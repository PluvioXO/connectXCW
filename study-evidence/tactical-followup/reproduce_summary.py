"""Verify and summarise the move-matched tactics and endgame experiments."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.dqn_study import tactical_candidates
from scripts.tactical_study import CHECKPOINTS, SEEDS, STRATA, canonical_board, jobs
from src.game.endgame import action_values, replay_board
from src.training.tactical_dqn import tactical_values

VARIANTS = ("plain", "tactical")
MODES = ("greedy", "guarded", "mirrored_average")
VARIANT_NAMES = {"plain": "Plain Double DQN", "tactical": "With tactical supervision"}
COLORS = {"plain": "#326c9a", "tactical": "#bd5709"}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def by_seed(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "by_seed": values.tolist()}


def gap(values):
    """Two uncertainty views for paired differences with shared openings."""
    rows = np.asarray(values, dtype=float)
    result = by_seed(rows.mean(axis=1))
    seed_means = np.asarray(result["by_seed"])
    assert len(seed_means) == 5
    half_width = 2.7764451051977987 * seed_means.std(ddof=1) / np.sqrt(len(seed_means))
    result["seed_t_ci95"] = [result["mean"] - half_width, result["mean"] + half_width]
    samples = np.random.default_rng(103915).integers(0, rows.shape[1],
                                                     size=(20_000, rows.shape[1]))
    draws = rows.mean(axis=0)[samples].mean(axis=1)
    result["openings_only_ci95"] = np.quantile(draws, [.025, .975]).tolist()
    return result


def verify_testpack(base, pack):
    earlier = root_paths(base)
    assert pack["exclusion_files_sha256"] == {p.name: digest(p) for p in earlier}
    previous = {canonical_board(replay_board(g["opening"])) for p in earlier for g in read(p)["games"]}
    assert len(previous) == pack["previous_openings_excluded_up_to_reflection"]
    assert pack["train_seeds"] == list(SEEDS) and pack["checkpoints"] == list(CHECKPOINTS)
    assert Counter(row["group"] for row in pack["oracle"]) == {group: 96 for group in STRATA}
    for collection in (pack["curve"], pack["final"], [o["prefix"] for o in pack["oracle"]]):
        keys = {canonical_board(replay_board(prefix)) for prefix in collection}
        assert len(keys) == len(collection) and previous.isdisjoint(keys)
        previous |= keys
    for row in pack["oracle"]:
        board = replay_board(row["prefix"])
        values = {c: v for c, v in enumerate(row["values_by_column"]) if v is not None}
        assert values == action_values(board) and len(set(values.values())) >= 2
        approx = tactical_values(board)
        assert all(values[c] == approx[c] for c in values if np.isfinite(approx[c]))
        if row["group"] == "multi-step":
            assert not np.isfinite(approx).any()


def root_paths(directory):
    root = directory.resolve().parents[1]
    return [
        root / "pilot-evidence/ddqn-pilot-51/evaluation/random-000000.json",
        root / "study-evidence/dqn-study/evaluation/ddqn-51/curve-network-0-random.json",
        root / "study-evidence/dqn-study/evaluation/ddqn-51/final-network-0-random.json",
    ]


def verify_games(record, prefixes, diagnose=False):
    results = []
    tallies = Counter()
    boards = [replay_board(prefix) for prefix in prefixes]
    assert not any(board.is_terminal() for board in boards)
    for index, game in enumerate(record["games"]):
        pair, seat = index // 2, index % 2 + 1
        assert (game["pair"], game["player"], game["opening"]) == (pair, seat, prefixes[pair])
        board = boards[pair]
        for ply, action in enumerate(game["moves_after_opening"]):
            assert not board.is_terminal() and action in board.get_legal_moves()
            if board.current_player == seat:
                if record["mode"] != "greedy":
                    assert action in tactical_candidates(board)
                if diagnose:
                    legal = board.get_legal_moves()
                    labels = tactical_values(board)[legal]
                    chosen = labels[legal.index(action)]
                    if np.any(labels == 1):
                        tallies["win_opportunities"] += 1
                        tallies["missed_wins"] += int(chosen != 1)
                    elif np.any(labels == -1) and not np.all(labels == -1):
                        tallies["defence_opportunities"] += 1
                        tallies["avoidable_immediate_exposures"] += int(chosen == -1)
                        if len(prefixes[pair]) + ply < 30:
                            tallies["defence_opportunities_before_30_moves"] += 1
                            tallies["avoidable_exposures_before_30_moves"] += int(chosen == -1)
            board = board.make_move(action)
        assert board.is_terminal() and board.get_result(seat) == game["reward"]
        results.append(game["reward"])
    assert len(results) == 2 * len(prefixes)
    counts = Counter(results)
    assert (record["wins"], record["draws"], record["losses"]) == (counts[1], counts[0], counts[-1])
    scores = (np.asarray(results, dtype=float).reshape(-1, 2) + 1.) / 2.
    assert scores.mean() == record["score"]
    return scores, dict(tallies)


def outcome_scores(oracle, moves):
    labels = [row["values_by_column"] for row in oracle]
    choices = [labels[i][move] for i, move in enumerate(moves)]
    assert all(value in (-1, 0, 1) for value in choices)
    max_values = [max(value for value in row if value is not None) for row in labels]
    regret = np.asarray(max_values) - choices
    return regret == 0, regret



def verify_lookahead(directory, pack, training):
    assert (directory / "lookahead-exit-code").read_text().strip() == "0"
    pack_path = directory / "lookahead-pack.json"
    manifest = read(pack_path)
    prefixes = manifest["openings"]
    other = [g["opening"] for p in root_paths(directory) for g in read(p)["games"]]
    other += pack["curve"] + pack["final"]
    candidates = [canonical_board(replay_board(moves)) for moves in prefixes]
    assert len(set(candidates)) == len(candidates) == 24
    assert set(candidates).isdisjoint(canonical_board(replay_board(moves)) for moves in other)
    assert manifest["references_testpack_sha256"] == digest(directory / "testpack.json")
    assert manifest["seeds"] == list(SEEDS)

    audit, comparisons, n = {}, {}, 0
    for depth in (2, 3):
        per_depth = comparisons[str(depth)] = {}
        controls = {}
        control = directory / "lookahead/tactics-only" / f"tactics-only-None-minimax-{depth}.json"
        report = read(control)
        assert report["pack_sha256"] == digest(pack_path) and report["depth"] == depth
        tactics, _ = verify_games(report, prefixes)
        per_depth["tactics_only"] = float(tactics.mean())
        audit[str(control.relative_to(directory))] = digest(control)
        n += len(report["games"])
        for variant in VARIANTS:
            outputs = per_depth[variant] = {}
            for step, mode in ((80_000, "guarded"), (80_000, "lookahead"), (0, "lookahead")):
                if step == 0 and variant == "tactical":
                    continue
                rows = []
                for seed in SEEDS:
                    run = f"{variant}-{seed}"
                    file = directory / "lookahead" / run / f"{mode}-{step}-minimax-{depth}.json"
                    report = read(file)
                    assert (report["mode"], report["depth"], report["moves"]) == (mode, depth, step)
                    assert (report["variant"], report["seed"]) == (variant, seed)
                    assert report["pack_sha256"] == digest(pack_path)
                    assert report["checkpoint_sha256"] == training[run]["checkpoints"][step]
                    replay_mode = dict(report, mode="greedy") if mode == "lookahead" else report
                    scores, _ = verify_games(replay_mode, prefixes)
                    rows.append(scores.mean(axis=1))
                    audit[str(file.relative_to(directory))] = digest(file)
                    n += len(report["games"])
                key = "initial_lookahead" if step == 0 else mode
                controls[variant, key] = np.stack(rows)
                outputs[key] = by_seed(np.stack(rows).mean(axis=1))
            outputs["lookahead_minus_guarded"] = gap(controls[variant, "lookahead"] -
                                                       controls[variant, "guarded"])
            outputs["lookahead_minus_tactics"] = gap(controls[variant, "lookahead"] - tactics.mean(axis=1))
        assert ("plain", "initial_lookahead") in controls
        for variant in VARIANTS:
            per_depth[variant]["lookahead_training_gain"] = gap(controls[variant, "lookahead"] -
                                                                  controls["plain", "initial_lookahead"])
    assert len(audit) == 52 and n == 2496
    return {"distinct_openings": len(prefixes), "pack_sha256": digest(pack_path),
            "games_replayed": n, "file_hashes": audit, "scores": comparisons}

def summarize(directory, figures, reference_weights):
    for name in ("train-exit-code", "evaluate-exit-code", "trace-exit-code"):
        assert (directory / name).read_text().strip() == "0"
    pack = read(directory / "testpack.json")
    pack_hash = digest(directory / "testpack.json")
    verify_testpack(directory, pack)
    oracle = pack["oracle"]
    group_masks = {group: np.asarray([r["group"] == group for r in oracle]) for group in STRATA}
    records, positions, certificate_rates, training, file_hashes = {}, {}, {}, {}, {}
    diagnostics = {}
    n_games = 0
    for seed in SEEDS:
        initial = []
        for variant in VARIANTS:
            run = f"{variant}-{seed}"
            base = directory / "runs" / run / "training"
            cfg = read(base / "config.json")
            assert cfg["variant"] == variant and cfg["seed"] == seed and cfg["moves"] == 80_000
            assert cfg["decay_moves"] == 64_000 and cfg["save_every"] == 20_000
            assert cfg["tactical_weight"] == (variant == "tactical")
            logs = [json.loads(line) for line in (base / "training.jsonl").read_text().splitlines()]
            assert [row["moves"] for row in logs] == list(range(1000, 80_001, 1000))
            assert all(np.isfinite(row["mean_loss"]) for row in logs if row["mean_loss"] is not None)
            assert all(row["updates"] == max(0, (row["moves"] - 1024) // 4 + 1) for row in logs)
            assert logs[-1]["updates"] == 19_745
            assert 0 < logs[-1]["certified_actions"] < logs[-1]["legal_actions"]
            training[run] = {"config": cfg, "last": logs[-1], "checkpoints": {}}
            certificate_rates[run] = logs[-1]["certified_actions"] / logs[-1]["legal_actions"]
            for step in CHECKPOINTS:
                ckpt = base / f"move-{step:06d}.pt"
                ckpt_hash = digest(ckpt)
                training[run]["checkpoints"][step] = ckpt_hash
                if step == 0:
                    initial.append(torch.load(ckpt, map_location="cpu", weights_only=True)["weights"])
                stage = directory / "evaluation-with-traces" / run / f"step-{step}"
                for split, mode, opponent in jobs(step):
                    path = stage / f"{split}-{mode}-{step}-{opponent}.json"
                    result = read(path)
                    assert result["testpack_sha256"] == pack_hash and result["checkpoint_sha256"] == ckpt_hash
                    assert (result["variant"], result["seed"], result["steps"]) == (variant, seed, step)
                    assert (result["split"], result["mode"], result["opponent"]) == (split, mode, opponent)
                    prefixes = pack[split][:32] if opponent == "minimax-3" else pack[split]
                    key = (variant, seed, step, split, mode, opponent)
                    diagnose = step == 80_000 and split == "final" and opponent == "minimax-2"
                    records[key], tallies = verify_games(result, prefixes, diagnose=diagnose)
                    if diagnose:
                        diagnostics[f"{run}/{mode}"] = tallies
                    n_games += len(result["games"])
                    file_hashes[str(path.relative_to(directory))] = digest(path)
                if step in (0, 40_000, 80_000):
                    path = stage / f"oracle-{step}.json"
                    static = read(path)
                    assert static["testpack_sha256"] == pack_hash and static["checkpoint_sha256"] == ckpt_hash
                    assert len(static["actions"]) == len(oracle)
                    assert [r["id"] for r in static["actions"]] == list(range(len(oracle)))
                    errors = [r["q_reflection_mae"] for r in static["actions"]]
                    assert all(np.isfinite(error) and 0 <= error <= 2 for error in errors)
                    for r, action in zip(oracle, static["actions"]):
                        assert action["guarded"] in tactical_candidates(replay_board(r["prefix"]))
                    positions[(variant, seed, step)] = {
                        mode: outcome_scores(oracle, [r[mode] for r in static["actions"]]) for mode in MODES
                    }
                    positions[(variant, seed, step)]["reflection_mae"] = errors
                    file_hashes[str(path.relative_to(directory))] = digest(path)
        assert initial[0].keys() == initial[1].keys()
        assert all(torch.equal(v, initial[1][k]) for k, v in initial[0].items())

    control = {}
    for opponent in ("random", "minimax-2", "minimax-3"):
        path = directory / "evaluation-with-traces/tactics-only" / f"final-tactics-only-None-{opponent}.json"
        report = read(path)
        assert report["testpack_sha256"] == pack_hash
        assert (report["opponent"], report["mode"], report["split"]) == (opponent, "tactics-only", "final")
        control[opponent], _ = verify_games(report, pack["final"][:32] if opponent == "minimax-3" else pack["final"])
        n_games += len(report["games"])
        file_hashes[str(path.relative_to(directory))] = digest(path)
    assert n_games == 10_400 and len(file_hashes) == 153

    for name in sorted(file_hashes):
        path = directory / name
        original_path = directory / name.replace("evaluation-with-traces/", "evaluation/", 1)
        result, original = read(path), read(original_path)
        if "completed_training_games" in result:
            assert original.pop("games") == result.pop("completed_training_games")
            if "games" in result:
                result.pop("games")
        assert result == original, name

    def paired(variant, step, split, mode, opponent):
        return np.stack([records[variant, seed, step, split, mode, opponent].mean(axis=1) for seed in SEEDS])

    primary = gap(paired("tactical", 80_000, "final", "greedy", "minimax-2") -
                  paired("plain", 80_000, "final", "greedy", "minimax-2"))
    summary = {
        "testpack_sha256": pack_hash, "seeds": SEEDS, "trained_moves": 800_000,
        "games_replayed": n_games, "all_scores_match_initial_evaluation": True, "evaluation_manifest": file_hashes,
        "certified_action_fraction": certificate_rates, "training": training, "diagnostics": diagnostics,
        "learning_curve": {}, "final": {}, "oracle": {}, "primary_tactical_minus_plain": primary,
        "tactics_only_match_scores": {key: float(values.mean()) for key, values in control.items()},
    }
    for variant in VARIANTS:
        summary["learning_curve"][variant] = [
            by_seed(paired(variant, step, "curve", "greedy", "minimax-2").mean(axis=1))
            for step in CHECKPOINTS
        ]
        final = summary["final"][variant] = {}
        for opponent in ("random", "minimax-2", "minimax-3"):
            after = paired(variant, 80000, "final", "greedy", opponent)
            final[opponent] = {"greedy": by_seed(after.mean(axis=1))}
            if opponent in ("random", "minimax-2"):
                before = paired(variant, 0, "final", "greedy", opponent)
                final[opponent]["initial"] = by_seed(before.mean(axis=1))
                final[opponent]["learning_gain"] = gap(after - before)
                assert np.array_equal(paired("plain", 0, "final", "greedy", opponent),
                                      paired("tactical", 0, "final", "greedy", opponent))
            if opponent != "random":
                with_guard = paired(variant, 80000, "final", "guarded", opponent)
                final[opponent]["guarded"] = by_seed(with_guard.mean(axis=1))
                final[opponent]["guard_gain"] = gap(with_guard - after)
                final[opponent]["guarded_minus_tactics_only"] = gap(with_guard - control[opponent].mean(axis=1))
        terminal = summary["oracle"][variant] = {}
        for step in (0, 40_000, 80_000):
            terminal[step] = {}
            for mode in MODES:
                optimal, regret = zip(*(positions[(variant, seed, step)][mode] for seed in SEEDS))
                optimal, regret = np.asarray(optimal), np.asarray(regret)
                terminal[step][mode] = {
                    group: {"optimal_action": by_seed(optimal[:, mask].mean(axis=1)),
                            "regret": by_seed(regret[:, mask].mean(axis=1))}
                    for group, mask in group_masks.items()
                }
            terminal[step]["reflection_mae"] = by_seed([
                np.mean(positions[(variant, seed, step)]["reflection_mae"]) for seed in SEEDS
            ])

    summary["oracle_delta_greedy"] = {}
    for group, mask in group_masks.items():
        values = np.stack([
            positions[("tactical", seed, 80000)]["greedy"][0][mask].astype(float) -
            positions[("plain", seed, 80000)]["greedy"][0][mask].astype(float)
            for seed in SEEDS
        ])
        summary["oracle_delta_greedy"][group] = gap(values)

    ref_path = directory / "oracle-reference.json"
    ref = read(ref_path)
    assert ref["testpack_sha256"] == pack_hash and ref["simulations"] == 64
    assert ref["reference_sha256"] == digest(reference_weights)
    ref_metrics = summary["inherited_oracle"] = {"simulations": 64, "weights_sha256": ref["reference_sha256"]}
    for mode in ("prior", "puct"):
        assert len(ref[mode]) == len(oracle)
        optimal, regret = outcome_scores(oracle, ref[mode])
        ref_metrics[mode] = {g: {"optimal_action": float(optimal[mask].mean()),
                                "regret": float(regret[mask].mean())} for g, mask in group_masks.items()}

    summary["lookahead"] = verify_lookahead(directory, pack, training)
    oracle_verification = read(directory / "oracle-verification.json")
    assert oracle_verification["positions_verified"] == 288
    assert oracle_verification["testpack_sha256"] == pack_hash
    summary["independent_oracle_verification"] = oracle_verification

    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot(summary, figures)
    print(json.dumps({"games_replayed": n_games, "primary": primary, "final": summary["final"],
                      "oracle_delta": summary["oracle_delta_greedy"], "reference": ref_metrics}, indent=2))


def plot(summary, figures):
    fig, (curve, motifs) = plt.subplots(1, 2, figsize=(7.1, 3.0), layout="constrained")
    for variant in VARIANTS:
        values = 100 * np.asarray([row["by_seed"] for row in summary["learning_curve"][variant]])
        curve.plot(np.asarray(CHECKPOINTS) / 1000, values, color=COLORS[variant], lw=.7, alpha=.27)
        curve.plot(np.asarray(CHECKPOINTS) / 1000, values.mean(axis=1), color=COLORS[variant],
                   marker="o", markersize=3, linewidth=1.6, label=VARIANT_NAMES[variant])
    curve.set(xlabel="Training moves (thousands)", ylabel="Game score (%)",
              title="(a) Greedy play vs depth-two minimax", xticks=(0, 20, 40, 60, 80), ylim=(0, None))
    curve.legend(loc="upper left", frameon=False, fontsize=8)

    x = np.arange(len(STRATA))
    rng = np.random.default_rng(150)
    updates = summary["oracle"]
    for variant, offset in zip(VARIANTS, (-.18, .18)):
        seed_scores = np.asarray([updates[variant][80000]["greedy"][group]["optimal_action"]["by_seed"]
                                  for group in STRATA]) * 100
        motifs.bar(x + offset, seed_scores.mean(axis=1), width=.32, color=COLORS[variant], alpha=.82)
        jitter = rng.uniform(-.1, .1, size=seed_scores.shape)
        motifs.plot((x + offset)[:, None] + jitter, seed_scores, color="#152536", marker="o", ms=2, ls="none")
    motifs.set(xticks=x, xticklabels=("Immediate\nwin", "Immediate\ndefence", "Multi-step"),
               ylabel="Optimal action (%)", title="(b) Solved endgames at 80k moves", ylim=(0, 103))
    for ax in (curve, motifs):
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    for ext in ("pdf", "png"):
        fig.savefig(figures / f"tactical-followup.{ext}", dpi=230)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("figures", type=Path)
    parser.add_argument("--reference-weights", type=Path, default=Path("models/large.pt"))
    args = parser.parse_args()
    summarize(args.directory, args.figures, args.reference_weights)


if __name__ == "__main__":
    main()
