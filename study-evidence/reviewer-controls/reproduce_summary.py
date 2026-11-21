"""Replay the final match traces and regenerate the coursework evidence summary."""

import argparse
from collections import Counter
from json import loads
from pathlib import Path

import numpy as np
import torch

from scripts.dqn_study import tactical_candidates
from scripts.q_controls import (
    FOLLOWUP, HORIZON_SEEDS, SEEDS, defence_choices, defence_pack, digest,
    opening_pack, read, save_json,
)
from src.game.endgame import replay_board
from src.training.tactical_dqn import tactical_values

HORIZON_ARMS = {"n1": ("plain", 1), "n3": ("plain", 3)}


def rates(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
            "by_seed": np.asarray(values).tolist()}


def seed_interval(values):
    critical = {5: 2.7764451051977987, 10: 2.2621571627409915}[len(values)]
    result = rates(values)
    half = critical * result["std"] / np.sqrt(len(values))
    result["seed_t_ci95"] = [result["mean"] - half, result["mean"] + half]
    return result


def match_interval(differences):
    result = seed_interval(differences.mean(axis=1))
    rng = np.random.default_rng(186418)
    seeds, openings = differences.shape
    opening_ids = rng.integers(openings, size=(20_000, openings))
    seed_ids = rng.integers(seeds, size=(20_000, seeds))
    seed_weights = np.stack([(seed_ids == i).mean(axis=1) for i in range(seeds)], axis=1)
    draws = differences.mean(axis=0)[opening_ids].mean(axis=1)
    crossed = np.take_along_axis(seed_weights @ differences, opening_ids, axis=1).mean(axis=1)
    result["openings_only_ci95"] = np.quantile(draws, [0.025, 0.975]).tolist()
    result["seeds_and_openings_ci95"] = np.quantile(crossed, [0.025, 0.975]).tolist()
    return result


def replay_matches(path, pack, metadata):
    record = read(path)
    assert all(record[key] == expected for key, expected in metadata.items()), path
    prefixes = read(pack)["openings"]
    scores = []
    rewards = []
    assert len(record["games"]) == 2 * len(prefixes)
    for i, game in enumerate(record["games"]):
        pair, seat = i // 2, 1 + i % 2
        assert (game["pair"], game["player"], game["opening"]) == (pair, seat, prefixes[pair])
        board = replay_board(prefixes[pair])
        for action in game["moves_after_opening"]:
            assert not board.is_terminal() and action in board.get_legal_moves()
            if record["mode"] == "guarded" and board.current_player == seat:
                assert action in tactical_candidates(board)
            board = board.make_move(action)
        assert board.is_terminal() and board.get_result(seat) == game["reward"]
        reward = game["reward"]
        scores.append((reward + 1) / 2)
        rewards.append(reward)
    counts = Counter(rewards)
    assert (record["wins"], record["draws"], record["losses"]) == (counts[1], counts[0], counts[-1])
    assert np.mean(scores) == record["score"]
    return np.asarray(scores).reshape(-1, 2).mean(axis=1)


def analyse_defence(root, directory, added_seeds):
    pack_path = directory / "defence-positions.json"
    pack = read(pack_path)
    assert pack == defence_pack(root)
    rows = pack["positions"]
    checkpoints = sorted((root / FOLLOWUP / "runs").glob("*/training/move-080000.pt"))
    pack_hash = digest(pack_path)
    choices_path = directory / "defence-choices.json"
    choices = read(choices_path)
    assert choices["pack_sha256"] == pack_hash
    assert choices["checkpoints"] == defence_choices(rows, checkpoints)
    records = choices["checkpoints"]
    seeds = SEEDS + added_seeds
    if added_seeds:
        extension = directory / "extension"
        added_paths = [extension / "runs" / f"{arm}-{seed}/training/move-080000.pt"
                       for arm in ("n1", "tactical") for seed in added_seeds]
        added_choices_path = extension / "defence-choices.json"
        added_choices = read(added_choices_path)
        assert added_choices["pack_sha256"] == pack_hash
        assert added_choices["checkpoints"] == defence_choices(rows, added_paths)
        assert not records.keys() & added_choices["checkpoints"].keys()
        assert all(int(source.rsplit("-", 1)[1]) in SEEDS for row in rows for source in row["sources"])
        records = {**records, **added_choices["checkpoints"]}
        checkpoints += added_paths

    def errors(positions, records):
        mistakes = {}
        for run, record in records.items():
            assert len(record["choices"]) == 2 * len(positions)
            counts = []
            for i, row in enumerate(positions):
                a, b = record["choices"][2 * i:2 * i + 2]
                b = 6 - b
                assert a in row["legal"] and b in row["legal"]
                counts.append((int(a in row["exposes"]) + int(b in row["exposes"])) / 2)
            mistakes[run] = np.asarray(counts)
        return {variant: np.stack([mistakes[f"{variant}-{seed}"] for seed in seeds])
                for variant in ("plain", "tactical")}

    def compare(mistakes, mask):
        plain = mistakes["plain"][:, mask].mean(axis=1)
        tactical = mistakes["tactical"][:, mask].mean(axis=1)
        return {"positions": int(np.count_nonzero(mask)), "plain": rates(plain),
                "tactical": rates(tactical), "difference": seed_interval(tactical - plain)}

    mistakes = errors(rows, records)
    plies = np.asarray([len(r["prefix"]) for r in rows])
    phases = {f"{lo}-{hi - 1}": compare(mistakes, (plies >= lo) & (plies < hi))
              for lo, hi in ((0, 10), (10, 20), (20, 30))}
    exclusions = []
    for i, seed in enumerate(SEEDS):
        mask = [all(not source.endswith(f"-{seed}") for source in r["sources"]) for r in rows]
        gap = mistakes["tactical"][i, mask] - mistakes["plain"][i, mask]
        exclusions.append({"seed": seed, "positions": sum(mask), "difference": float(gap.mean())})

    solved = []
    for row in read(root / FOLLOWUP / "testpack.json")["oracle"]:
        if row["group"] != "block-now":
            continue
        board = replay_board(row["prefix"])
        legal = board.get_legal_moves()
        labels = tactical_values(board)
        bad = [c for c in legal if labels[c] == -1]
        assert 0 < len(bad) < len(legal) and not np.any(labels == 1)
        assert all(row["values_by_column"][c] == -1 for c in bad)
        solved.append({"prefix": row["prefix"], "legal": legal, "exposes": bad})
    contrast = compare(errors(solved, defence_choices(solved, checkpoints)), np.ones(len(solved), bool))
    result = {"match_boards": compare(mistakes, plies < 30), "by_ply": phases,
              "excluding_own_seed_sources": exclusions,
              "solved_defence_same_metric_exploratory": contrast,
              "positions_sha256": pack_hash, "choices_sha256": digest(choices_path)}
    if added_seeds:
        result.update(seeds=list(seeds), added_choices_sha256=digest(added_choices_path),
                      added_seeds_only=compare({k: v[len(SEEDS):] for k, v in mistakes.items()},
                                               plies < 30))
    return result


def verify_training(directory, seeds, arms):
    stats = []
    for seed in seeds:
        initial_weights = []
        matched_configs = []
        for arm, (variant, horizon) in arms.items():
            run = directory / "runs" / f"{arm}-{seed}" / "training"
            config = read(run / "config.json")
            assert config["seed"] == seed and config["n_step"] == horizon
            assert config["moves"] == 80_000 and config["decay_moves"] == 64_000
            assert config["variant"] == variant and config["tactical_weight"] == (variant == "tactical")
            matched_configs.append({k: v for k, v in config.items()
                                    if k not in ("n_step", "output", "variant", "tactical_weight")})
            with (run / "training.jsonl").open() as stream:
                logs = [loads(line) for line in stream]
            assert [row["moves"] for row in logs] == list(range(1000, 80_001, 1000))
            assert all(row["updates"] == max(0, (row["moves"] - 1024) // 4 + 1)
                       and len(row["sampled_horizons"]) == horizon
                       and sum(row["sampled_horizons"]) == 128 * row["updates"] for row in logs)
            initial = torch.load(run / "move-000000.pt", weights_only=True, map_location="cpu")
            assert initial["steps"] == initial["updates"] == 0
            assert initial["seed"] == seed and initial["variant"] == variant and initial["n_step"] == horizon
            initial_weights.append(initial["weights"])
            final = torch.load(run / "move-080000.pt", weights_only=True, map_location="cpu")
            tail = logs[-1]
            assert final["seed"] == seed and final["variant"] == variant and final["n_step"] == horizon
            assert final["steps"] == tail["moves"] == 80_000 and final["updates"] == tail["updates"] == 19745
            assert final["games"] == tail["games"]
            stats.append({"seed": seed, "n_step": horizon, "games": final["games"],
                          "seconds": tail["seconds"], "sampled_horizons": tail["sampled_horizons"],
                          "run_files_sha256": {p.name: digest(p) for p in sorted(run.iterdir()) if p.is_file()}})
            if "tactical" in arms:
                stats[-1]["arm"] = arm
        assert all(config == matched_configs[0] for config in matched_configs[1:])
        assert all(weights.keys() == initial_weights[0].keys()
                   and all(torch.equal(initial_weights[0][key], value) for key, value in weights.items())
                   for weights in initial_weights[1:])
    return stats


def summarize(root, directory):
    pack = directory / "openings.json"
    assert read(pack) == opening_pack(root)
    metadata = {"pack_sha256": digest(pack)}
    match_hashes = {}
    added_match_hashes = {}
    extension = directory / "extension"

    def arm(location, name, mode, depth, checkpoint=None, **expected):
        if checkpoint:
            expected["checkpoint_sha256"] = digest(checkpoint)
        path = location / "matches" / f"{name}.json"
        hashes = match_hashes if location == directory else added_match_hashes
        hashes[path.name] = digest(path)
        return replay_matches(path, pack, {**metadata, "mode": mode, "depth": depth, **expected})

    def search_matches(location, seeds, checkpoint_root, prefix):
        result = {}
        for mode in ("initial", "lookahead", "guarded"):
            step = 0 if mode == "initial" else 80_000
            result[mode] = np.stack([
                arm(location, f"{mode}-{seed}", mode, 3,
                    checkpoint_root / "runs" / f"{prefix}-{seed}/training/move-{step:06d}.pt",
                    seed=seed, steps=step, variant="plain", updates=19745 if step else 0)
                for seed in seeds
            ])
        return result

    def horizon_matches(location, seeds):
        return {n: np.stack([
            arm(location, f"n{n}-{seed}", "horizon", 2,
                location / "runs" / f"n{n}-{seed}/training/move-080000.pt",
                seed=seed, steps=80_000, variant="plain", updates=19745, n_step=n)
            for seed in seeds
        ]) for n in (1, 3)}

    def search_results(rows, seeds):
        trained = rows["lookahead"]
        return {"seeds": list(seeds), "heuristic": float(heuristic.mean()),
                **{mode: rates(matrix.mean(axis=1)) for mode, matrix in rows.items()},
                "lookahead_minus_heuristic": match_interval(trained - heuristic),
                "lookahead_minus_initial": match_interval(trained - rows["initial"]),
                "lookahead_minus_guarded": match_interval(trained - rows["guarded"])}

    def horizon_results(rows, seeds):
        return {"seeds": list(seeds), "one_step": rates(rows[1].mean(axis=1)),
                "three_step": rates(rows[3].mean(axis=1)), "difference": match_interval(rows[3] - rows[1])}

    heuristic = arm(directory, "heuristic", "heuristic", 3)
    search = search_matches(directory, SEEDS, root / FOLLOWUP, "plain")
    training = verify_training(directory, HORIZON_SEEDS, HORIZON_ARMS)
    horizons = horizon_matches(directory, HORIZON_SEEDS)
    assert set(match_hashes) == {p.name for p in (directory / "matches").glob("*.json")}
    added_seeds = ()
    added = {}
    plan_path = extension / "plan.json"
    if plan_path.is_file():
        plan = read(plan_path)
        added_seeds = tuple(plan["added_seeds"])
        assert len(added_seeds) == len(set(added_seeds)) == 5
        assert set(added_seeds).isdisjoint(SEEDS + HORIZON_SEEDS)
        assert all((extension / f"{job}-exit-code").read_text().strip() == "0"
                   for job in ("train", "evaluate"))
        assert plan["inputs_sha256"] == {
            "openings.json": digest(pack), "defence-positions.json": digest(directory / "defence-positions.json"),
            "solved-positions.json": digest(root / FOLLOWUP / "testpack.json"),
        }
        arms = {**HORIZON_ARMS, "tactical": ("tactical", 1)}
        assert plan["three_arm_block"] == {k: {"variant": v, "n_step": n} for k, (v, n) in arms.items()}
        assert plan["budget"] == {"moves": 80_000, "updates": 19745, "epsilon_decay_moves": 64_000}
        added_training = verify_training(extension, added_seeds, arms)
        extra_search = search_matches(extension, added_seeds, extension, "n1")
        extra_horizons = horizon_matches(extension, added_seeds)
        assert set(added_match_hashes) == {p.name for p in (extension / "matches").glob("*.json")}
        assert len(added_match_hashes) == 25
        added = {"extension": {"plan_sha256": digest(plan_path), "added_seeds": list(added_seeds),
                               "training": added_training, "match_trace_sha256": added_match_hashes,
                               "search": search_results(extra_search, added_seeds),
                               "horizon": horizon_results(extra_horizons, added_seeds)}}
        training += [row for row in added_training if row["arm"] in HORIZON_ARMS]
        search = {k: np.concatenate((rows, extra_search[k])) for k, rows in search.items()}
        horizons = {n: np.concatenate((rows, extra_horizons[n])) for n, rows in horizons.items()}
    return {
        "opening_pairs": len(heuristic), "opening_pack_sha256": digest(pack),
        "replayed_games": 2 * len(heuristic) * (len(match_hashes) + len(added_match_hashes)),
        "match_trace_sha256": match_hashes,
        "search": search_results(search, SEEDS + added_seeds),
        "horizon": {**horizon_results(horizons, HORIZON_SEEDS + added_seeds), "training": training},
        "defence": analyse_defence(root, directory, added_seeds),
        **added,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    folder = Path(__file__).resolve().parent
    save_json(args.output, summarize(folder.parents[1], folder))
