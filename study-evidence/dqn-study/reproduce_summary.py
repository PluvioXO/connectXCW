"""Replay the study from archived game traces and recreate its figures."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.dqn_study import opening_sets, tactical_candidates

SEEDS = (51, 52, 53)
METHODS = ("dqn", "ddqn")
OPPONENTS = ("random", "minimax")
CHECKPOINTS = (0, 1000, 2000, 3000, 4000, 5000)
LABELS = {"dqn": "DQN", "ddqn": "Double DQN", "random": "Uniform random",
          "minimax": "Depth-two minimax"}
COLORS = {"dqn": "#087c8b", "ddqn": "#7151a1"}


def replay(report, openings):
    rewards = []
    immediate_loss = 0
    for index, game in enumerate(report["games"]):
        pair, seat = index // 2, index % 2 + 1
        board, prefix = openings[pair]
        assert (game["pair"], game["player"], game["opening"]) == (pair, seat, prefix)
        moves = game["moves_after_opening"]
        for ply, move in enumerate(moves):
            assert not board.is_terminal() and move in board.get_legal_moves()
            if board.current_player == seat:
                if report["mode"] != "network":
                    assert move in tactical_candidates(board)
                if game["reward"] == -1 and ply == len(moves) - 2:
                    child = board.make_move(move)
                    for alternative in tactical_candidates(board):
                        alt = board.make_move(alternative)
                        if alt.is_terminal() or all(
                            alt.make_move(reply).get_winner() != alt.current_player
                            for reply in alt.get_legal_moves()
                        ):
                            assert not child.is_terminal()
                            immediate_loss += 1
                            break
            board = board.make_move(move)
        assert board.is_terminal() and board.get_result(seat) == game["reward"]
        rewards.append(game["reward"])
    assert len(rewards) == 2 * len(openings)
    assert [report[k] for k in ("wins", "draws", "losses")] == [rewards.count(r) for r in (1, 0, -1)]
    pairs = ((np.asarray(rewards).reshape(-1, 2) + 1) / 2).mean(axis=1)
    assert pairs.mean() == report["score"]
    assert report["mode"] == "network" or immediate_loss == 0
    return pairs, immediate_loss


def describe_by_seed(values):
    scores = values.mean(axis=1)
    return {"mean": float(scores.mean()), "by_seed": scores.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("figures", type=Path)
    args = parser.parse_args()
    directory = args.directory
    curve_openings, final_openings = opening_sets(directory / "excluded-pilot-openings.json")
    opens = {"curve": curve_openings, "final": final_openings}
    assert not ({b for b, _ in curve_openings} & {b for b, _ in final_openings})
    for name in ("train-exit-code", "evaluate-exit-code"):
        assert (directory / name).read_text().strip() == "0"
    paired = {}
    avoidable = {}
    files = {}
    training = {}
    n_games = 0
    for seed in SEEDS:
        initial = []
        for method in METHODS:
            run = f"{method}-{seed}"
            base = directory / "runs" / run / "training"
            settings = json.loads((base / "config.json").read_text())
            assert settings["seed"] == seed and settings["method"] == method
            assert settings["decay_games"] == 4000 and settings["save_every"] == 1000
            rows = [json.loads(line) for line in (base / "training.jsonl").read_text().splitlines()]
            assert [r["game"] for r in rows] == list(range(1, 5001))
            assert all(r["mean_loss"] is None or np.isfinite(r["mean_loss"]) for r in rows)
            assert all(r["updates"] == max(0, (r["steps"] - 1024) // 4 + 1) for r in rows)
            training[run] = {"moves": rows[-1]["steps"], "updates": rows[-1]["updates"],
                             "config": settings}
            weights = torch.load(base / "game-000000.pt", map_location="cpu", weights_only=True)
            initial.append(weights["weights"])
            digests = {g: sha256((base / f"game-{g:06d}.pt").read_bytes()).hexdigest()
                       for g in CHECKPOINTS}
            training[run]["checkpoint_sha256"] = digests
            for path in sorted((directory / "evaluation" / run).glob("*.json")):
                record = json.loads(path.read_text())
                split, mode, game, opponent = (record[k] for k in
                                              ("split", "mode", "training_games", "opponent"))
                assert record["method"] == method and record["seed"] == seed
                assert record["checkpoint_sha256"] == digests[game]
                key = (method, seed, split, mode, game, opponent)
                assert key not in paired
                paired[key], avoidable[key] = replay(record, opens[split])
                files[path.relative_to(directory).as_posix()] = sha256(path.read_bytes()).hexdigest()
                n_games += len(record["games"])
        assert initial[0].keys() == initial[1].keys()
        assert all(torch.equal(p, initial[1][k]) for k, p in initial[0].items())
    tactics = {}
    for opponent in OPPONENTS:
        path = directory / "evaluation" / "tactics-only" / f"final-tactics-only-None-{opponent}.json"
        record = json.loads(path.read_text())
        assert record["mode"] == "tactics-only" and record["opponent"] == opponent
        values, immediate = replay(record, final_openings)
        tactics[opponent] = {"score": float(values.mean()),
                             "pair_scores": values.tolist(), "avoidable_losses": immediate}
        files[path.relative_to(directory).as_posix()] = sha256(path.read_bytes()).hexdigest()
        n_games += len(record["games"])
    assert n_games == 11008 and len(files) == 122

    def scores(method, split, mode, checkpoint, opponent):
        return np.stack([paired[(method, seed, split, mode, checkpoint, opponent)] for seed in SEEDS])

    summary = {"seeds": SEEDS, "curve_opening_seed": 621903, "final_opening_seed": 736401,
               "curve_pairs": len(curve_openings), "final_pairs": len(final_openings),
               "games_replayed": n_games, "evaluations": len(files),
               "evaluations_sha256": files, "training": training,
               "tactics_only": tactics, "curve": {}, "final": {}}
    bootstrap_ids = np.random.default_rng(20261119).integers(0, 64, (50_000, 64))

    def describe_contrast(values):
        result = describe_by_seed(values)
        per_opening = values.mean(axis=0)
        result["opening_ci95"] = np.quantile(per_opening[bootstrap_ids].mean(axis=1), [.025, .975]).tolist()
        return result

    summary["contrast_interval"] = "Shared opening-pair bootstrap conditional on these three trained seed pairs."
    for opponent in OPPONENTS:
        summary["curve"][opponent] = {
            method: [describe_by_seed(scores(method, "curve", "network", g, opponent))
                     for g in CHECKPOINTS] for method in METHODS
        }
        final = summary["final"][opponent] = {}
        for method in METHODS:
            initial = scores(method, "final", "network", 0, opponent)
            guarded_initial = scores(method, "final", "guarded", 0, opponent)
            plain = scores(method, "final", "network", 5000, opponent)
            guarded = scores(method, "final", "guarded", 5000, opponent)
            final[method] = {
                "initial": describe_by_seed(initial), "initial_guarded": describe_by_seed(guarded_initial),
                "trained": describe_by_seed(plain), "trained_guarded": describe_by_seed(guarded),
                "training_gain": describe_contrast(plain - initial),
                "training_gain_with_guard": describe_contrast(guarded - guarded_initial),
                "guard_gain": describe_contrast(guarded - plain),
                "guarded_minus_tactics_only": describe_contrast(
                    guarded - np.asarray(tactics[opponent]["pair_scores"])),
                "avoidable_losses": [avoidable[(method, seed, "final", "network", 5000, opponent)]
                                     for seed in SEEDS],
            }
            for mode in ("network", "guarded"):
                assert np.array_equal(scores("dqn", "final", mode, 0, opponent),
                                      scores("ddqn", "final", mode, 0, opponent))
        final["ddqn_minus_dqn"] = describe_contrast(scores("ddqn", "final", "network", 5000, opponent) -
                                                   scores("dqn", "final", "network", 5000, opponent))

    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), layout="constrained", sharey=True)
    for ax, opponent in zip(axes, OPPONENTS):
        ax.set_title(LABELS[opponent])
        for method in METHODS:
            values = np.array([r["by_seed"] for r in summary["curve"][opponent][method]]) * 100
            ax.plot(CHECKPOINTS, values, color=COLORS[method], alpha=.22, lw=.9)
            ax.plot(CHECKPOINTS, values.mean(axis=1), color=COLORS[method], marker="o", ms=3, lw=1.6,
                    label=LABELS[method])
        ax.axhline(50, color="#64748b", ls="--", lw=.75)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set(xlabel="Self-play games", xticks=(0, 1000, 3000, 5000), ylim=(0, 103))
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Score (%)")
    axes[1].legend(frameon=False, loc="upper left")
    for ext in ("png", "pdf"):
        fig.savefig(args.figures / f"dqn-study.{ext}", dpi=220)
    print(json.dumps({k: summary[k] for k in ("games_replayed", "final", "tactics_only")}, indent=2))


if __name__ == "__main__":
    main()
