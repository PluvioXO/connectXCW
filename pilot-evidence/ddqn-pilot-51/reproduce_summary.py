from pathlib import Path
from hashlib import sha256
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.double_dqn import paired_openings
from src.game.board import Connect4Board

evidence = Path(sys.argv[1])
figures = Path(sys.argv[2])
read = lambda path: json.loads(path.read_text())
training = [json.loads(line) for line in (evidence / "training/training.jsonl").read_text().splitlines()]
config = read(evidence / "training/config.json")
assert config["seed"] == 51 and config["method"] == "ddqn" and config["games"] == 5000
assert [row["game"] for row in training] == list(range(1, 5001))
assert all(7 <= row["steps"] - previous <= 42
           for row, previous in zip(training, [0] + [r["steps"] for r in training[:-1]]))
assert training[-1]["updates"] == (training[-1]["steps"] - 1024) // 4 + 1
assert all(row["mean_loss"] is None or np.isfinite(row["mean_loss"]) for row in training)
assert (evidence / "exit-code").read_text().strip() == "0"
openings = list(paired_openings(32, 80421))
indices = np.random.default_rng(20261119).integers(0, 32, (50_000, 32))
games = list(range(0, 5001, 1000))
summary = {"seed": 51, "opening_seed": 80421, "opening_pairs": 32,
           "training_games": len(training), "training_moves": training[-1]["steps"],
           "updates": training[-1]["updates"], "config": config, "evaluations": {},
           "interval": "Percentile bootstrap across 32 opening pairs. Conditional on one training seed.",
           "started_utc": (evidence / "started-utc").read_text().strip(),
           "training_ended_utc": (evidence / "training-ended-utc").read_text().strip(),
           "ended_utc": (evidence / "ended-utc").read_text().strip()}

fig, ax = plt.subplots(figsize=(7.0, 3.1), layout="constrained")
for opponent, label, color in [("random", "Uniform random", "#087c8b"),
                               ("minimax", "Depth-two minimax", "#7151a1")]:
    records, all_pairs = [], []
    for count in games:
        result = read(evidence / "evaluation" / f"{opponent}-{count:06d}.json")
        assert result["training_games"] == count and result["opponent"] == opponent
        assert result["opening_seed"] == 80421 and result["pairs"] == 32
        assert result["training_seed"] == 51 and result["method"] == "ddqn"
        checkpoint = evidence / "training" / f"game-{count:06d}.pt"
        assert sha256(checkpoint.read_bytes()).hexdigest() == result["checkpoint_sha256"]
        rewards = []
        for index, game in enumerate(result["games"]):
            pair, seat = index // 2, index % 2 + 1
            board, prefix = openings[pair]
            assert (game["pair"], game["player"], game["opening"]) == (pair, seat, prefix)
            for move in game["moves_after_opening"]:
                assert not board.is_terminal() and move in board.get_legal_moves()
                board = board.make_move(move)
            reward = board.get_result(seat)
            assert board.is_terminal() and reward == game["reward"]
            rewards.append(reward)
        assert len(rewards) == 64
        pairs = ((np.array(rewards).reshape(32, 2) + 1) / 2).mean(axis=1)
        assert result["score"] == pairs.mean()
        assert [result[k] for k in ("wins", "draws", "losses")] == [rewards.count(x) for x in (1, 0, -1)]
        all_pairs.append(pairs)
        records.append({"training_games": count, "score": float(pairs.mean()),
                        "ci95": np.quantile(pairs[indices].mean(axis=1), [0.025, 0.975]).tolist(),
                        "wins": result["wins"], "draws": result["draws"], "losses": result["losses"],
                        "checkpoint_sha256": result["checkpoint_sha256"]})
    diff = all_pairs[-1] - all_pairs[0]
    summary["evaluations"][opponent] = {"checkpoints": records, "final_minus_initial": float(diff.mean()),
                                       "difference_ci95": np.quantile(diff[indices].mean(axis=1), [0.025, 0.975]).tolist()}
    y = np.array([r["score"] for r in records]) * 100
    lo, hi = np.array([r["ci95"] for r in records]).T * 100
    ax.errorbar(np.array(games), y, yerr=(y-lo,hi-y), marker="o", lw=1.5,
                capsize=3, color=color, label=label)
ax.axhline(50, color="#64748b", ls="--", lw=0.9)
ax.set(xlabel="Completed self-play games", ylabel="Score (%)", xticks=games, ylim=(0, 103))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=.18)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2)
for ext in ("png", "pdf"):
    fig.savefig(figures / f"ddqn-pilot.{ext}", dpi=220)
summary["games_replayed"] = 2 * len(games) * 64

# On final minimax losses, test whether another legal move could avoid defeat
# on the next turn. This diagnostic makes no claim about eventual game value.
final_minimax = read(evidence / "evaluation/minimax-005000.json")
avoidable = 0
for game in final_minimax["games"]:
    if game["reward"] != -1 or len(game["moves_after_opening"]) < 2:
        continue
    board = Connect4Board()
    for move in game["opening"] + game["moves_after_opening"][:-2]:
        board = board.make_move(move)
    assert board.current_player == game["player"]
    alternatives = [board.make_move(c) for c in board.get_legal_moves()]
    avoidable += any(child.is_terminal() or all(
        child.make_move(c).get_winner() != child.current_player
        for c in child.get_legal_moves()
    ) for child in alternatives)
summary["final_minimax_avoidable_next_move_losses"] = avoidable
(evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
