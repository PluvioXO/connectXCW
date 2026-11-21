"""Run a move-matched Double DQN experiment with or without tactical labels."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.double_dqn import positive_int
from src.game.board import Connect4Board
from src.training.double_dqn import advance
from src.training.tactical_dqn import FixedBudgetLearner, tactical_values


def train(args):
    args.output.mkdir(parents=True)
    weight = float(args.variant == "tactical")
    learner = FixedBudgetLearner(args.device, args.seed, tactical_weight=weight, n_step=args.n_step)
    config = dict(vars(args), output=str(args.output), torch_version=torch.__version__,
                  tactical_weight=weight, batch_size=learner.batch_size, warmup=learner.warmup,
                  update_every=learner.update_every, sync_every=learner.sync_every,
                  replay_capacity=learner.replay_capacity, learning_rate=3e-4)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def save(games):
        path = args.output / f"move-{learner.steps:06d}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save({"weights": learner.online.state_dict(), "steps": learner.steps,
                    "updates": learner.updates, "games": games, "seed": args.seed,
                    "variant": args.variant, "n_step": args.n_step}, temporary)
        temporary.replace(path)

    start = time.monotonic()
    games, labelled, legal_actions, total_loss, n_loss = 0, 0, 0, 0., 0
    board = Connect4Board()
    save(games)
    with (args.output / "training.jsonl").open("w", buffering=1) as log:
        for step in range(1, args.moves + 1):
            epsilon = max(0.05, 1 - 0.95 * (step - 1) / args.decay_moves)
            labels = tactical_values(board)
            labelled += int(np.count_nonzero(np.isfinite(labels)))
            legal_actions += len(board.get_legal_moves())
            action, exploratory = learner.behavior_action(board, epsilon)
            board, experience = advance(board, action)
            loss = learner.observe(experience, labels, exploratory=exploratory,
                                   last_step=(step == args.moves))
            if loss is not None:
                total_loss += loss
                n_loss += 1
            if board.is_terminal():
                games += 1
                board = Connect4Board()
            if step % args.log_every == 0 or step == args.moves:
                info = {"moves": step, "games": games, "epsilon": epsilon,
                        "updates": learner.updates, "certified_actions": labelled,
                        "legal_actions": legal_actions, "seconds": time.monotonic() - start,
                        "mean_loss": total_loss / n_loss if n_loss else None,
                        "sampled_horizons": learner.sampled_horizons[1:].tolist()}
                log.write(json.dumps(info, allow_nan=False) + "\n")
                total_loss, n_loss = 0., 0
            if step % args.save_every == 0 or step == args.moves:
                save(games)
                print(f"Saved {step:,} moves after {games:,} complete games", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("plain", "tactical"), required=True)
    parser.add_argument("--n-step", type=int, choices=(1, 3), default=1)
    parser.add_argument("--moves", type=positive_int, default=80_000)
    parser.add_argument("--decay-moves", type=positive_int, default=64_000)
    parser.add_argument("--save-every", type=positive_int, default=20_000)
    parser.add_argument("--log-every", type=positive_int, default=1000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
