"""Bounded fine-tuning experiment using connect4-alphazero's existing code."""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from src.game.board import Connect4Board
from src.mcts.search import BatchedMCTS
from src.neural_net.model import Connect4Net
from src.training.replay_buffer import ReplayBuffer
from src.training.self_play import BatchedSelfPlay
from src.training.trainer import Trainer
from src.utils.config import MCTSConfig, TrainingConfig


def save_checkpoint(model, path, **metadata):
    payload = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "num_blocks": 5,
        "num_filters": 128,
        **metadata,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = Connect4Net(num_blocks=checkpoint["num_blocks"],
                        num_filters=checkpoint["num_filters"]).cuda().eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def opening_boards(count, seed):
    rng = random.Random(seed)
    boards = [Connect4Board()] if count else []
    seen = {(0, 0)}
    while len(boards) < count:
        board = Connect4Board()
        for _ in range(rng.choice((2, 4, 6, 8, 10))):
            moves = [board.make_move(c) for c in board.get_legal_moves()]
            moves = [b for b in moves if not b.is_terminal()]
            if not moves:
                break
            board = rng.choice(moves)
        key = (board._mask, board._position)
        if key not in seen:
            seen.add(key)
            boards.append(board)
    return boards


def arena(candidate, baseline, pairs, sims, seed, deadline, baseline_sims=None):
    """Play both sides of every opening, batching independent game trees."""
    searches = [BatchedMCTS(baseline, MCTSConfig(num_simulations=baseline_sims or sims)),
                BatchedMCTS(candidate, MCTSConfig(num_simulations=sims))]
    games = [(board, side) for board in opening_boards(pairs, seed) for side in (1, 2)]
    wins = losses = draws = 0
    while games:
        if time.monotonic() >= deadline:
            raise TimeoutError("Experiment time budget reached")
        moves = {}
        for owner in (0, 1):
            indices = [i for i, (b, side) in enumerate(games)
                       if int(b.current_player == side) == owner]
            if indices:
                policies = searches[owner].search_batch([games[i][0] for i in indices])
                moves.update({i: int(np.argmax(p)) for i, p in zip(indices, policies)})
        active = []
        for index, (board, side) in enumerate(games):
            board = board.make_move(moves[index])
            if board.is_terminal():
                winner = board.get_winner()
                draws += int(winner is None)
                wins += int(winner == side)
                losses += int(winner is not None and winner != side)
            else:
                active.append((board, side))
        games = active
    assert wins + losses + draws == pairs * 2
    return {"wins": wins, "losses": losses, "draws": draws,
            "score": (wins + draws / 2) / (pairs * 2), "games": pairs * 2,
            "simulations": sims, "baseline_simulations": baseline_sims or sims,
            "opening_seed": seed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seconds", type=int, default=1050)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--baseline-sims", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.seconds
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError("Stopped at time limit")))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    baseline = load_model("models/large.pt")
    report = {"seed": args.seed, "learning_rate": args.lr,
              "starting_checkpoint": "models/large.pt", "rounds": []}
    try:
        if args.evaluate:
            report["evaluation"] = arena(load_model(args.evaluate), baseline,
                                         args.pairs, args.sims, args.seed, deadline,
                                         args.baseline_sims)
        else:
            candidate = copy.deepcopy(baseline)
            # Paired deterministic play must score identical models equally.
            check = arena(candidate, baseline, 2, 8, 123, deadline)
            assert check["wins"] == check["losses"] and check["score"] == 0.5, check
            print("ARENA_SELF_CHECK", check, flush=True)
            trainer = Trainer(candidate, TrainingConfig(batch_size=256, learning_rate=args.lr))
            replay = ReplayBuffer(100000)
            for iteration in range(1, 4):
                if deadline - time.monotonic() < 180:
                    break
                t = time.monotonic()
                samples = BatchedSelfPlay(candidate.eval(), MCTSConfig(num_simulations=128),
                                          batch_size=32).generate_games(96)
                for sample in samples:
                    replay.add(sample)
                print("SELF_PLAY", iteration, len(samples), time.monotonic() - t, flush=True)
                if time.monotonic() >= deadline:
                    raise TimeoutError("Time budget reached after self-play")
                metrics = trainer.train(replay, epochs=4)
                assert all(np.isfinite(v) for v in metrics.values()), metrics
                candidate.eval()
                path = args.output / f"candidate_{iteration}.pt"
                save_checkpoint(candidate, path, iteration=iteration, seed=args.seed,
                                learning_rate=args.lr, optimizer_state_dict=trainer.optimizer.state_dict())
                result = arena(candidate, baseline, 16, 64, 12345, deadline)
                report["rounds"].append({"iteration": iteration, "checkpoint": str(path),
                                         "new_samples": len(samples), "buffer_samples": len(replay),
                                         "loss": metrics, "arena": result})
                print("ROUND", json.dumps(report["rounds"][-1]), flush=True)
                (args.output / "report.json").write_text(json.dumps(report, indent=2))
    except TimeoutError as exc:
        report["stop_reason"] = str(exc)
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print("FINISHED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
