"""Play the two unchanged submission archives with their real time budgets."""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import tarfile
import tempfile
import time
from types import SimpleNamespace

from kaggle_environments import make
from src.game.board import Connect4Board


def openings(count, seed):
    rng = random.Random(seed)
    result, seen = [[]], {(0, 0)}
    while len(result) < count:
        board, moves = Connect4Board(), []
        for _ in range(rng.choice((2, 4, 6, 8, 10))):
            legal = [c for c in board.get_legal_moves() if not board.make_move(c).is_terminal()]
            col = rng.choice(legal)
            moves.append(col)
            board = board.make_move(col)
        key = board._mask, board._position
        if key not in seen:
            seen.add(key)
            result.append(moves)
    return result


def load_archive(path, directory, name):
    folder = directory / name
    folder.mkdir()
    with tarfile.open(path) as archive:
        assert set(archive.getnames()) == {"main.py", "weights.npz"}
        for filename in archive.getnames():
            (folder / filename).write_bytes(archive.extractfile(filename).read())
    spec = importlib.util.spec_from_file_location(name, folder / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._WEIGHTS_PATH = str(folder / "weights.npz")
    return module


def play_game(modules, opening, candidate_side):
    env = make("connectx", debug=False)
    for col in opening:
        env.step([col, col])
        assert not env.done
    durations = [[], []]
    banks = [60., 60.]
    moves, failure = [], None
    started = time.monotonic()
    while not env.done:
        side = 0 if env.state[0].status == "ACTIVE" else 1
        owner = int(side == candidate_side)
        board = list(env.state[0].observation.board)
        observation = SimpleNamespace(board=board, mark=side + 1,
                                      remainingOverageTime=banks[owner])
        t = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                action = modules[owner].my_agent(observation, env.configuration)
            elapsed = time.perf_counter() - t
            durations[owner].append(elapsed)
            banks[owner] -= max(0, elapsed - env.configuration.actTimeout)
            if banks[owner] < 0:
                raise TimeoutError("Kaggle overage bank exhausted")
            if not isinstance(action, int) or not 0 <= action < 7 or board[action] != 0:
                raise ValueError(f"Invalid action: {action!r}")
        except Exception as exc:
            failure = {"agent": "candidate" if owner else "reference", "error": repr(exc)}
            reward = -1 if owner else 1
            break
        moves.append(action)
        env.step([action, action])
    else:
        assert all(s.status == "DONE" for s in env.state)
        reward = env.state[candidate_side].reward
        assert reward in (-1, 0, 1)
    timings = {}
    for name, values in zip(("reference", "candidate"), durations):
        timings[name] = {"moves": len(values), "max_seconds": max(values, default=0),
                         "mean_seconds": sum(values) / len(values) if values else 0,
                         "moves_over_two_seconds": sum(v > 2 for v in values)}
    return {"opening": opening, "candidate_player": candidate_side + 1,
            "candidate_reward": reward, "actions": moves, "failure": failure,
            "timings": timings, "elapsed_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    assert args.pairs > 0 and 0 <= args.worker < args.workers
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
              "worker": args.worker, "pairs_total": args.pairs, "opening_seed": 82026,
              "games": []}
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)
        modules = [load_archive(args.reference, directory, "reference"),
                   load_archive(args.candidate, directory, "candidate")]
        assert modules[0]._TIME_BUDGET_SECS == modules[1]._TIME_BUDGET_SECS == 1.9
        for index, opening in enumerate(openings(args.pairs, 82026)):
            if index % args.workers != args.worker:
                continue
            for side in (0, 1):
                for module in modules:
                    module._weights = None
                result = play_game(modules, opening, side)
                result["pair"] = index
                report["games"].append(result)
                temporary = args.output.with_suffix(".tmp")
                temporary.write_text(json.dumps(report, indent=2))
                temporary.replace(args.output)
                print("GAME", index, side + 1, result["candidate_reward"],
                      result["elapsed_seconds"], result["failure"], flush=True)


if __name__ == "__main__":
    main()
