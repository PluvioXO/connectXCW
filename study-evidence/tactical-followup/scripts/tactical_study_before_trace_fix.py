"""Freeze Connect Four evaluation positions or evaluate a trained checkpoint family."""

import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch

from scripts.dqn_study import tactical_candidates
from src.agents.minimax_agent import MinimaxAgent
from src.game.board import Connect4Board
from src.game.constants import COLS, ROWS
from src.game.endgame import action_values, replay_board
from src.training.double_dqn import make_network
from src.training.tactical_dqn import tactical_values

SEEDS = (61, 62, 63, 64, 65)
CHECKPOINTS = (0, 20_000, 40_000, 60_000, 80_000)
STRATA = ("win-now", "block-now", "multi-step")
OPENING_SEED = 417507
FINAL_SEED = 295813
ORACLE_SEED = 563829


def canonical_board(board):
    state = board.encode()
    return min(state.tobytes(), state[..., ::-1].tobytes())


def random_prefix(rng, length):
    board = Connect4Board()
    moves = []
    for _ in range(length):
        children = [(col, board.make_move(col)) for col in board.get_legal_moves()]
        live = [(col, child) for col, child in children if not child.is_terminal()]
        if not live:
            return None, None
        col, board = rng.choice(live)
        moves.append(col)
    return board, moves


def sample_openings(count, seed, excluded):
    rng = random.Random(seed)
    openings = []
    while len(openings) < count:
        board, moves = random_prefix(rng, rng.choice((2, 4, 6, 8, 10)))
        if board is None:
            continue
        key = canonical_board(board)
        if key not in excluded:
            excluded.add(key)
            openings.append(moves)
    return openings


def diagnostic_group(values, tactical):
    if len(set(values.values())) < 2:
        return None
    if np.any(tactical == 1):
        return "win-now"
    if max(values.values()) >= 0 and np.any(tactical == -1):
        return "block-now"
    if not np.isfinite(tactical).any():
        return "multi-step"
    return None


def sample_oracle(count, excluded):
    rng = random.Random(ORACLE_SEED)
    counts = Counter()
    puzzles = []
    for _ in range(200_000):
        if len(puzzles) == count * len(STRATA):
            return puzzles
        ply = ROWS * COLS - rng.choice((8, 9, 10, 11, 12))
        board, prefix = random_prefix(rng, ply)
        if board is None or len(board.get_legal_moves()) < 2:
            continue
        key = canonical_board(board)
        if key in excluded:
            continue
        labels = action_values(board)
        tactical = tactical_values(board)
        assert all(labels[c] == tactical[c] for c in labels if np.isfinite(tactical[c]))
        group = diagnostic_group(labels, tactical)
        if group is None or counts[group] >= count:
            continue
        counts[group] += 1
        excluded.add(key)
        puzzles.append({"prefix": prefix, "group": group,
                        "values_by_column": [labels.get(c) for c in range(COLS)]})
    raise RuntimeError(f"Oracle strata did not fill: {dict(counts)}")


def prepare(args):
    excluded = {canonical_board(replay_board(game["opening"]))
                for path in args.excluded_reports
                for game in json.loads(path.read_text())["games"]}
    n_excluded = len(excluded)
    curve = sample_openings(24, OPENING_SEED, excluded)
    final = sample_openings(64, FINAL_SEED, excluded)
    puzzles = sample_oracle(96, excluded)
    payload = {"train_seeds": list(SEEDS), "checkpoints": list(CHECKPOINTS),
               "curve_seed": OPENING_SEED, "final_seed": FINAL_SEED, "oracle_seed": ORACLE_SEED,
               "previous_openings_excluded_up_to_reflection": n_excluded,
               "exclusion_files_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                                          for p in args.excluded_reports},
               "curve": curve, "final": final, "oracle": puzzles,
               "primary": {"checkpoint": 80000, "opponent": "minimax-2",
                           "split": "final", "mode": "greedy",
                           "contrast": "tactical minus plain"},
               "secondary": ["train and guard ablation", "depth-three opponent",
                             "solved action ranking", "reflection-averaged Q on solved positions"]}
    with args.output.open("x") as stream:
        json.dump(payload, stream, separators=(",", ":"))
        stream.write("\n")


@torch.inference_mode()
def evaluate_games(net, mode, opponent, prefixes, seed):
    if opponent != "random":
        depth = int(opponent.rsplit("-", maxsplit=1)[1])
        opponent_move = lru_cache(16_384)(MinimaxAgent(max_depth=depth).select_move)
    results = []
    for index, prefix in enumerate(prefixes):
        start = replay_board(prefix)
        for player in (1, 2):
            other_rng = random.Random(f"opponent:{seed}:{index}:{player}")
            agent_rng = random.Random(f"agent:{seed}:{index}:{player}")
            board = start
            moves = []
            while not board.is_terminal():
                if board.current_player == player:
                    choices = board.get_legal_moves() if mode == "greedy" else tactical_candidates(board)
                    if net is None:
                        move = agent_rng.choice(choices)
                    else:
                        scores = net(torch.from_numpy(board.encode()).unsqueeze(0))[0]
                        move = choices[int(scores[choices].argmax())]
                elif opponent == "random":
                    move = other_rng.choice(board.get_legal_moves())
                else:
                    move = opponent_move(board)
                moves.append(move)
                board = board.make_move(move)
            results.append({"pair": index, "player": player, "opening": prefix,
                            "moves_after_opening": moves, "reward": int(board.get_result(player))})
    counts = Counter(r["reward"] for r in results)
    return {"games": results, "wins": counts[1], "draws": counts[0], "losses": counts[-1],
            "score": (counts[1] + counts[0] / 2) / len(results)}


@torch.inference_mode()
def evaluate_oracle(net, records):
    boards = [replay_board(r["prefix"]) for r in records]
    features = torch.from_numpy(np.stack([b.encode() for b in boards]))
    raw = net(features).numpy()
    reflected = net(features.flip(-1)).flip(-1).numpy()
    mirror_average = (raw + reflected) / 2
    result = []
    for index, board in enumerate(boards):
        legal = board.get_legal_moves()
        guarded = tactical_candidates(board)
        result.append({"id": index, "greedy": legal[int(np.argmax(raw[index, legal]))],
                       "mirrored_average": legal[int(np.argmax(mirror_average[index, legal]))],
                       "guarded": guarded[int(np.argmax(raw[index, guarded]))],
                       "q_reflection_mae": float(np.abs(raw[index, legal] - reflected[index, legal]).mean())})
    return result


def save_json(path, payload):
    with path.open("x") as stream:
        json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def jobs(step):
    yield "curve", "greedy", "minimax-2"
    if step == 0:
        yield "final", "greedy", "minimax-2"
        yield "final", "greedy", "random"
    elif step == CHECKPOINTS[-1]:
        yield "final", "greedy", "random"
        for mode in ("greedy", "guarded"):
            yield "final", mode, "minimax-2"
            yield "final", mode, "minimax-3"


def evaluate(args):
    testpack = json.loads(args.testpack.read_text())
    args.output.mkdir(parents=True)
    digest = sha256(args.testpack.read_bytes()).hexdigest()

    def series(net, split, mode, opponent, meta):
        prefixes = testpack[split]
        if opponent == "minimax-3":
            prefixes = prefixes[:32]
        result = evaluate_games(net, mode, opponent, prefixes, testpack[f"{split}_seed"])
        result.update(meta, split=split, mode=mode, opponent=opponent, testpack_sha256=digest)
        path = args.output / f"{split}-{mode}-{meta['steps']}-{opponent}.json"
        save_json(path, result)
        print(f"{path.stem}: {result['score']:.5f}", flush=True)

    if args.training is None:
        for opponent in ("random", "minimax-2", "minimax-3"):
            series(None, "final", "tactics-only", opponent, {"steps": None})
        return

    net = make_network().eval()
    steps = (args.step,) if args.step is not None else CHECKPOINTS
    for step in steps:
        path = args.training / f"move-{step:06d}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(checkpoint["weights"])
        meta = {name: checkpoint[name] for name in ("steps", "updates", "games", "seed", "variant")}
        assert meta["steps"] == step
        meta["checkpoint_sha256"] = sha256(path.read_bytes()).hexdigest()
        for split, mode, opponent in jobs(step):
            series(net, split, mode, opponent, meta)
        if step in (0, 40_000, 80_000):
            save_json(args.output / f"oracle-{step}.json", dict(
                meta, testpack_sha256=digest, actions=evaluate_oracle(net, testpack["oracle"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--excluded-reports", nargs="+", type=Path, required=True)
    setup.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("evaluate")
    run.add_argument("--training", type=Path)
    run.add_argument("--testpack", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--step", choices=CHECKPOINTS, type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
