"""Compare a learned two-ply leaf evaluator with a tactical filter."""

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
from scripts.tactical_study import SEEDS, canonical_board, sample_openings, save_json
from src.agents.minimax_agent import MinimaxAgent
from src.game.endgame import replay_board
from src.training.double_dqn import make_network

OPENING_SEED = 174963


@torch.inference_mode()
def two_ply_move(network, board, *, leaf_evaluator=None):
    mover = board.current_player
    choices = board.get_legal_moves()
    branches = []
    leaves = []
    for action in choices:
        child = board.make_move(action)
        if child.is_terminal():
            result = child.get_result(mover)
            if result == 1:
                return action
            branches.append(result)
            continue
        replies = [child.make_move(reply) for reply in child.get_legal_moves()]
        if any(reply.get_winner() == 3 - mover for reply in replies):
            branches.append(-1.)
            continue
        start = len(leaves)
        leaves.extend(reply for reply in replies if not reply.is_terminal())
        branches.append((start, len(leaves), any(reply.is_terminal() for reply in replies)))

    estimates = []
    if leaves:
        if leaf_evaluator is None:
            values = network(torch.from_numpy(np.stack([leaf.encode() for leaf in leaves]))).numpy()
            estimates = [float(row[leaf.get_legal_moves()].max()) for row, leaf in zip(values, leaves)]
        else:
            estimates = leaf_evaluator(leaves)
    scores = []
    for branch in branches:
        if isinstance(branch, tuple):
            lo, hi, has_draw = branch
            scores.append(min(estimates[lo:hi] + ([0.] if has_draw else [])))
        else:
            scores.append(branch)
    return choices[int(np.argmax(scores))]


@torch.inference_mode()
def evaluate(network, mode, depth, prefixes, *, leaf_evaluator=None):
    minimax = lru_cache(16_384)(MinimaxAgent(max_depth=depth).select_move)
    results = []
    for index, prefix in enumerate(prefixes):
        initial = replay_board(prefix)
        for seat in (1, 2):
            board = initial
            moves = []
            rng = random.Random(f"lookahead:{OPENING_SEED}:{index}:{seat}")
            while not board.is_terminal():
                if board.current_player != seat:
                    action = minimax(board)
                elif mode == "tactics-only":
                    action = rng.choice(tactical_candidates(board))
                elif mode == "guarded":
                    choices = tactical_candidates(board)
                    scores = network(torch.from_numpy(board.encode()).unsqueeze(0))[0]
                    action = choices[int(scores[choices].argmax())]
                else:
                    action = two_ply_move(network, board, leaf_evaluator=leaf_evaluator)
                moves.append(action)
                board = board.make_move(action)
            results.append({"pair": index, "player": seat, "opening": prefix,
                            "moves_after_opening": moves, "reward": int(board.get_result(seat))})
    counts = Counter(g["reward"] for g in results)
    return {"games": results, "score": (counts[1] + counts[0] / 2) / len(results),
            "wins": counts[1], "draws": counts[0], "losses": counts[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--testpack", type=Path, required=True)
    setup.add_argument("--prior-reports", nargs="+", type=Path, required=True)
    setup.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("evaluate")
    run.add_argument("--pack", type=Path, required=True)
    run.add_argument("--training", type=Path)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        testpack = json.loads(args.testpack.read_text())
        prior = [game["opening"] for path in args.prior_reports
                 for game in json.loads(path.read_text())["games"]]
        prior += testpack["curve"] + testpack["final"]
        excluded = {canonical_board(replay_board(moves)) for moves in prior}
        save_json(args.output, {"seed": OPENING_SEED, "seeds": list(SEEDS),
                                "references_testpack_sha256": sha256(args.testpack.read_bytes()).hexdigest(),
                                "openings": sample_openings(24, OPENING_SEED, excluded)})
        return

    torch.set_num_threads(1)
    pack = json.loads(args.pack.read_text())
    args.output.mkdir(parents=True)
    pack_hash = sha256(args.pack.read_bytes()).hexdigest()
    if args.training:
        net = make_network().eval()
        seeds = [(80_000, "guarded"), (80_000, "lookahead")]
        if args.training.parent.name.startswith("plain-"):
            seeds += [(0, "lookahead")]
    else:
        net, seeds = None, [(None, "tactics-only")]

    for step, mode in seeds:
        metadata = {"moves": step, "mode": mode, "pack_sha256": pack_hash}
        if step is not None:
            path = args.training / f"move-{step:06d}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            assert checkpoint["steps"] == step
            net.load_state_dict(checkpoint["weights"])
            metadata.update(seed=checkpoint["seed"], variant=checkpoint["variant"],
                            checkpoint_sha256=sha256(path.read_bytes()).hexdigest())
        for depth in (2, 3):
            result = evaluate(net, mode, depth, pack["openings"])
            result.update(metadata, depth=depth)
            path = args.output / f"{mode}-{step}-minimax-{depth}.json"
            save_json(path, result)
            print(f"{path.stem}: {result['score']:.5f}", flush=True)


if __name__ == "__main__":
    main()
