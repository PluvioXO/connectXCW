"""Evaluate fixed Connect Four checkpoints and a two-ply tactical filter."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random

import torch

from scripts.double_dqn import paired_openings
from src.agents.minimax_agent import MinimaxAgent
from src.game.board import Connect4Board
from src.training.double_dqn import make_network


def tactical_candidates(board):
    children = {move: board.make_move(move) for move in board.get_legal_moves()}
    wins = [move for move, child in children.items()
            if child.get_winner() == board.current_player]
    if wins:
        return wins
    safe = [move for move, child in children.items() if child.is_terminal() or all(
        child.make_move(reply).get_winner() != child.current_player
        for reply in child.get_legal_moves()
    )]
    return safe or list(children)


def opening_sets(exclusion):
    pilot = set()
    for game in json.loads(exclusion.read_text())["games"]:
        board = Connect4Board()
        for column in game["opening"]:
            board = board.make_move(column)
        pilot.add(board)
    curve = list(paired_openings(32, 621903, pilot))
    held_out = list(paired_openings(64, 736401, pilot | {board for board, _ in curve}))
    return curve, held_out


@torch.inference_mode()
def evaluate(network, mode, opponent, openings, seed):
    minimax = MinimaxAgent(max_depth=2)
    games = []
    for pair, (start, prefix) in enumerate(openings):
        for player in (1, 2):
            opponent_rng = random.Random(f"opponent:{seed}:{pair}:{player}")
            candidate_rng = random.Random(f"candidate:{seed}:{pair}:{player}")
            board, moves = start, []
            while not board.is_terminal():
                if board.current_player == player:
                    legal = (board.get_legal_moves() if mode == "network"
                             else tactical_candidates(board))
                    if mode == "tactics-only":
                        action = candidate_rng.choice(legal)
                    else:
                        state = torch.from_numpy(board.encode()).unsqueeze(0)
                        action = legal[int(network(state)[0, legal].argmax())]
                elif opponent == "random":
                    action = opponent_rng.choice(board.get_legal_moves())
                else:
                    action = minimax.select_move(board)
                moves.append(action)
                board = board.make_move(action)
            games.append({"pair": pair, "opening": prefix, "player": player,
                          "moves_after_opening": moves, "reward": board.get_result(player)})
    counts = Counter(game["reward"] for game in games)
    return {"wins": counts[1], "draws": counts[0], "losses": counts[-1],
            "score": (counts[1] + 0.5 * counts[0]) / len(games), "games": games}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--excluded-pilot-openings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    curve, held_out = opening_sets(args.excluded_pilot_openings)
    if args.training is None:
        jobs = [("final", held_out, None, "tactics-only")]
    else:
        jobs = [("curve", curve, game, "network") for game in range(0, 5001, 1000)]
        jobs += [("final", held_out, game, mode)
                 for game in (0, 5000) for mode in ("network", "guarded")]

    network = make_network().eval() if args.training else None
    for split, openings, game, mode in jobs:
        meta = {"training_games": game, "checkpoint_sha256": None}
        if game is not None:
            checkpoint_path = args.training / f"game-{game:06d}.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            network.load_state_dict(checkpoint["weights"])
            meta.update(checkpoint_sha256=sha256(checkpoint_path.read_bytes()).hexdigest(),
                        method=checkpoint["method"], seed=checkpoint["seed"],
                        moves=checkpoint["steps"], updates=checkpoint["updates"])
        for opponent in ("random", "minimax"):
            result = evaluate(network, mode, opponent, openings,
                              621903 if split == "curve" else 736401)
            result.update(meta, split=split, mode=mode, opponent=opponent)
            name = f"{split}-{mode}-{game}-{opponent}"
            with (args.output / f"{name}.json").open("x") as output:
                json.dump(result, output, separators=(",", ":"))
                output.write("\n")
            print(f"{name}: {result['score']:.4f}", flush=True)


if __name__ == "__main__":
    main()
