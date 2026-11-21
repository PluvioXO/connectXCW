"""Evaluate the inherited reference at fixed search cost on the solved test set."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from scripts.tactical_study import save_json
from src.game.endgame import replay_board
from src.mcts.search import BatchedMCTS
from src.neural_net.model import Connect4Net
from src.utils.config import MCTSConfig


@torch.inference_mode()
def evaluate(args):
    pack = json.loads(args.testpack.read_text())
    boards = [replay_board(row["prefix"]) for row in pack["oracle"]]
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    network = Connect4Net(num_blocks=checkpoint["num_blocks"],
                          num_filters=checkpoint["num_filters"],
                          input_planes=checkpoint.get("input_planes", 3))
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    search = BatchedMCTS(network, MCTSConfig(num_simulations=args.simulations))
    priors, visits = [], []
    for start in range(0, len(boards), 32):
        group = boards[start:start+32]
        logits, _ = network(torch.from_numpy(np.stack([board.encode() for board in group])))
        for board, row in zip(group, logits):
            legal = board.get_legal_moves()
            priors.append(legal[int(row[legal].argmax())])
        for board, row in zip(group, search.search_batch(group)):
            legal = board.get_legal_moves()
            visits.append(legal[int(row[legal].argmax())])
        print(f"Scored {start+len(group)} of {len(boards)} positions", flush=True)
    save_json(args.output, {"reference_sha256": sha256(args.weights.read_bytes()).hexdigest(),
                           "testpack_sha256": sha256(args.testpack.read_bytes()).hexdigest(),
                           "simulations": args.simulations, "prior": priors, "puct": visits})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--testpack", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--simulations", type=int, default=64)
    args = parser.parse_args()
    if args.simulations < 1:
        parser.error("Simulations must be positive")
    torch.set_num_threads(1)
    evaluate(args)


if __name__ == "__main__":
    main()
