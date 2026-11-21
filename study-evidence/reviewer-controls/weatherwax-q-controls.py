"""Freeze and run the final matched controls for the Connect Four Q learner."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.learned_lookahead import evaluate
from scripts.tactical_study import SEEDS, canonical_board, evaluate_games, sample_openings, save_json
from src.agents.minimax_agent import _heuristic
from src.game.endgame import replay_board
from src.training.double_dqn import make_network
from src.training.tactical_dqn import tactical_values

OPENING_SEED = 935072
HORIZON_SEEDS = (71, 72, 73, 74, 75)
FOLLOWUP = Path("study-evidence/tactical-followup")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def opening_pack(root):
    paths = [
        Path("pilot-evidence/ddqn-pilot-51/evaluation/random-000000.json"),
        Path("study-evidence/dqn-study/evaluation/ddqn-51/curve-network-0-random.json"),
        Path("study-evidence/dqn-study/evaluation/ddqn-51/final-network-0-random.json"),
    ]
    prior = [g["opening"] for p in paths for g in read(root / p)["games"]]
    tactical = read(root / FOLLOWUP / "testpack.json")
    prior += tactical["curve"] + tactical["final"] + [r["prefix"] for r in tactical["oracle"]]
    prior += read(root / FOLLOWUP / "lookahead-pack.json")["openings"]
    paths += [FOLLOWUP / "testpack.json", FOLLOWUP / "lookahead-pack.json"]
    excluded = {canonical_board(replay_board(moves)) for moves in prior}
    return {
        "seed": OPENING_SEED,
        "prior_files_sha256": {str(p): digest(root / p) for p in paths},
        "excluded_up_to_reflection": len(excluded),
        "openings": sample_openings(128, OPENING_SEED, excluded),
        "search_seeds": list(SEEDS), "horizon_seeds": list(HORIZON_SEEDS),
        "search_primary": "trained Q vs window heuristic in the same two plies vs minimax-3",
        "search_secondary": ["trained Q vs random Q in two plies", "two plies vs guarded trained Q"],
        "horizon_primary": "greedy up-to-three-step vs one-step DDQN vs minimax-2 at 80000 moves",
    }


def defence_pack(root):
    positions = {}
    inputs = {}
    for variant in ("plain", "tactical"):
        for seed in SEEDS:
            run = f"{variant}-{seed}"
            path = (FOLLOWUP / "evaluation-with-traces" / run / "step-80000"
                    / "final-greedy-80000-minimax-2.json")
            inputs[str(path)] = digest(root / path)
            for game in read(root / path)["games"]:
                prefix = game["opening"].copy()
                board = replay_board(prefix)
                for action in game["moves_after_opening"]:
                    legal = board.get_legal_moves()
                    assert action in legal and not board.is_terminal()
                    if len(prefix) < 30 and board.current_player == game["player"]:
                        values = tactical_values(board)
                        bad = [c for c in legal if values[c] == -1]
                        if 0 < len(bad) < len(legal) and not np.any(values == 1):
                            key = canonical_board(board)
                            if key not in positions:
                                positions[key] = {"prefix": prefix.copy(), "legal": legal,
                                                  "exposes": bad, "sources": []}
                            sources = positions[key]["sources"]
                            if run not in sources:
                                sources.append(run)
                    prefix.append(action)
                    board = board.make_move(action)
                assert board.is_terminal() and board.get_result(game["player"]) == game["reward"]
    return {"sources_sha256": inputs, "positions": list(positions.values())}


def heuristic_leaves(leaves):
    scores = [_heuristic(b) for b in leaves]
    return [h / (1 + abs(h)) for h in scores]


def load_network(path, moves):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["steps"] == moves
    net = make_network().eval()
    net.load_state_dict(checkpoint["weights"])
    return net, checkpoint


def match(pack, output, checkpoint_path, mode):
    start = time.monotonic()
    net, metadata = None, {}
    if checkpoint_path:
        moves = 0 if mode == "initial" else 80_000
        net, checkpoint = load_network(checkpoint_path, moves)
        metadata = {k: checkpoint[k] for k in ("seed", "steps", "updates", "variant")}
        metadata["checkpoint_sha256"] = digest(checkpoint_path)
        if mode == "horizon":
            metadata["n_step"] = checkpoint["n_step"]
    prefixes = read(pack)["openings"]
    if mode == "horizon":
        depth = 2
        result = evaluate_games(net, "greedy", "minimax-2", prefixes, metadata["seed"])
    else:
        depth = 3
        result = evaluate(net, mode, depth, prefixes,
                          leaf_evaluator=heuristic_leaves if mode == "heuristic" else None)
    result.update(metadata, mode=mode, depth=depth, pack_sha256=digest(pack),
                  seconds=time.monotonic() - start)
    save_json(output, result)
    print(f"{output.name}: {result['score']:.5f} in {result['seconds']:.1f}s", flush=True)


@torch.inference_mode()
def diagnose(pack, checkpoints):
    rows = read(pack)["positions"]
    boards = [replay_board(prefix) for row in rows for prefix in
              (row["prefix"], [6 - c for c in row["prefix"]])]
    states = torch.from_numpy(np.stack([b.encode() for b in boards]))
    mask = torch.zeros((len(boards), 7), dtype=torch.bool)
    for i, board in enumerate(boards):
        mask[i, board.get_legal_moves()] = True
    results = {}
    for path in checkpoints:
        net, checkpoint = load_network(path, 80_000)
        moves = [net(states[i:i+128]).masked_fill(~mask[i:i+128], -torch.inf).argmax(dim=1)
                 for i in range(0, len(states), 128)]
        name = f"{checkpoint['variant']}-{checkpoint['seed']}"
        assert name not in results
        results[name] = {"checkpoint_sha256": digest(path), "choices": torch.cat(moves).tolist()}
    return {"pack_sha256": digest(pack), "checkpoints": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("openings", "defence"):
        setup = commands.add_parser(name)
        setup.add_argument("--root", type=Path, default=Path("."))
        setup.add_argument("--output", type=Path, required=True)
    games = commands.add_parser("matches")
    games.add_argument("--mode", choices=("heuristic", "initial", "lookahead", "guarded", "horizon"),
                       required=True)
    games.add_argument("--checkpoint", type=Path)
    diagnosis = commands.add_parser("diagnose")
    diagnosis.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    for command in (games, diagnosis):
        command.add_argument("--pack", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "openings":
        save_json(args.output, opening_pack(args.root))
    elif args.command == "defence":
        save_json(args.output, defence_pack(args.root))
    elif args.command == "diagnose":
        save_json(args.output, diagnose(args.pack, args.checkpoints))
    else:
        if (args.checkpoint is None) != (args.mode == "heuristic"):
            parser.error("Specify a checkpoint for every match mode except heuristic")
        match(args.pack, args.output, args.checkpoint, args.mode)


if __name__ == "__main__":
    main()
