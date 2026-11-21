"""Train or evaluate the separate Connect Four DQN agent."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import torch

from src.agents.minimax_agent import MinimaxAgent
from src.game.board import Connect4Board
from src.training.double_dqn import Learner, advance, greedy_move, make_network


def train(args):
    args.output.mkdir(parents=True)
    learner = Learner(args.device, args.seed, double=args.method == "ddqn")
    config = {k: v for k, v in vars(args).items() if k != "output"}
    config.update(torch_version=torch.__version__, batch_size=learner.batch_size,
                  warmup=learner.warmup, sync_every=learner.sync_every,
                  update_every=learner.update_every, replay_capacity=learner.replay_capacity)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def save(game):
        path = args.output / f"game-{game:06d}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save({"weights": learner.online.state_dict(), "games": game,
                    "steps": learner.steps, "updates": learner.updates,
                    "method": args.method, "seed": args.seed}, temporary)
        temporary.replace(path)

    save(0)
    with (args.output / "training.jsonl").open("w", buffering=1) as log:
        for game in range(1, args.games + 1):
            epsilon = max(0.05, 1 - 0.95 * (game - 1) / args.decay_games)
            board = Connect4Board()
            losses = []
            while not board.is_terminal():
                board, transition = advance(board, learner.action(board, epsilon))
                loss = learner.observe(transition)
                if loss is not None:
                    losses.append(loss)
            log.write(json.dumps({"game": game, "steps": learner.steps,
                                  "updates": learner.updates, "epsilon": epsilon,
                                  "winner": board.get_winner(),
                                  "mean_loss": sum(losses) / len(losses) if losses else None}) + "\n")
            if game % args.save_every == 0 or game == args.games:
                save(game)
                print(f"Saved game {game}, {learner.steps} moves, {learner.updates} updates",
                      flush=True)


def paired_openings(count, seed, excluded=()):
    rng = random.Random(seed)
    seen = set(excluded)
    emitted = 0
    while emitted < count:
        board, prefix = Connect4Board(), []
        length = 0 if not seen else rng.choice((2, 4, 6, 8, 10))
        for _ in range(length):
            candidates = [(c, board.make_move(c)) for c in board.get_legal_moves()]
            col, board = rng.choice([(c, b) for c, b in candidates if not b.is_terminal()])
            prefix.append(col)
        if board not in seen:
            seen.add(board)
            emitted += 1
            yield board, prefix


def evaluate(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    network = make_network()
    network.load_state_dict(checkpoint["weights"])
    network.to(args.device).eval()
    opponent = MinimaxAgent(max_depth=2)
    excluded = set()
    for report_path in args.exclude_openings_from:
        for game in json.loads(report_path.read_text())["games"]:
            board = Connect4Board()
            for column in game["opening"]:
                board = board.make_move(column)
            excluded.add(board)
    games = []
    for pair, (start, prefix) in enumerate(paired_openings(args.pairs, args.seed, excluded)):
        for player in (1, 2):
            rng = random.Random(f"{args.seed}:{pair}:{player}")
            board, moves = start, []
            while not board.is_terminal():
                if board.current_player == player:
                    action = greedy_move(network, board, args.device)
                elif args.opponent == "random":
                    action = rng.choice(board.get_legal_moves())
                else:
                    action = opponent.select_move(board)
                moves.append(action)
                board = board.make_move(action)
            games.append({"pair": pair, "opening": prefix, "player": player,
                          "moves_after_opening": moves, "reward": board.get_result(player)})

    counts = Counter(game["reward"] for game in games)
    report = {"checkpoint": args.checkpoint.name,
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "training_seed": checkpoint["seed"], "method": checkpoint["method"],
              "training_games": checkpoint["games"], "opponent": args.opponent,
              "opening_seed": args.seed, "pairs": args.pairs,
              "excluded_opening_boards": len(excluded),
              "wins": counts[1], "draws": counts[0], "losses": counts[-1],
              "score": (counts[1] + 0.5 * counts[0]) / len(games), "games": games}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Score: {report['score']:.1%} over {len(games)} games")


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--method", choices=("ddqn", "dqn"), default="ddqn")
    training.add_argument("--games", type=positive_int, required=True)
    training.add_argument("--decay-games", type=positive_int, required=True)
    training.add_argument("--save-every", type=positive_int, default=100)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--opponent", choices=("random", "minimax"), required=True)
    evaluation.add_argument("--pairs", type=positive_int, default=64)
    evaluation.add_argument("--exclude-openings-from", type=Path, action="append", default=[])
    for command in (training, evaluation):
        command.add_argument("--seed", type=int, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
