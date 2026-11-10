"""Round-robin benchmark tournament for Connect 4 agents.

Usage:
    python scripts/evaluate.py                                         # no AlphaZero
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
    python scripts/evaluate.py --num-games 100 --depth 3 5
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --output logs/results.json
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from tqdm import tqdm  # noqa: E402

from src.agents.base_agent import Agent  # noqa: E402
from src.agents.mcts_agent import MCTSAgent  # noqa: E402
from src.agents.minimax_agent import MinimaxAgent  # noqa: E402
from src.agents.random_agent import RandomAgent  # noqa: E402
from src.game.board import Connect4Board  # noqa: E402
from src.game.constants import PLAYER_1, PLAYER_2  # noqa: E402


def play_game(agent1: Agent, agent2: Agent) -> int:
    """Play one game between two agents.

    Args:
        agent1: Agent playing as Player 1.
        agent2: Agent playing as Player 2.

    Returns:
        +1 if Player 1 wins, -1 if Player 2 wins, 0 for draw.
    """
    board = Connect4Board()
    while not board.is_terminal():
        if board.current_player == PLAYER_1:
            col = agent1.select_move(board)
        else:
            col = agent2.select_move(board)
        board = board.make_move(col)

    winner = board.get_winner()
    if winner == PLAYER_1:
        return 1
    elif winner == PLAYER_2:
        return -1
    return 0


def run_matchup(
    agent_a: Agent,
    agent_b: Agent,
    num_games_per_side: int,
) -> tuple[int, int, int]:
    """Run a symmetric matchup between two agents.

    Each agent plays num_games_per_side games as P1 and num_games_per_side as P2.
    Results are from agent_a's perspective.

    Args:
        agent_a: First agent.
        agent_b: Second agent.
        num_games_per_side: Number of games per side (total = 2x this).

    Returns:
        Tuple of (wins_a, draws, losses_a).
    """
    wins = 0
    draws = 0
    losses = 0
    total = num_games_per_side * 2

    with tqdm(total=total, desc=f"{agent_a.name} vs {agent_b.name}", leave=False) as pbar:
        # agent_a as Player 1
        for _ in range(num_games_per_side):
            result = play_game(agent_a, agent_b)
            if result == 1:
                wins += 1
            elif result == 0:
                draws += 1
            else:
                losses += 1
            pbar.update(1)

        # agent_a as Player 2
        for _ in range(num_games_per_side):
            result = play_game(agent_b, agent_a)
            if result == -1:
                wins += 1
            elif result == 0:
                draws += 1
            else:
                losses += 1
            pbar.update(1)

    return wins, draws, losses


def _build_agent_pool(args: argparse.Namespace) -> list[Agent]:
    """Construct the list of agents to benchmark.

    Args:
        args: Parsed CLI arguments.

    Returns:
        List of agent instances.
    """
    agents: list[Agent] = []

    if not args.az_only:
        agents.append(RandomAgent())

    depths = args.depth if args.depth else [1, 3, 5]
    for d in depths:
        agents.append(MinimaxAgent(max_depth=d))

    if not args.az_only:
        agents.append(MCTSAgent(num_simulations=args.mcts_sims))

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"Warning: checkpoint not found: {ckpt_path}. Skipping AlphaZero.")
        else:
            from src.agents.alphazero_agent import AlphaZeroAgent
            from src.utils.config import MCTSConfig
            config = MCTSConfig(num_simulations=args.mcts_sims)
            agents.append(AlphaZeroAgent(ckpt_path, mcts_config=config))

    return agents


def _print_results(results: list[dict]) -> None:
    """Print a formatted results table.

    Args:
        results: List of matchup result dicts.
    """
    header = f"{'Matchup':<40} {'Wins':>6} {'Draws':>6} {'Losses':>7} {'Win%':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        total = r["wins"] + r["draws"] + r["losses"]
        win_pct = 100.0 * r["wins"] / total if total > 0 else 0.0
        matchup = f"{r['agent_a']} vs {r['agent_b']}"
        print(
            f"{matchup:<40} {r['wins']:>6} {r['draws']:>6} {r['losses']:>7} {win_pct:>6.1f}%"
        )
    print("=" * len(header))


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Round-robin benchmark tournament for Connect 4 agents."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to AlphaZero model checkpoint .pt file.",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=50,
        help="Number of games per matchup per side (default: 50). Total = 2x this.",
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=200,
        help="MCTS simulations for MCTS-based agents (default: 200).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        nargs="+",
        default=None,
        help="Minimax depths to include (default: 1 3 5). E.g. --depth 1 3",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON results.",
    )
    parser.add_argument(
        "--az-only",
        action="store_true",
        default=False,
        help="Only run AlphaZero vs each opponent (skips Random, MCTS, and Minimax vs Minimax).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: build agents, run tournament, print results."""
    args = _parse_args()
    agents = _build_agent_pool(args)

    print(f"\nAgents in tournament: {[a.name for a in agents]}")
    print(f"Games per matchup (each side): {args.num_games} (total: {args.num_games * 2})\n")

    results: list[dict] = []

    if args.az_only and args.checkpoint:
        # Only run AlphaZero vs each opponent, skipping Minimax vs Minimax.
        az_agent = agents[-1]
        opponents = agents[:-1]
        for opp in opponents:
            wins, draws, losses = run_matchup(az_agent, opp, args.num_games)
            results.append(
                {
                    "agent_a": az_agent.name,
                    "agent_b": opp.name,
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                }
            )
    else:
        # Round-robin: each pair plays once.
        for i, agent_a in enumerate(agents):
            for agent_b in agents[i + 1:]:
                wins, draws, losses = run_matchup(agent_a, agent_b, args.num_games)
                results.append(
                    {
                        "agent_a": agent_a.name,
                        "agent_b": agent_b.name,
                        "wins": wins,
                        "draws": draws,
                        "losses": losses,
                    }
                )

    _print_results(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
