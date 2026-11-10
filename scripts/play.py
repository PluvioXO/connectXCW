"""Terminal play interface for Connect 4 AlphaZero agent.

Usage:
    python scripts/play.py                                         # auto-discover checkpoint
    python scripts/play.py --checkpoint checkpoints/baseline_b2_f32.pt
    python scripts/play.py --checkpoint checkpoints/best_model.pt --sims 100
    python scripts/play.py --no-mcts                               # policy-only (instant moves)
"""

import argparse
import sys
from pathlib import Path

# Allow imports from the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch  # noqa: E402

from src.game.board import Connect4Board  # noqa: E402
from src.game.constants import PLAYER_1, PLAYER_2  # noqa: E402
from src.mcts.search import MCTS, select_move  # noqa: E402
from src.neural_net.model import Connect4Net  # noqa: E402
from src.utils.config import MCTSConfig  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Play Connect 4 against a trained AlphaZero agent."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint .pt file. Auto-discovered if omitted.",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=50,
        help="MCTS simulations per AI move (default: 50). Higher = stronger but slower.",
    )
    parser.add_argument(
        "--no-mcts",
        action="store_true",
        help="Use raw policy head only (instant moves, no MCTS search).",
    )
    return parser.parse_args()


def _find_checkpoint(explicit: str | None) -> Path:
    """Locate a checkpoint file, or exit with a helpful message.

    Args:
        explicit: Explicit path provided by the user, or None for auto-discovery.

    Returns:
        Path to an existing checkpoint file.
    """
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"Checkpoint not found: {p}")
        return p

    root = _PROJECT_ROOT / "checkpoints"
    candidates = sorted(root.glob("baseline_b*.pt"))
    best = root / "best_model.pt"
    if best.exists():
        candidates.append(best)

    for p in candidates:
        if p.exists():
            return p

    sys.exit(
        "No checkpoint found in checkpoints/.\n"
        "Run the baseline training first:\n"
        "  pytest tests/test_training.py -v -k slow -s\n"
        "Or specify a path:\n"
        "  python scripts/play.py --checkpoint path/to/model.pt"
    )


def _load_model(path: Path) -> tuple[Connect4Net, dict]:
    """Load a model from a checkpoint file.

    Handles both the baseline format (from the slow integration test) and
    the coach format (from scripts/train.py).

    Args:
        path: Path to the checkpoint .pt file.

    Returns:
        Tuple of (loaded model in eval mode, raw checkpoint dict).
    """
    ckpt = torch.load(path, weights_only=False)
    model = Connect4Net(
        num_blocks=ckpt["num_blocks"],
        num_filters=ckpt["num_filters"],
        input_planes=ckpt.get("input_planes", 3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _get_ai_move(
    board: Connect4Board,
    model: Connect4Net,
    mcts_config: MCTSConfig,
    no_mcts: bool,
) -> int:
    """Compute the AI's chosen column.

    Args:
        board: Current game state.
        model: Trained neural network.
        mcts_config: MCTS configuration parameters.
        no_mcts: If True, use raw policy head only.

    Returns:
        Column index (0-6) chosen by the AI.
    """
    if no_mcts:
        state_np = board.encode()
        state = torch.tensor(state_np.tolist(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(state)
        mask = torch.full((7,), float("-inf"))
        for c in board.get_legal_moves():
            mask[c] = logits[0, c]
        col = int(torch.argmax(mask).item())
    else:
        print("AI is thinking...", end="\r", flush=True)
        mcts = MCTS(model, mcts_config)
        visits = mcts.search(board, add_dirichlet_noise=False)
        col = select_move(visits, temperature=0.1)

    print(f"AI plays column {col}  ")  # trailing spaces clear "thinking..." line
    return col


def _prompt_human(board: Connect4Board) -> int:
    """Prompt the human player to enter a column, with validation.

    Args:
        board: Current game state (used to determine legal moves).

    Returns:
        A legal column index chosen by the human.
    """
    legal = board.get_legal_moves()
    while True:
        raw = input(f"Your move {legal}: ").strip()
        if raw.isdigit() and int(raw) in legal:
            return int(raw)
        print(f"  Invalid. Choose from {legal}.")


def _print_result(board: Connect4Board, human_player: int) -> None:
    """Print the game outcome from the human's perspective.

    Args:
        board: Terminal game state.
        human_player: PLAYER_1 or PLAYER_2 constant for the human.
    """
    winner = board.get_winner()
    if winner is None:
        print("\nIt's a draw.")
    elif winner == human_player:
        print("\nYou win!")
    else:
        print("\nAI wins.")


def _play_game(
    model: Connect4Net,
    human_player: int,
    mcts_config: MCTSConfig,
    no_mcts: bool,
) -> None:
    """Run one full game to completion.

    Args:
        model: Trained neural network.
        human_player: PLAYER_1 or PLAYER_2 — which side the human controls.
        mcts_config: MCTS configuration parameters.
        no_mcts: If True, AI uses raw policy head only.
    """
    board = Connect4Board()
    human_symbol = "X" if human_player == PLAYER_1 else "O"
    ai_symbol = "O" if human_symbol == "X" else "X"
    print(f"\nYou are {human_symbol}. AI is {ai_symbol}.\n")

    while not board.is_terminal():
        print(board)
        if board.current_player == human_player:
            col = _prompt_human(board)
        else:
            col = _get_ai_move(board, model, mcts_config, no_mcts)
        board = board.make_move(col)

    print(board)
    _print_result(board, human_player)


def main() -> None:
    """Entry point: parse args, load model, run play loop."""
    args = _parse_args()

    checkpoint_path = _find_checkpoint(args.checkpoint)
    model, ckpt = _load_model(checkpoint_path)

    # Print model info banner
    blocks = ckpt["num_blocks"]
    filters = ckpt["num_filters"]
    trained = ckpt.get("trained_utc", ckpt.get("iteration", "unknown"))
    label = f"iter={trained}" if isinstance(trained, int) else f"trained={trained}"
    mode = "policy-only" if args.no_mcts else f"MCTS {args.sims} sims"
    print(f"\nConnect 4 — AlphaZero ({blocks} blocks, {filters} filters, {label})")
    print(f"AI mode: {mode}")
    print(f"Checkpoint: {checkpoint_path.name}\n")

    mcts_config = MCTSConfig(num_simulations=args.sims)

    # Ask who goes first
    first = input("Who goes first? [h]uman/[a]i (default: human): ").strip().lower()
    human_player = PLAYER_2 if first.startswith("a") else PLAYER_1

    while True:
        _play_game(model, human_player, mcts_config, args.no_mcts)
        again = input("\nPlay again? [y/N]: ").strip().lower()
        if again != "y":
            break
        # Swap sides for rematch
        swap = input("Swap sides? [y/N]: ").strip().lower()
        if swap == "y":
            human_player = PLAYER_2 if human_player == PLAYER_1 else PLAYER_1

    print("Thanks for playing!")


if __name__ == "__main__":
    main()
