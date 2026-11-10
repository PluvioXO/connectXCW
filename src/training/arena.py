"""Arena evaluation: pits two models against each other.

Each model pair plays num_games games total: half with model_new as
Player 1 and half with model_old as Player 1. Results are used to
decide whether to accept the new model.
"""

from src.game.board import Connect4Board
from src.game.constants import PLAYER_1, PLAYER_2
from src.mcts.search import MCTS, select_move
from src.neural_net.model import Connect4Net
from src.utils.config import MCTSConfig

# Fixed low temperature for near-deterministic arena play
_ARENA_TEMPERATURE: float = 0.1


def pit(
    model_new: Connect4Net,
    model_old: Connect4Net,
    num_games: int,
    mcts_config: MCTSConfig,
) -> tuple[int, int, int]:
    """Pit two models against each other over num_games games.

    Games are split evenly: half with model_new as P1, half as P2.
    If num_games is odd, the extra game is assigned to model_new as P1.

    Args:
        model_new: Candidate new model.
        model_old: Current best model.
        num_games: Total number of games to play.
        mcts_config: MCTS configuration (simulations, c_puct, etc.).
            Temperature is overridden to 0.1 for arena play.

    Returns:
        Tuple of (wins_new, losses_new, draws) from model_new's perspective.
        wins_new + losses_new + draws == num_games.
    """
    mcts_new = MCTS(model_new, mcts_config)
    mcts_old = MCTS(model_old, mcts_config)

    wins_new = 0
    losses_new = 0
    draws = 0

    games_as_p1 = (num_games + 1) // 2  # model_new is P1
    games_as_p2 = num_games // 2        # model_new is P2

    for _ in range(games_as_p1):
        result = _play_one_game(mcts_new, mcts_old, _ARENA_TEMPERATURE)
        if result == 1:
            wins_new += 1
        elif result == -1:
            losses_new += 1
        else:
            draws += 1

    for _ in range(games_as_p2):
        result = _play_one_game(mcts_old, mcts_new, _ARENA_TEMPERATURE)
        # model_new is P2; result is from P1's perspective, so flip sign
        if result == -1:
            wins_new += 1
        elif result == 1:
            losses_new += 1
        else:
            draws += 1

    return wins_new, losses_new, draws


def _play_one_game(mcts_p1: MCTS, mcts_p2: MCTS, temperature: float) -> int:
    """Play a single game between two MCTS agents.

    Args:
        mcts_p1: MCTS instance for Player 1.
        mcts_p2: MCTS instance for Player 2.
        temperature: Temperature for move selection.

    Returns:
        +1 if Player 1 wins, -1 if Player 2 wins, 0 for a draw.
    """
    board = Connect4Board()

    while not board.is_terminal():
        if board.current_player == PLAYER_1:
            visit_counts = mcts_p1.search(board, add_dirichlet_noise=False)
        else:
            visit_counts = mcts_p2.search(board, add_dirichlet_noise=False)

        col = select_move(visit_counts, temperature)
        board = board.make_move(col)

    winner = board.get_winner()
    if winner == PLAYER_1:
        return 1
    elif winner == PLAYER_2:
        return -1
    return 0
