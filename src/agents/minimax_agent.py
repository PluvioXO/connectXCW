"""Minimax agent with alpha-beta pruning for Connect 4.

Uses a heuristic evaluation function based on window scoring and a
transposition table for caching. Searches in center-first column order
for better alpha-beta pruning.
"""

from __future__ import annotations

from src.agents.base_agent import Agent
from src.game.board import Connect4Board
from src.game.constants import BITS_PER_COL, COLS, ROWS, WIN_LENGTH

# Column search order: center-first for better pruning
_CENTER_FIRST: list[int] = [3, 2, 4, 1, 5, 0, 6]

# Heuristic score constants
_WIN_SCORE: float = 10_000.0
_THREE_OPEN_SCORE: float = 100.0
_TWO_OPEN_SCORE: float = 10.0
_OPP_THREE_OPEN_SCORE: float = -80.0

# Transposition table flag values
_EXACT = "exact"
_LOWER = "lower"
_UPPER = "upper"


class MinimaxAgent(Agent):
    """Alpha-beta minimax agent with heuristic evaluation.

    Args:
        max_depth: Maximum search depth (plies). Common values: 1, 3, 5.
    """

    def __init__(self, max_depth: int = 3) -> None:
        """Initialize the minimax agent.

        Args:
            max_depth: Maximum search depth in plies.
        """
        self.max_depth = max_depth
        # Transposition table: hash -> (depth, value, flag)
        self._tt: dict[int, tuple[int, float, str]] = {}

    @property
    def name(self) -> str:
        """Return the agent's display name."""
        return f"Minimax-{self.max_depth}"

    def select_move(self, board: Connect4Board) -> int:
        """Select the best move via alpha-beta minimax search.

        Args:
            board: Current game state.

        Returns:
            The column index of the best move found.
        """
        self._tt.clear()
        legal = board.get_legal_moves()

        best_col = legal[0]
        # Prefer center column as tiebreak
        for col in _CENTER_FIRST:
            if col in legal:
                best_col = col
                break

        best_score = -float("inf")
        alpha = -float("inf")
        beta = float("inf")

        for col in _CENTER_FIRST:
            if col not in legal:
                continue
            child = board.make_move(col)
            # Negate: child is opponent's turn, so negate their score
            score = -self._negamax(child, self.max_depth - 1, -beta, -alpha)
            if score > best_score:
                best_score = score
                best_col = col
            alpha = max(alpha, score)

        return best_col

    def _negamax(
        self,
        board: Connect4Board,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        """Negamax with alpha-beta pruning and transposition table.

        Returns the score from the current player's perspective.

        Args:
            board: Current game state.
            depth: Remaining search depth.
            alpha: Lower bound for pruning.
            beta: Upper bound for pruning.

        Returns:
            Score from the current player's perspective.
        """
        key = hash(board)
        tt_entry = self._tt.get(key)
        if tt_entry is not None:
            cached_depth, cached_val, flag = tt_entry
            if cached_depth >= depth:
                if flag == _EXACT:
                    return cached_val
                if flag == _LOWER:
                    alpha = max(alpha, cached_val)
                elif flag == _UPPER:
                    beta = min(beta, cached_val)
                if alpha >= beta:
                    return cached_val

        if board.is_terminal():
            winner = board.get_winner()
            if winner is None:
                return 0.0
            # Current player is the loser (the winner just moved)
            return -_WIN_SCORE

        if depth == 0:
            return _heuristic(board)

        legal = board.get_legal_moves()
        original_alpha = alpha
        best = -float("inf")

        for col in _CENTER_FIRST:
            if col not in legal:
                continue
            child = board.make_move(col)
            score = -self._negamax(child, depth - 1, -beta, -alpha)
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        # Store in transposition table
        if best <= original_alpha:
            flag = _UPPER
        elif best >= beta:
            flag = _LOWER
        else:
            flag = _EXACT
        self._tt[key] = (depth, best, flag)

        return best


def _heuristic(board: Connect4Board) -> float:
    """Score the board from the current player's perspective.

    Evaluates windows of 4 cells across all directions. Positive = good
    for the current player. Does not check terminal states.

    Args:
        board: A non-terminal game state.

    Returns:
        Heuristic score from the current player's perspective.
    """
    grid = _board_to_grid(board)
    current = board.current_player
    opponent = 3 - current

    score = 0.0
    for window in _all_windows(grid):
        score += _score_window(window, current, opponent)

    return score


def _board_to_grid(board: Connect4Board) -> list[list[int]]:
    """Convert a Connect4Board to a 2D list grid[row][col].

    Grid values: 0 = empty, 1 = Player 1's piece, 2 = Player 2's piece.

    Args:
        board: Game state to convert.

    Returns:
        List of rows (top row first), each row is a list of 7 integers.
    """
    # Reconstruct from bitboard internals via the encode method
    # encode() gives (3, ROWS, COLS) — plane 0 = current player, plane 1 = opponent
    encoded = board.encode()
    current = board.current_player
    opponent = 3 - current

    grid: list[list[int]] = [[0] * COLS for _ in range(ROWS)]
    for row in range(ROWS):
        for col in range(COLS):
            if encoded[0, row, col]:
                grid[row][col] = current
            elif encoded[1, row, col]:
                grid[row][col] = opponent

    return grid


def _all_windows(grid: list[list[int]]) -> list[list[int]]:
    """Generate all windows of length WIN_LENGTH from the board.

    Covers horizontal, vertical, and both diagonal directions.

    Args:
        grid: 2D list of piece values (0/1/2).

    Returns:
        List of windows, each a list of WIN_LENGTH integers.
    """
    windows: list[list[int]] = []

    for row in range(ROWS):
        for col in range(COLS - WIN_LENGTH + 1):
            windows.append([grid[row][col + i] for i in range(WIN_LENGTH)])

    for col in range(COLS):
        for row in range(ROWS - WIN_LENGTH + 1):
            windows.append([grid[row + i][col] for i in range(WIN_LENGTH)])

    # Diagonal down-right (\)
    for row in range(ROWS - WIN_LENGTH + 1):
        for col in range(COLS - WIN_LENGTH + 1):
            windows.append([grid[row + i][col + i] for i in range(WIN_LENGTH)])

    # Diagonal down-left (/)
    for row in range(WIN_LENGTH - 1, ROWS):
        for col in range(COLS - WIN_LENGTH + 1):
            windows.append([grid[row - i][col + i] for i in range(WIN_LENGTH)])

    return windows


def _score_window(window: list[int], current: int, opponent: int) -> float:
    """Score a single window of 4 cells from the current player's perspective.

    Args:
        window: List of WIN_LENGTH cell values (0=empty, current, opponent).
        current: Current player's piece value.
        opponent: Opponent's piece value.

    Returns:
        Score contribution from this window.
    """
    cur_count = window.count(current)
    opp_count = window.count(opponent)
    empty_count = window.count(0)

    if opp_count > 0 and cur_count > 0:
        return 0.0  # Mixed window — no score

    if cur_count == 4:
        return _WIN_SCORE
    if cur_count == 3 and empty_count == 1:
        return _THREE_OPEN_SCORE
    if cur_count == 2 and empty_count == 2:
        return _TWO_OPEN_SCORE

    if opp_count == 3 and empty_count == 1:
        return _OPP_THREE_OPEN_SCORE

    return 0.0
