"""Connect 4 bitboard game engine.

Bitboard layout — each column uses 7 bits (6 rows + 1 sentinel):

  bit positions:
   5 12 19 26 33 40 47
   4 11 18 25 32 39 46
   3 10 17 24 31 38 45
   2  9 16 23 30 37 44
   1  8 15 22 29 36 43
   0  7 14 21 28 35 42

Win detection uses bit-shifting and AND:
  Horizontal: shift 7  (one column right)
  Vertical:   shift 1  (one row up)
  Diagonal \\: shift 8  (one column right + one row up)
  Diagonal /: shift 6  (one column right + one row down)

State representation:
  _position — current player's pieces
  _mask     — all placed pieces (both players combined)

After each make_move call, the roles swap: the new _position holds
the *previous* player's pieces (which are now the opponent of the
next mover).  This XOR trick is from Pascal Pons' C++ solver.
"""

from __future__ import annotations

import numpy as np

from src.game.constants import (
    BITS_PER_COL,
    COLS,
    COLUMN_MASK,
    PLAYER_1,
    PLAYER_2,
    ROWS,
    WIN_LENGTH,
)


class Connect4Board:
    """Immutable Connect 4 board using a bitboard representation.

    Two Python ints represent each player's pieces. Moves return new
    board instances; the original is never mutated.

    Attributes:
        _mask: Bitmask of all placed pieces (both players).
        _position: Bitmask of the current player's pieces.
        _num_moves: Total number of moves played so far.
    """

    def __init__(self) -> None:
        """Create an empty board. Player 1 moves first."""
        self._mask: int = 0
        self._position: int = 0
        self._num_moves: int = 0

    @classmethod
    def _from_state(cls, mask: int, position: int, num_moves: int) -> "Connect4Board":
        """Create a board directly from raw bitboard state.

        Args:
            mask: Bitmask of all placed pieces.
            position: Bitmask of the current player's pieces.
            num_moves: Number of moves played so far.

        Returns:
            A Connect4Board initialised with the given state.
        """
        board = cls.__new__(cls)
        board._mask = mask
        board._position = position
        board._num_moves = num_moves
        return board

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_player(self) -> int:
        """Return 1 or 2 — the player whose turn it is to move."""
        return PLAYER_1 if self._num_moves % 2 == 0 else PLAYER_2

    def make_move(self, column: int) -> "Connect4Board":
        """Place the current player's piece in the specified column.

        The piece drops to the lowest available row (gravity). Returns
        a new board; the original is unchanged (immutable).

        Args:
            column: Column index (0–6) to place the piece.

        Returns:
            A new Connect4Board with the piece placed and the current
            player toggled.

        Raises:
            ValueError: If the column is full or out of range.
        """
        if column < 0 or column >= COLS:
            raise ValueError(f"Column {column} is out of range (0–{COLS - 1}).")
        if not self._column_has_space(column):
            raise ValueError(f"Column {column} is full.")

        # The lowest empty bit in this column sits just above the current top.
        new_piece = (self._mask + self._bottom_mask_col(column)) & self._col_mask(column)
        new_mask = self._mask | new_piece

        # XOR trick: new_position becomes the *opponent's* pieces, which
        # are the new current player's pieces after the turn flips.
        # opponent_pieces = self._mask ^ self._position (unchanged by this move)
        new_position = self._mask ^ self._position

        return Connect4Board._from_state(
            mask=new_mask,
            position=new_position,
            num_moves=self._num_moves + 1,
        )

    def get_legal_moves(self) -> list[int]:
        """Return list of column indices that are not full.

        Returns:
            List of integers in range [0, 6] representing playable columns,
            in ascending order.
        """
        return [col for col in range(COLS) if self._column_has_space(col)]

    def is_terminal(self) -> bool:
        """Return True if the game is over (win or draw).

        Returns:
            True if the previous move produced four in a row, or if the
            board is completely full (draw).
        """
        return self._last_player_won() or self._is_draw()

    def get_winner(self) -> int | None:
        """Return the winning player number, or None if no winner.

        Returns:
            1 or 2 if that player has four in a row, None for draw or
            ongoing game.
        """
        if not self._last_player_won():
            return None
        # The player who just moved (previous player) is the winner.
        # After a move, num_moves is already incremented:
        #   odd num_moves  → P1 just moved → P1 wins
        #   even num_moves → P2 just moved → P2 wins
        return PLAYER_1 if self._num_moves % 2 == 1 else PLAYER_2

    def get_result(self, player: int) -> float:
        """Return the game outcome from the perspective of the given player.

        Args:
            player: Player number (1 or 2) to evaluate for.

        Returns:
            +1.0 for a win, -1.0 for a loss, 0.0 for draw or ongoing.
        """
        winner = self.get_winner()
        if winner is None:
            return 0.0
        return 1.0 if winner == player else -1.0

    def encode(self) -> np.ndarray:
        """Encode the board as a (3, 6, 7) float32 array for NN input.

        Plane 0: current player's pieces.
        Plane 1: opponent's pieces.
        Plane 2: turn indicator — all 1s if current player is P1, else 0s.

        The encoding is always from the current player's perspective.

        Returns:
            Float32 array of shape (3, ROWS, COLS).
        """
        state = np.zeros((3, ROWS, COLS), dtype=np.float32)
        opponent_position = self._mask ^ self._position

        for col in range(COLS):
            base = col * BITS_PER_COL
            for row in range(ROWS):
                bit = 1 << (base + row)
                if self._position & bit:
                    state[0, row, col] = 1.0
                if opponent_position & bit:
                    state[1, row, col] = 1.0

        state[2] = 1.0 if self.current_player == PLAYER_1 else 0.0
        return state

    def encode_flipped(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a horizontally-flipped encoding for data augmentation.

        Connect 4 is left-right symmetric, so mirroring the board is a
        free augmentation that doubles the training dataset.

        Returns:
            A tuple of:
            - flipped_state: (3, 6, 7) float32 array mirrored left–right.
            - flip_indices: (7,) int64 array mapping each column index to
              its flipped counterpart: [6, 5, 4, 3, 2, 1, 0].
        """
        state = self.encode()
        flipped_state = np.flip(state, axis=2).copy()
        flip_indices = np.arange(COLS - 1, -1, -1, dtype=np.int64)
        return flipped_state, flip_indices

    def clone(self) -> "Connect4Board":
        """Return an independent deep copy of this board.

        Returns:
            A new Connect4Board with identical state.
        """
        return Connect4Board._from_state(
            mask=self._mask,
            position=self._position,
            num_moves=self._num_moves,
        )

    def __str__(self) -> str:
        """Pretty-print the board.

        Uses '.' for empty cells, 'X' for Player 1, 'O' for Player 2.
        The top row is printed first; column indices appear at the bottom.
        """
        p1_bits, p2_bits = self._get_player_bits()
        rows_out: list[str] = []
        for row in range(ROWS - 1, -1, -1):
            row_chars: list[str] = []
            for col in range(COLS):
                bit = 1 << (col * BITS_PER_COL + row)
                if p1_bits & bit:
                    row_chars.append("X")
                elif p2_bits & bit:
                    row_chars.append("O")
                else:
                    row_chars.append(".")
            rows_out.append(" ".join(row_chars))
        rows_out.append(" ".join(str(c) for c in range(COLS)))
        return "\n".join(rows_out)

    def __eq__(self, other: object) -> bool:
        """Compare boards by their full state."""
        if not isinstance(other, Connect4Board):
            return NotImplemented
        return (
            self._mask == other._mask
            and self._position == other._position
            and self._num_moves == other._num_moves
        )

    def __hash__(self) -> int:
        """Hash based on the full board state."""
        return hash((self._mask, self._position, self._num_moves))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_player_bits(self) -> tuple[int, int]:
        """Return (p1_bits, p2_bits) reconstructed from bitboard state.

        Returns:
            Tuple of (player1_bitboard, player2_bitboard).
        """
        if self.current_player == PLAYER_1:
            return self._position, self._mask ^ self._position
        return self._mask ^ self._position, self._position

    @staticmethod
    def _bottom_mask_col(col: int) -> int:
        """Return the bitmask for the bottom cell of the given column."""
        return 1 << (col * BITS_PER_COL)

    @staticmethod
    def _col_mask(col: int) -> int:
        """Return the bitmask for all valid (non-sentinel) rows in a column."""
        return COLUMN_MASK << (col * BITS_PER_COL)

    def _column_has_space(self, col: int) -> bool:
        """Return True if the column has at least one empty row.

        A column is full when its top row (ROWS-1) is already occupied.
        """
        top_row_bit = 1 << (col * BITS_PER_COL + ROWS - 1)
        return (self._mask & top_row_bit) == 0

    def _last_player_won(self) -> bool:
        """Return True if the player who just moved has four in a row."""
        if self._num_moves < WIN_LENGTH * 2 - 1:
            return False
        # After a move, _position holds the *new* current player's pieces,
        # which are the previous mover's *opponent*.
        # mask ^ position gives the previous mover's pieces.
        return _has_four_in_a_row(self._mask ^ self._position)

    def _is_draw(self) -> bool:
        """Return True if the board is full with no winner."""
        return self._num_moves == ROWS * COLS


def _has_four_in_a_row(bits: int) -> bool:
    """Check if a bitboard contains four consecutive set bits in any direction.

    Uses the AND-shift trick: if (bits & (bits >> k)) & (... >> 2k) is
    non-zero, there are at least four consecutive bits in that direction.

    Args:
        bits: Bitboard for a single player.

    Returns:
        True if any four-in-a-row exists, False otherwise.
    """
    # Horizontal (columns are BITS_PER_COL=7 bits apart)
    m = bits & (bits >> BITS_PER_COL)
    if m & (m >> (2 * BITS_PER_COL)):
        return True

    # Diagonal \\ (right one col + up one row = +7+1 = +8)
    m = bits & (bits >> (BITS_PER_COL + 1))
    if m & (m >> (2 * (BITS_PER_COL + 1))):
        return True

    # Diagonal / (right one col + down one row = +7-1 = +6)
    m = bits & (bits >> (BITS_PER_COL - 1))
    if m & (m >> (2 * (BITS_PER_COL - 1))):
        return True

    # Vertical (rows are 1 bit apart)
    m = bits & (bits >> 1)
    if m & (m >> 2):
        return True

    return False
