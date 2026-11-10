"""Tests for the Connect 4 bitboard game engine (src/game/board.py)."""

import numpy as np
import pytest

from src.game.board import Connect4Board, _has_four_in_a_row
from src.game.constants import COLS, PLAYER_1, PLAYER_2, ROWS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def play(moves: list[int]) -> Connect4Board:
    """Build a board from a list of column moves."""
    board = Connect4Board()
    for col in moves:
        board = board.make_move(col)
    return board


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_current_player_is_player_one(self, empty_board):
        assert empty_board.current_player == PLAYER_1

    def test_all_columns_legal_on_empty_board(self, empty_board):
        assert empty_board.get_legal_moves() == list(range(COLS))

    def test_empty_board_not_terminal(self, empty_board):
        assert not empty_board.is_terminal()

    def test_no_winner_on_empty_board(self, empty_board):
        assert empty_board.get_winner() is None

    def test_empty_board_str_has_correct_shape(self, empty_board):
        lines = str(empty_board).split("\n")
        # ROWS board rows + 1 index line
        assert len(lines) == ROWS + 1

    def test_empty_board_str_all_dots(self, empty_board):
        lines = str(empty_board).split("\n")
        for line in lines[:-1]:  # skip index line
            assert set(line.split()) == {"."}


# ---------------------------------------------------------------------------
# Player alternation
# ---------------------------------------------------------------------------

class TestPlayerAlternation:
    def test_player_alternates_after_each_move(self, empty_board):
        board = empty_board
        for expected in [PLAYER_2, PLAYER_1, PLAYER_2, PLAYER_1]:
            board = board.make_move(0)
            assert board.current_player == expected

    def test_immutability_original_unchanged(self, empty_board):
        original = empty_board
        _ = original.make_move(3)
        assert original.current_player == PLAYER_1
        assert original.get_legal_moves() == list(range(COLS))


# ---------------------------------------------------------------------------
# Legal moves
# ---------------------------------------------------------------------------

class TestLegalMoves:
    def test_full_column_not_legal(self):
        # Fill column 0 (6 rows) by alternating P1 on col 0 and P2 on col 1
        board = play([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        assert 0 not in board.get_legal_moves()

    def test_legal_moves_count_after_full_column(self):
        # Both players alternate filling col 0 (P1 rows 0,2,4; P2 rows 1,3,5).
        # No 4-in-a-row occurs; col 0 is full after 6 moves → 6 columns remain.
        board = play([0, 0, 0, 0, 0, 0])
        assert len(board.get_legal_moves()) == COLS - 1

    def test_make_move_invalid_column_raises(self, empty_board):
        with pytest.raises(ValueError, match="out of range"):
            empty_board.make_move(-1)
        with pytest.raises(ValueError, match="out of range"):
            empty_board.make_move(COLS)

    def test_make_move_full_column_raises(self):
        board = play([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        with pytest.raises(ValueError, match="full"):
            board.make_move(0)


# ---------------------------------------------------------------------------
# Win detection — parametrized over all four directions
# ---------------------------------------------------------------------------

class TestWinDetection:
    @pytest.mark.parametrize("win_col_start", [0, 1, 2, 3])
    def test_horizontal_win_p1(self, win_col_start):
        """P1 wins with 4 horizontal pieces starting at win_col_start."""
        # Choose a filler column that does not overlap with the win columns.
        filler = 0 if win_col_start + 3 >= COLS - 1 else COLS - 1
        moves: list[int] = []
        for i in range(4):
            moves.append(win_col_start + i)  # P1
            if i < 3:
                moves.append(filler)         # P2
        board = play(moves)
        assert board.is_terminal()
        assert board.get_winner() == PLAYER_1

    def test_vertical_win_p1(self):
        """P1 wins with 4 vertical pieces in column 3."""
        # P1: col 3, P2: col 4 — four times
        moves = [3, 4, 3, 4, 3, 4, 3]
        board = play(moves)
        assert board.is_terminal()
        assert board.get_winner() == PLAYER_1

    def test_vertical_win_p2(self):
        """P2 wins with 4 vertical pieces in column 2."""
        # P1: col 0, P2: col 2 — P2 fills column 2 first
        moves = [0, 2, 0, 2, 0, 2, 1, 2]
        board = play(moves)
        assert board.is_terminal()
        assert board.get_winner() == PLAYER_2

    def test_diagonal_forward_slash_win_p1(self):
        """P1 wins on a / diagonal."""
        # Build a / diagonal starting at (row=0,col=0) going up-right
        # P1 at (0,0),(1,1),(2,2),(3,3)
        # Setup: stack pieces so correct rows are occupied
        # Col 0: P1 at row 0 → play col 0 first
        # Col 1: P1 at row 1 → need one filler first
        # Col 2: P1 at row 2 → need two fillers first
        # Col 3: P1 at row 3 → need three fillers first
        moves = [
            0,           # P1@(0,0)
            1,           # P2@(0,1) filler
            1,           # P1@(1,1)
            2,           # P2@(0,2) filler
            3,           # P1@(0,3) filler for col2 height
            2,           # P2@(1,2) filler
            2,           # P1@(2,2)
            3,           # P2@(1,3) filler
            3,           # P1@(2,3) filler
            6,           # P2 filler (safe column)
            3,           # P1@(3,3)
        ]
        board = play(moves)
        assert board.is_terminal()
        assert board.get_winner() == PLAYER_1

    def test_diagonal_backslash_win_p1(self):
        """P1 wins on a \\ diagonal: (col=3,row=0),(col=2,row=1),(col=1,row=2),(col=0,row=3).

        Construction: P2 fills fillers so each col has the right height before P1's
        diagonal piece.  Col 2 needs 1 piece below P1's, col 1 needs 2, col 0 needs 3.
        """
        moves = [
            3,   # P1@(col=3, row=0) ← diagonal piece
            6,   # P2 filler
            2,   # P1@(col=2, row=0) filler (raise col 2 height)
            1,   # P2@(col=1, row=0) filler (raise col 1 height)
            1,   # P1@(col=1, row=1) filler (raise col 1 height)
            0,   # P2@(col=0, row=0) filler (raise col 0 height)
            2,   # P1@(col=2, row=1) ← diagonal piece
            0,   # P2@(col=0, row=1) filler (raise col 0 height)
            1,   # P1@(col=1, row=2) ← diagonal piece
            0,   # P2@(col=0, row=2) filler (raise col 0 height)
            0,   # P1@(col=0, row=3) ← diagonal piece → win!
        ]
        board = play(moves)
        assert board.is_terminal()
        assert board.get_winner() == PLAYER_1

    def test_no_false_win_detection(self):
        """Three in a row does NOT trigger a win."""
        board = play([0, 6, 1, 6, 2])  # P1 has cols 0,1,2 — only 3
        assert not board.is_terminal()
        assert board.get_winner() is None

    def test_win_stops_game(self):
        """After a win, is_terminal returns True."""
        board = play([0, 6, 1, 6, 2, 6, 3])
        assert board.is_terminal()

    def test_get_result_winner_plus_one(self):
        board = play([0, 6, 1, 6, 2, 6, 3])
        assert board.get_result(PLAYER_1) == 1.0
        assert board.get_result(PLAYER_2) == -1.0

    def test_get_result_ongoing_is_zero(self, empty_board):
        assert empty_board.get_result(PLAYER_1) == 0.0
        assert empty_board.get_result(PLAYER_2) == 0.0


# ---------------------------------------------------------------------------
# Draw detection
# ---------------------------------------------------------------------------

class TestDrawDetection:
    def test_draw_condition_requires_full_board(self):
        """_is_draw is True only when all 42 cells are filled."""
        assert ROWS * COLS == 42  # sanity check on board dimensions
        # A board with 41 moves is not a draw (one cell remains)
        board = play([0, 0, 0, 0, 0, 0])  # 6 moves, col 0 full, 36 remain
        assert not board._is_draw()

    def test_num_moves_counts_correctly(self):
        board = play([0, 1, 2, 3])
        # 4 moves played
        assert board._num_moves == 4


# ---------------------------------------------------------------------------
# Clone and equality
# ---------------------------------------------------------------------------

class TestCloneAndEquality:
    def test_clone_equals_original(self, empty_board):
        clone = empty_board.clone()
        assert clone == empty_board

    def test_clone_is_independent(self, empty_board):
        clone = empty_board.clone()
        _ = clone.make_move(0)
        # Original unchanged
        assert empty_board.get_legal_moves() == list(range(COLS))

    def test_different_states_not_equal(self, empty_board, board_with_moves):
        assert empty_board != board_with_moves([0])

    def test_hash_equal_for_same_board(self, empty_board):
        clone = empty_board.clone()
        assert hash(clone) == hash(empty_board)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

class TestEncoding:
    def test_encode_empty_board_all_zeros_except_plane2(self, empty_board):
        enc = empty_board.encode()
        assert enc.shape == (3, ROWS, COLS)
        assert np.all(enc[0] == 0)
        assert np.all(enc[1] == 0)
        # P1 moves first → plane 2 all ones
        assert np.all(enc[2] == 1.0)

    def test_encode_plane2_zero_for_p2_turn(self):
        board = play([0])  # P2's turn
        enc = board.encode()
        assert np.all(enc[2] == 0.0)

    def test_encode_current_player_piece_in_plane0(self):
        # P1 plays col 3 → after move it's P2's turn
        # encode() is from P2's (current player's) perspective
        board = play([3])
        enc = board.encode()
        # Plane 0 = current (P2) pieces = empty
        assert np.all(enc[0] == 0)
        # Plane 1 = opponent (P1) pieces → col 3 row 0 has a piece
        assert enc[1, 0, 3] == 1.0
        assert enc[1].sum() == 1.0

    def test_encode_dtype_is_float32(self, empty_board):
        assert empty_board.encode().dtype == np.float32

    def test_encode_flipped_mirrors_board(self):
        board = play([0])  # piece in col 0 (leftmost)
        flipped, flip_idx = board.encode_flipped()
        # Piece should now appear in col 6 (rightmost) after flip
        assert flipped[1, 0, 6] == 1.0  # opponent's piece, flipped
        assert np.array_equal(flip_idx, np.array([6, 5, 4, 3, 2, 1, 0]))

    def test_encode_flipped_shape(self, empty_board):
        flipped, flip_idx = empty_board.encode_flipped()
        assert flipped.shape == (3, ROWS, COLS)
        assert flip_idx.shape == (COLS,)

    def test_encode_matches_known_position(self):
        # P1 plays col 3 (bottom of col 3), then P2 plays col 3 (row 1 of col 3)
        board = play([3, 3])
        enc = board.encode()
        # Now it's P1's turn again
        # Plane 0 = P1's pieces: col3 row0 = 1
        assert enc[0, 0, 3] == 1.0
        # Plane 1 = P2's pieces: col3 row1 = 1
        assert enc[1, 1, 3] == 1.0
        # No other pieces
        assert enc[0].sum() == 1.0
        assert enc[1].sum() == 1.0


# ---------------------------------------------------------------------------
# String representation
# ---------------------------------------------------------------------------

class TestStringRepresentation:
    def test_str_shows_p1_piece(self):
        board = play([3])  # P1 → col 3 bottom row
        s = str(board)
        # Bottom row is the second-to-last line (last line is column indices)
        rows = s.split("\n")
        bottom_row = rows[-2]
        chars = bottom_row.split()
        # P1 pieces are shown as 'X'
        assert chars[3] == "X"

    def test_str_index_line(self, empty_board):
        rows = str(empty_board).split("\n")
        index_line = rows[-1]
        assert index_line == "0 1 2 3 4 5 6"

    def test_str_p1_is_x_p2_is_o(self):
        # P1 plays col 0, P2 plays col 0 (above P1)
        board = play([0, 0])
        s = str(board)
        lines = s.split("\n")
        # Bottom row (index -2): P1 at col 0 → 'X'
        assert lines[-2].split()[0] == "X"
        # Second from bottom row (index -3): P2 at col 0 → 'O'
        assert lines[-3].split()[0] == "O"


# ---------------------------------------------------------------------------
# _has_four_in_a_row helper
# ---------------------------------------------------------------------------

class TestHasFourInARow:
    def test_horizontal_four(self):
        # Four bits in columns 0–3 at row 0
        from src.game.constants import BITS_PER_COL
        bits = sum(1 << (col * BITS_PER_COL) for col in range(4))
        assert _has_four_in_a_row(bits)

    def test_vertical_four(self):
        # Four bits in rows 0–3 at column 0
        bits = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
        assert _has_four_in_a_row(bits)

    def test_three_not_four(self):
        # Only three horizontal bits
        from src.game.constants import BITS_PER_COL
        bits = sum(1 << (col * BITS_PER_COL) for col in range(3))
        assert not _has_four_in_a_row(bits)

    def test_empty_not_four(self):
        assert not _has_four_in_a_row(0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @pytest.mark.parametrize("col", range(COLS))
    def test_win_in_each_column_vertical(self, col):
        """P1 can win vertically in any column."""
        filler = (col + 1) % COLS  # a different column for P2
        moves = [col, filler, col, filler, col, filler, col]
        board = play(moves)
        assert board.get_winner() == PLAYER_1

    @pytest.mark.parametrize("start_col", [0, 1, 2, 3])
    def test_horizontal_win_bottom_row(self, start_col):
        """P1 wins horizontally in the bottom row from start_col."""
        filler = 0 if start_col + 3 >= COLS - 1 else COLS - 1
        moves: list[int] = []
        for i in range(4):
            moves.append(start_col + i)
            if i < 3:
                moves.append(filler)
        board = play(moves)
        assert board.get_winner() == PLAYER_1
