"""Exact late-game win/draw/loss values for legal actions."""

from functools import lru_cache

from src.game.board import Connect4Board


def replay_board(moves):
    board = Connect4Board()
    for move in moves:
        if board.is_terminal():
            raise ValueError("Moves extend beyond a terminal state")
        board = board.make_move(move)
    return board


@lru_cache(maxsize=100_000)
def exact_value(board):
    """Return the undiscounted minimax value to the player about to move."""
    if board.is_terminal():
        return int(board.get_result(board.current_player))
    best = -1
    for col in board.get_legal_moves():
        best = max(best, -exact_value(board.make_move(col)))
        if best == 1:
            break
    return best


def action_values(board):
    if board.is_terminal():
        raise ValueError("A terminal state has no action values")
    return {col: -exact_value(board.make_move(col)) for col in board.get_legal_moves()}
