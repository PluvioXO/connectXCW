from scripts.dqn_study import tactical_candidates
from src.game.board import Connect4Board


def position(*moves):
    board = Connect4Board()
    for move in moves:
        board = board.make_move(move)
    return board


def test_chooses_immediate_win_before_blocking():
    assert tactical_candidates(position(0, 1, 0, 1, 0, 1)) == [0]


def test_removes_moves_allowing_immediate_loss():
    assert tactical_candidates(position(6, 0, 6, 0, 5, 0)) == [0]
    assert tactical_candidates(Connect4Board()) == list(range(7))
