import numpy as np

from src.game.endgame import action_values, replay_board
from src.training.tactical_dqn import tactical_values


def test_exact_values_distinguish_legal_choices_beyond_short_tactics():
    prefix = [4, 6, 1, 6, 4, 2, 3, 2, 2, 4, 6, 2, 2, 2, 5,
              6, 3, 6, 0, 6, 0, 0, 5, 5, 5, 5, 4, 0, 5, 3]
    board = replay_board(prefix)
    values = action_values(board)
    assert np.isnan(tactical_values(board)).all()
    assert values == {0: -1, 1: -1, 3: 1, 4: -1}
    mirrored = action_values(replay_board([6 - c for c in prefix]))
    assert mirrored == {6 - c: value for c, value in values.items()}


def test_two_ply_lookahead_prefers_a_win_or_forced_block():
    import torch

    from scripts.learned_lookahead import two_ply_move

    class Constant:
        def __call__(self, batch):
            return torch.zeros(len(batch), 7)

    assert two_ply_move(Constant(), replay_board([0, 1, 0, 1, 0, 1])) == 0
    assert two_ply_move(Constant(), replay_board([6, 0, 6, 0, 5, 0])) == 0
