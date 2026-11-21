import numpy as np
import torch

from scripts.double_dqn import paired_openings
from src.game.board import Connect4Board
from src.training.double_dqn import Learner, advance, greedy_move, targets


def test_bootstrap_masks_illegal_columns_and_switches_perspective():
    def online(x):
        return x.new_tensor([0.9, 0.4, 0.2, -0.4]).repeat(len(x), 1)

    def frozen(x):
        return x.new_tensor([0.9, 0.3, 0.8, -0.1]).repeat(len(x), 1)

    legal = torch.tensor([[False, True, True, True], [False] * 4, [False] * 4])
    rewards = torch.tensor([0., 1., 0.])
    states = torch.zeros(3, 1)
    torch.testing.assert_close(targets(online, frozen, states, rewards, legal),
                               torch.tensor([-0.3, 1., 0.]))
    torch.testing.assert_close(targets(online, frozen, states, rewards, legal, double=False),
                               torch.tensor([-0.8, 1., 0.]))


def test_terminal_win_uses_previous_mover_and_disables_bootstrap():
    board = Connect4Board()
    for column in (0, 1, 0, 1, 0, 1):
        board = board.make_move(column)
    successor, transition = advance(board, 0)
    assert successor.is_terminal() and successor.get_winner() == 1
    assert transition.reward == 1 and not transition.next_legal.any()
    assert successor.get_result(successor.current_player) == -1


def test_full_column_is_unavailable_and_target_network_is_delayed():
    board = Connect4Board()
    for _ in range(6):
        board = board.make_move(0)
    assert greedy_move(lambda x: x.new_tensor([[10., 1., 0., 0., 0., 0., 0.]]),
                       board, "cpu") == 1

    _, transition = advance(board, 1)
    np.testing.assert_array_equal(transition.next_legal, [False] + [True] * 6)
    learner = Learner("cpu", 7)
    learner.batch_size = learner.warmup = learner.update_every = 1
    learner.sync_every = 2
    initial = [p.detach().clone() for p in learner.target.parameters()]
    assert np.isfinite(learner.observe(transition))
    assert all(torch.equal(old, p) for old, p in zip(initial, learner.target.parameters()))
    assert any(not torch.equal(old, p) for old, p in zip(initial, learner.online.parameters()))
    assert np.isfinite(learner.observe(transition))
    assert all(torch.equal(p, q) for p, q in zip(learner.online.parameters(),
                                               learner.target.parameters()))


def test_paired_openings_are_repeatable_distinct_and_nonterminal():
    openings = list(paired_openings(24, 42))
    assert openings == list(paired_openings(24, 42))
    assert len({board for board, _ in openings}) == 24
    assert all(not board.is_terminal() and len(moves) % 2 == 0 for board, moves in openings)
    previous = {board for board, _ in openings}
    held_out = list(paired_openings(24, 42, previous))
    assert len(held_out) == 24 and previous.isdisjoint(board for board, _ in held_out)
