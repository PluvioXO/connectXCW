import numpy as np
import torch

from scripts.learned_lookahead import two_ply_move
from src.game.board import Connect4Board
from src.training.double_dqn import advance, targets
from src.training.tactical_dqn import FixedBudgetLearner, certified_loss, tactical_values


def position(*moves):
    board = Connect4Board()
    for move in moves:
        board = board.make_move(move)
    return board


def test_certificates_only_claim_proven_outcomes():
    values = tactical_values(position(6, 0, 6, 0, 5, 0))
    assert np.isnan(values[0])
    assert np.all(values[1:] == -1)
    assert tactical_values(position(0, 1, 0, 1, 0, 1))[0] == 1
    assert np.all(np.isnan(tactical_values(Connect4Board())))


def test_only_certified_actions_receive_gradient():
    predictions = torch.tensor([[0., 0., 0.]], requires_grad=True)
    labels = torch.tensor([[-1., float("nan"), 1.]])
    loss = certified_loss(predictions, labels)
    assert loss == 0.5
    loss.backward()
    torch.testing.assert_close(predictions.grad, torch.tensor([[.5, 0., -.5]]))
    assert certified_loss(predictions, torch.full_like(labels, float("nan"))) == 0.


def test_multistep_targets_alternate_bootstrap_sign_and_skip_terminal():
    states = torch.zeros(4, 1)
    legal = torch.tensor([[True, True], [True, True], [True, True], [False, False]])
    online = lambda batch: torch.tensor([[1., 0.]]).expand(len(batch), -1)
    frozen = lambda batch: torch.tensor([[0.5, 0.8]]).expand(len(batch), -1)
    values = targets(online, frozen, states, torch.tensor([0., 0., 0., -1.]), legal,
                     horizons=torch.tensor([1, 2, 3, 2]))
    torch.testing.assert_close(values, torch.tensor([-.5, .5, -.5, -1.]))


def test_multistep_replay_cuts_before_exploration_and_after_terminal():
    learner = FixedBudgetLearner("cpu", 1, tactical_weight=0, n_step=3)
    board = Connect4Board()
    moves = (0, 1, 0, 1, 0, 1, 0)
    for step, col in enumerate(moves):
        labels = np.full(7, step)
        board, transition = advance(board, col)
        learner.observe(transition, labels, exploratory=(step == 2))
    assert learner.steps == learner.stored == len(moves) and not learner.pending
    assert [entry.horizon for entry in learner.replay] == [2, 1, 3, 3, 3, 2, 1]
    assert [entry.reward for entry in learner.replay] == [0, 0, 0, 0, 1, -1, 1]
    assert [entry.next_legal.any() for entry in learner.replay] == [True] * 4 + [False] * 3
    np.testing.assert_array_equal(learner.certificates[:7, 0], np.arange(7))


def test_two_ply_search_handles_forced_defence_with_either_leaf_evaluator():
    board = position(6, 0, 6, 0, 5, 0)
    assert two_ply_move(lambda states: torch.zeros(len(states), 7), board) == 0
    assert two_ply_move(None, board, leaf_evaluator=lambda leaves: [0.] * len(leaves)) == 0
