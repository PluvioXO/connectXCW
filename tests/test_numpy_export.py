"""Check the actual NumPy submission against the training game and network."""

from types import SimpleNamespace
import io

import numpy as np
import pytest
import torch

from src.export import kaggle_agent_numpy as agent
from src.game.board import Connect4Board
from scripts.kaggle_submit import _extract_weights_npz


def kaggle_board(board):
    encoded = board.encode()
    grid = encoded[0] * board.current_player + encoded[1] * (3 - board.current_player)
    return grid[::-1].reshape(-1).astype(int).tolist()


def test_board_encoding_and_wins_match_training_engine():
    rng = np.random.default_rng(42)
    for _ in range(100):
        board = Connect4Board()
        while not board.is_terminal():
            board = board.make_move(int(rng.choice(board.get_legal_moves())))
            flat = kaggle_board(board)
            np.testing.assert_array_equal(agent._encode_board(flat, board.current_player)[0],
                                          board.encode())
            for mark in (1, 2):
                assert agent._check_win(flat, mark) == (board.get_winner() == mark)


@pytest.mark.parametrize("moves,expected", [([0, 6, 1, 6, 2, 5], 3),
                                           ([6, 0, 6, 1, 5, 2], 3)])
def test_tactical_move_does_not_need_model(monkeypatch, moves, expected):
    board = Connect4Board()
    for col in moves:
        board = board.make_move(col)
    def fail(*args):
        raise AssertionError("Forced move should not run the network")
    monkeypatch.setattr(agent, "_mcts_search", fail)
    observation = SimpleNamespace(board=kaggle_board(board), mark=board.current_player)
    assert agent.my_agent(observation, None) == expected


def test_numpy_network_matches_pytorch(tiny_model, monkeypatch):
    weights = {k: v.detach().numpy().copy() for k, v in tiny_model.state_dict().items()
               if "num_batches_tracked" not in k}
    weights["num_blocks"] = np.array(2)
    agent._fold_bn(weights)
    monkeypatch.setattr(agent, "_weights", weights)
    board = Connect4Board()
    for col in [3, 2, 3, 4, 1, 4]:
        board = board.make_move(col)
    state = agent._encode_board(kaggle_board(board), board.current_player)
    logits, value = agent._predict(state)
    with torch.no_grad():
        expected_logits, expected_value = tiny_model(torch.from_numpy(state))
    np.testing.assert_allclose(logits, expected_logits[0].numpy(), atol=2e-5, rtol=2e-5)
    assert value == pytest.approx(expected_value.item(), abs=2e-5)


def test_initial_network_call_is_inside_search_budget(monkeypatch):
    clock = [0.0]
    calls = []
    def predict(state):
        calls.append(1)
        clock[0] += 0.1
        return np.array([0, 1, 2, 5, 2, 1, 0], dtype=np.float32), 0.0
    monkeypatch.setattr(agent.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(agent, "_predict", predict)
    move, _, sims = agent._mcts_search([0] * 42, 1, 0.1)
    assert len(calls) == 1 and sims == 0
    assert move == 3


def test_search_avoids_giving_opponent_a_winning_square(monkeypatch):
    board = [0, 0, 0, 0, 0, 0, 0,
             2, 0, 0, 0, 0, 0, 0,
             2, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0,
             2, 0, 2, 0, 1, 1, 2,
             1, 1, 2, 2, 2, 1, 1]
    logits = np.array([0, 100, 0, 0, 0, 0, 0], dtype=np.float32)
    monkeypatch.setattr(agent, "_predict", lambda _: (logits, 0.0))
    move, _, _ = agent._mcts_search(board, 1, 0)
    assert move in (0, 2, 3, 4, 5, 6)


def test_packaging_removes_subnormal_weights(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"model_state_dict": {"weight": torch.tensor([1e-40, -1e-40, 0.25])},
                "num_blocks": 2, "num_filters": 32}, path)
    with np.load(io.BytesIO(_extract_weights_npz(path))) as weights:
        np.testing.assert_array_equal(weights["weight"], [0, 0, 0.25])


def test_search_uses_exact_value_for_immediate_wins(monkeypatch):
    board = Connect4Board()
    for col in [0, 6, 1, 6, 2, 5]:
        board = board.make_move(col)
    def fail(*args):
        raise AssertionError("An exact win needs no network estimate")
    monkeypatch.setattr(agent, "_predict", fail)
    node = agent._MCTSNode(kaggle_board(board), board.current_player)
    assert agent._expand(node) == 1.0
    assert list(node.children) == [3]
    assert node.children[3].terminal
