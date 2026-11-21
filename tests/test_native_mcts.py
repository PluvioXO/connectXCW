"""The native backend must preserve the Python search's visit distributions."""
import shutil

import numpy as np
import pytest
import torch

from src.game.board import Connect4Board
from src.mcts.build_native import main as build_native
from src.mcts.search import BatchedMCTS
from src.utils.config import MCTSConfig


@pytest.fixture(scope='module', autouse=True)
def compiled_kernel():
    if not shutil.which('g++'):
        pytest.skip('Optional native search requires a C++ compiler')
    build_native()


def positions():
    boards = [Connect4Board()]
    for sequence in ([0, 0, 0, 0, 0, 0], [0, 6, 1, 6, 2, 6], [6, 0, 6, 1, 5, 2]):
        board = Connect4Board()
        for action in sequence:
            board = board.make_move(action)
        boards.append(board)
    rng = np.random.default_rng(7021)
    for _ in range(24):
        board = Connect4Board()
        for _ in range(int(rng.integers(1, 39))):
            safe = [board.make_move(c) for c in board.get_legal_moves()]
            safe = [b for b in safe if not b.is_terminal()]
            if not safe:
                break
            board = safe[int(rng.integers(len(safe)))]
        boards.append(board)
    return boards


@pytest.mark.parametrize('simulations', [0, 1, 48, 128])
@pytest.mark.parametrize('noise', [False, True])
def test_native_matches_python_exactly(tiny_model, monkeypatch, simulations, noise):
    torch.set_num_threads(1)
    boards = positions()
    results, statistics = [], []
    for backend in ['0', '1']:
        monkeypatch.setenv('CONNECTX_NATIVE_MCTS', backend)
        np.random.seed(701)
        search = BatchedMCTS(tiny_model, MCTSConfig(num_simulations=simulations))
        results.append(search.search_batch(boards, add_dirichlet_noise=noise))
        statistics.append(search.stats)
    np.testing.assert_array_equal(results[0], results[1])
    for key in ('searched_positions', 'simulations', 'network_batches', 'network_positions'):
        assert statistics[0][key] == statistics[1][key]


def test_native_rejects_invalid_roots_and_handles_empty(tiny_model, monkeypatch):
    monkeypatch.setenv('CONNECTX_NATIVE_MCTS', '1')
    search = BatchedMCTS(tiny_model, MCTSConfig(num_simulations=8))
    assert search.search_batch([]) == []
    board = Connect4Board()
    for col in [0, 1, 0, 1, 0, 1, 0]:
        board = board.make_move(col)
    with pytest.raises(ValueError, match='Terminal'):
        search.search_batch([board])
    board = Connect4Board._from_state(1 << 6, 0, 1)
    with pytest.raises(ValueError, match='Invalid bitboard'):
        search.search_batch([board])
    with pytest.raises(ValueError, match='gravity'):
        search.search_batch([Connect4Board._from_state(1 << 5, 0, 1)])


def test_native_rejects_nonfinite_network_output(tiny_model, monkeypatch):
    monkeypatch.setenv('CONNECTX_NATIVE_MCTS', '1')
    with torch.no_grad():
        next(tiny_model.parameters()).fill_(float('nan'))
    search = BatchedMCTS(tiny_model, MCTSConfig(num_simulations=8))
    with pytest.raises(RuntimeError, match='Non-finite|Invalid network'):
        search.search_batch([Connect4Board()])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph validation requires a GPU')
def test_cuda_graphs_refresh_inputs_and_masks(tiny_model):
    from src.mcts.cuda_evaluator import CudaEvaluator
    import torch.nn.functional as F
    model = tiny_model.cuda().eval()
    evaluator = CudaEvaluator(model, torch.device('cuda'))
    rng = np.random.default_rng(177)
    for count in [1, 7, 32, 64, 7, 1]:
        states = rng.integers(0, 2, (count, 3, 6, 7)).astype(np.float32)
        legal = rng.integers(0, 2, (count, 7)).astype(bool)
        legal[:, 3] = True
        with torch.no_grad():
            logits, values = model(torch.from_numpy(states).cuda())
            policies = F.softmax(logits.masked_fill(~torch.from_numpy(legal).cuda(), float('-inf')), dim=-1)
        actual_policy, actual_value = evaluator(states, legal)
        np.testing.assert_array_equal(actual_policy, policies.cpu().numpy())
        np.testing.assert_array_equal(actual_value, values.cpu().numpy().reshape(-1))
