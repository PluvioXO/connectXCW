"""ctypes bridge for the native tree kernel; PyTorch inference remains unchanged."""
import ctypes as C
from functools import lru_cache
from pathlib import Path
import os
import time

import numpy as np
import torch
import torch.nn.functional as F


@lru_cache(maxsize=1)
def library():
    path = Path(__file__).with_name('_native_search.so')
    if not path.exists():
        raise RuntimeError('Build native MCTS first: python -m src.mcts.build_native')
    lib = C.CDLL(str(path))
    lib.cx_version.restype = C.c_int
    assert lib.cx_version() == 1, 'Native MCTS ABI mismatch'
    lib.cx_error.restype = C.c_char_p
    lib.cx_create.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_double]
    lib.cx_create.restype = C.c_void_p
    lib.cx_destroy.argtypes = [C.c_void_p]
    lib.cx_destroy.restype = None
    lib.cx_prepare.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
    lib.cx_prepare.restype = C.c_int
    lib.cx_finish.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_double, C.c_int]
    lib.cx_finish.restype = C.c_int
    lib.cx_visits.argtypes = [C.c_void_p, C.c_void_p]
    lib.cx_visits.restype = None
    return lib


def search_batch(search, boards, add_noise):
    started = time.perf_counter()
    lib = library()
    size = len(boards)
    if search._device.type == 'cuda' and os.environ.get('CONNECTX_CUDA_GRAPHS') == '1':
        if not hasattr(search, '_cuda_evaluator'):
            from src.mcts.cuda_evaluator import CudaEvaluator
            search._cuda_evaluator = CudaEvaluator(search.model, search._device)
        evaluator = search._cuda_evaluator
    else:
        evaluator = None
    masks = np.array([b._mask for b in boards], dtype=np.uint64)
    positions = np.array([b._position for b in boards], dtype=np.uint64)
    moves = np.array([b._num_moves for b in boards], dtype=np.int32)
    handle = lib.cx_create(masks.ctypes.data, positions.ctypes.data, moves.ctypes.data,
                           size, search.config.num_simulations, search.config.c_puct)
    if not handle:
        raise ValueError(lib.cx_error().decode())
    try:
        states = np.empty((size, 3, 6, 7), dtype=np.float32)
        legal = np.empty((size, 7), dtype=np.bool_)
        root_legal = None
        for step in range(search.config.num_simulations + 1):
            count = lib.cx_prepare(handle, states.ctypes.data, legal.ctypes.data)
            noise = None
            policies = values = None
            if count:
                if evaluator is not None:
                    policies, values = evaluator(states[:count], legal[:count])
                else:
                    with torch.no_grad():
                        logits, tensor_values = search.model(torch.from_numpy(states[:count]).to(search._device))
                        mask = torch.from_numpy(legal[:count]).to(search._device)
                        policies = F.softmax(logits.masked_fill(~mask, float('-inf')), dim=-1).cpu().numpy()
                        values = tensor_values.cpu().numpy().reshape(-1)
                search.stats['network_batches'] += 1
                search.stats['network_positions'] += count
                if step == 0:
                    root_legal = legal.copy()
                    if add_noise:
                        noise = np.zeros((size, 7), dtype=np.float64)
                        for i in range(size):
                            columns = np.flatnonzero(legal[i])
                            noise[i, columns] = np.random.dirichlet([search.config.dirichlet_alpha] * len(columns))
            code = lib.cx_finish(handle, None if policies is None else policies.ctypes.data,
                                  None if values is None else values.ctypes.data,
                                  None if noise is None else noise.ctypes.data,
                                  search.config.dirichlet_epsilon, int(step > 0))
            if code:
                raise RuntimeError(lib.cx_error().decode())
        visits = np.empty((size, 7), dtype=np.float32)
        lib.cx_visits(handle, visits.ctypes.data)
        empty = visits.sum(1) == 0
        visits[empty] = root_legal[empty]
        visits /= visits.sum(1, keepdims=True)
        search.stats['searched_positions'] += size
        search.stats['simulations'] += size * search.config.num_simulations
        search.stats['search_seconds'] += time.perf_counter() - started
        return [row.copy() for row in visits]
    finally:
        lib.cx_destroy(handle)
