"""Self-contained Kaggle Connect-X agent — pure numpy, zero external deps.

Only stdlib + numpy. No onnxruntime, no torch, no scipy.

Weights are loaded from a .npz file bundled in the submission archive.
The path is set by kaggle_submit.py --tar at packaging time.

Kaggle extracts the tar.gz to /kaggle_simulations/agent/ — main.py and
weights.npz both land there, so the default path works as-is.

Board encoding note (CRITICAL):
  Kaggle: board_flat[kaggle_row * 7 + col], row 0 = TOP
  Network: state[plane, network_row, col], row 0 = BOTTOM (gravity)
  Conversion: network_row = (ROWS - 1) - kaggle_row
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

# -------------------------------------------------------------------------
# Sentinels — replaced by kaggle_submit.py --tar at packaging time
# -------------------------------------------------------------------------
_CWD = '/kaggle_simulations/agent/'
if os.path.exists(_CWD):
    sys.path.append(_CWD)
else:
    _CWD = ''

_WEIGHTS_PATH: str = _CWD + "weights.npz"
_TIME_BUDGET_SECS: float = 2.0  # sentinel

# -------------------------------------------------------------------------
# Section 1: Board logic
# -------------------------------------------------------------------------

_ROWS: int = 6
_COLS: int = 7
_WIN: int = 4


def _get_legal_moves(board_flat: list[int]) -> list[int]:
    return [c for c in range(_COLS) if board_flat[c] == 0]


def _make_move(board_flat: list[int], col: int, mark: int) -> list[int]:
    board = list(board_flat)
    for row in range(_ROWS - 1, -1, -1):
        if board[row * _COLS + col] == 0:
            board[row * _COLS + col] = mark
            return board
    raise ValueError(f"Column {col} is full")


def _check_win(board_flat: list[int], mark: int) -> bool:
    bits = sum(1 << ((i % _COLS) * 7 + _ROWS - 1 - i // _COLS)
               for i, cell in enumerate(board_flat) if cell == mark)
    for shift in (1, 7, 6, 8):
        pairs = bits & (bits >> shift)
        if pairs & (pairs >> (2 * shift)):
            return True
    return False


def _is_terminal(board_flat: list[int]) -> bool:
    return (
        _check_win(board_flat, 1)
        or _check_win(board_flat, 2)
        or len(_get_legal_moves(board_flat)) == 0
    )


def _get_result(board_flat: list[int], mark: int) -> float:
    if _check_win(board_flat, mark):
        return 1.0
    if _check_win(board_flat, 3 - mark):
        return -1.0
    return 0.0


def _encode_board(board_flat: list[int], current_mark: int) -> np.ndarray:
    """Encode board as (1, 3, 6, 7) float32. Row flip: Kaggle 0=top, network 0=bottom."""
    state = np.zeros((1, 3, _ROWS, _COLS), dtype=np.float32)
    for kaggle_row in range(_ROWS):
        network_row = (_ROWS - 1) - kaggle_row
        for col in range(_COLS):
            cell = board_flat[kaggle_row * _COLS + col]
            if cell == current_mark:
                state[0, 0, network_row, col] = 1.0
            elif cell == (3 - current_mark):
                state[0, 1, network_row, col] = 1.0
    state[0, 2, :, :] = 1.0 if current_mark == 1 else 0.0
    return state


# -------------------------------------------------------------------------
# Section 2: Numpy ResNet forward pass
# -------------------------------------------------------------------------

_weights: dict | None = None


def _load_weights() -> dict:
    global _weights
    if _weights is None:
        npz = np.load(_WEIGHTS_PATH)
        _weights = {k: npz[k] for k in npz.files}
        _fold_bn(_weights)
    return _weights


def _fold_bn(w: dict) -> None:
    """Pre-compute BN scale/shift at load time so inference uses 2 ops instead of 5.

    For each BN layer: scale = gamma / sqrt(var + eps), shift = beta - mean * scale.
    Stored as new keys "<layer>.scale" and "<layer>.shift".
    """
    num_blocks = int(w["num_blocks"])
    bn_keys = (
        ["stem.1"]
        + [f"tower.{i}.bn1" for i in range(num_blocks)]
        + [f"tower.{i}.bn2" for i in range(num_blocks)]
        + ["policy_head.1", "value_head.1"]
    )
    for key in bn_keys:
        scale = (w[f"{key}.weight"] / np.sqrt(w[f"{key}.running_var"] + 1e-5)).astype(np.float32)
        shift = (w[f"{key}.bias"] - w[f"{key}.running_mean"] * scale).astype(np.float32)
        w[f"{key}.scale"] = scale
        w[f"{key}.shift"] = shift


def _bn_fast(x: np.ndarray, scale: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Apply pre-folded BN: y = x * scale + shift."""
    return x * scale[None, :, None, None] + shift[None, :, None, None]


def _conv2d(x: np.ndarray, w: np.ndarray, pad: int = 1) -> np.ndarray:
    """2D convolution via im2col + matmul. No bias (BN follows).

    Args:
        x: (1, C, H, W) float32
        w: (F, C, kH, kW) float32
        pad: zero-padding on each spatial side

    Returns:
        (1, F, H_out, W_out) float32
    """
    if pad > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    _, C, H, W = x.shape
    F, _, kH, kW = w.shape
    H_out = H - kH + 1
    W_out = W - kW + 1

    # Vectorised im2col via stride tricks — no Python loop over spatial positions.
    shape   = (H_out, W_out, C, kH, kW)
    strides = (x.strides[2], x.strides[3], x.strides[1], x.strides[2], x.strides[3])
    col = np.lib.stride_tricks.as_strided(x[0], shape=shape, strides=strides)
    col = col.reshape(H_out * W_out, C * kH * kW)

    out = col @ w.reshape(F, -1).T
    return out.T.reshape(1, F, H_out, W_out)


def _bn(x, gamma, beta, mean, var, eps=1e-5):
    """Batch norm (eval mode) for (1, C, H, W) tensors."""
    return (
        gamma[None, :, None, None]
        * (x - mean[None, :, None, None])
        / np.sqrt(var[None, :, None, None] + eps)
        + beta[None, :, None, None]
    )


def _predict(state: np.ndarray) -> tuple[np.ndarray, float]:
    """Run ResNet forward pass on (1, 3, 6, 7) board encoding.

    Returns:
        (policy_logits, value): (7,) float32 logits and scalar in [-1, 1]
    """
    w = _load_weights()
    num_blocks = int(w["num_blocks"])

    # Stem
    x = _conv2d(state, w["stem.0.weight"], pad=1)
    x = np.maximum(_bn_fast(x, w["stem.1.scale"], w["stem.1.shift"]), 0)

    # Residual tower
    for i in range(num_blocks):
        residual = x
        out = _conv2d(x, w[f"tower.{i}.conv1.weight"], pad=1)
        out = np.maximum(_bn_fast(out, w[f"tower.{i}.bn1.scale"], w[f"tower.{i}.bn1.shift"]), 0)
        out = _conv2d(out, w[f"tower.{i}.conv2.weight"], pad=1)
        out = _bn_fast(out, w[f"tower.{i}.bn2.scale"], w[f"tower.{i}.bn2.shift"])
        x = np.maximum(out + residual, 0)

    # Policy head: 1×1 conv → BN → ReLU → flatten → linear
    p = _conv2d(x, w["policy_head.0.weight"], pad=0)
    p = np.maximum(_bn_fast(p, w["policy_head.1.scale"], w["policy_head.1.shift"]), 0).reshape(1, -1)
    policy_logits = (p @ w["policy_head.4.weight"].T + w["policy_head.4.bias"])[0]

    # Value head: 1×1 conv → BN → ReLU → flatten → linear → ReLU → linear → tanh
    v = _conv2d(x, w["value_head.0.weight"], pad=0)
    v = np.maximum(_bn_fast(v, w["value_head.1.scale"], w["value_head.1.shift"]), 0).reshape(1, -1)
    v = np.maximum(v @ w["value_head.4.weight"].T + w["value_head.4.bias"], 0)
    v = v @ w["value_head.6.weight"].T + w["value_head.6.bias"]
    value = float(np.tanh(v[0, 0]))

    return policy_logits, value


# -------------------------------------------------------------------------
# Section 3: Inline MCTS
# -------------------------------------------------------------------------

_C_PUCT: float = 2.0


class _MCTSNode:
    __slots__ = ["board_flat", "mark", "parent", "action", "prior",
                 "visit_count", "value_sum", "children", "terminal"]

    def __init__(self, board_flat, mark, parent=None, action=None, prior=0.0):
        self.board_flat = board_flat
        self.mark = mark
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: dict = {}
        self.terminal: bool = _is_terminal(board_flat)

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _puct_score(parent: _MCTSNode, child: _MCTSNode) -> float:
    exploration = (
        _C_PUCT * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    )
    return -child.q_value + exploration


def _expand(node: _MCTSNode) -> float:
    if node.terminal:
        return _get_result(node.board_flat, node.mark)
    if node.children:
        return 0.0
    legal = _get_legal_moves(node.board_flat)
    winning = [col for col in legal
               if _check_win(_make_move(node.board_flat, col, node.mark), node.mark)]
    if winning:
        for col in winning:
            node.children[col] = _MCTSNode(
                _make_move(node.board_flat, col, node.mark), 3 - node.mark,
                parent=node, action=col, prior=1.0 / len(winning),
            )
        return 1.0
    state = _encode_board(node.board_flat, node.mark)
    policy_logits, value = _predict(state)
    mask = np.full(_COLS, -1e9, dtype=np.float32)
    for c in legal:
        mask[c] = policy_logits[c]
    mask -= mask.max()
    exp = np.exp(mask)
    probs = exp / exp.sum()
    next_mark = 3 - node.mark
    for col in legal:
        node.children[col] = _MCTSNode(
            board_flat=_make_move(node.board_flat, col, node.mark),
            mark=next_mark, parent=node, action=col, prior=float(probs[col]),
        )
    return value


def _backup(node: _MCTSNode, value: float) -> None:
    current = node
    v = value
    while current is not None:
        current.visit_count += 1
        current.value_sum += v
        v = -v
        current = current.parent


def _mcts_search(board_flat: list[int], mark: int, time_budget_secs: float) -> tuple[int, float, int]:
    deadline = time.perf_counter() + time_budget_secs
    root = _MCTSNode(board_flat=list(board_flat), mark=mark)
    started = time.perf_counter()
    _backup(root, _expand(root))
    safe = {col: child for col, child in root.children.items()
            if not any(_check_win(_make_move(child.board_flat, reply, child.mark), child.mark)
                       for reply in _get_legal_moves(child.board_flat))}
    if safe:
        root.children = safe
    longest_sim = time.perf_counter() - started
    sims = 0
    while time.perf_counter() + longest_sim * 1.5 < deadline:
        started = time.perf_counter()
        node = root
        while node.children and not node.terminal:
            best_score = -float("inf")
            best_child = None
            for child in node.children.values():
                score = _puct_score(node, child)
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child  # type: ignore[assignment]
        _backup(node, _expand(node))
        sims += 1
        longest_sim = max(longest_sim, time.perf_counter() - started)
    best_move = max(root.children, key=lambda c: (root.children[c].visit_count,
                                                root.children[c].prior))
    return best_move, root.q_value, sims


# -------------------------------------------------------------------------
# Section 4: Agent entry point
# -------------------------------------------------------------------------

def my_agent(observation, configuration) -> int:
    """Kaggle Connect-X agent — pure numpy ResNet + MCTS, no onnxruntime."""
    started = time.perf_counter()
    board, mark = observation.board, observation.mark
    legal = _get_legal_moves(board)
    for player in (mark, 3 - mark):
        for col in legal:
            if _check_win(_make_move(board, col, player), player):
                return col
    if len(legal) == 1:
        return legal[0]
    budget = _TIME_BUDGET_SECS - 0.2 - (time.perf_counter() - started)
    best_move, value, sims = _mcts_search(
        board, mark, max(0, budget)
    )
    print(f"[agent] sims={sims}  value={value:+.3f}  move={best_move}")
    return best_move
