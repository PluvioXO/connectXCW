"""Self-contained Kaggle Connect-X submission with inline MCTS.

Zero imports from src/. Only numpy, onnxruntime, math, stdlib.

This file is copied verbatim into Kaggle submissions by kaggle_submit.py,
which replaces the _MODEL_PATH and _NUM_MCTS_SIMS sentinel lines.

Board encoding note (CRITICAL):
  Kaggle: board_flat[kaggle_row * 7 + col], row 0 = TOP (physical top)
  Network: state[plane, network_row, col], row 0 = BOTTOM (gravity side)
  Conversion: network_row = (ROWS - 1) - kaggle_row

_make_move note:
  Gravity means pieces fall to the bottom. In Kaggle's row-major layout,
  the bottom row has the highest row index (ROWS - 1). Scan from ROWS-1
  downward to find the first empty cell.
"""

from __future__ import annotations

import math

import numpy as np
import onnxruntime as ort

# -------------------------------------------------------------------------
# Module-level constants — replaced by kaggle_submit.py at packaging time
# -------------------------------------------------------------------------
_MODEL_PATH: str = "model.onnx"
_NUM_MCTS_SIMS: int = 200

# -------------------------------------------------------------------------
# Section 1: Board logic
# -------------------------------------------------------------------------

_ROWS: int = 6
_COLS: int = 7
_WIN: int = 4


def _get_legal_moves(board_flat: list[int]) -> list[int]:
    """Return column indices that are not full.

    In Kaggle format, board_flat[col] is the top cell of column col (row 0).
    A column is full when its top cell is non-zero.
    """
    return [c for c in range(_COLS) if board_flat[c] == 0]


def _make_move(board_flat: list[int], col: int, mark: int) -> list[int]:
    """Place mark in column col using gravity, returning a new board.

    Gravity: pieces fall to the bottom. In Kaggle's row-major layout,
    the bottom row has the highest row index (ROWS - 1 = 5).
    Scans from the bottom upward to find the lowest empty cell.
    """
    board = list(board_flat)
    for row in range(_ROWS - 1, -1, -1):
        if board[row * _COLS + col] == 0:
            board[row * _COLS + col] = mark
            return board
    raise ValueError(f"Column {col} is full")


def _check_win(board_flat: list[int], mark: int) -> bool:
    """Return True if mark has four consecutive pieces in any direction."""
    board = [board_flat[r * _COLS : (r + 1) * _COLS] for r in range(_ROWS)]

    # Horizontal
    for r in range(_ROWS):
        for c in range(_COLS - _WIN + 1):
            if all(board[r][c + i] == mark for i in range(_WIN)):
                return True

    # Vertical
    for r in range(_ROWS - _WIN + 1):
        for c in range(_COLS):
            if all(board[r + i][c] == mark for i in range(_WIN)):
                return True

    # Diagonal \ (top-left to bottom-right: row index increases)
    for r in range(_ROWS - _WIN + 1):
        for c in range(_COLS - _WIN + 1):
            if all(board[r + i][c + i] == mark for i in range(_WIN)):
                return True

    # Diagonal / (bottom-left to top-right: row index decreases)
    for r in range(_WIN - 1, _ROWS):
        for c in range(_COLS - _WIN + 1):
            if all(board[r - i][c + i] == mark for i in range(_WIN)):
                return True

    return False


def _is_terminal(board_flat: list[int]) -> bool:
    """Return True if the game is over (win or full board)."""
    return (
        _check_win(board_flat, 1)
        or _check_win(board_flat, 2)
        or len(_get_legal_moves(board_flat)) == 0
    )


def _get_result(board_flat: list[int], mark: int) -> float:
    """Return +1.0 if mark won, -1.0 if opponent won, 0.0 for draw."""
    if _check_win(board_flat, mark):
        return 1.0
    if _check_win(board_flat, 3 - mark):
        return -1.0
    return 0.0


def _encode_board(board_flat: list[int], current_mark: int) -> np.ndarray:
    """Encode board as (1, 3, 6, 7) float32 array for ONNX inference.

    Plane 0: current player's pieces.
    Plane 1: opponent's pieces.
    Plane 2: turn indicator — all 1s if current_mark == 1, else 0s.

    Critical row flip:
        Kaggle row 0 = physical top (where pieces land last).
        Network row 0 = physical bottom (where pieces land first).
        network_row = (ROWS - 1) - kaggle_row
    """
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
# Section 2: Inline MCTS
# -------------------------------------------------------------------------

_C_PUCT: float = 2.0


class _MCTSNode:
    """A node in the MCTS search tree.

    value_sum is accumulated from this node's own mark's (current player's)
    perspective. PUCT must negate child.q_value when evaluating from the
    parent's perspective (zero-sum game).
    """

    __slots__ = [
        "board_flat",
        "mark",
        "parent",
        "action",
        "prior",
        "visit_count",
        "value_sum",
        "children",
    ]

    def __init__(
        self,
        board_flat: list[int],
        mark: int,
        parent: "_MCTSNode | None" = None,
        action: "int | None" = None,
        prior: float = 0.0,
    ) -> None:
        self.board_flat = board_flat
        self.mark = mark
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: dict[int, "_MCTSNode"] = {}

    @property
    def q_value(self) -> float:
        """W / N from this node's mark's perspective. 0.0 if unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _puct_score(parent: _MCTSNode, child: _MCTSNode) -> float:
    """PUCT score for selecting child, evaluated from parent's perspective.

    UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))

    Negates child.q_value because child stores value from child's
    perspective (opposite of parent's in a zero-sum game).
    """
    exploration = (
        _C_PUCT
        * child.prior
        * math.sqrt(parent.visit_count)
        / (1 + child.visit_count)
    )
    return -child.q_value + exploration


def _expand(node: _MCTSNode, session: ort.InferenceSession) -> float:
    """Expand a leaf node using the ONNX model.

    Returns the value estimate from node's mark's perspective:
    - Terminal node: exact game result.
    - Non-terminal leaf: neural network value estimate.
    - Already-expanded node (e.g. root called twice): returns 0.0.
    """
    if _is_terminal(node.board_flat):
        return _get_result(node.board_flat, node.mark)

    # Guard against double-expansion (root pre-expand then sim expand)
    if node.children:
        return 0.0

    state = _encode_board(node.board_flat, node.mark)
    # session.run() returns outputs in output_names order: ["policy_logits", "value"]
    policy_logits, value_arr = session.run(None, {"board_state": state})
    # policy_logits: (1, 7) float32, value_arr: (1, 1) float32

    legal = _get_legal_moves(node.board_flat)
    logits = policy_logits[0].copy()  # (7,) float32

    # Mask illegal moves to near-zero probability before softmax
    mask = np.full(_COLS, -1e9, dtype=np.float32)
    for c in legal:
        mask[c] = logits[c]

    # Numerically stable softmax
    mask -= mask.max()
    exp = np.exp(mask)
    probs = exp / exp.sum()

    next_mark = 3 - node.mark  # opponent moves from child positions
    for col in legal:
        child_board = _make_move(node.board_flat, col, node.mark)
        node.children[col] = _MCTSNode(
            board_flat=child_board,
            mark=next_mark,
            parent=node,
            action=col,
            prior=float(probs[col]),
        )

    return float(value_arr[0, 0])


def _backup(node: _MCTSNode, value: float) -> None:
    """Propagate value up the tree, flipping sign at each level.

    Each level switches player perspective, so the sign of value flips.
    node.value_sum always accumulates from that node's own mark's perspective.
    """
    current: _MCTSNode | None = node
    v = value
    while current is not None:
        current.visit_count += 1
        current.value_sum += v
        v = -v
        current = current.parent


def _mcts_search(
    session: ort.InferenceSession,
    board_flat: list[int],
    mark: int,
    num_simulations: int,
) -> int:
    """Run MCTS from the given position and return the best column.

    Args:
        session: ONNX Runtime inference session (pre-loaded).
        board_flat: Flat Kaggle board (42 ints, row 0 = top).
        mark: Current player's mark (1 or 2).
        num_simulations: Number of MCTS simulations to run.

    Returns:
        Best column index (0-6) based on visit counts.
    """
    root = _MCTSNode(board_flat=list(board_flat), mark=mark)
    _expand(root, session)

    for _ in range(num_simulations):
        # SELECT: traverse tree using PUCT until reaching a leaf or terminal
        node = root
        while node.children and not _is_terminal(node.board_flat):
            best_score = -float("inf")
            best_child = None
            for child in node.children.values():
                score = _puct_score(node, child)
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child  # type: ignore[assignment]

        # EXPAND & EVALUATE
        value = _expand(node, session)

        # BACKUP
        _backup(node, value)

    # Select move with highest visit count (deterministic)
    return max(root.children, key=lambda c: root.children[c].visit_count)


# -------------------------------------------------------------------------
# Section 3: Agent entry point
# -------------------------------------------------------------------------

_session: ort.InferenceSession | None = None


def my_agent(observation, configuration) -> int:
    """Kaggle Connect-X agent using ONNX-backed MCTS.

    First call loads the ONNX model (~10s on Kaggle). Subsequent calls
    reuse the cached session.

    Args:
        observation: Kaggle observation with .board (list of 42 ints,
            row 0 = top) and .mark (1 or 2 — this agent's player number).
        configuration: Kaggle config (unused; constants hardcoded above).

    Returns:
        Column index (0-6) to play.
    """
    global _session
    if _session is None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        _session = ort.InferenceSession(_MODEL_PATH, sess_options=opts)
    return _mcts_search(_session, observation.board, observation.mark, _NUM_MCTS_SIMS)
