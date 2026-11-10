"""Monte Carlo Tree Search with neural network guidance (AlphaZero style).

The MCTS uses the neural network to evaluate leaf nodes and guide exploration
via the PUCT formula. After all simulations, it returns a visit-count
distribution used as a training policy target.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from src.game.board import Connect4Board
from src.game.constants import COLS
from src.neural_net.model import Connect4Net, get_policy
from src.utils.config import MCTSConfig


class MCTSNode:
    """A node in the MCTS search tree.

    Attributes:
        board: Connect4Board at this position.
        parent: Parent node (None for root).
        action: The column (0-6) that led to this node from its parent.
        prior: P(s,a) from the neural network policy.
        visit_count: N(s,a) — how many times this node was visited.
        value_sum: W(s,a) — total value accumulated through this node.
        children: Dict mapping action (int) → child MCTSNode.
    """

    def __init__(
        self,
        board: Connect4Board,
        parent: MCTSNode | None = None,
        action: int | None = None,
        prior: float = 0.0,
    ) -> None:
        """Initialize a node.

        Args:
            board: The game state at this node.
            parent: Parent node, or None if this is the root.
            action: The column played to reach this node.
            prior: Prior probability P(s,a) from the neural network.
        """
        self.board = board
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: dict[int, MCTSNode] = {}

    @property
    def is_leaf(self) -> bool:
        """Return True if this node has no children yet."""
        return len(self.children) == 0

    @property
    def is_terminal(self) -> bool:
        """Return True if the board is in a terminal state."""
        return self.board.is_terminal()

    @property
    def q_value(self) -> float:
        """Return Q(s,a) = W / N. Returns 0.0 if unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """Monte Carlo Tree Search guided by a neural network.

    Uses the PUCT formula to balance exploration and exploitation.
    The neural network provides policy priors and value estimates at leaf nodes,
    replacing traditional random rollouts.

    Args:
        model: Trained (or random) Connect4Net used for evaluation.
        config: MCTSConfig containing hyperparameters.
    """

    def __init__(self, model: Connect4Net, config: MCTSConfig) -> None:
        """Initialize MCTS.

        Args:
            model: Neural network for evaluating positions.
            config: MCTS hyperparameters.
        """
        self.model = model
        self.config = config
        self.model.eval()

    def search(self, board: Connect4Board, add_dirichlet_noise: bool = False) -> np.ndarray:
        """Run MCTS simulations from the given board position.

        Args:
            board: The root board position to search from.
            add_dirichlet_noise: If True, add Dirichlet noise to root priors
                for exploration (used during self-play training).

        Returns:
            A (7,) float32 array of visit-count probabilities, summing to 1.0.
            Illegal moves have probability 0.

        Raises:
            ValueError: If board is already in a terminal state.
        """
        if board.is_terminal():
            raise ValueError("Cannot search from a terminal board position.")

        root = MCTSNode(board)
        self._expand_and_evaluate(root)

        if add_dirichlet_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.config.num_simulations):
            node = self._select(root)
            value = self._expand_and_evaluate(node)
            self._backup(node, value)

        # Build visit count array over all 7 columns
        visit_array = np.zeros(COLS, dtype=np.float32)
        for action, child in root.children.items():
            visit_array[action] = child.visit_count

        total = visit_array.sum()
        if total == 0:
            # Fallback: uniform over legal moves (shouldn't happen in practice)
            legal = board.get_legal_moves()
            visit_array[legal] = 1.0
            total = len(legal)

        return visit_array / total

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse the tree using PUCT until reaching a leaf or terminal node.

        Args:
            node: The node to start selection from.

        Returns:
            The selected leaf or terminal node.
        """
        current = node
        while not current.is_leaf and not current.is_terminal:
            best_score = -float("inf")
            best_child = None
            for child in current.children.values():
                score = self._puct_score(current, child)
                if score > best_score:
                    best_score = score
                    best_child = child
            current = best_child  # type: ignore[assignment]
        return current

    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """Expand a leaf node and return its value estimate.

        For terminal nodes, returns the exact game result.
        For non-terminal leaves, runs the neural network and creates children.
        If the node already has children (i.e. it's the root being expanded
        for the first time via search()), returns the NN value without
        re-expanding.

        Args:
            node: The node to expand.

        Returns:
            Value estimate from the current player's perspective:
            +1.0 = current player wins, -1.0 = current player loses, 0.0 = draw.
        """
        if node.is_terminal:
            winner = node.board.get_winner()
            if winner is None:
                return 0.0
            # After a win, the current_player is the *loser* (the winner just moved)
            return node.board.get_result(node.board.current_player)

        # Already expanded (can happen on root)
        if not node.is_leaf:
            return 0.0

        board_tensor = self._encode_board(node.board)
        legal_mask = self._build_legal_moves_mask(node.board)

        with torch.no_grad():
            policy, value = get_policy(self.model, board_tensor, legal_mask)

        policy_np = np.array(policy.tolist(), dtype=np.float32)
        value_scalar = float(value.item())

        legal_moves = node.board.get_legal_moves()
        for action in legal_moves:
            child_board = node.board.make_move(action)
            prior = float(policy_np[action])
            node.children[action] = MCTSNode(
                board=child_board,
                parent=node,
                action=action,
                prior=prior,
            )

        return value_scalar

    def _backup(self, node: MCTSNode, value: float) -> None:
        """Propagate value back up the tree, flipping sign at each level.

        The sign is flipped because Q values are from the perspective of the
        node's current player, and each level switches perspective.

        Args:
            node: The leaf node where backup starts.
            value: The value to propagate (current player's perspective at leaf).
        """
        current: MCTSNode | None = node
        v = value
        while current is not None:
            current.visit_count += 1
            current.value_sum += v
            v = -v
            current = current.parent

    def _add_dirichlet_noise(self, node: MCTSNode) -> None:
        """Add Dirichlet noise to root node's children priors for exploration.

        Args:
            node: The root node whose children receive noise.
        """
        if not node.children:
            return

        alpha = self.config.dirichlet_alpha
        epsilon = self.config.dirichlet_epsilon
        num_children = len(node.children)

        noise = np.random.dirichlet([alpha] * num_children)
        for i, child in enumerate(node.children.values()):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

    def _puct_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        """Compute the PUCT score for selecting a child node.

        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))

        Args:
            parent: The parent node.
            child: The child node to score.

        Returns:
            PUCT score (higher = more promising to explore).
        """
        exploration = (
            self.config.c_puct
            * child.prior
            * math.sqrt(parent.visit_count)
            / (1 + child.visit_count)
        )
        # Negate child.q_value: child stores value from child's player's perspective,
        # but PUCT selects from the PARENT's player's perspective (opposite sign).
        return -child.q_value + exploration

    def _encode_board(self, board: Connect4Board) -> torch.Tensor:
        """Encode a board as a (1, 3, 6, 7) tensor for neural network input.

        Uses tolist() conversion to work around the missing NumPy C bridge
        in this PyTorch build.

        Args:
            board: The board to encode.

        Returns:
            Float32 tensor of shape (1, 3, 6, 7).
        """
        device = next(self.model.parameters()).device
        return torch.tensor(board.encode().tolist(), dtype=torch.float32).unsqueeze(0).to(device)

    def _build_legal_moves_mask(self, board: Connect4Board) -> np.ndarray:
        """Build a boolean mask of legal moves.

        Args:
            board: The current game state.

        Returns:
            Boolean ndarray of shape (7,), True for legal columns.
        """
        mask = np.zeros(COLS, dtype=bool)
        for col in board.get_legal_moves():
            mask[col] = True
        return mask


def select_move(visit_counts: np.ndarray, temperature: float) -> int:
    """Select a move from MCTS visit counts using temperature scaling.

    At temperature ≈ 0: deterministically selects the most-visited move.
    At temperature = 1: samples proportionally to visit counts.
    Higher temperature flattens the distribution (more exploration).

    Args:
        visit_counts: Array of visit counts for each column (length 7).
            Zero counts are valid (for illegal or unvisited moves).
        temperature: Controls randomness. Values near 0 are deterministic.

    Returns:
        Selected column index (0–6).

    Raises:
        ValueError: If all visit counts are zero.
    """
    if visit_counts.sum() == 0:
        raise ValueError("All visit counts are zero — no move available.")

    if temperature < 0.01:
        return int(np.argmax(visit_counts))

    adjusted = visit_counts.astype(np.float64) ** (1.0 / temperature)
    total = adjusted.sum()
    probs = adjusted / total
    return int(np.random.choice(len(probs), p=probs))


def _visit_distribution(root: MCTSNode, board: Connect4Board) -> np.ndarray:
    """Extract a normalised visit-count distribution from a searched root node.

    Args:
        root: The root node after MCTS search.
        board: The board at the root (used as fallback for legal moves).

    Returns:
        A (7,) float32 array of visit-count probabilities, summing to 1.0.
    """
    visit_array = np.zeros(COLS, dtype=np.float32)
    for action, child in root.children.items():
        visit_array[action] = child.visit_count

    total = visit_array.sum()
    if total == 0:
        legal = board.get_legal_moves()
        visit_array[legal] = 1.0
        total = len(legal)

    return visit_array / total


class BatchedMCTS(MCTS):
    """MCTS variant that processes multiple trees in lock-step for GPU efficiency.

    Instead of one forward pass per simulation, `search_batch` advances M MCTS
    trees together and collects all leaf evaluations into a single batched
    GPU forward pass per simulation step. This keeps GPU utilisation high
    (batch size M instead of 1) and is the recommended approach for training
    with large models on a GPU.

    This class is additive — all existing MCTS tests and the base `MCTS` class
    are unchanged.

    Args:
        model: Trained Connect4Net. Should be on the target GPU/CPU device.
        config: MCTS hyperparameters.
        batch_size: Number of MCTS trees to advance in parallel (M). Larger
            values improve GPU utilisation but use more memory.
    """

    def __init__(self, model: Connect4Net, config: MCTSConfig, batch_size: int = 32) -> None:
        """Initialise batched MCTS.

        Args:
            model: Neural network for evaluating positions.
            config: MCTS hyperparameters.
            batch_size: Number of parallel game trees.
        """
        super().__init__(model, config)
        self._batch_size = batch_size
        self._device = next(model.parameters()).device

    def search_batch(
        self, boards: list[Connect4Board], add_dirichlet_noise: bool = False
    ) -> list[np.ndarray]:
        """Run batched MCTS across multiple board positions simultaneously.

        All M trees advance one simulation step together. Leaf evaluations are
        batched into a single GPU forward pass per step.

        Args:
            boards: List of board positions to search from. Must all be
                non-terminal.
            add_dirichlet_noise: If True, add Dirichlet noise to root priors
                for training exploration.

        Returns:
            List of (7,) float32 visit-count probability arrays, one per board,
            in the same order as the input list.
        """
        if not boards:
            return []

        roots = [MCTSNode(board=b) for b in boards]

        # Initial expansion of all roots (one batched NN call)
        self._batch_expand_and_evaluate(roots, add_noise=add_dirichlet_noise)

        for _ in range(self.config.num_simulations):
            # Phase 1: CPU traversal — select a leaf from each tree
            leaves = [self._select(root) for root in roots]
            # Phase 2: single GPU call — expand all leaves together
            values = self._batch_expand_and_evaluate(leaves)
            # Phase 3: CPU backup
            for leaf in leaves:
                self._backup(leaf, values[id(leaf)])

        return [_visit_distribution(root, board) for root, board in zip(roots, boards)]

    def _batch_expand_and_evaluate(
        self, nodes: list[MCTSNode], add_noise: bool = False
    ) -> dict[int, float]:
        """Expand a batch of leaf nodes, evaluating them with one GPU call.

        Terminal nodes are evaluated with exact game results (no NN needed).
        Already-expanded nodes return 0.0 (safety guard).
        Remaining leaf nodes are evaluated in a single batched forward pass.

        Args:
            nodes: Nodes to expand/evaluate.
            add_noise: If True, add Dirichlet noise to the root prior of each
                newly expanded node (used during self-play training).

        Returns:
            Dict mapping ``id(node)`` → value (float) for every node in
            ``nodes``.  Values are from each node's current player's perspective.
        """
        result: dict[int, float] = {}
        to_eval: list[MCTSNode] = []
        legal_masks: list[np.ndarray] = []

        for node in nodes:
            if node.is_terminal:
                winner = node.board.get_winner()
                if winner is None:
                    result[id(node)] = 0.0
                else:
                    result[id(node)] = node.board.get_result(node.board.current_player)
            elif not node.is_leaf:
                # Already expanded (can happen on root re-entry)
                result[id(node)] = 0.0
            else:
                legal_mask = self._build_legal_moves_mask(node.board)
                to_eval.append(node)
                legal_masks.append(legal_mask)

        if to_eval:
            states = np.stack([n.board.encode() for n in to_eval])
            batch_tensor = torch.from_numpy(states).to(self._device)
            mask_tensor = torch.from_numpy(np.stack(legal_masks)).to(self._device)

            with torch.no_grad():
                logits, values = self.model(batch_tensor)
                policies = F.softmax(logits.masked_fill(~mask_tensor, float("-inf")), dim=-1).cpu().numpy()
                values = values.cpu().numpy().reshape(-1)

            for i, node in enumerate(to_eval):
                policy_np = policies[i]
                value_scalar = float(values[i])

                legal_moves = node.board.get_legal_moves()
                for action in legal_moves:
                    child_board = node.board.make_move(action)
                    prior = float(policy_np[action])
                    node.children[action] = MCTSNode(
                        board=child_board,
                        parent=node,
                        action=action,
                        prior=prior,
                    )

                if add_noise:
                    self._add_dirichlet_noise(node)

                result[id(node)] = value_scalar

        return result
