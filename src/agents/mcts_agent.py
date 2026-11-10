"""Pure MCTS agent using random rollouts (no neural network).

Uses UCT (Upper Confidence bound for Trees) for node selection and
random playouts for evaluation. Standalone implementation with its own
lightweight node structure.
"""

from __future__ import annotations

import math
import random

import numpy as np

from src.agents.base_agent import Agent
from src.game.board import Connect4Board
from src.game.constants import PLAYER_1, PLAYER_2

# UCT exploration constant (sqrt(2) is theoretically optimal)
_UCT_C: float = math.sqrt(2)


class _PureMCTSNode:
    """Lightweight MCTS node for random-rollout (no NN) MCTS.

    Attributes:
        board: Game state at this node.
        parent: Parent node, or None for the root.
        action: Column played to reach this node.
        visit_count: Number of times this node was visited.
        value_sum: Accumulated value from this node's perspective.
        children: Map from action (int) to child node.
        untried_actions: Legal moves not yet expanded.
    """

    def __init__(
        self,
        board: Connect4Board,
        parent: _PureMCTSNode | None = None,
        action: int | None = None,
    ) -> None:
        self.board = board
        self.parent = parent
        self.action = action
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: dict[int, _PureMCTSNode] = {}
        self.untried_actions: list[int] = (
            [] if board.is_terminal() else board.get_legal_moves()
        )

    @property
    def q_value(self) -> float:
        """Return mean value. Returns 0.0 if unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_fully_expanded(self) -> bool:
        """Return True when all legal moves have been expanded."""
        return len(self.untried_actions) == 0

    def uct_score(self, parent_visits: int) -> float:
        """Compute the UCT score for this node (from parent's perspective).

        Args:
            parent_visits: Visit count of the parent node.

        Returns:
            UCT score (higher = more promising to explore).
        """
        if self.visit_count == 0:
            return float("inf")
        # Negate: child stores value from child's player's perspective
        exploitation = -self.q_value
        exploration = _UCT_C * math.sqrt(math.log(parent_visits) / self.visit_count)
        return exploitation + exploration


class PureMCTS:
    """Monte Carlo Tree Search with random rollouts.

    Args:
        num_simulations: Number of simulations per move.
        seed: Optional random seed for reproducibility.
    """

    def __init__(self, num_simulations: int = 200, seed: int | None = None) -> None:
        """Initialize PureMCTS.

        Args:
            num_simulations: Number of simulations to run per search call.
            seed: Optional seed for the random number generator.
        """
        self.num_simulations = num_simulations
        self._rng = random.Random(seed)

    def search(self, board: Connect4Board) -> int:
        """Run MCTS simulations and return the best move column.

        Args:
            board: Current game state.

        Returns:
            The column index with the most visits after all simulations.
        """
        root = _PureMCTSNode(board)

        for _ in range(self.num_simulations):
            node = self._select(root)
            if not node.board.is_terminal():
                node = self._expand(node)
            value = self._rollout(node)
            self._backup(node, value)

        # Choose child with most visits
        best_action = max(root.children, key=lambda a: root.children[a].visit_count)
        return best_action

    def _select(self, node: _PureMCTSNode) -> _PureMCTSNode:
        """Select a leaf node using UCT.

        Args:
            node: Root of the subtree to select from.

        Returns:
            Selected leaf or unexpanded node.
        """
        current = node
        while current.is_fully_expanded and not current.board.is_terminal():
            best_score = -float("inf")
            best_child = None
            for child in current.children.values():
                score = child.uct_score(current.visit_count)
                if score > best_score:
                    best_score = score
                    best_child = child
            current = best_child  # type: ignore[assignment]
        return current

    def _expand(self, node: _PureMCTSNode) -> _PureMCTSNode:
        """Expand one untried action from the node.

        Args:
            node: Node with untried actions available.

        Returns:
            Newly created child node.
        """
        action = self._rng.choice(node.untried_actions)
        node.untried_actions.remove(action)
        child_board = node.board.make_move(action)
        child = _PureMCTSNode(child_board, parent=node, action=action)
        node.children[action] = child
        return child

    def _rollout(self, node: _PureMCTSNode) -> float:
        """Perform a random playout from the node's board state.

        Args:
            node: Starting node for the rollout.

        Returns:
            Game result from the node's current player's perspective:
            +1.0 win, -1.0 loss, 0.0 draw.
        """
        board = node.board
        rollout_player = board.current_player

        while not board.is_terminal():
            legal = board.get_legal_moves()
            col = self._rng.choice(legal)
            board = board.make_move(col)

        return board.get_result(rollout_player)

    def _backup(self, node: _PureMCTSNode, value: float) -> None:
        """Propagate the rollout value back up the tree, flipping sign.

        Args:
            node: Leaf node where backup starts.
            value: Game result from the leaf node's player's perspective.
        """
        current: _PureMCTSNode | None = node
        v = value
        while current is not None:
            current.visit_count += 1
            current.value_sum += v
            v = -v
            current = current.parent


class MCTSAgent(Agent):
    """Agent using pure MCTS with random rollouts (no neural network).

    Args:
        num_simulations: Number of MCTS simulations per move (default: 200).
        seed: Optional random seed for reproducibility.
    """

    def __init__(self, num_simulations: int = 200, seed: int | None = None) -> None:
        """Initialize the pure MCTS agent.

        Args:
            num_simulations: Simulations to run per move.
            seed: Optional random seed.
        """
        self._mcts = PureMCTS(num_simulations=num_simulations, seed=seed)
        self._num_simulations = num_simulations

    @property
    def name(self) -> str:
        """Return the agent's display name."""
        return f"PureMCTS-{self._num_simulations}"

    def select_move(self, board: Connect4Board) -> int:
        """Select a move using pure MCTS with random rollouts.

        Args:
            board: Current game state.

        Returns:
            Best column index found by MCTS.
        """
        return self._mcts.search(board)
