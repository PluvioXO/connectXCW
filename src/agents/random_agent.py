"""Random agent: selects uniformly from legal moves."""

import random

from src.agents.base_agent import Agent
from src.game.board import Connect4Board


class RandomAgent(Agent):
    """Agent that selects a uniformly random legal move.

    Args:
        seed: Optional random seed for reproducibility.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the random agent.

        Args:
            seed: Optional seed for the random number generator.
        """
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        """Return the agent's display name."""
        return "Random"

    def select_move(self, board: Connect4Board) -> int:
        """Select a uniformly random legal move.

        Args:
            board: Current game state.

        Returns:
            A randomly chosen legal column index.
        """
        return self._rng.choice(board.get_legal_moves())
