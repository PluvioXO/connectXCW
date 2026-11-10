"""Abstract base class for Connect 4 agents."""

from abc import ABC, abstractmethod

from src.game.board import Connect4Board


class Agent(ABC):
    """Abstract base class for all Connect 4 agents.

    All agents must implement select_move to return a legal column index.
    """

    @property
    def name(self) -> str:
        """Return the agent's display name."""
        return self.__class__.__name__

    @abstractmethod
    def select_move(self, board: Connect4Board) -> int:
        """Return a legal column index (0-6) for the current board state.

        Args:
            board: The current game state. It is the current player's turn.

        Returns:
            A legal column index in range [0, 6].
        """
