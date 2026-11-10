"""Replay buffer for AlphaZero self-play training data.

Stores (state, policy, value) training samples with FIFO eviction
when the buffer exceeds its maximum capacity.
"""

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingSample:
    """A single training example from self-play.

    Attributes:
        state: Board encoding of shape (3, 6, 7), float32.
        policy: MCTS visit-count distribution of shape (7,), sums to 1.0.
        value: Game outcome from the current player's perspective:
            +1.0 = win, -1.0 = loss, 0.0 = draw.
    """

    state: np.ndarray    # (3, 6, 7) float32
    policy: np.ndarray   # (7,) float32, sums to 1.0
    value: float         # +1.0, -1.0, or 0.0


class ReplayBuffer:
    """FIFO replay buffer for AlphaZero training samples.

    When the buffer is full, the oldest samples are evicted first.
    Mini-batches are sampled uniformly at random.

    Args:
        max_size: Maximum number of samples to store.
    """

    def __init__(self, max_size: int) -> None:
        """Initialize the replay buffer.

        Args:
            max_size: Maximum number of TrainingSample entries to hold.
        """
        self._max_size = max_size
        self._buffer: deque[TrainingSample] = deque()

    def add(self, sample: TrainingSample) -> None:
        """Add a sample to the buffer, evicting the oldest if full.

        Args:
            sample: TrainingSample to add.
        """
        if len(self._buffer) >= self._max_size:
            self._buffer.popleft()
        self._buffer.append(sample)

    def sample(self, batch_size: int) -> list[TrainingSample]:
        """Sample a mini-batch of training examples uniformly at random.

        If batch_size exceeds the buffer length, returns all samples
        (shuffled).

        Args:
            batch_size: Number of samples to draw.

        Returns:
            List of TrainingSample objects (may be shorter than batch_size
            if the buffer has fewer samples).
        """
        actual_size = min(batch_size, len(self._buffer))
        return random.sample(list(self._buffer), actual_size)

    def __len__(self) -> int:
        """Return the current number of samples in the buffer."""
        return len(self._buffer)
