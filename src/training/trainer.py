"""Neural network trainer for AlphaZero.

Trains the Connect4Net on (state, policy, value) samples from the
replay buffer using combined MSE + soft cross-entropy loss.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.neural_net.model import Connect4Net
from src.training.replay_buffer import ReplayBuffer, TrainingSample
from src.utils.config import TrainingConfig


class Trainer:
    """Trains Connect4Net on replay buffer data.

    Uses Adam optimizer with the combined AlphaZero loss:
        L = MSE(value) + CrossEntropy(policy) + weight_decay * ||θ||²

    The policy loss uses soft cross-entropy (KL divergence form) because
    policy targets are probability distributions, not one-hot labels.

    Args:
        model: The Connect4Net to train (modified in-place).
        training_config: Hyperparameters (batch_size, lr, weight_decay, etc.).
    """

    def __init__(self, model: Connect4Net, training_config: TrainingConfig) -> None:
        """Initialize the trainer.

        Args:
            model: Neural network to train.
            training_config: Training hyperparameters.
        """
        self._model = model
        self._config = training_config
        self._optimizer = optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

    @property
    def optimizer(self) -> optim.Adam:
        """The Adam optimizer (exposed for checkpoint save/load)."""
        return self._optimizer

    def train_step(self, batch: list[TrainingSample]) -> dict[str, float]:
        """Perform a single gradient update on one mini-batch.

        Args:
            batch: List of TrainingSample objects forming the mini-batch.

        Returns:
            Dict with keys 'policy_loss', 'value_loss', 'total_loss'.
        """
        self._model.train()

        device = next(self._model.parameters()).device

        # Convert numpy arrays → tensors via .tolist() (numpy C bridge unavailable)
        states = torch.tensor(
            np.stack([s.state for s in batch]).tolist(), dtype=torch.float32
        ).to(device)
        target_policies = torch.tensor(
            np.stack([s.policy for s in batch]).tolist(), dtype=torch.float32
        ).to(device)
        target_values = torch.tensor(
            [s.value for s in batch], dtype=torch.float32
        ).unsqueeze(1).to(device)

        policy_logits, value = self._model(states)

        # Value loss: MSE between predicted scalar and game outcome
        value_loss = F.mse_loss(value, target_values)

        # Policy loss: soft cross-entropy (not F.cross_entropy — targets are distributions)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(target_policies * log_probs).sum(dim=-1).mean()

        total_loss = value_loss + policy_loss

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "total_loss": float(total_loss.item()),
        }

    def train_epoch(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
        """Run one training epoch over the replay buffer.

        Performs max(1, len(buffer) // batch_size) gradient steps,
        sampling a fresh mini-batch for each step.

        Args:
            replay_buffer: Buffer to sample mini-batches from.

        Returns:
            Dict with averaged 'policy_loss', 'value_loss', 'total_loss'.
        """
        batch_size = self._config.batch_size
        num_steps = max(1, len(replay_buffer) // batch_size)

        total_metrics: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "total_loss": 0.0,
        }

        for _ in range(num_steps):
            batch = replay_buffer.sample(batch_size)
            metrics = self.train_step(batch)
            for key in total_metrics:
                total_metrics[key] += metrics[key]

        return {key: val / num_steps for key, val in total_metrics.items()}

    def train(self, replay_buffer: ReplayBuffer, epochs: int) -> dict[str, float]:
        """Run multiple training epochs.

        Args:
            replay_buffer: Buffer to sample mini-batches from.
            epochs: Number of full passes over the buffer.

        Returns:
            Dict with averaged metrics across all epochs:
            'policy_loss', 'value_loss', 'total_loss'.
        """
        logger = logging.getLogger(__name__)
        total_metrics: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "total_loss": 0.0,
        }

        for epoch in range(epochs):
            epoch_metrics = self.train_epoch(replay_buffer)
            for key in total_metrics:
                total_metrics[key] += epoch_metrics[key]
            logger.info(
                "  Epoch %d/%d — policy: %.4f  value: %.4f  total: %.4f",
                epoch + 1, epochs,
                epoch_metrics["policy_loss"],
                epoch_metrics["value_loss"],
                epoch_metrics["total_loss"],
            )

        return {key: val / epochs for key, val in total_metrics.items()}
