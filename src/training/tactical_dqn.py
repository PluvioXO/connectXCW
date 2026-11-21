"""Double DQN with an optional loss on tactically certified action values."""

from dataclasses import replace

import numpy as np
import torch
from torch.nn import functional as F

from src.game.constants import COLS
from src.training.double_dqn import Learner, targets


def tactical_values(board):
    """Exact values when an action ends play or allows an immediate loss.

    NaN means unknown. An action that survives the next reply can still lose
    later. The labels use the same undiscounted, adversarial return as the TD
    target.
    """
    values = np.full(COLS, np.nan, dtype=np.float32)
    mover = board.current_player
    for col in board.get_legal_moves():
        child = board.make_move(col)
        if child.is_terminal():
            values[col] = child.get_result(mover)
        elif any(child.make_move(reply).get_winner() == child.current_player
                 for reply in child.get_legal_moves()):
            values[col] = -1
    return values


def certified_loss(predictions, certificates):
    """Average Huber loss over exact labels, with zero loss for empty masks."""
    mask = torch.isfinite(certificates)
    labels = torch.where(mask, certificates, predictions.detach())
    return F.smooth_l1_loss(predictions, labels, reduction="sum") / mask.sum().clamp(min=1)


class FixedBudgetLearner(Learner):
    """Replay one- or multi-ply returns on a fixed environment-step schedule."""

    def __init__(self, device, seed, *, tactical_weight, n_step=1):
        super().__init__(device, seed, double=True)
        self.tactical_weight = tactical_weight
        self.certificates = np.full((self.replay_capacity, COLS), np.nan, dtype=np.float32)
        self.n_step = n_step
        self.pending = []
        self.stored = 0
        self.sampled_horizons = np.zeros(n_step + 1, dtype=np.int64)

    def observe(self, transition, certificates, *, exploratory=False, last_step=False):
        ready = [(transition, certificates)]
        if self.n_step > 1:
            # Start a new return at an exploratory move, cutting older returns before it.
            ready = self.pending if exploratory else []
            self.pending = [] if exploratory else [
                (replace(prior,
                         reward=prior.reward + (-1)**prior.horizon * transition.reward,
                         successor=transition.successor, next_legal=transition.next_legal,
                         horizon=prior.horizon + 1), labels)
                for prior, labels in self.pending
            ]
            self.pending.append((transition, certificates))
            if last_step or not transition.next_legal.any():
                ready.extend(self.pending)
                self.pending.clear()
            elif self.pending[0][0].horizon == self.n_step:
                ready.append(self.pending.pop(0))

        for entry, labels in ready:
            index = self.stored % self.replay_capacity
            if self.stored < self.replay_capacity:
                self.replay.append(entry)
            else:
                self.replay[index] = entry
            self.certificates[index] = labels
            self.stored += 1
        self.steps += 1
        if (self.steps < self.warmup or self.steps % self.update_every
                or len(self.replay) < self.batch_size):
            return None

        indices = self.rng.sample(range(len(self.replay)), self.batch_size)
        batch = [self.replay[i] for i in indices]

        def tensor(field, dtype):
            return torch.as_tensor(np.stack([getattr(t, field) for t in batch]),
                                   dtype=dtype, device=self.device)

        horizons = None
        if self.n_step > 1:
            lengths = np.array([entry.horizon for entry in batch])
            self.sampled_horizons += np.bincount(lengths, minlength=self.n_step + 1)
            horizons = torch.as_tensor(lengths, device=self.device)
        else:
            self.sampled_horizons[1] += self.batch_size
        expected = targets(
            self.online, self.target, tensor("successor", torch.float32),
            tensor("reward", torch.float32), tensor("next_legal", torch.bool),
            horizons=horizons,
        )
        values = self.online(tensor("state", torch.float32))
        actions = tensor("action", torch.long)
        chosen = values.gather(1, actions[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(chosen, expected)
        if self.tactical_weight:
            known = torch.as_tensor(self.certificates[indices], device=self.device)
            loss = loss + self.tactical_weight * certified_loss(values, known)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())
