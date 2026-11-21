"""Double DQN targets and replay for alternating, zero-sum Connect Four."""

from copy import deepcopy
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.game.board import Connect4Board
from src.game.constants import COLS, ROWS


def make_network():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        nn.Flatten(), nn.Linear(32 * ROWS * COLS, 128), nn.ReLU(),
        nn.Linear(128, COLS), nn.Tanh(),
    )


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    successor: np.ndarray
    next_legal: np.ndarray
    horizon: int = 1


def advance(board: Connect4Board, action: int):
    successor = board.make_move(action)
    legal = np.zeros(COLS, dtype=bool)
    if not successor.is_terminal():
        legal[successor.get_legal_moves()] = True
    transition = Transition(
        board.encode().astype(np.uint8), action,
        successor.get_result(board.current_player),
        successor.encode().astype(np.uint8), legal,
    )
    return successor, transition


@torch.no_grad()
def targets(online, target, successors, rewards, legal, *, double=True, horizons=None):
    values = rewards.clone()
    live = legal.any(dim=1)
    if live.any():
        next_states = successors[live]
        target_q = target(next_states)
        selector = online(next_states) if double else target_q
        best = selector.masked_fill(~legal[live], -torch.inf).argmax(dim=1, keepdim=True)
        continuation = target_q.gather(1, best).squeeze(1)
        # After an odd number of plies, the successor describes the opponent.
        sign = -1 if horizons is None else 1 - 2 * (horizons[live] % 2)
        values[live] += sign * continuation
    return values


@torch.no_grad()
def greedy_move(network, board, device):
    if board.is_terminal():
        raise ValueError("Cannot select a move on a terminal board")
    legal = board.get_legal_moves()
    state = torch.from_numpy(board.encode()).unsqueeze(0).to(device)
    return legal[int(network(state)[0, legal].argmax())]


class Learner:
    batch_size = 128
    warmup = 1024
    update_every = 4
    sync_every = 500
    replay_capacity = 50_000

    def __init__(self, device, seed, *, double=True):
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.device = device
        self.double = double
        self.online = make_network().to(device)
        self.target = deepcopy(self.online).requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=3e-4)
        self.replay: list[Transition] = []
        self.steps = 0
        self.updates = 0

    def action(self, board, epsilon):
        return self.behavior_action(board, epsilon)[0]

    def behavior_action(self, board, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.choice(board.get_legal_moves()), True
        return greedy_move(self.online, board, self.device), False

    def observe(self, transition):
        if len(self.replay) < self.replay_capacity:
            self.replay.append(transition)
        else:
            self.replay[self.steps % self.replay_capacity] = transition
        self.steps += 1
        if len(self.replay) < self.warmup or self.steps % self.update_every:
            return None

        batch = self.rng.sample(self.replay, self.batch_size)
        def tensor(field, dtype):
            return torch.as_tensor(np.stack([getattr(t, field) for t in batch]),
                                   dtype=dtype, device=self.device)

        states = tensor("state", torch.float32)
        actions = tensor("action", torch.long)
        expected = targets(
            self.online, self.target, tensor("successor", torch.float32),
            tensor("reward", torch.float32), tensor("next_legal", torch.bool),
            double=self.double,
        )
        chosen = self.online(states).gather(1, actions[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(chosen, expected)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())
