"""DQN agent: 11 -> 128 -> 128 -> 5 ReLU MLP with replay + target network.

Standard value-based DQN: epsilon-greedy behaviour, experience replay and a
periodically-synced target network, trained with SmoothL1 loss.
"""

from __future__ import annotations

import json
import os
import random
from collections import deque, namedtuple
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from . import config

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class QNetwork(nn.Module):
    def __init__(
        self,
        state_size: int = config.STATE_SIZE,
        action_size: int = config.ACTION_SIZE,
        hidden: int = config.HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        device: str = config.DEVICE,
        seed: Optional[int] = None,
    ) -> None:
        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        self.q_network = QNetwork().to(self.device)
        self.target_network = QNetwork().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(),
                                    lr=config.LEARNING_RATE)
        self.replay_buffer = deque(maxlen=config.REPLAY_BUFFER_SIZE)
        self.epsilon = config.EPSILON_START
        self.learn_steps = 0

    # ------------------------------------------------------------------ #
    def select_action(self, state: torch.Tensor, is_test: bool = False) -> torch.Tensor:
        state = state.to(self.device)
        if not is_test and random.random() < self.epsilon:
            return torch.tensor(
                [[random.randrange(config.ACTION_SIZE)]],
                dtype=torch.long, device=self.device,
            )
        with torch.no_grad():
            return self.q_network(state).max(1)[1].view(1, 1)

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.append(
            Transition(state, action, reward, next_state, done)
        )

    def learn(self) -> Optional[float]:
        if len(self.replay_buffer) < config.BATCH_SIZE:
            return None

        batch = Transition(*zip(
            *random.sample(self.replay_buffer, config.BATCH_SIZE)
        ))
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device).bool().view(-1)

        q_values = self.q_network(state_batch).gather(1, action_batch)

        next_q_values = torch.zeros(config.BATCH_SIZE, device=self.device)
        non_final = ~done_batch
        if non_final.any():
            next_q_values[non_final] = (
                self.target_network(next_state_batch[non_final])
                .max(1)[0]
                .detach()
            )

        target_q_values = reward_batch + config.GAMMA * next_q_values
        loss = nn.functional.smooth_l1_loss(
            q_values, target_q_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_steps += 1
        return float(loss.item())

    # ------------------------------------------------------------------ #
    def update_epsilon(self) -> None:
        if self.epsilon > config.EPSILON_END:
            self.epsilon = max(self.epsilon * config.EPSILON_DECAY,
                               config.EPSILON_END)

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.q_network.state_dict(), path + ".pt")
        torch.save(self.target_network.state_dict(), path + "_target.pt")

    def load(self, path: str) -> None:
        self.q_network.load_state_dict(
            torch.load(path + ".pt", map_location=self.device)
        )
        target_path = path + "_target.pt"
        if os.path.exists(target_path):
            self.target_network.load_state_dict(
                torch.load(target_path, map_location=self.device)
            )
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())


def save_checkpoint_meta(path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + "_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
