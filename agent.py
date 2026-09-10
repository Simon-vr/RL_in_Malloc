from config import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class DQNAgent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.q_network = self._build_model().to(self.device)
        self.target_network = self._build_model().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.epsilon = EPSILON_START

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(STATE_SIZE, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, ACTION_SIZE)
        )

    def select_action(self, state, is_test=False):
        state = state.to(self.device)
        if is_test:
            with torch.no_grad():
                return self.q_network(state).max(1)[1].view(1, 1)
        if random.random() < self.epsilon:
            return torch.tensor([[random.randrange(ACTION_SIZE)]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            return self.q_network(state).max(1)[1].view(1, 1)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append(Transition(state, action, reward, next_state, done))

    def learn(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        transitions = random.sample(self.replay_buffer, BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)

        q_values = self.q_network(state_batch).gather(1, action_batch)
        next_q_values = torch.zeros(BATCH_SIZE, device=self.device)
        non_final_mask = ~done_batch
        next_q_values[non_final_mask] = self.target_network(next_state_batch[non_final_mask]).max(1)[0].detach()

        target_q_values = reward_batch + (GAMMA * next_q_values)
        loss = nn.functional.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)
        torch.save(self.target_network.state_dict(), filename + "_target")

    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
        self.target_network.load_state_dict(torch.load(filename + "_target"))
