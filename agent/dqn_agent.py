# agent/dqn_agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=128, output_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, env, lr=1e-3, gamma=0.99, batch_size=64, buffer_size=20000, device=None):
        self.env = env
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay = ReplayBuffer(buffer_size)

        self.policy_net = QNetwork(input_dim=env.observation_space.shape[0]).to(self.device)
        self.target_net = QNetwork(input_dim=env.observation_space.shape[0]).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.steps = 0
        self.eps_start = 1.0
        self.eps_min = 0.05
        self.eps_decay_steps = 50000

    def select_action(self, state):
        eps = max(self.eps_min, self.eps_start - (self.steps / self.eps_decay_steps))
        self.steps += 1
        if random.random() < eps:
            return self.env.action_space.sample()
        state_v = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.policy_net(state_v)
            return int(torch.argmax(q).item())

    def push_transition(self, *args):
        self.replay.push(*args)

    def update(self):
        if len(self.replay) < self.batch_size:
            return None
        batch = self.replay.sample(self.batch_size)
        state = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        action = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        done = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(state).gather(1, action)
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
            expected = reward + self.gamma * next_q * (1.0 - done)

        loss = nn.MSELoss()(q_values, expected)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.sync_target()
