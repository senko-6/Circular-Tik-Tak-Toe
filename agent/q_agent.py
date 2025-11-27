# agent/q_agent.py
import random
import pickle
import numpy as np
from collections import defaultdict
from env.utils import board_to_state, idx_to_coord

class QAgent:
    def __init__(self, env, lr=0.7, gamma=0.99, epsilon=1.0, eps_min=0.05, eps_decay=0.9997):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.q = defaultdict(lambda: np.zeros(env.action_space.n, dtype=np.float32))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state_key = tuple(state.tolist())
        return int(np.argmax(self.q[state_key]))

    def update(self, state, action, reward, next_state, done):
        s = tuple(state.tolist())
        ns = tuple(next_state.tolist())
        best_next = 0 if done else np.max(self.q[ns])
        td = reward + self.gamma * best_next - self.q[s][action]
        self.q[s][action] += self.lr * td
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q), f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q = defaultdict(lambda: np.zeros(self.env.action_space.n, dtype=np.float32), data)
