# train/train_dqn.py
import numpy as np
from env.circular_env import CircularTicTacToeEnv
from agent.dqn_agent import DQNAgent
from tqdm import trange

def train(episodes=20000, sync_every=500):
    env = CircularTicTacToeEnv(opponent_policy='random')
    agent = DQNAgent(env)
    for ep in trange(episodes):
        state = env.reset()
        done = False
        total_r = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.push_transition(state, action, reward, next_state, float(done))
            loss = agent.update()
            state = next_state
            total_r += reward
        if ep % sync_every == 0:
            agent.sync_target()
        if ep % 1000 == 0:
            print(f"Ep {ep} total reward {total_r}")
    agent.save("models/dqn_model.pth")
    print("DQN training finished and model saved.")

if __name__ == "__main__":
    train()
