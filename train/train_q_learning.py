# train/train_q_learning.py
import numpy as np
from env.circular_env import CircularTicTacToeEnv
from agent.q_agent import QAgent
from tqdm import trange

def train(episodes=20000, eval_every=2000):
    env = CircularTicTacToeEnv(opponent_policy='random')
    agent = QAgent(env)
    wins = 0
    for ep in trange(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                if info.get('result') == 'win':
                    wins += 1
        if (ep+1) % eval_every == 0:
            print(f"Episode {ep+1} wins so far: {wins}/{eval_every}")
            wins = 0
    agent.save("models/q_agent_policy.pkl")
    print("Training finished and policy saved.")

if __name__ == "__main__":
    train()
