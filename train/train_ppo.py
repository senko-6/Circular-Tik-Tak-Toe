# train/train_ppo.py
"""
PPO training with stable-baselines3.
Install SB3: pip install stable-baselines3[extra]
"""
import gym
import numpy as np
from env.circular_env import CircularTicTacToeEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class SB3Wrapper(gym.Env):
    # Wrap our env to produce float32 obs for sb3
    def __init__(self):
        super().__init__()
        self.env = CircularTicTacToeEnv(opponent_policy='random')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset().astype('float32')

    def step(self, action):
        obs, reward, done, info = self.env.step(int(action))
        return obs.astype('float32'), float(reward), done, info

    def render(self, mode='human'):
        return self.env.render(mode)

def train(total_timesteps=200000):
    vec_env = make_vec_env(SB3Wrapper, n_envs=8)
    model = PPO('MlpPolicy', vec_env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("models/ppo_circular")
    print("PPO training complete.")

if __name__ == "__main__":
    train()
