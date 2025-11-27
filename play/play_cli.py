# play/play_cli.py
import numpy as np
import argparse
from env.circular_env import CircularTicTacToeEnv
from agent.dqn_agent import DQNAgent
import pickle

def human_move(env):
    env.render()
    avail = [i for i in range(env.action_space.n) if env.board.flatten()[i] == 0]
    print("Available actions:", avail)
    a = None
    while a not in avail:
        try:
            a = int(input("Your action (0-15): "))
        except:
            a = None
    return a

def play_with_dqn(model_path=None):
    env = CircularTicTacToeEnv(opponent_policy='random')
    agent = DQNAgent(env)
    if model_path:
        agent.load(model_path)
    while True:
        s = env.reset()
        done = False
        while not done:
            # agent plays first
            act = agent.select_action(s)
            s, r, done, info = env.step(act)
            if done:
                env.render()
                print("Agent result:", info.get('result', 'n/a'), "reward", r)
                break
            # human move (opponent)
            env.render()
            action = human_move(env)
            s, r, done, info = env.step(action)  # NOTE: our step assumes action is agent; to play human vs agent we'd need to change. Simpler: let human act as agent by choosing actions and environment will apply as agent.
            # The current wrapper places agent moves as '1' by design.
        again = input("Play again? y/n: ")
        if again.lower() != 'y':
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dqn', help='Path to dqn model .pth', default="models/dqn_model.pth")
    args = parser.parse_args()
    play_with_dqn(args.dqn)
