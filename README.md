# Circular Tic Tac Toe

Files:
- env/circular_env.py    -> Gym env with reward shaping
- agent/q_agent.py       -> Tabular Q-learning
- agent/dqn_agent.py     -> DQN agent (PyTorch)
- train/*.py             -> training scripts
- play/play_cli.py       -> simple CLI to interact

Install:
    pip install -r requirements.txt

Train Q-learning:
    python train/train_q_learning.py

Train DQN:
    python train/train_dqn.py

Train PPO (requires stable-baselines3):
    python train/train_ppo.py

Play (simple CLI):
    python play/play_cli.py --dqn models/dqn_model.pth

Notes:
- The environment is 4x4 (4 rings x 4 sectors). Reward shaping implemented per design.
- You can change opponent_policy when creating the env to 'greedy_block' for harder opponents.
