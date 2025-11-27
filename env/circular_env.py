# env/circular_env.py
import gym
from gym import spaces
import numpy as np
from .utils import board_to_state, available_actions, idx_to_coord, coord_to_idx

RINGS = 4
SECTORS = 4
BOARD_SIZE = RINGS * SECTORS

class CircularTicTacToeEnv(gym.Env):
    """
    Gym-compatible environment for Circular Tic Tac Toe (4x4).
    State: flattened 16-length vector with values {0: empty, 1: agent, -1: opponent}
    Action: Discrete(16) positions (0..15)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, opponent_policy='random'):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(BOARD_SIZE,), dtype=np.int8)
        self.action_space = spaces.Discrete(BOARD_SIZE)
        self.opponent_policy = opponent_policy
        self.reset()

    def reset(self):
        self.board = np.zeros((RINGS, SECTORS), dtype=np.int8)
        self.current_player = 1  # 1 -> agent, -1 -> opponent
        self.done = False
        self.last_move = None
        return self._get_obs()

    def _get_obs(self):
        return self.board.flatten().astype(np.int8)

    # -------------- win/detection utilities --------------
    def _check_line_win(self, line):
        """Return 1 if agent wins on line, -1 if opponent, else 0."""
        s = np.sum(line)
        if s == RINGS:       # all 1s along radial? Not used like that; use row/col checks below
            return 1
        if s == -RINGS:
            return -1
        return 0

    def _is_win(self, player):
        """
        Winning conditions:
         - any full ring (row) with all player's marks
         - any full sector (column) with all player's marks
         - two main diagonals (top-left to bottom-right and top-right to bottom-left)
        """
        b = self.board
        p = player
        # full ring (row)
        for r in range(RINGS):
            if np.all(b[r, :] == p):
                return True
        # full sector (col / radial)
        for s in range(SECTORS):
            if np.all(b[:, s] == p):
                return True
        # diagonals
        if np.all(np.diag(b) == p):
            return True
        if np.all(np.diag(np.fliplr(b)) == p):
            return True
        return False

    def _is_draw(self):
        return np.all(self.board != 0)

    def _detect_n_in_line(self, player, n=2):
        """
        Detect if player has at least one line with exactly n marks and others empty (threat).
        Return True if at least one such pattern exists (rows, cols, diag).
        """
        b = self.board
        # rows
        for r in range(RINGS):
            row = b[r, :]
            if np.count_nonzero(row == player) == n and np.count_nonzero(row == 0) == (SECTORS - n):
                return True
        # cols
        for s in range(SECTORS):
            col = b[:, s]
            if np.count_nonzero(col == player) == n and np.count_nonzero(col == 0) == (RINGS - n):
                return True
        # main diag
        d = np.diag(b)
        if np.count_nonzero(d == player) == n and np.count_nonzero(d == 0) == (len(d) - n):
            return True
        d2 = np.diag(np.fliplr(b))
        if np.count_nonzero(d2 == player) == n and np.count_nonzero(d2 == 0) == (len(d2) - n):
            return True
        return False

    def opponent_about_to_win(self):
        return self._detect_n_in_line(-1, n=RINGS-1) or self._detect_n_in_line(-1, n=3)

    # -------------- reward function --------------
    def compute_reward(self, action, info=None):
        """
        Reward shaping:
        - Win: +20
        - Lose: -20
        - Draw: +2
        - Block opponent's immediate win: +6
        - Create 2-in-line: +3
        - Create 3-in-line: +5
        - Invalid move: -8
        - Neutral valid move: 0
        """
        r = 0
        # invalid
        r_idx = action
        r_coord = idx_to_coord(r_idx, RINGS, SECTORS)
        if self.board[r_coord] != 0:
            return -8

        # apply move temporarily to evaluate shaping
        temp = self.board.copy()
        temp[r_coord] = 1  # agent played here

        # check if this blocks opponent immediate win:
        # opponent had a threat: n = RINGS-1 or 3
        opp_threat_before = self._detect_n_in_line(-1, n=RINGS-1) or self._detect_n_in_line(-1, n=3)
        # after move, check if opponent still has that threat
        b_after = temp
        def detect_n_in_line_board(board, player, n):
            # rows
            for rr in range(RINGS):
                row = board[rr, :]
                if np.count_nonzero(row == player) == n and np.count_nonzero(row == 0) == (SECTORS - n):
                    return True
            for ss in range(SECTORS):
                col = board[:, ss]
                if np.count_nonzero(col == player) == n and np.count_nonzero(col == 0) == (RINGS - n):
                    return True
            d = np.diag(board)
            if np.count_nonzero(d == player) == n and np.count_nonzero(d == 0) == (len(d) - n):
                return True
            d2 = np.diag(np.fliplr(board))
            if np.count_nonzero(d2 == player) == n and np.count_nonzero(d2 == 0) == (len(d2) - n):
                return True
            return False

        opp_threat_after = detect_n_in_line_board(b_after, -1, RINGS-1) or detect_n_in_line_board(b_after, -1, 3)
        if opp_threat_before and not opp_threat_after:
            r += 6

        # create 3-in-line
        if detect_n_in_line_board(b_after, 1, 3):
            r += 5

        # create 2-in-line
        if detect_n_in_line_board(b_after, 1, 2):
            r += 3

        return r

    # -------------- step/reset/render --------------
    def step(self, action):
        """
        Agent acts as player 1 (1). Opponent is -1, plays after agent.
        Returns obs, reward, done, info
        """
        info = {}
        # invalid
        r_idx = int(action)
        r_coord = idx_to_coord(r_idx, RINGS, SECTORS)
        if self.board[r_coord] != 0:
            reward = -8
            done = False
            # still allow to continue; penalize and return state
            return self._get_obs(), reward, done, info

        # agent move
        self.board[r_coord] = 1
        self.last_move = r_coord

        # terminal?
        if self._is_win(1):
            return self._get_obs(), 20, True, {'result': 'win'}
        if self._is_draw():
            return self._get_obs(), 2, True, {'result': 'draw'}

        # shaped reward for the move (non-terminal)
        shaped = self.compute_reward(action)

        # opponent move
        opp_action = self._opponent_move()
        if opp_action is not None:
            self.board[opp_action] = -1

        # after opponent move, check terminal
        if self._is_win(-1):
            return self._get_obs(), -20, True, {'result': 'lose'}
        if self._is_draw():
            return self._get_obs(), 2, True, {'result': 'draw'}

        return self._get_obs(), shaped, False, {}

    def _opponent_move(self):
        """Simple opponent policies: 'random' or 'blocking' or 'minimax' (not implemented)."""
        avail = available_actions(self.board)
        if not avail:
            return None
        if self.opponent_policy == 'random':
            return np.random.choice(avail)
        elif self.opponent_policy == 'greedy_block':
            # if opponent can win, play that; else random; this function returns index to place -1
            for a in avail:
                r,c = idx_to_coord(a,RINGS,SECTORS)
                temp = self.board.copy()
                temp[r,c] = -1
                # if opponent would win, take it
                if self._is_win_board(temp, -1):
                    return a
            # else block agent if agent can win next
            for a in avail:
                r,c = idx_to_coord(a,RINGS,SECTORS)
                temp = self.board.copy()
                temp[r,c] = 1
                if self._is_win_board(temp, 1):
                    return a
            return np.random.choice(avail)
        else:
            return np.random.choice(avail)

    def _is_win_board(self, board, player):
        # same logic as _is_win but for provided board
        b = board
        p = player
        for r in range(RINGS):
            if np.all(b[r, :] == p):
                return True
        for s in range(SECTORS):
            if np.all(b[:, s] == p):
                return True
        if np.all(np.diag(b) == p):
            return True
        if np.all(np.diag(np.fliplr(b)) == p):
            return True
        return False

    def render(self, mode='human'):
        # simple text rendering
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print("Board:")
        for r in range(RINGS):
            print(" ".join(symbols[int(x)] for x in self.board[r, :]))
        print("---")
