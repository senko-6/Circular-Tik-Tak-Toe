# env/utils.py
import numpy as np

def board_to_state(board):
    """Convert 2D board to immutable tuple for dict keys."""
    return tuple(board.flatten().tolist())

def available_actions(board):
    """Return list of action indices that are empty."""
    flat = board.flatten()
    return [i for i, v in enumerate(flat) if v == 0]

def idx_to_coord(idx, rings=4, sectors=4):
    r = idx // sectors
    s = idx % sectors
    return r, s

def coord_to_idx(r, s, rings=4, sectors=4):
    return r * sectors + s
