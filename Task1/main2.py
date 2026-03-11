import numpy as np
import random

GRID_SIZE = 5
ACTIONS = ["U", "D", "L", "R"]
ACTION_MAP = {
    "U": (0, 1),   # Up increases y
    "D": (0, -1),  # Down decreases y
    "L": (-1, 0),  # Left decreases x
    "R": (1, 0),   # Right increases x
}

GOAL = (4, 4)
START = (0, 0)
ROADBLOCKS = {(2, 1), (2, 3)}
GAMMA = 0.9
STEP_REWARD = -1.0
GOAL_REWARD = 10.0

def get_all_states():
    """Return list of all valid (non-roadblock) states."""
    states = []
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if (x, y) not in ROADBLOCKS:
                states.append((x, y))
    return states


def move(state, action):
    """
    Attempt to move from state in the given direction.
    Returns new state, or same state if hitting wall/roadblock.
    """
    dx, dy = ACTION_MAP[action]
    nx, ny = state[0] + dx, state[1] + dy
    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in ROADBLOCKS:
        return (nx, ny)
    return state


def get_reward(state, next_state):
    """
    -1 per step. +10 if reaching goal (net +9).
    0 if already at goal (terminal).
    """
    if state == GOAL:
        return 0.0
    if next_state == GOAL:
        return STEP_REWARD + GOAL_REWARD  # -1 + 10 = +9
    return STEP_REWARD

def stochastic_transition(state, action):
    """
    Simulate one step in the environment.
    Agent calls this as a black box — doesn't see the probabilities.
    """
    if action == "U":
        transitions = [("U", 0.8), ("L", 0.1), ("R", 0.1)]
    elif action == "D":
        transitions = [("D", 0.8), ("L", 0.1), ("R", 0.1)]
    elif action == "L":
        transitions = [("L", 0.8), ("U", 0.1), ("D", 0.1)]
    elif action == "R":
        transitions = [("R", 0.8), ("U", 0.1), ("D", 0.1)]

    r = random.random()
    cum_prob = 0.0
    for a, p in transitions:
        cum_prob += p
        if r <= cum_prob:
            return move(state, a)

    return state