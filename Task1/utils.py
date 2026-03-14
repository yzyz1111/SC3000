import json

def load_data():
    with open('Data/G.json') as f: G = json.load(f)
    with open('Data/Coord.json') as f: Coord = json.load(f)
    with open('Data/Dist.json') as f: Dist = json.load(f)
    with open('Data/Cost.json') as f: Cost = json.load(f)
    return G, Coord, Dist, Cost

import random

# Part 2: Gridworld

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
ROADBLOCKS = {(1, 2), (3, 2)}
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

def get_transition_probs(state, action):
    if state == GOAL:
        return [(GOAL, 1.0)]

    if action == "U":
        outcomes = [("U", 0.8), ("L", 0.1), ("R", 0.1)]
    elif action == "D":
        outcomes = [("D", 0.8), ("L", 0.1), ("R", 0.1)]
    elif action == "L":
        outcomes = [("L", 0.8), ("U", 0.1), ("D", 0.1)]
    elif action == "R":
        outcomes = [("R", 0.8), ("U", 0.1), ("D", 0.1)]

    # Aggregate — multiple outcomes can land on same state
    trans = {}
    for a, p in outcomes:
        ns = move(state, a)
        trans[ns] = trans.get(ns, 0.0) + p

    return list(trans.items())

ARROW = {"U": "↑", "D": "↓", "L": "←", "R": "→", "G": "★"}
def print_gridworld(V, policy, iters, title):
    print(f"\n{'='*50}")
    print(f" {title} — Converged in {iters} iterations")
    print(f"{'='*50}")
    
    # Value Function
    print("\n Optimal Value Function:")
    print("   " + "".join(f"  x={x}   " for x in range(GRID_SIZE)))
    for y in range(GRID_SIZE - 1, -1, -1):
        row = ""
        for x in range(GRID_SIZE):
            if (x, y) in ROADBLOCKS:
                row += " BLOCK "
            else:
                row += f" {V[(x,y)]:6.2f} "
        print(f"y={y} |{row}|")
    
    # Policy
    print("\n Optimal Policy:")
    print("   " + "".join(f"  x={x} " for x in range(GRID_SIZE)))
    for y in range(GRID_SIZE - 1, -1, -1):
        row = ""
        for x in range(GRID_SIZE):
            if (x, y) in ROADBLOCKS:
                row += " BLK "
            else:
                row += f"  {ARROW[policy[(x,y)]]}  "
        print(f"y={y} |{row}|")
    
    print(f"\n Value at Start {START}: {V[START]:.4f}\n")