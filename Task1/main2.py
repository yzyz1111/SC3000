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

EPISODES = 200000

def init_Q():
    Q = {}
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if (x, y) not in ROADBLOCKS:
                for a in ACTIONS:
                    Q[(x, y, a)] = 0.0
    return Q

EPSILON = 0.1

def epsilon_greedy(Q, state, epsilon=EPSILON):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: Q[(state[0], state[1], a)])

def generate_episode(Q, max_steps=2000):
    episode = []
    state = START
    steps = 0

    while state != GOAL and steps < max_steps:
        action = epsilon_greedy(Q, state)
        next_state = stochastic_transition(state, action)
        reward = get_reward(state, next_state)
        episode.append((state, action, reward))
        state = next_state
        steps += 1

    return episode

def monte_carlo(episodes=EPISODES):
    Q = init_Q()
    N = {(x, y, a): 0 for x in range(GRID_SIZE)
                       for y in range(GRID_SIZE)
                       for a in ACTIONS
                       if (x,y) not in ROADBLOCKS}

    for _ in range(episodes):
        episode = generate_episode(Q)
        G = 0
        visited = set()

        for state, action, reward in reversed(episode):
            G = reward + GAMMA * G
            if (state, action) not in visited:
                visited.add((state, action))
                N[(state[0], state[1], action)] += 1
                # Incremental mean — no need to store all returns
                Q[(state[0], state[1], action)] += (G - Q[(state[0], state[1], action)]) / N[(state[0], state[1], action)]

    return Q

def extract_policy(Q):
    policy = {}
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if (x, y) not in ROADBLOCKS and (x, y) != GOAL:
                policy[(x, y)] = max(ACTIONS, key=lambda a: Q[(x, y, a)])
    return policy

Q = monte_carlo()
policy = extract_policy(Q)

print("── Monte Carlo Policy ──")
for y in range(GRID_SIZE-1, -1, -1):
    row = ""
    for x in range(GRID_SIZE):
        if (x, y) == GOAL:
            row += " G  "
        elif (x, y) in ROADBLOCKS:
            row += " X  "
        else:
            row += f" {policy[(x,y)]}  "
    print(row)

print("\n── Q Values at START (0,0) ──")
for a in ACTIONS:
    print(f"Q({START}, {a}) = {Q[(0, 0, a)]:.3f}")

ALPHA = 0.1
EPISODES_QL = 100000

def q_learning(episodes=EPISODES_QL):
    Q = init_Q()

    for ep in range(episodes):
        state = START

        while state != GOAL:
            action = epsilon_greedy(Q, state)
            next_state = stochastic_transition(state, action)
            reward = get_reward(state, next_state)

            # Q-learning update
            best_next = max(Q[(next_state[0], next_state[1], a)] for a in ACTIONS) if next_state != GOAL else 0.0
            Q[(state[0], state[1], action)] += ALPHA * (reward + GAMMA * best_next - Q[(state[0], state[1], action)])

            state = next_state

    return Q

Q_ql = q_learning()
policy_ql = extract_policy(Q_ql)

print("── Q-Learning Policy ──")
for y in range(GRID_SIZE-1, -1, -1):
    row = ""
    for x in range(GRID_SIZE):
        if (x, y) == GOAL:
            row += " G  "
        elif (x, y) in ROADBLOCKS:
            row += " X  "
        else:
            row += f" {policy_ql[(x,y)]}  "
    print(row)