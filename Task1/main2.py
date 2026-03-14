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

def value_iteration(theta=1e-8):
    states = get_all_states()
    
    # Step 1: Initialize all values to 0
    V = {s: 0.0 for s in states}
    
    iterations = 0
    
    # Step 2: Loop until convergence
    while True:
        delta = 0.0
        
        for s in states:
            # Goal is terminal — value stays 0
            if s == GOAL:
                continue
            
            old_v = V[s]
            
            # Try all 4 actions, keep the best
            best_val = float('-inf')
            for a in ACTIONS:
                # Compute Q(s, a)
                q_val = 0.0
                for next_state, prob in get_transition_probs(s, a):
                    r = get_reward(s, next_state)
                    q_val += prob * (r + GAMMA * V[next_state])
                best_val = max(best_val, q_val)
            
            # Update V(s) with the best action's value
            V[s] = best_val
            delta = max(delta, abs(V[s] - old_v))
        
        iterations += 1
        
        if delta < theta:
            break
    
    # Step 3: Extract the optimal policy
    policy = {}
    for s in states:
        if s == GOAL:
            policy[s] = "G"
            continue
        
        best_action = None
        best_val = float('-inf')
        for a in ACTIONS:
            q_val = 0.0
            for next_state, prob in get_transition_probs(s, a):
                r = get_reward(s, next_state)
                q_val += prob * (r + GAMMA * V[next_state])
            if q_val > best_val:
                best_val = q_val
                best_action = a
        policy[s] = best_action
    
    return V, policy, iterations

ARROW = {"U": "↑", "D": "↓", "L": "←", "R": "→", "G": "★"}

def print_results(V, policy, iters, title):
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
    
    print(f"\n Value at Start {START}: {V[START]:.4f}")

def policy_evaluation(policy, V, states, theta=1e-8):
    """Given a fixed policy, compute its value function."""
    while True:
        delta = 0.0
        for s in states:
            if s == GOAL:
                continue
            
            old_v = V[s]
            
            # Only evaluate the ONE action the policy says to take
            a = policy[s]
            val = 0.0
            for next_state, prob in get_transition_probs(s, a):
                r = get_reward(s, next_state)
                val += prob * (r + GAMMA * V[next_state])
            
            V[s] = val
            delta = max(delta, abs(V[s] - old_v))
        
        if delta < theta:
            break
    
    return V


def policy_improvement(V, states):
    """Given a value function, pick the best action at every state."""
    policy = {}
    for s in states:
        if s == GOAL:
            policy[s] = "G"
            continue
        
        best_action = None
        best_val = float('-inf')
        for a in ACTIONS:
            q_val = 0.0
            for next_state, prob in get_transition_probs(s, a):
                r = get_reward(s, next_state)
                q_val += prob * (r + GAMMA * V[next_state])
            if q_val > best_val:
                best_val = q_val
                best_action = a
        policy[s] = best_action
    
    return policy


def policy_iteration():
    states = get_all_states()
    
    # Step 1: Start with arbitrary policy (all Up)
    policy = {s: "U" for s in states}
    policy[GOAL] = "G"
    V = {s: 0.0 for s in states}
    
    iterations = 0
    
    while True:
        # Step 2: Evaluate current policy
        V = policy_evaluation(policy, V, states)
        
        # Step 3: Improve policy greedily
        new_policy = policy_improvement(V, states)
        
        iterations += 1
        
        # Step 4: If policy didn't change, we're done
        if all(new_policy[s] == policy[s] for s in states):
            break
        
        policy = new_policy
    
    return V, policy, iterations

V_vi, policy_vi, iters_vi = value_iteration()
print_results(V_vi, policy_vi, iters_vi, "Value Iteration")

V_pi, policy_pi, iters_pi = policy_iteration()
print_results(V_pi, policy_pi, iters_pi, "Policy Iteration")
