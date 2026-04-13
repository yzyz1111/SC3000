from utils import *

# Task 2.1 Value Iteration and Policy Iteration

# Value Iteration

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

# Policy Iteration

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


# Task 2.2 Monte Carlo Control

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

ALPHA = 0.1
EPISODES_QL = 100000

# Task 2.3 Q-Learning

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