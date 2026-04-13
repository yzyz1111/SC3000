from utils import *
from part1 import astar, task1_2, astar_energy
from part2 import *
G, Coord, Dist, Cost = load_data()

# Part 1 Finding a Shortest Path with An Energy Budget

# Source, Target, Energy Budget
S = '1'
T = '50'
B = 287932

# Task 1.1
def task1_1results():
    distance, path = astar(G, Dist, Coord, S, T)
    path_str = "->".join(["S"] + path[1:-1] + ["T"])
    print("── Task 1.1 ──")
    print(f"Shortest path:     {path_str}")
    print(f"Shortest distance: {distance}\n")

# Task 1.2
def task1_2results():
    result = task1_2(G, Dist, Cost, S, T, B)
    print("── Task 1.2 ──")
    if result:
        path, distance, expanded, energy = result
        path_str = "->".join(["S"] + path[1:-1] + ["T"])
        print(f"Shortest path:     {path_str}")
        print(f"Shortest distance: {distance}")
        print(f"Expanded nodes:    {expanded}")
        print(f"Total energy:      {energy}\n")

# Task 1.3
def task1_3results():
    result = astar_energy(G, Dist, Cost, Coord, S, T, B)
    print("── Task 1.3 ──")
    if result[0] == float('inf'):
        print("No valid path found within energy budget")
        return
    distance, path, energy, expanded = result
    path_str = "->".join(["S"] + path[1:-1] + ["T"])
    print(f"Shortest path:     {path_str}")
    print(f"Shortest distance: {distance}")
    print(f"Expanded nodes:    {expanded}")
    print(f"Total energy:      {energy}\n")

task1_1results()

task1_2results()

task1_3results()

# Part 2 Solving MDP and Reinforcement Learning Problems Usinga Grid World

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

# Task 2.1 Value Iteration
print("── Task 2.1 ──")
V_vi, policy_vi, iters_vi = value_iteration()
print_gridworld(V_vi, policy_vi, iters_vi, "Value Iteration")

# Task 2.1 Policy Iteration
V_pi, policy_pi, iters_pi = policy_iteration()
print_gridworld(V_pi, policy_pi, iters_pi, "Policy Iteration")

# Task 2.2 Monte Carlo Control
print("── Task 2.2 ──")
Q_mc = monte_carlo()
policy_mc = extract_policy(Q_mc)
policy_mc[GOAL] = "G"
V_mc = {(x, y): max(Q_mc[(x, y, a)] for a in ACTIONS)
        for x in range(GRID_SIZE) for y in range(GRID_SIZE)
        if (x, y) not in ROADBLOCKS}
print_gridworld(V_mc, policy_mc, EPISODES, "Monte Carlo")

print("\n── Q Values at START (0,0) ──")
for a in ACTIONS:
    print(f"Q({START}, {a}) = {Q_mc[(0, 0, a)]:.3f}\n")

# Task 2.3 Q-Learning
print("── Task 2.3 ──")
Q_ql = q_learning()
policy_ql = extract_policy(Q_ql)
policy_ql[GOAL] = "G"
V_ql = {(x, y): max(Q_ql[(x, y, a)] for a in ACTIONS)
        for x in range(GRID_SIZE) for y in range(GRID_SIZE)
        if (x, y) not in ROADBLOCKS}
print_gridworld(V_ql, policy_ql, EPISODES_QL, "Q-Learning")    