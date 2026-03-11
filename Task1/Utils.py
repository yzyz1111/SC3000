import json
import math
import heapq
import os

# Constants
S = '1'
T = '50'
B = 287932


def load_data():
    with open('Data/G.json') as f: G = json.load(f)
    with open('Data/Coord.json') as f: Coord = json.load(f)
    with open('Data/Dist.json') as f: Dist = json.load(f)
    with open('Data/Cost.json') as f: Cost = json.load(f)
    return G, Coord, Dist, Cost

# ── Task 1.1: A* Shortest Path ────────────────────────────
def heuristic(Coord, v, target):
    x1, y1 = Coord[v]
    x2, y2 = Coord[target]
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def reconstruct_path(prev, target):
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1]

def astar(G, Dist, Coord, source, target):
    pq = [(0, 0, source)]
    g_score = {source: 0}
    visited = set()
    prev = {source: None}

    while pq:
        f, g, v = heapq.heappop(pq)
        if v in visited: continue
        visited.add(v)
        if v == target:
            return g, reconstruct_path(prev, target)
        for w in G[v]:
            new_g = g + Dist[f"{v},{w}"]
            if w not in g_score or new_g < g_score[w]:
                g_score[w] = new_g
                prev[w] = v
                new_f = new_g + heuristic(Coord, w, target)
                heapq.heappush(pq, (new_f, new_g, w))

    return float('inf'), []

def astar_energy(G, Dist, Cost, Coord, source, target, budget):
    pq = [(0, 0, 0, source)]   # (f, g, energy, node)
    g_score = {source: 0}
    best_energy = {source: 0}
    prev = {source: None}
    visited = set()

    while pq:
        f, g, energy, v = heapq.heappop(pq)

        if (v, energy) in visited:
            continue
        visited.add((v, energy))

        if v == target:
            return g, reconstruct_path(prev, target), energy
        
        for w in G[v]:
            new_g = g + Dist[f"{v},{w}"]
            new_energy = energy + Cost[f"{v},{w}"]

            if new_energy <= budget and (w not in best_energy or new_energy < best_energy[w]):
                best_energy[w] = new_energy
                prev[w] = v
                new_f = new_g + heuristic(Coord, w, target)
                heapq.heappush(pq, (new_f, new_g, new_energy, w))

    return float('inf'), [], 0

# ── Task 1.2: UCS with Energy Constraint ─────────────────
def task1_2(G, Dist, Cost, source, target, budget) -> tuple[list[str], float, int, int]:
    pq = [(0, 0, source, [])]
    energy_cost = {source: 0}
    expanded = 0

    while pq:
        node_distance, node_energy, current_node, path = heapq.heappop(pq)
        expanded += 1
        if current_node == target:
            return (path + [current_node], node_distance, expanded, node_energy)
        for neighbor in G[current_node]:
            new_dist   = node_distance + Dist[f"{current_node},{neighbor}"]
            new_energy = node_energy   + Cost[f"{current_node},{neighbor}"]
            if new_energy <= budget and (neighbor not in energy_cost or new_energy < energy_cost[neighbor]):
                heapq.heappush(pq, (new_dist, new_energy, neighbor, path + [current_node]))
                energy_cost[neighbor] = new_energy

    return None
