import json
import math
import heapq
import os

# Constants
S = '1'
T = '50'
B = 287932

# ── Data Loading ──────────────────────────────────────────
def load_dict(file: str) -> dict:
    if not os.path.isfile(file):
        print(f"{file} is missing!")
        exit(1)
    with open(file) as f:
        return json.load(f)

def load_data():
    G     = load_dict('Data/G.json')
    Coord = load_dict('Data/Coord.json')
    Dist  = load_dict('Data/Dist.json')
    Cost  = load_dict('Data/Cost.json')
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

# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    G, Coord, Dist, Cost = load_data()

    # Task 1.1
    distance, path = astar(G, Dist, Coord, S, T)
    path_str = "->".join(["S"] + path[1:-1] + ["T"])
    print("── Task 1.1 ──")
    print(f"Shortest path:     {path_str}")
    print(f"Shortest distance: {distance}\n")

    # Task 1.2
    result = task1_2(G, Dist, Cost, S, T, B)
    print("── Task 1.2 ──")
    if result:
        path, shortest_dist, expanded, total_energy = result
        path_str = "->".join(["S"] + path[1:-1] + ["T"])
        print(f"Shortest path:     {path_str}")
        print(f"Shortest distance: {shortest_dist}")
        print(f"Expanded nodes:    {expanded}")
        print(f"Total energy:      {total_energy}")
    else:
        print("No valid path found within energy budget")