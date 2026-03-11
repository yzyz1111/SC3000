
import json 
import math
import heapq
def load_Data():
    with open('Data/G.json') as f: G = json.load(f)
    with open('Data/Coord.json') as f: Coord = json.load(f)
    with open('Data/Dist.json') as f: Dist = json.load(f)
    return G, Coord, Dist

G, Coord, Dist = load_Data()

print(list(G.keys())[:5])
print(list(G.items())[:5])
print('\n')

print(list(Coord.keys())[:5])
print(list(Coord.items())[:5])
print('\n')

print(list(Dist.keys())[:5])
print(list(Dist.items())[:5])
print('\n')

def heuristic(Coord, v, target):
    x1, y1 = Coord[v]
    x2, y2 = Coord[target]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

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
        
        if v in visited:
            continue
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

    return float('inf')

source, target = "1", "50"
distance, path = astar(G, Dist, Coord, source, target)
path_str = "->".join(["S"] + path[1:-1] + ["T"])
print(f"Shortest path: {path_str}")
print(f"Shortest distance: {distance}")
