import heapq
from utils import astar, load_data, heuristic, reconstruct_path
G, Coord, Dist, Cost = load_data()

S = '1'
T = '50'
B = 287932

# Task 1.1
distance, path = astar(G, Dist, Coord, S, T)
path_str = "->".join(["S"] + path[1:-1] + ["T"])
print("── Task 1.1 ──")
print(f"Shortest path:     {path_str}")
print(f"Shortest distance: {distance}\n")

# Task 1.2
def task1_2(G, Dist, Cost, source, target, budget) -> tuple[list[str], float, int, int]:
    
    # (total_distance, total_energy, node, path)
    pq = [(0, 0, source, [])]
    
    energy_cost = {source: 0}
    
    expanded = 0

    while pq:
        node_distance, node_energy, current_node, path = heapq.heappop(pq)
        expanded += 1

        # Goal reached
        if current_node == target:
            return (path + [current_node], node_distance, expanded, node_energy)

        # Expand neighbors
        for neighbor in G[current_node]:
            new_dist   = node_distance + Dist[f"{current_node},{neighbor}"]
            new_energy = node_energy   + Cost[f"{current_node},{neighbor}"]

            # Only proceed if within budget AND better energy found
            if new_energy <= budget and (neighbor not in energy_cost or new_energy < energy_cost[neighbor]):
                heapq.heappush(pq, (new_dist, new_energy, neighbor, path + [current_node]))
                energy_cost[neighbor] = new_energy

    return None  # No valid path found

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