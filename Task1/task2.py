import heapq
import os
import json

S = '1'
T = '50'
B = 287932

def load_dict(file: str) -> dict:
    if not os.path.isfile(file):
        print(f"{file} is missing!")
        exit(1)
    with open(file) as json_file:
        return json.load(json_file)

def task2(G, Dist, Cost, source, target, budget) -> tuple[list[str], float, int, int]:
    """
    UCS to find shortest distance path from source to target
    with energy cost <= budget

    Returns:
        tuple: (path, total_distance, expanded, total_energy)
    """
    
    # (total_distance, total_energy, node, path)  -- energy as tiebreaker
    pq = [(0, 0, source, [])]
    
    # node -> minimum energy used to reach it
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


# Load dictionaries
G     = load_dict('G.json')
Coord = load_dict('Coord.json')
Dist  = load_dict('Dist.json')
Cost  = load_dict('Cost.json')

# Run Task 2
result = task2(G, Dist, Cost, S, T, B)

if result:
    path, shortest_dist, expanded, total_energy = result
    print(f"Path:           {path}")
    print(f"Shortest Dist:  {shortest_dist}")
    print(f"Expanded Nodes: {expanded}")
    print(f"Total Energy:   {total_energy}")
else:
    print("No valid path found within energy budget")