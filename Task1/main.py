from utils import astar, load_data, heuristic, reconstruct_path, task1_2, astar_energy
G, Coord, Dist, Cost = load_data()

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
        path, shortest_dist, expanded, total_energy = result
        path_str = "->".join(["S"] + path[1:-1] + ["T"])
        print(f"Shortest path:     {path_str}")
        print(f"Shortest distance: {shortest_dist}")
        print(f"Expanded nodes:    {expanded}")
        print(f"Total energy:      {total_energy}")
    else:
        print("No valid path found within energy budget")

def task1_3results():
    distance, path, energy = astar_energy(G, Dist, Cost, Coord, S, T, B)
    path_str = "->".join(["S"] + path[1:-1] + ["T"])
    print("── Task 1.3 ──")
    print(f"Shortest path:     {path_str}")
    print(f"Shortest distance: {distance}")
    print(f"Total energy:      {energy}")

task1_1results()

task1_2results()

task1_3results()