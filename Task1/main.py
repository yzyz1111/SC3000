from Utils import astar, load_Data, heuristic, reconstruct_path, task1_2
G, Coord, Dist, Cost = load_Data()

S = '1'
T = '50'
B = 287932

# Task 1.1
def task1_1Results():
    distance, path = astar(G, Dist, Coord, S, T)
    path_str = "->".join(["S"] + path[1:-1] + ["T"])
    print("── Task 1.1 ──")
    print(f"Shortest path:     {path_str}")
    print(f"Shortest distance: {distance}\n")

# Task 1.2
def task1_2Results():
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

task1_1Results()

task1_2Results()