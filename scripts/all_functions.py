import itertools
import time
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
import networkx as nx

# Define a class to represent points with x and y coordinates


class Points:
    def __init__(self, x_coordinate, y_coordinate):
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate

    def __str__(self):
        return f"({self.x_coordinate}, {self.y_coordinate})"

# Function to calculate distance between two points using a modified distance formula


def calculate_distance(point1, point2):
    x_distance = 1 / 0.9 * abs(point1.x_coordinate - point2.x_coordinate)
    y_distance = abs(point1.y_coordinate - point2.y_coordinate)
    return max(x_distance, y_distance)

# Function to read points from a file and calculate distances between all pairs of points


def get_distance_dict(file_path):
    global points_dict
    points_dict = {}
    points_dict[0] = Points(0, 0)  # Add origin point
    with open(file_path, "r") as file:
        num_of_points = int(file.readline().strip())
        for i in range(1, num_of_points + 1):
            _, x, y = file.readline().split()
            points_dict[i] = Points(float(x), float(y))

    # Generate all combinations of points and calculate distances between them
    distance_dict = {(0, 0): 0}
    for combination in itertools.permutations(points_dict.keys(), 2):
        distance_dict[combination] = calculate_distance(
            points_dict[combination[0]], points_dict[combination[1]]
        )

    return distance_dict, points_dict

# Function implementing the 2-opt local search algorithm for TSP


def twoOpt(tour, distance_dict, total):
    start_time = time.time()
    tour = tour
    n = len(tour)
    improvement = True
    old_distance = calculate_total_distance(
        distance_dict=distance_dict, route=tour, nn=True
    )
    while improvement:
        improvement = False
        for i in range(1, n - 1):
            if time.time() - start_time > total * 60:
                print("Time exceeded " + str(total) + "minutes")
                return tour

            for j in range(i + 1, n - 1):
                new_tour = tour[0:i] + tour[i: j + 1][::-1] + tour[j + 1: n]
                new_distance = calculate_total_distance(
                    distance_dict=distance_dict, route=new_tour, nn=True
                )
                if new_distance < old_distance:
                    tour = new_tour
                    improvement = True
                    break

            if improvement:
                old_distance = new_distance
                break
    return tour

# Function to plot the optimized tour using matplotlib


def plot_tour(points, route, nn=False):
    plt.figure(figsize=(10, 6))

    # Plot points
    for point in points.values():
        plt.plot(point.x_coordinate, point.y_coordinate, "bo")

    # Plot lines connecting the points in the tour
    if nn:
        for k in range(len(route) - 1):
            i = route[k]
            j = route[k + 1]
            plt.plot(
                [points[i].x_coordinate, points[j].x_coordinate],
                [points[i].y_coordinate, points[j].y_coordinate],
                "b-",
            )
    else:
        for i, j in route:
            plt.plot(
                [points[i].x_coordinate, points[j].x_coordinate],
                [points[i].y_coordinate, points[j].y_coordinate],
                "b-",
            )

    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Optimal Tour")
    plt.grid(True)
    plt.show()

# Function to generate an initial solution using the Nearest Neighbor heuristic


def nearest_neighbour_v2(distance_dict, points):
    print("NN started")

    point_to_connections = {p: {} for p in points}
    for (p1, p2), dist in distance_dict.items():
        point_to_connections[p1][p2] = dist
        point_to_connections[p2][p1] = dist

    # Helper function to find the nearest unvisited point
    def find_nearest_point(point, unvisited_points):
        closest_point, min_distance = None, float("inf")
        connections = point_to_connections[point]
        for p, dist in connections.items():
            if p in unvisited_points and dist < min_distance:
                closest_point, min_distance = p, dist
        return closest_point

    # Build the initial tour starting from the origin
    tour = [0]
    unvisited_points = set(points)
    unvisited_points.remove(0)
    current_point = 0
    while unvisited_points:
        nearest_point = find_nearest_point(current_point, unvisited_points)
        tour.append(nearest_point)
        unvisited_points.remove(nearest_point)
        current_point = nearest_point

    tour.append(0)  # Return to the origin
    return tour

# Function to calculate the total distance of a route


def calculate_total_distance(distance_dict, route, nn=False):
    total_distance = 0
    if nn:
        for k in range(len(route) - 1):
            i = route[k]
            j = route[k + 1]
            total_distance += distance_dict[(int(i), int(j))]
    else:
        for travel in route:
            total_distance += distance_dict[travel]

    return total_distance

# Function to load an initial solution into the Gurobi model


def load_initial_solution(model, initial_tour, vars, u):
    for var in vars.values():
        var.start = 0

    # Set starting values for the decision variables based on the initial tour
    tour_edges = zip(initial_tour[:-1], initial_tour[1:])
    for i, j in tour_edges:
        if (i, j) in vars:
            vars[i, j].start = 1

    # Set starting values for the subtour elimination variables
    for index, city in enumerate(initial_tour[:-1]):
        u[city].start = index + 1

# Function to optimize the TSP problem using Gurobi with an initial solution


def optimize_tsp_with_initial_solution(distance_dict, points, initial_tour, time):
    model = Model("TSP")

    # Decision variables: x[i, j] is 1 if the path is part of the route, else 0
    vars = {}
    for i, j in distance_dict.keys():
        vars[i, j] = model.addVar(
            obj=distance_dict[i, j], vtype=GRB.BINARY, name=f"x_{i}_{j}"
        )

    # Subtour elimination variables: u[i] is the position of point i in the tour
    u = model.addVars(points, vtype=GRB.CONTINUOUS, name="u")

    # Constraints: Each city must be entered and left exactly once
    for i in points:
        model.addConstr(
            quicksum(vars[i, j] for j in points if (i, j) in vars) == 1,
            name=f"enter_{i}",
        )
        model.addConstr(
            quicksum(vars[j, i] for j in points if (j, i) in vars) == 1,
            name=f"leave_{i}",
        )

    # Subtour elimination constraints (skip for the start city 0)
    for i in points:
        for j in points:
            if i != j and (i != 0 and j != 0) and (i, j) in vars:
                model.addConstr(
                    u[i] - u[j] + len(points) * vars[i, j] <= len(points) - 1,
                    name=f"subtour_{i}_{j}",
                )

    # Constraint for point 0 to start and end the tour
    model.addConstr(
        quicksum(vars[0, j] for j in points if (0, j) in vars) == 1, name="leave_0"
    )
    model.addConstr(
        quicksum(vars[i, 0] for i in points if (i, 0) in vars) == 1, name="enter_0"
    )

    # Load initial solution
    print("Initial solution loaded...")
    load_initial_solution(model, initial_tour, vars, u)

    # Set the model to focus on finding a feasible solution quickly
    model.Params.timeLimit = time * 60
    model.setParam("MIPFocus", 1)

    # Optimize the model
    model.optimize()

    # Check if a feasible solution is found
    if model.SolCount > 0:
        mip_gap = model.MIPGap
        print(f"The solution is within {mip_gap:.2%} of the optimal value.")
        solution = model.getAttr("X", vars)
        route = [(i, j) for i, j in solution if solution[i, j] > 0.5]
        return route
    else:
        # Handle cases where no feasible solution is found
        if model.status == GRB.TIME_LIMIT:
            print("No feasible solution found within the time limit.")
        else:
            print("Optimization was unsuccessful. Status code:", model.status)
        return None

# Function to create a graph from distance data


def create_graph(distance_dict):
    G = nx.Graph()
    for (city1, city2), dist in distance_dict.items():
        G.add_edge(city1, city2, weight=dist)
    return G

# Function to compute a lower bound for the TSP using a 1-tree approach


def one_tree_lower_bound(distance_dict, start_city='A'):
    G = create_graph(distance_dict)

    # Remove the start city from the graph
    G.remove_node(start_city)

    # Compute Minimum Spanning Tree (MST) on the remaining graph
    mst = nx.minimum_spanning_tree(G, weight='weight')

    # Sum the weights of the MST
    mst_cost = sum(data['weight'] for _, _, data in mst.edges(data=True))

    # Reconnect the start city by finding the two shortest edges connecting it to the MST
    shortest_edges = []
    for node in G.nodes():
        edge_weight = distance_dict.get((start_city, node), float('inf'))
        shortest_edges.append(edge_weight)

    shortest_edges.sort()

    # Add the two shortest edges to the MST cost
    if len(shortest_edges) < 2:
        raise ValueError("Not enough edges to form a 1-tree.")

    one_tree_cost = mst_cost + shortest_edges[0] + shortest_edges[1]

    return one_tree_cost

# Function to generate all possible 3-opt variants for a given path


def generate_3opt_variants(path, i, j, k):
    variants = [
        path[:i] + path[i:j][::-1] + path[j:k] +
        path[k:],  # Case 1: reverse (i,j)
        path[:i] + path[i:j] + path[j:k][::-1] + \
        path[k:],  # Case 2: reverse (j,k)
        path[:i] + path[j:k] + path[i:j] + \
        path[k:],  # Case 3: swap (i,j) and (j,k)
        # Case 4: reverse (i,j) and swap with (j,k)
        path[:i] + path[j:k] + path[i:j][::-1] + path[k:],
        path[:i] + path[i:j][::-1] + path[j:k][::-1] + \
        path[k:],  # Case 5: reverse (i,j) and (j,k)
        path[:i] + path[k-1:j-1:-1] + path[i:j][::-1] + \
        path[k:],  # Case 6: reverse (j,k) and swap with (i,j)
        path[:i] + path[k-1:j-1:-1] + path[i:j] + \
        path[k:],  # Case 7: reverse all three segments
    ]

    return variants

# Function to improve a given path using 3-opt optimization


def improve_3opt(path, distance_dict, total=5):
    start_time = time.time()
    n = len(path)
    old_distance = calculate_total_distance(
        route=path, distance_dict=distance_dict, nn=True)
    improvement = True

    while improvement:
        improvement = False
        for (i, j, k) in itertools.combinations(range(1, n), 3):
            if k - j == 1 or j - i == 1:
                continue  # Skip adjacent indices

            variants = generate_3opt_variants(path, i, j, k)

            for variant in variants:
                new_distance = calculate_total_distance(
                    route=variant, distance_dict=distance_dict, nn=True)

                if (time.time() - start_time) > total * 60:
                    print("Time exceeded 5 minutes")
                    return path

                if new_distance < old_distance:
                    path = variant
                    old_distance = new_distance
                    improvement = True
                    break  # Exit inner loop to restart with the new path

            if improvement:
                break  # Exit outer loop to restart with the new path

    return path

# Function to solve TSP using 3-opt optimization


def solve_tsp_3opt(distance_dict, initial_tour, total=5):
    optimized_path = improve_3opt(initial_tour, distance_dict, total)
    return optimized_path
