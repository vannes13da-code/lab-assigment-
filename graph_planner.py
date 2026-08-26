import sys
# Node class
# Stores the information required by the A* algorithm.
class Node:
    def __init__(self, state, parent=None, action=None, g=0, h=0):
        self.state = state          # Current graph node
        self.parent = parent        # Previous node in the path
        self.action = action        # Movement from parent to current node
        self.g = g                  # Cost from the start node
        self.h = h                  # Estimated cost to the goal
        self.f = g + h              # Total A* cost


# Priority queue for A*
# Stores nodes and removes the node with the smallest f value.
class PriorityQueueFrontier:
    def __init__(self):
        self.frontier = []

    def add(self, node):
        # Add a node to the frontier.
        self.frontier.append(node)

    def empty(self):
        # Check whether the frontier is empty.
        return len(self.frontier) == 0

    def contains_state(self, state):
        # Check whether a state is already in the frontier.
        return any(node.state == state for node in self.frontier)

    def remove(self):
        # Remove and return the node with the smallest f value.
        if self.empty():
            raise Exception("Priority queue is empty.")

        min_index = 0

        for i in range(1, len(self.frontier)):
            if self.frontier[i].f < self.frontier[min_index].f:
                min_index = i

        return self.frontier.pop(min_index)


# Street graph
# Reads the graph file and stores nodes and edges.
class StreetGraph:
    def __init__(self, filename):
        self.nodes = {}
        self.graph = {}

        # Open the graph file.
        with open(filename, "r", encoding="utf-8") as file:
            lines = file.read().splitlines()

        section = None

        # Read the file line by line.
        for line_number, line in enumerate(lines, start=1):
            line = line.strip()

            # Ignore empty lines.
            if not line:
                continue

            # Start reading nodes.
            if line == "NODOS":
                section = "nodes"
                continue

            # Start reading edges.
            if line == "ARISTAS":
                section = "edges"
                continue

            # Read node information.
            if section == "nodes":
                parts = line.split()

                if len(parts) != 3:
                    raise ValueError(
                        f"Invalid node format on line {line_number}: {line}"
                    )

                node_id = parts[0]
                x = float(parts[1])
                y = float(parts[2])

                self.nodes[node_id] = (x, y)
                self.graph[node_id] = {}

            # Read edge information.
            elif section == "edges":
                parts = line.split()

                if len(parts) < 4:
                    raise ValueError(
                        f"Invalid edge format on line {line_number}: {line}"
                    )

                u = parts[0]
                v = parts[1]
                distance = float(parts[2])
                road_type = parts[3]

                # Check that both nodes exist.
                if u not in self.nodes or v not in self.nodes:
                    raise ValueError(
                        f"Undefined node in edge on line {line_number}: {line}"
                    )

                # Store the edge information.
                edge_data = {
                    "distance": distance,
                    "type": road_type
                }

                # Add the edge in both directions because
                # the graph is undirected.
                self.graph[u][v] = edge_data
                self.graph[v][u] = edge_data

            else:
                raise ValueError(
                    f"Invalid data before NODOS or ARISTAS "
                    f"on line {line_number}: {line}"
                )

        # Check that the graph contains nodes.
        if not self.nodes:
            raise ValueError("No nodes were found in the file.")

        self.num_explored = 0

    def neighbors(self, state):
        # Return all neighboring nodes of the current state.
        return list(self.graph[state].items())

    def heuristic(self, state, goal):
        # Calculate the Euclidean distance between two nodes.
        x1, y1 = self.nodes[state]
        x2, y2 = self.nodes[goal]

        dx = x2 - x1
        dy = y2 - y1

        return (dx ** 2 + dy ** 2) ** 0.5

    def cost(self, edge_data, cost_type):
        # Calculate the cost of using an edge.
        distance = edge_data["distance"]
        road_type = edge_data["type"]

        # Use the actual distance as the cost.
        if cost_type == "distance":
            return distance

        # Modify the cost according to the road type.
        if cost_type == "custom":
            factors = {
                "avenue": 1.5,
                "residential": 0.8,
                "alley": 0.9,
                "normal": 1.0
            }

            factor = factors.get(road_type, 1.0)

            return distance * factor

        raise ValueError(
            "Invalid cost type. Use 'distance' or 'custom'."
        )

    def solve(self, start_id, goal_id, cost_type="distance"):
        # Check that the start node exists.
        if start_id not in self.nodes:
            raise ValueError(
                f"Start node '{start_id}' does not exist."
            )

        # Check that the goal node exists.
        if goal_id not in self.nodes:
            raise ValueError(
                f"Goal node '{goal_id}' does not exist."
            )

        # Reset the number of explored nodes.
        self.num_explored = 0

        # Create the initial node.
        start_node = Node(
            state=start_id,
            parent=None,
            action=None,
            g=0,
            h=self.heuristic(start_id, goal_id)
        )

        # Add the initial node to the frontier.
        frontier = PriorityQueueFrontier()
        frontier.add(start_node)

        # Store the best known cost to each node.
        g_score = {
            start_id: 0
        }

        # Store the nodes that have already been explored.
        explored = set()

        # Continue searching while there are nodes to explore.
        while not frontier.empty():
            node = frontier.remove()
            self.num_explored += 1

            # Ignore outdated nodes if a better path was found.
            if node.g > g_score.get(
                node.state,
                float("inf")
            ):
                continue

            # Check whether the goal has been reached.
            if node.state == goal_id:
                actions = []
                path = []

                current = node

                # Reconstruct the path from goal to start.
                while current is not None:
                    path.append(current.state)

                    if current.action is not None:
                        actions.append(current.action)

                    current = current.parent

                # Reverse the lists to obtain start-to-goal order.
                path.reverse()
                actions.reverse()

                return actions, path, node.g

            # Mark the current node as explored.
            explored.add(node.state)

            # Explore all neighboring nodes.
            for neighbor, edge_data in self.neighbors(node.state):

                # Calculate the cost of reaching the neighbor.
                step_cost = self.cost(
                    edge_data,
                    cost_type
                )

                tentative_g = node.g + step_cost

                # Check whether this is a better path.
                if tentative_g < g_score.get(
                    neighbor,
                    float("inf")
                ):
                    g_score[neighbor] = tentative_g

                    # Allow the node to be explored again if
                    # a better path has been found.
                    if neighbor in explored:
                        explored.remove(neighbor)

                    # Calculate the heuristic value.
                    h = self.heuristic(
                        neighbor,
                        goal_id
                    )

                    # Create the child node.
                    child = Node(
                        state=neighbor,
                        parent=node,
                        action=(node.state, neighbor),
                        g=tentative_g,
                        h=h
                    )

                    # Add the child to the frontier.
                    frontier.add(child)

        # No path was found.
        raise Exception(
            f"No route was found from "
            f"'{start_id}' to '{goal_id}'."
        )

    def real_distance(self, path):
        # Calculate the actual distance of a complete path.
        total = 0.0

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]

            total += self.graph[u][v]["distance"]

        return total


# Main function
# Loads the graph, selects the start and goal,
# runs A*, and displays the results.
def main():

    # Check that the graph file was provided.
    if len(sys.argv) != 2:
        sys.exit(
            "Usage: python graph_planner.py grafo.txt"
        )

    # Get the file name from the command line.
    filename = sys.argv[1]

    # Load the graph.
    graph = StreetGraph(filename)

    # Define the starting node.
    start = "yachay_university"

    # Define the destination node.
    goal = "Quito"

    print("=" * 60)
    print("A* STREET GRAPH")
    print("=" * 60)

    print(f"Graph file: {filename}")
    print(f"Nodes loaded: {len(graph.nodes)}")
    print(f"Start: {start}")
    print(f"Goal: {goal}")

    # Find the shortest route using actual distance.
    print("\nROUTE 1: SHORTEST DISTANCE ")

    actions, path, total_distance = graph.solve(
        start,
        goal,
        cost_type="distance"
    )

    print(
        f"Path: {' -> '.join(path)}"
    )

    print(
        f"Total distance: {total_distance:.2f}"
    )

    print(
        f"Nodes explored: {graph.num_explored}"
    )

    # Find the route using the custom road-type cost.
    print("\n ROUTE 2: CUSTOM COST")

    actions_custom, path_custom, total_custom = graph.solve(
        start,
        goal,
        cost_type="custom"
    )

    print(
        f"Path: {' -> '.join(path_custom)}"
    )

    print(
        f"Custom cost: {total_custom:.2f}"
    )

    print(
        f"Real distance: "
        f"{graph.real_distance(path_custom):.2f}"
    )

    print(
        f"Nodes explored: {graph.num_explored}"
    )

    # Display the edges used in the custom route.
    print("\n ROUTE DETAILS")

    for u, v in actions_custom:
        distance = graph.graph[u][v]["distance"]
        road_type = graph.graph[u][v]["type"]

        print(
            f"{u} -> {v} | "
            f"distance = {distance:.2f} | "
            f"type = {road_type}"
        )


# Run the main function when the file is executed directly.
if __name__ == "__main__":
    main()