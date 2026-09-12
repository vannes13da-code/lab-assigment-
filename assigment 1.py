import math
import matplotlib.pyplot as plt
class Node:
    def __init__(self, state, parent=None, action=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g # cost from start to this node
        self.h = h # heuristic
        self.f = g + h # sum of cost and heuristic
class PriorityQueueFrontier:

    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):

        if self.empty():
            raise Exception("Priority queue is empty.")

        #search for the node with the lowest f value
        min_index = 0

        for i in range(1, len(self.frontier)):

            if self.frontier[i].f < self.frontier[min_index].f:
                min_index = i

        # delete and return the node with the lowest f value
        return self.frontier.pop(min_index)
class StreetGraph:

    def __init__(self):
        self.nodes = {

            "yachay_university": (0, 0),

            "B": (0, 3),
            "C": (2, 4.5),
            "D": (3.5, 6),
            "E": (3, 7),
            "F": (5.5, 5),
            "G": (6, 1.5),
            "H": (6.5, 2.5),
            "I": (7.5, 6),
            "J": (5.5, 6.5),
            "K": (7.5, 9),
            "L": (9.5, 3.5),
            "M": (7.5, 1.5),
            "N": (6.5, 0.5),

            "Quito": (8.2, 0.5)
        }
        self.graph = {}

        for node in self.nodes:
            self.graph[node] = {}
        edges = [

            ("yachay_university", "B", 300, "residential"),

            ("B", "C", 250, "residential"),
            ("C", "D", 212, "residential"),

            ("D", "E", 112, "residential"),
            ("D", "J", 206, "residential"),
            ("D", "F", 224, "avenue"),

            ("E", "J", 255, "residential"),
            ("E", "K", 492, "avenue"),

            ("F", "G", 354, "avenue"),
            ("F", "H", 269, "residential"),
            ("F", "I", 224, "avenue"),
            ("F", "J", 150, "alley"),

            ("G", "H", 112, "alley"),
            ("G", "M", 150, "residential"),
            ("G", "N", 112, "alley"),

            ("H", "I", 364, "residential"),
            ("H", "L", 316, "residential"),
            ("H", "M", 141, "alley"),

            ("I", "J", 206, "residential"),
            ("I", "K", 300, "avenue"),
            ("I", "L", 320, "residential"),

            ("J", "K", 320, "avenue"),

            ("L", "M", 283, "residential"),
            ("L", "Quito", 327, "avenue"),

            ("M", "N", 141, "alley"),
            ("M", "Quito", 122, "avenue"),

            ("N", "Quito", 170, "avenue")
        ]
    # add edges to the graph
        for u, v, distance, road_type in edges:

            edge_data = {
                "distance": distance,
                "type": road_type
            }
            self.graph[u][v] = edge_data
            self.graph[v][u] = edge_data

        self.num_explored = 0

    def neighbors(self, state):

        return list(self.graph[state].items())
#-------------------------------------------------

    def heuristic(self, state, goal):

        x1, y1 = self.nodes[state]
        x2, y2 = self.nodes[goal]

        dx = x2 - x1
        dy = y2 - y1
        #euclidean distance as heuristic
        return math.sqrt(dx ** 2 + dy ** 2)

    def cost(self, edge_data, cost_type):

        distance = edge_data["distance"]
        road_type = edge_data["type"]
        if cost_type == "distance":

            return distance

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
            "Invalid cost type."
        )

    def solve(self, start_id, goal_id, cost_type="distance"):

        if start_id not in self.nodes:

            raise ValueError(
                f"Start node '{start_id}' does not exist."
            )

        if goal_id not in self.nodes:

            raise ValueError(
                f"Goal node '{goal_id}' does not exist."
            )

        # reset the number of explored nodes for this search
        self.num_explored = 0

        start_node = Node(

            state=start_id,

            parent=None,

            action=None,

            g=0,

            h=self.heuristic(
                start_id,
                goal_id
            )
        )

        frontier = PriorityQueueFrontier()

        frontier.add(start_node)
        #cost from start to each node
        g_score = {

            start_id: 0
        }

        #explored nodes
        explored = set()

       #search loop 
        while not frontier.empty():

            node = frontier.remove()

            self.num_explored += 1
            #ignore an old version of the node if a better path already exists
            if node.g > g_score.get(
                node.state,
                float("inf")
            ):
                continue

            if node.state == goal_id:

                actions = []
                path = []

                current = node

                # Reconstruct the path from the goal to the start by following parent nodes 
                while current is not None:

                    path.append(
                        current.state
                    )

                    if current.action is not None:

                        actions.append(
                            current.action
                        )

                    current = current.parent

                path.reverse()
                actions.reverse()

                return actions, path, node.g

            # mark the node as explored

            explored.add(node.state)

            # Explore neighbors

            for neighbor, edge_data in self.neighbors(
                node.state
            ):

                step_cost = self.cost(
                    edge_data,
                    cost_type
                )

                tentative_g = (
                    node.g + step_cost
                )

                # ¿found a better path to the neighbor?
                if tentative_g < g_score.get(
                    neighbor,
                    float("inf")
                ):

                    g_score[neighbor] = tentative_g

                    if neighbor in explored:

                        explored.remove(
                            neighbor
                        )

                    h = self.heuristic(
                        neighbor,
                        goal_id
                    )

                    child = Node(

                        state=neighbor,

                        parent=node,

                        action=(
                            node.state,
                            neighbor
                        ),

                        g=tentative_g,

                        h=h
                    )

                    frontier.add(child)
        #done searching, no path found

        raise Exception(
            f"No route was found from "
            f"'{start_id}' to '{goal_id}'."
        )


    def real_distance(self, path):

        total = 0.0

        for i in range(
            len(path) - 1
        ):

            u = path[i]
            v = path[i + 1]

            total += self.graph[u][v][
                "distance"
            ]

        return total


  #funtion to plot the graph and the path

def plot_graph(
    graph,
    path=None,
    title="Grafo de calles"
):

    plt.figure(
        figsize=(14, 10)
    )


    drawn_edges = set()

    for u in graph.graph:

        for v, edge_data in graph.graph[u].items():

            #
            edge = tuple(
                sorted([u, v])
            )

            if edge in drawn_edges:
                continue

            drawn_edges.add(edge)

            x1, y1 = graph.nodes[u]
            x2, y2 = graph.nodes[v]

            road_type = edge_data["type"]

            
            if road_type == "avenue":

                line_style = "-"
                line_width = 2.5

            elif road_type == "residential":

                line_style = "--"
                line_width = 1.5

            elif road_type == "alley":

                line_style = ":"
                line_width = 2

            else:

                line_style = "-"
                line_width = 1.5

            plt.plot(

                [x1, x2],

                [y1, y2],

                linestyle=line_style,

                linewidth=line_width,

                color="gray",

                zorder=1
            )

            xm = (x1 + x2) / 2
            ym = (y1 + y2) / 2

            plt.text( xm , ym , 
                f"{edge_data['distance']:.0f}",
                fontsize=7,
                color="dimgray",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7
                ),
                zorder=2
            )

#Ruta A* 

    if path is not None:

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            x1, y1 = graph.nodes[u]
            x2, y2 = graph.nodes[v]
            plt.plot(
                [x1, x2],
                [y1, y2],
                color="orange",
                linewidth=3,
                solid_capstyle="round",
                zorder=4
            )

    # dibujar nodos
    for node, (x, y) in graph.nodes.items():

        # Nodo inicial
        if node == "yachay_university":

            plt.scatter(

                x,
                y,

                s=220,

                color="green",

                edgecolors="black",

                linewidths=1.5,

                zorder=5
            )

        # Nodo final
        elif node == "Quito":

            plt.scatter(

                x,
                y,

                s=220,

                color="red",

                edgecolors="black",

                linewidths=1.5,

                zorder=5
            )

        # Nodos normales
        else:

            plt.scatter(

                x,
                y,

                s=100,

                color="steelblue",

                edgecolors="black",

                linewidths=1,

                zorder=5
            )

        # Nombre del nodo
        plt.annotate(

            node,

            (x, y),

            xytext=(7, 7),

            textcoords="offset points",

            fontsize=9,

            fontweight="bold"
        )

    plt.title(
        title,
        fontsize=16,
        fontweight="bold"
    )
    plt.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    plt.axis("equal")

    # Recuadro de leyenda 

    plt.plot(
        [],
        [],
        color="gray",
        linestyle="-",
        linewidth=2.5,
        label="Avenue"
    )

    plt.plot(
        [],
        [],
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Residential"
    )

    plt.plot(
        [],
        [],
        color="gray",
        linestyle=":",
        linewidth=2,
        label="Alley"
    )

    if path is not None:

        plt.plot(
            [],
            [],
            color="orange",
            linewidth=3,
            label="Ruta A*"
        )

    plt.scatter(
        [],
        [],
        color="green",
        s=100,
        label="Inicio"
    )

    plt.scatter(
        [],
        [],
        color="red",
        s=100,
        label="Destino"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


def main():

    graph = StreetGraph()
    start = "yachay_university"
    goal = "Quito"

    #distance and custom cost path
    print("A* Algorithm")
    print()

    print(
        f"Start: {start}"
    )

    print(
        f"Goal: {goal}"
    )

    print()

    print(
        " lowest distance"
    )
    actions_distance, path_distance, total_distance = (
        graph.solve(
            start,
            goal,
            cost_type="distance"
        )
    )

    print(
        "path:"
    )

    print(
        " -> ".join(path_distance)
    )

    print()

    print(
        f"Distance total: "
        f"{total_distance:.2f}"
    )

    print(
        f"Nodes explored: "
        f"{graph.num_explored}"
    )

    print()

    print(
        " custom cost"
    )
    actions_custom, path_custom, total_custom = (
        graph.solve(
            start,
            goal,
            cost_type="custom"
        )
    )

    print(
        "Path:"
    )

    print(
        " -> ".join(path_custom)
    )

    print()

    print(
        f"Custom cost: "
        f"{total_custom:.2f}"
    )

    print(
        f"Real distance: "
        f"{graph.real_distance(path_custom):.2f}"
    )

    print(
        f"Nodes explored: "
        f"{graph.num_explored}"
    )

    print()



    print(
        "Detail of the route:"
    )


    for u, v in actions_custom:

        distance = graph.graph[u][v][
            "distance"
        ]

        road_type = graph.graph[u][v][
            "type"
        ]

        print(

            f"{u} -> {v} "
            f"distance = {distance:.2f} "
            f"type = {road_type}"
        )

    print()
# The graph with the custom path highlighted
    plot_graph(

        graph,

        path=path_custom,
    )

#11
if __name__ == "__main__":
    main()