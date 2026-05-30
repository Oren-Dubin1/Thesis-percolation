import networkx as nx
from ortools.sat.python import cp_model
from utils import *
from percolation_improved import Graph

def optimal_coloring(G):
    nodes = list(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)

    model = cp_model.CpModel()

    color = [model.NewIntVar(0, n - 1, f"color_{i}") for i in range(n)]
    used = [model.NewBoolVar(f"used_{c}") for c in range(n)]

    for u, v in G.edges():
        model.Add(color[index[u]] != color[index[v]])

    for i in range(n):
        for c in range(n):
            is_c = model.NewBoolVar(f"is_{i}_{c}")
            model.Add(color[i] == c).OnlyEnforceIf(is_c)
            model.Add(color[i] != c).OnlyEnforceIf(is_c.Not())
            model.AddImplication(is_c, used[c])

    model.Add(used[0] == 1)
    for c in range(n - 1):
        model.Add(used[c] >= used[c + 1])

    model.Minimize(sum(used))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None

    coloring = {nodes[i]: solver.Value(color[i]) for i in range(n)}
    chromatic_number = len(set(coloring.values()))

    return chromatic_number, coloring

def color_witnesses(G : Graph):
    answer, order, wits = G.is_percolating(document_steps=True)
    assert answer

    colors = {v : -1 for v in range(G.graph.number_of_nodes())}
    for wit in wits:
        for v in wit:
            if colors[v] != -1:
                continue
            neighbors = [colors[u] for u in G.graph.neighbors(v)]
            for color in range(1, G.graph.number_of_nodes()):
                if color not in neighbors:
                    colors[v] = color
                    break

    return max(colors.values()), colors



if __name__ == "__main__":
    _max = -1
    max_greedy = -1
    max_nx = -1
    for n in range(7, 20):
        try:
            for g in iter_stored_graphs_for_n(n):
                chromatic_number, coloring = optimal_coloring(g)
                print(f"Graph with {g.number_of_edges()} edges has chromatic number {chromatic_number}.")
                _max = max(_max, chromatic_number)
                greedy, colors = color_witnesses(Graph(g))
                print(f"Greedy coloring based on witnesses gives {greedy} colors.")
                max_greedy = max(max_greedy, greedy)

                nx_greedy = max(nx.greedy_color(g).values()) + 1
                print(f"NetworkX greedy coloring gives {nx_greedy} colors.")
                max_nx = max(max_nx, nx_greedy)

                if greedy > 5:
                    answer, order, wits = Graph(g).is_percolating(document_steps=True)
                    print(f"Graph is percolating: {answer}")
                    print(f"Witness order: {order}")
                    print(f"Witnesses: {wits}")
                    for u,v in g.edges():
                        print(u, v)

                    print(colors)

                    raise Exception("Found a graph with greedy coloring > 4!")

        except FileNotFoundError:
            pass

    print(f"Maximum chromatic number found: {_max}")
    print(f"Maximum greedy coloring found: {max_greedy}")
    print(f"Maximum networkx greedy coloring found: {max_nx}")

    # G = nx.complete_multipartite_graph(2,2,2)
    # G.add_edge(0,1)
    # print(color_witnesses(Graph(G)))
