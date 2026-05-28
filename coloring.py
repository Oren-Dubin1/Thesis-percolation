import networkx as nx
from ortools.sat.python import cp_model
from utils import *

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

if __name__ == "__main__":
    _max = -1
    for n in range(7, 26):
        try:
            for g in iter_stored_graphs_for_n(n):
                chromatic_number, coloring = optimal_coloring(g)
                print(f"Graph with {g.number_of_edges()} edges has chromatic number {chromatic_number}.")
                _max = max(_max, chromatic_number)

        except FileNotFoundError:
            pass

    print(f"Maximum chromatic number found: {_max}")