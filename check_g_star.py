import itertools
import math
import networkx as nx


def compute_e_values(F: nx.Graph) -> dict[int, int]:
    """
    e_i = min_{S subset V(F), |S|=i} |E(F) \\ E(F-S)| - 1.

    Equivalently: minimum number of edges of F incident to at least
    one vertex of S, minus 1.
    """
    vertices = list(F.nodes())
    v = len(vertices)

    e = {0: 0}

    for i in range(1, v):
        best = math.inf

        for S_tuple in itertools.combinations(vertices, i):
            S = set(S_tuple)

            incident_edges = 0
            for u, w in F.edges():
                if u in S or w in S:
                    incident_edges += 1

            best = min(best, incident_edges - 1)

        e[i] = best

    return e


def compute_g_star(F: nx.Graph, n_max: int | None = None, print_values: bool = True) -> list[int]:
    """
    Computes g^*(0), ..., g^*(n_max), where

        g^*(0) = 0
        g^*(n) = min_{1 <= j <= v-1} g^*(n-j) + e_j.

    If n_max is None, computes up to 3v.
    """
    v = F.number_of_nodes()

    if n_max is None:
        n_max = 3 * v

    e = compute_e_values(F)

    g = [math.inf] * (n_max + 1)
    g[0] = 0

    for n in range(1, n_max + 1):
        for j in range(1, min(v - 1, n) + 1):
            g[n] = min(g[n], g[n - j] + e[j])

    if print_values:
        print("e_i values:")
        for i in range(v):
            print(f"e_{i} = {e[i]}")

        print("\ng^*(i) values:")
        for i in range(n_max + 1):
            print(f"g^*({i}) = {g[i]}")

    return g

if __name__ == '__main__':
    """
    checks the function g^* in the paper https://arxiv.org/pdf/2305.11043
    for the octahedron.
    turns out to be g*(n) ~= 2.2n
    """
    F = nx.complete_multipartite_graph(2, 2, 2)

    g = compute_g_star(F, n_max=100000)
    for i, val in enumerate(g):
        if i == 0: continue
        print(f'g^*({i})/i = {val / i}')