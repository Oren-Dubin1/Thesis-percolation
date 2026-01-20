import itertools
import networkx as nx
import random

import numpy as np

from Graphs import PercolationGraph
import os


def sample_3n_6(n, seed=None):
    """Return a random 3n - 6 graph (networkx Graph) with minimal degree >= 3."""
    if seed is not None:
        random.seed(seed)

    all_edges = list(itertools.combinations(range(n), 2))

    while True:
        selected_edges = random.sample(list(all_edges), 3 * n - 6)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(selected_edges)
        if all(deg >= 3 for _, deg in G.degree()):
            break

    graph = Graph(G)

    return graph

class Graph:
    def __init__(self, graph):
        self.graph = graph
        self.n = self.graph.number_of_nodes()
        self.helper_matrix = None
        self.index_map = None
        self.local_addition_matrix = None


        if self.graph is not None:
            self.build_helper_matrix()
            self.original_graph = graph.copy()

    def build_helper_matrix(self):
        """
                Build helper matrix:
                - nodes are frozenset pairs {i,j} (i<j)
                - edge weight between two nodes is number of edges between the two pairs in G
                (only for disjoint pairs)
                """
        assert self.graph is not None, "Graph must be set before building helper graph."
        H = np.zeros((self.n * (self.n - 1) // 2, self.n * (self.n - 1) // 2), dtype=int)
        nodes = [frozenset(pair) for pair in itertools.combinations(self.graph.nodes(), 2)]
        self.index_map = {node: idx for idx, node in enumerate(nodes)}

        for a, b in itertools.combinations(nodes, 2):
            if a & b:
                continue  # pairs are not disjoint
            # count edges between the two pairs (up to 4)
            weight = 0
            for u in a:
                for v in b:
                    if self.graph.has_edge(u, v):
                        weight += 1
            if weight > 0:
                idx_a = self.index_map[a]
                idx_b = self.index_map[b]
                H[idx_a][idx_b] = weight
                H[idx_b][idx_a] = weight

        self.helper_matrix = H

        return H

    def set_local_addition_matrix(self):
        # For each edge addition (u,v) in the original graph, determine which helper matrix entries are affected. Return a dict of (u,v) -> matrix which is zero everywhere except for the affected entries.
        assert self.graph is not None, "Graph must be set before building local addition matrices."
        L = {}
        nodes = [frozenset(pair) for pair in itertools.combinations(self.graph.nodes(), 2)]

        index_map = self.index_map or {node: idx for idx, node in enumerate(nodes)}

        for u, v in itertools.combinations(self.graph.nodes(), 2):
            local_matrix = np.zeros((self.n * (self.n - 1) // 2, self.n * (self.n - 1) // 2), dtype=int)
            for pair in nodes:
                for other_pair in nodes:
                    if pair & other_pair:
                        continue  # pairs are not disjoint
                    if u in pair and v in other_pair or v in pair and u in other_pair:
                        idx_a = index_map[pair]
                        idx_b = index_map[other_pair]
                        local_matrix[idx_a][idx_b] = 1
                        local_matrix[idx_b][idx_a] = 1

            L[(u, v)] = local_matrix
            L[(v, u)] = local_matrix  # undirected edge
        self.local_addition_matrix = L
        return L

    def is_percolating_one_step(self, return_vertices=True):
        # Check if there exists a triangle in the helper matrix with weights 3,4,4
        H = self.helper_matrix
        if H is None:
            H = self.build_helper_matrix()

        size = H.shape[0]
        for i in range(size):
            for j in range(i + 1, size):
                for k in range(j + 1, size):
                    w1 = H[i][j]
                    w2 = H[j][k]
                    w3 = H[k][i]
                    weights = sorted([w1, w2, w3])
                    if weights == [3, 4, 4]:
                        # Retrieve the corresponding vertex pairs

                        nodes = list(self.index_map.keys())
                        if w1 == 3:
                            A = nodes[i]
                            B = nodes[j]
                        elif w2 == 3:
                            A = nodes[j]
                            B = nodes[k]
                        else:
                            A = nodes[k]
                            B = nodes[i]

                        for u in A:
                            for v in B:
                                if not self.graph.has_edge(u, v):
                                    if return_vertices:
                                        return u,v

                        else:
                            return True
        return None

    def is_percolating(self):
        # Check if the graph is percolating by checking all possible edge additions
        L = self.local_addition_matrix
        if L is None:
            L = self.set_local_addition_matrix()
        H_original = self.helper_matrix
        if H_original is None:
            H_original = self.build_helper_matrix()

        while True:
            result = self.is_percolating_one_step()
            if result is None:
                break  # No more percolating configurations found

            u,v = result
            # Update helper matrix
            self.helper_matrix += L[(u, v)]
            self.graph.add_edge(u, v)

        percolated = self.graph.number_of_edges() == self.n * (self.n - 1) // 2
        # Restore original graph and helper matrix
        self.graph = self.original_graph.copy()
        self.helper_matrix = H_original.copy()
        return percolated





if __name__ == "__main__":
    G = sample_3n_6(10, seed=42)
    print("Graph edges:", G.graph.number_of_edges())
    print("Is percolating:", G.is_percolating())
