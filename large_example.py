import itertools
import unittest

import networkx as nx
import numpy as np

import main
from Graphs import PercolationGraph
from main import print_graph, is_k222_percolating
import Graphs


class CreateLargeGraph:
    def __init__(self, n: int, init_graph: Graphs.PercolationGraph):
        self.n = n
        self.graph = PercolationGraph(init_graph)  # Must be K222 percolating


    def enlarge(self):
        while self.graph.number_of_nodes() < self.n:
            print(f'current number of nodes={self.graph.number_of_nodes()}')
            # Choose a removable edge
            edges = list(self.graph.edges())
            u, v = edges[np.random.randint(len(edges))]

            self.graph.remove_edge(u, v)

            # Try to find two other nodes a, b such that adding w to u,v,a,b creates a K_{2,2,2}^-
            candidate_nodes = set(self.graph.nodes()) - {u, v}
            found = False

            for a, b in itertools.combinations(candidate_nodes, 2):
                # Add temporary node w with edges to u, v, a, b
                w = max(self.graph.nodes()) + 1
                self.graph.add_node(w)
                self.graph.add_edges_from([(w, u), (w, v), (w, a), (w, b)])

                if self.graph.is_k222_percolating():
                    found = True
                    break  # Success, graph is now enlarged

                # Otherwise, undo and try next
                self.graph.remove_node(w)

            if not found:
                # If no valid (a, b) worked, restore the edge and try another iteration
                self.graph.add_edge(u, v)


    def check_conjecture_3n_6(self, edge):
        if self.graph.number_of_edges() > 3 * self.graph.number_of_nodes() - 6:
            return

        answer = self.graph.is_k_222_percolating_without_edge(edge)
        if answer:
            self.graph.remove_edge(*edge)
            self.graph.print_graph()
            raise "Found percolating graph on < 3n-6 edges."


    def find_k222_without_two_edges(self, edge):
        """
        Find a 6-node set S such that:
          – S contains both endpoints of `edge`
          – the induced subgraph on S is a K_{2,2,2} minus exactly TWO cross edges,
                one of which must be `edge`.
        Returns:
            (S, missing_edges_set) or None
        """

        u, v = tuple(sorted(edge))
        nodes = list(self.graph.nodes())

        # Only 6-sets containing u and v
        remaining = [x for x in nodes if x not in (u, v)]

        for subset4 in itertools.combinations(remaining, 4):
            S = {u, v, *subset4}
            S_list = list(S)

            # Enumerate all 2,2,2 partitions of six nodes
            for A in itertools.combinations(S_list, 2):
                rem1 = [x for x in S_list if x not in A]
                for B in itertools.combinations(rem1, 2):
                    C = [x for x in rem1 if x not in B]

                    # Compute the 12 cross edges once
                    cross_edges = (
                            set(itertools.product(A, B)) |
                            set(itertools.product(A, C)) |
                            set(itertools.product(B, C))
                    )
                    cross_edges = {tuple(sorted(e)) for e in cross_edges}

                    # which cross edges are missing?
                    missing = {e for e in cross_edges if not self.graph.has_edge(*e)}

                    # We need exactly two missing cross edges
                    if len(missing) != 2:
                        continue

                    # One of them must be the tracked edge
                    if (u, v) not in missing:
                        continue

                    missing.remove(edge)
                    return S, missing

        return None

    @staticmethod
    def get_opposite(subgraph, node, edge_to_add=None):
        assert subgraph.number_of_nodes() == 6

        comp = nx.complement(subgraph)
        if edge_to_add: comp.remove_edge(*edge_to_add)
        return next(comp.neighbors(node))


    def decide_vertices_to_connect(self, edge):
        # By assumption - G-e does not percolate. G does percolate
        assert edge in self.graph.edges()

        u,v = edge
        S, f = self.find_k222_without_two_edges(edge)
        S = list(S)
        subgraph = self.graph.subgraph(S)
        # First case in the proof - (u,v)
        if subgraph.degree[u] == 4 and subgraph.degree[v] == 4:
            pass


    def smart_enlarge(self):
        while self.graph.number_of_nodes() < self.n:
            print(f'current number of nodes={self.graph.number_of_nodes()}')
            edge_to_remove = np.random.choice(self.graph.edges)
            self.check_conjecture_3n_6(edge_to_remove)  # Along the way check

            vertices_to_connect = self.decide_vertices_to_connect(edge_to_remove)
            new_node = self.graph.number_of_nodes()
            self.graph.add_node(new_node)
            self.graph.add_edges_from([(new_node, vertex) for vertex in vertices_to_connect])
            assert self.graph.is_k222_percolating()  # Only for testing - remove after

