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

    def decide_vertices_to_connect(self, edge):
        assert 0 <= edge <= self.graph.number_of_edges()

        # check if G - e percolates
        G = PercolationGraph(self.graph).copy()
        G.remove_edge(*edge)
        if is_k222_percolating(G):
            return G.nodes[0], G.nodes[1], G.nodes[2]








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


class TestCreate(unittest.TestCase):
    def test_enlarge(self):
        G = PercolationGraph(nx.complete_multipartite_graph(2, 2, 2))
        G.remove_edge(0, 2)
        G.add_edge(0, 1)
        assert is_k222_percolating(G), "Initial Graph not percolating"

        creator = CreateLargeGraph(10, G)
        creator.enlarge()
        assert is_k222_percolating(creator.graph), "Final Graph not percolating"
        print_graph(creator.graph)

    def test_decide_vertices_to_connect(self):
        G = PercolationGraph(nx.complete_multipartite_graph(2, 2, 2))
        G.add_edge(0, 1)
        G.add_edge(2, 3)
        print(G)


if __name__ == '__main__':
    unittest.main()
