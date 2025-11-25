from large_example import CreateLargeGraph
import unittest

import networkx as nx
import numpy as np

import main
from Graphs import PercolationGraph
from main import print_graph, is_k222_percolating
import Graphs

class TestCreate(unittest.TestCase):
    def test_enlarge(self):
        G = PercolationGraph(nx.complete_multipartite_graph(2, 2, 2))
        G.remove_edge(0, 2)
        G.add_edge(0, 1)
        assert is_k222_percolating(G), "Initial Graph not percolating"

        creator = CreateLargeGraph(5, G)
        creator.enlarge()
        assert is_k222_percolating(creator.graph), "Final Graph not percolating"
        creator.graph.print_graph()

    def test_decide_vertices_to_connect(self):
        G = PercolationGraph(nx.complete_multipartite_graph(2, 2, 2))
        G.add_edge(0, 1)
        G.add_edge(2, 3)
        print(G)

    def test_find_k222_without_two_edges(self):
        G = PercolationGraph(nx.complete_multipartite_graph(2, 2, 2))
        G.remove_edge(0, 2)
        G.remove_edge(0,3)
        G.add_nodes_from([6,7])
        G.add_edges_from([(6,1),(6,2),(6,4), (7,6), (7,4), (7,2), (0,1)])

        creator = CreateLargeGraph(10, G)
        S, edge = creator.find_k222_without_two_edges((0,2))
        self.assertEqual(edge, {(0,3)})
        self.assertEqual(S, {0,1,2,3,4,5})

    def test_get_opposite(self):
        G = PercolationGraph(nx.complete_multipartite_graph(2, 2, 2))
        G.remove_edge(0, 2)
        self.assertEqual(CreateLargeGraph.get_opposite(G, 3, (0,2)), 2)

if __name__ == '__main__':
    unittest.main()

