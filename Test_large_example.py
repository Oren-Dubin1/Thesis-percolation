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
        G.remove_edge(0, 2)
        for edge in G.edges:
            G = PercolationGraph(nx.complete_multipartite_graph(2, 2, 2))
            G.remove_edge(0, 2)
            creator = CreateLargeGraph(5, G)
            vertices = creator.decide_vertices_to_connect(edge)
            G.add_edges_from([(6, a) for a in vertices])
            answer, graph = G.is_k222_percolating(return_final_graph=True)
            self.assertTrue(graph.has_edge(*edge))

            G.add_edge(0,1)
            self.assertTrue(G.is_k222_percolating())




    def test_get_opposite(self):
        G = PercolationGraph(nx.complete_multipartite_graph(2, 2, 2))
        G.remove_edge(0, 2)
        self.assertEqual(CreateLargeGraph.get_opposite(G, 3, (0,2)), 2)

    def test_smart_enlarge(self):
        G = PercolationGraph(nx.complete_multipartite_graph(2, 2, 2))
        G.remove_edge(0, 2)
        G.add_edge(0,1)

        creator = CreateLargeGraph(10, G)
        creator.smart_enlarge()
        assert creator.graph.is_k222_percolating()

if __name__ == '__main__':
    unittest.main()

