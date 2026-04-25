import networkx as nx

from VXchecker import VXChecker
import unittest
from Graphs import PercolationGraph

class VXcheckerTesting(unittest.TestCase):

    def test_check_edge_can_be_added(self):
        G = PercolationGraph(nx.complete_multipartite_graph(2, 2, 2))
        self.assertFalse(VXChecker(graph=G, v=0, x=1).check_edge_can_be_added())

        answer, vertex = VXChecker(graph=G, v=0, x=1).check_edge_can_be_added(return_vertex=True)
        self.assertFalse(answer)
        self.assertIsNone(vertex)

        G.remove_edge(2,5)
        checker = VXChecker(graph=G, v=0, x=1)
        answer, vertex = checker.check_edge_can_be_added(return_vertex=True)
        self.assertTrue(answer)
        self.assertEqual(vertex, 2)
        G.add_edge(2,5)

        G.remove_edge(2,4)
        checker = VXChecker(graph=G, v=0, x=1)
        self.assertFalse(checker.check_edge_can_be_added())

        G = PercolationGraph(nx.complete_graph(10))
        checker = VXChecker(graph=G, v=0, x=1)
        self.assertFalse(checker.check_edge_can_be_added())

        G = PercolationGraph(nx.complete_graph(6))
        checker = VXChecker(graph=G, v=0, x=1)
        self.assertFalse(checker.check_edge_can_be_added())


    def test_enumerate_possible_edges(self):
        G = PercolationGraph(nx.complete_multipartite_graph(2,2,2))
        checker = VXChecker(graph=G, v=0, x=1)
        checker.enumerate_possible_edges()

if __name__ == '__main__':
    unittest.main()