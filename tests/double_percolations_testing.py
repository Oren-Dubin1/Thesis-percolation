from double_percolations import DoublePercolation
import networkx as nx
import unittest
import numpy as np

class TestDoublePercolation(unittest.TestCase):

    def test_double_percolation_true(self):
        # Create a graph that is known to be double percolating
        G = nx.complete_graph(6)
        G.remove_edge(0, 1)  # Remove one edge to allow K5^- percolation
        DP = DoublePercolation(G)
        self.assertTrue(DP.is_double_percolating(), "Test Failed: Graph should be double percolating")

    def test_double_percolation_false(self):
        # Create a sparse graph that should not be double percolating
        G = nx.path_graph(6)  # A simple path graph
        DP = DoublePercolation(G)
        self.assertFalse(DP.is_double_percolating(), "Test Failed: Sparse graph shouldn't be double percolating")


if __name__ == '__main__':
        unittest.main()