import unittest
import networkx as nx
from Graphs import PercolationGraph
from main import print_graph


class TestPercolationGraph(unittest.TestCase):

    def test_graph_inheritance(self):
        G = PercolationGraph()
        G.add_edges_from([(0, 1), (1, 2)])
        self.assertTrue(isinstance(G, nx.Graph))
        self.assertEqual(len(G.nodes), 3)

    def test_k222_minus_detection_true(self):
        # Create a K_{2,2,2} and remove one inter-part edge
        G = nx.complete_multipartite_graph(2, 2, 2)
        G.remove_edge(0, 2)  # valid edge between part 1 and part 2
        PG = PercolationGraph()
        PG.add_edges_from(G.edges())
        nodes = list(PG.nodes())
        self.assertTrue(PG.is_k222_minus_subgraph((0, 2), nodes))

    def test_k222_minus_detection_false(self):
        G = nx.complete_multipartite_graph(2, 2, 2)
        # G.remove_edge(0, 1)
        # G.remove_edge(2, 3)
        PG = PercolationGraph()
        PG.add_edges_from(G.edges())
        nodes = list(PG.nodes())
        self.assertFalse(PG.is_k222_minus_subgraph((0, 1), nodes))  # Two missing edges

    def test_k222_percolating_true(self):
        G = nx.complete_multipartite_graph(2, 2, 2)
        PG = PercolationGraph(G)
        PG.add_edge(0,1)
        self.assertTrue(PG.is_k222_percolating())

    def test_k222_percolating_false(self):
        PG = PercolationGraph()
        PG.add_edges_from([(0, 1), (1, 2), (2, 3)])  # A sparse path
        self.assertFalse(PG.is_k222_percolating())

    def test_is_k5_percolating(self):
        # Test 1: K5 minus one edge – should percolate
        G1 = nx.complete_graph(5)
        G1.remove_edge(0, 1)
        G1 = PercolationGraph(G1)
        assert G1.is_k5_percolating() == True, "Test 1 Failed: K5^- should percolate"

        # Test 2: Disconnected graph – should not percolate
        G2 = nx.Graph()
        G2.add_nodes_from(range(5))
        G2 = PercolationGraph(G2)
        assert G2.is_k5_percolating() == False, "Test 2 Failed: Empty graph shouldn't percolate"

        # Test 3: K5 – already complete
        G3 = nx.complete_graph(5)
        G3 = PercolationGraph(G3)
        assert G3.is_k5_percolating() == True, "Test 3 Failed: K5 is trivially percolating"

        G4 = nx.complete_graph(5)
        G4.remove_edge(0, 1)
        G4.add_node(5)
        G4.add_edge(0, 5)
        G4.add_edge(1, 5)
        G4.add_edge(2, 5)
        G4.add_edge(3, 5)
        G4 = PercolationGraph(G4)

        assert G4.is_k5_percolating(), "Test 4 Failed: G4 is percolating"

        print("Basic unit tests passed ✅")


if __name__ == '__main__':
    unittest.main()
