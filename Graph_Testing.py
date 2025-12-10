import unittest
import time
import networkx as nx
from Graphs import PercolationGraph
from main import print_graph
import numpy as np



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
        assert G1.is_k5_percolating(), "Test 1 Failed: K5^- should percolate"

        # Test 2: Disconnected graph – should not percolate
        G2 = nx.Graph()
        G2.add_nodes_from(range(5))
        G2 = PercolationGraph(G2)
        assert not G2.is_k5_percolating(), "Test 2 Failed: Empty graph shouldn't percolate"

        # Test 3: K5 – already complete
        G3 = nx.complete_graph(5)
        G3 = PercolationGraph(G3)
        assert G3.is_k5_percolating(), "Test 3 Failed: K5 is trivially percolating"

        G4 = nx.complete_graph(5)
        G4.remove_edge(0, 1)
        G4.add_node(5)
        G4.add_edge(0, 5)
        G4.add_edge(1, 5)
        G4.add_edge(2, 5)
        G4.add_edge(3, 5)
        G4 = PercolationGraph(G4)

        assert G4.is_k5_percolating(), "Test 4 Failed: G4 is percolating"

        G5 = nx.complete_multipartite_graph(2, 2, 2)
        PG = PercolationGraph(G5)
        PG.add_edge(0, 1)
        answer = PG.is_k222_percolating((0, 1))
        assert PG.is_k5_percolating()
        assert answer, "Test 5 Failed: K_222^+ should percolate"

        G = nx.complete_multipartite_graph(2, 2, 2)
        G.add_edge(0,1)
        G.add_node(6)
        G.add_edge(0,6)
        G.add_edge(1,6)
        answer, graph = PercolationGraph(G).is_k5_percolating(return_final_graph=True)
        self.assertFalse(answer)
        self.assertEqual(graph.number_of_edges(), 17)
        self.assertTrue(graph.has_edge(6,1))
        self.assertFalse(graph.has_edge(6,2))

    def test_again_is_k222_perc(self):
        G = nx.Graph()
        G.add_nodes_from(range(10))
        G.add_edges_from([
            (0, 3), (0, 4), (0, 5), (0, 6), (0, 8), (0, 9),
    (1, 3), (1, 4), (1, 5), (1, 6), (1, 8), (1, 9),
    (2, 4), (2, 5), (2, 6), (2, 7),
    (3, 4), (3, 5), (3, 7), (3, 8),
    (4, 6), (4, 9),
    (7, 8), (7, 9),
    (10, 0), (10, 1), (10, 2), (10, 5), (10,4)
])
        PG = PercolationGraph(G)
        answer = PG.is_k222_percolating((0, 3))
        self.assertTrue(answer)


    def test_init(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        PG = PercolationGraph(G)
        self.assertEqual(list(PG.edges()), [(0,1),(1,2)])  # Output: [(0, 1), (1, 2)]

    def test_is_k_222_percolating_without_edge(self):
        G = nx.complete_multipartite_graph(2, 2, 2)
        G = PercolationGraph(G)
        G.add_edge(0, 1)
        G.add_node(6)
        G.add_edge(0, 6)
        G.add_edge(1, 6)
        G.add_edge(2, 6)
        answer, graph = G.is_k_222_percolating_without_edge((0,6), return_final_graph=True)
        self.assertFalse(answer)
        self.assertEqual(graph.number_of_edges(), 17)
        self.assertEqual(graph.number_of_nodes(), 7)

    # def test_is_k_222_percolating_large_example(self):
    #     G = nx.complete_graph(2)
    #     G = PercolationGraph(G)
    #     for edge in G.edges():
    #         if np.random.random() < 0.5:
    #             G.remove_edge(*edge)
    #
    #     print(G.is_k222_percolating())

    def test_find_k222_without_two_edges(self):
        G = PercolationGraph(nx.complete_multipartite_graph(2, 2, 2))
        G.remove_edge(0, 2)
        G.remove_edge(0,3)
        G.add_nodes_from([6,7])
        G.add_edges_from([(6,1),(6,2),(6,4), (7,6), (7,4), (7,2), (0,1)])

        S, edge = G.find_k222_without_two_edges((0,2))
        self.assertTrue(S.is_k222_minus_subgraph(edge, S.nodes))
        self.assertEqual(edge, (0,3))

        G = PercolationGraph(nx.complete_multipartite_graph(2, 2, 2))
        G.remove_edge(0, 2)
        G.remove_edge(0,3)
        S, edge = G.find_k222_without_two_edges((0,2))
        self.assertTrue(S.is_k222_minus_subgraph(edge, S.nodes))

        G = PercolationGraph(nx.Graph())
        G.add_edges_from([
            (0, 3), (0, 4), (0, 5), (0, 1), (0, 6),
            (1, 2), (1, 4), (1, 5), (1, 6), (1, 7),
            (2, 4), (2, 5), (2, 6),
            (3, 4), (3, 6)
        ])
        S, f = G.find_k222_without_two_edges((0,5))
        self.assertIsNone(S)

        G = PercolationGraph(nx.complete_graph(10))
        S, f = G.find_k222_without_two_edges((0,1))
        self.assertIsNone(f)
        self.assertIsNone(S)

    def test_find_k222_without_two_edges_non_induced(self):
        G = PercolationGraph(nx.complete_multipartite_graph(2, 2, 2))
        G.remove_edge(0, 2)
        G.remove_edge(0,3)
        G.add_nodes_from([6,7])
        G.add_edges_from([(6,1),(6,2),(6,4), (7,6), (7,4), (7,2), (0,1)])

        S, edge = G.find_k222_without_two_edges((0,2), False)
        self.assertTrue(S.is_k222_minus_subgraph(edge, S.nodes))
        self.assertEqual(edge, (0,3))

        G = PercolationGraph(nx.complete_multipartite_graph(2, 2, 2))
        G.remove_edge(0, 2)
        G.remove_edge(0,3)
        S, edge = G.find_k222_without_two_edges((0,2))
        self.assertTrue(S.is_k222_minus_subgraph(edge, S.nodes))

        G = PercolationGraph(nx.Graph())
        G.add_edges_from([
            (0, 3), (0, 4), (0, 5), (0, 1), (0, 6),
            (1, 2), (1, 4), (1, 5), (1, 6), (1, 7),
            (2, 4), (2, 5), (2, 6),
            (3, 4), (3, 6)
        ])
        S, f = G.find_k222_without_two_edges((0,5), False)
        self.assertIsNone(S)

        G = PercolationGraph(nx.complete_graph(10))
        G.remove_edge(0,1)
        S, f = G.find_k222_without_two_edges((0,1), False)
        self.assertEqual(f, (2,4))
        self.assertEqual(S.number_of_edges(), 11)


    def test_is_rigid(self):
        G = PercolationGraph(nx.complete_graph(5))
        self.assertTrue(G.is_rigid())

        G = PercolationGraph(nx.complete_multipartite_graph(2,2,2))
        self.assertTrue(G.is_rigid())

        G = PercolationGraph(nx.Graph())
        G.add_edges_from([(0,1),(1,2),(2,3),(0,3)])
        self.assertFalse(G.is_rigid())

        G = PercolationGraph(nx.complete_graph(200))
        G.add_edges_from([(200,0), (200,4)])
        self.assertFalse(G.is_rigid())
        G.add_edge(200,100)
        self.assertTrue(G.is_rigid())



if __name__ == '__main__':
    unittest.main()
