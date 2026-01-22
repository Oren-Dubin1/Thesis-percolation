# test percolation_improved using unittest
import unittest
import networkx as nx
import numpy as np

from percolation_improved import Graph

class TestPercolationImproved(unittest.TestCase):
    def test_build_helper_matrix_creates_expected_weights(self):
        # Build graph so pair {0,1} vs {2,3} has three cross edges
        G = nx.Graph()
        G.add_nodes_from(range(4))
        G.add_edge(0, 2)
        G.add_edge(0, 3)
        G.add_edge(1, 2)
        graph_obj = Graph(graph=G)
        H = graph_obj.build_helper_matrix()
        A = frozenset({0, 1})
        B = frozenset({2, 3})
        idx_A = graph_obj.index_map[A]
        idx_B = graph_obj.index_map[B]
        self.assertEqual(H[idx_A][idx_B], 3)

    def test_set_local_addition_matrix_updates_correct_entries(self):
        # start with 4 vertices and no original edges
        G = nx.Graph()
        G.add_nodes_from(range(4))
        graph_obj = Graph(graph=G)
        graph_obj.build_helper_matrix()  # helper matrix nodes exist
        graph_obj.set_local_addition_matrix()

        # add original edge (0,2)
        local_matrix = graph_obj.local_addition_matrix[(0, 2)]

        # Check that the correct entries are updated
        A = frozenset({0, 1})
        B = frozenset({2, 3})
        idx_A = graph_obj.index_map[A]
        idx_B = graph_obj.index_map[B]

        self.assertEqual(local_matrix[idx_A][idx_B], 1)
        self.assertEqual(local_matrix[idx_B][idx_A], 1)

    def test_is_percolating_one_step_detects_triangle(self):
        G = nx.complete_multipartite_graph(2,2,2)
        G.remove_edge(0,3)
        G.add_edge(0,1)
        graph_obj = Graph(graph=G)
        u,v = graph_obj.is_percolating_one_step()
        self.assertEqual((u,v), (0,3))

    def test_is_percolating(self):
        G = nx.complete_multipartite_graph(2, 2, 2)
        G.remove_edge(0, 3)
        graph_obj = Graph(graph=G)
        self.assertFalse(graph_obj.is_percolating())

        G.add_edge(0, 1)
        graph_obj = Graph(graph=G)
        self.assertTrue(graph_obj.is_percolating())

        graph_obj = Graph(graph=nx.complete_graph(20))
        self.assertTrue(graph_obj.is_percolating())

        PG = nx.Graph()
        PG.add_edges_from([(0, 1), (1, 2), (2, 3)])  # A sparse path
        graph_obj = Graph(graph=PG)
        self.assertFalse(graph_obj.is_percolating())

        G = nx.Graph()
        G.add_nodes_from(range(11))
        G.add_edges_from([
            (0, 3), (0, 4), (0, 5), (0, 6), (0, 8), (0, 9),
            (1, 3), (1, 4), (1, 5), (1, 6), (1, 8), (1, 9),
            (2, 4), (2, 5), (2, 6), (2, 7),
            (3, 4), (3, 5), (3, 7), (3, 8),
            (4, 6), (4, 9),
            (7, 8), (7, 9),
            (10, 0), (10, 1), (10, 2), (10, 5), (10, 4)
        ])
        PG = Graph(G)
        answer = PG.is_percolating()
        self.assertTrue(answer)

    def test_is_rigid(self):
        G = Graph(nx.complete_graph(5))
        self.assertTrue(G.is_rigid())

        G = Graph(nx.complete_multipartite_graph(2,2,2))
        self.assertTrue(G.is_rigid())

        G.graph.remove_edge(0,2)
        G.build_helper_matrix()
        self.assertFalse(G.is_rigid())

        G.graph.add_edge(0,1)
        G.build_helper_matrix()
        self.assertTrue(G.is_rigid())

        G = nx.Graph()
        G.add_edges_from([(0,1),(1,2),(2,3),(0,3)])
        graph = Graph(G)
        self.assertFalse(graph.is_rigid())

        G = nx.complete_graph(200)
        G.add_edges_from([(200,0), (200,4)])
        graph = Graph(G, build_helper=False)
        self.assertFalse(graph.is_rigid())
        graph.graph.add_edge(200,100)

        self.assertTrue(graph.is_rigid())

    def test_is_k5_percolating(self):
        # Test 1: K5 minus one edge – should percolate
        G1 = nx.complete_graph(5)
        G1.remove_edge(0, 1)
        G1 = Graph(G1)
        assert G1.is_k5_percolating(), "Test 1 Failed: K5^- should percolate"

        # Test 2: Disconnected graph – should not percolate
        G2 = nx.Graph()
        G2.add_nodes_from(range(5))
        G2 = Graph(G2)
        assert not G2.is_k5_percolating(), "Test 2 Failed: Empty graph shouldn't percolate"

        # Test 3: K5 – already complete
        G3 = nx.complete_graph(5)
        G3 = Graph(G3)
        assert G3.is_k5_percolating(), "Test 3 Failed: K5 is trivially percolating"

        G4 = nx.complete_graph(5)
        G4.remove_edge(0, 1)
        G4.add_node(5)
        G4.add_edge(0, 5)
        G4.add_edge(1, 5)
        G4.add_edge(2, 5)
        G4.add_edge(3, 5)
        G4 = Graph(G4)

        assert G4.is_k5_percolating(), "Test 4 Failed: G4 is percolating"

        G5 = nx.complete_multipartite_graph(2, 2, 2)
        G5.add_edge(0, 1)
        PG = Graph(G5)
        answer = PG.is_percolating()
        assert PG.is_k5_percolating()
        assert answer, "Test 5 Failed: K_222^+ should percolate"

        G = nx.complete_multipartite_graph(2, 2, 2)
        G.add_edge(0,1)
        G.add_node(6)
        G.add_edge(0,6)
        G.add_edge(1,6)
        answer, graph = Graph(G).is_k5_percolating(return_final_graph=True)
        self.assertFalse(answer)
        self.assertEqual(graph.number_of_edges(), 17)
        self.assertTrue(graph.has_edge(6,1))
        self.assertFalse(graph.has_edge(6,2))

    def test_is_k5_percolating_one_step(self):
            # K5 minus one edge: should detect and return missing edge
            G = nx.complete_graph(5)
            G.remove_edge(0, 1)
            graph_obj = Graph(graph=G)
            self.assertTrue(graph_obj.is_k5_percolating_one_step(return_edge=False))
            missing = graph_obj.is_k5_percolating_one_step(return_edge=True)
            self.assertIsInstance(missing, tuple)
            self.assertEqual(frozenset(missing), frozenset((0, 1)))

            # Complete K5: should not report a missing edge (edges == 10)
            G2 = nx.complete_graph(5)
            graph2 = Graph(graph=G2)
            self.assertFalse(graph2.is_k5_percolating_one_step(return_edge=False))
            self.assertFalse(graph2.is_k5_percolating_one_step(return_edge=True))

            # Graph with no 5-subset of 9 edges
            G3 = nx.path_graph(6)
            graph3 = Graph(graph=G3)
            self.assertFalse(graph3.is_k5_percolating_one_step(return_edge=False))
            self.assertFalse(graph3.is_k5_percolating_one_step(return_edge=True))

    def test_double_percolation_true(self):
        # Create a graph that is known to be double percolating
        G = nx.complete_graph(6)
        G.remove_edge(0, 1)  # Remove one edge to allow K5^- percolation
        DP = Graph(G)
        self.assertTrue(DP.is_double_percolating(), "Test Failed: Graph should be double percolating")

    def test_double_percolation_false(self):
        # Create a sparse graph that should not be double percolating
        G = nx.path_graph(6)  # A simple path graph
        DP = Graph(G)
        self.assertFalse(DP.is_double_percolating(), "Test Failed: Sparse graph shouldn't be double percolating")



if __name__ == '__main__':
    unittest.main()