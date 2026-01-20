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
        graph_obj = Graph(graph=G, n=4)
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
        graph_obj = Graph(graph=G, n=4)
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
        graph_obj = Graph(graph=G, n=6)
        u,v = graph_obj.is_percolating_one_step()
        self.assertEqual((u,v), (0,3))

    def test_is_percolating(self):
        G = nx.complete_multipartite_graph(2, 2, 2)
        G.remove_edge(0, 3)
        graph_obj = Graph(graph=G, n=6)
        self.assertFalse(graph_obj.is_percolating())

        G.add_edge(0, 1)
        graph_obj = Graph(graph=G, n=6)
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



if __name__ == '__main__':
    unittest.main()