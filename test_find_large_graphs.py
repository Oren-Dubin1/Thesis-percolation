# python
import unittest
from unittest.mock import patch, MagicMock
import networkx as nx

from Find_large_graphs import LargeGraphFinder


class TestLargeGraphFinder(unittest.TestCase):
    def test_init_defaults(self):
        finder = LargeGraphFinder()
        self.assertIsNone(finder.graph)
        self.assertIsNone(finder.helper_graph)

    def test_sample_gnp_sets_graph_and_is_reproducible(self):
        finder = LargeGraphFinder(n=5, p=0.5)
        g1 = finder.sample_gnp(seed=123)
        # call again with same seed -> reproducible
        g2 = finder.sample_gnp(seed=123)

        self.assertIsInstance(g1, nx.Graph)
        self.assertEqual(len(g1.nodes()), 5)
        # adjacency sets should match for same seed
        self.assertEqual(set(map(frozenset, g1.edges())), set(map(frozenset, g2.edges())))

    def test_print_graph_calls_PercolationGraph_print(self):
        finder = LargeGraphFinder()
        g = nx.path_graph(3)
        finder.graph = g

        # patch the PercolationGraph class imported into the module
        with patch("Find_large_graphs.PercolationGraph") as MockPG:
            instance = MockPG.return_value
            instance.print_graph = MagicMock()
            finder.print_graph()
            instance.print_graph.assert_called_once()


    def test__build_helper_graph_creates_expected_weights(self):
        # Build graph so pair {0,1} vs {2,3} has three cross edges
        G = nx.Graph()
        G.add_nodes_from(range(4))
        G.add_edge(0, 2)
        G.add_edge(0, 3)
        G.add_edge(1, 2)
        finder = LargeGraphFinder(graph=G)
        H = finder._build_helper_graph()
        A = frozenset({0, 1})
        B = frozenset({2, 3})
        self.assertIn(A, H.nodes())
        self.assertIn(B, H.nodes())
        self.assertTrue(H.has_edge(A, B))
        self.assertEqual(H[A][B]['weight'], 3)

    def test_update_helper_for_original_edge_adds_edges_and_updates_weights(self):
        # start with 4 vertices and no original edges
        G = nx.Graph()
        G.add_nodes_from(range(4))
        finder = LargeGraphFinder(graph=G)
        H = finder._build_helper_graph()  # helper graph nodes exist, no helper-edges yet
        # add original edge (0,2)
        applied = finder.update_helper_for_original_edge(0, 2)
        # expect at least one helper-edge created: between {0,1} and {2,3}
        A = frozenset({0, 1})
        B = frozenset({2, 3})
        self.assertTrue(H.has_edge(A, B))
        self.assertEqual(H[A][B]['weight'], 1)
        self.assertGreaterEqual(applied, 1)

        # add another original edge that touches same helper-edge to increment its weight
        finder.update_helper_for_original_edge(1, 3)
        # weight between A and B should increase
        self.assertEqual(H[A][B]['weight'], 2)

    def test__is_k222_percolating_one_step_detects_triangle_and_returns_expected_format(self):
        finder = LargeGraphFinder()
        # create helper graph triangle of disjoint pairs with weights 3,4,4
        H = nx.Graph()
        A = frozenset({0, 1})
        B = frozenset({2, 3})
        C = frozenset({4, 5})
        H.add_nodes_from([A, B, C])
        H.add_edge(A, B, weight=3)
        H.add_edge(B, C, weight=4)
        H.add_edge(C, A, weight=4)

        finder.helper_graph = H
        result = finder._is_k222_percolating_one_step()
        # result should be (four_vertices, u0, u1, v0, v1, w0, w1)
        self.assertIsInstance(result, tuple)
        four_vertices, u0, u1, v0, v1, w0, w1 = result
        # four_vertices should be two sorted tuples corresponding to the weight-3 edge (A,B)
        self.assertEqual(four_vertices, (tuple(sorted(A)), tuple(sorted(B))))
        self.assertEqual((u0, u1), tuple(sorted(A)))
        self.assertEqual((v0, v1), tuple(sorted(B)))
        self.assertEqual((w0, w1), tuple(sorted(C)))

    def test_k222_percolating_true(self):
        G = nx.complete_multipartite_graph(2, 2, 2)
        G.add_edge(0, 1)
        F = LargeGraphFinder(graph=G)
        self.assertTrue(F.is_k222_percolating())

    def test_k222_percolating_false(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])  # A sparse path
        F = LargeGraphFinder(graph=G)
        self.assertFalse(F.is_k222_percolating())

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
            (10, 0), (10, 1), (10, 2), (10, 5), (10, 4)
        ])
        F = LargeGraphFinder(graph=G)
        answer = F.is_k222_percolating()
        self.assertTrue(answer)


if __name__ == "__main__":
    unittest.main()
