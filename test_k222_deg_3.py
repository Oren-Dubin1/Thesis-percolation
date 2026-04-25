import unittest
import networkx as nx
from k222_deg_3 import *


class TestK222WithDegree3Vertices(unittest.TestCase):
    def setUp(self):
        self.builder = K222WithDegree3Vertices()

    def test_base_graph_is_k222(self):
        G = self.builder.graph()

        self.assertEqual(set(G.nodes()), set(range(6)))
        self.assertEqual(G.number_of_edges(), 12)

        A = [0, 1]
        B = [2, 3]
        C = [4, 5]

        # No edges inside parts
        self.assertFalse(G.has_edge(A[0], A[1]))
        self.assertFalse(G.has_edge(B[0], B[1]))
        self.assertFalse(G.has_edge(C[0], C[1]))

        # All edges between different parts
        for u in A:
            for v in B:
                self.assertTrue(G.has_edge(u, v))
        for u in A:
            for v in C:
                self.assertTrue(G.has_edge(u, v))
        for u in B:
            for v in C:
                self.assertTrue(G.has_edge(u, v))

        degrees = dict(G.degree())
        for v in range(6):
            self.assertEqual(degrees[v], 4)

    def test_add_degree3_vertex(self):
        new_v = self.builder.add_degree3_vertex([0, 2, 4])
        G = self.builder.graph()

        self.assertIn(new_v, G.nodes())
        self.assertEqual(new_v, 6)

        self.assertTrue(G.has_edge(new_v, 0))
        self.assertTrue(G.has_edge(new_v, 2))
        self.assertTrue(G.has_edge(new_v, 4))
        self.assertEqual(G.degree(new_v), 3)

        self.assertEqual(G.number_of_nodes(), 7)
        self.assertEqual(G.number_of_edges(), 15)


    def test_add_degree3_vertex_requires_exactly_three_neighbors(self):
        with self.assertRaises(ValueError):
            self.builder.add_degree3_vertex([0, 2])

        with self.assertRaises(ValueError):
            self.builder.add_degree3_vertex([0, 2, 4, 5])

    def test_add_degree3_vertex_requires_distinct_neighbors(self):
        with self.assertRaises(ValueError):
            self.builder.add_degree3_vertex([0, 0, 2])

    def test_add_degree3_vertex_requires_existing_neighbors(self):
        with self.assertRaises(ValueError):
            self.builder.add_degree3_vertex([0, 2, 10])

    def test_add_vertices_by_rule(self):
        added = self.builder.add_vertices_by_rule(3, rule="ACC")
        G = self.builder.graph()

        self.assertEqual(added, [6, 7, 8])
        for v in added:
            self.assertEqual(G.degree(v), 3)
            self.assertTrue(G.has_edge(v, 0) or G.has_edge(v, 1))
            self.assertTrue(G.has_edge(v, 4) or G.has_edge(v, 5))

    def test_add_vertices_by_rule_invalid_rule(self):
        with self.assertRaises(ValueError):
            self.builder.add_vertices_by_rule(2, rule="XYZ")

    def test_copy_returns_independent_graph(self):
        self.builder.add_degree3_vertex([0, 2, 4], "ABC")

        G_copy = self.builder.copy()
        self.assertIsInstance(G_copy, nx.Graph)

        G_copy.add_node(100)

        original = self.builder.graph()
        self.assertNotIn(100, original.nodes())
        self.assertIn(100, G_copy.nodes())

    def test_add_vertices_by_rule_AAB(self):
        """Test AAB rule: 2 from A, 1 from B"""
        added = self.builder.add_vertices_by_rule(2, rule="AAB")
        G = self.builder.graph()

        self.assertEqual(added, [6, 7])
        for v in added:
            self.assertEqual(G.degree(v), 3)
            self.assertTrue(G.has_edge(v, 0))
            self.assertTrue(G.has_edge(v, 1))
            self.assertTrue(G.has_edge(v, 2) or G.has_edge(v, 3))

    def test_add_vertices_by_rule_AAC(self):
        """Test AAC rule: 2 from A, 1 from C"""
        added = self.builder.add_vertices_by_rule(2, rule="AAC")
        G = self.builder.graph()

        self.assertEqual(added, [6, 7])
        for v in added:
            self.assertEqual(G.degree(v), 3)
            self.assertTrue(G.has_edge(v, 0))
            self.assertTrue(G.has_edge(v, 1))
            self.assertTrue(G.has_edge(v, 4) or G.has_edge(v, 5))

    def test_add_vertices_by_rule_ABB(self):
        """Test ABB rule: 1 from A, 2 from B"""
        added = self.builder.add_vertices_by_rule(2, rule="ABB")
        G = self.builder.graph()

        self.assertEqual(added, [6, 7])
        for v in added:
            self.assertEqual(G.degree(v), 3)
            self.assertTrue(G.has_edge(v, 0) or G.has_edge(v, 1))
            self.assertTrue(G.has_edge(v, 2))
            self.assertTrue(G.has_edge(v, 3))

    def test_add_vertices_by_rule_ACC(self):
        """Test ACC rule: 1 from A, 2 from C"""
        added = self.builder.add_vertices_by_rule(2, rule="ACC")
        G = self.builder.graph()

        self.assertEqual(added, [6, 7])
        for v in added:
            self.assertEqual(G.degree(v), 3)
            self.assertTrue(G.has_edge(v, 0) or G.has_edge(v, 1))
            self.assertTrue(G.has_edge(v, 4))
            self.assertTrue(G.has_edge(v, 5))

    def test_add_vertices_by_rule_BBC(self):
        """Test BBC rule: 2 from B, 1 from C"""
        added = self.builder.add_vertices_by_rule(2, rule="BBC")
        G = self.builder.graph()

        self.assertEqual(added, [6, 7])
        for v in added:
            self.assertEqual(G.degree(v), 3)
            self.assertTrue(G.has_edge(v, 2))
            self.assertTrue(G.has_edge(v, 3))
            self.assertTrue(G.has_edge(v, 4) or G.has_edge(v, 5))

    def test_add_vertices_by_rule_BCC(self):
        """Test BCC rule: 1 from B, 2 from C"""
        added = self.builder.add_vertices_by_rule(2, rule="BCC")
        G = self.builder.graph()

        self.assertEqual(added, [6, 7])
        for v in added:
            self.assertEqual(G.degree(v), 3)
            self.assertTrue(G.has_edge(v, 2) or G.has_edge(v, 3))
            self.assertTrue(G.has_edge(v, 4))
            self.assertTrue(G.has_edge(v, 5))

    def test_add_vertices_by_rules_single_batch(self):
        result = self.builder.add_vertices_by_rules(((3, "AAB"),))

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

        G = self.builder.graph()
        self.assertEqual(G.number_of_nodes(), 9)    # 6 base + 3 new
        self.assertEqual(G.number_of_edges(), 21)   # 12 base + 9 new

        for v in result:
            self.assertIsInstance(v, int)
            self.assertIn(v, G.nodes())
            self.assertEqual(G.degree(v), 3)

    def test_add_vertices_by_rules_multiple_batches(self):
        result = self.builder.add_vertices_by_rules((
            (2, "AAB"),
            (3, "AAC"),
            (1, "BCC"),
        ))

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 6)  # 2 + 3 + 1

        G = self.builder.graph()
        self.assertEqual(G.number_of_nodes(), 12)   # 6 base + 6 new
        self.assertEqual(G.number_of_edges(), 30)    # 12 base + 18 new

        self.assertEqual(len(set(result)), 6)  # all vertex labels should be distinct

        for v in result:
            self.assertIsInstance(v, int)
            self.assertIn(v, G.nodes())
            self.assertEqual(G.degree(v), 3)

    def test_add_vertices_by_rules_empty_input(self):
        result = self.builder.add_vertices_by_rules(())

        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

        G = self.builder.graph()
        self.assertEqual(G.number_of_nodes(), 6)
        self.assertEqual(G.number_of_edges(), 12)

    def test_add_vertices_by_rules_zero_vertices(self):
        result = self.builder.add_vertices_by_rules(((0, "AAB"),))

        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

        G = self.builder.graph()
        self.assertEqual(G.number_of_nodes(), 6)
        self.assertEqual(G.number_of_edges(), 12)

    def test_add_vertices_by_rules_all_rules(self):
        result = self.builder.add_vertices_by_rules((
            (1, "AAB"),
            (1, "AAC"),
            (1, "ABB"),
            (1, "ACC"),
            (1, "BBC"),
            (1, "BCC"),
        ))

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 6)

        G = self.builder.graph()
        self.assertEqual(G.number_of_nodes(), 12)   # 6 base + 6 new
        self.assertEqual(G.number_of_edges(), 30)    # 12 base + 18 new

        for v in result:
            self.assertIn(v, G.nodes())
            self.assertEqual(G.degree(v), 3)

    def test_add_vertices_by_rules_preserves_order(self):
        result = self.builder.add_vertices_by_rules((
            (3, "AAB"),
            (4, "AAC"),
        ))

        self.assertEqual(result, list(range(6, 13)))

    def test_add_vertices_by_all_rules_zero(self):
        result = self.builder.add_vertices_by_all_rules(0)

        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

        G = self.builder.graph()
        self.assertEqual(G.number_of_nodes(), 6)
        self.assertEqual(G.number_of_edges(), 12)

    def test_add_vertices_by_all_rules_one_each(self):
        result = self.builder.add_vertices_by_all_rules(1)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 6)

        G = self.builder.graph()
        self.assertEqual(G.number_of_nodes(), 12)   # 6 base + 6 new
        self.assertEqual(G.number_of_edges(), 30)    # 12 base + 18 new

        self.assertEqual(result, list(range(6, 12)))

        for v in result:
            self.assertIn(v, G.nodes())
            self.assertEqual(G.degree(v), 3)

    def test_add_vertices_by_all_rules_multiple_each(self):
        result = self.builder.add_vertices_by_all_rules(2)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 12)

        G = self.builder.graph()
        self.assertEqual(G.number_of_nodes(), 18)   # 6 base + 12 new
        self.assertEqual(G.number_of_edges(), 48)   # 12 base + 36 new

        self.assertEqual(result, list(range(6, 18)))

        self.assertEqual(len(set(result)), 12)
        for v in result:
            self.assertIn(v, G.nodes())
            self.assertEqual(G.degree(v), 3)

    def test_add_vertices_by_all_rules_respects_all_rules(self):
        result = self.builder.add_vertices_by_all_rules(1)
        G = self.builder.graph()

        # Each returned vertex should be connected to exactly 3 base vertices
        for v in result:
            base_neighbors = [u for u in G.neighbors(v) if u < 6]
            self.assertEqual(len(base_neighbors), 3)


if __name__ == "__main__":
    unittest.main()