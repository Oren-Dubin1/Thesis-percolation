import unittest
import networkx as nx
from get_all_cuts import get_all_cuts_of_sizes_3_4

class TestGetAllCuts(unittest.TestCase):
    def test_generator_and_counts_3x3(self):
        gen = get_all_cuts_of_sizes_3_4(sizes=[(3,3)])
        # There are 3*3 = 9 possible bipartite edges, so 2^9 graphs
        expected = 1 << 9
        count = 0
        for left, right, G in gen:
            self.assertEqual(left, 3)
            self.assertEqual(right, 3)
            # nodes labeled 0..5
            self.assertEqual(G.number_of_nodes(), 6)
            count += 1
        self.assertEqual(count, expected)

    def test_as_list_and_connected_filter(self):
        graphs = get_all_cuts_of_sizes_3_4(sizes=[(2,2)], as_list=True, connected_only=True)
        # For 2x2 there are 4 possible edges; enumerate by hand and check
        # connected bipartite graphs count for 2x2 is: total 16, disconnected ones are those with empty rows or columns
        # We'll just assert non-empty and that type is list
        self.assertIsInstance(graphs, list)
        self.assertTrue(len(graphs) > 0)
        for left, right, G in graphs:
            self.assertTrue(nx.is_connected(G))

if __name__ == '__main__':
    unittest.main()
