

import unittest
import networkx as nx
from unittest.mock import patch

from k222_k5_percolation_with_v_x import (
    remove_vertex_relabel,
    all_pair_partitions_6,
    k5_witnesses,
    k222_witnesses,
    addable_edges_with_witnesses,
    witness_uses_pair_vx,
    edge_addable_without_v_one_step,
    brute_force_sequences,
    percolate_without_v,
    witnesses_for_edge,
    check_conj_all_graphs, find_vx_witness_for_ij,

)


def make_complete_graph(nodes):
    if type(nodes) is int:
        nodes = list(range(1, nodes+1))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            G.add_edge(u, v)
    return G

class DummyWitness:
    def __init__(self, vertices, parts=None, rule="K222"):
        self.vertices = tuple(vertices)
        self.parts = parts
        self.rule = rule

    def __repr__(self):
        return f"DummyWitness(vertices={self.vertices}, parts={self.parts}, rule={self.rule})"


class TestPercolationBruteforce(unittest.TestCase):
    def test_remove_vertex_relabel_basic(self):
        # Graph: 1-2, 2-3, 3-4
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3, 4])
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])
        H, mp = remove_vertex_relabel(G, 2)

        # Remaining nodes: [1,3,4] -> relabeled to [1,2,3]
        self.assertEqual(mp, {1: 1, 3: 2, 4: 3})

        self.assertEqual(set(H.nodes()), {1, 2, 3})
        self.assertEqual(set(map(tuple, H.edges())), {(2, 3)})

    def test_all_pair_partitions_6_count_unique(self):
        V6 = (1, 2, 3, 4, 5, 6)
        parts = list(all_pair_partitions_6(V6))

        # Number of perfect matchings of K6 is 15
        self.assertEqual(len(parts), 15)
        self.assertEqual(len(set(parts)), 15)

        for P in parts:
            self.assertEqual(len(P), 3)
            used = []
            for pair in P:
                self.assertEqual(len(pair), 2)
                used.extend(pair)
            self.assertEqual(sorted(used), list(V6))

    def test_k5_witness_exists_in_K5_minus(self):
        # Build K5 on {1,2,3,4,5} missing edge (1,2)
        G = make_complete_graph(5)
        G.remove_edge(1, 2)

        wits = k5_witnesses(G, (1, 2))
        self.assertTrue(len(wits) >= 1)
        self.assertTrue(any(w.rule == "K5" and set(w.vertices) == {1, 2, 3, 4, 5} for w in wits))

    def test_k5_witness_absent_if_two_edges_missing(self):
        G = make_complete_graph(5)
        G.remove_edge(1, 2)
        G.remove_edge(3, 4)

        wits = k5_witnesses(G, (1, 2))
        self.assertEqual(wits, [])

    def test_witness_for_edge(self):
        G = make_complete_graph(5)
        G.remove_edge(1, 2)
        G.add_node(6)
        G.add_edges_from([(6,i) for i in range(1,6)])  # connect 5 to all others

        wits = witnesses_for_edge(G, (1, 2))
        self.assertTrue(len(wits) >= 1)
        for w in wits:
            self.assertIn(1, w.vertices)
            self.assertIn(2, w.vertices)

    def test_k222_witness_exists_in_complete_tripartite(self):
        # Construct K_{2,2,2} on vertices 1..6 with parts (1,2),(3,4),(5,6), then remove (1,3).
        G = nx.Graph()
        G.add_nodes_from(range(1, 7))
        parts = [(1, 2), (3, 4), (5, 6)]

        for (a, b) in parts:
            for (c, d) in parts:
                if (a, b) == (c, d):
                    continue
                for u in (a, b):
                    for v in (c, d):
                        G.add_edge(u, v)

        G.remove_edge(1, 3)

        wits = k222_witnesses(G, (1, 3))
        self.assertEqual(G.number_of_edges(), 11)
        self.assertTrue(len(wits) >= 1)
        self.assertTrue(any(w.rule == "K222" and set(w.vertices) == {1, 2, 3, 4, 5, 6} for w in wits))

        # Verify 1 and 3 are in different parts for at least one witness on {1..6}
        ok_any = False
        for w in wits:
            if set(w.vertices) != {1, 2, 3, 4, 5, 6} or w.parts is None:
                continue
            part_of = {}
            for idx, pr in enumerate(w.parts):
                part_of[pr[0]] = idx
                part_of[pr[1]] = idx
            if part_of[1] != part_of[3]:
                ok_any = True
                break
        self.assertTrue(ok_any)

    def test_witness_uses_pair_vx(self):
        # In K6 missing (3,4), there should exist K222 witnesses pairing (1,2).
        G = make_complete_graph(6)
        G.remove_edge(3, 4)
        wits = k222_witnesses(G, (3, 4))
        self.assertTrue(len(wits) >= 1)
        self.assertTrue(any(witness_uses_pair_vx(w, 1, 2) for w in wits))

    def test_addable_edges_with_witnesses_finds_expected(self):
        G = make_complete_graph(5)
        G.remove_edge(1, 2)
        addable = addable_edges_with_witnesses(G)
        self.assertTrue((1, 2) in addable or (2, 1) in addable)
        e = (1, 2) if (1, 2) in addable else (2, 1)
        self.assertTrue(any(w.rule == "K5" for w in addable[e]))

    def test_edge_addable_without_v_one_step_true_for_K5_minus_in_G_minus_v(self):
        # Removing v=6 yields K5^- on 1..5 missing (1,2)
        G = make_complete_graph(5)
        G.remove_edge(1, 2)
        G.add_node(6)

        self.assertTrue(edge_addable_without_v_one_step(G, (1, 2), v=6))

    def test_edge_addable_without_v_one_step_false_when_G_minus_v_lacks_K5_minus(self):
        # Construct G with nodes 1..6 where nodes 1..5 are missing two edges
        # so G - v (with v=6) is not a K5^-; edge (1,2) should therefore not
        # be addable in one step without using v.
        G = make_complete_graph(5)
        G.remove_edge(1, 2)
        G.remove_edge(3, 4)
        G.add_node(6)

        self.assertFalse(edge_addable_without_v_one_step(G, (1, 2), v=6))


    def test_k5_step_adds_edge_when_v_not_in_witness(self):
        # K5^- on {1,2,3,4,5} missing (1,2); v=6 is forbidden but not in witness.
        G = make_complete_graph([1, 2, 3, 4, 5])
        G.remove_edge(1, 2)
        G.add_node(6)  # forbidden vertex not participating

        # sanity: witness exists in full graph for (1,2)
        self.assertTrue(len(k5_witnesses(G, (1, 2))) >= 1)

        H = percolate_without_v(G, v=6)
        self.assertTrue(H.has_edge(1, 2), "Expected (1,2) to be percolated without using v=6")

    def test_k5_step_blocked_when_forbidden_vertex_in_witness(self):
        # Same K5^- but now forbid v=3 which *must* appear in any K5 witness on {1..5}.
        G = make_complete_graph([1, 2, 3, 4, 5])
        G.remove_edge(1, 2)

        # sanity: witness exists for (1,2)
        self.assertTrue(len(k5_witnesses(G, (1, 2))) >= 1)

        H = percolate_without_v(G, v=3)
        self.assertFalse(H.has_edge(1, 2), "Expected (1,2) NOT to be added since any witness uses v=3")

    def test_k222_step_adds_missing_cross_edge_when_v_not_in_witness(self):
        # Build K222 on {0..5} with parts (0,1),(2,3),(4,5) and remove cross edge (0,2).
        G = nx.complete_multipartite_graph(2,2,2)
        G.remove_edge(0, 2)
        G.add_node(6)  # forbidden vertex not in witness

        # sanity: K222 witness exists for (0,2)
        self.assertTrue(len(k222_witnesses(G, (0, 2))) >= 1)

        H = percolate_without_v(G, v=6)
        self.assertTrue(H.has_edge(0, 2), "Expected (0,2) to be percolated without using v=6")

    def test_k222_step_blocked_when_forbidden_vertex_is_in_witness(self):
        # Same K222^- missing (0,2), but forbid v=0 (endpoint), so witness uses it => blocked.
        G = nx.complete_multipartite_graph(2,2,2)
        G.remove_edge(0, 2)

        # sanity: K222 witness exists for (0,2)
        self.assertTrue(len(k222_witnesses(G, (0, 2))) >= 1)

        H = percolate_without_v(G, v=0)
        self.assertFalse(H.has_edge(0, 2), "Expected (0,2) NOT to be added since any witness contains v=0")

    def test_edges_incident_to_v_are_never_added(self):
        # Since any witness for (v,u) must include v (as an endpoint), forbidding v blocks such edges.
        # Construct a graph where (6,1) would be addable by K5 if v were allowed:
        # Put {1,2,3,4,6} as K5^- missing (1,6), and forbid v=6.
        G = make_complete_graph([1, 2, 3, 4, 6])
        G.remove_edge(1, 6)
        G.add_node(5)  # extra node irrelevant

        # sanity: K5 witness exists for (1,6) in the full graph
        self.assertTrue(len(k5_witnesses(G, (1, 6))) >= 1)

        H = percolate_without_v(G, v=6)
        self.assertFalse(H.has_edge(1, 6), "Expected edges incident to forbidden v not to be added")

    def test_return_trace_witnesses_never_use_v(self):
        # Use a case where some edges get added, and ensure trace witnesses exclude v.
        G = make_complete_graph([1, 2, 3, 4, 5])
        G.remove_edge(1, 2)
        G.add_node(6)

        H, trace = percolate_without_v(G, v=6, return_trace=True)

        # At least (1,2) should be added
        self.assertTrue(H.has_edge(1, 2))
        self.assertTrue(len(trace) >= 1)

        for e, w in trace:
            self.assertNotIn(6, w.vertices, f"Witness for edge {e} illegally used v=6")

        # Also check that all traced edges are indeed present in H
        for e, _ in trace:
            self.assertTrue(H.has_edge(*e))


    def test_bruteforce_finds_event_in_constructed_instance(self):
        """
        Construct a graph where edge (3,6) is addable by a K222 witness using pair {v,x}={1,2},
        and simultaneously (3,6) is addable in G-v in one step (via K5^- on {2,3,4,5,6}).
        """
        v, x = 1, 2
        n = 6
        G0 = nx.Graph()
        G0.add_nodes_from(range(1, n + 1))

        # Make {2,3,4,5,6} complete minus (3,6)
        for u in [2, 3, 4, 5, 6]:
            for w in [2, 3, 4, 5, 6]:
                if u < w and (u, w) != (3, 6):
                    G0.add_edge(u, w)

        # Ensure cross edges for K222 with parts {1,2},{4,5},{3,6}, missing only (3,6)
        for u in [1, 2]:
            for w in [4, 5, 3, 6]:
                G0.add_edge(u, w)

        for u in [4, 5]:
            for w in [3, 6]:
                if (u, w) != (3, 6):
                    G0.add_edge(u, w)

        self.assertFalse(G0.has_edge(3, 6))

        events = brute_force_sequences(
            G0=G0,
            v=v,
            x=x,
            target_vertex=6,
            max_steps=1,      # should be addable immediately
            stop_after=3000,
        )

        self.assertTrue(any(ev.added_edge == (3, 6) and ev.witness.rule == "K222" for ev in events))
        self.assertTrue(any(ev.added_edge == (3, 6) and ev.could_add_without_v for ev in events))

    def test_detects_edge_ij_added_by_k222_with_vx_pair(self):
        # Build a K222^- on 6 vertices with parts {v,x}, {i,a}, {b,c}
        # Missing edge is (i,b), which is NOT incident to v or x.
        v, x = 0, 1
        i, a = 2, 3
        b, c = 4, 5

        G = nx.Graph()
        G.add_nodes_from(range(6))

        parts = [(v, x), (i, a), (b, c)]

        # Add all cross edges between different parts
        for p in range(3):
            for q in range(p + 1, 3):
                for u in parts[p]:
                    for w in parts[q]:
                        G.add_edge(u, w)

        # Remove one cross edge not incident to v or x
        # Choose (i,b) = (2,4)
        G.remove_edge(i, b)

        res = find_vx_witness_for_ij(G, v=v, x=x)
        self.assertIsNotNone(res)

        (edge, wit) = res
        self.assertEqual(set(edge), {i, b})  # should find the missing edge
        self.assertTrue(any(set(pair) == {v, x} for pair in wit.parts))

    def test_returns_none_when_no_k222_witness_pairs_vx(self):
        # Build a K222^- where v and x are NOT paired in the witness.
        # Parts: {v,i}, {x,a}, {b,c}. Missing edge: (i,b).
        v, x = 0, 1
        i, a = 2, 3
        b, c = 4, 5

        G = nx.Graph()
        G.add_nodes_from(range(6))

        parts = [(v, i), (x, a), (b, c)]

        for p in range(3):
            for q in range(p + 1, 3):
                for u in parts[p]:
                    for w in parts[q]:
                        G.add_edge(u, w)

        # Remove a cross edge not incident to v or x: (i,b) = (2,4)
        G.remove_edge(i, b)

        res = find_vx_witness_for_ij(G, v=v, x=x)
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
