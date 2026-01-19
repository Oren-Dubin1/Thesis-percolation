import networkx as nx
import itertools

import numpy as np


class PercolationGraph(nx.Graph):
    def __init__(self, base_graph=None, **kwargs):
        super().__init__(**kwargs)
        if base_graph is not None:
            self.update(base_graph)

    def is_k222_minus_subgraph(self, missing_edge, node_set):
        """Check whether the subgraph induced by node_set is a K_{2,2,2} minus the missing edge."""
        nodes = list(node_set)
        for A in itertools.combinations(nodes, 2):
            rest = [v for v in nodes if v not in A]
            for B in itertools.combinations(rest, 2):
                C = [v for v in rest if v not in B]
                # Generate all cross-part edges
                cross_edges = set(itertools.product(A, B)) | \
                              set(itertools.product(A, C)) | \
                              set(itertools.product(B, C))
                cross_edges = {tuple(sorted(e)) for e in cross_edges}

                if tuple(sorted(missing_edge)) not in cross_edges:
                    continue

                required_edges = cross_edges - {tuple(sorted(missing_edge))}
                if all(self.has_edge(*e) for e in required_edges):
                    return True
        return False

    @staticmethod
    def change_missing_edges(missing_edges, tracked_edge):
        # If tracking an edge, defer its addition to the end
        if tracked_edge is not None:
            tracked_edge_sorted = tuple(sorted(tracked_edge))
            if tracked_edge_sorted in missing_edges:
                missing_edges.remove(tracked_edge_sorted)
                missing_edges.append(tracked_edge_sorted)
        return missing_edges


    def is_k222_percolating(self, return_final_graph=False):
        if self.number_of_nodes() < 6: return False
        G = PercolationGraph(base_graph=self).copy()
        n = G.number_of_nodes()
        nodes = list(G.nodes())

        def all_edges(nodelist):
            return set(tuple(sorted(e)) for e in itertools.combinations(nodelist, 2))

        changed = True

        while changed:
            changed = False
            current_edges = set(tuple(sorted(e)) for e in G.edges())
            missing_edges = list(all_edges(nodes) - current_edges)

            for u, v in missing_edges:
                candidate = G.copy()
                candidate.add_edge(u, v)

                for S in itertools.combinations(nodes, 6):
                    if u not in S or v not in S:
                        continue
                    if PercolationGraph(candidate).is_k222_minus_subgraph((u, v), S):
                        G.add_edge(u, v)
                        changed = True
                        break

                if changed:
                    break

        if return_final_graph:
            return G.number_of_edges() == n * (n - 1) // 2,  G
        return G.number_of_edges() == n * (n - 1) // 2

    def is_k_222_percolating_without_edge(self, tracked_edge, return_final_graph=False):
        G = PercolationGraph(base_graph=self).copy()
        G.remove_edge(*tracked_edge)
        if return_final_graph:
            answer, graph = G.is_k222_percolating(return_final_graph=True)
            return answer, graph

        answer = G.is_k222_percolating(return_final_graph=False)
        return answer

    def is_k5_minus_subgraph(self, missing_edge, node_set):
        """Check if the subgraph induced by node_set is a K_5 minus missing_edge."""
        if len(node_set) < 5:
            return False
        for subset in itertools.combinations(node_set, 5):
            sub = self.subgraph(subset)
            if sub.number_of_edges() == 9 and not sub.has_edge(*missing_edge):
                return True
        return False

    def is_k5_percolating(self, return_final_graph=False):
        """Simulate K_5 percolation closure."""
        G = self.copy()
        nodes = list(G.nodes())
        n = len(nodes)

        all_possible_edges = {tuple(sorted(e)) for e in itertools.combinations(nodes, 2)}

        changed = True
        while changed:
            changed = False
            current_edges = {tuple(sorted(e)) for e in G.edges()}
            missing_edges = all_possible_edges - current_edges

            for u, v in missing_edges:
                for S in itertools.combinations(nodes, 5):
                    if u not in S or v not in S:
                        continue
                    if G.subgraph(S).number_of_edges() == 9 and not G.has_edge(u, v):
                        G.add_edge(u, v)
                        changed = True
                        break
                if changed:
                    break
        if return_final_graph:
            return G.number_of_edges() == n * (n - 1) // 2, G
        return G.number_of_edges() == n * (n - 1) // 2

    def print_graph(self, file=None):
        for node in self.nodes():
            print(node, file=file)
        for e in self.edges():
            print(*e, file=file)


    def find_k222_without_two_edges(self, edge, induced=True):
        """
        Find a 6-node set S such that:
          – S contains both endpoints of `edge`
          – the induced subgraph on S is a K_{2,2,2} minus exactly TWO cross edges,
                one of which MUST be `edge`.

        Return:
          A new graph H induced by S, with the tracked edge ADDED BACK
          and ONLY the other missing edge kept absent.

          So H is a K_{2,2,2}^- with *the other* missing edge.
        """
        u, v = tuple(sorted(edge))
        nodes = list(self.nodes())
        remaining = [x for x in nodes if x not in (u, v)]

        for subset4 in itertools.combinations(remaining, 4):
            S = {u, v, *subset4}
            S_list = list(S)

            for A in itertools.combinations(S_list, 2):
                rem1 = [x for x in S_list if x not in A]
                for B in itertools.combinations(rem1, 2):
                    C = [x for x in rem1 if x not in B]

                    # Build all 12 cross edges of K2,2,2
                    cross_edges = (
                            set(itertools.product(A, B)) |
                            set(itertools.product(A, C)) |
                            set(itertools.product(B, C))
                    )
                    cross_edges = {tuple(sorted(e)) for e in cross_edges}

                    # Which cross edges are missing in the original graph?
                    missing = {e for e in cross_edges
                               if not self.has_edge(*e)}

                    if induced and len(missing) != 2:
                        continue

                    if not induced and len(missing) > 2:
                        continue
                    # Must contain the tracked edge
                    if (u, v) not in missing:
                        continue

                    # Identify the OTHER missing cross-edge
                    try:
                        missing_other = [e for e in missing if e != (u, v)][0]
                    except IndexError:  # In case the graph is full
                        missing_other = [e for e in cross_edges if e != (u, v)][0]
                        if induced:
                            raise IndexError

                    # -------------------------
                    # BUILD CORRECTED SUBGRAPH
                    # -------------------------

                    H = PercolationGraph(nx.Graph())
                    H.add_nodes_from(S)

                    # Add ALL cross-edges EXCEPT `missing_other`
                    for e in cross_edges:
                        if e == missing_other:
                            continue  # keep this one absent
                        H.add_edge(*e)

                    # Now H is K2,2,2 minus ONLY missing_other
                    assert H.number_of_edges() == 11
                    return H, missing_other

        return None, None

    def is_rigid(self):
        n = self.number_of_nodes()
        m = self.number_of_edges()

        # Random generic positions in R^3
        P = np.random.random((n, 3))

        # Rigidity matrix: m rows (edges), 3n columns (coords)
        R = np.zeros((m, 3 * n))

        for row_idx, (u, v) in enumerate(self.edges()):
            # vector difference in R^3
            diff = P[u] - P[v]  # shape (3,)
            R[row_idx, 3 * u: 3 * u + 3] = diff
            R[row_idx, 3 * v: 3 * v + 3] = -diff

        # Rank test for rigidity:
        rank = np.linalg.matrix_rank(R)
        # maximal infinitesimal rigidity rank in R^3:
        return rank == 3 * n - 6

    def is_k222_graph_non_induced(self):
        """Return True if the graph is exactly a K_{2,2,2} (6 nodes, 12 cross edges)."""
        if self.number_of_nodes() != 6:
            return False

        nodes = list(self.nodes())
        # edges in the induced subgraph on these 6 nodes
        sub_edges = {tuple(sorted(e)) for e in self.subgraph(nodes).edges()}

        for A in itertools.combinations(nodes, 2):
            rest = [v for v in nodes if v not in A]
            for B in itertools.combinations(rest, 2):
                C = [v for v in rest if v not in B]

                cross_edges = (
                        set(itertools.product(A, B)) |
                        set(itertools.product(A, C)) |
                        set(itertools.product(B, C))
                )
                cross_edges = {tuple(sorted(e)) for e in cross_edges}

                if sub_edges >= cross_edges:
                    return True

        return False


if __name__ == "__main__":
    G = PercolationGraph(nx.complete_multipartite_graph(2, 2, 2))
    G.remove_edge(2,4)
    G.remove_edge(3,5)
    G.add_edge(4,5)
    G.add_edges_from([(1,2), (1,3), (1,4), (1,5), (1,6), (1,7)])
    G.add_edges_from([(6,2), (6,4), (7,6), (7,3)])
    G.print_graph()
    answer, graph = G.is_k222_percolating(return_final_graph=True)
    print("Is graph with v Percolating?", answer)
    print('Final graph has edge (7,5)???', graph.has_edge(7,5))
    G.remove_node(0)
    answer, graph = G.is_k222_percolating(return_final_graph=True)
    print("Is graph without v Percolating?", answer)
    print('Final graph has edge (7,5)???', graph.has_edge(7,5))





