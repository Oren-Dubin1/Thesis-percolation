import networkx as nx
import itertools


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

    def is_k5_percolating(self):
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

        return G.number_of_edges() == n * (n - 1) // 2

    def print_graph(self, file=None):
        for i in range(self.number_of_nodes()):
            print(i, file=file)
        for e in self.edges():
            print(*e, file=file)
