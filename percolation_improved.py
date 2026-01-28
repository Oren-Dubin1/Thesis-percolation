import itertools
import networkx as nx
import random

import numpy as np
from networkx.classes import non_edges

from Graphs import PercolationGraph
import os
import matplotlib.pyplot as plt
from double_percolations import DoublePercolation
import os



def read_graphs_from_edgelist(path):
    """Given a folder path, find all .edgelist files and return a list of LargeGraphFinder
    instances with each graph loaded. If `path` is a single .edgelist file, return a
    list containing a single finder.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    if os.path.isfile(path):
        files = [path]
    else:
        # list .edgelist files in directory, sorted for determinism
        files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith('.edgelist')]

    graphs = []
    for fpath in files:
        graphs.append(read_graph_from_edgelist(fpath))

    return graphs

def read_graph_from_edgelist(path):
    """Read a graph from an edgelist file and set it as self.graph."""
    graph = nx.read_edgelist(path, nodetype=int)
    G = Graph(graph)
    return G


def sample_3n_6(n, seed=None):
    """Construct a 3n-6 graph with minimal degree >= 3 using the shuffled-edge pass approach.

    Algorithm (per user):
    - Create a shuffled list of all possible edges.
    - Iterate once over that list and add each edge if it is incident to a vertex whose current
      degree is less than 3 (and the edge is not already present).
    - After the pass, if some vertices still have degree < 3, do targeted additions using remaining
      edges to satisfy degree constraints.
    - Finally, add arbitrary remaining edges from the shuffled list until exactly 3n-6 edges are present.
    """
    if n < 4:
        raise ValueError("n must be at least 4 to construct a 3n-6 graph with min degree >= 3")
    if seed is not None:
        random.seed(seed)

    nodes = list(range(n))
    all_edges = [tuple(e) for e in itertools.combinations(nodes, 2)]
    random.shuffle(all_edges)

    G = nx.Graph()
    G.add_nodes_from(nodes)

    degree = {u: 0 for u in nodes}
    edge_set = set()

    target_edges = 3 * n - 6

    # Single pass: add edge if it helps a vertex with degree < 3
    for (u, v) in all_edges:
        if len(edge_set) >= target_edges:
            break
        if (u, v) in edge_set or (v, u) in edge_set:
            continue
        if degree[u] < 3 or degree[v] < 3:
            G.add_edge(u, v)
            edge_set.add((u, v))
            degree[u] += 1
            degree[v] += 1

    # Final fill: add arbitrary remaining edges until we have target_edges
    if len(edge_set) < target_edges:
        for (u, v) in all_edges:
            if len(edge_set) >= target_edges:
                break
            if (u, v) in edge_set or (v, u) in edge_set:
                continue
            G.add_edge(u, v)
            edge_set.add((u, v))
            # update degree counts
            degree[u] += 1
            degree[v] += 1

    # Final validation
    if len(edge_set) != target_edges:
        return sample_3n_6(n, seed=seed + 1 if seed is not None else None)
    if any(d < 3 for d in degree.values()):
        return sample_3n_6(n, seed=seed + 1 if seed is not None else None)

    graph = Graph(G)
    return graph



def run_percolation_experiments(n=None,
                                seed=None,
                                output_dir='percolating graphs',
                                log_file='percolating_graphs.txt',
                                max_tries=1000,
                                double_percolation=False
                                ):
    """
    For each p (either explicit `p_values` or generated from start/stop/step), repeatedly sample G(n,p)
    until `is_k222_percolating()` returns True or `max_tries` is reached.
    Saves each successful graph as an edgelist in `output_dir` and appends a summary to `log_file`.
    """
    import os
    import time
    from datetime import datetime


    output_dir = output_dir + f'/n_{n}'

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_file)
    count_percolated = 0
    results = []
    for attempts in range(max_tries):
        attempt_seed = None if seed is None else seed + attempts
        assert attempt_seed is None
        graph = sample_3n_6(n, seed=attempt_seed)
        print(f'\nAttempt {attempts} at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', end='')

        if double_percolation:
            percolated = graph.is_double_percolating()
        else:
            percolated = graph.is_percolating()

        if not percolated:
            continue

        p_fname = f"{attempts}".replace('.', '_')
        if percolated:
            fname = f"percolating_{p_fname}_2.edgelist"
            path = os.path.join(output_dir, fname)
            nx.write_edgelist(graph.original_graph, path, data=False)
            summary = (f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | attempts={attempts} | "
                       f"nodes={graph.graph.number_of_nodes()} edges={graph.graph.number_of_edges()} "
                       f"| file={fname}\n")
            results.append({'attempts': attempts, 'file': path})
            count_percolated += 1
            print(f' Percolated and saved graph. Total percolated so far: {count_percolated}')
        else:
            summary = (f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | attempts={attempts} | "
                       f"FAILED to percolate within {max_tries} attempts\n")
            results.append({'attempts': attempts, 'file': None})

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(summary)

        time.sleep(0.01)

    print()
    return results

class Graph:
    def __init__(self, graph, build_helper=True, build_local_addition=False):
        self.graph = graph.copy() if graph is not None else None
        self.helper_matrix = None
        self.marked_vertices = None
        self.index_map = None


        if self.graph is not None:
            self.n = self.graph.number_of_nodes()
            self.original_graph = graph.copy()
            if build_helper:
                self.build_helper_matrix()
                self.build_marked_vertices()

            if build_local_addition:
                self.set_local_addition_matrix()
            else:
                self.local_addition_matrix = None


    def restore_graph(self):
        # Restore the graph object and rebuild helper structures.
        # Copy the original graph back to avoid mutating the stored original.
        self.graph = self.original_graph.copy()
        # Rebuild helper matrix and index map from the restored graph
        self.helper_matrix = self.build_helper_matrix()

        # Rebuild the marked vertices array
        self.build_marked_vertices()


    def build_marked_vertices(self):
        assert self.helper_matrix is not None, "Helper matrix must be built before building marked vertices."
        marked = [False] * self.helper_matrix.shape[0]
        for idx, node in enumerate(self.index_map.keys()):
            u, v = list(node)
            if self.graph.has_edge(u, v):
                marked[idx] = True
        self.marked_vertices = marked

    def build_helper_matrix(self):
        """
                Build helper matrix:
                - nodes are frozenset pairs {i,j} (i<j)
                - edge weight between two nodes is number of edges between the two pairs in G
                (only for disjoint pairs)
                """
        assert self.graph is not None, "Graph must be set before building helper graph."
        H = np.zeros((self.n * (self.n - 1) // 2, self.n * (self.n - 1) // 2), dtype=int)
        nodes = [frozenset(pair) for pair in itertools.combinations(self.graph.nodes(), 2)]
        self.index_map = {node: idx for idx, node in enumerate(nodes)}

        for a, b in itertools.combinations(nodes, 2):
            if a & b:
                continue  # pairs are not disjoint
            # count edges between the two pairs (up to 4)
            weight = 0
            for u in a:
                for v in b:
                    if self.graph.has_edge(u, v):
                        weight += 1
            if weight > 0:
                idx_a = self.index_map[a]
                idx_b = self.index_map[b]
                H[idx_a][idx_b] = weight
                H[idx_b][idx_a] = weight

        self.helper_matrix = H

        return H

    def set_local_addition_matrix(self):
        # For each edge addition (u,v) in the original graph, determine which helper matrix entries are affected. Return a dict of (u,v) -> matrix which is zero everywhere except for the affected entries.
        assert self.graph is not None, "Graph must be set before building local addition matrices."
        L = {}
        nodes = [frozenset(pair) for pair in itertools.combinations(self.graph.nodes(), 2)]

        index_map = self.index_map or {node: idx for idx, node in enumerate(nodes)}

        for u, v in itertools.combinations(self.graph.nodes(), 2):
            local_matrix = np.zeros((self.n * (self.n - 1) // 2, self.n * (self.n - 1) // 2), dtype=int)
            for pair in nodes:
                for other_pair in nodes:
                    if pair & other_pair:
                        continue  # pairs are not disjoint
                    if u in pair and v in other_pair or v in pair and u in other_pair:
                        idx_a = index_map[pair]
                        idx_b = index_map[other_pair]
                        local_matrix[idx_a][idx_b] = 1
                        local_matrix[idx_b][idx_a] = 1

            L[(u, v)] = local_matrix
            L[(v, u)] = local_matrix  # undirected edge
        self.local_addition_matrix = L
        return L

    def is_percolating_one_step(self, return_witness=False, k_222_plus=False):
        # Check if there exists a triangle in the helper matrix with weights 3,4,4
        # If k_222_plus is True, check if graph is k_222+ percolating
        H = self.helper_matrix
        if H is None:
            H = self.build_helper_matrix()

        if k_222_plus and self.marked_vertices is None:
            self.build_marked_vertices()

        size = H.shape[0]
        for i in range(size):
            for j in range(i + 1, size):
                for k in range(j + 1, size):
                    w1 = H[i][j]
                    w2 = H[j][k]
                    w3 = H[k][i]
                    weights = sorted([w1, w2, w3])
                    if weights == [4,4,4] and k_222_plus:
                        # Ensure there is at least one edge to be added
                        if self.marked_vertices[i] and self.marked_vertices[j] and self.marked_vertices[k]:
                            continue

                        nodes = list(self.index_map.keys())
                        return nodes[i], nodes[j], nodes[k]

                    if weights == [3, 4, 4]:
                        nodes = list(self.index_map.keys())
                        if k_222_plus:
                            # Ensure at least one vertex in the triangle is marked
                            if self.marked_vertices[i] or self.marked_vertices[j] or self.marked_vertices[k]:
                                return nodes[i], nodes[j], nodes[k]

                            else:
                                continue

                        if w1 == 3:
                            A = nodes[i]
                            B = nodes[j]
                        elif w2 == 3:
                            A = nodes[j]
                            B = nodes[k]
                        else:
                            A = nodes[k]
                            B = nodes[i]

                        for u in A:
                            for v in B:
                                if not self.graph.has_edge(u, v):
                                    if not return_witness:
                                        return u,v
                                    else:
                                        return (u, v) , (nodes[i], nodes[j], nodes[k])


        return None

    def is_percolating(self,
                       k_222_plus=False,
                       print_steps=False,
                       return_final_graph=False,
                       document_steps=False):

        assert not (k_222_plus and document_steps), "If k_222_plus is True, document_steps must be False."
        if self.n < 6:
            raise ValueError("Graph must have at least 6 vertices to check for k_222 percolation.")
        # Check if the graph is percolating by checking all possible edge additions
        L = self.local_addition_matrix
        if L is None:
            L = self.set_local_addition_matrix()

        witnesses = []
        order_of_additions = []


        while True:
            result = self.is_percolating_one_step(k_222_plus=k_222_plus, return_witness=document_steps)
            if result is None:
                break  # No more percolating configurations found

            if document_steps:
                result, witness = result
                order_of_additions.append(result)
                # witness is a tuple of frozensets representing the pairs involved. Add as a simple tuple of tuples.
                witnesses.append((*list(witness[0]),*list(witness[1]), *list(witness[2])))

            if k_222_plus:
                p1, p2, p3 = result
                # Add all edges between the vertices in the three pairs
                vertices = [*list(p1), *list(p2), *list(p3)]
                for u,v in itertools.combinations(vertices, 2):
                    if not self.graph.has_edge(u,v):
                        self.helper_matrix += L[(u, v)]
                        self.marked_vertices[self.index_map[frozenset({u,v})]] = True
                        self.graph.add_edge(u, v)
                        if print_steps:
                            print(f"Added edge ({u}, {v}) for k_222+ percolation.")
                continue

            u,v = result
            # Update helper matrix
            self.helper_matrix += L[(u, v)]
            self.graph.add_edge(u, v)
            if print_steps:
                print(f"Added edge ({u}, {v}) for percolation.")

        percolated = self.graph.number_of_edges() == self.n * (self.n - 1) // 2
        if return_final_graph:
            final_graph = self.graph.copy()
            # Restore original graph and helper matrix
            self.restore_graph()
            if document_steps:
                return percolated, final_graph, order_of_additions, witnesses
            return percolated, final_graph
        # Restore original graph and helper matrix
        self.restore_graph()
        if document_steps:
            return percolated, order_of_additions, witnesses
        return percolated



    def is_rigid(self, return_rigidity_matrix=False, return_rank=False):
        return PercolationGraph(self.graph).is_rigid(return_rigidity_matrix=return_rigidity_matrix, return_rank=return_rank)

    def is_k5_percolating(self, return_final_graph=False):
        return PercolationGraph(self.graph).is_k5_percolating(return_final_graph=return_final_graph)


    def is_k5_percolating_one_step(self, return_edge):
        for nodes in itertools.combinations(self.graph.nodes, 5):
            if self.graph.subgraph(nodes).number_of_edges() == 9:
                if not return_edge:
                    return True

                for u in nodes:
                    for v in nodes:
                        if u != v and not self.graph.has_edge(u,v):
                            return u, v
        return None


    def is_double_percolating(self):
        # Check if the graph is percolating by checking all possible edge additions
        L = self.local_addition_matrix
        if L is None:
            L = self.set_local_addition_matrix()
        H_original = self.helper_matrix
        if H_original is None:
            H_original = self.build_helper_matrix()

        while True:
            result = self.is_k5_percolating_one_step(return_edge=True)
            if result is None:
                result = self.is_percolating_one_step()

            if result is None:
                break

            u, v = result
            # Update helper matrix
            self.helper_matrix += L[(u, v)]
            self.graph.add_edge(u, v)

        percolated = self.graph.number_of_edges() == self.n * (self.n - 1) // 2
        # Restore original graph and helper matrix
        self.graph = self.original_graph.copy()
        self.helper_matrix = H_original.copy()
        return percolated

    def edge_percolates_in_process(self, edge: tuple[int, int], k222_plus: bool=False) -> bool:
        # Not optimized - it would be better to stop the percolation process as soon as the edge is added
        assert edge not in self.graph.edges(), "Edge is already present in the graph."
        result, order, wits = self.is_percolating(k_222_plus=k222_plus, document_steps=True)
        return edge in order



if __name__ == "__main__":
    n = 13
    # run_percolation_experiments(n=n, max_tries=10000, output_dir='Double percolating Graphs', double_percolation=True)
    # graph = read_graphs_from_edgelist(f'percolating Graphs/n_{n}')[0]
    # result, wits = graph.is_percolating(k_222_plus=False, document_steps=True)
    # print(f'Percolated: {result}, Witnesses: {wits}')


    G = nx.Graph()

    # nodes
    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8])

    # edges
    G.add_edges_from([
        (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 4), (2, 5),
        (3, 4), (3, 5),

        (6, 7), (7, 8), (6, 8),
        (0, 6), (0, 8),
        (1, 7), (1, 8),
        (2, 6), (2, 7),
    ])

    G.add_edges_from([(6,9), (7,9), (8,9), (9,10),
                      (6,10), (7,10), (8,10)])

    print(G.subgraph([6,7,8,9,10]).number_of_edges())

    graph = Graph(G)
    is_rigid, rank = graph.is_rigid(return_rank=True)
    print(f'Graph is rigid: {is_rigid}, rank: {rank}')
    nx.draw(G, with_labels=True)
    plt.show()
    G.remove_edge(0,4)
    graph = Graph(G)
    is_rigid, rank = graph.is_rigid(return_rank=True)
    print(f'Graph is rigid: {is_rigid}, rank: {rank}')
    nx.draw(G, with_labels=True)
    plt.show()


