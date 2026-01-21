import itertools
import networkx as nx
import random

import numpy as np

from Graphs import PercolationGraph
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
        F = Graph(None)
        F.read_graph_from_edgelist(fpath)
        graphs.append(F)

    return graphs


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
    def __init__(self, graph, build_helper=True):
        self.graph = graph
        if self.graph is not None:
            self.n = self.graph.number_of_nodes()
            self.helper_matrix = None
            self.index_map = None
            self.local_addition_matrix = None
            if build_helper: self.build_helper_matrix()
            self.original_graph = graph.copy()

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

    def is_percolating_one_step(self, return_vertices=True):
        # Check if there exists a triangle in the helper matrix with weights 3,4,4
        H = self.helper_matrix
        if H is None:
            H = self.build_helper_matrix()

        size = H.shape[0]
        for i in range(size):
            for j in range(i + 1, size):
                for k in range(j + 1, size):
                    w1 = H[i][j]
                    w2 = H[j][k]
                    w3 = H[k][i]
                    weights = sorted([w1, w2, w3])
                    if weights == [3, 4, 4]:
                        # Retrieve the corresponding vertex pairs

                        nodes = list(self.index_map.keys())
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
                                    if return_vertices:
                                        return u,v

                        else:
                            return True
        return None

    def is_percolating(self):
        # Check if the graph is percolating by checking all possible edge additions
        L = self.local_addition_matrix
        if L is None:
            L = self.set_local_addition_matrix()
        H_original = self.helper_matrix
        if H_original is None:
            H_original = self.build_helper_matrix()

        while True:
            result = self.is_percolating_one_step()
            if result is None:
                break  # No more percolating configurations found

            u,v = result
            # Update helper matrix
            self.helper_matrix += L[(u, v)]
            self.graph.add_edge(u, v)

        percolated = self.graph.number_of_edges() == self.n * (self.n - 1) // 2
        # Restore original graph and helper matrix
        self.graph = self.original_graph.copy()
        self.helper_matrix = H_original.copy()
        return percolated


    def read_graph_from_edgelist(self, path):
        """Read a graph from an edgelist file and set it as self.graph."""
        self.graph = nx.read_edgelist(path, nodetype=int)
        self.original_graph = self.graph.copy()
        return self.graph

    def is_rigid(self):
        return PercolationGraph(self.graph).is_rigid()





if __name__ == "__main__":
    n = 20
    # run_percolation_experiments(n=n, max_tries=100000)
    graphs = read_graphs_from_edgelist(f'percolating graphs/n_{n}')
    flag = True
    for g in graphs:
        rig = g.is_rigid()
        flag = flag and rig

    print(f"All graphs rigid: {flag}")
