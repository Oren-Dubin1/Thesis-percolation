import itertools
import networkx as nx
import random
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

    finders = []
    for fpath in files:
        F = LargeGraphFinder()
        F.read_graph_from_edgelist(fpath)
        finders.append(F)

    return finders



class LargeGraphFinder:
    def __init__(self, graph=None, helper_graph=None, n=10):
        self.graph = graph
        self.initial_graph = graph.copy() if graph is not None else None
        self.helper_graph = helper_graph
        self.n = n

        self.all_edges = list(itertools.combinations(range(self.n), 2))



    def print_graph(self, file=None):
        PercolationGraph(self.graph).print_graph(file=file)

    def sample_gnp(self, seed=None, p =0.5):
        """Return a random G(n, p) (networkx Graph)."""
        if seed is not None:
            random.seed(seed)

        self.graph = nx.erdos_renyi_graph(self.n, p, seed=seed)
        self.initial_graph = self.graph.copy()
        self._build_helper_graph()
        return self.graph

    def sample_3n_6(self, seed=None):
        """Return a random 3n - 6 graph (networkx Graph) with minimal degree >= 3."""
        if seed is not None:
            random.seed(seed)
        while True:
            selected_edges = random.sample(list(self.all_edges), 3 * self.n - 6)
            G = nx.Graph()
            G.add_nodes_from(range(self.n))
            G.add_edges_from(selected_edges)
            if all(deg >= 3 for _, deg in G.degree()):
                break

        self.graph = G
        self.initial_graph = self.graph.copy()
        self._build_helper_graph()
        return self.graph

    def _build_helper_graph(self):
        """
        Build helper graph:
        - nodes are frozenset pairs {i,j} (i<j)
        - edge weight between two nodes is number of edges between the two pairs in G
        (only for disjoint pairs)
        """
        assert self.graph is not None, "Graph must be set before building helper graph."
        H = nx.Graph()
        nodes = [frozenset(pair) for pair in itertools.combinations(self.graph.nodes(), 2)]
        H.add_nodes_from(nodes)

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
                H.add_edge(a, b, weight=weight)

        self.helper_graph = H
        self.initial_helper_graph = H.copy()
        return self.helper_graph


    def update_helper_for_original_edge(self, u, v):
        """
        Incrementally update `self.helper_graph` to reflect adding the original edge (u, v).

        Notes:
        - This implementation assumes edges are only added (no removals).
        - For every helper-node A that contains u and helper-node B that contains v (and A and B disjoint),
          the helper-edge weight corresponding to cross-edges between A and B is incremented by 1.
        - If increment creates a new nonzero weight, the edge is added with weight 1.
        - Returns the number of helper-edges created or updated.
        """
        if self.helper_graph is None:
            raise RuntimeError("Helper graph not set. Build it first.")

        H = self.helper_graph

        # collect helper nodes that contain u and v
        nodes_u = [node for node in H.nodes() if u in node]
        nodes_v = [node for node in H.nodes() if v in node]

        # Prepare updates without mutating the graph while iterating adjacency
        updates = []  # tuples (A, B, new_weight) where new_weight > 0 to set

        seen = set()  # track processed unordered helper-edge pairs to avoid duplicates

        # 1) Update existing edges: iterate neighbors of nodes containing u and pick those that contain v
        for A in nodes_u:
            for B in H.neighbors(A):
                if v not in B:
                    continue
                key = (A, B) if id(A) <= id(B) else (B, A)
                if key in seen:
                    continue
                seen.add(key)
                cur = H[A][B].get('weight', 0)
                new = cur + 1
                updates.append((A, B, new))

        # 2) Create new helper-edges that didn't exist but should receive +1 on add
        for A in nodes_u:
            for B in nodes_v:
                if A is B:
                    continue
                if A & B:
                    continue  # not disjoint
                if H.has_edge(A, B):
                    continue  # already handled above
                # ensure we don't double-add the same unordered pair
                key = (A, B) if id(A) <= id(B) else (B, A)
                if key in seen:
                    continue
                seen.add(key)
                updates.append((A, B, 1))

        # Apply updates
        applied = 0
        for A, B, new_weight in updates:
            # add_edge will set/update the weight attribute
            H.add_edge(A, B, weight=new_weight)
            applied += 1

        return applied


    def _is_k222_percolating_one_step(self):
        """
        Search triangles in the helper graph with weights 3,4,4.
        If found, identify the edge with weight 3, take the two helper-nodes
        (each is a frozenset of two original vertices), return the four original
        vertices (as a sorted tuple) together with True.
        Return False if none found.
        """
        H = self.helper_graph
        if H is None:
            raise "Helper graph not set. Build it first."

        for u, v, w in itertools.combinations(H.nodes(), 3):
            if H.has_edge(u, v) and H.has_edge(v, w) and H.has_edge(w, u):
                wt_uv = H[u][v].get('weight', 0)
                wt_vw = H[v][w].get('weight', 0)
                wt_wu = H[w][u].get('weight', 0)
                weights_sorted = sorted([wt_uv, wt_vw, wt_wu])
                if weights_sorted == [3, 4, 4]:
                    # find which edge has weight 3 and return the four original vertices
                    if wt_uv == 3:
                        pair_a, pair_b = u, v
                    elif wt_vw == 3:
                        pair_a, pair_b = v, w
                    else:
                        pair_a, pair_b = w, u
                    four_vertices = tuple(sorted(pair_a)), tuple(sorted(pair_b))
                    u0, u1 = tuple(sorted(u))
                    v0, v1 = tuple(sorted(v))
                    w0, w1 = tuple(sorted(w))
                    return four_vertices, u0, u1, v0, v1, w0, w1
        return False

    def is_k222_percolating(self, print_steps=False):
        """
        Repeatedly apply one-step percolation until no more steps possible.
        Return True if at least one step was performed, else False.
        """
        if self.helper_graph is None:
            self._build_helper_graph()

        while True:
            result = self._is_k222_percolating_one_step()
            if result is False:
                break
            # perform the percolation step
            four_vertices, u0, u1, v0, v1, w0, w1 = result

            # add edge between the two vertices not connected by weight 3 edge
            edge = None
            for u in four_vertices[0]:
                for v in four_vertices[1]:
                    if not self.graph.has_edge(u, v):
                        self.graph.add_edge(u, v)
                        edge = (u, v)
                        if print_steps:
                            print(f"adding edge {edge} using witnesses {(u0, u1), (v0, v1), (w0, w1)}")
                        break

            # rebuild helper graph
            assert edge is not None
            self.update_helper_for_original_edge(*edge)

        return self.graph.number_of_edges() == self.graph.number_of_nodes() * (self.graph.number_of_nodes() - 1) // 2


    # python
    def run_percolation_experiments(self,
                                    n=None,
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

        orig_n = self.n
        if n is not None:
            self.n = n

        output_dir = output_dir + f'/n_{self.n}'

        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, log_file)

        results = []
        try:
            for attempts in range(max_tries):
                attempt_seed = None if seed is None else seed + attempts
                self.sample_3n_6(seed=attempt_seed)
                print(f'Attempt {attempts} at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                percolated = self.is_k222_percolating()
                if not percolated:
                    continue

                p_fname = f"{attempts}".replace('.', '_')
                if percolated:
                    fname = f"percolating_{p_fname}.edgelist"
                    path = os.path.join(output_dir, fname)
                    nx.write_edgelist(self.initial_graph, path, data=False)
                    summary = (f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | attempts={attempts} | "
                               f"nodes={self.graph.number_of_nodes()} edges={self.graph.number_of_edges()} "
                               f"| file={fname}\n")
                    results.append({'attempts': attempts, 'file': path})
                    print(summary)
                else:
                    summary = (f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | attempts={attempts} | "
                               f"FAILED to percolate within {max_tries} attempts\n")
                    results.append({'attempts': attempts, 'file': None})

                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(summary)

                time.sleep(0.01)
        finally:
            # restore original parameters
            self.n = orig_n

        return results

    def read_graph_from_edgelist(self, path):
        """Read a graph from an edgelist file and set it as self.graph."""
        self.graph = nx.read_edgelist(path, nodetype=int)
        self.initial_graph = self.graph.copy()
        return self.graph


if __name__ == "__main__":
    F = LargeGraphFinder(n=10)
    F.run_percolation_experiments(max_tries=100)
