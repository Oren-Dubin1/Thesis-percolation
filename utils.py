import itertools
import os
import random
import multiprocessing as mp

import networkx
import numpy as np
import networkx as nx
from pathlib import Path
import subprocess

from percolation_improved import Graph


def iter_stored_graphs_for_n(n, base_folder="percolating graphs"):
    folder = Path(base_folder) / f"n_{n}"

    if not folder.exists():
        return

    for path in sorted(folder.glob("*.edgelist")):
        G = nx.read_edgelist(path, nodetype=int)
        yield G


def check_all_rigid(n):
    for G in iter_stored_graphs_for_n(n):
        H = Graph(G)
        if not H.is_rigid():
            print(f"Graph {G.edges()} is not rigid.")
            raise RuntimeError(f"Found a non-rigid graph for n={n}.")
    print(f"All stored graphs for n={n} are rigid.")


def count_k222_steps(graph_or_wrapper) -> tuple[int, int]:
        """
        Run the percolation process (documented) and count how many K_{2,2,2}-type steps occurred,
        including how many appeared after K5 steps.

        Parameters
        ----------
        graph_or_wrapper : networkx.Graph or Graph
            The input graph (networkx.Graph) or a pre-built Graph wrapper.

        Returns
        -------
        tuple[int, int]
            (total_k222_count, k222_after_k5_count) where:
            - total_k222_count: total number of K222 steps in the percolation
            - k222_after_k5_count: number of K222 steps that appeared after at least one K5 step

        Notes
        -----
        - Uses `Graph.is_percolating(document_steps=True)` which returns
          (percolated, order_of_additions, witnesses).
        - Each `witness` entry created by `is_percolating` is a dict with "vertices"
          set to a tuple of length 6 for a K222 witness, and length 5 for a K5 witness.
        - The method restores the original graph state before returning, so this is safe to call.
        """
        # Normalize to Graph wrapper
        if isinstance(graph_or_wrapper, Graph):
            PG = graph_or_wrapper
        elif isinstance(graph_or_wrapper, nx.Graph):
            PG = Graph(graph_or_wrapper)
        else:
            raise TypeError("Expected a networkx.Graph or Graph instance")

        # Run documented percolation process
        percolated, order_of_additions, witnesses = PG.is_percolating(document_steps=True)

        # Count K222 witnesses and track those after K5
        k222_count = 0
        k222_after_k5_count = 0
        k5_seen = False

        for idx, w in enumerate(witnesses):
            witness_type = None

            # Determine witness type from dict structure
            if isinstance(w, dict):
                verts = w.get("vertices")
                if verts is not None:
                    if len(verts) == 6:
                        witness_type = "K222"
                        k222_count += 1
                        if k5_seen:
                            k222_after_k5_count += 1
                    elif len(verts) == 5:
                        witness_type = "K5"
                        k5_seen = True
            else:
                # Fallback for tuple-based witness format
                try:
                    if len(w) == 6:
                        witness_type = "K222"
                        k222_count += 1
                        if k5_seen:
                            k222_after_k5_count += 1
                    elif len(w) == 5:
                        witness_type = "K5"
                        k5_seen = True
                except Exception:
                    pass

        print(f"K222 steps performed: {k222_count}  "
              f"K222 after K5: {k222_after_k5_count}  "
              f"(percolated: {bool(percolated)})")

        if k222_after_k5_count > 0:
            print(f"Witness sequence: {witnesses}")

        return k222_count, k222_after_k5_count



def effective_resistance(G: nx.Graph, u, v) -> float:
    """
    Computes the effective resistance R_eff(u, v) in an unweighted graph G.

    Assumes:
        - G is connected
        - every edge has resistance 1

    Formula:
        R_eff(u,v) = (e_u - e_v)^T L^+ (e_u - e_v)
    where L^+ is the Moore-Penrose pseudoinverse of the Laplacian.
    """
    if u == v:
        return 0.0

    if not nx.is_connected(G):
        raise ValueError("Effective resistance is finite only inside a connected component.")

    nodes = list(G.nodes())
    idx = {node: i for i, node in enumerate(nodes)}

    L = nx.laplacian_matrix(G, nodelist=nodes).toarray().astype(float)
    L_pinv = np.linalg.pinv(L)

    b = np.zeros(len(nodes))
    b[idx[u]] = 1
    b[idx[v]] = -1

    return float(b @ L_pinv @ b)

def get_minimum_resistance_of_graph(G: nx.Graph) -> tuple[float, tuple[int, int]]:
    min_res = 2
    min_vertices = -1, -1
    for u, v in G.edges():
        res = effective_resistance(G, u, v)
        if res < min_res:
            min_res = res
            min_vertices = u, v

    return min_res, min_vertices



def check_edge_with_resistance_at_least_5_12(n : int):
    for G in iter_stored_graphs_for_n(n):
        for e in G.edges():
            H = G.copy()
            H.remove_edge(*e)
            answer, final_graph = Graph(H).is_percolating(return_final_graph=True)
            if final_graph.number_of_edges() != H.number_of_edges():
                continue

            R_eff, edge = get_minimum_resistance_of_graph(H)
            if R_eff < 5 / 12:
                print(f"Graph has edge {edge} with effective resistance {R_eff:.4f} < 5/12.")
                for u,v in H.edges():
                    print(u, v)
                raise RuntimeError(f"Found a graph for n={n} with an edge of effective resistance less than 5/12 = {5/12:.4f}.")



def save_unlabeled_graphs_up_to_n(max_n=9, out_dir="unlabeled_graphs"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    for n in range(1, max_n + 1):
        out_path = out_dir / f"unlabeled_graphs_n{n}.g6"

        print(f"Generating n={n} -> {out_path}")

        result = subprocess.run(
            ["wsl", "nauty-geng", str(n)],
            capture_output=True,
            text=True,
            check=True,
        )

        lines = []

        for line in result.stdout.splitlines():
            line = line.strip()

            if not line or line.startswith(">"):
                continue

            lines.append(line)

        with open(out_path, "w") as f:
            for line in lines:
                f.write(line + "\n")

        print(f"Saved {len(lines)} graphs")


def load_unlabeled_graphs_of_order(n: int, directory="unlabeled_graphs"):
    path = Path(directory) / f"unlabeled_graphs_n{n}.g6"

    if not path.exists():
        raise FileNotFoundError(
            f"No file found for n={n}: {path}"
        )

    graphs = []

    with open(path, "rb") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            G = nx.from_graph6_bytes(line)
            G = nx.convert_node_labels_to_integers(G)

            graphs.append(G)

    print(f"Loaded {len(graphs)} unlabeled graphs on {n} vertices")

    return graphs


if __name__ == '__main__':
    G = nx.read_edgelist(r'C:\Oren\Academy\Thesis\Thesis-percolation\percolating graphs\n_15\percolating_73.edgelist')

    print(G.number_of_nodes())
    print(G.number_of_edges())
    print(Graph(G).is_percolating(print_steps=True, document_steps=True))
    # G = nx.complete_multipartite_graph(2,2,2)
    # G.add_edges_from([('x', 0), ('x', 1), ('y', 0), ('y', 1), ('z', 0), ('z', 2), ('x', 'y'), ('y', 'z'), ('x', 'z')])
    # print(Graph(G).is_percolating(return_final_graph=True))





