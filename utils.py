import os
import random
import multiprocessing as mp
import networkx as nx
from pathlib import Path

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


def count_k222_steps(graph_or_wrapper) -> int:
    """
    Run the percolation process (documented) and count how many K_{2,2,2}-type steps occurred.

    Parameters
    ----------
    graph_or_wrapper : networkx.Graph or Graph
        The input graph (networkx.Graph) or a pre-built Graph wrapper.

    Returns
    -------
    int
        Number of K_{2,2,2} witness steps performed during the percolation process.

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

    # Count K222 witnesses: in this implementation K222 witnesses store 6 vertices
    k222_count = 0
    for w in witnesses:
        # witness may be a dict with key "vertices" (K222 -> 6 entries, K5 -> 5 entries)
        if isinstance(w, dict):
            verts = w.get("vertices")
            if verts is not None and len(verts) == 6:
                k222_count += 1
        else:
            # if older format used (tuple with 6 items), detect by length
            try:
                if len(w) == 6:
                    k222_count += 1
            except Exception:
                pass

    print(f"K222 steps performed: {k222_count}  (percolated: {bool(percolated)})")
    return k222_count

if __name__ == '__main__':
    n = 8
    for G in iter_stored_graphs_for_n(n):
        count_k222_steps(G)

