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
if __name__ == '__main__':
    n = 25
    check_all_rigid(n)

