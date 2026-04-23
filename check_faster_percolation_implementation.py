import random
import statistics
import time

import networkx as nx

from percolation_improved import Graph


def benchmark_percolation_functions(graph_objects, repeat=3, shuffle=True, seed=None, validate_results=True):
    """
    Benchmark Graph.is_percolating vs Graph.is_percolating_faster on the same inputs.

    Parameters
    ----------
    graph_objects : list[Graph] or list[nx.Graph]
        Graphs to test.
    repeat : int
        Number of benchmark repeats.
    shuffle : bool
        Shuffle graph order each repeat.
    seed : int | None
        Seed for deterministic shuffling.
    validate_results : bool
        If True, assert both methods return the same boolean per graph.

    Returns
    -------
    dict
        Timing summary + validation info.
    """
    rng = random.Random(seed)

    percolating_times = []
    faster_times = []
    mismatches = []

    for r in range(repeat):
        print(f'In {r + 1} iteration')
        order = list(range(len(graph_objects)))
        if shuffle:
            rng.shuffle(order)

        t1_total = 0.0
        t2_total = 0.0

        for idx in order:
            obj = graph_objects[idx]

            # Normalize to a base NetworkX graph
            if isinstance(obj, Graph):
                base_nx = obj.original_graph.copy()
            elif isinstance(obj, nx.Graph):
                base_nx = obj.copy()
            else:
                raise TypeError(
                    "graph_objects must contain Graph or networkx.Graph instances; "
                    f"got {type(obj)} at index {idx}"
                )

            # Run is_percolating on a fresh wrapper
            g1 = Graph(base_nx)
            start = time.perf_counter()
            res1 = g1.is_percolating_slow()
            t1_total += time.perf_counter() - start

            # Run is_percolating_faster on a fresh wrapper
            g2 = Graph(base_nx)
            start = time.perf_counter()
            res2 = g2.is_percolating()
            t2_total += time.perf_counter() - start

            if validate_results and res1 != res2:
                mismatches.append((r, idx, res1, res2))

        percolating_times.append(t1_total)
        faster_times.append(t2_total)

    mean_perc = statistics.mean(percolating_times)
    mean_fast = statistics.mean(faster_times)

    return {
        "is_percolating_times": percolating_times,
        "is_percolating_faster_times": faster_times,
        "is_percolating_mean": mean_perc,
        "is_percolating_faster_mean": mean_fast,
        "is_percolating_median": statistics.median(percolating_times),
        "is_percolating_faster_median": statistics.median(faster_times),
        "speedup_faster_over_percolating": (mean_perc / mean_fast) if mean_fast > 0 else float("inf"),
        "speedup_percolating_over_faster": (mean_fast / mean_perc) if mean_perc > 0 else float("inf"),
        "results_match": len(mismatches) == 0,
        "mismatches": mismatches,
    }

if __name__ == '__main__':
    graphs = []
    for _ in range(4):
        graphs.append(nx.erdos_renyi_graph(20, 0.05))

    print(benchmark_percolation_functions(graphs, repeat=7))