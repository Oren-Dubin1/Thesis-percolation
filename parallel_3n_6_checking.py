import os
import random
import multiprocessing as mp
import networkx as nx

from percolation_improved import *


def random_nonedge(G, rng):
    nodes = list(G.nodes)
    while True:
        u, v = rng.sample(nodes, 2)
        if u > v:
            u, v = v, u
        if not G.has_edge(u, v):
            return u, v


def check_one(args):
    n, i, base_seed, save_folder = args

    rng = random.Random(base_seed + i)

    # exactly 3n - 7 edges
    G0 = nx.gnm_random_graph(n, 3 * n - 7, seed=base_seed + i)

    graph_obj = Graph(G0.copy())
    if graph_obj.is_percolating():
        path = os.path.join(save_folder, f"counterexample_3n_minus_7_{i}.edgelist")
        nx.write_edgelist(G0, path, data=False)
        return ("counterexample", i, path)

    # add one random missing edge, giving 3n - 6 edges
    e = random_nonedge(G0, rng)
    G0.add_edge(*e)

    graph_obj = Graph(G0.copy())
    if graph_obj.is_percolating():
        path = os.path.join(save_folder, f"percolating_{i}.edgelist")
        nx.write_edgelist(G0, path, data=False)
        return ("percolating", i, path)

    return ("none", i, None)


def check_3n6_conjecture_parallel(
    n,
    num_tries=1000,
    workers=None,
    seed=0,
    chunksize=32,
):
    if workers is None:
        workers = max(1, os.cpu_count() - 1)

    save_folder = os.path.join("percolating graphs", f"n_{n}")
    os.makedirs(save_folder, exist_ok=True)

    tasks = [
        (n, i, seed, save_folder)
        for i in range(num_tries)
    ]

    found = []

    with mp.Pool(processes=workers) as pool:
        for done, result in enumerate(pool.imap_unordered(check_one, tasks, chunksize=chunksize), start=1):
            status, i, path = result

            if done % 10 == 0:
                print(f"Passed {done}/{num_tries}")

            if status == "counterexample":
                pool.terminate()
                raise RuntimeError(f"Found a 3n-7 percolating graph: {path}")

            if status == "percolating":
                found.append(path)
                print(f"Found a 3n-6 percolating graph: {path}")

    return found

if __name__ == "__main__":
    n = 22
    check_3n6_conjecture_parallel(n=22, num_tries=10000)