import itertools
import json
import os
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import networkx as nx
import numpy as np
import pulp
from tqdm import tqdm

from utils import load_unlabeled_graphs_of_order


GLOBAL_CLASS_CACHE = None
GLOBAL_M = None


def edge_list_kn(n: int):
    return list(itertools.combinations(range(n), 2))


def edge_index_dict(edges):
    return {e: i for i, e in enumerate(edges)}


def graph_from_mask(n: int, edges, mask: int):
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i, (u, v) in enumerate(edges):
        if (mask >> i) & 1:
            G.add_edge(u, v)

    return G


def build_iso_lookup(reps):
    buckets = defaultdict(list)

    for i, G in enumerate(reps):
        h = nx.weisfeiler_lehman_graph_hash(G)
        buckets[h].append((i, G))

    return buckets


def class_id(G, buckets):
    h = nx.weisfeiler_lehman_graph_hash(G)

    for i, H in buckets[h]:
        if nx.is_isomorphic(G, H):
            return i

    raise ValueError("Graph class not found")


def make_class_id_cached(n, edges, buckets):
    cache = {}

    def get(mask: int):
        if mask not in cache:
            G = graph_from_mask(n, edges, mask)
            cache[mask] = class_id(G, buckets)
        return cache[mask]

    return get


def complete_graph_mask_on_subset(subset, edge_to_index):
    mask = 0

    for u, v in itertools.combinations(subset, 2):
        e = tuple(sorted((u, v)))
        mask |= 1 << edge_to_index[e]

    return mask


def k222_mask_on_partition(A, B, C, edge_to_index):
    mask = 0

    for X, Y in [(A, B), (A, C), (B, C)]:
        for x in X:
            for y in Y:
                e = tuple(sorted((x, y)))
                mask |= 1 << edge_to_index[e]

    return mask


def class_cache_path(n: int):
    return f"class_cache_n{n}.uint16.dat"


def class_cache_exists(n: int, m: int):
    path = class_cache_path(n)
    expected_size = (1 << m) * np.dtype(np.uint16).itemsize

    return os.path.exists(path) and os.path.getsize(path) == expected_size


def class_cache_build_worker(args):
    n, start, end = args

    edges = edge_list_kn(n)
    m = len(edges)
    reps = load_unlabeled_graphs_of_order(n)
    buckets = build_iso_lookup(reps)
    class_of_mask = make_class_id_cached(n=n, edges=edges, buckets=buckets)

    total = 1 << m
    path = class_cache_path(n)

    cache = np.memmap(path, dtype=np.uint16, mode="r+", shape=(total,))

    for mask in range(start, end):
        cache[mask] = class_of_mask(mask)

    cache.flush()

    return end - start


def build_class_cache_memmap_parallel(n: int, m: int, num_workers=8, chunk_size=50_000):
    total = 1 << m
    path = class_cache_path(n)

    print(f"Creating class cache file: {path}")
    print(f"Total masks: {total:,}")
    print(f"Workers: {num_workers}")
    print(f"Chunk size: {chunk_size:,}")

    cache = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))
    cache.flush()
    del cache

    tasks = []

    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        tasks.append((n, start, end))

    with Pool(processes=num_workers) as pool:
        with tqdm(total=total, desc="Building class cache", unit="mask") as pbar:
            for done in pool.imap_unordered(class_cache_build_worker, tasks):
                pbar.update(done)

    print(f"Finished class cache: {path}")

    return path


def ensure_class_cache(n: int, m: int, num_workers=None):
    if class_cache_exists(n=n, m=m):
        print(f"Using existing class cache: {class_cache_path(n)}")
        return class_cache_path(n)

    return build_class_cache_memmap_parallel(n=n, m=m, num_workers=num_workers)


def open_class_cache(n: int, m: int):
    return np.memmap(class_cache_path(n), dtype=np.uint16, mode="r", shape=(1 << m,))


def init_worker_memmap(n, m):
    global GLOBAL_CLASS_CACHE, GLOBAL_M

    GLOBAL_CLASS_CACHE = open_class_cache(n=n, m=m)
    GLOBAL_M = m


def monotonicity_worker(args):
    start_mask, end_mask, worker_id = args

    class_cache = GLOBAL_CLASS_CACHE
    m = GLOBAL_M

    constraints = set()

    for mask in range(start_mask, end_mask):
        if (mask - start_mask) % 100_000 == 0:
            print(f"Monotonicity worker {worker_id}: {mask}/{end_mask}")

        id_A = int(class_cache[mask])

        for e_idx in range(m):
            if not ((mask >> e_idx) & 1):
                id_B = int(class_cache[mask | (1 << e_idx)])
                constraints.add((id_A, id_B))

    return constraints


def add_monotonicity_constraints_parallel(prob, r, n: int, m: int, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    total = 1 << m
    chunk_size = (total + num_workers - 1) // num_workers

    tasks = []

    for worker_id in range(num_workers):
        start_mask = worker_id * chunk_size
        end_mask = min(total, start_mask + chunk_size)

        if start_mask < end_mask:
            tasks.append((start_mask, end_mask, worker_id))

    print(f"Generating monotonicity constraints using {num_workers} workers...")

    constraints = set()

    with Pool(processes=num_workers, initializer=init_worker_memmap, initargs=(n, m)) as pool:
        for worker_constraints in tqdm(pool.imap_unordered(monotonicity_worker, tasks), total=len(tasks), desc="Monotonicity workers"):
            constraints.update(worker_constraints)

    print(f"Distinct monotonicity constraints: {len(constraints)}")
    print("Adding monotonicity constraints to PuLP model...")

    for id_A, id_B in constraints:
        prob += r[id_A] <= r[id_B]

    return len(constraints)


def elementary_submodularity_worker(args):
    start_A, end_A, worker_id = args

    class_cache = GLOBAL_CLASS_CACHE
    m = GLOBAL_M

    constraints = set()

    for A in range(start_A, end_A):
        id_A = int(class_cache[A])
        missing = [e for e in range(m) if not ((A >> e) & 1)]

        for i in range(len(missing)):
            e = missing[i]
            Ae = A | (1 << e)
            id_Ae = int(class_cache[Ae])

            for j in range(i + 1, len(missing)):
                f = missing[j]
                Af = A | (1 << f)
                Aef = Ae | (1 << f)

                id_Af = int(class_cache[Af])
                id_Aef = int(class_cache[Aef])

                left_1, left_2 = sorted((id_Ae, id_Af))
                constraints.add((left_1, left_2, id_A, id_Aef))

    return end_A - start_A, constraints


def add_all_elementary_submodularity_constraints_parallel(prob, r, n: int, m: int, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    total = 1 << m
    chunk_size = 100_000

    tasks = []

    worker_id = 0
    for start_A in range(0, total, chunk_size):
        end_A = min(total, start_A + chunk_size)
        tasks.append((start_A, end_A, worker_id))
        worker_id = (worker_id + 1) % num_workers

    print(f"Generating elementary submodularity constraints using {num_workers} workers...")
    print(f"Total A masks: {total:,}")
    print(f"Chunk size: {chunk_size:,}")

    constraints = set()

    with Pool(processes=num_workers, initializer=init_worker_memmap, initargs=(n, m)) as pool:
        with tqdm(total=total, desc="Generating submodularity", unit="A") as pbar:
            for done, worker_constraints in pool.imap_unordered(elementary_submodularity_worker, tasks):
                constraints.update(worker_constraints)
                pbar.update(done)

    print(f"Distinct elementary submodularity constraints: {len(constraints)}")
    print("Adding submodularity constraints to PuLP model...")

    for id_Ae, id_Af, id_A, id_Aef in tqdm(constraints, desc="Adding submodularity"):
        prob += r[id_Ae] + r[id_Af] >= r[id_A] + r[id_Aef]

    return len(constraints)


def graph_to_mask(G, edge_to_index):
    mask = 0

    for u, v in G.edges():
        e = tuple(sorted((u, v)))
        mask |= 1 << edge_to_index[e]

    return mask


def save_values_by_representatives(n, reps, r, filename):
    edges = edge_list_kn(n)
    edge_to_index = edge_index_dict(edges)
    m = len(edges)

    values = {}

    for i, G in enumerate(reps):
        mask = graph_to_mask(G, edge_to_index)
        values[format(mask, f"0{m}b")] = round(pulp.value(r[i]))

    with open(filename, "w") as f:
        json.dump(values, f, indent=4)

    print(f"Saved {len(values)} representatives to {filename}")


def build_base_problem(n: int, num_workers=None):
    edges = edge_list_kn(n)
    edge_to_index = edge_index_dict(edges)
    m = len(edges)

    print(f"n={n}, |E(K_n)|={m}")

    reps = load_unlabeled_graphs_of_order(n)
    print(f"Number of unlabeled graphs: {len(reps)}")

    ensure_class_cache(n=n, m=m, num_workers=num_workers)
    class_cache = open_class_cache(n=n, m=m)

    prob = pulp.LpProblem("symmetric_polymatroid_rank", pulp.LpMaximize)

    r = {
        i: pulp.LpVariable(f"r_{i}", lowBound=0, cat=pulp.LpInteger)
        for i in range(len(reps))
    }

    empty_mask = 0
    full_mask = (1 << m) - 1

    empty_id = int(class_cache[empty_mask])
    full_id = int(class_cache[full_mask])

    prob += r[empty_id] == 0

    print("Adding size bounds...")
    for i, G in enumerate(reps):
        prob += r[i] <= G.number_of_edges()

    print("Adding monotonicity constraints...")
    added = add_monotonicity_constraints_parallel(prob=prob, r=r, n=n, m=m, num_workers=num_workers)
    print("Added monotonicity constraints:", added)

    print("Adding K5 circuit constraint...")
    k5_mask = complete_graph_mask_on_subset(tuple(range(5)), edge_to_index)
    k5_id = int(class_cache[k5_mask])

    prob += r[k5_id] == 9

    for e_idx in range(m):
        if (k5_mask >> e_idx) & 1:
            k5_minus_e = k5_mask ^ (1 << e_idx)
            prob += r[int(class_cache[k5_minus_e])] == 9

    print("Adding K222 circuit constraint...")
    k222_mask = k222_mask_on_partition(A=(0, 1), B=(2, 3), C=(4, 5), edge_to_index=edge_to_index)
    k222_id = int(class_cache[k222_mask])

    prob += r[k222_id] == 11

    for e_idx in range(m):
        if (k222_mask >> e_idx) & 1:
            k222_minus_e = k222_mask ^ (1 << e_idx)
            prob += r[int(class_cache[k222_minus_e])] == 11

    prob += r[full_id]

    return prob, r, reps, m, full_id


def save_base_model(n: int, filename: str | None = None, num_workers=None):
    if filename is None:
        filename = f"base_model_n{n}.json"

    print(f"Saving base model to {filename}...")

    prob, r, reps, m, full_id = build_base_problem(n=n, num_workers=num_workers)
    prob.toJson(filename)

    print(f"Saved base model to {filename}")


def load_model(n: int, filename: str | None = None):
    if filename is None:
        filename = f"base_model_n{n}.json"

    print(f"Loading base model from {filename}...")

    var_dict, prob = pulp.LpProblem.fromJson(filename)

    r = {
        int(name.split("_")[1]): var
        for name, var in var_dict.items()
        if name.startswith("r_")
    }

    reps = load_unlabeled_graphs_of_order(n)
    return prob, r, reps


def get_solver():
    try:
        solver = pulp.HiGHS(msg=True)

        if solver.available():
            return solver
    except Exception:
        pass

    print("HiGHS unavailable. Falling back to CBC.")
    return pulp.PULP_CBC_CMD(msg=True)


def solve_with_all_elementary_submodularity(n: int, num_workers=None):
    edges = edge_list_kn(n)
    m = len(edges)

    ensure_class_cache(n=n, m=m, num_workers=num_workers)
    class_cache = open_class_cache(n=n, m=m)

    full_mask = (1 << m) - 1
    full_id = int(class_cache[full_mask])

    try:
        prob, r, reps = load_model(n, filename=full_model_filename(n))
        print("Loaded full model successfully")

    except FileNotFoundError:
        try:
            prob, r, reps = load_model(n)

        except FileNotFoundError:
            save_base_model(n=n, num_workers=num_workers)
            prob, r, reps = load_model(n)

        print("Loaded base model successfully")

        added = add_all_elementary_submodularity_constraints_parallel(
            prob=prob,
            r=r,
            n=n,
            m=m,
            num_workers=num_workers,
        )

        print("Added elementary submodularity constraints:", added)

        filename = full_model_filename(n)
        print(f"Saving full model to {filename}...")
        prob.toJson(filename)
        print("Saved full model")

    solver = get_solver()

    print("Solving...")
    prob.solve(solver)

    print("Status:", pulp.LpStatus[prob.status])
    print("Final symmetric LP value:", pulp.value(r[full_id]))

    save_values_by_representatives(
        n=n,
        reps=reps,
        r=r,
        filename=f"final_values_n{n}_representatives.json",
    )

    return prob, r, reps

def full_model_filename(n: int):
    return f"full_model_n{n}.json"


if __name__ == "__main__":
    solve_with_all_elementary_submodularity(n=8, num_workers=8)