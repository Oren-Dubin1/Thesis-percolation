import itertools
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import highspy
import networkx as nx
import numpy as np
from tqdm import tqdm

from utils import load_unlabeled_graphs_of_order


MATROIDS_DIR = Path(__file__).resolve().parent

GLOBAL_CLASS_CACHE = None
GLOBAL_M = None


def edge_list_kn(n: int):
    return list(itertools.combinations(range(n), 2))


def edge_index_dict(edges):
    return {e: i for i, e in enumerate(edges)}


def var(class_id: int):
    return f"r_{class_id}"


def class_cache_path(n: int):
    return MATROIDS_DIR / f"class_cache_n{n}.uint16.dat"


def symmetric_lp_filename(n: int, integer=False):
    kind = "integer" if integer else "lp"
    return MATROIDS_DIR / f"symmetric_full_n{n}_{kind}.lp"


def symmetric_values_filename(n: int, integer=False):
    kind = "integer" if integer else "lp"
    return MATROIDS_DIR / f"symmetric_values_n{n}_{kind}.json"


def graph_from_mask(n: int, edges, mask: int):
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i, (u, v) in enumerate(edges):
        if (mask >> i) & 1:
            G.add_edge(u, v)

    return G


def graph_to_mask(G, edge_to_index):
    mask = 0

    for u, v in G.edges():
        e = tuple(sorted((u, v)))
        mask |= 1 << edge_to_index[e]

    return mask


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


def class_cache_exists(n: int, m: int):
    path = class_cache_path(n)
    expected_size = (1 << m) * np.dtype(np.uint16).itemsize
    return path.exists() and path.stat().st_size == expected_size


def open_class_cache(n: int, m: int):
    return np.memmap(class_cache_path(n), dtype=np.uint16, mode="r", shape=(1 << m,))


def class_cache_build_worker(args):
    n, start, end = args

    edges = edge_list_kn(n)
    m = len(edges)

    reps = load_unlabeled_graphs_of_order(n)
    buckets = build_iso_lookup(reps)
    class_of_mask = make_class_id_cached(n=n, edges=edges, buckets=buckets)

    cache = np.memmap(class_cache_path(n), dtype=np.uint16, mode="r+", shape=(1 << m,))

    for mask in range(start, end):
        cache[mask] = class_of_mask(mask)

    cache.flush()
    return end - start


def build_class_cache_memmap_parallel(n: int, m: int, num_workers=None, chunk_size=50_000):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    total = 1 << m
    path = class_cache_path(n)

    print(f"Creating class cache file: {path}")
    print(f"Total masks: {total:,}")
    print(f"Workers: {num_workers}")

    cache = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))
    cache.flush()
    del cache

    tasks = [
        (n, start, min(total, start + chunk_size))
        for start in range(0, total, chunk_size)
    ]

    with Pool(processes=num_workers) as pool:
        with tqdm(total=total, desc="Building class cache", unit="mask") as pbar:
            for done in pool.imap_unordered(class_cache_build_worker, tasks):
                pbar.update(done)

    return path


def ensure_class_cache(n: int, m: int, num_workers=None):
    if class_cache_exists(n=n, m=m):
        print(f"Using existing class cache: {class_cache_path(n)}")
        return class_cache_path(n)

    return build_class_cache_memmap_parallel(n=n, m=m, num_workers=num_workers)


def init_worker_memmap(n, m):
    global GLOBAL_CLASS_CACHE, GLOBAL_M
    GLOBAL_CLASS_CACHE = open_class_cache(n=n, m=m)
    GLOBAL_M = m


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


def monotonicity_worker(args):
    start_mask, end_mask = args

    class_cache = GLOBAL_CLASS_CACHE
    m = GLOBAL_M
    constraints = set()

    for mask in range(start_mask, end_mask):
        id_A = int(class_cache[mask])

        for e_idx in range(m):
            if not ((mask >> e_idx) & 1):
                id_B = int(class_cache[mask | (1 << e_idx)])
                constraints.add((id_A, id_B))

    return constraints


def collect_monotonicity_constraints(n: int, m: int, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    total = 1 << m
    chunk_size = (total + num_workers - 1) // num_workers

    tasks = [
        (worker_id * chunk_size, min(total, (worker_id + 1) * chunk_size))
        for worker_id in range(num_workers)
        if worker_id * chunk_size < total
    ]

    constraints = set()

    with Pool(processes=num_workers, initializer=init_worker_memmap, initargs=(n, m)) as pool:
        for worker_constraints in tqdm(
            pool.imap_unordered(monotonicity_worker, tasks),
            total=len(tasks),
            desc="Collecting monotonicity",
        ):
            constraints.update(worker_constraints)

    print(f"Distinct monotonicity constraints: {len(constraints):,}")
    return constraints


def elementary_submodularity_worker(args):
    start_A, end_A = args

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


def collect_elementary_submodularity_constraints(n: int, m: int, num_workers=None, chunk_size=100_000):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    total = 1 << m

    tasks = [
        (start, min(total, start + chunk_size))
        for start in range(0, total, chunk_size)
    ]

    constraints = set()

    with Pool(processes=num_workers, initializer=init_worker_memmap, initargs=(n, m)) as pool:
        with tqdm(total=total, desc="Collecting submodularity", unit="A") as pbar:
            for done, worker_constraints in pool.imap_unordered(elementary_submodularity_worker, tasks):
                constraints.update(worker_constraints)
                pbar.update(done)

    print(f"Distinct elementary submodularity constraints: {len(constraints):,}")
    return constraints


def write_symmetric_lp(n: int, integer=False, num_workers=None):
    edges = edge_list_kn(n)
    edge_to_index = edge_index_dict(edges)
    m = len(edges)

    reps = load_unlabeled_graphs_of_order(n)
    print(f"Number of unlabeled graphs: {len(reps):,}")

    ensure_class_cache(n=n, m=m, num_workers=num_workers)
    class_cache = open_class_cache(n=n, m=m)

    empty_id = int(class_cache[0])
    full_id = int(class_cache[(1 << m) - 1])

    filename = symmetric_lp_filename(n=n, integer=integer)

    monotonicity = collect_monotonicity_constraints(n=n, m=m, num_workers=num_workers)
    submodularity = collect_elementary_submodularity_constraints(n=n, m=m, num_workers=num_workers)

    print(f"Writing symmetric LP to {filename}")

    constraint_id = 0

    with open(filename, "w", buffering=1024 * 1024 * 64) as f:
        f.write("Maximize\n")
        f.write(f" obj: {var(full_id)}\n\n")

        f.write("Subject To\n")

        f.write(f" c{constraint_id}: {var(empty_id)} = 0\n")
        constraint_id += 1

        for id_A, id_B in tqdm(monotonicity, desc="Writing monotonicity"):
            f.write(f" c{constraint_id}: {var(id_A)} - {var(id_B)} <= 0\n")
            constraint_id += 1

        k5_mask = complete_graph_mask_on_subset(tuple(range(5)), edge_to_index)
        k5_id = int(class_cache[k5_mask])

        f.write(f" c{constraint_id}: {var(k5_id)} = 9\n")
        constraint_id += 1

        for e_idx in range(m):
            if (k5_mask >> e_idx) & 1:
                k5_minus_e = k5_mask ^ (1 << e_idx)
                f.write(f" c{constraint_id}: {var(int(class_cache[k5_minus_e]))} = 9\n")
                constraint_id += 1

        k222_mask = k222_mask_on_partition(A=(0, 1), B=(2, 3), C=(4, 5), edge_to_index=edge_to_index)
        k222_id = int(class_cache[k222_mask])

        f.write(f" c{constraint_id}: {var(k222_id)} = 11\n")
        constraint_id += 1

        for e_idx in range(m):
            if (k222_mask >> e_idx) & 1:
                k222_minus_e = k222_mask ^ (1 << e_idx)
                f.write(f" c{constraint_id}: {var(int(class_cache[k222_minus_e]))} = 11\n")
                constraint_id += 1

        for id_Ae, id_Af, id_A, id_Aef in tqdm(submodularity, desc="Writing submodularity"):
            f.write(
                f" c{constraint_id}: "
                f"{var(id_Ae)} + {var(id_Af)} - {var(id_A)} - {var(id_Aef)} >= 0\n"
            )
            constraint_id += 1

        f.write("\nBounds\n")

        for i, G in enumerate(tqdm(reps, desc="Writing bounds")):
            f.write(f" 0 <= {var(i)} <= {G.number_of_edges()}\n")

        if integer:
            f.write("\nGenerals\n")
            for i in range(len(reps)):
                f.write(f" {var(i)}\n")

        f.write("\nEnd\n")

    print(f"Finished writing {filename}")
    print(f"Total constraints: {constraint_id:,}")
    print(f"File size: {os.path.getsize(filename) / 1024 ** 2:.2f} MB")

    return filename


def solve_lp_with_highs(filename):
    h = highspy.Highs()

    h.setOptionValue("output_flag", True)
    h.setOptionValue("threads", 4)

    print("Reading file...")
    status = h.readModel(str(filename))
    print("Read status:", status)

    # HiGHS may return kWarning for repeated columns or minor LP parser warnings.
    # Do not abort unless it is an actual error.
    if status == highspy.HighsStatus.kError:
        raise RuntimeError(f"HiGHS failed reading {filename}")

    print("Solving...")
    run_status = h.run()
    print("Run status:", run_status)

    model_status = h.getModelStatus()
    info = h.getInfo()

    print("Status:", h.modelStatusToString(model_status))
    print("Objective:", info.objective_function_value)

    return h


def save_highs_values_by_representatives(h, n: int, integer=False):
    lp = h.getLp()
    solution = h.getSolution()

    values_by_class = {}

    for name, value in zip(lp.col_names_, solution.col_value):
        if not name.startswith("r_"):
            continue

        class_id = int(name.split("_")[1])
        values_by_class[class_id] = round(value) if integer else value

    edges = edge_list_kn(n)
    edge_to_index = edge_index_dict(edges)
    reps = load_unlabeled_graphs_of_order(n)
    m = len(edges)

    values = {}

    for i, G in enumerate(reps):
        mask = graph_to_mask(G, edge_to_index)
        values[format(mask, f"0{m}b")] = values_by_class.get(i)

    filename = symmetric_values_filename(n=n, integer=integer)

    with open(filename, "w") as f:
        json.dump(values, f, indent=4)

    print(f"Saved values to {filename}")


def write_and_solve_symmetric_lp(n: int, integer=False, num_workers=None):
    filename = write_symmetric_lp(n=n, integer=integer, num_workers=num_workers)
    h = solve_lp_with_highs(filename)
    save_highs_values_by_representatives(h=h, n=n, integer=integer)
    return h


if __name__ == "__main__":
    write_and_solve_symmetric_lp(n=8, integer=False, num_workers=8)