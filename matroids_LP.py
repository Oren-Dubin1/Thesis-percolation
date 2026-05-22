import itertools
import json
import random
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import networkx as nx
import pulp
from utils import *


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

    raise ValueError("Graph class not found.")


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


def all_k222_masks(n: int, edge_to_index):
    masks = set()

    for subset in itertools.combinations(range(n), 6):
        subset = tuple(subset)

        for A in itertools.combinations(subset, 2):
            rest_after_A = tuple(x for x in subset if x not in A)

            for B in itertools.combinations(rest_after_A, 2):
                C = tuple(x for x in rest_after_A if x not in B)

                parts = tuple(sorted([
                    tuple(sorted(A)),
                    tuple(sorted(B)),
                    tuple(sorted(C)),
                ]))

                if tuple(sorted(A)) != parts[0]:
                    continue

                masks.add(k222_mask_on_partition(parts[0], parts[1], parts[2], edge_to_index))

    return masks


def precompute_class_cache(class_of_mask, m: int):
    print("Precomputing class IDs for all masks...")
    return [class_of_mask(mask) for mask in range(1 << m)]


def find_violated_cuts_worker(args):
    worker_id, trials, m, values, class_cache, seed = args

    rng = random.Random(seed + worker_id)
    found = set()

    for _ in range(trials):
        A = rng.randrange(1 << m)
        B = rng.randrange(1 << m)

        id_A = class_cache[A]
        id_B = class_cache[B]
        id_U = class_cache[A | B]
        id_I = class_cache[A & B]

        if values[id_A] + values[id_B] + 1e-7 < values[id_U] + values[id_I]:
            cut_key = tuple(sorted((id_A, id_B)) + [id_U, id_I])
            found.add(cut_key)

    return found


def add_random_submodularity_cuts_parallel(
    prob,
    r,
    m: int,
    values,
    class_cache,
    num_trials: int,
    num_workers: int | None = None,
    seed: int = 1234567,
):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    trials_per_worker = num_trials // num_workers
    remainder = num_trials % num_workers

    tasks = []

    for worker_id in range(num_workers):
        trials = trials_per_worker + (1 if worker_id < remainder else 0)
        tasks.append((worker_id, trials, m, values, class_cache, seed))

    with Pool(processes=num_workers) as pool:
        results = pool.map(find_violated_cuts_worker, tasks)

    violated = set()

    for cuts in results:
        violated.update(cuts)

    added = 0

    for id_A, id_B, id_U, id_I in violated:
        prob += r[id_A] + r[id_B] >= r[id_U] + r[id_I]
        added += 1

    return added


def build_base_problem(n: int):
    edges = edge_list_kn(n)
    edge_to_index = edge_index_dict(edges)
    m = len(edges)

    print(f"n={n}, |E(K_n)|={m}")

    reps = load_unlabeled_graphs_of_order(n)
    buckets = build_iso_lookup(reps)

    print(f"Number of unlabeled graphs: {len(reps)}")

    class_of_mask = make_class_id_cached(n=n, edges=edges, buckets=buckets)

    prob = pulp.LpProblem("symmetric_polymatroid_rank", pulp.LpMaximize)

    r = {
        i: pulp.LpVariable(f"r_{i}", lowBound=0, cat=pulp.LpInteger)
        for i in range(len(reps))
    }

    empty_mask = 0
    full_mask = (1 << m) - 1

    empty_id = class_of_mask(empty_mask)
    full_id = class_of_mask(full_mask)

    prob += r[empty_id] == 0

    print("Adding size bounds...")
    for i, G in enumerate(reps):
        prob += r[i] <= G.number_of_edges()

    print("Adding monotonicity constraints...")
    for mask in range(1 << m):
        if mask % 10000 == 0:
            print(f'Added {mask}/{(1 << m) * m} monotonicity constraints...')
        id_A = class_of_mask(mask)

        for e_idx in range(m):
            if not ((mask >> e_idx) & 1):
                id_B = class_of_mask(mask | (1 << e_idx))
                prob += r[id_A] <= r[id_B]

    print("Adding K5 circuit constraint...")

    k5_mask = complete_graph_mask_on_subset(tuple(range(5)), edge_to_index)
    k5_id = class_of_mask(k5_mask)

    # K5 has 10 edges. Circuit means rank exactly 9.
    prob += r[k5_id] == 9

    for e_idx in range(m):
        if (k5_mask >> e_idx) & 1:
            k5_minus_e = k5_mask ^ (1 << e_idx)
            prob += r[class_of_mask(k5_minus_e)] == 9

    print("Adding K222 circuit constraint...")

    k222_mask = k222_mask_on_partition(A=(0, 1), B=(2, 3), C=(4, 5), edge_to_index=edge_to_index)
    k222_id = class_of_mask(k222_mask)

    # K222 has 12 edges. Circuit means rank exactly 11.
    prob += r[k222_id] == 11

    for e_idx in range(m):
        if (k222_mask >> e_idx) & 1:
            k222_minus_e = k222_mask ^ (1 << e_idx)
            prob += r[class_of_mask(k222_minus_e)] == 11

    prob += r[full_id]

    return prob, r, reps, class_of_mask, m, full_id


def mask_to_binary(mask: int, m: int):
    return format(mask, f"0{m}b")


def save_values_by_binary_graph(n, reps, class_of_mask, m, r, filename):
    values = {}
    print(f"Saving ranks by binary graph to {filename}...")
    for mask in range(1 << m):
        class_id = class_of_mask(mask)
        rank = pulp.value(r[class_id])

        values[mask_to_binary(mask, m)] = round(rank)

    with open(filename, "w") as f:
        json.dump(values, f, indent=4)

    print(f"Saved ranks by binary graph to {filename}")


def add_all_elementary_submodularity_constraints(prob, r, m: int, class_cache):
    print("Adding all elementary submodularity constraints up to symmetry...")

    constraints = set()
    total = 1 << m

    for A in range(total):
        if A % 100_000 == 0:
            print(f"Processed {A}/{total}")

        id_A = class_cache[A]
        missing = [e for e in range(m) if not ((A >> e) & 1)]

        for i in range(len(missing)):
            e = missing[i]
            Ae = A | (1 << e)
            id_Ae = class_cache[Ae]

            for j in range(i + 1, len(missing)):
                f = missing[j]
                Af = A | (1 << f)
                Aef = Ae | (1 << f)

                id_Af = class_cache[Af]
                id_Aef = class_cache[Aef]

                left = tuple(sorted((id_Ae, id_Af)))
                constraints.add((left[0], left[1], id_A, id_Aef))

    print(f"Distinct elementary submodularity constraints: {len(constraints)}")

    for id_Ae, id_Af, id_A, id_Aef in constraints:
        prob += r[id_Ae] + r[id_Af] >= r[id_A] + r[id_Aef]

    return len(constraints)


def elementary_submodularity_worker(args):
    start_A, end_A, m, class_cache, worker_id = args

    constraints = set()

    for A in range(start_A, end_A):
        if (A - start_A) % 100_000 == 0:
            print(f"Worker {worker_id}: processed A={A}/{end_A}")

        id_A = class_cache[A]
        missing = [e for e in range(m) if not ((A >> e) & 1)]

        for i in range(len(missing)):
            e = missing[i]
            Ae = A | (1 << e)
            id_Ae = class_cache[Ae]

            for j in range(i + 1, len(missing)):
                f = missing[j]
                Af = A | (1 << f)
                Aef = Ae | (1 << f)

                id_Af = class_cache[Af]
                id_Aef = class_cache[Aef]

                left_1, left_2 = sorted((id_Ae, id_Af))
                constraints.add((left_1, left_2, id_A, id_Aef))

    return constraints


def add_all_elementary_submodularity_constraints_parallel(
    prob,
    r,
    m: int,
    class_cache,
    num_workers: int | None = None,
):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"Generating elementary submodularity constraints using {num_workers} workers...")

    total = 1 << m
    chunk_size = (total + num_workers - 1) // num_workers

    tasks = []

    for worker_id in range(num_workers):
        start_A = worker_id * chunk_size
        end_A = min(total, start_A + chunk_size)

        if start_A < end_A:
            tasks.append((start_A, end_A, m, class_cache, worker_id))

    with Pool(processes=num_workers) as pool:
        results = pool.map(elementary_submodularity_worker, tasks)

    constraints = set()

    for worker_constraints in results:
        constraints.update(worker_constraints)

    print(f"Distinct elementary submodularity constraints: {len(constraints)}")
    print("Adding constraints to PuLP model...")

    for id_Ae, id_Af, id_A, id_Aef in constraints:
        prob += r[id_Ae] + r[id_Af] >= r[id_A] + r[id_Aef]

    return len(constraints)

def solve_with_all_elementary_submodularity(
    n: int,
    num_workers: int | None = None,
):
    edges = edge_list_kn(n)
    m = len(edges)

    reps = load_unlabeled_graphs_of_order(n)
    buckets = build_iso_lookup(reps)

    class_of_mask = make_class_id_cached(n=n, edges=edges, buckets=buckets)

    full_mask = (1 << m) - 1
    full_id = class_of_mask(full_mask)

    try:
        prob, r = load_base_model(n)
    except FileNotFoundError:
        save_base_model(n)
        prob, r = load_base_model(n)

    print("Loaded base model successfully")

    class_cache = precompute_class_cache(class_of_mask=class_of_mask, m=m)

    added = add_all_elementary_submodularity_constraints_parallel(
        prob=prob,
        r=r,
        m=m,
        class_cache=class_cache,
        num_workers=num_workers,
    )

    print("Added elementary submodularity constraints:", added)

    solver = pulp.HiGHS(msg=True)

    print("Solving...")
    prob.solve(solver)

    print("Status:", pulp.LpStatus[prob.status])
    print("Final symmetric LP value:", pulp.value(r[full_id]))

    save_values_by_binary_graph(
        n=n,
        reps=reps,
        class_of_mask=class_of_mask,
        m=m,
        r=r,
        filename=f"final_values_n{n}_by_binary_graph.json",
    )

    return prob, r, reps

def save_base_model(n: int, filename: str | None = None):
    if filename is None:
        filename = f"base_model_n{n}.json"

    prob, r, reps, class_of_mask, m, full_id = build_base_problem(n)
    prob.toJson(filename)

    print(f"Saved base model to {filename}")


def load_base_model(n: int, filename: str | None = None):
    if filename is None:
        filename = f"base_model_n{n}.json"

    print(f'Loading base model from {filename}...')
    var_dict, prob = pulp.LpProblem.fromJson(filename)

    r = {
        int(name.split("_")[1]): var
        for name, var in var_dict.items()
        if name.startswith("r_")
    }

    return prob, r

if __name__ == "__main__":
    solve_with_all_elementary_submodularity(6)