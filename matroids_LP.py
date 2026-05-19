import itertools
import random
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import networkx as nx
import pulp


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


def all_unlabeled_graphs_on_n_vertices(n: int):
    if n > 7:
        raise ValueError("NetworkX graph_atlas_g only works up to n=7.")

    reps = []

    for G in nx.graph_atlas_g():
        if G.number_of_nodes() == n:
            reps.append(nx.convert_node_labels_to_integers(G))

    return reps


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

    reps = all_unlabeled_graphs_on_n_vertices(n)
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
        id_A = class_of_mask(mask)

        for e_idx in range(m):
            if not ((mask >> e_idx) & 1):
                id_B = class_of_mask(mask | (1 << e_idx))
                prob += r[id_A] <= r[id_B]

    print("Adding K5 dependence constraint...")
    k5_mask = complete_graph_mask_on_subset(tuple(range(5)), edge_to_index)
    prob += r[class_of_mask(k5_mask)] <= 9

    print("Adding K222 dependence constraint...")
    k222_mask = k222_mask_on_partition(
        A=(0, 1),
        B=(2, 3),
        C=(4, 5),
        edge_to_index=edge_to_index,
    )
    prob += r[class_of_mask(k222_mask)] <= 11

    prob += r[full_id]

    return prob, r, reps, class_of_mask, m, full_id


def solve_with_random_cuts_parallel(
    n: int,
    rounds: int = 20,
    cuts_per_round: int = 200_000,
    final_check_trials: int = 5_000_000,
    num_workers: int | None = None,
):

    edges = edge_list_kn(n)
    m = len(edges)

    reps = all_unlabeled_graphs_on_n_vertices(n)
    buckets = build_iso_lookup(reps)

    class_of_mask = make_class_id_cached(
        n=n,
        edges=edges,
        buckets=buckets,
    )

    full_mask = (1 << m) - 1
    full_id = class_of_mask(full_mask)

    try:
        prob, r = load_base_model(n)
    except FileNotFoundError:
        save_base_model(n)
        prob, r = load_base_model(n)

    print('Loaded base model successfully')

    class_cache = precompute_class_cache(class_of_mask=class_of_mask, m=m)

    solver = pulp.HiGHS(msg=True)

    for round_idx in range(rounds):
        print()
        print(f"=== Round {round_idx} ===")

        prob.solve(solver)

        print("Status:", pulp.LpStatus[prob.status])
        print("Current objective:", pulp.value(r[full_id]))

        values = {
            i: pulp.value(var)
            for i, var in r.items()
        }

        added = add_random_submodularity_cuts_parallel(
            prob=prob,
            r=r,
            m=m,
            values=values,
            class_cache=class_cache,
            num_trials=cuts_per_round,
            num_workers=num_workers,
            seed=1234567 + 1000 * round_idx,
        )

        print("Added submodularity cuts:", added)

        if added == 0:
            print()
            print("No cuts found. Running final longer check...")

            final_added = add_random_submodularity_cuts_parallel(
                prob=prob,
                r=r,
                m=m,
                values=values,
                class_cache=class_cache,
                num_trials=final_check_trials,
                num_workers=num_workers,
                seed=987654321 + 1000 * round_idx,
            )

            print("Final check added cuts:", final_added)

            if final_added == 0:
                print("Final check found no violated submodularity constraints.")
                break

    print()
    print("Final status:", pulp.LpStatus[prob.status])
    print("Final symmetric LP value:", pulp.value(r[full_id]))

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

    var_dict, prob = pulp.LpProblem.fromJson(filename)

    r = {
        int(name.split("_")[1]): var
        for name, var in var_dict.items()
        if name.startswith("r_")
    }

    return prob, r


if __name__ == "__main__":
    solve_with_random_cuts_parallel(
        n=7,
        rounds=40,
        cuts_per_round=50_000,
        num_workers=8,
    )
