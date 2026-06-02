import itertools
import json
import os
from multiprocessing import Pool, cpu_count

import highspy
import numpy as np
from tqdm import tqdm


INF = highspy.kHighsInf
GLOBAL_M = None


def edge_list_kn(n: int):
    return list(itertools.combinations(range(n), 2))


def edge_index_dict(edges):
    return {e: i for i, e in enumerate(edges)}


def base_model_filename(n: int):
    return f"nonsymmetric_highs_base_n{n}.mps"


def full_model_filename(n: int):
    return f"nonsymmetric_highs_full_n{n}.mps"


def final_values_filename(n: int):
    return f"nonsymmetric_highs_values_n{n}.json"


def init_worker(m):
    global GLOBAL_M
    GLOBAL_M = m


def complete_graph_mask_on_subset(subset, edge_to_index):
    mask = 0

    for u, v in itertools.combinations(subset, 2):
        mask |= 1 << edge_to_index[tuple(sorted((u, v)))]

    return mask


def k222_mask_on_partition(A, B, C, edge_to_index):
    mask = 0

    for X, Y in [(A, B), (A, C), (B, C)]:
        for x in X:
            for y in Y:
                mask |= 1 << edge_to_index[tuple(sorted((x, y)))]

    return mask


def partitions_222(S):
    S = tuple(S)
    seen = set()

    for A in itertools.combinations(S, 2):
        rem1 = tuple(x for x in S if x not in A)

        for B in itertools.combinations(rem1, 2):
            C = tuple(x for x in rem1 if x not in B)
            parts = tuple(sorted([tuple(sorted(A)), tuple(sorted(B)), tuple(sorted(C))]))

            if parts not in seen:
                seen.add(parts)
                yield parts


def all_k5_masks(n, edge_to_index):
    for S in itertools.combinations(range(n), 5):
        yield complete_graph_mask_on_subset(S, edge_to_index)


def all_k222_masks(n, edge_to_index):
    seen = set()

    for S in itertools.combinations(range(n), 6):
        for A, B, C in partitions_222(S):
            mask = k222_mask_on_partition(A=A, B=B, C=C, edge_to_index=edge_to_index)

            if mask not in seen:
                seen.add(mask)
                yield mask


def add_sparse_rows(h, row_lower, row_upper, starts, indices, values):
    h.addRows(
        len(row_lower),
        row_lower,
        row_upper,
        len(indices),
        starts,
        indices,
        values,
    )


def add_equality_constraint(h, coeffs, rhs):
    starts = np.array([0], dtype=np.int32)
    indices = np.array(list(coeffs.keys()), dtype=np.int32)
    values = np.array(list(coeffs.values()), dtype=np.float64)
    row_lower = np.array([rhs], dtype=np.float64)
    row_upper = np.array([rhs], dtype=np.float64)

    add_sparse_rows(h, row_lower, row_upper, starts, indices, values)


def monotonicity_worker(args):
    start, end = args
    m = GLOBAL_M

    rows = []

    for A in range(start, end):
        for e in range(m):
            if not ((A >> e) & 1):
                Ae = A | (1 << e)
                rows.append((A, Ae))

    num_rows = len(rows)

    row_lower = np.full(num_rows, -INF, dtype=np.float64)
    row_upper = np.zeros(num_rows, dtype=np.float64)
    starts = np.arange(0, 2 * num_rows, 2, dtype=np.int32)

    indices = np.empty(2 * num_rows, dtype=np.int32)
    values = np.empty(2 * num_rows, dtype=np.float64)

    for k, (A, Ae) in enumerate(rows):
        base = 2 * k
        indices[base:base + 2] = [A, Ae]
        values[base:base + 2] = [1.0, -1.0]

    return row_lower, row_upper, starts, indices, values


def add_monotonicity_parallel(h, m: int, num_workers=None, chunk_size=20_000):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    total = 1 << m
    tasks = [(start, min(total, start + chunk_size)) for start in range(0, total, chunk_size)]

    added = 0

    with Pool(processes=num_workers, initializer=init_worker, initargs=(m,)) as pool:
        for row_lower, row_upper, starts, indices, values in tqdm(
            pool.imap_unordered(monotonicity_worker, tasks),
            total=len(tasks),
            desc="Adding monotonicity",
        ):
            add_sparse_rows(h, row_lower, row_upper, starts, indices, values)
            added += len(row_lower)

    print(f"Added monotonicity constraints: {added:,}")
    return added


def submodularity_worker(args):
    start, end = args
    m = GLOBAL_M

    rows = []

    for A in range(start, end):
        missing = [e for e in range(m) if not ((A >> e) & 1)]

        for i in range(len(missing)):
            e = missing[i]
            Ae = A | (1 << e)

            for j in range(i + 1, len(missing)):
                f = missing[j]
                Af = A | (1 << f)
                Aef = Ae | (1 << f)

                rows.append((A, Aef, Ae, Af))

    num_rows = len(rows)

    row_lower = np.full(num_rows, -INF, dtype=np.float64)
    row_upper = np.zeros(num_rows, dtype=np.float64)
    starts = np.arange(0, 4 * num_rows, 4, dtype=np.int32)

    indices = np.empty(4 * num_rows, dtype=np.int32)
    values = np.empty(4 * num_rows, dtype=np.float64)

    for k, (A, Aef, Ae, Af) in enumerate(rows):
        base = 4 * k
        indices[base:base + 4] = [A, Aef, Ae, Af]
        values[base:base + 4] = [1.0, 1.0, -1.0, -1.0]

    return row_lower, row_upper, starts, indices, values


def add_submodularity_parallel(h, m: int, num_workers=None, chunk_size=200):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    total = 1 << m
    tasks = [(start, min(total, start + chunk_size)) for start in range(0, total, chunk_size)]

    added = 0

    with Pool(processes=num_workers, initializer=init_worker, initargs=(m,)) as pool:
        for row_lower, row_upper, starts, indices, values in tqdm(
            pool.imap_unordered(submodularity_worker, tasks),
            total=len(tasks),
            desc="Adding elementary submodularity",
        ):
            add_sparse_rows(h, row_lower, row_upper, starts, indices, values)
            added += len(row_lower)

    print(f"Added elementary submodularity constraints: {added:,}")
    return added


def add_k5_constraints(h, n, edge_to_index):
    added = 0
    m = len(edge_to_index)

    for mask in all_k5_masks(n, edge_to_index):
        add_equality_constraint(h, {mask: 1.0}, 9.0)
        added += 1

        for e in range(m):
            if (mask >> e) & 1:
                add_equality_constraint(h, {mask ^ (1 << e): 1.0}, 9.0)
                added += 1

    print(f"Added K5 and K5-minus-edge constraints: {added:,}")
    return added


def add_k222_constraints(h, n, edge_to_index):
    added = 0
    m = len(edge_to_index)

    for mask in all_k222_masks(n, edge_to_index):
        add_equality_constraint(h, {mask: 1.0}, 11.0)
        added += 1

        for e in range(m):
            if (mask >> e) & 1:
                add_equality_constraint(h, {mask ^ (1 << e): 1.0}, 11.0)
                added += 1

    print(f"Added K222 and K222-minus-edge constraints: {added:,}")
    return added


def build_base_highs_model(n: int, integer=False, num_workers=None):
    edges = edge_list_kn(n)
    edge_to_index = edge_index_dict(edges)
    m = len(edges)

    total = 1 << m
    full_mask = total - 1

    print(f"n={n}, |E(K_n)|={m}, variables={total:,}")

    h = highspy.Highs()
    h.setOptionValue("output_flag", True)

    lower = np.zeros(total, dtype=np.float64)
    upper = np.array([A.bit_count() for A in range(total)], dtype=np.float64)

    h.addVars(total, lower, upper)

    costs = np.zeros(total, dtype=np.float64)
    costs[full_mask] = 1.0

    h.changeColsCost(total, np.arange(total, dtype=np.int32), costs)

    if integer:
        integrality = np.array([highspy.HighsVarType.kInteger] * total, dtype=object)
        h.changeColsIntegrality(total, np.arange(total, dtype=np.int32), integrality)

    h.setMaximize()

    add_equality_constraint(h, {0: 1.0}, 0.0)

    add_monotonicity_parallel(h=h, m=m, num_workers=num_workers)

    add_k5_constraints(h=h, n=n, edge_to_index=edge_to_index)
    add_k222_constraints(h=h, n=n, edge_to_index=edge_to_index)

    h.writeModel(base_model_filename(n))
    print(f"Saved base model: {base_model_filename(n)}")

    return h, edges, full_mask


def load_highs_model(filename):
    h = highspy.Highs()
    h.setOptionValue("output_flag", True)
    h.readModel(filename)
    return h


def save_final_values(h, n: int):
    edges = edge_list_kn(n)
    m = len(edges)
    solution = h.getSolution()

    values = {
        format(mask, f"0{m}b"): solution.col_value[mask]
        for mask in range(1 << m)
    }

    filename = final_values_filename(n)

    with open(filename, "w") as f:
        json.dump(values, f, indent=4)

    print(f"Saved final values: {filename}")


def solve_nonsymmetric_highs(n: int, integer=False, add_submodularity=True, num_workers=None):
    edges = edge_list_kn(n)
    m = len(edges)
    full_mask = (1 << m) - 1

    base_file = base_model_filename(n)
    full_file = full_model_filename(n)

    if add_submodularity and os.path.exists(full_file):
        print(f"Loading full model: {full_file}")
        h = load_highs_model(full_file)

    elif os.path.exists(base_file):
        print(f"Loading base model: {base_file}")
        h = load_highs_model(base_file)

        if add_submodularity:
            add_submodularity_parallel(h=h, m=m, num_workers=num_workers)
            h.writeModel(full_file)
            print(f"Saved full model: {full_file}")

    else:
        h, edges, full_mask = build_base_highs_model(n=n, integer=integer, num_workers=num_workers)

        if add_submodularity:
            add_submodularity_parallel(h=h, m=m, num_workers=num_workers)
            h.writeModel(full_file)
            print(f"Saved full model: {full_file}")

    print("Solving...")
    h.run()

    info = h.getInfo()
    solution = h.getSolution()

    print("Objective:", info.objective_function_value)
    print("Rank full:", solution.col_value[full_mask])

    save_final_values(h=h, n=n)

    return h


if __name__ == "__main__":
    solve_nonsymmetric_highs(n=8, integer=True, add_submodularity=True)