import itertools
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import highspy
import networkx as nx
from tqdm import tqdm

from utils import load_unlabeled_graphs_of_order


MATROIDS_DIR = Path(__file__).resolve().parent


def edge_list_kn(n: int):
    return list(itertools.combinations(range(n), 2))


def edge_index_dict(edges):
    return {e: i for i, e in enumerate(edges)}


def var(i: int):
    return f"r_{i}"


def symmetric_lp_filename(n: int, integer=False):
    kind = "integer" if integer else "lp"
    return MATROIDS_DIR / f"symmetric_full_n{n}_{kind}.lp"


def symmetric_values_filename(n: int, integer=False):
    kind = "integer" if integer else "lp"
    return MATROIDS_DIR / f"symmetric_values_n{n}_{kind}.json"


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


def graph_to_mask(G, edge_to_index):
    mask = 0

    for u, v in G.edges():
        mask |= 1 << edge_to_index[tuple(sorted((u, v)))]

    return mask


def mask_to_graph(n, edges, mask):
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


def class_id_of_graph(G, buckets):
    h = nx.weisfeiler_lehman_graph_hash(G)

    for i, H in buckets[h]:
        if nx.is_isomorphic(G, H):
            return i

    raise ValueError("Graph class not found")


def build_mask_to_class_map(n, reps, edge_to_index):
    result = {}

    for i, G in enumerate(reps):
        result[graph_to_mask(G, edge_to_index)] = i

    return result


def class_id_after_adding_edges(base_graph, edges_to_add, buckets):
    H = base_graph.copy()
    H.add_edges_from(edges_to_add)
    return class_id_of_graph(H, buckets)


def linear_expr(coeffs):
    merged = defaultdict(float)

    for idx, coef in coeffs:
        merged[idx] += coef

    terms = []

    for idx, coef in sorted(merged.items()):
        if abs(coef) < 1e-12:
            continue

        name = var(idx)

        if not terms:
            if coef == 1:
                terms.append(name)
            elif coef == -1:
                terms.append(f"- {name}")
            else:
                terms.append(f"{coef:g} {name}")
        else:
            if coef == 1:
                terms.append(f"+ {name}")
            elif coef == -1:
                terms.append(f"- {name}")
            elif coef > 0:
                terms.append(f"+ {coef:g} {name}")
            else:
                terms.append(f"- {-coef:g} {name}")

    return " ".join(terms) if terms else "0"


def collect_constraints_from_reps(n, reps, buckets):
    edges = edge_list_kn(n)
    all_edges = {tuple(sorted(e)) for e in edges}

    monotonicity = set()
    submodularity = set()

    for id_A, G in enumerate(tqdm(reps, desc="Collecting constraints from reps")):
        present = {tuple(sorted(e)) for e in G.edges()}
        missing = sorted(all_edges - present)

        one_added = {}

        for e in missing:
            id_Ae = class_id_after_adding_edges(G, [e], buckets)
            one_added[e] = id_Ae
            monotonicity.add((id_A, id_Ae))

        for e, f in itertools.combinations(missing, 2):
            id_Ae = one_added[e]
            id_Af = one_added[f]
            id_Aef = class_id_after_adding_edges(G, [e, f], buckets)

            left_1, left_2 = sorted((id_Ae, id_Af))
            submodularity.add((left_1, left_2, id_A, id_Aef))

    print(f"Distinct monotonicity constraints: {len(monotonicity):,}")
    print(f"Distinct elementary submodularity constraints: {len(submodularity):,}")

    return monotonicity, submodularity


def write_symmetric_lp(n: int, integer=False):
    edges = edge_list_kn(n)
    edge_to_index = edge_index_dict(edges)

    reps = load_unlabeled_graphs_of_order(n)
    buckets = build_iso_lookup(reps)

    print(f"Number of unlabeled graphs: {len(reps):,}")

    empty = nx.empty_graph(n)
    full = nx.complete_graph(n)

    empty_id = class_id_of_graph(empty, buckets)
    full_id = class_id_of_graph(full, buckets)

    monotonicity, submodularity = collect_constraints_from_reps(n=n, reps=reps, buckets=buckets)

    filename = symmetric_lp_filename(n=n, integer=integer)
    constraint_id = 0

    with open(filename, "w", buffering=1024 * 1024 * 64) as f:
        f.write("Maximize\n")
        f.write(f" obj: {var(full_id)}\n\n")

        f.write("Subject To\n")

        f.write(f" c{constraint_id}: {var(empty_id)} = 0\n")
        constraint_id += 1

        for id_A, id_B in tqdm(monotonicity, desc="Writing monotonicity"):
            expr = linear_expr([(id_A, 1.0), (id_B, -1.0)])

            if expr != "0":
                f.write(f" c{constraint_id}: {expr} <= 0\n")
                constraint_id += 1

        k5_mask = complete_graph_mask_on_subset(tuple(range(5)), edge_to_index)
        k5_graph = mask_to_graph(n, edges, k5_mask)
        k5_id = class_id_of_graph(k5_graph, buckets)

        f.write(f" c{constraint_id}: {var(k5_id)} = 9\n")
        constraint_id += 1

        for e_idx in range(len(edges)):
            if (k5_mask >> e_idx) & 1:
                k5_minus_e = k5_mask ^ (1 << e_idx)
                k5_minus_graph = mask_to_graph(n, edges, k5_minus_e)
                k5_minus_id = class_id_of_graph(k5_minus_graph, buckets)
                f.write(f" c{constraint_id}: {var(k5_minus_id)} = 9\n")
                constraint_id += 1

        k222_mask = k222_mask_on_partition(A=(0, 1), B=(2, 3), C=(4, 5), edge_to_index=edge_to_index)
        k222_graph = mask_to_graph(n, edges, k222_mask)
        k222_id = class_id_of_graph(k222_graph, buckets)

        f.write(f" c{constraint_id}: {var(k222_id)} = 11\n")
        constraint_id += 1

        for e_idx in range(len(edges)):
            if (k222_mask >> e_idx) & 1:
                k222_minus_e = k222_mask ^ (1 << e_idx)
                k222_minus_graph = mask_to_graph(n, edges, k222_minus_e)
                k222_minus_id = class_id_of_graph(k222_minus_graph, buckets)
                f.write(f" c{constraint_id}: {var(k222_minus_id)} = 11\n")
                constraint_id += 1

        for id_Ae, id_Af, id_A, id_Aef in tqdm(submodularity, desc="Writing submodularity"):
            expr = linear_expr([
                (id_Ae, 1.0),
                (id_Af, 1.0),
                (id_A, -1.0),
                (id_Aef, -1.0),
            ])

            if expr != "0":
                f.write(f" c{constraint_id}: {expr} >= 0\n")
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


def write_and_solve_symmetric_lp(n: int, integer=False):
    filename = write_symmetric_lp(n=n, integer=integer)
    h = solve_lp_with_highs(filename)
    save_highs_values_by_representatives(h=h, n=n, integer=integer)
    return h


if __name__ == "__main__":
    write_and_solve_symmetric_lp(n=8, integer=False)