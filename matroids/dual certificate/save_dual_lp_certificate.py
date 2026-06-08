import json
import os
from pathlib import Path
import sys

import highspy
import networkx as nx
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from matroids.symmetric_write import *


def graph_to_mask(G, edge_to_index):
    mask = 0

    for u, v in G.edges():
        if u > v:
            u, v = v, u
        mask |= 1 << edge_to_index[(u, v)]

    return mask


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
    constraint_masks_filename = str(filename) + ".constraint_masks.json"
    variable_masks_filename = str(filename) + ".variable_masks.json"

    constraint_id = 0
    constraint_masks = {}
    variable_masks = {}

    for i, G in enumerate(reps):
        variable_masks[str(i)] = {
            "mask": int(graph_to_mask(G, edge_to_index)),
            "upper_bound": int(G.number_of_edges()),
            "variable": var(i),
        }

    with open(filename, "w", buffering=1024 * 1024 * 64) as f:
        f.write("Maximize\n")
        f.write(f" obj: {var(full_id)}\n\n")

        f.write("Subject To\n")

        f.write(f" c{constraint_id}: {var(empty_id)} = 0\n")
        constraint_masks[str(constraint_id)] = {
            "type": "empty_rank",
            "mask": int(graph_to_mask(reps[empty_id], edge_to_index)),
        }
        constraint_id += 1

        for id_A, id_B in tqdm(monotonicity, desc="Writing monotonicity"):
            expr = linear_expr([(id_A, 1.0), (id_B, -1.0)])

            if expr != "0":
                f.write(f" c{constraint_id}: {expr} <= 0\n")
                constraint_masks[str(constraint_id)] = {
                    "type": "monotonicity",
                    "small_mask": int(graph_to_mask(reps[id_A], edge_to_index)),
                    "large_mask": int(graph_to_mask(reps[id_B], edge_to_index)),
                }
                constraint_id += 1

        k5_mask = complete_graph_mask_on_subset(tuple(range(5)), edge_to_index)
        k5_graph = mask_to_graph(n, edges, k5_mask)
        k5_id = class_id_of_graph(k5_graph, buckets)

        f.write(f" c{constraint_id}: {var(k5_id)} = 9\n")
        constraint_masks[str(constraint_id)] = {
            "type": "K5",
            "mask": int(graph_to_mask(reps[k5_id], edge_to_index)),
            "original_labeled_mask": int(k5_mask),
        }
        constraint_id += 1

        for e_idx in range(len(edges)):
            if (k5_mask >> e_idx) & 1:
                k5_minus_e = k5_mask ^ (1 << e_idx)
                k5_minus_graph = mask_to_graph(n, edges, k5_minus_e)
                k5_minus_id = class_id_of_graph(k5_minus_graph, buckets)

                f.write(f" c{constraint_id}: {var(k5_minus_id)} = 9\n")
                constraint_masks[str(constraint_id)] = {
                    "type": "K5_minus_edge",
                    "mask": int(graph_to_mask(reps[k5_minus_id], edge_to_index)),
                    "original_labeled_mask": int(k5_minus_e),
                    "removed_edge_index": int(e_idx),
                    "removed_edge": list(edges[e_idx]),
                }
                constraint_id += 1

        k222_mask = k222_mask_on_partition(A=(0, 1), B=(2, 3), C=(4, 5), edge_to_index=edge_to_index)
        k222_graph = mask_to_graph(n, edges, k222_mask)
        k222_id = class_id_of_graph(k222_graph, buckets)

        f.write(f" c{constraint_id}: {var(k222_id)} = 11\n")
        constraint_masks[str(constraint_id)] = {
            "type": "K222",
            "mask": int(graph_to_mask(reps[k222_id], edge_to_index)),
            "original_labeled_mask": int(k222_mask),
        }
        constraint_id += 1

        for e_idx in range(len(edges)):
            if (k222_mask >> e_idx) & 1:
                k222_minus_e = k222_mask ^ (1 << e_idx)
                k222_minus_graph = mask_to_graph(n, edges, k222_minus_e)
                k222_minus_id = class_id_of_graph(k222_minus_graph, buckets)

                f.write(f" c{constraint_id}: {var(k222_minus_id)} = 11\n")
                constraint_masks[str(constraint_id)] = {
                    "type": "K222_minus_edge",
                    "mask": int(graph_to_mask(reps[k222_minus_id], edge_to_index)),
                    "original_labeled_mask": int(k222_minus_e),
                    "removed_edge_index": int(e_idx),
                    "removed_edge": list(edges[e_idx]),
                }
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
                constraint_masks[str(constraint_id)] = {
                    "type": "submodularity",
                    "Ae_mask": int(graph_to_mask(reps[id_Ae], edge_to_index)),
                    "Af_mask": int(graph_to_mask(reps[id_Af], edge_to_index)),
                    "A_mask": int(graph_to_mask(reps[id_A], edge_to_index)),
                    "Aef_mask": int(graph_to_mask(reps[id_Aef], edge_to_index)),
                }
                constraint_id += 1

        f.write("\nBounds\n")

        for i, G in enumerate(tqdm(reps, desc="Writing bounds")):
            f.write(f" 0 <= {var(i)} <= {G.number_of_edges()}\n")

        if integer:
            f.write("\nGenerals\n")
            for i in range(len(reps)):
                f.write(f" {var(i)}\n")

        f.write("\nEnd\n")

    with open(constraint_masks_filename, "w") as f:
        json.dump(constraint_masks, f, indent=2)

    with open(variable_masks_filename, "w") as f:
        json.dump(variable_masks, f, indent=2)

    print(f"Finished writing {filename}")
    print(f"Finished writing {constraint_masks_filename}")
    print(f"Finished writing {variable_masks_filename}")
    print(f"Total constraints: {constraint_id:,}")
    print(f"File size: {os.path.getsize(filename) / 1024 ** 2:.2f} MB")

    return filename, constraint_masks_filename, variable_masks_filename


def _strip_prefix(name, prefix):
    name = str(name)
    return name[len(prefix):] if name.startswith(prefix) else name


def save_dual_values(h, out_file, eps=1e-8):
    sol = h.getSolution()
    lp = h.getLp()

    row_duals = {}

    for i, y in enumerate(sol.row_dual):
        if abs(y) > eps:
            row_name = lp.row_names_[i]  # e.g. "c6638"
            row_id = _strip_prefix(row_name, "c")
            row_duals[str(row_id)] = float(y)

    col_duals = {}

    for i, y in enumerate(sol.col_dual):
        if abs(y) > eps:
            col_name = lp.col_names_[i]  # e.g. "r1043" or "r_1043"
            col_id = col_name.replace("r_", "").replace("r", "")
            col_duals[str(col_id)] = float(y)

    dual_values = {
        "row_duals": row_duals,
        "col_duals": col_duals,
    }

    with open(out_file, "w") as f:
        json.dump(dual_values, f, indent=2)

    print(f"Saved {len(row_duals)} nonzero row duals")
    print(f"Saved {len(col_duals)} nonzero column duals")
    print(f"Saved dual values to {out_file}")


def solve_lp_and_save_duals(filename, out_dir=None, eps=1e-8):
    filename = Path(filename)

    if out_dir is None:
        out_dir = filename.parent

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h = highspy.Highs()
    h.setOptionValue("output_flag", True)
    h.setOptionValue("threads", 4)
    h.setOptionValue("solver", "ipm")
    h.setOptionValue("simplex_strategy", 1)

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

    h.writeSolution(str(out_dir / "solution.pretty"), 1)
    h.writeSolution(str(out_dir / "solution.raw"), 0)
    h.writeBasis(str(out_dir / "basis.bas"))

    dual_values_file = out_dir / (filename.name + ".dual_values.json")

    save_dual_values(h=h, out_file=dual_values_file, eps=eps)

    return h, dual_values_file


def combine_constraint_masks_and_duals(constraint_masks_file, variable_masks_file, dual_values_file, out_file):
    with open(constraint_masks_file, "r") as f:
        constraint_masks = json.load(f)

    with open(variable_masks_file, "r") as f:
        variable_masks = json.load(f)

    with open(dual_values_file, "r") as f:
        dual_values = json.load(f)

    row_duals = dual_values["row_duals"]
    col_duals = dual_values["col_duals"]

    combined = {
        "row_terms": {},
        "column_bound_terms": {},
    }

    for row, value in row_duals.items():
        if row not in constraint_masks:
            raise KeyError(f"Dual row {row} does not appear in constraint_masks")

        combined["row_terms"][row] = {
            "dual_value": value,
            "constraint": constraint_masks[row],
        }

    for col, value in col_duals.items():
        if col not in variable_masks:
            raise KeyError(f"Column {col} does not appear in variable_masks")

        combined["column_bound_terms"][col] = {
            "dual_value": value,
            "variable": variable_masks[col],
        }

    with open(out_file, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Saved combined certificate to {out_file}")
    print(f"Number of active row constraints: {len(combined['row_terms'])}")
    print(f"Number of active column bounds: {len(combined['column_bound_terms'])}")

    return combined


if __name__ == "__main__":
    n = 6
    # filename, constraint_masks_filename, variable_masks_filename = write_symmetric_lp(n=n, integer=False)
    filename = symmetric_lp_filename(n=n, integer=False)
    constraint_masks_filename = str(filename) + ".constraint_masks.json"
    variable_masks_filename = str(filename) + ".variable_masks.json"
    h, dual_values_file = solve_lp_and_save_duals(
        filename=filename,
        out_dir=f"matroids/dual_output/certificate_n{n}",
    )

    combine_constraint_masks_and_duals(
        constraint_masks_file=constraint_masks_filename,
        variable_masks_file=variable_masks_filename,
        dual_values_file=dual_values_file,
        out_file=f"matroids/dual_output/certificate_n{n}/combined_certificate.json",
    )