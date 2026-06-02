import itertools
import os
from tqdm import tqdm
import cplex


def edge_list_kn(n: int):
    return list(itertools.combinations(range(n), 2))


def edge_index_dict(edges):
    return {e: i for i, e in enumerate(edges)}


def lp_filename(n: int):
    return f"nonsymmetric_full_n{n}.lp"


def var(mask: int):
    return f"r_{mask}"


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


def write_full_lp(n: int, integer=True):
    edges = edge_list_kn(n)
    edge_to_index = edge_index_dict(edges)
    m = len(edges)

    total = 1 << m
    full_mask = total - 1
    filename = lp_filename(n)

    print(f"Writing full nonsymmetric model to {filename}")
    print(f"n={n}, m={m}, variables={total:,}")

    constraint_id = 0

    with open(filename, "w", buffering=1024 * 1024 * 64) as f:
        f.write("Maximize\n")
        f.write(f" obj: {var(full_mask)}\n\n")

        f.write("Subject To\n")

        f.write(f" c{constraint_id}: {var(0)} = 0\n")
        constraint_id += 1

        print("Writing monotonicity constraints...")
        for A in tqdm(range(total), desc="Monotonicity"):
            for e in range(m):
                if not ((A >> e) & 1):
                    Ae = A | (1 << e)
                    f.write(f" c{constraint_id}: {var(A)} - {var(Ae)} <= 0\n")
                    constraint_id += 1

        print("Writing K5 constraints...")
        for mask in tqdm(list(all_k5_masks(n, edge_to_index)), desc="K5"):
            f.write(f" c{constraint_id}: {var(mask)} = 9\n")
            constraint_id += 1

            for e in range(m):
                if (mask >> e) & 1:
                    f.write(f" c{constraint_id}: {var(mask ^ (1 << e))} = 9\n")
                    constraint_id += 1

        print("Writing K222 constraints...")
        for mask in tqdm(list(all_k222_masks(n, edge_to_index)), desc="K222"):
            f.write(f" c{constraint_id}: {var(mask)} = 11\n")
            constraint_id += 1

            for e in range(m):
                if (mask >> e) & 1:
                    f.write(f" c{constraint_id}: {var(mask ^ (1 << e))} = 11\n")
                    constraint_id += 1

        print("Writing elementary submodularity constraints...")
        for A in tqdm(range(total), desc="Submodularity"):
            missing = [e for e in range(m) if not ((A >> e) & 1)]

            for i in range(len(missing)):
                e = missing[i]
                Ae = A | (1 << e)

                for j in range(i + 1, len(missing)):
                    f_idx = missing[j]
                    Af = A | (1 << f_idx)
                    Aef = Ae | (1 << f_idx)

                    f.write(
                        f" c{constraint_id}: "
                        f"{var(A)} + {var(Aef)} - {var(Ae)} - {var(Af)} <= 0\n"
                    )
                    constraint_id += 1

        f.write("\nBounds\n")

        print("Writing bounds...")
        for A in tqdm(range(total), desc="Bounds"):
            f.write(f" 0 <= {var(A)} <= {A.bit_count()}\n")

        if integer:
            f.write("\nGenerals\n")
            for A in tqdm(range(total), desc="Integer variables"):
                f.write(f" {var(A)}\n")

        f.write("\nEnd\n")

    print(f"Finished writing {filename}")
    print(f"Total constraints: {constraint_id:,}")
    print(f"File size: {os.path.getsize(filename) / 1024 ** 3:.2f} GB")

    return filename


def solve_lp_with_cplex(filename):
    cpx = cplex.Cplex()
    cpx.parameters.workdir.set(r"C:\Oren\Other\cplex_work")
    cpx.parameters.threads.set(4)
    cpx.parameters.lpmethod.set(cpx.parameters.lpmethod.values.barrier)
    cpx.parameters.barrier.crossover.set(0)

    print('Reading file...')
    cpx.read(filename)
    print('Solving...')
    cpx.solve()

    print("Status:", cpx.solution.get_status_string())
    print("Objective:", cpx.solution.get_objective_value())

    return cpx

import highspy


def solve_lp_with_highs(filename):
    h = highspy.Highs()

    h.setOptionValue("output_flag", True)
    h.setOptionValue("threads", 4)

    print("Reading file...")
    status = h.readModel(filename)

    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"Failed reading {filename}")

    print("Solving...")
    h.run()

    model_status = h.getModelStatus()
    info = h.getInfo()

    print("Status:", h.modelStatusToString(model_status))
    print("Objective:", info.objective_function_value)

    return h


if __name__ == "__main__":
    n = 7
    write_full_lp(n=n)
    solve_lp_with_highs(f"nonsymmetric_full_n{n}.lp")

# if __name__ == "__main__":
#