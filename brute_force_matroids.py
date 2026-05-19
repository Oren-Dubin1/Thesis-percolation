import networkx as nx
import igraph as ig

from itertools import combinations, permutations


def all_edge_sets_of_size(edges, r):
    for subset in combinations(edges, r):
        yield tuple(sorted(subset))


def contains_K5(G):
    for S in combinations(G.nodes(), 5):
        H = G.subgraph(S)

        if H.number_of_edges() == 10:
            return True

    return False


def contains_K222(G):
    template = nx.complete_multipartite_graph(2, 2, 2)

    GM = nx.algorithms.isomorphism.GraphMatcher(G, template)

    try:
        next(GM.subgraph_monomorphisms_iter())
        return True
    except StopIteration:
        return False


def satisfies_matroid_axiom(bases):
    if not bases:
        return False

    for B1 in bases:
        for B2 in bases:
            for e in B1 - B2:
                found = False

                for f in B2 - B1:
                    candidate = (B1 - {e}) | {f}

                    if candidate in bases:
                        found = True
                        break

                if not found:
                    return False

    return True


def generate_orbits_of_r_edge_graphs(n, r):
    edges = list(combinations(range(n), 2))

    representatives = []
    representative_graphs = []

    for subset in combinations(edges, r):
        G = ig.Graph(n=n, edges=list(subset))

        already_seen = False

        for H in representative_graphs:
            if G.isomorphic(H):
                already_seen = True
                break

        if already_seen:
            continue

        representatives.append(set(subset))
        representative_graphs.append(G)

    return representatives


def expand_orbit(n, edge_set):
    vertices = list(range(n))

    orbit = set()

    for perm in permutations(vertices):
        transformed = []

        for u, v in edge_set:
            a = perm[u]
            b = perm[v]

            if a > b:
                a, b = b, a

            transformed.append((a, b))

        orbit.add(frozenset(transformed))

    return orbit


def brute_force_matroids(n, min_rank):
    vertices = list(range(n))
    m = n * (n - 1) // 2

    for rank in range(min_rank, m + 1):
        print(f"Checking rank {rank}")

        orbit_reps = generate_orbits_of_r_edge_graphs(n, rank)

        valid_orbits = []

        for B in orbit_reps:
            G = nx.Graph()
            G.add_nodes_from(vertices)
            G.add_edges_from(B)

            if contains_K5(G) or contains_K222(G):
                continue

            valid_orbits.append(B)

        print(f"Valid basis orbits: {len(valid_orbits)}")

        total = 1 << len(valid_orbits)
        print(f"Checking {total} orbit-subsets")

        expanded_orbits = [expand_orbit(n, rep) for rep in valid_orbits]

        for mask in range(total):
            chosen_reps = []
            bases = set()

            for i in range(len(valid_orbits)):
                if (mask >> i) & 1:
                    chosen_reps.append(valid_orbits[i])
                    bases |= expanded_orbits[i]

            if not bases:
                continue

            if not satisfies_matroid_axiom(bases):
                continue

            print("=" * 80)
            print(f"FOUND MATROID OF RANK {rank}")
            print(f"Number of bases: {len(bases)}")
            print(f"Number of basis orbits: {len(chosen_reps)}")
            print("Basis orbit representatives:")

            for rep in chosen_reps:
                print(sorted(rep))


# if __name__ == "__main__":
#     n = 7
#     min_rank = 15
#     brute_force_matroids(n=n, min_rank=min_rank)

import itertools
import numpy as np
import galois


def ff_rank(M):
    R = M.row_reduce()
    return sum(bool(np.any(row != 0)) for row in R)


def pair_partitions(vertices):
    vertices = list(vertices)

    if not vertices:
        yield []
        return

    a = vertices[0]

    for i in range(1, len(vertices)):
        b = vertices[i]
        rest = vertices[1:i] + vertices[i + 1:]

        for pp in pair_partitions(rest):
            yield [(a, b)] + pp


def k222_edges(parts):
    edges = []

    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    edges.append(tuple(sorted((u, v))))

    return sorted(edges)


def all_k5_edge_sets(n):
    for S in itertools.combinations(range(n), 5):
        yield [
            tuple(sorted(e))
            for e in itertools.combinations(S, 2)
        ]


def all_k222_edge_sets(n):
    for S in itertools.combinations(range(n), 6):
        for parts in pair_partitions(S):
            yield k222_edges(parts)


def random_rank_matrix(F, rank, m):
    """
    Returns a rank x m matrix over F, hopefully of full row-rank.
    Columns are the edge vectors.
    """

    while True:
        M = F.Random((rank, m))

        if ff_rank(M) == rank:
            return M


def check_matrix(M, edge_to_index, forbidden_sets):
    """
    M has columns indexed by edges.
    A forbidden set is good iff its columns are dependent.
    """

    for edge_set in forbidden_sets:
        cols = [edge_to_index[e] for e in edge_set]
        submatrix = M[:, cols]

        if ff_rank(submatrix.T) == len(edge_set):
            return False, edge_set

    return True, None


def search_linear_matroid(n=7, rank=15, q=2, trials=10000, require_k5=True, require_k222=True):
    F = galois.GF(q)

    edges = [
        tuple(sorted(e))
        for e in itertools.combinations(range(n), 2)
    ]

    edge_to_index = {
        e: i
        for i, e in enumerate(edges)
    }

    forbidden_sets = []

    if require_k5:
        forbidden_sets.extend(list(all_k5_edge_sets(n)))

    if require_k222:
        forbidden_sets.extend(list(all_k222_edge_sets(n)))

    print("n:", n)
    print("field: GF(" + str(q) + ")")
    print("number of edges:", len(edges))
    print("target rank:", rank)
    print("number of forbidden sets:", len(forbidden_sets))

    m = len(edges)

    for trial in range(trials):
        M = random_rank_matrix(F=F, rank=rank, m=m)

        ok, bad_set = check_matrix(
            M=M,
            edge_to_index=edge_to_index,
            forbidden_sets=forbidden_sets
        )

        if trial % 100 == 0:
            print("trial", trial)

        if ok:
            print("=" * 80)
            print("FOUND")
            print("rank:", rank)
            print("field: GF(" + str(q) + ")")
            print("matrix shape:", M.shape)

            print("edges:")
            for i, e in enumerate(edges):
                print(i, e)

            print("matrix:")
            print(M)

            return M, edges

    print("FAILED")
    return None, edges


if __name__ == "__main__":
    search_linear_matroid(
        n=6,
        rank=12,
        q=2**8,
        trials=10000,
        require_k5=False,
        require_k222=True
    )
