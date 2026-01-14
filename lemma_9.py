import numpy as np
import itertools

def rigidity_edge_vector(p: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Build f_{ij} in R^{d*n} with block i = p_i - p_j, block j = p_j - p_i, others 0.
    p: (n,d)
    returns: (d*n,)
    """
    n, d = p.shape
    f = np.zeros((n, d), dtype=float)
    f[i] = p[i] - p[j]
    f[j] = p[j] - p[i]
    return f.reshape(n * d)

def build_edge_matrix(p: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    """
    Stack the f_e as columns: A is (d*n) x m, where m = number of edges.
    Then a dependence among the f_e is A @ c = 0.
    """
    cols = [rigidity_edge_vector(p, i, j) for (i, j) in edges]
    return np.column_stack(cols)  # (d*n, m)

def nullspace(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Orthonormal basis for ker(A) using SVD.
    Returns N of shape (m, k) where k = dim ker(A) and A @ N = 0.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # singular values are sorted descending
    rank = int(np.sum(s > tol))
    # Vt is (m,m), rows are right singular vectors. Nullspace is last m-rank columns of V.
    N = Vt[rank:, :].T
    return N

def find_one_dependence(p: np.ndarray, edges: list[tuple[int, int]], tol: float = 1e-10) -> np.ndarray | None:
    """
    Returns one nonzero coefficient vector c (length m) with A @ c = 0, or None if ker(A) is trivial.
    """
    A = build_edge_matrix(p, edges)
    N = nullspace(A, tol=tol)
    if N.shape[1] == 0:
        return None
    # Take any nonzero vector in the nullspace; random linear combination is usually fine.
    coeffs = N @ np.random.randn(N.shape[1])
    # Normalize for readability
    mx = np.max(np.abs(coeffs))
    if mx > 0:
        coeffs = coeffs / mx
    return coeffs

def edges_of_K222(parts: tuple[list[int], list[int], list[int]]) -> list[tuple[int, int]]:
    """
    parts = (A,B,C) with |A|=|B|=|C|=2, returns all edges between different parts.
    """
    A, B, C = parts
    edges = []
    for X, Y in [(A, B), (A, C), (B, C)]:
        for i in X:
            for j in Y:
                edges.append((min(i, j), max(i, j)))
    return edges

def demo_K222(n: int = 10, d: int = 2, seed: int = 0) -> None:
    """
    Demonstrates finding a dependence on a specific K_{2,2,2} copy.
    - In d=2: you should typically get a nontrivial nullspace (dim ~ 3) for K222.
    - In d=3: generically you get None (no dependence), because K222 is isostatic in 3D.
    """
    rng = np.random.default_rng(seed)
    p = rng.standard_normal((n, d))  # generic random placement

    # pick a specific 6-set as (2,2,2)
    A = [0, 1]
    B = [2, 3]
    C = [4, 5]
    edges = edges_of_K222((A, B, C))

    c = find_one_dependence(p, edges, tol=1e-10)
    print(f"Ambient: R^{d}, n={n}, K222 edges={len(edges)}")
    if c is None:
        print("No nontrivial dependence found (nullspace is trivial).")
        return

    # verify
    M = build_edge_matrix(p, edges)
    residual = np.linalg.norm(M @ c)
    print(f"Found dependence coefficients c (normalized):")
    print(np.round(c, 6))
    print(f"Residual ||A c|| = {residual:.3e}")

    # check if all coefficients are nonzero-ish
    nonzero = np.sum(np.abs(c) > 1e-6)
    print(f"Nonzero entries (>|1e-6|): {nonzero}/{len(c)}")

if __name__ == "__main__":
    # Try d=2 first (should find dependencies)
    demo_K222(n=10, d=2, seed=1)

    # Try d=3 (typically no dependencies for K222)
    demo_K222(n=10, d=3, seed=1)
