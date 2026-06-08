import json
from collections import defaultdict


def full_mask(n):
    return (1 << (n * (n - 1) // 2)) - 1


def add_term(expr, mask, coeff):
    expr[str(mask)] += float(coeff)


def add_expr(acc, expr, coeff):
    for var, value in expr.items():
        acc[str(var)] += coeff * float(value)


def constraint_to_expression(constraint):
    expr = defaultdict(float)
    t = constraint["type"]

    if t == "empty_rank":
        add_term(expr, constraint["mask"], 1)
        return dict(expr), 0.0, "="

    if t == "monotonicity":
        add_term(expr, constraint["small_mask"], 1)
        add_term(expr, constraint["large_mask"], -1)
        return dict(expr), 0.0, "<="

    if t == "K5":
        add_term(expr, constraint["mask"], 1)
        return dict(expr), 9.0, "="

    if t == "K5_minus_edge":
        add_term(expr, constraint["mask"], 1)
        return dict(expr), 9.0, "="

    if t == "K222":
        add_term(expr, constraint["mask"], 1)
        return dict(expr), 11.0, "="

    if t == "K222_minus_edge":
        add_term(expr, constraint["mask"], 1)
        return dict(expr), 11.0, "="

    if t == "submodularity":
        add_term(expr, constraint["Ae_mask"], 1)
        add_term(expr, constraint["Af_mask"], 1)
        add_term(expr, constraint["A_mask"], -1)
        add_term(expr, constraint["Aef_mask"], -1)
        return dict(expr), 0.0, ">="

    raise ValueError(f"Unknown constraint type: {t}")


def check_one_convention(cert, n, claimed_bound, row_sign, col_sign, target_sign, eps):
    lhs = defaultdict(float)
    rhs = 0.0

    for _, item in cert["row_terms"].items():
        y = row_sign * float(item["dual_value"])
        expr, b, _ = constraint_to_expression(item["constraint"])
        add_expr(lhs, expr, y)
        rhs += y * b

    for _, item in cert["column_bound_terms"].items():
        y = col_sign * float(item["dual_value"])
        variable = item["variable"]
        mask = str(variable["mask"])
        upper = float(variable["upper_bound"])

        lhs[mask] += y

        # If coefficient is positive, use r(G) <= upper.
        # If coefficient is negative, use r(G) >= 0, giving RHS 0.
        if y > 0:
            rhs += y * upper

    target = defaultdict(float)
    target[str(full_mask(n))] = target_sign

    residual = {
        var: lhs.get(var, 0.0) - target.get(var, 0.0)
        for var in set(lhs) | set(target)
        if abs(lhs.get(var, 0.0) - target.get(var, 0.0)) > eps
    }

    rhs_error = rhs - claimed_bound if target_sign == 1 else rhs + claimed_bound

    return residual, rhs, rhs_error


def verify_combined_certificate(combined_certificate_file, n, claimed_bound, eps=1e-6):
    with open(combined_certificate_file, "r") as f:
        cert = json.load(f)

    best = None

    for row_sign in [1, -1]:
        for col_sign in [1, -1]:
            for target_sign in [1, -1]:
                residual, rhs, rhs_error = check_one_convention(
                    cert=cert,
                    n=n,
                    claimed_bound=claimed_bound,
                    row_sign=row_sign,
                    col_sign=col_sign,
                    target_sign=target_sign,
                    eps=eps,
                )

                score = (len(residual), abs(rhs_error))

                if best is None or score < best[0]:
                    best = (score, row_sign, col_sign, target_sign, residual, rhs, rhs_error)

    score, row_sign, col_sign, target_sign, residual, rhs, rhs_error = best
    passed = len(residual) == 0 and abs(rhs_error) <= eps

    print("Certificate:", combined_certificate_file)
    print("n:", n)
    print("Objective mask:", full_mask(n))
    print("Claimed bound:", claimed_bound)
    print("Best row_sign:", row_sign)
    print("Best col_sign:", col_sign)
    print("Best target_sign:", target_sign)
    print("Computed RHS:", rhs)
    print("RHS error:", rhs_error)
    print("Residual variables:", len(residual))
    print("PASSED:", passed)

    if residual:
        print("\nLargest residuals:")
        for var, value in sorted(residual.items(), key=lambda x: -abs(x[1]))[:30]:
            print(var, value)

    return passed


if __name__ == "__main__":
    verify_combined_certificate(
        combined_certificate_file=r"C:\Oren\Academy\Thesis\Thesis-percolation\matroids\dual_output\certificate_n6\combined_certificate.json",
        n=6,
        claimed_bound=12,
        eps=1e-6,
    )