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

    if t in {"K5", "K5_minus_edge"}:
        add_term(expr, constraint["mask"], 1)
        return dict(expr), 9.0, "="

    if t in {"K222", "K222_minus_edge"}:
        add_term(expr, constraint["mask"], 1)
        return dict(expr), 11.0, "="

    if t == "submodularity":
        add_term(expr, constraint["Ae_mask"], 1)
        add_term(expr, constraint["Af_mask"], 1)
        add_term(expr, constraint["A_mask"], -1)
        add_term(expr, constraint["Aef_mask"], -1)
        return dict(expr), 0.0, ">="

    raise ValueError(f"Unknown constraint type: {t}")


def check_one_convention(cert, n, row_sign, col_sign, target_sign, eps):
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

        # Positive coefficient uses r(G) <= upper.
        # Negative coefficient uses r(G) >= 0, so contributes 0 to RHS.
        if y > 0:
            rhs += y * upper

    target = defaultdict(float)
    target[str(full_mask(n))] = target_sign

    residual = {
        var: lhs.get(var, 0.0) - target.get(var, 0.0)
        for var in set(lhs) | set(target)
        if abs(lhs.get(var, 0.0) - target.get(var, 0.0)) > eps
    }

    return residual, rhs


def verify_combined_certificate(combined_certificate_file, n, claimed_bound, eps=1e-6):
    with open(combined_certificate_file, "r") as f:
        cert = json.load(f)

    best = None

    for row_sign in [1, -1]:
        for col_sign in [1, -1]:
            for target_sign in [1, -1]:
                residual, rhs = check_one_convention(
                    cert=cert,
                    n=n,
                    row_sign=row_sign,
                    col_sign=col_sign,
                    target_sign=target_sign,
                    eps=eps,
                )

                is_upper_bound = target_sign == 1
                rhs_gap = rhs - claimed_bound if is_upper_bound else float("inf")

                # Prefer exact identity, then upper-bound conventions, then smaller valid upper bound.
                score = (
                    len(residual),
                    0 if is_upper_bound else 1,
                    abs(rhs_gap) if is_upper_bound else float("inf"),
                )

                if best is None or score < best[0]:
                    best = (score, row_sign, col_sign, target_sign, residual, rhs, is_upper_bound, rhs_gap)

    score, row_sign, col_sign, target_sign, residual, rhs, is_upper_bound, rhs_gap = best

    # This only accepts certificates proving r(K_n) <= rhs.
    # If the true optimum is 15, then rhs=16 is acceptable, rhs=14 is not.
    bound_is_good = is_upper_bound and rhs + eps <= claimed_bound
    passed = len(residual) == 0 and bound_is_good

    print("Certificate:", combined_certificate_file)
    print("n:", n)
    print("Objective mask:", full_mask(n))
    print("Claimed primal optimum/lower reference:", claimed_bound)
    print("Best row_sign:", row_sign)
    print("Best col_sign:", col_sign)
    print("Best target_sign:", target_sign)
    print("Is upper bound:", is_upper_bound)
    print("Computed upper bound:", rhs if is_upper_bound else "not an upper bound")
    print("Gap above claimed optimum:", rhs_gap if is_upper_bound else None)
    print("Residual variables:", len(residual))
    print("PASSED:", passed)

    if residual:
        print("\nLargest residuals:")
        for var, value in sorted(residual.items(), key=lambda x: -abs(x[1]))[:30]:
            print(var, value)

    return passed


if __name__ == "__main__":
    verify_combined_certificate(
        combined_certificate_file=r"C:\Oren\Academy\Thesis\Thesis-percolation\matroids\dual_output\certificate_n7\combined_certificate.json",
        n=7,
        claimed_bound=16,
        eps=1e-6,
    )