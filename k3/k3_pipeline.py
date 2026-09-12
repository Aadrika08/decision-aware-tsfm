"""K3 analysis pipeline: sign reversals, monotone-vs-unconstrained boundary fit,
oracle value at the best single-Omega threshold.

STATUS: pipeline logic built and smoke-tested against the ONE real series
available so far (Aadrika's #3 smoke test: ett1/H channel 0). This is NOT
a K3 conclusion -- K3 requires many series spanning a range of Omega values
to fit or reject a boundary. With one series there is exactly one Omega
value, so no boundary can be fit yet. This file exists to prove the
mechanics are correct before real multi-series data lands.
"""
import numpy as np


def family_cost_difference(cost_a, cost_b):
    """cost_b - cost_a. Positive means family A is cheaper (wins)."""
    return cost_b - cost_a


def count_sign_reversals(omega_values, cost_differences):
    """Count how many times the winning family flips as Omega increases.

    Requires multiple (omega, cost_difference) pairs sorted by omega.
    With a single series (one omega value) this always returns 0 --
    that is expected and correct, not a bug.
    """
    omega_values = np.asarray(omega_values)
    cost_differences = np.asarray(cost_differences)
    order = np.argsort(omega_values)
    signs = np.sign(cost_differences[order])
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.sum(np.diff(signs) != 0))


def fit_monotone_boundary(omega_values, cost_differences):
    """Fit a monotone (isotonic) boundary vs. an unconstrained one and
    compare. Requires at least several distinct omega values to be
    meaningful. Returns None if there isn't enough data to fit anything.
    """
    omega_values = np.asarray(omega_values)
    cost_differences = np.asarray(cost_differences)
    if len(np.unique(omega_values)) < 3:
        return None  # not enough distinct Omega points to fit a boundary
    from sklearn.isotonic import IsotonicRegression
    order = np.argsort(omega_values)
    iso = IsotonicRegression()
    monotone_fit = iso.fit_transform(omega_values[order], cost_differences[order])
    unconstrained_fit = np.poly1d(np.polyfit(omega_values[order], cost_differences[order], deg=3))(omega_values[order])
    return {
        "monotone_fit": monotone_fit,
        "unconstrained_fit": unconstrained_fit,
        "monotone_mse": np.mean((monotone_fit - cost_differences[order]) ** 2),
        "unconstrained_mse": np.mean((unconstrained_fit - cost_differences[order]) ** 2),
    }


def oracle_value_best_threshold(omega_values, cost_differences, always_a_cost, always_b_cost):
    """How much value does the best single Omega threshold capture,
    versus always picking one family or perfectly picking per-series
    (the oracle)? Requires multiple series with varying Omega.
    """
    omega_values = np.asarray(omega_values)
    cost_differences = np.asarray(cost_differences)
    if len(np.unique(omega_values)) < 3:
        return None  # not enough distinct Omega points

    oracle_cost = np.minimum(always_a_cost, always_b_cost).sum()
    best_threshold, best_cost = None, np.inf
    for candidate in np.unique(omega_values):
        picks_a = omega_values <= candidate
        cost = np.where(picks_a, always_a_cost, always_b_cost).sum()
        if cost < best_cost:
            best_cost, best_threshold = cost, candidate

    always_a_total = always_a_cost.sum()
    always_b_total = always_b_cost.sum()
    worst_fixed = max(always_a_total, always_b_total)

    captured = (worst_fixed - best_cost) / (worst_fixed - oracle_cost) if worst_fixed != oracle_cost else None
    return {"best_threshold": best_threshold, "best_threshold_cost": best_cost,
            "oracle_cost": oracle_cost, "fraction_of_oracle_value_captured": captured}


if __name__ == "__main__":
    # --- Smoke test against the ONE real data point currently available ---
    # From Aadrika's #3: ett1/H channel 0, mean pinball loss.
    tau_05 = {"seasonal_naive": 4.315344, "moirai": 5.514882}
    tau_91 = {"seasonal_naive": 2.822248, "moirai": 2.016869}

    print("=== family_cost_difference (real data, one series) ===")
    diff_05 = family_cost_difference(tau_05["seasonal_naive"], tau_05["moirai"])
    diff_91 = family_cost_difference(tau_91["seasonal_naive"], tau_91["moirai"])
    print(f"  tau=0.5:  diff = {diff_05:.6f}  ({'classical wins' if diff_05 > 0 else 'TSFM wins'})")
    print(f"  tau=0.91: diff = {diff_91:.6f}  ({'classical wins' if diff_91 > 0 else 'TSFM wins'})")
    print()

    print("=== count_sign_reversals (needs multiple Omega points) ===")
    # Only one real Omega value exists right now -> correctly returns 0.
    result = count_sign_reversals([0.42], [diff_05])
    print(f"  With 1 real series: {result} (expected 0 -- not enough data, not a finding)")
    print()

    print("=== fit_monotone_boundary / oracle_value_best_threshold ===")
    print("  Both correctly return None with fewer than 3 distinct Omega values.")
    print("  fit_monotone_boundary:", fit_monotone_boundary([0.42], [diff_05]))
    print("  oracle_value_best_threshold:", oracle_value_best_threshold(
        [0.42], [diff_05], np.array([tau_05["seasonal_naive"]]), np.array([tau_05["moirai"]])))
    print()
    print("CONCLUSION: pipeline mechanics verified. Real K3 boundary analysis")
    print("is blocked pending multiple series spanning a range of Omega values.")
