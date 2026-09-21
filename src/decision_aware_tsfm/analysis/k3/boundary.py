"""K3 — is a scalar Omega boundary meaningful?

Convention used throughout:
    delta = cost_tsfm - cost_classical
    delta > 0  -> classical family is cheaper at that series
    delta < 0  -> TSFM family is cheaper at that series

Fixes relative to the feasibility-sprint skeleton (#13):
  1. Monotone direction is pre-registered or both directions are reported;
     never silently chosen after seeing results.
  2. Minimum distinct-Omega requirements are explicit and configurable.
  3. Both threshold orientations are tested (classical-below and TSFM-below).
  4. Oracle value is measured against the BEST fixed family by default.
  5. Too little data returns an explicit `insufficient_data` result.

Malformed input (wrong shape, non-finite values) raises ValueError. That is
a different condition from "valid but insufficient" and must not be confused
with it.

PROVISIONAL: MIN_DISTINCT_OMEGA_THRESHOLD and MIN_DISTINCT_OMEGA_FLEXIBLE are
new analysis parameters. Their values should be agreed on the issue and
recorded in docs/decisions/ before the real multi-series run.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, KFold

MIN_DISTINCT_OMEGA_THRESHOLD = 3
MIN_DISTINCT_OMEGA_FLEXIBLE = 8
ORIENTATIONS = ("classical_below", "tsfm_below")
DIRECTIONS = ("increasing", "decreasing")


# ----------------------------------------------------------------- helpers
def insufficient_data(reason, n_distinct_omega, required):
    return {"status": "insufficient_data", "reason": reason,
            "n_distinct_omega": int(n_distinct_omega), "required": int(required)}


def _as_1d_finite(name, values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _validate(omega, *others):
    omega = _as_1d_finite("omega", omega)
    out = [omega]
    for i, other in enumerate(others):
        arr = _as_1d_finite(f"input_{i}", other)
        if arr.shape != omega.shape:
            raise ValueError(f"input_{i} length {len(arr)} != omega length {len(omega)}")
        out.append(arr)
    return out


def _collapse_ties(omega, delta):
    """Average delta over series sharing an identical Omega value."""
    unique, inverse = np.unique(omega, return_inverse=True)
    means = np.bincount(inverse, weights=delta) / np.bincount(inverse)
    return unique, means


# --------------------------------------------------------- 1. cost difference
def family_cost_difference(cost_classical, cost_tsfm):
    """delta = cost_tsfm - cost_classical. Positive means classical is cheaper."""
    c, t = _validate(cost_classical, cost_tsfm)
    return t - c


# ------------------------------------------------------- 2. sign reversals
def count_sign_reversals(omega, delta, min_distinct=MIN_DISTINCT_OMEGA_THRESHOLD):
    """Count flips in the winning family as Omega increases.

    Series with identical Omega are averaged first so ordering among ties
    cannot manufacture reversals. Exact zeros (a tie between families) are
    dropped rather than counted as a flip.
    """
    omega, delta = _validate(omega, delta)
    n_distinct = len(np.unique(omega))
    if n_distinct < min_distinct:
        return insufficient_data("too few distinct Omega values", n_distinct, min_distinct)

    _, mean_delta = _collapse_ties(omega, delta)
    signs = np.sign(mean_delta)
    signs = signs[signs != 0]
    reversals = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0
    return {"status": "ok", "sign_reversals": reversals,
            "n_distinct_omega": n_distinct, "n_series": len(omega)}


# ------------------------------------------- 3. monotone vs flexible fit
def _crossing_interval(x_sorted, fitted):
    """For a monotone fit, return the Omega interval where it changes sign."""
    signs = np.sign(fitted)
    idx = np.where(np.diff(signs) != 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    return [float(x_sorted[i]), float(x_sorted[i + 1])]


def _cv_mse(x, y, make_predict, groups, n_splits):
    splitter = GroupKFold(n_splits) if groups is not None else KFold(
        n_splits, shuffle=True, random_state=0)
    errors = []
    for train, test in splitter.split(x, y, groups):
        pred = make_predict(x[train], y[train])(x[test])
        errors.append((pred - y[test]) ** 2)
    return float(np.mean(np.concatenate(errors)))


def fit_monotone_boundary(omega, delta, direction="both", groups=None, n_splits=5,
                          min_distinct_monotone=MIN_DISTINCT_OMEGA_THRESHOLD,
                          min_distinct_flexible=MIN_DISTINCT_OMEGA_FLEXIBLE):
    """Compare monotone fits with an unconstrained flexible fit.

    direction: "increasing", "decreasing", or "both". Pass a single direction
    only if it was pre-registered before seeing the data.

    Fits are compared by out-of-fold MSE, not in-sample MSE (in-sample always
    favours the more flexible model). Pass `groups` (e.g. dataset/domain
    labels) so folds never split one group across train and test.
    """
    if direction not in ("both", *DIRECTIONS):
        raise ValueError(f"direction must be 'both', 'increasing' or 'decreasing'")
    omega, delta = _validate(omega, delta)
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != omega.shape:
            raise ValueError("groups length must match omega")

    n_distinct = len(np.unique(omega))
    if n_distinct < min_distinct_monotone:
        return insufficient_data("too few distinct Omega values for any fit",
                                 n_distinct, min_distinct_monotone)

    n_folds = min(n_splits, len(np.unique(groups)) if groups is not None else len(omega))
    if n_folds < 2:
        return insufficient_data("too few series/groups for cross-validation", n_distinct, 2)

    order = np.argsort(omega)
    x_sorted, y_sorted = omega[order], delta[order]
    result = {"status": "ok", "n_distinct_omega": n_distinct, "n_series": len(omega),
              "direction_mode": direction, "monotone": {}}

    for d in (DIRECTIONS if direction == "both" else (direction,)):
        inc = d == "increasing"

        def make(xtr, ytr, inc=inc):
            model = IsotonicRegression(increasing=inc, out_of_bounds="clip").fit(xtr, ytr)
            return model.predict

        fitted = make(x_sorted, y_sorted)(x_sorted)
        result["monotone"][d] = {
            "cv_mse": _cv_mse(omega, delta, make, groups, n_folds),
            "crossing_omega_interval": _crossing_interval(x_sorted, fitted),
        }

    if n_distinct < min_distinct_flexible:
        result["flexible"] = insufficient_data(
            "too few distinct Omega values for a flexible curve",
            n_distinct, min_distinct_flexible)
    else:
        degree = min(3, n_distinct - 2)

        def make_flex(xtr, ytr, degree=degree):
            return np.poly1d(np.polyfit(xtr, ytr, deg=degree))

        result["flexible"] = {"status": "ok", "degree": degree,
                              "cv_mse": _cv_mse(omega, delta, make_flex, groups, n_folds)}
    return result


# ------------------------------------------------- 4. oracle value captured
def oracle_value_best_threshold(omega, cost_classical, cost_tsfm, baseline="best_fixed",
                                min_distinct=MIN_DISTINCT_OMEGA_THRESHOLD):
    """How much of the oracle's advantage does the best single Omega threshold
    recover, relative to a fixed-family baseline?

        fraction = (baseline_cost - threshold_cost) / (baseline_cost - oracle_cost)

    baseline="best_fixed" (default) uses the cheaper of always-classical and
    always-TSFM. "worst_fixed" reproduces the sprint skeleton's definition and
    is kept only so the two can be compared while the definition is confirmed.

    Both orientations are searched. Candidate thresholds include the extremes,
    so "always one family" is always available and fraction >= 0 under the
    best_fixed baseline.
    """
    if baseline not in ("best_fixed", "worst_fixed"):
        raise ValueError("baseline must be 'best_fixed' or 'worst_fixed'")
    omega, c, t = _validate(omega, cost_classical, cost_tsfm)
    n_distinct = len(np.unique(omega))
    if n_distinct < min_distinct:
        return insufficient_data("too few distinct Omega values", n_distinct, min_distinct)

    always_classical, always_tsfm = float(c.sum()), float(t.sum())
    oracle = float(np.minimum(c, t).sum())
    base = min(always_classical, always_tsfm) if baseline == "best_fixed" \
        else max(always_classical, always_tsfm)

    u = np.unique(omega)
    candidates = np.concatenate(([u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]))

    per_orientation = {}
    for orient in ORIENTATIONS:
        best_thr, best_cost = None, np.inf
        for thr in candidates:
            below = omega <= thr
            if orient == "classical_below":
                cost = float(np.where(below, c, t).sum())
            else:
                cost = float(np.where(below, t, c).sum())
            if cost < best_cost:
                best_cost, best_thr = cost, float(thr)
        per_orientation[orient] = {"threshold": best_thr, "cost": best_cost}

    best_orient = min(ORIENTATIONS, key=lambda o: per_orientation[o]["cost"])
    gap = base - oracle
    out = {"status": "ok", "baseline": baseline, "n_distinct_omega": n_distinct,
           "always_classical_cost": always_classical, "always_tsfm_cost": always_tsfm,
           "oracle_cost": oracle, "baseline_cost": base,
           "orientations": per_orientation, "best_orientation": best_orient}
    if gap <= 0:
        out["status"] = "no_oracle_gap"
        out["fraction_of_oracle_value_captured"] = None
        out["note"] = "baseline already matches the oracle; nothing to capture"
        return out

    for orient in ORIENTATIONS:
        per_orientation[orient]["fraction_captured"] = \
            (base - per_orientation[orient]["cost"]) / gap
    out["fraction_of_oracle_value_captured"] = per_orientation[best_orient]["fraction_captured"]
    return out
