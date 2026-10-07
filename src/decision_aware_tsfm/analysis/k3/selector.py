"""E4: an Omega-plus-tau family selector that can defer.

The selector answers one question per (series, tau): use the classical family,
use the TSFM family, or defer because the evidence does not support a call.
Deferring is a real answer, not a failure -- K3 exists because a scalar Omega
boundary may not be defensible, and a selector that guesses anyway would hide
that.

Conventions follow docs/decisions/0006-k3-boundary-statistics.md:
  * the unit is the series, with origins already aggregated upstream;
  * sufficiency is counted in distinct DATASETS, not distinct Omega values;
  * both threshold orientations are searched, never just one;
  * fitting happens on a training subset only, then the spec is FROZEN -- it is
    a plain dict of numbers and applying it never refits;
  * an undefined Omega (null, with its reason) is a deferral, never a 0.0.

Deferral reasons are explicit so the frontier can be read:
  omega_undefined     -- no Omega for this series (flat or empty history)
  omega_out_of_range  -- Omega outside the range the rule was fitted on
  weak_evidence       -- Omega within `margin` of the threshold
  unfitted_tau        -- no rule was fitted for this tau
  insufficient_data   -- too few datasets at this tau to fit anything

PROVISIONAL: deferral is costed as the oracle (the deferred case is sent for
full validation across both families, so the better family is used). That makes
the frontier an upper bound on the value of deferring and ignores the compute
the second family costs. The compute price belongs in the E4 write-up; confirm
with Veer and Pratham before the real run.
"""
import numpy as np

from decision_aware_tsfm.analysis.k3.boundary import (
    MIN_DISTINCT_OMEGA_THRESHOLD, ORIENTATIONS, _validate, distinct_count)

CLASSICAL, TSFM, DEFER = "classical", "tsfm", "defer"
SPEC_VERSION = "k3-omega-tau-selector-v1"


def _as_object_array(name, values, length):
    arr = np.asarray(values, dtype=object)
    if arr.shape != (length,):
        raise ValueError(f"{name} must be one-dimensional of length {length}, "
                         f"got shape {arr.shape}")
    return arr


def _check_inputs(omega, tau, cost_classical, cost_tsfm, datasets):
    """Omega may contain None (undefined); the other columns may not."""
    tau, cost_classical, cost_tsfm = _validate(tau, cost_classical, cost_tsfm)
    omega = _as_object_array("omega", omega, len(tau))
    datasets = _as_object_array("datasets", datasets, len(tau))
    for value in omega:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
            raise ValueError("omega entries must be numbers or None")
        if not np.isfinite(float(value)):
            raise ValueError("omega contains non-finite values")
    if np.any((tau <= 0) | (tau >= 1)):
        raise ValueError("tau must lie strictly between 0 and 1")
    return omega, tau, cost_classical, cost_tsfm, datasets


def _pick(orientation, below):
    """Which family a rule chooses on the low side of its threshold."""
    if orientation == "classical_below":
        return CLASSICAL if below else TSFM
    return TSFM if below else CLASSICAL


def fit_selector(omega, tau, cost_classical, cost_tsfm, datasets,
                 min_distinct=MIN_DISTINCT_OMEGA_THRESHOLD):
    """Fit one threshold rule per tau on TRAINING data, then freeze.

    Returns a plain dict. Applying it is pure lookup: no data, no refitting,
    nothing that could leak the evaluation set back into the rule.
    """
    omega, tau, cost_classical, cost_tsfm, datasets = _check_inputs(
        omega, tau, cost_classical, cost_tsfm, datasets)

    rules, undefined_total = {}, int(sum(v is None for v in omega))
    for value in sorted(set(tau.tolist())):
        at_tau = tau == value
        known = at_tau & np.array([v is not None for v in omega])
        key = f"{value:.2f}"

        if not known.any():
            rules[key] = {"status": "insufficient_data", "reason": "no defined Omega",
                          "n_datasets": 0, "required": min_distinct}
            continue

        values = np.array([float(v) for v in omega[known]])
        classical, tsfm = cost_classical[known], cost_tsfm[known]
        n_datasets, unit = distinct_count(values, datasets[known])
        if n_datasets < min_distinct:
            rules[key] = {"status": "insufficient_data",
                          "reason": f"too few distinct Omega {unit}s",
                          "n_datasets": int(n_datasets), "required": min_distinct}
            continue

        unique = np.unique(values)
        # Midpoints only: the extremes are "always one family", which is the
        # fixed-family baseline rather than a rule that uses Omega.
        candidates = (unique[:-1] + unique[1:]) / 2
        best = None
        for orientation in ORIENTATIONS:
            for threshold in candidates:
                below = values <= threshold
                chooses_classical = (below if orientation == "classical_below" else ~below)
                cost = float(np.where(chooses_classical, classical, tsfm).sum())
                if best is None or cost < best[0]:
                    best = (cost, orientation, float(threshold))

        cost, orientation, threshold = best
        rules[key] = {
            "status": "ok", "orientation": orientation, "threshold": threshold,
            "omega_min": float(values.min()), "omega_max": float(values.max()),
            "n_datasets": int(n_datasets), "n_series": int(known.sum()),
            "training_cost": cost,
            "always_classical_cost": float(classical.sum()),
            "always_tsfm_cost": float(tsfm.sum()),
        }

    return {"spec_version": SPEC_VERSION, "frozen": True, "rules": rules,
            "taus": sorted(rules), "training_series": int(len(tau)),
            "training_undefined_omega": undefined_total,
            "min_distinct_datasets": min_distinct}


def fit_tau_only_baseline(tau, cost_classical, cost_tsfm):
    """Aadrika's proposed control (#45): the cheaper family per tau, ignoring Omega.

    If the Omega-plus-tau selector cannot beat this, Omega is adding nothing
    beyond cost asymmetry and should not be presented as if it were.
    """
    tau, cost_classical, cost_tsfm = _validate(tau, cost_classical, cost_tsfm)
    rules = {}
    for value in sorted(set(tau.tolist())):
        at_tau = tau == value
        classical = float(cost_classical[at_tau].sum())
        tsfm = float(cost_tsfm[at_tau].sum())
        rules[f"{value:.2f}"] = {
            "status": "ok", "family": CLASSICAL if classical <= tsfm else TSFM,
            "always_classical_cost": classical, "always_tsfm_cost": tsfm}
    return {"spec_version": "k3-tau-only-baseline-v1", "frozen": True,
            "rules": rules, "taus": sorted(rules)}


def apply_selector(spec, omega, tau, margin=0.0):
    """Apply a frozen spec to one (omega, tau). Returns (choice, reason)."""
    if not spec.get("frozen"):
        raise ValueError("Selector spec is not frozen")
    if margin < 0:
        raise ValueError("margin must be non-negative")

    rule = spec["rules"].get(f"{float(tau):.2f}")
    if rule is None:
        return DEFER, "unfitted_tau"
    if rule["status"] != "ok":
        return DEFER, "insufficient_data"
    if spec["spec_version"] != SPEC_VERSION:          # tau-only baseline
        return rule["family"], "tau_only"
    if omega is None:
        return DEFER, "omega_undefined"

    value = float(omega)
    if not rule["omega_min"] <= value <= rule["omega_max"]:
        return DEFER, "omega_out_of_range"
    if abs(value - rule["threshold"]) < margin:
        return DEFER, "weak_evidence"
    return _pick(rule["orientation"], value <= rule["threshold"]), "rule"


def evaluate_selector(spec, omega, tau, cost_classical, cost_tsfm, datasets,
                      margin=0.0):
    """Score a frozen spec on an UNSEEN subset. Never M5/PJM.

    Deferred cases are costed as the oracle (see the module note): deferring
    means validating both families, so the better one is used.
    """
    omega, tau, cost_classical, cost_tsfm, datasets = _check_inputs(
        omega, tau, cost_classical, cost_tsfm, datasets)

    chosen, reasons = [], {}
    for value, point in zip(omega, tau):
        choice, reason = apply_selector(spec, value, point, margin=margin)
        chosen.append(choice)
        reasons[reason] = reasons.get(reason, 0) + 1
    chosen = np.array(chosen, dtype=object)

    deferred = chosen == DEFER
    oracle_each = np.minimum(cost_classical, cost_tsfm)
    decided = np.where(chosen == CLASSICAL, cost_classical, cost_tsfm)
    realized = float(np.where(deferred, oracle_each, decided).sum())

    always_classical = float(cost_classical.sum())
    always_tsfm = float(cost_tsfm.sum())
    best_fixed = min(always_classical, always_tsfm)
    oracle = float(oracle_each.sum())
    gap = best_fixed - oracle

    result = {
        "status": "ok", "margin": float(margin), "n": int(len(tau)),
        "deferral_rate": float(deferred.mean()),
        "deferrals": int(deferred.sum()), "reasons": dict(sorted(reasons.items())),
        "realized_cost": realized, "oracle_cost": oracle,
        "best_fixed_cost": best_fixed, "always_classical_cost": always_classical,
        "always_tsfm_cost": always_tsfm,
        "datasets": sorted({str(d) for d in datasets}),
    }
    if gap <= 0:
        result["status"] = "no_oracle_gap"
        result["fraction_of_oracle_value_captured"] = None
        result["note"] = "best fixed family already matches the oracle"
    else:
        result["fraction_of_oracle_value_captured"] = (best_fixed - realized) / gap
    return result


def cost_deferral_frontier(spec, omega, tau, cost_classical, cost_tsfm, datasets,
                           margins=None):
    """E4: realized cost against deferral rate, as the defer margin widens."""
    if margins is None:
        margins = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    points = [evaluate_selector(spec, omega, tau, cost_classical, cost_tsfm,
                                datasets, margin=m) for m in sorted(margins)]
    return {"points": points,
            "margins": [p["margin"] for p in points],
            "deferral_rates": [p["deferral_rate"] for p in points],
            "realized_costs": [p["realized_cost"] for p in points]}


def leave_one_dataset_out(omega, tau, cost_classical, cost_tsfm, datasets,
                          margin=0.0, min_distinct=MIN_DISTINCT_OMEGA_THRESHOLD):
    """Fit on all datasets but one, evaluate on the held-out one, and rotate.

    The agreed fitting protocol (Veer, confirmed 2026-10-07). With only two
    datasets in the 0.40-0.50 band, a single fixed split can leave that band
    unrepresented on one side; rotating gives every dataset a turn as unseen.
    Also reports the tau-only baseline on the same folds, so Omega has to earn
    its place rather than merely perform.
    """
    omega, tau, cost_classical, cost_tsfm, datasets = _check_inputs(
        omega, tau, cost_classical, cost_tsfm, datasets)
    names = sorted({str(d) for d in datasets})
    if len(names) < 2:
        return {"status": "insufficient_data",
                "reason": "leave-one-dataset-out needs at least 2 datasets",
                "n_datasets": len(names)}

    folds = []
    for held_out in names:
        test = np.array([str(d) == held_out for d in datasets])
        train = ~test
        if not train.any() or not test.any():
            continue
        spec = fit_selector(omega[train], tau[train], cost_classical[train],
                            cost_tsfm[train], datasets[train], min_distinct=min_distinct)
        baseline = fit_tau_only_baseline(tau[train], cost_classical[train],
                                         cost_tsfm[train])
        folds.append({
            "held_out": held_out,
            "n_train": int(train.sum()), "n_test": int(test.sum()),
            "selector": evaluate_selector(spec, omega[test], tau[test],
                                          cost_classical[test], cost_tsfm[test],
                                          datasets[test], margin=margin),
            "tau_only": evaluate_selector(baseline, omega[test], tau[test],
                                          cost_classical[test], cost_tsfm[test],
                                          datasets[test], margin=margin),
        })

    selector_cost = sum(f["selector"]["realized_cost"] for f in folds)
    baseline_cost = sum(f["tau_only"]["realized_cost"] for f in folds)
    return {
        "status": "ok", "margin": float(margin), "folds": folds,
        "datasets": names,
        "selector_total_cost": selector_cost,
        "tau_only_total_cost": baseline_cost,
        "omega_advantage": baseline_cost - selector_cost,
        "mean_deferral_rate": float(np.mean([f["selector"]["deferral_rate"]
                                             for f in folds])) if folds else None,
    }
