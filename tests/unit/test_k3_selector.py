"""Synthetic ground-truth tests for the E4 Omega-plus-tau selector.

Invented data with planted answers. No real forecasts, no M5/PJM, no inference.
"""
import numpy as np
import pytest

from decision_aware_tsfm.analysis.k3.selector import (
    CLASSICAL, DEFER, TSFM, apply_selector, cost_deferral_frontier,
    evaluate_selector, fit_selector, fit_tau_only_baseline, leave_one_dataset_out)

SEED = 20261007
TAUS = (0.50, 0.91)


def planted(n_per_dataset=12, n_datasets=4, boundary=0.5, classical_below=True,
            advantage=0.5, noise=0.02, seed=SEED, omega_undefined=0):
    """Per tau, classical wins on one side of `boundary` and TSFM on the other."""
    rng = np.random.default_rng(seed)
    omega, tau, classical, tsfm, datasets = [], [], [], [], []
    for d in range(n_datasets):
        base_omega = np.linspace(0.1, 0.9, n_per_dataset) + rng.normal(0, 0.01, n_per_dataset)
        for value, point in [(o, t) for o in base_omega for t in TAUS]:
            base = rng.uniform(1.0, 2.0)
            cheaper_classical = (value <= boundary) if classical_below else (value > boundary)
            classical.append(base + (0.0 if cheaper_classical else advantage)
                             + rng.normal(0, noise))
            tsfm.append(base + (advantage if cheaper_classical else 0.0)
                        + rng.normal(0, noise))
            omega.append(float(value))
            tau.append(point)
            datasets.append(f"dataset_{d}")

    for i in range(omega_undefined):
        omega[i] = None
    return (np.array(omega, dtype=object), np.array(tau), np.array(classical),
            np.array(tsfm), np.array(datasets, dtype=object))


# ------------------------------------------------------------ fit and freeze
def test_fitted_rule_recovers_the_planted_boundary():
    spec = fit_selector(*planted())
    assert spec["frozen"] is True
    for tau in ("0.50", "0.91"):
        rule = spec["rules"][tau]
        assert rule["status"] == "ok"
        assert rule["orientation"] == "classical_below"
        assert abs(rule["threshold"] - 0.5) < 0.06


def test_reversed_orientation_recovered():
    spec = fit_selector(*planted(classical_below=False))
    assert spec["rules"]["0.50"]["orientation"] == "tsfm_below"


def test_spec_is_plain_data_and_applying_never_refits():
    import copy
    spec = fit_selector(*planted())
    before = copy.deepcopy(spec)
    for _ in range(3):
        apply_selector(spec, 0.2, 0.50)
    assert spec == before


def test_unfrozen_spec_refused():
    spec = fit_selector(*planted())
    spec["frozen"] = False
    with pytest.raises(ValueError, match="not frozen"):
        apply_selector(spec, 0.2, 0.50)


# ---------------------------------------------------------- deferral reasons
def test_defers_on_undefined_omega():
    spec = fit_selector(*planted())
    assert apply_selector(spec, None, 0.50) == (DEFER, "omega_undefined")


def test_defers_outside_the_fitted_range():
    spec = fit_selector(*planted())
    assert apply_selector(spec, 0.999, 0.50) == (DEFER, "omega_out_of_range")
    assert apply_selector(spec, 0.001, 0.50) == (DEFER, "omega_out_of_range")


def test_defers_on_weak_evidence_near_the_threshold():
    spec = fit_selector(*planted())
    threshold = spec["rules"]["0.50"]["threshold"]
    assert apply_selector(spec, threshold + 0.01, 0.50, margin=0.1)[1] == "weak_evidence"
    assert apply_selector(spec, threshold + 0.01, 0.50, margin=0.0)[1] == "rule"


def test_defers_on_an_unfitted_tau():
    spec = fit_selector(*planted())
    assert apply_selector(spec, 0.3, 0.25) == (DEFER, "unfitted_tau")


def test_defers_when_too_few_datasets():
    spec = fit_selector(*planted(n_datasets=2))
    assert spec["rules"]["0.50"]["status"] == "insufficient_data"
    assert apply_selector(spec, 0.3, 0.50) == (DEFER, "insufficient_data")


def test_choices_follow_the_planted_sides():
    spec = fit_selector(*planted())
    assert apply_selector(spec, 0.2, 0.50)[0] == CLASSICAL
    assert apply_selector(spec, 0.8, 0.50)[0] == TSFM


def test_negative_margin_refused():
    spec = fit_selector(*planted())
    with pytest.raises(ValueError, match="non-negative"):
        apply_selector(spec, 0.3, 0.50, margin=-0.1)


# -------------------------------------------------------------- evaluation
def test_evaluation_on_unseen_data_captures_planted_value():
    train = planted(seed=1)
    test = planted(seed=2)
    result = evaluate_selector(fit_selector(*train), *test)
    assert result["status"] == "ok"
    assert result["fraction_of_oracle_value_captured"] > 0.8
    assert result["deferral_rate"] < 0.1


def test_noise_only_data_captures_little():
    rng = np.random.default_rng(SEED)
    n = 400
    omega = np.array(rng.uniform(0.1, 0.9, n), dtype=object)
    tau = np.array(rng.choice(TAUS, n))
    base = rng.uniform(1, 2, n)
    classical = base + rng.normal(0, 0.3, n)
    tsfm = base + rng.normal(0, 0.3, n)          # winner unrelated to Omega
    datasets = np.array([f"dataset_{i % 4}" for i in range(n)], dtype=object)
    spec = fit_selector(omega, tau, classical, tsfm, datasets)
    result = evaluate_selector(spec, omega, tau, classical, tsfm, datasets)
    assert result["fraction_of_oracle_value_captured"] < 0.5


def test_constant_winner_reports_no_oracle_gap():
    omega, tau, classical, _, datasets = planted()
    tsfm = classical - 0.5                        # TSFM cheaper everywhere
    spec = fit_selector(omega, tau, classical, tsfm, datasets)
    result = evaluate_selector(spec, omega, tau, classical, tsfm, datasets)
    assert result["status"] == "no_oracle_gap"
    assert result["fraction_of_oracle_value_captured"] is None


def test_undefined_omega_counted_as_deferral_not_dropped():
    omega, tau, classical, tsfm, datasets = planted(omega_undefined=10)
    spec = fit_selector(omega, tau, classical, tsfm, datasets)
    result = evaluate_selector(spec, omega, tau, classical, tsfm, datasets)
    assert result["reasons"].get("omega_undefined") == 10
    assert result["n"] == len(tau)                # nothing silently dropped


# ---------------------------------------------------------------- frontier
def test_frontier_deferral_rate_is_monotone_in_margin():
    train, test = planted(seed=1), planted(seed=2)
    frontier = cost_deferral_frontier(fit_selector(*train), *test)
    rates = frontier["deferral_rates"]
    assert rates == sorted(rates)
    assert rates[0] < rates[-1]


def test_frontier_cost_falls_as_deferral_rises():
    """Deferring is costed as the oracle, so more deferral cannot cost more."""
    train, test = planted(seed=1), planted(seed=2)
    frontier = cost_deferral_frontier(fit_selector(*train), *test)
    costs = frontier["realized_costs"]
    assert costs[-1] <= costs[0] + 1e-9


# -------------------------------------------------- leave-one-dataset-out
def test_leave_one_dataset_out_runs_every_dataset_once():
    result = leave_one_dataset_out(*planted(n_datasets=5))
    assert result["status"] == "ok"
    assert [f["held_out"] for f in result["folds"]] == result["datasets"]
    assert all(f["n_train"] > 0 and f["n_test"] > 0 for f in result["folds"])


def test_leave_one_dataset_out_needs_two_datasets():
    result = leave_one_dataset_out(*planted(n_datasets=1))
    assert result["status"] == "insufficient_data"


def test_omega_beats_tau_only_when_a_real_boundary_exists():
    result = leave_one_dataset_out(*planted(n_datasets=5))
    assert result["omega_advantage"] > 0
    assert result["selector_total_cost"] < result["tau_only_total_cost"]


def test_omega_shows_no_advantage_on_noise():
    """If Omega carries nothing, it must not look better than tau alone."""
    rng = np.random.default_rng(SEED)
    n = 600
    omega = np.array(rng.uniform(0.1, 0.9, n), dtype=object)
    tau = np.array(rng.choice(TAUS, n))
    base = rng.uniform(1, 2, n)
    classical = base + rng.normal(0, 0.3, n)
    tsfm = base + 0.2 + rng.normal(0, 0.3, n)     # classical better on average
    datasets = np.array([f"dataset_{i % 5}" for i in range(n)], dtype=object)
    result = leave_one_dataset_out(omega, tau, classical, tsfm, datasets)
    assert result["omega_advantage"] < result["tau_only_total_cost"] * 0.05


# ------------------------------------------------------- input validation
def test_malformed_inputs_raise():
    omega, tau, classical, tsfm, datasets = planted()
    with pytest.raises(ValueError, match="length"):
        fit_selector(omega[:-1], tau, classical, tsfm, datasets)
    with pytest.raises(ValueError, match="tau must lie strictly"):
        fit_selector(omega, np.ones_like(tau), classical, tsfm, datasets)
    bad = omega.copy()
    bad[0] = "zero"
    with pytest.raises(ValueError, match="numbers or None"):
        fit_selector(bad, tau, classical, tsfm, datasets)


def test_tau_only_baseline_picks_the_cheaper_family():
    omega, tau, classical, tsfm, datasets = planted()
    baseline = fit_tau_only_baseline(tau, classical, tsfm)
    assert baseline["frozen"] is True
    for rule in baseline["rules"].values():
        cheaper = CLASSICAL if rule["always_classical_cost"] <= rule["always_tsfm_cost"] else TSFM
        assert rule["family"] == cheaper
    assert apply_selector(baseline, None, 0.50)[1] == "tau_only"


# ------------------- the frontier must be informative, not just monotone
def test_fuzzy_boundary_gives_a_partial_result_that_deferral_improves():
    """A realistic boundary: right more often than chance, far from perfect."""
    train = planted(seed=1, advantage=0.25, noise=0.30)
    test = planted(seed=2, advantage=0.25, noise=0.30)
    frontier = cost_deferral_frontier(fit_selector(*train), *test)
    captured = [p["fraction_of_oracle_value_captured"] for p in frontier["points"]]
    assert 0.3 < captured[0] < 0.8            # partial at zero deferral
    assert captured[-1] > captured[0] + 0.2   # deferring genuinely buys value
    assert frontier["realized_costs"][-1] < frontier["realized_costs"][0]


def test_total_deferral_captures_everything_by_construction():
    """The degenerate end: deferring everything is the oracle, because this
    frontier does not price the second family's compute. The E4 write-up has
    to say so rather than reporting 1.000 as a selector result."""
    train, test = planted(seed=1), planted(seed=2)
    result = evaluate_selector(fit_selector(*train), *test, margin=1.0)
    assert result["deferral_rate"] == 1.0
    assert result["fraction_of_oracle_value_captured"] == pytest.approx(1.0)
    assert result["realized_cost"] == pytest.approx(result["oracle_cost"])
