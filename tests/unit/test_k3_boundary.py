"""Synthetic ground-truth tests for K3 boundary analysis.

Each test plants a known answer and checks the analysis recovers it.
No real data, no M5/PJM, no model inference.
"""
import numpy as np
import pytest

from decision_aware_tsfm.analysis.k3 import (
    count_sign_reversals,
    family_cost_difference,
    fit_monotone_boundary,
    oracle_value_best_threshold,
)

SEED = 20260921


def planted(n=80, boundary=0.5, classical_below=True, noise=0.05, seed=SEED):
    """Series with a known family boundary at `boundary` in Omega."""
    rng = np.random.default_rng(seed)
    omega = np.sort(rng.uniform(0.0, 1.0, n))
    below = omega <= boundary
    base = rng.uniform(1.0, 2.0, n)
    advantage = 0.5
    cheaper_classical = below if classical_below else ~below
    cost_classical = base + np.where(cheaper_classical, 0.0, advantage) + rng.normal(0, noise, n)
    cost_tsfm = base + np.where(cheaper_classical, advantage, 0.0) + rng.normal(0, noise, n)
    return omega, cost_classical, cost_tsfm


# ------------------------------------------------ planted boundary recovered
def test_planted_boundary_recovered():
    omega, c, t = planted(boundary=0.5, classical_below=True)
    delta = family_cost_difference(c, t)

    rev = count_sign_reversals(omega, delta)
    assert rev["status"] == "ok" and rev["sign_reversals"] == 1

    ov = oracle_value_best_threshold(omega, c, t)
    assert ov["best_orientation"] == "classical_below"
    assert abs(ov["orientations"]["classical_below"]["threshold"] - 0.5) < 0.05
    assert ov["fraction_of_oracle_value_captured"] > 0.9


# ------------------------- fix 3: reversed orientation is found, not missed
def test_reversed_orientation_recovered():
    omega, c, t = planted(boundary=0.5, classical_below=False)
    ov = oracle_value_best_threshold(omega, c, t)
    assert ov["best_orientation"] == "tsfm_below"
    assert ov["fraction_of_oracle_value_captured"] > 0.9
    # the wrong orientation should capture essentially nothing
    assert ov["orientations"]["classical_below"]["fraction_captured"] < 0.1


# ------------------------------ fix 1: both directions reported, right one wins
def test_monotone_direction_both_reported():
    omega, c, t = planted(classical_below=True)  # delta falls as Omega rises
    fit = fit_monotone_boundary(omega, family_cost_difference(c, t))
    assert set(fit["monotone"]) == {"increasing", "decreasing"}
    assert fit["monotone"]["decreasing"]["cv_mse"] < fit["monotone"]["increasing"]["cv_mse"]
    interval = fit["monotone"]["decreasing"]["crossing_omega_interval"]
    assert interval is not None and interval[0] <= 0.55 and interval[1] >= 0.45


def test_preregistered_direction_fits_only_that_direction():
    omega, c, t = planted()
    fit = fit_monotone_boundary(omega, family_cost_difference(c, t), direction="decreasing")
    assert set(fit["monotone"]) == {"decreasing"}


def test_grouped_cross_validation_runs():
    omega, c, t = planted(n=60)
    groups = np.repeat(np.arange(6), 10)  # six synthetic "datasets"
    fit = fit_monotone_boundary(omega, family_cost_difference(c, t), groups=groups)
    assert fit["status"] == "ok"


# ------------------------------------------------- no boundary -> no boundary
def test_constant_winner_has_no_oracle_gap():
    rng = np.random.default_rng(SEED)
    omega = rng.uniform(0, 1, 50)
    c = rng.uniform(2, 3, 50)
    t = c - 0.5  # TSFM always cheaper, everywhere
    rev = count_sign_reversals(omega, family_cost_difference(c, t))
    assert rev["sign_reversals"] == 0
    ov = oracle_value_best_threshold(omega, c, t)
    assert ov["status"] == "no_oracle_gap"
    assert ov["fraction_of_oracle_value_captured"] is None


def test_unstructured_noise_captures_little():
    rng = np.random.default_rng(SEED)
    n = 200
    omega = rng.uniform(0, 1, n)
    base = rng.uniform(1, 2, n)
    c = base + rng.normal(0, 0.3, n)
    t = base + rng.normal(0, 0.3, n)  # winner is random, unrelated to Omega
    ov = oracle_value_best_threshold(omega, c, t)
    assert ov["fraction_of_oracle_value_captured"] < 0.3


# ------------------------------- fix 4: baseline is best fixed, not worst
def test_best_fixed_baseline_is_stricter_than_worst_fixed():
    omega, c, t = planted(boundary=0.8)  # unbalanced: most series favour one side
    best = oracle_value_best_threshold(omega, c, t, baseline="best_fixed")
    worst = oracle_value_best_threshold(omega, c, t, baseline="worst_fixed")
    assert best["baseline_cost"] <= worst["baseline_cost"]
    assert best["fraction_of_oracle_value_captured"] <= worst["fraction_of_oracle_value_captured"]


# ------------------------------- fix 2 + 5: explicit insufficient_data
def test_single_real_smoke_series_is_insufficient():
    # ett1/H channel 0 from the #3 release; omega value is a placeholder.
    omega, c, t = [0.42], [4.315344], [5.514882]
    delta = family_cost_difference(c, t)
    for result in (count_sign_reversals(omega, delta),
                   fit_monotone_boundary(omega, delta),
                   oracle_value_best_threshold(omega, c, t)):
        assert result["status"] == "insufficient_data"
        assert result["n_distinct_omega"] == 1


def test_flexible_fit_withheld_below_threshold():
    omega, c, t = planted(n=40)
    omega = np.round(omega * 5) / 5  # only ~6 distinct Omega values
    fit = fit_monotone_boundary(omega, family_cost_difference(c, t))
    assert fit["status"] == "ok"
    assert fit["flexible"]["status"] == "insufficient_data"


def test_ties_cannot_manufacture_reversals():
    omega = np.array([0.2, 0.2, 0.2, 0.6, 0.6, 0.9])
    delta = np.array([1.0, -0.1, 1.0, -1.0, -1.0, -1.0])  # tie group mean > 0
    assert count_sign_reversals(omega, delta)["sign_reversals"] == 1


# ------------------------------------------ malformed input fails loudly
def test_malformed_input_raises():
    with pytest.raises(ValueError):
        count_sign_reversals([0.1, 0.2, np.nan], [1, 2, 3])
    with pytest.raises(ValueError):
        oracle_value_best_threshold([0.1, 0.2], [1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        fit_monotone_boundary([0.1, 0.2, 0.3], [1, 2, 3], direction="sideways")


# --------------- Veer's sign-off condition: count per dataset, not per series
def test_many_series_from_one_dataset_are_not_enough(tmp_path=None):
    """30 series from a single dataset is one dataset's worth of evidence."""
    omega, c, t = planted(n=30)
    datasets = ["ett1_H"] * 30
    delta = family_cost_difference(c, t)

    for result in (count_sign_reversals(omega, delta, datasets=datasets),
                   fit_monotone_boundary(omega, delta, datasets=datasets),
                   oracle_value_best_threshold(omega, c, t, datasets=datasets)):
        assert result["status"] == "insufficient_data"
        assert result["unit"] == "dataset"
        assert result["n_distinct_omega"] == 1

    # the same data without labels still passes the weaker per-value count
    assert count_sign_reversals(omega, delta)["status"] == "ok"


def test_three_datasets_clear_the_monotone_floor(tmp_path=None):
    omega, c, t = planted(n=30)
    datasets = ["ett1_H"] * 10 + ["electricity_15T"] * 10 + ["solar_10T"] * 10
    result = count_sign_reversals(omega, family_cost_difference(c, t), datasets=datasets)
    assert result["status"] == "ok"
    assert result["unit"] == "dataset" and result["n_distinct_omega"] == 3


def test_flexible_fit_needs_eight_datasets(tmp_path=None):
    omega, c, t = planted(n=40)
    datasets = [f"dataset_{i % 5}" for i in range(40)]  # 5 datasets, below the 8 floor
    fit = fit_monotone_boundary(omega, family_cost_difference(c, t), datasets=datasets)
    assert fit["status"] == "ok" and fit["n_distinct_omega"] == 5
    assert fit["flexible"]["status"] == "insufficient_data"
    assert fit["flexible"]["unit"] == "dataset"


def test_mismatched_dataset_labels_raise(tmp_path=None):
    omega, c, t = planted(n=10)
    with pytest.raises(ValueError, match="datasets length"):
        count_sign_reversals(omega, family_cost_difference(c, t), datasets=["a", "b"])
