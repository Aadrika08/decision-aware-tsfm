"""Synthetic ground-truth tests for the E4 Omega-plus-tau selector.

Invented data with planted answers. No real forecasts, no M5/PJM, no inference.

unittest, to match the repository convention and the CI workflows. These also
run under pytest.
"""
import copy
import unittest

import numpy as np

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


def noise_only(n, seed=SEED, tsfm_offset=0.0, n_datasets=4):
    """Costs unrelated to Omega; tsfm_offset makes one family better on average."""
    rng = np.random.default_rng(seed)
    omega = np.array(rng.uniform(0.1, 0.9, n), dtype=object)
    tau = np.array(rng.choice(TAUS, n))
    base = rng.uniform(1, 2, n)
    classical = base + rng.normal(0, 0.3, n)
    tsfm = base + tsfm_offset + rng.normal(0, 0.3, n)
    datasets = np.array([f"dataset_{i % n_datasets}" for i in range(n)], dtype=object)
    return omega, tau, classical, tsfm, datasets


class FitAndFreezeTests(unittest.TestCase):
    def test_fitted_rule_recovers_the_planted_boundary(self):
        spec = fit_selector(*planted())
        self.assertIs(spec["frozen"], True)
        for tau in ("0.50", "0.91"):
            rule = spec["rules"][tau]
            self.assertEqual(rule["status"], "ok")
            self.assertEqual(rule["orientation"], "classical_below")
            self.assertLess(abs(rule["threshold"] - 0.5), 0.06)

    def test_reversed_orientation_recovered(self):
        spec = fit_selector(*planted(classical_below=False))
        self.assertEqual(spec["rules"]["0.50"]["orientation"], "tsfm_below")

    def test_spec_is_plain_data_and_applying_never_refits(self):
        spec = fit_selector(*planted())
        before = copy.deepcopy(spec)
        for _ in range(3):
            apply_selector(spec, 0.2, 0.50)
        self.assertEqual(spec, before)

    def test_unfrozen_spec_refused(self):
        spec = fit_selector(*planted())
        spec["frozen"] = False
        with self.assertRaisesRegex(ValueError, "not frozen"):
            apply_selector(spec, 0.2, 0.50)


class DeferralReasonTests(unittest.TestCase):
    def test_defers_on_undefined_omega(self):
        spec = fit_selector(*planted())
        self.assertEqual(apply_selector(spec, None, 0.50), (DEFER, "omega_undefined"))

    def test_defers_outside_the_fitted_range(self):
        spec = fit_selector(*planted())
        self.assertEqual(apply_selector(spec, 0.999, 0.50), (DEFER, "omega_out_of_range"))
        self.assertEqual(apply_selector(spec, 0.001, 0.50), (DEFER, "omega_out_of_range"))

    def test_defers_on_weak_evidence_near_the_threshold(self):
        spec = fit_selector(*planted())
        threshold = spec["rules"]["0.50"]["threshold"]
        self.assertEqual(
            apply_selector(spec, threshold + 0.01, 0.50, margin=0.1)[1], "weak_evidence")
        self.assertEqual(
            apply_selector(spec, threshold + 0.01, 0.50, margin=0.0)[1], "rule")

    def test_defers_on_an_unfitted_tau(self):
        spec = fit_selector(*planted())
        self.assertEqual(apply_selector(spec, 0.3, 0.25), (DEFER, "unfitted_tau"))

    def test_defers_when_too_few_datasets(self):
        spec = fit_selector(*planted(n_datasets=2))
        self.assertEqual(spec["rules"]["0.50"]["status"], "insufficient_data")
        self.assertEqual(apply_selector(spec, 0.3, 0.50), (DEFER, "insufficient_data"))

    def test_choices_follow_the_planted_sides(self):
        spec = fit_selector(*planted())
        self.assertEqual(apply_selector(spec, 0.2, 0.50)[0], CLASSICAL)
        self.assertEqual(apply_selector(spec, 0.8, 0.50)[0], TSFM)

    def test_negative_margin_refused(self):
        spec = fit_selector(*planted())
        with self.assertRaisesRegex(ValueError, "non-negative"):
            apply_selector(spec, 0.3, 0.50, margin=-0.1)


class EvaluationTests(unittest.TestCase):
    def test_evaluation_on_unseen_data_captures_planted_value(self):
        result = evaluate_selector(fit_selector(*planted(seed=1)), *planted(seed=2))
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["fraction_of_oracle_value_captured"], 0.8)
        self.assertLess(result["deferral_rate"], 0.1)

    def test_noise_only_data_captures_little(self):
        omega, tau, classical, tsfm, datasets = noise_only(400)
        spec = fit_selector(omega, tau, classical, tsfm, datasets)
        result = evaluate_selector(spec, omega, tau, classical, tsfm, datasets)
        self.assertLess(result["fraction_of_oracle_value_captured"], 0.5)

    def test_constant_winner_reports_no_oracle_gap(self):
        omega, tau, classical, _, datasets = planted()
        tsfm = classical - 0.5  # TSFM cheaper everywhere
        spec = fit_selector(omega, tau, classical, tsfm, datasets)
        result = evaluate_selector(spec, omega, tau, classical, tsfm, datasets)
        self.assertEqual(result["status"], "no_oracle_gap")
        self.assertIsNone(result["fraction_of_oracle_value_captured"])

    def test_undefined_omega_counted_as_deferral_not_dropped(self):
        omega, tau, classical, tsfm, datasets = planted(omega_undefined=10)
        spec = fit_selector(omega, tau, classical, tsfm, datasets)
        result = evaluate_selector(spec, omega, tau, classical, tsfm, datasets)
        self.assertEqual(result["reasons"].get("omega_undefined"), 10)
        self.assertEqual(result["n"], len(tau))  # nothing silently dropped


class FrontierTests(unittest.TestCase):
    def test_frontier_deferral_rate_is_monotone_in_margin(self):
        frontier = cost_deferral_frontier(fit_selector(*planted(seed=1)),
                                          *planted(seed=2))
        rates = frontier["deferral_rates"]
        self.assertEqual(rates, sorted(rates))
        self.assertLess(rates[0], rates[-1])

    def test_frontier_cost_falls_as_deferral_rises(self):
        """Priced at the oracle, more deferral cannot cost more."""
        frontier = cost_deferral_frontier(fit_selector(*planted(seed=1)),
                                          *planted(seed=2))
        costs = frontier["realized_costs"]
        self.assertLessEqual(costs[-1], costs[0] + 1e-9)

    def test_fuzzy_boundary_gives_a_partial_result_that_deferral_improves(self):
        """A realistic boundary: right more often than chance, far from perfect."""
        train = planted(seed=1, advantage=0.25, noise=0.30)
        test = planted(seed=2, advantage=0.25, noise=0.30)
        frontier = cost_deferral_frontier(fit_selector(*train), *test)
        captured = [p["fraction_of_oracle_value_captured"] for p in frontier["points"]]
        self.assertGreater(captured[0], 0.3)   # partial at zero deferral
        self.assertLess(captured[0], 0.8)
        self.assertGreater(captured[-1], captured[0] + 0.2)  # deferring buys value
        self.assertLess(frontier["realized_costs"][-1], frontier["realized_costs"][0])

    def test_total_deferral_captures_everything_by_construction(self):
        """The degenerate end: priced at the oracle, deferring everything is the
        oracle. The E4 write-up must say so rather than reporting 1.000 as a
        selector result."""
        result = evaluate_selector(fit_selector(*planted(seed=1)), *planted(seed=2),
                                   margin=1.0)
        self.assertEqual(result["deferral_rate"], 1.0)
        self.assertAlmostEqual(result["fraction_of_oracle_value_captured"], 1.0)
        self.assertAlmostEqual(result["realized_cost"], result["oracle_cost"])


class LeaveOneDatasetOutTests(unittest.TestCase):
    def test_runs_every_dataset_once(self):
        result = leave_one_dataset_out(*planted(n_datasets=5))
        self.assertEqual(result["status"], "ok")
        self.assertEqual([f["held_out"] for f in result["folds"]], result["datasets"])
        for fold in result["folds"]:
            self.assertGreater(fold["n_train"], 0)
            self.assertGreater(fold["n_test"], 0)

    def test_needs_two_datasets(self):
        result = leave_one_dataset_out(*planted(n_datasets=1))
        self.assertEqual(result["status"], "insufficient_data")

    def test_omega_beats_tau_only_when_a_real_boundary_exists(self):
        result = leave_one_dataset_out(*planted(n_datasets=5))
        self.assertGreater(result["omega_advantage"], 0)
        self.assertLess(result["selector_total_cost"], result["tau_only_total_cost"])

    def test_omega_shows_no_advantage_on_noise(self):
        """If Omega carries nothing, it must not look better than tau alone."""
        result = leave_one_dataset_out(*noise_only(600, tsfm_offset=0.2, n_datasets=5))
        self.assertLess(result["omega_advantage"],
                        result["tau_only_total_cost"] * 0.05)


class InputValidationTests(unittest.TestCase):
    def test_malformed_inputs_raise(self):
        omega, tau, classical, tsfm, datasets = planted()
        with self.assertRaisesRegex(ValueError, "length"):
            fit_selector(omega[:-1], tau, classical, tsfm, datasets)
        with self.assertRaisesRegex(ValueError, "tau must lie strictly"):
            fit_selector(omega, np.ones_like(tau), classical, tsfm, datasets)
        bad = omega.copy()
        bad[0] = "zero"
        with self.assertRaisesRegex(ValueError, "numbers or None"):
            fit_selector(bad, tau, classical, tsfm, datasets)

    def test_tau_only_baseline_picks_the_cheaper_family(self):
        _, tau, classical, tsfm, _ = planted()
        baseline = fit_tau_only_baseline(tau, classical, tsfm)
        self.assertIs(baseline["frozen"], True)
        for rule in baseline["rules"].values():
            cheaper = (CLASSICAL if rule["always_classical_cost"] <= rule["always_tsfm_cost"]
                       else TSFM)
            self.assertEqual(rule["family"], cheaper)
        self.assertEqual(apply_selector(baseline, None, 0.50)[1], "tau_only")


class IdenticalOmegaTests(unittest.TestCase):
    """Veer's report on #54: a verdict, not a crash."""

    def test_identical_omega_across_datasets_is_insufficient_not_a_crash(self):
        """Enough datasets, but all sharing one Omega value: a threshold has
        nowhere to sit, so this must be insufficient_data, not a TypeError."""
        spec = fit_selector([0.5] * 3, [0.5] * 3, [1.0] * 3, [2.0] * 3,
                            ["a", "b", "c"])
        rule = spec["rules"]["0.50"]
        self.assertEqual(rule["status"], "insufficient_data")
        self.assertEqual(rule["n_distinct_omega"], 1)
        self.assertEqual(rule["n_datasets"], 3)
        self.assertEqual(apply_selector(spec, 0.5, 0.50), (DEFER, "insufficient_data"))

    def test_two_distinct_omega_values_are_enough_to_place_a_threshold(self):
        spec = fit_selector([0.2, 0.2, 0.8, 0.8], [0.5] * 4, [1.0, 1.0, 5.0, 5.0],
                            [5.0, 5.0, 1.0, 1.0], ["a", "b", "c", "d"])
        rule = spec["rules"]["0.50"]
        self.assertEqual(rule["status"], "ok")
        self.assertGreater(rule["threshold"], 0.2)
        self.assertLess(rule["threshold"], 0.8)


class DeferralPricingTests(unittest.TestCase):
    """The caller's validation comparator, not the oracle."""

    def test_validation_priced_deferral_costs_more_than_the_oracle(self):
        """Validation picks on validation origins and pays that family's test
        cost, so it cannot beat hindsight. Pricing at the oracle flatters it."""
        omega, tau, classical, tsfm, datasets = planted(seed=2)
        spec = fit_selector(*planted(seed=1))
        validation = np.maximum(classical, tsfm)  # worst case: validation chose wrong

        oracle_priced = evaluate_selector(spec, omega, tau, classical, tsfm,
                                          datasets, margin=1.0)
        real_priced = evaluate_selector(spec, omega, tau, classical, tsfm, datasets,
                                        margin=1.0, deferral_cost=validation)

        self.assertEqual(oracle_priced["deferral_pricing"], "oracle_upper_bound")
        self.assertEqual(real_priced["deferral_pricing"], "validation_comparator")
        self.assertGreater(real_priced["realized_cost"], oracle_priced["realized_cost"])
        self.assertLess(real_priced["fraction_of_oracle_value_captured"], 1.0)

    def test_oracle_pricing_is_labelled_a_bound_not_a_result(self):
        result = evaluate_selector(fit_selector(*planted(seed=1)), *planted(seed=2))
        self.assertEqual(result["deferral_pricing"], "oracle_upper_bound")

    def test_deferral_cost_length_is_checked(self):
        omega, tau, classical, tsfm, datasets = planted()
        spec = fit_selector(omega, tau, classical, tsfm, datasets)
        with self.assertRaisesRegex(ValueError, "deferral_cost length"):
            evaluate_selector(spec, omega, tau, classical, tsfm, datasets,
                              deferral_cost=np.ones(3))

    def test_frontier_with_validation_pricing_need_not_fall_to_the_oracle(self):
        """With honest pricing, deferring everything no longer scores 1.000."""
        omega, tau, classical, tsfm, datasets = planted(seed=2)
        spec = fit_selector(*planted(seed=1))
        validation = np.maximum(classical, tsfm)
        frontier = cost_deferral_frontier(spec, omega, tau, classical, tsfm,
                                          datasets, deferral_cost=validation)
        self.assertLess(frontier["points"][-1]["fraction_of_oracle_value_captured"], 1.0)
        for point in frontier["points"]:
            self.assertEqual(point["deferral_pricing"], "validation_comparator")


if __name__ == "__main__":
    unittest.main()
