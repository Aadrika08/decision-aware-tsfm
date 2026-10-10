"""The statistic on hand-built examples with known answers. Numbers here are made up."""
import unittest
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from decision_aware_tsfm.analysis.k2.statistic import (
    MAX_SEED, bootstrap_interval, bootstrap_rhos, check_bootstrap_parameters, relative_delta,
    sign_counts, spearman, spearman_xy)


def table(omega, delta):
    """A family table whose Delta(A, B) equals the given values: A = 100, B = 100 - delta."""
    delta = np.asarray(delta, dtype=float)
    return pd.DataFrame({"omega": omega, "A": 100.0, "B": 100.0 - delta})


class DeltaAndSpearman(unittest.TestCase):
    def test_delta_is_relative_to_family_a_and_negative_when_a_is_better(self):
        np.testing.assert_allclose(relative_delta([2.0, 4.0], [3.0, 1.0]), [-50.0, 75.0])

    def test_spearman_known_answer(self):
        # Ranks of omega: 1..5. Ranks of Delta: 2,1,4,3,5. Sum of squared rank
        # differences = 4, so rho = 1 - 6*4 / (5*(25-1)) = 0.8.
        rho, p, n = spearman(table([0.1, 0.2, 0.3, 0.4, 0.5], [20, 10, 40, 30, 50]), "A", "B")
        self.assertAlmostEqual(rho, 0.8, places=12)
        self.assertEqual(n, 5)
        self.assertTrue(0 < p < 1)

    def test_perfect_monotone_relations(self):
        for sign, delta in ((1.0, [1, 2, 3, 4]), (-1.0, [4, 3, 2, 1])):
            with self.subTest(sign=sign):
                rho, _, _ = spearman(table([0.1, 0.2, 0.3, 0.9], delta), "A", "B")
                self.assertAlmostEqual(rho, sign, places=12)

    def test_non_finite_rows_are_left_out_and_counted_in_n(self):
        t = table([0.1, 0.2, 0.3, 0.4, 0.5, np.nan], [20, 10, 40, 30, 50, 60])
        t.loc[len(t)] = {"omega": 0.6, "A": 0.0, "B": 1.0}      # division by zero
        rho, _, n = spearman(t, "A", "B")
        self.assertEqual(n, 5)
        self.assertAlmostEqual(rho, 0.8, places=12)

    def test_fewer_than_three_units_gives_nan(self):
        rho, p, n = spearman_xy([0.1, 0.2], [1.0, 2.0])
        self.assertTrue(np.isnan(rho) and np.isnan(p))
        self.assertEqual(n, 2)

    def test_sign_counts(self):
        t = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, np.nan], "B": [2.0, 1.0, 3.0, 5.0, 1.0]})
        self.assertEqual(sign_counts(t, "A", "B"),
                         {"n": 4, "n_a_lower": 2, "n_b_lower": 1, "n_tie": 1})


class Bootstrap(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261004)
        self.x = rng.normal(size=20)
        self.y = 0.8 * self.x + 0.6 * rng.normal(size=20)      # a well-behaved example

    def test_same_seed_same_interval_and_different_seed_differs(self):
        a = bootstrap_interval(self.x, self.y, n_resamples=500, seed=7)
        b = bootstrap_interval(self.x, self.y, n_resamples=500, seed=7)
        c = bootstrap_interval(self.x, self.y, n_resamples=500, seed=8)
        self.assertEqual(a, b)
        self.assertNotEqual((a["ci_low"], a["ci_high"]), (c["ci_low"], c["ci_high"]))

    def test_interval_contains_the_point_estimate_on_a_well_behaved_example(self):
        rho, _, n = spearman_xy(self.x, self.y)
        ci = bootstrap_interval(self.x, self.y, n_resamples=2000, seed=1)
        self.assertLess(ci["ci_low"], rho)
        self.assertGreater(ci["ci_high"], rho)
        self.assertGreaterEqual(ci["ci_low"], -1.0)
        self.assertLessEqual(ci["ci_high"], 1.0)
        self.assertEqual(ci["n_degenerate"], 0)
        self.assertEqual(ci["n_resampled_units"], n)

    def test_interval_is_the_stated_percentiles_of_the_resample_rhos(self):
        rhos = bootstrap_rhos(self.x, self.y, 2000, 4)
        ci = bootstrap_interval(self.x, self.y, n_resamples=2000, seed=4, level=0.95)
        lo, hi = np.percentile(rhos, [2.5, 97.5])
        self.assertAlmostEqual(ci["ci_low"], lo, places=12)
        self.assertAlmostEqual(ci["ci_high"], hi, places=12)
        inside = np.mean((rhos >= ci["ci_low"]) & (rhos <= ci["ci_high"]))
        self.assertAlmostEqual(inside, 0.95, delta=0.005)
        half = bootstrap_interval(self.x, self.y, n_resamples=2000, seed=4, level=0.5)
        self.assertGreater(half["ci_low"], ci["ci_low"])
        self.assertLess(half["ci_high"], ci["ci_high"])

    def test_resample_rhos_equal_scipy_on_the_same_draws_including_ties(self):
        x = np.array([1, 2, 2, 3, 5, 8, 8, 9.0])
        y = np.array([3, 1, 4, 4, 2, 9, 7, 7.0])
        seed, n_resamples = 5, 300
        ours = bootstrap_rhos(x, y, n_resamples, seed)
        draws = np.random.default_rng(seed).integers(0, len(x), size=(n_resamples, len(x)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = np.array([stats.spearmanr(x[d], y[d])[0] for d in draws])
        self.assertTrue(np.array_equal(np.isnan(ours), np.isnan(ref)))
        np.testing.assert_allclose(ours[~np.isnan(ours)], ref[~np.isnan(ref)], atol=1e-12)

    def test_degenerate_resamples_are_counted_and_left_out(self):
        # Three units: a resample that draws one unit three times has no rank
        # variation, so rho is undefined. Count those draws independently.
        x, y = np.array([0.1, 0.2, 0.3]), np.array([1.0, 3.0, 2.0])
        seed, n_resamples = 3, 400
        draws = np.random.default_rng(seed).integers(0, 3, size=(n_resamples, 3))
        expected = int((draws == draws[:, :1]).all(axis=1).sum())
        self.assertGreater(expected, 0)
        ci = bootstrap_interval(x, y, n_resamples=n_resamples, seed=seed)
        self.assertEqual(ci["n_degenerate"], expected)
        self.assertTrue(np.isfinite(ci["ci_low"]) and np.isfinite(ci["ci_high"]))

    def test_cluster_resampling_moves_whole_clusters(self):
        # Four clusters of three identical rows. Resampling clusters is then the
        # same as resampling four units, row for row.
        x4, y4 = np.array([0.1, 0.2, 0.3, 0.4]), np.array([2.0, 1.0, 4.0, 3.0])
        clusters = np.repeat(["a", "b", "c", "d"], 3)
        by_cluster = bootstrap_rhos(np.repeat(x4, 3), np.repeat(y4, 3), 300, 11, clusters)
        by_unit = bootstrap_rhos(x4, y4, 300, 11)
        self.assertTrue(np.array_equal(np.isnan(by_cluster), np.isnan(by_unit)))
        keep = ~np.isnan(by_unit)
        np.testing.assert_allclose(by_cluster[keep], by_unit[keep], atol=1e-12)
        ci = bootstrap_interval(np.repeat(x4, 3), np.repeat(y4, 3), 300, 11, clusters=clusters)
        self.assertEqual(ci["n_resampled_units"], 4)

    def test_too_few_units_gives_no_interval(self):
        ci = bootstrap_interval([0.1, 0.2], [1.0, 2.0], n_resamples=50, seed=1)
        self.assertTrue(np.isnan(ci["ci_low"]) and np.isnan(ci["ci_high"]))

    def test_non_finite_input_is_refused(self):
        good = np.arange(5.0)
        for bad in (np.array([1, 2, np.nan, 4, 5.0]), np.array([1, 2, np.inf, 4, 5.0])):
            for x, y in ((bad, good), (good, bad)):
                with self.subTest(x=x, y=y), self.assertRaises(ValueError) as ctx:
                    bootstrap_interval(x, y, n_resamples=200, seed=1)
                self.assertIn("finite", str(ctx.exception))

    def test_unequal_lengths_and_bad_clusters_are_refused(self):
        with self.assertRaises(ValueError):
            bootstrap_interval(np.arange(5.0), np.arange(4.0), n_resamples=50, seed=1)
        with self.assertRaises(ValueError):
            bootstrap_interval(np.arange(5.0), np.arange(5.0), n_resamples=50, seed=1,
                               clusters=["a", "b"])

    def test_resample_count_and_seed_are_checked(self):
        self.assertEqual(check_bootstrap_parameters(1, 0), (1, 0))
        self.assertEqual(check_bootstrap_parameters(np.int64(200), MAX_SEED), (200, MAX_SEED))
        for n_resamples, seed, part in ((0, 1, "at least 1"), (-5, 1, "at least 1"),
                                        (10, -1, "between 0 and"), (10, MAX_SEED + 1, "between 0 and"),
                                        (2.5, 1, "whole number"), (10, "7", "whole number"),
                                        (True, 1, "whole number")):
            with self.subTest(n_resamples=n_resamples, seed=seed), \
                    self.assertRaises(ValueError) as ctx:
                check_bootstrap_parameters(n_resamples, seed)
            self.assertIn(part, str(ctx.exception))
        for n_resamples, seed in ((0, 1), (10, -1)):
            with self.subTest(n_resamples=n_resamples, seed=seed), self.assertRaises(ValueError):
                bootstrap_interval(self.x, self.y, n_resamples=n_resamples, seed=seed)


if __name__ == "__main__":
    unittest.main()
