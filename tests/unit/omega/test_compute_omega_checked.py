"""Tests for compute_omega_checked: undefined Omega is None with a reason,
never 0.0, and defined Omega equals compute_omega exactly.

Run from the repo root:
    python -m unittest tests.unit.omega.test_compute_omega_checked -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from decision_aware_tsfm.omega.compute import (  # noqa: E402
    NEAR_FLAT_SHARE,
    compute_omega,
    compute_omega_checked,
    mode_share,
)

RNG = np.random.default_rng(0)


class UndefinedOmegaTests(unittest.TestCase):
    def assertUndefined(self, result, status):
        value, got = result
        self.assertIsNone(value)
        self.assertEqual(got, status)

    def test_empty_history(self):
        self.assertUndefined(compute_omega_checked([]), "empty_history")
        self.assertUndefined(compute_omega_checked([1.0]), "empty_history")

    def test_all_nan_is_empty(self):
        self.assertUndefined(compute_omega_checked([np.nan] * 50), "empty_history")

    def test_two_points_diffs_is_empty(self):
        self.assertUndefined(compute_omega_checked([1.0, 2.0], diffs=True), "empty_history")

    def test_below_floor(self):
        self.assertUndefined(
            compute_omega_checked(RNG.normal(size=999), min_finite=1000), "below_floor"
        )

    def test_floor_counts_finite_points_only(self):
        x = RNG.normal(size=1500)
        x[:600] = np.nan
        self.assertUndefined(compute_omega_checked(x, min_finite=1000), "below_floor")

    def test_constant_is_flat(self):
        self.assertUndefined(compute_omega_checked(np.full(2000, 3.0)), "flat_history")

    def test_linear_trend_diffs_are_flat(self):
        self.assertUndefined(
            compute_omega_checked(np.arange(2000.0), diffs=True), "flat_history"
        )

    def test_flat_with_one_spike_is_near_flat(self):
        for n, pos in [(1000, 100), (4096, 2048), (20000, 18000)]:
            x = np.zeros(n)
            x[pos] = 5.0
            self.assertUndefined(compute_omega_checked(x), "near_flat")
            self.assertUndefined(compute_omega_checked(x, diffs=True), "near_flat")

    def test_step_series_diffs_are_near_flat_levels_are_not(self):
        x = np.r_[np.zeros(1000), np.ones(1000)]
        self.assertUndefined(compute_omega_checked(x, diffs=True), "near_flat")
        value, status = compute_omega_checked(x)
        self.assertEqual(status, "ok")
        self.assertGreater(value, 0.0)


class DefinedOmegaTests(unittest.TestCase):
    def test_matches_compute_omega_exactly(self):
        t = np.arange(3000)
        for x in [np.sin(2 * np.pi * t / 24) + RNG.normal(0, 0.3, 3000), RNG.normal(size=3000)]:
            for diffs in (False, True):
                value, status = compute_omega_checked(x, diffs=diffs)
                self.assertEqual(status, "ok")
                self.assertEqual(value, compute_omega(x, diffs=diffs))

    def test_nan_points_are_dropped_like_compute_omega(self):
        x = np.sin(np.arange(2000) / 7.0)
        x[::50] = np.nan
        value, status = compute_omega_checked(x)
        self.assertEqual(status, "ok")
        self.assertEqual(value, compute_omega(x))

    def test_defined_values_are_never_zero(self):
        for x in [RNG.normal(size=1000), (RNG.random(3000) < 0.02) * 1.0]:
            value, status = compute_omega_checked(x)
            self.assertEqual(status, "ok")
            self.assertGreater(value, 0.0)

    def test_intermittent_series_below_threshold_stays_defined(self):
        x = (RNG.random(3000) < 0.02) * 1.0
        self.assertLess(mode_share(x), NEAR_FLAT_SHARE)
        self.assertEqual(compute_omega_checked(x)[1], "ok")


class ModeShareTests(unittest.TestCase):
    def test_values(self):
        self.assertIsNone(mode_share([]))
        self.assertEqual(mode_share([1.0, 1.0, 2.0, np.nan]), 2 / 3)
        self.assertEqual(mode_share(np.zeros(10)), 1.0)


if __name__ == "__main__":
    unittest.main()
