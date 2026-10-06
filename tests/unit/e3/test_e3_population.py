"""Closed-form break-even errors and the F-win count as a function of error size."""
import unittest

import numpy as np
import pandas as pd
from scipy import integrate, stats

from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID
from decision_aware_tsfm.analysis.experiments.e3.design import (
    ALTERNATIVES, F_VARIANTS, MEMBERS, innovation_dist)
from decision_aware_tsfm.analysis.experiments.e3.population import (
    ERROR_GRID, breakeven_error, cost_gap, f_win_counts, near_knife_edge, population_delta)
from decision_aware_tsfm.analysis.experiments.e3.scoring import expected_pinball

from . import _shared

LOCATION_VARIANTS = ("loc_small", "primary", "loc_large")


class CostGap(unittest.TestCase):
    def test_gap_equals_adaptive_quadrature_of_cdf_minus_tau(self):
        for mname in ("gaussian", "skewnorm_n8", "student_t3"):
            dist = innovation_dist(MEMBERS[mname])
            for a, b, tau in ((-2.3, -1.5, 0.01), (0.0, 0.25, 0.5), (1.6, 2.4, 0.95), (0.4, -0.7, 0.25)):
                with self.subTest(member=mname, a=a, b=b):
                    ref, _ = integrate.quad(lambda x: dist.cdf(x) - tau, a, b, epsabs=1e-13)
                    self.assertAlmostEqual(cost_gap(dist, a, b, tau), ref, places=11)
        self.assertEqual(cost_gap(stats.norm(), 0.3, 0.3, 0.5), 0.0)

    def test_gap_equals_the_difference_of_two_expected_costs(self):
        # expected_pinball integrates from minus infinity: a different route
        for mname, tau in (("skewnorm_p8", 0.01), ("student_t5", 0.91)):
            dist = innovation_dist(MEMBERS[mname])
            a, b = float(stats.norm.ppf(tau)), float(dist.ppf(tau)) + 0.25
            with self.subTest(member=mname, tau=tau):
                self.assertAlmostEqual(cost_gap(dist, a, b, tau),
                                       expected_pinball(dist, b, tau) - expected_pinball(dist, a, tau),
                                       places=8)

    def test_every_row_of_the_population_table_satisfies_the_identity(self):
        # J_F - J_G = integral from z to Q + b of (CDF - tau), for all 264 rows
        e = _shared.expected()
        self.assertEqual(len(e), len(MEMBERS) * len(TAU_GRID) * 3)
        own = np.array([population_delta(r.member, r.tau, F_VARIANTS[r.variant].location_error)
                        for r in e.itertuples()])
        np.testing.assert_allclose(e.exp_delta.to_numpy(), own, rtol=0, atol=1e-8)
        np.testing.assert_array_equal((e.population_winner == "F").to_numpy(), own < 0)


class BreakEven(unittest.TestCase):
    def test_gaussian_members_and_symmetric_medians_have_no_room_for_F(self):
        b = _shared.breakeven()
        g = b[b.member.isin(["gaussian", "gaussian_b"])]
        self.assertTrue((g.breakeven_pos == 0).all() and (g.breakeven_neg == 0).all())
        med = b[b.member.isin(["student_t5", "student_t3"]) & (b.tau == 0.5)]
        self.assertEqual(len(med), 2)
        self.assertTrue((med.breakeven_pos == 0).all())

    def test_when_G_errs_towards_the_same_side_the_break_even_is_the_quantile_gap(self):
        b = _shared.breakeven()
        same_side = b[b.g_shape_error > 1e-9]
        self.assertGreater(len(same_side), 20)
        np.testing.assert_allclose(same_side.breakeven_pos, same_side.gaussian_quantile - same_side.true_quantile,
                                   atol=1e-12)
        other = b[b.g_shape_error < -1e-9]
        np.testing.assert_allclose(other.breakeven_neg, other.true_quantile - other.gaussian_quantile,
                                   atol=1e-12)

    def test_break_even_is_where_the_two_expected_costs_cross(self):
        # checked with expected_pinball (integration from minus infinity), not with cost_gap
        for mname, tau in (("skewnorm_p8", 0.01), ("student_t3", 0.09), ("skewnorm_n8", 0.95),
                           ("skewnorm_n2", 0.5)):
            dist = innovation_dist(MEMBERS[mname])
            z, q = float(stats.norm.ppf(tau)), float(dist.ppf(tau))
            jg = expected_pinball(dist, z, tau)
            for sign in (1, -1):
                with self.subTest(member=mname, tau=tau, sign=sign):
                    be = breakeven_error(dist, tau, sign)
                    self.assertGreater(be, 0.0)
                    self.assertAlmostEqual(expected_pinball(dist, q + sign * be, tau), jg, places=8)
                    self.assertLess(expected_pinball(dist, q + sign * 0.9 * be, tau), jg)      # F cheaper
                    self.assertGreater(expected_pinball(dist, q + sign * 1.1 * be, tau), jg)   # G cheaper

    def test_mirror_members_have_mirrored_break_evens(self):
        b = _shared.breakeven().set_index(["member", "tau"])
        for tau in TAU_GRID:
            with self.subTest(tau=tau):
                mirror = round(1 - tau, 2)
                self.assertAlmostEqual(b.loc[("skewnorm_p8", tau), "breakeven_pos"],
                                       b.loc[("skewnorm_n8", mirror), "breakeven_neg"], places=9)
                self.assertAlmostEqual(b.loc[("student_t3", tau), "breakeven_pos"],
                                       b.loc[("student_t3", mirror), "breakeven_neg"], places=9)


class WinCounts(unittest.TestCase):
    def test_hand_built_count(self):
        be = pd.DataFrame({"member": [ALTERNATIVES[0]] * 4 + ["gaussian"],
                           "tau": [0.01, 0.5, 0.5, 0.99, 0.5],
                           "breakeven_pos": [0.30, 0.10, 0.25, 0.60, 0.0],
                           "breakeven_neg": [0.05, 0.10, 0.00, 0.40, 0.0]})
        w = f_win_counts(be, errors=(0.0, 0.10, 0.25, 0.5)).set_index("location_error")
        self.assertEqual(w.n_cells.tolist(), [4, 4, 4, 4])                    # Gaussian row not a cell
        self.assertEqual(w.n_F_wins_positive_error.tolist(), [4, 3, 2, 1])    # strict: 0.10 and 0.25 are ties
        self.assertEqual(w.n_F_wins_positive_error_tails.tolist(), [2, 2, 2, 1])
        self.assertEqual(w.n_F_wins_positive_error_centre.tolist(), [2, 1, 0, 0])
        self.assertEqual(w.n_F_wins_negative_error.tolist(), [3, 1, 1, 0])
        self.assertEqual(w.declared_variant.tolist(), ["", "loc_small", "primary", "loc_large"])

    def test_counts_at_the_declared_errors_equal_the_population_table(self):
        w = f_win_counts(_shared.breakeven()).set_index("location_error")
        e = _shared.expected()
        e = e[e.member.isin(ALTERNATIVES)]
        for vname in LOCATION_VARIANTS:
            with self.subTest(variant=vname):
                n = int((e[e.variant == vname].population_winner == "F").sum())
                self.assertEqual(w.loc[F_VARIANTS[vname].location_error, "n_F_wins_positive_error"], n)
        self.assertTrue((w.n_cells == len(ALTERNATIVES) * len(TAU_GRID)).all())

    def test_count_falls_as_the_error_grows_and_error_zero_leaves_only_exact_ties(self):
        w = f_win_counts(_shared.breakeven())
        self.assertEqual(w.location_error.tolist(), list(ERROR_GRID))
        self.assertTrue((np.diff(w.n_F_wins_positive_error) <= 0).all())
        # with no error F is the true quantile and loses only the two exact ties (t medians)
        self.assertEqual(int(w.n_F_wins_positive_error.iloc[0]), 66 - 2)
        self.assertLess(int(w.n_F_wins_positive_error.iloc[-1]), 5)

    def test_cells_near_the_primary_error_are_listed_with_their_margin(self):
        b = _shared.breakeven()
        k = near_knife_edge(b, 0.25, band=0.05)
        alt = b[b.member.isin(ALTERNATIVES)]
        self.assertEqual(len(k), int(((alt.breakeven_pos - 0.25).abs() <= 0.05).sum()))
        np.testing.assert_allclose(k.margin, k.breakeven_pos - 0.25)
        self.assertTrue((np.diff(k.margin) >= 0).all())
        self.assertTrue((k.margin < 0).any() and (k.margin > 0).any())


if __name__ == "__main__":
    unittest.main()
