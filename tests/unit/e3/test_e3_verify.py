"""Verification tables: the costs depend on the innovations only."""
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID
from decision_aware_tsfm.analysis.experiments.e3.design import (
    F_VARIANTS, MEMBERS, N_CONTEXT, N_TOTAL, STRUCTURES, innovation_dist, sine_signal)
from decision_aware_tsfm.analysis.experiments.e3.generator import describe_series
from decision_aware_tsfm.analysis.experiments.e3.scoring import score_dataset
from decision_aware_tsfm.analysis.experiments.e3.verify import (
    compare_with_engine, innovation_only_costs, no_series_simulation, no_series_vs_matched,
    sign_by_match_status)

from . import _shared

TAUS = np.asarray(TAU_GRID)


def _values(series):
    return pd.concat([pd.DataFrame({"series_id": sid, "position": np.arange(x.size, dtype="int64"),
                                    "value": x}) for sid, x in series.items()], ignore_index=True)


class InnovationOnly(unittest.TestCase):
    def test_costs_and_winners_equal_the_engine_for_every_variant(self):
        u = _shared.units_all()
        innov = innovation_only_costs(_shared.dataset()[0], _shared.scored_rows())
        c = compare_with_engine(u, innov).set_index("variant")
        self.assertEqual(set(c.index), set(F_VARIANTS) | {"all"})
        self.assertEqual(c.loc["all", "n_units"], len(_shared.SCORED_MEMBERS) * len(TAU_GRID) * len(F_VARIANTS))
        self.assertLess(c.max_abs_diff_J_G.max(), 1e-12)
        self.assertLess(c.max_abs_diff_J_F.max(), 1e-12)
        self.assertEqual(int(c.n_winner_mismatches.sum()), 0)

    def test_hand_computed_cost_from_innovations(self):
        # one series, G at tau 0.25, written out without any E3 helper
        rows = _shared.scored_rows()
        r = rows[rows.member == "skewnorm_p8"].iloc[0]
        x = _shared.values_of(r.series_id)
        e = x[N_CONTEXT:] - 0.5 * x[N_CONTEXT - 1:-1]                    # phi = 0.50
        u = e - stats.norm.ppf(0.25)
        want_g = float(np.mean(np.where(u >= 0, 0.25 * u, -0.75 * u)))
        q = innovation_dist(MEMBERS["skewnorm_p8"]).ppf(0.25) + 0.25
        u = e - q
        want_f = float(np.mean(np.where(u >= 0, 0.25 * u, -0.75 * u)))
        got = innovation_only_costs(_shared.dataset()[0], rows[rows.member == "skewnorm_p8"],
                                    variants=["primary"])
        got = got[np.isclose(got.tau, 0.25)].iloc[0]
        self.assertAlmostEqual(got.J_G_innov, want_g, places=12)
        self.assertAlmostEqual(got.J_F_innov, want_f, places=12)
        self.assertEqual(got.winner_innov, "F" if want_f < want_g else "G")

    def test_a_disagreement_with_the_engine_is_reported(self):
        u = _shared.units_all().copy()
        innov = innovation_only_costs(_shared.dataset()[0], _shared.scored_rows())
        u.loc[u.index[0], "J_F"] += 0.01
        u.loc[u.index[1], "winner"] = "F" if u.winner.iloc[1] == "G" else "G"
        c = compare_with_engine(u, innov).set_index("variant")
        self.assertAlmostEqual(c.loc["all", "max_abs_diff_J_F"], 0.01, places=9)
        self.assertEqual(c.loc["all", "n_winner_mismatches"], 1)
        with self.assertRaises(ValueError):
            compare_with_engine(u.iloc[5:], innov)

    def test_same_innovations_in_two_structures_give_the_same_engine_costs(self):
        # The identity behind the note's status paragraph, shown through the ENGINE: put one
        # innovation sequence into an AR(1) series and into the sine series. Omega differs,
        # the series differ, the costs and winners of G and F do not.
        mb = MEMBERS["skewnorm_n8"]
        e = innovation_dist(mb).rvs(size=N_TOTAL, random_state=np.random.default_rng(77))
        ar, sine = STRUCTURES["ar1_phi0.95"], STRUCTURES["sine_sd1.0"]
        x_ar, state = np.empty(N_TOTAL), 0.0
        for t in range(N_TOTAL):
            state = ar.phi * state + e[t]
            x_ar[t] = state
        x_sine = sine_signal(N_TOTAL) + e
        rows = pd.DataFrame([describe_series(x_ar, ar, mb, 0), describe_series(x_sine, sine, mb, 0)])
        values = _values(dict(zip(rows.series_id, (x_ar, x_sine))))
        u = score_dataset(values, rows, variants=["primary", "empirical"])
        a = u[u.structure == ar.name].sort_values(["variant", "tau"]).reset_index(drop=True)
        b = u[u.structure == sine.name].sort_values(["variant", "tau"]).reset_index(drop=True)
        self.assertGreater(abs(rows.omega_levels.iloc[0] - rows.omega_levels.iloc[1]), 0.1)
        prim = a.variant == "primary"
        np.testing.assert_allclose(a.J_G, b.J_G, atol=1e-12)
        np.testing.assert_allclose(a.J_F[prim], b.J_F[prim], atol=1e-12)
        self.assertTrue((a.winner[prim] == b.winner[prim]).all())
        # the empirical F estimates its quantiles from the context, where AR(1) loses position 0
        np.testing.assert_allclose(a.J_F[~prim], b.J_F[~prim], atol=2e-3)


class NoSeries(unittest.TestCase):
    def test_reproducible_and_shaped_as_documented(self):
        a = no_series_simulation(n_replicates=200, n_test=300, seed=5)
        b = no_series_simulation(n_replicates=200, n_test=300, seed=5)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), (len(MEMBERS) - 1) * len(TAU_GRID))
        self.assertEqual(set(a[a.pair_kind == "baseline"].member), {"gaussian_b"})
        self.assertTrue((a.rate == a.n_sign_differs / 200).all())
        self.assertFalse(a.equals(no_series_simulation(n_replicates=200, n_test=300, seed=6)))
        with self.assertRaises(ValueError):
            no_series_simulation(variant="empirical", n_replicates=10)

    def test_rates_follow_the_population_winners_without_any_series(self):
        s = no_series_simulation(n_replicates=400).set_index(["member", "tau"])
        # population reversal cells with a wide margin: the sign differs in most replicates
        self.assertGreater(s.loc[("skewnorm_n8", 0.99), "rate"], 0.8)
        self.assertGreater(s.loc[("skewnorm_p8", 0.01), "rate"], 0.8)
        # no reversal and a wide margin: it hardly ever differs
        self.assertLess(s.loc[("skewnorm_p8", 0.75), "rate"], 0.03)
        self.assertLess(s.loc[("gaussian_b", 0.5), "rate"], 0.03)
        # two Gaussian draws at an extreme tau: noise alone gives a clearly non-zero rate
        self.assertTrue(0.08 < s.loc[("gaussian_b", 0.01), "rate"] < 0.4)

    def test_comparison_with_matched_pairs_on_a_hand_built_example(self):
        ns = pd.DataFrame({"variant": "primary", "member": ["a", "b", "gaussian_b"] * 2,
                           "pair_kind": ["alternative", "alternative", "baseline"] * 2,
                           "tau": [0.05] * 3 + [0.5] * 3, "rate": [0.6, 0.2, 0.1, 0.0, 0.1, 0.0]})
        pairs = pd.DataFrame({"variant": "primary", "tau": [0.05] * 5,
                              "pair_kind": ["alternative"] * 4 + ["baseline"],
                              "status": ["matched", "matched", "matched", "failed", "matched"],
                              "sign_differs": [True, False, False, True, True]})
        t = no_series_vs_matched(ns, pairs).set_index("tau")
        r = t.loc[0.05]
        self.assertAlmostEqual(r.alternative_rate_no_series, 0.4)
        self.assertAlmostEqual(r.alternative_rate_matched_pairs, 1 / 3)
        self.assertEqual((r.alternative_n_matched_pairs, r.baseline_n_matched_pairs), (3, 1))
        self.assertEqual((r.baseline_rate_no_series, r.baseline_rate_matched_pairs), (0.1, 1.0))
        self.assertAlmostEqual(t.loc[0.5, "alternative_rate_no_series"], 0.05)
        self.assertEqual(len(t), len(TAU_GRID))


class MatchStatus(unittest.TestCase):
    def test_rates_for_matched_and_failed_pairs(self):
        p = pd.DataFrame({"variant": "primary", "pair_kind": "alternative", "tau": 0.05,
                          "status": ["matched"] * 4 + ["failed"] * 2,
                          "sign_differs": [True, True, True, False, True, False]})
        r = sign_by_match_status(p).iloc[0]
        self.assertEqual((r.n_matched, r.n_differs_matched, r.rate_matched), (4, 3, 0.75))
        self.assertEqual((r.n_failed_match, r.n_differs_failed_match, r.rate_failed_match), (2, 1, 0.5))
        self.assertAlmostEqual(r.fisher_p, stats.fisher_exact([[3, 1], [1, 1]])[1])
        only = sign_by_match_status(p[p.status == "matched"]).iloc[0]
        self.assertTrue(np.isnan(only.rate_failed_match) and np.isnan(only.fisher_p))


if __name__ == "__main__":
    unittest.main()
