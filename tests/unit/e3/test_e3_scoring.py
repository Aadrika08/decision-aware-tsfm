"""Engine scoring, the sign-difference counter and the population check."""
import unittest

import numpy as np
import pandas as pd
from scipy import integrate, stats

from decision_aware_tsfm.analysis.decision.fixture import expected_pinball as gaussian_closed_form
from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID
from decision_aware_tsfm.analysis.experiments.e3.design import (
    F_VARIANTS, MEMBERS, MODEL_G, N_CONTEXT, N_TOTAL, STRUCTURES, innovation_dist)
from decision_aware_tsfm.analysis.experiments.e3.forecasters import series_frame
from decision_aware_tsfm.analysis.experiments.e3.scoring import (
    arm_interval, count_sign_differences, expected_pinball, member_delta_summary,
    population_cell_status, sign_by_population_status, sign_difference_vs_baseline, sign_pairs,
    wilson_interval)

from . import _shared

F_PRIMARY = F_VARIANTS["primary"].model


class EngineCosts(unittest.TestCase):
    def test_cost_at_median_is_half_the_mean_absolute_error(self):
        u = _shared.units()
        rows = _shared.scored_rows().set_index("series_id")
        for sid in rows.index:
            with self.subTest(series=sid):
                r = rows.loc[sid]
                x = _shared.values_of(sid)
                fr = series_frame(x, sid, STRUCTURES[r.structure], MEMBERS[r.member], [MODEL_G, F_PRIMARY])
                y = x[N_CONTEXT:]
                got = u[(u.series_id == sid) & (u.tau == 0.5)].iloc[0]
                for col, model in (("J_G", MODEL_G), ("J_F", F_PRIMARY)):
                    q50 = fr.loc[fr.model == model, "q_0.50"].to_numpy()
                    half_mae = 0.5 * float(np.mean(np.abs(y - q50)))        # plain numpy, no engine
                    self.assertAlmostEqual(got[col], half_mae, delta=1e-12)

    def test_units_cover_every_tau_once_and_use_all_test_origins(self):
        u = _shared.units()
        self.assertEqual(len(u), len(_shared.SCORED_MEMBERS) * len(TAU_GRID))
        self.assertFalse(u.duplicated(["series_id", "tau"]).any())
        self.assertTrue((u.n_test_origins == N_TOTAL - N_CONTEXT).all())
        self.assertTrue((u.decision_origin == N_CONTEXT).all())

    def test_omega_seen_by_the_engine_is_the_context_window_omega(self):
        u = _shared.units()
        om = _shared.dataset()[1].set_index("series_id").omega_levels
        np.testing.assert_array_equal(u.omega.to_numpy(), om.loc[u.series_id].to_numpy())

    def test_delta_and_winner_are_consistent(self):
        u = _shared.units()
        np.testing.assert_allclose(u.delta, u.J_F - u.J_G, atol=0)
        # engine rule: ties within 1e-12 relative go to the "classical" slot, which is G
        tol = 1e-12 * np.maximum(1.0, np.maximum(u.J_F.abs(), u.J_G.abs()))
        self.assertTrue(((u.winner == "F") == (u.J_F < u.J_G - tol)).all())
        self.assertEqual(set(u.winner), {"F", "G"})
        np.testing.assert_allclose(u.delta_scaled, u.delta / u.scale)
        self.assertTrue((u.scale > 0).all())

    def test_delta_rel_is_relative_to_the_cost_of_G(self):
        u = _shared.units()
        np.testing.assert_allclose(u.delta_rel, (u.J_F - u.J_G) / u.J_G, rtol=1e-12)
        self.assertGreater(float(np.abs(u.delta_rel - (u.J_F - u.J_G) / u.J_F).max()), 1e-4)
        # a hand case through the same code path is not available without the engine, so
        # also pin the sign convention: delta_rel < 0 exactly where F is cheaper
        self.assertTrue(((u.delta_rel < 0) == (u.J_F < u.J_G)).all())


def _pairs(rows):
    cols = ["variant", "pair_kind", "tau", "structure", "seed", "member_b", "status",
            "winner_a", "winner_b"]
    p = pd.DataFrame(rows, columns=cols)
    p["sign_differs"] = p.winner_a != p.winner_b
    p["delta_scaled_a"] = 0.01
    p["delta_scaled_b"] = -0.02
    return p


class SignDifferenceCounter(unittest.TestCase):
    def test_hand_built_example(self):
        p = _pairs([
            # tau 0.05: three matched alternative pairs, two with opposite winners; one failed match
            ("primary", "alternative", 0.05, "s", 0, "m1", "matched", "G", "F"),
            ("primary", "alternative", 0.05, "s", 1, "m1", "matched", "G", "G"),
            ("primary", "alternative", 0.05, "s", 2, "m1", "matched", "F", "G"),
            ("primary", "alternative", 0.05, "s", 3, "m1", "failed", "G", "F"),
            # tau 0.5: two matched alternative pairs, none differ
            ("primary", "alternative", 0.50, "s", 0, "m1", "matched", "G", "G"),
            ("primary", "alternative", 0.50, "s", 1, "m1", "matched", "F", "F"),
            # baseline at tau 0.05: two matched, one differs
            ("primary", "baseline", 0.05, "s", 0, "b", "matched", "G", "G"),
            ("primary", "baseline", 0.05, "s", 1, "b", "matched", "G", "F"),
        ])
        c = count_sign_differences(p, n_boot=50).set_index(["pair_kind", "tau"])
        a = c.loc[("alternative", 0.05)]
        self.assertEqual((a.n_pairs, a.n_failed_match, a.n_matched, a.n_sign_differs), (4, 1, 3, 2))
        self.assertAlmostEqual(a.rate, 2 / 3)
        b = c.loc[("alternative", 0.50)]
        self.assertEqual((b.n_pairs, b.n_failed_match, b.n_matched, b.n_sign_differs), (2, 0, 2, 0))
        self.assertEqual(b.rate, 0.0)
        base = c.loc[("baseline", 0.05)]
        self.assertEqual((base.n_matched, base.n_sign_differs, base.rate), (2, 1, 0.5))
        self.assertEqual(len(c), 3)

    def test_failed_matches_never_enter_the_sign_count(self):
        p = _pairs([("primary", "alternative", 0.5, "s", k, "m1", "failed", "G", "F") for k in range(4)])
        c = count_sign_differences(p, n_boot=10).iloc[0]
        self.assertEqual((c.n_pairs, c.n_failed_match, c.n_matched, c.n_sign_differs), (4, 4, 0, 0))
        self.assertTrue(np.isnan(c.rate))

    def test_sign_pairs_joins_reference_and_member_winners(self):
        units = pd.DataFrame({
            "variant": "primary", "tau": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "series_id": ["ref", "alt", "base"] * 2,
            "winner": ["G", "F", "G", "G", "G", "F"],
            "delta": [1.0, -1.0, 1.0, 2.0, 2.0, -2.0],
            "delta_scaled": [0.1, -0.1, 0.1, 0.2, 0.2, -0.2],
            "delta_rel": [0.1, -0.1, 0.1, 0.2, 0.2, -0.2]})
        matches = pd.DataFrame({
            "pair_kind": ["alternative", "baseline", "negative_control"], "seed": 0,
            "structure_a": "s", "member_b": ["m1", "gaussian_b", "gaussian"],
            "series_id_a": "ref", "series_id_b": ["alt", "base", "other"],
            "omega_levels_a": 0.3, "omega_levels_b": [0.31, 0.29, 0.6],
            "omega_levels_abs_diff": [0.01, 0.01, 0.3], "status": ["matched", "matched", "failed"]})
        p = sign_pairs(units, matches).set_index(["pair_kind", "tau"])
        self.assertEqual(len(p), 4)                                    # negative control not paired
        self.assertTrue(p.loc[("alternative", 0.05)].sign_differs)
        self.assertFalse(p.loc[("alternative", 0.5)].sign_differs)
        self.assertFalse(p.loc[("baseline", 0.05)].sign_differs)
        self.assertTrue(p.loc[("baseline", 0.5)].sign_differs)
        self.assertEqual(p.loc[("alternative", 0.05)].delta_b, -1.0)

    def test_sign_differs_follows_the_engine_winner_on_exact_ties(self):
        # Delta exactly 0 is a tie, and the engine gives ties to G. 49 of the 42,240 units of
        # the full run are such ties, so the pairing must compare WINNERS, not signs of Delta.
        units = pd.DataFrame({
            "variant": "primary", "tau": 0.5, "series_id": ["ref", "g_side", "f_side"],
            "winner": ["G", "G", "F"], "delta": [0.0, 0.3, -0.3],
            "delta_scaled": [0.0, 0.1, -0.1], "delta_rel": [0.0, 0.1, -0.1]})
        matches = pd.DataFrame({
            "pair_kind": "alternative", "seed": 0, "structure_a": "s", "member_b": ["m1", "m2"],
            "series_id_a": "ref", "series_id_b": ["g_side", "f_side"],
            "omega_levels_a": 0.3, "omega_levels_b": 0.3, "omega_levels_abs_diff": 0.0,
            "status": "matched"})
        p = sign_pairs(units, matches).set_index("member_b")
        self.assertFalse(p.loc["m1", "sign_differs"])        # tie (G) against G
        self.assertTrue(p.loc["m2", "sign_differs"])         # tie (G) against F
        # and with the tie on the member side
        swapped = matches.assign(series_id_a=["g_side", "f_side"], series_id_b="ref")
        q = sign_pairs(units, swapped).set_index("member_b")
        self.assertFalse(q.loc["m1", "sign_differs"])
        self.assertTrue(q.loc["m2", "sign_differs"])

    def test_bootstrap_resamples_whole_groups_not_single_pairs(self):
        # 10 groups of 6 pairs; inside a group every pair has the same outcome (the shared
        # reference decides it). 5 groups differ. Resampling groups gives rates on a grid of
        # tenths with standard deviation 0.158; resampling the 60 pairs would give 0.065.
        rows = [("primary", "alternative", 0.5, "s", g, f"m{j}", "matched", "G", "F" if g < 5 else "G")
                for g in range(10) for j in range(6)]
        c = count_sign_differences(_pairs(rows), n_boot=2000).iloc[0]
        self.assertEqual((c.n_matched, c.n_sign_differs, c.n_groups), (60, 30, 10))
        self.assertEqual(c.interval_method, "group_bootstrap")
        self.assertLessEqual(c.boot_lo, 0.2 + 1e-12)
        self.assertGreaterEqual(c.boot_hi, 0.8 - 1e-12)
        self.assertAlmostEqual(c.boot_lo * 10, round(c.boot_lo * 10), places=9)     # on the grid of tenths
        self.assertGreater(c.boot_hi - c.boot_lo, 2 * (c.wilson_hi - c.wilson_lo))

    def test_failed_matches_change_neither_the_rate_nor_any_interval(self):
        matched = [("primary", "alternative", 0.5, "s", g, f"m{j}", "matched", "G", "F" if g < 3 else "G")
                   for g in range(8) for j in range(2)]
        failed = [("primary", "alternative", 0.5, "s", g, "m9", "failed", "G", "F") for g in range(8)]
        with_failed = count_sign_differences(_pairs(matched + failed), n_boot=500).iloc[0]
        without = count_sign_differences(_pairs(matched), n_boot=500).iloc[0]
        self.assertEqual((with_failed.n_pairs, with_failed.n_failed_match), (24, 8))
        for col in ("n_matched", "n_sign_differs", "rate", "wilson_lo", "wilson_hi", "boot_lo",
                    "boot_hi", "interval_lo", "interval_hi", "n_groups"):
            self.assertEqual(with_failed[col], without[col], col)
        self.assertLess(with_failed.boot_hi, 1.0)

    def test_all_equal_outcomes_get_a_wilson_interval_not_a_zero_width_one(self):
        rows = [("primary", "alternative", 0.75, "s", g, f"m{j}", "matched", "G", "G")
                for g in range(20) for j in range(5)]
        c = count_sign_differences(_pairs(rows), n_boot=200).iloc[0]
        self.assertEqual((c.n_matched, c.n_sign_differs, c.rate, c.n_groups), (100, 0, 0.0, 20))
        self.assertEqual(c.interval_method, "wilson_on_groups")
        self.assertTrue(np.isnan(c.boot_lo) and np.isnan(c.boot_hi))
        lo, hi = wilson_interval(0, 20)                       # groups, not pairs
        self.assertEqual((c.interval_lo, c.interval_hi), (lo, hi))
        self.assertGreater(c.interval_hi, 0.1)
        self.assertEqual(arm_interval(7, 7, 4, 1.0, 1.0), (*wilson_interval(4, 4), "wilson_on_groups"))
        self.assertEqual(arm_interval(3, 7, 4, 0.2, 0.6), (0.2, 0.6, "group_bootstrap"))

    def test_wilson_interval_matches_scipy(self):
        for k, n in ((0, 7), (5, 10), (22, 37), (30, 30)):
            with self.subTest(k=k, n=n):
                ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
                lo, hi = wilson_interval(k, n)
                self.assertAlmostEqual(lo, ci.low, places=10)
                self.assertAlmostEqual(hi, ci.high, places=10)

    def test_member_summary_share(self):
        u = pd.DataFrame({"variant": "primary", "member": "m", "tau": 0.5,
                          "series_id": list("abcd"), "winner": ["F", "F", "G", "F"],
                          "delta_scaled": [-1.0, -1.0, 1.0, -1.0], "delta_rel": [-2.0, -2.0, 2.0, -2.0]})
        s = member_delta_summary(u).iloc[0]
        self.assertEqual((s.n_series, s.share_F_wins, s.delta_scaled_mean, s.delta_rel_mean),
                         (4, 0.75, -0.5, -1.0))


def _pair(kind, tau, seed, member, status, wa, wb, structure="s"):
    return ("primary", kind, tau, structure, seed, member, status, wa, wb)


class DifferenceFromBaseline(unittest.TestCase):
    def test_counts_rates_and_criterion_on_a_hand_built_example(self):
        rows = []
        for g in range(12):
            # alternatives: 2 pairs per group, both differ in groups 0..8  -> 18 of 24
            rows += [_pair("alternative", 0.05, g, f"m{j}", "matched", "G", "F" if g < 9 else "G")
                     for j in range(2)]
            # baseline: 1 pair per group, differs in groups 0..2          -> 3 of 12
            rows.append(_pair("baseline", 0.05, g, "gaussian_b", "matched", "G", "F" if g < 3 else "G"))
        rows.append(_pair("alternative", 0.05, 0, "m7", "failed", "G", "G"))      # must be ignored
        d = sign_difference_vs_baseline(_pairs(rows), n_boot=2000).iloc[0]
        self.assertEqual((d.alt_n_matched, d.alt_n_sign_differs, d.base_n_matched, d.base_n_sign_differs),
                         (24, 18, 12, 3))
        self.assertAlmostEqual(d["diff"], 0.75 - 0.25)
        self.assertEqual(d.diff_method, "joint_group_bootstrap")
        self.assertTrue(0 < d.diff_lo < 0.5 < d.diff_hi < 1)
        self.assertTrue(d.above_baseline)
        self.assertEqual(d.n_groups, 12)

    def test_the_two_rates_come_from_the_same_resample(self):
        # In every group the alternative pair and the baseline pair have the same outcome, so
        # the difference is 0 in EVERY joint resample. Resampling the two arms separately
        # would give a wide interval around 0 instead.
        rows = []
        for g in range(20):
            w = "F" if g < 10 else "G"
            rows += [_pair("alternative", 0.5, g, "m1", "matched", "G", w),
                     _pair("baseline", 0.5, g, "gaussian_b", "matched", "G", w)]
        d = sign_difference_vs_baseline(_pairs(rows), n_boot=500).iloc[0]
        self.assertEqual(d.diff_method, "joint_group_bootstrap")
        self.assertEqual((d["diff"], d.diff_lo, d.diff_hi), (0.0, 0.0, 0.0))
        self.assertFalse(d.above_baseline)
        self.assertLess(d.alt_lo, 0.35)
        self.assertGreater(d.alt_hi, 0.65)

    def test_zero_counts_use_the_wilson_fallback(self):
        rows = []
        for g in range(30):
            rows += [_pair("alternative", 0.75, g, f"m{j}", "matched", "G", "G") for j in range(3)]
            rows.append(_pair("baseline", 0.75, g, "gaussian_b", "matched", "G", "G"))
        d = sign_difference_vs_baseline(_pairs(rows), n_boot=200).iloc[0]
        lo, hi = wilson_interval(0, 30)
        self.assertEqual((d.alt_method, d.base_method), ("wilson_on_groups", "wilson_on_groups"))
        self.assertEqual(d.diff_method, "newcombe_from_arm_intervals")
        self.assertAlmostEqual(d.diff_lo, -hi)
        self.assertAlmostEqual(d.diff_hi, hi)
        self.assertGreater(d.diff_hi, 0.05)                      # not "0.0 to 0.0"
        self.assertFalse(d.above_baseline)

    def test_one_empty_arm_widens_the_difference_interval(self):
        # baseline 0 of 30: its uncertainty (Wilson upper end) must enter the lower end of the difference
        rows = []
        for g in range(30):
            rows += [_pair("alternative", 0.25, g, f"m{j}", "matched", "G", "F" if g < 3 else "G")
                     for j in range(2)]
            rows.append(_pair("baseline", 0.25, g, "gaussian_b", "matched", "G", "G"))
        d = sign_difference_vs_baseline(_pairs(rows), n_boot=1000).iloc[0]
        self.assertEqual((d.alt_method, d.base_method), ("group_bootstrap", "wilson_on_groups"))
        self.assertEqual(d.diff_method, "newcombe_from_arm_intervals")
        self.assertAlmostEqual(d["diff"], 0.1)
        base_hi = wilson_interval(0, 30)[1]
        self.assertAlmostEqual(d.diff_lo, 0.1 - np.sqrt((0.1 - d.alt_lo) ** 2 + base_hi ** 2))
        self.assertLess(d.diff_lo, d.alt_lo)
        self.assertEqual(d.above_baseline, d.diff_lo > 0)


class PopulationStatus(unittest.TestCase):
    def _expected(self):
        # tau 0.05: m1 reverses (F wins on it, G on the reference); m2 does not
        return pd.DataFrame({
            "variant": "primary", "tau": 0.05,
            "member": ["gaussian", "gaussian_b", "m1", "m2"],
            "population_winner": ["G", "G", "F", "G"],
            "exp_delta_rel": [0.04, 0.04, -0.10, 0.002]})

    def test_cell_status_compares_each_member_with_the_reference(self):
        c = population_cell_status(self._expected()).set_index("member")
        self.assertEqual(c.cell_status.to_dict(), {"gaussian": "same_winner", "gaussian_b": "same_winner",
                                                   "m1": "reversal", "m2": "same_winner"})
        self.assertTrue((c.pop_delta_rel_reference == 0.04).all())
        self.assertEqual(c.loc["m1", "pop_delta_rel_member"], -0.10)

    def test_sign_differences_are_split_by_status_with_effect_sizes(self):
        rows = ([_pair("alternative", 0.05, g, "m1", "matched", "G", "F") for g in range(4)]
                + [_pair("alternative", 0.05, g, "m2", "matched", "G", "F" if g == 0 else "G")
                   for g in range(4)]
                + [_pair("alternative", 0.05, 9, "m2", "failed", "G", "F")]
                + [_pair("baseline", 0.05, g, "gaussian_b", "matched", "G", "G") for g in range(4)])
        p = _pairs(rows)
        p["delta_rel_a"] = 0.05
        p["delta_rel_b"] = np.where(p.member_b == "m1", -0.2, 0.01)
        t = sign_by_population_status(p, self._expected()).set_index(["pair_kind", "cell_status"])
        rev, same = t.loc[("alternative", "reversal")], t.loc[("alternative", "same_winner")]
        self.assertEqual((rev.n_cells, rev.n_matched, rev.n_sign_differs, rev.rate), (1, 4, 4, 1.0))
        self.assertEqual((same.n_cells, same.n_matched, same.n_sign_differs, same.rate), (1, 4, 1, 0.25))
        self.assertEqual((rev.pop_delta_rel_member_min, rev.pop_delta_rel_member_max), (-0.10, -0.10))
        self.assertEqual(same.pop_delta_rel_member_max, 0.002)
        self.assertAlmostEqual(rev.realised_delta_rel_member_mean, -0.2)
        self.assertAlmostEqual(same.realised_delta_rel_reference_mean, 0.05)
        self.assertEqual(t.loc[("baseline", "same_winner")].n_sign_differs, 0)
        self.assertNotIn(("baseline", "reversal"), t.index)
        # every matched difference is counted exactly once
        self.assertEqual(int(t.n_sign_differs.sum()), int(p[p.status == "matched"].sign_differs.sum()))

    def test_real_table_has_no_reversal_on_gaussian_members_and_18_alternative_cells(self):
        c = population_cell_status(_shared.expected())
        prim = c[c.variant == "primary"]
        gauss = prim[prim.member.isin(["gaussian", "gaussian_b"])]
        self.assertTrue((gauss.cell_status == "same_winner").all())
        self.assertEqual(int((prim.cell_status == "reversal").sum()), 18)
        self.assertTrue(prim[prim.cell_status == "reversal"].tau.isin([0.01, 0.05, 0.09, 0.91, 0.95, 0.99]).all())


def _pdf_route(dist, c, tau):
    """Expected pinball by integrating the density on each side of c: a different
    route from scoring.expected_pinball, which integrates the distribution function."""
    lo, _ = integrate.quad(lambda x: (c - x) * dist.pdf(x), -np.inf, c, limit=300)
    hi, _ = integrate.quad(lambda x: (x - c) * dist.pdf(x), c, np.inf, limit=300)
    return (1 - tau) * lo + tau * hi


class PopulationCheck(unittest.TestCase):
    def test_table_matches_an_independent_integration(self):
        # a subset of cells here (this route is slow); every cell is checked against the
        # finite-interval identity in test_e3_population
        e = _shared.expected()
        z = stats.norm.ppf(np.asarray(TAU_GRID))
        for mname in ("gaussian", "skewnorm_p8", "skewnorm_n2", "student_t3"):
            dist = innovation_dist(MEMBERS[mname])
            q = dist.ppf(np.asarray(TAU_GRID))
            for k, tau in enumerate(TAU_GRID):
                if tau not in (0.01, 0.09, 0.5, 0.95):
                    continue
                rows = e[(e.member == mname) & (e.tau == tau)].set_index("variant")
                with self.subTest(member=mname, tau=tau):
                    jg = _pdf_route(dist, z[k], tau)
                    self.assertAlmostEqual(rows.exp_J_G.iloc[0], jg, places=7)
                    for vname in ("loc_small", "primary", "loc_large"):
                        jf = _pdf_route(dist, q[k] + F_VARIANTS[vname].location_error, tau)
                        r = rows.loc[vname]
                        self.assertAlmostEqual(r.exp_J_F, jf, places=7)
                        self.assertAlmostEqual(r.exp_delta, jf - jg, places=7)
                        self.assertAlmostEqual(r.exp_delta_rel, (jf - jg) / jg, places=6)
                        self.assertEqual(r.population_winner, "F" if jf < jg else "G")
                        self.assertEqual((r.gaussian_quantile, r.true_quantile), (z[k], q[k]))

    def test_large_iid_sample_costs_agree_with_the_table(self):
        # 400,000 draws per member, plain numpy pinball: within 5 standard errors
        e = _shared.expected().set_index(["variant", "member", "tau"])
        taus = np.asarray(TAU_GRID)
        for mname in ("skewnorm_p8", "student_t5"):
            dist = innovation_dist(MEMBERS[mname])
            x = dist.rvs(size=400_000, random_state=np.random.default_rng(17))
            for vname in ("loc_small", "primary", "loc_large"):
                c = dist.ppf(taus) + F_VARIANTS[vname].location_error
                u = x[:, None] - c[None, :]
                loss = np.maximum(taus * u, (taus - 1) * u)
                mean, se = loss.mean(axis=0), loss.std(axis=0, ddof=1) / np.sqrt(x.size)
                for k, tau in enumerate(TAU_GRID):
                    with self.subTest(member=mname, variant=vname, tau=tau):
                        self.assertLess(abs(mean[k] - e.loc[(vname, mname, tau), "exp_J_F"]), 5 * se[k])
                        self.assertLess(5 * se[k] / mean[k], 0.05)       # the check is a tight one

    def test_expected_pinball_matches_the_gaussian_closed_form(self):
        d = stats.norm()
        for tau in (0.01, 0.25, 0.5, 0.95):
            for c in (-1.0, 0.0, 0.25, 2.0):
                with self.subTest(tau=tau, c=c):
                    self.assertAlmostEqual(expected_pinball(d, c, tau),
                                           float(gaussian_closed_form(c, 0.0, tau)), places=8)

    def test_true_quantile_minimises_expected_pinball(self):
        for mname in ("skewnorm_p8", "student_t3"):
            d = innovation_dist(MEMBERS[mname])
            for tau in (0.05, 0.5, 0.91):
                with self.subTest(member=mname, tau=tau):
                    q = float(d.ppf(tau))
                    best = expected_pinball(d, q, tau)
                    self.assertLess(best, expected_pinball(d, q + 0.05, tau))
                    self.assertLess(best, expected_pinball(d, q - 0.05, tau))

    def test_on_gaussian_members_G_wins_in_population_at_every_tau(self):
        e = _shared.expected()
        g = e[e.member.isin(["gaussian", "gaussian_b"])]
        self.assertEqual(len(g), 2 * 3 * len(TAU_GRID))
        self.assertTrue((g.population_winner == "G").all())
        self.assertTrue((g.exp_delta > 0).all())
        np.testing.assert_allclose(g.gaussian_quantile, g.true_quantile, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
