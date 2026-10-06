"""Generator, innovation laws and the Omega match check."""
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from decision_aware_tsfm.analysis.experiments.e3.design import (
    BASELINE, BURN_IN, INNOVATION_VARIANCE_ATOL, MEMBERS, N_CONTEXT, N_TOTAL, OMEGA_TOLERANCE,
    REFERENCE, STRUCTURES, Member, innovation_dist)
from decision_aware_tsfm.analysis.experiments.e3.generator import (
    NEGATIVE_CONTROLS, check_pair, context_innovations, describe_series, group_table,
    make_dataset, make_series, match_table, member_rng, tail_summary)
from decision_aware_tsfm.omega.compute import compute_omega

from . import _shared

# Population excess kurtosis of each member's innovation law (scipy, analytic).
# The sample variance of n draws has standard deviation sqrt((kurtosis + 2) / n),
# which is finite for every member except student_t3.
FINITE_KURTOSIS_MEMBERS = tuple(n for n in MEMBERS if n != "student_t3")


class Determinism(unittest.TestCase):
    def test_same_seed_gives_identical_series(self):
        for sname in ("ar1_phi0.80", "sine_sd1.0"):
            for mname in ("gaussian", "skewnorm_n8", "student_t3"):
                with self.subTest(structure=sname, member=mname):
                    a = make_series(STRUCTURES[sname], MEMBERS[mname], 5)
                    b = make_series(STRUCTURES[sname], MEMBERS[mname], 5)
                    np.testing.assert_array_equal(a, b)
                    self.assertEqual(a.size, N_TOTAL)

    def test_seed_structure_and_member_each_change_the_stream(self):
        st, mb = STRUCTURES["ar1_phi0.80"], MEMBERS["gaussian"]
        base = make_series(st, mb, 5)
        self.assertFalse(np.array_equal(base, make_series(st, mb, 6)))
        self.assertFalse(np.array_equal(base, make_series(st, MEMBERS["gaussian_b"], 5)))
        # structure: compare the DRAWS, not the series. Two AR(1) structures with the same
        # sigma give different series whatever the stream is, so recover the innovations
        # (x_t - phi * x_{t-1}) and require those to differ.
        other = STRUCTURES["ar1_phi0.50"]
        e_a = context_innovations(base, st)
        e_b = context_innovations(make_series(other, mb, 5), other)
        self.assertFalse(np.allclose(e_a, e_b))
        self.assertNotEqual(member_rng(5, st, mb).random(), member_rng(5, other, mb).random())

    def test_dataset_tables_are_reproducible(self):
        v1, s1 = make_dataset((3,), structures=["ar1_phi0.50"])
        v2, s2 = make_dataset((3,), structures=["ar1_phi0.50"])
        pd.testing.assert_frame_equal(v1, v2)
        pd.testing.assert_frame_equal(s1, s2)


class InnovationLaws(unittest.TestCase):
    def test_population_mean_zero_and_variance_one_for_every_member(self):
        for name, mb in MEMBERS.items():
            with self.subTest(member=name):
                d = innovation_dist(mb)
                self.assertAlmostEqual(float(d.mean()), 0.0, delta=INNOVATION_VARIANCE_ATOL)
                self.assertAlmostEqual(float(d.var()), 1.0, delta=INNOVATION_VARIANCE_ATOL)

    def test_variance_by_independent_numerical_integration(self):
        # second moment integrated from the density, not taken from scipy's formula
        for name, mb in MEMBERS.items():
            with self.subTest(member=name):
                d = innovation_dist(mb)
                self.assertAlmostEqual(d.expect(lambda x: x), 0.0, delta=1e-6)
                self.assertAlmostEqual(d.expect(lambda x: x * x), 1.0, delta=1e-6)

    def test_student_t_without_a_variance_is_refused(self):
        for df in (2.0, 1.5, 1.0):
            with self.subTest(df=df):
                with self.assertRaises(ValueError):
                    innovation_dist(Member("bad", "student_t", 99, df))

    def test_sample_innovation_variance_is_within_its_sampling_noise(self):
        # members with finite kurtosis: within 5 standard deviations of 1, per series
        s = _shared.dataset()[1]
        for r in s[s.member.isin(FINITE_KURTOSIS_MEMBERS)].itertuples():
            with self.subTest(series=r.series_id):
                kurt = float(innovation_dist(MEMBERS[r.member]).stats(moments="k"))
                sd = np.sqrt((kurt + 2.0) / N_CONTEXT)
                rel = r.innov_variance / STRUCTURES[r.structure].sigma ** 2
                self.assertLess(abs(rel - 1.0), 5 * sd)

    def test_student_t3_sample_variance_does_not_settle_at_this_length(self):
        # The population variance of student_t3 is exactly 1 (tested above), but its fourth
        # moment is infinite, so the realised variance of 2000 draws is far more variable
        # than a Gaussian's. Equal population variance is NOT equal realised variance here.
        d = _shared.omega_draws().pivot(index="seed", columns="member", values="innov_variance")
        iqr = d.quantile(0.75) - d.quantile(0.25)
        self.assertGreater(iqr["student_t3"], 2.5 * iqr["gaussian"])
        self.assertGreater(float((d["student_t3"] - 1).abs().max()), 0.5)
        self.assertLess(float((d["gaussian"] - 1).abs().max()), 0.2)
        self.assertTrue(np.isinf(float(innovation_dist(MEMBERS["student_t3"]).stats(moments="k"))))

    def test_population_shape_differs_in_the_intended_direction(self):
        sk = {n: float(innovation_dist(m).stats(moments="s")) for n, m in MEMBERS.items()}
        self.assertEqual(sk["gaussian"], 0.0)
        self.assertGreater(sk["skewnorm_p8"], sk["skewnorm_p2"])
        self.assertGreater(sk["skewnorm_p2"], 0.3)
        self.assertAlmostEqual(sk["skewnorm_n8"], -sk["skewnorm_p8"], places=12)
        self.assertAlmostEqual(sk["skewnorm_n2"], -sk["skewnorm_p2"], places=12)
        self.assertEqual(sk["student_t5"], 0.0)
        self.assertAlmostEqual(float(innovation_dist(MEMBERS["student_t5"]).stats(moments="k")), 6.0)


class SeriesConstruction(unittest.TestCase):
    def test_ar1_recursion_recovers_the_drawn_innovations(self):
        for sname in ("ar1_phi0.50", "ar1_phi0.95"):
            for mname in ("gaussian", "skewnorm_p8"):
                with self.subTest(structure=sname, member=mname):
                    st, mb = STRUCTURES[sname], MEMBERS[mname]
                    x = make_series(st, mb, 2)
                    e = innovation_dist(mb).rvs(size=BURN_IN + N_TOTAL,
                                                random_state=member_rng(2, st, mb))[BURN_IN:]
                    np.testing.assert_allclose(x[1:] - st.phi * x[:-1], st.sigma * e[1:], atol=1e-9)

    def test_ar1_series_includes_the_burn_in_history(self):
        # the whole recursion written out here, starting from zero BEFORE the burn-in draws
        for sname in ("ar1_phi0.50", "ar1_phi0.95"):
            with self.subTest(structure=sname):
                st, mb = STRUCTURES[sname], MEMBERS["skewnorm_n2"]
                e = st.sigma * innovation_dist(mb).rvs(size=BURN_IN + N_TOTAL,
                                                       random_state=member_rng(3, st, mb))
                own, state = np.empty(e.size), 0.0
                for t in range(e.size):
                    state = st.phi * state + e[t]
                    own[t] = state
                x = make_series(st, mb, 3)
                np.testing.assert_allclose(x, own[BURN_IN:], atol=1e-9)
                self.assertGreater(abs(x[0] - e[BURN_IN]), 1e-3)      # position 0 carries history

    def test_ar1_starts_in_the_stationary_distribution(self):
        # variance of the FIRST stored value across seeds is sigma^2 / (1 - phi^2) = 10.26,
        # not sigma^2 = 1 as it would be for a series started from a zero state
        st, mb = STRUCTURES["ar1_phi0.95"], MEMBERS["gaussian"]
        first = np.array([make_series(st, mb, seed, n_total=5)[0] for seed in range(100, 500)])
        self.assertAlmostEqual(float(first.var()), 1.0 / (1.0 - st.phi ** 2), delta=3.0)

    def test_context_innovations_use_the_context_window_only(self):
        st = STRUCTURES["ar1_phi0.80"]
        x = make_series(st, MEMBERS["student_t5"], 1)
        y = x.copy()
        y[N_CONTEXT:] += 100.0
        np.testing.assert_array_equal(context_innovations(x, st), context_innovations(y, st))
        self.assertEqual(context_innovations(x, st).size, N_CONTEXT - 1)
        self.assertEqual(context_innovations(x, STRUCTURES["sine_sd1.0"]).size, N_CONTEXT)

    def test_persisted_omega_is_the_team_function_on_the_context_window(self):
        s = _shared.dataset()[1]
        for sid in s.series_id.iloc[[0, 9, 20, 40, 65]]:
            with self.subTest(series=sid):
                x, row = _shared.values_of(sid), s.set_index("series_id").loc[sid]
                self.assertEqual(row.omega_levels, compute_omega(x[:N_CONTEXT], diffs=False))
                self.assertEqual(row.omega_diffs, compute_omega(x[:N_CONTEXT], diffs=True))
                self.assertNotEqual(row.omega_levels, compute_omega(x, diffs=False))

    def test_no_persisted_statistic_uses_the_test_period(self):
        for sname, mname in (("ar1_phi0.80", "skewnorm_p8"), ("sine_sd1.0", "student_t5")):
            with self.subTest(structure=sname, member=mname):
                st, mb = STRUCTURES[sname], MEMBERS[mname]
                x = make_series(st, mb, 6)
                y = x.copy()
                y[N_CONTEXT:] = 50.0 + 7.0 * np.arange(N_TOTAL - N_CONTEXT) ** 0.5
                a, b = describe_series(x, st, mb, 6), describe_series(y, st, mb, 6)
                self.assertEqual(a, b)
                ctx = x[:N_CONTEXT]
                self.assertEqual(a["skewness"], float(stats.skew(ctx)))
                self.assertEqual(a["excess_kurtosis"], float(stats.kurtosis(ctx)))
                self.assertNotEqual(a["skewness"], float(stats.skew(x)))

    def test_every_series_records_its_settings(self):
        s = _shared.dataset()[1]
        need = {"omega_levels", "omega_diffs", "skewness", "excess_kurtosis", "tail_weight",
                "tail_asym", "seed", "structure", "phi", "sigma", "member", "innovation_kind",
                "innovation_param", "n_total", "n_context", "n_test", "burn_in",
                "rng_seed_sequence", "innov_variance", "innov_skewness", "innov_excess_kurtosis"}
        self.assertFalse(need - set(s.columns))
        self.assertFalse(s.series_id.duplicated().any())
        self.assertTrue((s.n_total == N_TOTAL).all() and (s.n_context == N_CONTEXT).all())

    def test_sample_skewness_kurtosis_and_tails_differ_as_intended(self):
        s = _shared.dataset()[1]
        for (sname, seed), g in s[s.structure != "sine_sd2.0"].groupby(["structure", "seed"]):
            with self.subTest(structure=sname, seed=seed):
                r = g.set_index("member")
                self.assertGreater(r.innov_skewness["skewnorm_p8"], 0.6)
                self.assertGreater(r.innov_skewness["skewnorm_p8"], r.innov_skewness["skewnorm_p2"])
                self.assertGreater(r.innov_skewness["skewnorm_p2"], 0.2)
                self.assertLess(r.innov_skewness["skewnorm_n8"], -0.6)
                self.assertLess(r.innov_skewness["skewnorm_n2"], -0.2)
                self.assertLess(abs(r.innov_skewness["gaussian"]), 0.2)
                self.assertGreater(r.innov_excess_kurtosis["student_t5"],
                                   r.innov_excess_kurtosis["gaussian"] + 1.0)
                self.assertGreater(r.innov_tail_weight["student_t3"], r.innov_tail_weight["student_t5"])
                self.assertGreater(r.innov_tail_weight["student_t5"], r.innov_tail_weight["gaussian"])
                self.assertGreater(r.innov_tail_asym["skewnorm_p8"], 0.1)
                self.assertLess(r.innov_tail_asym["skewnorm_n8"], -0.1)

    def test_tail_summary_on_known_inputs(self):
        z = stats.norm.ppf(np.linspace(0.0005, 0.9995, 1000))       # a Gaussian-shaped grid
        t = tail_summary(z)
        self.assertAlmostEqual(t["tail_weight"], (2 * 2.3263) / (2 * 0.6745), places=1)
        self.assertAlmostEqual(t["tail_asym"], 0.0, places=6)
        self.assertGreater(tail_summary(np.exp(z))["tail_asym"], 0.5)   # long right tail


class MatchCheck(unittest.TestCase):
    def test_check_pair_boundary_and_length_rule(self):
        self.assertEqual(check_pair(0.300, 0.329, 2000, 2000)[0], "matched")
        self.assertEqual(check_pair(0.300, 0.331, 2000, 2000)[0], "failed")
        self.assertEqual(check_pair(0.331, 0.300, 2000, 2000)[0], "failed")
        status, reason = check_pair(0.300, 0.300, 2000, 1000)
        self.assertEqual(status, "failed")
        self.assertIn("unequal length", reason)
        self.assertEqual(check_pair(float("nan"), 0.3, 2000, 2000)[0], "failed")

    def test_status_is_the_tolerance_rule_applied_to_the_persisted_omegas(self):
        # No claim that design pairs match: in the Omega-only simulation two Gaussian draws of
        # the sine structure fail this tolerance about 14 percent of the time, student_t3
        # about 40 percent. The check is that the status is exactly the stated rule.
        s, m = _shared.dataset()[1].set_index("series_id"), _shared.matches()
        d = m[m.pair_kind != "negative_control"]
        self.assertEqual(len(d), 4 * 2 * 7)
        gap = (s.omega_levels.loc[d.series_id_a].to_numpy() - s.omega_levels.loc[d.series_id_b].to_numpy())
        np.testing.assert_array_equal(d.omega_levels_abs_diff.to_numpy(), np.abs(gap))
        np.testing.assert_array_equal((d.status == "matched").to_numpy(), np.abs(gap) <= OMEGA_TOLERANCE)
        self.assertTrue((d.expected_match).all())
        np.testing.assert_array_equal(d.as_expected.to_numpy(), (d.status == "matched").to_numpy())
        # at phi = 0.50 the tolerance is more than 3 standard deviations of the difference
        self.assertTrue((d[d.structure_a == "ar1_phi0.50"].status == "matched").all())

    def test_pair_kind_separates_the_baseline_from_the_alternatives(self):
        m = _shared.matches()
        d = m[m.pair_kind != "negative_control"]
        self.assertTrue((d.member_a == REFERENCE).all())
        np.testing.assert_array_equal((d.pair_kind == "baseline").to_numpy(),
                                      (d.member_b == BASELINE).to_numpy())
        self.assertEqual(d.pair_kind.value_counts().to_dict(),
                         {"alternative": 4 * 2 * 6, "baseline": 4 * 2 * 1})
        self.assertEqual(set(d[d.pair_kind == "alternative"].member_b),
                         {n for n, mb in MEMBERS.items() if mb.role == "alternative"})

    def test_negative_controls_are_reported_as_mismatches(self):
        m = _shared.matches()
        nc = m[m.pair_kind == "negative_control"]
        self.assertEqual(len(nc), len(NEGATIVE_CONTROLS) * len(_shared.SEEDS))
        self.assertTrue((nc.status == "failed").all())
        self.assertTrue((~nc.expected_match).all())
        self.assertTrue(nc.as_expected.all())
        self.assertTrue((nc.omega_levels_abs_diff > 2 * OMEGA_TOLERANCE).all())

    def test_every_generated_pair_is_kept_whatever_its_status(self):
        s, m = _shared.dataset()[1], _shared.matches()
        n_scored = sum(st.scored for st in STRUCTURES.values())
        expected = n_scored * len(_shared.SEEDS) * (len(MEMBERS) - 1)
        self.assertEqual(int((m.pair_kind != "negative_control").sum()), expected)
        self.assertEqual(set(m.status), {"matched", "failed"})
        g = group_table(s, m)
        self.assertEqual(len(g), n_scored * len(_shared.SEEDS))
        self.assertEqual(int(g.n_pairs_failed.sum()),
                         int(((m.pair_kind != "negative_control") & (m.status == "failed")).sum()))

    def test_deliberately_mismatched_group_is_reported_failed_not_dropped(self):
        st, bad_st = STRUCTURES["ar1_phi0.50"], STRUCTURES["ar1_phi0.95"]
        rows = [describe_series(make_series(st, MEMBERS[n], 0), st, MEMBERS[n], 0)
                for n in ("gaussian", "gaussian_b")]
        # a member generated with the wrong phi but filed in the phi = 0.50 group
        wrong = describe_series(make_series(bad_st, MEMBERS["skewnorm_p8"], 0), bad_st,
                                MEMBERS["skewnorm_p8"], 0)
        wrong["structure"] = st.name
        table = pd.DataFrame(rows + [wrong])
        m = match_table(table)
        self.assertEqual(len(m), 2)
        bad = m[m.member_b == "skewnorm_p8"].iloc[0]
        self.assertEqual(bad.status, "failed")
        self.assertFalse(bad.as_expected)
        self.assertIn("exceeds tolerance", bad.reason)
        self.assertEqual(m[m.member_b == "gaussian_b"].iloc[0].status, "matched")
        g = group_table(table, m).iloc[0]
        self.assertEqual((g.status, g.failed_members, g.n_pairs_failed), ("failed", "skewnorm_p8", 1))

    def test_unequal_lengths_inside_a_group_fail_the_match(self):
        st = STRUCTURES["sine_sd1.0"]
        a = describe_series(make_series(st, MEMBERS["gaussian"], 0), st, MEMBERS["gaussian"], 0)
        b = describe_series(make_series(st, MEMBERS["gaussian_b"], 0, n_total=2000), st,
                            MEMBERS["gaussian_b"], 0, n_context=1000)
        m = match_table(pd.DataFrame([a, b]))
        self.assertEqual(m.iloc[0].status, "failed")
        self.assertIn("unequal length", m.iloc[0].reason)


if __name__ == "__main__":
    unittest.main()
