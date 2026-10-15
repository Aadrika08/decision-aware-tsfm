"""The Omega-only simulation: draws, failure rates, tests and their power."""
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from decision_aware_tsfm.analysis.experiments.e3.design import (
    DEFAULT_SEEDS, MEMBERS, N_CONTEXT, OMEGA_TOLERANCE, STRUCTURES)
from decision_aware_tsfm.analysis.experiments.e3.generator import context_innovations, make_series
from decision_aware_tsfm.analysis.experiments.e3.omega_only import (
    N_SIM_SEEDS, SIM_SEED_START, detectable_rate_difference, mcnemar_exact, mcnemar_power,
    sim_seeds, simulate, summarise, tolerance_in_sd_units)
from decision_aware_tsfm.analysis.experiments.e3.scoring import wilson_interval
from decision_aware_tsfm.omega.compute import compute_omega

from . import _shared


def _draws(table):
    """Long frame from {member: omega list}; one structure, seeds 0..n-1, variance column filled."""
    parts = [pd.DataFrame({"structure": "s", "member": m, "seed": range(len(v)),
                           "omega_levels": v, "innov_variance": np.linspace(0.9, 1.1, len(v))})
             for m, v in table.items()]
    return pd.concat(parts, ignore_index=True)


class Simulation(unittest.TestCase):
    def test_seeds_are_many_and_disjoint_from_the_design_seeds(self):
        s = sim_seeds()
        self.assertEqual((s.size, s[0], s[-1]), (N_SIM_SEEDS, SIM_SEED_START, SIM_SEED_START + N_SIM_SEEDS - 1))
        self.assertGreaterEqual(N_SIM_SEEDS, 2000)
        self.assertFalse(set(s.tolist()) & set(DEFAULT_SEEDS))

    def test_each_draw_is_the_design_generator_and_the_team_omega_on_the_context(self):
        d = simulate([1000, 1001], structures=["ar1_phi0.80", "sine_sd1.0"],
                     members=["gaussian", "student_t3"])
        self.assertEqual(len(d), 2 * 2 * 2)
        for r in d.itertuples():
            with self.subTest(structure=r.structure, member=r.member, seed=r.seed):
                st = STRUCTURES[r.structure]
                x = make_series(st, MEMBERS[r.member], r.seed)            # full design length
                self.assertEqual(r.omega_levels, compute_omega(x[:N_CONTEXT], diffs=False))
                self.assertNotEqual(r.omega_levels, compute_omega(x, diffs=False))
                self.assertEqual(r.innov_variance, float(np.var(context_innovations(x, st))))

    def test_simulation_is_reproducible(self):
        a = simulate([1000, 1001, 1002], structures=["sine_sd1.0"], members=["gaussian_b"])
        b = simulate([1000, 1001, 1002], structures=["sine_sd1.0"], members=["gaussian_b"])
        pd.testing.assert_frame_equal(a, b)


class Summary(unittest.TestCase):
    def test_failure_rates_and_discordant_counts_on_a_hand_built_example(self):
        ref = [0.301, 0.299] * 5                                                # mean 0.30, not constant
        base = [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.34, 0.25]     # fails on seeds 8, 9
        alt = [0.35, 0.35, 0.35, 0.35, 0.30, 0.30, 0.30, 0.30, 0.36, 0.30]      # fails on 0, 1, 2, 3, 8
        s = summarise(_draws({"gaussian": ref, "gaussian_b": base, "student_t3": alt}),
                      tolerance=0.03, n_design_seeds=30).set_index("member")
        self.assertTrue(np.isnan(s.loc["gaussian", "fail_rate"]))
        self.assertEqual((s.loc["gaussian_b", "n_failed"], s.loc["gaussian_b", "fail_rate"]), (2, 0.2))
        self.assertTrue(np.isnan(s.loc["gaussian_b", "mcnemar_p"]))            # the baseline is not tested against itself
        t = s.loc["student_t3"]
        self.assertEqual((t.n_failed, t.fail_rate, t.fail_rate_baseline), (5, 0.5, 0.2))
        self.assertAlmostEqual(t.fail_rate_diff, 0.3)
        self.assertEqual((t.n_member_only_fails, t.n_baseline_only_fails), (4, 1))
        self.assertAlmostEqual(t.mcnemar_p, stats.binomtest(4, 5, 0.5).pvalue)
        self.assertEqual((t.fail_lo, t.fail_hi), wilson_interval(5, 10))
        self.assertAlmostEqual(t.omega_mean, np.mean(alt))
        self.assertAlmostEqual(t.omega_sd, np.std(alt, ddof=1))
        self.assertAlmostEqual(t.mean_diff_vs_reference, np.mean(alt) - np.mean(ref))
        self.assertAlmostEqual(t.power_run, mcnemar_power(10, 0.4, 0.1))
        self.assertAlmostEqual(t.power_design, mcnemar_power(30, 0.4, 0.1))

    def test_failure_uses_the_registered_tolerance_and_the_same_seed_reference(self):
        d = _shared.omega_draws()
        w = d.pivot(index="seed", columns="member", values="omega_levels")
        s = summarise(d).set_index("member")
        for m in ("gaussian_b", "student_t3"):
            with self.subTest(member=m):
                own = float((np.abs(w[m] - w["gaussian"]) > OMEGA_TOLERANCE).mean())
                self.assertEqual(s.loc[m, "fail_rate"], own)
        self.assertEqual(s.loc["student_t3", "welch_p"],
                         float(stats.ttest_ind(w["student_t3"], w["gaussian"], equal_var=False).pvalue))
        self.assertEqual(s.loc["student_t3", "brown_forsythe_p"],
                         float(stats.levene(w["student_t3"], w["gaussian"], center="median").pvalue))

    def test_on_the_sine_the_innovation_law_changes_omega_noise_and_the_match_rate(self):
        # the corrected claim, on 300 fresh seeds: student_t3 is NOT just sampling noise
        s = summarise(_shared.omega_draws()).set_index("member")
        t, b = s.loc["student_t3"], s.loc["gaussian_b"]
        self.assertGreater(t.sd_ratio_vs_reference, 1.8)
        self.assertLess(t.brown_forsythe_p, 1e-6)
        self.assertGreater(t.fail_rate, b.fail_rate + 0.12)
        self.assertLess(t.mcnemar_p, 1e-4)
        self.assertGreater(t.mean_diff_vs_reference, 0.0)
        # two Gaussian draws: same spread, and still a clearly non-zero failure rate
        self.assertTrue(0.8 < b.sd_ratio_vs_reference < 1.25)
        self.assertTrue(0.05 < b.fail_rate < 0.25)
        # mechanism: on the sine, Omega falls when the realised noise variance is high
        self.assertLess(t.corr_omega_innov_variance, -0.3)
        self.assertLess(s.loc["gaussian", "corr_omega_innov_variance"], -0.15)

    def test_tolerance_in_sd_units_uses_the_sd_of_the_difference(self):
        rng = np.random.default_rng(5)
        a, b = 0.25 + 0.015 * rng.standard_normal(4000), 0.25 + 0.015 * rng.standard_normal(4000)
        t = tolerance_in_sd_units(_draws({"gaussian": a, "gaussian_b": b}), tolerance=0.03).iloc[0]
        self.assertAlmostEqual(t.omega_sd_gaussian, np.std(a, ddof=1))
        self.assertAlmostEqual(t.sd_of_difference_of_two_draws, np.std(b - a, ddof=1))
        self.assertAlmostEqual(t.tolerance_in_single_draw_sd, 2.0, delta=0.1)
        self.assertAlmostEqual(t.tolerance_in_difference_sd, 2.0 / np.sqrt(2), delta=0.07)
        self.assertEqual(t.gaussian_pair_fail_rate, float((np.abs(b - a) > 0.03).mean()))
        self.assertAlmostEqual(t.normal_theory_fail_rate, 2 * stats.norm.sf(np.sqrt(2)), delta=0.02)
        self.assertAlmostEqual(t.gaussian_pair_fail_rate, t.normal_theory_fail_rate, delta=0.03)
        self.assertAlmostEqual(t.tolerance_for_2_difference_sd, 2 * np.std(b - a, ddof=1))


class Power(unittest.TestCase):
    def test_exact_p_value_matches_scipy(self):
        for n10, n01 in ((0, 0), (5, 0), (3, 3), (21, 6), (1374, 321)):
            with self.subTest(n10=n10, n01=n01):
                want = 1.0 if n10 + n01 == 0 else stats.binomtest(n10, n10 + n01, 0.5).pvalue
                self.assertAlmostEqual(mcnemar_exact(n10, n01), want)

    def test_exact_power_agrees_with_a_simulation(self):
        rng = np.random.default_rng(11)
        n, p10, p01, reps = 30, 0.35, 0.08, 1500
        hits = 0
        for _ in range(reps):
            c = rng.multinomial(n, [p10, p01, 1 - p10 - p01])
            hits += mcnemar_exact(c[0], c[1]) <= 0.05
        self.assertAlmostEqual(mcnemar_power(n, p10, p01), hits / reps, delta=0.04)

    def test_power_properties(self):
        self.assertLessEqual(mcnemar_power(30, 0.1, 0.1), 0.05)           # no difference: at most alpha
        self.assertEqual(mcnemar_power(30, 0.0, 0.0), 0.0)
        self.assertLess(mcnemar_power(30, 0.3, 0.1), mcnemar_power(90, 0.3, 0.1))
        self.assertAlmostEqual(mcnemar_power(30, 0.3, 0.1), mcnemar_power(30, 0.1, 0.3))
        # large-sample formula: close to a simulation of the normal-approximation test
        rng = np.random.default_rng(12)
        n, p10, p01 = 2000, 0.12, 0.09
        c = rng.multinomial(n, [p10, p01, 1 - p10 - p01], size=4000)
        z = (c[:, 0] - c[:, 1]) / np.sqrt(c[:, 0] + c[:, 1])
        self.assertAlmostEqual(mcnemar_power(n, p10, p01), float((np.abs(z) > 1.96).mean()), delta=0.03)

    def test_detectable_difference_reaches_the_power_target_and_shrinks_with_n(self):
        for n in (30, 4000):
            with self.subTest(n=n):
                d = detectable_rate_difference(n, 0.14)
                p1 = 0.14 + d
                self.assertGreaterEqual(mcnemar_power(n, p1 * 0.86, 0.14 * (1 - p1)), 0.80 - 1e-6)
                p0 = 0.14 + 0.8 * d
                self.assertLess(mcnemar_power(n, p0 * 0.86, 0.14 * (1 - p0)), 0.80)
        self.assertGreater(detectable_rate_difference(30, 0.14), 5 * detectable_rate_difference(4000, 0.14))
        self.assertTrue(np.isnan(detectable_rate_difference(3, 0.5)))


if __name__ == "__main__":
    unittest.main()
