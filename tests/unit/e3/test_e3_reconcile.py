"""Part A: the Omega reconciliation sweep."""
import importlib.util
import pathlib
import unittest

import numpy as np
import pandas as pd

from decision_aware_tsfm.analysis.experiments.e3.omega_reconcile import (
    PERIOD, noisy_sine, omega_sweep, reproduce_abdullah, reproduce_dayson, sd_for_target)
from decision_aware_tsfm.omega.compute import compute_omega

SCRIPT = pathlib.Path(__file__).parents[3] / "scripts" / "validate_omega_synthetics.py"


class Reconcile(unittest.TestCase):
    def test_reproduction_equals_the_teammates_script(self):
        spec = importlib.util.spec_from_file_location("validate_omega_synthetics", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        signal = mod.generate_synthetic_signals()["Moderately noisy signal"]
        self.assertEqual(reproduce_dayson(), compute_omega(np.asarray(signal, dtype=np.float64)))

    def test_omega_falls_as_noise_grows(self):
        s = omega_sweep(lengths=(1000,), sds=(0.25, 0.5, 1.0, 2.0), n_seeds=5)
        self.assertEqual(list(s.noise_sd), [0.25, 0.5, 1.0, 2.0])
        self.assertTrue((np.diff(s.omega_mean) < 0).all())
        self.assertTrue((s.omega_min <= s.omega_mean).all() and (s.omega_mean <= s.omega_max).all())

    def test_signal_is_a_unit_amplitude_period_24_sine_plus_scaled_noise(self):
        t = np.arange(480)
        clean = noisy_sine(480, 0.0, np.random.default_rng(3))
        np.testing.assert_allclose(clean, np.sin(2 * np.pi * t / 24), atol=1e-12)
        self.assertEqual(PERIOD, 24)
        self.assertAlmostEqual(float(np.abs(clean).max()), 1.0, places=12)
        np.testing.assert_allclose(clean[:24], clean[24:48], atol=1e-12)
        # the noise is the generator's standard normal stream times sd
        noisy = noisy_sine(480, 0.5, np.random.default_rng(3))
        np.testing.assert_allclose(noisy - clean, 0.5 * np.random.default_rng(3).standard_normal(480),
                                   atol=1e-12)
        self.assertFalse(np.array_equal(noisy, noisy_sine(480, 0.5, np.random.default_rng(4))))

    def test_sweep_cell_equals_a_hand_computation_over_seeds_0_to_n(self):
        s = omega_sweep(lengths=(1000,), sds=(0.5,), n_seeds=4).iloc[0]
        t = np.arange(1000)
        own = np.array([compute_omega(np.sin(2 * np.pi * t / 24)
                                      + 0.5 * np.random.default_rng(k).standard_normal(1000))
                        for k in range(4)])
        self.assertEqual((s.n, s.noise_sd, s.n_seeds), (1000, 0.5, 4))
        self.assertAlmostEqual(s.omega_mean, own.mean(), places=12)
        self.assertAlmostEqual(s.omega_sd, own.std(ddof=1), places=12)
        self.assertEqual((s.omega_min, s.omega_max), (own.min(), own.max()))

    def test_abdullah_setting_is_n_2000_noise_sd_1p5(self):
        r = reproduce_abdullah(n_seeds=4)                 # defaults for everything else
        t = np.arange(2000)
        own = np.array([compute_omega(np.sin(2 * np.pi * t / 24)
                                      + 1.5 * np.random.default_rng(k).standard_normal(2000))
                        for k in range(4)])
        self.assertEqual((r["n"], r["noise_sd"], r["n_seeds"]), (2000, 1.5, 4))
        self.assertAlmostEqual(r["omega_mean"], float(own.mean()), places=12)
        self.assertEqual(r["share_in_0p13_0p15"], float(((own >= 0.13) & (own <= 0.15)).mean()))

    def test_interpolated_noise_level_on_a_known_curve(self):
        s = pd.DataFrame({"n": 1000, "noise_sd": [0.5, 1.0, 1.5], "omega_mean": [0.5, 0.3, 0.1]})
        self.assertAlmostEqual(sd_for_target(s, 1000, 0.32), 0.95)
        self.assertAlmostEqual(sd_for_target(s, 1000, 0.2), 1.25)
        self.assertTrue(np.isnan(sd_for_target(s, 1000, 0.9)))


if __name__ == "__main__":
    unittest.main()
