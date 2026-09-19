import json
import inspect
import os
import subprocess
import sys
import unittest

import numpy as np

from decision_aware_tsfm.cache.adapters import load_adapter, validate_adapter


def fixture_job(**updates):
    job = dict(model="fixture_classical", family="classical", adapter="fixture",
               adapter_version="1", checkpoint=None, device="cpu", batch_size=1,
               quantile_method="constant", settings={"prediction": 7.5}, frequency="H",
               context_length=4, horizon=3, seed=42,
               start_timestamp="2020-01-01T00:00:00",
               origin_timestamp="2020-01-01T04:00:00",
               context_start_timestamp="2020-01-01T00:00:00", fixture=True)
    job.update(updates)
    return job


def seasonal_job(**updates):
    job = fixture_job(model="seasonal_naive", adapter="seasonal_naive_smoke", fixture=False,
                      context_length=2048, horizon=24,
                      quantile_method="empirical_lag24_linear",
                      settings={"season_length": 24, "residual_lag": 24})
    job.update(updates)
    return job


class AdapterTest(unittest.TestCase):
    def test_constant_fixture_is_planted_and_repeatable(self):
        adapter = load_adapter(fixture_job())
        first, measurement = adapter.predict(np.arange(4.0), fixture_job())
        second, _ = adapter.predict(np.arange(4.0), fixture_job(seed=999))
        np.testing.assert_array_equal(first, np.full((3, 11), 7.5))
        np.testing.assert_array_equal(second, first)
        self.assertEqual(measurement["device"], "cpu")
        self.assertEqual(measurement["peak_memory_bytes"], 0)
        self.assertTrue(measurement["hardware_platform"])

    def test_fixture_reads_context_only(self):
        job = fixture_job()
        adapter = load_adapter(job)
        context = np.arange(4.0)
        expected, _ = adapter.predict(context, job)
        observed, _ = adapter.predict(context, job)
        np.testing.assert_array_equal(observed, expected)
        self.assertEqual(tuple(inspect.signature(adapter.predict).parameters), ("context", "job"))

    def test_historical_seasonal_arithmetic(self):
        context = np.arange(2048.0)
        quantiles, measurement = load_adapter(seasonal_job()).predict(context, seasonal_job())
        expected = np.repeat(context[-24:, None] + 24, 11, axis=1)
        np.testing.assert_array_equal(quantiles, expected)
        self.assertEqual(measurement["device"], "cpu")

    def test_rejects_unknowns_mixes_and_unqualified_work(self):
        cases = (
            (fixture_job(adapter="mystery"), "Unknown adapter"),
            (fixture_job(model="not_a_fixture"), "fixture_*"),
            (fixture_job(settings={"prediction": True}), "finite"),
            (fixture_job(batch_size=False), "batch_size"),
            (seasonal_job(fixture=True), "fixture=false"),
            (seasonal_job(horizon=168), "horizon=24"),
            (seasonal_job(family="tsfm"), "family='classical'"),
            (seasonal_job(settings={"season_length": 12, "residual_lag": 24}), "settings"),
            (fixture_job(adapter="autoets", model="AutoETS"), "not qualified"),
            (fixture_job(adapter="autoarima", model="AutoARIMA"), "not qualified"),
        )
        for job, message in cases:
            with self.subTest(job=job), self.assertRaisesRegex(ValueError, message):
                validate_adapter(job)

    def test_rejects_bad_context_without_accepting_actuals(self):
        adapter = load_adapter(fixture_job())
        with self.assertRaisesRegex(ValueError, "context_length"):
            adapter.predict(np.arange(3.0), fixture_job())
        with self.assertRaises(TypeError):
            adapter.predict(np.arange(4.0), fixture_job(), np.arange(3.0))

    def test_import_does_not_load_model_or_legacy_smoke_modules(self):
        code = """
import json, sys
from decision_aware_tsfm.cache.adapters import validate_adapter
validate_adapter(dict(model='fixture_classical', family='classical', adapter='fixture',
    adapter_version='1', checkpoint=None, device='cpu', batch_size=1,
    quantile_method='constant', settings={'prediction': 1.0}, frequency='H',
    context_length=2, horizon=1, seed=1, start_timestamp='2020-01-01T00:00:00',
    origin_timestamp='2020-01-01T02:00:00',
    context_start_timestamp='2020-01-01T00:00:00', fixture=True))
print(json.dumps({name: name in sys.modules for name in (
    'torch', 'uni2ts', 'decision_aware_tsfm.cache.forecast_smoke')}))
"""
        env = dict(os.environ, PYTHONPATH="src")
        output = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
        self.assertEqual(json.loads(output), {"torch": False, "uni2ts": False,
                                              "decision_aware_tsfm.cache.forecast_smoke": False})


if __name__ == "__main__":
    unittest.main()
