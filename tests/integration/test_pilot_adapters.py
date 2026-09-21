import importlib.metadata
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from decision_aware_tsfm.cache.adapters import load_adapter, validate_adapter
from decision_aware_tsfm.cache.executor import audit
from decision_aware_tsfm.cache.pilot import pilot_models
from decision_aware_tsfm.cache.planner import plan
from decision_aware_tsfm.cache.schema import TAUS


def pilot_job(adapter="autoarima_native", **updates):
    config = next(model for model in pilot_models("cpu") if model["adapter"] == adapter)
    job = {**config, "frequency": "H", "context_length": 2048, "horizon": 24, "seed": 42,
           "start_timestamp": "2020-01-01T00:00:00",
           "context_start_timestamp": "2020-01-01T00:00:00",
           "origin_timestamp": "2020-03-26T08:00:00", "fixture": True}
    job.update(updates)
    return job


class _Native:
    def __init__(self, **settings):
        self.settings = settings

    def fit(self, values):
        self.model_ = {"arma": (1, 0, 0, 0, 0, 0, 0), "coef": {"ar1": .2},
                       "sigma2": 1.0, "distribution": "normal"}
        return self

    def predict(self, h, level):
        result = {"mean": np.full(h, 50.0)}
        for width in level:
            result[f"lo-{width}"] = np.full(h, 50.0 - width)
            result[f"hi-{width}"] = np.full(h, 50.0 + width)
        return result


class _ETS(_Native):
    def fit(self, values):
        self.model_ = {"components": ["A", "N", "N"], "par": np.array([.1]),
                       "distribution": "normal"}
        return self

    def simulate(self, h, n_paths, seed, error_distribution):
        assert (n_paths, seed, error_distribution) == (4096, 42, None)
        return np.repeat(np.arange(n_paths, dtype=float)[:, None], h, axis=1)


class PilotAdapterTest(unittest.TestCase):
    def test_canonical_configs_are_fresh_and_device_is_moirai_only(self):
        first = pilot_models()
        second = pilot_models()
        self.assertEqual([model["model"] for model in first],
                         ["autoarima_native", "autoets_simulated", "seasonal_naive_native", "moirai_pilot"])
        self.assertEqual([model["device"] for model in first], ["cpu", "cpu", "cpu", "cuda"])
        first[0]["settings"]["max_p"] = 99
        self.assertEqual(second[0]["settings"]["max_p"], 5)

    @patch("importlib.metadata.version", return_value="2.1.1")
    def test_native_and_simulated_predictions_are_real_ordered_quantiles(self, _version):
        module = types.ModuleType("statsforecast.models")
        module.AutoARIMA = _Native
        module.AutoETS = _ETS
        module.SeasonalNaive = _Native
        package = types.ModuleType("statsforecast")
        package.models = module
        context = np.arange(2048.0)
        with patch.dict(sys.modules, {"statsforecast": package, "statsforecast.models": module}):
            native, native_measurement = load_adapter(pilot_job()).predict(context, pilot_job())
            simulated, simulated_measurement = load_adapter(pilot_job("autoets_simulated")).predict(
                context, pilot_job("autoets_simulated"))
        self.assertEqual(native.shape, (24, 11))
        self.assertTrue((np.diff(native, axis=1) >= 0).all())
        self.assertTrue((np.diff(simulated, axis=1) >= 0).all())
        self.assertAlmostEqual(simulated[0, 5], np.quantile(np.arange(4096.0), .5, method="linear"))
        self.assertGreater(native_measurement["fitting_seconds"], 0)
        self.assertEqual(simulated_measurement["fitted_model"]["components"], ["A", "N", "N"])
        np.testing.assert_array_equal(context, np.arange(2048.0))

    @patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError)
    def test_validation_for_planning_and_audit_does_not_require_statsforecast(self, _version):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            workload = {"schema_version": 1, "split": "development", "fixture": True,
                        "quantiles": list(TAUS), "rows": [{
                            "source_revision": "synthetic-1", "source_sha256": "a" * 64,
                            "source_path": "source.arrow", "dataset": "synthetic",
                            "config": "hourly", "series_id": "series", "item_id": "item",
                            "channel": 0, "frequency": "H", "start_timestamp": "2020-01-01T00:00:00",
                            "source_length": 3000, "usable_start": 0, "usable_end": 3000,
                            "context_start": 0, "context_length": 2048, "origin": 2048,
                            "horizon": 24, "seed": 42, "training_end": 2000,
                            "validation_end": 2500,
                        }]}
            planned = plan(workload, {"schema_version": 1, "models": [pilot_models("cpu")[0]]},
                           base_dir=base)
            self.assertEqual(planned["planned_jobs"], 1)
            self.assertEqual(len(audit(planned, base / "cache")["missing"]), 1)

    @patch("importlib.metadata.version", return_value="2.0.0")
    def test_loading_rejects_version_and_validation_rejects_drift(self, _version):
        with self.assertRaisesRegex(RuntimeError, "2.1.1"):
            load_adapter(pilot_job())
        with patch("importlib.metadata.version", return_value="2.1.1"):
            for job, message in ((pilot_job(horizon=2), "horizon"),
                                 (pilot_job(seed=1), "seed"),
                                 (pilot_job(settings={}), "configuration")):
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    validate_adapter(job)
            adapter = load_adapter(pilot_job())
            with self.assertRaisesRegex(ValueError, "context_length"):
                adapter.predict(np.arange(2047.0), pilot_job())
            for context in (np.ones(2048, dtype=bool), np.ones(2048, dtype=complex)):
                with self.subTest(dtype=context.dtype), self.assertRaisesRegex(ValueError, "real numeric"):
                    adapter.predict(context, pilot_job())


if __name__ == "__main__":
    unittest.main()
