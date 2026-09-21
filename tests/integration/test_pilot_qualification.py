import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from decision_aware_tsfm.cache import qualification
from decision_aware_tsfm.cache.pilot import pilot_models


def models(device="cuda"):
    return pilot_models(device=device)


class FakeAdapter:
    def __init__(self, fail=False):
        self.fail = fail

    def predict(self, context, job):
        if self.fail and job["seed"] == 43:
            raise RuntimeError("intentional evidence")
        base = np.arange(job["horizon"], dtype=float)[:, None] + np.arange(11)[None, :]
        return base, {"fitting_seconds": 0.01, "warnings": [], "fitted_model": {"kind": "fake"},
                      "hardware_platform": "fake-platform", "device": job["device"],
                      "peak_memory_bytes": 0}


class QualificationTest(unittest.TestCase):
    def test_full_orchestration_persists_sources_forecasts_and_repeatability(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            report = qualification.qualify(output, device="cpu", models=models("cpu"),
                                               adapter_loader=lambda job: FakeAdapter())
            self.assertEqual(report["status"], "test_only")
            self.assertEqual(report["evidence_kind"], "test_harness")
            self.assertEqual(report["counts"], {"attempted": 72, "passed": 72, "failed": 0})
            self.assertEqual(len(list((output / "sources").glob("*.npy"))), 2)
            self.assertEqual(len(list((output / "forecasts").glob("*.npy"))), 72)
            repeats = [row["repeatability"] for row in report["attempts"] if row["repeat"] == 1]
            self.assertTrue(repeats and all(row["exact"] for row in repeats))
            self.assertEqual(json.loads((output / "report.json").read_text())["status"], "test_only")
            with self.assertRaises(FileExistsError):
                qualification.qualify(output, device="cpu", models=models("cpu"),
                                      adapter_loader=lambda job: FakeAdapter())

    def test_failure_continues_and_partial_coverage_is_nonpassing(self):
        with tempfile.TemporaryDirectory() as temporary:
            selected = [model for model in models() if model["model"] == "seasonal_naive_native"]
            report = qualification.qualify(Path(temporary) / "run", models=selected, horizons=(1,),
                                               adapter_loader=lambda job: FakeAdapter(fail=True))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["counts"], {"attempted": 6, "passed": 4, "failed": 2})
            self.assertEqual(len(report["failures"]), 2)

    def test_interrupt_still_writes_atomic_final_report(self):
        class InterruptAdapter:
            def predict(self, context, job):
                raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with self.assertRaises(KeyboardInterrupt):
                qualification.qualify(output, models=models()[:1], horizons=(1,),
                                      adapter_loader=lambda job: InterruptAdapter())
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["counts"], {"attempted": 1, "passed": 0, "failed": 1})
            self.assertIn("KeyboardInterrupt", report["failures"][0]["error"])

    def test_system_exit_cannot_leave_a_passing_report(self):
        class HardStopAdapter:
            def predict(self, context, job):
                raise SystemExit(7)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with self.assertRaisesRegex(SystemExit, "7"):
                qualification.qualify(output, models=models()[:1], horizons=(1,),
                                      adapter_loader=lambda job: HardStopAdapter())
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertLess(report["counts"]["attempted"], 6)
            self.assertIn("SystemExit", report["failures"][0]["error"])

    def test_rejects_noncanonical_config_before_output_and_bad_measurement_as_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "invalid-config"
            changed = models("cpu")[:1]
            changed[0]["settings"]["season_length"] = 12
            with self.assertRaisesRegex(ValueError, "canonical"):
                qualification.qualify(output, device="cpu", models=changed,
                                      adapter_loader=lambda job: FakeAdapter())
            self.assertFalse(output.exists())

            class BadMeasurement:
                def predict(self, context, job):
                    return np.zeros((job["horizon"], 11)), {"fitting_seconds": float("nan")}

            report = qualification.qualify(Path(temporary) / "bad-measurement", device="cpu",
                models=models("cpu")[:1], horizons=(1,), adapter_loader=lambda job: BadMeasurement())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["counts"]["failed"], 6)
            self.assertIn("fitting_seconds", report["failures"][0]["error"])


if __name__ == "__main__":
    unittest.main()
