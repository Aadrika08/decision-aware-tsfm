import math
import unittest

from decision_aware_tsfm.cache.budget import estimate_budget, measurement_key


def job(job_id, device, *, fixture=False, seed=1, origin=10, **changes):
    value = {
        "job_id": job_id,
        "model": "seasonal_naive" if device == "cpu" else "moirai",
        "family": "classical" if device == "cpu" else "tsfm",
        "adapter": None,
        "adapter_version": None,
        "checkpoint": "abc",
        "device": device,
        "batch_size": 1,
        "quantile_method": "linear",
        "settings": {"samples": 100},
        "horizon": 24,
        "context_length": 2048,
        "frequency": "H",
        "implementation_sha256": "code",
        "environment_sha256": "env",
        "fixture": fixture,
        "config": "ett1/H",
        "seed": seed,
        "origin": origin,
        "quantiles": [0.01, 0.5, 0.99],
    }
    value.update(changes)
    return value


def measurement(planned, seconds, output_bytes=100):
    return {
        "status": "verified",
        "timing_key": measurement_key(planned),
        "hardware": "x86-test" if planned["device"] == "cpu" else "A100-test",
        "device": planned["device"],
        "inference_seconds": seconds,
        "output_bytes": output_bytes,
        "peak_memory_bytes": 1000,
        "fixture": planned["fixture"],
        "job_id": planned["job_id"],
        "run_id": "run-1",
        "attempt": 1,
    }


class CacheBudgetTest(unittest.TestCase):
    def test_projects_cpu_gpu_and_storage_from_matching_measurements(self):
        cpu = job("cpu-1", "cpu")
        gpu = job("gpu-1", "cuda")
        report = estimate_budget(
            [cpu, job("cpu-2", "cpu", seed=9, origin=50), gpu],
            [measurement(cpu, 10, 200), measurement(gpu, 20, 500)],
            hardware={"cpu": "x86-test", "cuda": "A100-test"},
            loading_seconds={"cpu": 6, "cuda": 8},
            io_seconds={"cpu": 4, "cuda": 2},
            restart_fraction=0.1,
            failure_fraction=0.2,
        )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["scenario"], "real")
        self.assertAlmostEqual(report["resources"]["cpu"]["resource_hours"], 39 / 3600)
        self.assertAlmostEqual(report["resources"]["cuda"]["resource_hours"], 39 / 3600)
        self.assertIsNone(report["wall_clock_hours"])
        self.assertEqual(report["scope"], "resource_projection")
        self.assertEqual(report["resources"]["cpu"]["inference_seconds"], 20)
        self.assertEqual(report["resources"]["cpu"]["measured_sample_count"], 1)
        self.assertEqual(report["resources"]["cuda"]["measured_sample_count"], 1)
        self.assertEqual(report["storage_bytes"], 900)
        self.assertEqual(report["coverage"], {"matched_jobs": 3, "planned_jobs": 3})

    def test_signature_ignores_seed_and_origin_without_multiplying_quantiles(self):
        base = job("one", "cuda")
        repeat = job("two", "cuda", seed=999, origin=999)
        self.assertEqual(measurement_key(base), measurement_key(repeat))
        self.assertNotEqual(measurement_key(base), measurement_key(job("three", "cuda", settings={"samples": 200})))
        report = estimate_budget(
            [base], [measurement(base, 7)], hardware={"cpu": "x86-test", "cuda": "A100-test"},
            loading_seconds={"cpu": 0, "cuda": 0}, io_seconds={"cpu": 0, "cuda": 0},
        )
        self.assertAlmostEqual(report["resources"]["cuda"]["resource_hours"], 7 / 3600)

    def test_missing_or_incompatible_evidence_is_incomplete(self):
        planned = job("gpu", "cuda")
        wrong_hardware = measurement(planned, 4)
        wrong_hardware["hardware"] = "T4"
        report = estimate_budget(
            [planned], [wrong_hardware], hardware={"cpu": "x86-test", "cuda": "A100-test"},
            loading_seconds={"cpu": 0, "cuda": None}, io_seconds={"cpu": 0, "cuda": 0},
        )
        self.assertEqual(report["status"], "incomplete")
        self.assertIsNone(report["resources"]["cuda"]["resource_hours"])
        self.assertIsNone(report["storage_bytes"])
        self.assertIn("cuda: no compatible timing evidence", report["reasons"])
        self.assertIn("cuda: missing loading overhead evidence", report["reasons"])

    def test_rejects_malformed_measurements_and_unsupported_jobs(self):
        planned = job("gpu", "cuda")
        for bad_seconds in (0, -1, math.nan, math.inf, "4"):
            bad = measurement(planned, 4)
            bad["inference_seconds"] = bad_seconds
            with self.assertRaises(ValueError):
                estimate_budget([planned], [bad], hardware={"cuda": "A100-test"},
                                loading_seconds={"cuda": 0}, io_seconds={"cuda": 0})
        with self.assertRaises(ValueError):
            estimate_budget([job("x", "tpu")], [], hardware={}, loading_seconds={}, io_seconds={})

    def test_fixture_evidence_never_resolves_a_real_budget(self):
        real = job("real", "cuda")
        fixture = job("fixture", "cuda", fixture=True)
        incompatible = estimate_budget(
            [real], [measurement(fixture, 1)], hardware={"cuda": "A100-test"},
            loading_seconds={"cuda": 0}, io_seconds={"cuda": 0})
        self.assertEqual(incompatible["status"], "incomplete")
        fixture_report = estimate_budget(
            [fixture], [measurement(fixture, 1)], hardware={"cuda": "A100-test"},
            loading_seconds={"cuda": 0}, io_seconds={"cuda": 0})
        self.assertEqual(fixture_report["status"], "incomplete")
        self.assertEqual(fixture_report["scenario"], "fixture")
        self.assertIn("fixture projection is not a real compute budget", fixture_report["reasons"])

    def test_deduplicates_identical_evidence_and_rejects_conflicts(self):
        planned = job("gpu", "cuda")
        first = measurement(planned, 4, 20)
        report = estimate_budget([planned], [first, dict(first)], hardware={"cuda": "A100-test"},
                                 loading_seconds={"cuda": 0}, io_seconds={"cuda": 0})
        self.assertEqual(report["resources"]["cuda"]["measured_sample_count"], 1)
        conflict = dict(first, inference_seconds=5)
        with self.assertRaises(ValueError):
            estimate_budget([planned], [first, conflict], hardware={"cuda": "A100-test"},
                            loading_seconds={"cuda": 0}, io_seconds={"cuda": 0})
        with self.assertRaises(ValueError):
            estimate_budget([planned, dict(planned)], [first], hardware={"cuda": "A100-test"},
                            loading_seconds={"cuda": 0}, io_seconds={"cuda": 0})

    def test_independent_runs_are_distinct_samples_and_run_id_is_required(self):
        planned = job("gpu", "cuda")
        first = measurement(planned, 4)
        second = dict(first, run_id="run-2", inference_seconds=6)
        report = estimate_budget([planned], [first, second], hardware={"cuda": "A100-test"},
                                 loading_seconds={"cuda": 0}, io_seconds={"cuda": 0})
        self.assertEqual(report["resources"]["cuda"]["measured_sample_count"], 2)
        self.assertEqual(report["resources"]["cuda"]["inference_seconds"], 5)
        missing = dict(first)
        del missing["run_id"]
        with self.assertRaisesRegex(ValueError, "run_id"):
            estimate_budget([planned], [missing], hardware={"cuda": "A100-test"},
                            loading_seconds={"cuda": 0}, io_seconds={"cuda": 0})

    def test_quantile_grid_is_part_of_timing_signature(self):
        planned = job("gpu", "cuda", quantiles=[0.1, 0.5, 0.9])
        changed = dict(planned, quantiles=[0.5])
        self.assertNotEqual(measurement_key(planned), measurement_key(changed))


if __name__ == "__main__":
    unittest.main()
