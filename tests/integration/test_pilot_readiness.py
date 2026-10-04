import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from decision_aware_tsfm.cache.pilot import pilot_models
from decision_aware_tsfm.cache import executor
from decision_aware_tsfm.cache.readiness import inspect_pilot, main
from decision_aware_tsfm.cache.schema import TAUS


def handoff(folder):
    """Invented sources and Omega values, never research/approval evidence."""
    rows, omega_rows = [], []
    for index in range(2):
        source = folder / f"invented-{index}.npy"
        np.save(source, np.arange(4300, dtype=float) + index, allow_pickle=False)
        identity = dict(dataset="synthetic_fixture", config=f"invented_{index}/H",
                        series_id=f"stable-{index}", source_revision="invented-v1",
                        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        for origin in (4096, 4120):
            for h in (1, 24, 168):
                for seed in (42, 43):
                    rows.append(dict(identity, source_path=source.name, item_id=f"item-{index}",
                                     channel=0, frequency="H", start_timestamp="2020-01-01T00:00:00",
                                     source_length=4300, usable_start=0, usable_end=4300,
                                     context_start=origin-2048, context_length=2048, origin=origin,
                                     horizon=h, seed=seed, training_end=4096, validation_end=4300))
            for variant in ("current", "last_4096"):
                omega_rows.append(dict(identity, origin=origin, variant=variant,
                                       history_start=0 if variant == "current" else origin-4096,
                                       history_end=origin, omega=0.2 + 0.6*index))
    return (dict(schema_version=1, split="development", fixture=True, quantiles=list(TAUS), rows=rows),
            dict(schema_version=1, models=pilot_models()),
            dict(schema_version=1, fixture=True, variants={"current": "invented current",
                                                         "last_4096": "invented capped"}, rows=omega_rows))


class PilotReadinessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(executor._SOURCES.clear)
        self.addCleanup(executor._WINDOWS.clear)
        self.root = Path(self.temp.name)
        self.workload, self.models, self.omega = handoff(self.root)

    def inspect(self):
        return inspect_pilot(self.workload, self.models, self.omega, base_dir=self.root)

    def test_complete_readiness_and_first_slice_reuse_without_models(self):
        with patch("decision_aware_tsfm.cache.executor.load_adapter", side_effect=AssertionError("model import")):
            report, full, first = self.inspect()
        self.assertEqual(report["status"], "fixture_valid", report)
        self.assertEqual((full["planned_jobs"], full["planned_rows"]), (96, 6176))
        self.assertEqual((first["planned_jobs"], first["planned_rows"]), (24, 1544))
        self.assertEqual(report["source_windows_verified"], 12)
        self.assertEqual(report["first_slice"]["series_id"], "stable-0")
        self.assertEqual(report["first_slice"]["origin"], 4096)
        self.assertLess({j["job_id"] for j in first["jobs"]}, {j["job_id"] for j in full["jobs"]})

    def test_reordering_and_additional_omega_variant_preserve_ids(self):
        _, original, original_first = self.inspect()
        self.workload["rows"].reverse()
        self.omega["variants"]["additional"] = "invented sensitivity"
        self.omega["rows"] += [dict(row, variant="additional") for row in self.omega["rows"]
                              if row["variant"] == "current"]
        report, reordered, first = self.inspect()
        self.assertEqual(report["status"], "fixture_valid", report)
        self.assertEqual(original["jobs"], reordered["jobs"])
        self.assertEqual(original_first["jobs"], first["jobs"])

    def test_missing_coverage_and_bad_omega_fail_before_source_access(self):
        scenarios = ("missing_job", "missing_omega", "future_omega", "duplicate_omega",
                     "wrong_hash", "bad_cap", "nan_omega", "missing_description", "same_omega")
        for scenario in scenarios:
            self.workload, self.models, self.omega = handoff(self.root)
            if scenario == "missing_job": self.workload["rows"].pop()
            elif scenario == "missing_omega": self.omega["rows"].pop()
            elif scenario == "future_omega": self.omega["rows"][0]["history_end"] += 1
            elif scenario == "duplicate_omega": self.omega["rows"].append(self.omega["rows"][0])
            elif scenario == "wrong_hash": self.omega["rows"][0]["source_sha256"] = "0"*64
            elif scenario == "bad_cap": self.omega["rows"][1]["history_start"] = 1
            elif scenario == "nan_omega": self.omega["rows"][0]["omega"] = float("nan")
            elif scenario == "missing_description": self.omega["variants"].pop("current")
            else:
                for row in self.omega["rows"]: row["omega"] = 0.5
            with self.subTest(scenario=scenario), patch("decision_aware_tsfm.cache.readiness.load_source") as reader:
                report, planned, first = self.inspect()
                self.assertEqual(report["status"], "blocked", report)
                self.assertIsNone(planned)
                self.assertIsNone(first)
                reader.assert_not_called()

    def test_heldout_anywhere_and_real_counter_ids_rejected_before_reads(self):
        for update in ({"dataset": "M5"}, {"source_path": "pjm/source.npy"},
                       {"series_id": "old_idx_10"}, {"validation_end": 4100}):
            self.workload, self.models, self.omega = handoff(self.root)
            self.workload.update(fixture=False, approval_reference="invented-test-reference")
            self.models["approval_reference"] = "invented-test-reference"
            self.workload["rows"][-1].update(update)
            with self.subTest(update=update), patch("decision_aware_tsfm.cache.readiness.load_source") as reader:
                report, full, _ = self.inspect()
                self.assertEqual(report["status"], "blocked")
                self.assertIsNone(full)
                reader.assert_not_called()

    def test_source_corruption_and_nonfinite_values_never_yield_plans(self):
        source = self.root / self.workload["rows"][0]["source_path"]
        source.write_bytes(b"corrupt")
        report, full, _ = self.inspect()
        self.assertEqual(report["status"], "blocked")
        self.assertIn("checksum", report["reasons"][0])
        self.assertIsNone(full)
        self.workload, self.models, self.omega = handoff(self.root)
        values = np.arange(4300, dtype=float)
        values[4096] = np.nan
        np.save(source, values)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        for row in self.workload["rows"] + self.omega["rows"]:
            if row["series_id"] == "stable-0": row["source_sha256"] = digest
        report, full, _ = self.inspect()
        self.assertEqual(report["status"], "blocked")
        self.assertIn("finite", report["reasons"][0])
        self.assertIsNone(full)

    def test_wrong_roster_or_inconsistent_series_mapping_is_blocked(self):
        self.models["models"].pop()
        report, full, _ = self.inspect()
        self.assertEqual(report["status"], "blocked")
        self.assertIsNone(full)
        self.workload, self.models, self.omega = handoff(self.root)
        self.workload["rows"][-1]["item_id"] = "different-item"
        with patch("decision_aware_tsfm.cache.readiness.load_source") as reader:
            report, full, _ = self.inspect()
            self.assertEqual(report["status"], "blocked")
            self.assertIn("inconsistent source", report["reasons"][0])
            reader.assert_not_called()

    def test_aliased_source_with_inconsistent_length_is_not_deduplicated(self):
        original = self.workload["rows"][0]
        for row in self.workload["rows"]:
            if row["series_id"] == "stable-1":
                row.update(source_path=original["source_path"], source_sha256=original["source_sha256"],
                           source_length=4368)
        for row in self.omega["rows"]:
            if row["series_id"] == "stable-1": row["source_sha256"] = original["source_sha256"]
        report, full, _ = self.inspect()
        self.assertEqual(report["status"], "blocked")
        self.assertIsNone(full)
        self.assertIn("one-dimensional length", report["reasons"][0])

    def test_cli_preserves_inputs_and_writes_blocked_or_valid_artifacts(self):
        args = []
        for name, value in (("workload", self.workload), ("models", self.models), ("omega", self.omega)):
            path = self.root / f"{name}.json"
            path.write_text(json.dumps(value))
            args += [f"--{name}", str(path)]
        output = self.root / "ready"
        self.assertEqual(main(args + ["--output-directory", str(output)]), 0)
        before = (output / "plan.json").read_bytes()
        self.assertEqual(main(args + ["--output-directory", str(output)]), 2)
        self.assertEqual(before, (output / "plan.json").read_bytes())
        (self.root / "omega.json").unlink()
        blocked = self.root / "blocked"
        self.assertEqual(main(args + ["--output-directory", str(blocked)]), 2)
        self.assertEqual(json.loads((blocked / "report.json").read_text())["status"], "blocked")
        self.assertFalse((blocked / "plan.json").exists())
        self.assertFalse((blocked / "first-plan.json").exists())


if __name__ == "__main__":
    unittest.main()
