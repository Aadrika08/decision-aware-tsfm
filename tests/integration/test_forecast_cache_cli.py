"""Exercise the published commands with invented data, never model inference."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/forecast_cache.py"


class ForecastCacheCLITest(unittest.TestCase):
    def invoke(self, *args, success=True):
        result = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                                cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0 if success else 2, (result.stderr + result.stdout)[:2000])
        return json.loads(result.stdout) if result.stdout else result.stderr

    def test_fixture_plan_generate_audit_and_missing_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            self.invoke("fixture", fixture)
            args = ["--workload", fixture / "workload.json", "--models", fixture / "models.json"]
            planned = self.invoke("plan", *args)
            self.assertTrue(planned["fixture"])
            self.assertEqual(planned["planned_jobs"], 24)
            self.assertEqual(planned["planned_rows"], 1544)
            cache = root / "cache"
            self.invoke("audit", *args, "--cache", cache, success=False)
            self.invoke("generate", *args, "--cache", cache, "--workers", 1)
            self.assertEqual(self.invoke("audit", *args, "--cache", cache)["status"], "complete")
            again = self.invoke("generate", *args, "--cache", cache, "--workers", 1)
            self.assertEqual(again["generated"], 0)
            self.assertEqual(again["skipped"], 24)
            incomplete = self.invoke("budget", *args, success=False)
            self.assertEqual(incomplete["status"], "incomplete")
            self.assertEqual(incomplete["scenario"], "fixture")
            self.assertTrue(incomplete["reasons"])

    def test_audit_uses_saved_plan_without_original_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            self.invoke("fixture", fixture)
            args = ["--workload", fixture / "workload.json", "--models", fixture / "models.json"]
            frozen = root / "plan.json"
            planned = self.invoke("plan", *args, "--output", frozen)
            cache = root / "cache"
            self.invoke("generate", *args, "--cache", cache, "--workers", 1)
            # A consumer can audit an archive using the original plan. It must not
            # replan using that consumer's different environment or require sources.
            for job in planned["jobs"]:
                job["source_path"] = str(root / "unavailable.npy")
            frozen.write_text(json.dumps(planned))
            report = self.invoke("audit", "--plan-file", frozen, "--cache", cache)
            self.assertEqual(report["verified_jobs"], 24)

    def test_fixture_refuses_overwrite_and_plan_rejects_heldout_without_data(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture"
            self.invoke("fixture", fixture)
            self.invoke("fixture", fixture, success=False)
            path = fixture / "workload.json"
            workload = json.loads(path.read_text())
            workload["rows"][0]["dataset"] = "Salesforce/PJM"
            workload["rows"][0]["source_path"] = "/no/such/file.npy"
            path.write_text(json.dumps(workload))
            error = self.invoke("plan", "--workload", path, "--models", fixture / "models.json", success=False)
            self.assertIn("held", error.lower())

    def test_duplicate_json_keys_and_output_over_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture"
            self.invoke("fixture", fixture)
            workload = fixture / "workload.json"
            original = workload.read_bytes()
            args = ["--workload", workload, "--models", fixture / "models.json"]
            self.invoke("plan", *args, "--output", workload, success=False)
            self.assertEqual(workload.read_bytes(), original)
            workload.write_text('{"schema_version":1,"schema_version":2}')
            self.assertIn("Duplicate JSON key", self.invoke("plan", *args, success=False))

    def test_reports_stay_outside_cache_and_plan_outputs_are_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            self.invoke("fixture", fixture)
            args = ["--workload", fixture / "workload.json", "--models", fixture / "models.json"]
            plan_output = root / "plan.json"
            self.invoke("plan", *args, "--output", plan_output)
            original = plan_output.read_bytes()
            self.invoke("plan", *args, "--output", plan_output, success=False)
            self.assertEqual(plan_output.read_bytes(), original)
            cache = root / "cache"
            self.invoke("audit", *args, "--cache", cache, "--output", cache / "report.json",
                        success=False)
            self.assertFalse((cache / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
