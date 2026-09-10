"""Release regression checks; requires the downloaded v0.1.0-smoke asset."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import verify_smoke_release as verify


@unittest.skipUnless((verify.ROOT / "cache/forecast-smoke-sample.zip").exists(),
                     "Download the pinned release asset for release integration tests")
class ReleaseTest(unittest.TestCase):
    archive = verify.ROOT / "cache/forecast-smoke-sample.zip"

    def test_published_release_and_historical_ledger_scope(self):
        result = verify.verify(self.archive)
        self.assertEqual(result["rows"], 96)
        self.assertEqual(result["completed_rerun_ledgers"], 1)
        self.assertEqual(len(result["unrelated_ledger_jobs"]), 1)
        self.assertFalse(result["torch_imported"])
        self.assertIn("blocked", result["full_grid_budget_status"])

    def test_changed_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.zip"
            path.write_bytes(self.archive.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify.verify(path)

    def test_report_metric_disagreement_is_rejected(self):
        original = verify.blob
        report = json.loads(original("docs/forecast-smoke-results.json"))
        report["costs"][0]["mean_pinball"] += 1

        def altered(path):
            return json.dumps(report).encode() if path.endswith("results.json") else original(path)

        with patch.object(verify, "blob", side_effect=altered):
            with self.assertRaises(AssertionError):
                verify.verify(self.archive)

    def test_missing_generation_provenance_is_rejected(self):
        # Substitute the expected hash only inside this deliberate corrupt fixture
        # so the provenance checks, beyond archive integrity, are exercised.
        output = io.BytesIO()
        with zipfile.ZipFile(self.archive) as source, zipfile.ZipFile(output, "w") as target:
            for name in source.namelist():
                if "1788896289708726716.json" not in name:
                    target.writestr(name, source.read(name))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-ledger.zip"
            path.write_bytes(output.getvalue())
            with patch.object(verify, "ZIP_SHA", verify.sha(output.getvalue())):
                with self.assertRaisesRegex(ValueError, "Missing generation provenance"):
                    verify.verify(path)


if __name__ == "__main__":
    unittest.main()
