"""Regression checks for the repository's stable import boundaries."""

from pathlib import Path
import unittest


class PackageLayoutTest(unittest.TestCase):
    def test_research_modules_are_importable_from_owned_package_areas(self):
        from decision_aware_tsfm.omega.compute import compute_omega

        self.assertTrue(callable(compute_omega))

    def test_all_owned_module_files_have_stable_locations(self):
        source_root = Path(__file__).parents[2] / "src" / "decision_aware_tsfm"
        expected_modules = [
            "data/gift_eval_manifest.py",
            "cache/forecast_smoke.py",
            "cache/release_verification.py",
            "analysis/k2/replication.py",
            "analysis/k3/matched_omega_pair.py",
        ]
        self.assertTrue(all((source_root / module).is_file() for module in expected_modules))

    def test_repository_root_has_no_research_python_modules(self):
        repository_root = Path(__file__).parents[2]
        root_modules = sorted(path.name for path in repository_root.glob("*.py"))
        self.assertEqual(root_modules, [])
