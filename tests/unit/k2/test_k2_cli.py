"""The command on tiny hand-built inputs: output safety, the Omega-column switch,
the manifest-version seam, the optional figure, and the version-pin warning.
All numbers in these inputs are made up."""
import importlib.util
import json
import os
import unittest
from unittest import mock

import pandas as pd

from decision_aware_tsfm.analysis.k2 import environment
from decision_aware_tsfm.analysis.k2.families import METRICS, model_to_family
from decision_aware_tsfm.analysis.k2.loading import sha256_file

from ._shared import (SECOND_OMEGA, dominant_results, run_cli, scratch_dir, tiny_manifest,
                      tiny_results, write_tiny_inputs)

EXPECTED_TABLES = {"summary_stats.csv", "within_family_sensitivity.csv", "joined_table.csv",
                   "correlations_dataset_level.csv", "sign_counts.csv", "dataset_deltas.csv",
                   "correlations_config_level.csv", "joined_config_table.csv",
                   "mse_sensitivity.csv", "leave_one_model_out.csv",
                   "leave_one_model_out_summary.csv", "model_score_diagnostics.csv",
                   "paper_like_for_like.csv", "within_family_sensitivity_relabelled.csv"}


class Command(unittest.TestCase):
    def setUp(self):
        self.dir = scratch_dir()
        self.manifest, self.results = write_tiny_inputs(self.dir)

    def argv(self, outdir, *extra):
        return ["--manifest", self.manifest, "--results", self.results, "--outdir", outdir,
                "--bootstrap-resamples", "200", *extra]

    def rho(self, outdir, comparison="zero-shot vs deep"):
        s = pd.read_csv(os.path.join(outdir, "summary_stats.csv"))
        return float(s[(s.metric == "sMAPE") & (s.comparison == comparison)].spearman_rho.iloc[0])

    def test_one_command_writes_every_output_and_a_run_manifest(self):
        out = os.path.join(self.dir, "out")
        code, _ = run_cli(self.argv(out))
        self.assertEqual(code, 0)
        self.assertEqual(set(os.listdir(out)), EXPECTED_TABLES | {"run_manifest.json"})
        with open(os.path.join(out, "run_manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        self.assertEqual(m["inputs"]["results"]["sha256"], sha256_file(self.results))
        self.assertEqual(m["inputs"]["manifest"]["sha256"], sha256_file(self.manifest))
        self.assertEqual(m["model_to_family"], model_to_family())
        self.assertEqual(m["command"]["argv"][1:], self.argv(out))
        self.assertEqual(m["parameters"]["omega_col"], "omega_levels")
        self.assertEqual(m["parameters"]["bootstrap_resamples"], 200)
        self.assertEqual(m["roster"]["models_ignored"], ["not_in_any_family"])
        self.assertEqual(m["held_out"]["results_rows_excluded"], 0)
        for name in EXPECTED_TABLES:
            self.assertEqual(m["outputs_sha256"][name], sha256_file(os.path.join(out, name)))
        for key in ("python", "packages", "team_pins", "pin_mismatches"):
            self.assertIn(key, m["environment"])
        self.assertEqual(self.rho(out), 1.0)
        self.assertEqual(self.rho(out, "zero-shot vs statistical"), -1.0)
        self.assertEqual([f for f in os.listdir(self.dir) if ".partial-" in f], [])
        # A clean run: the manifest is recorded as not frozen, and nothing is warned about.
        self.assertEqual((m["manifest_file"]["frozen"], m["manifest_file"]["layout_version"],
                          m["manifest_file"]["declared_version"]), (False, "dev-v1", None))
        self.assertFalse(m["parameters"]["allow_missing_omega"])
        self.assertEqual(m["degenerate_resample_warnings"], [])
        self.assertEqual(m["held_out"]["manifest_fuzzy_reports"], [])

    def test_clean_run_prints_no_warning(self):
        out = os.path.join(self.dir, "out")
        code, text = run_cli(self.argv(out, "--allow-missing-omega"))
        self.assertEqual(code, 0)
        self.assertIn("frozen: no", text)
        for absent in ("n CHANGES", "held-out rule: dropped", "possible held-out name",
                       "bootstrap resamples were degenerate"):
            self.assertNotIn(absent, text)

    def test_existing_output_directory_is_not_overwritten(self):
        out = os.path.join(self.dir, "out")
        self.assertEqual(run_cli(self.argv(out))[0], 0)
        before = {f: sha256_file(os.path.join(out, f)) for f in os.listdir(out)}
        code, _ = run_cli(self.argv(out, "--omega-col", SECOND_OMEGA))
        self.assertIn("already exists", str(code))
        after = {f: sha256_file(os.path.join(out, f)) for f in os.listdir(out)}
        self.assertEqual(before, after)
        # An existing directory that K2 did not write is refused as well, even if empty.
        empty = os.path.join(self.dir, "empty")
        os.makedirs(empty)
        code, _ = run_cli(self.argv(empty))
        self.assertIn("already exists", str(code))
        self.assertEqual(os.listdir(empty), [])

    def test_output_directory_inside_an_existing_run_is_refused(self):
        out = os.path.join(self.dir, "out")
        self.assertEqual(run_cli(self.argv(out))[0], 0)
        before = sorted(os.listdir(out))
        for nested in (os.path.join(out, "nested"), os.path.join(out, "a", "b", "c")):
            with self.subTest(nested=nested):
                code, _ = run_cli(self.argv(nested))
                self.assertTrue(str(code).startswith("ERROR"))
                self.assertIn("inside an existing K2 run directory", str(code))
                self.assertIn(out, str(code))
                self.assertFalse(os.path.exists(nested))
        self.assertEqual(sorted(os.listdir(out)), before)

    def test_check_outdir_accepts_a_new_directory_under_plain_directories(self):
        from decision_aware_tsfm.analysis.k2.pipeline import OutputExistsError, check_outdir
        plain = os.path.join(self.dir, "plain", "deeper", "run2")
        self.assertEqual(check_outdir(plain), plain)
        run = os.path.join(self.dir, "run")
        os.makedirs(run)
        with open(os.path.join(run, "run_manifest.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        self.assertEqual(check_outdir(os.path.join(self.dir, "beside_the_run")),
                         os.path.join(self.dir, "beside_the_run"))
        for inside in (os.path.join(run, "x"), os.path.join(run, "x", "y")):
            with self.subTest(inside=inside), self.assertRaises(OutputExistsError):
                check_outdir(inside)
        with self.assertRaises(OutputExistsError):
            check_outdir(run)

    def test_bad_resample_count_and_seed_are_clean_errors(self):
        out = os.path.join(self.dir, "out")
        base = ["--manifest", self.manifest, "--results", self.results, "--outdir", out]
        for extra, part in ((["--bootstrap-resamples", "0"], "at least 1"),
                            (["--bootstrap-resamples", "-3"], "at least 1"),
                            (["--seed", "-1"], "between 0 and")):
            with self.subTest(extra=extra):
                code, _ = run_cli(base + extra)
                self.assertTrue(str(code).startswith("ERROR: "), code)
                self.assertIn(part, str(code))
                self.assertFalse(os.path.exists(out))

    def test_omega_column_must_be_a_float_column(self):
        rows = tiny_manifest()["active_dev_series"]
        for row in rows:
            row.update(omega_flag=True, omega_text="0.5")
        d = scratch_dir()
        manifest, results = write_tiny_inputs(d, manifest={"active_dev_series": rows})
        for column in ("omega_flag", "omega_text"):
            with self.subTest(column=column):
                out = os.path.join(d, column)
                code, _ = run_cli(["--manifest", manifest, "--results", results, "--outdir", out,
                                   "--omega-col", column])
                self.assertTrue(str(code).startswith("ERROR"))
                self.assertIn("not a float column", str(code))
                self.assertFalse(os.path.exists(out))

    def test_colliding_manifest_column_is_a_clean_error(self):
        rows = tiny_manifest()["active_dev_series"]
        for row in rows:
            row["omega_sd"] = 0.123
        d = scratch_dir()
        manifest, results = write_tiny_inputs(d, manifest={"active_dev_series": rows})
        for column in ("omega_sd", "omega_levels"):
            with self.subTest(column=column):
                out = os.path.join(d, column)
                code, _ = run_cli(["--manifest", manifest, "--results", results, "--outdir", out,
                                   "--omega-col", column])
                self.assertTrue(str(code).startswith("ERROR"))
                self.assertIn("omega_sd", str(code))
                self.assertFalse(os.path.exists(out))

    def missing_omega_inputs(self):
        rows = tiny_manifest()["active_dev_series"]
        for row in rows:
            row["omega_late"] = None if row["subset"] == "solar_H" else row["omega_levels"]
        d = scratch_dir()
        manifest, results = write_tiny_inputs(d, manifest={"active_dev_series": rows})
        return d, ["--manifest", manifest, "--results", results, "--bootstrap-resamples", "50",
                   "--omega-col", "omega_late"]

    def test_omega_column_missing_for_a_dataset_stops_the_run(self):
        d, argv = self.missing_omega_inputs()
        out = os.path.join(d, "out")
        code, _ = run_cli(argv + ["--outdir", out])
        self.assertTrue(str(code).startswith("ERROR"))
        for part in ("solar: 1 of 1 series", "--allow-missing-omega", "Nothing was written"):
            self.assertIn(part, str(code))
        self.assertFalse(os.path.exists(out))
        self.assertEqual([f for f in os.listdir(d) if ".partial-" in f], [])

    def test_allow_missing_omega_prints_the_changed_n_and_records_it(self):
        d, argv = self.missing_omega_inputs()
        out = os.path.join(d, "out")
        code, text = run_cli(argv + ["--outdir", out, "--allow-missing-omega"])
        self.assertEqual(code, 0)
        for part in ("WARNING: --allow-missing-omega is in effect",
                     "n CHANGES: 4 base datasets have an Omega value, not 5",
                     "base datasets removed (no series has a value): ['solar']",
                     "solar: 1 of 1 series have no value"):
            self.assertIn(part, text)
        self.assertEqual(set(pd.read_csv(os.path.join(out, "summary_stats.csv")).n), {4})
        self.assertEqual(set(pd.read_csv(os.path.join(out, "correlations_dataset_level.csv")).n), {4})
        self.assertNotIn("solar", set(pd.read_csv(os.path.join(out, "joined_table.csv")).dataset))
        with open(os.path.join(out, "run_manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        self.assertTrue(m["parameters"]["allow_missing_omega"])
        self.assertEqual(m["omega_report"]["base_datasets_removed_no_selected_omega"], ["solar"])
        self.assertEqual(m["omega_report"]["n_base_datasets"], 4)

    def test_manifest_version_and_frozen_flag_are_recorded(self):
        d = scratch_dir()
        manifest, results = write_tiny_inputs(
            d, manifest=dict(tiny_manifest(), frozen=True, manifest_version="made-up-v7"))
        out2 = os.path.join(d, "out")
        code, text = run_cli(["--manifest", manifest, "--results", results, "--outdir", out2,
                              "--bootstrap-resamples", "50"])
        self.assertEqual(code, 0)
        self.assertIn("frozen: yes", text)
        self.assertIn("declared version: made-up-v7", text)
        with open(os.path.join(out2, "run_manifest.json"), encoding="utf-8") as f:
            record = json.load(f)["manifest_file"]
        self.assertEqual((record["frozen"], record["declared_version"]), (True, "made-up-v7"))

    def test_results_without_a_needed_metric_column_is_a_clean_error(self):
        d = scratch_dir()
        manifest, results = write_tiny_inputs(
            d, results=tiny_results().drop(columns=["eval_metrics/MSE[0.5]"]))
        out = os.path.join(d, "out")
        code, _ = run_cli(["--manifest", manifest, "--results", results, "--outdir", out])
        self.assertTrue(str(code).startswith("ERROR"))
        self.assertIn("eval_metrics/MSE[0.5]", str(code))
        self.assertFalse(os.path.exists(out))

    def test_command_prints_the_mse_table_beside_the_headline_and_writes_it(self):
        d = scratch_dir()
        manifest, results = write_tiny_inputs(d, results=dominant_results())
        out = os.path.join(d, "out")
        code, text = run_cli(["--manifest", manifest, "--results", results, "--outdir", out,
                              "--bootstrap-resamples", "50"])
        self.assertEqual(code, 0)
        self.assertIn("Do not cite the MSE rows above without this table", text)
        self.assertIn("dominated by one model, VisionTS: median share of the family sum 0.750 "
                      "(an even share is 0.333), highest zero-shot MSE on 5 of 5 datasets", text)
        self.assertLess(text.index("K2a (sMAPE, MSE)"), text.index("Do not cite the MSE rows"))
        self.assertIn("its family comparison is on sMAPE only", text)
        mse = pd.read_csv(os.path.join(out, "mse_sensitivity.csv"), keep_default_na=False)
        deep = mse[mse.comparison == "zero-shot vs deep"].set_index("variant")
        self.assertAlmostEqual(deep.loc["as_recorded", "spearman_rho"], -0.1, places=12)
        self.assertAlmostEqual(deep.loc["dominant_model_removed", "spearman_rho"], -1.0, places=12)
        self.assertEqual(deep.loc["dominant_model_removed", "model_removed"], "VisionTS")
        self.assertEqual(deep.loc["as_recorded", "model_removed"], "")
        # The recorded headline is the as-recorded row, untouched.
        summary = pd.read_csv(os.path.join(out, "summary_stats.csv"))
        self.assertEqual(float(summary[(summary.metric == "MSE")
                                       & (summary.comparison == "zero-shot vs deep")].spearman_rho.iloc[0]),
                         -0.1)
        with open(os.path.join(out, "run_manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        self.assertEqual(m["dominant_model"]["model"], "VisionTS")
        self.assertIn("mean/rows", m["summary_labels"])
        self.assertNotIn("paper's choice", json.dumps(m["summary_labels"]))
        labels = pd.read_csv(os.path.join(out, "within_family_sensitivity_relabelled.csv"))
        self.assertFalse(labels.within_family.str.contains("paper's choice").any())
        legacy = pd.read_csv(os.path.join(out, "within_family_sensitivity.csv"))
        self.assertEqual(legacy.within_family.iloc[0], "mean across family (paper's choice)")

    def test_omega_column_switch_changes_the_result(self):
        a, b = os.path.join(self.dir, "levels"), os.path.join(self.dir, "second")
        self.assertEqual(run_cli(self.argv(a))[0], 0)
        self.assertEqual(run_cli(self.argv(b, "--omega-col", SECOND_OMEGA))[0], 0)
        self.assertEqual(self.rho(a), 1.0)
        self.assertEqual(self.rho(b), -1.0)
        with open(os.path.join(b, "run_manifest.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["parameters"]["omega_col"], SECOND_OMEGA)
        cols = pd.read_csv(os.path.join(b, "joined_table.csv")).columns
        self.assertIn(SECOND_OMEGA, cols)

    def test_missing_omega_column_fails_with_the_available_columns_and_writes_nothing(self):
        out = os.path.join(self.dir, "out")
        code, _ = run_cli(self.argv(out, "--omega-col", "omega_last_4096_guess"))
        message = str(code)
        self.assertTrue(message.startswith("ERROR"))
        for name in ("omega_last_4096_guess", "omega_levels", "omega_diffs", SECOND_OMEGA):
            self.assertIn(name, message)
        self.assertFalse(os.path.exists(out))
        self.assertEqual([f for f in os.listdir(self.dir) if ".partial-" in f], [])

    def test_new_manifest_layout_fails_cleanly(self):
        out = os.path.join(self.dir, "out")
        code, _ = run_cli(self.argv(out, "--manifest-version", "frozen-v2"))
        self.assertIn("frozen-v2", str(code))
        self.assertIn("4,096", str(code))
        self.assertFalse(os.path.exists(out))

    def test_held_out_rows_in_the_results_are_excluded_and_recorded(self):
        fake = [{"dataset": "pjm_made_up/H/short", "model": m, **{c: 5.0 for c in METRICS.values()}}
                for m in ("TiRex", "PatchTST")]
        d = scratch_dir()
        manifest, results = write_tiny_inputs(d, results=tiny_results(fake))
        clean, dirty = os.path.join(self.dir, "clean"), os.path.join(d, "dirty")
        self.assertEqual(run_cli(self.argv(clean))[0], 0)
        self.assertEqual(run_cli(["--manifest", manifest, "--results", results, "--outdir", dirty,
                                  "--bootstrap-resamples", "200"])[0], 0)
        with open(os.path.join(dirty, "run_manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        self.assertEqual(m["held_out"]["results_rows_excluded"], 2)
        self.assertEqual(m["held_out"]["results_datasets_excluded"], ["pjm_made_up/H/short"])
        for name in sorted(EXPECTED_TABLES):
            with self.subTest(table=name):
                self.assertEqual(sha256_file(os.path.join(clean, name)),
                                 sha256_file(os.path.join(dirty, name)))

    def test_held_out_row_in_the_manifest_stops_the_run(self):
        from ._shared import tiny_manifest
        d = scratch_dir()
        bad = tiny_manifest([{"series_id": "x", "subset": "M5_made_up_subset", "channel": 0,
                              "length": 3000, "omega_levels": 0.5, "omega_diffs": 0.1,
                              SECOND_OMEGA: 0.5}])
        manifest, results = write_tiny_inputs(d, manifest=bad)
        out = os.path.join(d, "out")
        code, _ = run_cli(["--manifest", manifest, "--results", results, "--outdir", out])
        self.assertIn("held-out", str(code))
        self.assertFalse(os.path.exists(out))

    def test_same_seed_gives_identical_tables(self):
        a, b, c = (os.path.join(self.dir, n) for n in ("a", "b", "c"))
        for out, seed in ((a, "1"), (b, "1"), (c, "2")):
            self.assertEqual(run_cli(self.argv(out, "--seed", seed))[0], 0)
        for name in sorted(EXPECTED_TABLES):
            with self.subTest(table=name):
                self.assertEqual(sha256_file(os.path.join(a, name)),
                                 sha256_file(os.path.join(b, name)))
        self.assertNotEqual(sha256_file(os.path.join(a, "correlations_dataset_level.csv")),
                            sha256_file(os.path.join(c, "correlations_dataset_level.csv")))
        self.assertEqual(sha256_file(os.path.join(a, "summary_stats.csv")),
                         sha256_file(os.path.join(c, "summary_stats.csv")))

    @unittest.skipIf(importlib.util.find_spec("matplotlib") is not None,
                     "matplotlib is installed here; this checks the path without it")
    def test_figure_flag_without_matplotlib_reports_and_continues(self):
        out = os.path.join(self.dir, "out")
        code, text = run_cli(self.argv(out, "--figure"))
        self.assertEqual(code, 0)
        self.assertIn("matplotlib not installed", text)
        self.assertNotIn("replication_figure.png", os.listdir(out))
        with open(os.path.join(out, "run_manifest.json"), encoding="utf-8") as f:
            fig = json.load(f)["figure"]
        self.assertEqual(fig, {"requested": True, "written": False,
                               "reason": "matplotlib not installed"})

    def test_version_mismatch_warns_and_does_not_fail(self):
        out = os.path.join(self.dir, "out")
        with mock.patch.dict(environment.TEAM_PINS, {"numpy": "0.0.1"}):
            code, text = run_cli(self.argv(out))
        self.assertEqual(code, 0)
        self.assertIn("WARNING", text)
        self.assertIn("numpy: pinned 0.0.1, running", text)
        with open(os.path.join(out, "run_manifest.json"), encoding="utf-8") as f:
            mismatches = json.load(f)["environment"]["pin_mismatches"]
        # on an interpreter that is not the pinned one (for example Python 3.10) the real
        # mismatches are reported too; the patched numpy pin must be added to exactly those
        real = {m["name"] for m in environment.pin_mismatches(environment.collect_versions())}
        self.assertNotIn("numpy", real)
        self.assertEqual({m["name"] for m in mismatches}, real | {"numpy"})
        self.assertTrue(os.path.exists(os.path.join(out, "summary_stats.csv")))

    def test_missing_input_file_is_a_clean_error(self):
        out = os.path.join(self.dir, "out")
        code, _ = run_cli(["--manifest", os.path.join(self.dir, "nope.json"),
                           "--results", self.results, "--outdir", out])
        self.assertIn("manifest not found", str(code))
        self.assertFalse(os.path.exists(out))


class VersionPins(unittest.TestCase):
    def test_no_warning_when_versions_equal_the_pins(self):
        self.assertEqual(environment.pin_mismatches(dict(environment.TEAM_PINS)), [])
        self.assertEqual(environment.mismatch_warning([]), "")

    def test_mismatch_and_missing_package_are_named(self):
        running = dict(environment.TEAM_PINS, scipy="9.9.9", pyarrow=None)
        text = environment.mismatch_warning(environment.pin_mismatches(running))
        self.assertIn("scipy: pinned 1.11.4, running 9.9.9", text)
        self.assertIn("pyarrow: pinned 16.1.0, running not installed", text)
        self.assertNotIn("numpy", text)

    def test_a_difference_in_the_last_version_part_is_a_mismatch(self):
        for name, running in (("numpy", "1.26.5"), ("numpy", "1.26.40"), ("python", "3.11.14"),
                              ("scipy", "1.11.4.post1"), ("pandas", "2.2")):
            with self.subTest(name=name, running=running):
                versions = dict(environment.TEAM_PINS, **{name: running})
                self.assertEqual(environment.pin_mismatches(versions),
                                 [{"name": name, "pinned": environment.TEAM_PINS[name],
                                   "running": running}])

    def test_collect_versions_reports_every_pinned_name(self):
        self.assertEqual(set(environment.collect_versions()), set(environment.TEAM_PINS))


if __name__ == "__main__":
    unittest.main()
