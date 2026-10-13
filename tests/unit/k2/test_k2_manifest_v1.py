"""The two-file manifest layout 'manifest-v1' (series_manifest.json + omega_values.json).

Every input here is a tiny hand-built fixture written inside the test; every
number is made up. No data file is read from the repository except in the
last class, which checks that the default single-file path is unchanged.
Names containing a held-out token are names only; no held-out data is read.
"""
import inspect
import json
import os
import unittest

import numpy as np
import pandas as pd

from decision_aware_tsfm.analysis.decision.guards import HeldOutLeakError
from decision_aware_tsfm.analysis.k2 import cli
from decision_aware_tsfm.analysis.k2.families import METRICS, family_table
from decision_aware_tsfm.analysis.k2.loading import (
    DEFAULT_MANIFEST_VERSION, MANIFEST_VERSIONS, ManifestJoinError, ManifestLayoutError,
    OmegaColumnError, OmegaMissingError, _from_manifest_v1, aggregate_omega, omega_columns,
    prepare_results, read_manifest, sha256_file)
from decision_aware_tsfm.analysis.k2.pipeline import (DEFAULT_MANIFEST, join_omega_by_base,
                                                      run_k2)
from decision_aware_tsfm.analysis.k2.statistic import spearman

from ._shared import (TINY_SUBSETS, reference_run, run_cli, scratch_dir, tiny_manifest,
                      tiny_results, write_tiny_inputs)
from .test_k2_cli import EXPECTED_TABLES

FULL, SHORT = "omega_levels", "omega_levels_4096"
LISTED = ["M5_Retail", "PJM_Electricity"]          # the form the manifest's own list has


def pair(subset, name="a", full=0.5, short=0.5, full_status="ok", short_status="ok", **series):
    """One series as (row of the series file, row of the Omega file).

    A value of None is an undefined Omega; its status then says why.
    """
    series_id = series.pop("series_id", f"{subset}__{name}")
    row = {"series_id": series_id, "subset": subset, "dataset": "Made/Up",
           "config": subset.replace("_", "/"), "item_id": name, "channel": None,
           "length": 3000, "horizons": [1, 24], "status": "active_dev"}
    row.update(series)
    omega = {"series_id": series_id, FULL: full, FULL + "_status": full_status,
             SHORT: short, SHORT + "_status": short_status,
             "history_length_4096": 3000, "history_start_index_4096": 0}
    return row, omega


def tiny_v1(extra=()):
    """Five single-series subsets. omega_levels rises with the subset order
    (0.1 ... 0.5); omega_levels_4096 falls with it (0.9 ... 0.5)."""
    pairs = [pair(subset, full=0.1 * (i + 1), short=0.9 - 0.1 * i, length=2000 + i)
             for i, subset in enumerate(TINY_SUBSETS)]
    pairs.extend(extra)
    series = {"active_dev_series": [p[0] for p in pairs], "excluded_series": [],
              "held_out_strict": list(LISTED)}
    return series, {"omega_values": [p[1] for p in pairs]}


def write_v1(directory, series, omega, results=None):
    """Write the two manifest files and the tiny results: (series path, Omega path, results path)."""
    rpath = os.path.join(directory, "tiny_results_v1.csv")
    (tiny_results() if results is None else results).to_csv(rpath, index=False, lineterminator="\n")
    spath = os.path.join(directory, "tiny_series_manifest.json")
    opath = os.path.join(directory, "tiny_omega_values.json")
    for path, content in ((spath, series), (opath, omega)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f)
    return spath, opath, rpath


def run_v1(extra=(), *args, results=None):
    """Run the command on a tiny two-file manifest: (exit code or message, stdout, outdir)."""
    d = scratch_dir()
    spath, opath, rpath = write_v1(d, *tiny_v1(extra), results=results)
    out = os.path.join(d, "out")
    code, text = run_cli(["--manifest", spath, "--manifest-version", "manifest-v1",
                          "--omega-values", opath, "--results", rpath, "--outdir", out,
                          "--bootstrap-resamples", "50", *args])
    return code, text, out


def rho_smape_deep(outdir):
    s = pd.read_csv(os.path.join(outdir, "summary_stats.csv"))
    return float(s[(s.metric == "sMAPE") & (s.comparison == "zero-shot vs deep")].spearman_rho.iloc[0])


class TwoFileReader(unittest.TestCase):
    def test_canonical_table_from_the_two_files(self):
        m = _from_manifest_v1(*tiny_v1([pair("ett1_15T", "b", full=0.5, short=0.2)]))
        self.assertEqual(list(m.series.columns[:5]),
                         ["series_id", "source_group", "base", "freq", "length"])
        self.assertEqual(omega_columns(m.series), [FULL, SHORT])
        by_id = m.series.set_index("series_id")
        self.assertEqual(tuple(by_id.loc["ett1_15T__b", ["source_group", "base", "freq"]]),
                         ("ett1_15T", "ett1", "15T"))
        self.assertAlmostEqual(by_id.loc["ett1_15T__b", FULL], 0.5)
        self.assertAlmostEqual(by_id.loc["ett1_15T__b", SHORT], 0.2)
        self.assertAlmostEqual(by_id.loc["us_births_D__a", FULL], 0.5)
        self.assertAlmostEqual(by_id.loc["us_births_D__a", SHORT], 0.5)
        self.assertAlmostEqual(by_id.loc["ett1_H__a", SHORT], 0.9)
        self.assertEqual(by_id.loc["ett2_H__a", "length"], 2001)
        report = m.report
        self.assertEqual(report["manifest_version"], "manifest-v1")
        self.assertEqual((report["n_active_series_read"], report["n_omega_rows_read"],
                          report["n_series_used"]), (6, 6, 6))
        self.assertEqual(report["held_out_listed"], LISTED)
        self.assertEqual(report["held_out_rows_found"], 0)
        self.assertEqual(report["unmapped_subsets_not_used"], {})
        self.assertEqual(report["subsets_used"]["ett1_15T"], 1)
        self.assertEqual(report["manifest_file"]["layout_version"], "manifest-v1")
        self.assertFalse(report["manifest_file"]["frozen"])     # the file does not say it is

    def test_omega_per_base_dataset_from_the_joined_values(self):
        extra = [pair("ett1_15T", "b", full=0.5), pair("ett1_15T", "c", full=0.9)]
        om = aggregate_omega(_from_manifest_v1(*tiny_v1(extra)), FULL)
        ett1 = om.by_base.set_index("base").loc["ett1"]
        # ett1 has three series: 0.1 (ett1_H), 0.5 and 0.9 (ett1_15T). Each counts once.
        self.assertAlmostEqual(ett1[FULL], 0.5)
        self.assertEqual(ett1.n_series, 3)
        self.assertAlmostEqual(ett1.omega_sd, np.std([0.1, 0.5, 0.9], ddof=1))
        self.assertAlmostEqual(om.by_base_freq.set_index(["base", "freq"]).omega[("ett1", "15T")], 0.7)

    def test_join_is_by_series_id_not_by_row_order(self):
        series, omega = tiny_v1([pair("ett1_15T", "b", full=0.8, short=0.3)])
        shuffled = {"omega_values": list(reversed(omega["omega_values"]))}
        a = _from_manifest_v1(series, omega).series
        b = _from_manifest_v1(series, shuffled).series
        pd.testing.assert_frame_equal(a, b)
        self.assertAlmostEqual(b.set_index("series_id").loc["ett1_H__a", FULL], 0.1)
        self.assertAlmostEqual(b.set_index("series_id").loc["ett1_15T__b", FULL], 0.8)

    def test_join_that_is_not_one_to_one_is_refused(self):
        series, omega = tiny_v1()
        rows = omega["omega_values"]
        other_id = dict(rows[0], series_id="ett1_H__renamed")
        cases = {
            "an Omega row is missing": ({"omega_values": rows[1:]},
                                        ("1 series id(s) have no row in the Omega file", "ett1_H__a")),
            "an Omega row has no series": ({"omega_values": rows + [dict(rows[0], series_id="zz__a")]},
                                           ("1 Omega row id(s) are not in the series file", "zz__a")),
            "an id differs": ({"omega_values": [other_id] + rows[1:]},
                              ("1 series id(s) have no row in the Omega file",
                               "1 Omega row id(s) are not in the series file")),
            "an Omega id is repeated": ({"omega_values": rows + [dict(rows[0])]},
                                        ("1 series id(s) occur more than once in the Omega file",
                                         "ett1_H__a")),
        }
        for name, (bad_omega, parts) in cases.items():
            with self.subTest(case=name), self.assertRaises(ManifestJoinError) as ctx:
                _from_manifest_v1(series, bad_omega)
            for part in parts:
                self.assertIn(part, str(ctx.exception))
            self.assertIn("one to one", str(ctx.exception))
        repeated = dict(series, active_dev_series=series["active_dev_series"]
                        + [dict(series["active_dev_series"][0])])
        with self.assertRaises(ManifestJoinError) as ctx:
            _from_manifest_v1(repeated, omega)
        self.assertIn("1 series id(s) occur more than once in the series file", str(ctx.exception))
        self.assertIsInstance(ctx.exception, ManifestLayoutError)   # the command reports it cleanly

    def test_wrong_layout_is_refused_with_what_was_found(self):
        series, omega = tiny_v1()
        with self.assertRaises(ManifestLayoutError) as ctx:
            _from_manifest_v1({"series": []}, omega)
        self.assertIn("active_dev_series", str(ctx.exception))
        with self.assertRaises(ManifestLayoutError) as ctx:
            _from_manifest_v1(series, {"active_dev_series": []})
        self.assertIn("omega_values", str(ctx.exception))
        no_subset = dict(series, active_dev_series=[
            {k: v for k, v in row.items() if k != "subset"} for row in series["active_dev_series"]])
        with self.assertRaises(ManifestLayoutError) as ctx:
            _from_manifest_v1(no_subset, omega)
        self.assertIn("'subset'", str(ctx.exception))
        no_status = {"omega_values": [{k: v for k, v in row.items() if k != SHORT + "_status"}
                                      for row in omega["omega_values"]]}
        with self.assertRaises(ManifestLayoutError) as ctx:
            _from_manifest_v1(series, no_status)
        self.assertIn(SHORT + "_status", str(ctx.exception))
        # An Omega value sitting in the series file as well would be ambiguous.
        both = dict(series, active_dev_series=[dict(row, **{FULL: 0.5})
                                               for row in series["active_dev_series"]])
        with self.assertRaises(ManifestLayoutError) as ctx:
            _from_manifest_v1(both, omega)
        self.assertIn(FULL, str(ctx.exception))

    def test_unmapped_subset_is_reported_not_silently_dropped(self):
        m = _from_manifest_v1(*tiny_v1([pair("brand_new_subset_H"), pair("brand_new_subset_H", "b")]))
        self.assertEqual(m.report["unmapped_subsets_not_used"], {"brand_new_subset_H": 2})
        self.assertEqual((m.report["n_active_series_read"], m.report["n_series_used"]), (7, 5))
        self.assertNotIn("brand_new_subset_H", set(m.series.source_group))

    def test_reading_from_files_needs_both_and_only_for_this_layout(self):
        d = scratch_dir()
        spath, opath, _ = write_v1(d, *tiny_v1())
        m = read_manifest(spath, "manifest-v1", omega_values_path=opath)
        self.assertEqual(m.report["manifest_version"], "manifest-v1")
        self.assertEqual(sorted(m.series.source_group), sorted(TINY_SUBSETS))
        self.assertIn("manifest-v1", MANIFEST_VERSIONS)
        with self.assertRaises(ManifestLayoutError) as ctx:
            read_manifest(spath, "manifest-v1")
        self.assertIn("--omega-values", str(ctx.exception))
        mpath, _ = write_tiny_inputs(d)
        for kwargs in ({"omega_values_path": opath}, {"exclude_held_out": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ManifestLayoutError) as ctx:
                read_manifest(mpath, "dev-v1", **kwargs)
            self.assertIn("manifest-v1", str(ctx.exception))


class UndefinedOmega(unittest.TestCase):
    """null with a status: never 0.0, never left out without a count."""

    def manifest(self):
        # ett1: H 0.1; 15T 0.5 and one flat history. solar: its only series is empty at 4,096.
        extra = [pair("ett1_15T", "b", full=0.5, short=0.4),
                 pair("ett1_15T", "c", full=None, short=None, full_status="flat_history",
                      short_status="flat_history")]
        series, omega = tiny_v1(extra)
        solar = next(r for r in omega["omega_values"] if r["series_id"] == "solar_H__a")
        solar.update({SHORT: None, SHORT + "_status": "empty_history"})
        return _from_manifest_v1(series, omega)

    def test_null_stays_undefined_in_the_table_and_the_reader_counts_it_by_status(self):
        m = self.manifest()
        by_id = m.series.set_index("series_id")
        self.assertTrue(np.isnan(by_id.loc["ett1_15T__c", FULL]))        # not 0.0
        self.assertTrue(np.isnan(by_id.loc["solar_H__a", SHORT]))
        self.assertEqual(by_id.loc["ett1_15T__c", FULL + "_status"], "flat_history")
        self.assertEqual(len(m.series), 7)                               # the row is not dropped
        self.assertEqual(m.report["omega_undefined_by_column"],
                         {FULL: {"flat_history": 1},
                          SHORT: {"empty_history": 1, "flat_history": 1}})
        self.assertEqual(m.report["omega_exactly_zero_by_column"], {FULL: 0, SHORT: 0})

    def test_undefined_omega_stops_the_run_by_default_and_gives_count_and_status(self):
        with self.assertRaises(OmegaMissingError) as ctx:
            aggregate_omega(self.manifest(), FULL)
        message = str(ctx.exception)
        for part in ("has no value for 1 of 7 series", "ett1: 1 of 3 series",
                     "its Omega would use the remaining series", "by status: {'flat_history': 1}",
                     "--allow-missing-omega"):
            self.assertIn(part, message)

    def test_under_the_flag_the_series_is_left_out_with_count_and_status(self):
        om = aggregate_omega(self.manifest(), FULL, allow_missing=True)
        ett1 = om.by_base.set_index("base").loc["ett1"]
        self.assertAlmostEqual(ett1[FULL], 0.3)               # mean of 0.1 and 0.5; 0.2 if null were 0.0
        self.assertEqual(ett1.n_series, 2)                    # not 3
        self.assertAlmostEqual(ett1.omega_sd, np.std([0.1, 0.5], ddof=1))
        by_freq = om.by_base_freq.set_index(["base", "freq"])
        self.assertAlmostEqual(by_freq.loc[("ett1", "15T"), "omega"], 0.5)   # 0.25 if null were 0.0
        self.assertEqual(by_freq.loc[("ett1", "15T"), "n_series"], 1)
        report = om.report
        self.assertEqual(report["n_series_missing_selected_omega"], 1)
        self.assertEqual(report["missing_selected_omega_by_base"],
                         {"ett1": {"n_missing": 1, "n_series": 3}})
        self.assertEqual(report["undefined_selected_omega_by_status"], {"flat_history": 1})
        self.assertEqual(report["undefined_selected_omega_by_base"], {"ett1": {"flat_history": 1}})
        self.assertEqual(report["base_datasets_removed_no_selected_omega"], [])
        self.assertEqual(report["base_datasets_under_half_defined"], {})
        self.assertEqual(report["n_base_datasets"], 5)

    def test_dataset_with_no_defined_series_is_removed_and_reported(self):
        om = aggregate_omega(self.manifest(), SHORT, allow_missing=True)
        self.assertNotIn("solar", set(om.by_base.base))
        report = om.report
        self.assertEqual(report["base_datasets_removed_no_selected_omega"], ["solar"])
        self.assertEqual((report["n_base_datasets"], report["n_base_datasets_before_missing_omega"]),
                         (4, 5))
        self.assertEqual(report["n_series_missing_selected_omega"], 2)
        self.assertEqual(report["undefined_selected_omega_by_status"],
                         {"empty_history": 1, "flat_history": 1})
        self.assertEqual(report["undefined_selected_omega_by_base"],
                         {"ett1": {"flat_history": 1}, "solar": {"empty_history": 1}})
        # The other column still has solar, so the two runs differ in n and say so.
        full = aggregate_omega(self.manifest(), FULL, allow_missing=True)
        self.assertIn("solar", set(full.by_base.base))
        results, _ = prepare_results(tiny_results())
        table = family_table(results, METRICS["sMAPE"])
        self.assertEqual(spearman(join_omega_by_base(table, om), "zero-shot", "deep")[2], 4)
        self.assertEqual(spearman(join_omega_by_base(table, full), "zero-shot", "deep")[2], 5)

    def test_dataset_kept_on_fewer_than_half_of_its_series_is_listed(self):
        extra = [pair("ett2_15T", "b", full=None, full_status="flat_history"),
                 pair("ett2_15T", "c", full=None, full_status="empty_history"),
                 pair("ett1_15T", "b", full=None, full_status="flat_history")]
        om = aggregate_omega(_from_manifest_v1(*tiny_v1(extra)), FULL, allow_missing=True)
        # ett2: 1 of 3 series defined (listed). ett1: 1 of 2, exactly half (not listed).
        self.assertEqual(om.report["base_datasets_under_half_defined"],
                         {"ett2": {"n_defined": 1, "n_series": 3}})
        self.assertIn("ett2", set(om.by_base.base))            # kept, on its one defined series
        self.assertAlmostEqual(om.by_base.set_index("base").loc["ett2", FULL], 0.2)
        self.assertEqual(om.report["undefined_selected_omega_by_base"],
                         {"ett1": {"flat_history": 1},
                          "ett2": {"empty_history": 1, "flat_history": 1}})

    def test_value_and_status_must_agree(self):
        cases = {
            "null with status ok": pair("ett1_15T", "b", full=None, full_status="ok"),
            "a value with a status that says undefined": pair("ett1_15T", "b", full=0.4,
                                                              full_status="flat_history"),
            "0.0 with a status that says undefined": pair("ett1_15T", "b", full=0.0,
                                                          full_status="flat_history"),
            "a status that is not text": pair("ett1_15T", "b", full=0.4, full_status=None),
            "an infinite value": pair("ett1_15T", "b", full=float("inf")),
        }
        for name, bad in cases.items():
            with self.subTest(case=name), self.assertRaises(ManifestLayoutError) as ctx:
                _from_manifest_v1(*tiny_v1([bad]))
            self.assertIn(FULL, str(ctx.exception))
            self.assertIn("1 row(s)", str(ctx.exception))

    def test_zero_with_status_ok_is_a_value_and_is_counted(self):
        m = _from_manifest_v1(*tiny_v1([pair("ett1_15T", "b", full=0.0, short=0.3)]))
        self.assertEqual(m.report["omega_exactly_zero_by_column"], {FULL: 1, SHORT: 0})
        self.assertEqual(m.report["omega_undefined_by_column"], {FULL: {}, SHORT: {}})
        om = aggregate_omega(m, FULL)                           # no flag needed: nothing is undefined
        ett1 = om.by_base.set_index("base").loc["ett1"]
        self.assertAlmostEqual(ett1[FULL], 0.05)                # mean of 0.1 and 0.0
        self.assertEqual(ett1.n_series, 2)
        self.assertEqual(om.report["n_series_selected_omega_exactly_zero"], 1)
        self.assertEqual(om.report["selected_omega_exactly_zero_by_base"], {"ett1": 1})
        self.assertEqual(om.report["n_series_missing_selected_omega"], 0)
        other = aggregate_omega(m, SHORT)
        self.assertEqual(other.report["n_series_selected_omega_exactly_zero"], 0)
        self.assertEqual(other.report["selected_omega_exactly_zero_by_base"], {})


class ColumnSelection(unittest.TestCase):
    def test_the_4096_column_gives_its_own_result(self):
        # Delta(zero-shot, deep) is 10, 20, 30, 40, 50 in dataset order. omega_levels rises
        # in that order (rho = +1); omega_levels_4096 falls (rho = -1).
        results, _ = prepare_results(tiny_results())
        table = family_table(results, METRICS["sMAPE"])
        m = _from_manifest_v1(*tiny_v1())
        rho = {}
        for col in (FULL, SHORT):
            om = aggregate_omega(m, col)
            self.assertEqual(om.omega_col, col)
            rho[col], _, n = spearman(join_omega_by_base(table, om), "zero-shot", "deep")
            self.assertEqual(n, 5)
        self.assertAlmostEqual(rho[FULL], 1.0, places=12)
        self.assertAlmostEqual(rho[SHORT], -1.0, places=12)

    def test_spread_counts_and_undefined_follow_the_selected_column(self):
        extra = [pair("ett1_15T", "b", full=0.5, short=0.2),
                 pair("ett1_15T", "c", full=0.9, short=None, short_status="flat_history")]
        m = _from_manifest_v1(*tiny_v1(extra))
        full = aggregate_omega(m, FULL)                          # complete: no flag needed
        self.assertEqual(full.by_base.set_index("base").loc["ett1", "n_series"], 3)
        self.assertEqual(full.report["undefined_selected_omega_by_status"], {})
        with self.assertRaises(OmegaMissingError):
            aggregate_omega(m, SHORT)
        short = aggregate_omega(m, SHORT, allow_missing=True)
        ett1 = short.by_base.set_index("base").loc["ett1"]
        self.assertAlmostEqual(ett1[SHORT], 0.55)                # mean of 0.9 and 0.2
        self.assertEqual(ett1.n_series, 2)
        self.assertAlmostEqual(ett1.omega_sd, np.std([0.9, 0.2], ddof=1))
        self.assertEqual(short.report["undefined_selected_omega_by_status"], {"flat_history": 1})

    def test_status_and_history_columns_cannot_be_selected(self):
        m = _from_manifest_v1(*tiny_v1())
        for name in (FULL + "_status", "history_length_4096", "omega_levels_2048"):
            with self.subTest(column=name), self.assertRaises(OmegaColumnError) as ctx:
                aggregate_omega(m, name)
            self.assertIn(SHORT, str(ctx.exception))             # the usable ones are listed
        self.assertEqual(m.report["omega_columns_available"], [FULL, SHORT])
        self.assertEqual(m.report["omega_columns_rejected"], {})


class HeldOutGuard(unittest.TestCase):
    def test_held_out_name_in_any_text_field_is_refused(self):
        cases = {
            "subset": pair("M5_made_up_subset"),
            "config": pair("ett1_15T", "b", config="pjm_made_up/H"),
            "dataset": pair("ett1_15T", "b", dataset="Made/PJM_made_up"),
            "item_id": pair("ett1_15T", "b", item_id="m5_made_up_item"),
            "series_id": pair("ett1_15T", "b", series_id="m5_made_up__b_dim5"),
        }
        for field_name, bad in cases.items():
            with self.subTest(field=field_name), self.assertRaises(HeldOutLeakError) as ctx:
                _from_manifest_v1(*tiny_v1([bad]))
            message = str(ctx.exception)
            self.assertIn(f"{field_name} = ", message)
            self.assertIn("1 row(s)", message)
            self.assertIn("--exclude-held-out-matches", message)

    def test_channel_suffix_dim5_is_not_a_held_out_name(self):
        """'_dim5' contains the letters "m5" but is the manifest's channel 5, not M5."""
        extra = [pair("ett1_15T", "b", full=0.5, series_id="ett1_15T__b_dim5"),
                 pair("ett2_15T", "c", full=0.7, series_id="ett2_15T__c_dim50")]
        plain = _from_manifest_v1(*tiny_v1(extra))                # no opt-in needed
        self.assertIn("ett1_15T__b_dim5", set(plain.series.series_id))
        self.assertIn("ett2_15T__c_dim50", set(plain.series.series_id))
        report = plain.report
        self.assertEqual((report["held_out_rows_found"], report["held_out_rows_excluded"],
                          report["held_out_exclusions"]), (0, 0, []))
        self.assertEqual((report["n_active_series_read"], report["n_series_used"]), (7, 7))
        by_base = aggregate_omega(plain, FULL).by_base.set_index("base")
        self.assertAlmostEqual(by_base.loc["ett1", FULL], 0.3)   # 0.1 and 0.5: the dim5 value is used
        self.assertEqual(by_base.loc["ett1", "n_series"], 2)
        # Still visible: the punctuation-insensitive layer reports both ids and keeps them.
        self.assertEqual([(h["field"], h["value"], h["action"], h["in_a_mapped_subset"])
                          for h in report["held_out_fuzzy_reports"]],
                         [("series_id", "ett1_15T__b_dim5", "kept, reported only", True),
                          ("series_id", "ett2_15T__c_dim50", "kept, reported only", True)])
        # The opt-in has nothing to leave out, so it changes nothing.
        asked = _from_manifest_v1(*tiny_v1(extra), exclude_held_out=True)
        pd.testing.assert_frame_equal(plain.series, asked.series)
        self.assertEqual((asked.report["held_out_rows_found"], asked.report["held_out_rows_excluded"]),
                         (0, 0))

    def test_the_manifests_own_list_is_applied(self):
        bad = pair("ett1_15T", "b", item_id="store_M-5.Retail_001")
        with self.assertRaises(HeldOutLeakError) as ctx:
            _from_manifest_v1(*tiny_v1([bad]))
        self.assertIn("held_out_strict entry 'M5_Retail'", str(ctx.exception))
        series, omega = tiny_v1([bad])
        del series["held_out_strict"]
        m = _from_manifest_v1(series, omega)                     # reported, not refused
        self.assertEqual([(h["field"], h["value"]) for h in m.report["held_out_fuzzy_reports"]],
                         [("item_id", "store_M-5.Retail_001")])

    def test_opt_in_exclusion_leaves_the_rows_out_and_counts_them(self):
        extra = [pair("ett1_15T", "b", full=0.5),
                 pair("ett1_15T", "c", full=0.9, series_id="ett1_15T__c_m5"),
                 pair("ett2_15T", "d", full=0.7, item_id="pjm_made_up_item")]
        m = _from_manifest_v1(*tiny_v1(extra), exclude_held_out=True)
        self.assertEqual(sorted(m.series.series_id),
                         sorted([f"{s}__a" for s in TINY_SUBSETS] + ["ett1_15T__b"]))
        report = m.report
        self.assertTrue(report["held_out_exclusion_requested"])
        self.assertEqual((report["held_out_rows_found"], report["held_out_rows_excluded"]), (2, 2))
        self.assertEqual((report["n_active_series_read"], report["n_series_used"]), (8, 6))
        self.assertEqual(report["held_out_rows_excluded_by_subset"], {"ett1_15T": 1, "ett2_15T": 1})
        self.assertEqual(sorted((e["field"], e["value"], e["n_rows"])
                                for e in report["held_out_exclusions"]),
                         [("item_id", "pjm_made_up_item", 1), ("series_id", "ett1_15T__c_m5", 1)])
        # The excluded values are not used: ett1 is 0.1 and 0.5, ett2 only its H series.
        by_base = aggregate_omega(m, FULL).by_base.set_index("base")
        self.assertAlmostEqual(by_base.loc["ett1", FULL], 0.3)
        self.assertEqual(by_base.loc["ett1", "n_series"], 2)
        self.assertAlmostEqual(by_base.loc["ett2", FULL], 0.2)
        self.assertEqual(by_base.loc["ett2", "n_series"], 1)

    def test_exclusion_changes_nothing_on_a_clean_manifest(self):
        plain = _from_manifest_v1(*tiny_v1())
        asked = _from_manifest_v1(*tiny_v1(), exclude_held_out=True)
        pd.testing.assert_frame_equal(plain.series, asked.series)
        self.assertEqual((asked.report["held_out_rows_found"], asked.report["held_out_rows_excluded"],
                          asked.report["held_out_exclusions"]), (0, 0, []))
        self.assertFalse(plain.report["held_out_exclusion_requested"])


class Command(unittest.TestCase):
    """What a person running the command on the two files gets."""

    UNDEFINED = (pair("ett1_15T", "b", full=0.5, short=0.4),
                 pair("ett1_15T", "c", full=None, short=None, full_status="flat_history",
                      short_status="empty_history"))

    def test_both_columns_run_and_both_input_hashes_are_recorded(self):
        d = scratch_dir()
        spath, opath, rpath = write_v1(d, *tiny_v1())
        common = ["--manifest", spath, "--manifest-version", "manifest-v1", "--omega-values", opath,
                  "--results", rpath, "--bootstrap-resamples", "50"]
        full, short = os.path.join(d, "full"), os.path.join(d, "short")
        code, text = run_cli(common + ["--outdir", full])
        self.assertEqual(code, 0)
        self.assertIn("layout manifest-v1", text)
        self.assertIn(opath, text)
        self.assertEqual(run_cli(common + ["--outdir", short, "--omega-col", SHORT])[0], 0)
        self.assertEqual(rho_smape_deep(full), 1.0)
        self.assertEqual(rho_smape_deep(short), -1.0)
        for out, column in ((full, FULL), (short, SHORT)):
            self.assertEqual(set(os.listdir(out)), EXPECTED_TABLES | {"run_manifest.json"})
            with open(os.path.join(out, "run_manifest.json"), encoding="utf-8") as f:
                m = json.load(f)
            self.assertEqual(sorted(m["inputs"]), ["manifest", "omega_values", "results"])
            self.assertEqual(m["inputs"]["manifest"]["sha256"], sha256_file(spath))
            self.assertEqual(m["inputs"]["omega_values"]["sha256"], sha256_file(opath))
            self.assertEqual(m["inputs"]["omega_values"]["bytes"], os.path.getsize(opath))
            self.assertNotEqual(m["inputs"]["manifest"]["sha256"],
                                m["inputs"]["omega_values"]["sha256"])
            self.assertEqual(m["inputs"]["results"]["sha256"], sha256_file(rpath))
            self.assertEqual(m["parameters"]["omega_col"], column)
            self.assertEqual(m["parameters"]["manifest_version"], "manifest-v1")
            self.assertEqual(m["manifest_file"]["layout_version"], "manifest-v1")
            self.assertEqual(m["manifest_report"]["n_omega_rows_read"], 5)
            self.assertEqual(m["held_out"]["manifest_held_out_strict"], LISTED)
            self.assertEqual(m["omega_report"]["undefined_selected_omega_by_status"], {})
        omegas = pd.read_csv(os.path.join(short, "dataset_deltas.csv"))
        omegas = omegas[(omegas.metric == "sMAPE") & (omegas.comparison == "zero-shot vs deep")]
        self.assertEqual(dict(zip(omegas.dataset, omegas.omega.round(6))),
                         {"ett1": 0.9, "ett2": 0.8, "solar": 0.7, "electricity": 0.6,
                          "us_births": 0.5})

    def test_undefined_omega_stops_the_command_and_writes_nothing(self):
        code, _, out = run_v1(self.UNDEFINED)
        self.assertTrue(str(code).startswith("ERROR"))
        for part in ("ett1: 1 of 3 series", "by status: {'flat_history': 1}",
                     "--allow-missing-omega", "Nothing was written"):
            self.assertIn(part, str(code))
        self.assertFalse(os.path.exists(out))

    def test_allow_missing_prints_count_and_status_and_records_them(self):
        code, text, out = run_v1(self.UNDEFINED, "--allow-missing-omega", "--omega-col", SHORT)
        self.assertEqual(code, 0)
        for part in ("WARNING: --allow-missing-omega is in effect",
                     "ett1: 1 of 3 series have no value (status {'empty_history': 1})",
                     "undefined Omega by status, all datasets: {'empty_history': 1}"):
            self.assertIn(part, text)
        with open(os.path.join(out, "run_manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        self.assertEqual(m["omega_report"]["n_series_missing_selected_omega"], 1)
        self.assertEqual(m["omega_report"]["undefined_selected_omega_by_status"], {"empty_history": 1})
        self.assertEqual(m["omega_report"]["undefined_selected_omega_by_base"],
                         {"ett1": {"empty_history": 1}})
        self.assertEqual(m["manifest_report"]["omega_undefined_by_column"],
                         {FULL: {"flat_history": 1}, SHORT: {"empty_history": 1}})
        joined = pd.read_csv(os.path.join(out, "joined_table.csv"))
        ett1 = joined[joined.dataset == "ett1"].iloc[0]
        self.assertAlmostEqual(ett1[SHORT], 0.65)                # mean of 0.9 and 0.4, not of three
        self.assertEqual(ett1.n_series, 2)
        self.assertEqual(set(pd.read_csv(os.path.join(out, "summary_stats.csv")).n), {5})

    def test_dataset_left_with_no_defined_series_changes_n_and_is_printed(self):
        gone = pair("solar_10T", "b", full=None, full_status="flat_history")
        series, omega = tiny_v1([gone])
        solar = next(r for r in omega["omega_values"] if r["series_id"] == "solar_H__a")
        solar.update({FULL: None, FULL + "_status": "empty_history"})
        d = scratch_dir()
        spath, opath, rpath = write_v1(d, series, omega)
        out = os.path.join(d, "out")
        code, text = run_cli(["--manifest", spath, "--manifest-version", "manifest-v1",
                              "--omega-values", opath, "--results", rpath, "--outdir", out,
                              "--bootstrap-resamples", "50", "--allow-missing-omega"])
        self.assertEqual(code, 0)
        for part in ("n CHANGES: 4 base datasets have an Omega value, not 5",
                     "base datasets removed (no series has a value): ['solar']",
                     "solar: 2 of 2 series have no value (status {'empty_history': 1, "
                     "'flat_history': 1})"):
            self.assertIn(part, text)
        self.assertEqual(set(pd.read_csv(os.path.join(out, "summary_stats.csv")).n), {4})

    def test_dataset_kept_on_under_half_of_its_series_is_printed(self):
        extra = (pair("ett2_15T", "b", full=None, full_status="flat_history"),
                 pair("ett2_15T", "c", full=None, full_status="flat_history"))
        code, text, _ = run_v1(extra, "--allow-missing-omega")
        self.assertEqual(code, 0)
        self.assertIn("fewer than half of the series have a defined Omega in: "
                      "{'ett2': {'n_defined': 1, 'n_series': 3}}", text)

    def test_zero_omega_is_used_and_printed(self):
        code, text, out = run_v1((pair("ett1_15T", "b", full=0.0),))
        self.assertEqual(code, 0)
        self.assertIn("1 series have Omega exactly 0.0 with status 'ok' in column 'omega_levels'",
                      text)
        self.assertIn("{'ett1': 1}", text)
        joined = pd.read_csv(os.path.join(out, "joined_table.csv"))
        self.assertAlmostEqual(joined[joined.dataset == "ett1"].iloc[0][FULL], 0.05)
        code, text, _ = run_v1()
        self.assertNotIn("exactly 0.0", text)

    def test_held_out_name_stops_the_command_and_the_opt_in_leaves_it_out(self):
        bad = (pair("ett1_15T", "b", full=0.5),
               pair("ett1_15T", "c", full=0.9, series_id="M5_made_up__c_dim5"))
        code, _, out = run_v1(bad)
        self.assertTrue(str(code).startswith("ERROR"))
        self.assertIn("series_id = 'M5_made_up__c_dim5'", str(code))
        self.assertIn("held-out", str(code))
        self.assertFalse(os.path.exists(out))
        code, text, out = run_v1(bad, "--exclude-held-out-matches")
        self.assertEqual(code, 0)
        for part in ("[WARNING] held-out rule: left out 1 manifest row(s) with series_id = "
                     "'M5_made_up__c_dim5'", "token 'm5'", "false positive",
                     "1 manifest row(s) left out in total"):
            self.assertIn(part, text)
        with open(os.path.join(out, "run_manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        self.assertEqual(m["held_out"]["manifest_rows_found"], 1)
        self.assertEqual(m["manifest_report"]["held_out_rows_excluded"], 1)
        self.assertEqual(m["manifest_report"]["n_series_used"], 6)
        joined = pd.read_csv(os.path.join(out, "joined_table.csv"))
        ett1 = joined[joined.dataset == "ett1"].iloc[0]
        self.assertAlmostEqual(ett1[FULL], 0.3)                  # 0.1 and 0.5; 0.9 is left out
        self.assertEqual(ett1.n_series, 2)

    def test_dim5_series_id_runs_without_the_opt_in_and_is_printed(self):
        rows = (pair("ett1_15T", "b", full=0.5),
                pair("ett1_15T", "c", full=0.9, series_id="ett1_15T__c_dim5"))
        code, text, out = run_v1(rows)
        self.assertEqual(code, 0)
        self.assertIn("possible held-out name in the manifest, NOT refused: series_id = "
                      "'ett1_15T__c_dim5', 1 row(s), in a subset the analysis uses", text)
        self.assertNotIn("held-out rule: left out", text)
        self.assertNotIn("--exclude-held-out-matches is in effect", text)
        with open(os.path.join(out, "run_manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        self.assertEqual(m["held_out"]["manifest_rows_found"], 0)
        self.assertEqual(m["manifest_report"]["held_out_rows_excluded"], 0)
        self.assertEqual(m["manifest_report"]["n_series_used"], 7)
        joined = pd.read_csv(os.path.join(out, "joined_table.csv"))
        ett1 = joined[joined.dataset == "ett1"].iloc[0]
        self.assertAlmostEqual(ett1[FULL], 0.5)                  # 0.1, 0.5 and 0.9: all three used
        self.assertEqual(ett1.n_series, 3)
        # With the opt-in the same tables come out: there is nothing for it to leave out.
        code, _, out_asked = run_v1(rows, "--exclude-held-out-matches")
        self.assertEqual(code, 0)
        for name in sorted(EXPECTED_TABLES):
            self.assertEqual(sha256_file(os.path.join(out, name)),
                             sha256_file(os.path.join(out_asked, name)), name)

    def test_join_mismatch_and_missing_second_file_are_clean_errors(self):
        d = scratch_dir()
        series, omega = tiny_v1()
        spath, opath, rpath = write_v1(d, series, {"omega_values": omega["omega_values"][1:]})
        out = os.path.join(d, "out")
        base = ["--manifest", spath, "--manifest-version", "manifest-v1", "--results", rpath,
                "--outdir", out, "--bootstrap-resamples", "50"]
        code, _ = run_cli(base + ["--omega-values", opath])
        self.assertTrue(str(code).startswith("ERROR"))
        self.assertIn("one to one", str(code))
        code, _ = run_cli(base)
        self.assertTrue(str(code).startswith("ERROR"))
        self.assertIn("--omega-values", str(code))
        code, _ = run_cli(base + ["--omega-values", os.path.join(d, "not_there.json")])
        self.assertTrue(str(code).startswith("ERROR"))
        self.assertIn("not found", str(code))
        self.assertFalse(os.path.exists(out))
        self.assertEqual([f for f in os.listdir(d) if ".partial-" in f], [])

    def test_the_new_options_are_refused_with_the_single_file_layout(self):
        d = scratch_dir()
        mpath, rpath = write_tiny_inputs(d)
        _, opath, _ = write_v1(d, *tiny_v1())
        out = os.path.join(d, "out")
        base = ["--manifest", mpath, "--results", rpath, "--outdir", out,
                "--bootstrap-resamples", "50"]
        for extra in (["--omega-values", opath], ["--exclude-held-out-matches"]):
            with self.subTest(extra=extra):
                code, _ = run_cli(base + extra)
                self.assertTrue(str(code).startswith("ERROR"))
                self.assertIn("manifest-v1", str(code))
                self.assertFalse(os.path.exists(out))


class DefaultPathUnchanged(unittest.TestCase):
    """The single-file development manifest stays the default and gives the same tables."""

    def test_defaults_of_the_command_and_of_the_run(self):
        args = cli.build_parser().parse_args([])
        self.assertEqual(args.manifest, DEFAULT_MANIFEST)
        self.assertTrue(DEFAULT_MANIFEST.replace(os.sep, "/").endswith(
            "artifacts/reference/manifests/dev_manifest.json"))
        self.assertEqual((args.manifest_version, DEFAULT_MANIFEST_VERSION), ("dev-v1", "dev-v1"))
        self.assertEqual((args.omega_col, args.omega_values, args.exclude_held_out_matches,
                          args.allow_missing_omega), ("omega_levels", None, False, False))
        defaults = {name: p.default for name, p in inspect.signature(run_k2).parameters.items()}
        self.assertEqual((defaults["manifest_path"], defaults["manifest_version"],
                          defaults["omega_values_path"], defaults["exclude_held_out"],
                          defaults["omega_col"], defaults["allow_missing_omega"]),
                         (DEFAULT_MANIFEST, "dev-v1", None, False, "omega_levels", False))
        self.assertEqual(inspect.signature(read_manifest).parameters["version"].default, "dev-v1")

    def test_the_reference_run_reads_the_single_file_and_nothing_else(self):
        _, manifest = reference_run()
        self.assertEqual(sorted(manifest["inputs"]), ["manifest", "results"])
        self.assertEqual(manifest["inputs"]["manifest"]["path"],
                         "artifacts/reference/manifests/dev_manifest.json")
        self.assertEqual(manifest["inputs"]["manifest"]["sha256"], sha256_file(DEFAULT_MANIFEST))
        self.assertEqual(manifest["manifest_file"]["layout_version"], "dev-v1")
        self.assertEqual(manifest["parameters"]["manifest_version"], "dev-v1")
        self.assertEqual(manifest["manifest_report"]["manifest_version"], "dev-v1")
        for key in ("n_omega_rows_read", "held_out_rows_excluded", "omega_undefined_by_column"):
            self.assertNotIn(key, manifest["manifest_report"])
        for key in ("undefined_selected_omega_by_status", "n_series_selected_omega_exactly_zero"):
            self.assertNotIn(key, manifest["omega_report"])

    def test_same_values_in_either_layout_give_byte_identical_tables(self):
        # The tiny single-file manifest, and the same series split into the two files.
        single = tiny_manifest(second_column=False)
        series_rows, omega_rows = [], []
        for row in single["active_dev_series"]:
            series_rows.append({k: v for k, v in row.items() if not k.startswith("omega")})
            omega_rows.append({"series_id": row["series_id"],
                               "omega_levels": row["omega_levels"], "omega_levels_status": "ok",
                               "omega_diffs": row["omega_diffs"], "omega_diffs_status": "ok"})
        d = scratch_dir()
        mpath, rpath = write_tiny_inputs(d, manifest=single)
        spath, opath, _ = write_v1(d, {"active_dev_series": series_rows},
                                   {"omega_values": omega_rows})
        old, new = os.path.join(d, "single"), os.path.join(d, "two")
        tail = ["--results", rpath, "--bootstrap-resamples", "50"]
        self.assertEqual(run_cli(["--manifest", mpath, "--outdir", old] + tail)[0], 0)
        self.assertEqual(run_cli(["--manifest", spath, "--manifest-version", "manifest-v1",
                                  "--omega-values", opath, "--outdir", new] + tail)[0], 0)
        for name in sorted(EXPECTED_TABLES):
            with self.subTest(table=name):
                self.assertEqual(sha256_file(os.path.join(old, name)),
                                 sha256_file(os.path.join(new, name)))
        with open(os.path.join(old, "run_manifest.json"), encoding="utf-8") as f:
            self.assertEqual(sorted(json.load(f)["inputs"]), ["manifest", "results"])


if __name__ == "__main__":
    unittest.main()
