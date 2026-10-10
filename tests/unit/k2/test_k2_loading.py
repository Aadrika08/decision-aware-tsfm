"""Loaders, the manifest seam and the held-out guard, on tiny hand-built inputs.

Every number in these inputs is made up. Dataset names containing a held-out
token are names only; no held-out data is read anywhere.
"""
import os
import unittest

import numpy as np
import pandas as pd

from decision_aware_tsfm.analysis.decision.guards import HeldOutLeakError
from decision_aware_tsfm.analysis.k2.families import METRICS, family_table
from decision_aware_tsfm.analysis.k2.loading import (
    ManifestLayoutError, OmegaColumnError, OmegaMissingError, ResultsLayoutError, _from_dev_v1,
    aggregate_omega, load_results, manifest_file_record, omega_columns, prepare_results,
    read_manifest, rejected_omega_columns, sha256_file)
from decision_aware_tsfm.analysis.k2.pipeline import join_omega_by_base
from decision_aware_tsfm.analysis.k2.statistic import spearman

from ._shared import (SECOND_OMEGA, TINY_SUBSETS, scratch_dir, tiny_manifest, tiny_results,
                      write_tiny_inputs)


def series_row(subset, i=0, **omega):
    row = {"series_id": f"{subset}_idx_{i}", "subset": subset, "channel": 0, "length": 3000,
           "omega_levels": 0.5, "omega_diffs": 0.05, SECOND_OMEGA: 0.5}
    row.update(omega)
    return row


class ManifestReader(unittest.TestCase):
    def test_canonical_table_and_omega_per_base_dataset(self):
        extra = [series_row("ett1_15T", omega_levels=0.5), series_row("ett1_15T", 1, omega_levels=0.9)]
        m = _from_dev_v1(tiny_manifest(extra))
        self.assertEqual(list(m.series.columns[:5]),
                         ["series_id", "source_group", "base", "freq", "length"])
        om = aggregate_omega(m, "omega_levels")
        ett1 = om.by_base.set_index("base").loc["ett1"]
        # ett1 has three series: 0.1 (ett1_H), 0.5 and 0.9 (ett1_15T). Each counts once.
        self.assertAlmostEqual(ett1.omega_levels, 0.5)
        self.assertEqual(ett1.n_series, 3)
        self.assertAlmostEqual(ett1.omega_sd, np.std([0.1, 0.5, 0.9], ddof=1))
        bf = om.by_base_freq.set_index(["base", "freq"]).omega
        self.assertAlmostEqual(bf[("ett1", "H")], 0.1)
        self.assertAlmostEqual(bf[("ett1", "15T")], 0.7)
        self.assertEqual(om.report["n_base_datasets"], 5)
        self.assertEqual(om.report["n_base_frequency_subsets"], 6)

    def test_unmapped_subset_is_reported_not_silently_dropped(self):
        extra = [series_row("brand_new_subset_H"), series_row("brand_new_subset_H", 1)]
        m = _from_dev_v1(tiny_manifest(extra))
        self.assertEqual(m.report["unmapped_subsets_not_used"], {"brand_new_subset_H": 2})
        self.assertEqual(m.report["n_active_series_read"], 7)
        self.assertEqual(m.report["n_series_used"], 5)
        self.assertNotIn("brand_new_subset_H", set(m.series.source_group))

    def test_known_duplicate_is_checked_before_it_is_left_out(self):
        twin = series_row("jena_weather_10T", omega_levels=0.3)
        same = dict(twin, subset="jena_weather", series_id="jena_weather_idx_0")
        m = _from_dev_v1(tiny_manifest([twin, same]))
        self.assertEqual(m.report["known_duplicate_subsets_not_used"],
                         {"jena_weather": {"n_series": 1, "duplicate_of": "jena_weather_10T"}})
        self.assertEqual(m.report["unmapped_subsets_not_used"], {})
        # If the "duplicate" does not carry the same values it is reported as unmapped.
        different = dict(same, omega_levels=0.31)
        m2 = _from_dev_v1(tiny_manifest([twin, different]))
        self.assertEqual(m2.report["known_duplicate_subsets_not_used"], {})
        self.assertEqual(m2.report["unmapped_subsets_not_used"], {"jena_weather": 1})

    def test_mapped_subsets_absent_from_the_manifest_are_listed(self):
        m = _from_dev_v1(tiny_manifest())
        self.assertIn("m4_hourly", m.report["mapped_subsets_absent_from_manifest"])
        self.assertNotIn("ett1_H", m.report["mapped_subsets_absent_from_manifest"])

    def test_held_out_manifest_row_is_refused(self):
        for name in ("M5_made_up_subset", "pjm_made_up_subset_H"):
            with self.subTest(name=name), self.assertRaises(HeldOutLeakError):
                _from_dev_v1(tiny_manifest([series_row(name)]))

    def test_wrong_layout_is_refused_with_the_keys_found(self):
        with self.assertRaises(ManifestLayoutError) as ctx:
            _from_dev_v1({"series": []})
        self.assertIn("active_dev_series", str(ctx.exception))
        self.assertIn("series", str(ctx.exception))
        with self.assertRaises(ManifestLayoutError):
            _from_dev_v1({"active_dev_series": [{"series_id": "a", "length": 5}]})


class OmegaColumnSeam(unittest.TestCase):
    def test_missing_omega_column_lists_the_available_columns(self):
        m = _from_dev_v1(tiny_manifest())
        with self.assertRaises(OmegaColumnError) as ctx:
            aggregate_omega(m, "omega_not_there")
        message = str(ctx.exception)
        for name in ("omega_not_there", "omega_levels", "omega_diffs", SECOND_OMEGA):
            self.assertIn(name, message)

    def test_non_omega_columns_cannot_be_selected(self):
        m = _from_dev_v1(tiny_manifest())
        self.assertEqual(omega_columns(m.series), ["omega_levels", "omega_diffs", SECOND_OMEGA])
        with self.assertRaises(OmegaColumnError):
            aggregate_omega(m, "length")

    def test_switching_the_omega_column_changes_the_result_as_expected(self):
        # Delta(zero-shot, deep) is 10, 20, 30, 40, 50 in dataset order. omega_levels
        # rises in that order (rho = +1); the second column falls (rho = -1).
        results, _ = prepare_results(tiny_results())
        m = _from_dev_v1(tiny_manifest())
        table = family_table(results, METRICS["sMAPE"])
        rho = {}
        for col in ("omega_levels", SECOND_OMEGA):
            om = aggregate_omega(m, col)
            self.assertEqual(om.omega_col, col)
            rho[col], _, n = spearman(join_omega_by_base(table, om), "zero-shot", "deep")
            self.assertEqual(n, len(TINY_SUBSETS))
        self.assertAlmostEqual(rho["omega_levels"], 1.0, places=12)
        self.assertAlmostEqual(rho[SECOND_OMEGA], -1.0, places=12)

    def test_only_float_columns_are_omega_columns(self):
        rows = tiny_manifest()["active_dev_series"]
        for i, row in enumerate(rows):
            row.update(omega_flag=True, omega_text=str(row["omega_levels"]), omega_count=i,
                       omega_mixed=None if i == 0 else True)
        m = _from_dev_v1({"active_dev_series": rows})
        self.assertEqual(omega_columns(m.series), ["omega_levels", "omega_diffs", SECOND_OMEGA])
        rejected = m.report["omega_columns_rejected"]
        self.assertEqual(sorted(rejected),
                         ["omega_count", "omega_flag", "omega_mixed", "omega_text"])
        self.assertIn("bool", rejected["omega_flag"])
        self.assertIn("str", rejected["omega_text"])
        self.assertIn("int", rejected["omega_count"])
        for name in rejected:
            with self.subTest(column=name), self.assertRaises(OmegaColumnError) as ctx:
                aggregate_omega(m, name)
            message = str(ctx.exception)
            self.assertIn("not a float column", message)
            self.assertIn(name, message)
            self.assertIn("omega_levels", message)             # the usable ones are listed
        self.assertEqual(rejected_omega_columns(m.series), {})  # none is carried downstream

    def missing_manifest(self, missing):
        """Tiny manifest plus two more ett1 series; `missing` maps series_id -> value."""
        rows = tiny_manifest([series_row("ett1_15T", 0, omega_levels=0.5),
                              series_row("ett1_15T", 1, omega_levels=0.9)])["active_dev_series"]
        for row in rows:
            row["omega_late"] = missing.get(row["series_id"], row["omega_levels"])
        return _from_dev_v1({"active_dev_series": rows})

    def test_column_missing_for_a_whole_dataset_stops_and_names_it(self):
        m = self.missing_manifest({"solar_H_idx_0": None})
        with self.assertRaises(OmegaMissingError) as ctx:
            aggregate_omega(m, "omega_late")
        message = str(ctx.exception)
        for part in ("'omega_late' has no value for 1 of 7 series", "1 base dataset(s)",
                     "solar: 1 of 1 series", "would leave the analysis", "--allow-missing-omega"):
            self.assertIn(part, message)
        self.assertIsInstance(ctx.exception, OmegaColumnError)   # the command reports it cleanly

    def test_column_missing_for_some_series_stops_and_gives_the_counts(self):
        m = self.missing_manifest({"ett1_15T_idx_0": None, "ett2_H_idx_0": float("inf")})
        with self.assertRaises(OmegaMissingError) as ctx:
            aggregate_omega(m, "omega_late")
        message = str(ctx.exception)
        for part in ("2 of 7 series", "2 base dataset(s)", "ett1: 1 of 3 series",
                     "its Omega would use the remaining series", "ett2: 1 of 1 series"):
            self.assertIn(part, message)

    def test_allow_missing_drops_what_has_no_value_and_reports_the_changed_n(self):
        m = self.missing_manifest({"solar_H_idx_0": None, "ett1_H_idx_0": None})
        om = aggregate_omega(m, "omega_late", allow_missing=True)
        by_base = om.by_base.set_index("base")
        self.assertEqual(sorted(by_base.index), ["electricity", "ett1", "ett2", "us_births"])
        # ett1 keeps its two 15T series (0.5 and 0.9); the H series has no value.
        self.assertAlmostEqual(by_base.loc["ett1", "omega_late"], 0.7)
        self.assertEqual(by_base.loc["ett1", "n_series"], 2)
        self.assertAlmostEqual(by_base.loc["ett1", "omega_sd"], np.std([0.5, 0.9], ddof=1))
        self.assertEqual(sorted(zip(om.by_base_freq.base, om.by_base_freq.freq)),
                         [("electricity", "H"), ("ett1", "15T"), ("ett2", "H"), ("us_births", "D")])
        self.assertFalse(om.by_base.omega_late.isna().any())
        self.assertEqual(om.report["n_base_datasets"], 4)
        self.assertEqual(om.report["n_base_datasets_before_missing_omega"], 5)
        self.assertEqual(om.report["n_base_frequency_subsets"], 4)
        self.assertEqual(om.report["n_base_frequency_subsets_before_missing_omega"], 6)
        self.assertEqual(om.report["n_series_missing_selected_omega"], 2)
        self.assertEqual(om.report["base_datasets_removed_no_selected_omega"], ["solar"])
        self.assertEqual(om.report["missing_selected_omega_by_base"],
                         {"ett1": {"n_missing": 1, "n_series": 3},
                          "solar": {"n_missing": 1, "n_series": 1}})
        # The join then has four datasets, and says so through n.
        results, _ = prepare_results(tiny_results())
        _, _, n = spearman(join_omega_by_base(family_table(results, METRICS["sMAPE"]), om),
                           "zero-shot", "deep")
        self.assertEqual(n, 4)

    def test_series_counts_under_the_flag_count_only_series_with_a_value(self):
        # ett1 has three series: H (0.1) and two at 15T (0.5, 0.9). One 15T series has no
        # value, so its subset keeps one series and the dataset keeps two of its three.
        m = self.missing_manifest({"ett1_15T_idx_0": None})
        om = aggregate_omega(m, "omega_late", allow_missing=True)
        ett1 = om.by_base.set_index("base").loc["ett1"]
        self.assertEqual(ett1.n_series, 2)                       # not 3
        self.assertAlmostEqual(ett1.omega_late, 0.5)             # mean of 0.1 and 0.9
        self.assertAlmostEqual(ett1.omega_sd, np.std([0.1, 0.9], ddof=1))
        self.assertAlmostEqual(ett1.omega_levels, 0.5)           # other columns keep all three
        by_freq = om.by_base_freq.set_index(["base", "freq"])
        self.assertEqual(by_freq.loc[("ett1", "15T"), "n_series"], 1)     # not 2
        self.assertAlmostEqual(by_freq.loc[("ett1", "15T"), "omega"], 0.9)
        self.assertEqual(by_freq.loc[("ett1", "H"), "n_series"], 1)
        self.assertEqual(om.report["n_base_datasets"], 5)        # no dataset leaves
        self.assertEqual(om.report["n_base_frequency_subsets"], 6)
        self.assertEqual(om.report["base_datasets_removed_no_selected_omega"], [])
        self.assertEqual(om.report["missing_selected_omega_by_base"],
                         {"ett1": {"n_missing": 1, "n_series": 3}})

    def test_subset_with_no_value_leaves_the_frequency_table_but_the_dataset_stays(self):
        # The only ett1 H series has no value: the (ett1, H) subset goes, ett1 stays on 15T.
        m = self.missing_manifest({"ett1_H_idx_0": None})
        om = aggregate_omega(m, "omega_late", allow_missing=True)
        self.assertNotIn(("ett1", "H"), set(zip(om.by_base_freq.base, om.by_base_freq.freq)))
        self.assertIn(("ett1", "15T"), set(zip(om.by_base_freq.base, om.by_base_freq.freq)))
        self.assertIn("ett1", set(om.by_base.base))
        self.assertEqual(om.report["n_base_frequency_subsets"], 5)
        self.assertEqual(om.report["n_base_frequency_subsets_before_missing_omega"], 6)
        self.assertEqual(om.report["n_base_datasets"], 5)
        self.assertFalse(om.by_base_freq.omega.isna().any())

    def test_complete_column_is_unaffected_by_the_flag(self):
        m = _from_dev_v1(tiny_manifest())
        strict = aggregate_omega(m, "omega_levels")
        allowed = aggregate_omega(m, "omega_levels", allow_missing=True)
        pd.testing.assert_frame_equal(strict.by_base, allowed.by_base)
        pd.testing.assert_frame_equal(strict.by_base_freq, allowed.by_base_freq)
        self.assertEqual(strict.report["n_series_missing_selected_omega"], 0)
        self.assertEqual(strict.report["missing_selected_omega_by_base"], {})
        self.assertEqual(list(strict.by_base.n_series), [1, 1, 1, 1, 1])

    def test_manifest_column_named_like_the_computed_spread_is_refused(self):
        rows = tiny_manifest()["active_dev_series"]
        for row in rows:
            row["omega_sd"] = 0.123
        with self.assertRaises(ManifestLayoutError) as ctx:
            _from_dev_v1({"active_dev_series": rows})
        self.assertIn("omega_sd", str(ctx.exception))
        self.assertIn("computes itself", str(ctx.exception))

    def test_spread_and_frequency_tables_use_the_selected_column(self):
        # ett1: omega_levels 0.1, 0.5, 0.9; the second column 0.9, 0.2, 0.7.
        extra = [series_row("ett1_15T", 0, omega_levels=0.5, **{SECOND_OMEGA: 0.2}),
                 series_row("ett1_15T", 1, omega_levels=0.9, **{SECOND_OMEGA: 0.7})]
        m = _from_dev_v1(tiny_manifest(extra))
        second = aggregate_omega(m, SECOND_OMEGA)
        ett1 = second.by_base.set_index("base").loc["ett1"]
        self.assertAlmostEqual(ett1.omega_sd, np.sqrt(0.13))        # not 0.4, the levels spread
        self.assertAlmostEqual(ett1[SECOND_OMEGA], 0.6)
        bf = second.by_base_freq.set_index(["base", "freq"]).omega
        self.assertAlmostEqual(bf[("ett1", "15T")], 0.45)           # not 0.7
        self.assertAlmostEqual(bf[("ett1", "H")], 0.9)              # not 0.1
        self.assertAlmostEqual(bf[("solar", "H")], 0.7)             # not 0.3

    def test_manifest_file_record(self):
        dev = manifest_file_record(tiny_manifest(), "dev-v1")
        self.assertEqual((dev["frozen"], dev["declared_version"], dev["layout_version"]),
                         (False, None, "dev-v1"))
        self.assertIn("no boolean 'frozen' key", dev["frozen_basis"])
        self.assertEqual(dev["top_level_keys"], ["active_dev_series"])
        frozen = manifest_file_record(dict(tiny_manifest(), frozen=True, manifest_version="2026-11"),
                                      "dev-v1")
        self.assertEqual((frozen["frozen"], frozen["declared_version"]), (True, "2026-11"))
        self.assertIn("'frozen' key", frozen["frozen_basis"])
        for not_a_flag in ("yes", 1, None):
            with self.subTest(value=not_a_flag):
                self.assertFalse(manifest_file_record(dict(tiny_manifest(), frozen=not_a_flag),
                                                      "dev-v1")["frozen"])
        self.assertFalse(manifest_file_record(dict(tiny_manifest(), frozen=False), "dev-v1")["frozen"])
        self.assertEqual(_from_dev_v1(tiny_manifest()).report["manifest_file"], dev)

    def test_new_layout_raises_naming_what_is_needed(self):
        d = scratch_dir()
        mpath, _ = write_tiny_inputs(d)
        with self.assertRaises(NotImplementedError) as ctx:
            read_manifest(mpath, version="frozen-v2")
        message = str(ctx.exception)
        for needed in ("frozen manifest file", "dataset", "config", "series id", "4,096",
                       "SUBSET_MAP"):
            self.assertIn(needed, message)
        with self.assertRaises(ValueError):
            read_manifest(mpath, version="v9")

    def test_reading_the_current_layout_from_a_file(self):
        d = scratch_dir()
        mpath, _ = write_tiny_inputs(d)
        m = read_manifest(mpath)
        self.assertEqual(m.report["manifest_version"], "dev-v1")
        self.assertEqual(sorted(m.series.source_group), sorted(TINY_SUBSETS))


class ResultsLoader(unittest.TestCase):
    def test_configuration_name_is_split(self):
        r, report = prepare_results(tiny_results())
        row = r[r.dataset == "us_births/D/short"].iloc[0]
        self.assertEqual((row.base, row.freq, row.term), ("us_births", "D", "short"))
        self.assertEqual(report["held_out_rows_excluded"], 0)
        self.assertEqual(report["n_configs"], 5)

    def test_held_out_rows_are_excluded_before_anything_is_computed_and_counted(self):
        fake = [{"dataset": name, "model": "TiRex", **{c: 1.0 for c in METRICS.values()}}
                for name in ("M5_made_up/D/short", "pjm_made_up/H/short", "pjm_made_up/H/long")]
        r, report = prepare_results(tiny_results(fake))
        self.assertEqual(report["held_out_rows_excluded"], 3)
        self.assertEqual(report["held_out_datasets_excluded"],
                         ["M5_made_up/D/short", "pjm_made_up/H/long", "pjm_made_up/H/short"])
        self.assertEqual(report["n_rows_used"], report["n_rows_read"] - 3)
        self.assertFalse(r.dataset.str.lower().str.contains("m5|pjm").any())
        clean, _ = prepare_results(tiny_results())
        pd.testing.assert_frame_equal(r.reset_index(drop=True), clean.reset_index(drop=True))

    def test_duplicate_rows_and_bad_names_are_refused(self):
        dup = tiny_results().iloc[:1].to_dict("records")
        with self.assertRaises(ResultsLayoutError):
            prepare_results(tiny_results(dup))
        bad = [{"dataset": "no_slashes", "model": "TiRex", **{c: 1.0 for c in METRICS.values()}}]
        with self.assertRaises(ResultsLayoutError):
            prepare_results(tiny_results(bad))
        with self.assertRaises(ResultsLayoutError):
            prepare_results(pd.DataFrame({"model": ["a"]}))

    def test_load_from_file_and_hash(self):
        d = scratch_dir()
        _, rpath = write_tiny_inputs(d)
        r, report = load_results(rpath)
        self.assertEqual(report["n_rows_read"], len(tiny_results()))
        self.assertEqual(len(sha256_file(rpath)), 64)
        with open(os.path.join(d, "abc.txt"), "w", encoding="utf-8") as f:
            f.write("abc")
        self.assertEqual(sha256_file(os.path.join(d, "abc.txt")),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


if __name__ == "__main__":
    unittest.main()
