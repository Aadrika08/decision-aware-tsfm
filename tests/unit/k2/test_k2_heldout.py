"""The held-out name checks, on made-up names and tiny hand-built inputs.

Names containing a held-out token are names only. No held-out data is read
anywhere; every number in these inputs is made up.
"""
import json
import os
import unittest

from decision_aware_tsfm.analysis.decision.guards import HeldOutLeakError
from decision_aware_tsfm.analysis.k2.heldout import (
    HeldOutListError, check_listed, fuzzy_reasons, scan_names, squash, strict_reasons)
from decision_aware_tsfm.analysis.k2.loading import (
    _from_dev_v1, held_out_hits, identity_fields, prepare_results)

from ._shared import (SECOND_OMEGA, metric_row, run_cli, scratch_dir, tiny_manifest, tiny_results,
                      write_tiny_inputs)

LISTED = ["M5_Retail", "PJM_Electricity"]          # the form the manifest's own list has
ZERO_WIDTH_SPACE = "\u200b"


def series_row(subset, series_id=None, **extra):
    row = {"series_id": series_id or f"{subset}_idx_9", "subset": subset, "channel": 0,
           "length": 3000, "omega_levels": 0.99, "omega_diffs": 0.05, SECOND_OMEGA: 0.5}
    row.update(extra)
    return row


class NameRules(unittest.TestCase):
    def test_squash_keeps_letters_and_digits_only(self):
        self.assertEqual(squash("M-5_Retail"), "m5retail")
        self.assertEqual(squash("P.J.M Hourly"), "pjmhourly")
        self.assertEqual(squash("M5" + ZERO_WIDTH_SPACE), "m5")
        self.assertEqual(squash("_-. "), "")

    def test_strict_layer_known_answers(self):
        cases = {
            "M5_made_up": ["token 'm5'"],
            "pjm_made_up/H/short": ["token 'pjm'"],
            "ram5gb_cluster/H/short": ["token 'm5'"],          # a false positive, by design visible
            "electricity_H": [],
            "M-5_Retail": [],                                   # not strict without the list
            "P.J.M": [],
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(strict_reasons(name), expected)

    def test_listed_entries_match_on_letters_and_digits(self):
        for name in ("M-5_Retail", "m5 retail", "M5.Retail_item_7", "store/M_5-RETAIL/x"):
            with self.subTest(name=name):
                self.assertIn("held_out_strict entry 'M5_Retail'", strict_reasons(name, LISTED))
        self.assertEqual(strict_reasons("P.J.M_Electricity_zone", LISTED),
                         ["held_out_strict entry 'PJM_Electricity'"])
        self.assertEqual(strict_reasons("retail_m4", LISTED), [])
        # A name can match both ways; both reasons are given.
        self.assertEqual(strict_reasons("M5_Retail", LISTED),
                         ["token 'm5'", "held_out_strict entry 'M5_Retail'"])

    def test_fuzzy_layer_known_answers(self):
        cases = {"M-5_Retail": ["m5"], "m 5": ["m5"], "P.J.M": ["pjm"], "pj_m_hourly": ["pjm"],
                 "Walmart_M_5": ["m5"], "ett1_H_item_5": ["m5"], "electricity_H": [],
                 "m4_hourly": []}
        for name, tokens in cases.items():
            with self.subTest(name=name):
                self.assertEqual(fuzzy_reasons(name),
                                 [f"token {t!r} after removing punctuation" for t in tokens])

    def test_scan_separates_the_layers_and_removes_duplicates(self):
        names = ["M5_x", "M5_x", "M-5_Retail", "P.J.M", "solar_H", "item_5"]
        strict, fuzzy = scan_names(names, LISTED)
        self.assertEqual(sorted(strict), ["M-5_Retail", "M5_x"])
        self.assertEqual(sorted(fuzzy), ["P.J.M", "item_5"])
        strict, fuzzy = scan_names(names)                      # no list: M-5_Retail is fuzzy only
        self.assertEqual(sorted(strict), ["M5_x"])
        self.assertEqual(sorted(fuzzy), ["M-5_Retail", "P.J.M", "item_5"])

    def test_unusable_list_is_refused(self):
        self.assertEqual(check_listed(None), ())
        self.assertEqual(check_listed(LISTED), tuple(LISTED))
        for bad in ([""], ["_-"], [5], "M5_Retail", {"a": 1}):
            with self.subTest(bad=bad), self.assertRaises(HeldOutListError):
                check_listed(bad)


class ManifestGuard(unittest.TestCase):
    def test_every_text_field_is_an_identity_field(self):
        import pandas as pd
        rows = pd.DataFrame(tiny_manifest([series_row("ett1_H", dataset="x", status="active_dev",
                                                      horizons=[1, 24])])["active_dev_series"])
        rows["dataset"] = rows["dataset"].fillna("d")
        self.assertEqual(identity_fields(rows), ["series_id", "subset", "dataset", "status"])

    def test_held_out_series_id_under_a_benign_subset_is_refused_and_named(self):
        bad = tiny_manifest([series_row("electricity_H", "m5_retail_item_0001")])
        with self.assertRaises(HeldOutLeakError) as ctx:
            _from_dev_v1(bad)
        message = str(ctx.exception)
        for part in ("held-out", "series_id = 'm5_retail_item_0001'", "1 row(s)",
                     "'electricity_H'", "token 'm5'", "refused"):
            self.assertIn(part, message)

    def test_held_out_name_in_any_other_text_field_is_refused(self):
        for field_name in ("dataset", "config", "source"):
            with self.subTest(field=field_name):
                bad = tiny_manifest([series_row("ett1_H", "ett1_H_idx_9",
                                                **{field_name: "pjm_made_up_zone"})])
                with self.assertRaises(HeldOutLeakError) as ctx:
                    _from_dev_v1(bad)
                self.assertIn(f"{field_name} = 'pjm_made_up_zone'", str(ctx.exception))

    def test_the_manifests_own_list_is_applied(self):
        rows = [series_row("ett1_H", "store_M-5.Retail_001")]
        with_list = dict(tiny_manifest(rows), held_out_strict=LISTED)
        with self.assertRaises(HeldOutLeakError) as ctx:
            _from_dev_v1(with_list)
        self.assertIn("held_out_strict entry 'M5_Retail'", str(ctx.exception))
        # Without the list the same name is not refused, but it is reported.
        m = _from_dev_v1(tiny_manifest(rows))
        self.assertEqual(m.report["held_out_listed"], [])
        self.assertEqual([(h["field"], h["value"]) for h in m.report["held_out_fuzzy_reports"]],
                         [("series_id", "store_M-5.Retail_001")])
        self.assertEqual(with_list["held_out_strict"], LISTED)

    def test_spelling_variants_as_subset_names(self):
        refused = {"M-5_Retail": True, "m 5": False, "M5" + ZERO_WIDTH_SPACE: True,
                   "P.J.M": False, "pj_m_hourly": False, "Walmart_M_5": False}
        for name, is_refused in refused.items():
            with self.subTest(name=name):
                manifest = dict(tiny_manifest([series_row(name, "x")]), held_out_strict=LISTED)
                if is_refused:
                    with self.assertRaises(HeldOutLeakError):
                        _from_dev_v1(manifest)
                    continue
                m = _from_dev_v1(manifest)
                reports = m.report["held_out_fuzzy_reports"]
                self.assertEqual([(h["field"], h["value"], h["n_rows"], h["in_a_mapped_subset"])
                                  for h in reports], [("subset", name, 1, False)])
                self.assertEqual(reports[0]["action"], "kept, reported only")
                self.assertEqual(m.report["unmapped_subsets_not_used"], {name: 1})
                self.assertEqual(m.report["n_series_used"], 5)

    def test_fuzzy_match_in_a_used_subset_is_reported_and_kept(self):
        m = _from_dev_v1(tiny_manifest([series_row("ett1_H", "ett1_H_item_5")]))
        self.assertEqual(m.report["n_series_used"], 6)
        (hit,) = m.report["held_out_fuzzy_reports"]
        self.assertEqual((hit["field"], hit["value"], hit["in_a_mapped_subset"], hit["subsets"]),
                         ("series_id", "ett1_H_item_5", True, ["ett1_H"]))

    def test_clean_manifest_reports_nothing_and_lists_the_fields(self):
        m = _from_dev_v1(tiny_manifest())
        self.assertEqual(m.report["held_out_fuzzy_reports"], [])
        self.assertEqual(m.report["held_out_fields_checked"], ["series_id", "subset"])
        import pandas as pd
        rows = pd.DataFrame(tiny_manifest()["active_dev_series"])
        self.assertEqual(held_out_hits(rows, ["series_id", "subset"], LISTED), ([], []))

    def test_empty_list_entry_is_refused(self):
        with self.assertRaises(HeldOutListError):
            _from_dev_v1(dict(tiny_manifest(), held_out_strict=["M5_Retail", ""]))


class ResultsGuard(unittest.TestCase):
    def fake(self, *datasets):
        return [metric_row(d, m, 5.0) for d in datasets for m in ("TiRex", "PatchTST")]

    def test_exclusions_are_listed_with_counts_and_reasons(self):
        raw = tiny_results(self.fake("ram5gb_cluster/H/short", "M-5/D/short", "PJM/H/short"))
        r, report = prepare_results(raw)
        self.assertEqual(report["held_out_exclusions"], [
            {"dataset": "PJM/H/short", "n_rows": 2, "reasons": ["token 'pjm'"]},
            {"dataset": "ram5gb_cluster/H/short", "n_rows": 2, "reasons": ["token 'm5'"]}])
        self.assertEqual(report["held_out_rows_excluded"], 4)
        self.assertEqual(report["held_out_fuzzy_reports"], [
            {"dataset": "M-5/D/short", "n_rows": 2, "action": "kept, reported only",
             "reasons": ["token 'm5' after removing punctuation"]}])
        self.assertIn("M-5/D/short", set(r.dataset))             # reported, not dropped
        self.assertNotIn("ram5gb_cluster/H/short", set(r.dataset))

    def test_the_manifests_list_is_applied_to_the_results(self):
        raw = tiny_results(self.fake("M-5_Retail/D/short"))
        kept, _ = prepare_results(raw)
        self.assertIn("M-5_Retail/D/short", set(kept.dataset))
        dropped, report = prepare_results(raw, LISTED)
        self.assertNotIn("M-5_Retail/D/short", set(dropped.dataset))
        self.assertEqual(report["held_out_exclusions"][0]["reasons"],
                         ["held_out_strict entry 'M5_Retail'"])
        self.assertEqual(report["held_out_listed"], LISTED)


class CommandOutput(unittest.TestCase):
    """What a person running the command sees."""

    def run_with(self, manifest=None, results=None, *extra):
        d = scratch_dir()
        mpath, rpath = write_tiny_inputs(d, manifest=manifest, results=results)
        out = os.path.join(d, "out")
        code, text = run_cli(["--manifest", mpath, "--results", rpath, "--outdir", out,
                              "--bootstrap-resamples", "50", *extra])
        return code, text, out

    def test_dropped_results_rows_are_printed_and_fuzzy_names_are_printed_and_kept(self):
        fake = [metric_row(d, m, 5.0) for d in ("ram5gb_cluster/H/short", "M-5/D/short")
                for m in ("TiRex", "PatchTST")]
        manifest = tiny_manifest([series_row("ett1_H", "ett1_H_item_5", omega_levels=0.1)])
        code, text, out = self.run_with(manifest, tiny_results(fake))
        self.assertEqual(code, 0)
        for part in ("[WARNING] held-out rule: dropped 2 results row(s) of dataset "
                     "'ram5gb_cluster/H/short' (token 'm5')", "false positive",
                     "possible held-out name in the results, NOT dropped: dataset 'M-5/D/short', "
                     "2 row(s)",
                     "possible held-out name in the manifest, NOT refused: series_id = "
                     "'ett1_H_item_5', 1 row(s), in a subset the analysis uses"):
            self.assertIn(part, text)
        with open(os.path.join(out, "run_manifest.json"), encoding="utf-8") as f:
            held = json.load(f)["held_out"]
        self.assertEqual(held["results_rows_excluded"], 2)
        self.assertEqual([e["dataset"] for e in held["results_exclusions"]],
                         ["ram5gb_cluster/H/short"])
        self.assertEqual([e["dataset"] for e in held["results_fuzzy_reports"]], ["M-5/D/short"])
        self.assertEqual([e["value"] for e in held["manifest_fuzzy_reports"]], ["ett1_H_item_5"])
        self.assertEqual(held["manifest_fields_checked"], ["series_id", "subset"])

    def test_refused_manifest_names_the_row_and_writes_nothing(self):
        bad = tiny_manifest([series_row("electricity_H", "m5_retail_item_0001")])
        code, _, out = self.run_with(bad)
        self.assertTrue(str(code).startswith("ERROR"))
        self.assertIn("series_id = 'm5_retail_item_0001'", str(code))
        self.assertFalse(os.path.exists(out))

    def test_manifest_list_reaches_the_results_and_the_run_manifest(self):
        fake = [metric_row("M-5_Retail/D/short", "TiRex", 5.0)]
        manifest = dict(tiny_manifest(), held_out_strict=LISTED)
        code, text, out = self.run_with(manifest, tiny_results(fake))
        self.assertEqual(code, 0)
        self.assertIn("dropped 1 results row(s) of dataset 'M-5_Retail/D/short' "
                      "(held_out_strict entry 'M5_Retail')", text)
        with open(os.path.join(out, "run_manifest.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["held_out"]["manifest_held_out_strict"], LISTED)


if __name__ == "__main__":
    unittest.main()
