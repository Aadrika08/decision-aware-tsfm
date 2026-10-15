"""The series scale (#29 item 7a): in-sample seasonal-naive MAE, season 24, of each series'
history before its first origin in the workload. Used by the engine, the representatives
and everything downstream. SYNTHETIC data only.

Set SCALE_REAL_WORKLOAD to a real workload.json (Dayson's handoff) to also check that every
series there gets a usable scale.
"""
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from decision_aware_tsfm.analysis.decision import representatives as rep
from decision_aware_tsfm.analysis.decision import run_pilot as rp
from decision_aware_tsfm.analysis.decision import scale as sc
from decision_aware_tsfm.analysis.decision.costs import assign_roles
from decision_aware_tsfm.analysis.decision.fixture import (
    FAMILY_OF_MODEL, FIT_ONLY_DATASETS, FIXTURE_SCHEMA, FIXTURE_STATIC_THRESHOLD, REPRESENTATIVES,
    ROSTER_FAMILY_OF_MODEL, first_origin, make_fixture_scales, make_roster_fixture, series_values)
from decision_aware_tsfm.analysis.decision.guards import HeldOutLeakError
from decision_aware_tsfm.analysis.decision.pipeline import ZERO_SCALE_REASON, build_tables, run
from decision_aware_tsfm.analysis.experiments import bootstrap as bs

from . import _shared, _sources
from .test_decision_pilot_run import ORIGINS, make_bundle, pilot_frame, quiet

KEY = ("d", "H", "d/s0")


def _with_flat_history(scales, series_id, status=sc.STATUS_ZERO):
    """The fixture scale table with one series' history made flat (scale 0)."""
    out = scales.copy()
    hit = out.series_id == series_id
    out.loc[hit, ["scale", "status", "n_pairs"]] = [np.nan, status, 0 if status == sc.STATUS_NO_PAIRS else 977]
    return out


class Statistic(unittest.TestCase):
    def test_hand_computed_value(self):
        y = np.arange(30, dtype=float)                       # y_t - y_{t-24} = 24 for t = 24..29
        self.assertEqual(sc.seasonal_naive_mae(y), (24.0, 6))
        y = np.r_[np.zeros(24), [1.0, -3.0, 2.0]]
        self.assertEqual(sc.seasonal_naive_mae(y), (2.0, 3))

    def test_only_pairs_with_both_values_finite_count(self):
        y = np.arange(30, dtype=float)
        y[1] = np.nan                                        # removes the pair (25, 1)
        y[28] = np.inf                                       # removes the pair (28, 4)
        self.assertEqual(sc.seasonal_naive_mae(y), (24.0, 4))
        s, n = sc.seasonal_naive_mae(np.full(30, np.nan))
        self.assertTrue(np.isnan(s))
        self.assertEqual(n, 0)

    def test_short_history_has_no_pair(self):
        s, n = sc.seasonal_naive_mae(np.arange(24.0))
        self.assertTrue(np.isnan(s))
        self.assertEqual(n, 0)

    def test_row_statuses(self):
        y = np.r_[np.arange(48.0), np.full(10, 99.0)]
        self.assertEqual(sc.scale_row(KEY, y, 48)["status"], sc.STATUS_OK)
        flat = sc.scale_row(KEY, np.full(60, 3.0), 48)
        self.assertEqual((flat["status"], flat["n_pairs"]), (sc.STATUS_ZERO, 24))
        self.assertTrue(np.isnan(flat["scale"]))
        short = sc.scale_row(KEY, y, 20)
        self.assertEqual((short["status"], short["n_pairs"]), (sc.STATUS_NO_PAIRS, 0))
        with self.assertRaises(sc.ScaleError):
            sc.scale_row(KEY, y, 100)


class OnlyHistoryBeforeTheFirstOrigin(unittest.TestCase):
    def test_values_from_the_first_origin_on_do_not_move_the_scale(self):
        y = series_values(7)[("fx_tie", "H", "fx_tie/s0")]
        h = first_origin()
        base = sc.scale_row(KEY, y, h)
        changed = y.copy()
        changed[h:] = np.random.default_rng(0).standard_normal(y.size - h) * 1e6   # targets and later
        self.assertEqual(sc.scale_row(KEY, changed, h), base)
        changed[h - 1] += 1.0                                # the last history point does move it
        self.assertNotEqual(sc.scale_row(KEY, changed, h)["scale"], base["scale"])

    def test_scrambled_targets_leave_the_engine_scale_unchanged(self):
        fc, omega, _ = _shared.fixture_data()
        a = _shared.tables()
        scrambled = fc.copy()
        scrambled["y"] = np.random.default_rng(1).permutation(scrambled["y"].to_numpy()) * 7.0
        # y must be the same across models for a target, so scramble per target position
        pos = scrambled["origin"] + scrambled["step"] - 1
        first = scrambled.groupby([scrambled.series_id, pos])["y"].transform("first")
        scrambled["y"] = first
        b = build_tables(scrambled, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega,
                         fit_only_datasets=FIT_ONLY_DATASETS, scales=_shared.scales())
        for name in ("val", "test", "fit_units"):
            ka = a[name].set_index(["series_id", "horizon", "tau"])["scale"]
            kb = b[name].set_index(["series_id", "horizon", "tau"])["scale"]
            pd.testing.assert_series_equal(ka.sort_index(), kb.sort_index())
        self.assertFalse(np.allclose(a["test"]["J_tsfm"], b["test"]["J_tsfm"]))

    def test_one_scale_per_series_for_every_role_horizon_and_tau(self):
        t = _shared.tables()
        sc_ = make_fixture_scales(7).set_index("series_id")["scale"]
        for name in ("val", "test", "fit_units"):
            f = t[name]
            self.assertTrue((f["scale"].to_numpy() == sc_.loc[f["series_id"]].to_numpy()).all(), name)
        po = t["per_origin"]
        self.assertTrue((po.groupby("series_id")["scale"].nunique() == 1).all())
        self.assertEqual(t["settings"]["scale"], sc.SCALE_KIND)

    def test_history_reaching_into_the_targets_is_refused(self):
        fc, omega, _ = _shared.fixture_data()
        late = _shared.scales().assign(history_end=first_origin() + 1)
        with self.assertRaisesRegex(sc.ScaleError, "ends after the series' first origin"):
            build_tables(fc, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega,
                         fit_only_datasets=FIT_ONLY_DATASETS, scales=late)
        with self.assertRaisesRegex(sc.ScaleError, "ends after the series' first origin"):
            rep.best_of_family(make_roster_fixture(7)[0], FIXTURE_SCHEMA, ROSTER_FAMILY_OF_MODEL,
                               scales=late)


class ZeroAndUndefinedScale(unittest.TestCase):
    def test_all_zero_horizon_1_targets_with_a_varying_history_are_scored(self):
        fc, omega, _ = _shared.fixture_data()
        fc = fc.copy()
        pos = fc.origin + fc.step - 1
        h1 = set((fc.loc[fc.horizon == 1, "origin"]).unique())
        fc.loc[(fc.series_id == "fx_tie/s1") & pos.isin(h1), "y"] = 0.0
        self.assertEqual(float(fc.loc[(fc.series_id == "fx_tie/s1") & (fc.horizon == 1), "y"].abs().max()), 0.0)
        t = build_tables(fc, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega,
                         fit_only_datasets=FIT_ONLY_DATASETS, scales=_shared.scales())
        self.assertEqual(len(t["excluded_series_horizons"]), 0)
        res, _ = run(t, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD)
        mine = res[(res.series_id == "fx_tie/s1") & (res.horizon == 1) & (res.status == "selected")]
        self.assertGreater(len(mine), 0)
        self.assertTrue(np.isfinite(mine.scaled_regret).all())

    def test_flat_history_drops_the_series_with_its_reason(self):
        fc, omega, _ = _shared.fixture_data()
        for status in (sc.STATUS_ZERO, sc.STATUS_NO_PAIRS):
            with self.subTest(status=status):
                t = build_tables(fc, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega,
                                 fit_only_datasets=FIT_ONLY_DATASETS,
                                 scales=_with_flat_history(_shared.scales(), "fx_tie/s1", status))
                ex = t["excluded_series_horizons"]
                self.assertEqual(sorted(ex.horizon), [1, 24])                # every horizon of the series
                self.assertTrue((ex.series_id == "fx_tie/s1").all())
                self.assertTrue(ex.reason.str.startswith(ZERO_SCALE_REASON).all())
                self.assertTrue(ex.reason.str.endswith(f"({status})").all())
                self.assertNotIn("fx_tie/s1", set(t["units"].series_id))
                self.assertTrue((t["test"].scale > 0).all())
                res, _ = run(t, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD)
                self.assertTrue(np.isfinite(res.loc[res.status == "selected", "scaled_regret"]).all())

    def test_flat_history_in_a_fit_only_dataset_is_dropped_with_its_reason(self):
        fc, omega, _ = _shared.fixture_data()
        t = build_tables(fc, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega,
                         fit_only_datasets=FIT_ONLY_DATASETS,
                         scales=_with_flat_history(_shared.scales(), "fx_dfit/s0"))
        ex = t["excluded_series_horizons"]
        self.assertEqual(set(ex.used_for), {"fit"})
        self.assertNotIn("fx_dfit/s0", set(t["fit_units"].series_id))

    def test_representatives_count_and_skip_a_series_without_a_usable_scale(self):
        fc, _ = make_roster_fixture(7)
        p = rep.best_of_family(fc, FIXTURE_SCHEMA, ROSTER_FAMILY_OF_MODEL,
                               scales=_with_flat_history(_shared.scales(), "fx_tie/s1"))
        self.assertEqual(p["coverage"]["classical"]["n_units_dropped_zero_scale"], 2)   # two horizons
        full = rep.best_of_family(fc, FIXTURE_SCHEMA, ROSTER_FAMILY_OF_MODEL, scales=_shared.scales())
        self.assertEqual(full["coverage"]["classical"]["n_units_dropped_zero_scale"], 0)
        self.assertEqual(p["coverage"]["classical"]["n_units_scored"],
                         full["coverage"]["classical"]["n_units_scored"] - 2)


class MissingOrBadScaleInput(unittest.TestCase):
    def test_no_scale_table_fails_loudly_everywhere(self):
        fc, omega, _ = _shared.fixture_data()
        with self.assertRaisesRegex(sc.ScaleError, "no scale table"):
            build_tables(fc, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega,
                         fit_only_datasets=FIT_ONLY_DATASETS)
        with self.assertRaisesRegex(sc.ScaleError, "no scale table"):
            rep.best_of_family(make_roster_fixture(7)[0], FIXTURE_SCHEMA, ROSTER_FAMILY_OF_MODEL)
        with self.assertRaisesRegex(sc.ScaleError, "no scale table"):
            rep.fixed_named(make_roster_fixture(7)[0], FIXTURE_SCHEMA, ROSTER_FAMILY_OF_MODEL,
                            REPRESENTATIVES)

    def test_a_series_without_a_row_fails(self):
        fc, omega, _ = _shared.fixture_data()
        with self.assertRaisesRegex(sc.ScaleError, "1 series have no row"):
            build_tables(fc, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega,
                         fit_only_datasets=FIT_ONLY_DATASETS, scales=_shared.scales().iloc[1:])

    def test_malformed_tables_fail(self):
        good = _shared.scales()
        cases = {
            "lacks columns": good.drop(columns="history_end"),
            "more than one row": pd.concat([good, good.iloc[:1]], ignore_index=True),
            "disagree": good.assign(status=np.where(good.index == 0, sc.STATUS_ZERO, good.status)),
            "non-negative integer": good.assign(history_end=-1),
            "numeric or null": good.assign(scale="ten"),
        }
        for pattern, table in cases.items():
            with self.subTest(pattern), self.assertRaisesRegex(sc.ScaleError, pattern):
                sc.check_scale_table(table)

    def test_a_table_without_status_marks_zero_as_not_usable(self):
        t = sc.check_scale_table(pd.DataFrame(dict(dataset=["a", "a"], config="H", series_id=["x", "y"],
                                                   scale=[2.0, 0.0], history_end=[10, 10])))
        self.assertEqual(t.status.tolist(), [sc.STATUS_OK, sc.STATUS_NOT_POSITIVE])
        self.assertTrue(np.isnan(t.scale.iloc[1]))


class SourcesFromTheWorkload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fc, _ = pilot_frame()
        self.fc = fc[fc.dataset == "fx_tie"]
        self.workload, self.hashes = _sources.write_workload(self.root, self.fc)

    def tearDown(self):
        self.tmp.cleanup()

    def _doc(self):
        return json.loads(self.workload.read_text(encoding="utf-8"))

    def _save(self, doc):
        self.workload.write_text(json.dumps(doc), encoding="utf-8")

    def test_reads_every_source_and_checks_its_hash(self):
        s = sc.read_workload_sources(self.workload)
        self.assertEqual(len(s.scales), 3)
        self.assertEqual(s.report["sources_hash_verified"], 3)
        self.assertTrue((s.scales.history_end == ORIGINS[0]).all())
        h = _sources.stepped_history(ORIGINS[0])
        self.assertTrue((s.scales.scale == float(np.mean(np.abs(h[24:] - h[:-24])))).all())
        self.assertEqual(sc.check_targets_match_sources(self.fc, s)["series"], 3)

    def test_missing_source_file_fails(self):
        (self.root / self._doc()["rows"][0]["source_path"]).unlink()
        with self.assertRaisesRegex(sc.ScaleError, "is missing.*--sources"):
            sc.read_workload_sources(self.workload)

    def test_changed_source_fails_its_hash(self):
        p = self.root / self._doc()["rows"][0]["source_path"]
        a = np.load(p)
        a[0] += 1.0
        np.save(p, a)
        with self.assertRaisesRegex(sc.ScaleError, "hash mismatch"):
            sc.read_workload_sources(self.workload)

    def test_workload_without_source_fields_fails(self):
        doc = self._doc()
        for r in doc["rows"]:
            r.pop("source_path")
        self._save(doc)
        with self.assertRaisesRegex(sc.ScaleError, "lack \\['source_path'\\]"):
            sc.read_workload_sources(self.workload)

    def test_rows_of_one_series_must_agree_on_their_source(self):
        doc = self._doc()
        doc["rows"][0]["usable_start"] = 5
        self._save(doc)
        with self.assertRaisesRegex(sc.ScaleError, "disagree"):
            sc.read_workload_sources(self.workload)

    def test_usable_start_opens_the_history(self):
        doc = self._doc()
        sid = doc["rows"][0]["series_id"]
        for r in doc["rows"]:
            if r["series_id"] == sid:
                r["usable_start"] = 300
        self._save(doc)
        s = sc.read_workload_sources(self.workload)
        row = s.scales[s.scales.series_id == sid].iloc[0]
        h = _sources.stepped_history(ORIGINS[0])[300:]
        self.assertEqual((row.history_start, row.n_pairs), (300, ORIGINS[0] - 300 - 24))
        self.assertEqual(row.scale, float(np.mean(np.abs(h[24:] - h[:-24]))))

    def test_paths_must_stay_inside_the_sources_folder(self):
        doc = self._doc()
        for r in doc["rows"]:
            r["source_path"] = "../" + r["source_path"]
        self._save(doc)
        with self.assertRaisesRegex(sc.ScaleError, "relative path inside"):
            sc.read_workload_sources(self.workload)

    def test_sources_folder_can_be_named(self):
        elsewhere = self.root / "elsewhere"
        (self.root / "sources").rename(self.root / "moved")
        elsewhere.mkdir()
        (self.root / "moved").rename(elsewhere / "sources")
        with self.assertRaises(sc.ScaleError):
            sc.read_workload_sources(self.workload)
        s = sc.read_workload_sources(self.workload, elsewhere)
        self.assertEqual(s.report["sources_folder"], "elsewhere")
        self.assertEqual(len(s.scales), 3)

    def test_held_out_names_stop_before_any_file_is_opened(self):
        for field, value in (("series_id", "fx_tie/pjm_s0"), ("source_path", "sources/m5_x.npy")):
            doc = self._doc()
            doc["rows"][0][field] = value
            self._save(doc)
            with self.subTest(field=field), mock.patch.object(sc.np, "load", side_effect=AssertionError):
                with self.assertRaises(HeldOutLeakError):
                    sc.read_workload_sources(self.workload)

    def test_targets_that_differ_from_the_source_fail(self):
        s = sc.read_workload_sources(self.workload)
        bad = self.fc.copy()
        bad.loc[bad.index[0], "y"] += 1.0
        with self.assertRaisesRegex(sc.ScaleError, "differ from the source"):
            sc.check_targets_match_sources(bad, s)


class PilotRunInputs(unittest.TestCase):
    """run_pilot stops, with nothing written, when the history cannot be read or checked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        fc, omega = pilot_frame()
        self.bundle = self.base / "bundle"
        make_bundle(self.bundle, fc, omega)
        self.out = self.base / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def assertStops(self, pattern, **kw):
        with self.assertRaisesRegex(sc.ScaleError, pattern):
            rp.run_pilot(self.bundle, self.out, n_boot=0, **kw)
        self.assertFalse(self.out.exists())
        self.assertFalse(self.out.with_name("out.partial").exists())

    def _first_source(self):
        doc = json.loads((self.bundle / "workload.json").read_text(encoding="utf-8"))
        return self.bundle / doc["rows"][0]["source_path"]

    def test_missing_source_stops_and_the_command_says_so(self):
        self._first_source().unlink()
        self.assertStops("is missing")
        with self.assertRaises(SystemExit) as cm:
            quiet(rp.main, [str(self.bundle), "--out", str(self.out)])
        self.assertIn("ScaleError", str(cm.exception))

    def test_changed_source_stops(self):
        p = self._first_source()
        a = np.load(p)
        a[:5] = 0.0
        np.save(p, a)
        self.assertStops("hash mismatch")

    def test_source_other_than_the_bundle_names_stops(self):
        doc = json.loads((self.bundle / "workload.json").read_text(encoding="utf-8"))
        p = self._first_source()
        a = np.load(p)
        a[:5] = 0.0
        np.save(p, a)
        for r in doc["rows"]:
            if r["source_path"] == doc["rows"][0]["source_path"]:
                r["source_sha256"] = _sources.sha256(p)
        (self.bundle / "workload.json").write_text(json.dumps(doc), encoding="utf-8")
        self.assertStops("forecast from a source other than")

    def test_sources_option_reaches_a_moved_folder(self):
        (self.bundle / "sources").rename(self.base / "sources")
        self.assertStops("is missing")
        m = quiet(rp.run_pilot, self.bundle, self.out, n_boot=0, sources=self.base)
        self.assertEqual(m["inputs"]["scale_history"]["sources_folder"], self.base.name)
        self.assertNotIn(str(self.base), json.dumps(m))
        out2 = self.base / "out_cli"
        quiet(rp.main, [str(self.bundle), "--out", str(out2), "--sources", str(self.base), "--n-boot", "0"])
        self.assertTrue((out2 / "engine" / "series_scales.csv").is_file())

    def test_bundle_values_that_differ_from_the_sources_stop(self):
        forecasts, omega, _, _ = rp.load_bundle(self.bundle, "current")
        src = sc.read_workload_sources(self.bundle / "workload.json")
        self.assertEqual(sc.check_bundle_against_sources(forecasts, omega, src)["series"], 15)
        bad = forecasts.copy()
        bad.loc[bad.index[0], "y_true"] += 0.5
        with self.assertRaisesRegex(sc.ScaleError, "differ from the source"):
            sc.check_bundle_against_sources(bad, omega, src)


class SameScaleEverywhere(unittest.TestCase):
    def test_representatives_score_with_the_engine_scale(self):
        fc, omega = make_roster_fixture(7)
        lab = fc.assign(family=fc.model.map(ROSTER_FAMILY_OF_MODEL))
        rows = assign_roles(lab, 1 / 3)
        u = rep.validation_unit_losses(rows[rows.role == "validation"], _shared.scales())
        t = build_tables(fc, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega,
                         fit_only_datasets=(), scales=_shared.scales())
        v = t["val"][np.isclose(t["val"].tau, 0.5)][["series_id", "horizon", "scale"]]
        m = u.merge(v, on=["series_id", "horizon"], suffixes=("", "_engine"))
        self.assertEqual(len(m), len(u))
        self.assertTrue((m.scale == m.scale_engine).all())

    def test_bootstrap_keeps_the_series_scale_on_every_resample(self):
        t = _shared.tables()
        cost = bs.family_cost_table(t["per_origin"], "test")
        u = bs._unit_means(bs.OriginPanel(cost).resample(np.random.default_rng(3)))
        s = make_fixture_scales(7).set_index("series_id")["scale"]
        self.assertTrue(np.isin(u.scale.to_numpy(), s.to_numpy()).all())
        bad = cost.copy()
        bad.loc[bad.index[0], "scale"] *= 2
        with self.assertRaisesRegex(ValueError, "one value per series"):
            bs._unit_means(bs.OriginPanel(bad).identity())

    def test_e3_scales_each_series_on_its_context(self):
        from decision_aware_tsfm.analysis.experiments.e3.scoring import context_scales
        rng = np.random.default_rng(5)
        values = {"a": rng.standard_normal(3000), "b": rng.standard_normal(3000)}
        rows = pd.DataFrame(dict(dataset="e3", config="H", series_id=["a", "b"], n_context=2000))
        t = context_scales(values, rows).set_index("series_id")
        for sid, y in values.items():
            h = y[:2000]
            self.assertEqual(t.loc[sid, "history_end"], 2000)
            self.assertEqual(t.loc[sid, "scale"], float(np.mean(np.abs(h[24:] - h[:-24]))))

    def test_robustness_floor_is_three(self):
        p = inspect.signature(rep.representative_robustness).parameters
        self.assertEqual(p["min_fit_datasets"].default, 3)


@unittest.skipUnless(os.environ.get("SCALE_REAL_WORKLOAD"),
                     "set SCALE_REAL_WORKLOAD to a real workload.json to check its scales")
class RealWorkload(unittest.TestCase):
    def test_every_series_has_a_usable_scale(self):
        s = sc.read_workload_sources(Path(os.environ["SCALE_REAL_WORKLOAD"]))
        self.assertTrue((s.scales.status == sc.STATUS_OK).all(), s.scales[s.scales.status != sc.STATUS_OK])


if __name__ == "__main__":
    unittest.main()
