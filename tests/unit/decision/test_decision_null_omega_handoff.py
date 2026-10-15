"""Null Omega in a handoff bundle: a row that is present with a reason is not a
missing join (review of #44, 2026-10-10)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from decision_aware_tsfm.analysis.decision import handoff
from decision_aware_tsfm.analysis.decision.fixture import (
    FAMILY_OF_MODEL, FIT_ONLY_DATASETS, FIXTURE_SCHEMA, FIXTURE_STATIC_THRESHOLD, PRODUCER_SCHEMA,
    REPRESENTATIVES, to_producer_format)
from decision_aware_tsfm.analysis.decision.handoff import HandoffError, read_handoff
from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID
from decision_aware_tsfm.analysis.decision.pipeline import OmegaJoinError, build_tables, run
from decision_aware_tsfm.analysis.decision.schema import validate_selection_frame

from . import _shared
from .test_decision_producer import _sha, quiet_check, write_bundle

# the producer's vocabulary: omega.compute_omega_checked (data owner) and cache.readiness (#56)
REASONS = ("empty_history", "below_floor", "flat_history", "nonfinite")
SERIES = ("fx_tsfm_wins/s0", "fx_tsfm_wins/s1", "fx_classical_wins/s0", "fx_classical_wins/s1",
          "fx_dfit/s0", "fx_dfit/s6")
ORIGINS = (1001, 1025, 1049)     # horizon 24: one validation origin, then two test origins
SID = "fx_tsfm_wins/s0"
DECISION_ORIGIN = 1025           # first test origin: the Omega row the evaluation decision uses
OMEGA_BASED = ("static_omega", "d_omega")
OTHERS = ("always_classical", "always_tsfm", "pinball_validation", "crps_validation", "hindsight_oracle")
# The bundle below has three datasets, so a fold has two others: below the default floor of 3.
# These tests are about null Omega, so they ask for 2 and keep D_Omega fitted.
MIN_FIT_DATASETS = 2
CHECK_ARGS = ["--variant", "current", "--classical", "AutoETS", "--tsfm", "Moirai-small",
              "--static-threshold", "0.45", "--fit-only-datasets", "fx_dfit",
              "--min-fit-datasets", str(MIN_FIT_DATASETS)]


def _frame():
    """Two evaluation datasets and the fit-only one, two series each, representatives only."""
    fc = _shared.fixture_data()[0]
    keep = ((fc.horizon == 24) & fc.model.isin(list(REPRESENTATIVES.values()))
            & fc.series_id.isin(SERIES) & fc.origin.isin(ORIGINS))
    return to_producer_format(fc[keep])


def edit_sidecar(root, change):
    """Apply change(rows) to omega.json and record the new sidecar hash in handoff.json."""
    sidecar = json.loads((root / "omega.json").read_text(encoding="utf-8"))
    change(sidecar["rows"])
    (root / "omega.json").write_text(json.dumps(sidecar), encoding="utf-8")
    manifest = json.loads((root / "handoff.json").read_text(encoding="utf-8"))
    manifest["omega_sha256"] = _sha(root / "omega.json")
    (root / "handoff.json").write_text(json.dumps(manifest), encoding="utf-8")


def _row(rows, series_id, origin, variant):
    hit = [r for r in rows if (r["series_id"], r["origin"], r["variant"]) == (series_id, origin, variant)]
    assert len(hit) == 1, hit
    return hit[0]


def set_row(series_id=SID, origin=DECISION_ORIGIN, variant="current", **fields):
    return lambda rows: _row(rows, series_id, origin, variant).update(fields)


def drop_row(series_id=SID, origin=DECISION_ORIGIN, variant="current"):
    return lambda rows: rows.remove(_row(rows, series_id, origin, variant))


def _engine(root, variant="current"):
    fc, omega, rep = read_handoff(root, variant)
    t = build_tables(fc, PRODUCER_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega,
                     fit_only_datasets=FIT_ONLY_DATASETS, omega_col="omega", omega_variant=variant)
    res, d = run(t, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD, min_fit_datasets=MIN_FIT_DATASETS)
    return t, res, d, rep


class NullOmegaInBundle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "bundle"
        write_bundle(self.root, _frame())

    def tearDown(self):
        self.tmp.cleanup()

    def test_recognised_reasons_are_the_producers(self):
        self.assertEqual(tuple(handoff.UNDEFINED_OMEGA_REASONS), REASONS)

    def test_review_case_present_null_with_reason_is_read(self):
        # the case from the #44 review: this raised "1 forecast (series, origin) pairs have no Omega row"
        edit_sidecar(self.root, set_row(omega=None, omega_status="flat_history"))
        fc, omega, rep = read_handoff(self.root, "current")
        self.assertEqual(rep["partitions_hash_verified"], rep["partitions_listed"])
        row = omega[(omega.series_id == SID) & (omega.origin == DECISION_ORIGIN) & (omega.variant == "current")]
        self.assertEqual(len(row), 1)
        self.assertTrue(np.isnan(row.omega.iloc[0]))                 # undefined, not 0.0
        self.assertEqual(row.omega_status.iloc[0], "flat_history")   # the reason survives the read

    def test_every_recognised_reason_is_kept_and_counted(self):
        for reason in REASONS:
            with self.subTest(reason=reason):
                edit_sidecar(self.root, set_row(omega=None, omega_status=reason))
                _, omega, rep = read_handoff(self.root, "current")
                self.assertEqual(omega.loc[omega.omega.isna(), "omega_status"].tolist(), [reason])
                self.assertEqual(rep["omega_undefined_rows_for_variant"], 1)
                self.assertEqual(rep["omega_undefined_reasons_for_variant"], {reason: 1})
                self.assertEqual(rep["omega_rows_for_variant"], len(SERIES) * len(ORIGINS))

    def test_counts_are_for_the_variant_that_was_asked_for(self):
        edit_sidecar(self.root, set_row(omega=None, omega_status="below_floor"))
        t, res, _, rep = _engine(self.root, "last_4096")
        self.assertEqual(rep["omega_undefined_rows_for_variant"], 0)
        self.assertEqual(rep["omega_undefined_reasons_for_variant"], {})
        self.assertEqual(t["n_units_undefined_omega"], 0)
        self.assertEqual(len(t["undefined_omega"]), 0)
        self.assertTrue((res.status == "selected").all())

    def test_null_reaches_the_engine_as_undefined_and_only_omega_selectors_defer(self):
        edit_sidecar(self.root, set_row(omega=None, omega_status="flat_history"))
        t, res, _, _ = _engine(self.root)
        validate_selection_frame(res)
        self.assertEqual(t["n_units_undefined_omega"], len(TAU_GRID))          # one horizon x 11 taus
        self.assertTrue(t["units"].loc[t["units"].series_id == SID, "omega"].isna().all())
        mine = res[res.series_id == SID]
        self.assertEqual(len(mine), 7 * len(TAU_GRID))
        self.assertTrue(mine.omega.isna().all())                               # NaN, never 0.0
        for name in OMEGA_BASED:
            self.assertTrue((mine[mine.selector == name].status == "insufficient_data").all(), name)
        for name in OTHERS:
            self.assertTrue((mine[mine.selector == name].status == "selected").all(), name)
        rest = res[res.series_id != SID]
        self.assertTrue((rest.status == "selected").all())                     # nobody else is affected
        self.assertTrue((rest.omega == 0.5).all())

    def test_deferred_decision_is_listed_with_its_reason(self):
        edit_sidecar(self.root, set_row(omega=None, omega_status="empty_history"))
        t, _, _, _ = _engine(self.root)
        self.assertEqual(t["undefined_omega"].to_dict("records"), [dict(
            dataset="fx_tsfm_wins", config="H", series_id=SID, horizon=24,
            decision_origin=DECISION_ORIGIN, reason="empty_history", used_for="evaluation")])

    def test_null_in_fit_data_is_listed_and_left_out_of_the_fit(self):
        # fit-only datasets decide at their first origin
        edit_sidecar(self.root, set_row("fx_dfit/s0", ORIGINS[0], omega=None, omega_status="below_floor"))
        t, res, d, _ = _engine(self.root)
        self.assertEqual(t["undefined_omega"].to_dict("records"), [dict(
            dataset="fx_dfit", config="H", series_id="fx_dfit/s0", horizon=24,
            decision_origin=ORIGINS[0], reason="below_floor", used_for="fit")])
        self.assertNotIn("fx_dfit/s0", d.fitted_series)
        self.assertIn("fx_dfit/s6", d.fitted_series)
        self.assertEqual(t["n_units_undefined_omega"], 0)                      # no evaluation unit defers
        self.assertTrue((res.status == "selected").all())

    def test_null_away_from_the_decision_origin_is_counted_but_defers_nothing(self):
        edit_sidecar(self.root, set_row(origin=ORIGINS[0], omega=None, omega_status="flat_history"))
        t, res, _, rep = _engine(self.root)
        self.assertEqual(rep["omega_undefined_reasons_for_variant"], {"flat_history": 1})
        self.assertEqual(len(t["undefined_omega"]), 0)
        self.assertTrue((res.status == "selected").all())

    def test_null_without_a_recognised_reason_is_refused(self):
        absent = object()
        for status in (absent, None, "", "ok", "invented", "FLAT_HISTORY", 0, ["flat_history"]):
            with self.subTest(omega_status="<absent>" if status is absent else status):
                fields = {} if status is absent else dict(omega_status=status)
                edit_sidecar(self.root, set_row(omega=None, **fields))
                with self.assertRaisesRegex(HandoffError, "null Omega without a recognised omega_status"):
                    read_handoff(self.root, "current")

    def test_nan_literal_is_treated_as_null(self):
        edit_sidecar(self.root, set_row(omega=float("nan")))            # written as NaN, not null
        with self.assertRaisesRegex(HandoffError, "null Omega without a recognised omega_status"):
            read_handoff(self.root, "current")
        edit_sidecar(self.root, set_row(omega=float("nan"), omega_status="nonfinite"))
        self.assertEqual(read_handoff(self.root, "current")[2]["omega_undefined_reasons_for_variant"],
                         {"nonfinite": 1})

    def test_null_without_reason_is_refused_in_any_variant_and_at_any_origin(self):
        edit_sidecar(self.root, set_row("fx_classical_wins/s1", ORIGINS[2], "last_4096", omega=None))
        with self.assertRaisesRegex(HandoffError, "null Omega without a recognised omega_status"):
            read_handoff(self.root, "current")

    def test_absent_sidecar_row_is_still_refused(self):
        for origin in ORIGINS:                       # at the decision origin or anywhere else
            with self.subTest(origin=origin):
                write_bundle(self.root, _frame())
                edit_sidecar(self.root, drop_row(origin=origin))
                with self.assertRaisesRegex(HandoffError, r"1 forecast \(series, origin\) pairs have no Omega row"):
                    read_handoff(self.root, "current")

    def test_absent_row_is_told_apart_from_present_null(self):
        def change(rows):
            set_row(omega=None, omega_status="flat_history")(rows)
            set_row("fx_tsfm_wins/s1", omega=None, omega_status="nonfinite")(rows)
            drop_row("fx_classical_wins/s0")(rows)
        edit_sidecar(self.root, change)
        # two present nulls and one absent row: exactly one pair has no row
        with self.assertRaisesRegex(HandoffError, r"^1 forecast \(series, origin\) pairs have no Omega row"):
            read_handoff(self.root, "current")

    def test_defined_omega_with_an_undefined_reason_is_refused(self):
        for status in REASONS + ("invented", "", 0):
            with self.subTest(omega_status=status):
                edit_sidecar(self.root, set_row(omega=0.5, omega_status=status))
                with self.assertRaisesRegex(HandoffError, "defined Omega but omega_status"):
                    read_handoff(self.root, "current")

    def test_defined_omega_marked_ok_or_unmarked_is_read(self):
        def change(rows):
            for r in rows:
                if r["series_id"] != SID:
                    r["omega_status"] = "ok"         # the data owner's exporter writes this on every row
        edit_sidecar(self.root, change)
        t, res, _, rep = _engine(self.root)
        self.assertEqual(rep["omega_undefined_rows_for_variant"], 0)
        self.assertEqual(rep["omega_undefined_reasons_for_variant"], {})
        self.assertEqual(len(t["undefined_omega"]), 0)
        self.assertTrue((res.status == "selected").all())

    def test_bundle_without_any_status_still_has_the_status_column(self):
        _, omega, rep = read_handoff(self.root, "current")
        self.assertIn("omega_status", omega.columns)
        self.assertTrue(omega.omega_status.isna().all())
        self.assertEqual(rep["omega_undefined_rows_for_variant"], 0)

    def test_a_reason_does_not_excuse_the_other_sidecar_checks(self):
        cases = {
            "history after origin": (dict(history_end=DECISION_ORIGIN + 1), "after its origin"),
            "other source": (dict(source_sha256="0" * 64), "different source"),
        }
        for name, (fields, pattern) in cases.items():
            with self.subTest(name):
                write_bundle(self.root, _frame())
                edit_sidecar(self.root, set_row(omega=None, omega_status="flat_history", **fields))
                with self.assertRaisesRegex(HandoffError, pattern):
                    read_handoff(self.root, "current")

    def test_zero_is_not_an_undefined_placeholder(self):
        edit_sidecar(self.root, set_row(omega=0.0))
        fc, omega, _ = read_handoff(self.root, "current")
        with self.assertRaisesRegex(OmegaJoinError, "exactly 0.0"):
            build_tables(fc, PRODUCER_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega,
                         fit_only_datasets=FIT_ONLY_DATASETS, omega_col="omega", omega_variant="current")

    def test_check_command_reports_the_deferral_and_its_reason(self):
        edit_sidecar(self.root, set_row(omega=None, omega_status="flat_history"))
        out = Path(self.tmp.name) / "out"
        quiet_check([str(self.root)] + CHECK_ARGS + ["--out", str(out)])
        rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["omega_undefined_rows_for_variant"], 1)
        self.assertEqual(rep["omega_undefined_reasons_for_variant"], {"flat_history": 1})
        self.assertEqual(rep["undefined_omega_decisions"], {"evaluation/flat_history": 1})
        n = len(TAU_GRID)
        for name in OMEGA_BASED:
            self.assertEqual(rep["status_counts"][f"{name}/insufficient_data"], n, name)
            self.assertEqual(rep["status_counts"][f"{name}/selected"], 3 * n, name)
        for name in OTHERS:
            self.assertEqual(rep["status_counts"][f"{name}/selected"], 4 * n, name)
            self.assertNotIn(f"{name}/insufficient_data", rep["status_counts"])
        self.assertEqual(rep["column_check"]["unmatched"], 0)

    def test_check_command_report_on_a_bundle_without_nulls(self):
        out = Path(self.tmp.name) / "out"
        quiet_check([str(self.root)] + CHECK_ARGS + ["--out", str(out)])
        rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["omega_undefined_rows_for_variant"], 0)
        self.assertEqual(rep["undefined_omega_decisions"], {})

    def test_check_command_default_floor_leaves_a_three_dataset_bundle_unfitted(self):
        """Without --min-fit-datasets the command uses the recorded floor of 3 other datasets."""
        out = Path(self.tmp.name) / "out"
        args = [a for a in CHECK_ARGS if a not in ("--min-fit-datasets", str(MIN_FIT_DATASETS))]
        quiet_check([str(self.root)] + args + ["--out", str(out)])
        rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["d_omega"]["min_fit_datasets"], 3)
        for fold in rep["d_omega"]["folds"].values():
            self.assertIn("2 other dataset(s) with usable units, need 3", fold["not_fitted"])
        n = len(TAU_GRID)
        self.assertEqual(rep["status_counts"]["d_omega/insufficient_data"], 4 * n)
        self.assertEqual(rep["status_counts"]["static_omega/selected"], 4 * n)


class ReasonFromAPerSeriesOmegaTable(unittest.TestCase):
    """The engine without a bundle: the reason column is <omega_col>_status, as in the frozen manifest."""
    def _tables(self, omega):
        return build_tables(_shared.fixture_data()[0], FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL,
                            omega, fit_only_datasets=FIT_ONLY_DATASETS)

    def _omega(self, with_status):
        om = _shared.fixture_data()[1].copy()
        om.loc[om.series_id.isin([SID, "fx_dfit/s0"]), "omega_levels"] = np.nan
        if with_status:
            om["omega_levels_status"] = np.where(om.omega_levels.isna(), "flat_history", "ok")
        return om

    def test_reason_is_listed_per_deferred_decision(self):
        u = self._tables(self._omega(with_status=True))["undefined_omega"]
        self.assertEqual(sorted(zip(u.series_id, u.horizon, u.reason, u.used_for)),
                         [("fx_dfit/s0", 1, "flat_history", "fit"), ("fx_dfit/s0", 24, "flat_history", "fit"),
                          (SID, 1, "flat_history", "evaluation"), (SID, 24, "flat_history", "evaluation")])

    def test_decision_is_listed_without_a_reason_when_the_table_gives_none(self):
        t = self._tables(self._omega(with_status=False))
        u = t["undefined_omega"]
        self.assertEqual(sorted(zip(u.series_id, u.horizon, u.used_for)),
                         [("fx_dfit/s0", 1, "fit"), ("fx_dfit/s0", 24, "fit"),
                          (SID, 1, "evaluation"), (SID, 24, "evaluation")])
        self.assertTrue(u.reason.isna().all())
        self.assertEqual(t["n_units_undefined_omega"], 2 * len(TAU_GRID))

    def test_status_column_does_not_reach_the_selector_tables(self):
        t = self._tables(self._omega(with_status=True))
        ref = _shared.tables()
        for name in ("units", "fit_units", "decision_origins"):
            self.assertEqual(list(t[name].columns), list(ref[name].columns), name)

    def test_nothing_is_listed_on_the_ordinary_fixture(self):
        u = _shared.tables()["undefined_omega"]
        self.assertEqual(len(u), 0)
        self.assertEqual(list(u.columns), ["dataset", "config", "series_id", "horizon",
                                           "decision_origin", "reason", "used_for"])


if __name__ == "__main__":
    unittest.main()
