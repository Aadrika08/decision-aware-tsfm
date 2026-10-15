"""Agreed development roles (2026-10-07): earliest third of origins for validation,
the rest for evaluation, leave-one-dataset-out for fitting, and a series only
qualifies if its validation third is not empty."""
import unittest

import numpy as np
import pandas as pd

from decision_aware_tsfm.analysis.decision.fixture import (
    FAMILY_OF_MODEL, FIT_ONLY_DATASETS, FIXTURE_SCHEMA, FIXTURE_STATIC_THRESHOLD, REPRESENTATIVES)
from decision_aware_tsfm.analysis.decision.guards import FitEvalOverlapError, HeldOutLeakError
from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID
from decision_aware_tsfm.analysis.decision.pipeline import (
    NO_VALIDATION_REASON, build_tables, fit_pool, run)
from decision_aware_tsfm.analysis.decision.schema import UNIT_KEYS, validate_selection_frame
from decision_aware_tsfm.analysis.decision.selectors import (
    DOmegaLinear, LeaveOneDatasetOut, NotFitted, SelectorInput)

from . import _shared

EVAL_DATASETS = ("fx_classical_wins", "fx_tau_flip", "fx_tie", "fx_tsfm_wins")


def _tables(fc=None, **kw):
    fc, omega, _ = (_shared.fixture_data() if fc is None else (fc, _shared.fixture_data()[1], None))
    return build_tables(fc, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega,
                        fit_only_datasets=FIT_ONLY_DATASETS, scales=_shared.scales(), **kw)


class LeaveOneDatasetOutFitting(unittest.TestCase):
    def setUp(self):
        self.res, self.d = _shared.results()
        self.tables = _shared.tables()

    def test_default_is_one_fold_per_evaluation_dataset(self):
        self.assertIsInstance(self.d, LeaveOneDatasetOut)
        self.assertEqual(sorted(self.d.folds), list(EVAL_DATASETS))

    def test_no_fold_sees_its_own_dataset_and_each_sees_all_others(self):
        for ds, fold in self.d.folds.items():
            self.assertNotIn(ds, fold.fitted_datasets)
            want = sorted((set(EVAL_DATASETS) - {ds}) | set(FIT_ONLY_DATASETS))
            self.assertEqual(sorted(fold.fitted_datasets), want, ds)
            own = set(self.tables["units"].loc[self.tables["units"].dataset == ds, "series_id"])
            self.assertFalse(own & fold.fitted_series, ds)

    def test_choices_equal_an_independent_refit_per_dataset(self):
        """Recompute every D_Omega choice by hand: fit on the pool without d,
        apply the threshold rule to d's units."""
        pool = fit_pool(self.tables).dropna(subset=["omega"])
        units = self.tables["units"]
        got = self.res[self.res.selector == "d_omega"].set_index(UNIT_KEYS)["chosen_family"]
        checked = 0
        for ds in EVAL_DATASETS:
            f = DOmegaLinear.fit(pool[pool.dataset != ds])
            u = units[units.dataset == ds]
            bound = f.omega0 + f.beta * np.abs(u.tau - 0.5)
            want = np.where(u.omega > bound, "tsfm", "classical")
            have = got.loc[pd.MultiIndex.from_frame(u[UNIT_KEYS])].to_numpy()
            self.assertTrue((have == want).all(), ds)
            checked += len(u)
        self.assertEqual(checked, len(units))

    def test_note_names_the_fold(self):
        d = self.res[self.res.selector == "d_omega"]
        self.assertTrue((d.note.str.split().str[0] == "fold=" + d.dataset).all())

    def test_pool_holds_fit_only_and_evaluation_test_units(self):
        pool = fit_pool(self.tables)
        self.assertEqual(len(pool), len(self.tables["fit_units"]) + len(self.tables["test"]))
        ev = pool[pool.dataset.isin(EVAL_DATASETS)].merge(self.tables["test"], on=UNIT_KEYS,
                                                           suffixes=("", "_t"))
        self.assertTrue(np.allclose(ev.J_classical, ev.J_classical_t))
        self.assertTrue(np.allclose(ev.J_tsfm, ev.J_tsfm_t))

    def test_too_few_other_datasets_defers_with_reason(self):
        res, d = run(self.tables, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD, min_fit_datasets=5)
        self.assertTrue(all(isinstance(f, NotFitted) for f in d.folds.values()))
        dm = res[res.selector == "d_omega"]
        self.assertTrue((dm.status == "insufficient_data").all())
        self.assertTrue(dm.note.str.contains("4 other dataset").all())
        self.assertTrue((res[res.selector != "d_omega"].status == "selected").all())

    def test_default_minimum_is_three_other_datasets(self):
        """Record 0007 section 2: three other datasets fit a fold, two do not."""
        pool = fit_pool(self.tables).dropna(subset=["omega"])
        four = pool[pool.dataset.isin(EVAL_DATASETS)]
        d = LeaveOneDatasetOut.build(four, EVAL_DATASETS)
        self.assertEqual(d.min_fit_datasets, 3)
        self.assertFalse(any(isinstance(f, NotFitted) for f in d.folds.values()))
        for ds, fold in d.folds.items():
            self.assertEqual(len(fold.fitted_datasets), 3, ds)
        three = four[four.dataset != EVAL_DATASETS[-1]]
        d = LeaveOneDatasetOut.build(three, EVAL_DATASETS[:-1])
        self.assertTrue(all(isinstance(f, NotFitted) for f in d.folds.values()))
        for ds, fold in d.folds.items():
            self.assertIn("2 other dataset(s) with usable units, need 3",
                          fold.describe()["not_fitted"], ds)

    def test_run_uses_the_same_default_minimum(self):
        _, d = run(self.tables, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD)
        self.assertEqual(d.min_fit_datasets, 3)
        self.assertFalse(any(isinstance(f, NotFitted) for f in d.folds.values()))

    def test_a_fold_fitted_on_its_own_dataset_is_refused(self):
        units = self.tables["units"]
        leaky = DOmegaLinear.fit(fit_pool(self.tables))              # includes every dataset
        guard = LeaveOneDatasetOut(folds={ds: leaky for ds in EVAL_DATASETS}, min_fit_datasets=2)
        with self.assertRaises(FitEvalOverlapError):
            guard.select(SelectorInput(units=units, val=self.tables["val"]))

    def test_own_dataset_guard_does_not_rely_on_the_selector(self):
        """A replacement selector (e.g. the real D_Omega) may not check series
        overlap itself; the fold wrapper must still refuse a self-fitted fold."""
        class NoCheck:
            fitted_datasets = ("fx_tie",)

            def select(self, inp):
                raise AssertionError("must not be reached")

        guard = LeaveOneDatasetOut(folds={"fx_tie": NoCheck()}, min_fit_datasets=1)
        units = self.tables["units"]
        with self.assertRaises(FitEvalOverlapError):
            guard.select(SelectorInput(units=units[units.dataset == "fx_tie"], val=self.tables["val"]))

    def test_held_out_dataset_in_the_pool_is_refused(self):
        pool = fit_pool(self.tables)
        pool.loc[pool.dataset == "fx_dfit", "dataset"] = "PJM"
        with self.assertRaises(HeldOutLeakError):
            LeaveOneDatasetOut.build(pool, EVAL_DATASETS)

    def test_output_keeps_the_order_of_the_units(self):
        units = self.tables["units"].sample(frac=1.0, random_state=3)
        out = self.d.select(SelectorInput(units=units, val=self.tables["val"]))
        self.assertTrue((out[UNIT_KEYS].to_numpy() == units[UNIT_KEYS].to_numpy()).all())

    def test_old_scheme_still_available_and_unknown_scheme_refused(self):
        _, d = run(self.tables, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD, fit_scheme="fit_only_datasets")
        self.assertIsInstance(d, DOmegaLinear)
        self.assertEqual(d.fitted_datasets, FIT_ONLY_DATASETS)
        with self.assertRaisesRegex(ValueError, "fit_scheme"):
            run(self.tables, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD, fit_scheme="pooled")

    def test_describe_lists_every_fold(self):
        desc = self.d.describe()
        self.assertEqual(desc["scheme"], "leave_one_dataset_out")
        self.assertEqual(sorted(desc["folds"]), list(EVAL_DATASETS))


class EmptyValidationThird(unittest.TestCase):
    SID = "fx_tsfm_wins/s2"

    def _short(self, horizon):
        """Keep only the two earliest origins of one series at one horizon:
        floor(2/3) = 0 validation origins."""
        fc = _shared.fixture_data()[0]
        g = (fc.series_id == self.SID) & (fc.horizon == horizon)
        keep = sorted(fc.loc[g, "origin"].unique())[:2]
        return fc[~g | fc.origin.isin(keep)].copy()

    def test_series_without_validation_is_excluded_with_reason(self):
        t = _tables(self._short(24))
        ex = t["excluded_series_horizons"]
        self.assertEqual(len(ex), 1)
        row = ex.iloc[0]
        self.assertEqual((row.series_id, int(row.horizon), row.reason, row.used_for),
                         (self.SID, 24, NO_VALIDATION_REASON, "evaluation"))
        for name in ("units", "val", "test", "per_origin"):
            sub = t[name]
            self.assertFalse(((sub.series_id == self.SID) & (sub.horizon == 24)).any(), name)
        # the other horizon of the same series is unaffected
        self.assertTrue(((t["units"].series_id == self.SID) & (t["units"].horizon == 1)).any())

    def test_after_exclusion_validation_comparators_score_every_unit(self):
        res, _ = run(_tables(self._short(24)), REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD)
        validate_selection_frame(res)
        self.assertTrue((res.status == "selected").all())
        self.assertEqual(len(res), 7 * (len(_shared.tables()["units"]) - len(TAU_GRID)))

    def test_opting_out_keeps_the_series_and_validation_defers(self):
        t = _tables(self._short(24), exclude_without_validation=False)
        self.assertEqual(len(t["excluded_series_horizons"]), 0)
        self.assertFalse(t["settings"]["exclude_without_validation"])
        res, _ = run(t, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD)
        mine = res[(res.series_id == self.SID) & (res.horizon == 24)]
        for name in ("pinball_validation", "crps_validation"):
            self.assertTrue((mine[mine.selector == name].status == "insufficient_data").all(), name)

    def test_zero_val_fraction_excludes_nothing_for_this_reason(self):
        t = _tables(val_fraction=0.0)
        self.assertFalse(t["settings"]["exclude_without_validation"])
        self.assertEqual(len(t["excluded_series_horizons"]), 0)
        self.assertEqual(len(t["val"]), 0)

    def test_default_records_the_rule_and_excludes_nothing_on_the_fixture(self):
        t = _shared.tables()
        self.assertTrue(t["settings"]["exclude_without_validation"])
        self.assertEqual(len(t["excluded_series_horizons"]), 0)


if __name__ == "__main__":
    unittest.main()
