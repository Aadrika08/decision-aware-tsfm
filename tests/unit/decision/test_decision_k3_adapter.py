"""The K3 Omega-plus-tau selector (#26, analysis/k3/selector.py) in the D_Omega slot,
through decision.k3_adapter.

Synthetic fixture and hand-built tables only. No real cache, no M5 or PJM.
The classes that call the K3 selector need analysis/k3/selector.py and
scikit-learn; without them they are skipped and say so."""
import contextlib
import copy
import functools
import importlib
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import decision_aware_tsfm.analysis.decision as pkg
from decision_aware_tsfm.analysis.decision import k3_adapter, run_fixture
from decision_aware_tsfm.analysis.decision.costs import SERIES_KEYS
from decision_aware_tsfm.analysis.decision.evaluate import run_selectors, summarize
from decision_aware_tsfm.analysis.decision.fixture import (
    FAMILY_OF_MODEL, FIT_ONLY_DATASETS, FIXTURE_SCHEMA, FIXTURE_STATIC_THRESHOLD as THRESHOLD,
    REPRESENTATIVES)
from decision_aware_tsfm.analysis.decision.guards import FitEvalOverlapError, HeldOutLeakError
from decision_aware_tsfm.analysis.decision.k3_adapter import (
    DEFAULT_COST, DEFAULT_MARGIN, DEFAULT_MIN_DISTINCT, K3_SELECTOR_MODULE, K3SelectorAdapter,
    K3SelectorUnavailable, reason_of)
from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID
from decision_aware_tsfm.analysis.decision.pipeline import build_tables, fit_pool, run, seven_selectors
from decision_aware_tsfm.analysis.decision.schema import UNIT_KEYS, validate_selection_frame
from decision_aware_tsfm.analysis.decision.selectors import (
    FIT_POOL_COLUMNS, DOmegaLinear, LeaveOneDatasetOut, NotFitted, SelectorInput)

from . import _shared

try:
    k3 = importlib.import_module(K3_SELECTOR_MODULE)
    NO_K3 = ""
except ImportError as e:
    k3 = None
    NO_K3 = (f"K3 selector not importable here ({e}); these tests need "
             "analysis/k3/selector.py (PR #54, #32) and scikit-learn (PR #33)")
needs_k3 = unittest.skipIf(NO_K3, NO_K3)

EVAL_DATASETS = ("fx_classical_wins", "fx_tau_flip", "fx_tie", "fx_tsfm_wins")
SEVEN = {"always_classical", "always_tsfm", "static_omega", "pinball_validation",
         "crps_validation", "d_omega", "hindsight_oracle"}
POOL_OMEGAS = (0.125, 0.25, 0.75, 0.875)       # exact in binary, so the fitted threshold is exactly 0.5


def _keys(dataset, omegas, taus=(0.5, 0.91), horizon=1):
    return pd.DataFrame([dict(dataset=dataset, config="c", series_id=f"{dataset}/s{i}", horizon=horizon,
                              tau=t, omega=om)
                         for i, om in enumerate(omegas) for t in taus])


def _pool(datasets=("fa", "fb", "fc"), omegas=POOL_OMEGAS, horizon=1):
    """Hand-built fit pool with a planted boundary at omega 0.5. At tau 0.5
    classical is cheaper below it and TSFM above; at tau 0.91 the reverse.
    Costs are 1 (cheaper family) and 2, scale 1."""
    p = pd.concat([_keys(d, omegas, horizon=horizon) for d in datasets], ignore_index=True)
    classical_cheaper = np.where(p.tau == 0.5, p.omega <= 0.5, p.omega > 0.5)
    p["J_classical"] = np.where(classical_cheaper, 1.0, 2.0)
    p["J_tsfm"] = np.where(classical_cheaper, 2.0, 1.0)
    p["scale"] = 1.0
    return p[FIT_POOL_COLUMNS]


def _eval_tables(units):
    """Engine tables for hand-built units: classical costs 1 and TSFM 2 everywhere."""
    keys = units[UNIT_KEYS]
    return dict(units=units,
                val=keys.assign(J_classical=1.0, J_tsfm=1.0, crps_classical=1.0, crps_tsfm=1.0),
                test=keys.assign(J_classical=1.0, J_tsfm=2.0, scale=1.0, n_origins=4),
                n_val=keys[SERIES_KEYS].drop_duplicates().assign(n_validation_origins=2))


def _select(selector, units):
    return selector.select(SelectorInput(units=units, val=_eval_tables(units)["val"]))


def _fixture_tables(omega=None, fc=None, **kw):
    fc0, omega0, _ = _shared.fixture_data()
    kw.setdefault("fit_only_datasets", FIT_ONLY_DATASETS)
    return build_tables(fc0 if fc is None else fc, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL,
                        omega0 if omega is None else omega, **kw)


@functools.lru_cache(maxsize=None)
def _k3_results():
    return run(_shared.tables(), REPRESENTATIVES, THRESHOLD, d_omega_fit=K3SelectorAdapter.fit)


@functools.lru_cache(maxsize=None)
def _tables_without_fit_only():
    """The four evaluation datasets alone: every fold has exactly 3 other datasets."""
    fc = _shared.fixture_data()[0]
    return _fixture_tables(fc=fc[fc.dataset != "fx_dfit"], fit_only_datasets=())


def _d_omega(res):
    return res[res.selector == "d_omega"].reset_index(drop=True)


class WithoutTheK3Module(unittest.TestCase):
    """Runs in every environment, with or without scikit-learn."""

    def test_importing_the_decision_package_loads_neither_k3_nor_scikit_learn(self):
        pkg_dir = Path(pkg.__file__).parent
        mods = [f"decision_aware_tsfm.analysis.decision.{f.stem}" for f in pkg_dir.glob("*.py")
                if f.stem != "__init__"]
        self.assertIn("decision_aware_tsfm.analysis.decision.k3_adapter", mods)
        code = ("import sys, importlib\n"
                f"for m in {mods!r}: importlib.import_module(m)\n"
                "print(','.join(sorted(m for m in sys.modules if m.split('.')[0] == 'sklearn'\n"
                "                      or m.startswith('decision_aware_tsfm.analysis.k3'))))")
        src = str(pkg_dir.parents[2])
        path = os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": path}, check=True)
        self.assertEqual(out.stdout.strip(), "", out.stdout)

    def test_missing_k3_module_fails_with_a_clear_message(self):
        with mock.patch.dict(sys.modules, {K3_SELECTOR_MODULE: None}):
            with self.assertRaisesRegex(K3SelectorUnavailable, "scikit-learn"):
                K3SelectorAdapter.fit(_pool())
            with self.assertRaisesRegex(K3SelectorUnavailable, "analysis/k3/selector.py"):
                k3_adapter.k3_module()

    def test_held_out_names_are_refused_before_anything_else(self):
        # the guard does not depend on the K3 module being importable
        for column, name in (("dataset", "PJM"), ("dataset", "M5_Retail"), ("config", "pjm/H")):
            with self.subTest(column=column, name=name):
                pool = _pool()
                pool.loc[pool.dataset == "fb", column] = name
                with mock.patch.dict(sys.modules, {K3_SELECTOR_MODULE: None}):
                    with self.assertRaises(HeldOutLeakError):
                        K3SelectorAdapter.fit(pool)

    def test_default_run_is_still_the_placeholder(self):
        res, d = _shared.results()
        self.assertIsInstance(d, LeaveOneDatasetOut)
        self.assertTrue(all(isinstance(f, DOmegaLinear) for f in d.folds.values()))
        self.assertEqual(inspect.signature(run).parameters["d_omega_fit"].default, None)
        explicit, _ = run(_shared.tables(), REPRESENTATIVES, THRESHOLD, d_omega_fit=DOmegaLinear.fit)
        pd.testing.assert_frame_equal(explicit, res)
        self.assertTrue((res.status == "selected").all())
        self.assertFalse(res.note.str.contains("reason=").any())

    def test_old_scheme_default_is_still_the_placeholder(self):
        _, d = run(_shared.tables(), REPRESENTATIVES, THRESHOLD, fit_scheme="fit_only_datasets")
        self.assertIsInstance(d, DOmegaLinear)

    def test_run_hands_d_omega_fit_each_fold_pool_under_both_schemes(self):
        seen = []

        def stub(fit_units):
            seen.append(sorted(fit_units["dataset"].unique()))
            return NotFitted("d_omega", "stub")

        res, d = run(_shared.tables(), REPRESENTATIVES, THRESHOLD, d_omega_fit=stub)
        self.assertTrue(all(isinstance(f, NotFitted) for f in d.folds.values()))
        self.assertEqual(len(seen), len(EVAL_DATASETS))
        for ds, got in zip(EVAL_DATASETS, seen):                            # folds are built in sorted order
            self.assertEqual(got, sorted((set(EVAL_DATASETS) - {ds}) | set(FIT_ONLY_DATASETS)))
        self.assertTrue((_d_omega(res).note == "fold=" + _d_omega(res).dataset + " not fitted: stub").all())
        seen.clear()
        _, d = run(_shared.tables(), REPRESENTATIVES, THRESHOLD, fit_scheme="fit_only_datasets", d_omega_fit=stub)
        self.assertIsInstance(d, NotFitted)
        self.assertEqual(seen, [list(FIT_ONLY_DATASETS)])

    def test_summary_of_a_group_with_nothing_selected_says_so(self):
        res, _ = run(_shared.tables(), REPRESENTATIVES, THRESHOLD,
                     d_omega_fit=lambda fit_units: NotFitted("d_omega", "stub"))
        s = summarize(res, by=("selector", "dataset"))
        mine = s[s.selector == "d_omega"]
        self.assertTrue((mine.n_selected == 0).all())
        self.assertTrue(mine.frac_oracle_value.isna().all())
        self.assertTrue((mine.frac_note == "undefined: no selected units").all())
        base = summarize(_shared.results()[0], by=("selector", "dataset"))  # every other group is as before
        pd.testing.assert_frame_equal(s[s.selector != "d_omega"].reset_index(drop=True),
                                      base[base.selector != "d_omega"].reset_index(drop=True))

    def test_reason_of_reads_notes_with_and_without_the_fold_prefix(self):
        notes = pd.Series(["fold=fx_tie reason=rule", "reason=omega_out_of_range", "omega0=0.41 beta=+0.000", ""])
        self.assertEqual(reason_of(notes).tolist()[:2], ["rule", "omega_out_of_range"])
        self.assertTrue(reason_of(notes).iloc[2:].isna().all())

    def test_run_fixture_default_command_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(sys, "argv", ["run_fixture", "--out", tmp]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_fixture.main()
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             ["fixture_cache.parquet", "fixture_omega.csv", "planted_answers.csv",
                              "selection_results.parquet", "summary_by_dataset.csv"])
            res = pd.read_parquet(Path(tmp) / "selection_results.parquet")
        self.assertIn("D_Omega placeholder, leave-one-dataset-out folds:", buf.getvalue())
        self.assertNotIn("without a choice", buf.getvalue())
        pd.testing.assert_frame_equal(res, _shared.results()[0])


@needs_k3
class HandBuiltPool(unittest.TestCase):
    """Answers known by construction, so a wrong orientation, key or mapping
    cannot hide behind 'equals what the K3 function returned'."""

    def setUp(self):
        self.sel = K3SelectorAdapter.fit(_pool())

    def test_defaults_are_his_defaults_and_are_recorded(self):
        self.assertEqual(DEFAULT_MARGIN, inspect.signature(k3.apply_selector).parameters["margin"].default)
        self.assertEqual(DEFAULT_MIN_DISTINCT, inspect.signature(k3.fit_selector).parameters["min_distinct"].default)
        self.assertEqual(DEFAULT_COST, "raw")
        desc = self.sel.describe()
        self.assertEqual((desc["margin"], desc["min_distinct"], desc["cost"]), (0.0, 3, "raw"))
        self.assertEqual(desc["fitted_datasets"], ["fa", "fb", "fc"])
        self.assertEqual(sorted(desc["specs"]), ["1"])
        self.assertEqual(desc["specs"]["1"]["spec_version"], k3.SPEC_VERSION)
        self.assertEqual(desc["specs"]["1"]["min_distinct_datasets"], 3)

    def test_planted_boundary_and_both_orientations_are_fitted(self):
        rules = self.sel.specs[1]["rules"]
        self.assertEqual(sorted(rules), ["0.50", "0.91"])
        self.assertEqual((rules["0.50"]["orientation"], rules["0.50"]["threshold"]), ("classical_below", 0.5))
        self.assertEqual((rules["0.91"]["orientation"], rules["0.91"]["threshold"]), ("tsfm_below", 0.5))
        self.assertEqual((rules["0.50"]["omega_min"], rules["0.50"]["omega_max"]), (0.125, 0.875))
        self.assertEqual(rules["0.50"]["n_datasets"], 3)
        self.assertEqual(rules["0.50"]["n_series"], 12)              # one row per series at each tau
        self.assertEqual(rules["0.50"]["training_cost"], 12.0)       # raw cost: every series at cost 1

    def test_choices_and_deferrals_on_known_units(self):
        want = [  # omega, family at tau 0.5, family at tau 0.91, reason
            (0.1875, "classical", "tsfm", "rule"),
            (0.8125, "tsfm", "classical", "rule"),
            (0.5, "classical", "tsfm", "rule"),                      # on the threshold counts as below
            (0.125, "classical", "tsfm", "rule"),                    # lowest fitted Omega is in range
            (0.875, "tsfm", "classical", "rule"),                    # highest fitted Omega is in range
            (0.9375, None, None, "omega_out_of_range"),
            (0.0625, None, None, "omega_out_of_range"),
            (np.nan, None, None, "omega_undefined"),
        ]
        units = _keys("ev", [w[0] for w in want])
        out = _select(self.sel, units)
        self.assertEqual(list(out.columns), UNIT_KEYS + ["chosen_family", "status", "note"])
        self.assertTrue((out[UNIT_KEYS].to_numpy() == units[UNIT_KEYS].to_numpy()).all())
        for i, (om, at_50, at_91, reason) in enumerate(want):
            for tau, fam in ((0.5, at_50), (0.91, at_91)):
                r = out[(out.series_id == f"ev/s{i}") & (out.tau == tau)].iloc[0]
                with self.subTest(omega=om, tau=tau):
                    self.assertEqual(r.note, f"reason={reason}")
                    if fam is None:
                        self.assertEqual(r.status, "defer")
                        self.assertTrue(pd.isna(r.chosen_family))
                    else:
                        self.assertEqual((r.status, r.chosen_family), ("selected", fam))

    def test_margin_defers_near_the_threshold_and_is_recorded(self):
        units = _keys("ev", [0.4375, 0.5625, 0.1875], taus=(0.5,))          # 0.0625 from the threshold, twice
        self.assertEqual(_select(self.sel, units).status.tolist(), ["selected"] * 3)
        wide = K3SelectorAdapter.fit(_pool(), margin=0.1)
        out = _select(wide, units)
        self.assertEqual(out.status.tolist(), ["defer", "defer", "selected"])
        self.assertEqual(out.note.tolist(), ["reason=weak_evidence", "reason=weak_evidence", "reason=rule"])
        self.assertTrue(out.chosen_family.iloc[:2].isna().all())
        self.assertEqual(wide.describe()["margin"], 0.1)
        # his rule is strict: exactly margin away is not weak evidence
        edge = K3SelectorAdapter.fit(_pool(), margin=0.0625)
        self.assertEqual(_select(edge, units).status.tolist(), ["selected"] * 3)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            K3SelectorAdapter.fit(_pool(), margin=-0.1)

    def test_a_tau_or_horizon_without_a_rule_is_deferred(self):
        units = pd.concat([_keys("ev", [0.1875], taus=(0.25,)),
                           _keys("ev", [0.1875], taus=(0.5,), horizon=24)], ignore_index=True)
        out = _select(self.sel, units)
        self.assertEqual(out.status.tolist(), ["defer", "defer"])
        self.assertEqual(out.note.tolist(), ["reason=unfitted_tau", "reason=unfitted_horizon"])
        self.assertTrue(out.chosen_family.isna().all())

    def test_one_spec_per_horizon_each_fitted_on_its_own_units(self):
        """Horizon 24 gets the mirror image of horizon 1; pooling them would fit neither."""
        p24 = _pool(horizon=24).rename(columns={"J_classical": "J_tsfm", "J_tsfm": "J_classical"})
        both = pd.concat([_pool(), p24], ignore_index=True)
        sel = K3SelectorAdapter.fit(both)
        self.assertEqual(sorted(sel.specs), [1, 24])
        self.assertEqual(sel.specs[1], self.sel.specs[1])                   # unaffected by the other horizon
        self.assertEqual(sel.specs[24]["rules"]["0.50"]["orientation"], "tsfm_below")
        self.assertEqual(sel.specs[24]["rules"]["0.50"]["n_series"], 12)
        units = pd.concat([_keys("ev", [0.1875], taus=(0.5,)),
                           _keys("ev", [0.1875], taus=(0.5,), horizon=24)], ignore_index=True)
        self.assertEqual(_select(sel, units).chosen_family.tolist(), ["classical", "tsfm"])

    def test_fewer_datasets_than_min_distinct_defers_every_unit(self):
        sel = K3SelectorAdapter.fit(_pool(datasets=("fa", "fb")))
        for rule in sel.specs[1]["rules"].values():
            self.assertEqual((rule["status"], rule["n_datasets"], rule["required"]), ("insufficient_data", 2, 3))
        out = _select(sel, _keys("ev", [0.1875, 0.8125]))
        self.assertTrue((out.status == "defer").all())
        self.assertTrue((out.note == "reason=insufficient_data").all())
        self.assertTrue(out.chosen_family.isna().all())
        # min_distinct is passed through and recorded
        two = K3SelectorAdapter.fit(_pool(datasets=("fa", "fb")), min_distinct=2)
        self.assertEqual(two.describe()["min_distinct"], 2)
        self.assertEqual(_select(two, _keys("ev", [0.1875, 0.8125])).status.tolist(), ["selected"] * 4)
        for bad in (0, 2.5):
            with self.assertRaisesRegex(ValueError, "min_distinct"):
                K3SelectorAdapter.fit(_pool(), min_distinct=bad)

    def test_undefined_omega_in_the_pool_reaches_him_as_none(self):
        pool = _pool()
        extra = _keys("fa", [np.nan]).assign(series_id="fa/undefined", J_classical=5.0, J_tsfm=0.1, scale=1.0)
        sel = K3SelectorAdapter.fit(pd.concat([pool, extra[FIT_POOL_COLUMNS]], ignore_index=True))
        self.assertEqual(sel.specs[1]["training_undefined_omega"], 2)       # one series, two taus
        self.assertEqual(sel.specs[1]["rules"], self.sel.specs[1]["rules"])  # and it moved no rule
        self.assertIn("fa/undefined", sel.fitted_series)

    def test_units_with_a_missing_cost_are_left_out_of_the_fit(self):
        extra = _keys("fa", [0.5]).assign(series_id="fa/nocost", J_classical=np.nan, J_tsfm=1.0, scale=1.0)
        sel = K3SelectorAdapter.fit(pd.concat([_pool(), extra[FIT_POOL_COLUMNS]], ignore_index=True))
        self.assertEqual(sel.specs, self.sel.specs)
        self.assertNotIn("fa/nocost", sel.fitted_series)

    def test_cost_choice_changes_what_he_is_fitted_on(self):
        """One series with a large scale: raw cost lets it move the threshold, scaled cost does not."""
        pool = _pool()
        big = (pool.series_id == "fc/s1") & (pool.tau == 0.5)               # omega 0.25
        pool.loc[big, ["J_classical", "J_tsfm", "scale"]] = [2000.0, 1000.0, 1000.0]
        raw = K3SelectorAdapter.fit(pool)
        scaled = K3SelectorAdapter.fit(pool, cost="scaled")
        self.assertEqual(raw.specs[1]["rules"]["0.50"]["threshold"], 0.1875)
        self.assertEqual(scaled.specs[1]["rules"]["0.50"]["threshold"], 0.5)
        self.assertEqual(raw.specs[1]["rules"]["0.50"]["training_cost"], 1013.0)
        self.assertEqual(scaled.specs[1]["rules"]["0.50"]["training_cost"], 13.0)
        self.assertEqual((raw.describe()["cost"], scaled.describe()["cost"]), ("raw", "scaled"))
        unit = _keys("ev", [0.25], taus=(0.5,))
        self.assertEqual(_select(raw, unit).chosen_family.tolist(), ["tsfm"])
        self.assertEqual(_select(scaled, unit).chosen_family.tolist(), ["classical"])
        with self.assertRaisesRegex(ValueError, "cost"):
            K3SelectorAdapter.fit(pool, cost="regret")

    def test_taus_his_two_decimal_keys_cannot_tell_apart_are_refused(self):
        pool = _pool()
        pool.loc[pool.tau == 0.91, "tau"] = 0.505                           # would share the key "0.50" or "0.51"
        with self.assertRaisesRegex(ValueError, "two decimals"):
            K3SelectorAdapter.fit(pool)
        with self.assertRaisesRegex(ValueError, "two decimals"):
            _select(self.sel, _keys("ev", [0.1875], taus=(0.504,)))

    def test_duplicate_units_in_the_pool_are_refused(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            K3SelectorAdapter.fit(pd.concat([_pool(), _pool().iloc[:1]], ignore_index=True))

    def test_empty_pool_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "no usable units"):
            K3SelectorAdapter.fit(_pool().iloc[0:0])

    def test_refuses_to_evaluate_on_its_fit_series(self):
        with self.assertRaises(FitEvalOverlapError):
            _select(self.sel, _keys("fa", [0.1875]))

    def test_applying_never_changes_the_frozen_spec(self):
        before = copy.deepcopy(self.sel.specs)
        for _ in range(2):
            _select(self.sel, _keys("ev", [0.1875, 0.9375, np.nan]))
        self.assertEqual(self.sel.specs, before)
        self.assertTrue(all(s["frozen"] is True for s in self.sel.specs.values()))
        desc = self.sel.describe()
        desc["specs"]["1"]["rules"]["0.50"]["threshold"] = 0.0             # editing the description
        self.assertEqual(self.sel.specs, before)                           # does not reach the selector
        with self.assertRaises(Exception):
            self.sel.margin = 0.3                                          # frozen dataclass

    def test_description_is_json_serialisable(self):
        desc = self.sel.describe()
        self.assertEqual(json.loads(json.dumps(desc)), desc)

    def test_every_choice_equals_apply_selector_called_directly(self):
        units = pd.concat([_keys("ev", [0.1875, 0.5, 0.875, 0.9375, np.nan], taus=(0.25, 0.5, 0.91)),
                           _keys("ev", [0.3], taus=(0.5,), horizon=24)], ignore_index=True)
        for margin in (0.0, 0.1):
            sel = K3SelectorAdapter.fit(_pool(), margin=margin)
            out = _select(sel, units)
            p = _pool()
            spec = k3.fit_selector(p.omega.tolist(), p.tau.to_numpy(), p.J_classical.to_numpy(),
                                   p.J_tsfm.to_numpy(), p.dataset.to_numpy(dtype=object))
            for u, r in zip(units.itertuples(), out.itertuples()):
                if u.horizon != 1:
                    self.assertEqual((r.status, r.note), ("defer", "reason=unfitted_horizon"))
                    continue
                choice, reason = k3.apply_selector(spec, None if np.isnan(u.omega) else u.omega, u.tau,
                                                   margin=margin)
                self.assertEqual(r.note, f"reason={reason}")
                self.assertEqual(r.status, "defer" if choice == k3.DEFER else "selected")
                self.assertEqual(None if pd.isna(r.chosen_family) else r.chosen_family,
                                 None if choice == k3.DEFER else choice)


@needs_k3
class DeferralsInTheCommonSchema(unittest.TestCase):
    def setUp(self):
        self.units = _keys("ev", [0.1875, 0.8125, 0.9375, np.nan])
        self.t = _eval_tables(self.units)
        self.sel = K3SelectorAdapter.fit(_pool())

    def _run(self, selectors):
        t = self.t
        return run_selectors(selectors, t["units"], t["val"], t["test"], REPRESENTATIVES, t["n_val"])

    def test_deferred_rows_are_valid_and_carry_no_family_and_no_cost(self):
        res = validate_selection_frame(self._run([self.sel]))
        self.assertEqual(res.status.tolist(), ["selected"] * 4 + ["defer"] * 4)
        d = res[res.status == "defer"]
        for col in ("chosen_family", "chosen_model", "realized_cost", "regret", "scaled_regret"):
            self.assertTrue(d[col].isna().all(), col)
        # what does not depend on the choice is still reported for a deferred unit
        self.assertEqual(d.oracle_cost.tolist(), [1.0] * 4)
        self.assertEqual(d.oracle_family.tolist(), ["classical"] * 4)
        self.assertEqual(d.note.tolist(), ["reason=omega_out_of_range"] * 2 + ["reason=omega_undefined"] * 2)
        s = res[res.status == "selected"]
        self.assertEqual(s.chosen_family.tolist(), ["classical", "tsfm", "tsfm", "classical"])
        self.assertEqual(s.regret.tolist(), [0.0, 1.0, 1.0, 0.0])            # classical costs 1, TSFM 2

    def test_a_deferral_is_not_priced_as_the_oracle_or_as_anything_else(self):
        res = self._run([self.sel])
        d = res[res.status == "defer"]
        self.assertFalse(np.isfinite(d.realized_cost).any())
        self.assertFalse((d.realized_cost == d.oracle_cost).any())

    def test_all_seven_stay_valid_next_to_a_deferring_d_omega(self):
        res = self._run(seven_selectors(THRESHOLD, self.sel))
        self.assertEqual(set(res.selector), SEVEN)
        self.assertEqual(set(res[res.status == "defer"].selector), {"d_omega"})

    def test_summary_counts_selected_units_only(self):
        res = self._run(seven_selectors(THRESHOLD, self.sel))
        s = summarize(res, by=("selector",)).set_index("selector")
        self.assertEqual((s.loc["d_omega", "n_units"], s.loc["d_omega", "n_selected"]), (8, 4))
        self.assertAlmostEqual(s.loc["d_omega", "mean_scaled_regret"], 0.5, places=12)
        self.assertEqual((s.loc["always_tsfm", "n_units"], s.loc["always_tsfm", "n_selected"]), (8, 8))

    def test_summary_of_a_fully_deferred_group_says_nothing_was_selected(self):
        res = self._run(seven_selectors(THRESHOLD, K3SelectorAdapter.fit(_pool(datasets=("fa", "fb")))))
        self.assertTrue((_d_omega(res).status == "defer").all())
        s = summarize(res, by=("selector",)).set_index("selector")
        row = s.loc["d_omega"]
        self.assertEqual((row.n_units, row.n_selected), (8, 0))
        self.assertTrue(np.isnan(row.mean_scaled_regret))
        self.assertTrue(np.isnan(row.frac_oracle_value))
        self.assertEqual(row.frac_note, "undefined: no selected units")
        # classical is the oracle everywhere in these tables, so the others keep their own reason
        self.assertEqual(s.loc["always_tsfm", "frac_note"], "undefined: best-fixed equals oracle")


@needs_k3
class InsideTheFoldWrapper(unittest.TestCase):
    def _pool_with_eval(self, datasets=("fa", "fb", "fc")):
        ev = _keys("ev", [0.1875, 0.8125]).assign(J_classical=1.0, J_tsfm=2.0, scale=1.0)
        return pd.concat([_pool(datasets), ev[FIT_POOL_COLUMNS]], ignore_index=True)

    def test_no_fold_is_ever_fitted_on_its_own_dataset(self):
        pool, seen = self._pool_with_eval(), {}

        def recording_fit(fit_units):
            seen[len(seen)] = set(fit_units["dataset"])
            return K3SelectorAdapter.fit(fit_units)

        names = ("ev", "fa", "fb", "fc")
        d = LeaveOneDatasetOut.build(pool, names, fit=recording_fit)
        self.assertEqual(len(seen), 4)
        for i, ds in enumerate(names):                                      # folds are built in sorted order
            self.assertEqual(seen[i], set(names) - {ds})
            self.assertNotIn(ds, d.folds[ds].fitted_datasets)
            self.assertEqual(set(d.folds[ds].fitted_datasets), set(names) - {ds})
            self.assertFalse(set(pool.loc[pool.dataset == ds, "series_id"]) & d.folds[ds].fitted_series)

    def test_a_fold_fitted_on_its_own_dataset_is_refused(self):
        pool = self._pool_with_eval()
        leaky = K3SelectorAdapter.fit(pool)                                 # includes "ev"
        self.assertIn("ev", leaky.fitted_datasets)
        guard = LeaveOneDatasetOut(folds={"ev": leaky}, min_fit_datasets=2)
        units = pool.loc[pool.dataset == "ev", UNIT_KEYS + ["omega"]]
        with self.assertRaises(FitEvalOverlapError):
            guard.select(SelectorInput(units=units, val=units))

    def test_held_out_dataset_in_the_pool_is_refused_through_the_wrapper(self):
        pool = self._pool_with_eval()
        pool.loc[pool.dataset == "fb", "dataset"] = "PJM"
        with self.assertRaises(HeldOutLeakError):
            LeaveOneDatasetOut.build(pool, ["ev"], fit=K3SelectorAdapter.fit)

    def test_fold_with_two_other_datasets_is_fitted_and_defers_every_unit(self):
        """The wrapper needs 2 other datasets, his rule needs 3: the fold is built,
        every rule says insufficient_data, every unit is deferred."""
        pool = self._pool_with_eval(datasets=("fa", "fb"))
        d = LeaveOneDatasetOut.build(pool, ["ev"], fit=K3SelectorAdapter.fit)
        self.assertIsInstance(d.folds["ev"], K3SelectorAdapter)
        units = pool.loc[pool.dataset == "ev", UNIT_KEYS + ["omega"]]
        out = d.select(SelectorInput(units=units, val=units))
        self.assertTrue((out.status == "defer").all())
        self.assertTrue((out.note == "fold=ev reason=insufficient_data").all())

    def test_fold_with_one_other_dataset_is_not_fitted_at_all(self):
        pool = self._pool_with_eval(datasets=("fa",))
        d = LeaveOneDatasetOut.build(pool, ["ev"], fit=K3SelectorAdapter.fit)
        units = pool.loc[pool.dataset == "ev", UNIT_KEYS + ["omega"]]
        out = d.select(SelectorInput(units=units, val=units))
        self.assertTrue((out.status == "insufficient_data").all())          # the wrapper's own answer
        self.assertTrue(out.note.str.contains("1 other dataset").all())


@needs_k3
class OnTheFixtureThroughRun(unittest.TestCase):
    def setUp(self):
        self.res, self.d = _k3_results()
        self.tables = _shared.tables()

    def test_runs_end_to_end_and_only_the_d_omega_slot_differs(self):
        validate_selection_frame(self.res)
        self.assertEqual(set(self.res.selector), SEVEN)
        self.assertEqual(len(self.res), 7 * len(self.tables["units"]))
        base = _shared.results()[0]
        others = self.res.selector != "d_omega"
        pd.testing.assert_frame_equal(self.res[others], base[base.selector != "d_omega"])
        self.assertTrue(set(_d_omega(self.res).status) <= {"selected", "defer"})

    def test_one_adapter_per_fold_never_fitted_on_its_own_dataset(self):
        self.assertIsInstance(self.d, LeaveOneDatasetOut)
        self.assertEqual(sorted(self.d.folds), list(EVAL_DATASETS))
        units = self.tables["units"]
        for ds, fold in self.d.folds.items():
            self.assertIsInstance(fold, K3SelectorAdapter)
            self.assertEqual(sorted(fold.fitted_datasets),
                             sorted((set(EVAL_DATASETS) - {ds}) | set(FIT_ONLY_DATASETS)), ds)
            self.assertFalse(set(units.loc[units.dataset == ds, "series_id"]) & fold.fitted_series, ds)
            self.assertEqual(sorted(fold.specs), [1, 24])

    def test_every_choice_equals_an_independent_fit_and_apply_per_fold_and_horizon(self):
        """Build his arrays by hand from the pool (without the fold's dataset, one
        horizon at a time), call his two functions directly, compare every unit."""
        pool = fit_pool(self.tables).dropna(subset=["omega", "J_classical", "J_tsfm", "scale"])
        units = self.tables["units"]
        got = _d_omega(self.res).set_index(UNIT_KEYS)
        checked = 0
        for ds in EVAL_DATASETS:
            for h in sorted(units.horizon.unique()):
                p = pool[(pool.dataset != ds) & (pool.horizon == h)]
                self.assertNotIn(ds, set(p.dataset))
                spec = k3.fit_selector(p.omega.tolist(), p.tau.to_numpy(), p.J_classical.to_numpy(),
                                       p.J_tsfm.to_numpy(), p.dataset.to_numpy(dtype=object))
                self.assertEqual(self.d.folds[ds].specs[int(h)], spec, (ds, h))
                for u in units[(units.dataset == ds) & (units.horizon == h)].itertuples():
                    choice, reason = k3.apply_selector(spec, None if np.isnan(u.omega) else u.omega, u.tau)
                    r = got.loc[(u.dataset, u.config, u.series_id, u.horizon, u.tau)]
                    self.assertEqual(r.note, f"fold={ds} reason={reason}")
                    self.assertEqual(r.status, "defer" if choice == k3.DEFER else "selected")
                    self.assertEqual(None if pd.isna(r.chosen_family) else r.chosen_family,
                                     None if choice == k3.DEFER else choice)
                    checked += 1
        self.assertEqual(checked, len(units))

    def test_every_fold_fits_every_rule_on_four_datasets(self):
        for ds, fold in self.d.folds.items():
            for h, spec in fold.specs.items():
                self.assertEqual(spec["taus"], sorted(f"{t:.2f}" for t in TAU_GRID))
                for rule in spec["rules"].values():
                    self.assertEqual((rule["status"], rule["n_datasets"]), ("ok", 4), (ds, h))

    def test_note_names_the_fold_then_the_reason(self):
        d = _d_omega(self.res)
        self.assertTrue((d.note.str.split().str[0] == "fold=" + d.dataset).all())
        self.assertTrue(reason_of(d.note).notna().all())
        self.assertTrue((reason_of(d.note[d.status == "selected"]) == "rule").all())
        self.assertFalse((reason_of(d.note[d.status == "defer"]) == "rule").any())

    def test_output_keeps_the_order_of_the_units(self):
        units = self.tables["units"].sample(frac=1.0, random_state=3)
        out = self.d.select(SelectorInput(units=units, val=self.tables["val"]))
        self.assertTrue((out[UNIT_KEYS].to_numpy() == units[UNIT_KEYS].to_numpy()).all())

    def test_description_of_all_folds_is_json_serialisable_and_records_the_settings(self):
        desc = self.d.describe()
        self.assertEqual(json.loads(json.dumps(desc)), desc)
        self.assertEqual(sorted(desc["folds"]), list(EVAL_DATASETS))
        for fold in desc["folds"].values():
            self.assertEqual((fold["selector"], fold["margin"], fold["min_distinct"], fold["cost"]),
                             ("k3_omega_tau", 0.0, 3, "raw"))

    def test_does_not_use_hindsight(self):
        self.assertFalse(K3SelectorAdapter.needs_hindsight)
        self.assertFalse(self.d.needs_hindsight)

    def test_settings_reach_the_selector_through_run(self):
        fit = functools.partial(K3SelectorAdapter.fit, margin=0.05, min_distinct=4, cost="scaled")
        res, d = run(self.tables, REPRESENTATIVES, THRESHOLD, d_omega_fit=fit)
        validate_selection_frame(res)
        for fold in d.describe()["folds"].values():
            self.assertEqual((fold["margin"], fold["min_distinct"], fold["cost"]), (0.05, 4, "scaled"))

    def test_min_distinct_above_the_fold_size_defers_everything(self):
        fit = functools.partial(K3SelectorAdapter.fit, min_distinct=5)      # each fold has 4 other datasets
        res, _ = run(self.tables, REPRESENTATIVES, THRESHOLD, d_omega_fit=fit)
        validate_selection_frame(res)
        d = _d_omega(res)
        self.assertTrue((d.status == "defer").all())
        self.assertTrue((reason_of(d.note) == "insufficient_data").all())
        self.assertTrue((res[res.selector != "d_omega"].status == "selected").all())

    def test_old_scheme_with_one_fit_only_dataset_defers_everything(self):
        res, d = run(self.tables, REPRESENTATIVES, THRESHOLD, fit_scheme="fit_only_datasets",
                     d_omega_fit=K3SelectorAdapter.fit)
        self.assertIsInstance(d, K3SelectorAdapter)
        self.assertEqual(d.fitted_datasets, FIT_ONLY_DATASETS)
        validate_selection_frame(res)
        dm = _d_omega(res)
        self.assertTrue((dm.status == "defer").all())
        self.assertEqual(dm.note.unique().tolist(), ["reason=insufficient_data"])
        s = summarize(res, by=("selector", "dataset"))
        mine = s[s.selector == "d_omega"]
        self.assertTrue((mine.n_selected == 0).all())
        self.assertTrue(mine.mean_scaled_regret.isna().all())
        self.assertTrue(mine.frac_oracle_value.isna().all())

    def test_undefined_omega_is_a_deferral_with_its_reason(self):
        om = _shared.fixture_data()[1].copy()
        om.loc[om.series_id == "fx_tsfm_wins/s0", "omega_levels"] = np.nan
        res, _ = run(_fixture_tables(omega=om), REPRESENTATIVES, THRESHOLD, d_omega_fit=K3SelectorAdapter.fit)
        validate_selection_frame(res)
        mine = res[(res.series_id == "fx_tsfm_wins/s0") & (res.selector == "d_omega")]
        self.assertEqual(len(mine), 2 * len(TAU_GRID))
        self.assertTrue((mine.status == "defer").all())
        self.assertTrue((mine.note == "fold=fx_tsfm_wins reason=omega_undefined").all())
        self.assertTrue(mine.chosen_family.isna().all())
        self.assertTrue(mine.omega.isna().all())
        rest = res[(res.series_id != "fx_tsfm_wins/s0") & (res.selector == "d_omega")]
        self.assertFalse((reason_of(rest.note) == "omega_undefined").any())


@needs_k3
class PartlyDeferredOnTheFixture(unittest.TestCase):
    """Without the fit-only dataset every fold has exactly 3 other datasets and
    some evaluation Omega values fall outside what a fold was fitted on."""

    def setUp(self):
        self.tables = _tables_without_fit_only()
        self.res, self.d = run(self.tables, REPRESENTATIVES, THRESHOLD, d_omega_fit=K3SelectorAdapter.fit)

    def test_out_of_range_units_are_deferred_fold_by_fold(self):
        validate_selection_frame(self.res)
        d = _d_omega(self.res)
        n = 2 * len(TAU_GRID)                                               # units per series
        got = d.groupby(["dataset", "status"]).size().to_dict()
        self.assertEqual(got, {("fx_classical_wins", "defer"): 6 * n,       # 0.10-0.25, fold fitted on 0.42-0.82
                               ("fx_tau_flip", "selected"): 6 * n,
                               ("fx_tie", "selected"): 4 * n,
                               ("fx_tsfm_wins", "defer"): 5 * n,            # 0.66-0.82, fold fitted on 0.10-0.62
                               ("fx_tsfm_wins", "selected"): 1 * n})        # 0.62 is the fitted maximum
        self.assertEqual(set(reason_of(d.note[d.status == "defer"])), {"omega_out_of_range"})
        for fold in self.d.folds.values():
            for spec in fold.specs.values():
                self.assertTrue(all(r["n_datasets"] == 3 for r in spec["rules"].values()))

    def test_summary_and_regret_use_the_selected_units_only(self):
        d = _d_omega(self.res)
        s = summarize(self.res, by=("selector", "dataset"))
        s = s[s.selector == "d_omega"].set_index("dataset")
        n = 2 * len(TAU_GRID)
        self.assertEqual((s.loc["fx_tsfm_wins", "n_units"], s.loc["fx_tsfm_wins", "n_selected"]), (6 * n, n))
        sel = d[(d.dataset == "fx_tsfm_wins") & (d.status == "selected")]
        self.assertAlmostEqual(s.loc["fx_tsfm_wins", "mean_scaled_regret"], sel.scaled_regret.mean(), places=15)
        self.assertEqual(s.loc["fx_classical_wins", "n_selected"], 0)
        self.assertTrue(np.isnan(s.loc["fx_classical_wins", "mean_scaled_regret"]))
        self.assertEqual(s.loc["fx_classical_wins", "frac_note"], "undefined: no selected units")
        self.assertTrue(d[d.status == "defer"].regret.isna().all())
        self.assertTrue((d[d.status == "selected"].regret >= 0).all())


@needs_k3
class RunFixtureCommandWithTheK3Selector(unittest.TestCase):
    def _main(self, *args):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(sys, "argv", ["run_fixture", "--out", tmp, "--d-omega", "k3_selector", *args]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_fixture.main()
            files = sorted(p.name for p in Path(tmp).iterdir())
            res = pd.read_parquet(Path(tmp) / "selection_results.parquet")
            summary = pd.read_csv(Path(tmp) / "summary_by_dataset.csv", encoding="utf-8")
            folds = json.loads((Path(tmp) / "d_omega_folds.json").read_text(encoding="utf-8"))
        return buf.getvalue(), files, res, summary, folds

    def test_writes_the_frozen_folds_and_matches_run(self):
        out, files, res, summary, folds = self._main()
        self.assertEqual(files, ["d_omega_folds.json", "fixture_cache.parquet", "fixture_omega.csv",
                                 "planted_answers.csv", "selection_results.parquet", "summary_by_dataset.csv"])
        pd.testing.assert_frame_equal(res, _k3_results()[0])
        self.assertEqual(folds, _k3_results()[1].describe())
        self.assertIn("D_Omega K3 Omega-plus-tau selector", out)
        self.assertNotIn("D_Omega placeholder", out)

    def test_every_unit_deferred_still_reports(self):
        out, _, res, summary, folds = self._main("--k3-min-distinct", "5")
        validate_selection_frame(res)
        d = _d_omega(res)
        self.assertTrue((d.status == "defer").all())
        self.assertIn("D_Omega units without a choice", out)
        self.assertIn("insufficient_data", out)
        mine = summary[summary.selector == "d_omega"]
        self.assertTrue((mine.n_selected == 0).all())
        self.assertTrue(mine.mean_scaled_regret.isna().all())
        self.assertTrue((mine.frac_note == "undefined: no selected units").all())
        self.assertTrue(all(f["min_distinct"] == 5 for f in folds["folds"].values()))

    def test_some_units_deferred_still_reports(self):
        out, _, res, _, folds = self._main("--k3-margin", "0.2")
        validate_selection_frame(res)
        d = _d_omega(res)
        self.assertEqual(set(d.status), {"selected", "defer"})
        self.assertEqual(set(reason_of(d.note[d.status == "defer"])), {"weak_evidence"})
        self.assertIn("D_Omega units without a choice", out)
        self.assertIn("weak_evidence", out)
        self.assertTrue(all(f["margin"] == 0.2 for f in folds["folds"].values()))


if __name__ == "__main__":
    unittest.main()
