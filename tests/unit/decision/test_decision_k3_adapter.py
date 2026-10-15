"""The K3 Omega-plus-tau selector (#26, analysis/k3/selector.py) in the D_Omega slot,
through decision.k3_adapter.

Synthetic fixture and hand-built tables only. No real cache, no M5 or PJM.
The classes that call the K3 selector need analysis/k3/selector.py (PR #54)
and analysis/k3/boundary.py (PR #32); without them they are skipped and say
so. Deferral pricing (pilot decision 8) is tested with and without them."""
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
from decision_aware_tsfm.analysis.decision.evaluate import (
    CHARGE_COLUMNS, DEFERRAL_PRICING, charge_deferrals, run_selectors, summarize, summarize_charged)
from decision_aware_tsfm.analysis.decision.fixture import (
    FAMILY_OF_MODEL, FIT_ONLY_DATASETS, FIXTURE_SCHEMA, FIXTURE_STATIC_THRESHOLD as THRESHOLD,
    REPRESENTATIVES)
from decision_aware_tsfm.analysis.decision.guards import FitEvalOverlapError, HeldOutLeakError
from decision_aware_tsfm.analysis.decision.k3_adapter import (
    DEFAULT_COST, DEFAULT_K3_MIN_DISTINCT, DEFAULT_MARGIN, DEFAULT_MIN_FIT_DATASETS, K3_SELECTOR_MODULE,
    K3SelectorAdapter, K3SelectorUnavailable, build_k3_d_omega, reason_of)
from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID
from decision_aware_tsfm.analysis.decision.pipeline import (
    build_tables, fit_d_omega, fit_pool, run, seven_selectors)
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
             "analysis/k3/selector.py (PR #54) and analysis/k3/boundary.py (PR #32)")
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
            with self.assertRaisesRegex(K3SelectorUnavailable, "PR #54"):
                K3SelectorAdapter.fit(_pool())
            with self.assertRaisesRegex(K3SelectorUnavailable, "PR #54"):
                build_k3_d_omega(_shared.tables())
            res, _ = run(_shared.tables(), REPRESENTATIVES, THRESHOLD)        # the default never needs it
            pd.testing.assert_frame_equal(res, _shared.results()[0])
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
        self.assertEqual(DEFAULT_K3_MIN_DISTINCT,
                         inspect.signature(k3.fit_selector).parameters["min_distinct"].default)
        self.assertEqual(DEFAULT_COST, "scaled")                            # pilot decision 9
        desc = self.sel.describe()
        self.assertEqual((desc["margin"], desc["k3_min_distinct"], desc["cost"]), (0.0, 3, "scaled"))
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
        self.assertEqual(rules["0.50"]["training_cost"], 12.0)       # every series at cost 1, scale 1

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
        # k3_min_distinct is passed through and recorded
        two = K3SelectorAdapter.fit(_pool(datasets=("fa", "fb")), k3_min_distinct=2)
        self.assertEqual(two.describe()["k3_min_distinct"], 2)
        self.assertEqual(_select(two, _keys("ev", [0.1875, 0.8125])).status.tolist(), ["selected"] * 4)
        for bad in (0, 2.5):
            with self.assertRaisesRegex(ValueError, "k3_min_distinct"):
                K3SelectorAdapter.fit(_pool(), k3_min_distinct=bad)

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
        raw = K3SelectorAdapter.fit(pool, cost="raw")
        scaled = K3SelectorAdapter.fit(pool)
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

    def test_scaled_fit_refuses_a_non_positive_scale(self):
        for bad in (0.0, -1.0):
            pool = _pool()
            pool.loc[pool.index[0], "scale"] = bad
            with self.assertRaisesRegex(ValueError, "positive scale"):
                K3SelectorAdapter.fit(pool)
            K3SelectorAdapter.fit(pool, cost="raw")                          # raw cost does not divide

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

    def test_summary_refuses_deferrals_and_the_charged_summary_counts_selected_units_apart(self):
        res = self._run(seven_selectors(THRESHOLD, self.sel))
        with self.assertRaisesRegex(ValueError, "4 deferred row.*summarize_charged"):
            summarize(res, by=("selector",))
        s = summarize_charged(res, by=("selector",)).set_index("selector")
        self.assertEqual((s.loc["d_omega", "n_units"], s.loc["d_omega", "n_selected"]), (8, 4))
        self.assertEqual((s.loc["d_omega", "n_deferred"], s.loc["d_omega", "deferral_rate"]), (4, 0.5))
        self.assertAlmostEqual(s.loc["d_omega", "mean_scaled_regret"], 0.5, places=12)
        self.assertEqual((s.loc["always_tsfm", "n_units"], s.loc["always_tsfm", "n_selected"]), (8, 8))

    def test_summary_of_a_fully_deferred_group_says_nothing_was_selected(self):
        res = self._run(seven_selectors(THRESHOLD, K3SelectorAdapter.fit(_pool(datasets=("fa", "fb")))))
        self.assertTrue((_d_omega(res).status == "defer").all())
        s = summarize_charged(res, by=("selector",)).set_index("selector")
        row = s.loc["d_omega"]
        self.assertEqual((row.n_units, row.n_selected, row.n_deferred), (8, 0, 8))
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

    def test_fold_with_two_other_datasets_is_not_fitted_by_default(self):
        """Engine floor 3 (#61): two other datasets is too few, the fold is not built."""
        pool = self._pool_with_eval(datasets=("fa", "fb"))
        d = LeaveOneDatasetOut.build(pool, ["ev"], fit=K3SelectorAdapter.fit)
        self.assertIsInstance(d.folds["ev"], NotFitted)
        self.assertIn("2 other dataset(s) with usable units, need 3", d.folds["ev"].reason)

    def test_fold_with_two_other_datasets_under_an_engine_floor_of_2_meets_his_floor(self):
        """Engine floor lowered to 2, his floor still 3: the fold is built,
        every rule says insufficient_data, every unit is deferred."""
        pool = self._pool_with_eval(datasets=("fa", "fb"))
        d = LeaveOneDatasetOut.build(pool, ["ev"], fit=K3SelectorAdapter.fit, min_fit_datasets=2)
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
                spec = k3.fit_selector(p.omega.tolist(), p.tau.to_numpy(),
                                       p.J_classical.to_numpy() / p.scale.to_numpy(),
                                       p.J_tsfm.to_numpy() / p.scale.to_numpy(), p.dataset.to_numpy(dtype=object))
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
            self.assertEqual((fold["selector"], fold["margin"], fold["k3_min_distinct"], fold["cost"]),
                             ("k3_omega_tau", 0.0, 3, "scaled"))
        self.assertEqual(desc["min_fit_datasets"], 3)                       # the engine's floor, named apart

    def test_does_not_use_hindsight(self):
        self.assertFalse(K3SelectorAdapter.needs_hindsight)
        self.assertFalse(self.d.needs_hindsight)

    def test_settings_reach_the_selector_through_run(self):
        fit = functools.partial(K3SelectorAdapter.fit, margin=0.05, k3_min_distinct=4, cost="raw")
        res, d = run(self.tables, REPRESENTATIVES, THRESHOLD, d_omega_fit=fit)
        validate_selection_frame(res)
        for fold in d.describe()["folds"].values():
            self.assertEqual((fold["margin"], fold["k3_min_distinct"], fold["cost"]), (0.05, 4, "raw"))

    def test_min_distinct_above_the_fold_size_defers_everything(self):
        fit = functools.partial(K3SelectorAdapter.fit, k3_min_distinct=5)  # each fold has 4 other datasets
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
        s = summarize_charged(res, by=("selector", "dataset"))
        mine = s[s.selector == "d_omega"]
        self.assertTrue((mine.n_selected == 0).all())
        self.assertTrue(mine.mean_scaled_regret.isna().all())
        self.assertTrue(mine.frac_oracle_value.isna().all())
        # every deferral is charged pinball_validation's cost, so the charged numbers are those of pinball_validation
        pv = s[s.selector == "pinball_validation"].set_index("dataset")
        mine = mine.set_index("dataset")
        self.assertTrue((mine.deferral_rate == 1.0).all())
        self.assertTrue((mine.n_not_charged == 0).all())
        pd.testing.assert_series_equal(mine.mean_scaled_regret_charged, pv.mean_scaled_regret, check_names=False)

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
        s = summarize_charged(self.res, by=("selector", "dataset"))
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
            self.charged = pd.read_parquet(Path(tmp) / "selection_charged.parquet")
            self.per_tau = pd.read_csv(Path(tmp) / "summary_by_horizon_tau.csv", encoding="utf-8")
        return buf.getvalue(), files, res, summary, folds

    def test_writes_the_frozen_folds_and_matches_run(self):
        out, files, res, summary, folds = self._main()
        self.assertEqual(files, ["d_omega_folds.json", "fixture_cache.parquet", "fixture_omega.csv",
                                 "planted_answers.csv", "selection_charged.parquet", "selection_results.parquet",
                                 "summary_by_dataset.csv", "summary_by_horizon_tau.csv"])
        pd.testing.assert_frame_equal(res, _k3_results()[0])
        self.assertEqual(folds, _k3_results()[1].describe())
        pd.testing.assert_frame_equal(self.charged, charge_deferrals(res))
        self.assertEqual(len(self.per_tau), 7 * 2 * len(TAU_GRID))           # one row per selector, horizon and tau
        self.assertTrue((self.per_tau.deferral_pricing == DEFERRAL_PRICING).all())
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
        self.assertTrue((mine.deferral_rate == 1.0).all())                  # not dropped: every unit charged
        self.assertTrue(mine.mean_scaled_regret_charged.notna().all())
        self.assertTrue(all(f["k3_min_distinct"] == 5 for f in folds["folds"].values()))

    def test_some_units_deferred_still_reports(self):
        out, _, res, _, folds = self._main("--k3-margin", "0.2")
        validate_selection_frame(res)
        d = _d_omega(res)
        self.assertEqual(set(d.status), {"selected", "defer"})
        self.assertEqual(set(reason_of(d.note[d.status == "defer"])), {"weak_evidence"})
        self.assertIn("D_Omega units without a choice", out)
        self.assertIn("weak_evidence", out)
        self.assertTrue(all(f["margin"] == 0.2 for f in folds["folds"].values()))



# --------------------------------------------------------------------------- pilot decisions 6, 8, 9 (#29)

def _src_path():
    src = str(Path(pkg.__file__).parents[2])
    return os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)


class _DeferSeries:
    """Stub D_Omega: classical everywhere, defer on the named series. Lets the
    pricing be tested where the K3 module is not importable."""
    name, needs_hindsight = "d_omega", False

    def __init__(self, defer):
        self.defer = set(defer)

    def select(self, inp):
        d = inp.units[UNIT_KEYS].copy()
        mask = d.series_id.isin(self.defer).to_numpy()
        d["chosen_family"] = np.where(mask, None, "classical").astype(object)
        d["status"] = np.where(mask, "defer", "selected")
        d["note"] = np.where(mask, "reason=stub", "")
        return d


def _pricing_tables(val_unknown_s3=True):
    """Four series, taus 0.5 and 0.91. Test: classical 1, TSFM 3, scale 2, so the
    oracle is classical at cost 1. Validation makes pinball_validation choose
    TSFM (wrongly) on s0 and s1, classical on s2, and nothing on s3 (no
    validation costs) unless val_unknown_s3 is False (then classical)."""
    units = _keys("ev", [0.2, 0.4, 0.6, 0.8])
    keys = units[UNIT_KEYS]
    s = keys.series_id.str[-1].astype(int)
    vc = np.where(s <= 1, 1.0, 0.5)
    vt = np.where(s <= 1, 0.5, 1.0)
    if val_unknown_s3:
        vc, vt = np.where(s == 3, np.nan, vc), np.where(s == 3, np.nan, vt)
    return dict(units=units,
                val=keys.assign(J_classical=vc, J_tsfm=vt, crps_classical=vc, crps_tsfm=vt),
                test=keys.assign(J_classical=1.0, J_tsfm=3.0, scale=2.0, n_origins=4),
                n_val=keys[SERIES_KEYS].drop_duplicates().assign(n_validation_origins=2))


def _run_seven(t, d_omega):
    return run_selectors(seven_selectors(THRESHOLD, d_omega), t["units"], t["val"], t["test"],
                         REPRESENTATIVES, t["n_val"])


class DeferralsArePricedAtPinballValidation(unittest.TestCase):
    """Pilot decision 8. Runs in every environment (a stub defers)."""

    def setUp(self):
        self.t = _pricing_tables()
        self.res = _run_seven(self.t, _DeferSeries({"ev/s0", "ev/s2", "ev/s3"}))
        self.c = charge_deferrals(self.res)

    def test_a_deferred_unit_is_charged_pinball_validations_cost_for_the_same_unit(self):
        mine = self.c[self.c.selector == "d_omega"].set_index(UNIT_KEYS)
        pv = self.res[self.res.selector == "pinball_validation"].set_index(UNIT_KEYS)
        for key, r in mine.iterrows():
            with self.subTest(unit=key):
                if r.status == "selected":
                    self.assertFalse(r.deferred)
                    self.assertEqual(r.charged_cost, r.realized_cost)
                    self.assertIsNone(r.deferral_family)
                    self.assertTrue(np.isnan(r.deferral_cost))
                elif pv.loc[key, "status"] == "selected":
                    self.assertTrue(r.deferred)
                    self.assertEqual(r.deferral_family, pv.loc[key, "chosen_family"])
                    self.assertEqual(r.deferral_cost, pv.loc[key, "realized_cost"])
                    self.assertEqual(r.charged_cost, pv.loc[key, "realized_cost"])
                    self.assertEqual(r.charge_note, "deferral charged at pinball_validation")
        s0 = mine.xs("ev/s0", level="series_id")
        self.assertEqual(s0.deferral_family.tolist(), ["tsfm", "tsfm"])          # validation's (wrong) choice
        self.assertEqual(s0.charged_cost.tolist(), [3.0, 3.0])                   # its test cost, not the oracle's 1
        self.assertEqual(s0.charged_regret.tolist(), [2.0, 2.0])
        self.assertEqual(s0.charged_scaled_regret.tolist(), [1.0, 1.0])
        s2 = mine.xs("ev/s2", level="series_id")
        self.assertEqual((s2.deferral_family.tolist(), s2.charged_cost.tolist()), (["classical"] * 2, [1.0, 1.0]))

    def test_deferred_units_are_never_dropped(self):
        self.assertEqual(len(self.c), len(self.res))
        pd.testing.assert_frame_equal(self.c[list(self.res.columns)], self.res.reset_index(drop=True))
        self.assertEqual(list(self.c.columns), list(self.res.columns) + list(CHARGE_COLUMNS))
        mine = self.c[self.c.selector == "d_omega"]
        self.assertEqual(int(mine.deferred.sum()), 6)
        self.assertTrue((mine.deferred == (mine.status == "defer")).all())
        s = summarize_charged(self.res).set_index(["selector", "horizon", "tau"])
        for tau in (0.5, 0.91):
            row = s.loc[("d_omega", 1, tau)]
            self.assertEqual((row.n_units, row.n_selected, row.n_deferred, row.n_not_charged), (4, 1, 3, 1))
            self.assertEqual(row.deferral_rate, 0.75)

    def test_an_unpriceable_deferral_is_flagged_and_the_group_is_not_reported_as_complete(self):
        mine = self.c[self.c.selector == "d_omega"]
        s3 = mine[mine.series_id == "ev/s3"]
        self.assertTrue(s3.deferred.all())
        self.assertTrue(s3.charged_cost.isna().all())
        self.assertTrue(s3.deferral_family.isna().all())
        self.assertTrue((s3.charge_note == "deferral unpriced: pinball_validation made no choice for this unit").all())
        row = summarize_charged(self.res).set_index(["selector", "horizon", "tau"]).loc[("d_omega", 1, 0.5)]
        for col in ("mean_scaled_cost_charged", "mean_scaled_regret_charged", "frac_oracle_value_charged"):
            self.assertTrue(np.isnan(row[col]), col)
        self.assertEqual(row.frac_note_charged, "undefined: 1 unit(s) not charged")

    def test_charged_numbers_by_hand_when_every_deferral_is_priced(self):
        t = _pricing_tables(val_unknown_s3=False)
        res = _run_seven(t, _DeferSeries({"ev/s0", "ev/s2", "ev/s3"}))
        row = summarize_charged(res).set_index(["selector", "horizon", "tau"]).loc[("d_omega", 1, 0.5)]
        # charged: s0 3 (tsfm), s1 1 (selected classical), s2 1, s3 1; scale 2; oracle 1 everywhere
        self.assertEqual(row.n_not_charged, 0)
        self.assertAlmostEqual(row.mean_scaled_cost_charged, (3 + 1 + 1 + 1) / 2 / 4, places=15)
        self.assertAlmostEqual(row.mean_scaled_regret_charged, (2 / 2) / 4, places=15)
        self.assertEqual(row.mean_scaled_regret, 0.0)                         # the decided unit alone
        self.assertTrue(np.isnan(row.frac_oracle_value_charged))             # classical is best-fixed and oracle
        self.assertEqual(row.frac_note_charged, "undefined: best-fixed equals oracle")
        self.assertEqual(row.deferral_pricing, DEFERRAL_PRICING)

    def test_summarize_refuses_a_frame_with_deferrals(self):
        with self.assertRaisesRegex(ValueError, "6 deferred row.*summarize_charged"):
            summarize(self.res)

    def test_without_deferrals_the_charged_summary_adds_columns_and_changes_nothing(self):
        base = _shared.results()[0]
        plain = summarize(base)
        charged = summarize_charged(base)
        pd.testing.assert_frame_equal(charged[list(plain.columns)], plain)
        self.assertTrue((charged.n_deferred == 0).all())
        self.assertTrue((charged.deferral_rate == 0).all())
        np.testing.assert_allclose(charged.mean_scaled_regret_charged, charged.mean_scaled_regret, rtol=0, atol=1e-15)
        ok = plain.frac_oracle_value.notna()
        np.testing.assert_allclose(charged.frac_oracle_value_charged[ok], plain.frac_oracle_value[ok], rtol=1e-12)
        orc = charged[charged.selector == "hindsight_oracle"]
        self.assertTrue((orc.mean_scaled_regret_charged == 0).all())


class OraclePricingCannotBeProducedSilently(unittest.TestCase):
    """Pilot decision 8: the oracle price is an upper bound and is never what the engine charges."""

    def test_there_is_no_option_to_choose_the_price(self):
        self.assertEqual(list(inspect.signature(charge_deferrals).parameters), ["res"])
        self.assertEqual(list(inspect.signature(summarize_charged).parameters), ["res", "by"])
        self.assertNotIn("deferral", " ".join(inspect.signature(build_k3_d_omega).parameters))
        self.assertEqual(DEFERRAL_PRICING, "validation_comparator")

    def test_without_pinball_validation_rows_there_is_no_fallback(self):
        t = _pricing_tables(val_unknown_s3=False)
        res = run_selectors([_DeferSeries({"ev/s0"})], t["units"], t["val"], t["test"], REPRESENTATIVES, t["n_val"])
        with self.assertRaisesRegex(ValueError, "never at the oracle"):
            charge_deferrals(res)
        with self.assertRaisesRegex(ValueError, "never at the oracle"):
            summarize_charged(res)

    def test_a_deferred_row_that_already_carries_a_cost_is_refused(self):
        """A frame whose deferrals were priced upstream (for example at the oracle) is not schema-valid."""
        res = _run_seven(_pricing_tables(val_unknown_s3=False), _DeferSeries({"ev/s0"}))
        bad = res.copy()
        d = bad.status == "defer"
        bad.loc[d, "realized_cost"] = bad.loc[d, "oracle_cost"]
        with self.assertRaisesRegex(ValueError, "must not carry a family or a cost"):
            charge_deferrals(bad)

    def test_the_charge_is_not_the_oracle_cost_and_every_row_names_its_pricing(self):
        t = _pricing_tables(val_unknown_s3=False)
        c = charge_deferrals(_run_seven(t, _DeferSeries({"ev/s0", "ev/s1"})))
        d = c[c.deferred]
        self.assertTrue((d.charged_cost > d.oracle_cost).all())               # validation chose wrongly here
        self.assertTrue((c.deferral_pricing == "validation_comparator").all())
        self.assertFalse((c.deferral_pricing == "oracle_upper_bound").any())


@needs_k3
class AgreesWithHisEvaluationOneTauAtATime(unittest.TestCase):
    """Pilot decisions 8 and 9: per (horizon, tau), the engine's charged numbers
    equal his evaluate_selector given scaled costs and deferral_cost from
    pinball_validation; and differ from his oracle-priced upper bound."""

    def setUp(self):
        omegas = [0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125]
        units = _keys("ev", omegas)
        keys = units[UNIT_KEYS]
        i = keys.series_id.str[-1].astype(int).to_numpy()
        below = units.omega.to_numpy() <= 0.5
        classical_cheaper = np.where(units.tau.to_numpy() == 0.5, below, ~below)
        scale = 1.0 + i
        jc = np.where(classical_cheaper, 1.0, 2.0 + 0.25 * i) * scale
        jt = np.where(classical_cheaper, 2.0 + 0.5 * i, 1.0) * scale
        jt[(i == 0) & (units.tau.to_numpy() == 0.91)] = 0.5                # best-fixed is not the oracle
        vwrong = (i == 1) | (i == 3)                                          # validation picks wrongly on s1, s3
        vc = np.where(classical_cheaper ^ vwrong, 0.5, 1.0)
        vt = np.where(classical_cheaper ^ vwrong, 1.0, 0.5)
        self.t = dict(units=units,
                      val=keys.assign(J_classical=vc, J_tsfm=vt, crps_classical=vc, crps_tsfm=vt),
                      test=keys.assign(J_classical=jc, J_tsfm=jt, scale=scale, n_origins=4),
                      n_val=keys[SERIES_KEYS].drop_duplicates().assign(n_validation_origins=2))
        self.sel = K3SelectorAdapter.fit(_pool(), margin=0.2)
        self.res = _run_seven(self.t, self.sel)

    def test_each_tau_matches_his_evaluate_selector_with_the_validation_price(self):
        s = summarize_charged(self.res).set_index(["selector", "horizon", "tau"])
        c = charge_deferrals(self.res)
        pv = c[c.selector == "pinball_validation"].set_index(UNIT_KEYS)
        test = self.t["test"].set_index(UNIT_KEYS)
        n_def = 0
        for tau in (0.5, 0.91):
            with self.subTest(tau=tau):
                u = self.t["units"][self.t["units"].tau == tau]
                k = [tuple(r) for r in u[UNIT_KEYS].itertuples(index=False)]
                sc = test.loc[k, "scale"].to_numpy()
                jc, jt = test.loc[k, "J_classical"].to_numpy() / sc, test.loc[k, "J_tsfm"].to_numpy() / sc
                price = pv.loc[k, "realized_cost"].to_numpy() / sc
                his = k3.evaluate_selector(self.sel.specs[1], u.omega.tolist(), u.tau.to_numpy(), jc, jt,
                                           u.dataset.to_numpy(dtype=object), margin=0.2, deferral_cost=price)
                row = s.loc[("d_omega", 1, tau)]
                self.assertEqual(his["deferral_pricing"], DEFERRAL_PRICING)
                self.assertEqual((his["deferrals"], his["deferral_rate"]), (row.n_deferred, row.deferral_rate))
                self.assertAlmostEqual(his["realized_cost"] / his["n"], row.mean_scaled_cost_charged, places=12)
                self.assertGreater(his["best_fixed_cost"], his["oracle_cost"])
                self.assertAlmostEqual(his["fraction_of_oracle_value_captured"], row.frac_oracle_value_charged,
                                       places=12)
                bound = k3.evaluate_selector(self.sel.specs[1], u.omega.tolist(), u.tau.to_numpy(), jc, jt,
                                             u.dataset.to_numpy(dtype=object), margin=0.2)
                self.assertEqual(bound["deferral_pricing"], "oracle_upper_bound")
                self.assertLess(bound["realized_cost"], his["realized_cost"])  # the bound flatters a deferral
                n_def += his["deferrals"]
        self.assertGreater(n_def, 0)

    def test_per_tau_numbers_differ_from_one_pooled_over_taus(self):
        by_tau = summarize_charged(self.res).set_index(["selector", "horizon", "tau"])
        pooled = summarize_charged(self.res, by=("selector", "horizon")).set_index(["selector", "horizon"])
        fr = [by_tau.loc[("d_omega", 1, t), "frac_oracle_value_charged"] for t in (0.5, 0.91)]
        self.assertNotAlmostEqual(fr[0], fr[1], places=6)
        self.assertEqual(len(by_tau.loc["d_omega"]), 2)
        self.assertEqual(pooled.loc[("d_omega", 1), "n_units"], 12)


@needs_k3
class FittedOnScaledCost(unittest.TestCase):
    """Pilot decision 9: one dataset with costs 1000 times larger must not move the fit."""

    def _pools(self):
        agree = _pool(datasets=("fa", "fb", "fc"))
        against = _pool(datasets=("fd",)).rename(columns={"J_classical": "J_tsfm", "J_tsfm": "J_classical"})
        big = against.copy()
        big[["J_classical", "J_tsfm", "scale"]] *= 1000.0
        return (pd.concat([agree, against], ignore_index=True)[FIT_POOL_COLUMNS],
                pd.concat([agree, big], ignore_index=True)[FIT_POOL_COLUMNS])

    def test_a_dataset_1000_times_larger_does_not_flip_the_scaled_fit(self):
        same, large = self._pools()
        ref = K3SelectorAdapter.fit(same)
        scaled = K3SelectorAdapter.fit(large)                                 # the default
        self.assertEqual(scaled.cost, "scaled")
        self.assertEqual(scaled.specs, ref.specs)
        self.assertEqual(ref.specs[1]["rules"]["0.50"]["orientation"], "classical_below")    # three datasets to one
        raw = K3SelectorAdapter.fit(large, cost="raw")
        self.assertEqual(raw.specs[1]["rules"]["0.50"]["orientation"], "tsfm_below")        # fd alone decides
        unit = _keys("ev", [0.25], taus=(0.5,))
        self.assertEqual(_select(scaled, unit).chosen_family.tolist(), ["classical"])
        self.assertEqual(_select(raw, unit).chosen_family.tolist(), ["tsfm"])

    def test_through_the_entry_point_on_the_fixture(self):
        tables = dict(_shared.tables())
        fu = tables["fit_units"].copy()
        fu[["J_classical", "J_tsfm", "scale"]] *= 1000.0
        big = dict(tables, fit_units=fu)

        def rules(d):
            return {(ds, h, k): (r["status"], r.get("orientation"), r.get("threshold"))
                    for ds, f in d.folds.items() for h, sp in f.specs.items() for k, r in sp["rules"].items()}

        self.assertEqual(rules(build_k3_d_omega(big)), rules(build_k3_d_omega(tables)))
        self.assertNotEqual(rules(build_k3_d_omega(big, cost="raw")), rules(build_k3_d_omega(tables, cost="raw")))


class TwoFloorsNamedApart(unittest.TestCase):
    """Pilot decision 6 and #61: the engine floor counts datasets, default 3; his floor is separate."""

    def test_every_engine_floor_default_is_3(self):
        self.assertEqual(DEFAULT_MIN_FIT_DATASETS, 3)
        for f in (LeaveOneDatasetOut.build, run, fit_d_omega, build_k3_d_omega):
            with self.subTest(f=f.__qualname__):
                self.assertEqual(inspect.signature(f).parameters["min_fit_datasets"].default, 3)
        self.assertEqual(inspect.signature(build_k3_d_omega).parameters["k3_min_distinct"].default, 3)
        self.assertEqual(inspect.signature(K3SelectorAdapter.fit).parameters["k3_min_distinct"].default, 3)
        self.assertNotIn("min_distinct", inspect.signature(K3SelectorAdapter.fit).parameters)

    def test_entry_point_refuses_a_floor_that_is_not_a_whole_number(self):
        for bad in (0, 2.5):
            with self.assertRaisesRegex(ValueError, "min_fit_datasets"):
                build_k3_d_omega(_shared.tables(), min_fit_datasets=bad)
        with self.assertRaisesRegex(ValueError, "fit_scheme"):
            build_k3_d_omega(_shared.tables(), fit_scheme="pooled")

    @needs_k3
    def test_entry_point_applies_the_engine_floor_then_his(self):
        ev = _keys("ev", [0.1875, 0.8125])
        t = _eval_tables(ev)
        t["fit_units"] = _pool(datasets=("fa", "fb"))
        d = build_k3_d_omega(t)
        self.assertIsInstance(d.folds["ev"], NotFitted)                      # engine floor: 2 < 3 datasets
        self.assertEqual(d.describe()["min_fit_datasets"], 3)
        d2 = build_k3_d_omega(t, min_fit_datasets=2)
        self.assertIsInstance(d2.folds["ev"], K3SelectorAdapter)
        out = d2.select(SelectorInput(units=ev, val=t["val"]))
        self.assertTrue((out.note == "fold=ev reason=insufficient_data").all())  # his floor: 2 < 3
        self.assertEqual(d2.describe()["folds"]["ev"]["k3_min_distinct"], 3)
        self.assertIn("datasets", d2.describe()["folds"]["ev"]["k3_min_distinct_counts"])

    @needs_k3
    def test_his_floor_counts_datasets_not_omega_values(self):
        """What the adapter's description says: 3 datasets sharing 2 Omega values pass his floor."""
        sel = K3SelectorAdapter.fit(_pool(datasets=("fa", "fb", "fc"), omegas=(0.25, 0.75)))
        rule = sel.specs[1]["rules"]["0.50"]
        self.assertEqual((rule["status"], rule["n_datasets"]), ("ok", 3))
        one = K3SelectorAdapter.fit(_pool(datasets=("fa", "fb", "fc"), omegas=(0.25,)))
        self.assertEqual(one.specs[1]["rules"]["0.50"]["status"], "insufficient_data")   # 1 Omega value


class EntryPointGuards(unittest.TestCase):
    def test_held_out_evaluation_units_are_refused_before_the_k3_import(self):
        for column, name in (("dataset", "PJM"), ("dataset", "m5_daily"), ("config", "M5/D")):
            with self.subTest(column=column, name=name):
                t = dict(_shared.tables())
                u = t["units"].copy()
                u.loc[u.index[:3], column] = name
                with mock.patch.dict(sys.modules, {K3_SELECTOR_MODULE: None}):
                    with self.assertRaises(HeldOutLeakError):
                        build_k3_d_omega(dict(t, units=u))

    def test_requesting_k3_without_the_module_fails_even_when_no_fold_would_be_fitted(self):
        ev = _keys("ev", [0.1875, 0.8125])
        t = dict(_eval_tables(ev), fit_units=_pool(datasets=("fa",)))       # 1 other dataset: no fold is fitted
        with mock.patch.dict(sys.modules, {K3_SELECTOR_MODULE: None}):
            with self.assertRaises(K3SelectorUnavailable):
                build_k3_d_omega(t)
            d = fit_d_omega(t)                                               # the placeholder path is unaffected
        self.assertIsInstance(d.folds["ev"], NotFitted)

    @needs_k3
    def test_held_out_fit_units_are_refused(self):
        t = dict(_shared.tables())
        fu = t["fit_units"].copy()
        fu["dataset"] = "PJM_hourly"
        with self.assertRaises(HeldOutLeakError):
            build_k3_d_omega(dict(t, fit_units=fu))

    @needs_k3
    def test_entry_point_equals_run_with_the_same_fit(self):
        d = build_k3_d_omega(_shared.tables())
        res = _run_seven(_shared.tables(), d)
        pd.testing.assert_frame_equal(res, _k3_results()[0])
        self.assertEqual(d.describe(), _k3_results()[1].describe())
        _, d_old = run(_shared.tables(), REPRESENTATIVES, THRESHOLD, fit_scheme="fit_only_datasets",
                       d_omega_fit=K3SelectorAdapter.fit)
        d_new = build_k3_d_omega(_shared.tables(), fit_scheme="fit_only_datasets")
        self.assertEqual(d_new.describe(), d_old.describe())


class WithoutScikitLearnOrTheK3Module(unittest.TestCase):
    def _cli(self, block, tmp):
        code = ("import sys\n"
                f"for m in {block!r}: sys.modules[m] = None\n"
                f"sys.argv = ['run_fixture', '--out', {tmp!r}, '--d-omega', 'k3_selector']\n"
                "from decision_aware_tsfm.analysis.decision import run_fixture\n"
                "run_fixture.main()\n"
                "print('sklearn loaded:', any(m.split('.')[0] == 'sklearn' and sys.modules[m] is not None\n"
                "                             for m in list(sys.modules)))\n")
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": _src_path()})

    def test_k3_option_without_the_k3_module_stops_with_a_clear_error_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._cli([K3_SELECTOR_MODULE], str(Path(tmp) / "run"))
            self.assertEqual(out.returncode, 2, out.stderr)
            self.assertIn("cannot import decision_aware_tsfm.analysis.k3.selector", out.stderr)
            self.assertIn("PR #54", out.stderr)
            self.assertNotIn("Traceback", out.stderr)
            self.assertFalse((Path(tmp) / "run").exists())

    @needs_k3
    def test_k3_option_runs_with_scikit_learn_blocked(self):
        """#32 imports scikit-learn inside the functions that need it; the selector needs none of them."""
        with tempfile.TemporaryDirectory() as tmp:
            out = self._cli(["sklearn"], str(Path(tmp) / "run"))
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("sklearn loaded: False", out.stdout)
            self.assertTrue((Path(tmp) / "run" / "summary_by_horizon_tau.csv").exists())

if __name__ == "__main__":
    unittest.main()
