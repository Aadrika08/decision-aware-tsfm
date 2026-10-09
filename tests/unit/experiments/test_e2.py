"""E2: fit / freeze / evaluate separation, frozen settings, held-out gate and
ledger, selector code hash, metrics.

Only the synthetic fixture and invented dataset names are used. A dataset
called 'm5_invented_for_test' below is fixture rows renamed; no real held-out
data exists in this repository or is read here.
"""
import contextlib
import hashlib
import inspect
import io
import json
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from decision_aware_tsfm.analysis.decision.evaluate import summarize
from decision_aware_tsfm.analysis.decision.fixture import (
    FAMILY_OF_MODEL, FIT_ONLY_DATASETS, FIXTURE_SCHEMA, FIXTURE_STATIC_THRESHOLD, REPRESENTATIVES)
from decision_aware_tsfm.analysis.decision.guards import FitEvalOverlapError, HeldOutLeakError
from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID
from decision_aware_tsfm.analysis.decision.pipeline import build_tables
from decision_aware_tsfm.analysis.decision.selectors import _decisions
from decision_aware_tsfm.analysis.experiments import e2

from . import _shared

KIND = "d_omega_linear_placeholder"
INVENTED_HELD_OUT = "m5_invented_for_test"
EVAL_FRAMES = ("units", "val", "test", "n_val")


@lru_cache(maxsize=None)
def _base_spec():
    return e2.fit(_shared.tables(), selector_kind=KIND, static_threshold=FIXTURE_STATIC_THRESHOLD)


_BASE_DIR = tempfile.TemporaryDirectory()        # removed when the test process ends


@lru_cache(maxsize=None)
def _base_result():
    """(frozen specification, its evaluation on the fixture), computed once."""
    path = Path(_BASE_DIR.name) / "frozen.json"
    e2.freeze(_base_spec(), path)
    frozen = e2.load_frozen(path)
    return frozen, e2.evaluate(frozen, _shared.tables())


def _fit(tables=None, fit_units=None, **kw):
    if tables is None and fit_units is None and not kw:
        return _base_spec()
    t = _shared.tables() if tables is None else tables
    if fit_units is not None:
        t = dict(t, fit_units=fit_units)
    args = dict(selector_kind=KIND, static_threshold=FIXTURE_STATIC_THRESHOLD)
    args.update(kw)
    return e2.fit(t, **args)


@lru_cache(maxsize=None)
def _h24_tables(**build_kw):
    """Horizon-24 fixture tables built with other settings (small, so quick to build)."""
    fc, omega, _ = _shared.fixture_data()
    kw = dict(representatives=REPRESENTATIVES, omega=omega)
    kw.update(build_kw)
    reps, om = kw.pop("representatives"), kw.pop("omega")
    return build_tables(fc[fc.horizon == 24], FIXTURE_SCHEMA, dict(reps), FAMILY_OF_MODEL, om,
                        fit_only_datasets=FIT_ONLY_DATASETS, **kw)


def _forge(payload) -> dict:
    """A frozen document with a CORRECT hash for any payload, using only the standard library."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {"payload": payload, "sha256": hashlib.sha256(text.encode()).hexdigest()}


def _select_all(family):
    def select(self, inp):
        return _decisions(inp.units, np.full(len(inp.units), family, dtype=object))
    return select


class _WithFrozen(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.spec = _fit()
        self.path = self.dir / "frozen.json"
        self.sha = e2.freeze(self.spec, self.path)
        self.frozen = e2.load_frozen(self.path)
        self.tables = _shared.tables()

    def _register(self, adapter):
        previous = e2.SELECTOR_KINDS.get(adapter.kind)
        e2.register_selector_kind(adapter)
        if previous is None:
            self.addCleanup(e2.SELECTOR_KINDS.pop, adapter.kind, None)
        else:
            self.addCleanup(e2.SELECTOR_KINDS.__setitem__, adapter.kind, previous)

    def _freeze_kind(self, kind, name, file="custom.json", **kw):
        path = self.dir / file
        e2.freeze(_fit(selector_kind=kind, selector_name=name, **kw), path)
        return e2.load_frozen(path)

    def _forged_file(self, mutate, name="forged.json"):
        p = self.spec.payload
        mutate(p)
        f = self.dir / name
        f.write_text(json.dumps(_forge(p)), encoding="utf-8")
        return f


class FitAndFreeze(_WithFrozen):
    def test_spec_contains_everything_needed_to_select(self):
        p = self.frozen.payload
        self.assertEqual(set(p), set(e2.PAYLOAD_KEYS))
        self.assertEqual(p["schema"], "e2-frozen-v2")
        self.assertEqual(p["selector_kind"], KIND)
        self.assertEqual(set(p["selector_params"]), {"omega0", "beta"})
        self.assertEqual(p["static_threshold"], FIXTURE_STATIC_THRESHOLD)
        self.assertEqual(p["fit_datasets"], ["fx_dfit"])
        fu = self.tables["fit_units"]
        self.assertEqual(len(p["fit_series"]), fu.series_id.nunique())
        self.assertEqual({s[2] for s in p["fit_series"]}, set(fu.series_id))
        self.assertEqual(p["n_fit_units"], len(fu))
        self.assertRegex(p["selector_code_sha256"], r"^[0-9a-f]{64}$")

    def test_settings_are_recorded_from_the_fit_tables_not_typed_in(self):
        self.assertEqual(list(inspect.signature(e2.fit).parameters),
                         ["fit_tables", "selector_kind", "static_threshold", "selector_name"])
        p = self.frozen.payload
        self.assertEqual(p["representatives"], REPRESENTATIVES)
        self.assertEqual(p["tau_grid"], list(TAU_GRID))
        self.assertEqual(p["omega_col"], "omega_levels")
        self.assertIsNone(p["omega_variant"])
        self.assertEqual(p["val_fraction"], 1 / 3)
        self.assertEqual(p["crps_method"], "trapezoid")
        self.assertEqual(p["cache_schema_version"], FIXTURE_SCHEMA)
        other = _fit(_h24_tables(representatives=tuple({"classical": "SeasonalNaive",
                                                        "tsfm": "Moirai-small"}.items()),
                                 val_fraction=0.6, crps_method="uniform")).payload
        self.assertEqual(other["representatives"], {"classical": "SeasonalNaive", "tsfm": "Moirai-small"})
        self.assertEqual((other["val_fraction"], other["crps_method"]), (0.6, "uniform"))

    def test_fit_refuses_tables_without_a_settings_record(self):
        t = {k: v for k, v in self.tables.items() if k != "settings"}
        with self.assertRaises(e2.SettingsMismatchError):
            _fit(t)
        broken = dict(self.tables, settings=dict(self.tables["settings"], models_scored=("AutoETS",)))
        with self.assertRaises(e2.SettingsMismatchError):
            _fit(broken)
        with self.assertRaises(TypeError):
            e2.fit(self.tables["fit_units"], selector_kind=KIND, static_threshold=0.45,
                   representatives=REPRESENTATIVES, omega_col="omega_levels")      # the old call

    def test_placeholder_recovers_planted_boundary_of_fit_fixture(self):
        # fx_dfit is built with classical winning at omega <= 0.40 and TSFM at >= 0.50
        p = self.frozen.payload["selector_params"]
        self.assertGreater(p["omega0"], 0.40)
        self.assertLess(p["omega0"], 0.50)

    def test_hash_round_trip(self):
        self.assertEqual(self.frozen.sha256, self.sha)
        self.assertEqual(self.frozen.sha256, self.spec.sha256)
        self.assertEqual(self.frozen.payload, self.spec.payload)
        self.assertEqual(len(self.sha), 64)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["sha256"], self.sha)

    def test_freeze_does_not_overwrite(self):
        with self.assertRaises(FileExistsError):
            e2.freeze(self.spec, self.path)

    def test_freeze_takes_only_a_spec_made_by_fit(self):
        text = e2._canonical(self.spec.payload)
        with self.assertRaises(e2.NotFrozenError):
            e2.FitSpec(text)                                    # no public way to build one by hand
        for thing in (self.spec.payload, text, self.frozen, None):
            with self.subTest(given=type(thing).__name__):
                with self.assertRaises(TypeError):
                    e2.freeze(thing, self.dir / "x.json")
        with self.assertRaises(AttributeError):
            self.spec._text = "{}"

    def test_fit_is_deterministic(self):
        again = e2.fit(self.tables, selector_kind=KIND, static_threshold=FIXTURE_STATIC_THRESHOLD)
        self.assertIsNot(again, self.spec)
        self.assertEqual(again.sha256, self.spec.sha256)

    def test_fit_refuses_held_out_names(self):
        fu = self.tables["fit_units"]
        cases = {
            "dataset m5": fu.assign(dataset="m5_invented"),
            "dataset PJM upper case": fu.assign(dataset="Invented_PJM_load"),
            "dataset full-width letters": fu.assign(dataset="\uff2d\uff15_invented"),
            "config": fu.assign(config="pjm/H"),
            "series id": fu.assign(series_id="m5_invented/" + fu.series_id),
            "one row only": fu.assign(dataset=np.where(np.arange(len(fu)) == 3, "x_m5", fu.dataset)),
        }
        for name, bad in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(HeldOutLeakError):
                    _fit(fit_units=bad)

    def test_fit_has_no_held_out_switch(self):
        with self.assertRaises(TypeError):
            _fit(allow_held_out=True)

    def test_fit_checks_the_tau_grid_per_series_and_horizon(self):
        fu = self.tables["fit_units"]
        cases = {
            "tau missing everywhere": fu[fu.tau != 0.95],
            "tau missing at one horizon only": fu[~((fu.tau == 0.99) & (fu.horizon == 24))],
            "tau missing for one series": fu[~((fu.tau == 0.5) & (fu.series_id == fu.series_id.iloc[0]))],
            "extra tau 0.3": pd.concat([fu, fu[fu.tau == 0.5].assign(tau=0.3)], ignore_index=True),
        }
        for name, bad in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(e2.TauGridError):
                    _fit(fit_units=bad)
        with self.assertRaises(ValueError):
            _fit(selector_kind="no_such_kind")


class TamperAndFrozenOnly(_WithFrozen):
    def _rewrite(self, mutate):
        doc = json.loads(self.path.read_text(encoding="utf-8"))
        mutate(doc)
        self.path.write_text(json.dumps(doc), encoding="utf-8")

    def test_tampered_file_is_refused(self):
        def beta(d): d["payload"]["selector_params"]["beta"] = 0.25
        def omega0(d): d["payload"]["selector_params"]["omega0"] += 1e-9
        def threshold(d): d["payload"]["static_threshold"] = 0.5
        def fit_list(d): d["payload"]["fit_series"] = d["payload"]["fit_series"][1:]
        def reps(d): d["payload"]["representatives"]["tsfm"] = "another-model"
        def taus(d): d["payload"]["tau_grid"] = d["payload"]["tau_grid"][:-1]
        def code(d): d["payload"]["selector_code_sha256"] = "0" * 64
        def the_hash(d): d["sha256"] = "0" * 64
        def extra_key(d): d["note"] = "edited"
        original = self.path.read_text(encoding="utf-8")
        for mutate in (beta, omega0, threshold, fit_list, reps, taus, code, the_hash, extra_key):
            with self.subTest(change=mutate.__name__):
                self.path.write_text(original, encoding="utf-8")
                self._rewrite(mutate)
                with self.assertRaises(e2.FrozenFileError):
                    e2.load_frozen(self.path)
        self.path.write_text(original, encoding="utf-8")
        self.assertEqual(e2.load_frozen(self.path).sha256, self.sha)

    def test_reformatted_but_unchanged_file_still_loads(self):
        self._rewrite(lambda d: None)             # different whitespace, same content
        self.assertEqual(e2.load_frozen(self.path).sha256, self.sha)

    def test_broken_or_missing_file_is_refused(self):
        with self.assertRaises(e2.FrozenFileError):
            e2.load_frozen(self.dir / "absent.json")
        bad = self.dir / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with self.assertRaises(e2.FrozenFileError):
            e2.load_frozen(bad)

    def test_file_with_a_correct_hash_naming_held_out_fit_data_is_refused(self):
        """Each list is checked on its own: datasets, configs and series ids."""
        def datasets(p):
            p["fit_datasets"] = ["m5_invented"]
            p["fit_series"] = [["m5_invented", s[1], s[2]] for s in p["fit_series"]]

        def configs(p):
            p["fit_configs"] = ["pjm_invented"]
            p["fit_series"] = [[s[0], "pjm_invented", s[2]] for s in p["fit_series"]]

        def series(p):
            p["fit_series"] = [[s[0], s[1], "m5_invented/" + s[2].split("/")[1]] for s in p["fit_series"]]

        for mutate in (datasets, configs, series):
            with self.subTest(held_out_in=mutate.__name__):
                f = self._forged_file(mutate, f"{mutate.__name__}.json")
                with self.assertRaises(HeldOutLeakError):
                    e2.load_frozen(f)

    def test_file_with_a_correct_hash_but_an_invalid_specification_is_refused(self):
        def empty_fit_lists(p): p.update(fit_datasets=[], fit_configs=[], fit_series=[])
        def no_units(p): p["n_fit_units"] = 0
        def lists_disagree(p): p["fit_datasets"] = ["nothing"]
        def old_schema(p): p["schema"] = "e2-frozen-v1"
        def baseline_name(p): p["selector_name"] = "hindsight_oracle"
        def val_fraction(p): p["val_fraction"] = 1.5
        for mutate in (empty_fit_lists, no_units, lists_disagree, old_schema, baseline_name, val_fraction):
            with self.subTest(change=mutate.__name__):
                f = self._forged_file(mutate, f"{mutate.__name__}.json")
                with self.assertRaises(e2.FrozenFileError):
                    e2.load_frozen(f)

    def test_edited_parameters_with_a_recomputed_hash_are_accepted(self):
        """Documents the limit stated in the module docstring: the hash has no
        key, so it catches accidents and nothing else."""
        f = self._forged_file(lambda p: p["selector_params"].update(omega0=0.2))
        forged = e2.load_frozen(f)
        self.assertEqual(forged.payload["selector_params"]["omega0"], 0.2)
        self.assertNotEqual(forged.sha256, self.sha)
        r = e2.evaluate(forged, self.tables)
        self.assertEqual(r.frozen_sha256, forged.sha256)       # the result names the file that was used
        self.assertIn("not a security control", e2.__doc__)
        self.assertNotIn("cannot be skipped", " ".join(e2.__doc__.split()))

    def test_evaluate_cannot_run_without_a_frozen_file(self):
        for name, thing in (("FitSpec", self.spec), ("dict", self.spec.payload),
                            ("path", self.path), ("None", None)):
            with self.subTest(given=name):
                with self.assertRaises(e2.NotFrozenError):
                    e2.evaluate(thing, self.tables)
        with self.assertRaises(e2.NotFrozenError):
            e2.FrozenSpec("{}", "0" * 64, self.path)        # no public way to build one by hand

    def test_evaluate_refuses_when_file_is_deleted_or_edited_after_loading(self):
        original = self.path.read_text(encoding="utf-8")
        self._rewrite(lambda d: d["payload"]["selector_params"].update(beta=0.5))
        with self.assertRaises(e2.FrozenFileError):
            e2.evaluate(self.frozen, self.tables)
        self.path.write_text(original, encoding="utf-8")
        self.assertEqual(e2.evaluate(self.frozen, self.tables).frozen_sha256, self.sha)
        self.path.unlink()
        with self.assertRaises(e2.FrozenFileError):
            e2.evaluate(self.frozen, self.tables)

    def test_loaded_spec_cannot_be_modified(self):
        with self.assertRaises(AttributeError):
            self.frozen._text = "{}"
        with self.assertRaises(AttributeError):
            self.frozen.sha256 = "x"
        p = self.frozen.payload
        p["selector_params"]["beta"] = 99.0                 # edits a copy only
        self.assertNotEqual(self.frozen.payload["selector_params"]["beta"], 99.0)

    def test_evaluate_has_no_fitting_inputs(self):
        self.assertEqual(list(inspect.signature(e2.evaluate).parameters),
                         ["frozen", "tables", "allow_held_out", "ledger_path"])


class FrozenSettings(_WithFrozen):
    """Fix 3. Evaluation tables must have been built the way the frozen file says."""

    def test_reviewer_tables_with_another_classical_representative_are_refused(self):
        """Frozen: classical = AutoETS. Tables built with classical =
        SeasonalNaive were accepted, labelled AutoETS, and scored SeasonalNaive."""
        t = _h24_tables(representatives=tuple({"classical": "SeasonalNaive", "tsfm": "Moirai-small"}.items()))
        self.assertEqual(t["settings"]["representatives"]["classical"], "SeasonalNaive")
        with self.assertRaises(e2.SettingsMismatchError) as cm:
            e2.evaluate(self.frozen, t)
        self.assertIn("representatives", str(cm.exception))
        self.assertIn("SeasonalNaive", str(cm.exception))

    def test_every_frozen_setting_is_enforced(self):
        fc, omega, _ = _shared.fixture_data()
        other = omega.assign(other=1 - omega.omega_levels)
        variants = pd.concat([omega.assign(variant="full"), omega.assign(variant="last_4096")],
                             ignore_index=True)
        cases = {
            "val_fraction": _h24_tables(val_fraction=0.6),
            "omega_col": build_tables(fc[fc.horizon == 24], FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL,
                                      other, fit_only_datasets=FIT_ONLY_DATASETS, omega_col="other"),
            "omega_variant": build_tables(fc[fc.horizon == 24], FIXTURE_SCHEMA, REPRESENTATIVES,
                                          FAMILY_OF_MODEL, variants, fit_only_datasets=FIT_ONLY_DATASETS,
                                          omega_variant="last_4096"),
            "crps_method": _h24_tables(crps_method="uniform"),
        }
        for key, t in cases.items():
            with self.subTest(setting=key):
                with self.assertRaises(e2.SettingsMismatchError) as cm:
                    e2.evaluate(self.frozen, t)
                self.assertIn(key, str(cm.exception))
        same = self.tables["settings"]
        for key, value in (("taus", same["taus"][:-1]), ("taus", same["taus"] + (0.999,)),
                           ("models_scored", ("AutoETS", "Moirai-small", "SeasonalNaive")),
                           ("models_scored", ("AutoETS",)),
                           ("representatives", {"classical": "AutoETS"})):
            with self.subTest(setting=key, value=value):
                with self.assertRaises(e2.SettingsMismatchError):
                    e2.evaluate(self.frozen, dict(self.tables, settings=dict(same, **{key: value})))

    def test_tables_without_a_settings_record_are_refused(self):
        t = {k: v for k, v in self.tables.items() if k != "settings"}
        with self.assertRaises(e2.SettingsMismatchError) as cm:
            e2.evaluate(self.frozen, t)
        self.assertIn("settings", str(cm.exception))
        for missing in ("crps_method", "models_scored"):
            s = {k: v for k, v in self.tables["settings"].items() if k != missing}
            with self.subTest(missing=missing):
                with self.assertRaises(e2.SettingsMismatchError):
                    e2.evaluate(self.frozen, dict(self.tables, settings=s))

    def test_matching_tables_are_accepted_and_the_settings_are_returned(self):
        r = e2.evaluate(self.frozen, self.tables)
        self.assertEqual(r.settings["representatives"], REPRESENTATIVES)
        self.assertEqual(set(r.selection.chosen_model.dropna()), set(REPRESENTATIVES.values()))
        # the engine's own schema version is recorded but may differ (another cache layout)
        e2.evaluate(self.frozen, dict(self.tables, settings=dict(self.tables["settings"],
                                                                 schema_version="producer-v1")))


class Separation(_WithFrozen):
    def test_evaluate_refuses_datasets_used_in_fit(self):
        with self.assertRaises(FitEvalOverlapError) as cm:
            e2.evaluate(self.frozen, _shared.tables_without_fit_split())
        self.assertIn("fx_dfit", str(cm.exception))

    def test_dataset_name_check_works_without_the_series_check(self):
        """The fit dataset under its own name but with series ids nobody has
        seen: only the dataset-name check can refuse this."""
        t = dict(_shared.tables_without_fit_split())
        for k in _shared.FRAME_KEYS:
            f = t[k].copy()
            hit = f["dataset"] == "fx_dfit"
            f.loc[hit, "series_id"] = f.loc[hit, "series_id"].str.replace("fx_dfit/", "unseen/", regex=False)
            t[k] = f
        with self.assertRaises(FitEvalOverlapError) as cm:
            e2.evaluate(self.frozen, t)
        self.assertIn("datasets", str(cm.exception))
        self.assertNotIn("series", str(cm.exception))

    def test_evaluate_refuses_series_used_in_fit_under_another_dataset_name(self):
        t = _shared.renamed(_shared.tables_without_fit_split(), "fx_dfit", "fx_other",
                            rename_series=False)
        with self.assertRaises(FitEvalOverlapError) as cm:
            e2.evaluate(self.frozen, t)
        self.assertIn("series", str(cm.exception))
        self.assertNotIn("datasets", str(cm.exception))
        # one overlapping series is enough
        one = self.tables["fit_units"].series_id.iloc[0]
        t = _shared.tables_without_fit_split()
        keep = dict(t)
        for k in EVAL_FRAMES:
            f = t[k]
            keep[k] = f[(f.dataset != "fx_dfit") | (f.series_id == one)].assign(
                dataset=lambda d: d.dataset.replace("fx_dfit", "fx_other"))
        with self.assertRaises(FitEvalOverlapError) as cm:
            e2.evaluate(self.frozen, keep)
        self.assertIn("1 evaluation series", str(cm.exception))

    def test_recapitalised_or_padded_names_are_still_recognised(self):
        def recase(f):
            out = dict(_shared.tables_without_fit_split())
            for k in _shared.FRAME_KEYS:
                x = out[k].copy()
                hit = x.dataset == "fx_dfit"
                x.loc[hit, "dataset"] = x.loc[hit, "dataset"].map(f)
                x.loc[hit, "series_id"] = x.loc[hit, "series_id"].map(f)
                out[k] = x
            return out
        for name, f in (("upper case", str.upper), ("trailing space", lambda s: s + " "),
                        ("leading tab", lambda s: "\t" + s)):
            with self.subTest(change=name):
                with self.assertRaises(FitEvalOverlapError):
                    e2.evaluate(self.frozen, recase(f))
        self.assertEqual(e2.normalize_name(" FX_Dfit\t"), "fx_dfit")

    def test_fully_renamed_copy_of_fit_data_is_not_caught(self):
        """Documents the limit of the check: it compares names, not values."""
        t = _shared.renamed(_shared.tables_without_fit_split(), "fx_dfit", "fx_other")
        r = e2.evaluate(self.frozen, t)
        fit_rows = r.unit_regret[r.unit_regret.dataset == "fx_other"]
        self.assertEqual(len(fit_rows), 12 * 11)               # the 12 fit series were evaluated (horizon 24)
        self.assertEqual({s.replace("fx_other/", "fx_dfit/") for s in fit_rows.series_id},
                         set(self.tables["fit_units"].series_id))
        self.assertIn("not detected", e2.__doc__)

    def test_tau_grid_is_checked_for_every_series_at_every_horizon(self):
        def without(mask_fn, frames=("units", "val", "test")):
            t = dict(self.tables)
            for k in frames:
                t[k] = self.tables[k][~mask_fn(self.tables[k])]
            return t
        extra = dict(self.tables)
        for k in ("units", "val", "test"):
            extra[k] = pd.concat([self.tables[k], self.tables[k][self.tables[k].tau == 0.5].assign(tau=0.3)],
                                 ignore_index=True)
        cases = {
            "tau missing everywhere": without(lambda f: f.tau == 0.99),
            "the reviewer's case: tau 0.99 missing at horizon 24 only":
                without(lambda f: (f.tau == 0.99) & (f.horizon == 24)),
            "tau missing for one series": without(lambda f: (f.tau == 0.5) & (f.series_id == "fx_tie/s0")),
            "tau missing in the test costs only": without(lambda f: (f.tau == 0.5) & (f.horizon == 1),
                                                          frames=("test",)),
            "extra tau": extra,
        }
        for name, t in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(e2.TauGridError):
                    e2.evaluate(self.frozen, t)

    def test_best_fixed_is_chosen_on_the_evaluated_units_only(self):
        """The reviewer's construction: units restricted to one dataset, test
        table left whole. The result must be the one for that dataset alone."""
        proper = e2.evaluate(self.frozen, _shared.subset(self.tables, ["fx_tau_flip"]))
        loose = dict(self.tables)
        loose["units"] = self.tables["units"][self.tables["units"].dataset == "fx_tau_flip"]
        got = e2.evaluate(self.frozen, loose)
        pd.testing.assert_frame_equal(got.selection, proper.selection)
        pd.testing.assert_frame_equal(got.regret_summary, proper.regret_summary)
        whole = e2.evaluate(self.frozen, self.tables).selection
        whole = whole[whole.dataset == "fx_tau_flip"].reset_index(drop=True)
        self.assertFalse(whole.best_fixed_family.equals(proper.selection.best_fixed_family))


class HeldOutGate(_WithFrozen):
    def setUp(self):
        super().setUp()
        self.named = _shared.renamed(self.tables, "fx_tau_flip", INVENTED_HELD_OUT)
        self.ledger = self.dir / "ledger" / "held_out.jsonl"

    def test_held_out_name_refused_without_flag(self):
        with self.assertRaises(HeldOutLeakError):
            e2.evaluate(self.frozen, self.named)
        with self.assertRaises(HeldOutLeakError):
            e2.evaluate(self.frozen, self.named, allow_held_out=False, ledger_path=self.ledger)
        only_config = dict(self.tables)
        for k in EVAL_FRAMES:
            only_config[k] = self.tables[k].assign(config="PJM_invented")
        with self.assertRaises(HeldOutLeakError):
            e2.evaluate(self.frozen, only_config)
        only_series = _shared.renamed(self.tables, "fx_tau_flip", "fx_other")
        for k in EVAL_FRAMES:
            only_series[k] = only_series[k].assign(
                series_id=lambda d: d.series_id.str.replace("fx_other", "pjm_invented"))
        with self.assertRaises(HeldOutLeakError):
            e2.evaluate(self.frozen, only_series)
        self.assertFalse(self.ledger.exists())                  # a refused evaluation is not logged

    def test_flag_must_be_the_bool_true(self):
        for value in (1, "yes", 1.0, [True], None):
            with self.subTest(allow_held_out=value):
                with self.assertRaises(TypeError):
                    e2.evaluate(self.frozen, self.named, allow_held_out=value, ledger_path=self.ledger)
        self.assertFalse(self.ledger.exists())

    def test_flag_needs_a_ledger_path(self):
        with self.assertRaises(e2.LedgerRequiredError):
            e2.evaluate(self.frozen, self.named, allow_held_out=True)
        with self.assertRaises(e2.LedgerRequiredError):
            e2.evaluate(self.frozen, self.tables, allow_held_out=True)      # even with nothing held out

    def test_flag_alone_is_not_enough_without_a_frozen_file(self):
        with self.assertRaises(e2.NotFrozenError):
            e2.evaluate(self.spec, self.named, allow_held_out=True, ledger_path=self.ledger)
        self.assertFalse(self.ledger.exists())

    def test_invented_held_out_name_evaluates_with_flag_frozen_file_and_ledger(self):
        r = e2.evaluate(self.frozen, self.named, allow_held_out=True, ledger_path=self.ledger)
        self.assertEqual(r.held_out_evaluated[0], INVENTED_HELD_OUT)
        self.assertTrue(all(INVENTED_HELD_OUT in n for n in r.held_out_evaluated))
        self.assertIn(INVENTED_HELD_OUT, set(r.selection.dataset))
        self.assertEqual(r.frozen_sha256, self.sha)
        # same rows under another name, so the costs are those of the plain run
        plain = e2.evaluate(self.frozen, self.tables)
        self.assertEqual(plain.held_out_evaluated, ())
        self.assertIsNone(plain.ledger_entry)
        a = r.unit_regret.sort_values(["series_id", "horizon", "tau"]).regret.to_numpy()
        b = (plain.unit_regret.assign(series_id=lambda d: d.series_id.str.replace(
            "fx_tau_flip", INVENTED_HELD_OUT)).sort_values(["series_id", "horizon", "tau"])
            .regret.to_numpy())
        self.assertTrue(np.array_equal(a, b))
        self.assertGreater(a.max(), 0)

    def test_every_held_out_evaluation_appends_one_ledger_line(self):
        r1 = e2.evaluate(self.frozen, self.named, allow_held_out=True, ledger_path=self.ledger)
        lines = e2.read_ledger(self.ledger)
        self.assertEqual(len(lines), 1)
        entry = lines[0]
        self.assertEqual(entry, r1.ledger_entry)
        self.assertEqual(entry["frozen_sha256"], self.sha)
        self.assertEqual(entry["selector_code_sha256"], self.frozen.payload["selector_code_sha256"])
        self.assertEqual(entry["datasets"], sorted(set(self.named["units"].dataset)))
        self.assertIn(INVENTED_HELD_OUT, entry["held_out_names"])
        self.assertEqual(entry["n_units"], len(self.named["units"]))
        self.assertRegex(entry["time_utc"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00$")
        # a second frozen file evaluated on the same held-out data leaves a second line
        other = e2.load_frozen(self._forged_file(lambda p: p["selector_params"].update(omega0=0.7)))
        e2.evaluate(other, self.named, allow_held_out=True, ledger_path=self.ledger)
        e2.evaluate(self.frozen, self.named, allow_held_out=True, ledger_path=self.ledger)
        lines = e2.read_ledger(self.ledger)
        self.assertEqual([x["frozen_sha256"] for x in lines], [self.sha, other.sha256, self.sha])
        # no held-out names present: nothing is appended, flag or no flag
        e2.evaluate(self.frozen, self.tables, allow_held_out=True, ledger_path=self.ledger)
        self.assertEqual(len(e2.read_ledger(self.ledger)), 3)

    def test_ledger_line_is_written_before_any_selector_runs(self):
        class Crash:
            name, needs_hindsight = "crash", False

            def select(self, inp):
                raise RuntimeError("selector failed")

        class Adapter:
            kind = "test_crash"

            def fit(self, fit_units):
                return {}

            def build(self, params, fit_series, fit_datasets):
                return Crash()

        self._register(Adapter())
        frozen = self._freeze_kind("test_crash", "crash")
        with self.assertRaises(RuntimeError):
            e2.evaluate(frozen, self.named, allow_held_out=True, ledger_path=self.ledger)
        self.assertEqual([x["frozen_sha256"] for x in e2.read_ledger(self.ledger)], [frozen.sha256])

    def test_fit_still_refuses_that_name(self):
        fu = self.tables["fit_units"].assign(dataset=INVENTED_HELD_OUT)
        with self.assertRaises(HeldOutLeakError):
            _fit(fit_units=fu)


def _selection(rows):
    """rows: (selector, unit, status, realized, oracle, best_fixed, scale) at horizon 1, tau 0.5."""
    d = pd.DataFrame(rows, columns=["selector", "series_id", "status", "realized_cost", "oracle_cost",
                                    "best_fixed_cost", "scale"])
    d["dataset"], d["config"], d["horizon"], d["tau"] = "hand", "c", 1, 0.5
    return d


class MetricsOnHandBuiltTables(unittest.TestCase):
    """Known answers, including units a selector did not decide."""

    def test_regret_summary_sums_over_selected_units_only(self):
        s = _selection([("s", "u1", "selected", 2.0, 1.0, 3.0, 1.0),
                        ("s", "u2", "selected", 4.0, 4.0, 4.0, 2.0),
                        ("s", "u3", "insufficient_data", np.nan, 5.0, 50.0, 1.0)])
        r = e2.regret_summary(s).iloc[0]
        # scaled sums over u1 and u2: SEL 2 + 2, ORACLE 1 + 2, BEST_FIXED 3 + 2
        self.assertEqual((r.n_units, r.n_selected), (3, 2))
        self.assertAlmostEqual(r.mean_scaled_regret, 0.5, places=15)
        self.assertAlmostEqual(r.regret_over_oracle_cost, 1 / 3, places=15)
        self.assertAlmostEqual(r.frac_oracle_value_recovered, 0.5, places=15)
        self.assertTrue(r.regret_over_oracle_cost_defined and r.frac_oracle_value_recovered_defined)
        self.assertEqual((r.regret_over_oracle_cost_reason, r.frac_oracle_value_recovered_reason), ("", ""))
        self.assertNotIn("normalized_regret", e2.regret_summary(s).columns)   # neither is given that name

    def test_each_undefined_case_has_its_own_reason(self):
        s = _selection([("none", "u1", "insufficient_data", np.nan, 1.0, 2.0, 1.0),
                        ("none", "u2", "defer", np.nan, 1.0, 2.0, 1.0),
                        ("no_gap", "u1", "selected", 2.0, 2.0, 2.0, 1.0),
                        ("no_gap", "u2", "selected", 3.0, 1.0, 1.0, 1.0),
                        ("free", "u1", "selected", 1.0, 0.0, 4.0, 2.0)])
        with np.errstate(all="raise"):
            r = e2.regret_summary(s).set_index("selector")
        none = r.loc["none"]
        self.assertEqual((none.n_units, none.n_selected), (2, 0))
        self.assertTrue(np.isnan(none.mean_scaled_regret) and np.isnan(none.regret_over_oracle_cost)
                        and np.isnan(none.frac_oracle_value_recovered))
        self.assertEqual(none.regret_over_oracle_cost_reason, "undefined: no selected units in this group")
        self.assertEqual(none.frac_oracle_value_recovered_reason, "undefined: no selected units in this group")
        gap = r.loc["no_gap"]                                   # best fixed equals the oracle in total
        self.assertAlmostEqual(gap.regret_over_oracle_cost, 2 / 3, places=15)
        self.assertFalse(gap.frac_oracle_value_recovered_defined)
        self.assertTrue(np.isnan(gap.frac_oracle_value_recovered))
        self.assertIn("best-fixed family equals the oracle", gap.frac_oracle_value_recovered_reason)
        free = r.loc["free"]                                    # the oracle pays nothing
        self.assertFalse(free.regret_over_oracle_cost_defined)
        self.assertIn("oracle cost is zero", free.regret_over_oracle_cost_reason)
        self.assertAlmostEqual(free.frac_oracle_value_recovered, (2.0 - 0.5) / 2.0, places=15)
        self.assertAlmostEqual(free.mean_scaled_regret, 0.5, places=15)

    def test_gap_to_validation_leaves_undecided_units_out(self):
        s = _selection([("a", "u1", "selected", 3.0, 1.0, 1.0, 2.0),
                        ("a", "u2", "selected", 9.0, 1.0, 1.0, 1.0),
                        ("a", "u3", "selected", 2.0, 1.0, 1.0, 2.0),
                        ("a", "u4", "insufficient_data", np.nan, 1.0, 1.0, 1.0),
                        ("b", "u1", "selected", 1.0, 1.0, 1.0, 2.0),
                        ("b", "u2", "insufficient_data", np.nan, 1.0, 1.0, 1.0),
                        ("b", "u3", "selected", 1.0, 1.0, 1.0, 2.0),
                        ("b", "u4", "selected", 7.0, 1.0, 1.0, 1.0)])
        g = e2.gap_to_validation(s, "a", reference="b").iloc[0]
        # u1: (3 - 1) / 2 = 1.0 ; u3: (2 - 1) / 2 = 0.5 ; u2 and u4 are left out, not counted as 0
        self.assertEqual((g.n_units, g.n_both_selected), (4, 2))
        self.assertAlmostEqual(g.gap_to_validation, 0.75, places=15)
        self.assertEqual((g.selector, g.reference), ("a", "b"))
        with self.assertRaises(ValueError):
            e2.gap_to_validation(s, "a", reference="absent")

    def test_worst_decile_on_hand_built_regrets(self):
        reg = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 5.0]      # 12 series
        sel = pd.DataFrame(dict(selector="s", dataset="d", config="c",
                                series_id=[f"x{i}" for i in range(12)], horizon=1, tau=0.5,
                                status="selected", scaled_regret=reg))
        w = e2.worst_decile_regret(sel, "s").iloc[0]
        self.assertEqual((w.n_series, w.n_in_worst), (12, 2))                    # ceil(1.2) = 2
        self.assertAlmostEqual(w.worst_decile_regret, (5.0 + 0.6) / 2)
        self.assertAlmostEqual(w.max_series_regret, 5.0)
        self.assertAlmostEqual(w.mean_series_regret, sum(reg) / 12)
        for n, k in ((1, 1), (9, 1), (10, 1), (11, 2), (20, 2), (21, 3)):
            with self.subTest(n_series=n):
                got = e2.worst_decile_regret(sel.iloc[:1].loc[[0] * n].assign(
                    series_id=[f"y{i}" for i in range(n)]), "s").iloc[0]
                self.assertEqual(got.n_in_worst, k)

    def test_p90_is_the_ninetieth_percentile(self):
        sel = pd.DataFrame(dict(selector="s", dataset="d", config="c",
                                series_id=[f"x{i:02d}" for i in range(11)], horizon=1, tau=0.5,
                                status="selected", scaled_regret=np.arange(11.0)))
        w = e2.worst_decile_regret(sel, "s").iloc[0]
        self.assertAlmostEqual(w.p90_series_regret, 9.0, places=12)             # the median would be 5
        self.assertAlmostEqual(w.worst_decile_regret, 9.5, places=12)           # mean of 10 and 9
        undecided = sel.assign(status=["selected"] * 10 + ["insufficient_data"])
        w = e2.worst_decile_regret(undecided, "s").iloc[0]
        self.assertEqual(w.n_series, 10)
        self.assertAlmostEqual(w.p90_series_regret, 8.1, places=12)             # 0..9, linear interpolation
        with self.assertRaises(ValueError):
            e2.worst_decile_regret(sel.assign(scaled_regret=[np.nan] + [1.0] * 10), "s")


class EvaluateAndMetrics(_WithFrozen):
    def setUp(self):
        super().setUp()
        self.r = _base_result()[1]

    def test_parameters_unchanged_by_evaluate(self):
        self.before_bytes = self.path.read_bytes()
        self.before_payload = self.frozen.payload
        e2.evaluate(self.frozen, self.tables)
        self.assertEqual(self.frozen.payload, self.before_payload)
        self.assertEqual(self.frozen.sha256, self.sha)
        self.assertEqual(self.path.read_bytes(), self.before_bytes)
        self.assertEqual(e2.load_frozen(self.path).payload, self.before_payload)

    def test_same_selection_as_the_engine_pipeline(self):
        # E2 freezes one rule fitted on the fit-only datasets, so compare with the
        # engine run under that same scheme (the engine default is leave-one-dataset-out)
        eng, _ = _shared.results(fit_scheme="fit_only_datasets")
        pd.testing.assert_frame_equal(self.r.selection, eng)

    def test_engine_default_differs_from_e2_only_in_d_omega(self):
        lodo, _ = _shared.results()
        e2sel = self.r.selection
        keep = lodo.selector != "d_omega"
        pd.testing.assert_frame_equal(lodo[keep].reset_index(drop=True),
                                      e2sel[e2sel.selector != "d_omega"].reset_index(drop=True))

    def test_on_the_fixture_the_placeholder_equals_the_static_rule(self):
        """beta is 0 and omega0 equals the fixture's static threshold, so the two
        selectors are the same rule here. The fixture cannot tell them apart."""
        p = self.frozen.payload
        self.assertEqual((p["selector_params"]["beta"], p["selector_params"]["omega0"]), (0.0, 0.45))
        s = self.r.selection
        a = s[s.selector == "d_omega"].chosen_family.to_numpy()
        b = s[s.selector == "static_omega"].chosen_family.to_numpy()
        self.assertTrue((a == b).all())

    def test_static_rule_uses_the_frozen_threshold(self):
        """Frozen with 0.60 instead of the fixture's 0.45: TSFM exactly where omega > 0.60."""
        path = self.dir / "t60.json"
        e2.freeze(_fit(static_threshold=0.60), path)
        r = e2.evaluate(e2.load_frozen(path), self.tables)
        s = r.selection[r.selection.selector == "static_omega"]
        self.assertTrue(((s.omega > 0.60) == (s.chosen_family == "tsfm")).all())
        base = self.r.selection[self.r.selection.selector == "static_omega"]
        changed = int((s.chosen_family.to_numpy() != base.chosen_family.to_numpy()).sum())
        self.assertEqual(changed, int(((base.omega > 0.45) & (base.omega <= 0.60)).sum()))
        self.assertGreater(changed, 50)
        self.assertTrue(s.note.str.contains("0.6").all())

    def test_regret_never_negative_and_oracle_zero(self):
        s = self.r.selection
        self.assertTrue((s.regret >= 0).all())
        self.assertTrue((s[s.selector == "hindsight_oracle"].regret == 0).all())
        self.assertGreater(s[s.selector == "always_classical"].regret.max(), 0.1)
        u = self.r.unit_regret
        self.assertEqual(len(u), len(self.tables["units"]))
        self.assertTrue((u.regret >= 0).all())
        self.assertTrue(np.allclose(u.regret, u.realized_cost - u.oracle_cost, rtol=0, atol=0))
        self.assertTrue(np.allclose(u.scaled_regret, u.regret / u.scale))

    def test_both_regret_figures_on_the_fixture(self):
        n = self.r.regret_summary
        self.assertEqual(len(n), 7 * 2 * 11)
        orc = n[n.selector == "hindsight_oracle"]
        self.assertTrue((orc.regret_over_oracle_cost == 0).all())
        self.assertTrue((orc.frac_oracle_value_recovered[orc.frac_oracle_value_recovered_defined] == 1).all())
        self.assertTrue((n.regret_over_oracle_cost[n.regret_over_oracle_cost_defined] >= 0).all())
        self.assertTrue((n.frac_oracle_value_recovered[n.frac_oracle_value_recovered_defined] <= 1 + 1e-12).all())
        # the fraction is the engine's, with the same undefined groups; the mean regret is the engine's
        eng = summarize(self.r.selection)
        m = n.merge(eng, on=["selector", "horizon", "tau"], validate="1:1")
        self.assertTrue((m.frac_oracle_value_recovered_defined == m.frac_oracle_value.notna()).all())
        ok = m[m.frac_oracle_value_recovered_defined]
        self.assertGreater(len(ok), 100)
        self.assertTrue(np.allclose(ok.frac_oracle_value_recovered, ok.frac_oracle_value, atol=1e-12))
        self.assertTrue(np.allclose(m.mean_scaled_regret_x, m.mean_scaled_regret_y, atol=1e-15))
        # the two figures are different quantities: independent recomputation of regret / oracle cost
        s = self.r.selection
        d = s[(s.selector == "d_omega") & (s.horizon == 24) & (s.tau == 0.09)]
        want = (d.realized_cost / d.scale).sum() / (d.oracle_cost / d.scale).sum() - 1
        row = n[(n.selector == "d_omega") & (n.horizon == 24) & (n.tau == 0.09)].iloc[0]
        self.assertAlmostEqual(row.regret_over_oracle_cost, want, places=12)
        self.assertGreater(row.regret_over_oracle_cost, 0.02)
        self.assertLess(row.frac_oracle_value_recovered, -10)

    def test_best_fixed_family_recovers_zero_of_the_oracle_value(self):
        n = self.r.regret_summary
        n = n[n.frac_oracle_value_recovered_defined & n.selector.isin(["always_classical", "always_tsfm"])]
        best = n.groupby(["horizon", "tau"]).frac_oracle_value_recovered.max()
        self.assertEqual(len(best), 22)
        self.assertTrue(np.allclose(best, 0.0, atol=1e-12))

    def test_undefined_fraction_is_reported_not_divided(self):
        # exact-tie dataset alone: best-fixed equals the oracle in every group
        with np.errstate(all="raise"):
            r = e2.evaluate(self.frozen, _shared.subset(self.tables, ["fx_tie"]))
        n = r.regret_summary
        self.assertFalse(n.frac_oracle_value_recovered_defined.any())
        self.assertTrue(n.frac_oracle_value_recovered.isna().all())
        self.assertTrue(n.frac_oracle_value_recovered_reason.str.startswith("undefined: best-fixed").all())
        self.assertTrue((n.regret_over_oracle_cost == 0).all())            # still defined, and zero
        self.assertFalse(np.isinf(n.select_dtypes("number").to_numpy()).any())
        # a dominance dataset: also undefined, and regret itself is still a number
        r = e2.evaluate(self.frozen, _shared.subset(self.tables, ["fx_classical_wins"]))
        self.assertTrue(r.regret_summary.mean_scaled_regret.notna().all())
        full = self.r.regret_summary
        self.assertTrue((full.frac_oracle_value_recovered_reason[full.frac_oracle_value_recovered_defined] == "").all())

    def test_gap_to_validation(self):
        g = self.r.gap_to_validation
        self.assertEqual(len(g), 2 * 11)
        self.assertTrue((g.reference == "pinball_validation").all())
        self.assertTrue((g.n_both_selected == 22).all())
        eng = summarize(self.r.selection).set_index(["selector", "horizon", "tau"]).mean_scaled_regret
        for row in g.itertuples():
            with self.subTest(horizon=row.horizon, tau=row.tau):
                want = (eng[("d_omega", row.horizon, row.tau)]
                        - eng[("pinball_validation", row.horizon, row.tau)])
                self.assertAlmostEqual(row.gap_to_validation, want, places=14)
        self.assertGreater(g.gap_to_validation.abs().max(), 1e-4)
        self_gap = e2.gap_to_validation(self.r.selection, "pinball_validation")
        self.assertTrue((self_gap.gap_to_validation == 0).all())

    def test_worst_decile_is_per_series_and_at_least_the_mean(self):
        w = self.r.worst_decile
        self.assertEqual(len(w), 2 * 11)
        self.assertTrue((w.n_series == 22).all())
        self.assertTrue((w.n_in_worst == 3).all())                                # ceil(2.2)
        self.assertTrue((w.worst_decile_regret >= w.mean_series_regret - 1e-15).all())
        self.assertTrue((w.worst_decile_regret <= w.max_series_regret + 1e-15).all())
        # grouping by horizon only averages a series over its 11 taus first
        wh = e2.worst_decile_regret(self.r.selection, "d_omega", by=("horizon",))
        self.assertEqual(len(wh), 2)
        s = self.r.selection
        s = s[(s.selector == "d_omega") & (s.horizon == 24)]
        per_series = s.groupby("series_id").scaled_regret.mean().sort_values(ascending=False)
        self.assertAlmostEqual(wh[wh.horizon == 24].worst_decile_regret.iloc[0],
                               per_series.iloc[:3].mean(), places=14)

    def test_selector_timing_is_labelled_as_not_reproducible(self):
        c = self.r.selector_cost
        self.assertEqual(list(c.selector), ["always_classical", "always_tsfm", "static_omega",
                                            "pinball_validation", "crps_validation", "d_omega",
                                            "hindsight_oracle"])
        self.assertTrue(np.isfinite(c.seconds).all())
        self.assertTrue(c.measured.str.contains("single run").all())
        self.assertTrue(c.measured.str.contains("not reproducible").all())
        self.assertTrue((c.n_units == len(self.tables["units"])).all())

    def test_every_result_carries_the_frozen_hash(self):
        for name in ("unit_regret", "regret_summary", "gap_to_validation", "worst_decile", "selector_cost"):
            with self.subTest(table=name):
                self.assertEqual(set(getattr(self.r, name).frozen_sha256), {self.sha})
        self.assertEqual(self.r.selector_code_sha256, self.frozen.payload["selector_code_sha256"])
        out = self.dir / "result"
        files = self.r.write(out)
        self.assertEqual(sorted(files), sorted(p.name for p in out.iterdir()))
        self.assertIn(f"selection_{self.sha[:12]}.csv", files)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["frozen_sha256"], self.sha)
        self.assertEqual(manifest["n_units"], 484)
        self.assertIn("not a security control", manifest["hash_note"])
        self.assertEqual(set(pd.read_csv(out / "regret_summary.csv").frozen_sha256), {self.sha})

    def test_cli_runs_twice_into_the_same_folder(self):
        out = self.dir / "cli"
        for run in (1, 2):
            fixture, build = _shared.cli_patches()
            with fixture, build, contextlib.redirect_stdout(io.StringIO()) as printed:
                e2.main(["--out", str(out), "--n-boot", "0"])
            text = printed.getvalue()
            self.assertIn("SYNTHETIC FIXTURE", text)
            self.assertIn("choose differently: 0 of 484", text)
            frozen = sorted(out.glob("frozen_*.json"))
            self.assertEqual(len(frozen), 1, run)               # named by content, reused, not rewritten
            self.assertEqual(frozen[0].name, f"frozen_{self.sha[:12]}.json")
        self.assertEqual(json.loads((out / "manifest.json").read_text(encoding="utf-8"))["frozen_sha256"], self.sha)


class DropInSelector(_WithFrozen):
    """Any object with the same fit / build / select shape works unchanged."""

    def test_custom_kind_goes_through_fit_freeze_evaluate(self):
        calls = {"fit": 0}

        class ThresholdOnTau:
            name, needs_hindsight = "tau_gate", False

            def __init__(self, cut):
                self.cut = cut

            def select(self, inp):
                fam = np.where(np.abs(inp.units.tau.to_numpy() - 0.5) < self.cut, "tsfm", "classical")
                return _decisions(inp.units, fam.astype(object))

        class Adapter:
            kind = "test_tau_gate"

            def fit(self, fit_units):
                calls["fit"] += 1
                return {"cut": 0.2}

            def build(self, params, fit_series, fit_datasets):
                return ThresholdOnTau(params["cut"])

        self._register(Adapter())
        r = e2.evaluate(self._freeze_kind("test_tau_gate", "tau_gate"), self.tables)
        self.assertEqual(calls["fit"], 1)                      # evaluate never fits
        self.assertEqual(r.selector_name, "tau_gate")
        got = r.selection[r.selection.selector == "tau_gate"]
        self.assertEqual(len(got), len(self.tables["units"]))
        self.assertTrue((got[np.abs(got.tau - 0.5) < 0.2].chosen_family == "tsfm").all())
        self.assertTrue((got[np.abs(got.tau - 0.5) > 0.2].chosen_family == "classical").all())
        self.assertNotIn("d_omega", set(r.selection.selector))

    def test_frozen_selector_never_receives_test_costs(self):
        seen = {}

        class Peek:
            name, needs_hindsight = "peek", False

            def select(self, inp):
                seen["test"] = inp.test
                seen["val_columns"] = sorted(inp.val.columns)
                seen["unit_columns"] = sorted(inp.units.columns)
                return _decisions(inp.units, np.full(len(inp.units), "classical", dtype=object))

        class Adapter:
            kind = "test_peek"

            def fit(self, fit_units):
                return {}

            def build(self, params, fit_series, fit_datasets):
                return Peek()

        self._register(Adapter())
        e2.evaluate(self._freeze_kind("test_peek", "peek"), self.tables)
        self.assertIsNone(seen["test"])
        self.assertEqual(seen["unit_columns"], sorted(["dataset", "config", "series_id", "horizon", "tau",
                                                       "omega"]))
        self.assertEqual(seen["val_columns"], sorted(self.tables["val"].columns))
        self.assertNotIn("n_origins", seen["unit_columns"])

    def test_selector_cannot_change_the_callers_tables(self):
        class Mutating:
            name, needs_hindsight = "mut", False

            def select(self, inp):
                inp.units["omega"] = 0.0
                inp.val["J_classical"] = 0.0
                return _decisions(inp.units, np.full(len(inp.units), "classical", dtype=object))

        class Adapter:
            kind = "test_mut"

            def fit(self, fit_units):
                fit_units["omega"] = 0.0
                return {}

            def build(self, params, fit_series, fit_datasets):
                return Mutating()

        self._register(Adapter())
        t = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in self.tables.items()}
        r = e2.evaluate(self._freeze_kind("test_mut", "mut", tables=t), t)
        for k in ("units", "val", "fit_units"):
            pd.testing.assert_frame_equal(t[k], self.tables[k])
        orc = r.selection[r.selection.selector == "hindsight_oracle"]
        self.assertGreater(orc.omega.min(), 0.05)              # later selectors saw the real omega

    def test_selector_name_must_match_the_frozen_name_at_evaluation(self):
        state = {"name": "gate"}

        class Named:
            needs_hindsight = False
            select = _select_all("classical")

            def __init__(self, name):
                self.name = name

        class Adapter:
            kind = "test_named"

            def fit(self, fit_units):
                return {}

            def build(self, params, fit_series, fit_datasets):
                return Named(state["name"])

        self._register(Adapter())
        with self.assertRaisesRegex(ValueError, "built selector is named 'gate'"):
            _fit(selector_kind="test_named", selector_name="other")
        frozen = self._freeze_kind("test_named", "gate")
        e2.evaluate(frozen, self.tables)
        state["name"] = "static_omega"                         # would collide with a baseline's rows
        with self.assertRaisesRegex(ValueError, "built selector is named 'static_omega', frozen name is 'gate'"):
            e2.evaluate(frozen, self.tables)

    def test_selector_that_wants_hindsight_or_a_baseline_name_is_refused(self):
        state = {"hindsight": True}

        class Cheat:
            name = "cheat"
            select = _select_all("classical")

            @property
            def needs_hindsight(self):
                return state["hindsight"]

        class Adapter:
            kind = "test_cheat"

            def fit(self, fit_units):
                return {}

            def build(self, params, fit_series, fit_datasets):
                return Cheat()

        self._register(Adapter())
        with self.assertRaisesRegex(ValueError, "hindsight"):
            _fit(selector_kind="test_cheat", selector_name="cheat")
        state["hindsight"] = False
        frozen = self._freeze_kind("test_cheat", "cheat")
        state["hindsight"] = True
        with self.assertRaisesRegex(ValueError, "hindsight"):
            e2.evaluate(frozen, self.tables)
        with self.assertRaisesRegex(ValueError, "baseline"):
            _fit(selector_name="hindsight_oracle")
        with self.assertRaises(TypeError):
            e2.register_selector_kind(object())

    def test_selector_code_changed_after_freezing_is_refused(self):
        """The reviewer's route: register the kind again with other code. The
        frozen file and its hash are unchanged; the behaviour is not."""
        base = e2.evaluate(self.frozen, self.tables)

        class Swap:
            kind = KIND

            def fit(self, fit_units):
                return {}

            def build(self, params, fit_series, fit_datasets):
                class AlwaysTSFMNamedDOmega:
                    name, needs_hindsight = "d_omega", False
                    select = _select_all("tsfm")
                return AlwaysTSFMNamedDOmega()

        self._register(Swap())
        with self.assertRaises(e2.SelectorCodeError) as cm:
            e2.evaluate(self.frozen, self.tables)
        self.assertIn("changed after freezing", str(cm.exception))
        self.assertIsInstance(cm.exception, e2.FrozenFileError)
        e2.register_selector_kind(e2.DOmegaLinearPlaceholder())             # the original code again
        again = e2.evaluate(self.frozen, self.tables)
        pd.testing.assert_frame_equal(again.selection, base.selection)
        self.assertEqual(again.selector_code_sha256, base.selector_code_sha256)

    def test_code_hash_covers_the_adapter_class_and_the_selector_module(self):
        adapter = e2.SELECTOR_KINDS[KIND]
        selector = adapter.build({"omega0": 0.4, "beta": 0.0}, ["x"], ["d"])
        h = e2.selector_code_sha256(adapter, selector)
        self.assertEqual(h, self.frozen.payload["selector_code_sha256"])
        import decision_aware_tsfm.analysis.decision.selectors as mod
        parts = [f"adapter class\x1f{e2.__name__}.DOmegaLinearPlaceholder\x1f"
                 + inspect.getsource(e2.DOmegaLinearPlaceholder),
                 f"selector module\x1f{mod.__name__}\x1f" + inspect.getsource(mod)]
        self.assertEqual(h, hashlib.sha256("\x1e".join(parts).encode()).hexdigest())

    def test_selector_without_readable_source_cannot_be_frozen(self):
        scope = {"_select_all": _select_all}
        exec("class Typed:\n    name, needs_hindsight = 'typed', False\n"
             "    select = _select_all('classical')\n"
             "class Adapter:\n    kind = 'test_typed'\n"
             "    def fit(self, fit_units):\n        return {}\n"
             "    def build(self, params, fit_series, fit_datasets):\n        return Typed()\n", scope)
        self._register(scope["Adapter"]())
        with self.assertRaises(e2.SelectorCodeError):
            _fit(selector_kind="test_typed", selector_name="typed")

    def test_guards_belong_to_the_interface_not_to_the_selector(self):
        """The placeholder selector has its own held-out and overlap checks. A
        selector without any must still be protected by fit and evaluate."""
        calls = {"fit": 0, "select": 0}

        class Unguarded:
            name, needs_hindsight = "unguarded", False

            def select(self, inp):
                calls["select"] += 1
                return _decisions(inp.units, np.full(len(inp.units), "classical", dtype=object))

        class Adapter:
            kind = "test_unguarded"

            def fit(self, fit_units):
                calls["fit"] += 1
                return {}

            def build(self, params, fit_series, fit_datasets):
                return Unguarded()

        self._register(Adapter())
        fu = self.tables["fit_units"]
        for col in ("dataset", "config", "series_id"):
            with self.subTest(held_out_in=col):
                with self.assertRaises(HeldOutLeakError):
                    _fit(fit_units=fu.assign(**{col: "m5_invented/" + fu[col]}),
                         selector_kind="test_unguarded", selector_name="unguarded")
        self.assertEqual(calls["fit"], 0)          # refused before the selector saw the data

        frozen = self._freeze_kind("test_unguarded", "unguarded")
        self.assertEqual(calls["fit"], 1)
        with self.assertRaises(FitEvalOverlapError):
            e2.evaluate(frozen, _shared.tables_without_fit_split())
        with self.assertRaises(FitEvalOverlapError):
            e2.evaluate(frozen, _shared.renamed(_shared.tables_without_fit_split(), "fx_dfit",
                                                "fx_other", rename_series=False))
        with self.assertRaises(HeldOutLeakError):
            e2.evaluate(frozen, _shared.renamed(self.tables, "fx_tau_flip", INVENTED_HELD_OUT))
        self.assertEqual(calls["select"], 0)       # refused before the selector ran
        e2.evaluate(frozen, self.tables)
        self.assertEqual(calls["select"], 1)

    def test_non_json_or_nan_parameters_cannot_be_frozen(self):
        class Adapter:
            kind = "test_nan"

            def fit(self, fit_units):
                return {"w": float("nan")}

            def build(self, params, fit_series, fit_datasets):
                raise AssertionError

        self._register(Adapter())
        with self.assertRaises(ValueError):
            _fit(selector_kind="test_nan")


if __name__ == "__main__":
    unittest.main()
