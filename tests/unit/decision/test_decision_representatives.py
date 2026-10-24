"""Within-family representatives (#41): chosen on validation origins only, then frozen.

Most tests use a small hand-built cache with flat forecasts, so every expected
number can be worked out by hand: with y = 10 and a forecast q at every
quantile, the pinball loss at tau 0.50 is 0.5 * |10 - q| and the scale is 10.
Those caches have no history, so they get a STATED scale table (hand_scales: the
series' constant |y|); the roster fixture gets its computed one (#29 item 7a).
A unit with n origins spaced one horizon apart has floor(n / 3) validation
origins and no gap, so with 12 origins the validation origins are indices 0..3.
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from decision_aware_tsfm.analysis.decision import representatives as rep
from decision_aware_tsfm.analysis.decision.cache_io import CacheError, qcol, restrict_to_representatives
from decision_aware_tsfm.analysis.decision.costs import assign_roles
from decision_aware_tsfm.analysis.decision.evaluate import summarize
from decision_aware_tsfm.analysis.decision.fixture import (
    FIT_ONLY_DATASETS, FIXTURE_SCHEMA, FIXTURE_STATIC_THRESHOLD, PRODUCER_SCHEMA, REPRESENTATIVES,
    ROSTER_EXTRA_MODEL, ROSTER_FAMILY_OF_MODEL, make_roster_fixture, to_producer_format)
from decision_aware_tsfm.analysis.decision.guards import HeldOutLeakError
from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID
from decision_aware_tsfm.analysis.decision.pipeline import build_tables, run
from decision_aware_tsfm.analysis.decision.run_representatives import main as run_representatives
from decision_aware_tsfm.analysis.decision.scale import read_workload_sources
from decision_aware_tsfm.analysis.experiments.e2 import FrozenFileError

from . import _shared, _sources
from .test_decision_producer import write_bundle

QCOLS = [qcol(t) for t in TAU_GRID]
TAUS = np.asarray(TAU_GRID)
FAMILY = {"A": "classical", "B": "classical", "C": "classical", "T": "tsfm"}
UNIT = dict(dataset="d1", series_id="d1/s0", horizon=1, n=12, y=10.0)
N_VAL = 4                                   # floor(12 / 3)


def flat(validation, evaluation, n_val=N_VAL):
    """A model that forecasts `validation` on validation origins and `evaluation` after."""
    return lambda unit, k: validation if k < n_val else evaluation


def make_cache(models, units=(UNIT,), seeds=(0,)):
    """Canonical cache. models: {name: f(unit, origin_index) -> forecast}, where a
    forecast is one number (every quantile) or 11 numbers, and None means the
    model has no forecast there. Origins are spaced one horizon apart."""
    rows = []
    for u in units:
        for model, f in models.items():
            for k in range(u["n"]):
                q = f(u, k)
                if q is None:
                    continue
                q = np.broadcast_to(np.asarray(q, dtype=float), (len(TAU_GRID),))
                for seed in seeds:
                    for step in range(1, u["horizon"] + 1):
                        r = dict(dataset=u["dataset"], config="H", series_id=u["series_id"], model=model,
                                 seed=seed, horizon=u["horizon"], origin=k * u["horizon"], step=step,
                                 y=u["y"])
                        r.update(zip(QCOLS, q))
                        rows.append(r)
    fc = pd.DataFrame(rows)
    for c in ("seed", "horizon", "origin", "step"):
        fc[c] = fc[c].astype("int64")
    return fc


def hand_scales(fc):
    """A stated scale per series: the fixture's computed table for the roster fixture,
    and the series' constant |y| for a hand-built cache (10, or 0 for a zero series)."""
    if fc["dataset"].str.startswith("fx_").all():
        return _shared.scales()
    g = fc.groupby(["dataset", "config", "series_id"], sort=True)
    return pd.DataFrame(dict(scale=g["y"].apply(lambda y: float(np.abs(y).mean())),
                             history_end=g["origin"].min())).reset_index()


def best(fc, family=FAMILY, **kw):
    kw.setdefault("scales", hand_scales(fc))
    return rep.best_of_family(fc, FIXTURE_SCHEMA, family, **kw)


def scores(payload):
    return {r["model"]: r for r in payload["scores"]}


def median_pinball_by_model(fc, rows):
    """Mean pinball at tau 0.50 per model over the given rows, straight from the columns."""
    d = fc[rows]
    return (0.5 * (d["y"] - d[qcol(0.50)]).abs()).groupby(d["model"]).mean()


# A is best on validation origins, B is best on evaluation origins and overall
SPLIT = {"A": flat(9.0, 0.0), "B": flat(8.0, 9.0), "C": flat(7.0, 7.0), "T": flat(9.5, 9.5)}


class ValidationOnly(unittest.TestCase):
    def setUp(self):
        self.fc = make_cache(SPLIT)
        self.payload = best(self.fc)

    def test_the_validation_winner_is_chosen_not_the_evaluation_winner(self):
        k = self.fc.origin // self.fc.horizon
        classical = self.fc.model != "T"
        on_validation = median_pinball_by_model(self.fc, classical & (k < N_VAL))
        on_evaluation = median_pinball_by_model(self.fc, classical & (k >= N_VAL))
        overall = median_pinball_by_model(self.fc, classical)
        # the three windows disagree, so the assertion below can tell them apart
        self.assertEqual(on_validation.idxmin(), "A")
        self.assertEqual(on_evaluation.idxmin(), "B")
        self.assertEqual(overall.idxmin(), "B")
        self.assertEqual(self.payload["representatives"], {"classical": "A", "tsfm": "T"})
        self.assertEqual(self.payload["reasons"]["classical"], "lowest_validation_loss")

    def test_scores_are_the_hand_computed_validation_losses(self):
        s = scores(self.payload)
        for model, raw in (("A", 0.5), ("B", 1.0), ("C", 1.5), ("T", 0.25)):
            self.assertAlmostEqual(s[model]["mean_raw_loss"], raw, places=12)
            self.assertAlmostEqual(s[model]["mean_scaled_loss"], raw / 10.0, places=12)
        self.assertAlmostEqual(s["B"]["loss_minus_best"], 0.05, places=12)
        self.assertEqual([m for m, r in s.items() if r["chosen"]], ["A", "T"])

    def test_changing_every_evaluation_forecast_changes_no_score(self):
        other = self.fc.copy()
        late = (other.origin // other.horizon) >= N_VAL
        other.loc[late, QCOLS] = 123.0
        p = best(other)
        self.assertEqual(p["scores"], self.payload["scores"])
        self.assertEqual(p["representatives"], self.payload["representatives"])
        self.assertNotEqual(p["input"]["sha256"], self.payload["input"]["sha256"])

    def test_a_validation_forecast_does_change_the_score(self):
        other = self.fc.copy()
        early = (other.model == "A") & ((other.origin // other.horizon) < N_VAL)
        other.loc[early, QCOLS] = 6.0
        self.assertEqual(best(other)["representatives"]["classical"], "B")

    def test_counts_and_statement_are_recorded(self):
        p = self.payload
        self.assertEqual(p["roles_scored"], ["validation"])
        self.assertEqual(p["n_origins_by_role"], {"validation": 4, "test": 8, "gap": 0})
        self.assertIn("validation origins only", p["origins_statement"])
        self.assertAlmostEqual(p["val_fraction"], 1 / 3)
        self.assertEqual(p["coverage"]["classical"], dict(
            n_candidates=3, n_units_scored=1, n_series_scored=1, n_datasets_scored=1,
            n_validation_origins_scored=4, n_units_dropped_zero_scale=0))
        self.assertEqual(p["candidates"], {"classical": ["A", "B", "C"], "tsfm": ["T"]})
        self.assertEqual((p["selection"]["tau"], p["selection"]["loss"]), (0.5, "scaled"))

    def test_gap_origins_are_not_scored(self):
        """Horizon 3 at every origin: 12 origins, 4 validation candidates, E = 4, and
        origins 2 and 3 reach the evaluation period, so only origins 0 and 1 are scored."""
        def model(lo):
            return lambda unit, k: 9.0 if k < 2 else lo
        fc = make_cache({"A": model(0.0), "B": lambda unit, k: 8.0, "T": model(9.0)},
                        units=[dict(UNIT, horizon=3)])
        fc["origin"] = fc["origin"] // 3              # one origin per step: windows overlap
        p = best(fc)
        self.assertEqual(p["n_origins_by_role"], {"validation": 2, "test": 8, "gap": 2})
        self.assertEqual(p["representatives"]["classical"], "A")
        self.assertAlmostEqual(scores(p)["A"]["mean_raw_loss"], 0.5, places=12)

    def test_window_is_the_engine_validation_window(self):
        """On the roster fixture each candidate's score equals the mean of the engine's
        own validation costs (build_tables 'val', tau 0.50) for that representative."""
        fc, omega = make_roster_fixture(7)
        p = best(fc, ROSTER_FAMILY_OF_MODEL)
        for model in p["candidates"]["classical"]:
            t = build_tables(fc, FIXTURE_SCHEMA, {"classical": model, "tsfm": "Moirai-small"},
                             ROSTER_FAMILY_OF_MODEL, omega, scales=_shared.scales())
            v = t["val"][np.isclose(t["val"].tau, 0.5)]
            self.assertEqual(len(v), scores(p)[model]["n_units_scored"])
            self.assertAlmostEqual(float((v.J_classical / v.scale).mean()),
                                   scores(p)[model]["mean_scaled_loss"], places=12)
            self.assertAlmostEqual(float(v.J_classical.mean()), scores(p)[model]["mean_raw_loss"], places=12)
        self.assertAlmostEqual(float((v.J_tsfm / v.scale).mean()),
                               scores(p)["Moirai-small"]["mean_scaled_loss"], places=12)


class Aggregation(unittest.TestCase):
    def test_scaled_loss_is_the_default_and_raw_is_the_alternative(self):
        """Small series y = 10, large series y = 1000. A is far better on the small one
        in relative terms; B is slightly better on the large one in absolute terms."""
        units = [dict(UNIT, dataset="small", series_id="small/s0", y=10.0),
                 dict(UNIT, dataset="large", series_id="large/s0", y=1000.0)]
        err = {"A": {"small": 1.0, "large": 100.0}, "B": {"small": 4.0, "large": 90.0},
               "T": {"small": 1.0, "large": 1.0}}
        fc = make_cache({m: (lambda u, k, e=e: u["y"] - e[u["dataset"]]) for m, e in err.items()}, units)
        scaled, raw = best(fc), best(fc, loss="raw")
        self.assertEqual(scaled["representatives"]["classical"], "A")
        self.assertEqual(raw["representatives"]["classical"], "B")
        for p in (scaled, raw):
            self.assertEqual(p["best_by_loss"]["classical"], {"scaled": "A", "raw": "B"})
            s = scores(p)
            self.assertAlmostEqual(s["A"]["mean_scaled_loss"], 0.05, places=12)
            self.assertAlmostEqual(s["B"]["mean_scaled_loss"], (0.2 + 0.045) / 2, places=12)
            self.assertAlmostEqual(s["A"]["mean_raw_loss"], (0.5 + 50.0) / 2, places=12)
            self.assertAlmostEqual(s["B"]["mean_raw_loss"], (2.0 + 45.0) / 2, places=12)
        self.assertEqual(raw["selection"]["loss"], "raw")
        self.assertAlmostEqual(scores(raw)["A"]["loss_minus_best"], 25.25 - 23.5, places=12)

    def test_score_is_the_mean_over_units_not_over_rows(self):
        """One unit with 4 validation rows where A is much better, one with 20 where B
        is a little better. Per unit A wins; pooled over rows B would."""
        units = [dict(UNIT, series_id="d1/short", n=12), dict(UNIT, series_id="d1/long", n=60)]
        err = {"A": {"d1/short": 1.0, "d1/long": 2.0}, "B": {"d1/short": 5.0, "d1/long": 1.0},
               "T": {"d1/short": 1.0, "d1/long": 1.0}}
        fc = make_cache({m: (lambda u, k, e=e: u["y"] - e[u["series_id"]]) for m, e in err.items()}, units)
        roles = assign_roles(fc, 1 / 3)
        pooled = median_pinball_by_model(roles, (roles.role == "validation") & (roles.model != "T"))
        self.assertEqual(pooled.idxmin(), "B")                       # the other reading disagrees
        p = best(fc)
        self.assertEqual(p["representatives"]["classical"], "A")
        self.assertAlmostEqual(scores(p)["A"]["mean_scaled_loss"], (0.05 + 0.10) / 2, places=12)
        self.assertAlmostEqual(scores(p)["B"]["mean_scaled_loss"], (0.25 + 0.05) / 2, places=12)
        self.assertEqual(p["coverage"]["classical"]["n_validation_origins_scored"], 4 + 20)

    def test_only_tau_050_is_read(self):
        """A has the better median and useless tails; B the reverse. Any other tau, or
        an average over taus, would pick B."""
        a = np.where(TAUS < 0.5, -40.0, np.where(TAUS > 0.5, 60.0, 9.0))
        b = np.where(TAUS > 0.5, 13.0, 7.0)
        fc = make_cache({"A": lambda u, k: a, "B": lambda u, k: b, "T": lambda u, k: 9.5})
        p = best(fc)
        self.assertEqual(p["representatives"]["classical"], "A")
        self.assertAlmostEqual(scores(p)["A"]["mean_raw_loss"], 0.5, places=12)
        self.assertAlmostEqual(scores(p)["B"]["mean_raw_loss"], 1.5, places=12)

    def test_seeds_are_averaged_per_window(self):
        fc = make_cache(SPLIT, seeds=(42, 43))
        self.assertEqual(best(fc)["scores"], best(make_cache(SPLIT))["scores"])
        self.assertEqual(best(fc)["input"]["seeds"], [42, 43])

    def test_zero_scale_unit_is_counted_and_not_scored(self):
        units = [UNIT, dict(UNIT, series_id="d1/zero", y=0.0)]
        fc = make_cache({"A": lambda u, k: u["y"] - 1.0, "B": lambda u, k: u["y"] - 2.0,
                         "T": lambda u, k: u["y"]}, units)
        p = best(fc)
        self.assertEqual(p["coverage"]["classical"]["n_units_dropped_zero_scale"], 1)
        self.assertEqual(p["coverage"]["classical"]["n_units_scored"], 1)
        self.assertAlmostEqual(scores(p)["A"]["mean_scaled_loss"], 0.05, places=12)


class TiesAndSingleMembers(unittest.TestCase):
    def test_exact_tie_goes_to_the_first_name_whatever_the_input_order(self):
        same = lambda u, k: 9.0
        for order in (("B", "A"), ("A", "B")):
            fc = make_cache({order[0]: same, order[1]: same, "C": lambda u, k: 5.0, "T": same})
            p = best(fc)
            self.assertEqual(p["representatives"]["classical"], "A", order)
            self.assertEqual(p["reasons"]["classical"], "tie_first_by_name")
            self.assertEqual(scores(p)["B"]["loss_minus_best"], 0.0)

    def test_a_difference_inside_the_tolerance_is_a_tie(self):
        fc = make_cache({"A": lambda u, k: 9.0, "B": lambda u, k: 9.0 + 1e-13, "T": lambda u, k: 9.0})
        p = best(fc)
        self.assertLess(scores(p)["B"]["mean_scaled_loss"], scores(p)["A"]["mean_scaled_loss"])
        self.assertEqual((p["representatives"]["classical"], p["reasons"]["classical"]),
                         ("A", "tie_first_by_name"))

    def test_a_real_difference_is_not_a_tie(self):
        fc = make_cache({"A": lambda u, k: 9.0, "B": lambda u, k: 9.0 + 1e-6, "T": lambda u, k: 9.0})
        p = best(fc)
        self.assertEqual((p["representatives"]["classical"], p["reasons"]["classical"]),
                         ("B", "lowest_validation_loss"))

    def test_single_member_family_returns_it_as_only_member(self):
        p = best(make_cache(SPLIT))
        self.assertEqual((p["representatives"]["tsfm"], p["reasons"]["tsfm"]), ("T", "only_member"))
        one = best(make_cache({"C": SPLIT["C"], "T": SPLIT["T"]}))
        self.assertEqual(one["representatives"], {"classical": "C", "tsfm": "T"})
        self.assertEqual(one["reasons"], {"classical": "only_member", "tsfm": "only_member"})

    def test_single_member_needs_no_validation_origin(self):
        """Two origins give no validation third; the only member is still returned."""
        p = best(make_cache({"C": SPLIT["C"], "T": SPLIT["T"]}, units=[dict(UNIT, n=2)]))
        self.assertEqual(p["reasons"], {"classical": "only_member", "tsfm": "only_member"})
        self.assertIsNone(scores(p)["C"]["mean_scaled_loss"])
        self.assertEqual(p["coverage"]["classical"]["n_units_scored"], 0)

    def test_several_candidates_and_no_validation_origin_is_refused(self):
        with self.assertRaisesRegex(CacheError, "cannot be chosen"):
            best(make_cache(SPLIT, units=[dict(UNIT, n=2)]))


class Coverage(unittest.TestCase):
    EASY = dict(UNIT, series_id="d1/easy")
    HARD = dict(UNIT, series_id="d1/hard")
    ERR = {"A": {"d1/easy": 0.5, "d1/hard": 5.0}, "B": {"d1/easy": 2.0, "d1/hard": 6.0},
           "C": {"d1/easy": 0.6, "d1/hard": 5.5}, "T": {"d1/easy": 1.0, "d1/hard": 1.0}}

    def _fc(self, c_missing):
        def model(name):
            def f(u, k):
                if name == "C" and c_missing(u, k):
                    return None
                return u["y"] - self.ERR[name][u["series_id"]]
            return f
        return make_cache({m: model(m) for m in self.ERR}, [self.EASY, self.HARD])

    def test_a_model_missing_a_hard_series_does_not_win_by_missing_it(self):
        p = best(self._fc(lambda u, k: u["series_id"] == "d1/hard"))
        s = scores(p)
        # on its own units C would average 0.03, below A's 0.275 over both
        self.assertEqual(p["representatives"]["classical"], "A")
        self.assertEqual({m: s[m]["n_units_covered"] for m in "ABC"}, {"A": 2, "B": 2, "C": 1})
        self.assertEqual({s[m]["n_units_scored"] for m in "ABC"}, {1})
        self.assertAlmostEqual(s["A"]["mean_scaled_loss"], 0.025, places=12)
        self.assertAlmostEqual(s["C"]["mean_scaled_loss"], 0.030, places=12)
        self.assertEqual(p["coverage"]["classical"]["n_units_scored"], 1)

    def test_a_model_missing_some_origins_of_a_unit_does_not_cover_it(self):
        """C lacks the hard series' last evaluation origin. Its origins there are not
        the ones the roles came from, so that unit is compared for nobody."""
        p = best(self._fc(lambda u, k: u["series_id"] == "d1/hard" and k == 11))
        s = scores(p)
        self.assertEqual({m: s[m]["n_units_covered"] for m in "ABC"}, {"A": 2, "B": 2, "C": 1})
        self.assertEqual(p["coverage"]["classical"]["n_units_scored"], 1)
        self.assertEqual(p["representatives"]["classical"], "A")

    def test_full_coverage_scores_every_unit(self):
        p = best(self._fc(lambda u, k: False))
        self.assertEqual(p["coverage"]["classical"]["n_units_scored"], 2)
        self.assertAlmostEqual(scores(p)["A"]["mean_scaled_loss"], (0.025 + 0.25) / 2, places=12)


class Guards(unittest.TestCase):
    def setUp(self):
        self.fc = make_cache(SPLIT)

    def test_held_out_dataset_or_config_is_refused_before_anything_is_scored(self):
        """The refusal must come from the guard at the start of the selection, not from
        the check on the finished payload: by then every loss would have been computed."""
        named = {"classical": "A", "tsfm": "T"}
        reached = AssertionError("held-out rows reached the role assignment")
        for column, name in (("dataset", "M5"), ("dataset", "gift/pjm_hourly"), ("config", "m5_daily"),
                             ("config", "PJM")):
            fc = self.fc.assign(**{column: name})
            with mock.patch.object(rep, "assign_roles", side_effect=reached), \
                    mock.patch.object(rep, "per_origin_costs", side_effect=reached):
                with self.assertRaisesRegex(HeldOutLeakError, "representative selection", msg=name):
                    best(fc)
                with self.assertRaisesRegex(HeldOutLeakError, "representative selection", msg=name):
                    rep.fixed_named(fc, FIXTURE_SCHEMA, FAMILY, named, scales=hand_scales(self.fc))
            with self.assertRaises(HeldOutLeakError, msg=name):
                rep.representative_robustness(fc, FIXTURE_SCHEMA, FAMILY, None, 0.45, tsfm="T",
                                              scales=hand_scales(self.fc))

    def test_input_fingerprint_ignores_row_order_and_sees_every_value(self):
        fc = make_cache(SPLIT, units=[UNIT, dict(UNIT, series_id="d1/s1", horizon=2, n=9)], seeds=(42, 43))
        want = rep.input_fingerprint(fc)
        self.assertEqual(want, best(fc)["input"]["sha256"])
        for seed in (0, 1):
            self.assertEqual(rep.input_fingerprint(fc.sample(frac=1.0, random_state=seed)), want)
        last = fc.index[-1]
        for column, value in (("y", 10.5), (qcol(0.99), 99.0), (qcol(0.01), -1.0), ("model", "Z"),
                              ("seed", 44), ("series_id", "d1/other")):
            other = fc.copy()
            other.loc[last, column] = value
            self.assertNotEqual(rep.input_fingerprint(other), want, column)

    def test_scoring_refuses_evaluation_rows(self):
        rows = assign_roles(self.fc.assign(family=self.fc.model.map(FAMILY)), 1 / 3)
        self.assertEqual(set(rows.role), {"validation", "test"})
        sc = hand_scales(self.fc)
        with self.assertRaisesRegex(rep.EvaluationRowsError, "test"):
            rep.validation_unit_losses(rows, sc)
        with self.assertRaisesRegex(rep.EvaluationRowsError, "test"):
            rep.validation_unit_losses(rows[rows.role == "test"], sc)
        with self.assertRaisesRegex(rep.EvaluationRowsError, "no role column"):
            rep.validation_unit_losses(rows.drop(columns="role"), sc)
        ok = rep.validation_unit_losses(rows[rows.role == "validation"], sc)
        self.assertEqual(len(ok), 4)

    def test_scoring_refuses_gap_rows(self):
        rows = assign_roles(self.fc.assign(family=self.fc.model.map(FAMILY)), 1 / 3)
        with self.assertRaisesRegex(rep.EvaluationRowsError, "gap"):
            rep.validation_unit_losses(rows.assign(role="gap"), hand_scales(self.fc))

    def test_family_labels_must_be_complete_and_consistent(self):
        with self.assertRaisesRegex(CacheError, "no family label"):
            best(self.fc, {m: f for m, f in FAMILY.items() if m != "B"})
        with self.assertRaisesRegex(CacheError, "disagree"):
            best(self.fc.assign(family="tsfm"))
        with self.assertRaisesRegex(CacheError, "no tsfm model"):
            best(self.fc[self.fc.model != "T"])

    def test_bad_settings_are_refused(self):
        for bad in (0.0, 1.0, -0.1):
            with self.assertRaisesRegex(ValueError, "val_fraction"):
                best(self.fc, val_fraction=bad)
        with self.assertRaisesRegex(ValueError, "loss"):
            best(self.fc, loss="mase")

    def test_same_payload_for_shuffled_input(self):
        fc = make_cache(SPLIT, units=[UNIT, dict(UNIT, series_id="d1/s1", horizon=2, n=9)], seeds=(42, 43))
        want = best(fc)
        for seed in (0, 1, 2):
            self.assertEqual(best(fc.sample(frac=1.0, random_state=seed)), want)
        self.assertEqual(best(fc[list(reversed(fc.columns))].iloc[::-1]), want)

    def test_same_payload_for_shuffled_input_when_sums_depend_on_order(self):
        """Losses spread over many orders of magnitude within one window: the float sum
        over a window's steps depends on the order of its rows, so only a fixed
        internal order gives the same scores for every row order of the input."""
        fc = make_cache(SPLIT, units=[dict(UNIT, horizon=60)])
        noise = np.random.default_rng(3).lognormal(0.0, 8.0, len(fc))
        fc[QCOLS] = np.repeat((10.0 - noise)[:, None], len(QCOLS), axis=1)
        want = best(fc)
        for seed in range(6):
            self.assertEqual(best(fc.sample(frac=1.0, random_state=seed)), want, seed)

    def test_validation_fraction_is_used_and_recorded(self):
        half = best(self.fc, val_fraction=0.5)
        self.assertEqual(half["val_fraction"], 0.5)
        self.assertEqual(half["n_origins_by_role"], {"validation": 6, "test": 6, "gap": 0})
        # A forecasts 9 on origins 0..3 and 0 after: (4 * 0.5 + 2 * 5.0) / 6
        self.assertAlmostEqual(scores(half)["A"]["mean_raw_loss"], 2.0, places=12)
        self.assertEqual(half["representatives"]["classical"], "B")


class FixedNamed(unittest.TestCase):
    def setUp(self):
        self.fc = make_cache(SPLIT)

    def _named(self, reps, fc=None, family=FAMILY):
        fc = self.fc if fc is None else fc
        return rep.fixed_named(fc, FIXTURE_SCHEMA, family, reps, scales=hand_scales(fc))

    def test_named_representatives_are_taken_and_scores_still_recorded(self):
        p = self._named({"classical": "C", "tsfm": "T"})
        self.assertEqual(p["rule"], "fixed_named")
        self.assertEqual(p["representatives"], {"classical": "C", "tsfm": "T"})
        self.assertEqual(p["reasons"], {"classical": "named_by_caller", "tsfm": "named_by_caller"})
        self.assertEqual(p["best_by_loss"]["classical"], {"scaled": "A", "raw": "A"})
        self.assertEqual([m for m, r in scores(p).items() if r["chosen"]], ["C", "T"])
        self.assertAlmostEqual(scores(p)["C"]["loss_minus_best"], 0.1, places=12)
        self.assertEqual(p["scores"], [dict(r, chosen=r["model"] in ("C", "T"),
                                            loss_minus_best=r["loss_minus_best"])
                                       for r in best(self.fc)["scores"]])

    def test_refused_exactly_as_the_engine_refuses(self):
        cases = [
            ({"classical": "A"}, self.fc, FAMILY),                                  # a family missing
            ({"classical": "T", "tsfm": "T"}, self.fc, FAMILY),                     # wrong label
            ({"classical": "A", "tsfm": "nobody"}, self.fc, dict(FAMILY, nobody="tsfm")),   # no rows
            ({"classical": "A", "tsfm": "T"},                                       # other targets
             self.fc[~((self.fc.model == "T") & (self.fc.origin == 11))], FAMILY)]
        for reps, fc, family in cases:
            with self.assertRaises(CacheError) as engine:
                restrict_to_representatives(fc, reps, family)
            with self.assertRaises(CacheError) as here:
                self._named(reps, fc, family)
            self.assertEqual(str(here.exception), str(engine.exception))


class FrozenFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "sub" / "representatives.json"
        self.fc = make_cache(SPLIT)
        self.payload = best(self.fc, source={"kind": "test"})
        self.sha = rep.freeze(self.payload, self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def _edit(self, change, rehash=True):
        doc = json.loads(self.path.read_text(encoding="utf-8"))
        change(doc["payload"])
        if rehash:
            doc["sha256"] = rep.content_sha256(doc["payload"])
        self.path.write_text(json.dumps(doc), encoding="utf-8")

    def test_round_trip(self):
        self.assertEqual(rep.load_frozen(self.path), self.payload)
        self.assertEqual(rep.load_representatives(self.path), {"classical": "A", "tsfm": "T"})
        self.assertEqual(self.sha, rep.content_sha256(self.payload))
        raw = self.path.read_bytes()
        self.assertTrue(raw.endswith(b"}\n"))
        self.assertNotIn(b"\r", raw)
        doc = json.loads(raw.decode("utf-8"))
        self.assertEqual(set(doc), {"payload", "sha256"})
        self.assertEqual(doc["sha256"], self.sha)
        self.assertEqual(set(doc["payload"]), set(rep.PAYLOAD_KEYS))
        self.assertEqual(doc["payload"]["input"]["source"], {"kind": "test"})
        self.assertEqual(doc["payload"]["code_sha256"], rep.code_sha256())
        self.assertEqual(doc["payload"]["input"]["sha256"], rep.input_fingerprint(self.fc))

    def test_frozen_dict_runs_the_engine(self):
        fc, omega = make_roster_fixture(7)
        path = Path(self.tmp.name) / "roster.json"
        rep.freeze(best(fc, ROSTER_FAMILY_OF_MODEL), path)
        reps = rep.load_representatives(path)
        t = build_tables(fc, FIXTURE_SCHEMA, reps, ROSTER_FAMILY_OF_MODEL, omega,
                         fit_only_datasets=FIT_ONLY_DATASETS, scales=_shared.scales())
        res, _ = run(t, reps, FIXTURE_STATIC_THRESHOLD)
        self.assertEqual(set(res.chosen_model.dropna()), set(reps.values()))
        self.assertEqual(t["settings"]["representatives"], reps)

    def test_a_frozen_file_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            rep.freeze(self.payload, self.path)

    def test_edited_file_is_refused(self):
        edits = {
            "representative": lambda p: p["representatives"].update(classical="B"),
            "score": lambda p: p["scores"][0].update(mean_scaled_loss=0.0),
            "rule": lambda p: p.update(rule="fixed_named"),
            "val_fraction": lambda p: p.update(val_fraction=0.5),
            "input": lambda p: p["input"].update(sha256="0" * 64)}
        original = self.path.read_text(encoding="utf-8")
        for name, change in edits.items():
            self.path.write_text(original, encoding="utf-8")
            self._edit(change, rehash=False)
            with self.assertRaisesRegex(FrozenFileError, "modified after freezing", msg=name):
                rep.load_frozen(self.path)
            with self.assertRaises(FrozenFileError, msg=name):
                rep.load_representatives(self.path)

    def test_incomplete_or_inconsistent_file_is_refused_even_with_a_matching_hash(self):
        edits = {
            "field missing": lambda p: p.pop("scores"),
            "statement missing": lambda p: p.pop("origins_statement"),
            "extra field": lambda p: p.update(note="x"),
            "other schema": lambda p: p.update(schema="e2-frozen-v2"),
            "unknown rule": lambda p: p.update(rule="best_on_test"),
            "test origins scored": lambda p: p.update(roles_scored=["validation", "test"]),
            "no validation": lambda p: p.update(val_fraction=0.0),
            "family missing": lambda p: p["representatives"].pop("tsfm"),
            "empty name": lambda p: p["representatives"].update(tsfm=""),
            "not a candidate": lambda p: p["representatives"].update(classical="Z"),
            "another candidate": lambda p: p["representatives"].update(classical="B"),
            "score row dropped": lambda p: p["scores"].pop(1),
            "bad fingerprint": lambda p: p["input"].update(sha256="abc"),
            "bad code hash": lambda p: p.update(code_sha256=None),
            "other tau": lambda p: p["selection"].update(tau=0.9),
            "unknown reason": lambda p: p["reasons"].update(classical="looked good")}
        original = self.path.read_text(encoding="utf-8")
        for name, change in edits.items():
            self.path.write_text(original, encoding="utf-8")
            self._edit(change)
            with self.assertRaises(FrozenFileError, msg=name):
                rep.load_frozen(self.path)

    def test_file_naming_held_out_input_is_refused(self):
        self._edit(lambda p: p["input"]["datasets"].append("PJM"))
        with self.assertRaises(HeldOutLeakError):
            rep.load_frozen(self.path)

    def test_truncated_missing_or_foreign_file_is_refused(self):
        text = self.path.read_text(encoding="utf-8")
        self.path.write_text(text[: len(text) // 2], encoding="utf-8")
        with self.assertRaises(FrozenFileError):
            rep.load_frozen(self.path)
        self.path.write_text(json.dumps({"representatives": {"classical": "A", "tsfm": "T"}}),
                             encoding="utf-8")
        with self.assertRaises(FrozenFileError):
            rep.load_frozen(self.path)
        with self.assertRaises(FrozenFileError):
            rep.load_frozen(self.path.with_name("absent.json"))

    def test_an_invalid_selection_cannot_be_frozen(self):
        bad = dict(self.payload, roles_scored=["test"])
        with self.assertRaises(FrozenFileError):
            rep.freeze(bad, self.path.with_name("bad.json"))
        self.assertFalse(self.path.with_name("bad.json").exists())


def quiet(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        run_representatives(argv)
    return out.getvalue()


class Command(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fixture_run_writes_prints_and_reproduces(self):
        path = self.root / "representatives.json"
        text = quiet(["--fixture", "--out", str(path)])
        self.assertIn("SYNTHETIC FIXTURE", text)
        self.assertIn("(written)", text)
        for model in ROSTER_FAMILY_OF_MODEL:
            self.assertIn(model, text)
        first = path.read_bytes()
        fc, _ = make_roster_fixture(7)
        want = best(fc, ROSTER_FAMILY_OF_MODEL,
                    source=dict(kind="synthetic roster fixture", fixture=True, seed=7))
        self.assertEqual(rep.load_frozen(path), want)
        self.assertEqual(set(want["candidates"]["classical"]), {"AutoETS", "SeasonalNaive", ROSTER_EXTRA_MODEL})
        self.assertEqual(want["candidates"]["tsfm"], ["Moirai-small"])
        # run again: same file, not rewritten; another selection for the same file: refused
        self.assertIn("reproduced, left unchanged", quiet(["--fixture", "--out", str(path)]))
        self.assertEqual(path.read_bytes(), first)
        with self.assertRaises(SystemExit) as cm:
            quiet(["--fixture", "--loss", "raw", "--out", str(path)])
        self.assertIn("different selection", str(cm.exception))
        self.assertEqual(path.read_bytes(), first)

    def test_handoff_bundle_and_parquet_give_the_api_result(self):
        fc = make_cache(SPLIT, seeds=(42, 43)).assign(origin=lambda d: d["origin"] + 48)  # room for history
        frame = to_producer_format(fc, FAMILY)
        workload, hashes = _sources.write_workload(self.root / "bundle", fc)
        write_bundle(self.root / "bundle", frame, source_hashes=hashes)
        text = quiet([str(self.root / "bundle"), "--variant", "current", "--out", str(self.root / "b.json")])
        self.assertIn("SYNTHETIC FIXTURE", text)                     # write_bundle marks fixture=True
        got = rep.load_frozen(self.root / "b.json")
        scales = read_workload_sources(workload).scales
        h = _sources.stepped_history(48)                    # the history before origin 48
        self.assertEqual(float(scales["scale"].iloc[0]), float(np.mean(np.abs(h[24:] - h[:-24]))))
        want = rep.best_of_family(frame, PRODUCER_SCHEMA, FAMILY, source=got["input"]["source"],
                                  scales=scales)
        self.assertEqual(got, want)
        self.assertEqual(got["representatives"], {"classical": "A", "tsfm": "T"})
        self.assertEqual(got["input"]["source"]["kind"], "handoff bundle")
        self.assertEqual(got["input"]["source"]["partitions_hash_verified"], frame.job_id.nunique())
        # the same rows from canonical and from producer layout have one fingerprint
        self.assertEqual(got["input"]["sha256"], best(fc)["input"]["sha256"])

        frame.to_parquet(self.root / "cache.parquet", index=False)
        text = quiet([str(self.root / "cache.parquet"), "--workload", str(workload),
                      "--out", str(self.root / "p.json")])
        self.assertNotIn("SYNTHETIC", text)
        self.assertEqual(rep.load_frozen(self.root / "p.json")["scores"], want["scores"])

    def test_fixed_named_from_the_command_line(self):
        path = self.root / "named.json"
        quiet(["--fixture", "--rule", "fixed_named", "--classical", "AutoETS", "--tsfm", "Moirai-small",
               "--out", str(path)])
        self.assertEqual(rep.load_representatives(path), REPRESENTATIVES)
        self.assertEqual(rep.load_frozen(path)["rule"], "fixed_named")

    def test_bad_arguments_are_refused(self):
        out = str(self.root / "x.json")
        for argv in (["--out", out], ["--fixture", "somewhere", "--out", out],
                     ["--fixture", "--classical", "AutoETS", "--out", out],
                     ["--fixture", "--rule", "fixed_named", "--tsfm", "Moirai-small", "--out", out]):
            with self.assertRaises(SystemExit, msg=argv), contextlib.redirect_stderr(io.StringIO()):
                quiet(argv)
        self.assertFalse(Path(out).exists())


class Robustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fc, cls.omega = make_roster_fixture(7)
        cls.table = rep.representative_robustness(
            cls.fc, FIXTURE_SCHEMA, ROSTER_FAMILY_OF_MODEL, cls.omega, FIXTURE_STATIC_THRESHOLD,
            tsfm="Moirai-small", fit_only_datasets=FIT_ONLY_DATASETS, scales=_shared.scales())

    def test_one_block_of_seven_selectors_per_classical_candidate(self):
        t = self.table
        self.assertEqual(sorted(t.classical_representative.unique()),
                         sorted(m for m, f in ROSTER_FAMILY_OF_MODEL.items() if f == "classical"))
        self.assertEqual(len(t), 3 * 7)
        self.assertTrue((t.groupby("classical_representative").selector.nunique() == 7).all())
        self.assertTrue((t[t.selector == "hindsight_oracle"].mean_scaled_regret == 0).all())

    def test_each_block_is_the_engine_run_for_that_representative(self):
        want = summarize(_shared.results()[0], by=("selector",))
        got = (self.table[self.table.classical_representative == "AutoETS"]
               .drop(columns="classical_representative").reset_index(drop=True))
        pd.testing.assert_frame_equal(got, want)
        # the choice of representative moves the headline on this fixture
        r = self.table.pivot(index="selector", columns="classical_representative", values="mean_scaled_regret")
        self.assertFalse(np.allclose(r["AutoETS"], r["SeasonalNaive"]))

    def test_candidates_can_be_named(self):
        t = rep.representative_robustness(
            self.fc, FIXTURE_SCHEMA, ROSTER_FAMILY_OF_MODEL, self.omega, FIXTURE_STATIC_THRESHOLD,
            tsfm="Moirai-small", classical=["SeasonalNaive"], by=("selector", "horizon"),
            fit_only_datasets=FIT_ONLY_DATASETS, scales=_shared.scales())
        self.assertEqual(set(t.classical_representative), {"SeasonalNaive"})
        self.assertEqual(len(t), 7 * 2)


if __name__ == "__main__":
    unittest.main()
