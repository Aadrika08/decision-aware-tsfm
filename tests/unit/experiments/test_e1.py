"""E1: decision map, Delta orientation, sign reversals, symmetry table, tau-grid completeness."""
import contextlib
import io
import json
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from decision_aware_tsfm.analysis.decision.evaluate import summarize
from decision_aware_tsfm.analysis.decision.fixture import planted_answers
from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID
from decision_aware_tsfm.analysis.experiments import e1

from . import _shared

PAIRS = [(0.01, 0.99), (0.05, 0.95), (0.09, 0.91), (0.25, 0.75), (0.40, 0.60)]
# Planted expected-cost gaps on fx_tau_flip are below 0.004 at tau 0.25 and 0.75
# and above 0.014 everywhere else. Units are compared with the planted winner
# only where the planted gap is at least this large.
CLEAR_MARGIN = 0.01
KEYS = ["dataset", "config", "series_id", "horizon", "tau"]
OVERVIEW = Path(__file__).resolve().parents[3] / "docs" / "project-overview.md"


def _tiny_tables(rows):
    """rows: (series, tau, omega, J_classical, J_tsfm, scale). One dataset, horizon 1."""
    d = pd.DataFrame(rows, columns=["series_id", "tau", "omega", "J_classical", "J_tsfm", "scale"])
    d.insert(0, "config", "c")
    d.insert(0, "dataset", "hand")
    d.insert(3, "horizon", 1)
    d["n_origins"] = 5
    return dict(units=d[KEYS + ["omega"]], test=d[KEYS + ["J_classical", "J_tsfm", "scale", "n_origins"]])


def _drop(tbl, mask_fn, keys=("units", "test", "val")):
    t = dict(tbl)
    for k in keys:
        t[k] = tbl[k][~mask_fn(tbl[k])]
    return t


@lru_cache(maxsize=None)
def _fixture_out():
    return e1.run_e1(_shared.tables(), _shared.results()[0])


class TauGrid(unittest.TestCase):
    def setUp(self):
        self.tables = _shared.tables()
        self.res, _ = _shared.results()
        self.out = _fixture_out()

    def test_all_eleven_taus_present_on_both_sides_of_half(self):
        self.assertEqual(len(TAU_GRID), 11)
        for name in ("map", "summary"):
            for h, g in self.out[name].groupby("horizon"):
                with self.subTest(table=name, horizon=h):
                    taus = sorted(g.tau.unique())
                    self.assertEqual(taus, sorted(TAU_GRID))
                    self.assertEqual(sum(t < 0.5 for t in taus), 5)
                    self.assertEqual(sum(t > 0.5 for t in taus), 5)
                    self.assertIn(0.5, taus)

    def test_summary_has_one_row_per_horizon_and_tau(self):
        s = self.out["summary"]
        self.assertEqual(len(s), 2 * 11)
        self.assertFalse(s.duplicated(["horizon", "tau"]).any())
        self.assertEqual(int(s.n_units.sum()), len(self.tables["units"]))
        self.assertEqual(set(s.n_units), {22})

    def test_missing_tau_in_selection_fails_loudly(self):
        for tau in (0.91, 0.01, 0.5):
            with self.subTest(tau=tau):
                res = self.res[self.res.tau != tau]
                with self.assertRaises(e1.MissingTauError) as cm:
                    e1.run_e1(self.tables, res)
                self.assertIn(str(tau), str(cm.exception))

    def test_missing_tau_for_one_selector_at_one_horizon_fails(self):
        drop = (self.res.selector == "d_omega") & (self.res.horizon == 24) & (self.res.tau == 0.09)
        with self.assertRaises(e1.MissingTauError):
            e1.run_e1(self.tables, self.res[~drop])

    def test_missing_tau_in_unit_costs_fails_loudly(self):
        t = dict(self.tables)
        t["test"] = t["test"][~((t["test"].tau == 0.95) & (t["test"].horizon == 1))]
        with self.assertRaises(e1.MissingTauError):
            e1.run_e1(t, self.res)

    def test_tau_missing_for_one_dataset_only_fails(self):
        """The reviewer's construction: tau 0.95 absent for fx_tie in the tables
        and in the selection. Checked per horizon only, this passed, and the
        symmetry row for 0.05 / 0.95 compared 22 units with 18."""
        t = _drop(self.tables, lambda f: (f.tau == 0.95) & (f.dataset == "fx_tie"))
        res = self.res[~((self.res.tau == 0.95) & (self.res.dataset == "fx_tie"))]
        with self.assertRaises(e1.MissingTauError) as cm:
            e1.run_e1(t, res)
        self.assertIn("fx_tie", str(cm.exception))
        self.assertIn("0.95", str(cm.exception))

    def test_tau_missing_for_one_series_or_in_units_only_fails(self):
        one = _drop(self.tables, lambda f: (f.tau == 0.95) & (f.series_id == "fx_tsfm_wins/s0"))
        with self.assertRaises(e1.MissingTauError) as cm:
            e1.run_e1(one, self.res)
        self.assertIn("fx_tsfm_wins/s0", str(cm.exception))
        with self.assertRaises(e1.MissingTauError):
            e1.run_e1(_drop(self.tables, lambda f: f.tau == 0.95, keys=("units",)), self.res)

    def test_unexpected_tau_fails(self):
        extra = self.res[self.res.tau == 0.5].assign(tau=0.3)
        with self.assertRaises(e1.MissingTauError) as cm:
            e1.run_e1(self.tables, pd.concat([self.res, extra], ignore_index=True))
        self.assertIn("0.3", str(cm.exception))
        t = dict(self.tables)
        for k in ("units", "test"):
            t[k] = pd.concat([t[k], t[k][t[k].tau == 0.5].assign(tau=0.3)], ignore_index=True)
        with self.assertRaises(e1.MissingTauError):
            e1.run_e1(t, self.res)

    def test_selector_on_other_units_than_the_map_fails(self):
        res = self.res[~((self.res.selector == "d_omega") & (self.res.dataset == "fx_tie"))]
        with self.assertRaises(e1.UnequalPopulationError) as cm:
            e1.run_e1(self.tables, res)
        self.assertIn("d_omega", str(cm.exception))

    def test_empty_results_fail(self):
        with self.assertRaises(e1.MissingTauError):
            e1.run_e1(self.tables, self.res.iloc[0:0])

    def test_assert_full_grid_levels(self):
        f = pd.DataFrame(dict(horizon=1, dataset=["a"] * 3 + ["b"] * 2, tau=[0.1, 0.5, 0.9, 0.1, 0.5]))
        grid = (0.1, 0.5, 0.9)
        e1.assert_full_grid(f, "pooled", grid, by=())
        e1.assert_full_grid(f, "per horizon", grid, by=("horizon",))           # b's gap is hidden
        with self.assertRaises(e1.MissingTauError) as cm:
            e1.assert_full_grid(f, "per dataset", grid, by=("dataset", "horizon"))
        self.assertIn("0.9", str(cm.exception))
        self.assertIn("'b'", str(cm.exception))
        with self.assertRaises(e1.MissingTauError):
            e1.assert_full_grid(f[f.tau != 0.5], "pooled", grid, by=())
        e1.assert_full_grid(f.assign(tau=f.tau + 1e-12), "float noise", grid, by=("horizon",))


class DeltaOrientation(unittest.TestCase):
    """Fix 6: Delta = J_TSFM - J_classical, negative means TSFM has the lower cost."""

    def test_project_overview_still_defines_it_this_way(self):
        text = OVERVIEW.read_text(encoding="utf-8")
        self.assertIn("Delta(x, tau) = J_TSFM(x, tau) \u2212 J_classical(x, tau)", text)
        self.assertIn("`Delta < 0` \u2192 the TSFM family is decision-optimal", text)

    def test_delta_is_tsfm_minus_classical_on_scaled_cost(self):
        t = _tiny_tables([("a", 0.5, 0.3, 2.0, 1.0, 10.0),      # TSFM cheaper
                          ("b", 0.5, 0.4, 1.0, 4.0, 2.0),       # classical cheaper
                          ("c", 0.5, 0.5, 3.0, 3.0, 6.0)])      # tie
        m = e1.decision_map(t).set_index("series_id")
        self.assertAlmostEqual(m.loc["a", "delta"], -0.1, places=15)
        self.assertEqual(m.loc["a", "lower_cost_family"], "tsfm")
        self.assertAlmostEqual(m.loc["b", "delta"], 1.5, places=15)
        self.assertEqual(m.loc["b", "lower_cost_family"], "classical")
        self.assertEqual(m.loc["c", "delta"], 0.0)
        self.assertEqual(m.loc["c", "lower_cost_family"], "tie")
        self.assertAlmostEqual(m.loc["a", "scaled_cost_classical"], 0.2, places=15)
        self.assertAlmostEqual(m.loc["a", "scaled_cost_tsfm"], 0.1, places=15)
        s = e1.summarize_map(e1.decision_map(t)).iloc[0]
        self.assertEqual((s.n_units, s.n_tsfm_lower, s.n_classical_lower, s.n_ties), (3, 1, 1, 1))
        self.assertAlmostEqual(s.share_tsfm_lower, 1 / 3)
        self.assertAlmostEqual(s.mean_delta, (-0.1 + 1.5 + 0.0) / 3)

    def test_ties_are_judged_on_scaled_cost(self):
        """The reviewer's case: costs 5e-13 and 1e-13 on a series of scale
        1e-12. One family is five times cheaper; an absolute tolerance on the
        raw costs called it a tie."""
        m = e1.decision_map(_tiny_tables([("a", 0.5, 0.3, 5e-13, 1e-13, 1e-12),
                                          ("b", 0.5, 0.4, 5.0, 5.0 + 1e-13, 1.0)])).set_index("series_id")
        self.assertFalse(m.loc["a", "is_tie"])
        self.assertEqual(m.loc["a", "lower_cost_family"], "tsfm")
        self.assertAlmostEqual(m.loc["a", "delta"], -0.4, places=12)
        self.assertTrue(m.loc["b", "is_tie"])                   # equal within 1e-12 of the scaled cost

    def test_non_finite_or_missing_inputs_raise(self):
        for name, row in (("inf cost", ("a", 0.5, 0.3, np.inf, 1.0, 1.0)),
                          ("nan cost", ("a", 0.5, 0.3, np.nan, 1.0, 1.0)),
                          ("zero scale", ("a", 0.5, 0.3, 1.0, 2.0, 0.0)),
                          ("inf omega", ("a", 0.5, np.inf, 1.0, 2.0, 1.0))):
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    e1.decision_map(_tiny_tables([row]))

    def test_nan_omega_is_kept_as_undefined(self):
        m = e1.decision_map(_tiny_tables([("a", 0.5, np.nan, 1.0, 2.0, 1.0)]))
        self.assertTrue(np.isnan(m["omega"].iloc[0]))
        self.assertTrue(np.isfinite(m["delta"].iloc[0]))


class DecisionMapOnFixture(unittest.TestCase):
    def setUp(self):
        self.tables = _shared.tables()
        self.res, _ = _shared.results()
        self.out = _fixture_out()

    def test_tau_flip_matches_planted_answers(self):
        planted = planted_answers()
        planted = planted[planted.dataset == "fx_tau_flip"].set_index("tau")
        gap = planted.exp_J_tsfm - planted.exp_J_classical       # same sign as Delta
        # the planted design itself: TSFM at the median, classical at both extremes
        self.assertEqual(planted.loc[0.5, "planted_pinball_winner"], "tsfm")
        self.assertEqual(planted.loc[0.01, "planted_pinball_winner"], "classical")
        self.assertEqual(planted.loc[0.99, "planted_pinball_winner"], "classical")
        self.assertLess(gap[0.5], 0)                             # TSFM cheaper means Delta below zero
        s = self.out["summary_by_dataset"]
        s = s[s.dataset == "fx_tau_flip"]
        checked = 0
        for r in s.itertuples():
            if abs(gap[r.tau]) < CLEAR_MARGIN:
                continue
            checked += 1
            with self.subTest(horizon=r.horizon, tau=r.tau):
                want = planted.loc[r.tau, "planted_pinball_winner"]
                self.assertEqual(r.n_tsfm_lower, r.n_units if want == "tsfm" else 0)
                self.assertEqual(np.sign(r.mean_delta), np.sign(gap[r.tau]))
        self.assertEqual(checked, 2 * 9)       # 9 of 11 taus have a clear planted gap

    def test_tau_flip_is_tsfm_near_median_and_classical_in_both_tails(self):
        s = self.out["summary_by_dataset"]
        s = s[s.dataset == "fx_tau_flip"].set_index(["horizon", "tau"])
        for h in (1, 24):
            for tau in (0.40, 0.50, 0.60):
                with self.subTest(horizon=h, tau=tau):
                    self.assertEqual(s.loc[(h, tau), "share_tsfm_lower"], 1.0)
                    self.assertLess(s.loc[(h, tau), "mean_delta"], 0)
            for tau in (0.01, 0.05, 0.09, 0.91, 0.95, 0.99):
                with self.subTest(horizon=h, tau=tau):
                    self.assertEqual(s.loc[(h, tau), "share_tsfm_lower"], 0.0)
                    self.assertGreater(s.loc[(h, tau), "mean_delta"], 0)

    def test_exact_tie_dataset_is_reported_as_ties(self):
        m = self.out["map"]
        tie = m[m.dataset == "fx_tie"]
        self.assertTrue(tie.is_tie.all())
        self.assertTrue((tie.lower_cost_family == "tie").all())
        self.assertFalse(m[m.dataset != "fx_tie"].is_tie.any())
        s = self.out["summary_by_dataset"]
        s = s[s.dataset == "fx_tie"]
        self.assertTrue((s.n_ties == s.n_units).all())
        self.assertTrue((s.share_tsfm_lower == 0).all())
        self.assertTrue((s.n_sign_reversals == 0).all())

    def test_delta_agrees_with_engine_oracle(self):
        """Delta < 0 exactly where the engine's oracle picks TSFM."""
        orc = self.res[self.res.selector == "hindsight_oracle"]
        m = self.out["map"].merge(orc, on=e1.UNIT_KEYS, validate="1:1")
        self.assertEqual(len(m), 484)
        self.assertTrue(((m.delta < 0) & ~m.is_tie == (m.oracle_family == "tsfm")).all())
        self.assertEqual(int((m.oracle_family == "tsfm").sum()), int((m.lower_cost_family == "tsfm").sum()))
        self.assertGreater(int((m.oracle_family == "tsfm").sum()), 100)
        self.assertTrue(np.allclose(np.minimum(m.scaled_cost_classical, m.scaled_cost_tsfm) * m.scale_x,
                                    m.oracle_cost, rtol=1e-12, atol=0))

    def test_selector_regret_columns_match_engine_summary(self):
        eng = summarize(self.res).set_index(["selector", "horizon", "tau"]).mean_scaled_regret
        s = self.out["summary"].set_index(["horizon", "tau"])
        for sel in self.res.selector.unique():
            with self.subTest(selector=sel):
                col = s[f"regret_{sel}"]
                want = eng.xs(sel, level="selector").reindex(col.index)
                self.assertTrue(np.allclose(col, want, rtol=1e-12, atol=1e-15))
                self.assertEqual(set(s[f"n_selected_{sel}"]), {22})
        self.assertTrue((s["regret_hindsight_oracle"] == 0).all())

    def test_map_has_the_columns_the_boundary_fit_needs_and_no_fit(self):
        self.assertEqual(list(self.out["map"].columns), e1.BOUNDARY_INPUT_COLUMNS)
        fitted = [c for t in self.out.values() for c in t.columns
                  if "boundary" in c or "omega_star" in c or "beta" in c]
        self.assertEqual(fitted, [])

    def test_cli_writes_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            fixture, build = _shared.cli_patches()
            with fixture, build, contextlib.redirect_stdout(io.StringIO()) as printed:
                e1.main(["--out", d, "--n-boot", "20"])
            text = printed.getvalue()
            self.assertIn("SYNTHETIC FIXTURE", text)
            self.assertIn("NOMINAL 95%", text)
            self.assertIn("20 resamples", text)
            names = {p.name for p in Path(d).iterdir()}
            self.assertEqual(names, {"decision_map.parquet", "decision_map.csv", "summary_by_horizon_tau.csv",
                                     "summary_by_dataset_horizon_tau.csv", "symmetry.csv",
                                     "mean_delta_intervals.csv", "manifest.json"})
            m = pd.read_parquet(Path(d) / "decision_map.parquet")
            self.assertEqual(sorted(m.tau.unique()), sorted(TAU_GRID))
            pd.testing.assert_frame_equal(m, self.out["map"])
            manifest = json.loads((Path(d) / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("synthetic", manifest["data"])
            self.assertIn("J_tsfm/scale - J_classical/scale", manifest["delta_definition"])
            self.assertIn("delta < 0 means TSFM has the lower cost", manifest["delta_definition"])
            self.assertNotIn("delta_sign", manifest)
            self.assertEqual((manifest["n_boot"], manifest["time_blocks"]), (20, "per_series"))
            ci = pd.read_csv(Path(d) / "mean_delta_intervals.csv")
            self.assertEqual(len(ci), 22)
            self.assertEqual(set(ci.n_boot), {20})
            self.assertEqual(sorted(ci.block_lengths_used.unique()), ["h1: L=7, n=320", "h24: L=3, n=20"])
            summary = pd.read_csv(Path(d) / "summary_by_horizon_tau.csv")
            self.assertTrue(np.allclose(ci.sort_values(["horizon", "tau"]).mean_delta,
                                        summary.sort_values(["horizon", "tau"]).mean_delta, atol=1e-12))


class SelectorRegret(unittest.TestCase):
    def test_undecided_units_are_not_counted_as_selected(self):
        """Four units, one of them undecided. The mean is over three and n says three."""
        sel = pd.DataFrame(dict(selector="s", horizon=1, tau=0.5,
                                status=["selected", "selected", "selected", "insufficient_data"],
                                scaled_regret=[0.1, 0.2, 0.3, np.nan]))
        other = sel.assign(selector="t", status="insufficient_data", scaled_regret=np.nan)
        r = e1.selector_regret(pd.concat([sel, other], ignore_index=True)).iloc[0]
        self.assertAlmostEqual(r.regret_s, 0.2, places=15)
        self.assertEqual(r.n_selected_s, 3)
        self.assertEqual(r.n_selected_t, 0)
        self.assertTrue(np.isnan(r.regret_t))


class SignReversals(unittest.TestCase):
    def test_known_counts(self):
        cases = [
            ("no boundary", [0.1, 0.2, 0.3], [1, 2, 3], 0),
            ("one clean boundary", [0.1, 0.2, 0.3, 0.4], [-1, -2, 3, 1], 1),
            ("three reversals", [0.1, 0.2, 0.3, 0.4, 0.5], [-1, -1, 1, -1, 1], 3),
            ("input order does not matter", [0.5, 0.1, 0.4, 0.2, 0.3], [1, -1, -1, -1, 1], 3),
            ("tie between opposite signs", [0.1, 0.2, 0.3], [1, 0, -1], 1),
            ("tie between equal signs", [0.1, 0.2, 0.3], [1, 0, 1], 0),
            ("only ties", [0.1, 0.2], [0, 0], 0),
            ("single unit", [0.1], [5], 0),
            ("empty", [], [], 0),
            ("equal omega averaged to plus", [0.1, 0.2, 0.2, 0.3], [-1, 3, -1, -1], 2),
            ("equal omega averaged to minus", [0.1, 0.2, 0.2, 0.3], [-1, 1, -3, -1], 0),
            ("equal omega averaged to zero is skipped", [0.1, 0.2, 0.2, 0.3], [-1, 1, -1, 1], 1),
        ]
        for name, om, de, want in cases:
            with self.subTest(case=name):
                self.assertEqual(e1.count_sign_reversals(om, de), want)
                self.assertEqual(e1.count_sign_reversals(om, [-x for x in de]), want)   # orientation-free

    def test_explicit_tie_mask_overrides_tiny_values(self):
        om, de = [0.1, 0.2, 0.3], [1.0, -1e-18, 1.0]
        self.assertEqual(e1.count_sign_reversals(om, de), 2)
        self.assertEqual(e1.count_sign_reversals(om, de, is_tie=[False, True, False]), 0)

    def test_summary_uses_it_per_group(self):
        rows = [(f"s{i}", 0.5, om, 1.0 + d, 1.0, 1.0)
                for i, (om, d) in enumerate([(0.1, -0.5), (0.2, 0.5), (0.3, -0.5), (0.4, 0.5)])]
        rows += [(f"s{i}", 0.6, om, 1.0 + d, 1.0, 1.0)
                 for i, (om, d) in enumerate([(0.1, -0.5), (0.2, -0.5), (0.3, 0.5), (0.4, 0.5)])]
        s = e1.summarize_map(e1.decision_map(_tiny_tables(rows))).set_index("tau")
        self.assertEqual(s.loc[0.5, "n_sign_reversals"], 3)
        self.assertEqual(s.loc[0.6, "n_sign_reversals"], 1)

    def test_bad_input_raises(self):
        with self.assertRaises(ValueError):
            e1.count_sign_reversals([0.1, 0.2], [1.0])
        with self.assertRaises(ValueError):
            e1.count_sign_reversals([0.1, np.nan], [1.0, 2.0])


class Symmetry(unittest.TestCase):
    def _summary(self):
        share = {0.01: 0.0, 0.05: 0.1, 0.09: 0.2, 0.25: 0.5, 0.40: 0.9, 0.50: 1.0,
                 0.60: 0.9, 0.75: 0.3, 0.91: 0.6, 0.95: 0.1, 0.99: 0.25}
        delta = {0.01: -2.0, 0.05: -1.0, 0.09: -0.5, 0.25: 0.0, 0.40: 1.0, 0.50: 2.0,
                 0.60: 1.0, 0.75: -0.75, 0.91: 0.25, 0.95: -1.0, 0.99: -0.5}
        # sign reversals differ on the two sides of every pair: k at tau below 0.5, 10 + k above
        return pd.DataFrame([dict(horizon=7, tau=t, n_units=20, share_tsfm_lower=share[t],
                                  mean_delta=delta[t], n_sign_reversals=k if t < 0.5 else 10 + k)
                             for k, t in enumerate(TAU_GRID)])

    def test_pairs(self):
        got = e1.tau_pairs()
        self.assertEqual(len(got), 5)
        for (a, b), (wa, wb) in zip(got, PAIRS):
            self.assertAlmostEqual(a, wa)
            self.assertAlmostEqual(b, wb)

    def test_hand_built_differences_are_upper_minus_lower(self):
        sym = e1.symmetry_table(self._summary()).set_index("tau_lower")
        self.assertEqual(len(sym), 5)
        want_share = {0.01: 0.25, 0.05: 0.0, 0.09: 0.4, 0.25: -0.2, 0.40: 0.0}
        want_delta = {0.01: 1.5, 0.05: 0.0, 0.09: 0.75, 0.25: -0.75, 0.40: 0.0}
        grid = list(TAU_GRID)
        for lo, hi in PAIRS:
            with self.subTest(pair=(lo, hi)):
                self.assertAlmostEqual(sym.loc[lo, "tau_upper"], hi)
                self.assertAlmostEqual(sym.loc[lo, "diff_share_tsfm_lower"], want_share[lo])
                self.assertAlmostEqual(sym.loc[lo, "diff_mean_delta"], want_delta[lo])
                self.assertAlmostEqual(sym.loc[lo, "distance_from_half"], 0.5 - lo)
                self.assertEqual((sym.loc[lo, "n_units_lower"], sym.loc[lo, "n_units_upper"]), (20, 20))
                self.assertEqual(sym.loc[lo, "n_sign_reversals_lower"], grid.index(lo))
                self.assertEqual(sym.loc[lo, "n_sign_reversals_upper"], 10 + grid.index(hi))

    def test_unequal_populations_are_refused(self):
        """A pair computed on 20 units below 0.5 and 18 above mixes a tau effect
        with a population effect."""
        s = self._summary()
        s.loc[s.tau == 0.95, "n_units"] = 18
        with self.assertRaises(e1.UnequalPopulationError) as cm:
            e1.symmetry_table(s)
        self.assertIn("20", str(cm.exception))
        self.assertIn("18", str(cm.exception))
        s = self._summary()
        s.loc[s.tau == 0.25, "n_units"] = 19                    # the lower side this time
        with self.assertRaises(e1.UnequalPopulationError):
            e1.symmetry_table(s)

    def test_symmetric_map_gives_zero(self):
        s = self._summary()
        for lo, hi in PAIRS:
            for c in ("share_tsfm_lower", "mean_delta"):
                s.loc[s.tau == hi, c] = s.loc[s.tau == lo, c].to_numpy()
        sym = e1.symmetry_table(s)
        self.assertTrue((sym.diff_share_tsfm_lower == 0).all())
        self.assertTrue((sym.diff_mean_delta == 0).all())

    def test_missing_side_fails(self):
        s = self._summary()
        with self.assertRaises(e1.MissingTauError):
            e1.symmetry_table(s[s.tau != 0.75])
        with self.assertRaises(e1.MissingTauError):
            e1.tau_pairs([0.1, 0.5, 0.8])
        with self.assertRaises(e1.MissingTauError):
            e1.tau_pairs([0.5, 0.8])

    def test_planted_asymmetry_through_map_summary_and_table(self):
        """Known answer from costs to the symmetry row. At tau 0.25 both units
        have TSFM lower (delta -0.2, -0.4); at 0.75 one has (delta -0.1) and
        one has classical lower (delta +0.5)."""
        rows = [("a", 0.25, 0.3, 1.2, 1.0, 1.0), ("b", 0.25, 0.4, 1.4, 1.0, 1.0),
                ("a", 0.50, 0.3, 1.0, 1.0, 1.0), ("b", 0.50, 0.4, 1.0, 1.0, 1.0),
                ("a", 0.75, 0.3, 1.1, 1.0, 1.0), ("b", 0.75, 0.4, 1.0, 1.5, 1.0)]
        summary = e1.summarize_map(e1.decision_map(_tiny_tables(rows)))
        sym = e1.symmetry_table(summary, grid=(0.25, 0.5, 0.75))
        self.assertEqual(len(sym), 1)
        r = sym.iloc[0]
        self.assertEqual((r.tau_lower, r.tau_upper, r.n_units_lower, r.n_units_upper), (0.25, 0.75, 2, 2))
        self.assertAlmostEqual(r.share_tsfm_lower_at_tau_lower, 1.0)
        self.assertAlmostEqual(r.share_tsfm_lower_at_tau_upper, 0.5)
        self.assertAlmostEqual(r.diff_share_tsfm_lower, -0.5)
        self.assertAlmostEqual(r.mean_delta_at_tau_lower, -0.3)
        self.assertAlmostEqual(r.mean_delta_at_tau_upper, 0.2)
        self.assertAlmostEqual(r.diff_mean_delta, 0.5)

    def test_fixture_symmetry_rows_against_the_map_itself(self):
        out = _fixture_out()
        sym, m = out["symmetry"], out["map"]
        self.assertEqual(len(sym), 2 * 5)
        self.assertTrue((sym.n_units_lower == 22).all() and (sym.n_units_upper == 22).all())
        # recomputed from the unit rows of the map, not from the summary the table was built from
        cell = m.assign(tsfm=(m.delta < 0) & ~m.is_tie).groupby(["horizon", "tau"]).agg(
            share=("tsfm", "mean"), delta=("delta", "mean"))
        for r in sym.itertuples():
            with self.subTest(horizon=r.horizon, pair=(r.tau_lower, r.tau_upper)):
                self.assertAlmostEqual(r.tau_lower + r.tau_upper, 1.0)
                lo, hi = cell.loc[(r.horizon, r.tau_lower)], cell.loc[(r.horizon, r.tau_upper)]
                self.assertAlmostEqual(r.diff_mean_delta, hi.delta - lo.delta, places=14)
                self.assertAlmostEqual(r.diff_share_tsfm_lower, hi.share - lo.share, places=14)
        self.assertGreater(sym.diff_share_tsfm_lower.abs().max(), 0.1)        # not all zero


class UndefinedOmega(unittest.TestCase):
    """A unit whose Omega is undefined (NaN) stays in the map and is only left
    out of what is ordered by Omega."""

    @classmethod
    def setUpClass(cls):
        from decision_aware_tsfm.analysis.decision.fixture import (
            FAMILY_OF_MODEL, FIT_ONLY_DATASETS, FIXTURE_SCHEMA, FIXTURE_STATIC_THRESHOLD,
            REPRESENTATIVES, make_fixture)
        from decision_aware_tsfm.analysis.decision.pipeline import build_tables, run
        fc, om, _ = make_fixture(seed=7)
        om = om.copy()
        om.loc[om.series_id == "fx_tsfm_wins/s0", "omega_levels"] = np.nan
        cls.tables = build_tables(fc, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, om,
                                  fit_only_datasets=FIT_ONLY_DATASETS)
        cls.selection, _ = run(cls.tables, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD)
        cls.out = e1.run_e1(cls.tables, cls.selection)

    def test_unit_is_kept_with_its_costs_and_a_nan_omega(self):
        m = self.out["map"]
        mine = m[m.series_id == "fx_tsfm_wins/s0"]
        self.assertEqual(len(mine), 2 * len(TAU_GRID))
        self.assertTrue(mine.omega.isna().all())
        self.assertTrue(np.isfinite(mine.delta).all())
        # costs do not depend on Omega, so Delta and the lower-cost family equal the ordinary fixture's
        ref = _fixture_out()["map"]
        ref = ref[ref.series_id == "fx_tsfm_wins/s0"].sort_values(["horizon", "tau"])
        mine = mine.sort_values(["horizon", "tau"])
        np.testing.assert_array_equal(mine.delta.to_numpy(), ref.delta.to_numpy())
        self.assertEqual(mine.lower_cost_family.tolist(), ref.lower_cost_family.tolist())
        self.assertFalse(m[m.series_id != "fx_tsfm_wins/s0"].omega.isna().any())

    def test_summary_counts_it_and_reversals_ignore_it(self):
        s, m = self.out["summary"], self.out["map"]
        self.assertTrue((s.n_undefined_omega == 1).all())
        self.assertTrue((s.n_units == 22).all())                        # still in every unit count
        for r in s.itertuples():
            g = m[(m.horizon == r.horizon) & (m.tau == r.tau) & m.omega.notna()]
            self.assertEqual(r.n_sign_reversals, e1.count_sign_reversals(g.omega, g.delta, g.is_tie))
        self.assertTrue((s.n_selected_static_omega == 21).all())        # the Omega rule deferred on it
        self.assertTrue((s.n_selected_always_tsfm == 22).all())

    def test_ordinary_fixture_has_none(self):
        self.assertTrue((_fixture_out()["summary"].n_undefined_omega == 0).all())

    def test_infinite_omega_is_refused(self):
        t = dict(self.tables)
        u = t["units"].copy()
        u.loc[u.index[0], "omega"] = np.inf
        t["units"] = u
        with self.assertRaisesRegex(ValueError, "omega must be finite"):
            e1.decision_map(t)


if __name__ == "__main__":
    unittest.main()
