"""The seven comparators against planted answers and an independent recomputation."""
import unittest

import numpy as np
import pandas as pd

from decision_aware_tsfm.analysis.decision.cache_io import qcol
from decision_aware_tsfm.analysis.decision.costs import assign_roles
from decision_aware_tsfm.analysis.decision.evaluate import run_selectors, summarize
from decision_aware_tsfm.analysis.decision.fixture import (
    FIXTURE_STATIC_THRESHOLD as THRESHOLD, REPRESENTATIVES)
from decision_aware_tsfm.analysis.decision.guards import FitEvalOverlapError, HeldOutLeakError
from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID
from decision_aware_tsfm.analysis.decision.pipeline import run, seven_selectors
from decision_aware_tsfm.analysis.decision.selectors import DOmegaLinear, SelectorInput

from . import _shared

SEVEN = {"always_classical", "always_tsfm", "static_omega", "pinball_validation",
         "crps_validation", "d_omega", "hindsight_oracle"}


def _brute_force(fc):
    """Independent recomputation of test-period mean pinball per unit, written
    without the costs module: explicit loops, explicit origin split."""
    out = {}
    fc = fc[fc.dataset != "fx_dfit"]
    for (ds, sid, h), g in fc.groupby(["dataset", "series_id", "horizon"]):
        origins = np.sort(g.origin.unique())
        n_val = int(np.floor(len(origins) / 3))
        gt = g[g.origin.isin(set(origins[n_val:].tolist()))]
        for tau in TAU_GRID:
            J = {}
            for fam, model in REPRESENTATIVES.items():
                m = gt[gt.model == model]
                u = m.y.to_numpy() - m[qcol(tau)].to_numpy()
                J[fam] = float(np.mean([max(tau * x, (tau - 1) * x) for x in u]))
            out[(sid, h, tau)] = J
    return out


class SevenComparators(unittest.TestCase):
    def setUp(self):
        self.res, self.d_omega = _shared.results()
        self.tables = _shared.tables()

    def test_all_seven_run_on_every_unit_and_every_tau(self):
        self.assertEqual(set(self.res.selector), SEVEN)
        self.assertEqual(self.res.groupby("selector").size().nunique(), 1)
        self.assertEqual(sorted(self.res.tau.unique()), sorted(TAU_GRID))
        self.assertTrue((self.res.status == "selected").all())

    def test_oracle_zero_regret_and_all_regret_nonnegative(self):
        self.assertTrue((self.res[self.res.selector == "hindsight_oracle"].regret == 0).all())
        self.assertTrue((self.res.regret >= 0).all())

    def test_engine_matches_independent_recomputation(self):
        bf = _brute_force(_shared.fixture_data()[0])
        orc = self.res[self.res.selector == "hindsight_oracle"]
        self.assertEqual(len(orc), len(bf))
        for r in orc.itertuples():
            J = bf[(r.series_id, r.horizon, r.tau)]
            self.assertAlmostEqual(r.oracle_cost / min(J.values()), 1.0, delta=1e-10)
            want = "tsfm" if J["tsfm"] < J["classical"] * (1 - 1e-12) else "classical"
            self.assertEqual(r.oracle_family, want)

    def test_planted_winners_recovered_where_statistically_clear(self):
        """The oracle's winner equals the analytic winner wherever the analytic
        margin exceeds 4 standard errors of the unit's paired mean difference."""
        fc, _, planted = _shared.fixture_data()
        orc = self.res[self.res.selector == "hindsight_oracle"].merge(planted, on=["dataset", "tau"])
        asserted = 0
        for r in orc.itertuples():
            if r.dataset == "fx_tie":
                self.assertEqual(r.oracle_family, "classical")      # exact tie -> classical
                self.assertEqual(r.regret, 0)
                continue
            g = fc[(fc.series_id == r.series_id) & (fc.horizon == r.horizon)]
            origins = np.sort(g.origin.unique())
            g = g[g.origin.isin(origins[len(origins) // 3:])]
            d = []
            for model in (REPRESENTATIVES["classical"], REPRESENTATIVES["tsfm"]):
                m = g[g.model == model]
                u = m.y.to_numpy() - m[qcol(r.tau)].to_numpy()
                d.append(np.maximum(r.tau * u, (r.tau - 1) * u))
            diff = d[0] - d[1]
            se = diff.std(ddof=1) / np.sqrt(diff.size)
            if abs(r.exp_J_classical - r.exp_J_tsfm) > 4 * se:
                self.assertEqual(r.oracle_family, r.planted_pinball_winner, (r.dataset, r.series_id, r.tau))
                asserted += 1
        self.assertGreaterEqual(asserted, 200)          # not vacuous

    def test_fixed_family_and_omega_rules(self):
        res = self.res
        self.assertTrue((res[res.selector == "always_classical"].chosen_family == "classical").all())
        self.assertTrue((res[res.selector == "always_tsfm"].chosen_family == "tsfm").all())
        st = res[res.selector == "static_omega"]
        self.assertTrue(((st.omega > THRESHOLD) == (st.chosen_family == "tsfm")).all())

    def test_static_rule_pays_regret_in_tails_of_tau_flip(self):
        st = self.res.query("selector == 'static_omega' and dataset == 'fx_tau_flip'")
        self.assertTrue((st[st.tau.isin([0.01, 0.05, 0.99])].regret > 0).all())

    def test_crps_choice_does_not_depend_on_tau(self):
        c = self.res[self.res.selector == "crps_validation"]
        self.assertTrue((c.groupby(["series_id", "horizon"]).chosen_family.nunique() == 1).all())

    def test_non_hindsight_selectors_cannot_see_test_outcomes(self):
        """Swap the two families' test costs. Only the oracle may change its choices."""
        t = self.tables
        sels = seven_selectors(THRESHOLD, DOmegaLinear.fit(t["fit_units"]))
        a = run_selectors(sels, t["units"], t["val"], t["test"], REPRESENTATIVES, t["n_val"])
        t2 = t["test"].rename(columns={"J_classical": "J_tsfm", "J_tsfm": "J_classical"})
        b = run_selectors(sels, t["units"], t["val"], t2, REPRESENTATIVES, t["n_val"])
        for s in {x.name for x in sels} - {"hindsight_oracle"}:
            self.assertTrue((a[a.selector == s].chosen_family.to_numpy()
                             == b[b.selector == s].chosen_family.to_numpy()).all(), s)
        o = "hindsight_oracle"
        self.assertFalse((a[a.selector == o].chosen_family.to_numpy()
                          == b[b.selector == o].chosen_family.to_numpy()).all())

    def test_non_hindsight_selectors_run_without_test_table(self):
        t = self.tables
        inp = SelectorInput(units=t["units"], val=t["val"], test=None)
        for s in seven_selectors(THRESHOLD, DOmegaLinear.fit(t["fit_units"])):
            if not s.needs_hindsight:
                s.select(inp)

    def test_d_omega_recovers_planted_boundary(self):
        # fitted on the fit-only dataset alone, where the boundary is planted at 0.45
        _, d = run(self.tables, REPRESENTATIVES, THRESHOLD, fit_scheme="fit_only_datasets")
        self.assertAlmostEqual(d.omega0, 0.45, places=12)
        self.assertEqual(d.beta, 0.0)
        self.assertEqual(d.fitted_datasets, ("fx_dfit",))

    def test_d_omega_refuses_held_out(self):
        fu = self.tables["fit_units"].copy()
        fu.loc[fu.index[:3], "dataset"] = "M5_Retail"
        with self.assertRaises(HeldOutLeakError):
            DOmegaLinear.fit(fu)

    def test_d_omega_refuses_to_evaluate_on_its_fit_series(self):
        fu = self.tables["fit_units"]
        d = DOmegaLinear.fit(fu)
        with self.assertRaises(FitEvalOverlapError):
            d.select(SelectorInput(units=fu[["dataset", "config", "series_id", "horizon", "tau", "omega"]], val=fu))

    def test_d_omega_empty_fit_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "no usable units"):
            DOmegaLinear.fit(self.tables["fit_units"].iloc[0:0])

    def test_normalised_value_undefined_on_exact_tie(self):
        s = summarize(self.res, by=("selector", "dataset"))
        tie = s[s.dataset == "fx_tie"]
        self.assertTrue(tie.frac_oracle_value.isna().all())
        self.assertTrue((tie.frac_note != "").all())
        flip = s[(s.dataset == "fx_tau_flip") & (s.selector == "hindsight_oracle")]
        self.assertAlmostEqual(flip.frac_oracle_value.iloc[0], 1.0, places=12)

    def test_units_without_validation_origins_are_insufficient_data(self):
        t = self.tables
        val = t["val"][t["val"].series_id != "fx_tsfm_wins/s0"]
        res = run_selectors(seven_selectors(THRESHOLD, DOmegaLinear.fit(t["fit_units"])),
                            t["units"], val, t["test"], REPRESENTATIVES, t["n_val"])
        s0 = res[res.series_id == "fx_tsfm_wins/s0"]
        for name in ("pinball_validation", "crps_validation"):
            self.assertTrue((s0[s0.selector == name].status == "insufficient_data").all())
        self.assertTrue((s0[s0.selector == "always_tsfm"].status == "selected").all())


class TableSettings(unittest.TestCase):
    def test_tables_record_how_they_were_built(self):
        s = _shared.tables()["settings"]
        self.assertEqual(s["representatives"], REPRESENTATIVES)
        self.assertAlmostEqual(s["val_fraction"], 1 / 3)
        self.assertEqual(s["omega_col"], "omega_levels")
        self.assertIsNone(s["omega_variant"])
        self.assertEqual(s["taus"], tuple(TAU_GRID))
        self.assertEqual(s["fit_only_datasets"], ("fx_dfit",))
        self.assertEqual(set(s["models_scored"]), set(REPRESENTATIVES.values()))   # non-representatives are not scored


class OriginRoles(unittest.TestCase):
    def test_validation_origins_overlapping_test_period_become_gap(self):
        # origin = first target, so a window covers origin .. origin+23
        po = pd.DataFrame({"dataset": "d", "config": "c", "series_id": "s", "horizon": 24,
                           "origin": np.arange(100, 400)})          # stride 1: windows overlap
        r = assign_roles(po, 1 / 3).drop_duplicates("origin").set_index("origin").role
        self.assertEqual(r[r == "test"].index.min(), 200)           # floor(300 / 3) = 100 candidates
        self.assertTrue((r.loc[100:176] == "validation").all())     # 176 covers 176..199, before 200
        self.assertTrue((r.loc[177:199] == "gap").all())            # would reach 200 or later
        self.assertTrue((r.loc[200:] == "test").all())
        self.assertEqual(int((r == "validation").sum()), 77)


if __name__ == "__main__":
    unittest.main()
