"""The seven comparators against planted answers and an independent recomputation."""
import numpy as np
import pandas as pd
import pytest

from decision_aware_tsfm.analysis.decision.cache_io import qcol
from decision_aware_tsfm.analysis.decision.costs import assign_roles
from decision_aware_tsfm.analysis.decision.evaluate import run_selectors, summarize
from decision_aware_tsfm.analysis.decision.fixture import (
    FIXTURE_STATIC_THRESHOLD as THRESHOLD, REPRESENTATIVES)
from decision_aware_tsfm.analysis.decision.guards import FitEvalOverlapError, HeldOutLeakError
from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID
from decision_aware_tsfm.analysis.decision.pipeline import seven_selectors
from decision_aware_tsfm.analysis.decision.selectors import DOmegaLinear, SelectorInput

SEVEN = {"always_classical", "always_tsfm", "static_omega", "pinball_validation",
         "crps_validation", "d_omega", "hindsight_oracle"}


def test_all_seven_run_on_every_unit_and_every_tau(results, tables):
    res = results[0]
    assert set(res.selector) == SEVEN
    assert res.groupby("selector").size().nunique() == 1
    assert sorted(res.tau.unique()) == sorted(TAU_GRID)
    assert (res.status == "selected").all()


def test_oracle_zero_regret_and_all_regret_nonnegative(results):
    res = results[0]
    assert (res[res.selector == "hindsight_oracle"].regret == 0).all()
    assert (res.regret >= 0).all()


def _brute_force(fc):
    """Independent recomputation of test-period mean pinball per unit, written
    without the costs module: explicit loops, explicit origin split."""
    out = {}
    fc = fc[fc.dataset != "fx_dfit"]
    for (ds, sid, h), g in fc.groupby(["dataset", "series_id", "horizon"]):
        origins = np.sort(g.origin.unique())
        n_val = int(np.floor(len(origins) / 3))
        test_o = set(origins[n_val:].tolist())
        gt = g[g.origin.isin(test_o)]
        for tau in TAU_GRID:
            J = {}
            for fam, model in REPRESENTATIVES.items():
                m = gt[gt.model == model]
                u = m.y.to_numpy() - m[qcol(tau)].to_numpy()
                J[fam] = float(np.mean([max(tau * x, (tau - 1) * x) for x in u]))
            out[(sid, h, tau)] = J
    return out


def test_engine_matches_independent_recomputation(results, fixture_data):
    res = results[0]
    bf = _brute_force(fixture_data[0])
    orc = res[res.selector == "hindsight_oracle"]
    assert len(orc) == len(bf)
    for r in orc.itertuples():
        J = bf[(r.series_id, r.horizon, r.tau)]
        assert r.oracle_cost == pytest.approx(min(J.values()), rel=1e-10)
        want = "tsfm" if J["tsfm"] < J["classical"] * (1 - 1e-12) else "classical"
        assert r.oracle_family == want


def test_planted_winners_recovered_where_statistically_clear(results, fixture_data):
    """Assert the oracle's winner equals the analytic winner only where the
    analytic margin exceeds 4 standard errors of the unit's paired mean difference."""
    fc, _, planted = fixture_data
    res = results[0]
    orc = res[res.selector == "hindsight_oracle"].merge(planted, on=["dataset", "tau"])
    asserted = 0
    for r in orc.itertuples():
        if r.dataset == "fx_tie":
            assert r.oracle_family == "classical" and r.regret == 0     # exact tie -> classical
            continue
        g = fc[(fc.series_id == r.series_id) & (fc.horizon == r.horizon)]
        origins = np.sort(g.origin.unique()); test_o = origins[len(origins) // 3:]
        g = g[g.origin.isin(test_o)]
        d = []
        for model in (REPRESENTATIVES["classical"], REPRESENTATIVES["tsfm"]):
            m = g[g.model == model]; u = m.y.to_numpy() - m[qcol(r.tau)].to_numpy()
            d.append(np.maximum(r.tau * u, (r.tau - 1) * u))
        diff = d[0] - d[1]
        se = diff.std(ddof=1) / np.sqrt(diff.size)
        if abs(r.exp_J_classical - r.exp_J_tsfm) > 4 * se:
            assert r.oracle_family == r.planted_pinball_winner, (r.dataset, r.series_id, r.tau)
            asserted += 1
    print(f'\nplanted winners asserted on {asserted} units')
    assert asserted >= 200, asserted          # not vacuous


def test_fixed_family_and_omega_rules(results):
    res = results[0]
    assert (res[res.selector == "always_classical"].chosen_family == "classical").all()
    assert (res[res.selector == "always_tsfm"].chosen_family == "tsfm").all()
    st = res[res.selector == "static_omega"]
    assert ((st.omega > THRESHOLD) == (st.chosen_family == "tsfm")).all()


def test_static_rule_pays_regret_in_tails_of_tau_flip(results):
    st = results[0].query("selector == 'static_omega' and dataset == 'fx_tau_flip'")
    tails = st[st.tau.isin([0.01, 0.05, 0.99])]
    assert (tails.regret > 0).all()


def test_crps_choice_does_not_depend_on_tau(results):
    c = results[0][results[0].selector == "crps_validation"]
    assert (c.groupby(["series_id", "horizon"]).chosen_family.nunique() == 1).all()


def test_non_hindsight_selectors_cannot_see_test_outcomes(tables):
    """Swap the two families' test costs. Only the oracle may change its choices."""
    d = DOmegaLinear.fit(tables["fit_units"])
    sels = seven_selectors(THRESHOLD, d)
    a = run_selectors(sels, tables["units"], tables["val"], tables["test"], REPRESENTATIVES, tables["n_val"])
    t2 = tables["test"].rename(columns={"J_classical": "J_tsfm", "J_tsfm": "J_classical"})
    b = run_selectors(sels, tables["units"], tables["val"], t2, REPRESENTATIVES, tables["n_val"])
    for s in {x.name for x in sels} - {"hindsight_oracle"}:
        assert (a[a.selector == s].chosen_family.to_numpy() == b[b.selector == s].chosen_family.to_numpy()).all(), s
    o = "hindsight_oracle"
    assert not (a[a.selector == o].chosen_family.to_numpy() == b[b.selector == o].chosen_family.to_numpy()).all()


def test_non_hindsight_selectors_run_without_test_table(tables):
    d = DOmegaLinear.fit(tables["fit_units"])
    inp = SelectorInput(units=tables["units"], val=tables["val"], test=None)
    for s in seven_selectors(THRESHOLD, d):
        if not s.needs_hindsight:
            s.select(inp)


def test_d_omega_recovers_planted_boundary(results):
    d = results[1]
    assert d.omega0 == pytest.approx(0.45) and d.beta == 0.0
    assert d.fitted_datasets == ("fx_dfit",)


def test_d_omega_refuses_held_out(tables):
    fu = tables["fit_units"].copy(); fu.loc[fu.index[:3], "dataset"] = "M5_Retail"
    with pytest.raises(HeldOutLeakError):
        DOmegaLinear.fit(fu)


def test_d_omega_refuses_to_evaluate_on_its_fit_series(tables):
    d = DOmegaLinear.fit(tables["fit_units"])
    fu = tables["fit_units"]
    with pytest.raises(FitEvalOverlapError):
        d.select(SelectorInput(units=fu[["dataset", "config", "series_id", "horizon", "tau", "omega"]], val=fu))


def test_normalised_value_undefined_on_exact_tie(results):
    s = summarize(results[0], by=("selector", "dataset"))
    tie = s[s.dataset == "fx_tie"]
    assert tie.frac_oracle_value.isna().all()
    assert (tie.frac_note != "").all()
    flip = s[(s.dataset == "fx_tau_flip") & (s.selector == "hindsight_oracle")]
    assert flip.frac_oracle_value.iloc[0] == pytest.approx(1.0)


def test_validation_origins_overlapping_test_period_become_gap():
    po = pd.DataFrame({"dataset": "d", "config": "c", "series_id": "s", "horizon": 24,
                       "origin": np.arange(100, 400)})          # stride 1, horizon 24: windows overlap
    r = assign_roles(po, 1 / 3).drop_duplicates("origin").set_index("origin").role
    assert r[r == "test"].index.min() == 200                    # floor(300 / 3) = 100 validation candidates
    assert (r.loc[100:176] == "validation").all()               # 176 + 24 = 200: last target is origin 200, allowed
    assert (r.loc[177:199] == "gap").all()                      # targets would reach into the test period
    assert (r.loc[200:] == "test").all()
    assert (r == "validation").sum() == 77


def test_units_without_validation_origins_are_insufficient_data(tables):
    val = tables["val"][tables["val"].series_id != "fx_tsfm_wins/s0"]
    d = DOmegaLinear.fit(tables["fit_units"])
    res = run_selectors(seven_selectors(THRESHOLD, d), tables["units"], val, tables["test"],
                        REPRESENTATIVES, tables["n_val"])
    s0 = res[res.series_id == "fx_tsfm_wins/s0"]
    for name in ("pinball_validation", "crps_validation"):
        assert (s0[s0.selector == name].status == "insufficient_data").all()
    assert (s0[s0.selector == "always_tsfm"].status == "selected").all()


def test_d_omega_empty_fit_fails_clearly(tables):
    with pytest.raises(ValueError, match="no usable units"):
        DOmegaLinear.fit(tables["fit_units"].iloc[0:0])
