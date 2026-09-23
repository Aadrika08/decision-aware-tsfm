"""Deterministic fixture cache with planted answers.

Every series is y_t = mu_t + e_t, e_t ~ N(0, 1), mu_t a known daily cycle.
Every model forecasts  q_tau = mu_t + bias + spread * z_tau,  z_tau the
standard-normal tau-quantile. Because the noise is Gaussian, the EXPECTED
pinball loss of each model at each tau has a closed form:

    c = bias + spread * z_tau
    E[pinball_tau] = -c * (tau - Phi(c)) + phi(c)

so the winning family at each tau is known before any data is drawn.

Datasets
  fx_tsfm_wins       TSFM exact, classical biased and too wide. TSFM wins every tau.
  fx_classical_wins  mirror image.
  fx_tie             identical forecasts for both families. Exact tie.
  fx_tau_flip        TSFM centred but too narrow; classical correct width, slightly
                     biased. TSFM wins near the median, classical wins the tails.
  fx_dfit            fit-only set for D_Omega: omega <= 0.40 classical wins,
                     omega >= 0.50 TSFM wins.

Omega values are LABELS assigned to fixture series, not computed from them
(white-noise series would all score near 0.06).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .cache_io import qcol
from .metrics import TAU_GRID, crps_weights

FIXTURE_SCHEMA = "fixture-v0"
FAMILY_OF_MODEL = {"AutoETS": "classical", "SeasonalNaive": "classical", "Moirai-small": "tsfm"}
REPRESENTATIVES = {"classical": "AutoETS", "tsfm": "Moirai-small"}   # fixture only, not a decision
FIT_ONLY_DATASETS = ("fx_dfit",)
# NOT the published threshold. Placeholder so the static-rule slot runs on the
# fixture. The real value must be read from arXiv:2511.08884 and recorded.
FIXTURE_STATIC_THRESHOLD = 0.45

GOOD, BAD = (0.0, 1.0), (1.5, 2.0)          # (bias, spread)
DESIGNS = {
    "fx_tsfm_wins":      dict(omegas=[0.62, 0.66, 0.70, 0.74, 0.78, 0.82], tsfm=GOOD, classical=BAD),
    "fx_classical_wins": dict(omegas=[0.10, 0.13, 0.16, 0.19, 0.22, 0.25], tsfm=BAD, classical=GOOD),
    "fx_tie":            dict(omegas=[0.42, 0.43, 0.44, 0.46], tsfm=(0.2, 1.2), classical=(0.2, 1.2)),
    "fx_tau_flip":       dict(omegas=[0.52, 0.54, 0.56, 0.58, 0.60, 0.62], tsfm=(0.0, 0.6), classical=(0.3, 1.0)),
}
DFIT_OMEGAS = [0.10, 0.16, 0.22, 0.28, 0.34, 0.40, 0.50, 0.57, 0.64, 0.71, 0.78, 0.85]
SEASONAL_NAIVE = (2.0, 2.5)                 # non-representative, always worst
HORIZON_ORIGINS = {24: [1000 + 24 * k for k in range(30)],   # non-overlapping windows
                   1: [1000 + k for k in range(480)]}


def expected_pinball(bias: float, spread: float, tau) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    c = bias + spread * stats.norm.ppf(tau)
    return -c * (tau - stats.norm.cdf(c)) + stats.norm.pdf(c)


def _design_table() -> list[tuple[str, str, float, dict]]:
    rows = []
    for ds, d in DESIGNS.items():
        for j, om in enumerate(d["omegas"]):
            rows.append((ds, f"{ds}/s{j}", om, {"tsfm": d["tsfm"], "classical": d["classical"]}))
    for j, om in enumerate(DFIT_OMEGAS):
        fam = {"tsfm": GOOD, "classical": BAD} if om > 0.45 else {"tsfm": BAD, "classical": GOOD}
        rows.append(("fx_dfit", f"fx_dfit/s{j}", om, fam))
    return rows


def make_fixture(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (forecasts in canonical form, omega table, planted answers)."""
    taus = np.asarray(TAU_GRID)
    z = stats.norm.ppf(taus)
    need = max(o + h for h, os in HORIZON_ORIGINS.items() for o in os) + 1
    parts, omega_rows = [], []
    for i, (ds, sid, om, fam) in enumerate(_design_table()):
        rng = np.random.default_rng(seed * 100_003 + i)
        t = np.arange(need)
        mu = 10 + 3 * np.sin(2 * np.pi * t / 24)
        y = mu + rng.standard_normal(need)
        omega_rows.append(dict(dataset=ds, series_id=sid, omega_levels=om))
        models = {"AutoETS": fam["classical"], "Moirai-small": fam["tsfm"], "SeasonalNaive": SEASONAL_NAIVE}
        for h, origins in HORIZON_ORIGINS.items():
            o = np.repeat(origins, h)
            s = np.tile(np.arange(1, h + 1), len(origins))
            tgt = o + s
            for model, (b, sp) in models.items():
                Q = mu[tgt][:, None] + b + sp * z[None, :]
                f = pd.DataFrame(Q, columns=[qcol(x) for x in taus])
                f.insert(0, "y", y[tgt])
                f.insert(0, "step", s)
                f.insert(0, "origin", o)
                f.insert(0, "horizon", h)
                f.insert(0, "model", model)
                f.insert(0, "series_id", sid)
                f.insert(0, "config", "H")
                f.insert(0, "dataset", ds)
                parts.append(f)
    fc = pd.concat(parts, ignore_index=True)
    for c in ("horizon", "origin", "step"):
        fc[c] = fc[c].astype("int64")
    return fc, pd.DataFrame(omega_rows), planted_answers()


def planted_answers() -> pd.DataFrame:
    """Analytic expected cost and winner per (dataset, tau), for pinball and CRPS."""
    taus = np.asarray(TAU_GRID)
    w = crps_weights(taus, "trapezoid")
    out = []
    for ds, d in DESIGNS.items():
        ec = expected_pinball(*d["classical"], taus)
        et = expected_pinball(*d["tsfm"], taus)
        crps_c, crps_t = 2 * (w * ec).sum(), 2 * (w * et).sum()
        for k, tau in enumerate(taus):
            tie = np.isclose(ec[k], et[k], rtol=0, atol=1e-15)
            out.append(dict(dataset=ds, tau=tau, exp_J_classical=ec[k], exp_J_tsfm=et[k],
                            planted_pinball_winner="classical" if (tie or ec[k] < et[k]) else "tsfm",
                            exp_crps_classical=crps_c, exp_crps_tsfm=crps_t,
                            planted_crps_winner="classical" if crps_c <= crps_t else "tsfm"))
    return pd.DataFrame(out)
