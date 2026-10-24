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

Time convention: origin is the first forecast target, target = origin + step - 1
(shared with the producer, #45).

Also here:
  to_producer_format()  exports any canonical frame in the producer-v1 layout,
                        so every fixture test can run through the production path.
  flat_fixture_producer()  a rebuild, from its description, of the cache owner's
                        flat decision fixture: y = 10, classical = 8, TSFM = 11 at
                        every quantile. Expected mean pinball: classical 2*tau,
                        TSFM 1 - tau, so the winner flips at tau = 1/3.
  make_roster_fixture() the fixture plus a third classical model, so the cache has
                        three classical candidates and one TSFM like the pilot
                        roster. Used by the representative selection (#41).
  series_values()       the whole drawn series behind every fixture series, history
                        before the first origin included.
  make_fixture_scales() the scale table of every fixture series (scale.py: seasonal-
                        naive MAE, season 24, of its values before its first origin,
                        #29 item 7a), which the engine and the representatives need.
  stated_scales()       a STATED scale table for hand-built caches with no history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .cache_io import qcol
from .metrics import TAU_GRID, crps_weights
from .scale import STATUS_OK, scale_row, scale_table

FIXTURE_SCHEMA = "canonical-v1"
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
# origin = first target. Same target positions as before the convention change.
HORIZON_ORIGINS = {24: [1001 + 24 * k for k in range(30)],   # non-overlapping windows
                   1: [1001 + k for k in range(480)]}


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


def _length() -> int:
    return max(o + h - 1 for h, os in HORIZON_ORIGINS.items() for o in os) + 1


def _cycle(need: int) -> np.ndarray:
    return 10 + 3 * np.sin(2 * np.pi * np.arange(need) / 24)


def _draw(seed: int, i: int, need: int) -> np.ndarray:
    return _cycle(need) + np.random.default_rng(seed * 100_003 + i).standard_normal(need)


def series_values(seed: int = 7) -> dict:
    """{(dataset, config, series_id): y at positions 0 .. last target} for every fixture
    series: the very draw make_fixture takes its targets from, history included."""
    need = _length()
    return {(ds, "H", sid): _draw(seed, i, need) for i, (ds, sid, _, _) in enumerate(_design_table())}


def first_origin() -> int:
    """The earliest origin of the fixture (every series has the same origins)."""
    return min(o for os in HORIZON_ORIGINS.values() for o in os)


def make_fixture_scales(seed: int = 7) -> pd.DataFrame:
    """The scale table of every fixture series: seasonal-naive MAE (season 24) of its
    values before the fixture's first origin. About 1.13 for every series (two N(0, 1)
    draws a day apart; the daily cycle cancels), where mean |y| was about 10."""
    h = first_origin()
    return scale_table(scale_row(key, y, h) for key, y in series_values(seed).items())


def make_fixture(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (forecasts in canonical form, omega table, planted answers)."""
    taus = np.asarray(TAU_GRID)
    z = stats.norm.ppf(taus)
    need = _length()
    parts, omega_rows = [], []
    for i, (ds, sid, om, fam) in enumerate(_design_table()):
        mu = _cycle(need)
        y = _draw(seed, i, need)
        omega_rows.append(dict(dataset=ds, config="H", series_id=sid, omega_levels=om))
        models = {"AutoETS": fam["classical"], "Moirai-small": fam["tsfm"], "SeasonalNaive": SEASONAL_NAIVE}
        for h, origins in HORIZON_ORIGINS.items():
            o = np.repeat(origins, h)
            s = np.tile(np.arange(1, h + 1), len(origins))
            tgt = o + s - 1
            for model, (b, sp) in models.items():
                Q = mu[tgt][:, None] + b + sp * z[None, :]
                f = pd.DataFrame(Q, columns=[qcol(x) for x in taus])
                f.insert(0, "y", y[tgt])
                f.insert(0, "step", s)
                f.insert(0, "seed", 0)
                f.insert(0, "origin", o)
                f.insert(0, "horizon", h)
                f.insert(0, "model", model)
                f.insert(0, "series_id", sid)
                f.insert(0, "config", "H")
                f.insert(0, "dataset", ds)
                parts.append(f)
    fc = pd.concat(parts, ignore_index=True)
    for c in ("seed", "horizon", "origin", "step"):
        fc[c] = fc[c].astype("int64")
    return fc, pd.DataFrame(omega_rows), planted_answers()


ROSTER_EXTRA_MODEL = "AutoARIMA"
ROSTER_EXTRA = (0.8, 1.2)                   # (bias, spread), the same in every dataset
ROSTER_FAMILY_OF_MODEL = {**FAMILY_OF_MODEL, ROSTER_EXTRA_MODEL: "classical"}


def make_roster_fixture(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (forecasts in canonical form, omega table): make_fixture plus the
    classical model ROSTER_EXTRA_MODEL. Its forecast is SeasonalNaive's moved to
    the (bias, spread) in ROSTER_EXTRA. make_fixture itself is unchanged."""
    fc, omega, _ = make_fixture(seed)
    z = stats.norm.ppf(np.asarray(TAU_GRID))
    qcols = [qcol(x) for x in TAU_GRID]
    extra = fc[fc.model == "SeasonalNaive"].copy()
    extra["model"] = ROSTER_EXTRA_MODEL
    extra[qcols] = (extra[qcols].to_numpy() + (ROSTER_EXTRA[0] - SEASONAL_NAIVE[0])
                    + (ROSTER_EXTRA[1] - SEASONAL_NAIVE[1]) * z[None, :])
    return pd.concat([fc, extra], ignore_index=True), omega


PRODUCER_SCHEMA = "producer-v1"
PRODUCER_COLUMNS = (["job_id", "dataset", "config", "series_id", "model", "family",
                     "origin", "horizon", "seed", "step", "y_true", "timestamp"]
                    + [qcol(x) for x in TAU_GRID])


def to_producer_format(fc: pd.DataFrame, family_of_model=None,
                       start: str = "2020-01-01T00:00:00", freq: str = "h") -> pd.DataFrame:
    """Export a canonical frame in the producer-v1 layout (column order included).
    Timestamps are start + (origin + step - 1) * freq."""
    fam = family_of_model or FAMILY_OF_MODEL
    out = fc.copy()
    if "seed" not in out.columns:
        out["seed"] = np.int64(0)
    out["family"] = out["model"].map(fam)
    out["job_id"] = (out["dataset"] + "|" + out["config"] + "|" + out["series_id"] + "|"
                     + out["model"] + "|h" + out["horizon"].astype(str)
                     + "|o" + out["origin"].astype(str) + "|s" + out["seed"].astype(str))
    pos = (out["origin"] + out["step"] - 1).to_numpy()
    ts = pd.Timestamp(start) + pd.to_timedelta(pos, unit=freq)
    out["timestamp"] = pd.DatetimeIndex(ts).strftime("%Y-%m-%dT%H:%M:%S")
    out = out.rename(columns={"y": "y_true"})
    for c in ("origin", "horizon", "seed", "step"):
        out[c] = out[c].astype("int64")
    out["y_true"] = out["y_true"].astype("float64")
    return out[PRODUCER_COLUMNS].reset_index(drop=True)


FLAT_FAMILY_OF_MODEL = {"flat-classical": "classical", "flat-tsfm": "tsfm"}
FLAT_REPRESENTATIVES = {"classical": "flat-classical", "tsfm": "flat-tsfm"}


def flat_fixture_producer(origins=(48, 72), horizon: int = 24, seeds=(0, 1)) -> pd.DataFrame:
    """Rebuild of the cache owner's flat fixture in producer-v1 layout."""
    rows = []
    for model, value in (("flat-classical", 8.0), ("flat-tsfm", 11.0)):
        for o in origins:
            for sd in seeds:
                for st in range(1, horizon + 1):
                    r = dict(dataset="fx_flat", config="H", series_id="fx_flat/s0", model=model,
                             seed=sd, horizon=horizon, origin=o, step=st, y=10.0)
                    r.update({qcol(x): value for x in TAU_GRID})
                    rows.append(r)
    fc = pd.DataFrame(rows)
    return to_producer_format(fc, FLAT_FAMILY_OF_MODEL)


FLAT_SCALE = 1.0


def stated_scales(frame: pd.DataFrame, scale: float = FLAT_SCALE) -> pd.DataFrame:
    """A STATED scale table, not a computed one, for hand-built caches with no history
    (the cache owner's flat fixture is y = 10 everywhere, so its seasonal-naive MAE would
    be 0): every series of `frame` gets `scale`, with a history that ends at its first
    origin. Scale 1 makes the scaled costs equal the raw ones. Tests only."""
    first = frame.groupby(["dataset", "config", "series_id"], sort=True)["origin"].min()
    return scale_table([dict(dataset=d, config=c, series_id=sid, scale=float(scale), status=STATUS_OK,
                             history_start=0, history_end=int(o), n_pairs=0)
                        for (d, c, sid), o in first.items()])


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
