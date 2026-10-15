"""Idealised forecasters for E3, exported in the engine's canonical-v1 frame.

These are NOT models and nothing here fits or runs one. Both forecasters are
given the true one-step conditional mean and the true scale; they differ only
in the SHAPE of their quantiles (see design.py):

  G  Gaussian shape:             q_tau = m_t + sigma * z_tau
  F  shape of the true law, with a stated imperfection:
       location variants         q_tau = m_t + sigma * (Q_member(tau) + location_error)
       empirical variant         q_tau = m_t + empirical tau-quantile of the
                                 context-window innovations (positions < N_CONTEXT only,
                                 so nothing from the test period is used)

Horizon is 1: origin = target position, step = 1 (engine convention:
target position = origin + step - 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from decision_aware_tsfm.analysis.decision.cache_io import qcol
from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID

from .design import (CONFIG, DATASET, F_VARIANTS, HORIZON, MEMBERS, MODEL_G, N_CONTEXT,
                     STRUCTURES, FVariant, Member, Structure, innovation_dist)
from .generator import conditional_mean, context_innovations

TAUS = np.asarray(TAU_GRID, dtype=float)
QCOLS = [qcol(t) for t in TAU_GRID]


def true_innovation_quantiles(member: Member, taus=TAUS) -> np.ndarray:
    """Quantiles of the member's standardised innovation law (sigma = 1)."""
    return innovation_dist(member).ppf(np.asarray(taus, dtype=float))


def quantile_offsets(model: str, values: np.ndarray, structure: Structure, member: Member,
                     n_context: int = N_CONTEXT) -> np.ndarray:
    """What the named forecaster adds to the conditional mean, one value per tau."""
    if model == MODEL_G:
        return structure.sigma * stats.norm.ppf(TAUS)
    v: FVariant = next(x for x in F_VARIANTS.values() if x.model == model)
    if v.kind == "location":
        return structure.sigma * (true_innovation_quantiles(member) + v.location_error)
    if v.kind == "empirical":
        return np.quantile(context_innovations(values, structure, n_context), TAUS)
    raise ValueError(f"unknown forecaster kind {v.kind!r}")


def series_frame(values: np.ndarray, sid: str, structure: Structure, member: Member,
                 models, n_context: int = N_CONTEXT) -> pd.DataFrame:
    """canonical-v1 rows for one series: every model on every test target."""
    x = np.asarray(values, dtype=float)
    pos = np.arange(n_context, x.size, dtype="int64")
    m = conditional_mean(x, structure)[pos]
    parts = []
    for model in models:
        Q = m[:, None] + quantile_offsets(model, x, structure, member, n_context)[None, :]
        f = pd.DataFrame(Q, columns=QCOLS)
        f.insert(0, "y", x[pos])
        f.insert(0, "step", np.int64(1))
        f.insert(0, "origin", pos)
        f.insert(0, "horizon", np.int64(HORIZON))
        f.insert(0, "seed", np.int64(0))          # model seed; the forecasters are deterministic
        f.insert(0, "model", model)
        f.insert(0, "series_id", sid)
        f.insert(0, "config", CONFIG)
        f.insert(0, "dataset", DATASET)
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def group_frame(values_by_series: dict, series_rows: pd.DataFrame, models) -> pd.DataFrame:
    """canonical-v1 frame for several series. values_by_series: series_id -> array."""
    parts = [series_frame(values_by_series[r.series_id], r.series_id, STRUCTURES[r.structure],
                          MEMBERS[r.member], models, int(r.n_context))
             for r in series_rows.itertuples()]
    return pd.concat(parts, ignore_index=True)
