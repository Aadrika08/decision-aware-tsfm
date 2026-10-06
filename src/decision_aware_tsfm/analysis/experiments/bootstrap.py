"""Moving-block and cluster bootstrap for statistics of the per-origin cost table.

SCAFFOLDING. This module gives resampling with stated semantics and a
percentile interval. It does not do hypothesis tests, multiple-comparison
corrections, or bias-corrected intervals.

READ THIS BEFORE QUOTING AN INTERVAL. The intervals are approximate. In the
coverage simulation that ships with this module (coverage_sim.py; numbers in
docs/results/experiments/2026-10-04-e1-e2-bootstrap.md) the nominal 95%
interval covers the known answer close to 95% of the time only when origin
costs are independent. Under serial dependence, and whenever series share a
shock at the same origin, it covers less often. An interval from this module
must not be called a "95% confidence interval" without that caveat; the
accurate name is "nominal 95% percentile block-bootstrap interval".

Why blocks. Origin follows the engine convention: it is the index of the first
forecast target, and a window covers origin .. origin + horizon - 1. Two
windows of the same series at origins a < b share at least one target when
b - a < horizon. Costs at such origins are computed partly from the same
observations, so they are not independent, and an interval that resamples
single origins is too narrow. Resampling blocks of consecutive origins keeps
that dependence inside each block. It does not remove the problem: dependence
between blocks is still cut, so the block interval is wider than the naive
one and still too narrow when dependence is strong.

What is resampled, from the outside in:
  1. datasets  (cluster="dataset" or "dataset+series"): whole datasets, with
     replacement. Every series of a drawn dataset comes with it.
  2. series    (cluster="series" or "dataset+series"): whole series, with
     replacement, WITHIN each dataset (the number of series per dataset is
     kept). Every origin of a drawn series comes with it.
  3. origins   (block_length is not None): within each (series, horizon), a
     moving-block resample of the ordered origins. All rows of one origin
     (every tau, every cost column) move together, so dependence across tau is
     kept as well. With time_blocks="per_series" (the default) every series
     draws its own block positions. With time_blocks="shared_within_dataset"
     one set of block positions is drawn per (drawn dataset, horizon) and
     applied to every series of that dataset, which requires that those series
     have identical origins.

What is held fixed: the set of units and the tau grid, the number of series per
dataset, the number of origins per series, Omega, the split into validation and
test origins, the fitted selector, each selector's chosen family per unit, and
(in the default regret statistic) the hindsight-oracle family per unit. See
fixed_choice_costs and stat_mean_scaled_regret.

What the interval is an interval for: the value the statistic would take on
average over evaluation samples drawn the way the resampling draws them, with
everything in the previous paragraph unchanged. For stat_mean_delta that is
the mean scaled cost difference between the two families' representatives.
For stat_mean_scaled_regret it is the mean scaled cost gap between the family
a selector chose and the family that was cheaper on this test sample.

What the interval is known NOT to cover:
  * the variability of fitting or validating a selector on another sample;
  * the selection of the oracle family from the same test costs (reported
    regret is biased upward as an estimate of regret against the truly better
    family, and the interval is centred on the reported value, bias included);
  * shocks shared by series at the same origin, when time_blocks="per_series":
    independent block positions average such shocks away;
  * shocks shared across datasets at the same calendar time, in any mode;
  * dataset-level uncertainty when there is one dataset. M5 and PJM are one
    dataset each, so cluster="dataset" cannot be used for the transfer test and
    raises DegenerateResampleError;
  * serial dependence longer than the block. See default_block_length.

Stacking stages errs the other way. When series are resampled as clusters AND
their origins in blocks, the noise within a series is counted twice: the
variation between the series' means already contains it. The interval is then
wider than needed when the series are alike (in the simulation the nominal 95%
interval covered about 99% in that case). Resampling series alone
(block_length=None) already carries all dependence within a series.

A stage that cannot resample raises DegenerateResampleError instead of
returning the original table again and again (which would give an interval of
width zero that looks like certainty).

Every resampled frame carries two extra columns:
  boot_dataset  integer id of the drawn dataset copy
  boot_series   integer id of the drawn series copy
A series drawn twice appears under two different boot_series ids, so
statistics must group by boot_series, not by series_id.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..decision.schema import UNIT_KEYS
from ..decision.selectors import argmin_family

SERIES_ID_KEYS = ["dataset", "config", "series_id"]
PANEL_KEYS = SERIES_ID_KEYS + ["horizon"]
CLUSTER_LEVELS = ("none", "series", "dataset", "dataset+series")
TIME_BLOCK_MODES = ("per_series", "shared_within_dataset")
BOOT_DATASET, BOOT_SERIES = "boot_dataset", "boot_series"

# Floors below which a stage is refused. They guard against degenerate output;
# passing them does not make an interval accurate (see the coverage simulation).
MIN_BLOCK_STARTS = 5     # distinct block start positions, n_origins - L + 1
MIN_BLOCKS = 2           # a series must be at least this many blocks long
MIN_CLUSTERS = 2         # clusters at any requested cluster level


class DegenerateResampleError(ValueError):
    """A requested resampling stage has too little to resample."""


class MisalignedOriginsError(ValueError):
    """Shared time blocks were requested but series of a dataset have different origins."""


# --------------------------------------------------------------------------- block length

def n_overlapping_origins(horizon: int, origin_step: int) -> int:
    """Number of consecutive origins, counting an origin itself, whose windows
    share at least one target with that origin's window: ceil(horizon / step).

    Origins a and a + j*step overlap when j*step < horizon. With horizon 24
    and step 1 this is 24; with horizon 24 and step 24 (non-overlapping
    windows) it is 1.
    """
    if horizon < 1 or origin_step < 1:
        raise ValueError("horizon and origin_step must be positive integers")
    return int(math.ceil(horizon / origin_step))


def check_block(n_origins: int, block_length: int, where: str = "") -> None:
    """Raise DegenerateResampleError unless a moving-block resample of
    n_origins origins with this block length has something to resample:

        n_origins - block_length + 1 >= MIN_BLOCK_STARTS   distinct block starts
        n_origins >= MIN_BLOCKS * block_length             at least two blocks

    With one possible start every resample is the original series. With a
    series shorter than two blocks the variation between blocks, which is what
    the bootstrap measures, was not observed.
    """
    n, L = int(n_origins), int(block_length)
    if n < 1:
        raise ValueError("n_origins must be positive")
    if L < 1:
        raise ValueError(f"block_length must be a positive integer, got {block_length}")
    starts = n - L + 1
    if starts < MIN_BLOCK_STARTS or n < MIN_BLOCKS * L:
        raise DegenerateResampleError(
            f"{where + ': ' if where else ''}cannot block-resample {n} origins with block length {L}: "
            f"{max(starts, 0)} block start position(s) (need at least {MIN_BLOCK_STARTS}) and "
            f"{n / L:.2f} blocks (need at least {MIN_BLOCKS}). Use more origins, or pass "
            "block_length=None and resample series or datasets as clusters instead")


def _block_rule(horizon: int, origin_step: int, n_origins: int) -> int:
    growth = int(math.ceil(n_origins ** (1.0 / 3.0) - 1e-9))
    return max(1, n_overlapping_origins(horizon, origin_step), growth)


def default_block_length(horizon: int, origin_step: int, n_origins: int) -> int:
    """Default block length, in origins:

        max(ceil(horizon / origin_step), ceil(n_origins ** (1/3)))

    The first term is the overlap floor: a block is at least as long as the
    run of origins whose windows share targets. The second term only stops the
    rule from collapsing to single origins when windows do not overlap (the
    first term is then 1). Neither term is fitted to the dependence in the
    costs, and neither gives calibrated intervals: in the coverage simulation
    the nominal 95% interval with this default covered clearly less than 95%
    for overlapping windows and for strongly autocorrelated costs (numbers in
    the result note). The rule is a documented starting point, not a
    guarantee.

    The value is not capped at n_origins. If the series is too short for it,
    check_block raises DegenerateResampleError: a capped block would be the
    whole series, and every resample would be the original.
    """
    if n_origins < 1:
        raise ValueError("n_origins must be positive")
    L = _block_rule(horizon, origin_step, n_origins)
    check_block(n_origins, L, f"default block length for horizon {horizon}, origin step {origin_step}")
    return L


def origin_step(origins) -> int:
    """Smallest gap between consecutive distinct origins (1 if there is one origin).
    The smallest gap is used so that irregular spacing gives the longer block."""
    o = np.unique(np.asarray(origins))
    if o.size < 2:
        return 1
    return int(np.diff(o).min())


# --------------------------------------------------------------------------- index resamplers

def moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Positions 0..n-1 resampled as moving blocks.

    ceil(n / L) block starts are drawn uniformly, with replacement, from
    0 .. n - L (both ends included). Each start s contributes s, s+1, ...,
    s+L-1. The concatenation is cut to length n, so only the last block can be
    shorter than L. Blocks do not wrap around the end of the series, so the
    first and last L - 1 positions are drawn less often than the middle ones.

    Raises DegenerateResampleError when check_block(n, L) fails.
    """
    L = int(block_length)
    check_block(n, L)
    n_blocks = int(math.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(L)[None, :]).ravel()
    return idx[:n]


def cluster_indices(n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """n_clusters cluster positions drawn uniformly with replacement.
    Raises DegenerateResampleError for fewer than MIN_CLUSTERS clusters."""
    if n_clusters < MIN_CLUSTERS:
        raise DegenerateResampleError(f"cannot resample {n_clusters} cluster(s); need at least "
                                      f"{MIN_CLUSTERS}")
    return rng.integers(0, n_clusters, size=n_clusters)


# --------------------------------------------------------------------------- panel

@dataclass
class _Group:                       # one (series, horizon)
    starts: np.ndarray              # first row position of each origin
    counts: np.ndarray              # number of rows of each origin
    origins: np.ndarray             # the origin values, ascending
    horizon: int
    step: int


class OriginPanel:
    """A per-origin table indexed for resampling.

    frame must have dataset, config, series_id, horizon, origin, and any
    number of further key or value columns. Rows are sorted once; resamples
    are then taken by position. Missing values in the five key columns raise.
    """

    def __init__(self, frame: pd.DataFrame):
        need = PANEL_KEYS + ["origin"]
        missing = [c for c in need if c not in frame.columns]
        if missing:
            raise ValueError(f"per-origin table is missing columns {missing}")
        if frame.empty:
            raise ValueError("per-origin table is empty")
        for c in (BOOT_DATASET, BOOT_SERIES):
            if c in frame.columns:
                raise ValueError(f"column {c!r} is reserved for the resampler")
        bad = [c for c in need if frame[c].isna().any()]
        if bad:
            raise ValueError(f"per-origin table has missing values in key columns {bad}")
        self.frame = frame.sort_values(need, kind="mergesort").reset_index(drop=True)
        f = self.frame
        # one entry per (series, horizon, origin), in sorted order
        first = ~f.duplicated(need).to_numpy()
        starts = np.flatnonzero(first)
        counts = np.diff(np.append(starts, len(f)))
        heads = f.loc[starts, need].reset_index(drop=True)
        self.datasets: list[str] = sorted(heads["dataset"].unique())
        # dataset -> list of series keys ; series key -> list of _Group
        self.series_of: dict[str, list[tuple]] = {d: [] for d in self.datasets}
        self.groups_of: dict[tuple, list[_Group]] = {}
        for key, g in heads.groupby(PANEL_KEYS, sort=True):
            skey = tuple(key[:3])
            if skey not in self.groups_of:
                self.groups_of[skey] = []
                self.series_of[key[0]].append(skey)
            pos = g.index.to_numpy()
            o = g["origin"].to_numpy()
            self.groups_of[skey].append(_Group(starts=starts[pos], counts=counts[pos], origins=o,
                                               horizon=int(key[3]), step=origin_step(o)))
        # column arrays, so a resample is built by position without pandas indexing
        self._columns = {c: (f[c].to_numpy() if isinstance(f[c].dtype, np.dtype) else f[c].array)
                         for c in f.columns}
        self._checked: set = set()
        self._layouts: dict[str, dict[int, _Group]] = {}

    # ------------------------------------------------------------------ block lengths

    def block_length_for(self, grp: _Group, block_length, where: str = "") -> int | None:
        """Block length for one (series, horizon). Raises DegenerateResampleError
        when the series cannot be block-resampled with it."""
        if block_length is None:
            return None
        n = grp.starts.size
        if isinstance(block_length, str):
            if block_length != "auto":
                raise ValueError(f"unknown block_length {block_length!r}")
            L = _block_rule(grp.horizon, grp.step, n)
        else:
            L = int(block_length)
        check_block(n, L, where)
        return L

    def block_lengths(self, block_length) -> pd.DataFrame:
        """The block length each (series, horizon) gets. For reporting."""
        rows = []
        for skey, groups in self.groups_of.items():
            for g in groups:
                where = f"series {skey}, horizon {g.horizon}"
                rows.append(dict(zip(PANEL_KEYS, skey + (g.horizon,)),
                                 n_origins=int(g.starts.size), origin_step=g.step,
                                 block_length=self.block_length_for(g, block_length, where)))
        return pd.DataFrame(rows)

    def describe_block_lengths(self, block_length) -> str:
        """Short text for result tables, e.g. 'h1: L=7, n=320; h24: L=3, n=20'."""
        if block_length is None:
            return "origins not resampled"
        t = self.block_lengths(block_length)
        parts = []
        for h, g in t.groupby("horizon", sort=True):
            rng = lambda s: str(int(s.min())) if s.min() == s.max() else f"{int(s.min())}-{int(s.max())}"
            parts.append(f"h{int(h)}: L={rng(g.block_length)}, n={rng(g.n_origins)}")
        return "; ".join(parts)

    # ------------------------------------------------------------------ checks

    def _shared_layout(self, ds: str) -> dict[int, _Group]:
        """horizon -> one representative group, after checking that every series
        of the dataset has the same origins at that horizon."""
        if ds in self._layouts:
            return self._layouts[ds]
        ref: dict[int, tuple] = {}
        for skey in self.series_of[ds]:
            for g in self.groups_of[skey]:
                if g.horizon not in ref:
                    ref[g.horizon] = (skey, g)
                    continue
                rkey, rg = ref[g.horizon]
                if rg.origins.size != g.origins.size or not np.array_equal(rg.origins, g.origins):
                    raise MisalignedOriginsError(
                        f"time_blocks='shared_within_dataset' needs identical origins for all series "
                        f"of a dataset at a horizon; in dataset {ds!r}, horizon {g.horizon}, series "
                        f"{rkey[2]!r} has {rg.origins.size} origins from {rg.origins[0]} to "
                        f"{rg.origins[-1]} and series {skey[2]!r} has {g.origins.size} from "
                        f"{g.origins[0]} to {g.origins[-1]}")
        self._layouts[ds] = {h: g for h, (_, g) in ref.items()}
        return self._layouts[ds]

    def check_resamplable(self, block_length="auto", cluster: str = "none",
                          time_blocks: str = "per_series") -> None:
        """Raise unless every requested stage can resample. Called before any draw.

        ValueError               unknown option, or nothing requested at all
        DegenerateResampleError  fewer than MIN_CLUSTERS datasets, a dataset with
                                 fewer than MIN_CLUSTERS series, or a series that
                                 fails check_block
        MisalignedOriginsError   shared time blocks with unequal origins
        """
        if cluster not in CLUSTER_LEVELS:
            raise ValueError(f"cluster must be one of {CLUSTER_LEVELS}, got {cluster!r}")
        if time_blocks not in TIME_BLOCK_MODES:
            raise ValueError(f"time_blocks must be one of {TIME_BLOCK_MODES}, got {time_blocks!r}")
        key = (block_length, cluster, time_blocks)
        if key in self._checked:
            return
        if block_length is None and cluster == "none":
            raise ValueError("nothing would be resampled: block_length is None and cluster is 'none'")
        if block_length is None and time_blocks != "per_series":
            raise ValueError("time_blocks='shared_within_dataset' needs a block_length")
        if cluster in ("dataset", "dataset+series") and len(self.datasets) < MIN_CLUSTERS:
            raise DegenerateResampleError(
                f"cluster={cluster!r} resamples datasets, but the table has {len(self.datasets)} "
                f"dataset(s) {self.datasets}; need at least {MIN_CLUSTERS}. Every resample would "
                "contain the same dataset")
        if cluster in ("series", "dataset+series"):
            few = {d: len(s) for d, s in self.series_of.items() if len(s) < MIN_CLUSTERS}
            if few:
                raise DegenerateResampleError(
                    f"cluster={cluster!r} resamples series within each dataset, but these datasets "
                    f"have fewer than {MIN_CLUSTERS} series: {few}")
        if block_length is not None:
            for skey, groups in self.groups_of.items():
                for g in groups:
                    self.block_length_for(g, block_length, f"series {skey}, horizon {g.horizon}")
            if time_blocks == "shared_within_dataset":
                for ds in self.datasets:
                    self._shared_layout(ds)
        self._checked.add(key)

    # ------------------------------------------------------------------ resampling

    def identity(self) -> pd.DataFrame:
        """The original table with boot columns filled in as if every dataset
        and every series had been drawn exactly once."""
        f = self.frame.copy()
        f[BOOT_DATASET] = pd.factorize(f["dataset"], sort=True)[0].astype("int64")
        key = f["dataset"] + "\x1f" + f["config"] + "\x1f" + f["series_id"]
        f[BOOT_SERIES] = pd.factorize(key, sort=True)[0].astype("int64")
        return f

    def resample(self, rng: np.random.Generator, block_length="auto", cluster: str = "none",
                 time_blocks: str = "per_series") -> pd.DataFrame:
        """One resampled table. See the module docstring for the stages."""
        self.check_resamplable(block_length, cluster, time_blocks)
        by_dataset = cluster in ("dataset", "dataset+series")
        by_series = cluster in ("series", "dataset+series")
        shared = block_length is not None and time_blocks == "shared_within_dataset"

        if by_dataset:
            drawn = [self.datasets[i] for i in cluster_indices(len(self.datasets), rng)]
        else:
            drawn = list(self.datasets)

        pos_parts, bd_parts, bs_parts = [], [], []
        series_copy = 0
        for d_copy, ds in enumerate(drawn):
            shared_idx: dict[int, np.ndarray] = {}
            if shared:
                # one draw per horizon, reused by every series of this dataset copy
                for h, g in sorted(self._shared_layout(ds).items()):
                    shared_idx[h] = moving_block_indices(
                        g.starts.size, self.block_length_for(g, block_length), rng)
            series = self.series_of[ds]
            if by_series:
                series = [series[i] for i in cluster_indices(len(series), rng)]
            for skey in series:
                for grp in self.groups_of[skey]:
                    n = grp.starts.size
                    if block_length is None:
                        oi = np.arange(n)
                    elif shared:
                        oi = shared_idx[grp.horizon]
                    else:
                        oi = moving_block_indices(n, self.block_length_for(grp, block_length), rng)
                    cnt = grp.counts[oi]
                    total = int(cnt.sum())
                    offs = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                    pos_parts.append(np.repeat(grp.starts[oi], cnt) + offs)
                    bd_parts.append(np.full(total, d_copy, dtype="int64"))
                    bs_parts.append(np.full(total, series_copy, dtype="int64"))
                series_copy += 1
        pos = np.concatenate(pos_parts)
        data = {c: a.take(pos) for c, a in self._columns.items()}
        data[BOOT_DATASET] = np.concatenate(bd_parts)
        data[BOOT_SERIES] = np.concatenate(bs_parts)
        return pd.DataFrame(data, copy=False)


# --------------------------------------------------------------------------- interval

INTERVAL_KIND = "percentile block bootstrap, approximate (nominal level, not verified coverage)"


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: np.ndarray        # statistic on the original table
    lower: np.ndarray
    upper: np.ndarray
    level: float                # NOMINAL level; see the module docstring
    n_boot: int
    seed: int
    cluster: str
    block_length: object        # "auto", an integer, or None (origins not resampled)
    time_blocks: str
    block_lengths_used: str     # per horizon, e.g. "h24: L=3, n=20"
    replicates: np.ndarray      # (n_boot, k)
    labels: tuple               # index of the statistic if it returned a Series, else ()

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower

    def to_frame(self) -> pd.DataFrame:
        d = pd.DataFrame({"estimate": self.estimate, "lower": self.lower, "upper": self.upper})
        if self.labels:
            d.insert(0, "label", list(self.labels))
        d["nominal_level"], d["n_boot"], d["boot_seed"] = self.level, self.n_boot, self.seed
        d["cluster"] = self.cluster
        d["block_length"] = str(self.block_length)
        d["block_lengths_used"] = self.block_lengths_used
        d["time_blocks"] = self.time_blocks
        d["interval_kind"] = INTERVAL_KIND
        return d


def _as_vector(x) -> tuple[np.ndarray, tuple]:
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float), tuple(x.index.tolist())
    return np.atleast_1d(np.asarray(x, dtype=float)), ()


def bootstrap_ci(frame: pd.DataFrame, statistic: Callable[[pd.DataFrame], object], *,
                 n_boot: int = 1000, level: float = 0.95, seed: int = 0,
                 block_length="auto", cluster: str = "none",
                 time_blocks: str = "per_series") -> BootstrapInterval:
    """Percentile bootstrap interval for statistic(frame). APPROXIMATE: `level`
    is the nominal level, not a verified coverage (module docstring).

    frame      per-origin table (see OriginPanel).
    statistic  function of a table with the same columns plus boot_dataset and
               boot_series. Returns a float, or a pandas Series when several
               quantities should be bootstrapped on the same resamples (for
               example one value per tau).
    block_length  "auto" (default_block_length per series and horizon), an
               integer, or None to leave origins as they are. block_length=1
               resamples single origins independently; it exists only as the
               naive comparison.
    cluster    "none", "series", "dataset" or "dataset+series".
    time_blocks  "per_series": every series draws its own block positions.
               "shared_within_dataset": one draw per (dataset, horizon), used by
               all its series, so a shock that hits every series at the same
               origin is resampled together. Needs identical origins within a
               dataset and raises MisalignedOriginsError otherwise.
    seed       the interval is a deterministic function of (frame, statistic,
               n_boot, level, seed, block_length, cluster, time_blocks).

    The interval is the (alpha/2, 1 - alpha/2) quantile pair of the resampled
    statistics, alpha = 1 - level. A statistic that returns a non-finite value
    on the original table or on any resample raises, so an undefined quantity
    is never silently dropped. A stage that cannot resample raises
    DegenerateResampleError before any resample is drawn.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between 0 and 1")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    panel = OriginPanel(frame)
    panel.check_resamplable(block_length, cluster, time_blocks)
    est, labels = _as_vector(statistic(panel.identity()))
    if not np.isfinite(est).all():
        raise ValueError("statistic is not finite on the original table")
    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, est.size), dtype=float)
    for b in range(n_boot):
        v, lab = _as_vector(statistic(panel.resample(rng, block_length, cluster, time_blocks)))
        if v.size != est.size or lab != labels:
            raise ValueError("statistic changed shape or labels on a resample; with cluster "
                             "resampling, reindex its output to a fixed set of labels")
        reps[b] = v
    if not np.isfinite(reps).all():
        raise ValueError(f"statistic was not finite on {int((~np.isfinite(reps)).any(axis=1).sum())} "
                         f"of {n_boot} resamples")
    a = 1.0 - level
    lower, upper = np.quantile(reps, [a / 2.0, 1.0 - a / 2.0], axis=0)
    return BootstrapInterval(estimate=est, lower=lower, upper=upper, level=level, n_boot=n_boot,
                             seed=seed, cluster=cluster, block_length=block_length,
                             time_blocks=time_blocks,
                             block_lengths_used=panel.describe_block_lengths(block_length),
                             replicates=reps, labels=labels)


# --------------------------------------------------------------------------- cost tables and statistics

COST_COLUMNS = ["pinball_classical", "pinball_tsfm", "abs_y_sum", "n"]
FIXED_COLUMNS = ["pinball_selected", "pinball_oracle"]
ORACLE_CHECK_RTOL = 1e-9


def family_cost_table(per_origin: pd.DataFrame, role: str | None = "test") -> pd.DataFrame:
    """Per-origin costs with the two families side by side.

    Input: the engine's per_origin table (one row per series, horizon, origin,
    family, tau). Output: one row per (series, horizon, tau, origin) with
    pinball_classical, pinball_tsfm (sums over the window's targets),
    abs_y_sum and n (number of targets). role selects origins by their role
    column; None keeps every origin.
    """
    d = per_origin if role is None else per_origin[per_origin["role"] == role]
    if d.empty:
        raise ValueError(f"no per-origin rows with role {role!r}")
    keys = UNIT_KEYS + ["origin"]
    w = d.pivot_table(index=keys, columns="family", values="pinball_sum", aggfunc="first")
    if w.isna().any().any() or set(w.columns) != {"classical", "tsfm"}:
        raise ValueError("some origins do not have both families")
    w.columns = [f"pinball_{c}" for c in w.columns]
    meta = d[d["family"] == "classical"].set_index(keys)[["abs_y_sum", "n"]]
    return w.join(meta).reset_index()[keys + COST_COLUMNS]


def _pick(family, cost: pd.DataFrame) -> np.ndarray:
    return np.where(np.asarray(family) == "classical", cost["pinball_classical"], cost["pinball_tsfm"])


def attach_sample_oracle(cost: pd.DataFrame) -> pd.DataFrame:
    """Add oracle_family and pinball_oracle to a cost table.

    oracle_family is the family with the lower mean cost for the unit
    (dataset, config, series_id, horizon, tau) ON THE ROWS GIVEN, with the
    engine's tie rule (ties go to classical). It is computed once, here, and
    then stays attached to every origin of the unit, so resampling cannot
    change it. For tables that did not come from fixed_choice_costs.
    """
    c = cost.copy()
    if c[COST_COLUMNS].isna().any().any():
        raise ValueError("cost table has missing values")
    g = c.groupby(UNIT_KEYS, sort=False)[COST_COLUMNS].transform("sum")
    c["oracle_family"] = argmin_family(g["pinball_classical"] / g["n"], g["pinball_tsfm"] / g["n"])
    c["pinball_oracle"] = _pick(c["oracle_family"], c)
    return c


def fixed_choice_costs(per_origin: pd.DataFrame, selection: pd.DataFrame, selector: str) -> pd.DataFrame:
    """family_cost_table on TEST origins plus, per origin of every unit,

      pinball_selected  cost of the family `selector` chose for the unit
      pinball_oracle    cost of the unit's hindsight-oracle family

    both read from the selection frame (chosen_family, oracle_family), which
    the engine computed on the real test costs. This is where the two family
    labels are frozen for the bootstrap: they are attached to every origin of
    the unit once, so no resample can change them. Units the selector did not
    decide (status other than 'selected') raise. The function also checks
    that, on the rows given, the oracle family really is the cheaper one for
    every unit; if not, the selection frame and the per-origin table do not
    belong together and it raises.
    """
    s = selection[selection["selector"] == selector]
    if s.empty:
        raise ValueError(f"selector {selector!r} not in the selection frame")
    if (s["status"] != "selected").any():
        raise ValueError(f"selector {selector!r} left {int((s['status'] != 'selected').sum())} "
                         "units undecided; a fixed-choice cost is not defined for them")
    t = family_cost_table(per_origin, "test")
    t = t.merge(s[UNIT_KEYS + ["chosen_family", "oracle_family"]], on=UNIT_KEYS, how="left",
                validate="m:1")
    if t[["chosen_family", "oracle_family"]].isna().any().any():
        raise ValueError("some per-origin units have no decision in the selection frame")
    t["pinball_selected"] = _pick(t["chosen_family"], t)
    t["pinball_oracle"] = _pick(t["oracle_family"], t)
    u = t.groupby(UNIT_KEYS, sort=False)[["pinball_classical", "pinball_tsfm", "pinball_oracle"]].sum()
    low = np.minimum(u["pinball_classical"], u["pinball_tsfm"])
    if (u["pinball_oracle"] > low + ORACLE_CHECK_RTOL * np.maximum(1.0, low.abs())).any():
        raise ValueError("the selection frame's oracle family is not the cheaper family on these "
                         "per-origin rows; the two tables do not belong together")
    return t


def _unit_means(df: pd.DataFrame, extra=()) -> pd.DataFrame:
    """Mean cost per resampled unit (boot_series, horizon, tau). Missing costs raise."""
    cols = COST_COLUMNS + list(extra)
    absent = [c for c in cols if c not in df.columns]
    if absent:
        raise ValueError(f"cost table lacks columns {absent}; build it with fixed_choice_costs "
                         "or attach_sample_oracle")
    if np.isnan(df[cols].to_numpy(dtype=float)).any():
        raise ValueError("cost table has missing values in " +
                         str([c for c in cols if df[c].isna().any()]))
    g = df.groupby([BOOT_SERIES, "horizon", "tau"], sort=True)[cols].sum()
    out = pd.DataFrame({"J_classical": g["pinball_classical"] / g["n"],
                        "J_tsfm": g["pinball_tsfm"] / g["n"],
                        "scale": g["abs_y_sum"] / g["n"]})
    for c in extra:
        out[c] = g[c] / g["n"]
    return out.reset_index()


def _mean(r: pd.Series, u: pd.DataFrame, by):
    return float(r.mean()) if by is None else r.groupby(u[by]).mean()


def stat_mean_scaled_regret(df: pd.DataFrame, by=None):
    """Mean over units of (J_selected - J_oracle) / scale, with BOTH families fixed.

    df comes from fixed_choice_costs (or a hand-built table with
    pinball_selected and pinball_oracle, see attach_sample_oracle). The family
    the selector chose and the hindsight-oracle family are the ones determined
    on the original sample; only their costs are recomputed on the rows given.
    On the original table this equals the engine's mean scaled regret. On a
    resample a unit's value can be negative (the fixed oracle family need not
    be the cheaper one on that resample), which is what lets the bootstrap
    distribution be centred on the reported regret.

    The resulting interval describes how the cost gap between two FIXED
    families per unit varies with the evaluation sample. It is not an interval
    for regret against the truly better family: the oracle family was picked
    on the same test costs, so the reported regret is biased upward (it is
    positive on average even for two families with equal expected cost), and
    this interval is centred on that biased value.

    With `by` (for example "tau") a Series is returned, one value per group.
    """
    u = _unit_means(df, extra=FIXED_COLUMNS)
    r = (u["pinball_selected"] - u["pinball_oracle"]) / u["scale"]
    return _mean(r, u, by)


def stat_mean_scaled_regret_recomputed_oracle(df: pd.DataFrame, by=None):
    """NOT THE DEFAULT. Mean over units of (J_selected - min(J_classical, J_tsfm)) / scale
    with the minimum recomputed on the rows given.

    Biased upward under resampling: each unit's value is forced to be >= 0 on
    every resample, so units whose two families are close contribute positive
    amounts that they do not contribute to the reported regret. The percentile
    interval built from this statistic can lie entirely above the reported
    regret. It is NOT a confidence interval for the reported regret and must
    not be presented as one. It is kept only to show the size of that effect.
    """
    u = _unit_means(df, extra=("pinball_selected",))
    r = (u["pinball_selected"] - np.minimum(u["J_classical"], u["J_tsfm"])) / u["scale"]
    return _mean(r, u, by)


def stat_mean_delta(df: pd.DataFrame, by=None):
    """Mean over units of Delta = (J_tsfm - J_classical) / scale.

    Orientation of docs/project-overview.md: Delta = J_TSFM - J_classical, and
    Delta < 0 means the TSFM family has the lower cost. Same as e1.decision_map.
    """
    u = _unit_means(df)
    r = (u["J_tsfm"] - u["J_classical"]) / u["scale"]
    return _mean(r, u, by)
