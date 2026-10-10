"""Coverage simulation for the block bootstrap intervals in bootstrap.py.

    python -m decision_aware_tsfm.analysis.experiments.coverage_sim --out artifacts/runs/bootstrap_coverage

SYNTHETIC. Every design below is simulated from a process whose answer is
known in closed form, so the question "how often does the nominal 95% interval
contain the truth" has an exact reference. Nothing here is a result about real
forecasts.

For each design: draw a data set, compute the bootstrap interval with
bootstrap.bootstrap_ci, record whether it contains the known true value and
how wide it is; repeat `--reps` times. Reported per design:

  coverage     share of replications whose interval contains the truth
  coverage_se  binomial standard error of that share
  width_ratio  mean interval width / true width, where the true width is
               2 * 1.959964 * (true standard deviation of the estimator). A
               ratio of 1 is the width a correct normal-theory 95% interval has.

Designs
  overlap   one series, horizon 24, an origin every step. cost[o] is the sum of
            24 independent N(0,1) errors, so neighbouring windows share 23 of 24
            terms. Mean cost; truth 0.
  ar1       one series, horizon 1, AR(1) costs with coefficient phi. Truth 0.
  ar1_h24   one series, horizon 24, origins 24 apart (non-overlapping windows),
            20 origins: the shape of the fixture's test set. AR(1) costs.
  shock     20 series x 60 origins in one dataset. cost = common shock of the
            origin (shared by all series) + independent noise. Mean cost; truth
            0. Run with independent block positions per series and with block
            positions shared within the dataset, with and without series
            clustering. shock_none is the same without the common shock.
  regret    two families per series. "clear": TSFM truly cheaper by about 0.3 on
            every series, selector always classical, so the true mean regret is
            that gap (computed exactly in make_regret).
            "equal": both families have the same expected cost, so regret
            against the truly better family is 0 for any selector; the
            selector is always classical, or the sample-best family
            ("samplebest", reported regret exactly 0). Both regret statistics
            are run: the default (oracle family fixed) and the recomputed-oracle
            one. contains_estimate is the share of intervals that contain the
            value they are reported with. The "clear" design is also run with
            series resampled as clusters, with and without the block stage, to
            show what stacking the two stages does to the width.

New designs are appended at the end of DESIGNS: the data seed of a design
depends on its position, so appending leaves earlier rows unchanged.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import bootstrap

Z95 = 1.959964


def _frame(cost, horizon, step=1, series="s", dataset="d"):
    n = len(cost)
    return pd.DataFrame(dict(dataset=dataset, config="c", series_id=series, horizon=horizon, tau=0.5,
                             origin=np.arange(n) * step, cost=cost))


def _mean_cost(d):
    return float(d["cost"].to_numpy().mean())


def _sd_of_mean(autocov, n):
    """Standard deviation of the mean of n consecutive values of a stationary
    process with autocovariances autocov[0..n-1]."""
    k = np.arange(1, n)
    return float(np.sqrt((n * autocov[0] + 2.0 * np.sum((n - k) * autocov[1:n])) / n ** 2))


def _ar1(phi, n, rng):
    x = np.empty(n)
    x[0] = rng.standard_normal() / np.sqrt(1.0 - phi ** 2)
    e = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


# each maker returns (frame, statistic, truth, true_sd or None)

def make_overlap(rng, n, horizon=24):
    e = rng.standard_normal(n + horizon - 1)
    cost = np.convolve(e, np.ones(horizon), mode="valid")
    acov = np.array([max(horizon - k, 0) for k in range(n)], dtype=float)
    return _frame(cost, horizon), _mean_cost, 0.0, _sd_of_mean(acov, n)


def make_ar1(rng, n, phi, horizon=1, step=1):
    acov = phi ** np.arange(n) / (1.0 - phi ** 2)
    return _frame(_ar1(phi, n, rng), horizon, step), _mean_cost, 0.0, _sd_of_mean(acov, n)


def make_shock(rng, n_series=20, n=60, shock=1.0):
    common = shock * rng.standard_normal(n)
    f = pd.concat([_frame(common + rng.standard_normal(n), 24, 24, series=f"s{i:02d}")
                   for i in range(n_series)], ignore_index=True)
    return f, _mean_cost, 0.0, float(np.sqrt(shock ** 2 / n + 1.0 / (n * n_series)))


def _folded_mean(mu, sigma):
    """E|mu + sigma * N(0,1)|."""
    return float(sigma * np.sqrt(2.0 / np.pi) * np.exp(-mu ** 2 / (2.0 * sigma ** 2))
                 + mu * (1.0 - 2.0 * stats.norm.cdf(-mu / sigma)))


def make_regret(rng, n_series, n, gap, selector, statistic):
    """Costs: classical ~ |1 + 0.3 N|, tsfm ~ |1 - gap + 0.3 N|, independent
    over origins and series. scale is 1. The true mean regret of
    always-classical against the truly cheaper family is
    E|1 + 0.3 N| - E|1 - gap + 0.3 N| (0 when gap is 0)."""
    rows = []
    for s in range(n_series):
        f = _frame(np.zeros(n), 1, series=f"s{s:02d}").drop(columns="cost")
        f["pinball_classical"] = np.abs(1.0 + 0.3 * rng.standard_normal(n))
        f["pinball_tsfm"] = np.abs(1.0 - gap + 0.3 * rng.standard_normal(n))
        f["abs_y_sum"], f["n"] = 1.0, 1
        rows.append(f)
    c = bootstrap.attach_sample_oracle(pd.concat(rows, ignore_index=True))
    c["pinball_selected"] = c["pinball_oracle"] if selector == "samplebest" else c["pinball_classical"]
    stat = (bootstrap.stat_mean_scaled_regret if statistic == "fixed"
            else bootstrap.stat_mean_scaled_regret_recomputed_oracle)
    return c, stat, _folded_mean(1.0, 0.3) - _folded_mean(1.0 - gap, 0.3), None


MAKERS = dict(overlap=make_overlap, ar1=make_ar1, shock=make_shock, regret=make_regret)

# (label, maker, maker kwargs, bootstrap_ci kwargs)
DESIGNS = [
    ("overlap H=24 stride 1 n=480, auto block", "overlap", dict(n=480), dict()),
    ("overlap H=24 stride 1 n=480, single origins (naive)", "overlap", dict(n=480), dict(block_length=1)),
    ("overlap H=24 stride 1 n=480, block 48", "overlap", dict(n=480), dict(block_length=48)),
    ("overlap H=24 stride 1 n=120, auto block", "overlap", dict(n=120), dict()),
    ("ar1 phi=0.0 n=320 H=1, auto block", "ar1", dict(n=320, phi=0.0), dict()),
    ("ar1 phi=0.5 n=320 H=1, auto block", "ar1", dict(n=320, phi=0.5), dict()),
    ("ar1 phi=0.9 n=320 H=1, auto block", "ar1", dict(n=320, phi=0.9), dict()),
    ("ar1_h24 phi=0.0 n=20 H=24 step 24, auto block", "ar1", dict(n=20, phi=0.0, horizon=24, step=24), dict()),
    ("ar1_h24 phi=0.5 n=20 H=24 step 24, auto block", "ar1", dict(n=20, phi=0.5, horizon=24, step=24), dict()),
    ("shock 20x60, per-series blocks, cluster none", "shock", dict(), dict()),
    ("shock 20x60, per-series blocks, cluster series", "shock", dict(), dict(cluster="series")),
    ("shock 20x60, shared blocks, cluster none", "shock", dict(),
     dict(time_blocks="shared_within_dataset")),
    ("shock 20x60, shared blocks, cluster series", "shock", dict(),
     dict(time_blocks="shared_within_dataset", cluster="series")),
    ("shock_none 20x60, per-series blocks, cluster none", "shock", dict(shock=0.0), dict()),
    ("shock_none 20x60, shared blocks, cluster none", "shock", dict(shock=0.0),
     dict(time_blocks="shared_within_dataset")),
    ("regret clear gap 0.3, always classical, fixed oracle", "regret",
     dict(n_series=30, n=40, gap=0.3, selector="classical", statistic="fixed"), dict()),
    ("regret clear gap 0.3, always classical, fixed oracle, cluster series", "regret",
     dict(n_series=30, n=40, gap=0.3, selector="classical", statistic="fixed"), dict(cluster="series")),
    ("regret equal families, always classical, fixed oracle", "regret",
     dict(n_series=50, n=40, gap=0.0, selector="classical", statistic="fixed"), dict()),
    ("regret equal families, always classical, recomputed oracle", "regret",
     dict(n_series=50, n=40, gap=0.0, selector="classical", statistic="recomputed"), dict()),
    ("regret equal families, sample-best selector, fixed oracle", "regret",
     dict(n_series=50, n=40, gap=0.0, selector="samplebest", statistic="fixed"), dict()),
    ("regret equal families, sample-best selector, recomputed oracle", "regret",
     dict(n_series=50, n=40, gap=0.0, selector="samplebest", statistic="recomputed"), dict()),
    ("regret clear gap 0.3, always classical, fixed oracle, cluster series, origins not resampled",
     "regret", dict(n_series=30, n=40, gap=0.3, selector="classical", statistic="fixed"),
     dict(cluster="series", block_length=None)),
]


def run_design(index: int, reps: int, n_boot: int, seed: int, level: float = 0.95) -> dict:
    """Run one design. Data seed is seed + 100003 * index + r; bootstrap seed is r."""
    label, maker, mk, bk = DESIGNS[index]
    cover = contains_est = 0
    widths, estimates, used = [], [], ""
    true_sd = None
    for r in range(reps):
        frame, stat, truth, true_sd = MAKERS[maker](np.random.default_rng(seed + 100003 * index + r), **mk)
        ci = bootstrap.bootstrap_ci(frame, stat, n_boot=n_boot, level=level, seed=r, **bk)
        lo, hi, est = float(ci.lower[0]), float(ci.upper[0]), float(ci.estimate[0])
        cover += lo <= truth <= hi
        contains_est += lo <= est <= hi
        widths.append(hi - lo)
        estimates.append(est)
        used = ci.block_lengths_used
    p = cover / reps
    true_width = None if true_sd is None else 2.0 * Z95 * true_sd
    return dict(design=label, reps=reps, n_boot=n_boot, nominal_level=level, block_lengths_used=used,
                cluster=bk.get("cluster", "none"), time_blocks=bk.get("time_blocks", "per_series"),
                truth=truth, mean_estimate=float(np.mean(estimates)),
                mean_width=float(np.mean(widths)), true_width=true_width,
                width_ratio=None if true_width is None else float(np.mean(widths) / true_width),
                coverage=p, coverage_se=float(np.sqrt(p * (1.0 - p) / reps)),
                contains_estimate=contains_est / reps)


def _job(args):
    return run_design(*args)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Coverage of the block bootstrap intervals (synthetic)")
    ap.add_argument("--out", default="artifacts/runs/bootstrap_coverage")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", default="", help="run only designs whose label contains this text")
    a = ap.parse_args(argv)
    todo = [(i, a.reps, a.n_boot, a.seed) for i, d in enumerate(DESIGNS) if a.only in d[0]]
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            rows = list(ex.map(_job, todo))
    else:
        rows = [_job(t) for t in todo]
    t = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "coverage.csv", index=False, lineterminator="\n")
    (out / "manifest.json").write_text(json.dumps(dict(
        experiment="bootstrap coverage simulation", data="synthetic, closed-form truth",
        reps=a.reps, n_boot=a.n_boot, seed=a.seed, nominal_level=0.95, files=["coverage.csv"]),
        indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print("SYNTHETIC. Coverage of the NOMINAL 95% percentile block-bootstrap interval, "
          f"{a.reps} replications x {a.n_boot} resamples, seed {a.seed}.")
    print(t[["design", "block_lengths_used", "truth", "mean_estimate", "mean_width", "true_width",
             "width_ratio", "coverage", "coverage_se", "contains_estimate"]].round(4).to_string(index=False))
    print(f"\nwrote coverage.csv and manifest.json to {out}")


if __name__ == "__main__":
    main()
