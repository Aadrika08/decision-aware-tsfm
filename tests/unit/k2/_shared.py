"""Shared helpers for the K2 tests.

Two kinds of input are used and they are kept apart:
  * the saved reference inputs (published results and the Omega manifest), for
    the regression tests against the recorded K2 tables;
  * tiny hand-built tables with made-up numbers, for mechanism tests. They are
    synthetic and say nothing about real model families.
"""
import atexit
import contextlib
import io
import json
import os
import pathlib
import shutil
import tempfile
from functools import lru_cache

import pandas as pd

from decision_aware_tsfm.analysis.k2 import cli
from decision_aware_tsfm.analysis.k2.families import METRICS, MSE_MEDIAN_FORECAST_COLUMN
from decision_aware_tsfm.analysis.k2.loading import aggregate_omega, load_results, read_manifest
from decision_aware_tsfm.analysis.k2.pipeline import DEFAULT_MANIFEST, DEFAULT_RESULTS, run_k2

ROOT = pathlib.Path(__file__).resolve().parents[3]
RECORDED = ROOT / "docs" / "results" / "k2"
REFERENCE_RESAMPLES = 2000          # enough for the mechanism checks; the note uses the default


def scratch_dir() -> str:
    d = tempfile.mkdtemp(prefix="k2-test-")
    atexit.register(shutil.rmtree, d, ignore_errors=True)
    return d


@lru_cache(maxsize=None)
def reference_results():
    return load_results(DEFAULT_RESULTS)


@lru_cache(maxsize=None)
def reference_manifest():
    return read_manifest(DEFAULT_MANIFEST)


@lru_cache(maxsize=None)
def reference_omega():
    return aggregate_omega(reference_manifest(), "omega_levels")


@lru_cache(maxsize=None)
def reference_run():
    """One full run on the saved reference inputs: (output directory, run manifest)."""
    out = os.path.join(scratch_dir(), "run")
    manifest = run_k2(outdir=out, n_resamples=REFERENCE_RESAMPLES, echo=lambda *_: None)
    return out, manifest


def run_cli(argv):
    """Run the command in process. Returns (exit code or message, captured stdout)."""
    buf = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buf):
        try:
            cli.main(argv)
        except SystemExit as exc:
            code = exc.code
    return code, buf.getvalue()


# ---- synthetic inputs (made-up numbers) -----------------------------------
TINY_SUBSETS = ["ett1_H", "ett2_H", "solar_H", "electricity_H", "us_births_D"]
TINY_CONFIGS = ["ett1/H/short", "ett2/H/short", "solar/H/short", "electricity/H/short",
                "us_births/D/short"]
SECOND_OMEGA = "omega_second_column_placeholder"   # not the teammate's column name
# Every metric column the command reads: the four recorded ones and MSE[0.5].
METRIC_COLUMNS = list(METRICS.values()) + [MSE_MEDIAN_FORECAST_COLUMN]


def metric_row(dataset, model, score, **overrides):
    """One results row with the same made-up score in every metric column."""
    row = {"dataset": dataset, "model": model, **{c: float(score) for c in METRIC_COLUMNS}}
    row.update(overrides)
    return row


def tiny_manifest(extra_rows=(), second_column=True) -> dict:
    """Five single-series subsets. omega_levels rises with the subset order; the
    second Omega column falls with it."""
    rows = []
    for i, subset in enumerate(TINY_SUBSETS):
        row = {"series_id": f"{subset}_idx_0", "subset": subset, "channel": 0,
               "length": 2000 + i, "omega_levels": 0.1 * (i + 1), "omega_diffs": 0.05}
        if second_column:
            row[SECOND_OMEGA] = 0.9 - 0.1 * i
        rows.append(row)
    rows.extend(extra_rows)
    return {"active_dev_series": rows}


def tiny_results(extra_rows=()) -> pd.DataFrame:
    """Zero-shot scores 1.0 everywhere; the deep family's score falls with the
    dataset order, so Delta(zero-shot, deep) = 10, 20, 30, 40, 50."""
    rows = []
    for i, config in enumerate(TINY_CONFIGS):
        scores = {"TiRex": 1.0, "Moirai_small": 1.0,                  # zero-shot
                  "PatchTST": 1.0 - 0.1 * (i + 1), "DLinear": 1.0 - 0.1 * (i + 1),   # deep
                  "Naive": 2.0 + i,                                     # statistical
                  "Chronos_small": [1.5, 1.9, 1.6, 1.8, 1.7][i],          # pretrained
                  "not_in_any_family": 1000.0}
        for model, score in scores.items():
            rows.append(metric_row(config, model, score))
    rows.extend(extra_rows)
    return pd.DataFrame(rows)


def write_tiny_inputs(directory, manifest=None, results=None):
    manifest = tiny_manifest() if manifest is None else manifest
    results = tiny_results() if results is None else results
    mpath = os.path.join(directory, "tiny_manifest.json")
    rpath = os.path.join(directory, "tiny_results.csv")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    results.to_csv(rpath, index=False)
    return mpath, rpath


# ---- a second hand-built results table, for the pooling and summary tests ---
# Five base datasets with two configurations each (short, long). Three zero-shot
# models with the (short, long) scores below; every other family has one model
# with a constant score, so Delta rises with the zero-shot family score and the
# rank correlation with Omega is the rank correlation of that score.
POOLED_ZERO_SHOT = {
    #                 TiRex      Moirai_small  VisionTS
    "ett1/H":        ((1, 19),   (3, 5),       (4, 28)),
    "ett2/H":        ((2, 4),    (9, 13),      (14, 24)),
    "solar/H":       ((6, 6),    (7, 5),       (3, 21)),
    "electricity/H": ((20, 4),   (13, 13),     (5, 17)),
    "us_births/D":   ((5, 16),   (17, 5),      (6, 8)),
}
POOLED_OTHER = {"PatchTST": 10.0, "Naive": 20.0, "Chronos_small": 10.0}
# Zero-shot family score per base dataset (Omega 0.1 ... 0.5), worked by hand:
#   summary          ett1  ett2  solar  elec  us_births   ranks        rho
#   mean, rows       10    11    8      12    9.5         3,4,1,5,2   -0.1
#   median, rows     4.5   11    6      13    7           1,4,2,5,3   +0.5
#   min, rows        1     2     3      4     5           1,2,3,4,5   +1.0
#   median, models   10    11    6      12    10.5        2,4,1,5,3   +0.3
#   min, models      4     3     6      11    7           2,1,3,5,4   +0.8
# rho = 1 - 6 * sum(d^2) / (5 * 24), with sum(d^2) = 22, 10, 0, 14, 4.
POOLED_SCORE = {
    ("mean", "rows"): [10, 11, 8, 12, 9.5], ("median", "rows"): [4.5, 11, 6, 13, 7],
    ("min", "rows"): [1, 2, 3, 4, 5], ("median", "models"): [10, 11, 6, 12, 10.5],
    ("min", "models"): [4, 3, 6, 11, 7]}
POOLED_RHO = {("mean", "rows"): -0.1, ("median", "rows"): 0.5, ("min", "rows"): 1.0,
              ("median", "models"): 0.3, ("min", "models"): 0.8}
# Versus the deep family (constant 10): datasets where zero-shot is lower, higher, tied.
POOLED_SIGNS_VS_DEEP = {("mean", "rows"): (2, 2, 1), ("median", "rows"): (3, 2, 0),
                        ("min", "rows"): (5, 0, 0), ("median", "models"): (1, 3, 1),
                        ("min", "models"): (4, 1, 0)}


def pooled_results() -> pd.DataFrame:
    rows = []
    for base_freq, scores in POOLED_ZERO_SHOT.items():
        for term_index, term in enumerate(("short", "long")):
            config = f"{base_freq}/{term}"
            for model, pair in zip(("TiRex", "Moirai_small", "VisionTS"), scores):
                rows.append(metric_row(config, model, pair[term_index]))
            for model, score in POOLED_OTHER.items():
                rows.append(metric_row(config, model, score))
    return pd.DataFrame(rows)


# ---- a third hand-built table, for the one-model-dominates tests ------------
# One configuration per base dataset. Zero-shot: TiRex 2, Moirai_small 4,
# VisionTS 3 in every column, except that VisionTS has a large MSE[mean].
DOMINANT_MSE_MEAN = [6.0, 24.0, 9.0, 114.0, 18.0]     # VisionTS, MSE[mean] column only
DOMINANT_DEEP = [2.0, 2.5, 3.5, 4.0, 5.5]             # PatchTST, every column
DOMINANT_STATISTICAL = [20.0, 21.0, 22.0, 23.0, 24.0]  # Naive, every column
DOMINANT_PRETRAINED = [9.0, 8.0, 7.0, 6.0, 5.0]       # Chronos_small, every column
# Worked by hand, zero-shot versus deep:
#   every column but MSE[mean]: zero-shot mean = (2 + 4 + 3) / 3 = 3 on every dataset, so
#     Delta = 100 * (3 - deep) / 3 falls as deep rises with Omega: rho = -1; zero-shot is
#     lower where deep > 3, on 3 of 5 datasets. Removing VisionTS leaves (2 + 4) / 2 = 3.
#   MSE[mean]: zero-shot mean = (6 + v) / 3 = 4, 10, 5, 40, 8; deep / mean = 0.5, 0.25, 0.7,
#     0.1, 0.6875; Delta = 50, 75, 30, 90, 31.25; ranks 3,4,1,5,2; rho = -0.1; zero-shot is
#     lower on 0 of 5. VisionTS carries 0.5, 0.8, 0.6, 0.95, 0.75 of the family sum (median 0.75).
#   MSE[mean] without TiRex: mean = (4 + v) / 2 = 5, 14, 6.5, 59, 11; Delta ranks 3,4,1,5,2.
#   MSE[mean] without Moirai_small: mean = (2 + v) / 2 = 4, 13, 5.5, 58, 10; ranks 3,4,1,5,2.


def dominant_results() -> pd.DataFrame:
    mse_mean = METRICS["MSE"]
    rows = []
    for i, config in enumerate(TINY_CONFIGS):
        rows.append(metric_row(config, "TiRex", 2.0))
        rows.append(metric_row(config, "Moirai_small", 4.0))
        rows.append(metric_row(config, "VisionTS", 3.0, **{mse_mean: DOMINANT_MSE_MEAN[i]}))
        rows.append(metric_row(config, "PatchTST", DOMINANT_DEEP[i]))
        rows.append(metric_row(config, "Naive", DOMINANT_STATISTICAL[i]))
        rows.append(metric_row(config, "Chronos_small", DOMINANT_PRETRAINED[i]))
    return pd.DataFrame(rows)


def tiny_omega(extra_rows=(), column="omega_levels"):
    """Omega tables from the tiny manifest (omega_levels = 0.1 ... 0.5 in subset order)."""
    from decision_aware_tsfm.analysis.k2.loading import _from_dev_v1
    return aggregate_omega(_from_dev_v1(tiny_manifest(extra_rows)), column)
