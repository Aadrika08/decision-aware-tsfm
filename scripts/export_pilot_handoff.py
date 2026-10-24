"""
Export the bounded development-pilot handoff for Aadrika's readiness checker.

Produces, in a new output directory (refuses to overwrite):
  sources/<series>.npy  one raw 1-D float64 array per selected series, missing
                        values kept in place (array position = timestamp)
  workload.json         one row per series x origin x horizon x seed
  omega.json            one row per series x origin x variant (current, last_4096)
  summary.md            human-readable dataset / selection / exclusion summary

Follows docs/runbooks/development-pilot.md, production-cache-handoff.md,
configured-forecast-cache.md ("Inputs and identities") and ADR 0006.

Usage (from the stage0 folder):
    python export_pilot_handoff.py --output HANDOFF
    python export_pilot_handoff.py --output HANDOFF --per-config-cap 3
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk

try:
       from omega import compute_omega_checked  # stage0 folder
except ImportError:  # repo layout after main's restructure
       import sys
       sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
       from decision_aware_tsfm.omega.compute import compute_omega_checked


from stage0prep_fixed_v2 import (
    PRED_LENGTH_MAP,
    base_freq_code,
    compute_test_trim,
    compute_windows,
    split_channels,
)

DATA_DIR = Path("./gift_eval_data")

# Data revision of Salesforce/GiftEval actually downloaded (first line of every
# gift_eval_data/.cache/huggingface/download/*.metadata file; all identical).
SOURCE_REVISION = "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"
SOURCE_REPOSITORY = "Salesforce/GiftEval"
NAMING_SCHEME = "dataset-config-v1"

# Pilot scope per development-pilot.md: ETT, Electricity, Solar, Jena at hourly.
PILOT_CONFIGS = ["ett1/H", "ett2/H", "electricity/H", "solar/H", "jena_weather/H"]

# Datasets where every channel/series is included. Others are capped and picked
# evenly across their Omega range ("Dayson chooses ... rows with Omega variation").
TAKE_ALL = {"ett1/H", "ett2/H", "jena_weather/H"}

QUANTILES = [0.01, 0.05, 0.09, 0.25, 0.4, 0.5, 0.6, 0.75, 0.91, 0.95, 0.99]
HORIZONS = [1, 24, 168]
SEEDS = [42, 43]
CONTEXT_LENGTH = 2048          # pilot runbook: hourly context 2,048
OMEGA_FLOOR = 1000             # source paper's stability floor
CAP_4096 = 4096

# Bounds convention (proposed, pending Aadrika's confirmation): forecasts sit in
# GIFT-Eval's held-back region. training_end = GIFT-Eval train/test boundary,
# validation_end = end of series.


def dataset_config(rel_path: str) -> tuple[str, str]:
    """ADR 0006 mapping for nested folders: ett1/H -> ("ett1", "H")."""
    dataset, config = rel_path.split("/", 1)
    return dataset, config


def legacy_series_id(rel_path: str, item_id: str, channel) -> str:
    """Keep the PR #48 series IDs unchanged (Aadrika: opaque strings, no renaming)."""
    subset = rel_path.replace("/", "_")
    if channel is None:
        return f"{subset}__{item_id}"
    return f"{subset}__{item_id}_dim{channel}"


def safe_filename(series_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", series_id) + ".npy"


def load_config(rel_path: str):
    """Returns (list of series dicts, raw_freq string) for one configuration.
    Each series dict: item_id, channel (None if univariate), raw (1-D float64),
    start (ISO string, timezone-naive)."""
    ds = load_from_disk(str(DATA_DIR / rel_path))
    if hasattr(ds, "keys") and "train" in ds.keys():
        ds = ds["train"]
    raw_freq = str(ds[0]["freq"])
    out = []
    for row in ds:
        target = np.asarray(row["target"], dtype=np.float64)
        start = pd.Timestamp(row["start"])
        if start.tzinfo is not None:
            raise ValueError(f"{rel_path}: timezone-aware start not supported by the runner")
        for ch, arr in split_channels(target):
            out.append({
                "item_id": str(row["item_id"]),
                "channel": ch,
                "raw": arr,
                "start": start.isoformat(),
            })
    return out, raw_freq


def candidate_origins(raw: np.ndarray, training_end: int, windows: int, pred_len: int):
    """GIFT-Eval short-term evaluation-window starts, spaced pred_len apart,
    kept only if the 2,048 context and the longest (168) target fit, stay in
    the validation block, and contain no missing values.
    Returns (kept, dropped) where dropped is a list of (origin, reason)."""
    length = len(raw)
    kept, dropped = [], []
    first = length - pred_len * windows
    for k in range(windows):
        origin = first + pred_len * k
        h_max = max(HORIZONS)
        if origin + h_max > length:
            dropped.append((origin, "target_past_end"))
        elif origin < training_end:
            dropped.append((origin, "before_training_end"))
        elif origin - CONTEXT_LENGTH < 0:
            dropped.append((origin, "context_before_start"))
        elif not np.all(np.isfinite(raw[origin - CONTEXT_LENGTH:origin])):
            dropped.append((origin, "missing_in_context"))
        elif not np.all(np.isfinite(raw[origin:origin + h_max])):
            dropped.append((origin, "missing_in_target"))
        else:
            kept.append(origin)
    return kept, dropped


def pick_across_range(items, key, cap):
    """Pick `cap` items evenly spaced across the sorted `key` values."""
    ranked = sorted(items, key=key)
    if len(ranked) <= cap:
        return ranked
    idx = sorted(set(round(i * (len(ranked) - 1) / (cap - 1)) for i in range(cap)))
    return [ranked[i] for i in idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-config-cap", type=int, default=10,
                        help="series per capped dataset (Electricity, Solar)")
    parser.add_argument("--approval-reference", default=None,
                        help="genuine recorded approval; omit until one exists")
    args = parser.parse_args()

    out_dir = Path(args.output)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists -- choose a new output directory")
    (out_dir / "sources").mkdir(parents=True)

    workload_rows, omega_rows, summary_lines = [], [], []
    totals = {"series": 0, "origins": 0, "dropped_origins": {}, "omega_null": {}}

    for rel_path in PILOT_CONFIGS:
        dataset, config = dataset_config(rel_path)
        series, raw_freq = load_config(rel_path)
        freq_code = base_freq_code(raw_freq)
        subset = rel_path.replace("/", "_")

        # Same test-window trim as stage0prep_fixed_v2 (raw min length).
        min_length = min(len(s["raw"]) for s in series)
        test_trim = compute_test_trim(subset, freq_code, min_length)
        pred_len = PRED_LENGTH_MAP[freq_code]        # short-term, e.g. 48 hourly
        windows = compute_windows(min_length, pred_len)

        # Selection Omega: training history only, so selection never sees
        # the forecast region.
        for s in series:
            training_end = max(0, len(s["raw"]) - test_trim)
            s["training_end"] = training_end
            s["select_omega"], _ = compute_omega_checked(
                s["raw"][:training_end], min_finite=OMEGA_FLOOR)

        eligible = [s for s in series if s["select_omega"] is not None]
        if rel_path in TAKE_ALL:
            chosen = eligible
        else:
            chosen = pick_across_range(eligible, lambda s: s["select_omega"],
                                       args.per_config_cap)

        config_origins = 0
        for s in chosen:
            raw = s["raw"]
            series_id = legacy_series_id(rel_path, s["item_id"], s["channel"])
            # Runner needs a nonnegative integer channel; univariate -> 0,
            # recorded in summary.md (Aadrika, #45).
            channel = 0 if s["channel"] is None else int(s["channel"])

            origins, dropped = candidate_origins(raw, s["training_end"], windows, pred_len)
            for _, reason in dropped:
                totals["dropped_origins"][reason] = totals["dropped_origins"].get(reason, 0) + 1
            if not origins:
                summary_lines.append(f"- {series_id}: no valid origins, not exported")
                continue

            npy_name = safe_filename(series_id)
            npy_path = out_dir / "sources" / npy_name
            np.save(npy_path, np.ascontiguousarray(raw, dtype=np.float64), allow_pickle=False)
            source_sha256 = hashlib.sha256(npy_path.read_bytes()).hexdigest()

            identity = {
                "dataset": dataset, "config": config, "series_id": series_id,
                "item_id": s["item_id"], "channel": channel,
                "source_revision": SOURCE_REVISION,
            }
            staged = {
                "source_path": f"sources/{npy_name}", "source_sha256": source_sha256,
                "source_length": len(raw), "start_timestamp": s["start"],
                "frequency": raw_freq,
            }
            bounds = {
                "usable_start": 0, "usable_end": len(raw),
                "training_end": s["training_end"], "validation_end": len(raw),
            }

            for origin in origins:
                for horizon in HORIZONS:
                    for seed in SEEDS:
                        workload_rows.append({
                            **identity, **staged, **bounds,
                            "context_start": origin - CONTEXT_LENGTH,
                            "context_length": CONTEXT_LENGTH,
                            "origin": origin, "horizon": horizon, "seed": seed,
                        })
                # Omega: history ends at the origin; last_4096 caps the same cutoff.
                for variant, h_start in (("current", 0),
                                         ("last_4096", max(0, origin - CAP_4096))):
                    value, reason = compute_omega_checked(raw[h_start:origin],
                                                          min_finite=OMEGA_FLOOR)
                    if value is None:
                        totals["omega_null"][reason] = totals["omega_null"].get(reason, 0) + 1
                    omega_rows.append({
                        "dataset": dataset, "config": config, "series_id": series_id,
                        "source_revision": SOURCE_REVISION, "source_sha256": source_sha256,
                        "origin": origin, "variant": variant,
                        "history_start": h_start, "history_end": origin,
                        "omega": None if value is None else round(value, 6),
                        "omega_status": reason,
                    })
            totals["series"] += 1
            totals["origins"] += len(origins)
            config_origins += len(origins)

        summary_lines.append(
            f"- {dataset}/{config}: {len(series)} series available, {len(eligible)} with "
            f"defined training Omega, {len(chosen)} chosen, {config_origins} origins "
            f"(test_trim {test_trim}, {windows} GIFT-Eval windows of {pred_len})")

    header = {"naming_scheme": NAMING_SCHEME, "source_repository": SOURCE_REPOSITORY}
    workload = {
        "schema_version": 1, "split": "development", "fixture": False,
        **header, "quantiles": QUANTILES, "rows": workload_rows,
    }
    if args.approval_reference:
        workload["approval_reference"] = args.approval_reference
    omega = {
        "schema_version": 1, "fixture": False, **header,
        "variants": {
            "current": "Spectral predictability Omega on raw levels (omega.compute_omega_checked, "
                       "Hann taper, DC excluded, normalised spectral entropy) over history "
                       "[0, origin); null with omega_status when the history is empty, flat, "
                       f"or has fewer than {OMEGA_FLOOR} finite observations.",
            "last_4096": "Same method and floor applied to at most the last 4096 observations "
                         "before the origin: history [max(0, origin-4096), origin).",
        },
        "rows": omega_rows,
    }
    (out_dir / "workload.json").write_text(json.dumps(workload, indent=2), encoding="utf-8")
    (out_dir / "omega.json").write_text(json.dumps(omega, indent=2), encoding="utf-8")

    jobs = len(workload_rows) * 4
    summary = [
        "# Development pilot handoff summary",
        "",
        f"Source: {SOURCE_REPOSITORY} at revision {SOURCE_REVISION}. Naming: {NAMING_SCHEME}.",
        "",
        "## Configurations",
        *[line for line in summary_lines if line],
        "",
        "## Totals",
        f"- Series exported: {totals['series']}",
        f"- Origins: {totals['origins']}",
        f"- Workload rows (x 3 horizons x 2 seeds): {len(workload_rows)}",
        f"- Forecast jobs with the 4 pilot models: {jobs}",
        f"- Omega rows: {len(omega_rows)}; null by reason: {totals['omega_null'] or 'none'}",
        f"- Candidate origins dropped by reason: {totals['dropped_origins'] or 'none'}",
        "",
        "## Conventions",
        "- Origins: GIFT-Eval short-term evaluation windows, kept only where the 2,048 "
        "context and 168-step target fit with no missing values.",
        "- Bounds: training_end = GIFT-Eval train/test boundary; validation_end = end of "
        "series (proposed, pending confirmation).",
        "- Series IDs unchanged from PR #48. Univariate series (native channel null) use "
        "channel 0 in the workload.",
        "- Selection: ETT1, ETT2, Jena take every channel; Electricity and Solar take "
        f"{args.per_config_cap} series evenly across their training-history Omega range.",
        "- Selector role assignments are not included yet (pending Veer/Abdullah/Pratham).",
        f"- approval_reference: {args.approval_reference or 'NOT SET (no recorded approval yet)'}",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n".join(summary))


if __name__ == "__main__":
    main()