"""Test helpers: write each series' history as a source file plus a workload.json that
names it, the way Dayson's handoff does (source_path relative to the workload's folder,
source_sha256 of the file). The scale (#29 item 7a) is computed from these files.
SYNTHETIC: for wiring tests only.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["dataset", "config", "series_id"]


def stepped_history(n: int) -> np.ndarray:
    """10 + (t mod 7): not periodic with period 24, so its seasonal-naive MAE is above 0."""
    return 10.0 + (np.arange(n) % 7)


def source_values(frame: pd.DataFrame, y_col: str = "y", history=None) -> dict:
    """{(dataset, config, series_id): array}. Target positions (origin + step - 1) hold the
    frame's realised values; positions before the series' first origin hold its history:
    history[key][:first] when history is a dict holding the key, stepped_history otherwise.
    Positions between targets that no window covers are NaN."""
    out = {}
    for key, g in frame.groupby(KEYS, sort=True):
        pos = (g["origin"] + g["step"] - 1).to_numpy(dtype=np.int64)
        first = int(g["origin"].min())
        arr = np.full(int(pos.max()) + 1, np.nan)
        arr[:first] = (history[key][:first] if history is not None and key in history
                       else stepped_history(first))
        arr[pos] = g[y_col].to_numpy(dtype=float)
        out[key] = arr
    return out


def file_name(key) -> str:
    return "__".join(str(k).replace("/", "_") for k in key) + ".npy"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_sources(folder: Path, values: dict) -> dict:
    """Write each array to folder/sources/<name>.npy. Returns {key: (relative path, sha256)}."""
    folder = Path(folder)
    (folder / "sources").mkdir(parents=True, exist_ok=True)
    out = {}
    for key, arr in values.items():
        rel = f"sources/{file_name(key)}"
        np.save(folder / rel, np.asarray(arr, dtype=np.float64), allow_pickle=False)
        out[key] = (rel, sha256(folder / rel))
    return out


def workload_doc(frame: pd.DataFrame, written: dict, values: dict) -> dict:
    """A development workload: one row per (series, origin, horizon, seed) of the frame,
    each naming its series' source file and hash."""
    cols = KEYS + ["origin", "horizon", "seed"]
    have = [c for c in cols if c in frame.columns]
    rows = frame[have].drop_duplicates().sort_values(have)
    out = []
    for r in rows.to_dict("records"):
        key = tuple(r[k] for k in KEYS)
        rel, sha = written[key]
        row = {k: (int(v) if k in ("origin", "horizon", "seed") else v) for k, v in r.items()}
        row.setdefault("seed", 0)
        row.update(source_path=rel, source_sha256=sha, source_length=int(len(values[key])),
                   usable_start=0)
        out.append(row)
    return dict(schema_version=1, split="development", fixture=True, rows=out)


def write_workload(folder: Path, frame: pd.DataFrame, y_col: str = "y", history=None,
                   name: str = "workload.json") -> tuple[Path, dict]:
    """Sources and workload.json for every series of frame. Returns (workload path,
    {key: sha256})."""
    values = source_values(frame, y_col, history)
    written = write_sources(folder, values)
    path = Path(folder) / name
    path.write_text(json.dumps(workload_doc(frame, written, values)), encoding="utf-8")
    return path, {k: sha for k, (_, sha) in written.items()}
