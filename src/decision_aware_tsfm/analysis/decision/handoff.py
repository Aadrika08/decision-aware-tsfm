"""Read a producer handoff bundle (#45 / #47) for the decision engine.

A bundle directory contains
    handoff.json          partition paths and SHA-256 hashes, sidecar hashes, naming header
    <partition>.parquet   producer-v1 payload, one job per file, job provenance in the
                          Arrow schema metadata key "job"
    omega.json            Omega rows keyed by (dataset, config, series_id, origin, variant),
                          each with source_revision, source_sha256, history_start, history_end

Rules applied here, all from the producer's handoff runbook:
    - read only the partitions listed in handoff.json, and check each file's hash
    - read each partition with pyarrow.parquet.ParquetFile(path).read()
    - exact v1 column names, order and Arrow types
    - every job carries provenance; its identity must match its rows
    - naming scheme and source repository agree across manifest, jobs and Omega sidecar
    - an Omega row's history may not end after its origin
    - every forecast (series, origin) has an Omega row for the chosen variant with the
      same source revision and source hash
Nothing here imports forecasting code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .cache_io import CacheError
from .metrics import TAU_GRID

PRODUCER_ARROW_SCHEMA = pa.schema(
    [(n, pa.string()) for n in ("job_id", "dataset", "config", "series_id", "model", "family")]
    + [(n, pa.int64()) for n in ("origin", "horizon", "seed", "step")]
    + [("y_true", pa.float64()), ("timestamp", pa.string())]
    + [(f"q_{t:.2f}", pa.float64()) for t in TAU_GRID])

JOB_IDENTITY = ("dataset", "config", "series_id", "model", "origin", "horizon", "seed")
JOB_PROVENANCE = ("naming_scheme", "source_repository", "source_revision", "source_sha256")
OMEGA_FIELDS = ("dataset", "config", "series_id", "origin", "variant", "source_revision",
                "source_sha256", "history_start", "history_end", "omega")
SERIES_ORIGIN = ["dataset", "config", "series_id", "origin"]


class HandoffError(CacheError):
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_handoff(root, variant: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (producer-v1 forecast rows, Omega rows for all variants, report)."""
    root = Path(root)
    manifest = json.loads((root / "handoff.json").read_text(encoding="utf-8"))
    hashes = manifest.get("partition_hashes")
    if not hashes:
        raise HandoffError("handoff.json lists no partitions")
    naming = manifest.get("naming_scheme")
    repo = manifest.get("source_repository")

    frames, jobs = [], []
    for rel, expected in sorted(hashes.items()):
        path = root / rel
        if not path.is_file():
            raise HandoffError(f"listed partition is missing: {rel}")
        if file_sha256(path) != expected:
            raise HandoffError(f"hash mismatch for {rel}")
        table = pq.ParquetFile(path).read()
        if not table.schema.remove_metadata().equals(PRODUCER_ARROW_SCHEMA):
            raise HandoffError(f"{rel}: columns, order or Arrow types differ from producer v1")
        meta = table.schema.metadata or {}
        if b"job" not in meta:
            raise HandoffError(f"{rel}: no job provenance in schema metadata")
        job = json.loads(meta[b"job"])
        missing = [k for k in JOB_PROVENANCE + JOB_IDENTITY + ("job_id",) if k not in job]
        if missing:
            raise HandoffError(f"{rel}: job provenance lacks {missing}")
        df = table.to_pandas()
        for k in JOB_IDENTITY + ("job_id",):
            if not (df[k] == job[k]).all():
                raise HandoffError(f"{rel}: rows disagree with job metadata on {k!r}")
        if job["naming_scheme"] != naming or job["source_repository"] != repo:
            raise HandoffError(f"{rel}: naming scheme or source repository differs from handoff.json")
        frames.append(df)
        jobs.append({k: job[k] for k in SERIES_ORIGIN + ["source_revision", "source_sha256"]})
    forecasts = pd.concat(frames, ignore_index=True)

    omega_path = root / "omega.json"
    if manifest.get("omega_sha256") and file_sha256(omega_path) != manifest["omega_sha256"]:
        raise HandoffError("hash mismatch for omega.json")
    sidecar = json.loads(omega_path.read_text(encoding="utf-8"))
    if sidecar.get("naming_scheme") != naming or sidecar.get("source_repository") != repo:
        raise HandoffError("omega.json naming scheme or source repository differs from handoff.json")
    omega = pd.DataFrame(sidecar["rows"])
    lacking = [k for k in OMEGA_FIELDS if k not in omega.columns]
    if lacking:
        raise HandoffError(f"omega.json rows lack {lacking}")
    if (omega["history_end"] > omega["origin"]).any():
        raise HandoffError("omega.json has history ending after its origin")
    if variant not in set(omega["variant"]):
        raise HandoffError(f"omega variant {variant!r} not in {sorted(set(omega['variant']))}")

    prov = pd.DataFrame(jobs).drop_duplicates()
    if prov.duplicated(SERIES_ORIGIN).any():
        raise HandoffError("one (series, origin) has jobs with different source revision/hash")
    om_v = omega[omega["variant"] == variant]
    joined = prov.merge(om_v, on=SERIES_ORIGIN, how="left", suffixes=("", "_omega"))
    if joined["omega"].isna().any():
        raise HandoffError(f"{int(joined['omega'].isna().sum())} forecast (series, origin) pairs "
                           f"have no Omega row for variant {variant!r}")
    mismatch = ((joined["source_revision"] != joined["source_revision_omega"])
                | (joined["source_sha256"] != joined["source_sha256_omega"]))
    if mismatch.any():
        raise HandoffError(f"{int(mismatch.sum())} Omega rows come from a different source "
                           "revision or hash than the forecasts they would join to")

    report = dict(
        fixture=bool(manifest.get("fixture", False)),
        evidence_kind=manifest.get("evidence_kind"),
        naming_scheme=naming, source_repository=repo,
        partitions_listed=len(hashes), partitions_hash_verified=len(frames),
        unlisted_parquet_files=sum(1 for p in root.rglob("*.parquet")
                                   if p.relative_to(root).as_posix() not in hashes),
        rows=int(len(forecasts)), jobs=int(forecasts["job_id"].nunique()),
        series=int(forecasts[["dataset", "config", "series_id"]].drop_duplicates().shape[0]),
        series_origins=int(len(prov)), models=sorted(forecasts["model"].unique().tolist()),
        seeds=sorted(int(s) for s in forecasts["seed"].unique()),
        omega_variant=variant, omega_rows_for_variant=int(len(om_v)),
        omega_sha256=file_sha256(omega_path), handoff_sha256=file_sha256(root / "handoff.json"))
    return forecasts, omega[list(OMEGA_FIELDS)], report
