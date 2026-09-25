"""Export Veer's pinned analytic fixture as a synthetic v1 producer handoff."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .planner import target_timestamps
from .schema import QCOLS, SCHEMA, TAUS, validate_payload


SOURCE_COMMIT = "dd9505d2389a191327e4fb5da937a8c0b289a6a1"
SOURCE_REPOSITORY = "synthetic:Veer-PR-44/analysis.decision.fixture"
GENERATOR_SEED = 7
EXPORTED_SEEDS = (42, 43)  # Coverage copies of one draw, not independent forecasts.
START_TIMESTAMP = "2020-01-01T00:00:00"


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_package(checkout: Path):
    checkout = Path(checkout).resolve()
    try:
        commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"], text=True)
    except subprocess.CalledProcessError as error:
        raise ValueError("Decision source must be a Git checkout") from error
    if commit != SOURCE_COMMIT or dirty:
        raise ValueError("Decision source must be the clean pinned PR #44 commit")
    package = checkout / "src/decision_aware_tsfm/analysis/decision"
    fixture_file = package / "fixture.py"
    if not fixture_file.is_file():
        raise ValueError("Decision fixture module is missing")
    # Load only the analysis package under a private name; no forecasters are imported.
    name = "_handoff_decision_fixture"
    spec = importlib.util.spec_from_file_location(name, package / "__init__.py",
                                                   submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        fixture = importlib.import_module(f"{name}.fixture")
        return fixture.make_fixture(seed=GENERATOR_SEED), _file_sha(fixture_file)
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]


def _bounded(frame: pd.DataFrame) -> pd.DataFrame:
    needed = {"dataset", "config", "series_id", "model", "horizon", "origin", "step", "y", *QCOLS}
    if needed - set(frame.columns):
        raise ValueError(f"Fixture is missing columns: {sorted(needed - set(frame.columns))}")
    for name in ("horizon", "origin", "step"):
        if not pd.api.types.is_integer_dtype(frame[name]) or frame[name].isna().any():
            raise ValueError(f"Fixture {name} must contain non-null integers")
    frame = frame.loc[frame.horizon == 24].copy()
    if frame.empty:
        raise ValueError("Fixture has no horizon-24 rows")
    origins = frame.groupby(["dataset", "config", "series_id"])["origin"].transform(
        lambda values: values.isin(sorted(values.unique())[:2]))
    frame = frame.loc[origins].sort_values(
        ["dataset", "config", "series_id", "origin", "model", "step"], kind="stable")
    counts = frame.groupby(["dataset", "config", "series_id"])["origin"].nunique()
    if frame.empty or frame["dataset"].nunique() < 2 or (counts != 2).any():
        raise ValueError("Fixture needs two origins per series across at least two datasets")
    return frame


def _jobs(frame: pd.DataFrame, omega: pd.DataFrame, generator_sha: str):
    if tuple(f"q_{tau:.2f}" for tau in TAUS) != QCOLS:
        raise ValueError("Producer quantile grid changed")
    omega_lookup = omega.set_index(["dataset", "series_id"])["omega_levels"]
    exporter_sha = _file_sha(Path(__file__))
    partitions, omega_rows = [], {}
    for (dataset, config, series_id, old_origin, model), group in frame.groupby(
            ["dataset", "config", "series_id", "origin", "model"], sort=True):
        if not all(isinstance(x, str) and x for x in (dataset, config, series_id, model)):
            raise ValueError("Fixture identities must be nonempty strings")
        group = group.sort_values("step")
        horizon = int(group.horizon.iloc[0])
        origin = int(old_origin) + 1  # Veer: last observation; producer: first target.
        if group.step.tolist() != list(range(1, horizon + 1)) or len(group) != horizon:
            raise ValueError("Fixture job has missing or duplicate steps")
        if model not in {"AutoETS", "SeasonalNaive", "Moirai-small"}:
            raise ValueError(f"Unexpected fixture model {model!r}")
        family = "tsfm" if model == "Moirai-small" else "classical"
        source_sha = _sha(_json({"generator_commit": SOURCE_COMMIT,
                                 "generator_file_sha256": generator_sha,
                                 "generator_seed": GENERATOR_SEED,
                                 "dataset": dataset, "series_id": series_id}))
        key = (dataset, config, series_id, origin)
        if (key, "current") not in omega_rows:
            value = float(omega_lookup.loc[(dataset, series_id)])
            for variant, start in (("current", 0), ("last_4096", max(0, origin - 4096))):
                omega_rows[(key, variant)] = dict(dataset=dataset, config=config,
                    series_id=series_id, source_revision=SOURCE_COMMIT,
                    source_sha256=source_sha, origin=origin, variant=variant,
                    history_start=start, history_end=origin, omega=value)
        for seed in EXPORTED_SEEDS:
            identity = dict(schema_version=1, fixture=True, evidence_kind="synthetic_fixture",
                exporter_file_sha256=exporter_sha,
                naming_scheme="dataset-config-v1", source_repository=SOURCE_REPOSITORY,
                source_revision=SOURCE_COMMIT, source_sha256=source_sha,
                source_hash_kind="generator_commit_file_seed_and_series_identity",
                generator_seed=GENERATOR_SEED, seed_role="duplicated_fixture_coverage",
                dataset=dataset, config=config, series_id=series_id, model=model, family=family,
                origin=origin, horizon=horizon, seed=seed, frequency="h",
                start_timestamp=START_TIMESTAMP)
            identity["origin_timestamp"] = (
                pd.Timestamp(START_TIMESTAMP) + pd.Timedelta(hours=origin)).isoformat()
            job_id = _sha(_json(identity))
            values = dict(job_id=[job_id] * horizon, dataset=[dataset] * horizon,
                config=[config] * horizon, series_id=[series_id] * horizon,
                model=[model] * horizon, family=[family] * horizon,
                origin=[origin] * horizon, horizon=[horizon] * horizon,
                seed=[seed] * horizon, step=group.step.tolist(),
                y_true=group.y.to_numpy(dtype=float), timestamp=target_timestamps(identity))
            values.update({name: group[name].to_numpy(dtype=float) for name in QCOLS})
            table = validate_payload(pa.table(values, schema=SCHEMA))
            table = table.replace_schema_metadata({
                b"fixture": b"true", b"evidence_kind": b"synthetic_fixture",
                b"job": _json({**identity, "job_id": job_id}),
                b"payload_sha256": _sha(_json(table.to_pydict())).encode(),
            })
            partitions.append((f"model={model}/{job_id}.parquet", table))
    return partitions, [omega_rows[key] for key in sorted(omega_rows)]


def export_handoff(source: Path, output: Path) -> dict:
    """Create one deterministic, visibly synthetic bundle in a new directory."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    (forecasts, omega, answers), generator_sha = _source_package(source)
    partitions, omega_rows = _jobs(_bounded(forecasts), omega, generator_sha)
    answer_rows = answers.sort_values(["dataset", "tau"]).to_dict("records")
    sidecar = dict(schema_version=1, fixture=True, evidence_kind="synthetic_omega_labels",
        naming_scheme="dataset-config-v1", source_repository=SOURCE_REPOSITORY,
        source_revision=SOURCE_COMMIT, generator_file_sha256=generator_sha,
        variants={"current": "Veer fixture omega_levels label; not measured",
                  "last_4096": "Same Veer fixture omega_levels label; not measured"},
        rows=omega_rows)
    expected = dict(fixture=True, evidence_kind="analytic_expected_costs_not_finite_sample_scores",
        source_revision=SOURCE_COMMIT, generator_file_sha256=generator_sha, rows=answer_rows)
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for relative, table in partitions:
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="zstd")
        hashes[relative] = _file_sha(path)
    (output / "omega.json").write_bytes(_json(sidecar) + b"\n")
    (output / "planted_answers.json").write_bytes(_json(expected) + b"\n")
    manifest = dict(schema_version=1, fixture=True, evidence_kind="synthetic_fixture",
        exporter_file_sha256=_file_sha(Path(__file__)),
        naming_scheme="dataset-config-v1", source_repository=SOURCE_REPOSITORY,
        source_revision=SOURCE_COMMIT, generator_file_sha256=generator_sha,
        generator_seed=GENERATOR_SEED, exported_seeds=list(EXPORTED_SEEDS),
        seed_role="duplicate coverage, not independent model runs",
        origin_semantics="zero-based first forecast target",
        timestamp_semantics="target position = origin + step - 1; hourly from 2020-01-01",
        horizon=24, origins_per_series=2, partition_hashes=hashes,
        omega_sha256=_file_sha(output / "omega.json"),
        planted_answers_sha256=_file_sha(output / "planted_answers.json"))
    (output / "handoff.json").write_bytes(_json(manifest) + b"\n")
    return manifest
