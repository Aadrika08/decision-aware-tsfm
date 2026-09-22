"""Tests for the K3 input loader. Invented fixture data only; no M5/PJM, no inference."""
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from decision_aware_tsfm.analysis.k3.loader import K3InputError, load_k3_inputs
from decision_aware_tsfm.cache.schema import QCOLS, SCHEMA, decision_fixture

REV = "fixture-revision"
SHA = "a" * 64


def write_job(path, job_id, series_id, origin, model, family, y=10.0, pred=9.0,
              horizon=2, seed=42, job_meta=None, fixture=False, timestamps=None):
    rows = []
    for step in range(1, horizon + 1):
        row = dict(job_id=job_id, dataset="synthetic", config="hourly", series_id=series_id,
                   model=model, family=family, origin=origin, horizon=horizon, seed=seed,
                   step=step, y_true=y,
                   timestamp=(timestamps or {}).get(step, f"2020-01-01T{step:02d}:00:00"))
        row.update({c: pred for c in QCOLS})
        rows.append(row)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    meta = {}
    if job_meta is not None:
        meta[b"job"] = json.dumps(job_meta).encode()
    if fixture:
        meta[b"fixture"] = b"true"
    pq.write_table(table.replace_schema_metadata(meta or None), path)


def build_cache(tmp_path, series=(("s0", 0.2), ("s1", 0.5), ("s2", 0.8)), origin=5000,
                fixture=False, source_sha=SHA, actual_override=None):
    hashes = {}
    for sid, _ in series:
        for model, family, pred in (("classical_m", "classical", 9.0), ("tsfm_m", "tsfm", 10.5)):
            name = f"{sid}_{model}.parquet"
            y = actual_override.get((sid, model), 10.0) if actual_override else 10.0
            meta = {"source_revision": REV, "source_sha256": source_sha}
            write_job(tmp_path / name, f"{sid}:{model}", sid, origin, model, family,
                      y=y, pred=pred, job_meta=meta, fixture=fixture)
            hashes[name] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    return hashes


def omega_doc(series=(("s0", 0.2), ("s1", 0.5), ("s2", 0.8)), origin=5000, fixture=False,
              variants=("current", "last_4096"), mutate=None):
    rows = []
    for sid, value in series:
        for v in variants:
            start = 0 if v == "current" else max(0, origin - 4096)
            rows.append(dict(dataset="synthetic", config="hourly", series_id=sid,
                             source_revision=REV, source_sha256=SHA, origin=origin,
                             variant=v, history_start=start, history_end=origin,
                             omega=value if v == "current" else value + 0.01))
    doc = {"schema_version": 1, "fixture": fixture,
           "variants": {"current": "full history", "last_4096": "last 4096 steps"},
           "rows": rows}
    if mutate:
        mutate(doc)
    return doc


def run(tmp_path, hashes, doc, variant="current", allow_fixture=False):
    path = tmp_path / "omega.json"
    path.write_text(json.dumps(doc))
    return load_k3_inputs(tmp_path, hashes, path, variant, allow_fixture=allow_fixture)


# ------------------------------------------------------------- happy paths
def test_real_inputs_join_and_report(tmp_path):
    table, omega, report = run(tmp_path, build_cache(tmp_path), omega_doc())
    assert report["status"] == "ok"
    assert report["series"] == 3 and report["comparisons"] == 3
    assert report["distinct_omega"] == 3
    assert report["provenance_checked_jobs"] == 6
    assert omega[("synthetic", "hourly", "s1", 5000)] == 0.5


def test_variant_selection_is_explicit(tmp_path):
    _, omega, report = run(tmp_path, build_cache(tmp_path), omega_doc(), variant="last_4096")
    assert report["variant"] == "last_4096"
    assert omega[("synthetic", "hourly", "s1", 5000)] == pytest.approx(0.51)


def test_aadrika_decision_fixture_loads_as_fixture_valid(tmp_path):
    pq.write_table(decision_fixture(), tmp_path / "fixture.parquet")
    hashes = {"fixture.parquet": hashlib.sha256((tmp_path / "fixture.parquet").read_bytes()).hexdigest()}
    doc = omega_doc(series=(("fixture_series_0", 0.4),), origin=1000, fixture=True)
    for row in doc["rows"]:
        row.update(dataset="synthetic_fixture")
    _, _, report = run(tmp_path, hashes, doc, allow_fixture=True)
    assert report["status"] == "fixture_valid"
    assert report["distinct_omega"] == 1


# ---------------------------------------------------------- rejection rules
def test_partition_hash_mismatch_rejected(tmp_path):
    hashes = build_cache(tmp_path)
    hashes[next(iter(hashes))] = "b" * 64
    with pytest.raises(K3InputError, match="SHA-256 mismatch"):
        run(tmp_path, hashes, omega_doc())


def test_unlisted_files_are_never_read(tmp_path):
    hashes = build_cache(tmp_path)
    write_job(tmp_path / "stale.parquet", "stale", "s9", 5000, "x", "classical")
    _, _, report = run(tmp_path, hashes, omega_doc())
    assert report["series"] == 3  # stale partition ignored, no globbing


def test_missing_omega_row_rejected(tmp_path):
    doc = omega_doc(series=(("s0", 0.2), ("s1", 0.5)))
    with pytest.raises(K3InputError, match="no Omega row"):
        run(tmp_path, build_cache(tmp_path), doc)


def test_future_leakage_rejected(tmp_path):
    def leak(doc):
        doc["rows"][0]["history_end"] = 5001
    with pytest.raises(K3InputError, match="end <= origin"):
        run(tmp_path, build_cache(tmp_path), omega_doc(mutate=leak))


def test_missing_variant_for_a_source_rejected(tmp_path):
    def drop(doc):
        doc["rows"] = [r for r in doc["rows"] if not (r["series_id"] == "s1" and r["variant"] == "last_4096")]
    with pytest.raises(K3InputError, match="missing"):
        run(tmp_path, build_cache(tmp_path), omega_doc(mutate=drop))


def test_inconsistent_capped_bounds_rejected(tmp_path):
    def shift(doc):
        next(r for r in doc["rows"] if r["variant"] == "last_4096")["history_start"] = 100
    with pytest.raises(K3InputError, match="bounds inconsistent"):
        run(tmp_path, build_cache(tmp_path), omega_doc(mutate=shift))


def test_source_revision_mismatch_rejected(tmp_path):
    hashes = build_cache(tmp_path, source_sha="c" * 64)
    with pytest.raises(K3InputError, match="different source revision"):
        run(tmp_path, hashes, omega_doc())


def test_fixture_refused_without_opt_in(tmp_path):
    with pytest.raises(K3InputError, match="Fixture Omega refused"):
        run(tmp_path, build_cache(tmp_path), omega_doc(fixture=True))


def test_missing_family_rejected(tmp_path):
    hashes = build_cache(tmp_path)
    for name in [n for n in hashes if n.startswith("s2_tsfm")]:
        del hashes[name]
    with pytest.raises(K3InputError, match="missing a family"):
        run(tmp_path, hashes, omega_doc())


def test_actuals_disagreeing_across_models_rejected(tmp_path):
    hashes = build_cache(tmp_path, actual_override={("s1", "tsfm_m"): 11.0})
    with pytest.raises(K3InputError, match="Actuals or timestamps differ"):
        run(tmp_path, hashes, omega_doc())


def test_undeclared_variant_rejected(tmp_path):
    with pytest.raises(K3InputError, match="not declared"):
        run(tmp_path, build_cache(tmp_path), omega_doc(), variant="differences")


def test_duplicate_omega_row_rejected(tmp_path):
    def dup(doc):
        doc["rows"].append(dict(doc["rows"][0]))
    with pytest.raises(K3InputError, match="duplicate"):
        run(tmp_path, build_cache(tmp_path), omega_doc(mutate=dup))


def test_missing_omega_value_rejected(tmp_path):
    def blank(doc):
        doc["rows"][0]["omega"] = None  # undefined Omega needs an upstream decision
    with pytest.raises(K3InputError, match="missing or wrong type"):
        run(tmp_path, build_cache(tmp_path), omega_doc(mutate=blank))


def test_distinct_omega_count_reported_not_hidden(tmp_path):
    same = (("s0", 0.4), ("s1", 0.4), ("s2", 0.4))
    _, _, report = run(tmp_path, build_cache(tmp_path, series=same), omega_doc(series=same))
    assert report["distinct_omega"] == 1
