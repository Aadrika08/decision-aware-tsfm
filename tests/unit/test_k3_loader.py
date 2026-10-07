"""Tests for the K3 input loader. Invented fixture data only; no M5/PJM, no inference."""
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from decision_aware_tsfm.analysis.k3.loader import K3InputError, load_k3_inputs
from decision_aware_tsfm.cache.schema import QCOLS, SCHEMA, decision_fixture  # noqa: F811
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


# ------------------ Aadrika's review: provenance mandatory for real inputs
def test_real_partition_without_provenance_rejected(tmp_path):
    """A real job carrying no source metadata must fail, not be skipped."""
    hashes = build_cache(tmp_path)
    name = "s0_classical_m.parquet"
    write_job(tmp_path / name, "s0:classical_m", "s0", 5000, "classical_m", "classical",
              pred=9.0, job_meta=None)  # no job metadata at all
    hashes[name] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    with pytest.raises(K3InputError, match="provenance is mandatory"):
        run(tmp_path, hashes, omega_doc())


def test_fixture_inputs_do_not_require_provenance(tmp_path):
    pq.write_table(decision_fixture(), tmp_path / "fixture.parquet")
    hashes = {"fixture.parquet": hashlib.sha256((tmp_path / "fixture.parquet").read_bytes()).hexdigest()}
    doc = omega_doc(series=(("fixture_series_0", 0.4),), origin=1000, fixture=True)
    for row in doc["rows"]:
        row.update(dataset="synthetic_fixture")
    _, _, report = run(tmp_path, hashes, doc, allow_fixture=True)
    assert report["status"] == "fixture_valid"
    assert report["provenance_required"] is False


def test_real_inputs_report_provenance_required(tmp_path):
    _, _, report = run(tmp_path, build_cache(tmp_path), omega_doc())
    assert report["provenance_required"] is True
    assert report["provenance_checked_jobs"] == 6


# ----------- Veer's second condition: coverage per Omega band, every result
def test_band_coverage_reported(tmp_path):
    series = (("s0", 0.22), ("s1", 0.45), ("s2", 0.78))
    _, _, report = run(tmp_path, build_cache(tmp_path, series=series), omega_doc(series=series))
    bands = report["omega_band_coverage"]
    assert bands["below_0.40"]["series_origins"] == 1
    assert bands["0.40_to_0.50"]["series_origins"] == 1
    assert bands["above_0.50"]["series_origins"] == 1
    assert bands["0.40_to_0.50"]["range"] == [0.40, 0.50]


def test_empty_band_is_visible_not_omitted(tmp_path):
    """Eight values all above 0.50 must not look like boundary evidence."""
    series = tuple((f"s{i}", 0.55 + i * 0.03) for i in range(8))
    _, _, report = run(tmp_path, build_cache(tmp_path, series=series), omega_doc(series=series))
    bands = report["omega_band_coverage"]
    assert report["distinct_omega"] == 8
    assert bands["above_0.50"]["series_origins"] == 8
    assert bands["below_0.40"]["series_origins"] == 0
    assert bands["0.40_to_0.50"]["series_origins"] == 0
    assert bands["0.40_to_0.50"]["datasets"] == []


def test_band_boundaries_are_half_open(tmp_path):
    series = (("s0", 0.40), ("s1", 0.50), ("s2", 0.30))
    _, _, report = run(tmp_path, build_cache(tmp_path, series=series), omega_doc(series=series))
    bands = report["omega_band_coverage"]
    assert bands["0.40_to_0.50"]["series_origins"] == 1  # 0.40 lands here
    assert bands["above_0.50"]["series_origins"] == 1    # 0.50 lands here
    assert bands["below_0.40"]["series_origins"] == 1


def test_distinct_datasets_reported_for_boundary_counting(tmp_path):
    _, _, report = run(tmp_path, build_cache(tmp_path), omega_doc())
    assert report["distinct_datasets"] == 1  # all three series share one dataset
    assert report["distinct_omega"] == 3


# ---------------- Aadrika's review: naming and repository validation (#45)
def test_legacy_slash_dataset_rejected_under_v1(tmp_path):
    """dataset="Salesforce/GiftEval" is a repository, not a production dataset."""
    def legacy(doc):
        for row in doc["rows"]:
            row.update(dataset="Salesforce/GiftEval", config="ett1/H")
    hashes = build_cache(tmp_path)
    with pytest.raises(K3InputError, match="legacy slash form"):
        run(tmp_path, hashes, omega_doc(mutate=legacy))


def test_legacy_naming_accepted_when_explicitly_requested(tmp_path):
    from decision_aware_tsfm.analysis.k3.loader import NAMING_LEGACY
    import json as _json
    hashes = build_cache(tmp_path)
    doc = omega_doc()
    path = tmp_path / "omega.json"
    path.write_text(_json.dumps(doc))
    _, _, report = load_k3_inputs(tmp_path, hashes, path, "current", naming=NAMING_LEGACY)
    assert report["naming"] == NAMING_LEGACY


def test_empty_config_is_valid(tmp_path):
    """A known source with no frequency variant carries config=""."""
    series = (("s0", 0.2), ("s1", 0.5), ("s2", 0.8))
    hashes = {}
    for sid, _ in series:
        for model, family, pred in (("classical_m", "classical", 9.0), ("tsfm_m", "tsfm", 10.5)):
            name = f"{sid}_{model}.parquet"
            rows = []
            for step in (1, 2):
                row = dict(job_id=f"{sid}:{model}", dataset="hospital", config="",
                           series_id=sid, model=model, family=family, origin=5000,
                           horizon=2, seed=42, step=step, y_true=10.0,
                           timestamp=f"2020-01-01T{step:02d}:00:00")
                row.update({c: pred for c in QCOLS})
                rows.append(row)
            table = pa.Table.from_pylist(rows, schema=SCHEMA)
            table = table.replace_schema_metadata(
                {b"job": json.dumps({"source_revision": REV, "source_sha256": SHA}).encode()})
            pq.write_table(table, tmp_path / name)
            hashes[name] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()

    def rename(doc):
        for row in doc["rows"]:
            row.update(dataset="hospital", config="")
    _, _, report = run(tmp_path, hashes, omega_doc(series=series, mutate=rename))
    assert report["status"] == "ok"


def test_source_repository_mismatch_rejected(tmp_path):
    def repo(doc):
        doc["source_repository"] = "SomeoneElse/OtherData"
    hashes = build_cache(tmp_path)
    for name in list(hashes):
        table = pq.ParquetFile(tmp_path / name).read()
        job = json.loads(table.schema.metadata[b"job"])
        job["source_repository"] = "Salesforce/GiftEval"
        pq.write_table(table.replace_schema_metadata({b"job": json.dumps(job).encode()}),
                       tmp_path / name)
        hashes[name] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    with pytest.raises(K3InputError, match="different source repositories"):
        run(tmp_path, hashes, omega_doc(mutate=repo))


def test_source_repository_reported_when_declared(tmp_path):
    def repo(doc):
        doc["source_repository"] = "Salesforce/GiftEval"
    _, _, report = run(tmp_path, build_cache(tmp_path), omega_doc(mutate=repo))
    assert report["source_repository"] == "Salesforce/GiftEval"


def test_blank_repository_rejected(tmp_path):
    def blank(doc):
        doc["source_repository"] = "   "
    with pytest.raises(K3InputError, match="source_repository"):
        run(tmp_path, build_cache(tmp_path), omega_doc(mutate=blank))
