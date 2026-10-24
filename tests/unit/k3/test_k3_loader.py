"""Tests for the K3 input loader. Invented fixture data only; no M5/PJM, no inference.

unittest, to match the repository convention and the CI workflows. These also
run under pytest.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

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
                             omega_status="ok",
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


def _undefine(series_id, status="flat_history"):
    """Mark one series' Omega undefined, with a status, in both variants."""
    def mutate(doc):
        for row in doc["rows"]:
            if row["series_id"] == series_id:
                row["omega"] = None
                row["omega_status"] = status
    return mutate


class LoaderTestCase(unittest.TestCase):
    """Gives each test its own directory, standing in for pytest's tmp_path."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.tmp = Path(self._dir.name)

class HappyPathsTests(LoaderTestCase):
    """happy paths."""

    def test_real_inputs_join_and_report(self):
        table, omega, report = run(self.tmp, build_cache(self.tmp), omega_doc())
        self.assertTrue(report["status"] == "ok")
        self.assertTrue(report["series"] == 3 and report["comparisons"] == 3)
        self.assertTrue(report["distinct_omega"] == 3)
        self.assertTrue(report["provenance_checked_jobs"] == 6)
        self.assertTrue(omega[("synthetic", "hourly", "s1", 5000)] == 0.5)


    def test_variant_selection_is_explicit(self):
        _, omega, report = run(self.tmp, build_cache(self.tmp), omega_doc(), variant="last_4096")
        self.assertTrue(report["variant"] == "last_4096")
        self.assertAlmostEqual(omega[("synthetic", "hourly", "s1", 5000)], 0.51)


    def test_aadrika_decision_fixture_loads_as_fixture_valid(self):
        pq.write_table(decision_fixture(), self.tmp / "fixture.parquet")
        hashes = {"fixture.parquet": hashlib.sha256((self.tmp / "fixture.parquet").read_bytes()).hexdigest()}
        doc = omega_doc(series=(("fixture_series_0", 0.4),), origin=1000, fixture=True)
        for row in doc["rows"]:
            row.update(dataset="synthetic_fixture")
        _, _, report = run(self.tmp, hashes, doc, allow_fixture=True)
        self.assertTrue(report["status"] == "fixture_valid")
        self.assertTrue(report["distinct_omega"] == 1)


class RejectionRulesTests(LoaderTestCase):
    """rejection rules."""

    def test_partition_hash_mismatch_rejected(self):
        hashes = build_cache(self.tmp)
        hashes[next(iter(hashes))] = "b" * 64
        with self.assertRaisesRegex(K3InputError, "SHA-256 mismatch"):
            run(self.tmp, hashes, omega_doc())


    def test_unlisted_files_are_never_read(self):
        hashes = build_cache(self.tmp)
        write_job(self.tmp / "stale.parquet", "stale", "s9", 5000, "x", "classical")
        _, _, report = run(self.tmp, hashes, omega_doc())
        self.assertTrue(report["series"] == 3)  # stale partition ignored, no globbing


    def test_missing_omega_row_rejected(self):
        doc = omega_doc(series=(("s0", 0.2), ("s1", 0.5)))
        with self.assertRaisesRegex(K3InputError, "no Omega row"):
            run(self.tmp, build_cache(self.tmp), doc)


    def test_future_leakage_rejected(self):
        def leak(doc):
            doc["rows"][0]["history_end"] = 5001
        with self.assertRaisesRegex(K3InputError, "end <= origin"):
            run(self.tmp, build_cache(self.tmp), omega_doc(mutate=leak))


    def test_missing_variant_for_a_source_rejected(self):
        def drop(doc):
            doc["rows"] = [r for r in doc["rows"] if not (r["series_id"] == "s1" and r["variant"] == "last_4096")]
        with self.assertRaisesRegex(K3InputError, "missing"):
            run(self.tmp, build_cache(self.tmp), omega_doc(mutate=drop))


    def test_inconsistent_capped_bounds_rejected(self):
        def shift(doc):
            next(r for r in doc["rows"] if r["variant"] == "last_4096")["history_start"] = 100
        with self.assertRaisesRegex(K3InputError, "bounds inconsistent"):
            run(self.tmp, build_cache(self.tmp), omega_doc(mutate=shift))


    def test_source_revision_mismatch_rejected(self):
        hashes = build_cache(self.tmp, source_sha="c" * 64)
        with self.assertRaisesRegex(K3InputError, "different source revision"):
            run(self.tmp, hashes, omega_doc())


    def test_fixture_refused_without_opt_in(self):
        with self.assertRaisesRegex(K3InputError, "Fixture Omega refused"):
            run(self.tmp, build_cache(self.tmp), omega_doc(fixture=True))


    def test_missing_family_rejected(self):
        hashes = build_cache(self.tmp)
        for name in [n for n in hashes if n.startswith("s2_tsfm")]:
            del hashes[name]
        with self.assertRaisesRegex(K3InputError, "missing a family"):
            run(self.tmp, hashes, omega_doc())


    def test_actuals_disagreeing_across_models_rejected(self):
        hashes = build_cache(self.tmp, actual_override={("s1", "tsfm_m"): 11.0})
        with self.assertRaisesRegex(K3InputError, "Actuals or timestamps differ"):
            run(self.tmp, hashes, omega_doc())


    def test_undeclared_variant_rejected(self):
        with self.assertRaisesRegex(K3InputError, "not declared"):
            run(self.tmp, build_cache(self.tmp), omega_doc(), variant="differences")


    def test_duplicate_omega_row_rejected(self):
        def dup(doc):
            doc["rows"].append(dict(doc["rows"][0]))
        with self.assertRaisesRegex(K3InputError, "duplicate"):
            run(self.tmp, build_cache(self.tmp), omega_doc(mutate=dup))


    def test_bare_null_omega_rejected(self):
        """A null without a reason is indistinguishable from a dropped field."""
        def blank(doc):
            for row in doc["rows"]:
                if row["series_id"] == "s1":
                    row["omega"] = None
                    row.pop("omega_status", None)
        with self.assertRaisesRegex(K3InputError, "needs a non-empty omega_status"):
            run(self.tmp, build_cache(self.tmp), omega_doc(mutate=blank))


    def test_distinct_omega_count_reported_not_hidden(self):
        same = (("s0", 0.4), ("s1", 0.4), ("s2", 0.4))
        _, _, report = run(self.tmp, build_cache(self.tmp, series=same), omega_doc(series=same))
        self.assertTrue(report["distinct_omega"] == 1)


class AadrikaProvenanceMandatoryForRealInputsTests(LoaderTestCase):
    """Aadrika's review: provenance mandatory for real inputs."""

    def test_real_partition_without_provenance_rejected(self):
        """A real job carrying no source metadata must fail, not be skipped."""
        hashes = build_cache(self.tmp)
        name = "s0_classical_m.parquet"
        write_job(self.tmp / name, "s0:classical_m", "s0", 5000, "classical_m", "classical",
                  pred=9.0, job_meta=None)  # no job metadata at all
        hashes[name] = hashlib.sha256((self.tmp / name).read_bytes()).hexdigest()
        with self.assertRaisesRegex(K3InputError, "provenance is mandatory"):
            run(self.tmp, hashes, omega_doc())


    def test_fixture_inputs_do_not_require_provenance(self):
        pq.write_table(decision_fixture(), self.tmp / "fixture.parquet")
        hashes = {"fixture.parquet": hashlib.sha256((self.tmp / "fixture.parquet").read_bytes()).hexdigest()}
        doc = omega_doc(series=(("fixture_series_0", 0.4),), origin=1000, fixture=True)
        for row in doc["rows"]:
            row.update(dataset="synthetic_fixture")
        _, _, report = run(self.tmp, hashes, doc, allow_fixture=True)
        self.assertTrue(report["status"] == "fixture_valid")
        self.assertIs(report["provenance_required"], False)


    def test_real_inputs_report_provenance_required(self):
        _, _, report = run(self.tmp, build_cache(self.tmp), omega_doc())
        self.assertIs(report["provenance_required"], True)
        self.assertTrue(report["provenance_checked_jobs"] == 6)


class VeerSecondConditionCoveragePerOmegaBandEveryResultTests(LoaderTestCase):
    """Veer's second condition: coverage per Omega band, every result."""

    def test_band_coverage_reported(self):
        series = (("s0", 0.22), ("s1", 0.45), ("s2", 0.78))
        _, _, report = run(self.tmp, build_cache(self.tmp, series=series), omega_doc(series=series))
        bands = report["omega_band_coverage"]
        self.assertTrue(bands["below_0.40"]["series_origins"] == 1)
        self.assertTrue(bands["0.40_to_0.50"]["series_origins"] == 1)
        self.assertTrue(bands["above_0.50"]["series_origins"] == 1)
        self.assertTrue(bands["0.40_to_0.50"]["range"] == [0.40, 0.50])


    def test_empty_band_is_visible_not_omitted(self):
        """Eight values all above 0.50 must not look like boundary evidence."""
        series = tuple((f"s{i}", 0.55 + i * 0.03) for i in range(8))
        _, _, report = run(self.tmp, build_cache(self.tmp, series=series), omega_doc(series=series))
        bands = report["omega_band_coverage"]
        self.assertTrue(report["distinct_omega"] == 8)
        self.assertTrue(bands["above_0.50"]["series_origins"] == 8)
        self.assertTrue(bands["below_0.40"]["series_origins"] == 0)
        self.assertTrue(bands["0.40_to_0.50"]["series_origins"] == 0)
        self.assertTrue(bands["0.40_to_0.50"]["datasets"] == [])


    def test_band_boundaries_are_half_open(self):
        series = (("s0", 0.40), ("s1", 0.50), ("s2", 0.30))
        _, _, report = run(self.tmp, build_cache(self.tmp, series=series), omega_doc(series=series))
        bands = report["omega_band_coverage"]
        self.assertTrue(bands["0.40_to_0.50"]["series_origins"] == 1)  # 0.40 lands here
        self.assertTrue(bands["above_0.50"]["series_origins"] == 1)  # 0.50 lands here
        self.assertTrue(bands["below_0.40"]["series_origins"] == 1)


    def test_distinct_datasets_reported_for_boundary_counting(self):
        _, _, report = run(self.tmp, build_cache(self.tmp), omega_doc())
        self.assertTrue(report["distinct_datasets"] == 1)  # all three series share one dataset
        self.assertTrue(report["distinct_omega"] == 3)


class AadrikaNamingRepositoryValidation45Tests(LoaderTestCase):
    """Aadrika's review: naming and repository validation (#45)."""

    def test_legacy_slash_dataset_rejected_under_v1(self):
        """dataset="Salesforce/GiftEval" is a repository, not a production dataset."""
        def legacy(doc):
            for row in doc["rows"]:
                row.update(dataset="Salesforce/GiftEval", config="ett1/H")
        hashes = build_cache(self.tmp)
        with self.assertRaisesRegex(K3InputError, "legacy slash form"):
            run(self.tmp, hashes, omega_doc(mutate=legacy))


    def test_legacy_naming_accepted_when_explicitly_requested(self):
        from decision_aware_tsfm.analysis.k3.loader import NAMING_LEGACY
        import json as _json
        hashes = build_cache(self.tmp)
        doc = omega_doc()
        path = self.tmp / "omega.json"
        path.write_text(_json.dumps(doc))
        _, _, report = load_k3_inputs(self.tmp, hashes, path, "current", naming=NAMING_LEGACY)
        self.assertTrue(report["naming"] == NAMING_LEGACY)


    def test_empty_config_is_valid(self):
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
                pq.write_table(table, self.tmp / name)
                hashes[name] = hashlib.sha256((self.tmp / name).read_bytes()).hexdigest()

        def rename(doc):
            for row in doc["rows"]:
                row.update(dataset="hospital", config="")
        _, _, report = run(self.tmp, hashes, omega_doc(series=series, mutate=rename))
        self.assertTrue(report["status"] == "ok")


    def test_source_repository_mismatch_rejected(self):
        def repo(doc):
            doc["source_repository"] = "SomeoneElse/OtherData"
        hashes = build_cache(self.tmp)
        for name in list(hashes):
            table = pq.ParquetFile(self.tmp / name).read()
            job = json.loads(table.schema.metadata[b"job"])
            job["source_repository"] = "Salesforce/GiftEval"
            pq.write_table(table.replace_schema_metadata({b"job": json.dumps(job).encode()}),
                           self.tmp / name)
            hashes[name] = hashlib.sha256((self.tmp / name).read_bytes()).hexdigest()
        with self.assertRaisesRegex(K3InputError, "different source repositories"):
            run(self.tmp, hashes, omega_doc(mutate=repo))


    def test_source_repository_reported_when_declared(self):
        def repo(doc):
            doc["source_repository"] = "Salesforce/GiftEval"
        _, _, report = run(self.tmp, build_cache(self.tmp), omega_doc(mutate=repo))
        self.assertTrue(report["source_repository"] == "Salesforce/GiftEval")


    def test_blank_repository_rejected(self):
        def blank(doc):
            doc["source_repository"] = "   "
        with self.assertRaisesRegex(K3InputError, "source_repository"):
            run(self.tmp, build_cache(self.tmp), omega_doc(mutate=blank))


class UndefinedOmegaNullWithReasonCarriedThroughForDeferralTests(LoaderTestCase):
    """undefined Omega: null with a status, carried through for deferral."""


    def test_null_omega_with_reason_is_carried_through(self):
        _, omega, report = run(self.tmp, build_cache(self.tmp),
                               omega_doc(mutate=_undefine("s1")))
        self.assertTrue(report["status"] == "ok")
        self.assertIsNone(omega[("synthetic", "hourly", "s1", 5000)])
        self.assertTrue(report["undefined_omega"] == 1)
        self.assertEqual(report["undefined_omega_statuses"], {"flat_history": 1})


    def test_null_omega_excluded_from_counts_and_bands(self):
        _, _, report = run(self.tmp, build_cache(self.tmp), omega_doc(mutate=_undefine("s1")))
        self.assertTrue(report["distinct_omega"] == 2)  # s1 no longer counted
        self.assertTrue(report["omega_range"] == [0.2, 0.8])  # s1's 0.5 gone from the range
        bands = report["omega_band_coverage"]
        self.assertTrue(bands["undefined"]["series_origins"] == 1)
        self.assertTrue(bands["undefined"]["datasets"] == ["synthetic"])
        self.assertEqual(
            sum(bands[b]["series_origins"]
                for b in ("below_0.40", "0.40_to_0.50", "above_0.50")), 2)


    def test_zero_offered_as_undefined_rejected(self):
        """0.0 must not be dressed up as undefined; it reads as a real low Omega."""
        def zero(doc):
            for row in doc["rows"]:
                if row["series_id"] == "s1":
                    row["omega"] = 0.0
                    row["omega_status"] = "flat_history"
        with self.assertRaisesRegex(K3InputError, "a defined value must be"):
            run(self.tmp, build_cache(self.tmp), omega_doc(mutate=zero))


    def test_blank_reason_rejected(self):
        with self.assertRaisesRegex(K3InputError, "needs a non-empty omega_status"):
            run(self.tmp, build_cache(self.tmp), omega_doc(mutate=_undefine("s1", "   ")))


    def test_undefined_in_one_variant_only_rejected(self):
        def half(doc):
            for row in doc["rows"]:
                if row["series_id"] == "s1" and row["variant"] == "current":
                    row["omega"] = None
                    row["omega_status"] = "flat_history"
        with self.assertRaisesRegex(K3InputError, "undefined in some variants but not others"):
            run(self.tmp, build_cache(self.tmp), omega_doc(mutate=half))


    def test_unknown_status_is_reported_not_rejected(self):
        """manifest-v2 added near_flat and broke consumers that hardcode the
        known set, so a status we have not seen before must still load."""
        _, _, report = run(self.tmp, build_cache(self.tmp),
                           omega_doc(mutate=_undefine("s1", "near_flat")))
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["undefined_omega_statuses"], {"near_flat": 1})

    def test_statuses_are_counted_per_value(self):
        def mixed(doc):
            for row in doc["rows"]:
                if row["series_id"] in ("s0", "s1"):
                    row["omega"] = None
                    row["omega_status"] = ("flat_history" if row["series_id"] == "s0"
                                           else "empty_history")
        _, _, report = run(self.tmp, build_cache(self.tmp), omega_doc(mutate=mixed))
        self.assertEqual(report["undefined_omega"], 2)
        self.assertEqual(report["undefined_omega_statuses"],
                         {"empty_history": 1, "flat_history": 1})

    def test_all_omega_undefined_reports_empty_range(self):
        def nothing(doc):
            for row in doc["rows"]:
                row["omega"] = None
                row["omega_status"] = "flat_history"
        _, _, report = run(self.tmp, build_cache(self.tmp), omega_doc(mutate=nothing))
        self.assertTrue(report["distinct_omega"] == 0)
        self.assertTrue(report["distinct_datasets"] == 0)
        self.assertIsNone(report["omega_range"])
        self.assertTrue(report["undefined_omega"] == 3)


class VeerReviewOn35Tests(LoaderTestCase):
    """Items raised in Veer's review of #35."""

    def test_defined_omega_must_carry_status_ok(self):
        """A number under flat_history is a stand-in, not a measurement."""
        def bad(doc):
            for row in doc["rows"]:
                if row["series_id"] == "s1":
                    row["omega_status"] = "flat_history"
        with self.assertRaisesRegex(K3InputError, "a defined value must be"):
            run(self.tmp, build_cache(self.tmp), omega_doc(mutate=bad))

    def test_null_omega_may_not_claim_status_ok(self):
        def bad(doc):
            for row in doc["rows"]:
                if row["series_id"] == "s1":
                    row["omega"] = None
                    row["omega_status"] = "ok"
        with self.assertRaisesRegex(K3InputError, "'ok' on a null omega"):
            run(self.tmp, build_cache(self.tmp), omega_doc(mutate=bad))

    def test_omega_outside_zero_one_refused(self):
        for bad_value in (1.5, -0.1):
            def out(doc, v=bad_value):
                doc["rows"][0]["omega"] = v
            with self.assertRaisesRegex(K3InputError, r"outside \[0, 1\]"):
                run(self.tmp, build_cache(self.tmp), omega_doc(mutate=out))

    def test_held_out_dataset_refused(self):
        """The engine refuses m5 and pjm before any cost; so must this."""
        for name in ("m5", "M5_daily", "retail_pjm"):
            def rename(doc, n=name):
                for row in doc["rows"]:
                    row["dataset"] = n
            with self.assertRaisesRegex(K3InputError, "Held-out data refused"):
                run(self.tmp, build_cache(self.tmp), omega_doc(mutate=rename))

    def test_held_out_config_refused(self):
        def rename(doc):
            for row in doc["rows"]:
                row["config"] = "PJM"
        with self.assertRaisesRegex(K3InputError, "Held-out data refused"):
            run(self.tmp, build_cache(self.tmp), omega_doc(mutate=rename))

    def test_actuals_differing_across_seeds_refused(self):
        """Same target position, two seeds, two different y_true."""
        hashes = build_cache(self.tmp)
        for model, family, pred in (("classical_m", "classical", 9.0),
                                    ("tsfm_m", "tsfm", 10.5)):
            name = f"s0_{model}_seed43.parquet"
            write_job(self.tmp / name, f"s0:{model}:43", "s0", 5000, model, family,
                      y=11.0, pred=pred, seed=43,   # same targets, different actual
                      job_meta={"source_revision": REV, "source_sha256": SHA})
            hashes[name] = hashlib.sha256((self.tmp / name).read_bytes()).hexdigest()
        with self.assertRaisesRegex(K3InputError, "same target position"):
            run(self.tmp, hashes, omega_doc())

    def test_actuals_differing_across_horizons_refused(self):
        """Horizon 3 covers the same first two targets as horizon 2."""
        hashes = build_cache(self.tmp)
        for model, family in (("classical_m", "classical"), ("tsfm_m", "tsfm")):
            name = f"s0_{model}_h3.parquet"
            write_job(self.tmp / name, f"s0:{model}:h3", "s0", 5000, model, family,
                      y=12.0, pred=9.0, horizon=3,
                      job_meta={"source_revision": REV, "source_sha256": SHA})
            hashes[name] = hashlib.sha256((self.tmp / name).read_bytes()).hexdigest()
        with self.assertRaisesRegex(K3InputError, "same target position"):
            run(self.tmp, hashes, omega_doc())

    def test_sidecar_hash_checked_when_supplied(self):
        hashes = build_cache(self.tmp)
        path = self.tmp / "omega.json"
        path.write_text(json.dumps(omega_doc()))
        with self.assertRaisesRegex(K3InputError, "SHA-256 does not match"):
            load_k3_inputs(self.tmp, hashes, path, "current", omega_sha256="f" * 64)

    def test_sidecar_hash_accepted_when_correct(self):
        hashes = build_cache(self.tmp)
        path = self.tmp / "omega.json"
        path.write_text(json.dumps(omega_doc()))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _, _, report = load_k3_inputs(self.tmp, hashes, path, "current",
                                      omega_sha256=digest)
        self.assertIs(report["omega_sha256_verified"], True)


class PlantedBugsVeerCaughtTests(LoaderTestCase):
    """The four planted bugs Veer's run found my tests did not cover."""

    def test_listed_partition_absent_refused(self):
        hashes = build_cache(self.tmp)
        hashes["never_written.parquet"] = "a" * 64
        with self.assertRaisesRegex(K3InputError, "not found"):
            run(self.tmp, hashes, omega_doc())

    def test_fixture_cache_refused_without_opt_in(self):
        """Fixture *cache*, real Omega: still has to be opted into."""
        hashes = build_cache(self.tmp, fixture=True)
        with self.assertRaisesRegex(K3InputError, "Fixture cache refused"):
            run(self.tmp, hashes, omega_doc())

    def test_left_over_omega_rows_are_counted(self):
        extra = (("s0", 0.2), ("s1", 0.5), ("s2", 0.8), ("s3", 0.9))
        _, _, report = run(self.tmp, build_cache(self.tmp), omega_doc(series=extra))
        self.assertEqual(report["unused_omega_sources"], 1)

    def test_timestamp_mismatch_refused(self):
        hashes = {}
        for sid, _ in (("s0", 0.2), ("s1", 0.5), ("s2", 0.8)):
            for model, family, pred in (("classical_m", "classical", 9.0),
                                        ("tsfm_m", "tsfm", 10.5)):
                name = f"{sid}_{model}.parquet"
                stamps = ({1: "2020-06-06T05:00:00"}
                          if (sid, model) == ("s1", "tsfm_m") else None)
                write_job(self.tmp / name, f"{sid}:{model}", sid, 5000, model, family,
                          pred=pred, timestamps=stamps,
                          job_meta={"source_revision": REV, "source_sha256": SHA})
                hashes[name] = hashlib.sha256((self.tmp / name).read_bytes()).hexdigest()
        with self.assertRaisesRegex(K3InputError, "same target position"):
            run(self.tmp, hashes, omega_doc())


if __name__ == "__main__":
    unittest.main()
