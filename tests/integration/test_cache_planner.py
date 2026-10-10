from pathlib import Path
import tempfile
import unittest

from decision_aware_tsfm.cache.planner import IDENTITY_FIELDS, job_identity, plan, target_timestamps
from decision_aware_tsfm.cache.schema import TAUS


def row(**updates):
    value = dict(source_revision="revision-1", source_sha256="a" * 64,
                 source_path="inputs/source.arrow", dataset="GiftEval", config="ett1/H",
                 series_id="ett1_H_item_0_dim0", item_id="item_0", channel=0,
                 frequency="H", start_timestamp="2020-01-01T00:00:00",
                 source_length=5000, usable_start=0, usable_end=4000,
                 context_start=0, context_length=2048, origin=2048, horizon=24,
                 seed=42, training_end=2000, validation_end=3000)
    value.update(updates)
    return value


def workload(rows=None, *, fixture=False, **updates):
    value = dict(schema_version=1, split="development", fixture=fixture,
                 quantiles=list(TAUS), rows=rows or [row()])
    if not fixture:
        value["approval_reference"] = "issue-17"
    value.update(updates)
    return value


def models(*, fixture=False, **updates):
    model = (dict(model="fixture_classical", family="classical", adapter="fixture",
                  adapter_version="1", checkpoint=None, device="cpu", batch_size=1,
                  quantile_method="constant", settings={"prediction": 5.0}) if fixture else
             dict(model="seasonal_naive", family="classical", adapter="seasonal_naive_smoke",
                  adapter_version="1", checkpoint=None, device="cpu", batch_size=1,
                  quantile_method="empirical_lag24_linear",
                  settings={"season_length": 24, "residual_lag": 24}))
    value = {"schema_version": 1, "models": [model]}
    if not fixture:
        value["approval_reference"] = "model-approval"
    value.update(updates)
    return value


class PlannerTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())

    def test_plans_exact_jobs_without_reading_source(self):
        report = plan(workload(), models(), base_dir=self.base)
        self.assertEqual(report["schema_version"], 2)
        self.assertFalse(report["fixture"])
        self.assertEqual(report["planned_jobs"], 1)
        self.assertEqual(report["planned_rows"], 24)
        self.assertEqual(report["counts_by_model"], {"seasonal_naive": 1})
        self.assertEqual(report["provenance"]["implementation_sha256"],
                         report["jobs"][0]["implementation_sha256"])
        self.assertIn("python", report["provenance"]["environment"])
        self.assertIn("git_commit", report["provenance"])
        job = report["jobs"][0]
        self.assertEqual(job["source_path"], str((self.base / "inputs/source.arrow").resolve()))
        self.assertEqual(job["approval_reference"], "issue-17")
        self.assertEqual(job["model_approval_reference"], "model-approval")
        self.assertEqual(job["origin_timestamp"], "2020-03-26T08:00:00")
        self.assertEqual(len(target_timestamps(job)), 24)
        self.assertNotIn("job_id", job_identity(job))

    def test_ids_ignore_order_path_approvals_and_omega_but_track_science(self):
        first = row(series_id="stable-a", omega=0.1)
        second = row(source_revision="revision-2", source_path="elsewhere.arrow",
                     series_id="stable-b", item_id="item_1", channel=1, origin=2072,
                     context_start=24, training_end=2072, omega=0.9)
        initial = plan(workload([first, second]), models(), base_dir=self.base)
        changed = [dict(second, source_path="moved/two.arrow", omega=-9),
                   dict(first, source_path="moved/one.arrow", omega=99)]
        reordered = plan(workload(changed, approval_reference="other"),
                         models(approval_reference="other-model"), base_dir=self.base)
        self.assertEqual([j["job_id"] for j in initial["jobs"]],
                         [j["job_id"] for j in reordered["jobs"]])
        revised = plan(workload([dict(first, source_revision="revision-x")]), models(), base_dir=self.base)
        self.assertNotEqual(initial["jobs"][0]["job_id"], revised["jobs"][0]["job_id"])
        fixture_one = plan(workload(fixture=True), models(fixture=True), base_dir=self.base)
        fixture_models = models(fixture=True)
        fixture_models["models"][0]["settings"] = {"prediction": 6.0}
        fixture_two = plan(workload(fixture=True), fixture_models, base_dir=self.base)
        self.assertNotEqual(fixture_one["jobs"][0]["job_id"], fixture_two["jobs"][0]["job_id"])
        altered_models = models()
        altered_models["models"][0]["settings"] = {"season_length": 12, "residual_lag": 24}
        with self.assertRaises(ValueError):
            plan(workload(), altered_models, base_dir=self.base)

    def test_production_names_and_repository_provenance(self):
        names = dict(naming_scheme="dataset-config-v1", source_repository="Salesforce/GiftEval")
        ett = plan(workload([row(dataset="ett1", config="H")], **names), models(),
                   base_dir=self.base)["jobs"][0]
        self.assertEqual((ett["dataset"], ett["config"]), ("ett1", "H"))
        self.assertEqual((ett["naming_scheme"], ett["source_repository"]),
                         ("dataset-config-v1", "Salesforce/GiftEval"))
        self.assertEqual(target_timestamps(ett)[0], ett["origin_timestamp"])
        self.assertEqual(target_timestamps(ett)[1], "2020-03-26T09:00:00")
        self.assertEqual(ett["origin"], ett["context_start"] + ett["context_length"])
        other_repo = plan(workload([row(dataset="ett1", config="H")],
                                   **dict(names, source_repository="other/GiftEval")),
                          models(), base_dir=self.base)["jobs"][0]
        self.assertNotEqual(ett["job_id"], other_repo["job_id"])
        m4 = plan(workload([row(dataset="m4", config="daily")], **names), models(),
                  base_dir=self.base)["jobs"][0]
        self.assertEqual((m4["dataset"], m4["config"]), ("m4", "daily"))
        absent = plan(workload([row(dataset="synthetic_no_variant", config="")], **names),
                      models(), base_dir=self.base)["jobs"][0]
        self.assertEqual(absent["config"], "")

    def test_production_namespaces_allow_item_reuse_and_distinct_channels(self):
        names = dict(naming_scheme="dataset-config-v1", source_repository="Salesforce/GiftEval")
        rows = [row(dataset="m4", config="daily", series_id="stable-0"),
                row(dataset="m4", config="monthly", series_id="stable-0",
                    source_revision="monthly-v1", source_sha256="b" * 64),
                row(dataset="m4", config="daily", series_id="stable-1", channel=1)]
        jobs = plan(workload(rows, **names), models(), base_dir=self.base)["jobs"]
        self.assertEqual(len(jobs), 3)
        self.assertEqual({job["item_id"] for job in jobs}, {"item_0"})
        self.assertEqual({(job["config"], job["channel"], job["series_id"]) for job in jobs},
                         {("daily", 0, "stable-0"), ("monthly", 0, "stable-0"),
                          ("daily", 1, "stable-1")})

    def test_legacy_identity_and_incomplete_naming_rejected(self):
        legacy = plan(workload(), models(), base_dir=self.base)["jobs"][0]
        self.assertEqual(job_identity(legacy), {field: legacy[field] for field in IDENTITY_FIELDS})
        self.assertNotIn("source_repository", legacy)
        for header in ({"naming_scheme": "dataset-config-v1"},
                       {"source_repository": "Salesforce/GiftEval"},
                       {"naming_scheme": "other", "source_repository": "Salesforce/GiftEval"},
                       {"naming_scheme": "dataset-config-v1", "source_repository": ""},
                       {"naming_scheme": "dataset-config-v1", "source_repository": " Salesforce/GiftEval"},
                       {"naming_scheme": "dataset-config-v1", "source_repository": "Salesforce/GiftEval "},
                       {"naming_scheme": "dataset-config-v1", "source_repository": "M5"}):
            with self.subTest(header=header), self.assertRaises(ValueError):
                plan(workload(**header), models(), base_dir=self.base)
        names = dict(naming_scheme="dataset-config-v1", source_repository="Salesforce/GiftEval")
        for bad in (row(dataset="ett1", config=None), row(dataset="GiftEval", config="ett1/H"),
                    row(dataset="ett1", config=" H"), row(dataset="ett1", config="H",
                        source_repository="Salesforce/GiftEval")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                plan(workload([bad], **names), models(), base_dir=self.base)
        with self.assertRaises(ValueError):
            plan(workload([row(config="")]), models(), base_dir=self.base)

    def test_stable_id_cannot_change_source_item_or_channel(self):
        names = dict(naming_scheme="dataset-config-v1", source_repository="Salesforce/GiftEval")
        for update in ({"source_revision": "revision-2"}, {"source_sha256": "b" * 64},
                       {"item_id": "item-2"}, {"channel": 1}):
            conflicting = row(seed=43, **update)
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, "stable series ID"):
                plan(workload([row(), conflicting]), models(), base_dir=self.base)
            with self.subTest(update=update, naming="production"), self.assertRaisesRegex(ValueError, "stable series ID"):
                plan(workload([row(dataset="ett1", config="H"),
                               dict(conflicting, dataset="ett1", config="H")], **names),
                     models(), base_dir=self.base)

    def test_rejects_heldout_aliases_duplicate_logical_rows_and_model_labels(self):
        for update in ({"dataset": "M5"}, {"source_path": "data/pjm/source.arrow"},
                       {"source_revision": "release-M5-final"}, {"series_id": "old_idx_4"}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                plan(workload([row(**update)]), models(), base_dir=self.base)
        alias = dict(row(), series_id="another-stable-name")
        with self.assertRaises(ValueError):
            plan(workload([row(), alias]), models(), base_dir=self.base)
        duplicate_models = models()
        duplicate_models["models"] *= 2
        with self.assertRaises(ValueError):
            plan(workload(), duplicate_models, base_dir=self.base)

    def test_fixture_is_explicit_and_does_not_need_approval(self):
        result = plan(workload(fixture=True), models(fixture=True), base_dir=self.base)
        self.assertTrue(result["fixture"])
        self.assertNotIn("approval_reference", result["jobs"][0])
        calendar = row(frequency="ME", start_timestamp="2020-01-31T00:00:00",
                       context_length=1000, origin=1000, horizon=3,
                       validation_end=1003, training_end=1000)
        calendar_job = plan(workload([calendar], fixture=True), models(fixture=True),
                            base_dir=self.base)["jobs"][0]
        self.assertEqual(target_timestamps(calendar_job),
                         ["2103-05-31T00:00:00", "2103-06-30T00:00:00", "2103-07-31T00:00:00"])
        with self.assertRaises(ValueError):
            plan(workload(fixture=False, approval_reference=""), models(), base_dir=self.base)

    def test_rejects_invalid_bounds_types_paths_and_calendar_alignment(self):
        cases = [
            {"channel": True}, {"source_length": 0}, {"context_length": 999},
            {"origin": 2049}, {"horizon": True}, {"usable_end": 5001},
            {"training_end": 2049}, {"validation_end": 2071},
            {"source_path": "../escape.arrow"}, {"frequency": "not-an-offset"},
            {"start_timestamp": "2020-01-01T00:00:00+00:00"},
            {"frequency": "ME", "start_timestamp": "2020-01-01T00:00:00"},
            {"frequency": "0H"}, {"start_timestamp": "NaT"},
        ]
        for update in cases:
            with self.subTest(update=update), self.assertRaises(ValueError):
                plan(workload([row(**update)]), models(), base_dir=self.base)

    def test_rejects_bad_model_headers_and_protects_row_metadata(self):
        for update in ({"batch_size": True}, {"batch_size": 0}, {"device": "tpu"},
                       {"model": ""}, {"family": 1}, {"adapter_version": None}):
            configured = models()
            configured["models"][0].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                plan(workload(), configured, base_dir=self.base)
        configured = models()
        configured["models"][0]["fixture"] = True
        with self.assertRaises(ValueError):
            plan(workload(), configured, base_dir=self.base)
        for field, value in (("naming_scheme", "other"), ("source_repository", "other/repo")):
            configured = models()
            configured["models"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "protected"):
                plan(workload(), configured, base_dir=self.base)

    def test_heldout_rejection_precedes_bad_path_validation(self):
        with self.assertRaisesRegex(ValueError, "held out"):
            plan(workload([row(dataset="PJM", source_path="/no/such.npy")]),
                 models(), base_dir=self.base)


if __name__ == "__main__":
    unittest.main()
