"""Regression against the recorded K2 tables, and the numbers the hardening note reports.

These tests run the real pipeline on the saved reference inputs (published
GIFT-Eval results and the current Omega manifest). The recorded correlations
are read from docs/results/k2/ and compared, not retyped. The reference
numbers that are typed here are the counts 97, 71, 89, 29, 20 and 28
(configurations, joined configurations, subsets, datasets), each next to an
independent recount, and the formatted p-value "0.850".
"""
import json
import os
import unittest

import numpy as np
import pandas as pd

from decision_aware_tsfm.analysis.decision.guards import is_held_out
from scipy import stats

from decision_aware_tsfm.analysis.k2.families import (
    GROUPS, METRICS, MSE_MEDIAN_FORECAST_COLUMN, family_table)
from decision_aware_tsfm.analysis.k2.loading import sha256_file
from decision_aware_tsfm.analysis.k2.pipeline import (
    DEFAULT_MANIFEST, DEFAULT_RESULTS, config_join_report, config_keys, join_omega_by_base,
    join_omega_by_config)

from ._shared import (RECORDED, REFERENCE_RESAMPLES, reference_manifest, reference_omega,
                      reference_results, reference_run)


def read(name):
    return pd.read_csv(os.path.join(reference_run()[0], name))


def lf_bytes(path):
    """File content with CRLF folded to LF, so the comparison with the recorded
    file is byte for byte on any platform and under any git line-ending setting."""
    with open(path, "rb") as f:
        return f.read().replace(b"\r\n", b"\n")


class RecordedNumbersReproduce(unittest.TestCase):
    def test_summary_stats_equal_the_recorded_file(self):
        new, old = read("summary_stats.csv"), pd.read_csv(RECORDED / "summary_stats.csv")
        pd.testing.assert_frame_equal(new, old, check_exact=True)
        self.assertEqual(lf_bytes(os.path.join(reference_run()[0], "summary_stats.csv")),
                         lf_bytes(RECORDED / "summary_stats.csv"))

    def test_within_family_sensitivity_equals_the_recorded_file(self):
        name = "within_family_sensitivity.csv"
        pd.testing.assert_frame_equal(read(name), pd.read_csv(RECORDED / name), check_exact=True)
        self.assertEqual(lf_bytes(os.path.join(reference_run()[0], name)), lf_bytes(RECORDED / name))

    def test_joined_table_values_equal_the_recorded_file(self):
        # The recorded file has the same columns in a different order, so compare by name.
        new, old = read("joined_table.csv"), pd.read_csv(RECORDED / "joined_table.csv")
        self.assertEqual(sorted(new.columns), sorted(old.columns))
        key = ["dataset", "model_family"]
        new = new.sort_values(key).reset_index(drop=True)
        old = old.sort_values(key).reset_index(drop=True)[list(new.columns)]
        pd.testing.assert_frame_equal(new, old, check_exact=True)

    def test_full_precision_table_rounds_to_the_recorded_headline(self):
        full = read("correlations_dataset_level.csv")
        head = full[(full.within_family == "mean") & (full.pooling == "rows")]
        old = pd.read_csv(RECORDED / "summary_stats.csv")
        merged = old.merge(head, on=["metric", "comparison"], suffixes=("_old", ""))
        self.assertEqual(len(merged), len(old))
        for row in merged.itertuples():
            with self.subTest(metric=row.metric, comparison=row.comparison):
                self.assertEqual(round(row.spearman_rho, 3), row.spearman_rho_old)
                self.assertEqual(round(row.p_value, 4), row.p_value_old)
                self.assertEqual(row.n, row.n_old)

    def test_recorded_sensitivity_rows_are_the_row_pooled_ones(self):
        full = read("correlations_dataset_level.csv")
        full = full[(full.metric == "sMAPE") & (full.comparison == "zero-shot vs deep")
                    & (full.pooling == "rows")].set_index("within_family")
        old = pd.read_csv(RECORDED / "within_family_sensitivity.csv")
        for agg, row in zip(("mean", "median", "min"), old.itertuples()):
            with self.subTest(within_family=agg):
                self.assertEqual(round(full.loc[agg, "spearman_rho"], 3), row.spearman_rho)
                self.assertEqual(round(full.loc[agg, "p_value"], 4), row.p_value)

    def test_median_model_p_value_rounds_to_0850_not_0851(self):
        full = read("correlations_dataset_level.csv")
        p = float(full[(full.metric == "sMAPE") & (full.comparison == "zero-shot vs deep")
                       & (full.within_family == "median") & (full.pooling == "rows")].p_value.iloc[0])
        recorded = float(pd.read_csv(RECORDED / "within_family_sensitivity.csv")
                         .set_index("within_family").loc["median model in family", "p_value"])
        self.assertEqual(round(p, 4), recorded)
        self.assertEqual(f"{p:.3f}", "0.850")


class NewTables(unittest.TestCase):
    def test_every_interval_contains_its_point_estimate_and_n_is_reported(self):
        for name in ("correlations_dataset_level.csv", "correlations_config_level.csv"):
            t = read(name)
            with self.subTest(table=name):
                self.assertFalse(t[["spearman_rho", "ci_low", "ci_high", "n"]].isna().any().any())
                self.assertTrue((t.ci_low <= t.spearman_rho).all())
                self.assertTrue((t.spearman_rho <= t.ci_high).all())
                self.assertTrue((t.ci_low >= -1).all() and (t.ci_high <= 1).all())
                self.assertTrue((t.n_resamples == REFERENCE_RESAMPLES).all())
                self.assertTrue((t.n_resampled_units == 20).all())

    def test_sensitivity_covers_every_metric_comparison_and_summary(self):
        t = read("correlations_dataset_level.csv")
        self.assertEqual(len(t), 4 * 3 * 5)
        self.assertEqual(len(t.drop_duplicates(["metric", "comparison", "within_family", "pooling"])),
                         len(t))
        self.assertTrue((t.n == 20).all())

    def test_family_mean_does_not_depend_on_pooling_here(self):
        results, _ = reference_results()
        for mname, mcol in METRICS.items():
            with self.subTest(metric=mname):
                rows = family_table(results, mcol, "mean", pooling="rows")
                models = family_table(results, mcol, "mean", pooling="models")
                np.testing.assert_allclose(rows.to_numpy(), models.to_numpy(), rtol=1e-9)

    def test_sign_counts_agree_with_the_saved_deltas(self):
        signs = read("sign_counts.csv")
        self.assertEqual(len(signs), 4 * 3 * 5)
        self.assertTrue((signs.n_zero_shot_lower + signs.n_other_lower + signs.n_tie == signs.n).all())
        deltas = read("dataset_deltas.csv")
        counted = (deltas.assign(lower=deltas.delta < 0)
                   .groupby(["metric", "comparison"]).lower.sum())
        mean_rows = signs[(signs.within_family == "mean") & (signs.pooling == "rows")]
        for row in mean_rows.itertuples():
            with self.subTest(metric=row.metric, comparison=row.comparison):
                self.assertEqual(row.n_zero_shot_lower, counted[(row.metric, row.comparison)])
                self.assertEqual(row.n, 20)


def independent_rho(metric_col, other, without=()):
    """Family-mean correlation recomputed with plain pandas and scipy, no package tables.

    Returns (rho, number of datasets where the zero-shot family mean is lower).
    """
    raw = pd.read_csv(DEFAULT_RESULTS)
    raw["base"] = raw.dataset.str.split("/").str[0]
    series = reference_manifest().series
    omega = series.groupby("base").omega_levels.mean()
    means = {}
    for family in ("zero-shot", other):
        models = [m for m in GROUPS[family] if m not in without]
        means[family] = raw[raw.model.isin(models)].groupby("base")[metric_col].mean()
    t = pd.DataFrame(means).join(omega.rename("omega"), how="inner")
    delta = 100.0 * (t["zero-shot"] - t[other]) / t["zero-shot"]
    return float(stats.spearmanr(t.omega, delta)[0]), int((t["zero-shot"] < t[other]).sum())


class SensitivityTables(unittest.TestCase):
    """The MSE sensitivity, leave-one-model-out and paper tables on the reference inputs.

    Nothing is typed in: each value is compared with another table of the same
    run or with the independent recomputation above.
    """
    KEYS = ["spearman_rho", "p_value", "ci_low", "ci_high", "n"]

    def headline(self, metric):
        full = read("correlations_dataset_level.csv")
        return full[(full.metric == metric) & (full.within_family == "mean")
                    & (full.pooling == "rows")].set_index("comparison")

    def test_recorded_and_mae_rows_equal_the_headline_rows(self):
        mse = read("mse_sensitivity.csv")
        signs = read("sign_counts.csv")
        for variant, metric in (("as_recorded", "MSE"), ("mae_for_comparison", "MAE")):
            rows = mse[mse.variant == variant].set_index("comparison")
            head = self.headline(metric)
            count = signs[(signs.metric == metric) & (signs.within_family == "mean")
                          & (signs.pooling == "rows")].set_index("comparison")
            self.assertEqual(len(rows), 3)
            for comparison in rows.index:
                with self.subTest(variant=variant, comparison=comparison):
                    for key in self.KEYS:
                        self.assertEqual(rows.loc[comparison, key], head.loc[comparison, key])
                    self.assertEqual(rows.loc[comparison, "n_zero_shot_lower"],
                                     count.loc[comparison, "n_zero_shot_lower"])
        recorded = pd.read_csv(RECORDED / "summary_stats.csv")
        recorded = recorded[recorded.metric == "MSE"].set_index("comparison")
        rows = mse[mse.variant == "as_recorded"].set_index("comparison")
        for comparison in rows.index:
            self.assertEqual(round(rows.loc[comparison, "spearman_rho"], 3),
                             recorded.loc[comparison, "spearman_rho"])

    def test_other_variants_equal_an_independent_recomputation(self):
        mse = pd.read_csv(os.path.join(reference_run()[0], "mse_sensitivity.csv"),
                          keep_default_na=False)
        removed = set(mse.model_removed) - {""}
        self.assertEqual(len(removed), 1)
        (dominant,) = removed
        self.assertEqual(reference_run()[1]["dominant_model"]["model"], dominant)
        n_present = reference_run()[1]["roster"]["families"]["zero-shot"]["n_present"]
        cases = {"dominant_model_removed": (METRICS["MSE"], (dominant,), n_present - 1),
                 "median_forecast_column": (MSE_MEDIAN_FORECAST_COLUMN, (), n_present),
                 "median_forecast_column_dominant_model_removed":
                     (MSE_MEDIAN_FORECAST_COLUMN, (dominant,), n_present - 1)}
        for variant, (column, without, n_models) in cases.items():
            for other in ("deep", "statistical", "pretrained"):
                with self.subTest(variant=variant, other=other):
                    row = mse[(mse.variant == variant)
                              & (mse.comparison == f"zero-shot vs {other}")].iloc[0]
                    rho, lower = independent_rho(column, other, without)
                    self.assertAlmostEqual(row.spearman_rho, rho, places=12)
                    self.assertEqual(row.n_zero_shot_lower, lower)
                    self.assertEqual((row.metric_column, row.n_zero_shot_models), (column, n_models))
                    self.assertTrue(row.ci_low <= row.spearman_rho <= row.ci_high)

    def test_dominant_model_is_the_largest_share_in_the_diagnostics(self):
        diag = read("model_score_diagnostics.csv")
        z = diag[(diag.metric == "MSE") & (diag.family == "zero-shot")]
        top = z.sort_values("median_share_of_family_sum", ascending=False).iloc[0]
        self.assertEqual(top.model, reference_run()[1]["dominant_model"]["model"])
        # Independent recount of the share and of "highest on how many datasets".
        results, _ = reference_results()
        bases = set(reference_omega().by_base.base)
        r = results[results.base.isin(bases) & results.model.isin(GROUPS["zero-shot"])]
        per_model = r.groupby(["base", "model"])[METRICS["MSE"]].mean().unstack("model")
        share = per_model[top.model] / per_model.sum(axis=1)
        self.assertAlmostEqual(top.median_share_of_family_sum, float(share.median()), places=12)
        self.assertEqual(top.n_datasets_highest_in_family,
                         int((per_model.idxmax(axis=1) == top.model).sum()))
        self.assertEqual(top.n_models_in_family, per_model.shape[1])
        self.assertGreater(top.median_share_of_family_sum, 5 * top.even_share)

    def test_leave_one_model_out_rows(self):
        lomo = read("leave_one_model_out.csv")
        roster = reference_run()[1]["roster"]["families"]
        n_zero_shot = roster["zero-shot"]["n_present"]
        n_others = sum(roster[f]["n_present"] for f in ("deep", "statistical", "pretrained"))
        self.assertEqual(len(lomo), 5 * (3 * n_zero_shot + n_others))
        self.assertEqual(lomo.n.min(), lomo.n.max())
        for metric in METRICS:
            head = self.headline(metric)
            rows = lomo[lomo.metric == metric]
            for comparison in head.index:
                with self.subTest(metric=metric, comparison=comparison):
                    sub = rows[rows.comparison == comparison]
                    self.assertEqual(set(sub.rho_all_models), {head.loc[comparison, "spearman_rho"]})
                    self.assertEqual(set(sub.ci_low_all_models), {head.loc[comparison, "ci_low"]})
        np.testing.assert_allclose(lomo.rho_change, lomo.spearman_rho - lomo.rho_all_models, atol=1e-15)
        # The dominant-model row is the same computation as the MSE sensitivity table.
        mse = pd.read_csv(os.path.join(reference_run()[0], "mse_sensitivity.csv"),
                          keep_default_na=False)
        removed = mse[mse.variant == "dominant_model_removed"].set_index("comparison")
        dominant = removed.model_removed.iloc[0]
        for comparison in removed.index:
            row = lomo[(lomo.metric == "MSE") & (lomo.comparison == comparison)
                       & (lomo.model_removed == dominant)].iloc[0]
            for key in self.KEYS + ["n_zero_shot_lower"]:
                self.assertEqual(row[key], removed.loc[comparison, key])

    def test_leave_one_model_out_equals_an_independent_recomputation(self):
        lomo = read("leave_one_model_out.csv")
        # The largest change in each (metric, comparison, family) group, and two fixed rows.
        idx = lomo.groupby(["metric", "comparison", "family_of_removed_model"]).rho_change.apply(
            lambda s: s.abs().idxmax())
        picked = lomo.loc[sorted(set(idx) | {0, len(lomo) - 1})]
        self.assertGreaterEqual(len(picked), 30)
        for row in picked.itertuples():
            with self.subTest(metric=row.metric, comparison=row.comparison, model=row.model_removed):
                rho, lower = independent_rho(row.metric_column, row.comparison.split(" vs ")[1],
                                             (row.model_removed,))
                self.assertAlmostEqual(row.spearman_rho, rho, places=12)
                self.assertEqual(row.n_zero_shot_lower, lower)

    def test_summary_is_derived_from_the_rows(self):
        lomo = read("leave_one_model_out.csv")
        summary = read("leave_one_model_out_summary.csv")
        self.assertEqual(len(summary), 5 * 6)
        for row in summary.itertuples():
            with self.subTest(metric=row.metric, comparison=row.comparison,
                              family=row.family_of_removed_model):
                g = lomo[(lomo.metric == row.metric) & (lomo.comparison == row.comparison)
                         & (lomo.family_of_removed_model == row.family_of_removed_model)]
                self.assertEqual((row.rho_min, row.rho_max), (g.spearman_rho.min(), g.spearman_rho.max()))
                self.assertEqual(row.model_at_rho_max,
                                 g.loc[g.spearman_rho.idxmax()].model_removed)
                self.assertAlmostEqual(row.max_abs_rho_change, g.rho_change.abs().max(), places=15)
                self.assertEqual(row.n_sign_changes,
                                 int((np.sign(g.spearman_rho) != np.sign(row.rho_all_models)).sum()))
                full_excludes = row.ci_low_all_models > 0 or row.ci_high_all_models < 0
                self.assertEqual(row.n_interval_status_changes,
                                 int((((g.ci_low > 0) | (g.ci_high < 0)) != full_excludes).sum()))
                self.assertEqual((row.n_zero_shot_lower_min, row.n_zero_shot_lower_max),
                                 (g.n_zero_shot_lower.min(), g.n_zero_shot_lower.max()))
                self.assertEqual(row.n_models_removed_in_turn, len(g))

    def test_paper_table_carries_the_smape_rows(self):
        paper = read("paper_like_for_like.csv").set_index("comparison")
        smape, mse = self.headline("sMAPE"), self.headline("MSE")
        for comparison in paper.index:
            with self.subTest(comparison=comparison):
                row = paper.loc[comparison]
                self.assertEqual((row.paper_metric, row.replication_metric), ("sMAPE", "sMAPE"))
                self.assertEqual(row.replication_rho, smape.loc[comparison, "spearman_rho"])
                self.assertEqual((row.ci_low, row.ci_high),
                                 (smape.loc[comparison, "ci_low"], smape.loc[comparison, "ci_high"]))
                self.assertEqual(row.not_like_for_like_mse_mean_rho,
                                 mse.loc[comparison, "spearman_rho"])
                self.assertEqual(row.n, smape.loc[comparison, "n"])

    def test_relabelled_table_has_the_recorded_numbers_under_accurate_labels(self):
        new = read("within_family_sensitivity_relabelled.csv")
        old = pd.read_csv(RECORDED / "within_family_sensitivity.csv")
        rows = new[new.pooling == "rows"].reset_index(drop=True)
        self.assertEqual(list(rows.label_in_recorded_table), list(old.within_family))
        self.assertEqual(list(rows.spearman_rho), list(old.spearman_rho))
        self.assertEqual(list(rows.p_value), list(old.p_value))
        self.assertEqual(list(rows.n), list(old.n))
        self.assertFalse(new.within_family.str.contains("paper's choice").any())
        self.assertEqual(len(new), 5)

    def test_no_degenerate_warning_and_manifest_records_on_the_reference_run(self):
        manifest = reference_run()[1]
        self.assertEqual(manifest["degenerate_resample_warnings"], [])
        self.assertFalse(manifest["manifest_file"]["frozen"])
        self.assertEqual(manifest["manifest_file"]["layout_version"], "dev-v1")
        self.assertEqual(manifest["held_out"]["manifest_fields_checked"],
                         ["series_id", "subset", "status"])
        with open(DEFAULT_MANIFEST, encoding="utf-8") as f:
            listed = json.load(f)["held_out_strict"]
        self.assertEqual(manifest["held_out"]["manifest_held_out_strict"], listed)
        self.assertEqual(manifest["held_out"]["results_exclusions"], [])
        self.assertEqual(manifest["held_out"]["results_fuzzy_reports"], [])
        self.assertEqual(manifest["held_out"]["manifest_fuzzy_reports"], [])
        self.assertEqual(manifest["omega_report"]["n_series_missing_selected_omega"], 0)


class ConfigurationLevel(unittest.TestCase):
    """The join counts the hardening note reports, recomputed independently of the pipeline."""

    def test_join_counts(self):
        results, _ = reference_results()
        series = reference_manifest().series
        configs = results[["dataset", "base", "freq"]].drop_duplicates()
        self.assertEqual(len(configs), 97)
        have_subset = set(zip(series.base, series.freq))
        have_base = set(series.base)
        n_by_frequency = sum((b, f) in have_subset for b, f in zip(configs.base, configs.freq))
        n_by_base = int(configs.base.isin(have_base).sum())
        self.assertEqual(len(have_subset), 29)
        self.assertEqual(len(have_base), 20)

        report = config_join_report(results, reference_omega())
        self.assertEqual(report["n_configs_in_results"], 97)
        by_freq = report["matchings"]["dataset_frequency"]
        by_base = report["matchings"]["dataset_broadcast"]
        self.assertEqual(by_freq["n_configs_joined"], n_by_frequency)
        self.assertEqual(by_base["n_configs_joined"], n_by_base)
        self.assertEqual((n_by_frequency, n_by_base), (71, 89))
        self.assertEqual(by_freq["n_base_datasets_joined"], 20)
        self.assertEqual(by_base["n_base_datasets_joined"], 20)
        self.assertEqual(len(by_freq["configs_not_joined"]), 97 - 71)
        self.assertEqual(len(by_base["configs_not_joined"]), 97 - 89)
        self.assertEqual(report, reference_run()[1]["config_join_report"])

    def test_correlation_rows_use_the_joined_counts(self):
        t = read("correlations_config_level.csv")
        self.assertEqual(len(t), 2 * 4 * 3 * 3)
        self.assertEqual(set(t[t.matching == "dataset_frequency"].n), {71})
        self.assertEqual(set(t[t.matching == "dataset_broadcast"].n), {89})

    def test_frequency_matched_omega_is_the_subset_mean(self):
        results, _ = reference_results()
        omega = reference_omega()
        keys = config_keys(results)
        t = join_omega_by_config(family_table(results, METRICS["sMAPE"], unit="config"),
                                 keys, omega, "dataset_frequency")
        series = reference_manifest().series
        expected = series[(series.base == "electricity") & (series.freq == "15T")].omega_levels.mean()
        got = t[(t.base == "electricity") & (t.freq == "15T")].omega
        self.assertEqual(len(got), 3)                      # short, medium, long
        np.testing.assert_allclose(got.to_numpy(), expected)
        self.assertEqual(len(t[(t.base == "electricity") & (t.freq == "W")]), 0)   # no Omega for it

    def test_dataset_level_join_keeps_twenty_datasets(self):
        results, _ = reference_results()
        t = join_omega_by_base(family_table(results, METRICS["sMAPE"]), reference_omega())
        self.assertEqual(len(t), 20)
        self.assertEqual(results.base.nunique(), 28)


class RunManifest(unittest.TestCase):
    def setUp(self):
        self.out, self.manifest = reference_run()

    def test_input_hashes_and_versions_are_recorded(self):
        with open(os.path.join(self.out, "run_manifest.json"), encoding="utf-8") as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk["inputs"]["results"]["sha256"], sha256_file(DEFAULT_RESULTS))
        self.assertEqual(on_disk["inputs"]["manifest"]["sha256"], sha256_file(DEFAULT_MANIFEST))
        self.assertEqual(on_disk["inputs"]["results"]["path"], "artifacts/reference/k2/gift_results.csv")
        env = on_disk["environment"]
        self.assertEqual(set(env["packages"]), {"numpy", "pandas", "pyarrow", "scipy"})
        self.assertEqual(env["packages"]["numpy"], np.__version__)
        self.assertEqual(env["packages"]["pandas"], pd.__version__)

    def test_held_out_count_in_the_reference_results_is_recorded(self):
        raw = pd.read_csv(DEFAULT_RESULTS, usecols=["dataset"])
        n = int(raw.dataset.map(is_held_out).sum())
        self.assertEqual(self.manifest["held_out"]["results_rows_excluded"], n)
        self.assertEqual(self.manifest["results_report"]["n_rows_used"], len(raw) - n)
        self.assertEqual(self.manifest["held_out"]["manifest_rows_found"], 0)

    def test_roster_and_manifest_reports(self):
        roster = self.manifest["roster"]
        self.assertEqual(sum(f["n_mapped"] for f in roster["families"].values()),
                         len(self.manifest["model_to_family"]))
        self.assertEqual(roster["n_models_in_results"],
                         roster["n_models_ignored"]
                         + sum(f["n_present"] for f in roster["families"].values()))
        report = self.manifest["manifest_report"]
        self.assertEqual(report["unmapped_subsets_not_used"], {})
        self.assertEqual(list(report["known_duplicate_subsets_not_used"]), ["jena_weather"])
        self.assertEqual(report["n_series_used"]
                         + report["known_duplicate_subsets_not_used"]["jena_weather"]["n_series"],
                         report["n_active_series_read"])


if __name__ == "__main__":
    unittest.main()
