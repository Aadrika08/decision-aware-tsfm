"""Value-level tests of every output table, on tiny hand-built inputs with known answers.

The inputs are made up; the expected values are worked by hand in tests/unit/k2/_shared.py
and in the comments here. These are mechanism checks and say nothing about real model families.
"""
import math
import os
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from decision_aware_tsfm.analysis.k2 import pipeline, sensitivity
from decision_aware_tsfm.analysis.k2.families import (
    METRICS, MSE_MEDIAN_FORECAST_COLUMN, SUMMARY_LABELS, WITHIN_FAMILY_LABELS, family_table)
from decision_aware_tsfm.analysis.k2.loading import _from_dev_v1, aggregate_omega, prepare_results
from decision_aware_tsfm.analysis.k2.statistic import bootstrap_interval

from ._shared import (DOMINANT_DEEP, DOMINANT_MSE_MEAN, POOLED_RHO, POOLED_SCORE,
                      POOLED_SIGNS_VS_DEEP, SECOND_OMEGA, dominant_results,
                      pooled_results, run_cli, scratch_dir, tiny_manifest, tiny_omega,
                      tiny_results, write_tiny_inputs)

BASES = ["ett1", "ett2", "solar", "electricity", "us_births"]       # Omega 0.1 ... 0.5
# The tables hold base datasets in alphabetical order, and a bootstrap draw refers to a
# position, so a hand-built bootstrap has to use the same order.
ALPHABETICAL = np.argsort(BASES)
SUMMARIES = list(POOLED_RHO)
DEEP = "zero-shot vs deep"


def prepared(raw):
    return prepare_results(raw)[0]


def pick(table, **where):
    for column, value in where.items():
        table = table[table[column] == value]
    return table


def one(table, **where):
    rows = pick(table, **where)
    assert len(rows) == 1, (where, len(rows))
    return rows.iloc[0]


class PooledFixture(unittest.TestCase):
    """The hand-worked family scores really are what the fixture gives."""

    def test_family_scores_match_the_hand_calculation(self):
        results = prepared(pooled_results())
        for (agg, pooling), expected in POOLED_SCORE.items():
            with self.subTest(summary=agg, pooling=pooling):
                t = family_table(results, METRICS["sMAPE"], agg, pooling=pooling)
                np.testing.assert_allclose(t.loc[BASES, "zero-shot"].to_numpy(), expected)
                np.testing.assert_allclose(t.loc[BASES, "deep"].to_numpy(), 10.0)


class DatasetLevelTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = prepared(pooled_results())
        cls.omega = tiny_omega()
        cls.corr = pipeline.dataset_level_correlations(cls.results, cls.omega, 300, 5)
        cls.signs = pipeline.dataset_sign_counts(cls.results, cls.omega)

    def test_every_summary_column_has_its_own_known_rho(self):
        # Distinct answers per column, so a column that computes another summary is seen.
        self.assertEqual(len(set(POOLED_RHO.values())), 5)
        for metric in METRICS:
            for (agg, pooling), rho in POOLED_RHO.items():
                with self.subTest(metric=metric, summary=agg, pooling=pooling):
                    row = one(self.corr, metric=metric, comparison=DEEP, within_family=agg,
                              pooling=pooling)
                    self.assertAlmostEqual(row.spearman_rho, rho, places=12)
                    self.assertEqual(row.n, 5)
                    self.assertEqual(row.n_resampled_units, 5)

    def test_interval_is_the_bootstrap_of_the_same_points_with_the_given_seed(self):
        omega = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        for (agg, pooling), scores in POOLED_SCORE.items():
            with self.subTest(summary=agg, pooling=pooling):
                scores = np.asarray(scores, dtype=float)
                delta = 100.0 * (scores - 10.0) / scores
                expected = bootstrap_interval(omega[ALPHABETICAL], delta[ALPHABETICAL],
                                              n_resamples=300, seed=5)
                row = one(self.corr, metric="sMAPE", comparison=DEEP, within_family=agg,
                          pooling=pooling)
                self.assertEqual((row.ci_low, row.ci_high, row.n_degenerate, row.seed),
                                 (expected["ci_low"], expected["ci_high"],
                                  expected["n_degenerate"], 5))

    def test_sign_counts_known_answers_for_both_poolings(self):
        self.assertNotEqual(POOLED_SIGNS_VS_DEEP[("median", "rows")],
                            POOLED_SIGNS_VS_DEEP[("median", "models")])
        for (agg, pooling), (lower, higher, tie) in POOLED_SIGNS_VS_DEEP.items():
            with self.subTest(summary=agg, pooling=pooling):
                row = one(self.signs, metric="MAE", comparison=DEEP, within_family=agg,
                          pooling=pooling)
                self.assertEqual((row.n, row.n_zero_shot_lower, row.n_other_lower, row.n_tie),
                                 (5, lower, higher, tie))
        # Versus the statistical family (constant 20) the zero-shot family is always lower.
        for agg, pooling in SUMMARIES:
            row = one(self.signs, metric="MAE", comparison="zero-shot vs statistical",
                      within_family=agg, pooling=pooling)
            self.assertEqual((row.n_zero_shot_lower, row.n_other_lower, row.n_tie), (5, 0, 0))

    def test_legacy_tables_on_the_fixture(self):
        summary = pipeline.summary_stats(self.results, self.omega)
        self.assertEqual(len(summary), 12)
        for row in summary.itertuples():
            self.assertEqual((row.n, row.spearman_rho), (5, -0.1))
        recorded = pipeline.recorded_within_family_sensitivity(self.results, self.omega)
        self.assertEqual(list(recorded.within_family), list(WITHIN_FAMILY_LABELS.values()))
        self.assertEqual(list(recorded.spearman_rho), [-0.1, 0.5, 1.0])     # rows pooling

    def test_relabelled_sensitivity_table(self):
        t = sensitivity.within_family_relabelled(self.results, self.omega)
        self.assertEqual(list(t.within_family), list(SUMMARY_LABELS.values()))
        self.assertEqual(list(zip(t.summary, t.pooling)), SUMMARIES)
        self.assertEqual(list(t.spearman_rho), [POOLED_RHO[k] for k in SUMMARIES])
        self.assertTrue((t.n == 5).all())
        self.assertEqual(list(t.label_in_recorded_table),
                         list(WITHIN_FAMILY_LABELS.values()) + ["", ""])
        for label in t.within_family:
            self.assertNotIn("paper's choice", label)
        self.assertIn("the paper does not state one", t.within_family.iloc[0])
        self.assertIn("model-configuration row", t.within_family.iloc[1])
        self.assertIn("model-configuration row", t.within_family.iloc[2])


class DeltasTable(unittest.TestCase):
    def test_delta_is_relative_to_the_zero_shot_score(self):
        # tiny_results: zero-shot 1.0; deep 0.9, 0.8, ..., 0.5; statistical 2, 3, ..., 6.
        t = pipeline.dataset_deltas(prepared(tiny_results()), tiny_omega())
        self.assertEqual(len(t), 4 * 3 * 5)
        deep = pick(t, metric="sMAPE", comparison=DEEP).set_index("dataset").loc[BASES]
        np.testing.assert_allclose(deep.delta, [10, 20, 30, 40, 50])          # 100 (1 - d) / 1
        np.testing.assert_allclose(deep.omega, [0.1, 0.2, 0.3, 0.4, 0.5])
        np.testing.assert_allclose(deep.zero_shot_score, 1.0)
        np.testing.assert_allclose(deep.other_score, [0.9, 0.8, 0.7, 0.6, 0.5])
        stat = pick(t, metric="CRPS", comparison="zero-shot vs statistical")
        np.testing.assert_allclose(stat.set_index("dataset").loc[BASES].delta,
                                   [-100, -200, -300, -400, -500])            # 100 (1 - s) / 1

    def test_delta_when_the_zero_shot_score_is_not_one(self):
        # pooled fixture, family mean: zero-shot 10, 11, 8, 12, 9.5 against statistical 20.
        t = pipeline.dataset_deltas(prepared(pooled_results()), tiny_omega())
        got = pick(t, metric="MSE", comparison="zero-shot vs statistical").set_index("dataset")
        expected = [100 * (z - 20) / z for z in (10, 11, 8, 12, 9.5)]
        np.testing.assert_allclose(got.loc[BASES].delta, expected)
        self.assertAlmostEqual(got.loc["solar", "delta"], -150.0)


class ConfigurationLevelTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # ett1 gets a second frequency subset (15T, two series) and a configuration on it.
        extra_series = [
            {"series_id": "ett1_15T_idx_0", "subset": "ett1_15T", "channel": 0, "length": 3000,
             "omega_levels": 0.5, "omega_diffs": 0.05, SECOND_OMEGA: 0.2},
            {"series_id": "ett1_15T_idx_1", "subset": "ett1_15T", "channel": 0, "length": 3000,
             "omega_levels": 0.9, "omega_diffs": 0.05, SECOND_OMEGA: 0.7}]
        cls.manifest = _from_dev_v1(tiny_manifest(extra_series))
        raw = pooled_results()
        extra_config = raw[raw.dataset == "ett1/H/short"].assign(dataset="ett1/15T/short")
        cls.results = prepared(pd.concat([raw, extra_config], ignore_index=True))

    def test_omega_tables_follow_the_selected_column(self):
        levels = aggregate_omega(self.manifest, "omega_levels")
        second = aggregate_omega(self.manifest, SECOND_OMEGA)
        for om, by_freq, by_base, spread in (
                (levels, {"15T": 0.7, "H": 0.1}, 0.5, np.std([0.1, 0.5, 0.9], ddof=1)),
                (second, {"15T": 0.45, "H": 0.9}, 0.6, np.std([0.9, 0.2, 0.7], ddof=1))):
            with self.subTest(column=om.omega_col):
                bf = om.by_base_freq.set_index(["base", "freq"]).omega
                self.assertAlmostEqual(bf[("ett1", "15T")], by_freq["15T"])
                self.assertAlmostEqual(bf[("ett1", "H")], by_freq["H"])
                ett1 = om.by_base.set_index("base").loc["ett1"]
                self.assertAlmostEqual(ett1[om.omega_col], by_base)
                self.assertAlmostEqual(ett1.omega_sd, spread)
                self.assertEqual(ett1.n_series, 3)
        self.assertAlmostEqual(np.std([0.9, 0.2, 0.7], ddof=1), math.sqrt(0.13))

    def test_joined_config_table_holds_both_omega_values_per_configuration(self):
        for column, by_freq, broadcast, solar in (("omega_levels", {"15T": 0.7, "H": 0.1}, 0.5, 0.3),
                                                  (SECOND_OMEGA, {"15T": 0.45, "H": 0.9}, 0.6, 0.7)):
            with self.subTest(column=column):
                t = pipeline.joined_config_table(self.results, aggregate_omega(self.manifest, column))
                self.assertEqual(len(t), 11 * 4)                 # 11 configurations, 4 families
                for config, freq in (("ett1/15T/short", "15T"), ("ett1/H/short", "H"),
                                     ("ett1/H/long", "H")):
                    rows = pick(t, config=config)
                    np.testing.assert_allclose(rows.omega_dataset_frequency, by_freq[freq])
                    np.testing.assert_allclose(rows.omega_dataset_broadcast, broadcast)
                rows = pick(t, config="solar/H/long")
                np.testing.assert_allclose(rows.omega_dataset_frequency, solar)
                np.testing.assert_allclose(rows.omega_dataset_broadcast, solar)
                zero_shot = one(t, config="ett1/H/long", model_family="zero-shot")
                self.assertAlmostEqual(zero_shot.sMAPE, (19 + 5 + 28) / 3)

    def expected_config_rho(self, column, matching, agg):
        """Recomputed here from the fixture definition, without the package's tables."""
        series = self.manifest.series
        raw = pooled_results()
        raw = pd.concat([raw, raw[raw.dataset == "ett1/H/short"].assign(dataset="ett1/15T/short")])
        xs, ys, clusters = [], [], []
        for config in sorted(raw.dataset.unique()):
            base, freq, _ = config.split("/")
            rows = series[series.base == base]
            if matching == "dataset_frequency":
                rows = rows[rows.freq == freq]
            scores = raw[(raw.dataset == config)
                         & raw.model.isin(["TiRex", "Moirai_small", "VisionTS"])][METRICS["sMAPE"]]
            z = {"mean": scores.mean(), "median": scores.median(), "min": scores.min()}[agg]
            xs.append(rows[column].mean())
            ys.append(100.0 * (z - 10.0) / z)
            clusters.append(base)
        return np.array(xs), np.array(ys), np.array(clusters)

    def test_configuration_level_values_follow_column_matching_and_seed(self):
        for column in ("omega_levels", SECOND_OMEGA):
            omega = aggregate_omega(self.manifest, column)
            table = pipeline.config_level_correlations(self.results, omega, 200, 3)
            other_seed = pipeline.config_level_correlations(self.results, omega, 200, 4)
            self.assertEqual(len(table), 2 * 4 * 3 * 3)
            for matching in ("dataset_frequency", "dataset_broadcast"):
                for agg in ("mean", "median", "min"):
                    with self.subTest(column=column, matching=matching, summary=agg):
                        x, y, clusters = self.expected_config_rho(column, matching, agg)
                        row = one(table, matching=matching, metric="sMAPE", comparison=DEEP,
                                  within_family=agg)
                        self.assertEqual(row.n, 11)
                        self.assertEqual(row.n_resampled_units, 5)
                        self.assertAlmostEqual(row.spearman_rho, stats.spearmanr(x, y)[0], places=12)
                        ci = bootstrap_interval(x, y, n_resamples=200, seed=3, clusters=clusters)
                        self.assertAlmostEqual(row.ci_low, ci["ci_low"], places=12)
                        self.assertAlmostEqual(row.ci_high, ci["ci_high"], places=12)
                        self.assertEqual(row.seed, 3)
                        other = one(other_seed, matching=matching, metric="sMAPE", comparison=DEEP,
                                    within_family=agg)
                        ci4 = bootstrap_interval(x, y, n_resamples=200, seed=4, clusters=clusters)
                        self.assertAlmostEqual(other.ci_low, ci4["ci_low"], places=12)
                        self.assertAlmostEqual(other.ci_high, ci4["ci_high"], places=12)
                        self.assertEqual(other.seed, 4)
            # The two seeds give different intervals somewhere, so the seed is really used.
            self.assertFalse(np.allclose(table.ci_low, other_seed.ci_low)
                             and np.allclose(table.ci_high, other_seed.ci_high))

    def test_the_two_columns_give_different_frequency_matched_correlations(self):
        rho = {}
        for column in ("omega_levels", SECOND_OMEGA):
            table = pipeline.config_level_correlations(
                self.results, aggregate_omega(self.manifest, column), 50, 1)
            rho[column] = one(table, matching="dataset_frequency", metric="sMAPE",
                              comparison=DEEP, within_family="mean").spearman_rho
        self.assertGreater(abs(rho["omega_levels"] - rho[SECOND_OMEGA]), 0.3)


class DominantModelFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = prepared(dominant_results())
        cls.omega = tiny_omega()
        cls.diag = sensitivity.model_score_diagnostics(cls.results, cls.omega)
        cls.mse = sensitivity.mse_sensitivity(cls.results, cls.omega, "VisionTS", 300, 5)
        cls.lomo = sensitivity.leave_one_model_out(cls.results, cls.omega, 300, 5)
        cls.summary = sensitivity.leave_one_model_out_summary(cls.lomo)

    # ---- diagnostics -----------------------------------------------------
    def test_share_of_the_family_sum(self):
        vision = one(self.diag, metric="MSE", model="VisionTS")
        # v / (2 + 4 + v) for v = 6, 24, 9, 114, 18 is 0.5, 0.8, 0.6, 0.95, 0.75: median 0.75.
        self.assertAlmostEqual(vision.median_share_of_family_sum, 0.75)
        self.assertAlmostEqual(vision.even_share, 1 / 3)
        self.assertEqual((vision.n_models_in_family, vision.n_datasets,
                          vision.n_datasets_highest_in_family), (3, 5, 5))
        # v / median(2, 4, v) = v / 4 = 1.5, 6, 2.25, 28.5, 4.5: median 4.5.
        self.assertAlmostEqual(vision.median_ratio_to_family_median_model, 4.5)
        self.assertAlmostEqual(vision.median_rmse_over_mae, math.sqrt(18) / 3)
        self.assertAlmostEqual(vision.median_mse_mean_over_mse_median, 6.0)      # 18 / 3
        tirex = one(self.diag, metric="MSE", model="TiRex")
        # 2 / (6 + v) = 1/6, 1/15, 2/15, 1/60, 1/12: median 1/12.
        self.assertAlmostEqual(tirex.median_share_of_family_sum, 1 / 12)
        self.assertEqual(tirex.n_datasets_highest_in_family, 0)
        self.assertAlmostEqual(tirex.median_mse_mean_over_mse_median, 1.0)
        # On the MSE[0.5] column the shares are 2/9, 4/9, 3/9 on every dataset.
        for model, share in (("TiRex", 2 / 9), ("Moirai_small", 4 / 9), ("VisionTS", 3 / 9)):
            row = one(self.diag, metric="MSE[0.5]", model=model)
            self.assertAlmostEqual(row.median_share_of_family_sum, share)
        single = one(self.diag, metric="sMAPE", model="PatchTST")
        self.assertEqual((single.family, single.median_share_of_family_sum), ("deep", 1.0))

    def test_dominant_model_is_the_largest_median_share(self):
        self.assertEqual(sensitivity.dominant_model(self.diag), "VisionTS")
        self.assertEqual(sensitivity.dominant_model(self.diag, metric="MSE[0.5]"), "Moirai_small")
        self.assertEqual(sensitivity.dominant_model(self.diag, family="deep"), "PatchTST")
        with self.assertRaises(ValueError):
            sensitivity.dominant_model(self.diag, metric="no such metric")

    # ---- MSE sensitivity ---------------------------------------------------
    def test_mse_sensitivity_known_answers(self):
        self.assertEqual(len(self.mse), 5 * 3)
        expected = {   # variant: (rho, zero-shot lower, column, model removed, zero-shot models)
            "as_recorded": (-0.1, 0, METRICS["MSE"], "", 3),
            "dominant_model_removed": (-1.0, 3, METRICS["MSE"], "VisionTS", 2),
            "median_forecast_column": (-1.0, 3, MSE_MEDIAN_FORECAST_COLUMN, "", 3),
            "median_forecast_column_dominant_model_removed":
                (-1.0, 3, MSE_MEDIAN_FORECAST_COLUMN, "VisionTS", 2),
            "mae_for_comparison": (-1.0, 3, METRICS["MAE"], "", 3)}
        for variant, (rho, lower, column, removed, n_models) in expected.items():
            with self.subTest(variant=variant):
                row = one(self.mse, variant=variant, comparison=DEEP)
                self.assertAlmostEqual(row.spearman_rho, rho, places=12)
                self.assertEqual((row.n, row.n_zero_shot_lower, row.n_other_lower, row.n_tie),
                                 (5, lower, 5 - lower, 0))
                self.assertEqual((row.metric_column, row.model_removed, row.n_zero_shot_models),
                                 (column, removed, n_models))

    def test_mse_sensitivity_intervals_are_the_bootstrap_of_the_hand_worked_points(self):
        omega = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        deep = np.array(DOMINANT_DEEP)
        recorded_mean = (6.0 + np.array(DOMINANT_MSE_MEAN)) / 3.0            # 4, 10, 5, 40, 8
        np.testing.assert_allclose(recorded_mean, [4, 10, 5, 40, 8])
        delta = 100.0 * (recorded_mean - deep) / recorded_mean
        np.testing.assert_allclose(delta, [50, 75, 30, 90, 31.25])
        expected = bootstrap_interval(omega[ALPHABETICAL], delta[ALPHABETICAL],
                                      n_resamples=300, seed=5)
        row = one(self.mse, variant="as_recorded", comparison=DEEP)
        self.assertEqual((row.ci_low, row.ci_high, row.n_degenerate),
                         (expected["ci_low"], expected["ci_high"], expected["n_degenerate"]))
        removed = one(self.mse, variant="dominant_model_removed", comparison=DEEP)
        self.assertEqual((removed.ci_low, removed.ci_high), (-1.0, -1.0))
        self.assertNotEqual((row.ci_low, row.ci_high), (-1.0, -1.0))

    def test_mse_sensitivity_refuses_a_model_that_is_not_there(self):
        for model in ("PatchTST", "granite-flowstate-r1", "nobody"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                sensitivity.mse_sensitivity(self.results, self.omega, model, 50, 1)

    # ---- leave one model out -----------------------------------------------
    def test_leave_one_model_out_known_answers(self):
        # 5 metrics; 3 zero-shot models x 3 comparisons + 3 single-model families x 1.
        self.assertEqual(len(self.lomo), 5 * (3 * 3 + 3))
        for model, rho, lower in (("TiRex", -0.1, 0), ("Moirai_small", -0.1, 0),
                                  ("VisionTS", -1.0, 3)):
            with self.subTest(model=model):
                row = one(self.lomo, metric="MSE", comparison=DEEP, model_removed=model)
                self.assertAlmostEqual(row.spearman_rho, rho, places=12)
                self.assertAlmostEqual(row.rho_all_models, -0.1, places=12)
                self.assertAlmostEqual(row.rho_change, rho + 0.1, places=12)
                self.assertEqual((row.n, row.n_zero_shot_lower, row.n_zero_shot_lower_all_models,
                                  row.n_models_left_in_family, row.family_of_removed_model),
                                 (5, lower, 0, 2, "zero-shot"))
        # Every other column: the zero-shot mean without one model is 3.5, 2.5 or 3 on every
        # dataset, so rho stays -1 and only the sign counts move. deep = 2, 2.5, 3.5, 4, 5.5:
        # above 3.5 on 2 datasets (one tie), above 2.5 on 3 (one tie), above 3 on 3.
        for model, lower in (("TiRex", 2), ("Moirai_small", 3), ("VisionTS", 3)):
            for metric in ("sMAPE", "MAE", "CRPS", "MSE[0.5]"):
                with self.subTest(model=model, metric=metric):
                    row = one(self.lomo, metric=metric, comparison=DEEP, model_removed=model)
                    self.assertAlmostEqual(row.spearman_rho, -1.0, places=12)
                    self.assertAlmostEqual(row.rho_change, 0.0, places=12)
                    self.assertEqual((row.n_zero_shot_lower, row.n_zero_shot_lower_all_models),
                                     (lower, 3))

    def test_leave_one_model_out_uses_the_same_family_scores_as_family_table(self):
        results = prepared(pooled_results())
        mapped = results.assign(family=results.model.map(
            {"TiRex": "zero-shot", "Moirai_small": "zero-shot", "VisionTS": "zero-shot",
             "PatchTST": "deep", "Naive": "statistical", "Chronos_small": "pretrained"}))
        for model in ("TiRex", "VisionTS"):
            with self.subTest(model=model):
                fast = sensitivity.family_means(mapped[mapped.model != model], METRICS["CRPS"])
                slow = family_table(results, METRICS["CRPS"],
                                    groups=sensitivity.groups_without(model))
                pd.testing.assert_frame_equal(fast, slow, check_exact=True)
        # ett1 without VisionTS: rows 1, 19, 3, 5, so the mean is 7.
        self.assertEqual(fast.loc["ett1", "zero-shot"], 7.0)
        table = sensitivity.leave_one_model_out(results, tiny_omega(), 50, 1)
        row = one(table, metric="CRPS", comparison=DEEP, model_removed="VisionTS")
        # Without VisionTS the zero-shot means are 7, 7, 6, 12.5, 10.75 (Omega order):
        # two tied at 7, ranks 2.5, 2.5, 1, 5, 4 against 1..5.
        expected = stats.spearmanr([0.1, 0.2, 0.3, 0.4, 0.5], [7, 7, 6, 12.5, 10.75])[0]
        self.assertAlmostEqual(row.spearman_rho, expected, places=12)
        self.assertAlmostEqual(row.rho_all_models, -0.1, places=12)

    def test_removing_the_only_model_of_a_family_gives_an_empty_cell(self):
        rows = pick(self.lomo, model_removed="PatchTST")
        self.assertEqual(len(rows), 5)
        self.assertEqual(set(rows.comparison), {DEEP})
        self.assertTrue((rows.n == 0).all() and rows.spearman_rho.isna().all())
        self.assertTrue((rows.n_models_left_in_family == 0).all())
        self.assertEqual(set(pick(self.lomo, model_removed="Naive").comparison),
                         {"zero-shot vs statistical"})

    def test_summary_known_answers(self):
        # Only the zero-shot family has more than one model, so it is the only one summarised.
        self.assertEqual(len(self.summary), 5 * 3)
        self.assertEqual(set(self.summary.family_of_removed_model), {"zero-shot"})
        row = one(self.summary, metric="MSE", comparison=DEEP)
        self.assertAlmostEqual(row.rho_all_models, -0.1, places=12)
        self.assertAlmostEqual(row.rho_min, -1.0, places=12)
        self.assertAlmostEqual(row.rho_max, -0.1, places=12)
        self.assertAlmostEqual(row.max_abs_rho_change, 0.9, places=12)
        self.assertEqual((row.model_at_rho_min, row.model_at_max_abs_change), ("VisionTS", "VisionTS"))
        self.assertEqual((row.n_models_removed_in_turn, row.n_sign_changes), (3, 0))
        self.assertEqual((row.n_zero_shot_lower_all_models, row.n_zero_shot_lower_min,
                          row.n_zero_shot_lower_max), (0, 0, 3))
        # as recorded the interval includes zero; without VisionTS it is [-1, -1]: one change.
        self.assertTrue(row.ci_low_all_models < 0 < row.ci_high_all_models)
        self.assertEqual(row.n_interval_status_changes, 1)
        quiet = one(self.summary, metric="sMAPE", comparison=DEEP)
        self.assertAlmostEqual(quiet.rho_min, -1.0, places=12)
        self.assertAlmostEqual(quiet.rho_max, -1.0, places=12)
        self.assertAlmostEqual(quiet.max_abs_rho_change, 0.0, places=12)
        self.assertEqual((quiet.n_sign_changes, quiet.n_interval_status_changes), (0, 0))
        self.assertEqual((quiet.n_zero_shot_lower_min, quiet.n_zero_shot_lower_max), (2, 3))

    def test_summary_counts_sign_changes_and_names_the_models(self):
        lomo = pd.DataFrame({
            "metric": "m", "comparison": "c", "family_of_removed_model": "f",
            "model_removed": ["a", "b", "c", "d"], "spearman_rho": [-0.5, 0.2, -0.1, -0.9],
            "rho_all_models": -0.3, "rho_change": [-0.2, 0.5, 0.2, -0.6],
            "ci_low": [-0.8, -0.2, -0.6, -0.95], "ci_high": [-0.1, 0.6, 0.4, -0.5],
            "ci_low_all_models": -0.7, "ci_high_all_models": 0.2,
            "n_zero_shot_lower": [10, 12, 9, 11], "n_zero_shot_lower_all_models": 10})
        (row,) = sensitivity.leave_one_model_out_summary(lomo).itertuples()
        self.assertEqual((row.rho_min, row.model_at_rho_min), (-0.9, "d"))
        self.assertEqual((row.rho_max, row.model_at_rho_max), (0.2, "b"))
        self.assertEqual((row.max_abs_rho_change, row.model_at_max_abs_change), (0.6, "d"))
        self.assertEqual(row.n_sign_changes, 1)                    # only b is positive
        self.assertEqual(row.n_interval_status_changes, 2)         # a and d exclude zero
        self.assertEqual((row.n_zero_shot_lower_min, row.n_zero_shot_lower_max), (9, 12))
        self.assertEqual(row.n_models_removed_in_turn, 4)

    # ---- the paper table ---------------------------------------------------
    def test_paper_table_uses_smape_and_keeps_mse_in_a_separate_column(self):
        t = sensitivity.paper_like_for_like(self.results, self.omega, 300, 5)
        self.assertEqual(list(t.comparison), [DEEP, "zero-shot vs statistical",
                                              "zero-shot vs pretrained"])
        self.assertEqual(list(t.paper_rho[:2]), [-0.556, -0.234])
        self.assertTrue(np.isnan(t.paper_rho.iloc[2]))             # the paper gives no number
        self.assertTrue((t.paper_metric == "sMAPE").all() and (t.replication_metric == "sMAPE").all())
        self.assertTrue((t.paper_n_datasets == 28).all() and (t.n == 5).all())
        deep = t.iloc[0]
        self.assertAlmostEqual(deep.replication_rho, -1.0, places=12)          # sMAPE
        self.assertAlmostEqual(deep.not_like_for_like_mse_mean_rho, -0.1, places=12)
        self.assertEqual((deep.ci_low, deep.ci_high), (-1.0, -1.0))
        self.assertEqual(list(t.paper_rho_inside_interval), [False, False, None])
        # pretrained falls 9, 8, 7, 6, 5 with Omega, so Delta rises: rho = +1.
        self.assertAlmostEqual(t.iloc[2].replication_rho, 1.0, places=12)


class DegenerateResamples(unittest.TestCase):
    def test_cells_above_the_threshold_are_listed(self):
        tables = {
            "a.csv": pd.DataFrame({"metric": ["x", "y", "z"], "n_degenerate": [0, 5, 6],
                                   "n_resamples": [100, 100, 100]}),
            "b.csv": pd.DataFrame({"other": [1]})}
        cells = pipeline.degenerate_cells(tables, threshold=0.05)
        self.assertEqual(cells, [{"table": "a.csv", "row": {"metric": "z"}, "n_degenerate": 6,
                                  "n_resamples": 100, "share": 0.06}])
        self.assertEqual(pipeline.degenerate_cells(tables, threshold=0.5), [])

    def test_command_warns_when_a_cell_has_no_usable_resample(self):
        # The deep family is always half the zero-shot score, so Delta is 50 on every dataset
        # and the correlation is undefined in every resample.
        raw = tiny_results()
        deep = raw.model.isin(["PatchTST", "DLinear"])
        metric_columns = [c for c in raw.columns if c not in ("dataset", "model")]
        raw.loc[deep, metric_columns] = 0.5
        d = scratch_dir()
        mpath, rpath = write_tiny_inputs(d, results=raw)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code, text = run_cli(["--manifest", mpath, "--results", rpath, "--outdir",
                                  os.path.join(d, "out"), "--bootstrap-resamples", "40"])
        self.assertEqual(code, 0)
        self.assertIn("WARNING: in ", text)
        self.assertIn("bootstrap resamples were degenerate", text)
        self.assertIn("40 of 40 (100.0%)", text)
        import json
        with open(os.path.join(d, "out", "run_manifest.json"), encoding="utf-8") as f:
            warned = json.load(f)["degenerate_resample_warnings"]
        self.assertTrue(all(w["n_degenerate"] == 40 for w in warned
                            if w["row"].get("comparison") == DEEP
                            and w["table"] == "correlations_dataset_level.csv"))
        self.assertIn("correlations_dataset_level.csv", {w["table"] for w in warned})


if __name__ == "__main__":
    unittest.main()
