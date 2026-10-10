"""Sensitivity tables that sit beside the recorded K2 headline. None of them
changes the headline computation.

  model_score_diagnostics   how much of a family's mean score each model carries
  mse_sensitivity           the MSE rows as recorded, with the dominant model
                            removed, and on the MSE[0.5] column
  leave_one_model_out       every headline correlation with one model removed
  leave_one_model_out_summary   the range of those, per headline cell
  paper_like_for_like       the published correlations beside the replication
                            on the metric the paper used (sMAPE)
  within_family_relabelled  the recorded sensitivity table with accurate labels
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..decision.guards import assert_no_held_out
from .families import (COMPARISONS, GROUPS, METRICS, MSE_MEDIAN_FORECAST_COLUMN,
                       REFERENCE_FAMILY, SUMMARY_LABELS, WITHIN_FAMILY_LABELS, family_table,
                       model_to_family)
from .joins import correlation_row, join_omega_by_base
from .loading import OmegaTables
from .statistic import DEFAULT_RESAMPLES, DEFAULT_SEED, finite_pairs, sign_counts, spearman

# Metrics for the leave-one-model-out table: the four recorded ones plus the
# MSE of the median forecast, so the alternative MSE column is covered as well.
SENSITIVITY_METRICS: dict[str, str] = {**METRICS, "MSE[0.5]": MSE_MEDIAN_FORECAST_COLUMN}

MSE_COLUMN = METRICS["MSE"]
MAE_COLUMN = METRICS["MAE"]

# (variant, plain description, metric column, remove the dominant model?)
MSE_VARIANTS: tuple[tuple[str, str, str, bool], ...] = (
    ("as_recorded", "MSE[mean] column, every zero-shot model (the recorded headline)",
     MSE_COLUMN, False),
    ("dominant_model_removed", "MSE[mean] column, dominant zero-shot model removed",
     MSE_COLUMN, True),
    ("median_forecast_column", "MSE[0.5] column, every zero-shot model",
     MSE_MEDIAN_FORECAST_COLUMN, False),
    ("median_forecast_column_dominant_model_removed",
     "MSE[0.5] column, the same model removed", MSE_MEDIAN_FORECAST_COLUMN, True),
    ("mae_for_comparison", "MAE[0.5] column, every zero-shot model (for comparison)",
     MAE_COLUMN, False),
)

# What the source paper (arXiv:2511.08884v1, section 5, "Model-Family-Specific
# Patterns") states for the correlation of Omega with Delta sMAPE: "Spearman
# correlations ranging from -0.234 for statistical to -0.556 for deep learning".
# It gives no number for the pretrained comparison and no p-values. Its
# analysis uses 28 GIFT-Eval datasets and sMAPE only.
PAPER_SOURCE = "arXiv:2511.08884v1, section 5"
PAPER_METRIC = "sMAPE"
PAPER_N_DATASETS = 28
PAPER_PUBLISHED_RHO: dict[str, float | None] = {
    "deep": -0.556, "statistical": -0.234, "pretrained": None}


def _groups(groups):
    return GROUPS if groups is None else groups


def groups_without(model: str, groups: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """The family mapping with one model taken out of whichever family holds it."""
    return {family: [m for m in models if m != model] for family, models in _groups(groups).items()}


def present_models(results: pd.DataFrame, groups=None) -> dict[str, list[str]]:
    """Mapped models that have rows in the results, per family, in mapping order."""
    present = set(results.model.unique())
    return {family: [m for m in models if m in present] for family, models in _groups(groups).items()}


def _analysed(results: pd.DataFrame, omega: OmegaTables) -> pd.DataFrame:
    """Rows of mapped models on the base datasets that have an Omega value."""
    fam = model_to_family()
    r = results[results.base.isin(set(omega.by_base.base))].copy()
    r["family"] = r.model.map(fam)
    return r[r.family.notna()]


# --------------------------------------------------------------------------
# how much of a family mean one model carries
# --------------------------------------------------------------------------
def model_score_diagnostics(results: pd.DataFrame, omega: OmegaTables) -> pd.DataFrame:
    """Per metric and model: its share of the family's summed score.

    Each model is averaged over a base dataset's configurations. On each
    dataset a model's share is its score divided by the sum over the family's
    models, so with k models an even split is 1/k. The table reports the
    median share across datasets, the count of datasets where the model has
    the family's highest score, and the median ratio to the family's median
    model. Two unit checks are added for every model, on its rows in the
    analysed datasets: the median of sqrt(MSE[mean]) / MAE[0.5], and the median
    of MSE[mean] / MSE[0.5].
    """
    r = _analysed(results, omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        checks = pd.DataFrame({
            "model": r.model,
            "rmse_over_mae": np.sqrt(r[MSE_COLUMN]) / r[MAE_COLUMN],
            "mse_mean_over_mse_median": r[MSE_COLUMN] / r[MSE_MEDIAN_FORECAST_COLUMN],
        }).groupby("model").median()
    rows = []
    for mname, mcol in SENSITIVITY_METRICS.items():
        per_model = r.groupby(["family", "base", "model"])[mcol].mean()
        for family in per_model.index.get_level_values("family").unique():
            t = per_model[family].unstack("model")
            share = t.div(t.sum(axis=1), axis=0)
            ratio = t.div(t.median(axis=1), axis=0)
            highest = t.idxmax(axis=1).value_counts()
            for model in t.columns:
                rows.append(dict(
                    metric=mname, metric_column=mcol, family=family, model=model,
                    n_models_in_family=int(t.shape[1]), n_datasets=int(t[model].notna().sum()),
                    even_share=1.0 / t.shape[1],
                    median_share_of_family_sum=float(share[model].median()),
                    n_datasets_highest_in_family=int(highest.get(model, 0)),
                    median_ratio_to_family_median_model=float(ratio[model].median()),
                    median_rmse_over_mae=float(checks.loc[model, "rmse_over_mae"]),
                    median_mse_mean_over_mse_median=float(
                        checks.loc[model, "mse_mean_over_mse_median"])))
    out = pd.DataFrame(rows)
    return out.sort_values(["metric", "family", "median_share_of_family_sum"],
                           ascending=[True, True, False], kind="stable").reset_index(drop=True)


def dominant_model(diagnostics: pd.DataFrame, metric: str = "MSE",
                   family: str = REFERENCE_FAMILY) -> str:
    """The family's model with the largest median share of the family sum on one metric."""
    d = diagnostics[(diagnostics.metric == metric) & (diagnostics.family == family)]
    if d.empty:
        raise ValueError(f"no diagnostics for metric {metric!r} and family {family!r}")
    return str(d.sort_values("median_share_of_family_sum", ascending=False, kind="stable")
               .model.iloc[0])


# --------------------------------------------------------------------------
# one correlation cell, used by every table below
# --------------------------------------------------------------------------
def family_means(rows_with_family: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Family mean per base dataset from rows that already carry a 'family' column.

    The same numbers as family_table(results, metric_col) on the same rows (a
    test checks they are equal); used where the table is rebuilt once per
    removed model, to avoid mapping models to families every time.
    """
    return rows_with_family.groupby(["base", "family"])[metric_col].mean().unstack("family")


def _joined(results, omega, metric_col, groups) -> pd.DataFrame:
    """Family-mean table per base dataset under one mapping, joined to Omega."""
    return join_omega_by_base(family_table(results, metric_col, groups=groups), omega)


def _cell(results, omega, metric_col, other, groups, n_resamples, seed) -> dict:
    """Family-mean correlation with Omega for zero-shot versus `other`, with signs."""
    return _cell_from(_joined(results, omega, metric_col, groups), other, n_resamples, seed)


def _cell_from(t: pd.DataFrame, other: str, n_resamples, seed) -> dict:
    """The correlation cell from an already joined table.

    If a family has no model left there is nothing to correlate: n is 0 and the
    statistics are NaN.
    """
    if REFERENCE_FAMILY not in t.columns or other not in t.columns:
        nan = float("nan")
        return dict(n=0, spearman_rho=nan, p_value=nan, ci_low=nan, ci_high=nan, ci_level=nan,
                    n_resampled_units=0, n_resamples=int(n_resamples), n_degenerate=0,
                    seed=int(seed), n_zero_shot_lower=0, n_other_lower=0, n_tie=0)
    x, y, _ = finite_pairs(t, REFERENCE_FAMILY, other)
    signs = sign_counts(t, REFERENCE_FAMILY, other)
    return dict(**correlation_row(x, y, n_resamples, seed),
                n_zero_shot_lower=signs["n_a_lower"], n_other_lower=signs["n_b_lower"],
                n_tie=signs["n_tie"])


def mse_sensitivity(results: pd.DataFrame, omega: OmegaTables, model_removed: str,
                    n_resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """The MSE correlations under each variant in MSE_VARIANTS, family mean, per base dataset.

    model_removed is the zero-shot model taken out in the "removed" variants
    (the command passes the dominant model found by model_score_diagnostics).
    """
    n_all = len(present_models(results)[REFERENCE_FAMILY])
    if model_removed not in present_models(results)[REFERENCE_FAMILY]:
        raise ValueError(f"{model_removed!r} is not a {REFERENCE_FAMILY} model present in the results")
    rows = []
    for variant, label, mcol, remove in MSE_VARIANTS:
        groups = groups_without(model_removed) if remove else None
        for other in COMPARISONS:
            rows.append(dict(
                variant=variant, variant_description=label, metric_column=mcol,
                model_removed=model_removed if remove else "",
                n_zero_shot_models=n_all - int(remove),
                comparison=f"{REFERENCE_FAMILY} vs {other}",
                **_cell(results, omega, mcol, other, groups, n_resamples, seed)))
    return pd.DataFrame(rows)


def leave_one_model_out(results: pd.DataFrame, omega: OmegaTables,
                        n_resamples: int = DEFAULT_RESAMPLES,
                        seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Every family-mean correlation recomputed with one model removed.

    Removing a zero-shot model changes all three comparisons. Removing a model
    of another family changes only the comparison against that family. Models
    absent from the results are skipped (removing them changes nothing). The
    all-model value is repeated on each row so the change can be read off.
    """
    present = present_models(results)
    assert_no_held_out(results.base.unique(), "K2 leave one model out")
    mapped = results[results.model.isin(set(model_to_family()))].copy()
    mapped["family"] = mapped.model.map(model_to_family())
    rows = []
    for mname, mcol in SENSITIVITY_METRICS.items():
        all_models = join_omega_by_base(family_means(mapped, mcol), omega)
        full = {other: _cell_from(all_models, other, n_resamples, seed) for other in COMPARISONS}
        for family, models in present.items():
            affected = COMPARISONS if family == REFERENCE_FAMILY else (family,)
            for model in models:
                without = join_omega_by_base(
                    family_means(mapped[mapped.model != model], mcol), omega)
                for other in affected:
                    cell = _cell_from(without, other, n_resamples, seed)
                    rows.append(dict(
                        metric=mname, metric_column=mcol,
                        comparison=f"{REFERENCE_FAMILY} vs {other}",
                        family_of_removed_model=family, model_removed=model,
                        n_models_left_in_family=len(models) - 1, **cell,
                        rho_all_models=full[other]["spearman_rho"],
                        rho_change=cell["spearman_rho"] - full[other]["spearman_rho"],
                        ci_low_all_models=full[other]["ci_low"],
                        ci_high_all_models=full[other]["ci_high"],
                        n_zero_shot_lower_all_models=full[other]["n_zero_shot_lower"]))
    return pd.DataFrame(rows)


def _excludes_zero(low, high) -> bool:
    return bool(low > 0 or high < 0)


def leave_one_model_out_summary(lomo: pd.DataFrame) -> pd.DataFrame:
    """Per metric, comparison and family of the removed model: how far rho moves.

    n_sign_changes counts removals after which rho has the other sign.
    n_interval_status_changes counts removals after which the bootstrap interval
    switches between excluding zero and including it.
    """
    rows = []
    keys = ["metric", "comparison", "family_of_removed_model"]
    for (metric, comparison, family), g in lomo.groupby(keys, sort=False):
        g = g[g.spearman_rho.notna()]
        if g.empty:          # the family had a single model; nothing to summarise
            continue
        full = float(g.rho_all_models.iloc[0])
        full_excludes = _excludes_zero(g.ci_low_all_models.iloc[0], g.ci_high_all_models.iloc[0])
        lo, hi = g.loc[g.spearman_rho.idxmin()], g.loc[g.spearman_rho.idxmax()]
        far = g.loc[g.rho_change.abs().idxmax()]
        excludes = [_excludes_zero(a, b) for a, b in zip(g.ci_low, g.ci_high)]
        rows.append(dict(
            metric=metric, comparison=comparison, family_of_removed_model=family,
            n_models_removed_in_turn=int(len(g)), rho_all_models=full,
            ci_low_all_models=float(g.ci_low_all_models.iloc[0]),
            ci_high_all_models=float(g.ci_high_all_models.iloc[0]),
            rho_min=float(lo.spearman_rho), model_at_rho_min=lo.model_removed,
            rho_max=float(hi.spearman_rho), model_at_rho_max=hi.model_removed,
            max_abs_rho_change=float(abs(far.rho_change)), model_at_max_abs_change=far.model_removed,
            n_sign_changes=int((np.sign(g.spearman_rho) != np.sign(full)).sum()),
            n_interval_status_changes=int(sum(e != full_excludes for e in excludes)),
            n_zero_shot_lower_all_models=int(g.n_zero_shot_lower_all_models.iloc[0]),
            n_zero_shot_lower_min=int(g.n_zero_shot_lower.min()),
            n_zero_shot_lower_max=int(g.n_zero_shot_lower.max())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# what the paper published, beside the like-for-like replication
# --------------------------------------------------------------------------
def paper_like_for_like(results: pd.DataFrame, omega: OmegaTables,
                        n_resamples: int = DEFAULT_RESAMPLES,
                        seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Published correlation beside the replication on the paper's metric (sMAPE).

    The MSE[mean] value is carried in a separate column, named so that it is
    not mistaken for a replication of the published figure: the paper computes
    its family comparison on sMAPE only.
    """
    rows = []
    for other in COMPARISONS:
        like = _cell(results, omega, METRICS[PAPER_METRIC], other, None, n_resamples, seed)
        mse = _cell(results, omega, MSE_COLUMN, other, None, n_resamples, seed)
        published = PAPER_PUBLISHED_RHO[other]
        rows.append(dict(
            comparison=f"{REFERENCE_FAMILY} vs {other}", paper_source=PAPER_SOURCE,
            paper_metric=PAPER_METRIC, paper_n_datasets=PAPER_N_DATASETS,
            paper_rho=np.nan if published is None else published,
            replication_metric=PAPER_METRIC, n=like["n"],
            replication_rho=like["spearman_rho"], p_value=like["p_value"],
            ci_low=like["ci_low"], ci_high=like["ci_high"], ci_level=like["ci_level"],
            n_resamples=like["n_resamples"], n_degenerate=like["n_degenerate"], seed=like["seed"],
            paper_rho_inside_interval=(None if published is None
                                       else bool(like["ci_low"] <= published <= like["ci_high"])),
            not_like_for_like_mse_mean_rho=mse["spearman_rho"]))
    return pd.DataFrame(rows)


def within_family_relabelled(results: pd.DataFrame, omega: OmegaTables) -> pd.DataFrame:
    """The recorded sensitivity table (sMAPE, zero-shot vs deep) with accurate labels.

    Same numbers and rounding as within_family_sensitivity.csv for the three
    recorded rows, plus the two model-level rows. The legacy file keeps its
    recorded labels so that it stays byte-identical to the record.
    """
    rows = []
    for (agg, pooling), label in SUMMARY_LABELS.items():
        t = join_omega_by_base(family_table(results, METRICS["sMAPE"], agg, pooling=pooling), omega)
        rho, p, n = spearman(t, REFERENCE_FAMILY, "deep")
        rows.append(dict(within_family=label, summary=agg, pooling=pooling,
                         label_in_recorded_table=WITHIN_FAMILY_LABELS[agg] if pooling == "rows" else "",
                         n=n, spearman_rho=round(rho, 3), p_value=round(p, 4)))
    return pd.DataFrame(rows)
