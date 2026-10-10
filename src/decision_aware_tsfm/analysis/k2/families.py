"""Model families, metric columns, and the per-family score table for K2.

The family lists are the ones the existing K2 record used. They are data, not
code: the run manifest records the exact model-to-family mapping of each run.
"""
from __future__ import annotations

import pandas as pd

from ..decision.guards import assert_no_held_out

# Family membership as used by the existing K2 record. The four lists were
# compared name by name with Appendix A.1 of the source paper (arXiv:2511.08884v1)
# on 2026-10-04 and agree, except that the paper writes "Chronos_bolt_base" and
# "Chronos_bolt_small" where the GIFT-Eval result files (and this mapping) write
# "chronos_bolt_base" and "chronos_bolt_small".
GROUPS: dict[str, list[str]] = {
    "zero-shot": [
        "TimesFM-2.5", "FlowState-9.1M", "Kairos_10m", "Kairos_23m", "Kairos_50m",
        "sundial_base_128m", "YingLong_110m", "YingLong_300m", "YingLong_50m",
        "YingLong_6m", "granite-flowstate-r1", "TiRex",
        "DeOSAlphaTimeGPTPredictor-2025", "Toto_Open_Base_1.0", "TabPFN-TS",
        "Moirai_base", "Moirai_large", "Moirai_small", "VisionTS",
    ],
    "pretrained": [
        "Moirai2", "chronos_bolt_base", "chronos_bolt_small", "Chronos_large",
        "Chronos_base", "Chronos_small", "TimesFM", "timesfm_2_0_500m",
        "TTM-R1-Pretrained", "TTM-R2-Pretrained", "Lag-Llama",
    ],
    "deep": [
        "xLSTM-Mixer", "PatchTST", "TFT", "N-BEATS", "DLinear", "TIDE",
        "DeepAR", "Crossformer", "iTransformer",
    ],
    "statistical": [
        "Auto_Arima", "Seasonal_Naive", "Auto_Theta", "Auto_ETS", "Naive",
    ],
}

METRICS: dict[str, str] = {
    "sMAPE": "eval_metrics/sMAPE[0.5]",
    "MSE": "eval_metrics/MSE[mean]",
    "MAE": "eval_metrics/MAE[0.5]",             # pinball at tau=0.5 == MAE / 2
    "CRPS": "eval_metrics/mean_weighted_sum_quantile_loss",
}

# The published GIFT-Eval column with the squared error of the median (0.5
# quantile) forecast. The recorded MSE rows use MSE[mean], the squared error of
# the mean forecast. Used only by the sensitivity tables, never by the headline.
MSE_MEDIAN_FORECAST_COLUMN = "eval_metrics/MSE[0.5]"

REFERENCE_FAMILY = "zero-shot"
COMPARISONS: tuple[str, ...] = ("deep", "statistical", "pretrained")

# Within-family summaries. These labels are the ones in the existing record and
# are written only to the legacy table within_family_sensitivity.csv, which a
# regression test compares byte for byte with the recorded file. Two of them are
# not accurate and must not be used in new tables:
#   * "paper's choice": the source paper does not say how a family score is
#     formed for its correlations (see the hardening note, section 3);
#   * "median model" and "best model": the recorded computation is over
#     model-configuration rows, not over models.
# New tables use SUMMARY_LABELS below.
WITHIN_FAMILY_LABELS: dict[str, str] = {
    "mean": "mean across family (paper's choice)",
    "median": "median model in family",
    "min": "best model in family",
}

# Corrected labels for new tables, keyed by (within-family summary, pooling).
SUMMARY_LABELS: dict[tuple[str, str], str] = {
    ("mean", "rows"): "mean across family (this replication's choice; the paper does not state one)",
    ("median", "rows"): "median over the family's model-configuration rows",
    ("min", "rows"): "lowest single model-configuration row in the family",
    ("median", "models"): "median model in family (each model averaged over configurations first)",
    ("min", "models"): "best model in family (each model averaged over configurations first)",
}

# What the summary is taken over.
#   "rows"   : every (model, configuration) row of the family inside the unit.
#              This is what the existing record computed.
#   "models" : each model is first averaged over the unit's configurations,
#              then the summary is taken over models.
# The two agree for the mean when every model has every configuration, and
# they agree for every summary when the unit is a single configuration.
POOLINGS: tuple[str, ...] = ("rows", "models")

UNIT_COLUMN = {"base": "base", "config": "dataset"}


class FamilyMappingError(ValueError):
    """The model-to-family mapping is not a function."""


def model_to_family(groups: dict[str, list[str]] | None = None) -> dict[str, str]:
    """Invert family -> models into model -> family; a model in two families is an error."""
    groups = GROUPS if groups is None else groups
    out: dict[str, str] = {}
    for family, models in groups.items():
        for model in models:
            if model in out:
                raise FamilyMappingError(
                    f"model {model!r} is listed in both {out[model]!r} and {family!r}")
            out[model] = family
    return out


def family_table(results: pd.DataFrame, metric_col: str, within_family: str = "mean",
                 unit: str = "base", pooling: str = "rows",
                 groups: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """One score per (unit, model family).

    results needs the columns model, <metric_col> and the unit column ('base'
    for one row per base dataset, 'dataset' for one row per GIFT-Eval
    configuration). Models that are not in the mapping are ignored. Returns a
    table indexed by unit with one column per family.
    """
    if within_family not in WITHIN_FAMILY_LABELS:
        raise ValueError(f"within_family must be one of {sorted(WITHIN_FAMILY_LABELS)}, "
                         f"got {within_family!r}")
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
    if unit not in UNIT_COLUMN:
        raise ValueError(f"unit must be one of {sorted(UNIT_COLUMN)}, got {unit!r}")
    key = UNIT_COLUMN[unit]
    assert_no_held_out(results[key].unique(), "K2 family table")
    fam = model_to_family(groups)
    r = results.copy()
    r["family"] = r.model.map(fam)
    r = r[r.family.notna()]
    if pooling == "models":
        r = r.groupby([key, "family", "model"])[metric_col].mean().reset_index()
    return r.groupby([key, "family"])[metric_col].agg(within_family).unstack("family")


def roster_report(results: pd.DataFrame,
                  groups: dict[str, list[str]] | None = None) -> dict:
    """Which mapped models are present in the results, and how many models are ignored."""
    groups = GROUPS if groups is None else groups
    fam = model_to_family(groups)
    present = set(results.model.unique())
    families = {}
    for family, models in groups.items():
        families[family] = {
            "n_mapped": len(models),
            "n_present": sum(m in present for m in models),
            "absent_from_results": sorted(m for m in models if m not in present),
        }
    ignored = sorted(present - set(fam))
    return {"families": families, "n_models_in_results": len(present),
            "n_models_ignored": len(ignored), "models_ignored": ignored}
