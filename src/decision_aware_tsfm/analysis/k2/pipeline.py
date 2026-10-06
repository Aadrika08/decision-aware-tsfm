"""Compute every K2 table from the saved inputs and write one output directory.

No forecasting model is run. All scores are the published GIFT-Eval per-model
results; Omega comes from the manifest.

The tables in the recorded (legacy) format are never changed by the newer
analyses: the sensitivity tables are computed beside them (sensitivity.py).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .environment import environment_record, mismatch_warning
from .families import (COMPARISONS, GROUPS, METRICS, REFERENCE_FAMILY, SUMMARY_LABELS,
                       WITHIN_FAMILY_LABELS, family_table, model_to_family, roster_report)
from .figure import FigureUnavailable, write_replication_figure
from .heldout import FUZZY_RULE, STRICT_RULE
from .joins import (CONFIG_MATCHINGS, config_join_report, config_keys, join_omega_by_base,
                    join_omega_by_config)
from .joins import correlation_row as _correlation_row
from .loading import (DEFAULT_MANIFEST_VERSION, DEFAULT_OMEGA_COLUMN, OmegaTables,
                      ResultsLayoutError, aggregate_omega, load_results, read_manifest, sha256_file)
from .sensitivity import (MSE_VARIANTS, PAPER_METRIC, PAPER_SOURCE, SENSITIVITY_METRICS,
                          dominant_model, leave_one_model_out, leave_one_model_out_summary,
                          model_score_diagnostics, mse_sensitivity, paper_like_for_like,
                          within_family_relabelled)
from .statistic import (DEFAULT_LEVEL, DEFAULT_RESAMPLES, DEFAULT_SEED, DEGENERATE_WARN_FRACTION,
                        check_bootstrap_parameters, finite_pairs, relative_delta, sign_counts,
                        spearman)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS = str(PROJECT_ROOT / "artifacts" / "reference" / "k2" / "gift_results.csv")
DEFAULT_MANIFEST = str(PROJECT_ROOT / "artifacts" / "reference" / "manifests" / "dev_manifest.json")
DEFAULT_OUTDIR = str(PROJECT_ROOT / "artifacts" / "runs" / "k2")

# (within-family summary, pooling). The mean is the headline; for the mean the
# two poolings agree whenever every model has every configuration.
SUMMARIES: tuple[tuple[str, str], ...] = (
    ("mean", "rows"), ("median", "rows"), ("min", "rows"), ("median", "models"), ("min", "models"))

FIGURE_NAME = "replication_figure.png"
MANIFEST_NAME = "run_manifest.json"


class OutputExistsError(FileExistsError):
    """The output directory already exists, or lies inside a run; K2 never overwrites a run."""


class InvalidParameterError(ValueError):
    """A run parameter has a value K2 cannot use."""


def check_outdir(outdir: str) -> str:
    """Absolute output path, refused if it exists or lies inside an existing run directory."""
    outdir = os.path.abspath(outdir)
    if os.path.lexists(outdir):
        raise OutputExistsError(
            f"output directory already exists and is never overwritten: {outdir}\n"
            "  pass a new --outdir")
    parent = Path(outdir).parent
    for ancestor in (parent, *parent.parents):
        if (ancestor / MANIFEST_NAME).exists():
            raise OutputExistsError(
                f"output directory {outdir} lies inside an existing K2 run directory "
                f"({ancestor}, which holds {MANIFEST_NAME}). A run directory holds exactly one "
                "run; it already exists and nothing is written into it.\n  pass a new --outdir")
    return outdir


def degenerate_cells(tables: dict, threshold: float = DEGENERATE_WARN_FRACTION) -> list[dict]:
    """Table rows whose share of degenerate bootstrap resamples is above the threshold."""
    cells = []
    for name, table in tables.items():
        if not {"n_degenerate", "n_resamples"} <= set(table.columns):
            continue
        share = table.n_degenerate / table.n_resamples.where(table.n_resamples > 0)
        for i in table.index[share > threshold]:
            row = table.loc[i]
            labels = {c: row[c] for c in table.columns if isinstance(row[c], str) and row[c]
                      and c not in ("metric_column", "variant_description", "paper_source")}
            cells.append({"table": name, "row": labels, "n_degenerate": int(row.n_degenerate),
                          "n_resamples": int(row.n_resamples), "share": float(share[i])})
    return cells


# --------------------------------------------------------------------------
# tables in the recorded (legacy) format
# --------------------------------------------------------------------------
def summary_stats(results: pd.DataFrame, omega: OmegaTables) -> pd.DataFrame:
    """The twelve headline correlations, rounded as in the existing record."""
    rows = []
    for mname, mcol in METRICS.items():
        t = join_omega_by_base(family_table(results, mcol), omega)
        for other in COMPARISONS:
            rho, p, n = spearman(t, REFERENCE_FAMILY, other)
            rows.append(dict(metric=mname, comparison=f"{REFERENCE_FAMILY} vs {other}",
                             n=n, spearman_rho=round(rho, 3), p_value=round(p, 4)))
    return pd.DataFrame(rows)


def recorded_within_family_sensitivity(results: pd.DataFrame, omega: OmegaTables) -> pd.DataFrame:
    """The recorded sensitivity table: sMAPE, zero-shot vs deep, row pooling."""
    rows = []
    for agg in ("mean", "median", "min"):
        t = join_omega_by_base(family_table(results, METRICS["sMAPE"], agg), omega)
        rho, p, n = spearman(t, REFERENCE_FAMILY, "deep")
        rows.append(dict(within_family=WITHIN_FAMILY_LABELS[agg], n=n,
                         spearman_rho=round(rho, 3), p_value=round(p, 4)))
    return pd.DataFrame(rows)


def joined_table(results: pd.DataFrame, omega: OmegaTables) -> pd.DataFrame:
    """Per base dataset and family: mean scores joined to the dataset's Omega summary."""
    fam = model_to_family()
    r = results.copy()
    r["family"] = r.model.map(fam)
    r = r[r.family.notna()]
    long = r.groupby(["base", "family"])[list(METRICS.values())].mean().reset_index()
    long.columns = ["dataset", "model_family"] + list(METRICS)
    joined = long.merge(omega.by_base.rename(columns={"base": "dataset"}), on="dataset")
    joined["pinball_tau0.5"] = joined["MAE"] / 2
    return joined.round(5)


# --------------------------------------------------------------------------
# new tables
# --------------------------------------------------------------------------
def dataset_level_correlations(results: pd.DataFrame, omega: OmegaTables,
                               n_resamples: int = DEFAULT_RESAMPLES,
                               seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Every metric, comparison and within-family summary at one row per base dataset.

    Full precision, with a bootstrap interval that resamples base datasets.
    """
    rows = []
    for mname, mcol in METRICS.items():
        for agg, pooling in SUMMARIES:
            t = join_omega_by_base(family_table(results, mcol, agg, pooling=pooling), omega)
            for other in COMPARISONS:
                x, y, _ = finite_pairs(t, REFERENCE_FAMILY, other)
                rows.append(dict(unit="base_dataset", metric=mname,
                                 comparison=f"{REFERENCE_FAMILY} vs {other}",
                                 within_family=agg, pooling=pooling,
                                 **_correlation_row(x, y, n_resamples, seed)))
    return pd.DataFrame(rows)


def dataset_sign_counts(results: pd.DataFrame, omega: OmegaTables) -> pd.DataFrame:
    """On how many of the analysed base datasets the zero-shot family has the lower error."""
    rows = []
    for mname, mcol in METRICS.items():
        for agg, pooling in SUMMARIES:
            t = join_omega_by_base(family_table(results, mcol, agg, pooling=pooling), omega)
            for other in COMPARISONS:
                c = sign_counts(t, REFERENCE_FAMILY, other)
                rows.append(dict(metric=mname, comparison=f"{REFERENCE_FAMILY} vs {other}",
                                 within_family=agg, pooling=pooling, n=c["n"],
                                 n_zero_shot_lower=c["n_a_lower"], n_other_lower=c["n_b_lower"],
                                 n_tie=c["n_tie"]))
    return pd.DataFrame(rows)


def dataset_deltas(results: pd.DataFrame, omega: OmegaTables) -> pd.DataFrame:
    """The points behind the headline correlations: Omega and Delta per base dataset (family mean)."""
    frames = []
    for mname, mcol in METRICS.items():
        t = join_omega_by_base(family_table(results, mcol), omega)
        for other in COMPARISONS:
            frames.append(pd.DataFrame({
                "dataset": t.index, "omega": t["omega"].to_numpy(), "metric": mname,
                "comparison": f"{REFERENCE_FAMILY} vs {other}",
                "zero_shot_score": t[REFERENCE_FAMILY].to_numpy(),
                "other_score": t[other].to_numpy(),
                "delta": relative_delta(t[REFERENCE_FAMILY], t[other])}))
    return pd.concat(frames, ignore_index=True)


def config_level_correlations(results: pd.DataFrame, omega: OmegaTables,
                              n_resamples: int = DEFAULT_RESAMPLES,
                              seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Sensitivity only: the same correlation at one row per GIFT-Eval configuration.

    Configurations of one base dataset share data and (within a frequency) one
    Omega value, so they are not independent. The p-value here treats them as
    independent and is reported for comparison only. The interval resamples
    base datasets (all configurations of a drawn dataset come along).
    """
    keys = config_keys(results)
    rows = []
    for matching in CONFIG_MATCHINGS:
        for mname, mcol in METRICS.items():
            for agg in ("mean", "median", "min"):
                t = join_omega_by_config(family_table(results, mcol, agg, unit="config"),
                                         keys, omega, matching)
                for other in COMPARISONS:
                    x, y, ok = finite_pairs(t, REFERENCE_FAMILY, other)
                    clusters = t.base.to_numpy()[ok]
                    rows.append(dict(unit="configuration", matching=matching, metric=mname,
                                     comparison=f"{REFERENCE_FAMILY} vs {other}",
                                     within_family=agg,
                                     **_correlation_row(x, y, n_resamples, seed, clusters)))
    return pd.DataFrame(rows)


def joined_config_table(results: pd.DataFrame, omega: OmegaTables) -> pd.DataFrame:
    """Per configuration and family: mean scores and the Omega value under each matching rule."""
    fam = model_to_family()
    r = results.copy()
    r["family"] = r.model.map(fam)
    r = r[r.family.notna()]
    long = r.groupby(["dataset", "family"])[list(METRICS.values())].mean().reset_index()
    long.columns = ["config", "model_family"] + list(METRICS)
    keys = config_keys(results).rename(columns={"dataset": "config"})
    out = keys.merge(long, on="config")
    bf = omega.by_base_freq[["base", "freq", "omega"]].rename(
        columns={"omega": "omega_dataset_frequency"})
    bb = omega.by_base[["base", omega.omega_col]].rename(
        columns={omega.omega_col: "omega_dataset_broadcast"})
    out = out.merge(bf, on=["base", "freq"], how="left").merge(bb, on="base", how="left")
    return out.sort_values(["config", "model_family"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def _rel(path: str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _input_record(path: str) -> dict:
    return {"path": _rel(path), "sha256": sha256_file(path), "bytes": os.path.getsize(path)}


def _code_record() -> dict:
    here = Path(__file__).resolve().parent
    return {f.name: sha256_file(str(f)) for f in sorted(here.glob("*.py"))}


def _echo_held_out(echo, results_report: dict, manifest_report: dict) -> None:
    """Print every held-out exclusion and every fuzzy match, so none is hidden in a file."""
    for item in results_report["held_out_exclusions"]:
        echo(f"  [WARNING] held-out rule: dropped {item['n_rows']} results row(s) of dataset "
             f"{item['dataset']!r} ({', '.join(item['reasons'])}). If this dataset is not held-out "
             "data, this is a false positive of the substring rule and the rows are missing "
             "from the analysis.")
    for item in results_report["held_out_fuzzy_reports"]:
        echo(f"  [WARNING] possible held-out name in the results, NOT dropped: dataset "
             f"{item['dataset']!r}, {item['n_rows']} row(s) ({', '.join(item['reasons'])}). "
             "Check it by hand.")
    for item in manifest_report.get("held_out_fuzzy_reports", []):
        used = "in a subset the analysis uses" if item["in_a_mapped_subset"] else "in an unused subset"
        echo(f"  [WARNING] possible held-out name in the manifest, NOT refused: {item['field']} = "
             f"{item['value']!r}, {item['n_rows']} row(s), {used}, subset(s) {item['subsets']} "
             f"({', '.join(item['reasons'])}). Check it by hand.")


def _echo_missing_omega(echo, omega: OmegaTables) -> None:
    report = omega.report
    if not report["n_series_missing_selected_omega"]:
        return
    echo("\n" + "!" * 68)
    echo(f"WARNING: --allow-missing-omega is in effect. Omega column {omega.omega_col!r} has no "
         f"value for {report['n_series_missing_selected_omega']} series.")
    echo(f"  n CHANGES: {report['n_base_datasets']} base datasets have an Omega value, not "
         f"{report['n_base_datasets_before_missing_omega']}; "
         f"{report['n_base_frequency_subsets']} dataset-frequency subsets, not "
         f"{report['n_base_frequency_subsets_before_missing_omega']}.")
    if report["base_datasets_removed_no_selected_omega"]:
        echo(f"  base datasets removed (no series has a value): "
             f"{report['base_datasets_removed_no_selected_omega']}")
    for base, c in report["missing_selected_omega_by_base"].items():
        echo(f"  {base}: {c['n_missing']} of {c['n_series']} series have no value")
    echo("  Results of this run are not comparable with a run on the full set of datasets.")
    echo("!" * 68)


def run_k2(results_path: str = DEFAULT_RESULTS, manifest_path: str = DEFAULT_MANIFEST,
           outdir: str = DEFAULT_OUTDIR, omega_col: str = DEFAULT_OMEGA_COLUMN,
           manifest_version: str = DEFAULT_MANIFEST_VERSION,
           n_resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED,
           figure: bool = False, command: list[str] | None = None, echo=print,
           allow_missing_omega: bool = False) -> dict:
    """Regenerate every K2 output into a new directory and return the run manifest."""
    try:
        n_resamples, seed = check_bootstrap_parameters(n_resamples, seed)
    except ValueError as exc:
        raise InvalidParameterError(str(exc)) from exc
    outdir = check_outdir(outdir)
    for path, what in ((results_path, "results"), (manifest_path, "manifest")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{what} not found at {path}")

    env = environment_record()
    warning = mismatch_warning(env["pin_mismatches"])
    if warning:
        echo(warning)

    # The manifest is read first: its held_out_strict list is also applied to the results.
    manifest = read_manifest(manifest_path, manifest_version)
    results, results_report = load_results(results_path, manifest.report.get("held_out_listed", ()))
    absent = sorted(set(SENSITIVITY_METRICS.values()) - set(results.columns))
    if absent:
        raise ResultsLayoutError(f"results have no column(s) {absent}; K2 needs "
                                 f"{sorted(SENSITIVITY_METRICS.values())}")
    omega = aggregate_omega(manifest, omega_col, allow_missing=allow_missing_omega)
    roster = roster_report(results)

    file_record = manifest.report.get("manifest_file", {})
    echo(f"\nmanifest: {manifest_path} (layout {manifest_version}, Omega column {omega_col})")
    echo(f"  frozen: {'yes' if file_record.get('frozen') else 'no'} "
         f"({file_record.get('frozen_basis', 'not recorded')}); "
         f"declared version: {file_record.get('declared_version')}")
    echo(f"  {int(omega.by_base.n_series.sum())} series across {len(omega.by_base)} base datasets")
    if manifest.report.get("unmapped_subsets_not_used"):
        echo("  [warn] manifest subsets with no mapping to GIFT-Eval, not used: "
             f"{manifest.report['unmapped_subsets_not_used']}")
    if manifest.report.get("known_duplicate_subsets_not_used"):
        echo("  known duplicate subsets, checked identical and not used: "
             f"{manifest.report['known_duplicate_subsets_not_used']}")
    echo(f"results: {results_path}")
    echo(f"  {results_report['n_rows_used']} rows, {results_report['n_models']} models, "
         f"{results_report['n_configs']} configurations; held-out rows excluded: "
         f"{results_report['held_out_rows_excluded']}")
    _echo_held_out(echo, results_report, manifest.report)
    for family, info in roster["families"].items():
        if info["absent_from_results"]:
            echo(f"  [warn] {family}: mapped models absent from the results: "
                 f"{info['absent_from_results']}")
    _echo_missing_omega(echo, omega)

    diagnostics = model_score_diagnostics(results, omega)
    dominant = dominant_model(diagnostics)
    lomo = leave_one_model_out(results, omega, n_resamples, seed)
    tables = {
        # recorded (legacy) format; compared byte for byte with docs/results/k2/
        "summary_stats.csv": summary_stats(results, omega),
        "within_family_sensitivity.csv": recorded_within_family_sensitivity(results, omega),
        "joined_table.csv": joined_table(results, omega),
        # first hardening run
        "correlations_dataset_level.csv": dataset_level_correlations(results, omega, n_resamples, seed),
        "sign_counts.csv": dataset_sign_counts(results, omega),
        "dataset_deltas.csv": dataset_deltas(results, omega),
        "correlations_config_level.csv": config_level_correlations(results, omega, n_resamples, seed),
        "joined_config_table.csv": joined_config_table(results, omega),
        # sensitivity tables beside the headline
        "mse_sensitivity.csv": mse_sensitivity(results, omega, dominant, n_resamples, seed),
        "leave_one_model_out.csv": lomo,
        "leave_one_model_out_summary.csv": leave_one_model_out_summary(lomo),
        "model_score_diagnostics.csv": diagnostics,
        "paper_like_for_like.csv": paper_like_for_like(results, omega, n_resamples, seed),
        "within_family_sensitivity_relabelled.csv": within_family_relabelled(results, omega),
    }
    join_report = config_join_report(results, omega)
    degenerate = degenerate_cells(tables)

    echo("\n" + "=" * 68)
    echo("K2a (sMAPE, MSE) and K2b (MAE = 2x pinball at tau=0.5)")
    echo("Negative rho = zero-shot advantage grows with Omega")
    echo("=" * 68)
    echo(tables["summary_stats.csv"].to_string(index=False))
    head = tables["correlations_dataset_level.csv"]
    head = head[(head.within_family == "mean") & (head.pooling == "rows")]
    echo(f"\n{int(100 * DEFAULT_LEVEL)}% percentile bootstrap intervals "
         f"({n_resamples} resamples of base datasets, seed {seed}):")
    echo(head[["metric", "comparison", "n", "spearman_rho", "ci_low", "ci_high",
               "n_degenerate"]].round(3).to_string(index=False))

    share = diagnostics[(diagnostics.metric == "MSE") & (diagnostics.model == dominant)].iloc[0]
    echo("\nDo not cite the MSE rows above without this table. The MSE[mean] family mean of the "
         f"zero-shot family is dominated by one model, {dominant}: median share of the family sum "
         f"{share.median_share_of_family_sum:.3f} (an even share is {share.even_share:.3f}), "
         f"highest zero-shot MSE on {share.n_datasets_highest_in_family} of {share.n_datasets} "
         "datasets.")
    echo(tables["mse_sensitivity.csv"][
        ["variant", "comparison", "n", "spearman_rho", "ci_low", "ci_high", "n_zero_shot_lower"]]
        .round(3).to_string(index=False))
    echo(f"\nLike for like with the paper ({PAPER_SOURCE}; its family comparison is on "
         f"{PAPER_METRIC} only):")
    echo(tables["paper_like_for_like.csv"][
        ["comparison", "paper_rho", "replication_rho", "ci_low", "ci_high", "n"]]
        .round(3).to_string(index=False))
    summary = tables["leave_one_model_out_summary.csv"]
    echo("\nLeave one model out (range of rho when each model is removed in turn):")
    echo(summary[["metric", "comparison", "family_of_removed_model", "rho_all_models", "rho_min",
                  "rho_max", "model_at_max_abs_change", "n_sign_changes"]]
         .round(3).to_string(index=False))
    if degenerate:
        echo(f"\nWARNING: in {len(degenerate)} table row(s) more than "
             f"{100 * DEGENERATE_WARN_FRACTION:.0f}% of the bootstrap resamples were degenerate "
             "(Omega or Delta constant in the resample) and were left out of the interval. "
             "These intervals rest on fewer resamples than requested:")
        for cell in degenerate:
            echo(f"  {cell['table']}: {cell['row']}: {cell['n_degenerate']} of "
                 f"{cell['n_resamples']} ({100 * cell['share']:.1f}%)")

    parent = os.path.dirname(outdir)
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=os.path.basename(outdir) + ".partial-", dir=parent)
    try:
        for name, table in tables.items():
            table.to_csv(os.path.join(tmp, name), index=False, lineterminator="\n")

        figure_record = {"requested": bool(figure), "written": False, "reason": "not requested"}
        if figure:
            panels = {m: join_omega_by_base(family_table(results, METRICS[m]), omega)
                      for m in ("sMAPE", "MSE")}
            try:
                write_replication_figure(panels, os.path.join(tmp, FIGURE_NAME))
                figure_record.update(written=True, reason="")
            except FigureUnavailable as exc:
                figure_record["reason"] = str(exc)
                echo(f"\nfigure not written: {exc}")

        outputs = {name: sha256_file(os.path.join(tmp, name)) for name in sorted(os.listdir(tmp))}
        run_manifest = {
            "analysis": "K2a/K2b replication of the Omega relationship (no forecasting model is run)",
            "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "command": {"orig_argv": list(getattr(sys, "orig_argv", [])),
                        "argv": list(command) if command is not None else list(sys.argv),
                        "cwd": os.getcwd(), "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
            "parameters": {"omega_col": omega_col, "manifest_version": manifest_version,
                           "allow_missing_omega": bool(allow_missing_omega),
                           "bootstrap_resamples": int(n_resamples), "bootstrap_seed": int(seed),
                           "ci_level": DEFAULT_LEVEL, "reference_family": REFERENCE_FAMILY,
                           "comparisons": list(COMPARISONS), "metrics": dict(METRICS),
                           "sensitivity_metrics": dict(SENSITIVITY_METRICS),
                           "mse_variants": [v[0] for v in MSE_VARIANTS],
                           "degenerate_warn_fraction": DEGENERATE_WARN_FRACTION},
            "inputs": {"results": _input_record(results_path),
                       "manifest": _input_record(manifest_path)},
            "manifest_file": file_record,
            "code_sha256": _code_record(),
            "environment": env,
            "held_out": {"rule": STRICT_RULE,
                         "fuzzy_rule": FUZZY_RULE,
                         "tokens": results_report["held_out_tokens"],
                         "manifest_held_out_strict": manifest.report.get("held_out_listed", []),
                         "manifest_fields_checked": manifest.report.get("held_out_fields_checked", []),
                         "results_rows_excluded": results_report["held_out_rows_excluded"],
                         "results_datasets_excluded": results_report["held_out_datasets_excluded"],
                         "results_exclusions": results_report["held_out_exclusions"],
                         "results_fuzzy_reports": results_report["held_out_fuzzy_reports"],
                         "manifest_rows_found": manifest.report.get("held_out_rows_found", 0),
                         "manifest_fuzzy_reports": manifest.report.get("held_out_fuzzy_reports", [])},
            "results_report": results_report,
            "manifest_report": manifest.report,
            "omega_report": omega.report,
            "config_join_report": join_report,
            "roster": roster,
            "model_to_family": model_to_family(),
            "family_groups": {k: list(v) for k, v in GROUPS.items()},
            "summary_labels": {f"{agg}/{pooling}": label
                               for (agg, pooling), label in SUMMARY_LABELS.items()},
            "dominant_model": {"metric": "MSE", "family": REFERENCE_FAMILY, "model": dominant,
                               "rule": "largest median share of the family's summed score "
                                       "across the analysed base datasets"},
            "degenerate_resample_warnings": degenerate,
            "figure": figure_record,
            "outputs_sha256": outputs,
        }
        with open(os.path.join(tmp, MANIFEST_NAME), "w", encoding="utf-8") as f:
            json.dump(run_manifest, f, indent=2, sort_keys=True, default=_json_default)
            f.write("\n")
        os.chmod(tmp, 0o755)
        if os.path.lexists(outdir):
            raise OutputExistsError(f"output directory appeared during the run: {outdir}")
        os.rename(tmp, outdir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    echo(f"\nwrote {len(tables)} tables and {MANIFEST_NAME} to {outdir}")
    return run_manifest


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")
