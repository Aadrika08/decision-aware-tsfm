#!/usr/bin/env python3
"""
K2a / K2b replication, self-contained.

K2a: does the published Omega relationship reproduce on the source paper's own
     metrics (sMAPE, MSE)?
K2b: does it survive the move to pinball loss at tau = 0.5?

No forecasting model is run. All scores come from the published GIFT-Eval
per-model result files.

Usage
-----
    # first time: download the published results (~3 MB, 129 models)
    python run_k2.py --fetch-results

    # run the analysis
    python run_k2.py --manifest path/to/dev_manifest.json

    # everything, with outputs written to ./out
    python run_k2.py --manifest ../data/dev_manifest.json --outdir out --figure

Requirements: pandas, numpy, scipy, matplotlib (only for --figure), requests
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(HERE, "gift_results.csv")
DEFAULT_MANIFEST = os.path.join(HERE, "dev_manifest.json")
RESULTS_SPACE = "https://huggingface.co/spaces/Salesforce/GIFT-Eval"

# ---------------------------------------------------------------------------
# Model families. Taken verbatim from the source paper's Appendix A.1
# (arXiv:2511.08884), which uses GIFT-Eval's own four-way taxonomy.
# "zero-shot" = published weights carry no leakage against the benchmark.
# "pretrained" = had to be re-pretrained on a leakage-free corpus first.
# The paper's Omega relationship is reported for the zero-shot family only.
# ---------------------------------------------------------------------------
GROUPS = {
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

METRICS = {
    "sMAPE": "eval_metrics/sMAPE[0.5]",
    "MSE": "eval_metrics/MSE[mean]",
    "MAE": "eval_metrics/MAE[0.5]",             # pinball at tau=0.5 == MAE / 2
    "CRPS": "eval_metrics/mean_weighted_sum_quantile_loss",
}

# Dayson's stage0prep subset directory names -> GIFT-Eval (base dataset, frequency).
# GIFT-Eval configs are base/frequency/term; one raw directory backs all three terms.
SUBSET_MAP = {
    "bitbrains_fast_storage_5T": ("bitbrains_fast_storage", "5T"),
    "bitbrains_fast_storage_H": ("bitbrains_fast_storage", "H"),
    "bitbrains_rnd_5T": ("bitbrains_rnd", "5T"),
    "bitbrains_rnd_H": ("bitbrains_rnd", "H"),
    "bizitobs_application": ("bizitobs_application", "10S"),
    "bizitobs_l2c_5T": ("bizitobs_l2c", "5T"),
    "bizitobs_l2c_H": ("bizitobs_l2c", "H"),
    "bizitobs_service": ("bizitobs_service", "10S"),
    "electricity_15T": ("electricity", "15T"),
    "electricity_H": ("electricity", "H"),
    "electricity_D": ("electricity", "D"),
    "ett1_15T": ("ett1", "15T"), "ett1_H": ("ett1", "H"), "ett1_D": ("ett1", "D"),
    "ett2_15T": ("ett2", "15T"), "ett2_H": ("ett2", "H"), "ett2_D": ("ett2", "D"),
    "hierarchical_sales_D": ("hierarchical_sales", "D"),
    "jena_weather_10T": ("jena_weather", "10T"),
    "jena_weather_H": ("jena_weather", "H"),
    "jena_weather_D": ("jena_weather", "D"),
    "kdd_cup_2018_with_missing_H": ("kdd_cup_2018", "H"),
    "LOOP_SEATTLE_5T": ("loop_seattle", "5T"),
    "LOOP_SEATTLE_H": ("loop_seattle", "H"),
    "M_DENSE_H": ("m_dense", "H"),
    "m4_daily": ("m4_daily", "D"),
    "m4_hourly": ("m4_hourly", "H"),
    "m4_monthly": ("m4_monthly", "M"),
    "m4_weekly": ("m4_weekly", "W"),
    "saugeenday_D": ("saugeen", "D"),
    "saugeenday_W": ("saugeen", "W"),
    "solar_10T": ("solar", "10T"),
    "solar_H": ("solar", "H"),
    "SZ_TAXI_15T": ("sz_taxi", "15T"),
    "us_births_D": ("us_births", "D"),
    "us_births_W": ("us_births", "W"),
}
# "jena_weather" with no frequency suffix is a duplicate directory of
# jena_weather_10T (identical length and Omega). Deliberately not mapped.


def fetch_results(out_path):
    """Download and concatenate every published per-model result file."""
    import requests
    print(f"Listing models at {RESULTS_SPACE} ...")
    tree = requests.get(
        "https://huggingface.co/api/spaces/Salesforce/GIFT-Eval/tree/main/results",
        timeout=60).json()
    names = [e["path"].split("/")[-1] for e in tree if e["type"] == "directory"]
    print(f"  {len(names)} model directories")
    frames, failed = [], []
    for i, name in enumerate(names, 1):
        url = (f"https://huggingface.co/spaces/Salesforce/GIFT-Eval/"
               f"resolve/main/results/{name}/all_results.csv")
        try:
            d = pd.read_csv(url)
            if "dataset" not in d.columns:
                failed.append(name); continue
            frames.append(d)
        except Exception:
            failed.append(name)
        if i % 25 == 0:
            print(f"  {i}/{len(names)}")
    r = pd.concat(frames, ignore_index=True)
    r.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(r)} rows, {r.model.nunique()} models, "
          f"{r.dataset.nunique()} configs")
    if failed:
        print(f"  skipped (no parseable results file): {failed}")


def load_omega(manifest_path):
    """One Omega per base dataset, averaged over that dataset's series and channels."""
    with open(manifest_path) as f:
        raw = json.load(f)
    a = pd.DataFrame(raw["active_dev_series"])
    unmapped = sorted(set(a.subset) - set(SUBSET_MAP) - {"jena_weather"})
    if unmapped:
        print(f"  [warn] subsets in manifest with no SUBSET_MAP entry, dropped: {unmapped}")
    a = a[a.subset.isin(SUBSET_MAP)].copy()
    a["base"] = a.subset.map(lambda s: SUBSET_MAP[s][0])
    return a.groupby("base").agg(
        omega_levels=("omega_levels", "mean"),
        omega_diffs=("omega_diffs", "mean"),
        omega_sd=("omega_levels", "std"),
        n_series=("series_id", "count"),
        median_length=("length", "median"),
    ).reset_index()


def family_table(results, metric_col, within_family="mean"):
    """Per base dataset, one score per model family."""
    fam = {m: k for k, v in GROUPS.items() for m in v}
    r = results.copy()
    r["family"] = r.model.map(fam)
    r = r[r.family.notna()]
    return (r.groupby(["base", "family"])[metric_col]
             .agg(within_family).unstack("family"))


def spearman(tab, A, B):
    d = 100 * (tab[A] - tab[B]) / tab[A]
    ok = np.isfinite(d) & np.isfinite(tab["omega"])
    if ok.sum() < 3:
        return np.nan, np.nan, int(ok.sum())
    rho, p = stats.spearmanr(tab["omega"][ok], d[ok])
    return rho, p, int(ok.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="dev_manifest.json from stage0prep (default: ./dev_manifest.json)")
    ap.add_argument("--results", default=DEFAULT_RESULTS,
                    help="concatenated GIFT-Eval results (default: ./gift_results.csv)")
    ap.add_argument("--outdir", default=".", help="where to write CSV outputs")
    ap.add_argument("--omega-col", default="omega_levels",
                    choices=["omega_levels", "omega_diffs"])
    ap.add_argument("--fetch-results", action="store_true",
                    help="download the published results, then exit")
    ap.add_argument("--figure", action="store_true", help="also write the replication figure")
    args = ap.parse_args()

    if args.fetch_results:
        fetch_results(args.results); return

    for path, what in [(args.results, "results"), (args.manifest, "manifest")]:
        if not os.path.exists(path):
            hint = "  run with --fetch-results first" if what == "results" else \
                   "  pass --manifest /path/to/dev_manifest.json"
            sys.exit(f"ERROR: {what} not found at {path}\n{hint}")
    os.makedirs(args.outdir, exist_ok=True)

    results = pd.read_csv(args.results)
    results["base"] = results.dataset.str.split("/").str[0]
    om = load_omega(args.manifest)
    print(f"\nmanifest: {args.manifest}")
    print(f"  {int(om.n_series.sum())} series across {len(om)} base datasets")

    # ---- K2a / K2b -------------------------------------------------------
    rows = []
    for mname, mcol in METRICS.items():
        t = family_table(results, mcol).merge(
            om.set_index("base")[[args.omega_col]].rename(
                columns={args.omega_col: "omega"}),
            left_index=True, right_index=True)
        for B in ["deep", "statistical", "pretrained"]:
            rho, p, n = spearman(t, "zero-shot", B)
            rows.append(dict(metric=mname, comparison=f"zero-shot vs {B}",
                             n=n, spearman_rho=round(rho, 3), p_value=round(p, 4)))
    summary = pd.DataFrame(rows)
    print("\n" + "=" * 68)
    print("K2a (sMAPE, MSE) and K2b (MAE = 2x pinball at tau=0.5)")
    print("Negative rho = zero-shot advantage grows with Omega (published direction)")
    print("=" * 68)
    print(summary.to_string(index=False))
    print("\npublished: -0.556 vs deep, -0.234 vs statistical, pretrained between")
    summary.to_csv(os.path.join(args.outdir, "summary_stats.csv"), index=False)

    # ---- within-family sensitivity ---------------------------------------
    print("\n" + "=" * 68)
    print("Sensitivity to the within-family summary (sMAPE, zero-shot vs deep)")
    print("=" * 68)
    sens = []
    for agg, label in [("mean", "mean across family (paper's choice)"),
                       ("median", "median model in family"),
                       ("min", "best model in family")]:
        t = family_table(results, METRICS["sMAPE"], agg).merge(
            om.set_index("base")[[args.omega_col]].rename(
                columns={args.omega_col: "omega"}),
            left_index=True, right_index=True)
        rho, p, n = spearman(t, "zero-shot", "deep")
        sens.append(dict(within_family=label, n=n,
                         spearman_rho=round(rho, 3), p_value=round(p, 4)))
        print(f"  {label:<38} rho = {rho:+.3f}   p = {p:.3f}")
    pd.DataFrame(sens).to_csv(
        os.path.join(args.outdir, "within_family_sensitivity.csv"), index=False)

    # ---- joined table ----------------------------------------------------
    fam = {m: k for k, v in GROUPS.items() for m in v}
    r2 = results.copy(); r2["family"] = r2.model.map(fam); r2 = r2[r2.family.notna()]
    long = r2.groupby(["base", "family"])[list(METRICS.values())].mean().reset_index()
    long.columns = ["dataset", "model_family"] + list(METRICS)
    joined = long.merge(om.rename(columns={"base": "dataset"}), on="dataset")
    joined["pinball_tau0.5"] = joined["MAE"] / 2
    joined.round(5).to_csv(os.path.join(args.outdir, "joined_table.csv"), index=False)
    print(f"\nwrote summary_stats.csv, within_family_sensitivity.csv, joined_table.csv "
          f"to {os.path.abspath(args.outdir)}")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for ax, mname in zip(axes, ["sMAPE", "MSE"]):
            t = family_table(results, METRICS[mname]).merge(
                om.set_index("base")[[args.omega_col]].rename(
                    columns={args.omega_col: "omega"}),
                left_index=True, right_index=True)
            for B, c, lbl in [("deep", "#c0392b", "vs deep-learning"),
                              ("statistical", "#2471a3", "vs statistical")]:
                d = 100 * (t["zero-shot"] - t[B]) / t["zero-shot"]
                ok = np.isfinite(d)
                rho, p, _ = spearman(t, "zero-shot", B)
                ax.scatter(t.omega[ok], d[ok], s=38, color=c, alpha=.8,
                           label=f"{lbl}  rho={rho:+.3f}, p={p:.3f}")
                z = np.polyfit(t.omega[ok], d[ok], 1)
                xs = np.linspace(0.05, 0.8, 50)
                ax.plot(xs, np.polyval(z, xs), color=c, lw=1.2, ls="--", alpha=.7)
            ax.axhline(0, color="#555", lw=.8)
            ax.axvline(0.4, color="#999", lw=.8, ls=":")
            ax.axvline(0.5, color="#999", lw=.8, ls=":")
            ax.set_xlabel("Spectral predictability $\\Omega$")
            ax.set_ylabel("$\\Delta$ = 100(A-B)/A   negative = zero-shot better")
            ax.set_title(f"K2a replication on {mname}  (n={len(t)} datasets)", fontsize=11)
            ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=.25)
        plt.tight_layout()
        out = os.path.join(args.outdir, "replication_figure.png")
        plt.savefig(out, dpi=180)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
