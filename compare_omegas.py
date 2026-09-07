import json
import numpy as np
import pandas as pd


def analyze_omega_comparison(manifest_path: str = "dev_manifest.json"):
    with open(manifest_path, "r") as f:
        data = json.load(f)

    active_series = data.get("active_dev_series", [])
    df = pd.DataFrame(active_series)

    # Compute comparison metrics
    df["omega_delta"] = df["omega_levels"] - df["omega_diffs"]
    df["relative_drop_%"] = (
        df["omega_delta"] / df["omega_levels"].replace(0, np.nan)
    ) * 100

    print("=" * 65)
    print("SPECTRAL PREDICTABILITY COMPARISON (Omega_levels vs Omega_diffs)")
    print("=" * 65)
    print(f"Total Active Series Analyzed (N >= 1000): {len(df):,}\n")

    print(" MEAN & MEDIAN METRICS")
    print(f"  Mean Omega (Levels)     : {float(df['omega_levels'].mean()):.4f}")
    print(f"  Mean Omega (Diffs)      : {float(df['omega_diffs'].mean()):.4f}")
    print(f"  Mean Predictability Drop : {float(df['omega_delta'].mean()):.4f} (Levels - Diffs)")
    print(f"  Median Predictability Drop: {float(df['omega_delta'].median()):.4f}\n")

    print(" CATEGORICAL BREAKDOWN OF SERIES")
    drop_count = int((df["omega_delta"] > 0.05).sum())
    gain_count = int((df["omega_delta"] < -0.05).sum())
    similar_count = len(df) - drop_count - gain_count

    print(
        f"  Series where Levels > Diffs (Trend/Seasonal Dominant) : {drop_count:,} ({drop_count/len(df)*100:.1f}%)"
    )
    print(
        f"  Series where Diffs > Levels (Velocity/Momentum Signal): {gain_count:,} ({gain_count/len(df)*100:.1f}%)"
    )
    print(
        f"  Series where Levels approx Diffs (|Delta| <= 0.05)    : {similar_count:,} ({similar_count/len(df)*100:.1f}%)\n"
    )

    print(" SUBSET LEVEL COMPARISON (Top 10 Subsets)")
    subset_summary = (
        df.groupby("subset")
        .agg(
            count=("series_id", "count"),
            avg_levels=("omega_levels", "mean"),
            avg_diffs=("omega_diffs", "mean"),
            avg_drop=("omega_delta", "mean"),
        )
        .reset_index()
        .sort_values(by="count", ascending=False)
    )

    print(
        f"{'Subset Name':<35} | {'Count':<6} | {'Omega Lev':<9} | {'Omega Dif':<9} | {'Drop':<8}"
    )
    print("-" * 75)
    for _, row in subset_summary.head(10).iterrows():
        print(
            f"{str(row['subset'])[:34]:<35} | {int(row['count']):<6} | {float(row['avg_levels']):<9.4f} | {float(row['avg_diffs']):<9.4f} | {float(row['avg_drop']):<8.4f}"
        )

    print("=" * 65)


if __name__ == "__main__":
    analyze_omega_comparison()