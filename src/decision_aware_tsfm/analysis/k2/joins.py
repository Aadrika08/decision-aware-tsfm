"""Joining a per-family score table to Omega, and the correlation row every table shares."""
from __future__ import annotations

import pandas as pd

from .loading import OmegaTables
from .statistic import bootstrap_interval, spearman_xy

# How a GIFT-Eval configuration (base/frequency/term) is given an Omega value.
#   dataset_frequency : Omega of the manifest subset with the same base dataset and
#                       frequency. Configurations whose frequency subset has no
#                       Omega in the manifest do not join.
#   dataset_broadcast : the base dataset's Omega (the one the headline analysis
#                       uses) copied to every configuration of that base dataset,
#                       including frequencies the manifest has no Omega for.
CONFIG_MATCHINGS: tuple[str, ...] = ("dataset_frequency", "dataset_broadcast")


def join_omega_by_base(table: pd.DataFrame, omega: OmegaTables) -> pd.DataFrame:
    """Family table indexed by base dataset, inner-joined to that dataset's Omega."""
    om = omega.by_base.set_index("base")[[omega.omega_col]].rename(
        columns={omega.omega_col: "omega"})
    return table.merge(om, left_index=True, right_index=True)


def config_keys(results: pd.DataFrame) -> pd.DataFrame:
    return (results[["dataset", "base", "freq", "term"]].drop_duplicates()
            .sort_values("dataset").reset_index(drop=True))


def join_omega_by_config(table: pd.DataFrame, keys: pd.DataFrame, omega: OmegaTables,
                         matching: str) -> pd.DataFrame:
    """Family table indexed by configuration, inner-joined to Omega under one matching rule."""
    t = keys.merge(table.reset_index(), on="dataset", how="inner")
    if matching == "dataset_frequency":
        om = omega.by_base_freq[["base", "freq", "omega"]]
        return t.merge(om, on=["base", "freq"], how="inner", validate="m:1")
    if matching == "dataset_broadcast":
        om = omega.by_base[["base", omega.omega_col]].rename(columns={omega.omega_col: "omega"})
        return t.merge(om, on="base", how="inner", validate="m:1")
    raise ValueError(f"matching must be one of {CONFIG_MATCHINGS}, got {matching!r}")


def config_join_report(results: pd.DataFrame, omega: OmegaTables) -> dict:
    """How many configurations get an Omega value under each matching rule."""
    keys = config_keys(results)
    report = {"n_configs_in_results": int(len(keys)),
              "n_base_datasets_in_results": int(keys.base.nunique()), "matchings": {}}
    for matching in CONFIG_MATCHINGS:
        joined = join_omega_by_config(keys[["dataset"]].set_index("dataset"), keys, omega, matching)
        report["matchings"][matching] = {
            "n_configs_joined": int(len(joined)),
            "n_base_datasets_joined": int(joined.base.nunique()),
            "n_distinct_omega_values": int(joined.omega.nunique()),
            "configs_not_joined": sorted(set(keys.dataset) - set(joined.dataset)),
        }
    return report


def correlation_row(x, y, n_resamples, seed, clusters=None) -> dict:
    """rho, p, n and the bootstrap interval for one table cell."""
    rho, p, n = spearman_xy(x, y)
    ci = bootstrap_interval(x, y, n_resamples=n_resamples, seed=seed, clusters=clusters)
    return dict(n=n, spearman_rho=rho, p_value=p, ci_low=ci["ci_low"], ci_high=ci["ci_high"],
                ci_level=ci["ci_level"], n_resampled_units=ci["n_resampled_units"],
                n_resamples=ci["n_resamples"], n_degenerate=ci["n_degenerate"], seed=ci["seed"])
