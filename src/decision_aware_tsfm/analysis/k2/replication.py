#!/usr/bin/env python3
"""
K2a / K2b replication: entry point.

K2a: does the published Omega relationship reproduce? The source paper
     (arXiv:2511.08884) computes its family comparison on sMAPE only, so the
     like-for-like replication is the sMAPE one. Its published -0.556 (zero-shot
     versus deep learning) is an sMAPE figure. MSE appears in the paper only in
     its controlled experiments; the MSE rows here are an extension, and they
     depend on one model's MSE[mean] column (see mse_sensitivity.csv).
K2b: does it survive the move to pinball loss at tau = 0.5?

The paper does not say how a family score is formed from its models for these
correlations. The family mean used here is this replication's choice.

No forecasting model is run. All scores come from the published GIFT-Eval
per-model result files.

Usage (one command regenerates every output into a new directory):

    PYTHONPATH=src python -m decision_aware_tsfm.analysis.k2.replication --outdir artifacts/runs/<new name>

The code lives in sibling modules, one job each:

    loading.py      results file, manifest reader (versioned), Omega tables, held-out guard
    heldout.py      held-out name checks (strict, acted on; fuzzy, reported only)
    families.py     model-to-family mapping, metric columns, per-family score table
    statistic.py    Delta, Spearman correlation, bootstrap interval, sign counts
    joins.py        joining family scores to Omega
    sensitivity.py  MSE sensitivity, leave one model out, like-for-like with the paper
    pipeline.py     every output table and the run manifest
    environment.py  running versions against the team pins
    figure.py       optional figure (matplotlib is not a pinned dependency)
    fetch.py        download of the published results (network)
    cli.py          argument parsing

This module keeps the names that scripts/run_k2.py, the console script and
earlier notes refer to.
"""
from .cli import main
from .families import GROUPS, METRICS, family_table
from .fetch import RESULTS_SPACE, fetch_results
from .loading import SUBSET_MAP, aggregate_omega, read_manifest
from .pipeline import DEFAULT_MANIFEST, DEFAULT_RESULTS, PROJECT_ROOT
from .statistic import spearman

__all__ = ["DEFAULT_MANIFEST", "DEFAULT_RESULTS", "GROUPS", "METRICS", "PROJECT_ROOT",
           "RESULTS_SPACE", "SUBSET_MAP", "family_table", "fetch_results", "load_omega",
           "main", "spearman"]


def load_omega(manifest_path, omega_col="omega_levels", version="dev-v1"):
    """One Omega per base dataset, averaged over that dataset's series and channels."""
    return aggregate_omega(read_manifest(manifest_path, version), omega_col).by_base


if __name__ == "__main__":
    main()
