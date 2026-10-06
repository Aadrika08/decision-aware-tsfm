"""Command-line entry for K2.

One command regenerates every K2 output from the saved inputs:

    PYTHONPATH=src python -m decision_aware_tsfm.analysis.k2.replication --outdir <new directory>

The output directory must not exist and must not lie inside an existing run
directory; a run is never overwritten. Use --omega-col to select a different
Omega column of the manifest (for example the 4,096-history variant once the
frozen manifest carries it) and --manifest-version to select the manifest
layout. The Omega column must be a float column with a value for every series;
if it is not, the command stops and names the base datasets affected.
"""
from __future__ import annotations

import argparse
import sys

from ..decision.guards import HeldOutLeakError
from .fetch import fetch_results
from .heldout import HeldOutListError
from .loading import (DEFAULT_MANIFEST_VERSION, DEFAULT_OMEGA_COLUMN, MANIFEST_VERSIONS,
                      ManifestLayoutError, OmegaColumnError, ResultsLayoutError)
from .pipeline import (DEFAULT_MANIFEST, DEFAULT_OUTDIR, DEFAULT_RESULTS, InvalidParameterError,
                       OutputExistsError, run_k2)
from .statistic import DEFAULT_RESAMPLES, DEFAULT_SEED

EXPECTED_ERRORS = (OutputExistsError, FileNotFoundError, FileExistsError, OmegaColumnError,
                   ManifestLayoutError, ResultsLayoutError, HeldOutLeakError, HeldOutListError,
                   InvalidParameterError, NotImplementedError)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="decision-aware-run-k2", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="Omega manifest (default: artifacts/reference/manifests/dev_manifest.json)")
    ap.add_argument("--manifest-version", default=DEFAULT_MANIFEST_VERSION,
                    choices=list(MANIFEST_VERSIONS),
                    help="layout of the manifest file (default: %(default)s)")
    ap.add_argument("--results", default=DEFAULT_RESULTS,
                    help="concatenated GIFT-Eval results (default: artifacts/reference/k2/gift_results.csv)")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR,
                    help="new directory for the outputs; must not exist (default: artifacts/runs/k2)")
    ap.add_argument("--omega-col", default=DEFAULT_OMEGA_COLUMN,
                    help="name of the manifest's Omega column to use; must be a float column "
                         "(default: %(default)s)")
    ap.add_argument("--allow-missing-omega", action="store_true",
                    help="continue when the Omega column has no value for some series; series "
                         "and datasets without a value are left out and the changed n is "
                         "printed (default: stop and list the datasets affected)")
    ap.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_RESAMPLES,
                    help="bootstrap resamples per interval, at least 1 (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="bootstrap seed, 0 or larger (default: %(default)s)")
    ap.add_argument("--fetch-results", action="store_true",
                    help="download the published results to --results, then exit")
    ap.add_argument("--figure", action="store_true",
                    help="also write the replication figure (needs matplotlib, which is not pinned)")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.fetch_results:
            fetch_results(args.results)
            return
        run_k2(results_path=args.results, manifest_path=args.manifest, outdir=args.outdir,
               omega_col=args.omega_col, manifest_version=args.manifest_version,
               n_resamples=args.bootstrap_resamples, seed=args.seed, figure=args.figure,
               allow_missing_omega=args.allow_missing_omega,
               command=list(sys.argv) if argv is None else ["decision-aware-run-k2", *argv])
    except EXPECTED_ERRORS as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
