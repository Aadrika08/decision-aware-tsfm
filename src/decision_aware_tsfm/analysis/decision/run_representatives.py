"""Choose one representative model per family from a forecast cache and freeze it (#41).

    PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_representatives BUNDLE \
        --variant current --out artifacts/runs/representatives/representatives.json

BUNDLE is a producer handoff bundle (a directory with handoff.json; read with
every hash checked, --variant names the Omega variant its reader verifies) or a
producer-v1 parquet file or directory. Families come from the cache's own
family column.

    --rule fixed_named --classical MODEL --tsfm MODEL    name the representatives
    --loss raw                                           average unscaled loss
    --fixture                                            the SYNTHETIC roster fixture

Only validation origins are scored (representatives.py). Run again with the
same input and --out, the command reproduces the file and leaves it alone; a
file that would differ is never overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .fixture import FIXTURE_SCHEMA, PRODUCER_SCHEMA, ROSTER_FAMILY_OF_MODEL, make_roster_fixture
from .handoff import read_handoff
from .representatives import (LOSSES, RULES, best_of_family, content_sha256, fixed_named, freeze,
                              load_frozen, score_table)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache", nargs="?", type=Path, help="handoff bundle, or producer-v1 parquet file/directory")
    ap.add_argument("--fixture", action="store_true", help="use the synthetic roster fixture instead of a cache")
    ap.add_argument("--seed", type=int, default=7, help="seed of the synthetic fixture")
    ap.add_argument("--variant", help="Omega variant of a handoff bundle, e.g. current or last_4096")
    ap.add_argument("--rule", choices=RULES, default="best_of_family")
    ap.add_argument("--classical", help="classical representative (fixed_named only)")
    ap.add_argument("--tsfm", help="TSFM representative (fixed_named only)")
    ap.add_argument("--loss", choices=LOSSES, default="scaled")
    ap.add_argument("--val-fraction", type=float, default=1 / 3)
    ap.add_argument("--out", type=Path, required=True, help="the frozen JSON file to write")
    a = ap.parse_args(argv)
    if a.fixture == (a.cache is not None):
        ap.error("give a cache path or --fixture, not both and not neither")
    named = {"classical": a.classical, "tsfm": a.tsfm}
    if a.rule == "fixed_named" and not all(named.values()):
        ap.error("--rule fixed_named needs both --classical and --tsfm")
    if a.rule != "fixed_named" and any(named.values()):
        ap.error("--classical and --tsfm are only for --rule fixed_named")
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

    if a.fixture:
        forecasts, _ = make_roster_fixture(a.seed)
        schema, family_of_model = FIXTURE_SCHEMA, ROSTER_FAMILY_OF_MODEL
        source = dict(kind="synthetic roster fixture", fixture=True, seed=a.seed)
    else:
        if (a.cache / "handoff.json").is_file():
            if not a.variant:
                ap.error("--variant is required for a handoff bundle")
            forecasts, _, report = read_handoff(a.cache, a.variant)
            source = dict(kind="handoff bundle", fixture=report["fixture"], omega_variant=a.variant,
                          handoff_sha256=report["handoff_sha256"],
                          partitions_hash_verified=report["partitions_hash_verified"])
        else:
            forecasts = pd.read_parquet(a.cache)
            source = dict(kind="producer-v1 parquet", fixture=False)
        if "family" not in forecasts.columns:
            raise SystemExit("the cache has no family column; a producer-v1 cache labels every model")
        schema = PRODUCER_SCHEMA
        family_of_model = forecasts.groupby("model")["family"].first().to_dict()

    kw = dict(val_fraction=a.val_fraction, loss=a.loss, source=source)
    payload = (fixed_named(forecasts, schema, family_of_model, named, **kw) if a.rule == "fixed_named"
               else best_of_family(forecasts, schema, family_of_model, **kw))
    if a.out.exists():
        if load_frozen(a.out) != payload:
            raise SystemExit(f"{a.out} exists and holds a different selection; a frozen file is not "
                             "overwritten. Choose a new --out")
        state = "reproduced, left unchanged"
    else:
        freeze(payload, a.out)
        state = "written"

    if source["fixture"]:
        print("SYNTHETIC FIXTURE: checks the interface only, not a research result.")
    inp, sel = payload["input"], payload["selection"]
    print(f"input: {inp['n_rows']:,} forecast rows, {len(inp['datasets'])} datasets, horizons "
          f"{inp['horizons']}, seeds {inp['seeds']}; fingerprint {inp['sha256'][:12]}")
    print(f"origins by role: {payload['n_origins_by_role']}; validation fraction "
          f"{payload['val_fraction']:.4f}")
    print(f"rule {payload['rule']}, loss {sel['loss']}, tau {sel['tau']}. {payload['origins_statement']}")
    print("\nscore per candidate (mean over the units every candidate of the family covers):")
    print(score_table(payload).to_string(index=False))
    print()
    for fam, model in payload["representatives"].items():
        best = payload["best_by_loss"][fam]
        print(f"{fam}: {model} ({payload['reasons'][fam]}); lowest scaled loss {best['scaled']}, "
              f"lowest raw loss {best['raw']}; {payload['coverage'][fam]}")
    print(f"\nfrozen file {a.out} ({state}); sha256 {content_sha256(payload)}")


if __name__ == "__main__":
    main()
