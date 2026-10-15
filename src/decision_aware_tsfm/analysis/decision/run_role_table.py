"""Write the validation / gap / evaluation role table from the decision engine, or check one.

    PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_role_table export \
        --workload HANDOFF/workload.json --out artifacts/runs/role-table
    PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_role_table check \
        artifacts/runs/role-table --workload HANDOFF/workload.json

The origins come from exactly one of
    --workload FILE              the producer's workload.json
    --cache PATH --schema NAME   a forecast cache (producer-v1 or canonical-v1)
    --handoff DIR --variant V    a producer handoff bundle

export writes role_table.csv, decision_origins.csv, excluded_series_horizons.csv,
summary.csv and role_table.json to --out (a new directory). check reads a table (that
directory, or one CSV file) and exits non-zero if it differs from what the engine
assigns to the origins. Nothing here reads a forecast value or runs a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from . import role_table as rt
from .handoff import file_sha256


def _git_commit() -> str | None:
    done = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


def _tree_sha256(path: Path) -> str:
    """A file's hash, or for a directory the hash of its sorted (relative path, file hash) lines."""
    if path.is_file():
        return file_sha256(path)
    lines = [f"{p.relative_to(path).as_posix()} {file_sha256(p)}\n"
             for p in sorted(path.rglob("*")) if p.is_file()]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def read_origins(a):
    """(origins, provenance of the input) for the one input kind that was named."""
    if a.workload is not None:
        origins = rt.read_workload_origins(a.workload)
        source = dict(kind="workload", name=a.workload.name, sha256=file_sha256(a.workload))
    elif a.cache is not None:
        if a.schema is None:
            raise SystemExit("--cache needs --schema (producer-v1 or canonical-v1)")
        origins = rt.cache_origins(a.cache, a.schema)
        source = dict(kind="cache", name=a.cache.name, schema=a.schema, sha256=_tree_sha256(a.cache))
    else:
        if a.variant is None:
            raise SystemExit("--handoff needs --variant (the bundle reader checks one Omega variant)")
        origins, report = rt.handoff_origins(a.handoff, a.variant)
        source = dict(kind="handoff", name="handoff.json", variant=a.variant,
                      sha256=report["handoff_sha256"], fixture=report["fixture"])
    # fit-only datasets get no roles (record 0007, section 1): all their origins are used for fitting
    origins = origins[~origins["dataset"].isin(a.fit_only_datasets)]
    return origins, source


def _input_options(p):
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--workload", type=Path, help="producer workload.json")
    g.add_argument("--cache", type=Path, help="forecast cache (parquet file or directory)")
    g.add_argument("--handoff", type=Path, help="producer handoff bundle directory")
    p.add_argument("--schema", help="cache schema, with --cache")
    p.add_argument("--variant", help="Omega variant, with --handoff")
    p.add_argument("--val-fraction", type=float, default=1 / 3)
    p.add_argument("--fit-only-datasets", nargs="*", default=[])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export", help="write the role table the engine assigns")
    _input_options(ex)
    ex.add_argument("--out", type=Path, required=True)
    ck = sub.add_parser("check", help="compare a role table with the engine")
    ck.add_argument("table", type=Path, help="directory written by export, or a role table CSV")
    _input_options(ck)
    a = ap.parse_args(argv)

    if a.command == "export":
        if a.out.exists():
            raise SystemExit(f"--out {a.out} already exists; choose a new directory")
        origins, source = read_origins(a)
        result = rt.build_role_table(origins, a.val_fraction)
        sidecar = rt.write_role_table(result, a.out, dict(
            input=source, fit_only_datasets=sorted(a.fit_only_datasets), code_commit=_git_commit()))
        print(rt.summarize(result).to_string(index=False))
        print(json.dumps(dict(out=str(a.out), val_fraction=sidecar["val_fraction"], input=source,
                              **sidecar["counts"]), indent=2, sort_keys=True))
        return

    origins, source = read_origins(a)
    folder = a.table if a.table.is_dir() else None
    table = rt.read_csv(folder / rt.FILES["table"] if folder else a.table)
    side = {name: rt.read_csv(folder / rt.FILES[name]) if folder and (folder / rt.FILES[name]).is_file()
            else None for name in ("excluded", "decision_origins")}
    report = rt.check_role_table(table, origins, a.val_fraction, excluded=side["excluded"],
                                 decision_origins=side["decision_origins"])
    problems = [rt.describe_differences(report)] if not report["ok"] else []
    if folder and (folder / rt.SIDECAR).is_file():
        recorded = json.loads((folder / rt.SIDECAR).read_text(encoding="utf-8"))["val_fraction"]
        if recorded != a.val_fraction:
            problems.append(f"the table's sidecar records val_fraction {recorded}, the check used {a.val_fraction}")
    print(json.dumps(dict(
        ok=not problems, input=source, val_fraction=report["val_fraction"],
        table_rows=report["table_rows"], engine_rows=report["engine_rows"],
        matched_rows=report["matched_rows"], **{k: len(report[k]) for k in rt.DIFFERENCES},
        examples={k: report[k][:3] for k in rt.DIFFERENCES if report[k]}), indent=2, sort_keys=True))
    if problems:
        raise SystemExit("role table check FAILED: " + "; ".join(problems))
    print("role table check passed: the table and the engine agree on every row")


if __name__ == "__main__":
    main()
