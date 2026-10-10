"""Configured forecast planning, execution, audit, and measured budgeting."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from .budget import estimate_budget
from .fixtures import create_fixture
from .planner import plan


def read_json(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_pairs,
                      parse_constant=invalid_constant)


def atomic_output(path, text, *, refuse_existing=False):
    path = Path(path)
    if refuse_existing and path.exists():
        raise ValueError("Refusing to overwrite an existing immutable plan output")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if refuse_existing:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise ValueError("Refusing to overwrite an existing immutable plan output") from error
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("fixture", help="Create marked invented inputs in a new directory")
    fixture.add_argument("directory", type=Path)
    for name in ("plan", "generate", "audit", "budget"):
        command = commands.add_parser(name)
        command.add_argument("--workload", type=Path, required=name == "plan")
        command.add_argument("--models", type=Path, required=name == "plan")
        if name != "plan":
            command.add_argument("--plan-file", type=Path, help="Use an immutable saved plan, including its original provenance")
        command.add_argument("--output", type=Path)
        if name in ("generate", "audit"):
            command.add_argument("--cache", type=Path, required=True)
        if name == "generate":
            command.add_argument("--workers", type=int, default=2, help="Bounded CPU worker count")
            command.add_argument("--limit-jobs", type=int)
            command.add_argument("--repair", action="store_true", help="Quarantine corrupt files before regenerating")
            command.add_argument("--real", action="store_true", help="Run only explicitly approved, qualified real inputs")
        if name == "budget":
            command.add_argument("--measurements", type=Path, nargs="*", default=[],
                                 help="Run reports with measurements, or JSON lists of measurements")
            command.add_argument("--hardware", type=Path, help="JSON cpu/cuda hardware labels")
            command.add_argument("--overheads", type=Path, help="JSON loading_seconds/io_seconds by cpu/cuda")
            command.add_argument("--restart-fraction", type=float, default=0.0)
            command.add_argument("--failure-fraction", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        if getattr(args, "output", None):
            protected = [getattr(args, name, None) for name in
                         ("workload", "models", "plan_file", "hardware", "overheads")]
            protected.extend(getattr(args, "measurements", []))
            if args.output.resolve() in {path.resolve() for path in protected if path}:
                raise ValueError("--output must not overwrite an input file")
        if args.command == "fixture":
            result = create_fixture(args.directory)
        else:
            if getattr(args, "plan_file", None):
                if args.workload or args.models:
                    raise ValueError("Use either --plan-file or --workload with --models")
                planned = read_json(args.plan_file)
            else:
                if not args.workload or not args.models:
                    raise ValueError("Supply --plan-file or both --workload and --models")
                planned = plan(read_json(args.workload), read_json(args.models), base_dir=args.workload.resolve().parent)
            if getattr(args, "output", None) and args.output.resolve() in {
                    Path(job["source_path"]).resolve() for job in planned["jobs"]}:
                raise ValueError("--output must not overwrite a staged source file")
            if (getattr(args, "output", None) and getattr(args, "cache", None)
                    and args.output.resolve().is_relative_to(args.cache.resolve())):
                raise ValueError("--output must be outside --cache")
            if args.command == "plan":
                result = planned
            elif args.command in ("generate", "audit"):
                from .executor import audit, generate
                result = (audit(planned, args.cache) if args.command == "audit" else
                          generate(planned, args.cache, workers=args.workers, repair=args.repair,
                                   limit_jobs=args.limit_jobs, real=args.real))
            else:
                measurements = []
                for path in args.measurements:
                    report = read_json(path)
                    records = report.get("measurements") if isinstance(report, dict) else report
                    if not isinstance(records, list):
                        raise ValueError(f"Expected a measurements list in {path}")
                    report_run_id = report.get("run_id") if isinstance(report, dict) else None
                    for record in records:
                        if not isinstance(record, dict):
                            raise ValueError(f"Expected measurement objects in {path}")
                        record = dict(record)
                        if report_run_id is not None:
                            if record.get("run_id", report_run_id) != report_run_id:
                                raise ValueError(f"Measurement run_id conflicts with report {path}")
                            record["run_id"] = report_run_id
                        measurements.append(record)
                overheads = read_json(args.overheads) if args.overheads else {}
                result = estimate_budget(planned["jobs"], measurements,
                    hardware=read_json(args.hardware) if args.hardware else {},
                    loading_seconds=overheads.get("loading_seconds", {}),
                    io_seconds=overheads.get("io_seconds", {}),
                    restart_fraction=args.restart_fraction, failure_fraction=args.failure_fraction)
        output = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if getattr(args, "output", None):
            atomic_output(args.output, output, refuse_existing=args.command == "plan")
        print(output, end="")
        return 2 if result.get("status") in ("incomplete", "failed") else 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"forecast-cache: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
