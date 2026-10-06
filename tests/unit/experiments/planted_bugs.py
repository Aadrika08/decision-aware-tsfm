"""Plant one bug at a time and report whether the experiments test suite catches it.

    python tests/unit/experiments/planted_bugs.py            (from the tree root)
    python tests/unit/experiments/planted_bugs.py --only M1 M5 --jobs 2

This is a tool, not a test (unittest discovery does not collect it). It never
writes to the working tree: the package source and this test directory are
copied to a temporary folder, each bug is planted in the COPY, the suite runs
there, and the file is restored in a `finally` block and compared byte for
byte with the original before the next bug. At the end a clean run of the copy
must pass, and the three source files in the working tree must still have the
hashes they had at the start.

M1 to M16 are the independent reviewer's sixteen bugs, re-expressed against the
current code (same defect, current line). N1 to N9 were added with the fixes.
A bug counts as caught when the suite exits non-zero; the failing tests are
listed. "SURVIVED" means the suite passed with the bug in place.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG = "src/decision_aware_tsfm/analysis/experiments"
SOURCES = ("bootstrap.py", "e1.py", "e2.py")
SUITE = ["-m", "unittest", "discover", "-s", "tests/unit/experiments", "-t", "."]

# (id, what the bug is, file, text to find exactly once, replacement)
BUGS = [
    ("M1", "bootstrap_ci uses quantiles (a, 1-a): a 90% interval labelled 95%", "bootstrap.py",
     "np.quantile(reps, [a / 2.0, 1.0 - a / 2.0], axis=0)", "np.quantile(reps, [a, 1.0 - a], axis=0)"),
    ("M2", "moving_block_indices never draws the last admissible block start", "bootstrap.py",
     "starts = rng.integers(0, n - L + 1, size=n_blocks)",
     "starts = rng.integers(0, max(1, n - L), size=n_blocks)"),
    ("M3", "cluster='dataset+series' skips the series stage", "bootstrap.py",
     'by_series = cluster in ("series", "dataset+series")', 'by_series = cluster in ("series",)'),
    ("M4", "e1.mean_delta_intervals uses all origins, not test only", "e1.py",
     'cost = bootstrap.family_cost_table(tables["per_origin"], role="test")',
     'cost = bootstrap.family_cost_table(tables["per_origin"], role=None)'),
    ("M5", "e2.evaluate hands test costs to the frozen selector", "e2.py",
     "test=test.copy() if s.needs_hindsight else None)", "test=test.copy())"),
    ("M6", "_check_payload drops the held-out check on fit configs and fit series", "e2.py",
     '    _assert_no_held_out(p["fit_configs"], f"{context}: fit configs")\n'
     '    _assert_no_held_out([s[2] for s in p["fit_series"]], f"{context}: fit series")\n', ""),
    ("M7", "e2.evaluate: dataset-name overlap check removed", "e2.py",
     "    if ds_overlap:\n"
     '        raise FitEvalOverlapError(f"evaluation datasets were used in fitting: {ds_overlap[:5]}")\n', ""),
    ("M8", "p90_series_regret is actually the median", "e2.py",
     "p90_series_regret=float(np.quantile(v, 0.90))", "p90_series_regret=float(np.quantile(v, 0.50))"),
    ("M9", "symmetry table reads n_units_upper from the lower row", "e1.py",
     'n_units_upper=int(b["n_units"])', 'n_units_upper=int(a["n_units"])'),
    ("M10", "static threshold hard-coded to 0.45, frozen value ignored", "e2.py",
     'StaticOmegaRule(p["static_threshold"])', "StaticOmegaRule(0.45)"),
    ("M11", "e1.selector_regret counts undecided units in n_selected", "e1.py",
     '    s = selection[selection["status"] == "selected"]\n    m = s.groupby',
     "    s = selection\n    m = s.groupby"),
    ("M12", "e1.decision_map: tie tolerance loosened from 1e-12 to 1e-4", "e1.py",
     "tie = np.abs(a - b) <= TIE_RTOL * np.maximum", "tie = np.abs(a - b) <= 1e-4 * np.maximum"),
    ("M13", "e2.evaluate: selector-name check removed", "e2.py",
     '    if selector.name != p["selector_name"]:\n'
     '        raise ValueError(f"built selector is named {selector.name!r}, frozen name is "\n'
     "                         f\"{p['selector_name']!r}\")\n", ""),
    ("M14", "scalar stat_mean_scaled_regret returns the median", "bootstrap.py",
     '    r = (u["pinball_selected"] - u["pinball_oracle"]) / u["scale"]\n    return _mean(r, u, by)',
     '    r = (u["pinball_selected"] - u["pinball_oracle"]) / u["scale"]\n'
     "    return float(r.median()) if by is None else _mean(r, u, by)"),
    ("M15", "gap_to_validation counts undecided units as gap 0", "e2.py",
     'm["gap"] = np.where(both, (m["realized_cost"] - m["realized_cost_ref"]) / m["scale"], np.nan)',
     'm["gap"] = np.where(both, (m["realized_cost"] - m["realized_cost_ref"]) / m["scale"], 0.0)'),
    ("M16", "regret summary sums over all units, not only selected ones", "e2.py",
     '    r = selection[selection["status"] == "selected"]\n    sums = ', "    r = selection\n    sums = "),
    ("N1", "shared time blocks are drawn per series after all", "bootstrap.py",
     "oi = shared_idx[grp.horizon]",
     "oi = moving_block_indices(n, self.block_length_for(grp, block_length), rng)"),
    ("N2", "default regret statistic recomputes the oracle per resample", "bootstrap.py",
     '    r = (u["pinball_selected"] - u["pinball_oracle"]) / u["scale"]\n    return _mean(r, u, by)',
     '    r = (u["pinball_selected"] - np.minimum(u["J_classical"], u["J_tsfm"])) / u["scale"]\n'
     "    return _mean(r, u, by)"),
    ("N3", "check_block accepts a single block start (zero-width intervals again)", "bootstrap.py",
     "    if starts < MIN_BLOCK_STARTS or n < MIN_BLOCKS * L:", "    if starts < 1:"),
    ("N4", "evaluate does not compare crps_method with the frozen value", "e2.py",
     '"crps_method": "crps_method", ', ""),
    ("N5", "selector code hash is computed but not compared", "e2.py",
     '    if code != p["selector_code_sha256"]:', "    if False:"),
    ("N6", "held-out evaluation is not written to the ledger", "e2.py",
     "        _append_ledger(Path(ledger_path), entry)", "        pass"),
    ("N7", "Delta orientation flipped back to classical minus TSFM in the map", "e1.py",
     'out["delta"] = out["scaled_cost_tsfm"] - out["scaled_cost_classical"]',
     'out["delta"] = out["scaled_cost_classical"] - out["scaled_cost_tsfm"]'),
    ("N8", "evaluate does not restrict the test table to the evaluated units", "e2.py",
     'test = tables["test"].merge(units[UNIT_KEYS], on=UNIT_KEYS, how="inner", validate="1:1")',
     'test = tables["test"]'),
    ("N9", "allow_held_out accepts any truthy value", "e2.py",
     "    if not isinstance(allow_held_out, bool):", "    if False:"),
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_tree(dst: Path) -> None:
    shutil.copytree(ROOT / "src", dst / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (dst / "tests" / "unit").mkdir(parents=True)
    for rel in ("tests/__init__.py", "tests/unit/__init__.py"):
        shutil.copy(ROOT / rel, dst / rel)
    shutil.copytree(ROOT / "tests/unit/experiments", dst / "tests/unit/experiments",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (dst / "docs").mkdir()
    shutil.copy(ROOT / "docs/project-overview.md", dst / "docs/project-overview.md")


def _run_suite(tree: Path):
    env = {**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, *SUITE], cwd=tree, env=env, capture_output=True, text=True)
    failed = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\S+) \(([\w.]+)\)", r.stderr, flags=re.M)))
    last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "(no output)"
    return r.returncode, last, sorted({".".join(path.split(".")[-2:]) for _, path in failed})


def _plant_and_run(tree: Path, bug) -> dict:
    ident, what, fn, old, new = bug
    target = tree / PKG / fn
    original = target.read_bytes()
    text = original.decode("utf-8")
    if text.count(old) != 1:
        return dict(id=ident, what=what, outcome=f"NOT PLANTED: text found {text.count(old)} times", tests=[])
    try:
        target.write_text(text.replace(old, new), encoding="utf-8")
        code, last, failed = _run_suite(tree)
    finally:
        target.write_bytes(original)
    if target.read_bytes() != original:
        raise RuntimeError(f"{target} was not restored after {ident}")
    outcome = "caught" if code != 0 else "SURVIVED"
    print(f"  {ident}: {outcome} ({last})", file=sys.stderr, flush=True)       # progress
    return dict(id=ident, what=what, outcome=outcome, last=last, tests=failed)


def _worker(bugs) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="planted_bugs_") as tmp:
        tree = Path(tmp)
        _copy_tree(tree)
        before = {f: _sha(tree / PKG / f) for f in SOURCES}
        rows = [_plant_and_run(tree, b) for b in bugs]
        after = {f: _sha(tree / PKG / f) for f in SOURCES}
        if after != before:
            raise RuntimeError("the copy differs from its starting state after the last bug")
        code, last, failed = _run_suite(tree)
        rows.append(dict(id="clean", what="no bug planted, same copy, after all restores",
                         outcome="passes" if code == 0 else "FAILS", last=last, tests=failed))
        return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=[], help="bug ids to run, e.g. M1 M5 N2")
    ap.add_argument("--jobs", type=int, default=1, help="copies of the tree to work in at once")
    a = ap.parse_args(argv)
    bugs = [b for b in BUGS if not a.only or b[0] in a.only]
    start = {f: _sha(ROOT / PKG / f) for f in SOURCES}
    chunks = [bugs[i::a.jobs] for i in range(a.jobs)]
    with ThreadPoolExecutor(a.jobs) as ex:
        results = [row for rows in ex.map(_worker, chunks) for row in rows]
    order = {b[0]: i for i, b in enumerate(BUGS)}
    bug_rows = sorted((r for r in results if r["id"] != "clean"), key=lambda r: order[r["id"]])
    clean_rows = [r for r in results if r["id"] == "clean"]
    print("| bug | what was planted | outcome | tests that failed |")
    print("|---|---|---|---|")
    for r in bug_rows:
        tests = ", ".join(r["tests"][:3]) + (f" (+{len(r['tests']) - 3} more)" if len(r["tests"]) > 3 else "")
        print(f"| {r['id']} | {r['what']} | {r['outcome']} | {tests} |")
    caught = sum(r["outcome"] == "caught" for r in bug_rows)
    print(f"\n{caught} of {len(bug_rows)} planted bugs caught; "
          f"{sum(r['outcome'] == 'SURVIVED' for r in bug_rows)} survived; "
          f"{sum(r['outcome'].startswith('NOT PLANTED') for r in bug_rows)} could not be planted.")
    for r in clean_rows:
        print(f"clean run of a copy after all restores: {r['outcome']} ({r['last']})")
    end = {f: _sha(ROOT / PKG / f) for f in SOURCES}
    print("working tree source files unchanged by this script: "
          + ("yes" if end == start else f"NO, changed: {[f for f in SOURCES if end[f] != start[f]]}"))
    ok = caught == len(bug_rows) and all(r["outcome"] == "passes" for r in clean_rows) and end == start
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
