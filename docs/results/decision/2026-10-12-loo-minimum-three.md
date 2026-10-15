# Decision engine: a leave-one-dataset-out fold needs 3 other datasets by default

Status: default changed and tested on the synthetic fixture only. No real cache has been read.
Not a result.
Scope: `analysis/decision/selectors.py` (`LeaveOneDatasetOut.build`), `analysis/decision/pipeline.py`
(`run`), `analysis/decision/run_handoff_check.py`, `tests/unit/decision/test_decision_roles.py`,
`tests/unit/decision/test_decision_null_omega_handoff.py`. Part of #38.

## Why

Decision record 0007 section 2 (PR #57) now says a fold needs at least 3 other datasets with usable
rows. Abdullah asked for 3 on #57 on 2026-10-11 so that the engine and his K3 selector carry one
number (record 0008 section 3: 3 datasets with distinct Omega before a threshold is fitted), and
Veer agreed on 2026-10-12. The engine's default was 2. With exactly 3 datasets in total the two
disagreed: the engine built the fold and the selector inside it answered `insufficient_data`.

This supersedes one line of `2026-10-09-development-roles.md` ("`min_fit_datasets` (default 2)")
and its open choice 2 ("A fold needs at least 2 other datasets. Alternative: 1.").

## What changed

- `LeaveOneDatasetOut.build(..., min_fit_datasets=3)` and `run(..., min_fit_datasets=3)`. Both were 2.
  The argument is unchanged, so a caller can still pass another number.
- A fold with fewer than 3 other datasets is not fitted. Its units report `insufficient_data` with
  the note `leave-one-dataset-out: N other dataset(s) with usable units, need 3`. The other six
  comparators still run.
- `run_handoff_check` has a new option `--min-fit-datasets` (default 3) and passes it to `run`.
  Before, the command could only use the engine's default.
- Two new tests in `test_decision_roles.py`: three other datasets fit every fold and two do not,
  with the reason text checked; `run` uses the same default.
- `test_decision_null_omega_handoff.py` builds a bundle with three datasets (two evaluation, one
  fit-only), so each fold has two other datasets. Six of its tests relied on the old default to keep
  D_Omega fitted. They are about null Omega, not about the floor, so the file now asks for 2
  explicitly (`MIN_FIT_DATASETS = 2`, passed to `run` and as `--min-fit-datasets 2`). One new test
  runs the command without the option and checks that the same bundle is then left unfitted with
  `2 other dataset(s) with usable units, need 3`.

## Commands

Team pinned environment (Python 3.11.15 and 3.10.20, NumPy 1.26.4, pandas 2.2.3, PyArrow 16.1.0,
SciPy 1.11.4), from the repository root:

- `PYTHONPATH=src python -m unittest discover -s tests/unit/decision -t .`
- `PYTHONPATH=src python -m unittest discover -s tests/unit/experiments -t .`
- `PYTHONPATH=src python -m unittest discover -s tests/unit/e3 -t .`
- `PYTHONPATH=src python -m unittest discover -s tests/unit/k2 -t .`
- `PYTHONPATH=src python -m unittest tests.unit.test_package_layout`
- `PYTHONPATH=src python -m unittest discover -s tests/integration -t .`
- `PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_fixture --out <dir>`

## What was measured (2026-10-12, identical on Python 3.11 and 3.10)

- decision 155 OK (1 skipped; 152 on main plus the 3 new tests), experiments 169 OK, e3 107 OK,
  k2 182 OK, package layout 3 OK, integration 102 OK (17 skipped).
- With only the two defaults changed, 6 tests in `test_decision_null_omega_handoff.py` failed
  (for example `d_omega/insufficient_data` 44 where 11 was expected). That is the three-dataset
  bundle described above.
- The 2 new tests in `test_decision_roles.py` fail on main's code (default 2) and pass here.
- 8 of 8 planted bugs caught: either default set back to 2, either default raised, the comparison
  off by one, `run` ignoring its argument, the command's default set back to 2, the command
  ignoring its option.
- Default fixture run: all 5 output files and the printed tables are byte-identical to main. Every
  fixture fold has 4 other datasets, so no fold changes.
- `run_handoff_check` on the released bundle `consumer-handoff-45.zip`, variants `current` and
  `last_4096`, main against this branch: `selection_results.parquet` is byte-identical, 0 unmatched
  comparisons, and `report.json` differs only in `d_omega.min_fit_datasets` (2 on main, 3 here).
  That bundle has 5 datasets, so every fold has 4 others.
- The pilot workload also has 5 datasets, so the new default does not bind there either.

## Not covered

- No real forecast has been scored. The floor only matters for a run with 3 or fewer datasets.
- The K3 selector adapter branch (`feat/37-k3-selector-adapter`) is not on main and has its own
  copy of `run` with the old default of 2. It needs this change when it is brought up to main.
