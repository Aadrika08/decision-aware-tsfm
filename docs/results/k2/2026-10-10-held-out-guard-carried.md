# K2: carry the held-out guard module so the K2 suite runs on its own

Status: packaging change only. No K2 code, test or result changed.
Scope: two files added, `src/decision_aware_tsfm/analysis/decision/__init__.py` (5 lines) and
`src/decision_aware_tsfm/analysis/decision/guards.py` (25 lines). Part of #39.

## Why

The K2 code and tests import the held-out guard (`HELD_OUT_TOKENS`, `HeldOutLeakError`,
`is_held_out`, `assert_no_held_out`) from `analysis/decision/guards.py`. That module arrives with
the decision engine (#44), which is not on main yet. On this branch alone, 6 of the 8 K2 test files
could not be imported (`ModuleNotFoundError: No module named 'decision_aware_tsfm.analysis.decision'`)
and only 24 tests ran, 6 of them as import errors.

`guards.py` is the only engine module K2 uses (9 import lines across K2 code and tests, all of
them `decision.guards`). It imports nothing but the standard library.

## What changed

The two files are copied byte for byte from the engine branch at `cab61d4`. When #44 merges, git
sees identical files on both sides, so there is no conflict (checked with `git merge-tree`).

## Commands

Team pinned environment (Python 3.11, NumPy 1.26.4, pandas 2.2.3, PyArrow 16.1.0, SciPy 1.11.4):

- `PYTHONPATH=src python -m unittest discover -s tests/unit/k2 -t .`
- `PYTHONPATH=src python -m decision_aware_tsfm.analysis.k2.replication --outdir <dir>`

## Verified in this change

- K2 suite on this branch with the two files: 149 tests pass, on Python 3.11 and 3.10.
- Team tests on this branch: 3 layout tests pass; 96 integration tests pass (17 skipped), on both.
- K2 rerun (with the whole engine branch merged in, Python 3.11): `summary_stats.csv` and
  `within_family_sensitivity.csv` are byte-identical to the saved copies in `docs/results/k2/`.
  `joined_table.csv` has the same 80 rows and 12 columns with every value equal, but the saved
  copy has a different column order; it predates this branch and was not regenerated.

## Open

- The other 11 tables the K2 command writes have no saved copy to compare with.
- If `guards.py` changes on the engine branch before #44 merges, copy it here again.
