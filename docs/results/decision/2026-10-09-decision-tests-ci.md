# Decision tests in CI (review of #44, item 2)

Status: workflow file added and its steps run locally. It has not run on GitHub: Actions jobs on
this repository are currently not started (the job annotation on 2026-10-09 reads "The job was not
started because recent account payments have failed or your spending limit needs to be increased").
Scope: one new file, `.github/workflows/decision-tests.yml`. `forecast-cache.yml` and
`repository-structure.yml` are unchanged. Part of #37.

## What it does

On every pull request and on pushes to main: Ubuntu, Python 3.11, the team pins (NumPy 1.26.4,
pandas 2.2.3, PyArrow 16.1.0, SciPy 1.11.4), `pip install --no-deps -e .`, then
`python -m unittest discover -s tests/unit/<suite> -t .` for each of `decision`, `experiments`,
`e3` and `k2` that exists on the branch. A branch with none of them passes with a message, so the
job does not block PRs that carry no decision-side code. Any failing suite fails the job; the
remaining suites still run so one failure does not hide another.

The decision tests were moved from pytest to unittest on 2026-10-06, so no pytest install is needed.

## Verified in this change

The workflow's own `run` steps were read out of the YAML file and executed with `bash -e` in a
fresh clone and a fresh Python 3.11 environment (3.11.15), four times:

- This branch: decision 117 tests, OK (1 skipped without `DECISION_HANDOFF_BUNDLE`), job exit 0.
- A throwaway tree of main plus the engine, E1/E2, E3, K2 and roster branches: decision 117,
  experiments 165, e3 107, k2 149, all OK, job exit 0.
- main as it is today (no decision-side suites): prints "nothing to run", job exit 0.
- This branch with one test deliberately made to fail: 118 tests, 1 failure, job exit 1.

The three team layout tests (`tests/unit/test_*.py`) still pass with the new file present, on
Python 3.11 and 3.10.

## Open

- The first real run on GitHub can only be checked once Actions jobs start again.
- Python 3.10 is run locally for every change but is not in this workflow. Adding it would double
  the minutes used; left as a team decision.
