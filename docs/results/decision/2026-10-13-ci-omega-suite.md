# CI: run the Omega unit tests (2026-10-13)

Status: one word added to `.github/workflows/decision-tests.yml`. Checked locally by running the
workflow's own step. `tests/unit/omega` is not on main yet (it arrives with PR #48), so on main the
job runs the same four suites as before.

## What changed

The `unit` job loops over a list of suite folders under `tests/unit/` and runs each one that exists
on the branch. The list was `decision experiments e3 k2`. It is now `decision experiments e3 k2 omega`.
Nothing else in the workflow changed. `forecast-cache.yml` and `repository-structure.yml` are untouched.

Why: PR #48 adds 14 unittest tests for `compute_omega_checked` under `tests/unit/omega`, and no
workflow ran them.

## Verified in this change

The step's shell script was read out of the YAML file and run with `bash -e`, with the team pins
(NumPy 1.26.4, pandas 2.2.3, PyArrow 16.1.0, SciPy 1.11.4), on Python 3.11 and 3.10, same counts:

- This branch (main `39d49e7` plus the change, no `tests/unit/omega`): decision 152 (1 skipped),
  experiments 169, e3 107, k2 182. Job exit 0. Same suites and counts as before the change.
- Main merged with PR #48 at `961dd57`, with this workflow file: the four suites as above, then
  omega 14. Job exit 0.
- The same tree with one Omega test made to fail on purpose (Python 3.11): 15 tests, 1 failure,
  exit 1.
- Package layout 3 OK, integration 102 OK (17 skipped), both Pythons.

## Not done

- The K3 suite and scikit-learn are not added here. That waits for PR #32.
