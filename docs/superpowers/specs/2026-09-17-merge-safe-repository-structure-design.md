# Merge-Safe Repository Structure — Design

## Purpose

Reorganize the repository so four contributors can work in separate areas without
unnecessary merge conflicts. This is a behavior-preserving maintenance change: it
must not change datasets, held-out splits, horizons, tau values, Omega, metrics, or
experimental gates.

## Chosen approach

Use one importable Python package for reusable research logic, thin command-line
entry points for people running workflows, and separate locations for tests,
documentation, and generated artifacts. Existing historical result files remain
available, but future run output is kept out of Git unless it is a small,
human-readable result record.

## Directory contracts

```text
src/decision_aware_tsfm/
  contracts/        shared constants and schemas; coordinated changes only
  omega/            Omega calculation and validation support
  data/             GIFT-Eval manifest preparation
  cache/            forecast-cache generation and release verification
  analysis/k2/      K2 replication
  analysis/k3/      matched-Omega and future K3 analysis
scripts/            thin executable wrappers and one-off checks
tests/unit/         deterministic module-level tests
tests/integration/  workflow/release checks
docs/results/       small, immutable human-readable result records
artifacts/reference/ tracked historical input/output artifacts
artifacts/runs/      ignored generated run data and caches
```

## Compatibility and migration rules

- `pyproject.toml` declares the package and standard command names.
- Existing root scripts move to `scripts/`; workflow documentation is updated to
  use package commands.
- Existing test files move under `tests/` and import package modules directly.
- Existing raw artifacts move out of the root without altering their content.
- The superseded pre-trim manifest script is removed rather than kept as an
  executable alternative; Git history and the decision records preserve it.
- The unmerged K3 threshold-pipeline branch is out of scope for this move. Its
  future merge target is `src/decision_aware_tsfm/analysis/k3/`.

## Merge boundaries

| Area | Primary owner | Coordination requirement |
|---|---|---|
| `omega/`, `data/` | Dayson | Coordinate before changing `contracts/` |
| `cache/` | Aadrika | Keep cache-schema changes documented |
| `analysis/k2/` | Veer | Do not modify K3 or cache internals |
| `analysis/k3/` | Abdullah | Do not merge threshold work until its dependencies and tests are ready |
| `contracts/`, `configs/`, dependencies, decisions | project lead | Dedicated issue and review required |

## Verification

The restructure is acceptable only if package modules compile, the deterministic
Omega validation still runs, structural imports work, existing smoke/release tests
can be discovered from their new locations, and Git reports only intentional
changes.
