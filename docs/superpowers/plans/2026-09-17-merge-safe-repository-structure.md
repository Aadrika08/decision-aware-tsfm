# Merge-Safe Repository Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Reorganize repository material into stable ownership areas without changing research behavior.

**Architecture:** Reusable Python modules live in `src/decision_aware_tsfm`; runnable commands are thin wrappers under `scripts`; tests, result records, and artifacts have separate locations. A package manifest exposes stable commands and `.gitignore` prevents new run data from becoming merge-conflict-prone tracked files.

**Tech Stack:** Python, standard-library `unittest`, existing NumPy/PyArrow/Scipy project dependencies, GitHub CODEOWNERS.

## Global Constraints

- Preserve data, held-out splits, horizons, tau grid, Omega definition, metrics, and experiment gates exactly.
- Do not include `IMPLEMENTATION_STATUS_AUDIT.md`; it is a pre-existing untracked user artifact.
- Keep the K3 threshold-pipeline branch out of scope.
- Move files with Git-aware renames whenever possible.
- Run package tests with `PYTHONPATH=src` until contributors install the package with `pip install -e .`.

---

### Task 1: Establish the package boundary

**Files:**
- Create: `tests/unit/test_package_layout.py`
- Create: `src/decision_aware_tsfm/__init__.py`
- Create: `src/decision_aware_tsfm/contracts/__init__.py`
- Create: `pyproject.toml`

**Interfaces:**
- Consumes: existing reusable modules in the repository root.
- Produces: import path `decision_aware_tsfm` and package command registration.

- [x] **Step 1: Write a failing package-layout test**

```python
from decision_aware_tsfm.omega.compute import compute_omega

def test_omega_module_is_importable_from_the_package():
    assert callable(compute_omega)
```

- [x] **Step 2: Run the test and verify it fails because the package does not exist**

Run: `PYTHONPATH=src python -m unittest tests/unit/test_package_layout.py -v`

Expected: an import failure for `decision_aware_tsfm`.

- [x] **Step 3: Create the package and move reusable modules without changing their logic**

Create the package metadata and move Omega, data preparation, cache, K2, and K3
logic into their responsible package directories. Update internal imports to package
imports.

- [x] **Step 4: Run the package-layout test**

Run: `PYTHONPATH=src python -m unittest tests/unit/test_package_layout.py -v`

Expected: PASS.

### Task 2: Separate execution, tests, results, and artifacts

**Files:**
- Move: root executable/check scripts to `scripts/`
- Move: `test_*.py` to `tests/unit/` or `tests/integration/`
- Move: K2 results to `docs/results/k2/`
- Move: smoke records to `docs/results/cache/`
- Move: reference data to `artifacts/reference/`
- Create: `artifacts/README.md`, `docs/results/README.md`
- Modify: `.gitignore`, `README.md`, `docs/forecast-smoke.md`, `docs/forecast-smoke-verification.md`

**Interfaces:**
- Consumes: package modules created in Task 1.
- Produces: documented commands and a root containing only repository-level files.

- [x] **Step 1: Move test modules and update imports to package paths**
- [x] **Step 2: Move command-line behavior into `scripts/` wrappers**
- [x] **Step 3: Move historical result and reference artifacts without editing their contents**
- [x] **Step 4: Update documentation to package command names and artifact locations**
- [x] **Step 5: Add ignored future-run storage and result-record guidance**

### Task 3: Add ownership guardrails

**Files:**
- Create: `.github/CODEOWNERS`
- Modify: `CONTRIBUTING.md`

**Interfaces:**
- Consumes: directory contracts from the design.
- Produces: explicit reviewers and contributor rules for cross-boundary changes.

- [x] **Step 1: Assign each feature area to its current owner**
- [x] **Step 2: Mark contracts, configs, dependencies, and decisions as shared review areas**
- [x] **Step 3: Document one-feature/one-area PR rules and immutable result-record naming**

### Task 4: Verify the behavior-preserving restructure

**Files:**
- Verify: all moved Python modules, tests, and documentation links.

- [x] **Step 1: Compile every Python source file**

Run: `PYTHONPATH=src python -m compileall -q src scripts tests`

Expected: exit code 0.

- [x] **Step 2: Run deterministic Omega validation**

Run: `PYTHONPATH=src python scripts/validate_omega_synthetics.py`

Expected: four signal results print successfully.

- [x] **Step 3: Run package layout and available unit tests**

Run: `PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v`

Expected: environment-independent tests pass; unavailable optional dependencies are reported separately.

- [x] **Step 4: Check Git integrity and accidental artifact tracking**

Run: `git diff --check && git status --short`

Expected: no whitespace errors and only intended structure changes, excluding the pre-existing audit report.
