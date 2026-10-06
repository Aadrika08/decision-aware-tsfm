# Decision engine: producer-v1 ingress, seeds, per-origin Omega, unittest suite

Status: interface check on synthetic data only. No real forecasts were read. Not a result.
Scope: the #44 consumer changes listed in #45 (producer ingress, origin alignment,
seed retention, per-origin Omega join). Part of #38. Supersedes nothing in the
2026-09-24 note; its fixture results are unchanged (see below).

## Commands

Team pinned environment (Python 3.11, NumPy 1.26.4, pandas 2.2.3, PyArrow 16.1.0, SciPy 1.11.4):

- Tests: `PYTHONPATH=src python -m unittest discover -s tests/unit/decision -t .`
- Check a producer bundle:
  `PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_handoff_check BUNDLE --variant current --classical AutoETS --tsfm Moirai-small --static-threshold 0.45 --fit-only-datasets fx_dfit --out artifacts/runs/decision-handoff`
- Same check as a test: set `DECISION_HANDOFF_BUNDLE=BUNDLE` and run the test command.

## What changed

- `cache_io`: new `producer-v1` schema. Checks every rule in the producer contract
  (exact columns and dtypes, no nulls, finite values, `(job_id, step)` unique, one
  identity per job, family in {classical, tsfm}, one family per model, ISO 8601
  timestamps that map one-to-one to target positions and increase with them).
  `y_true` becomes the internal `y`. Seed is part of the cache key. Realised values
  must agree per target position across models, seeds and horizons.
- Time convention: the engine now uses the producer's (origin = first target,
  target = origin + step - 1). `fixture-v0` (origin = last observed) is refused with
  an explanation instead of being silently misread.
- Seeds: each seed is scored separately, then averaged per window. A window counts
  once whatever its seed count. The two families must have the same seed set per window.
- Omega: joined on (dataset, config, series_id[, origin]) with a named variant.
  With per-origin Omega the value used is the one at the decision origin (first test
  origin; first origin for fit-only datasets). A table with variants requires the
  variant to be named.
- `handoff`: bundle reader. Reads only listed partitions, checks each SHA-256, the exact
  Arrow schema, job provenance (mandatory) against the rows, naming scheme and source
  repository across manifest, jobs and `omega.json`, Omega history not ending after
  origin, and matching source revision and hash between forecasts and Omega rows.
- `run_handoff_check`: one-command consumer check; also recomputes every test cost
  straight from the raw quantile columns.
- D_Omega with no fit-only data now reports `insufficient_data` (note "not fitted")
  instead of stopping the other six comparators.
- Bug fixed: when no series had any validation origin, the validation comparators
  crashed; they now report `insufficient_data`.
- Tests rewritten from pytest to unittest (the team's runner); pytest still runs them.

## Verified in this change

- 76 tests, 74 pass and 2 skip, under unittest on the pinned versions. Under pytest:
  same outcome. Skips: the real-bundle test (needs `DECISION_HANDOFF_BUNDLE`) and the
  producer `decision_fixture()` test (needs `cache/schema.py` in the checkout).
- Copied into the `feat/45-cache-consumer-handoff` tree: compileall passes, the
  package-layout, cache-schema and consumer-handoff tests pass, and with
  `DECISION_HANDOFF_BUNDLE` pointing at the rebuilt bundle all 76 decision tests run
  and pass with none skipped. `decision_fixture()` gives classical 2*tau and TSFM
  1-tau exactly.
- Six deliberate bugs each cause failures: seeds added instead of averaged, Omega
  taken at the last origin, partition hash check removed, Omega history check removed,
  seed-set mismatch allowed, engine reading q_0.50 for every tau.
- Fixture results unchanged by the convention switch: all 3,388 rows identical to the
  #44 version, largest difference 0.0; D_Omega placeholder still omega0 = 0.45, beta = 0.
- Producer bundle rebuilt with the producer's own exporter (`handoff_fixture.py`,
  SHA-256 799aca3d...) from the pinned #44 `fixture.py` (SHA-256 e4f00736...), with
  the exporter's git-checkout check bypassed: 408 partitions, all hashes verified;
  9,792 rows; 34 series; 68 series-origins; seeds 42 and 43; both Omega variants.
  Engine costs match direct recomputation from the raw columns on all 242 units
  (largest relative difference 3.3e-16). Producer path and canonical path give
  identical results on the same windows (1,694 rows, largest difference 0.0).
- The rebuilt `handoff.json` hash (f01f634f...) differs from the one recorded in the
  producer note (a6e1ec82...), although both pinned source files match. Cause not
  established; acceptance should be run on the producer's own bundle.

## Design choices needing sign-off

1. Seeds: score per seed, then average per window (alternative: average quantiles
   across seeds, which scores a forecast no model produced).
2. Omega at the decision origin (alternative: average over origins, which lets later
   history into the decision).
3. D_Omega with no fit data reports insufficient_data rather than failing the run.

## Open

- The team's CI does not run these tests. Proposed one-line addition to the
  "Forecast cache" workflow (shared file, needs its owner's review):
  `python -m unittest discover -s tests/unit/decision -t .`
- CODEOWNERS has no entry for `analysis/decision/`.
- The one-third validation split is still a proposal; the development roles must be
  agreed before real analysis.
- With two origins per series (as in the bundle), validation comparators have no
  validation origins and report insufficient_data by design.
