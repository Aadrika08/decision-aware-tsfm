# Decision engine: invariants, schema, fixture, seven comparators

Status: runs on fixture only. No real cache has been read. Not a result.

## Commands
    python -m pytest -q tests/unit/decision
    PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_fixture --out artifacts/runs/decision_fixture

## Verified in this change
- 62 tests pass. Four deliberate bugs (pinball sign, tie-break direction,
  validation placed after test, gap guard removed) each cause failures.
- Engine oracle costs match an independent loop-based recomputation to 1e-10 relative.
- Oracle winner equals the analytic planted winner on all 233 units where the
  analytic margin exceeds 4 standard errors.
- D_Omega placeholder fitted on fx_dfit returns omega0 = 0.45, beta = 0, the planted boundary.
- No module in analysis/decision imports an inference library (static and runtime checks).

- Separate check: run_k2.py rerun from a fresh GIFT-Eval download (132 models,
  97 configs) against the current manifest reproduces all 12 K2a/K2b
  correlations. Median-model sensitivity p = 0.8502 (0.850, not 0.851).

## Design choices needing sign-off
1. Decision unit = (series, horizon, tau). Per-origin costs kept separately for the bootstrap.
2. Validation = earliest floor(n/3) origins per series; origins whose targets reach
   into the test period are dropped ("gap").
3. Exact ties go to classical.
4. CRPS from the 11 quantiles by trapezoid rule over [0.01, 0.99].
5. Best-fixed family chosen per (horizon, tau) on scaled test cost; scale = mean |y|.
6. Fraction of oracle value is reported as undefined when best-fixed equals oracle.

## Dependencies
numpy, pandas, scipy, pyarrow (runner only), pytest

## Open inputs
- Static Omega threshold: placeholder 0.45 in the fixture run. Real value to be read
  from arXiv:2511.08884 and recorded.
- Cache schema mapping (cache_io._to_canonical): waiting on origin/step/timestamp and
  dataset/config-to-subset answers.
- D_Omega: placeholder until the K3/E4 frozen selector exists.
- Within-family representatives: fixture uses a fixed mapping; the primary rule is unsettled.
