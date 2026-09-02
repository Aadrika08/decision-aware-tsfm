# Five-day feasibility sprint

Each day has one deliverable and one pass condition. **The sprint is not a mini-paper.**
Its job is to falsify expensive assumptions before the full grid is launched.

All four students work in parallel; the day structure below is the schedule of *results*,
not a queue.

Sources: [`implementation-plan.pdf`](implementation-plan.pdf) §2, §3, and the smoke-test
portion of §4; [`proposal.pdf`](proposal.pdf) §10.

---

## Shared rules

These are not guidelines.

- All four students work in parallel.
- **Do not run the complete forecasting grid during this sprint.**
- Use only a small development subset for testing.
- **Do not use M5 or PJM for fitting or tuning.** They remain held out for the final project.
- Save every result, configuration, seed, runtime, and dataset split used.
- **Dayson should share the validated Omega implementation as early as possible.**
- **Aadrika should provide a small forecast-cache sample** so Veer and Abdullah can test
  downstream calculations.
- **A failed gate is a valid result. Do not change the experiment merely to force a
  positive outcome.**

### Dependency map

```
Dayson (K1) ──── Omega implementation ────┬──→ Veer (K2a/K2b)
                                          └──→ Abdullah (K3, matched-Omega)

Aadrika (cache smoke test) ─── cache sample + schema ──┬──→ Veer (K2b, pinball at 0.5)
                                                       └──→ Abdullah (K3 family costs)
```

Both downstream owners are blocked on inputs from the other two. Ship the interface early
and imperfect rather than late and polished.

---

## Day schedule and pass conditions

| Day | Work and pass condition |
|-----|-------------------------|
| **1** | Implement and validate Omega against synthetic signals of known structure; execute K1; decide levels versus first differences and document it. **Pass:** pure sine ≈ 0.87, daily plus weekly cycle ≈ 0.79, moderate noise ≈ 0.32, white noise ≈ 0.06. |
| **2** | Execute K2 using published GIFT-Eval per-model results. Compute Omega on training splits, join, and check whether the published symmetric-cost relationship reproduces. No forecasting model is run. **Pass:** the published direction and rough magnitude recover. **This is the decisive gate.** |
| **3** | One classical and one TSFM end to end on one development dataset; cache the quantile grid; compute pinball-tau and downstream regret; re-estimate the compute budget from measured throughput. **Pass:** a cached quantile grid exists and full-grid compute is projected within budget. |
| **4** | Analytically construct and numerically verify one matched-Omega pair; run structural K3 at tau ∈ {0.5, 0.91} on a small subset. **Pass:** the Omega match verifies numerically and a boundary can be estimated. |
| **5** | Fit the minimal selector on a development subset, freeze it, and smoke-test on a held-out subset. **Pass:** the frozen selector runs end to end on data it has not seen. |

### Go / no-go memo

The sprint ends with five numbers:

1. Omega on the four synthetic signals, against expected.
2. The K2 relationship, against the published value.
3. Measured throughput extrapolated to the full grid.
4. The verified Omega match for the synthetic pair.
5. The fraction of oracle value captured by the best scalar threshold.

Exact numerical success thresholds for the full project are pre-registered **after** this
sprint, using measured noise and compute rather than arbitrary targets.

---

## Dayson — Data preparation and K1: validate Omega

**References:** Implementation plan §2 (Stage 0 — pre-registration and data preparation),
§3 (K1). Proposal §10, Day 1.

### Task

1. Prepare the small development dataset subset used during the sprint.
2. Record dataset names, frequency, usable history length, and proposed horizons.
3. Exclude configurations with fewer than 1,000 usable timestamps.
4. Implement Omega according to the source paper's definition.
5. Test Omega on:
   - Pure sine wave: expected around **0.87**
   - Daily plus weekly cycle: expected around **0.79**
   - Moderately noisy signal: expected around **0.32**
   - White noise: expected around **0.06**
6. Compare Omega calculated on raw levels and first differences.
7. Document which version should be used and why, especially for random-walk-like series.

### Why this is required

Omega is the main cheap input used by the proposed selector. If its implementation is
incorrect, or if it gives misleading values for drifting series, every later boundary and
selector result will be unreliable. The matched-Omega experiment also assumes that Omega
contains only the intended spectral and second-order information.

### Deliverables

- Reusable Omega implementation.
- Small development-dataset manifest.
- List of excluded short series.
- Table containing the four synthetic-signal results.
- Short levels-versus-first-differences decision note (→ `docs/decisions/`).

### Completion condition

The four synthetic tests behave approximately as expected, all deviations are reported, and
the levels-versus-differences choice is documented.

---

## Veer — K2a and K2b: replicate the published relationship

**References:** Implementation plan §3 (K2a, K2b). Proposal §10, Day 2.

### Task

1. Obtain the published GIFT-Eval per-model result files.
2. Use Omega values supplied by Dayson for the corresponding training splits.
3. Join Omega with the published sMAPE and MSE results.
4. Check whether the published direction is reproduced: whether TSFMs gain relative
   advantage in higher-Omega regions.
5. Record the direction, rough magnitude, number of usable configurations, and obvious
   failure cases.
6. **Start K2b only after the K2a result is known.**
7. Test whether the relationship survives under pinball loss at tau = 0.5 using available
   probabilistic outputs or the small cache supplied by Aadrika.
8. If compatible quantile outputs are unavailable, **report that as a K2b input limitation;
   do not label it a K2a replication failure.**

### Why this is required

The complete project extends an existing Omega-based result. The team must first confirm
that the original relationship can be reproduced. Separating K2a and K2b also tells us
whether a failure comes from the original premise or only from changing the evaluation
metric.

### Deliverables

- Clean joined table of dataset, Omega, model family, sMAPE, and MSE.
- One replication plot or summary table.
- K2a pass/fail conclusion with evidence.
- K2b comparison at tau = 0.5, or a precise report of the missing input.

### Completion condition

The team can clearly state whether the original relationship reproduces and whether the
move to pinball loss changes it.

---

## Aadrika — Small end-to-end forecasting and cache smoke test

**References:** Implementation plan §4 (Stage 2 — generate all predictions), **feasibility-
sized version only**. Proposal §10, Day 3.

### Task

1. Select one development dataset from Dayson's manifest.
2. Run one classical forecaster and one TSFM end to end.
3. Prefer the plan's simple representatives for the smoke test — Seasonal Naive and
   Chronos-Bolt-base — provided both expose compatible quantiles.
4. Generate and save the **full required quantile grid**, not only point forecasts.
5. Store predictions in Parquet with keys for dataset, configuration, window, model,
   horizon, and seed.
6. Confirm that rerunning the command **skips completed rows** instead of duplicating them.
7. Calculate pinball loss and downstream regret **from the saved cache**, without rerunning
   either model.
8. Measure seconds per batch, wall-clock time, rows produced, and projected compute for the
   complete grid.

### Why this is required

Forecast generation is the only expensive part of the complete project. The team must prove
that the quantile interface, caching, resume behaviour, and downstream cost calculation
work before launching a large GPU job. Measured throughput is also needed to confirm that
the full experiment fits the available compute budget.

### Deliverables

- Small quantile cache in Parquet format.
- One classical and one TSFM forecast result.
- Pinball-loss and regret calculation from cached data.
- Runtime and full-grid compute estimate.
- Short cache-schema description for Veer and Abdullah.

### Completion condition

The cache can be created, reopened, analysed without model inference, and resumed without
duplicate rows. The full-grid compute estimate must be based on **measured** throughput.

---

## Abdullah Adnan — K3, matched-Omega test, and minimal-selector smoke test

**References:** Implementation plan §3 (K3), §6 (E3 — matched-Omega sufficiency test).
Proposal §10, Days 4 and 5.

### Task

1. Construct one pair of synthetic time series with matched spectral or second-order
   structure but different skewness or tail behaviour.
2. Use Dayson's implementation to verify that their Omega values match numerically.
3. Use the small family-cost results from Aadrika to prepare the K3 analysis pipeline.
4. At tau = 0.5 and tau = 0.91:
   - Calculate the family cost difference.
   - Count sign reversals where possible.
   - Compare a monotone boundary estimate with an unconstrained estimate.
   - Measure the oracle value captured by the best single-Omega threshold.
5. Fit the smallest possible selector using only Omega and tau.
6. Freeze the selector and smoke-test it on a small subset not used for fitting. This must
   be a **development-domain holdout, not M5 or PJM.**
7. Report whether the boundary can be estimated; **do not force a boundary if the
   scalar-threshold structure is weak.**

### Why this is required

A fitted regression will always produce some curve, even when no meaningful boundary
exists. K3 checks whether the boundary is real before the team builds the full paper around
it. The matched-Omega pair tests the deeper question: whether Omega can ever be sufficient
when two series have the same spectrum but different distributional shapes.

### Deliverables

- Synthetic matched-Omega pair and generation settings.
- Numerical Omega-match table.
- Preliminary boundary result at tau = 0.5 and tau = 0.91.
- Oracle-value calculation for the best scalar threshold.
- Frozen-selector smoke-test result on unseen sample data.

### Completion condition

The Omega match is numerically verified, the K3 analysis runs end to end, and the selector
can be frozen and applied to data not used for fitting. **A finding that no stable scalar
boundary exists is acceptable.**
