# 0002 — Issue #3 forecast smoke configuration

- Date: 2026-09-09
- Owner: Aadrika Maurya
- Issue: https://github.com/Aadrika08/decision-aware-tsfm/issues/3
- Status: accepted for the smoke test by Aadrika in the implementation request

## Decision

Use SeasonalNaive and Moirai-1.1-R-small on channel 0 of GIFT-Eval `ett1/H`.
Moirai is already in the implementation plan's provisional roster. This selects
a representative for this smoke test only; it does not freeze the final roster.

Keep all eleven tau values: 0.01, 0.05, 0.09, 0.25, 0.40, 0.50, 0.60, 0.75,
0.91, 0.95, 0.99. Chronos-Bolt's current quantile API clamps requests outside
its trained 0.1–0.9 range to its endpoint predictions. Those outputs cannot
validate the intended tail interface. Moirai samples a distribution; estimate
all eleven quantiles from the same 4,096 samples, using NumPy linear quantiles.
Sampling uncertainty remains, particularly in the tails; this run tests the
interface, not calibration or superiority.

Use the pinned dataset and checkpoint revisions in `forecast_smoke.py`.
The dataset is one item with shape `(7, 17420)`, hourly from 2016-07-01.
The existing development manifest is used only to confirm dataset membership.
Its flattened length and Omega values are not inputs to this workflow. No K1
or K3 files are modified.

The GIFT-Eval short configuration has prediction length 48 and 20 evaluation
windows, placing the training cutoff at `17420 - 48 * 21 = 16412` and the
validation cutoff at 16460. Our two 24-hour smoke windows partition that
validation block: `[16412, 16436)` and `[16436, 16460)`. This is not a
GIFT-Eval benchmark reproduction. Both models see the preceding 2,048 points;
the second origin may use observations already realized in the first window.
Training-only here means observations preceding each forecast origin.

SeasonalNaive repeats the last 24 observations and adds empirical quantiles
of lag-24 residuals from that origin's context. This wrapper is specified only
for the 24-step smoke horizon. Moirai uses patch size 32, batch size 1,
4,096 samples, no covariates, and no fitting or hyperparameter selection.
The two window seeds are 42 and 43. PyTorch 2.4.1's CUDA categorical sampler
raises under strict deterministic mode. CUDA generation therefore uses seeded
deterministic-warn mode and saves the actual warnings with each job; CPU mode
remains strict. Resume reproducibility is tested on the recorded environment,
not guaranteed across hardware/library versions. Device choice is part of the
job identity. Cached quantiles are the downstream source of truth.

Run on the available local RTX 4050, with explicit hardware measurements.
No paid or larger cloud machine is requested. Do not call these T4 timings.

## Consequences

Each complete job has 24 rows and eleven quantiles, aligned actuals, a model
family, source/implementation/environment hashes, and a fixed seed. The full
smoke cache has four jobs and 96 rows. Hashes include model settings, so changed
configurations cannot silently reuse predictions. Writes are atomic and a
single writer owns the cache. Raw predictions remain outside Git.

The cache's oracle is the better of these two fixed representatives over both
windows, separately at each tau. Report pinball and regret in raw target units,
without claiming normalized full-family or deployment regret. Throughput
projections apply only to these models, context, horizon and measured hardware;
they cannot establish the complete roster's budget.

## Sources

- [GIFT-Eval split implementation](https://github.com/SalesforceAIResearch/gift-eval/blob/main/src/gift_eval/data.py)
- [Chronos-Bolt quantile interface](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py)
- [Moirai 2.0.0 package's sampling implementation](https://github.com/SalesforceAIResearch/uni2ts/blob/2.0.0/src/uni2ts/model/moirai/forecast.py)
