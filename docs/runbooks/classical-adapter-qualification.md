# Classical adapter qualification boundary

Update, 2026-09-20: the four-model synthetic qualification methods are now accepted
in [ADR 0005](../decisions/0005-pilot-adapter-qualification.md) and implemented under
separate adapter names in #20. Use the [qualification runbook](pilot-adapter-qualification.md)
for those configurations. The historical interface and original proposal below
explain the boundary preserved by #17; they are not the latest qualification status.

Issue #17 provides the adapter fixture and preserves two historical smoke paths. It does
not qualify the final four-model workload or produce new forecast evidence.

## Version 1 interface

`load_adapter(job)` returns an object with:

```python
quantiles, measurement = adapter.predict(context, job)
```

`context` is a finite one-dimensional array of exactly `context_length` past observations.
There is no actuals argument. `quantiles` has shape `(horizon, 11)` in the tau order defined
by `cache.schema.TAUS`. Every job supplies these flat fields:

| Field | Supported value |
|---|---|
| `adapter_version` | `"1"` |
| `adapter` | `fixture`, `seasonal_naive_smoke`, or `moirai_smoke` |
| identity | `model`, `family`, `checkpoint`, `fixture` |
| execution | `device`, `batch_size`, `seed` |
| forecast | `frequency`, `context_length`, `horizon`, `quantile_method`, `settings` |
| time provenance | `start_timestamp`, `context_start_timestamp`, `origin_timestamp` as ISO-8601 |

The deliberately invented `fixture` adapter requires a `fixture_*` model label,
`fixture=true`, CPU, no checkpoint, `quantile_method="constant"`, and exactly
`settings={"prediction": <finite number>}`. Its output is test data, not model evidence.

The preserved historical paths are intentionally narrow:

| Adapter | Exact admitted configuration |
|---|---|
| `seasonal_naive_smoke` | `model="seasonal_naive"`, classical, CPU, hourly, context 2048, horizon 24, batch 1, no checkpoint, `empirical_lag24_linear`, `settings={"season_length": 24, "residual_lag": 24}` |
| `moirai_smoke` | `model="moirai_1.1_R_small"`, TSFM, CPU or CUDA, hourly, context 2048, horizon 24, batch 1, pinned checkpoint `0c24ab99db2c1a70ea2a0fc03bf113329772ac64`, `sample_linear`, repository `Salesforce/moirai-1.1-R-small`, 4096 samples, patch size 32, target dimension 1, no covariates |

Moirai retains the historical seed setup and records sampling warnings. CUDA categorical
sampling was historically run with deterministic algorithms in warning mode, so seeds and
warnings are provenance rather than a claim of bitwise repeatability.

## Proposed AutoETS and AutoARIMA qualification

`autoets` and `autoarima` currently raise a precise unqualified-adapter error. Before either
is admitted, issue #16 should review a native predictive-distribution route from the chosen
pinned implementation. Qualification should freeze the package and model configuration,
define how its predictive distribution maps to all 11 requested quantiles, and test shape,
ordering, seeded repeatability where stochastic, and context-only operation on development
fixtures. Residual resampling or Gaussian tails should not be introduced as an undocumented
substitute; either would change the forecast distribution and needs explicit approval.

The concrete review candidate is the native predictive-distribution interface in
[StatsForecast `models.py`](https://github.com/Nixtla/statsforecast/blob/main/python/statsforecast/models.py),
inspected 2026-09-19. This is a proposed qualification path, not an implemented or approved
scientific default:

| Native central interval | Lower quantile | Upper quantile |
|---:|---:|---:|
| 20% | 0.40 | 0.60 |
| 50% | 0.25 | 0.75 |
| 82% | 0.09 | 0.91 |
| 90% | 0.05 | 0.95 |
| 98% | 0.01 | 0.99 |

Qualification must also obtain a justified 0.50 quantile. A forecast mean cannot stand in
for the median when the predictive distribution may be skewed. A future approved ADR must
pin the released StatsForecast package version, exact model configuration, and explicit
season length from the approved workload configuration after verifying that version's
interfaces. Native ETS point forecasts may be means rather than medians, so that route
cannot claim compatibility until the 0.50 mapping is established. No dependency or model
method is added by this proposal.

The historical SeasonalNaive residual construction supports only hourly context 2048 and
horizon 24. Its lag-24 residuals cannot be extrapolated to horizons 1 or 168 without a new
qualification. Producing all 11 ordered quantiles demonstrates an interface property, not
calibration or decision quality.
