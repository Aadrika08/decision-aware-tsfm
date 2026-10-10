# 0005 — Four-model adapter qualification

- Date: 2026-09-20
- Owner: Aadrika
- Issue: [#20](https://github.com/Aadrika08/decision-aware-tsfm/issues/20), parent [#16](https://github.com/Aadrika08/decision-aware-tsfm/issues/16)
- Status: accepted for synthetic-input qualification by the user's implementation request following the concrete proposal; [recorded agreement](https://github.com/Aadrika08/decision-aware-tsfm/issues/16#issuecomment-5748473632).

## Scope

Qualify actual Moirai-small, AutoARIMA, AutoETS and SeasonalNaive implementations
against invented hourly inputs before admitting a real development workload.
Context length is 2,048, horizons are 1/24/168 observations, seeds are 42/43,
classical season length is 24 and batch size is one. Keep the eleven taus from
the existing cache schema. Classical models run on CPU; Moirai uses the local
RTX 4050. CPU Moirai is an explicit alternate execution device, not GPU evidence.

This decision authorizes the methods below and their separate dependency pins.
It does not freeze development datasets, series/origins, splits, Omega variants,
the final model roster, or the full-workload budget. Dayson supplies the exact
approved workload. M5/PJM remain untouched. The historical smoke configurations
and release retain their existing meaning under ADR 0002.

## Predictive distributions

Use StatsForecast **2.1.1**, upstream source
`264166ec0ddbfd481ee8e0b71e5ca42d2c0676fc`. Pin its dependency environment separately
in `requirements/qualification.txt`, retaining all historical smoke package pins.

- **AutoARIMA:** native Gaussian predictive quantiles, no Box–Cox transform,
  bias adjustment, conformal wrapper or exogenous variables. Use the version's
  stepwise AICc defaults, explicitly serialized in `cache/pilot.py`. Native central
  interval levels 20/50/82/90/98 yield the ten nonmedian quantiles; the Gaussian
  forecast center is the median. These are quantiles of the fitted predictive
  approximation, not a calibration guarantee.
- **AutoETS:** automatic `ZZZ` selection, Gaussian fitted innovations, native
  damping selection, no conformal wrapper. Draw 4,096 native paths with the job
  seed and use NumPy linear empirical quantiles for every tau. The sampled median
  avoids assuming that an ETS forecast mean is its median. Monte Carlo tail error
  remains. Store fitted model components and parameters.
- **SeasonalNaive:** a new, separately named native Gaussian seasonal-random-walk
  adapter, using native horizon-dependent uncertainty and the same interval-level
  mapping. This changes the new pilot's uncertainty method from the historical
  empirical lag-24 smoke wrapper. Preserve both labels and their distinct keys.
- **Moirai-small:** retain repository `Salesforce/moirai-1.1-R-small`, checkpoint
  `0c24ab99db2c1a70ea2a0fc03bf113329772ac64`, patch size 32, 4,096 samples, one target
  and no covariates. Admit the three horizons only through the new `moirai_pilot`
  configuration. Sample quantiles use NumPy's linear method.

## Qualification evidence

Use two fixed invented contexts: positive seasonal/trend/noise and constant
negative observations. Preserve their generation formula, RNG seed, source bytes
and hashes. Exercise every model/horizon/context with seeds 42 and 43, repeating
seed 42 once: 72 prediction attempts for a complete run. Synthetic source metadata
uses `fixture=true` even when predictions come from actual model implementations;
this must never become evidence for a production compute budget.

Check exact `(horizon, 11)` shape, finite values, ordered quantiles, unchanged
context and same-seed repeatability. CPU repeats require exact equality. Record
GPU maximum absolute differences against a fixed `1e-6` tolerance in raw synthetic
target units; also retain deterministic-algorithm warnings. This tolerance is an
engineering check on these fixed inputs, not an experimental gate or cross-device
reproducibility claim. Report failures without changing data or method to pass.

Record package/code identity, source/forecast hashes, hardware, adapter loading,
fitting and prediction timings, fitted classical specifications, warnings and
failures. Save generated arrays and detailed logs in ignored `artifacts/runs/`;
commit a small dated result note. Partial coverage remains explicitly partial.

## Sources

- [StatsForecast 2.1.1 release](https://github.com/Nixtla/statsforecast/releases/tag/v2.1.1)
- [Pinned ARIMA intervals](https://github.com/Nixtla/statsforecast/blob/264166ec0ddbfd481ee8e0b71e5ca42d2c0676fc/python/statsforecast/arima.py#L1390-L1477)
- [Pinned AutoETS simulation API](https://github.com/Nixtla/statsforecast/blob/264166ec0ddbfd481ee8e0b71e5ca42d2c0676fc/python/statsforecast/models.py#L888-L954)
- [Pinned ETS simulation](https://github.com/Nixtla/statsforecast/blob/264166ec0ddbfd481ee8e0b71e5ca42d2c0676fc/python/statsforecast/ets.py#L1370-L1456)
- [Pinned SeasonalNaive intervals](https://github.com/Nixtla/statsforecast/blob/264166ec0ddbfd481ee8e0b71e5ca42d2c0676fc/python/statsforecast/models.py#L3758-L3786)
