# Cache handoff for Veer — issue #3

This documents the actual published v1 smoke cache. It is usable now for decision
engine fixtures; it does not freeze the production workload, stable-ID algorithm,
model roster, or new forecasting methods. Follow-up under parent #3: make the
existing schema directly consumable and document adapter evidence.

## Columns and types

One row is one forecast step for one model, series, origin, horizon and seed.
Column order below matches the released Parquet files exactly.

| Columns | Arrow dtype | Meaning |
|---|---|---|
| `job_id` | string | Content hash for real v1 jobs; `(job_id, step)` is unique |
| `dataset` | string | Source repository, e.g. `Salesforce/GiftEval` |
| `config` | string | Source configuration, e.g. `ett1/H` |
| `series_id` | string | Series/channel identity within the source configuration |
| `model` | string | `seasonal_naive` or `moirai_1.1_R_small` in the release |
| `family` | string | Exactly `classical` or `tsfm` |
| `origin` | int64 | Zero-based index of the first forecast target |
| `horizon` | int64 | Number of forecast steps in this job |
| `seed` | int64 | Recorded random seed |
| `step` | int64 | One-based forecast step, 1 through horizon |
| `y_true` | float64 | Observed target, in original units |
| `timestamp` | string | ISO 8601 target time; released data is hourly and timezone-naive |
| `q_0.01`, `q_0.05`, `q_0.09`, `q_0.25`, `q_0.40`, `q_0.50`, `q_0.60`, `q_0.75`, `q_0.91`, `q_0.95`, `q_0.99` | float64 each | Eleven quantile forecasts, in that order |

The physical Arrow fields allow nulls, but the contract forbids null values.
Actuals/forecasts must be finite. Quantiles must be nondecreasing; ties are valid.
Do not infer dtypes from partition directory names: use
`pyarrow.parquet.ParquetFile(path).read()` for each partition and concatenate.

Schema metadata keys `job` and `measurement` contain UTF-8 JSON. `job` records
source revision/hash, item/channel, frequency, context and split bounds,
checkpoint/settings, quantile method, code/environment hashes and job identity.
`measurement` records inference seconds, device, GPU allocation and warnings.
The payload validator checks shape/types/values/completeness only. Use the
existing release verifier for real v1 provenance and identity verification.

## Model evidence as of 2026-09-19

This table describes the historical handoff. The later
[2026-09-21 synthetic qualification](../results/cache/2026-09-21-pilot-adapter-qualification.md)
passed all four approved pilot adapters at horizons 1/24/168; use the
[development pilot runbook](development-pilot.md) for the current integration steps.

| Model | Evidence available | Limits / action needed |
|---|---|---|
| Moirai-1.1-R-small | Passed all eleven quantiles in the released test | One ETT channel, h=24, context=2048, 4096 distribution samples, NumPy linear quantiles. Tail calibration and other horizons are not established. |
| SeasonalNaive | Passed all eleven quantiles in the released test | Last-day forecast plus empirical lag-24 residual quantiles. This wrapper was specified only for h=24. |
| AutoETS, AutoARIMA | No adapter-contract pass in our evidence | Need an approved predictive-distribution method and tests. |
| TimesFM 2.5, TTM r2 | No adapter-contract pass in our evidence | Need quantile-interface verification before admission. |
| Chronos-Bolt-base/small | Not cleared for the full quantile grid | ADR 0002 records native 0.1–0.9 coverage and endpoint clamping outside it. No fix has been verified here. This blocks the current interface, not proof that every possible approach is impossible. Do not fabricate tails or silently substitute repeated endpoints. |

Sources: `docs/decisions/0002-forecast-smoke.md`, the published
[v0.1.0-smoke release](https://github.com/Aadrika08/decision-aware-tsfm/releases/tag/v0.1.0-smoke),
and `docs/results/cache/forecast-smoke-release-verification.json`.
These are historical smoke passes, not production approvals.

## Build fixtures immediately

With `src` on PYTHONPATH and NumPy/PyArrow installed:

```python
from decision_aware_tsfm.cache.schema import SCHEMA, TAUS, decision_fixture, validate_payload
table = decision_fixture()
validate_payload(table)
```

This four-row fixture is invented, clearly marked, and contains no model output.
For both steps, y=10; classical predicts 8 at every quantile and TSFM predicts 11.
Expected mean pinball: classical = `2*tau`; TSFM = `1-tau`.
At tau=0.5: costs 1 and 0.5, regrets 0.5 and 0.
At tau=0.91: costs 1.82 and 0.09, regrets 1.73 and 0.
The oracle is the lower mean cost at each tau. Veer owns the evaluation engine;
this helper only supplies forecasts with known answers.

## Joins and decisions still to coordinate

- Do not join on the current manifest's global counter or row position. The
  existing smoke ID is `item_0_dim0`; Dayson must supply an explicit mapping to
  the final stable identity. Preserve source/config/item/channel and source revision.
- Omega is not a prediction column. Join a separately versioned Omega table to
  the matching series AND history/cutoff definition. Do not attach full-series
  Omega to an earlier origin if it contains future observations.
- Keep horizons, origins, seeds and evaluation splits explicit; match actuals
  and windows across models before comparisons. Do not pool incompatible runs.
- The four-model pilot has one TSFM. Its within-family choice is trivial, not a
  comparison among TSFMs. Team decision: add a contract-verified second TSFM or
  defer comparative within-TSFM selection until expansion. No roster change here.
- Production manifest hashes and stable-ID mapping need agreement before real
  partitions replace fixtures; any incompatible payload change gets a new version.

## Verification command

```text
python -m unittest tests.integration.test_cache_schema -v
```

Set PYTHONPATH to `src` if the package is not installed. The release compatibility
test reads all four real files when `cache/forecast-smoke-sample.zip` is present;
otherwise it reports a skip. The fixture tests require neither Torch nor a GPU.
