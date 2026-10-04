# Configured forecasting and cache — #17 / parent #16

This implements the fixture foundation for [#17](https://github.com/Aadrika08/decision-aware-tsfm/issues/17).
It builds on the published schema branch and preserves the v1 consumer columns.
The production sprint supplied on 2026-09-19 has SHA-256
`a23c02eef4147b7c41d727ed0d18bc6b13f3c0aea1ff64515898ac224e34cc5c`.

The agreed first real pilot is Moirai-small, SeasonalNaive, AutoETS and AutoARIMA,
with the local RTX 4050 and CPU as the timing hosts. Comparative within-TSFM
selection waits for expansion. The methods are now accepted in
[ADR 0005](../decisions/0005-pilot-adapter-qualification.md), with a
[completed synthetic qualification](../results/cache/2026-09-21-pilot-adapter-qualification.md).
The approved real workload is still missing. No real pilot or full grid is claimed here.
The [development pilot runbook](development-pilot.md) now provides handoff validation,
the first comparison slice, Linux measurement commands and consumer acceptance checks.

## Run the complete fixture flow

Use Python 3.11 and the existing smoke environment. The configured commands and
fixture tests require NumPy, PyArrow and pandas; forecasting dependencies load only
when a real adapter needs them. A separate consumer can use the saved plan without
matching the generating machine's environment.

```bash
python scripts/forecast_cache.py fixture artifacts/runs/cache-fixture
python scripts/forecast_cache.py plan --workload artifacts/runs/cache-fixture/workload.json --models artifacts/runs/cache-fixture/models.json --output artifacts/runs/cache-fixture/plan.json
python scripts/forecast_cache.py generate --plan-file artifacts/runs/cache-fixture/plan.json --cache artifacts/runs/cache-fixture/predictions --workers 2 --limit-jobs 3
python scripts/forecast_cache.py generate --plan-file artifacts/runs/cache-fixture/plan.json --cache artifacts/runs/cache-fixture/predictions --workers 2 --output artifacts/runs/cache-fixture/resumed.json
python scripts/forecast_cache.py audit --plan-file artifacts/runs/cache-fixture/plan.json --cache artifacts/runs/cache-fixture/predictions
```

The limited first generation returns exit code **2** because the planned cache is
incomplete. The resumed generation and complete audit return 0. Rerunning the
completed generation skips every job without loading sources or models.
The `fixture` command refuses to overwrite its destination: choose a new directory
when repeating the demonstration. `--output` writes a JSON report as well as printing it.
Saved plans refuse replacement. Reports are published atomically outside the cache;
an output path cannot replace configuration, source data, or cache partitions.

The invented workload has two constant series, two origins, horizons 1/24/168,
two deterministic fixture adapters, **24 jobs / 1,544 rows / eleven quantiles**.
The first series has actuals 10: classical predicts 8, TSFM predicts 11, so their
mean pinball costs are `2*tau` and `1-tau`. The second series has actuals 7, giving
costs `1-tau` and `4*(1-tau)`. These are planted checks, not research measurements.

## Inputs and identities

The JSON workload uses `schema_version: 1`, `split: "development"`, an explicit
`fixture` boolean, the exact eleven-value `quantiles` list, and nonempty `rows`.
Each row contains:

| Group | Fields |
|---|---|
| Stable source identity | `dataset`, `config`, `series_id`, `item_id`, `channel`, `source_revision` |
| Staged source | `source_path`, `source_sha256`, `source_length`, `start_timestamp`, `frequency` |
| Usable bounds | `usable_start`, `usable_end`, `training_end`, `validation_end` |
| Forecast window | `context_start`, `context_length`, `origin`, `horizon`, `seed` |

`source_path` is relative to the workload file and remains under that directory.
For this version it identifies a non-pickled, one-dimensional NumPy `.npy` file;
`source_sha256` hashes those exact staged bytes. This is an explicit staging
interface, not a GIFT-Eval downloader or an automatic replacement for Dayson's
manifest. Dayson's export must preserve the original source revision, item/channel
mapping and chronology. It must not flatten channels or remove internal timestamps.

All indices are zero-based and end bounds are exclusive. Context ends at `origin`;
the target is `[origin, origin+horizon)`. The entire forecast must fit in the approved
validation block. Context has at least 1,000 observations. Timestamps are derived
from the source start and its pandas frequency offset, including calendar offsets;
this version requires timezone-naive input. Horizons count observations, not hours.

The model JSON uses `schema_version: 1` and a `models` list. Each entry contains
`model`, `family`, `adapter`, `adapter_version`, `checkpoint`, `device`, `batch_size`,
`quantile_method`, and `settings`. Examples come from the fixture command. See
[historical adapter qualifications](classical-adapter-qualification.md) for the
original smoke methods and the [pilot qualification runbook](pilot-adapter-qualification.md)
for the accepted four-model configurations and synthetic results.

Planning never loads source arrays or model weights. It rejects duplicate logical
windows, invalid bounds, unsupported adapter configurations and held-out identities.
Old global-counter IDs are not accepted for real workloads. No production ID mapping
is invented here.

The saved plan uses job-metadata version **2**, while forecast columns remain **v1**.
Job IDs hash source and forecast configuration, seeds, model/distribution settings,
and implementation/environment fingerprints. Source paths, whole-workload hashes,
approval references and Omega columns do not affect those IDs. Thus relocation,
an added unrelated workload row, or an additional Omega variant cannot invalidate
unchanged predictions. Source revisions and forecast-affecting settings do.

Each plan records actual environment versions, code revision and content hashes.
Generation rejects stale implementation/environment identities. Read-only audit
accepts the original `--plan-file`, including on a different consumer environment;
it needs neither the source files nor forecasting libraries.

## Execution, failures and audit

One OS-released lock protects the coordinator. CPU execution uses a bounded spawn
process pool; GPU execution uses one coordinator worker. Model instances are reused
for compatible jobs. Batch sizes above one remain unqualified for execution; the
runner does not label sequential calls as measured model batching.

Each job becomes one Parquet partition under
`model=<escaped model>/config=<escaped config>/<job_id>.parquet`.
Publication writes and syncs a temporary file in the same directory, then atomically
renames it. A temporary `.partial` file never counts as a prediction.

Metadata contains the exact job identity, measured provenance and payload checksum.
The consumer payload uses the existing `SCHEMA` and `validate_payload()` contract.
Use `pyarrow.parquet.ParquetFile(path).read()` to avoid partition-type inference.

The run directory preserves a saved plan, per-attempt events and a final run report.
At most two automatic attempts are allowed for classified transient failures.
Invalid inputs and invalid forecasts are terminal. Completed files remain reusable
even when a later ledger or invocation fails. Failed attempts retain their elapsed
time and error; successful job timings do not stand in for failed work.

Corrupt completed partitions fail loudly. `generate --repair` first preserves the
corrupt bytes in `quarantine/`, then regenerates the affected job. Valid jobs are
never overwritten. Audit reports missing, corrupt, duplicate and unexpected files,
plus disagreements in actual targets across comparable models. Extra files do not
silently expand the selected experiment.

## Budget reports

```bash
python scripts/forecast_cache.py budget --plan-file artifacts/runs/cache-fixture/plan.json --measurements artifacts/runs/cache-fixture/resumed.json
```

This intentionally exits 2 with an incomplete estimate. Hardware and overhead
evidence are missing, and fixture timings never establish a real compute budget.

For a measured scenario, provide `--hardware hardware.json` containing exact
`cpu` and/or `cuda` labels from the run measurements, and `--overheads overheads.json`
containing `loading_seconds` and `io_seconds` maps for the planned schedule. Include
source preparation and verification in I/O accounting. Supply explicit
`--restart-fraction` and `--failure-fraction` margins; defaults are zero and do not
represent team-approved margins. Pass all relevant run reports after an interruption.

Timing compatibility includes model/distribution configuration, context, horizon,
frequency, dataset configuration, device, batch size, code/environment and fixture
status. RTX 4050 observations cannot fill a T4 timing gap. Repeated copies of the
same evidence must not increase the sample count. Quantiles do not multiply jobs.
Measurements retain their run ID and attempt, so independent runs remain separate
samples while repeated copies of one run are counted once.

Resource consumption is reported separately for CPU and GPU:

`hours = (sum(jobs * matching mean inference time) + loading + I/O) * (1 + restart_fraction + failure_fraction) / 3600`

These resource sums are not parallel elapsed wall time; run reports retain measured
wall time. Missing measurements leave the corresponding estimates incomplete.
Resource coverage alone is not approval to run the full grid.

## Remaining #16 integration inputs

- Dayson: approved exact workload, stable source mapping, deduplicated Jena data,
  usable-series counts, all split/window bounds, and separately versioned Omega
  variants with their history cutoffs. Hospital currently has no usable series;
  Aadrika does not silently replace it or impose a high-frequency-only selection.
- Veer and Abdullah: acceptance of the common fixture, then actual pilot partitions.
- Aadrika: run the approved pilot on the local RTX 4050/CPU, measure all retained
  configurations, and obtain the full-workload schedule and accepted margins.

Real generation additionally requires `--real` and workload/model
`approval_reference` fields pointing to genuine recorded decisions. These fields
are provenance, not a substitute for review. Do not invent them to bypass a gate.
M5/PJM and the full grid remain excluded. #3 stays open for its full-grid budget;
#16 stays open for the real workload, pilot acceptance and the measured budget.

## Verification

```bash
PYTHONPATH=src python -m unittest tests.integration.test_cache_schema tests.integration.test_cache_adapters tests.integration.test_cache_planner tests.integration.test_cache_executor tests.integration.test_cache_budget tests.integration.test_forecast_cache_cli -v
```

The GitHub workflow runs these portable fixture checks on Linux and Windows. The
historical release additionally needs `cache/forecast-smoke-sample.zip`; the release
tests explicitly skip when that artifact is absent. No fixture test performs real
forecast inference or opens held-out data.
