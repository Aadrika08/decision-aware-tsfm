# Aadrika's forecast/cache smoke test — #3

Scope: one development channel, two models, two 24-hour windows. No full grid,
M5/PJM, Omega implementation, or K3 changes. Configuration and cache decisions:
[ADR 0002](../decisions/0002-forecast-smoke.md). Sub-issues: #8 and #9.

## Run

From this checkout on Linux (the cache writer uses `flock`):

```bash
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python requirements/smoke.txt
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/decision-aware-forecast-smoke dry-run
.venv/bin/decision-aware-forecast-smoke generate --limit-jobs 3
.venv/bin/decision-aware-forecast-smoke generate
.venv/bin/decision-aware-forecast-smoke analyze
.venv/bin/decision-aware-forecast-smoke dry-run --project-jobs 1000
```

Generation downloads the pinned ETT Arrow file and Moirai checkpoint if absent.
It defaults to CUDA for Moirai; `--device cpu` is explicit and produces distinct
Moirai job IDs. The lock targets the tested Python 3.11 Linux environment.
Analysis needs only NumPy and PyArrow, not Torch, model weights, or source data.
Use the same checkout and device argument when consuming a transferred cache.

## Cache handoff to #2 and #4

Default: `artifacts/runs/cache/forecast-smoke/`. Each file lives at
`model=<model>/dataset=ett1_H/<job_id>.parquet`. Share the four files selected by
`planned_jobs()` together with this checkout and the `runs/` ledgers; do not
glob stale jobs left by an older script/configuration. Raw predictions are
ignored by Git. The [v0.1.0-smoke release](https://github.com/Aadrika08/decision-aware-tsfm/releases/tag/v0.1.0-smoke)
publishes `forecast-smoke-sample.zip`, containing the measured four-job sample
and its run ledgers. See [independent release verification](../results/cache/forecast-smoke-verification.md)
for the checksum, a Windows-compatible verification command, and the precise
limits of the available evidence.

One row is one forecast step. Identity columns: `job_id`, `dataset`, `config`,
`series_id`, `model`, `family`, `origin`, `horizon`, `seed`; payload: `step`
(1–24), hourly target `timestamp`, `y_true`, and `q_0.01`, `q_0.05`, `q_0.09`,
`q_0.25`, `q_0.40`, `q_0.50`, `q_0.60`, `q_0.75`, `q_0.91`, `q_0.95`, `q_0.99`.
Parquet schema metadata `job` contains the split, context, channel, revisions,
hashes, quantile method and settings; `measurement` contains inference time,
device, peak GPU allocation and any sampling warnings. Run ledgers additionally
record installed versions, load/write/wall time and generated/skipped jobs.
Ledgers are finalized on successful command exit; completed Parquet jobs survive
an abrupt exit, but that interrupted invocation's final wall time is unavailable.

Read a file with `pyarrow.parquet.ParquetFile(path).read()` to avoid inferring
Hive partition types that conflict with the explicit identity columns. Prefer
`read_job()` for configuration, completeness and quantile validation.

The writer locks the cache, validates complete jobs, then publishes each file by
atomic rename. Reruns validate and skip existing jobs. Partial temporary files
are not predictions. Corrupt completed files fail loudly: preserve the evidence
and choose a fresh `--cache` directory rather than silently replacing them.
Changing the script, lock, settings or source changes job IDs.

Pinball is `max(tau*(y-q), (tau-1)*(y-q))`. Mean costs use all 48 target steps
per representative. Regret subtracts the better fixed representative's mean
cost at that tau, not a per-step oracle. Units are raw target units, equivalent
to decision cost with underage/overage costs summing to one. These are interface
samples, not enough series for K2b/K3 conclusions or full-family comparisons.

## Reproduce the resume check

After the interrupted/resumed commands above:

```bash
.venv/bin/decision-aware-forecast-smoke generate --cache artifacts/runs/cache/forecast-smoke-uninterrupted
.venv/bin/python - <<'PY'
import numpy as np
from decision_aware_tsfm.cache import forecast_smoke as s
for job in s.planned_jobs():
    a, q, _ = s.read_job('artifacts/runs/cache/forecast-smoke', job)
    b, r, _ = s.read_job('artifacts/runs/cache/forecast-smoke-uninterrupted', job)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(q, r)
print('All four jobs match exactly')
PY
```

Use fresh cache paths to repeat generation, not the already-completed files.
The offline tests also simulate interruption before atomic publication, reject
incomplete/crossing/stale caches, check leakage and pinball arithmetic, and prove
that completed-cache resume never calls either loader.

## Measured result (2026-09-09 local time)

Python 3.11.15, Torch 2.4.1, RTX 4050 Laptop GPU; pinned environment recorded in
[forecast-smoke-results.json](../results/cache/forecast-smoke-results.json).

- 4 complete jobs, 96 rows, 11 quantiles; all actuals and quantiles matched
  exactly between resumed and uninterrupted runs (maximum absolute gap 0).
- Completed-cache rerun skipped all four jobs with zero data/model loading.
  A fresh analysis process did not import Torch.
- Mean pinball at tau 0.5: SeasonalNaive 4.315344, Moirai 5.514882. At tau
  0.91: 2.822248 versus 2.016869. No research gate is inferred from these values.
- Uninterrupted run: 3.6815 seconds excluding initial downloads/environment
  installation; model loading 2.7137 seconds. Moirai batches: 0.3867 seconds
  first and 0.1064 seconds second. Peak allocated GPU memory: 1,180,523,520 bytes.
- The separately resumed Moirai jobs each paid first-inference overhead:
  0.4340 and 0.3947 seconds. Their mean projects 1,000 otherwise-identical jobs
  to 0.1151 GPU-hours, excluding loading and I/O. SeasonalNaive projects to
  0.0001021 CPU-hours. Two timings are not a robust throughput benchmark.

CUDA categorical sampling warns that its kernel lacks strict determinism. Seeds
and exact local repeatability were verified; cross-device/version repeatability
is not guaranteed. CPU generation requests strict deterministic algorithms but
has not been benchmarked here.

## Full-grid budget remains open

The plan proposes approximately 36 configurations, 8 models and 3 horizons;
it does not freeze the retained series/window counts. For each model and horizon,
the required estimate is `sum(number_of_jobs * measured_seconds_per_job) / 3600`,
plus loading, I/O and safety margin. Quantiles come from one distribution draw
and must not multiply the inference count by eleven.

This run measures only SeasonalNaive and Moirai at h=24 on this GPU. Other
models, horizons, and T4 throughput are unmeasured. Freeze the actual counts and
collect approved tiny timing probes before confirming the 22–30 GPU-hour budget.
Do not run the full grid to fill that gap. Parent #3 remains open for that budget
decision and owner-reviewed integration.
