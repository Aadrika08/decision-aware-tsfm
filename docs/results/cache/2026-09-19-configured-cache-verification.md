# Configured cache fixture verification — 2026-09-19

Issue [#17](https://github.com/Aadrika08/decision-aware-tsfm/issues/17), parent
[#16](https://github.com/Aadrika08/decision-aware-tsfm/issues/16).
Verified implementation commit: `ee274714d929cf423c1534ef233e71e2cf6c7087`.
This is implementation evidence using invented data, not a development pilot or
scientific result.

## Observed results

The independent local run passed **60 tests with zero failures and zero skips**,
including the published historical smoke archive. The suite took 15.064 seconds
on Linux with Python 3.11.15. Package versions, implementation/environment hashes,
release checksum, commands and timings are saved in the
[machine-readable record](2026-09-19-configured-cache-verification.json).

The documented two-worker fixture flow produced:

| Check | Observed |
|---|---:|
| Planned jobs / rows | 24 / 1,544 |
| Quantiles per forecast row | 11 |
| Limited first run | 3 generated, incomplete as expected |
| Resume | 21 generated, 3 skipped |
| Repeat after completion | 0 generated, 24 skipped |
| Completion audit | 24 verified jobs |
| Resumed versus uninterrupted payloads | Equal for every job |
| Completed partition bytes after repeat | Unchanged |
| Budget without real timing/overhead evidence | Incomplete, as required |

The suite also checks corrupted and missing partitions, quarantine/repair,
duplicate planning, transient and terminal failures, interruption, errors after
publication without repeated inference, exclusive writer locking, held-out guards,
source checksums/types, forecast bounds, provenance, report overwrite protection,
independent timing samples and incompatible hardware measurements.

## Reproduce

Use the pinned smoke environment and run from the repository root:

```bash
PYTHONPATH=src python -m unittest discover -v
```

The historical archive must be present at `cache/forecast-smoke-sample.zip` to
avoid release-test skips. Obtain it from release `v0.1.0-smoke`; the tests verify
its pinned checksum. Follow the
[configured cache runbook](../../runbooks/configured-forecast-cache.md) for fixture,
plan, limited generation, resume, audit and budget commands. Choose a fresh fixture
directory. The JSON record contains the exact commands for this run, including a
second uninterrupted generation. Automated integration tests reproduce the
payload comparison and completed-job reuse checks.

Raw local reports and partitions are under
`artifacts/runs/fixture-verification-20260919-203915/`; test output is under
`artifacts/runs/implementation-verification/`. These generated artifacts are ignored
by Git. The recorded summaries and reproduction commands are committed.

## Remaining work and limits

Linux and Windows fixture/release checks are configured in the PR workflow; the
local results above do not establish Windows execution. See the PR's CI results
for the remote platform checks.

The first Windows CI run exposed two platform-specific failures in the existing
contention and corruption/repair tests: Windows forbids reading a locked byte,
and a failed Parquet constructor retained an open file through its traceback.
The follow-up fix checks lock-file size without reading the locked region and
closes the underlying file explicitly on every Parquet read path. The full local
suite then passed all 60 tests again in 13.518 seconds. The recorded fixture run
above remains tied to its original implementation commit; the latest PR checks
cover the follow-up fix.

The source staging interface currently accepts a single real numeric NPY series
with preserved chronology and explicit source identity. Model batching above one
remains unqualified. CPU jobs use bounded processes; GPU jobs run serially. The
budget reports resource consumption, not an inferred parallel wall-clock schedule.

The historical Moirai-small and SeasonalNaive configurations remain narrowly
qualified to their existing smoke paths. AutoETS/AutoARIMA distribution methods,
the stable-ID approved workload, real RTX 4050/CPU pilot, downstream consumer
acceptance and measured production budget remain parent #16 work. Comparative
within-TSFM selection awaits expansion. No new real forecasts or held-out data
were used, and no full-grid budget or completion claim is made.
