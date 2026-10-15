# First measured real pilot slice — October 15

Owner: Aadrika. Part of #29/#30, parent #16. Generation source is frozen at
`d2e1b1b40334ee89207be3eab4078b0d291d346a`; documentation changes are isolated.
Authority is [Veer's October 15 bounded-pilot decision](https://github.com/Aadrika08/decision-aware-tsfm/issues/29#issuecomment-6185355042),
explicitly accepted by the operator. No Pratham signature is claimed.

## Real result

All **24 jobs / 1,544 forecast rows** completed and passed the saved-plan audit.
No missing, corrupt, unexpected or duplicate partitions, mismatched actuals or
failed jobs were found. All eleven quantiles are finite and ordered. Both Omega
variants pass the existing consumer ingress checks. This is real CPU/CUDA model
execution on the approved Electricity series, not a synthetic fixture.

| Measurement | Observed |
| --- | ---: |
| End-to-end measured wall time | 454.1865 seconds |
| Maximum process RSS | 1,557,468 KiB |
| Sampled device memory maximum | 2,421 MiB |
| Sampled GPU utilization maximum | 78% |
| GPU sample count / interval | 454 / one second |
| CPU | AMD Ryzen 7 7735HS, one worker / BLAS thread |
| GPU | RTX 4050 Laptop, driver 610.57.04 |

Six jobs per model (three horizons, two seeds) used the same first origin:

| Model | Sum of inference seconds | Mean per job |
| --- | ---: | ---: |
| AutoARIMA native | 435.7050 | 72.6175 |
| AutoETS simulation | 9.8365 | 1.6394 |
| SeasonalNaive native | 0.0055 | 0.0009 |
| Moirai-small sampling | 1.3137 | 0.2190 |

Inference includes classical fitting and quantile construction. Model/source
loading, publication and verification add other work; the table is not an
end-to-end speed comparison. RSS is process high-water memory, not a summed
process tree. GPU samples are device-wide, include other processes and can miss
short peaks; they are distinct from PyTorch allocations.

## Download and reproduce checks

The [real first-slice release](https://github.com/Aadrika08/decision-aware-tsfm/releases/tag/pilot-first-2026-10-15)
contains the cache bundle, separate evidence archive and `SHA256SUMS-first`.
GitHub's uploaded asset digests equal the local hashes:

| Artifact | SHA-256 |
| --- | --- |
| pilot-first-bundle-2026-10-15.tar.gz | b86feb412b216b1316f7ee465c1e561719aaab1d1c0b3f24f1a61410248479bf |
| pilot-first-evidence-2026-10-15.tar.gz | 5516cf81e07c35c23818b81035a7046aaddc58c3bf5e078f546f61687ae85c10 |
| Bundle handoff.json | 96a6dcd295d91f3a4887c27fa3fd315dfec2c3e664675db35f1457857d6719c7 |

The bundle contains the original inputs, separately labelled approved execution
inputs, all 55 source arrays, both Omega variants, the five-file role export,
the immutable first plan, copied-bundle audit, hashes and local ingress reports.
Evidence retains the untouched original ZIP, preflight, full/first plans, role
exports, measurement/generation reports and raw GNU-time/GPU/stdout/stderr logs.
The [JSON record](2026-10-15-real-pilot/first-slice.json) retains exact values.

After unpacking the bundle, use the preserved qualification Python and current
source code on `PYTHONPATH=src`:

```bash
python scripts/forecast_cache.py audit --plan-file first-bundle/plan.json --cache first-bundle --output first-bundle-audit.json
python -c "from decision_aware_tsfm.analysis.decision.handoff import read_handoff; print(read_handoff('first-bundle', 'current')[2])"
python -c "from decision_aware_tsfm.analysis.decision.handoff import read_handoff; print(read_handoff('first-bundle', 'last_4096')[2])"
```

The original raw workload SHA is `1474953e...`; adding only its approval field
produced raw SHA `32f1e574...`. Both exact files and hashes are retained. The
planner's canonical full/subset workload hashes are different identities again;
do not equate them with raw file hashes. The original workload at bundle root
lets the analysis identify the exact #29 packet; approved inputs document the
actual generation provenance. The approved execution copy was not moved after planning.

## Remaining execution and acceptance

The first slice was delivered before starting the approved **21,696-job /
1,395,776-row** continuation in the same cache. All scientific settings remain
unchanged. The measured schedule keeps one CPU worker, one BLAS thread and batch one.

This slice has only one series-origin. It cannot support the validation
comparators or K3 boundary analysis; `run_pilot` correctly requires complete
origin coverage for each represented series/horizon. Local ingress checks do
not substitute for Veer's or Abdullah's explicit acceptance of the real cache.

AutoARIMA dominates this slice. Although the budget estimator's twelve timing
signatures are all covered, this is not representative timing across 55 series
or 904 origins. No full-pilot finish date or final-study compute budget is inferred.

The first slice alone used **0.124013 CPU resource hours and 0.001450 CUDA
resource hours**, with **479,885 bytes** of Parquet output. Resource occupancy
sums measured inference plus explicit loading/I/O components; it is distinct
from the 454-second measured elapsed time. A separate proposed 10% restart plus
10% failure scenario gives 0.148816 CPU and 0.001740 CUDA resource hours.
These allowances are not observed costs or approval for the full study.

## Continuation snapshot and operation

The [timestamped continuation record](2026-10-15-real-pilot/continuation.json)
records the running full pilot and its monitor. A controlled SIGTERM interrupted
the continuation after 24 verified cache hits, before another partition was
published. Its 196.1125 seconds of canceled attempt time are retained separately
from successful inference. The partial audit found 24 valid and 21,672 missing
partitions, with no corruption, unexpected files, duplicates or mismatched actuals.
The resumed run skipped all 24 existing partitions; their hashes still match.

A detached completion monitor is active in the frozen execution worktree. After
generation, it requires a complete audit, a rerun generating zero jobs, recovery
of two disposable missing/corrupt copies, and resource accounting before packaging.
It uploads a draft release, verifies its asset digests, then publishes and requests
Veer's and Abdullah's actual consumer acceptance on #29/#30. It stops on error;
it neither closes these issues nor infers acceptance.

For local live status, use the frozen worktree; the JSON record in this PR is a
snapshot, not a live counter:

```bash
cd cache/worktrees/real-pilot-29
cat artifacts/runs/pilot-2026-10-15/completion-status.json
tail -n 20 artifacts/runs/pilot-2026-10-15/finish-supervisor.log
```

Keep this host awake and preserve the execution checkout, inputs and run paths.
The detached processes survive the terminal session, but do not restart after a
reboot. If the status is `stopped_on_error`, preserve the reports and investigate
before restarting; do not launch a second generator or edit scientific settings.

#29 remains open until the full audit, recovery/completed-rerun checks and both
owner acceptances. #30 remains open until real coverage, actual resource costs
and the measured base plus separately labelled proposed allowances are reported.
M5/PJM and the full study grid remain untouched.
