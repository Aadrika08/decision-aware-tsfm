# Partial pilot status and measured parallel execution, 23 October

Owner: Aadrika. [#71](https://github.com/Aadrika08/decision-aware-tsfm/issues/71),
under [#29](https://github.com/Aadrika08/decision-aware-tsfm/issues/29) / #30.

## Current state and completed coordination

The October 20 team report assumed the pilot had continued running. It was
explicitly stopped on October 16; both original services are inactive/dead.
The [status correction](https://github.com/Aadrika08/decision-aware-tsfm/issues/29#issuecomment-6370014290)
records that the old October 20–21 scenario is no longer a current estimate.

A fresh full-plan audit verifies **575 / 21,696 jobs**, with **21,121 missing**
and zero corrupt, unexpected, duplicate or actual-value-mismatched partitions.
Saved counts are AutoARIMA 557, AutoETS 6, Moirai 6 and SeasonalNaive 6.
This is incomplete coverage, not a failed completeness guard or a finished pilot.
The [audit summary](2026-10-23-parallel-pilot-resume/saved-cache-audit-summary.json)
records the counts; the complete path-level report is retained in the ignored artifacts.

PR #69 was merged at `7e73c3fb58e4d649e2a7ac76cb419fe2c6c7e41e` after checking
both owners' approvals, four successful checks and no unresolved review threads.
Those approvals concern first-slice interfaces and historical evidence; they are
not full-cache acceptance. Abdullah's requested complete Omega input was already
in the durable first-slice bundle: `first-bundle/omega.json`, 904 records for each
of `current` and `last_4096`. The status correction supplies its link and SHA.

## Small execution change

`measure_pilot.py --workers N` now forwards a positive integer to the existing
CPU process pool and records `cpu_workers`. The Python API defaults to one.
Invalid counts fail before output creation or spawning. BLAS remains one thread
per worker; GPU execution remains serial. No forecast adapter, executor, planner,
schema, model, seed, horizon, source, split or Omega definition changed.

The original 21,696-job plan still passes the executor's code/environment and
canonical-identity validation. Implementation hash remains
`5aa32a3436d5a9cc3ac93d79f1f9892dc6db1969d932bfa33ade0aac07967e36`;
environment hash remains
`86d8741605a3e5f29262eb6bdbadae072285a5935f0255c685c659a9d546f8fe`.
The wrapper has a separate recorded hash, so scheduling changes remain visible.

A separate source-relocation check copied and hashed all 55 arrays, changed only
source paths, revalidated all 21,696 identities, and audited the existing cache
against the relocated plan: the same 575 partitions pass with zero anomalies.
The original plan bytes remain unchanged. The
[relocation record](2026-10-23-parallel-pilot-resume/relocation-verification.json)
covers portability without running a model.

## Real parallel smoke, separate from the production cache

On the qualified Python 3.11.15 environment, a fresh disposable cache regenerated
four already-approved CPU jobs: AutoARIMA and AutoETS, both seeds, horizon 168,
on the first electricity series-origin. It used two CPU workers, one BLAS thread,
`RuntimeMaxSec=270`, `TimeoutStopSec=15`, `KillMode=control-group`, restart disabled,
MemoryHigh 2,500 MiB, MemoryMax 3 GiB and MemorySwapMax 1 GiB.

**4 generated / 0 failed / 0 skipped; 672 rows; full subset audit complete.**
All payload columns of all four jobs exactly equal their saved reference partitions.
Measured elapsed time was **84.1046 seconds**. GNU time maximum process RSS was
**212,204 KiB**; systemd's service-cgroup MemoryPeak was **591,183,872 bytes**
(about 564 MiB). These are different measurements. The observed memory-event
counts were zero. The completed smoke service was stopped; the full pilot was
not restarted.

The [verification record](2026-10-23-parallel-pilot-resume/smoke-verification.json)
pins the four IDs and scope. Its subset-plan `counts_by_model` header used detailed
objects; direct per-model counts, canonical jobs and total rows were checked.
The preserved original smoke plan is not rewritten; future subset examples use
the planner's integer count format. This metadata presentation does not change
the verified forecasts, and is not used for any throughput admission.

This is a functional parallelism check on one series-origin, not representative
throughput, eight-worker RAM qualification or a measured two-day completion bound.
No benchmark partitions were added to the 575-job production cache.

## Prepared transfer artifact

The ignored `artifacts/runs/parallel-71-20261023/` directory contains
`pilot-resume-2026-10-23.tar.gz` (48,498,484 bytes) and its `.sha256` file.
Archive SHA-256:
`823c44d67eb155b9cf96e1c2dea270635b4ad29c288d83216b9f56394727437a`.
It preserves 913 original stopped-run files (476,641,178 bytes before compression),
excluding `.writer.lock`, plus separately labelled acceleration proof. Every original
file was rehashed inside the archive and compared again with the unchanged source.

The [transfer manifest](2026-10-23-parallel-pilot-resume/transfer-manifest.json)
pins code `1937c7ba11e045449ef5469685b405ab51da58f3`. The archive includes original
and approved inputs, all saved partitions and attempts, source arrays, roles and
plans. It is staged locally for transfer to the selected host, not uploaded as a
completed consumer cache. The Python environment and pinned Moirai checkpoint
are installed/verified separately using the runbook. PR #72 publishes the code;
#29/#30 remain open.

## Runtime choice and remaining work

The [saved resource baseline](2026-10-23-parallel-pilot-resume/resource-baseline.json)
has 557 AutoARIMA measurements: mean 80.5998 s, median 69.6704 s, largest
observed 334.1606 s. Most are horizon 1. The 4,867 remaining AutoARIMA jobs imply
108.97 serial worker-hours, or 13.62 hours with ideal eight-worker scaling.
Other models, startup, contention, audits and packaging add work. No full laptop
run is justified under the operator's six-hour limit.

For a stronger host, use the [parallel resume runbook](../../runbooks/pilot-parallel-resume.md).
The proposed target is 16 vCPUs / about eight CPU cores, 64 GiB RAM and one
modest NVIDIA GPU. Admit the actual host only after a declared benchmark covering
all five datasets and three horizons sustains at least **139.1 AutoARIMA jobs/hour**,
with acceptable memory and separate checks of other-model time. That criterion
uses a proposed 20% allowance plus six hours reserved for other work; it is not
a guarantee. No host was rented and no representative remote benchmark has run.

Raw seed-copying and cross-horizon reuse were not introduced. Although saved
deterministic pairs match, proper derived provenance, timing accounting and
identity migration require a separate implementation. The existing process pool
is the smaller route to the requested two-day alternative.

The user-stopped wrapper's elapsed time **42,260.6709 s** is known from its
interrupted report; its maximum RSS is unknown. This must not be confused with
the earlier OOM-killed attempt, whose wrapper totals remain unknown. Preserve
both, plus the 25 original recovered measurements, as separate attempts.

Validation for this change: **54 measurement, executor, planner and budget tests
passed** under the qualified environment. Five Bash blocks and two embedded
Python snippets in the runbook pass syntax checks; remote commands are not
claimed executed. The real smoke and relocation checks above are separate proof.

Still required: actual admitted-host execution, full audit, completed zero-generation
rerun, disposable missing/corrupt recovery, complete hashed publication, both
consumers' acceptance of identical complete bytes and the final measured budget.
Neither #29/#30 nor parent #16 is complete.
