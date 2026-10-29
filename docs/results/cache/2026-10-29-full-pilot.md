# Full pilot forecasts and producer handoff — 2026-10-29

Owner: Aadrika. References: #29, #30; parent #16.

**All approved forecasts are complete. Consumer-owner acceptance is pending.**
The cache contains 21,696 verified jobs and 1,395,776 forecast rows. The decision
reader passed producer checks for both Omega variants. The K3 reader reached a
900-second operational timeout without returning its final report; this is an
incomplete consumer check, not an invalid-cache finding.

## Download and identity

[Full pilot release](https://github.com/Aadrika08/decision-aware-tsfm/releases/tag/pilot-full-2026-10-29).
Download both `pilot-full-*-l4-20261029.tar.gz` archives and `SHA256SUMS-full`.

- Forecast bundle SHA-256: `4ec7b9ad45c921581562cc7bbb118c0988b318179fd4e5cfac97ae904cfc03a5`.
- `full-bundle/handoff.json` SHA-256: `82d2234855ba4b67df0aa0b2994946c1a46c70498375600378a6e6eb5b7aa0a9`.
- Evidence archive SHA-256: `f983217fdce4538d9bc059576ed413d7521dd31478833ea4e84eb2fc0ac4bf99`.
  Its digest is also included in `SHA256SUMS-full`; its internal
  manifest hashes every captured file and checks all 913 original transport files
  and all 21,696 primary partitions.
- `independent-audit.json` and `independent-verification.json` record an additional
  restore and audit on a separate filesystem using the same scientific source.
- `publication-readback.json` records a fresh GitHub download of all nine
  archive/report/checksum assets, matching local hashes and GitHub asset digests.

The evidence archive retains original inputs, prior attempts, native logs,
qualification, hardware records, relocation proof, all closeout checks, budgets,
the successful decision-reader checks and the unsuccessful K3 timing probe.
Historical prepared scripts and superseded status files are retained as history;
they are not evidence that their proposed executions occurred.

## Frozen scientific scope

Resumed-generation source commit: `1937c7ba11e045449ef5469685b405ab51da58f3`.
Earlier saved jobs retain their original recorded revisions and measurements.
The run used the same 99 package pins and Moirai checkpoint
`0c24ab99db2c1a70ea2a0fc03bf113329772ac64` throughout this grant.

The approved workload has 55 hourly series from electricity, ETT1, ETT2, solar and
Jena Weather; 904 origins; context 2,048; horizons 1/24/168; seeds 42/43; batch one;
and taus 0.01/0.05/0.09/0.25/0.40/0.50/0.60/0.75/0.91/0.95/0.99.
The models are `autoarima_native`, `autoets_simulated`, `seasonal_naive_native`
and `moirai_pilot`. `current` Omega remains primary and `last_4096` remains the
sensitivity variant. No datasets, windows, splits, model settings, metrics or
Omega definitions changed. M5/PJM were not used.

## Execution and verification

The renewed L4 passed all 72 fresh qualification cases and the declared 30-job
real-data benchmark. An initial admission was refused while the provider's
Jupyter allocation was inactive; it generated no forecasts. Restarting the
provided account reacquired the same L4, verified by UUID and provider device
mapping before admission.

The measured resume ran on 2026-10-29 from 06:54:56 UTC, preserving the exact
18,084 saved jobs and generating exactly 3,612 missing Moirai jobs. Native run ID:
`de468d62e51c489ba060ef58bb9feb51`. Wrapper wall time: **748.571297 seconds**.
Its source, model and input hashes stayed fixed. Technical closeout completed at
07:18:22 UTC, and the service exited successfully.

| Check | Observed outcome |
|---|---|
| Primary generation | 3,612 generated; 18,084 skipped; zero failures |
| Full cache audit | 21,696/21,696; no missing, corrupt, duplicate, unexpected or mismatched-actual entries |
| Preserved baseline | All 18,084 saved partition bytes retained; all 25 recovered original-run measurements retained |
| Completed rerun | Zero generated; 21,696 skipped; zero failures |
| Disposable recovery | One missing and one deliberately corrupt partition; normal generation refused; repair generated exactly two; final audit clean |
| Primary cache after recovery | Original partition hashes unchanged |
| Independent restored audit | 21,696/21,696 on the downloaded bundle, using source `1937c7b` |
| Decision reader, producer execution | Both variants passed; all 55 source hashes and 1,395,776 target rows checked |
| K3 reader, producer execution | Timed out at 900.13 seconds, exit 124; no final report or data-rejection error returned |

The decision probe used source `1937c7b`, completed in 102.56 seconds, and had
maximum process RSS 2,046,808 KiB. K3 used Abdullah's merged #35 source
`e3b7e17e54e94439fc78a38057273473e6bc7985`; maximum process RSS was 4,078,520 KiB.
Both used Python 3.11.15, NumPy 1.26.4, pandas 2.2.3, PyArrow 16.1.0 and SciPy 1.11.4.
These were sequential, cache-only producer probes, not the owners' acceptance.

K3's `join_omega` filters the full table once per job (`loader.py:266–270` at the
pinned revision). That is a likely scaling bottleneck at 21,696 jobs and
1,395,776 rows, but the timeout was not profiled to establish its exact cause.
The original reader and cache were left unchanged. Retain the timeout as an
incomplete check; do not bypass provenance checks or count it as acceptance.

## Measured pilot resources

Accounting joins each published partition's `(run_id, job_id, attempt)` to its
native verified event. It covers all 21,696 jobs across eight run/device cohorts,
including the 25 recovered original CPU measurements. Failed-job placeholder
timings were excluded from successful-job totals and remain in the raw history.

| Published-job cohort | Device | Jobs | Component resource-hours |
|---|---|---:|---:|
| Recovered original, `65373ea1…` | CPU | 25 | 0.677559 |
| First slice, `85bf5431…` | CPU | 18 | 0.124013 |
| First slice, `85bf5431…` | RTX 4050 CUDA | 6 | 0.001450 |
| User-stopped laptop run, `fb1f7fc7…` | CPU | 526 | 11.713168 |
| First L4 attempt, `c2fc4225…` | CPU | 337 | 8.194477 |
| Second L4 attempt, `b3fdbda0…` | CPU | 15,366 | 150.537832 |
| Second L4 attempt, `b3fdbda0…` | L4 CUDA | 1,806 | 0.177024 |
| Renewed L4 resume, `de468d62…` | L4 CUDA | 3,612 | 0.125172 |

| Scenario | CPU resource-hours | Hours attributed to CUDA jobs |
|---|---:|---:|
| Measured successful-job components | 171.247048 | 0.303645 |
| Proposed 10% restart + 10% failure allowances | 205.496458 | 0.364374 |

These totals add inference, model loading and source/I/O components across
heterogeneous hosts and workers. They are not elapsed time, active GPU compute
time, billed machine hours, or the complete cost of every attempt. The allowances
are additive assumptions, not observed failure rates. Primary Parquet storage is
447,582,553 bytes; bundles, sources, raw logs and backups require additional space.

The complete new resume's maximum process RSS was 2,557,808 KiB. This is separate
from PyTorch allocation and sampled device memory; device samples can include
other processes. CPU per-job memory zeros are uninstrumented fields. Original
laptop wrapper records identify an AMD Ryzen 7 7735HS and RTX 4050; L4 wrapper
records identify an AMD EPYC 7R13 host. Keep the recovered run's native OS label
and missing dedicated wrapper explicit instead of inferring missing measurements.

Known wrapper walls include 454.186536 seconds for the first slice, 196.112490
for the early manual interruption, 42,260.670885 for the user-stopped laptop run,
3,794.490525 for L4 attempt one, 69,806.754223 for the failed second L4 attempt,
and 748.571297 for the successful new resume. These wrappers overlap the job
components; do not add them to those sums. Historical OOM/interrupted costs that
were not captured remain unknown. Qualification, benchmark, audit, zero-rerun,
repair, reader checks, setup and transfer are additional workflow work.

This is a bounded four-model pilot budget. It does not establish costs for an
expanded roster (including a second TSFM), new domains/frequencies, M5/PJM,
selector fitting/evaluation, or the full-study workload. Freeze and measure those
separately before making a full-study compute commitment.

## Owner handoff

Use the [consumer acceptance checklist](2026-10-15-real-pilot/consumer-acceptance.md)
for the required assertions and attestation fields. Its October 15 download names
and producer revision are historical; use the commands below for this release.
The exact producer probe scripts and their raw outputs are included under
`full-evidence/operations/new-grant-20261029/` and
`full-evidence/closeout-resume-3-new-grant/consumer-probes/`.

Set `PRODUCER` and `K3` to clean checkouts at the recorded revisions, and `PY` to
an existing environment with the versions above. These commands only read saved
forecasts. Record any owner-agreed revision change instead of silently replacing
the pins. Run each owner's reader separately and retain its actual exit status.

```bash
set -eu
test "$(git -C "$PRODUCER" rev-parse HEAD)" = 1937c7ba11e045449ef5469685b405ab51da58f3
test -z "$(git -C "$PRODUCER" status --porcelain)"
test "$(git -C "$K3" rev-parse HEAD)" = e3b7e17e54e94439fc78a38057273473e6bc7985
test -z "$(git -C "$K3" status --porcelain)"
mkdir pilot-full-acceptance-20261029
cd pilot-full-acceptance-20261029
gh release download pilot-full-2026-10-29 --repo Aadrika08/decision-aware-tsfm \
  --pattern 'pilot-full-*-l4-20261029.tar.gz' --pattern SHA256SUMS-full
sha256sum --check SHA256SUMS-full
tar -xzf pilot-full-bundle-l4-20261029.tar.gz
tar -xzf pilot-full-evidence-l4-20261029.tar.gz
export BUNDLE="$PWD/full-bundle"
export PROBES="$PWD/full-evidence/operations/new-grant-20261029"
export PYTHONDONTWRITEBYTECODE=1
sha256sum "$BUNDLE/handoff.json"
"$PY" -c 'import platform, importlib.metadata as m; print(platform.python_version()); print({p: m.version(p) for p in ("numpy", "pandas", "pyarrow", "scipy")})'
```

Veer's reader:

```bash
PYTHONPATH="$PRODUCER/src" /usr/bin/time -v -o decision.time.txt \
  "$PY" -u "$PROBES/probe_decision.py" > decision.jsonl 2> decision.stderr.log
```

Abdullah's reader (the producer's 900-second attempt did not complete):

```bash
PYTHONPATH="$K3/src" /usr/bin/time -v -o k3.time.txt \
  "$PY" -u "$PROBES/probe_k3.py" > k3.jsonl 2> k3.stderr.log
```

Veer and Abdullah each need to record on #29 the same two archive hashes and
handoff hash, their clean consumer revision, environment, exact commands and
exit statuses, both Omega-variant reports, and explicit acceptance or rejection.
Abdullah's full-cache check also needs the timeout resolved or a completed run
with its actual resource usage recorded. Producer results do not supply those
attestations. #29 and parent #16 remain open; #30's published evidence is ready
for review with the stated accounting and final-study coverage limits.
