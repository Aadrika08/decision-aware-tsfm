# Pilot preparation verification — 2026-09-21

Implementation: `14d9890`. PR [#31](https://github.com/Aadrika08/decision-aware-tsfm/pull/31).
Parent #16; preparation #28; real execution/acceptance #29; resource budget #30.
Reproduction and the proposed handoff format are in the
[development pilot runbook](../../runbooks/development-pilot.md).
The [JSON evidence](2026-09-21-pilot-preparation.json) records hashes, environment,
hardware, counts and the missing upstream inputs.

## Verified preparation

- All **88 local tests passed**, with no skips, including historical smoke-release
  compatibility. Portable readiness/cache tests run in Linux/Windows CI; the new
  Linux-specific measurement tests explicitly skip on Windows.
- The model-free readiness fixture plans **96 jobs / 6,176 rows** across two
  invented series/configurations and two origins. Its first complete comparison
  slice has **24 jobs / 1,544 rows**, with identical job IDs in the full plan.
  No forecast model is loaded by this check; its Omega values are invented.
- A separate native measurement run executes the existing two-constant-model
  fixture: **24 jobs / 1,544 rows**, all audited. A rerun generates zero jobs and
  skips all 24; every published partition hash is unchanged.
- On a disposable copy, one file was removed and another deliberately corrupted.
  Audit identifies 22 verified jobs and both unverified jobs. Its `missing` list
  includes corrupt partitions, so the diagnostic lists two missing and one
  corrupt. Repair regenerates exactly two jobs, preserves the corrupt bytes and
  restores matching payloads. The published original remains untouched.
- Independent review found two edge cases: source-window deduplication omitted
  declared source length, and an installed but unusable GPU probe blocked CPU-only
  measurement. Both are fixed with regression tests.

The native fixture measurement took **2.158 seconds** including the wrapper and
recorded **124,544 KiB maximum process RSS** on the local Ryzen 7 7735HS. Device-wide
GPU samples are recorded separately; this fixture did not perform GPU inference.
These measurements verify instrumentation only. They are not production throughput,
Moirai memory requirements, or evidence for the full-study compute budget.

Raw evidence is retained in ignored
`artifacts/runs/pilot-preparation-final/`: source fixtures, full/first plans,
measurement logs/reports, original/disposable caches, integrity checks and test log.
The committed JSON identifies the exact hashes of the principal evidence files.

## Real handoff remains unavailable

The current upstream `origin/feat/1-omega-validation:dev_manifest.json` still has
SHA-256 `a944b4a9b02ae2bc72e4ff6cd4fbe869a778d2ad3a2bc80498574919f7d35b4b`.
All **8,782 active rows** use counter-based `_idx_` identifiers. Their fields are
`series_id`, `subset`, `channel`, `length`, `omega_levels`, `omega_diffs`, `horizons`
and `status`; they do not supply the required forecast workload/source provenance.

Dayson must provide stable identities, staged source checksums/revisions, exact
forecast and split bounds, a separately described 4,096-history Omega comparison,
and the recorded pilot decision. The exchange format in the runbook is proposed
for agreement; it does not assert a Stage 0 schema freeze or a scientific default.
Hospital, Jena duplication and the US Births discrepancy remain Stage 0 decisions.

No real pilot, downstream consumer acceptance or real budget is claimed. No
development or held-out arrays were read. #29/#30 remain open; the full-study
requirements under #16/#3 also remain open.
