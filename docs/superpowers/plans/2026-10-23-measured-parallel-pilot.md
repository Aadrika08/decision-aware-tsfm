# Measured parallel pilot implementation plan

> **For agentic workers:** Use superpowers:executing-plans or superpowers:subagent-driven-development to implement these tasks. Keep the original run unchanged.

**Goal:** Prepare measured CPU parallelism and a portable, resumable handoff for the approved pilot without starting a run that exceeds the operator's limits.

**Architecture:** Reuse the existing bounded CPU process pool and serial GPU phase. Add a validated worker option only to the measurement wrapper, whose bytes are excluded from the canonical forecast implementation fingerprint. Preserve all existing job IDs and partitions; keep timing evidence separated by attempt and hardware.

**Tech stack:** Existing Python 3.11.15 qualification environment, unittest, StatsForecast 2.1.1, current cache executor; no new dependencies.

**Spec:** https://github.com/Aadrika08/decision-aware-tsfm/issues/71 and the operator's 23 October instruction: no full laptop run exceeding six hours; prepare a stronger-machine route targeting at most two days.

## Global constraints

- Keep 21,696 jobs / 1,395,776 forecast rows, 55 series / 904 origins, all four models, seeds 42/43, horizons 1/24/168, context 2,048, eleven taus and existing splits/Omega definitions.
- M5/PJM remain untouched. No paid host provisioning. No full laptop run.
- Preserve the stopped d2e1b1b checkout, all 575 saved partitions and every original input/attempt. A small disposable real smoke may run at most 300 seconds.
- Default CPU workers remains one; BLAS threads remain one per worker. Record any opt-in parallel attempt separately. No seed copying, shared fitting or identity migration.
- Machine-time projections are conditional admission scenarios, not guarantees or measured full-study budgets. Historical killed-attempt unknowns stay unknown.

## Review focus

- Invalid worker values (zero, negative, bool, non-integer) must fail before output creation or process spawn.
- CLI and Python API must forward the same value and record it; omission must preserve one-worker behavior.
- Existing GPU identity, output isolation, signal cleanup and measurement failures must remain enforced.
- The implementation fingerprint and all saved job IDs must stay compatible.
- No six-hour or 48-hour claim may rest on ideal scaling alone; remote full execution requires a measured admission benchmark and memory checks.

### Task 1: Worker-aware measurement

**Files:** `src/decision_aware_tsfm/cache/measurement.py`, `tests/integration/test_pilot_measurement.py`.

**Interface:** `measure(plan_file, cache, output_directory, *, real=False, workers=1)` and CLI `--workers`; report `cpu_workers` explicitly. Existing callers continue unchanged.

- [x] Run the existing focused measurement suite as baseline.
- [x] Add tests for opt-in forwarding/reporting, invalid values before side effects, and CLI behavior; observe expected failures.
- [x] Add the minimal validation, command argument and metadata, retaining all existing safeguards.
- [x] Run measurement and cache-executor integration tests; compare current implementation fingerprint with the saved approved plan.

### Task 2: Evidence and transfer handoff

**Files:** new dated result record under `docs/results/cache/`, `docs/runbooks/pilot-parallel-resume.md`; ignored artifacts under `artifacts/runs/`.

- [x] Record the fresh 575-valid / 21,121-missing audit and stopped-run status; publish the correction on #29.
- [ ] Preserve source hashes, original plan and cache bytes in a hashed transfer artifact; validate relocation without altering canonical identities.
- [x] Run one disposable real CPU smoke with two workers, a hard 300-second limit and bounded memory; compare saved reference predictions if feasible. Do not treat this as representative throughput.
- [x] Supply exact setup, audit, bounded benchmark, measured admission and resume commands. Recommend CPU capacity first; retain the same modest single-GPU requirement.
- [ ] Independently review the diff and handoff; run required checks, commit/push and open a PR closing #71 only. Leave #29/#30 open.
