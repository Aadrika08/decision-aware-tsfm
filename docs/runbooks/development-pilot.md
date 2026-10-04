# Four-model development pilot — parent #16

Owner: Aadrika. Preparation: [#28](https://github.com/Aadrika08/decision-aware-tsfm/issues/28).
Real execution/acceptance: [#29](https://github.com/Aadrika08/decision-aware-tsfm/issues/29).
Resource evidence/budget: [#30](https://github.com/Aadrika08/decision-aware-tsfm/issues/30).

The user approved this bounded implementation plan after the four-model synthetic
qualification. Schema, executor and qualification PRs #18/#19/#27 are integrated
into `infra/merge-safe-repository-structure`. This does not approve a final study
roster or provide Dayson's missing data decisions. Preserve the historical smoke.

## Dayson's required handoff

Supply the existing [workload/model document contract](configured-forecast-cache.md#inputs-and-identities):
immutable source revisions and staged NPY checksums; stable source/item/channel
identities; source start/frequency; usable, context, training and validation bounds;
exact origins/horizons/seeds; the unchanged eleven taus; and recorded approval
references. Supply a human-readable dataset/exclusion summary alongside the files.

The proposed first groups are ETT, Electricity, Solar and Jena, at their approved
hourly configurations for this pilot. Dayson chooses and freezes the exact rows
with Omega variation. The tool checks multiple series/configurations and complete
comparison coverage; it cannot infer the approved dataset roster from folder names.
Human review of the manifest against its decision remains necessary.

Resolve Jena's duplicate source, Hospital's zero usable series and the US Births
review/manifest discrepancy in Stage 0. Long-history M4 daily/weekly/monthly and
Hierarchical Sales daily are not excluded by an invented frequency policy. Wider
frequency/model qualification is a later milestone. Aadrika does not edit Stage 0,
select a scientific Omega default, or manufacture stable IDs from counter IDs.

### Proposed Omega exchange document

The readiness checker accepts the following cache-side exchange format. It is a
proposal for Dayson's export, not a claim that Stage 0 already emits or has frozen
this schema. Agree the mapping before admitting a real pilot. Values remain in
this sidecar; they are never inserted into forecast payloads or forecast identities.

```json
{
  "schema_version": 1,
  "fixture": false,
  "variants": {
    "current": "Exact current method, levels/differences choice and implementation revision",
    "last_4096": "Same method/revision applied to at most the last 4096 observations"
  },
  "rows": [
    {
      "dataset": "source repository",
      "config": "approved configuration",
      "series_id": "upstream stable identity",
      "source_revision": "immutable source revision",
      "source_sha256": "64 hexadecimal characters",
      "origin": 5000,
      "variant": "current",
      "history_start": 0,
      "history_end": 5000,
      "omega": 0.4
    }
  ]
}
```

The example is illustrative and incomplete: supply one row per declared variant
and source/origin. The corresponding `last_4096` row above starts at 904 and ends
at 5000. History bounds are zero-based, end-exclusive and cannot extend past the
forecast origin. Both roles use the same cutoff; capped history starts at
`max(current.history_start, current.history_end - 4096)`. The checker reports actual
history lengths: a cap is not proof that every series has 4,096 observations.

Additional named variants, including separately described differences, are
supported when present for every source/origin. Do not relabel full-history Omega
as a capped value. Undefined/nonfinite values require an upstream decision, not
fabricated numbers. Descriptions must identify calculation provenance; this tool
validates supplied values/joins and does not recompute or certify their method.

## Validate and plan without inference

Use the separate pinned [qualification environment](pilot-adapter-qualification.md).
Prepare a model document containing `schema_version: 1`, the genuine
`approval_reference` for the bounded real pilot, and the four configurations from
`decision_aware_tsfm.cache.pilot.pilot_models()`. ADR 0005 approved synthetic
qualification; do not present it alone as approval of a real workload.

```bash
PYTHONPATH=src python scripts/check_pilot_readiness.py --workload HANDOFF/workload.json --models HANDOFF/models.json --omega HANDOFF/omega.json --output-directory artifacts/runs/pilot-ready
```

This validates the entire metadata/coverage/Omega handoff before opening sources,
then checks staged hashes, numeric shapes and finite context/target windows using
the existing executor loader. It never loads forecast models. It requires the
four qualified methods, hourly context 2,048, horizons 1/24/168, seeds 42/43, batch
one and complete model comparisons for every source/origin.

Success saves `report.json`, the immutable `plan.json` and `first-plan.json`.
The first plan selects the first stable series ID, then dataset/configuration and
earliest origin, and includes all four models, horizons and seeds: 24 jobs / 1,544
forecast rows. Its job IDs also exist in the full plan. Quantiles are columns.
Relative source paths retain their original base through absolute paths in plans.

Missing/invalid handoffs exit 2 with a blocked report and no executable plans.
Invented inputs produce `fixture_valid`, which is not real-pilot approval. Existing
output directories are refused. Keep the input files immutable and retain the
report's workload, model, Omega, implementation and environment fingerprints.

## Progressive measured generation

The wrapper fixes one CPU worker, one OpenBLAS/OMP/MKL thread and the original
batch-one adapter settings. Moirai runs on the sole visible local RTX 4050;
ambiguous physical GPU selection is rejected. It requires Linux and
`/usr/bin/time`; GPU sampling uses `nvidia-smi` at one-second intervals.

```bash
PYTHONPATH=src python scripts/measure_pilot.py --plan-file artifacts/runs/pilot-ready/first-plan.json --cache artifacts/runs/pilot-cache --output-directory artifacts/runs/pilot-first-measurement --real
PYTHONPATH=src python scripts/forecast_cache.py audit --plan-file artifacts/runs/pilot-ready/first-plan.json --cache artifacts/runs/pilot-cache --output artifacts/runs/pilot-first-audit.json
PYTHONPATH=src python scripts/measure_pilot.py --plan-file artifacts/runs/pilot-ready/plan.json --cache artifacts/runs/pilot-cache --output-directory artifacts/runs/pilot-full-measurement --real
PYTHONPATH=src python scripts/forecast_cache.py audit --plan-file artifacts/runs/pilot-ready/plan.json --cache artifacts/runs/pilot-cache --output artifacts/runs/pilot-full-audit.json
```

Complete the first audit and deliver that comparison slice before extending the
cache. Publish complete per-origin slices and the matching sub-plan; do not
describe partially generated comparisons as a complete consumer artifact. Keep
cache generation under its existing lock; concurrent readers use published files.

Outputs include copied plan, `generation.json`, stdout/stderr, GNU time output,
raw GPU samples/probe and `report.json`. Generation exit status and measurement
completeness are separate. Interrupted runs preserve reports and stop children;
rerun with a fresh measurement directory and the same cache to resume. All run
reports, including failed attempts, belong in the evidence bundle.

Maximum process RSS is an operating-system process high-water measurement, not
summed process-tree memory. Sampled device memory includes other GPU processes
and can miss short peaks; PyTorch allocations remain a different per-job metric.
The historical CPU zero sentinel is not measured memory. Record GPU utilization
during GPU inference separately from the preceding CPU-only phase; investigate
sustained utilization below 30%, with above 60% an operational target rather than
a reason to alter research settings. Parallel tuning is outside this milestone.

## Consumer acceptance and budget

Give each consumer the same saved plan, verified cache, Omega sidecar, input
fingerprints, per-partition SHA-256 list and audit report. Archive those files with
relative paths and give the archive its own checksum. No custom payload conversion.

- Veer: run his cache-only pinball/decision interface with matched actuals/windows.
- Abdullah: load the stable-ID/Omega join and run his K3 input checks.
- Each records command, code/environment, artifact hashes, result and explicit
  acceptance on #29. A fixture pass or someone else's assertion is not acceptance.
- Verify interrupted resume, a completed rerun with zero generated jobs, and
  missing/corrupt partition detection on a disposable copy. Preserve corrupt
  bytes before repair; never damage the published cache for testing.

Reuse `forecast_cache.py budget` with the full pilot plan and both generation
reports. Construct the hardware labels expected by those reports and retain the
wrapper's actual CPU/GPU identity with them. Use only evidence from that measured
machine/configuration; the old CPU platform label alone is not a hardware identity.
Provide explicit schedule totals for loading and source/publication/verification
I/O without double counting fitting (already included in inference).

Publish the zero-margin measured base and a second scenario using
`--restart-fraction 0.10 --failure-fraction 0.10`. The latter is a **proposed 20%
combined resource allowance**, not an accepted production margin. Report CPU/GPU
resource hours separately from wall time, output storage and the slowest measured
component. Missing real measurements leave the estimate incomplete; synthetic
qualification and fixture timings cannot fill coverage gaps.

#28 can close when preparation is verified. #29 requires the real pilot and both
consumer acceptances; #30 requires real resource evidence and its bounded budget.
#16/#3 remain open for their full-study dependencies. The sole pilot TSFM does not
resolve comparative within-TSFM representative selection. No M5/PJM or full grid.
