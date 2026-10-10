# Four-model synthetic qualification — 2026-09-21

Issue [#20](https://github.com/Aadrika08/decision-aware-tsfm/issues/20), parent
[#16](https://github.com/Aadrika08/decision-aware-tsfm/issues/16).
Methods: [ADR 0005](../../decisions/0005-pilot-adapter-qualification.md).
Inference implementation commit: `66a0d1f` (the later CI-only test fix leaves the
recorded inference-source hashes unchanged).

## Result

**All 72 actual-model prediction attempts passed.** Four models covered horizons
1/24/168, two fixed invented 2,048-step hourly contexts, and seeds 42/43 with one
additional seed-42 repeat per case. Every output had eleven finite, ordered
quantiles. All 24 repeated-seed comparisons were exactly equal in this run, and
the adapters preserved their input contexts.

An independent artifact check verified both source hashes, all 72 forecast hashes,
shapes, finite values and ordering, all 24 repeat comparisons, the twelve complete
model/horizon groups, and the current inference code against the recorded hashes.
The [JSON record](2026-09-21-pilot-adapter-qualification.json) contains exact
configurations, source definitions, package/code fingerprints, timings and warnings.

The complete run took **353.27 seconds** on an AMD Ryzen 7 7735HS and NVIDIA RTX
4050 Laptop GPU with 6,141 MiB. NumPy/SciPy classical execution used one configured
OpenBLAS/OMP/MKL thread. Moirai used CUDA and the cached pinned checkpoint with
network access disabled through the offline Hugging Face settings.

| Model | Attempts passed | h=1 prediction seconds | h=24 prediction seconds | h=168 prediction seconds |
|---|---:|---:|---:|---:|
| AutoARIMA native Gaussian | 18/18 | 96.793 | 104.953 | 106.022 |
| AutoETS seeded simulation | 18/18 | 4.311 | 6.513 | 21.774 |
| SeasonalNaive native Gaussian | 18/18 | 0.0042 | 0.0039 | 0.0038 |
| Moirai-small | 18/18 | 4.363 | 0.660 | 0.991 |

Each timing cell is the **sum of six attempts**, mixing the positive seasonal
signal and negative constant signal. Classical prediction times include fitting;
first-call overhead is visible in Moirai's h=1 group. These are synthetic
qualification observations, not representative production throughput. Loading
and fitting times are separately recorded in JSON.

Moirai's largest observed PyTorch allocation was **1,264,442,880 bytes** at h=168.
This is allocated tensor memory, not total device reservation. CPU peak memory is
not instrumented; inherited zero-valued fields must not be interpreted as measured
zero memory usage or used to approve a CPU memory budget.

The GPU emitted a warning that its cumulative-sum kernel lacks a deterministic
implementation under deterministic-warn mode. Exact observed repeats do not imply
bitwise reproducibility on every run, device or library version. The warning is
preserved in the result and original run report.

## Reproduce

Build the separate hash-pinned environment using the
[qualification runbook](../../runbooks/pilot-adapter-qualification.md). This lock
retains all 85 historical smoke package pins and adds 14 dependencies. All 99
installed packages passed the dependency consistency check.

```bash
PYTHONPATH=src python -m unittest discover -v
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/qualify_pilot_adapters.py artifacts/runs/pilot-qualification-2026-09-21 --device cuda
```

Use a new output directory for a rerun. Offline mode requires the pinned Moirai
checkpoint already present in the local model cache. The local suite covers 69
tests, including the historical smoke release; Linux and Windows CI also cover the
portable cache and qualification orchestration tests. CI uses mocks for model
execution and is distinct from the actual CPU/GPU run reported here.

Raw arrays and `report.json` are retained under
`artifacts/runs/pilot-qualification-2026-09-21/`. Logs, tests, CPU identification and
the earlier one-horizon API preflight are under
`artifacts/runs/qualification-verification/`. Generated artifacts remain ignored by
Git; this note and the compact JSON result are committed.

## Limits and next dependency

This qualifies the approved adapter interfaces on two synthetic contexts. It does
not establish calibration, superiority, a final roster, a real development pilot,
or a production compute budget. No development dataset or M5/PJM data was read.
The historical 96-row smoke artifact remains separately verified.

Next, Dayson must supply the frozen workload with stable source identities and
exact forecast/split bounds. Aadrika can then run the real pilot, obtain Veer and
Abdullah's consumer acceptance, and combine representative measurements with the
final workload and accepted margins. The full grid remains gated under #16.
