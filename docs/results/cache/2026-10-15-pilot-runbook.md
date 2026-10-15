# Pilot runbook for a rented GPU (fallback) - October 15

Owner: Veer. Part of [#29](https://github.com/Aadrika08/decision-aware-tsfm/issues/29).
The pilot belongs to Aadrika. This runbook is the fallback if her run has not
started by 16 October and the team agrees that Veer runs it on a rented or
borrowed machine. It writes down the exact steps; it does not run anything.
No forecasting model was loaded, no real forecast was made and M5/PJM were not
opened while writing it.

Everything below comes from the code on main `d209ae7`
(`src/decision_aware_tsfm/cache/`, unchanged at `1261e8b` along with `scripts/` and `requirements/`), Aadrika's preflight in PR
[#58](https://github.com/Aadrika08/decision-aware-tsfm/pull/58)
(`docs/results/cache/2026-10-12-pilot-execution-preflight.md` and its
`report.json`), the [development pilot runbook](../../runbooks/development-pilot.md),
the [qualification runbook](../../runbooks/pilot-adapter-qualification.md),
[ADR 0005](../../decisions/0005-pilot-adapter-qualification.md) and the dated
notes in this folder. Numbers marked "checked on 15 Oct" come from runs made
while writing this note (Python 3.11.15, NumPy 1.26.4, pandas 2.2.3,
PyArrow 16.1.0, no forecasting packages); see "How this runbook was checked".

## 0. Before starting

- The approval exists: the team decision record on #29,
  `https://github.com/Aadrika08/decision-aware-tsfm/issues/29#issuecomment-6185355042`
  (15 Oct). It names the ZIP hash, the `workload.json` hash and the four models.
  The planner only checks that `approval_reference` is a non-empty string
  (`cache/planner.py`, `_approval`); whether this record is enough approval is a
  team question, not a code one. #58 asked for Pratham's written approval.
- Tell Aadrika and the team before starting, so two people do not run the same pilot.
- The rented machine must be Linux on x86_64 with exactly one NVIDIA GPU. Section 1 says why.

## 1. Hardware

### What the repo records

| What | Value | Where |
| --- | --- | --- |
| GPU used for Moirai | NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB total, 5,796 MiB free at probe, driver 610.57.04 | name: `2026-09-21-pilot-adapter-qualification.json` and the `gpu_probe` in `2026-09-21-pilot-preparation.json`; memory and driver: #58 note |
| CPU used for classical models | AMD Ryzen 7 7735HS, 16 logical CPUs, Linux x86_64 | `2026-09-21-pilot-adapter-qualification.json` |
| Moirai largest PyTorch allocation | 1,264,442,880 bytes (about 1.18 GiB) at horizon 168; this is allocated tensor memory, not total GPU use | `2026-09-21-pilot-adapter-qualification.md` |
| Synthetic qualification, 72 attempts, all four models | 353.27 s wall time | same note |
| Free disk on Aadrika's machine at preflight | 43,367,694,336 bytes | #58 `report.json`, `runtime.disk_available_bytes` |
| Process memory | Not measured for any real model. CPU peak memory is not instrumented (the zero values in the JSON are placeholders, not measurements). The only RSS figure, 124,544 KiB, is for the invented two-constant fixture | `2026-09-21-pilot-preparation.md` |
| Real-data runtime, memory, output size | None recorded. #30 is still open; the first 24-job slice is what produces the first real numbers | #58 note |

Per-attempt prediction times in the synthetic qualification (sum of six attempts
divided by six, from `per_model_horizon` in the qualification JSON):
AutoARIMA 16.1 to 17.7 s, AutoETS 0.7 to 3.6 s, SeasonalNaive under 0.001 s,
Moirai 0.11 to 0.73 s (the 0.73 s group includes the first-call warm-up).
These are invented 2,048-step inputs, not the real series.

Rough arithmetic only, not a budget: the pilot has 5,424 jobs per model
(21,696 / 4). At the synthetic averages that is about 25.8 CPU hours for
AutoARIMA, 2.7 for AutoETS and 0.5 GPU hours for Moirai. AutoARIMA dominates,
so CPU cores matter more than GPU size. With one worker (what `measure_pilot.py`
forces) the full pilot would take more than a day; with 8 CPU workers and even
scaling, about 3.6 hours for the classical jobs. Replace this with the real
first-slice timings as soon as they exist.

### What to rent

- Linux x86_64. The locked environment (`requirements/qualification.txt`)
  pins `triton==3.0.0` and the `nvidia-*-cu12` packages with hashes. Checked on
  15 Oct: pip finds no macOS arm64 wheel for `triton==3.0.0` or
  `nvidia-cublas-cu12==12.1.3.1`, and does find a `manylinux_2_17_x86_64`
  wheel for triton. `measure_pilot.py` also refuses anything but Linux
  ("Pilot resource measurement requires Linux") and needs `/usr/bin/time`
  (GNU time) and `/proc/cpuinfo`.
- Exactly one NVIDIA GPU on the machine. `measure_pilot.py` stops with
  "GPU measurement requires exactly one physical CUDA device" when
  `nvidia-smi` lists more than one. At least 6 GB of GPU memory matches the
  qualified run. torch 2.4.1 in the lock uses CUDA 12.1 libraries, so the
  driver must support CUDA 12.
- As many CPU cores as is affordable (8 to 16 is reasonable), because of AutoARIMA.
- Disk: no measured size for the environment or the cache. Aadrika had about
  43 GB free; aim for at least that. The first slice records `output_bytes` per
  job, which gives the real cache size estimate (times 21,696 / 24).
- RAM: no measured figure. `measure_pilot.py` records maximum process RSS for
  the first slice; use that before choosing `--workers`.

### Can the classical models run on a CPU-only machine, separately from Moirai?

On a CPU-only **Linux x86_64** machine: yes, if it has the identical
environment. On Veer's Mac (Apple M4 Pro): no, not with the pinned lock.

The reason is the job identity. `job_id` is the SHA-256 of the job identity
(`planner.py`, `job_identity`, `IDENTITY_FIELDS`), which includes every field of
`MODEL_FIELDS` (`model, family, adapter, adapter_version, checkpoint, device,
batch_size, quantile_method, settings`) plus `implementation_sha256` (the five
files `planner.py, executor.py, adapters.py, pilot.py, schema.py`) and
`environment_sha256` (Python version, including the patch number, and the
versions of numpy, scipy, pandas, pyarrow, torch, uni2ts, statsforecast,
statsmodels, coreforecast, utilsforecast and threadpoolctl). Checked on 15 Oct:

- Adding or changing `approval_reference` does **not** change any job_id.
- Changing Moirai's `device` from `cuda` to `cpu` gives every Moirai job a new
  job_id (classical job_ids stay the same), and the readiness check then blocks:
  "Pilot requires exactly the four qualified CPU/RTX 4050 model configurations".
  So never change `device`.
- Changing only the Python patch version (3.11.15 to 3.11.14) changes every job_id.
- `generate` refuses a plan made in a different environment or code version:
  "Plan implementation or environment fingerprint differs from this executor".

So a cache made on another machine is consistent with Aadrika's plan only when
that machine has Python 3.11.15, the same locked package versions and the same
five cache files. Then the job_ids are identical to hers, because none of the
five files changed between the main she tested (`7ecc09a`) and `d209ae7`
(checked with `git diff --stat 7ecc09a d209ae7 -- src/decision_aware_tsfm/cache`).
The CPU machine still keeps `device: "cuda"` for Moirai in the plan; it just does
not run those jobs (see the split option in section 4).

## 2. Set up the machine

Python version: 3.11. Aadrika's environment is Python 3.11.15
(`2026-09-21-pilot-preparation.json`); use exactly 3.11.15 so the fingerprint
matches. `uv` lists `cpython-3.11.15-linux-x86_64-gnu` as a download (checked 15 Oct).

On the Mac, package the code without putting a GitHub login on the rented machine:

```bash
cd ~/Documents/algoverse/decision-aware-tsfm
git fetch --all --prune
git bundle create ~/pilot-code.bundle origin/main
scp ~/pilot-code.bundle USER@HOST:pilot-code.bundle
scp "$HOME/Downloads/Handoff from Algoverse AI Research.zip" USER@HOST:HANDOFF.zip
```

On the rented machine (use `tmux new -s pilot` so a dropped connection does not stop a run):

```bash
nvidia-smi                                   # must list exactly one GPU
test -x /usr/bin/time || sudo apt-get install -y time   # GNU time, needed by measure_pilot.py
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11.15

mkdir -p ~/pilot/tools && cd ~/pilot
git init decision-aware-tsfm
git -C decision-aware-tsfm fetch ~/pilot-code.bundle refs/remotes/origin/main
git -C decision-aware-tsfm checkout --detach FETCH_HEAD
git -C decision-aware-tsfm rev-parse HEAD   # the current origin/main (1261e8b on 15 Oct). What must match is implementation_sha256 below.

uv venv --python 3.11.15 ~/pilot/venv
uv pip sync --python ~/pilot/venv/bin/python --require-hashes ~/pilot/decision-aware-tsfm/requirements/qualification.txt
uv pip check --python ~/pilot/venv/bin/python
export PILOT_PY=$HOME/pilot/venv/bin/python
```

Installing the package itself is not needed. The pilot commands run it from the
checkout with `PYTHONPATH=src`, exactly as in #58 (`scripts/forecast_cache.py`
also adds `src` to the path itself). If you still want it installed, this does
not change any job_id, because the fingerprint lists only Python and the eleven
packages above:

```bash
uv pip install --python "$PILOT_PY" --no-deps -e ~/pilot/decision-aware-tsfm
```

Check the fingerprints (expected values: `environment_sha256` is the value
recorded for Aadrika's environment in `2026-09-21-pilot-preparation.json`,
recomputed on 15 Oct from its recorded versions; `implementation_sha256` was
computed on 15 Oct from `d209ae7`):

```bash
cd ~/pilot/decision-aware-tsfm
PYTHONPATH=src "$PILOT_PY" -c "from decision_aware_tsfm.cache.planner import _environment, _implementation, canonical, digest; e = _environment(); print(e); print('environment_sha256', digest(canonical(e).encode())); print('implementation_sha256', _implementation())"
# environment_sha256    86d8741605a3e5f29262eb6bdbadae072285a5935f0255c685c659a9d546f8fe
# implementation_sha256 5aa32a3436d5a9cc3ac93d79f1f9892dc6db1969d932bfa33ade0aac07967e36
```

If either differs, stop: the cache would not line up with Aadrika's plan. In the
fingerprint output, a `torch` version with a suffix such as `2.4.1+cu124` means
torch came from somewhere other than the lock.

Confirm CUDA (expect `2.4.1+cu121 12.1 True` and the GPU name):

```bash
"$PILOT_PY" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Download the pinned Moirai checkpoint once, then check it against the hashes
in #58 (`model.safetensors`: 55,320,200 bytes, sha256 `cc46831272ea07e99d78fe05c09e730c7c76a1b272cddbdaf5173dae371cb238`;
`config.json`: sha256 `37b9eca5a0c3e58c2cdac3ecdbf897acbd009a20d6ba0f8a2ae0e8d92002fd47`):

```bash
SNAP=$("$PILOT_PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Salesforce/moirai-1.1-R-small', revision='0c24ab99db2c1a70ea2a0fc03bf113329772ac64'))")
sha256sum "$SNAP/model.safetensors" "$SNAP/config.json"
stat -L -c %s "$SNAP/model.safetensors"
```

The adapter loads exactly this repository and revision (`adapters.py`,
`MOIRAI_REPOSITORY`, `MOIRAI_CHECKPOINT`). After this step all runs use
`HF_HUB_OFFLINE=1`, so nothing else is downloaded.

Recommended: rerun the repo's own synthetic qualification on this GPU (invented
inputs only, 72 attempts, 353.27 s on the RTX 4050). ADR 0005 qualified the
RTX 4050; a different GPU is a change worth recording on #29.

```bash
cd ~/pilot/decision-aware-tsfm
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "$PILOT_PY" scripts/qualify_pilot_adapters.py artifacts/runs/pilot-qualification-rental --device cuda
# expect: status=passed attempted=72 passed=72 failed=0
```

## 3. Inputs

Dayson's ZIP stores paths with Windows backslashes (`sources\electricity_H__MT_003.npy`).
Unpack it with this script, which turns them into normal folders, checks the ZIP
hash, the four metadata hashes recorded in #58 and all 55 source hashes listed in
`workload.json`. The files are data from a teammate, so the script lives outside
the data folder and runs with `-I`.

```bash
cat > ~/pilot/tools/unpack_handoff.py <<'PY'
import hashlib, json, sys, zipfile
from pathlib import Path, PurePosixPath
archive, out = Path(sys.argv[1]), Path(sys.argv[2])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(archive) == "70e814bf323178df84b771f920f6a6114c6796cb5b143ba28ccce4833eb56ba0", "zip hash differs from #58"
out.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        name = PurePosixPath(info.filename.replace("\\", "/"))
        if info.is_dir() or name.name == "":
            continue
        if name.is_absolute() or ".." in name.parts:
            sys.exit(f"unsafe entry: {info.filename}")
        target = out.joinpath(*name.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(z.read(info))
expected = {"workload.json": "1474953eb7898c3f65986b6c54974af02b52b0be63eb39b58bdc246d3d979516",
            "models.json": "75ed1fc5ab57935563436d3aa81012ae59b34089e9fed0e16b641eeba784098e",
            "omega.json": "dd31de6c67927214478ff9d627eac31bacbe06fc8b12a3fa04c364196b195885",
            "summary.md": "2107cc3c26e61eb9b287040c9fa004296c66cbe4ef378e83db8f3a9231cdeade"}
for name, value in expected.items():
    assert sha(out / name) == value, f"{name} hash differs from #58"
rows = json.loads((out / "workload.json").read_text(encoding="utf-8"))["rows"]
sources = {r["source_path"]: r["source_sha256"] for r in rows}
bad = [p for p, h in sources.items() if sha(out / p) != h]
assert not bad, f"source hashes differ: {bad}"
print(f"ok: zip and 4 metadata hashes match #58; {len(sources)} source files match workload.json")
PY
cd ~/pilot/decision-aware-tsfm
mkdir -p artifacts/runs/pilot-handoff-2026-10-12
"$PILOT_PY" -I ~/pilot/tools/unpack_handoff.py ~/HANDOFF.zip artifacts/runs/pilot-handoff-2026-10-12/HANDOFF
# expect: ok: zip and 4 metadata hashes match #58; 55 source files match workload.json
```

### Add the approval reference

The edit is one new line near the top of both `workload.json` and `models.json`,
right after `"schema_version": 1,`:

```json
"approval_reference": "https://github.com/Aadrika08/decision-aware-tsfm/issues/29#issuecomment-6185355042",
```

Do it on a copy, never on the original folder. This script copies the whole
folder (sources included, because `source_path` is relative to the workload
file) and adds the line to both files:

```bash
cat > ~/pilot/tools/approve_handoff.py <<'PY'
import json, shutil, sys
from pathlib import Path
source, target, workload_ref, model_ref = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
for ref in (workload_ref, model_ref):
    if not ref.startswith("https://github.com/Aadrika08/decision-aware-tsfm/"):
        sys.exit("approval_reference must be the real approval comment URL on the team repo")
shutil.copytree(source, target)
for name, ref in (("workload.json", workload_ref), ("models.json", model_ref)):
    document = json.loads((target / name).read_text(encoding="utf-8"))
    if "approval_reference" in document:
        sys.exit(f"{name} already has approval_reference")
    document = {"schema_version": document.pop("schema_version"), "approval_reference": ref, **document}
    (target / name).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
print("approved copy written to", target)
PY
APPROVAL=https://github.com/Aadrika08/decision-aware-tsfm/issues/29#issuecomment-6185355042
"$PILOT_PY" -I ~/pilot/tools/approve_handoff.py artifacts/runs/pilot-handoff-2026-10-12/HANDOFF artifacts/runs/pilot-handoff-2026-10-12/HANDOFF-approved "$APPROVAL" "$APPROVAL"
sha256sum artifacts/runs/pilot-handoff-2026-10-12/HANDOFF-approved/*.json
```

What this changes, by design (checked on 15 Oct, in memory, with placeholder
text, no plan kept):

- The planner hashes the whole workload document, top-level keys included
  (`planner.py`, `hashed_workload = dict(workload)`), so `workload_sha256` changes,
  and so does the file hash `1474953e...` recorded in #58. #58 recorded file
  hashes only; it saved no real plan, so there is no earlier workload hash to keep.
  Record the new file hashes from the `sha256sum` line above on #29.
- No job_id changes: `approval_reference` is not an identity field. It is copied
  into each job as `approval_reference` and `model_approval_reference`.
- The script rewrites the two files with plain line endings (the originals use
  Windows line endings). The parsed content is the same apart from the new key;
  the planner reads parsed content, not bytes.

The plans store **absolute** source paths. Do not move or rename
`HANDOFF-approved` after the next step.

## 4. Run

All commands run from `~/pilot/decision-aware-tsfm`. Use a new output folder
for every check or attempt and keep every failed or interrupted attempt; the
scripts refuse to overwrite their outputs.

```bash
cd ~/pilot/decision-aware-tsfm
export PILOT_PY=$HOME/pilot/venv/bin/python
export PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### 4a. Readiness check and plan (this is the dry run)

There is no `--dry-run` flag in `cli.py`. The readiness check is the dry run:
it validates the whole handoff, verifies every source window and writes the
full and first-slice plans without loading any model.

```bash
"$PILOT_PY" scripts/check_pilot_readiness.py --workload artifacts/runs/pilot-handoff-2026-10-12/HANDOFF-approved/workload.json --models artifacts/runs/pilot-handoff-2026-10-12/HANDOFF-approved/models.json --omega artifacts/runs/pilot-handoff-2026-10-12/HANDOFF-approved/omega.json --output-directory artifacts/runs/pilot-ready
```

Expected (checked on 15 Oct with placeholder approval text): `"status": "ready"`,
21,696 planned jobs, 1,395,776 rows, 55 series, 904 origins, 2,712 source windows
verified, and a first slice of 24 jobs / 1,544 rows on `electricity_H__MT_003`,
origin 34104. The folder holds `report.json`, `plan.json` and `first-plan.json`.
A plain plan without the readiness checks is also possible
(`scripts/forecast_cache.py plan --workload ... --models ... --output ...`), but
the pilot uses the readiness plans.

Quick look at the plan (expect 21696 jobs, 5424 per model, and the environment
hash from section 2):

```bash
"$PILOT_PY" -c "import json; p = json.load(open('artifacts/runs/pilot-ready/plan.json', encoding='utf-8')); print(p['planned_jobs'], p['counts_by_model'], p['provenance']['environment_sha256'])"
```

### 4b. First 24-job slice (Aadrika's #58 commands, only `PILOT_PY` changed)

Do not type `set -e` in the interactive shell: later commands exit 2 by design and would close the shell. If you already typed it, run `set +e`.

```bash
export PILOT_PY=$HOME/pilot/venv/bin/python
export PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PILOT_PY" scripts/check_pilot_readiness.py --workload artifacts/runs/pilot-handoff-2026-10-12/HANDOFF-approved/workload.json --models artifacts/runs/pilot-handoff-2026-10-12/HANDOFF-approved/models.json --omega artifacts/runs/pilot-handoff-2026-10-12/HANDOFF-approved/omega.json --output-directory artifacts/runs/pilot-ready
"$PILOT_PY" scripts/measure_pilot.py --plan-file artifacts/runs/pilot-ready/first-plan.json --cache artifacts/runs/pilot-cache --output-directory artifacts/runs/pilot-first-measurement --real
"$PILOT_PY" scripts/forecast_cache.py audit --plan-file artifacts/runs/pilot-ready/first-plan.json --cache artifacts/runs/pilot-cache --output artifacts/runs/pilot-first-audit.json
```

If 4a already ran, skip its line here: the readiness output folder must be new.
`measure_pilot.py` always uses one CPU worker, one BLAS thread, batch one and the
single GPU; it prints `{"status": "complete", "evidence_kind": "real_pilot", ...}`
on success and saves `report.json`, `generation.json`, `time.txt`, `gpu.csv`
and logs in its output folder. The audit must say `"status": "complete"` with
`"verified_jobs": 24`.

Then package and deliver the first slice (section 5, with `first-plan.json` and
`pilot-first-audit.json`) before running the rest, as #58 asks.

### 4c. The rest of the pilot, with resume

Both options below skip the 24 finished jobs (a verified partition is a
"verified cache hit") and only run what is missing. Rerunning after an
interruption is the same command with a new output name.

Option A, as written in the development pilot runbook: one worker, with
resource measurement for #30. By the rough arithmetic in section 1 this takes
more than a day.

```bash
"$PILOT_PY" scripts/measure_pilot.py --plan-file artifacts/runs/pilot-ready/plan.json --cache artifacts/runs/pilot-cache --output-directory artifacts/runs/pilot-full-measurement --real
```

Option B, faster: CPU jobs in parallel. `--workers` is the number of CPU worker
processes (default 2); Moirai jobs always run one at a time in the main process,
after the CPU jobs (`executor.py`, `generate`). Keep one BLAS thread per worker,
as the measured runs did.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PILOT_PY" scripts/forecast_cache.py generate --plan-file artifacts/runs/pilot-ready/plan.json --cache artifacts/runs/pilot-cache --workers 8 --real --output artifacts/runs/pilot-full-generation-1.json
```

Flags, from `cli.py`:

- `--real`: required for real inputs. Without it: "Real generation requires real=True".
- `--workers N`: CPU worker processes. Use at most the number of cores.
- `--limit-jobs N`: run only the first N jobs that are still missing (in job_id
  order, so models are mixed). Useful for working in chunks. The command then
  exits with code 2 and `"status": "incomplete"`, because the audit at the end
  covers the whole plan; that is expected (checked on 15 Oct on the fixture).
- `--repair`: move corrupt partitions to `pilot-cache/quarantine/` and run them
  again. Without it, a corrupt file stops the run (see section 6).
- `--output`: the run report (status, generated, skipped, failed, per-job
  measurements, audit). It must be outside `--cache`.

After an interruption: Option A, rerun with a new `--output-directory`
(for example `pilot-full-measurement-2`). Option B, rerun with a new `--output`
(for example `pilot-full-generation-2.json`). A finished cache rerun reports
`generated: 0` and every job skipped (checked on 15 Oct on the fixture).

Split option, only if the classical jobs run on a second, CPU-only Linux x86_64
machine: set it up exactly as section 2 (both fingerprints must match; the CUDA
packages in the lock install on a machine without a GPU). Split the plan by
device on the GPU machine, then copy `artifacts/runs/plan-cpu.json` and
`artifacts/runs/pilot-handoff-2026-10-12/HANDOFF-approved/` to the **same
absolute path** on the CPU machine, because plans store absolute source paths:

```bash
cat > ~/pilot/tools/split_plan.py <<'PY'
import json, sys
from pathlib import Path
plan_path, field, value, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
plan = json.loads(plan_path.read_text(encoding="utf-8"))
jobs = [job for job in plan["jobs"] if str(job[field]) == value]
if not jobs or out.exists():
    sys.exit("no matching jobs, or output exists")
sub = dict(plan, jobs=jobs, planned_jobs=len(jobs), planned_rows=sum(j["horizon"] for j in jobs),
           counts_by_model={m: sum(j["model"] == m for j in jobs) for m in sorted({j["model"] for j in jobs})})
out.write_text(json.dumps(sub, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(out, sub["planned_jobs"], sub["planned_rows"])
PY
"$PILOT_PY" -I ~/pilot/tools/split_plan.py artifacts/runs/pilot-ready/plan.json device cpu artifacts/runs/plan-cpu.json
"$PILOT_PY" -I ~/pilot/tools/split_plan.py artifacts/runs/pilot-ready/plan.json device cuda artifacts/runs/plan-cuda.json
# on the CPU machine (with the exports from the top of section 4 and the three *_NUM_THREADS=1 exports):
"$PILOT_PY" scripts/forecast_cache.py generate --plan-file artifacts/runs/plan-cpu.json --cache artifacts/runs/pilot-cache-cpu --workers 8 --real --output artifacts/runs/pilot-cpu-generation-1.json
# on the GPU machine:
"$PILOT_PY" scripts/forecast_cache.py generate --plan-file artifacts/runs/plan-cuda.json --cache artifacts/runs/pilot-cache --real --output artifacts/runs/pilot-cuda-generation-1.json
# when both have finished, pull the CPU cache into the main cache on the GPU machine, then run the full audit in 4d:
rsync -a --ignore-existing --exclude .writer.lock CPUHOST:pilot/decision-aware-tsfm/artifacts/runs/pilot-cache-cpu/ artifacts/runs/pilot-cache/
```

Each half-plan report ends `incomplete`, with the other half's files under `unexpected`; that is expected. Judge only by the full-plan audit in 4d. The CPU half also re-runs the 18 classical first-slice jobs into its own cache; `--ignore-existing` keeps the delivered first-slice files in the main cache unchanged.

The device counts are 16,272 CPU jobs and 5,424 CUDA jobs (checked on 15 Oct).
Job_ids and file paths do not depend on which plan file a job came from; on the
fixture, two half-plans run into two caches and merged gave the same partition
paths as one run, and the full-plan audit was complete (checked on 15 Oct).

### 4d. Audit, completeness and corruption checks

```bash
"$PILOT_PY" scripts/forecast_cache.py audit --plan-file artifacts/runs/pilot-ready/plan.json --cache artifacts/runs/pilot-cache --output artifacts/runs/pilot-full-audit.json
```

The audit opens no sources and no models. It must say `"status": "complete"`
and `"verified_jobs": 21696`, with every list empty:

- `missing`: planned jobs with no valid file at their path (a corrupt file is listed here too).
- `corrupt`: files that fail to read, fail the checksum in their metadata, have the
  wrong columns, crossing quantiles or misaligned timestamps.
- `unexpected`: files that belong to no planned job, or sit at the wrong path.
- `duplicate`: one job found in more than one file.
- `mismatched_actuals`: two jobs for the same series, origin and horizon with different `y_true`.

The development pilot runbook also asks for a resume and corruption test on a
**disposable copy**, never on the real cache. On a copy, delete one partition
and overwrite another, then run `audit` (it reports both), `generate` without
`--repair` (it stops), and `generate --repair --real` (it quarantines the bad
file and makes the two jobs again, so this runs two real forecasts). On the
fixture: 22 verified, 2 missing, 1 corrupt, then repair generated 2 and skipped
22 (checked on 15 Oct).

Optional, for #30 (Aadrika's issue): `scripts/forecast_cache.py budget` takes the
plan and the run reports (`--measurements`), plus `--hardware` and `--overheads`
JSON files and `--restart-fraction` / `--failure-fraction`; see the development
pilot runbook for how to fill them.

## 5. Hand-off to the engine and to Abdullah's K3 loader

Both consumers read the same bundle: a folder with `handoff.json` (a list of
partition paths and their SHA-256 hashes, the naming header and the Omega hash),
the Parquet partitions, and `omega.json`. The engine's reader is
`analysis/decision/handoff.py` (`read_handoff`). Abdullah's loader
(`load_k3_inputs(base_dir, partition_hashes, omega_path, variant, ..., omega_sha256=...)`
on branch `origin/exp/24-k3-input-loader`) takes the same `partition_hashes`
and Omega file. The development pilot runbook lists what each consumer gets:
saved plan, verified cache, Omega sidecar, input fingerprints, per-partition
SHA-256 list and audit report.

Cache layout (`executor.py`, `job_path`):
`pilot-cache/model=<model>/config=<config>/<job_id>.parquet`, one job per file,
v1 columns (`job_id, dataset, config, series_id, model, family, origin, horizon,
seed, step, y_true, timestamp, q_0.01 ... q_0.99`), with the job identity,
the measurement and a payload checksum in the Parquet metadata. `pilot-cache/runs/<run_id>/`
holds each run's `plan.json`, `events.jsonl` and `run.json`; `pilot-cache/quarantine/`
holds any repaired files. Only planned partitions go in the bundle, because the
engine counts any other Parquet file as unlisted.

```bash
cat > ~/pilot/tools/bundle_cache.py <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
from urllib.parse import quote
cache, plan_path, omega_path, audit_path, ready_dir, out = map(Path, sys.argv[1:7])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan = json.loads(plan_path.read_text(encoding="utf-8"))
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if audit["status"] != "complete" or audit["verified_jobs"] != plan["planned_jobs"]:
    sys.exit("audit is not complete for this plan; do not package")
out.mkdir(parents=True, exist_ok=False)
hashes = {}
for job in plan["jobs"]:
    rel = f"model={quote(str(job['model']), safe='')}/config={quote(str(job['config']), safe='')}/{job['job_id']}.parquet"
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache / rel, out / rel)
    hashes[rel] = sha(out / rel)
for source, name in ((omega_path, "omega.json"), (plan_path, "plan.json"), (audit_path, "audit.json"),
                     (ready_dir / "report.json", "readiness-report.json")):
    shutil.copy2(source, out / name)
first = plan["jobs"][0]
manifest = dict(schema_version=1, fixture=plan["fixture"],
    evidence_kind="fixture" if plan["fixture"] else "real_pilot_cache",
    naming_scheme=first.get("naming_scheme"), source_repository=first.get("source_repository"),
    workload_sha256=plan["workload_sha256"], approval_reference=first.get("approval_reference"),
    model_approval_reference=first.get("model_approval_reference"),
    planned_jobs=plan["planned_jobs"], planned_rows=plan["planned_rows"],
    provenance=plan["provenance"], partition_hashes=hashes,
    omega_sha256=sha(out / "omega.json"), plan_sha256=sha(out / "plan.json"),
    audit_sha256=sha(out / "audit.json"), readiness_report_sha256=sha(out / "readiness-report.json"))
(out / "handoff.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"partitions": len(hashes), "handoff_sha256": sha(out / "handoff.json")}))
PY
cd ~/pilot/decision-aware-tsfm
# first slice:
"$PILOT_PY" -I ~/pilot/tools/bundle_cache.py artifacts/runs/pilot-cache artifacts/runs/pilot-ready/first-plan.json artifacts/runs/pilot-handoff-2026-10-12/HANDOFF-approved/omega.json artifacts/runs/pilot-first-audit.json artifacts/runs/pilot-ready artifacts/runs/pilot-first-bundle
# full pilot, after the full audit:
"$PILOT_PY" -I ~/pilot/tools/bundle_cache.py artifacts/runs/pilot-cache artifacts/runs/pilot-ready/plan.json artifacts/runs/pilot-handoff-2026-10-12/HANDOFF-approved/omega.json artifacts/runs/pilot-full-audit.json artifacts/runs/pilot-ready artifacts/runs/pilot-full-bundle
```

The bundle's `omega.json` is the whole sidecar (all 904 origins); the engine
joins only the origins it has forecasts for. Pack each bundle and give the
archive its own checksum, and pack the evidence separately (every run report,
including failed or interrupted ones):

```bash
cd artifacts/runs
tar -czf pilot-full-bundle.tar.gz pilot-full-bundle && sha256sum pilot-full-bundle.tar.gz > pilot-full-bundle.tar.gz.sha256
tar -czf pilot-evidence.tar.gz pilot-ready pilot-first-measurement pilot-first-audit.json pilot-full-audit.json pilot-cache/runs pilot-qualification-rental
sha256sum pilot-evidence.tar.gz > pilot-evidence.tar.gz.sha256
```

Add to the evidence list whatever else exists: `pilot-full-measurement*`
(Option A), `pilot-full-generation-*.json` (Option B), the split-option plans
and reports, and `pilot-cache/quarantine` if a repair happened.
Copy both archives and their `.sha256` files back with `scp`. Consumer checks
(this one also runs on the Mac with `../envs/py311/bin/python`):

```bash
# engine reader (Veer); prints partitions verified, rows, jobs, series, models, seeds and the Omega join
cd ~/pilot/decision-aware-tsfm
PYTHONPATH=src "$PILOT_PY" -c "import json, sys; from decision_aware_tsfm.analysis.decision.handoff import read_handoff; f, o, r = read_handoff(sys.argv[1], 'current'); print(json.dumps(r, indent=2))" artifacts/runs/pilot-full-bundle
```

For the full pilot expect `partitions_hash_verified` 21,696, `rows` 1,395,776,
`unlisted_parquet_files` 0 and models `autoarima_native, autoets_simulated,
moirai_pilot, seasonal_naive_native`. Abdullah runs `load_k3_inputs` on the
same folder with `json.load(open(bundle/'handoff.json'))['partition_hashes']`,
`bundle/'omega.json'` and the variant. Each consumer records the command, code
version, hashes and result on #29. The selector role table (from #60) is a
separate file and is not part of the bundle.

## 6. If something fails

Only messages the code itself prints or records. Never change the workload,
models, `device`, taus or seeds to get past one of these.

| Message | Where | Meaning and what to do |
| --- | --- | --- |
| `Real workload requires approval_reference` / `Real models requires approval_reference` | planner | The approval line is missing in that file. Redo section 3 on a fresh copy. |
| `Pilot requires exactly the four qualified CPU/RTX 4050 model configurations` | readiness | `models.json` differs from `pilot_models()` (for example a changed `device`). Use Dayson's file plus only the approval line. |
| `Readiness output directory must be new` | readiness | Pick a new `--output-directory`. |
| `Source checksum differs from the planned source` | readiness, generate | A source file changed. Unpack again from the ZIP. |
| `Plan implementation or environment fingerprint differs from this executor` | generate | The plan was made with other code or another environment. Run the readiness check again in this environment, into a new folder. |
| `Real generation requires real=True` | generate | Add `--real`. |
| `Corrupt cache partition requires repair=True: <path>` | generate | Run `audit` to see the list, then rerun with `--repair`. The bad file is kept in `quarantine/`. |
| `Another cache generator holds the coordinator lock` | generate | Another `generate` is using the same cache. Wait for it or stop it; never run two on one cache. |
| `--output must be outside --cache` | cli | Put the report next to the cache, not inside it. |
| `CUDA requested but unavailable; explicitly choose device='cpu'` | Moirai adapter, recorded per job in the run report | CUDA is not working on this machine. Fix the driver or machine; do not switch to CPU (it changes job_ids and readiness blocks it). |
| `statsforecast==2.1.1 is required, found ...` | classical adapters | The environment is not the lock. Redo `uv pip sync`. |
| `Pilot resource measurement requires Linux` | measure_pilot | Use the Linux machine. |
| `Required measurement tool /usr/bin/time is missing` | measure_pilot | `sudo apt-get install -y time`. |
| `GPU measurement requires exactly one physical CUDA device` | measure_pilot | The machine has more than one GPU. Use a single-GPU machine. |
| `CUDA_VISIBLE_DEVICES does not select the measured GPU` | measure_pilot | Unset `CUDA_VISIBLE_DEVICES` or set it to `0`. |
| `CUDA measurement requires CUBLAS_WORKSPACE_CONFIG=:4096:8` | measure_pilot | Export it exactly as in section 4. |
| `nvidia-smi identity probe failed` | measure_pilot | `nvidia-smi` does not work; fix the driver. |
| `Measurement output, cache and input files must be separate` | measure_pilot | Keep the output folder, the cache and the inputs apart. |
| `"status": "incomplete"` with entries under `failures` | generate run report | Each failure has `phase` (`source_preparation`, `model_loading`, `inference`, `execution`, `publication`; shape and crossing-quantile errors are recorded as `execution`), `error` and `transient`. Transient errors (timeouts, connection resets) are retried once; others are not. Rerunning the same command tries every failed job again. A `model_loading` failure for `moirai_pilot` with `HF_HUB_OFFLINE=1` points back to the checkpoint step in section 2. |
| `Moirai forecast starts at the wrong timestamp`, `Invalid Moirai samples`, `Adapter returned crossing quantiles`, `Adapter returned invalid quantile shape or values` | adapters, executor | A model output failed a check. Keep the run report and raise it on #29; do not edit outputs. |

Exit codes: the scripts exit 2 for blocked, incomplete or failed runs and 0 when
complete. A `--limit-jobs` run exits 2 by design.

## How this runbook was checked

On 15 Oct, in Veer's Python 3.11.15 environment without forecasting packages,
and only in a temporary folder:

- `--help` of `forecast_cache.py` (all subcommands), `check_pilot_readiness.py`,
  `measure_pilot.py` and `qualify_pilot_adapters.py`: every flag above exists.
- The ZIP script on the original ZIP: all hashes matched #58.
- Readiness on an approved copy with placeholder text: `ready`, with the counts in 4a.
  The placeholder plans were deleted; they are not evidence.
- Identity checks in memory: approval text does not change job_ids;
  Moirai `device` and the Python patch version do; the recorded environment
  versions hash to `86d87416...`.
- On the invented fixture (constant predictions, no real model): plan,
  `generate --workers 2 --limit-jobs 5`, resume, rerun with zero generated,
  audit, corruption and `--repair`, the lock, the plan split and merge, and the
  bundle script read back by the engine's `read_handoff`.
- Not checked: anything needing CUDA, torch, uni2ts or statsforecast; the
  `uv` install on Linux; Abdullah's loader on the bundle.
