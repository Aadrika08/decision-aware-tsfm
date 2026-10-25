# Resume the approved pilot on a stronger CPU host

Owner: Aadrika. Execution preparation: [#71](https://github.com/Aadrika08/decision-aware-tsfm/issues/71).
Pilot and final budget remain [#29](https://github.com/Aadrika08/decision-aware-tsfm/issues/29)
and [#30](https://github.com/Aadrika08/decision-aware-tsfm/issues/30).
This supplements [development-pilot.md](development-pilot.md) and the detailed
fallback setup in [PR #65](https://github.com/Aadrika08/decision-aware-tsfm/pull/65).
The verified partial transfer is now available in the
[pilot-resume release](https://github.com/Aadrika08/decision-aware-tsfm/releases/tag/pilot-resume-2026-10-23).
See the [25 October handoff record](../results/cache/2026-10-25-compute-handoff.md)
for its download verification and the remaining evidence requirements.

The original laptop run is stopped. The October 23 audit verifies **575 jobs**;
21,121 remain missing, with zero corrupt/unexpected/duplicate partitions or
actual-value mismatches. A new attempt must skip these saved jobs, not replace them.

## Choose the CPU capacity first

Use Linux x86-64, **16 vCPUs / about eight physical cores, 64 GiB RAM, at least
60 GiB free persistent disk, and exactly one NVIDIA GPU with at least 6 GiB VRAM**.
Start with eight CPU workers, each with one BLAS thread. A larger GPU alone does
not address AutoARIMA fitting time. Examples with the required shape are
[AWS g4dn.4xlarge](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
(16 vCPUs, eight cores, 64 GiB, one T4) and
[GCP g2-standard-16](https://docs.cloud.google.com/compute/docs/gpus)
(16 vCPUs, 64 GB, one L4). Specifications were checked October 23; availability,
price and actual throughput must be checked before renting. No machine has been provisioned.

Saved AutoARIMA inference has n=557, mean 80.5998 s, median 69.6704 s and
largest observed 334.1606 s. Its 4,867 remaining jobs imply **108.97 serial
CPU-hours**, or **13.62 hours with ideal eight-worker scaling**, before other
models and checks. This does not establish six-hour completion on the laptop.
Even deterministic-seed reuse alone would leave about 50.4 serial CPU-hours.
Sharing fits across horizons could reduce that further, but is not implemented
or qualified here. No seeds are copied or dropped.

## Install the exact environment

First install `uv`, GNU time (`/usr/bin/time`) and a compatible NVIDIA driver
using [PR #65, section 2](https://github.com/Aadrika08/decision-aware-tsfm/pull/65).
The commands below assume those prerequisites succeeded.

Use the source commit in the transfer manifest / reviewed #71 PR. The five
fingerprinted forecast files must match the original plan; the measurement
wrapper can differ and records its own hash. Do not install latest versions.

```bash
git clone https://github.com/Aadrika08/decision-aware-tsfm.git
cd decision-aware-tsfm
git checkout --detach 1937c7ba11e045449ef5469685b405ab51da58f3
uv python install 3.11.15
uv venv --python 3.11.15 .venv
uv pip sync --python .venv/bin/python --require-hashes requirements/qualification.txt
uv pip check --python .venv/bin/python
export PILOT_PY="$PWD/.venv/bin/python"
export PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi
test -x /usr/bin/time
```

Download the exact checkpoint before offline execution, using PR #65's commands:
repository `Salesforce/moirai-1.1-R-small`, revision
`0c24ab99db2c1a70ea2a0fc03bf113329772ac64`.
`model.safetensors` must be 55,320,200 bytes with SHA-256
`cc46831272ea07e99d78fe05c09e730c7c76a1b272cddbdaf5173dae371cb238`;
`config.json` SHA-256 is
`37b9eca5a0c3e58c2cdac3ecdbf897acbd009a20d6ba0f8a2ae0e8d92002fd47`.
Then set `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`. Require successful CUDA and
checkpoint qualification **before launching the full mixed-model service, including
its CPU phase**, so a later GPU failure cannot waste the CPU run. For example:

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$PILOT_PY" -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
"$PILOT_PY" scripts/qualify_pilot_adapters.py artifacts/runs/new-host-qualification --device cuda
# Require status=passed, attempted=72, passed=72, failed=0.
```

Record the hardware, checkpoint hashes and qualification result. No M5/PJM data
is needed. Qualification failures are retained; they do not pass admission.

## Verify the transport and relocate paths only

The transfer is labelled **partial paused pilot**, not a completed consumer
handoff. It preserves the entire stopped run, original inputs and attempts.
Download `pilot-resume-2026-10-23.tar.gz` and its `.sha256` file from the release.
The archive must be 48,498,484 bytes with SHA-256
`823c44d67eb155b9cf96e1c2dea270635b4ad29c288d83216b9f56394727437a`.
Check the archive SHA-256 before extracting into a new directory. Set `PILOT_RUN`
to the extracted `pilot-run` directory. Keep that directory fixed after relocation.
The archive's `transport-sha256.json` maps relative file paths to exact hashes.
Verify every file before using it:

```bash
export PILOT_RUN=/absolute/path/to/extracted/pilot-run
"$PILOT_PY" - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['PILOT_RUN']).resolve()
manifest = json.loads((root.parent / 'transport-sha256.json').read_text())
for name, expected in manifest.items():
    path = (root / name).resolve()
    assert path.is_relative_to(root) and path.is_file(), name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
print('transport hashes verified:', len(manifest))
PY
```

The saved full plan is `ready/plan.json`; keep its bytes unchanged. Create a
separate relocated copy. Paths are excluded from canonical job identity, but
source bytes, windows, seeds, settings and quantiles are included:

```bash
"$PILOT_PY" - <<'PY'
import copy, hashlib, json, os
from pathlib import Path
from decision_aware_tsfm.cache.executor import _planned_jobs
from decision_aware_tsfm.cache.planner import canonical, job_identity
root = Path(os.environ['PILOT_RUN']).resolve()
original = json.loads((root / 'ready/plan.json').read_text())
relocated = copy.deepcopy(original)
sources = root / 'HANDOFF-approved/sources'
checked = {}
for before, job in zip(original['jobs'], relocated['jobs']):
    path = (sources / Path(job['source_path']).name).resolve()
    assert path.is_relative_to(sources.resolve())
    if path not in checked:
        checked[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert checked[path] == job['source_sha256']
    job['source_path'] = str(path)
    assert job_identity(job) == job_identity(before)
    assert hashlib.sha256(canonical(job_identity(job)).encode()).hexdigest() == job['job_id']
assert len(checked) == 55 and len(relocated['jobs']) == 21696
_planned_jobs(relocated, generate=True)
with (root / 'plan.relocated.json').open('x') as stream:
    json.dump(relocated, stream, indent=2, allow_nan=False)
print('55 sources verified; all 21,696 identities preserved')
PY
"$PILOT_PY" scripts/forecast_cache.py audit \
  --plan-file "$PILOT_RUN/plan.relocated.json" --cache "$PILOT_RUN/cache" \
  --output "$PILOT_RUN/audit-on-new-host.json" > "$PILOT_RUN/audit-on-new-host.stdout.log"
# Expected exit 2: incomplete, 575 verified, 21,121 missing, all anomaly counts zero.
```

The code/environment check must find implementation
`5aa32a3436d5a9cc3ac93d79f1f9892dc6db1969d932bfa33ade0aac07967e36`
and environment
`86d8741605a3e5f29262eb6bdbadae072285a5935f0255c685c659a9d546f8fe`.
Stop on any mismatch. Do not change pins or rebuild job IDs to bypass it.

## Admit the host before the full run

Select one predeclared series-origin per dataset, all three horizons, seed 42:
15 AutoARIMA jobs / 965 rows. Build this **subset of existing jobs**, updating
`jobs`, `planned_jobs`, `planned_rows` and `counts_by_model` (integer counts).
Record the selected IDs before timing. Benchmark into a new empty disposable cache;
using the production cache with a subset plan would label other saved jobs unexpected.
Do not select contexts by their known speed.

Run the measured wrapper with `--workers 8 --real` under a bounded transient
service. For an initial check use `RuntimeMaxSec=1800`, `TimeoutStopSec=15`,
`KillMode=control-group`, `Restart=no`, `MemoryHigh=24G`, `MemoryMax=32G`,
`MemorySwapMax=4G` on a 64 GiB host. Leave time for cleanup. A timeout, any failure,
missing benchmark job or resource-pressure event is not a passing throughput result.
Record cold-start cost. Broaden the declared benchmark if results are unstable.

For the proposed 48-hour execution scenario after setup/admission, reserve
three hours for the other models inside the generator and three hours for
shutdown, audits, recovery and packaging afterward. Apply the **proposed** 10%
restart + 10% failure allowance to remaining AutoARIMA work:

`required rate = 4867 * 1.20 / (48 - 6) = 139.057 jobs/hour`.

Require at least **139.1 completed AutoARIMA jobs/hour** on the declared benchmark,
all five datasets/all horizons represented, no failed jobs and acceptable memory
pressure. Measure Moirai/AutoETS on the new host as well and confirm their
three-hour slot is sufficient; otherwise recompute the rate threshold and timers.
The complete generator gets 45 hours (42 AutoARIMA + three other models), leaving
three hours including shutdown for the final checks. Setup, qualification, the
admission benchmark and human review have separate elapsed-time/cost records. This is an admission model, not a statistical bound or
guarantee. Stop or revise the host choice if it fails; keep the failed evidence.

GNU time's maximum RSS is a maximum process figure, not summed pool memory.
Also capture `systemctl --user show UNIT -p MemoryPeak -p MemoryCurrent -p ControlGroup`
and the cgroup's `memory.peak` / `memory.events` during execution. Device-wide
GPU samples and per-job Torch allocations are different measurements.

## Resume only after admission and a recorded scheduling amendment

The default remains one CPU worker. Record the selected host, eight workers,
resource caps and benchmark on #29/#71 before this distinct parallel attempt.
For an admitted 64 GiB host, use a fresh named service and fresh output directory:

```bash
systemd-run --user --unit=decision-aware-pilot-71 --service-type=exec \
  --working-directory="$PWD" --property=Restart=no --property=KillMode=control-group \
  --property=MemoryHigh=24G --property=MemoryMax=32G --property=MemorySwapMax=4G \
  --property=RuntimeMaxSec=45h --property=TimeoutStopSec=15 \
  --property=RemainAfterExit=yes \
  --property="StandardOutput=append:$PILOT_RUN/parallel-service.log" \
  --property="StandardError=append:$PILOT_RUN/parallel-service.log" \
  --setenv="PYTHONPATH=$PWD/src" --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --setenv=HF_HUB_OFFLINE=1 --setenv=HF_DATASETS_OFFLINE=1 \
  "$PILOT_PY" "$PWD/scripts/measure_pilot.py" \
  --plan-file "$PILOT_RUN/plan.relocated.json" --cache "$PILOT_RUN/cache" \
  --output-directory "$PILOT_RUN/parallel-attempt-1" --workers 8 --real
```

The timer stops work; it does not make incomplete work complete. Do not run the
old laptop-specific launchers or monitor on the new host: they have absolute
paths, old service names and old accounting assumptions. Observe this service
directly, post job counts, and preserve all native events and raw measurements.
No implicit reboot restart is configured; keep the host/session available.

After generation, follow the [portable closeout commands](pilot-closeout.md).
Require the full 21,696-job audit, zero-generation completed
rerun and disposable missing/corrupt recovery. Budget each measured hardware/attempt
separately, retaining old unknown OOM costs and avoiding duplicate successful-job
accounting. Publish the hashed complete bundle, then obtain both actual consumers'
acceptance of the same bytes. #29/#30 and the full-study gates are still open.
