# Close the approved pilot after compute finishes

Owner: Aadrika; #29/#30, parent #16. **Prepared instructions, not completed evidence.**
Do not execute generation, repair or qualification while the run is excluded.
The [parallel resume runbook](pilot-parallel-resume.md) covers host admission.
Keep `current` primary and `last_4096` sensitivity, as approved on #29. The newer
Omega coverage diagnostic does not amend that choice.

Use the existing partial transfer release `pilot-resume-2026-10-23` (archive SHA
`823c44d67eb155b9cf96e1c2dea270635b4ad29c288d83216b9f56394727437a`).
Preserve that archive and the original `ready/plan.json`; generate into its staged copy.
Never execute the archived laptop-specific `finish_pilot_service.py` on the new host.

## 1. Parameters and completion gate — future, cache-only

Use the exact qualified environment and checked source described in the resume runbook.
Set these to the actual admitted host's paths; every closeout output directory must be new.

```bash
set -euo pipefail
export PILOT_RUN=/absolute/path/to/staged/pilot-run
export PILOT_PY=/absolute/path/to/qualified/venv/bin/python
export PLAN="$PILOT_RUN/plan.relocated.json" CACHE="$PILOT_RUN/cache"
export FINAL_ATTEMPT="$PILOT_RUN/parallel-attempt-1"
export CLOSEOUT=/absolute/path/to/new/closeout-ATTEMPT
export PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir "$CLOSEOUT"
"$PILOT_PY" - <<'PY'
import json, os
from pathlib import Path
r = Path(os.environ['FINAL_ATTEMPT'])
m = json.loads((r/'report.json').read_text()); g = json.loads((r/'generation.json').read_text())
assert m['status'] == m['measurement_status'] == 'complete' and m['generation_exit_code'] == 0
assert g['status'] == 'complete' and g['failed'] == 0 and g['generated'] + g['skipped'] == 21696
PY
"$PILOT_PY" scripts/forecast_cache.py audit --plan-file "$PLAN" --cache "$CACHE" \
  --output "$CLOSEOUT/full-audit.json" > "$CLOSEOUT/full-audit.stdout.log"
"$PILOT_PY" - <<'PY'
import hashlib, json, os
from pathlib import Path
r = Path(os.environ['PILOT_RUN']); c = Path(os.environ['CLOSEOUT'])
a = json.loads((c/'full-audit.json').read_text())
assert a['status'] == 'complete' and a['planned_jobs'] == a['verified_jobs'] == 21696
assert all(not a[k] for k in ('missing','corrupt','unexpected','duplicate','mismatched_actuals'))
t = json.loads((r.parent/'transport-sha256.json').read_text())
saved = {n:h for n,h in t.items() if n.startswith('cache/model=') and n.endswith('.parquet')}
assert len(saved) == 575
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest() == h for n,h in saved.items())
PY
```

Failure of any gate stops closeout. A stopped unit or a partial 575-job audit is not completion.

## 2. Completed rerun — deferred until the gate passes

This command can run models if the cache is incomplete: never bypass section 1.

```bash
"$PILOT_PY" scripts/measure_pilot.py --plan-file "$PLAN" --cache "$CACHE" \
  --output-directory "$CLOSEOUT/zero-rerun" --workers 8 --real
"$PILOT_PY" - <<'PY'
import json, os
from pathlib import Path
r = Path(os.environ['CLOSEOUT'])/'zero-rerun'
g = json.loads((r/'generation.json').read_text()); m = json.loads((r/'report.json').read_text())
assert g['status'] == 'complete' and (g['generated'],g['skipped'],g['failed']) == (0,21696,0)
assert m['status'] == m['measurement_status'] == 'complete'
PY
```

## 3. Disposable missing/corrupt recovery — **deferred real inference**

Only two approved jobs are repaired. Keep their costs separate from the 21,696 primary jobs.
This prepares a fresh two-job copy outside the production cache; original bytes are recorded.

```bash
export RECOVERY="$CLOSEOUT/recovery"
"$PILOT_PY" - <<'PY'
import collections, hashlib, json, os, shutil
from pathlib import Path
from decision_aware_tsfm.cache.executor import _planned_jobs, job_path
p = json.loads(Path(os.environ['PLAN']).read_text()); out = Path(os.environ['RECOVERY'])
out.mkdir(); src = Path(os.environ['CACHE']).resolve(); dst = (out/'cache').resolve()
assert not dst.is_relative_to(src) and not src.is_relative_to(dst)
jobs = [min((j for j in p['jobs'] if j['model']==m and j['horizon']==1 and j['seed']==42),
            key=lambda j:(j['dataset'],j['series_id'],j['origin'],j['job_id']))
        for m in ('seasonal_naive_native','moirai_pilot')]
sub = dict(p,jobs=jobs,planned_jobs=2,planned_rows=2,counts_by_model=dict(collections.Counter(j['model'] for j in jobs)))
_planned_jobs(sub,generate=True)
(out/'plan.json').write_text(json.dumps(sub,indent=2)+'\n')
hashes = {}
for j in jobs:
    a,b = job_path(src,j),job_path(dst,j); b.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(a,b); hashes[j['job_id']] = hashlib.sha256(a.read_bytes()).hexdigest()
(out/'original-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
job_path(dst,jobs[0]).unlink()
job_path(dst,jobs[1]).write_bytes(b'Intentional corruption of disposable recovery copy\n')
PY
expect2() { if "$@"; then return 1; else test "$?" -eq 2; fi; }
expect2 "$PILOT_PY" scripts/forecast_cache.py audit --plan-file "$RECOVERY/plan.json" \
  --cache "$RECOVERY/cache" --output "$RECOVERY/before.json" > "$RECOVERY/before.log" 2>&1
"$PILOT_PY" - <<'PY'
import json, os
from pathlib import Path
a = json.loads((Path(os.environ['RECOVERY'])/'before.json').read_text())
assert a['status']=='incomplete' and len(a['missing'])==2 and len(a['corrupt'])==1
assert not a['unexpected'] and not a['duplicate'] and not a['mismatched_actuals']
PY
expect2 "$PILOT_PY" scripts/forecast_cache.py generate --plan-file "$RECOVERY/plan.json" \
  --cache "$RECOVERY/cache" --workers 1 --real > "$RECOVERY/refused.log" 2>&1
# The next command performs two real forecasts; run only after compute is authorised.
"$PILOT_PY" scripts/forecast_cache.py generate --plan-file "$RECOVERY/plan.json" \
  --cache "$RECOVERY/cache" --workers 1 --repair --real --output "$RECOVERY/repair.json" \
  > "$RECOVERY/repair.log" 2>&1
"$PILOT_PY" scripts/forecast_cache.py audit --plan-file "$RECOVERY/plan.json" \
  --cache "$RECOVERY/cache" --output "$RECOVERY/after.json" > "$RECOVERY/after.log"
"$PILOT_PY" - <<'PY'
import hashlib, json, os
from pathlib import Path
from decision_aware_tsfm.cache.executor import job_path
r = Path(os.environ['RECOVERY']); p = json.loads((r/'plan.json').read_text())
g = json.loads((r/'repair.json').read_text()); a = json.loads((r/'after.json').read_text())
assert g['status']=='complete' and (g['generated'],g['failed'])==(2,0)
assert a['status']=='complete' and a['verified_jobs']==2
assert all(not a[k] for k in ('missing','corrupt','unexpected','duplicate','mismatched_actuals'))
assert any(p.read_bytes()==b'Intentional corruption of disposable recovery copy\n' for p in (r/'cache/quarantine').glob('*.parquet'))
old = json.loads((r/'original-sha256.json').read_text())
assert all(hashlib.sha256(job_path(os.environ['CACHE'],j).read_bytes()).hexdigest()==old[j['job_id']] for j in p['jobs'])
PY
```

Preserve refused-run native events, quarantine bytes, repair wall time and all exit codes.
The repair is unwrapped: it does not establish GNU time RSS or sampled GPU peaks.

## 4. Primary-job accounting — cache-only; retain separate hardware/attempt cohorts

Select native verified measurements by the published Parquet's `(run_id,job_id,attempt)`.
Do not concatenate duplicate historical reports. Group by `(run_id,device,hardware)`;
the CPU `hardware` label is only `platform.platform()`, not physical CPU identity.
Attach actual wrapper/preflight hardware evidence to each run before resource review.
This collector may be used now with `ALLOW_PARTIAL=1` and a new temporary output;
its output remains explicitly partial. Full closeout uses the default full-cache gate.

```bash
export ACCOUNTING_OUT="$CLOSEOUT/accounting"
"$PILOT_PY" - <<'PY'
import collections, hashlib, json, os, re
from pathlib import Path
from decision_aware_tsfm.cache.executor import audit, job_path, read_job
from decision_aware_tsfm.cache.budget import _verified_measurements, measurement_key
p = json.loads(Path(os.environ['PLAN']).read_text()); cache = Path(os.environ['CACHE']).resolve()
out = Path(os.environ['ACCOUNTING_OUT']); out.mkdir(parents=True,exist_ok=False)
a = audit(p,cache); assert all(not a[k] for k in ('corrupt','unexpected','duplicate','mismatched_actuals'))
full = a['status']=='complete' and a['verified_jobs']==21696
assert full or os.environ.get('ALLOW_PARTIAL')=='1', 'Full cache required; partial accounting is preparation only'
selected = {}; native = {}; logs = {}; groups = collections.defaultdict(list)
shared = ('run_id','job_id','attempt','hardware','device','fixture','inference_seconds','model_load_seconds','source_load_seconds','peak_memory_bytes')
for j in p['jobs']:
    path = job_path(cache,j)
    if not path.exists(): continue
    embedded = json.loads(read_job(cache,j).schema.metadata[b'measurement'])
    rid = embedded['run_id']; assert re.fullmatch('[0-9a-f]{32}',rid)
    if rid not in native:
        log = (cache/'runs'/rid/'events.jsonl').resolve(); assert log.is_relative_to(cache)
        raw = log.read_bytes(); prefix = raw[:raw.rfind(b'\n')+1]; records = {}
        for line in prefix.splitlines():
            event = json.loads(line)
            if event['status']!='verified': continue
            m = event['measurement']; key = (m['run_id'],m['job_id'],m['attempt'])
            assert key not in records, 'Duplicate native verified event requires review'
            records[key] = m
        native[rid] = records
        logs[rid] = dict(path=str(log),sha256=hashlib.sha256(raw).hexdigest(),incomplete_tail_bytes=len(raw)-len(prefix))
    key = (rid,j['job_id'],embedded['attempt']); m = native[rid][key]
    assert all(m[k]==embedded[k] for k in shared)
    assert m['timing_key']==measurement_key(j) and m['output_bytes']==path.stat().st_size
    assert j['job_id'] not in selected; selected[j['job_id']] = m
    groups[(rid,m['device'],m['hardware'])].append((j,m))
assert len(selected)==a['verified_jobs'] and (not full or len(selected)==21696)
_verified_measurements(list(selected.values()))
inventory = []
for (rid,device,label),items in sorted(groups.items()):
    name = rid+'-'+device+'-'+hashlib.sha256(label.encode()).hexdigest()[:12]
    d = out/'cohorts'/name; d.mkdir(parents=True); jobs = [j for j,m in items]; ms = [m for j,m in items]
    sub = dict(p,jobs=jobs,planned_jobs=len(jobs),planned_rows=sum(j['horizon'] for j in jobs),counts_by_model=dict(collections.Counter(j['model'] for j in jobs)))
    values = {'plan':sub,'measurements':ms,'hardware':{device:label},'overheads':{'loading_seconds':{device:sum(m['model_load_seconds'] for m in ms)},'io_seconds':{device:sum(m['source_load_seconds']+m['io_seconds'] for m in ms)}}}
    for name,value in values.items(): (d/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    inventory.append(dict(directory=str(d),run_id=rid,device=device,measurement_label=label,jobs=len(jobs),physical_hardware_evidence='REQUIRED: link wrapper/preflight records; do not infer identity from platform label'))
summary = dict(status='complete_primary_job_accounting' if full else 'partial',verified_jobs=len(selected),missing_jobs=len(a['missing']),published_partition_bytes=sum(m['output_bytes'] for m in selected.values()),cohorts=inventory,native_logs=logs,workflow_accounting_status='incomplete: original OOM attempt totals remain unknown')
(out/'inventory.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
PY
for cohort in "$ACCOUNTING_OUT"/cohorts/*; do
  for scenario in base proposed; do
    margin=0; test "$scenario" = base || margin=0.10
    "$PILOT_PY" scripts/forecast_cache.py budget --plan-file "$cohort/plan.json" \
      --measurements "$cohort/measurements.json" --hardware "$cohort/hardware.json" \
      --overheads "$cohort/overheads.json" --restart-fraction "$margin" --failure-fraction "$margin" \
      --output "$cohort/budget-$scenario.json" > "$cohort/budget-$scenario.stdout.log"
  done
done
```

The old published 575 jobs comprise 24 first-slice, 25 recovered-original and 526
user-stopped-service measurements. Preserve those original run IDs. Benchmark,
repair and zero-rerun costs are additional workflow costs, not primary jobs.
List every attempt with actual command, worker count, CPU/GPU identity, status,
wall/RSS/sample values or explicit unknown reasons. The user-stop wrapper wall
is 42,260.6709 s; the earlier OOM wrapper totals remain unknown. Do not add wrapper
wall to summed components, or infer parallel elapsed time from resource sums.

## 5. Complete bundle and owner acceptance — future, cache-only

After all gates, use the archived helper without modifying its bytes:

```bash
"$PILOT_PY" - <<'PY'
import hashlib, json, os
from pathlib import Path
r = Path(os.environ['PILOT_RUN'])
m = json.loads((r.parent/'transport-sha256.json').read_text())
assert hashlib.sha256((r/'package_real_cache.py').read_bytes()).hexdigest()==m['package_real_cache.py']
PY
"$PILOT_PY" "$PILOT_RUN/package_real_cache.py" "$PILOT_RUN" full > "$CLOSEOUT/package.log"
```

It writes a new `full-bundle`, audits all partitions, retains original and approved
inputs, 55 sources, both Omega variants, roles and hashes, and checks both reader variants.
It intentionally packages the unchanged original `ready/plan.json`: audit uses
canonical identity without opening its old absolute source paths. Preserve the
relocated execution plan and verification as additional execution evidence.

Do not substitute PR #65's minimal `bundle_cache.py`: it omits sources/roles and
approval-derivation hashes needed by the current source/scale acceptance checks.
Archive the complete bundle and a separate evidence tree containing every old/new
attempt, raw logs, hardware attestations, relocation proof, checks above and budgets.
Create archive SHA-256s; verify uploaded draft assets' digests before publication.
Never replace the historical first-slice or partial-resume assets.

Use the existing [full consumer acceptance checklist](../results/cache/2026-10-15-real-pilot/consumer-acceptance.md).
Its initial checkout/download block has historical revision pins and a proposed
October 15 full-release name. Replace those with the actual published full-release
assets and owner-agreed reader revisions; do not run that setup block unchanged.
The expected cache counts and source/role/hash checks still apply. Record the
actual revisions; run consumers/variants
sequentially after generation. Both owners must independently attest the same
archive and `handoff.json` hashes, actual commands/environments/exits/reports and
explicit acceptance on #29. Producer checks and the Omega coverage diagnostic
are not those attestations. #29/#30/#16 remain open for their real conditions.
