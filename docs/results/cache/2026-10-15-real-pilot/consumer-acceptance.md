# Future full-pilot consumer acceptance — #29 / #30

**Status: future, unexecuted checklist.** The full bundle did not exist when this
checklist was prepared. Run sequentially **after generation and packaging complete**.
Local ingress probes are not either owner's acceptance and do not close #29.
These commands read saved forecasts; they choose no representatives, thresholds or scores.

Use existing Python 3.11 with NumPy 1.26.4, pandas 2.2.3, PyArrow 16.1.0 and
SciPy 1.11.4. No dependency installation, model loading or inference is required.
Set `PRODUCER` and `K3` to clean checkouts and `PY` to that existing environment:

```bash
set -eu
export PRODUCER=/absolute/path/to/producer-checkout
export K3=/absolute/path/to/abdullah-checkout
export PY=/absolute/path/to/existing/environment/bin/python
export PYTHONDONTWRITEBYTECODE=1
test "$(git -C "$PRODUCER" rev-parse HEAD)" = d2e1b1b40334ee89207be3eab4078b0d291d346a
test -z "$(git -C "$PRODUCER" status --porcelain)"
test "$(git -C "$K3" rev-parse HEAD)" = e3b7e17e54e94439fc78a38057273473e6bc7985
test -z "$(git -C "$K3" status --porcelain)"
mkdir pilot-full-acceptance
cd pilot-full-acceptance
gh release download pilot-full-2026-10-15 --repo Aadrika08/decision-aware-tsfm \
  --pattern 'pilot-full-*.tar.gz' --pattern SHA256SUMS-full
sha256sum --check SHA256SUMS-full
tar -xzf pilot-full-bundle-2026-10-15.tar.gz
tar -xzf pilot-full-evidence-2026-10-15.tar.gz
export BUNDLE="$PWD/full-bundle"
sha256sum "$BUNDLE/handoff.json"
"$PY" -c 'import platform, importlib.metadata as m; print(platform.python_version()); print({p: m.version(p) for p in ("numpy", "pandas", "pyarrow", "scipy")})'
```

The K3 pin matched the live PR #35 head when checked on October 16 (IST).
Confirm the owner-agreed revision again before acceptance; record any changed pin and rerun.
The historical `2496744` runbook pin lacks later provenance/hash checks.

Retain `full-evidence` package/audit reports and `full-bundle/handoff.json`,
`audit.json`, `readiness-report.json`, `preflight.json`, `consumer-ingress-checks.json`
and `role_table/`. The existing packager verifies source copies, approval-only
derivations and role-table discovery; its reports are local producer evidence.
Recheck the saved plan without generating or writing an output file:

```bash
PYTHONPATH="$PRODUCER/src" "$PY" "$PRODUCER/scripts/forecast_cache.py" audit \
  --plan-file "$BUNDLE/plan.json" --cache "$BUNDLE"
```

Expected: `status=complete`, `planned_jobs=verified_jobs=21696`, and empty missing,
corrupt, unexpected, duplicate and mismatched-actual collections. This checks job
identities, context/target bounds and exact timestamps against the frozen plan.

Veer: `read_handoff(root, variant)` returns forecasts, Omega rows and a report;
`load_cache(forecasts, schema_version="producer-v1")` validates canonical ingress.
The existing source helper verifies all source hashes and realised target positions.

```bash
PYTHONPATH="$PRODUCER/src" "$PY" - <<'PY'
import gc, json, os
from pathlib import Path
from decision_aware_tsfm.cache.schema import QCOLS
from decision_aware_tsfm.analysis.decision.handoff import file_sha256, read_handoff
from decision_aware_tsfm.analysis.decision.cache_io import load_cache
from decision_aware_tsfm.analysis.decision.scale import read_workload_sources, check_bundle_against_sources
root = Path(os.environ["BUNDLE"])
m = json.loads((root / "handoff.json").read_text(encoding="utf-8"))
for name, field in (("plan.json", "plan_sha256"), ("audit.json", "audit_sha256"),
                    ("readiness-report.json", "readiness_report_sha256"),
                    ("workload.json", "original_workload_file_sha256"),
                    ("execution-inputs/workload.json", "approved_workload_file_sha256"),
                    ("execution-inputs/models.json", "approved_models_file_sha256"),
                    ("role_table/role_table.csv", "roles_sha256")):
    assert file_sha256(root / name) == m[field], name
sources = read_workload_sources(root / "workload.json")
assert sources.report["sources_hash_verified"] == 55
for variant in ("current", "last_4096"):
    forecasts, omega, report = read_handoff(root, variant)
    canonical = load_cache(forecasts, schema_version="producer-v1")
    assert len(QCOLS) == 11 and all(canonical[c].equals(forecasts[c]) for c in QCOLS)
    for key, expected in {"fixture": False, "partitions_listed": 21696,
        "partitions_hash_verified": 21696, "unlisted_parquet_files": 0,
        "rows": 1395776, "jobs": 21696, "series": 55, "series_origins": 904,
        "omega_rows_for_variant": 904, "omega_undefined_rows_for_variant": 0,
        "seeds": [42, 43]}.items():
        assert report[key] == expected, key
    assert report["models"] == ["autoarima_native", "autoets_simulated", "moirai_pilot", "seasonal_naive_native"]
    assert sorted(canonical.horizon.unique().tolist()) == [1, 24, 168]
    source_check = check_bundle_against_sources(forecasts, omega, sources)
    assert source_check == {"series": 55, "target_rows_checked": 1395776, "source_hashes_match_bundle": 55}
    print(json.dumps({"consumer": "Veer", "report": report, "source_check": source_check}, sort_keys=True))
    del forecasts, canonical, omega
    gc.collect()
PY
```

Abdullah: use his checkout's ordinary import. `load_k3_inputs` returns the table,
Omega by stable series/origin key, and its report. Keep fixture opt-in disabled.

```bash
PYTHONPATH="$K3/src" "$PY" - <<'PY'
import gc, hashlib, json, os
from pathlib import Path
from decision_aware_tsfm.cache.schema import TAUS
from decision_aware_tsfm.analysis.k3.loader import load_k3_inputs
root = Path(os.environ["BUNDLE"])
m = json.loads((root / "handoff.json").read_text(encoding="utf-8"))
for variant in ("current", "last_4096"):
    table, omega, report = load_k3_inputs(root, m["partition_hashes"], root / "omega.json", variant,
        allow_fixture=False, naming=m["naming_scheme"], omega_sha256=m["omega_sha256"])
    for key, expected in {"status": "ok", "provenance_required": True,
        "omega_sha256_verified": True, "partitions": 21696, "rows": 1395776,
        "provenance_checked_jobs": 21696, "comparisons": 5424, "series": 55,
        "series_origins": 904, "horizons": [1, 24, 168], "seeds": [42, 43],
        "unused_omega_sources": 0, "undefined_omega": 0, "taus": list(TAUS)}.items():
        assert report[key] == expected, key
    assert len(omega) == 904
    print(json.dumps({"consumer": "Abdullah", "handoff_sha256": hashlib.sha256(
        (root / "handoff.json").read_bytes()).hexdigest(), "report": report}, sort_keys=True))
    del table, omega
    gc.collect()
PY
```

Each owner records on #29: name/date; release tag, both archive SHA-256s and
`handoff.json` SHA-256; clean consumer commit; Python/package versions; exact
commands, exit codes and complete reports for **both** variants; any warnings,
limitations or rejection reason; and an explicit acceptance/rejection statement.
Reports must retain model/family, seed, horizon and all eleven tau identities.
Loader Omega-band diagnostics do not authorise scientific threshold choices.
Local passes cannot substitute for either owner's attestation or close #29/#30.

Full-bundle reader time/memory is unmeasured; run the consumers and variants
sequentially, after generation, and retain failures rather than changing inputs.
