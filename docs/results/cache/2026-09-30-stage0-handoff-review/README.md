# Reproduce the September 30 handoff review

The two scripts here are the exact audit/probe scripts used for the
[dated report](../2026-09-30-stage0-handoff-review.md). Their hashes are preserved
in its [evidence JSON](../2026-09-30-stage0-handoff-review.json). They are frozen
reproduction material for the listed revisions, not general input validators or
a replacement for the existing pilot-readiness checker.

Use a checkout containing this directory, Git, and Python 3.11 with NumPy 1.26.4,
pandas 2.2.3, PyArrow 16.1.0 and SciPy 1.11.4. No forecasting libraries or weights
are needed. Fetch the repository's branches so the pinned consumer commits are
available locally. Authenticate `gh` to an account with access to this repository
if downloading the metadata.

Run the following from the repository root. It creates a fresh output directory,
checks out the unchanged consumers, obtains the five hash-pinned metadata/code
files from PR #48, regenerates the producer fixture, and checks the recorded
results. It does not read raw source arrays or compute forecasts/Omega. The
original manifest is about 88 MB; it is downloaded metadata, not committed here.

Optionally set `REVIEW_INPUTS` to a directory containing the five original input
files to use local copies instead of downloading them. Their recorded hashes
are required in either case. Each replay retains its own logs and outputs.

Verified on October 1 using local copies of the original hash-pinned metadata:
24 targeted tests passed, both consumer results matched the record, and the
regenerated `handoff.json` retained its recorded SHA-256. The metadata-download
branch uses the same GitHub API endpoints as the original collection.

```bash
python - <<'PY'
import hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

repo = Path.cwd()
material = repo / "docs/results/cache/2026-09-30-stage0-handoff-review"
record = json.loads(material.with_suffix(".json").read_text())
output_parent = repo / "artifacts/runs"
output_parent.mkdir(parents=True, exist_ok=True)
output = Path(tempfile.mkdtemp(prefix="handoff-review-", dir=output_parent))
(output / "inputs").mkdir()
(output / "sources").mkdir()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def run(label, args, pythonpath=None):
    env = dict(os.environ, DECISION_FIXTURE_SOURCE=str(output / "sources/decision"))
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    with (output / (label + ".log")).open("w") as log:
        subprocess.run(list(map(str, args)), env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
for name in ("audit_manifest.py", "probe_consumers.py"):
    assert sha(material / name) == record["artifact_hashes"][name], name
    shutil.copy2(material / name, output / name)
for mode, number in (("decision", "44"), ("k3", "35")):
    commit = record["input_fingerprints"]["heads"][number]
    run("checkout-" + mode, ["git", "worktree", "add", "--detach",
                            output / "sources" / mode, commit])
ref = record["input_fingerprints"]["source_commit"]
for name, expected in record["input_fingerprints"]["files"].items():
    target = output / "inputs" / name
    if os.environ.get("REVIEW_INPUTS"):
        shutil.copy2(Path(os.environ["REVIEW_INPUTS"]) / name, target)
    else:
        with target.open("wb") as stream:
            subprocess.run(["gh", "api", "-H", "Accept: application/vnd.github.raw+json",
                f"repos/Aadrika08/decision-aware-tsfm/contents/{name}?ref={ref}"],
                stdout=stream, check=True)
    assert target.stat().st_size == expected["bytes"], name
    assert sha(target) == expected["sha256"], name
run("audit", [sys.executable, output / "audit_manifest.py"], repo / "src")
run("tests", [sys.executable, "-m", "unittest",
    "tests.integration.test_cache_planner", "tests.integration.test_pilot_readiness",
    "tests.integration.test_consumer_handoff", "-v"], repo / "src")
run("export", [sys.executable, "scripts/export_consumer_fixture.py",
    "--decision-source", output / "sources/decision",
    "--output", output / "consumer-fixture"], repo / "src")
for mode in ("decision", "k3"):
    run(mode, [sys.executable, output / "probe_consumers.py", mode],
        output / "sources" / mode / "src")
assert json.loads((output / "manifest-audit.json").read_text()) == record["manifest_audit"]
for mode in ("decision", "k3"):
    assert json.loads((output / (mode + "-probe.json")).read_text()) == record["consumer_probes"][mode]
assert sha(output / "consumer-fixture/handoff.json") == record["fixture"]["handoff_sha256"]
print("Verified pinned handoff review and fixture:", output)
PY
```

The exported `consumer-fixture/` is the handoff for both consumers: 408 partitions,
9,792 rows, the two planted Omega variants and `handoff.json` with partition hashes.
Use its manifest hashes to compare payloads; repacking a ZIP may change archive
metadata and its outer hash. Forecast caches remain excluded from Git.

The script intentionally reproduces the old consumer rejections and K3 validation
gaps recorded in the report. Matching those outcomes does not mean the defects
are fixed. Updated consumers need a new dated probe and their owners' acceptance.
The fresh native-readiness report has a new timestamp; the original evidence is
never overwritten. This replay neither supplies nor approves a real pilot workload.
