"""Assert-based snapshot checks using synthetic files only; never reads the real cache."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("snapshot", Path(__file__).parents[2] / "scripts/pilot_checkpoint.py")
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def rejected(call):
    try:
        call()
    except (ValueError, KeyError):
        return
    raise AssertionError("Invalid backup was accepted")


class PilotCheckpointTest(unittest.TestCase):
    def test_checkpoint_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:

            root = Path(temporary) / "node"; root.mkdir()
            def put(name, value):
                p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(value); return p
            part = put("transfer/pilot-run/cache/model=m/config=H/a.parquet", b"fake-partition-a")
            log = put("transfer/pilot-run/cache/runs/abc/events.jsonl", b'{"a":1}\n{"unfinished":')
            csv = put("transfer/pilot-run/parallel-attempt-1/gpu.csv", b"a,b\n1,2\nunfinished")
            put("transfer/pilot-run/plan.relocated.json", b'{"fake":true}')
            put("host-inventory.json", b'{"cpu":"fake"}')
            put("setup.log", b"synthetic setup\n")
            put("preflight.py", b"# synthetic source\n")
            put("qualification/report.json", b'{"fixture":true}')
            put("benchplans/plan.json", b'{"fixture":true}')
            put("operations/supervisor.py", b"# synthetic source\n")
            final = put("closeout/pilot-full-bundle-l4-20261025.tar.gz", b"fake finalized archive")
            put("closeout/SHA256SUMS-closeout", (hashlib.sha256(final.read_bytes()).hexdigest()+"  "+final.name+"\n").encode())
            put("closeout/status.json", b'{"fixture":true}')
            put("closeout/unfinalized.tar.gz", b"still writing")
            put("closeout/package-inputs/full-bundle/audit.json", b'{"fixture":true}')
            for name in ["transfer/pilot-run/cache/.writer.lock", "transfer/pilot-run/cache/data.partial",
                         "transfer/pilot-run/cache/out.tmp", "transfer/pilot-run/HANDOFF-approved/sources/a.npy",
                         "venv/package.py", "code/private.py", ".ssh/id_ed25519", "operations/relay_key",
                         "operations/.env", "operations/credentials.json", "operations/config.json", "operations/private-key",
                         "closeout/package-inputs/full-bundle/model=m/config=H/a.parquet",
                         "operations/checkpoint.safetensors"]:
                put(name, b"DO NOT COPY")
            outside = Path(temporary) / "outside"; outside.write_bytes(b"OUTSIDE SECRET")
            (root / "operations/link").symlink_to(outside)
            (root / "operations/linked-directory").symlink_to(outside.parent, target_is_directory=True)
            if hasattr(os, "mkfifo"): os.mkfifo(root / "operations/fifo")
            first = s.snapshot(root, 0)
            backup = root / "backups"
            assert s.verify(backup/"snapshot-0.tar.gz", backup/"snapshot-0.receipt.json")["verified"]
            with tarfile.open(backup/"snapshot-0.tar.gz") as tar:
                manifest = json.load(tar.extractfile(s.MANIFEST))
                assert len(manifest["files"]) == 14
                assert first["deferred_archives"] == 1
                assert "closeout/pilot-full-bundle-l4-20261025.tar.gz" in manifest["files"]
                assert "closeout/unfinalized.tar.gz" not in manifest["files"]
                assert not any(b"DO NOT COPY" in tar.extractfile(n).read() or b"OUTSIDE SECRET" in tar.extractfile(n).read() for n in manifest["files"])
                m = manifest["files"]["transfer/pilot-run/cache/runs/abc/events.jsonl"]
                assert m["truncated_bytes"] == len(b'{"unfinished":')
                assert tar.extractfile("transfer/pilot-run/cache/runs/abc/events.jsonl").read() == b'{"a":1}\n'
                assert tar.extractfile("transfer/pilot-run/parallel-attempt-1/gpu.csv").read() == b"a,b\n1,2\n"
            log.write_bytes(b'{"a":1}\n{"a":2}\n')
            csv.write_bytes(b"a,b\n1,2\n3,4\n")
            for name in ('parallel-attempt-1/report.json', 'admission.json', 'pipeline-status.json',
                         'watchdog-events.jsonl', 'provenance-verified.json', 'checkpoint-verified.json'):
                put(name, b'{"fixture":true}\n')
            put("transfer/pilot-run/cache/model=m/config=H/b.parquet", b"fake-partition-b")
            second = s.snapshot(root, 1)
            with tarfile.open(backup/"snapshot-1.tar.gz") as tar:
                names = set(tar.getnames())
                assert 'parallel-attempt-1/report.json' in names and 'watchdog-events.jsonl' in names
            assert first["generation_id"] != second["generation_id"] and second["cache_partition_files"] == 2
            assert second["truncated_files"] == 0
            s.verify(backup/"snapshot-0.tar.gz", backup/"snapshot-0.receipt.json")
            s.verify(backup/"snapshot-1.tar.gz", backup/"snapshot-1.receipt.json")
            verified = s.verify(backup/"snapshot-1.tar.gz", backup/"snapshot-1.receipt.json")
            ack = s.acknowledge(root, verified)
            assert json.loads((root/"backup-ack.json").read_text()) == ack and ack["verified"]
            rejected(lambda: s.acknowledge(root, dict(verified, archive_sha256="0"*64)))
            rejected(lambda: s.acknowledge(root, dict(verified, generation_id="wrong")))
            rejected(lambda: s.acknowledge(root, dict(verified, verified=False)))
            rejected(lambda: s.acknowledge(root, s.verify(backup/"snapshot-0.tar.gz", backup/"snapshot-0.receipt.json")))
            renamed = Path(temporary)/"renamed-generation.tar.gz"; renamed.write_bytes((backup/"snapshot-1.tar.gz").read_bytes())
            s.verify(renamed, backup/"snapshot-1.receipt.json")
            renamed.write_bytes(renamed.read_bytes()+b"corrupt")
            rejected(lambda: s.verify(renamed, backup/"snapshot-1.receipt.json"))
            receipt_before = (backup/"snapshot-0.receipt.json").read_bytes()
            original_atomic = s.atomic_bytes
            def fail_receipt(path, data):
                if path.name.endswith(".receipt.json"):
                    raise RuntimeError("synthetic receipt-publication failure")
                original_atomic(path, data)
            s.atomic_bytes = fail_receipt
            try:
                s.snapshot(root, 0)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Injected failure did not happen")
            s.atomic_bytes = original_atomic
            assert (backup/"snapshot-0.receipt.json").read_bytes() == receipt_before
            rejected(lambda: s.verify(backup/"snapshot-0.tar.gz", backup/"snapshot-0.receipt.json"))
            s.verify(backup/"snapshot-1.tar.gz", backup/"snapshot-1.receipt.json")
            s.snapshot(root, 0)
            assert len(list(backup.glob("snapshot-*.tar.gz"))) == 2
            assert not list(backup.glob("*.partial"))
            for bad, kind in [("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)]:
                malicious = Path(temporary)/"malicious.tar.gz"
                with tarfile.open(malicious,"w:gz") as tar:
                    info=tarfile.TarInfo(bad); info.type=kind; info.size=1 if kind==tarfile.REGTYPE else 0
                    if kind==tarfile.SYMTYPE: info.linkname="/outside"
                    tar.addfile(info,io.BytesIO(b"x") if info.size else None)
                rec = Path(temporary)/"bad.receipt.json"
                rec.write_text(json.dumps({"schema_version":1,"slot":0,"archive_sha256":hashlib.sha256(malicious.read_bytes()).hexdigest(),"archive_bytes":malicious.stat().st_size}))
                rejected(lambda: s.verify(malicious,rec))
            for field in ("hash", "count"):
                malicious = Path(temporary)/"wrong-manifest.tar.gz"
                with tarfile.open(backup/"snapshot-1.tar.gz") as src, tarfile.open(malicious,"w:gz") as dst:
                    for member in src:
                        data = src.extractfile(member).read()
                        if member.name == s.MANIFEST:
                            m = json.loads(data)
                            if field == "hash": next(iter(m["files"].values()))["sha256"] = "0"*64
                            else: m["file_count"] += 1
                            data = json.dumps(m).encode(); member.size=len(data)
                        dst.addfile(member,io.BytesIO(data))
                rec.write_text(json.dumps(dict(second,archive_sha256=hashlib.sha256(malicious.read_bytes()).hexdigest(),archive_bytes=malicious.stat().st_size)))
                rejected(lambda: s.verify(malicious,rec))
            print(json.dumps({"status":"passed","fixture_only":True,"model_imports":False,
                              "checks":["cumulative_updates","complete_log_prefixes","source_exclusions","symlinks",
                                        "slot_retention","renamed_download","sha_mismatch","receipt_atomic_last","unsafe_members",
                                        "member_hashes_and_counts","ack_matches_latest","ack_rejects_mismatch_or_regression"]}))

if __name__ == "__main__":
    unittest.main()
