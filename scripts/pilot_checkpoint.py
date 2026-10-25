"""Cumulative off-host-backup snapshots, without model imports or credentials.

Create: python snapshot.py create ROOT 0|1
Verify downloaded files (renaming is fine): python snapshot.py verify ARCHIVE RECEIPT
Accept a cloud-verified receipt on stdin: python snapshot.py ack ROOT
Commands print compact JSON. Fetch the receipt first; a concurrent
slot replacement must fail its SHA check, never be accepted as that generation.
"""
import argparse
from datetime import datetime, timezone
import errno
import hashlib
import io
import json
import ntpath
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import tempfile
import uuid


BASE_SHA256 = "823c44d67eb155b9cf96e1c2dea270635b4ad29c288d83216b9f56394727437a"
BASE_RELEASE = "pilot-resume-2026-10-23"
MANIFEST = "snapshot-manifest.json"
BASE_DIRS = {"HANDOFF", "HANDOFF-approved", "ready", "roles", "sources",
             "first-bundle", "full-bundle", "full-evidence", "first-measurement",
             "full-interruption", "full-resume", "full-service-resume"}
PRIVATE_DIRS = {".git", ".ssh", ".aws", ".config", ".cache", ".local", ".venv",
                "venv", "uv", "python", "hf", "huggingface", "checkpoints",
                "backups", "auth", "secrets", "credentials", "keys", "config", "configs",
                "__pycache__", "tmp", "temp"}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def private(name):
    lower = name.lower()
    return (name.startswith(".") or lower.startswith("id_")
            or lower in {"authorized_keys", "known_hosts", "config.json", "config.yaml", "config.yml", "config.toml", "keys.json"}
            or lower.endswith((".key", ".pem", ".pub", ".p12", ".pfx"))
            or re.search(r"(?:^|[-_.])(?:secret|token|password|credential|credentials|(?:ssh|relay|backup|private|api|access)[-_]?key)(?:[-_.]|$)", lower))


def temporary(name):
    return name == ".writer.lock" or name.endswith((".partial", ".tmp", ".temp", ".swp", "~"))


def sources(root):
    pilot = root / "transfer" / "pilot-run"
    starts = []
    if pilot.is_dir() and not pilot.is_symlink() and not pilot.parent.is_symlink():
        starts.extend(p for p in pilot.iterdir() if p.name not in BASE_DIRS)
    for p in root.iterdir():
        name = p.name.lower().replace("_", "-")
        if p.name == "operations" or name.startswith(("qualification", "bench", "host-inventory", "setup", "preflight", "audit", "closeout", "parallel-attempt", "admission", "pipeline", "watchdog", "provenance-verified", "checkpoint-verified", "backup-ack")):
            starts.append(p)
    found = set()
    for start in starts:
        if start.is_symlink() or private(start.name) or temporary(start.name):
            continue
        if start.is_dir():
            candidates = []
            for directory, dirs, files in os.walk(start, followlinks=False):
                packaged = "package-inputs" in Path(directory).relative_to(root).parts
                dirs[:] = sorted(d for d in dirs if d.lower() not in PRIVATE_DIRS
                                 and not private(d) and not (Path(directory) / d).is_symlink()
                                 and not (packaged and (d in {"HANDOFF", "HANDOFF-approved", "sources", "cache", "ready", "roles"} or d.startswith("model="))))
                candidates.extend(Path(directory) / name for name in sorted(files))
        else:
            candidates = [start]
        for path in candidates:
            if path.is_symlink() or private(path.name) or temporary(path.name):
                continue
            rel = path.relative_to(root).as_posix()
            if "package-inputs" in path.relative_to(root).parts:
                diagnostic = path.suffix.lower() in {".log", ".txt"} or (path.suffix.lower() == ".json" and (path.name == "handoff.json" or any(s in path.name.lower() for s in ("audit", "report", "check", "status"))))
                if not diagnostic:
                    continue
            in_cache = rel.startswith("transfer/pilot-run/cache/")
            if any(p.startswith("closeout") for p in path.relative_to(root).parts) and path.name.lower().endswith((".tar.gz", ".tgz")):
                found.add(path)
                continue
            if not in_cache and path.name.lower().endswith((".npy", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".pyc", ".tar", ".tar.gz", ".tgz", ".zip")):
                continue
            found.add(path)
    # Capture mutable ledgers after partitions; this is still file-by-file, not a frozen filesystem.
    return sorted(found, key=lambda p: (p.suffix.lower() in {".jsonl", ".csv"}, p.as_posix()))


def finalized_hash(path):
    """A closeout archive is final only when its captured bytes match a published checksum."""
    hashes = set()
    for sidecar in (Path(str(path) + ".sha256"), path.parent / "SHA256SUMS-closeout", path.parent / "SHA256SUMS-full", path.parent / "SHA256SUMS"):
        if not sidecar.is_file() or sidecar.is_symlink():
            continue
        for line in sidecar.read_text().splitlines():
            parts = line.split(maxsplit=1)
            if parts and re.fullmatch("[0-9a-f]{64}", parts[0]) and ((len(parts) == 1 and sidecar == Path(str(path) + ".sha256")) or (len(parts) == 2 and Path(parts[1].lstrip(" *")).name == path.name)):
                hashes.add(parts[0])
    return next(iter(hashes)) if len(hashes) == 1 else None


def atomic_bytes(path, data):
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def snapshot(root, slot):
    import fcntl  # The producer is Linux; local verification does not require fcntl.
    root = Path(root).resolve(strict=True)
    if type(slot) is not int or slot not in (0, 1):
        raise ValueError("slot must be 0 or 1")
    backup = root / "backups"
    if backup.is_symlink():
        raise ValueError("backup directory must not be a symlink")
    backup.mkdir(exist_ok=True)
    with (backup / ".snapshot.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        now = datetime.now(timezone.utc)
        generation = now.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
        manifest = {"schema_version": 1, "generation_id": generation,
                    "started_at": now.isoformat(), "base_release": BASE_RELEASE,
                    "base_archive_sha256": BASE_SHA256, "source_root": str(root),
                    "consistency": "file-by-file captured bytes; restore then audit, not completion proof",
                    "files": {}, "vanished_files": [], "deferred_archives": []}
        fd, temp = tempfile.mkstemp(prefix=f".snapshot-{slot}.", suffix=".partial", dir=backup)
        archive = backup / f"snapshot-{slot}.tar.gz"
        try:
            with os.fdopen(fd, "wb") as raw_archive:
                with tarfile.open(fileobj=raw_archive, mode="w:gz", compresslevel=3) as tar:
                    for path in sources(root):
                        rel = path.relative_to(root).as_posix()
                        try:
                            if not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
                                continue
                            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
                            with os.fdopen(descriptor, "rb") as stream:
                                info = os.fstat(stream.fileno())
                                if not stat.S_ISREG(info.st_mode):
                                    continue
                                data = stream.read()
                        except OSError as error:
                            if error.errno not in (errno.ENOENT, errno.ELOOP):
                                raise
                            manifest["vanished_files"].append(rel)
                            continue
                        source_bytes = len(data)
                        if not rel.startswith("transfer/pilot-run/cache/") and path.name.lower().endswith((".tar.gz", ".tgz")):
                            if hashlib.sha256(data).hexdigest() != finalized_hash(path):
                                manifest["deferred_archives"].append(rel)
                                continue
                        if path.suffix.lower() in {".jsonl", ".csv"}:
                            data = data[:data.rfind(b"\n") + 1]
                        manifest["files"][rel] = {"sha256": hashlib.sha256(data).hexdigest(),
                            "bytes": len(data), "source_read_bytes": source_bytes,
                            "truncated_bytes": source_bytes - len(data)}
                        member = tarfile.TarInfo(rel)
                        member.size, member.mode, member.mtime = len(data), info.st_mode & 0o777, int(info.st_mtime)
                        tar.addfile(member, io.BytesIO(data))
                    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
                    manifest["file_count"] = len(manifest["files"])
                    manifest["file_bytes"] = sum(v["bytes"] for v in manifest["files"].values())
                    encoded = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
                    member = tarfile.TarInfo(MANIFEST)
                    member.size, member.mode = len(encoded), 0o600
                    tar.addfile(member, io.BytesIO(encoded))
                raw_archive.flush()
                os.fsync(raw_archive.fileno())
            digest = sha_file(temp)
            receipt = {"schema_version": 1, "slot": slot, "generation_id": generation,
                "created_at": manifest["finished_at"],
                "archive_name": archive.name, "archive_sha256": digest, "archive_bytes": os.path.getsize(temp),
                "base_archive_sha256": BASE_SHA256, "file_count": manifest["file_count"],
                "file_bytes": manifest["file_bytes"],
                "cache_partition_files": sum(p.startswith("transfer/pilot-run/cache/model=") and p.endswith(".parquet") for p in manifest["files"]),
                "truncated_files": sum(v["truncated_bytes"] > 0 for v in manifest["files"].values())}
            receipt["deferred_archives"] = len(manifest["deferred_archives"])
            os.replace(temp, archive)
            atomic_bytes(backup / (archive.name + ".sha256"), (digest + "  " + archive.name + "\n").encode())
            # Publish the receipt last: consumers trust only a downloaded archive matching this SHA/generation.
            atomic_bytes(backup / f"snapshot-{slot}.receipt.json", (json.dumps(receipt, sort_keys=True) + "\n").encode())
            return receipt
        finally:
            if os.path.exists(temp):
                os.unlink(temp)


def verify(archive, receipt):
    archive = Path(archive)
    receipt = json.loads(Path(receipt).read_text())
    expected = receipt.get("archive_sha256", "")
    if not re.fullmatch("[0-9a-f]{64}", expected) or sha_file(archive) != expected:
        raise ValueError("Archive SHA-256 differs from receipt")
    if archive.stat().st_size != receipt["archive_bytes"]:
        raise ValueError("Archive byte count differs from receipt")
    if receipt.get("schema_version") != 1 or type(receipt.get("slot")) is not int or receipt["slot"] not in (0, 1):
        raise ValueError("Unsupported receipt schema or slot")
    observed, manifest = {}, None
    with tarfile.open(archive, "r:gz") as tar:
        seen = set()
        for member in tar:
            name = member.name
            path = PurePosixPath(name)
            if (not path.parts or not member.isfile() or path.is_absolute() or ntpath.isabs(name)
                    or ".." in path.parts or "\\" in name or "\x00" in name
                    or path.as_posix() != name or ":" in path.parts[0] or name in seen):
                raise ValueError("Unsafe or duplicate archive member: " + name)
            seen.add(name)
            stream = tar.extractfile(member)
            if name == MANIFEST:
                manifest = json.load(stream)
                continue
            h, size, last = hashlib.sha256(), 0, b""
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk); size += len(chunk); last = chunk[-1:]
            if path.suffix.lower() in {".jsonl", ".csv"} and size and last != b"\n":
                raise ValueError("Incomplete line in archived log: " + name)
            observed[name] = {"sha256": h.hexdigest(), "bytes": size}
    if not manifest or manifest.get("schema_version") != 1 or manifest.get("base_archive_sha256") != BASE_SHA256 or receipt.get("base_archive_sha256") != BASE_SHA256:
        raise ValueError("Missing manifest or wrong base archive")
    if manifest["generation_id"] != receipt["generation_id"] or manifest["finished_at"] != receipt["created_at"] or set(observed) != set(manifest["files"]):
        raise ValueError("Generation or member inventory differs")
    for name, actual in observed.items():
        wanted = manifest["files"][name]
        if actual != {k: wanted[k] for k in ("sha256", "bytes")} or wanted["source_read_bytes"] - wanted["bytes"] != wanted["truncated_bytes"] or wanted["truncated_bytes"] < 0:
            raise ValueError("Member hash/bytes differ: " + name)
    counts = {"file_count": len(observed), "file_bytes": sum(v["bytes"] for v in observed.values())}
    if any(manifest[k] != v or receipt[k] != v for k, v in counts.items()):
        raise ValueError("Manifest/receipt counts differ")
    if receipt["cache_partition_files"] != sum(p.startswith("transfer/pilot-run/cache/model=") and p.endswith(".parquet") for p in observed):
        raise ValueError("Partition count differs")
    if receipt["truncated_files"] != sum(v["truncated_bytes"] > 0 for v in manifest["files"].values()):
        raise ValueError("Truncation count differs")
    if receipt["deferred_archives"] != len(manifest["deferred_archives"]):
        raise ValueError("Deferred archive count differs")
    return dict(receipt, verified=True)


def acknowledge(root, received):
    """An SSH relay supplies only the receipt it verified after cloud read-back."""
    import fcntl
    root = Path(root).resolve(strict=True)
    slot = received.get("slot")
    if received.get("verified") is not True or type(slot) is not int or slot not in (0, 1):
        raise ValueError("A verified receipt with slot 0 or 1 is required")
    backup = root / "backups"
    if backup.is_symlink():
        raise ValueError("backup directory must not be a symlink")
    with (backup / ".snapshot.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        local = json.loads((backup / f"snapshot-{slot}.receipt.json").read_text())
        if {k: v for k, v in received.items() if k != "verified"} != local:
            raise ValueError("Cloud receipt does not match the latest local slot generation/SHA")
        target = root / "backup-ack.json"
        if target.exists():
            old = json.loads(target.read_text())
            if datetime.fromisoformat(local["created_at"]) < datetime.fromisoformat(old["created_at"]):
                raise ValueError("Acknowledgement would regress to an older snapshot")
        acknowledged = dict(local, verified=True, acknowledged_at=datetime.now(timezone.utc).isoformat())
        atomic_bytes(target, (json.dumps(acknowledged, sort_keys=True) + "\n").encode())
        return acknowledged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create"); create.add_argument("root", type=Path); create.add_argument("slot", type=int, choices=(0, 1))
    check = commands.add_parser("verify"); check.add_argument("archive", type=Path); check.add_argument("receipt", type=Path)
    ack = commands.add_parser("ack"); ack.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "create":
        result = snapshot(args.root, args.slot)
    elif args.command == "verify":
        result = verify(args.archive, args.receipt)
    else:
        result = acknowledge(args.root, json.load(sys.stdin))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
