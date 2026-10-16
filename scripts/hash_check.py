"""SHA-256 of the frozen manifest files, both as committed in git and as on disk.

Run from the repo root:
    python scripts/hash_check.py            # committed at HEAD
    python scripts/hash_check.py manifest-v2   # committed at a tag

Post the "committed" hashes. They are what anyone gets on a clean checkout,
whatever their operating system. "disk" should match; if it does not, the
working copy differs from the commit (for example, line endings).
"""
import hashlib
import subprocess
import sys
from pathlib import Path

FILES = [
    "artifacts/reference/manifests/series_manifest.json",
    "artifacts/reference/manifests/omega_values.json",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    ok = True
    for path in FILES:
        committed = subprocess.run(
            ["git", "show", f"{ref}:{path}"], capture_output=True, check=False
        )
        if committed.returncode != 0:
            print(f"{path}\n  committed ({ref}): NOT FOUND")
            ok = False
            continue
        c = sha256(committed.stdout)
        disk_path = Path(path)
        d = sha256(disk_path.read_bytes()) if disk_path.exists() else "missing"
        same = "same" if c == d else "DIFFERENT"
        print(f"{path}\n  committed ({ref}): {c}\n  disk:            {d}  [{same}]")
        ok = ok and c == d
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
