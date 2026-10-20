"""Held-out guard: no M5 or PJM subset anywhere in the frozen manifest.

Run from the repo root:
    python scripts/guard_check.py
    python scripts/guard_check.py path/to/excluded_series.json   # also check the excluded list

The excluded list is a release asset on manifest-v2, not a committed file, so
it is checked only when its path is given (or when it sits next to the manifest).
The check is on subset names, not series IDs, so IDs ending in _dim5 are fine.
"""
import json
import sys
from pathlib import Path

MANIFEST_DIR = Path("artifacts/reference/manifests")
HELD_OUT = ("m5", "pjm")


def subsets(path: Path, key: str) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {row["subset"] for row in json.load(f)[key]}


def main() -> int:
    checked = {"active": subsets(MANIFEST_DIR / "series_manifest.json", "active_dev_series")}

    excluded_path = Path(sys.argv[1]) if len(sys.argv) > 1 else MANIFEST_DIR / "excluded_series.json"
    if excluded_path.exists():
        checked["excluded"] = subsets(excluded_path, "excluded_series")
    else:
        print(f"Note: {excluded_path} not found, excluded list not checked "
              "(download it from the manifest-v2 release and pass its path).")

    violations = sorted(
        s for group in checked.values() for s in group
        if any(h in s.lower() for h in HELD_OUT)
    )
    assert not violations, f"HELD-OUT LEAKAGE: {violations}"
    counts = ", ".join(f"{len(v)} {k} subsets" for k, v in checked.items())
    print(f"Guard passed: zero M5/PJM subsets ({counts}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
