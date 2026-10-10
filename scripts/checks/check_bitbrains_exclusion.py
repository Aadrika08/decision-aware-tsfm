import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "artifacts" / "reference" / "manifests" / "dev_manifest.json"

with MANIFEST.open(encoding="utf-8") as f:
    d = json.load(f)

excluded = d["excluded_series"]
bb_excluded = [s for s in excluded if "bitbrains_fast_storage_H" in s["subset"]]
print(f"bitbrains_fast_storage_H entries in EXCLUDED list: {len(bb_excluded)}")
if bb_excluded:
    print("Sample:", bb_excluded[0])





active = d["active_dev_series"]
bb_5t_active = [s for s in active if s["subset"] == "bitbrains_fast_storage_5T"]
bb_5t_excluded = [s for s in excluded if s["subset"] == "bitbrains_fast_storage_5T"]
print(f"\nbitbrains_fast_storage_5T -> active: {len(bb_5t_active)}, excluded: {len(bb_5t_excluded)}")
if bb_5t_active:
    print("Sample active length:", bb_5t_active[0]["length"])
