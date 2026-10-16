import json

with open("series_manifest.json") as f:
    d = json.load(f)

all_subsets = set(s["subset"] for s in d["active_dev_series"] + d["excluded_series"])
violations = [s for s in all_subsets if "m5" in s.lower() or "pjm" in s.lower()]
assert len(violations) == 0, f"HELD-OUT LEAKAGE: {violations}"
print("Guard passed: zero M5/PJM rows.")