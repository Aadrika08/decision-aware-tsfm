import json
def summarize(path, label):
    with open(path) as f:
        d = json.load(f)
    active = d["active_dev_series"]
    print(f"\n=== {label} ({path}) ===")
    for subset in ["ett1_H", "jena_weather", "bitbrains_fast_storage_H"]:
        matches = [s for s in active if s["subset"] == subset]
        if matches:
            mean_omega = sum(s["omega_levels"] for s in matches) / len(matches)
            print(f"  {subset:<28} n_series={len(matches):<6} mean_omega_levels={mean_omega:.4f}")
        else:
            print(f"  {subset:<28} NOT FOUND")

summarize("dev_manifest_OLD_buggy.json", "OLD (buggy)")
summarize("dev_manifest.json", "NEW (fixed)")

def totals(path, label):
    with open(path) as f:
        d = json.load(f)
    n_active = len(d["active_dev_series"])
    n_excluded = len(d["excluded_series"])
    print(f"{label}: active={n_active}, excluded={n_excluded}, total={n_active + n_excluded}")
totals("dev_manifest_OLD_buggy.json", "OLD (buggy)")
totals("dev_manifest.json", "NEW (fixed)")