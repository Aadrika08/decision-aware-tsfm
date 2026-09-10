import json
import re
import numpy as np
from datasets import load_from_disk

with open("dev_manifest.json") as f:
    d = json.load(f)

# Maps manifest subset_name -> real on-disk folder. Needed because
# stage0prep_fixed.py builds subset_name by flattening nested folders
# with underscores (e.g. real path electricity/H becomes subset name
# "electricity_H"), so subset_name != real folder path whenever a
# dataset has sub-frequency folders. Flat, single-level subsets
# (m4_daily, covid_deaths) are the same either way.
#
# Entries marked (verified) have been run and matched exactly.
# Entries marked (inferred) follow the same nesting pattern but haven't
# been run yet -- confirm the real folder name before trusting them.
SUBSET_PATHS = {
    "m4_daily": "m4_daily",                                   # verified
    "covid_deaths": "covid_deaths",                           # verified
    "electricity_H": "electricity/H",                         # verified
    "electricity_15T": "electricity/15T",                     # inferred
    "electricity_D": "electricity/D",                         # inferred
    "bitbrains_fast_storage_H": "bitbrains_fast_storage/H",    # inferred
    "bitbrains_fast_storage_5T": "bitbrains_fast_storage/5T",  # inferred
    "bitbrains_rnd_H": "bitbrains_rnd/H",                      # inferred
    "bitbrains_rnd_5T": "bitbrains_rnd/5T",                    # inferred
    "ett1_H": "ett1/H",                                        # inferred
    "ett2_H": "ett2/H",                                        # inferred
    "jena_weather": "jena_weather",                            # inferred (flat)
}

# Verified 2026-09-09 on: m4_daily (mixed short/long), covid_deaths
# (uniformly short, trim correctly no-ops), electricity_H (uniformly
# long, trim correctly applies). Swap `subset` below to check others
# add any missing entry to SUBSET_PATHS first.
subset = "m4_daily"
data_dir = f"./gift_eval_data/{SUBSET_PATHS.get(subset, subset)}"
test_trim = 168


def idx_of(series_id: str) -> int:
    """Recover original within-subset extraction order from the series_id
    suffix (`..._idx_<n>`), so entries can be matched by identity instead
    of by sorted length. Sorted-length comparison silently breaks on ties,
    which are common with small integer lengths.
    """
    return int(re.search(r"_idx_(\d+)$", series_id).group(1))


recs = (
    [(s["series_id"], s["length"]) for s in d["active_dev_series"] if s["subset"] == subset]
    + [(s["series_id"], s["length"]) for s in d["excluded_series"] if s["subset"] == subset]
)
recs.sort(key=lambda r: idx_of(r[0]))
manifest_lengths_ordered = [length for _, length in recs]

ds = load_from_disk(data_dir)
if hasattr(ds, "keys") and "train" in ds.keys():
    ds = ds["train"]
target_col = next(c for c in ["target", "values", "series_value", "target_values"] if c in ds.column_names)

expected_ordered = []
for v in ds[target_col]:
    if v is None or len(v) == 0:
        continue
    ts = np.array(v, dtype=np.float64)
    ts = ts[np.isfinite(ts)]
    L = len(ts)
    if test_trim > 0 and L > test_trim:
        L = L - test_trim
    expected_ordered.append(L)

print("counts:", len(expected_ordered), len(manifest_lengths_ordered))
print("expected (first 5):", expected_ordered[:5])
print("manifest (first 5):", manifest_lengths_ordered[:5])

mismatches = [
    (i, e, m) for i, (e, m) in enumerate(zip(expected_ordered, manifest_lengths_ordered)) if e != m
]
print("Exact index-for-index match?", len(mismatches) == 0)
if mismatches:
    print(f"{len(mismatches)} mismatches. First 5:", mismatches[:5])