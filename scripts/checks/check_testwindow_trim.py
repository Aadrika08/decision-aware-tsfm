import json
import math
import re
from pathlib import Path
import numpy as np
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "artifacts" / "reference" / "manifests" / "dev_manifest.json"

with MANIFEST.open(encoding="utf-8") as f:
    d = json.load(f)

SUBSET_PATHS = {
    "m4_daily": "m4_daily",
    "m4_hourly": "m4_hourly",
    "m4_weekly": "m4_weekly",
    "m4_monthly": "m4_monthly",
    "m4_quarterly": "m4_quarterly",
    "m4_yearly": "m4_yearly",
    "covid_deaths": "covid_deaths",
    "electricity_H": "electricity/H",
    "electricity_15T": "electricity/15T",
    "electricity_D": "electricity/D",
    "bitbrains_fast_storage_H": "bitbrains_fast_storage/H",
    "bitbrains_fast_storage_5T": "bitbrains_fast_storage/5T",
    "bitbrains_rnd_H": "bitbrains_rnd/H",
    "bitbrains_rnd_5T": "bitbrains_rnd/5T",
    "ett1_H": "ett1/H",
    "ett2_H": "ett2/H",
    "jena_weather": "jena_weather",
}

subset = "covid_deaths"
data_dir = f"./gift_eval_data/{SUBSET_PATHS.get(subset, subset)}"

TEST_SPLIT = 0.1
MAX_WINDOW = 20
M4_PRED_LENGTH_MAP = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
PRED_LENGTH_MAP = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
TERM_MULTIPLIERS = {"short": 1, "medium": 10, "long": 15}


def maybe_reconvert_freq(freq):
    return {"Y": "A", "YE": "A", "QE": "Q", "ME": "M", "h": "H", "min": "T", "s": "S", "us": "U"}.get(freq, freq)


def base_freq_code(freq):
    freq = freq.strip()
    freq = re.sub(r"^\d+", "", freq)
    freq = freq.split("-")[0]
    return maybe_reconvert_freq(freq)


def compute_windows(min_length, prediction_length):
    if prediction_length <= 0:
        return 1
    w = math.ceil(TEST_SPLIT * min_length / prediction_length)
    return min(max(1, w), MAX_WINDOW)


def compute_test_trim(subset_name, freq_code, min_length):
    is_m4 = "m4" in subset_name
    pred_map = M4_PRED_LENGTH_MAP if is_m4 else PRED_LENGTH_MAP
    if freq_code is None or freq_code not in pred_map:
        return int(min_length * TEST_SPLIT)
    base_pred_len = pred_map[freq_code]
    if is_m4:
        return base_pred_len * TERM_MULTIPLIERS["short"] * 1
    trims = []
    for multiplier in TERM_MULTIPLIERS.values():
        pl = base_pred_len * multiplier
        w = compute_windows(min_length, pl)
        trims.append(pl * w)
    return max(trims)


def split_channels(raw):
    """Same split as the GIFT-Eval manifest builder. Multivariate targets are
    (num_channels, num_timesteps), not one flattened series."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 1:
        return [raw]
    elif raw.ndim == 2:
        return [raw[ch] for ch in range(raw.shape[0])]
    return []


def idx_of(series_id):
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

raw_series = []
for v in ds[target_col]:
    if v is None or len(v) == 0:
        continue
    raw = np.array(v, dtype=np.float64)
    for channel_arr in split_channels(raw):
        if len(channel_arr) > 0:
            raw_series.append(channel_arr)  # keep raw, NaN-strip later

min_length = min(len(ts) for ts in raw_series)  # from raw (pre-NaN-strip) lengths

raw_freq = None
if "freq" in ds.column_names:
    val = ds["freq"][0]
    if val:
        raw_freq = str(val)
freq_code = base_freq_code(raw_freq) if raw_freq else None

test_trim = compute_test_trim(subset, freq_code, min_length)

print(f"subset: {subset}")
print(f"detected freq: {raw_freq!r} -> code {freq_code!r}")
print(f"min_length across subset (per-channel): {min_length}")
print(f"computed test_trim: {test_trim}")
print()

# Trim the RAW array first (matching GIFT-Eval's position-based split),
# then NaN-strip in the same order as the manifest builder. This is the only
# place expected_ordered is computed; no second pass overwrites it.
expected_ordered = []
for raw_ts in raw_series:
    trimmed = raw_ts[:-test_trim] if len(raw_ts) > test_trim else raw_ts[:0]
    ts = trimmed[np.isfinite(trimmed)]
    expected_ordered.append(len(ts))

print("counts:", len(expected_ordered), len(manifest_lengths_ordered))
print("expected (first 5):", expected_ordered[:5])
print("manifest (first 5):", manifest_lengths_ordered[:5])

mismatches = [(i, e, m) for i, (e, m) in enumerate(zip(expected_ordered, manifest_lengths_ordered)) if e != m]
print("Exact index-for-index match?", len(mismatches) == 0)
if mismatches:
    print(f"{len(mismatches)} mismatches. First 5:", mismatches[:5])
