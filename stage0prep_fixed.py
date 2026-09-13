import glob
import json
import math
import os
import re
import numpy as np
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from datasets import load_from_disk
from omega import compute_omega


DATA_DIR = "./gift_eval_data"

STRICT_HELD_OUT = ["m5", "pjm"]

# Pre-registered horizons from Stage 0. No longer used to size the
# test-window trim -- kept for the manifest's "horizons" field, which
# downstream analysis still reads.
HORIZONS = [1, 24, 168]

# --- Exact GIFT-Eval test-window replication -------------------------------
# Copied from GIFT-Eval's own src/gift_eval/data.py (Aksu et al.). The real
# test region for a (dataset, frequency, term) config is
# prediction_length * windows, NOT a flat fraction of each series' own
# length. windows is computed once per dataset config from the MINIMUM
# series length in that config, rounded up to a whole number of
# prediction-length blocks, capped at MAX_WINDOW. M4 datasets always use
# exactly 1 window regardless of this formula.

TEST_SPLIT = 0.1
MAX_WINDOW = 20

M4_PRED_LENGTH_MAP = {
    "A": 6,
    "Q": 8,
    "M": 18,
    "W": 13,
    "D": 14,
    "H": 48,
}

PRED_LENGTH_MAP = {
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
}

# short/medium/long multipliers from GIFT-Eval's Term enum. Our pipeline
# doesn't track "term" per subset the way GIFT-Eval's Dataset class does --
# a single raw directory here (e.g. ett1/H) backs all three term configs
# in the real benchmark. For non-M4 datasets we compute the test region
# under all three terms and trim by the LARGEST, guaranteeing Omega never
# overlaps the true test window under any term the published K2a results
# might use. This can overtrim subsets that only ever have a short-term
# entry in GIFT-Eval's own results table, but never undertrims (no
# leakage), and avoids hand-transcribing which terms exist per subset.
TERM_MULTIPLIERS = {"short": 1, "medium": 10, "long": 15}


def maybe_reconvert_freq(freq: str) -> str:
    """Map newer pandas frequency aliases back to the older ones GIFT-Eval's
    maps use, e.g. 'h' -> 'H', 'min' -> 'T'. Copied from data.py.
    """
    deprecated_map = {
        "Y": "A",
        "YE": "A",
        "QE": "Q",
        "ME": "M",
        "h": "H",
        "min": "T",
        "s": "S",
        "us": "U",
    }
    return deprecated_map.get(freq, freq)


def base_freq_code(freq: str) -> str:
    """Reduce a raw frequency string to the base code PRED_LENGTH_MAP /
    M4_PRED_LENGTH_MAP key on. Strips a leading numeric multiplier
    ('15T' -> 'T', '10S' -> 'S') and any anchor suffix
    ('A-DEC' -> 'A', 'W-THU' -> 'W'), then normalizes deprecated aliases.
    """
    freq = freq.strip()
    freq = re.sub(r"^\d+", "", freq)
    freq = freq.split("-")[0]
    return maybe_reconvert_freq(freq)


def detect_subset_freq(dir_path: str, ds) -> str | None:
    """Read the per-series 'freq' field GIFT-Eval's HF datasets store.
    Tries the primary Dataset object first, then falls back to reading it
    directly from an arrow file if the primary load failed. Returns None
    if it can't be determined, in which case the caller falls back to a
    flat 10%-of-min-length approximation for that subset only.
    """
    try:
        if ds is not None and "freq" in ds.column_names:
            val = ds["freq"][0]
            if val:
                return str(val)
    except Exception:
        pass

    try:
        arrow_files = glob.glob(os.path.join(dir_path, "*.arrow"))
        for af in arrow_files:
            if "cache" in af:
                continue
            table = feather.read_table(af)
            if "freq" in table.column_names:
                val = table.column("freq")[0].as_py()
                if val:
                    return str(val)
    except Exception:
        pass

    return None


def compute_windows(min_length: int, prediction_length: int) -> int:
    if prediction_length <= 0:
        return 1
    w = math.ceil(TEST_SPLIT * min_length / prediction_length)
    return min(max(1, w), MAX_WINDOW)


def compute_test_trim(subset_name: str, freq_code: str | None, min_length: int) -> int:
    """Returns the number of trailing timesteps to trim from every series in
    this subset, matching (or conservatively exceeding) GIFT-Eval's actual
    test-window size. See module docstring above for the M4 special case
    and the max-over-terms approximation for everything else.
    """
    is_m4 = "m4" in subset_name
    pred_map = M4_PRED_LENGTH_MAP if is_m4 else PRED_LENGTH_MAP

    if freq_code is None or freq_code not in pred_map:
        print(
            f"  [WARN] {subset_name}: could not resolve frequency "
            f"({freq_code!r}) to a known prediction-length code; "
            f"falling back to a flat 10% trim for this subset."
        )
        return int(min_length * TEST_SPLIT)

    base_pred_len = pred_map[freq_code]

    if is_m4:
        # M4 is only ever published at its native (short-term) horizon --
        # GIFT-Eval's results table has no medium/long columns for any M4
        # subset. windows is forced to 1 regardless of term.
        prediction_length = base_pred_len * TERM_MULTIPLIERS["short"]
        return prediction_length * 1

    trims = []
    for multiplier in TERM_MULTIPLIERS.values():
        prediction_length = base_pred_len * multiplier
        windows = compute_windows(min_length, prediction_length)
        trims.append(prediction_length * windows)
    return max(trims)


# --- End exact GIFT-Eval test-window replication ----------------------------


def split_channels(raw: np.ndarray) -> list[tuple[int | None, np.ndarray]]:
    """Split a target array into per-channel 1D series.

    GIFT-Eval / GluonTS store multivariate targets as shape (num_channels,
    num_timesteps). Univariate targets are 1D. Passing a 2D array straight
    into downstream code (e.g. `arr[np.isfinite(arr)]`) silently flattens it
    in row-major order -- i.e. channel 0 in full, then channel 1 in full,
    etc. -- producing one bogus concatenated series instead of N real ones.
    This is the bug Veer flagged (ett1_H length 121,940 == 17,420 * 7).

    Returns a list of (channel_index_or_None, 1D array) tuples. channel_index
    is None for series that were already univariate.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 1:
        return [(None, raw)]
    elif raw.ndim == 2:
        # Confirmed convention for gift-eval multivariate targets: rows are
        # channels, columns are timesteps. If a dataset ever violates this,
        # the exact-division check upstream (channel count divides length)
        # is how it would be caught.
        return [(ch, raw[ch]) for ch in range(raw.shape[0])]
    else:
        print(f"  [WARN] unexpected target ndim={raw.ndim}, shape={raw.shape}; skipping")
        return []


def extract_targets_from_dataset(ds) -> list[tuple[int | None, np.ndarray]]:
    """Extracts target time series arrays from a Hugging Face Dataset object.

    Returns (channel_index_or_None, 1D array) tuples -- see split_channels.
    """
    series_list = []

    target_col = None
    for col in ["target", "values", "series_value", "target_values"]:
        if col in ds.column_names:
            target_col = col
            break

    if target_col:
        for val in ds[target_col]:
            if val is not None and len(val) > 0:
                raw = np.array(val, dtype=np.float64)
                series_list.extend(split_channels(raw))

    return series_list


def extract_targets_from_arrow_file(file_path: str) -> list[tuple[int | None, np.ndarray]]:
    series_list = []
    table = None

    # Try Feather Reader
    try:
        table = feather.read_table(file_path)
    except Exception:
        # Try Stream Reader
        try:
            with open(file_path, "rb") as f:
                reader = ipc.RecordBatchStreamReader(f)
                table = reader.read_all()
        except Exception as e:
            print(f"Failed to read {file_path} via fallbacks: {e}")
            return []

    if table is not None:
        target_col = None
        for col in ["target", "values", "series_value", "target_values"]:
            if col in table.column_names:
                target_col = col
                break

        if target_col:
            for item in table.column(target_col):
                arr = item.as_py()
                if arr is not None and len(arr) > 0:
                    raw = np.array(arr, dtype=np.float64)
                    series_list.extend(split_channels(raw))

    return series_list


def load_train_only(dir_path: str):
    """Prefer an explicit train split if the dataset exposes one.

    NOTE: if `load_from_disk` already returns only a train-portion Dataset
    (no explicit split dict), this can't detect that from here. The trim in
    process_all_gift_eval is now the exact GIFT-Eval windows-based test
    region (see compute_test_trim above), confirmed against GIFT-Eval's
    own src/gift_eval/data.py rather than the paper's prose summary.
    """
    ds = load_from_disk(dir_path)
    if hasattr(ds, "keys") and "train" in ds.keys():
        return ds["train"]
    return ds


def process_all_gift_eval():
    dev_manifest = []
    excluded_series = []

    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' not found")
        return
    # Find directories containing dataset info
    dataset_dirs = glob.glob(
        os.path.join(DATA_DIR, "**", "dataset_info.json"), recursive=True
    )
    print(f"Found {len(dataset_dirs)} dataset partitions. Processing...")

    series_counter = 0

    for info_file in dataset_dirs:
        dir_path = os.path.dirname(info_file)
        normalized_path = dir_path.replace("\\", "/").lower()
        # Enforce  exclusions (M5, PJM)
        if any(held_out in normalized_path for held_out in STRICT_HELD_OUT):
            continue
        rel_path = os.path.relpath(dir_path, DATA_DIR)
        subset_name = rel_path.replace("\\", "_").replace("/", "_")

        extracted_series = []
        ds = None

        # Attempt primary HF load_from_disk
        try:
            ds = load_train_only(dir_path)
            extracted_series = extract_targets_from_dataset(ds)
        except Exception:
            ds = None
            arrow_files = glob.glob(
                os.path.join(dir_path, "*.arrow"), recursive=True
            )
            for af in arrow_files:
                if "cache" in af:
                    continue
                extracted_series.extend(extract_targets_from_arrow_file(af))

        # min_length uses RAW series length (before NaN-stripping), matching
        # GIFT-Eval's own data.py: _min_series_length reads pyarrow list
        # lengths directly, with no NaN-filtering. Some datasets (e.g.
        # electricity_15T) store every series at the same nominal array
        # length but pad missing history with NaN -- NaN-stripping before
        # computing min_length would badly undercount for those series and
        # undertrim the whole subset. Filter out only genuinely empty
        # (zero-length) raw series here.
        raw_series = [(ch, ts) for ch, ts in extracted_series if len(ts) > 0]

        if not raw_series:
            continue

        min_length = min(len(ts) for _, ts in raw_series)
        raw_freq = detect_subset_freq(dir_path, ds)
        freq_code = base_freq_code(raw_freq) if raw_freq else None
        test_trim = compute_test_trim(subset_name, freq_code, min_length)

        for channel_idx, raw_ts in raw_series:
            # Trim off the RAW array's tail first, matching GIFT-Eval's
            # position-based split (gluonts trims by raw timestep offset,
            # not by count of non-missing values). NaN-strip afterward,
            # for the floor check and for compute_omega.
            if test_trim > 0:
                trimmed = raw_ts[:-test_trim] if len(raw_ts) > test_trim else raw_ts[:0]
            else:
                trimmed = raw_ts
            ts = trimmed[np.isfinite(trimmed)]

            length = len(ts)
            series_counter += 1
            if channel_idx is not None:
                series_id = f"{subset_name}_ch{channel_idx}_idx_{series_counter}"
            else:
                series_id = f"{subset_name}_idx_{series_counter}"

            # 1,000 timestep floor
            if length < 1000:
                excluded_series.append(
                    {
                        "series_id": series_id,
                        "subset": subset_name,
                        "channel": channel_idx,
                        "length": length,
                        "reason": "Below 1000 timestep floor",
                    }
                )
                continue

            omega_lev = compute_omega(ts, diffs=False)
            omega_dif = compute_omega(ts, diffs=True)
            dev_manifest.append(
                {
                    "series_id": series_id,
                    "subset": subset_name,
                    "channel": channel_idx,
                    "length": length,
                    "omega_levels": round(omega_lev, 4),
                    "omega_diffs": round(omega_dif, 4),
                    "horizons": HORIZONS,
                    "status": "active_dev",
                }
            )

    # Export dev_manifest.json
    with open("dev_manifest.json", "w") as f:
        json.dump(
            {
                "active_dev_series": dev_manifest,
                "excluded_series": excluded_series,
                "held_out_strict": ["M5_Retail", "PJM_Electricity"],
            },
            f,
            indent=4,
        )
    print(f"\nManifest Generation Complete")
    print(f"Active Series (>=1000 ts): {len(dev_manifest)}")
    print(f"Excluded Series (<1000 ts): {len(excluded_series)}")


if __name__ == "__main__":
    process_all_gift_eval()