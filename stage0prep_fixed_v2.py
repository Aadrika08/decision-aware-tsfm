import glob
import hashlib
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

# Known duplicate raw directories to skip entirely. jena_weather has a flat
# copy of its 10T data sitting alongside the proper frequency subfolders
# (jena_weather/10T, /D, /H) -- confirmed byte-identical via NaN-aware
# comparison. Left on disk (not deleted) but excluded here so it doesn't
# double-count in the manifest. Path is relative to DATA_DIR, forward
# slashes, matched against the directory a dataset_info.json sits in
# directly (not a prefix match, so it doesn't also skip the real
# subfolders).
SKIP_DUPLICATE_DIRS = {"jena_weather"}

HORIZONS = [1, 24, 168]

FOUR096_CONTEXT = 4096  # matches the source paper's TSFM context cap

# Matches Aadrika's cache consumer contract (docs/runbooks/cache-consumer-
# contract.md): "dataset" = source repository, e.g. "Salesforce/GiftEval".
# "config" = source configuration, e.g. "ett1/H". Assumes every dataset
# here traces to the same upstream repo -- flagged for confirmation with
# Aadrika; if any subset (e.g. Bitbrains, sourced from Grid Workloads
# Archive rather than GIFT-Eval's own HF repo) actually has a different
# source repository, this needs a per-dataset mapping instead of one
# constant.
GIFT_EVAL_SOURCE_REPO = "Salesforce/GiftEval"

# --- Exact GIFT-Eval test-window replication -------------------------------
TEST_SPLIT = 0.1
MAX_WINDOW = 20

M4_PRED_LENGTH_MAP = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
PRED_LENGTH_MAP = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
TERM_MULTIPLIERS = {"short": 1, "medium": 10, "long": 15}


def maybe_reconvert_freq(freq: str) -> str:
    deprecated_map = {
        "Y": "A", "YE": "A", "QE": "Q", "ME": "M",
        "h": "H", "min": "T", "s": "S", "us": "U",
    }
    return deprecated_map.get(freq, freq)


def base_freq_code(freq: str) -> str:
    freq = freq.strip()
    freq = re.sub(r"^\d+", "", freq)
    freq = freq.split("-")[0]
    return maybe_reconvert_freq(freq)


def detect_subset_freq(dir_path: str, ds) -> str | None:
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
    """Split a target array into per-channel 1D series. Returns
    (channel_index_or_None, 1D array) tuples. channel_index is None for
    series that were already univariate.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 1:
        return [(None, raw)]
    elif raw.ndim == 2:
        return [(ch, raw[ch]) for ch in range(raw.shape[0])]
    else:
        print(f"  [WARN] unexpected target ndim={raw.ndim}, shape={raw.shape}; skipping")
        return []


def extract_targets_from_dataset(ds) -> list[tuple[int | None, np.ndarray, str]]:
    """Extracts target time series arrays from a Hugging Face Dataset object.

    Returns (channel_index_or_None, 1D array, item_id) tuples. item_id comes
    from the dataset's own 'item_id' field when present (GIFT-Eval's native
    per-series identifier), falling back to a positional placeholder
    ('row{n}') only when the field is missing -- this fallback is the one
    case where series identity is still order-dependent, and is flagged in
    the manifest via item_id_is_fallback.
    """
    series_list = []

    target_col = None
    for col in ["target", "values", "series_value", "target_values"]:
        if col in ds.column_names:
            target_col = col
            break

    has_item_id = "item_id" in ds.column_names

    if target_col:
        for row_idx, val in enumerate(ds[target_col]):
            if val is not None and len(val) > 0:
                raw = np.array(val, dtype=np.float64)
                if has_item_id:
                    item_id = str(ds[row_idx]["item_id"])
                    fallback = False
                else:
                    item_id = f"row{row_idx}"
                    fallback = True
                for ch, arr in split_channels(raw):
                    series_list.append((ch, arr, item_id, fallback))

    return series_list


def extract_targets_from_arrow_file(file_path: str) -> list[tuple[int | None, np.ndarray, str, bool]]:
    series_list = []
    table = None

    try:
        table = feather.read_table(file_path)
    except Exception:
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

        has_item_id = "item_id" in table.column_names

        if target_col:
            target_column = table.column(target_col)
            item_id_column = table.column("item_id") if has_item_id else None
            for row_idx in range(len(target_column)):
                arr = target_column[row_idx].as_py()
                if arr is not None and len(arr) > 0:
                    raw = np.array(arr, dtype=np.float64)
                    if has_item_id:
                        item_id = str(item_id_column[row_idx].as_py())
                        fallback = False
                    else:
                        item_id = f"row{row_idx}"
                        fallback = True
                    for ch, channel_arr in split_channels(raw):
                        series_list.append((ch, channel_arr, item_id, fallback))

    return series_list


def load_train_only(dir_path: str):
    ds = load_from_disk(dir_path)
    if hasattr(ds, "keys") and "train" in ds.keys():
        return ds["train"]
    return ds


def get_dataset_config(rel_path: str) -> tuple[str, str]:
    """Returns (dataset, config) matching Aadrika's cache consumer contract:
    dataset is the source repository, config is the source configuration in
    its natural slash form (e.g. "ett1/H"), not underscore-joined. Replaces
    an earlier version that split subset_name into two pieces (dataset=
    "ett1", config="H") -- that matched neither the contract's target
    format nor the existing smoke test's dataset=ett1_H convention, it was
    a third, invented format.
    """
    return GIFT_EVAL_SOURCE_REPO, rel_path


def compute_source_checksum(dir_path: str) -> str:
    """Hash of every arrow file's bytes in this subset directory, sorted for
    determinism. Directory-level, not per-series -- cheap, and a reasonable
    reading of the contract's "preserve source revision" language, though
    not explicitly confirmed as the intended granularity.
    """
    arrow_files = sorted(glob.glob(os.path.join(dir_path, "*.arrow")))
    hasher = hashlib.sha256()
    for af in arrow_files:
        with open(af, "rb") as f:
            hasher.update(f.read())
    return f"sha256:{hasher.hexdigest()}"


def process_all_gift_eval():
    dev_manifest = []
    excluded_series = []
    omega_values = []

    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' not found")
        return

    dataset_dirs = glob.glob(
        os.path.join(DATA_DIR, "**", "dataset_info.json"), recursive=True
    )
    print(f"Found {len(dataset_dirs)} dataset partitions. Processing...")

    for info_file in dataset_dirs:
        dir_path = os.path.dirname(info_file)
        normalized_path = dir_path.replace("\\", "/").lower()

        if any(held_out in normalized_path for held_out in STRICT_HELD_OUT):
            continue

        rel_path = os.path.relpath(dir_path, DATA_DIR).replace("\\", "/")
        if rel_path in SKIP_DUPLICATE_DIRS:
            print(f"  [SKIP] {rel_path}: known duplicate of a frequency subfolder, excluded")
            continue

        subset_name = rel_path.replace("/", "_")
        dataset, config = get_dataset_config(rel_path)

        extracted_series = []
        ds = None

        try:
            ds = load_train_only(dir_path)
            extracted_series = extract_targets_from_dataset(ds)
        except Exception:
            ds = None
            arrow_files = glob.glob(os.path.join(dir_path, "*.arrow"), recursive=True)
            for af in arrow_files:
                if "cache" in af:
                    continue
                extracted_series.extend(extract_targets_from_arrow_file(af))

        # min_length uses RAW series length (before NaN-stripping), matching
        # GIFT-Eval's own data.py: _min_series_length reads pyarrow list
        # lengths directly, with no NaN-filtering. NaN-stripping first would
        # badly undercount for series with internal missing data (found on
        # electricity_15T) and undertrim the whole subset.
        raw_series = [(ch, ts, item_id, fb) for ch, ts, item_id, fb in extracted_series if len(ts) > 0]

        if not raw_series:
            continue

        min_length = min(len(ts) for _, ts, _, _ in raw_series)
        raw_freq = detect_subset_freq(dir_path, ds)
        freq_code = base_freq_code(raw_freq) if raw_freq else None
        test_trim = compute_test_trim(subset_name, freq_code, min_length)
        source_checksum = compute_source_checksum(dir_path)

        for channel_idx, raw_ts, item_id, id_is_fallback in raw_series:
            raw_length = len(raw_ts)

            # Trim the RAW array's tail first (matching GIFT-Eval's
            # position-based split), NaN-strip after.
            if test_trim > 0:
                trimmed = raw_ts[:-test_trim] if raw_length > test_trim else raw_ts[:0]
            else:
                trimmed = raw_ts
            ts = trimmed[np.isfinite(trimmed)]
            length = len(ts)

            split_boundary_index = raw_length - test_trim if raw_length > test_trim else 0

            # Stable series ID: derived from subset + native item_id (+
            # channel for multivariate), not a running counter. Doesn't
            # shift if anything upstream reorders. NOTE: the existing
            # released smoke cache uses bare "item_0_dim0" (no subset
            # prefix) -- this is deliberately different (prefixed, so
            # series_id alone is self-describing/globally unique even
            # without the dataset+config columns attached). Flagged as a
            # proposal for the stable-ID mapping Aadrika asked for, not
            # assumed final.
            if channel_idx is not None:
                series_id = f"{subset_name}__{item_id}_dim{channel_idx}"
            else:
                series_id = f"{subset_name}__{item_id}"

            base_entry = {
                "series_id": series_id,
                "subset": subset_name,
                "dataset": dataset,
                "config": config,
                "frequency": raw_freq,
                "freq_code": freq_code,
                "item_id": item_id,
                "item_id_is_fallback": id_is_fallback,
                "channel": channel_idx,
                "source_checksum": source_checksum,
                "raw_length": raw_length,
                "test_trim": test_trim,
                "split_boundary_index": split_boundary_index,
                "length": length,
            }


            if length < 1000:
                excluded_series.append(
                    {**base_entry, "reason": "Below 1000 timestep floor"}
                )
                continue

            omega_lev = compute_omega(ts, diffs=False)
            omega_dif = compute_omega(ts, diffs=True)

            # 4,096-step column: sliced from the same RAW trimmed array
            # (before NaN-stripping), same ordering as the main trim fix --
            # NOT from the already-NaN-stripped `ts`. This matters whenever
            # a series has internal missing data; slicing the NaN-stripped
            # array would take a different, longer-history window than
            # slicing the raw one. history_start_index_4096 is recorded
            # against the trimmed (not raw) array's own indexing, i.e. 0 is
            # the start of the train portion. Per the cache contract's join
            # rule ("do not attach full-series Omega to an earlier origin
            # if it contains future observations"), this cutoff must be
            # preserved alongside the Omega value, not dropped.
            if len(trimmed) > FOUR096_CONTEXT:
                ts_4096_raw = trimmed[-FOUR096_CONTEXT:]
                history_start_index_4096 = len(trimmed) - FOUR096_CONTEXT
            else:
                ts_4096_raw = trimmed
                history_start_index_4096 = 0
            ts_4096 = ts_4096_raw[np.isfinite(ts_4096_raw)]

            omega_lev_4096 = compute_omega(ts_4096, diffs=False)
            omega_dif_4096 = compute_omega(ts_4096, diffs=True)

            dev_manifest.append(
                {**base_entry, "horizons": HORIZONS, "status": "active_dev"}
            )

            omega_values.append(
                {
                    "series_id": series_id,
                    "omega_levels": round(omega_lev, 4),
                    "omega_diffs": round(omega_dif, 4),
                    "omega_levels_4096": round(omega_lev_4096, 4),
                    "omega_diffs_4096": round(omega_dif_4096, 4),
                    "history_length_4096": len(ts_4096),
                    "history_start_index_4096": history_start_index_4096,
                }
            )

    # Series manifest: identity, checksums, windows/splits -- what the
    # forecast cache actually keys against. Kept separate from Omega values
    # per Aadrika's note: "Omega stays in a separate join, so adding a
    # variant won't invalidate unchanged forecasts."
    with open("series_manifest.json", "w") as f:
        json.dump(
            {
                "active_dev_series": dev_manifest,
                "excluded_series": excluded_series,
                "held_out_strict": ["M5_Retail", "PJM_Electricity"],
            },
            f,
            indent=4,
        )

    with open("omega_values.json", "w") as f:
        json.dump({"omega_values": omega_values}, f, indent=4)

    print(f"\nManifest Generation Complete")
    print(f"Active Series (>=1000 ts): {len(dev_manifest)}")
    print(f"Excluded Series (<1000 ts): {len(excluded_series)}")
    print(f"Omega values written for {len(omega_values)} active series")


if __name__ == "__main__":
    process_all_gift_eval()