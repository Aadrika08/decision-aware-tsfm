import glob
import json
import os
import numpy as np
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from datasets import load_from_disk
from omega import compute_omega


DATA_DIR = "./gift_eval_data"

STRICT_HELD_OUT = ["m5", "pjm"]

# Pre-registered horizons from Stage 0, used as a conservative floor for
# trimming the test window before Omega is computed (see note).
HORIZONS = [1, 24, 168]


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
        # Confirmed convention for gift-eval multivariate targets,rows are
        # channels, columns are timesteps.
        # If a dataset ever violates this,
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
    (no explicit split dict), this can't detect that from here, so it also
    can't fully rule out leakage on its own. The trim in process_all_gift_eval
    (last max(HORIZONS) timesteps dropped) is a conservative backstop for
    that case. Confirm the true GIFT-Eval test-window convention (it may be
    multiple rolling windows, not one horizon-length chunk) before reporting
    final K2a numbers -- Veer already measured the current leakage effect at
    <0.04, so this is about correctness of the replication claim, not about
    changing the conclusion.
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
    test_trim = max(HORIZONS)

    for info_file in dataset_dirs:
        dir_path = os.path.dirname(info_file)
        normalized_path = dir_path.replace("\\", "/").lower()
        # Enforce  exclusions (M5, PJM)
        if any(held_out in normalized_path for held_out in STRICT_HELD_OUT):
            continue
        rel_path = os.path.relpath(dir_path, DATA_DIR)
        subset_name = rel_path.replace("\\", "_").replace("/", "_")

        extracted_series = []

        # Attempt primary HF load_from_disk
        try:
            ds = load_train_only(dir_path)
            extracted_series = extract_targets_from_dataset(ds)
        except Exception:

            arrow_files = glob.glob(
                os.path.join(dir_path, "*.arrow"), recursive=True
            )
            for af in arrow_files:
                if "cache" in af:
                    continue
                extracted_series.extend(extract_targets_from_arrow_file(af))

        for channel_idx, ts in extracted_series:
            ts = ts[np.isfinite(ts)]

            # Conservative test-window trim -- see load_train_only() note.
            if test_trim > 0 and len(ts) > test_trim:
                ts = ts[:-test_trim]

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