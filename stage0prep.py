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


def extract_targets_from_dataset(ds) -> list[np.ndarray]:
    """Extracts target time series arrays from a Hugging Face Dataset object."""
    series_list = []

    target_col = None
    for col in ["target", "values", "series_value", "target_values"]:
        if col in ds.column_names:
            target_col = col
            break

    if target_col:
        for val in ds[target_col]:
            if val is not None and len(val) > 0:
                series_list.append(np.array(val, dtype=np.float64))

    return series_list


def extract_targets_from_arrow_file(file_path: str) -> list[np.ndarray]:
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
                    series_list.append(np.array(arr, dtype=np.float64))

    return series_list


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

        # Attempt primary HF load_from_disk
        try:
            ds = load_from_disk(dir_path)
            extracted_series = extract_targets_from_dataset(ds)
        except Exception:

            arrow_files = glob.glob(
                os.path.join(dir_path, "*.arrow"), recursive=True
            )
            for af in arrow_files:
                if "cache" in af:
                    continue
                extracted_series.extend(extract_targets_from_arrow_file(af))

        for ts in extracted_series:
            ts = ts[np.isfinite(ts)]
            length = len(ts)
            series_counter += 1
            series_id = f"{subset_name}_idx_{series_counter}"

            # 1,000 timestep floor
            if length < 1000:
                excluded_series.append(
                    {
                        "series_id": series_id,
                        "subset": subset_name,
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
                    "length": length,
                    "omega_levels": round(omega_lev, 4),
                    "omega_diffs": round(omega_dif, 4),
                    "horizons": [1, 24, 168],
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