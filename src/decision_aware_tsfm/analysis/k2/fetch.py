"""Download the published GIFT-Eval per-model result files (network; not used by the analysis)."""
from __future__ import annotations

import os

import pandas as pd

RESULTS_SPACE = "https://huggingface.co/spaces/Salesforce/GIFT-Eval"


def fetch_results(out_path: str) -> None:
    """Download and concatenate every published per-model result file.

    Refuses to replace an existing file: the saved results are a frozen input,
    and the leaderboard changes over time.
    """
    if os.path.exists(out_path):
        raise FileExistsError(
            f"{out_path} already exists. It is a frozen input and is not overwritten; "
            "pass --results with a new path to download a fresh copy.")
    import requests
    print(f"Listing models at {RESULTS_SPACE} ...")
    tree = requests.get(
        "https://huggingface.co/api/spaces/Salesforce/GIFT-Eval/tree/main/results",
        timeout=60).json()
    names = [e["path"].split("/")[-1] for e in tree if e["type"] == "directory"]
    print(f"  {len(names)} model directories")
    frames, failed = [], []
    for i, name in enumerate(names, 1):
        url = (f"https://huggingface.co/spaces/Salesforce/GIFT-Eval/"
               f"resolve/main/results/{name}/all_results.csv")
        try:
            d = pd.read_csv(url)
            if "dataset" not in d.columns:
                failed.append(name)
                continue
            frames.append(d)
        except Exception:
            failed.append(name)
        if i % 25 == 0:
            print(f"  {i}/{len(names)}")
    r = pd.concat(frames, ignore_index=True)
    r.to_csv(out_path, index=False, lineterminator="\n")
    print(f"Wrote {out_path}: {len(r)} rows, {r.model.nunique()} models, "
          f"{r.dataset.nunique()} configs")
    if failed:
        print(f"  skipped (no parseable results file): {failed}")
