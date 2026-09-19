"""Write an explicitly invented workload for pipeline/consumer checks."""
import io
import json
from pathlib import Path

import numpy as np

from .planner import digest
from .schema import TAUS


def create_fixture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, actual in enumerate((10.0, 7.0)):
        output = io.BytesIO()
        np.save(output, np.full(2400, actual), allow_pickle=False)
        data = output.getvalue()
        name = f"invented-{index}.npy"
        (directory / name).write_bytes(data)
        for origin in (2048, 2216):
            for horizon in (1, 24, 168):
                rows.append(dict(
                    dataset="synthetic_fixture", config="hourly", series_id=f"fixture_series_{index}",
                    item_id=f"fixture_item_{index}", channel=0, frequency="H",
                    source_revision="invented-v1", source_sha256=digest(data), source_path=name,
                    start_timestamp="2020-01-01T00:00:00", source_length=2400,
                    usable_start=0, usable_end=2400, context_start=origin - 2048,
                    context_length=2048, origin=origin, horizon=horizon, seed=42 + index,
                    training_end=2048, validation_end=2400,
                ))
    workload = dict(schema_version=1, split="development", fixture=True,
                    quantiles=list(TAUS), rows=rows,
                    provenance={"description": "Invented constant series; not model or research evidence"})
    models = dict(schema_version=1, models=[
        dict(model=f"fixture_{family}", family=family, adapter="fixture", adapter_version="1",
             checkpoint=None, device="cpu", batch_size=1, quantile_method="constant",
             settings={"prediction": prediction})
        for family, prediction in (("classical", 8.0), ("tsfm", 11.0))
    ])
    for name, value in (("workload.json", workload), ("models.json", models)):
        (directory / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return {"fixture": True, "workload": str(directory / "workload.json"),
            "models": str(directory / "models.json"),
            "description": "24 invented jobs / 1544 rows; no real dataset or forecaster"}
