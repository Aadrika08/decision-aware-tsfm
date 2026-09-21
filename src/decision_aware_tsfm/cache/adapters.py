"""Bounded forecast adapters for issue #17.

Validation is dependency-free. Model libraries are imported only when the historical
Moirai adapter is loaded.
"""
from __future__ import annotations

from datetime import datetime
import os
import platform
import random
from typing import Any, Protocol
import warnings

import numpy as np

from .schema import TAUS

MOIRAI_REPOSITORY = "Salesforce/moirai-1.1-R-small"
MOIRAI_CHECKPOINT = "0c24ab99db2c1a70ea2a0fc03bf113329772ac64"

_REQUIRED = {
    "model", "family", "adapter", "adapter_version", "checkpoint", "device",
    "batch_size", "quantile_method", "settings", "frequency", "context_length",
    "horizon", "seed", "start_timestamp", "origin_timestamp",
    "context_start_timestamp", "fixture",
}
_SEASONAL_SETTINGS = {"season_length": 24, "residual_lag": 24}
_MOIRAI_SETTINGS = {
    "repository": MOIRAI_REPOSITORY,
    "patch_size": 32,
    "samples": 4096,
    "target_dim": 1,
    "covariates": 0,
}


class Adapter(Protocol):
    def predict(self, context: np.ndarray, job: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Return [horizon, 11] quantiles and an execution measurement."""


def _require_common(job: dict[str, Any]) -> None:
    missing = sorted(_REQUIRED - job.keys())
    if missing:
        raise ValueError(f"Adapter job is missing keys: {', '.join(missing)}")
    if job["adapter_version"] != "1":
        raise ValueError("Only adapter_version='1' is supported")
    if job["family"] not in ("classical", "tsfm"):
        raise ValueError("family must be 'classical' or 'tsfm'")
    if not isinstance(job["context_length"], int) or job["context_length"] < 1:
        raise ValueError("context_length must be a positive integer")
    if not isinstance(job["horizon"], int) or job["horizon"] < 1:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(job["seed"], int) or isinstance(job["seed"], bool):
        raise ValueError("seed must be an integer")
    if not isinstance(job["batch_size"], int) or isinstance(job["batch_size"], bool) or job["batch_size"] < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(job["settings"], dict):
        raise ValueError("settings must be a dictionary")
    for key in ("start_timestamp", "origin_timestamp", "context_start_timestamp"):
        try:
            datetime.fromisoformat(job[key])
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an ISO-8601 timestamp") from None


def validate_adapter(job: dict[str, Any]) -> None:
    """Validate a job without importing Torch or any model package."""
    if not isinstance(job, dict):
        raise ValueError("Adapter job must be a flat dictionary")
    _require_common(job)
    name = job["adapter"]
    if name in ("autoets", "autoarima"):
        raise ValueError(f"Adapter {name!r} is not qualified: approve and test a predictive-distribution method first")
    if name == "fixture":
        if job["fixture"] is not True or not str(job["model"]).startswith("fixture_"):
            raise ValueError("fixture adapter requires fixture=true and a fixture_* model label")
        if job["device"] != "cpu" or job["checkpoint"] is not None:
            raise ValueError("fixture adapter requires device='cpu' and checkpoint=null")
        if job["quantile_method"] != "constant" or set(job["settings"]) != {"prediction"}:
            raise ValueError("fixture adapter requires quantile_method='constant' and settings={'prediction': finite number}")
        try:
            if isinstance(job["settings"]["prediction"], (bool, np.bool_)):
                raise TypeError
            finite = np.isfinite(float(job["settings"]["prediction"]))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError("fixture prediction must be finite")
        return
    if name == "seasonal_naive_smoke":
        expected = (job["model"] == "seasonal_naive" and job["family"] == "classical"
                    and job["fixture"] is False and job["checkpoint"] is None
                    and job["device"] == "cpu" and job["batch_size"] == 1
                    and job["frequency"] == "H" and job["context_length"] == 2048
                    and job["horizon"] == 24
                    and job["quantile_method"] == "empirical_lag24_linear"
                    and job["settings"] == _SEASONAL_SETTINGS)
        if not expected:
            raise ValueError("seasonal_naive_smoke requires model='seasonal_naive', family='classical', "
                             "fixture=false, CPU, batch_size=1, hourly context_length=2048, horizon=24, "
                             "quantile_method='empirical_lag24_linear', and lag-24 settings")
        return
    if name == "moirai_smoke":
        expected = (job["model"] == "moirai_1.1_R_small" and job["family"] == "tsfm"
                    and job["fixture"] is False and job["checkpoint"] == MOIRAI_CHECKPOINT
                    and job["device"] in ("cpu", "cuda") and job["batch_size"] == 1
                    and job["frequency"] == "H" and job["context_length"] == 2048
                    and job["horizon"] == 24 and job["quantile_method"] == "sample_linear"
                    and job["settings"] == _MOIRAI_SETTINGS)
        if not expected:
            raise ValueError("moirai_smoke requires the pinned Moirai 1.1 R-small checkpoint, tsfm family, "
                             "fixture=false, batch_size=1, hourly context_length=2048, horizon=24, "
                             "sample_linear quantiles, 4096 samples, and patch_size=32")
        return
    raise ValueError(f"Unknown adapter: {name!r}")


def _context(context: np.ndarray, job: dict[str, Any]) -> np.ndarray:
    values = np.asarray(context, dtype=np.float64)
    if values.shape != (job["context_length"],) or not np.isfinite(values).all():
        raise ValueError(f"context must contain exactly context_length={job['context_length']} finite values")
    return values


class _FixtureAdapter:
    def predict(self, context: np.ndarray, job: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        validate_adapter(job)
        _context(context, job)
        quantiles = np.full((job["horizon"], len(TAUS)), float(job["settings"]["prediction"]))
        return quantiles, {"hardware_platform": platform.platform(), "device": "cpu",
                           "peak_memory_bytes": 0}


class _SeasonalNaiveAdapter:
    def predict(self, context: np.ndarray, job: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        validate_adapter(job)
        values = _context(context, job)
        residuals = values[24:] - values[:-24]
        quantiles = values[-24:, None] + np.quantile(residuals, TAUS, method="linear")[None, :]
        return quantiles, {"hardware_platform": platform.platform(), "device": "cpu",
                           "peak_memory_bytes": 0}


class _MoiraiAdapter:
    def __init__(self, job: dict[str, Any]):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

        if job["device"] == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; explicitly choose device='cpu'")
        torch.use_deterministic_algorithms(True, warn_only=job["device"] == "cuda")
        module = MoiraiModule.from_pretrained(MOIRAI_REPOSITORY, revision=MOIRAI_CHECKPOINT)
        model = MoiraiForecast(module=module, prediction_length=24, context_length=2048,
                               patch_size=32, num_samples=4096, target_dim=1,
                               feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).eval()
        self.predictor = model.create_predictor(batch_size=1, device=job["device"])

    def predict(self, context: np.ndarray, job: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        validate_adapter(job)
        values = _context(context, job)
        import pandas as pd
        import torch
        from gluonts.dataset.common import ListDataset

        random.seed(job["seed"])
        np.random.seed(job["seed"])
        torch.manual_seed(job["seed"])
        if job["device"] == "cuda":
            torch.cuda.manual_seed_all(job["seed"])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        dataset = ListDataset([{"item_id": job["model"], "target": values,
                                "start": pd.Period(job["context_start_timestamp"], freq="h")}], freq="h")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            forecast = next(iter(self.predictor.predict(dataset)))
        if forecast.start_date != pd.Period(job["origin_timestamp"], freq="h"):
            raise ValueError("Moirai forecast starts at the wrong timestamp")
        samples = np.asarray(forecast.samples)
        if samples.shape != (4096, 24) or not np.isfinite(samples).all():
            raise ValueError(f"Invalid Moirai samples: {samples.shape}")
        if job["device"] == "cuda":
            torch.cuda.synchronize()
        device = torch.cuda.get_device_name(0) if job["device"] == "cuda" else "cpu"
        peak = torch.cuda.max_memory_allocated() if job["device"] == "cuda" else 0
        return np.quantile(samples, TAUS, axis=0, method="linear").T, {
            "hardware_platform": platform.platform(), "device": device,
            "peak_memory_bytes": peak, "warnings": sorted({str(w.message) for w in caught}),
        }


def load_adapter(job: dict[str, Any]) -> Adapter:
    """Validate and construct the configured adapter."""
    validate_adapter(job)
    if job["adapter"] == "fixture":
        return _FixtureAdapter()
    if job["adapter"] == "seasonal_naive_smoke":
        return _SeasonalNaiveAdapter()
    return _MoiraiAdapter(job)
