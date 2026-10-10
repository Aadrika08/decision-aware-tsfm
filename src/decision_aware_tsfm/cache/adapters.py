"""Bounded smoke and pilot forecast adapters.

Validation is dependency-free. Model libraries are imported only when an adapter is loaded.
"""
from __future__ import annotations

from datetime import datetime
import importlib.metadata
import os
import platform
import random
import time
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
_LEVELS = [20, 50, 82, 90, 98]
_PILOT_ADAPTERS = {"autoarima_native", "autoets_simulated", "seasonal_naive_native", "moirai_pilot"}


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
    if name in _PILOT_ADAPTERS:
        from .pilot import pilot_models

        expected = {model["adapter"]: model for model in pilot_models(job["device"])}[name]
        configured = {key: job[key] for key in expected}
        if configured != expected or job["frequency"] != "H" or job["context_length"] != 2048:
            raise ValueError(f"{name} does not match the qualified hourly pilot configuration")
        if job["horizon"] not in (1, 24, 168) or job["seed"] not in (42, 43):
            raise ValueError("pilot horizon must be 1, 24, or 168 and seed must be 42 or 43")
        if not isinstance(job["fixture"], bool):
            raise ValueError("pilot jobs require an explicit fixture boolean")
        return
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
    raw = np.asarray(context)
    if raw.dtype.kind not in "iuf":
        raise ValueError("context must contain real numeric values")
    values = raw.astype(np.float64, copy=False)
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


def _warnings(caught) -> list[dict[str, str]]:
    return sorted(({"category": w.category.__name__, "message": str(w.message)} for w in caught),
                  key=lambda value: (value["category"], value["message"]))


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _fitted_specification(model, kind: str) -> dict[str, Any]:
    fitted = model.model_
    keys = (("arma", "coef", "sigma2", "distribution", "aic", "aicc") if kind == "arima"
            else ("components", "m", "n", "n_params", "sigma", "sigma2", "distribution",
                  "aic", "aicc", "bic", "par"))
    return {"kind": kind, "season_length": 24,
            **{key: _json_value(fitted[key]) for key in keys if key in fitted}}


def _native_quantiles(forecast: dict[str, Any]) -> np.ndarray:
    columns = [forecast[f"lo-{int(round((1 - 2 * tau) * 100))}"] if tau < .5
               else forecast["mean"] if tau == .5
               else forecast[f"hi-{int(round((2 * tau - 1) * 100))}"] for tau in TAUS]
    return np.column_stack(columns)


class _StatsForecastAdapter:
    def __init__(self, kind: str):
        self.kind = kind
        from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive

        self.model_class = {"arima": AutoARIMA, "ets": AutoETS, "seasonal": SeasonalNaive}[kind]

    def predict(self, context: np.ndarray, job: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        validate_adapter(job)
        values = _context(context, job)
        settings = job["settings"]
        if self.kind == "arima":
            model = self.model_class(**{key: value for key, value in settings.items() if key != "levels"})
        elif self.kind == "ets":
            model = self.model_class(season_length=24, model="ZZZ", damped=None, phi=None,
                                     distribution="normal", prediction_intervals=None)
        else:
            model = self.model_class(season_length=24)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(values)
            fitting_seconds = max(time.perf_counter() - started, 1e-12)
            if self.kind == "ets":
                paths = np.asarray(model.simulate(h=job["horizon"], n_paths=4096,
                                                   seed=job["seed"], error_distribution=None))
                if paths.shape != (4096, job["horizon"]):
                    raise ValueError(f"Invalid AutoETS simulation shape: {paths.shape}")
                quantiles = np.quantile(paths, TAUS, axis=0, method="linear").T
            else:
                quantiles = _native_quantiles(model.predict(h=job["horizon"], level=_LEVELS))
        return quantiles, {"hardware_platform": platform.platform(), "device": "cpu",
                           "peak_memory_bytes": 0, "fitting_seconds": fitting_seconds,
                           "warnings": _warnings(caught),
                           "fitted_model": _fitted_specification(model, self.kind)}


class _MoiraiAdapter:
    def __init__(self, job: dict[str, Any]):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

        if job["device"] == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; explicitly choose device='cpu'")
        torch.use_deterministic_algorithms(True, warn_only=job["device"] == "cuda")
        module = MoiraiModule.from_pretrained(MOIRAI_REPOSITORY, revision=MOIRAI_CHECKPOINT)
        model = MoiraiForecast(module=module, prediction_length=job["horizon"], context_length=2048,
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
        expected = (4096, job["horizon"])
        if samples.shape != expected or not np.isfinite(samples).all():
            raise ValueError(f"Invalid Moirai samples: {samples.shape}; expected {expected}")
        if job["device"] == "cuda":
            torch.cuda.synchronize()
        device = torch.cuda.get_device_name(0) if job["device"] == "cuda" else "cpu"
        peak = torch.cuda.max_memory_allocated() if job["device"] == "cuda" else 0
        measurement = {
            "hardware_platform": platform.platform(), "device": device,
            "peak_memory_bytes": peak, "warnings": sorted({str(w.message) for w in caught}),
        }
        if job["adapter"] == "moirai_pilot":
            measurement.update(warnings=_warnings(caught), fitting_seconds=0.0,
                               fitted_model={"repository": MOIRAI_REPOSITORY,
                                             "checkpoint": MOIRAI_CHECKPOINT})
        return np.quantile(samples, TAUS, axis=0, method="linear").T, measurement


def load_adapter(job: dict[str, Any]) -> Adapter:
    """Validate and construct the configured adapter."""
    validate_adapter(job)
    if job["adapter"] == "fixture":
        return _FixtureAdapter()
    if job["adapter"] == "seasonal_naive_smoke":
        return _SeasonalNaiveAdapter()
    if job["adapter"] in {"autoarima_native", "autoets_simulated", "seasonal_naive_native"}:
        try:
            version = importlib.metadata.version("statsforecast")
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError("statsforecast==2.1.1 is required") from None
        if version != "2.1.1":
            raise RuntimeError(f"statsforecast==2.1.1 is required, found {version}")
    if job["adapter"] == "autoarima_native":
        return _StatsForecastAdapter("arima")
    if job["adapter"] == "autoets_simulated":
        return _StatsForecastAdapter("ets")
    if job["adapter"] == "seasonal_naive_native":
        return _StatsForecastAdapter("seasonal")
    if job["adapter"] == "moirai_pilot":
        return _MoiraiAdapter(job)
    return _MoiraiAdapter(job)
