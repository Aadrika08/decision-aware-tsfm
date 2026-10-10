"""Canonical candidate model configurations for the bounded issue #20 pilot."""

from copy import deepcopy

from .adapters import MOIRAI_CHECKPOINT, MOIRAI_REPOSITORY


_ARIMA_SETTINGS = {
    "d": None, "D": None, "max_p": 5, "max_q": 5, "max_P": 2, "max_Q": 2,
    "max_order": 5, "max_d": 2, "max_D": 1, "start_p": 2, "start_q": 2,
    "start_P": 1, "start_Q": 1, "stationary": False, "seasonal": True,
    "ic": "aicc", "stepwise": True, "nmodels": 94, "trace": False,
    "approximation": False, "method": None, "truncate": None, "test": "kpss",
    "test_kwargs": None, "seasonal_test": "seas", "seasonal_test_kwargs": None,
    "allowdrift": True, "allowmean": True, "blambda": None, "biasadj": False,
    "season_length": 24, "distribution": "normal", "prediction_intervals": None,
    "levels": [20, 50, 82, 90, 98],
}


def pilot_models(device: str = "cuda") -> list[dict]:
    """Return fresh planner-compatible configs for the four pilot candidates."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    common = {"family": "classical", "adapter_version": "1", "checkpoint": None,
              "device": "cpu", "batch_size": 1}
    models = [
        {**common, "model": "autoarima_native", "adapter": "autoarima_native",
         "quantile_method": "native_gaussian", "settings": _ARIMA_SETTINGS},
        {**common, "model": "autoets_simulated", "adapter": "autoets_simulated",
         "quantile_method": "simulation_linear",
         "settings": {"season_length": 24, "model": "ZZZ", "distribution": "normal",
                      "damped": None, "phi": None, "prediction_intervals": None,
                      "samples": 4096}},
        {**common, "model": "seasonal_naive_native", "adapter": "seasonal_naive_native",
         "quantile_method": "native_gaussian",
         "settings": {"season_length": 24, "distribution": "normal",
                      "prediction_intervals": None, "levels": [20, 50, 82, 90, 98]}},
        {"model": "moirai_pilot", "family": "tsfm", "adapter": "moirai_pilot",
         "adapter_version": "1", "checkpoint": MOIRAI_CHECKPOINT, "device": device,
         "batch_size": 1, "quantile_method": "sample_linear",
         "settings": {"repository": MOIRAI_REPOSITORY, "patch_size": 32, "samples": 4096,
                      "target_dim": 1, "covariates": 0}},
    ]
    return deepcopy(models)
