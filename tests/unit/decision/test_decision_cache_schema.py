"""Loud failures on bad input, and the output schema's own checks."""
import numpy as np
import pandas as pd
import pytest

from decision_aware_tsfm.analysis.decision.cache_io import (
    CacheError, DuplicateCacheKeyError, load_cache, qcol, restrict_to_representatives)
from decision_aware_tsfm.analysis.decision.fixture import (
    FAMILY_OF_MODEL, FIT_ONLY_DATASETS, FIXTURE_SCHEMA, REPRESENTATIVES)
from decision_aware_tsfm.analysis.decision.guards import HeldOutLeakError
from decision_aware_tsfm.analysis.decision.pipeline import build_tables
from decision_aware_tsfm.analysis.decision.schema import SchemaError, validate_selection_frame
from decision_aware_tsfm.analysis.decision.selectors import StaticOmegaRule


@pytest.fixture
def small(fixture_data):
    fc = fixture_data[0]
    return fc[(fc.dataset == "fx_tie") & (fc.horizon == 24)].copy()


def test_duplicate_cache_key_fails(small):
    with pytest.raises(DuplicateCacheKeyError):
        load_cache(pd.concat([small, small.iloc[:1]]), FIXTURE_SCHEMA)


def test_crossing_quantiles_fail(small):
    bad = small.copy()
    bad.loc[bad.index[0], qcol(0.99)] = bad.loc[bad.index[0], qcol(0.01)] - 1
    with pytest.raises(CacheError, match="crossing"):
        load_cache(bad, FIXTURE_SCHEMA)


def test_float_noise_is_not_a_crossing(small):
    ok = small.copy()
    i = ok.index[0]
    ok.loc[i, qcol(0.05)] = ok.loc[i, qcol(0.01)] - 1e-13     # 1e-13 below: rounding noise
    load_cache(ok, FIXTURE_SCHEMA)


def test_missing_quantile_fails(small):
    bad = small.copy()
    bad.loc[bad.index[3], qcol(0.5)] = np.nan
    with pytest.raises(CacheError, match="missing"):
        load_cache(bad, FIXTURE_SCHEMA)


def test_missing_quantile_column_fails(small):
    with pytest.raises(CacheError, match="missing columns"):
        load_cache(small.drop(columns=qcol(0.91)), FIXTURE_SCHEMA)


def test_inconsistent_y_across_models_fails(small):
    bad = small.copy()
    i = bad.index[bad.model == "Moirai-small"][0]
    bad.loc[i, "y"] += 1.0
    with pytest.raises(CacheError, match="different y"):
        load_cache(bad, FIXTURE_SCHEMA)


def test_family_coverage_mismatch_fails(small):
    fc = load_cache(small, FIXTURE_SCHEMA)
    drop = fc.index[(fc.model == "Moirai-small")][:5]
    with pytest.raises(CacheError, match="not covered"):
        restrict_to_representatives(fc.drop(drop), REPRESENTATIVES, FAMILY_OF_MODEL)


def test_missing_y_dropped_for_both_families_and_counted(small):
    fc = small.copy()
    tgt = fc.iloc[0][["series_id", "origin", "step"]]
    m = (fc.series_id == tgt.series_id) & (fc.origin == tgt.origin) & (fc.step == tgt.step)
    fc.loc[m, "y"] = np.nan
    out = restrict_to_representatives(load_cache(fc, FIXTURE_SCHEMA), REPRESENTATIVES, FAMILY_OF_MODEL)
    assert out.attrs["n_targets_dropped_missing_y"] == 1
    assert len(out) == (small.model != "SeasonalNaive").sum() - 2


def test_unknown_cache_schema_fails(small):
    with pytest.raises(NotImplementedError):
        load_cache(small, "aadrika-v1")


def test_wrong_representative_family_fails(small):
    with pytest.raises(CacheError):
        restrict_to_representatives(small, {"classical": "Moirai-small", "tsfm": "AutoETS"},
                                    FAMILY_OF_MODEL)


def test_missing_omega_fails(fixture_data):
    fc, omega, _ = fixture_data
    with pytest.raises(ValueError, match="no omega"):
        build_tables(fc, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega.iloc[1:],
                     fit_only_datasets=FIT_ONLY_DATASETS)


@pytest.mark.parametrize("name", ["m5_retail", "PJM_Electricity", "data/M5/daily"])
def test_held_out_dataset_blocked_from_development_pipeline(small, fixture_data, name):
    fc = small.copy(); fc["dataset"] = name
    with pytest.raises(HeldOutLeakError):
        build_tables(fc, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, fixture_data[1])


def test_static_rule_has_no_default_threshold():
    with pytest.raises(TypeError):
        StaticOmegaRule()


# ---- output schema -----------------------------------------------------------

def test_valid_results_pass(results):
    validate_selection_frame(results[0])


def _corrupt(res, **changes):
    r = res.copy()
    i = r.index[(r.status == "selected") & (r.selector == changes.pop("_selector", "always_tsfm"))][0]
    for k, v in changes.items():
        r.loc[i, k] = v
    return r


def test_negative_regret_rejected(results):
    with pytest.raises(SchemaError, match="negative regret"):
        validate_selection_frame(_corrupt(results[0], regret=-1e-9))


def test_oracle_nonzero_regret_rejected(results):
    with pytest.raises(SchemaError, match="oracle"):
        validate_selection_frame(_corrupt(results[0], _selector="hindsight_oracle", regret=1e-9))


def test_selected_row_without_cost_rejected(results):
    with pytest.raises(SchemaError):
        validate_selection_frame(_corrupt(results[0], realized_cost=np.nan))


def test_duplicate_unit_rejected(results):
    r = results[0]
    with pytest.raises(SchemaError, match="duplicate"):
        validate_selection_frame(pd.concat([r, r.iloc[:1]], ignore_index=True))


def test_extra_column_rejected(results):
    with pytest.raises(SchemaError, match="columns"):
        validate_selection_frame(results[0].assign(extra=1))
