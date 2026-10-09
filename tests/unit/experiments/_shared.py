"""Shared, cached fixture objects for the experiment-layer tests.

Built once per test process. Everything here is the synthetic fixture.
"""
from functools import lru_cache

from decision_aware_tsfm.analysis.decision.fixture import (
    FAMILY_OF_MODEL, FIT_ONLY_DATASETS, FIXTURE_SCHEMA, FIXTURE_STATIC_THRESHOLD,
    REPRESENTATIVES, make_fixture)
from decision_aware_tsfm.analysis.decision.pipeline import build_tables, run

FRAME_KEYS = ("per_origin", "units", "val", "test", "n_val", "decision_origins", "fit_units")


@lru_cache(maxsize=None)
def fixture_data():
    return make_fixture(seed=7)


@lru_cache(maxsize=None)
def tables():
    fc, omega, _ = fixture_data()
    return build_tables(fc, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega,
                        fit_only_datasets=FIT_ONLY_DATASETS)


@lru_cache(maxsize=None)
def tables_without_fit_split():
    """Every fixture dataset, fx_dfit included, placed on the evaluation side.
    Horizon 24 only, which keeps the build short; the settings are the fixture's."""
    fc, omega, _ = fixture_data()
    return build_tables(fc[fc.horizon == 24], FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega,
                        fit_only_datasets=())


def cli_patches():
    """Patches that let a command-line entry point reuse the cached fixture
    instead of rebuilding it. The stand-ins check that the entry point asks for
    exactly what the cache holds, so the wiring is still tested."""
    from unittest import mock

    def fake_fixture(seed=7):
        assert seed == 7, seed
        return fixture_data()

    def fake_build(fc, schema, representatives, family_of_model, omega, **kw):
        assert fc is fixture_data()[0] and omega is fixture_data()[1]
        assert (schema, representatives, family_of_model) == (FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL)
        assert kw == {"fit_only_datasets": FIT_ONLY_DATASETS}, kw
        return tables()

    return (mock.patch("decision_aware_tsfm.analysis.decision.fixture.make_fixture", fake_fixture),
            mock.patch("decision_aware_tsfm.analysis.decision.pipeline.build_tables", fake_build))


@lru_cache(maxsize=None)
def results(fit_scheme="leave_one_dataset_out"):
    return run(tables(), REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD, fit_scheme=fit_scheme)


def subset(tbl, datasets):
    """Copy of the tables restricted to some datasets."""
    out = dict(tbl)
    for k in FRAME_KEYS:
        out[k] = tbl[k][tbl[k]["dataset"].isin(datasets)].reset_index(drop=True)
    return out


def renamed(tbl, old, new, rename_series=True):
    """Copy of the tables with dataset `old` called `new`. With rename_series the
    series ids get the new prefix as well, so nothing of the old name is left."""
    out = dict(tbl)
    for k in FRAME_KEYS:
        f = tbl[k].copy()
        hit = f["dataset"] == old
        f.loc[hit, "dataset"] = new
        if rename_series:
            f.loc[hit, "series_id"] = f.loc[hit, "series_id"].str.replace(old, new, regex=False)
        out[k] = f
    return out
