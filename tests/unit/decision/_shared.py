"""Shared, cached fixture objects for the decision-engine tests.

Built once per test process, so the full fixture pipeline runs once.
Run with the package installed or with PYTHONPATH=src.
"""
from functools import lru_cache

from decision_aware_tsfm.analysis.decision.fixture import (
    FAMILY_OF_MODEL, FIT_ONLY_DATASETS, FIXTURE_SCHEMA, FIXTURE_STATIC_THRESHOLD,
    REPRESENTATIVES, make_fixture, make_fixture_scales)
from decision_aware_tsfm.analysis.decision.pipeline import build_tables, run


@lru_cache(maxsize=None)
def fixture_data():
    return make_fixture(seed=7)


@lru_cache(maxsize=None)
def scales():
    """The fixture's series scale table (history before the first origin, #29 item 7a)."""
    return make_fixture_scales(seed=7)


@lru_cache(maxsize=None)
def tables():
    fc, omega, _ = fixture_data()
    return build_tables(fc, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega,
                        fit_only_datasets=FIT_ONLY_DATASETS, scales=scales())


@lru_cache(maxsize=None)
def results():
    return run(tables(), REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD)


def small():
    """Horizon-24 rows of the exact-tie dataset (fresh copy each call)."""
    fc = fixture_data()[0]
    return fc[(fc.dataset == "fx_tie") & (fc.horizon == 24)].copy()
