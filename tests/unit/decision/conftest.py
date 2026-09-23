import pathlib
import sys

import pytest

# lets `pytest` run from the repo root without PYTHONPATH=src
SRC = pathlib.Path(__file__).resolve().parents[3] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from decision_aware_tsfm.analysis.decision.fixture import (
    FAMILY_OF_MODEL, FIT_ONLY_DATASETS, FIXTURE_SCHEMA, FIXTURE_STATIC_THRESHOLD,
    REPRESENTATIVES, make_fixture)
from decision_aware_tsfm.analysis.decision.pipeline import build_tables, run



@pytest.fixture(scope="session")
def fixture_data():
    return make_fixture(seed=7)


@pytest.fixture(scope="session")
def tables(fixture_data):
    fc, omega, _ = fixture_data
    return build_tables(fc, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega,
                        fit_only_datasets=FIT_ONLY_DATASETS)


@pytest.fixture(scope="session")
def results(tables):
    res, d = run(tables, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD)
    return res, d
