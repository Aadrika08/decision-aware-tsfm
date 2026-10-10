"""Shared, cached objects for the E3 tests (built once per test process)."""
from functools import lru_cache

from decision_aware_tsfm.analysis.experiments.e3 import design
from decision_aware_tsfm.analysis.experiments.e3.generator import make_dataset, match_table
from decision_aware_tsfm.analysis.experiments.e3.omega_only import sim_seeds, simulate
from decision_aware_tsfm.analysis.experiments.e3.population import breakeven_table
from decision_aware_tsfm.analysis.experiments.e3.scoring import expected_delta_table, score_dataset

SEEDS = (0, 1)
SCORED_MEMBERS = ("gaussian", "gaussian_b", "skewnorm_p8")
SCORED_STRUCTURE = "ar1_phi0.50"
OMEGA_SIM_SEEDS = 300
OMEGA_SIM_MEMBERS = ("gaussian", "gaussian_b", "student_t3")


@lru_cache(maxsize=None)
def dataset():
    """(values, series_table) for every structure and member, seeds 0 and 1."""
    return make_dataset(SEEDS)


@lru_cache(maxsize=None)
def matches():
    return match_table(dataset()[1])


def values_of(series_id):
    v = dataset()[0]
    return v.loc[v.series_id == series_id].sort_values("position")["value"].to_numpy()


@lru_cache(maxsize=None)
def scored_rows():
    s = dataset()[1]
    return s[(s.structure == SCORED_STRUCTURE) & (s.seed == 0) & s.member.isin(SCORED_MEMBERS)]


@lru_cache(maxsize=None)
def units_all():
    """Engine unit costs for one small group, every F variant."""
    return score_dataset(dataset()[0], scored_rows())


@lru_cache(maxsize=None)
def units():
    """Engine unit costs for one small group, primary F variant."""
    u = units_all()
    return u[u.variant == design.PRIMARY_VARIANT].reset_index(drop=True)


@lru_cache(maxsize=None)
def expected():
    """Population cost table (numerical integration; takes several seconds, so built once)."""
    return expected_delta_table()


@lru_cache(maxsize=None)
def breakeven():
    return breakeven_table()


@lru_cache(maxsize=None)
def omega_draws():
    """A small Omega-only simulation on the sine structure (300 fresh seeds, three members)."""
    return simulate(sim_seeds(OMEGA_SIM_SEEDS), structures=["sine_sd1.0"], members=OMEGA_SIM_MEMBERS)
