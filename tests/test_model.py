import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sirtv.model import SIRTVParams, simulate
from sirtv.equilibria import r0, total_population_equilibrium
from sirtv.config import BASELINE, Y0


def test_total_population_converges_to_lam_over_mu():
    """N' = lam - mu*N implies N(t) -> lam/mu (paper Section 4.1)."""
    sol = simulate(BASELINE, Y0, t_span=(0, 2000))
    N_final = sol.y[:, -1].sum()
    N_eq = total_population_equilibrium(BASELINE)
    assert abs(N_final - N_eq) / N_eq < 0.01


def test_r0_below_one_disease_dies_out():
    p = SIRTVParams(**{**BASELINE.__dict__, "beta": 0.0005})
    assert r0(p) < 1
    sol = simulate(p, Y0, t_span=(0, 2000))
    assert sol.y[1, -1] < 1e-3  # infected -> 0


def test_r0_above_one_disease_persists():
    assert r0(BASELINE) > 1
    sol = simulate(BASELINE, Y0, t_span=(0, 2000))
    assert sol.y[1, -1] > 1  # infected settles at a positive endemic level


def test_populations_stay_nonnegative():
    sol = simulate(BASELINE, Y0, t_span=(0, 2000))
    assert np.all(sol.y >= -1e-6)


def test_dynamic_N_differs_from_hardcoded_N_bug():
    """
    Regression test for the original MATLAB bug: hardcoding N=1000 in the
    infection term gives materially different trajectories once N drifts
    away from 1000, which it does here since S+I+T+R+V -> lam/mu = 100000,
    not 1000.
    """
    sol = simulate(BASELINE, Y0, t_span=(0, 500))
    N_over_time = sol.y.sum(axis=0)
    # N should have moved substantially away from the old hardcoded value
    assert abs(N_over_time[-1] - 1000) > 500
