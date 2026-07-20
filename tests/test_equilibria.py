import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sirtv.equilibria import dfe, jacobian, dfe_eigenvalues, is_dfe_stable, r0
from sirtv.model import SIRTVParams
from sirtv.config import BASELINE


def test_dfe_satisfies_steady_state():
    """Plugging Q0 into the RHS should give (approximately) zero derivative
    for S and V (I=T=R=0 trivially satisfy their equations)."""
    from sirtv.model import rhs
    Q0 = dfe(BASELINE)
    d = rhs(0, Q0, BASELINE)
    assert abs(d[0]) < 1e-6  # dS/dt ~ 0
    assert abs(d[4]) < 1e-6  # dV/dt ~ 0


def test_dfe_stability_matches_r0_threshold():
    """Paper Section 4.3: DFE is stable iff R0 < 1."""
    p_stable = SIRTVParams(**{**BASELINE.__dict__, "beta": 0.0005})
    p_unstable = BASELINE

    assert r0(p_stable) < 1
    assert is_dfe_stable(p_stable)

    assert r0(p_unstable) > 1
    assert not is_dfe_stable(p_unstable)


def test_jacobian_eigenvalue_matches_lambda2_formula():
    """
    Paper Section 4.3: lambda2 = (gamma1 + tau + mu) * (R0 - 1) should equal
    the "infection" eigenvalue of J(Q0).
    """
    Q0 = dfe(BASELINE)
    J = jacobian(Q0, BASELINE)
    eigs = np.linalg.eigvals(J)

    expected_lambda2 = (BASELINE.gamma1 + BASELINE.tau + BASELINE.mu) * (r0(BASELINE) - 1)
    assert np.any(np.abs(eigs.real - expected_lambda2) < 1e-6)


def test_four_eigenvalues_are_always_negative():
    """lambda1, lambda3, lambda4, lambda5 should always be negative
    regardless of R0 (paper Section 4.3)."""
    eigs = np.sort(dfe_eigenvalues(BASELINE).real)
    # at least 4 of the 5 eigenvalues must be negative regardless of R0
    assert np.sum(eigs < 0) >= 4
