"""
Equilibria, R0, and stability analysis for the SIRTV model.

This module implements the closed-form results derived analytically in the
paper (disease-free equilibrium, R0, Jacobian, eigenvalues), rather than the
approximations used in the original MATLAB script (see NOTE below).
"""

import numpy as np
from .model import SIRTVParams


def dfe(p: SIRTVParams):
    """
    Disease-free equilibrium Q0 = (S0, 0, 0, 0, V0).

    Closed form (paper, Section 4.2):
        S0 = lam*(omegaV + mu) / (mu*(omegaV + mu + nu))
        V0 = nu*lam       / (mu*(omegaV + mu + nu))

    NOTE (bug fix vs. the original MATLAB code):
    The MATLAB script instead computed
        S0 = lam / (nu + mu)
        V0 = nu * S0 / (omegaV + mu)
    which omits the mu*(omegaV + mu + nu) denominator term and does not
    solve the actual S'=0, V'=0 system simultaneously -- it is only a valid
    equilibrium in the limiting case omegaV -> 0. Because R0 is evaluated
    at S0/N0, using the wrong S0 silently biases every R0 value the MATLAB
    script produces (and, downstream, the "sensitivity analysis" section,
    which uses yet a *third*, further simplified formula -- see
    sensitivity.py). Here we use the closed-form solution actually derived
    in the paper.
    """
    S0 = p.lam * (p.omegaV + p.mu) / (p.mu * (p.omegaV + p.mu + p.nu))
    V0 = p.nu * p.lam / (p.mu * (p.omegaV + p.mu + p.nu))
    return np.array([S0, 0.0, 0.0, 0.0, V0])


def total_population_equilibrium(p: SIRTVParams):
    """N* = lam/mu, the equilibrium of N' = lam - mu*N (paper, Section 4.1)."""
    return p.lam / p.mu


def r0(p: SIRTVParams):
    """
    Basic reproduction number (paper, Section 4):
        R0 = beta*(1-sigma) * (S0/N0) / (gamma1 + tau + mu)

    N0 here is the equilibrium total population lam/mu, consistent with the
    disease-free equilibrium above (S0 + V0 = lam/mu, since I=T=R=0 there).
    """
    Q0 = dfe(p)
    S0 = Q0[0]
    N0 = total_population_equilibrium(p)
    return p.beta * (1 - p.sigma) * (S0 / N0) / (p.gamma1 + p.tau + p.mu)


def jacobian(y, p: SIRTVParams):
    """
    Jacobian of the SIRTV system at state y = [S, I, T, R, V],
    matching the analytical Jacobian in the paper (Section 4.3).
    """
    S, I, T, R, V = y
    N = max(S + I + T + R + V, 1e-12)
    b = p.beta * (1 - p.sigma)

    dF_dS = b * (I / N - S * I / N**2)
    dF_dI = b * (S / N - S * I / N**2)
    dF_dT = -b * S * I / N**2
    dF_dR = -b * S * I / N**2
    dF_dV = -b * S * I / N**2

    J = np.array([
        [-dF_dS - p.nu - p.mu,              -dF_dI,                          -dF_dT,               -dF_dR + p.omegaR, -dF_dV + p.omegaV],
        [ dF_dS,                             dF_dI - (p.gamma1 + p.tau + p.mu), dF_dT,               dF_dR,             dF_dV],
        [ 0,                                 p.tau,                          -(p.gamma2 + p.mu),    0,                 0],
        [ 0,                                 p.gamma1,                        p.gamma2,             -(p.omegaR + p.mu), 0],
        [ p.nu,                              0,                               0,                    0,                 -(p.omegaV + p.mu)],
    ])
    return J


def dfe_eigenvalues(p: SIRTVParams):
    """Eigenvalues of J(Q0). Local stability of the DFE holds iff all < 0."""
    Q0 = dfe(p)
    return np.linalg.eigvals(jacobian(Q0, p))


def is_dfe_stable(p: SIRTVParams):
    return bool(np.all(np.real(dfe_eigenvalues(p)) < 0))
