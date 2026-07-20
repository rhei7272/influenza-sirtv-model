"""
SIRTV (Susceptible-Infected-Treated-Recovered-Vaccinated) epidemic model.

This is a Python port of an original MATLAB implementation, with one
important correctness fix applied (see NOTE below).
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class SIRTVParams:
    """Parameters for the SIRTV model."""
    lam: float      # recruitment (birth) rate
    beta: float     # transmission rate
    sigma: float    # social distancing effect, in [0, 1]
    nu: float       # vaccination rate
    gamma1: float   # recovery rate (untreated, "gamma" in the paper)
    tau: float      # treatment rate
    gamma2: float   # recovery rate under treatment ("gamma_T" in the paper)
    omegaR: float   # waning immunity rate, recovered -> susceptible
    omegaV: float   # waning immunity rate, vaccinated -> susceptible
    mu: float       # natural death rate


def rhs(t, y, p: SIRTVParams):
    """
    Right-hand side of the SIRTV ODE system.

    y = [S, I, T, R, V]

    NOTE (bug fix vs. the original MATLAB code):
    The MATLAB `model()` function hardcoded N = 1000 in the force-of-infection
    term (`N = 1000;`), regardless of the actual state. This is inconsistent
    with the model as derived in the paper, where N = S + I + T + R + V is a
    *dynamic* quantity satisfying N' = lam - mu*N (Section 4.1), and with the
    Jacobian (Section 4.3), which differentiates the infection term treating
    N as a function of the state. Hardcoding N silently breaks that
    consistency: R0 and the equilibria computed elsewhere in the paper assume
    N = lam/mu at the disease-free equilibrium, not a fixed constant. Here we
    use the true dynamic N.
    """
    S, I, T, R, V = y
    N = S + I + T + R + V
    infection = p.beta * (1 - p.sigma) * S * I / N

    dS = p.lam - infection - p.nu * S + p.omegaR * R + p.omegaV * V - p.mu * S
    dI = infection - (p.gamma1 + p.tau + p.mu) * I
    dT = p.tau * I - (p.gamma2 + p.mu) * T
    dR = p.gamma1 * I + p.gamma2 * T - (p.omegaR + p.mu) * R
    dV = p.nu * S - (p.omegaV + p.mu) * V

    return [dS, dI, dT, dR, dV]


def simulate(p: SIRTVParams, y0, t_span=(0, 300), t_eval=None, **kwargs):
    """
    Integrate the SIRTV model.

    Parameters
    ----------
    p : SIRTVParams
    y0 : array-like, initial [S, I, T, R, V]
    t_span : (t0, tf)
    t_eval : optional array of times to sample at

    Returns
    -------
    scipy.integrate.OdeResult
    """
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 600)
    return solve_ivp(
        rhs, t_span, y0, args=(p,), t_eval=t_eval,
        method="RK45", rtol=1e-8, atol=1e-8, **kwargs
    )
