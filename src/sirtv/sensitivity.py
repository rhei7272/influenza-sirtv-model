"""
PRCC (Partial Rank Correlation Coefficient) sensitivity analysis of R0.
"""

import numpy as np
from scipy.stats import rankdata
from .model import SIRTVParams
from .equilibria import r0


PARAM_NAMES = ["beta", "sigma", "nu", "gamma1", "tau", "mu"]


def sample_parameters(n=500, seed=1, lam=1000.0, gamma2=0.09, omegaR=0.06, omegaV=0.01):
    """
    Draw uniform samples for the six parameters PRCC is computed over.

    NOTE (bug fix vs. the original MATLAB code): the script sampled
    beta in [0.0003, 0.0007] -- three orders of magnitude below the paper's
    own stated realistic range for beta of 0.2-1.5 (Table 2), and well
    below the value needed for R0 > 1 (see config.py). Ranges below are
    taken directly from the paper's Table 2 instead.
    """
    rng = np.random.default_rng(seed)

    beta = rng.uniform(0.2, 1.5, n)
    sigma = rng.uniform(0.0, 1.0, n)
    nu = rng.uniform(0.01, 0.5, n)
    gamma1 = rng.uniform(0.1, 0.5, n)
    tau = rng.uniform(0.05, 0.3, n)
    mu = rng.uniform(0.01, 0.03, n)

    samples = np.column_stack([beta, sigma, nu, gamma1, tau, mu])
    return samples


def r0_for_samples(samples, lam=1000.0, gamma2=0.09, omegaR=0.06, omegaV=0.01):
    """
    Compute R0 for each row of `samples` using the model's actual closed-form
    R0 (equilibria.r0), not a simplified stand-in.

    NOTE (bug fix vs. the original MATLAB code):
    The MATLAB sensitivity section computed
        R0_vals = beta*(1-sigma) .* (lam ./ (nu+mu)) ./ (gamma1+tau+mu)
    i.e. it substitutes S0 ~= lam/(nu+mu) directly into the R0 formula. This
    is a *different, cruder* approximation than even the (already incorrect)
    S0 used earlier in the same script for Q0 -- so the main script's R0 and
    the "sensitivity analysis" R0 were computed two inconsistent ways in the
    same file. Both diverge further from the paper's actual closed-form R0
    (which uses S0/N0 with the correct DFE, see equilibria.dfe). Here every
    R0 value is computed with the single, correct closed-form expression so
    the PRCC ranks the parameters' influence on the model's actual threshold
    quantity, not an approximation of it.
    """
    r0_vals = np.empty(samples.shape[0])
    for i, row in enumerate(samples):
        beta, sigma, nu, gamma1, tau, mu = row
        p = SIRTVParams(lam=lam, beta=beta, sigma=sigma, nu=nu,
                         gamma1=gamma1, tau=tau, gamma2=gamma2,
                         omegaR=omegaR, omegaV=omegaV, mu=mu)
        r0_vals[i] = r0(p)
    return r0_vals


def compute_prcc(X, y):
    """
    Partial Rank Correlation Coefficients of each column of X against y.
    Direct Python translation of the MATLAB `compute_prcc` (rank + partial
    linear regression residual correlation) -- this part of the original
    code was correct and is preserved as-is.
    """
    n, k = X.shape
    XR = np.column_stack([rankdata(X[:, j]) for j in range(k)])
    yR = rankdata(y)

    prcc = np.zeros(k)
    for j in range(k):
        others = [c for c in range(k) if c != j]
        X_other = np.column_stack([np.ones(n), XR[:, others]])

        bx, *_ = np.linalg.lstsq(X_other, XR[:, j], rcond=None)
        rx = XR[:, j] - X_other @ bx

        by, *_ = np.linalg.lstsq(X_other, yR, rcond=None)
        ry = yR - X_other @ by

        prcc[j] = (rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry))
    return prcc


def run_sensitivity_analysis(n=500, seed=1):
    """Convenience wrapper: sample, evaluate R0, compute PRCC."""
    samples = sample_parameters(n=n, seed=seed)
    r0_vals = r0_for_samples(samples)
    prcc = compute_prcc(samples, r0_vals)
    return dict(zip(PARAM_NAMES, prcc))
