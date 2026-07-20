from .model import SIRTVParams

# Baseline ("outbreak") parameters.
#
# NOTE (bug fix vs. the original MATLAB script): the script used
# beta = 0.005, likely tuned by trial-and-error against its own *buggy*
# disease-free equilibrium formula (see equilibria.py) rather than the
# paper's own stated "realistic range" for beta of 0.2-1.5 (Table 2). Once
# the correct DFE / R0 formula is used, beta = 0.005 never produces an
# outbreak (R0 stays far below 1) -- so the script's Figure 1 "R0 > 1" curve
# didn't actually correspond to the beta value it printed. beta is chosen
# here from within the paper's own stated range so that R0 > 1. All other
# parameters keep the original script's values.
BASELINE = SIRTVParams(
    lam=1000.0,
    beta=1.3,        # paper's stated range: 0.2-1.5 (was 0.005 in MATLAB)
    sigma=0.3,
    nu=0.1,
    gamma1=0.08,
    tau=0.02,
    gamma2=0.09,
    omegaR=0.06,
    omegaV=0.01,
    mu=0.01,
)

# A "controlled" scenario with stronger vaccination/social distancing,
# giving R0 < 1 (disease dies out), for contrast in the dashboard/report.
CONTROLLED = SIRTVParams(
    lam=1000.0,
    beta=1.3,
    sigma=0.6,
    nu=0.3,
    gamma1=0.08,
    tau=0.02,
    gamma2=0.09,
    omegaR=0.06,
    omegaV=0.01,
    mu=0.01,
)

Y0 = [200.0, 1.0, 0.0, 0.0, 0.0]
