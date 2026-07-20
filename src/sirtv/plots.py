import os
import numpy as np
import matplotlib.pyplot as plt

from .model import SIRTVParams, simulate
from .equilibria import r0, total_population_equilibrium
from .sensitivity import run_sensitivity_analysis, PARAM_NAMES
from .config import BASELINE, Y0

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "figures")


def _ensure_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def fig_threshold_behavior(save=True):
    """R0 < 1 (dies out) vs R0 > 1 (outbreak), matching paper Figure 1."""
    _ensure_dir()
    low_r0 = SIRTVParams(**{**BASELINE.__dict__, "beta": 0.001})  # forces R0 < 1
    high_r0 = BASELINE  # R0 > 1 at baseline

    sol_low = simulate(low_r0, Y0)
    sol_high = simulate(high_r0, Y0)

    plt.figure(figsize=(7, 5))
    plt.plot(sol_low.t, sol_low.y[1], label=f"R0={r0(low_r0):.2f} (dies out)", color="blue")
    plt.plot(sol_high.t, sol_high.y[1], label=f"R0={r0(high_r0):.2f} (outbreak)", color="red")
    plt.xlabel("Time")
    plt.ylabel("Infected")
    plt.title("Infected Population (R0 < 1 vs R0 > 1)")
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(os.path.join(FIG_DIR, "fig1_threshold.png"), dpi=150, bbox_inches="tight")
    return plt.gcf()


def fig_all_compartments(save=True):
    """S, I, T, R, V over time at baseline (R0 > 1), matching paper Figure 2."""
    _ensure_dir()
    sol = simulate(BASELINE, Y0)
    labels = ["S", "I", "T", "R", "V"]
    colors = ["b", "r", "m", "g", "c"]

    plt.figure(figsize=(7, 5))
    for i, (lab, c) in enumerate(zip(labels, colors)):
        plt.plot(sol.t, sol.y[i], color=c, label=lab)
    plt.xlabel("t")
    plt.ylabel("Population")
    plt.title(f"SIRTV Model Dynamics (R0={r0(BASELINE):.2f} > 1)")
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(os.path.join(FIG_DIR, "fig2_compartments.png"), dpi=150, bbox_inches="tight")
    return plt.gcf()


def fig_total_population(save=True):
    """N(t) -> lam/mu, matching paper Figure 3."""
    _ensure_dir()
    sol = simulate(BASELINE, Y0)
    N = sol.y.sum(axis=0)
    N_eq = total_population_equilibrium(BASELINE)

    plt.figure(figsize=(7, 5))
    plt.plot(sol.t, N, color="black", label="N(t)")
    plt.axhline(N_eq, color="gray", linestyle="--", label=f"lam/mu = {N_eq:.0f}")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("Total Population N(t)")
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(os.path.join(FIG_DIR, "fig3_total_population.png"), dpi=150, bbox_inches="tight")
    return plt.gcf()


def fig_prcc(save=True, n=500, seed=1):
    """PRCC bar chart, matching paper Figure 4 (with corrected R0 formula)."""
    _ensure_dir()
    prcc = run_sensitivity_analysis(n=n, seed=seed)

    plt.figure(figsize=(7, 5))
    plt.bar(PARAM_NAMES, [prcc[k] for k in PARAM_NAMES], color="#3399CC")
    plt.ylabel("PRCC")
    plt.title("PRCC Sensitivity Analysis of R0")
    plt.grid(True, axis="y")
    if save:
        plt.savefig(os.path.join(FIG_DIR, "fig4_prcc.png"), dpi=150, bbox_inches="tight")
    return plt.gcf()


def generate_all_figures():
    fig_threshold_behavior()
    fig_all_compartments()
    fig_total_population()
    fig_prcc()
    print(f"Figures saved to {os.path.abspath(FIG_DIR)}")


if __name__ == "__main__":
    generate_all_figures()
