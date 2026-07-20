"""
Interactive dashboard for the SIRTV influenza transmission model.

Run locally with:
    streamlit run dashboard/app.py

Deploy for free at https://streamlit.io/cloud by pointing it at this file
in your GitHub repo.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sirtv.model import SIRTVParams, simulate
from sirtv.equilibria import r0, total_population_equilibrium
from sirtv.config import BASELINE, Y0

st.set_page_config(page_title="SIRTV Influenza Model", layout="wide")

st.title("Influenza Transmission Model (SIRTV)")
st.markdown(
    "An interactive version of a 5-compartment (Susceptible-Infected-"
    "Treated-Recovered-Vaccinated) epidemic model. Adjust the intervention "
    "parameters on the left and watch the basic reproduction number "
    "**R0** and the epidemic curve update live."
)

st.sidebar.header("Intervention parameters")
beta = st.sidebar.slider("Transmission rate (β)", 0.2, 1.5, float(BASELINE.beta), 0.01)
sigma = st.sidebar.slider("Social distancing (σ)", 0.0, 1.0, float(BASELINE.sigma), 0.01)
nu = st.sidebar.slider("Vaccination rate (ν)", 0.01, 0.5, float(BASELINE.nu), 0.01)
tau = st.sidebar.slider("Treatment rate (τ)", 0.05, 0.3, float(BASELINE.tau), 0.01)

st.sidebar.header("Other parameters (fixed at paper defaults)")
gamma1 = st.sidebar.slider("Recovery rate, untreated (γ)", 0.1, 0.5, float(BASELINE.gamma1), 0.01)
mu = st.sidebar.slider("Natural death rate (μ)", 0.01, 0.03, float(BASELINE.mu), 0.001)

params = SIRTVParams(
    lam=BASELINE.lam, beta=beta, sigma=sigma, nu=nu, gamma1=gamma1,
    tau=tau, gamma2=BASELINE.gamma2, omegaR=BASELINE.omegaR,
    omegaV=BASELINE.omegaV, mu=mu,
)

R0_value = r0(params)

col1, col2, col3 = st.columns(3)
col1.metric("R0", f"{R0_value:.2f}", help="Values above 1 mean the disease can spread; below 1, it dies out.")
col2.metric("Outcome", "Outbreak" if R0_value > 1 else "Dies out")
col3.metric("Equilibrium population", f"{total_population_equilibrium(params):,.0f}")

sol = simulate(params, Y0, t_span=(0, 400))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].plot(sol.t, sol.y[1], color="crimson", linewidth=2)
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Infected")
axes[0].set_title(f"Infected population over time (R0 = {R0_value:.2f})")
axes[0].grid(True, alpha=0.3)

labels = ["S", "I", "T", "R", "V"]
colors = ["#1f77b4", "#d62728", "#e377c2", "#2ca02c", "#17becf"]
for i, (lab, c) in enumerate(zip(labels, colors)):
    axes[1].plot(sol.t, sol.y[i], color=c, label=lab)
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Population")
axes[1].set_title("All compartments")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

st.pyplot(fig)

st.markdown(
    "---\n"
    "**Try it:** push vaccination (ν) and social distancing (σ) up until "
    "R0 drops below 1 — notice how much less vaccination is needed than "
    "social distancing to hit the same R0 reduction, matching the paper's "
    "sensitivity analysis finding that ν has the strongest effect on R0."
)
