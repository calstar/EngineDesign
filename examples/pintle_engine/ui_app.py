"""Streamlit UI for the pintle engine pipeline."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import streamlit as st

from pintle_pipeline.io import load_config
from pintle_models.runner import PintleEngineRunner

PSI_TO_PA = 6894.76
PA_TO_PSI = 1.0 / PSI_TO_PA

CONFIG_PATH = Path(__file__).parent / "config_minimal.yaml"


@lru_cache(maxsize=1)
def get_runner() -> PintleEngineRunner:
    config = load_config(str(CONFIG_PATH))
    return PintleEngineRunner(config)


def summarize_results(results: Dict[str, Any]) -> None:
    Pc_psi = results["Pc"] * PA_TO_PSI
    thrust_kN = results["F"] / 1000.0
    mdot_total = results["mdot_total"]
    mdot_O = results["mdot_O"]
    mdot_F = results["mdot_F"]
    MR = results["MR"]
    Isp = results["Isp"]
    cstar = results["cstar_actual"]
    v_exit = results["v_exit"]
    P_exit_psi = results["P_exit"] * PA_TO_PSI

    st.metric("Thrust", f"{thrust_kN:.2f} kN")
    st.metric("Specific Impulse", f"{Isp:.1f} s")
    st.metric("Chamber Pressure", f"{Pc_psi:.1f} psi")
    st.metric("Total Mass Flow", f"{mdot_total:.3f} kg/s")
    st.metric("Oxidizer Flow", f"{mdot_O:.3f} kg/s")
    st.metric("Fuel Flow", f"{mdot_F:.3f} kg/s")
    st.metric("Mixture Ratio (O/F)", f"{MR:.3f}")
    st.metric("c* (actual)", f"{cstar:.1f} m/s")
    st.metric("Exit Velocity", f"{v_exit:.1f} m/s")
    st.metric("Exit Pressure", f"{P_exit_psi:.2f} psi")


def forward_view(runner: PintleEngineRunner) -> None:
    st.header("Forward Mode: Tank Pressures → Performance")

    col1, col2 = st.columns(2)
    with col1:
        P_tank_O_psi = st.slider(
            "LOX Tank Pressure [psi]",
            min_value=200.0,
            max_value=1500.0,
            value=1305.0,
            step=5.0,
        )
    with col2:
        P_tank_F_psi = st.slider(
            "Fuel Tank Pressure [psi]",
            min_value=200.0,
            max_value=1500.0,
            value=974.0,
            step=5.0,
        )

    if st.button("Compute Performance", type="primary"):
        try:
            results = runner.evaluate(
                P_tank_O_psi * PSI_TO_PA,
                P_tank_F_psi * PSI_TO_PA,
            )
            summarize_results(results)
        except Exception as exc:
            st.error(f"Pipeline evaluation failed: {exc}")


def _thrust_difference(
    scale: float,
    runner: PintleEngineRunner,
    base_pressures: Tuple[float, float],
    target_thrust_kN: float,
) -> float:
    P_tank_O_base, P_tank_F_base = base_pressures
    P_tank_O = scale * P_tank_O_base
    P_tank_F = scale * P_tank_F_base
    results = runner.evaluate(P_tank_O, P_tank_F)
    thrust_kN = results["F"] / 1000.0
    return thrust_kN - target_thrust_kN


def inverse_view(runner: PintleEngineRunner) -> None:
    from scipy.optimize import brentq

    st.header("Inverse Mode: Target Thrust → Tank Pressures")

    target_thrust_kN = st.number_input(
        "Desired Thrust [kN]",
        min_value=0.1,
        value=6.65,
        step=0.1,
    )

    col1, col2 = st.columns(2)
    with col1:
        base_O_psi = st.number_input(
            "Baseline LOX Tank Pressure [psi]",
            min_value=200.0,
            value=1305.0,
            step=10.0,
        )
    with col2:
        base_F_psi = st.number_input(
            "Baseline Fuel Tank Pressure [psi]",
            min_value=200.0,
            value=974.0,
            step=10.0,
        )

    if st.button("Solve for Tank Pressures", type="primary"):
        base_pressures_pa = (base_O_psi * PSI_TO_PA, base_F_psi * PSI_TO_PA)
        baseline_results = runner.evaluate(*base_pressures_pa)
        baseline_thrust = baseline_results["F"] / 1000.0

        low, high = (0.2, 5.0)
        f_low = _thrust_difference(low, runner, base_pressures_pa, target_thrust_kN)
        f_high = _thrust_difference(high, runner, base_pressures_pa, target_thrust_kN)

        iterations = 0
        max_expand = 10
        while f_low * f_high > 0 and iterations < max_expand:
            if target_thrust_kN > baseline_thrust:
                high *= 1.5
                f_high = _thrust_difference(high, runner, base_pressures_pa, target_thrust_kN)
            else:
                low *= 0.5
                f_low = _thrust_difference(low, runner, base_pressures_pa, target_thrust_kN)
            iterations += 1

        if f_low * f_high > 0:
            st.error("Could not bracket solution. Adjust target thrust or baseline pressures.")
            return

        try:
            scale = brentq(
                _thrust_difference,
                low,
                high,
                args=(runner, base_pressures_pa, target_thrust_kN),
                xtol=1e-4,
                rtol=1e-4,
                maxiter=100,
            )
        except Exception as exc:
            st.error(f"Failed to find tank pressures: {exc}")
            return

        P_tank_O_solution = scale * base_pressures_pa[0]
        P_tank_F_solution = scale * base_pressures_pa[1]
        results = runner.evaluate(P_tank_O_solution, P_tank_F_solution)

        st.subheader("Required Tank Pressures")
        st.metric("LOX Tank Pressure", f"{P_tank_O_solution * PA_TO_PSI:.1f} psi")
        st.metric("Fuel Tank Pressure", f"{P_tank_F_solution * PA_TO_PSI:.1f} psi")
        st.subheader("Performance at Solution")
        summarize_results(results)


def main():
    st.set_page_config(page_title="Pintle Engine Pipeline", layout="wide")
    st.title("Pintle Injector Engine Pipeline")
    st.caption(f"Configuration: {CONFIG_PATH}")

    runner = get_runner()

    tab1, tab2 = st.tabs(["Forward", "Inverse"])
    with tab1:
        forward_view(runner)
    with tab2:
        inverse_view(runner)


if __name__ == "__main__":
    main()
