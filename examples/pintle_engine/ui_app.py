# -*- coding: utf-8 -*-
"""Streamlit UI for the pintle engine pipeline."""

from __future__ import annotations

import math
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import sys
import warnings

import copy

# Suppress font warnings (corrupted font files on system)
warnings.filterwarnings("ignore", message=".*stringOffset.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="plotly")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml
from rocketpy import Function

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pintle_pipeline.io import load_config
from pintle_pipeline.config_schemas import PintleEngineConfig
from pintle_pipeline.time_series import generate_pressure_profile
from pintle_models.runner import PintleEngineRunner
from examples.pintle_engine.interactive_pipeline import solve_for_thrust, solve_for_thrust_and_MR, ThrustSolveError
from examples.pintle_engine.flight_sim import setup_flight, detect_tank_underfill_time
from examples.pintle_engine.copv_pressure.copv_solve_both import (
    size_or_check_copv_for_polytropic_N2,
)
# Import chamber geometry module
chamber_path = project_root / "chamber"
if str(chamber_path) not in sys.path:
    sys.path.insert(0, str(chamber_path))
from chamber_geometry import chamber_geometry_calc
# Import chamber geometry solver with CEA
try:
    from chamber.chamber_geometry_solver import solve_chamber_geometry_with_cea
except ImportError:
    # Fallback if import fails
    solve_chamber_geometry_with_cea = None

# Import chamber profiles for multi-layer visualization
try:
    from pintle_models.chamber_profiles import (
        calculate_complete_chamber_geometry,
        visualize_chamber_cross_section,
    )
except ImportError:
    calculate_complete_chamber_geometry = None
    visualize_chamber_cross_section = None
# Import graphite geometry sizing
from pintle_pipeline.graphite_geometry import size_graphite_insert
from pintle_pipeline.graphite_cooling import compute_graphite_recession, calculate_throat_recession_multiplier


# RocketPy imports (optional, only needed for flight sim)
try:
    from rocketpy import Function
    ROCKETPY_AVAILABLE = True
except ImportError:
    ROCKETPY_AVAILABLE = False
    Function = None

PSI_TO_PA = 6894.76
PA_TO_PSI = 1.0 / PSI_TO_PA

CONFIG_PATH = Path(__file__).parent / "config_minimal.yaml"

FLUID_LIBRARY: Dict[str, Dict[str, float]] = {
    "LOX": {
        "name": "LOX",
        "density": 1140.0,
        "viscosity": 1.8e-4,
        "surface_tension": 0.013,
        "vapor_pressure": 101325.0,
    },
    "LH2": {
        "name": "LH2",
        "density": 70.0,
        "viscosity": 1.3e-4,
        "surface_tension": 0.002,
        "vapor_pressure": 70000.0,
    },
    "RP-1": {
        "name": "RP-1",
        "density": 780.0,
        "viscosity": 2.0e-3,
        "surface_tension": 0.025,
        "vapor_pressure": 1000.0,
    },
    "Alcohol": {
        "name": "Ethanol",
        "density": 789.0,
        "viscosity": 1.2e-3,
        "surface_tension": 0.022,
        "vapor_pressure": 5800.0,
    },
}

OXIDIZER_OPTIONS = list(FLUID_LIBRARY.keys()) + ["Custom"]
FUEL_OPTIONS = list(FLUID_LIBRARY.keys()) + ["Custom"]

INJECTOR_OPTIONS = {
    "Pintle": {
        "type": "pintle",
        "supported": True,
        "description": "Axial LOX orifices with radial fuel annulus.",
    },
    "Coaxial": {
        "type": "coaxial",
        "supported": True,
        "description": "Shear coaxial: central oxidizer core with annular fuel (optional swirl).",
    },
    "Impinging": {
        "type": "impinging",
        "supported": True,
        "description": "Impinging element injector using opposing jets for atomization.",
    },
}

SENSOR_UNITS = {
    "pressure": {
        "psi": (PA_TO_PSI, "%.2f"),
        "kPa": (1e-3, "%.2f"),
        "MPa": (1e-6, "%.4f"),
    },
    "length": {
        "m": (1.0, "%.5f"),
        "mm": (1000.0, "%.3f"),
    },
    "mass_flow": {
        "kg/s": (1.0, "%.4f"),
        "g/s": (1000.0, "%.1f"),
    },
}

DEFAULT_GEOMETRIES = {
    "pintle": {
        "lox": {
            "n_orifices": 16,
            "d_orifice": 1.779e-3,
            "theta_orifice": 30.0,
            "A_entry": 1.0e-5,
            "d_hydraulic": 1.779e-3,
        },
        "fuel": {
            "d_pintle_tip": 0.01905,
            "d_reservoir_inner": 0.019676,
            "h_gap": 3.13e-4,
            "A_entry": 1.5e-5,
            "d_hydraulic": 6.26e-4,
        },
    },
    "coaxial": {
        "core": {
            "n_ports": 12,
            "d_port": 1.4e-3,
            "length": 0.015,
        },
        "annulus": {
            "inner_diameter": 5.0e-3,
            "gap_thickness": 8.0e-4,
            "swirl_angle": 20.0,
        },
    },
    "impinging": {
        "oxidizer": {
            "n_elements": 8,
            "d_jet": 1.2e-3,
            "impingement_angle": 60.0,
            "spacing": 4.0e-3,
        },
        "fuel": {
            "n_elements": 8,
            "d_jet": 1.2e-3,
            "impingement_angle": 60.0,
            "spacing": 4.0e-3,
        },
    },
}


@lru_cache(maxsize=1)
def load_default_runner() -> PintleEngineRunner:
    config = load_config(str(CONFIG_PATH))
    return PintleEngineRunner(config)


def get_default_config_dict() -> Dict[str, Any]:
    config = load_config(str(CONFIG_PATH))
    # Use exclude_none=False to preserve all fields including None values
    return config.model_dump(exclude_none=False)


def format_value(value: float, unit_type: str, unit_label: str) -> str:
    factor, fmt = SENSOR_UNITS[unit_type][unit_label]
    return fmt % (value * factor)


def to_elapsed_seconds(time_series: pd.Series) -> np.ndarray:
    """Convert a time column to elapsed seconds from the first value.
    Supports numeric, numeric-like strings, and datetimes (ISO, etc.)."""
    if pd.api.types.is_numeric_dtype(time_series):
        arr = np.asarray(time_series, dtype=float)
        t0 = float(arr[0])
        return (arr - t0).astype(float)
    numeric_coerced = pd.to_numeric(time_series, errors="coerce")
    if numeric_coerced.notna().mean() > 0.7:
        arr = numeric_coerced.fillna(method="ffill").to_numpy(dtype=float)
        t0 = float(arr[0])
        return (arr - t0).astype(float)
    try:
        dt_series = pd.to_datetime(time_series, errors="raise", utc=True)
    except Exception:
        dt_series = pd.to_datetime(time_series, errors="raise")
    t0 = dt_series.iloc[0]
    return (dt_series - t0).dt.total_seconds().to_numpy(dtype=float)


def build_rp_function(times_s: np.ndarray, values: np.ndarray, interpolation: str = "linear"):
    """Build RocketPy Function from time/value arrays with sorting and stacking."""
    if not ROCKETPY_AVAILABLE or Function is None:
        raise ImportError("RocketPy not available. Install with: pip install rocketpy")
    order = np.argsort(times_s)
    t_sorted = np.asarray(times_s, dtype=float)[order]
    v_sorted = np.asarray(values, dtype=float)[order]
    source = np.column_stack((t_sorted, v_sorted))
    return Function(source, interpolation=interpolation)


def _series_to_np(series_obj) -> np.ndarray:
    """RocketPy series to numpy, handling versions with/without get_source."""
    try:
        data = np.asarray(series_obj.get_source(), dtype=float)
        # If get_source returned (N, 2), we likely want the second column (values)
        # RocketPy Function source is typically [[x0, y0], [x1, y1], ...]
        if data.ndim == 2 and data.shape[1] >= 2:
            return data[:, 1]
        return data
    except Exception:
        return np.asarray(series_obj, dtype=float)


def _to_1d(arr_like) -> np.ndarray:
    arr = np.asarray(arr_like)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        squeezed = np.squeeze(arr)
        arr = squeezed if squeezed.ndim == 1 else np.ravel(arr)
    return arr.astype(float, copy=False)


def extract_flight_series(flight) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, altitude, and vertical velocity arrays aligned and 1-D."""
    t_series = _to_1d(_series_to_np(getattr(flight, "time", [])))
    z_series = _to_1d(_series_to_np(getattr(flight, "z", [])))
    vz_series = _to_1d(_series_to_np(getattr(flight, "vz", [])))
    n = int(min(len(t_series), len(z_series), len(vz_series)))
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    return t_series[:n], z_series[:n], vz_series[:n]


def plot_flight_results(t: np.ndarray, z: np.ndarray, vz: np.ndarray, *, key_suffix: str = "") -> None:
    """Plot altitude/velocity vs time if series are non-empty."""
    if t.size == 0:
        st.warning("Flight produced empty time series.")
        return
    df = pd.DataFrame({"time": t, "Altitude (m)": z, "Vertical Velocity (m/s)": vz})
    st.plotly_chart(px.line(df, x="time", y="Altitude (m)", title="Altitude vs Time"), width='stretch', key=f"flight_alt_plot{key_suffix}")
    st.plotly_chart(px.line(df, x="time", y="Vertical Velocity (m/s)", title="Vertical Velocity vs Time"), width='stretch', key=f"flight_vel_plot{key_suffix}")


def render_rocket_view(flight) -> None:
    """Render a static rocket view using RocketPy's draw."""
    try:
        import matplotlib.pyplot as plt  # local import to avoid global dependency if not needed
        # Clear previous figures to avoid overlay
        plt.close('all')
        result = getattr(flight, "rocket", None)
        if result is None:
            st.info("Rocket object not available for drawing.")
            return
        maybe_fig = result.draw()
        fig = maybe_fig if hasattr(maybe_fig, "savefig") else plt.gcf()
        st.pyplot(fig, clear_figure=True, width='stretch')
    except Exception as exc:
        st.info(f"Rocket view unavailable: {exc}")


def plot_additional_rocket_plots(flight, t_series: np.ndarray, *, key_suffix: str = "") -> None:
    """Try to render extra rocket/flight plots if the series exist on the Flight object."""
    if t_series.size == 0:
        return
    plots: list[tuple[str, str]] = []
    # Candidate attributes and labels
    candidates = [
        ("ax", "Axial Acceleration"),
        ("ay", "Lateral Acceleration Y"),
        ("az", "Lateral Acceleration Z"),
        ("alpha", "Angle of Attack"),
        ("beta", "Sideslip Angle"),
        ("mach_number", "Mach Number"),
    ]
    for attr, label in candidates:
        series_obj = getattr(flight, attr, None)
        if series_obj is not None:
            try:
                vals = _to_1d(_series_to_np(series_obj))
                if vals.size > 0:
                    plots.append((label, vals))
            except Exception:
                pass
    if not plots:
        st.info("No additional flight plots available.")
        return
    for label, vals in plots:
        n = min(len(t_series), len(vals))
        if n > 0:
            df = pd.DataFrame({"time": t_series[:n], label: vals[:n]})
            st.plotly_chart(px.line(df, x="time", y=label, title=label), width='stretch', key=f"flight_{label.replace(' ', '_').lower()}{key_suffix}")


# ============================================================================
# End Flight Simulation Helper Functions
# ============================================================================


def length_number_input(
    label: str,
    value_m: float,
    *,
    key: str,
    min_m: float = 0.0,
    max_m: float = 1.0,
    step_m: Optional[float] = None,
    allow_none: bool = False,
) -> Optional[float]:
    """Render a number_input in the configured length units and return meters."""

    unit = st.session_state.get("display_length_unit", "mm")
    factor, fmt = SENSOR_UNITS["length"][unit]
    display_label = f"{label} [{unit}]"

    value_m = 0.0 if value_m is None else float(value_m)
    min_display = float(min_m * factor)
    max_display = float(max_m * factor)
    value_display = float(value_m * factor)
    step_display = float(step_m * factor) if step_m is not None else None

    input_kwargs = {
        "min_value": min_display,
        "max_value": max_display,
        "value": value_display,
        "format": fmt,
        "key": key,
    }
    if step_display is not None:
        input_kwargs["step"] = step_display

    result_display = st.number_input(display_label, **input_kwargs)

    if allow_none and result_display <= 0:
        return None

    return float(result_display / factor)


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
    diagnostics = results.get("diagnostics", {})

    st.metric("Thrust", f"{thrust_kN:.2f} kN")
    st.metric("Specific Impulse", f"{Isp:.1f} s")
    pressure_unit = st.session_state.get("display_pressure_unit", "psi")
    mass_unit = st.session_state.get("display_mass_unit", "kg/s")

    st.metric("Chamber Pressure", format_value(results["Pc"], "pressure", pressure_unit) + f" {pressure_unit}")
    st.metric("Total Mass Flow", format_value(mdot_total, "mass_flow", mass_unit) + f" {mass_unit}")
    st.metric("Oxidizer Flow", format_value(mdot_O, "mass_flow", mass_unit) + f" {mass_unit}")
    st.metric("Fuel Flow", format_value(mdot_F, "mass_flow", mass_unit) + f" {mass_unit}")
    st.metric("Mixture Ratio (O/F)", f"{MR:.3f}")
    st.metric("c* (actual)", f"{cstar:.1f} m/s")
    st.metric("Exit Velocity", f"{v_exit:.1f} m/s")
    st.metric("Exit Pressure", format_value(results["P_exit"], "pressure", pressure_unit) + f" {pressure_unit}")
    
    # Thrust coefficient
    Cf_actual = results.get("Cf_actual", results.get("Cf", np.nan))
    Cf_ideal = results.get("Cf_ideal", np.nan)
    Cf_theoretical = results.get("Cf_theoretical", np.nan)
    if np.isfinite(Cf_actual):
        st.metric("Cf (actual)", f"{Cf_actual:.4f}")
        if np.isfinite(Cf_ideal):
            st.caption(f"Cf ideal: {Cf_ideal:.4f}")
        if np.isfinite(Cf_theoretical) and Cf_theoretical > 0:
            efficiency_pct = (Cf_actual / Cf_theoretical) * 100.0
            # Efficiency should be <= 100% (nozzle losses)
            efficiency_pct = min(efficiency_pct, 100.0)  # Cap at 100%
            st.caption(f"Cf theoretical: {Cf_theoretical:.4f} | Nozzle Efficiency: {efficiency_pct:.1f}%")
            if efficiency_pct > 100.0 or Cf_actual > Cf_ideal:
                st.warning("⚠️ Cf_actual > Cf_ideal may indicate calculation error (exit temp/velocity issue)")
    
    # Temperatures
    Tc = results.get("Tc", np.nan)
    T_throat = results.get("T_throat", np.nan)
    T_exit = results.get("T_exit", np.nan)
    if np.isfinite(Tc):
        st.metric("Chamber Temp", f"{Tc:.1f} K ({Tc-273.15:.1f} °C)")
    if np.isfinite(T_throat):
        st.metric("Throat Temp", f"{T_throat:.1f} K ({T_throat-273.15:.1f} °C)")
    if np.isfinite(T_exit):
        st.metric("Exit Temp", f"{T_exit:.1f} K ({T_exit-273.15:.1f} °C)")
    
    # Pressure and Temperature Profiles
    st.subheader("Chamber Profiles")
    
    # Pressure profile plot
    pressure_profile = results.get("pressure_profile")
    temp_profile = results.get("temperature_profile")
    
    if pressure_profile and isinstance(pressure_profile, dict):
        positions_p = pressure_profile.get("positions", [])
        pressures = pressure_profile.get("pressures", [])
        if positions_p and pressures and len(positions_p) == len(pressures):
            fig_pressure = go.Figure()
            fig_pressure.add_trace(go.Scatter(
                x=positions_p,
                y=np.array(pressures) * PA_TO_PSI,  # Convert to psi
                mode='lines+markers',
                name='Pressure',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            fig_pressure.update_layout(
                title="Pressure Profile Along Chamber",
                xaxis_title="Position from Injection [m]",
                yaxis_title="Pressure [psi]",
                height=300
            )
            st.plotly_chart(fig_pressure, width='stretch')
            
            # Show key pressures
            P_inj = pressure_profile.get("P_injection", np.nan) * PA_TO_PSI
            P_mid = pressure_profile.get("P_mid", np.nan) * PA_TO_PSI
            P_th = pressure_profile.get("P_throat", np.nan) * PA_TO_PSI
            if np.isfinite(P_inj):
                st.caption(f"P_injection: {P_inj:.1f} psi | P_mid: {P_mid:.1f} psi | P_throat: {P_th:.1f} psi")
    
    # Temperature profile plot
    if temp_profile and isinstance(temp_profile, dict):
        positions_t = temp_profile.get("positions", [])
        temperatures = temp_profile.get("temperatures", [])
        if positions_t and temperatures and len(positions_t) == len(temperatures):
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(
                x=positions_t,
                y=temperatures,
                mode='lines+markers',
                name='Temperature',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ))
            fig_temp.update_layout(
                title="Temperature Profile Along Chamber",
                xaxis_title="Position from Injection [m]",
                yaxis_title="Temperature [K]",
                height=300
            )
            st.plotly_chart(fig_temp, width='stretch')
            
            # Show key temperatures
            T_inj = temp_profile.get("T_injection", np.nan)
            T_mid = temp_profile.get("T_mid", np.nan)
            T_th = temp_profile.get("T_throat", T_throat)
            if np.isfinite(T_inj):
                st.caption(f"T_injection: {T_inj:.1f} K | T_mid: {T_mid:.1f} K | T_throat: {T_th:.1f} K")
    
    # Combined pressure-temperature profile
    if (pressure_profile and isinstance(pressure_profile, dict) and 
        temp_profile and isinstance(temp_profile, dict)):
        positions_p = pressure_profile.get("positions", [])
        positions_t = temp_profile.get("positions", [])
        pressures = pressure_profile.get("pressures", [])
        temperatures = temp_profile.get("temperatures", [])
        
        if (positions_p and positions_t and pressures and temperatures and
            len(positions_p) == len(pressures) and len(positions_t) == len(temperatures)):
            # Use common positions (interpolate if needed)
            if len(positions_p) == len(positions_t):
                fig_combined = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Pressure trace
                fig_combined.add_trace(
                    go.Scatter(
                        x=positions_p,
                        y=np.array(pressures) * PA_TO_PSI,
                        mode='lines+markers',
                        name='Pressure [psi]',
                        line=dict(color='blue', width=2),
                        marker=dict(size=6)
                    ),
                    secondary_y=False,
                )
                
                # Temperature trace
                fig_combined.add_trace(
                    go.Scatter(
                        x=positions_t,
                        y=temperatures,
                        mode='lines+markers',
                        name='Temperature [K]',
                        line=dict(color='red', width=2),
                        marker=dict(size=6)
                    ),
                    secondary_y=True,
                )
                
                fig_combined.update_xaxes(title_text="Position from Injection [m]")
                fig_combined.update_yaxes(title_text="Pressure [psi]", secondary_y=False)
                fig_combined.update_yaxes(title_text="Temperature [K]", secondary_y=True)
                fig_combined.update_layout(
                    title="Pressure and Temperature Profiles Along Chamber",
                    height=400
                )
                st.plotly_chart(fig_combined, width='stretch')
    
    # Injector Pressure Drop
    injector_pressure = results.get("injector_pressure")
    if injector_pressure and isinstance(injector_pressure, dict):
        st.subheader("Injector Pressure Drop")
        col1, col2, col3 = st.columns(3)
        with col1:
            P_inj_O = injector_pressure.get("P_injector_O")
            delta_p_O = injector_pressure.get("delta_p_injector_O")
            if P_inj_O is not None and np.isfinite(P_inj_O):
                st.metric("P_injector_O", f"{P_inj_O * PA_TO_PSI:.1f} psi")
            if delta_p_O is not None and np.isfinite(delta_p_O):
                st.metric("ΔP_injector_O", f"{delta_p_O * PA_TO_PSI:.1f} psi")
        with col2:
            P_inj_F = injector_pressure.get("P_injector_F")
            delta_p_F = injector_pressure.get("delta_p_injector_F")
            if P_inj_F is not None and np.isfinite(P_inj_F):
                st.metric("P_injector_F", f"{P_inj_F * PA_TO_PSI:.1f} psi")
            if delta_p_F is not None and np.isfinite(delta_p_F):
                st.metric("ΔP_injector_F", f"{delta_p_F * PA_TO_PSI:.1f} psi")
        with col3:
            delta_p_feed_O = injector_pressure.get("delta_p_feed_O")
            delta_p_feed_F = injector_pressure.get("delta_p_feed_F")
            if delta_p_feed_O is not None and np.isfinite(delta_p_feed_O):
                st.metric("ΔP_feed_O", f"{delta_p_feed_O * PA_TO_PSI:.1f} psi")
            if delta_p_feed_F is not None and np.isfinite(delta_p_feed_F):
                st.metric("ΔP_feed_F", f"{delta_p_feed_F * PA_TO_PSI:.1f} psi")
    
    # Chamber Intrinsics
    chamber_intrinsics = results.get("chamber_intrinsics")
    if chamber_intrinsics and isinstance(chamber_intrinsics, dict):
        st.subheader("Chamber Intrinsics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            Lstar = chamber_intrinsics.get("Lstar")
            if Lstar is not None and np.isfinite(Lstar):
                st.metric("L*", f"{Lstar*1000:.1f} mm")
            residence = chamber_intrinsics.get("residence_time")
            if residence is not None and np.isfinite(residence):
                st.metric("Residence Time", f"{residence*1000:.2f} ms")
        with col2:
            v_mean = chamber_intrinsics.get("velocity_mean")
            if v_mean is not None and np.isfinite(v_mean):
                st.metric("Mean Velocity", f"{v_mean:.1f} m/s")
            v_throat = chamber_intrinsics.get("velocity_throat")
            if v_throat is not None and np.isfinite(v_throat):
                st.metric("Throat Velocity", f"{v_throat:.0f} m/s")
        with col3:
            mach = chamber_intrinsics.get("mach_number")
            if mach is not None and np.isfinite(mach):
                st.metric("Mach Number", f"{mach:.3f}")
            Re = chamber_intrinsics.get("reynolds_number")
            if Re is not None and np.isfinite(Re):
                st.metric("Reynolds Number", f"{Re:.0f}")
        with col4:
            rho = chamber_intrinsics.get("density")
            if rho is not None and np.isfinite(rho):
                st.metric("Gas Density", f"{rho:.2f} kg/m³")
            sound = chamber_intrinsics.get("sound_speed")
            if sound is not None and np.isfinite(sound):
                st.metric("Sound Speed", f"{sound:.0f} m/s")

    eta_cstar = diagnostics.get("eta_cstar")
    if eta_cstar is not None:
        st.metric("η₍c*₎", f"{eta_cstar:.3f}")
        mixture_eff = diagnostics.get("mixture_efficiency")
        cooling_eff = diagnostics.get("cooling_efficiency")
        additional_rows = []
        if mixture_eff is not None:
            additional_rows.append(f"Mixture coupling: {mixture_eff:.3f}")
        if cooling_eff is not None:
            additional_rows.append(f"Cooling coupling: {cooling_eff:.3f}")
        turb_mix = diagnostics.get("turbulence_intensity_mix")
        if turb_mix is not None:
            additional_rows.append(f"Injector turbulence intensity: {turb_mix:.3f}")
        gas_turb = diagnostics.get("cooling", {}).get("metadata", {}).get("gas_turbulence_intensity")
        if gas_turb is not None:
            additional_rows.append(f"Chamber turbulence intensity: {gas_turb:.3f}")
        if additional_rows:
            st.caption(" | ".join(additional_rows))

    cooling = results.get("cooling", {})
    if cooling:
        st.subheader("Cooling Summary")
        regen = cooling.get("regen")
        if regen and regen.get("enabled", False):
            st.caption("Regenerative cooling")
            st.write(
                f"Coolant outlet temperature: {regen['coolant_outlet_temperature']:.1f} K"
            )
            st.write(
                f"Heat removed: {regen['heat_removed']/1000:.1f} kW | Hot-side heat flux: {regen['overall_heat_flux']/1000:.1f} kW/m²"
            )
            if "mdot_coolant" in regen:
                st.write(
                    f"Coolant flow through channels: {format_value(regen['mdot_coolant'], 'mass_flow', st.session_state.get('display_mass_unit', 'kg/s'))} {st.session_state.get('display_mass_unit', 'kg/s')}"
                )
            if "wall_temperature_coolant" in regen:
                st.write(
                    f"Wall temperature (hot/cool): {regen['wall_temperature_hot']:.1f} K / {regen['wall_temperature_coolant']:.1f} K"
                )
            if regen.get("film_effectiveness", 0.0) > 0:
                st.write(f"Film effectiveness contribution: {regen['film_effectiveness']:.2f}")
        film = cooling.get("film")
        if film and film.get("enabled", False):
            st.caption("Film cooling")
            st.write(
                f"Mass fraction: {film['mass_fraction']:.3f} | Effectiveness: {film['effectiveness']:.2f}"
            )
            st.write(
                f"Film mass flow: {film['mdot_film']:.3f} kg/s | Heat-flux factor: {film['heat_flux_factor']:.2f}"
            )
            if film.get("blowing_ratio") is not None:
                st.write(
                    f"Blowing ratio: {film['blowing_ratio']:.3f} | Turbulence multiplier: {film.get('turbulence_multiplier', np.nan):.2f}"
                )
        ablative = cooling.get("ablative")
        if ablative and ablative.get("enabled", False):
            st.caption("Ablative cooling")
            # Use cooling_power if available, fallback to heat_removed for backward compatibility
            cooling_power = ablative.get("cooling_power", ablative.get("heat_removed", 0.0))
            st.write(
                f"Recession rate: {ablative['recession_rate']*1e6:.3f} µm/s | Effective heat flux: {ablative['effective_heat_flux']/1000:.1f} kW/m² | Cooling power: {cooling_power/1000:.1f} kW"
            )
            if ablative.get("turbulence_multiplier") is not None:
                st.write(
                    f"Turbulence multiplier: {ablative['turbulence_multiplier']:.2f} | Incident heat flux: {ablative.get('incident_heat_flux', np.nan)/1000:.1f} kW/m²"
                )
            if ablative.get("below_pyrolysis") is not None:
                below_pyro = ablative.get("below_pyrolysis", False)
                st.write(
                    f"Status: {'Below pyrolysis (no ablation)' if below_pyro else 'Above pyrolysis (ablating)'}"
                )

        metadata = cooling.get("metadata", {})
        gas_turb = metadata.get("gas_turbulence_intensity") if isinstance(metadata, dict) else None
        if gas_turb is not None and np.isfinite(gas_turb):
            st.caption("Chamber turbulence")
            st.write(f"Gas-side turbulence intensity: {gas_turb:.3f}")

    # Graphite Insert Parameters
    graphite_cfg = None
    try:
        # Try to get config from default runner
        runner = load_default_runner()
        graphite_cfg = runner.config.graphite_insert if hasattr(runner.config, 'graphite_insert') and runner.config.graphite_insert else None
    except:
        pass
    
    if graphite_cfg and graphite_cfg.enabled:
        st.subheader("Graphite Throat Insert")
        st.caption("Material properties and configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Material Density", f"{graphite_cfg.material_density:.0f} kg/m³")
            st.metric("Heat of Ablation", f"{graphite_cfg.heat_of_ablation/1e6:.2f} MJ/kg")
            st.metric("Thermal Conductivity", f"{graphite_cfg.thermal_conductivity:.1f} W/(m·K)")
        with col2:
            st.metric("Specific Heat", f"{graphite_cfg.specific_heat:.0f} J/(kg·K)")
            st.metric("Initial Thickness", f"{graphite_cfg.initial_thickness*1000:.1f} mm")
            st.metric("Surface Temp Limit", f"{graphite_cfg.surface_temperature_limit:.0f} K")
        with col3:
            st.metric("Oxidation Temperature", f"{graphite_cfg.oxidation_temperature:.0f} K")
            st.metric("Oxidation Rate", f"{graphite_cfg.oxidation_rate*1e6:.3f} µm/s")
            st.metric("Coverage Fraction", f"{graphite_cfg.coverage_fraction*100:.0f}%")
        
        if graphite_cfg.recession_multiplier is not None:
            st.caption(f"Recession multiplier: {graphite_cfg.recession_multiplier:.2f}x (fixed)")
        else:
            st.caption("Recession multiplier: Calculated from Bartz correlation")
        
        if graphite_cfg.char_layer_thickness > 0:
            st.caption(f"Char layer: {graphite_cfg.char_layer_thickness*1000:.1f} mm thick, {graphite_cfg.char_layer_conductivity:.1f} W/(m·K) conductivity")

    # Stability Analysis
    stability = results.get("stability")
    if stability and isinstance(stability, dict):
        st.subheader("Stability Analysis")
        
        is_stable = stability.get("is_stable", False)
        status_color = "🟢" if is_stable else "🔴"
        st.markdown(f"**Overall Status:** {status_color} {'STABLE' if is_stable else 'INSTABILITY RISK'}")
        
        # Chugging analysis
        chugging = stability.get("chugging", {})
        if chugging:
            st.caption("**Combustion Stability - Chugging**")
            col1, col2, col3 = st.columns(3)
            with col1:
                freq = chugging.get("frequency", np.nan)
                if np.isfinite(freq):
                    st.metric("Frequency", f"{freq:.1f} Hz")
            with col2:
                damping = chugging.get("damping_ratio", np.nan)
                if np.isfinite(damping):
                    st.metric("Damping Ratio", f"{damping:.3f}")
            with col3:
                margin = chugging.get("stability_margin", np.nan)
                if np.isfinite(margin):
                    margin_color = "normal" if margin > 0 else "off"
                    st.metric("Stability Margin", f"{margin:.3f}", delta=None if margin > 0 else "⚠️ Risk")
        
        # Acoustic modes
        acoustic = stability.get("acoustic", {})
        if acoustic:
            st.caption("**Acoustic Modes**")
            long_modes = acoustic.get("longitudinal_modes", [])
            trans_modes = acoustic.get("transverse_modes", [])
            
            if long_modes and len(long_modes) > 0:
                st.write("Longitudinal modes:")
                modes_str = ", ".join([f"{f:.0f} Hz" for f in long_modes[:5] if np.isfinite(f)])
                st.caption(modes_str)
            
            if trans_modes and len(trans_modes) > 0:
                st.write("Transverse modes:")
                modes_str = ", ".join([f"{f:.0f} Hz" for f in trans_modes[:5] if np.isfinite(f)])
                st.caption(modes_str)
        
        # Feed system stability
        feed_sys = stability.get("feed_system", {})
        if feed_sys:
            st.caption("**Feed System Stability**")
            col1, col2, col3 = st.columns(3)
            with col1:
                pogo = feed_sys.get("pogo_frequency", np.nan)
                if np.isfinite(pogo):
                    st.metric("POGO Frequency", f"{pogo:.1f} Hz")
            with col2:
                surge = feed_sys.get("surge_frequency", np.nan)
                if np.isfinite(surge):
                    st.metric("Surge Frequency", f"{surge:.1f} Hz")
            with col3:
                margin_feed = feed_sys.get("stability_margin", np.nan)
                if np.isfinite(margin_feed):
                    margin_color = "normal" if margin_feed > 1.0 else "off"
                    st.metric("Feed Margin", f"{margin_feed:.2f}", delta=None if margin_feed > 1.0 else "⚠️ Risk")
        
        # Issues and recommendations
        issues = stability.get("issues", [])
        if issues:
            st.warning("**Potential Issues:**")
            for issue in issues:
                st.write(f"  • {issue}")
        
        recommendations = stability.get("recommendations", [])
        if recommendations:
            st.info("**Recommendations:**")
            for rec in recommendations:
                st.write(f"  → {rec}")

    # Burn Analysis (for time-series data)
    # Note: This would need to be called separately for time-series evaluations
    timeseries_results = st.session_state.get("timeseries_results")
    if timeseries_results and isinstance(timeseries_results, dict):
        if "F" in timeseries_results and "time" in timeseries_results or "times" in timeseries_results:
            st.subheader("Burn Analysis")
            try:
                from pintle_pipeline.burn_analysis import analyze_burn_degradation
                
                # Extract time history
                times = timeseries_results.get("times") or timeseries_results.get("time")
                if times is not None and len(times) > 1:
                    burn_results = analyze_burn_degradation(
                        np.asarray(times),
                        np.asarray(timeseries_results.get("F", [])),
                        np.asarray(timeseries_results.get("Pc", [])),
                        np.asarray(timeseries_results.get("Isp", [])),
                        np.asarray(timeseries_results.get("MR", [])),
                        np.asarray(timeseries_results.get("mdot_total", [])),
                        A_throat_history=timeseries_results.get("A_throat"),
                        recession_history=timeseries_results.get("recession_throat"),
                    )
                    
                    if "error" not in burn_results:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            duration = burn_results.get("burn_time", np.nan)
                            if np.isfinite(duration):
                                st.metric("Burn Duration", f"{duration:.2f} s")
                            impulse = burn_results.get("total_impulse", np.nan)
                            if np.isfinite(impulse):
                                st.metric("Total Impulse", f"{impulse/1000:.2f} kN·s")
                        with col2:
                            thrust_pct = burn_results.get("changes", {}).get("thrust_pct", np.nan)
                            if np.isfinite(thrust_pct):
                                st.metric("Thrust Change", f"{thrust_pct:.2f}%")
                            isp_pct = burn_results.get("changes", {}).get("Isp_pct", np.nan)
                            if np.isfinite(isp_pct):
                                st.metric("Isp Change", f"{isp_pct:.2f}%")
                        with col3:
                            pc_pct = burn_results.get("changes", {}).get("Pc_pct", np.nan)
                            if np.isfinite(pc_pct):
                                st.metric("Pc Change", f"{pc_pct:.2f}%")
                            throat_pct = burn_results.get("changes", {}).get("A_throat_pct", np.nan)
                            if np.isfinite(throat_pct):
                                st.metric("Throat Area Growth", f"{throat_pct:.2f}%")
                        
                        # Degradation rates
                        rates = burn_results.get("degradation_rates", {})
                        if rates:
                            st.caption("**Degradation Rates:**")
                            if "thrust" in rates and np.isfinite(rates["thrust"]):
                                st.write(f"Thrust: {rates['thrust']:.2f} N/s")
                            if "Pc" in rates and np.isfinite(rates["Pc"]):
                                st.write(f"Pc: {rates['Pc']/1000:.2f} kPa/s")
            except Exception as e:
                st.caption(f"Burn analysis unavailable: {e}")


def init_session_state() -> None:
    defaults = {
        "forward_result": None,
        "inverse_result": None,
        "timeseries_results": None,
        "custom_plot_datasets": {},
        "last_custom_dataset": None,
        "ui_previous_injector_type": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def store_dataset(label: str, df: pd.DataFrame) -> None:
    datasets: Dict[str, pd.DataFrame] = st.session_state.setdefault("custom_plot_datasets", {})
    base_label = label
    suffix = 2
    while label in datasets:
        label = f"{base_label} ({suffix})"
        suffix += 1
    datasets[label] = df.copy()
    st.session_state["last_custom_dataset"] = label


def estimate_pressurant_volume(config: PintleEngineConfig) -> Optional[float]:
    press = getattr(config, "press_tank", None)
    if press is None:
        return None
    for attr in ("press_volume", "volume_m3", "tank_volume_m3"):
        val = getattr(press, attr, None)
        if val is not None:
            return float(val)
    radius = getattr(press, "press_radius", None)
    height = getattr(press, "press_h", None)
    if radius is not None and height is not None:
        return float(math.pi * float(radius) ** 2 * float(height))
    return None


def infer_pressurant_R(config: PintleEngineConfig, default: float = 296.803) -> float:
    fluids = getattr(config, "fluids", None) or {}
    press_cfg = None
    if isinstance(fluids, dict):
        press_cfg = fluids.get("pressurant")
    else:
        try:
            press_cfg = fluids.get("pressurant")
        except Exception:
            press_cfg = None
    if press_cfg is not None:
        for attr in ("R", "R_specific"):
            val = getattr(press_cfg, attr, None)
            if val is not None:
                return float(val)
        molar_mass = getattr(press_cfg, "molar_mass", None)
        if molar_mass not in (None, 0):
            return float(8.31446261815324 / float(molar_mass))
    return float(default)


def create_single_run_dataframe(results: Dict[str, Any], context: str) -> pd.DataFrame:
    diag = results.get("diagnostics", {}) if isinstance(results, dict) else {}
    cooling = diag.get("cooling", {}) if isinstance(diag, dict) else {}
    regen = cooling.get("regen", {}) if isinstance(cooling, dict) else {}
    film = cooling.get("film", {}) if isinstance(cooling, dict) else {}
    ablative = cooling.get("ablative", {}) if isinstance(cooling, dict) else {}

    theta_rad = diag.get("theta") if isinstance(diag, dict) else None
    theta_deg = float(theta_rad * 180.0 / np.pi) if theta_rad is not None else np.nan

    row = {
        "Context": context,
        "Timestamp": datetime.now(timezone.utc).isoformat(),
        "Pc (psi)": results.get("Pc", np.nan) * PA_TO_PSI,
        "Thrust (kN)": results.get("F", np.nan) / 1000.0,
        "Isp (s)": results.get("Isp", np.nan),
        "MR": results.get("MR", np.nan),
        "mdot_O (kg/s)": results.get("mdot_O", np.nan),
        "mdot_F (kg/s)": results.get("mdot_F", np.nan),
        "mdot_total (kg/s)": results.get("mdot_total", np.nan),
        "cstar_actual (m/s)": results.get("cstar_actual", np.nan),
        "gamma": results.get("gamma", np.nan),
        "Spray Angle (deg)": theta_deg,
        "J": diag.get("J", np.nan),
        "We_O": diag.get("We_O", np.nan),
        "We_F": diag.get("We_F", np.nan),
        "D32_O (µm)": diag.get("D32_O", np.nan) * 1e6 if diag.get("D32_O") is not None else np.nan,
        "D32_F (µm)": diag.get("D32_F", np.nan) * 1e6 if diag.get("D32_F") is not None else np.nan,
        "Evaporation Length (mm)": diag.get("x_star", np.nan) * 1000.0 if diag.get("x_star") is not None else np.nan,
        "Mixture Efficiency": diag.get("mixture_efficiency", np.nan),
        "Cooling Efficiency": diag.get("cooling_efficiency", np.nan),
        "Regen Heat Removed (kW)": regen.get("heat_removed", np.nan) / 1000.0 if regen else np.nan,
        "Regen Outlet Temp (K)": regen.get("coolant_outlet_temperature", np.nan) if regen else np.nan,
        "Film Heat Removed (kW)": film.get("heat_removed", np.nan) / 1000.0 if film else np.nan,
        "Film Effectiveness": film.get("effectiveness", np.nan) if film else np.nan,
        "Ablative Cooling Power (kW)": ablative.get("cooling_power", ablative.get("heat_removed", np.nan)) / 1000.0 if ablative else np.nan,
        "Ablative Recession (µm/s)": ablative.get("recession_rate", np.nan) * 1e6 if ablative and ablative.get("recession_rate") is not None else np.nan,
        "Injector Turbulence O": diag.get("turbulence_intensity_O", np.nan),
        "Injector Turbulence F": diag.get("turbulence_intensity_F", np.nan),
        "Injector Turbulence Mix": diag.get("turbulence_intensity_mix", np.nan),
        "Film Turbulence Multiplier": film.get("turbulence_multiplier", np.nan) if film else np.nan,
        "Ablative Turbulence Multiplier": ablative.get("turbulence_multiplier", np.nan) if ablative else np.nan,
        "Cd_O": results.get("Cd_O", diag.get("Cd_O", np.nan)),
        "Cd_F": results.get("Cd_F", diag.get("Cd_F", np.nan)),
    }

    return pd.DataFrame([row])


def compute_timeseries_dataframe(
    runner: PintleEngineRunner,
    times: np.ndarray,
    P_tank_O_psi: np.ndarray,
    P_tank_F_psi: np.ndarray,
) -> Tuple[pd.DataFrame, list[str]]:
    P_tank_O_pa = np.asarray(P_tank_O_psi) * PSI_TO_PA
    P_tank_F_pa = np.asarray(P_tank_F_psi) * PSI_TO_PA

    # Check if ablative geometry tracking is enabled
    ablative_cfg = runner.config.ablative_cooling
    use_time_varying = (
        ablative_cfg is not None 
        and ablative_cfg.enabled 
        and ablative_cfg.track_geometry_evolution
        and len(times) >= 2
    )
    
    # Check if fully-coupled solver should be used
    use_coupled_solver = st.session_state.get("use_coupled_solver", True) if hasattr(st, 'session_state') else True
    
    if use_time_varying:
        # Use time-varying method for ablative geometry evolution
        # NEW: Use fully-coupled solver by default (integrates all systems)
        results = runner.evaluate_arrays_with_time(
            times, 
            P_tank_O_pa, 
            P_tank_F_pa,
            use_coupled_solver=use_coupled_solver,  # Fully-coupled solver
        )
    else:
        # Use standard method
        results = runner.evaluate_arrays(P_tank_O_pa, P_tank_F_pa)

    df_dict = {
        "time": np.asarray(times),
        "P_tank_O (psi)": np.asarray(P_tank_O_psi, dtype=float),
        "P_tank_F (psi)": np.asarray(P_tank_F_psi, dtype=float),
        "Pc (psi)": np.asarray(results["Pc"], dtype=float) * PA_TO_PSI,
        "Thrust (kN)": np.asarray(results["F"], dtype=float) / 1000.0,
        "Isp (s)": np.asarray(results["Isp"], dtype=float),
        "MR": np.asarray(results["MR"], dtype=float),
        "mdot_O (kg/s)": np.asarray(results["mdot_O"], dtype=float),
        "mdot_F (kg/s)": np.asarray(results["mdot_F"], dtype=float),
        "mdot_total (kg/s)": np.asarray(results["mdot_total"], dtype=float),
        "cstar_actual (m/s)": np.asarray(results["cstar_actual"], dtype=float),
        "gamma": np.asarray(results["gamma"], dtype=float),
        "Cd_O": np.asarray(results.get("Cd_O", np.full(len(times), np.nan)), dtype=float),
        "Cd_F": np.asarray(results.get("Cd_F", np.full(len(times), np.nan)), dtype=float),
    }

    regen_heat_flux = []
    regen_heat_removed = []
    regen_out_temp = []
    film_mass_flow = []
    film_effectiveness = []
    film_heat_removed = []
    film_heat_factor = []
    film_turbulence_multiplier = []
    ablative_recession = []
    ablative_heat_removed = []
    ablative_heat_flux = []
    injector_turbulence_mix = []
    
    # Graphite insert recession data
    graphite_recession_rate = []
    graphite_oxidation_rate = []
    graphite_ablation_rate = []
    graphite_surface_temp = []
    
    # Injector pressure diagnostics
    P_injector_O = []
    P_injector_F = []
    delta_p_injector_O = []
    delta_p_injector_F = []
    
    # Stability metrics
    chugging_frequency = []
    stability_margin = []
    damping_ratio = []
    pogo_frequency = []
    surge_frequency = []
    
    # Chamber intrinsics
    Lstar_list = []
    residence_time_list = []
    velocity_mean_list = []
    mach_number_list = []
    
    diag_list = results.get("diagnostics", [])
    errors: list[str] = []
    
    # Get graphite config for computing recession if not in diagnostics
    graphite_cfg = runner.config.graphite_insert if hasattr(runner, 'config') else None
    use_graphite = graphite_cfg is not None and graphite_cfg.enabled

    for i, diag in enumerate(diag_list):
        if isinstance(diag, dict) and "error" in diag:
            errors.append(str(diag["error"]))
        cooling = diag.get("cooling", {}) if isinstance(diag, dict) else {}
        regen = cooling.get("regen", {})
        film = cooling.get("film", {})
        ablative = cooling.get("ablative", {})

        regen_heat_flux.append(regen.get("overall_heat_flux", np.nan) / 1000.0 if regen else np.nan)
        regen_heat_removed.append(regen.get("heat_removed", np.nan) / 1000.0 if regen else np.nan)
        regen_out_temp.append(regen.get("coolant_outlet_temperature", np.nan) if regen else np.nan)
        film_mass_flow.append(film.get("mdot_film", np.nan) if film else np.nan)
        film_effectiveness.append(film.get("effectiveness", np.nan) if film else np.nan)
        film_heat_removed.append(film.get("heat_removed", np.nan) / 1000.0 if film else np.nan)
        film_heat_factor.append(film.get("heat_flux_factor", np.nan) if film else np.nan)
        film_turbulence_multiplier.append(film.get("turbulence_multiplier", np.nan) if film else np.nan)
        ablative_recession.append(ablative.get("recession_rate", np.nan) * 1e6 if ablative else np.nan)
        # Use cooling_power if available, fallback to heat_removed for backward compatibility
        ablative_cooling_power = ablative.get("cooling_power", ablative.get("heat_removed", np.nan)) if ablative else np.nan
        ablative_heat_removed.append(ablative_cooling_power / 1000.0 if not np.isnan(ablative_cooling_power) else np.nan)
        ablative_heat_flux.append(ablative.get("effective_heat_flux", np.nan) / 1000.0 if ablative else np.nan)
        injector_turbulence_mix.append(diag.get("turbulence_intensity_mix", np.nan) if isinstance(diag, dict) else np.nan)
        
        # Graphite insert recession data
        graphite = cooling.get("graphite", {}) if cooling else {}
        if graphite and graphite.get("enabled", False):
            # Extract from diagnostics if available
            graphite_recession_rate.append(graphite.get("recession_rate", np.nan) * 1e6)  # Convert to µm/s
            graphite_oxidation_rate.append(graphite.get("oxidation_rate", np.nan) * 1e6)  # Convert to µm/s
            graphite_ablation_rate.append(graphite.get("recession_rate_thermal", np.nan) * 1e6)  # Convert to µm/s
            graphite_surface_temp.append(graphite.get("surface_temperature", np.nan))
        elif use_graphite and i < len(results.get("Pc", [])):
            # Compute graphite recession if not in diagnostics but graphite is enabled
            try:
                from pintle_pipeline.graphite_cooling import compute_graphite_recession, calculate_throat_recession_multiplier
                
                Pc = float(results["Pc"][i]) if i < len(results["Pc"]) else 2e6
                Tc = float(results["Tc"][i]) if i < len(results["Tc"]) else 3000.0
                gamma = float(results["gamma"][i]) if i < len(results["gamma"]) else 1.2
                R = float(results["R"][i]) if i < len(results["R"]) else 300.0
                mdot_total = float(results["mdot_total"][i]) if i < len(results["mdot_total"]) else 0.1
                
                # Get throat area
                if "A_throat" in results:
                    throat_area = float(results["A_throat"][i])
                else:
                    throat_area = runner.config.chamber.A_throat if hasattr(runner.config, 'chamber') else 0.000857892
                
                # Estimate chamber heat flux
                if ablative and ablative.get("enabled", False):
                    chamber_heat_flux = ablative.get("incident_heat_flux", 1e6)
                else:
                    # Estimate from chamber conditions
                    chamber_heat_flux = 1e6 * (Pc / 2e6) ** 0.8
                
                # Calculate throat heat flux multiplier
                rho_chamber = Pc / (R * Tc) if R > 0 and Tc > 0 else 1.0
                D_throat = np.sqrt(4.0 * throat_area / np.pi) if throat_area > 0 else 0.033
                A_chamber = np.pi * (D_throat * 3) ** 2 / 4.0  # Approximate chamber area
                v_chamber = mdot_total / (rho_chamber * A_chamber) if rho_chamber > 0 and A_chamber > 0 else 50.0
                v_throat = np.sqrt(gamma * R * Tc / (gamma + 1)) if gamma > 0 and R > 0 and Tc > 0 else 1000.0
                
                throat_mult = calculate_throat_recession_multiplier(
                    Pc, v_chamber, v_throat, chamber_heat_flux, gamma
                )
                peak_heat_flux = chamber_heat_flux * throat_mult
                
                # Estimate surface temperature
                surface_temp = Tc * 0.85
                
                # Compute graphite recession
                graphite_results = compute_graphite_recession(
                    net_heat_flux=peak_heat_flux,
                    throat_temperature=surface_temp,
                    gas_temperature=Tc,
                    graphite_config=graphite_cfg,
                    throat_area=throat_area,
                    pressure=Pc,
                )
                
                graphite_recession_rate.append(graphite_results.get("recession_rate", 0.0) * 1e6)  # Convert to µm/s
                graphite_oxidation_rate.append(graphite_results.get("oxidation_rate", 0.0) * 1e6)  # Convert to µm/s
                graphite_ablation_rate.append(graphite_results.get("recession_rate_thermal", 0.0) * 1e6)  # Convert to µm/s
                graphite_surface_temp.append(graphite_results.get("surface_temperature", surface_temp))
            except Exception as e:
                # If computation fails, use NaN
                graphite_recession_rate.append(np.nan)
                graphite_oxidation_rate.append(np.nan)
                graphite_ablation_rate.append(np.nan)
                graphite_surface_temp.append(np.nan)
        else:
            # No graphite data available
            graphite_recession_rate.append(np.nan)
            graphite_oxidation_rate.append(np.nan)
            graphite_ablation_rate.append(np.nan)
            graphite_surface_temp.append(np.nan)
        
        # Injector pressure diagnostics
        P_injector_O.append(diag.get("P_injector_O", np.nan) if isinstance(diag, dict) else np.nan)
        P_injector_F.append(diag.get("P_injector_F", np.nan) if isinstance(diag, dict) else np.nan)
        delta_p_injector_O.append(diag.get("delta_p_injector_O", np.nan) if isinstance(diag, dict) else np.nan)
        delta_p_injector_F.append(diag.get("delta_p_injector_F", np.nan) if isinstance(diag, dict) else np.nan)

    if regen_heat_flux:
        df_dict["Regen Heat Flux (kW/m²)"] = np.asarray(regen_heat_flux, dtype=float)
        df_dict["Regen Heat Removed (kW)"] = np.asarray(regen_heat_removed, dtype=float)
        df_dict["Regen Coolant Outlet (K)"] = np.asarray(regen_out_temp, dtype=float)
    if film_mass_flow:
        df_dict["Film Mass Flow (kg/s)"] = np.asarray(film_mass_flow, dtype=float)
        df_dict["Film Effectiveness"] = np.asarray(film_effectiveness, dtype=float)
        df_dict["Film Heat Removed (kW)"] = np.asarray(film_heat_removed, dtype=float)
        df_dict["Film Heat Flux Factor"] = np.asarray(film_heat_factor, dtype=float)
        df_dict["Film Turbulence Multiplier"] = np.asarray(film_turbulence_multiplier, dtype=float)
        df_dict["Injector Turbulence Mix"] = np.asarray(injector_turbulence_mix, dtype=float)
    if ablative_recession:
        df_dict["Ablative Recession (µm/s)"] = np.asarray(ablative_recession, dtype=float)
        df_dict["Ablative Cooling Power (kW)"] = np.asarray(ablative_heat_removed, dtype=float)
        df_dict["Ablative Heat Flux (kW/m²)"] = np.asarray(ablative_heat_flux, dtype=float)
    
    # Add graphite insert recession data
    if graphite_recession_rate and any(np.isfinite(graphite_recession_rate)):
        df_dict["Graphite Recession Rate (µm/s)"] = np.asarray(graphite_recession_rate, dtype=float)
        df_dict["Graphite Oxidation Rate (µm/s)"] = np.asarray(graphite_oxidation_rate, dtype=float)
        df_dict["Graphite Ablation Rate (µm/s)"] = np.asarray(graphite_ablation_rate, dtype=float)
        df_dict["Graphite Surface Temperature (K)"] = np.asarray(graphite_surface_temp, dtype=float)
        
        # Calculate cumulative recession if time data is available
        times_array = np.asarray(times)
        if len(times_array) == len(graphite_recession_rate):
            dt = np.diff(times_array, prepend=times_array[0])
            cumulative_oxidation = np.cumsum(np.asarray(graphite_oxidation_rate, dtype=float) * dt * 1e-6)  # Convert µm/s to m/s, then integrate
            cumulative_ablation = np.cumsum(np.asarray(graphite_ablation_rate, dtype=float) * dt * 1e-6)  # Convert µm/s to m/s, then integrate
            cumulative_total = np.cumsum(np.asarray(graphite_recession_rate, dtype=float) * dt * 1e-6)  # Convert µm/s to m/s, then integrate
            
            df_dict["Cumulative Graphite Oxidation Recession (mm)"] = cumulative_oxidation * 1000.0
            df_dict["Cumulative Graphite Ablation Recession (mm)"] = cumulative_ablation * 1000.0
            df_dict["Cumulative Graphite Total Recession (mm)"] = cumulative_total * 1000.0

    # Add injector pressure drop data
    if P_injector_O and any(np.isfinite(P_injector_O)):
        df_dict["P_injector_O (psi)"] = np.asarray(P_injector_O, dtype=float) * PA_TO_PSI
        df_dict["P_injector_F (psi)"] = np.asarray(P_injector_F, dtype=float) * PA_TO_PSI
        df_dict["ΔP_injector_O (psi)"] = np.asarray(delta_p_injector_O, dtype=float) * PA_TO_PSI
        df_dict["ΔP_injector_F (psi)"] = np.asarray(delta_p_injector_F, dtype=float) * PA_TO_PSI
    
    # Add ablative geometry evolution data if available
    if "Lstar" in results:
        df_dict["L* (mm)"] = np.asarray(results["Lstar"], dtype=float) * 1000.0
        df_dict["Chamber Volume (cm³)"] = np.asarray(results["V_chamber"], dtype=float) * 1e6
        df_dict["Throat Area (mm²)"] = np.asarray(results["A_throat"], dtype=float) * 1e6
        df_dict["Cumulative Chamber Recession (µm)"] = np.asarray(results["recession_chamber"], dtype=float) * 1e6
        df_dict["Cumulative Throat Recession (µm)"] = np.asarray(results["recession_throat"], dtype=float) * 1e6
        if "recession_graphite" in results:
            df_dict["Cumulative Graphite Recession (µm)"] = np.asarray(results["recession_graphite"], dtype=float) * 1e6
        if "graphite_thickness_remaining" in results:
            df_dict["Graphite Thickness Remaining (mm)"] = np.asarray(results["graphite_thickness_remaining"], dtype=float) * 1000.0
        if "throat_recession_multiplier" in results:
            df_dict["Throat Recession Multiplier"] = np.asarray(results["throat_recession_multiplier"], dtype=float)
        
        # Add multi-layer thermal analysis metrics
        if "T_ablative_surface" in results:
            df_dict["Ablative Surface Temp (K)"] = np.asarray(results["T_ablative_surface"], dtype=float)
        if "T_stainless_chamber" in results:
            df_dict["Stainless Steel Temp - Chamber (K)"] = np.asarray(results["T_stainless_chamber"], dtype=float)
        if "T_graphite_surface" in results:
            df_dict["Graphite Surface Temp (K)"] = np.asarray(results["T_graphite_surface"], dtype=float)
        if "T_stainless_throat" in results:
            df_dict["Stainless Steel Temp - Throat (K)"] = np.asarray(results["T_stainless_throat"], dtype=float)
    
    # Extract stability metrics from diagnostics (if stored per point)
    # Note: For full time-series stability analysis, this would need to call comprehensive_stability_analysis
    # for each point, which can be expensive. For now, we check if it's already in diagnostics.
    if any(isinstance(d, dict) and "stability" in d for d in diag_list):
        for diag in diag_list:
            if isinstance(diag, dict):
                stability = diag.get("stability")
                if stability and isinstance(stability, dict):
                    chugging = stability.get("chugging", {})
                    feed_sys = stability.get("feed_system", {})
                    chugging_frequency.append(chugging.get("frequency", np.nan))
                    stability_margin.append(chugging.get("stability_margin", np.nan))
                    damping_ratio.append(chugging.get("damping_ratio", np.nan))
                    pogo_frequency.append(feed_sys.get("pogo_frequency", np.nan))
                    surge_frequency.append(feed_sys.get("surge_frequency", np.nan))
                else:
                    # Pad with NaN if not available for this point
                    chugging_frequency.append(np.nan)
                    stability_margin.append(np.nan)
                    damping_ratio.append(np.nan)
                    pogo_frequency.append(np.nan)
                    surge_frequency.append(np.nan)
        
        # Only add if we have data
        if chugging_frequency and any(np.isfinite(chugging_frequency)):
            df_dict["Chugging Frequency (Hz)"] = np.asarray(chugging_frequency, dtype=float)
            df_dict["Stability Margin"] = np.asarray(stability_margin, dtype=float)
            df_dict["Damping Ratio"] = np.asarray(damping_ratio, dtype=float)
        if pogo_frequency and any(np.isfinite(pogo_frequency)):
            df_dict["POGO Frequency (Hz)"] = np.asarray(pogo_frequency, dtype=float)
            df_dict["Surge Frequency (Hz)"] = np.asarray(surge_frequency, dtype=float)

    mixture_eff = [diag.get("mixture_efficiency", np.nan) if isinstance(diag, dict) else np.nan for diag in diag_list]
    cooling_eff = [diag.get("cooling_efficiency", np.nan) if isinstance(diag, dict) else np.nan for diag in diag_list]
    df_dict["Mixture Efficiency"] = np.asarray(mixture_eff, dtype=float)
    df_dict["Cooling Efficiency"] = np.asarray(cooling_eff, dtype=float)

    # Ensure all arrays have the same length
    n_expected = len(times)
    for key, arr in df_dict.items():
        arr_array = np.asarray(arr)
        if len(arr_array) != n_expected:
            if len(arr_array) < n_expected:
                # Pad with NaN
                padding = np.full(n_expected - len(arr_array), np.nan, dtype=arr_array.dtype)
                df_dict[key] = np.concatenate([arr_array, padding])
            else:
                # Truncate
                df_dict[key] = arr_array[:n_expected]

    df = pd.DataFrame(df_dict)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df, errors


def display_time_series_summary(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No time-series data available.")
        return

    duration = float(df["time"].iloc[-1] - df["time"].iloc[0])
    thrust_column = df["Thrust (kN)"]
    pc_column = df["Pc (psi)"]
    avg_thrust = float(thrust_column.mean())
    max_thrust = float(thrust_column.max())
    total_impulse = float(np.trapezoid(thrust_column * 1000.0, df["time"])) / 1000.0  # kN·s

    col1, col2, col3 = st.columns(3)
    col1.metric("Burn duration", f"{duration:.2f} s")
    col2.metric("Average thrust", f"{avg_thrust:.2f} kN")
    col3.metric("Total impulse", f"{total_impulse:.2f} kN·s")

    col4, col5 = st.columns(2)
    col4.metric("Peak thrust", f"{max_thrust:.2f} kN")
    col5.metric("Min chamber pressure", f"{pc_column.min():.1f} psi")


def plot_time_series_results(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No data to plot.")
        return

    thrust_fig = px.line(df, x="time", y="Thrust (kN)", markers=True, title="Thrust vs Time")
    st.plotly_chart(thrust_fig, width="stretch", key="ts_thrust")

    pc_fig = px.line(df, x="time", y="Pc (psi)", markers=True, title="Chamber Pressure vs Time")
    st.plotly_chart(pc_fig, width="stretch", key="ts_pc")

    mdot_fig = px.line(
        df,
        x="time",
        y=["mdot_O (kg/s)", "mdot_F (kg/s)", "mdot_total (kg/s)"],
        title="Mass Flow Rates",
    )
    st.plotly_chart(mdot_fig, width="stretch", key="ts_mdot")

    mr_fig = px.line(df, x="time", y="MR", markers=False, title="Mixture Ratio vs Time")
    st.plotly_chart(mr_fig, width="stretch", key="ts_mr")
    
    # Injector pressure drop plots
    if "ΔP_injector_O (psi)" in df.columns and not df["ΔP_injector_O (psi)"].isna().all():
        injector_p_fig = px.line(
            df,
            x="time",
            y=["ΔP_injector_O (psi)", "ΔP_injector_F (psi)"],
            markers=False,
            title="Injector Pressure Drop vs Time",
            labels={"value": "Pressure Drop [psi]", "variable": "Propellant"},
        )
        st.plotly_chart(injector_p_fig, width="stretch", key="ts_injector_p")
    
    if "P_injector_O (psi)" in df.columns and not df["P_injector_O (psi)"].isna().all():
        injector_p_abs_fig = px.line(
            df,
            x="time",
            y=["P_injector_O (psi)", "P_injector_F (psi)", "Pc (psi)"],
            markers=False,
            title="Injector and Chamber Pressures vs Time",
            labels={"value": "Pressure [psi]", "variable": "Location"},
        )
        st.plotly_chart(injector_p_abs_fig, width="stretch", key="ts_injector_p_abs")
    
    # Chamber intrinsics plots if available
    if "L* (mm)" in df.columns and not df["L* (mm)"].isna().all():
        lstar_fig = px.line(df, x="time", y="L* (mm)", markers=False, title="Characteristic Length (L*) vs Time")
        st.plotly_chart(lstar_fig, width="stretch", key="ts_lstar")
    
    # Stability metrics plots if available
    if "Chugging Frequency (Hz)" in df.columns and not df["Chugging Frequency (Hz)"].isna().all():
        stability_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Chugging Frequency", "Stability Margin", "Damping Ratio", "POGO/Surge Frequency"),
            vertical_spacing=0.15,
        )
        
        # Chugging frequency
        stability_fig.add_trace(
            go.Scatter(x=df["time"], y=df["Chugging Frequency (Hz)"], mode='lines+markers', name='Chugging', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Stability margin
        if "Stability Margin" in df.columns and not df["Stability Margin"].isna().all():
            stability_fig.add_trace(
                go.Scatter(x=df["time"], y=df["Stability Margin"], mode='lines+markers', name='Margin', line=dict(color='green')),
                row=1, col=2
            )
            # Add zero line
            stability_fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        # Damping ratio
        if "Damping Ratio" in df.columns and not df["Damping Ratio"].isna().all():
            stability_fig.add_trace(
                go.Scatter(x=df["time"], y=df["Damping Ratio"], mode='lines+markers', name='Damping', line=dict(color='orange')),
                row=2, col=1
            )
        
        # POGO/Surge frequency
        if "POGO Frequency (Hz)" in df.columns and not df["POGO Frequency (Hz)"].isna().all():
            stability_fig.add_trace(
                go.Scatter(x=df["time"], y=df["POGO Frequency (Hz)"], mode='lines+markers', name='POGO', line=dict(color='purple')),
                row=2, col=2
            )
        if "Surge Frequency (Hz)" in df.columns and not df["Surge Frequency (Hz)"].isna().all():
            stability_fig.add_trace(
                go.Scatter(x=df["time"], y=df["Surge Frequency (Hz)"], mode='lines+markers', name='Surge', line=dict(color='brown')),
                row=2, col=2
            )
        
        stability_fig.update_xaxes(title_text="Time [s]", row=2, col=1)
        stability_fig.update_xaxes(title_text="Time [s]", row=2, col=2)
        stability_fig.update_yaxes(title_text="Frequency [Hz]", row=1, col=1)
        stability_fig.update_yaxes(title_text="Margin", row=1, col=2)
        stability_fig.update_yaxes(title_text="Ratio", row=2, col=1)
        stability_fig.update_yaxes(title_text="Frequency [Hz]", row=2, col=2)
        stability_fig.update_layout(title="Stability Metrics vs Time", height=600, showlegend=True)
        st.plotly_chart(stability_fig, width="stretch", key="ts_stability")

    if "Regen Heat Flux (kW/m²)" in df.columns and not df["Regen Heat Flux (kW/m²)"].isna().all():
        regen_fig = px.line(
            df,
            x="time",
            y="Regen Heat Flux (kW/m²)",
            markers=False,
            title="Regenerative Cooling Heat Flux",
        )
        st.plotly_chart(regen_fig, width="stretch", key="ts_regen_flux")

    if "Film Mass Flow (kg/s)" in df.columns and not df["Film Mass Flow (kg/s)"].isna().all():
        film_fig = px.line(
            df,
            x="time",
            y="Film Mass Flow (kg/s)",
            markers=False,
            title="Film Cooling Mass Flow",
        )
        st.plotly_chart(film_fig, width="stretch", key="ts_film")

    if "Film Heat Removed (kW)" in df.columns and not df["Film Heat Removed (kW)"].isna().all():
        film_heat_fig = px.line(
            df,
            x="time",
            y="Film Heat Removed (kW)",
            markers=False,
            title="Film Cooling Heat Removal",
        )
        st.plotly_chart(film_heat_fig, width="stretch", key="ts_film_heat")

    if "Film Heat Flux Factor" in df.columns and not df["Film Heat Flux Factor"].isna().all():
        film_factor_fig = px.line(
            df,
            x="time",
            y="Film Heat Flux Factor",
            markers=False,
            title="Film Heat Flux Reduction Factor",
        )
        st.plotly_chart(film_factor_fig, width="stretch", key="ts_film_factor")

    if "Ablative Recession (µm/s)" in df.columns and not df["Ablative Recession (µm/s)"].isna().all():
        abl_fig = px.line(
            df,
            x="time",
            y="Ablative Recession (µm/s)",
            markers=False,
            title="Ablative Recession Rate",
        )
        st.plotly_chart(abl_fig, width="stretch", key="ts_ablative")

    if "Ablative Cooling Power (kW)" in df.columns and not df["Ablative Cooling Power (kW)"].isna().all():
        abl_heat_fig = px.line(
            df,
            x="time",
            y="Ablative Cooling Power (kW)",
            markers=False,
            title="Ablative Cooling Power",
        )
        st.plotly_chart(abl_heat_fig, width="stretch", key="ts_abl_heat")

    if "Injector Turbulence Mix" in df.columns and not df["Injector Turbulence Mix"].isna().all():
        turb_fig = px.line(
            df,
            x="time",
            y="Injector Turbulence Mix",
            markers=False,
            title="Injector Mixed Turbulence Intensity",
        )
        st.plotly_chart(turb_fig, width="stretch", key="ts_turb_mix")
    
    # Ablative geometry evolution plots
    if "L* (mm)" in df.columns and not df["L* (mm)"].isna().all():
        st.subheader("🔥 Ablative Geometry Evolution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            lstar_fig = px.line(
                df,
                x="time",
                y="L* (mm)",
                markers=False,
                title="Characteristic Length (L*) Evolution",
            )
            lstar_fig.update_layout(yaxis_title="L* [mm]", xaxis_title="Time [s]")
            st.plotly_chart(lstar_fig, width='stretch', key="ts_lstar_evol")
        
        with col2:
            recession_fig = go.Figure()
            recession_fig.add_trace(go.Scatter(
                x=df["time"],
                y=df["Cumulative Chamber Recession (µm)"],
                name="Chamber",
                mode="lines",
                line=dict(color="purple", width=2),
            ))
            recession_fig.add_trace(go.Scatter(
                x=df["time"],
                y=df["Cumulative Throat Recession (µm)"],
                name="Throat",
                mode="lines",
                line=dict(color="orange", width=2),
            ))
            recession_fig.update_layout(
                title="Cumulative Ablative Recession",
                xaxis_title="Time [s]",
                yaxis_title="Recession [µm]",
                legend=dict(x=0.02, y=0.98),
            )
            st.plotly_chart(recession_fig, width='stretch', key="ts_recession_cumul")
        
        col3, col4 = st.columns(2)
        
        with col3:
            geom_fig = go.Figure()
            V_pct_change = (df["Chamber Volume (cm³)"] / df["Chamber Volume (cm³)"].iloc[0] - 1) * 100
            A_pct_change = (df["Throat Area (mm²)"] / df["Throat Area (mm²)"].iloc[0] - 1) * 100
            geom_fig.add_trace(go.Scatter(
                x=df["time"],
                y=V_pct_change,
                name="Chamber Volume",
                mode="lines",
                line=dict(color="red", width=2),
            ))
            geom_fig.add_trace(go.Scatter(
                x=df["time"],
                y=A_pct_change,
                name="Throat Area",
                mode="lines",
                line=dict(color="blue", width=2),
            ))
            geom_fig.update_layout(
                title="Geometry Growth",
                xaxis_title="Time [s]",
                yaxis_title="Change [%]",
                legend=dict(x=0.02, y=0.98),
            )
            st.plotly_chart(geom_fig, width='stretch', key="ts_geom_growth_pct")
        
        with col4:
            if "Throat Recession Multiplier" in df.columns and not df["Throat Recession Multiplier"].isna().all():
                mult_fig = px.line(
                    df,
                    x="time",
                    y="Throat Recession Multiplier",
                    markers=False,
                    title="Throat Recession Multiplier (Physics-Based)",
                )
                mult_fig.update_layout(yaxis_title="Multiplier", xaxis_title="Time [s]")
                mult_mean = df["Throat Recession Multiplier"].mean()
                mult_fig.add_hline(
                    y=mult_mean,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"Mean: {mult_mean:.2f}",
                )
                st.plotly_chart(mult_fig, width='stretch', key="ts_throat_mult_phys")
        
        # Performance degradation summary
        if len(df) > 1:
            thrust_initial = df["Thrust (kN)"].iloc[0]
            thrust_final = df["Thrust (kN)"].iloc[-1]
            thrust_loss_pct = (thrust_final / thrust_initial - 1) * 100
            
            lstar_initial = df["L* (mm)"].iloc[0]
            lstar_final = df["L* (mm)"].iloc[-1]
            lstar_change_pct = (lstar_final / lstar_initial - 1) * 100
            
            st.info(
                f"**Ablative Geometry Impact:**\n\n"
                f"- L* increased by **{lstar_change_pct:+.2f}%** ({lstar_initial:.2f} → {lstar_final:.2f} mm)\n"
                f"- Throat area grew by **{A_pct_change.iloc[-1]:+.3f}%**\n"
                f"- Thrust degraded by **{thrust_loss_pct:+.2f}%** ({thrust_initial:.3f} → {thrust_final:.3f} kN)\n"
                f"- Total chamber recession: **{df['Cumulative Chamber Recession (µm)'].iloc[-1]:.1f} µm**\n"
                f"- Total throat recession: **{df['Cumulative Throat Recession (µm)'].iloc[-1]:.1f} µm**"
            )


def custom_plot_builder() -> None:
    st.header("Custom Plot Builder")

    datasets: Dict[str, pd.DataFrame] = st.session_state.get("custom_plot_datasets", {})
    if not datasets:
        st.info("Run a forward or time-series analysis to populate datasets for plotting.")
        return

    dataset_names = list(datasets.keys())
    default_dataset = st.session_state.get("last_custom_dataset")
    if default_dataset not in dataset_names:
        default_index = 0
    else:
        default_index = dataset_names.index(default_dataset)

    dataset_name = st.selectbox("Select dataset", dataset_names, index=default_index)
    df = datasets[dataset_name]

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        st.warning("Selected dataset has no numeric columns to plot.")
        return

    x_col_default = "time" if "time" in numeric_columns else numeric_columns[0]
    x_col = st.selectbox("X-axis", numeric_columns, index=numeric_columns.index(x_col_default))

    plot_type = st.selectbox("Plot type", ["Line", "Scatter", "Heatmap", "Contour"])

    if plot_type in {"Heatmap", "Contour"}:
        heatmap_x = st.selectbox("X-axis", numeric_columns, index=numeric_columns.index(x_col_default), key="heatmap_x_axis")
        y_candidates = [col for col in numeric_columns if col != heatmap_x]
        if not y_candidates:
            st.warning("Select a dataset with at least two numeric columns for heatmap/contour plots.")
            return
        heatmap_y = st.selectbox("Y-axis", y_candidates, index=0, key="heatmap_y_axis")
        z_candidates = [col for col in numeric_columns if col not in {heatmap_x, heatmap_y}]
        if not z_candidates:
            st.warning("No remaining numeric columns available for the value axis. Choose different X/Y axes.")
            return
        heatmap_z = st.selectbox("Value (Z) axis", z_candidates, index=0, key="heatmap_z_axis")
        agg_choice = st.selectbox("Aggregation", ["mean", "median", "sum", "min", "max"], key="heatmap_agg")
        agg_map = {
            "mean": np.mean,
            "median": np.median,
            "sum": np.sum,
            "min": np.min,
            "max": np.max,
        }
        pivot = df.pivot_table(index=heatmap_y, columns=heatmap_x, values=heatmap_z, aggfunc=agg_map[agg_choice])
        pivot = pivot.replace([np.inf, -np.inf], np.nan)
        if pivot.isna().all().all():
            st.warning("Pivot table contains only NaN values; adjust axis selections or aggregation.")
            return
        pivot = pivot.sort_index().sort_index(axis=1)

        if plot_type == "Heatmap":
            fig = px.imshow(
                pivot,
                x=pivot.columns,
                y=pivot.index,
                color_continuous_scale="Viridis",
                aspect="auto",
                origin="lower",
                labels=dict(x=heatmap_x, y=heatmap_y, color=heatmap_z),
                title=f"{heatmap_z} heatmap ({agg_choice})",
            )
        else:
            fig = go.Figure(
                data=go.Contour(
                    x=pivot.columns,
                    y=pivot.index,
                    z=pivot.values,
                    contours_coloring="heatmap",
                    colorbar=dict(title=heatmap_z),
                )
            )
            fig.update_layout(
                title=f"{heatmap_z} contour ({agg_choice})",
                xaxis_title=heatmap_x,
                yaxis_title=heatmap_y,
            )

        st.plotly_chart(fig, width="stretch", key="custom_plot_builder")

    else:
        y_col_default = [col for col in numeric_columns if col != x_col][:1] or [x_col]
        y_cols = st.multiselect("Y-axis", numeric_columns, default=y_col_default)
        if not y_cols:
            st.warning("Select at least one Y-axis variable.")
            return

        # Secondary axis options
        use_secondary = False
        primary_y_cols = y_cols
        secondary_y_cols = []
        
        if len(y_cols) > 1:
            use_secondary = st.checkbox("Use secondary Y-axis", value=False, help="Assign some variables to a secondary Y-axis")
            if use_secondary:
                primary_y_cols = st.multiselect(
                    "Primary Y-axis variables", 
                    y_cols, 
                    default=y_cols[:1],
                    help="Variables plotted on the left Y-axis"
                )
                secondary_y_cols = [col for col in y_cols if col not in primary_y_cols]
                if not primary_y_cols:
                    st.warning("Select at least one variable for the primary Y-axis.")
                    return
                if not secondary_y_cols:
                    st.warning("Select at least one variable for the secondary Y-axis, or uncheck 'Use secondary Y-axis'.")
                    return

        color_options = ["None"] + [col for col in numeric_columns if col not in {x_col, *y_cols}]
        color_by = st.selectbox("Color by (optional)", color_options)

        if color_by != "None" and len(y_cols) > 1:
            st.warning("When using a color grouping, select a single Y-axis variable.")
            return

        # Scaling options
        st.subheader("Axis Scaling")
        col1, col2 = st.columns(2)
        with col1:
            x_scale = st.selectbox("X-axis scale", ["linear", "log", "symlog"], index=0)
        with col2:
            y_scale = st.selectbox("Y-axis scale", ["linear", "log", "symlog"], index=0)
            if use_secondary:
                y2_scale = st.selectbox("Secondary Y-axis scale", ["linear", "log", "symlog"], index=0)
        
        show_markers = st.checkbox("Show markers", value=(plot_type == "Scatter"))

        # Create figure with or without secondary axis
        if use_secondary:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add primary Y-axis traces
            for y_col in primary_y_cols:
                if plot_type == "Line":
                    trace = go.Scatter(
                        x=df[x_col],
                        y=df[y_col],
                        name=y_col,
                        mode="lines" + ("+markers" if show_markers else ""),
                    )
                else:  # Scatter
                    trace = go.Scatter(
                        x=df[x_col],
                        y=df[y_col],
                        name=y_col,
                        mode="markers",
                        marker=dict(size=8),
                    )
                fig.add_trace(trace, secondary_y=False)
            
            # Add secondary Y-axis traces
            for y_col in secondary_y_cols:
                if plot_type == "Line":
                    trace = go.Scatter(
                        x=df[x_col],
                        y=df[y_col],
                        name=y_col,
                        mode="lines" + ("+markers" if show_markers else ""),
                        line=dict(dash="dash"),
                    )
                else:  # Scatter
                    trace = go.Scatter(
                        x=df[x_col],
                        y=df[y_col],
                        name=y_col,
                        mode="markers",
                        marker=dict(size=8, symbol="diamond"),
                    )
                fig.add_trace(trace, secondary_y=True)
            
            # Update axis labels and scales
            primary_label = ", ".join(primary_y_cols) if len(primary_y_cols) <= 2 else f"{len(primary_y_cols)} variables"
            secondary_label = ", ".join(secondary_y_cols) if len(secondary_y_cols) <= 2 else f"{len(secondary_y_cols)} variables"
            
            fig.update_xaxes(title_text=x_col, type=x_scale if x_scale != "symlog" else "log")
            fig.update_yaxes(title_text=primary_label, secondary_y=False, type=y_scale if y_scale != "symlog" else "log")
            fig.update_yaxes(title_text=secondary_label, secondary_y=True, type=y2_scale if y2_scale != "symlog" else "log")
            
            fig.update_layout(
                title="Multi-axis plot",
                hovermode="x unified",
            )
            
        else:
            # Single axis plot
            if plot_type == "Line":
                if len(y_cols) == 1:
                    fig = px.line(
                        df,
                        x=x_col,
                        y=y_cols[0],
                        color=None if color_by == "None" else color_by,
                        markers=show_markers,
                        title=f"{y_cols[0]} vs {x_col}",
                    )
                else:
                    melted = df.melt(id_vars=x_col, value_vars=y_cols, var_name="Series", value_name="Value")
                    fig = px.line(
                        melted,
                        x=x_col,
                        y="Value",
                        color="Series",
                        markers=show_markers,
                        title="Multi-series plot",
                    )
            else:  # Scatter
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_cols[0],
                    color=None if color_by == "None" else color_by,
                    title=f"{y_cols[0]} vs {x_col}",
                )
                if show_markers:
                    fig.update_traces(marker=dict(size=8))
            
            # Apply scaling
            if x_scale == "log":
                fig.update_xaxes(type="log")
            elif x_scale == "symlog":
                fig.update_xaxes(type="log")
            
            if y_scale == "log":
                fig.update_yaxes(type="log")
            elif y_scale == "symlog":
                fig.update_yaxes(type="log")

        st.plotly_chart(fig, width="stretch", key="custom_plot_builder")

    with st.expander("Dataset preview"):
        st.dataframe(df)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download dataset as CSV",
            data=csv_bytes,
            file_name=f"{dataset_name.replace(' ', '_').lower()}_data.csv",
            mime="text/csv",
        )


def copv_view(config_obj: PintleEngineConfig) -> None:
    st.header("COPV Sizing & Verification (Shared N₂ Tank)")
    st.write(
        "Select a time-series dataset with mass-flow and tank-pressure traces, then size or check a shared "
        "pressurant COPV that feeds both oxidizer and fuel tanks."
    )

    datasets: Dict[str, pd.DataFrame] = st.session_state.get("custom_plot_datasets", {})
    required_cols = {"time", "mdot_O (kg/s)", "mdot_F (kg/s)", "P_tank_O (psi)", "P_tank_F (psi)"}
    eligible = {name: df for name, df in datasets.items() if required_cols.issubset(df.columns)}
    if not eligible:
        st.info(
            "No eligible datasets found. Run a time-series evaluation (generated or uploaded) so the "
            "resulting dataframe includes time, mdot_O, mdot_F, and tank pressure columns."
        )
        return

    dataset_names = list(eligible.keys())
    default_dataset = st.session_state.get("last_custom_dataset")
    default_index = dataset_names.index(default_dataset) if default_dataset in dataset_names else 0

    default_volume_m3 = estimate_pressurant_volume(config_obj) or 0.02
    default_volume_l = max(default_volume_m3 * 1000.0, 0.1)
    default_R = infer_pressurant_R(config_obj)

    fluids_cfg = getattr(config_obj, "fluids", None) or {}

    def _fluid_temp(name: str, fallback: float) -> float:
        fluid = None
        if isinstance(fluids_cfg, dict):
            fluid = fluids_cfg.get(name)
        else:
            try:
                fluid = fluids_cfg.get(name)
            except Exception:
                fluid = None
        temp_val = getattr(fluid, "temperature", None) if fluid is not None else None
        return float(temp_val) if temp_val is not None else float(fallback)

    ox_temp_default = _fluid_temp("oxidizer", 300.0)
    fuel_temp_default = _fluid_temp("fuel", 300.0)

    df_selected: Optional[pd.DataFrame] = None
    dataset_name = dataset_names[default_index]
    with st.form("copv_sizing_form"):
        dataset_name = st.selectbox(
            "Dataset (must include time, mdot, and tank pressure columns)",
            dataset_names,
            index=default_index,
        )
        df_selected = eligible[dataset_name].copy().sort_values("time").reset_index(drop=True)
        if df_selected.empty:
            st.warning("Selected dataset is empty. Choose a different dataset.")
        else:
            duration = float(df_selected["time"].iloc[-1] - df_selected["time"].iloc[0])
            st.caption(f"{len(df_selected)} samples | duration ≈ {duration:.2f} s")

        sizing_mode = st.radio(
            "Sizing mode",
            ["Known COPV volume → solve for initial pressure", "Known initial pressure → solve for COPV volume"],
            horizontal=False,
        )

        copv_volume_m3: Optional[float]
        copv_P0_Pa: Optional[float]

        if sizing_mode.startswith("Known COPV volume"):
            copv_volume_l = st.number_input(
                "COPV free volume [L]",
                min_value=0.01,
                value=float(default_volume_l),
                step=0.1,
            )
            copv_volume_m3 = copv_volume_l / 1000.0
            copv_P0_Pa = None
        else:
            tank_pressures = df_selected[["P_tank_O (psi)", "P_tank_F (psi)"]].to_numpy(dtype=float)
            finite_pressures = tank_pressures[np.isfinite(tank_pressures)]
            if finite_pressures.size == 0:
                max_required_psi = 1000.0
            else:
                max_required_psi = float(finite_pressures.max())
            default_P0_psi = max(50.0, max_required_psi * 1.15)
            copv_P0_psi = st.number_input(
                "Initial COPV pressure [psi]",
                min_value=50.0,
                value=float(default_P0_psi),
                step=10.0,
            )
            copv_P0_Pa = copv_P0_psi * PSI_TO_PA
            copv_volume_m3 = None

        col_params = st.columns(3)
        with col_params[0]:
            n = st.number_input("Polytropic exponent n", min_value=1.0, max_value=1.5, value=1.2, step=0.01)
        with col_params[1]:
            T0_K = st.number_input("Initial COPV temperature [K]", min_value=50.0, value=300.0, step=1.0)
        with col_params[2]:
            R_pressurant = st.number_input(
                "Pressurant gas constant [J/(kg·K)]",
                min_value=10.0,
                value=float(default_R),
                step=1.0,
            )

        branch_temp_cols = st.columns(2)
        with branch_temp_cols[0]:
            ox_temp_K = st.number_input(
                "Oxidizer tank gas temperature [K]",
                min_value=30.0,
                value=float(ox_temp_default),
                step=1.0,
            )
        with branch_temp_cols[1]:
            fuel_temp_K = st.number_input(
                "Fuel tank gas temperature [K]",
                min_value=30.0,
                value=float(fuel_temp_default),
                step=1.0,
            )
        Tp_K = 0.5 * (ox_temp_K + fuel_temp_K)

        use_real_gas = st.checkbox("Use real-gas Z lookup for N₂", value=True)
        default_z_table = Path(__file__).parent / "copv_pressure" / "n2_Z_lookup.csv"
        z_lookup_path = st.text_input("N₂ Z-table CSV", value=str(default_z_table))

        run_btn = st.form_submit_button("Run COPV sizing/check")

    if not run_btn:
        return

    if df_selected is None or df_selected.empty:
        st.error("Cannot run COPV sizing without a populated dataset.")
        return

    try:
        solver_results = size_or_check_copv_for_polytropic_N2(
            df=df_selected,
            config=config_obj,
            n=n,
            T0_K=T0_K,
            Tp_K=Tp_K,
            use_real_gas=use_real_gas,
            n2_Z_csv=z_lookup_path,
            pressurant_R=R_pressurant,
            branch_temperatures_K={
                "oxidizer": ox_temp_K,
                "fuel": fuel_temp_K,
            },
            copv_volume_m3=copv_volume_m3,
            copv_P0_Pa=copv_P0_Pa,
        )
    except Exception as exc:
        st.error(f"Failed to run COPV sizing: {exc}")
        return

    st.success("COPV computation complete.")

    metrics_cols = st.columns(4)
    with metrics_cols[0]:
        st.metric("Initial COPV pressure", f"{solver_results['P0_Pa'] * PA_TO_PSI:.1f} psi")
    with metrics_cols[1]:
        st.metric("COPV free volume", f"{solver_results['copv_volume_m3'] * 1000.0:.2f} L")
    with metrics_cols[2]:
        st.metric("Initial N₂ mass", f"{solver_results['m0_kg']:.3f} kg")
    with metrics_cols[3]:
        st.metric("Total delivered N₂", f"{solver_results['total_delivered_mass_kg']:.3f} kg")
    st.metric(
        "Minimum pressure margin (all tanks)",
        f"{solver_results['min_margin_Pa'] * PA_TO_PSI:.1f} psi",
    )

    branch_margins = solver_results.get("branch_min_margins_Pa", {})
    if branch_margins:
        st.subheader("Per-branch minimum margins")
        cols = st.columns(len(branch_margins))
        for col, (name, margin_Pa) in zip(cols, branch_margins.items()):
            col.metric(f"{name.capitalize()} margin", f"{margin_Pa * PA_TO_PSI:.1f} psi")

    time_vals = np.asarray(solver_results["time_s"], dtype=float)
    copv_pressure = np.asarray(solver_results["PH_trace_Pa"], dtype=float) * PA_TO_PSI
    pressure_fig = make_subplots(specs=[[{"secondary_y": True}]])
    pressure_fig.add_trace(
        go.Scatter(
            x=time_vals,
            y=copv_pressure,
            name="COPV pressure",
            line=dict(color="green"),
        ),
        secondary_y=False,
    )
    for name, branch in solver_results["branches"].items():
        branch_pressure = np.asarray(branch["P_tank_Pa"], dtype=float) * PA_TO_PSI
        pressure_fig.add_trace(
            go.Scatter(
                x=time_vals,
                y=branch_pressure,
                name=f"{name.capitalize()} tank",
                line=dict(dash="dot"),
            ),
            secondary_y=True,
        )
    pressure_fig.update_yaxes(title_text="COPV pressure [psi]", secondary_y=False)
    pressure_fig.update_yaxes(title_text="Tank pressure [psi]", secondary_y=True, showgrid=False)
    pressure_fig.update_layout(xaxis_title="Time [s]")
    st.plotly_chart(pressure_fig, width='stretch')

    mass_fig = go.Figure()
    mass_fig.add_trace(
        go.Scatter(
            x=time_vals,
            y=np.asarray(solver_results["combined_M_delivered_kg"], dtype=float),
            name="Combined delivered mass",
        )
    )
    for name, branch in solver_results["branches"].items():
        mass_fig.add_trace(
            go.Scatter(
                x=time_vals,
                y=np.asarray(branch["M_delivered_kg"], dtype=float),
                name=f"{name.capitalize()} delivered mass",
                line=dict(dash="dash"),
            )
        )
    mass_fig.update_layout(xaxis_title="Time [s]", yaxis_title="Delivered gas mass [kg]")
    st.plotly_chart(mass_fig, width='stretch')

    branch_rows = []
    for name, branch in solver_results["branches"].items():
        branch_rows.append(
            {
                "Branch": name.capitalize(),
                "Tank volume [L]": branch["tank_volume_m3"] * 1000.0,
                "Initial ullage [L]": branch["Vg0_m3"] * 1000.0,
                "Initial propellant mass [kg]": branch["initial_mass_kg"],
                "Gas temperature [K]": branch.get("gas_temperature_K", np.nan),
                "Peak tank pressure [psi]": np.max(branch["P_tank_Pa"]) * PA_TO_PSI,
                "Delivered gas mass [kg]": branch["M_delivered_kg"][-1],
            }
        )
    if branch_rows:
        summary_df = pd.DataFrame(branch_rows).set_index("Branch")
        st.subheader("Branch summary")
        st.dataframe(summary_df)

    augmented_df = df_selected.copy()
    augmented_df["COPV_pressure_Pa"] = np.asarray(solver_results["PH_trace_Pa"], dtype=float)
    augmented_df["COPV_pressure_psi"] = copv_pressure
    augmented_df["COPV_mass_delivered_kg"] = np.asarray(solver_results["combined_M_delivered_kg"], dtype=float)
    dataset_store: Dict[str, pd.DataFrame] = st.session_state.setdefault("custom_plot_datasets", {})
    dataset_store[dataset_name] = augmented_df
    st.session_state["last_custom_dataset"] = dataset_name

    export_df = pd.DataFrame(
        {
            "time_s": time_vals,
            "COPV_pressure_Pa": np.asarray(solver_results["PH_trace_Pa"], dtype=float),
            "COPV_pressure_psi": copv_pressure,
            "combined_gas_mass_kg": np.asarray(solver_results["combined_M_delivered_kg"], dtype=float),
        }
    )
    for name, branch in solver_results["branches"].items():
        export_df[f"P_tank_{name}_Pa"] = np.asarray(branch["P_tank_Pa"], dtype=float)
        export_df[f"P_tank_{name}_psi"] = export_df[f"P_tank_{name}_Pa"] * PA_TO_PSI
        export_df[f"M_delivered_{name}_kg"] = np.asarray(branch["M_delivered_kg"], dtype=float)
        export_df[f"m_g_required_{name}_kg"] = np.asarray(branch["m_g_req_kg"], dtype=float)

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download COPV traces (CSV)",
        data=csv_bytes,
        file_name="copv_sizing_results.csv",
        mime="text/csv",
    )



def plots_analysis_view(runner: PintleEngineRunner) -> None:
    if isinstance(runner, PintleEngineConfig):
        runner = PintleEngineRunner(runner)
    st.header("Plots & Analysis")
    st.write(
        "Run forward or time-series analyses, then use the controls below to explore correlations "
        "and summaries. For bespoke plots, visit the Custom Plot Builder tab."
    )

    datasets: Dict[str, pd.DataFrame] = st.session_state.get("custom_plot_datasets", {})
    if not datasets:
        st.info("No datasets available yet. Execute a forward or time-series run to populate this tab.")
        return

    dataset_names = list(datasets.keys())
    default_dataset = st.session_state.get("last_custom_dataset")
    if default_dataset not in dataset_names:
        default_index = 0
    else:
        default_index = dataset_names.index(default_dataset)

    dataset_name = st.selectbox("Dataset", dataset_names, index=default_index, key="analysis_dataset_select")
    df = datasets[dataset_name]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.info("Selected dataset does not have enough numeric columns for advanced analysis.")
    else:
        col_metrics = st.columns(3)
        if "Thrust (kN)" in df.columns:
            col_metrics[0].metric("Avg Thrust", f"{df['Thrust (kN)'].mean():.2f} kN")
            col_metrics[1].metric("Peak Thrust", f"{df['Thrust (kN)'].max():.2f} kN")
        if "Pc (psi)" in df.columns:
            col_metrics[2].metric("Avg Pc", f"{df['Pc (psi)'].mean():.1f} psi")

        if "time" in df.columns and {"Thrust (kN)", "Pc (psi)"}.issubset(df.columns):
            time_fig = px.line(
                df,
                x="time",
                y=["Thrust (kN)", "Pc (psi)"],
                markers=False,
                title="Thrust & Chamber Pressure vs Time",
            )
            st.plotly_chart(time_fig, width="stretch", key="analysis_timeplot")

        max_dims = min(len(numeric_cols), 6)
        analysis_columns = numeric_cols[:max_dims]
        sample_df = df[analysis_columns].dropna().copy()
        if len(sample_df) > 500:
            sample_df = sample_df.sample(500, random_state=0)
        scatter_fig = px.scatter_matrix(
            sample_df,
            dimensions=analysis_columns,
            title=f"Scatter matrix (first {max_dims} numeric columns)",
        )
        st.plotly_chart(scatter_fig, width="stretch", key="analysis_scatter_matrix")

        corr = df[numeric_cols].corr().fillna(0.0)
        corr_fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu",
            origin="lower",
            title="Correlation heatmap",
        )
        st.plotly_chart(corr_fig, width="stretch", key="analysis_corr_heatmap")

        st.subheader("Quick scatter/line plot")
        quick_cols = [col for col in numeric_cols]
        x_choice = st.selectbox("X-axis", quick_cols, index=0, key="analysis_x_axis")
        y_choices = [col for col in quick_cols if col != x_choice]
        if not y_choices:
            st.warning("Not enough numeric columns for plotting.")
        else:
            y_choice = st.selectbox("Y-axis", y_choices, key="analysis_y_axis")
            chart_type = st.radio("Chart type", ["Line", "Scatter"], horizontal=True, key="analysis_chart_type")
            color_choice = st.selectbox("Color group (optional)", ["None"] + [col for col in quick_cols if col not in {x_choice, y_choice}], key="analysis_color")
            use_markers = st.checkbox("Show markers", value=(chart_type == "Scatter"), key="analysis_markers")

            if chart_type == "Line":
                quick_fig = px.line(
                    df,
                    x=x_choice,
                    y=y_choice,
                    color=None if color_choice == "None" else color_choice,
                    markers=use_markers,
                    title=f"{y_choice} vs {x_choice}",
                )
            else:
                quick_fig = px.scatter(
                    df,
                    x=x_choice,
                    y=y_choice,
                    color=None if color_choice == "None" else color_choice,
                    title=f"{y_choice} vs {x_choice}",
                )
                if use_markers:
                    quick_fig.update_traces(marker=dict(size=8))

            st.plotly_chart(quick_fig, width="stretch", key="analysis_quick_plot")

        hist_col = st.selectbox("Histogram variable", numeric_cols, key="analysis_hist_var")
        hist_fig = px.histogram(df, x=hist_col, nbins=50, title=f"Histogram of {hist_col}")
        st.plotly_chart(hist_fig, width="stretch", key="analysis_histogram")

    with st.expander("Dataset preview"):
        st.dataframe(df.head(50))
        st.download_button(
            "Download dataset as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{dataset_name.replace(' ', '_').lower()}_analysis.csv",
            mime="text/csv",
        )


def forward_view(runner: PintleEngineRunner) -> None:
    if isinstance(runner, PintleEngineConfig):
        runner = PintleEngineRunner(runner)
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
            st.session_state["forward_result"] = results
            summarize_results(results)
        except Exception as exc:
            st.error(f"Pipeline evaluation failed: {exc}")
            return

        dataset_label = f"Forward Run {datetime.now().strftime('%H:%M:%S')}"
        store_dataset(dataset_label, create_single_run_dataframe(results, context="Forward"))


def inverse_view(runner: PintleEngineRunner, config_label: str) -> None:
    if isinstance(runner, PintleEngineConfig):
        runner = PintleEngineRunner(runner)
    st.header("Inverse Mode: Target Performance → Tank Pressures")
    
    # Mode selector
    inverse_mode = st.radio(
        "Inverse solver mode",
        ["Thrust only", "Thrust + O/F ratio"],
        horizontal=True,
        help="'Thrust only' scales baseline pressures uniformly. 'Thrust + O/F' solves for independent LOX and fuel pressures.",
    )
    
    st.markdown("---")

    target_thrust_kN = st.number_input(
        "Desired Thrust [kN]",
        min_value=0.1,
        value=6.65,
        step=0.1,
    )
    
    if inverse_mode == "Thrust + O/F ratio":
        target_MR = st.number_input(
            "Desired Mixture Ratio (O/F)",
            min_value=0.5,
            max_value=10.0,
            value=2.36,
            step=0.1,
            help="Target oxidizer-to-fuel mass ratio",
        )
        
        st.markdown("**Initial Guess (optional):**")
        col1, col2 = st.columns(2)
        with col1:
            guess_O_psi = st.number_input(
                "LOX Tank Pressure [psi]",
                min_value=200.0,
                value=1305.0,
                step=10.0,
                key="guess_O",
            )
        with col2:
            guess_F_psi = st.number_input(
                "Fuel Tank Pressure [psi]",
                min_value=200.0,
                value=974.0,
                step=10.0,
                key="guess_F",
            )
        
        if st.button("Solve for Tank Pressures (2D)", type="primary", key="solve_2d"):
            try:
                (P_tank_O_solution, P_tank_F_solution), results, diagnostics = solve_for_thrust_and_MR(
                    runner,
                    target_thrust_kN,
                    target_MR,
                    initial_guess_psi=(guess_O_psi, guess_F_psi),
                )
            except ThrustSolveError as exc:
                st.error(str(exc))
                diag = exc.diagnostics
                if "history" in diag and diag["history"]["thrust"]:
                    with st.expander("Iteration history"):
                        hist_df = pd.DataFrame({
                            "Iteration": range(len(diag["history"]["thrust"])),
                            "P_tank_O [psi]": diag["history"]["P_tank_O"],
                            "P_tank_F [psi]": diag["history"]["P_tank_F"],
                            "Thrust [kN]": diag["history"]["thrust"],
                            "O/F": diag["history"]["MR"],
                        })
                        st.dataframe(hist_df)
                return
            except Exception as exc:
                st.error(f"Failed to find tank pressures: {exc}")
                return

            st.success(f"✅ Converged in {diagnostics['iterations']} iterations!")
            
            st.subheader("Required Tank Pressures")
            col_sol1, col_sol2 = st.columns(2)
            with col_sol1:
                st.metric("LOX Tank Pressure", f"{P_tank_O_solution:.1f} psi")
            with col_sol2:
                st.metric("Fuel Tank Pressure", f"{P_tank_F_solution:.1f} psi")
            
            st.subheader("Performance at Solution")
            col_perf1, col_perf2, col_perf3 = st.columns(3)
            with col_perf1:
                st.metric("Thrust", f"{diagnostics['final_thrust']:.2f} kN", 
                         delta=f"{diagnostics['thrust_error_pct']:.2f}% error")
            with col_perf2:
                st.metric("O/F Ratio", f"{diagnostics['final_MR']:.3f}",
                         delta=f"{diagnostics['MR_error_pct']:.2f}% error")
            with col_perf3:
                st.metric("Iterations", f"{diagnostics['iterations']}")
            
            st.session_state["inverse_result"] = {"results": results, "diagnostics": diagnostics}
            summarize_results(results)
            dataset_label = f"Inverse 2D {datetime.now().strftime('%H:%M:%S')}"
            store_dataset(dataset_label, create_single_run_dataframe(results, context="Inverse 2D"))
            
            with st.expander("Convergence history"):
                hist_df = pd.DataFrame({
                    "Iteration": range(len(diagnostics["history"]["thrust"])),
                    "P_tank_O [psi]": diagnostics["history"]["P_tank_O"],
                    "P_tank_F [psi]": diagnostics["history"]["P_tank_F"],
                    "Thrust [kN]": diagnostics["history"]["thrust"],
                    "O/F": diagnostics["history"]["MR"],
                    "Thrust Error": [f"{e*100:.2f}%" for e in diagnostics["history"]["thrust_error"]],
                    "O/F Error": [f"{e*100:.2f}%" for e in diagnostics["history"]["MR_error"]],
                })
                st.dataframe(hist_df)
                
                # Plot convergence
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(diagnostics["history"]["thrust"]))),
                    y=[abs(e)*100 for e in diagnostics["history"]["thrust_error"]],
                    name="Thrust Error %",
                    mode="lines+markers",
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(len(diagnostics["history"]["MR"]))),
                    y=[abs(e)*100 for e in diagnostics["history"]["MR_error"]],
                    name="O/F Error %",
                    mode="lines+markers",
                ))
                fig.update_layout(
                    title="Convergence History",
                    xaxis_title="Iteration",
                    yaxis_title="Absolute Error (%)",
                    yaxis_type="log",
                    height=400,
                )
                st.plotly_chart(fig, width='stretch', key="inverse_2d_convergence")
        
        return  # Exit early for 2D mode
    
    # Original 1D mode (thrust only)
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

    if st.button("Solve for Tank Pressures (1D)", type="primary", key="solve_1d"):
        try:
            (P_tank_O_solution, P_tank_F_solution), results, diagnostics = solve_for_thrust(
                runner,
                target_thrust_kN,
                (base_O_psi, base_F_psi),
            )
        except ThrustSolveError as exc:
            st.error(str(exc))
            diag = exc.diagnostics
            st.info(
                f"Achievable thrust range: {diag['min_thrust']:.2f} - {diag['max_thrust']:.2f} kN\n"
                f"Baseline thrust (scale=1.0): {diag['baseline_thrust']:.2f} kN"
            )
            invalid = diag.get("invalid_samples") or []
            if invalid:
                with st.expander("Failed scale evaluations"):
                    st.write(
                        "The solver skipped the following scale factors because the forward pipeline failed "
                        "(e.g., no Pc solution):"
                    )
                    st.dataframe(
                        {
                            "Scale": [f"{scale:.3f}" for scale, _ in invalid],
                            "Error": [msg for _, msg in invalid],
                        }
                    )
            return
        except Exception as exc:
            st.error(f"Failed to find tank pressures: {exc}")
            return

        st.subheader("Required Tank Pressures")
        st.metric("LOX Tank Pressure", f"{P_tank_O_solution * PA_TO_PSI:.1f} psi")
        st.metric("Fuel Tank Pressure", f"{P_tank_F_solution * PA_TO_PSI:.1f} psi")

        st.subheader("Performance at Solution")
        st.session_state["inverse_result"] = {"results": results, "diagnostics": diagnostics}
        summarize_results(results)
        dataset_label = f"Inverse Run {datetime.now().strftime('%H:%M:%S')}"
        store_dataset(dataset_label, create_single_run_dataframe(results, context="Inverse"))

        st.info(
            f"Configuration: {config_label}\n"
            f"Baseline thrust (scale=1.0): {diagnostics['baseline_thrust']:.2f} kN\n"
            f"Achievable thrust range: {diagnostics['min_thrust']:.2f} - {diagnostics['max_thrust']:.2f} kN"
        )

        invalid = diagnostics.get("invalid_samples") or []
        if invalid:
            with st.expander("Failed scale evaluations"):
                st.write(
                    "Some sampled scale factors failed during bracketing. They are listed here for reference."
                )
                st.dataframe(
                    {
                        "Scale": [f"{scale:.3f}" for scale, _ in invalid],
                        "Error": [msg for _, msg in invalid],
                    }
                )

        with st.expander("Diagnostics: Thrust vs Scale"):
            st.write("This table shows sampled thrust values used to bracket the solution.")
            diag_table = {
                "Scale": list(diagnostics["sample_scales"]),
                "Thrust [kN]": list(diagnostics["sample_thrusts"]),
            }
            st.dataframe(diag_table)


def timeseries_view(runner: PintleEngineRunner, config_label: str) -> None:
    if isinstance(runner, PintleEngineConfig):
        runner = PintleEngineRunner(runner)
    st.header("Time-Series Evaluation: Pressure Curve → Thrust Curve")
    st.write(
        "Generate a pressure profile or upload a CSV to evaluate thrust and performance over time."
    )

    mode = st.radio("Input method", ["Generate profile", "Upload CSV"], horizontal=True)

    if mode == "Generate profile":
        with st.form("timeseries_generate_form"):
            duration = st.number_input("Duration [s]", min_value=0.1, value=5.0)
            n_steps = st.number_input("Number of samples", min_value=2, max_value=2000, value=101, step=1)

            st.subheader("LOX pressure profile")
            col1, col2 = st.columns(2)
            with col1:
                lox_start = st.number_input("Start pressure [psi]", value=1305.0, key="lox_start")
                lox_model = st.selectbox("Profile type", ["linear", "exponential", "power"], key="lox_model")
            with col2:
                lox_end = st.number_input("End pressure [psi]", value=900.0, key="lox_end")
                if lox_model == "exponential":
                    lox_decay = st.slider("Decay constant", min_value=0.1, max_value=10.0, value=3.0, step=0.1, key="lox_decay")
                elif lox_model == "power":
                    lox_power = st.slider("Power exponent", min_value=0.1, max_value=5.0, value=2.0, step=0.1, key="lox_power")
                else:
                    lox_decay = lox_power = None

            st.subheader("Fuel pressure profile")
            col3, col4 = st.columns(2)
            with col3:
                fuel_start = st.number_input("Start pressure [psi]", value=974.0, key="fuel_start")
                fuel_model = st.selectbox("Profile type", ["linear", "exponential", "power"], key="fuel_model")
            with col4:
                fuel_end = st.number_input("End pressure [psi]", value=650.0, key="fuel_end")
                if fuel_model == "exponential":
                    fuel_decay = st.slider("Decay constant", min_value=0.1, max_value=10.0, value=3.0, step=0.1, key="fuel_decay")
                elif fuel_model == "power":
                    fuel_power = st.slider("Power exponent", min_value=0.1, max_value=5.0, value=2.0, step=0.1, key="fuel_power")
                else:
                    fuel_decay = fuel_power = None

            submitted = st.form_submit_button("Run Time-Series")

        if submitted:
            try:
                times, lox_profile = generate_pressure_profile(
                    lox_model,
                    lox_start,
                    lox_end,
                    duration,
                    int(n_steps),
                    decay_constant=lox_decay if lox_model == "exponential" else None,
                    power=lox_power if lox_model == "power" else None,
                )
                _, fuel_profile = generate_pressure_profile(
                    fuel_model,
                    fuel_start,
                    fuel_end,
                    duration,
                    int(n_steps),
                    decay_constant=fuel_decay if fuel_model == "exponential" else None,
                    power=fuel_power if fuel_model == "power" else None,
                )
            except Exception as exc:
                st.error(f"Failed to generate profiles: {exc}")
                return

            df, errors = compute_timeseries_dataframe(runner, times, lox_profile, fuel_profile)
            if errors:
                st.error("Errors encountered during time-series evaluation:")
                for err in errors:
                    st.write(f"- {err}")
                return

            st.session_state["timeseries_results"] = {"data": df, "meta": {"source": "profile", "config": config_label}}
            store_dataset("Time Series (generated)", df)

            display_time_series_summary(df)
            plot_time_series_results(df)

            if errors:
                st.warning("Some time steps did not converge. Affected rows contain NaNs in the output dataset.")

            with st.expander("Data table"):
                st.dataframe(df)
                st.download_button(
                    "Download results as CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="time_series_generated.csv",
                    mime="text/csv",
                )

            st.success("Time-series evaluation complete.")

    else:  # Upload CSV
        uploaded_csv = st.file_uploader(
            "Upload CSV (columns: time [s – optional], P_tank_O [psi], P_tank_F [psi])",
            type=["csv"],
            key="timeseries_upload",
        )

        if uploaded_csv is None:
            st.info("Upload a CSV to generate time-series plots.")
            return

        try:
            df_input = pd.read_csv(uploaded_csv)
        except Exception as exc:
            st.error(f"Failed to read CSV: {exc}")
            return

        required_cols = {"P_tank_O", "P_tank_F"}
        if not required_cols.issubset(df_input.columns):
            st.error(f"CSV must contain columns: {required_cols}")
            return

        if "time" not in df_input.columns:
            time_step = st.number_input("Time step between samples [s]", min_value=0.001, value=1.0, key="timeseries_time_step")
            df_input["time"] = np.arange(len(df_input)) * time_step

        df_input = df_input.sort_values("time")

        if st.button("Run Time-Series", type="primary"):
            # Validate pressure ranges
            P_O_min = df_input["P_tank_O"].min()
            P_O_max = df_input["P_tank_O"].max()
            P_F_min = df_input["P_tank_F"].min()
            P_F_max = df_input["P_tank_F"].max()
            
            st.info(f"Pressure ranges in CSV: LOX [{P_O_min:.0f} - {P_O_max:.0f}] psi, Fuel [{P_F_min:.0f} - {P_F_max:.0f}] psi")
            
            # Warn if pressures are too low
            if P_O_min < 800 or P_F_min < 600:
                st.warning(
                    f"⚠️ **Low tank pressures detected!** This engine is designed for ~1305 psi LOX and ~974 psi fuel. "
                    f"Your CSV has LOX as low as {P_O_min:.0f} psi and fuel as low as {P_F_min:.0f} psi. "
                    f"The solver may fail at these low pressures because there isn't enough pressure to sustain combustion after feed losses and injector drops. "
                    f"Consider scaling up your pressures by 2-3x."
                )
            
            df, errors = compute_timeseries_dataframe(
                runner,
                df_input["time"].to_numpy(dtype=float),
                df_input["P_tank_O"].to_numpy(dtype=float),
                df_input["P_tank_F"].to_numpy(dtype=float),
            )
            
            # Count failures
            n_total = len(df)
            n_failed = df["Pc (psi)"].isna().sum()
            n_success = n_total - n_failed
            
            if n_failed > 0:
                st.error(f"❌ Solver failed for {n_failed}/{n_total} time steps ({n_failed/n_total*100:.1f}%)")
                if errors:
                    with st.expander("View error details"):
                        for err in errors[:10]:  # Show first 10 errors
                            st.write(f"- {err}")
                        if len(errors) > 10:
                            st.write(f"... and {len(errors)-10} more errors")
                
                if n_success == 0:
                    st.error("All time steps failed. Cannot generate plots. Please check your tank pressures and configuration.")
                    return
                else:
                    st.warning(f"Proceeding with {n_success} successful evaluations. Failed rows contain NaN values.")
            else:
                st.success(f"✅ All {n_total} time steps evaluated successfully!")

            st.session_state["timeseries_results"] = {"data": df, "meta": {"source": "csv", "config": config_label}}
            store_dataset("Time Series (uploaded)", df)

            display_time_series_summary(df)
            plot_time_series_results(df)

            with st.expander("Data table"):
                st.dataframe(df)
                st.download_button(
                    "Download results as CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="time_series_results.csv",
                    mime="text/csv",
                )

            st.success("Time-series evaluation complete.")


def flight_sim_view(runner: PintleEngineRunner, config_obj: PintleEngineConfig, config_label: str) -> None:
    if isinstance(runner, PintleEngineConfig):
        runner = PintleEngineRunner(runner)
    st.header("Flight Simulation")
    st.write("Use engine performance to simulate a basic rocket flight and report apogee and velocity.")

    source = st.radio("Performance source", ["Tank pressures (constant thrust)", "Dataset (time-varying)"], horizontal=True, key="flight_perf_source")

    # Tank pressures inputs shown only for constant source
    if source == "Tank pressures (constant thrust)":
        col1, col2 = st.columns(2)
        with col1:
            P_tank_O_psi = st.number_input(
                "LOX Tank Pressure [psi]",
                min_value=50.0,
                max_value=3000.0,
                value=1305.0,
                step=5.0,
                key="flight_lox_tank_psi",
            )
        with col2:
            P_tank_F_psi = st.number_input(
                "Fuel Tank Pressure [psi]",
                min_value=50.0,
                max_value=3000.0,
                value=974.0,
                step=5.0,
                key="flight_fuel_tank_psi",
            )
    else:
        # Dataset selection and column mapping
        datasets: Dict[str, pd.DataFrame] = st.session_state.get("custom_plot_datasets", {})
        if not datasets:
            st.warning("No datasets available. Run a time-series or forward analysis first to populate datasets.")
            return
        ds_names = list(datasets.keys())
        default_ds = st.session_state.get("last_custom_dataset") or ds_names[0]
        dataset_name = st.selectbox("Dataset for thrust and O/F", ds_names, index=ds_names.index(default_ds) if default_ds in ds_names else 0, key="flight_ds_select")
        ds_df = datasets[dataset_name]
        num_cols = ds_df.select_dtypes(include=[np.number]).columns.tolist()
        # sensible defaults
        time_col = "time" if "time" in ds_df.columns else num_cols[0]
        thrust_candidates = [c for c in ds_df.columns if "Thrust" in c]
        thrust_col = thrust_candidates[0] if thrust_candidates else num_cols[1 if len(num_cols) > 1 else 0]
        mdot_o_col = "mdot_O (kg/s)" if "mdot_O (kg/s)" in ds_df.columns else None
        mdot_f_col = "mdot_F (kg/s)" if "mdot_F (kg/s)" in ds_df.columns else None
        mr_col = "MR" if "MR" in ds_df.columns else None
        mdot_total_col = "mdot_total (kg/s)" if "mdot_total (kg/s)" in ds_df.columns else None

        st.markdown("#### Map dataset columns")
        colm = st.columns(4)
        with colm[0]:
            time_col = st.selectbox("Time column [s]", ds_df.columns.tolist(), index=ds_df.columns.tolist().index(time_col) if time_col in ds_df.columns else 0, key="flight_ds_time_col")
        with colm[1]:
            thrust_col = st.selectbox("Thrust column", ds_df.columns.tolist(), index=ds_df.columns.tolist().index(thrust_col) if thrust_col in ds_df.columns else 0, key="flight_ds_thrust_col")
        with colm[2]:
            mr_col = st.selectbox("MR (O/F) column (optional)", ["None"] + ds_df.columns.tolist(), index=(0 if mr_col is None else ds_df.columns.tolist().index(mr_col)+1), key="flight_ds_mr_col")
        with colm[3]:
            mdot_total_col = st.selectbox("Total mdot column (optional)", ["None"] + ds_df.columns.tolist(), index=(0 if mdot_total_col is None else ds_df.columns.tolist().index(mdot_total_col)+1), key="flight_ds_mdtot_col")
        colm2 = st.columns(2)
        with colm2[0]:
            mdot_o_col = st.selectbox("mdot_O column (optional)", ["None"] + ds_df.columns.tolist(), index=(0 if mdot_o_col is None else ds_df.columns.tolist().index(mdot_o_col)+1), key="flight_ds_mdot_o_col")
        with colm2[1]:
            mdot_f_col = st.selectbox("mdot_F column (optional)", ["None"] + ds_df.columns.tolist(), index=(0 if mdot_f_col is None else ds_df.columns.tolist().index(mdot_f_col)+1), key="flight_ds_mdot_f_col")

    with st.form("flight_sim_form"):
        if source == "Tank pressures (constant thrust)":
            col3, col4, col5 = st.columns(3)
            with col3:
                default_burn = float(config_obj.thrust.burn_time) if getattr(config_obj, "thrust", None) and config_obj.thrust else 5.0
                burn_time = st.number_input("Burn time [s]", min_value=0.5, value=default_burn, step=0.5, key="flight_burn_time")
            with col4:
                default_m_lox = float(getattr(getattr(config_obj, "lox_tank", None), "mass", None)) if getattr(getattr(config_obj, "lox_tank", None), "mass", None) is not None else 20.0
                m_lox = st.number_input("Initial LOX mass [kg]", min_value=0.1, value=default_m_lox, step=0.1, key="flight_m_lox")
            with col5:
                default_m_fuel = float(getattr(getattr(config_obj, "fuel_tank", None), "mass", None)) if getattr(getattr(config_obj, "fuel_tank", None), "mass", None) is not None else 4.0
                m_fuel = st.number_input("Initial Fuel mass [kg]", min_value=0.1, value=default_m_fuel, step=0.1, key="flight_m_fuel")

            st.markdown("### Environment")
            env_date = None
            env_hour = 12
            if getattr(config_obj, "environment", None) and config_obj.environment:
                try:
                    y, m, d, h = list(config_obj.environment.date)
                    env_date = datetime(y, m, d).date()
                    env_hour = int(h)
                except Exception:
                    env_date = datetime.now().date()
                    env_hour = 12
            else:
                env_date = datetime.now().date()
                env_hour = 12

            cold1, cold2 = st.columns(2)
            with cold1:
                sel_date = st.date_input("Launch date", value=env_date, key="flight_env_date")
            with cold2:
                sel_hour = st.number_input("Launch hour [0-23]", min_value=0, max_value=23, value=env_hour, step=1, key="flight_env_hour")
        else:
            # Dataset mode: user defines propellant fill; burn time comes from dataset
            burn_time = None
            colpf1, colpf2 = st.columns(2)
            with colpf1:
                default_m_lox = float(getattr(getattr(config_obj, "lox_tank", None), "mass", None)) if getattr(getattr(config_obj, "lox_tank", None), "mass", None) is not None else 20.0
                m_lox = st.number_input("Initial LOX mass [kg]", min_value=0.1, value=default_m_lox, step=0.1, key="flight_m_lox_ds")
            with colpf2:
                default_m_fuel = float(getattr(getattr(config_obj, "fuel_tank", None), "mass", None)) if getattr(getattr(config_obj, "fuel_tank", None), "mass", None) is not None else 4.0
                m_fuel = st.number_input("Initial Fuel mass [kg]", min_value=0.1, value=default_m_fuel, step=0.1, key="flight_m_fuel_ds")
            sel_date = None
            sel_hour = None

        run_btn = st.form_submit_button("Run Flight Simulation", type="primary")

    # Show editors regardless of whether the form is submitted; simulation will run only if run_btn is True.

    # Create a working copy of config with overrides for flight-related fields
    try:
        working = copy.deepcopy(config_obj).model_dump()
        if source == "Tank pressures (constant thrust)":
            # ensure optional sections exist
            working.setdefault("thrust", {"burn_time": burn_time})
            working["thrust"]["burn_time"] = float(burn_time)
            # update masses from user inputs (tanks now hold masses)
            lox_tank_work = working.setdefault("lox_tank", {})
            fuel_tank_work = working.setdefault("fuel_tank", {})
            lox_tank_work["mass"] = float(m_lox)
            fuel_tank_work["mass"] = float(m_fuel)
            # update environment date
            if "environment" not in working or working["environment"] is None:
                working["environment"] = {}
            working["environment"]["date"] = [int(sel_date.year), int(sel_date.month), int(sel_date.day), int(sel_hour)]
        else:
            # dataset mode: always use user-defined masses; burn time set later from dataset
            lox_tank_work = working.setdefault("lox_tank", {})
            fuel_tank_work = working.setdefault("fuel_tank", {})
            lox_tank_work["mass"] = float(m_lox)
            fuel_tank_work["mass"] = float(m_fuel)
    except Exception as exc:
        st.error(f"Invalid flight configuration: {exc}")
        return

    # Collapsible configuration editor
    st.subheader("Flight configuration")
    with st.expander("Environment", expanded=False):
        env = working.get("environment", {}) or {}
        env.setdefault("latitude", 35.0)
        env.setdefault("longitude", -117.0)
        env.setdefault("elevation", 0.0)
        env.setdefault("p_amb", 101325.0)
        colE1, colE2 = st.columns(2)
        with colE1:
            env_lat = st.number_input("Latitude [deg]", value=float(env.get("latitude", 35.0)), key="flight_env_lat")
            env_elev = st.number_input("Elevation [m]", value=float(env.get("elevation", 0.0)), key="flight_env_elev")
        with colE2:
            env_lon = st.number_input("Longitude [deg]", value=float(env.get("longitude", -117.0)), key="flight_env_lon")
            env_pamb = st.number_input("Ambient pressure [Pa]", value=float(env.get("p_amb", 101325.0)), key="flight_env_pamb")
        env["latitude"] = float(env_lat)
        env["longitude"] = float(env_lon)
        env["elevation"] = float(env_elev)
        env["p_amb"] = float(env_pamb)
        working["environment"] = env

    with st.expander("Rocket", expanded=False):
        rocket = working.get("rocket") or {}
        rocket.setdefault("mass", 90.72)
        rocket.setdefault("inertia", [8.0, 8.0, 0.5])
        rocket.setdefault("radius", 0.1)
        rocket.setdefault("cm_wo_motor", 1.0)
        rocket.setdefault("dry_mass", 12.0)
        rocket.setdefault("motor_inertia", [0.1, 0.1, 0.1])
        colR1, colR2, colR3 = st.columns(3)
        with colR1:
            r_mass = st.number_input("Rocket mass [kg]", value=float(rocket.get("mass", 90.72)), key="flight_rocket_mass")
            r_radius = st.number_input("Rocket radius [m]", value=float(rocket.get("radius", 0.1)), key="flight_rocket_radius")
        with colR2:
            r_cm = st.number_input("CM without motor [m]", value=float(rocket.get("cm_wo_motor", 1.0)), key="flight_rocket_cm")
            r_dry = st.number_input("Rocket dry mass [kg]", value=float(rocket.get("dry_mass", 12.0)), key="flight_rocket_dry")
        with colR3:
            mi_x = st.number_input("Motor inertia X", value=float(rocket.get("motor_inertia", [0.1, 0.1, 0.1])[0]), key="flight_motor_inertia_x")
            mi_y = st.number_input("Motor inertia Y", value=float(rocket.get("motor_inertia", [0.1, 0.1, 0.1])[1]), key="flight_motor_inertia_y")
            mi_z = st.number_input("Motor inertia Z", value=float(rocket.get("motor_inertia", [0.1, 0.1, 0.1])[2]), key="flight_motor_inertia_z")
        rocket["mass"] = float(r_mass)
        rocket["radius"] = float(r_radius)
        rocket["cm_wo_motor"] = float(r_cm)
        rocket["dry_mass"] = float(r_dry)
        rocket["motor_inertia"] = [float(mi_x), float(mi_y), float(mi_z)]

        # Fins
        fins = (rocket.get("fins") or {})
        fins.setdefault("no_fins", 3)
        fins.setdefault("root_chord", 0.2)
        fins.setdefault("tip_chord", 0.1)
        fins.setdefault("fin_span", 0.3)
        fins.setdefault("fin_position", 0.0)
        colF1, colF2, colF3 = st.columns(3)
        with colF1:
            fins["no_fins"] = int(st.number_input("Fin count", value=int(fins["no_fins"]), min_value=1, step=1, key="flight_fins_count"))
            fins["root_chord"] = float(st.number_input("Root chord [m]", value=float(fins["root_chord"]), key="flight_fins_root"))
        with colF2:
            fins["tip_chord"] = float(st.number_input("Tip chord [m]", value=float(fins["tip_chord"]), key="flight_fins_tip"))
            fins["fin_span"] = float(st.number_input("Fin span [m]", value=float(fins["fin_span"]), key="flight_fins_span"))
        with colF3:
            fins["fin_position"] = float(st.number_input("Fin position [m]", value=float(fins["fin_position"]), key="flight_fins_pos"))
        rocket["fins"] = fins
        working["rocket"] = rocket

    with st.expander("Tanks", expanded=False):
        lox_tank = (working.get("lox_tank") or {})
        fuel_tank = (working.get("fuel_tank") or {})
        press_tank = (working.get("press_tank") or {})
        # LOX
        lox_tank.setdefault("lox_h", 1.14)
        lox_tank.setdefault("lox_radius", 0.0762)
        lox_tank.setdefault("ox_tank_pos", 0.6)
        colL1, colL2, colL3 = st.columns(3)
        with colL1:
            lox_tank["lox_h"] = float(st.number_input("LOX tank height [m]", value=float(lox_tank["lox_h"]), key="flight_lox_h"))
        with colL2:
            lox_tank["lox_radius"] = float(st.number_input("LOX tank radius [m]", value=float(lox_tank["lox_radius"]), key="flight_lox_radius"))
        with colL3:
            lox_tank["ox_tank_pos"] = float(st.number_input("LOX tank position [m]", value=float(lox_tank["ox_tank_pos"]), key="flight_lox_pos"))
        # Fuel
        fuel_tank.setdefault("rp1_h", 0.609)
        fuel_tank.setdefault("rp1_radius", 0.0762)
        fuel_tank.setdefault("fuel_tank_pos", -0.2)
        colFu1, colFu2, colFu3 = st.columns(3)
        with colFu1:
            fuel_tank["rp1_h"] = float(st.number_input("Fuel tank height [m]", value=float(fuel_tank["rp1_h"]), key="flight_rp1_h"))
        with colFu2:
            fuel_tank["rp1_radius"] = float(st.number_input("Fuel tank radius [m]", value=float(fuel_tank["rp1_radius"]), key="flight_rp1_radius"))
        with colFu3:
            fuel_tank["fuel_tank_pos"] = float(st.number_input("Fuel tank position [m]", value=float(fuel_tank["fuel_tank_pos"]), key="flight_rp1_pos"))
        # Pressurant (optional)
        if press_tank is None:
            press_tank = {}
        press_tank.setdefault("press_h", 0.457)
        press_tank.setdefault("press_radius", 0.0762)
        press_tank.setdefault("pres_tank_pos", 1.2)
        colP1, colP2, colP3 = st.columns(3)
        with colP1:
            press_tank["press_h"] = float(st.number_input("Pressurant tank height [m]", value=float(press_tank["press_h"]), key="flight_press_h"))
        with colP2:
            press_tank["press_radius"] = float(st.number_input("Pressurant tank radius [m]", value=float(press_tank["press_radius"]), key="flight_press_radius"))
        with colP3:
            press_tank["pres_tank_pos"] = float(st.number_input("Pressurant tank position [m]", value=float(press_tank["pres_tank_pos"]), key="flight_press_pos"))
        working["lox_tank"] = lox_tank
        working["fuel_tank"] = fuel_tank
        working["press_tank"] = press_tank

    with st.expander("Nozzle", expanded=False):
        nozzle = working.get("nozzle") or {}
        nozzle.setdefault("A_throat", 0.00156235266901)
        nozzle.setdefault("A_exit", 0.00831498636119)
        nozzle.setdefault("expansion_ratio", 6.54)
        nozzle.setdefault("efficiency", 0.98)
        colN1, colN2 = st.columns(2)
        with colN1:
            nozzle["A_throat"] = float(st.number_input("Throat area [m²]", value=float(nozzle["A_throat"]), key="flight_noz_at"))
            nozzle["expansion_ratio"] = float(st.number_input("Expansion ratio (Ae/At)", value=float(nozzle["expansion_ratio"]), key="flight_noz_er"))
        with colN2:
            nozzle["A_exit"] = float(st.number_input("Exit area [m²]", value=float(nozzle["A_exit"]), key="flight_noz_ae"))
            nozzle["efficiency"] = float(st.number_input("Nozzle efficiency", value=float(nozzle["efficiency"]), key="flight_noz_eta"))
        working["nozzle"] = nozzle

    with st.expander("Fluids (properties)", expanded=False):
        fluids = working.get("fluids") or {}
        ox = fluids.get("oxidizer") or {}
        fu = fluids.get("fuel") or {}
        ox.setdefault("name", "LOX")
        ox.setdefault("density", 1140.0)
        ox.setdefault("temperature", 90.0)
        fu.setdefault("name", "RP-1")
        fu.setdefault("density", 780.0)
        fu.setdefault("temperature", 293.0)
        colOx1, colOx2, colOx3 = st.columns(3)
        with colOx1:
            ox["name"] = st.text_input("Oxidizer name", value=str(ox["name"]), key="flight_ox_name")
        with colOx2:
            ox["density"] = float(st.number_input("Oxidizer density [kg/m³]", value=float(ox["density"]), key="flight_ox_density"))
        with colOx3:
            ox["temperature"] = float(st.number_input("Oxidizer temperature [K]", value=float(ox["temperature"]), key="flight_ox_temp"))
        colFu1, colFu2, colFu3 = st.columns(3)
        with colFu1:
            fu["name"] = st.text_input("Fuel name", value=str(fu["name"]), key="flight_fu_name")
        with colFu2:
            fu["density"] = float(st.number_input("Fuel density [kg/m³]", value=float(fu["density"]), key="flight_fu_density"))
        with colFu3:
            fu["temperature"] = float(st.number_input("Fuel temperature [K]", value=float(fu["temperature"]), key="flight_fu_temp"))
        fluids["oxidizer"] = ox
        fluids["fuel"] = fu
        working["fluids"] = fluids

    # Validate edited working config
    try:
        config_for_flight = PintleEngineConfig(**working)
    except Exception as exc:
        st.error(f"Edited flight configuration is invalid: {exc}")
        return

    if run_btn:
        # If dataset-driven, build Functions and override burn time
        if source == "Dataset (time-varying)":
            try:
                # Convert time column and thrust values
                t_vals = to_elapsed_seconds(ds_df[time_col])
                thrust_vals = np.asarray(ds_df[thrust_col], dtype=float)
            except Exception as exc:
                st.error(f"Invalid dataset columns: {exc}")
                return
            # Unit handling for thrust
            if "(kN)" in thrust_col:
                thrust_vals_SI = thrust_vals * 1000.0
            elif "(N)" in thrust_col:
                thrust_vals_SI = thrust_vals
            else:
                # Assume N if unitless; provide a toggle?
                thrust_vals_SI = thrust_vals
            # mdot handling
            if mdot_o_col != "None" and mdot_f_col != "None":
                mdot_O_vals = np.asarray(ds_df[mdot_o_col], dtype=float)
                mdot_F_vals = np.asarray(ds_df[mdot_f_col], dtype=float)
            elif mr_col != "None" and mdot_total_col != "None":
                MR_vals = np.asarray(ds_df[mr_col], dtype=float)
                mdot_total_vals = np.asarray(ds_df[mdot_total_col], dtype=float)
                mdot_O_vals = mdot_total_vals * (MR_vals / (1.0 + MR_vals))
                mdot_F_vals = mdot_total_vals * (1.0 / (1.0 + MR_vals))
            else:
                st.error("Provide either mdot_O and mdot_F columns, or MR and total mdot.")
                return

            # Sort by time and drop duplicates if needed
            order = np.argsort(t_vals)
            t_vals = t_vals[order]
            thrust_vals_SI = thrust_vals_SI[order]
            mdot_O_vals = mdot_O_vals[order]
            mdot_F_vals = mdot_F_vals[order]

            # Deduce burn time from dataset
            if len(t_vals) < 2:
                st.error("Dataset must contain at least two time samples to define a burn duration.")
                return
            burn_time_ds = float(np.max(t_vals) - np.min(t_vals))
            if not np.isfinite(burn_time_ds) or burn_time_ds <= 0.0:
                st.error("Dataset time axis must increase (duration must be > 0 s). Check the selected time column.")
                return
            working["thrust"]["burn_time"] = burn_time_ds
            # Use user-defined propellant masses (set earlier in the form)
            try:
                config_for_flight = PintleEngineConfig(**working)
            except Exception as exc:
                st.error(f"Edited flight configuration is invalid: {exc}")
                return

            # Build RocketPy Functions
            thrust_func = build_rp_function(t_vals, thrust_vals_SI)
            mdot_O_func = build_rp_function(t_vals, mdot_O_vals)
            mdot_F_func = build_rp_function(t_vals, mdot_F_vals)

            # Run flight directly with dataset-driven inputs
            try:
                sim_result = setup_flight(config_for_flight, thrust_func, mdot_O_func, mdot_F_func, plot_results=False)
            except Exception as exc:
                st.error(f"Flight simulation failed: {exc}")
                return

            apogee = sim_result.get("apogee")
            max_v = sim_result.get("max_velocity")
            flight = sim_result.get("flight")

            colm1, colm2 = st.columns(2)
            colm1.metric("Apogee", f"{apogee:.1f} m" if isinstance(apogee, (int, float)) else "N/A")
            if isinstance(max_v, (int, float)) and max_v is not None:
                colm2.metric("Max Velocity", f"{max_v:.1f} m/s")
            else:
                colm2.metric("Max Velocity", "N/A")

            # Plot altitude/velocity
            try:
                t_series, z_series, vz_series = extract_flight_series(flight)
                plot_flight_results(t_series, z_series, vz_series, key_suffix="_ds")
            except Exception as exc:
                st.warning(f"Could not extract time series: {exc}")
                return
            with st.expander("Rocket view (render)"):
                render_rocket_view(flight)
            with st.expander("Additional rocket plots"):
                plot_additional_rocket_plots(flight, t_series, key_suffix="_ds")
            with st.expander("Thrust curve"):
                thrust_df = pd.DataFrame({"time": t_vals, "Thrust (N)": thrust_vals_SI})
                st.plotly_chart(px.line(thrust_df, x="time", y="Thrust (N)", title="Thrust Curve (dataset)"), width='stretch', key="flight_thrust_plot_ds")
            return

        # Evaluate engine performance at specified tank pressures
        try:
            results = runner.evaluate(P_tank_O_psi * PSI_TO_PA, P_tank_F_psi * PSI_TO_PA)
        except Exception as exc:
            st.error(f"Engine performance evaluation failed: {exc}")
            return

        F = float(results.get("F", 0.0))
        mdot_O = float(results.get("mdot_O", 0.0))
        mdot_F = float(results.get("mdot_F", 0.0))

        if F <= 0 or mdot_O <= 0 or mdot_F <= 0:
            st.error("Non-physical engine outputs (thrust or mass flows <= 0). Check inputs.")
            return

        # Build constant thrust curve [(t, F)]
        thrust_curve = [(0.0, F), (float(burn_time), F)]

        try:
            sim_result = setup_flight(config_for_flight, thrust_curve, mdot_O, mdot_F, plot_results=False)
        except Exception as exc:
            st.error(f"Flight simulation failed: {exc}")
            return

        apogee = sim_result.get("apogee")
        max_v = sim_result.get("max_velocity")
        flight = sim_result.get("flight")

        colm1, colm2 = st.columns(2)
        colm1.metric("Apogee", f"{apogee:.1f} m" if isinstance(apogee, (int, float)) else "N/A")
        if isinstance(max_v, (int, float)) and max_v is not None:
            colm2.metric("Max Velocity", f"{max_v:.1f} m/s")
        else:
            colm2.metric("Max Velocity", "N/A")

        # Extract and plot time series
        try:
            t_series, z_series, vz_series = extract_flight_series(flight)
            plot_flight_results(t_series, z_series, vz_series)
        except Exception as exc:
            st.warning(f"Could not extract time series: {exc}")
            return

        with st.expander("Thrust curve"):
            thrust_df = pd.DataFrame({"time": [0.0, float(burn_time)], "Thrust (N)": [F, F]})
            thrust_fig = px.line(thrust_df, x="time", y="Thrust (N)", title="Thrust Curve (assumed constant)")
            st.plotly_chart(thrust_fig, width='stretch', key="flight_thrust_plot")
        with st.expander("Rocket view (render)"):
            render_rocket_view(flight)
        with st.expander("Additional rocket plots"):
            plot_additional_rocket_plots(flight, t_series)
    else:
        st.info("Edit configuration above, then click Run Flight Simulation.")

def detect_fluid_choice(fluid: Dict[str, Any]) -> str:
    name = fluid.get("name")
    if name in FLUID_LIBRARY:
        defaults = FLUID_LIBRARY[name]
        tolerance = 1e-3
        if all(abs(float(fluid[key]) - defaults[key]) <= tolerance * max(1.0, abs(defaults[key])) for key in ["density", "viscosity", "surface_tension", "vapor_pressure"]):
            return name
    for candidate, defaults in FLUID_LIBRARY.items():
        tolerance = 1e-3
        if all(abs(float(fluid[key]) - defaults[key]) <= tolerance * max(1.0, abs(defaults[key])) for key in ["density", "viscosity", "surface_tension", "vapor_pressure"]):
            return candidate
    return "Custom"


def _clear_chamber_design_form_state(config_label: str) -> None:
    """Clear chamber design form input values from session state to force reset to config defaults."""
    keys_to_clear = [
        # Chamber design form keys
        f"chamber_pc_design_{config_label}",
        f"chamber_thrust_design_{config_label}",
        f"chamber_force_coefficient_{config_label}",
        f"chamber_diameter_inner_{config_label}",
        f"chamber_diameter_exit_{config_label}",
        f"chamber_l_star_{config_label}",
        f"chamber_unit_system_{config_label}",
        # Config editor form keys (chamber length and Lstar)
        f"chamber_length_{config_label}",
        f"chamber_lstar_{config_label}",
        # Also clear old key without config_label for backward compatibility
        "chamber_lstar",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def load_config_state(uploaded_file) -> Tuple[PintleEngineConfig, str]:
    if "config_dict" not in st.session_state:
        st.session_state["config_dict"] = get_default_config_dict()
        st.session_state["config_label"] = str(CONFIG_PATH)

    if uploaded_file is not None:
        try:
            config_text = uploaded_file.getvalue().decode("utf-8")
            config_dict = yaml.safe_load(config_text)
            # Validate the config
            PintleEngineConfig(**config_dict)
            # Store the raw YAML dict to preserve all fields including optional ones
            st.session_state["config_dict"] = config_dict
            old_config_label = st.session_state.get("config_label", "default")
            st.session_state["config_label"] = uploaded_file.name
            # Clear form state for both old and new config labels to ensure clean reset
            _clear_chamber_design_form_state(old_config_label)
            _clear_chamber_design_form_state(uploaded_file.name)
            # Mark that config was just loaded so chamber design form can reset
            st.session_state["config_just_loaded"] = True
        except Exception as exc:
            raise ValueError(f"Failed to load uploaded configuration: {exc}") from exc

    try:
        # Create config object from the dict (this validates it)
        config_obj = PintleEngineConfig(**st.session_state["config_dict"])
        # Update session state with the validated config to ensure all fields are present
        # Use exclude_none=False to preserve all fields
        st.session_state["config_dict"] = config_obj.model_dump(exclude_none=False)
    except Exception as exc:
        raise ValueError(f"Invalid configuration state: {exc}") from exc

    ox_choice = detect_fluid_choice(config_obj.fluids["oxidizer"].model_dump())
    fuel_choice = detect_fluid_choice(config_obj.fluids["fuel"].model_dump())
    st.session_state.setdefault("oxidizer_choice", ox_choice)
    st.session_state.setdefault("fuel_choice", fuel_choice)
    st.session_state.setdefault("injector_choice", "Pintle")

    return config_obj, st.session_state["config_label"]


def config_editor(config: PintleEngineConfig) -> PintleEngineConfig:
    # Always use session_state["config_dict"] if it exists, as it's the source of truth
    # This ensures values loaded from YAML files are properly displayed
    if "config_dict" in st.session_state:
        working_copy = copy.deepcopy(st.session_state["config_dict"])
    else:
        # Fallback to the config object passed in (shouldn't normally happen)
        config_dict_fallback = config.model_dump(exclude_none=False)
        working_copy = copy.deepcopy(config_dict_fallback)
    if "pintle_geometry" in working_copy and "injector" not in working_copy:
        working_copy["injector"] = {
            "type": "pintle",
            "geometry": working_copy.pop("pintle_geometry"),
        }

    injector_block = working_copy.setdefault("injector", {"type": "pintle", "geometry": {}})
    current_type = injector_block.get("type", "pintle")
    injector_display = next((name for name, info in INJECTOR_OPTIONS.items() if info["type"] == current_type), "Pintle")

    st.sidebar.markdown("### Injector")
    injector_choice = st.sidebar.selectbox(
        "Injector Type",
        list(INJECTOR_OPTIONS.keys()),
        index=list(INJECTOR_OPTIONS.keys()).index(injector_display),
        key="injector_choice_select",
    )
    st.session_state["injector_choice"] = injector_choice
    injector_info = INJECTOR_OPTIONS[injector_choice]
    injector_type = injector_info["type"]
    st.sidebar.caption(injector_info["description"])

    previous_type = st.session_state.get("active_injector_type", current_type)
    if injector_block.get("type") != injector_type:
        injector_block["type"] = injector_type
        injector_block["geometry"] = copy.deepcopy(DEFAULT_GEOMETRIES.get(injector_type, {}))

    if previous_type != injector_type:
        for key in list(st.session_state.keys()):
            if key.startswith("geom_"):
                del st.session_state[key]
    st.session_state["active_injector_type"] = injector_type

    injector_dict = injector_block

    with st.sidebar.form("config_form"):
        st.markdown("### Propellants")

        ox_choice = st.selectbox(
            "Oxidizer",
            OXIDIZER_OPTIONS,
            index=OXIDIZER_OPTIONS.index(st.session_state.get("oxidizer_choice", "Custom")),
            key="oxidizer_choice",
        )
        ox = working_copy["fluids"]["oxidizer"]
        if ox_choice != "Custom" and ox_choice in FLUID_LIBRARY:
            ox.update(FLUID_LIBRARY[ox_choice])
        ox["name"] = ox_choice if ox_choice != "Custom" else ox.get("name", "Custom Oxidizer")
        ox["density"] = st.number_input("Oxidizer density [kg/m³]", min_value=200.0, max_value=3000.0, value=float(ox.get("density") or 1140.0))
        ox["viscosity"] = st.number_input("Oxidizer viscosity [Pa·s]", min_value=1e-5, max_value=1e-2, value=float(ox.get("viscosity") or 1.8e-4))
        ox["surface_tension"] = st.number_input("Oxidizer surface tension [N/m]", min_value=1e-3, max_value=0.05, value=float(ox.get("surface_tension") or 0.013))
        ox["vapor_pressure"] = st.number_input("Oxidizer vapor pressure [Pa]", min_value=0.0, max_value=3e6, value=float(ox.get("vapor_pressure") or 101325.0))

        fuel_choice = st.selectbox(
            "Fuel",
            FUEL_OPTIONS,
            index=FUEL_OPTIONS.index(st.session_state.get("fuel_choice", "Custom")),
            key="fuel_choice",
        )
        fuel = working_copy["fluids"]["fuel"]
        if fuel_choice != "Custom" and fuel_choice in FLUID_LIBRARY:
            fuel.update(FLUID_LIBRARY[fuel_choice])
        fuel["name"] = fuel_choice if fuel_choice != "Custom" else fuel.get("name", "Custom Fuel")
        fuel["density"] = st.number_input("Fuel density [kg/m³]", min_value=200.0, max_value=3000.0, value=float(fuel.get("density") or 780.0))
        fuel["viscosity"] = st.number_input("Fuel viscosity [Pa·s]", min_value=1e-5, max_value=1e-2, value=float(fuel.get("viscosity") or 2.0e-3))
        fuel["surface_tension"] = st.number_input("Fuel surface tension [N/m]", min_value=1e-3, max_value=0.05, value=float(fuel.get("surface_tension") or 0.025))
        fuel["vapor_pressure"] = st.number_input("Fuel vapor pressure [Pa]", min_value=0.0, max_value=3e6, value=float(fuel.get("vapor_pressure") or 1000.0))

        st.markdown("### Injector Geometry")
        if injector_type == "pintle":
            st.markdown("#### Pintle Geometry")
            pintle_geom = injector_dict.setdefault("geometry", copy.deepcopy(DEFAULT_GEOMETRIES["pintle"]))
            lox_geom = pintle_geom.setdefault("lox", {})
            fuel_geom = pintle_geom.setdefault("fuel", {})
            lox_geom.setdefault("n_orifices", 12)
            lox_geom.setdefault("d_orifice", 1.5e-3)
            fuel_geom.setdefault("d_pintle_tip", 0.02)
            fuel_geom.setdefault("h_gap", 0.0005)

            lox_geom["n_orifices"] = int(
                st.number_input(
                    "Number of LOX orifices",
                    min_value=1,
                    max_value=128,
                    value=int(lox_geom["n_orifices"]),
                    step=1,
                    format="%d",
                    key="geom_pintle_lox_n_orifices",
                )
            )
            lox_geom["d_orifice"] = length_number_input(
                "LOX orifice diameter",
                float(lox_geom["d_orifice"]),
                min_m=1e-5,
                max_m=5e-2,
                step_m=1e-5,
                key="geom_pintle_lox_d_orifice",
            )
            fuel_geom["d_pintle_tip"] = length_number_input(
                "Fuel pintle tip diameter",
                float(fuel_geom["d_pintle_tip"]),
                min_m=1e-3,
                max_m=0.1,
                step_m=1e-4,
                key="geom_pintle_fuel_tip_d",
            )
            fuel_geom["h_gap"] = length_number_input(
                "Fuel gap height",
                float(fuel_geom["h_gap"]),
                min_m=1e-5,
                max_m=5e-3,
                step_m=1e-5,
                key="geom_pintle_fuel_gap",
            )

        elif injector_type == "coaxial":
            st.markdown("#### Coaxial Geometry")
            coax_geom = injector_dict.setdefault("geometry", copy.deepcopy(DEFAULT_GEOMETRIES["coaxial"]))
            core_geom = coax_geom.setdefault("core", {})
            ann_geom = coax_geom.setdefault("annulus", {})

            core_geom.setdefault("n_ports", 12)
            core_geom.setdefault("d_port", 1.4e-3)
            core_geom.setdefault("length", 0.015)
            ann_geom.setdefault("inner_diameter", 5.0e-3)
            ann_geom.setdefault("gap_thickness", 8.0e-4)
            ann_geom.setdefault("swirl_angle", 20.0)

            core_geom["n_ports"] = int(
                st.number_input(
                    "Core ports",
                    min_value=1,
                    max_value=256,
                    value=int(core_geom["n_ports"]),
                    step=1,
                    format="%d",
                    key="geom_coaxial_core_ports",
                )
            )
            core_geom["d_port"] = length_number_input(
                "Core port diameter",
                float(core_geom["d_port"]),
                min_m=1e-5,
                max_m=2e-2,
                step_m=1e-5,
                key="geom_coaxial_core_d",
            )
            core_length_val = length_number_input(
                "Core port length (0 = auto)",
                float(core_geom.get("length") or 0.0),
                min_m=0.0,
                max_m=0.1,
                step_m=1e-4,
                key="geom_coaxial_core_length",
            )
            core_geom["length"] = core_length_val if core_length_val and core_length_val > 0 else None

            ann_geom["inner_diameter"] = length_number_input(
                "Annulus inner diameter",
                float(ann_geom["inner_diameter"]),
                min_m=1e-3,
                max_m=0.1,
                step_m=1e-4,
                key="geom_coaxial_annulus_id",
            )
            ann_geom["gap_thickness"] = length_number_input(
                "Annulus gap thickness",
                float(ann_geom["gap_thickness"]),
                min_m=1e-5,
                max_m=1e-2,
                step_m=1e-5,
                key="geom_coaxial_annulus_gap",
            )
            ann_geom["swirl_angle"] = st.slider(
                "Swirl angle [deg]",
                min_value=0.0,
                max_value=80.0,
                value=float(ann_geom["swirl_angle"]),
                step=1.0,
                key="geom_coaxial_swirl_angle",
            )

        elif injector_type == "impinging":
            st.markdown("#### Impinging Geometry")
            imp_geom = injector_dict.setdefault("geometry", copy.deepcopy(DEFAULT_GEOMETRIES["impinging"]))
            ox_geom = imp_geom.setdefault("oxidizer", {})
            fuel_geom_imp = imp_geom.setdefault("fuel", {})

            for branch, geom in ("Oxidizer", ox_geom), ("Fuel", fuel_geom_imp):
                geom.setdefault("n_elements", 8)
                geom.setdefault("d_jet", 1.2e-3)
                geom.setdefault("impingement_angle", 60.0)
                geom.setdefault("spacing", 4.0e-3)

                st.subheader(f"{branch} Jets")
                branch_key = branch.lower().replace(" ", "_")
                geom["n_elements"] = int(
                    st.number_input(
                        f"{branch} elements",
                        min_value=1,
                        max_value=128,
                        value=int(geom["n_elements"]),
                        step=1,
                        format="%d",
                        key=f"geom_impinging_{branch_key}_elements",
                    )
                )
                geom["d_jet"] = length_number_input(
                    f"{branch} jet diameter",
                    float(geom["d_jet"]),
                    min_m=1e-5,
                    max_m=1e-2,
                    step_m=1e-5,
                    key=f"geom_impinging_{branch_key}_jet_d",
                )
                geom["impingement_angle"] = st.slider(
                    f"{branch} impingement angle [deg]",
                    min_value=20.0,
                    max_value=180.0,
                    value=float(geom["impingement_angle"]),
                    step=1.0,
                    key=f"geom_impinging_{branch_key}_imp_angle",
                )
                geom["spacing"] = length_number_input(
                    f"{branch} jet spacing",
                    float(geom["spacing"]),
                    min_m=1e-4,
                    max_m=0.05,
                    step_m=1e-4,
                    key=f"geom_impinging_{branch_key}_spacing",
                )

        st.markdown("### Feed System")
        feed = working_copy["feed_system"]
        ox_feed = feed["oxidizer"]
        fuel_feed = feed["fuel"]
        ox_feed["d_inlet"] = length_number_input(
            "LOX inlet diameter",
            float(ox_feed["d_inlet"]),
            min_m=1e-3,
            max_m=0.05,
            step_m=1e-4,
            key="feed_lox_d_inlet",
        )
        ox_feed["K0"] = st.number_input("LOX loss coefficient K0", min_value=0.0, max_value=10.0, value=float(ox_feed["K0"]))
        fuel_feed["d_inlet"] = length_number_input(
            "Fuel inlet diameter",
            float(fuel_feed["d_inlet"]),
            min_m=1e-3,
            max_m=0.05,
            step_m=1e-4,
            key="feed_fuel_d_inlet",
        )
        fuel_feed["K0"] = st.number_input("Fuel loss coefficient K0", min_value=0.0, max_value=10.0, value=float(fuel_feed["K0"]))

        st.markdown("### Regenerative Cooling")
        regen = working_copy["regen_cooling"]
        regen["enabled"] = st.checkbox("Enable regenerative cooling", value=bool(regen["enabled"]))
        regen["n_channels"] = int(st.number_input("Channels", min_value=1, max_value=400, value=int(regen["n_channels"])) )
        regen["channel_width"] = length_number_input(
            "Channel width",
            float(regen["channel_width"]),
            min_m=1e-4,
            max_m=5e-3,
            step_m=1e-4,
            key="regen_channel_width",
        )
        regen["channel_height"] = length_number_input(
            "Channel height",
            float(regen["channel_height"]),
            min_m=1e-4,
            max_m=5e-3,
            step_m=1e-4,
            key="regen_channel_height",
        )
        regen["use_heat_transfer"] = st.checkbox("Enable coupled heat-transfer", value=bool(regen.get("use_heat_transfer", False)))
        regen["wall_thickness"] = length_number_input(
            "Wall thickness",
            float(regen.get("wall_thickness", 0.002)),
            min_m=1e-4,
            max_m=0.02,
            step_m=1e-4,
            key="regen_wall_thickness",
        )
        regen["wall_thermal_conductivity"] = st.number_input("Wall conductivity [W/(m·K)]", min_value=10.0, max_value=600.0, value=float(regen.get("wall_thermal_conductivity", 300.0)))
        regen["chamber_inner_diameter"] = length_number_input(
            "Chamber inner diameter",
            float(regen.get("chamber_inner_diameter", 0.08)),
            min_m=0.01,
            max_m=0.5,
            step_m=1e-3,
            key="regen_chamber_inner_d",
        )

        st.markdown("### Film Cooling")
        film_cfg = working_copy.setdefault("film_cooling", {
            "enabled": False,
            "mass_fraction": 0.05,
            "injection_temperature": None,
            "effectiveness_ref": 0.4,
            "decay_length": 0.1,
            "apply_to_fraction_of_length": 1.0,
            "slot_height": 3.0e-4,
            "reference_blowing_ratio": 0.5,
            "blowing_exponent": 0.6,
            "turbulence_reference_intensity": 0.08,
            "turbulence_sensitivity": 1.0,
            "turbulence_exponent": 1.0,
            "turbulence_min_multiplier": 0.5,
            "reference_wall_temperature": 1100.0,
            "density_override": None,
            "cp_override": None,
        })
        film_cfg["enabled"] = st.checkbox("Enable film cooling", value=bool(film_cfg.get("enabled", False)), key="film_enabled")
        film_cfg["mass_fraction"] = st.number_input(
            "Film mass fraction (of fuel)",
            min_value=0.0,
            max_value=0.5,
            value=float(film_cfg.get("mass_fraction", 0.05)),
            key="film_mass_fraction",
        )
        film_cfg["effectiveness_ref"] = st.slider(
            "Reference effectiveness",
            min_value=0.0,
            max_value=1.0,
            value=float(film_cfg.get("effectiveness_ref", 0.4)),
            step=0.01,
            key="film_effectiveness_ref",
        )
        film_cfg["decay_length"] = length_number_input(
            "Effectiveness decay length",
            float(film_cfg.get("decay_length", 0.1)),
            min_m=0.005,
            max_m=2.0,
            step_m=0.01,
            key="film_decay_length",
        )
        film_cfg["apply_to_fraction_of_length"] = st.number_input(
            "Fraction of chamber length covered",
            min_value=0.1,
            max_value=1.5,
            value=float(film_cfg.get("apply_to_fraction_of_length", 1.0)),
            key="film_length_fraction",
        )
        film_cfg["slot_height"] = length_number_input(
            "Film slot height",
            float(film_cfg.get("slot_height", 3.0e-4)),
            min_m=5e-5,
            max_m=5e-3,
            step_m=5e-5,
            key="film_slot_height",
        )
        film_cfg["reference_blowing_ratio"] = st.number_input(
            "Reference blowing ratio",
            min_value=0.05,
            max_value=5.0,
            value=float(film_cfg.get("reference_blowing_ratio", 0.5)),
            key="film_blowing_ratio",
        )
        film_cfg["blowing_exponent"] = st.number_input(
            "Blowing exponent",
            min_value=0.1,
            max_value=2.0,
            value=float(film_cfg.get("blowing_exponent", 0.6)),
            key="film_blowing_exponent",
        )
        film_cfg["turbulence_reference_intensity"] = st.number_input(
            "Reference turbulence intensity",
            min_value=0.01,
            max_value=0.5,
            value=float(film_cfg.get("turbulence_reference_intensity", 0.08)),
            step=0.01,
            key="film_turbulence_reference",
        )
        film_cfg["turbulence_sensitivity"] = st.number_input(
            "Turbulence sensitivity",
            min_value=0.0,
            max_value=5.0,
            value=float(film_cfg.get("turbulence_sensitivity", 1.0)),
            step=0.1,
            key="film_turbulence_sensitivity",
        )
        film_cfg["turbulence_exponent"] = st.number_input(
            "Turbulence exponent",
            min_value=0.1,
            max_value=3.0,
            value=float(film_cfg.get("turbulence_exponent", 1.0)),
            step=0.1,
            key="film_turbulence_exponent",
        )
        film_cfg["turbulence_min_multiplier"] = st.number_input(
            "Minimum effectiveness multiplier",
            min_value=0.1,
            max_value=1.0,
            value=float(film_cfg.get("turbulence_min_multiplier", 0.5)),
            step=0.05,
            key="film_turbulence_min_multiplier",
        )
        film_cfg["reference_wall_temperature"] = st.number_input(
            "Reference wall temperature [K]",
            min_value=300.0,
            max_value=4000.0,
            value=float(film_cfg.get("reference_wall_temperature", 1100.0)),
            key="film_reference_wall_temperature",
        )
        film_temp_override = film_cfg.get("injection_temperature")
        film_cfg["injection_temperature"] = st.number_input(
            "Film injection temperature override [K] (0 = use fuel temp)",
            min_value=0.0,
            max_value=2000.0,
            value=float(film_temp_override or 0.0),
            key="film_injection_temp",
        ) or None
        density_override_val = st.number_input(
            "Film density override [kg/m³] (0 = use fuel density)",
            min_value=0.0,
            max_value=2000.0,
            value=float(film_cfg.get("density_override", 0.0) or 0.0),
            key="film_density_override",
        )
        film_cfg["density_override"] = density_override_val or None
        cp_override_val = st.number_input(
            "Film specific heat override [J/(kg·K)] (0 = use fuel cp)",
            min_value=0.0,
            max_value=6000.0,
            value=float(film_cfg.get("cp_override", 0.0) or 0.0),
            key="film_cp_override",
        )
        film_cfg["cp_override"] = cp_override_val or None

        st.markdown("### Ablative Cooling")
        ablative_cfg = working_copy.setdefault("ablative_cooling", {
            "enabled": False,
            "material_density": 1600.0,
            "heat_of_ablation": 2.5e6,
            "thermal_conductivity": 0.35,
            "specific_heat": 1500.0,
            "initial_thickness": 0.01,
            "surface_temperature_limit": 1200.0,
            "coverage_fraction": 1.0,
            "pyrolysis_temperature": 900.0,
            "blowing_efficiency": 0.8,
            "use_physics_based_blowing": True,
            "blowing_coefficient": 0.5,
            "blowing_min_reduction_factor": 0.1,
            "surface_emissivity": 0.85,
            "ambient_temperature": 300.0,
            "radiative_sink_minimum_threshold": 400.0,
            "radiative_sink_fallback_temperature": 600.0,
            "turbulence_reference_intensity": 0.08,
            "turbulence_sensitivity": 1.5,
            "turbulence_exponent": 1.0,
            "turbulence_max_multiplier": 3.0,
        })
        ablative_cfg["enabled"] = st.checkbox("Enable ablative liner", value=bool(ablative_cfg.get("enabled", False)), key="ablative_enabled")
        ablative_cfg["material_density"] = st.number_input("Ablator density [kg/m³]", min_value=200.0, max_value=4000.0, value=float(ablative_cfg.get("material_density", 1600.0)))
        ablative_cfg["heat_of_ablation"] = st.number_input("Heat of ablation [J/kg]", min_value=1e6, max_value=1e8, value=float(ablative_cfg.get("heat_of_ablation", 2.5e6)))
        ablative_cfg["thermal_conductivity"] = st.number_input("Ablator conductivity [W/(m·K)]", min_value=0.05, max_value=5.0, value=float(ablative_cfg.get("thermal_conductivity", 0.35)), key="ablative_conductivity")
        ablative_cfg["specific_heat"] = st.number_input("Ablator specific heat [J/(kg·K)]", min_value=200.0, max_value=4000.0, value=float(ablative_cfg.get("specific_heat", 1500.0)), key="ablative_specific_heat")
        ablative_cfg["initial_thickness"] = length_number_input(
            "Initial thickness",
            float(ablative_cfg.get("initial_thickness", 0.01)),
            min_m=0.001,
            max_m=0.05,
            step_m=0.001,
            key="ablative_thickness",
        )
        ablative_cfg["surface_temperature_limit"] = st.number_input("Surface temperature limit [K]", min_value=500.0, max_value=2500.0, value=float(ablative_cfg.get("surface_temperature_limit", 1200.0)), key="ablative_surface_temp")
        ablative_cfg["coverage_fraction"] = st.number_input("Surface coverage fraction", min_value=0.1, max_value=1.0, value=float(ablative_cfg.get("coverage_fraction", 1.0)), key="ablative_coverage_fraction")
        ablative_cfg["pyrolysis_temperature"] = st.number_input("Pyrolysis temperature [K]", min_value=300.0, max_value=2000.0, value=float(ablative_cfg.get("pyrolysis_temperature", 900.0)), key="ablative_pyrolysis_temperature")
        st.markdown("**Blowing Effect (Pyrolysis Gas Protection):**")
        ablative_cfg["use_physics_based_blowing"] = st.checkbox(
            "Use physics-based blowing (B = m_dot_pyrolysis/m_dot_external)",
            value=bool(ablative_cfg.get("use_physics_based_blowing", True)),
            help="If enabled, computes blowing parameter B from actual pyrolysis mass flux. If disabled, uses constant blowing_efficiency factor.",
            key="ablative_use_physics_blowing"
        )
        
        if ablative_cfg["use_physics_based_blowing"]:
            ablative_cfg["blowing_coefficient"] = st.number_input(
                "Blowing coefficient c",
                min_value=0.1,
                max_value=2.0,
                value=float(ablative_cfg.get("blowing_coefficient", 0.5)),
                step=0.1,
                help="Coefficient in f(B) = 1/(1 + c*B). Typical range: 0.3-0.8. Higher = stronger blowing effect.",
                key="ablative_blowing_coefficient"
            )
            ablative_cfg["blowing_min_reduction_factor"] = st.number_input(
                "Minimum reduction factor (max blowing effectiveness)",
                min_value=0.01,
                max_value=0.5,
                value=float(ablative_cfg.get("blowing_min_reduction_factor", 0.1)),
                step=0.01,
                help="Minimum fraction of convective heat that remains (max reduction = 1 - this value). Default 0.1 = max 90% reduction.",
                key="ablative_blowing_min_reduction"
            )
        else:
            ablative_cfg["blowing_efficiency"] = st.number_input(
                "Blowing efficiency (legacy constant factor)",
                min_value=0.0,
                max_value=1.0,
                value=float(ablative_cfg.get("blowing_efficiency", 0.8)),
                help="Constant effectiveness factor (used when physics-based blowing is disabled)",
                key="ablative_blowing_efficiency"
            )
        
        st.markdown("**Radiative Heat Transfer:**")
        ablative_cfg["surface_emissivity"] = st.number_input(
            "Surface emissivity",
            min_value=0.0,
            max_value=1.0,
            value=float(ablative_cfg.get("surface_emissivity", 0.85)),
            step=0.05,
            help="Emissivity for radiative heat transfer (0-1, typical 0.8-0.9 for charred ablators)",
            key="ablative_surface_emissivity"
        )
        ablative_cfg["ambient_temperature"] = st.number_input(
            "Ambient temperature [K]",
            min_value=100.0,
            max_value=1000.0,
            value=float(ablative_cfg.get("ambient_temperature", 300.0)),
            step=10.0,
            help="Ambient/surrounding temperature for radiative heat transfer. For radiation to space, use ~300K.",
            key="ablative_ambient_temp"
        )
        ablative_cfg["radiative_sink_minimum_threshold"] = st.number_input(
            "Radiative sink minimum threshold [K]",
            min_value=200.0,
            max_value=600.0,
            value=float(ablative_cfg.get("radiative_sink_minimum_threshold", 400.0)),
            step=10.0,
            help="If ambient_temperature is below this, uses fallback temperature (heated steel layer)",
            key="ablative_rad_sink_min"
        )
        ablative_cfg["radiative_sink_fallback_temperature"] = st.number_input(
            "Radiative sink fallback temperature [K]",
            min_value=400.0,
            max_value=1000.0,
            value=float(ablative_cfg.get("radiative_sink_fallback_temperature", 600.0)),
            step=10.0,
            help="Fallback temperature when ambient is too low (represents heated steel layer behind ablator)",
            key="ablative_rad_sink_fallback"
        )
        
        st.markdown("**Turbulence Effects:**")
        ablative_cfg["turbulence_reference_intensity"] = st.number_input("Reference turbulence intensity", min_value=0.01, max_value=0.5, value=float(ablative_cfg.get("turbulence_reference_intensity", 0.08)), step=0.01, key="ablative_turbulence_reference")
        ablative_cfg["turbulence_sensitivity"] = st.number_input("Turbulence sensitivity", min_value=0.0, max_value=5.0, value=float(ablative_cfg.get("turbulence_sensitivity", 1.5)), step=0.1, key="ablative_turbulence_sensitivity")
        ablative_cfg["turbulence_exponent"] = st.number_input("Turbulence exponent", min_value=0.1, max_value=3.0, value=float(ablative_cfg.get("turbulence_exponent", 1.0)), step=0.1, key="ablative_turbulence_exponent")
        ablative_cfg["turbulence_max_multiplier"] = st.number_input("Max heat-flux multiplier", min_value=1.0, max_value=5.0, value=float(ablative_cfg.get("turbulence_max_multiplier", 3.0)), step=0.1, key="ablative_turbulence_multiplier")
        
        st.markdown("**Geometry Evolution (Time-Varying L*):**")
        ablative_cfg["track_geometry_evolution"] = st.checkbox(
            "Enable geometry evolution tracking", 
            value=bool(ablative_cfg.get("track_geometry_evolution", True)),
            help="Track cumulative recession and update chamber/throat geometry over time",
            key="ablative_track_geometry"
        )
        
        throat_mult_raw = ablative_cfg.get("throat_recession_multiplier", None)
        use_physics_mult = st.checkbox(
            "Use physics-based throat multiplier (Bartz correlation)",
            value=(throat_mult_raw is None),
            help="If unchecked, you can specify a fixed multiplier (typically 1.2-2.0)",
            key="ablative_use_physics_mult"
        )
        
        if use_physics_mult:
            ablative_cfg["throat_recession_multiplier"] = None
            st.info("Throat multiplier calculated from flow conditions (velocity & pressure ratios)")
        else:
            ablative_cfg["throat_recession_multiplier"] = st.number_input(
                "Throat recession multiplier",
                min_value=1.0,
                max_value=3.0,
                value=float(throat_mult_raw if throat_mult_raw is not None else 1.3),
                step=0.1,
                help="Throat recedes faster than chamber (typically 1.2-2.0x)",
                key="ablative_throat_mult_fixed"
            )
        
        ablative_cfg["char_layer_conductivity"] = st.number_input(
            "Char layer conductivity [W/(m·K)]",
            min_value=0.01,
            max_value=2.0,
            value=float(ablative_cfg.get("char_layer_conductivity", 0.2)),
            step=0.01,
            help="Thermal conductivity of protective char layer",
            key="ablative_char_conductivity"
        )
        ablative_cfg["char_layer_thickness"] = length_number_input(
            "Char layer thickness",
            float(ablative_cfg.get("char_layer_thickness", 0.001)),
            min_m=0.0001,
            max_m=0.01,
            step_m=0.0001,
            key="ablative_char_thickness",
        )
        
        ablative_cfg["nozzle_ablative"] = st.checkbox(
            "Enable nozzle exit ablation",
            value=bool(ablative_cfg.get("nozzle_ablative", False)),
            help="If enabled, nozzle exit also recedes (A_exit grows). If disabled, only throat recedes (expansion ratio decreases)",
            key="ablative_nozzle_ablative"
        )

        st.markdown("### Chamber & Nozzle")
        chamber = working_copy["chamber"]
        nozzle = working_copy["nozzle"]
        config_label = st.session_state.get("config_label", "default")
        
        # Get values directly from config FIRST - ALWAYS pull from config, never calculate
        chamber_volume = chamber.get("volume")
        if chamber_volume is None:
            chamber_volume = 1e-3
        chamber_A_throat = chamber.get("A_throat")
        if chamber_A_throat is None:
            chamber_A_throat = 0.000857892  # Default throat area
        chamber_length = chamber.get("length")
        if chamber_length is None:
            chamber_length = 0.5
        chamber_lstar = chamber.get("Lstar")
        if chamber_lstar is None:
            chamber_lstar = 0.5
        
        # Make keys include a hash of the config values so they change when config changes
        # This makes length and Lstar work identically to volume and A_throat (which have no keys)
        # When config changes, the hash changes, so Streamlit treats it as a new widget and uses the value parameter
        import hashlib
        # Hash the chamber section of config to create unique keys that change when config changes
        chamber_str = str(sorted(chamber.items())) if chamber else ""
        config_hash = hashlib.md5(f"{config_label}_{chamber_str}".encode()).hexdigest()[:8]
        length_key = f"chamber_length_{config_label}_{config_hash}"
        lstar_key = f"chamber_lstar_{config_label}_{config_hash}"
        
        # Now render inputs - they will use config values since keys are cleared if values don't match
        chamber["volume"] = st.number_input("Chamber volume [m³]", min_value=1e-6, max_value=1.0, value=float(chamber_volume), format="%.6f")
        chamber["A_throat"] = st.number_input("Throat area [m²]", min_value=1e-5, max_value=0.01, value=float(chamber_A_throat), format="%.6f")
        chamber["length"] = length_number_input(
            "Chamber length",
            float(chamber_length),
            min_m=0.01,
            max_m=3.0,
            step_m=0.01,
            key=length_key,
        )
        chamber["Lstar"] = length_number_input(
            "Characteristic length L*",
            float(chamber_lstar),
            min_m=0.1,
            max_m=5.0,
            step_m=0.05,
            key=lstar_key,
        )
        nozzle_expansion_ratio = nozzle.get("expansion_ratio")
        if nozzle_expansion_ratio is None:
            nozzle_expansion_ratio = 6.5  # Default expansion ratio
        nozzle["expansion_ratio"] = st.number_input("Expansion ratio (Ae/At)", min_value=1.0, max_value=200.0, value=float(nozzle_expansion_ratio), format="%.4f")

        st.markdown("### Combustion & Efficiency")
        st.info(
            "**Note:** L* (characteristic length) is set in Chamber Geometry. "
            "If coupling is enabled with high efficiency floors, L* changes may have minimal visible effect on performance. "
            "Disable coupling or lower the floors below to observe L* impact more clearly."
        )
        combustion = working_copy.setdefault("combustion", {})
        efficiency = combustion.setdefault("efficiency", {})
        eff_model = efficiency.get("model", "exponential")
        model_options = ["exponential", "linear", "constant"]
        eff_model = st.selectbox(
            "η₍c*₎ model",
            model_options,
            index=model_options.index(eff_model) if eff_model in model_options else 0,
            key="comb_eff_model",
        )
        efficiency["model"] = eff_model
        efficiency["C"] = st.number_input(
            "Efficiency constant C",
            min_value=0.0,
            max_value=1.0,
            value=float(efficiency.get("C", 0.3)),
            key="comb_eff_C",
        )
        efficiency["K"] = st.number_input(
            "Efficiency exponent K",
            min_value=0.0,
            max_value=2.0,
            value=float(efficiency.get("K", 0.15)),
            key="comb_eff_K",
        )

        st.markdown("#### Efficiency Coupling")
        col_eff1, col_eff2, col_eff3 = st.columns(3)
        efficiency["use_mixture_coupling"] = col_eff1.checkbox(
            "Mixture coupling",
            value=bool(efficiency.get("use_mixture_coupling", False)),
            key="comb_eff_use_mixture",
        )
        efficiency["use_cooling_coupling"] = col_eff2.checkbox(
            "Cooling coupling",
            value=bool(efficiency.get("use_cooling_coupling", False)),
            key="comb_eff_use_cooling",
        )
        efficiency["use_turbulence_coupling"] = col_eff3.checkbox(
            "Turbulence coupling",
            value=bool(efficiency.get("use_turbulence_coupling", False)),
            key="comb_eff_use_turbulence",
        )
        
        st.markdown("#### Efficiency Floors")
        col_floor1, col_floor2, col_floor3 = st.columns(3)
        efficiency["mixture_efficiency_floor"] = col_floor1.number_input(
            "Mixture floor",
            min_value=0.0,
            max_value=1.0,
            value=float(efficiency.get("mixture_efficiency_floor", 0.25)),
            help="Minimum mixture efficiency (0 = no floor, 1 = always 100%)",
            key="comb_eff_mix_floor_main",
        )
        efficiency["cooling_efficiency_floor"] = col_floor2.number_input(
            "Cooling floor",
            min_value=0.0,
            max_value=1.0,
            value=float(efficiency.get("cooling_efficiency_floor", 0.25)),
            help="Minimum cooling efficiency (0 = no floor, 1 = always 100%)",
            key="comb_eff_cooling_floor_main",
        )
        efficiency["turbulence_efficiency_floor"] = col_floor3.number_input(
            "Turbulence floor",
            min_value=0.0,
            max_value=1.0,
            value=float(efficiency.get("turbulence_efficiency_floor", 0.3)),
            help="Minimum turbulence efficiency (0 = no floor, 1 = always 100%)",
            key="comb_eff_turb_floor_main",
        )

        with st.expander("Advanced mixing & turbulence parameters"):
            efficiency["target_smd_microns"] = st.number_input(
                "Target SMD [μm]",
                min_value=1.0,
                max_value=200.0,
                value=float(efficiency.get("target_smd_microns", 40.0)),
                key="comb_eff_target_smd",
            )
            efficiency["smd_penalty_exponent"] = st.number_input(
                "SMD penalty exponent",
                min_value=0.0,
                max_value=5.0,
                value=float(efficiency.get("smd_penalty_exponent", 1.5)),
                key="comb_eff_smd_exp",
            )
            efficiency["xstar_limit_mm"] = st.number_input(
                "Evaporation limit x* [mm]",
                min_value=1.0,
                max_value=200.0,
                value=float(efficiency.get("xstar_limit_mm", 40.0)),
                key="comb_eff_xstar_limit",
            )
            efficiency["xstar_penalty_exponent"] = st.number_input(
                "x* penalty exponent",
                min_value=0.0,
                max_value=5.0,
                value=float(efficiency.get("xstar_penalty_exponent", 1.0)),
                key="comb_eff_xstar_exp",
            )
            efficiency["we_reference"] = st.number_input(
                "Reference Weber number",
                min_value=1.0,
                max_value=500.0,
                value=float(efficiency.get("we_reference", 25.0)),
                key="comb_eff_we_ref",
            )
            efficiency["we_penalty_exponent"] = st.number_input(
                "Weber penalty exponent",
                min_value=0.0,
                max_value=5.0,
                value=float(efficiency.get("we_penalty_exponent", 1.0)),
                key="comb_eff_we_exp",
            )
            efficiency["target_turbulence_intensity"] = st.number_input(
                "Target turbulence intensity",
                min_value=0.01,
                max_value=0.5,
                value=float(efficiency.get("target_turbulence_intensity", 0.08)),
                key="comb_eff_turb_target",
            )
            efficiency["turbulence_penalty_exponent"] = st.number_input(
                "Turbulence penalty exponent",
                min_value=0.0,
                max_value=5.0,
                value=float(efficiency.get("turbulence_penalty_exponent", 1.0)),
                key="comb_eff_turb_exp",
            )

        submitted = st.form_submit_button("Apply configuration changes")

    if submitted:
        try:
            working_copy["combustion"]["cea"]["ox_name"] = working_copy["fluids"]["oxidizer"].get("name", "Oxidizer")
            working_copy["combustion"]["cea"]["fuel_name"] = working_copy["fluids"]["fuel"].get("name", "Fuel")
            new_config = PintleEngineConfig(**working_copy)
            st.session_state["config_dict"] = working_copy
            # Clear form state to ensure inputs reset to new config values
            config_label = st.session_state.get("config_label", "default")
            _clear_chamber_design_form_state(config_label)
            # Mark that config was updated from config_editor so chamber design form can reset
            st.session_state["config_updated_from_editor"] = True
            st.session_state["config_updated"] = True
            st.success("Configuration updated. Chamber Design tab will refresh with new values.")
            st.rerun()  # Force rerun to update chamber design tab
            return new_config
        except Exception as exc:
            st.error(f"Invalid configuration: {exc}")
            return config

    return PintleEngineConfig(**st.session_state["config_dict"])


def chamber_design_view(config_obj: PintleEngineConfig, runner: Optional[PintleEngineRunner] = None) -> PintleEngineConfig:
    """Chamber Design tab - calculates chamber geometry and updates config."""
    st.header("Chamber Design")
    st.markdown("Calculate chamber geometry from design parameters and update configuration.")
    
    # Get config label for unique form keys
    config_label = st.session_state.get("config_label", "default")
    
    # Check if config was just updated or loaded - if so, clear form state to force reset to new defaults
    if st.session_state.get("config_updated_from_editor", False) or st.session_state.get("config_just_loaded", False):
        _clear_chamber_design_form_state(config_label)
        # Clear the flags after clearing form state
        if "config_just_loaded" in st.session_state:
            del st.session_state["config_just_loaded"]
        if "config_updated_from_editor" in st.session_state:
            del st.session_state["config_updated_from_editor"]
    
    # Check if we have stored calculation results from a previous run
    has_stored_results = "chamber_calc_results" in st.session_state
    stored_results = None
    if has_stored_results:
        stored_results = st.session_state["chamber_calc_results"]
    
    # Get current config values for defaults
    config_dict = config_obj.model_dump()
    
    # Conversion factors (use module-level PSI_TO_PA and PA_TO_PSI)
    LBF_TO_N = 4.44822
    N_TO_LBF = 1.0 / LBF_TO_N
    IN_TO_M = 0.0254
    M_TO_IN = 1.0 / IN_TO_M
    
    # Get default values from config if they exist (in metric)
    pc_design_default_metric = 2.068e6  # Pa (300 psi default)
    thrust_design_default_metric = 6000.0  # N
    force_coefficient_default = 1.4
    diameter_inner_default_metric = 0.08636  # m (3.4 inches)
    diameter_exit_default_metric = 0.1016  # m (4 inches)
    l_star_default_metric = 1.27  # m
    MR_default = 2.5  # Default mixture ratio
    
    # Extract values from config
    chamber = config_dict.get("chamber", {})
    nozzle = config_dict.get("nozzle", {})
    regen = config_dict.get("regen_cooling", {})
    
    # Get L* from config (or calculate from volume/A_throat if available)
    if chamber.get("Lstar") is not None:
        l_star_default_metric = float(chamber["Lstar"])
    elif chamber.get("volume") is not None and chamber.get("A_throat") is not None:
        volume = float(chamber["volume"])
        A_throat = float(chamber["A_throat"])
        if volume > 0 and A_throat > 0:
            l_star_default_metric = volume / A_throat
    
    # Get design pressure from config
    if chamber.get("design_pressure") is not None:
        pc_design_default_metric = float(chamber["design_pressure"])
    
    # Get design thrust from config
    if chamber.get("design_thrust") is not None:
        thrust_design_default_metric = float(chamber["design_thrust"])
    
    # Get force coefficient from config
    if chamber.get("design_force_coefficient") is not None:
        force_coefficient_default = float(chamber["design_force_coefficient"])
    
    # Get chamber inner diameter from config - check chamber.chamber_inner_diameter first, then regen_cooling, then calculate
    if chamber.get("chamber_inner_diameter") is not None:
        diameter_inner_default_metric = float(chamber["chamber_inner_diameter"])
    elif regen.get("chamber_inner_diameter") is not None:
        diameter_inner_default_metric = float(regen["chamber_inner_diameter"])
    elif chamber.get("volume") is not None and chamber.get("length") is not None:
        volume = float(chamber["volume"])
        length = float(chamber["length"])
        if volume > 0 and length > 0:
            area = volume / length
            diameter_inner_default_metric = np.sqrt(4.0 * area / np.pi)
    
    # Get exit diameter from config - check nozzle.exit_diameter first, then A_exit, then calculate
    if nozzle.get("exit_diameter") is not None:
        diameter_exit_default_metric = float(nozzle["exit_diameter"])
    elif nozzle.get("A_exit") is not None:
        A_exit = float(nozzle["A_exit"])
        diameter_exit_default_metric = np.sqrt(4.0 * A_exit / np.pi)
    elif nozzle.get("expansion_ratio") is not None and chamber.get("A_throat") is not None:
        expansion_ratio = float(nozzle["expansion_ratio"])
        A_throat = float(chamber["A_throat"])
        if expansion_ratio > 0 and A_throat > 0:
            A_exit = expansion_ratio * A_throat
            diameter_exit_default_metric = np.sqrt(4.0 * A_exit / np.pi)
    
    # Input section
    # Note: Streamlit forms submit on Enter by default - user must click button to calculate
    st.info("💡 **Tip:** Press Enter in any field to update values, but click the button below to calculate geometry.")
    
    # Move radio buttons outside form so they trigger immediate reruns
    # Read current values first (with defaults) to use in hash
    import hashlib
    base_config_str = f"{pc_design_default_metric}_{thrust_design_default_metric}_{force_coefficient_default}_{diameter_inner_default_metric}_{diameter_exit_default_metric}_{l_star_default_metric}"
    base_hash = hashlib.md5(f"{config_label}_{base_config_str}".encode()).hexdigest()[:8]
    
    st.subheader("Calculation Method")
    
    # Method selector: Manual Force Coefficient vs CEA-Based Solver
    calculation_method = st.radio(
        "Calculation Method",
        ["Manual Force Coefficient", "CEA-Based Solver (Automatic Cf)"],
        horizontal=True,
        help="Manual: Specify force coefficient directly. CEA-Based: Automatically calculates optimal Cf from thermochemistry.",
        key=f"chamber_calc_method_{config_label}_{base_hash}"
    )
    
    # Check if CEA solver is available
    use_cea_solver = (calculation_method == "CEA-Based Solver (Automatic Cf)" and 
                     solve_chamber_geometry_with_cea is not None and 
                     runner is not None and 
                     hasattr(runner, 'cea_cache'))
    
    if calculation_method == "CEA-Based Solver (Automatic Cf)":
        if solve_chamber_geometry_with_cea is None:
            st.error("⚠️ CEA-based solver not available. Please use 'Manual Force Coefficient' method.")
            calculation_method = "Manual Force Coefficient"
            use_cea_solver = False
        elif runner is None or not hasattr(runner, 'cea_cache'):
            st.warning("⚠️ Runner or CEA cache not available. Please ensure engine is configured. Falling back to manual method.")
            calculation_method = "Manual Force Coefficient"
            use_cea_solver = False
        else:
            st.info("ℹ️ CEA-based solver will automatically calculate the optimal thrust coefficient (Cf) from thermochemistry.")
    
    st.subheader("Design Parameters")
    
    # Unit system toggle - outside form for immediate updates
    unit_system = st.radio(
        "Unit System",
        ["Metric", "Imperial"],
        horizontal=True,
        key=f"chamber_unit_system_{config_label}_{base_hash}"
    )
    
    # Create hash that includes unit system and calculation method so form inputs reset when these change
    chamber_design_config_str = f"{base_config_str}_{calculation_method}_{unit_system}"
    config_hash = hashlib.md5(f"{config_label}_{chamber_design_config_str}".encode()).hexdigest()[:8]
    
    with st.form(f"chamber_design_form_{config_label}_{config_hash}", clear_on_submit=False):
        
        # Convert defaults based on unit system
        if unit_system == "Imperial":
            pc_design_default = pc_design_default_metric * PA_TO_PSI  # PSI (using module-level constant)
            thrust_design_default = thrust_design_default_metric * N_TO_LBF  # lbf
            diameter_inner_default = diameter_inner_default_metric * M_TO_IN  # inches
            diameter_exit_default = diameter_exit_default_metric * M_TO_IN  # inches
            l_star_default = l_star_default_metric * M_TO_IN  # inches
            
            # Pressure limits in PSI
            pc_min = 100000.0 * PA_TO_PSI
            pc_max = 20e6 * PA_TO_PSI
            pc_step = 10000.0 * PA_TO_PSI
            
            # Thrust limits in lbf
            thrust_min = 100.0 * N_TO_LBF
            thrust_max = 100000.0 * N_TO_LBF
            thrust_step = 100.0 * N_TO_LBF
            
            # Length limits in inches
            length_min = 0.01 * M_TO_IN
            length_max = 1.0 * M_TO_IN
            length_step = 0.001 * M_TO_IN
            l_star_min = 0.1 * M_TO_IN
            l_star_max = 5.0 * M_TO_IN
            l_star_step = 0.01 * M_TO_IN
            
            pc_unit = "PSI"
            thrust_unit = "lbf"
            length_unit = "in"
        else:  # Metric
            pc_design_default = pc_design_default_metric  # Pa
            thrust_design_default = thrust_design_default_metric  # N
            diameter_inner_default = diameter_inner_default_metric  # m
            diameter_exit_default = diameter_exit_default_metric  # m
            l_star_default = l_star_default_metric  # m
            
            pc_min = 100000.0
            pc_max = 20e6
            pc_step = 10000.0
            
            thrust_min = 100.0
            thrust_max = 100000.0
            thrust_step = 100.0
            
            length_min = 0.01
            length_max = 1.0
            length_step = 0.001
            l_star_min = 0.1
            l_star_max = 5.0
            l_star_step = 0.01
            
            pc_unit = "Pa"
            thrust_unit = "N"
            length_unit = "m"
        
        col1, col2 = st.columns(2)
        with col1:
            pc_design = st.number_input(
                f"Design Chamber Pressure [{pc_unit}]",
                min_value=pc_min,
                max_value=pc_max,
                value=pc_design_default,
                step=pc_step,
                format="%.2f" if unit_system == "Imperial" else "%.0f",
                key=f"chamber_pc_design_{config_label}_{config_hash}"
            )
            thrust_design = st.number_input(
                f"Design Thrust [{thrust_unit}]",
                min_value=thrust_min,
                max_value=thrust_max,
                value=thrust_design_default,
                step=thrust_step,
                format="%.2f" if unit_system == "Imperial" else "%.1f",
                key=f"chamber_thrust_design_{config_label}_{config_hash}"
            )
            
            # Show force coefficient input only for manual method
            if not use_cea_solver:
                force_coefficient = st.number_input(
                    "Force Coefficient",
                    min_value=1.0,
                    max_value=2.0,
                    value=force_coefficient_default,
                    step=0.1,
                    format="%.2f",
                    help="Thrust coefficient (Cf) - typically 1.2-1.8 for rocket nozzles",
                    key=f"chamber_force_coefficient_{config_label}_{config_hash}"
                )
                MR_input = MR_default  # Not used, but initialize for consistency
            else:
                # For CEA solver, we need mixture ratio instead
                MR_input = st.number_input(
                    "Mixture Ratio (O/F)",
                    min_value=1.0,
                    max_value=10.0,
                    value=MR_default,
                    step=0.1,
                    format="%.2f",
                    help="Oxidizer-to-fuel mass ratio for CEA thermochemistry lookup",
                    key=f"chamber_MR_{config_label}_{config_hash}"
                )
                force_coefficient = None  # Will be calculated by CEA solver
        
        with col2:
            diameter_inner = st.number_input(
                f"Chamber Inner Diameter [{length_unit}]",
                min_value=length_min,
                max_value=length_max,
                value=diameter_inner_default,
                step=length_step,
                format="%.4f" if unit_system == "Imperial" else "%.6f",
                key=f"chamber_diameter_inner_{config_label}_{config_hash}"
            )
            diameter_exit = st.number_input(
                f"Exit Diameter [{length_unit}]",
                min_value=length_min,
                max_value=length_max,
                value=diameter_exit_default,
                step=length_step,
                format="%.4f" if unit_system == "Imperial" else "%.6f",
                key=f"chamber_diameter_exit_{config_label}_{config_hash}"
            )
            l_star = st.number_input(
                f"L* (Characteristic Length) [{length_unit}]",
                min_value=l_star_min,
                max_value=l_star_max,
                value=l_star_default,
                step=l_star_step,
                format="%.4f",
                key=f"chamber_l_star_{config_label}_{config_hash}"
            )
        
        calculate_button = st.form_submit_button("Calculate Chamber Geometry", type="primary", width='stretch')
    
    if calculate_button:
        try:
            # Convert inputs to metric for calculation
            if unit_system == "Imperial":
                pc_design_metric = pc_design * PSI_TO_PA  # Convert PSI to Pa (using module-level constant)
                thrust_design_metric = thrust_design * LBF_TO_N
                diameter_inner_metric = diameter_inner * IN_TO_M
                diameter_exit_metric = diameter_exit * IN_TO_M
                l_star_metric = l_star * IN_TO_M
            else:
                pc_design_metric = pc_design
                thrust_design_metric = thrust_design
                diameter_inner_metric = diameter_inner
                diameter_exit_metric = diameter_exit
                l_star_metric = l_star
            
            # Calculate chamber geometry
            solver_info = None
            # MR_input is already defined in the form section above
            if use_cea_solver:
                # Use CEA-based solver
                with st.spinner("Calculating chamber geometry with CEA solver (this may take a moment)..."):
                    try:
                        # Get nozzle efficiency from config to use corrected Cf
                        nozzle_efficiency = runner.config.nozzle.efficiency if runner and runner.config else 0.98
                        pts, table_data, total_chamber_length, solver_info = solve_chamber_geometry_with_cea(
                            pc_design=pc_design_metric,
                            thrust_design=thrust_design_metric,
                            cea_cache=runner.cea_cache,
                            MR=MR_input,
                            diameter_inner=diameter_inner_metric,
                            diameter_exit=diameter_exit_metric,
                            l_star=l_star_metric,
                            nozzle_efficiency=nozzle_efficiency,
                            do_plot=False,
                            steps=200,
                            verbose=False
                        )
                        # Extract calculated Cf from solver info (this is now the corrected Cf)
                        force_coefficient = solver_info.get('final_Cf', force_coefficient_default)
                        Cf_ideal = solver_info.get('final_Cf_ideal', force_coefficient)
                        st.success(f"✓ CEA solver converged: Cf = {force_coefficient:.4f} (corrected, with efficiency={nozzle_efficiency:.3f})")
                        st.caption(f"Cf_ideal = {Cf_ideal:.4f} (from CEA)")
                    except Exception as e:
                        st.error(f"CEA solver failed: {e}")
                        st.info("Falling back to manual force coefficient method...")
                        use_cea_solver = False
                        # Fall back to manual method
                        force_coefficient = force_coefficient_default
            
            if not use_cea_solver:
                # Use manual force coefficient method (original behavior)
                with st.spinner("Calculating chamber geometry..."):
                    pts, table_data, total_chamber_length = chamber_geometry_calc(
                        pc_design=pc_design_metric,
                        thrust_design=thrust_design_metric,
                        force_coeffcient=force_coefficient,
                        diameter_inner=diameter_inner_metric,
                        diameter_exit=diameter_exit_metric,
                        l_star=l_star_metric,
                        do_plot=False,
                        steps=200
                    )
            
            # Extract calculated values from table_data
            # table_data format: [headers, row1, row2, ...]
            # Each row: [parameter_name, metric_value, metric_unit, imperial_value, imperial_unit]
            calculated_values = {}
            for row in table_data[1:]:  # Skip header row
                param_name = row[0]
                metric_value_str = str(row[1]).strip()
                # Parse the metric value (handles scientific notation)
                try:
                    # Python's float() handles scientific notation automatically
                    metric_value = float(metric_value_str)
                    calculated_values[param_name] = metric_value
                except (ValueError, TypeError):
                    # Skip if we can't parse it
                    pass
            
            # Also use the directly returned total_chamber_length as a fallback
            if "Total Chamber Length" not in calculated_values and total_chamber_length is not None:
                calculated_values["Total Chamber Length"] = total_chamber_length
            
            # Update config with calculated values
            config_dict = config_obj.model_dump(exclude_none=False)
            updated = False
            
            # Ensure chamber section exists
            if "chamber" not in config_dict:
                config_dict["chamber"] = {}
            
            # Update chamber section - always update these if calculated
            if "Chamber Volume" in calculated_values:
                config_dict["chamber"]["volume"] = calculated_values["Chamber Volume"]
                updated = True
            if "Throat Area" in calculated_values:
                config_dict["chamber"]["A_throat"] = calculated_values["Throat Area"]
                updated = True
            if "L*" in calculated_values:
                config_dict["chamber"]["Lstar"] = calculated_values["L*"]
                updated = True
            # Update chamber length with total chamber length (cylindrical + contraction)
            if "Total Chamber Length" in calculated_values:
                config_dict["chamber"]["length"] = calculated_values["Total Chamber Length"]
                updated = True
            
            # Store design parameters used for geometry calculation
            config_dict["chamber"]["design_pressure"] = pc_design_metric
            config_dict["chamber"]["design_thrust"] = thrust_design_metric
            config_dict["chamber"]["design_force_coefficient"] = force_coefficient
            if use_cea_solver and solver_info:
                # Store CEA solver info
                config_dict["chamber"]["design_MR"] = MR_input
                config_dict["chamber"]["design_cea_solver_used"] = True
                config_dict["chamber"]["design_cea_solver_info"] = {
                    "converged": solver_info.get("converged", False),
                    "iterations": solver_info.get("iterations", 0),
                    "final_Cf": solver_info.get("final_Cf", force_coefficient),
                    "final_eps": solver_info.get("final_eps", 0.0),
                }
            # Also update chamber_inner_diameter and exit_diameter from form inputs
            config_dict["chamber"]["chamber_inner_diameter"] = diameter_inner_metric
            updated = True
            
            # Update nozzle section - ALWAYS update these if calculated
            if "nozzle" not in config_dict:
                config_dict["nozzle"] = {}
            if "Throat Area" in calculated_values:
                config_dict["nozzle"]["A_throat"] = calculated_values["Throat Area"]
                updated = True
            if "Exit Area" in calculated_values:
                config_dict["nozzle"]["A_exit"] = calculated_values["Exit Area"]
                updated = True
            if "Expansion Ratio" in calculated_values:
                config_dict["nozzle"]["expansion_ratio"] = calculated_values["Expansion Ratio"]
                updated = True
            # Update exit_diameter from form input
            config_dict["nozzle"]["exit_diameter"] = diameter_exit_metric
            updated = True
            
            # Update combustion.cea.expansion_ratio if it exists
            if "combustion" in config_dict and "cea" in config_dict["combustion"]:
                if "Expansion Ratio" in calculated_values and "expansion_ratio" in config_dict["combustion"]["cea"]:
                    config_dict["combustion"]["cea"]["expansion_ratio"] = calculated_values["Expansion Ratio"]
                    updated = True
            
            # Store calculation results in session state so they persist after rerun
            # Note: config_dict contains the updated values but they are NOT automatically loaded
            st.session_state["chamber_calc_results"] = {
                "pts": pts,
                "table_data": table_data,
                "calculated_values": calculated_values,
                "config_dict": config_dict,
                "updated": updated  # This just indicates values were calculated, not that they're loaded
            }
            
            # Generate DXF file for download
            dxf_bytes = None
            try:
                import tempfile
                import os
                
                # Generate DXF to a temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.dxf', delete=False) as tmp_file:
                    tmp_dxf_path = tmp_file.name
                
                # Generate DXF using chamber_geometry_calc
                _, _, _ = chamber_geometry_calc(
                    pc_design=pc_design_metric,
                    thrust_design=thrust_design_metric,
                    force_coeffcient=force_coefficient,
                    diameter_inner=diameter_inner_metric,
                    diameter_exit=diameter_exit_metric,
                    l_star=l_star_metric,
                    do_plot=False,
                    steps=200,
                    export_dxf=tmp_dxf_path
                )
                
                # Read the DXF file
                with open(tmp_dxf_path, 'rb') as f:
                    dxf_bytes = f.read()
                
                # Clean up temporary file
                os.unlink(tmp_dxf_path)
                
                # Store DXF bytes in session state
                st.session_state["chamber_dxf_bytes"] = dxf_bytes
            except ImportError:
                st.warning("ezdxf library is required for DXF export. Install it with: pip install ezdxf")
            except Exception as e:
                st.warning(f"Error generating DXF: {str(e)}")
            
            # Store DXF bytes in results for display
            results_for_display = {
                "pts": pts,
                "table_data": table_data,
                "calculated_values": calculated_values,
                "config_dict": config_dict,
                "updated": updated,
                "dxf_bytes": dxf_bytes
            }
            
            # Store results in session state FIRST (before any rerun)
            st.session_state["chamber_calc_results"] = {
                "pts": pts,
                "table_data": table_data,
                "calculated_values": calculated_values,
                "config_dict": config_dict,
                "updated": updated,
                "dxf_bytes": dxf_bytes,
                "solver_info": solver_info,
                "use_cea_solver": use_cea_solver
            }
            if dxf_bytes:
                st.session_state["chamber_dxf_bytes"] = dxf_bytes
            
            # CRITICAL: Update session state config_dict IMMEDIATELY with ALL calculated values
            # This is THE source of truth - download button and runner will use this
            st.session_state["config_dict"] = config_dict
            st.session_state["config_updated"] = True
            # Mark that config was updated from chamber design so sidebar can refresh
            st.session_state["config_updated_from_chamber_design"] = True
            
            # Display results immediately after calculation
            results_for_display["solver_info"] = solver_info
            results_for_display["use_cea_solver"] = use_cea_solver
            _display_chamber_results(results_for_display)
            
            # Show success message
            st.success("✓ Chamber geometry calculated and config updated! Download button below has the updated config.")
            
        except Exception as e:
            st.error(f"Error calculating chamber geometry: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # Display stored results if they exist (from previous calculation, not the one we just did)
    # Only display if we didn't just calculate (to avoid double display)
    if has_stored_results and stored_results and not calculate_button:
        # Add DXF bytes to stored results if available
        display_results = stored_results.copy()
        if "chamber_dxf_bytes" in st.session_state:
            display_results["dxf_bytes"] = st.session_state["chamber_dxf_bytes"]
        else:
            display_results["dxf_bytes"] = None
        
        # Display the results (they persist in session state after form submission)
        _display_chamber_results(display_results)
    
    # Always show download config button (uses current config_dict from session state - THE SOURCE OF TRUTH)
    st.markdown("---")
    st.subheader("Download Configuration")
    # ALWAYS use st.session_state["config_dict"] as it's the single source of truth
    # This ensures the downloaded config matches what was calculated
    if "config_dict" in st.session_state:
        current_config_dict = st.session_state["config_dict"]
    else:
        # Fallback only if config_dict doesn't exist (shouldn't happen)
        current_config_dict = config_obj.model_dump(exclude_none=False)
        st.session_state["config_dict"] = current_config_dict
    
    config_yaml = yaml.dump(current_config_dict, default_flow_style=False, sort_keys=False, allow_unicode=True)
    st.download_button(
        label="📥 Download Current Config (YAML)",
        data=config_yaml,
        file_name=f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
        mime="application/x-yaml",
        key="chamber_design_download_config_always",
        use_container_width=True,
        help="Download the current configuration (includes any changes from sidebar or chamber design)"
    )
    
    # Return updated config object from session state
    if "config_dict" in st.session_state:
        try:
            return PintleEngineConfig(**st.session_state["config_dict"])
        except Exception:
            # If validation fails, return the original
            return config_obj
    return config_obj


def _display_chamber_results(results: dict) -> None:
    """Helper function to display chamber calculation results."""
    pts = results["pts"]
    table_data = results["table_data"]
    config_dict = results.get("config_dict", {})
    updated = results.get("updated", False)
    solver_info = results.get("solver_info", None)
    use_cea_solver = results.get("use_cea_solver", False)
    
    # Display CEA solver info if used
    if use_cea_solver and solver_info:
        st.subheader("CEA Solver Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Converged", "✓ Yes" if solver_info.get("converged", False) else "✗ No")
        with col2:
            st.metric("Iterations", solver_info.get("iterations", 0))
        with col3:
            st.metric("Final Cf (Ideal)", f"{solver_info.get('final_Cf', 0):.4f}")
        with col4:
            st.metric("Final Expansion Ratio", f"{solver_info.get('final_eps', 0):.2f}")
        
        # Show convergence history if available
        if solver_info.get("convergence_history"):
            conv_history = solver_info["convergence_history"]
            if len(conv_history) > 1:
                with st.expander("Convergence History"):
                    conv_df = pd.DataFrame(conv_history)
                    st.dataframe(conv_df[['iteration', 'A_throat', 'Cf', 'eps', 'residual']], 
                               use_container_width=True, hide_index=True)
    
    # Display results
    st.subheader("Chamber Contour")
    
    # Try to create enhanced multi-layer visualization if available
    if (calculate_complete_chamber_geometry is not None and 
        hasattr(config_obj, 'ablative_cooling') and config_obj.ablative_cooling and
        hasattr(config_obj, 'graphite_insert') and config_obj.graphite_insert and
        hasattr(config_obj, 'stainless_steel_case') and config_obj.stainless_steel_case):
        
        try:
            # Get current geometry from results
            V_chamber = results.get("chamber_volume", config_obj.chamber.volume)
            A_throat = results.get("A_throat", config_obj.chamber.A_throat)
            L_chamber = results.get("chamber_length", config_obj.chamber.length)
            D_chamber_initial = config_obj.chamber.chamber_inner_diameter
            D_throat_initial = np.sqrt(4.0 * A_throat / np.pi)
            
            # Get recession values (default to 0 if not available)
            recession_chamber = results.get("recession_chamber", 0.0) if isinstance(results.get("recession_chamber"), (int, float)) else 0.0
            recession_graphite = results.get("recession_graphite", 0.0) if isinstance(results.get("recession_graphite"), (int, float)) else 0.0
            
            # Calculate complete geometry
            geometry = calculate_complete_chamber_geometry(
                V_chamber=V_chamber,
                A_throat=A_throat,
                L_chamber=L_chamber,
                D_chamber_initial=D_chamber_initial,
                D_throat_initial=D_throat_initial,
                ablative_config=config_obj.ablative_cooling,
                graphite_config=config_obj.graphite_insert,
                stainless_config=config_obj.stainless_steel_case,
                recession_chamber=recession_chamber,
                recession_graphite=recession_graphite,
                n_points=100,
            )
            
            # Create enhanced plotly visualization
            fig_contour = go.Figure()
            positions = np.array(geometry["positions"])
            
            # Gas boundary (chamber)
            D_gas = np.array(geometry["D_gas_chamber"]) / 2.0  # Convert to radius
            fig_contour.add_trace(go.Scatter(
                x=positions,
                y=D_gas,
                mode='lines',
                name='Gas Boundary (Chamber)',
                line=dict(color='orange', width=3),
                fill='tozeroy',
                fillcolor='rgba(255, 165, 0, 0.2)',
            ))
            fig_contour.add_trace(go.Scatter(
                x=positions,
                y=-D_gas,
                mode='lines',
                name='Gas Boundary (Lower)',
                line=dict(color='orange', width=3),
                fill='tozeroy',
                fillcolor='rgba(255, 165, 0, 0.2)',
                showlegend=False,
            ))
            
            # Ablative layer
            if geometry["ablative_thickness"][0] > 0:
                D_ablative = np.array(geometry["D_ablative_outer"]) / 2.0
                fig_contour.add_trace(go.Scatter(
                    x=positions,
                    y=D_ablative,
                    mode='lines',
                    name='Phenolic Ablator',
                    line=dict(color='brown', width=2, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(139, 69, 19, 0.3)',
                ))
                fig_contour.add_trace(go.Scatter(
                    x=positions,
                    y=-D_ablative,
                    mode='lines',
                    name='Phenolic Ablator (Lower)',
                    line=dict(color='brown', width=2, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(139, 69, 19, 0.3)',
                    showlegend=False,
                ))
            
            # Stainless steel case (chamber)
            if geometry["stainless_thickness"] > 0:
                D_stainless = np.array(geometry["D_stainless_outer"]) / 2.0
                fig_contour.add_trace(go.Scatter(
                    x=positions,
                    y=D_stainless,
                    mode='lines',
                    name='Stainless Steel Case',
                    line=dict(color='gray', width=2, dash='dot'),
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.2)',
                ))
                fig_contour.add_trace(go.Scatter(
                    x=positions,
                    y=-D_stainless,
                    mode='lines',
                    name='Stainless Steel (Lower)',
                    line=dict(color='gray', width=2, dash='dot'),
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.2)',
                    showlegend=False,
                ))
            
            # Throat region with graphite
            throat_pos = positions[-1]
            D_throat = geometry["D_throat_current"] / 2.0
            D_graphite = geometry["D_graphite_outer"] / 2.0
            D_stainless_throat = geometry["D_stainless_throat_outer"] / 2.0
            
            # Draw throat region
            if geometry["graphite_thickness"] > 0:
                # Graphite insert (circular cross-section at throat)
                theta = np.linspace(0, 2*np.pi, 100)
                fig_contour.add_trace(go.Scatter(
                    x=[throat_pos] * len(theta),
                    y=D_graphite * np.cos(theta),
                    mode='lines',
                    name='Graphite Insert',
                    line=dict(color='black', width=2),
                ))
                # Throat (gas boundary at throat)
                fig_contour.add_trace(go.Scatter(
                    x=[throat_pos] * len(theta),
                    y=D_throat * np.cos(theta),
                    mode='lines',
                    name='Throat',
                    line=dict(color='red', width=3),
                ))
                # Stainless steel at throat
                if geometry["stainless_thickness"] > 0:
                    fig_contour.add_trace(go.Scatter(
                        x=[throat_pos] * len(theta),
                        y=D_stainless_throat * np.cos(theta),
                        mode='lines',
                        name='Stainless Steel (Throat)',
                        line=dict(color='gray', width=2, dash='dot'),
                    ))
            
            # Add centerline
            x_min, x_max = positions[0], positions[-1]
            fig_contour.add_trace(go.Scatter(
                x=[x_min, x_max],
                y=[0, 0],
                mode='lines',
                name='Centerline',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ))
            
            fig_contour.update_layout(
                title="Complete Chamber Cross-Section (Chamber + Ablative + Graphite + Stainless Steel)",
                xaxis_title="Axial Position [m]",
                yaxis_title="Radius [m]",
                height=600,
                showlegend=True,
                yaxis=dict(scaleanchor="x", scaleratio=1),  # Equal aspect ratio
            )
            st.plotly_chart(fig_contour, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not generate enhanced multi-layer visualization: {e}")
            # Fall back to simple contour
            fig_contour = go.Figure()
            fig_contour.add_trace(go.Scatter(
                x=pts[:, 0],
                y=pts[:, 1],
                mode='lines',
                name='Chamber Contour',
                line=dict(color='red', width=2)
            ))
            fig_contour.add_trace(go.Scatter(
                x=pts[:, 0],
                y=-pts[:, 1],
                mode='lines',
                name='Lower Contour',
                line=dict(color='red', width=2),
                showlegend=False
            ))
            x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
            fig_contour.add_trace(go.Scatter(
                x=[x_min, x_max],
                y=[0, 0],
                mode='lines',
                name='Centerline',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ))
            fig_contour.update_layout(
                title="Chamber Contour (Half-Section)",
                xaxis_title="Axial Position [m]",
                yaxis_title="Radius [m]",
                height=500,
                showlegend=True
            )
            st.plotly_chart(fig_contour, use_container_width=True)
    else:
        # Simple contour plot (fallback)
        fig_contour = go.Figure()
        fig_contour.add_trace(go.Scatter(
            x=pts[:, 0],
            y=pts[:, 1],
            mode='lines',
            name='Chamber Contour',
            line=dict(color='red', width=2)
        ))
        fig_contour.add_trace(go.Scatter(
            x=pts[:, 0],
            y=-pts[:, 1],
            mode='lines',
            name='Lower Contour',
            line=dict(color='red', width=2),
            showlegend=False
        ))
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        fig_contour.add_trace(go.Scatter(
            x=[x_min, x_max],
            y=[0, 0],
            mode='lines',
            name='Centerline',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False
        ))
        fig_contour.update_layout(
            title="Chamber Contour (Half-Section)",
            xaxis_title="Axial Position [m]",
            yaxis_title="Radius [m]",
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig_contour, use_container_width=True)
    
    # Display geometry parameters table
    st.subheader("Chamber Geometry Parameters")
    if table_data:
        df = pd.DataFrame(table_data[1:], columns=table_data[0])
        st.dataframe(df, width='stretch')
    
    # # Show update status
    # if updated:
    #     # st.success("✓ Configuration values have been updated. The page will reload to apply changes.")
    #     None
    
    # DXF download button
    if "dxf_bytes" in results and results["dxf_bytes"] is not None:
        st.download_button(
            label="Download Chamber Contour (DXF)",
            data=results["dxf_bytes"],
            file_name="chamber_contour.dxf",
            mime="application/dxf",
            key="chamber_dxf_download"  # Unique key to prevent duplicate element error
        )


def graphite_insert_view(config_obj: PintleEngineConfig, runner: Optional[PintleEngineRunner] = None) -> PintleEngineConfig:
    """Graphite Insert tab - sizes graphite throat insert geometry and updates config."""
    st.header("Graphite Insert Sizing")
    st.markdown("Size graphite throat insert based on heat flux, recession, and thermal requirements.")
    
    # Get config label for unique form keys
    config_label = st.session_state.get("config_label", "default")
    
    # Get current config values for defaults
    config_dict = config_obj.model_dump()
    graphite_config = config_dict.get("graphite_insert", {})
    chamber_config = config_dict.get("chamber", {})
    
    # Get defaults from config
    throat_diameter = np.sqrt(4.0 * chamber_config.get("A_throat", 0.000857892) / np.pi)
    burn_time = 10.0  # Default burn time
    if config_dict.get("thrust") and config_dict["thrust"].get("burn_time"):
        burn_time = float(config_dict["thrust"]["burn_time"])
    
    # Input method selection - OUTSIDE the form so it updates immediately
    st.subheader("Input Method")
    input_method = st.radio(
        "How to get design point values?",
        ["From Engine Run", "Manual Input"],
        horizontal=True,
        key=f"graphite_input_method_{config_label}",
        help="'From Engine Run' will use a time-series dataset to evaluate the engine over the full burn. 'Manual Input' lets you enter values directly."
    )
    
    # Handle dataset selection for "From Engine Run"
    df_selected: Optional[pd.DataFrame] = None
    dataset_name: Optional[str] = None
    
    if input_method == "From Engine Run":
        if runner is None:
            st.warning("⚠️ Runner not available. Please use 'Manual Input' mode or ensure engine is configured.")
            input_method = "Manual Input"  # Force manual mode
        else:
            # Get available datasets (like COPV does)
            datasets: Dict[str, pd.DataFrame] = st.session_state.get("custom_plot_datasets", {})
            required_cols = {"time", "P_tank_O (psi)", "P_tank_F (psi)"}
            eligible = {name: df for name, df in datasets.items() if required_cols.issubset(df.columns)}
            
            if not eligible:
                st.info(
                    "No eligible time-series datasets found. Run a time-series evaluation (generated or uploaded) so the "
                    "resulting dataframe includes time and tank pressure columns. Alternatively, use 'Manual Input' mode."
                )
                input_method = "Manual Input"  # Fall back to manual
            else:
                dataset_names = list(eligible.keys())
                default_dataset = st.session_state.get("last_custom_dataset")
                default_index = dataset_names.index(default_dataset) if default_dataset in dataset_names else 0
                
                dataset_name = st.selectbox(
                    "Time-Series Dataset (must include time and tank pressure columns)",
                    dataset_names,
                    index=default_index,
                    key=f"graphite_dataset_{config_label}",
                )
                df_selected = eligible[dataset_name].copy().sort_values("time").reset_index(drop=True)
                if df_selected.empty:
                    st.warning("Selected dataset is empty. Choose a different dataset or use 'Manual Input' mode.")
                    input_method = "Manual Input"
                else:
                    duration = float(df_selected["time"].iloc[-1] - df_selected["time"].iloc[0])
                    st.caption(f"Dataset: {dataset_name} | {len(df_selected)} samples | duration ≈ {duration:.2f} s")
    
    # Create form for input
    with st.form(f"graphite_insert_form_{config_label}", clear_on_submit=False):
        
        st.subheader("Graphite Material Properties")
        col1, col2, col3 = st.columns(3)
        with col1:
            thermal_conductivity = st.number_input(
                "Thermal Conductivity [W/(m·K)]",
                min_value=10.0,
                max_value=500.0,
                value=float(graphite_config.get("thermal_conductivity", 100.0)),
                step=10.0,
                key=f"graphite_k_{config_label}"
            )
            density = st.number_input(
                "Density [kg/m³]",
                min_value=1000.0,
                max_value=3000.0,
                value=float(graphite_config.get("material_density", 1800.0)),
                step=50.0,
                key=f"graphite_rho_{config_label}"
            )
        with col2:
            specific_heat = st.number_input(
                "Specific Heat [J/(kg·K)]",
                min_value=100.0,
                max_value=2000.0,
                value=float(graphite_config.get("specific_heat", 710.0)),
                step=50.0,
                key=f"graphite_cp_{config_label}"
            )
            backface_temp_max = st.number_input(
                "Max Backface Temp [K]",
                min_value=300.0,
                max_value=1000.0,
                value=600.0,
                step=50.0,
                help="Maximum allowable temperature at backface (for metal substrate/adhesive)",
                key=f"graphite_T_back_{config_label}"
            )
        with col3:
            mechanical_thickness = st.number_input(
                "Mechanical Thickness [mm]",
                min_value=0.1,
                max_value=10.0,
                value=float(graphite_config.get("initial_thickness", 0.005)) * 1000.0,
                step=0.1,
                help="Minimum thickness for mechanical support, groove, seating",
                key=f"graphite_t_mech_{config_label}"
            )
            safety_factor = st.number_input(
                "Safety Factor [%]",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=5.0,
                help="Safety factor as percentage (typically 30-50%)",
                key=f"graphite_sf_{config_label}"
            )
        
        st.subheader("Operating Conditions")
        col1, col2 = st.columns(2)
        with col1:
            burn_time_input = st.number_input(
                "Burn Time [s]",
                min_value=0.1,
                max_value=1000.0,
                value=burn_time,
                step=1.0,
                key=f"graphite_burn_time_{config_label}"
            )
            throat_diam_input = st.number_input(
                "Throat Diameter [mm]",
                min_value=1.0,
                max_value=500.0,
                value=throat_diameter * 1000.0,
                step=0.1,
                key=f"graphite_Dt_{config_label}"
            )
        with col2:
            use_transient = st.checkbox(
                "Use Transient Conduction Model",
                value=False,
                help="Include transient thermal penetration depth calculation",
                key=f"graphite_transient_{config_label}"
            )
            if use_transient:
                transient_eta = st.number_input(
                    "Transient Safety Factor (η)",
                    min_value=1.5,
                    max_value=5.0,
                    value=2.5,
                    step=0.1,
                    help="Safety factor for transient penetration depth (typically 2-3)",
                    key=f"graphite_eta_{config_label}"
                )
            else:
                transient_eta = 2.5
        
        # Manual input section (shown if manual mode or if runner unavailable)
        peak_heat_flux_manual = None
        surface_temp_manual = None
        recession_rate_manual = None
        
        if input_method == "Manual Input":
            st.subheader("Manual Design Point Values")
            col1, col2 = st.columns(2)
            with col1:
                peak_heat_flux_manual = st.number_input(
                    "Peak Heat Flux [MW/m²]",
                    min_value=0.1,
                    max_value=50.0,
                    value=5.0,
                    step=0.5,
                    help="Peak convective heat flux at throat",
                    key=f"graphite_q_peak_{config_label}"
                ) * 1e6  # Convert to W/m²
                surface_temp_manual = st.number_input(
                    "Surface Temperature [K]",
                    min_value=500.0,
                    max_value=3000.0,
                    value=2000.0,
                    step=50.0,
                    help="Surface temperature from wall model",
                    key=f"graphite_T_surf_{config_label}"
                )
            with col2:
                recession_rate_manual = st.number_input(
                    "Recession Rate [µm/s]",
                    min_value=0.1,
                    max_value=1000.0,
                    value=60.0,
                    step=5.0,
                    help="Net recession rate from wall model",
                    key=f"graphite_rdot_{config_label}"
                ) * 1e-6  # Convert to m/s
        
        calculate_button = st.form_submit_button("Size Graphite Insert", type="primary")
    
    # Perform calculation if button clicked
    if calculate_button:
        try:
            # Get values from form
            mechanical_thickness_m = mechanical_thickness / 1000.0  # Convert mm to m
            throat_diameter_m = throat_diam_input / 1000.0  # Convert mm to m
            safety_factor_frac = safety_factor / 100.0  # Convert % to fraction
            
            # Get design point values
            if input_method == "From Engine Run" and runner is not None and df_selected is not None:
                # Process time series dataset
                with st.spinner("Running engine over time series..."):
                    times = df_selected["time"].values
                    P_tank_O_psi_array = df_selected["P_tank_O (psi)"].values
                    P_tank_F_psi_array = df_selected["P_tank_F (psi)"].values
                    
                    # Convert to Pa
                    P_tank_O_pa = P_tank_O_psi_array * PSI_TO_PA
                    P_tank_F_pa = P_tank_F_psi_array * PSI_TO_PA
                    
                    # Run engine for all time steps
                    results_dict = runner.evaluate_arrays(P_tank_O_pa, P_tank_F_pa)
                    diagnostics_list = results_dict.get("diagnostics", [])
                    
                    # Extract time series of heat flux, surface temp, and recession rate
                    heat_flux_series = []
                    surface_temp_series = []
                    recession_rate_series = []
                    oxidation_rate_series = []
                    ablation_rate_series = []
                    
                    graphite_cfg_temp = config_obj.graphite_insert
                    throat_area = np.pi * (throat_diameter_m / 2) ** 2
                    
                    for i in range(len(times)):
                        # Get diagnostics for this time step
                        if i < len(diagnostics_list) and isinstance(diagnostics_list[i], dict):
                            diagnostics = diagnostics_list[i]
                        else:
                            diagnostics = {}
                        
                        cooling = diagnostics.get("cooling", {})
                        
                        # Get results for this time step
                        Pc = float(results_dict["Pc"][i]) if i < len(results_dict["Pc"]) else 2e6
                        Tc = float(results_dict["Tc"][i]) if i < len(results_dict["Tc"]) else 3000.0
                        gamma = float(results_dict["gamma"][i]) if i < len(results_dict["gamma"]) else 1.2
                        R = float(results_dict["R"][i]) if i < len(results_dict["R"]) else 300.0
                        mdot_total = float(results_dict["mdot_total"][i]) if i < len(results_dict["mdot_total"]) else 0.1
                        
                        # Get chamber heat flux (from ablative or estimate)
                        ablative = cooling.get("ablative", {})
                        if ablative.get("enabled", False):
                            chamber_heat_flux = ablative.get("incident_heat_flux", 1e6)
                        else:
                            # Estimate from chamber conditions
                            # Rough estimate: q'' ~ 0.026 * Pr^0.4 * (rho*u)^0.8 * (T_g - T_w)
                            # Simplified: use typical value scaled by pressure
                            chamber_heat_flux = 1e6 * (Pc / 2e6) ** 0.8
                        
                        # Calculate throat heat flux multiplier
                        
                        # Estimate velocities
                        rho_chamber = Pc / (R * Tc)
                        A_chamber = np.pi * (throat_diameter_m * 3) ** 2 / 4.0  # Approximate chamber area
                        v_chamber = mdot_total / (rho_chamber * A_chamber) if rho_chamber > 0 else 50.0
                        v_throat = np.sqrt(gamma * R * Tc / (gamma + 1))  # Sonic velocity
                        
                        throat_mult = calculate_throat_recession_multiplier(
                            Pc, v_chamber, v_throat, chamber_heat_flux, gamma
                        )
                        peak_heat_flux_i = chamber_heat_flux * throat_mult
                        
                        # Get surface temperature (estimate from throat conditions)
                        surface_temp_i = Tc * 0.85  # Approximate throat temperature
                        
                        # Get recession rate from graphite cooling model
                        if graphite_cfg_temp and graphite_cfg_temp.enabled:
                            graphite_results = compute_graphite_recession(
                                net_heat_flux=peak_heat_flux_i,
                                throat_temperature=surface_temp_i,
                                gas_temperature=Tc,
                                graphite_config=graphite_cfg_temp,
                                throat_area=throat_area,
                                pressure=Pc,
                            )
                            recession_rate_i = graphite_results.get("recession_rate", 6e-5)
                            oxidation_rate_i = graphite_results.get("oxidation_rate", 0.0)
                            ablation_rate_i = graphite_results.get("recession_rate_thermal", 0.0)
                        else:
                            # Default estimate
                            recession_rate_i = 6e-5  # m/s
                            oxidation_rate_i = 0.0
                            ablation_rate_i = 0.0
                        
                        heat_flux_series.append(peak_heat_flux_i)
                        surface_temp_series.append(surface_temp_i)
                        recession_rate_series.append(recession_rate_i)
                        oxidation_rate_series.append(oxidation_rate_i)
                        ablation_rate_series.append(ablation_rate_i)
                    
                    # Convert to numpy arrays
                    heat_flux_series = np.array(heat_flux_series)
                    surface_temp_series = np.array(surface_temp_series)
                    recession_rate_series = np.array(recession_rate_series)
                    oxidation_rate_series = np.array(oxidation_rate_series)
                    ablation_rate_series = np.array(ablation_rate_series)
                    
                    # Use actual burn time from dataset
                    actual_burn_time = float(times[-1] - times[0])
                    if actual_burn_time > 0:
                        burn_time_input = actual_burn_time
                    
                    # Integrate recession rate over time to get total recession allowance
                    # This is more accurate than assuming constant max rate
                    dt = np.diff(times, prepend=times[0])
                    total_recession_allowance = float(np.sum(recession_rate_series * dt))
                    
                    # Calculate effective average recession rate for sizing function
                    # (sizing function expects recession_rate * burn_time, so we back-calculate)
                    if actual_burn_time > 0:
                        effective_recession_rate = total_recession_allowance / actual_burn_time
                    else:
                        effective_recession_rate = float(np.max(recession_rate_series))
                    
                    # Use peak values for sizing (heat flux and surface temp)
                    peak_heat_flux = float(np.max(heat_flux_series))
                    surface_temp = float(surface_temp_series[np.argmax(heat_flux_series)])  # Surface temp at peak heat flux
                    
                    # Use the effective recession rate (which represents the integrated total)
                    recession_rate = effective_recession_rate
                    
                    st.success(f"✓ Time series analysis complete: {len(times)} time steps, "
                              f"Peak heat flux = {peak_heat_flux/1e6:.2f} MW/m², "
                              f"Integrated total recession = {total_recession_allowance*1000:.2f} mm, "
                              f"Effective recession rate = {recession_rate*1e6:.2f} µm/s")
                    
                    # Store time series for verification
                    st.session_state["graphite_timeseries"] = {
                        "time": times,
                        "heat_flux": heat_flux_series,
                        "surface_temp": surface_temp_series,
                        "recession_rate": recession_rate_series,
                        "oxidation_rate": oxidation_rate_series,
                        "ablation_rate": ablation_rate_series,
                    }
            else:
                # Use manual input values
                if peak_heat_flux_manual is not None:
                    peak_heat_flux = peak_heat_flux_manual
                if surface_temp_manual is not None:
                    surface_temp = surface_temp_manual
                if recession_rate_manual is not None:
                    recession_rate = recession_rate_manual
            
            # Perform sizing calculation
            with st.spinner("Calculating graphite insert size..."):
                sizing = size_graphite_insert(
                    peak_heat_flux=peak_heat_flux,
                    surface_temperature=surface_temp,
                    recession_rate=recession_rate,
                    burn_time=burn_time_input,
                    thermal_conductivity=thermal_conductivity,
                    backface_temperature_max=backface_temp_max,
                    throat_diameter=throat_diameter_m,
                    density=density if use_transient else None,
                    specific_heat=specific_heat if use_transient else None,
                    mechanical_thickness=mechanical_thickness_m,
                    safety_factor=safety_factor_frac,
                    transient=use_transient,
                    transient_eta=transient_eta if use_transient else 2.5,
                )
            
            # Display results
            st.subheader("Sizing Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Initial Thickness", f"{sizing.initial_thickness*1000:.2f} mm")
                st.metric("Recession Allowance", f"{sizing.recession_allowance*1000:.2f} mm")
                st.metric("Conduction Thickness", f"{sizing.conduction_thickness*1000:.2f} mm")
            with col2:
                st.metric("Safety Margin", f"{sizing.safety_margin*1000:.2f} mm")
                st.metric("Total Axial Length", f"{sizing.total_axial_length*1000:.2f} mm")
                st.metric("Upstream Length", f"{sizing.axial_half_length_upstream*1000:.2f} mm")
            with col3:
                st.metric("Downstream Length", f"{sizing.axial_half_length_downstream*1000:.2f} mm")
                st.metric("Conduction Model", sizing.conduction_model)
                st.metric("Sizing Method", sizing.sizing_method)
            
            # Throat growth check
            st.subheader("Throat Growth Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Initial Throat Diameter", f"{sizing.throat_diameter_initial*1000:.3f} mm")
                st.metric("Final Throat Diameter", f"{sizing.throat_diameter_end*1000:.3f} mm")
            with col2:
                st.metric("Area Change", f"{sizing.throat_area_change*1e6:.3f} mm²")
                area_change_color = "normal" if not sizing.throat_area_change_excessive else "inverse"
                st.metric("Area Change [%]", f"{sizing.throat_area_change_pct:.2f}%", 
                         delta=None if not sizing.throat_area_change_excessive else "Exceeds 3% threshold")
            
            # Integrity warnings
            if sizing.integrity_note != "All checks passed":
                st.warning(f"⚠️ {sizing.integrity_note}")
            else:
                st.success("✓ All integrity checks passed")
            
            # Time series verification (if available)
            if "graphite_timeseries" in st.session_state and input_method == "From Engine Run":
                from pintle_pipeline.graphite_geometry import verify_transient_thickness
                
                st.subheader("Time Series Verification")
                ts_data = st.session_state["graphite_timeseries"]
                times = ts_data["time"]
                recession_rate_series = ts_data["recession_rate"]
                oxidation_rate_series = ts_data.get("oxidation_rate", np.zeros_like(recession_rate_series))
                ablation_rate_series = ts_data.get("ablation_rate", np.zeros_like(recession_rate_series))
                
                # Verify thickness throughout burn
                verification = verify_transient_thickness(
                    time=times,
                    recession_rate=recession_rate_series,
                    initial_thickness=sizing.initial_thickness,
                    mechanical_thickness=mechanical_thickness_m,
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Consumed", f"{verification['consumed']*1000:.2f} mm")
                    st.metric("Remaining at End", f"{verification['remaining']*1000:.2f} mm")
                with col2:
                    st.metric("Minimum Remaining", f"{verification['min_remaining']*1000:.2f} mm")
                    status_color = "normal" if verification['meets_mechanical'] else "inverse"
                    st.metric("Meets Mechanical", "✓ Yes" if verification['meets_mechanical'] else "✗ No", 
                             delta=None if verification['meets_mechanical'] else "Below minimum")
                with col3:
                    required_min = mechanical_thickness_m * 1000.0
                    margin = (verification['min_remaining'] * 1000.0) - required_min
                    st.metric("Safety Margin", f"{margin:.2f} mm")
                
                # Calculate cumulative recession due to oxidation vs ablation
                dt = np.diff(times, prepend=times[0])
                cumulative_oxidation = np.cumsum(oxidation_rate_series * dt)
                cumulative_ablation = np.cumsum(ablation_rate_series * dt)
                cumulative_total = cumulative_oxidation + cumulative_ablation
                
                # Create dataframe showing recession breakdown over time
                recession_df = pd.DataFrame({
                    "Time [s]": times,
                    "Oxidation Recession [mm]": cumulative_oxidation * 1000.0,
                    "Ablation Recession [mm]": cumulative_ablation * 1000.0,
                    "Total Recession [mm]": cumulative_total * 1000.0,
                    "Oxidation Rate [µm/s]": oxidation_rate_series * 1e6,
                    "Ablation Rate [µm/s]": ablation_rate_series * 1e6,
                    "Total Rate [µm/s]": recession_rate_series * 1e6,
                })
                
                st.subheader("Recession Breakdown Over Burn")
                st.caption("Cumulative recession due to oxidation vs thermal ablation")
                st.dataframe(recession_df, use_container_width=True, height=400)
                
                # Summary metrics
                total_oxidation = float(cumulative_oxidation[-1]) * 1000.0  # mm
                total_ablation = float(cumulative_ablation[-1]) * 1000.0  # mm
                total_recession = float(cumulative_total[-1]) * 1000.0  # mm
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Oxidation Recession", f"{total_oxidation:.3f} mm",
                             delta=f"{(total_oxidation/total_recession*100):.1f}%" if total_recession > 0 else None)
                with col2:
                    st.metric("Total Ablation Recession", f"{total_ablation:.3f} mm",
                             delta=f"{(total_ablation/total_recession*100):.1f}%" if total_recession > 0 else None)
                with col3:
                    st.metric("Total Recession", f"{total_recession:.3f} mm")
                
                # Plot time series
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=("Heat Flux & Surface Temperature", "Recession Rate Breakdown", "Cumulative Recession"),
                    vertical_spacing=0.12,
                    specs=[[{"secondary_y": True}], [{}], [{}]],
                )
                
                # Heat flux and surface temp
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=ts_data["heat_flux"] / 1e6,  # Convert to MW/m²
                        name="Heat Flux",
                        line=dict(color="red"),
                    ),
                    row=1, col=1,
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=ts_data["surface_temp"],
                        name="Surface Temp",
                        line=dict(color="orange"),
                    ),
                    row=1, col=1,
                    secondary_y=True,
                )
                
                # Recession rate breakdown
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=recession_rate_series * 1e6,  # Convert to µm/s
                        name="Total Rate",
                        line=dict(color="blue", width=2),
                    ),
                    row=2, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=oxidation_rate_series * 1e6,  # Convert to µm/s
                        name="Oxidation Rate",
                        line=dict(color="green"),
                    ),
                    row=2, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=ablation_rate_series * 1e6,  # Convert to µm/s
                        name="Ablation Rate",
                        line=dict(color="purple"),
                    ),
                    row=2, col=1,
                )
                
                # Cumulative recession
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=cumulative_total * 1000.0,  # Convert to mm
                        name="Total Cumulative",
                        line=dict(color="blue", width=2),
                    ),
                    row=3, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=cumulative_oxidation * 1000.0,  # Convert to mm
                        name="Oxidation Cumulative",
                        line=dict(color="green"),
                    ),
                    row=3, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=cumulative_ablation * 1000.0,  # Convert to mm
                        name="Ablation Cumulative",
                        line=dict(color="purple"),
                    ),
                    row=3, col=1,
                )
                
                fig.update_xaxes(title_text="Time [s]", row=3, col=1)
                fig.update_xaxes(title_text="Time [s]", row=2, col=1)
                fig.update_xaxes(title_text="Time [s]", row=1, col=1)
                fig.update_yaxes(title_text="Heat Flux [MW/m²]", row=1, col=1, secondary_y=False)
                fig.update_yaxes(title_text="Temperature [K]", row=1, col=1, secondary_y=True)
                fig.update_yaxes(title_text="Recession Rate [µm/s]", row=2, col=1)
                fig.update_yaxes(title_text="Cumulative Recession [mm]", row=3, col=1)
                fig.update_layout(height=800, showlegend=True)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Update config button
            st.subheader("Update Configuration")
            if st.button("Update Config with Calculated Values", type="primary", 
                        key=f"graphite_update_config_{config_label}"):
                # Update config_dict
                if "graphite_insert" not in config_dict:
                    config_dict["graphite_insert"] = {}
                
                config_dict["graphite_insert"]["enabled"] = True
                config_dict["graphite_insert"]["initial_thickness"] = float(sizing.initial_thickness)
                config_dict["graphite_insert"]["thermal_conductivity"] = float(thermal_conductivity)
                config_dict["graphite_insert"]["material_density"] = float(density)
                config_dict["graphite_insert"]["specific_heat"] = float(specific_heat)
                
                # Update session state
                st.session_state["config_dict"] = config_dict
                st.success("✓ Configuration updated! The runner will be recreated on next evaluation.")
                st.info("💡 Note: Axial length is not stored in config - it's calculated from heat flux profile or simple rule.")
            
            # Store results for display
            st.session_state["graphite_sizing_results"] = sizing
            
        except Exception as e:
            st.error(f"Error during sizing calculation: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Show stored results if available
    if "graphite_sizing_results" in st.session_state and not calculate_button:
        sizing = st.session_state["graphite_sizing_results"]
        st.subheader("Previous Sizing Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Initial Thickness", f"{sizing.initial_thickness*1000:.2f} mm")
            st.metric("Recession Allowance", f"{sizing.recession_allowance*1000:.2f} mm")
        with col2:
            st.metric("Conduction Thickness", f"{sizing.conduction_thickness*1000:.2f} mm")
            st.metric("Total Axial Length", f"{sizing.total_axial_length*1000:.2f} mm")
        with col3:
            st.metric("Area Change [%]", f"{sizing.throat_area_change_pct:.2f}%")
            if sizing.integrity_note != "All checks passed":
                st.warning(f"⚠️ {sizing.integrity_note}")
    
    return PintleEngineConfig(**st.session_state.get("config_dict", config_dict))


def flight_sim_view(runner: PintleEngineRunner, config_obj: PintleEngineConfig, config_label: str) -> None:
    if not ROCKETPY_AVAILABLE:
        st.error("RocketPy is not installed. Install it with: `pip install rocketpy`")
        st.info("Flight simulation requires RocketPy for trajectory propagation.")
        return
    
    if isinstance(runner, PintleEngineConfig):
        runner = PintleEngineRunner(runner)
    st.header("Flight Simulation")
    st.write("Use engine performance to simulate a basic rocket flight and report apogee and velocity.")

    source = st.radio("Performance source", ["Tank pressures (constant thrust)", "Dataset (time-varying)"], horizontal=True, key="flight_perf_source")

    # Tank pressures inputs shown only for constant source
    if source == "Tank pressures (constant thrust)":
        col1, col2 = st.columns(2)
        with col1:
            P_tank_O_psi = st.number_input(
                "LOX Tank Pressure [psi]",
                min_value=50.0,
                max_value=3000.0,
                value=1305.0,
                step=5.0,
                key="flight_lox_tank_psi",
            )
        with col2:
            P_tank_F_psi = st.number_input(
                "Fuel Tank Pressure [psi]",
                min_value=50.0,
                max_value=3000.0,
                value=974.0,
                step=5.0,
                key="flight_fuel_tank_psi",
            )
    else:
        # Dataset selection and column mapping
        datasets: Dict[str, pd.DataFrame] = st.session_state.get("custom_plot_datasets", {})
        if not datasets:
            st.warning("No datasets available. Run a time-series or forward analysis first to populate datasets.")
            return
        ds_names = list(datasets.keys())
        default_ds = st.session_state.get("last_custom_dataset") or ds_names[0]
        dataset_name = st.selectbox("Dataset for thrust and O/F", ds_names, index=ds_names.index(default_ds) if default_ds in ds_names else 0, key="flight_ds_select")
        ds_df = datasets[dataset_name]
        num_cols = ds_df.select_dtypes(include=[np.number]).columns.tolist()
        # sensible defaults
        time_col = "time" if "time" in ds_df.columns else num_cols[0]
        thrust_candidates = [c for c in ds_df.columns if "Thrust" in c]
        thrust_col = thrust_candidates[0] if thrust_candidates else num_cols[1 if len(num_cols) > 1 else 0]
        mdot_o_col = "mdot_O (kg/s)" if "mdot_O (kg/s)" in ds_df.columns else None
        mdot_f_col = "mdot_F (kg/s)" if "mdot_F (kg/s)" in ds_df.columns else None
        mr_col = "MR" if "MR" in ds_df.columns else None
        mdot_total_col = "mdot_total (kg/s)" if "mdot_total (kg/s)" in ds_df.columns else None

        st.markdown("#### Map dataset columns")
        colm = st.columns(4)
        with colm[0]:
            time_col = st.selectbox("Time column [s]", ds_df.columns.tolist(), index=ds_df.columns.tolist().index(time_col) if time_col in ds_df.columns else 0, key="flight_ds_time_col")
        with colm[1]:
            thrust_col = st.selectbox("Thrust column", ds_df.columns.tolist(), index=ds_df.columns.tolist().index(thrust_col) if thrust_col in ds_df.columns else 0, key="flight_ds_thrust_col")
        with colm[2]:
            mr_col = st.selectbox("MR (O/F) column (optional)", ["None"] + ds_df.columns.tolist(), index=(0 if mr_col is None else ds_df.columns.tolist().index(mr_col)+1), key="flight_ds_mr_col")
        with colm[3]:
            mdot_total_col = st.selectbox("Total mdot column (optional)", ["None"] + ds_df.columns.tolist(), index=(0 if mdot_total_col is None else ds_df.columns.tolist().index(mdot_total_col)+1), key="flight_ds_mdtot_col")
        colm2 = st.columns(2)
        with colm2[0]:
            mdot_o_col = st.selectbox("mdot_O column (optional)", ["None"] + ds_df.columns.tolist(), index=(0 if mdot_o_col is None else ds_df.columns.tolist().index(mdot_o_col)+1), key="flight_ds_mdot_o_col")
        with colm2[1]:
            mdot_f_col = st.selectbox("mdot_F column (optional)", ["None"] + ds_df.columns.tolist(), index=(0 if mdot_f_col is None else ds_df.columns.tolist().index(mdot_f_col)+1), key="flight_ds_mdot_f_col")

    with st.form("flight_sim_form"):
        if source == "Tank pressures (constant thrust)":
            col3, col4, col5 = st.columns(3)
            with col3:
                default_burn = float(config_obj.thrust.burn_time) if getattr(config_obj, "thrust", None) and config_obj.thrust else 5.0
                burn_time = st.number_input("Burn time [s]", min_value=0.5, value=default_burn, step=0.5, key="flight_burn_time")
            with col4:
                default_m_lox = float(getattr(getattr(config_obj, "lox_tank", None), "mass", None)) if getattr(getattr(config_obj, "lox_tank", None), "mass", None) is not None else 20.0
                m_lox = st.number_input("Initial LOX mass [kg]", min_value=0.1, value=default_m_lox, step=0.1, key="flight_m_lox")
            with col5:
                default_m_fuel = float(getattr(getattr(config_obj, "fuel_tank", None), "mass", None)) if getattr(getattr(config_obj, "fuel_tank", None), "mass", None) is not None else 4.0
                m_fuel = st.number_input("Initial Fuel mass [kg]", min_value=0.1, value=default_m_fuel, step=0.1, key="flight_m_fuel")

            st.markdown("### Environment")
            env_date = None
            env_hour = 12
            if getattr(config_obj, "environment", None) and config_obj.environment:
                try:
                    y, m, d, h = list(config_obj.environment.date)
                    env_date = datetime(y, m, d).date()
                    env_hour = int(h)
                except Exception:
                    env_date = datetime.now().date()
                    env_hour = 12
            else:
                env_date = datetime.now().date()
                env_hour = 12

            cold1, cold2 = st.columns(2)
            with cold1:
                sel_date = st.date_input("Launch date", value=env_date, key="flight_env_date")
            with cold2:
                sel_hour = st.number_input("Launch hour [0-23]", min_value=0, max_value=23, value=env_hour, step=1, key="flight_env_hour")
        else:
            # Dataset mode: user defines propellant fill; burn time comes from dataset
            burn_time = None
            colpf1, colpf2 = st.columns(2)
            with colpf1:
                default_m_lox = float(getattr(getattr(config_obj, "lox_tank", None), "mass", None)) if getattr(getattr(config_obj, "lox_tank", None), "mass", None) is not None else 20.0
                m_lox = st.number_input("Initial LOX mass [kg]", min_value=0.1, value=default_m_lox, step=0.1, key="flight_m_lox_ds")
            with colpf2:
                default_m_fuel = float(getattr(getattr(config_obj, "fuel_tank", None), "mass", None)) if getattr(getattr(config_obj, "fuel_tank", None), "mass", None) is not None else 4.0
                m_fuel = st.number_input("Initial Fuel mass [kg]", min_value=0.1, value=default_m_fuel, step=0.1, key="flight_m_fuel_ds")
            sel_date = None
            sel_hour = None

        run_btn = st.form_submit_button("Run Flight Simulation", type="primary")

    # Show editors regardless of whether the form is submitted; simulation will run only if run_btn is True.

    # Create a working copy of config with overrides for flight-related fields
    try:
        working = copy.deepcopy(config_obj).model_dump()
        if source == "Tank pressures (constant thrust)":
            # ensure optional sections exist (handle None values)
            if working.get("thrust") is None:
                working["thrust"] = {}
            working["thrust"]["burn_time"] = float(burn_time)
            # update masses from user inputs (tanks now hold masses)
            if working.get("lox_tank") is None:
                working["lox_tank"] = {}
            if working.get("fuel_tank") is None:
                working["fuel_tank"] = {}
            lox_tank_work = working["lox_tank"]
            fuel_tank_work = working["fuel_tank"]
            lox_tank_work["mass"] = float(m_lox)
            fuel_tank_work["mass"] = float(m_fuel)
            # update environment date
            if working.get("environment") is None:
                working["environment"] = {}
            working["environment"]["date"] = [int(sel_date.year), int(sel_date.month), int(sel_date.day), int(sel_hour)]
        else:
            # dataset mode: always use user-defined masses; burn time set later from dataset
            if working.get("lox_tank") is None:
                working["lox_tank"] = {}
            if working.get("fuel_tank") is None:
                working["fuel_tank"] = {}
            lox_tank_work = working["lox_tank"]
            fuel_tank_work = working["fuel_tank"]
            lox_tank_work["mass"] = float(m_lox)
            fuel_tank_work["mass"] = float(m_fuel)
            # Initialize environment with default date if not set (will be editable in expander)
            if working.get("environment") is None:
                working["environment"] = {}
            if working["environment"].get("date") is None:
                now = datetime.now()
                working["environment"]["date"] = [int(now.year), int(now.month), int(now.day), 12]
    except Exception as exc:
        st.error(f"Invalid flight configuration: {exc}")
        return

    # Collapsible configuration editor
    st.subheader("Flight configuration")
    with st.expander("Environment", expanded=False):
        env = working.get("environment") if working.get("environment") is not None else {}
        env.setdefault("latitude", 35.0)
        env.setdefault("longitude", -117.0)
        env.setdefault("elevation", 0.0)
        env.setdefault("p_amb", 101325.0)
        
        # Handle date - get from existing or use default
        env_date = None
        env_hour = 12
        if env.get("date") is not None:
            try:
                y, m, d, h = list(env["date"])
                env_date = datetime(y, m, d).date()
                env_hour = int(h)
            except Exception:
                env_date = datetime.now().date()
                env_hour = 12
        else:
            env_date = datetime.now().date()
            env_hour = 12
        
        colE1, colE2 = st.columns(2)
        with colE1:
            env_lat = st.number_input("Latitude [deg]", value=float(env.get("latitude", 35.0)), key="flight_env_lat")
            env_elev = st.number_input("Elevation [m]", value=float(env.get("elevation", 0.0)), key="flight_env_elev")
            env_date_input = st.date_input("Launch date", value=env_date, key="flight_env_date_expander")
        with colE2:
            env_lon = st.number_input("Longitude [deg]", value=float(env.get("longitude", -117.0)), key="flight_env_lon")
            env_pamb = st.number_input("Ambient pressure [Pa]", value=float(env.get("p_amb", 101325.0)), key="flight_env_pamb")
            env_hour_input = st.number_input("Launch hour [0-23]", min_value=0, max_value=23, value=env_hour, step=1, key="flight_env_hour_expander")
        env["latitude"] = float(env_lat)
        env["longitude"] = float(env_lon)
        env["elevation"] = float(env_elev)
        env["p_amb"] = float(env_pamb)
        env["date"] = [int(env_date_input.year), int(env_date_input.month), int(env_date_input.day), int(env_hour_input)]
        working["environment"] = env

    with st.expander("Rocket", expanded=False):
        rocket = working.get("rocket") if working.get("rocket") is not None else {}
        rocket.setdefault("mass", 90.72)
        rocket.setdefault("inertia", [8.0, 8.0, 0.5])
        rocket.setdefault("radius", 0.1)
        rocket.setdefault("cm_wo_motor", 1.0)
        rocket.setdefault("motor_inertia", [0.1, 0.1, 0.1])
        # Motor config (nested)
        motor = rocket.setdefault("motor", {})
        motor.setdefault("dry_mass", 12.0)
        colR1, colR2, colR3 = st.columns(3)
        with colR1:
            r_mass = st.number_input("Rocket mass [kg]", value=float(rocket.get("mass", 90.72)), key="flight_rocket_mass")
            r_radius = st.number_input("Rocket radius [m]", value=float(rocket.get("radius", 0.1)), key="flight_rocket_radius")
        with colR2:
            r_cm = st.number_input("CM without motor [m]", value=float(rocket.get("cm_wo_motor", 1.0)), key="flight_rocket_cm")
            r_dry = st.number_input("Rocket dry mass [kg]", value=float(motor.get("dry_mass", 12.0)), key="flight_rocket_dry")
        with colR3:
            mi_x = st.number_input("Motor inertia X", value=float(rocket.get("motor_inertia", [0.1, 0.1, 0.1])[0]), key="flight_motor_inertia_x")
            mi_y = st.number_input("Motor inertia Y", value=float(rocket.get("motor_inertia", [0.1, 0.1, 0.1])[1]), key="flight_motor_inertia_y")
            mi_z = st.number_input("Motor inertia Z", value=float(rocket.get("motor_inertia", [0.1, 0.1, 0.1])[2]), key="flight_motor_inertia_z")
        rocket["mass"] = float(r_mass)
        rocket["radius"] = float(r_radius)
        rocket["cm_wo_motor"] = float(r_cm)
        motor["dry_mass"] = float(r_dry)
        rocket["motor"] = motor
        rocket["motor_inertia"] = [float(mi_x), float(mi_y), float(mi_z)]

        # Fins
        fins = rocket.get("fins") if rocket.get("fins") is not None else {}
        fins.setdefault("no_fins", 3)
        fins.setdefault("root_chord", 0.2)
        fins.setdefault("tip_chord", 0.1)
        fins.setdefault("fin_span", 0.3)
        fins.setdefault("fin_position", 0.0)
        colF1, colF2, colF3 = st.columns(3)
        with colF1:
            fins["no_fins"] = int(st.number_input("Fin count", value=int(fins["no_fins"]), min_value=1, step=1, key="flight_fins_count"))
            fins["root_chord"] = float(st.number_input("Root chord [m]", value=float(fins["root_chord"]), key="flight_fins_root"))
        with colF2:
            fins["tip_chord"] = float(st.number_input("Tip chord [m]", value=float(fins["tip_chord"]), key="flight_fins_tip"))
            fins["fin_span"] = float(st.number_input("Fin span [m]", value=float(fins["fin_span"]), key="flight_fins_span"))
        with colF3:
            fins["fin_position"] = float(st.number_input("Fin position [m]", value=float(fins["fin_position"]), key="flight_fins_pos"))
        rocket["fins"] = fins
        working["rocket"] = rocket

    with st.expander("Tanks", expanded=False):
        lox_tank = working.get("lox_tank") if working.get("lox_tank") is not None else {}
        fuel_tank = working.get("fuel_tank") if working.get("fuel_tank") is not None else {}
        press_tank = working.get("press_tank") if working.get("press_tank") is not None else {}
        # LOX
        lox_tank.setdefault("lox_h", 1.14)
        lox_tank.setdefault("lox_radius", 0.0762)
        lox_tank.setdefault("ox_tank_pos", 0.6)
        colL1, colL2, colL3 = st.columns(3)
        with colL1:
            lox_tank["lox_h"] = float(st.number_input("LOX tank height [m]", value=float(lox_tank["lox_h"]), key="flight_lox_h"))
        with colL2:
            lox_tank["lox_radius"] = float(st.number_input("LOX tank radius [m]", value=float(lox_tank["lox_radius"]), key="flight_lox_radius"))
        with colL3:
            lox_tank["ox_tank_pos"] = float(st.number_input("LOX tank position [m]", value=float(lox_tank["ox_tank_pos"]), key="flight_lox_pos"))
        # Fuel
        fuel_tank.setdefault("rp1_h", 0.609)
        fuel_tank.setdefault("rp1_radius", 0.0762)
        fuel_tank.setdefault("fuel_tank_pos", -0.2)
        colFu1, colFu2, colFu3 = st.columns(3)
        with colFu1:
            fuel_tank["rp1_h"] = float(st.number_input("Fuel tank height [m]", value=float(fuel_tank["rp1_h"]), key="flight_rp1_h"))
        with colFu2:
            fuel_tank["rp1_radius"] = float(st.number_input("Fuel tank radius [m]", value=float(fuel_tank["rp1_radius"]), key="flight_rp1_radius"))
        with colFu3:
            fuel_tank["fuel_tank_pos"] = float(st.number_input("Fuel tank position [m]", value=float(fuel_tank["fuel_tank_pos"]), key="flight_rp1_pos"))
        # Pressurant (optional)
        if press_tank is None:
            press_tank = {}
        press_tank.setdefault("press_h", 0.457)
        press_tank.setdefault("press_radius", 0.0762)
        press_tank.setdefault("pres_tank_pos", 1.2)
        colP1, colP2, colP3 = st.columns(3)
        with colP1:
            press_tank["press_h"] = float(st.number_input("Pressurant tank height [m]", value=float(press_tank["press_h"]), key="flight_press_h"))
        with colP2:
            press_tank["press_radius"] = float(st.number_input("Pressurant tank radius [m]", value=float(press_tank["press_radius"]), key="flight_press_radius"))
        with colP3:
            press_tank["pres_tank_pos"] = float(st.number_input("Pressurant tank position [m]", value=float(press_tank["pres_tank_pos"]), key="flight_press_pos"))
        working["lox_tank"] = lox_tank
        working["fuel_tank"] = fuel_tank
        working["press_tank"] = press_tank

    with st.expander("Nozzle", expanded=False):
        nozzle = working.get("nozzle") if working.get("nozzle") is not None else {}
        nozzle.setdefault("A_throat", 0.00156235266901)
        nozzle.setdefault("A_exit", 0.00831498636119)
        nozzle.setdefault("expansion_ratio", 6.54)
        nozzle.setdefault("efficiency", 0.98)
        colN1, colN2 = st.columns(2)
        with colN1:
            nozzle["A_throat"] = float(st.number_input("Throat area [m²]", value=float(nozzle["A_throat"]), key="flight_noz_at"))
            nozzle["expansion_ratio"] = float(st.number_input("Expansion ratio (Ae/At)", value=float(nozzle["expansion_ratio"]), key="flight_noz_er"))
        with colN2:
            nozzle["A_exit"] = float(st.number_input("Exit area [m²]", value=float(nozzle["A_exit"]), key="flight_noz_ae"))
            nozzle["efficiency"] = float(st.number_input("Nozzle efficiency", value=float(nozzle["efficiency"]), key="flight_noz_eta"))
        working["nozzle"] = nozzle

    with st.expander("Fluids (properties)", expanded=False):
        fluids = working.get("fluids") if working.get("fluids") is not None else {}
        ox = fluids.get("oxidizer") if fluids.get("oxidizer") is not None else {}
        fu = fluids.get("fuel") if fluids.get("fuel") is not None else {}
        ox.setdefault("name", "LOX")
        ox.setdefault("density", 1140.0)
        ox.setdefault("temperature", 90.0)
        fu.setdefault("name", "RP-1")
        fu.setdefault("density", 780.0)
        fu.setdefault("temperature", 293.0)
        colOx1, colOx2, colOx3 = st.columns(3)
        with colOx1:
            ox["name"] = st.text_input("Oxidizer name", value=str(ox["name"]), key="flight_ox_name")
        with colOx2:
            ox["density"] = float(st.number_input("Oxidizer density [kg/m³]", value=float(ox["density"]), key="flight_ox_density"))
        with colOx3:
            ox["temperature"] = float(st.number_input("Oxidizer temperature [K]", value=float(ox["temperature"]), key="flight_ox_temp"))
        colFu1, colFu2, colFu3 = st.columns(3)
        with colFu1:
            fu["name"] = st.text_input("Fuel name", value=str(fu["name"]), key="flight_fu_name")
        with colFu2:
            fu["density"] = float(st.number_input("Fuel density [kg/m³]", value=float(fu["density"]), key="flight_fu_density"))
        with colFu3:
            fu["temperature"] = float(st.number_input("Fuel temperature [K]", value=float(fu["temperature"]), key="flight_fu_temp"))
        fluids["oxidizer"] = ox
        fluids["fuel"] = fu
        working["fluids"] = fluids

    # Validate edited working config
    try:
        config_for_flight = PintleEngineConfig(**working)
    except Exception as exc:
        st.error(f"Edited flight configuration is invalid: {exc}")
        return

    if run_btn:
        # If dataset-driven, build Functions and override burn time
        if source == "Dataset (time-varying)":
            try:
                # Convert time column and thrust values
                t_vals = to_elapsed_seconds(ds_df[time_col])
                thrust_vals = np.asarray(ds_df[thrust_col], dtype=float)
            except Exception as exc:
                st.error(f"Invalid dataset columns: {exc}")
                return
            # Unit handling for thrust
            if "(kN)" in thrust_col:
                thrust_vals_SI = thrust_vals * 1000.0
            elif "(N)" in thrust_col:
                thrust_vals_SI = thrust_vals
            else:
                # Assume N if unitless; provide a toggle?
                thrust_vals_SI = thrust_vals
            # mdot handling
            if mdot_o_col != "None" and mdot_f_col != "None":
                mdot_O_vals = np.asarray(ds_df[mdot_o_col], dtype=float)
                mdot_F_vals = np.asarray(ds_df[mdot_f_col], dtype=float)
            elif mr_col != "None" and mdot_total_col != "None":
                MR_vals = np.asarray(ds_df[mr_col], dtype=float)
                mdot_total_vals = np.asarray(ds_df[mdot_total_col], dtype=float)
                mdot_O_vals = mdot_total_vals * (MR_vals / (1.0 + MR_vals))
                mdot_F_vals = mdot_total_vals * (1.0 / (1.0 + MR_vals))
            else:
                st.error("Provide either mdot_O and mdot_F columns, or MR and total mdot.")
                return

            # Sort by time and drop duplicates if needed
            order = np.argsort(t_vals)
            t_vals = t_vals[order]
            thrust_vals_SI = thrust_vals_SI[order]
            mdot_O_vals = mdot_O_vals[order]
            mdot_F_vals = mdot_F_vals[order]
            
            # Store original arrays for later (before truncation)
            t_vals_original = t_vals.copy()
            thrust_vals_original = thrust_vals_SI.copy()
            mdot_O_vals_original = mdot_O_vals.copy()
            mdot_F_vals_original = mdot_F_vals.copy()

            # Deduce burn time from dataset
            if len(t_vals) < 2:
                st.error("Dataset must contain at least two time samples to define a burn duration.")
                return
            burn_time_ds = float(np.max(t_vals) - np.min(t_vals))
            if not np.isfinite(burn_time_ds) or burn_time_ds <= 0.0:
                st.error("Dataset time axis must increase (duration must be > 0 s). Check the selected time column.")
                return
            
            # Check for tank underfill and truncate dataset if needed
            m_lox0 = float(m_lox)
            m_fuel0 = float(m_fuel)
            
            # Normalize time to start at 0
            t_min = float(np.min(t_vals))
            t_vals_normalized = t_vals - t_min
            
            # Build temporary Functions for underfill detection
            mdot_lox_temp = build_rp_function(t_vals_normalized, mdot_O_vals)
            mdot_fuel_temp = build_rp_function(t_vals_normalized, mdot_F_vals)
            
            # Detect underfill by integrating mdot arrays
            lox_cutoff = detect_tank_underfill_time(mdot_lox_temp, m_lox0, burn_time_ds)
            fuel_cutoff = detect_tank_underfill_time(mdot_fuel_temp, m_fuel0, burn_time_ds)
            
            # Find earliest cutoff
            cutoff_time = None
            if lox_cutoff is not None and fuel_cutoff is not None:
                cutoff_time = min(lox_cutoff, fuel_cutoff)
            elif lox_cutoff is not None:
                cutoff_time = lox_cutoff
            elif fuel_cutoff is not None:
                cutoff_time = fuel_cutoff
            
            # Truncate arrays if needed
            if cutoff_time is not None and cutoff_time < burn_time_ds:
                # Find index where time exceeds cutoff
                trunc_idx = np.searchsorted(t_vals_normalized, cutoff_time, side='right')
                if trunc_idx < len(t_vals_normalized):
                    t_vals_normalized = t_vals_normalized[:trunc_idx+1]
                    thrust_vals_SI = thrust_vals_SI[:trunc_idx+1]
                    mdot_O_vals = mdot_O_vals[:trunc_idx+1]
                    mdot_F_vals = mdot_F_vals[:trunc_idx+1]
                    burn_time_ds = float(t_vals_normalized[-1])
                    st.info(f"Truncated burn at {cutoff_time:.2f} s due to propellant depletion")
            
            # Build RocketPy Functions
            thrust_func = build_rp_function(t_vals_normalized, thrust_vals_SI)
            mdot_lox_func = build_rp_function(t_vals_normalized, mdot_O_vals)
            mdot_fuel_func = build_rp_function(t_vals_normalized, mdot_F_vals)
            
            # Update config with dataset-derived burn time and masses
            if working.get("thrust") is None:
                working["thrust"] = {}
            working["thrust"]["burn_time"] = float(burn_time_ds)
            
            # Re-validate config
            try:
                config_for_flight = PintleEngineConfig(**working)
            except Exception as exc:
                st.error(f"Invalid flight configuration after dataset processing: {exc}")
                return
            
            # Run flight simulation
            with st.spinner("Running flight simulation..."):
                try:
                    result = setup_flight(config_for_flight, thrust_func, mdot_lox_func, mdot_fuel_func, plot_results=False)
                    apogee = result["apogee"]
                    max_velocity = result["max_velocity"]
                    flight_obj = result["flight"]
                    trunc_info = result.get("truncation_info", {})
                    
                    st.success("Flight simulation completed!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Apogee", f"{apogee:.1f} m")
                    with col2:
                        st.metric("Max Velocity", f"{max_velocity:.1f} m/s")
                    
                    # Add truncated data to dataset
                    truncated_df = pd.DataFrame({
                        "Time (s)": t_vals_normalized,
                        "Thrust_truncated (N)": thrust_vals_SI,
                        "mdot_O (kg/s)": mdot_O_vals,
                        "mdot_F (kg/s)": mdot_F_vals,
                    })
                    
                    # Extract flight data
                    flight_time, flight_z, flight_vz = extract_flight_series(flight_obj)
                    if flight_time.size > 0:
                        truncated_df["Altitude above sea level (m)"] = np.interp(
                            t_vals_normalized, 
                            flight_time, 
                            flight_z
                        )
                        truncated_df["Vertical Velocity (m/s)"] = np.interp(
                            t_vals_normalized,
                            flight_time,
                            flight_vz
                        )
                    
                    st.subheader("Truncated Dataset")
                    st.dataframe(truncated_df)
                    
                    # Plot flight results
                    if flight_time.size > 0:
                        plot_flight_results(flight_time, flight_z, flight_vz)
                        
                        # Create comprehensive flight dataframe for download/plotting
                        flight_df = pd.DataFrame({
                            "Time (s)": flight_time,
                            "Altitude (m)": flight_z,
                            "Velocity_z (m/s)": flight_vz
                        })
                        
                        # Determine effective cutoff time for CSV generation
                        # Use the one from simulation result if available, otherwise fall back to configured burn time
                        sim_cutoff = trunc_info.get("cutoff_time") if trunc_info.get("truncated") else None
                        # Also respect the burn time from config (which matches dataset duration/truncation)
                        config_burn_time = config_for_flight.thrust.burn_time
                        
                        # The effective cutoff is the minimum of available limits
                        effective_cutoff = config_burn_time
                        if sim_cutoff is not None:
                            effective_cutoff = min(effective_cutoff, sim_cutoff)

                        # Interpolate input thrust/mdot onto flight timeline
                        # Ensure we zero out values after the effective cutoff time
                        f_thrust = [float(thrust_func(t)) if t <= effective_cutoff else 0.0 for t in flight_time]
                        f_mdot_o = [float(mdot_lox_func(t)) if t <= effective_cutoff else 0.0 for t in flight_time]
                        f_mdot_f = [float(mdot_fuel_func(t)) if t <= effective_cutoff else 0.0 for t in flight_time]
                        
                        flight_df["Thrust (N)"] = f_thrust
                        flight_df["mdot_O (kg/s)"] = f_mdot_o
                        flight_df["mdot_F (kg/s)"] = f_mdot_f

                        with st.expander("Thrust curve (flight)"):
                            st.plotly_chart(
                                px.line(flight_df, x="Time (s)", y="Thrust (N)", title="Thrust vs Time (Flight)"), 
                                width='stretch', 
                                key="flight_thrust_plot_ds_actual"
                            )

                        render_rocket_view(flight_obj)
                        plot_additional_rocket_plots(flight_obj, flight_time)
                        
                        # Download button for full flight data
                        csv_data = flight_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Flight Dataset (CSV)",
                            data=csv_data,
                            file_name="flight_simulation_data_dataset_mode.csv",
                            mime="text/csv"
                        )
                        
                except Exception as exc:
                    st.error(f"Flight simulation failed: {exc}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            # Tank pressures mode: constant thrust
            # Evaluate engine at specified tank pressures to get actual thrust and mdot values
            try:
                eval_result = runner.evaluate(
                    P_tank_O=P_tank_O_psi * PSI_TO_PA,
                    P_tank_F=P_tank_F_psi * PSI_TO_PA
                )
                thrust_val = float(eval_result.get("F", 1000.0))
                mdot_lox_actual = float(eval_result["mdot_O"])
                mdot_fuel_actual = float(eval_result["mdot_F"])
            except Exception as exc:
                st.warning(f"Could not evaluate engine for thrust and mdot values: {exc}. Using estimates.")
                # Fallback to estimates
                thrust_val = 1000.0
                mdot_lox_actual = float(m_lox) / float(burn_time) * 0.8
                mdot_fuel_actual = float(m_fuel) / float(burn_time) * 0.2
            
            # Build constant functions
            times_const = np.array([0.0, float(burn_time)])
            thrust_const = np.array([thrust_val, thrust_val])
            mdot_lox_const = np.array([mdot_lox_actual, mdot_lox_actual])
            mdot_fuel_const = np.array([mdot_fuel_actual, mdot_fuel_actual])
            
            thrust_func = build_rp_function(times_const, thrust_const)
            mdot_lox_func = build_rp_function(times_const, mdot_lox_const)
            mdot_fuel_func = build_rp_function(times_const, mdot_fuel_const)
            
            # Update config
            if working.get("thrust") is None:
                working["thrust"] = {}
            working["thrust"]["burn_time"] = float(burn_time)
            if working.get("lox_tank") is None:
                working["lox_tank"] = {}
            if working.get("fuel_tank") is None:
                working["fuel_tank"] = {}
            working["lox_tank"]["mass"] = float(m_lox)
            working["fuel_tank"]["mass"] = float(m_fuel)
            
            # Re-validate config
            try:
                config_for_flight = PintleEngineConfig(**working)
            except Exception as exc:
                st.error(f"Invalid flight configuration: {exc}")
                return
            
            # Run flight simulation
            with st.spinner("Running flight simulation..."):
                try:
                    result = setup_flight(config_for_flight, thrust_func, mdot_lox_func, mdot_fuel_func, plot_results=False)
                    apogee = result["apogee"]
                    max_velocity = result["max_velocity"]
                    flight_obj = result["flight"]
                    trunc_info = result.get("truncation_info", {})
                    
                    st.success("Flight simulation completed!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Apogee", f"{apogee:.1f} m")
                    with col2:
                        st.metric("Max Velocity", f"{max_velocity:.1f} m/s")
                    
                    # Extract and plot flight data
                    flight_time, flight_z, flight_vz = extract_flight_series(flight_obj)
                    if flight_time.size > 0:
                        # Add flight data to truncated_df or create new DF for download
                        # To align time series, we might need to interpolate since RocketPy uses adaptive steps
                        # For now, let's create a separate dataframe for flight results
                        flight_df = pd.DataFrame({
                            "Time (s)": flight_time,
                            "Altitude (m)": flight_z,
                            "Velocity_z (m/s)": flight_vz
                        })
                        
                        # Determine effective cutoff time for CSV generation
                        # Use the one from simulation result if available, otherwise use configured burn time
                        effective_cutoff = trunc_info.get("cutoff_time") if trunc_info.get("truncated") else config_for_flight.thrust.burn_time

                        # Also interpolate thrust/mdot onto this timeline for a complete dataset
                        # thrust_func(t), mdot_lox_func(t), mdot_fuel_func(t) are RocketPy Functions
                        # Ensure we zero out values after the effective cutoff time
                        f_thrust = [float(thrust_func(t)) if t <= effective_cutoff else 0.0 for t in flight_time]
                        f_mdot_o = [float(mdot_lox_func(t)) if t <= effective_cutoff else 0.0 for t in flight_time]
                        f_mdot_f = [float(mdot_fuel_func(t)) if t <= effective_cutoff else 0.0 for t in flight_time]
                        
                        flight_df["Thrust (N)"] = f_thrust
                        flight_df["mdot_O (kg/s)"] = f_mdot_o
                        flight_df["mdot_F (kg/s)"] = f_mdot_f
                        
                        plot_flight_results(flight_time, flight_z, flight_vz)
                        
                        with st.expander("Thrust curve (flight)"):
                            st.plotly_chart(
                                px.line(flight_df, x="Time (s)", y="Thrust (N)", title="Thrust vs Time (Flight)"), 
                                width='stretch', 
                                key="flight_thrust_plot_actual"
                            )
                        
                        render_rocket_view(flight_obj)
                        plot_additional_rocket_plots(flight_obj, flight_time)
                        
                        # Download button
                        csv_data = flight_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Flight Dataset (CSV)",
                            data=csv_data,
                            file_name="flight_simulation_data.csv",
                            mime="text/csv"
                        )

                except Exception as exc:
                    st.error(f"Flight simulation failed: {exc}")
                    import traceback
                    st.code(traceback.format_exc())


def main():
    st.set_page_config(page_title="Pintle Injector Engine Pipeline", layout="wide")
    st.title("Pintle Injector Engine Pipeline")

    st.sidebar.header("Configuration")
    uploaded_config = st.sidebar.file_uploader("Upload custom YAML config", type=["yaml", "yml"])

    try:
        config_obj, config_label = load_config_state(uploaded_config)
        st.sidebar.success(f"Using configuration: {config_label}")
    except ValueError as exc:
        st.sidebar.error(str(exc))
        st.stop()

    init_session_state()

    st.sidebar.subheader("Edit Configuration")
    st.sidebar.write("### Display Units")
    st.session_state["display_pressure_unit"] = st.sidebar.selectbox(
        "Pressure unit",
        list(SENSOR_UNITS["pressure"].keys()),
        index=list(SENSOR_UNITS["pressure"].keys()).index(st.session_state.get("display_pressure_unit", "psi")),
    )
    st.session_state["display_mass_unit"] = st.sidebar.selectbox(
        "Mass flow unit",
        list(SENSOR_UNITS["mass_flow"].keys()),
        index=list(SENSOR_UNITS["mass_flow"].keys()).index(st.session_state.get("display_mass_unit", "kg/s")),
    )
    st.session_state["display_length_unit"] = st.sidebar.selectbox(
        "Length unit",
        list(SENSOR_UNITS["length"].keys()),
        index=list(SENSOR_UNITS["length"].keys()).index(st.session_state.get("display_length_unit", "mm")),
    )

    # Check if config was updated in session state (e.g., from chamber design tab or config editor)
    # This can happen if the page reruns after a config update
    # Load the updated config BEFORE config_editor so it doesn't get overwritten
    if "config_dict" in st.session_state and st.session_state.get("config_updated", False):
        # Use the updated config from session state
        config_dict = st.session_state["config_dict"]
        config_obj = PintleEngineConfig(**config_dict)
        if st.session_state.get("config_updated_from_chamber_design", False):
            st.sidebar.info("✓ Configuration updated from Chamber Design")
            st.session_state["config_updated_from_chamber_design"] = False
        elif st.session_state.get("config_updated_from_editor", False):
            st.sidebar.info("✓ Configuration updated from sidebar")
        # Clear the flag after using it
        st.session_state["config_updated"] = False

    config_obj = config_editor(config_obj)
    # Use exclude_none=False to preserve all fields including None values
    # BUT: Don't overwrite config_dict if it was just updated by chamber_design_view
    # Only update if config wasn't just updated from chamber design
    if not st.session_state.get("config_updated_from_chamber_design", False):
        config_dict = config_obj.model_dump(exclude_none=False)
        st.session_state["config_dict"] = config_dict
    else:
        # Use the updated config_dict from chamber design
        config_dict = st.session_state.get("config_dict", config_obj.model_dump(exclude_none=False))
        config_obj = PintleEngineConfig(**config_dict)
    
    st.session_state["config_label"] = config_label

    # Import json for config hashing
    import json
    
    # Get cached runner info
    cached_runner = st.session_state.get("cached_runner")
    cached_config_hash = st.session_state.get("cached_config_hash")
    
    # Get config_dict BEFORE chamber design tab runs to compare later
    config_dict_before = st.session_state.get("config_dict", config_dict)
    config_str_before = json.dumps(config_dict_before, sort_keys=True, default=str)
    config_hash_before = hash(config_str_before)
    
    # Create/update runner using config_dict from session state (source of truth)
    runner_config_dict = st.session_state.get("config_dict", config_dict)
    runner_config_str = json.dumps(runner_config_dict, sort_keys=True, default=str)
    runner_config_hash = hash(runner_config_str)
    
    if cached_runner is None or cached_config_hash != runner_config_hash:
        # Config changed or no cached runner - create new one
        # Use config_dict from session state (source of truth) to create runner
        runner_config_obj = PintleEngineConfig(**runner_config_dict)
        runner = PintleEngineRunner(runner_config_obj)
        st.session_state["cached_runner"] = runner
        st.session_state["cached_config_hash"] = runner_config_hash
    else:
        # Reuse cached runner
        runner = cached_runner

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Forward Mode",
        "Inverse Mode",
        "Time-Series Analysis",
        "COPV Sizing",
        "Plots & Analysis",
        "Custom Plot Builder",
        "Flight Simulation",
        "Chamber Design",
        "Graphite Insert",
    ])
    
    # Run chamber design tab - it updates st.session_state["config_dict"] when Calculate is pressed
    # The download button uses st.session_state["config_dict"] which is the source of truth
    with tab8:
        config_obj = chamber_design_view(config_obj, runner)
        # chamber_design_view updates st.session_state["config_dict"] with calculated values
    
    # Run graphite insert tab - it updates st.session_state["config_dict"] when Calculate is pressed
    with tab9:
        config_obj = graphite_insert_view(config_obj, runner)
        # graphite_insert_view updates st.session_state["config_dict"] with calculated values
    
    # CRITICAL: After chamber design tab runs, check if config was updated and recreate runner
    # This ensures Forward Mode and other tabs use the updated geometry immediately
    updated_config_dict = st.session_state.get("config_dict", config_dict)
    updated_config_str = json.dumps(updated_config_dict, sort_keys=True, default=str)
    updated_config_hash = hash(updated_config_str)
    
    # If config was updated by chamber design (hash changed), recreate runner with new config
    if config_hash_before != updated_config_hash:
        runner_config_obj = PintleEngineConfig(**updated_config_dict)
        runner = PintleEngineRunner(runner_config_obj)
        st.session_state["cached_runner"] = runner
        st.session_state["cached_config_hash"] = updated_config_hash
        # Show info message that runner was updated
        st.sidebar.success("✓ Runner updated with new chamber geometry!")
    
    with tab1:
        forward_view(runner)
    with tab2:
        inverse_view(runner, config_label)
    with tab3:
        timeseries_view(runner, config_label)
    with tab4:
        copv_view(config_obj)
    with tab5:
        plots_analysis_view(runner)
    with tab6:
        custom_plot_builder()
    with tab7:
        flight_sim_view(runner, config_obj, config_label)


if __name__ == "__main__":
    main()
