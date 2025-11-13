# -*- coding: utf-8 -*-
"""Streamlit UI for the pintle engine pipeline."""

from __future__ import annotations

import math
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import sys

import copy

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pintle_pipeline.io import load_config
from pintle_pipeline.config_schemas import PintleEngineConfig
from pintle_pipeline.time_series import generate_pressure_profile
from pintle_models.runner import PintleEngineRunner
from examples.pintle_engine.interactive_pipeline import solve_for_thrust, solve_for_thrust_and_MR, ThrustSolveError
from examples.pintle_engine.flight_sim import setup_flight

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
    return config.model_dump()


def format_value(value: float, unit_type: str, unit_label: str) -> str:
    factor, fmt = SENSOR_UNITS[unit_type][unit_label]
    return fmt % (value * factor)


# ============================================================================
# Flight Simulation Helper Functions
# ============================================================================

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
        return np.asarray(series_obj.get_source(), dtype=float)
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
    st.plotly_chart(px.line(df, x="time", y="Altitude (m)", title="Altitude vs Time"), use_container_width=True, key=f"flight_alt_plot{key_suffix}")
    st.plotly_chart(px.line(df, x="time", y="Vertical Velocity (m/s)", title="Vertical Velocity vs Time"), use_container_width=True, key=f"flight_vel_plot{key_suffix}")


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
        st.pyplot(fig, clear_figure=True, use_container_width=True)
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
            st.plotly_chart(px.line(df, x="time", y=label, title=label), use_container_width=True, key=f"flight_{label.replace(' ', '_').lower()}{key_suffix}")


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
            st.write(
                f"Recession rate: {ablative['recession_rate']*1e6:.3f} µm/s | Effective heat flux: {ablative['effective_heat_flux']/1000:.1f} kW/m²"
            )
            if ablative.get("turbulence_multiplier") is not None:
                st.write(
                    f"Turbulence multiplier: {ablative['turbulence_multiplier']:.2f} | Incident heat flux: {ablative.get('incident_heat_flux', np.nan)/1000:.1f} kW/m²"
                )

        metadata = cooling.get("metadata", {})
        gas_turb = metadata.get("gas_turbulence_intensity") if isinstance(metadata, dict) else None
        if gas_turb is not None and np.isfinite(gas_turb):
            st.caption("Chamber turbulence")
            st.write(f"Gas-side turbulence intensity: {gas_turb:.3f}")


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
        "Ablative Heat Removed (kW)": ablative.get("heat_removed", np.nan) / 1000.0 if ablative else np.nan,
        "Ablative Recession (µm/s)": ablative.get("recession_rate", np.nan) * 1e6 if ablative and ablative.get("recession_rate") is not None else np.nan,
        "Injector Turbulence O": diag.get("turbulence_intensity_O", np.nan),
        "Injector Turbulence F": diag.get("turbulence_intensity_F", np.nan),
        "Injector Turbulence Mix": diag.get("turbulence_intensity_mix", np.nan),
        "Film Turbulence Multiplier": film.get("turbulence_multiplier", np.nan) if film else np.nan,
        "Ablative Turbulence Multiplier": ablative.get("turbulence_multiplier", np.nan) if ablative else np.nan,
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
    
    if use_time_varying:
        # Use time-varying method for ablative geometry evolution
        results = runner.evaluate_arrays_with_time(times, P_tank_O_pa, P_tank_F_pa)
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
    diag_list = results.get("diagnostics", [])
    errors: list[str] = []

    for diag in diag_list:
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
        ablative_heat_removed.append(ablative.get("heat_removed", np.nan) / 1000.0 if ablative else np.nan)
        ablative_heat_flux.append(ablative.get("effective_heat_flux", np.nan) / 1000.0 if ablative else np.nan)
        injector_turbulence_mix.append(diag.get("turbulence_intensity_mix", np.nan) if isinstance(diag, dict) else np.nan)

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
        df_dict["Ablative Heat Removed (kW)"] = np.asarray(ablative_heat_removed, dtype=float)
        df_dict["Ablative Heat Flux (kW/m²)"] = np.asarray(ablative_heat_flux, dtype=float)

    # Add ablative geometry evolution data if available
    if "Lstar" in results:
        df_dict["L* (mm)"] = np.asarray(results["Lstar"], dtype=float) * 1000.0
        df_dict["Chamber Volume (cm³)"] = np.asarray(results["V_chamber"], dtype=float) * 1e6
        df_dict["Throat Area (mm²)"] = np.asarray(results["A_throat"], dtype=float) * 1e6
        df_dict["Cumulative Chamber Recession (µm)"] = np.asarray(results["recession_chamber"], dtype=float) * 1e6
        df_dict["Cumulative Throat Recession (µm)"] = np.asarray(results["recession_throat"], dtype=float) * 1e6
        if "throat_recession_multiplier" in results:
            df_dict["Throat Recession Multiplier"] = np.asarray(results["throat_recession_multiplier"], dtype=float)

    mixture_eff = [diag.get("mixture_efficiency", np.nan) if isinstance(diag, dict) else np.nan for diag in diag_list]
    cooling_eff = [diag.get("cooling_efficiency", np.nan) if isinstance(diag, dict) else np.nan for diag in diag_list]
    df_dict["Mixture Efficiency"] = np.asarray(mixture_eff, dtype=float)
    df_dict["Cooling Efficiency"] = np.asarray(cooling_eff, dtype=float)

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
    total_impulse = float(np.trapz(thrust_column * 1000.0, df["time"])) / 1000.0  # kN·s

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

    if "Ablative Heat Removed (kW)" in df.columns and not df["Ablative Heat Removed (kW)"].isna().all():
        abl_heat_fig = px.line(
            df,
            x="time",
            y="Ablative Heat Removed (kW)",
            markers=False,
            title="Ablative Heat Removal",
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
            st.plotly_chart(lstar_fig, use_container_width=True, key="ts_lstar_evol")
        
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
            st.plotly_chart(recession_fig, use_container_width=True, key="ts_recession_cumul")
        
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
            st.plotly_chart(geom_fig, use_container_width=True, key="ts_geom_growth_pct")
        
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
                st.plotly_chart(mult_fig, use_container_width=True, key="ts_throat_mult_phys")
        
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

        color_options = ["None"] + [col for col in numeric_columns if col not in {x_col, *y_cols}]
        color_by = st.selectbox("Color by (optional)", color_options)

        if color_by != "None" and len(y_cols) > 1:
            st.warning("When using a color grouping, select a single Y-axis variable.")
            return

        log_x = st.checkbox("Logarithmic X-axis", value=False)
        log_y = st.checkbox("Logarithmic Y-axis", value=False)
        show_markers = st.checkbox("Show markers", value=(plot_type == "Scatter"))

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

        if log_x:
            fig.update_xaxes(type="log")
        if log_y:
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
                st.plotly_chart(fig, use_container_width=True, key="inverse_2d_convergence")
        
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


def load_config_state(uploaded_file) -> Tuple[PintleEngineConfig, str]:
    if "config_dict" not in st.session_state:
        st.session_state["config_dict"] = get_default_config_dict()
        st.session_state["config_label"] = str(CONFIG_PATH)

    if uploaded_file is not None:
        try:
            config_text = uploaded_file.getvalue().decode("utf-8")
            config_dict = yaml.safe_load(config_text)
            PintleEngineConfig(**config_dict)
            st.session_state["config_dict"] = config_dict
            st.session_state["config_label"] = uploaded_file.name
        except Exception as exc:
            raise ValueError(f"Failed to load uploaded configuration: {exc}") from exc

    try:
        config_obj = PintleEngineConfig(**st.session_state["config_dict"])
    except Exception as exc:
        raise ValueError(f"Invalid configuration state: {exc}") from exc

    ox_choice = detect_fluid_choice(config_obj.fluids["oxidizer"].model_dump())
    fuel_choice = detect_fluid_choice(config_obj.fluids["fuel"].model_dump())
    st.session_state.setdefault("oxidizer_choice", ox_choice)
    st.session_state.setdefault("fuel_choice", fuel_choice)
    st.session_state.setdefault("injector_choice", "Pintle")

    return config_obj, st.session_state["config_label"]


def config_editor(config: PintleEngineConfig) -> PintleEngineConfig:
    working_copy = copy.deepcopy(st.session_state.get("config_dict", config.model_dump()))
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
        ox["density"] = st.number_input("Oxidizer density [kg/m³]", min_value=200.0, max_value=3000.0, value=float(ox["density"]))
        ox["viscosity"] = st.number_input("Oxidizer viscosity [Pa·s]", min_value=1e-5, max_value=1e-2, value=float(ox["viscosity"]))
        ox["surface_tension"] = st.number_input("Oxidizer surface tension [N/m]", min_value=1e-3, max_value=0.05, value=float(ox["surface_tension"]))
        ox["vapor_pressure"] = st.number_input("Oxidizer vapor pressure [Pa]", min_value=0.0, max_value=3e6, value=float(ox["vapor_pressure"]))

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
        fuel["density"] = st.number_input("Fuel density [kg/m³]", min_value=200.0, max_value=3000.0, value=float(fuel["density"]))
        fuel["viscosity"] = st.number_input("Fuel viscosity [Pa·s]", min_value=1e-5, max_value=1e-2, value=float(fuel["viscosity"]))
        fuel["surface_tension"] = st.number_input("Fuel surface tension [N/m]", min_value=1e-3, max_value=0.05, value=float(fuel["surface_tension"]))
        fuel["vapor_pressure"] = st.number_input("Fuel vapor pressure [Pa]", min_value=0.0, max_value=3e6, value=float(fuel["vapor_pressure"]))

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
        ablative_cfg["blowing_efficiency"] = st.number_input("Blowing efficiency", min_value=0.0, max_value=1.0, value=float(ablative_cfg.get("blowing_efficiency", 0.8)), key="ablative_blowing_efficiency")
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

        st.markdown("### Chamber & Nozzle")
        chamber = working_copy["chamber"]
        nozzle = working_copy["nozzle"]
        
        # Chamber geometry specification mode (similar to injector type selector)
        st.markdown("**Chamber Geometry Specification:**")
        
        # Track previous mode to clear state on switch (like injector does)
        previous_geom_mode = st.session_state.get("active_chamber_geom_mode", "Volume + Throat Area")
        
        geom_mode = st.radio(
            "Choose how to specify chamber geometry",
            ["Volume + Throat Area", "L* (Characteristic Length)"],
            horizontal=True,
            key="chamber_geom_mode",
            help="L* = Volume / A_throat. Choose one mode to avoid conflicts."
        )
        
        # Clear conflicting keys when mode changes (like injector does)
        if previous_geom_mode != geom_mode:
            # Clear keys that might conflict
            for key in list(st.session_state.keys()):
                if key.startswith("chamber_volume_input") or key.startswith("chamber_lstar_input") or key.startswith("chamber_athroat_"):
                    # Don't delete, but mark for refresh
                    pass
        st.session_state["active_chamber_geom_mode"] = geom_mode
        
        if geom_mode == "Volume + Throat Area":
            # User specifies volume and throat, L* is calculated
            chamber["volume"] = st.number_input(
                "Chamber volume [m³]", 
                min_value=1e-6, 
                max_value=1.0, 
                value=float(chamber.get("volume", 1e-3)), 
                format="%.6f",
                key="chamber_volume_input"
            )
            chamber["A_throat"] = st.number_input(
                "Throat area [m²]", 
                min_value=1e-5, 
                max_value=0.01, 
                value=float(chamber.get("A_throat", 1e-3)), 
                format="%.6f",
                key="chamber_athroat_input"
            )
            # Calculate and display L* (but don't store in config to avoid override)
            calculated_lstar = chamber["volume"] / chamber["A_throat"]
            chamber["Lstar"] = None  # Set to None so solver calculates from V/A
            st.info(f"✓ Calculated L* = **{calculated_lstar:.4f} m** ({calculated_lstar*1000:.2f} mm)")
        else:
            # User specifies L* and throat, volume is calculated
            chamber["A_throat"] = st.number_input(
                "Throat area [m²]", 
                min_value=1e-5, 
                max_value=0.01, 
                value=float(chamber.get("A_throat", 1e-3)), 
                format="%.6f",
                key="chamber_athroat_lstar_mode"
            )
            
            # Get current L* value (calculate from volume if not set)
            if chamber.get("Lstar") is not None and float(chamber.get("Lstar", 0)) > 0:
                default_lstar = float(chamber["Lstar"])
            else:
                # Calculate from volume and throat
                current_volume = float(chamber.get("volume", 1e-3))
                current_athroat = float(chamber.get("A_throat", 1e-3))
                default_lstar = current_volume / current_athroat if current_athroat > 0 else 1.0
            
            # Use regular number_input instead of length_number_input for better control
            lstar_m = st.number_input(
                "Characteristic length L* [m]",
                min_value=0.1,
                max_value=5.0,
                value=default_lstar,
                step=0.05,
                format="%.4f",
                key="chamber_lstar_input"
            )
            chamber["Lstar"] = lstar_m
            # Calculate and store volume from L*
            chamber["volume"] = chamber["Lstar"] * chamber["A_throat"]
            st.info(f"✓ Calculated Volume = **{chamber['volume']:.6f} m³** ({chamber['volume']*1e6:.2f} cm³)")
        
        chamber["length"] = length_number_input(
            "Chamber length",
            float(chamber.get("length", 0.5)),
            min_m=0.01,
            max_m=3.0,
            step_m=0.01,
            key="chamber_length",
        )
        nozzle["expansion_ratio"] = st.number_input("Expansion ratio (Ae/At)", min_value=1.0, max_value=200.0, value=float(nozzle["expansion_ratio"]), format="%.4f")

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
            st.success("Configuration updated.")
            return new_config
        except Exception as exc:
            st.error(f"Invalid configuration: {exc}")
            return config

    return PintleEngineConfig(**st.session_state["config_dict"])


def flight_sim_view(runner: PintleEngineRunner, config_obj: PintleEngineConfig, config_label: str) -> None:
    """Flight simulation tab - integrate engine performance with RocketPy trajectory simulation."""
    if not ROCKETPY_AVAILABLE:
        st.error("RocketPy is not installed. Install it with: `pip install rocketpy`")
        st.info("Flight simulation requires RocketPy for trajectory propagation.")
        return
    
    st.header("Flight Simulation")
    st.write("Simulate rocket flight using engine performance data with RocketPy.")
    
    st.info("⚠️ Flight simulation integration is under development. Full UI coming soon!")
    
    st.markdown("""
    ### What's Available
    
    The flight simulation module (`flight_sim.py`) is ready and can:
    - Accept thrust curves from engine performance
    - Model tank geometries and propellant consumption
    - Simulate 6-DOF flight trajectories
    - Calculate apogee and max velocity
    - Generate flight plots (altitude, velocity, acceleration, etc.)
    
    ### How to Use (Python API)
    
    ```python
    from examples.pintle_engine.flight_sim import setup_flight
    from pintle_pipeline.io import load_config
    from pintle_models.runner import PintleEngineRunner
    
    # Load config and run engine
    config = load_config("config_minimal.yaml")
    runner = PintleEngineRunner(config)
    results = runner.evaluate(P_tank_O=1305*6894.76, P_tank_F=974*6894.76)
    
    # Create thrust curve
    burn_time = 5.0
    thrust_curve = [(0.0, results["F"]), (burn_time, results["F"])]
    
    # Run flight sim
    sim_result = setup_flight(
        config=config,
        thrust_curve=thrust_curve,
        mdot_lox=results["mdot_O"],
        mdot_fuel=results["mdot_F"]
    )
    
    print(f"Apogee: {sim_result['apogee']:.1f} m")
    print(f"Max velocity: {sim_result['max_velocity']:.1f} m/s")
    ```
    
    ### Configuration Requirements
    
    Flight simulation requires these additional config sections:
    - `environment`: Launch site (lat/lon/elevation), date
    - `rocket`: Mass, inertia, radius, fins
    - `lox_tank` / `fuel_tank`: Geometry (height, radius), position
    - `thrust`: Burn time
    
    See `examples/multi_body_rocket.yaml` for a complete example.
    """)
    
    # Simple test interface
    with st.expander("Quick Test (Constant Thrust)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            P_O_psi = st.number_input("LOX Tank Pressure [psi]", value=1305.0, min_value=100.0, max_value=3000.0)
            burn_time = st.number_input("Burn Time [s]", value=5.0, min_value=0.5, max_value=30.0)
        with col2:
            P_F_psi = st.number_input("Fuel Tank Pressure [psi]", value=974.0, min_value=100.0, max_value=3000.0)
            m_prop = st.number_input("Total Propellant Mass [kg]", value=24.0, min_value=1.0, max_value=100.0)
        
        if st.button("Run Quick Flight Sim", type="primary"):
            try:
                # Evaluate engine
                results = runner.evaluate(P_O_psi * PSI_TO_PA, P_F_psi * PSI_TO_PA)
                F = results["F"]
                mdot_O = results["mdot_O"]
                mdot_F = results["mdot_F"]
                
                st.success(f"Engine: {F/1000:.2f} kN thrust, {mdot_O+mdot_F:.3f} kg/s total flow")
                
                # Check if config has required fields
                if not hasattr(config_obj, "environment") or config_obj.environment is None:
                    st.error("Config missing 'environment' section. Cannot run flight sim.")
                    return
                if not hasattr(config_obj, "rocket") or config_obj.rocket is None:
                    st.error("Config missing 'rocket' section. Cannot run flight sim.")
                    return
                
                # Create thrust curve
                thrust_curve = [(0.0, F), (burn_time, F)]
                
                # Run flight sim
                sim_result = setup_flight(config_obj, thrust_curve, mdot_O, mdot_F, plot_results=False)
                
                # Display results
                col_a, col_b = st.columns(2)
                col_a.metric("Apogee", f"{sim_result['apogee']:.1f} m")
                if sim_result['max_velocity'] is not None:
                    col_b.metric("Max Velocity", f"{sim_result['max_velocity']:.1f} m/s")
                
                # Extract and plot
                flight = sim_result['flight']
                t_series, z_series, vz_series = extract_flight_series(flight)
                plot_flight_results(t_series, z_series, vz_series)
                
                with st.expander("Rocket View"):
                    render_rocket_view(flight)
                    
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

    config_obj = config_editor(config_obj)
    st.session_state["config_dict"] = config_obj.model_dump()
    st.session_state["config_label"] = config_label

    runner = PintleEngineRunner(config_obj)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Forward Mode",
        "Inverse Mode",
        "Time-Series Analysis",
        "Plots & Analysis",
        "Custom Plot Builder",
        "Flight Simulation",
    ])
    with tab1:
        forward_view(runner)
    with tab2:
        inverse_view(runner, config_label)
    with tab3:
        timeseries_view(runner, config_label)
    with tab4:
        plots_analysis_view(runner)
    with tab5:
        custom_plot_builder()
    with tab6:
        flight_sim_view(runner, config_obj, config_label)


if __name__ == "__main__":
    main()
