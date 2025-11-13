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
from rocketpy import Function

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pintle_pipeline.io import load_config
from pintle_pipeline.config_schemas import PintleEngineConfig
from pintle_pipeline.time_series import generate_pressure_profile
from pintle_models.runner import PintleEngineRunner
from examples.pintle_engine.interactive_pipeline import solve_for_thrust, ThrustSolveError
from examples.pintle_engine.flight_sim import setup_flight

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


def build_rp_function(times_s: np.ndarray, values: np.ndarray, interpolation: str = "linear") -> Function:
    """Build RocketPy Function from time/value arrays with sorting and stacking."""
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
        ("az", "Acceleration (m/s²)"),
        ("M", "Mach"),
        ("q", "Dynamic Pressure (Pa)"),
        ("totalMass", "Total Mass (kg)"),
        ("mass", "Total Mass (kg)"),
    ]
    for attr, label in candidates:
        series_obj = getattr(flight, attr, None)
        if series_obj is None:
            continue
        try:
            y = _to_1d(_series_to_np(series_obj))
            n = int(min(len(t_series), len(y)))
            if n <= 1:
                continue
            df = pd.DataFrame({"time": t_series[:n], label: y[:n]})
            st.plotly_chart(
                px.line(df, x="time", y=label, title=f"{label} vs Time"),
                use_container_width=True,
                key=f"flight_extra_{attr}{key_suffix}",
            )
        except Exception:
            continue


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
            df, errors = compute_timeseries_dataframe(
                runner,
                df_input["time"].to_numpy(dtype=float),
                df_input["P_tank_O"].to_numpy(dtype=float),
                df_input["P_tank_F"].to_numpy(dtype=float),
            )
            if errors:
                st.error("Errors encountered during time-series evaluation:")
                for err in errors:
                    st.write(f"- {err}")
                return

            st.session_state["timeseries_results"] = {"data": df, "meta": {"source": "csv", "config": config_label}}
            store_dataset("Time Series (uploaded)", df)

            display_time_series_summary(df)
            plot_time_series_results(df)

            if errors:
                st.warning("Some time steps did not converge. Affected rows contain NaNs in the output dataset.")

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
            
            # Detect underfill by integrating mdot arrays
            def detect_underfill_from_arrays(times, mdot_array, m_initial):
                """Detect when tank would be depleted by integrating mdot array."""
                if len(times) < 2:
                    return None
                # Use trapezoidal integration
                cumulative_mass = np.zeros_like(times)
                for i in range(1, len(times)):
                    dt = times[i] - times[i-1]
                    cumulative_mass[i] = cumulative_mass[i-1] + (mdot_array[i-1] + mdot_array[i]) * dt / 2.0
                # Find where cumulative mass exceeds initial mass
                depletion_idx = np.where(cumulative_mass >= m_initial)[0]
                if len(depletion_idx) > 0:
                    return float(times[depletion_idx[0]])
                return None
            
            lox_cutoff = detect_underfill_from_arrays(t_vals_normalized, mdot_O_vals, m_lox0)
            fuel_cutoff = detect_underfill_from_arrays(t_vals_normalized, mdot_F_vals, m_fuel0)
            
            # Find earliest cutoff time
            cutoff_time = None
            cutoff_reason = None
            if lox_cutoff is not None and fuel_cutoff is not None:
                if lox_cutoff <= fuel_cutoff:
                    cutoff_time = lox_cutoff
                    cutoff_reason = "LOX"
                else:
                    cutoff_time = fuel_cutoff
                    cutoff_reason = "fuel"
            elif lox_cutoff is not None:
                cutoff_time = lox_cutoff
                cutoff_reason = "LOX"
            elif fuel_cutoff is not None:
                cutoff_time = fuel_cutoff
                cutoff_reason = "fuel"
            
            # Truncate arrays at cutoff time
            if cutoff_time is not None:
                # Find indices where time <= cutoff
                valid_mask = t_vals_normalized <= cutoff_time
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    # Check if the last valid point is exactly at cutoff_time
                    last_valid_idx = valid_indices[-1]
                    last_valid_time = t_vals_normalized[last_valid_idx]
                    
                    if abs(last_valid_time - cutoff_time) < 1e-6:
                        # Last point is already at cutoff, just set last values to 0
                        t_vals_trunc = t_vals_normalized[:last_valid_idx+1].copy()
                        thrust_vals_trunc = thrust_vals_SI[:last_valid_idx+1].copy()
                        mdot_O_vals_trunc = mdot_O_vals[:last_valid_idx+1].copy()
                        mdot_F_vals_trunc = mdot_F_vals[:last_valid_idx+1].copy()
                        # Set last point to 0
                        thrust_vals_trunc[-1] = 0.0
                        mdot_O_vals_trunc[-1] = 0.0
                        mdot_F_vals_trunc[-1] = 0.0
                    else:
                        # Need to add a point at cutoff_time with 0 values
                        t_vals_trunc = np.concatenate([t_vals_normalized[:last_valid_idx+1], [cutoff_time]])
                        thrust_vals_trunc = np.concatenate([thrust_vals_SI[:last_valid_idx+1], [0.0]])
                        mdot_O_vals_trunc = np.concatenate([mdot_O_vals[:last_valid_idx+1], [0.0]])
                        mdot_F_vals_trunc = np.concatenate([mdot_F_vals[:last_valid_idx+1], [0.0]])
                    
                    # Update arrays
                    t_vals_normalized = t_vals_trunc
                    thrust_vals_SI = thrust_vals_trunc
                    mdot_O_vals = mdot_O_vals_trunc
                    mdot_F_vals = mdot_F_vals_trunc
                    
                    # Show warning
                    st.warning(f"{cutoff_reason.capitalize()} tank underfill detected at t={cutoff_time:.3f} s. Dataset truncated.")
                    
                    # Update burn time
                    burn_time_ds = float(cutoff_time)
            
            # Restore original time offset for display
            t_vals = t_vals_normalized + t_min
            
            # Always add truncated thrust column to original dataset
            # (If no truncation occurred, it will just match the original thrust)
            cutoff_time_original = (cutoff_time + t_min) if cutoff_time is not None else None
            
            # Create a function to interpolate truncated thrust at any time
            def get_truncated_thrust(t_query):
                """Get truncated thrust value at time t_query using interpolation."""
                if cutoff_time_original is not None and t_query > cutoff_time_original:
                    return 0.0
                # Find the index in truncated data (or original if no truncation)
                idx = np.searchsorted(t_vals, t_query, side='right') - 1
                if idx < 0:
                    return 0.0
                if idx >= len(thrust_vals_SI) - 1:
                    return thrust_vals_SI[-1] if len(thrust_vals_SI) > 0 else 0.0
                # Linear interpolation
                t1, t2 = t_vals[idx], t_vals[idx+1]
                f1, f2 = thrust_vals_SI[idx], thrust_vals_SI[idx+1]
                if abs(t2 - t1) > 1e-9:
                    return f1 + (f2 - f1) * (t_query - t1) / (t2 - t1)
                return f1
            
            # Add truncated thrust column to original dataset
            ds_df_with_truncated = ds_df.copy()
            # Apply truncated thrust function to each row's time value
            ds_df_with_truncated["Thrust_truncated (N)"] = ds_df[time_col].apply(get_truncated_thrust)
            
            # Update the dataset in session state
            datasets[dataset_name] = ds_df_with_truncated
            st.session_state["custom_plot_datasets"] = datasets
            if cutoff_time is not None:
                st.info(f"Added 'Thrust_truncated (N)' column to dataset '{dataset_name}' (truncated at t={cutoff_time:.3f} s).")
            else:
                st.info(f"Added 'Thrust_truncated (N)' column to dataset '{dataset_name}' (no truncation needed).")
            
            working["thrust"]["burn_time"] = burn_time_ds
            # Use user-defined propellant masses (set earlier in the form)
            try:
                config_for_flight = PintleEngineConfig(**working)
            except Exception as exc:
                st.error(f"Edited flight configuration is invalid: {exc}")
                return

            # Build RocketPy Functions from (potentially truncated) data
            thrust_func = build_rp_function(t_vals, thrust_vals_SI)
            mdot_O_func = build_rp_function(t_vals, mdot_O_vals)
            mdot_F_func = build_rp_function(t_vals, mdot_F_vals)

            # Run flight directly with dataset-driven inputs
            # (Dataset is already truncated above if needed)
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
                
                # Add flight simulation data to dataset
                if len(t_series) > 0 and len(z_series) > 0 and len(vz_series) > 0:
                    # Get the current dataset (should already have truncated thrust column)
                    ds_df_current = datasets.get(dataset_name)
                    if ds_df_current is None:
                        # Fallback to original if somehow not in session state
                        ds_df_current = ds_df_with_truncated if 'ds_df_with_truncated' in locals() else ds_df
                    
                    # Get launch elevation for cumulative altitude (altitude above sea level)
                    launch_elevation = float(config_for_flight.environment.elevation) if config_for_flight.environment else 0.0
                    
                    # Create interpolation functions for flight data
                    # Use numpy interpolation for better performance and accuracy
                    from scipy.interpolate import interp1d
                    
                    # Ensure time series are sorted and unique
                    t_series_sorted, sort_idx = np.unique(t_series, return_index=True)
                    z_series_sorted = z_series[sort_idx]
                    vz_series_sorted = vz_series[sort_idx]
                    
                    # Create interpolation functions (extrapolate to constant values outside range)
                    if len(t_series_sorted) > 1:
                        # Use linear interpolation, extrapolate to boundary values
                        interp_z = interp1d(t_series_sorted, z_series_sorted, kind='linear', 
                                           bounds_error=False, fill_value=(z_series_sorted[0], z_series_sorted[-1]))
                        interp_vz = interp1d(t_series_sorted, vz_series_sorted, kind='linear',
                                            bounds_error=False, fill_value=(0.0, 0.0))
                    else:
                        # Fallback if insufficient data
                        interp_z = lambda t: z_series_sorted[0] if len(z_series_sorted) > 0 else 0.0
                        interp_vz = lambda t: 0.0
                    
                    def get_altitude(t_query):
                        """Get cumulative altitude (height above sea level) at time t_query using interpolation."""
                        if len(t_series_sorted) == 0:
                            return launch_elevation
                        # Get interpolated altitude above launch
                        z_above_launch = float(interp_z(t_query))
                        # Return cumulative: launch elevation + altitude above launch
                        return launch_elevation + max(0.0, z_above_launch)
                    
                    def get_vertical_velocity(t_query):
                        """Get vertical velocity at time t_query using interpolation."""
                        if len(t_series_sorted) == 0:
                            return 0.0
                        # Get interpolated velocity (already handles bounds)
                        return float(interp_vz(t_query))
                    
                    # Add flight data columns to dataset
                    # Ensure dataset is sorted by time to avoid interpolation issues
                    ds_df_flight = ds_df_current.copy()
                    ds_df_flight = ds_df_flight.sort_values(by=time_col).reset_index(drop=True)
                    
                    # Apply interpolation to sorted dataset
                    ds_df_flight["Altitude above sea level (m)"] = ds_df_flight[time_col].apply(get_altitude)
                    ds_df_flight["Vertical Velocity (m/s)"] = ds_df_flight[time_col].apply(get_vertical_velocity)
                    
                    # Add scalar metrics as constant columns (for reference)
                    if isinstance(apogee, (int, float)):
                        ds_df_flight["Apogee (m)"] = apogee
                    if isinstance(max_v, (int, float)) and max_v is not None:
                        ds_df_flight["Max Velocity (m/s)"] = max_v
                    
                    # Update the dataset in session state
                    datasets[dataset_name] = ds_df_flight
                    st.session_state["custom_plot_datasets"] = datasets
                    st.info(f"Added flight simulation data (altitude, velocity) to dataset '{dataset_name}'.")
            except Exception as exc:
                st.warning(f"Could not extract time series: {exc}")
                return
            with st.expander("Rocket view (render)"):
                render_rocket_view(flight)
            with st.expander("Additional rocket plots"):
                plot_additional_rocket_plots(flight, t_series, key_suffix="_ds")
            with st.expander("Thrust curve"):
                thrust_df = pd.DataFrame({"time": t_vals, "Thrust (N)": thrust_vals_SI})
                st.plotly_chart(px.line(thrust_df, x="time", y="Thrust (N)", title="Thrust Curve (dataset)"), use_container_width=True, key="flight_thrust_plot_ds")
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

        # Display truncation info if tank underfill was detected
        truncation_info = sim_result.get("truncation_info")
        if truncation_info and truncation_info.get("truncated"):
            st.warning(truncation_info.get("message", "Tank underfill detected and truncated."))

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
            st.plotly_chart(thrust_fig, use_container_width=True, key="flight_thrust_plot")
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

        st.markdown("### Chamber & Nozzle")
        chamber = working_copy["chamber"]
        nozzle = working_copy["nozzle"]
        chamber["volume"] = st.number_input("Chamber volume [m³]", min_value=1e-6, max_value=1.0, value=float(chamber.get("volume", 1e-3)), format="%.6f")
        chamber["A_throat"] = st.number_input("Throat area [m²]", min_value=1e-5, max_value=0.01, value=float(chamber["A_throat"]), format="%.6f")
        chamber["length"] = length_number_input(
            "Chamber length",
            float(chamber.get("length", 0.5)),
            min_m=0.01,
            max_m=3.0,
            step_m=0.01,
            key="chamber_length",
        )
        chamber["Lstar"] = length_number_input(
            "Characteristic length L*",
            float(chamber["Lstar"]),
            min_m=0.1,
            max_m=5.0,
            step_m=0.05,
            key="chamber_lstar",
        )
        nozzle["expansion_ratio"] = st.number_input("Expansion ratio (Ae/At)", min_value=1.0, max_value=200.0, value=float(nozzle["expansion_ratio"]), format="%.4f")

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
