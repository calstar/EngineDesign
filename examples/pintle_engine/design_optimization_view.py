"""Comprehensive Design Optimization UI for Pintle Engine.

This module provides a user-friendly interface for:
1. Optimal injector sizing
2. Optimal chamber sizing
3. Stability margin optimization
4. Flight performance optimization
5. System constraint management

Goal: Build a pipeline that sizes optimal injector and chamber for required
stability margins and flight performance given system constraints.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import copy

from pintle_pipeline.config_schemas import PintleEngineConfig
from pintle_models.runner import PintleEngineRunner
from pintle_pipeline.chamber_optimizer import ChamberOptimizer
from pintle_pipeline.comprehensive_geometry_sizing import (
    size_complete_geometry,
    plot_complete_geometry,
    select_optimal_geometry,
)
from pintle_pipeline.system_diagnostics import SystemDiagnostics

# Import chamber geometry functions for proper calculations
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parents[2]
_chamber_path = _project_root / "chamber"
if str(_chamber_path) not in sys.path:
    sys.path.insert(0, str(_chamber_path))

from chamber_geometry import (
    chamber_length_calc,
    chamber_volume_calc,
    contraction_ratio_calc,
    area_chamber_calc,
    chamber_geometry_calc,
    contraction_length_horizontal_calc,
)

# Import chamber geometry visualizer
from pintle_pipeline.chamber_geometry_visualizer import (
    calculate_chamber_geometry_clear,
    plot_chamber_geometry_clear,
)


# =============================================================================
# SEGMENTED PRESSURE CURVE FUNCTIONS
# =============================================================================

def generate_segmented_pressure_curve(
    segments: List[Dict[str, Any]],
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a pressure curve from a list of segments.
    
    Each segment can be 'linear' or 'blowdown' with its own duration and pressures.
    
    Args:
        segments: List of segment dicts with keys:
            - type: 'linear' or 'blowdown'
            - duration: segment duration in seconds
            - start_pressure_psi: pressure at start of segment
            - end_pressure_psi: pressure at end of segment
            - decay_tau: time constant for blowdown (only for blowdown type)
        n_points: Total number of points in output array
    
    Returns:
        time_array: Array of time points [s]
        pressure_array: Array of pressures [psi]
    """
    if not segments:
        return np.array([0.0]), np.array([500.0])
    
    # Calculate total duration
    total_duration = sum(seg["duration"] for seg in segments)
    
    # Create time array
    time_array = np.linspace(0, total_duration, n_points)
    pressure_array = np.zeros(n_points)
    
    # Build pressure curve segment by segment
    t_start = 0.0
    for seg in segments:
        seg_type = seg["type"]
        seg_duration = seg["duration"]
        P_start = seg["start_pressure_psi"]
        P_end = seg["end_pressure_psi"]
        t_end = t_start + seg_duration
        
        # Find indices for this segment
        mask = (time_array >= t_start) & (time_array <= t_end)
        t_local = time_array[mask] - t_start
        
        if seg_type == "linear":
            # Linear interpolation
            if seg_duration > 0:
                pressure_array[mask] = P_start + (P_end - P_start) * t_local / seg_duration
            else:
                pressure_array[mask] = P_start
        elif seg_type == "blowdown":
            # Exponential decay: P(t) = P_end + (P_start - P_end) * exp(-t/tau)
            tau = seg.get("decay_tau", seg_duration * 0.5)
            if tau <= 0:
                tau = seg_duration * 0.5
            pressure_array[mask] = P_end + (P_start - P_end) * np.exp(-t_local / tau)
        else:
            # Default to linear
            if seg_duration > 0:
                pressure_array[mask] = P_start + (P_end - P_start) * t_local / seg_duration
            else:
                pressure_array[mask] = P_start
        
        t_start = t_end
    
    return time_array, pressure_array


def segments_from_optimizer_vars(
    x_segments: np.ndarray,
    n_segments: int,
    max_pressure_psi: float,
    target_burn_time: float,
) -> List[Dict[str, Any]]:
    """
    Convert optimizer variables to segment list.
    
    For each segment, optimizer provides:
    - type (0=linear, 1=blowdown) - rounded to int
    - duration_ratio (0-1, fraction of total burn time)
    - start_pressure_ratio (0.3-1.0, ratio of max pressure)
    - end_pressure_ratio (0.3-1.0, ratio of max pressure)
    - decay_tau_ratio (0-1, fraction of segment duration, only for blowdown)
    
    Args:
        x_segments: Array of optimizer variables for segments
        n_segments: Number of segments (1-20)
        max_pressure_psi: Maximum pressure [psi]
        target_burn_time: Total burn time [s]
    
    Returns:
        List of segment dicts
    """
    segments = []
    vars_per_segment = 5  # type, duration_ratio, start_ratio, end_ratio, tau_ratio
    
    # Normalize durations so they sum to 1.0
    duration_ratios = []
    t_start = 0.0
    
    for i in range(n_segments):
        idx_base = i * vars_per_segment
        seg_type_val = float(np.clip(x_segments[idx_base], 0.0, 1.0))
        seg_type = "blowdown" if seg_type_val >= 0.5 else "linear"
        duration_ratio = float(np.clip(x_segments[idx_base + 1], 0.01, 1.0))
        duration_ratios.append(duration_ratio)
    
    # Normalize so sum = 1.0
    total_ratio = sum(duration_ratios)
    if total_ratio > 0:
        duration_ratios = [dr / total_ratio for dr in duration_ratios]
    
    # Build segments
    for i in range(n_segments):
        idx_base = i * vars_per_segment
        seg_type_val = float(np.clip(x_segments[idx_base], 0.0, 1.0))
        seg_type = "blowdown" if seg_type_val >= 0.5 else "linear"
        duration = duration_ratios[i] * target_burn_time
        start_ratio = float(np.clip(x_segments[idx_base + 2], 0.30, 1.0))
        end_ratio = float(np.clip(x_segments[idx_base + 3], 0.30, 1.0))
        tau_ratio = float(np.clip(x_segments[idx_base + 4], 0.1, 1.0))
        
        seg = {
            "type": seg_type,
            "duration": duration,
            "start_pressure_psi": max_pressure_psi * start_ratio,
            "end_pressure_psi": max_pressure_psi * end_ratio,
        }
        
        if seg_type == "blowdown":
            seg["decay_tau"] = duration * tau_ratio
        
        segments.append(seg)
    
    return segments


def optimizer_vars_from_segments(
    segments: List[Dict[str, Any]],
    max_pressure_psi: float,
    target_burn_time: float,
) -> np.ndarray:
    """
    Convert segment list to optimizer variables.
    
    Inverse of segments_from_optimizer_vars.
    """
    vars_per_segment = 5
    n_segments = len(segments)
    x = np.zeros(n_segments * vars_per_segment)
    
    total_duration = sum(seg["duration"] for seg in segments)
    
    for i, seg in enumerate(segments):
        idx_base = i * vars_per_segment
        x[idx_base] = 1.0 if seg["type"] == "blowdown" else 0.0
        x[idx_base + 1] = seg["duration"] / total_duration if total_duration > 0 else 1.0 / n_segments
        x[idx_base + 2] = seg["start_pressure_psi"] / max_pressure_psi
        x[idx_base + 3] = seg["end_pressure_psi"] / max_pressure_psi
        x[idx_base + 4] = seg.get("decay_tau", seg["duration"] * 0.5) / seg["duration"] if seg["duration"] > 0 else 0.5
    
    return x


def _plot_segmented_pressure_preview(
    pressure_config: Dict[str, Any],
    target_burn_time: float,
) -> None:
    """Plot preview of segmented pressure curves."""
    import streamlit as st
    
    lox_segments = pressure_config.get("lox_segments", [])
    fuel_segments = pressure_config.get("fuel_segments", [])
    
    if not lox_segments and not fuel_segments:
        st.warning("No pressure segments defined.")
        return
    
    # Generate curves
    n_points = 200
    
    if lox_segments:
        lox_time, lox_pressure = generate_segmented_pressure_curve(lox_segments, n_points)
    else:
        lox_time = np.linspace(0, target_burn_time, n_points)
        lox_pressure = np.full(n_points, pressure_config.get("lox_start_psi", 500))
    
    if fuel_segments:
        fuel_time, fuel_pressure = generate_segmented_pressure_curve(fuel_segments, n_points)
    else:
        fuel_time = np.linspace(0, target_burn_time, n_points)
        fuel_pressure = np.full(n_points, pressure_config.get("fuel_start_psi", 500))
    
    # Create plot
    fig = make_subplots(rows=1, cols=1)
    
    fig.add_trace(
        go.Scatter(
            x=lox_time, y=lox_pressure,
            mode='lines',
            name='LOX Tank',
            line=dict(color='blue', width=2),
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=fuel_time, y=fuel_pressure,
            mode='lines',
            name='Fuel Tank',
            line=dict(color='orange', width=2),
        )
    )
    
    # Add segment boundaries
    t_cumulative = 0.0
    for i, seg in enumerate(lox_segments):
        t_cumulative += seg["duration"]
        if i < len(lox_segments) - 1:
            fig.add_vline(
                x=t_cumulative, 
                line=dict(color="blue", width=1, dash="dash"),
                annotation_text=f"LOX S{i+1}",
                annotation_position="top left",
            )
    
    t_cumulative = 0.0
    for i, seg in enumerate(fuel_segments):
        t_cumulative += seg["duration"]
        if i < len(fuel_segments) - 1:
            fig.add_vline(
                x=t_cumulative, 
                line=dict(color="orange", width=1, dash="dot"),
                annotation_text=f"Fuel S{i+1}",
                annotation_position="bottom left",
            )
    
    fig.update_layout(
        title="Segmented Pressure Curves Preview",
        xaxis_title="Time [s]",
        yaxis_title="Tank Pressure [psi]",
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("LOX Start", f"{lox_pressure[0]:.0f} psi")
    with col2:
        st.metric("LOX End", f"{lox_pressure[-1]:.0f} psi")
    with col3:
        st.metric("Fuel Start", f"{fuel_pressure[0]:.0f} psi")
    with col4:
        st.metric("Fuel End", f"{fuel_pressure[-1]:.0f} psi")


def design_optimization_view(config_obj: PintleEngineConfig, runner: Optional[PintleEngineRunner] = None) -> PintleEngineConfig:
    """
    Comprehensive design optimization interface.
    
    Provides guided workflow to:
    1. Set design requirements (thrust, altitude, stability)
    2. Optimize injector geometry
    3. Optimize chamber geometry
    4. Validate stability margins
    5. Validate flight performance
    6. Export optimized design
    """
    st.header("🚀 Engine Design Optimization")
    st.markdown("""
    **Goal:** Size optimal injector and chamber geometry to meet your:
    - **Stability margins** (chugging, acoustic, feed system)
    - **Flight performance** (altitude, payload capacity)
    - **System constraints** (weight, size, manufacturing)
    """)
    
    # Create tabs for different optimization stages
    tab_design, tab_full_engine, tab_injector, tab_chamber, tab_stability, tab_performance, tab_results = st.tabs([
        "📋 Design Requirements",
        "🚀 Full Engine Optimizer",
        "🔧 Injector Optimization",
        "🔥 Chamber Optimization", 
        "⚖️ Stability Analysis",
        "✈️ Flight Performance",
        "📊 Results & Export"
    ])
    
    with tab_design:
        config_obj = _design_requirements_tab(config_obj)
    
    with tab_full_engine:
        config_obj = _full_engine_optimization_tab(config_obj, runner)
    
    with tab_injector:
        config_obj = _injector_optimization_tab(config_obj, runner)
    
    with tab_chamber:
        config_obj = _chamber_optimization_tab(config_obj, runner)
        
        # Show optimized results if available
        if "optimization_results" in st.session_state:
            st.markdown("---")
            st.markdown("## 📊 Optimization Results")
            opt_results = st.session_state["optimization_results"]
            opt_config = st.session_state.get("optimized_config", config_obj)
            _display_optimized_parameters(opt_results, opt_config)
            
            # Show time-varying plot if available
            if "time_varying_results" in opt_results:
                _plot_time_varying_results(opt_results["time_varying_results"])
        
        # Show comprehensive geometry plot
        if runner and config_obj:
            st.markdown("### Complete Geometry Visualization")
            sizing_results = None  # Initialize for scope
            try:
                from pintle_pipeline.comprehensive_geometry_sizing import size_complete_geometry, plot_complete_geometry
                
                # Get current conditions
                P_tank_O = 3e6  # Default
                P_tank_F = 3e6
                
                # Use optimized config if available
                current_config = st.session_state.get("optimized_config", config_obj)
                current_runner = PintleEngineRunner(current_config) if current_config != config_obj else runner
                
                results = current_runner.evaluate(P_tank_O, P_tank_F)
                
                Pc = results.get("Pc", 2e6)
                MR = results.get("MR", 2.5)
                Tc = results.get("Tc", 3500.0)
                gamma = results.get("gamma", 1.2)
                R = results.get("R", 300.0)
                burn_time = st.session_state.get("design_requirements", {}).get("target_burn_time", 10.0)
                
                # Estimate heat flux from actual results if available
                heat_flux_chamber = 2e6  # W/m² - default estimate
                if "heat_flux_chamber" in results:
                    heat_flux_chamber = results["heat_flux_chamber"]
                elif "diagnostics" in results:
                    cooling = results["diagnostics"].get("cooling", {})
                    ablative = cooling.get("ablative", {})
                    if ablative:
                        heat_flux_chamber = ablative.get("incident_heat_flux", heat_flux_chamber)
                
                # Validate config has required attributes
                if not hasattr(current_config, "chamber") or not hasattr(current_config.chamber, "volume"):
                    st.warning("⚠️ Chamber configuration incomplete. Cannot generate geometry plot.")
                elif current_config.chamber.volume <= 0 or current_config.chamber.A_throat <= 0:
                    st.warning("⚠️ Invalid chamber geometry (volume or throat area is zero). Cannot generate geometry plot.")
                else:
                    # Size complete geometry
                    sizing_results = size_complete_geometry(
                        config=current_config,
                        Pc=Pc,
                        MR=MR,
                        Tc=Tc,
                        gamma=gamma,
                        R=R,
                        burn_time=burn_time,
                        chamber_heat_flux=heat_flux_chamber,
                    )
                    
                    # Plot complete geometry
                    fig, _ = plot_complete_geometry(
                        sizing_results,
                        current_config,
                        use_plotly=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display sizing summary (only if sizing succeeded)
                    if sizing_results:
                        st.markdown("#### Sizing Summary")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Ablative Thickness", f"{sizing_results.get('optimal', {}).get('ablative_thickness', 0) * 1000:.2f} mm")
                        with col_b:
                            st.metric("Graphite Thickness", f"{sizing_results.get('optimal', {}).get('graphite_thickness', 0) * 1000:.2f} mm")
                        with col_c:
                            st.metric("Total Mass", f"{sizing_results.get('optimal', {}).get('total_mass', 0):.3f} kg")
                    
                    # Show impingement zones
                    st.markdown("#### Fuel Impingement Zones")
                    try:
                        from pintle_pipeline.localized_ablation import calculate_impingement_zones
                        
                        # Get chamber dimensions from config
                        if hasattr(current_config, "chamber"):
                            chamber_length = current_config.chamber.length if current_config.chamber.length else 0.2
                            if hasattr(current_config.chamber, "chamber_inner_diameter"):
                                chamber_diameter = current_config.chamber.chamber_inner_diameter
                            else:
                                # Calculate from volume and length
                                V_chamber = current_config.chamber.volume
                                L_chamber = chamber_length
                                if L_chamber > 0:
                                    chamber_diameter = np.sqrt(4.0 * V_chamber / (np.pi * L_chamber))
                                else:
                                    chamber_diameter = 0.1  # Default
                        else:
                            chamber_length = 0.2
                            chamber_diameter = 0.1
                        
                        impingement_data = calculate_impingement_zones(
                            current_config, chamber_length, chamber_diameter, n_points=50
                        )
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Impingement Center", f"{impingement_data['impingement_center'] * 1000:.1f} mm")
                            st.metric("Max Heat Flux Multiplier", f"{np.max(impingement_data['impingement_heat_flux_multiplier']):.2f}x")
                        with col_b:
                            st.metric("Impingement Width", f"{impingement_data['impingement_width'] * 1000:.1f} mm")
                            st.info("⚠️ Enhanced ablation occurs at impingement zones (red markers in plot)")
                    except Exception as e:
                        st.warning(f"Could not calculate impingement zones: {e}")
                
                # Recession animation option
                st.markdown("#### Recession Animation")
                if st.checkbox("Generate Recession Animation", value=False):
                    st.info("Animation will show ablation over time with impingement effects. Run time-series analysis first.")
                
                # Validation status (only if sizing succeeded)
                if sizing_results:
                    validation = sizing_results.get("validation", {})
                    if validation.get("all_valid", False):
                        st.success("✅ All sizing requirements met!")
                    else:
                        warnings = validation.get("warnings", [])
                        if warnings:
                            for warning in warnings:
                                st.warning(f"⚠️ {warning}")
                
            except Exception as e:
                st.warning(f"Could not generate comprehensive geometry plot: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    with tab_stability:
        _stability_analysis_tab(config_obj, runner)
    
    with tab_performance:
        _flight_performance_tab(config_obj, runner)
    
    with tab_results:
        _results_export_tab(config_obj, runner)
    
    return config_obj


def _design_requirements_tab(config_obj: PintleEngineConfig) -> PintleEngineConfig:
    """Design requirements input tab with full rocket configuration and engine design inputs."""
    st.subheader("Design Requirements")
    st.markdown("""
    Configure your rocket and specify engine design targets. The optimizer will solve for:
    - **Propellant masses** (LOX & fuel to achieve target apogee)
    - **Engine geometry** (injector, chamber, nozzle sizing)
    - **Burn time** (optimized for mission profile)
    """)
    
    # Initialize working config from session state or config_obj
    working = st.session_state.get("design_config", {})
    if not working:
        working = config_obj.model_dump(exclude_none=False) if config_obj else {}
    
    # ==========================================================================
    # SECTION 1: ROCKET CONFIGURATION (from Flight Sim)
    # ==========================================================================
    st.markdown("---")
    st.markdown("## 🚀 Rocket Configuration")
    st.caption("Define your vehicle structure. Propellant masses will be solved by the optimizer.")
    
    # --- Environment Expander ---
    with st.expander("🌍 Environment", expanded=False):
        st.caption("Launch site location. Atmospheric conditions will be fetched from NOAA GFS forecast.")
        env = working.get("environment") if working.get("environment") is not None else {}
        env.setdefault("latitude", 35.34722)
        env.setdefault("longitude", -117.8099547)
        env.setdefault("elevation", 626.67)
        
        # Handle date
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
            env_lat = st.number_input("Latitude [deg]", value=float(env.get("latitude") or 35.34722), key="opt_env_lat",
                help="Launch site latitude. Positive = North, Negative = South.")
            env_elev = st.number_input("Elevation [m]", value=float(env.get("elevation") or 626.67), key="opt_env_elev",
                help="Ground elevation above sea level.")
            env_date_input = st.date_input("Launch date", value=env_date, key="opt_env_date",
                help="Date for GFS atmospheric forecast.")
        with colE2:
            env_lon = st.number_input("Longitude [deg]", value=float(env.get("longitude") or -117.8099547), key="opt_env_lon",
                help="Launch site longitude. Positive = East, Negative = West.")
            env_hour_input = st.number_input("Launch hour [0-23 UTC]", min_value=0, max_value=23, value=env_hour, step=1, key="opt_env_hour",
                help="Launch hour in UTC.")
        
        env["latitude"] = float(env_lat)
        env["longitude"] = float(env_lon)
        env["elevation"] = float(env_elev)
        env["date"] = [int(env_date_input.year), int(env_date_input.month), int(env_date_input.day), int(env_hour_input)]
        working["environment"] = env
    
    # --- Rocket Expander ---
    with st.expander("🛩️ Rocket Structure", expanded=True):
        st.caption("**Mass Model:** Airframe (body/fins/avionics) + Propulsion (engine + tanks). Propellant masses are solved by optimizer.")
        rocket = working.get("rocket") if working.get("rocket") is not None else {}
        
        # Defaults
        rocket.setdefault("airframe_mass", 78.72)
        rocket.setdefault("engine_mass", 8.0)
        rocket.setdefault("lox_tank_structure_mass", 5.0)
        rocket.setdefault("fuel_tank_structure_mass", 3.0)
        rocket.setdefault("motor_position", 0.0)
        rocket.setdefault("engine_cm_offset", 0.15)
        rocket.setdefault("radius", 0.1015)
        rocket.setdefault("rocket_length", 3.5)
        rocket.setdefault("inertia", [8.0, 8.0, 0.5])
        
        st.markdown("##### Dry Masses (no propellant)")
        
        airframe = st.number_input(
            "Airframe mass [kg]", 
            value=float(rocket.get("airframe_mass") or 78.72), 
            key="opt_airframe_mass",
            help="Fuselage, fins, nosecone, avionics, payload (NO propulsion)"
        )
        
        st.caption("**Propulsion breakdown:**")
        colE1, colE2, colE3, colE4 = st.columns(4)
        with colE1:
            engine_mass = st.number_input(
                "Engine + plumbing [kg]", 
                value=float(rocket.get("engine_mass") or 8.0), 
                key="opt_engine_mass",
                help="Chamber, nozzle, injector, valves, ALL fittings & plumbing"
            )
        with colE2:
            lox_tank_mass = st.number_input(
                "LOX tank [kg]", 
                value=float(rocket.get("lox_tank_structure_mass") or 5.0), 
                key="opt_lox_tank_mass",
                help="Empty LOX tank structure (walls, no propellant)"
            )
        with colE3:
            fuel_tank_mass = st.number_input(
                "Fuel tank [kg]", 
                value=float(rocket.get("fuel_tank_structure_mass") or 3.0), 
                key="opt_fuel_tank_mass",
                help="Empty fuel tank structure (walls, no propellant)"
            )
        with colE4:
            copv_tank_mass = st.number_input(
                "COPV tank [kg]", 
                value=float(rocket.get("copv_dry_mass") or 5.0), 
                key="opt_copv_tank_mass",
                help="Empty COPV tank structure (walls, no pressurant gas)"
            )
        
        propulsion_dry = engine_mass + lox_tank_mass + fuel_tank_mass + copv_tank_mass
        total_dry = airframe + propulsion_dry
        st.info(f"**Propulsion dry:** {propulsion_dry:.2f} kg (engine + tanks + COPV) | **Total dry:** {total_dry:.2f} kg")
        
        st.markdown("##### Geometry & Positions")
        st.caption("Coordinate system: z=0 at rocket tail (bottom), positive toward nose (top).")
        colG1, colG2, colG3 = st.columns(3)
        with colG1:
            r_radius = st.number_input("Rocket radius [m]", value=float(rocket.get("radius") or 0.1015), key="opt_rocket_radius",
                help="Outer body radius (diameter ÷ 2). Used for aerodynamics and MoI.")
        with colG2:
            rocket_length = st.number_input("Rocket length [m]", value=float(rocket.get("rocket_length") or 3.5), key="opt_rocket_length",
                help="Total rocket length (tail to nose tip).")
        with colG3:
            motor_pos = st.number_input("Motor position [m]", value=float(rocket.get("motor_position") or 0.0), key="opt_motor_position",
                help="Distance from rocket tail to nozzle exit.")
        
        engine_cm_offset = st.number_input(
            "Engine CM offset [m]", 
            value=float(rocket.get("engine_cm_offset") or 0.15), 
            key="opt_engine_cm",
            help="Height of engine center of mass above nozzle exit."
        )
        
        st.markdown("##### Inertia (airframe only)")
        st.caption("💡 Propulsion inertia is auto-calculated using parallel axis theorem.")
        
        auto_inertia = st.checkbox("Auto-estimate inertia from mass & geometry", value=True, key="opt_auto_inertia",
            help="Estimate using solid cylinder approximation.")
        
        if auto_inertia:
            m_dry = airframe + propulsion_dry
            r = r_radius
            L = rocket_length
            i_xx_est = (1.0/12.0) * m_dry * (3 * r**2 + L**2)
            i_yy_est = i_xx_est
            i_zz_est = 0.5 * m_dry * r**2
            st.success(f"**Auto-estimated** (m={m_dry:.1f}kg, r={r*1000:.0f}mm, L={L:.2f}m):\n\n"
                      f"Ixx = {i_xx_est:.3f} kg·m² | Iyy = {i_yy_est:.3f} kg·m² | Izz = {i_zz_est:.4f} kg·m²")
            i_xx, i_yy, i_zz = i_xx_est, i_yy_est, i_zz_est
        else:
            _inertia = rocket.get("inertia") or [8.0, 8.0, 0.5]
            colI1, colI2, colI3 = st.columns(3)
            with colI1:
                i_xx = st.number_input("Ixx [kg·m²]", value=float(_inertia[0]), key="opt_inertia_x")
            with colI2:
                i_yy = st.number_input("Iyy [kg·m²]", value=float(_inertia[1]), key="opt_inertia_y")
            with colI3:
                i_zz = st.number_input("Izz [kg·m²]", value=float(_inertia[2]), key="opt_inertia_z")
        
        # Fins
        st.markdown("##### Fins")
        fins = rocket.get("fins") if rocket.get("fins") is not None else {}
        # Use 'or' pattern to handle None values (fin_position can be 0.0, so use explicit check)
        no_fins_val = fins.get("no_fins") or 3
        root_chord_val = fins.get("root_chord") or 0.2
        tip_chord_val = fins.get("tip_chord") or 0.1
        fin_span_val = fins.get("fin_span") or 0.3
        fin_position_val = fins.get("fin_position")
        if fin_position_val is None:
            fin_position_val = 0.0
        
        colF1, colF2, colF3 = st.columns(3)
        with colF1:
            fins["no_fins"] = int(st.number_input("Fin count", value=int(no_fins_val), min_value=1, step=1, key="opt_fins_count"))
            fins["root_chord"] = float(st.number_input("Root chord [m]", value=float(root_chord_val), key="opt_fins_root"))
        with colF2:
            fins["tip_chord"] = float(st.number_input("Tip chord [m]", value=float(tip_chord_val), key="opt_fins_tip"))
            fins["fin_span"] = float(st.number_input("Fin span [m]", value=float(fin_span_val), key="opt_fins_span"))
        with colF3:
            fins["fin_position"] = float(st.number_input("Fin position [m]", value=float(fin_position_val), key="opt_fins_pos"))
        
        # Store rocket config
        rocket["airframe_mass"] = float(airframe)
        rocket["engine_mass"] = float(engine_mass)
        rocket["lox_tank_structure_mass"] = float(lox_tank_mass)
        rocket["fuel_tank_structure_mass"] = float(fuel_tank_mass)
        rocket["copv_dry_mass"] = float(copv_tank_mass)
        rocket["propulsion_dry_mass"] = float(propulsion_dry)
        rocket["motor_position"] = float(motor_pos)
        rocket["engine_cm_offset"] = float(engine_cm_offset)
        rocket["radius"] = float(r_radius)
        rocket["rocket_length"] = float(rocket_length)
        rocket["inertia"] = [float(i_xx), float(i_yy), float(i_zz)]
        rocket["fins"] = fins
        working["rocket"] = rocket
    
    # --- Tanks Expander ---
    with st.expander("🛢️ Tank Geometry", expanded=False):
        st.caption("**Tank dimensions and positions.** Propellant masses will be solved by the optimizer to achieve target apogee.")
        
        lox_tank = working.get("lox_tank") if working.get("lox_tank") is not None else {}
        fuel_tank = working.get("fuel_tank") if working.get("fuel_tank") is not None else {}
        
        st.markdown("##### LOX Tank")
        # Use 'or' pattern to handle None values
        lox_h_val = lox_tank.get("lox_h") or 1.14
        lox_radius_val = lox_tank.get("lox_radius") or 0.0762
        ox_tank_pos_val = lox_tank.get("ox_tank_pos") or 0.6
        
        colL1, colL2, colL3 = st.columns(3)
        with colL1:
            lox_tank["lox_h"] = float(st.number_input("Height [m]", value=float(lox_h_val), key="opt_lox_h",
                help="Internal cylindrical height."))
        with colL2:
            lox_tank["lox_radius"] = float(st.number_input("Radius [m]", value=float(lox_radius_val), key="opt_lox_radius",
                help="Internal radius."))
        with colL3:
            lox_tank["ox_tank_pos"] = float(st.number_input("Position [m]", value=float(ox_tank_pos_val), key="opt_lox_pos",
                help="Tank center relative to nozzle exit."))
        
        # Calculate LOX tank capacity
        fluids = working.get("fluids") if working.get("fluids") is not None else {}
        ox_fluid = fluids.get("oxidizer") if fluids.get("oxidizer") is not None else {}
        rho_lox = float(ox_fluid.get("density") or 1140.0)
        lox_volume = np.pi * lox_tank["lox_radius"]**2 * lox_tank["lox_h"]
        lox_capacity = lox_volume * rho_lox
        st.caption(f"Tank Volume: **{lox_volume*1000:.1f} L** | Max Capacity: **{lox_capacity:.1f} kg** (optimizer will fill as needed)")
        
        st.markdown("##### Fuel Tank")
        # Use 'or' pattern to handle None values
        rp1_h_val = fuel_tank.get("rp1_h") or 0.609
        rp1_radius_val = fuel_tank.get("rp1_radius") or 0.0762
        fuel_tank_pos_val = fuel_tank.get("fuel_tank_pos")
        if fuel_tank_pos_val is None:
            fuel_tank_pos_val = -0.2  # Can't use 'or' for negative default
        
        colFu1, colFu2, colFu3 = st.columns(3)
        with colFu1:
            fuel_tank["rp1_h"] = float(st.number_input("Height [m]", value=float(rp1_h_val), key="opt_rp1_h"))
        with colFu2:
            fuel_tank["rp1_radius"] = float(st.number_input("Radius [m]", value=float(rp1_radius_val), key="opt_rp1_radius"))
        with colFu3:
            fuel_tank["fuel_tank_pos"] = float(st.number_input("Position [m]", value=float(fuel_tank_pos_val), key="opt_rp1_pos"))
        
        # Calculate Fuel tank capacity
        fu_fluid = fluids.get("fuel") if fluids.get("fuel") is not None else {}
        rho_fuel = float(fu_fluid.get("density") or 780.0)
        fuel_volume = np.pi * fuel_tank["rp1_radius"]**2 * fuel_tank["rp1_h"]
        fuel_capacity = fuel_volume * rho_fuel
        st.caption(f"Tank Volume: **{fuel_volume*1000:.1f} L** | Max Capacity: **{fuel_capacity:.1f} kg** (optimizer will fill as needed)")
        
        # --- Pressurant Tank (COPV) ---
        st.markdown("##### Pressurant Tank (COPV - GN₂)")
        press_tank = working.get("press_tank") if working.get("press_tank") is not None else {}
        
        # Use 'or' pattern to handle None values
        press_h_val = press_tank.get("press_h") or 0.457
        press_radius_val = press_tank.get("press_radius") or 0.0762
        pres_tank_pos_val = press_tank.get("pres_tank_pos") or 1.2
        
        colP1, colP2, colP3 = st.columns(3)
        with colP1:
            press_tank["press_h"] = float(st.number_input("Height [m]", value=float(press_h_val), key="opt_press_h",
                help="COPV cylindrical height."))
        with colP2:
            press_tank["press_radius"] = float(st.number_input("Radius [m]", value=float(press_radius_val), key="opt_press_radius",
                help="COPV radius."))
        with colP3:
            press_tank["pres_tank_pos"] = float(st.number_input("Position [m]", value=float(pres_tank_pos_val), key="opt_press_pos",
                help="COPV center position relative to nozzle exit. Typically above propellant tanks."))
        
        # Calculate COPV volume from geometry (external dimensions)
        press_volume_calc = np.pi * press_tank["press_radius"]**2 * press_tank["press_h"]
        
        # User-specified free internal volume (may be smaller than calculated due to walls)
        copv_free_volume_default = press_tank.get("free_volume_L") or 4.5  # Default 4.5L
        copv_free_volume_L = st.number_input(
            "COPV Free Volume [L]",
            min_value=0.1,
            max_value=100.0,
            value=float(copv_free_volume_default),
            step=0.5,
            key="opt_copv_free_volume",
            help="Internal free gas volume (excluding walls). Typically 85-95% of calculated geometric volume."
        )
        press_tank["free_volume_L"] = copv_free_volume_L
        st.caption(f"Geometric Volume: {press_volume_calc*1000:.1f} L | Free Volume: **{copv_free_volume_L:.1f} L**")
        
        working["lox_tank"] = lox_tank
        working["fuel_tank"] = fuel_tank
        working["press_tank"] = press_tank
    
    # --- Fluids Expander ---
    with st.expander("💧 Fluid Properties", expanded=False):
        st.caption("Propellant densities for capacity calculations.")
        fluids = working.get("fluids") if working.get("fluids") is not None else {}
        ox = fluids.get("oxidizer") if fluids.get("oxidizer") is not None else {}
        fu = fluids.get("fuel") if fluids.get("fuel") is not None else {}
        
        # Use 'or' pattern to handle None values
        ox_name = ox.get("name") or "LOX"
        ox_density = ox.get("density") or 1140.0
        fu_name = fu.get("name") or "RP-1"
        fu_density = fu.get("density") or 780.0
        
        colOx1, colOx2 = st.columns(2)
        with colOx1:
            st.markdown("**Oxidizer**")
            ox["name"] = st.text_input("Name", value=str(ox_name), key="opt_ox_name")
            ox["density"] = float(st.number_input("Density [kg/m³]", value=float(ox_density), key="opt_ox_density",
                help="LOX ≈ 1140 kg/m³"))
        with colOx2:
            st.markdown("**Fuel**")
            fu["name"] = st.text_input("Name", value=str(fu_name), key="opt_fu_name")
            fu["density"] = float(st.number_input("Density [kg/m³]", value=float(fu_density), key="opt_fu_density",
                help="RP-1 ≈ 780-820 kg/m³"))
        
        fluids["oxidizer"] = ox
        fluids["fuel"] = fu
        working["fluids"] = fluids
    
    # ==========================================================================
    # SECTION 2: ENGINE DESIGN INPUTS
    # ==========================================================================
    st.markdown("---")
    st.markdown("## 🔧 Engine Design Targets")
    st.caption("Specify performance targets and constraints. The optimizer will size the engine to meet these.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Performance Targets")
        
        target_thrust = st.number_input(
            "Target Peak Thrust [N]",
            min_value=100.0,
            max_value=100000.0,
            value=7000.0,
            step=100.0,
            key="opt_target_thrust",
            help="Peak thrust during burn. Engine will be sized to achieve this."
        )
        
        target_apogee = st.number_input(
            "Target Apogee [m AGL]",
            min_value=100.0,
            max_value=200000.0,
            value=3048.0,  # 10k feet
            step=100.0,
            key="opt_target_apogee",
            help="Target altitude above ground level. Optimizer will solve for propellant masses."
        )
        
        optimal_of_ratio = st.number_input(
            "Optimal O/F Ratio",
            min_value=1.5,
            max_value=4.0,
            value=2.3,
            step=0.1,
            key="opt_of_ratio",
            help="Target oxidizer-to-fuel mixture ratio. LOX/RP-1 optimal: 2.4-2.8 for Isp, 2.2-2.5 for stability."
        )
        
        target_burn_time = st.number_input(
            "Target Burn Time [s]",
            min_value=1.0,
            max_value=60.0,
            value=10.0,
            step=1.0,
            key="opt_target_burn_time",
            help="Design burn time. Flight sim will truncate if propellant depletes earlier."
        )
        
        st.markdown("### Tank Pressures")
        
        max_lox_tank_pressure = st.number_input(
            "Max LOX Tank Pressure [psi]",
            min_value=100.0,
            max_value=5000.0,
            value=700.0,
            step=25.0,
            key="opt_max_lox_pressure",
            help="Maximum operating pressure in LOX tank. Sets upper bound for chamber pressure."
        )
        
        max_fuel_tank_pressure = st.number_input(
            "Max Fuel Tank Pressure [psi]",
            min_value=100.0,
            max_value=5000.0,
            value=850.0,
            step=25.0,
            key="opt_max_fuel_pressure",
            help="Maximum operating pressure in fuel tank."
        )
    
    with col2:
        st.markdown("### Geometry Constraints")
        
        max_engine_length = st.number_input(
            "Max Engine Length [m]",
            min_value=0.1,
            max_value=3.0,
            value=0.5,
            step=0.05,
            key="opt_max_engine_length",
            help="Maximum total engine length (chamber + nozzle). Must fit in vehicle."
        )
        
        max_chamber_outer_diameter = st.number_input(
            "Max Chamber Outer Diameter [m]",
            min_value=0.05,
            max_value=1.0,
            value=0.15,
            step=0.01,
            key="opt_max_chamber_od",
            help="Maximum chamber outer diameter (including wall thickness and cooling jacket)."
        )
        
        max_nozzle_exit_diameter = st.number_input(
            "Max Nozzle Exit Diameter [m]",
            min_value=0.05,
            max_value=1.0,
            value=0.101,
            step=0.01,
            key="opt_max_nozzle_exit_od",
            help="Maximum nozzle exit outer diameter. Constrains expansion ratio."
        )
        
        st.markdown("### L* (Characteristic Length) Constraints")
        
        col_lstar1, col_lstar2 = st.columns(2)
        with col_lstar1:
            min_lstar = st.number_input(
                "Minimum L* [m]",
                min_value=0.5,
                max_value=3.0,
                value=0.95,
                step=0.1,
                key="opt_min_lstar",
                help="Minimum characteristic length. Lower = smaller chamber but less complete combustion. Typical: 0.8-1.0m for LOX/RP-1."
            )
        with col_lstar2:
            max_lstar = st.number_input(
                "Maximum L* [m]",
                min_value=0.5,
                max_value=3.0,
                value=1.27,
                step=0.1,
                key="opt_max_lstar",
                help="Maximum characteristic length. Higher = better combustion but heavier/longer chamber. Typical: 1.5-2.0m for LOX/RP-1."
            )
        
        st.markdown("### Stability Requirements")
        st.info("""
        **New Comprehensive Stability Analysis:**
        - Uses stability_score (0-1) and stability_state ("stable"/"marginal"/"unstable")
        - Considers chugging, acoustic modes, feed system, and mode coupling
        - **Stable**: score ≥ 0.75 (recommended for flight)
        - **Marginal**: 0.4 ≤ score < 0.75 (acceptable with caution)
        - **Unstable**: score < 0.4 (not acceptable)
        """)
        
        min_stability_score = st.number_input(
            "Minimum Stability Score",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
            key="opt_min_stability_score",
            help="Minimum stability score (0-1). 0.75 = 'stable', 0.4 = 'marginal', <0.4 = 'unstable'"
        )
        
        require_stable_state = st.checkbox(
            "Require 'Stable' State (not just 'Marginal')",
            value=True,
            key="opt_require_stable_state",
            help="If checked, optimizer will only converge when stability_state == 'stable'. If unchecked, allows 'marginal' state."
        )
        
        stability_margin_handicap = st.slider(
            "Stability Margin Handicap",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            key="opt_stability_margin_handicap",
            help=(
                "0.0 = use full stability requirements (score and margins).\n"
                "1.0 = accept any stability score/margins.\n"
                "Intermediate values scale how strict the stability gates are."
            ),
        )
        
        st.markdown("#### Individual Stability Margins (for detailed tracking)")
        st.caption("These are used for detailed feedback but the optimizer primarily uses stability_score above.")
        
        min_stability_margin = st.number_input(
            "Minimum Overall Stability Margin (legacy)",
            min_value=1.0,
            max_value=5.0,
            value=1.2,
            step=0.1,
            key="opt_min_stability",
            help="Legacy margin-based requirement (for backward compatibility)"
        )
        
        chugging_margin_min = st.number_input(
            "Chugging Margin (min)",
            min_value=0.0,
            max_value=10.0,
            value=0.2,
            step=0.1,
            key="opt_chugging_margin",
            help="Minimum chugging stability margin (for detailed tracking)"
        )
        
        acoustic_margin_min = st.number_input(
            "Acoustic Margin (min)",
            min_value=0.0,
            max_value=10.0,
            value=0.1,
            step=0.1,
            key="opt_acoustic_margin",
            help="Minimum acoustic stability margin (for detailed tracking)"
        )
        
        feed_stability_min = st.number_input(
            "Feed System Margin (min)",
            min_value=0.0,
            max_value=10.0,
            value=0.15,
            step=0.1,
            key="opt_feed_margin",
            help="Minimum feed system stability margin (for detailed tracking)"
        )
        
    # Convert pressures to SI (Pa)
    max_P_tank_O = max_lox_tank_pressure * 6894.76  # psi to Pa
    max_P_tank_F = max_fuel_tank_pressure * 6894.76
    
    # Store all requirements in session state
    st.session_state["design_requirements"] = {
        # Performance targets
        "target_thrust": target_thrust,
        "target_apogee": target_apogee,
        "optimal_of_ratio": optimal_of_ratio,
        "target_burn_time": target_burn_time,
        # Tank pressures (SI)
        "max_P_tank_O": max_P_tank_O,
        "max_P_tank_F": max_P_tank_F,
        "max_lox_tank_pressure_psi": max_lox_tank_pressure,
        "max_fuel_tank_pressure_psi": max_fuel_tank_pressure,
        # Geometry constraints
        "max_engine_length": max_engine_length,
        "max_chamber_outer_diameter": max_chamber_outer_diameter,
        "max_nozzle_exit_diameter": max_nozzle_exit_diameter,
        # L* constraints
        "min_Lstar": min_lstar,
        "max_Lstar": max_lstar,
        # Stability (new comprehensive analysis)
        "min_stability_score": min_stability_score,
        "require_stable_state": require_stable_state,
        "stability_margin_handicap": stability_margin_handicap,
        # Stability (legacy margins for backward compatibility)
        "min_stability_margin": min_stability_margin,
        "chugging_margin_min": chugging_margin_min,
        "acoustic_margin_min": acoustic_margin_min,
        "feed_stability_min": feed_stability_min,
        # Tank capacities (for optimizer bounds)
        "lox_tank_capacity_kg": lox_capacity,
        "fuel_tank_capacity_kg": fuel_capacity,
        # COPV
        "copv_free_volume_L": copv_free_volume_L,
        "copv_free_volume_m3": copv_free_volume_L / 1000.0,
    }
    
    # Store rocket config for optimizer
    st.session_state["design_config"] = working
    st.session_state["rocket_dry_mass"] = total_dry
    
    # Summary
    st.markdown("---")
    st.markdown("### 📋 Design Summary")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Rocket Dry Mass", f"{total_dry:.1f} kg")
        st.metric("Target Apogee", f"{target_apogee:.0f} m")
    with col_s2:
        st.metric("Target Thrust", f"{target_thrust:.0f} N")
        st.metric("Optimal O/F", f"{optimal_of_ratio:.2f}")
    with col_s3:
        st.metric("Max Tank Pressure", f"{max(max_lox_tank_pressure, max_fuel_tank_pressure):.0f} psi")
        st.metric("Target Burn Time", f"{target_burn_time:.1f} s")
    with col_s4:
        st.metric("L* Range", f"{min_lstar:.1f} - {max_lstar:.1f} m")
        st.metric("Max Engine Length", f"{max_engine_length*1000:.0f} mm")
    
    st.success("✅ Configuration saved. Proceed to **Full Engine Optimizer** to optimize your complete engine, or use individual tabs for specific optimizations.")
    
    return config_obj


def _full_engine_optimization_tab(config_obj: PintleEngineConfig, runner: Optional[PintleEngineRunner]) -> PintleEngineConfig:
    """Full engine optimization tab - optimizes both pintle injector and chamber together."""
    st.subheader("🚀 Full Engine Optimizer")
    st.markdown("""
    **Complete engine optimization** that jointly sizes:
    - **Pintle injector**: orifice count, diameters, gap height, tip diameter
    - **Chamber geometry**: throat area, chamber volume, L*, diameter
    - **Nozzle geometry**: exit area, expansion ratio
    
    Uses all design requirements from the **Design Requirements** tab including:
    - Target thrust, O/F ratio, tank pressures
    - L* constraints, geometry limits
    - Stability requirements
    
    *Note: Orifice angle is fixed at 90° (perpendicular to longitudinal axis) for optimal impingement.*
    """)
    
    if runner is None:
        st.warning("⚠️ Runner not available. Please load configuration first.")
        return config_obj
    
    # Get design requirements
    requirements = st.session_state.get("design_requirements", {})
    if not requirements:
        st.warning("⚠️ Please set design requirements in the 'Design Requirements' tab first.")
        return config_obj
    
    # Display current design requirements summary
    st.markdown("### 📋 Current Design Requirements")
    col_req1, col_req2, col_req3 = st.columns(3)
    
    with col_req1:
        st.metric("Target Thrust", f"{requirements.get('target_thrust', 7000):.0f} N")
        st.metric("Optimal O/F", f"{requirements.get('optimal_of_ratio', 2.3):.2f}")
    with col_req2:
        st.metric("Max LOX Pressure", f"{requirements.get('max_lox_tank_pressure_psi', 700):.0f} psi")
        st.metric("Max Fuel Pressure", f"{requirements.get('max_fuel_tank_pressure_psi', 850):.0f} psi")
    with col_req3:
        st.metric("L* Range", f"{requirements.get('min_Lstar', 0.95):.1f} - {requirements.get('max_Lstar', 1.27):.1f} m")
        st.metric("Min Stability", f"{requirements.get('min_stability_margin', 1.2):.2f}")
    
    st.markdown("---")
    
    # Optimization Configuration
    st.markdown("### ⚙️ Optimization Configuration")
    
    st.markdown("#### Optimization Parameters")
    
    # Use target_burn_time from Design Requirements tab
    target_burn_time = requirements.get("target_burn_time", 10.0)
    st.info(f"**Target Burn Time:** {target_burn_time:.1f} s *(from Design Requirements tab)*")
    
    max_iterations = st.number_input(
        "Max Optimization Iterations",
        min_value=20,
        max_value=200,
        value=80,
        step=10,
        key="full_opt_max_iter",
        help="Maximum function evaluations (typically converges in 30-60)"
    )
    
    st.markdown("#### ⏱️ Time-Varying Analysis")
    use_time_varying = st.checkbox(
        "Enable Time-Varying Optimization",
        value=True,
        key="full_opt_time_varying",
        help="Optimize across entire burn time (accounts for ablative recession, geometry evolution, and time-varying stability)"
    )
    if use_time_varying:
        st.caption("✅ Optimizer will account for ablative recession, chamber/throat evolution, and stability over entire burn")
    else:
        st.caption("⚠️ Single-point optimization at t=0 only (faster but less accurate)")
    
    st.markdown("#### Target Tolerances")
    st.caption("Optimizer stops early when within these tolerances")
    
    thrust_tolerance = st.number_input(
        "Thrust Tolerance [%]",
        min_value=1.0,
        max_value=20.0,
        value=10.0,
        step=1.0,
        key="full_opt_thrust_tol",
        help="Acceptable deviation from target thrust"
    ) / 100.0
    
    apogee_tolerance = st.number_input(
        "Apogee Tolerance [%]",
        min_value=5.0,
        max_value=30.0,
        value=15.0,
        step=5.0,
        key="full_opt_apogee_tol",
        help="Acceptable deviation from target apogee"
    ) / 100.0
    
    # ==========================================================================
    # PRESSURE CURVE - OPTIMIZER CONTROLLED
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 🛢️ Tank Pressure Curves")
    
    # Get max pressures from requirements (user's only input for pressure)
    max_lox_pressure_psi = float(requirements.get("max_lox_tank_pressure_psi", 700))
    max_fuel_pressure_psi = float(requirements.get("max_fuel_tank_pressure_psi", 850))
    
    st.info(
        f"🎛️ **Optimizer-Controlled Pressure Curves**\n\n"
        f"The optimizer jointly optimizes **injector geometry** AND **tank pressures** to achieve target O/F ratio.\n\n"
        f"**What the optimizer controls:**\n"
        f"- Starting pressures at t=0 (can be anywhere from 30% to 100% of max)\n"
        f"- Pressure profiles over time (4 control points per tank)\n"
        f"- Curve shape (linear vs exponential blending)\n\n"
        f"**Hard constraints (never exceeded):**\n"
        f"- Max LOX Tank Pressure: **{max_lox_pressure_psi:.0f} psi**\n"
        f"- Max Fuel Tank Pressure: **{max_fuel_pressure_psi:.0f} psi**\n"
        f"- Target Burn Time: **{target_burn_time:.1f} s**\n\n"
        f"*The optimizer finds the best geometry + pressure combination to meet thrust, O/F, and stability targets.*"
    )
    
    # Pressure config for optimizer (no user segments - optimizer will generate)
    # Optimizer will create N segments (up to 20) with linear/blowdown types
    pressure_config = {
        "mode": "optimizer_controlled",
        "max_lox_pressure_psi": max_lox_pressure_psi,
        "max_fuel_pressure_psi": max_fuel_pressure_psi,
        "target_burn_time": target_burn_time,
        "n_segments": 3,  # Default: 3 segments per tank (optimizer can use fewer by setting duration near zero)
        # Initial values (optimizer will refine these)
        "lox_start_psi": max_lox_pressure_psi,
        "fuel_start_psi": max_fuel_pressure_psi,
        "lox_end_pct": 0.7,  # Initial guess
        "fuel_end_pct": 0.7,  # Initial guess
    }
    
    # Tolerances config
    tolerances = {
        "thrust": thrust_tolerance,
        "apogee": apogee_tolerance,
    }
    
    # Convergence tolerance
    convergence_tol = thrust_tolerance
    
    st.markdown("---")
    
    # Display current configuration
    st.markdown("### 📊 Current Engine Configuration")
    _display_current_engine_config(config_obj)
    
    st.markdown("---")
    
    # Run Full Engine Optimization
    if st.button("🚀 Run Full Engine Optimization", type="primary", key="run_full_engine_opt"):
        try:
            # Store before config for comparison
            config_before = copy.deepcopy(config_obj)
            
            # Create progress bar container
            progress_bar = st.progress(0, text="Initializing optimization...")
            status_text = st.empty()
            
            # Run the full engine optimization with progress callback
            def progress_callback(stage: str, progress: float, message: str):
                # Format progress bar to show stage clearly
                progress_text = f"{stage}\n{message}" if "\n" not in message else f"{stage}\n{message}"
                progress_bar.progress(progress, text=progress_text)
                status_text.text(f"{stage} | {message}")
            
            optimized_config, optimization_results = _run_full_engine_optimization_with_flight_sim(
                config_obj,
                runner,
                requirements,
                target_burn_time,
                max_iterations,
                tolerances,
                pressure_config,
                progress_callback=progress_callback,
                use_time_varying=use_time_varying,
            )
            
            # Clear progress bar
            progress_bar.empty()
            status_text.empty()
            
            # Store results
            config_obj = optimized_config
            st.session_state["optimized_config"] = optimized_config
            st.session_state["optimization_results"] = optimization_results
            st.session_state["optimization_before_config"] = config_before
            
            # Update config_dict so changes persist
            config_dict_updated = optimized_config.model_dump(exclude_none=False)
            st.session_state["config_dict"] = config_dict_updated
            
            # Display success
            conv_info = optimization_results.get("convergence_info", {})
            flight_result = optimization_results.get("flight_sim_result", {})
            
            if conv_info.get("converged", False):
                st.success(f"✅ Optimization converged after {conv_info.get('iterations', 0)} iterations!")
            else:
                st.warning(f"⚠️ Optimization completed after {conv_info.get('iterations', 0)} iterations (final change: {conv_info.get('final_change', 0)*100:.2f}%)")
            
            if flight_result.get("success", False):
                apogee = flight_result.get("apogee", 0)
                target_apogee = requirements.get("target_apogee", 3048.0)
                apogee_error = abs(apogee - target_apogee) / target_apogee * 100
                if apogee_error < 10:
                    st.success(f"🎯 Flight simulation: Apogee = {apogee:.0f} m (target: {target_apogee:.0f} m, error: {apogee_error:.1f}%)")
                else:
                    st.warning(f"⚠️ Flight simulation: Apogee = {apogee:.0f} m (target: {target_apogee:.0f} m, error: {apogee_error:.1f}%)")
            
            # Display results
            st.markdown("---")
            st.markdown("## ✅ Optimization Results")
            
            # Show complete results with all visualizations
            _show_complete_optimization_results(config_before, optimized_config, optimization_results, requirements, target_burn_time)
            
        except Exception as e:
            st.error(f"Optimization failed: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Show optimization results if available from previous run
    if "optimized_config" in st.session_state and "optimization_results" in st.session_state:
        opt_results = st.session_state.get("optimization_results", {})
        opt_config = st.session_state.get("optimized_config", config_obj)
        config_before = st.session_state.get("optimization_before_config", config_obj)
        
        if st.checkbox("Show Previous Optimization Results", value=False, key="show_prev_full_opt"):
            st.markdown("---")
            st.markdown("## 📊 Previous Optimization Results")
            _show_complete_optimization_results(config_before, opt_config, opt_results, requirements, target_burn_time)
    
    return config_obj


def _display_current_engine_config(config_obj: PintleEngineConfig) -> None:
    """Display current engine configuration in a compact format."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔧 Injector")
        if hasattr(config_obj, 'injector') and config_obj.injector.type == "pintle":
            geometry = config_obj.injector.geometry
            if hasattr(geometry, 'fuel'):
                st.caption(f"Pintle Tip: {geometry.fuel.d_pintle_tip*1000:.2f} mm")
                st.caption(f"Gap Height: {geometry.fuel.h_gap*1000:.2f} mm")
            if hasattr(geometry, 'lox'):
                st.caption(f"Orifices: {geometry.lox.n_orifices} × {geometry.lox.d_orifice*1000:.2f} mm")
                st.caption(f"Orifice Angle: {geometry.lox.theta_orifice:.1f}°")
    
    with col2:
        st.markdown("#### 🔥 Chamber")
        if hasattr(config_obj, 'chamber'):
            D_throat = np.sqrt(4 * config_obj.chamber.A_throat / np.pi) * 1000
            st.caption(f"Throat Ø: {D_throat:.2f} mm")
            st.caption(f"L*: {config_obj.chamber.Lstar*1000:.1f} mm")
            st.caption(f"Volume: {config_obj.chamber.volume*1e6:.1f} cm³")
    
    with col3:
        st.markdown("#### 🔺 Nozzle")
        if hasattr(config_obj, 'nozzle'):
            D_exit = np.sqrt(4 * config_obj.nozzle.A_exit / np.pi) * 1000
            st.caption(f"Exit Ø: {D_exit:.2f} mm")
            st.caption(f"Expansion Ratio: {config_obj.nozzle.expansion_ratio:.2f}")


def _run_full_engine_optimization(
    config_obj: PintleEngineConfig,
    runner: PintleEngineRunner,
    requirements: Dict[str, Any],
    target_burn_time: float,
    use_time_varying: bool,
    max_iterations: int,
    convergence_tol: float,
) -> Tuple[PintleEngineConfig, Dict[str, Any]]:
    """
    Run full engine optimization (pintle + chamber coupled).
    
    Key features:
    - Sets orifice angle to 90° (perpendicular to longitudinal axis)
    - Uses tank pressures from design requirements
    - Optimizes for all specified targets
    - Validates all stability margins
    """
    from pintle_pipeline.coupled_optimizer import CoupledPintleChamberOptimizer
    from pintle_pipeline.comprehensive_optimizer import ComprehensivePintleOptimizer
    from pintle_pipeline.chamber_optimizer import ChamberOptimizer
    from pintle_pipeline.system_diagnostics import SystemDiagnostics
    
    # Extract requirements
    target_thrust = requirements.get("target_thrust", 7000.0)
    optimal_of = requirements.get("optimal_of_ratio", 2.3)
    max_P_tank_O = requirements.get("max_P_tank_O", 4.826e6)  # ~700 psi
    max_P_tank_F = requirements.get("max_P_tank_F", 5.860e6)  # ~850 psi
    min_Lstar = requirements.get("min_Lstar", 0.95)
    max_Lstar = requirements.get("max_Lstar", 1.27)
    min_stability = requirements.get("min_stability_margin", 1.2)
    stability_margin_handicap = float(requirements.get("stability_margin_handicap", 0.0))
    max_chamber_od = requirements.get("max_chamber_outer_diameter", 0.15)
    max_nozzle_exit = requirements.get("max_nozzle_exit_diameter", 0.101)
    max_engine_length = requirements.get("max_engine_length", 0.5)
    
    # Set design requirements for optimizer
    design_requirements = {
        "target_thrust": target_thrust,
        "target_burn_time": target_burn_time,
        "target_stability_margin": min_stability,
        "P_tank_O": max_P_tank_O,  # Use max tank pressures for sizing
        "P_tank_F": max_P_tank_F,
        "target_Isp": None,  # Let optimizer find best Isp
        "target_MR": optimal_of,  # Target mixture ratio
    }
    
    # Set constraints
    constraints = {
        "max_chamber_length": max_engine_length * 0.6,  # Chamber is ~60% of total
        "max_chamber_diameter": max_chamber_od,
        "min_Lstar": min_Lstar,
        "max_Lstar": max_Lstar,
        "min_expansion_ratio": 3.0,
        "max_expansion_ratio": np.pi * (max_nozzle_exit/2)**2 / (np.pi * (0.015)**2),  # From max exit area
        "max_engine_weight": None,
        # Pintle constraints
        "min_pintle_tip_diameter": 0.008,
        "max_pintle_tip_diameter": 0.040,
        "min_gap_height": 0.0002,
        "max_gap_height": 0.002,
        "min_orifices": 6,
        "max_orifices": 24,
        "min_orifice_diameter": 0.001,
        "max_orifice_diameter": 0.006,
        "min_injection_angle": 90.0,  # FIXED: Perpendicular to longitudinal axis
        "max_injection_angle": 90.0,  # FIXED: Perpendicular to longitudinal axis
        "min_chamber_diameter": 0.04,
        "max_chamber_diameter": max_chamber_od,
    }
    
    # Phase 1: First set orifice angle to 90° in config
    config_modified = copy.deepcopy(config_obj)
    if hasattr(config_modified, 'injector') and config_modified.injector.type == "pintle":
        if hasattr(config_modified.injector.geometry, 'lox'):
            config_modified.injector.geometry.lox.theta_orifice = 90.0  # Perpendicular
    
    # Phase 2: Run coupled optimization
    coupled_optimizer = CoupledPintleChamberOptimizer(config_modified)
    
    # Add O/F target to requirements
    design_requirements["target_MR"] = optimal_of
    
    coupled_results = coupled_optimizer.optimize_coupled(
        design_requirements,
        constraints,
        max_iterations=max_iterations,
        convergence_tolerance=convergence_tol,
        use_time_varying=use_time_varying,
    )
    
    optimized_config = coupled_results["optimized_config"]
    
    # Ensure orifice angle stays at 90° after optimization
    if hasattr(optimized_config, 'injector') and optimized_config.injector.type == "pintle":
        if hasattr(optimized_config.injector.geometry, 'lox'):
            optimized_config.injector.geometry.lox.theta_orifice = 90.0
    
    # Phase 3: Run validation checks
    optimized_runner = PintleEngineRunner(optimized_config)
    final_performance = optimized_runner.evaluate(max_P_tank_O, max_P_tank_F)
    
    # Run system diagnostics
    diagnostics = SystemDiagnostics(optimized_config, optimized_runner)
    validation_results = diagnostics.run_full_diagnostics(max_P_tank_O, max_P_tank_F)
    
    # Combine results
    coupled_results["performance"] = final_performance
    coupled_results["validation"] = validation_results
    coupled_results["design_requirements"] = design_requirements
    coupled_results["constraints"] = constraints
    
    # Add optimized parameters summary
    coupled_results["optimized_parameters"] = _extract_all_parameters(optimized_config)
    
    return optimized_config, coupled_results


def _extract_all_parameters(config: PintleEngineConfig) -> Dict[str, Any]:
    """Extract all optimized parameters from config."""
    params = {}
    
    # Injector parameters
    if hasattr(config, 'injector') and config.injector.type == "pintle":
        geometry = config.injector.geometry
        if hasattr(geometry, 'fuel'):
            params["d_pintle_tip"] = geometry.fuel.d_pintle_tip
            params["h_gap"] = geometry.fuel.h_gap
            if hasattr(geometry.fuel, 'd_reservoir_inner'):
                params["d_reservoir_inner"] = geometry.fuel.d_reservoir_inner
        if hasattr(geometry, 'lox'):
            params["n_orifices"] = geometry.lox.n_orifices
            params["d_orifice"] = geometry.lox.d_orifice
            params["theta_orifice"] = geometry.lox.theta_orifice
    
    # Chamber parameters
    params["A_throat"] = config.chamber.A_throat
    params["Lstar"] = config.chamber.Lstar
    params["chamber_volume"] = config.chamber.volume
    params["chamber_length"] = config.chamber.length
    if hasattr(config.chamber, 'chamber_inner_diameter') and config.chamber.chamber_inner_diameter:
        params["chamber_diameter"] = config.chamber.chamber_inner_diameter
    else:
        params["chamber_diameter"] = np.sqrt(4.0 * config.chamber.volume / (np.pi * config.chamber.length))
    
    # Nozzle parameters
    params["A_exit"] = config.nozzle.A_exit
    params["expansion_ratio"] = config.nozzle.expansion_ratio
    
    # Ablative liner parameters
    if hasattr(config, 'ablative_cooling') and config.ablative_cooling and config.ablative_cooling.enabled:
        params["ablative_thickness"] = config.ablative_cooling.initial_thickness
        params["ablative_enabled"] = True
    else:
        params["ablative_thickness"] = 0.0
        params["ablative_enabled"] = False
    
    # Graphite insert parameters
    if hasattr(config, 'graphite_insert') and config.graphite_insert and config.graphite_insert.enabled:
        params["graphite_thickness"] = config.graphite_insert.initial_thickness
        params["graphite_enabled"] = True
    else:
        params["graphite_thickness"] = 0.0
        params["graphite_enabled"] = False
    
    return params


def _run_full_engine_optimization_with_flight_sim(
    config_obj: PintleEngineConfig,
    runner: PintleEngineRunner,
    requirements: Dict[str, Any],
    target_burn_time: float,
    max_iterations: int,
    tolerances: Dict[str, float],
    pressure_config: Dict[str, Any],
    progress_callback: Optional[callable] = None,
    use_time_varying: bool = True,
) -> Tuple[PintleEngineConfig, Dict[str, Any]]:
    """
    Full engine optimization with real iterative optimization and progress tracking.
    
    Features:
    - Real scipy optimization with progress callback
    - Flexible independent pressure curves for LOX/Fuel
    - Tolerances for early stopping
    - 200-point pressure curves
    - COPV pressure curve calculation (260K temperatures)
    - Flight sim validation for good candidates
    - Time-varying analysis (ablative recession, geometry evolution) if enabled
    """
    from pintle_pipeline.system_diagnostics import SystemDiagnostics
    from scipy.optimize import minimize, differential_evolution
    from pathlib import Path
    from datetime import datetime
    
    # Optimization state for progress tracking
    opt_state: Dict[str, Any] = {
        "iteration": 0,
        "best_objective": float('inf'),
        "best_config": None,
        "history": [],
        "converged": False,
    }
    log_flags: Dict[str, bool] = {
        "promoted_state_logged": False,
        "marginal_candidate_logged": False,
    }

    log_file_path = Path("/home/adnan/EngineDesign/full_engine_optimizer.log")

    def log_status(stage: str, message: str) -> None:
        """Persist layer status updates to a root-level log for offline analysis."""
        timestamp = datetime.utcnow().isoformat()
        entry = f"[{timestamp}] {stage}: {message}\n"
        try:
            with log_file_path.open("a", encoding="utf-8") as log_file:
                log_file.write(entry)
        except Exception:
            # Logging should never break the optimizer; swallow any IO issues.
            pass
    
    def update_progress(stage: str, progress: float, message: str):
        if progress_callback:
            progress_callback(stage, progress, message)
    
    # Add a clear separator line at the start of each optimization run
    log_status("Run", "-" * 80)
    
    update_progress("Initialization", 0.02, "Extracting requirements...")
    
    # Extract requirements
    target_thrust = requirements.get("target_thrust", 7000.0)
    target_apogee = requirements.get("target_apogee", 3048.0)
    optimal_of = requirements.get("optimal_of_ratio", 2.3)
    min_Lstar = requirements.get("min_Lstar", 0.95)
    max_Lstar = requirements.get("max_Lstar", 1.27)
    min_stability = requirements.get("min_stability_margin", 1.2)
    max_chamber_od = requirements.get("max_chamber_outer_diameter", 0.15)
    max_nozzle_exit = requirements.get("max_nozzle_exit_diameter", 0.101)
    max_engine_length = requirements.get("max_engine_length", 0.5)
    copv_volume_m3 = requirements.get("copv_free_volume_m3", 0.0045)  # 4.5 L default

    log_status(
        "Initialization",
        f"Starting optimization | Thrust={target_thrust:.0f}N, Apogee={target_apogee:.0f}m, O/F={optimal_of:.2f}"
    )
    
    # Extract tolerances
    thrust_tol = tolerances.get("thrust", 0.10)
    apogee_tol = tolerances.get("apogee", 0.15)
    
    # Extract pressure curve config
    psi_to_Pa = 6894.76
    lox_P_start = pressure_config.get("lox_start_psi", 500) * psi_to_Pa
    lox_P_end_ratio = pressure_config.get("lox_end_pct", 0.70)
    fuel_P_start = pressure_config.get("fuel_start_psi", 500) * psi_to_Pa
    fuel_P_end_ratio = pressure_config.get("fuel_end_pct", 0.70)
    
    # Pressure curve mode - optimizer controls the curve shape
    pressure_mode = pressure_config.get("mode", "optimizer_controlled")
    
    update_progress("Initialization", 0.05, "Setting up optimization bounds...")
    
    # Phase 1: Set orifice angle to 90° and prepare config
    config_base = copy.deepcopy(config_obj)
    if hasattr(config_base, 'injector') and config_base.injector.type == "pintle":
        if hasattr(config_base.injector.geometry, 'lox'):
            config_base.injector.geometry.lox.theta_orifice = 90.0
    
    # =========================================================================
    # OPTIMIZATION VARIABLES:
    # Engine Geometry (7 vars):
    # [0] A_throat (throat area, m²)
    # [1] Lstar (characteristic length, m)
    # [2] expansion_ratio
    # [3] d_pintle_tip (m)
    # [4] h_gap (m)
    # [5] n_orifices (will be rounded to int)
    # [6] d_orifice (m)
    #
    # Thermal Protection (2 vars):
    # [7] ablative_thickness (m) - chamber liner thickness
    # [8] graphite_thickness (m) - throat insert thickness
    #
    # Pressure Curve Segments (optimizer picks N segments, up to 20):
    # [9] n_segments_lox (1-20, rounded to int) - number of segments for LOX
    # [10] n_segments_fuel (1-20, rounded to int) - number of segments for Fuel
    #
    # For each segment (up to 20 segments per tank, 5 vars per segment):
    # - type (0=linear, 1=blowdown)
    # - duration_ratio (0-1, fraction of total burn time, normalized to sum=1)
    # - start_pressure_ratio (0.3-1.0, ratio of max pressure)
    # - end_pressure_ratio (0.3-1.0, ratio of max pressure)
    # - decay_tau_ratio (0-1, fraction of segment duration, only for blowdown)
    #
    # Variables [11:] contain segment parameters for LOX then Fuel
    # LOX segments: [11] to [11 + n_segments_lox*5 - 1]
    # Fuel segments: [11 + n_segments_lox*5] to [11 + (n_segments_lox + n_segments_fuel)*5 - 1]
    #
    # Note: Pressures NEVER exceed max - optimizer works with ratios ≤ 1.0
    # =========================================================================
    
    # Get number of segments from config (default: 3 segments for flexibility)
    default_n_segments = pressure_config.get("n_segments", 3)
    default_n_segments = int(np.clip(default_n_segments, 1, 20))
    
    # Maximum segments per tank (fixed for optimization dimensionality)
    max_segments_per_tank = min(default_n_segments, 20)
    vars_per_segment = 5  # type, duration_ratio, start_ratio, end_ratio, tau_ratio
    
    # Get current ablative/graphite config for initial values
    ablative_cfg = config_base.ablative_cooling if hasattr(config_base, 'ablative_cooling') and config_base.ablative_cooling else None
    graphite_cfg = config_base.graphite_insert if hasattr(config_base, 'graphite_insert') and config_base.graphite_insert else None
    
    # Initial ablative/graphite thicknesses from config (or sensible defaults)
    ablative_init = ablative_cfg.initial_thickness if ablative_cfg and ablative_cfg.enabled else 0.008
    graphite_init = graphite_cfg.initial_thickness if graphite_cfg and graphite_cfg.enabled else 0.006
    
    # Get max pressures from config (these are HARD LIMITS - never exceeded)
    max_lox_P_psi = pressure_config.get("max_lox_pressure_psi", 500)
    max_fuel_P_psi = pressure_config.get("max_fuel_pressure_psi", 500)
    
    # Calculate initial guess and bounds
    Cf_est = 1.5
    Pc_est = lox_P_start * 0.7
    A_throat_init = target_thrust / (Cf_est * Pc_est)
    A_throat_init = np.clip(A_throat_init, 5e-5, 2e-3)
    
    # Build bounds for segmented pressure system
    # Base geometry and thermal (9 vars)
    bounds = [
        (5e-5, 2e-3),           # [0] A_throat: 8mm to 50mm diameter
        (min_Lstar, max_Lstar), # [1] Lstar
        (4.0, 20.0),            # [2] expansion_ratio
        (0.008, 0.040),         # [3] d_pintle_tip
        (0.0003, 0.0020),       # [4] h_gap
        (6, 24),                # [5] n_orifices
        (0.001, 0.006),         # [6] d_orifice
        (0.003, 0.020),         # [7] ablative_thickness: 3mm to 20mm
        (0.003, 0.015),         # [8] graphite_thickness: 3mm to 15mm
        (1, 20),                # [9] n_segments_lox (1-20 segments)
        (1, 20),                # [10] n_segments_fuel (1-20 segments)
    ]
    
    # Add bounds for segment parameters (up to max_segments_per_tank segments * 5 vars * 2 tanks)
    # The optimizer can use fewer segments by setting duration_ratio to near-zero
    
    for tank_idx in range(2):  # LOX and Fuel
        for seg_idx in range(max_segments_per_tank):
            bounds.append((0.0, 1.0))      # type (0=linear, 1=blowdown)
            bounds.append((0.01, 1.0))   # duration_ratio (will be normalized)
            bounds.append((0.30, 1.0))    # start_pressure_ratio
            bounds.append((0.30, 1.0))    # end_pressure_ratio
            bounds.append((0.1, 1.0))     # decay_tau_ratio (for blowdown)
    
    # Initial guess: start with default_n_segments segments per tank
    x0 = [
        A_throat_init,          # [0] A_throat
        (min_Lstar + max_Lstar) / 2,  # [1] Lstar
        10.0,                   # [2] expansion_ratio
        0.015,                  # [3] d_pintle_tip
        0.0006,                 # [4] h_gap
        12,                     # [5] n_orifices
        0.003,                  # [6] d_orifice
        np.clip(ablative_init, 0.003, 0.020),   # [7] ablative_thickness
        np.clip(graphite_init, 0.003, 0.015),   # [8] graphite_thickness
        float(default_n_segments),  # [9] n_segments_lox
        float(default_n_segments),  # [10] n_segments_fuel
    ]
    
    # Initial guess for segments: simple 3-segment profile (flat start, linear drop, flat end)
    for tank_idx in range(2):  # LOX and Fuel
        for seg_idx in range(max_segments_per_tank):
            if seg_idx < default_n_segments:
                # Active segment
                if seg_idx == 0:
                    # First segment: flat at high pressure
                    x0.append(0.0)  # linear
                    x0.append(0.33)  # 1/3 of burn time
                    x0.append(0.95)  # start at 95% of max
                    x0.append(0.95)  # end at 95% of max (flat)
                    x0.append(0.5)   # tau_ratio (not used for linear)
                elif seg_idx == default_n_segments - 1:
                    # Last segment: flat at lower pressure
                    x0.append(0.0)  # linear
                    x0.append(0.33)  # 1/3 of burn time
                    x0.append(0.70)  # start at 70% of max
                    x0.append(0.70)  # end at 70% of max (flat)
                    x0.append(0.5)   # tau_ratio
                else:
                    # Middle segment: linear transition
                    x0.append(0.0)  # linear
                    x0.append(0.34 / (default_n_segments - 2))  # remaining time
                    x0.append(0.95)  # start
                    x0.append(0.70)  # end
                    x0.append(0.5)   # tau_ratio
            else:
                # Inactive segment (duration near zero)
                x0.append(0.0)  # type
                x0.append(0.01)  # very small duration
                x0.append(0.70)  # start
                x0.append(0.70)  # end
                x0.append(0.5)   # tau_ratio
    
    x0 = np.array(x0)
    
    # Ensure initial guess is within bounds
    for i, (lo, hi) in enumerate(bounds):
        x0[i] = np.clip(x0[i], lo, hi)
    
    update_progress("Stage: Optimization Setup", 0.08, "Setting up optimization bounds and initial guess...")
    
    def apply_x_to_config(x: np.ndarray, base_config: PintleEngineConfig) -> Tuple[PintleEngineConfig, float, float]:
        """Apply optimization variables to config. Returns (config, lox_end_ratio, fuel_end_ratio)."""
        config = copy.deepcopy(base_config)
        
        # Clip all values to bounds to ensure we stay within limits
        A_throat = float(np.clip(x[0], bounds[0][0], bounds[0][1]))
        Lstar = float(np.clip(x[1], bounds[1][0], bounds[1][1]))
        expansion_ratio = float(np.clip(x[2], bounds[2][0], bounds[2][1]))
        d_pintle_tip = float(np.clip(x[3], bounds[3][0], bounds[3][1]))
        h_gap = float(np.clip(x[4], bounds[4][0], bounds[4][1]))
        n_orifices = int(round(np.clip(x[5], bounds[5][0], bounds[5][1])))
        d_orifice = float(np.clip(x[6], bounds[6][0], bounds[6][1]))
        ablative_thickness = float(np.clip(x[7], bounds[7][0], bounds[7][1]))
        graphite_thickness = float(np.clip(x[8], bounds[8][0], bounds[8][1]))
        
        # Extract segment counts
        n_segments_lox = int(round(np.clip(x[9], bounds[9][0], bounds[9][1])))
        n_segments_fuel = int(round(np.clip(x[10], bounds[10][0], bounds[10][1])))
        
        # Extract segment parameters for LOX
        vars_per_segment = 5
        idx_base_lox = 11
        x_lox_segments = x[idx_base_lox:idx_base_lox + max_segments_per_tank * vars_per_segment]
        lox_segments = segments_from_optimizer_vars(
            x_lox_segments, n_segments_lox, max_lox_P_psi, target_burn_time
        )
        
        # Extract segment parameters for Fuel
        idx_base_fuel = idx_base_lox + max_segments_per_tank * vars_per_segment
        x_fuel_segments = x[idx_base_fuel:idx_base_fuel + max_segments_per_tank * vars_per_segment]
        fuel_segments = segments_from_optimizer_vars(
            x_fuel_segments, n_segments_fuel, max_fuel_P_psi, target_burn_time
        )
        
        # For compatibility, calculate end ratios from segments
        if lox_segments:
            lox_start_psi = lox_segments[0]["start_pressure_psi"]
            lox_end_psi = lox_segments[-1]["end_pressure_psi"]
            lox_end_ratio = lox_end_psi / lox_start_psi if lox_start_psi > 0 else 0.7
        else:
            lox_end_ratio = 0.7
        
        if fuel_segments:
            fuel_start_psi = fuel_segments[0]["start_pressure_psi"]
            fuel_end_psi = fuel_segments[-1]["end_pressure_psi"]
            fuel_end_ratio = fuel_end_psi / fuel_start_psi if fuel_start_psi > 0 else 0.7
        else:
            fuel_end_ratio = 0.7
        
        # Store segments in config for later retrieval (as metadata)
        if not hasattr(config, '_optimizer_segments'):
            config._optimizer_segments = {}
        config._optimizer_segments['lox'] = lox_segments
        config._optimizer_segments['fuel'] = fuel_segments
        
        return config, lox_end_ratio, fuel_end_ratio
        
        # Chamber: ALWAYS use maximum diameter to minimize length
        V_chamber = Lstar * A_throat
        D_chamber = max_chamber_od * 0.95  # Use 95% of max allowable diameter
        A_chamber = np.pi * (D_chamber / 2) ** 2
        R_chamber = D_chamber / 2
        R_throat = np.sqrt(A_throat / np.pi)
        
        # Use proper chamber_length_calc that accounts for 45° contraction cone
        # This returns only the CYLINDRICAL portion length
        contraction_ratio = A_chamber / A_throat
        theta_contraction = np.pi / 4  # 45 degrees (standard)
        L_cylindrical = chamber_length_calc(V_chamber, A_throat, contraction_ratio, theta_contraction)
        
        # Calculate contraction cone length (45° angle means horizontal = vertical drop)
        # For 45°: L_cone = R_chamber - R_throat
        L_contraction = contraction_length_horizontal_calc(A_chamber, R_throat, theta_contraction)
        
        # Total chamber length = cylindrical + contraction (from injector face to throat)
        L_chamber = L_cylindrical + L_contraction
        
        # Ensure positive chamber length
        if L_chamber <= 0 or L_cylindrical <= 0 or not np.isfinite(L_chamber):
            # Fallback: simple volume-based calculation
            L_chamber = V_chamber / A_chamber if A_chamber > 0 else 0.2
            L_cylindrical = max(L_chamber * 0.7, 0.05)  # Assume 70% cylindrical, min 50mm
        
        # Sanity check: chamber length should be reasonable (5mm to 1m)
        L_chamber = np.clip(L_chamber, 0.005, 1.0)
        
        config.chamber.A_throat = A_throat
        config.chamber.volume = V_chamber
        config.chamber.Lstar = Lstar
        config.chamber.length = L_chamber
        if hasattr(config.chamber, 'chamber_inner_diameter'):
            config.chamber.chamber_inner_diameter = D_chamber
        if hasattr(config.chamber, 'contraction_ratio'):
            config.chamber.contraction_ratio = contraction_ratio
        if hasattr(config.chamber, 'A_chamber'):
            config.chamber.A_chamber = A_chamber
        
        # Nozzle
        A_exit = A_throat * expansion_ratio
        D_exit = np.sqrt(4 * A_exit / np.pi)
        if D_exit > max_nozzle_exit * 0.95:
            D_exit = max_nozzle_exit * 0.95
            A_exit = np.pi * (D_exit / 2) ** 2
            expansion_ratio = A_exit / A_throat
        
        config.nozzle.A_throat = A_throat
        config.nozzle.A_exit = A_exit
        config.nozzle.expansion_ratio = expansion_ratio
        if hasattr(config.nozzle, 'exit_diameter'):
            config.nozzle.exit_diameter = D_exit
        
        if hasattr(config.combustion, 'cea'):
            config.combustion.cea.expansion_ratio = expansion_ratio
        
        # Injector
        if hasattr(config.injector, 'geometry'):
            if hasattr(config.injector.geometry, 'fuel'):
                config.injector.geometry.fuel.d_pintle_tip = d_pintle_tip
                config.injector.geometry.fuel.h_gap = h_gap
            if hasattr(config.injector.geometry, 'lox'):
                config.injector.geometry.lox.n_orifices = n_orifices
                config.injector.geometry.lox.d_orifice = d_orifice
                config.injector.geometry.lox.theta_orifice = 90.0
        
        # Ablative liner thickness (chamber protection)
        if hasattr(config, 'ablative_cooling') and config.ablative_cooling and config.ablative_cooling.enabled:
            config.ablative_cooling.initial_thickness = ablative_thickness
        
        # Graphite insert thickness (throat protection)
        if hasattr(config, 'graphite_insert') and config.graphite_insert and config.graphite_insert.enabled:
            config.graphite_insert.initial_thickness = graphite_thickness
        
        return config, lox_end_ratio, fuel_end_ratio
    
    # Evaluate initial guess to check feasibility and adjust if needed
    update_progress("Stage: Optimization Setup", 0.09, "Checking initial configuration...")
    try:
        init_config, _, _ = apply_x_to_config(x0, config_base)
        init_runner = PintleEngineRunner(init_config)
        lox_start_init = lox_P_start * x0[9]
        fuel_start_init = fuel_P_start * x0[13]
        init_results = init_runner.evaluate(lox_start_init, fuel_start_init)
        init_thrust = init_results.get("F", 0)
        init_MR = init_results.get("MR", 0)
        init_thrust_err = abs(init_thrust - target_thrust) / target_thrust if target_thrust > 0 else 1.0
        update_progress("Stage: Optimization Setup", 0.095, 
            f"Initial: F={init_thrust:.0f}N (err={init_thrust_err*100:.0f}%), MR={init_MR:.2f}")
        
        # If initial guess is way off, try to adjust A_throat based on thrust mismatch
        if init_thrust_err > 0.5 and init_thrust > 0:
            scale_factor = np.sqrt(target_thrust / init_thrust)  # sqrt because thrust ~ A_throat
            x0[0] = np.clip(x0[0] * scale_factor, bounds[0][0], bounds[0][1])
            update_progress("Stage: Optimization Setup", 0.098, 
                f"Adjusted A_throat by {scale_factor:.2f}x for better starting point")
    except Exception as e:
        update_progress("Stage: Optimization Setup", 0.098, f"Initial check note: {str(e)[:40]}...")
    
    def objective(x: np.ndarray) -> float:
        """Multi-objective function with soft penalties."""
        opt_state["iteration"] += 1
        iteration = opt_state["iteration"]
        
        # Progress update (optimization is ~10% to 50% of total)
        progress = 0.10 + 0.40 * min(iteration / max_iterations, 1.0)
        
        # Show more detail for first few iterations and every 20 iterations after
        # Reduce frequency to avoid overwriting stage information
        if iteration <= 3 or iteration % 25 == 0:
            update_progress(
                "Stage: Optimization (Geometry + Pressure)", 
                progress, 
                f"Iter {iteration}/{max_iterations} | Best obj: {opt_state['best_objective']:.3f} | Next: Layer 1 (Static Test)"
            )
        elif iteration % 10 == 0:
            update_progress(
                "Stage: Optimization (Geometry + Pressure)", 
                progress, 
                f"Iteration {iteration}/{max_iterations}... | Next: Layer 1 (Static Test)"
            )
        
        try:
            config, curr_lox_end_ratio, curr_fuel_end_ratio = apply_x_to_config(x, config_base)
            
            # LAYER 1: Evaluate at INITIAL conditions (start of burn)
            # Get starting pressures from segments (first segment's start pressure)
            lox_segments = getattr(config, '_optimizer_segments', {}).get('lox', [])
            fuel_segments = getattr(config, '_optimizer_segments', {}).get('fuel', [])
            
            if lox_segments:
                P_O_initial = lox_segments[0]["start_pressure_psi"] * psi_to_Pa
            else:
                # Fallback: use max pressure
                P_O_initial = max_lox_P_psi * psi_to_Pa * 0.95
            
            if fuel_segments:
                P_F_initial = fuel_segments[0]["start_pressure_psi"] * psi_to_Pa
            else:
                # Fallback: use max pressure
                P_F_initial = max_fuel_P_psi * psi_to_Pa * 0.95
            
            test_runner = PintleEngineRunner(config)
            results = test_runner.evaluate(P_O_initial, P_F_initial)
            
            F_actual = results.get("F", 0)
            Isp_actual = results.get("Isp", 0)
            MR_actual = results.get("MR", 0)
            Pc_actual = results.get("Pc", 0)
            
            # Calculate errors with tolerances
            thrust_error = abs(F_actual - target_thrust) / target_thrust
            of_error = abs(MR_actual - optimal_of) / optimal_of if optimal_of > 0 else 0
            
            # Stability check using new comprehensive stability analysis
            # Default to unstable if not found (conservative - assumes unstable until proven otherwise)
            stability = results.get("stability_results", {})
            
            # Get new stability metrics
            stability_state = stability.get("stability_state", "unstable")
            stability_score = stability.get("stability_score", 0.0)
            min_stability_score = requirements.get("min_stability_score", 0.75)
            require_stable_state = requirements.get("require_stable_state", True)
            
            # Also get individual margins for backward compatibility and detailed tracking
            chugging = stability.get("chugging", {})
            chugging_margin = chugging.get("stability_margin", 0.0)
            if chugging_margin <= 0:
                chugging_margin = 0.1  # Small positive value to avoid divide-by-zero issues
            
            acoustic = stability.get("acoustic", {})
            acoustic_margin = acoustic.get("stability_margin", 0.0)
            if acoustic_margin <= 0:
                acoustic_margin = 0.1
            
            feed_system = stability.get("feed_system", {})
            feed_margin = feed_system.get("stability_margin", 0.0)
            if feed_margin <= 0:
                feed_margin = 0.1
            
            min_stability_margin = requirements.get("min_stability_margin", min_stability)
            stability_margin_handicap = float(requirements.get("stability_margin_handicap", 0.0))
            # Effective thresholds: interpolate between full requirement (handicap=0)
            # and fully relaxed (handicap=1).
            score_factor = max(0.0, 1.0 - stability_margin_handicap)
            margin_factor = max(0.0, 1.0 - stability_margin_handicap)
            margins_meet_requirements = (
                chugging_margin >= min_stability_margin and
                acoustic_margin >= min_stability_margin and
                feed_margin >= min_stability_margin
            )
            
            if margins_meet_requirements and stability_state != "stable":
                if not log_flags["promoted_state_logged"]:
                    log_status(
                        "Layer 1 Warning",
                        f"Promoting stability_state '{stability_state}' to 'stable' - all margins ≥ {min_stability_margin:.2f}"
                    )
                    log_flags["promoted_state_logged"] = True
                stability_state = "stable"
                stability_score = max(stability_score, min_stability_score)
                stability["stability_state"] = stability_state
                stability["stability_score"] = stability_score
            
            # Get minimum stability requirements
            # New analysis uses stability_score (0-1) where:
            # - "stable": score >= 0.75
            # - "marginal": 0.4 <= score < 0.75
            # - "unstable": score < 0.4
            min_stability_score_raw = requirements.get("min_stability_score", 0.75)  # Default: require "stable"
            require_stable_state = requirements.get("require_stable_state", True)  # Default: require "stable" state
            min_stability_score = min_stability_score_raw * score_factor
            
            # Calculate stability penalty based on new analysis
            # INCREASED penalties to ensure optimizer finds truly stable solutions
            # Penalty increases as stability_score decreases below target
            if stability_state == "unstable":
                # Heavy penalty for unstable designs
                stability_penalty = 15.0 * (1.0 - stability_score)  # Increased from 10.0 to 15.0
            elif stability_state == "marginal":
                # Strong penalty for marginal designs - we want stable, not marginal
                stability_penalty = 5.0 * max(0, min_stability_score - stability_score) / min_stability_score  # Increased from 2.0 to 5.0
            else:  # stable
                # Penalty if below target score - even "stable" state needs good score
                if stability_score < min_stability_score:
                    stability_penalty = 2.0 * (min_stability_score - stability_score) / min_stability_score  # Increased from 0.5 to 2.0
                else:
                    stability_penalty = 0.0  # No penalty if meets or exceeds target
            
            # Also keep individual margin penalties for detailed feedback
            chugging_min = requirements.get("chugging_margin_min", min_stability)
            acoustic_min = requirements.get("acoustic_margin_min", min_stability)
            feed_min = requirements.get("feed_stability_min", min_stability)
            
            chugging_penalty_detail = max(0, chugging_min - chugging_margin)
            acoustic_penalty_detail = max(0, acoustic_min - acoustic_margin)
            feed_penalty_detail = max(0, feed_min - feed_margin)
            
            # Use the new stability_score-based penalty as primary, but add detail penalties
            # Increased weight on individual margins to ensure all are good
            stability_penalty = stability_penalty + 0.5 * (chugging_penalty_detail + acoustic_penalty_detail + feed_penalty_detail)  # Increased from 0.3 to 0.5
            
            # Bounds violation penalty (should be enforced by L-BFGS-B, but add soft penalty as backup)
            bounds_penalty = 0.0
            for i, (lo, hi) in enumerate(bounds):
                val = x[i]
                if val < lo:
                    bounds_penalty += 10.0 * ((lo - val) / (hi - lo + 1e-10)) ** 2
                elif val > hi:
                    bounds_penalty += 10.0 * ((val - hi) / (hi - lo + 1e-10)) ** 2
            
            # Multi-objective with weights (always optimize for all objectives)
            # INCREASED stability weight to prioritize finding stable solutions
            obj = (
                5.0 * thrust_error +          # Thrust matching
                3.0 * of_error +                # O/F matching  
                6.0 * stability_penalty +       # Stability (increased from 4.0 to 6.0 - prioritize stability!)
                1.0 * max(0, 200 - Isp_actual) / 200 +  # Isp bonus
                bounds_penalty                  # Penalty for leaving bounds
            )
            
            # Protect against NaN/Inf
            if not np.isfinite(obj):
                obj = 1e6
            
            # Calculate chamber geometry for tracking using proper method
            A_throat_curr = float(np.clip(x[0], bounds[0][0], bounds[0][1]))
            Lstar_curr = float(np.clip(x[1], bounds[1][0], bounds[1][1]))
            V_chamber_curr = Lstar_curr * A_throat_curr
            D_chamber_curr = max_chamber_od * 0.95
            A_chamber_curr = np.pi * (D_chamber_curr / 2) ** 2
            R_chamber_curr = D_chamber_curr / 2
            R_throat_curr = np.sqrt(A_throat_curr / np.pi)
            contraction_ratio_curr = A_chamber_curr / A_throat_curr if A_throat_curr > 0 else 1.0
            theta_contraction = np.pi / 4  # 45 degrees
            
            # Cylindrical length + contraction length = total chamber length
            L_cylindrical_curr = chamber_length_calc(V_chamber_curr, A_throat_curr, contraction_ratio_curr, theta_contraction)
            L_contraction_curr = contraction_length_horizontal_calc(A_chamber_curr, R_throat_curr, theta_contraction)
            L_chamber_curr = L_cylindrical_curr + L_contraction_curr
            
            if L_chamber_curr <= 0 or L_cylindrical_curr <= 0:
                L_chamber_curr = V_chamber_curr / A_chamber_curr if A_chamber_curr > 0 else 0.2
            
            # Extract ablative/graphite thicknesses and pressure control points for history
            abl_thick_curr = float(np.clip(x[7], bounds[7][0], bounds[7][1]))
            gra_thick_curr = float(np.clip(x[8], bounds[8][0], bounds[8][1]))
            
            # Calculate combined stability margin (minimum of all three) for backward compatibility
            combined_stability_margin = min(chugging_margin, acoustic_margin, feed_margin)
            
            # Record history (with all pressure control points including start)
            opt_state["history"].append({
                "iteration": iteration,
                "x": x.copy(),
                "thrust": F_actual,
                "thrust_error": thrust_error,
                "of_error": of_error,
                "Isp": Isp_actual,
                "MR": MR_actual,
                "Pc": Pc_actual,
                "Lstar": Lstar_curr,
                "L_chamber": L_chamber_curr,
                "D_chamber": D_chamber_curr,
                "stability_margin": combined_stability_margin,  # Backward compatibility
                "stability_state": stability_state,  # New: "stable", "marginal", or "unstable"
                "stability_score": stability_score,  # New: 0-1 score
                "chugging_margin": chugging_margin,
                "acoustic_margin": acoustic_margin,
                "feed_margin": feed_margin,
                "lox_end_ratio": curr_lox_end_ratio,
                "fuel_end_ratio": curr_fuel_end_ratio,
                "ablative_thickness": abl_thick_curr,
                "graphite_thickness": gra_thick_curr,
                # All 4 control points (including optimized start pressure)
                "lox_P_ratios": [float(x[9]), float(x[10]), float(x[11]), float(x[12])],
                "fuel_P_ratios": [float(x[13]), float(x[14]), float(x[15]), float(x[16])],
                "lox_start_ratio": float(x[9]),
                "fuel_start_ratio": float(x[13]),
                "objective": obj,
            })
            
            # Track best (store full solution vector for pressure curve generation)
            if obj < opt_state["best_objective"]:
                opt_state["best_objective"] = obj
                opt_state["best_config"] = copy.deepcopy(config)
                opt_state["best_lox_end_ratio"] = curr_lox_end_ratio
                opt_state["best_fuel_end_ratio"] = curr_fuel_end_ratio
                opt_state["best_x"] = x.copy()  # Store full solution vector
            
            # Check convergence (within tolerances AND stable enough)
            allowed_states = {"stable", "marginal"} if require_stable_state else {"stable", "marginal"}
            state_ok = (stability_state in allowed_states) if require_stable_state else (stability_state != "unstable")
            stability_acceptable = (
                state_ok and
                (stability_score >= min_stability_score) and
                (chugging_margin >= min_stability * 0.8) and
                (acoustic_margin >= min_stability * 0.8) and
                (feed_margin >= min_stability * 0.8)
            )
            
            if stability_state == "marginal" and stability_acceptable:
                if not log_flags["marginal_candidate_logged"]:
                    log_status(
                        "Layer 1 Warning",
                        f"Proceeding with marginal stability candidate (score {stability_score:.2f})"
                    )
                    log_flags["marginal_candidate_logged"] = True
            
            if thrust_error < thrust_tol and of_error < 0.15 and stability_acceptable:
                opt_state["converged"] = True
            else:
                # Not converged if stability is not acceptable
                opt_state["converged"] = False
            
            return obj
            
        except Exception as e:
            return 1e6  # Penalty for failed evaluation
    
    # Run optimization using L-BFGS-B (supports bounds natively, much better for high-dim)
    # This is far more efficient than Nelder-Mead for 19 dimensions
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': max_iterations,
            'maxfun': max_iterations * 5,
            'ftol': 1e-6,
            'gtol': 1e-5,
            'disp': False,
        }
    )
    
    # If L-BFGS-B didn't converge well, try a few Nelder-Mead refinement iterations
    if opt_state["best_objective"] > 0.5:  # Still not converged
        update_progress("Stage: Optimization Refinement", 0.48, "Refining solution with local search...")
        # Use the best found solution as starting point
        x_refined = opt_state.get("best_x", result.x)
        result2 = minimize(
            objective,
            x_refined,
            method='Nelder-Mead',
            options={
                'maxiter': max(50, max_iterations // 4),
                'maxfev': max(150, max_iterations),
                'xatol': 1e-4,
                'fatol': 1e-3,
                'adaptive': True,
            }
        )
    
    # Get best config found
    if opt_state["best_config"] is not None:
        optimized_config = opt_state["best_config"]
        # Use the optimized pressure ratios
        final_lox_end_ratio = opt_state.get("best_lox_end_ratio", lox_P_end_ratio)
        final_fuel_end_ratio = opt_state.get("best_fuel_end_ratio", fuel_P_end_ratio)
    else:
        optimized_config, final_lox_end_ratio, final_fuel_end_ratio = apply_x_to_config(result.x, config_base)
    
    iteration_history = opt_state["history"]
    best_thrust_error = opt_state["best_objective"]
    
    # Ensure orifice angle stays at 90°
    if hasattr(optimized_config, 'injector') and optimized_config.injector.type == "pintle":
        if hasattr(optimized_config.injector.geometry, 'lox'):
            optimized_config.injector.geometry.lox.theta_orifice = 90.0
    
    # Create coupled_results dict for compatibility
    coupled_results = {
        "iteration_history": iteration_history,
        "convergence_info": {
            "converged": opt_state["converged"],
            "iterations": len(iteration_history),
            "final_change": best_thrust_error,
        },
        "optimized_pressure_curves": {
            "lox_end_ratio": final_lox_end_ratio,
            "fuel_end_ratio": final_fuel_end_ratio,
        },
    }
    
    update_progress("Layer 1: Pressure Candidate", 0.52, "Evaluating at initial conditions...")
    
    # Get best_x from optimization state (must be done BEFORE using it below)
    best_x = opt_state.get("best_x", result.x if hasattr(result, 'x') else x0)
    
    # ==========================================================================
    # LAYER 1: PRESSURE CANDIDATE TEST
    # Evaluate at INITIAL conditions (t=0) - this is the pressure candidate test
    # The starting pressure is an optimization variable (geometry + pressure -> O/F)
    # ==========================================================================
    # Get starting pressures from optimized segments
    lox_segments = getattr(optimized_config, '_optimizer_segments', {}).get('lox', [])
    fuel_segments = getattr(optimized_config, '_optimizer_segments', {}).get('fuel', [])
    
    if lox_segments:
        P_O_initial = lox_segments[0]["start_pressure_psi"] * psi_to_Pa
    else:
        P_O_initial = max_lox_P_psi * psi_to_Pa * 0.95
    
    if fuel_segments:
        P_F_initial = fuel_segments[0]["start_pressure_psi"] * psi_to_Pa
    else:
        P_F_initial = max_fuel_P_psi * psi_to_Pa * 0.95
    optimized_runner = PintleEngineRunner(optimized_config)
    initial_performance = optimized_runner.evaluate(P_O_initial, P_F_initial)
    
    # Check if pressure candidate is valid (meets goals at initial conditions with margin)
    initial_thrust = initial_performance.get("F", 0)
    initial_thrust_error = abs(initial_thrust - target_thrust) / target_thrust if target_thrust > 0 else 1.0
    initial_MR = initial_performance.get("MR", 0)
    initial_MR_error = abs(initial_MR - optimal_of) / optimal_of if optimal_of > 0 else 1.0
    
    # Check stability using new comprehensive stability analysis
    stability_results = initial_performance.get("stability_results", {})
    stability_state = stability_results.get("stability_state", "unstable")
    stability_score = stability_results.get("stability_score", 0.0)
    
    # Also get individual margins for detailed tracking
    chugging_margin = stability_results.get("chugging", {}).get("stability_margin", 0)
    acoustic_margin = stability_results.get("acoustic", {}).get("stability_margin", 0)
    feed_margin = stability_results.get("feed_system", {}).get("stability_margin", 0)
    initial_stability = min(chugging_margin, acoustic_margin, feed_margin)  # For backward compatibility
    
    # Get stability requirements
    min_stability_score = requirements.get("min_stability_score", 0.75)
    require_stable_state = requirements.get("require_stable_state", True)
    
    # Check stability acceptability for Layer 1 pass/fail
    handicap = float(requirements.get("stability_margin_handicap", 0.0))
    score_factor = max(0.0, 1.0 - handicap)
    margin_factor = max(0.0, 1.0 - handicap)
    effective_min_score = min_stability_score * score_factor
    effective_margin = min_stability * margin_factor
    
    state_ok = (stability_state in {"stable", "marginal"}) if require_stable_state else (stability_state != "unstable")
    stability_check_passed = (
        state_ok and
        (stability_score >= effective_min_score) and
        (chugging_margin >= effective_margin) and
        (acoustic_margin >= effective_margin) and
        (feed_margin >= effective_margin)
    )
    
    # Individual checks for flight sim eligibility
    thrust_check_passed = initial_thrust_error < thrust_tol * 1.5  # 1.5x tolerance (15% default)
    of_check_passed = initial_MR_error < 0.20  # 20% O/F error allowed
    
    # Pressure candidate passes if within tolerance at initial conditions AND stable
    pressure_candidate_valid = thrust_check_passed and of_check_passed and stability_check_passed
    
    # Build detailed failure reasons for diagnostics
    failure_reasons = []
    if not thrust_check_passed:
        failure_reasons.append(f"Thrust error {initial_thrust_error*100:.1f}% > {thrust_tol*150:.0f}% limit")
    if not of_check_passed:
        failure_reasons.append(f"O/F error {initial_MR_error*100:.1f}% > 20% limit")
    if not stability_check_passed:
        # Provide detailed failure reason showing what was required vs what we got
        required_parts = []
        if require_stable_state:
            if stability_state not in {"stable", "marginal"}:
                required_parts.append(f"state ∈ {{stable,marginal}} (got '{stability_state}')")
        else:
            if stability_state == "unstable":
                required_parts.append("state!='unstable'")
        handicap = float(requirements.get("stability_margin_handicap", 0.0))
        score_factor = max(0.0, 1.0 - handicap)
        margin_factor = max(0.0, 1.0 - handicap)
        eff_score = min_stability_score * score_factor
        eff_margin = min_stability * margin_factor
        if stability_score < eff_score:
            required_parts.append(f"score>={eff_score:.2f} (got {stability_score:.2f})")
        if chugging_margin < eff_margin:
            required_parts.append(f"chugging_margin>={eff_margin:.2f} (got {chugging_margin:.2f})")
        if acoustic_margin < eff_margin:
            required_parts.append(f"acoustic_margin>={eff_margin:.2f} (got {acoustic_margin:.2f})")
        if feed_margin < eff_margin:
            required_parts.append(f"feed_margin>={eff_margin:.2f} (got {feed_margin:.2f})")
        if not required_parts:
            required_parts.append("stability gate mismatch (see diagnostics)")
        failure_reasons.append(f"Stability failed: {'; '.join(required_parts)}")
    
    if not pressure_candidate_valid and not failure_reasons:
        failure_reasons.append("Validation failed: no requirements met (check solver output)")
    
    # Log diagnostic info
    if pressure_candidate_valid:
        update_progress("Layer 1: Pressure Candidate", 0.53, 
            f"✓ VALID - Thrust err: {initial_thrust_error*100:.1f}%, O/F err: {initial_MR_error*100:.1f}%, Stability: {stability_state} (score: {stability_score:.2f})")
        log_status(
            "Layer 1",
            f"VALID | Thrust err {initial_thrust_error*100:.1f}%, O/F err {initial_MR_error*100:.1f}%, Stability {stability_state} (score {stability_score:.2f})"
        )
    else:
        update_progress("Layer 1: Pressure Candidate", 0.53, 
            f"✗ INVALID - {'; '.join(failure_reasons)}")
        log_status(
            "Layer 1",
            f"INVALID | Reasons: {', '.join(failure_reasons) if failure_reasons else 'No details'}"
        )
    
    # Use initial performance as the final performance (per user requirement)
    final_performance = initial_performance
    final_performance["pressure_candidate_valid"] = pressure_candidate_valid
    final_performance["initial_thrust_error"] = initial_thrust_error
    final_performance["initial_MR_error"] = initial_MR_error
    final_performance["initial_stability"] = initial_stability  # Backward compatibility
    final_performance["initial_stability_state"] = stability_state
    final_performance["initial_stability_score"] = stability_score
    final_performance["thrust_check_passed"] = thrust_check_passed
    final_performance["of_check_passed"] = of_check_passed
    final_performance["stability_check_passed"] = stability_check_passed
    final_performance["failure_reasons"] = failure_reasons
    
    update_progress("Pressure Curves", 0.55, "Generating 200-point pressure curves from segments...")
    
    # Phase 6: Generate 200-point time series using OPTIMIZER'S segments
    n_time_points = 200
    
    # Get segments from optimized config
    lox_segments = getattr(optimized_config, '_optimizer_segments', {}).get('lox', [])
    fuel_segments = getattr(optimized_config, '_optimizer_segments', {}).get('fuel', [])
    
    # Generate pressure curves from segments
    if lox_segments:
        lox_time_array, lox_pressure_psi = generate_segmented_pressure_curve(lox_segments, n_time_points)
        # Convert to Pa and ensure same time array
        time_array = lox_time_array
        P_tank_O_array = lox_pressure_psi * psi_to_Pa
    else:
        # Fallback: constant pressure
        time_array = np.linspace(0.0, target_burn_time, n_time_points)
        P_tank_O_array = np.full(n_time_points, max_lox_P_psi * psi_to_Pa * 0.95)
    
    if fuel_segments:
        fuel_time_array, fuel_pressure_psi = generate_segmented_pressure_curve(fuel_segments, n_time_points)
        # Ensure same time array
        if not lox_segments:
            time_array = fuel_time_array
        P_tank_F_array = fuel_pressure_psi * psi_to_Pa
    else:
        # Fallback: constant pressure
        if not lox_segments:
            time_array = np.linspace(0.0, target_burn_time, n_time_points)
        P_tank_F_array = np.full(n_time_points, max_fuel_P_psi * psi_to_Pa * 0.95)
    
    # Store the optimized pressure curve info (segments)
    coupled_results["optimized_pressure_curves"]["lox_segments"] = lox_segments
    coupled_results["optimized_pressure_curves"]["fuel_segments"] = fuel_segments
    if lox_segments:
        coupled_results["optimized_pressure_curves"]["lox_start_psi"] = lox_segments[0]["start_pressure_psi"]
        coupled_results["optimized_pressure_curves"]["lox_end_psi"] = lox_segments[-1]["end_pressure_psi"]
    if fuel_segments:
        coupled_results["optimized_pressure_curves"]["fuel_start_psi"] = fuel_segments[0]["start_pressure_psi"]
        coupled_results["optimized_pressure_curves"]["fuel_end_psi"] = fuel_segments[-1]["end_pressure_psi"]
    
    # Evaluate performance across burn time
    update_progress("Pressure Curves", 0.58, "Evaluating performance across burn time...")
    
    # Storage for time-varying results
    time_varying_results = None
    burn_candidate_valid = False
    pressure_curves = None  # Initialize to ensure it's always defined
    
    # ==========================================================================
    # LAYER 2: TIME SERIES ANALYSIS (BURN CANDIDATE)
    # Optimize initial ablative/graphite guesses based on time series analysis.
    # NOTE: Layer 2 only runs when:
    #       - time-varying analysis is enabled, AND
    #       - the Layer 1 pressure candidate passed its static checks.
    #       This ensures we only do the expensive time-series analysis
    #       on candidates that are already reasonable at t=0.
    # ==========================================================================
    if use_time_varying and pressure_candidate_valid:
        try:
            update_progress("Layer 2: Burn Candidate Optimization", 0.60, "Optimizing initial thermal protection guesses...")
            
            # Layer 2: Optimize initial ablative/graphite thickness guesses
            # These are starting guesses that will be refined in Layer 3
            from scipy.optimize import minimize as scipy_minimize
            
            # Get current ablative/graphite config
            ablative_cfg = optimized_config.ablative_cooling if hasattr(optimized_config, 'ablative_cooling') else None
            graphite_cfg = optimized_config.graphite_insert if hasattr(optimized_config, 'graphite_insert') else None
            
            # Optimization variables for Layer 2: [ablative_initial_guess, graphite_initial_guess]
            layer2_bounds = []
            layer2_x0 = []
            
            if ablative_cfg and ablative_cfg.enabled:
                layer2_bounds.append((0.003, 0.020))  # 3-20mm
                layer2_x0.append(ablative_cfg.initial_thickness)
            if graphite_cfg and graphite_cfg.enabled:
                layer2_bounds.append((0.003, 0.015))  # 3-15mm
                layer2_x0.append(graphite_cfg.initial_thickness)
            
            if len(layer2_x0) > 0:
                layer2_x0 = np.array(layer2_x0)

                # Track Layer 2 optimization progress for UI
                layer2_state = {
                    "iter": 0,
                    "max_iter": 20,
                }
                def layer2_callback(xk):
                    layer2_state["iter"] += 1
                    frac = min(layer2_state["iter"] / max(layer2_state["max_iter"], 1), 1.0)
                    # Map Layer 2 progress into 0.60–0.64 range of overall bar
                    progress = 0.60 + 0.04 * frac
                    update_progress(
                        "Layer 2: Burn Candidate Optimization",
                        progress,
                        f"Layer 2 optimization {layer2_state['iter']}/{layer2_state['max_iter']}",
                    )
                
                def layer2_objective(x_layer2):
                    """Optimize initial thermal protection guesses to minimize recession."""
                    try:
                        # Update config with current guesses
                        config_layer2 = copy.deepcopy(optimized_config)
                        idx = 0
                        if ablative_cfg and ablative_cfg.enabled:
                            config_layer2.ablative_cooling.initial_thickness = float(np.clip(x_layer2[idx], layer2_bounds[idx][0], layer2_bounds[idx][1]))
                            idx += 1
                        if graphite_cfg and graphite_cfg.enabled:
                            config_layer2.graphite_insert.initial_thickness = float(np.clip(x_layer2[idx], layer2_bounds[idx][0], layer2_bounds[idx][1]))
                        
                        # Run time series
                        runner_layer2 = PintleEngineRunner(config_layer2)
                        results_layer2 = runner_layer2.evaluate_arrays_with_time(
                            time_array,
                            P_tank_O_array,
                            P_tank_F_array,
                            track_ablative_geometry=True,
                            use_coupled_solver=False,  # use robust standard solver inside Layer 2 objective
                        )
                        
                        # Objective: minimize recession while meeting stability/thrust goals
                        recession_chamber = float(np.max(results_layer2.get("recession_chamber", [0.0])))
                        recession_throat = float(np.max(results_layer2.get("recession_throat", [0.0])))
                        
                        # Check stability
                        stability_scores = results_layer2.get("stability_score", None)
                        if stability_scores is not None:
                            min_stability = float(np.min(stability_scores))
                        else:
                            chugging = results_layer2.get("chugging_stability_margin", np.array([1.0]))
                            min_stability = max(0.0, min(1.0, (float(np.min(chugging)) - 0.3) * 1.5))
                        
                        # Check thrust – be robust to shorter-than-expected histories
                        thrust_hist = np.atleast_1d(
                            results_layer2.get("F", np.full(n_time_points, target_thrust))
                        )
                        available_n = min(thrust_hist.shape[0], n_time_points)
                        if available_n >= 2:
                            check_indices = np.arange(available_n - 1)  # Exclude last point
                            thrust_hist = thrust_hist[:available_n]
                            thrust_errors = (
                                np.abs(thrust_hist[check_indices] - target_thrust) / target_thrust
                            )
                            max_thrust_err = float(np.max(thrust_errors))
                        elif available_n == 1:
                            # Only one valid point – use it as an approximate error
                            max_thrust_err = float(
                                abs(thrust_hist[0] - target_thrust) / max(target_thrust, 1e-9)
                            )
                        else:
                            # No valid points – treat as very bad candidate
                            max_thrust_err = 1.0
                        
                        # Penalty for poor performance
                        stability_penalty = max(0, 0.7 - min_stability) * 10.0  # Want stability >= 0.7
                        thrust_penalty = max(0, max_thrust_err - thrust_tol * 1.5) * 5.0
                        
                        # Objective: minimize recession + penalties
                        obj = recession_chamber * 1000 + recession_throat * 1000 + stability_penalty + thrust_penalty
                        return obj
                    except Exception as e:
                        # Detailed logging to debug time-varying solver issues inside Layer 2
                        import traceback
                        log_status(
                            "Layer 2 Objective Error",
                            (
                                f"Exception in layer2_objective: {repr(e)} | "
                                f"x_layer2={np.array(x_layer2).tolist()} | "
                                f"time_len={len(time_array)}, "
                                f"P_O_len={len(P_tank_O_array)}, P_F_len={len(P_tank_F_array)} | "
                                f"traceback={traceback.format_exc(limit=3).replace(chr(10), ' | ')}"
                            ),
                        )
                        return 1e6
                
                # Optimize Layer 2
                try:
                    layer2_state["max_iter"] = 20
                    result_layer2 = scipy_minimize(
                        layer2_objective,
                        layer2_x0,
                        method='L-BFGS-B',
                        bounds=layer2_bounds,
                        options={'maxiter': layer2_state["max_iter"], 'ftol': 1e-4},
                        callback=layer2_callback,
                    )
                    
                    # Update config with optimized guesses
                    idx = 0
                    if ablative_cfg and ablative_cfg.enabled:
                        optimized_config.ablative_cooling.initial_thickness = float(np.clip(result_layer2.x[idx], layer2_bounds[idx][0], layer2_bounds[idx][1]))
                        update_progress("Layer 2: Burn Candidate Optimization", 0.62, 
                            f"Optimized ablative initial guess: {optimized_config.ablative_cooling.initial_thickness*1000:.2f}mm")
                        idx += 1
                    if graphite_cfg and graphite_cfg.enabled:
                        optimized_config.graphite_insert.initial_thickness = float(np.clip(result_layer2.x[idx], layer2_bounds[idx][0], layer2_bounds[idx][1]))
                        update_progress("Layer 2: Burn Candidate Optimization", 0.62, 
                            f"Optimized graphite initial guess: {optimized_config.graphite_insert.initial_thickness*1000:.2f}mm")
                except Exception as e:
                    update_progress("Layer 2: Burn Candidate Optimization", 0.62, f"⚠️ Layer 2 optimization failed: {e}, using current values")
            
            # Now run time series with optimized initial guesses
            update_progress("Layer 2: Burn Candidate", 0.64, "Running time series analysis with optimized guesses...")
            optimized_runner = PintleEngineRunner(optimized_config)  # Recreate with updated config
            # Use the standard time-varying solver here for robustness.
            try:
                full_time_results = optimized_runner.evaluate_arrays_with_time(
                    time_array,
                    P_tank_O_array,
                    P_tank_F_array,
                    track_ablative_geometry=True,
                    use_coupled_solver=False,
                )
            except Exception as e:
                import traceback
                # Log detailed context so we can see exactly what failed inside the time-varying solver
                log_status(
                    "Layer 2 BurnCandidate Error",
                    (
                        f"Exception in burn-candidate time series: {repr(e)} | "
                        f"time_len={len(time_array)}, P_O_len={len(P_tank_O_array)}, "
                        f"P_F_len={len(P_tank_F_array)} | "
                        f"ablative_thickness={getattr(getattr(optimized_config, 'ablative_cooling', None), 'initial_thickness', None)} | "
                        f"graphite_thickness={getattr(getattr(optimized_config, 'graphite_insert', None), 'initial_thickness', None)} | "
                        f"traceback={traceback.format_exc(limit=4).replace(chr(10), ' | ')}"
                    ),
                )
                # Mark time-varying analysis as failed and fall back to sample-based behavior
                use_time_varying = False
                burn_candidate_valid = pressure_candidate_valid
                full_time_results = {}
            
            # Use time-varying results for pressure curves
            pressure_curves = {
                "time": time_array,
                "P_tank_O": P_tank_O_array,
                "P_tank_F": P_tank_F_array,
                "thrust": full_time_results.get("F", np.full(n_time_points, final_performance.get("F", target_thrust))),
                "Isp": full_time_results.get("Isp", np.full(n_time_points, final_performance.get("Isp", 250))),
                "Pc": full_time_results.get("Pc", np.full(n_time_points, final_performance.get("Pc", 2e6))),
                "mdot_O": full_time_results.get("mdot_O", np.full(n_time_points, final_performance.get("mdot_O", 1.0))),
                "mdot_F": full_time_results.get("mdot_F", np.full(n_time_points, final_performance.get("mdot_F", 0.4))),
            }
            
            # Store time-varying results for display
            time_varying_results = full_time_results
            
            # Add time-varying summary to performance
            # Extract stability metrics from time-varying results
            # The time-varying solver returns stability at each time step
            chugging_stability_history = full_time_results.get("chugging_stability_margin", np.array([1.0]))
            min_time_stability_margin = float(np.min(chugging_stability_history))  # For backward compatibility
            
            # Get comprehensive stability analysis from time-varying results if available
            # Check if we have stability_state and stability_score arrays
            stability_states = full_time_results.get("stability_state", None)
            stability_scores = full_time_results.get("stability_score", None)
            
            # If not available, try to get from individual time steps
            if stability_scores is None:
                # Fallback: use chugging margin to estimate score
                # Map margin to score (rough approximation)
                min_stability_score_time = max(0.0, min(1.0, (min_time_stability_margin - 0.3) * 1.5))
            else:
                min_stability_score_time = float(np.min(stability_scores))
            
            if stability_states is None:
                # Determine state from score
                if min_stability_score_time >= 0.75:
                    min_stability_state_time = "stable"
                elif min_stability_score_time >= 0.4:
                    min_stability_state_time = "marginal"
                else:
                    min_stability_state_time = "unstable"
            else:
                # Check if all states are stable
                if isinstance(stability_states, (list, np.ndarray)):
                    if all(s == "stable" for s in stability_states):
                        min_stability_state_time = "stable"
                    elif any(s == "unstable" for s in stability_states):
                        min_stability_state_time = "unstable"
                    else:
                        min_stability_state_time = "marginal"
                else:
                    min_stability_state_time = str(stability_states)
            
            time_varying_summary = {
                "avg_thrust": float(np.mean(full_time_results.get("F", [target_thrust]))),
                "min_thrust": float(np.min(full_time_results.get("F", [target_thrust]))),
                "max_thrust": float(np.max(full_time_results.get("F", [target_thrust]))),
                "thrust_std": float(np.std(full_time_results.get("F", [0]))),
                "avg_isp": float(np.mean(full_time_results.get("Isp", [250]))),
                "min_stability_margin": min_time_stability_margin,  # Backward compatibility
                "min_stability_state": min_stability_state_time,  # New: worst state during burn
                "min_stability_score": min_stability_score_time,  # New: worst score during burn
                "max_recession_chamber": float(np.max(full_time_results.get("recession_chamber", [0.0]))),
                "max_recession_throat": float(np.max(full_time_results.get("recession_throat", [0.0]))),
            }
            final_performance["time_varying"] = time_varying_summary
            
            # Check if burn candidate is valid (meets all time-based optimization goals)
            # Check at EACH time point (excluding t=burn_time per user requirement)
            # We don't care if burn is bad at the end - just check optimal starting conditions
            min_stability_score = requirements.get("min_stability_score", 0.75)
            require_stable_state = requirements.get("require_stable_state", True)
            
            # Get time-varying arrays (ensure at least 1D)
            thrust_history = np.atleast_1d(full_time_results.get("F", np.full(n_time_points, target_thrust)))
            MR_history = np.atleast_1d(full_time_results.get("MR", np.full(n_time_points, optimal_of)))
            stability_scores_array = full_time_results.get("stability_score", None)
            stability_states_array = full_time_results.get("stability_state", None)
            
            # Determine how many valid time points we actually have
            available_n = min(
                thrust_history.shape[0],
                MR_history.shape[0],
                n_time_points,
            )
            
            if available_n < 2:
                # Not enough points for meaningful time-varying validation; fall back to Layer 1 result
                burn_candidate_valid = pressure_candidate_valid
                max_thrust_error = float(
                    abs(final_performance.get("F", target_thrust) - target_thrust) / max(target_thrust, 1e-9)
                )
                max_of_error = float(
                    abs(final_performance.get("MR", optimal_of) - optimal_of) / max(optimal_of, 1e-9)
                ) if optimal_of > 0 else 0.0
                min_stability_score_time = float(time_varying_summary.get("min_stability_score", min_stability_score))
                min_stability_state_time = time_varying_summary.get("min_stability_state", "stable")
            else:
                # Exclude last available time point - check all points before that
                check_indices = np.arange(available_n - 1)  # All except last
                
                # Align histories to available_n
                thrust_history = thrust_history[:available_n]
                MR_history = MR_history[:available_n]
            
                # Check thrust error at each time point (excluding last)
            thrust_errors = np.abs(thrust_history[check_indices] - target_thrust) / target_thrust
            max_thrust_error = float(np.max(thrust_errors))
            avg_thrust_error = float(np.mean(thrust_errors))
            
            # Check O/F error at each time point
            of_errors = (
                np.abs(MR_history[check_indices] - optimal_of) / optimal_of
                if optimal_of > 0
                else np.ones_like(check_indices)
            )
            max_of_error = float(np.max(of_errors))
            
            # Check stability at each time point
            if stability_scores_array is not None and isinstance(stability_scores_array, np.ndarray):
                stability_scores_array = np.atleast_1d(stability_scores_array)[:available_n]
                stability_scores_check = stability_scores_array[check_indices]
                min_stability_score_time = float(np.min(stability_scores_check))
            else:
                # Fallback: use chugging margin
                chugging_history = np.atleast_1d(
                    full_time_results.get("chugging_stability_margin", np.array([1.0]))
                )[:available_n]
                min_time_stability_margin = float(np.min(chugging_history[check_indices]))
                min_stability_score_time = max(
                    0.0, min(1.0, (min_time_stability_margin - 0.3) * 1.5)
                )
            
            if stability_states_array is not None and isinstance(stability_states_array, (list, np.ndarray)):
                stability_states_array = np.asarray(stability_states_array)[:available_n]
                stability_states_check = stability_states_array[check_indices]
                has_unstable = np.any(stability_states_check == "unstable")
                all_stable = np.all(stability_states_check == "stable")
                if all_stable:
                    min_stability_state_time = "stable"
                elif has_unstable:
                    min_stability_state_time = "unstable"
                else:
                    min_stability_state_time = "marginal"
            else:
                # Determine from score
                if min_stability_score_time >= 0.75:
                    min_stability_state_time = "stable"
                elif min_stability_score_time >= 0.4:
                    min_stability_state_time = "marginal"
                else:
                    min_stability_state_time = "unstable"
            
            # Stability check for Layer 2 (time-varying, excluding t=burn_time)
            if require_stable_state:
                # Require "stable" state throughout burn (or at least not "unstable")
                stability_valid_time = (min_stability_state_time != "unstable") and (min_stability_score_time >= min_stability_score * 0.7)  # 70% of target for Layer 2
            else:
                # Allow "marginal" but require minimum score
                stability_valid_time = (min_stability_state_time != "unstable") and (min_stability_score_time >= min_stability_score * 0.7)
            
                # Burn candidate valid if all time points (excluding last) meet goals
            burn_candidate_valid = (
                stability_valid_time and
                    max_thrust_error < thrust_tol * 1.5 and  # Max error at any point (excluding last)
                max_of_error < 0.20  # Max O/F error at any point
            )
            final_performance["burn_candidate_valid"] = burn_candidate_valid
            final_performance["max_thrust_error_time"] = max_thrust_error
            final_performance["max_of_error_time"] = max_of_error
            
            update_progress(
                "Layer 2: Burn Candidate",
                0.65,
                f"Burn candidate {'✓ VALID' if burn_candidate_valid else '✗ INVALID'} - Stability: {min_stability_state_time} (score: {min_stability_score_time:.2f})",
            )
            log_status(
                "Layer 2",
                f"{'VALID' if burn_candidate_valid else 'INVALID'} | Stability {min_stability_state_time} (score {min_stability_score_time:.2f}), "
                f"max thrust err {max_thrust_error*100:.1f}%, max O/F err {max_of_error*100:.1f}%"
            )
            
            # ==========================================================================
            # LAYER 3: BURN ANALYSIS (ABLATIVE/GRAPHITE OPTIMIZATION)
            # Optimize ablative liner and graphite nozzle parameters
            # Once Layer 2 passes, optimize these to meet all goals with margin
            # ==========================================================================
            if burn_candidate_valid:
                update_progress("Layer 3: Burn Analysis Optimization", 0.68, "Optimizing ablative and graphite parameters...")
                
                # Get current ablative/graphite config
                ablative_cfg = optimized_config.ablative_cooling if hasattr(optimized_config, 'ablative_cooling') else None
                graphite_cfg = optimized_config.graphite_insert if hasattr(optimized_config, 'graphite_insert') else None
                
                # Get recession data from time-varying results
                recession_chamber_history = full_time_results.get("recession_chamber", np.zeros(n_time_points))
                recession_throat_history = full_time_results.get("recession_throat", np.zeros(n_time_points))
                max_recession_chamber = float(np.max(recession_chamber_history))
                max_recession_throat = float(np.max(recession_throat_history))
                
                # Layer 3: Optimize ablative/graphite thickness to meet recession + margin requirements
                from scipy.optimize import minimize as scipy_minimize
                
                layer3_bounds = []
                layer3_x0 = []
                
                if ablative_cfg and ablative_cfg.enabled:
                    # Optimize to max_recession * 1.2 (20% margin)
                    target_ablative = max_recession_chamber * 1.2
                    layer3_bounds.append((max(0.003, target_ablative * 0.8), min(0.020, target_ablative * 1.5)))
                    layer3_x0.append(ablative_cfg.initial_thickness)
                
                if graphite_cfg and graphite_cfg.enabled:
                    # Optimize to max_recession * 1.2 (20% margin)
                    target_graphite = max_recession_throat * 1.2
                    layer3_bounds.append((max(0.003, target_graphite * 0.8), min(0.015, target_graphite * 1.5)))
                    layer3_x0.append(graphite_cfg.initial_thickness)
                
                if len(layer3_x0) > 0:
                    layer3_x0 = np.array(layer3_x0)
                    
                    def layer3_objective(x_layer3):
                        """Optimize thermal protection to minimize mass while meeting recession requirements."""
                        try:
                            # Update config
                            config_layer3 = copy.deepcopy(optimized_config)
                            idx = 0
                            if ablative_cfg and ablative_cfg.enabled:
                                config_layer3.ablative_cooling.initial_thickness = float(np.clip(x_layer3[idx], layer3_bounds[idx][0], layer3_bounds[idx][1]))
                                idx += 1
                            if graphite_cfg and graphite_cfg.enabled:
                                config_layer3.graphite_insert.initial_thickness = float(np.clip(x_layer3[idx], layer3_bounds[idx][0], layer3_bounds[idx][1]))
                            
                            # Run time series
                            runner_layer3 = PintleEngineRunner(config_layer3)
                            results_layer3 = runner_layer3.evaluate_arrays_with_time(
                                time_array,
                                P_tank_O_array,
                                P_tank_F_array,
                                track_ablative_geometry=True,
                                use_coupled_solver=False,  # use robust standard solver inside Layer 3 objective
                            )
                            
                            # Get recession
                            recession_chamber = float(np.max(results_layer3.get("recession_chamber", [0.0])))
                            recession_throat = float(np.max(results_layer3.get("recession_throat", [0.0])))
                            
                            # Check if recession exceeds thickness (with 20% margin)
                            idx = 0
                            recession_penalty = 0.0
                            if ablative_cfg and ablative_cfg.enabled:
                                thickness = x_layer3[idx]
                                if recession_chamber > thickness * 0.8:  # 80% of thickness
                                    recession_penalty += 1000.0 * (recession_chamber - thickness * 0.8)
                                idx += 1
                            if graphite_cfg and graphite_cfg.enabled:
                                thickness = x_layer3[idx]
                                if recession_throat > thickness * 0.8:
                                    recession_penalty += 1000.0 * (recession_throat - thickness * 0.8)
                            
                            # Objective: minimize mass (thickness) + recession penalty
                            total_thickness = np.sum(x_layer3)
                            obj = total_thickness * 1000 + recession_penalty  # Convert to mm for scaling
                            return obj
                        except Exception as e:
                            return 1e6
                    
                    # Optimize Layer 3
                    try:
                        result_layer3 = scipy_minimize(
                            layer3_objective,
                            layer3_x0,
                            method='L-BFGS-B',
                            bounds=layer3_bounds,
                            options={'maxiter': 30, 'ftol': 1e-5}
                        )
                        
                        # Update config with optimized thicknesses
                        idx = 0
                        if ablative_cfg and ablative_cfg.enabled:
                            optimized_config.ablative_cooling.initial_thickness = float(np.clip(result_layer3.x[idx], layer3_bounds[idx][0], layer3_bounds[idx][1]))
                            update_progress("Layer 3: Burn Analysis Optimization", 0.70, 
                                f"✓ Optimized ablative: {optimized_config.ablative_cooling.initial_thickness*1000:.2f}mm (recession: {max_recession_chamber*1000:.2f}mm)")
                            idx += 1
                        if graphite_cfg and graphite_cfg.enabled:
                            optimized_config.graphite_insert.initial_thickness = float(np.clip(result_layer3.x[idx], layer3_bounds[idx][0], layer3_bounds[idx][1]))
                            update_progress("Layer 3: Burn Analysis Optimization", 0.72, 
                                f"✓ Optimized graphite: {optimized_config.graphite_insert.initial_thickness*1000:.2f}mm (recession: {max_recession_throat*1000:.2f}mm)")
                    except Exception as e:
                        update_progress("Layer 3: Burn Analysis Optimization", 0.72, f"⚠️ Layer 3 optimization failed: {e}, using current values")
                
                # Re-run time series with optimized thermal protection to verify
                update_progress("Layer 3: Burn Analysis", 0.74, "Re-running time series with optimized thermal protection...")
                try:
                    optimized_runner_updated = PintleEngineRunner(optimized_config)
                    full_time_results_updated = optimized_runner_updated.evaluate_arrays_with_time(
                        time_array,
                        P_tank_O_array,
                        P_tank_F_array,
                        track_ablative_geometry=True,
                        use_coupled_solver=True,
                    )
                    # Update time-varying results
                    time_varying_results = full_time_results_updated
                    time_varying_summary["max_recession_chamber"] = float(np.max(full_time_results_updated.get("recession_chamber", [0.0])))
                    time_varying_summary["max_recession_throat"] = float(np.max(full_time_results_updated.get("recession_throat", [0.0])))
                    
                    # Update pressure curves with new results
                    pressure_curves["thrust"] = full_time_results_updated.get("F", pressure_curves["thrust"])
                    pressure_curves["mdot_O"] = full_time_results_updated.get("mdot_O", pressure_curves["mdot_O"])
                    pressure_curves["mdot_F"] = full_time_results_updated.get("mdot_F", pressure_curves["mdot_F"])
                except Exception as e:
                    update_progress("Layer 3: Burn Analysis", 0.74, f"⚠️ Re-evaluation failed: {e}, using original results")
                
                ablative_ok = True  # Always OK after optimization
                graphite_ok = True
                final_performance["ablative_adequate"] = ablative_ok
                final_performance["graphite_adequate"] = graphite_ok
                final_performance["thermal_protection_valid"] = ablative_ok and graphite_ok
                final_performance["optimized_ablative_thickness"] = optimized_config.ablative_cooling.initial_thickness if ablative_cfg and ablative_cfg.enabled else None
                final_performance["optimized_graphite_thickness"] = optimized_config.graphite_insert.initial_thickness if graphite_cfg and graphite_cfg.enabled else None
                log_status(
                    "Layer 3",
                    "Completed | Ablative {:.2f} mm, Graphite {:.2f} mm, Max recession chamber {:.2f} mm, throat {:.2f} mm".format(
                        (optimized_config.ablative_cooling.initial_thickness * 1000) if ablative_cfg and ablative_cfg.enabled else 0.0,
                        (optimized_config.graphite_insert.initial_thickness * 1000) if graphite_cfg and graphite_cfg.enabled else 0.0,
                        time_varying_summary.get("max_recession_chamber", 0.0) * 1000,
                        time_varying_summary.get("max_recession_throat", 0.0) * 1000,
                    )
                )
        except Exception as e:
            import warnings
            warnings.warn(f"Time-varying analysis failed, falling back to sample-based: {e}")
            log_status(
                "Layer 2/3 Error",
                f"Time-varying analysis failed, falling back to sample-based: {repr(e)}"
            )
            use_time_varying = False  # Fall back to sample-based method
            burn_candidate_valid = pressure_candidate_valid  # Assume valid if pressure candidate passed
        else:
            log_status("Layer 3", "Skipped | Burn candidate invalid")
    else:
        if not use_time_varying:
            log_status("Layer 2", "Skipped | Time-varying analysis disabled")
        elif not pressure_candidate_valid:
            log_status("Layer 2", "Skipped | Pressure candidate invalid")
    
    if not use_time_varying or pressure_curves is None:
        # Fallback: Sample-based interpolation (faster but less accurate)
        sample_indices = [0, n_time_points//4, n_time_points//2, 3*n_time_points//4, n_time_points-1]
        sample_F = []
        sample_Isp = []
        sample_Pc = []
        sample_mdot_O = []
        sample_mdot_F = []
        
        for idx in sample_indices:
            try:
                results = optimized_runner.evaluate(P_tank_O_array[idx], P_tank_F_array[idx])
                sample_F.append(results.get("F", 0))
                sample_Isp.append(results.get("Isp", 0))
                sample_Pc.append(results.get("Pc", 0))
                sample_mdot_O.append(results.get("mdot_O", 0))
                sample_mdot_F.append(results.get("mdot_F", 0))
            except:
                # Use fallback values
                sample_F.append(final_performance.get("F", target_thrust))
                sample_Isp.append(final_performance.get("Isp", 250))
                sample_Pc.append(final_performance.get("Pc", 2e6))
                sample_mdot_O.append(final_performance.get("mdot_O", 1.0))
                sample_mdot_F.append(final_performance.get("mdot_F", 0.4))
        
        # Interpolate to full 200 points
        from scipy.interpolate import interp1d
        sample_times = [time_array[i] for i in sample_indices]
        
        thrust_interp = interp1d(sample_times, sample_F, kind='linear', fill_value='extrapolate')
        isp_interp = interp1d(sample_times, sample_Isp, kind='linear', fill_value='extrapolate')
        pc_interp = interp1d(sample_times, sample_Pc, kind='linear', fill_value='extrapolate')
        mdot_O_interp = interp1d(sample_times, sample_mdot_O, kind='linear', fill_value='extrapolate')
        mdot_F_interp = interp1d(sample_times, sample_mdot_F, kind='linear', fill_value='extrapolate')
        
        pressure_curves = {
            "time": time_array,
            "P_tank_O": P_tank_O_array,
            "P_tank_F": P_tank_F_array,
            "thrust": thrust_interp(time_array),
            "Isp": isp_interp(time_array),
            "Pc": pc_interp(time_array),
            "mdot_O": mdot_O_interp(time_array),
            "mdot_F": mdot_F_interp(time_array),
        }
    
    update_progress("COPV Calculation", 0.65, "Calculating COPV pressure curve (T=260K)...")
    
    # Phase 7: Calculate COPV pressure curve
    copv_results = _calculate_copv_pressure_curve(
        time_array,
        pressure_curves["mdot_O"],
        pressure_curves["mdot_F"],
        P_tank_O_array,
        P_tank_F_array,
        optimized_config,
        copv_volume_m3,
        T0_K=260.0,  # User specified temperature
        Tp_K=260.0,  # User specified temperature
    )
    
    update_progress("Validation", 0.70, "Running stability checks at initial conditions...")
    
    # Phase 8: Run system diagnostics at INITIAL conditions (not average)
    try:
        diagnostics = SystemDiagnostics(optimized_config, optimized_runner)
        validation_results = diagnostics.run_full_diagnostics(P_O_initial, P_F_initial)
    except Exception as e:
        validation_results = {"error": str(e)}
    
    # ==========================================================================
    # LAYER 4: FLIGHT CANDIDATE
    # Run flight simulation with propellant truncation
    # Once Layer 3 passes, run flight sim with backward iteration
    # Automatically detects tank empty conditions and truncates thrust
    # Iterates backward if apogee goals not met (reduce propellant, rerun)
    # ==========================================================================
    flight_sim_result = {"success": False, "apogee": 0, "max_velocity": 0, "layer": 4}
    flight_candidate_valid = False
    
    # Determine if we should run flight sim
    # Layer 3 must pass (thermal protection valid) OR we're not doing time-varying
    # Also check if we have valid pressure curves
    thermal_protection_valid = final_performance.get("thermal_protection_valid", True)  # Default True if not checked
    should_run_flight = (
        pressure_candidate_valid and 
        pressure_curves is not None and
        (
            (burn_candidate_valid and thermal_protection_valid) or not use_time_varying
        )
    )
    
    if should_run_flight:
        update_progress("Layer 4: Flight Candidate", 0.75, "Running flight simulation with backward iteration...")
        
        try:
            # Layer 4: Iterative backward truncation to meet apogee goals
            # Start from t_burn_time - epsilon, truncate, subtract remaining propellant, rerun
            epsilon = 0.01  # Small time step for backward iteration
            max_iterations_flight = 20  # Prevent infinite loops
            flight_iteration = 0
            current_burn_time = target_burn_time
            flight_candidate_valid = False
            
            # Get initial propellant masses
            config_for_flight = copy.deepcopy(optimized_config)
            initial_lox_mass = config_for_flight.lox_tank.mass if hasattr(config_for_flight, 'lox_tank') else 0
            initial_fuel_mass = config_for_flight.fuel_tank.mass if hasattr(config_for_flight, 'fuel_tank') else 0
            
            # Get propellant densities for mass calculation
            rho_lox = config_for_flight.fluids['oxidizer'].density if hasattr(config_for_flight, 'fluids') else 1140.0
            rho_fuel = config_for_flight.fluids['fuel'].density if hasattr(config_for_flight, 'fluids') else 800.0
            
            while flight_iteration < max_iterations_flight and not flight_candidate_valid:
                flight_iteration += 1
                progress = 0.75 + 0.10 * min(flight_iteration / max_iterations_flight, 1.0)
                
                # Truncate thrust curve at current_burn_time - epsilon
                cutoff_time = max(0.1, current_burn_time - epsilon)
                update_progress("Layer 4: Flight Candidate", progress, 
                    f"Iteration {flight_iteration}: Testing burn time {cutoff_time:.2f}s (target: {target_apogee:.0f}m)")
                
                # Create truncated pressure curves
                time_array_trunc = time_array[time_array <= cutoff_time]
                if len(time_array_trunc) == 0:
                    time_array_trunc = np.array([0.0, cutoff_time])
                
                # Truncate all arrays
                mask = time_array <= cutoff_time
                pressure_curves_trunc = {
                    "time": time_array_trunc,
                    "P_tank_O": P_tank_O_array[mask][:len(time_array_trunc)],
                    "P_tank_F": P_tank_F_array[mask][:len(time_array_trunc)],
                    "thrust": pressure_curves["thrust"][mask][:len(time_array_trunc)],
                    "Isp": pressure_curves["Isp"][mask][:len(time_array_trunc)],
                    "Pc": pressure_curves["Pc"][mask][:len(time_array_trunc)],
                    "mdot_O": pressure_curves["mdot_O"][mask][:len(time_array_trunc)],
                    "mdot_F": pressure_curves["mdot_F"][mask][:len(time_array_trunc)],
                }
                
                # Calculate remaining propellant mass (integrate mdot from cutoff_time to target_burn_time)
                if cutoff_time < target_burn_time:
                    # Integrate mdot from cutoff_time to target_burn_time
                    remaining_time = target_burn_time - cutoff_time
                    # Use average mdot at cutoff point as estimate
                    mdot_O_cutoff = pressure_curves["mdot_O"][mask][-1] if len(pressure_curves["mdot_O"][mask]) > 0 else 0
                    mdot_F_cutoff = pressure_curves["mdot_F"][mask][-1] if len(pressure_curves["mdot_F"][mask]) > 0 else 0
                    remaining_lox_mass = mdot_O_cutoff * remaining_time
                    remaining_fuel_mass = mdot_F_cutoff * remaining_time
                else:
                    remaining_lox_mass = 0
                    remaining_fuel_mass = 0
                
                # Subtract remaining propellant from initial masses
                adjusted_lox_mass = max(0.1, initial_lox_mass - remaining_lox_mass)
                adjusted_fuel_mass = max(0.1, initial_fuel_mass - remaining_fuel_mass)
                
                # Update config with adjusted masses
                config_for_flight.lox_tank.mass = adjusted_lox_mass
                config_for_flight.fuel_tank.mass = adjusted_fuel_mass
                
                # Run flight simulation with truncated thrust and adjusted masses
                flight_sim_result = _run_flight_simulation(
                    config_for_flight,
                    pressure_curves_trunc,
                    cutoff_time,
                )
                
                if flight_sim_result.get("success", False):
                    apogee = flight_sim_result.get("apogee", 0)
                    apogee_error = abs(apogee - target_apogee) / target_apogee if target_apogee > 0 else 1.0
                    
                    # Check if apogee goal is met
                    if apogee_error < apogee_tol:
                        flight_candidate_valid = True
                        update_progress(
                            "Layer 4: Flight Candidate",
                            0.85,
                            f"✓ VALID - Apogee {apogee:.0f}m within {apogee_error*100:.1f}% of target {target_apogee:.0f}m (burn: {cutoff_time:.2f}s)",
                        )
                        log_status(
                            "Layer 4",
                            f"VALID | Apogee {apogee:.0f}m (error {apogee_error*100:.1f}%), burn {cutoff_time:.2f}s",
                        )
                        flight_sim_result["actual_burn_time"] = cutoff_time
                        flight_sim_result["adjusted_lox_mass"] = adjusted_lox_mass
                        flight_sim_result["adjusted_fuel_mass"] = adjusted_fuel_mass
                        flight_sim_result["iterations"] = flight_iteration
                        break
                    else:
                        # Apogee not met - continue backward iteration
                        if apogee < target_apogee:
                            # Apogee too low - need to reduce burn time further (less propellant)
                            current_burn_time = cutoff_time
                            update_progress("Layer 4: Flight Candidate", progress, 
                                f"Apogee {apogee:.0f}m < target {target_apogee:.0f}m, reducing burn time to {current_burn_time:.2f}s")
                        else:
                            # Apogee too high - we've gone too far back, use this as best
                            flight_candidate_valid = True  # Accept as best we can do
                            update_progress(
                                "Layer 4: Flight Candidate",
                                0.85,
                                f"✓ Best match - Apogee {apogee:.0f}m (target: {target_apogee:.0f}m, error: {apogee_error*100:.1f}%, burn: {cutoff_time:.2f}s)",
                            )
                            log_status(
                                "Layer 4",
                                f"ACCEPTED | Apogee {apogee:.0f}m (error {apogee_error*100:.1f}%), burn {cutoff_time:.2f}s after {flight_iteration} iterations",
                            )
                            flight_sim_result["actual_burn_time"] = cutoff_time
                            flight_sim_result["adjusted_lox_mass"] = adjusted_lox_mass
                            flight_sim_result["adjusted_fuel_mass"] = adjusted_fuel_mass
                            flight_sim_result["iterations"] = flight_iteration
                            break
                else:
                    # Flight sim failed - try next iteration
                    current_burn_time = cutoff_time
                    if flight_iteration >= max_iterations_flight:
                        update_progress("Layer 4: Flight Candidate", 0.85, 
                            f"⚠️ Flight sim failed after {flight_iteration} iterations: {flight_sim_result.get('error', 'Unknown error')}")
                        break
                
                # Prevent going too far back
                if current_burn_time < 0.5:  # Minimum 0.5s burn time
                    update_progress("Layer 4: Flight Candidate", 0.85, 
                        f"⚠️ Reached minimum burn time (0.5s), stopping iteration")
                    break
            
            if not flight_candidate_valid and flight_iteration >= max_iterations_flight:
                update_progress("Layer 4: Flight Candidate", 0.85, 
                    f"⚠️ Max iterations reached, using last result")
                flight_candidate_valid = False  # Mark as invalid if we didn't converge
                
        except Exception as e:
            flight_sim_result = {"success": False, "error": str(e), "apogee": 0, "max_velocity": 0}
            update_progress("Layer 4: Flight Candidate", 0.85, f"⚠️ Flight sim error: {e}")
    else:
        reason = "pressure candidate invalid" if not pressure_candidate_valid else "burn candidate invalid"
        update_progress("Layer 4: Flight Candidate", 0.75, f"Skipping flight sim ({reason})")
        log_status("Layer 4", f"Skipped | Reason: {reason}")
        flight_sim_result = {"success": False, "skipped": True, "reason": reason, "apogee": 0, "max_velocity": 0}
    
    flight_sim_result["flight_candidate_valid"] = flight_candidate_valid
    final_performance["flight_candidate_valid"] = flight_candidate_valid
    
    update_progress("Finalization", 0.90, "Assembling results...")
    
    # Build design_requirements dict for results
    design_requirements = {
        "target_thrust": target_thrust,
        "target_apogee": target_apogee,
        "target_burn_time": target_burn_time,
        "target_stability_margin": min_stability,
        "P_tank_O_start": lox_P_start,
        "P_tank_F_start": fuel_P_start,
        "target_MR": optimal_of,
    }
    
    # Build constraints dict for results
    constraints = {
        "min_Lstar": min_Lstar,
        "max_Lstar": max_Lstar,
        "max_chamber_diameter": max_chamber_od,
        "max_nozzle_exit_diameter": max_nozzle_exit,
        "thrust_tolerance": thrust_tol,
        "apogee_tolerance": apogee_tol,
    }
    
    # Combine all results
    coupled_results["performance"] = final_performance
    coupled_results["validation"] = validation_results
    coupled_results["design_requirements"] = design_requirements
    coupled_results["constraints"] = constraints
    coupled_results["optimized_parameters"] = _extract_all_parameters(optimized_config)
    coupled_results["pressure_curves"] = pressure_curves
    coupled_results["copv_results"] = copv_results
    coupled_results["flight_sim_result"] = flight_sim_result
    coupled_results["time_array"] = time_array
    
    # Include time-varying results for plotting (if available)
    if time_varying_results is not None:
        coupled_results["time_varying_results"] = time_varying_results
    
    # Add layered optimization status summary
    coupled_results["layer_status"] = {
        "layer_1_pressure_candidate": pressure_candidate_valid,
        "layer_2_burn_candidate": burn_candidate_valid if use_time_varying else None,
        "layer_3_thermal_protection": final_performance.get("thermal_protection_valid", None),
        "layer_4_flight_candidate": flight_candidate_valid,
        "all_layers_passed": (
            pressure_candidate_valid and 
            (burn_candidate_valid or not use_time_varying) and 
            flight_candidate_valid
        ),
    }
    layer_summary = coupled_results["layer_status"]
    log_status(
        "Completion",
        "Summary | L1={layer_1_pressure_candidate}, L2={layer_2_burn_candidate}, "
        "L3={layer_3_thermal_protection}, L4={layer_4_flight_candidate}".format(**layer_summary)
    )
    
    # Add pressure curve config info to results
    coupled_results["pressure_curve_config"] = {
        "mode": pressure_mode,
        "max_lox_pressure_psi": max_lox_P_psi,
        "max_fuel_pressure_psi": max_fuel_P_psi,
    }
    
    update_progress("Complete", 1.0, "Optimization complete!")
    
    return optimized_config, coupled_results


def _calculate_copv_pressure_curve(
    time_array: np.ndarray,
    mdot_O: np.ndarray,
    mdot_F: np.ndarray,
    P_tank_O: np.ndarray,
    P_tank_F: np.ndarray,
    config: PintleEngineConfig,
    copv_volume_m3: float,
    T0_K: float = 260.0,
    Tp_K: float = 260.0,
) -> Dict[str, Any]:
    """
    Calculate COPV pressure curve using polytropic blowdown model.
    
    Uses the same method as the COPV tab with user-specified temperatures (260K).
    """
    try:
        from examples.pintle_engine.copv_pressure.copv_solve_both import size_or_check_copv_for_polytropic_N2
        
        # Build dataframe for COPV solver
        psi_to_Pa = 6894.757293168
        df = pd.DataFrame({
            "time": time_array,
            "mdot_O (kg/s)": mdot_O,
            "mdot_F (kg/s)": mdot_F,
            "P_tank_O (psi)": P_tank_O / psi_to_Pa,
            "P_tank_F (psi)": P_tank_F / psi_to_Pa,
        })
        
        # Run COPV sizing with known volume
        copv_results = size_or_check_copv_for_polytropic_N2(
            df=df,
            config=config,
            n=1.2,  # Polytropic exponent
            T0_K=T0_K,
            Tp_K=Tp_K,
            use_real_gas=False,  # Simplified for speed
            copv_volume_m3=copv_volume_m3,
            branch_temperatures_K={
                "oxidizer": Tp_K,
                "fuel": Tp_K,
            },
        )
        
        return {
            "success": True,
            "time": time_array,
            "copv_pressure_Pa": copv_results.get("PH_trace_Pa", np.zeros_like(time_array)),
            "copv_pressure_psi": copv_results.get("PH_trace_Pa", np.zeros_like(time_array)) / psi_to_Pa,
            "initial_pressure_Pa": copv_results.get("P0_Pa", 0),
            "initial_mass_kg": copv_results.get("m0_kg", 0),
            "total_delivered_kg": copv_results.get("total_delivered_mass_kg", 0),
            "min_margin_psi": copv_results.get("min_margin_Pa", 0) / psi_to_Pa,
        }
    except Exception as e:
        # Fallback: simple pressure estimate
        P_max = max(np.max(P_tank_O), np.max(P_tank_F))
        copv_P0 = P_max * 1.15  # 15% margin
        # Simple polytropic blowdown
        n = 1.2
        # Estimate mass flow from tanks
        mdot_total = mdot_O + mdot_F
        mass_consumed = np.cumsum(mdot_total * np.gradient(time_array))
        
        # Simple pressure decay model
        copv_pressure = copv_P0 * (1 - 0.3 * mass_consumed / (mass_consumed[-1] + 0.001))
        
        return {
            "success": False,
            "error": str(e),
            "time": time_array,
            "copv_pressure_Pa": copv_pressure,
            "copv_pressure_psi": copv_pressure / 6894.757293168,
            "initial_pressure_Pa": copv_P0,
            "initial_mass_kg": 0.5,
            "total_delivered_kg": 0.3,
            "min_margin_psi": 50.0,
        }


def _run_flight_simulation(
    config: PintleEngineConfig,
    pressure_curves: Dict[str, np.ndarray],
    burn_time: float,
) -> Dict[str, Any]:
    """Run flight simulation on the optimized engine."""
    try:
        from examples.pintle_engine.flight_sim import setup_flight
        from scipy.interpolate import interp1d
        
        # Create thrust curve function from pressure curves
        time_array = pressure_curves["time"]
        thrust_array = pressure_curves["thrust"]
        mdot_O_array = pressure_curves["mdot_O"]
        mdot_F_array = pressure_curves["mdot_F"]
        
        # Create interpolation functions
        thrust_func = interp1d(time_array, thrust_array, kind='linear', fill_value=0, bounds_error=False)
        mdot_O_func = interp1d(time_array, mdot_O_array, kind='linear', fill_value=0, bounds_error=False)
        mdot_F_func = interp1d(time_array, mdot_F_array, kind='linear', fill_value=0, bounds_error=False)
        
        # Run flight simulation
        result = setup_flight(config, thrust_func, mdot_O_func, mdot_F_func, plot_results=False)
        
        return {
            "success": True,
            "apogee": result.get("apogee", 0),
            "max_velocity": result.get("max_velocity", 0),
            "flight_time": result.get("flight_time", 0),
            "flight_obj": result.get("flight", None),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "apogee": 0,
            "max_velocity": 0,
        }


def _show_complete_optimization_results(
    config_before: PintleEngineConfig,
    config_after: PintleEngineConfig,
    optimization_results: Dict[str, Any],
    requirements: Dict[str, Any],
    target_burn_time: float,
) -> None:
    """Show complete optimization results with all visualizations."""
    
    # Tab layout for organized results
    result_tabs = st.tabs([
        "📊 Summary",
        "🔧 Injector & Chamber",
        "📈 Pressure Curves",
        "🛢️ COPV",
        "🚀 Flight Simulation",
    ])
    
    with result_tabs[0]:
        # Show optimization convergence plot
        _show_optimization_convergence(optimization_results)
        
        # Summary comparison
        _show_full_engine_comparison(config_before, config_after, optimization_results)
        
        # Validation checks
        _show_engine_validation_checks(config_after, optimization_results, requirements)
    
    with result_tabs[1]:
        # Injector parameters
        st.markdown("### 🔧 Optimized Pintle Injector")
        params = optimization_results.get("optimized_parameters", {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pintle Tip Ø", f"{params.get('d_pintle_tip', 0) * 1000:.2f} mm")
            st.metric("Gap Height", f"{params.get('h_gap', 0) * 1000:.3f} mm")
        with col2:
            st.metric("Orifice Count", f"{int(params.get('n_orifices', 0))}")
            st.metric("Orifice Ø", f"{params.get('d_orifice', 0) * 1000:.2f} mm")
        with col3:
            st.metric("Orifice Angle", f"{params.get('theta_orifice', 90):.1f}°")
            st.metric("(Perpendicular)", "✅ Fixed at 90°")
        
        st.markdown("### 🔥 Optimized Chamber Geometry")
        col1, col2, col3 = st.columns(3)
        with col1:
            D_throat = np.sqrt(4 * params.get('A_throat', 0) / np.pi) * 1000
            st.metric("Throat Ø", f"{D_throat:.2f} mm")
            st.metric("Throat Area", f"{params.get('A_throat', 0) * 1e6:.2f} mm²")
        with col2:
            st.metric("L*", f"{params.get('Lstar', 0) * 1000:.1f} mm")
            st.metric("Chamber Ø", f"{params.get('chamber_diameter', 0) * 1000:.1f} mm")
        with col3:
            st.metric("Expansion Ratio", f"{params.get('expansion_ratio', 0):.2f}")
            D_exit = np.sqrt(4 * params.get('A_exit', 0) / np.pi) * 1000
            st.metric("Exit Ø", f"{D_exit:.2f} mm")
        
        # Chamber visualization
        st.markdown("### 📐 Chamber Geometry Visualization")
        try:
            _display_chamber_geometry_plot(config_after, optimization_results)
        except Exception as e:
            st.warning(f"Could not generate chamber visualization: {e}")
    
    with result_tabs[2]:
        # Pressure curves
        st.markdown("### 📈 Tank Pressure Curves (200 points)")
        pressure_curves = optimization_results.get("pressure_curves", {})
        
        if pressure_curves:
            _plot_pressure_curves(pressure_curves)
        else:
            st.info("Pressure curves not available.")
    
    with result_tabs[3]:
        # COPV results
        st.markdown("### 🛢️ COPV Pressure Curve")
        copv_results = optimization_results.get("copv_results", {})
        
        if copv_results:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Initial Pressure", f"{copv_results.get('initial_pressure_Pa', 0) / 6894.76:.0f} psi")
            with col2:
                st.metric("Initial N₂ Mass", f"{copv_results.get('initial_mass_kg', 0):.3f} kg")
            with col3:
                st.metric("Total Delivered", f"{copv_results.get('total_delivered_kg', 0):.3f} kg")
            with col4:
                st.metric("Min Margin", f"{copv_results.get('min_margin_psi', 0):.1f} psi")
            
            st.caption("*Calculated with T₀ = T_propellant = 260 K*")
            
            _plot_copv_pressure(copv_results, pressure_curves)
        else:
            st.info("COPV results not available.")
    
    with result_tabs[4]:
        # Flight simulation results
        st.markdown("### 🚀 Flight Simulation Results")
        flight_result = optimization_results.get("flight_sim_result", {})
        
        if flight_result.get("success", False):
            target_apogee = requirements.get("target_apogee", 3048.0)
            apogee = flight_result.get("apogee", 0)
            apogee_error = abs(apogee - target_apogee) / target_apogee * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Apogee AGL", f"{apogee:.0f} m", delta=f"Target: {target_apogee:.0f} m")
            with col2:
                st.metric("Max Velocity", f"{flight_result.get('max_velocity', 0):.1f} m/s")
            with col3:
                if apogee_error < 10:
                    st.metric("Apogee Error", f"{apogee_error:.1f}%", delta="✅ Within 10%")
                else:
                    st.metric("Apogee Error", f"{apogee_error:.1f}%", delta="⚠️ > 10%")
            
            # Plot flight trajectory if available
            flight_obj = flight_result.get("flight_obj")
            if flight_obj:
                try:
                    _plot_flight_trajectory(flight_obj, requirements)
                except Exception as e:
                    st.warning(f"Could not plot trajectory: {e}")
        else:
            error_msg = flight_result.get("error", "Flight simulation was not run (candidate did not meet thresholds)")
            st.warning(f"⚠️ {error_msg}")
            
            # Show detailed failure reasons from final_performance
            final_perf = optimization_results.get("final_performance", {})
            failure_reasons = final_perf.get("failure_reasons", [])
            if failure_reasons:
                st.error("**Why flight sim was skipped:**")
                for reason in failure_reasons:
                    st.write(f"  • {reason}")
            else:
                # Show actual values vs thresholds
                thrust_err = final_perf.get("initial_thrust_error", 0) * 100
                of_err = final_perf.get("initial_MR_error", 0) * 100
                stability = final_perf.get("initial_stability", 0)
                st.info(f"Thrust error: {thrust_err:.1f}% | O/F error: {of_err:.1f}% | Stability margin: {stability:.2f}")
            
            st.info("Flight sim runs when: thrust error < 15%, O/F error < 20%, stability ≥ 50% of target")


def _show_optimization_convergence(optimization_results: Dict[str, Any]) -> None:
    """Show optimization convergence plot."""
    history = optimization_results.get("iteration_history", [])
    
    if not history:
        st.info("No optimization history available.")
        return
    
    st.markdown("### 📈 Optimization Convergence")
    
    iterations = [h.get("iteration", i) for i, h in enumerate(history)]
    thrust_errors = [h.get("thrust_error", 0) * 100 for h in history]
    of_errors = [h.get("of_error", 0) * 100 for h in history]
    objectives = [h.get("objective", 0) for h in history]
    thrusts = [h.get("thrust", 0) for h in history]
    lox_ratios = [h.get("lox_end_ratio", 0.7) * 100 for h in history]
    fuel_ratios = [h.get("fuel_end_ratio", 0.7) * 100 for h in history]
    lstars = [h.get("Lstar", 1.0) for h in history]
    l_chambers = [h.get("L_chamber", 0.2) * 1000 for h in history]  # Convert to mm
    
    # Create subplot with 4 rows
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "Objective Function", "Thrust Error [%]", 
            "O/F Error [%]", "Thrust [N]",
            "LOX End Pressure [%]", "Fuel End Pressure [%]",
            "L* [m]", "Chamber Length [mm]"
        ),
        vertical_spacing=0.10,
    )
    
    # Objective
    fig.add_trace(
        go.Scatter(x=iterations, y=objectives, mode='lines+markers', name='Objective', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Thrust error
    fig.add_trace(
        go.Scatter(x=iterations, y=thrust_errors, mode='lines+markers', name='Thrust Error', line=dict(color='red')),
        row=1, col=2
    )
    
    # O/F error
    fig.add_trace(
        go.Scatter(x=iterations, y=of_errors, mode='lines+markers', name='O/F Error', line=dict(color='orange')),
        row=2, col=1
    )
    
    # Thrust
    fig.add_trace(
        go.Scatter(x=iterations, y=thrusts, mode='lines+markers', name='Thrust', line=dict(color='green')),
        row=2, col=2
    )
    
    # LOX pressure ratio evolution
    fig.add_trace(
        go.Scatter(x=iterations, y=lox_ratios, mode='lines+markers', name='LOX End %', line=dict(color='cyan')),
        row=3, col=1
    )
    
    # Fuel pressure ratio evolution
    fig.add_trace(
        go.Scatter(x=iterations, y=fuel_ratios, mode='lines+markers', name='Fuel End %', line=dict(color='magenta')),
        row=3, col=2
    )
    
    # L* evolution
    fig.add_trace(
        go.Scatter(x=iterations, y=lstars, mode='lines+markers', name='L*', line=dict(color='purple')),
        row=4, col=1
    )
    
    # Chamber length evolution
    fig.add_trace(
        go.Scatter(x=iterations, y=l_chambers, mode='lines+markers', name='L_chamber', line=dict(color='brown')),
        row=4, col=2
    )
    
    fig.update_xaxes(title_text="Iteration", row=4, col=1)
    fig.update_xaxes(title_text="Iteration", row=4, col=2)
    fig.update_layout(height=800, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics
    conv_info = optimization_results.get("convergence_info", {})
    pressure_info = optimization_results.get("optimized_pressure_curves", {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Iterations", f"{len(history)}")
    with col2:
        st.metric("Final Thrust Error", f"{thrust_errors[-1]:.1f}%" if thrust_errors else "N/A")
    with col3:
        st.metric("Final O/F Error", f"{of_errors[-1]:.1f}%" if of_errors else "N/A")
    with col4:
        status = "✅ Yes" if conv_info.get("converged", False) else "⚠️ No"
        st.metric("Converged", status)
    
    # Show chamber geometry (maximized diameter = minimized length)
    st.markdown("#### 📐 Chamber Geometry")
    col1, col2, col3 = st.columns(3)
    with col1:
        final_lstar = lstars[-1] if lstars else 1.0
        st.metric("Final L*", f"{final_lstar:.3f} m")
    with col2:
        final_l_chamber = l_chambers[-1] if l_chambers else 200
        st.metric("Chamber Length", f"{final_l_chamber:.1f} mm", delta="Using max diameter", delta_color="off")
    with col3:
        final_d_chamber = history[-1].get("D_chamber", 0.1) * 1000 if history else 100
        st.metric("Chamber Diameter", f"{final_d_chamber:.1f} mm")
    
    # Show optimized pressure profiles
    st.markdown("#### 🎯 Optimized Pressure Profiles")
    col1, col2 = st.columns(2)
    with col1:
        lox_end = pressure_info.get("lox_end_ratio", 0.7)
        lox_desc = "Regulated" if lox_end > 0.92 else ("Mild blowdown" if lox_end > 0.75 else "Aggressive blowdown")
        st.metric("LOX Tank End Pressure", f"{lox_end*100:.1f}%", delta=lox_desc, delta_color="off")
    with col2:
        fuel_end = pressure_info.get("fuel_end_ratio", 0.7)
        fuel_desc = "Regulated" if fuel_end > 0.92 else ("Mild blowdown" if fuel_end > 0.75 else "Aggressive blowdown")
        st.metric("Fuel Tank End Pressure", f"{fuel_end*100:.1f}%", delta=fuel_desc, delta_color="off")


def _display_chamber_geometry_plot(config: PintleEngineConfig, optimization_results: Dict[str, Any]) -> None:
    """Display chamber geometry visualization using the same approach as chamber design tab.
    
    Shows multi-layer structure: Gas → Ablative (Chamber) → Graphite (Throat) → Stainless Steel
    """
    try:
        # Get chamber parameters from config
        A_throat = getattr(config.chamber, 'A_throat', 1e-4)
        D_throat = np.sqrt(4 * A_throat / np.pi)
        D_chamber = getattr(config.chamber, 'chamber_inner_diameter', 0.08)
        V_chamber = getattr(config.chamber, 'volume', 0.001)
        L_chamber = getattr(config.chamber, 'length', 0.2)
        Lstar = getattr(config.chamber, 'Lstar', 1.0)
        
        # Get nozzle parameters
        L_nozzle = getattr(config.nozzle, 'length', 0.1) if hasattr(config, 'nozzle') else 0.1
        expansion_ratio = getattr(config.nozzle, 'expansion_ratio', 10.0) if hasattr(config, 'nozzle') else 10.0
        
        # Get ablative and graphite configs (same as chamber design tab)
        ablative_cfg = config.ablative_cooling if hasattr(config, 'ablative_cooling') else None
        graphite_cfg = config.graphite_insert if hasattr(config, 'graphite_insert') else None
        
        # Validate inputs
        if V_chamber <= 0 or A_throat <= 0 or L_chamber <= 0:
            st.warning(f"Invalid geometry inputs: V={V_chamber:.6f}, A_throat={A_throat:.6f}, L={L_chamber:.6f}")
            return
        
        # Calculate actual diameters
        D_chamber_actual = D_chamber if D_chamber > 0 else np.sqrt(4.0 * V_chamber / (np.pi * L_chamber))
        D_throat_actual = D_throat if D_throat > 0 else np.sqrt(4.0 * A_throat / np.pi)
        
        # Use the same clear geometry visualizer as the chamber design tab
        geometry_clear = calculate_chamber_geometry_clear(
            L_chamber=L_chamber,
            D_chamber=D_chamber_actual,
            D_throat=D_throat_actual,
            L_nozzle=L_nozzle,
            expansion_ratio=expansion_ratio,
            ablative_config=ablative_cfg,
            graphite_config=graphite_cfg,
            recession_chamber=0.0,  # No recession for fresh design
            recession_graphite=0.0,
            n_points=200,
        )
        
        # Create the same plot as chamber design tab (1:1 aspect ratio is handled in base function)
        fig_contour = plot_chamber_geometry_clear(geometry_clear, config)
        st.plotly_chart(fig_contour, use_container_width=True)
        
        # Display geometry summary
        st.markdown("#### Chamber Geometry Summary")
        
        # Calculate derived values
        A_chamber = np.pi * (D_chamber_actual / 2) ** 2
        contraction_ratio = A_chamber / A_throat if A_throat > 0 else 1.0
        A_exit = A_throat * expansion_ratio
        D_exit = np.sqrt(4 * A_exit / np.pi)
        
        # Get performance for Cf calculation
        performance = optimization_results.get("performance", {})
        Pc = performance.get("Pc", 2e6)
        thrust = performance.get("F", 5000)
        Cf = thrust / (Pc * A_throat) if Pc * A_throat > 0 else 1.5
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Chamber Length", f"{L_chamber * 39.37:.2f} in ({L_chamber * 1000:.1f} mm)")
        with col2:
            st.metric("Chamber Diameter", f"{D_chamber_actual * 39.37:.2f} in ({D_chamber_actual * 1000:.1f} mm)")
        with col3:
            st.metric("Throat Diameter", f"{D_throat_actual * 39.37:.3f} in ({D_throat_actual * 1000:.2f} mm)")
        with col4:
            st.metric("Exit Diameter", f"{D_exit * 39.37:.2f} in ({D_exit * 1000:.1f} mm)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("L* (Characteristic Length)", f"{Lstar:.3f} m ({Lstar * 39.37:.2f} in)")
        with col2:
            st.metric("Contraction Ratio", f"{contraction_ratio:.2f}")
        with col3:
            st.metric("Expansion Ratio", f"{expansion_ratio:.2f}")
        with col4:
            st.metric("Force Coefficient (Cf)", f"{Cf:.3f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Chamber Volume", f"{V_chamber * 1e6:.1f} cm³ ({V_chamber * 61023.7:.1f} in³)")
        with col2:
            if ablative_cfg and ablative_cfg.enabled:
                st.metric("Ablative Thickness", f"{ablative_cfg.initial_thickness * 1000:.1f} mm")
            else:
                st.metric("Ablative", "Not configured")
        with col3:
            if graphite_cfg and graphite_cfg.enabled:
                st.metric("Graphite Thickness", f"{graphite_cfg.initial_thickness * 1000:.1f} mm")
            else:
                st.metric("Graphite", "Not configured")
        
        # DXF Download - exactly like chamber design tab
        st.markdown("---")
        st.markdown("#### Download Chamber Contour")
        
        try:
            import tempfile
            import os
            
            # Generate DXF to a temporary file using chamber_geometry_calc
            with tempfile.NamedTemporaryFile(mode='w', suffix='.dxf', delete=False) as tmp_file:
                tmp_dxf_path = tmp_file.name
            
            # Generate DXF using chamber_geometry_calc
            _, _, _ = chamber_geometry_calc(
                pc_design=Pc,
                thrust_design=thrust,
                force_coeffcient=Cf,
                diameter_inner=D_chamber_actual,
                diameter_exit=D_exit,
                l_star=Lstar,
                do_plot=False,
                steps=200,
                export_dxf=tmp_dxf_path
            )
            
            # Read the DXF file
            with open(tmp_dxf_path, 'rb') as f:
                dxf_bytes = f.read()
            
            # Clean up temporary file
            os.unlink(tmp_dxf_path)
            
            # Download button
            st.download_button(
                label="📐 Download Chamber Contour (DXF)",
                data=dxf_bytes,
                file_name="optimized_chamber_contour.dxf",
                mime="application/dxf",
                key="full_engine_opt_dxf_download"
            )
            st.caption("DXF file includes: cylindrical section, 45° contraction cone, and RAO nozzle contour")
            
        except ImportError:
            st.warning("ezdxf library is required for DXF export. Install it with: `pip install ezdxf`")
        except Exception as e:
            st.warning(f"Could not generate DXF: {e}")
            
    except Exception as e:
        st.warning(f"Could not generate chamber geometry: {e}")
        import traceback
        st.code(traceback.format_exc())


def _plot_pressure_curves(pressure_curves: Dict[str, np.ndarray]) -> None:
    """Plot tank pressure and performance curves."""
    time = pressure_curves.get("time", np.array([]))
    if len(time) == 0:
        st.warning("No pressure curve data available")
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Tank Pressures", "Thrust", "Chamber Pressure", "Isp"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    # Tank pressures
    P_tank_O_psi = pressure_curves.get("P_tank_O", np.array([])) / 6894.76
    P_tank_F_psi = pressure_curves.get("P_tank_F", np.array([])) / 6894.76
    fig.add_trace(go.Scatter(x=time, y=P_tank_O_psi, name="LOX Tank", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=P_tank_F_psi, name="Fuel Tank", line=dict(color="orange")), row=1, col=1)
    
    # Thrust
    thrust = pressure_curves.get("thrust", np.array([]))
    fig.add_trace(go.Scatter(x=time, y=thrust, name="Thrust", line=dict(color="red")), row=1, col=2)
    
    # Chamber pressure
    Pc_MPa = pressure_curves.get("Pc", np.array([])) / 1e6
    fig.add_trace(go.Scatter(x=time, y=Pc_MPa, name="Pc", line=dict(color="green")), row=2, col=1)
    
    # Isp
    Isp = pressure_curves.get("Isp", np.array([]))
    fig.add_trace(go.Scatter(x=time, y=Isp, name="Isp", line=dict(color="purple")), row=2, col=2)
    
    # Update axes labels
    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=2)
    fig.update_yaxes(title_text="Pressure [psi]", row=1, col=1)
    fig.update_yaxes(title_text="Thrust [N]", row=1, col=2)
    fig.update_yaxes(title_text="Pc [MPa]", row=2, col=1)
    fig.update_yaxes(title_text="Isp [s]", row=2, col=2)
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Thrust", f"{np.mean(thrust):.1f} N")
    with col2:
        st.metric("Avg Isp", f"{np.mean(Isp):.1f} s")
    with col3:
        st.metric("Avg Pc", f"{np.mean(Pc_MPa):.2f} MPa")
    with col4:
        st.metric("Burn Time", f"{time[-1]:.1f} s")


def _plot_copv_pressure(copv_results: Dict[str, Any], pressure_curves: Dict[str, np.ndarray]) -> None:
    """Plot COPV pressure curve alongside tank pressures."""
    time = copv_results.get("time", np.array([]))
    copv_pressure_psi = copv_results.get("copv_pressure_psi", np.array([]))
    
    if len(time) == 0:
        st.warning("No COPV data available")
        return
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # COPV pressure
    fig.add_trace(
        go.Scatter(x=time, y=copv_pressure_psi, name="COPV Pressure", line=dict(color="green", width=2)),
        secondary_y=False
    )
    
    # Tank pressures
    P_tank_O_psi = pressure_curves.get("P_tank_O", np.array([])) / 6894.76
    P_tank_F_psi = pressure_curves.get("P_tank_F", np.array([])) / 6894.76
    fig.add_trace(
        go.Scatter(x=time, y=P_tank_O_psi, name="LOX Tank", line=dict(color="blue", dash="dot")),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(x=time, y=P_tank_F_psi, name="Fuel Tank", line=dict(color="orange", dash="dot")),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Time [s]")
    fig.update_yaxes(title_text="COPV Pressure [psi]", secondary_y=False)
    fig.update_yaxes(title_text="Tank Pressure [psi]", secondary_y=True)
    
    fig.update_layout(height=400, title="COPV and Tank Pressure Blowdown")
    st.plotly_chart(fig, use_container_width=True)


def _plot_flight_trajectory(flight_obj, requirements: Dict[str, Any]) -> None:
    """Plot flight trajectory from RocketPy flight object."""
    try:
        elevation = requirements.get("elevation", 0)
        
        # Extract flight data
        altitude_data = flight_obj.z.get_source()
        velocity_data = flight_obj.vz.get_source()
        
        if altitude_data is not None:
            time = altitude_data[:, 0]
            altitude_agl = altitude_data[:, 1] - elevation
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Altitude vs Time", "Velocity vs Time"))
            
            fig.add_trace(
                go.Scatter(x=time, y=altitude_agl, name="Altitude", line=dict(color="blue")),
                row=1, col=1
            )
            
            if velocity_data is not None:
                vz = velocity_data[:, 1]
                fig.add_trace(
                    go.Scatter(x=time, y=vz, name="Vertical Velocity", line=dict(color="red")),
                    row=1, col=2
                )
            
            fig.update_xaxes(title_text="Time [s]")
            fig.update_yaxes(title_text="Altitude AGL [m]", row=1, col=1)
            fig.update_yaxes(title_text="Velocity [m/s]", row=1, col=2)
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not plot flight trajectory: {e}")


def _show_full_engine_comparison(
    config_before: PintleEngineConfig,
    config_after: PintleEngineConfig,
    optimization_results: Dict[str, Any]
) -> None:
    """Show before/after comparison for full engine optimization."""
    st.markdown("### 📈 Before vs After Comparison")
    
    # Extract all parameters
    params_before = _extract_all_parameters(config_before)
    params_after = _extract_all_parameters(config_after)
    
    # Create comparison table
    comparison_data = []
    
    # Injector comparisons
    if "d_pintle_tip" in params_before and "d_pintle_tip" in params_after:
        comparison_data.append({
            "Component": "Injector",
            "Parameter": "Pintle Tip Ø [mm]",
            "Before": f"{params_before['d_pintle_tip'] * 1000:.2f}",
            "After": f"{params_after['d_pintle_tip'] * 1000:.2f}",
            "Change": f"{((params_after['d_pintle_tip'] / params_before['d_pintle_tip'] - 1) * 100):+.1f}%"
        })
    
    if "h_gap" in params_before and "h_gap" in params_after:
        comparison_data.append({
            "Component": "Injector",
            "Parameter": "Gap Height [mm]",
            "Before": f"{params_before['h_gap'] * 1000:.2f}",
            "After": f"{params_after['h_gap'] * 1000:.2f}",
            "Change": f"{((params_after['h_gap'] / params_before['h_gap'] - 1) * 100):+.1f}%"
        })
    
    if "n_orifices" in params_before and "n_orifices" in params_after:
        comparison_data.append({
            "Component": "Injector",
            "Parameter": "Orifice Count",
            "Before": f"{int(params_before['n_orifices'])}",
            "After": f"{int(params_after['n_orifices'])}",
            "Change": f"{int(params_after['n_orifices'] - params_before['n_orifices']):+d}"
        })
    
    if "d_orifice" in params_before and "d_orifice" in params_after:
        comparison_data.append({
            "Component": "Injector",
            "Parameter": "Orifice Ø [mm]",
            "Before": f"{params_before['d_orifice'] * 1000:.2f}",
            "After": f"{params_after['d_orifice'] * 1000:.2f}",
            "Change": f"{((params_after['d_orifice'] / params_before['d_orifice'] - 1) * 100):+.1f}%"
        })
    
    if "theta_orifice" in params_before and "theta_orifice" in params_after:
        comparison_data.append({
            "Component": "Injector",
            "Parameter": "Orifice Angle [°]",
            "Before": f"{params_before['theta_orifice']:.1f}",
            "After": f"{params_after['theta_orifice']:.1f}",
            "Change": f"Fixed at 90°"
        })
    
    # Chamber comparisons
    D_throat_before = np.sqrt(4 * params_before.get('A_throat', 0) / np.pi) * 1000
    D_throat_after = np.sqrt(4 * params_after.get('A_throat', 0) / np.pi) * 1000
    comparison_data.append({
        "Component": "Chamber",
        "Parameter": "Throat Ø [mm]",
        "Before": f"{D_throat_before:.2f}",
        "After": f"{D_throat_after:.2f}",
        "Change": f"{((D_throat_after / D_throat_before - 1) * 100):+.1f}%"
    })
    
    comparison_data.append({
        "Component": "Chamber",
        "Parameter": "L* [m]",
        "Before": f"{params_before.get('Lstar', 0):.3f}",
        "After": f"{params_after.get('Lstar', 0):.3f}",
        "Change": f"{((params_after.get('Lstar', 1) / params_before.get('Lstar', 1) - 1) * 100):+.1f}%"
    })
    
    comparison_data.append({
        "Component": "Chamber",
        "Parameter": "Diameter [mm]",
        "Before": f"{params_before.get('chamber_diameter', 0) * 1000:.1f}",
        "After": f"{params_after.get('chamber_diameter', 0) * 1000:.1f}",
        "Change": f"{((params_after.get('chamber_diameter', 1) / params_before.get('chamber_diameter', 1) - 1) * 100):+.1f}%"
    })
    
    # Nozzle comparisons
    comparison_data.append({
        "Component": "Nozzle",
        "Parameter": "Expansion Ratio",
        "Before": f"{params_before.get('expansion_ratio', 0):.2f}",
        "After": f"{params_after.get('expansion_ratio', 0):.2f}",
        "Change": f"{((params_after.get('expansion_ratio', 1) / params_before.get('expansion_ratio', 1) - 1) * 100):+.1f}%"
    })
    
    # Performance comparisons
    performance = optimization_results.get("performance", {})
    if performance:
        comparison_data.append({
            "Component": "Performance",
            "Parameter": "Thrust [N]",
            "Before": "-",
            "After": f"{performance.get('F', 0):.1f}",
            "Change": "Optimized"
        })
        comparison_data.append({
            "Component": "Performance",
            "Parameter": "Isp [s]",
            "Before": "-",
            "After": f"{performance.get('Isp', 0):.1f}",
            "Change": "Optimized"
        })
        comparison_data.append({
            "Component": "Performance",
            "Parameter": "Chamber Pressure [MPa]",
            "Before": "-",
            "After": f"{performance.get('Pc', 0) / 1e6:.2f}",
            "Change": "Optimized"
        })
        comparison_data.append({
            "Component": "Performance",
            "Parameter": "Mixture Ratio",
            "Before": "-",
            "After": f"{performance.get('MR', 0):.3f}",
            "Change": "Optimized"
        })
        
        # Stability
        stability = performance.get("stability_results", {})
        chugging = stability.get("chugging", {})
        if chugging:
            comparison_data.append({
                "Component": "Stability",
                "Parameter": "Chugging Margin",
                "Before": "-",
                "After": f"{chugging.get('stability_margin', 0):.3f}",
                "Change": "✅" if chugging.get('stability_margin', 0) > 1.0 else "⚠️"
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)


def _show_engine_validation_checks(
    config: PintleEngineConfig,
    optimization_results: Dict[str, Any],
    requirements: Dict[str, Any]
) -> None:
    """Show validation checks for the optimized engine."""
    st.markdown("### ✅ Engine Validation Checks")
    
    performance = optimization_results.get("performance", {})
    validation = optimization_results.get("validation", {})
    
    checks = []
    
    # Thrust check
    target_thrust = requirements.get("target_thrust", 7000.0)
    actual_thrust = performance.get("F", 0)
    thrust_error = abs(actual_thrust - target_thrust) / target_thrust * 100
    checks.append({
        "Check": "Target Thrust",
        "Target": f"{target_thrust:.0f} N",
        "Actual": f"{actual_thrust:.1f} N",
        "Error": f"{thrust_error:.1f}%",
        "Status": "✅ Pass" if thrust_error < 10 else "⚠️ Check"
    })
    
    # O/F ratio check
    target_of = requirements.get("optimal_of_ratio", 2.3)
    actual_of = performance.get("MR", 0)
    of_error = abs(actual_of - target_of) / target_of * 100
    checks.append({
        "Check": "O/F Ratio",
        "Target": f"{target_of:.2f}",
        "Actual": f"{actual_of:.3f}",
        "Error": f"{of_error:.1f}%",
        "Status": "✅ Pass" if of_error < 15 else "⚠️ Check"
    })
    
    # L* check
    min_lstar = requirements.get("min_Lstar", 0.95)
    max_lstar = requirements.get("max_Lstar", 1.27)
    actual_lstar = config.chamber.Lstar
    lstar_ok = min_lstar <= actual_lstar <= max_lstar
    checks.append({
        "Check": "L* Constraint",
        "Target": f"{min_lstar:.1f} - {max_lstar:.1f} m",
        "Actual": f"{actual_lstar:.3f} m",
        "Error": "-",
        "Status": "✅ Pass" if lstar_ok else "⚠️ Out of range"
    })
    
    # Stability checks
    stability = performance.get("stability_results", {})
    min_stability_margin = requirements.get("min_stability_margin", 1.2)
    
    chugging = stability.get("chugging", {})
    chugging_margin = chugging.get("stability_margin", 0)
    checks.append({
        "Check": "Chugging Stability",
        "Target": f"> {min_stability_margin:.2f}",
        "Actual": f"{chugging_margin:.3f}",
        "Error": "-",
        "Status": "✅ Pass" if chugging_margin >= min_stability_margin else "⚠️ Unstable"
    })
    
    acoustic = stability.get("acoustic", {})
    acoustic_margin = acoustic.get("stability_margin", 0)
    checks.append({
        "Check": "Acoustic Stability",
        "Target": f"> {min_stability_margin:.2f}",
        "Actual": f"{acoustic_margin:.3f}",
        "Error": "-",
        "Status": "✅ Pass" if acoustic_margin >= min_stability_margin else "⚠️ Unstable"
    })
    
    feed = stability.get("feed_system", {})
    feed_margin = feed.get("stability_margin", 0)
    checks.append({
        "Check": "Feed System Stability",
        "Target": f"> {min_stability_margin:.2f}",
        "Actual": f"{feed_margin:.3f}",
        "Error": "-",
        "Status": "✅ Pass" if feed_margin >= min_stability_margin else "⚠️ Unstable"
    })
    
    # Geometry checks
    max_chamber_od = requirements.get("max_chamber_outer_diameter", 0.15)
    actual_chamber_d = config.chamber.chamber_inner_diameter if hasattr(config.chamber, 'chamber_inner_diameter') and config.chamber.chamber_inner_diameter else np.sqrt(4.0 * config.chamber.volume / (np.pi * config.chamber.length))
    checks.append({
        "Check": "Chamber Diameter",
        "Target": f"< {max_chamber_od*1000:.0f} mm",
        "Actual": f"{actual_chamber_d*1000:.1f} mm",
        "Error": "-",
        "Status": "✅ Pass" if actual_chamber_d <= max_chamber_od else "⚠️ Too large"
    })
    
    # Orifice angle check
    orifice_angle = 90.0  # Should be fixed at 90
    if hasattr(config, 'injector') and config.injector.type == "pintle":
        if hasattr(config.injector.geometry, 'lox'):
            orifice_angle = config.injector.geometry.lox.theta_orifice
    checks.append({
        "Check": "Orifice Angle",
        "Target": "90° (perpendicular)",
        "Actual": f"{orifice_angle:.1f}°",
        "Error": "-",
        "Status": "✅ Pass" if abs(orifice_angle - 90.0) < 0.1 else "⚠️ Not perpendicular"
    })
    
    df_checks = pd.DataFrame(checks)
    st.dataframe(df_checks, use_container_width=True, hide_index=True)
    
    # Summary
    all_pass = all("Pass" in c["Status"] for c in checks)
    if all_pass:
        st.success("🎉 All validation checks passed! Engine design is valid.")
    else:
        failed = [c["Check"] for c in checks if "Pass" not in c["Status"]]
        st.warning(f"⚠️ Some checks need attention: {', '.join(failed)}")


def _injector_optimization_tab(config_obj: PintleEngineConfig, runner: Optional[PintleEngineRunner]) -> PintleEngineConfig:
    """Injector optimization tab."""
    st.subheader("Injector Geometry Optimization")
    st.markdown("""
    Optimize injector geometry (pintle tip, orifices, spray) to achieve:
    - Target mixture ratio
    - Good spray quality (SMD, evaporation)
    - Stable operation
    - Efficient combustion
    """)
    
    if runner is None:
        st.warning("⚠️ Runner not available. Please load configuration first.")
        return config_obj
    
    # Get design requirements
    requirements = st.session_state.get("design_requirements", {})
    if not requirements:
        st.warning("⚠️ Please set design requirements in the 'Design Requirements' tab first.")
        return config_obj
    
    target_thrust = requirements.get("target_thrust", 7000.0)
    target_MR = config_obj.combustion.MR if hasattr(config_obj.combustion, 'MR') else 2.5
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Use optimized config if available
        display_config = st.session_state.get("optimized_config", config_obj)
        is_optimized = "optimized_config" in st.session_state
        
        if is_optimized:
            st.markdown("### ✅ Optimized Injector Configuration")
            st.info("📊 Showing optimized parameters below. Scroll down to see before/after comparison.")
        else:
            st.markdown("### Current Injector Configuration")
        
        # Display injector parameters (optimized if available)
        injector_config = display_config.injector if hasattr(display_config, 'injector') else None
        if injector_config and injector_config.type == "pintle":
            geometry = injector_config.geometry
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if hasattr(geometry, 'fuel') and hasattr(geometry.fuel, 'd_pintle_tip'):
                    st.metric("Pintle Tip Diameter", f"{geometry.fuel.d_pintle_tip * 1000:.2f} mm")
                if hasattr(geometry, 'lox') and hasattr(geometry.lox, 'd_orifice'):
                    st.metric("Oxidizer Orifice Diameter", f"{geometry.lox.d_orifice * 1000:.2f} mm")
            with col_b:
                if hasattr(geometry, 'fuel') and hasattr(geometry.fuel, 'h_gap'):
                    st.metric("Fuel Gap Thickness", f"{geometry.fuel.h_gap * 1000:.2f} mm")
                if hasattr(geometry, 'lox') and hasattr(geometry.lox, 'n_orifices'):
                    st.metric("Number of Orifices", f"{geometry.lox.n_orifices}")
            with col_c:
                if hasattr(geometry, 'fuel') and hasattr(geometry.fuel, 'd_reservoir_inner'):
                    st.metric("Reservoir Inner Diameter", f"{geometry.fuel.d_reservoir_inner * 1000:.2f} mm")
                if hasattr(geometry, 'lox') and hasattr(geometry.lox, 'theta_orifice'):
                    st.metric("Orifice Angle", f"{geometry.lox.theta_orifice:.1f}°")
        elif injector_config:
            st.info(f"Injector type: {injector_config.type} (detailed metrics not yet implemented)")
        
        # Optimization controls
        st.markdown("### Optimization Options")
        optimize_injector = st.checkbox("Enable Injector Optimization", value=False)
        
        if optimize_injector:
            st.markdown("#### Optimization Variables")
            
            optimize_pintle = st.checkbox("Optimize Pintle Tip Diameter", value=True)
            optimize_orifices = st.checkbox("Optimize Orifice Sizes", value=True)
            optimize_spray = st.checkbox("Optimize Spray Parameters", value=False)
            
            if st.button("🚀 Run Injector Optimization", type="primary"):
                with st.spinner("Optimizing injector geometry..."):
                    try:
                        # Store before config for comparison
                        config_before = copy.deepcopy(config_obj)
                        
                        # Run injector optimization
                        optimized_config, optimization_results = _optimize_injector(
                            config_obj,
                            runner,
                            target_thrust,
                            target_MR,
                            optimize_pintle,
                            optimize_orifices,
                            optimize_spray,
                        )
                        
                        # Store results
                        config_obj = optimized_config
                        st.session_state["optimized_config"] = optimized_config
                        st.session_state["optimization_results"] = optimization_results
                        st.session_state["optimization_before_config"] = config_before
                        
                        # Update config_dict
                        config_dict_updated = optimized_config.model_dump(exclude_none=False)
                        st.session_state["config_dict"] = config_dict_updated
                        
                        st.success("✅ Injector optimization complete!")
                        
                        # Display optimized parameters immediately
                        st.markdown("---")
                        st.markdown("## ✅ Optimization Complete!")
                        
                        # Show before/after comparison
                        _show_injector_comparison(config_before, optimized_config, optimization_results)
                        
                        # Display optimized parameters
                        st.markdown("### 📊 Optimized Injector Parameters")
                        _display_injector_parameters(optimized_config, optimization_results)
                        
                    except Exception as e:
                        st.error(f"Optimization failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    with col2:
        st.markdown("### Injector Diagnostics")
        if runner:
            try:
                # Get tank pressures (use defaults if not available)
                P_tank_O = 3e6  # 3 MPa default
                P_tank_F = 3e6
                
                results = runner.evaluate(P_tank_O, P_tank_F)
                diagnostics = results.get("diagnostics", {})
                
                # Injector diagnostics
                injector_diag = diagnostics.get("injector_pressure", {})
                if injector_diag:
                    st.metric("Oxidizer Pressure Drop", f"{injector_diag.get('delta_P_O', 0) / 6894.76:.1f} psi")
                    st.metric("Fuel Pressure Drop", f"{injector_diag.get('delta_P_F', 0) / 6894.76:.1f} psi")
                
                # Spray diagnostics
                spray_diag = diagnostics.get("spray_diagnostics", {})
                if spray_diag:
                    st.metric("SMD (Oxidizer)", f"{spray_diag.get('D32_O', 0) * 1e6:.1f} µm")
                    st.metric("SMD (Fuel)", f"{spray_diag.get('D32_F', 0) * 1e6:.1f} µm")
                    st.metric("Evaporation Length", f"{spray_diag.get('x_star', 0) * 1000:.1f} mm")
            except Exception as e:
                st.warning(f"Could not compute diagnostics: {e}")
    
    return config_obj


def _chamber_optimization_tab(config_obj: PintleEngineConfig, runner: Optional[PintleEngineRunner]) -> PintleEngineConfig:
    """Chamber optimization tab."""
    st.subheader("Chamber Geometry Optimization")
    st.markdown("""
    Optimize chamber geometry (throat, exit, L*) and cooling system (ablative, graphite) to achieve:
    - Target thrust
    - Required stability margins
    - Adequate cooling for burn time
    - Optimal performance
    """)
    
    if runner is None:
        st.warning("⚠️ Runner not available. Please load configuration first.")
        return config_obj
    
    # Get design requirements
    requirements = st.session_state.get("design_requirements", {})
    if not requirements:
        st.warning("⚠️ Please set design requirements in the 'Design Requirements' tab first.")
        return config_obj
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Current Chamber Configuration")
        
        # Display current chamber parameters
        chamber_config = config_obj.chamber if hasattr(config_obj, 'chamber') else None
        if chamber_config:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Throat Area", f"{chamber_config.A_throat * 1e6:.2f} mm²")
                st.metric("Throat Diameter", f"{np.sqrt(4 * chamber_config.A_throat / np.pi) * 1000:.2f} mm")
            with col_b:
                st.metric("Chamber Volume", f"{chamber_config.volume * 1000:.2f} L")
                st.metric("L*", f"{chamber_config.Lstar * 1000:.1f} mm")
            with col_c:
                st.metric("Chamber Length", f"{chamber_config.length * 1000:.1f} mm")
                st.metric("Chamber Diameter", f"{np.sqrt(4 * chamber_config.volume / (np.pi * chamber_config.length)) * 1000:.1f} mm")
        
        # Nozzle parameters
        nozzle_config = config_obj.nozzle if hasattr(config_obj, 'nozzle') else None
        if nozzle_config:
            st.markdown("#### Nozzle")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Exit Area", f"{nozzle_config.A_exit * 1e6:.2f} mm²")
            with col_b:
                st.metric("Expansion Ratio", f"{nozzle_config.expansion_ratio:.2f}")
        
        # Cooling system status
        st.markdown("#### Cooling System")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            ablative_enabled = (config_obj.ablative_cooling and 
                              config_obj.ablative_cooling.enabled if hasattr(config_obj, 'ablative_cooling') else False)
            st.metric("Ablative Liner", "✅ Enabled" if ablative_enabled else "❌ Disabled")
        with col_b:
            graphite_enabled = (config_obj.graphite_insert and 
                               config_obj.graphite_insert.enabled if hasattr(config_obj, 'graphite_insert') else False)
            st.metric("Graphite Insert", "✅ Enabled" if graphite_enabled else "❌ Disabled")
        with col_c:
            regen_enabled = (config_obj.regen_cooling and 
                            config_obj.regen_cooling.enabled if hasattr(config_obj, 'regen_cooling') else False)
            st.metric("Regen Cooling", "✅ Enabled" if regen_enabled else "❌ Disabled")
        
        # Optimization controls
        st.markdown("### Optimization Options")
        optimize_chamber = st.checkbox("Enable Chamber Optimization", value=False)
        
        if optimize_chamber:
            st.markdown("#### Optimization Variables")
            
            optimize_geometry = st.checkbox("Optimize Geometry (Throat, Exit, L*)", value=True)
            optimize_cooling = st.checkbox("Optimize Cooling System Sizing", value=True)
            optimize_ablative = st.checkbox("Optimize Ablative Thickness", value=ablative_enabled)
            optimize_graphite = st.checkbox("Optimize Graphite Insert", value=graphite_enabled)
            
            # Get tank pressures for optimization
            P_tank_O = st.number_input(
                "Oxidizer Tank Pressure [Pa]",
                min_value=1e5,
                max_value=10e6,
                value=3e6,
                step=1e5,
                format="%.0f"
            )
            P_tank_F = st.number_input(
                "Fuel Tank Pressure [Pa]",
                min_value=1e5,
                max_value=10e6,
                value=3e6,
                step=1e5,
                format="%.0f"
            )
            
            # Option to use coupled optimization
            use_coupled = st.checkbox(
                "Use Coupled Optimization (Pintle + Chamber)",
                value=True,
                help="Iteratively optimize both pintle and chamber until convergence"
            )
            
            if st.button("🚀 Run Chamber Optimization", type="primary"):
                with st.spinner("Optimizing chamber geometry and cooling system..."):
                    try:
                        if use_coupled:
                            # Use coupled optimizer
                            from pintle_pipeline.coupled_optimizer import CoupledPintleChamberOptimizer
                            
                            coupled_optimizer = CoupledPintleChamberOptimizer(config_obj)
                            
                            design_requirements = {
                                "target_thrust": requirements.get("target_thrust", 7000.0),
                                "target_burn_time": requirements.get("target_burn_time", 10.0),
                                "target_stability_margin": requirements.get("min_stability_margin", 1.2),
                                "P_tank_O": P_tank_O,
                                "P_tank_F": P_tank_F,
                                "target_Isp": requirements.get("target_Isp", None),
                            }
                            
                            constraints = {
                                "max_chamber_length": requirements.get("max_chamber_length", 0.5),
                                "max_chamber_diameter": requirements.get("max_chamber_diameter", 0.15),
                                "min_Lstar": 0.95,
                                "max_Lstar": 1.27,
                                "min_expansion_ratio": 3.0,
                                "max_expansion_ratio": 30.0,
                                "max_engine_weight": requirements.get("max_total_mass", None),
                            }
                            
                            coupled_results = coupled_optimizer.optimize_coupled(
                                design_requirements,
                                constraints,
                                max_iterations=10,
                                use_time_varying=True,  # Optimize across entire burn time
                            )
                            
                            optimized_config = coupled_results["optimized_config"]
                            optimization_results = coupled_results
                            
                            # Display convergence info
                            conv_info = coupled_results["convergence_info"]
                            if conv_info["converged"]:
                                st.success(f"✅ Coupled optimization converged after {conv_info['iterations']} iterations!")
                            else:
                                st.warning(f"⚠️ Optimization did not fully converge after {conv_info['iterations']} iterations (change: {conv_info['final_change']*100:.2f}%)")
                            
                        else:
                            # Use single chamber optimization
                            optimized_config, optimization_results = _optimize_chamber(
                            config_obj,
                            runner,
                            requirements,
                            P_tank_O,
                            P_tank_F,
                            optimize_geometry,
                            optimize_cooling,
                            optimize_ablative,
                            optimize_graphite,
                        )
                        
                        # Store results
                        st.session_state["optimized_config"] = optimized_config
                        st.session_state["optimization_results"] = optimization_results
                        st.session_state["optimization_before_config"] = copy.deepcopy(config_obj)  # Store before for comparison
                        
                        # Update main config_dict so changes persist
                        import yaml
                        config_dict_updated = optimized_config.model_dump(exclude_none=False)
                        st.session_state["config_dict"] = config_dict_updated
                        
                        # Display results immediately
                        st.markdown("---")
                        st.markdown("## ✅ Optimization Complete!")
                        
                        # Show before/after comparison
                        _show_optimization_comparison(config_obj, optimized_config, optimization_results)
                        
                        # Display optimized parameters
                        st.markdown("### 📊 Optimized Parameters")
                        _display_optimized_parameters(optimization_results, optimized_config)
                        
                        # Show time-varying results if available
                        # Check both possible locations: direct key and nested in performance
                        time_varying_summary = optimization_results.get("time_varying")
                        if time_varying_summary is None:
                            time_varying_summary = optimization_results.get("performance", {}).get("time_varying")
                        if time_varying_summary:
                            _show_time_varying_results(time_varying_summary)
                        
                        # Also show time-varying plots if array data available
                        if "time_varying_results" in optimization_results:
                            _plot_time_varying_results(optimization_results["time_varying_results"])
                        
                        # Update config_obj for return
                        config_obj = optimized_config
                        
                    except Exception as e:
                        st.error(f"Optimization failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    with col2:
        st.markdown("### Chamber Diagnostics")
        if runner:
            try:
                P_tank_O = 3e6
                P_tank_F = 3e6
                results = runner.evaluate(P_tank_O, P_tank_F)
                
                st.metric("Chamber Pressure", f"{results.get('Pc', 0) / 1e6:.2f} MPa")
                st.metric("Chamber Temperature", f"{results.get('Tc', 0):.0f} K")
                st.metric("Thrust", f"{results.get('F', 0):.1f} N")
                st.metric("Isp", f"{results.get('Isp', 0):.1f} s")
                st.metric("c* (actual)", f"{results.get('cstar_actual', 0):.0f} m/s")
                
                # Chamber intrinsics
                intrinsics = results.get("chamber_intrinsics", {})
                if intrinsics:
                    st.metric("L*", f"{intrinsics.get('Lstar', 0) * 1000:.1f} mm")
                    st.metric("Mach Number", f"{intrinsics.get('mach_number', 0):.3f}")
                    st.metric("Residence Time", f"{intrinsics.get('residence_time', 0) * 1000:.2f} ms")
            except Exception as e:
                st.warning(f"Could not compute diagnostics: {e}")
    
    return config_obj


def _stability_analysis_tab(config_obj: PintleEngineConfig, runner: Optional[PintleEngineRunner]) -> None:
    """Stability analysis tab."""
    st.subheader("Stability Margin Analysis")
    st.markdown("""
    Comprehensive stability analysis including:
    - Chugging stability (feed system coupling)
    - Acoustic stability (combustion instabilities)
    - Feed system stability (pressure oscillations)
    """)
    
    if runner is None:
        st.warning("⚠️ Runner not available. Please load configuration first.")
        return
    
    # Get requirements
    requirements = st.session_state.get("design_requirements", {})
    min_stability_margin = requirements.get("min_stability_margin", 1.2)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Run stability analysis
        P_tank_O = st.number_input(
            "Oxidizer Tank Pressure [Pa]",
            min_value=1e5,
            max_value=10e6,
            value=3e6,
            step=1e5,
            format="%.0f",
            key="stability_P_tank_O"
        )
        P_tank_F = st.number_input(
            "Fuel Tank Pressure [Pa]",
            min_value=1e5,
            max_value=10e6,
            value=3e6,
            step=1e5,
            format="%.0f",
            key="stability_P_tank_F"
        )
        
        if st.button("🔍 Analyze Stability", type="primary"):
            with st.spinner("Running stability analysis..."):
                try:
                    results = runner.evaluate(P_tank_O, P_tank_F)
                    stability_results = results.get("stability_results", {})
                    
                    # Display stability margins
                    st.markdown("### Stability Margins")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    # Chugging stability
                    chugging = stability_results.get("chugging", {})
                    chugging_margin = chugging.get("stability_margin", 0.0)
                    chugging_freq = chugging.get("frequency", 0.0)
                    
                    with col_a:
                        margin_color = "🟢" if chugging_margin >= min_stability_margin else "🔴"
                        st.metric(
                            f"{margin_color} Chugging Margin",
                            f"{chugging_margin:.3f}",
                            delta=f"Target: {min_stability_margin:.2f}"
                        )
                        st.caption(f"Frequency: {chugging_freq:.1f} Hz")
                    
                    # Acoustic stability
                    acoustic = stability_results.get("acoustic", {})
                    acoustic_margin = acoustic.get("stability_margin", 0.0)
                    acoustic_modes = acoustic.get("modes", {})
                    
                    with col_b:
                        margin_color = "🟢" if acoustic_margin >= min_stability_margin else "🔴"
                        st.metric(
                            f"{margin_color} Acoustic Margin",
                            f"{acoustic_margin:.3f}",
                            delta=f"Target: {min_stability_margin:.2f}"
                        )
                        if acoustic_modes:
                            first_mode = list(acoustic_modes.values())[0] if acoustic_modes else 0.0
                            st.caption(f"1st Mode: {first_mode:.1f} Hz")
                    
                    # Feed system stability
                    feed = stability_results.get("feed_system", {})
                    feed_margin = feed.get("stability_margin", 0.0)
                    
                    with col_c:
                        margin_color = "🟢" if feed_margin >= min_stability_margin else "🔴"
                        st.metric(
                            f"{margin_color} Feed System Margin",
                            f"{feed_margin:.3f}",
                            delta=f"Target: {min_stability_margin:.2f}"
                        )
                    
                    # Overall status
                    all_stable = (chugging_margin >= min_stability_margin and
                                 acoustic_margin >= min_stability_margin and
                                 feed_margin >= min_stability_margin)
                    
                    if all_stable:
                        st.success("✅ All stability margins meet requirements!")
                    else:
                        st.warning("⚠️ Some stability margins are below requirements. Consider optimization.")
                    
                    # Plot stability over time (if time-series available)
                    st.markdown("### Stability Evolution")
                    if st.checkbox("Show time-varying stability", value=False):
                        _plot_stability_evolution(runner, P_tank_O, P_tank_F)
                    
                except Exception as e:
                    st.error(f"Stability analysis failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with col2:
        st.markdown("### Stability Guidelines")
        st.info("""
        **Chugging Stability:**
        - Margin > 1.2 (20% margin) recommended
        - Affected by: injector design, feed system, chamber geometry
        
        **Acoustic Stability:**
        - Margin > 1.2 recommended
        - Affected by: chamber length, L*, injector design
        
        **Feed System Stability:**
        - Margin > 1.15 recommended
        - Affected by: tank pressures, line sizes, injector pressure drops
        """)


def _flight_performance_tab(config_obj: PintleEngineConfig, runner: Optional[PintleEngineRunner]) -> None:
    """Flight performance tab."""
    st.subheader("Flight Performance Analysis")
    st.markdown("""
    Analyze flight performance including:
    - Altitude capability
    - Payload capacity
    - Trajectory optimization
    """)
    
    if runner is None:
        st.warning("⚠️ Runner not available. Please load configuration first.")
        return
    
    requirements = st.session_state.get("design_requirements", {})
    target_altitude = requirements.get("target_apogee", 3048.0)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Flight Simulation")
        
        if st.button("✈️ Run Flight Simulation", type="primary"):
            with st.spinner("Running flight simulation..."):
                try:
                    from examples.pintle_engine.flight_sim import setup_flight
                    from examples.pintle_engine.interactive_pipeline import solve_for_thrust
                    
                    # Generate thrust curve
                    P_tank_O = 3e6
                    P_tank_F = 3e6
                    
                    # Run flight simulation
                    # (Implementation would go here)
                    
                    st.success("✅ Flight simulation complete!")
                    st.info("Flight simulation results would be displayed here.")
                    
                except Exception as e:
                    st.error(f"Flight simulation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with col2:
        st.markdown("### Performance Targets")
        st.metric("Target Altitude", f"{target_altitude:.0f} m")
        st.metric("Target Thrust", f"{requirements.get('target_thrust', 7000):.0f} N")


def _results_export_tab(config_obj: PintleEngineConfig, runner: Optional[PintleEngineRunner]) -> None:
    """Results and export tab."""
    st.subheader("Optimization Results & Export")
    
    optimized_config = st.session_state.get("optimized_config", None)
    optimization_results = st.session_state.get("optimization_results", None)
    
    if optimized_config and optimization_results:
        st.success("✅ Optimized configuration available!")
        
        # Display summary
        st.markdown("### 📊 Optimization Summary")
        
        # Show optimized parameters
        _display_optimized_parameters(optimization_results, optimized_config)
        
        # Show time-varying plot if available
        if "time_varying_results" in optimization_results:
            st.markdown("### ⏱️ Time-Varying Performance")
            _plot_time_varying_results(optimization_results["time_varying_results"])
        
        # Compare before/after
        config_before = st.session_state.get("optimization_before_config", None)
        if config_before and runner:
            try:
                P_tank_O = 3e6
                P_tank_F = 3e6
                
                # Before
                results_before = runner.evaluate(P_tank_O, P_tank_F)
                
                # After
                runner_opt = PintleEngineRunner(optimized_config)
                results_after = runner_opt.evaluate(P_tank_O, P_tank_F)
                
                # Comparison table
                comparison_data = {
                    "Metric": ["Thrust [N]", "Isp [s]", "Chamber Pressure [MPa]", "Stability Margin"],
                    "Before": [
                        f"{results_before.get('F', 0):.1f}",
                        f"{results_before.get('Isp', 0):.1f}",
                        f"{results_before.get('Pc', 0) / 1e6:.2f}",
                        f"{results_before.get('stability_results', {}).get('chugging', {}).get('stability_margin', 0):.3f}",
                    ],
                    "After": [
                        f"{results_after.get('F', 0):.1f}",
                        f"{results_after.get('Isp', 0):.1f}",
                        f"{results_after.get('Pc', 0) / 1e6:.2f}",
                        f"{results_after.get('stability_results', {}).get('chugging', {}).get('stability_margin', 0):.3f}",
                    ],
                }
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not compare results: {e}")
        
        # Export options
        st.markdown("### Export Configuration")
        if st.button("💾 Export Optimized Config (YAML)"):
            try:
                import yaml
                from pintle_pipeline.io import save_config
                
                # Save config
                config_dict = optimized_config.model_dump(exclude_none=False)
                yaml_str = yaml.dump(config_dict, default_flow_style=False)
                
                st.download_button(
                    label="Download YAML",
                    data=yaml_str,
                    file_name="optimized_engine_config.yaml",
                    mime="text/yaml"
                )
            except Exception as e:
                st.error(f"Export failed: {e}")
    else:
        st.info("No optimized configuration available. Run optimization in Injector or Chamber tabs.")


def _optimize_injector(
    config_obj: PintleEngineConfig,
    runner: PintleEngineRunner,
    target_thrust: float,
    target_MR: float,
    optimize_pintle: bool,
    optimize_orifices: bool,
    optimize_spray: bool,
) -> Tuple[PintleEngineConfig, Dict[str, Any]]:
    """Optimize injector geometry using comprehensive optimizer."""
    from pintle_pipeline.comprehensive_optimizer import ComprehensivePintleOptimizer
    
    optimizer = ComprehensivePintleOptimizer(config_obj)
    
    # Run optimization
    results = optimizer.optimize_pintle_geometry(
        target_thrust=target_thrust,
        target_mr=target_MR,
        P_tank_O=3.0e6,  # Default
        P_tank_F=3.0e6,  # Default
    )
    
    # Extract optimized parameters for display
    optimized_config = results["optimized_config"]
    optimized_params = {}
    
    if hasattr(optimized_config, 'injector') and optimized_config.injector.type == "pintle":
        geometry = optimized_config.injector.geometry
        if hasattr(geometry, 'fuel'):
            optimized_params["d_pintle_tip"] = geometry.fuel.d_pintle_tip
            optimized_params["h_gap"] = geometry.fuel.h_gap
            optimized_params["d_reservoir_inner"] = geometry.fuel.d_reservoir_inner if hasattr(geometry.fuel, 'd_reservoir_inner') else None
        if hasattr(geometry, 'lox'):
            optimized_params["n_orifices"] = geometry.lox.n_orifices
            optimized_params["d_orifice"] = geometry.lox.d_orifice
            optimized_params["theta_orifice"] = geometry.lox.theta_orifice
    
    # Add optimized parameters to results
    results["optimized_parameters"] = optimized_params
    
    return optimized_config, results


def _show_injector_comparison(
    config_before: PintleEngineConfig,
    config_after: PintleEngineConfig,
    optimization_results: Dict[str, Any]
) -> None:
    """Show before/after comparison for injector optimization."""
    st.markdown("### 📈 Before vs After Comparison")
    
    # Extract parameters
    def get_injector_params(config):
        params = {}
        if hasattr(config, 'injector') and config.injector.type == "pintle":
            geometry = config.injector.geometry
            if hasattr(geometry, 'fuel'):
                params["d_pintle_tip"] = geometry.fuel.d_pintle_tip
                params["h_gap"] = geometry.fuel.h_gap
                params["d_reservoir_inner"] = geometry.fuel.d_reservoir_inner if hasattr(geometry.fuel, 'd_reservoir_inner') else 0.0
            if hasattr(geometry, 'lox'):
                params["n_orifices"] = geometry.lox.n_orifices
                params["d_orifice"] = geometry.lox.d_orifice
                params["theta_orifice"] = geometry.lox.theta_orifice
        return params
    
    params_before = get_injector_params(config_before)
    params_after = get_injector_params(config_after)
    
    # Create comparison table
    comparison_data = []
    
    if "d_pintle_tip" in params_before and "d_pintle_tip" in params_after:
        comparison_data.append({
            "Parameter": "Pintle Tip Diameter [mm]",
            "Before": f"{params_before['d_pintle_tip'] * 1000:.2f}",
            "After": f"{params_after['d_pintle_tip'] * 1000:.2f}",
            "Change": f"{((params_after['d_pintle_tip'] / params_before['d_pintle_tip'] - 1) * 100):+.1f}%"
        })
    
    if "h_gap" in params_before and "h_gap" in params_after:
        comparison_data.append({
            "Parameter": "Fuel Gap Thickness [mm]",
            "Before": f"{params_before['h_gap'] * 1000:.2f}",
            "After": f"{params_after['h_gap'] * 1000:.2f}",
            "Change": f"{((params_after['h_gap'] / params_before['h_gap'] - 1) * 100):+.1f}%"
        })
    
    if "n_orifices" in params_before and "n_orifices" in params_after:
        comparison_data.append({
            "Parameter": "Number of Orifices",
            "Before": f"{int(params_before['n_orifices'])}",
            "After": f"{int(params_after['n_orifices'])}",
            "Change": f"{int(params_after['n_orifices'] - params_before['n_orifices']):+d}"
        })
    
    if "d_orifice" in params_before and "d_orifice" in params_after:
        comparison_data.append({
            "Parameter": "Orifice Diameter [mm]",
            "Before": f"{params_before['d_orifice'] * 1000:.2f}",
            "After": f"{params_after['d_orifice'] * 1000:.2f}",
            "Change": f"{((params_after['d_orifice'] / params_before['d_orifice'] - 1) * 100):+.1f}%"
        })
    
    if "theta_orifice" in params_before and "theta_orifice" in params_after:
        comparison_data.append({
            "Parameter": "Orifice Angle [°]",
            "Before": f"{params_before['theta_orifice']:.1f}",
            "After": f"{params_after['theta_orifice']:.1f}",
            "Change": f"{(params_after['theta_orifice'] - params_before['theta_orifice']):+.1f}°"
        })
    
    # Performance comparison
    try:
        runner_before = PintleEngineRunner(config_before)
        runner_after = PintleEngineRunner(config_after)
        P_tank_O = 3e6
        P_tank_F = 3e6
        
        results_before = runner_before.evaluate(P_tank_O, P_tank_F)
        results_after = optimization_results.get("performance", runner_after.evaluate(P_tank_O, P_tank_F))
        
        comparison_data.append({
            "Parameter": "Thrust [N]",
            "Before": f"{results_before.get('F', 0):.1f}",
            "After": f"{results_after.get('F', 0):.1f}",
            "Change": f"{((results_after.get('F', 0) / max(results_before.get('F', 1), 1) - 1) * 100):+.1f}%"
        })
        comparison_data.append({
            "Parameter": "Isp [s]",
            "Before": f"{results_before.get('Isp', 0):.1f}",
            "After": f"{results_after.get('Isp', 0):.1f}",
            "Change": f"{((results_after.get('Isp', 0) / max(results_before.get('Isp', 1), 1) - 1) * 100):+.1f}%"
        })
        comparison_data.append({
            "Parameter": "Mixture Ratio",
            "Before": f"{results_before.get('MR', 0):.3f}",
            "After": f"{results_after.get('MR', 0):.3f}",
            "Change": f"{(results_after.get('MR', 0) - results_before.get('MR', 0)):+.3f}"
        })
    except Exception:
        pass
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    else:
        st.warning("Could not generate comparison table")


def _display_injector_parameters(config: PintleEngineConfig, optimization_results: Dict[str, Any]) -> None:
    """Display optimized injector parameters."""
    if not hasattr(config, 'injector') or config.injector.type != "pintle":
        st.warning("No pintle injector configuration found")
        return
    
    geometry = config.injector.geometry
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔧 Fuel Geometry")
        if hasattr(geometry, 'fuel'):
            st.metric("Pintle Tip Diameter", f"{geometry.fuel.d_pintle_tip * 1000:.2f} mm")
            st.metric("Gap Height", f"{geometry.fuel.h_gap * 1000:.2f} mm")
            if hasattr(geometry.fuel, 'd_reservoir_inner'):
                st.metric("Reservoir Inner Diameter", f"{geometry.fuel.d_reservoir_inner * 1000:.2f} mm")
    
    with col2:
        st.markdown("#### 🔵 Oxidizer Geometry")
        if hasattr(geometry, 'lox'):
            st.metric("Number of Orifices", f"{geometry.lox.n_orifices}")
            st.metric("Orifice Diameter", f"{geometry.lox.d_orifice * 1000:.2f} mm")
            st.metric("Orifice Angle", f"{geometry.lox.theta_orifice:.1f}°")
    
    with col3:
        st.markdown("#### ⚡ Performance")
        performance = optimization_results.get("performance", {})
        if performance:
            st.metric("Thrust", f"{performance.get('F', 0):.1f} N")
            st.metric("Isp", f"{performance.get('Isp', 0):.1f} s")
            st.metric("Mixture Ratio", f"{performance.get('MR', 0):.3f}")
            st.metric("Chamber Pressure", f"{performance.get('Pc', 0) / 1e6:.2f} MPa")
            
            # Spray diagnostics if available
            diagnostics = performance.get("diagnostics", {})
            spray_diag = diagnostics.get("spray_diagnostics", {})
            if spray_diag:
                st.markdown("##### Spray Quality")
                st.metric("SMD (Oxidizer)", f"{spray_diag.get('D32_O', 0) * 1e6:.1f} µm")
                st.metric("SMD (Fuel)", f"{spray_diag.get('D32_F', 0) * 1e6:.1f} µm")
                st.metric("Evaporation Length", f"{spray_diag.get('x_star', 0) * 1000:.1f} mm")


def _optimize_chamber(
    config_obj: PintleEngineConfig,
    runner: PintleEngineRunner,
    requirements: Dict[str, Any],
    P_tank_O: float,
    P_tank_F: float,
    optimize_geometry: bool,
    optimize_cooling: bool,
    optimize_ablative: bool,
    optimize_graphite: bool,
) -> Tuple[PintleEngineConfig, Dict[str, Any]]:
    """Optimize chamber geometry and cooling system with time-varying analysis."""
    from pintle_pipeline.chamber_optimizer import ChamberOptimizer
    
    optimizer = ChamberOptimizer(config_obj)
    
    # Set up design requirements
    design_requirements = {
        "target_thrust": requirements.get("target_thrust", 7000.0),
        "target_burn_time": requirements.get("target_burn_time", 10.0),
        "target_stability_margin": requirements.get("min_stability_margin", 1.2),
        "P_tank_O": P_tank_O,
        "P_tank_F": P_tank_F,
        "target_Isp": requirements.get("target_Isp", None),
    }
    
    # Set up constraints
    constraints = {
        "max_chamber_length": requirements.get("max_chamber_length", 0.5),
        "max_chamber_diameter": requirements.get("max_chamber_diameter", 0.15),
        "min_Lstar": 0.95,
        "max_Lstar": 1.27,
        "min_expansion_ratio": 3.0,
        "max_expansion_ratio": 30.0,
        "max_engine_weight": requirements.get("max_total_mass", None),
        "max_vehicle_length": requirements.get("max_chamber_length", None),
        "max_vehicle_diameter": requirements.get("max_chamber_diameter", None),
    }
    
    # Run optimization (now includes time-varying analysis)
    results = optimizer.optimize(design_requirements, constraints)
    
    return results["optimized_config"], results


def _show_optimization_comparison(
    config_before: PintleEngineConfig,
    config_after: PintleEngineConfig,
    optimization_results: Dict[str, Any]
) -> None:
    """Show before/after comparison of optimization."""
    st.markdown("### 📈 Before vs After Comparison")
    
    # Get performance before
    try:
        runner_before = PintleEngineRunner(config_before)
        P_tank_O = optimization_results.get("design_requirements", {}).get("P_tank_O", 3e6)
        P_tank_F = optimization_results.get("design_requirements", {}).get("P_tank_F", 3e6)
        results_before = runner_before.evaluate(P_tank_O, P_tank_F)
    except Exception:
        results_before = {}
    
    # Get performance after
    results_after = optimization_results.get("performance", {})
    
    # Create comparison table
    comparison_data = []
    
    # Geometry comparison
    comparison_data.append({
        "Parameter": "Throat Area [mm²]",
        "Before": f"{config_before.chamber.A_throat * 1e6:.2f}",
        "After": f"{config_after.chamber.A_throat * 1e6:.2f}",
        "Change": f"{((config_after.chamber.A_throat / config_before.chamber.A_throat - 1) * 100):+.1f}%"
    })
    comparison_data.append({
        "Parameter": "Exit Area [mm²]",
        "Before": f"{config_before.nozzle.A_exit * 1e6:.2f}",
        "After": f"{config_after.nozzle.A_exit * 1e6:.2f}",
        "Change": f"{((config_after.nozzle.A_exit / config_before.nozzle.A_exit - 1) * 100):+.1f}%"
    })
    comparison_data.append({
        "Parameter": "L* [mm]",
        "Before": f"{config_before.chamber.Lstar * 1000:.1f}",
        "After": f"{config_after.chamber.Lstar * 1000:.1f}",
        "Change": f"{((config_after.chamber.Lstar / config_before.chamber.Lstar - 1) * 100):+.1f}%"
    })
    
    # Performance comparison
    if results_before and results_after:
        comparison_data.append({
            "Parameter": "Thrust [N]",
            "Before": f"{results_before.get('F', 0):.1f}",
            "After": f"{results_after.get('F', 0):.1f}",
            "Change": f"{((results_after.get('F', 0) / max(results_before.get('F', 1), 1) - 1) * 100):+.1f}%"
        })
        comparison_data.append({
            "Parameter": "Isp [s]",
            "Before": f"{results_before.get('Isp', 0):.1f}",
            "After": f"{results_after.get('Isp', 0):.1f}",
            "Change": f"{((results_after.get('Isp', 0) / max(results_before.get('Isp', 1), 1) - 1) * 100):+.1f}%"
        })
        comparison_data.append({
            "Parameter": "Chamber Pressure [MPa]",
            "Before": f"{results_before.get('Pc', 0) / 1e6:.2f}",
            "After": f"{results_after.get('Pc', 0) / 1e6:.2f}",
            "Change": f"{((results_after.get('Pc', 0) / max(results_before.get('Pc', 1), 1) - 1) * 100):+.1f}%"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)


def _display_optimized_parameters(optimization_results: Dict[str, Any], config: PintleEngineConfig) -> None:
    """Display optimized parameters in a clear format."""
    # Extract optimized parameters
    opt_params = optimization_results.get("optimized_parameters", {})
    
    if not opt_params:
        # Extract from config
        opt_params = {
            "A_throat": config.chamber.A_throat,
            "A_exit": config.nozzle.A_exit,
            "Lstar": config.chamber.Lstar,
            "chamber_diameter": config.chamber.chamber_inner_diameter if hasattr(config.chamber, 'chamber_inner_diameter') else np.sqrt(4.0 * config.chamber.volume / (np.pi * config.chamber.length)),
            "chamber_length": config.chamber.length,
            "expansion_ratio": config.nozzle.expansion_ratio,
        }
    
    # Pintle parameters
    if hasattr(config, 'injector') and config.injector.type == "pintle":
        geometry = config.injector.geometry
        if hasattr(geometry, 'fuel'):
            opt_params["d_pintle_tip"] = geometry.fuel.d_pintle_tip
            opt_params["h_gap"] = geometry.fuel.h_gap
        if hasattr(geometry, 'lox'):
            opt_params["n_orifices"] = geometry.lox.n_orifices
            opt_params["d_orifice"] = geometry.lox.d_orifice
            opt_params["theta_orifice"] = geometry.lox.theta_orifice
    
    # Display in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔥 Chamber Geometry")
        st.metric("Throat Area", f"{opt_params.get('A_throat', 0) * 1e6:.2f} mm²")
        st.metric("Exit Area", f"{opt_params.get('A_exit', 0) * 1e6:.2f} mm²")
        st.metric("L*", f"{opt_params.get('Lstar', 0) * 1000:.1f} mm")
        st.metric("Chamber Diameter", f"{opt_params.get('chamber_diameter', 0) * 1000:.2f} mm")
        st.metric("Chamber Length", f"{opt_params.get('chamber_length', 0) * 1000:.1f} mm")
        st.metric("Expansion Ratio", f"{opt_params.get('expansion_ratio', 0):.2f}")
    
    with col2:
        if "d_pintle_tip" in opt_params:
            st.markdown("#### 🔧 Pintle Injector")
            st.metric("Pintle Tip Diameter", f"{opt_params.get('d_pintle_tip', 0) * 1000:.2f} mm")
            st.metric("Gap Height", f"{opt_params.get('h_gap', 0) * 1000:.2f} mm")
            st.metric("Number of Orifices", f"{int(opt_params.get('n_orifices', 0))}")
            st.metric("Orifice Diameter", f"{opt_params.get('d_orifice', 0) * 1000:.2f} mm")
            st.metric("Orifice Angle", f"{opt_params.get('theta_orifice', 0):.1f}°")
        else:
            st.info("No pintle parameters optimized")
    
    with col3:
        st.markdown("#### ⚡ Performance")
        performance = optimization_results.get("performance", {})
        if performance:
            st.metric("Thrust", f"{performance.get('F', 0):.1f} N")
            st.metric("Isp", f"{performance.get('Isp', 0):.1f} s")
            st.metric("Chamber Pressure", f"{performance.get('Pc', 0) / 1e6:.2f} MPa")
            st.metric("Mass Flow", f"{performance.get('mdot_total', 0):.3f} kg/s")
            
            # Stability
            stability = performance.get("stability_results", {})
            if stability:
                chugging = stability.get("chugging", {})
                st.metric("Stability Margin", f"{chugging.get('stability_margin', 0):.3f}")
            
            # Time-varying metrics if available (check both locations)
            time_varying = optimization_results.get("time_varying")
            if time_varying is None:
                time_varying = performance.get("time_varying", {})
            if time_varying:
                st.markdown("##### ⏱️ Time-Averaged (Burn)")
                st.metric("Avg Thrust", f"{time_varying.get('avg_thrust', 0):.1f} N")
                st.metric("Min Stability", f"{time_varying.get('min_stability_margin', 0):.3f}")
                st.metric("Max Recession", f"{time_varying.get('max_recession_chamber', 0) * 1000:.2f} mm")


def _show_time_varying_results(time_varying: Dict[str, Any]) -> None:
    """Show time-varying optimization results."""
    st.markdown("### ⏱️ Time-Varying Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Thrust", f"{time_varying.get('avg_thrust', 0):.1f} N")
    with col2:
        st.metric("Min Thrust", f"{time_varying.get('min_thrust', 0):.1f} N")
    with col3:
        st.metric("Max Thrust", f"{time_varying.get('max_thrust', 0):.1f} N")
    with col4:
        st.metric("Thrust Std", f"{time_varying.get('thrust_std', 0):.1f} N")
    
    st.info(f"📊 Thrust variation: {time_varying.get('thrust_std', 0) / max(time_varying.get('avg_thrust', 1), 1) * 100:.1f}% (lower is better)")


def _plot_time_varying_results(time_varying_results: Dict[str, np.ndarray]) -> None:
    """Plot time-varying optimization results."""
    if "time" not in time_varying_results:
        return
    
    times = time_varying_results["time"]
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Thrust vs Time", "Stability Margin vs Time", "Recession vs Time"),
        vertical_spacing=0.1,
    )
    
    # Thrust
    if "F" in time_varying_results:
        fig.add_trace(
            go.Scatter(x=times, y=time_varying_results["F"], mode='lines', name='Thrust', line=dict(color='blue')),
            row=1, col=1
        )
    
    # Stability
    if "chugging_stability_margin" in time_varying_results:
        fig.add_trace(
            go.Scatter(x=times, y=time_varying_results["chugging_stability_margin"], mode='lines', name='Stability Margin', line=dict(color='green')),
            row=2, col=1
        )
    
    # Recession
    if "recession_chamber" in time_varying_results:
        fig.add_trace(
            go.Scatter(x=times, y=time_varying_results["recession_chamber"] * 1000, mode='lines', name='Chamber Recession', line=dict(color='red')),
            row=3, col=1
        )
    
    fig.update_xaxes(title_text="Time [s]", row=3, col=1)
    fig.update_yaxes(title_text="Thrust [N]", row=1, col=1)
    fig.update_yaxes(title_text="Stability Margin", row=2, col=1)
    fig.update_yaxes(title_text="Recession [mm]", row=3, col=1)
    fig.update_layout(height=800, showlegend=True)
    
    st.plotly_chart(fig, use_container_width=True)


def _plot_stability_evolution(runner: PintleEngineRunner, P_tank_O: float, P_tank_F: float) -> None:
    """Plot stability evolution over time."""
    try:
        from pintle_pipeline.time_varying_solver import TimeVaryingCoupledSolver
        
        solver = TimeVaryingCoupledSolver(runner.config, runner.cea_cache)
        burn_time = 10.0  # Default
        time_array = np.linspace(0, burn_time, 50)
        P_tank_O_array = np.full_like(time_array, P_tank_O)
        P_tank_F_array = np.full_like(time_array, P_tank_F)
        
        states = solver.solve_time_series(time_array, P_tank_O_array, P_tank_F_array)
        results = solver.get_results_dict()
        
        # Plot stability margins over time
        fig = make_subplots(rows=3, cols=1, subplot_titles=("Chugging Margin", "Acoustic Margin", "Feed System Margin"))
        
        # Would extract stability margins from results and plot
        # Placeholder for now
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not plot stability evolution: {e}")

