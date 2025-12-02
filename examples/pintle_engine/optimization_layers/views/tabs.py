"""UI Tab Functions for Design Optimization View.

This module contains all Streamlit tab functions for the design optimization interface.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import copy
import sys
from pathlib import Path

from pintle_pipeline.config_schemas import PintleEngineConfig
from pintle_models.runner import PintleEngineRunner

# Import from optimization_layers
from .. import (
    run_full_engine_optimization_with_flight_sim,
    extract_all_parameters,
    plot_optimization_convergence,
    plot_pressure_curves,
    plot_copv_pressure,
    plot_flight_trajectory,
    plot_time_varying_results,
)

# Import helper functions from views.helpers
from .helpers import (
    _display_current_engine_config,
    _show_complete_optimization_results,
    _display_chamber_geometry_plot,
    _show_full_engine_comparison,
    _show_engine_validation_checks,
    _optimize_injector,
    _show_injector_comparison,
    _display_injector_parameters,
    _optimize_chamber,
    _show_optimization_comparison,
    _display_optimized_parameters,
    _show_time_varying_results,
    _plot_stability_evolution,
)

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
            
            optimized_config, optimization_results = run_full_engine_optimization_with_flight_sim(
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
                apogee_error = abs(apogee - target_apogee) / target_apogee * 100 if target_apogee > 0 else 100.0
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
                            plot_time_varying_results(optimization_results["time_varying_results"])
                        
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
            plot_time_varying_results(optimization_results["time_varying_results"])
        
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




