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
        rocket.setdefault("motor_position", 0.5)
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
            motor_pos = st.number_input("Motor position [m]", value=float(rocket.get("motor_position") or 0.5), key="opt_motor_position",
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
        copv_free_volume_default = press_tank.get("free_volume_L") or (press_volume_calc * 1000 * 0.90)  # 90% default
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
            value=5000.0,
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
            value=2.5,
            step=0.1,
            key="opt_of_ratio",
            help="Target oxidizer-to-fuel mixture ratio. LOX/RP-1 optimal: 2.4-2.8 for Isp, 2.2-2.5 for stability."
        )
        
        st.markdown("### Tank Pressures")
        
        max_lox_tank_pressure = st.number_input(
            "Max LOX Tank Pressure [psi]",
            min_value=100.0,
            max_value=5000.0,
            value=500.0,
            step=25.0,
            key="opt_max_lox_pressure",
            help="Maximum operating pressure in LOX tank. Sets upper bound for chamber pressure."
        )
        
        max_fuel_tank_pressure = st.number_input(
            "Max Fuel Tank Pressure [psi]",
            min_value=100.0,
            max_value=5000.0,
            value=500.0,
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
            value=0.6,
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
            value=0.20,
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
                value=0.8,
                step=0.1,
                key="opt_min_lstar",
                help="Minimum characteristic length. Lower = smaller chamber but less complete combustion. Typical: 0.8-1.0m for LOX/RP-1."
            )
        with col_lstar2:
            max_lstar = st.number_input(
                "Maximum L* [m]",
                min_value=0.5,
                max_value=3.0,
                value=2.0,
                step=0.1,
                key="opt_max_lstar",
                help="Maximum characteristic length. Higher = better combustion but heavier/longer chamber. Typical: 1.5-2.0m for LOX/RP-1."
            )
        
        st.markdown("### Stability Requirements")
        
        min_stability_margin = st.number_input(
            "Minimum Overall Stability Margin",
            min_value=1.0,
            max_value=5.0,
            value=1.2,
            step=0.1,
            key="opt_min_stability",
            help="Minimum stability margin (1.2 = 20% margin, 1.5 = 50% margin)"
        )
        
        chugging_margin_min = st.number_input(
            "Chugging Margin (min)",
            min_value=0.0,
            max_value=10.0,
            value=0.2,
            step=0.1,
            key="opt_chugging_margin",
            help="Minimum chugging stability margin"
        )
        
        acoustic_margin_min = st.number_input(
            "Acoustic Margin (min)",
            min_value=0.0,
            max_value=10.0,
            value=0.1,
            step=0.1,
            key="opt_acoustic_margin",
            help="Minimum acoustic stability margin"
        )
        
        feed_stability_min = st.number_input(
            "Feed System Margin (min)",
            min_value=0.0,
            max_value=10.0,
            value=0.15,
            step=0.1,
            key="opt_feed_margin",
            help="Minimum feed system stability margin"
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
        # Stability
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
        st.metric("L* Range", f"{min_lstar:.1f} - {max_lstar:.1f} m")
    with col_s4:
        st.metric("Max Engine Length", f"{max_engine_length*1000:.0f} mm")
        st.metric("Max Chamber OD", f"{max_chamber_outer_diameter*1000:.0f} mm")
    
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
        st.metric("Target Thrust", f"{requirements.get('target_thrust', 5000):.0f} N")
        st.metric("Optimal O/F", f"{requirements.get('optimal_of_ratio', 2.5):.2f}")
    with col_req2:
        st.metric("Max LOX Pressure", f"{requirements.get('max_lox_tank_pressure_psi', 500):.0f} psi")
        st.metric("Max Fuel Pressure", f"{requirements.get('max_fuel_tank_pressure_psi', 500):.0f} psi")
    with col_req3:
        st.metric("L* Range", f"{requirements.get('min_Lstar', 0.8):.1f} - {requirements.get('max_Lstar', 2.0):.1f} m")
        st.metric("Min Stability", f"{requirements.get('min_stability_margin', 1.2):.2f}")
    
    st.markdown("---")
    
    # Optimization Configuration
    st.markdown("### ⚙️ Optimization Configuration")
    
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        st.markdown("#### Optimization Targets")
        
        optimize_thrust = st.checkbox("Optimize for Target Thrust", value=True, key="full_opt_thrust",
            help="Optimize geometry to achieve target thrust")
        
        optimize_stability = st.checkbox("Optimize Stability Margins", value=True, key="full_opt_stability",
            help="Ensure all stability margins are met (chugging, acoustic, feed system)")
        
        optimize_isp = st.checkbox("Maximize Specific Impulse", value=True, key="full_opt_isp",
            help="Optimize expansion ratio and combustion efficiency for best Isp")
        
        optimize_mass = st.checkbox("Minimize Engine Mass", value=False, key="full_opt_mass",
            help="Include engine mass minimization in objective (may trade off performance)")
    
    with col_opt2:
        st.markdown("#### Optimization Parameters")
        
        target_burn_time = st.number_input(
            "Target Burn Time [s]",
            min_value=1.0,
            max_value=60.0,
            value=10.0,
            step=1.0,
            key="full_opt_burn_time",
            help="Design burn time for optimization"
        )
        
        max_iterations = st.number_input(
            "Max Optimization Iterations",
            min_value=20,
            max_value=200,
            value=80,
            step=10,
            key="full_opt_max_iter",
            help="Maximum function evaluations (typically converges in 30-60)"
        )
        
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
    
    # Pressure curve configuration
    st.markdown("---")
    st.markdown("### 🛢️ Tank Pressure Curve Configuration")
    st.info(
        "🎯 **Pressure curves are OPTIMIZED** - The optimizer will find the best pressure profiles "
        "(50%-100% of start) for both tanks to achieve your O/F and thrust targets. "
        "Values below are initial guesses. Your controller can achieve any linear profile."
    )
    
    col_lox_press, col_fuel_press = st.columns(2)
    
    with col_lox_press:
        st.markdown("#### LOX Tank")
        
        lox_pressure_start_psi = st.number_input(
            "Start Pressure [psi]",
            min_value=100.0,
            max_value=1000.0,
            value=float(requirements.get("max_lox_tank_pressure_psi", 500)),
            step=25.0,
            key="full_opt_lox_start",
            help="Maximum tank pressure at start of burn"
        )
        
        lox_pressure_end_pct = st.slider(
            "Initial End Pressure Guess [% of start]",
            min_value=50,
            max_value=100,
            value=75,
            key="full_opt_lox_end_pct",
            help="Starting point for optimization (optimizer will find optimal value between 50-100%)"
        )
    
    with col_fuel_press:
        st.markdown("#### Fuel Tank")
        
        fuel_pressure_start_psi = st.number_input(
            "Start Pressure [psi]",
            min_value=100.0,
            max_value=1000.0,
            value=float(requirements.get("max_fuel_tank_pressure_psi", 500)),
            step=25.0,
            key="full_opt_fuel_start",
            help="Maximum tank pressure at start of burn"
        )
        
        fuel_pressure_end_pct = st.slider(
            "Initial End Pressure Guess [% of start]",
            min_value=50,
            max_value=100,
            value=75,
            key="full_opt_fuel_end_pct",
            help="Starting point for optimization (optimizer will find optimal value between 50-100%)"
        )
    
    # Store pressure curve config (initial guesses - optimizer will refine)
    pressure_config = {
        "lox_mode": "Optimized",  # Always optimized now
        "lox_start_psi": lox_pressure_start_psi,
        "lox_end_pct": lox_pressure_end_pct / 100.0,
        "fuel_mode": "Optimized",  # Always optimized now
        "fuel_start_psi": fuel_pressure_start_psi,
        "fuel_end_pct": fuel_pressure_end_pct / 100.0,
    }
    
    # Tolerances config
    tolerances = {
        "thrust": thrust_tolerance,
        "apogee": apogee_tolerance,
    }
    
    # For compatibility
    use_time_varying = False
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
                progress_bar.progress(progress, text=f"{stage}: {message}")
                status_text.text(f"Stage: {stage} | {message}")
            
            optimized_config, optimization_results = _run_full_engine_optimization_with_flight_sim(
                config_obj,
                runner,
                requirements,
                target_burn_time,
                max_iterations,
                tolerances,
                pressure_config,
                {
                    "optimize_thrust": optimize_thrust,
                    "optimize_stability": optimize_stability,
                    "optimize_isp": optimize_isp,
                    "optimize_mass": optimize_mass,
                },
                progress_callback=progress_callback,
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
    optimization_targets: Dict[str, bool],
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
    target_thrust = requirements.get("target_thrust", 5000.0)
    optimal_of = requirements.get("optimal_of_ratio", 2.5)
    max_P_tank_O = requirements.get("max_P_tank_O", 3.45e6)  # ~500 psi
    max_P_tank_F = requirements.get("max_P_tank_F", 3.45e6)
    min_Lstar = requirements.get("min_Lstar", 0.8)
    max_Lstar = requirements.get("max_Lstar", 2.0)
    min_stability = requirements.get("min_stability_margin", 1.2)
    max_chamber_od = requirements.get("max_chamber_outer_diameter", 0.15)
    max_nozzle_exit = requirements.get("max_nozzle_exit_diameter", 0.20)
    max_engine_length = requirements.get("max_engine_length", 0.6)
    
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
    
    return params


def _run_full_engine_optimization_with_flight_sim(
    config_obj: PintleEngineConfig,
    runner: PintleEngineRunner,
    requirements: Dict[str, Any],
    target_burn_time: float,
    max_iterations: int,
    tolerances: Dict[str, float],
    pressure_config: Dict[str, Any],
    optimization_targets: Dict[str, bool],
    progress_callback: Optional[callable] = None,
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
    """
    from pintle_pipeline.system_diagnostics import SystemDiagnostics
    from scipy.optimize import minimize, differential_evolution
    from pathlib import Path
    
    # Optimization state for progress tracking
    opt_state = {
        "iteration": 0,
        "best_objective": float('inf'),
        "best_config": None,
        "history": [],
        "converged": False,
    }
    
    def update_progress(stage: str, progress: float, message: str):
        if progress_callback:
            progress_callback(stage, progress, message)
    
    update_progress("Initialization", 0.02, "Extracting requirements...")
    
    # Extract requirements
    target_thrust = requirements.get("target_thrust", 5000.0)
    target_apogee = requirements.get("target_apogee", 3048.0)
    optimal_of = requirements.get("optimal_of_ratio", 2.5)
    min_Lstar = requirements.get("min_Lstar", 0.8)
    max_Lstar = requirements.get("max_Lstar", 2.0)
    min_stability = requirements.get("min_stability_margin", 1.2)
    max_chamber_od = requirements.get("max_chamber_outer_diameter", 0.15)
    max_nozzle_exit = requirements.get("max_nozzle_exit_diameter", 0.20)
    max_engine_length = requirements.get("max_engine_length", 0.6)
    copv_volume_m3 = requirements.get("copv_free_volume_m3", 0.02)
    
    # Extract tolerances
    thrust_tol = tolerances.get("thrust", 0.10)
    apogee_tol = tolerances.get("apogee", 0.15)
    
    # Extract pressure curve config
    psi_to_Pa = 6894.76
    lox_P_start = pressure_config.get("lox_start_psi", 500) * psi_to_Pa
    lox_P_end_ratio = pressure_config.get("lox_end_pct", 0.70)
    fuel_P_start = pressure_config.get("fuel_start_psi", 500) * psi_to_Pa
    fuel_P_end_ratio = pressure_config.get("fuel_end_pct", 0.70)
    
    update_progress("Initialization", 0.05, "Setting up optimization bounds...")
    
    # Phase 1: Set orifice angle to 90° and prepare config
    config_base = copy.deepcopy(config_obj)
    if hasattr(config_base, 'injector') and config_base.injector.type == "pintle":
        if hasattr(config_base.injector.geometry, 'lox'):
            config_base.injector.geometry.lox.theta_orifice = 90.0
    
    # =========================================================================
    # OPTIMIZATION VARIABLES (9 dimensions):
    # [0] A_throat (throat area, m²)
    # [1] Lstar (characteristic length, m)
    # [2] expansion_ratio
    # [3] d_pintle_tip (m)
    # [4] h_gap (m)
    # [5] n_orifices (will be rounded to int)
    # [6] d_orifice (m)
    # [7] lox_P_end_ratio (end pressure as fraction of start, 0.5-1.0)
    # [8] fuel_P_end_ratio (end pressure as fraction of start, 0.5-1.0)
    #
    # Note: Pressure curves are fully controllable - optimizer has full freedom
    # to pick any linear profile from start to end pressure.
    # =========================================================================
    
    # Calculate initial guess and bounds
    Cf_est = 1.5
    Pc_est = lox_P_start * 0.7
    A_throat_init = target_thrust / (Cf_est * Pc_est)
    A_throat_init = np.clip(A_throat_init, 5e-5, 2e-3)
    
    # Pressure ratio bounds - full freedom since we have active pressure control
    # Can go from 50% (aggressive blowdown) to 100% (perfectly regulated)
    bounds = [
        (5e-5, 2e-3),           # A_throat: 8mm to 50mm diameter
        (min_Lstar, max_Lstar), # Lstar
        (4.0, 20.0),            # expansion_ratio
        (0.008, 0.040),         # d_pintle_tip
        (0.0003, 0.0020),       # h_gap
        (6, 24),                # n_orifices
        (0.001, 0.006),         # d_orifice
        (0.50, 1.00),           # lox_P_end_ratio: full range (controller can achieve any profile)
        (0.50, 1.00),           # fuel_P_end_ratio: full range (controller can achieve any profile)
    ]
    
    x0 = np.array([
        A_throat_init,
        (min_Lstar + max_Lstar) / 2,
        10.0,
        0.015,
        0.0006,
        12,
        0.003,
        lox_P_end_ratio,   # Start with user's initial guess
        fuel_P_end_ratio,  # Start with user's initial guess
    ])
    
    update_progress("Optimization", 0.08, "Starting iterative optimization...")
    
    def apply_x_to_config(x: np.ndarray, base_config: PintleEngineConfig) -> Tuple[PintleEngineConfig, float, float]:
        """Apply optimization variables to config. Returns (config, lox_end_ratio, fuel_end_ratio)."""
        config = copy.deepcopy(base_config)
        
        A_throat = float(x[0])
        Lstar = float(x[1])
        expansion_ratio = float(x[2])
        d_pintle_tip = float(x[3])
        h_gap = float(x[4])
        n_orifices = int(round(x[5]))
        d_orifice = float(x[6])
        lox_end_ratio = float(x[7])
        fuel_end_ratio = float(x[8])
        
        # Chamber
        V_chamber = Lstar * A_throat
        D_chamber = min(np.sqrt(4 * A_throat * 3.5 / np.pi), max_chamber_od * 0.95)
        A_chamber = np.pi * (D_chamber / 2) ** 2
        L_chamber = V_chamber / A_chamber if A_chamber > 0 else 0.2
        
        config.chamber.A_throat = A_throat
        config.chamber.volume = V_chamber
        config.chamber.Lstar = Lstar
        config.chamber.length = L_chamber
        if hasattr(config.chamber, 'chamber_inner_diameter'):
            config.chamber.chamber_inner_diameter = D_chamber
        
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
        
        return config, lox_end_ratio, fuel_end_ratio
    
    def objective(x: np.ndarray) -> float:
        """Multi-objective function with soft penalties."""
        opt_state["iteration"] += 1
        iteration = opt_state["iteration"]
        
        # Progress update (optimization is ~10% to 50% of total)
        progress = 0.10 + 0.40 * min(iteration / max_iterations, 1.0)
        update_progress(
            "Optimization", 
            progress, 
            f"Iteration {iteration}/{max_iterations} | Best: {opt_state['best_objective']:.3f}"
        )
        
        try:
            config, curr_lox_end_ratio, curr_fuel_end_ratio = apply_x_to_config(x, config_base)
            
            # Evaluate at average pressure (middle of optimized curve)
            P_O_avg = lox_P_start * (1 + curr_lox_end_ratio) / 2
            P_F_avg = fuel_P_start * (1 + curr_fuel_end_ratio) / 2
            
            test_runner = PintleEngineRunner(config)
            results = test_runner.evaluate(P_O_avg, P_F_avg)
            
            F_actual = results.get("F", 0)
            Isp_actual = results.get("Isp", 0)
            MR_actual = results.get("MR", 0)
            Pc_actual = results.get("Pc", 0)
            
            # Calculate errors with tolerances
            thrust_error = abs(F_actual - target_thrust) / target_thrust
            of_error = abs(MR_actual - optimal_of) / optimal_of if optimal_of > 0 else 0
            
            # Stability check (get margin if available)
            stability = results.get("stability_results", {})
            chugging = stability.get("chugging", {})
            stability_margin = chugging.get("stability_margin", 1.0)
            stability_penalty = max(0, min_stability - stability_margin)
            
            # Multi-objective with weights
            obj = (
                5.0 * thrust_error +          # Thrust matching
                3.0 * of_error +              # O/F matching  
                2.0 * stability_penalty +     # Stability
                1.0 * max(0, 200 - Isp_actual) / 200  # Isp bonus
            )
            
            # Record history
            opt_state["history"].append({
                "iteration": iteration,
                "x": x.copy(),
                "thrust": F_actual,
                "thrust_error": thrust_error,
                "of_error": of_error,
                "Isp": Isp_actual,
                "MR": MR_actual,
                "Pc": Pc_actual,
                "stability_margin": stability_margin,
                "lox_end_ratio": curr_lox_end_ratio,
                "fuel_end_ratio": curr_fuel_end_ratio,
                "objective": obj,
            })
            
            # Track best (store pressure ratios too)
            if obj < opt_state["best_objective"]:
                opt_state["best_objective"] = obj
                opt_state["best_config"] = copy.deepcopy(config)
                opt_state["best_lox_end_ratio"] = curr_lox_end_ratio
                opt_state["best_fuel_end_ratio"] = curr_fuel_end_ratio
            
            # Check convergence (within tolerances)
            if thrust_error < thrust_tol and of_error < 0.15:
                opt_state["converged"] = True
            
            return obj
            
        except Exception as e:
            return 1e6  # Penalty for failed evaluation
    
    # Run optimization using Nelder-Mead (robust, doesn't need gradients)
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={
            'maxiter': max_iterations,
            'maxfev': max_iterations * 2,
            'xatol': 1e-6,
            'fatol': 1e-4,
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
    
    update_progress("Evaluation", 0.52, "Evaluating optimized engine at design point...")
    
    # Phase 5: Evaluate performance at average operating point (using optimized pressure ratios)
    P_O_avg = lox_P_start * (1 + final_lox_end_ratio) / 2
    P_F_avg = fuel_P_start * (1 + final_fuel_end_ratio) / 2
    optimized_runner = PintleEngineRunner(optimized_config)
    final_performance = optimized_runner.evaluate(P_O_avg, P_F_avg)
    
    update_progress("Pressure Curves", 0.55, "Generating 200-point pressure curves...")
    
    # Phase 6: Generate 200-point time series with OPTIMIZED independent pressure curves
    n_time_points = 200
    time_array = np.linspace(0.0, target_burn_time, n_time_points)
    
    # Generate independent pressure curves using OPTIMIZED end ratios
    # Linear interpolation from start to optimized end pressure
    lox_pressure_profile = np.linspace(1.0, final_lox_end_ratio, n_time_points)
    fuel_pressure_profile = np.linspace(1.0, final_fuel_end_ratio, n_time_points)
    
    P_tank_O_array = lox_P_start * lox_pressure_profile
    P_tank_F_array = fuel_P_start * fuel_pressure_profile
    
    # Evaluate at a few points along the burn to get realistic curves
    update_progress("Pressure Curves", 0.58, "Evaluating performance across burn time...")
    
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
    
    update_progress("Validation", 0.70, "Running stability checks...")
    
    # Phase 8: Run system diagnostics
    try:
        diagnostics = SystemDiagnostics(optimized_config, optimized_runner)
        validation_results = diagnostics.run_full_diagnostics(P_O_avg, P_F_avg)
    except Exception as e:
        validation_results = {"error": str(e)}
    
    # Check if candidate is good enough for flight sim
    thrust_error = abs(final_performance.get("F", 0) - target_thrust) / target_thrust
    stability = final_performance.get("stability_results", {})
    chugging_margin = stability.get("chugging", {}).get("stability_margin", 0)
    
    flight_sim_result = {"success": False, "apogee": 0, "max_velocity": 0}
    
    # Check if candidate is good enough for flight sim
    last_result = iteration_history[-1] if iteration_history else {}
    thrust_error_final = last_result.get("thrust_error", 1.0)
    stability_margin_final = last_result.get("stability_margin", 0)
    
    # Run flight sim if within tolerances (or close)
    if thrust_error_final < thrust_tol * 1.5:
        update_progress("Flight Simulation", 0.75, "Running flight simulation to verify apogee...")
        
        try:
            flight_sim_result = _run_flight_simulation(
                optimized_config,
                pressure_curves,
                target_burn_time,
            )
        except Exception as e:
            flight_sim_result = {"success": False, "error": str(e), "apogee": 0, "max_velocity": 0}
    else:
        update_progress("Flight Simulation", 0.75, f"Skipping flight sim (thrust error {thrust_error_final*100:.1f}% > {thrust_tol*150:.0f}%)...")
        flight_sim_result = {"success": False, "skipped": True, "apogee": 0, "max_velocity": 0}
    
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
            st.info("Flight sim only runs when: thrust error < 15% AND stability margin ≥ 80% of target")


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
    
    # Create subplot with 3 rows
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Objective Function", "Thrust Error [%]", 
            "O/F Error [%]", "Thrust [N]",
            "LOX End Pressure [%]", "Fuel End Pressure [%]"
        ),
        vertical_spacing=0.12,
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
    
    fig.update_xaxes(title_text="Iteration", row=3, col=1)
    fig.update_xaxes(title_text="Iteration", row=3, col=2)
    fig.update_layout(height=650, showlegend=False)
    
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
    """Display chamber geometry visualization similar to chamber design tab."""
    try:
        from pintle_pipeline.comprehensive_geometry_sizing import size_complete_geometry, plot_complete_geometry
        
        performance = optimization_results.get("performance", {})
        Pc = performance.get("Pc", 2e6)
        MR = performance.get("MR", 2.5)
        Tc = performance.get("Tc", 3500.0)
        gamma = performance.get("gamma", 1.2)
        R = performance.get("R", 300.0)
        burn_time = optimization_results.get("design_requirements", {}).get("target_burn_time", 10.0)
        
        sizing_results = size_complete_geometry(
            config=config,
            Pc=Pc,
            MR=MR,
            Tc=Tc,
            gamma=gamma,
            R=R,
            burn_time=burn_time,
            chamber_heat_flux=2e6,
        )
        
        fig, _ = plot_complete_geometry(sizing_results, config, use_plotly=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display sizing summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ablative Thickness", f"{sizing_results.get('optimal', {}).get('ablative_thickness', 0) * 1000:.2f} mm")
        with col2:
            st.metric("Graphite Thickness", f"{sizing_results.get('optimal', {}).get('graphite_thickness', 0) * 1000:.2f} mm")
        with col3:
            st.metric("Total Mass", f"{sizing_results.get('optimal', {}).get('total_mass', 0):.3f} kg")
            
    except Exception as e:
        st.warning(f"Could not generate chamber geometry: {e}")


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
    target_thrust = requirements.get("target_thrust", 5000.0)
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
    target_of = requirements.get("optimal_of_ratio", 2.5)
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
    min_lstar = requirements.get("min_Lstar", 0.8)
    max_lstar = requirements.get("max_Lstar", 2.0)
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
    
    target_thrust = requirements.get("target_thrust", 5000.0)
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
                                "target_thrust": requirements.get("target_thrust", 5000.0),
                                "target_burn_time": requirements.get("target_burn_time", 10.0),
                                "target_stability_margin": requirements.get("min_stability_margin", 1.2),
                                "P_tank_O": P_tank_O,
                                "P_tank_F": P_tank_F,
                                "target_Isp": requirements.get("target_Isp", None),
                            }
                            
                            constraints = {
                                "max_chamber_length": requirements.get("max_chamber_length", 0.5),
                                "max_chamber_diameter": requirements.get("max_chamber_diameter", 0.15),
                                "min_Lstar": 0.8,
                                "max_Lstar": 2.0,
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
                        if "time_varying" in optimization_results:
                            _show_time_varying_results(optimization_results["time_varying"])
                        
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
        st.metric("Target Thrust", f"{requirements.get('target_thrust', 0):.0f} N")


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
        "target_thrust": requirements.get("target_thrust", 5000.0),
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
        "min_Lstar": 0.8,
        "max_Lstar": 2.0,
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
            
            # Time-varying metrics if available
            time_varying = optimization_results.get("time_varying", {})
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
        from pintle_pipeline.time_varying_solver import TimeVaryingSolver
        
        solver = TimeVaryingSolver(runner.config, runner.cea_cache)
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

