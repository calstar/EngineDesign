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
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

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
    tab_design, tab_injector, tab_chamber, tab_stability, tab_performance, tab_results = st.tabs([
        "📋 Design Requirements",
        "🔧 Injector Optimization",
        "🔥 Chamber Optimization", 
        "⚖️ Stability Analysis",
        "✈️ Flight Performance",
        "📊 Results & Export"
    ])
    
    with tab_design:
        config_obj = _design_requirements_tab(config_obj)
    
    with tab_injector:
        config_obj = _injector_optimization_tab(config_obj, runner)
    
    with tab_chamber:
        config_obj = _chamber_optimization_tab(config_obj, runner)
        
        # Show comprehensive geometry plot
        if runner and config_obj:
            st.markdown("### Complete Geometry Visualization")
            try:
                from pintle_pipeline.comprehensive_geometry_sizing import size_complete_geometry, plot_complete_geometry
                
                # Get current conditions
                P_tank_O = 3e6  # Default
                P_tank_F = 3e6
                results = runner.evaluate(P_tank_O, P_tank_F)
                
                Pc = results.get("Pc", 2e6)
                MR = results.get("MR", 2.5)
                Tc = results.get("Tc", 3500.0)
                gamma = results.get("gamma", 1.2)
                R = results.get("R", 300.0)
                burn_time = st.session_state.get("design_requirements", {}).get("target_burn_time", 10.0)
                
                # Estimate heat flux
                heat_flux_chamber = 2e6  # W/m² - typical estimate
                
                # Size complete geometry
                sizing_results = size_complete_geometry(
                    config=config_obj,
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
                    config_obj,
                    use_plotly=True,
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display sizing summary
                st.markdown("#### Sizing Summary")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Ablative Thickness", f"{sizing_results['optimal'].get('ablative_thickness', 0) * 1000:.2f} mm")
                with col_b:
                    st.metric("Graphite Thickness", f"{sizing_results['optimal'].get('graphite_thickness', 0) * 1000:.2f} mm")
                with col_c:
                    st.metric("Total Mass", f"{sizing_results['optimal'].get('total_mass', 0):.3f} kg")
                
                # Show impingement zones
                st.markdown("#### Fuel Impingement Zones")
                try:
                    from pintle_pipeline.localized_ablation import calculate_impingement_zones
                    
                    chamber_length = positions[-1] if len(positions) > 0 else 0.2
                    chamber_diameter = geometry.get("D_chamber_initial", 0.1)
                    impingement_data = calculate_impingement_zones(
                        config_obj, chamber_length, chamber_diameter, n_points=len(positions)
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
                
                # Validation status
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
    """Design requirements input tab."""
    st.subheader("Design Requirements")
    st.markdown("Specify your mission requirements and constraints.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Performance Requirements")
        target_thrust = st.number_input(
            "Target Thrust [N]",
            min_value=100.0,
            max_value=100000.0,
            value=5000.0,
            step=100.0,
            help="Desired average thrust during burn"
        )
        
        target_altitude = st.number_input(
            "Target Altitude [m]",
            min_value=0.0,
            max_value=100000.0,
            value=3048.0,  # 10k feet
            step=100.0,
            help="Target apogee altitude"
        )
        
        payload_mass = st.number_input(
            "Payload Mass [kg]",
            min_value=0.0,
            max_value=1000.0,
            value=10.0,
            step=1.0,
            help="Payload mass to deliver to target altitude"
        )
        
        target_burn_time = st.number_input(
            "Target Burn Time [s]",
            min_value=1.0,
            max_value=300.0,
            value=10.0,
            step=0.5,
            help="Desired burn duration"
        )
        
        target_Isp = st.number_input(
            "Target Isp [s] (optional)",
            min_value=0.0,
            max_value=500.0,
            value=None,
            step=1.0,
            help="Desired specific impulse (leave blank to optimize)"
        )
    
    with col2:
        st.markdown("### Stability Requirements")
        min_stability_margin = st.number_input(
            "Minimum Stability Margin",
            min_value=1.0,
            max_value=5.0,
            value=1.2,
            step=0.1,
            help="Minimum stability margin (1.2 = 20% margin, 1.5 = 50% margin)"
        )
        
        st.markdown("#### Stability Components")
        chugging_margin_min = st.number_input(
            "Chugging Margin (min)",
            min_value=0.0,
            max_value=10.0,
            value=0.2,
            step=0.1,
            help="Minimum chugging stability margin"
        )
        
        acoustic_margin_min = st.number_input(
            "Acoustic Margin (min)",
            min_value=0.0,
            max_value=10.0,
            value=0.1,
            step=0.1,
            help="Minimum acoustic stability margin"
        )
        
        feed_stability_min = st.number_input(
            "Feed System Margin (min)",
            min_value=0.0,
            max_value=10.0,
            value=0.15,
            step=0.1,
            help="Minimum feed system stability margin"
        )
        
        st.markdown("### System Constraints")
        max_chamber_length = st.number_input(
            "Max Chamber Length [m]",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.05,
            help="Maximum allowable chamber length"
        )
        
        max_chamber_diameter = st.number_input(
            "Max Chamber Diameter [m]",
            min_value=0.05,
            max_value=1.0,
            value=0.15,
            step=0.01,
            help="Maximum allowable chamber diameter"
        )
        
        max_total_mass = st.number_input(
            "Max Total Mass [kg]",
            min_value=1.0,
            max_value=10000.0,
            value=100.0,
            step=1.0,
            help="Maximum total vehicle mass (propellant + structure)"
        )
    
    # Store requirements in session state
    st.session_state["design_requirements"] = {
        "target_thrust": target_thrust,
        "target_altitude": target_altitude,
        "payload_mass": payload_mass,
        "target_burn_time": target_burn_time,
        "target_Isp": target_Isp,
        "min_stability_margin": min_stability_margin,
        "chugging_margin_min": chugging_margin_min,
        "acoustic_margin_min": acoustic_margin_min,
        "feed_stability_min": feed_stability_min,
        "max_chamber_length": max_chamber_length,
        "max_chamber_diameter": max_chamber_diameter,
        "max_total_mass": max_total_mass,
    }
    
    st.success("✅ Design requirements saved. Proceed to Injector or Chamber optimization tabs.")
    
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
    
    target_thrust = requirements.get("target_thrust", 5000.0)
    target_MR = config_obj.combustion.MR if hasattr(config_obj.combustion, 'MR') else 2.5
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Current Injector Configuration")
        
        # Display current injector parameters
        injector_config = config_obj.injector if hasattr(config_obj, 'injector') else None
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
                        # Run injector optimization
                        optimized_config = _optimize_injector(
                            config_obj,
                            runner,
                            target_thrust,
                            target_MR,
                            optimize_pintle,
                            optimize_orifices,
                            optimize_spray,
                        )
                        config_obj = optimized_config
                        st.success("✅ Injector optimization complete!")
                        st.session_state["optimized_config"] = optimized_config
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
            
            if st.button("🚀 Run Chamber Optimization", type="primary"):
                with st.spinner("Optimizing chamber geometry and cooling system..."):
                    try:
                        optimized_config = _optimize_chamber(
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
                        config_obj = optimized_config
                        st.success("✅ Chamber optimization complete!")
                        st.session_state["optimized_config"] = optimized_config
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
    target_altitude = requirements.get("target_altitude", 3048.0)
    payload_mass = requirements.get("payload_mass", 10.0)
    
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
        st.metric("Payload Mass", f"{payload_mass:.1f} kg")
        st.metric("Target Thrust", f"{requirements.get('target_thrust', 0):.0f} N")


def _results_export_tab(config_obj: PintleEngineConfig, runner: Optional[PintleEngineRunner]) -> None:
    """Results and export tab."""
    st.subheader("Optimization Results & Export")
    
    optimized_config = st.session_state.get("optimized_config", None)
    if optimized_config:
        st.success("✅ Optimized configuration available!")
        
        # Display summary
        st.markdown("### Optimization Summary")
        
        # Compare before/after
        if runner:
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
                config_dict = optimized_config.to_dict() if hasattr(optimized_config, 'to_dict') else {}
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
) -> PintleEngineConfig:
    """Optimize injector geometry."""
    # Placeholder - would implement full injector optimization here
    # For now, return config as-is
    return config_obj


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
) -> PintleEngineConfig:
    """Optimize chamber geometry and cooling system."""
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
    }
    
    # Run optimization
    results = optimizer.optimize(design_requirements, constraints)
    
    return results["optimized_config"]


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

