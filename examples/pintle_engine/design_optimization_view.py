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
# MODULAR OPTIMIZATION LAYERS (refactored)
# These modules contain the extracted layer logic for better maintainability
# =============================================================================
from optimization_layers import (
    # Helpers - used directly throughout this file
    generate_segmented_pressure_curve,
    segments_from_optimizer_vars,
    optimizer_vars_from_segments,
    # Layer functions
    create_layer1_apply_x_to_config,
    run_layer2_burn_candidate,
    run_layer3_thermal_protection,
    run_layer4_flight_simulation,
    # Display functions
    plot_pressure_curves,
    plot_copv_pressure,
    plot_flight_trajectory,
    plot_optimization_convergence,
    plot_time_varying_results,
    # COPV and flight helpers
    calculate_copv_pressure_curve,
    run_flight_simulation,
    # Utilities
    extract_all_parameters,
    # Main optimizer
    run_full_engine_optimization_with_flight_sim,
)

# Import UI tab functions from views
from optimization_layers.views import (
    _design_requirements_tab,
    _full_engine_optimization_tab,
    _injector_optimization_tab,
    _chamber_optimization_tab,
    _stability_analysis_tab,
    _flight_performance_tab,
    _results_export_tab,
)

# Alias for backward compatibility with internal function names
_calculate_copv_pressure_curve = calculate_copv_pressure_curve
_run_flight_simulation = run_flight_simulation
_extract_all_parameters = extract_all_parameters
_run_full_engine_optimization_with_flight_sim = run_full_engine_optimization_with_flight_sim


# =============================================================================
# HELPER FUNCTIONS - PLOTTING AND VISUALIZATION
# (Helper functions moved to optimization_layers/helpers.py)
# =============================================================================

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
    
    st.plotly_chart(fig, use_container_width=True, key="optimization_history_plot")
    
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


# =============================================================================
# =============================================================================
# MAIN UI FUNCTIONS
# =============================================================================
# =============================================================================

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
                plot_time_varying_results(opt_results["time_varying_results"])
        
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
                    st.plotly_chart(fig, use_container_width=True, key="geometry_sizing_plot")
                    
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
                                if L_chamber > 0 and V_chamber > 0:
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

