"""Display and plotting functions for optimization results.

This module contains functions for visualizing optimization results,
pressure curves, flight trajectories, and engine comparisons.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from pintle_pipeline.config_schemas import PintleEngineConfig


def plot_pressure_curves(pressure_curves: Dict[str, np.ndarray]) -> None:
    """Plot tank pressure and performance curves."""
    time = pressure_curves.get("time", np.array([]))
    if len(time) == 0:
        st.warning("No pressure curve data available")
        return
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Tank Pressures", "Thrust", "Chamber Pressure", "Isp"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    P_tank_O_psi = pressure_curves.get("P_tank_O", np.array([])) / 6894.76
    P_tank_F_psi = pressure_curves.get("P_tank_F", np.array([])) / 6894.76
    fig.add_trace(go.Scatter(x=time, y=P_tank_O_psi, name="LOX Tank", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=P_tank_F_psi, name="Fuel Tank", line=dict(color="orange")), row=1, col=1)
    
    thrust = pressure_curves.get("thrust", np.array([]))
    fig.add_trace(go.Scatter(x=time, y=thrust, name="Thrust", line=dict(color="red")), row=1, col=2)
    
    Pc_MPa = pressure_curves.get("Pc", np.array([])) / 1e6
    fig.add_trace(go.Scatter(x=time, y=Pc_MPa, name="Pc", line=dict(color="green")), row=2, col=1)
    
    Isp = pressure_curves.get("Isp", np.array([]))
    fig.add_trace(go.Scatter(x=time, y=Isp, name="Isp", line=dict(color="purple")), row=2, col=2)
    
    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=2)
    fig.update_yaxes(title_text="Pressure [psi]", row=1, col=1)
    fig.update_yaxes(title_text="Thrust [N]", row=1, col=2)
    fig.update_yaxes(title_text="Pc [MPa]", row=2, col=1)
    fig.update_yaxes(title_text="Isp [s]", row=2, col=2)
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True, key="pressure_curves_plot")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Thrust", f"{np.mean(thrust):.1f} N")
    with col2:
        st.metric("Avg Isp", f"{np.mean(Isp):.1f} s")
    with col3:
        st.metric("Avg Pc", f"{np.mean(Pc_MPa):.2f} MPa")
    with col4:
        st.metric("Burn Time", f"{time[-1]:.1f} s")


def plot_copv_pressure(copv_results: Dict[str, Any], pressure_curves: Dict[str, np.ndarray]) -> None:
    """Plot COPV pressure curve alongside tank pressures."""
    time = copv_results.get("time", np.array([]))
    copv_pressure_psi = copv_results.get("copv_pressure_psi", np.array([]))
    
    if len(time) == 0:
        st.warning("No COPV data available")
        return
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=time, y=copv_pressure_psi, name="COPV Pressure", line=dict(color="green", width=2)),
        secondary_y=False
    )
    
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
    st.plotly_chart(fig, use_container_width=True, key="copv_pressure_plot")


def plot_flight_trajectory(flight_obj, requirements: Dict[str, Any]) -> None:
    """Plot flight trajectory from RocketPy flight object."""
    try:
        elevation = requirements.get("elevation", 0)
        
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
            st.plotly_chart(fig, use_container_width=True, key="flight_trajectory_plot")
    except Exception as e:
        st.warning(f"Could not plot flight trajectory: {e}")


def plot_optimization_convergence(optimization_results: Dict[str, Any]) -> None:
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
    l_chambers = [h.get("L_chamber", 0.2) * 1000 for h in history]
    
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
    
    fig.add_trace(
        go.Scatter(x=iterations, y=objectives, mode='lines+markers', name='Objective', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=thrust_errors, mode='lines+markers', name='Thrust Error', line=dict(color='red')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=of_errors, mode='lines+markers', name='O/F Error', line=dict(color='orange')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=thrusts, mode='lines+markers', name='Thrust', line=dict(color='green')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=lox_ratios, mode='lines+markers', name='LOX End %', line=dict(color='cyan')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=fuel_ratios, mode='lines+markers', name='Fuel End %', line=dict(color='magenta')),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=lstars, mode='lines+markers', name='L*', line=dict(color='purple')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=l_chambers, mode='lines+markers', name='L_chamber', line=dict(color='brown')),
        row=4, col=2
    )
    
    fig.update_xaxes(title_text="Iteration", row=4, col=1)
    fig.update_xaxes(title_text="Iteration", row=4, col=2)
    fig.update_layout(height=800, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True, key="optimization_summary_plot")
    
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


def plot_time_varying_results(time_varying_results: Dict[str, np.ndarray]) -> None:
    """Plot time-varying results (stability, recession, etc.)."""
    time = time_varying_results.get("time", np.array([]))
    if len(time) == 0:
        st.warning("No time-varying data available")
        return
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Stability Margin", "Chamber Recession",
            "Throat Recession", "Mass Flow Rates"
        ),
        vertical_spacing=0.15,
    )
    
    stability = time_varying_results.get("chugging_stability_margin", np.array([]))
    if len(stability) > 0:
        fig.add_trace(
            go.Scatter(x=time[:len(stability)], y=stability, name="Chugging Margin", line=dict(color="blue")),
            row=1, col=1
        )
    
    recession_chamber = time_varying_results.get("recession_chamber", np.array([]))
    if len(recession_chamber) > 0:
        fig.add_trace(
            go.Scatter(x=time[:len(recession_chamber)], y=recession_chamber * 1000, name="Chamber", line=dict(color="red")),
            row=1, col=2
        )
    
    recession_throat = time_varying_results.get("recession_throat", np.array([]))
    if len(recession_throat) > 0:
        fig.add_trace(
            go.Scatter(x=time[:len(recession_throat)], y=recession_throat * 1000, name="Throat", line=dict(color="orange")),
            row=2, col=1
        )
    
    mdot_O = time_varying_results.get("mdot_O", np.array([]))
    mdot_F = time_varying_results.get("mdot_F", np.array([]))
    if len(mdot_O) > 0:
        fig.add_trace(
            go.Scatter(x=time[:len(mdot_O)], y=mdot_O, name="LOX", line=dict(color="blue")),
            row=2, col=2
        )
    if len(mdot_F) > 0:
        fig.add_trace(
            go.Scatter(x=time[:len(mdot_F)], y=mdot_F, name="Fuel", line=dict(color="orange")),
            row=2, col=2
        )
    
    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=2)
    fig.update_yaxes(title_text="Stability Margin", row=1, col=1)
    fig.update_yaxes(title_text="Recession [mm]", row=1, col=2)
    fig.update_yaxes(title_text="Recession [mm]", row=2, col=1)
    fig.update_yaxes(title_text="Mass Flow [kg/s]", row=2, col=2)
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True, key="time_varying_results_plot")

