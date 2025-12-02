"""COPV and Flight Simulation Helper Functions.

This module contains:
- COPV pressure curve calculation
- Flight simulation execution
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd

from pintle_pipeline.config_schemas import PintleEngineConfig


def calculate_copv_pressure_curve(
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
        
        psi_to_Pa = 6894.757293168
        df = pd.DataFrame({
            "time": time_array,
            "mdot_O (kg/s)": mdot_O,
            "mdot_F (kg/s)": mdot_F,
            "P_tank_O (psi)": P_tank_O / psi_to_Pa,
            "P_tank_F (psi)": P_tank_F / psi_to_Pa,
        })
        
        copv_results = size_or_check_copv_for_polytropic_N2(
            df=df,
            config=config,
            n=1.2,
            T0_K=T0_K,
            Tp_K=Tp_K,
            use_real_gas=False,
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
        copv_P0 = P_max * 1.15
        n = 1.2
        mdot_total = mdot_O + mdot_F
        mass_consumed = np.cumsum(mdot_total * np.gradient(time_array))
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


def run_flight_simulation(
    config: PintleEngineConfig,
    pressure_curves: Dict[str, np.ndarray],
    burn_time: float,
) -> Dict[str, Any]:
    """Run flight simulation on the optimized engine."""
    try:
        from examples.pintle_engine.flight_sim import setup_flight
        from scipy.interpolate import interp1d
        
        time_array = pressure_curves["time"]
        thrust_array = pressure_curves["thrust"]
        mdot_O_array = pressure_curves["mdot_O"]
        mdot_F_array = pressure_curves["mdot_F"]
        
        thrust_func = interp1d(time_array, thrust_array, kind='linear', fill_value=0, bounds_error=False)
        mdot_O_func = interp1d(time_array, mdot_O_array, kind='linear', fill_value=0, bounds_error=False)
        mdot_F_func = interp1d(time_array, mdot_F_array, kind='linear', fill_value=0, bounds_error=False)
        
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

