"""Layer 4: Flight Simulation and Validation

Run flight simulation with propellant truncation to validate trajectory
performance. Once Layer 3 passes, run flight sim with backward iteration.
Automatically detects tank empty conditions and truncates thrust.
Iterates backward if apogee goals not met (reduce propellant, rerun).
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np
import copy

from pintle_pipeline.config_schemas import PintleEngineConfig


def run_layer4_flight_simulation(
    optimized_config: PintleEngineConfig,
    pressure_curves: Dict[str, np.ndarray],
    time_array: np.ndarray,
    P_tank_O_array: np.ndarray,
    P_tank_F_array: np.ndarray,
    target_burn_time: float,
    target_apogee: float,
    apogee_tol: float,
    update_progress: Callable,
    log_status: Callable,
    run_flight_simulation_func: Callable,
) -> Dict[str, Any]:
    """
    Run Layer 4: Flight Simulation with backward iteration.
    
    Args:
        optimized_config: The optimized engine configuration
        pressure_curves: Dict with time-varying performance data
        time_array: Time array for the burn
        P_tank_O_array: LOX tank pressure array
        P_tank_F_array: Fuel tank pressure array
        target_burn_time: Target burn time [s]
        target_apogee: Target apogee altitude [m]
        apogee_tol: Apogee tolerance (fraction)
        update_progress: Progress callback
        log_status: Logging callback
        run_flight_simulation_func: Function to run actual flight sim
    
    Returns:
        Dict with flight simulation results
    """
    flight_sim_result = {
        "success": False, 
        "apogee": 0, 
        "max_velocity": 0, 
        "layer": 4,
        "flight_candidate_valid": False,
    }
    
    try:
        # Layer 4: Iterative backward truncation to meet apogee goals
        epsilon = 0.01  # Small time step for backward iteration
        max_iterations_flight = 20
        flight_iteration = 0
        current_burn_time = target_burn_time
        flight_candidate_valid = False
        
        # Get initial propellant masses
        config_for_flight = copy.deepcopy(optimized_config)
        initial_lox_mass = config_for_flight.lox_tank.mass if hasattr(config_for_flight, 'lox_tank') else 0
        initial_fuel_mass = config_for_flight.fuel_tank.mass if hasattr(config_for_flight, 'fuel_tank') else 0
        
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
            
            # Calculate remaining propellant mass
            if cutoff_time < target_burn_time:
                remaining_time = target_burn_time - cutoff_time
                mdot_O_cutoff = pressure_curves["mdot_O"][mask][-1] if len(pressure_curves["mdot_O"][mask]) > 0 else 0
                mdot_F_cutoff = pressure_curves["mdot_F"][mask][-1] if len(pressure_curves["mdot_F"][mask]) > 0 else 0
                remaining_lox_mass = mdot_O_cutoff * remaining_time
                remaining_fuel_mass = mdot_F_cutoff * remaining_time
            else:
                remaining_lox_mass = 0
                remaining_fuel_mass = 0
            
            # Adjust masses
            adjusted_lox_mass = max(0.1, initial_lox_mass - remaining_lox_mass)
            adjusted_fuel_mass = max(0.1, initial_fuel_mass - remaining_fuel_mass)
            
            config_for_flight.lox_tank.mass = adjusted_lox_mass
            config_for_flight.fuel_tank.mass = adjusted_fuel_mass
            
            # Run flight simulation
            flight_sim_result = run_flight_simulation_func(
                config_for_flight,
                pressure_curves_trunc,
                cutoff_time,
            )
            
            if flight_sim_result.get("success", False):
                apogee = flight_sim_result.get("apogee", 0)
                apogee_error = abs(apogee - target_apogee) / target_apogee if target_apogee > 0 else 1.0
                
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
                    if apogee < target_apogee:
                        current_burn_time = cutoff_time
                    else:
                        flight_candidate_valid = True
                        update_progress(
                            "Layer 4: Flight Candidate",
                            0.85,
                            f"✓ Best match - Apogee {apogee:.0f}m (target: {target_apogee:.0f}m, error: {apogee_error*100:.1f}%, burn: {cutoff_time:.2f}s)",
                        )
                        log_status(
                            "Layer 4",
                            f"ACCEPTED | Apogee {apogee:.0f}m (error {apogee_error*100:.1f}%), burn {cutoff_time:.2f}s",
                        )
                        flight_sim_result["actual_burn_time"] = cutoff_time
                        flight_sim_result["adjusted_lox_mass"] = adjusted_lox_mass
                        flight_sim_result["adjusted_fuel_mass"] = adjusted_fuel_mass
                        flight_sim_result["iterations"] = flight_iteration
                        break
            else:
                current_burn_time = cutoff_time
                if flight_iteration >= max_iterations_flight:
                    break
            
            if current_burn_time < 0.5:
                update_progress("Layer 4: Flight Candidate", 0.85, 
                    "⚠️ Reached minimum burn time (0.5s), stopping iteration")
                break
        
        flight_sim_result["flight_candidate_valid"] = flight_candidate_valid
        
    except Exception as e:
        flight_sim_result = {
            "success": False, 
            "error": str(e), 
            "apogee": 0, 
            "max_velocity": 0,
            "flight_candidate_valid": False,
        }
        update_progress("Layer 4: Flight Candidate", 0.85, f"⚠️ Flight sim error: {e}")
    
    return flight_sim_result

