"""Layer 4: Flight Simulation and Validation

Run flight simulation to validate trajectory performance and adjust tank fills
(propellant masses) to hit apogee targets.

This layer:
1. Starts from the current optimized engine configuration (Layers 1–3)
2. Runs flight simulation with **full** pressure curves
3. Lets `flight_sim.py` handle truncation when tanks run out
4. Iteratively reduces propellant masses if apogee is too high
5. Accepts the best match if apogee is below target and masses cannot be
   increased beyond their initial values

We optimize tank fills, not burn time. Burn time naturally follows from
propellant mass and the thrust / mdot curves.
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
    Run Layer 4: Flight Simulation with tank-fill iteration.

    We:
    - Keep the full time history of thrust / mdot curves
    - Let the flight sim detect when tanks run out and truncate internally
    - Iteratively reduce LOX/fuel masses if apogee is too high
    """
    result: Dict[str, Any] = {
        "success": False,
        "apogee": 0.0,
        "max_velocity": 0.0,
        "layer": 4,
        "flight_candidate_valid": False,
    }

    try:
        # Copy config so we can safely tweak tank masses
        config_for_flight = copy.deepcopy(optimized_config)

        # Initial propellant masses from the optimized config
        initial_lox_mass = float(getattr(config_for_flight.lox_tank, "mass", 0.0))
        initial_fuel_mass = float(getattr(config_for_flight.fuel_tank, "mass", 0.0))

        # If tanks are not configured, just bail out gracefully
        if initial_lox_mass <= 0.0 or initial_fuel_mass <= 0.0:
            update_progress(
                "Layer 4: Flight Candidate",
                0.80,
                "Skipping flight sim: tank masses not configured",
            )
            return result

        # Start from full propellant, then reduce if apogee is too high
        current_lox_mass = initial_lox_mass
        current_fuel_mass = initial_fuel_mass

        mass_reduction_step = 0.05  # 5% reduction per iteration
        max_iterations = 20
        best_error = float("inf")
        best_result: Optional[Dict[str, Any]] = None

        for i in range(1, max_iterations + 1):
            progress = 0.75 + 0.10 * (i / max_iterations)

            # Apply current masses
            config_for_flight.lox_tank.mass = current_lox_mass
            config_for_flight.fuel_tank.mass = current_fuel_mass

            update_progress(
                "Layer 4: Flight Candidate",
                progress,
                f"Iteration {i}: LOX={current_lox_mass:.2f} kg, Fuel={current_fuel_mass:.2f} kg "
                f"(target apogee {target_apogee:.0f} m)",
            )

            # Run flight simulation with full curves; it will truncate when tanks are empty
            sim = run_flight_simulation_func(
                config_for_flight,
                pressure_curves,
                target_burn_time,
            )

            success = bool(sim.get("success", False))
            apogee = float(sim.get("apogee", 0.0) or 0.0)
            max_velocity = float(sim.get("max_velocity", 0.0) or 0.0)

            if not success:
                # If sim fails, try again with slightly less propellant; if it keeps failing,
                # we still keep the best attempt seen so far.
                current_lox_mass = max(0.1, current_lox_mass * (1.0 - mass_reduction_step))
                current_fuel_mass = max(0.1, current_fuel_mass * (1.0 - mass_reduction_step))
                continue

            # Compute fractional apogee error
            if target_apogee > 0.0:
                error_frac = abs(apogee - target_apogee) / target_apogee
            else:
                error_frac = 1.0

            # Track the best candidate
            if error_frac < best_error:
                best_error = error_frac
                best_result = {
                    "success": True,
                    "apogee": apogee,
                    "max_velocity": max_velocity,
                    "layer": 4,
                    "flight_candidate_valid": error_frac < apogee_tol,
                    "iterations": i,
                    "adjusted_lox_mass": current_lox_mass,
                    "adjusted_fuel_mass": current_fuel_mass,
                }

            # If we're within tolerance, we can stop
            if error_frac < apogee_tol:
                update_progress(
                    "Layer 4: Flight Candidate",
                    0.85,
                    f"Apogee {apogee:.0f} m within {error_frac * 100:.1f}% of target "
                    f"{target_apogee:.0f} m; accepting tank fills.",
                )
                break

            # If apogee is too high, reduce tank fills and try again
            if apogee > target_apogee:
                current_lox_mass = max(0.1, current_lox_mass * (1.0 - mass_reduction_step))
                current_fuel_mass = max(0.1, current_fuel_mass * (1.0 - mass_reduction_step))
            else:
                # Apogee too low – cannot increase beyond initial masses, so stop
                update_progress(
                    "Layer 4: Flight Candidate",
                    0.85,
                    f"Apogee {apogee:.0f} m is below target {target_apogee:.0f} m; "
                    "cannot increase tank fills beyond initial values, using best match.",
                )
                break

        # Finalize result from the best candidate, if any
        if best_result is not None:
            result.update(best_result)
        else:
            # No successful sim; keep default structure but mark failure
            result["success"] = False
            result["flight_candidate_valid"] = False

    except Exception as exc:
        update_progress(
            "Layer 4: Flight Candidate",
            0.85,
            f"Flight sim error: {exc}",
        )
        result.update(
            {
                "success": False,
                "error": str(exc),
                "apogee": 0.0,
                "max_velocity": 0.0,
                "flight_candidate_valid": False,
            }
        )

    return result

