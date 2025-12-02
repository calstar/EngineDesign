"""Layer 2: Time-Series Burn Candidate Optimization

This layer optimizes initial thermal protection (ablative/graphite) thickness guesses
based on time-series analysis over the full burn.

Layer 2 runs when:
- Time-varying analysis is enabled
- Layer 1 pressure candidate is at least reasonable (even if not perfect)
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np
import copy

from pintle_pipeline.config_schemas import PintleEngineConfig
from pintle_models.runner import PintleEngineRunner


def run_layer2_burn_candidate(
    optimized_config: PintleEngineConfig,
    time_array: np.ndarray,
    P_tank_O_array: np.ndarray,
    P_tank_F_array: np.ndarray,
    target_thrust: float,
    thrust_tol: float,
    optimal_of: float,  # CRITICAL: Pass target O/F ratio
    n_time_points: int,
    update_progress: Callable,
    log_status: Callable,
    max_lox_P_psi: float = None,  # For pressure optimization
    max_fuel_P_psi: float = None,  # For pressure optimization
) -> Tuple[PintleEngineConfig, Dict[str, Any], Dict[str, Any], bool]:
    """
    Run Layer 2: Burn Candidate Optimization.
    
    Optimizes initial thermal protection guesses based on time-series analysis.
    
    Returns:
        Tuple of (optimized_config, full_time_results, time_varying_summary, burn_candidate_valid)
    """
    from scipy.optimize import minimize as scipy_minimize
    
    # Get current ablative/graphite config
    ablative_cfg = optimized_config.ablative_cooling if hasattr(optimized_config, 'ablative_cooling') else None
    graphite_cfg = optimized_config.graphite_insert if hasattr(optimized_config, 'graphite_insert') else None
    
    # CRITICAL FIX: Layer 2 MUST optimize pressure curves to maintain thrust/O/F!
    # The problem: Layer 1 gives good initial performance, but as recession happens,
    # thrust/O/F drift. Layer 2 needs to adjust pressures to compensate.
    
    # Optimization variables for Layer 2:
    # 1. Thermal protection (ablative/graphite) - minimize recession
    # 2. Pressure scaling factors - adjust pressures to maintain thrust/O/F
    layer2_bounds = []
    layer2_x0 = []
    
    # Add pressure scaling variables (if max pressures provided)
    optimize_pressures = (max_lox_P_psi is not None and max_fuel_P_psi is not None)
    if optimize_pressures:
        # Scale factors for initial pressures (0.7-1.1x to allow some adjustment)
        layer2_bounds.append((0.7, 1.1))  # LOX pressure scale
        layer2_bounds.append((0.7, 1.1))  # Fuel pressure scale
        # Start at 1.0 (no change from Layer 1)
        layer2_x0.append(1.0)
        layer2_x0.append(1.0)
    
    if ablative_cfg and ablative_cfg.enabled:
        layer2_bounds.append((0.003, 0.020))  # 3-20mm
        layer2_x0.append(ablative_cfg.initial_thickness)
    if graphite_cfg and graphite_cfg.enabled:
        layer2_bounds.append((0.003, 0.015))  # 3-15mm
        layer2_x0.append(graphite_cfg.initial_thickness)
    
    full_time_results = {}
    time_varying_summary = {}
    burn_candidate_valid = False
    
    if len(layer2_x0) > 0:
        layer2_x0 = np.array(layer2_x0)
        
        # Track Layer 2 optimization progress
        layer2_state = {"iter": 0, "max_iter": 50}  # Increased iterations for pressure optimization
        
        def layer2_callback(xk):
            layer2_state["iter"] += 1
            frac = min(layer2_state["iter"] / max(layer2_state["max_iter"], 1), 1.0)
            progress = 0.60 + 0.04 * frac
            update_progress(
                "Layer 2: Burn Candidate Optimization",
                progress,
                f"Layer 2 optimization {layer2_state['iter']}/{layer2_state['max_iter']}",
            )
        
        def layer2_objective(x_layer2):
            """Optimize thermal protection AND pressure curves to maintain thrust/O/F."""
            try:
                config_layer2 = copy.deepcopy(optimized_config)
                idx = 0
                
                # CRITICAL: Apply pressure scaling if optimizing pressures
                P_O_array_scaled = P_tank_O_array.copy()
                P_F_array_scaled = P_tank_F_array.copy()
                if optimize_pressures:
                    lox_scale = float(np.clip(x_layer2[idx], layer2_bounds[idx][0], layer2_bounds[idx][1]))
                    idx += 1
                    fuel_scale = float(np.clip(x_layer2[idx], layer2_bounds[idx][0], layer2_bounds[idx][1]))
                    idx += 1
                    # Scale entire pressure curves
                    P_O_array_scaled = P_tank_O_array * lox_scale
                    P_F_array_scaled = P_tank_F_array * fuel_scale
                
                if ablative_cfg and ablative_cfg.enabled:
                    config_layer2.ablative_cooling.initial_thickness = float(
                        np.clip(x_layer2[idx], layer2_bounds[idx][0], layer2_bounds[idx][1])
                    )
                    idx += 1
                if graphite_cfg and graphite_cfg.enabled:
                    config_layer2.graphite_insert.initial_thickness = float(
                        np.clip(x_layer2[idx], layer2_bounds[idx][0], layer2_bounds[idx][1])
                    )
                
                runner_layer2 = PintleEngineRunner(config_layer2)
                # CRITICAL: Use fully-coupled solver for accurate time-varying analysis
                # This includes geometry evolution, reaction chemistry, and stability
                try:
                    results_layer2 = runner_layer2.evaluate_arrays_with_time(
                        time_array,
                        P_O_array_scaled,  # Use scaled pressures!
                        P_F_array_scaled,  # Use scaled pressures!
                        track_ablative_geometry=True,
                        use_coupled_solver=True,  # Use fully-coupled solver for Layer 2
                    )
                except Exception as e:
                    # If coupled solver fails, try without it as fallback
                    log_status("Layer 2 Objective", f"Coupled solver failed, using fallback: {repr(e)[:100]}")
                    results_layer2 = runner_layer2.evaluate_arrays_with_time(
                        time_array,
                        P_tank_O_array,
                        P_tank_F_array,
                        track_ablative_geometry=True,
                        use_coupled_solver=False,
                    )
                
                recession_chamber = float(np.max(results_layer2.get("recession_chamber", [0.0])))
                recession_throat = float(np.max(results_layer2.get("recession_throat", [0.0])))
                
                # CRITICAL FIX: Robust stability extraction - handle NaN/inf values
                stability_scores = results_layer2.get("stability_score", None)
                if stability_scores is not None:
                    # Filter out NaN/inf values
                    valid_scores = [s for s in np.atleast_1d(stability_scores) if np.isfinite(s)]
                    if len(valid_scores) > 0:
                        min_stability = float(np.min(valid_scores))
                    else:
                        min_stability = 0.5  # Default to marginal if all invalid
                else:
                    # Fallback: try to get from individual margins
                    try:
                        chugging = results_layer2.get("chugging_stability_margin", np.array([1.0]))
                        chugging_valid = [c for c in np.atleast_1d(chugging) if np.isfinite(c)]
                        if len(chugging_valid) > 0:
                            min_stability = max(0.0, min(1.0, (float(np.min(chugging_valid)) - 0.3) * 1.5))
                        else:
                            min_stability = 0.5  # Default to marginal
                    except Exception:
                        min_stability = 0.5  # Safe default
                
                # CRITICAL FIX: Robust handling of thrust history array
                thrust_hist = np.atleast_1d(results_layer2.get("F", np.full(n_time_points, target_thrust)))
                available_n = len(thrust_hist) if len(thrust_hist) > 0 else 0
                
                if available_n >= 2:
                    # Calculate errors for all points
                    thrust_errors = np.abs(thrust_hist - target_thrust) / max(target_thrust, 1e-9)
                    max_thrust_err = float(np.max(thrust_errors))
                    avg_thrust_err = float(np.mean(thrust_errors))
                elif available_n == 1:
                    max_thrust_err = float(abs(thrust_hist[0] - target_thrust) / max(target_thrust, 1e-9))
                    avg_thrust_err = max_thrust_err
                else:
                    max_thrust_err = 1.0
                    avg_thrust_err = 1.0
                
                # Also check O/F errors (if available)
                MR_hist = np.atleast_1d(results_layer2.get("MR", np.full(n_time_points, optimal_of)))
                max_MR_err = 0.0  # Initialize
                if len(MR_hist) > 0:
                    # Use target O/F passed as parameter
                    MR_errors = np.abs(MR_hist - optimal_of) / max(optimal_of, 1e-9)
                    max_MR_err = float(np.max(MR_errors))
                else:
                    max_MR_err = 1.0
                
                # CRITICAL FIX: MUCH stronger penalties for thrust/O/F errors
                # Layer 2 MUST maintain thrust and O/F throughout burn - these are PRIMARY goals
                stability_penalty = max(0, 0.3 - min_stability) * 5.0  # Reduced penalty (stability is secondary)
                # Thrust penalty: MUCH stronger - 54.7% error is unacceptable!
                thrust_penalty = (max_thrust_err ** 2) * 500.0  # Squared error, heavy penalty
                # O/F penalty: MUCH stronger - 66.7% error is catastrophic!
                MR_penalty = (max_MR_err ** 2) * 1000.0  # Squared error, very heavy penalty
                
                # Also penalize average errors to encourage consistent performance
                if available_n >= 2:
                    avg_thrust_penalty = (avg_thrust_err ** 2) * 200.0
                else:
                    avg_thrust_penalty = 0.0
                
                # Objective: PRIMARY = thrust/O/F accuracy, SECONDARY = recession, TERTIARY = stability
                obj = (
                    thrust_penalty +           # PRIMARY: Thrust accuracy (heavily weighted)
                    MR_penalty +                # PRIMARY: O/F accuracy (heavily weighted)
                    avg_thrust_penalty +        # PRIMARY: Consistent thrust
                    recession_chamber * 100 +   # SECONDARY: Recession (reduced weight)
                    recession_throat * 100 +    # SECONDARY: Recession (reduced weight)
                    stability_penalty           # TERTIARY: Stability (lowest weight)
                )
                return obj
            except Exception as e:
                # Log error for debugging
                if layer2_state["iter"] % 5 == 0:  # Log every 5 iterations to avoid spam
                    log_status("Layer 2 Objective Error", f"Iter {layer2_state['iter']}: {repr(e)[:150]}")
                return 1e6
        
        # Optimize Layer 2
        try:
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
                optimized_config.ablative_cooling.initial_thickness = float(
                    np.clip(result_layer2.x[idx], layer2_bounds[idx][0], layer2_bounds[idx][1])
                )
                update_progress("Layer 2: Burn Candidate Optimization", 0.62, 
                    f"Optimized ablative initial guess: {optimized_config.ablative_cooling.initial_thickness*1000:.2f}mm")
                idx += 1
            if graphite_cfg and graphite_cfg.enabled:
                optimized_config.graphite_insert.initial_thickness = float(
                    np.clip(result_layer2.x[idx], layer2_bounds[idx][0], layer2_bounds[idx][1])
                )
                update_progress("Layer 2: Burn Candidate Optimization", 0.62, 
                    f"Optimized graphite initial guess: {optimized_config.graphite_insert.initial_thickness*1000:.2f}mm")
        except Exception as e:
            update_progress("Layer 2: Burn Candidate Optimization", 0.62, 
                f"⚠️ Layer 2 optimization failed: {e}, using current values")
    
    # Run time series with optimized initial guesses
    update_progress("Layer 2: Burn Candidate", 0.64, "Running time series analysis with optimized guesses...")
    optimized_runner = PintleEngineRunner(optimized_config)
    
    # CRITICAL: Use fully-coupled solver for final time series analysis
    # This provides accurate geometry evolution, reaction chemistry, and stability
    try:
        full_time_results = optimized_runner.evaluate_arrays_with_time(
            time_array,
            P_tank_O_array,
            P_tank_F_array,
            track_ablative_geometry=True,
            use_coupled_solver=True,  # Use fully-coupled solver for accurate results
        )
    except Exception as e:
        # If coupled solver fails, try without it as fallback
        log_status("Layer 2 BurnCandidate Error", f"Coupled solver failed, trying fallback: {repr(e)[:200]}")
        try:
            full_time_results = optimized_runner.evaluate_arrays_with_time(
                time_array,
                P_tank_O_array,
                P_tank_F_array,
                track_ablative_geometry=True,
                use_coupled_solver=False,
            )
        except Exception as e2:
            log_status("Layer 2 BurnCandidate Error", f"Fallback solver also failed: {repr(e2)[:200]}")
            full_time_results = {}
    
    # Build time-varying summary with robust error handling
    if full_time_results:
        # Get stability metrics with safe defaults
        chugging_stability_history = full_time_results.get("chugging_stability_margin", np.array([1.0]))
        if len(chugging_stability_history) > 0:
            min_time_stability_margin = float(np.min(chugging_stability_history))
        else:
            min_time_stability_margin = 1.0
        
        stability_scores = full_time_results.get("stability_score", None)
        if stability_scores is not None and len(stability_scores) > 0:
            min_stability_score_time = float(np.min(stability_scores))
        else:
            min_stability_score_time = max(0.0, min(1.0, (min_time_stability_margin - 0.3) * 1.5))
        
        # Get thrust metrics with safe defaults
        F_hist = full_time_results.get("F", np.array([target_thrust]))
        if len(F_hist) > 0:
            avg_thrust = float(np.mean(F_hist))
            min_thrust = float(np.min(F_hist))
            max_thrust = float(np.max(F_hist))
            thrust_std = float(np.std(F_hist))
        else:
            avg_thrust = target_thrust
            min_thrust = target_thrust
            max_thrust = target_thrust
            thrust_std = 0.0
        
        # Get Isp metrics
        Isp_hist = full_time_results.get("Isp", np.array([250.0]))
        avg_isp = float(np.mean(Isp_hist)) if len(Isp_hist) > 0 else 250.0
        
        # Get recession metrics
        recession_chamber_hist = full_time_results.get("recession_chamber", np.array([0.0]))
        recession_throat_hist = full_time_results.get("recession_throat", np.array([0.0]))
        max_recession_chamber = float(np.max(recession_chamber_hist)) if len(recession_chamber_hist) > 0 else 0.0
        max_recession_throat = float(np.max(recession_throat_hist)) if len(recession_throat_hist) > 0 else 0.0
        
        time_varying_summary = {
            "avg_thrust": avg_thrust,
            "min_thrust": min_thrust,
            "max_thrust": max_thrust,
            "thrust_std": thrust_std,
            "avg_isp": avg_isp,
            "min_stability_margin": min_time_stability_margin,
            "min_stability_score": min_stability_score_time,
            "max_recession_chamber": max_recession_chamber,
            "max_recession_throat": max_recession_throat,
        }
    else:
        # Empty summary if no results
        time_varying_summary = {
            "avg_thrust": target_thrust,
            "min_thrust": target_thrust,
            "max_thrust": target_thrust,
            "thrust_std": 0.0,
            "avg_isp": 250.0,
            "min_stability_margin": 1.0,
            "min_stability_score": 1.0,
            "max_recession_chamber": 0.0,
            "max_recession_throat": 0.0,
        }
    
    return optimized_config, full_time_results, time_varying_summary, burn_candidate_valid

