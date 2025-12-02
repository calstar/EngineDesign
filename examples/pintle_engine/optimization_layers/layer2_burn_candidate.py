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
    n_time_points: int,
    update_progress: Callable,
    log_status: Callable,
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
    
    # Optimization variables for Layer 2
    layer2_bounds = []
    layer2_x0 = []
    
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
        layer2_state = {"iter": 0, "max_iter": 20}
        
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
            """Optimize initial thermal protection guesses to minimize recession."""
            try:
                config_layer2 = copy.deepcopy(optimized_config)
                idx = 0
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
                results_layer2 = runner_layer2.evaluate_arrays_with_time(
                    time_array,
                    P_tank_O_array,
                    P_tank_F_array,
                    track_ablative_geometry=True,
                    use_coupled_solver=False,
                )
                
                recession_chamber = float(np.max(results_layer2.get("recession_chamber", [0.0])))
                recession_throat = float(np.max(results_layer2.get("recession_throat", [0.0])))
                
                stability_scores = results_layer2.get("stability_score", None)
                if stability_scores is not None:
                    min_stability = float(np.min(stability_scores))
                else:
                    chugging = results_layer2.get("chugging_stability_margin", np.array([1.0]))
                    min_stability = max(0.0, min(1.0, (float(np.min(chugging)) - 0.3) * 1.5))
                
                thrust_hist = np.atleast_1d(results_layer2.get("F", np.full(n_time_points, target_thrust)))
                available_n = min(thrust_hist.shape[0], n_time_points)
                if available_n >= 2:
                    check_indices = np.arange(available_n - 1)
                    thrust_hist = thrust_hist[:available_n]
                    thrust_errors = np.abs(thrust_hist[check_indices] - target_thrust) / target_thrust
                    max_thrust_err = float(np.max(thrust_errors))
                elif available_n == 1:
                    max_thrust_err = float(abs(thrust_hist[0] - target_thrust) / max(target_thrust, 1e-9))
                else:
                    max_thrust_err = 1.0
                
                stability_penalty = max(0, 0.7 - min_stability) * 10.0
                thrust_penalty = max(0, max_thrust_err - thrust_tol * 1.5) * 5.0
                
                obj = recession_chamber * 1000 + recession_throat * 1000 + stability_penalty + thrust_penalty
                return obj
            except Exception:
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
    
    try:
        full_time_results = optimized_runner.evaluate_arrays_with_time(
            time_array,
            P_tank_O_array,
            P_tank_F_array,
            track_ablative_geometry=True,
            use_coupled_solver=False,
        )
    except Exception as e:
        log_status("Layer 2 BurnCandidate Error", f"Exception in burn-candidate time series: {repr(e)}")
        full_time_results = {}
    
    # Build time-varying summary
    if full_time_results:
        chugging_stability_history = full_time_results.get("chugging_stability_margin", np.array([1.0]))
        min_time_stability_margin = float(np.min(chugging_stability_history))
        
        stability_scores = full_time_results.get("stability_score", None)
        if stability_scores is None:
            min_stability_score_time = max(0.0, min(1.0, (min_time_stability_margin - 0.3) * 1.5))
        else:
            min_stability_score_time = float(np.min(stability_scores))
        
        time_varying_summary = {
            "avg_thrust": float(np.mean(full_time_results.get("F", [target_thrust]))),
            "min_thrust": float(np.min(full_time_results.get("F", [target_thrust]))),
            "max_thrust": float(np.max(full_time_results.get("F", [target_thrust]))),
            "thrust_std": float(np.std(full_time_results.get("F", [0]))),
            "avg_isp": float(np.mean(full_time_results.get("Isp", [250]))),
            "min_stability_margin": min_time_stability_margin,
            "min_stability_score": min_stability_score_time,
            "max_recession_chamber": float(np.max(full_time_results.get("recession_chamber", [0.0]))),
            "max_recession_throat": float(np.max(full_time_results.get("recession_throat", [0.0]))),
        }
    
    return optimized_config, full_time_results, time_varying_summary, burn_candidate_valid

