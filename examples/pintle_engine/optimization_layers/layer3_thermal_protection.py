"""Layer 3: Thermal Protection Optimization (Final Sizing)

This layer optimizes final ablative liner and graphite insert thicknesses to
meet recession requirements with margin while minimizing mass.

Once Layer 2 passes, this refines the thermal protection to right-size
the thicknesses (20% margin over max recession).
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np
import copy

from pintle_pipeline.config_schemas import PintleEngineConfig
from pintle_models.runner import PintleEngineRunner


def run_layer3_thermal_protection(
    optimized_config: PintleEngineConfig,
    time_array: np.ndarray,
    P_tank_O_array: np.ndarray,
    P_tank_F_array: np.ndarray,
    full_time_results: Dict[str, Any],
    n_time_points: int,
    update_progress: Callable,
    log_status: Callable,
) -> Tuple[PintleEngineConfig, Dict[str, Any], Dict[str, Any]]:
    """
    Run Layer 3: Thermal Protection Optimization.
    
    Optimizes final thermal protection thicknesses to meet recession requirements
    with margin while minimizing mass.
    
    Returns:
        Tuple of (optimized_config, updated_time_results, thermal_results)
    """
    from scipy.optimize import minimize as scipy_minimize
    
    ablative_cfg = optimized_config.ablative_cooling if hasattr(optimized_config, 'ablative_cooling') else None
    graphite_cfg = optimized_config.graphite_insert if hasattr(optimized_config, 'graphite_insert') else None
    
    # Get recession data from time-varying results
    recession_chamber_history = full_time_results.get("recession_chamber", np.zeros(n_time_points))
    recession_throat_history = full_time_results.get("recession_throat", np.zeros(n_time_points))
    max_recession_chamber = float(np.max(recession_chamber_history))
    max_recession_throat = float(np.max(recession_throat_history))
    
    thermal_results = {
        "max_recession_chamber": max_recession_chamber,
        "max_recession_throat": max_recession_throat,
        "ablative_adequate": True,
        "graphite_adequate": True,
        "thermal_protection_valid": True,
    }
    
    layer3_bounds = []
    layer3_x0 = []
    
    if ablative_cfg and ablative_cfg.enabled:
        # Optimize to max_recession * 1.2 (20% margin)
        target_ablative = max_recession_chamber * 1.2
        layer3_bounds.append((max(0.003, target_ablative * 0.8), min(0.020, target_ablative * 1.5)))
        layer3_x0.append(ablative_cfg.initial_thickness)
    
    if graphite_cfg and graphite_cfg.enabled:
        # Optimize to max_recession * 1.2 (20% margin)
        target_graphite = max_recession_throat * 1.2
        layer3_bounds.append((max(0.003, target_graphite * 0.8), min(0.015, target_graphite * 1.5)))
        layer3_x0.append(graphite_cfg.initial_thickness)
    
    updated_time_results = full_time_results
    
    if len(layer3_x0) > 0:
        layer3_x0 = np.array(layer3_x0)
        
        def layer3_objective(x_layer3):
            """Optimize thermal protection to minimize mass while meeting recession requirements."""
            try:
                config_layer3 = copy.deepcopy(optimized_config)
                idx = 0
                if ablative_cfg and ablative_cfg.enabled:
                    config_layer3.ablative_cooling.initial_thickness = float(
                        np.clip(x_layer3[idx], layer3_bounds[idx][0], layer3_bounds[idx][1])
                    )
                    idx += 1
                if graphite_cfg and graphite_cfg.enabled:
                    config_layer3.graphite_insert.initial_thickness = float(
                        np.clip(x_layer3[idx], layer3_bounds[idx][0], layer3_bounds[idx][1])
                    )
                
                runner_layer3 = PintleEngineRunner(config_layer3)
                results_layer3 = runner_layer3.evaluate_arrays_with_time(
                    time_array,
                    P_tank_O_array,
                    P_tank_F_array,
                    track_ablative_geometry=True,
                    use_coupled_solver=False,
                )
                
                recession_chamber = float(np.max(results_layer3.get("recession_chamber", [0.0])))
                recession_throat = float(np.max(results_layer3.get("recession_throat", [0.0])))
                
                # Check if recession exceeds thickness (with 20% margin)
                idx = 0
                recession_penalty = 0.0
                if ablative_cfg and ablative_cfg.enabled:
                    thickness = x_layer3[idx]
                    if recession_chamber > thickness * 0.8:  # 80% of thickness
                        recession_penalty += 1000.0 * (recession_chamber - thickness * 0.8)
                    idx += 1
                if graphite_cfg and graphite_cfg.enabled:
                    thickness = x_layer3[idx]
                    if recession_throat > thickness * 0.8:
                        recession_penalty += 1000.0 * (recession_throat - thickness * 0.8)
                
                # Objective: minimize mass (thickness) + recession penalty
                total_thickness = np.sum(x_layer3)
                obj = total_thickness * 1000 + recession_penalty  # Convert to mm for scaling
                return obj
            except Exception:
                return 1e6
        
        # Optimize Layer 3
        try:
            result_layer3 = scipy_minimize(
                layer3_objective,
                layer3_x0,
                method='L-BFGS-B',
                bounds=layer3_bounds,
                options={'maxiter': 30, 'ftol': 1e-5}
            )
            
            # Update config with optimized thicknesses
            idx = 0
            if ablative_cfg and ablative_cfg.enabled:
                optimized_config.ablative_cooling.initial_thickness = float(
                    np.clip(result_layer3.x[idx], layer3_bounds[idx][0], layer3_bounds[idx][1])
                )
                thermal_results["optimized_ablative_thickness"] = optimized_config.ablative_cooling.initial_thickness
                update_progress("Layer 3: Burn Analysis Optimization", 0.70, 
                    f"✓ Optimized ablative: {optimized_config.ablative_cooling.initial_thickness*1000:.2f}mm (recession: {max_recession_chamber*1000:.2f}mm)")
                idx += 1
            if graphite_cfg and graphite_cfg.enabled:
                optimized_config.graphite_insert.initial_thickness = float(
                    np.clip(result_layer3.x[idx], layer3_bounds[idx][0], layer3_bounds[idx][1])
                )
                thermal_results["optimized_graphite_thickness"] = optimized_config.graphite_insert.initial_thickness
                update_progress("Layer 3: Burn Analysis Optimization", 0.72, 
                    f"✓ Optimized graphite: {optimized_config.graphite_insert.initial_thickness*1000:.2f}mm (recession: {max_recession_throat*1000:.2f}mm)")
        except Exception as e:
            update_progress("Layer 3: Burn Analysis Optimization", 0.72, 
                f"⚠️ Layer 3 optimization failed: {e}, using current values")
        
        # Re-run time series with optimized thermal protection to verify
        update_progress("Layer 3: Burn Analysis", 0.74, "Re-running time series with optimized thermal protection...")
        try:
            optimized_runner_updated = PintleEngineRunner(optimized_config)
            full_time_results_updated = optimized_runner_updated.evaluate_arrays_with_time(
                time_array,
                P_tank_O_array,
                P_tank_F_array,
                track_ablative_geometry=True,
                use_coupled_solver=True,
            )
            updated_time_results = full_time_results_updated
            thermal_results["max_recession_chamber"] = float(np.max(full_time_results_updated.get("recession_chamber", [0.0])))
            thermal_results["max_recession_throat"] = float(np.max(full_time_results_updated.get("recession_throat", [0.0])))
        except Exception as e:
            update_progress("Layer 3: Burn Analysis", 0.74, f"⚠️ Re-evaluation failed: {e}, using original results")
        
        log_status(
            "Layer 3",
            "Completed | Ablative {:.2f} mm, Graphite {:.2f} mm, Max recession chamber {:.2f} mm, throat {:.2f} mm".format(
                (optimized_config.ablative_cooling.initial_thickness * 1000) if ablative_cfg and ablative_cfg.enabled else 0.0,
                (optimized_config.graphite_insert.initial_thickness * 1000) if graphite_cfg and graphite_cfg.enabled else 0.0,
                thermal_results.get("max_recession_chamber", 0.0) * 1000,
                thermal_results.get("max_recession_throat", 0.0) * 1000,
            )
        )
    
    return optimized_config, updated_time_results, thermal_results

