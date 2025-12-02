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
import logging
import time
from datetime import datetime

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
    objective_callback: Optional[Callable[[int, float, float], None]] = None,
) -> Tuple[PintleEngineConfig, Dict[str, Any], Dict[str, Any]]:
    """
    Run Layer 3: Thermal Protection Optimization.

    Optimizes final thermal protection thicknesses to meet recession requirements
    with margin while minimizing mass.

    Returns:
        Tuple of (optimized_config, updated_time_results, thermal_results)
    """
    from scipy.optimize import minimize as scipy_minimize

    # ------------------------------------------------------------------
    # Set up Layer 3 logging (mirrors Layer 2 style)
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"layer3_thermal_{timestamp}.log"

    layer3_logger = logging.getLogger("layer3_thermal")
    layer3_logger.setLevel(logging.INFO)
    layer3_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    layer3_logger.addHandler(file_handler)
    layer3_logger.propagate = False

    layer3_logger.info("=" * 70)
    layer3_logger.info("Layer 3: Thermal Protection Optimization")
    layer3_logger.info("=" * 70)
    layer3_logger.info(f"Log file: {log_file_path}")
    layer3_logger.info(f"Time points: {n_time_points}")

    ablative_cfg = optimized_config.ablative_cooling if hasattr(optimized_config, "ablative_cooling") else None
    graphite_cfg = optimized_config.graphite_insert if hasattr(optimized_config, "graphite_insert") else None

    # Get recession data from time-varying results
    recession_chamber_history = np.atleast_1d(full_time_results.get("recession_chamber", np.zeros(n_time_points)))
    recession_throat_history = np.atleast_1d(full_time_results.get("recession_throat", np.zeros(n_time_points)))
    max_recession_chamber = float(np.max(recession_chamber_history)) if recession_chamber_history.size else 0.0
    max_recession_throat = float(np.max(recession_throat_history)) if recession_throat_history.size else 0.0

    layer3_logger.info(
        "Initial max recession (chamber / throat): "
        f"{max_recession_chamber*1000:.3f} mm / {max_recession_throat*1000:.3f} mm"
    )

    thermal_results = {
        "max_recession_chamber": max_recession_chamber,
        "max_recession_throat": max_recession_throat,
        "ablative_adequate": True,
        "graphite_adequate": True,
        "thermal_protection_valid": True,
        "log_file": log_file_path,
    }

    layer3_bounds = []
    layer3_x0 = []

    if ablative_cfg and ablative_cfg.enabled:
        # Optimize to max_recession * 1.2 (20% margin)
        target_ablative = max_recession_chamber * 1.2
        bounds_abl = (max(0.003, target_ablative * 0.8), min(0.020, target_ablative * 1.5))
        layer3_bounds.append(bounds_abl)
        layer3_x0.append(float(ablative_cfg.initial_thickness))
        layer3_logger.info(
            "Ablative enabled: initial=%.3f mm, target≈%.3f mm, bounds=[%.3f, %.3f] mm",
            ablative_cfg.initial_thickness * 1000.0,
            target_ablative * 1000.0,
            bounds_abl[0] * 1000.0,
            bounds_abl[1] * 1000.0,
        )

    if graphite_cfg and graphite_cfg.enabled:
        # Optimize to max_recession * 1.2 (20% margin)
        target_graphite = max_recession_throat * 1.2
        bounds_gra = (max(0.003, target_graphite * 0.8), min(0.015, target_graphite * 1.5))
        layer3_bounds.append(bounds_gra)
        layer3_x0.append(float(graphite_cfg.initial_thickness))
        layer3_logger.info(
            "Graphite enabled: initial=%.3f mm, target≈%.3f mm, bounds=[%.3f, %.3f] mm",
            graphite_cfg.initial_thickness * 1000.0,
            target_graphite * 1000.0,
            bounds_gra[0] * 1000.0,
            bounds_gra[1] * 1000.0,
        )

    updated_time_results = full_time_results

    if len(layer3_x0) > 0:
        layer3_x0 = np.array(layer3_x0, dtype=float)

        # Track optimization evaluations for optional streaming to UI
        layer3_state: Dict[str, Any] = {
            "eval_index": 0,
            "best_objective": float("inf"),
        }

        def layer3_objective(x_layer3: np.ndarray) -> float:
            """Optimize thermal protection to minimize mass while meeting recession requirements."""
            eval_start = time.time()
            layer3_state["eval_index"] += 1
            eval_idx = int(layer3_state["eval_index"])

            try:
                x_layer3 = np.asarray(x_layer3, dtype=float)
                if not np.all(np.isfinite(x_layer3)):
                    layer3_logger.warning(
                        "Eval %d received non-finite thickness vector %s; returning large penalty.",
                        eval_idx,
                        repr(x_layer3),
                    )
                    return 1e6

                config_layer3 = copy.deepcopy(optimized_config)
                idx_param = 0
                chosen_thicknesses = []

                if ablative_cfg and ablative_cfg.enabled:
                    t_abl = float(np.clip(x_layer3[idx_param], layer3_bounds[idx_param][0], layer3_bounds[idx_param][1]))
                    config_layer3.ablative_cooling.initial_thickness = t_abl
                    chosen_thicknesses.append(t_abl)
                    idx_param += 1

                if graphite_cfg and graphite_cfg.enabled:
                    t_gra = float(
                        np.clip(x_layer3[idx_param], layer3_bounds[idx_param][0], layer3_bounds[idx_param][1])
                    )
                    config_layer3.graphite_insert.initial_thickness = t_gra
                    chosen_thicknesses.append(t_gra)

                runner_layer3 = PintleEngineRunner(config_layer3)
                results_layer3 = runner_layer3.evaluate_arrays_with_time(
                    time_array,
                    P_tank_O_array,
                    P_tank_F_array,
                    track_ablative_geometry=True,
                    use_coupled_solver=False,
                )

                recession_chamber = float(np.max(np.atleast_1d(results_layer3.get("recession_chamber", [0.0]))))
                recession_throat = float(np.max(np.atleast_1d(results_layer3.get("recession_throat", [0.0]))))

                # Check if recession exceeds thickness (with 20% margin)
                idx_param = 0
                recession_penalty = 0.0

                if ablative_cfg and ablative_cfg.enabled:
                    thickness = x_layer3[idx_param]
                    if recession_chamber > thickness * 0.8:  # 80% of thickness
                        recession_penalty += 1000.0 * (recession_chamber - thickness * 0.8)
                    idx_param += 1

                if graphite_cfg and graphite_cfg.enabled:
                    thickness = x_layer3[idx_param]
                    if recession_throat > thickness * 0.8:
                        recession_penalty += 1000.0 * (recession_throat - thickness * 0.8)

                # Objective: minimize mass (thickness) + recession penalty
                total_thickness = float(np.sum(x_layer3))
                obj = total_thickness * 1000.0 + recession_penalty  # Convert to mm for scaling

                # Optional: stream objective history to external callback (e.g., UI plot)
                if objective_callback is not None:
                    try:
                        best_obj = float(layer3_state["best_objective"])
                        if obj < best_obj:
                            best_obj = obj
                            layer3_state["best_objective"] = best_obj
                        objective_callback(eval_idx, float(obj), float(best_obj))
                    except Exception:
                        # Never let UI/consumer callback break the optimizer loop
                        pass

                eval_time = time.time() - eval_start
                layer3_logger.info(
                    "Eval %03d: thicknesses=%s mm, recession (chamber/throat)=%.3f/%.3f mm, "
                    "penalty=%.3f, obj=%.3f, dt=%.2fs",
                    eval_idx,
                    [t * 1000.0 for t in chosen_thicknesses],
                    recession_chamber * 1000.0,
                    recession_throat * 1000.0,
                    recession_penalty,
                    obj,
                    eval_time,
                )

                return obj
            except Exception as exc:
                eval_time = time.time() - eval_start
                layer3_logger.error("Exception in eval %03d (%.2fs): %r", eval_idx, eval_time, exc)
                import traceback

                layer3_logger.error(traceback.format_exc())
                return 1e6

        # Optimize Layer 3
        try:
            layer3_logger.info("Starting Layer 3 optimization using L-BFGS-B...")
            result_layer3 = scipy_minimize(
                layer3_objective,
                layer3_x0,
                method="L-BFGS-B",
                bounds=layer3_bounds,
                options={"maxiter": 30, "ftol": 1e-5},
            )
            layer3_logger.info(
                "Optimization finished: success=%s, final_obj=%.6f, nit=%s, nfev=%s",
                result_layer3.success,
                float(result_layer3.fun) if np.isfinite(result_layer3.fun) else float("nan"),
                getattr(result_layer3, "nit", "N/A"),
                getattr(result_layer3, "nfev", "N/A"),
            )

            # Update config with optimized thicknesses
            idx_param = 0
            if ablative_cfg and ablative_cfg.enabled:
                optimized_config.ablative_cooling.initial_thickness = float(
                    np.clip(result_layer3.x[idx_param], layer3_bounds[idx_param][0], layer3_bounds[idx_param][1])
                )
                thermal_results["optimized_ablative_thickness"] = optimized_config.ablative_cooling.initial_thickness
                update_progress(
                    "Layer 3: Burn Analysis Optimization",
                    0.70,
                    (
                        f"✓ Optimized ablative: "
                        f"{optimized_config.ablative_cooling.initial_thickness*1000:.2f}mm "
                        f"(recession: {max_recession_chamber*1000:.2f}mm)"
                    ),
                )
                layer3_logger.info(
                    "Optimized ablative thickness: %.3f mm",
                    optimized_config.ablative_cooling.initial_thickness * 1000.0,
                )
                idx_param += 1

            if graphite_cfg and graphite_cfg.enabled:
                optimized_config.graphite_insert.initial_thickness = float(
                    np.clip(result_layer3.x[idx_param], layer3_bounds[idx_param][0], layer3_bounds[idx_param][1])
                )
                thermal_results["optimized_graphite_thickness"] = optimized_config.graphite_insert.initial_thickness
                update_progress(
                    "Layer 3: Burn Analysis Optimization",
                    0.72,
                    (
                        f"✓ Optimized graphite: "
                        f"{optimized_config.graphite_insert.initial_thickness*1000:.2f}mm "
                        f"(recession: {max_recession_throat*1000:.2f}mm)"
                    ),
                )
                layer3_logger.info(
                    "Optimized graphite thickness: %.3f mm",
                    optimized_config.graphite_insert.initial_thickness * 1000.0,
                )
        except Exception as e:
            layer3_logger.error("Layer 3 optimization failed: %r", e)
            import traceback

            layer3_logger.error(traceback.format_exc())
            update_progress(
                "Layer 3: Burn Analysis Optimization",
                0.72,
                f"⚠️ Layer 3 optimization failed: {e}, using current values",
            )

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
            thermal_results["max_recession_chamber"] = float(
                np.max(np.atleast_1d(full_time_results_updated.get("recession_chamber", [0.0])))
            )
            thermal_results["max_recession_throat"] = float(
                np.max(np.atleast_1d(full_time_results_updated.get("recession_throat", [0.0])))
            )
            layer3_logger.info(
                "Post-optimization max recession (chamber / throat): "
                "%.3f mm / %.3f mm",
                thermal_results["max_recession_chamber"] * 1000.0,
                thermal_results["max_recession_throat"] * 1000.0,
            )
        except Exception as e:
            layer3_logger.error("Re-evaluation after optimization failed: %r", e)
            import traceback

            layer3_logger.error(traceback.format_exc())
            update_progress(
                "Layer 3: Burn Analysis",
                0.74,
                f"⚠️ Re-evaluation failed: {e}, using original results",
            )

        log_status(
            "Layer 3",
            (
                "Completed | Ablative {:.2f} mm, Graphite {:.2f} mm, "
                "Max recession chamber {:.2f} mm, throat {:.2f} mm "
                "(see {} for detailed log)".format(
                    (optimized_config.ablative_cooling.initial_thickness * 1000)
                    if ablative_cfg and ablative_cfg.enabled
                    else 0.0,
                    (optimized_config.graphite_insert.initial_thickness * 1000)
                    if graphite_cfg and graphite_cfg.enabled
                    else 0.0,
                    thermal_results.get("max_recession_chamber", 0.0) * 1000,
                    thermal_results.get("max_recession_throat", 0.0) * 1000,
                    log_file_path,
                )
            ),
        )

    layer3_logger.info("Layer 3 optimization complete. Log saved to: %s", log_file_path)
    # Clean up handlers so repeated calls don't leak file descriptors
    layer3_logger.handlers.clear()

    return optimized_config, updated_time_results, thermal_results

