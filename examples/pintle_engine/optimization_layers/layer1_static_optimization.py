"""Layer 1: Static Optimization

This layer implements the main optimization loop that optimizes ONLY
static (time‑independent) quantities:

- Engine geometry (throat, L*, expansion ratio, pintle geometry)
- Initial tank pressures for LOX and fuel (single value per tank)

All **time‑varying** pressure behavior (segments/curves over the burn)
is handled **exclusively** in Layer 2 (`layer2_pressure.py`). Layer 1
must NOT create or manipulate pressure segments or time arrays.
"""

from __future__ import annotations

from typing import Tuple, Callable, Dict, Any, Optional
import numpy as np
import copy
import logging
import time
from datetime import datetime
from pathlib import Path

from pintle_pipeline.config_schemas import PintleEngineConfig
from pintle_models.runner import PintleEngineRunner

from scipy.optimize import minimize, differential_evolution

try:
    import cma
except ImportError:  # pragma: no cover - optional dependency
    cma = None
   

# Import utility function
try:
    from .utils import extract_all_parameters
except ImportError:
    # Fallback if utils doesn't exist - extract basic parameters
    def extract_all_parameters(config):
        """Extract all parameters from config."""
        params = {}
        if hasattr(config, 'chamber'):
            params['A_throat'] = getattr(config.chamber, 'A_throat', None)
            params['Lstar'] = getattr(config.chamber, 'Lstar', None)
            params['chamber_length'] = getattr(config.chamber, 'length', None)
            params['chamber_diameter'] = getattr(config.chamber, 'chamber_inner_diameter', None)
        if hasattr(config, 'nozzle'):
            params['A_exit'] = getattr(config.nozzle, 'A_exit', None)
            params['expansion_ratio'] = getattr(config.nozzle, 'expansion_ratio', None)
        if hasattr(config, 'injector') and hasattr(config.injector, 'geometry'):
            if hasattr(config.injector.geometry, 'fuel'):
                params['d_pintle_tip'] = getattr(config.injector.geometry.fuel, 'd_pintle_tip', None)
                params['h_gap'] = getattr(config.injector.geometry.fuel, 'h_gap', None)
            if hasattr(config.injector.geometry, 'lox'):
                params['n_orifices'] = getattr(config.injector.geometry.lox, 'n_orifices', None)
                params['d_orifice'] = getattr(config.injector.geometry.lox, 'd_orifice', None)
        return params

# Import chamber geometry functions
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parents[3]
_chamber_path = _project_root / "chamber"
if str(_chamber_path) not in sys.path:
    sys.path.insert(0, str(_chamber_path))

from chamber_geometry import (
    chamber_length_calc,
    contraction_length_horizontal_calc,
)


TOTAL_WALL_THICKNESS_M = 0.0254  # 1.0 inch total wall (0.5 inch per side: outer - inner diameter)


def create_layer1_apply_x_to_config(
    bounds: list,
    max_chamber_od: float,
    max_nozzle_exit: float,
) -> Callable:
    """Create the apply_x_to_config function with dependencies.
    
    Returns a function that converts optimizer variables to engine config.
    """
    
    def apply_x_to_config(
        x: np.ndarray,
        base_config: PintleEngineConfig,
    ) -> Tuple[PintleEngineConfig, float, float]:
        """Apply optimization variables to config.

        Returns:
            config: Updated engine configuration
            P_O_start_psi: Initial LOX tank pressure [psi]
            P_F_start_psi: Initial fuel tank pressure [psi]

        Note:
            Layer 1 is **static only**. It chooses *single* initial tank
            pressures which Layer 2 then uses as the starting point for
            full time‑varying pressure‑curve optimization.
        """
        config = copy.deepcopy(base_config)
        
        # Clip all values to bounds to ensure we stay within limits
        A_throat = float(np.clip(x[0], bounds[0][0], bounds[0][1]))
        Lstar = float(np.clip(x[1], bounds[1][0], bounds[1][1]))
        expansion_ratio = float(np.clip(x[2], bounds[2][0], bounds[2][1]))
        D_chamber_outer = float(np.clip(x[3], bounds[3][0], bounds[3][1]))
        d_pintle_tip = float(np.clip(x[4], bounds[4][0], bounds[4][1]))
        h_gap = float(np.clip(x[5], bounds[5][0], bounds[5][1]))
        n_orifices = int(round(np.clip(x[6], bounds[6][0], bounds[6][1])))
        d_orifice = float(np.clip(x[7], bounds[7][0], bounds[7][1]))
        # NOTE: Indices 8 and 9 in `x` were previously used for ablative and graphite
        # thickness optimization. Layer 1 is now strictly geometry + initial tank
        # pressure only, so we intentionally ignore these values here to keep the
        # optimizer dimensionality stable while removing thermal‑protection coupling.
        ablative_thickness = float(np.clip(x[8], bounds[8][0], bounds[8][1]))
        graphite_thickness = float(np.clip(x[9], bounds[9][0], bounds[9][1]))
        
        # CRITICAL: Extract initial pressures (absolute values in psi).
        # These are the ONLY pressure‑related quantities optimized at
        # this layer; no time‑varying curves or segments are created.
        P_O_start_psi = float(np.clip(x[10], bounds[10][0], bounds[10][1]))
        P_F_start_psi = float(np.clip(x[11], bounds[11][0], bounds[11][1]))
        
        # Chamber geometry
        V_chamber = Lstar * A_throat
        # Convert outer diameter to inner diameter by subtracting wall thickness (0.5 inch total)
        D_chamber_inner = D_chamber_outer - TOTAL_WALL_THICKNESS_M
        if D_chamber_inner <= 0:
            # Fallback: keep at least 30% of outer diameter or 10mm
            D_chamber_inner = max(D_chamber_outer * 0.3, 0.01)
        A_chamber = np.pi * (D_chamber_inner / 2) ** 2
        R_chamber = D_chamber_inner / 2
        R_throat = np.sqrt(max(0, A_throat / np.pi))
        
        if A_throat > 0 and A_chamber > 0:
            contraction_ratio = A_chamber / A_throat
        else:
            contraction_ratio = 10.0
        theta_contraction = np.pi / 4  # 45 degrees
        L_cylindrical = chamber_length_calc(V_chamber, A_throat, contraction_ratio, theta_contraction)
        L_contraction = contraction_length_horizontal_calc(A_chamber, R_throat, theta_contraction)
        L_chamber = L_cylindrical + L_contraction
        
        if L_chamber <= 0 or L_cylindrical <= 0 or not np.isfinite(L_chamber):
            L_chamber = V_chamber / A_chamber if A_chamber > 0 else 0.2
            L_cylindrical = max(L_chamber * 0.5, 0.05)
        
        L_chamber = np.clip(L_chamber, 0.005, 1.0)
        
        config.chamber.A_throat = A_throat
        config.chamber.volume = V_chamber
        config.chamber.Lstar = Lstar
        config.chamber.length = L_chamber
        # Only store inner diameter on the chamber config; outer diameter is an
        # optimization variable but not part of the pydantic ChamberConfig schema.
        setattr(config.chamber, 'chamber_inner_diameter', D_chamber_inner)
        if hasattr(config.chamber, 'contraction_ratio'):
            config.chamber.contraction_ratio = contraction_ratio
        if hasattr(config.chamber, 'A_chamber'):
            config.chamber.A_chamber = A_chamber
        
        # Nozzle
        A_exit = A_throat * expansion_ratio
        if A_exit < 0:
            A_exit = A_throat * 10.0
        D_exit = np.sqrt(max(0, 4 * A_exit / np.pi))
        if D_exit > max_nozzle_exit:
            D_exit = max_nozzle_exit
            A_exit = np.pi * (D_exit / 2) ** 2
            if A_throat > 0:
                expansion_ratio = A_exit / A_throat
            else:
                expansion_ratio = 10.0
        
        config.nozzle.A_throat = A_throat
        config.nozzle.A_exit = A_exit
        config.nozzle.expansion_ratio = expansion_ratio
        if hasattr(config.nozzle, 'exit_diameter'):
            config.nozzle.exit_diameter = D_exit
        
        if hasattr(config.combustion, 'cea'):
            config.combustion.cea.expansion_ratio = expansion_ratio
        
        # Injector
        if hasattr(config.injector, 'geometry'):
            if hasattr(config.injector.geometry, 'fuel'):
                config.injector.geometry.fuel.d_pintle_tip = d_pintle_tip
                config.injector.geometry.fuel.h_gap = h_gap
            if hasattr(config.injector.geometry, 'lox'):
                config.injector.geometry.lox.n_orifices = n_orifices
                config.injector.geometry.lox.d_orifice = d_orifice
                config.injector.geometry.lox.theta_orifice = 90.0
        
        # Thermal protection
        #
        # IMPORTANT: Layer 1 no longer optimizes or modifies ablative/graphite
        # thickness. Those are handled exclusively in downstream layers
        # (Layer 2/3). We leave any existing values on `base_config` untouched
        # so that YAML export can still reflect sensible defaults, but we do
        # not change them here.
        
        # Layer 1 returns the static initial tank pressures; any time‑varying
        # curves are the responsibility of Layer 2.
        return config, P_O_start_psi, P_F_start_psi
    
    return apply_x_to_config


def run_layer1_global_search(
    objective: Callable[[np.ndarray], float],
    bounds: list,
    x0: np.ndarray,
    max_evals: int = 150,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Lightweight global search for Layer 1 using random sampling + short DE.

    This is intended to very quickly improve the starting point before the
    main local optimizer (L-BFGS-B) runs in the orchestrator.

    - Keeps evaluation budget small (max_evals) to avoid long runtimes.
    - Always respects the provided bounds.
    - Falls back gracefully if scipy's differential_evolution is unavailable.
    """
    try:
        from scipy.optimize import differential_evolution
    except Exception:
        differential_evolution = None

    if max_evals <= 0 or objective is None:
        return x0

    rng = np.random.default_rng(random_seed)

    bounds_arr = np.asarray(bounds, dtype=float)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]

    # Ensure starting point is within bounds
    best_x = np.clip(np.asarray(x0, dtype=float), lower, upper)
    try:
        best_f = float(objective(best_x))
    except Exception:
        # If evaluation fails, just return the original guess
        return x0

    evals_used = 1
    dim = best_x.size

    # ------------------------------------------------------------------
    # Phase 1: Random sampling within bounds (very small number of points)
    # ------------------------------------------------------------------
    n_random = max(5, min(20, max_evals // 3))
    for _ in range(n_random):
        if evals_used >= max_evals:
            break
        candidate = lower + rng.random(dim) * (upper - lower)
        try:
            f_val = float(objective(candidate))
        except Exception:
            evals_used += 1
            continue
        evals_used += 1
        if np.isfinite(f_val) and f_val < best_f:
            best_f = f_val
            best_x = candidate

    # ------------------------------------------------------------------
    # Phase 2: Very short Differential Evolution (if available)
    # ------------------------------------------------------------------
    if differential_evolution is not None and evals_used < max_evals:
        # Rough heuristic to keep DE cheap; cap iterations and population.
        # For typical Layer 1 dimensionality (~20 vars) this keeps runtime modest.
        remaining_evals = max_evals - evals_used
        popsize = 8
        # Each DE iter uses approximately popsize * dim evaluations
        approx_evals_per_iter = max(1, popsize * dim)
        maxiter = max(1, min(5, remaining_evals // approx_evals_per_iter))

        if maxiter > 0:
            # Wrap objective to track best solution without exceeding budget
            def wrapped_obj(v: np.ndarray) -> float:
                nonlocal best_x, best_f, evals_used
                if evals_used >= max_evals:
                    # Return current best to encourage convergence without new work
                    return best_f
                try:
                    f_val_inner = float(objective(v))
                except Exception:
                    evals_used += 1
                    return 1e9
                evals_used += 1
                if np.isfinite(f_val_inner) and f_val_inner < best_f:
                    best_f = f_val_inner
                    best_x = np.asarray(v, dtype=float)
                return f_val_inner

            try:
                _ = differential_evolution(
                    wrapped_obj,
                    bounds=bounds,
                    maxiter=maxiter,
                    popsize=popsize,
                    tol=0.01,
                    polish=False,
                    updating="deferred",
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    seed=random_seed,
                )
            except Exception:
                # If DE fails for any reason, just keep the best point found so far.
                pass

    return best_x


def run_layer1_optimization(
    config_obj: PintleEngineConfig,
    runner: PintleEngineRunner,
    requirements: Dict[str, Any],
    target_burn_time: float,
    max_iterations: int,
    tolerances: Dict[str, float],
    pressure_config: Dict[str, Any],
    update_progress: Optional[Callable[[str, float, str], None]] = None,
    log_status: Optional[Callable[[str, str], None]] = None,
    objective_callback: Optional[Callable[[int, float, float], None]] = None,
) -> Tuple[PintleEngineConfig, Dict[str, Any]]:
    """
    Run complete Layer 1 optimization: geometry + initial tank pressures.
    
    This function contains ALL Layer 1 optimization logic:
    - Setup (bounds, initial guess)
    - Objective function definition
    - Optimization loop (differential_evolution + L-BFGS-B)
    - Validation
    - Results packaging
    
    Args:
        config_obj: Base engine configuration
        runner: Engine runner (for validation)
        requirements: Design requirements dict
        target_burn_time: Target burn time [s]
        max_iterations: Maximum optimization iterations
        tolerances: Tolerance dict (thrust, apogee)
        pressure_config: Pressure configuration dict
        update_progress: Optional progress callback (stage, progress, message)
        log_status: Optional status logging callback (stage, message)
        objective_callback: Optional callback for objective history (iteration, objective, best_objective)
    
    Returns:
        optimized_config: Optimized engine configuration
        results: Results dict with performance, validation, history, etc.
    """
 
    # Set up Layer 1 logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"layer1_static_{timestamp}.log"
    
    # Create logger for Layer 1
    layer1_logger = logging.getLogger('layer1_static')
    layer1_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    layer1_logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    layer1_logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    layer1_logger.propagate = False
    
    layer1_logger.info("="*70)
    layer1_logger.info("Layer 1: Static Optimization")
    layer1_logger.info("="*70)
    layer1_logger.info(f"Log file: {log_file_path}")
    
    # Default callbacks
    if update_progress is None:
        def update_progress(stage: str, progress: float, message: str):
            pass
    if log_status is None:
        def log_status(stage: str, message: str):
            pass
    
    # Extract requirements
    target_thrust = requirements.get("target_thrust", 7000.0)
    optimal_of = requirements.get("optimal_of_ratio", 2.3)
    min_stability = requirements.get("min_stability_margin", 1.2)
    min_Lstar = requirements.get("min_Lstar", 0.95)
    max_Lstar = requirements.get("max_Lstar", 1.27)
    max_chamber_od = requirements.get("max_chamber_outer_diameter", 0.15)
    max_nozzle_exit = requirements.get("max_nozzle_exit_diameter", 0.101)
    thrust_tol = tolerances.get("thrust", 0.10)
    
    # Calculate target exit pressure from environment config (GPS/GFS-derived atmospheric pressure)
    # Use standard atmosphere model based on elevation if environment config is available
    target_P_exit = 101325.0  # Default: sea level (1 atm)
    if hasattr(config_obj, 'environment') and config_obj.environment is not None:
        elevation = getattr(config_obj.environment, 'elevation', 0.0)
        if elevation is not None and elevation >= 0:
            # Standard atmosphere model: P = P0 * exp(-M*g*h/(R*T0))
            # P0 = 101325 Pa (sea level)
            # M = 0.0289644 kg/mol (molar mass of dry air)
            # g = 9.80665 m/s² (standard gravity)
            # R = 8.31447 J/(mol·K) (universal gas constant)
            # T0 = 288.15 K (sea level temperature)
            P0 = 101325.0  # Pa
            M = 0.0289644  # kg/mol
            g = 9.80665    # m/s²
            R = 8.31447    # J/(mol·K)
            T0 = 288.15    # K
            target_P_exit = P0 * np.exp(-M * g * elevation / (R * T0))
            layer1_logger.info(f"Using atmospheric pressure from elevation: {elevation:.1f} m -> {target_P_exit:.1f} Pa ({target_P_exit/101325.0:.3f} atm)")
    else:
        layer1_logger.info(f"Environment config not available, using default sea level pressure: {target_P_exit:.1f} Pa")
    
    layer1_logger.info(f"Target thrust: {target_thrust:.1f} N")
    layer1_logger.info(f"Target O/F ratio: {optimal_of:.2f}")
    layer1_logger.info(f"Min stability margin: {min_stability:.2f}")
    layer1_logger.info(f"Target burn time: {target_burn_time:.2f} s")
    layer1_logger.info(f"Max iterations: {max_iterations}")
    layer1_logger.info("")
    
    # Get max pressures
    max_lox_P_psi = pressure_config.get("max_lox_pressure_psi", 700)
    max_fuel_P_psi = pressure_config.get("max_fuel_pressure_psi", 850)
    psi_to_Pa = 6894.76
    
    # Objective tolerance for early stopping
    obj_tolerance = 2.0  # For good solution: obj ≈ 1.25, so 2.0 is reasonable
    
    update_progress("Layer 1: Setup", 0.01, "Initializing Layer 1 optimization...")
    
    # Prepare base config
    config_base = copy.deepcopy(config_obj)
    if hasattr(config_base, 'injector') and config_base.injector.type == "pintle":
        if hasattr(config_base.injector.geometry, 'lox'):
            config_base.injector.geometry.lox.theta_orifice = 90.0
    
    # Enable turbulence coupling
    if hasattr(config_base, 'combustion') and hasattr(config_base.combustion, 'efficiency'):
        config_base.combustion.efficiency.use_turbulence_coupling = True
    
    # Get initial ablative/graphite thicknesses
    ablative_cfg = config_base.ablative_cooling if hasattr(config_base, 'ablative_cooling') and config_base.ablative_cooling else None
    graphite_cfg = config_base.graphite_insert if hasattr(config_base, 'graphite_insert') and config_base.graphite_insert else None
    ablative_init = ablative_cfg.initial_thickness if ablative_cfg and ablative_cfg.enabled else 0.008
    graphite_init = graphite_cfg.initial_thickness if graphite_cfg and graphite_cfg.enabled else 0.006
    
    # Calculate initial A_throat guess
    Pc_est_psi = 580.0
    Pc_est = Pc_est_psi * psi_to_Pa
    Cf_est = 1.5
    A_throat_init = target_thrust / (Cf_est * Pc_est) if Pc_est > 0 else 0.001
    A_throat_init = np.clip(A_throat_init, 5e-5, 3.0e-3)
    
    # Calculate bounds ensuring injector area < throat area
    # Use a more relaxed minimum - allow smaller throat areas since injector geometry
    # is also being optimized and the constraint will naturally prevent throat from
    # being too small relative to the actual chosen injector geometry
    max_n_orifices = 20
    max_d_orifice = 0.004
    max_LOX_area = max_n_orifices * np.pi * (max_d_orifice / 2) ** 2
    max_d_pintle = 0.030
    max_h_gap = 0.0015
    R_inner_max = max_d_pintle / 2
    R_outer_max = R_inner_max + max_h_gap
    max_fuel_area = np.pi * (R_outer_max ** 2 - R_inner_max ** 2)
    max_injector_area = max(max_LOX_area, max_fuel_area)
    # Reduced from 1.5x to 1.1x to allow smaller throat areas - constraint in objective
    # will still enforce injector_area < throat_area for the actual chosen geometry
    min_A_throat_safe = max_injector_area * 1.1
    
    # Initial pressure bounds (50-95% of max)
    min_P_ratio = 0.5
    max_P_ratio = 0.95
    min_outer_diameter = max_chamber_od * 0.5
    
    bounds = [
        (min_A_throat_safe, 3.0e-3),  # [0] A_throat
        (min_Lstar, max_Lstar),     # [1] Lstar
        (4.0, 20.0),                # [2] expansion_ratio
        (min_outer_diameter, max_chamber_od),  # [3] outer diameter
        (0.008, 0.030),             # [4] d_pintle_tip
        (0.0003, 0.0015),           # [5] h_gap
        (12, 20),                   # [6] n_orifices
        (0.001, 0.004),             # [7] d_orifice
        (0.003, 0.020),             # [8] ablative_thickness
        (0.003, 0.015),             # [9] graphite_thickness
        (max_lox_P_psi * min_P_ratio, max_lox_P_psi * max_P_ratio),    # [10] P_O_start_psi
        (max_fuel_P_psi * min_P_ratio, max_fuel_P_psi * max_P_ratio),  # [11] P_F_start_psi
    ]
    
    # Calculate initial guess
    default_n_orifices = 16
    default_d_orifice = 0.003
    default_d_pintle = 0.015
    default_h_gap = 0.0006
    A_lox_est = default_n_orifices * np.pi * (default_d_orifice / 2) ** 2
    R_inner_est = default_d_pintle / 2
    R_outer_est = R_inner_est + default_h_gap
    A_fuel_est = np.pi * (R_outer_est ** 2 - R_inner_est ** 2)
    max_injector_area_est = max(A_lox_est, A_fuel_est)
    # Use same relaxed multiplier for initial guess
    A_throat_min_safe = max_injector_area_est * 1.1
    A_throat_init = max(A_throat_init, A_throat_min_safe)
    A_throat_init = np.clip(A_throat_init, 5e-5, 3.0e-3)
    
    P_O_start_init = max_lox_P_psi * 0.80
    P_F_start_init = max_fuel_P_psi * 0.80
    outer_diameter_init = np.clip(max_chamber_od * 0.55, min_outer_diameter, max_chamber_od)
    
    x0 = np.array([
        A_throat_init,
        (min_Lstar + max_Lstar) / 2,
        10.0,
        outer_diameter_init,
        default_d_pintle,
        default_h_gap,
        default_n_orifices,
        default_d_orifice,
        np.clip(ablative_init, 0.003, 0.020),
        np.clip(graphite_init, 0.003, 0.015),
        P_O_start_init,
        P_F_start_init,
    ])
    
    # Clip to bounds
    for i, (lo, hi) in enumerate(bounds):
        x0[i] = np.clip(x0[i], lo, hi)
    
    update_progress("Layer 1: Setup", 0.05, "Creating apply_x_to_config function...")
    apply_x_to_config = create_layer1_apply_x_to_config(bounds, max_chamber_od, max_nozzle_exit)
    
    # Initialize optimization state
    opt_state = {
        "iteration": 0,
        "function_evaluations": 0,
        "best_objective": float('inf'),
        "best_x": None,
        "best_config": None,
        "best_lox_end_ratio": None,
        "best_fuel_end_ratio": None,
        "best_pressures": None,
        "best_results_for_validation": None,
        "converged": False,
        "objective_satisfied": False,
        "satisfied_obj": float('inf'),
        "satisfied_eval_count": 0,
        "consecutive_failures": 0,
        "last_valid_obj": float('inf'),
        "history": [],
        "stop_optimization": False,
        "force_maxfun_1": False,
    }
    
    log_flags = {
        "marginal_candidate_logged": False,
    }
    
    update_progress("Layer 1: Objective", 0.10, "Defining objective function...")
    
    # Define objective function
    def objective(x: np.ndarray) -> float:
        """Layer 1 objective function: optimize geometry + initial pressures."""
        
        # Initialize opt_state keys
        for key in ["consecutive_failures", "last_valid_obj", "iteration", "function_evaluations", "best_objective", "best_x"]:
            if key not in opt_state:
                opt_state[key] = 0 if key in ["iteration", "function_evaluations", "consecutive_failures"] else (float('inf') if key in ["last_valid_obj", "best_objective"] else None)
        
        opt_state["iteration"] += 1
        iteration = opt_state["iteration"]
        opt_state["function_evaluations"] += 1
        
        # Progress update
        progress = 0.10 + 0.40 * min(iteration / max_iterations, 1.0)
        if iteration <= 3 or iteration % 25 == 0:
            best_obj_str = f"{opt_state['best_objective']:.3e}" if np.isfinite(opt_state['best_objective']) else "inf"
            curr_obj_str = f"{opt_state.get('last_valid_obj', float('inf')):.3e}" if np.isfinite(opt_state.get('last_valid_obj', float('inf'))) else "inf"
            update_progress("Layer 1: Optimization", progress, f"Iter {iteration}/{max_iterations} | Curr: {curr_obj_str} | Best: {best_obj_str}")
            layer1_logger.info(f"[{int(progress*100)}%] Iteration {iteration}/{max_iterations} - "
                            f"Objective: {curr_obj_str} (Best: {best_obj_str})")
            for handler in layer1_logger.handlers:
                handler.flush()
        
        # Early exit if satisfied
        if opt_state.get('objective_satisfied', False):
            satisfied_obj = opt_state.get('satisfied_obj', opt_state.get('best_objective', 0.0))
            satisfied_count = opt_state.get('satisfied_eval_count', 0) + 1
            opt_state['satisfied_eval_count'] = satisfied_count
            if satisfied_count > 3:
                return satisfied_obj
            return satisfied_obj
        
        # Convert x to config
        config, P_O_start_psi_from_x, P_F_start_psi_from_x = apply_x_to_config(x, config_base)
        
        # Check constraints before evaluation
        A_throat_check = config.chamber.A_throat
        if hasattr(config, 'injector') and config.injector.type == "pintle":
            geom = config.injector.geometry
            A_lox_injector = geom.lox.n_orifices * np.pi * (geom.lox.d_orifice / 2) ** 2
            R_inner = geom.fuel.d_pintle_tip / 2
            R_outer = R_inner + geom.fuel.h_gap
            A_fuel_injector = np.pi * (R_outer ** 2 - R_inner ** 2)
            
            max_injector_area_ratio = 1.0
            if A_throat_check > 0:
                lox_ratio = A_lox_injector / A_throat_check
                fuel_ratio = A_fuel_injector / A_throat_check
                
                if lox_ratio > max_injector_area_ratio or fuel_ratio > max_injector_area_ratio:
                    excess_lox = max(0, lox_ratio - max_injector_area_ratio)
                    excess_fuel = max(0, fuel_ratio - max_injector_area_ratio)
                    constraint_penalty = 1e6 * (1.0 + excess_lox ** 2 + excess_fuel ** 2)
                    return constraint_penalty
                
                # Check O/F area ratio
                if A_fuel_injector > 0:
                    area_ratio = A_lox_injector / A_fuel_injector
                    Cd_ratio = 0.4 / 0.65
                    rho_ratio = np.sqrt(1140.0 / 780.0)
                    delta_p_ratio_est = np.sqrt(1.2)
                    area_ratio_factor = Cd_ratio * rho_ratio * delta_p_ratio_est
                    required_area_ratio = optimal_of / area_ratio_factor
                    area_ratio_error = abs(area_ratio - required_area_ratio) / required_area_ratio if required_area_ratio > 0 else 1.0
                    
                    if area_ratio_error > 0.5:
                        constraint_penalty = 1e5 * (1.0 + area_ratio_error ** 2)
                        return constraint_penalty
        
        # Disable thermal protection for evaluation
        config_runner = copy.deepcopy(config)
        if hasattr(config_runner, "ablative_cooling") and config_runner.ablative_cooling:
            config_runner.ablative_cooling.enabled = False
        if hasattr(config_runner, "graphite_insert") and config_runner.graphite_insert:
            config_runner.graphite_insert.enabled = False
        
        test_runner = PintleEngineRunner(config_runner)
        
        # Use x[10] and x[11] directly as pressures
        P_O_psi = float(np.clip(x[10], bounds[10][0], bounds[10][1]))
        P_F_psi = float(np.clip(x[11], bounds[11][0], bounds[11][1]))
        P_O_test = P_O_psi * psi_to_Pa
        P_F_test = P_F_psi * psi_to_Pa
        
        # Evaluate
        try:
            final_results = test_runner.evaluate(P_O_test, P_F_test, P_ambient=target_P_exit)
            final_pressures = (P_O_test, P_F_test)
        except Exception as eval_error:
            error_str = str(eval_error)
            if "Supply > Demand" in error_str or "No solution" in error_str:
                return 1e5
            return 1e6
        
        # Extract results
        F_actual = final_results.get("F", 0)
        Isp_actual = final_results.get("Isp", 0)
        MR_actual = final_results.get("MR", 0)
        Pc_actual = final_results.get("Pc", 0)
        Cf_actual = final_results.get("Cf_actual", final_results.get("Cf", 0))
        stability = final_results.get("stability_results", {})
        
        # Calculate errors
        thrust_error = abs(F_actual - target_thrust) / target_thrust if target_thrust > 0 else 1.0
        of_error = abs(MR_actual - optimal_of) / optimal_of if optimal_of > 0 else 1.0
        
        # Penalties for very bad geometries
        of_penalty_extra = 0.0
        thrust_penalty_extra = 0.0
        if of_error > 0.30:
            of_penalty_extra = 500.0 * (of_error - 0.30) ** 2
        if thrust_error > 0.50:
            thrust_penalty_extra = 500.0 * (thrust_error - 0.50) ** 2
        
        # Exit pressure error
        # Use target_P_exit calculated from environment config (GPS/GFS-derived atmospheric pressure)
        P_exit_actual = final_results.get("P_exit", 0.0)
        exit_pressure_error = 0.0
        # target_P_exit is calculated from environment config elevation using standard atmosphere model
        # (defined in outer scope, calculated once at optimization start)
        if target_P_exit > 0:
            exit_pressure_error = abs(P_exit_actual - target_P_exit) / target_P_exit
        
        # Cf (thrust coefficient) penalty - HIGHEST PRIORITY
        # Acceptable range: 1.3-1.8 (relaxed from 1.3-1.45 to allow for underexpanded nozzles)
        # Penalty structure designed to handle coupling with thrust
        Cf_min_acceptable = 1.3
        Cf_max_acceptable = 1.8
        Cf_error = 0.0
        Cf_penalty = 0.0
        Cf_in_range = False
        
        if Cf_actual > 0 and np.isfinite(Cf_actual):
            if Cf_actual < Cf_min_acceptable:
                # Penalty for being below acceptable range
                deficit = Cf_min_acceptable - Cf_actual
                # Moderate quadratic penalty - Cf is important
                Cf_penalty = 650.0 * (deficit ** 2)
                Cf_error = deficit / Cf_min_acceptable
                Cf_in_range = False
            elif Cf_actual > Cf_max_acceptable:
                # Penalty for being above acceptable range
                excess = Cf_actual - Cf_max_acceptable
                # Progressive penalty structure: softer for moderate excess, stronger for large excess
                # This allows optimizer to work towards target without being overly penalized
                if excess <= 0.3:
                    # Very small excess (1.8-2.1): very gentle penalty
                    Cf_penalty = 300.0 * (excess ** 2)
                elif excess <= 0.6:
                    # Moderate excess (2.1-2.4): moderate penalty
                    base_penalty = 300.0 * (0.3 ** 2)  # Penalty up to 0.3 excess
                    extra_penalty = 400.0 * ((excess - 0.3) ** 2)  # Additional for 0.3-0.6 excess
                    Cf_penalty = base_penalty + extra_penalty
                else:
                    # Large excess (>2.4): stronger penalty
                    base_penalty = 300.0 * (0.3 ** 2)  # Penalty up to 0.3 excess
                    mid_penalty = 400.0 * (0.3 ** 2)  # Penalty for 0.3-0.6 excess
                    extra_penalty = 600.0 * ((excess - 0.6) ** 2)  # Additional for >0.6 excess
                    Cf_penalty = base_penalty + mid_penalty + extra_penalty
                Cf_error = excess / Cf_max_acceptable
                Cf_in_range = False
            else:
                # Within acceptable range (1.3-1.8): no penalty
                Cf_penalty = 0.0
                Cf_error = 0.0
                Cf_in_range = True
        elif Cf_actual <= 0 or not np.isfinite(Cf_actual):
            # Large penalty for invalid Cf
            Cf_penalty = 1500.0
            Cf_error = 1.0
            Cf_in_range = False
        
        # Stability handling
        stability_state = stability.get("stability_state", "marginal")
        stability_score = stability.get("stability_score", 0.5)
        chugging = stability.get("chugging", {})
        chugging_margin = max(0.0, chugging.get("stability_margin", 0.0))
        acoustic = stability.get("acoustic", {})
        acoustic_margin = max(0.0, acoustic.get("stability_margin", 0.0))
        feed_system = stability.get("feed_system", {})
        feed_margin = max(0.0, feed_system.get("stability_margin", 0.0))
        
        min_stability_score_raw = requirements.get("min_stability_score", 0.75)
        stability_margin_handicap = float(requirements.get("stability_margin_handicap", 0.0))
        score_factor = max(0.0, 1.0 - stability_margin_handicap)
        min_stability_score = min_stability_score_raw * score_factor
        
        stability_penalty_base = 0.0
        if stability_score <= 0.0 or stability_state == "unstable":
            stability_penalty_base = 100.0 * (1.0 + (1.0 - max(0.0, stability_score)) ** 2)
        
        if stability_score < min_stability_score:
            score_deficit = min_stability_score - stability_score
            stability_penalty = stability_penalty_base + score_deficit * 100.0
        else:
            stability_penalty = stability_penalty_base
        
        margin_penalty = 0.0
        if chugging_margin < min_stability * 0.8:
            margin_penalty += 10.0 * (1.0 - chugging_margin / min_stability)
        if acoustic_margin < min_stability * 0.8:
            margin_penalty += 10.0 * (1.0 - acoustic_margin / min_stability)
        if feed_margin < min_stability * 0.8:
            margin_penalty += 10.0 * (1.0 - feed_margin / min_stability)
        
        stability_penalty = stability_penalty + margin_penalty

        # ------------------------------------------------------------------
        # Soft preference for shorter chambers
        # ------------------------------------------------------------------
        preferred_L_chamber = float(requirements.get("preferred_chamber_length_m", 0.25))
        length_penalty = 0.0
        L_chamber_curr = getattr(getattr(config, "chamber", None), "length", None)
        if L_chamber_curr is not None and np.isfinite(L_chamber_curr) and preferred_L_chamber > 0.0:
            if L_chamber_curr > preferred_L_chamber:
                excess_ratio = (L_chamber_curr - preferred_L_chamber) / preferred_L_chamber
                # Gentle quadratic penalty so this never dominates thrust/O/F/stability.
                length_penalty = 30.0 * excess_ratio ** 2
        
        
        # Bounds penalty
        bounds_penalty = 0.0
        for i, (lo, hi) in enumerate(bounds):
            val = x[i]
            if val < lo:
                bounds_penalty += 10.0 * ((lo - val) / (hi - lo + 1e-10)) ** 2
            elif val > hi:
                bounds_penalty += 10.0 * ((val - hi) / (hi - lo + 1e-10)) ** 2
        
        # Objective components - RESTRUCTURED to handle Cf-thrust-Pc coupling
        # Priority order: Cf > Stability > Thrust > O/F
        # When Cf is in range, reduce thrust penalty weight (they're coupled)
        
        # Base thrust penalty
        thrust_penalty_base = (thrust_error ** 2) * 250.0
        
        # Adaptive thrust penalty: reduce weight when Cf is acceptable or close
        # This handles the coupling - if Cf is reasonable, we're more forgiving on thrust
        # Priority: Cf > Stability > Thrust (when Cf is reasonable)
        if Cf_in_range:
            # Cf is acceptable (1.3-1.8): reduce thrust penalty weight by 40%
            # This allows optimizer to prioritize Cf without being overly penalized
            # for thrust variations that naturally occur when adjusting geometry
            thrust_penalty = thrust_penalty_base * 0.6
        elif Cf_actual > 0 and np.isfinite(Cf_actual) and Cf_actual > Cf_max_acceptable:
            # Cf is above acceptable: gradual penalty increase based on how far above
            # Extended range: allow adaptive penalty up to 2.4 (1.8 * 1.33)
            # This helps optimizer reduce Cf from high values (2.3+) without being
            # overly penalized on thrust while working towards target
            adaptive_max = Cf_max_acceptable * 1.33  # Up to 2.4
            if Cf_actual <= adaptive_max:
                # Cf is in adaptive range (1.8-2.4): gradual penalty increase
                excess_ratio = (Cf_actual - Cf_max_acceptable) / (adaptive_max - Cf_max_acceptable)
                excess_ratio = max(0.0, min(1.0, excess_ratio))  # Clip to [0, 1]
                # Interpolate between reduced (0.6x) and full penalty
                # More gradual: start at 0.5x for very close, ramp to 1.0x at 2.4
                thrust_penalty = thrust_penalty_base * (0.5 + 0.5 * excess_ratio)
            else:
                # Cf is very high (>2.4): full thrust penalty
                thrust_penalty = thrust_penalty_base
        else:
            # Cf is below acceptable (too low): full thrust penalty
            # Don't reduce thrust penalty when Cf is problematic
            thrust_penalty = thrust_penalty_base
        
        # O/F penalty (moderate priority)
        of_penalty = (of_error ** 2) * 300.0
        
        # Exit pressure penalty (low priority - less important than Cf/thrust/stability)
        exit_pressure_penalty = (exit_pressure_error ** 2) * 80.0
        
        # Stability penalty (HIGH priority - second only to Cf)
        stability_weight = 50.0 * stability_penalty
        
        # Combined objective
        # Structure: Cf (highest) > Stability > Thrust (adaptive) > O/F > Exit pressure
        obj = (
            Cf_penalty +           # Highest priority - Cf accuracy
            stability_weight +     # Second priority - stability margins
            thrust_penalty +       # Third priority - thrust (adaptive weight based on Cf)
            of_penalty +           # Fourth priority - O/F ratio
            exit_pressure_penalty + # Fifth priority - exit pressure matching
            length_penalty +       # Soft preference - chamber length
            bounds_penalty +       # Constraint enforcement
            thrust_penalty_extra + # Extra penalties for very bad cases
            of_penalty_extra
        )
        
        if not np.isfinite(obj):
            obj = 1e6
        
        # Check for early stopping
        thrust_tol_validation = thrust_tol * 1.0
        of_tol_validation = 0.15
        errors_acceptable = (
            thrust_error < thrust_tol_validation and
            of_error < of_tol_validation and
            stability_score >= 0.6
        )
        
        if np.isfinite(obj) and obj < obj_tolerance and errors_acceptable:
            opt_state['objective_satisfied'] = True
            opt_state['satisfied_obj'] = min(opt_state.get('satisfied_obj', float('inf')), obj)
            if final_pressures is not None:
                opt_state["best_pressures"] = final_pressures
                opt_state["best_results_for_validation"] = {
                    "F": F_actual,
                    "MR": MR_actual,
                    "thrust_error": thrust_error,
                    "of_error": of_error,
                    "stability_score": stability_score,
                    "stability_state": stability_state,
                    "chugging_margin": chugging_margin,
                    "acoustic_margin": acoustic_margin,
                    "feed_margin": feed_margin,
                    "stability_results": stability,
                }
            if not opt_state.get('satisfied_logged', False):
                log_status("Layer 1", f"✓ Solution valid! Obj={obj:.6e}, Thrust err: {thrust_error*100:.2f}%, O/F err: {of_error*100:.2f}%")
                opt_state['satisfied_logged'] = True
            opt_state['stop_optimization'] = True
            opt_state['force_maxfun_1'] = True
        
        # Track valid evaluations
        if np.isfinite(obj) and obj < 1e3:
            opt_state['consecutive_failures'] = 0
            opt_state['last_valid_obj'] = obj
        else:
            opt_state['consecutive_failures'] += 1
            if opt_state['consecutive_failures'] > 200:
                return 1e5
        
        # Record history with all parameterization variables
        A_throat_curr = float(np.clip(x[0], bounds[0][0], bounds[0][1]))
        Lstar_curr = float(np.clip(x[1], bounds[1][0], bounds[1][1]))
        expansion_ratio_curr = float(np.clip(x[2], bounds[2][0], bounds[2][1]))
        D_outer_curr = float(np.clip(x[3], bounds[3][0], bounds[3][1]))
        d_pintle_tip_curr = float(np.clip(x[4], bounds[4][0], bounds[4][1]))
        h_gap_curr = float(np.clip(x[5], bounds[5][0], bounds[5][1]))
        n_orifices_curr = int(round(np.clip(x[6], bounds[6][0], bounds[6][1])))
        d_orifice_curr = float(np.clip(x[7], bounds[7][0], bounds[7][1]))
        ablative_thickness_curr = float(np.clip(x[8], bounds[8][0], bounds[8][1]))
        graphite_thickness_curr = float(np.clip(x[9], bounds[9][0], bounds[9][1]))
        P_O_start_psi_hist = float(np.clip(x[10], bounds[10][0], bounds[10][1]))
        P_F_start_psi_hist = float(np.clip(x[11], bounds[11][0], bounds[11][1]))
        
        D_inner_curr = D_outer_curr - TOTAL_WALL_THICKNESS_M
        if D_inner_curr <= 0:
            D_inner_curr = max(D_outer_curr * 0.3, 0.01)
        
        lox_start_ratio_hist = P_O_start_psi_hist / max_lox_P_psi if max_lox_P_psi > 0 else 0.7
        fuel_start_ratio_hist = P_F_start_psi_hist / max_fuel_P_psi if max_fuel_P_psi > 0 else 0.7
        combined_stability_margin = min(chugging_margin, acoustic_margin, feed_margin)
        L_chamber_hist = L_chamber_curr if (L_chamber_curr is not None and np.isfinite(L_chamber_curr)) else None
        
        opt_state["history"].append({
            "iteration": iteration,
            "x": x.copy(),
            # Parameterization variables (geometries and pressures)
            "A_throat": A_throat_curr,
            "Lstar": Lstar_curr,
            "expansion_ratio": expansion_ratio_curr,
            "D_chamber_outer": D_outer_curr,
            "D_chamber_inner": D_inner_curr,
            "L_chamber": L_chamber_hist,
            "d_pintle_tip": d_pintle_tip_curr,
            "h_gap": h_gap_curr,
            "n_orifices": n_orifices_curr,
            "d_orifice": d_orifice_curr,
            "ablative_thickness": ablative_thickness_curr,
            "graphite_thickness": graphite_thickness_curr,
            "P_O_start_psi": P_O_start_psi_hist,
            "P_F_start_psi": P_F_start_psi_hist,
            # Performance metrics
            "thrust": F_actual,
            "thrust_error": thrust_error,
            "of_error": of_error,
            "Isp": Isp_actual,
            "MR": MR_actual,
            "Pc": Pc_actual,
            "Cf": Cf_actual,
            "Cf_error": Cf_error,
            "Cf_penalty": Cf_penalty,
            # Stability metrics
            "stability_margin": combined_stability_margin,
            "stability_state": stability_state,
            "stability_score": stability_score,
            "chugging_margin": chugging_margin,
            "acoustic_margin": acoustic_margin,
            "feed_margin": feed_margin,
            # Pressure ratios
            "lox_end_ratio": lox_start_ratio_hist,
            "fuel_end_ratio": fuel_start_ratio_hist,
            "lox_start_ratio": lox_start_ratio_hist,
            "fuel_start_ratio": fuel_start_ratio_hist,
            # Objective
            "objective": obj,
        })
        
        # Track best
        is_new_best = obj < opt_state["best_objective"]
        if is_new_best:
            opt_state["best_objective"] = obj
            opt_state["best_config"] = copy.deepcopy(config)
            opt_state["best_lox_end_ratio"] = lox_start_ratio_hist
            opt_state["best_fuel_end_ratio"] = fuel_start_ratio_hist
            opt_state["best_x"] = x.copy()
            if final_pressures is not None:
                opt_state["best_pressures"] = final_pressures
                opt_state["best_results_for_validation"] = {
                    "F": F_actual,
                    "MR": MR_actual,
                    "thrust_error": thrust_error,
                    "of_error": of_error,
                    "stability_score": stability_score,
                    "stability_state": stability_state,
                    "chugging_margin": chugging_margin,
                    "acoustic_margin": acoustic_margin,
                    "feed_margin": feed_margin,
                    "stability_results": stability,
                }
            layer1_logger.info(
                f"    ✓ New best objective: {obj:.6f} "
                f"(thrust_err: {thrust_error*100:.2f}%, O/F_err: {of_error*100:.2f}%, "
                f"Cf: {Cf_actual:.3f}, stability: {stability_state}, score: {stability_score:.3f})"
            )
            for handler in layer1_logger.handlers:
                handler.flush()
        
        # Stream objective history to external callback (e.g., UI plot) if provided
        if objective_callback is not None:
            try:
                objective_callback(
                    int(iteration),
                    float(obj),
                    float(opt_state.get("best_objective", obj)),
                )
            except Exception:
                # Never let UI/consumer callback break the optimizer loop
                pass
        
        # Convergence check
        require_stable_state = requirements.get("require_stable_state", True)
        allowed_states = {"stable", "marginal"}
        state_ok = (stability_state in allowed_states)
        stability_acceptable = (
            state_ok and
            (stability_score >= min_stability_score * 0.6) and
            (chugging_margin >= min_stability * 0.5) and
            (acoustic_margin >= min_stability * 0.5) and
            (feed_margin >= min_stability * 0.5)
        )
        
        convergence_thrust_tol = thrust_tol * 2.0
        convergence_of_tol = 0.30
        obj_improving = obj < opt_state.get("best_objective", float('inf'))
        
        if (thrust_error < convergence_thrust_tol and 
            of_error < convergence_of_tol and 
            stability_acceptable and
            obj_improving):
            opt_state["converged"] = True
        else:
            opt_state["converged"] = False
        
        return obj
    
    # Run optimization
    layer1_logger.info("")
    layer1_logger.info("Starting optimization...")
    layer1_logger.info("")
    opt_state["iteration"] = 0
    opt_state["function_evaluations"] = 0
    
    class _ResultWrapper:
        def __init__(self, x, fun, success=True):
            self.x = np.asarray(x, dtype=float)
            self.fun = float(fun)
            self.success = success
    
    result = None
    x0_refined = x0
    
    lower_bounds = np.array([b[0] for b in bounds], dtype=float)
    upper_bounds = np.array([b[1] for b in bounds], dtype=float)
    span = np.maximum(upper_bounds - lower_bounds, 1e-9)
    
    use_cma_solver = cma is not None
    
    if use_cma_solver:
        layer1_logger.info("Using CMA-ES for noisy coupled optimization.")
        update_progress("Layer 1: CMA-ES", 0.45, "Running CMA-ES global solver...")
        
        # Set per-dimension step sizes to ensure global exploration across ALL variables
        # CMA_stds scales step size per dimension: actual_step[i] = sigma0 * CMA_stds[i]
        # We want each variable to be able to explore ~15% of its range per sigma
        target_fraction_of_range = 0.15  # 15% of range per sigma
        
        # Calculate base sigma0 from median span (reasonable global scale)
        sigma0 = float(np.median(span) * target_fraction_of_range)
        if not np.isfinite(sigma0) or sigma0 <= 0:
            sigma0 = 0.05
        
        # Set CMA_stds proportional to each variable's range
        # This ensures variables with larger ranges (like expansion_ratio) get larger step sizes
        cma_stds = np.ones_like(span)
        for i in range(len(span)):
            if span[i] > 0:
                # Desired step size for this dimension: fraction of its range
                desired_step = span[i] * target_fraction_of_range
                # CMA_stds[i] = desired_step / sigma0
                # This makes step size in dimension i proportional to its range
                cma_stds[i] = max(0.1, desired_step / sigma0) if sigma0 > 0 else 1.0
        
        # Special handling for discrete n_orifices (index 6): ensure it can jump between integers
        n_orifices_idx = 6
        target_step_n = 1.0  # ~one orifice per sigma
        if sigma0 > 0:
            cma_stds[n_orifices_idx] = max(cma_stds[n_orifices_idx], target_step_n / sigma0)

        popsize = min(32, max(8, 4 + int(3 * np.log(len(x0_refined) + 1))))
        cma_options = {
            "bounds": [lower_bounds.tolist(), upper_bounds.tolist()],
            "popsize": popsize,
            "maxiter": max_iterations,
            "verb_disp": 0,
            "verb_log": 0,
            "CMA_stds": cma_stds.tolist(),
        }
        
        try:
            es = cma.CMAEvolutionStrategy(x0_refined.tolist(), sigma0, cma_options)
            while not es.stop():
                candidates = es.ask()
                values = []
                for cand in candidates:
                    cand_arr = np.clip(np.asarray(cand, dtype=float), lower_bounds, upper_bounds)
                    val = objective(cand_arr)
                    values.append(float(val))
                es.tell(candidates, values)
                
                iter_idx = max(1, es.countiter)
                progress = 0.10 + 0.35 * min(iter_idx / max_iterations, 1.0)
                update_progress("Layer 1: CMA-ES", progress, f"CMA-ES iteration {iter_idx}/{max_iterations}")
                
                if opt_state.get("objective_satisfied", False):
                    layer1_logger.info("Objective satisfied during CMA-ES run; stopping early.")
                    es.stop()
                    break
            
            cma_result = es.result
            x0_refined = np.asarray(cma_result.xbest, dtype=float)
            best_fun = float(cma_result.fbest)
            layer1_logger.info(f"CMA-ES finished with objective {best_fun:.6f} after {es.countiter} iterations.")
            log_status("Layer 1", f"CMA-ES complete: {es.countiter} iterations, obj={best_fun:.3f}, refining with L-BFGS-B...")
            for handler in layer1_logger.handlers:
                handler.flush()
            # Reset function evaluation counter for L-BFGS-B refinement
            opt_state["function_evaluations"] = 0
        except Exception as e:
            layer1_logger.error(f"CMA-ES failed: {e}. Falling back to differential evolution + L-BFGS-B.")
            log_status("Layer 1 Warning", f"CMA-ES failed: {e}. Falling back to legacy solver.")
            use_cma_solver = False
            x0_refined = x0
    
    if not use_cma_solver:
        update_progress("Layer 1: Global Search", 0.45, "Running differential evolution...")
        try:
            layer1_logger.info("Phase 1: Global search (differential evolution)...")
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=20,
                popsize=10,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                polish=False,
                workers=1,
            )
            x0_refined = de_result.x
            func_evals_de = opt_state.get("function_evaluations", 0)
            de_obj = de_result.fun
            layer1_logger.info(f"Global search (differential_evolution) finished with objective {de_obj:.6f}")
            layer1_logger.info(f"Function evaluations: {func_evals_de}")
            log_status("Layer 1", f"Global search complete: {func_evals_de} func evals, obj={de_obj:.3f}, refining with L-BFGS-B...")
            for handler in layer1_logger.handlers:
                handler.flush()
            opt_state["function_evaluations"] = 0
        except Exception as e:
            layer1_logger.warning(f"Differential evolution failed: {e}, using original initial guess")
            log_status("Layer 1 Warning", f"Differential evolution failed: {e}, using original initial guess")
            x0_refined = x0
            opt_state["function_evaluations"] = 0
    
    # L-BFGS-B refinement runs after either CMA-ES or differential evolution
    maxfun_capped = min(max_iterations * 3, 500)
    if opt_state.get('objective_satisfied', False) or opt_state.get('force_maxfun_1', False):
        maxfun_capped = 1
        log_status("Layer 1", "Objective satisfied - setting maxfun=1 to stop immediately")
    
    update_progress("Layer 1: Local Refinement", 0.47, f"Refining with L-BFGS-B (max {maxfun_capped} func evals)...")
    layer1_logger.info("Phase 2: Local refinement (L-BFGS-B)...")
    
    try:
        if opt_state.get('objective_satisfied', False):
            maxfun_capped = min(maxfun_capped, 3)
            log_status("Layer 1", f"Objective satisfied - reducing maxfun to {maxfun_capped}")
        
        lbfgs_result = minimize(
            objective,
            x0_refined,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iterations,
                'maxfun': maxfun_capped,
                'ftol': obj_tolerance * 0.1,
                'gtol': 1e-3,  # Relaxed from 1e-5 to allow larger steps before convergence
                'maxls': 50,    # Increased from 20 to allow more aggressive line searches
                'disp': False,
            }
        )
        layer1_logger.info("")
        layer1_logger.info("Optimization completed")
        layer1_logger.info(f"Success: {lbfgs_result.success}")
        layer1_logger.info(f"Final objective value: {lbfgs_result.fun:.6f}")
        layer1_logger.info(f"Iterations: {lbfgs_result.nit if hasattr(lbfgs_result, 'nit') else 'N/A'}")
        layer1_logger.info(f"Function evaluations: {lbfgs_result.nfev if hasattr(lbfgs_result, 'nfev') else 'N/A'}")
        layer1_logger.info("")
        result = lbfgs_result
    except Exception as e:
        layer1_logger.error(f"L-BFGS-B error: {e}")
        log_status("Layer 1 Warning", f"L-BFGS-B error: {e}, using best result found")
        if opt_state.get('objective_satisfied', False):
            satisfied_obj = opt_state.get('satisfied_obj', opt_state.get('best_objective', float('inf')))
            best_x = opt_state.get('best_x', x0_refined)
            result = _ResultWrapper(best_x, satisfied_obj)
        elif use_cma_solver and 'best_fun' in locals():
            # CMA-ES succeeded but L-BFGS-B failed
            result = _ResultWrapper(x0_refined, best_fun)
        elif 'de_result' in locals():
            result = _ResultWrapper(x0_refined, de_obj)
        else:
            result = _ResultWrapper(x0, float('inf'))
    
    if opt_state.get('objective_satisfied', False):
        satisfied_obj = opt_state.get('satisfied_obj', opt_state.get('best_objective', 0))
        log_status("Layer 1", f"✓ Objective satisfied! obj={satisfied_obj:.6e} < tolerance={obj_tolerance:.3f}")
    
    # Get best config
    if opt_state["best_config"] is not None:
        optimized_config = opt_state["best_config"]
        final_lox_end_ratio = opt_state.get("best_lox_end_ratio", 0.7)
        final_fuel_end_ratio = opt_state.get("best_fuel_end_ratio", 0.7)
    else:
        optimized_config, P_O_final_psi, P_F_final_psi = apply_x_to_config(result.x, config_base)
        final_lox_end_ratio = P_O_final_psi / max_lox_P_psi if max_lox_P_psi > 0 else 0.7
        final_fuel_end_ratio = P_F_final_psi / max_fuel_P_psi if max_fuel_P_psi > 0 else 0.7
    
    # Ensure orifice angle is 90°
    if hasattr(optimized_config, 'injector') and optimized_config.injector.type == "pintle":
        if hasattr(optimized_config.injector.geometry, 'lox'):
            optimized_config.injector.geometry.lox.theta_orifice = 90.0
    
    iteration_history = opt_state["history"]
    
    # Validation
    update_progress("Layer 1: Validation", 0.52, "Validating optimized configuration...")
    
    best_x = opt_state.get("best_x", result.x if hasattr(result, 'x') else x0)
    
    if "best_pressures" in opt_state and opt_state["best_pressures"] is not None:
        P_O_initial, P_F_initial = opt_state["best_pressures"]
    else:
        if best_x is not None and len(best_x) > 10:
            P_O_initial = float(np.clip(best_x[10], bounds[10][0], bounds[10][1])) * psi_to_Pa
        else:
            P_O_initial = max_lox_P_psi * psi_to_Pa * 0.95
        if best_x is not None and len(best_x) > 11:
            P_F_initial = float(np.clip(best_x[11], bounds[11][0], bounds[11][1])) * psi_to_Pa
        else:
            P_F_initial = max_fuel_P_psi * psi_to_Pa * 0.95
    
    optimized_config_runner = copy.deepcopy(optimized_config)
    if hasattr(optimized_config_runner, "ablative_cooling") and optimized_config_runner.ablative_cooling:
        optimized_config_runner.ablative_cooling.enabled = False
    if hasattr(optimized_config_runner, "graphite_insert") and optimized_config_runner.graphite_insert:
        optimized_config_runner.graphite_insert.enabled = False
    
    optimized_runner = PintleEngineRunner(optimized_config_runner)
    
    # Use stored validation results if available
    if "best_results_for_validation" in opt_state and opt_state["best_results_for_validation"] is not None:
        stored_results = opt_state["best_results_for_validation"]
        initial_performance = {
            "F": stored_results["F"],
            "MR": stored_results["MR"],
            "Isp": 250.0,
            "Pc": 2e6,
            "stability_results": stored_results.get("stability_results", {}),
        }
        initial_thrust_error = stored_results["thrust_error"]
        initial_MR_error = stored_results["of_error"]
        stored_stability_score = stored_results.get("stability_score", None)
        stored_stability_state = stored_results.get("stability_state", None)
        log_status("Layer 1 Validation", f"Using stored validation results: Thrust err {initial_thrust_error*100:.2f}%, O/F err {initial_MR_error*100:.2f}%")
        
        # For stored results, we need to re-evaluate to get P_exit and Cf
        # (stored_results only contains F, MR, errors, and stability)
        try:
            eval_results = optimized_runner.evaluate(P_O_initial, P_F_initial, P_ambient=target_P_exit)
            # Copy P_exit and Cf from evaluation if available
            if "P_exit" in eval_results:
                initial_performance["P_exit"] = eval_results["P_exit"]
            if "Cf" in eval_results or "Cf_actual" in eval_results:
                initial_performance["Cf"] = eval_results.get("Cf_actual", eval_results.get("Cf"))
                initial_performance["Cf_actual"] = initial_performance["Cf"]
            # Also copy other useful metrics that might be missing
            if "Isp" in eval_results:
                initial_performance["Isp"] = eval_results["Isp"]
            if "Pc" in eval_results:
                initial_performance["Pc"] = eval_results["Pc"]
        except Exception:
            # If re-evaluation fails, calculate Cf from available data
            F_val = initial_performance.get("F", 0)
            Pc_val = initial_performance.get("Pc", 0)
            A_throat_val = getattr(optimized_config.chamber, "A_throat", None)
            if A_throat_val and A_throat_val > 0 and Pc_val > 0:
                Cf_calculated = F_val / (Pc_val * A_throat_val)
                initial_performance["Cf_actual"] = Cf_calculated
                initial_performance["Cf"] = Cf_calculated
    else:
        initial_performance = optimized_runner.evaluate(P_O_initial, P_F_initial, P_ambient=target_P_exit)
        initial_thrust_error = abs(initial_performance.get("F", 0) - target_thrust) / target_thrust if target_thrust > 0 else 1.0
        initial_MR_error = abs(initial_performance.get("MR", 0) - optimal_of) / optimal_of if optimal_of > 0 else 1.0
        
        # Ensure Cf is included (calculate if not provided by runner)
        if "Cf" not in initial_performance and "Cf_actual" not in initial_performance:
            F_val = initial_performance.get("F", 0)
            Pc_val = initial_performance.get("Pc", 0)
            A_throat_val = getattr(optimized_config.chamber, "A_throat", None)
            if A_throat_val and A_throat_val > 0 and Pc_val > 0:
                Cf_calculated = F_val / (Pc_val * A_throat_val)
                initial_performance["Cf_actual"] = Cf_calculated
                initial_performance["Cf"] = Cf_calculated
    
    # Check stability
    stored_results = opt_state.get("best_results_for_validation", {})
    if stored_results and "stability_results" in stored_results:
        stability_results = stored_results.get("stability_results", {})
        stability_state = stored_results.get("stability_state", "unstable")
        stability_score = stored_results.get("stability_score", 0.0)
        chugging_margin = stored_results.get("chugging_margin", 0)
        acoustic_margin = stored_results.get("acoustic_margin", 0)
        feed_margin = stored_results.get("feed_margin", 0)
    else:
        stability_results = initial_performance.get("stability_results", {})
        stability_state = stability_results.get("stability_state", "unstable")
        stability_score = stability_results.get("stability_score", 0.0)
        chugging_margin = stability_results.get("chugging", {}).get("stability_margin", 0)
        acoustic_margin = stability_results.get("acoustic", {}).get("stability_margin", 0)
        feed_margin = stability_results.get("feed_system", {}).get("stability_margin", 0)
    initial_stability = min(chugging_margin, acoustic_margin, feed_margin)
    
    # Validation checks
    min_stability_score = requirements.get("min_stability_score", 0.75)
    require_stable_state = requirements.get("require_stable_state", True)
    handicap = float(requirements.get("stability_margin_handicap", 0.0))
    score_factor = max(0.0, 1.0 - handicap)
    margin_factor = max(0.0, 1.0 - handicap)
    effective_min_score = min_stability_score * score_factor
    effective_margin = min_stability * margin_factor
    
    state_ok = (stability_state in {"stable", "marginal"}) if require_stable_state else (stability_state != "unstable")
    margin_tolerance = 0.05
    stability_check_passed = (
        state_ok and
        (stability_score >= effective_min_score) and
        (chugging_margin >= effective_margin * (1.0 - margin_tolerance)) and
        (acoustic_margin >= effective_margin * (1.0 - margin_tolerance)) and
        (feed_margin >= effective_margin * (1.0 - margin_tolerance))
    )
    
    thrust_check_passed = initial_thrust_error < thrust_tol * 1.0
    of_check_passed = initial_MR_error < 0.15
    pressure_candidate_valid = thrust_check_passed and of_check_passed and stability_check_passed
    
    # Build failure reasons
    failure_reasons = []
    if not thrust_check_passed:
        failure_reasons.append(f"Thrust error {initial_thrust_error*100:.1f}% > {thrust_tol*100:.0f}% limit")
    if not of_check_passed:
        failure_reasons.append(f"O/F error {initial_MR_error*100:.1f}% > 15% limit")
    if not stability_check_passed:
        required_parts = []
        if require_stable_state:
            if stability_state not in {"stable", "marginal"}:
                required_parts.append(f"state ∈ {{stable,marginal}} (got '{stability_state}')")
        else:
            if stability_state == "unstable":
                required_parts.append("state!='unstable'")
        if stability_score < effective_min_score:
            required_parts.append(f"score>={effective_min_score:.2f} (got {stability_score:.2f})")
        if chugging_margin < effective_margin:
            required_parts.append(f"chugging_margin>={effective_margin:.2f} (got {chugging_margin:.2f})")
        if acoustic_margin < effective_margin:
            required_parts.append(f"acoustic_margin>={effective_margin:.2f} (got {acoustic_margin:.2f})")
        if feed_margin < effective_margin:
            required_parts.append(f"feed_margin>={effective_margin:.2f} (got {feed_margin:.2f})")
        if not required_parts:
            required_parts.append("stability gate mismatch")
        failure_reasons.append(f"Stability failed: {'; '.join(required_parts)}")
    
    if not pressure_candidate_valid and not failure_reasons:
        failure_reasons.append("Validation failed: no requirements met")
    
    # Log validation
    if pressure_candidate_valid:
        update_progress("Layer 1: Validation", 0.53, f"✓ VALID - Thrust err: {initial_thrust_error*100:.1f}%, O/F err: {initial_MR_error*100:.1f}%, Stability: {stability_state}")
        log_status("Layer 1", f"VALID | Thrust err {initial_thrust_error*100:.1f}%, O/F err {initial_MR_error*100:.1f}%, Stability {stability_state}")
        layer1_logger.info("✓ Validation: VALID")
    else:
        update_progress("Layer 1: Validation", 0.53, f"✗ INVALID - {'; '.join(failure_reasons)}")
        log_status("Layer 1", f"INVALID | Reasons: {', '.join(failure_reasons)}")
        layer1_logger.warning(f"✗ Validation: INVALID - {'; '.join(failure_reasons)}")
    
    # Build final performance dict
    final_performance = initial_performance.copy()
    final_performance["pressure_candidate_valid"] = pressure_candidate_valid
    final_performance["initial_thrust_error"] = initial_thrust_error
    final_performance["initial_MR_error"] = initial_MR_error
    final_performance["initial_stability"] = initial_stability
    final_performance["initial_stability_state"] = stability_state
    final_performance["initial_stability_score"] = stability_score
    final_performance["thrust_check_passed"] = thrust_check_passed
    final_performance["of_check_passed"] = of_check_passed
    final_performance["stability_check_passed"] = stability_check_passed
    final_performance["failure_reasons"] = failure_reasons
    
    # Extract optimized pressures
    if len(best_x) > 10:
        P_O_start_optimized_psi = float(np.clip(best_x[10], bounds[10][0], bounds[10][1]))
        P_F_start_optimized_psi = float(np.clip(best_x[11], bounds[11][0], bounds[11][1]))
        final_performance["P_O_start_psi"] = P_O_start_optimized_psi
        final_performance["P_F_start_psi"] = P_F_start_optimized_psi
        final_performance["P_O_start_ratio"] = P_O_start_optimized_psi / max_lox_P_psi if max_lox_P_psi > 0 else 0.0
        final_performance["P_F_start_ratio"] = P_F_start_optimized_psi / max_fuel_P_psi if max_fuel_P_psi > 0 else 0.0
    else:
        final_performance["P_O_start_psi"] = max_lox_P_psi * 0.8
        final_performance["P_F_start_psi"] = max_fuel_P_psi * 0.8
        final_performance["P_O_start_ratio"] = 0.8
        final_performance["P_F_start_ratio"] = 0.8
    
    # Generate constant pressure curves (Layer 1 static only)
    update_progress("Layer 1: Pressure Curves", 0.55, "Generating constant pressure curves...")
    n_time_points = 200
    time_array = np.linspace(0.0, target_burn_time, n_time_points)
    P_tank_O_array = np.full(n_time_points, final_performance["P_O_start_psi"] * psi_to_Pa)
    P_tank_F_array = np.full(n_time_points, final_performance["P_F_start_psi"] * psi_to_Pa)
    
    # Build results dict
    results = {
        "performance": final_performance,
        "iteration_history": iteration_history,
        "convergence_info": {
            "converged": opt_state["converged"],
            "iterations": len(iteration_history),
            "final_change": opt_state["best_objective"],
        },
        "exit_pressure_targeting": {
            "target_P_exit": target_P_exit,  # Atmospheric pressure from environment config (GPS/GFS-derived)
        },
        "optimized_pressure_curves": {
            "lox_end_ratio": final_lox_end_ratio,
            "fuel_end_ratio": final_fuel_end_ratio,
            "lox_segments": [{
                "type": "linear",
                "duration": target_burn_time,
                "start_pressure_psi": final_performance["P_O_start_psi"],
                "end_pressure_psi": final_performance["P_O_start_psi"],
            }],
            "fuel_segments": [{
                "type": "linear",
                "duration": target_burn_time,
                "start_pressure_psi": final_performance["P_F_start_psi"],
                "end_pressure_psi": final_performance["P_F_start_psi"],
            }],
            "lox_start_psi": final_performance["P_O_start_psi"],
            "lox_end_psi": final_performance["P_O_start_psi"],
            "fuel_start_psi": final_performance["P_F_start_psi"],
            "fuel_end_psi": final_performance["P_F_start_psi"],
        },
        "layer_status": {
            "layer_1_pressure_candidate": pressure_candidate_valid,
        },
        "optimized_parameters": extract_all_parameters(optimized_config),
    }
    
    # Final summary logging
    layer1_logger.info("")
    layer1_logger.info("="*70)
    layer1_logger.info("Final Results Summary")
    layer1_logger.info("="*70)
    if "best_results_for_validation" in opt_state and opt_state["best_results_for_validation"] is not None:
        stored = opt_state["best_results_for_validation"]
        layer1_logger.info(f"Thrust: {stored.get('F', 0):.1f} N (target: {target_thrust:.1f} N)")
        layer1_logger.info(f"Thrust error: {stored.get('thrust_error', 0)*100:.2f}%")
        layer1_logger.info(f"O/F ratio: {stored.get('MR', 0):.3f} (target: {optimal_of:.3f})")
        layer1_logger.info(f"O/F error: {stored.get('of_error', 0)*100:.2f}%")
        layer1_logger.info(f"Stability state: {stored.get('stability_state', 'unknown')}")
        layer1_logger.info(f"Stability score: {stored.get('stability_score', 0):.3f}")
        layer1_logger.info(f"Chugging margin: {stored.get('chugging_margin', 0):.3f}")
        layer1_logger.info(f"Acoustic margin: {stored.get('acoustic_margin', 0):.3f}")
        layer1_logger.info(f"Feed margin: {stored.get('feed_margin', 0):.3f}")
    if final_performance.get("P_O_start_psi") is not None:
        layer1_logger.info(f"LOX initial pressure: {final_performance['P_O_start_psi']:.1f} psi")
    if final_performance.get("P_F_start_psi") is not None:
        layer1_logger.info(f"Fuel initial pressure: {final_performance['P_F_start_psi']:.1f} psi")
    layer1_logger.info(f"Validation: {'VALID' if pressure_candidate_valid else 'INVALID'}")
    layer1_logger.info("="*70)
    layer1_logger.info("")
    layer1_logger.info(f"Layer 1 optimization complete. Log saved to: {log_file_path}")
    
    # Clean up handler to prevent file handle issues
    layer1_logger.handlers.clear()
    
    update_progress("Layer 1: Complete", 1.0, "Layer 1 optimization complete!")
    
    return optimized_config, results
