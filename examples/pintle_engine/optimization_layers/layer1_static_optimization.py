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

from typing import Tuple, Callable
import numpy as np
import copy

from pintle_pipeline.config_schemas import PintleEngineConfig

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

