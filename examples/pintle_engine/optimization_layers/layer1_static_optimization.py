"""Layer 1: Static Optimization

This layer implements the main optimization loop that jointly optimizes:
- Engine geometry (throat, L*, expansion ratio, pintle geometry)
- Pressure curve parameters (segmented curves for LOX and fuel)
- Initial thermal protection guesses

This is where the pressure input curves are iterated over during optimization.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np
import copy

from pintle_pipeline.config_schemas import PintleEngineConfig
from pintle_models.runner import PintleEngineRunner

# Import helpers
from .helpers import segments_from_optimizer_vars

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


def create_layer1_apply_x_to_config(
    bounds: list,
    max_segments_per_tank: int,
    max_lox_P_psi: float,
    max_fuel_P_psi: float,
    target_burn_time: float,
    max_chamber_od: float,
    max_nozzle_exit: float,
) -> Callable:
    """Create the apply_x_to_config function with dependencies.
    
    Returns a function that converts optimizer variables to engine config.
    """
    
    def apply_x_to_config(x: np.ndarray, base_config: PintleEngineConfig) -> Tuple[PintleEngineConfig, float, float]:
        """Apply optimization variables to config. Returns (config, lox_end_ratio, fuel_end_ratio)."""
        config = copy.deepcopy(base_config)
        
        # Clip all values to bounds to ensure we stay within limits
        A_throat = float(np.clip(x[0], bounds[0][0], bounds[0][1]))
        Lstar = float(np.clip(x[1], bounds[1][0], bounds[1][1]))
        expansion_ratio = float(np.clip(x[2], bounds[2][0], bounds[2][1]))
        d_pintle_tip = float(np.clip(x[3], bounds[3][0], bounds[3][1]))
        h_gap = float(np.clip(x[4], bounds[4][0], bounds[4][1]))
        n_orifices = int(round(np.clip(x[5], bounds[5][0], bounds[5][1])))
        d_orifice = float(np.clip(x[6], bounds[6][0], bounds[6][1]))
        ablative_thickness = float(np.clip(x[7], bounds[7][0], bounds[7][1]))
        graphite_thickness = float(np.clip(x[8], bounds[8][0], bounds[8][1]))
        
        # Extract segment counts
        n_segments_lox = int(round(np.clip(x[9], bounds[9][0], bounds[9][1])))
        n_segments_fuel = int(round(np.clip(x[10], bounds[10][0], bounds[10][1])))
        
        # Extract segment parameters for LOX
        vars_per_segment = 5
        idx_base_lox = 11
        max_lox_idx = min(idx_base_lox + max_segments_per_tank * vars_per_segment, len(x))
        x_lox_segments = x[idx_base_lox:max_lox_idx]
        n_segments_lox = min(n_segments_lox, len(x_lox_segments) // vars_per_segment)
        if n_segments_lox < 1:
            n_segments_lox = 1
        lox_segments = segments_from_optimizer_vars(
            x_lox_segments, n_segments_lox, max_lox_P_psi, target_burn_time
        )
        
        # Extract segment parameters for Fuel
        idx_base_fuel = idx_base_lox + max_segments_per_tank * vars_per_segment
        max_fuel_idx = min(idx_base_fuel + max_segments_per_tank * vars_per_segment, len(x))
        x_fuel_segments = x[idx_base_fuel:max_fuel_idx]
        n_segments_fuel = min(n_segments_fuel, len(x_fuel_segments) // vars_per_segment)
        if n_segments_fuel < 1:
            n_segments_fuel = 1
        fuel_segments = segments_from_optimizer_vars(
            x_fuel_segments, n_segments_fuel, max_fuel_P_psi, target_burn_time
        )
        
        # Calculate end ratios
        if lox_segments:
            lox_start_psi = lox_segments[0]["start_pressure_psi"]
            lox_end_psi = lox_segments[-1]["end_pressure_psi"]
            lox_end_ratio = lox_end_psi / max_lox_P_psi if max_lox_P_psi > 0 else 0.7
            if lox_end_psi > lox_start_psi:
                lox_end_ratio = lox_start_psi / max_lox_P_psi if max_lox_P_psi > 0 else 0.7
        else:
            lox_end_ratio = 0.7
        
        if fuel_segments:
            fuel_start_psi = fuel_segments[0]["start_pressure_psi"]
            fuel_end_psi = fuel_segments[-1]["end_pressure_psi"]
            fuel_end_ratio = fuel_end_psi / max_fuel_P_psi if max_fuel_P_psi > 0 else 0.7
            if fuel_end_psi > fuel_start_psi:
                fuel_end_ratio = fuel_start_psi / max_fuel_P_psi if max_fuel_P_psi > 0 else 0.7
        else:
            fuel_end_ratio = 0.7
        
        # Store segments in config for later retrieval
        if not hasattr(config, '_optimizer_segments'):
            config._optimizer_segments = {}
        config._optimizer_segments['lox'] = lox_segments
        config._optimizer_segments['fuel'] = fuel_segments
        
        # Chamber geometry
        V_chamber = Lstar * A_throat
        D_chamber = max_chamber_od
        A_chamber = np.pi * (D_chamber / 2) ** 2
        R_chamber = D_chamber / 2
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
        if hasattr(config.chamber, 'chamber_inner_diameter'):
            config.chamber.chamber_inner_diameter = D_chamber
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
        if hasattr(config, 'ablative_cooling') and config.ablative_cooling and config.ablative_cooling.enabled:
            config.ablative_cooling.initial_thickness = ablative_thickness
        
        if hasattr(config, 'graphite_insert') and config.graphite_insert and config.graphite_insert.enabled:
            config.graphite_insert.initial_thickness = graphite_thickness
        
        return config, lox_end_ratio, fuel_end_ratio
    
    return apply_x_to_config

