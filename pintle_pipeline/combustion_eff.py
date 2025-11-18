"""Combustion efficiency models (L* correction)

This module provides both:
1. Simple efficiency model (eta_cstar) - backward compatible
2. Advanced physics-based model (via combustion_physics module)
"""

import numpy as np
from typing import Optional, Dict, Any
from .config_schemas import CombustionEfficiencyConfig
from .constants import (
    DEFAULT_CHAMBER_PRESS_PA,
    DEFAULT_CHAMBER_TEMP_K,
    DEFAULT_CSTAR_IDEAL_M_S,
    DEFAULT_GAMMA_ND,
    DEFAULT_GAS_CONST_J_KG_K,
    DEFAULT_MIXTURE_RATIO_ND,
    DEFAULT_TURBULENCE_INTENSITY_ND,
)


def calculate_Lstar(
    V_chamber: float,
    A_throat: float,
    Lstar_override: Optional[float] = None
) -> float:
    """
    Calculate characteristic length L*.
    
    L* = V_chamber / A_throat
    
    Parameters:
    -----------
    V_chamber : float
        Chamber volume [m³]
    A_throat : float
        Throat area [m²]
    Lstar_override : float, optional
        Override value if provided in config
    
    Returns:
    --------
    Lstar : float [m]
    """
    if Lstar_override is not None:
        return float(Lstar_override)
    
    if A_throat <= 0:
        raise ValueError("A_throat must be positive")
    
    Lstar = V_chamber / A_throat
    return float(Lstar)


def eta_cstar(
    Lstar: float,
    config: CombustionEfficiencyConfig,
    spray_quality_good: bool = True,
    mixture_efficiency: float = 1.0,
    cooling_efficiency: float = 1.0,
    use_advanced_model: bool = True,
    advanced_params: Optional[Dict[str, Any]] = {
        "Pc": DEFAULT_CHAMBER_PRESS_PA,
        "Tc": DEFAULT_CHAMBER_TEMP_K,
        "cstar_ideal": DEFAULT_CSTAR_IDEAL_M_S,
        "gamma": DEFAULT_GAMMA_ND,
        "R": DEFAULT_GAS_CONST_J_KG_K,
        "MR": DEFAULT_MIXTURE_RATIO_ND,
        "Ac": 0.005666174318,
        "m_dot_total": 3,
        "Dinj": 0.002,
        "spray_diagnostics": None,
        "turbulence_intensity": DEFAULT_TURBULENCE_INTENSITY_ND,
    },
) -> float:
    """
    Calculate combustion efficiency based on characteristic length L*.
    
    Can use either:
    1. Simple model: η_c* = 1 - C × e^(-K×L*) (default, backward compatible)
    2. Advanced model: Physics-based with kinetics, mixing, turbulence (if use_advanced_model=True)
    
    This corrects CEA's infinite-area equilibrium assumption for finite chambers.
    
    NOTE: CEA uses EQUILIBRIUM flow (not frozen). The correction accounts for:
    - Finite residence time (L*)
    - Incomplete mixing
    - Finite-rate chemistry effects
    - Heat losses
    
    Parameters:
    -----------
    Lstar : float
        Characteristic length [m]
    config : CombustionEfficiencyConfig
        Efficiency configuration
    spray_quality_good : bool
        Whether spray constraints are satisfied (affects efficiency if enabled)
    mixture_efficiency : float
        Mixture efficiency factor (from spray diagnostics)
    cooling_efficiency : float
        Cooling efficiency factor (from heat transfer)
    use_advanced_model : bool
        If True, use advanced physics-based model
    advanced_params : dict, optional
        Parameters for advanced model:
        - Pc: Chamber pressure [Pa]
        - Tc: Chamber temperature [K]
        - cstar_ideal: Ideal c* [m/s]
        - gamma: Specific heat ratio
        - R: Gas constant [J/(kg·K)]
        - MR: Mixture ratio
        - spray_diagnostics: Spray diagnostics dict
        - turbulence_intensity: Turbulence intensity (0-1)
    
    Returns:
    --------
    eta : float
        Combustion efficiency (0-1)
    """
    # Use advanced model if requested and parameters provided
    if use_advanced_model and advanced_params is not None:
        try:
            from .combustion_physics import calculate_combustion_efficiency_advanced
            
            # Extract parameters
            Pc = advanced_params.get("Pc", DEFAULT_CHAMBER_PRESS_PA)
            Tc = advanced_params.get("Tc", DEFAULT_CHAMBER_TEMP_K)
            cstar_ideal = advanced_params.get("cstar_ideal", DEFAULT_CSTAR_IDEAL_M_S)
            gamma = advanced_params.get("gamma", DEFAULT_GAMMA_ND)
            R = advanced_params.get("R", DEFAULT_GAS_CONST_J_KG_K)
            MR = advanced_params.get("MR", DEFAULT_MIXTURE_RATIO_ND)
            Ac = advanced_params.get("Ac", None)
            m_dot_total = advanced_params.get("m_dot_total", None)
            Dinj = advanced_params.get("Dinj", None)
            spray_diagnostics = advanced_params.get("spray_diagnostics", None)
            turbulence_intensity = advanced_params.get("turbulence_intensity", DEFAULT_TURBULENCE_INTENSITY_ND)
            
            # Validate required parameters
            if Ac is None or m_dot_total is None:
                raise ValueError("Ac and m_dot_total are required for advanced combustion efficiency model")
            if Dinj is None:
                Dinj = float(np.sqrt(4.0 * Ac / np.pi))
            Dinj = float(max(Dinj, 1e-6))
            
            # Calculate advanced efficiency
            results = calculate_combustion_efficiency_advanced(
                Lstar, Pc, Tc, cstar_ideal, gamma, R, MR, config,
                Ac, Dinj, m_dot_total,
                spray_diagnostics, turbulence_intensity
            )
            
            eta = results["eta_total"]
            
            # Apply cooling efficiency (external factor)
            cooling_eff = float(np.clip(cooling_efficiency, config.cooling_efficiency_floor, 1.0))
            eta *= cooling_eff
            
            # Final clamp
            lower_bound = min(config.mixture_efficiency_floor, config.cooling_efficiency_floor)
            eta = np.clip(eta, lower_bound, 1.0)
            
            return float(eta)
            
        except ImportError:
            # Fall back to simple model if advanced module not available
            print("the advanced module is not available, falling back to simple model")
            pass
        except Exception as exc:
            import traceback
            import warnings
            traceback.print_exc()
            warnings.warn(f"Advanced eta_cstar failed: {exc}")
            pass
    # Simple model (backward compatible)
    if config.model == "constant":
        eta = 1.0 - config.C
    elif config.model == "linear":
        eta = 1.0 - config.C * (1.0 - Lstar / 1.0)  # Normalized to 1 m
        eta = np.clip(eta, 0.0, 1.0)
    else:  # exponential (default)
        # Less conservative: reduce C factor by 50% for more realistic efficiency
        # Typical rocket engines achieve 85-95% efficiency, not 55-77%
        C_adjusted = config.C * 0.5  # Reduce penalty by 50%
        eta = 1.0 - C_adjusted * np.exp(-config.K * Lstar)
        # Ensure minimum efficiency of 85% for well-designed engines
        eta = max(eta, 0.85)
    
    # Apply spray quality correction if enabled
    if config.use_spray_correction and not spray_quality_good:
        eta *= config.spray_penalty_factor

    mixture_efficiency = float(np.clip(mixture_efficiency, config.mixture_efficiency_floor, 1.0))
    cooling_efficiency = float(np.clip(cooling_efficiency, config.cooling_efficiency_floor, 1.0))
    eta *= mixture_efficiency * cooling_efficiency
    
    # Clamp to reasonable range - ensure minimum 85% for well-designed engines
    # This prevents efficiency from dropping too low due to conservative models
    lower_bound = max(min(config.mixture_efficiency_floor, config.cooling_efficiency_floor), 0.85)
    eta = np.clip(eta, lower_bound, 1.0)
    
    return float(eta)


def calculate_actual_chamber_temp(
    Tc_ideal: float,
    eta: float,
    gamma: float
) -> float:
    """
    Calculate actual chamber temperature accounting for combustion efficiency.
    
    T_c,actual = T_c,ideal × [η / (1 - (1-η) × (γ-1)/γ)]
    
    Parameters:
    -----------
    Tc_ideal : float
        Ideal chamber temperature from CEA [K]
    eta : float
        Combustion efficiency
    gamma : float
        Specific heat ratio
    
    Returns:
    --------
    Tc_actual : float [K]
    """
    if gamma <= 1:
        return Tc_ideal
    
    denominator = 1.0 - (1.0 - eta) * (gamma - 1.0) / gamma
    if denominator <= 0:
        return Tc_ideal
    
    Tc_actual = Tc_ideal * (eta / denominator)
    return float(Tc_actual)


def calculate_frozen_flow_correction(
    Lstar: float,
    gamma_ideal: float,
    alpha: float = 0.1
) -> float:
    """
    Calculate frozen flow correction factor for gamma.
    
    γ_actual = γ_ideal × [1 - α × (1 - η_c*)]
    
    This accounts for incomplete chemical reactions in the nozzle.
    
    Parameters:
    -----------
    Lstar : float
        Characteristic length [m]
    gamma_ideal : float
        Ideal gamma from CEA
    alpha : float
        Frozen flow parameter (default 0.1)
    
    Returns:
    --------
    correction_factor : float
        Factor to multiply gamma_ideal by
    """
    # Estimate efficiency from L* (simplified)
    # Using default C=0.3, K=0.15
    eta_est = 1.0 - 0.3 * np.exp(-0.15 * Lstar)
    
    correction = 1.0 - alpha * (1.0 - eta_est)
    correction = np.clip(correction, 0.9, 1.0)  # Reasonable bounds
    
    return float(correction)
