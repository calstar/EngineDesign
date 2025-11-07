"""Combustion efficiency models (L* correction)"""

import numpy as np
from typing import Optional
from .config_schemas import CombustionEfficiencyConfig


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
    spray_quality_good: bool = True
) -> float:
    """
    Calculate combustion efficiency based on characteristic length L*.
    
    Model: η_c* = 1 - C × e^(-K×L*)
    
    This corrects CEA's infinite-area equilibrium assumption for finite chambers.
    
    Parameters:
    -----------
    Lstar : float
        Characteristic length [m]
    config : CombustionEfficiencyConfig
        Efficiency configuration
    spray_quality_good : bool
        Whether spray constraints are satisfied (affects efficiency if enabled)
    
    Returns:
    --------
    eta : float
        Combustion efficiency (0-1)
    """
    if config.model == "constant":
        eta = 1.0 - config.C
    elif config.model == "linear":
        eta = 1.0 - config.C * (1.0 - Lstar / 1.0)  # Normalized to 1 m
        eta = np.clip(eta, 0.0, 1.0)
    else:  # exponential (default)
        eta = 1.0 - config.C * np.exp(-config.K * Lstar)
    
    # Apply spray quality correction if enabled
    if config.use_spray_correction and not spray_quality_good:
        eta *= config.spray_penalty_factor
    
    # Clamp to reasonable range
    eta = np.clip(eta, 0.5, 1.0)  # Minimum 50% efficiency
    
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
