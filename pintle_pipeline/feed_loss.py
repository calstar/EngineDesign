"""Generalized feed system pressure loss model with K_eff(P)"""

import numpy as np
from .config_schemas import FeedSystemConfig


def delta_p_feed(
    mdot: float,
    rho: float,
    config: FeedSystemConfig,
    P_tank: float
) -> float:
    """
    Calculate feed system pressure loss using generalized K_eff(P) model.
    
    Δp_feed = K_eff(P) × (ρ/2) × (ṁ/(ρ×A_hyd))²
    
    where K_eff(P) = K0 + K1 × φ(P)
    
    Parameters:
    -----------
    mdot : float
        Mass flow rate [kg/s]
    rho : float
        Fluid density [kg/m³]
    config : FeedSystemConfig
        Feed system configuration
    P_tank : float
        Tank pressure [Pa] (used for pressure-dependent K_eff)
    
    Returns:
    --------
    delta_p : float
        Pressure loss [Pa]
    """
    # Calculate effective loss coefficient
    if config.phi_type == "none":
        K_eff = config.K0
    elif config.phi_type == "sqrtP":
        K_eff = config.K0 + config.K1 * np.sqrt(P_tank)
    elif config.phi_type == "logP":
        K_eff = config.K0 + config.K1 * np.log(P_tank)
    else:
        raise ValueError(f"Unknown phi_type: {config.phi_type}")
    
    # Calculate area from inlet diameter if A_hydraulic not explicitly set
    # A_hydraulic should be calculated from d_inlet if not provided
    if hasattr(config, 'd_inlet') and config.d_inlet > 0:
        A_area = np.pi * (config.d_inlet / 2) ** 2
    else:
        A_area = config.A_hydraulic
    
    # Calculate velocity
    velocity = mdot / (rho * A_area)
    
    # Calculate pressure loss
    delta_p = K_eff * (rho / 2) * velocity**2
    
    return float(delta_p)



