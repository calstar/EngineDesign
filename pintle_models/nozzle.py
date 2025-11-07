"""Nozzle model: thrust coefficient and thrust calculation"""

import numpy as np
from pintle_pipeline.config_schemas import NozzleConfig
from pintle_pipeline.cea_cache import CEACache


def calculate_thrust(
    Pc: float,
    MR: float,
    mdot_total: float,
    cea_cache: CEACache,
    nozzle_config: NozzleConfig,
    Pa: float = 101325.0
) -> dict:
    """
    Calculate engine thrust with high fidelity.
    
    Thrust consists of two components:
    1. Momentum thrust: ṁ × v_exit
    2. Pressure thrust: (P_exit - Pa) × A_exit
    
    F = ṁ × v_exit + (P_exit - Pa) × A_exit
    
    Or using thrust coefficient:
    F = Cf × Pc × At
    
    where Cf accounts for both momentum and pressure components.
    
    Parameters:
    -----------
    Pc : float
        Chamber pressure [Pa]
    MR : float
        Mixture ratio (O/F)
    mdot_total : float
        Total mass flow rate [kg/s]
    cea_cache : CEACache
        CEA cache for thermochemical properties
    nozzle_config : NozzleConfig
        Nozzle configuration
    Pa : float
        Ambient pressure [Pa] (default: sea level)
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - F: Total thrust [N]
        - F_momentum: Momentum thrust component [N]
        - F_pressure: Pressure thrust component [N]
        - Cf: Thrust coefficient
        - Cf_ideal: Ideal thrust coefficient from CEA
        - P_exit: Exit pressure [Pa]
        - v_exit: Exit velocity [m/s]
        - Isp: Specific impulse [s]
    """
    # Get CEA properties
    cea_props = cea_cache.eval(MR, Pc, Pa, nozzle_config.expansion_ratio)
    Cf_ideal = cea_props["Cf_ideal"]
    gamma = cea_props["gamma"]
    Tc = cea_props["Tc"]
    R = cea_props["R"]
    
    # Apply nozzle efficiency
    Cf = nozzle_config.efficiency * Cf_ideal
    
    # Calculate exit pressure using isentropic relations
    eps = nozzle_config.expansion_ratio
    
    # For supersonic nozzles (eps > 1), we need to solve the area-Mach relation:
    # A/A* = (1/M) × [(2/(gamma+1)) × (1 + (gamma-1)/2 × M²)]^((gamma+1)/(2(gamma-1)))
    # Then use isentropic relation: P/Pc = [1 + (gamma-1)/2 × M²]^(-gamma/(gamma-1))
    
    if gamma > 1 and eps > 1:
        # Solve for exit Mach number from area ratio (supersonic solution)
        # Area-Mach relation: A/A* = (1/M) × [(2/(gamma+1)) × (1 + (gamma-1)/2 × M²)]^((gamma+1)/(2(gamma-1)))
        # For supersonic flow (M > 1), we need to solve this iteratively
        
        # Initial guess: For large eps, M_exit is large
        # Approximation: For M >> 1, A/A* ≈ (gamma+1)^((gamma+1)/(2(gamma-1))) / (2^(1/(gamma-1)) × M)
        # Better: Use iterative solution starting from M = 2.0
        M_exit = 2.0  # Start with reasonable supersonic guess
        
        # Newton-Raphson iteration
        for iteration in range(20):  # Usually converges in 3-5 iterations
            # Calculate A/A* for current M
            term = (2.0 / (gamma + 1.0)) * (1.0 + (gamma - 1.0) / 2.0 * M_exit**2)
            A_Astar = (1.0 / M_exit) * (term ** ((gamma + 1.0) / (2.0 * (gamma - 1.0))))
            
            # Error
            error = A_Astar - eps
            
            if abs(error) < 1e-8:
                break
            
            # Derivative d(A/A*)/dM
            # d/dM[(1/M) × term^((gamma+1)/(2(gamma-1)))]
            # = -A_Astar/M + A_Astar × (gamma+1)/(2(gamma-1)) × (1/term) × d(term)/dM
            dterm_dM = (gamma - 1.0) * M_exit
            dA_dM = -A_Astar / M_exit + A_Astar * ((gamma + 1.0) / (2.0 * (gamma - 1.0))) * (dterm_dM / term)
            
            if abs(dA_dM) > 1e-10:
                M_exit = M_exit - error / dA_dM
            else:
                break
            
            # Ensure supersonic and reasonable
            M_exit = max(M_exit, 1.01)
            M_exit = min(M_exit, 10.0)  # Cap at M=10 (unrealistic but prevents divergence)
        
        # Use isentropic relation for exit pressure
        # P_exit/Pc = [1 + (gamma-1)/2 × M_exit²]^(-gamma/(gamma-1))
        P_exit = Pc * (1.0 + (gamma - 1.0) / 2.0 * M_exit**2) ** (-gamma / (gamma - 1.0))
        P_exit = max(P_exit, Pa)  # Can't be less than ambient
    else:
        P_exit = Pa
    
    # Calculate exit temperature from isentropic relation
    if P_exit < Pc and gamma > 1:
        T_exit = Tc * ((P_exit / Pc) ** ((gamma - 1) / gamma))
    else:
        T_exit = Tc  # No expansion
    
    # Calculate exit velocity from energy equation
    cp = gamma * R / (gamma - 1)  # Specific heat at constant pressure
    if Tc > T_exit:
        v_exit = np.sqrt(2 * cp * (Tc - T_exit))
    else:
        v_exit = 0.0
    
    # Calculate thrust components
    F_momentum = mdot_total * v_exit
    F_pressure = (P_exit - Pa) * nozzle_config.A_exit
    F_total = F_momentum + F_pressure
    
    # Also calculate using thrust coefficient method for validation
    F_cf = Cf * Pc * nozzle_config.A_throat
    
    # Use the more accurate method (momentum + pressure)
    F = F_total
    
    # Calculate Isp
    g0 = 9.80665  # m/s²
    Isp = F / (mdot_total * g0) if mdot_total > 0 else 0
    
    return {
        "F": float(F),
        "F_momentum": float(F_momentum),
        "F_pressure": float(F_pressure),
        "F_cf_method": float(F_cf),  # For comparison
        "Cf": float(Cf),
        "Cf_ideal": float(Cf_ideal),
        "P_exit": float(P_exit),
        "v_exit": float(v_exit),
        "T_exit": float(T_exit),
        "Isp": float(Isp),
    }
