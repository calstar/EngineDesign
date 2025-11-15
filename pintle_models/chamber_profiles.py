"""Chamber pressure and temperature profiles along length"""

import numpy as np
from typing import Dict, Optional, Any, Tuple
from pintle_pipeline.numerical_robustness import NumericalStability, PhysicalConstraints


def calculate_chamber_pressure_profile(
    Pc: float,
    Lstar: float,
    mdot_total: float,
    gamma: float,
    R: float,
    Tc: float,
    A_throat: float,
    n_points: int = 20,
) -> Dict[str, Any]:
    """
    Calculate pressure profile along chamber length.
    
    Pressure drops from injection to throat due to:
    1. Momentum addition from combustion
    2. Friction losses (minor)
    3. Area contraction to throat
    
    For quasi-1D flow with heat addition:
        dP/dx = -rho * u * du/dx - (gamma - 1) * rho * dq/dx
    where dq/dx is heat release rate from reaction.
    
    Simplified model: P decreases linearly with heat addition.
    
    Parameters:
    -----------
    Pc : float
        Chamber pressure at throat [Pa] (reference pressure)
    Lstar : float
        Characteristic length [m]
    mdot_total : float
        Total mass flow [kg/s]
    gamma : float
        Specific heat ratio
    R : float
        Gas constant [J/(kg·K)]
    Tc : float
        Chamber temperature [K]
    A_throat : float
        Throat area [m²]
    n_points : int
        Number of points along chamber (default: 20)
    
    Returns:
    --------
    profile : dict
        - positions: Array of positions along chamber [m] (0 = injection, Lstar = throat)
        - pressures: Array of pressures [Pa]
        - P_injection: Pressure at injection plane [Pa]
        - P_mid: Pressure at mid-chamber [Pa]
        - P_throat: Pressure at throat [Pa] (= Pc)
    """
    # Create position array
    positions = np.linspace(0.0, Lstar, n_points)
    
    # Chamber gas properties
    rho_chamber = Pc / (R * Tc)
    
    # Estimate cross-sectional area (assume cylindrical chamber)
    # Approximate: A_chamber = V / L* ≈ A_throat * expansion_ratio
    # Use average area for velocity calculation
    A_chamber_avg = A_throat * 3.0  # Rough estimate (typical expansion from throat)
    u_chamber = mdot_total / (rho_chamber * A_chamber_avg) if rho_chamber > 0 else 0.0
    
    # Pressure profile: drops from injection to throat
    # Simplified: P(x) = P_injection * (1 - x/L*)^alpha
    # where alpha accounts for momentum addition and friction
    
    # At injection: higher pressure due to momentum of incoming propellants
    # Pressure ratio: P_injection / P_throat ≈ 1.05-1.15 for typical engines
    P_injection_ratio = 1.10  # 10% higher at injection
    
    # Alpha factor: how pressure drops (typically 0.1-0.3)
    # Lower alpha = more gradual drop (more momentum addition)
    alpha = 0.15  # Typical value
    
    # Normalized positions (0 = injection, 1 = throat)
    x_norm = positions / Lstar if Lstar > 0 else np.zeros_like(positions)
    
    # Pressure profile
    # P(x) = P_injection * (1 - x_norm)^alpha
    # At throat (x_norm=1): P = Pc
    # So: Pc = P_injection * (1 - 1)^alpha = P_injection * 0 = 0 (problem!)
    # Better: P(x) = P_injection - (P_injection - Pc) * x_norm^alpha
    P_injection = Pc * P_injection_ratio
    pressures = P_injection - (P_injection - Pc) * (x_norm ** alpha)
    
    # Ensure monotonic (pressure decreases toward throat)
    for i in range(1, len(pressures)):
        if pressures[i] > pressures[i-1]:
            pressures[i] = pressures[i-1] * 0.999
    
    # Validate all pressures
    pressures = np.clip(pressures, Pc * 0.8, Pc * 1.2)  # Physical bounds
    
    # Key points
    P_mid = float(pressures[len(pressures) // 2])
    P_injection_val = float(pressures[0])
    P_throat_val = float(pressures[-1])  # Should equal Pc
    
    return {
        "positions": positions.tolist(),
        "pressures": pressures.tolist(),
        "P_injection": P_injection_val,
        "P_mid": P_mid,
        "P_throat": P_throat_val,
    }


def calculate_chamber_intrinsics(
    Pc: float,
    Tc: float,
    mdot_total: float,
    gamma: float,
    R: float,
    V_chamber: float,
    A_throat: float,
    Lstar: float,
    MR: float,
) -> Dict[str, Any]:
    """
    Calculate chamber intrinsic properties.
    
    Parameters:
    -----------
    Pc : float
        Chamber pressure [Pa]
    Tc : float
        Chamber temperature [K]
    mdot_total : float
        Total mass flow [kg/s]
    gamma : float
        Specific heat ratio
    R : float
        Gas constant [J/(kg·K)]
    V_chamber : float
        Chamber volume [m³]
    A_throat : float
        Throat area [m²]
    Lstar : float
        Characteristic length [m]
    MR : float
        Mixture ratio
    
    Returns:
    --------
    intrinsics : dict
        - Lstar: Characteristic length [m]
        - residence_time: Gas residence time [s]
        - velocity_mean: Mean gas velocity [m/s]
        - velocity_throat: Velocity at throat (sonic) [m/s]
        - density: Chamber gas density [kg/m³]
        - sound_speed: Sound speed in chamber [m/s]
        - mach_number: Mean Mach number
        - reynolds_number: Reynolds number
    """
    # Gas density
    rho, rho_valid = NumericalStability.safe_divide(Pc, R * Tc, 1.0, "rho")
    if not rho_valid.passed:
        rho = 1.0  # Fallback
    
    # Sound speed
    sound_speed_squared = gamma * R * Tc
    sound_speed, sound_valid = NumericalStability.safe_sqrt(sound_speed_squared, "sound_speed")
    if not sound_valid.passed:
        sound_speed = 1000.0  # Fallback
    
    # Mean velocity (from mass flow and density)
    # Assume average cross-sectional area
    # A_avg ≈ V_chamber / Lstar
    A_avg, A_valid = NumericalStability.safe_divide(V_chamber, Lstar, A_throat, "A_avg")
    if not A_valid.passed:
        A_avg = A_throat * 3.0  # Fallback
    
    velocity_mean, v_valid = NumericalStability.safe_divide(
        mdot_total, rho * A_avg, 0.0, "velocity_mean"
    )
    if not v_valid.passed:
        velocity_mean = 50.0  # Fallback [m/s]
    
    # Throat velocity (sonic)
    # v_throat = sqrt(gamma * R * Tc / (gamma + 1))
    throat_velocity_factor, factor_valid = NumericalStability.safe_divide(
        gamma, gamma + 1.0, 0.5, "gamma/(gamma+1)"
    )
    if factor_valid.passed:
        throat_velocity_squared = gamma * R * Tc * throat_velocity_factor
        velocity_throat, throat_v_valid = NumericalStability.safe_sqrt(
            throat_velocity_squared, "velocity_throat"
        )
        if not throat_v_valid.passed:
            velocity_throat = sound_speed * 0.9  # Fallback (sonic)
    else:
        velocity_throat = sound_speed * 0.9  # Fallback
    
    # Mach number
    mach_number, mach_valid = NumericalStability.safe_divide(
        velocity_mean, sound_speed, 0.01, "mach_number"
    )
    if not mach_valid.passed:
        mach_number = 0.01  # Fallback
    
    # Residence time
    # tau = V_chamber / (mdot_total / rho) = V_chamber * rho / mdot_total
    residence_time, tau_valid = NumericalStability.safe_divide(
        V_chamber * rho, mdot_total, 0.001, "residence_time"
    )
    if not tau_valid.passed:
        residence_time = 0.001  # Fallback [s]
    
    # Reynolds number (use mean properties)
    # Re = rho * u * L / mu
    # Estimate viscosity from temperature (Sutherland's law approximation)
    mu_ref = 1.8e-5  # Pa·s at 273 K (air, approximate for combustion gases)
    T_ref = 273.0  # K
    S = 110.4  # Sutherland constant [K] (for air, approximate)
    viscosity = mu_ref * ((Tc / T_ref) ** 1.5) * ((T_ref + S) / (Tc + S))
    
    # Characteristic length for Re (use Lstar)
    reynolds_number, Re_valid = NumericalStability.safe_divide(
        rho * velocity_mean * Lstar, viscosity, 1000.0, "reynolds_number"
    )
    if not Re_valid.passed:
        reynolds_number = 10000.0  # Fallback
    
    # Validate all values
    residence_time = max(residence_time, 1e-6)  # Minimum 1 µs
    mach_number = max(min(mach_number, 0.3), 0.001)  # Subsonic
    reynolds_number = max(reynolds_number, 100.0)  # Minimum Re
    
    return {
        "Lstar": float(Lstar),
        "residence_time": float(residence_time),
        "velocity_mean": float(velocity_mean),
        "velocity_throat": float(velocity_throat),
        "density": float(rho),
        "sound_speed": float(sound_speed),
        "mach_number": float(mach_number),
        "reynolds_number": float(reynolds_number),
        "viscosity": float(viscosity),
        "volume": float(V_chamber),
        "A_throat": float(A_throat),
        "A_avg": float(A_avg),
    }

