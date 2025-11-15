"""Advanced combustion physics models for realistic performance prediction.

This module provides physics-based corrections to CEA equilibrium results
to account for:
1. Finite residence time effects (L*)
2. Mixing quality (spray, turbulence)
3. Reaction kinetics (pressure, temperature dependent)
4. Finite-rate chemistry
5. Heat loss effects
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
from pintle_pipeline.config_schemas import CombustionEfficiencyConfig


def calculate_residence_time(
    Lstar: float,
    Pc: float,
    cstar: float,
    gamma: float,
    R: float,
    Tc: float,
) -> float:
    """
    Calculate characteristic residence time in chamber.
    
    τ_res = L* / v_characteristic
    
    where v_characteristic is based on chamber conditions.
    
    Parameters:
    -----------
    Lstar : float
        Characteristic length [m]
    Pc : float
        Chamber pressure [Pa]
    cstar : float
        Characteristic velocity [m/s]
    gamma : float
        Specific heat ratio
    R : float
        Gas constant [J/(kg·K)]
    Tc : float
        Chamber temperature [K]
    
    Returns:
    --------
    tau_res : float
        Residence time [s]
    """
    # Characteristic velocity in chamber (subsonic)
    # Use sound speed as characteristic velocity
    sound_speed = np.sqrt(gamma * R * Tc)
    
    # Residence time
    tau_res = Lstar / sound_speed
    
    return float(tau_res)


def calculate_reaction_time_scale(
    Pc: float,
    Tc: float,
    MR: float,
    gamma: float,
) -> float:
    """
    Estimate chemical reaction time scale.
    
    Uses Arrhenius-like scaling with pressure and temperature.
    Higher pressure → faster reactions (collision frequency)
    Higher temperature → faster reactions (activation energy)
    
    τ_chem ≈ A × P^(-n) × exp(Ea / (R_gas × T))
    
    Simplified model:
    τ_chem ≈ τ_ref × (P_ref / P)^n × exp(Ea_norm × (T_ref / T - 1))
    
    Parameters:
    -----------
    Pc : float
        Chamber pressure [Pa]
    Tc : float
        Chamber temperature [K]
    MR : float
        Mixture ratio
    gamma : float
        Specific heat ratio
    
    Returns:
    --------
    tau_chem : float
        Chemical reaction time scale [s]
    """
    # Reference conditions (typical rocket chamber)
    P_ref = 4.0e6  # 4 MPa
    T_ref = 3500.0  # 3500 K
    
    # Pressure exponent (typically 0.5-1.0 for gas-phase reactions)
    n_pressure = 0.7
    
    # Normalized activation energy (dimensionless)
    # Higher for more complex reactions (e.g., hydrocarbon combustion)
    # Lower for simpler reactions (e.g., H2/O2)
    # Typical range: 5-15
    if MR < 1.5:  # Fuel-rich (more complex chemistry)
        Ea_norm = 12.0
    elif MR > 3.0:  # Oxidizer-rich (simpler chemistry)
        Ea_norm = 8.0
    else:  # Near-stoichiometric
        Ea_norm = 10.0
    
    # Reference reaction time (typical: 1-10 ms)
    tau_ref = 0.005  # 5 ms
    
    # Pressure effect (higher pressure → faster reactions)
    pressure_factor = (P_ref / max(Pc, 1e5)) ** n_pressure
    
    # Temperature effect (higher temperature → faster reactions)
    temp_factor = np.exp(Ea_norm * (T_ref / max(Tc, 1000.0) - 1.0))
    
    tau_chem = tau_ref * pressure_factor * temp_factor
    
    # Clamp to reasonable range (0.1 ms to 100 ms)
    tau_chem = np.clip(tau_chem, 0.0001, 0.1)
    
    return float(tau_chem)


def calculate_damkohler_number(
    tau_res: float,
    tau_chem: float,
) -> float:
    """
    Calculate Damköhler number (ratio of residence time to reaction time).
    
    Da = τ_res / τ_chem
    
    Da >> 1: Fast chemistry (equilibrium achieved)
    Da ~ 1: Finite-rate chemistry (partial equilibrium)
    Da << 1: Slow chemistry (far from equilibrium)
    
    Parameters:
    -----------
    tau_res : float
        Residence time [s]
    tau_chem : float
        Chemical reaction time scale [s]
    
    Returns:
    --------
    Da : float
        Damköhler number
    """
    if tau_chem <= 0:
        return np.inf  # Instantaneous reactions
    
    Da = tau_res / tau_chem
    return float(Da)


def calculate_mixing_efficiency(
    SMD: float,
    evaporation_length: float,
    chamber_length: float,
    turbulence_intensity: float,
    target_smd: float = 50e-6,  # 50 microns
) -> float:
    """
    Calculate mixing efficiency based on spray quality and evaporation.
    
    Poor mixing (large droplets, long evaporation) → lower efficiency.
    Good mixing (small droplets, short evaporation) → higher efficiency.
    
    Parameters:
    -----------
    SMD : float
        Sauter Mean Diameter [m]
    evaporation_length : float
        Evaporation length [m]
    chamber_length : float
        Chamber length [m]
    turbulence_intensity : float
        Turbulence intensity (0-1)
    target_smd : float
        Target SMD for good atomization [m]
    
    Returns:
    --------
    eta_mix : float
        Mixing efficiency (0-1)
    """
    # Droplet size effect
    # Smaller droplets → better mixing
    if SMD > 0 and target_smd > 0:
        smd_ratio = min(SMD / target_smd, 10.0)  # Cap at 10x
        smd_factor = 1.0 / (1.0 + 0.5 * (smd_ratio - 1.0))
    else:
        smd_factor = 0.5  # Unknown → assume poor
    
    # Evaporation length effect
    # If evaporation length > chamber length, incomplete evaporation
    if chamber_length > 0:
        evap_ratio = min(evaporation_length / chamber_length, 2.0)  # Cap at 2x
        evap_factor = 1.0 / (1.0 + 0.3 * (evap_ratio - 1.0))
    else:
        evap_factor = 0.5
    
    # Turbulence effect (enhances mixing)
    turbulence_factor = 0.7 + 0.3 * min(turbulence_intensity / 0.2, 1.0)
    
    # Combined mixing efficiency
    eta_mix = smd_factor * evap_factor * turbulence_factor
    
    # Clamp to reasonable range
    eta_mix = np.clip(eta_mix, 0.3, 1.0)
    
    return float(eta_mix)


def calculate_combustion_efficiency_advanced(
    Lstar: float,
    Pc: float,
    Tc: float,
    cstar_ideal: float,
    gamma: float,
    R: float,
    MR: float,
    config: CombustionEfficiencyConfig,
    spray_diagnostics: Optional[Dict] = None,
    turbulence_intensity: float = 0.08,
) -> Dict[str, float]:
    """
    Advanced combustion efficiency calculation with physics-based corrections.
    
    Accounts for:
    1. Finite residence time (L*)
    2. Chemical reaction kinetics (pressure, temperature dependent)
    3. Mixing quality (spray, evaporation)
    4. Turbulence effects
    
    Model:
    η_c* = η_L* × η_kinetics × η_mixing × η_turbulence × η_cooling
    
    where:
    - η_L*: L*-based efficiency (finite residence time)
    - η_kinetics: Reaction kinetics efficiency (Damköhler number)
    - η_mixing: Mixing efficiency (spray quality)
    - η_turbulence: Turbulence enhancement
    - η_cooling: Cooling losses (if provided)
    
    Parameters:
    -----------
    Lstar : float
        Characteristic length [m]
    Pc : float
        Chamber pressure [Pa]
    Tc : float
        Chamber temperature [K]
    cstar_ideal : float
        Ideal c* from CEA [m/s]
    gamma : float
        Specific heat ratio
    R : float
        Gas constant [J/(kg·K)]
    MR : float
        Mixture ratio
    config : CombustionEfficiencyConfig
        Efficiency configuration
    spray_diagnostics : dict, optional
        Spray diagnostics (SMD, evaporation length, etc.)
    turbulence_intensity : float
        Turbulence intensity (0-1)
    
    Returns:
    --------
    results : dict
        - eta_total: Total combustion efficiency
        - eta_Lstar: L*-based efficiency
        - eta_kinetics: Kinetics efficiency
        - eta_mixing: Mixing efficiency
        - eta_turbulence: Turbulence efficiency
        - Da: Damköhler number
        - tau_res: Residence time [s]
        - tau_chem: Chemical reaction time [s]
    """
    # 1. L*-based efficiency (finite residence time)
    if config.model == "constant":
        eta_Lstar = 1.0 - config.C
    elif config.model == "linear":
        eta_Lstar = 1.0 - config.C * (1.0 - Lstar / 1.0)
        eta_Lstar = np.clip(eta_Lstar, 0.0, 1.0)
    else:  # exponential (default)
        eta_Lstar = 1.0 - config.C * np.exp(-config.K * Lstar)
    
    # Clamp L* efficiency
    eta_Lstar = np.clip(eta_Lstar, config.mixture_efficiency_floor, 1.0)
    
    # 2. Reaction kinetics efficiency (Damköhler number)
    tau_res = calculate_residence_time(Lstar, Pc, cstar_ideal, gamma, R, Tc)
    tau_chem = calculate_reaction_time_scale(Pc, Tc, MR, gamma)
    Da = calculate_damkohler_number(tau_res, tau_chem)
    
    # Efficiency based on Damköhler number
    # Da >> 1: equilibrium (eta → 1)
    # Da ~ 1: finite-rate (eta ~ 0.8-0.95)
    # Da << 1: slow chemistry (eta ~ 0.5-0.8)
    if Da > 10.0:
        eta_kinetics = 1.0  # Fast chemistry, equilibrium achieved
    elif Da > 1.0:
        eta_kinetics = 0.95 + 0.05 * (1.0 - np.exp(-(Da - 1.0) / 2.0))
    elif Da > 0.1:
        eta_kinetics = 0.8 + 0.15 * (Da / 1.0)
    else:
        eta_kinetics = 0.5 + 0.3 * (Da / 0.1)
    
    eta_kinetics = np.clip(eta_kinetics, 0.5, 1.0)
    
    # 3. Mixing efficiency
    if spray_diagnostics is not None:
        SMD = spray_diagnostics.get("D32_O", 0.0) or spray_diagnostics.get("D32_F", 0.0) or 100e-6
        x_star = spray_diagnostics.get("x_star", 0.0) or 0.1
        chamber_length = Lstar  # Approximate
        eta_mixing = calculate_mixing_efficiency(
            SMD, x_star, chamber_length, turbulence_intensity,
            target_smd=config.target_smd_microns * 1e-6 if hasattr(config, 'target_smd_microns') else 50e-6
        )
    else:
        eta_mixing = 1.0  # Assume perfect mixing if no diagnostics
    
    # Apply spray quality penalty if enabled
    if config.use_spray_correction:
        spray_quality_good = spray_diagnostics.get("constraints_satisfied", True) if spray_diagnostics else True
        if not spray_quality_good:
            eta_mixing *= config.spray_penalty_factor
    
    eta_mixing = np.clip(eta_mixing, config.mixture_efficiency_floor, 1.0)
    
    # 4. Turbulence efficiency (enhancement)
    # Moderate turbulence enhances mixing, excessive turbulence can reduce efficiency
    if turbulence_intensity < 0.05:
        eta_turbulence = 0.9  # Low turbulence → poor mixing
    elif turbulence_intensity < 0.15:
        eta_turbulence = 0.95 + 0.05 * (turbulence_intensity / 0.15)  # Optimal range
    else:
        eta_turbulence = 1.0 - 0.1 * ((turbulence_intensity - 0.15) / 0.35)  # Excessive turbulence
    
    eta_turbulence = np.clip(eta_turbulence, 0.85, 1.0)
    
    # 5. Combined efficiency
    eta_total = eta_Lstar * eta_kinetics * eta_mixing * eta_turbulence
    
    # Apply cooling efficiency if provided (external)
    # This would be multiplied in by the caller
    
    # Final clamp
    lower_bound = min(config.mixture_efficiency_floor, config.cooling_efficiency_floor)
    eta_total = np.clip(eta_total, lower_bound, 1.0)
    
    return {
        "eta_total": float(eta_total),
        "eta_Lstar": float(eta_Lstar),
        "eta_kinetics": float(eta_kinetics),
        "eta_mixing": float(eta_mixing),
        "eta_turbulence": float(eta_turbulence),
        "Da": float(Da),
        "tau_res": float(tau_res),
        "tau_chem": float(tau_chem),
    }


def calculate_equilibrium_shift(
    Pc: float,
    Tc: float,
    MR: float,
    Lstar: float,
) -> Dict[str, float]:
    """
    Calculate how far from equilibrium the combustion is.
    
    Returns metrics for:
    - Equilibrium completeness (0-1)
    - Reaction progress
    - Composition shift
    
    Parameters:
    -----------
    Pc : float
        Chamber pressure [Pa]
    Tc : float
        Chamber temperature [K]
    MR : float
        Mixture ratio
    Lstar : float
        Characteristic length [m]
    
    Returns:
    --------
    results : dict
        - equilibrium_completeness: How close to equilibrium (0-1)
        - reaction_progress: Progress of main reactions (0-1)
        - composition_shift: Shift from ideal composition
    """
    # Estimate equilibrium completeness based on residence time
    # Higher pressure and temperature → faster approach to equilibrium
    # Longer L* → more time to reach equilibrium
    
    # Normalized pressure (relative to typical 4 MPa)
    P_norm = Pc / 4.0e6
    
    # Normalized temperature (relative to typical 3500 K)
    T_norm = Tc / 3500.0
    
    # Normalized L*
    Lstar_norm = Lstar / 1.0  # Relative to 1 m
    
    # Equilibrium completeness factor
    # Higher P, T, L* → closer to equilibrium
    completeness = 1.0 - np.exp(-0.5 * P_norm * T_norm * Lstar_norm)
    completeness = np.clip(completeness, 0.0, 1.0)
    
    # Reaction progress (simplified)
    # Assumes main reactions are 80% complete at typical conditions
    progress = 0.8 * completeness
    
    # Composition shift (how much composition differs from equilibrium)
    # Lower completeness → larger shift
    shift = 1.0 - completeness
    
    return {
        "equilibrium_completeness": float(completeness),
        "reaction_progress": float(progress),
        "composition_shift": float(shift),
    }

