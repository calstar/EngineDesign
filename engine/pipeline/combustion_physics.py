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
from engine.pipeline.config_schemas import CombustionEfficiencyConfig

def calculate_eta_Lstar(
    Tc: float,
    Pc: float,
    R: float,
    m_dot_total: float,
    Ac: float,
    SMD: float,
    L_star: float,
    mu: float = 7e-5,
    Bm: float = None,  # Now optional - calculated from pressure-based formulation if not provided
    phi: float = 3.0,
    D0: float = 2e-5,
    rho_l: float = 800.0,
    gamma: float = None,  # For calculating cp_gas if available
    fuel_props: dict = None,  # Fuel properties from config (T_boil, L_vap, P_sat_coeffs)
) -> float:
    """
    Compute evaporation-based efficiency using L* and a d^2-law evaporation model.

    Parameters
    ----------
    Tc : float
        Chamber temperature [K]
    Pc : float
        Chamber pressure [Pa]
    R : float
        Gas constant of mixture [J/(kg·K)]
    m_dot_total : float
        Total mass flow rate [kg/s]
    Ac : float
        Chamber cross-sectional area [m^2]
    SMD : float
        Sauter mean diameter d32 [m]
    L_star : float
        Characteristic length (L*) [m]
    mu : float, optional
        Dynamic viscosity [Pa·s]
    Bm : float, optional
        Spalding mass number [-]. If None, calculated from pressure-based formulation.
    phi : float, optional
        LOX penalty constant [-]
    D0 : float, optional
        Reference diffusivity at 300 K, 1 atm [m^2/s]
    rho_l : float, optional
        Liquid fuel density [kg/m^3]
    gamma : float, optional
        Specific heat ratio for calculating cp_gas
    fuel_props : dict, optional
        Fuel properties from config:
        - T_boil: Boiling point [K]
        - L_vap: Latent heat [J/kg]
        - A_antoine, B_antoine, C_antoine: Antoine equation coefficients for P_sat

    Returns
    -------
    eta_Lstar : float
        Evaporation/mixing efficiency associated with length L* [-]
    """
    # Gas density and bulk velocity
    rho_g = Pc / (R * Tc)              # kg/m^3
    U = m_dot_total / (rho_g * Ac)     # m/s

    # Effective diffusivity at Tc, Pc
    D_eff = D0 * (Tc / 300.0)**1.75 * (101325.0 / Pc)  # m^2/s

    # Dimensionless groups
    Sc = mu / (rho_g * D_eff)
    Re = rho_g * U * float(SMD) / mu          # based on droplet diameter

    Sh = 2.0 + 0.6 * Re**0.5 * Sc**(1.0/3.0)

    # Calculate Spalding number if not provided
    # Using PRESSURE-BASED mass fraction formulation (recommended for rocket conditions)
    # Bm = Ys / (1 - Ys) where Ys = P_sat(T_s) / Pc
    if Bm is None:
        # Extract fuel properties from config or use RP-1 defaults
        if fuel_props is not None:
            T_boil = fuel_props.get("T_boil", fuel_props.get("boiling_point", 489.0))
            L_vap = fuel_props.get("L_vap", fuel_props.get("latent_heat", 300e3))
            # Antoine coefficients for vapor pressure (if available)
            A_ant = fuel_props.get("A_antoine", 6.9)  # RP-1 approx
            B_ant = fuel_props.get("B_antoine", 1400.0)
            C_ant = fuel_props.get("C_antoine", -60.0)
        else:
            # RP-1 defaults
            T_boil = 489.0  # K
            L_vap = 300e3   # J/kg
            A_ant = 6.9     # Approximate Antoine A for RP-1
            B_ant = 1400.0  # Approximate Antoine B
            C_ant = -60.0   # Approximate Antoine C
        
        # Estimate droplet surface temperature (approximation: between T_boil and Tc)
        # At high heat transfer rates, T_s approaches wet-bulb temperature
        T_s = min(T_boil + 50.0, 0.7 * Tc + 0.3 * T_boil)  # Weighted average
        
        # Calculate saturation vapor pressure at surface temperature
        # Using Clausius-Clapeyron approximation: P_sat = P_ref * exp(-L_vap/R_fuel * (1/T - 1/T_ref))
        # or Antoine equation: log10(P_sat) = A - B/(C + T)
        # Use Clausius-Clapeyron for simplicity (more robust)
        R_fuel = 8314.0 / 170.0  # J/(kg·K) for RP-1 (M ~ 170 g/mol)
        P_sat_boil = 101325.0  # At boiling point, P_sat = 1 atm
        P_sat = P_sat_boil * np.exp(-L_vap / R_fuel * (1.0/T_s - 1.0/T_boil))
        
        # Mass fraction at surface: Ys = P_sat / Pc (for dilute species, ideal mixing)
        Ys = P_sat / max(Pc, 1e3)  # Prevent division by zero
        
        # Pressure-based Spalding mass transfer number
        # Bm = Ys / (1 - Ys)
        # This naturally accounts for high chamber pressure reducing evaporation
        Ys = np.clip(Ys, 0.0, 0.95)  # Prevent division by zero, cap at 95%
        Bm = Ys / (1.0 - Ys)
        
        # CRITICAL: Clamp Bm to physically plausible range for rocket conditions
        # Without clamping, Bm can exceed 5-10 and cause unrealistic efficiency collapse
        Bm = np.clip(Bm, 0.01, 2.0)  # Upper bound of 2.0 as recommended

    # Evaporation constant K [m^2/s]
    K = ((8.0 * D_eff * rho_g) / rho_l) * Sh * np.log(1.0 + Bm)

    # LOX penalty → effective evaporation constant K_eff [m^2/s]
    K_eff = K / (1.0 + phi)

    # Length-based Damköhler number:
    # Da_L = (K_eff * L_star) / (U * SMD^2)   (dimensionless)
    Da_L = K_eff * L_star / (U * SMD**2)

    # Efficiency from d^2-law over length L*
    eta_Lstar = 1.0 - np.exp(-Da_L)


    return eta_Lstar


def calculate_residence_time(
    Lstar: float,
    Pc: float,
    cstar: float,
    gamma: float,
    R: float,
    Tc: float,
    Ac: float,
    At: float,
    m_dot_total: float,
) -> float:
    """
    Calculate characteristic residence time in chamber.
    
    τ_res = V_chamber * rho / mdot = (L* * At) * rho / mdot
    
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
    Ac : float
        Chamber area (m^2)
    At : float
        Throat area (m^2)
    m_dot_total : float
        Total mass flow rate (kg/s)
    
    Returns:
    --------
    tau_res : float
        Residence time [s]
    """
    # Gas density at chamber conditions
    rho_g = Pc / (R * Tc) if R > 0 and Tc > 0 else 1.0
    
    # Residence time = Volume * rho / mdot
    # Since L* = Volume / At, then Volume = L* * At
    tau_res = (Lstar * At * rho_g) / m_dot_total if m_dot_total > 0 else 0.001
    
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
    n_pressure = 1.5
    
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
    tau_ref = 5e-5  # 5 ms
    
    # Pressure effect (higher pressure → faster reactions)
    pressure_factor = (P_ref / max(Pc, 1e5)) ** n_pressure
    
    # Temperature effect (higher temperature → faster reactions)
    temp_factor = np.exp(Ea_norm * (T_ref / max(Tc, 1000.0) - 1.0))
    
    tau_chem = tau_ref * pressure_factor * temp_factor
    
    # Clamp to reasonable range (0.1 ms to 100 ms)
    tau_chem = np.clip(tau_chem, 0.1e-5, 1e-2)
    # print(f"tau_chem: {tau_chem}")
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
    Tc: float,
    Pc: float,
    R: float,
    Ac: float,
    At: float,
    Dinj: float,
    m_dot_total: float,
    Lstar: float,
    target_smd: float = 50e-6,  # 50 microns
    beta: float = 8.0,  # recirculation/mixing strength factor
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
    Tc : float
        Chamber temperature [K]
    Pc : float
        Chamber pressure [Pa]
    R : float
        Gas constant [J/(kg·K)]
    Ac : float
        Chamber area [m^2]
    Dinj : float
        Characteristic injector diameter (e.g., pintle tip) [m]
    m_dot_total : float
        Total mass flow rate [kg/s]
    Lstar : float
        Characteristic length [m]
    target_smd : float
        Target SMD for good atomization [m]
    beta : float
        Recirculation enhancement factor (dimensionless)
    
    Returns:
    --------
    eta_mix : float
        Mixing efficiency (0-1)
    """
    # Calculate gas density and bulk velocity from chamber conditions
    rho_g = Pc / max(R * Tc, 1e-6)
    U = m_dot_total / max(rho_g * Ac, 1e-8)

    # --- turbulence-based transport properties ---
    # Use representative hot-gas viscosity for Reynolds number calculation
    mu_g = 7.0e-5  # Pa·s, representative hot-gas viscosity

    # Calculate Reynolds number based on injector diameter and bulk flow
    Re = rho_g * U * Dinj / max(mu_g, 1e-8)
    #print(f"rho_g: {rho_g}, U: {U}, Dinj: {Dinj}, mu_g: {mu_g}")
    Re = max(Re, 1.0)

    # Estimate turbulence intensity from canonical high-Re pipe flow correlation
    I_est = np.clip(0.055 * Re ** (-0.0407), 0.02, 0.3)

    # Use maximum of estimated and user-provided turbulence intensity
    I_eff = max(I_est, turbulence_intensity)

    # Integral length scale: typically 7% of injector diameter for pipe flow
    Lt = max(0.07 * Dinj, 1e-5)

    # k-epsilon model constant (standard value)
    C_mu = 0.09

    # Turbulent kinetic energy: k = (3/2) * (u'^2) where u' = U * I
    k_est = 1.5 * (U * I_eff) ** 2

    # Dissipation rate: epsilon = C_mu^(3/4) * k^(3/2) / Lt (k-epsilon scaling)
    epsilon_est = C_mu ** 0.75 * k_est ** 1.5 / max(Lt, 1e-6)
    epsilon_est = max(epsilon_est, 1e-8)

    # Eddy viscosity: mu_t = rho * C_mu * k^2 / epsilon
    mu_t = rho_g * C_mu * (k_est ** 2) / epsilon_est
    mu_t = max(mu_t, 0.0)

    # Turbulent diffusivity: D_t = mu_t / rho (turbulent Schmidt number ≈ 1)
    D_t = mu_t / max(rho_g, 1e-8)

    # Molecular diffusivity: scales with temperature^1.75 and inversely with pressure
    D_m = 2.0e-5 * (Tc / 300.0) ** 1.75 * (101325.0 / max(Pc, 1e3))

    # Total effective diffusivity: sum of molecular and turbulent contributions
    D_total = max(D_m + D_t, 1e-8)

    # Physics-based evaporation factor
    from engine.pipeline.physics_based_replacements import (
        calculate_evaporation_factor_physics,
        calculate_smd_factor_physics,
        calculate_recirculation_length_physics,
    )
    
    # Calculate Reynolds number for physics-based calculations
    Re_chamber = rho_g * U * np.sqrt(4.0 * Ac / np.pi) / max(mu_g, 1e-8)
    
    if chamber_length > 0:
        evap_factor = calculate_evaporation_factor_physics(
            evaporation_length=evaporation_length,
            chamber_length=chamber_length,
            SMD=SMD,
            target_smd=target_smd,
            Pc=Pc,
            Tc=Tc,
        )
    else:
        evap_factor = 0.5  # Unknown
    # print(f"evaporation_length: {evaporation_length}, evap_factor: {evap_factor}")
    
    # Correct residence time for mixing: tau = V * rho / mdot = (Lstar * At) * rho / mdot
    # Using Lstar / U where U = mdot / (rho * Ac) gives V * rho * (Ac/At) / mdot, which is wrong.
    # We should use U_throat_equiv = mdot / (rho * At)
    U_throat_equiv = m_dot_total / max(rho_g * At, 1e-8)
    tau_res_eff = (Lstar / max(U_throat_equiv, 1e-4)) * (1.0 / evap_factor)

    # Physics-based recirculation length
    Dc = np.sqrt(4.0 * Ac / np.pi)
    # Need injector velocity for recirculation calculation
    # Estimate from mass flow: U_inj ≈ mdot / (rho × A_inj)
    # For now, use chamber velocity as proxy
    U_inj_estimate = U  # Approximate
    L_recirc = calculate_recirculation_length_physics(
        D_chamber=Dc,
        d_pintle_tip=Dinj,  # Use injector diameter as proxy for pintle tip
        fuel_velocity=U_inj_estimate,
        lox_velocity=U_inj_estimate,
        Re_chamber=Re_chamber,
    )
    # print(f"Dc: {Dc}, L_recirc: {L_recirc}")

    # Physics-based SMD factor
    # Calculate injector Reynolds and Weber for physics-based calculation
    Re_injector = Re_chamber  # Approximate
    We_injector = 20.0  # Typical for pintle injectors
    smd_factor = calculate_smd_factor_physics(
        SMD=SMD,
        target_smd=target_smd,
        Re_injector=Re_injector,
        We_injector=We_injector,
    )

    #print(f"D_total: {D_total}")
    tau_mix = (L_recirc ** 2) / (beta * D_total)
    tau_mix = tau_mix / max(smd_factor, 0.1)
    tau_mix = max(tau_mix, 1e-6)

    Da_mix = tau_res_eff / tau_mix
    Da_mix = np.clip(Da_mix, 0.0, 50.0)
    # print(f'tau_res_eff: {tau_res_eff}, tau_mix: {tau_mix}, Da_mix: {Da_mix}')
    eta_m = 1.0 - np.exp(-Da_mix)
    
    return float(np.clip(eta_m, 0.0, 1.0))

def calculate_combustion_efficiency_advanced(
    Lstar: float,
    Pc: float,
    Tc: float,
    cstar_ideal: float,
    gamma: float,
    R: float,
    MR: float,
    config: CombustionEfficiencyConfig,
    Ac: float,
    At: float,
    Dinj: float,
    m_dot_total: float,
    spray_diagnostics: Optional[Dict] = None,
    turbulence_intensity: float = 0.08,
    chamber_length: Optional[float] = None,
    Tc_kinetics: Optional[float] = None,
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
        Chamber temperature [K] (Used for residence time - Ideal is conservative)
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
    Ac : float
        Chamber area [m^2]
    At : float
        Throat area [m^2]
    Dinj : float
        Characteristic injector diameter [m]
    m_dot_total : float
        Total mass flow rate [kg/s]
    spray_diagnostics : dict, optional
        Spray diagnostics (SMD, evaporation length, etc.)
    turbulence_intensity : float
        Turbulence intensity (0-1)
    chamber_length : float, optional
        Physical chamber length [m]
    Tc_kinetics : float, optional
        Temperature to use for reaction kinetics [K]. If None, uses Tc.
        (Actual/Effective is conservative)
    
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
    # Use Tc_kinetics for reaction-rate limited processes if provided
    T_react = Tc_kinetics if Tc_kinetics is not None else Tc

    # 1. L*-based efficiency (finite residence time)
    # Uses Tc (Ideal) for conservative residence time (shorter)
    if config.model == "constant":
        eta_Lstar = 1.0 - config.C
    elif config.model == "linear":
        eta_Lstar = 1.0 - config.C * (1.0 - Lstar / 1.0)
        eta_Lstar = np.clip(eta_Lstar, 0.0, 1.0)
    else:  # exponential (default)
        if spray_diagnostics is not None:
            SMD = spray_diagnostics.get("D32_O", 0.0) or spray_diagnostics.get("D32_F", 0.0) or 100e-6
        else:
            SMD = 100e-6  # Default SMD if diagnostics not available
            raise ValueError("SMD is required for eta_Lstar calculation")
        
        # Use Tc (Ideal) for residence time part, but T_react (Actual) for evaporation physics
        eta_Lstar = calculate_eta_Lstar(Tc, Pc, R, m_dot_total, Ac, SMD, Lstar)
    
    # Clamp L* efficiency
    eta_Lstar = np.clip(eta_Lstar, config.mixture_efficiency_floor, 1.0)
    
    # 2. Reaction kinetics efficiency (Damköhler number)
    # tau_res uses Tc (Ideal) -> shorter time (conservative)
    tau_res = calculate_residence_time(Lstar, Pc, cstar_ideal, gamma, R, Tc, Ac, At, m_dot_total)
    # tau_chem uses T_react (Actual) -> longer time (conservative)
    tau_chem = calculate_reaction_time_scale(Pc, T_react, MR, gamma)
    Da = calculate_damkohler_number(tau_res, tau_chem)
    
    # Efficiency based on Damköhler number
    eta_kinetics = 1 - np.exp(-Da**0.5)
    eta_kinetics = np.clip(eta_kinetics, 0.5, 1.0)
    
    # 3. Mixing efficiency
    if spray_diagnostics is not None:
        SMD = spray_diagnostics.get("D32_O", 0.0) or spray_diagnostics.get("D32_F", 0.0) or 100e-6
        x_star = spray_diagnostics.get("x_star", 0.0) or 0.1
        
        if chamber_length is None:
            raise ValueError("chamber_length is required for mixing efficiency calculation")
            
        # Use Tc (Ideal) for mixing physics to be conservative on residence time (shorter window)
        eta_mixing = calculate_mixing_efficiency(
            SMD, x_star, chamber_length, turbulence_intensity,
            Tc, Pc, R, Ac, At, Dinj, m_dot_total, Lstar,
            target_smd=config.target_smd_microns * 1e-6 if hasattr(config, 'target_smd_microns') else 50e-6
        )
    else:
        eta_mixing = 1.0  # Assume perfect mixing if no diagnostics
    
    # Apply spray quality penalty if enabled
    if config.use_spray_correction and spray_diagnostics is not None:
        spray_quality_good = spray_diagnostics.get("constraints_satisfied", True)
        if not spray_quality_good:
            eta_mixing *= config.spray_penalty_factor
    
    eta_mixing = np.clip(eta_mixing, config.mixture_efficiency_floor, 1.0)
    
    # 4. Turbulence efficiency (enhancement)
    if turbulence_intensity < 0.05:
        eta_turbulence = 0.9
    elif turbulence_intensity < 0.15:
        eta_turbulence = 0.95 + 0.05 * (turbulence_intensity / 0.15)
    else:
        eta_turbulence = 1.0 - 0.1 * ((turbulence_intensity - 0.15) / 0.35)
    
    eta_turbulence = np.clip(eta_turbulence, 0.85, 1.0)
    
    # 5. Combined efficiency
    print(f"[ETA_DEBUG] INPUTS: Pc={Pc/1e6:.3f} MPa, Tc_ideal={Tc:.0f} K, T_react={T_react:.0f} K, Lstar={Lstar:.3f} m")
    print(f"[ETA_DEBUG] INPUTS: SMD={SMD*1e6:.1f} µm, Ac={Ac*1e6:.2f} mm², Dinj={Dinj*1e3:.2f} mm")
    print(f"[ETA_DEBUG] INPUTS: m_dot_total={m_dot_total:.4f} kg/s, turbulence_intensity={turbulence_intensity:.4f}")
    print(f"[ETA_DEBUG] DERIVED: U={m_dot_total/(Pc/(R*Tc)*Ac):.2f} m/s, Da={Da:.4f}, tau_res={tau_res*1e3:.3f} ms, tau_chem={tau_chem*1e6:.3f} µs")
    print(f"[ETA_DEBUG] OUTPUTS: eta_Lstar={eta_Lstar:.4f}, eta_kinetics={eta_kinetics:.4f}, eta_mixing={eta_mixing:.4f}, eta_turbulence={eta_turbulence:.4f}")
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

