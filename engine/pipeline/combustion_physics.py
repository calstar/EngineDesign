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
import warnings
from engine.pipeline.config_schemas import CombustionEfficiencyConfig

def calculate_eta_Lstar(
    Tc: float,
    Pc: float,
    R: float,
    m_dot_total: float,
    Ac: float,
    At: float,
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
    At : float
        Geometric throat area [m^2] (Consistent with nozzle solver geometric At)
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
    Da_L : float
        Length-based Damköhler number [-]
    """
    # ... (logic) ...
    # (re-reading end of function)
    # Gas density and bulk velocity (Chamber-average approximation)
    rho_ch = Pc / (R * Tc)              # [kg/m^3]
    
    # U_bulk: chamber-average convective scale for droplet Re calculation
    U_bulk = m_dot_total / (rho_ch * Ac) 
    
    # G_throat: Throat mass flux for residence-time scaling
    G_throat = m_dot_total / At if At > 0 else 1.0

    # Effective diffusivity at Tc, Pc
    D_eff = D0 * (Tc / 300.0)**1.75 * (101325.0 / Pc)  # m^2/s

    # Dimensionless groups
    Sc = mu / (rho_ch * D_eff)
    Re = rho_ch * U_bulk * float(SMD) / mu          # based on droplet diameter and bulk flow

    Sh = 2.0 + 0.6 * Re**0.5 * Sc**(1.0/3.0)

    # Calculate Spalding number if not provided
    # Using centralized spalding module for consistent calculations
    if Bm is None:
        from engine.pipeline.spalding import (
            calculate_droplet_surface_temperature,
            calculate_spalding_pressure_based,
        )
        
        # Calculate T_s and Spalding number using centralized functions
        # Use pressure-based formulation with reference pressure for solver stability
        T_s, _ = calculate_droplet_surface_temperature(Tc, Pc, fuel_props)
        Bm = calculate_spalding_pressure_based(
            Tc, Pc, T_s, fuel_props, use_reference_pressure=True
        )

    # Evaporation constant K [m^2/s]
    K = ((8.0 * D_eff * rho_ch) / rho_l) * Sh * np.log(1.0 + Bm)

    # LOX penalty → effective evaporation constant K_eff [m^2/s]
    K_eff = K / (1.0 + phi)

    # Length-based Damköhler number:
    # tau_res = L* * rho_ch / G_throat = (V/At) * rho_ch / (mdot/At) = V*rho/mdot
    # t_evap = SMD^2 / K_eff (approximate d^2 law time scale)
    # Da_L = tau_res / t_evap = (K_eff * L_star * rho_ch) / (G_throat * SMD^2)
    # Using G_throat (throat mass flux) ensures correct residence-time scaling for L*.
    Da_L = (K_eff * L_star * rho_ch) / (G_throat * SMD**2) if G_throat > 0 else 0.0

    # Efficiency from d^2-law over length L*
    eta_Lstar = 1.0 - np.exp(-Da_L)


    return eta_Lstar, Da_L


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
    # Gas density at chamber conditions (Chamber approximation)
    rho_ch = Pc / (R * Tc) if R > 0 and Tc > 0 else 1.0
    
    # Residence time = Volume * rho / mdot
    # Since L* = Volume / At, then Volume = L* * At
    # tau_res = (L* * At * rho_ch) / mdot = L* * rho_ch / (mdot/At) = L* * rho_ch / G_throat
    G_throat = m_dot_total / At if At > 0 else 1.0
    tau_res = (Lstar * rho_ch) / G_throat if G_throat > 0 else 0.001
    
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
    u_fuel: Optional[float] = None,
    u_lox: Optional[float] = None,
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
    u_fuel : float, optional
        Fuel injection velocity [m/s]
    u_lox : float, optional
        LOX injection velocity [m/s]
    target_smd : float
        Target SMD for good atomization [m]
    beta : float
        Recirculation enhancement factor (dimensionless)
    
    Returns:
    --------
    eta_mix : float
        Mixing efficiency (0-1)
    """
    # Calculate gas density and bulk velocity from chamber conditions (Chamber approximation)
    rho_ch = Pc / max(R * Tc, 1e-6)
    
    # U_bulk: chamber-average convective scale
    U_bulk = m_dot_total / max(rho_ch * Ac, 1e-8)

    # --- turbulence-based transport properties ---
    # Use representative hot-gas viscosity for Reynolds number calculation
    mu_g = 7.0e-5  # Pa·s, representative hot-gas viscosity

    # Calculate Reynolds number based on injector diameter and bulk flow
    Re = rho_ch * U_bulk * Dinj / max(mu_g, 1e-8)
    #print(f"rho_ch: {rho_ch}, U_bulk: {U_bulk}, Dinj: {Dinj}, mu_g: {mu_g}")
    Re = max(Re, 1.0)

    # Estimate turbulence intensity from canonical high-Re pipe flow correlation
    I_est = np.clip(0.055 * Re ** (-0.0407), 0.02, 0.3)

    # Use maximum of estimated and user-provided turbulence intensity
    I_eff = max(I_est, turbulence_intensity)

    # Integral length scale: typically 7% of injector diameter for pipe flow
    Lt = max(0.07 * Dinj, 1e-5)

    # k-epsilon model constant (standard value)
    C_mu = 0.09

    # Turbulent kinetic energy: k = (3/2) * (u'^2) where u' = U_bulk * I
    k_est = 1.5 * (U_bulk * I_eff) ** 2

    # Dissipation rate: epsilon = C_mu^(3/4) * k^(3/2) / Lt (k-epsilon scaling)
    epsilon_est = C_mu ** 0.75 * k_est ** 1.5 / max(Lt, 1e-6)
    epsilon_est = max(epsilon_est, 1e-8)

    # Eddy viscosity: mu_t = rho * C_mu * k^2 / epsilon
    mu_t = rho_ch * C_mu * (k_est ** 2) / epsilon_est
    mu_t = max(mu_t, 0.0)

    # Turbulent diffusivity: D_t = mu_t / rho (turbulent Schmidt number ≈ 1)
    D_t = mu_t / max(rho_ch, 1e-8)

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
    Re_chamber = rho_ch * U_bulk * np.sqrt(4.0 * Ac / np.pi) / max(mu_g, 1e-8)
    
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
    
    # Correct residence time for mixing: tau_res = V * rho / mdot = (Lstar * At) * rho / mdot
    # Using G_throat = mdot / At: tau_res = Lstar * rho / G_throat
    G_throat = m_dot_total / max(At, 1e-8)
    tau_res_eff = (Lstar * rho_ch / max(G_throat, 1e-4)) * (1.0 / evap_factor)

    # Physics-based recirculation length
    Dc = np.sqrt(4.0 * Ac / np.pi)
    
    # Use provided injection velocities (near-field) if available, otherwise fall back to chamber bulk velocity
    if u_fuel is None or u_lox is None:
        # Log/Warn if fallback is used to indicate proxy usage
        # In a real system, use a proper logger
        # print(f"[MIXING_WARN] u_fuel or u_lox missing, falling back to U_bulk={U_bulk:.2f} m/s proxy")
        pass

    u_f_inj = u_fuel if u_fuel is not None else U_bulk
    u_o_inj = u_lox if u_lox is not None else U_bulk
    
    # Sanity checks for injection velocities
    u_f_inj = float(np.clip(u_f_inj, 1.0, 500.0))
    u_o_inj = float(np.clip(u_o_inj, 1.0, 500.0))
    
    L_recirc = calculate_recirculation_length_physics(
        D_chamber=Dc,
        d_pintle_tip=Dinj,  # Use injector diameter as proxy for pintle tip
        fuel_velocity=u_f_inj,
        lox_velocity=u_o_inj,
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
    
    # Debug prints for mixing efficiency diagnosis
    print(f"[MIXING_DEBUG] === Mixing Efficiency Breakdown ===")
    print(f"[MIXING_DEBUG] Gas Properties: rho_ch={rho_ch:.4f} kg/m³, U_bulk={U_bulk:.2f} m/s")
    print(f"[MIXING_DEBUG] Turbulence: I_est={I_est:.4f}, I_eff={I_eff:.4f}, Re={Re:.0f}")
    print(f"[MIXING_DEBUG] k-epsilon: k={k_est:.2f} m²/s², epsilon={epsilon_est:.2f} m²/s³, Lt={Lt*1e3:.2f} mm")
    print(f"[MIXING_DEBUG] Diffusivity: D_m={D_m:.2e} m²/s, D_t={D_t:.2e} m²/s, D_total={D_total:.2e} m²/s")
    print(f"[MIXING_DEBUG] Geometry: Dc={Dc*1e3:.1f} mm, L_recirc={L_recirc*1e3:.1f} mm")
    print(f"[MIXING_DEBUG] Factors: evap_factor={evap_factor:.4f}, smd_factor={smd_factor:.4f}")
    print(f"[MIXING_DEBUG] Time Scales: tau_res_eff={tau_res_eff*1e3:.3f} ms, tau_mix={tau_mix*1e3:.3f} ms")
    print(f"[MIXING_DEBUG] Da_mix={Da_mix:.4f} -> eta_m_raw={1.0 - np.exp(-Da_mix):.4f}")
    
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
    u_fuel: Optional[float] = None,
    u_lox: Optional[float] = None,
    spray_diagnostics: Optional[Dict] = None,
    turbulence_intensity: float = 0.08,
    chamber_length: Optional[float] = None,
    Tc_kinetics: Optional[float] = None,
    fuel_props: Optional[Dict] = None,
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
        eta_Lstar, Da_L = calculate_eta_Lstar(Tc, Pc, R, m_dot_total, Ac, At, SMD, Lstar, fuel_props=fuel_props)

    if config.model != "exponential":
        # Back-calculate Da_L for logging if using non-exponential models
        Da_L = np.inf if eta_Lstar > 0.999 else -np.log(max(1.0 - eta_Lstar, 1e-10))

    
    # Clamp L* efficiency
    eta_Lstar_raw = eta_Lstar
    eta_Lstar = np.clip(eta_Lstar, config.mixture_efficiency_floor, 1.0)
    if eta_Lstar != eta_Lstar_raw:
        warnings.warn(f"[ETA_CLAMP] eta_Lstar clamped from {eta_Lstar_raw:.4f} to {eta_Lstar:.4f}")
    
    # 2. Reaction kinetics efficiency (Damköhler number)
    # tau_res uses Tc (Ideal) -> shorter time (conservative)
    tau_res = calculate_residence_time(Lstar, Pc, cstar_ideal, gamma, R, Tc, Ac, At, m_dot_total)
    # tau_chem uses T_react (Actual) -> longer time (conservative)
    tau_chem = calculate_reaction_time_scale(Pc, T_react, MR, gamma)
    Da = calculate_damkohler_number(tau_res, tau_chem)
    
    # Efficiency based on Damköhler number
    eta_kinetics_raw = 1 - np.exp(-Da**0.5)
    eta_kinetics = np.clip(eta_kinetics_raw, 0.5, 1.0)
    if eta_kinetics != eta_kinetics_raw:
        warnings.warn(f"[ETA_CLAMP] eta_kinetics clamped from {eta_kinetics_raw:.4f} to {eta_kinetics:.4f}")
    
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
            u_fuel=u_fuel, u_lox=u_lox,
            target_smd=config.target_smd_microns * 1e-6 if hasattr(config, 'target_smd_microns') else 50e-6
        )
    else:
        eta_mixing = 1.0  # Assume perfect mixing if no diagnostics
    
    # Apply spray quality penalty if enabled
    if config.use_spray_correction and spray_diagnostics is not None:
        spray_quality_good = spray_diagnostics.get("constraints_satisfied", True)
        if not spray_quality_good:
            eta_mixing *= config.spray_penalty_factor
    
    eta_mixing_raw = eta_mixing
    eta_mixing = np.clip(eta_mixing, config.mixture_efficiency_floor, 1.0)
    if eta_mixing != eta_mixing_raw:
        warnings.warn(f"[ETA_CLAMP] eta_mixing clamped from {eta_mixing_raw:.4f} to {eta_mixing:.4f}")
    
    # 4. Turbulence efficiency (enhancement)
    if turbulence_intensity < 0.05:
        eta_turbulence_raw = 0.9
    elif turbulence_intensity < 0.15:
        eta_turbulence_raw = 0.95 + 0.05 * (turbulence_intensity / 0.15)
    else:
        eta_turbulence_raw = 1.0 - 0.1 * ((turbulence_intensity - 0.15) / 0.35)
    
    eta_turbulence = np.clip(eta_turbulence_raw, 0.85, 1.0)
    if eta_turbulence != eta_turbulence_raw:
        warnings.warn(f"[ETA_CLAMP] eta_turbulence clamped from {eta_turbulence_raw:.4f} to {eta_turbulence:.4f}")
    
    # 5. Combined efficiency
    print(f"[ETA_DEBUG] INPUTS: Pc={Pc/1e6:.3f} MPa, Tc_ideal={Tc:.0f} K, T_react={T_react:.0f} K, Lstar={Lstar:.3f} m")
    print(f"[ETA_DEBUG] INPUTS: SMD={SMD*1e6:.1f} µm, Ac={Ac*1e6:.2f} mm², At={At*1e6:.2f} mm², Dinj={Dinj*1e3:.2f} mm")
    print(f"[ETA_DEBUG] INPUTS: m_dot_total={m_dot_total:.4f} kg/s, u_fuel_inj={u_fuel if u_fuel else 0:.1f} m/s, u_lox_inj={u_lox if u_lox else 0:.1f} m/s")
    
    rho_ch = Pc / (R * Tc)
    U_bulk = m_dot_total / (rho_ch * Ac)
    G_throat = m_dot_total / At
    
    print(f"[ETA_DEBUG] DERIVED: rho_ch={rho_ch:.4f} kg/m³, U_bulk={U_bulk:.2f} m/s, G_throat={G_throat:.1f} kg/m²s")
    print(f"[ETA_DEBUG] DERIVED: tau_res_ch={tau_res*1e3:.3f} ms, Da_kinetics={Da:.4f}, Da_L={Da_L:.4f}")
    print(f"[ETA_DEBUG] OUTPUTS: eta_Lstar={eta_Lstar:.4f}, eta_kinetics={eta_kinetics:.4f}, eta_mixing={eta_mixing:.4f}, eta_turbulence={eta_turbulence:.4f}")
    
    eta_total_raw = eta_Lstar * eta_kinetics * eta_mixing * eta_turbulence
    
    # Apply cooling efficiency if provided (external)
    # This would be multiplied in by the caller
    
    # Final clamp
    lower_bound = min(config.mixture_efficiency_floor, config.cooling_efficiency_floor)
    eta_total = np.clip(eta_total_raw, lower_bound, 1.0)
    if eta_total != eta_total_raw:
        warnings.warn(f"[ETA_CLAMP] eta_total clamped from {eta_total_raw:.4f} to {eta_total:.4f}")
    
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

