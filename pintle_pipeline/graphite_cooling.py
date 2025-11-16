"""Graphite throat insert cooling and recession model - Physics-based with oxidation heat feedback"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
from pintle_pipeline.config_schemas import GraphiteInsertConfig

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant
R_GAS = 8.314  # J/(mol·K) - universal gas constant
MW_C = 0.012  # kg/mol - molar mass of carbon
MW_O2 = 0.032  # kg/mol - molar mass of oxygen


def calculate_throat_recession_multiplier(
    chamber_pressure: float,
    chamber_velocity: float,
    throat_velocity: float,
    chamber_heat_flux: float,
    gamma: float = 1.2,
) -> float:
    """
    Calculate throat recession multiplier based on local flow conditions using Bartz correlation.
    
    Throat recession is typically 1.2-2.5x higher than chamber due to:
    1. Higher velocity → Higher convective heat transfer
    2. Sonic conditions → Maximum heat flux
    3. Pressure gradient → Enhanced mass transfer
    4. Turbulence amplification near throat
    
    Uses Bartz correlation for heat flux ratio:
        q_throat / q_chamber ∝ (V_throat / V_chamber)^0.8 × (P_throat / P_chamber)^0.2
    
    This is used to scale chamber heat flux to throat conditions for graphite insert recession calculations.
    
    Parameters:
    -----------
    chamber_pressure : float
        Chamber pressure [Pa]
    chamber_velocity : float
        Chamber gas velocity [m/s]
    throat_velocity : float
        Throat gas velocity (sonic) [m/s]
    chamber_heat_flux : float
        Chamber wall heat flux [W/m²] (used for validation, not in calculation)
    gamma : float
        Specific heat ratio
    
    Returns:
    --------
    multiplier : float
        Throat recession multiplier (typically 1.2-2.5)
    """
    if chamber_velocity <= 0 or throat_velocity <= 0:
        return 1.3  # Default fallback
    
    # Velocity ratio effect (dominant factor)
    velocity_ratio = throat_velocity / chamber_velocity
    velocity_factor = velocity_ratio ** 0.8
    
    # Pressure ratio effect (throat is at critical pressure)
    # P_throat / P_chamber ≈ (2/(γ+1))^(γ/(γ-1))
    pressure_ratio = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    pressure_factor = pressure_ratio ** 0.2
    
    # Heat flux ratio from Bartz correlation
    heat_flux_ratio = velocity_factor * pressure_factor
    
    # Recession rate is proportional to heat flux
    # Add a base factor for enhanced turbulence at throat
    turbulence_enhancement = 1.1
    
    multiplier = heat_flux_ratio * turbulence_enhancement
    
    # Clamp to reasonable bounds (1.2 to 2.5)
    multiplier = float(np.clip(multiplier, 1.2, 2.5))
    
    return multiplier


def compute_graphite_recession(
    net_heat_flux: float,
    throat_temperature: float,
    gas_temperature: float,
    graphite_config: GraphiteInsertConfig,
    throat_area: float,
    pressure: float,
    gas_density: Optional[float] = None,
    gas_viscosity: Optional[float] = None,
    oxygen_mass_fraction: Optional[float] = None,
    characteristic_length: Optional[float] = None,
    gas_velocity: Optional[float] = None,
    heat_transfer_coefficient: Optional[float] = None,
    backside_temperature: Optional[float] = None,
    effective_thickness: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate graphite throat insert recession rate using physics-based models with oxidation heat feedback.
    
    Implements the theory from graphite_oxidation_feedback.tex:
    - Energy balance: q''_in + q''_fb - q''_rad = q''_cond + m''_th H*_th
    - Oxidation kinetics: kinetic-limited and diffusion-limited rates
    - Feedback fraction: f_fb based on Damköhler number and blowing parameter
    - Iterative solution for surface temperature
    
    Graphite recession is driven by:
    1. Chemical oxidation (C + O2 -> CO/CO2) - dominant mechanism
    2. Thermal ablation (sublimation) - only at very high temperatures (>2800 K)
    
    Parameters:
    -----------
    net_heat_flux : float
        Reference convective heat flux [W/m²] at initial T_s (used to estimate h_g if not provided)
    throat_temperature : float
        Initial guess for throat surface temperature [K]
    gas_temperature : float
        Free-stream gas temperature [K]
    graphite_config : GraphiteInsertConfig
        Graphite insert configuration
    throat_area : float
        Throat area [m²]
    pressure : float
        Chamber/throat pressure [Pa]
    gas_density : float, optional
        Gas density [kg/m³]. If None, estimated from ideal gas law.
    gas_viscosity : float, optional
        Gas dynamic viscosity [Pa·s]. If None, estimated (~4e-5 Pa·s for combustion products).
    oxygen_mass_fraction : float, optional
        Oxygen mass fraction in free stream. If None, estimated (~0.3 for LOX/RP-1).
    characteristic_length : float, optional
        Characteristic length for Sherwood number [m]. If None, uses throat diameter.
    gas_velocity : float, optional
        Gas velocity [m/s]. If None, estimated from sonic conditions.
    heat_transfer_coefficient : float, optional
        Convective heat transfer coefficient h_g [W/(m²·K)]. If None, estimated from net_heat_flux.
    backside_temperature : float, optional
        Backside temperature for conduction [K]. If None, uses default 300 K.
    effective_thickness : float, optional
        Effective thickness for conduction [m]. If None, uses char_layer_thickness + 0.001 m.
    
    Returns:
    --------
    dict
        Recession metrics including recession rate [m/s], mass flux [kg/(m²·s)],
        surface temperature [K], and detailed heat transfer breakdown.
    """
    if not graphite_config.enabled or throat_area <= 0:
        return {
            "enabled": False,
            "recession_rate": 0.0,
            "mass_flux": 0.0,
            "surface_temperature": throat_temperature,
            "heat_removed": 0.0,
            "oxidation_rate": 0.0,
            "oxidation_mass_flux": 0.0,
            "thermal_mass_flux": 0.0,
            "feedback_fraction": 0.0,
            "q_feedback": 0.0,
            "q_radiation": 0.0,
            "q_conduction": 0.0,
        }
    
    # Get optional config fields with safe defaults
    emissivity = getattr(graphite_config, "emissivity", 0.8) or 0.8
    T_env = getattr(graphite_config, "ambient_temperature", 300.0) or 300.0
    f_fb_min = getattr(graphite_config, "feedback_fraction_min", 0.0) or 0.0
    f_fb_max = getattr(graphite_config, "feedback_fraction_max", 0.2) or 0.2
    pressure_exponent = getattr(graphite_config, "oxidation_pressure_exponent", 0.5) or 0.5
    mixture_mw = getattr(graphite_config, "mixture_mw", 0.024) or 0.024  # kg/mol for hot products
    stoichiometry_ratio = getattr(graphite_config, "oxidation_stoichiometry_ratio", 1.0) or 1.0  # mol C per mol O2 (1.0 for CO2, 2.0 for CO)
    
    # Oxidation enthalpy: tie to stoichiometry if not explicitly set
    # CO2 channel (ratio=1.0): ~32.8 MJ/kgC, CO channel (ratio=2.0): ~10.1 MJ/kgC
    if hasattr(graphite_config, "oxidation_enthalpy") and graphite_config.oxidation_enthalpy is not None:
        oxidation_enthalpy = graphite_config.oxidation_enthalpy
    else:
        # Default based on stoichiometry ratio
        if stoichiometry_ratio <= 1.1:
            oxidation_enthalpy = 32.8e6  # J/kg C for CO2 channel
        else:
            oxidation_enthalpy = 10.1e6  # J/kg C for CO channel
    
    # Ablation surface temperature (pinned when thermal ablation is active)
    T_abl = getattr(graphite_config, "ablation_surface_temperature", 3000.0) or 3000.0  # K
    
    # Estimate missing gas properties
    if gas_density is None:
        # Estimate from ideal gas law: rho = P / (R_gas * T)
        R_avg = 300.0  # J/(kg·K) - average gas constant for combustion products
        gas_density = pressure / (R_avg * gas_temperature)
        gas_density = max(gas_density, 0.1)
    
    if gas_viscosity is None:
        gas_viscosity = 4.0e-5  # Pa·s - typical for combustion products at high T
    
    if oxygen_mass_fraction is None:
        oxygen_mass_fraction = 0.3  # Typical for LOX/RP-1 at stoichiometric
    
    if characteristic_length is None:
        # Estimate from throat area
        D_throat = np.sqrt(4.0 * throat_area / np.pi)
        characteristic_length = D_throat
    
    if gas_velocity is None:
        # Estimate sonic velocity at throat
        gamma_est = 1.2  # Typical for combustion products
        R_est = 300.0  # J/(kg·K)
        gas_velocity = np.sqrt(gamma_est * R_est * gas_temperature)
    
    # Estimate or extract heat transfer coefficient
    if heat_transfer_coefficient is None:
        # Reconstruct h_g from net_heat_flux assuming it was calculated at initial T_s
        delta_T_initial = max(gas_temperature - throat_temperature, 10.0)
        heat_transfer_coefficient = net_heat_flux / delta_T_initial
        heat_transfer_coefficient = max(heat_transfer_coefficient, 100.0)  # Minimum reasonable value
    else:
        heat_transfer_coefficient = max(heat_transfer_coefficient, 100.0)
    
    # Material properties
    rho_s = graphite_config.material_density  # kg/m³
    k_s = graphite_config.thermal_conductivity  # W/(m·K)
    cp_s = graphite_config.specific_heat  # J/(kg·K)
    T_back = backside_temperature if backside_temperature is not None else 300.0  # K - backside temperature (cooled)
    if effective_thickness is None:
        effective_thickness = graphite_config.char_layer_thickness + 0.001  # m
    effective_thickness = max(effective_thickness, 0.001)  # Minimum thickness
    
    # Oxidation kinetics parameters
    Ea = graphite_config.activation_energy  # J/mol
    T_ref = graphite_config.oxidation_reference_temperature  # K
    P_ref = graphite_config.oxidation_reference_pressure  # Pa
    
    # Reference mass flux at (T_ref, P_ref)
    # Convert recession rate to mass flux
    j_ref = graphite_config.oxidation_rate * rho_s  # kg/(m²·s) at reference conditions
    
    # Calculate Reynolds number once (flow property, independent of oxidation)
    Re = gas_density * gas_velocity * characteristic_length / gas_viscosity
    Re = max(Re, 100.0)  # Clamp once at the top
    
    # Calculate skin friction coefficient from Re (used for blowing parameter)
    Cf = 0.026 * (Re ** -0.25)  # Turbulent pipe correlation
    Cf = max(Cf, 0.001)  # Minimum
    
    # Initialize variables for return values
    Da = 0.0
    B_m = 0.0
    
    # Iterative solution for surface temperature
    T_s = throat_temperature  # Initial guess
    max_iter = 50
    tol = 1.0  # K - convergence tolerance
    damp = 0.5  # Damping factor for Newton step
    feedback_max_iter = 10  # Max iterations for feedback loop convergence
    feedback_tol = 1e-6  # Relative tolerance for feedback loop convergence
    
    for iter in range(max_iter):
        T_s_old = T_s
        
        # 1. CONVECTIVE HEAT FLUX: q''_in = h_g * (T_g - T_s)
        # Note: q_in can be negative if T_s > T_g (physically valid - wall cooling gas)
        q_in = heat_transfer_coefficient * (gas_temperature - T_s)
        
        # 2. RADIATIVE COOLING
        q_rad = emissivity * SIGMA * (T_s**4 - T_env**4)
        q_rad = max(q_rad, 0.0)
        
        # 3. OXIDATION KINETICS
        m_dot_ox = 0.0
        k_m_molar = 0.0
        X_O2 = 0.0
        p_O2 = 0.0
        
        if T_s > graphite_config.oxidation_temperature:
            # Convert oxygen mass fraction to mole fraction
            X_O2 = oxygen_mass_fraction * (mixture_mw / MW_O2)
            X_O2 = np.clip(X_O2, 0.0, 1.0)
            
            # Oxygen partial pressure
            p_O2 = X_O2 * pressure
            p_O2 = max(p_O2, 1.0)  # Minimum to avoid numerical issues
            
            # Kinetic-limited rate using reference mass flux
            # m''_ox,kin = j_ref * exp(-Ea/R * (1/T_s - 1/T_ref)) * (p_O2/P_ref)^n
            theta = np.exp(-Ea / R_GAS * (1.0 / T_s - 1.0 / T_ref))
            m_dot_ox_kin = j_ref * theta * (p_O2 / P_ref) ** pressure_exponent
            m_dot_ox_kin = max(m_dot_ox_kin, 0.0)
            
            # Diffusion-limited rate (molar basis)
            # Use film temperature for transport properties to keep Re, Sc, Sh, k_m, C_tot consistent
            T_film = 0.5 * (gas_temperature + T_s)
            
            # Estimate oxygen diffusivity: D_O2 ~ 1e-4 m²/s at high T, scales with T^1.5/P
            D_O2 = 1e-4 * (T_film / 1500.0) ** 1.5 * (1e6 / pressure)  # m²/s
            
            # Schmidt number: Sc = mu / (rho * D)
            Sc = gas_viscosity / (gas_density * D_O2)
            Sc = max(Sc, 0.1)  # Reasonable bounds
            
            # Sherwood number (Chilton-Colburn)
            Sh = 0.023 * (Re ** 0.8) * (Sc ** (1.0/3.0))
            Sh = max(Sh, 2.0)  # Minimum for laminar flow
            
            # Molar mass transfer coefficient [m/s]
            k_m_molar = Sh * D_O2 / characteristic_length
            
            # Total molar concentration [mol/m³] - use film temperature
            C_tot = pressure / (R_GAS * T_film)
            C_tot = max(C_tot, 1.0)  # Minimum
            
            # Surface oxygen mole fraction (assume zero at surface due to reaction)
            X_O2_s = 0.0
            
            # Molar flux of O2 [mol/(m²·s)] - use driving force (X_O2 - X_O2_s)
            N_O2 = k_m_molar * (X_O2 - X_O2_s) * C_tot
            N_O2 = max(N_O2, 0.0)
            
            # Convert to carbon mass flux: m''_ox,diff = nu_C_per_O2 * MW_C * N_O2
            m_dot_ox_diff = stoichiometry_ratio * MW_C * N_O2  # kg/(m²·s)
            m_dot_ox_diff = max(m_dot_ox_diff, 0.0)
            
            # Oxidation rate is minimum of kinetic and diffusion limits
            m_dot_ox = min(m_dot_ox_kin, m_dot_ox_diff)
            m_dot_ox = max(m_dot_ox, 0.0)
        else:
            m_dot_ox = 0.0
        
        # 4. INITIALIZE FEEDBACK LOOP VARIABLES
        # The feedback loop couples: f_fb ↔ q_fb ↔ m_dot_th ↔ B_m ↔ f_fb
        # We need to iterate this until convergence within each T_s iteration
        f_fb = f_fb_min
        q_fb = 0.0
        m_dot_th = 0.0
        Da = 0.0
        B_m = 0.0
        
        # Calculate Damköhler number (depends only on T_s and oxidation, not on m_dot_th)
        if m_dot_ox > 0 and T_s > graphite_config.oxidation_temperature:
            # Surface reaction rate constant (molar, simplified)
            if m_dot_ox_kin > 0:
                # k_surf_molar ~ m_dot_ox_kin / (MW_C * C_tot * p_O2^n)
                k_surf_approx = m_dot_ox_kin / (MW_C * C_tot * (p_O2 ** pressure_exponent))
                k_surf_approx = max(k_surf_approx, 1e-10)
            else:
                k_surf_approx = 1e-10
            
            # Damköhler number: Da = (k_surf * p_O2^n) / (k_m_molar * C_O2,inf)
            # Compare surface reaction rate to mass transfer rate
            if k_m_molar > 0 and p_O2 > 0:
                # Surface reaction rate: k_surf * C_O2,s^n (C_O2,s ~ p_O2^n for surface)
                # Mass transfer rate: k_m_molar * C_O2,inf
                C_O2_inf = X_O2 * C_tot  # Molar concentration of O2 in free stream
                C_O2_surf = (p_O2 / pressure) * C_tot  # Surface O2 concentration (simplified)
                Da = (k_surf_approx * (C_O2_surf ** pressure_exponent)) / (k_m_molar * C_O2_inf)
                Da = max(Da, 1e-6)
            else:
                Da = 0.0
        
        # 5. CONDUCTION INTO SOLID (depends only on T_s)
        q_cond = k_s * (T_s - T_back) / max(effective_thickness, 0.001)
        
        # 6. ITERATE FEEDBACK LOOP: f_fb ↔ q_fb ↔ m_dot_th ↔ B_m
        # This inner loop converges the coupling between feedback and thermal ablation
        is_ablating = False
        H_star_th = graphite_config.heat_of_ablation
        
        for fb_iter in range(feedback_max_iter):
            f_fb_old = f_fb
            m_dot_th_old = m_dot_th
            
            # Calculate feedback fraction from current B_m
            if m_dot_ox > 0 and T_s > graphite_config.oxidation_temperature and Da > 0:
                # Blowing parameter: B_m = m''_tot / (rho_g * v_tau)
                v_tau = gas_velocity * np.sqrt(Cf / 2.0)
                v_tau = max(v_tau, 1.0)
                m_dot_tot = m_dot_ox + m_dot_th
                B_m = m_dot_tot / (gas_density * v_tau)
                B_m = max(B_m, 0.0)
                
                # Feedback fraction
                f_fb = f_fb_min + (f_fb_max - f_fb_min) * (Da / (1.0 + Da)) * (1.0 / (1.0 + B_m))
                f_fb = float(np.clip(f_fb, f_fb_min, f_fb_max))
            else:
                f_fb = 0.0
                B_m = 0.0
            
            # Calculate feedback heat flux
            q_fb = f_fb * m_dot_ox * oxidation_enthalpy
            
            # ENERGY BALANCE: q''_in + q''_fb - q''_rad = q''_cond + m''_th * H*_th
            q_net_available = q_in + q_fb - q_rad - q_cond
            
            # THERMAL ABLATION LOGIC
            if q_net_available > 0 and T_s >= T_abl:
                # Pin surface temperature at ablation temperature
                if not is_ablating:
                    is_ablating = True
                    T_s = T_abl
                    # Recalculate heat fluxes at pinned temperature
                    q_in = heat_transfer_coefficient * (gas_temperature - T_s)
                    q_rad = emissivity * SIGMA * (T_s**4 - T_env**4)
                    q_rad = max(q_rad, 0.0)
                    q_cond = k_s * (T_s - T_back) / max(effective_thickness, 0.001)
                    # Recalculate oxidation at new T_s (important!)
                    # This requires recalculating oxidation kinetics
                    if T_s > graphite_config.oxidation_temperature:
                        # Recalculate oxidation with new T_s
                        theta = np.exp(-Ea / R_GAS * (1.0 / T_s - 1.0 / T_ref))
                        m_dot_ox_kin_new = j_ref * theta * (p_O2 / P_ref) ** pressure_exponent
                        m_dot_ox_kin_new = max(m_dot_ox_kin_new, 0.0)
                        # Diffusion-limited rate uses film temperature (update if needed)
                        T_film = 0.5 * (gas_temperature + T_s)
                        D_O2 = 1e-4 * (T_film / 1500.0) ** 1.5 * (1e6 / pressure)
                        Sc = gas_viscosity / (gas_density * D_O2)
                        Sc = max(Sc, 0.1)
                        Sh = 0.023 * (Re ** 0.8) * (Sc ** (1.0/3.0))
                        Sh = max(Sh, 2.0)
                        k_m_molar = Sh * D_O2 / characteristic_length
                        C_tot = pressure / (R_GAS * T_film)
                        C_tot = max(C_tot, 1.0)
                        N_O2 = k_m_molar * (X_O2 - 0.0) * C_tot
                        N_O2 = max(N_O2, 0.0)
                        m_dot_ox_diff_new = stoichiometry_ratio * MW_C * N_O2
                        m_dot_ox_diff_new = max(m_dot_ox_diff_new, 0.0)
                        m_dot_ox = min(m_dot_ox_kin_new, m_dot_ox_diff_new)
                        m_dot_ox = max(m_dot_ox, 0.0)
                        # Update Da with new oxidation rate
                        if m_dot_ox > 0:
                            k_surf_approx = m_dot_ox / (MW_C * C_tot * (p_O2 ** pressure_exponent))
                            k_surf_approx = max(k_surf_approx, 1e-10)
                            C_O2_inf = X_O2 * C_tot
                            C_O2_surf = (p_O2 / pressure) * C_tot
                            Da = (k_surf_approx * (C_O2_surf ** pressure_exponent)) / (k_m_molar * C_O2_inf)
                            Da = max(Da, 1e-6)
                        else:
                            Da = 0.0
                        # Recalculate q_fb with new m_dot_ox
                        q_fb = f_fb * m_dot_ox * oxidation_enthalpy
                        q_net_available = q_in + q_fb - q_rad - q_cond
                
                # Effective heat of sublimation (latent + sensible)
                delta_T = max(T_s - 300.0, 0.0)
                H_star_th = graphite_config.heat_of_ablation + cp_s * delta_T
                H_star_th = max(H_star_th, 1e6)
                
                # Solve for thermal ablation mass flux from energy balance
                if q_net_available > 0:
                    m_dot_th = q_net_available / H_star_th
                    m_dot_th = max(m_dot_th, 0.0)
                else:
                    m_dot_th = 0.0
            elif T_s > 2800.0 and q_net_available > 0:
                # Transition region: allow some thermal ablation but still iterate on T_s
                delta_T = max(T_s - 300.0, 0.0)
                H_star_th = graphite_config.heat_of_ablation + cp_s * delta_T
                H_star_th = max(H_star_th, 1e6)
                m_dot_th = q_net_available / H_star_th
                m_dot_th = max(m_dot_th, 0.0)
            else:
                m_dot_th = 0.0
            
            # Check convergence of feedback loop
            if fb_iter > 0:
                f_fb_change = abs(f_fb - f_fb_old) / max(abs(f_fb_old), f_fb_min, 1e-10)
                m_dot_th_change = abs(m_dot_th - m_dot_th_old) / max(abs(m_dot_th_old), 1e-10)
                if f_fb_change < feedback_tol and m_dot_th_change < feedback_tol:
                    break
        
        # 9. ENERGY BALANCE RESIDUAL
        # If ablating, residual should be zero (T_s is pinned, m_dot_th balances energy)
        if is_ablating:
            residual = 0.0
        else:
            residual = q_in + q_fb - q_rad - q_cond - m_dot_th * H_star_th
        
        # 10. NEWTON-RAPHSON UPDATE FOR T_s
        # Skip Newton update if ablating (T_s is pinned)
        if iter > 0 and not is_ablating:
            # Derivatives for Newton step
            dq_in_dT = -heat_transfer_coefficient  # d/dT_s [h_g * (T_g - T_s)]
            dq_rad_dT = 4.0 * emissivity * SIGMA * T_s**3
            dq_cond_dT = k_s / max(effective_thickness, 0.001)
            
            # Derivative of oxidation feedback (simplified - assume f_fb and m_dot_ox change slowly)
            dq_fb_dT = 0.0
            if m_dot_ox > 0:
                # Simplified: d(m_dot_ox)/dT_s ~ m_dot_ox * (Ea / (R_GAS * T_s^2))
                dm_dot_ox_dT = m_dot_ox * (Ea / (R_GAS * T_s**2)) * 0.1  # Small factor for stability
                dq_fb_dT = f_fb * oxidation_enthalpy * dm_dot_ox_dT
            
            # Derivative of thermal ablation term
            dm_dot_th_dT = 0.0
            if m_dot_th > 0:
                # d(m_dot_th * H_star_th)/dT_s = m_dot_th * cp_s + (dm_dot_th/dT_s) * H_star_th
                # For stability, approximate dm_dot_th/dT_s as small
                dm_dot_th_dT = m_dot_th * cp_s * 0.1  # Small factor for stability
            
            # Total derivative
            dresidual_dT = dq_in_dT + dq_fb_dT - dq_rad_dT - dq_cond_dT - dm_dot_th_dT
            
            # Newton step with damping for stability
            if abs(dresidual_dT) > 1e-6:
                T_s = T_s - damp * residual / dresidual_dT
            else:
                # Fallback: simple bisection
                if residual > 0:
                    T_s = T_s + 10.0
                else:
                    T_s = T_s - 10.0
        
        # Bound surface temperature
        T_s = np.clip(T_s, 300.0, graphite_config.surface_temperature_limit)
        
        # Check convergence
        if abs(T_s - T_s_old) < tol:
            break
    
    # Final calculations with converged T_s
    # Recalculate with final T_s for consistency
    q_in = heat_transfer_coefficient * (gas_temperature - T_s)
    q_rad = emissivity * SIGMA * (T_s**4 - T_env**4)
    q_rad = max(q_rad, 0.0)
    
    # Recalculate oxidation with final T_s
    m_dot_ox = 0.0
    m_dot_ox_kin = 0.0
    m_dot_ox_diff = 0.0
    X_O2 = 0.0
    p_O2 = 0.0
    k_m_molar = 0.0
    C_tot = 0.0
    
    if T_s > graphite_config.oxidation_temperature:
        X_O2 = oxygen_mass_fraction * (mixture_mw / MW_O2)
        X_O2 = np.clip(X_O2, 0.0, 1.0)
        p_O2 = X_O2 * pressure
        p_O2 = max(p_O2, 1.0)
        
        theta = np.exp(-Ea / R_GAS * (1.0 / T_s - 1.0 / T_ref))
        m_dot_ox_kin = j_ref * theta * (p_O2 / P_ref) ** pressure_exponent
        m_dot_ox_kin = max(m_dot_ox_kin, 0.0)
        
        # Use film temperature for transport properties
        T_film = 0.5 * (gas_temperature + T_s)
        D_O2 = 1e-4 * (T_film / 1500.0) ** 1.5 * (1e6 / pressure)
        Sc = gas_viscosity / (gas_density * D_O2)
        Sc = max(Sc, 0.1)
        Sh = 0.023 * (Re ** 0.8) * (Sc ** (1.0/3.0))
        Sh = max(Sh, 2.0)
        k_m_molar = Sh * D_O2 / characteristic_length
        C_tot = pressure / (R_GAS * T_film)
        C_tot = max(C_tot, 1.0)
        
        # Surface oxygen mole fraction (assume zero at surface due to reaction)
        X_O2_s = 0.0
        
        # Use same driving force (X_O2 - X_O2_s) as in the loop
        N_O2 = k_m_molar * (X_O2 - X_O2_s) * C_tot
        N_O2 = max(N_O2, 0.0)
        m_dot_ox_diff = stoichiometry_ratio * MW_C * N_O2
        m_dot_ox_diff = max(m_dot_ox_diff, 0.0)
        m_dot_ox = min(m_dot_ox_kin, m_dot_ox_diff)
        m_dot_ox = max(m_dot_ox, 0.0)
    
    # Recalculate feedback fraction and thermal ablation with converged T_s
    # Use same feedback loop logic for consistency
    f_fb = f_fb_min
    q_fb = 0.0
    m_dot_th = 0.0
    Da = 0.0
    B_m = 0.0
    q_cond = k_s * (T_s - T_back) / max(effective_thickness, 0.001)
    
    # Calculate Da with final oxidation rate
    if m_dot_ox > 0 and T_s > graphite_config.oxidation_temperature:
        if m_dot_ox_kin > 0:
            k_surf_approx = m_dot_ox_kin / (MW_C * C_tot * (p_O2 ** pressure_exponent))
            k_surf_approx = max(k_surf_approx, 1e-10)
        else:
            k_surf_approx = 1e-10
        
        if k_m_molar > 0 and p_O2 > 0:
            C_O2_inf = X_O2 * C_tot
            C_O2_surf = (p_O2 / pressure) * C_tot
            Da = (k_surf_approx * (C_O2_surf ** pressure_exponent)) / (k_m_molar * C_O2_inf)
            Da = max(Da, 1e-6)
        else:
            Da = 0.0
    
    # Final feedback loop iteration (should converge quickly since T_s is converged)
    T_abl = getattr(graphite_config, "ablation_surface_temperature", 3000.0) or 3000.0
    is_ablating_final = (T_s >= T_abl)
    H_star_th = graphite_config.heat_of_ablation
    
    for fb_iter in range(feedback_max_iter):
        f_fb_old = f_fb
        m_dot_th_old = m_dot_th
        
        # Calculate feedback fraction from current B_m
        if m_dot_ox > 0 and T_s > graphite_config.oxidation_temperature and Da > 0:
            v_tau = gas_velocity * np.sqrt(Cf / 2.0)
            v_tau = max(v_tau, 1.0)
            m_dot_tot = m_dot_ox + m_dot_th
            B_m = m_dot_tot / (gas_density * v_tau)
            B_m = max(B_m, 0.0)
            f_fb = f_fb_min + (f_fb_max - f_fb_min) * (Da / (1.0 + Da)) * (1.0 / (1.0 + B_m))
            f_fb = float(np.clip(f_fb, f_fb_min, f_fb_max))
        else:
            f_fb = 0.0
            B_m = 0.0
        
        q_fb = f_fb * m_dot_ox * oxidation_enthalpy
        q_net_available = q_in + q_fb - q_rad - q_cond
        
        # Thermal ablation
        if q_net_available > 0 and T_s >= T_abl:
            delta_T = max(T_s - 300.0, 0.0)
            H_star_th = graphite_config.heat_of_ablation + cp_s * delta_T
            H_star_th = max(H_star_th, 1e6)
            m_dot_th = q_net_available / H_star_th
            m_dot_th = max(m_dot_th, 0.0)
        elif T_s > 2800.0 and q_net_available > 0:
            delta_T = max(T_s - 300.0, 0.0)
            H_star_th = graphite_config.heat_of_ablation + cp_s * delta_T
            H_star_th = max(H_star_th, 1e6)
            m_dot_th = q_net_available / H_star_th
            m_dot_th = max(m_dot_th, 0.0)
        else:
            m_dot_th = 0.0
        
        # Check convergence
        if fb_iter > 0:
            f_fb_change = abs(f_fb - f_fb_old) / max(abs(f_fb_old), f_fb_min, 1e-10)
            m_dot_th_change = abs(m_dot_th - m_dot_th_old) / max(abs(m_dot_th_old), 1e-10)
            if f_fb_change < feedback_tol and m_dot_th_change < feedback_tol:
                break
    
    # TOTAL RECESSION RATE
    recession_rate_ox = m_dot_ox / rho_s
    recession_rate_th = m_dot_th / rho_s
    recession_rate_total = recession_rate_ox + recession_rate_th
    
    # Total mass flux
    mass_flux_total = m_dot_ox + m_dot_th
    
    # Heat removed - ONLY count feedback fraction and thermal ablation
    # Do NOT count full oxidation enthalpy as "heat removed from solid"
    heat_removed_ablation = (q_fb + m_dot_th * graphite_config.heat_of_ablation) * throat_area * graphite_config.coverage_fraction
    heat_removed_conduction = q_cond * throat_area * graphite_config.coverage_fraction
    heat_removed_total = heat_removed_ablation + heat_removed_conduction
    
    return {
        "enabled": True,
        "recession_rate": float(recession_rate_total),
        "mass_flux": float(mass_flux_total),
        "surface_temperature": float(T_s),
        "effective_heat_flux": float(q_net_available),
        "radiative_relief": float(q_rad),
        "conduction_loss": float(q_cond),
        "heat_removed": float(heat_removed_total),
        "oxidation_rate": float(recession_rate_ox),
        "oxidation_mass_flux": float(m_dot_ox),
        "thermal_mass_flux": float(m_dot_th),
        "recession_rate_thermal": float(recession_rate_th),
        "mass_flux_thermal": float(m_dot_th),
        "coverage_area": float(throat_area * graphite_config.coverage_fraction),
        "feedback_fraction": float(f_fb),
        "q_feedback": float(q_fb),
        "q_radiation": float(q_rad),
        "q_conduction": float(q_cond),
        "q_convective": float(q_in),
        "damkohler_number": float(Da),
        "blowing_parameter": float(B_m),
    }
