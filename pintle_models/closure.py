"""Closure logic: solve branch flows with spray constraints"""

import numpy as np
from typing import Tuple, Dict, Any
from pintle_pipeline.config_schemas import (
    PintleEngineConfig,
    PintleGeometryConfig,
    DischargeConfig,
    SprayConfig,
    FeedSystemConfig,
)
from pintle_pipeline.feed_loss import delta_p_feed
from pintle_pipeline.regen_cooling import delta_p_regen_channels
from pintle_models.geometry import get_effective_areas, get_hydraulic_diameters
from pintle_models.discharge import cd_from_re, calculate_reynolds_number
from pintle_models.spray import (
    momentum_flux_ratio,
    thrust_momentum_ratio,
    spray_angle_from_J,
    spray_angle_from_TMR,
    weber_number,
    ohnesorge_number,
    smd_lefebvre,
    tau_evap,
    xstar,
    check_spray_constraints,
)


def flows(
    P_tank_O: float,
    P_tank_F: float,
    Pc: float,
    config: PintleEngineConfig
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Solve for mass flow rates through pintle injector with spray constraints.
    
    This is the closure logic that iteratively solves for ṁ_O and ṁ_F
    while satisfying both hydraulic flow laws and spray physics constraints.
    
    Flow path:
    1. Tank pressures (P_tank_O, P_tank_F) - INPUT
    2. Feed system losses → P_injector = P_tank - Δp_feed
    3. Injector flow → mdot from P_injector - Pc
    4. Spray constraints → adjust if needed
    
    Parameters:
    -----------
    P_tank_O : float
        Oxidizer TANK pressure [Pa] - INPUT
    P_tank_F : float
        Fuel TANK pressure [Pa] - INPUT
    Pc : float
        Chamber pressure [Pa] - SOLVED (this is a guess during root finding)
    config : PintleEngineConfig
        Complete engine configuration
    
    Returns:
    --------
    mdot_O : float [kg/s]
        Oxidizer mass flow rate
    mdot_F : float [kg/s]
        Fuel mass flow rate
    diagnostics : dict
        Diagnostic information (J, θ, D32, x*, constraint violations, etc.)
    """
    # Extract configurations
    geom = config.pintle_geometry
    discharge_O = config.discharge["oxidizer"]
    discharge_F = config.discharge["fuel"]
    feed_O = config.feed_system["oxidizer"]
    feed_F = config.feed_system["fuel"]
    spray_cfg = config.spray
    fluids = config.fluids
    
    # Get fluid properties
    rho_O = fluids["oxidizer"].density
    mu_O = fluids["oxidizer"].viscosity
    sigma_O = fluids["oxidizer"].surface_tension
    
    rho_F = fluids["fuel"].density
    mu_F = fluids["fuel"].viscosity
    sigma_F = fluids["fuel"].surface_tension
    
    # Get geometry
    A_LOX, A_fuel = get_effective_areas(geom)
    d_hyd_O, d_hyd_F = get_hydraulic_diameters(geom)
    
    # Initial guess for mass flow rates (from simple orifice flow)
    # This will be refined iteratively
    mdot_O_guess = 0.1  # kg/s
    mdot_F_guess = 0.1  # kg/s
    
    # Closure iteration loop
    max_iter = config.solver.closure.max_iterations
    Cd_reduction = config.solver.closure.Cd_reduction_factor
    
    # Effective discharge coefficients (may be reduced if constraints violated)
    Cd_O_eff = discharge_O.Cd_inf
    Cd_F_eff = discharge_F.Cd_inf
    
    diagnostics = {
        "iterations": 0,
        "constraints_satisfied": False,
        "violations": [],
        "J": None,
        "TMR": None,
        "theta": None,
        "We_O": None,
        "We_F": None,
        "D32_O": None,
        "D32_F": None,
        "x_star": None,
    }
    
    for iteration in range(max_iter):
        # Step 1: Calculate feed losses (requires mdot guess)
        # For first iteration, use guess; for subsequent, use previous result
        if iteration == 0:
            mdot_O = mdot_O_guess
            mdot_F = mdot_F_guess
        
        # Step 2: Calculate feed losses from tank to injector
        # Feed losses depend on mdot, so we iterate
        # Use simple fixed-point iteration
        for feed_iter in range(3):  # 2-3 iterations usually sufficient
            # Calculate feed system pressure losses
            delta_p_feed_O = delta_p_feed(mdot_O, rho_O, feed_O, P_tank_O)
            
            # For fuel: add regenerative cooling pressure drop if enabled
            delta_p_feed_F_base = delta_p_feed(mdot_F, rho_F, feed_F, P_tank_F)
            if config.regen_cooling is not None and config.regen_cooling.enabled:
                delta_p_regen = delta_p_regen_channels(
                    mdot_F, rho_F, mu_F, config.regen_cooling, P_tank_F
                )
                delta_p_feed_F = delta_p_feed_F_base + delta_p_regen
            else:
                delta_p_feed_F = delta_p_feed_F_base
            
            # Injector inlet pressures (after feed losses)
            P_inj_O = P_tank_O - delta_p_feed_O
            P_inj_F = P_tank_F - delta_p_feed_F
            
            # Recalculate mdot with new injector pressures (quick update)
            if feed_iter < 2:  # Don't update on last iteration
                delta_p_inj_O = max(0, P_inj_O - Pc)
                delta_p_inj_F = max(0, P_inj_F - Pc)
                
                # Quick mdot update (without full spray calculation)
                u_O_quick = np.sqrt(2 * delta_p_inj_O / rho_O) if delta_p_inj_O > 0 else 0
                u_F_quick = np.sqrt(2 * delta_p_inj_F / rho_F) if delta_p_inj_F > 0 else 0
                
                Re_O_quick = calculate_reynolds_number(rho_O, u_O_quick, d_hyd_O, mu_O)
                Re_F_quick = calculate_reynolds_number(rho_F, u_F_quick, d_hyd_F, mu_F)
                
                # Quick Cd update with Reynolds coupling and inlet conditions
                # Pass inlet pressures for pressure-dependent Cd (compressibility effects)
                # Temperature: use tank temperature (assume isothermal for now, could enhance with heat transfer)
                T_tank_O = 90.0  # K - LOX tank temperature (typical)
                T_tank_F = 300.0  # K - RP-1 tank temperature (typical)
                Cd_O_quick_base = cd_from_re(Re_O_quick, discharge_O, P_inlet=P_inj_O, T_inlet=T_tank_O)
                Cd_F_quick_base = cd_from_re(Re_F_quick, discharge_F, P_inlet=P_inj_F, T_inlet=T_tank_F)
                Cd_O_quick = min(Cd_O_quick_base, Cd_O_eff)
                Cd_F_quick = min(Cd_F_quick_base, Cd_F_eff)
                
                mdot_O = Cd_O_quick * A_LOX * np.sqrt(2 * rho_O * delta_p_inj_O)
                mdot_F = Cd_F_quick * A_fuel * np.sqrt(2 * rho_F * delta_p_inj_F)
        
        # Step 3: Calculate injector-available pressure drop
        delta_p_inj_O = max(0, P_inj_O - Pc)
        delta_p_inj_F = max(0, P_inj_F - Pc)
        
        # CRITICAL: If injector pressure is below chamber pressure, flow is impossible
        # This should cause the solver to fail or find a different solution
        if P_inj_F < Pc:
            # Fuel cannot flow - this is a physical constraint violation
            # Set fuel flow to zero and let solver handle it
            mdot_F = 0.0
            # Continue with zero fuel flow - solver will need to adjust Pc or fail
        if P_inj_O < Pc:
            # Oxidizer cannot flow - this is a physical constraint violation
            mdot_O = 0.0
        
        # Step 4: Calculate velocities (initial guess)
        u_O = np.sqrt(2 * delta_p_inj_O / rho_O) if delta_p_inj_O > 0 else 0
        u_F = np.sqrt(2 * delta_p_inj_F / rho_F) if delta_p_inj_F > 0 else 0
        
        # Step 5: Calculate Reynolds numbers (recalculate with actual velocities)
        Re_O = calculate_reynolds_number(rho_O, u_O, d_hyd_O, mu_O)
        Re_F = calculate_reynolds_number(rho_F, u_F, d_hyd_F, mu_F)
        
        # Step 6: Calculate discharge coefficients with Reynolds coupling and inlet conditions
        # Enhanced Cd model: Cd(Re, P_inlet, T_inlet)
        # - Re dependence: viscous effects
        # - P_inlet dependence: compressibility effects (for LOX at high pressure)
        # - T_inlet dependence: viscosity changes with temperature
        T_tank_O = 90.0  # K - LOX tank temperature (typical cryogenic)
        T_tank_F = 300.0  # K - RP-1 tank temperature (typical ambient)
        Cd_O_base = cd_from_re(Re_O, discharge_O, P_inlet=P_inj_O, T_inlet=T_tank_O)
        Cd_F_base = cd_from_re(Re_F, discharge_F, P_inlet=P_inj_F, T_inlet=T_tank_F)
        
        # Apply spray constraint reduction (use the more restrictive)
        # Cd_eff accounts for spray constraints, Cd_base accounts for Re
        Cd_O = min(Cd_O_base, Cd_O_eff)
        Cd_F = min(Cd_F_base, Cd_F_eff)
        
        # Step 7: Calculate hydraulic mass flow rates
        # Only calculate if pressure drop is positive (flow is possible)
        if delta_p_inj_O > 0:
            mdot_O = Cd_O * A_LOX * np.sqrt(2 * rho_O * delta_p_inj_O)
        else:
            mdot_O = 0.0  # No flow if injector pressure <= chamber pressure
        
        if delta_p_inj_F > 0:
            mdot_F = Cd_F * A_fuel * np.sqrt(2 * rho_F * delta_p_inj_F)
        else:
            mdot_F = 0.0  # No flow if injector pressure <= chamber pressure
        
        # Step 8: Recalculate velocities with actual mdot
        u_O = mdot_O / (rho_O * A_LOX) if A_LOX > 0 else 0
        u_F = mdot_F / (rho_F * A_fuel) if A_fuel > 0 else 0
        
        # Step 9: Calculate spray parameters
        J = momentum_flux_ratio(rho_O, u_O, rho_F, u_F)
        MR = mdot_O / mdot_F if mdot_F > 0 else np.inf
        TMR = thrust_momentum_ratio(J, MR)
        
        # Spray angle
        if spray_cfg.spray_angle.model == "J":
            theta = spray_angle_from_J(J, spray_cfg.spray_angle.k, spray_cfg.spray_angle.n)
        else:  # TMR
            theta = spray_angle_from_TMR(TMR)
        
        # Weber numbers
        We_O = weber_number(rho_O, u_O, geom.lox.d_orifice, sigma_O)
        We_F = weber_number(rho_F, u_F, d_hyd_F, sigma_F)
        
        # Sauter Mean Diameter
        Oh_O = ohnesorge_number(mu_O, rho_O, sigma_O, geom.lox.d_orifice)
        Oh_F = ohnesorge_number(mu_F, rho_F, sigma_F, d_hyd_F)
        
        D32_O = smd_lefebvre(
            geom.lox.d_orifice,
            We_O,
            Oh_O,
            spray_cfg.smd.C,
            spray_cfg.smd.m,
            spray_cfg.smd.p
        )
        D32_F = smd_lefebvre(
            d_hyd_F,
            We_F,
            Oh_F,
            spray_cfg.smd.C,
            spray_cfg.smd.m,
            spray_cfg.smd.p
        )
        
        # Evaporation length
        U_rel = np.sqrt(u_O**2 + u_F**2)  # Relative velocity magnitude
        tau_evap_O = tau_evap(D32_O, spray_cfg.evaporation.K)
        x_star_O = xstar(U_rel, tau_evap_O)
        
        # Use worst-case (longest) evaporation length
        x_star = max(x_star_O, xstar(U_rel, tau_evap(D32_F, spray_cfg.evaporation.K)))
        
        # Step 10: Check spray constraints
        constraints_ok, violations = check_spray_constraints(We_O, We_F, x_star, spray_cfg)
        
        # Update diagnostics
        diagnostics.update({
            "iterations": iteration + 1,
            "constraints_satisfied": constraints_ok,
            "violations": violations,
            "J": J,
            "TMR": TMR,
            "theta": theta,
            "We_O": We_O,
            "We_F": We_F,
            "D32_O": D32_O,
            "D32_F": D32_F,
            "x_star": x_star,
        })
        
        # Step 11: If constraints satisfied, we're done
        if constraints_ok:
            break
        
        # Step 12: If constraints violated, reduce effective Cd and iterate
        Cd_O_eff *= Cd_reduction
        Cd_F_eff *= Cd_reduction
        
        # Prevent Cd from going too low
        Cd_O_eff = max(Cd_O_eff, discharge_O.Cd_min)
        Cd_F_eff = max(Cd_F_eff, discharge_F.Cd_min)
    
    return mdot_O, mdot_F, diagnostics



