"""Nozzle model: thrust coefficient and thrust calculation with shifting equilibrium"""

import numpy as np
from typing import Dict, Optional, Any
from engine.pipeline.cea_cache import CEACache
from engine.pipeline.numerical_robustness import (
    PhysicalConstraints,
    NumericalStability,
    PhysicsValidator,
)
from engine.core.mach_solver import solve_exit_mach_robust


def calculate_chamber_temperature_profile(
    Tc: float,
    Lstar: float,
    reaction_progress: Optional[Dict] = None,
    n_points: int = 10,
) -> Dict[str, Any]:
    """
    Calculate temperature profile along chamber length.
    
    Temperature increases as reaction progresses from injection to throat.
    Uses reaction progress to model temperature rise.
    
    Parameters:
    -----------
    Tc : float
        Chamber temperature at throat (equilibrium) [K]
    Lstar : float
        Characteristic length [m]
    reaction_progress : dict, optional
        Reaction progress dict with progress_injection, progress_mid, progress_throat
    n_points : int
        Number of points along chamber (default: 10)
    
    Returns:
    --------
    profile : dict
        - positions: Array of positions along chamber [m] (0 = injection, Lstar = throat)
        - temperatures: Array of temperatures [K]
        - progress: Array of reaction progress (0-1)
        - T_injection: Temperature at injection plane [K]
        - T_mid: Temperature at mid-chamber [K]
        - T_throat: Temperature at throat [K]
    """
    # Default reaction progress if not provided
    if reaction_progress is None:
        progress_injection = 0.0
        progress_mid = 0.5
        progress_throat = 1.0
    else:
        progress_injection = reaction_progress.get("progress_injection", 0.0)
        progress_mid = reaction_progress.get("progress_mid", 0.5)
        progress_throat = reaction_progress.get("progress_throat", 1.0)
    
    # Initial temperature at injection (before significant reaction)
    # Assume reactants enter at ~300-500 K (propellant temperature)
    T_injection_guess = 400.0  # K, typical propellant injection temperature
    
    # Temperature at throat is equilibrium temperature (Tc)
    T_throat = Tc
    
    # Interpolate temperature based on reaction progress
    # T = T_injection + progress * (T_throat - T_injection)
    # But account for heat release: more progress = more heat = higher temp
    
    # Create position array
    positions = np.linspace(0.0, Lstar, n_points)
    
    # Reaction progress along chamber (linear interpolation)
    progress_array = np.linspace(progress_injection, progress_throat, n_points)
    
    # Temperature profile
    # Simple model: T = T_injection + progress * (T_throat - T_injection)
    # More accurate: account for heat release rate
    temperatures = T_injection_guess + progress_array * (T_throat - T_injection_guess)
    
    # Refine: temperature rise is not linear with progress due to heat release
    # Use a power law to account for rapid initial heating
    # T = T_injection + (T_throat - T_injection) * progress^alpha
    # where alpha < 1 means faster initial heating
    alpha = 0.7  # Empirical: faster initial heating
    temperatures = T_injection_guess + (T_throat - T_injection_guess) * (progress_array ** alpha)
    
    # Mid-chamber temperature
    T_mid = T_injection_guess + (T_throat - T_injection_guess) * (progress_mid ** alpha)
    
    # Validate all temperatures
    temperatures = np.clip(temperatures, 200.0, 5000.0)  # Physical bounds
    T_injection = float(temperatures[0])
    T_mid = float(np.clip(T_mid, 200.0, 5000.0))
    
    return {
        "positions": positions.tolist(),
        "temperatures": temperatures.tolist(),
        "progress": progress_array.tolist(),
        "T_injection": T_injection,
        "T_mid": T_mid,
        "T_throat": T_throat,
    }


def calculate_thrust(
    Pc: float,
    MR: float,
    mdot_total: float,
    cea_cache: CEACache,
    config: Any,
    Pa: float = 101325.0,
    reaction_progress: Optional[Dict] = None,
    use_shifting_equilibrium: bool = True,
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
    config : PintleEngineConfig
        Complete engine configuration
    Pa : float
        Ambient pressure [Pa] (default: sea level)
    reaction_progress : dict, optional
        Reaction progress for shifting equilibrium
    use_shifting_equilibrium : bool
        Enable shifting equilibrium model
    
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
    # Extract geometry from config.chamber_geometry
    cg = config.chamber_geometry
    if cg is None:
        raise ValueError("config.chamber_geometry must be provided")

    A_throat = cg.A_throat
    A_exit = cg.A_exit
    eps = cg.expansion_ratio
    efficiency = cg.nozzle_efficiency

    # Validate geometry inputs
    if A_throat is None or A_throat <= 0:
        raise ValueError(f"Invalid A_throat: {A_throat}")
    if A_exit is None or A_exit <= 0:
        raise ValueError(f"Invalid A_exit: {A_exit}")
    if eps is None or eps <= 1.0:
        raise ValueError(f"Invalid expansion ratio: eps={eps}")

    # Verify geometric consistency: eps = A_exit / A_throat
    eps_calc = A_exit / A_throat
    if not np.isclose(eps, eps_calc, rtol=1e-4):
        raise ValueError(
            f"Geometric inconsistency: config.chamber_geometry.expansion_ratio ({eps:.6f}) "
            f"does not match A_exit / A_throat ({eps_calc:.6f})"
        )

    # Get CEA properties (now with eps parameter for 3D cache)
    cea_props = cea_cache.eval(MR, Pc, Pa, eps)
    Cf_ideal = cea_props["Cf_ideal"]
    gamma = cea_props["gamma"]
    Tc = cea_props["Tc"]
    R = cea_props["R"]
    


    
    # Validate inputs
    gamma_val = float(gamma)
    eps_val = float(eps)
    Pc_val = float(Pc)

    print(
        f"[NOZZLE][CEA] Pc={Pc_val:.3e} Pa, Pa={Pa:.3e} Pa, MR={MR:.4f}, eps={eps_val:.3f} | "
        f"Cf_ideal={Cf_ideal:.4f}, gamma={gamma_val:.4f}, Tc={Tc:.1f} K, R={R:.2f}"
    )


    # estimate what c*_actual is implied by Pc, At, mdot
    cstar_implied = (Pc_val * A_throat) / max(mdot_total, 1e-12)

    print(
        f"[NOZ-CSTAR] c*_implied_from_mdot={cstar_implied:.1f} m/s | "
        f"c*_ideal_from_CEA={cea_props['cstar_ideal']:.1f} m/s | ratio={cstar_implied/cea_props['cstar_ideal']:.4f}"
    )

    
    # Apply nozzle efficiency
    Cf = efficiency * Cf_ideal
    
    # Calculate exit pressure using isentropic relations
    
    # For supersonic nozzles (eps > 1), we need to solve the area-Mach relation:
    # A/A* = (1/M) × [(2/(gamma+1)) × (1 + (gamma-1)/2 × M²)]^((gamma+1)/(2(gamma-1)))
    # Then use isentropic relation: P/Pc = [1 + (gamma-1)/2 × M²]^(-gamma/(gamma-1))
    
    gamma_check = PhysicalConstraints.validate_gamma(gamma_val)
    if not gamma_check.passed and gamma_check.severity == "error":
        raise ValueError(f"Invalid gamma: {gamma_check.message}")
    
    # Initialize M_exit early to ensure it's always defined
    # CRITICAL: Must initialize to a valid supersonic value, not 0.0
    M_exit = 2.0  # Default supersonic value
    M_exit_calculated = False  # Track if M_exit was successfully calculated
    
    # CRITICAL: Ensure M_exit is always calculated - if conditions are invalid, use fallback
    if not (gamma_val > 1.0 and eps_val > 1.0):
        import warnings
        warnings.warn(f"Invalid nozzle conditions: gamma={gamma_val:.4f}, eps={eps_val:.4f}. Using fallback M_exit=2.0")
        M_exit = 2.0
        M_exit_calculated = True
    
    P_exit = Pa  # Initialize to ambient
    T_exit = Tc  # Initialize to chamber temp
    v_exit = 0.0  # Initialize
    
    if gamma_val > 1 and eps_val > 1:
        # Solve for exit Mach number from area ratio (supersonic solution)
        # Using consolidated solver from mach_solver module
        M_exit, M_converged = solve_exit_mach_robust(eps_val, gamma_val)
        M_exit_calculated = True
        
        # Use isentropic relation for exit pressure
        # P_exit/Pc = [1 + (gamma-1)/2 × M_exit²]^(-gamma/(gamma-1))
        pressure_exponent = -gamma_val / (gamma_val - 1.0)
        pressure_factor = (1.0 + (gamma_val - 1.0) / 2.0 * M_exit**2) ** pressure_exponent
        P_exit = Pc_val * pressure_factor
        
        # Calculate exit temperature from Mach number (isentropic, consistent with pressure)
        # T_exit/Tc = [1 + (gamma-1)/2 × M_exit²]^(-1) = 1 / [1 + (gamma-1)/2 × M_exit²]
        # This is the CORRECT isentropic relation, consistent with pressure calculation
        temperature_factor = 1.0 / (1.0 + (gamma_val - 1.0) / 2.0 * M_exit**2)
        T_exit = Tc * temperature_factor
        
        # Calculate exit velocity from Mach number (consistent with T_exit and P_exit)
        # v_exit = M_exit × sqrt(gamma × R × T_exit)
        # This is the CORRECT relation for isentropic flow
        # CRITICAL: Use chamber gamma and R for initial calculation, then update if shifting equilibrium
        sound_speed_exit_squared = gamma_val * R * T_exit
        sound_speed_exit, sound_valid = NumericalStability.safe_sqrt(sound_speed_exit_squared, "sound_speed_exit")
        if sound_valid.passed and np.isfinite(sound_speed_exit) and sound_speed_exit > 0 and M_exit > 0:
            v_exit = M_exit * sound_speed_exit
        else:
            # If sound speed calculation fails, use fallback or recalculate
            import warnings
            warnings.warn(f"[WARNING] Cannot calculate v_exit from M_exit in initial calculation.")
            v_exit = 0.0
        
        # Validate exit pressure
        if not np.isfinite(P_exit) or P_exit < 0:
            P_exit = Pa  # Fallback to ambient
        
        # Physics: P_exit can be < Pa (overexpanded) or > Pa (underexpanded)
        # Don't clamp - that's a physics result, not an error
        # Only ensure it's positive
        if P_exit < 0:
            raise ValueError(f"Non-physical exit pressure: P_exit={P_exit} Pa")
        
        # Validate exit temperature
        # Physics: T_exit must be positive and less than Tc (isentropic expansion)
        if not np.isfinite(T_exit) or T_exit <= 0 or T_exit >= Tc:
            raise ValueError(f"Non-physical exit temperature: T_exit={T_exit} K, Tc={Tc} K, M_exit={M_exit:.4f}")
    else:
        # Invalid conditions - use fallback values
        import warnings
        warnings.warn(f"Invalid nozzle conditions: gamma={gamma_val:.4f}, eps={eps_val:.4f}. Using fallback values.")
        # Calculate reasonable fallback M_exit from eps
        if eps_val > 1.0 and gamma_val > 1.0:
            # Use consolidated Mach solver
            M_exit, M_converged = solve_exit_mach_robust(eps_val, gamma_val)
            M_exit_calculated = True
        else:
            M_exit = 2.0  # Default supersonic value
            M_exit_calculated = True  # Mark as calculated
        P_exit = Pa
        T_exit = Tc * 0.5  # Conservative estimate
        # Calculate v_exit from M_exit
        sound_speed_exit_squared = gamma_val * R * T_exit
        sound_speed_exit, sound_valid = NumericalStability.safe_sqrt(sound_speed_exit_squared, "sound_speed_exit")
        if sound_valid.passed and np.isfinite(sound_speed_exit) and sound_speed_exit > 0:
            v_exit = M_exit * sound_speed_exit
        else:
            v_exit = 0.0
    
    print(
        f"[NOZZLE][ISO] M_exit={M_exit:.4f} | "
        f"T_exit={T_exit:.1f} K ({T_exit/Tc:.3f} Tc), "
        f"P_exit={P_exit:.3e} Pa ({P_exit/Pc_val:.4f} Pc)"
    )

    # Apply shifting equilibrium if enabled
    # PROPER ITERATIVE APPROACH: As gas expands, equilibrium composition shifts.
    # Gamma and R change between chamber (equilibrium) and exit (shifting).
    # Must iterate to find self-consistent solution: M_exit, P_exit, T_exit, gamma_exit, R_exit
    gamma_exit = gamma_val
    R_exit = R
    equilibrium_factor = 1.0
    
    if use_shifting_equilibrium and P_exit < Pc_val:
        try:
            from engine.pipeline.reaction_chemistry import calculate_shifting_equilibrium_properties
            
            # Get reaction progress at chamber (if provided)
            progress_chamber = 1.0  # Default: assume equilibrium at chamber
            if reaction_progress is not None:
                progress_chamber = reaction_progress.get("progress_throat", 1.0)
            
            # Reaction rate factor: Physics-based approach using Damköhler number
            # The equilibrium_factor returned by calculate_shifting_equilibrium_properties
            # is computed as Da/(1+Da) based on actual reaction and expansion time scales.
            # No longer using hardcoded 0.1 - let physics determine the value.
            #
            # Allow user override via config if available, otherwise use physics-based default
            reaction_rate_factor = None  # Will use default in function (physics-based)
            if config is not None and hasattr(config, 'nozzle'):
                if hasattr(config.nozzle, 'reaction_rate_factor'):
                    reaction_rate_factor = config.nozzle.reaction_rate_factor
            
            # ITERATIVE SHIFTING EQUILIBRIUM SOLUTION
            # Physics-based: iterate to find self-consistent M_exit, P_exit, T_exit, gamma_exit, R_exit
            # NO ARBITRARY CONSTANTS OR CLAMPING - all from physics equations
            gamma_exit_iter = gamma_val
            R_exit_iter = R
            M_exit_iter = M_exit
            P_exit_iter = P_exit
            T_exit_iter = T_exit
            
            # Use MR parameter passed to calculate_thrust (no hardcoded values)
            # MR is already available as a function parameter - use it directly
            MR_for_shifting = MR
            
            max_iterations = 20  # More iterations for convergence
            tolerance = 1e-6  # Stricter tolerance
            
            for iteration in range(max_iterations):
                gamma_exit_old = gamma_exit_iter
                
                # Calculate shifting equilibrium properties based on current exit conditions
                # Only pass reaction_rate_factor if explicitly set in config
                if reaction_rate_factor is not None:
                    shifting_props = calculate_shifting_equilibrium_properties(
                        Pc_val,
                        Tc,
                        gamma_val,
                        R,
                        P_exit_iter,  # Use current P_exit
                        progress_chamber,
                        reaction_rate_factor,
                        cea_cache,
                        MR=MR_for_shifting,  # Pass MR - no hardcoded values
                    )
                else:
                    # Use default (physics-based Da/(1+Da))
                    shifting_props = calculate_shifting_equilibrium_properties(
                        Pc_val,
                        Tc,
                        gamma_val,
                        R,
                        P_exit_iter,
                        progress_chamber,
                        cea_cache=cea_cache,
                        MR=MR_for_shifting,
                    )
                
                gamma_exit_iter = shifting_props["gamma_exit"]
                R_exit_iter = shifting_props["R_exit"]
                equilibrium_factor = shifting_props["equilibrium_factor"]
                
                # Check convergence
                gamma_change = abs(gamma_exit_iter - gamma_exit_old) / max(gamma_exit_old, 1.0)
                if gamma_change < tolerance:
                    break
                
                # Recalculate M_exit with new gamma_exit using consolidated solver
                if eps_val > 1.0 and gamma_exit_iter > 1.0:
                    M_exit_iter, _ = solve_exit_mach_robust(eps_val, gamma_exit_iter)
                
                # Recalculate P_exit and T_exit with new gamma_exit and M_exit_iter
                # Using isentropic relations (physics-based, no arbitrary factors)
                pressure_exponent_new = -gamma_exit_iter / (gamma_exit_iter - 1.0)
                pressure_factor_new = (1.0 + (gamma_exit_iter - 1.0) / 2.0 * M_exit_iter**2) ** pressure_exponent_new
                P_exit_iter = Pc_val * pressure_factor_new
                
                # Physics: P_exit cannot be negative
                if P_exit_iter < 0:
                    raise ValueError(f"Non-physical exit pressure: P_exit={P_exit_iter} Pa")
                
                # Note: P_exit can be < Pa (overexpanded) or > Pa (underexpanded)
                # Don't clamp to Pa - that's a physics result
                
                temperature_factor_new = 1.0 / (1.0 + (gamma_exit_iter - 1.0) / 2.0 * M_exit_iter**2)
                T_exit_iter = Tc * temperature_factor_new
                
                # Physics validation: T_exit must be positive and less than Tc
                if T_exit_iter <= 0 or T_exit_iter >= Tc:
                    raise ValueError(f"Non-physical exit temperature: T_exit={T_exit_iter} K, Tc={Tc} K")
            
            # Use converged values
            gamma_exit = gamma_exit_iter
            R_exit = R_exit_iter
            M_exit = M_exit_iter
            P_exit = P_exit_iter
            T_exit = T_exit_iter
            
        except Exception as e:
            # Don't silently fail - shifting equilibrium failure indicates physics issue
            import warnings
            warnings.warn(f"Shifting equilibrium calculation failed: {e}. This may indicate invalid nozzle conditions. Using chamber gamma as fallback.")
            # Fallback: use chamber gamma (assumes frozen flow)
            gamma_exit = gamma_val
            R_exit = R
            equilibrium_factor = 0.0  # Indicate frozen assumption
    
    # CRITICAL: Recalculate v_exit with final exit properties (gamma_exit, R_exit, T_exit, M_exit)
    # v_exit = M_exit × sqrt(gamma_exit × R_exit × T_exit)
    # This ensures velocity is consistent with exit conditions
    # FOR ISENTROPIC FLOW, v_exit MUST be calculated from M_exit - energy equation is NOT valid!
    sound_speed_exit_final = np.sqrt(gamma_exit * R_exit * T_exit) if gamma_exit > 1.0 and R_exit > 0 and T_exit > 0 else None
    if sound_speed_exit_final is not None and np.isfinite(sound_speed_exit_final) and sound_speed_exit_final > 0 and M_exit > 0:
        v_exit = M_exit * sound_speed_exit_final
    else:
        # If sound speed calculation fails, we have invalid exit conditions
        import warnings
        warnings.warn(f"[CRITICAL] Cannot calculate v_exit from M_exit after shifting equilibrium: gamma_exit={gamma_exit:.4f}, R_exit={R_exit:.2f}, T_exit={T_exit:.1f} K, M_exit={M_exit:.4f}")
        v_exit = 0.0
    
    # Final validation of exit temperature
    T_exit_check = PhysicalConstraints.validate_temperature(T_exit, "T_exit")
    if not T_exit_check.passed and T_exit_check.severity == "error":
        # Don't fallback to Tc (that's way too high) - use conservative estimate
        T_exit = max(Tc * 0.3, 500.0)  # At least 30% of chamber temp or 500K minimum
    
    # CRITICAL: ALWAYS recalculate exit velocity from M_exit to ensure physical consistency
    # v_exit MUST be calculated from M_exit for isentropic flow: v_exit = M_exit × sqrt(gamma_exit × R_exit × T_exit)
    # This ensures velocity-Mach consistency and correct thrust calculation
    # This is the ONLY physically correct method for isentropic nozzle flow
    # CRITICAL FIX: Ensure M_exit is valid BEFORE calculating v_exit
    if M_exit <= 0.0 or M_exit < 1.0:
        import warnings
        warnings.warn(f"[CRITICAL] M_exit is invalid ({M_exit:.6f}) before final v_exit calculation. Recalculating from eps={eps_val:.4f}, gamma_exit={gamma_exit:.4f}")
        # Recalculate M_exit from area-Mach relation
        if eps_val > 1.0 and gamma_exit > 1.0:
            # Use consolidated solver
            M_exit, _ = solve_exit_mach_robust(eps_val, gamma_exit)
    
    # Now calculate v_exit from M_exit (ALWAYS use this method for isentropic flow)
    if M_exit > 0.0 and M_exit >= 1.0:
        sound_speed_exit_consistent = np.sqrt(gamma_exit * R_exit * T_exit) if gamma_exit > 1.0 and R_exit > 0 and T_exit > 0 else None
        if sound_speed_exit_consistent is not None and np.isfinite(sound_speed_exit_consistent) and sound_speed_exit_consistent > 0:
            v_exit_from_M = M_exit * sound_speed_exit_consistent
            # ALWAYS use M_exit-based velocity - it's the physically correct method for isentropic flow
            # The energy equation is only a fallback and can give incorrect results
            v_exit = v_exit_from_M
        else:
            # If sound speed calculation fails, we have a problem
            import warnings
            warnings.warn(f"[WARNING] Cannot calculate exit velocity from M_exit: gamma_exit={gamma_exit:.4f}, R_exit={R_exit:.2f}, T_exit={T_exit:.1f} K")
    
    # CRITICAL: Final validation - ensure M_exit is always valid before calculating thrust
    # This MUST happen before v_exit validation to ensure v_exit is calculated correctly
    if M_exit <= 0.0 or M_exit < 1.0:
        import warnings
        warnings.warn(f"[CRITICAL] M_exit is invalid ({M_exit:.6f}) before thrust calculation. Recalculating from eps={eps_val:.4f}, gamma_exit={gamma_exit:.4f}")
        # Recalculate M_exit from area-Mach relation using consolidated solver
        if eps_val > 1.0 and gamma_exit > 1.0:
            M_exit, _ = solve_exit_mach_robust(eps_val, gamma_exit)
        else:
            M_exit = 2.0  # Fallback
        # Recalculate v_exit with corrected M_exit
        sound_speed_exit_final = np.sqrt(gamma_exit * R_exit * T_exit) if gamma_exit > 1.0 and R_exit > 0 and T_exit > 0 else None
        if sound_speed_exit_final is not None and np.isfinite(sound_speed_exit_final) and sound_speed_exit_final > 0:
            v_exit = M_exit * sound_speed_exit_final
        else:
            import warnings
            warnings.warn(f"[WARNING] Cannot recalculate v_exit: gamma_exit={gamma_exit:.4f}, R_exit={R_exit:.2f}, T_exit={T_exit:.1f} K")
    
    # Validate exit velocity
    v_exit_check = PhysicalConstraints.validate_velocity(v_exit, "v_exit")
    if not v_exit_check.passed and v_exit_check.severity == "error":
        # Recalculate from M_exit as last resort
        if M_exit > 0.0 and M_exit >= 1.0:
            sound_speed_exit_final = np.sqrt(gamma_exit * R_exit * T_exit) if gamma_exit > 1.0 and R_exit > 0 and T_exit > 0 else None
            if sound_speed_exit_final is not None and np.isfinite(sound_speed_exit_final) and sound_speed_exit_final > 0:
                v_exit = M_exit * sound_speed_exit_final
            else:
                v_exit = 0.0
        else:
            v_exit = 0.0  # Fallback
    
    # CRITICAL: Final v_exit recalculation RIGHT BEFORE thrust calculation
    # This ensures v_exit is ALWAYS calculated from M_exit for isentropic flow
    # v_exit = M_exit × sqrt(gamma_exit × R_exit × T_exit)
    sound_speed_final = np.sqrt(gamma_exit * R_exit * T_exit) if gamma_exit > 1.0 and R_exit > 0 and T_exit > 0 else None
    if sound_speed_final is not None and np.isfinite(sound_speed_final) and sound_speed_final > 0 and M_exit > 0 and M_exit >= 1.0:
        v_exit = M_exit * sound_speed_final
    else:
        import warnings
        warnings.warn(f"[CRITICAL] Cannot calculate v_exit from M_exit before thrust: gamma_exit={gamma_exit:.4f}, R_exit={R_exit:.2f}, T_exit={T_exit:.1f} K, M_exit={M_exit:.4f}")
        # Last resort: use zero velocity (will give zero thrust, which is better than wrong thrust)
        v_exit = 0.0
    
    # Calculate thrust components with validation
    g0 = 9.80665

    PcAt = Pc_val * A_throat
    F_mom = mdot_total * v_exit
    F_pres = (P_exit - Pa) * A_exit
    F_sum = F_mom + F_pres

    Cf_from_sum = F_sum / PcAt
    Isp_from_sum = F_sum / (mdot_total * g0)

    # Expected “effective exhaust velocity” should be ~ Isp*g0
    c_eff = F_sum / mdot_total

    print(
        f"[NOZZLE][THRUSTCHK] PcAt={PcAt:.1f} N | "
        f"mdot={mdot_total:.4f} kg/s, v_exit={v_exit:.1f} m/s, c_eff=F/mdot={c_eff:.1f} m/s | "
        f"F_mom={F_mom:.1f} N, F_pres={F_pres:.1f} N, F_sum={F_sum:.1f} N | "
        f"Cf_sum={Cf_from_sum:.3f}, Isp_sum={Isp_from_sum:.1f} s"
    )

    # Compare to CEA-ish expectations
    print(
        f"[NOZZLE][EXPECT] c*_ideal={cea_props['cstar_ideal']:.1f} m/s, "
        f"Cf_ideal={Cf_ideal:.3f} => expected c_eff~Cf*c*={Cf_ideal*cea_props['cstar_ideal']:.1f} m/s"
    )

    F_momentum = mdot_total * v_exit
    F_pressure = (P_exit - Pa) * A_exit
    F_total = F_momentum + F_pressure

    print(
        "[NOZZLE FINAL]",
        f"Pc={Pc_val:.3e} Pa, At={A_throat:.3e} m^2, Ae={A_exit:.3e} m^2, eps={A_exit/A_throat:.3f}",
        f"M_exit={M_exit:.4f}, gamma_exit={gamma_exit:.4f}, R_exit={R_exit:.2f}",
        f"T_exit={T_exit:.1f} K, Tc={Tc:.1f} K",
        f"v_exit={v_exit:.1f} m/s",
        f"P_exit={P_exit:.3e} Pa, Pa={Pa:.3e} Pa",
        f"mdot={mdot_total:.4f} kg/s",
        f"F_mom={F_momentum:.1f} N, F_pres={F_pressure:.1f} N",
        f"PcAt={Pc_val*A_throat:.1f} N",
        f"Cf_actual={F_total/(Pc_val*A_throat):.3f}, Cf_ideal={Cf_ideal:.3f}"
    )

    
    # Validate thrust components
    if not all(np.isfinite([F_momentum, F_pressure, F_total])):
        # Fallback to thrust coefficient method
        F_total = Cf * Pc_val * A_throat
    
    # Also calculate using thrust coefficient method for validation
    F_cf = Cf * Pc_val * A_throat
    
    # Validate thrust equation
    thrust_check = PhysicsValidator.validate_thrust_equation(
        F_momentum, F_pressure, F_total, tolerance=1e-2  # 1% tolerance
    )
    if not thrust_check.passed and thrust_check.severity == "error":
        # Use thrust coefficient method if momentum+pressure fails
        F_total = F_cf
    
    # Use the more accurate method (momentum + pressure)
    F = F_total

    # Calculate ACTUAL thrust coefficient from thrust equation
    # Cf_actual = F / (Pc * A_throat)
    # This is the measured value, not the theoretical
    Cf_actual, cf_actual_valid = NumericalStability.safe_divide(
        F, Pc_val * A_throat, 0.0, "Cf_actual"
    )
    if not cf_actual_valid.passed:
        Cf_actual = Cf  # Fallback to theoretical
    
    # Validate Cf_actual is reasonable
    # Typical range: 1.2-2.0 for most nozzles
    # If Cf_actual > Cf_ideal significantly, it may indicate:
    # 1. Underexpanded nozzle (P_exit > Pa) - this is OK and gives extra pressure thrust
    # 2. Calculation error (T_exit too high → v_exit too low → but that gives LOW thrust, not high)
    # 3. Exit velocity calculation error
    if Cf_actual > Cf_ideal * 1.2:  # More than 20% higher
        import warnings
        warnings.warn(
            f"Cf_actual ({Cf_actual:.4f}) significantly > Cf_ideal ({Cf_ideal:.4f}). "
            f"This may indicate underexpanded nozzle (P_exit={P_exit/1e6:.2f} MPa > Pa={Pa/1e6:.2f} MPa) "
            f"or calculation error. Check exit temperature (T_exit={T_exit:.1f} K, Tc={Tc:.1f} K)."
        )

    # Calculate throat temperature (at M=1, choked flow)
    # T_throat/Tc = 2/(gamma+1) for isentropic flow
    gamma_throat = gamma_val  # Use chamber gamma at throat (before expansion)
    throat_temp_ratio, throat_ratio_valid = NumericalStability.safe_divide(
        2.0, gamma_throat + 1.0, 0.5, "2/(gamma+1)"
    )
    if throat_ratio_valid.passed:
        T_throat = Tc * throat_temp_ratio
    else:
        T_throat = Tc * 0.85  # Fallback (typical for gamma ~1.2)

    # Validate throat temperature
    T_throat_check = PhysicalConstraints.validate_temperature(T_throat, "T_throat")
    if not T_throat_check.passed and T_throat_check.severity == "error":
        T_throat = Tc * 0.85  # Fallback

    # Calculate throat pressure (critical pressure at M=1)
    # P_throat/Pc = [2/(gamma+1)]^(gamma/(gamma-1))
    if gamma_throat > 1.0:
        pressure_exponent_throat = gamma_throat / (gamma_throat - 1.0)
        pressure_ratio_throat = throat_temp_ratio ** pressure_exponent_throat
        P_throat = Pc_val * pressure_ratio_throat
    else:
        P_throat = Pc_val * 0.6  # Fallback

    # Calculate Isp with validation
    g0 = 9.80665  # m/s²
    Isp, isp_valid = NumericalStability.safe_divide(F, mdot_total * g0, 0.0, "Isp")
    if not isp_valid.passed:
        Isp = 0.0

    # Validate Isp
    isp_check = PhysicalConstraints.validate_isp(Isp)
    if not isp_check.passed and isp_check.severity == "error":
        Isp = 0.0  # Fallback

    # Calculate chamber temperature profile if reaction progress available
    temperature_profile = None
    if reaction_progress is not None:
        try:
            # Get Lstar from config if available
            Lstar = None
            if config is not None and hasattr(config, 'chamber_geometry'):
                V_chamber = getattr(config.chamber_geometry, 'volume', None)
                A_throat = getattr(config.chamber_geometry, 'A_throat', None)
                if V_chamber is not None and A_throat is not None and A_throat > 0:
                    Lstar = V_chamber / A_throat
            
            if Lstar is not None:
                temperature_profile = calculate_chamber_temperature_profile(
                    Tc, Lstar, reaction_progress, n_points=20
                )
        except Exception as e:
            import warnings
            warnings.warn(f"Temperature profile calculation failed: {e}")

    results = {
        "F": float(F),
        "F_momentum": float(F_momentum),
        "F_pressure": float(F_pressure),
        "F_cf_method": float(F_cf),  # For comparison
        "Cf": float(Cf_actual),  # Return actual Cf (measured from thrust)
        "Cf_actual": float(Cf_actual),  # Explicit actual value
        "Cf_ideal": float(Cf_ideal),  # Ideal from CEA
        "Cf_theoretical": float(Cf),  # Theoretical (efficiency-adjusted ideal)
        "P_exit": float(P_exit),
        "P_throat": float(P_throat),
        "v_exit": float(v_exit),
        "T_exit": float(T_exit),
        "T_throat": float(T_throat),
        "temperature_profile": temperature_profile,  # Full profile along chamber
        "Isp": float(Isp),
        "gamma_chamber": float(gamma_val),
        "gamma_exit": float(gamma_exit),
        "R_chamber": float(R),
        "R_exit": float(R_exit),
        "equilibrium_factor": float(equilibrium_factor),
        "M_exit": float(M_exit) if M_exit > 1.0 else float(1.0 + 1e-6),  # Physics requirement: must be supersonic
    }
    
    # CRITICAL: Final check - ensure M_exit is always valid before returning
    # Recalculate M_exit if it's invalid
    if results["M_exit"] <= 0.0 or results["M_exit"] < 1.0:
        import warnings
        warnings.warn(f"[CRITICAL] M_exit in results is invalid ({results['M_exit']:.6f}). Recalculating from eps={eps_val:.4f}, gamma_exit={gamma_exit:.4f}")
        if eps_val > 1.0 and gamma_exit > 1.0:
            M_exit_recalc, _ = solve_exit_mach_robust(eps_val, gamma_exit)
        else:
            M_exit_recalc = 2.0
        results["M_exit"] = float(M_exit_recalc)
    
    return results
