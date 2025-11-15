"""Nozzle model: thrust coefficient and thrust calculation with shifting equilibrium"""

import numpy as np
from typing import Dict, Optional, Any
from pintle_pipeline.config_schemas import NozzleConfig
from pintle_pipeline.cea_cache import CEACache
from pintle_pipeline.numerical_robustness import (
    PhysicalConstraints,
    NumericalStability,
    PhysicsValidator,
)


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
    nozzle_config: NozzleConfig,
    Pa: float = 101325.0,
    eps: float = None,
    reaction_progress: Optional[Dict] = None,
    use_shifting_equilibrium: bool = True,
    config: Optional[Any] = None,
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
    eps : float, optional
        Expansion ratio (A_exit / A_throat). If None, uses nozzle_config.expansion_ratio.
        For ablative nozzles, this changes over time as throat/exit areas evolve.
    
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
    # Use provided eps or default from config
    if eps is None:
        eps = nozzle_config.expansion_ratio
    else:
        # Validate provided eps
        if eps <= 1.0:
            import warnings
            warnings.warn(f"Invalid expansion ratio: eps={eps:.4f} (should be > 1). Using config value {nozzle_config.expansion_ratio:.4f} instead.")
            eps = nozzle_config.expansion_ratio
    
    # Final validation
    if eps <= 1.0:
        raise ValueError(f"Invalid expansion ratio: eps={eps:.4f}. Must be > 1.0 for supersonic nozzle.")
    
    # Get CEA properties (now with eps parameter for 3D cache)
    cea_props = cea_cache.eval(MR, Pc, Pa, eps)
    Cf_ideal = cea_props["Cf_ideal"]
    gamma = cea_props["gamma"]
    Tc = cea_props["Tc"]
    R = cea_props["R"]
    
    # Apply nozzle efficiency
    Cf = nozzle_config.efficiency * Cf_ideal
    
    # Calculate exit pressure using isentropic relations
    # eps is already set above (from parameter or config)
    
    # For supersonic nozzles (eps > 1), we need to solve the area-Mach relation:
    # A/A* = (1/M) × [(2/(gamma+1)) × (1 + (gamma-1)/2 × M²)]^((gamma+1)/(2(gamma-1)))
    # Then use isentropic relation: P/Pc = [1 + (gamma-1)/2 × M²]^(-gamma/(gamma-1))
    
    # Validate inputs
    gamma_val = float(gamma)
    eps_val = float(eps)
    Pc_val = float(Pc)
    
    gamma_check = PhysicalConstraints.validate_gamma(gamma_val)
    if not gamma_check.passed and gamma_check.severity == "error":
        raise ValueError(f"Invalid gamma: {gamma_check.message}")
    
    if gamma_val > 1 and eps_val > 1:
        # Solve for exit Mach number from area ratio (supersonic solution)
        # Area-Mach relation: A/A* = (1/M) × [(2/(gamma+1)) × (1 + (gamma-1)/2 × M²)]^((gamma+1)/(2(gamma-1)))
        # For supersonic flow (M > 1), we need to solve this iteratively
        
        # Improved initial guess using asymptotic expansion
        # For supersonic flow, we need M > 1
        # For large eps: M ≈ (gamma+1)^((gamma+1)/(2(gamma-1))) / (2^(1/(gamma-1)) × eps)
        # For small eps near 1: M ≈ 1 + sqrt(2*(eps-1)/(gamma+1))
        # CRITICAL: Must start supersonic (M > 1) to find supersonic solution
        if eps_val > 10.0:
            # Large expansion ratio - use asymptotic formula
            prefactor = ((gamma_val + 1.0) / 2.0) ** ((gamma_val + 1.0) / (2.0 * (gamma_val - 1.0)))
            M_exit = prefactor / (eps_val ** (1.0 / (gamma_val - 1.0)))
        elif eps_val > 1.5:
            # Moderate expansion ratio - use improved approximation
            M_exit = 1.0 + np.sqrt(2.0 * (eps_val - 1.0) / (gamma_val + 1.0))
        else:
            # Small expansion ratio (near 1) - use linear approximation
            M_exit = 1.0 + 0.5 * (eps_val - 1.0)
        
        # CRITICAL: Ensure supersonic initial guess (M > 1.0)
        # If eps < 1, something is wrong (shouldn't happen for rocket nozzle)
        if eps_val <= 1.0:
            import warnings
            warnings.warn(f"Invalid expansion ratio: eps={eps_val:.4f} (should be > 1). Using M=2.0 as fallback.")
            M_exit = 2.0
        else:
            M_exit = max(M_exit, 1.1)  # Ensure clearly supersonic (M > 1.1)
            M_exit = min(M_exit, 10.0)  # Cap at M=10
        
        # Enhanced Newton-Raphson with convergence safeguards
        tolerance = 1e-10  # Stricter tolerance
        max_iterations = 50  # More iterations for robustness
        convergence_history = []
        
        for iteration in range(max_iterations):
            # Calculate A/A* for current M
            term = (2.0 / (gamma_val + 1.0)) * (1.0 + (gamma_val - 1.0) / 2.0 * M_exit**2)
            exponent = (gamma_val + 1.0) / (2.0 * (gamma_val - 1.0))
            A_Astar = (1.0 / M_exit) * (term ** exponent)
            
            # Error
            error = A_Astar - eps_val
            convergence_history.append(abs(error))
            
            if abs(error) < tolerance:
                break
            
            # Derivative d(A/A*)/dM (analytical)
            # d/dM[(1/M) × term^exponent]
            # = -A_Astar/M + A_Astar × exponent × (1/term) × d(term)/dM
            dterm_dM = (gamma_val - 1.0) * M_exit
            dA_dM = -A_Astar / M_exit + A_Astar * exponent * (dterm_dM / term)
            
            # Safeguard against zero or very small derivative
            if abs(dA_dM) < 1e-12:
                # Use bisection-like fallback
                if error > 0:
                    M_exit = M_exit * 0.99  # Reduce M
                else:
                    M_exit = M_exit * 1.01  # Increase M
            else:
                # Newton step with damping for stability
                step = error / dA_dM
                # Limit step size to prevent overshoot
                step = np.clip(step, -0.5 * M_exit, 0.5 * M_exit)
                M_exit = M_exit - step
            
            # CRITICAL: Ensure supersonic and reasonable bounds
            # Must stay supersonic (M > 1.0) - if it goes subsonic, something is wrong
            if M_exit < 1.0:
                import warnings
                warnings.warn(f"Mach number solver converged to subsonic M={M_exit:.3f} (eps={eps_val:.2f}). Resetting to supersonic guess.")
                # Reset to supersonic guess
                M_exit = 1.0 + np.sqrt(2.0 * (eps_val - 1.0) / (gamma_val + 1.0))
                M_exit = max(M_exit, 1.1)
            
            M_exit = max(M_exit, 1.01)  # Ensure supersonic
            M_exit = min(M_exit, 10.0)  # Cap at M=10
        
        # Validate convergence
        if abs(error) > tolerance * 10:  # Allow 10x tolerance for warning
            import warnings
            warnings.warn(f"Mach number solver did not fully converge: |error| = {abs(error):.2e} after {max_iterations} iterations")
        
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
        sound_speed_exit_squared = gamma_val * R * T_exit
        sound_speed_exit, sound_valid = NumericalStability.safe_sqrt(sound_speed_exit_squared, "sound_speed_exit")
        if sound_valid.passed and np.isfinite(sound_speed_exit) and sound_speed_exit > 0:
            v_exit = M_exit * sound_speed_exit
        else:
            # Fallback to energy equation if sound speed calculation fails
            cp = gamma_val * R / max(gamma_val - 1.0, 0.01)
            delta_T = Tc - T_exit
            if delta_T > 0 and cp > 0:
                v_exit_squared = 2.0 * cp * delta_T
                v_exit, v_valid = NumericalStability.safe_sqrt(v_exit_squared, "v_exit")
                if not v_valid.passed:
                    v_exit = 0.0
            else:
                v_exit = 0.0
        
        # Validate exit pressure
        if not np.isfinite(P_exit) or P_exit < 0:
            P_exit = Pa  # Fallback to ambient
        
        P_exit = max(P_exit, Pa)  # Can't be less than ambient
        
        # Validate exit temperature
        # Typical: T_exit/Tc = 0.3-0.7 for well-expanded nozzles
        T_exit_min = Tc * 0.2  # Minimum reasonable
        T_exit_max = Tc * 0.95  # Maximum reasonable
        if not np.isfinite(T_exit) or T_exit < T_exit_min or T_exit > T_exit_max:
            import warnings
            warnings.warn(f"Exit temperature {T_exit:.1f} K seems unphysical (Tc={Tc:.1f} K, M_exit={M_exit:.2f}). Recalculating.")
            # Recalculate with conservative estimate
            T_exit = Tc * 0.5  # Conservative fallback
            # Recalculate velocity with corrected temperature
            sound_speed_exit_squared = gamma_val * R * T_exit
            sound_speed_exit, sound_valid = NumericalStability.safe_sqrt(sound_speed_exit_squared, "sound_speed_exit")
            if sound_valid.passed and np.isfinite(sound_speed_exit) and sound_speed_exit > 0:
                v_exit = M_exit * sound_speed_exit
    else:
        P_exit = Pa
        T_exit = Tc  # No expansion
        v_exit = 0.0
    
    # Apply shifting equilibrium if enabled
    # As gas expands, equilibrium composition shifts. Gamma changes
    # between chamber value (equilibrium) and exit (may be shifted)
    gamma_exit = gamma_val
    R_exit = R
    equilibrium_factor = 1.0
    
    if use_shifting_equilibrium and P_exit < Pc_val:
        try:
            from pintle_pipeline.reaction_chemistry import calculate_shifting_equilibrium_properties
            
            # Get reaction progress at chamber (if provided)
            progress_chamber = 1.0  # Default: assume equilibrium at chamber
            if reaction_progress is not None:
                progress_chamber = reaction_progress.get("progress_throat", 1.0)
            
            # Reaction rate factor (0 = frozen, 1 = complete equilibrium)
            # Default: 0.1 means mostly frozen but some shifting
            reaction_rate_factor = 0.1
            if config is not None and hasattr(config, 'combustion') and hasattr(config.combustion, 'efficiency'):
                # Could add config parameter for this
                pass
            
            # Calculate shifting equilibrium properties
            shifting_props = calculate_shifting_equilibrium_properties(
                Pc_val,
                Tc,
                gamma_val,
                R,
                P_exit,
                progress_chamber,
                reaction_rate_factor,
            )
            
            gamma_exit = shifting_props["gamma_exit"]
            R_exit = shifting_props["R_exit"]
            equilibrium_factor = shifting_props["equilibrium_factor"]
            
            # Recalculate T_exit with shifting gamma
            T_exit_shifting = shifting_props["T_exit"]
            
        except Exception as e:
            # Don't silently fail - shifting equilibrium failure indicates physics issue
            import warnings
            warnings.warn(f"Shifting equilibrium calculation failed: {e}. This may indicate invalid nozzle conditions. Using chamber gamma as fallback.")
            # Fallback: use chamber gamma (assumes frozen flow)
            gamma_exit = gamma_val
            R_exit = R
            equilibrium_factor = 0.0  # Indicate frozen assumption
    
    # Use shifting equilibrium gamma for exit calculations
    gamma_for_exit = gamma_exit
    
    # Recalculate exit properties with shifting gamma if it changed significantly
    if abs(gamma_exit - gamma_val) > 0.01:  # Significant difference
        # Recalculate M_exit with new gamma (simplified correction)
        # For more accuracy, we'd re-solve area-Mach relation
        gamma_ratio = gamma_exit / gamma_val
        M_correction = 1.0 + 0.1 * (gamma_ratio - 1.0)  # Small correction
        M_exit = M_exit * M_correction
        M_exit = np.clip(M_exit, 1.01, 10.0)
        
        # Recalculate P_exit with new gamma
        pressure_exponent_new = -gamma_exit / (gamma_exit - 1.0)
        pressure_factor_new = (1.0 + (gamma_exit - 1.0) / 2.0 * M_exit**2) ** pressure_exponent_new
        P_exit = Pc_val * pressure_factor_new
        P_exit = max(P_exit, Pa)
        
        # Recalculate T_exit with new gamma (consistent with M_exit)
        temperature_factor_new = 1.0 / (1.0 + (gamma_exit - 1.0) / 2.0 * M_exit**2)
        T_exit = Tc * temperature_factor_new
        
        # Recalculate v_exit with new gamma and R
        sound_speed_exit_squared_new = gamma_exit * R_exit * T_exit
        sound_speed_exit_new, sound_valid_new = NumericalStability.safe_sqrt(sound_speed_exit_squared_new, "sound_speed_exit_new")
        if sound_valid_new.passed and np.isfinite(sound_speed_exit_new) and sound_speed_exit_new > 0:
            v_exit = M_exit * sound_speed_exit_new
        else:
            # Fallback to energy equation
            cp_new = gamma_exit * R_exit / max(gamma_exit - 1.0, 0.01)
            delta_T_new = Tc - T_exit
            if delta_T_new > 0 and cp_new > 0:
                v_exit_squared_new = 2.0 * cp_new * delta_T_new
                v_exit, v_valid_new = NumericalStability.safe_sqrt(v_exit_squared_new, "v_exit_new")
                if not v_valid_new.passed:
                    v_exit = 0.0
            else:
                v_exit = 0.0
    
    # Final validation of exit temperature
    T_exit_check = PhysicalConstraints.validate_temperature(T_exit, "T_exit")
    if not T_exit_check.passed and T_exit_check.severity == "error":
        # Don't fallback to Tc (that's way too high) - use conservative estimate
        T_exit = max(Tc * 0.3, 500.0)  # At least 30% of chamber temp or 500K minimum
    
    # Validate exit velocity
    v_exit_check = PhysicalConstraints.validate_velocity(v_exit, "v_exit")
    if not v_exit_check.passed and v_exit_check.severity == "error":
        v_exit = 0.0  # Fallback
    
    # Calculate thrust components with validation
    F_momentum = mdot_total * v_exit
    F_pressure = (P_exit - Pa) * nozzle_config.A_exit
    F_total = F_momentum + F_pressure
    
    # Validate thrust components
    if not all(np.isfinite([F_momentum, F_pressure, F_total])):
        # Fallback to thrust coefficient method
        F_total = Cf * Pc_val * nozzle_config.A_throat
    
    # Also calculate using thrust coefficient method for validation
    F_cf = Cf * Pc_val * nozzle_config.A_throat
    
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
        F, Pc_val * nozzle_config.A_throat, 0.0, "Cf_actual"
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
            if config is not None and hasattr(config, 'chamber'):
                V_chamber = getattr(config.chamber, 'volume', None)
                A_throat = getattr(config.chamber, 'A_throat', None)
                if V_chamber is not None and A_throat is not None and A_throat > 0:
                    Lstar = V_chamber / A_throat
            
            if Lstar is not None:
                temperature_profile = calculate_chamber_temperature_profile(
                    Tc, Lstar, reaction_progress, n_points=20
                )
        except Exception as e:
            import warnings
            warnings.warn(f"Temperature profile calculation failed: {e}")

    return {
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
        "M_exit": float(M_exit) if 'M_exit' in locals() else np.nan,
    }
