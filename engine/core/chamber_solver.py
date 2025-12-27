"""Chamber pressure solver: solve supply(Pc) = demand(Pc)"""

import numpy as np
from scipy.optimize import brentq, newton
from typing import Tuple, Dict, Any, List, Optional

from engine.pipeline.config_schemas import PintleEngineConfig, ensure_chamber_geometry
from engine.pipeline.combustion_eff import eta_cstar, calculate_Lstar
from engine.pipeline.cea_cache import CEACache
from engine.pipeline.thermal.film_cooling import compute_film_cooling
from engine.pipeline.thermal.regen_cooling import (
    compute_regen_heat_transfer,
    estimate_hot_wall_heat_flux,
)
from engine.pipeline.thermal.ablative_cooling import compute_ablative_response
from engine.pipeline.numerical_robustness import (
    PhysicalConstraints,
    NumericalStability,
    PhysicsValidator,
    validate_engine_state,
)
from engine.pipeline.constants import (
    DEFAULT_CHAMBER_TEMP_K,
    DEFAULT_CSTAR_IDEAL_M_S,
    DEFAULT_GAMMA_ND,
    DEFAULT_GAS_CONST_J_KG_K,
    DEFAULT_TURBULENCE_INTENSITY_ND,
)
from engine.core.closure import flows


class ChamberSolver:
    """Solves for chamber pressure by balancing supply and demand"""
    
    def __init__(self, config: PintleEngineConfig, cea_cache: CEACache):
        self.config = config
        self.cea_cache = cea_cache
        
        # Ensure chamber_geometry exists
        cg = ensure_chamber_geometry(config)
        
        # Calculate L* once (use config value if provided, otherwise calculate)
        self.Lstar = calculate_Lstar(
            cg.volume,
            cg.A_throat,
            Lstar_override=cg.Lstar
        )
        self.injector_diameter = self._infer_injector_diameter()
        
        # Cache for spray quality (updated during solve)
        self.spray_quality_good = True
    
    def residual(self, Pc: float, P_tank_O: float, P_tank_F: float) -> float:
        """
        Calculate residual: supply(Pc) - demand(Pc)
        
        Residual = 0 when supply equals demand.
        
        Parameters:
        -----------
        Pc : float
            Chamber pressure guess [Pa]
        P_tank_O : float
            Oxidizer tank pressure [Pa]
        P_tank_F : float
            Fuel tank pressure [Pa]
        
        Returns:
        --------
        residual : float [kg/s]
        """
        # Validate inputs
        Pc_val = float(Pc)
        P_tank_O_val = float(P_tank_O)
        P_tank_F_val = float(P_tank_F)
        
        # Physical constraint checks
        if not np.isfinite(Pc_val) or Pc_val <= 0:
            return np.nan  # Invalid pressure
        if not np.isfinite(P_tank_O_val) or P_tank_O_val <= 0:
            return np.nan
        if not np.isfinite(P_tank_F_val) or P_tank_F_val <= 0:
            return np.nan
        
        # Supply side: mass flow from injector (via closure)
        # flows() takes TANK PRESSURES and solves for mdot
        # It internally calculates:
        #   1. Feed losses: P_tank → P_injector
        #   2. Injector flow: P_injector - Pc → mdot
        #   3. Spray constraints: validates and adjusts if needed
        try:
            mdot_O, mdot_F, diagnostics = flows(
                P_tank_O_val,  # Tank pressure (INPUT)
                P_tank_F_val,  # Tank pressure (INPUT)
                Pc_val,        # Chamber pressure (GUESS - being solved for)
                self.config
            )
        except Exception as e:
            # If flows() fails, return NaN to signal invalid point
            import warnings
            warnings.warn(f"flows() failed in residual at Pc={Pc_val/1e6:.2f} MPa: {e}")
            return np.nan
        
        # Validate mass flows
        if not (np.isfinite(mdot_O) and np.isfinite(mdot_F)):
            return np.nan
        
        mdot_supply = mdot_O + mdot_F
        
        # Update spray quality for efficiency calculation
        self.spray_quality_good = diagnostics.get("constraints_satisfied", True)
        
        # Demand side: mass flow required by combustion
        MR, mr_valid = NumericalStability.safe_divide(mdot_O, mdot_F, 2.5, "MR")
        if not mr_valid.passed:
            return np.nan
        
        # Get CEA properties (IDEAL - infinite area equilibrium)
        # For 3D cache, use default expansion ratio from config (doesn't affect chamber properties much)
        try:
            cg = ensure_chamber_geometry(self.config)
            eps_default = cg.expansion_ratio
            cea_props = self.cea_cache.eval(MR, Pc_val, 101325.0, eps_default)
        except Exception as e:
            # Log the error for debugging but return NaN to signal failure
            import warnings
            warnings.warn(f"CEA cache eval failed in residual: {e}")
            return np.nan
        
        cstar_ideal = cea_props.get("cstar_ideal", 0.0)
        if not np.isfinite(cstar_ideal) or cstar_ideal <= 0:
            return np.nan
        
        # Apply combustion efficiency for FINITE CHAMBER
        # This corrects CEA's infinite-area assumption
        mixture_eff = self._compute_mixture_efficiency(diagnostics)

        cooling_results, cooling_eff, _ = self._evaluate_cooling_models(
            Pc_val,
            mdot_O,
            mdot_F,
            cea_props,
            diagnostics,
        )

        geometry = self._get_chamber_geometry()
        advanced_params = {
            "Pc": Pc_val,
            "Tc": cea_props.get("Tc", DEFAULT_CHAMBER_TEMP_K),
            "cstar_ideal": cea_props.get("cstar_ideal", DEFAULT_CSTAR_IDEAL_M_S),
            "gamma": cea_props.get("gamma", DEFAULT_GAMMA_ND),
            "R": cea_props.get("R", DEFAULT_GAS_CONST_J_KG_K),
            "MR": MR,
            "Ac": geometry["area_cross"],
            "At": cg.A_throat,  # Added At for residence time calculation
            "chamber_length": geometry["length"],  # Added for mixing models
            "Dinj": self.injector_diameter,
            "m_dot_total": mdot_supply,
            "spray_diagnostics": diagnostics,
            "turbulence_intensity": diagnostics.get("turbulence_intensity_mix", DEFAULT_TURBULENCE_INTENSITY_ND),
            "fuel_props": self._get_fuel_props(),
        }

        # Decide which efficiency model to use based on pressure gating
        pc_gate = getattr(self.config.combustion.efficiency, 'Pc_gate', 1000000.0)
        use_advanced = self.config.combustion.efficiency.use_advanced_model
        
        if use_advanced and Pc_val < pc_gate:
            # Below gate pressure, use simple model for stability
            if Pc_val < 1.1e5: # Only log at bottom bound (Pc_min) to avoid spam during iterations
                print(f"[ETA_GATE] Pc={Pc_val/1e6:.2f} MPa < Pc_gate={pc_gate/1e6:.2f} MPa -> using simple eta_cstar")
            use_advanced = False

        eta = eta_cstar(
            self.Lstar,
            self.config.combustion.efficiency,
            self.spray_quality_good,
            mixture_efficiency=mixture_eff,
            cooling_efficiency=cooling_eff,
            use_advanced_model=use_advanced,
            advanced_params=advanced_params,
        )
        
        # Validate efficiency
        if not np.isfinite(eta) or eta <= 0 or eta > 1.0:
            return np.nan
        
        # Actual c* accounting for finite chamber volume
        cstar_actual = eta * cstar_ideal
        
        # Demand: mdot = Pc * At / c*_actual
        # This uses the chamber-driven c*, not the ideal CEA value
        cg = ensure_chamber_geometry(self.config)
        mdot_demand, demand_valid = NumericalStability.safe_divide(
            Pc_val * cg.A_throat,
            cstar_actual,
            0.0,
            "mdot_demand"
        )
        if not demand_valid.passed:
            return np.nan
        
        residual = mdot_supply - mdot_demand
        
        # Validate residual is finite
        if not np.isfinite(residual):
            return np.nan
        
        return float(residual)
    
    def solve(
        self,
        P_tank_O: float,
        P_tank_F: float,
        Pc_guess: float = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Solve for chamber pressure.
        
        Parameters:
        -----------
        P_tank_O : float
            Oxidizer tank pressure [Pa]
        P_tank_F : float
            Fuel tank pressure [Pa]
        Pc_guess : float, optional
            Initial guess for chamber pressure [Pa]
        
        Returns:
        --------
        Pc : float [Pa]
            Solved chamber pressure
        diagnostics : dict
            Solution diagnostics
        """
        # Determine bounds
        # Realistic bounds: Pc must be less than both tank pressures (accounting for feed losses)
        Pc_min = 100000.0  # 100 kPa minimum
        
        # Estimate maximum feed losses for bounds calculation
        # Use rough estimates: assume maximum flow gives ~10-20% pressure drop
        # This is conservative but better than fixed 5% margin
        # Actual feed losses will be calculated during solve
        feed_loss_margin = 0.15  # 15% margin for feed losses (conservative estimate)
        Pc_max = min(P_tank_O, P_tank_F) * (1.0 - feed_loss_margin)
        
        # If fuel pressure is much higher than oxidizer, we might need to allow
        # Pc up to oxidizer pressure (since oxidizer flow limits the system)
        # But this is already handled by min(P_tank_O, P_tank_F)
        
        # Clamp to config bounds
        Pc_min = max(Pc_min, self.config.solver.Pc_bounds[0])
        Pc_max = min(Pc_max, self.config.solver.Pc_bounds[1])
        
        if Pc_max <= Pc_min:
            raise ValueError(f"Invalid pressure bounds: Pc_max ({Pc_max}) <= Pc_min ({Pc_min})")
        
        # Initial guess
        if Pc_guess is None:
            Pc_guess = (Pc_min + Pc_max) / 2
        
        # Create residual function with tank pressures bound
        def residual_func(Pc):
            return self.residual(Pc, P_tank_O, P_tank_F)
        
        # Check residual signs at bounds before solving
        residual_min = residual_func(Pc_min)
        residual_max = residual_func(Pc_max)
        
        # Check for NaN values and provide better error messages
        if not np.isfinite(residual_min):
            # Try to diagnose the issue
            try:
                # Test a few points to see where it fails
                test_Pc = (Pc_min + Pc_max) / 2
                test_res = residual_func(test_Pc)
                if not np.isfinite(test_res):
                    raise ValueError(
                        f"Residual function returns non-finite values. "
                        f"Pc_min={Pc_min/1e6:.2f} MPa, Pc_max={Pc_max/1e6:.2f} MPa. "
                        f"Check injector geometry, feed system, or CEA cache."
                    )
            except Exception as e:
                raise ValueError(
                    f"Residual function evaluation failed at bounds. "
                    f"Pc_min={Pc_min/1e6:.2f} MPa, Pc_max={Pc_max/1e6:.2f} MPa. "
                    f"Error: {e}"
                )
        
        if not np.isfinite(residual_max):
            raise ValueError(
                f"Residual function returns non-finite at Pc_max={Pc_max/1e6:.2f} MPa. "
                f"Check that tank pressures are sufficient and injector geometry is valid."
            )
        
        # brentq requires opposite signs at bounds
        if np.sign(residual_min) == np.sign(residual_max):
            # No root in interval - this happens when:
            # 1. Supply > demand at all Pc (both positive) - need higher Pc but limited by tank pressure
            # 2. Supply < demand at all Pc (both negative) - can't supply enough flow
            
            if residual_min > 0 and residual_max > 0:
                # Supply > Demand at all Pc
                # This means injector supplies more flow than combustion can demand
                # Common causes:
                # 1. Injector too large (orifice areas too big)
                # 2. Throat too small (can't flow enough to balance supply)
                # 3. Combustion efficiency too low (reduces demand)
                # 4. Pc_max too conservative (we could go slightly higher)
                
                # Initialize skip_solve flag
                skip_solve = False
                
                # Check if residual is small at Pc_max (near solution)
                residual_tolerance = 0.1  # kg/s - accept if within 0.1 kg/s
                
                if residual_max < residual_tolerance:
                    # Residual is small - we're very close to solution
                    # Use Pc_max as solution with warning
                    import warnings
                    warnings.warn(
                        f"Supply slightly > Demand at Pc_max. "
                        f"Using Pc_max ({Pc_max/1e6:.2f} MPa) as solution. "
                        f"Residual: {residual_max:.4f} kg/s. "
                        f"Injector may be slightly oversized or throat slightly undersized."
                    )
                    # Skip to solution validation - use Pc_max as solution
                    Pc = Pc_max
                    success = True
                    # Skip the root finding loop below
                    skip_solve = True
                else:
                    # Residual is significant - diagnose the issue
                    # Get diagnostics at Pc_max to understand supply/demand
                    try:
                        mdot_O_test, mdot_F_test, diag_test = flows(
                            P_tank_O, P_tank_F, Pc_max, self.config
                        )
                        mdot_supply_test = mdot_O_test + mdot_F_test
                        
                        # Get demand at Pc_max
                        MR_test = mdot_O_test / mdot_F_test if mdot_F_test > 0 else np.inf
                        cg = ensure_chamber_geometry(self.config)
                        eps_default = cg.expansion_ratio
                        cea_props_test = self.cea_cache.eval(MR_test, Pc_max, 101325.0, eps_default)
                        cstar_ideal_test = cea_props_test.get("cstar_ideal", 0.0)
                        
                        # Calculate efficiency
                        eta_test = eta_cstar(
                            self.Lstar,
                            self.config.combustion.efficiency,
                            diag_test.get("constraints_satisfied", True),
                            mixture_efficiency=diag_test.get("mixture_efficiency", 1.0),
                            cooling_efficiency=diag_test.get("cooling_efficiency", 1.0),
                        )
                        cstar_actual_test = eta_test * cstar_ideal_test
                        cg = ensure_chamber_geometry(self.config)
                        mdot_demand_test = (Pc_max * cg.A_throat) / cstar_actual_test if cstar_actual_test > 0 else np.inf
                        
                        # Calculate what Pc would balance (extrapolate)
                        # residual = supply - demand
                        # At Pc_max: residual = mdot_supply - mdot_demand
                        # Demand scales with Pc: mdot_demand ∝ Pc
                        # Supply decreases slightly with Pc: mdot_supply decreases as Pc increases
                        # Rough estimate: if we increase Pc by ΔPc, demand increases more than supply
                        
                        # Estimate required Pc (rough extrapolation)
                        # Assume linear relationship near Pc_max
                        if mdot_demand_test > 0 and mdot_supply_test > mdot_demand_test:
                            # We need more Pc to increase demand
                            # mdot_demand = Pc * At / c*, so Pc_needed = mdot_supply * c* / At
                            cg = ensure_chamber_geometry(self.config)
                            Pc_estimate = mdot_supply_test * cstar_actual_test / cg.A_throat
                            
                            raise ValueError(
                                f"No solution: Supply > Demand at all Pc. "
                                f"Residual at Pc_min: {residual_min:.4f} kg/s, at Pc_max: {residual_max:.4f} kg/s. "
                                f"\nDiagnostics at Pc_max ({Pc_max/1e6:.2f} MPa):"
                                f"\n  - Supply: {mdot_supply_test:.4f} kg/s (mdot_O={mdot_O_test:.4f}, mdot_F={mdot_F_test:.4f})"
                                f"\n  - Demand: {mdot_demand_test:.4f} kg/s (c*_actual={cstar_actual_test:.1f} m/s, At={cg.A_throat*1e6:.2f} mm²)"
                                f"\n  - Estimated Pc needed: {Pc_estimate/1e6:.2f} MPa (vs Pc_max={Pc_max/1e6:.2f} MPa)"
                                f"\nPossible fixes:"
                                f"\n  1. Reduce injector orifice areas (currently oversized)"
                                f"\n  2. Increase throat area (currently undersized)"
                                f"\n  3. Increase tank pressures to allow higher Pc_max"
                                f"\n  4. Check combustion efficiency (low efficiency reduces demand)"
                            )
                        else:
                            raise ValueError(
                                f"No solution: Supply > Demand at all Pc. "
                                f"Residual: [{residual_min:.4f}, {residual_max:.4f}] kg/s. "
                                f"Could not compute detailed diagnostics."
                            )
                    except ValueError:
                        # Re-raise explicit ValueErrors from above
                        raise
                    except Exception as diag_e:
                        # Diagnostics failed - provide generic error
                        raise ValueError(
                            f"No solution: Supply > Demand at all Pc. "
                            f"Residual at bounds: [{residual_min:.4f}, {residual_max:.4f}] kg/s. "
                            f"Pc_max ({Pc_max/1e6:.2f} MPa) limited by tank pressure. "
                            f"Possible causes: Injector oversized, throat undersized, or combustion efficiency too low. "
                            f"Diagnostic error: {diag_e}"
                        )
                    
            else:
                # Supply < Demand at all Pc (both negative)
                raise ValueError(
                    f"No solution: Supply < Demand at all Pc. "
                    f"Residual at bounds: [{residual_min:.4f}, {residual_max:.4f}] kg/s. "
                    f"Insufficient mass flow. Check tank pressures and injector geometry."
                )
        
        # Check if we already have a solution (from small residual case above)
        # skip_solve is defined in the if-else block above, default to False if not set
        if 'skip_solve' not in locals():
            skip_solve = False
        
        if not skip_solve:
            # Validate bracket before solving
            bracket_check = NumericalStability.check_bracket(residual_func, Pc_min, Pc_max)
            if not bracket_check.passed:
                raise ValueError(f"Invalid bracket for root finding: {bracket_check.message}")
            
            # Track convergence history for diagnostics
            convergence_history = []
            
            # Enhanced residual function with convergence tracking
            def tracked_residual_func(Pc):
                res = residual_func(Pc)
                convergence_history.append(float(res))
                return res
            
            # Solve using bracketed secant (brentq) - safe and robust
            try:
                if self.config.solver.method == "brentq":
                    Pc, result = brentq(
                        tracked_residual_func,
                        Pc_min,
                        Pc_max,
                        xtol=self.config.solver.tolerance,
                        rtol=self.config.solver.tolerance * 1e-3,  # Relative tolerance
                        maxiter=self.config.solver.max_iterations,
                        full_output=True
                    )
                    success = result.converged
                    
                    # Validate convergence
                    conv_check = NumericalStability.check_convergence(
                        convergence_history,
                        self.config.solver.tolerance,
                        min_iterations=3
                    )
                    if not conv_check.passed and conv_check.severity == "error":
                        raise RuntimeError(f"Convergence validation failed: {conv_check.message}")
                        
                else:
                    # Fallback to Newton's method (less robust)
                    Pc = newton(
                        tracked_residual_func,
                        Pc_guess,
                        tol=self.config.solver.tolerance,
                        maxiter=self.config.solver.max_iterations
                    )
                    success = True
                    
                    # Validate convergence for Newton
                    conv_check = NumericalStability.check_convergence(
                        convergence_history,
                        self.config.solver.tolerance,
                        min_iterations=3
                    )
                    if not conv_check.passed and conv_check.severity == "error":
                        raise RuntimeError(f"Convergence validation failed: {conv_check.message}")
                        
            except ValueError as e:
                # Re-raise ValueError (bracket issues, etc.)
                raise
            except Exception as e:
                raise RuntimeError(f"Chamber pressure solver failed: {e}")
        else:
            # We're using Pc_max as solution (small residual case)
            # Already set Pc = Pc_max and success = True above
            convergence_history = [residual_max]  # Store for diagnostics
        
        # Validate solution
        Pc_val = float(Pc)
        if not np.isfinite(Pc_val):
            raise RuntimeError(f"Solver returned non-finite pressure: {Pc_val}")
        
        Pc_check = PhysicalConstraints.validate_pressure(Pc_val, "Pc_solution")
        if not Pc_check.passed and Pc_check.severity == "error":
            raise RuntimeError(f"Solution validation failed: {Pc_check.message}")
        
        # Get final diagnostics
        mdot_O, mdot_F, closure_diag = flows(P_tank_O, P_tank_F, Pc_val, self.config)
        MR = mdot_O / mdot_F if mdot_F > 0 else 0
        
        # Use current expansion ratio for 3D cache
        # FIXED: Add safety check for division by zero
        cg = ensure_chamber_geometry(self.config)
        eps_current = cg.A_exit / cg.A_throat if cg.A_throat and cg.A_exit and cg.A_throat > 0 else cg.expansion_ratio
        cea_props = self.cea_cache.eval(MR, Pc_val, 101325.0, eps_current)
        
        # Calculate total mass flow rate (needed for various calculations below)
        mdot_total = mdot_O + mdot_F
        
        # Calculate cooling effects early (needed for conservative reaction kinetics)
        cooling_results, cooling_eff, effective_Tc = self._evaluate_cooling_models(
            Pc_val,
            mdot_O,
            mdot_F,
            cea_props,
            closure_diag,
        )

        # Calculate reaction progress through chamber (if finite-rate chemistry enabled)
        reaction_progress = None
        if getattr(self.config.combustion.efficiency, 'use_finite_rate_chemistry', True):
            try:
                from engine.pipeline.reaction_chemistry import calculate_chamber_reaction_progress
                
                # Pass spray diagnostics if available for better evaporation/mixing estimates
                spray_diagnostics = closure_diag if closure_diag else None
                
                # Use conservative "Worst of Both Worlds" temperatures:
                # Tc (Ideal) for residence time (shorter time is conservative)
                # effective_Tc (Actual) for kinetics (slower chemistry is conservative)
                reaction_progress = calculate_chamber_reaction_progress(
                    self.Lstar,
                    Pc_val,
                    cea_props["Tc"], # Ideal Tc (Residence Time)
                    cea_props["cstar_ideal"],
                    cea_props["gamma"],
                    cea_props["R"],
                    MR,
                    self.config,
                    spray_diagnostics=spray_diagnostics,
                    Tc_kinetics=effective_Tc, # Actual Tc (Kinetics)
                )
            except Exception as e:
                # Don't silently fail - raise error or log warning
                import warnings
                warnings.warn(f"Reaction progress calculation failed: {e}. This may indicate invalid engine conditions.")
                # Minimal fallback - but indicate uncertainty
                # CRITICAL FIX: Correct residence time formula
                rho_chamber = Pc_val / (cea_props["R"] * cea_props["Tc"]) if cea_props["R"] > 0 and cea_props["Tc"] > 0 else 1.0
                # Use actual mdot_total from closure (calculated above)
                cg = ensure_chamber_geometry(self.config)
                tau_residence_correct = self.Lstar * rho_chamber * cg.A_throat / mdot_total if mdot_total > 0 else 0.001
                reaction_progress = {
                    "progress_throat": 1.0,  # Assume equilibrium
                    "tau_residence": tau_residence_correct,
                    "calculation_failed": True,
                }
        
        mixture_eff = self._compute_mixture_efficiency(closure_diag)

        # Use advanced combustion efficiency model if enabled
        pc_gate = getattr(self.config.combustion.efficiency, 'Pc_gate', 1000000.0)
        use_advanced = getattr(self.config.combustion.efficiency, 'use_advanced_model', True)
        
        # Apply pressure gating for final diagnostics consistency
        if use_advanced and Pc_val < pc_gate:
            print(f"[ETA_GATE] Final diagnostics: Pc={Pc_val/1e6:.2f} MPa < Pc_gate={pc_gate/1e6:.2f} MPa -> using simple eta_cstar")
            use_advanced = False

        advanced_params = None
        
        if use_advanced:
            geometry = self._get_chamber_geometry()
            
            advanced_params = {
                "Pc": Pc_val,
                "Tc": cea_props["Tc"],  # Ideal Tc (Conservative Residence Time)
                "Tc_kinetics": effective_Tc, # Actual Tc (Conservative Kinetics)
                "cstar_ideal": cea_props.get("cstar_ideal", DEFAULT_CSTAR_IDEAL_M_S),
                "gamma": cea_props.get("gamma", DEFAULT_GAMMA_ND),
                "R": cea_props.get("R", DEFAULT_GAS_CONST_J_KG_K),
                "MR": MR,
                "Ac": geometry["area_cross"],
                "At": cg.A_throat,
                "chamber_length": geometry["length"],
                "Dinj": self.injector_diameter,
                "m_dot_total": mdot_total,
                "u_fuel": closure_diag.get("u_F"),
                "u_lox": closure_diag.get("u_O"),
                "spray_diagnostics": closure_diag,
                "turbulence_intensity": closure_diag.get("turbulence_intensity_mix", DEFAULT_TURBULENCE_INTENSITY_ND),
                "fuel_props": self._get_fuel_props(),
            }
        
        eta = eta_cstar(
            self.Lstar,
            self.config.combustion.efficiency,
            self.spray_quality_good,
            mixture_efficiency=mixture_eff,
            cooling_efficiency=cooling_eff,
            use_advanced_model=use_advanced,
            advanced_params=advanced_params,
        )
        
        # Comprehensive validation of final solution
        cstar_actual = eta * cea_props["cstar_ideal"]
        gamma = cea_props["gamma"]
        R = cea_props["R"]

        print(
            f"[CSTAR] eta={eta:.4f} | c*_ideal={cea_props['cstar_ideal']:.1f} m/s -> c*_actual={cstar_actual:.1f} m/s | "
            f"ratio={cstar_actual/cea_props['cstar_ideal']:.4f}"
        )   
        
        # Calculate Isp for validation
        # CRITICAL FIX: Correct Isp formula
        # Isp = F / (mdot * g0) = (Cf * Pc * At) / (mdot * g0)
        # OR equivalently: Isp = cstar * Cf / g0
        # The previous formula with gamma * sqrt(...) was incorrect
        g0 = 9.80665
        Cf_ideal = cea_props.get("Cf_ideal", 1.5)  # Get from CEA, default to typical value
        cg = ensure_chamber_geometry(self.config)
        Cf_actual = cg.nozzle_efficiency * Cf_ideal  # Account for nozzle efficiency
        # Use correct formula: Isp = Cf * Pc * At / (mdot * g0)
        Isp = (Cf_actual * Pc_val * cg.A_throat) / (mdot_total * g0) if mdot_total > 0 else 0.0
        
        # Validate engine state (use effective temperature after cooling)
        validation_results = validate_engine_state(
            Pc_val, MR, mdot_total, cstar_actual, gamma, effective_Tc, Isp, 0.0  # F not calculated yet
        )
        
        # Check for critical errors
        critical_errors = [r for r in validation_results if r.severity == "error" and not r.passed]
        if critical_errors:
            error_msgs = [r.message for r in critical_errors]
            raise RuntimeError(f"Solution validation failed:\n" + "\n".join(error_msgs))
        
        diagnostics = {
            "Pc": Pc_val,
            "mdot_O": mdot_O,
            "mdot_F": mdot_F,
            "mdot_total": mdot_total,
            "MR": MR,
            "cstar_ideal": cea_props["cstar_ideal"],
            "Tc_ideal": cea_props["Tc"],  # Store original ideal temperature
            "cstar_actual": cstar_actual,
            "eta_cstar": eta,
            "mixture_efficiency": mixture_eff,
            "cooling_efficiency": cooling_eff,
            "Tc": effective_Tc,  # Use effective temperature after cooling (accounts for energy removal)
            "Tc_ideal": cea_props["Tc"],  # Store original CEA temperature for reference
            "gamma": gamma,
            "R": R,
            "M": cea_props.get("M"),  # Molecular weight [kg/kmol]
            "spray_quality_good": self.spray_quality_good,
            "validation_results": validation_results,  # Include validation results
            "convergence_history": convergence_history,  # Include convergence history
            **closure_diag,
        }

        diagnostics["cooling"] = cooling_results

        return Pc_val, diagnostics

    def _compute_mixture_efficiency(self, closure_diag: Dict[str, Any]) -> float:
        eff_cfg = self.config.combustion.efficiency
        if not eff_cfg.use_mixture_coupling or not isinstance(closure_diag, dict):
            return 1.0

        actual_smd = max(
            float(closure_diag.get("D32_O") or 0.0),
            float(closure_diag.get("D32_F") or 0.0),
        ) * 1e6  # convert to microns

        if not np.isfinite(actual_smd) or actual_smd <= 0:
            smd_factor = 1.0
        else:
            # Less conservative: use square root of ratio to reduce penalty severity
            # If actual SMD is 2x target, old: factor = (0.5)^1.0 = 0.5
            # New: factor = (0.5)^0.5 = 0.707 (less severe)
            ratio = eff_cfg.target_smd_microns / max(actual_smd, 1e-9)
            # Use reduced exponent: 0.5 instead of full exponent for less severe penalty
            # CRITICAL FIX: Remove arbitrary 0.5 multiplier and 0.5 floor
            # Use the configured exponent directly, and use configurable floor
            effective_exponent = eff_cfg.smd_penalty_exponent  # Remove arbitrary 0.5 multiplier
            smd_factor = np.clip(ratio ** effective_exponent, eff_cfg.mixture_efficiency_floor, 1.0)  # Use configurable floor

        x_star_m = float(closure_diag.get("x_star") or 0.0)
        x_star_mm = x_star_m * 1000.0
        if not np.isfinite(x_star_mm) or x_star_mm <= 0:
            xstar_factor = 1.0
        else:
            # Less conservative: reduce penalty severity
            excess = max(0.0, x_star_mm - eff_cfg.xstar_limit_mm) / max(eff_cfg.xstar_limit_mm, 1e-3)
            # CRITICAL FIX: Remove arbitrary 0.5 multiplier and 0.5 floor
            effective_exponent = eff_cfg.xstar_penalty_exponent  # Remove arbitrary 0.5 multiplier
            xstar_factor = float(np.clip(np.exp(-effective_exponent * excess), eff_cfg.mixture_efficiency_floor, 1.0))  # Use configurable floor

        we_o = float(closure_diag.get("We_O") or 0.0)
        we_f = float(closure_diag.get("We_F") or 0.0)
        we_min_actual = min(value for value in [we_o, we_f] if np.isfinite(value) and value > 0) if any(np.isfinite(value) and value > 0 for value in [we_o, we_f]) else None
        if we_min_actual is None:
            we_factor = 1.0
        else:
            # Less conservative: use square root for less severe penalty
            we_ratio = we_min_actual / eff_cfg.we_reference
            # CRITICAL FIX: Remove arbitrary 0.5 multiplier and 0.7 floor
            effective_exponent = eff_cfg.we_penalty_exponent  # Remove arbitrary 0.5 multiplier
            we_factor = np.clip(we_ratio ** effective_exponent, eff_cfg.mixture_efficiency_floor, 1.0)  # Use configurable floor
 
        turbulence_factor = 1.0
        if eff_cfg.use_turbulence_coupling:
            I_mix = float(closure_diag.get("turbulence_intensity_mix") or 0.0)
            if I_mix > 0 and eff_cfg.target_turbulence_intensity > 0:
                ratio = I_mix / eff_cfg.target_turbulence_intensity
                turbulence_factor = ratio ** eff_cfg.turbulence_penalty_exponent
            else:
                turbulence_factor = eff_cfg.turbulence_efficiency_floor
            turbulence_factor = float(np.clip(turbulence_factor, eff_cfg.turbulence_efficiency_floor, 1.0))

        mixture_eff = smd_factor * xstar_factor * we_factor * turbulence_factor
        # CRITICAL FIX: Remove arbitrary 0.85 floor - use configured floor value
        # The configured floor should be set appropriately, not overridden
        mixture_eff_floor = eff_cfg.mixture_efficiency_floor  # Use configured value, don't override
        mixture_eff = float(np.clip(mixture_eff, mixture_eff_floor, 1.0))
        return mixture_eff

    def _compute_cooling_efficiency(
        self,
        cooling_results: Dict[str, Any],
        mdot_total: float,
        Tc: float,
        gamma: float,
        R: float,
    ) -> float:
        eff_cfg = self.config.combustion.efficiency
        if not eff_cfg.use_cooling_coupling:
            return 1.0

        total_heat_removed = 0.0
        for source in cooling_results.values():
            if isinstance(source, dict):
                total_heat_removed += float(source.get("heat_removed", 0.0))

        if total_heat_removed <= 0:
            return 1.0

        cp = gamma * R / max(gamma - 1.0, 1e-6)
        available_energy = mdot_total * cp * max(Tc, 1.0)
        if available_energy <= 0:
            return 1.0

        factor = 1.0 - total_heat_removed / available_energy
        return float(np.clip(factor, eff_cfg.cooling_efficiency_floor, 1.0))

    def _evaluate_cooling_models(
        self,
        Pc: float,
        mdot_O: float,
        mdot_F: float,
        cea_props: Dict[str, float],
        closure_diag: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float, float]:
        config = self.config
        mdot_total = mdot_O + mdot_F
        cooling_results: Dict[str, Any] = {}
        Tc = float(cea_props["Tc"])

        if mdot_total <= 0:
            closure_diag["cooling"] = cooling_results
            return cooling_results, 1.0, Tc

        fuel_fluid = config.fluids["fuel"]
        geometry = self._get_chamber_geometry()
        Pc_val = float(Pc)
        gamma = float(cea_props["gamma"])
        R = float(cea_props["R"])

        rho_g = max(Pc_val / (R * max(Tc, 1.0)), 1e-6)
        area_cross = geometry["area_cross"]
        velocity_g = mdot_total / (rho_g * area_cross)

        regen_cfg = config.regen_cooling
        from engine.pipeline.constants import DEFAULT_HOT_GAS_VISC_PA_S
        mu_g_config = regen_cfg.hot_gas_viscosity if regen_cfg is not None else DEFAULT_HOT_GAS_VISC_PA_S
        
        # Calculate viscosity using Huzel's formula if molecular weight is available
        M = cea_props.get("M")  # Molecular weight [kg/kmol]
        if M is not None and M > 0 and Tc > 0:
            from engine.pipeline.thermal.regen_cooling import calculate_gas_viscosity_huzel
            mu_g_calculated = calculate_gas_viscosity_huzel(Tc, M)
        else:
            mu_g_calculated = mu_g_config  # Fallback to config if M not available
        
        # Use calculated viscosity for calculations (more accurate)
        mu_g = mu_g_calculated
        
        k_g = regen_cfg.hot_gas_thermal_conductivity if regen_cfg is not None else 0.1
        Pr_g = (
            regen_cfg.hot_gas_prandtl
            if (regen_cfg is not None and regen_cfg.hot_gas_prandtl > 0)
            else mu_g * gamma * R / max(k_g * (gamma - 1.0), 1e-6)
        )

        Re_g = rho_g * velocity_g * geometry["diameter"] / max(mu_g, 1e-8)
        if Re_g < 2000:
            Nu_g = 4.36
        else:
            Nu_g = 0.023 * (Re_g ** 0.8) * (Pr_g ** 0.4)

        turbulence_intensity_calc = 0.05
        if Re_g > 0:
            turbulence_intensity_calc = float(np.clip(0.16 * Re_g ** -0.125, 0.02, 0.25))

        turbulence_boost = 1.0
        if regen_cfg is not None:
            # CRITICAL FIX: Remove arbitrary 0.8 exponent - turbulence effect on heat transfer
            # Turbulence increases Nu, but the relationship is complex
            # For now, use linear scaling: Nu_turbulent ≈ Nu_laminar × (1 + turbulence_intensity)
            turbulence_boost = 1.0 + max(regen_cfg.gas_turbulence_intensity, 0.0)  # Remove arbitrary 0.8 exponent
            turbulence_intensity_calc = max(
                turbulence_intensity_calc,
                float(np.clip(regen_cfg.gas_turbulence_intensity, 0.0, 0.5)),
            )

        h_hot_base = Nu_g * k_g / geometry["diameter"] * turbulence_boost

        gas_props = {
            "Pc": Pc_val,
            "Tc": Tc,
            "gamma": gamma,
            "R": R,
            "rho": rho_g,
            "velocity": velocity_g,
            "length": geometry["length"],
            "circumference": geometry["circumference"],
            "area": geometry["area"],
            "area_cross": area_cross,
            "h_hot_base": h_hot_base,
            "turbulence_intensity": turbulence_intensity_calc,
        }

        film_cfg = config.film_cooling
        film_results = {
            "enabled": False,
            "mdot_available_for_regen": mdot_F,
            "effective_gas_temperature": Tc,
            "heat_removed": 0.0,
        }

        if film_cfg is not None and film_cfg.enabled:
            film_results = compute_film_cooling(
                mdot_total,
                mdot_F,
                gas_props,
                film_cfg,
                fuel_fluid,
            )
            cooling_results["film"] = film_results

        effective_Tc = float(film_results.get("effective_gas_temperature", Tc))
        gas_props_regen = {
            "Pc": Pc_val,
            "Tc": effective_Tc,
            "gamma": gamma,
            "R": R,
            "M": cea_props.get("M"),  # Molecular weight [kg/kmol] for viscosity calculation
            "chamber_area": geometry["area_cross"],
            "A_throat": ensure_chamber_geometry(config).A_throat,
            "chamber_length": geometry["length"],
            "turbulence_intensity": turbulence_intensity_calc,
        }

        coolant_props = {
            "density": float(fuel_fluid.density),
            "viscosity": float(fuel_fluid.viscosity),
            "cp": float(fuel_fluid.specific_heat),
            "thermal_conductivity": float(fuel_fluid.thermal_conductivity),
            "temperature": float(fuel_fluid.temperature),
        }

        mdot_coolant = float(film_results.get("mdot_available_for_regen", mdot_F))

        if regen_cfg is not None and regen_cfg.enabled:
            regen_results = compute_regen_heat_transfer(
                mdot_coolant,
                coolant_props,
                gas_props_regen,
                regen_cfg,
                mdot_total,
            )
            regen_results["mdot_coolant"] = mdot_coolant
            cooling_results["regen"] = regen_results

        abl_cfg = config.ablative_cooling
        if abl_cfg is not None and abl_cfg.enabled:
            hot_flux = estimate_hot_wall_heat_flux(
                gas_props_regen,
                regen_cfg,
                abl_cfg.surface_temperature_limit,
                mdot_total,
            )
            abl_area = geometry["area"] * np.clip(abl_cfg.coverage_fraction, 0.0, 1.0)
            ablative_results = compute_ablative_response(
                hot_flux["heat_flux_total"],
                abl_cfg.surface_temperature_limit,
                abl_cfg,
                abl_area,
                turbulence_intensity_calc,
                heat_flux_conv=hot_flux.get("heat_flux_conv"),
                heat_flux_rad=hot_flux.get("heat_flux_rad"),
                gas_mass_flow_rate=mdot_total,
            )
            ablative_results["incident_heat_flux"] = hot_flux["heat_flux_total"]
            
            # Calculate effective gas temperature after ablative cooling
            # Energy removed from gas: Q = mdot_total × cp × ΔT
            # Therefore: ΔT = Q / (mdot_total × cp)
            abl_heat_removed = ablative_results.get("heat_removed", 0.0)
            if abl_heat_removed > 0 and mdot_total > 0:
                cp = gamma * R / max(gamma - 1.0, 1e-6)  # Specific heat [J/(kg·K)]
                delta_T_abl = abl_heat_removed / max(mdot_total * cp, 1e-6)
                effective_Tc = max(effective_Tc - delta_T_abl, 1.0)  # Update effective temperature
                ablative_results["temperature_reduction"] = float(delta_T_abl)
            else:
                ablative_results["temperature_reduction"] = 0.0
            
            ablative_results["effective_gas_temperature"] = float(effective_Tc)
            cooling_results["ablative"] = ablative_results

        cooling_eff = self._compute_cooling_efficiency(
            cooling_results,
            mdot_total,
            effective_Tc,
            gamma,
            R,
        )

        # Store metadata for diagnostics
        metadata = cooling_results.setdefault("metadata", {})
        metadata["gas_turbulence_intensity"] = turbulence_intensity_calc
        metadata["effective_gas_temperature"] = float(effective_Tc)
        metadata["original_gas_temperature"] = float(Tc)
        metadata["gas_viscosity"] = float(mu_g)  # Viscosity used in calculations (calculated from Huzel if available)
        metadata["gas_viscosity_config"] = float(mu_g_config)  # Viscosity from config (for reference)
        metadata["gas_viscosity_calculated"] = float(mu_g_calculated)  # Viscosity from Huzel formula (for reference)
        closure_diag["cooling"] = cooling_results

        return cooling_results, cooling_eff, effective_Tc
    
    def _infer_injector_diameter(self) -> float:
        """Estimate a characteristic injector diameter for mixing models."""
        injector_cfg = getattr(self.config, "injector", None)
        diameter = None
        injector_type = getattr(injector_cfg, "type", None) if injector_cfg is not None else None
        try:
            if injector_type == "pintle":
                diameter = injector_cfg.geometry.fuel.d_pintle_tip
            elif injector_type == "coaxial":
                diameter = injector_cfg.geometry.core.d_port
            elif injector_type == "impinging":
                diameter = injector_cfg.geometry.oxidizer.d_jet
        except AttributeError:
            diameter = None
        
        if diameter is None or diameter <= 0:
            geometry = self._get_chamber_geometry()
            diameter = np.sqrt(4.0 * geometry["area_cross"] / np.pi)
        return float(max(diameter, 1e-5))

    def _get_chamber_geometry(self) -> Dict[str, float]:
        """
        Extract physical chamber geometry from configuration.
        
        Returns a dictionary with:
        - length: Total physical length [m]
        - diameter: Chamber inner diameter [m]
        - area_cross: Cross-sectional area [m²]
        - circumference: Chamber circumference [m]
        - area: Total wetted surface area [m²] (cylindrical + contraction)
        """
        # cg is guaranteed to exist because __init__ calls ensure_chamber_geometry
        cg = ensure_chamber_geometry(self.config)
        regen_cfg = self.config.regen_cooling

        # 1. Physical diameter from unified config
        diameter = cg.chamber_diameter
        
        # Fallback to regen if not in unified (though ensure_chamber_geometry should handle it)
        if (diameter is None or diameter <= 0) and regen_cfg is not None and regen_cfg.chamber_inner_diameter is not None:
            diameter = regen_cfg.chamber_inner_diameter
            
        # Final fallback
        if diameter is None or diameter <= 0:
            diameter = 0.08
            
        diameter = max(diameter, 1e-6)
        area_cross = np.pi * (diameter / 2.0)**2
        circumference = np.pi * diameter
        
        # 2. Physical lengths
        length_total = cg.length
        length_cyl = cg.length_cylindrical
        length_cont = cg.length_contraction
        
        # 3. Wetted Surface Area
        # If we have the breakdown (cylindrical + contraction), calculate accurately
        if length_cyl is not None and length_cont is not None:
            # Wetted area = Cylindrical part + Contraction part (frustum of a cone)
            # Area_cyl = pi * D * L_cyl
            area_cyl = circumference * length_cyl
            
            # Area_cont = lateral area of a frustum = pi * (r1 + r2) * slant_height
            r1 = diameter / 2.0
            # Estimate throat radius from A_throat if available
            A_throat = cg.A_throat if cg.A_throat and cg.A_throat > 0 else (area_cross / 3.0)
            r2 = np.sqrt(A_throat / np.pi)
            
            slant_height = np.sqrt((r1 - r2)**2 + length_cont**2)
            area_cont = np.pi * (r1 + r2) * slant_height
            
            area_wetted = area_cyl + area_cont
        else:
            # Fallback to simple cylinder if breakdown not available
            area_wetted = circumference * length_total

        return {
            "length": float(length_total),
            "diameter": float(diameter),
            "area_cross": float(area_cross),
            "circumference": float(circumference),
            "area": float(area_wetted),
        }


    def _get_fuel_props(self) -> Optional[Dict[str, float]]:
        """
        Extract fuel properties from configuration for evaporation model.
        
        Returns a dictionary with:
        - boiling_point: Fuel boiling point [K]
        - latent_heat: Latent heat of vaporization [J/kg]
        - molecular_weight: Molecular weight [g/mol]
        - Pc_ref: Reference pressure for stable Bm calculation [Pa]
        
        Returns None if fuel config is not available.
        """
        try:
            fuel_cfg = self.config.fluids.get("fuel")
            if fuel_cfg is None:
                return None
            
            # Extract as dict with fallbacks to RP-1 defaults
            return {
                "boiling_point": getattr(fuel_cfg, "boiling_point", 489.0),
                "latent_heat": getattr(fuel_cfg, "latent_heat", 300e3),
                "molecular_weight": getattr(fuel_cfg, "molecular_weight", 170.0),
                "Pc_ref": getattr(fuel_cfg, "Pc_ref", 2.5e6),
            }
        except Exception:
            return None
