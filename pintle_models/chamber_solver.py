"""Chamber pressure solver: solve supply(Pc) = demand(Pc)"""

import numpy as np
from scipy.optimize import brentq, newton
from typing import Tuple, Dict, Any, Callable
from pintle_pipeline.config_schemas import PintleEngineConfig
from pintle_pipeline.feed_loss import delta_p_feed
from pintle_pipeline.combustion_eff import eta_cstar, calculate_Lstar
from pintle_pipeline.cea_cache import CEACache
from pintle_models.closure import flows


class ChamberSolver:
    """Solves for chamber pressure by balancing supply and demand"""
    
    def __init__(self, config: PintleEngineConfig, cea_cache: CEACache):
        self.config = config
        self.cea_cache = cea_cache
        
        # Calculate L* once (use config value if provided, otherwise calculate)
        self.Lstar = calculate_Lstar(
            config.chamber.volume,
            config.chamber.A_throat,
            Lstar_override=config.chamber.Lstar
        )
        
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
        # Supply side: mass flow from injector (via closure)
        # flows() takes TANK PRESSURES and solves for mdot
        # It internally calculates:
        #   1. Feed losses: P_tank → P_injector
        #   2. Injector flow: P_injector - Pc → mdot
        #   3. Spray constraints: validates and adjusts if needed
        mdot_O, mdot_F, diagnostics = flows(
            P_tank_O,  # Tank pressure (INPUT)
            P_tank_F,  # Tank pressure (INPUT)
            Pc,        # Chamber pressure (GUESS - being solved for)
            self.config
        )
        
        mdot_supply = mdot_O + mdot_F
        
        # Update spray quality for efficiency calculation
        self.spray_quality_good = diagnostics["constraints_satisfied"]
        
        # Demand side: mass flow required by combustion
        MR = mdot_O / mdot_F if mdot_F > 0 else 2.5  # Default MR
        
        # Get CEA properties (IDEAL - infinite area equilibrium)
        cea_props = self.cea_cache.eval(MR, Pc)
        cstar_ideal = cea_props["cstar_ideal"]
        
        # Apply combustion efficiency for FINITE CHAMBER
        # This corrects CEA's infinite-area assumption
        eta = eta_cstar(
            self.Lstar,
            self.config.combustion.efficiency,
            self.spray_quality_good
        )
        
        # Actual c* accounting for finite chamber volume
        cstar_actual = eta * cstar_ideal
        
        # Demand: mdot = Pc * At / c*_actual
        # This uses the chamber-driven c*, not the ideal CEA value
        if cstar_actual > 0:
            mdot_demand = (Pc * self.config.chamber.A_throat) / cstar_actual
        else:
            mdot_demand = 0.0
        
        residual = mdot_supply - mdot_demand
        return residual
    
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
        
        # brentq requires opposite signs at bounds
        if np.sign(residual_min) == np.sign(residual_max):
            # No root in interval - this happens when:
            # 1. Supply > demand at all Pc (both positive) - need higher Pc but limited by tank pressure
            # 2. Supply < demand at all Pc (both negative) - can't supply enough flow
            if residual_min > 0 and residual_max > 0:
                raise ValueError(
                    f"No solution: Supply > Demand at all Pc. "
                    f"Residual at bounds: [{residual_min:.4f}, {residual_max:.4f}] kg/s. "
                    f"Pc_max ({Pc_max/1e6:.2f} MPa) limited by tank pressure. "
                    f"Try increasing P_tank_O or reducing P_tank_F."
                )
            else:
                raise ValueError(
                    f"No solution: Supply < Demand at all Pc. "
                    f"Residual at bounds: [{residual_min:.4f}, {residual_max:.4f}] kg/s. "
                    f"Insufficient mass flow. Check tank pressures and injector geometry."
                )
        
        # Solve using bracketed secant (brentq) - safe and robust
        try:
            if self.config.solver.method == "brentq":
                Pc, result = brentq(
                    residual_func,
                    Pc_min,
                    Pc_max,
                    xtol=self.config.solver.tolerance,
                    maxiter=self.config.solver.max_iterations,
                    full_output=True
                )
                success = result.converged
            else:
                # Fallback to Newton's method (less robust)
                Pc = newton(
                    residual_func,
                    Pc_guess,
                    tol=self.config.solver.tolerance,
                    maxiter=self.config.solver.max_iterations
                )
                success = True
        except Exception as e:
            raise RuntimeError(f"Chamber pressure solver failed: {e}")
        
        # Get final diagnostics
        mdot_O, mdot_F, closure_diag = flows(P_tank_O, P_tank_F, Pc, self.config)
        MR = mdot_O / mdot_F if mdot_F > 0 else 0
        
        cea_props = self.cea_cache.eval(MR, Pc)
        eta = eta_cstar(
            self.Lstar,
            self.config.combustion.efficiency,
            self.spray_quality_good
        )
        
        diagnostics = {
            "Pc": Pc,
            "mdot_O": mdot_O,
            "mdot_F": mdot_F,
            "mdot_total": mdot_O + mdot_F,
            "MR": MR,
            "cstar_ideal": cea_props["cstar_ideal"],
            "cstar_actual": eta * cea_props["cstar_ideal"],
            "eta_cstar": eta,
            "Tc": cea_props["Tc"],
            "gamma": cea_props["gamma"],
            "R": cea_props["R"],
            "M": cea_props.get("M", 0),  # Mach number if available
            "spray_quality_good": self.spray_quality_good,
            **closure_diag,
        }
        
        return Pc, diagnostics


