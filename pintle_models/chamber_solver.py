"""Chamber pressure solver: solve supply(Pc) = demand(Pc)"""

import numpy as np
from scipy.optimize import brentq, newton
from typing import Tuple, Dict, Any

from pintle_pipeline.config_schemas import PintleEngineConfig
from pintle_pipeline.combustion_eff import eta_cstar, calculate_Lstar
from pintle_pipeline.cea_cache import CEACache
from pintle_pipeline.film_cooling import compute_film_cooling
from pintle_pipeline.regen_cooling import (
    compute_regen_heat_transfer,
    estimate_hot_wall_heat_flux,
)
from pintle_pipeline.ablative_cooling import compute_ablative_response
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
        mixture_eff = self._compute_mixture_efficiency(diagnostics)

        cooling_results, cooling_eff = self._evaluate_cooling_models(
            Pc,
            mdot_O,
            mdot_F,
            cea_props,
            diagnostics,
        )

        eta = eta_cstar(
            self.Lstar,
            self.config.combustion.efficiency,
            self.spray_quality_good,
            mixture_efficiency=mixture_eff,
            cooling_efficiency=cooling_eff,
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
        mixture_eff = self._compute_mixture_efficiency(closure_diag)
        cooling_results, cooling_eff = self._evaluate_cooling_models(
            Pc,
            mdot_O,
            mdot_F,
            cea_props,
            closure_diag,
        )

        eta = eta_cstar(
            self.Lstar,
            self.config.combustion.efficiency,
            self.spray_quality_good,
            mixture_efficiency=mixture_eff,
            cooling_efficiency=cooling_eff,
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
            "mixture_efficiency": mixture_eff,
            "cooling_efficiency": cooling_eff,
            "Tc": cea_props["Tc"],
            "gamma": cea_props["gamma"],
            "R": cea_props["R"],
            "M": cea_props.get("M"),  # Molecular weight [kg/kmol]
            "spray_quality_good": self.spray_quality_good,
            **closure_diag,
        }

        diagnostics["cooling"] = cooling_results

        return Pc, diagnostics

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
            ratio = eff_cfg.target_smd_microns / max(actual_smd, 1e-9)
            smd_factor = np.clip(ratio ** eff_cfg.smd_penalty_exponent, 0.0, 1.0)

        x_star_m = float(closure_diag.get("x_star") or 0.0)
        x_star_mm = x_star_m * 1000.0
        if not np.isfinite(x_star_mm) or x_star_mm <= 0:
            xstar_factor = 1.0
        else:
            excess = max(0.0, x_star_mm - eff_cfg.xstar_limit_mm) / max(eff_cfg.xstar_limit_mm, 1e-3)
            xstar_factor = float(np.exp(-eff_cfg.xstar_penalty_exponent * excess))

        we_o = float(closure_diag.get("We_O") or 0.0)
        we_f = float(closure_diag.get("We_F") or 0.0)
        we_min_actual = min(value for value in [we_o, we_f] if np.isfinite(value) and value > 0) if any(np.isfinite(value) and value > 0 for value in [we_o, we_f]) else None
        if we_min_actual is None:
            we_factor = 1.0
        else:
            we_ratio = we_min_actual / eff_cfg.we_reference
            we_factor = np.clip(we_ratio ** eff_cfg.we_penalty_exponent, 0.0, 1.0)
 
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
        mixture_eff = float(np.clip(mixture_eff, eff_cfg.mixture_efficiency_floor, 1.0))
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
    ) -> Tuple[Dict[str, Any], float]:
        config = self.config
        mdot_total = mdot_O + mdot_F
        cooling_results: Dict[str, Any] = {}

        if mdot_total <= 0:
            closure_diag["cooling"] = cooling_results
            return cooling_results, 1.0

        fuel_fluid = config.fluids["fuel"]
        geometry = self._get_chamber_geometry()

        Tc = float(cea_props["Tc"])
        Pc_val = float(Pc)
        gamma = float(cea_props["gamma"])
        R = float(cea_props["R"])

        rho_g = max(Pc_val / (R * max(Tc, 1.0)), 1e-6)
        area_cross = geometry["area_cross"]
        velocity_g = mdot_total / (rho_g * area_cross)

        regen_cfg = config.regen_cooling
        mu_g = regen_cfg.hot_gas_viscosity if regen_cfg is not None else 4.0e-5
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
            turbulence_boost = (1.0 + max(regen_cfg.gas_turbulence_intensity, 0.0)) ** 0.8
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
            "chamber_area": geometry["area_cross"],
            "A_throat": config.chamber.A_throat,
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
                effective_Tc,
                abl_cfg,
                abl_area,
                turbulence_intensity_calc,
            )
            ablative_results["incident_heat_flux"] = hot_flux["heat_flux_total"]
            cooling_results["ablative"] = ablative_results

        cooling_eff = self._compute_cooling_efficiency(
            cooling_results,
            mdot_total,
            effective_Tc,
            gamma,
            R,
        )

        cooling_results.setdefault("metadata", {})["gas_turbulence_intensity"] = turbulence_intensity_calc
        closure_diag["cooling"] = cooling_results

        return cooling_results, cooling_eff

    def _get_chamber_geometry(self) -> Dict[str, float]:
        chamber_cfg = self.config.chamber
        regen_cfg = self.config.regen_cooling

        length = chamber_cfg.length
        if length is None and regen_cfg is not None and regen_cfg.channel_length > 0:
            length = regen_cfg.channel_length
        if length is None:
            length = chamber_cfg.volume / max(chamber_cfg.A_throat, 1e-6)
        length = max(length, 1e-6)

        diameter = None
        if regen_cfg is not None and regen_cfg.chamber_inner_diameter is not None:
            diameter = regen_cfg.chamber_inner_diameter

        if diameter is None and chamber_cfg.volume > 0:
            area_cross = chamber_cfg.volume / length
            diameter = np.sqrt(max(4.0 * area_cross / np.pi, 1e-8))
        else:
            area_cross = np.pi * (max(diameter, 1e-6) ** 2) / 4.0

        diameter = max(diameter, 1e-6)
        circumference = np.pi * diameter
        area = circumference * length

        return {
            "length": float(length),
            "diameter": float(diameter),
            "area_cross": float(area_cross),
            "circumference": float(circumference),
            "area": float(area),
        }


