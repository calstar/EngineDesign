"""Main pipeline orchestrator - runs full tank pressure to thrust pipeline"""

import numpy as np
from typing import Dict, Any, Optional, Union
from pintle_pipeline.config_schemas import PintleEngineConfig
from pintle_pipeline.cea_cache import CEACache
from pintle_models.chamber_solver import ChamberSolver
from pintle_models.nozzle import calculate_thrust
from pintle_models.closure import flows
from pintle_pipeline.film_cooling import compute_film_cooling
from pintle_pipeline.regen_cooling import compute_regen_heat_transfer, estimate_hot_wall_heat_flux
from pintle_pipeline.ablative_cooling import compute_ablative_response
from pintle_pipeline.regen_cooling import RegenCoolingConfig


class PintleEngineRunner:
    """Main pipeline runner - orchestrates full tank pressure to thrust calculation"""
    
    def __init__(self, config: PintleEngineConfig):
        """
        Initialize pipeline runner.
        
        Parameters:
        -----------
        config : PintleEngineConfig
            Engine configuration
        """
        self.config = config
        
        # Initialize CEA cache
        self.cea_cache = CEACache(config.combustion.cea)
        
        # Initialize chamber solver
        self.solver = ChamberSolver(config, self.cea_cache)
    
    def evaluate(
        self,
        P_tank_O: float,
        P_tank_F: float,
        Pc_guess: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate engine performance at given tank pressures.
        
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
        results : dict
            Dictionary containing all performance metrics:
            - Pc: Chamber pressure [Pa]
            - mdot_O: Oxidizer mass flow [kg/s]
            - mdot_F: Fuel mass flow [kg/s]
            - MR: Mixture ratio (O/F)
            - F: Thrust [N]
            - Isp: Specific impulse [s]
            - cstar_actual: Actual characteristic velocity [m/s]
            - diagnostics: Detailed diagnostics dict
        """
        # Solve for chamber pressure
        Pc, diagnostics = self.solver.solve(P_tank_O, P_tank_F, Pc_guess)
        
        # Get final mass flow rates
        mdot_O = diagnostics["mdot_O"]
        mdot_F = diagnostics["mdot_F"]
        mdot_total = mdot_O + mdot_F
        MR = diagnostics["MR"]
        
        # Get CEA properties for nozzle calculation
        cea_props = self.cea_cache.eval(MR, Pc)
        cstar_actual = diagnostics["cstar_actual"]
        gamma = diagnostics["gamma"]
        R = diagnostics["R"]
        Tc = diagnostics["Tc"]
        
        # Calculate thrust
        # Use sea level ambient pressure (101325 Pa) as default
        Pa = 101325.0  # Pa - Ambient pressure (sea level)
        
        thrust_results = calculate_thrust(
            Pc,
            MR,
            mdot_total,
            self.cea_cache,
            self.config.nozzle,
            Pa
        )
        
        F = thrust_results["F"]
        Isp = thrust_results["Isp"]
        v_exit = thrust_results["v_exit"]
        P_exit = thrust_results["P_exit"]

        cooling_results = {}
        film_results = {
            "enabled": False,
            "mass_fraction": 0.0,
            "mdot_film": 0.0,
            "mdot_available_for_regen": mdot_F,
            "effectiveness": 0.0,
            "heat_flux_factor": 1.0,
            "film_temperature": self.config.fluids["fuel"].temperature,
        }

        fuel_fluid = self.config.fluids["fuel"]

        if self.config.film_cooling is not None:
            film_results = compute_film_cooling(
                mdot_total,
                mdot_F,
                diagnostics["Tc"],
                self.config.film_cooling,
                fuel_fluid.temperature,
            )
            cooling_results["film"] = film_results

        regen_results = {
            "enabled": False,
        }

        if self.config.regen_cooling is not None and self.config.regen_cooling.enabled:
            coolant_props = {
                "density": float(fuel_fluid.density),
                "viscosity": float(fuel_fluid.viscosity),
                "cp": float(fuel_fluid.specific_heat),
                "thermal_conductivity": float(fuel_fluid.thermal_conductivity),
                "temperature": float(fuel_fluid.temperature),
            }

            chamber_area = None
            if self.config.regen_cooling.channel_length > 0:
                chamber_area = self.config.chamber.volume / self.config.regen_cooling.channel_length

            gas_props = {
                "Pc": Pc,
                "Tc": diagnostics["Tc"],
                "gamma": diagnostics["gamma"],
                "R": diagnostics["R"],
                "chamber_area": chamber_area,
                "A_throat": self.config.chamber.A_throat,
            }

            mdot_coolant = max(mdot_F, 1e-9)

            regen_results = compute_regen_heat_transfer(
                mdot_coolant,
                coolant_props,
                gas_props,
                self.config.regen_cooling,
                mdot_total,
            )
            regen_results["mdot_coolant"] = mdot_coolant
            cooling_results["regen"] = regen_results

        if self.config.ablative_cooling is not None and self.config.ablative_cooling.enabled:
            abl_surface_temp = self.config.ablative_cooling.surface_temperature_limit
            hot_flux = estimate_hot_wall_heat_flux(
                gas_props,
                self.config.regen_cooling,
                abl_surface_temp,
                mdot_total,
            )
            net_heat_flux = hot_flux["heat_flux_total"]
            surface_temp = abl_surface_temp
            ablative_results = compute_ablative_response(
                net_heat_flux,
                surface_temp,
                diagnostics["Tc"],
                self.config.ablative_cooling,
            )
            cooling_results["ablative"] = ablative_results

        diagnostics["cooling"] = cooling_results
        
        # Compile results
        results = {
            "Pc": Pc,
            "mdot_O": mdot_O,
            "mdot_F": mdot_F,
            "mdot_total": mdot_total,
            "MR": MR,
            "F": F,
            "Isp": Isp,
            "v_exit": v_exit,
            "P_exit": P_exit,
            "cstar_actual": cstar_actual,
            "cstar_ideal": diagnostics["cstar_ideal"],
            "eta_cstar": diagnostics["eta_cstar"],
            "Tc": Tc,
            "gamma": gamma,
            "R": R,
            "cooling": cooling_results,
            "diagnostics": diagnostics,
        }
        
        return results
    
    def evaluate_arrays(
        self,
        P_tank_O: Union[np.ndarray, list],
        P_tank_F: Union[np.ndarray, list]
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate engine performance for arrays of tank pressures (time series).
        
        Parameters:
        -----------
        P_tank_O : array-like
            Array of oxidizer tank pressures [Pa]
        P_tank_F : array-like
            Array of fuel tank pressures [Pa]
        
        Returns:
        --------
        results : dict
            Dictionary with arrays of all performance metrics
        """
        P_tank_O = np.asarray(P_tank_O)
        P_tank_F = np.asarray(P_tank_F)
        
        if P_tank_O.shape != P_tank_F.shape:
            raise ValueError("P_tank_O and P_tank_F must have same shape")
        
        # Initialize result arrays
        n = P_tank_O.size
        results = {
            "Pc": np.full(n, np.nan),
            "mdot_O": np.full(n, np.nan),
            "mdot_F": np.full(n, np.nan),
            "mdot_total": np.full(n, np.nan),
            "MR": np.full(n, np.nan),
            "F": np.full(n, np.nan),
            "Isp": np.full(n, np.nan),
            "v_exit": np.full(n, np.nan),
            "P_exit": np.full(n, np.nan),
            "cstar_actual": np.full(n, np.nan),
            "cstar_ideal": np.full(n, np.nan),
            "eta_cstar": np.full(n, np.nan),
            "Tc": np.full(n, np.nan),
            "gamma": np.full(n, np.nan),
            "R": np.full(n, np.nan),
            "diagnostics": [],
        }
        
        # Evaluate at each point
        for i in range(n):
            try:
                point_results = self.evaluate(
                    float(P_tank_O.flat[i]),
                    float(P_tank_F.flat[i])
                )
                
                # Store scalar results
                for key in ["Pc", "mdot_O", "mdot_F", "mdot_total", "MR", "F", "Isp",
                           "v_exit", "P_exit", "cstar_actual", "cstar_ideal", "eta_cstar",
                           "Tc", "gamma", "R"]:
                    results[key][i] = point_results[key]
                
                # Store diagnostics
                results["diagnostics"].append(point_results["diagnostics"])
                
            except Exception as e:
                # If solve fails, leave NaN values
                results["diagnostics"].append({"error": str(e)})
                continue
        
        return results
