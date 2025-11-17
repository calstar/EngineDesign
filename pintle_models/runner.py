"""Main pipeline orchestrator - runs full tank pressure to thrust pipeline"""

import numpy as np
from typing import Dict, Any, Optional, Union
import copy

from pintle_pipeline.config_schemas import PintleEngineConfig
from pintle_pipeline.cea_cache import CEACache
from pintle_models.chamber_solver import ChamberSolver
from pintle_models.nozzle import calculate_thrust
from pintle_models.chamber_profiles import (
    calculate_chamber_pressure_profile,
    calculate_chamber_intrinsics,
)
from pintle_pipeline.ablative_geometry import (
    update_chamber_geometry_from_ablation,
    update_nozzle_exit_from_ablation,
    calculate_throat_recession_multiplier as calculate_ablative_throat_recession_multiplier,
    calculate_local_recession_rate,
)
from pintle_pipeline.graphite_cooling import (
    compute_graphite_recession,
    calculate_throat_recession_multiplier,
)
from pintle_pipeline.constants import DEFAULT_GAMMA_ND


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
        # Calculate current expansion ratio (for 3D CEA cache)
        eps_current = self.config.nozzle.A_exit / self.config.chamber.A_throat
        cea_props = self.cea_cache.eval(MR, Pc, 101325.0, eps_current)
        cstar_actual = diagnostics["cstar_actual"]
        gamma = diagnostics["gamma"]
        R = diagnostics["R"]
        Tc = diagnostics["Tc"]
        
        # Calculate thrust
        # Use sea level ambient pressure (101325 Pa) as default
        Pa = 101325.0  # Pa - Ambient pressure (sea level)
        
        # Calculate current expansion ratio (for 3D CEA cache)
        eps_current = self.config.nozzle.A_exit / self.config.chamber.A_throat
        
        # Check if shifting equilibrium is enabled
        use_shifting = getattr(self.config.combustion.efficiency, 'use_shifting_equilibrium', True)
        reaction_progress = diagnostics.get("reaction_progress", None)
        
        thrust_results = calculate_thrust(
            Pc,
            MR,
            mdot_total,
            self.cea_cache,
            self.config.nozzle,
            Pa,
            eps=eps_current,  # Pass current expansion ratio
            reaction_progress=reaction_progress,  # Pass reaction progress for shifting equilibrium
            use_shifting_equilibrium=use_shifting,
            config=self.config,
        )
        
        F = thrust_results["F"]
        Isp = thrust_results["Isp"]
        v_exit = thrust_results["v_exit"]
        P_exit = thrust_results["P_exit"]
        P_throat = thrust_results.get("P_throat", Pc * 0.6)  # Throat pressure
        T_exit = thrust_results.get("T_exit", Tc * 0.5)  # Exit temperature
        T_throat = thrust_results.get("T_throat", Tc * 0.85)  # Throat temperature
        Cf_actual = thrust_results.get("Cf_actual", thrust_results.get("Cf", 0.0))
        Cf_ideal = thrust_results.get("Cf_ideal", 0.0)
        Cf_theoretical = thrust_results.get("Cf_theoretical", 0.0)
        temperature_profile = thrust_results.get("temperature_profile", None)

        cooling_results = diagnostics.get("cooling", {})
        
        # Calculate chamber pressure profile along length
        pressure_profile = None
        try:
            pressure_profile = calculate_chamber_pressure_profile(
                Pc=Pc,
                Lstar=self.solver.Lstar,
                mdot_total=mdot_total,
                gamma=gamma,
                R=R,
                Tc=Tc,
                A_throat=self.config.chamber.A_throat,
                n_points=30,
            )
        except Exception as e:
            import warnings
            warnings.warn(f"Pressure profile calculation failed: {e}")
        
        # Calculate chamber intrinsics
        chamber_intrinsics = None
        try:
            chamber_intrinsics = calculate_chamber_intrinsics(
                Pc=Pc,
                Tc=Tc,
                mdot_total=mdot_total,
                gamma=gamma,
                R=R,
                V_chamber=self.config.chamber.volume,
                A_throat=self.config.chamber.A_throat,
                Lstar=self.solver.Lstar,
                MR=MR,
            )
        except Exception as e:
            import warnings
            warnings.warn(f"Chamber intrinsics calculation failed: {e}")
        
        # Extract injector pressure diagnostics from closure diagnostics
        injector_pressure_diagnostics = {
            "P_injector_O": diagnostics.get("P_injector_O"),
            "P_injector_F": diagnostics.get("P_injector_F"),
            "delta_p_injector_O": diagnostics.get("delta_p_injector_O"),
            "delta_p_injector_F": diagnostics.get("delta_p_injector_F"),
            "delta_p_feed_O": diagnostics.get("delta_p_feed_O"),
            "delta_p_feed_F": diagnostics.get("delta_p_feed_F"),
        }
        
        # Extract discharge coefficients
        Cd_O = diagnostics.get("Cd_O", np.nan)
        Cd_F = diagnostics.get("Cd_F", np.nan)
        
        # Calculate stability analysis if enabled
        stability_results = None
        try:
            from pintle_pipeline.stability_analysis import comprehensive_stability_analysis
            
            stability_results = comprehensive_stability_analysis(
                config=self.config,
                Pc=Pc,
                MR=MR,
                mdot_total=mdot_total,
                cstar=cstar_actual,
                gamma=gamma,
                R=R,
                Tc=Tc,
                diagnostics=diagnostics,
            )
        except Exception as e:
            import warnings
            warnings.warn(f"Stability analysis failed: {e}")
        
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
            "P_throat": P_throat,
            "T_exit": T_exit,
            "T_throat": T_throat,
            "Tc": Tc,
            "Cf": Cf_actual,  # Actual measured Cf
            "Cf_actual": Cf_actual,
            "Cf_ideal": Cf_ideal,
            "Cf_theoretical": Cf_theoretical,
            "temperature_profile": temperature_profile,
            "eps": eps_current,  # Expansion ratio (for 3D CEA cache)
            "A_throat": self.config.chamber.A_throat,
            "A_exit": self.config.nozzle.A_exit,
            "cstar_actual": cstar_actual,
            "cstar_ideal": diagnostics["cstar_ideal"],
            "eta_cstar": diagnostics["eta_cstar"],
            "gamma": gamma,
            "R": R,
            "Cd_O": Cd_O,
            "Cd_F": Cd_F,
            "cooling": cooling_results,
            "stability": stability_results,
            "pressure_profile": pressure_profile,
            "chamber_intrinsics": chamber_intrinsics,
            "injector_pressure": injector_pressure_diagnostics,
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
            "eps": np.full(n, np.nan),  # Expansion ratio
            "A_throat": np.full(n, np.nan),
            "A_exit": np.full(n, np.nan),
            "cstar_actual": np.full(n, np.nan),
            "cstar_ideal": np.full(n, np.nan),
            "eta_cstar": np.full(n, np.nan),
            "Tc": np.full(n, np.nan),
            "gamma": np.full(n, np.nan),
            "R": np.full(n, np.nan),
            "Cd_O": np.full(n, np.nan),
            "Cd_F": np.full(n, np.nan),
            "diagnostics": [],
        }
        
        # Evaluate at each point
        for i in range(n):
            try:
                point_results = self.evaluate(
                    float(P_tank_O.flat[i]),
                    float(P_tank_F.flat[i])
                )
                
                # Store scalar results (including eps for 3D CEA cache)
                for key in ["Pc", "mdot_O", "mdot_F", "mdot_total", "MR", "F", "Isp",
                           "v_exit", "P_exit", "eps", "A_throat", "A_exit",
                           "cstar_actual", "cstar_ideal", "eta_cstar",
                           "Tc", "gamma", "R", "Cd_O", "Cd_F"]:
                    results[key][i] = point_results[key]
                
                # Store diagnostics
                results["diagnostics"].append(point_results["diagnostics"])
                
            except Exception as e:
                # If solve fails, leave NaN values
                results["diagnostics"].append({"error": str(e)})
                continue
        
        return results
    
    def evaluate_arrays_with_time(
        self,
        times: Union[np.ndarray, list],
        P_tank_O: Union[np.ndarray, list],
        P_tank_F: Union[np.ndarray, list],
        track_ablative_geometry: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate engine performance over time with ablative geometry evolution.
        
        This method tracks cumulative ablative recession and updates chamber
        geometry (V_chamber, A_throat, L*) at each time step, providing
        accurate performance predictions for ablative engines.
        
        Parameters:
        -----------
        times : array-like
            Time points [s]
        P_tank_O : array-like
            Array of oxidizer tank pressures [Pa]
        P_tank_F : array-like
            Array of fuel tank pressures [Pa]
        track_ablative_geometry : bool, optional
            Override config setting for geometry tracking
        
        Returns:
        --------
        results : dict
            Dictionary with arrays of all performance metrics plus:
            - Lstar: Time-varying characteristic length [m]
            - V_chamber: Time-varying chamber volume [m³]
            - A_throat: Time-varying throat area [m²]
            - recession_chamber: Cumulative chamber recession [m]
            - recession_throat: Cumulative throat recession [m]
        """
        times = np.asarray(times)
        P_tank_O = np.asarray(P_tank_O)
        P_tank_F = np.asarray(P_tank_F)
        
        if times.shape != P_tank_O.shape or times.shape != P_tank_F.shape:
            raise ValueError("times, P_tank_O, and P_tank_F must have same shape")
        
        if len(times) < 2:
            raise ValueError("Need at least 2 time points for time-varying analysis")
        
        # Check if ablative geometry tracking is enabled
        ablative_cfg = self.config.ablative_cooling
        if track_ablative_geometry is None:
            track_ablative_geometry = (
                ablative_cfg is not None 
                and ablative_cfg.enabled 
                and ablative_cfg.track_geometry_evolution
            )
        
        # Initialize result arrays
        n = len(times)
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
            "Cd_O": np.full(n, np.nan),
            "Cd_F": np.full(n, np.nan),
            "Lstar": np.full(n, np.nan),
            "V_chamber": np.full(n, np.nan),
            "A_throat": np.full(n, np.nan),
            "A_exit": np.full(n, np.nan),
            "eps": np.full(n, np.nan),  # Expansion ratio
            "recession_chamber": np.full(n, 0.0),
            "recession_throat": np.full(n, 0.0),
            "recession_exit": np.full(n, 0.0),
            "throat_recession_multiplier": np.full(n, np.nan),
            "diagnostics": [],
        }
        
        # Initial geometry
        V_chamber_initial = self.config.chamber.volume
        A_throat_initial = self.config.chamber.A_throat
        A_exit_initial = self.config.nozzle.A_exit
        L_chamber = self.config.chamber.length if self.config.chamber.length else 0.18
        D_chamber_initial = np.sqrt(4 * V_chamber_initial / (np.pi * L_chamber))
        D_throat_initial = np.sqrt(4 * A_throat_initial / np.pi)
        D_exit_initial = np.sqrt(4 * A_exit_initial / np.pi)
        
        # Track cumulative recession
        cumulative_recession_chamber = 0.0
        cumulative_recession_throat = 0.0
        cumulative_recession_exit = 0.0
        
        # Create a mutable config copy for geometry updates
        config_copy = copy.deepcopy(self.config)
        
        # Evaluate at each time point
        for i in range(n):
            dt = times[i] - times[i-1] if i > 0 else 0.0
            
            try:
                # Update solver with current geometry
                solver_temp = ChamberSolver(config_copy, self.cea_cache)
                
                # Evaluate performance using the updated solver
                # (NOT self.evaluate which uses the original geometry!)
                Pc, diagnostics = solver_temp.solve(
                    float(P_tank_O[i]),
                    float(P_tank_F[i]),
                    Pc_guess=None
                )
                
                # Calculate thrust and performance
                Pa = 101325.0  # Ambient pressure
                
                # Calculate current expansion ratio (for 3D CEA cache)
                eps_current = config_copy.nozzle.A_exit / config_copy.chamber.A_throat
                
                thrust_results = calculate_thrust(
                    Pc,
                    diagnostics["MR"],
                    diagnostics["mdot_total"],
                    self.cea_cache,
                    config_copy.nozzle,
                    Pa,
                    eps=eps_current  # Pass current expansion ratio
                )
                thrust = thrust_results["F"]
                v_exit = thrust_results["v_exit"]
                P_exit = thrust_results["P_exit"]
                
                # Package results like evaluate() does
                point_results = {
                    "Pc": Pc,
                    "mdot_O": diagnostics["mdot_O"],
                    "mdot_F": diagnostics["mdot_F"],
                    "mdot_total": diagnostics["mdot_total"],
                    "MR": diagnostics["MR"],
                    "F": thrust,
                    "Isp": thrust / (diagnostics["mdot_total"] * 9.80665) if diagnostics["mdot_total"] > 0 else 0.0,
                    "v_exit": v_exit,
                    "P_exit": P_exit,
                    "cstar_actual": diagnostics["cstar_actual"],
                    "cstar_ideal": diagnostics["cstar_ideal"],
                    "eta_cstar": diagnostics["eta_cstar"],
                    "Tc": diagnostics["Tc"],
                    "gamma": diagnostics["gamma"],
                    "R": diagnostics["R"],
                    "Cd_O": diagnostics.get("Cd_O", np.nan),
                    "Cd_F": diagnostics.get("Cd_F", np.nan),
                    "diagnostics": diagnostics,
                }
                
                # Store scalar results
                for key in ["Pc", "mdot_O", "mdot_F", "mdot_total", "MR", "F", "Isp",
                           "v_exit", "P_exit", "cstar_actual", "cstar_ideal", "eta_cstar",
                           "Tc", "gamma", "R", "Cd_O", "Cd_F"]:
                    results[key][i] = point_results[key]
                
                # Store current geometry
                results["Lstar"][i] = solver_temp.Lstar
                results["V_chamber"][i] = config_copy.chamber.volume
                results["A_throat"][i] = config_copy.chamber.A_throat
                results["A_exit"][i] = config_copy.nozzle.A_exit
                results["eps"][i] = eps_current  # Store expansion ratio
                results["recession_chamber"][i] = cumulative_recession_chamber
                results["recession_throat"][i] = cumulative_recession_throat
                results["recession_exit"][i] = cumulative_recession_exit
                
                # Store diagnostics
                results["diagnostics"].append(point_results["diagnostics"])
                
                # Update geometry for next time step (if ablative tracking enabled)
                if track_ablative_geometry and dt > 0 and i < n - 1:
                    # Get ablative recession rate from diagnostics
                    cooling_diag = point_results.get("diagnostics", {}).get("cooling", {})
                    ablative_diag = cooling_diag.get("ablative", {})
                    
                    # Check for graphite insert (separate from chamber ablator)
                    graphite_cfg = config_copy.graphite_insert
                    use_graphite_throat = graphite_cfg is not None and graphite_cfg.enabled
                    
                    recession_rate_chamber = 0.0
                    recession_rate_throat = 0.0
                    
                    # Chamber recession (from ablative cooling)
                    if ablative_diag.get("enabled", False):
                        recession_rate_chamber = ablative_diag.get("recession_rate", 0.0)
                    
                    # Throat recession (from graphite insert if enabled, else from ablator)
                    if use_graphite_throat:
                        # Use graphite insert properties for throat recession
                        Pc = point_results["Pc"]
                        Tc = point_results["Tc"]
                        
                        # Get throat heat flux from cooling diagnostics
                        # Throat heat flux is typically higher than chamber
                        chamber_heat_flux = ablative_diag.get("incident_heat_flux", 1e6) if ablative_diag.get("enabled", False) else 1e6
                        
                        # Estimate throat heat flux (higher than chamber due to sonic conditions)
                        # Use Bartz correlation multiplier
                        gamma = point_results.get("gamma", DEFAULT_GAMMA_ND)
                        throat_heat_flux_mult = calculate_throat_recession_multiplier(
                            Pc, 50.0, 1000.0, chamber_heat_flux, gamma  # Approximate velocities
                        )
                        throat_heat_flux = chamber_heat_flux * throat_heat_flux_mult
                        
                        # Calculate graphite recession rate
                        graphite_results = compute_graphite_recession(
                            net_heat_flux=throat_heat_flux,
                            throat_temperature=Tc * 0.85,  # Approximate throat temperature
                            gas_temperature=Tc,
                            graphite_config=graphite_cfg,
                            throat_area=config_copy.chamber.A_throat,
                            pressure=Pc,
                        )
                        
                        recession_rate_throat = graphite_results.get("recession_rate", 0.0)
                        
                        # Apply graphite coverage fraction
                        recession_rate_throat *= graphite_cfg.coverage_fraction
                        
                        # If ablator also covers throat (partial coverage), add ablator contribution
                        if ablative_diag.get("enabled", False) and graphite_cfg.coverage_fraction < 1.0:
                            ablator_coverage = 1.0 - graphite_cfg.coverage_fraction
                            recession_rate_ablator_throat = recession_rate_chamber * throat_heat_flux_mult
                            recession_rate_throat += recession_rate_ablator_throat * ablator_coverage
                    elif ablative_diag.get("enabled", False):
                        # Use ablative properties for throat (original behavior)
                        # Calculate throat recession multiplier from flow conditions
                        if ablative_cfg.throat_recession_multiplier is not None:
                            throat_mult = ablative_cfg.throat_recession_multiplier
                        else:
                            # Calculate from physics
                            Pc = point_results["Pc"]
                            mdot_total = point_results["mdot_total"]
                            gamma = point_results["gamma"]
                            R = point_results["R"]
                            Tc = point_results["Tc"]
                            
                            # Chamber velocity
                            rho_chamber = Pc / (R * Tc)
                            A_chamber = np.pi * (D_chamber_initial ** 2) / 4.0
                            v_chamber = mdot_total / (rho_chamber * A_chamber) if rho_chamber > 0 else 0.0
                            
                            # Throat velocity (sonic)
                            v_throat = np.sqrt(gamma * R * Tc / (gamma + 1))
                            
                            # Heat flux (from cooling diagnostics)
                            chamber_heat_flux = ablative_diag.get("incident_heat_flux", 1e6)
                            
                            throat_mult = calculate_throat_recession_multiplier(
                                Pc, v_chamber, v_throat, chamber_heat_flux, gamma
                            )
                        
                        results["throat_recession_multiplier"][i] = throat_mult
                        recession_rate_throat = recession_rate_chamber * throat_mult
                    
                    # Update cumulative recession
                    recession_increment_chamber = recession_rate_chamber * dt
                    recession_increment_throat = recession_rate_throat * dt
                    
                    cumulative_recession_chamber += recession_increment_chamber
                    cumulative_recession_throat += recession_increment_throat
                    
                    # Update geometry
                    # Use ablative coverage for chamber, graphite coverage for throat
                    chamber_coverage = ablative_cfg.coverage_fraction if ablative_cfg and ablative_cfg.enabled else 1.0
                    throat_coverage = graphite_cfg.coverage_fraction if use_graphite_throat and graphite_cfg else chamber_coverage
                    
                    V_new, A_throat_new, D_chamber_new, D_throat_new, geom_diag = update_chamber_geometry_from_ablation(
                        V_chamber_initial,
                        A_throat_initial,
                        D_chamber_initial,
                        D_throat_initial,
                        L_chamber,
                        cumulative_recession_chamber,
                        cumulative_recession_throat,
                        chamber_coverage,  # Chamber coverage from ablator
                        None,  # Don't use multiplier here, we already calculated throat recession
                    )
                    
                    # Update config for next iteration
                    config_copy.chamber.volume = V_new
                    config_copy.chamber.A_throat = A_throat_new
                    
                    # Update L* if specified
                    if config_copy.chamber.Lstar is not None:
                        config_copy.chamber.Lstar = V_new / A_throat_new
                    
                    # Update nozzle exit geometry if nozzle is ablative
                    if ablative_cfg and ablative_cfg.enabled and ablative_cfg.nozzle_ablative:
                        # Nozzle exit recedes at similar rate to chamber (can be tuned)
                        # For now, assume exit recession rate = 0.8 × chamber rate (less severe than throat)
                        recession_increment_exit = recession_rate_chamber * 0.8 * dt
                        cumulative_recession_exit += recession_increment_exit
                        
                        A_exit_new, D_exit_new, exit_diag = update_nozzle_exit_from_ablation(
                            A_exit_initial,
                            D_exit_initial,
                            cumulative_recession_exit,
                            ablative_cfg.coverage_fraction,
                        )
                        
                        # Update nozzle config
                        config_copy.nozzle.A_exit = A_exit_new
                        
                        # Expansion ratio will be recalculated on next iteration
                
            except Exception as e:
                # If solve fails, leave NaN values
                print(f"[WARNING] Time step {i} (t={times[i]:.3f}s) failed: {e}")
                import traceback
                traceback.print_exc()
                results["diagnostics"].append({"error": str(e)})
                continue
        
        return results
