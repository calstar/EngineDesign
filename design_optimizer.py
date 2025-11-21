
import numpy as np
import copy
from typing import Dict, Any, Tuple, Optional, List
from scipy.optimize import minimize_scalar

from pintle_pipeline.config_schemas import PintleEngineConfig, ChamberConfig, NozzleConfig
from pintle_models.chamber_solver import ChamberSolver
from pintle_pipeline.stability_analysis import comprehensive_stability_analysis
from pintle_pipeline.cea_cache import CEACache

# Constants
g0 = 9.80665

class StandardAtmosphere:
    @staticmethod
    def get_pressure(altitude_m: float) -> float:
        """
        Calculate ambient pressure based on US Standard Atmosphere 1976.
        Valid for Troposphere and Stratosphere (< 32km).
        """
        # Constants
        P0 = 101325.0  # Pa
        T0 = 288.15    # K
        g = 9.80665    # m/s^2
        R = 287.05     # J/(kg K)
        
        # Layers
        # Troposphere (0 - 11km)
        if altitude_m <= 11000:
            L = -0.0065 # K/m
            return P0 * (1 + L * altitude_m / T0) ** (-g / (R * L))
        
        # Stratosphere (11km - 20km)
        elif altitude_m <= 20000:
            P11 = 22632.1 # Pa at 11km
            T11 = 216.65  # K
            return P11 * np.exp(-g * (altitude_m - 11000) / (R * T11))
            
        # Upper Stratosphere (20km - 32km)
        else:
            P20 = 5474.89 # Pa at 20km
            T20 = 216.65  # K
            L = 0.001 # K/m
            return P20 * (1 + L * (altitude_m - 20000) / T20) ** (-g / (R * L))

class EngineDesigner:
    def __init__(self, base_config_path: str):
        """
        Initialize with a base configuration file to use as a template.
        """
        import yaml
        with open(base_config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        self.base_config = PintleEngineConfig(**config_dict)
        
        # Initialize CEA cache
        self.cea_cache = CEACache(self.base_config.combustion.cea)
        # Cache is loaded/built in __init__

    def optimize(self, 
                 target_altitude: float, 
                 max_chamber_diameter: float,
                 max_exit_diameter: float,
                 max_lox_tank_pressure: float,
                 max_fuel_tank_pressure: float,
                 target_thrust: Optional[float] = None) -> Tuple[PintleEngineConfig, Dict[str, Any]]:
        """
        Find optimal geometry maximizing thrust (or matching target) under constraints.
        """
        
        print(f"Optimizing for Altitude: {target_altitude} m")
        print(f"Constraints: Dc_max={max_chamber_diameter:.3f}m, De_max={max_exit_diameter:.3f}m")
        
        P_amb = StandardAtmosphere.get_pressure(target_altitude)
        print(f"Ambient Pressure: {P_amb:.1f} Pa")
        
        # Grid Search Parameters
        # We iterate over Pc and MR to find the "sweet spot"
        # Pc range: 1 MPa to Max Available (considering feed drops)
        # Approximate feed drop ~ 20% of tank pressure for stability?
        # Max feasible Pc approx 0.7 * min(P_lox, P_fuel)
        max_available_pressure = min(max_lox_tank_pressure, max_fuel_tank_pressure)
        pc_max_limit = max_available_pressure * 0.85 # Initial guess of max Pc
        pc_min_limit = 1.0e6 # 1 MPa minimum
        
        pc_values = np.linspace(pc_min_limit, pc_max_limit, 10)
        mr_values = np.linspace(2.0, 2.8, 5) # Typical LOX/RP-1 range
        
        best_config = None
        best_score = -1.0
        best_metrics = {}
        
        for Pc in pc_values:
            for MR in mr_values:
                try:
                    candidate_config, metrics = self._evaluate_design_point(
                        Pc, MR, P_amb, 
                        max_chamber_diameter, max_exit_diameter,
                        max_lox_tank_pressure, max_fuel_tank_pressure,
                        target_thrust
                    )
                    
                    if metrics['is_valid']:
                        # Score: Thrust if maximizing, or -|Thrust - Target| if targeting
                        score = metrics['thrust']
                        if target_thrust:
                            score = -abs(metrics['thrust'] - target_thrust)
                            
                        if score > best_score:
                            best_score = score
                            best_config = candidate_config
                            best_metrics = metrics
                            print(f"New Best: Pc={Pc/1e6:.2f}MPa, MR={MR:.2f}, Thrust={metrics['thrust']/1000:.1f}kN, Isp={metrics['isp']:.1f}s")
                    else:
                         # print(f"Invalid Point: Pc={Pc/1e6:.2f}MPa, MR={MR:.2f}")
                         # print(f"  Reason: {metrics['reason']}")
                         pass
                            
                except Exception as e:
                    print(f"Failed design point Pc={Pc}, MR={MR}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
        if best_config is None:
            raise RuntimeError("No valid design found within constraints!")
            
        return best_config, best_metrics

    def _evaluate_design_point(self, 
                             Pc: float, 
                             MR: float, 
                             P_amb: float,
                             max_Dc: float, 
                             max_De: float,
                             P_tank_lox: float,
                             P_tank_fuel: float,
                             target_thrust: Optional[float]) -> Tuple[PintleEngineConfig, Dict]:
        
        # 1. Get Thermochemical Properties (C*, Gamma, Tc)
        # We use the CEA cache directly
        # eval signature: (MR, Pc, Pa, eps)
        state = self.cea_cache.eval(MR=MR, Pc=Pc, eps=1.0) # Get chamber properties first
        cstar = state['cstar_ideal']
        gamma = state['gamma']
        Tc = state['Tc']
        
        # 2. Determine Nozzle Geometry
        # Calculate optimal expansion ratio for P_exit = P_amb
        # Isentropic relation: P_exit/Pc = (1 + (g-1)/2 * M^2)^(-g/(g-1))
        # This is non-trivial to invert for epsilon, so we iterate or use approximation
        # Better: use CEA cache if it has expansion data, but simplified:
        
        # Find epsilon where P_exit ~ P_amb
        # Approx formula or root finding
        def get_px_pc(eps, g):
            # Approximate pressure ratio for area ratio eps
            # This is tough to do analytically without M. 
            # We will use a simplified standard nozzle relation helper or estimation.
            # For now, let's assume we want P_exit = P_amb.
            return P_amb / Pc

        # Given Pc/Pe, finding Epsilon is standard gas dynamics.
        # We can use the existing CEA cache to find optimal epsilon if 3D cache exists, 
        # or calculating it.
        # Let's use a robust approximation or scipy root find.
        from scipy.optimize import root_scalar
        
        def area_ratio_from_mach(M, g):
            return ((g+1)/2)**(-(g+1)/(2*(g-1))) * ((1 + (g-1)/2 * M**2)**((g+1)/(2*(g-1)))) / M

        def pressure_ratio_from_mach(M, g):
            return (1 + (g-1)/2 * M**2)**(-g/(g-1))
            
        target_pr = P_amb / Pc
        if target_pr > 1.0: target_pr = 0.9 # Can't expand to higher pressure
        
        # Find Mach number for this pressure ratio
        # M > 1 for nozzle exit
        def mach_res(M):
            return pressure_ratio_from_mach(M, gamma) - target_pr
            
        try:
            sol = root_scalar(mach_res, bracket=[1.0, 20.0], method='brentq')
            M_exit = sol.root
            epsilon = area_ratio_from_mach(M_exit, gamma)
        except:
            epsilon = 10.0 # Fallback
            
        # 3. Determine Throat Area (Scale)
        # If target_thrust is given: F = Pc * At * Cf
        # Cf approx sqrt(...) or from CEA
        # Let's estimate Cf
        # Cf = sqrt(2*g^2/(g-1) * (2/(g+1))^((g+1)/(g-1)) * (1 - (Pe/Pc)^((g-1)/g))) + (Pe-Pa)/Pc * eps
        # Ideally we want Pe=Pa, so second term is 0.
        
        pe_pc = pressure_ratio_from_mach(M_exit, gamma)
        cf_ideal = np.sqrt(2*gamma**2/(gamma-1) * (2/(gamma+1))**((gamma+1)/(gamma-1)) * (1 - pe_pc**((gamma-1)/gamma)))
        cf_est = cf_ideal * 0.98 # Efficiency
        
        if target_thrust:
            # F = Pc * At * Cf
            A_t = target_thrust / (Pc * cf_est)
        else:
            # Maximize thrust within diameter constraints
            # Constrained by Max Exit Diameter
            # A_exit_max = pi * (max_De/2)^2
            # A_t_max = A_exit_max / epsilon
            
            # Also constrained by Chamber Diameter?
            # Typically Dc ~ 2-3 * Dt. 
            # Let's set A_t based on Max Exit Diameter as the limit for "Max Thrust"
            A_exit_limit = np.pi * (max_De / 2)**2
            A_t = A_exit_limit / epsilon
            
            # Also check chamber diameter constraint
            # Assume Contraction Ratio (Ac/At) around 3 to 5
            CR = 3.0
            A_c = A_t * CR
            D_c_req = 2 * np.sqrt(A_c / np.pi)
            
            if D_c_req > max_Dc:
                # Must reduce size
                D_c_req = max_Dc
                A_c = np.pi * (D_c_req / 2)**2
                A_t = A_c / CR
                # Re-check exit diameter with new At
                A_exit_new = A_t * epsilon
                if A_exit_new > A_exit_limit:
                    # This implies our constraints are conflicting or epsilon is huge.
                    # If we are bound by chamber size, we might have a smaller exit than max allowed.
                    pass

        # Recalculate Geometry derived from A_t
        D_t = 2 * np.sqrt(A_t / np.pi)
        A_exit = A_t * epsilon
        D_exit = 2 * np.sqrt(A_exit / np.pi)
        
        # Check constraints again explicitly
        if D_exit > max_De * 1.01: # 1% tolerance
             # Clip to max exit
             D_exit = max_De
             A_exit = np.pi * (D_exit/2)**2
             epsilon = A_exit / A_t
        
        # Chamber Geometry
        # Use L* to size volume
        # Increased L* to 2.0 to lower chugging frequency (< 200 Hz)
        Lstar = 2.0 
        V_chamber = A_t * Lstar
        
        # Chamber Length/Diameter
        # Assume L/D ratio or standard Contraction Ratio
        # Ac/At usually 3-5
        CR = 3.5
        A_c = A_t * CR
        D_c = 2 * np.sqrt(A_c / np.pi)
        
        if D_c > max_Dc:
            D_c = max_Dc
            A_c = np.pi * (D_c/2)**2
            # If Ac is limited, length must increase to maintain V_chamber (L*)?
            # Or L* decreases? Usually we want to maintain L* for combustion efficiency.
            # So Length increases.
        
        L_c = V_chamber / A_c
        
        # 4. Update Configuration
        new_config = copy.deepcopy(self.base_config)
        
        # Update Chamber
        new_config.chamber.volume = float(V_chamber)
        new_config.chamber.A_throat = float(A_t)
        new_config.chamber.chamber_inner_diameter = float(D_c)
        new_config.chamber.length = float(L_c)
        new_config.chamber.Lstar = float(Lstar)
        
        # Update Nozzle
        new_config.nozzle.A_throat = float(A_t)
        new_config.nozzle.A_exit = float(A_exit)
        new_config.nozzle.expansion_ratio = float(epsilon)
        new_config.nozzle.exit_diameter = float(D_exit)
        
        # 5. Injector Sizing (Critical for Stability and Pressure Drop)
        # Calculate Mass Flow
        # mdot = Pc * At / cstar
        mdot = Pc * A_t / cstar
        mdot_ox = mdot * (MR / (1 + MR))
        mdot_fuel = mdot * (1 / (1 + MR))
        
        # Target Pressure Drops (20% of Pc typical for stability)
        # dP_inj = 0.2 * Pc
        # But we are constrained by Tank Pressure.
        # dP_allowable = P_tank - Pc - dP_feed
        
        # Feed line drops (Estimate)
        # Using config's feed system parameters if available
        # Or approx: dP_feed ~ 5-10 psi or dynamic
        # Let's assume we tune injector to use available margin up to 20%
        
        dP_feed_ox = 0.0 # Simplified, usually calculated by solver
        dP_feed_fuel = 0.0
        
        # We need to check if we have enough pressure
        P_inj_ox_avail = P_tank_lox - dP_feed_ox
        P_inj_fuel_avail = P_tank_fuel - dP_feed_fuel
        
        dP_inj_ox = P_inj_ox_avail - Pc
        dP_inj_fuel = P_inj_fuel_avail - Pc
        
        if dP_inj_ox < 0.10 * Pc or dP_inj_fuel < 0.10 * Pc:
            # Not enough margin for stable injection
            return new_config, {'is_valid': False, 'reason': f'Insufficient injection pressure margin: dP_ox/Pc={dP_inj_ox/Pc:.2f}, dP_fuel/Pc={dP_inj_fuel/Pc:.2f}'}
            
        # Clamp to reasonable max injector drop (don't waste energy if we have huge tank pressure)
        # Optimal is usually 15-20%
        dP_inj_target_ox = min(dP_inj_ox, 0.25 * Pc)
        dP_inj_target_fuel = min(dP_inj_fuel, 0.25 * Pc)
        
        # Size Injector Elements based on dP = K * mdot^2 / (rho * A^2)
        # A = mdot / sqrt(2 * rho * dP * Cd^2)
        # Assume Cd ~ 0.7
        Cd = 0.7
        rho_ox = new_config.fluids['oxidizer'].density
        rho_fuel = new_config.fluids['fuel'].density
        
        A_ox_total = mdot_ox / (Cd * np.sqrt(2 * rho_ox * dP_inj_target_ox))
        A_fuel_total = mdot_fuel / (Cd * np.sqrt(2 * rho_fuel * dP_inj_target_fuel))
        
        # Update Injector Geometry
        # Assume Pintle or Coaxial based on config type
        inj_type = new_config.injector.type
        
        # Update Feed System Geometry to Ensure Stability (Low Velocity to avoid Water Hammer issues)
        # Target velocity < 1.0 m/s to keep water hammer pressure low enough for stability margin
        target_feed_v = 0.5 # m/s
        
        # Oxidizer Feed Line
        rho_ox = new_config.fluids['oxidizer'].density
        A_feed_ox = mdot_ox / (rho_ox * target_feed_v)
        d_feed_ox = np.sqrt(4 * A_feed_ox / np.pi)
        
        # Fuel Feed Line
        rho_fuel = new_config.fluids['fuel'].density
        A_feed_fuel = mdot_fuel / (rho_fuel * target_feed_v)
        d_feed_fuel = np.sqrt(4 * A_feed_fuel / np.pi)
        
        # Update Config
        if isinstance(new_config.feed_system, dict):
             if 'oxidizer' in new_config.feed_system:
                 new_config.feed_system['oxidizer'].d_inlet = float(d_feed_ox)
             if 'fuel' in new_config.feed_system:
                 new_config.feed_system['fuel'].d_inlet = float(d_feed_fuel)
        else:
             # Pydantic model access
             if hasattr(new_config.feed_system, 'oxidizer'):
                 new_config.feed_system.oxidizer.d_inlet = float(d_feed_ox)
             if hasattr(new_config.feed_system, 'fuel'):
                 new_config.feed_system.fuel.d_inlet = float(d_feed_fuel)

        if inj_type == 'pintle':
            # Adjust LOX orifices or annulus
            # Simply scaling existing geometry is safer than redesigning form factor
            # Assume we scale area by scaling N_orifices or Diameter
            # Let's update total area params
            
            # LOX (Center)
            new_config.injector.geometry.lox.A_entry = float(A_ox_total / new_config.injector.geometry.lox.n_orifices)
            # Recalculate diameter for consistency
            new_config.injector.geometry.lox.d_orifice = float(2 * np.sqrt(new_config.injector.geometry.lox.A_entry / np.pi))
            
            # Fuel (Annulus/Gap)
            # Fuel area = pi * D_pintle * h_gap (approx)
            d_pintle = new_config.injector.geometry.fuel.d_pintle_tip
            h_gap = A_fuel_total / (np.pi * d_pintle)
            new_config.injector.geometry.fuel.h_gap = float(h_gap)
            new_config.injector.geometry.fuel.A_entry = float(A_fuel_total) # Reservoir entry
            
        elif inj_type == 'coaxial':
            # Update Core and Annulus
            pass # Logic would go here
            
        elif inj_type == 'impinging':
            pass
            
        # 6. Stability Check
        # Run the full stability analysis
        # Need a dummy solver result for the diagnostic dict
        stability_diagnostics = {
             'injector_pressure_drop_ratio': min(dP_inj_target_ox, dP_inj_target_fuel) / Pc,
             'P_tank_O': float(P_tank_lox),
             'mdot_O': float(mdot_ox),
             # Add other needed keys if analysis requires them
        }
        
        stability = comprehensive_stability_analysis(
            config=new_config,
            Pc=Pc,
            MR=MR,
            mdot_total=mdot,
            cstar=cstar,
            gamma=gamma,
            R=8314.46/24.0, # Approx R
            Tc=Tc,
            diagnostics=stability_diagnostics
        )
        
        is_stable = stability['is_stable']
        
        # If not stable, maybe penalize or reject
        # Ideally we would adjust geometry to fix it, but for now we filter
        if not is_stable:
            reason = f"Unstable: {stability.get('issues', [])}"
            return new_config, {'is_valid': False, 'reason': reason, 'stability': stability}
            
        # 7. Success
        thrust = mdot * cf_est * cstar # Approx
        isp = thrust / (mdot * g0)
        
        return new_config, {
            'is_valid': True,
            'thrust': float(thrust),
            'isp': float(isp),
            'Pc': float(Pc),
            'MR': float(MR),
            'stability': stability
        }

if __name__ == "__main__":
    # Example Usage
    import sys
    import os
    
    base_config = "examples/pintle_engine/config_minimal.yaml"
    if not os.path.exists(base_config):
        print(f"Config not found: {base_config}")
        sys.exit(1)
        
    optimizer = EngineDesigner(base_config)
    
    try:
        # Example constraints
        new_config, metrics = optimizer.optimize(
            target_altitude=10000.0, # 10km
            max_chamber_diameter=0.20, # 20cm
            max_exit_diameter=0.30,    # 30cm
            max_lox_tank_pressure=30e5, # 30 bar
            max_fuel_tank_pressure=30e5 # 30 bar
        )
        
        print("\nOptimization Successful!")
        print(f"Thrust: {metrics['thrust']:.1f} N")
        print(f"Isp: {metrics['isp']:.1f} s")
        print(f"Chamber Pressure: {metrics['Pc']/1e5:.2f} bar")
        print(f"Mixture Ratio: {metrics['MR']:.2f}")
        print(f"Throat Diameter: {np.sqrt(new_config.chamber.A_throat/np.pi)*2*1000:.1f} mm")
        print(f"Exit Diameter: {new_config.nozzle.exit_diameter*1000:.1f} mm")
        print(f"Chamber Diameter: {new_config.chamber.chamber_inner_diameter*1000:.1f} mm")
        
        # Save result
        with open("optimized_engine.yaml", "w") as f:
            import yaml
            # Convert pydantic to dict (v2 syntax)
            yaml.dump(new_config.model_dump(), f)
        print("Saved to optimized_engine.yaml")
        
    except Exception as e:
        print(f"Optimization failed: {e}")

