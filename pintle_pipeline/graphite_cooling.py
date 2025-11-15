"""Graphite throat insert cooling and recession model"""

from __future__ import annotations

from typing import Dict
import numpy as np
from pintle_pipeline.config_schemas import GraphiteInsertConfig

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant


def compute_graphite_recession(
    net_heat_flux: float,
    throat_temperature: float,
    gas_temperature: float,
    graphite_config: GraphiteInsertConfig,
    throat_area: float,
    pressure: float,
) -> Dict[str, float]:
    """
    Calculate graphite throat insert recession rate.
    
    Graphite recession is driven by:
    1. Thermal ablation (heat flux)
    2. Oxidation (chemical reaction above ~800 K)
    3. Erosion (mechanical removal)
    
    The primary mechanism is oxidation, unlike ablators which use pyrolysis.
    
    Parameters:
    -----------
    net_heat_flux : float
        Incident heat flux on throat [W/m²]
    throat_temperature : float
        Throat surface temperature [K]
    gas_temperature : float
        Free-stream gas temperature [K]
    graphite_config : GraphiteInsertConfig
        Graphite insert configuration
    throat_area : float
        Throat area [m²]
    pressure : float
        Chamber/throat pressure [Pa]
    
    Returns:
    --------
    dict
        Recession metrics including recession rate [m/s] and mass flux [kg/(m²·s)]
    """
    if not graphite_config.enabled or throat_area <= 0:
        return {
            "enabled": False,
            "recession_rate": 0.0,
            "mass_flux": 0.0,
            "surface_temperature": throat_temperature,
            "heat_removed": 0.0,
            "oxidation_rate": 0.0,
        }
    
    # Radiative cooling from surface
    radiative_relief = SIGMA * (throat_temperature ** 4 - 300 ** 4)  # 300K ambient
    radiative_relief = max(radiative_relief, 0.0)
    
    # Oxidation recession (dominant mechanism for graphite)
    # Oxidation rate increases with temperature above oxidation threshold
    if throat_temperature > graphite_config.oxidation_temperature:
        # Oxidation rate scales with temperature
        # Use Arrhenius-like scaling: rate ∝ exp(-E/T) where E is activation energy
        T_ratio = (throat_temperature - graphite_config.oxidation_temperature) / (
            graphite_config.surface_temperature_limit - graphite_config.oxidation_temperature
        )
        T_ratio = np.clip(T_ratio, 0.0, 1.0)
        
        # Oxidation rate increases with temperature
        # Also increases with pressure (more oxidizer available)
        P_effect = (pressure / 1e6) ** 0.5  # Normalized pressure effect
        oxidation_rate = graphite_config.oxidation_rate * (1.0 + 10.0 * T_ratio) * P_effect
        
        # Heat flux from oxidation (energy released per unit mass oxidized)
        # Graphite oxidation: C + O2 -> CO2, Δh ≈ 32 MJ/kg C
        delta_h_oxidation = 32e6  # J/kg (approximate)
        q_oxidation = oxidation_rate * graphite_config.material_density * delta_h_oxidation
    else:
        oxidation_rate = 0.0
        q_oxidation = 0.0
    
    # Net heat flux into graphite (after radiative cooling)
    q_net = max(net_heat_flux - radiative_relief, 0.0)
    
    # Thermal ablation component (heat-driven recession)
    # Energy required per unit mass ablated
    delta_T = max(throat_temperature - 300.0, 0.0)  # Temperature rise from ambient
    energy_per_mass = graphite_config.heat_of_ablation + graphite_config.specific_heat * delta_T
    
    # Thermal recession rate from heat flux
    if energy_per_mass > 0 and q_net > 0:
        mass_flux_thermal = q_net / energy_per_mass
        recession_rate_thermal = mass_flux_thermal / graphite_config.material_density
    else:
        recession_rate_thermal = 0.0
        mass_flux_thermal = 0.0
    
    # Total recession rate (thermal + oxidation)
    # Oxidation is typically dominant at high temperatures
    recession_rate_total = recession_rate_thermal + oxidation_rate
    
    # Total mass flux
    mass_flux_total = recession_rate_total * graphite_config.material_density
    
    # Heat removed by ablation
    heat_removed = q_net * throat_area * graphite_config.coverage_fraction
    
    # Surface temperature (limited by material limit)
    surface_temp = min(throat_temperature, graphite_config.surface_temperature_limit)
    
    return {
        "enabled": True,
        "recession_rate": float(recession_rate_total),
        "mass_flux": float(mass_flux_total),
        "surface_temperature": float(surface_temp),
        "effective_heat_flux": float(q_net),
        "radiative_relief": float(radiative_relief),
        "heat_removed": float(heat_removed),
        "oxidation_rate": float(oxidation_rate),
        "recession_rate_thermal": float(recession_rate_thermal),
        "mass_flux_thermal": float(mass_flux_thermal),
        "coverage_area": float(throat_area * graphite_config.coverage_fraction),
    }

