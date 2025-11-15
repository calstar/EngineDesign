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
            "q_oxidation": 0.0,
            "q_radiation": 0.0,
            "q_surface": 0.0,
        }
    
    # Get optional config fields with safe defaults for backward compatibility
    eps = getattr(graphite_config, "epsilon", 0.9)
    Tamb = getattr(graphite_config, "ambient_temperature", 300.0)
    include_qox = getattr(graphite_config, "include_oxidation_heat", True)
    Fv = getattr(graphite_config, "view_factor", 1.0)
    
    # Radiative cooling from surface
    # Radiation uses emissivity ε, view factor, and ambient temperature
    # Radiation is wall-to-ambient (not wall-to-gas). Convective heat transfer from gas
    # is handled by the caller through net_heat_flux. If gas_temperature > Tamb, the
    # convective component should already be included in net_heat_flux.
    # net_heat_flux is assumed to be incident non-radiative load from the caller
    radiative_relief = max(eps * Fv * SIGMA * (throat_temperature**4 - Tamb**4), 0.0)
    
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
        # Oxidation heat is exothermic and is added to the surface balance when include_oxidation_heat is True
        delta_h_oxidation = 32e6  # J/kg (approximate)
        q_oxidation = oxidation_rate * graphite_config.material_density * delta_h_oxidation
    else:
        oxidation_rate = 0.0
        q_oxidation = 0.0
    
    # Surface heat balance: incident flux minus radiation, plus oxidation heat if enabled
    # If the caller already included radiation or oxidation, they should disable include_oxidation_heat
    # or pass a net that excludes it to avoid double counting
    q_surface = net_heat_flux - radiative_relief
    if include_qox:
        q_surface += q_oxidation
    q_net = max(q_surface, 0.0)
    
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
        "q_oxidation": float(q_oxidation),
        "q_radiation": float(radiative_relief),
        "q_surface": float(q_surface),
    }

