"""Ablative cooling response model."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .config_schemas import AblativeCoolingConfig

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant


def compute_ablative_response(
    net_heat_flux: float,
    surface_temperature: float,
    gas_temperature: float,
    ablative_config: AblativeCoolingConfig,
    surface_area: float,
    turbulence_intensity: float,
) -> Dict[str, float]:
    """Estimate ablative recession rate and heat balance.

    Parameters
    ----------
    net_heat_flux : float
        Heat flux incident on ablative surface [W/m²].
    surface_temperature : float
        Estimated current surface temperature [K].
    gas_temperature : float
        Free-stream gas temperature [K].
    ablative_config : AblativeCoolingConfig
        Ablation model configuration.

    Returns
    -------
    dict
        Response metrics including recession rate [m/s] and mass flux [kg/(m²·s)].
    """
    if not ablative_config.enabled or surface_area <= 0:
        return {
            "enabled": False,
            "recession_rate": 0.0,
            "mass_flux": 0.0,
            "surface_temperature": surface_temperature,
            "heat_removed": 0.0,
            "turbulence_multiplier": 1.0,
        }

    # Radiative relief (surface assumed grey body)
    radiative_relief = (
        SIGMA
        * (surface_temperature ** 4 - ablative_config.surface_temperature_limit ** 4)
    )
    radiative_relief = max(radiative_relief, 0.0)

    turb_multiplier = 1.0
    if turbulence_intensity > 0 and ablative_config.turbulence_reference_intensity > 0:
        ratio = (turbulence_intensity / ablative_config.turbulence_reference_intensity) ** ablative_config.turbulence_exponent
        turb_multiplier = 1.0 + ablative_config.turbulence_sensitivity * ratio
    turb_multiplier = float(np.clip(turb_multiplier, 1.0, ablative_config.turbulence_max_multiplier))

    convective_reduction = 1.0 - np.clip(ablative_config.blowing_efficiency, 0.0, 1.0)
    effective_heat_flux = max(net_heat_flux * turb_multiplier * convective_reduction - radiative_relief, 0.0)

    if effective_heat_flux <= 0:
        return {
            "enabled": True,
            "recession_rate": 0.0,
            "mass_flux": 0.0,
            "surface_temperature": surface_temperature,
            "effective_heat_flux": effective_heat_flux,
            "radiative_relief": radiative_relief,
            "heat_removed": 0.0,
            "turbulence_multiplier": turb_multiplier,
        }

    delta_T_pyro = max(surface_temperature - ablative_config.pyrolysis_temperature, 0.0)
    energy_per_mass = ablative_config.heat_of_ablation + ablative_config.specific_heat * delta_T_pyro
    mass_flux = effective_heat_flux / max(energy_per_mass, 1e-6)
    recession_rate = mass_flux / ablative_config.material_density

    next_surface_temp = min(
        gas_temperature,
        ablative_config.surface_temperature_limit,
    )

    mass_flow = mass_flux * surface_area
    heat_removed = effective_heat_flux * surface_area

    return {
        "enabled": True,
        "recession_rate": float(recession_rate),
        "mass_flux": float(mass_flux),
        "surface_temperature": next_surface_temp,
        "effective_heat_flux": float(effective_heat_flux),
        "radiative_relief": float(radiative_relief),
        "heat_removed": float(heat_removed),
        "mass_flow": float(mass_flow),
        "coverage_area": float(surface_area),
        "turbulence_multiplier": turb_multiplier,
    }
