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
    if not ablative_config.enabled:
        return {
            "enabled": False,
            "recession_rate": 0.0,
            "mass_flux": 0.0,
            "surface_temperature": surface_temperature,
        }

    # Radiative relief (surface assumed grey body)
    radiative_relief = (
        SIGMA
        * (surface_temperature ** 4 - ablative_config.surface_temperature_limit ** 4)
    )
    radiative_relief = max(radiative_relief, 0.0)

    effective_heat_flux = max(net_heat_flux - radiative_relief, 0.0)

    if effective_heat_flux <= 0:
        return {
            "enabled": True,
            "recession_rate": 0.0,
            "mass_flux": 0.0,
            "surface_temperature": surface_temperature,
            "effective_heat_flux": effective_heat_flux,
            "radiative_relief": radiative_relief,
        }

    mass_flux = effective_heat_flux / ablative_config.heat_of_ablation
    recession_rate = mass_flux / ablative_config.material_density

    next_surface_temp = min(
        gas_temperature,
        ablative_config.surface_temperature_limit,
    )

    return {
        "enabled": True,
        "recession_rate": float(recession_rate),
        "mass_flux": float(mass_flux),
        "surface_temperature": next_surface_temp,
        "effective_heat_flux": float(effective_heat_flux),
        "radiative_relief": float(radiative_relief),
    }
