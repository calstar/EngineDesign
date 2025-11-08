"""Film cooling effectiveness model."""

from __future__ import annotations

from typing import Optional, Dict

import numpy as np

from .config_schemas import FilmCoolingConfig


def compute_film_cooling(
    mdot_total: float,
    mdot_fuel: float,
    gas_temperature: float,
    film_config: FilmCoolingConfig,
    fuel_temperature: float,
) -> Dict[str, float]:
    """Compute film cooling effectiveness and adjusted heat-flux factor.

    Parameters
    ----------
    mdot_total : float
        Total propellant mass flow [kg/s].
    mdot_fuel : float
        Fuel mass flow [kg/s] (before allocating film fraction).
    gas_temperature : float
        Chamber gas temperature [K].
    film_config : FilmCoolingConfig
        Film cooling configuration object.
    fuel_temperature : float
        Bulk fuel temperature [K] used for film injection if no override specified.

    Returns
    -------
    dict
        Dictionary containing film effectiveness, remaining coolant flow, and
        heat-flux reduction factor.
    """
    if not film_config.enabled:
        return {
            "enabled": False,
            "mass_fraction": 0.0,
            "mdot_film": 0.0,
            "mdot_available_for_regen": mdot_fuel,
            "effectiveness": 0.0,
            "heat_flux_factor": 1.0,
            "film_temperature": fuel_temperature,
        }

    mass_fraction = np.clip(film_config.mass_fraction, 0.0, 0.9)
    mdot_film = mass_fraction * mdot_fuel
    mdot_available = max(mdot_fuel - mdot_film, 1e-6)

    injection_temperature = (
        film_config.injection_temperature
        if film_config.injection_temperature is not None
        else fuel_temperature
    )

    # Simple empirical scaling of effectiveness with mass fraction and coverage
    base_effectiveness = film_config.effectiveness_ref
    if mass_fraction > 0:
        base_effectiveness *= (mass_fraction / max(1e-6, 0.05)) ** 0.5
    coverage_factor = np.clip(film_config.apply_to_fraction_of_length, 0.0, 1.5)
    effectiveness = np.clip(base_effectiveness * coverage_factor, 0.0, 0.95)

    # Effective recovery temperature after film dilution
    effective_gas_temperature = (
        gas_temperature - effectiveness * (gas_temperature - injection_temperature)
    )
    heat_flux_factor = max(1e-3, (effective_gas_temperature - injection_temperature) /
                           (gas_temperature - injection_temperature + 1e-9))

    return {
        "enabled": True,
        "mass_fraction": mass_fraction,
        "mdot_film": mdot_film,
        "mdot_available_for_regen": mdot_available,
        "effectiveness": effectiveness,
        "heat_flux_factor": float(np.clip(heat_flux_factor, 0.05, 1.0)),
        "film_temperature": injection_temperature,
        "effective_gas_temperature": effective_gas_temperature,
    }
