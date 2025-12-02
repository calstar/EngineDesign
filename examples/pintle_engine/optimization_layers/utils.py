"""Utility functions for optimization layers.

Contains parameter extraction and other helper utilities.
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np

from pintle_pipeline.config_schemas import PintleEngineConfig


def extract_all_parameters(config: PintleEngineConfig) -> Dict[str, Any]:
    """Extract all optimized parameters from config."""
    params = {}
    
    # Injector parameters
    if hasattr(config, 'injector') and config.injector.type == "pintle":
        geometry = config.injector.geometry
        if hasattr(geometry, 'fuel'):
            params["d_pintle_tip"] = geometry.fuel.d_pintle_tip
            params["h_gap"] = geometry.fuel.h_gap
            if hasattr(geometry.fuel, 'd_reservoir_inner'):
                params["d_reservoir_inner"] = geometry.fuel.d_reservoir_inner
        if hasattr(geometry, 'lox'):
            params["n_orifices"] = geometry.lox.n_orifices
            params["d_orifice"] = geometry.lox.d_orifice
            params["theta_orifice"] = geometry.lox.theta_orifice
    
    # Chamber parameters
    params["A_throat"] = config.chamber.A_throat
    params["Lstar"] = config.chamber.Lstar
    params["chamber_volume"] = config.chamber.volume
    params["chamber_length"] = config.chamber.length
    if hasattr(config.chamber, 'chamber_inner_diameter') and config.chamber.chamber_inner_diameter:
        params["chamber_diameter"] = config.chamber.chamber_inner_diameter
    else:
        # Safe calculation with validation
        if config.chamber.volume > 0 and config.chamber.length > 0:
            params["chamber_diameter"] = np.sqrt(4.0 * config.chamber.volume / (np.pi * config.chamber.length))
        else:
            if config.chamber.A_throat > 0:
                params["chamber_diameter"] = np.sqrt(4.0 * config.chamber.A_throat / np.pi) * 2.0
            else:
                params["chamber_diameter"] = 0.1
    
    # Nozzle parameters
    params["A_exit"] = config.nozzle.A_exit
    params["expansion_ratio"] = config.nozzle.expansion_ratio
    
    # Ablative liner parameters
    if hasattr(config, 'ablative_cooling') and config.ablative_cooling and config.ablative_cooling.enabled:
        params["ablative_thickness"] = config.ablative_cooling.initial_thickness
        params["ablative_enabled"] = True
    else:
        params["ablative_thickness"] = 0.0
        params["ablative_enabled"] = False
    
    # Graphite insert parameters
    if hasattr(config, 'graphite_insert') and config.graphite_insert and config.graphite_insert.enabled:
        params["graphite_thickness"] = config.graphite_insert.initial_thickness
        params["graphite_enabled"] = True
    else:
        params["graphite_thickness"] = 0.0
        params["graphite_enabled"] = False
    
    return params

