"""Ablative cooling response model with proper physics-based ablation."""

from __future__ import annotations

from typing import Dict, Union

import numpy as np

from engine.pipeline.config_schemas import AblativeCoolingConfig
from engine.pipeline.constants import STEFAN_BOLTZMANN_W_M2_K4, EPSILON_SMALL


def compute_ablative_response(
    net_heat_flux: float,
    surface_temperature: float,
    ablative_config: AblativeCoolingConfig,
    surface_area: float,
    turbulence_intensity: float,
    heat_flux_conv: float = None,
    heat_flux_rad: float = None,
    gas_mass_flow_rate: float = None,
) -> Dict[str, Union[float, bool]]:
    """Estimate ablative recession rate and heat balance with proper physics.

    This function models:
    1. Ablation only occurs when surface temperature exceeds pyrolysis temperature
    2. Proper radiative heat transfer (radiation FROM wall surface to environment)
    3. Energy balance accounting for all heat transfer mechanisms
    
    Heat Flux Definitions and Sign Conventions:
    - heat_flux_conv: NET convective flux from gas to wall [W/m²]
      Formula: q_conv = h_g × (Taw - Tw)
      Sign: Positive = heat flows INTO wall (gas → wall)
      This is already the net flux (gas → wall)
    
    - heat_flux_rad: NET radiative flux from gas to wall [W/m²]
      Formula: q_rad = ε × σ × (T_gas⁴ - T_wall⁴)
      Sign: Positive = heat flows INTO wall (gas → wall)
      This is already the net flux (gas → wall), accounting for wall temperature
      NOTE: If T_wall > T_gas, this will be negative (wall radiates to gas)
    
    - radiative_relief: Radiative flux from wall to environment [W/m²]
      Formula: q_relief = ε × σ × (T_wall⁴ - T_sink⁴)
      Sign: Always non-negative (clamped to ≥ 0)
      This is a separate cooling term (wall → environment), reduces net heat into wall
      Always subtracted from total heat flux (cooling effect)
    
    Energy Balance:
    effective_heat_flux = (q_conv_effective + q_rad_net) - q_relief
    where q_conv_effective includes turbulence and blowing effects
    
    Parameters
    ----------
    net_heat_flux : float
        Total heat flux incident on ablative surface [W/m²].
        This includes both convective and radiative components from the hot gas.
    surface_temperature : float
        Current surface temperature [K].
    ablative_config : AblativeCoolingConfig
        Ablation model configuration.
    surface_area : float
        Surface area of ablative material [m²].
    turbulence_intensity : float
        Gas turbulence intensity (0-1).
    heat_flux_conv : float, optional
        NET convective heat flux from gas to wall [W/m²].
        Formula: h_g × (Taw - Tw). Already accounts for wall temperature.
        Sign convention: Positive = heat flows INTO wall (gas → wall).
    heat_flux_rad : float, optional
        NET radiative heat flux from gas to wall [W/m²].
        Formula: ε × σ × (T_gas⁴ - T_wall⁴). Already accounts for wall temperature.
        Sign convention: Positive = heat flows INTO wall (gas → wall).
        Can be negative if T_wall > T_gas (wall radiates to gas), but this is unusual.
    gas_mass_flow_rate : float, optional
        External gas mass flow rate [kg/s] (for physics-based blowing calculation).
        If provided and use_physics_based_blowing=True, computes blowing parameter B.

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
            "cooling_power": 0.0,
            "heat_removed": 0.0,  # Backward compatibility
            "turbulence_multiplier": 1.0,
            "radiative_relief": 0.0,
            "heat_flux_from_gas_radiative": 0.0,
            "heat_flux_from_gas_convective": 0.0,
        }

    # ========================================================================
    # TURBULENCE EFFECTS
    # ========================================================================
    turb_multiplier = 1.0
    if turbulence_intensity > 0 and ablative_config.turbulence_reference_intensity > 0:
        ratio = (turbulence_intensity / ablative_config.turbulence_reference_intensity) ** ablative_config.turbulence_exponent
        turb_multiplier = 1.0 + ablative_config.turbulence_sensitivity * ratio
    turb_multiplier = float(np.clip(turb_multiplier, 1.0, ablative_config.turbulence_max_multiplier))

    # ========================================================================
    # CHECK IF ABOVE PYROLYSIS TEMPERATURE
    # ========================================================================
    # Ablation only occurs when surface temperature exceeds pyrolysis temperature
    # Below pyrolysis: no ablation → no pyrolysis gases → no blowing effect
    below_pyrolysis = surface_temperature < ablative_config.pyrolysis_temperature

    # ========================================================================
    # BLOWING EFFECT (pyrolysis gases reduce convective heat transfer)
    # ========================================================================
    # Physics-based: B = m_dot_pyrolysis / m_dot_external
    # Empirical function: f(B) = 1/(1 + c*B) where c is blowing_coefficient
    # Legacy: constant factor based on blowing_efficiency
    #
    # Note: If below pyrolysis, there's no pyrolysis gases, so no blowing effect
    if below_pyrolysis:
        # No ablation → no pyrolysis gases → no blowing effect
        convective_reduction = 1.0
        use_physics_blowing = False
    elif ablative_config.use_physics_based_blowing and gas_mass_flow_rate is not None and gas_mass_flow_rate > 0:
        # Physics-based blowing will be computed later (after we have heat fluxes)
        use_physics_blowing = True
        # convective_reduction will be computed in the physics-based block
    else:
        # Legacy constant factor
        use_physics_blowing = False
        convective_reduction = 1.0 - np.clip(ablative_config.blowing_efficiency, 0.0, 1.0)

    # ========================================================================
    # RADIATIVE SINK TEMPERATURE
    # ========================================================================
    # Use fallback temperature if ambient is too low (represents heated steel layer behind ablator)
    if ablative_config.ambient_temperature < ablative_config.radiative_sink_minimum_threshold:
        T_rad_sink = ablative_config.radiative_sink_fallback_temperature
    else:
        T_rad_sink = ablative_config.ambient_temperature

    # ========================================================================
    # RADIATIVE RELIEF (Wall → Environment)
    # ========================================================================
    # Radiation FROM the hot wall surface TO the environment (reduces net heat flux into wall)
    # This is SEPARATE from heat_flux_rad (which is gas → wall)
    # Formula: q_rad_wall_to_env = ε × σ × (T_surf⁴ - T_sink⁴)
    # Note: heat_flux_rad is already NET (gas → wall), so this is an additional cooling term
    radiative_relief = (
        ablative_config.surface_emissivity
        * STEFAN_BOLTZMANN_W_M2_K4
        * (surface_temperature ** 4 - T_rad_sink ** 4)
    )
    radiative_relief = max(radiative_relief, 0.0)

    # ========================================================================
    # EFFECTIVE HEAT FLUX
    # ========================================================================
    # Separate convective and radiative components (turbulence/blowing only affect convective)
    if heat_flux_conv is not None and heat_flux_rad is not None:
        q_conv_incident = heat_flux_conv
        q_rad_incident = heat_flux_rad
    else:
        # Estimate: typically 20% radiative, 80% convective
        q_conv_incident = net_heat_flux * 0.8
        q_rad_incident = net_heat_flux * 0.2
    
    # Compute physics-based blowing if enabled (only when NOT below pyrolysis)
    if use_physics_blowing:  # We already know not below_pyrolysis from earlier check
        # Step 1: Provisional mass flux assuming no blowing reduction
        # (only turbulence applied)
        q_conv_provisional = q_conv_incident * turb_multiplier
        q_rad_provisional = q_rad_incident
        q_total_provisional = max(q_conv_provisional + q_rad_provisional - radiative_relief, 0.0)
        
        if q_total_provisional > 0:
            delta_T_pyro = max(surface_temperature - ablative_config.pyrolysis_temperature, 0.0)
            energy_per_mass = ablative_config.heat_of_ablation + ablative_config.specific_heat * delta_T_pyro
            
            if energy_per_mass > 0:
                # Provisional mass flux [kg/(m²·s)]
                mass_flux_provisional = q_total_provisional / max(energy_per_mass, EPSILON_SMALL)
                # Pyrolysis gas mass flow rate [kg/s]
                m_dot_pyrolysis = mass_flux_provisional * surface_area
                
                # Step 2: Compute blowing parameter B = m_dot_pyrolysis / m_dot_external
                B = m_dot_pyrolysis / max(gas_mass_flow_rate, EPSILON_SMALL)
                
                # Step 3: Apply empirical function f(B) = 1/(1 + c*B)
                # This gives the fraction of convective heat transfer that remains
                blowing_reduction_factor = 1.0 / (1.0 + ablative_config.blowing_coefficient * B)
                # Cap reduction to prevent unrealistic blowing effectiveness
                # Maximum reduction = 1 - blowing_min_reduction_factor
                # (e.g., min_reduction=0.1 means max 90% reduction)
                convective_reduction = max(
                    blowing_reduction_factor,
                    ablative_config.blowing_min_reduction_factor
                )
            else:
                convective_reduction = 1.0
        else:
            convective_reduction = 1.0
    elif not use_physics_blowing:
        # Legacy constant factor already computed above
        pass
    
    # Apply turbulence and blowing ONLY to convective component
    q_conv_effective = q_conv_incident * turb_multiplier * convective_reduction
    
    # Total effective heat flux = (convective + radiative) - radiative relief
    effective_heat_flux = max(q_conv_effective + q_rad_incident - radiative_relief, 0.0)

    # ========================================================================
    # ABLATION PHYSICS
    # ========================================================================
    # Ablation only occurs when surface temperature exceeds pyrolysis temperature
    # (below_pyrolysis already computed earlier)
    if below_pyrolysis or effective_heat_flux <= 0:
        recession_rate = 0.0
        mass_flux = 0.0
        cooling_power = 0.0
    else:
        # Energy required per unit mass: heat of ablation + sensible heat
        delta_T_pyro = max(surface_temperature - ablative_config.pyrolysis_temperature, 0.0)
        energy_per_mass = ablative_config.heat_of_ablation + ablative_config.specific_heat * delta_T_pyro
        
        if energy_per_mass <= 0:
            recession_rate = 0.0
            mass_flux = 0.0
            cooling_power = 0.0
        else:
            # Mass flux: ṁ'' = q_effective / H_ablation
            mass_flux = effective_heat_flux / max(energy_per_mass, EPSILON_SMALL)
            # Recession rate: ṙ = ṁ'' / ρ (safe division)
            if ablative_config.material_density > 0:
                recession_rate = mass_flux / ablative_config.material_density
            else:
                recession_rate = 0.0  # Invalid density
            # Cooling power: P = q_effective × A [W]
            cooling_power = effective_heat_flux * surface_area

    # Build result dict
    result = {
        "enabled": True,
        "recession_rate": float(recession_rate),
        "mass_flux": float(mass_flux),
        "surface_temperature": float(surface_temperature),
        "effective_heat_flux": float(effective_heat_flux),
        "radiative_relief": float(radiative_relief),
        "cooling_power": float(cooling_power),  # Power [W], not energy
        "heat_removed": float(cooling_power),  # Backward compatibility alias
        "turbulence_multiplier": turb_multiplier,
        "below_pyrolysis": below_pyrolysis,
        "heat_flux_from_gas_radiative": float(q_rad_incident),
        "heat_flux_from_gas_convective": float(q_conv_incident),
        "pyrolysis_temperature": float(ablative_config.pyrolysis_temperature),
    }
    
    # Add optional fields
    if not below_pyrolysis and cooling_power > 0:
        result["mass_flow"] = float(mass_flux * surface_area)
        result["coverage_area"] = float(surface_area)
    
    return result
