"""Regenerative cooling channel pressure drop model"""

import numpy as np
from typing import Optional
from .config_schemas import RegenCoolingConfig


def calculate_channel_hydraulic_diameter(width: float, height: float) -> float:
    """
    Calculate hydraulic diameter for rectangular channel.
    
    d_hyd = 4 × A / P_wetted
    For rectangle: d_hyd = 2 × w × h / (w + h)
    
    Parameters:
    -----------
    width : float
        Channel width [m]
    height : float
        Channel height [m]
    
    Returns:
    --------
    d_hyd : float
        Hydraulic diameter [m]
    """
    if width <= 0 or height <= 0:
        raise ValueError("Channel dimensions must be positive")
    
    d_hyd = 2 * width * height / (width + height)
    return float(d_hyd)


def calculate_friction_factor(Re: float, d_hyd: float, roughness: float = 0.0) -> float:
    """
    Calculate Darcy-Weisbach friction factor.
    
    For smooth pipes (roughness = 0): Blasius correlation
    f = 0.316 / Re^0.25  (for Re < 10^5)
    
    For rough pipes: Colebrook equation (iterative)
    Or Swamee-Jain approximation
    
    Parameters:
    -----------
    Re : float
        Reynolds number
    d_hyd : float
        Hydraulic diameter [m]
    roughness : float
        Surface roughness [m] (default: 0 = smooth)
    
    Returns:
    --------
    f : float
        Friction factor
    """
    if Re <= 0:
        return 1.0  # Avoid division by zero
    
    if roughness == 0 or roughness / d_hyd < 1e-6:
        # Smooth pipe: Blasius correlation
        if Re < 1e5:
            f = 0.316 / (Re ** 0.25)
        else:
            # For higher Re, use more accurate correlation
            f = 0.184 / (Re ** 0.2)
    else:
        # Rough pipe: Swamee-Jain approximation
        relative_roughness = roughness / d_hyd
        f = 0.25 / (np.log10(relative_roughness / 3.7 + 5.74 / (Re ** 0.9))) ** 2
    
    return float(f)


def delta_p_regen_channels(
    mdot: float,
    rho: float,
    mu: float,
    config: RegenCoolingConfig,
    P_tank: float
) -> float:
    """
    Calculate pressure drop through regenerative cooling channels.
    
    Flow path:
    1. Inlet pipe (3/8" diameter)
    2. Split into N parallel channels
    3. N channels (each with friction)
    4. Merge back
    5. Pipe to injector
    
    Pressure drop is a function of tank pressure:
    - Higher tank pressure → higher mass flow → higher velocity → higher pressure drop
    - This is already captured through mdot(P_tank), but we add explicit pressure scaling
    - Channel length should match chamber length (cooling channels go across the length of the chamber)
    
    Parameters:
    -----------
    mdot : float
        Total mass flow rate [kg/s] (already depends on P_tank)
    rho : float
        Fluid density [kg/m³]
    mu : float
        Dynamic viscosity [Pa·s]
    config : RegenCoolingConfig
        Regenerative cooling configuration
    P_tank : float
        Tank pressure [Pa] - used for pressure-dependent scaling
    
    Returns:
    --------
    delta_p : float
        Total pressure drop [Pa]
    """
    # Note: Pressure dependence is already captured through mdot(P_tank)
    # Higher tank pressure → higher mdot → higher velocity → higher pressure drop
    # No additional scaling factor needed - the physics is already in the equations
    
    # 1. Inlet pipe (3/8" = 9.525 mm)
    d_inlet = config.d_inlet  # m
    A_inlet = np.pi * (d_inlet / 2) ** 2
    u_inlet = mdot / (rho * A_inlet)
    d_hyd_inlet = d_inlet
    Re_inlet = (rho * u_inlet * d_hyd_inlet) / mu
    f_inlet = calculate_friction_factor(Re_inlet, d_hyd_inlet, config.roughness)
    
    # Pressure drop in inlet pipe
    # Δp = f × (L/D) × (ρ/2) × u²
    delta_p_inlet = f_inlet * (config.L_inlet / d_hyd_inlet) * (rho / 2) * u_inlet ** 2
    
    # Add minor loss for inlet (entrance loss)
    K_entrance = 0.5  # Sharp entrance
    delta_p_inlet += K_entrance * (rho / 2) * u_inlet ** 2
    
    # 2. Split loss (manifold)
    # Loss coefficient for flow splitting
    K_split = config.K_manifold_split
    delta_p_split = K_split * (rho / 2) * u_inlet ** 2
    
    # 3. N parallel channels
    # Each channel gets mdot / N
    mdot_channel = mdot / config.n_channels
    
    # Channel geometry
    d_hyd_channel = calculate_channel_hydraulic_diameter(
        config.channel_width,
        config.channel_height
    )
    A_channel = config.channel_width * config.channel_height
    
    # Channel entrance loss (discharge coefficient effect)
    # Flow contracts entering small channel, creating entrance loss
    # Cd varies with Reynolds number (flow contraction depends on Re)
    # Calculate Re for entrance (use channel velocity estimate)
    u_channel_estimate = mdot_channel / (rho * A_channel)
    Re_channel_entrance = (rho * u_channel_estimate * d_hyd_channel) / mu
    
    # Dynamic Cd for entrance: Cd(Re) = Cd_∞ - a_Re/√Re (same model as injector)
    if Re_channel_entrance <= 0:
        Cd_entrance_dynamic = config.Cd_entrance_min
    else:
        Cd_entrance_dynamic = config.Cd_entrance_inf - config.a_Re_entrance / np.sqrt(Re_channel_entrance)
        Cd_entrance_dynamic = np.clip(Cd_entrance_dynamic, config.Cd_entrance_min, config.Cd_entrance_inf)
    
    # Effective area reduction: A_eff = Cd_entrance × A_channel
    A_eff_entrance = Cd_entrance_dynamic * A_channel
    u_channel_entrance = mdot_channel / (rho * A_eff_entrance)
    
    # Entrance loss: Δp = (1/Cd² - 1) × (ρ/2) × u²
    # This accounts for flow contraction at channel entrance
    K_entrance_channel = (1.0 / (Cd_entrance_dynamic ** 2) - 1.0)
    delta_p_entrance_channel = K_entrance_channel * (rho / 2) * u_channel_entrance ** 2
    
    # Channel flow (after entrance)
    u_channel = mdot_channel / (rho * A_channel)
    Re_channel = (rho * u_channel * d_hyd_channel) / mu
    f_channel = calculate_friction_factor(Re_channel, d_hyd_channel, config.roughness)
    
    # Pressure drop in one channel (friction)
    # Δp = f × (L/D) × (ρ/2) × u²
    # Channel length should be the chamber length (cooling channels go across the length of the chamber)
    delta_p_channel_friction = f_channel * (config.channel_length / d_hyd_channel) * (rho / 2) * u_channel ** 2
    
    # Channel exit loss (discharge coefficient effect)
    # Flow expands exiting channel, creating exit loss
    # Cd varies with Reynolds number (flow expansion depends on Re)
    # Re_channel is already calculated above
    
    # Dynamic Cd for exit: Cd(Re) = Cd_∞ - a_Re/√Re (same model as injector)
    if Re_channel <= 0:
        Cd_exit_dynamic = config.Cd_exit_min
    else:
        Cd_exit_dynamic = config.Cd_exit_inf - config.a_Re_exit / np.sqrt(Re_channel)
        Cd_exit_dynamic = np.clip(Cd_exit_dynamic, config.Cd_exit_min, config.Cd_exit_inf)
    
    K_exit_channel = (1.0 - Cd_exit_dynamic ** 2)
    delta_p_exit_channel = K_exit_channel * (rho / 2) * u_channel ** 2
    
    # Total channel pressure drop (entrance + friction + exit)
    delta_p_channel = delta_p_entrance_channel + delta_p_channel_friction + delta_p_exit_channel
    
    # All channels are parallel, so same pressure drop
    # (flow splits equally if channels are identical)
    
    # 4. Merge loss (manifold)
    # Loss coefficient for flow merging
    K_merge = config.K_manifold_merge
    delta_p_merge = K_merge * (rho / 2) * u_channel ** 2
    
    # 5. Pipe to injector (after merge)
    # Use d_outlet if specified, otherwise same as inlet
    d_outlet = config.d_outlet if config.d_outlet is not None else d_inlet
    A_outlet = np.pi * (d_outlet / 2) ** 2
    u_outlet = mdot / (rho * A_outlet)
    d_hyd_outlet = d_outlet
    Re_outlet = (rho * u_outlet * d_hyd_outlet) / mu
    f_outlet = calculate_friction_factor(Re_outlet, d_hyd_outlet, config.roughness)
    
    delta_p_outlet = f_outlet * (config.L_outlet / d_hyd_outlet) * (rho / 2) * u_outlet ** 2
    
    # Total pressure drop
    delta_p_total = (
        delta_p_inlet +
        delta_p_split +
        delta_p_channel +
        delta_p_merge +
        delta_p_outlet
    )
    
    return float(delta_p_total)


def calculate_regen_velocity_profile(
    mdot: float,
    rho: float,
    config: RegenCoolingConfig
) -> dict:
    """
    Calculate velocity profile through regen cooling system.
    
    Returns velocities at each section for diagnostics.
    
    Parameters:
    -----------
    mdot : float
        Total mass flow rate [kg/s]
    rho : float
        Fluid density [kg/m³]
    config : RegenCoolingConfig
        Regenerative cooling configuration
    
    Returns:
    --------
    velocities : dict
        Dictionary with velocities at each section [m/s]
    """
    d_inlet = config.d_inlet
    A_inlet = np.pi * (d_inlet / 2) ** 2
    u_inlet = mdot / (rho * A_inlet)
    
    mdot_channel = mdot / config.n_channels
    A_channel = config.channel_width * config.channel_height
    u_channel = mdot_channel / (rho * A_channel)
    
    d_outlet = config.d_outlet if config.d_outlet is not None else d_inlet
    A_outlet = np.pi * (d_outlet / 2) ** 2
    u_outlet = mdot / (rho * A_outlet)
    
    return {
        "u_inlet": float(u_inlet),
        "u_channel": float(u_channel),
        "u_outlet": float(u_outlet),
        "mdot_channel": float(mdot_channel),
    }
