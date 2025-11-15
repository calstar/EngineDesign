"""Stability analysis for combustion and feed system dynamics.

This module provides:
1. Combustion stability analysis (chugging, acoustic modes)
2. Feed system stability (POGO, surge, water hammer)
3. Stability margins and design recommendations
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, List
import numpy as np
from scipy.optimize import root_scalar
from pintle_pipeline.config_schemas import PintleEngineConfig


def calculate_chugging_frequency(
    chamber_volume: float,
    throat_area: float,
    cstar: float,
    gamma: float,
    Pc: float,
) -> Dict[str, float]:
    """
    Calculate chugging (low-frequency combustion instability) characteristics.
    
    Chugging occurs when feed system and combustion chamber couple, causing
    pressure oscillations at frequencies typically 10-100 Hz.
    
    Parameters:
    -----------
    chamber_volume : float
        Chamber volume [m³]
    throat_area : float
        Throat area [m²]
    cstar : float
        Characteristic velocity [m/s]
    gamma : float
        Specific heat ratio
    Pc : float
        Chamber pressure [Pa]
    
    Returns:
    --------
    results : dict
        - frequency: Chugging frequency [Hz]
        - period: Oscillation period [s]
        - stability_margin: Stability margin (positive = stable)
        - damping_ratio: Estimated damping ratio
    """
    # Characteristic time for chamber filling/emptying
    # τ = V_chamber / (mdot / rho) = V_chamber * rho / mdot
    # But mdot = Pc * At / cstar, so τ = V_chamber * rho * cstar / (Pc * At)
    # More accurately: τ = V_chamber / (A_throat × c*_effective)
    # where c*_effective accounts for actual gas density
    
    # Gas density at chamber conditions
    # Estimate using ideal gas law: Pc = rho * R * Tc
    # Need Tc - estimate from cstar: Tc ≈ cstar² / (gamma * R)
    # But we don't have Tc here, so use simplified: tau = V / (A * cstar) * (Pc / Pc_ref)^0.8
    
    # Simplified: τ ≈ V_chamber / (A_throat × characteristic_velocity)
    # where characteristic_velocity ≈ cstar
    # But account for gas compressibility: use actual residence time
    # τ = V_chamber * rho / mdot = V_chamber * Pc / (mdot * R * Tc)
    
    # More accurate: use L* formulation
    # τ ≈ L* / cstar (residence time)
    Lstar = chamber_volume / throat_area if throat_area > 0 else 0.1
    
    # Chugging frequency from residence time
    # f_chug ≈ 1 / (2π × τ_residence)
    # where τ_residence = L* / cstar
    if Lstar > 0 and cstar > 0:
        tau_residence = Lstar / cstar
        # Frequency (Hz): f = 1 / (2π × τ)
        frequency = 1.0 / (2.0 * np.pi * tau_residence) if tau_residence > 0 else 0.0
        
        # Alternative: use chamber volume formulation if more accurate
        # tau_chamber = chamber_volume * Pc / (throat_area * cstar)
        # But ensure it's physically reasonable
        tau_chamber_alt = chamber_volume * Pc / (throat_area * cstar) if throat_area > 0 else 0.0
        
        # Use the more accurate formulation (residence time based)
        # But clamp to reasonable values
        if tau_chamber_alt > 0 and tau_chamber_alt < 100.0:  # Reasonable bounds
            tau_chamber = tau_chamber_alt
            frequency = 1.0 / (2.0 * np.pi * tau_chamber)
        else:
            # Fall back to residence time
            tau_chamber = tau_residence
    else:
        tau_chamber = 0.01  # Fallback: 10 ms
        frequency = 1.0 / (2.0 * np.pi * tau_chamber)
    
    # Clamp frequency to reasonable range (1-500 Hz for chugging)
    frequency = np.clip(frequency, 1.0, 500.0)
    
    # Period
    period = 1.0 / frequency if frequency > 0 else np.inf
    
    # Stability criterion: system is stable if damping > 0
    # Damping comes from:
    # 1. Feed system resistance
    # 2. Combustion time lag
    # 3. Acoustic losses
    
    # Simplified damping estimate (empirical)
    # Higher Pc → more stable (higher pressure head)
    # Higher L* → more stable (longer residence time)
    Lstar = chamber_volume / throat_area
    damping_ratio = 0.1 * (Pc / 1e6) ** 0.3 * (Lstar / 1.0) ** 0.2
    
    # Stability margin: positive = stable
    # Margin = damping_ratio - critical_damping (0.05 typical)
    critical_damping = 0.05
    stability_margin = damping_ratio - critical_damping
    
    return {
        "frequency": float(frequency),
        "period": float(period),
        "stability_margin": float(stability_margin),
        "damping_ratio": float(damping_ratio),
        "tau_chamber": float(tau_chamber),
        "critical_damping": float(critical_damping),
    }


def calculate_acoustic_modes(
    chamber_length: float,
    chamber_diameter: float,
    gas_temperature: float,
    gamma: float,
    R: float,
) -> Dict[str, List[float]]:
    """
    Calculate acoustic resonance frequencies for longitudinal and transverse modes.
    
    Combustion instabilities can couple with acoustic modes, causing
    high-frequency oscillations (100-5000 Hz).
    
    Parameters:
    -----------
    chamber_length : float
        Chamber length [m]
    chamber_diameter : float
        Chamber diameter [m]
    gas_temperature : float
        Gas temperature [K]
    gamma : float
        Specific heat ratio
    R : float
        Gas constant [J/(kg·K)]
    
    Returns:
    --------
    results : dict
        - longitudinal_modes: List of longitudinal mode frequencies [Hz]
        - transverse_modes: List of first few transverse mode frequencies [Hz]
        - sound_speed: Sound speed in chamber [m/s]
    """
    # Sound speed
    sound_speed = np.sqrt(gamma * R * gas_temperature)
    
    # Longitudinal modes (1D acoustic waves)
    # f_n = n × c / (2 × L) for open-open or closed-closed
    # For rocket chambers (open-closed): f_n = (2n-1) × c / (4 × L)
    longitudinal_modes = []
    for n in range(1, 6):  # First 5 modes
        freq = (2 * n - 1) * sound_speed / (4 * chamber_length)
        longitudinal_modes.append(float(freq))
    
    # Transverse modes (radial modes in cylindrical chamber)
    # f_mn = α_mn × c / (π × D)
    # where α_mn are Bessel function roots
    # First few: α_01 ≈ 2.405, α_11 ≈ 3.832, α_21 ≈ 5.136
    alpha_values = [2.405, 3.832, 5.136, 6.380, 7.588]
    transverse_modes = []
    for alpha in alpha_values:
        freq = alpha * sound_speed / (np.pi * chamber_diameter)
        transverse_modes.append(float(freq))
    
    return {
        "longitudinal_modes": longitudinal_modes,
        "transverse_modes": transverse_modes,
        "sound_speed": float(sound_speed),
    }


def analyze_feed_system_stability(
    feed_line_length: float,
    feed_line_diameter: float,
    propellant_density: float,
    bulk_modulus: float,
    flow_velocity: float,
    pressure_drop: float,
) -> Dict[str, float]:
    """
    Analyze feed system stability (POGO, surge, water hammer).
    
    POGO (pogo oscillation) occurs when feed system and vehicle structure
    couple, causing low-frequency oscillations.
    
    Parameters:
    -----------
    feed_line_length : float
        Feed line length [m]
    feed_line_diameter : float
        Feed line diameter [m]
    propellant_density : float
        Propellant density [kg/m³]
    bulk_modulus : float
        Bulk modulus [Pa] (for liquids, ~1-2 GPa)
    flow_velocity : float
        Flow velocity [m/s]
    pressure_drop : float
        Pressure drop across feed system [Pa]
    
    Returns:
    --------
    results : dict
        - pogo_frequency: POGO frequency [Hz]
        - surge_frequency: Surge frequency [Hz]
        - water_hammer_pressure: Water hammer pressure spike [Pa]
        - stability_margin: Stability margin
    """
    # POGO frequency (feed system natural frequency)
    # f_pogo ≈ (1 / (2π)) × √(K_effective / m_effective)
    # Simplified: f_pogo ≈ c_sound / (4 × L)
    # where c_sound = √(bulk_modulus / density)
    
    sound_speed = np.sqrt(bulk_modulus / propellant_density)
    pogo_frequency = sound_speed / (4.0 * feed_line_length)
    
    # Surge frequency (sloshing in tanks/feed lines)
    # f_surge ≈ (1 / (2π)) × √(g / L_effective)
    # Simplified for feed lines
    g = 9.80665
    surge_frequency = (1.0 / (2.0 * np.pi)) * np.sqrt(g / feed_line_length)
    
    # Water hammer pressure spike
    # ΔP_hammer = ρ × c × Δv
    # For sudden valve closure
    delta_v = flow_velocity  # Assume full stop
    water_hammer_pressure = propellant_density * sound_speed * delta_v
    
    # Stability margin (simplified)
    # System is stable if pressure drop >> water hammer pressure
    stability_margin = pressure_drop / water_hammer_pressure if water_hammer_pressure > 0 else np.inf
    
    return {
        "pogo_frequency": float(pogo_frequency),
        "surge_frequency": float(surge_frequency),
        "water_hammer_pressure": float(water_hammer_pressure),
        "stability_margin": float(stability_margin),
        "sound_speed": float(sound_speed),
    }


def comprehensive_stability_analysis(
    config: PintleEngineConfig,
    Pc: float,
    MR: float,
    mdot_total: float,
    cstar: float,
    gamma: float,
    R: float,
    Tc: float,
    diagnostics: Dict,
) -> Dict[str, any]:
    """
    Comprehensive stability analysis combining all modes.
    
    Parameters:
    -----------
    config : PintleEngineConfig
        Engine configuration
    Pc : float
        Chamber pressure [Pa]
    MR : float
        Mixture ratio
    mdot_total : float
        Total mass flow [kg/s]
    cstar : float
        Characteristic velocity [m/s]
    gamma : float
        Specific heat ratio
    R : float
        Gas constant [J/(kg·K)]
    Tc : float
        Chamber temperature [K]
    diagnostics : dict
        Engine diagnostics (from solver)
    
    Returns:
    --------
    results : dict
        Complete stability analysis results
    """
    # Chamber geometry
    V_chamber = config.chamber.volume
    A_throat = config.chamber.A_throat
    Lstar = V_chamber / A_throat
    
    # Estimate chamber dimensions
    L_chamber = config.chamber.length if hasattr(config.chamber, 'length') and config.chamber.length else 0.18
    D_chamber = np.sqrt(4 * V_chamber / (np.pi * L_chamber))
    
    # Combustion stability
    chugging = calculate_chugging_frequency(V_chamber, A_throat, cstar, gamma, Pc)
    acoustic = calculate_acoustic_modes(L_chamber, D_chamber, Tc, gamma, R)
    
    # Feed system stability (simplified - use LOX feed as representative)
    if config.feed_system:
        # Handle both dict and object access
        if isinstance(config.feed_system, dict):
            lox_config = config.feed_system.get('lox', {})
            feed_length = lox_config.get('length', 1.0) if isinstance(lox_config, dict) else getattr(lox_config, 'length', 1.0)
            feed_diameter = lox_config.get('d_inlet', 0.01) if isinstance(lox_config, dict) else getattr(lox_config, 'd_inlet', 0.01)
        else:
            lox_config = getattr(config.feed_system, 'lox', None)
            if lox_config:
                feed_length = getattr(lox_config, 'length', 1.0)
                feed_diameter = getattr(lox_config, 'd_inlet', 0.01)
            else:
                feed_length = 1.0
                feed_diameter = 0.01
        
        # Get propellant density
        if hasattr(config, 'propellants') and config.propellants:
            if isinstance(config.propellants, dict):
                prop_density = config.propellants.get('oxidizer', {}).get('density', 1140.0)
            else:
                prop_density = getattr(config.propellants.oxidizer, 'density', 1140.0)
        else:
            prop_density = 1140.0  # Default LOX density [kg/m³]
        bulk_modulus = 1.5e9  # Typical for LOX [Pa]
        
        # Estimate flow velocity
        A_feed = np.pi * (feed_diameter / 2) ** 2
        mdot_ox = diagnostics.get("mdot_O", mdot_total * MR / (1 + MR))
        flow_velocity = mdot_ox / (prop_density * A_feed) if A_feed > 0 else 0.0
        
        # Estimate pressure drop
        P_tank_O = diagnostics.get("P_tank_O", Pc * 2.0)  # Rough estimate
        pressure_drop = P_tank_O - Pc
        
        feed_stability = analyze_feed_system_stability(
            feed_length, feed_diameter, prop_density, bulk_modulus,
            flow_velocity, pressure_drop
        )
    else:
        feed_stability = {
            "pogo_frequency": np.nan,
            "surge_frequency": np.nan,
            "water_hammer_pressure": np.nan,
            "stability_margin": np.nan,
        }
    
    # Overall stability assessment
    is_stable = (
        chugging["stability_margin"] > 0 and
        feed_stability.get("stability_margin", 1.0) > 1.0
    )
    
    # Identify potential issues
    issues = []
    if chugging["stability_margin"] < 0:
        issues.append("Chugging instability risk (low damping)")
    if chugging["frequency"] < 10 or chugging["frequency"] > 200:
        issues.append(f"Chugging frequency out of typical range: {chugging['frequency']:.1f} Hz")
    if feed_stability.get("stability_margin", 1.0) < 1.0:
        issues.append("Feed system stability risk (POGO/surge)")
    
    return {
        "is_stable": is_stable,
        "chugging": chugging,
        "acoustic": acoustic,
        "feed_system": feed_stability,
        "issues": issues,
        "recommendations": _generate_stability_recommendations(chugging, acoustic, feed_stability),
    }


def _generate_stability_recommendations(
    chugging: Dict,
    acoustic: Dict,
    feed_system: Dict,
) -> List[str]:
    """Generate stability improvement recommendations."""
    recommendations = []
    
    if chugging["stability_margin"] < 0:
        recommendations.append("Increase chamber pressure or L* to improve chugging stability")
        recommendations.append("Consider adding damping devices (baffles, acoustic liners)")
    
    if chugging["frequency"] < 10:
        recommendations.append("Very low chugging frequency - check feed system coupling")
    
    if feed_system.get("stability_margin", 1.0) < 1.0:
        recommendations.append("Add surge suppressors or accumulators to feed system")
        recommendations.append("Consider increasing feed line diameter to reduce flow velocity")
    
    # Check for mode coupling
    if chugging["frequency"] > 0:
        for mode_freq in acoustic["longitudinal_modes"][:3]:
            if abs(mode_freq - chugging["frequency"]) < 50:  # Within 50 Hz
                recommendations.append(f"Potential mode coupling: chugging ({chugging['frequency']:.1f} Hz) near acoustic mode ({mode_freq:.1f} Hz)")
    
    if not recommendations:
        recommendations.append("System appears stable - monitor during hot-fire testing")
    
    return recommendations

