"""Comprehensive thermal analysis demonstration for ablative and graphite systems.

This script demonstrates:
1. Multi-layer thermal conduction (stainless steel + phenolic + graphite)
2. Pyrolysis modeling with char layer formation
3. Vaporization at high temperatures
4. Ablative thickness sizing
5. Graphite insert sizing
6. Temperature profiles through wall
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from pintle_pipeline.io import load_config
from pintle_pipeline.thermal_analysis import (
    MaterialLayer,
    ThermalBoundaryConditions,
    analyze_multi_layer_system,
    calculate_required_ablative_thickness,
)
from pintle_pipeline.ablative_sizing import size_ablative_system, size_graphite_insert
from pintle_models.runner import PintleEngineRunner

# Load configuration
config = load_config('examples/pintle_engine/config_minimal.yaml')

print("=" * 80)
print("COMPREHENSIVE THERMAL ANALYSIS: ABLATIVE + GRAPHITE SYSTEM")
print("=" * 80)

# Run engine to get operating conditions
runner = PintleEngineRunner(config)
P_tank = 500.0 * 6894.76  # 500 psi
results = runner.evaluate(P_tank, P_tank)

Pc = results.get("Pc")
Tc = results.get("Tc")
mdot_total = results.get("mdot_total")

print(f"\nEngine Operating Conditions:")
print(f"  Pc: {Pc/1e6:.3f} MPa ({Pc/6894.76:.1f} psi)")
print(f"  Tc: {Tc:.1f} K")
print(f"  mdot: {mdot_total:.3f} kg/s")

# Estimate heat flux (using Bartz correlation approximation)
# q'' ≈ 0.026 * (Pc^0.8) * (mdot^0.2) / (Dt^0.9) * (Tc/Tw)^0.68
A_throat = config.chamber.A_throat
Dt = np.sqrt(4 * A_throat / np.pi)
Pc_bar = Pc / 1e5  # bar
mdot_kg_s = mdot_total
Tw_estimate = 1200.0  # K (typical ablative surface temp)

q_chamber = 0.026 * (Pc_bar ** 0.8) * (mdot_kg_s ** 0.2) / ((Dt * 100) ** 0.9) * ((Tc / Tw_estimate) ** 0.68)
q_chamber = q_chamber * 1e6  # Convert to W/m²

# Throat heat flux (higher)
q_throat = q_chamber * 1.5  # Typical throat multiplier

print(f"\nHeat Flux Estimates:")
print(f"  Chamber: {q_chamber/1e6:.2f} MW/m²")
print(f"  Throat:  {q_throat/1e6:.2f} MW/m²")

# ============================================================================
# 1. ABLATIVE LINER SIZING
# ============================================================================
print(f"\n{'='*80}")
print("1. ABLATIVE LINER SIZING")
print("=" * 80)

ablative_config = config.ablative_cooling
burn_time = 10.0  # seconds

sizing_results = size_ablative_system(
    heat_flux=q_chamber,
    burn_time=burn_time,
    ablative_config=ablative_config,
    backface_temp_limit=500.0,  # K - Max for stainless steel
    T_hot_gas=Tc,
    h_hot_gas=5000.0,  # W/(m²·K) - Typical for rocket chambers
    q_rad_hot=0.0,  # Negligible for LOX/RP-1
)

print(f"\nAblative Liner Sizing Results:")
print(f"  Required Thickness: {sizing_results['required_thickness']*1000:.2f} mm")
print(f"  Recession Allowance: {sizing_results['recession_allowance']*1000:.2f} mm")
print(f"  Conduction Thickness: {sizing_results['conduction_thickness']*1000:.2f} mm")
print(f"  Safety Margin: {sizing_results['safety_margin']*1000:.2f} mm")
print(f"  Recession Rate: {sizing_results['recession_rate']*1e6:.2f} µm/s")

thermal = sizing_results['thermal_analysis']
print(f"\nThermal Analysis:")
print(f"  Hot Surface Temp: {thermal['T_surface_hot']:.1f} K")
print(f"  Backface Temp: {thermal['T_backface']:.1f} K")
print(f"  Backface Limit: {sizing_results['backface_temp_limit']:.1f} K")
print(f"  Meets Requirements: {'YES' if sizing_results['meets_requirements'] else 'NO'}")

if thermal.get('pyrolysis', {}).get('pyrolysis_active', False):
    pyro = thermal['pyrolysis']
    print(f"\nPyrolysis Active:")
    print(f"  Pyrolysis Rate: {pyro['pyrolysis_rate']*1000:.3f} g/(m²·s)")
    print(f"  Char Thickness: {pyro['char_thickness']*1000:.2f} mm")

# ============================================================================
# 2. GRAPHITE INSERT SIZING
# ============================================================================
print(f"\n{'='*80}")
print("2. GRAPHITE INSERT SIZING")
print("=" * 80)

if hasattr(config, 'graphite_insert') and config.graphite_insert.enabled:
    graphite_config = config.graphite_insert
    
    # Estimate throat surface temperature and recession rate
    T_throat_surface = 2000.0  # K (typical for graphite)
    
    # Estimate recession rate from graphite cooling model
    from pintle_pipeline.graphite_cooling import compute_graphite_recession
    
    graphite_recession = compute_graphite_recession(
        net_heat_flux=q_throat,
        throat_temperature=T_throat_surface,
        gas_temperature=Tc,
        graphite_config=graphite_config,
        throat_area=A_throat,
        pressure=Pc,
    )
    
    recession_rate_graphite = graphite_recession.get("recession_rate", 1e-5)  # m/s
    
    graphite_sizing = size_graphite_insert(
        peak_heat_flux=q_throat,
        surface_temperature=T_throat_surface,
        recession_rate=recession_rate_graphite,
        burn_time=burn_time,
        graphite_config=graphite_config,
        backface_temp_limit=500.0,
    )
    
    sizing = graphite_sizing['sizing']
    print(f"\nGraphite Insert Sizing Results:")
    print(f"  Initial Thickness: {sizing.initial_thickness*1000:.2f} mm")
    print(f"  Recession Allowance: {sizing.recession_allowance*1000:.2f} mm")
    print(f"  Conduction Thickness: {sizing.conduction_thickness*1000:.2f} mm")
    print(f"  Safety Margin: {sizing.safety_margin*1000:.2f} mm")
    print(f"  Total Axial Length: {sizing.total_axial_length*1000:.2f} mm")
    print(f"  Throat Area Change: {sizing.throat_area_change_pct:.2f}%")
    
    thermal_graphite = graphite_sizing['thermal_analysis']
    print(f"\nThermal Analysis:")
    print(f"  Hot Surface Temp: {thermal_graphite['T_surface_hot']:.1f} K")
    print(f"  Backface Temp: {thermal_graphite['T_backface']:.1f} K")
    print(f"  Meets Requirements: {'YES' if graphite_sizing['meets_requirements'] else 'NO'}")
    
    if graphite_recession.get("oxidation_rate", 0.0) > 0:
        print(f"\nOxidation Active:")
        print(f"  Oxidation Rate: {graphite_recession['oxidation_rate']*1e6:.2f} µm/s")
        print(f"  Thermal Recession: {graphite_recession['recession_rate_thermal']*1e6:.2f} µm/s")

# ============================================================================
# 3. TEMPERATURE PROFILES
# ============================================================================
print(f"\n{'='*80}")
print("3. TEMPERATURE PROFILES THROUGH WALL")
print("=" * 80)

# Ablative + Stainless system
ablative_layer = MaterialLayer(
    name="Phenolic Ablator",
    thickness=sizing_results['required_thickness'],
    thermal_conductivity=ablative_config.thermal_conductivity,
    density=ablative_config.material_density,
    specific_heat=ablative_config.specific_heat,
    emissivity=0.85,
    pyrolysis_temp=ablative_config.pyrolysis_temperature,
)

stainless_layer = MaterialLayer(
    name="Stainless Steel",
    thickness=0.002,  # 2 mm
    thermal_conductivity=15.0,
    density=8000.0,
    specific_heat=500.0,
    emissivity=0.3,
)

layers_ablative = [ablative_layer, stainless_layer]

bc_chamber = ThermalBoundaryConditions(
    T_hot_gas=Tc,
    h_hot_gas=5000.0,
    q_rad_hot=0.0,
    T_ambient=300.0,
    h_ambient=10.0,
)

temp_profile_ablative = analyze_multi_layer_system(
    layers_ablative,
    bc_chamber,
    ablative_config=ablative_config,
)

print(f"\nAblative System Temperature Profile:")
profile = temp_profile_ablative['temperature_profile']
positions = profile['positions']
temperatures = profile['temperatures']
print(f"  Positions: {positions*1000} mm")
print(f"  Temperatures: {temperatures} K")
print(f"  Max temp: {np.max(temperatures):.1f} K (at {positions[np.argmax(temperatures)]*1000:.2f} mm)")
print(f"  Min temp: {np.min(temperatures):.1f} K (at {positions[np.argmin(temperatures)]*1000:.2f} mm)")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Thermal Analysis: Ablative + Graphite System', fontsize=14, fontweight='bold')

# Plot 1: Temperature profile through ablative wall
ax = axes[0, 0]
ax.plot(positions * 1000, temperatures, 'b-', linewidth=2, label='Temperature')
ax.axvline(positions[-1] - stainless_layer.thickness * 1000, color='r', linestyle='--', label='Ablative/Steel Interface')
ax.set_xlabel('Distance from Hot Surface (mm)')
ax.set_ylabel('Temperature (K)')
ax.set_title('Temperature Profile: Ablative + Stainless')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Heat flux components
ax = axes[0, 1]
components = ['Convective', 'Radiative', 'Total']
fluxes = [
    temp_profile_ablative['heat_flux'] * 0.9,  # Approximate
    temp_profile_ablative['heat_flux'] * 0.1,
    temp_profile_ablative['heat_flux'],
]
ax.bar(components, [f/1e6 for f in fluxes], color=['blue', 'red', 'green'])
ax.set_ylabel('Heat Flux (MW/m²)')
ax.set_title('Heat Flux Components')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Material consumption over time
ax = axes[1, 0]
times = np.linspace(0, burn_time, 100)
recession_ablative = sizing_results['recession_rate'] * times
ax.plot(times, recession_ablative * 1000, 'b-', linewidth=2, label='Ablative Recession')
if hasattr(config, 'graphite_insert') and config.graphite_insert.enabled:
    recession_graphite = recession_rate_graphite * times
    ax.plot(times, recession_graphite * 1000, 'r-', linewidth=2, label='Graphite Recession')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Recession (mm)')
ax.set_title('Material Recession Over Time')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 4: Thickness breakdown
ax = axes[1, 1]
thickness_components = ['Recession\nAllowance', 'Conduction\nThickness', 'Safety\nMargin']
thickness_values = [
    sizing_results['recession_allowance'] * 1000,
    sizing_results['conduction_thickness'] * 1000,
    sizing_results['safety_margin'] * 1000,
]
ax.bar(thickness_components, thickness_values, color=['orange', 'blue', 'green'])
ax.set_ylabel('Thickness (mm)')
ax.set_title('Ablative Thickness Breakdown')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save plot
output_path = Path(__file__).parent / "thermal_analysis_results.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Plot saved to: {output_path}")

plt.show()

print(f"\n{'='*80}")
print("THERMAL ANALYSIS COMPLETE")
print("=" * 80)

