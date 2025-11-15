"""Demonstrate how ablative recession changes L* and performance over time."""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pintle_pipeline.io import load_config
from pintle_pipeline.ablative_geometry import (
    calculate_Lstar_time_varying,
    estimate_performance_degradation,
)

# Load configuration
config_path = Path(__file__).parent / "config_minimal.yaml"
config = load_config(str(config_path))

print("=" * 80)
print("ABLATIVE LINER IMPACT ON L* AND PERFORMANCE")
print("=" * 80)

# Engine geometry
V_chamber = config.chamber.volume
A_throat = config.chamber.A_throat
L_chamber = config.chamber.length if config.chamber.length else 0.18  # m

# Calculate initial diameters
D_throat_initial = np.sqrt(4 * A_throat / np.pi)
D_chamber_initial = np.sqrt(4 * V_chamber / (np.pi * L_chamber))

print(f"\nInitial Geometry:")
print(f"  Chamber volume:    {V_chamber*1e6:.1f} cm³")
print(f"  Throat area:       {A_throat*1e6:.2f} mm²")
print(f"  Chamber diameter:  {D_chamber_initial*1000:.2f} mm")
print(f"  Throat diameter:   {D_throat_initial*1000:.2f} mm")
print(f"  Chamber length:    {L_chamber*1000:.1f} mm")
print(f"  Initial L*:        {V_chamber/A_throat*1000:.1f} mm")

# Ablative properties
ablative_cfg = config.ablative_cooling
recession_rate = 50e-6  # m/s (50 µm/s - typical for phenolic ablatives)
burn_time_max = 10.0  # seconds

print(f"\nAblative Properties:")
print(f"  Material:          {ablative_cfg.material_density:.0f} kg/m³ density")
print(f"  Initial thickness: {ablative_cfg.initial_thickness*1000:.1f} mm")
print(f"  Recession rate:    {recession_rate*1e6:.1f} µm/s (assumed)")
print(f"  Coverage fraction: {ablative_cfg.coverage_fraction*100:.0f}%")

# Time series analysis
times = np.linspace(0, burn_time_max, 100)
Lstar_values = []
V_chamber_values = []
A_throat_values = []
eta_values = []
thrust_degradation = []

for t in times:
    results = calculate_Lstar_time_varying(
        V_chamber,
        A_throat,
        recession_rate,
        t,
        D_chamber_initial,
        D_throat_initial,
        L_chamber,
        ablative_cfg.coverage_fraction,
    )
    
    Lstar_values.append(results["Lstar_final"])
    V_chamber_values.append(results["V_chamber_final"])
    A_throat_values.append(results["A_throat_final"])
    
    # Estimate efficiency degradation
    perf = estimate_performance_degradation(
        results["Lstar_initial"],
        results["Lstar_final"],
        config.combustion.efficiency.C,
        config.combustion.efficiency.K,
    )
    eta_values.append(perf["eta_final"])
    thrust_degradation.append(perf["cstar_degradation_pct"])

Lstar_values = np.array(Lstar_values)
V_chamber_values = np.array(V_chamber_values)
A_throat_values = np.array(A_throat_values)
eta_values = np.array(eta_values)
thrust_degradation = np.array(thrust_degradation)

# Final state
final_results = calculate_Lstar_time_varying(
    V_chamber,
    A_throat,
    recession_rate,
    burn_time_max,
    D_chamber_initial,
    D_throat_initial,
    L_chamber,
    ablative_cfg.coverage_fraction,
)

print(f"\n" + "=" * 80)
print(f"AFTER {burn_time_max:.1f} SECONDS OF BURN")
print("=" * 80)

print(f"\nGeometry Changes:")
print(f"  Total recession:     {final_results['total_recession']*1e6:.1f} µm")
print(f"  Chamber diameter:    {final_results['D_chamber_initial']*1000:.2f} → {final_results['D_chamber_final']*1000:.2f} mm  (+{(final_results['D_chamber_final']-final_results['D_chamber_initial'])*1000:.3f} mm)")
print(f"  Throat diameter:     {final_results['D_throat_initial']*1000:.2f} → {final_results['D_throat_final']*1000:.2f} mm  (+{(final_results['D_throat_final']-final_results['D_throat_initial'])*1000:.3f} mm)")
print(f"  Chamber volume:      {final_results['V_chamber_initial']*1e6:.2f} → {final_results['V_chamber_final']*1e6:.2f} cm³  (+{(final_results['V_chamber_final']-final_results['V_chamber_initial'])*1e6:.2f} cm³)")
print(f"  Throat area:         {final_results['A_throat_initial']*1e6:.2f} → {final_results['A_throat_final']*1e6:.2f} mm²  (+{(final_results['A_throat_final']-final_results['A_throat_initial'])*1e6:.2f} mm²)")

print(f"\nL* Changes:")
print(f"  Initial L*:          {final_results['Lstar_initial']*1000:.2f} mm")
print(f"  Final L*:            {final_results['Lstar_final']*1000:.2f} mm")
print(f"  Change:              {final_results['Lstar_change_pct']:+.2f}%")

perf_final = estimate_performance_degradation(
    final_results["Lstar_initial"],
    final_results["Lstar_final"],
    config.combustion.efficiency.C,
    config.combustion.efficiency.K,
)

print(f"\nPerformance Impact:")
print(f"  Initial η_c*:        {perf_final['eta_initial']:.4f}")
print(f"  Final η_c*:          {perf_final['eta_final']:.4f}")
print(f"  Efficiency loss:     {perf_final['eta_degradation_pct']:.2f}%")
print(f"  Approx thrust loss:  {perf_final['cstar_degradation_pct']:.2f}%")

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: L* vs time
ax = axes[0, 0]
ax.plot(times, Lstar_values * 1000, 'b-', linewidth=2)
ax.set_xlabel('Time [s]')
ax.set_ylabel('L* [mm]')
ax.set_title('Characteristic Length Evolution')
ax.grid(True, alpha=0.3)

# Plot 2: Geometry changes
ax = axes[0, 1]
ax.plot(times, (V_chamber_values / V_chamber - 1) * 100, 'r-', label='Chamber Volume', linewidth=2)
ax.plot(times, (A_throat_values / A_throat - 1) * 100, 'g-', label='Throat Area', linewidth=2)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Change [%]')
ax.set_title('Geometry Growth Due to Ablation')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Efficiency degradation
ax = axes[1, 0]
ax.plot(times, eta_values, 'purple', linewidth=2)
ax.set_xlabel('Time [s]')
ax.set_ylabel('η_c*')
ax.set_title('Combustion Efficiency Degradation')
ax.grid(True, alpha=0.3)

# Plot 4: Thrust degradation
ax = axes[1, 1]
ax.plot(times, thrust_degradation, 'orange', linewidth=2)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Thrust Loss [%]')
ax.set_title('Approximate Thrust Degradation')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
output_path = Path(__file__).parent / "ablative_Lstar_impact.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved plot to: {output_path}")

print("\n" + "=" * 80)
print("KEY FINDINGS:")
print("=" * 80)
print(f"1. Throat area grows by {(final_results['A_throat_final']/final_results['A_throat_initial']-1)*100:.2f}% over {burn_time_max}s burn")
print(f"2. L* increases by {final_results['Lstar_change_pct']:.2f}%, reducing combustion efficiency")
print(f"3. Performance degrades by ~{perf_final['cstar_degradation_pct']:.2f}% due to geometry changes")
print(f"4. This effect is NOT currently modeled in time-series simulations!")
print("\nTO FIX: Need to update chamber geometry at each time step in evaluate_arrays()")
print("=" * 80)

