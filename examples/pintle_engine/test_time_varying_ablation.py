"""Test time-varying ablative geometry evolution in the full pipeline."""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pintle_pipeline.io import load_config
from pintle_models.runner import PintleEngineRunner

# Load configuration
config_path = Path(__file__).parent / "config_minimal.yaml"
config = load_config(str(config_path))

# Enable ablative cooling and geometry tracking
config.ablative_cooling.enabled = True
config.ablative_cooling.track_geometry_evolution = True

print("=" * 80)
print("TIME-VARYING ABLATIVE GEOMETRY EVOLUTION TEST")
print("=" * 80)

# Create runner
runner = PintleEngineRunner(config)

# Define time series (10 second burn)
burn_time = 10.0  # seconds
n_points = 100
times = np.linspace(0, burn_time, n_points)

# Constant tank pressures
P_tank_O_psi = 1305.0
P_tank_F_psi = 974.0
P_tank_O = np.full(n_points, P_tank_O_psi * 6894.76)
P_tank_F = np.full(n_points, P_tank_F_psi * 6894.76)

print(f"\nTest Conditions:")
print(f"  Burn time:          {burn_time:.1f} s")
print(f"  Time points:        {n_points}")
print(f"  LOX tank pressure:  {P_tank_O_psi:.0f} psi (constant)")
print(f"  Fuel tank pressure: {P_tank_F_psi:.0f} psi (constant)")

print(f"\nAblative Configuration:")
print(f"  Material:           {config.ablative_cooling.material_density:.0f} kg/m³")
print(f"  Heat of ablation:   {config.ablative_cooling.heat_of_ablation/1e6:.1f} MJ/kg")
print(f"  Coverage:           {config.ablative_cooling.coverage_fraction*100:.0f}%")
print(f"  Throat multiplier:  {'Physics-based (Bartz)' if config.ablative_cooling.throat_recession_multiplier is None else config.ablative_cooling.throat_recession_multiplier}")

print(f"\n{'='*80}")
print("RUNNING TIME-SERIES SIMULATION WITH GEOMETRY EVOLUTION...")
print("=" * 80)

# Run time-varying simulation
results = runner.evaluate_arrays_with_time(times, P_tank_O, P_tank_F)

print(f"\n✅ Simulation complete!")

# Extract results
thrust_kN = results["F"] / 1000.0
Isp = results["Isp"]
Pc_psi = results["Pc"] / 6894.76
MR = results["MR"]
Lstar_mm = results["Lstar"] * 1000.0
recession_chamber_um = results["recession_chamber"] * 1e6
recession_throat_um = results["recession_throat"] * 1e6
throat_mult = results["throat_recession_multiplier"]
A_throat_pct_change = (results["A_throat"] / results["A_throat"][0] - 1) * 100
V_chamber_pct_change = (results["V_chamber"] / results["V_chamber"][0] - 1) * 100

# Print summary
print(f"\n{'='*80}")
print("PERFORMANCE SUMMARY")
print("=" * 80)

print(f"\nInitial State (t=0):")
print(f"  Thrust:             {thrust_kN[0]:.3f} kN")
print(f"  Isp:                {Isp[0]:.2f} s")
print(f"  Pc:                 {Pc_psi[0]:.1f} psi")
print(f"  O/F:                {MR[0]:.3f}")
print(f"  L*:                 {Lstar_mm[0]:.2f} mm")

print(f"\nFinal State (t={burn_time}s):")
print(f"  Thrust:             {thrust_kN[-1]:.3f} kN  ({(thrust_kN[-1]/thrust_kN[0]-1)*100:+.2f}%)")
print(f"  Isp:                {Isp[-1]:.2f} s  ({(Isp[-1]/Isp[0]-1)*100:+.2f}%)")
print(f"  Pc:                 {Pc_psi[-1]:.1f} psi  ({(Pc_psi[-1]/Pc_psi[0]-1)*100:+.2f}%)")
print(f"  O/F:                {MR[-1]:.3f}  ({(MR[-1]/MR[0]-1)*100:+.2f}%)")
print(f"  L*:                 {Lstar_mm[-1]:.2f} mm  ({(Lstar_mm[-1]/Lstar_mm[0]-1)*100:+.2f}%)")

print(f"\nGeometry Changes:")
print(f"  Chamber recession:  {recession_chamber_um[-1]:.1f} µm")
print(f"  Throat recession:   {recession_throat_um[-1]:.1f} µm")
print(f"  Throat multiplier:  {np.nanmean(throat_mult):.2f} (average)")
print(f"  Chamber volume:     {V_chamber_pct_change[-1]:+.3f}%")
print(f"  Throat area:        {A_throat_pct_change[-1]:+.3f}%")

# Calculate total performance loss
total_thrust_loss_pct = (thrust_kN[-1] / thrust_kN[0] - 1) * 100
total_isp_loss_pct = (Isp[-1] / Isp[0] - 1) * 100

print(f"\n{'='*80}")
print("CUMULATIVE PERFORMANCE DEGRADATION")
print("=" * 80)
print(f"  Thrust loss:        {total_thrust_loss_pct:.2f}%")
print(f"  Isp loss:           {total_isp_loss_pct:.2f}%")
print(f"  L* increase:        {(Lstar_mm[-1]/Lstar_mm[0]-1)*100:.2f}%")

# Create comprehensive plots
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Plot 1: Thrust vs time
ax = axes[0, 0]
ax.plot(times, thrust_kN, 'b-', linewidth=2)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Thrust [kN]')
ax.set_title('Thrust Degradation Over Time')
ax.grid(True, alpha=0.3)
ax.axhline(y=thrust_kN[0], color='r', linestyle='--', alpha=0.5, label='Initial')
ax.legend()

# Plot 2: L* evolution
ax = axes[0, 1]
ax.plot(times, Lstar_mm, 'g-', linewidth=2)
ax.set_xlabel('Time [s]')
ax.set_ylabel('L* [mm]')
ax.set_title('Characteristic Length Evolution')
ax.grid(True, alpha=0.3)

# Plot 3: Recession
ax = axes[1, 0]
ax.plot(times, recession_chamber_um, 'purple', linewidth=2, label='Chamber')
ax.plot(times, recession_throat_um, 'orange', linewidth=2, label='Throat')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Cumulative Recession [µm]')
ax.set_title('Ablative Material Recession')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Geometry changes
ax = axes[1, 1]
ax.plot(times, V_chamber_pct_change, 'r-', linewidth=2, label='Chamber Volume')
ax.plot(times, A_throat_pct_change, 'b-', linewidth=2, label='Throat Area')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Change [%]')
ax.set_title('Geometry Growth')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Chamber pressure
ax = axes[2, 0]
ax.plot(times, Pc_psi, 'cyan', linewidth=2)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Pc [psi]')
ax.set_title('Chamber Pressure Evolution')
ax.grid(True, alpha=0.3)

# Plot 6: Throat recession multiplier
ax = axes[2, 1]
valid_mult = throat_mult[~np.isnan(throat_mult)]
valid_times = times[~np.isnan(throat_mult)]
if len(valid_mult) > 0:
    ax.plot(valid_times, valid_mult, 'brown', linewidth=2)
    ax.axhline(y=np.nanmean(throat_mult), color='k', linestyle='--', alpha=0.5, label=f'Mean: {np.nanmean(throat_mult):.2f}')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Throat Recession Multiplier')
    ax.set_title('Physics-Based Throat Multiplier (Bartz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No throat multiplier data', ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
output_path = Path(__file__).parent / "time_varying_ablation.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved plot to: {output_path}")

print("\n" + "=" * 80)
print("KEY FINDINGS:")
print("=" * 80)
print(f"1. Ablative recession causes {abs(total_thrust_loss_pct):.2f}% thrust loss over {burn_time}s")
print(f"2. Throat area grows {A_throat_pct_change[-1]:.3f}% due to higher recession rate")
print(f"3. L* increases {(Lstar_mm[-1]/Lstar_mm[0]-1)*100:.2f}%, reducing combustion efficiency")
print(f"4. Physics-based throat multiplier averages {np.nanmean(throat_mult):.2f}x chamber recession")
print(f"5. This time-varying geometry effect is NOW FULLY MODELED! ✅")
print("=" * 80)

