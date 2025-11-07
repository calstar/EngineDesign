"""Compare CEA ideal (infinite area) vs actual (chamber-driven) performance"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pintle_pipeline.io import load_config
from pintle_models.runner import PintleEngineRunner
from pintle_pipeline.combustion_eff import (
    eta_cstar, calculate_Lstar, 
    calculate_actual_chamber_temp,
    calculate_frozen_flow_correction
)

# Load configuration
config_path = Path(__file__).parent / "config_minimal.yaml"
config = load_config(str(config_path))

# Initialize runner
runner = PintleEngineRunner(config)

print("=" * 70)
print("CEA IDEAL vs ACTUAL CHAMBER-DRIVEN PERFORMANCE")
print("=" * 70)

# Test single operating point
P_tank_O = 5.0e6
P_tank_F = 4.5e6

print(f"\nOperating Point:")
print(f"  P_tank_O = {P_tank_O/6894.76:.0f} psi")
print(f"  P_tank_F = {P_tank_F/6894.76:.0f} psi")

# Solve for actual chamber pressure
Pc, diagnostics = runner.solver.solve(P_tank_O, P_tank_F)
MR = diagnostics['MR']

print(f"\nSolved Chamber Pressure:")
print(f"  Pc = {Pc/6894.76:.0f} psi")
print(f"  MR = {MR:.2f}")

# Get CEA ideal properties
cea_props = runner.solver.cea_cache.eval(MR, Pc)

print(f"\n" + "=" * 70)
print("CEA IDEAL (Infinite Area Equilibrium)")
print("=" * 70)
print(f"  Tc_ideal     = {cea_props['Tc']:.1f} K ({(cea_props['Tc']-273.15)*9/5+32:.0f} °F)")
print(f"  c*_ideal     = {cea_props['cstar_ideal']:.1f} m/s ({cea_props['cstar_ideal']*3.28084:.0f} ft/s)")
print(f"  gamma_ideal  = {cea_props['gamma']:.4f}")
print(f"  R            = {cea_props['R']:.2f} J/(kg·K)")
print(f"  Cf_ideal     = {cea_props['Cf_ideal']:.4f}")

# Calculate chamber geometry parameters
V_chamber = config.chamber.volume
A_throat = config.chamber.A_throat
Lstar = calculate_Lstar(V_chamber, A_throat)

print(f"\n" + "=" * 70)
print("CHAMBER GEOMETRY")
print("=" * 70)
print(f"  V_chamber = {V_chamber*1e6:.2f} cm³")
print(f"  A_throat  = {A_throat*1e6:.2f} mm²")
print(f"  L*        = {Lstar:.3f} m ({Lstar*39.37:.1f} inches)")

# Calculate combustion efficiency
eta = eta_cstar(Lstar, config.combustion.efficiency, diagnostics.get('spray_quality_good', True))

print(f"\n" + "=" * 70)
print("COMBUSTION EFFICIENCY (Finite Chamber Correction)")
print("=" * 70)
print(f"  eta_c* = {eta:.4f} ({eta*100:.2f}%)")
print(f"  Loss due to finite L*: {(1-eta)*100:.2f}%")

# Calculate actual chamber properties
cstar_actual = eta * cea_props['cstar_ideal']
Tc_actual = calculate_actual_chamber_temp(cea_props['Tc'], eta, cea_props['gamma'])
gamma_frozen_factor = calculate_frozen_flow_correction(Lstar, cea_props['gamma'])
gamma_actual = cea_props['gamma'] * gamma_frozen_factor

print(f"\n" + "=" * 70)
print("ACTUAL CHAMBER-DRIVEN PERFORMANCE")
print("=" * 70)
print(f"  Tc_actual    = {Tc_actual:.1f} K ({(Tc_actual-273.15)*9/5+32:.0f} degF)")
print(f"    -> Delta_Tc      = {cea_props['Tc'] - Tc_actual:.1f} K ({((cea_props['Tc']-Tc_actual)*9/5):.0f} degF)")
print(f"  c*_actual    = {cstar_actual:.1f} m/s ({cstar_actual*3.28084:.0f} ft/s)")
print(f"    -> Delta_cstar      = {cea_props['cstar_ideal'] - cstar_actual:.1f} m/s ({(cea_props['cstar_ideal']-cstar_actual)*3.28084:.0f} ft/s)")
print(f"  gamma_actual = {gamma_actual:.4f} (with frozen flow correction)")
print(f"    -> Delta_gamma   = {gamma_actual - cea_props['gamma']:.4f}")
print(f"  Cf_actual    = {cea_props['Cf_ideal'] * eta:.4f} (approx eta x Cf_ideal)")

# Calculate performance impact
mdot_total = diagnostics['mdot_total']
F_ideal = cea_props['Cf_ideal'] * Pc * A_throat
F_actual = F_ideal * eta  # Simplified

Isp_ideal = F_ideal / (mdot_total * 9.80665)
Isp_actual = F_actual / (mdot_total * 9.80665)

print(f"\n" + "=" * 70)
print("PERFORMANCE IMPACT")
print("=" * 70)
print(f"  Thrust (ideal) = {F_ideal/1000:.2f} kN")
print(f"  Thrust (actual) = {F_actual/1000:.2f} kN")
print(f"    -> Loss       = {(F_ideal-F_actual)/1000:.2f} kN ({(1-eta)*100:.1f}%)")
print(f"  Isp (ideal)    = {Isp_ideal:.1f} s")
print(f"  Isp (actual)   = {Isp_actual:.1f} s")
print(f"    -> Loss       = {Isp_ideal-Isp_actual:.1f} s ({(1-eta)*100:.1f}%)")

# Create comparison plots for different L* values
print(f"\n" + "=" * 70)
print("GENERATING L* SENSITIVITY PLOTS")
print("=" * 70)

L_star_range = np.linspace(0.2, 3.0, 50)
eta_values = []
Tc_actual_values = []
gamma_actual_values = []
cstar_actual_values = []

for L in L_star_range:
    eta_L = eta_cstar(L, config.combustion.efficiency, True)
    eta_values.append(eta_L)
    
    Tc_act = calculate_actual_chamber_temp(cea_props['Tc'], eta_L, cea_props['gamma'])
    Tc_actual_values.append(Tc_act)
    
    gamma_act = cea_props['gamma'] * calculate_frozen_flow_correction(L, cea_props['gamma'])
    gamma_actual_values.append(gamma_act)
    
    cstar_act = eta_L * cea_props['cstar_ideal']
    cstar_actual_values.append(cstar_act)

eta_values = np.array(eta_values)
Tc_actual_values = np.array(Tc_actual_values)
gamma_actual_values = np.array(gamma_actual_values)
cstar_actual_values = np.array(cstar_actual_values)

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("CEA Ideal vs Actual Chamber-Driven Performance\n(LOX/RP-1, MR=2.2, Pc=2 MPa)", 
             fontsize=14, fontweight="bold")

# Plot 1: Combustion Efficiency vs L*
ax = axes[0, 0]
ax.plot(L_star_range, eta_values * 100, 'b-', linewidth=2.5)
ax.axvline(Lstar, color='r', linestyle='--', linewidth=2, label=f'Current L* = {Lstar:.2f} m')
ax.axhline(eta * 100, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Characteristic Length L* [m]')
ax.set_ylabel('Combustion Efficiency eta_c* [%]')
ax.set_title('Combustion Efficiency vs Chamber L*')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(50, 100)
ax.text(0.05, 0.05, f'Current: eta = {eta*100:.1f}%',
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Plot 2: c* (Ideal vs Actual)
ax = axes[0, 1]
ax.plot(L_star_range, cstar_actual_values, 'b-', linewidth=2.5, label='Actual (chamber-driven)')
ax.axhline(cea_props['cstar_ideal'], color='g', linestyle='--', linewidth=2, label='CEA Ideal (equilibrium)')
ax.axvline(Lstar, color='r', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel('Characteristic Length L* [m]')
ax.set_ylabel('Characteristic Velocity c* [m/s]')
ax.set_title('c* vs Chamber L*')
ax.grid(True, alpha=0.3)
ax.legend()
ax.text(0.05, 0.05, f'Loss: {cea_props["cstar_ideal"]-cstar_actual:.0f} m/s ({(1-eta)*100:.1f}%)', 
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Plot 3: Chamber Temperature (Ideal vs Actual)
ax = axes[1, 0]
Tc_actual_F = (Tc_actual_values - 273.15) * 9/5 + 32
Tc_ideal_F = (cea_props['Tc'] - 273.15) * 9/5 + 32
ax.plot(L_star_range, Tc_actual_F, 'b-', linewidth=2.5, label='Actual (incomplete combustion)')
ax.axhline(Tc_ideal_F, color='g', linestyle='--', linewidth=2, label='CEA Ideal (equilibrium)')
ax.axvline(Lstar, color='r', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel('Characteristic Length L* [m]')
ax.set_ylabel('Chamber Temperature Tc [°F]')
ax.set_title('Chamber Temperature vs L*')
ax.grid(True, alpha=0.3)
ax.legend()
ax.text(0.05, 0.95, f'Loss: {((cea_props["Tc"]-Tc_actual)*9/5):.0f} °F', 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Plot 4: Gamma (Equilibrium vs Frozen Flow)
ax = axes[1, 1]
ax.plot(L_star_range, gamma_actual_values, 'b-', linewidth=2.5, label='Actual (frozen flow effect)')
ax.axhline(cea_props['gamma'], color='g', linestyle='--', linewidth=2, label='CEA Equilibrium')
ax.axvline(Lstar, color='r', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel('Characteristic Length L* [m]')
ax.set_ylabel('Specific Heat Ratio gamma')
ax.set_title('Gamma: Equilibrium vs Frozen Flow')
ax.grid(True, alpha=0.3)
ax.legend()
ax.text(0.05, 0.05, 'Short L* -> more frozen -> higher gamma',
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
output_path = Path(__file__).parent / "ideal_vs_actual_comparison.png"
plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
print(f"\n[OK] Saved comparison plots to {output_path}")

print("\n" + "=" * 70)
print("[OK] Analysis complete!")
print("=" * 70)
print("\nKey Takeaways:")
print(f"  • CEA assumes infinite chamber (equilibrium)")
print(f"  • Our L* = {Lstar:.2f} m gives eta = {eta*100:.1f}%")
print(f"  • Performance loss: {(1-eta)*100:.1f}% due to finite chamber")
print(f"  • To improve: increase L* (larger chamber or smaller throat)")
print("=" * 70)

