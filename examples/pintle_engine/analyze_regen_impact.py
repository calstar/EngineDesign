"""Analyze why regen cooling impact is smaller than expected"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pintle_pipeline.io import load_config
from pintle_models.runner import PintleEngineRunner

config_path = Path(__file__).parent / "config_minimal.yaml"
config = load_config(str(config_path))

P_tank_O = 1305 * 6894.76
P_tank_F = 974 * 6894.76

print("=" * 80)
print("ANALYZING REGEN COOLING IMPACT")
print("=" * 80)

# Test WITH regen
print("\n1. WITH REGEN COOLING:")
print("-" * 80)
config.regen_cooling.enabled = True
runner_with = PintleEngineRunner(config)
results_with = runner_with.evaluate(P_tank_O, P_tank_F)

print(f"  mdot_O = {results_with['mdot_O']:.4f} kg/s")
print(f"  mdot_F = {results_with['mdot_F']:.4f} kg/s")
print(f"  mdot_total = {results_with['mdot_total']:.4f} kg/s")
print(f"  MR = {results_with['MR']:.3f}")
print(f"  Pc = {results_with['Pc']/6894.76:.1f} psi")
print(f"  F = {results_with['F']/1000:.3f} kN")
print(f"  Isp = {results_with['Isp']:.1f} s")
print(f"  cstar_actual = {results_with['cstar_actual']:.1f} m/s")
print(f"  v_exit = {results_with['v_exit']:.1f} m/s")

# Test WITHOUT regen
print("\n2. WITHOUT REGEN COOLING:")
print("-" * 80)
config.regen_cooling.enabled = False
runner_without = PintleEngineRunner(config)
results_without = runner_without.evaluate(P_tank_O, P_tank_F)

print(f"  mdot_O = {results_without['mdot_O']:.4f} kg/s")
print(f"  mdot_F = {results_without['mdot_F']:.4f} kg/s")
print(f"  mdot_total = {results_without['mdot_total']:.4f} kg/s")
print(f"  MR = {results_without['MR']:.3f}")
print(f"  Pc = {results_without['Pc']/6894.76:.1f} psi")
print(f"  F = {results_without['F']/1000:.3f} kN")
print(f"  Isp = {results_without['Isp']:.1f} s")
print(f"  cstar_actual = {results_without['cstar_actual']:.1f} m/s")
print(f"  v_exit = {results_without['v_exit']:.1f} m/s")

# Calculate changes
print("\n3. CHANGES:")
print("-" * 80)
print(f"  mdot_O: {results_with['mdot_O']:.4f} → {results_without['mdot_O']:.4f} kg/s ({(results_with['mdot_O']/results_without['mdot_O'] - 1)*100:+.1f}%)")
print(f"  mdot_F: {results_with['mdot_F']:.4f} → {results_without['mdot_F']:.4f} kg/s ({(results_with['mdot_F']/results_without['mdot_F'] - 1)*100:+.1f}%)")
print(f"  mdot_total: {results_with['mdot_total']:.4f} → {results_without['mdot_total']:.4f} kg/s ({(results_with['mdot_total']/results_without['mdot_total'] - 1)*100:+.1f}%)")
print(f"  MR: {results_with['MR']:.3f} → {results_without['MR']:.3f} ({(results_with['MR']/results_without['MR'] - 1)*100:+.1f}%)")
print(f"  Pc: {results_with['Pc']/6894.76:.1f} → {results_without['Pc']/6894.76:.1f} psi ({(results_with['Pc']/results_without['Pc'] - 1)*100:+.1f}%)")
print(f"  F: {results_with['F']/1000:.3f} → {results_without['F']/1000:.3f} kN ({(results_with['F']/results_without['F'] - 1)*100:+.1f}%)")
print(f"  Isp: {results_with['Isp']:.1f} → {results_without['Isp']:.1f} s ({(results_with['Isp']/results_without['Isp'] - 1)*100:+.1f}%)")
print(f"  cstar: {results_with['cstar_actual']:.1f} → {results_without['cstar_actual']:.1f} m/s ({(results_with['cstar_actual']/results_without['cstar_actual'] - 1)*100:+.1f}%)")
print(f"  v_exit: {results_with['v_exit']:.1f} → {results_without['v_exit']:.1f} m/s ({(results_with['v_exit']/results_without['v_exit'] - 1)*100:+.1f}%)")

# Analyze why impact is small
print("\n4. ANALYSIS:")
print("-" * 80)
print(f"  Fuel flow reduction: {(1 - results_with['mdot_F']/results_without['mdot_F'])*100:.1f}%")
print(f"  Total mass flow reduction: {(1 - results_with['mdot_total']/results_without['mdot_total'])*100:.1f}%")
print(f"  Chamber pressure reduction: {(1 - results_with['Pc']/results_without['Pc'])*100:.1f}%")
print(f"  Thrust reduction: {(1 - results_with['F']/results_without['F'])*100:.1f}%")

# The issue: if mdot_total drops by X%, and Pc drops by Y%, thrust should drop by roughly X% + Y%
# But we're seeing less impact. Why?
print("\n  Expected thrust reduction (if linear):")
expected_reduction = (1 - results_with['mdot_total']/results_without['mdot_total']) + (1 - results_with['Pc']/results_without['Pc'])
print(f"    ~{(1 - results_with['mdot_total']/results_without['mdot_total'])*100:.1f}% (mdot) + {(1 - results_with['Pc']/results_without['Pc'])*100:.1f}% (Pc) = {expected_reduction*100:.1f}%")
print(f"  Actual thrust reduction: {(1 - results_with['F']/results_without['F'])*100:.1f}%")

# Check if MR change is affecting cstar
print("\n  Mixture ratio change:")
print(f"    MR: {results_without['MR']:.3f} → {results_with['MR']:.3f}")
print(f"    This changes cstar: {results_without['cstar_actual']:.1f} → {results_with['cstar_actual']:.1f} m/s")
print(f"    cstar change: {(results_with['cstar_actual']/results_without['cstar_actual'] - 1)*100:+.1f}%")

# The problem: when fuel flow decreases, MR increases (more oxidizer per fuel)
# This might actually improve cstar slightly (closer to optimal MR), which partially offsets the loss
print("\n  ISSUE: When fuel flow decreases, MR increases.")
print("    This changes combustion properties (cstar, gamma, etc.)")
print("    The solver finds a new equilibrium that partially compensates for the loss.")
print("    This is why the impact seems smaller than expected.")

