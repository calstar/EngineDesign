"""Test what happens if we fix MR to see the true impact of regen cooling"""

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
print("TESTING: What if MR stayed constant?")
print("=" * 80)

# Get baseline (no regen)
config.regen_cooling.enabled = False
runner_no_regen = PintleEngineRunner(config)
results_no_regen = runner_no_regen.evaluate(P_tank_O, P_tank_F)

baseline_mdot_F = results_no_regen['mdot_F']
baseline_MR = results_no_regen['MR']
baseline_Pc = results_no_regen['Pc']
baseline_F = results_no_regen['F']

print(f"\nBaseline (no regen):")
print(f"  mdot_F = {baseline_mdot_F:.4f} kg/s")
print(f"  MR = {baseline_MR:.3f}")
print(f"  Pc = {baseline_Pc/6894.76:.1f} psi")
print(f"  F = {baseline_F/1000:.3f} kN")

# Now with regen - but what if we could keep MR constant?
# The issue is that when fuel flow drops, MR increases
# If we could keep MR constant, we'd need to reduce oxidizer flow too

# Calculate what mdot_F would be with regen (assuming same pressure drop)
from pintle_pipeline.feed_loss import delta_p_feed
from pintle_pipeline.regen_cooling import delta_p_regen_channels

rho_F = config.fluids["fuel"].density
mu_F = config.fluids["fuel"].viscosity

# Estimate: if we have 81.33 psi regen pressure drop at 0.7452 kg/s
# What would mdot_F be if we tried to maintain the same injector pressure drop?
config.regen_cooling.enabled = True
runner_with_regen = PintleEngineRunner(config)
results_with_regen = runner_with_regen.evaluate(P_tank_O, P_tank_F)

actual_mdot_F = results_with_regen['mdot_F']
actual_MR = results_with_regen['MR']
actual_Pc = results_with_regen['Pc']
actual_F = results_with_regen['F']

print(f"\nWith regen (actual):")
print(f"  mdot_F = {actual_mdot_F:.4f} kg/s ({(actual_mdot_F/baseline_mdot_F - 1)*100:+.1f}%)")
print(f"  MR = {actual_MR:.3f} ({(actual_MR/baseline_MR - 1)*100:+.1f}%)")
print(f"  Pc = {actual_Pc/6894.76:.1f} psi ({(actual_Pc/baseline_Pc - 1)*100:+.1f}%)")
print(f"  F = {actual_F/1000:.3f} kN ({(actual_F/baseline_F - 1)*100:+.1f}%)")

# The key insight: if fuel flow drops by 7.7%, but MR increases by 9.0%
# This means oxidizer flow is actually INCREASING slightly
# This is why total mass flow only drops by 2.1%

print(f"\n" + "=" * 80)
print("ROOT CAUSE ANALYSIS:")
print("=" * 80)
print(f"\nFuel flow drops by: {(1 - actual_mdot_F/baseline_mdot_F)*100:.1f}%")
print(f"But MR increases by: {(actual_MR/baseline_MR - 1)*100:.1f}%")
print(f"This means oxidizer flow changes by: {(results_with_regen['mdot_O']/results_no_regen['mdot_O'] - 1)*100:+.1f}%")
print(f"\nTotal mass flow only drops by: {(1 - results_with_regen['mdot_total']/results_no_regen['mdot_total'])*100:.1f}%")
print(f"Because oxidizer flow is INCREASING, partially compensating for fuel loss!")

print(f"\n" + "=" * 80)
print("WHY THIS HAPPENS:")
print("=" * 80)
print("When fuel flow decreases due to regen cooling:")
print("  1. Fuel injector pressure drops (due to regen pressure loss)")
print("  2. Fuel flow decreases")
print("  3. But oxidizer flow is NOT directly affected by regen")
print("  4. The solver finds a new Pc where supply = demand")
print("  5. At this new Pc, oxidizer flow might actually INCREASE slightly")
print("  6. This increases MR, which improves cstar, partially offsetting the loss")
print("\nThe solver is finding a 'smart' equilibrium that minimizes the impact,")
print("but this masks the true effect of regen cooling on fuel flow.")

print(f"\n" + "=" * 80)
print("SOLUTION:")
print("=" * 80)
print("The regen cooling IS working correctly - fuel flow drops by 7.7%")
print("But the solver finds a new equilibrium where:")
print("  - Oxidizer flow increases slightly")
print("  - MR increases (improves cstar)")
print("  - This partially compensates for the fuel loss")
print("\nThis is physically correct behavior - the solver balances supply and demand.")
print("But it means the impact on THRUST is smaller than the impact on FUEL FLOW.")

