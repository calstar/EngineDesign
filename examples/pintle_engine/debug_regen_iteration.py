"""Debug regen cooling iteration to see how pressure drop evolves"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pintle_pipeline.io import load_config
from pintle_pipeline.feed_loss import delta_p_feed
from pintle_pipeline.regen_cooling import delta_p_regen_channels

# Load config
config_path = Path(__file__).parent / "config_minimal.yaml"
config = load_config(str(config_path))

# Test conditions
P_tank_F = 974 * 6894.76  # Pa
rho_F = config.fluids["fuel"].density
mu_F = config.fluids["fuel"].viscosity

print("=" * 80)
print("DEBUGGING REGEN COOLING ITERATION")
print("=" * 80)

# Simulate iteration with different fuel mass flow rates
mdot_F_values = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]  # kg/s

print("\nIteration: mdot_F → delta_p_feed_base → delta_p_regen → delta_p_total → P_injector")
print("-" * 80)

for mdot_F in mdot_F_values:
    delta_p_feed_base = delta_p_feed(mdot_F, rho_F, config.feed_system["fuel"], P_tank_F)
    delta_p_regen = delta_p_regen_channels(
        mdot_F, rho_F, mu_F, config.regen_cooling, P_tank_F
    )
    delta_p_total = delta_p_feed_base + delta_p_regen
    P_injector = P_tank_F - delta_p_total
    
    print(f"mdot_F={mdot_F:.3f} kg/s: "
          f"feed={delta_p_feed_base/6894.76:.1f} psi, "
          f"regen={delta_p_regen/6894.76:.1f} psi, "
          f"total={delta_p_total/6894.76:.1f} psi, "
          f"P_inj={P_injector/6894.76:.1f} psi")

# Now check what happens at the actual solved values
print("\n" + "=" * 80)
print("AT ACTUAL SOLVED VALUES:")
print("=" * 80)

from pintle_models.runner import PintleEngineRunner

P_tank_O = 1305 * 6894.76
P_tank_F = 974 * 6894.76

runner = PintleEngineRunner(config)
results = runner.evaluate(P_tank_O, P_tank_F)

mdot_F_actual = results['mdot_F']
Pc_actual = results['Pc']

delta_p_feed_base_actual = delta_p_feed(mdot_F_actual, rho_F, config.feed_system["fuel"], P_tank_F)
delta_p_regen_actual = delta_p_regen_channels(
    mdot_F_actual, rho_F, mu_F, config.regen_cooling, P_tank_F
)
delta_p_total_actual = delta_p_feed_base_actual + delta_p_regen_actual
P_injector_actual = P_tank_F - delta_p_total_actual

print(f"\nSolved values:")
print(f"  mdot_F = {mdot_F_actual:.4f} kg/s")
print(f"  Pc = {Pc_actual/6894.76:.1f} psi")
print(f"  delta_p_feed_base = {delta_p_feed_base_actual/6894.76:.2f} psi")
print(f"  delta_p_regen = {delta_p_regen_actual/6894.76:.2f} psi")
print(f"  delta_p_total = {delta_p_total_actual/6894.76:.2f} psi")
print(f"  P_injector = {P_injector_actual/6894.76:.1f} psi")
print(f"  delta_p_injector = {(P_injector_actual - Pc_actual)/6894.76:.1f} psi")

# Compare to what it would be WITHOUT regen
print("\n" + "=" * 80)
print("COMPARISON: What if we used the SAME mdot_F but NO regen?")
print("=" * 80)

# Use the same mdot_F but calculate without regen
delta_p_feed_no_regen = delta_p_feed(mdot_F_actual, rho_F, config.feed_system["fuel"], P_tank_F)
P_injector_no_regen = P_tank_F - delta_p_feed_no_regen

print(f"\nWith same mdot_F = {mdot_F_actual:.4f} kg/s:")
print(f"  WITHOUT regen: P_injector = {P_injector_no_regen/6894.76:.1f} psi")
print(f"  WITH regen:    P_injector = {P_injector_actual/6894.76:.1f} psi")
print(f"  Difference:    {P_injector_no_regen - P_injector_actual:.1f} Pa = {(P_injector_no_regen - P_injector_actual)/6894.76:.2f} psi")

# The key question: what mdot_F would we get WITHOUT regen at the same Pc?
print("\n" + "=" * 80)
print("KEY QUESTION: What mdot_F would we get WITHOUT regen at Pc = {:.1f} psi?".format(Pc_actual/6894.76))
print("=" * 80)

# Disable regen and solve again
config.regen_cooling.enabled = False
runner_no_regen = PintleEngineRunner(config)
results_no_regen = runner_no_regen.evaluate(P_tank_O, P_tank_F)

mdot_F_no_regen = results_no_regen['mdot_F']
Pc_no_regen = results_no_regen['Pc']

print(f"\nWITHOUT regen:")
print(f"  mdot_F = {mdot_F_no_regen:.4f} kg/s")
print(f"  Pc = {Pc_no_regen/6894.76:.1f} psi")
print(f"  F = {results_no_regen['F']/1000:.3f} kN")

print(f"\nWITH regen:")
print(f"  mdot_F = {mdot_F_actual:.4f} kg/s")
print(f"  Pc = {Pc_actual/6894.76:.1f} psi")
print(f"  F = {results['F']/1000:.3f} kN")

print(f"\nImpact:")
print(f"  mdot_F reduction: {(mdot_F_no_regen - mdot_F_actual)/mdot_F_no_regen*100:.1f}%")
print(f"  Pc reduction: {(Pc_no_regen - Pc_actual)/Pc_no_regen*100:.1f}%")
print(f"  Thrust reduction: {(results_no_regen['F'] - results['F'])/results_no_regen['F']*100:.1f}%")

