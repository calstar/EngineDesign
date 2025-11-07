"""Check the impact of regen cooling on performance"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pintle_pipeline.io import load_config
from pintle_models.runner import PintleEngineRunner

# Load config
config_path = Path(__file__).parent / "config_minimal.yaml"
config = load_config(str(config_path))

# Test conditions
P_tank_O = 1305 * 6894.76  # psi to Pa
P_tank_F = 974 * 6894.76   # psi to Pa

print("=" * 80)
print("REGEN COOLING IMPACT ANALYSIS")
print("=" * 80)

# Test WITH regen cooling
print("\n1. WITH REGEN COOLING:")
print("-" * 80)
config.regen_cooling.enabled = True
runner = PintleEngineRunner(config)
results_with = runner.evaluate(P_tank_O, P_tank_F)
print(f"  Pc = {results_with['Pc']/6894.76:.1f} psi")
print(f"  mdot_O = {results_with['mdot_O']:.4f} kg/s")
print(f"  mdot_F = {results_with['mdot_F']:.4f} kg/s")
print(f"  mdot_total = {results_with['mdot_total']:.4f} kg/s")
print(f"  MR = {results_with['MR']:.3f}")
print(f"  F = {results_with['F']/1000:.3f} kN")
print(f"  Isp = {results_with['Isp']:.1f} s")

# Calculate pressure drops
from pintle_pipeline.feed_loss import delta_p_feed
from pintle_pipeline.regen_cooling import delta_p_regen_channels

mdot_F = results_with['mdot_F']
rho_F = config.fluids["fuel"].density
mu_F = config.fluids["fuel"].viscosity

delta_p_feed_F_base = delta_p_feed(mdot_F, rho_F, config.feed_system["fuel"], P_tank_F)
delta_p_regen = delta_p_regen_channels(mdot_F, rho_F, mu_F, config.regen_cooling, P_tank_F)
delta_p_feed_F_total = delta_p_feed_F_base + delta_p_regen
P_inj_F = P_tank_F - delta_p_feed_F_total

print(f"\n  Fuel Pressure Breakdown:")
print(f"    P_tank = {P_tank_F/6894.76:.1f} psi")
print(f"    Δp_feed_base = {delta_p_feed_F_base/6894.76:.2f} psi")
print(f"    Δp_regen = {delta_p_regen/6894.76:.2f} psi")
print(f"    Δp_feed_total = {delta_p_feed_F_total/6894.76:.2f} psi")
print(f"    P_injector = {P_inj_F/6894.76:.1f} psi")
print(f"    Pc = {results_with['Pc']/6894.76:.1f} psi")
print(f"    Δp_injector = {(P_inj_F - results_with['Pc'])/6894.76:.2f} psi")

# Test WITHOUT regen cooling
print("\n2. WITHOUT REGEN COOLING:")
print("-" * 80)
config.regen_cooling.enabled = False
runner_no_regen = PintleEngineRunner(config)
results_without = runner_no_regen.evaluate(P_tank_O, P_tank_F)
print(f"  Pc = {results_without['Pc']/6894.76:.1f} psi")
print(f"  mdot_O = {results_without['mdot_O']:.4f} kg/s")
print(f"  mdot_F = {results_without['mdot_F']:.4f} kg/s")
print(f"  mdot_total = {results_without['mdot_total']:.4f} kg/s")
print(f"  MR = {results_without['MR']:.3f}")
print(f"  F = {results_without['F']/1000:.3f} kN")
print(f"  Isp = {results_without['Isp']:.1f} s")

# Calculate pressure drops (no regen)
mdot_F_no_regen = results_without['mdot_F']
delta_p_feed_F_no_regen = delta_p_feed(mdot_F_no_regen, rho_F, config.feed_system["fuel"], P_tank_F)
P_inj_F_no_regen = P_tank_F - delta_p_feed_F_no_regen

print(f"\n  Fuel Pressure Breakdown:")
print(f"    P_tank = {P_tank_F/6894.76:.1f} psi")
print(f"    Δp_feed = {delta_p_feed_F_no_regen/6894.76:.2f} psi")
print(f"    P_injector = {P_inj_F_no_regen/6894.76:.1f} psi")
print(f"    Pc = {results_without['Pc']/6894.76:.1f} psi")
print(f"    Δp_injector = {(P_inj_F_no_regen - results_without['Pc'])/6894.76:.2f} psi")

# Compare
print("\n3. PERFORMANCE IMPACT:")
print("-" * 80)
print(f"  Thrust: {results_with['F']/1000:.3f} kN (with) vs {results_without['F']/1000:.3f} kN (without)")
print(f"    Change: {(results_with['F'] - results_without['F'])/1000:.3f} kN ({(results_with['F']/results_without['F'] - 1)*100:.1f}%)")
print(f"  Isp: {results_with['Isp']:.1f} s (with) vs {results_without['Isp']:.1f} s (without)")
print(f"    Change: {results_with['Isp'] - results_without['Isp']:.1f} s ({(results_with['Isp']/results_without['Isp'] - 1)*100:.1f}%)")
print(f"  Mass Flow: {results_with['mdot_total']:.4f} kg/s (with) vs {results_without['mdot_total']:.4f} kg/s (without)")
print(f"    Change: {results_with['mdot_total'] - results_without['mdot_total']:.4f} kg/s ({(results_with['mdot_total']/results_without['mdot_total'] - 1)*100:.1f}%)")
print(f"  Fuel Flow: {results_with['mdot_F']:.4f} kg/s (with) vs {results_without['mdot_F']:.4f} kg/s (without)")
print(f"    Change: {results_with['mdot_F'] - results_without['mdot_F']:.4f} kg/s ({(results_with['mdot_F']/results_without['mdot_F'] - 1)*100:.1f}%)")
print(f"  Chamber Pressure: {results_with['Pc']/6894.76:.1f} psi (with) vs {results_without['Pc']/6894.76:.1f} psi (without)")
print(f"    Change: {(results_with['Pc'] - results_without['Pc'])/6894.76:.1f} psi")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)
if results_with['F'] > results_without['F']:
    print("  WARNING: Thrust is HIGHER with regen cooling! This is unexpected.")
    print("  The regen cooling should REDUCE fuel flow and performance.")
else:
    print(f"  Regen cooling reduces thrust by {(1 - results_with['F']/results_without['F'])*100:.1f}%")
    print(f"  This is {'significant' if (1 - results_with['F']/results_without['F']) > 0.1 else 'small'} impact.")

if P_inj_F < results_with['Pc']:
    print(f"\n  CRITICAL: P_injector ({P_inj_F/6894.76:.1f} psi) < Pc ({results_with['Pc']/6894.76:.1f} psi)")
    print("  This means fuel cannot flow! The solver may be finding an unphysical solution.")
    print("  The regen cooling pressure drop is TOO HIGH for these conditions.")

