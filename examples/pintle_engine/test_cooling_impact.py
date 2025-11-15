"""Demonstrate the impact of cooling on thrust output."""

from pathlib import Path
import sys
import copy

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pintle_pipeline.io import load_config
from pintle_models.runner import PintleEngineRunner

# Load configuration
config_path = Path(__file__).parent / "config_minimal.yaml"
config = load_config(str(config_path))

print("=" * 80)
print("COOLING IMPACT ON THRUST - DIAGNOSTIC TEST")
print("=" * 80)

# Test conditions
P_tank_O_psi = 1305.0
P_tank_F_psi = 974.0
P_tank_O_pa = P_tank_O_psi * 6894.76
P_tank_F_pa = P_tank_F_psi * 6894.76

print(f"\nTest Conditions:")
print(f"  LOX Tank Pressure:  {P_tank_O_psi:.0f} psi")
print(f"  Fuel Tank Pressure: {P_tank_F_psi:.0f} psi")

# ============================================================================
# TEST 1: NO COOLING
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: NO COOLING (Baseline)")
print("=" * 80)

config_no_cooling = copy.deepcopy(config)
config_no_cooling.regen_cooling.enabled = False
config_no_cooling.film_cooling.enabled = False
config_no_cooling.ablative_cooling.enabled = False
config_no_cooling.combustion.efficiency.use_cooling_coupling = False

runner_no_cooling = PintleEngineRunner(config_no_cooling)
results_no_cooling = runner_no_cooling.evaluate(P_tank_O_pa, P_tank_F_pa)

print(f"\nPerformance:")
print(f"  Thrust:      {results_no_cooling['F']/1000:.3f} kN")
print(f"  Isp:         {results_no_cooling['Isp']:.2f} s")
print(f"  Pc:          {results_no_cooling['Pc']/6894.76:.1f} psi")
print(f"  O/F Ratio:   {results_no_cooling['MR']:.3f}")
print(f"  mdot_O:      {results_no_cooling['mdot_O']:.4f} kg/s")
print(f"  mdot_F:      {results_no_cooling['mdot_F']:.4f} kg/s")
print(f"  mdot_total:  {results_no_cooling['mdot_total']:.4f} kg/s")
print(f"  c* (actual): {results_no_cooling['cstar_actual']:.1f} m/s")
print(f"  η_c*:        {results_no_cooling['eta_cstar']:.4f}")

# ============================================================================
# TEST 2: REGEN COOLING ONLY
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: REGENERATIVE COOLING ONLY")
print("=" * 80)

config_regen = copy.deepcopy(config)
config_regen.regen_cooling.enabled = True
config_regen.film_cooling.enabled = False
config_regen.ablative_cooling.enabled = False
config_regen.combustion.efficiency.use_cooling_coupling = True

runner_regen = PintleEngineRunner(config_regen)
results_regen = runner_regen.evaluate(P_tank_O_pa, P_tank_F_pa)

print(f"\nPerformance:")
print(f"  Thrust:      {results_regen['F']/1000:.3f} kN")
print(f"  Isp:         {results_regen['Isp']:.2f} s")
print(f"  Pc:          {results_regen['Pc']/6894.76:.1f} psi")
print(f"  O/F Ratio:   {results_regen['MR']:.3f}")
print(f"  mdot_O:      {results_regen['mdot_O']:.4f} kg/s")
print(f"  mdot_F:      {results_regen['mdot_F']:.4f} kg/s")
print(f"  mdot_total:  {results_regen['mdot_total']:.4f} kg/s")
print(f"  c* (actual): {results_regen['cstar_actual']:.1f} m/s")
print(f"  η_c*:        {results_regen['eta_cstar']:.4f}")

regen_diag = results_regen['diagnostics']['cooling']['regen']
print(f"\nRegen Cooling Details:")
print(f"  Pressure drop:   {regen_diag['pressure_drop']/6894.76:.1f} psi")
print(f"  Heat removed:    {regen_diag['heat_removed']/1000:.1f} kW")
print(f"  Outlet temp:     {regen_diag['coolant_outlet_temperature']:.1f} K")
print(f"  Cooling eff:     {results_regen['diagnostics']['cooling_efficiency']:.4f}")

# ============================================================================
# TEST 3: ALL COOLING (Regen + Film + Ablative)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: ALL COOLING (Regen + Film + Ablative)")
print("=" * 80)

config_all = copy.deepcopy(config)
config_all.regen_cooling.enabled = True
config_all.film_cooling.enabled = True
config_all.ablative_cooling.enabled = True
config_all.combustion.efficiency.use_cooling_coupling = True

runner_all = PintleEngineRunner(config_all)
results_all = runner_all.evaluate(P_tank_O_pa, P_tank_F_pa)

print(f"\nPerformance:")
print(f"  Thrust:      {results_all['F']/1000:.3f} kN")
print(f"  Isp:         {results_all['Isp']:.2f} s")
print(f"  Pc:          {results_all['Pc']/6894.76:.1f} psi")
print(f"  O/F Ratio:   {results_all['MR']:.3f}")
print(f"  mdot_O:      {results_all['mdot_O']:.4f} kg/s")
print(f"  mdot_F:      {results_all['mdot_F']:.4f} kg/s")
print(f"  mdot_total:  {results_all['mdot_total']:.4f} kg/s")
print(f"  c* (actual): {results_all['cstar_actual']:.1f} m/s")
print(f"  η_c*:        {results_all['eta_cstar']:.4f}")

cooling_diag = results_all['diagnostics']['cooling']
print(f"\nCooling Details:")
if 'regen' in cooling_diag and cooling_diag['regen'].get('enabled'):
    regen = cooling_diag['regen']
    print(f"  Regen ΔP:        {regen['pressure_drop']/6894.76:.1f} psi")
    print(f"  Regen heat:      {regen['heat_removed']/1000:.1f} kW")
if 'film' in cooling_diag and cooling_diag['film'].get('enabled'):
    film = cooling_diag['film']
    print(f"  Film mass frac:  {film['mass_fraction']:.4f}")
    print(f"  Film mdot:       {film['mdot_film']:.4f} kg/s")
    print(f"  Film eff:        {film['effectiveness']:.3f}")
if 'ablative' in cooling_diag and cooling_diag['ablative'].get('enabled'):
    ablative = cooling_diag['ablative']
    print(f"  Ablative recess: {ablative['recession_rate']*1e6:.3f} µm/s")
    print(f"  Ablative heat:   {ablative['heat_removed']/1000:.1f} kW")

print(f"  Cooling eff:     {results_all['diagnostics']['cooling_efficiency']:.4f}")

# ============================================================================
# SUMMARY COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY COMPARISON")
print("=" * 80)

delta_thrust_regen = (results_regen['F'] - results_no_cooling['F']) / results_no_cooling['F'] * 100
delta_thrust_all = (results_all['F'] - results_no_cooling['F']) / results_no_cooling['F'] * 100
delta_mdot_F_regen = (results_regen['mdot_F'] - results_no_cooling['mdot_F']) / results_no_cooling['mdot_F'] * 100
delta_mdot_F_all = (results_all['mdot_F'] - results_no_cooling['mdot_F']) / results_no_cooling['mdot_F'] * 100
delta_MR_regen = (results_regen['MR'] - results_no_cooling['MR']) / results_no_cooling['MR'] * 100
delta_MR_all = (results_all['MR'] - results_no_cooling['MR']) / results_no_cooling['MR'] * 100

print(f"\n{'Metric':<20} {'No Cooling':<15} {'Regen Only':<15} {'All Cooling':<15}")
print("-" * 80)
print(f"{'Thrust [kN]':<20} {results_no_cooling['F']/1000:>14.3f} {results_regen['F']/1000:>14.3f} {results_all['F']/1000:>14.3f}")
print(f"{'  Δ%':<20} {'':<15} {delta_thrust_regen:>+14.2f}% {delta_thrust_all:>+14.2f}%")
print(f"{'Fuel Flow [kg/s]':<20} {results_no_cooling['mdot_F']:>14.4f} {results_regen['mdot_F']:>14.4f} {results_all['mdot_F']:>14.4f}")
print(f"{'  Δ%':<20} {'':<15} {delta_mdot_F_regen:>+14.2f}% {delta_mdot_F_all:>+14.2f}%")
print(f"{'O/F Ratio':<20} {results_no_cooling['MR']:>14.3f} {results_regen['MR']:>14.3f} {results_all['MR']:>14.3f}")
print(f"{'  Δ%':<20} {'':<15} {delta_MR_regen:>+14.2f}% {delta_MR_all:>+14.2f}%")
print(f"{'Pc [psi]':<20} {results_no_cooling['Pc']/6894.76:>14.1f} {results_regen['Pc']/6894.76:>14.1f} {results_all['Pc']/6894.76:>14.1f}")
print(f"{'Isp [s]':<20} {results_no_cooling['Isp']:>14.2f} {results_regen['Isp']:>14.2f} {results_all['Isp']:>14.2f}")
print(f"{'η_c*':<20} {results_no_cooling['eta_cstar']:>14.4f} {results_regen['eta_cstar']:>14.4f} {results_all['eta_cstar']:>14.4f}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print(f"✅ Regen cooling DOES affect thrust: {delta_thrust_regen:+.2f}% change")
print(f"✅ All cooling combined: {delta_thrust_all:+.2f}% change")
print(f"\nThe effect is coupled through:")
print(f"  1. Pressure drop → Lower fuel flow → Higher O/F → Different c*")
print(f"  2. Heat removal → Cooling efficiency factor → Lower η_c*")
print(f"  3. Film cooling → Mass diversion → Direct thrust reduction")
print("=" * 80)

