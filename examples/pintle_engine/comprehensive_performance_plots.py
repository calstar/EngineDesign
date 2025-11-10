"""Comprehensive performance plots - all metrics from pipeline"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pintle_pipeline.io import load_config
from pintle_models.runner import PintleEngineRunner
from pintle_pipeline.feed_loss import delta_p_feed
from pintle_models.geometry import get_effective_areas, get_hydraulic_diameters
from pintle_models.discharge import cd_from_re, calculate_reynolds_number

print("=" * 80)
print("COMPREHENSIVE PERFORMANCE PLOTS")
print("=" * 80)

# Load configuration
config_path = Path(__file__).parent / "config_minimal.yaml"
config = load_config(str(config_path))

# Initialize runner
runner = PintleEngineRunner(config)

# Test at target operating point
P_tank_O = 1305 * 6894.76  # psi to Pa
P_tank_F = 974 * 6894.76   # psi to Pa

print(f"\nTest Conditions:")
print(f"  P_tank_O = {P_tank_O/6894.76:.0f} psi")
print(f"  P_tank_F = {P_tank_F/6894.76:.0f} psi")

# Run pipeline
results = runner.evaluate(P_tank_O, P_tank_F)
diagnostics = results.get('diagnostics', {})
if isinstance(diagnostics, list):
    diagnostics = diagnostics[0] if diagnostics else {}

# Extract all data
Pc = results['Pc'] / 6894.76  # psi
mdot_O = results['mdot_O']
mdot_F = results['mdot_F']
mdot_total = mdot_O + mdot_F
MR = results['MR']
F = results['F'] / 1000  # kN
Isp = results['Isp']
cstar = results['cstar_actual']
eta_cstar = diagnostics.get('eta_cstar', np.nan)
Tc = diagnostics.get('Tc', np.nan)
gamma = diagnostics.get('gamma', np.nan)
R = diagnostics.get('R', np.nan)

# Calculate injector details
delta_p_feed_O = delta_p_feed(mdot_O, config.fluids["oxidizer"].density, 
                               config.feed_system["oxidizer"], P_tank_O) / 6894.76
delta_p_feed_F = delta_p_feed(mdot_F, config.fluids["fuel"].density,
                               config.feed_system["fuel"], P_tank_F) / 6894.76
P_inj_O = P_tank_O/6894.76 - delta_p_feed_O
P_inj_F = P_tank_F/6894.76 - delta_p_feed_F
delta_p_inj_O = P_inj_O - Pc
delta_p_inj_F = P_inj_F - Pc

A_LOX, A_fuel = get_effective_areas(config.injector.geometry)
d_hyd_O, d_hyd_F = get_hydraulic_diameters(config.injector.geometry)
rho_O = config.fluids["oxidizer"].density
rho_F = config.fluids["fuel"].density
u_O = mdot_O / (rho_O * A_LOX)
u_F = mdot_F / (rho_F * A_fuel)
mu_O = config.fluids["oxidizer"].viscosity
mu_F = config.fluids["fuel"].viscosity
Re_O = calculate_reynolds_number(rho_O, u_O, d_hyd_O, mu_O)
Re_F = calculate_reynolds_number(rho_F, u_F, d_hyd_F, mu_F)
Cd_O = cd_from_re(Re_O, config.discharge["oxidizer"])
Cd_F = cd_from_re(Re_F, config.discharge["fuel"])

# Spray diagnostics
J = diagnostics.get('J', np.nan)
TMR = diagnostics.get('TMR', np.nan)
theta = diagnostics.get('theta', np.nan)
We_O = diagnostics.get('We_O', np.nan)
We_F = diagnostics.get('We_F', np.nan)
D32_O = diagnostics.get('D32_O', np.nan)
D32_F = diagnostics.get('D32_F', np.nan)
x_star = diagnostics.get('x_star', np.nan)

print(f"\nResults:")
print(f"  Pc = {Pc:.1f} psi")
print(f"  MR = {MR:.3f}")
print(f"  mdot_total = {mdot_total:.3f} kg/s")
print(f"  F = {F:.3f} kN")
print(f"  Isp = {Isp:.1f} s")

# Create comprehensive figure
fig = plt.figure(figsize=(20, 16))

# 1. Mass Flow Rates
ax1 = plt.subplot(4, 4, 1)
bars = ax1.bar(['LOX', 'Fuel', 'Total'], 
               [mdot_O, mdot_F, mdot_total],
               color=['blue', 'orange', 'green'], alpha=0.7)
ax1.set_ylabel('Mass Flow [kg/s]')
ax1.set_title('Mass Flow Rates')
ax1.grid(True, alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 2. Pressure Breakdown (LOX)
ax2 = plt.subplot(4, 4, 2)
pressures_LOX = [P_tank_O/6894.76, P_inj_O, Pc]
labels_LOX = ['P_tank', 'P_inj', 'Pc']
colors_LOX = ['blue', 'lightblue', 'red']
ax2.barh(labels_LOX, pressures_LOX, color=colors_LOX, alpha=0.7)
ax2.set_xlabel('Pressure [psi]')
ax2.set_title('LOX Pressure Breakdown')
ax2.grid(True, alpha=0.3)
for i, (label, p) in enumerate(zip(labels_LOX, pressures_LOX)):
    ax2.text(p, i, f'{p:.0f}', ha='left', va='center', fontsize=9)

# 3. Pressure Breakdown (Fuel)
ax3 = plt.subplot(4, 4, 3)
pressures_F = [P_tank_F/6894.76, P_inj_F, Pc]
labels_F = ['P_tank', 'P_inj', 'Pc']
colors_F = ['orange', 'lightcoral', 'red']
ax3.barh(labels_F, pressures_F, color=colors_F, alpha=0.7)
ax3.set_xlabel('Pressure [psi]')
ax3.set_title('Fuel Pressure Breakdown')
ax3.grid(True, alpha=0.3)
for i, (label, p) in enumerate(zip(labels_F, pressures_F)):
    ax3.text(p, i, f'{p:.0f}', ha='left', va='center', fontsize=9)

# 4. Mixture Ratio
ax4 = plt.subplot(4, 4, 4)
target_MR = 2.36
ax4.bar(['Actual', 'Target'], [MR, target_MR], 
        color=['blue', 'red'], alpha=0.7)
ax4.set_ylabel('O/F Ratio')
ax4.set_title(f'Mixture Ratio')
ax4.grid(True, alpha=0.3)
ax4.text(0, MR, f'{MR:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax4.text(1, target_MR, f'{target_MR:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 5. Thrust Components
ax5 = plt.subplot(4, 4, 5)
F_momentum = results.get('F_momentum', 0) / 1000
F_pressure = results.get('F_pressure', 0) / 1000
target_F = 5.308
ax5.bar(['Momentum', 'Pressure', 'Total', 'Target'],
        [F_momentum, F_pressure, F, target_F],
        color=['blue', 'green', 'purple', 'red'], alpha=0.7)
ax5.set_ylabel('Thrust [kN]')
ax5.set_title('Thrust Components')
ax5.grid(True, alpha=0.3)
for i, val in enumerate([F_momentum, F_pressure, F, target_F]):
    ax5.text(i, val, f'{val:.2f}', ha='center', va='bottom', fontsize=9)

# 6. Performance Metrics
ax6 = plt.subplot(4, 4, 6)
target_Isp = 299.0
metrics = ['Isp [s]', 'c* [m/s]', 'η_c*']
values = [Isp, cstar/10, eta_cstar*100]  # Scale for visibility
targets = [target_Isp, None, None]
bars = ax6.bar(metrics, values, color=['blue', 'green', 'orange'], alpha=0.7)
ax6.set_ylabel('Value')
ax6.set_title('Performance Metrics')
ax6.grid(True, alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, values)):
    label = f'{val:.1f}' if i == 0 else f'{val:.0f}'
    ax6.text(bar.get_x() + bar.get_width()/2., val,
             label, ha='center', va='bottom', fontsize=9)

# 7. Chamber Properties
ax7 = plt.subplot(4, 4, 7)
if not np.isnan(Tc) and not np.isnan(gamma) and not np.isnan(R):
    props = ['Tc [K]', 'γ', 'R [J/kg·K]']
    vals = [Tc/100, gamma*10, R/10]  # Scale for visibility
    ax7.bar(props, vals, color=['red', 'blue', 'green'], alpha=0.7)
    ax7.set_ylabel('Value (scaled)')
    ax7.set_title('Chamber Properties')
    ax7.grid(True, alpha=0.3)
    for prop, val in zip(props, vals):
        ax7.text(prop, val, f'{val:.1f}', ha='center', va='bottom', fontsize=9)

# 8. Injector Velocities
ax8 = plt.subplot(4, 4, 8)
ax8.bar(['LOX', 'Fuel'], [u_O, u_F], color=['blue', 'orange'], alpha=0.7)
ax8.set_ylabel('Velocity [m/s]')
ax8.set_title('Injector Velocities')
ax8.grid(True, alpha=0.3)
ax8.text(0, u_O, f'{u_O:.1f}', ha='center', va='bottom', fontsize=10)
ax8.text(1, u_F, f'{u_F:.1f}', ha='center', va='bottom', fontsize=10)

# 9. Discharge Coefficients
ax9 = plt.subplot(4, 4, 9)
ax9.bar(['LOX', 'Fuel'], [Cd_O, Cd_F], color=['blue', 'orange'], alpha=0.7)
ax9.set_ylabel('Discharge Coefficient')
ax9.set_title(f'Cd (Re_O={Re_O:.0f}, Re_F={Re_F:.0f})')
ax9.grid(True, alpha=0.3)
ax9.text(0, Cd_O, f'{Cd_O:.3f}', ha='center', va='bottom', fontsize=10)
ax9.text(1, Cd_F, f'{Cd_F:.3f}', ha='center', va='bottom', fontsize=10)

# 10. Reynolds Numbers
ax10 = plt.subplot(4, 4, 10)
ax10.bar(['LOX', 'Fuel'], [Re_O/1000, Re_F/1000], color=['blue', 'orange'], alpha=0.7)
ax10.set_ylabel('Reynolds Number [×1000]')
ax10.set_title('Reynolds Numbers')
ax10.grid(True, alpha=0.3)
ax10.text(0, Re_O/1000, f'{Re_O:.0f}', ha='center', va='bottom', fontsize=9)
ax10.text(1, Re_F/1000, f'{Re_F:.0f}', ha='center', va='bottom', fontsize=9)

# 11. Pressure Drops
ax11 = plt.subplot(4, 4, 11)
ax11.bar(['Feed O', 'Feed F', 'Inj O', 'Inj F'],
         [delta_p_feed_O, delta_p_feed_F, delta_p_inj_O, delta_p_inj_F],
         color=['lightblue', 'lightcoral', 'blue', 'orange'], alpha=0.7)
ax11.set_ylabel('Pressure Drop [psi]')
ax11.set_title('Pressure Drops')
ax11.grid(True, alpha=0.3)
for i, val in enumerate([delta_p_feed_O, delta_p_feed_F, delta_p_inj_O, delta_p_inj_F]):
    ax11.text(i, val, f'{val:.1f}', ha='center', va='bottom', fontsize=9, rotation=90)

# 12. Spray Parameters
ax12 = plt.subplot(4, 4, 12)
if not np.isnan(J) and not np.isnan(theta):
    spray_params = ['J', 'θ [deg]']
    spray_vals = [J, theta*180/np.pi]
    ax12.bar(spray_params, spray_vals, color=['blue', 'green'], alpha=0.7)
    ax12.set_ylabel('Value')
    ax12.set_title('Spray Parameters')
    ax12.grid(True, alpha=0.3)
    for param, val in zip(spray_params, spray_vals):
        ax12.text(param, val, f'{val:.2f}', ha='center', va='bottom', fontsize=9)

# 13. Weber Numbers
ax13 = plt.subplot(4, 4, 13)
if not np.isnan(We_O) and not np.isnan(We_F):
    ax13.bar(['LOX', 'Fuel'], [We_O/1000, We_F/1000], color=['blue', 'orange'], alpha=0.7)
    ax13.set_ylabel('Weber Number [×1000]')
    ax13.set_title('Weber Numbers')
    ax13.grid(True, alpha=0.3)
    ax13.text(0, We_O/1000, f'{We_O:.0f}', ha='center', va='bottom', fontsize=9)
    ax13.text(1, We_F/1000, f'{We_F:.0f}', ha='center', va='bottom', fontsize=9)

# 14. Sauter Mean Diameters
ax14 = plt.subplot(4, 4, 14)
if not np.isnan(D32_O) and not np.isnan(D32_F):
    ax14.bar(['LOX', 'Fuel'], [D32_O*1e6, D32_F*1e6], color=['blue', 'orange'], alpha=0.7)
    ax14.set_ylabel('SMD [μm]')
    ax14.set_title('Sauter Mean Diameters')
    ax14.grid(True, alpha=0.3)
    ax14.text(0, D32_O*1e6, f'{D32_O*1e6:.2f}', ha='center', va='bottom', fontsize=9)
    ax14.text(1, D32_F*1e6, f'{D32_F*1e6:.2f}', ha='center', va='bottom', fontsize=9)

# 15. Injector Areas
ax15 = plt.subplot(4, 4, 15)
ax15.bar(['LOX', 'Fuel'], [A_LOX*1e6, A_fuel*1e6], color=['blue', 'orange'], alpha=0.7)
ax15.set_ylabel('Area [mm²]')
ax15.set_title('Injector Flow Areas')
ax15.grid(True, alpha=0.3)
ax15.text(0, A_LOX*1e6, f'{A_LOX*1e6:.2f}', ha='center', va='bottom', fontsize=9)
ax15.text(1, A_fuel*1e6, f'{A_fuel*1e6:.2f}', ha='center', va='bottom', fontsize=9)

# 16. Comparison to Target
ax16 = plt.subplot(4, 4, 16)
target_mdot = 1.81
comparisons = {
    'mdot [kg/s]': (mdot_total, target_mdot),
    'F [kN]': (F, target_F),
    'MR': (MR, target_MR),
    'Isp [s]': (Isp, target_Isp)
}
x_pos = np.arange(len(comparisons))
actual_vals = [v[0] for v in comparisons.values()]
target_vals = [v[1] for v in comparisons.values()]
x = np.arange(len(comparisons))
width = 0.35
ax16.bar(x - width/2, actual_vals, width, label='Actual', color='blue', alpha=0.7)
ax16.bar(x + width/2, target_vals, width, label='Target', color='red', alpha=0.7)
ax16.set_ylabel('Value')
ax16.set_title('Actual vs Target')
ax16.set_xticks(x)
ax16.set_xticklabels(list(comparisons.keys()), rotation=45, ha='right')
ax16.legend()
ax16.grid(True, alpha=0.3)

plt.tight_layout()
output_path = Path(__file__).parent / 'comprehensive_performance.png'
plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
print(f"\n[OK] Saved comprehensive performance plot to {output_path}")
plt.show()

# Now create 2D pressure sweep plots
print("\n" + "=" * 80)
print("GENERATING 2D PRESSURE SWEEP PLOTS")
print("=" * 80)

# Create 2D grid
P_tank_O_array = np.linspace(200*6894.76, 1200*6894.76, 20)  # 200-1200 psi
P_tank_F_array = np.linspace(200*6894.76, 1200*6894.76, 20)  # 200-1200 psi
P_O_mesh, P_F_mesh = np.meshgrid(P_tank_O_array, P_tank_F_array)
P_O_flat = P_O_mesh.flatten()
P_F_flat = P_F_mesh.flatten()

print(f"Evaluating {len(P_O_flat)} pressure combinations...")
results_2d = runner.evaluate_arrays(P_O_flat, P_F_flat)

# Reshape results
Pc_2d = results_2d['Pc'].reshape(P_O_mesh.shape) / 6894.76
MR_2d = results_2d['MR'].reshape(P_O_mesh.shape)
F_2d = results_2d['F'].reshape(P_O_mesh.shape) / 1000
Isp_2d = results_2d['Isp'].reshape(P_O_mesh.shape)
mdot_total_2d = (results_2d['mdot_O'] + results_2d['mdot_F']).reshape(P_O_mesh.shape)

# Convert meshgrid to psi
P_O_psi = P_O_mesh / 6894.76
P_F_psi = P_F_mesh / 6894.76

# Create 2D plots
fig2, axes = plt.subplots(2, 3, figsize=(18, 12))

# Thrust map
ax = axes[0, 0]
contour = ax.contourf(P_O_psi, P_F_psi, F_2d, levels=20, cmap='viridis')
ax.set_xlabel('P_tank_O [psi]')
ax.set_ylabel('P_tank_F [psi]')
ax.set_title('Thrust [kN]')
plt.colorbar(contour, ax=ax)

# Chamber pressure map
ax = axes[0, 1]
contour = ax.contourf(P_O_psi, P_F_psi, Pc_2d, levels=20, cmap='plasma')
ax.set_xlabel('P_tank_O [psi]')
ax.set_ylabel('P_tank_F [psi]')
ax.set_title('Chamber Pressure [psi]')
plt.colorbar(contour, ax=ax)

# Mixture ratio map
ax = axes[0, 2]
contour = ax.contourf(P_O_psi, P_F_psi, MR_2d, levels=20, cmap='coolwarm')
ax.set_xlabel('P_tank_O [psi]')
ax.set_ylabel('P_tank_F [psi]')
ax.set_title('Mixture Ratio (O/F)')
plt.colorbar(contour, ax=ax)

# Isp map
ax = axes[1, 0]
contour = ax.contourf(P_O_psi, P_F_psi, Isp_2d, levels=20, cmap='inferno')
ax.set_xlabel('P_tank_O [psi]')
ax.set_ylabel('P_tank_F [psi]')
ax.set_title('Specific Impulse [s]')
plt.colorbar(contour, ax=ax)

# Mass flow map
ax = axes[1, 1]
contour = ax.contourf(P_O_psi, P_F_psi, mdot_total_2d, levels=20, cmap='magma')
ax.set_xlabel('P_tank_O [psi]')
ax.set_ylabel('P_tank_F [psi]')
ax.set_title('Total Mass Flow [kg/s]')
plt.colorbar(contour, ax=ax)

# c* map
cstar_2d = results_2d['cstar_actual'].reshape(P_O_mesh.shape)
ax = axes[1, 2]
contour = ax.contourf(P_O_psi, P_F_psi, cstar_2d, levels=20, cmap='cividis')
ax.set_xlabel('P_tank_O [psi]')
ax.set_ylabel('P_tank_F [psi]')
ax.set_title('Characteristic Velocity c* [m/s]')
plt.colorbar(contour, ax=ax)

plt.tight_layout()
output_path = Path(__file__).parent / 'pressure_sweep_2d.png'
plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
print(f"[OK] Saved 2D pressure sweep plots to {output_path}")
plt.show()

print("\n[OK] All plots generated!")
