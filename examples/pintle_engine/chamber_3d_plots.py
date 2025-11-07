"""3D plots of chamber behavior: Pc vs MR and mdot_total"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pintle_pipeline.io import load_config
from pintle_models.runner import PintleEngineRunner

# Load configuration
config_path = Path(__file__).parent / "config_minimal.yaml"
config = load_config(str(config_path))

# Initialize runner
runner = PintleEngineRunner(config)

print("=" * 70)
print("3D CHAMBER BEHAVIOR PLOTS")
print("=" * 70)
print("\nGenerating 3D plots: Chamber Pressure vs O/F Ratio and Total Mass Flow")
print("-" * 70)

# Create a grid of tank pressures
P_tank_O_grid = np.linspace(3.0e6, 8.0e6, 15)  # 3-8 MPa
P_tank_F_grid = np.linspace(3.0e6, 7.5e6, 15)  # 3-7.5 MPa

# Create meshgrid
P_O_mesh, P_F_mesh = np.meshgrid(P_tank_O_grid, P_tank_F_grid)

# Flatten for evaluation
P_O_flat = P_O_mesh.flatten()
P_F_flat = P_F_mesh.flatten()

print(f"Evaluating {len(P_O_flat)} pressure combinations...")
results = runner.evaluate_arrays(P_O_flat, P_F_flat)

# Extract results
Pc_vals = results['Pc'] / 6894.76  # Convert to psi
MR_vals = results['MR']
mdot_total_vals = results['mdot_O'] + results['mdot_F']
F_vals = results['F'] / 1000  # Convert to kN
Isp_vals = results['Isp']

# Filter out NaN values (invalid pressure combinations)
valid_mask = ~np.isnan(Pc_vals)
Pc_valid = Pc_vals[valid_mask]
MR_valid = MR_vals[valid_mask]
mdot_valid = mdot_total_vals[valid_mask]
F_valid = F_vals[valid_mask]
Isp_valid = Isp_vals[valid_mask]

print(f"Valid points: {np.sum(valid_mask)}/{len(Pc_vals)}")

# Create 3D plots
fig = plt.figure(figsize=(16, 12))

# Plot 1: Chamber Pressure vs MR and mdot_total
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
scatter1 = ax1.scatter(
    MR_valid,
    mdot_valid,
    Pc_valid,
    c=Pc_valid,
    cmap='plasma',
    s=30,
    alpha=0.6
)
ax1.set_xlabel('Mixture Ratio (O/F)')
ax1.set_ylabel('Total Mass Flow [kg/s]')
ax1.set_zlabel('Chamber Pressure [psi]')
ax1.set_title('Chamber Pressure vs O/F and Mass Flow')
plt.colorbar(scatter1, ax=ax1, label='Pc [psi]')

# Plot 2: Thrust vs MR and mdot_total
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
scatter2 = ax2.scatter(
    MR_valid,
    mdot_valid,
    F_valid,
    c=F_valid,
    cmap='viridis',
    s=30,
    alpha=0.6
)
ax2.set_xlabel('Mixture Ratio (O/F)')
ax2.set_ylabel('Total Mass Flow [kg/s]')
ax2.set_zlabel('Thrust [kN]')
ax2.set_title('Thrust vs O/F and Mass Flow')
plt.colorbar(scatter2, ax=ax2, label='Thrust [kN]')

# Plot 3: Isp vs MR and mdot_total
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
scatter3 = ax3.scatter(
    MR_valid,
    mdot_valid,
    Isp_valid,
    c=Isp_valid,
    cmap='coolwarm',
    s=30,
    alpha=0.6
)
ax3.set_xlabel('Mixture Ratio (O/F)')
ax3.set_ylabel('Total Mass Flow [kg/s]')
ax3.set_zlabel('Specific Impulse [s]')
ax3.set_title('Isp vs O/F and Mass Flow')
plt.colorbar(scatter3, ax=ax3, label='Isp [s]')

# Plot 4: Thrust vs Pc and mdot_total (like reference file)
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
scatter4 = ax4.scatter(
    Pc_valid,
    mdot_valid,
    F_valid,
    c=MR_valid,
    cmap='viridis',
    s=30,
    alpha=0.6
)
ax4.set_xlabel('Chamber Pressure [psi]')
ax4.set_ylabel('Total Mass Flow [kg/s]')
ax4.set_zlabel('Thrust [kN]')
ax4.set_title('Thrust vs Pc and Mass Flow (Color: O/F Ratio)')
plt.colorbar(scatter4, ax=ax4, label='O/F Ratio')

plt.tight_layout()
output_path = Path(__file__).parent / "chamber_3d_plots.png"
plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
print(f"\n[OK] Saved 3D plots to {output_path}")

# Also create 2D projections
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle("2D Projections of Chamber Behavior", fontsize=16, fontweight="bold")

# Pc vs MR (colored by mdot)
ax = axes[0, 0]
scatter = ax.scatter(MR_valid, Pc_valid, c=mdot_valid, cmap='plasma', s=30, alpha=0.6)
ax.set_xlabel('Mixture Ratio (O/F)')
ax.set_ylabel('Chamber Pressure [psi]')
ax.set_title('Pc vs O/F Ratio (Color: Mass Flow)')
plt.colorbar(scatter, ax=ax, label='Mass Flow [kg/s]')
ax.grid(True, alpha=0.3)

# Pc vs mdot (colored by MR)
ax = axes[0, 1]
scatter = ax.scatter(mdot_valid, Pc_valid, c=MR_valid, cmap='viridis', s=30, alpha=0.6)
ax.set_xlabel('Total Mass Flow [kg/s]')
ax.set_ylabel('Chamber Pressure [psi]')
ax.set_title('Pc vs Mass Flow (Color: O/F Ratio)')
plt.colorbar(scatter, ax=ax, label='O/F Ratio')
ax.grid(True, alpha=0.3)

# Thrust vs MR (colored by mdot)
ax = axes[1, 0]
scatter = ax.scatter(MR_valid, F_valid, c=mdot_valid, cmap='plasma', s=30, alpha=0.6)
ax.set_xlabel('Mixture Ratio (O/F)')
ax.set_ylabel('Thrust [kN]')
ax.set_title('Thrust vs O/F Ratio (Color: Mass Flow)')
plt.colorbar(scatter, ax=ax, label='Mass Flow [kg/s]')
ax.grid(True, alpha=0.3)

# Thrust vs mdot (colored by MR)
ax = axes[1, 1]
scatter = ax.scatter(mdot_valid, F_valid, c=MR_valid, cmap='viridis', s=30, alpha=0.6)
ax.set_xlabel('Total Mass Flow [kg/s]')
ax.set_ylabel('Thrust [kN]')
ax.set_title('Thrust vs Mass Flow (Color: O/F Ratio)')
plt.colorbar(scatter, ax=ax, label='O/F Ratio')
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path_2d = Path(__file__).parent / "chamber_2d_projections.png"
plt.savefig(str(output_path_2d), dpi=300, bbox_inches='tight')
print(f"[OK] Saved 2D projections to {output_path_2d}")

print("\n" + "=" * 70)
print("[OK] 3D plots complete!")
print("=" * 70)

