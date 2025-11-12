"""Plot complete simulation results from t=0 to impact"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
df = pd.read_csv('out/realistic_test/telemetry.csv')
events_df = pd.read_csv('out/realistic_test/events.csv')

print("Creating plots...")

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Altitude vs Time (all bodies and canopies)
ax1 = plt.subplot(3, 3, 1)
for col in df.columns:
    if 'body:' in col and ':r_z' in col:
        body_name = col.split(':')[1].split(':')[0]
        ax1.plot(df['t'], df[col], label=f'Body: {body_name}', linewidth=2)
for col in df.columns:
    if 'canopy:' in col and ':p_z' in col:
        canopy_name = col.split(':')[1].split(':')[0]
        ax1.plot(df['t'], df[col], label=f'Canopy: {canopy_name}', linewidth=1.5, linestyle='--')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Altitude (m)')
ax1.set_title('Altitude vs Time')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
# Add event markers
for _, event in events_df.iterrows():
    t_event = event['t_event']
    if t_event <= df['t'].max():
        ax1.axvline(t_event, color='red', linestyle=':', alpha=0.5)
        ax1.text(t_event, ax1.get_ylim()[1]*0.95, event['event_type'], 
                rotation=90, fontsize=7, ha='right')

# 2. Velocity vs Time
ax2 = plt.subplot(3, 3, 2)
for col in df.columns:
    if 'body:' in col and ':v_z' in col:
        body_name = col.split(':')[1].split(':')[0]
        ax2.plot(df['t'], -df[col], label=f'Body: {body_name}', linewidth=2)  # Negative for downward
for col in df.columns:
    if 'canopy:' in col and ':v_z' in col:
        canopy_name = col.split(':')[1].split(':')[0]
        ax2.plot(df['t'], -df[col], label=f'Canopy: {canopy_name}', linewidth=1.5, linestyle='--')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Descent Velocity (m/s)')
ax2.set_title('Velocity vs Time')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# 3. System Force vs Time
ax3 = plt.subplot(3, 3, 3)
Fsys_mag = np.sqrt(df['Fsys_x']**2 + df['Fsys_y']**2 + df['Fsys_z']**2)
ax3.plot(df['t'], Fsys_mag/1000, 'k-', linewidth=2, label='Total System Force')
ax3.plot(df['t'], df['Fsys_z'].abs()/1000, 'r--', linewidth=1.5, label='Z-component')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Force (kN)')
ax3.set_title('System Force vs Time')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Tension in Lines
ax4 = plt.subplot(3, 3, 4)
for col in df.columns:
    if 'edge:' in col and ':T' in col:
        edge_name = col.split(':')[1].split(':')[0]
        ax4.plot(df['t'], df[col]/1000, label=edge_name, linewidth=2)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Tension (kN)')
ax4.set_title('Line Tensions vs Time')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# 5. Canopy Inflation (Area)
ax5 = plt.subplot(3, 3, 5)
for col in df.columns:
    if 'canopy:' in col and ':A' in col:
        canopy_name = col.split(':')[1].split(':')[0]
        ax5.plot(df['t'], df[col], label=canopy_name, linewidth=2)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Area (m²)')
ax5.set_title('Canopy Area vs Time')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Canopy Drag Coefficient
ax6 = plt.subplot(3, 3, 6)
for col in df.columns:
    if 'canopy:' in col and ':CD' in col:
        canopy_name = col.split(':')[1].split(':')[0]
        ax6.plot(df['t'], df[col], label=canopy_name, linewidth=2)
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('CD')
ax6.set_title('Drag Coefficient vs Time')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Per-Body Forces
ax7 = plt.subplot(3, 3, 7)
for col in df.columns:
    if 'body:' in col and ':Fz' in col:
        body_name = col.split(':')[1].split(':')[0]
        F_mag = df[col].abs()
        ax7.plot(df['t'], F_mag/1000, label=body_name, linewidth=2)
ax7.set_xlabel('Time (s)')
ax7.set_ylabel('Force (kN)')
ax7.set_title('Per-Body Forces (Z)')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# 8. Anchor Loads
ax8 = plt.subplot(3, 3, 8)
anchor_loads = {}
for col in df.columns:
    if 'anchor:' in col and 'Fmag_I' in col:
        anchor_name = col.split(':')[1] + ':' + col.split(':')[2].split(':')[0]
        if anchor_name not in anchor_loads:
            anchor_loads[anchor_name] = []
        anchor_loads[anchor_name].append(col)
for anchor_name, cols in anchor_loads.items():
    if len(cols) > 0:
        F_max = df[cols[0]].abs().max()
        if F_max > 100:  # Only plot significant loads
            ax8.plot(df['t'], df[cols[0]].abs()/1000, label=anchor_name, linewidth=2)
ax8.set_xlabel('Time (s)')
ax8.set_ylabel('Load (kN)')
ax8.set_title('Anchor Loads (Peak)')
ax8.legend(fontsize=7)
ax8.grid(True, alpha=0.3)

# 9. System Energy
ax9 = plt.subplot(3, 3, 9)
# Calculate kinetic + potential energy
KE = np.zeros(len(df))
PE = np.zeros(len(df))
g = 9.80665

for col in df.columns:
    if 'body:' in col and ':v_z' in col:
        body_name = col.split(':')[1].split(':')[0]
        # Find mass (approximate from body ID)
        if 'nosecone' in body_name:
            m = 5.0
        elif 'avionics' in body_name:
            m = 8.0 if '_half' not in body_name else 4.0
        elif 'motor' in body_name:
            m = 42.0
        else:
            m = 1.0
        v = df[col].abs()
        KE += 0.5 * m * v**2
    if 'body:' in col and ':r_z' in col:
        body_name = col.split(':')[1].split(':')[0]
        if 'nosecone' in body_name:
            m = 5.0
        elif 'avionics' in body_name:
            m = 8.0 if '_half' not in body_name else 4.0
        elif 'motor' in body_name:
            m = 42.0
        else:
            m = 1.0
        h = df[col]
        PE += m * g * h

total_energy = KE + PE
ax9.plot(df['t'], KE/1000, label='Kinetic', linewidth=2)
ax9.plot(df['t'], PE/1000, label='Potential', linewidth=2)
ax9.plot(df['t'], total_energy/1000, label='Total', linewidth=2, linestyle='--')
ax9.set_xlabel('Time (s)')
ax9.set_ylabel('Energy (kJ)')
ax9.set_title('System Energy')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('out/realistic_test/simulation_plots.png', dpi=150, bbox_inches='tight')
print("Saved plots to out/realistic_test/simulation_plots.png")

# Print summary
print("\n" + "="*60)
print("SIMULATION SUMMARY")
print("="*60)
print(f"Time range: {df['t'].min():.2f}s to {df['t'].max():.2f}s")

# Find altitude columns
alt_cols = [c for c in df.columns if ('body:' in c and ':r_z' in c) or ('canopy:' in c and ':p_z' in c)]
if alt_cols:
    print(f"Initial altitude: {df[alt_cols[0]].iloc[0]:.1f} m")
    print(f"Final altitude: {df[alt_cols[0]].iloc[-1]:.1f} m")

# Find velocity columns
vel_cols = [c for c in df.columns if ('body:' in c and ':v_z' in c) or ('canopy:' in c and ':v_z' in c)]
if vel_cols:
    print(f"Initial velocity: {df[vel_cols[0]].iloc[0]:.1f} m/s")
    print(f"Final velocity: {df[vel_cols[0]].iloc[-1]:.1f} m/s")
print(f"\nPeak system force: {Fsys_mag.max():.1f} N ({Fsys_mag.max()/1000:.2f} kN)")
print(f"Peak tension: {max([df[col].max() for col in df.columns if 'edge:' in col and ':T' in col]):.1f} N")
print(f"\nEvents:")
for _, event in events_df.iterrows():
    print(f"  {event['event_type']:15s} {event['id']:25s} at t={event['t_event']:.3f}s")

