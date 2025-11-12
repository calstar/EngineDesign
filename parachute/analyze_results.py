"""Analyze simulation results"""
import pandas as pd
import numpy as np

# Load telemetry
df = pd.read_csv('out/realistic_test/telemetry.csv')
events_df = pd.read_csv('out/realistic_test/events.csv')

print("=" * 60)
print("SIMULATION RESULTS ANALYSIS")
print("=" * 60)

print(f"\nTime Range: {df['t'].min():.3f}s to {df['t'].max():.3f}s")
print(f"Total Timesteps: {len(df)}")

print(f"\n{'='*60}")
print("EVENTS")
print(f"{'='*60}")
for idx, row in events_df.iterrows():
    print(f"{row['event_type']:15s} {row['id']:25s} at t={row['t_event']:.4f}s")
    if row['extra']:
        import json
        extra = json.loads(row['extra'])
        for k, v in extra.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")

print(f"\n{'='*60}")
print("PEAK FORCES")
print(f"{'='*60}")
# System forces
Fsys_mag = np.sqrt(df['Fsys_x']**2 + df['Fsys_y']**2 + df['Fsys_z']**2)
print(f"System Total Force: {Fsys_mag.max():.1f} N ({Fsys_mag.max()/1000:.1f} kN)")
print(f"  Fsys_x: {df['Fsys_x'].abs().max():.1f} N")
print(f"  Fsys_y: {df['Fsys_y'].abs().max():.1f} N")
print(f"  Fsys_z: {df['Fsys_z'].abs().max():.1f} N")

# Body forces
print(f"\nBody Forces:")
for col in df.columns:
    if 'body:' in col and ':F' in col and 'mag' not in col:
        max_val = df[col].abs().max()
        if max_val > 100:  # Only show significant forces
            print(f"  {col}: {max_val:.1f} N ({max_val/1000:.1f} kN)")

print(f"\n{'='*60}")
print("TENSIONS")
print(f"{'='*60}")
for col in df.columns:
    if 'edge:' in col and ':T' in col:
        max_tension = df[col].max()
        if max_tension > 100:  # Only show significant tensions
            print(f"  {col}: {max_tension:.1f} N ({max_tension/1000:.1f} kN)")

print(f"\n{'='*60}")
print("CANOPY INFLATION")
print(f"{'='*60}")
for col in df.columns:
    if 'canopy:' in col and ':A' in col:
        max_area = df[col].max()
        if max_area > 0.01:
            print(f"  {col}: max={max_area:.2f} m²")

print(f"\n{'='*60}")
print("ANCHOR LOADS (Peak)")
print(f"{'='*60}")
anchor_cols = [c for c in df.columns if 'anchor:' in c and 'Fmag' in c]
for col in anchor_cols:
    max_load = df[col].max()
    if max_load > 100:  # Only show significant loads
        print(f"  {col}: {max_load:.1f} N ({max_load/1000:.1f} kN)")

print(f"\n{'='*60}")
print("LAST TIMESTEP (Before instability)")
print(f"{'='*60}")
# Get last valid timestep (before explosion)
valid_idx = df[Fsys_mag < 1e6].index[-1] if len(df[Fsys_mag < 1e6]) > 0 else -1
print(f"Last valid timestep: t={df.loc[valid_idx, 't']:.4f}s")
print(f"System force: {Fsys_mag.iloc[valid_idx]:.1f} N")

