"""Detailed force analysis to find where 994kN is coming from"""
import pandas as pd
import numpy as np

df = pd.read_csv('out/realistic_test/telemetry.csv')

# Find last valid timestep (before explosion)
valid_mask = df['Fsys_z'].abs() < 1e6
if valid_mask.sum() > 0:
    valid_idx = df[valid_mask].index[-1]
    print(f"Last valid timestep: t={df.loc[valid_idx, 't']:.4f}s")
    print("=" * 60)
    
    # System forces
    Fsys_mag = np.sqrt(df['Fsys_x']**2 + df['Fsys_y']**2 + df['Fsys_z']**2)
    print(f"System Total Force: {Fsys_mag.iloc[valid_idx]:.1f} N ({Fsys_mag.iloc[valid_idx]/1000:.1f} kN)")
    print(f"  Fsys_x: {df.loc[valid_idx, 'Fsys_x']:.1f} N")
    print(f"  Fsys_y: {df.loc[valid_idx, 'Fsys_y']:.1f} N")
    print(f"  Fsys_z: {df.loc[valid_idx, 'Fsys_z']:.1f} N")
    
    print("\n" + "=" * 60)
    print("BODY FORCES (at last valid timestep)")
    print("=" * 60)
    for col in df.columns:
        if 'body:' in col and ':F' in col and 'mag' not in col:
            val = df.loc[valid_idx, col]
            if abs(val) > 100:
                print(f"  {col}: {val:.1f} N ({val/1000:.1f} kN)")
    
    print("\n" + "=" * 60)
    print("TENSIONS (at last valid timestep)")
    print("=" * 60)
    for col in df.columns:
        if 'edge:' in col and ':T' in col:
            val = df.loc[valid_idx, col]
            if val > 10:
                print(f"  {col}: {val:.1f} N ({val/1000:.1f} kN)")
    
    print("\n" + "=" * 60)
    print("CANOPY INFLATION (at last valid timestep)")
    print("=" * 60)
    for col in df.columns:
        if 'canopy:' in col and ':A' in col:
            val = df.loc[valid_idx, col]
            if val > 0.01:
                print(f"  {col}: {val:.2f} m²")
        if 'canopy:' in col and ':CD' in col:
            val = df.loc[valid_idx, col]
            if val > 0.01:
                print(f"  {col}: {val:.2f}")
    
    print("\n" + "=" * 60)
    print("ANCHOR LOADS (at last valid timestep)")
    print("=" * 60)
    for col in df.columns:
        if 'anchor:' in col and 'Fmag' in col:
            val = df.loc[valid_idx, col]
            if val > 100:
                print(f"  {col}: {val:.1f} N ({val/1000:.1f} kN)")
    
    print("\n" + "=" * 60)
    print("FORCE BREAKDOWN: Check if canopy drag is reasonable")
    print("=" * 60)
    # Calculate expected drag force for drogue
    # F_drag = 0.5 * rho * V^2 * A * CD
    rho = 1.225  # kg/m³
    # Get canopy velocity
    if 'canopy:drogue:v_z' in df.columns:
        v_drogue = df.loc[valid_idx, 'canopy:drogue:v_z']
        V = abs(v_drogue)  # Magnitude
        A = df.loc[valid_idx, 'canopy:drogue:A'] if 'canopy:drogue:A' in df.columns else 0
        CD = df.loc[valid_idx, 'canopy:drogue:CD'] if 'canopy:drogue:CD' in df.columns else 0
        F_drag_expected = 0.5 * rho * V**2 * A * CD
        print(f"Drogue canopy:")
        print(f"  Velocity: {v_drogue:.1f} m/s")
        print(f"  Area: {A:.2f} m²")
        print(f"  CD: {CD:.2f}")
        print(f"  Expected drag: {F_drag_expected:.1f} N ({F_drag_expected/1000:.1f} kN)")
    
    print("\n" + "=" * 60)
    print("FORCE HISTORY: Check when forces start growing")
    print("=" * 60)
    # Find when forces start growing
    Fsys_mag = np.sqrt(df['Fsys_x']**2 + df['Fsys_y']**2 + df['Fsys_z']**2)
    # Sample every 100 timesteps
    for i in range(0, min(valid_idx+1, len(df)), 100):
        t = df.loc[i, 't']
        F = Fsys_mag.iloc[i]
        print(f"  t={t:.3f}s: F={F:.1f} N ({F/1000:.1f} kN)")

