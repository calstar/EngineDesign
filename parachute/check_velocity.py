"""Check canopy velocity to see if it's causing huge drag"""
import pandas as pd
import numpy as np

df = pd.read_csv('out/realistic_test/telemetry.csv')

# Find last valid timestep
valid_mask = df['Fsys_z'].abs() < 1e6
if valid_mask.sum() > 0:
    valid_idx = df[valid_mask].index[-1]
    t = df.loc[valid_idx, 't']
    
    print(f"At t={t:.4f}s (last valid timestep):")
    print("=" * 60)
    
    # Check canopy velocity
    if 'canopy:drogue:v_z' in df.columns:
        v_drogue = df.loc[valid_idx, 'canopy:drogue:v_z']
        print(f"Drogue canopy velocity: {v_drogue:.1f} m/s")
    
    # Check body velocity
    if 'body:nosecone:v_z' in df.columns:
        v_nosecone = df.loc[valid_idx, 'body:nosecone:v_z']
        print(f"Nosecone velocity: {v_nosecone:.1f} m/s")
    
    # Check relative velocity
    if 'canopy:drogue:v_z' in df.columns and 'body:nosecone:v_z' in df.columns:
        v_rel = abs(v_drogue - v_nosecone)
        print(f"Relative velocity: {v_rel:.1f} m/s")
    
    # Calculate expected drag
    rho = 1.225
    A = df.loc[valid_idx, 'canopy:drogue:A'] if 'canopy:drogue:A' in df.columns else 0
    CD = df.loc[valid_idx, 'canopy:drogue:CD'] if 'canopy:drogue:CD' in df.columns else 0
    V = abs(v_drogue) if 'canopy:drogue:v_z' in df.columns else 50.0
    
    F_drag_expected = 0.5 * rho * V**2 * A * CD
    print(f"\nExpected drag force:")
    print(f"  V = {V:.1f} m/s")
    print(f"  A = {A:.2f} m²")
    print(f"  CD = {CD:.2f}")
    print(f"  F = 0.5 * {rho} * {V}² * {A} * {CD} = {F_drag_expected:.1f} N ({F_drag_expected/1000:.2f} kN)")
    
    # Check actual anchor load
    anchor_load = df.loc[valid_idx, 'anchor:nosecone:drogue_attach:Fmag_I'] if 'anchor:nosecone:drogue_attach:Fmag_I' in df.columns else 0
    print(f"\nActual anchor load: {anchor_load:.1f} N ({anchor_load/1000:.2f} kN)")
    print(f"Ratio: {anchor_load/F_drag_expected:.1f}x")
    
    # Check if velocity is growing
    print("\n" + "=" * 60)
    print("Velocity history (check for runaway):")
    print("=" * 60)
    for i in range(max(0, valid_idx-500), valid_idx+1, 50):
        if 'canopy:drogue:v_z' in df.columns:
            v = df.loc[i, 'canopy:drogue:v_z']
            t_val = df.loc[i, 't']
            print(f"  t={t_val:.3f}s: v={v:.1f} m/s")

