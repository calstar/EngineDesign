"""Diagnose what happens during and after separation"""
import pandas as pd
import numpy as np

df = pd.read_csv('out/realistic_test/telemetry.csv')
events_df = pd.read_csv('out/realistic_test/events.csv')

print("="*60)
print("SEPARATION DYNAMICS ANALYSIS")
print("="*60)

# Find separation event
separation_time = None
for _, event in events_df.iterrows():
    if event['event_type'] == 'separation':
        separation_time = event['t_event']
        print(f"Separation at t={separation_time:.3f}s")
        break

if separation_time:
    # Get data around separation
    sep_idx = df[df['t'] >= separation_time].index[0]
    print(f"\nBefore separation (t={df.loc[sep_idx-1, 't']:.3f}s):")
    
    # Check velocities
    if 'body:avionics:v_z' in df.columns:
        v_before = df.loc[sep_idx-1, 'body:avionics:v_z']
        print(f"  Avionics velocity: {v_before:.1f} m/s")
    
    print(f"\nAfter separation (t={df.loc[sep_idx, 't']:.3f}s):")
    if 'body:avionics:v_z' in df.columns:
        v_after = df.loc[sep_idx, 'body:avionics:v_z']
        print(f"  Avionics velocity: {v_after:.1f} m/s")
        print(f"  Change: {v_after - v_before:.1f} m/s")
    
    if 'body:avionics_half:v_z' in df.columns:
        v_half = df.loc[sep_idx, 'body:avionics_half:v_z']
        print(f"  Avionics_half velocity: {v_half:.1f} m/s")
        print(f"  Relative velocity: {abs(v_after - v_half):.1f} m/s (expected ~6 m/s)")

# Check altitude evolution
if 'body:nosecone:r_z' in df.columns:
    alt = df['body:nosecone:r_z']
    print(f"\nAltitude evolution:")
    print(f"  Initial: {alt.iloc[0]:.1f} m")
    if separation_time:
        sep_idx = df[df['t'] >= separation_time].index[0]
        print(f"  At separation: {alt.iloc[sep_idx]:.1f} m")
    print(f"  Final: {alt.iloc[-1]:.1f} m")
    
    # Check if altitude is increasing (unphysical)
    if alt.iloc[-1] > alt.iloc[0]:
        print(f"  WARNING: Altitude is INCREASING! This is unphysical.")
        print(f"  The system should be descending, not ascending.")
        # Find when it starts going up
        alt_diff = np.diff(alt)
        first_positive = np.where(alt_diff > 10)[0]  # 10m jump up
        if len(first_positive) > 0:
            idx = first_positive[0]
            print(f"  First upward jump: {alt_diff[idx]:.1f} m at t={df.loc[idx, 't']:.3f}s")

# Check forces around separation
Fsys_mag = np.sqrt(df['Fsys_x']**2 + df['Fsys_y']**2 + df['Fsys_z']**2)
if separation_time:
    sep_idx = df[df['t'] >= separation_time].index[0]
    print(f"\nForces:")
    print(f"  Before separation: {Fsys_mag.iloc[sep_idx-1]:.1f} N ({Fsys_mag.iloc[sep_idx-1]/1000:.2f} kN)")
    print(f"  After separation: {Fsys_mag.iloc[sep_idx]:.1f} N ({Fsys_mag.iloc[sep_idx]/1000:.2f} kN)")
    print(f"  Peak force: {Fsys_mag.max():.1f} N ({Fsys_mag.max()/1000:.2f} kN) at t={df.loc[Fsys_mag.idxmax(), 't']:.3f}s")

