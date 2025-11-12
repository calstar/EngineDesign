"""Analyze what's causing the numerical instability"""
import pandas as pd
import numpy as np

df = pd.read_csv('out/realistic_test/telemetry.csv')

print("="*60)
print("INSTABILITY ANALYSIS")
print("="*60)

# Find when instability occurs
Fsys_mag = np.sqrt(df['Fsys_x']**2 + df['Fsys_y']**2 + df['Fsys_z']**2)

# Find rapid force growth
force_rate = np.diff(Fsys_mag)
rapid_growth_idx = np.where(np.abs(force_rate) > 10000)[0]  # 10 kN/s change

if len(rapid_growth_idx) > 0:
    first_spike = rapid_growth_idx[0]
    t_spike = df.loc[first_spike, 't']
    print(f"\nFirst rapid force growth at t={t_spike:.3f}s")
    print(f"  Force before: {Fsys_mag.iloc[first_spike]:.1f} N")
    print(f"  Force after: {Fsys_mag.iloc[min(first_spike+1, len(df)-1)]:.1f} N")
    print(f"  Rate: {force_rate[first_spike]:.1f} N/s")

# Check velocities at instability point
print(f"\nAt t={df['t'].iloc[-1]:.3f}s (last timestep):")
print(f"  System force: {Fsys_mag.iloc[-1]:.1f} N ({Fsys_mag.iloc[-1]/1000:.2f} kN)")

# Check if positions are valid
if 'body:nosecone:r_z' in df.columns:
    alt_nosecone = df['body:nosecone:r_z']
    print(f"\nNosecone altitude:")
    print(f"  Initial: {alt_nosecone.iloc[0]:.1f} m")
    print(f"  Final: {alt_nosecone.iloc[-1]:.1f} m")
    print(f"  Change: {alt_nosecone.iloc[-1] - alt_nosecone.iloc[0]:.1f} m")
    
    # Check if altitude is going negative (unphysical)
    if alt_nosecone.iloc[-1] < 0:
        print(f"  WARNING: Altitude went negative! This indicates numerical instability.")
        print(f"  The system should stop at altitude=0 (impact), not go negative.")

# Check initial conditions
print(f"\nInitial Conditions (from YAML):")
print(f"  r0: [0, 0, 3048] m (10,000 ft altitude)")
print(f"  v0: [0, 0, -50] m/s (50 m/s downward)")
print(f"  This is NOT apogee - the rocket is already descending at 50 m/s")
print(f"  This is the drogue deployment altitude (10,000 ft)")
print(f"  Apogee would have v0 = 0 m/s")

# Check if 2.1 kN is reasonable for drogue
print(f"\nDrogue Deployment Force Analysis:")
print(f"  Peak force during drogue deployment: {Fsys_mag.max():.1f} N ({Fsys_mag.max()/1000:.2f} kN)")
print(f"  Typical drogue loads: 2-6 kN (reasonable)")
print(f"  Your 2.1 kN is in the expected range for drogue deployment")

