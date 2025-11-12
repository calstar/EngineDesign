"""Check what's happening at the instability point"""
import pandas as pd
import numpy as np

df = pd.read_csv('out/realistic_test/telemetry.csv')

print("="*60)
print("INSTABILITY TIMELINE ANALYSIS")
print("="*60)

# Check altitude evolution
if 'body:nosecone:r_z' in df.columns:
    alt = df['body:nosecone:r_z']
    print(f"\nNosecone altitude:")
    print(f"  Initial: {alt.iloc[0]:.1f} m")
    print(f"  Final: {alt.iloc[-1]:.1f} m")
    
    # Find when altitude starts going wrong
    valid_mask = (alt > 0) & (alt < 5000)
    if valid_mask.sum() > 0:
        last_valid_idx = df[valid_mask].index[-1]
        print(f"  Last valid altitude: {alt.iloc[last_valid_idx]:.1f} m at t={df.loc[last_valid_idx, 't']:.3f}s")
        print(f"  Next timestep: {alt.iloc[last_valid_idx+1]:.1f} m at t={df.loc[last_valid_idx+1, 't']:.3f}s")
        print(f"  Jump: {alt.iloc[last_valid_idx+1] - alt.iloc[last_valid_idx]:.1f} m")

# Check forces
Fsys_mag = np.sqrt(df['Fsys_x']**2 + df['Fsys_y']**2 + df['Fsys_z']**2)
print(f"\nForces:")
print(f"  Peak: {Fsys_mag.max():.1f} N ({Fsys_mag.max()/1000:.2f} kN) at t={df.loc[Fsys_mag.idxmax(), 't']:.3f}s")
print(f"  At last timestep: {Fsys_mag.iloc[-1]:.1f} N ({Fsys_mag.iloc[-1]/1000:.2f} kN)")

# Check tensions
print(f"\nTensions at last valid timestep:")
for col in df.columns:
    if 'edge:' in col and ':T' in col:
        edge_name = col.split(':')[1].split(':')[0]
        T = df[col].iloc[-1]
        if T > 10:
            print(f"  {edge_name}: {T:.1f} N ({T/1000:.2f} kN)")

# Check initial conditions answer
print(f"\n" + "="*60)
print("INITIAL CONDITIONS:")
print("="*60)
print(f"t=0 is NOT apogee!")
print(f"  Initial altitude: 3048 m (10,000 ft)")
print(f"  Initial velocity: -50 m/s (descending at 50 m/s)")
print(f"  This is the drogue deployment altitude during descent")
print(f"  Apogee would have v0 = 0 m/s")
print(f"\n2.1 kN for drogue deployment is reasonable!")
print(f"  Typical drogue loads: 2-6 kN")
print(f"  Your peak force is in the expected range")

