import pandas as pd
import numpy as np

df = pd.read_csv('out/simple_test/telemetry.csv')

print("First 20 rows (detailed):")
print(df[['t', 'edge:shockcord:T', 'edge:shockcord:L', 'edge:shockcord:x', 
          'canopy:main:A', 'body:rocket:Fz', 'Fsys_z']].head(20))

print("\n" + "="*60)
print("Checking for issues:")
print("="*60)

# Check when edge length becomes problematic
problem_idx = df[df['edge:shockcord:L'] > 100].index[0] if len(df[df['edge:shockcord:L'] > 100]) > 0 else None
if problem_idx is not None:
    print(f"\nFirst time edge length > 100m: t={df.iloc[problem_idx]['t']:.3f}s")
    print(f"  Tension: {df.iloc[problem_idx]['edge:shockcord:T']:.1f} N")
    print(f"  Extension: {df.iloc[problem_idx]['edge:shockcord:x']:.3f} m")
    print(f"  Canopy area: {df.iloc[problem_idx]['canopy:main:A']:.3f} m²")
    
    # Check drag force estimate
    if problem_idx > 0:
        prev_row = df.iloc[problem_idx-1]
        print(f"\nPrevious step:")
        print(f"  Edge length: {prev_row['edge:shockcord:L']:.3f} m")
        print(f"  Tension: {prev_row['edge:shockcord:T']:.1f} N")

# Calculate estimated drag force
print("\n" + "="*60)
print("Estimated drag forces:")
print("="*60)
rho = 1.225
CD = 2.0
for i in [0, 10, 50, 100, 200]:
    if i < len(df):
        row = df.iloc[i]
        A = row['canopy:main:A']
        # Estimate velocity magnitude (assuming horizontal at 67.97 m/s initially)
        V_est = 67.97  # Rough estimate
        F_drag_est = 0.5 * rho * CD * A * V_est * V_est
        print(f"t={row['t']:.3f}s: A={A:.3f}m², F_drag_est={F_drag_est/1000:.1f}kN")

print(f"\nKevlar max tension (cap): 200 kN")
print(f"Kevlar stiffness: 100,000 N/m")

