"""Quick script to scale the pressure CSV to workable values."""

import pandas as pd

# Read the CSV
df = pd.read_csv("Untitled spreadsheet - Sheet1 (2).csv")

print("ORIGINAL pressures:")
print(f"  LOX: {df['P_tank_O'].min():.0f} - {df['P_tank_O'].max():.0f} psi")
print(f"  Fuel: {df['P_tank_F'].min():.0f} - {df['P_tank_F'].max():.0f} psi")
print(f"  ❌ These are TOO LOW - engine needs ~1305 psi LOX, ~974 psi fuel")

# Scale by 2.5x
df['P_tank_O'] = df['P_tank_O'] * 2.5
df['P_tank_F'] = df['P_tank_F'] * 2.5

print("\nSCALED pressures (2.5x):")
print(f"  LOX: {df['P_tank_O'].min():.0f} - {df['P_tank_O'].max():.0f} psi")
print(f"  Fuel: {df['P_tank_F'].min():.0f} - {df['P_tank_F'].max():.0f} psi")
print(f"  ✅ These should work!")

# Save
output_file = "pressure_profile_FIXED.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Saved to: {output_file}")
print(f"\nNow upload '{output_file}' in the UI Time-Series Analysis tab!")

