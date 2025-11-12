"""Check if canopy drag calculation is correct"""
import numpy as np

# Test canopy drag calculation
rho = 1.225  # kg/m³
A = 1.02  # m² (drogue area at t=0.57s)
CD = 0.98  # drag coefficient
V = 50.0  # m/s (typical descent velocity)

# Expected drag force
F_drag = 0.5 * rho * V**2 * A * CD
print(f"Expected drag for drogue:")
print(f"  rho = {rho} kg/m³")
print(f"  A = {A} m²")
print(f"  CD = {CD}")
print(f"  V = {V} m/s")
print(f"  F_drag = 0.5 * {rho} * {V}² * {A} * {CD}")
print(f"  F_drag = {F_drag:.1f} N ({F_drag/1000:.2f} kN)")
print(f"\nBut we're seeing 979.7 kN on nosecone!")
print(f"That's {979700/F_drag:.0f}x too high!")
print(f"\nPossible issues:")
print(f"  1. Added mass reaction force is huge")
print(f"  2. Drag force is being multiplied somewhere")
print(f"  3. Forces are being double-counted")
print(f"  4. Velocity is much higher than expected")

