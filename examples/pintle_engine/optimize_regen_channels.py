"""Optimize regen cooling channel dimensions to achieve ~90 psi pressure drop"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from pintle_pipeline.regen_cooling import delta_p_regen_channels
from pintle_pipeline.config_schemas import RegenCoolingConfig

# Target conditions
mdot = 0.83  # kg/s (typical fuel flow)
rho = 820  # kg/m³ (RP-1 density)
mu = 0.0015  # Pa·s (RP-1 viscosity)
P_tank = 974 * 6894.76  # Pa
target_dp = 90 * 6894.76  # Pa (target pressure drop)

# Fixed parameters
d_inlet = 0.009525  # m (3/8")
L_inlet = 0.5  # m
channel_length = 0.18162  # m (chamber length)
L_outlet = 0.1  # m
roughness = 0.0
K_manifold_split = 0.5
K_manifold_merge = 0.3

# Try different configurations
print("=" * 80)
print("OPTIMIZING REGEN COOLING CHANNELS FOR ~90 PSI PRESSURE DROP")
print("=" * 80)

configs_to_try = [
    {"n_channels": 100, "width": 0.001, "height": 0.001},  # 100 channels, 1mm x 1mm (75.45 psi)
    {"n_channels": 100, "width": 0.0009, "height": 0.001},  # 100 channels, 0.9mm x 1mm
    {"n_channels": 100, "width": 0.001, "height": 0.0009},  # 100 channels, 1mm x 0.9mm
    {"n_channels": 100, "width": 0.0009, "height": 0.0009},  # 100 channels, 0.9mm x 0.9mm
    {"n_channels": 80, "width": 0.001, "height": 0.001},  # 80 channels, 1mm x 1mm
    {"n_channels": 90, "width": 0.001, "height": 0.001},  # 90 channels, 1mm x 1mm
]

best_config = None
best_error = float('inf')

for cfg in configs_to_try:
    n = cfg["n_channels"]
    w = cfg["width"]
    h = cfg["height"]
    
    # Create config
    regen_config = RegenCoolingConfig(
        enabled=True,
        d_inlet=d_inlet,
        L_inlet=L_inlet,
        n_channels=n,
        channel_width=w,
        channel_height=h,
        channel_length=channel_length,
        d_outlet=None,
        L_outlet=L_outlet,
        roughness=roughness,
        K_manifold_split=K_manifold_split,
        K_manifold_merge=K_manifold_merge,
        Cd_entrance_inf=0.8,
        a_Re_entrance=0.1,
        Cd_entrance_min=0.6,
        Cd_exit_inf=0.9,
        a_Re_exit=0.1,
        Cd_exit_min=0.7,
    )
    
    # Calculate pressure drop
    try:
        dp = delta_p_regen_channels(mdot, rho, mu, regen_config, P_tank)
        dp_psi = dp / 6894.76
        
        error = abs(dp_psi - 90)
        
        print(f"\n{n} channels, {w*1000:.1f}mm × {h*1000:.1f}mm:")
        print(f"  Pressure drop: {dp_psi:.2f} psi")
        print(f"  Error from target: {error:.2f} psi")
        
        # Calculate channel velocity
        A_channel = w * h
        mdot_per_channel = mdot / n
        u_channel = mdot_per_channel / (rho * A_channel)
        print(f"  Channel velocity: {u_channel:.2f} m/s")
        print(f"  Total channel area: {n * A_channel * 1e6:.2f} mm²")
        
        if error < best_error:
            best_error = error
            best_config = cfg.copy()
            best_config['dp_psi'] = dp_psi
            best_config['u_channel'] = u_channel
            best_config['total_area_mm2'] = n * A_channel * 1e6
    except Exception as e:
        print(f"\n{n} channels, {w*1000:.1f}mm × {h*1000:.1f}mm: ERROR - {e}")

if best_config:
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION:")
    print("=" * 80)
    print(f"  Number of channels: {best_config['n_channels']}")
    print(f"  Channel width: {best_config['width']*1000:.2f} mm")
    print(f"  Channel height: {best_config['height']*1000:.2f} mm")
    print(f"  Pressure drop: {best_config['dp_psi']:.2f} psi (target: 90 psi)")
    print(f"  Channel velocity: {best_config['u_channel']:.2f} m/s")
    print(f"  Total channel area: {best_config['total_area_mm2']:.2f} mm²")
    print("\n" + "=" * 80)

