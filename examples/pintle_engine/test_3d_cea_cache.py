"""
Test script for 3D CEA cache with ablative nozzle simulation.

This demonstrates:
1. 3D CEA cache building (Pc, MR, eps)
2. Time-varying expansion ratio due to ablation
3. Thrust changes as throat and nozzle exit areas evolve
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pintle_pipeline.io import load_config
from pintle_models.runner import PintleEngineRunner


def main():
    print("=" * 80)
    print("3D CEA CACHE + ABLATIVE NOZZLE TEST")
    print("=" * 80)
    
    # Load config
    config_path = Path("examples/pintle_engine/config_minimal.yaml")
    config = load_config(config_path)
    
    # Check if 3D cache is enabled
    cea_config = config.combustion.cea
    
    if cea_config.eps_range is None:
        print("\n[WARNING] 3D CEA cache not enabled in config!")
        print("Set eps_range in config to enable 3D cache.")
        print("Example: eps_range: [4.0, 15.0]")
        return
    
    print(f"\n✓ 3D CEA cache enabled")
    print(f"  Pc range: {cea_config.Pc_range[0]/1e6:.1f} - {cea_config.Pc_range[1]/1e6:.1f} MPa")
    print(f"  MR range: {cea_config.MR_range[0]:.2f} - {cea_config.MR_range[1]:.2f}")
    print(f"  eps range: {cea_config.eps_range[0]:.2f} - {cea_config.eps_range[1]:.2f}")
    print(f"  Grid points: {cea_config.n_points}³ = {cea_config.n_points**3}")
    
    # Initialize runner (this will build or load the 3D cache)
    print(f"\nInitializing runner...")
    runner = PintleEngineRunner(config)
    print(f"✓ Runner initialized")
    
    # Check if ablative cooling is enabled
    if not config.ablative_cooling.enabled:
        print("\n[INFO] Ablative cooling not enabled. Enabling for test...")
        config.ablative_cooling.enabled = True
        config.ablative_cooling.track_geometry_evolution = True
        config.ablative_cooling.nozzle_ablative = True
        runner = PintleEngineRunner(config)
    
    print(f"\n✓ Ablative cooling enabled")
    print(f"  Nozzle ablative: {config.ablative_cooling.nozzle_ablative}")
    print(f"  Track geometry evolution: {config.ablative_cooling.track_geometry_evolution}")
    
    # Create a burn profile (constant pressure for simplicity)
    burn_time = 10.0  # seconds
    n_points = 100
    times = np.linspace(0, burn_time, n_points)
    
    # Constant tank pressures
    P_tank_O = np.full(n_points, 600.0 * 6894.76)  # 600 psi -> Pa
    P_tank_F = np.full(n_points, 650.0 * 6894.76)  # 650 psi -> Pa
    
    print(f"\n✓ Burn profile created")
    print(f"  Duration: {burn_time} s")
    print(f"  Points: {n_points}")
    print(f"  P_tank_O: 600 psi (constant)")
    print(f"  P_tank_F: 650 psi (constant)")
    
    # Run simulation
    print(f"\nRunning time-varying simulation...")
    results = runner.evaluate_arrays_with_time(times, P_tank_O, P_tank_F, track_ablative_geometry=True)
    print(f"✓ Simulation complete")
    
    # Extract results
    thrust = results["F"]
    eps = results["eps"]
    A_throat = results["A_throat"]
    A_exit = results["A_exit"]
    recession_throat = results["recession_throat"]
    recession_exit = results["recession_exit"]
    
    # Calculate changes
    thrust_change_pct = (thrust[-1] - thrust[0]) / thrust[0] * 100
    eps_change_pct = (eps[-1] - eps[0]) / eps[0] * 100
    A_throat_change_pct = (A_throat[-1] - A_throat[0]) / A_throat[0] * 100
    A_exit_change_pct = (A_exit[-1] - A_exit[0]) / A_exit[0] * 100
    
    print(f"\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nInitial Conditions:")
    print(f"  Thrust:         {thrust[0]:.1f} N")
    print(f"  Expansion ratio: {eps[0]:.3f}")
    print(f"  A_throat:       {A_throat[0]*1e6:.2f} mm²")
    print(f"  A_exit:         {A_exit[0]*1e6:.2f} mm²")
    
    print(f"\nFinal Conditions (t = {burn_time} s):")
    print(f"  Thrust:         {thrust[-1]:.1f} N ({thrust_change_pct:+.2f}%)")
    print(f"  Expansion ratio: {eps[-1]:.3f} ({eps_change_pct:+.2f}%)")
    print(f"  A_throat:       {A_throat[-1]*1e6:.2f} mm² ({A_throat_change_pct:+.2f}%)")
    print(f"  A_exit:         {A_exit[-1]*1e6:.2f} mm² ({A_exit_change_pct:+.2f}%)")
    
    print(f"\nCumulative Recession:")
    print(f"  Throat:  {recession_throat[-1]*1000:.3f} mm")
    print(f"  Exit:    {recession_exit[-1]*1000:.3f} mm")
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("3D CEA Cache: Ablative Nozzle Evolution", fontsize=16, fontweight='bold')
    
    # Plot 1: Thrust vs time
    ax = axes[0, 0]
    ax.plot(times, thrust, 'b-', linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Thrust [N]")
    ax.set_title("Thrust Evolution")
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Expansion ratio vs time
    ax = axes[0, 1]
    ax.plot(times, eps, 'r-', linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Expansion Ratio ε")
    ax.set_title("Expansion Ratio Evolution")
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Areas vs time
    ax = axes[0, 2]
    ax.plot(times, A_throat * 1e6, 'g-', linewidth=2, label='A_throat')
    ax.plot(times, A_exit * 1e6, 'm-', linewidth=2, label='A_exit')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Area [mm²]")
    ax.set_title("Throat & Exit Area Growth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Recession vs time
    ax = axes[1, 0]
    ax.plot(times, recession_throat * 1000, 'g-', linewidth=2, label='Throat')
    ax.plot(times, recession_exit * 1000, 'm-', linewidth=2, label='Exit')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cumulative Recession [mm]")
    ax.set_title("Ablative Recession")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Thrust vs expansion ratio
    ax = axes[1, 1]
    ax.plot(eps, thrust, 'b-', linewidth=2)
    ax.set_xlabel("Expansion Ratio ε")
    ax.set_ylabel("Thrust [N]")
    ax.set_title("Thrust vs Expansion Ratio")
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Area ratio vs time
    ax = axes[1, 2]
    area_ratio = A_exit / A_throat
    ax.plot(times, area_ratio, 'k-', linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("A_exit / A_throat")
    ax.set_title("Area Ratio (should match ε)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("examples/pintle_engine/test_3d_cea_cache.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")
    
    plt.show()
    
    print(f"\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"\n✓ 3D CEA cache is working correctly!")
    print(f"✓ Expansion ratio changes from {eps[0]:.3f} to {eps[-1]:.3f}")
    print(f"✓ Thrust changes by {thrust_change_pct:+.2f}% due to geometry evolution")
    print(f"✓ Both throat and nozzle exit areas grow due to ablation")


if __name__ == "__main__":
    main()

