"""Diagnose Cf discrepancy and graphite insert behavior."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from pintle_pipeline.io import load_config
from pintle_models.runner import PintleEngineRunner

config = load_config('config_minimal.yaml')
runner = PintleEngineRunner(config)

print("=" * 80)
print("CF DISCREPANCY DIAGNOSIS")
print("=" * 80)

# Single point evaluation
results = runner.evaluate(3.5e6, 3.5e6)

Pc = results["Pc"]
MR = results["MR"]
mdot_total = results["mdot_total"]
F = results["F"]
v_exit = results["v_exit"]
P_exit = results["P_exit"]
M_exit = results["M_exit"]
Cf_actual = results["Cf_actual"]
Cf_ideal = results["Cf_ideal"]
A_throat = results["A_throat"]
A_exit = results["A_exit"]
eps = A_exit / A_throat
Pa = 101325.0

print(f"\n--- CONDITIONS ---")
print(f"Pc: {Pc/1e6:.3f} MPa")
print(f"MR: {MR:.3f}")
print(f"mdot: {mdot_total:.4f} kg/s")
print(f"eps: {eps:.4f}")
print(f"A_throat: {A_throat*1e6:.2f} mm²")
print(f"A_exit: {A_exit*1e6:.2f} mm²")

print(f"\n--- EXIT CONDITIONS ---")
print(f"M_exit: {M_exit:.4f}")
print(f"P_exit: {P_exit/1e6:.3f} MPa")
print(f"Pa: {Pa/1e6:.3f} MPa")
print(f"P_exit/Pa: {P_exit/Pa:.4f}")

print(f"\n--- THRUST CALCULATION ---")
F_momentum = mdot_total * v_exit
F_pressure = (P_exit - Pa) * A_exit
F_calculated = F_momentum + F_pressure
print(f"F_momentum: {F_momentum:.1f} N")
print(f"F_pressure: {F_pressure:.1f} N")
print(f"F_total (calculated): {F_calculated:.1f} N")
print(f"F (from results): {F:.1f} N")

print(f"\n--- CF ANALYSIS ---")
print(f"Cf_actual: {Cf_actual:.4f}")
print(f"Cf_ideal (CEA): {Cf_ideal:.4f}")
print(f"Ratio: {Cf_actual/Cf_ideal:.4f} ({((Cf_actual/Cf_ideal - 1.0)*100):.1f}% higher)")

# What should Cf be?
# Cf = F / (Pc * A_throat)
Cf_from_F = F / (Pc * A_throat)
print(f"Cf_from_F: {Cf_from_F:.4f}")

# Check if CEA Cf_ideal is for different conditions
print(f"\n--- CEA CF_IDEAL CHECK ---")
print(f"CEA Cf_ideal assumes:")
print(f"  - Equilibrium flow (gamma constant)")
print(f"  - Sea level ambient (Pa = 101325 Pa)")
print(f"  - Expansion ratio: {eps:.4f}")
print(f"  - Optimally expanded (P_exit = Pa)")

# Calculate what Cf should be theoretically
# For optimally expanded: Cf = sqrt(2*gamma^2/(gamma-1) * (2/(gamma+1))^((gamma+1)/(gamma-1)) * (1 - (Pa/Pc)^((gamma-1)/gamma)))
# But this is for equilibrium
gamma = results.get("gamma", 1.2)
gamma_exit = results.get("gamma_exit", gamma)
print(f"\n--- GAMMA COMPARISON ---")
print(f"gamma_chamber: {gamma:.4f}")
print(f"gamma_exit: {gamma_exit:.4f}")
if abs(gamma_exit - gamma) > 0.01:
    print(f"  [WARNING] Shifting equilibrium active - gamma changes!")
    print(f"  This means CEA's equilibrium Cf_ideal may not match actual conditions")

# Check graphite insert
print(f"\n" + "=" * 80)
print("GRAPHITE INSERT DIAGNOSIS")
print("=" * 80)

if hasattr(config, 'graphite_insert') and config.graphite_insert:
    graphite = config.graphite_insert
    print(f"\n--- GRAPHITE INSERT CONFIG ---")
    print(f"Enabled: {graphite.enabled}")
    print(f"Initial thickness: {graphite.initial_thickness*1000:.1f} mm")
    print(f"Oxidation rate: {graphite.oxidation_rate*1e6:.3f} µm/s")
    print(f"Coverage fraction: {graphite.coverage_fraction:.2f}")
    
    # Time series to check recession
    print(f"\n--- TIME SERIES ANALYSIS (10s burn) ---")
    times = np.linspace(0, 10, 101)
    P_tank_O = 3.5e6 * np.ones_like(times)
    P_tank_F = 3.5e6 * np.ones_like(times)
    
    time_results = runner.evaluate_arrays_with_time(
        times, P_tank_O, P_tank_F, track_ablative_geometry=True
    )
    
    recession_throat = time_results.get("recession_throat", np.zeros_like(times))
    recession_chamber = time_results.get("recession_chamber", np.zeros_like(times))
    A_throat_history = time_results.get("A_throat", np.full_like(times, A_throat))
    
    print(f"Initial throat area: {A_throat*1e6:.2f} mm²")
    print(f"Final throat area: {A_throat_history[-1]*1e6:.2f} mm²")
    print(f"Area change: {((A_throat_history[-1]/A_throat - 1.0)*100):.2f}%")
    print(f"\nThroat recession:")
    print(f"  Initial: {recession_throat[0]*1e6:.2f} µm")
    print(f"  Final: {recession_throat[-1]*1e6:.2f} µm")
    print(f"  Total: {recession_throat[-1]*1e6:.2f} µm")
    print(f"\nChamber recession:")
    print(f"  Initial: {recession_chamber[0]*1e6:.2f} µm")
    print(f"  Final: {recession_chamber[-1]*1e6:.2f} µm")
    print(f"  Total: {recession_chamber[-1]*1e6:.2f} µm")
    
    if recession_throat[-1] > 0:
        recession_rate_avg = recession_throat[-1] / times[-1]
        print(f"\nAverage throat recession rate: {recession_rate_avg*1e6:.3f} µm/s")
        print(f"  [EXPECTED] Graphite should have very low recession (< 0.1 µm/s)")
        
        if recession_rate_avg > 1e-7:  # > 0.1 µm/s
            print(f"  [WARNING] Throat recession rate is HIGH - graphite insert may not be working correctly!")
    
    # Check if throat area is staying constant
    A_throat_change = np.max(np.abs(np.diff(A_throat_history))) / A_throat * 100
    print(f"\nThroat area stability:")
    print(f"  Max change per step: {A_throat_change:.4f}%")
    if A_throat_change > 0.1:
        print(f"  [WARNING] Throat area is changing significantly - graphite insert should keep it constant!")
else:
    print("\n[INFO] Graphite insert not enabled in config")

print("\n" + "=" * 80)


