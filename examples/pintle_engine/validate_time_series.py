"""Comprehensive time-series validation for all physics parameters."""

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
print("TIME-SERIES VALIDATION")
print("=" * 80)

# 10 second burn
times = np.linspace(0, 10, 101)
P_tank_O = 3.5e6 * np.ones_like(times)
P_tank_F = 3.5e6 * np.ones_like(times)

print("\nRunning time-series simulation...")
results = runner.evaluate_arrays_with_time(
    times, P_tank_O, P_tank_F, track_ablative_geometry=True
)

print("\n" + "=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)

# 1. Thrust and Performance
print("\n--- THRUST & PERFORMANCE ---")
F = results.get("F", np.zeros_like(times))
Isp = results.get("Isp", np.zeros_like(times))
Pc = results.get("Pc", np.zeros_like(times))
MR = results.get("MR", np.zeros_like(times))

print(f"Initial F: {F[0]:.1f} N")
print(f"Final F: {F[-1]:.1f} N")
print(f"F change: {((F[-1]/F[0] - 1.0)*100):.2f}%")
print(f"Initial Isp: {Isp[0]:.1f} s")
print(f"Final Isp: {Isp[-1]:.1f} s")
print(f"Isp change: {((Isp[-1]/Isp[0] - 1.0)*100):.2f}%")

# Check for monotonicity (should decrease due to recession)
F_decreasing = np.all(np.diff(F) <= 0.1)  # Allow small numerical noise
if not F_decreasing:
    print(f"  [WARNING] Thrust is not monotonically decreasing (expected due to recession)")

# 2. Geometry Evolution
print("\n--- GEOMETRY EVOLUTION ---")
A_throat = results.get("A_throat", np.zeros_like(times))
A_exit = results.get("A_exit", np.zeros_like(times))
V_chamber = results.get("V_chamber", np.zeros_like(times))
eps = A_exit / A_throat

print(f"Initial A_throat: {A_throat[0]*1e6:.2f} mm²")
print(f"Final A_throat: {A_throat[-1]*1e6:.2f} mm²")
print(f"A_throat change: {((A_throat[-1]/A_throat[0] - 1.0)*100):.2f}%")

# Check graphite insert constraint
if hasattr(config, 'graphite_insert') and config.graphite_insert and config.graphite_insert.enabled:
    A_throat_change_pct = (A_throat[-1]/A_throat[0] - 1.0) * 100
    if abs(A_throat_change_pct) > 5.0:  # More than 5% change
        print(f"  [ERROR] Graphite insert enabled but throat area changed by {A_throat_change_pct:.2f}%!")
        print(f"  Graphite insert should keep throat area roughly constant (< 1% change)")
    else:
        print(f"  [OK] Throat area change ({A_throat_change_pct:.2f}%) is within graphite insert constraint")

print(f"Initial eps: {eps[0]:.4f}")
print(f"Final eps: {eps[-1]:.4f}")
print(f"eps change: {((eps[-1]/eps[0] - 1.0)*100):.2f}%")

# 3. Recession Rates
print("\n--- RECESSION RATES ---")
recession_throat = results.get("recession_throat", np.zeros_like(times))
recession_chamber = results.get("recession_chamber", np.zeros_like(times))

if len(recession_throat) > 1:
    recession_rate_throat = np.gradient(recession_throat, times)
    recession_rate_chamber = np.gradient(recession_chamber, times)
    
    print(f"Average throat recession rate: {np.mean(recession_rate_throat)*1e6:.3f} µm/s")
    print(f"Average chamber recession rate: {np.mean(recession_rate_chamber)*1e6:.3f} µm/s")
    
    if hasattr(config, 'graphite_insert') and config.graphite_insert and config.graphite_insert.enabled:
        max_throat_rate = np.max(recession_rate_throat) * 1e6
        if max_throat_rate > 0.1:  # > 0.1 µm/s
            print(f"  [ERROR] Throat recession rate ({max_throat_rate:.3f} µm/s) is too high for graphite!")
            print(f"  Graphite should have < 0.1 µm/s recession rate")
        else:
            print(f"  [OK] Throat recession rate ({max_throat_rate:.3f} µm/s) is within graphite limits")

# 4. Cf Validation
print("\n--- CF VALIDATION ---")
Cf_actual = results.get("Cf_actual", np.zeros_like(times))
Cf_ideal = results.get("Cf_ideal", np.zeros_like(times))

print(f"Initial Cf_actual: {Cf_actual[0]:.4f}")
print(f"Initial Cf_ideal: {Cf_ideal[0]:.4f}")
print(f"Initial ratio: {Cf_actual[0]/Cf_ideal[0]:.4f}")

# Check if Cf is reasonable over time
Cf_ratio = Cf_actual / Cf_ideal
if np.any(Cf_ratio > 2.0):
    print(f"  [WARNING] Cf_actual/Cf_ideal ratio exceeds 2.0 at some points")
    print(f"  Max ratio: {np.max(Cf_ratio):.4f}")

# 5. Stability Analysis (if available)
print("\n--- STABILITY ANALYSIS ---")
stability = results.get("stability", None)
if stability is not None:
    # Extract stability metrics if they're in the results
    print("  [INFO] Stability analysis results available")
else:
    print("  [INFO] Stability analysis not included in time-series results")

# 6. Turbulence Evolution
print("\n--- TURBULENCE EVOLUTION ---")
diagnostics_list = results.get("diagnostics", [])
if diagnostics_list and len(diagnostics_list) > 0:
    turbulence_intensity = []
    for diag in diagnostics_list:
        if isinstance(diag, dict):
            turb = diag.get("turbulence_intensity_mix", None)
            if turb is not None:
                turbulence_intensity.append(turb)
    
    if turbulence_intensity:
        turbulence_intensity = np.array(turbulence_intensity)
        print(f"Initial turbulence: {turbulence_intensity[0]:.4f}")
        print(f"Final turbulence: {turbulence_intensity[-1]:.4f}")
        print(f"Average turbulence: {np.mean(turbulence_intensity):.4f}")
    else:
        print("  [INFO] Turbulence intensity not found in diagnostics")
else:
    print("  [INFO] Diagnostics not available for turbulence analysis")

# 7. Mass Flow Validation
print("\n--- MASS FLOW VALIDATION ---")
mdot_total = results.get("mdot_total", np.zeros_like(times))
mdot_O = results.get("mdot_O", np.zeros_like(times))
mdot_F = results.get("mdot_F", np.zeros_like(times))

print(f"Initial mdot_total: {mdot_total[0]:.4f} kg/s")
print(f"Final mdot_total: {mdot_total[-1]:.4f} kg/s")
print(f"mdot change: {((mdot_total[-1]/mdot_total[0] - 1.0)*100):.2f}%")

# Mass flow should increase as throat area grows (for constant tank pressure)
if A_throat[-1] > A_throat[0]:
    expected_mdot_increase = (A_throat[-1]/A_throat[0] - 1.0) * 100
    actual_mdot_change = (mdot_total[-1]/mdot_total[0] - 1.0) * 100
    print(f"Expected mdot increase (from A_throat growth): {expected_mdot_increase:.2f}%")
    print(f"Actual mdot change: {actual_mdot_change:.2f}%")
    if abs(actual_mdot_change - expected_mdot_increase) > 10.0:
        print(f"  [WARNING] Mass flow change doesn't match throat area change")

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)




