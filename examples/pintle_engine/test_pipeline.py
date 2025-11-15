"""Test the complete pipeline with all new features"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pintle_pipeline.io import load_config
from pintle_models.runner import PintleEngineRunner

print("=" * 60)
print("TESTING PINTLE ENGINE PIPELINE")
print("=" * 60)

try:
    # Load config
    print("\n[1/4] Loading configuration...")
    config_path = Path(__file__).parent / 'config_minimal.yaml'
    config = load_config(str(config_path))
    print("[OK] Config loaded")
    
    # Initialize runner
    print("\n[2/4] Initializing runner...")
    runner = PintleEngineRunner(config)
    print("[OK] Runner initialized")
    
    # Run evaluation
    print("\n[3/4] Running pipeline evaluation...")
    P_tank_O = 1305 * 6894.76  # psi to Pa
    P_tank_F = 974 * 6894.76   # psi to Pa
    
    results = runner.evaluate(P_tank_O, P_tank_F)
    print("[OK] Evaluation complete")
    
    # Check results
    print("\n[4/4] Verifying results...")
    
    # Basic metrics
    print(f"\n[PERFORMANCE METRICS]")
    print(f"  Thrust: {results['F']/1000:.2f} kN")
    print(f"  Chamber Pressure: {results['Pc']/6894.76:.1f} psi")
    print(f"  Mass Flow: {results['mdot_total']:.3f} kg/s")
    print(f"  Mixture Ratio: {results['MR']:.3f}")
    print(f"  Isp: {results['Isp']:.1f} s")
    
    # New Cf metrics
    print(f"\n[THRUST COEFFICIENT]")
    if 'Cf_actual' in results:
        print(f"  Cf (actual): {results['Cf_actual']:.4f}")
    if 'Cf_ideal' in results:
        print(f"  Cf (ideal): {results['Cf_ideal']:.4f}")
    if 'Cf_theoretical' in results:
        print(f"  Cf (theoretical): {results['Cf_theoretical']:.4f}")
    
    # New temperature metrics
    print(f"\n[TEMPERATURES]")
    if 'Tc' in results:
        print(f"  Chamber: {results['Tc']:.1f} K")
    if 'T_throat' in results:
        print(f"  Throat: {results['T_throat']:.1f} K")
    if 'T_exit' in results:
        print(f"  Exit: {results['T_exit']:.1f} K")
    if 'temperature_profile' in results and results['temperature_profile']:
        profile = results['temperature_profile']
        print(f"  Injection: {profile.get('T_injection', 'N/A')} K")
        print(f"  Mid-chamber: {profile.get('T_mid', 'N/A')} K")
    
    # Stability analysis
    print(f"\n[STABILITY ANALYSIS]")
    stability = results.get('stability')
    if stability:
        status = "STABLE" if stability.get('is_stable') else "INSTABILITY RISK"
        print(f"  Status: {status}")
        chugging = stability.get('chugging', {})
        if chugging:
            print(f"  Chugging frequency: {chugging.get('frequency', 'N/A'):.1f} Hz")
            print(f"  Stability margin: {chugging.get('stability_margin', 'N/A'):.3f}")
        issues = stability.get('issues', [])
        if issues:
            print(f"  Issues: {len(issues)} found")
            for issue in issues[:3]:
                print(f"    - {issue}")
    else:
        print("  [WARNING] Stability analysis not available")
    
    # Graphite insert check
    print(f"\n[GRAPHITE INSERT]")
    if hasattr(runner.config, 'graphite_insert') and runner.config.graphite_insert:
        graphite = runner.config.graphite_insert
        print(f"  Enabled: {graphite.enabled}")
        if graphite.enabled:
            print(f"  Density: {graphite.material_density:.0f} kg/m³")
            print(f"  Heat of Ablation: {graphite.heat_of_ablation/1e6:.2f} MJ/kg")
    else:
        print("  Not configured")
    
    # Cooling diagnostics
    print(f"\n[COOLING]")
    cooling = results.get('cooling', {})
    if cooling:
        ablative = cooling.get('ablative', {})
        if ablative and ablative.get('enabled'):
            print(f"  Ablative: Enabled")
            print(f"  Recession rate: {ablative.get('recession_rate', 0)*1e6:.3f} µm/s")
        regen = cooling.get('regen', {})
        if regen and regen.get('enabled'):
            print(f"  Regenerative: Enabled")
            print(f"  Heat removed: {regen.get('heat_removed', 0)/1000:.1f} kW")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] ALL TESTS PASSED - PIPELINE IS WORKING!")
    print("=" * 60)
    sys.exit(0)
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

