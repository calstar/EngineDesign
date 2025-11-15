"""Test the new O/F-constrained inverse solver."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pintle_pipeline.io import load_config
from pintle_models.runner import PintleEngineRunner
from examples.pintle_engine.interactive_pipeline import solve_for_thrust_and_MR

# Load configuration
config_path = Path(__file__).parent / "config_minimal.yaml"
config = load_config(str(config_path))
runner = PintleEngineRunner(config)

print("=" * 80)
print("O/F-CONSTRAINED INVERSE SOLVER TEST")
print("=" * 80)

# Target performance
target_thrust_kN = 6.65
target_MR = 2.5  # Different from nominal ~2.36

print(f"\nTarget Performance:")
print(f"  Thrust: {target_thrust_kN:.2f} kN")
print(f"  O/F Ratio: {target_MR:.2f}")

print(f"\nSolving for tank pressures...")

try:
    (P_O_sol, P_F_sol), results, diagnostics = solve_for_thrust_and_MR(
        runner,
        target_thrust_kN,
        target_MR,
        initial_guess_psi=(1305.0, 974.0),
    )
    
    print(f"\n✅ CONVERGED in {diagnostics['iterations']} iterations!")
    print(f"\nRequired Tank Pressures:")
    print(f"  LOX:  {P_O_sol:.1f} psi")
    print(f"  Fuel: {P_F_sol:.1f} psi")
    
    print(f"\nAchieved Performance:")
    print(f"  Thrust: {diagnostics['final_thrust']:.2f} kN (error: {diagnostics['thrust_error_pct']:.3f}%)")
    print(f"  O/F:    {diagnostics['final_MR']:.3f} (error: {diagnostics['MR_error_pct']:.3f}%)")
    print(f"  Isp:    {results['Isp']:.1f} s")
    print(f"  Pc:     {results['Pc']/6894.76:.1f} psi")
    
    print(f"\nMass Flows:")
    print(f"  Oxidizer: {results['mdot_O']:.3f} kg/s")
    print(f"  Fuel:     {results['mdot_F']:.3f} kg/s")
    print(f"  Total:    {results['mdot_total']:.3f} kg/s")
    
    print(f"\n" + "=" * 80)
    print("CONVERGENCE HISTORY")
    print("=" * 80)
    print(f"{'Iter':<6} {'P_O [psi]':<12} {'P_F [psi]':<12} {'Thrust [kN]':<12} {'O/F':<8} {'F_err%':<10} {'MR_err%':<10}")
    print("-" * 80)
    
    for i in range(len(diagnostics['history']['thrust'])):
        print(
            f"{i:<6} "
            f"{diagnostics['history']['P_tank_O'][i]:<12.1f} "
            f"{diagnostics['history']['P_tank_F'][i]:<12.1f} "
            f"{diagnostics['history']['thrust'][i]:<12.2f} "
            f"{diagnostics['history']['MR'][i]:<8.3f} "
            f"{abs(diagnostics['history']['thrust_error'][i])*100:<10.3f} "
            f"{abs(diagnostics['history']['MR_error'][i])*100:<10.3f}"
        )
    
    print("\n✅ Test passed!")

except Exception as exc:
    print(f"\n❌ Test failed: {exc}")
    import traceback
    traceback.print_exc()

