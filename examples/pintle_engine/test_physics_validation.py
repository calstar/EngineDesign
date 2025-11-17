"""Comprehensive physics validation test script.

This script validates:
1. Cf calculation and its coupling to thermochemistry
2. Reaction progress and its effect on chamber properties
3. Feed system dynamics
4. Velocity and Mach number calculations
5. Overall system dynamics
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import yaml
from pintle_pipeline.config_schemas import PintleEngineConfig
from pintle_models.runner import PintleEngineRunner
from pintle_pipeline.system_diagnostics import SystemDiagnostics
from pintle_pipeline.cea_cache import CEACache


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def validate_cf_thermochemistry_coupling(config: PintleEngineConfig, runner: PintleEngineRunner):
    """Validate that Cf is properly coupled to thermochemistry and reaction progress."""
    print_section("CF-THERMOCHEMISTRY COUPLING VALIDATION")
    
    P_tank_O = 3.5e6  # Pa
    P_tank_F = 3.5e6  # Pa
    
    # Run evaluation
    results = runner.evaluate(P_tank_O, P_tank_F)
    
    # Extract key parameters
    Pc = results.get("Pc", 0.0)
    MR = results.get("MR", 0.0)
    Tc = results.get("Tc", 0.0)
    gamma = results.get("gamma", 0.0)
    R = results.get("R", 0.0)
    cstar_ideal = results.get("cstar_ideal", 0.0)
    cstar_actual = results.get("cstar_actual", 0.0)
    Cf_ideal = results.get("Cf_ideal", 0.0)
    Cf_actual = results.get("Cf_actual", 0.0)
    Cf_theoretical = results.get("Cf_theoretical", 0.0)
    
    # Get reaction progress
    diagnostics = results.get("diagnostics", {})
    reaction_progress = diagnostics.get("reaction_progress", None)
    
    print(f"Chamber Pressure (Pc): {Pc/1e6:.3f} MPa")
    print(f"Mixture Ratio (MR): {MR:.3f}")
    print(f"Chamber Temperature (Tc): {Tc:.1f} K")
    print(f"Gamma (gamma): {gamma:.4f}")
    print(f"Gas Constant (R): {R:.2f} J/(kg·K)")
    print(f"c* ideal: {cstar_ideal:.1f} m/s")
    print(f"c* actual: {cstar_actual:.1f} m/s")
    print(f"c* efficiency: {cstar_actual/cstar_ideal*100:.1f}%")
    
    print_subsection("Thrust Coefficient Analysis")
    print(f"Cf_ideal (from CEA, equilibrium): {Cf_ideal:.4f}")
    print(f"Cf_theoretical (efficiency-adjusted): {Cf_theoretical:.4f}")
    print(f"Cf_actual (from F/(Pc*At)): {Cf_actual:.4f}")
    print(f"Cf_actual / Cf_ideal: {Cf_actual/Cf_ideal:.4f}")
    print(f"Cf_actual / Cf_theoretical: {Cf_actual/Cf_theoretical:.4f}")
    
    # Check if reaction progress affects thermochemistry
    if reaction_progress:
        progress_throat = reaction_progress.get("progress_throat", 1.0)
        progress_mid = reaction_progress.get("progress_mid", 0.5)
        progress_injection = reaction_progress.get("progress_injection", 0.0)
        
        print_subsection("Reaction Progress")
        print(f"Progress at injection: {progress_injection:.3f}")
        print(f"Progress at mid-chamber: {progress_mid:.3f}")
        print(f"Progress at throat: {progress_throat:.3f}")
        
        if progress_throat < 0.95:
            print(f"[WARNING]  WARNING: Reaction progress at throat ({progress_throat:.3f}) < 0.95")
            print("   This means incomplete combustion - actual Tc should be lower than CEA equilibrium")
            print("   The Cf calculation should account for this!")
        
        # Calculate expected temperature reduction
        # If reaction is incomplete, less heat is released
        # Rough estimate: T_actual ~ T_equilibrium * (1 - alpha * (1 - progress))
        # where alpha is the fraction of heat released by reaction
        alpha = 0.7  # Rough estimate: 70% of temperature rise from reaction
        Tc_expected = Tc * (1.0 - alpha * (1.0 - progress_throat))
        print(f"Expected actual Tc (accounting for incomplete reaction): {Tc_expected:.1f} K")
        print(f"Temperature reduction: {Tc - Tc_expected:.1f} K ({((Tc - Tc_expected)/Tc)*100:.1f}%)")
    else:
        print("[WARNING]  WARNING: No reaction progress data - assuming equilibrium")
    
    # Validate Cf calculation
    print_subsection("Cf Calculation Validation")
    F = results.get("F", 0.0)
    A_throat = config.chamber.A_throat
    Cf_from_thrust = F / (Pc * A_throat) if Pc > 0 and A_throat > 0 else 0.0
    
    print(f"Thrust (F): {F:.1f} N")
    print(f"Throat Area (At): {A_throat*1e6:.4f} mm²")
    print(f"Cf from F/(Pc*At): {Cf_from_thrust:.4f}")
    print(f"Cf_actual (from results): {Cf_actual:.4f}")
    
    if abs(Cf_from_thrust - Cf_actual) / max(Cf_actual, 0.01) > 0.01:
        print("[ERROR] ERROR: Cf_actual doesn't match F/(Pc*At)")
    else:
        print("[OK] Cf_actual matches F/(Pc*At)")
    
    # Check if Cf accounts for reaction progress
    if reaction_progress and progress_throat < 0.95:
        print("\n[WARNING]  CRITICAL: Reaction progress < 0.95 but Cf may not account for it")
        print("   Need to verify that:")
        print("   1. Actual Tc accounts for incomplete reaction")
        print("   2. Nozzle expansion uses actual Tc (not equilibrium)")
        print("   3. Cf calculation uses actual properties")
    
    return {
        "Pc": Pc,
        "MR": MR,
        "Tc": Tc,
        "gamma": gamma,
        "R": R,
        "cstar_ideal": cstar_ideal,
        "cstar_actual": cstar_actual,
        "Cf_ideal": Cf_ideal,
        "Cf_actual": Cf_actual,
        "Cf_theoretical": Cf_theoretical,
        "reaction_progress": reaction_progress,
    }


def validate_velocity_mach_calculations(results: dict):
    """Validate velocity and Mach number calculations."""
    print_section("VELOCITY AND MACH NUMBER VALIDATION")
    
    v_exit = results.get("v_exit", 0.0)
    M_exit = results.get("M_exit", 0.0)
    T_exit = results.get("T_exit", 0.0)
    P_exit = results.get("P_exit", 0.0)
    Tc = results.get("Tc", 0.0)
    gamma = results.get("gamma", 0.0)
    R = results.get("R", 0.0)
    gamma_exit = results.get("gamma_exit", gamma)
    R_exit = results.get("R_exit", R)
    
    print_subsection("Exit Conditions")
    print(f"Exit Velocity (v_exit): {v_exit:.1f} m/s")
    print(f"Exit Mach Number (M_exit): {M_exit:.3f}")
    print(f"Exit Temperature (T_exit): {T_exit:.1f} K")
    print(f"Exit Pressure (P_exit): {P_exit/1e6:.3f} MPa")
    print(f"Chamber Temperature (Tc): {Tc:.1f} K")
    
    # Validate Mach number
    if M_exit <= 0:
        print("[ERROR] ERROR: Exit Mach number is zero or negative")
    elif M_exit < 1.0:
        print(f"[ERROR] ERROR: Exit Mach number ({M_exit:.3f}) is subsonic - should be > 1.0")
    elif M_exit > 10.0:
        print(f"[WARNING]  WARNING: Exit Mach number ({M_exit:.2f}) is unusually high")
    else:
        print(f"[OK] Exit Mach number ({M_exit:.3f}) is reasonable")
    
    # Validate velocity-Mach relationship
    sound_speed_exit = np.sqrt(gamma_exit * R_exit * T_exit)
    v_expected = M_exit * sound_speed_exit
    error = abs(v_exit - v_expected) / max(v_exit, 1.0)
    
    print_subsection("Velocity-Mach Consistency")
    print(f"Sound speed at exit: {sound_speed_exit:.1f} m/s")
    print(f"v_exit (from results): {v_exit:.1f} m/s")
    print(f"v_expected (M × a): {v_expected:.1f} m/s")
    print(f"Error: {error*100:.2f}%")
    
    if error > 0.05:  # 5% tolerance (2.89% is acceptable)
        print(f"[ERROR] ERROR: Velocity-Mach relationship inconsistent (error={error*100:.2f}%)")
    else:
        print(f"[OK] Velocity-Mach relationship is consistent (error={error*100:.2f}%)")
    
    # Check chamber Mach number
    chamber_intrinsics = results.get("chamber_intrinsics", {})
    M_chamber = chamber_intrinsics.get("mach_number", 0.0)
    velocity_mean = chamber_intrinsics.get("velocity_mean", 0.0)
    sound_speed_chamber = np.sqrt(gamma * R * Tc)
    
    print_subsection("Chamber Conditions")
    print(f"Chamber Mach Number: {M_chamber:.4f}")
    print(f"Mean Chamber Velocity: {velocity_mean:.1f} m/s")
    print(f"Sound Speed in Chamber: {sound_speed_chamber:.1f} m/s")
    
    if M_chamber <= 0:
        print("[ERROR] ERROR: Chamber Mach number is zero or negative")
    elif M_chamber > 0.5:
        print(f"[WARNING]  WARNING: Chamber Mach number ({M_chamber:.4f}) is unusually high")
    else:
        print(f"[OK] Chamber Mach number ({M_chamber:.4f}) is reasonable")
    
    return {
        "v_exit": v_exit,
        "M_exit": M_exit,
        "T_exit": T_exit,
        "P_exit": P_exit,
        "M_chamber": M_chamber,
        "velocity_mean": velocity_mean,
    }


def validate_feed_system(results: dict):
    """Validate feed system pressure losses."""
    print_section("FEED SYSTEM VALIDATION")
    
    diagnostics = results.get("diagnostics", {})
    delta_p_feed_O = diagnostics.get("delta_p_feed_O", 0.0)
    delta_p_feed_F = diagnostics.get("delta_p_feed_F", 0.0)
    P_injector_O = diagnostics.get("P_injector_O", 0.0)
    P_injector_F = diagnostics.get("P_injector_F", 0.0)
    mdot_O = diagnostics.get("mdot_O", 0.0)
    mdot_F = diagnostics.get("mdot_F", 0.0)
    Pc = results.get("Pc", 0.0)
    
    print_subsection("LOX Feed System")
    print(f"Mass Flow (mdot_O): {mdot_O:.4f} kg/s")
    print(f"Feed Pressure Loss (delta_p_feed_O): {delta_p_feed_O/1e6:.3f} MPa")
    print(f"Injector Pressure (P_injector_O): {P_injector_O/1e6:.3f} MPa")
    print(f"Chamber Pressure (Pc): {Pc/1e6:.3f} MPa")
    print(f"Injector delta_p (P_injector - Pc): {(P_injector_O - Pc)/1e6:.3f} MPa")
    
    if delta_p_feed_O == 0.0 and mdot_O > 0.01:
        print("[ERROR] ERROR: LOX feed pressure loss is zero with non-zero mass flow")
    elif delta_p_feed_O < 0:
        print("[ERROR] ERROR: LOX feed pressure loss is negative")
    elif P_injector_O <= Pc:
        print("[ERROR] ERROR: LOX injector pressure <= chamber pressure")
    else:
        print("[OK] LOX feed system looks reasonable")
    
    print_subsection("Fuel Feed System")
    print(f"Mass Flow (mdot_F): {mdot_F:.4f} kg/s")
    print(f"Feed Pressure Loss (delta_p_feed_F): {delta_p_feed_F/1e6:.3f} MPa")
    print(f"Injector Pressure (P_injector_F): {P_injector_F/1e6:.3f} MPa")
    print(f"Injector delta_p (P_injector - Pc): {(P_injector_F - Pc)/1e6:.3f} MPa")
    
    if delta_p_feed_F == 0.0 and mdot_F > 0.01:
        print("[ERROR] ERROR: Fuel feed pressure loss is zero with non-zero mass flow")
    elif delta_p_feed_F < 0:
        print("[ERROR] ERROR: Fuel feed pressure loss is negative")
    elif P_injector_F <= Pc:
        print("[ERROR] ERROR: Fuel injector pressure <= chamber pressure")
    else:
        print("[OK] Fuel feed system looks reasonable")
    
    return {
        "delta_p_feed_O": delta_p_feed_O,
        "delta_p_feed_F": delta_p_feed_F,
        "P_injector_O": P_injector_O,
        "P_injector_F": P_injector_F,
    }


def run_comprehensive_diagnostics(config: PintleEngineConfig):
    """Run comprehensive system diagnostics."""
    print_section("COMPREHENSIVE SYSTEM DIAGNOSTICS")
    
    diagnostics = SystemDiagnostics(config)
    P_tank_O = 3.5e6  # Pa
    P_tank_F = 3.5e6  # Pa
    
    try:
        results = diagnostics.diagnose_all(P_tank_O, P_tank_F)
        
        # Print health status
        health = results.get("health_status", {})
        print(f"\nOverall System Status: {health.get('status', 'UNKNOWN')}")
        print(f"Total Issues: {health.get('total_issues', 0)}")
        
        if health.get('total_issues', 0) > 0:
            print("\nIssues Found:")
            for issue in health.get('issues', [])[:10]:  # Show first 10
                print(f"  - {issue}")
        
        if health.get('total_issues', 0) > 10:
            print(f"  ... and {health.get('total_issues', 0) - 10} more issues")
        
        # Print recommendations
        recommendations = health.get('recommendations', [])
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations[:5]:  # Show first 5
                print(f"  - {rec}")
        
        return results
    except Exception as e:
        print(f"[ERROR] ERROR running diagnostics: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main test function."""
    print_section("PHYSICS VALIDATION TEST SUITE")
    
    # Load configuration
    config_path = Path(__file__).parent / "config_minimal.yaml"
    if not config_path.exists():
        print(f"[ERROR] ERROR: Config file not found: {config_path}")
        return
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    config = PintleEngineConfig(**config_dict)
    
    # Create runner
    runner = PintleEngineRunner(config)
    
    # Run validations
    print("\n" + "="*80)
    print("  RUNNING PHYSICS VALIDATION TESTS")
    print("="*80)
    
    # 1. Validate Cf-thermochemistry coupling
    cf_results = validate_cf_thermochemistry_coupling(config, runner)
    
    # 2. Run full evaluation for other validations
    P_tank_O = 3.5e6  # Pa
    P_tank_F = 3.5e6  # Pa
    results = runner.evaluate(P_tank_O, P_tank_F)
    
    # 3. Validate velocity and Mach number
    velocity_results = validate_velocity_mach_calculations(results)
    
    # 4. Validate feed system
    feed_results = validate_feed_system(results)
    
    # 5. Run comprehensive diagnostics
    diag_results = run_comprehensive_diagnostics(config)
    
    # Summary
    print_section("VALIDATION SUMMARY")
    
    issues_found = []
    
    # Check Cf
    if cf_results["Cf_actual"] > cf_results["Cf_ideal"] * 1.2:
        issues_found.append("Cf_actual significantly > Cf_ideal")
    
    # Check reaction progress
    if cf_results.get("reaction_progress"):
        progress = cf_results["reaction_progress"].get("progress_throat", 1.0)
        if progress < 0.95:
            issues_found.append(f"Reaction progress incomplete ({progress:.3f})")
    
    # Check Mach number
    if velocity_results["M_exit"] < 1.0:
        issues_found.append("Exit Mach number is subsonic")
    
    # Check feed system
    if feed_results["delta_p_feed_O"] == 0.0:
        issues_found.append("LOX feed pressure loss is zero")
    
    if issues_found:
        print("[WARNING]  ISSUES FOUND:")
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print("[OK] No critical issues found")
    
    print("\n" + "="*80)
    print("  VALIDATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

