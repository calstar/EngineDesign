"""Interactive CLI for the pintle engine pipeline.

Features:
- Forward mode: user provides tank pressures (psi) and receives performance outputs
- Inverse mode: user provides target thrust (kN) and pipeline solves required tank pressures

This script uses the existing PintleEngineRunner to evaluate the pipeline.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pintle_pipeline.io import load_config
from pintle_models.runner import PintleEngineRunner

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

PSI_TO_PA = 6894.76
PA_TO_PSI = 1.0 / PSI_TO_PA

CONFIG_PATH = Path(__file__).parent / "config_minimal.yaml"


def load_runner() -> PintleEngineRunner:
    """Load configuration and create a runner instance."""
    config = load_config(str(CONFIG_PATH))
    return PintleEngineRunner(config)


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def format_value(value: float, unit: str, precision: int = 2) -> str:
    return f"{value:.{precision}f} {unit}"


def summarize_results(results: Dict[str, Any]) -> None:
    """Pretty-print the main outputs from a pipeline evaluation."""
    Pc_psi = results["Pc"] * PA_TO_PSI
    thrust_kN = results["F"] / 1000.0
    mdot_total = results["mdot_total"]
    mdot_O = results["mdot_O"]
    mdot_F = results["mdot_F"]
    MR = results["MR"]
    Isp = results["Isp"]
    cstar = results["cstar_actual"]
    v_exit = results["v_exit"]
    P_exit_psi = results["P_exit"] * PA_TO_PSI

    print("Performance Summary:")
    print(f"  Thrust             : {format_value(thrust_kN, 'kN')}")
    print(f"  Specific Impulse   : {format_value(Isp, 's', 1)}")
    print(f"  Chamber Pressure   : {format_value(Pc_psi, 'psi', 1)}")
    print(f"  Total Mass Flow    : {format_value(mdot_total, 'kg/s', 3)}")
    print(f"    - Oxidizer       : {format_value(mdot_O, 'kg/s', 3)}")
    print(f"    - Fuel           : {format_value(mdot_F, 'kg/s', 3)}")
    print(f"  Mixture Ratio (O/F): {MR:.3f}")
    print(f"  c* (actual)        : {format_value(cstar, 'm/s', 1)}")
    print(f"  Exit Velocity      : {format_value(v_exit, 'm/s', 1)}")
    print(f"  Exit Pressure      : {format_value(P_exit_psi, 'psi', 2)}")


# -----------------------------------------------------------------------------
# Forward evaluation (tank pressures -> performance)
# -----------------------------------------------------------------------------

def forward_mode(runner: PintleEngineRunner) -> None:
    print_header("FORWARD MODE: Tank Pressures -> Performance")

    try:
        P_tank_O_psi = float(input("Enter LOX tank pressure [psi]: ").strip())
        P_tank_F_psi = float(input("Enter Fuel tank pressure [psi]: ").strip())
    except ValueError:
        print("Invalid numeric input. Returning to menu.")
        return

    if P_tank_O_psi <= 0 or P_tank_F_psi <= 0:
        print("Tank pressures must be positive. Returning to menu.")
        return

    P_tank_O = P_tank_O_psi * PSI_TO_PA
    P_tank_F = P_tank_F_psi * PSI_TO_PA

    try:
        results = runner.evaluate(P_tank_O, P_tank_F)
    except Exception as exc:
        print(f"Pipeline evaluation failed: {exc}")
        return

    print_header("RESULTS")
    summarize_results(results)


# -----------------------------------------------------------------------------
# Inverse evaluation (target thrust -> tank pressures)
# -----------------------------------------------------------------------------

def _thrust_difference(
    scale: float,
    runner: PintleEngineRunner,
    base_pressures: Tuple[float, float],
    target_thrust_kN: float,
) -> float:
    """Return thrust difference at given scale factor."""
    P_tank_O_base, P_tank_F_base = base_pressures
    P_tank_O = scale * P_tank_O_base
    P_tank_F = scale * P_tank_F_base
    results = runner.evaluate(P_tank_O, P_tank_F)
    thrust_kN = results["F"] / 1000.0
    return thrust_kN - target_thrust_kN


def solve_for_thrust(
    runner: PintleEngineRunner,
    target_thrust_kN: float,
    base_pressures_psi: Tuple[float, float],
    scale_bounds: Tuple[float, float] = (0.2, 5.0),
    max_expand: int = 10,
) -> Tuple[Tuple[float, float], Dict[str, Any]]:
    """Solve for tank pressures that achieve the target thrust.

    The solution scales the baseline tank pressures by a factor 'scale'.
    """

    base_O_psi, base_F_psi = base_pressures_psi
    if base_O_psi <= 0 or base_F_psi <= 0:
        raise ValueError("Baseline tank pressures must be positive.")

    base_pressures_pa = (base_O_psi * PSI_TO_PA, base_F_psi * PSI_TO_PA)

    # Evaluate baseline to understand where we stand relative to target
    baseline_results = runner.evaluate(*base_pressures_pa)
    baseline_thrust = baseline_results["F"] / 1000.0

    low, high = scale_bounds
    f_low = _thrust_difference(low, runner, base_pressures_pa, target_thrust_kN)
    f_high = _thrust_difference(high, runner, base_pressures_pa, target_thrust_kN)

    # Expand bounds until we bracket the root
    iterations = 0
    while f_low * f_high > 0 and iterations < max_expand:
        if target_thrust_kN > baseline_thrust:
            high *= 1.5
            f_high = _thrust_difference(high, runner, base_pressures_pa, target_thrust_kN)
        else:
            low *= 0.5
            f_low = _thrust_difference(low, runner, base_pressures_pa, target_thrust_kN)
        iterations += 1

    if f_low * f_high > 0:
        raise RuntimeError(
            "Could not bracket solution. Try adjusting target thrust or baseline pressures."
        )

    scale = brentq(
        _thrust_difference,
        low,
        high,
        args=(runner, base_pressures_pa, target_thrust_kN),
        xtol=1e-4,
        rtol=1e-4,
        maxiter=100,
    )

    # Evaluate at solution
    P_tank_O_solution = scale * base_pressures_pa[0]
    P_tank_F_solution = scale * base_pressures_pa[1]
    results = runner.evaluate(P_tank_O_solution, P_tank_F_solution)

    return (P_tank_O_solution, P_tank_F_solution), results


def inverse_mode(runner: PintleEngineRunner) -> None:
    print_header("INVERSE MODE: Target Thrust -> Tank Pressures")
    try:
        target_thrust_kN = float(input("Enter desired thrust [kN]: ").strip())
    except ValueError:
        print("Invalid thrust input. Returning to menu.")
        return

    if target_thrust_kN <= 0:
        print("Target thrust must be positive. Returning to menu.")
        return

    # Baseline pressures (psi)
    try:
        base_O_psi = float(
            input("Enter baseline LOX tank pressure [psi] (default 1305): ").strip() or "1305"
        )
        base_F_psi = float(
            input("Enter baseline Fuel tank pressure [psi] (default 974): ").strip() or "974"
        )
    except ValueError:
        print("Invalid baseline pressures. Returning to menu.")
        return

    try:
        (P_tank_O_solution, P_tank_F_solution), results = solve_for_thrust(
            runner,
            target_thrust_kN,
            (base_O_psi, base_F_psi),
        )
    except Exception as exc:
        print(f"Failed to find tank pressures for target thrust: {exc}")
        return

    P_tank_O_psi = P_tank_O_solution * PA_TO_PSI
    P_tank_F_psi = P_tank_F_solution * PA_TO_PSI

    print_header("SOLUTION")
    print(f"Required LOX tank pressure : {format_value(P_tank_O_psi, 'psi', 1)}")
    print(f"Required Fuel tank pressure: {format_value(P_tank_F_psi, 'psi', 1)}")
    summarize_results(results)


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def interactive_loop():
    runner = load_runner()
    print_header("PINTLE ENGINE PIPELINE - INTERACTIVE CLI")
    print(f"Configuration: {CONFIG_PATH}")

    while True:
        print("\nSelect an option:")
        print("  1) Forward mode  (tank pressures -> performance)")
        print("  2) Inverse mode  (target thrust   -> tank pressures)")
        print("  q) Quit")
        choice = input("Enter choice: ").strip().lower()

        if choice == "1":
            forward_mode(runner)
        elif choice == "2":
            inverse_mode(runner)
        elif choice in {"q", "quit", "exit"}:
            print("Exiting interactive CLI.")
            break
        else:
            print("Invalid choice. Please select 1, 2, or q.")


if __name__ == "__main__":
    try:
        interactive_loop()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
