#!/usr/bin/env python3
"""Test script for Layer 2 Pressure Curve Optimization.

This script allows you to test the Layer 2 optimizer with:
- A config YAML file
- Initial LOX and fuel pressures

Usage:
    python test_layer2.py <config.yaml> <initial_lox_pressure_psi> <initial_fuel_pressure_psi>
    
    Or edit the script to set parameters directly.

Example:
    python test_layer2.py config_minimal.yaml 500 500
"""

import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add project root to path
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from pintle_pipeline.io import load_config
from pintle_pipeline.config_schemas import PintleEngineConfig
from optimization_layers.layer2_pressure import run_layer2_pressure


def progress_callback(title: str, progress: float, message: str, logger: logging.Logger):
    """Simple progress callback for testing."""
    logger.info(f"[{progress*100:.1f}%] {title}: {message}")


def log_callback(title: str, message: str, logger: logging.Logger):
    """Simple log callback for testing."""
    logger.info(f"[LOG] {title}: {message}")


def plot_results(
    time_array: np.ndarray,
    P_tank_O: np.ndarray,
    P_tank_F: np.ndarray,
    summary: dict,
    output_file: str = None,
):
    """Plot pressure curves and summary metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Pressure curves
    ax1 = axes[0, 0]
    ax1.plot(time_array, P_tank_O / 1e6, 'b-', linewidth=2, label='LOX Tank')
    ax1.plot(time_array, P_tank_F / 1e6, 'r-', linewidth=2, label='Fuel Tank')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Tank Pressure [MPa]')
    ax1.set_title('Optimized Pressure Curves')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Pressure in PSI (more familiar units)
    ax2 = axes[0, 1]
    ax2.plot(time_array, P_tank_O / 6894.76, 'b-', linewidth=2, label='LOX Tank')
    ax2.plot(time_array, P_tank_F / 6894.76, 'r-', linewidth=2, label='Fuel Tank')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Tank Pressure [psi]')
    ax2.set_title('Optimized Pressure Curves (PSI)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Summary metrics (text)
    ax3 = axes[1, 0]
    ax3.axis('off')
    summary_text = f"""
Layer 2 Optimization Results
{'='*50}

Pressure Segments:
  LOX segments: {summary.get('lox_segments', 'N/A')}
  Fuel segments: {summary.get('fuel_segments', 'N/A')}

Initial Pressures:
  LOX: {summary.get('initial_lox_pressure_pa', 0)/6894.76:.1f} psi ({summary.get('initial_lox_pressure_pa', 0)/1e6:.2f} MPa)
  Fuel: {summary.get('initial_fuel_pressure_pa', 0)/6894.76:.1f} psi ({summary.get('initial_fuel_pressure_pa', 0)/1e6:.2f} MPa)

Final Pressures:
  LOX: {summary.get('lox_end_pressure_pa', 0)/6894.76:.1f} psi ({summary.get('lox_end_pressure_pa', 0)/1e6:.2f} MPa)
  Fuel: {summary.get('fuel_end_pressure_pa', 0)/6894.76:.1f} psi ({summary.get('fuel_end_pressure_pa', 0)/1e6:.2f} MPa)

Thrust:
  Peak/Initial: {summary.get('peak_thrust', 0):.1f} N
  Initial Actual: {summary.get('initial_thrust_actual', 0):.1f} N

Propellant Consumption:
  LOX: {summary.get('total_lox_mass_kg', 0):.3f} kg ({summary.get('lox_capacity_ratio', 0)*100:.1f}% of capacity)
  Fuel: {summary.get('total_fuel_mass_kg', 0):.3f} kg ({summary.get('fuel_capacity_ratio', 0)*100:.1f}% of capacity)
  Total: {summary.get('total_propellant_mass_kg', 0):.3f} kg

Impulse:
  Required: {summary.get('required_impulse', 0)/1000:.1f} kN·s
  Actual: {summary.get('total_impulse_actual', 0)/1000:.1f} kN·s
  Ratio: {summary.get('impulse_ratio', 0)*100:.1f}%

Burn Time: {summary.get('target_burn_time', 0):.2f} s
Time Points: {summary.get('n_time_points', 0)}
"""
    ax3.text(0.1, 0.5, summary_text, fontsize=10, family='monospace', 
             verticalalignment='center', transform=ax3.transAxes)
    
    # Plot 4: Pressure decay rate
    ax4 = axes[1, 1]
    if len(time_array) > 1:
        dt = np.diff(time_array)
        dP_O = -np.diff(P_tank_O) / dt  # Negative because pressure decreases
        dP_F = -np.diff(P_tank_F) / dt
        time_mid = (time_array[:-1] + time_array[1:]) / 2
        ax4.plot(time_mid, dP_O / 1e6, 'b-', linewidth=2, label='LOX Decay Rate')
        ax4.plot(time_mid, dP_F / 1e6, 'r-', linewidth=2, label='Fuel Decay Rate')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Pressure Decay Rate [MPa/s]')
        ax4.set_title('Pressure Decay Rate')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Test Layer 2 Pressure Curve Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with command line arguments
  python test_layer2.py config_minimal.yaml 500 500
  
  # With custom target apogee
  python test_layer2.py config_minimal.yaml 500 500 --target-apogee 30000
  
  # With target O/F ratio
  python test_layer2.py config_minimal.yaml 500 500 --target-of-ratio 2.5
  
  # With minimum stability margin
  python test_layer2.py config_minimal.yaml 500 500 --min-stability-margin 0.5
  
  # Save plot to file
  python test_layer2.py config_minimal.yaml 500 500 --plot-output results.png
        """
    )
    
    parser.add_argument(
        'config_file',
        type=str,
        help='Path to config YAML file'
    )
    
    parser.add_argument(
        'initial_lox_pressure_psi',
        type=float,
        help='Initial LOX tank pressure [psi]'
    )
    
    parser.add_argument(
        'initial_fuel_pressure_psi',
        type=float,
        help='Initial fuel tank pressure [psi]'
    )
    
    parser.add_argument(
        '--target-apogee',
        type=float,
        default=30000.0,
        help='Target apogee altitude [m] (default: 30000)'
    )
    
    parser.add_argument(
        '--peak-thrust',
        type=float,
        default=None,
        help='Peak/initial thrust target [N]. If not specified, uses config design_thrust'
    )
    
    parser.add_argument(
        '--plot-output',
        type=str,
        default=None,
        help='Save plot to file (e.g., results.png) instead of displaying'
    )
    
    parser.add_argument(
        '--min-pressure-psi',
        type=float,
        default=145.0,
        help='Minimum tank pressure [psi] (default: 145)'
    )
    
    parser.add_argument(
        '--n-time-points',
        type=int,
        default=200,
        help='Number of time points (default: 200)'
    )
    
    parser.add_argument(
        '--target-of-ratio',
        type=float,
        default=None,
        help='Target O/F (oxidizer-to-fuel) ratio for validation (optional)'
    )
    
    parser.add_argument(
        '--min-stability-margin',
        type=float,
        default=None,
        help='Minimum stability margin threshold (optional)'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save evaluation plots (overwrites layer2_evaluation_plot_*.png with each evaluation)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (default: test_layer2_YYYYMMDD_HHMMSS.log)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all terminal output (only log to file)'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if args.log_file:
        log_file_path = args.log_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"test_layer2_{timestamp}.log"
    
    # Create log directory if needed
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
        ]
    )
    
    # Also add console handler if not quiet
    if not args.quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger().addHandler(console_handler)
        print(f"Logging to: {log_file_path}")
    
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("Layer 2 Pressure Curve Optimization Test")
    logger.info("="*60)
    
    # Load config
    logger.info(f"Loading config from: {args.config_file}")
    try:
        config = load_config(args.config_file)
        logger.info("✓ Config loaded successfully")
    except Exception as e:
        logger.error(f"✗ Error loading config: {e}")
        return 1
    
    # Convert pressures from PSI to Pa
    initial_lox_pressure_pa = args.initial_lox_pressure_psi * 6894.76
    initial_fuel_pressure_pa = args.initial_fuel_pressure_psi * 6894.76
    min_pressure_pa = args.min_pressure_psi * 6894.76
    
    logger.info("")
    logger.info("Initial Pressures:")
    logger.info(f"  LOX: {args.initial_lox_pressure_psi:.1f} psi ({initial_lox_pressure_pa/1e6:.2f} MPa)")
    logger.info(f"  Fuel: {args.initial_fuel_pressure_psi:.1f} psi ({initial_fuel_pressure_pa/1e6:.2f} MPa)")
    
    # Extract parameters from config
    target_burn_time = config.thrust.burn_time if config.thrust else 10.0
    logger.info(f"")
    logger.info(f"Burn Time: {target_burn_time:.2f} s")
    
    # Get peak thrust
    if args.peak_thrust is not None:
        peak_thrust = args.peak_thrust
    else:
        peak_thrust = config.chamber.design_thrust if hasattr(config.chamber, 'design_thrust') else 7000.0
    logger.info(f"Peak Thrust Target: {peak_thrust:.1f} N")
    
    # Calculate rocket dry mass
    if config.rocket:
        rocket_dry_mass_kg = (
            config.rocket.airframe_mass +
            config.rocket.propulsion_dry_mass
        )
    else:
        # Fallback: estimate from tank structure masses if available
        rocket_dry_mass_kg = 100.0  # Default estimate
        logger.warning("⚠ Warning: No rocket config found, using default dry mass estimate")
    
    logger.info(f"Rocket Dry Mass: {rocket_dry_mass_kg:.2f} kg")
    
    # Get tank capacities
    if config.lox_tank and config.lox_tank.mass:
        max_lox_tank_capacity_kg = config.lox_tank.mass * 1.2  # 20% margin
    else:
        max_lox_tank_capacity_kg = 10.0  # Default
        logger.warning("⚠ Warning: No LOX tank config found, using default capacity")
    
    if config.fuel_tank and config.fuel_tank.mass:
        max_fuel_tank_capacity_kg = config.fuel_tank.mass * 1.2  # 20% margin
    else:
        max_fuel_tank_capacity_kg = 15.0  # Default
        logger.warning("⚠ Warning: No fuel tank config found, using default capacity")
    
    logger.info(f"Max LOX Tank Capacity: {max_lox_tank_capacity_kg:.2f} kg")
    logger.info(f"Max Fuel Tank Capacity: {max_fuel_tank_capacity_kg:.2f} kg")
    logger.info(f"Target Apogee: {args.target_apogee:.0f} m")
    if args.target_of_ratio is not None:
        logger.info(f"Target O/F Ratio: {args.target_of_ratio:.2f}")
    if args.min_stability_margin is not None:
        logger.info(f"Min Stability Margin: {args.min_stability_margin:.3f}")
    
    # Run Layer 2 optimization
    logger.info("")
    logger.info("="*60)
    logger.info("Running Layer 2 Pressure Curve Optimization...")
    logger.info("="*60)
    
    # Create wrapped callbacks that include logger
    def progress_wrapper(title: str, progress: float, message: str):
        progress_callback(title, progress, message, logger)
    
    def log_wrapper(title: str, message: str):
        log_callback(title, message, logger)
    
    try:
        optimized_config, time_array, P_tank_O, P_tank_F, summary, success = run_layer2_pressure(
            optimized_config=config,
            initial_lox_pressure_pa=initial_lox_pressure_pa,
            initial_fuel_pressure_pa=initial_fuel_pressure_pa,
            peak_thrust=peak_thrust,
            target_apogee_m=args.target_apogee,
            rocket_dry_mass_kg=rocket_dry_mass_kg,
            max_lox_tank_capacity_kg=max_lox_tank_capacity_kg,
            max_fuel_tank_capacity_kg=max_fuel_tank_capacity_kg,
            target_burn_time=target_burn_time,
            n_time_points=args.n_time_points,
            update_progress=progress_wrapper,
            log_status=log_wrapper,
            min_pressure_pa=min_pressure_pa,
            optimal_of_ratio=args.target_of_ratio,
            min_stability_margin=args.min_stability_margin,
            save_evaluation_plots=args.save_plots,
        )
        
        logger.info("")
        logger.info("="*60)
        if success:
            logger.info("✓ Layer 2 Optimization Completed Successfully!")
        else:
            logger.warning("⚠ Layer 2 Optimization Completed with Warnings")
        logger.info("="*60)
        
        # Print summary
        logger.info("")
        logger.info("Summary:")
        logger.info(f"  Success: {success}")
        logger.info(f"  LOX Segments: {summary.get('lox_segments', 'N/A')}")
        logger.info(f"  Fuel Segments: {summary.get('fuel_segments', 'N/A')}")
        logger.info(f"  Total Impulse: {summary.get('total_impulse_actual', 0)/1000:.1f} kN·s")
        logger.info(f"  Required Impulse: {summary.get('required_impulse', 0)/1000:.1f} kN·s")
        logger.info(f"  Impulse Ratio: {summary.get('impulse_ratio', 0)*100:.1f}%")
        logger.info(f"  Total Propellant: {summary.get('total_propellant_mass_kg', 0):.3f} kg")
        
        # Plot results
        if args.plot_output or True:  # Always plot
            plot_results(
                time_array,
                P_tank_O,
                P_tank_F,
                summary,
                output_file=args.plot_output,
            )
            if args.plot_output:
                logger.info(f"Plot saved to: {args.plot_output}")
        
        logger.info("")
        logger.info("="*60)
        logger.info(f"Test completed. Full log saved to: {log_file_path}")
        logger.info("="*60)
        
        return 0
        
    except Exception as e:
        logger.error("")
        logger.error(f"✗ Error during optimization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("")
        logger.error(f"Full error log saved to: {log_file_path}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

