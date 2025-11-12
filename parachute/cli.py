"""Command line entrypoints"""

import argparse
from pathlib import Path
from .model import load_config, System
from .engine import simulate, SimulationState
from .observers import write_telemetry_csv, write_events_csv, write_peaks_csv, detect_peaks


def create_initial_state(system: System) -> SimulationState:
    """Create initial simulation state from system config"""
    bodies = {bid: body for bid, body in system.bodies.items()}
    canopies = {cid: canopy for cid, canopy in system.canopies.items()}
    return SimulationState(bodies=bodies, canopies=canopies, t=0.0)


def cmd_simulate(args):
    """Simulate command"""
    # Load config
    system = load_config(args.config)
    
    # Create initial state
    state0 = create_initial_state(system)
    
    # Run simulation
    print(f"Running simulation from t={args.t0} to t={args.tf} with dt={args.dt}")
    states, events = simulate(
        system=system,
        t0=args.t0,
        tf=args.tf,
        dt=args.dt,
        state0=state0,
        ramp_shape=args.ramp_shape,
        enable_pickup_shrink=not args.no_pickup_shrink
    )
    
    print(f"Simulation complete: {len(states)} states, {len(events)} events")
    
    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write outputs
    telemetry_path = output_dir / "telemetry.csv"
    events_path = output_dir / "events.csv"
    peaks_path = output_dir / "peaks.csv"
    
    print(f"Writing telemetry to {telemetry_path}")
    write_telemetry_csv(states, system, telemetry_path, ramp_shape=args.ramp_shape, verbose=args.verbose, events=events)
    
    print(f"Writing events to {events_path}")
    write_events_csv(events, events_path)
    
    print(f"Detecting peaks...")
    peaks = detect_peaks(states, system, events, peak_window=args.peak_window)
    print(f"Writing peaks to {peaks_path}")
    write_peaks_csv(peaks, peaks_path)
    
    print(f"Done! Results in {output_dir}")


def cmd_trim(args):
    """Trim command - analyze existing simulation for trim state"""
    # TODO: Implement trim analysis
    print("Trim command not yet implemented")
    return


def main():
    parser = argparse.ArgumentParser(description="N-Canopy Parachute Dynamics Engine")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Run simulation')
    sim_parser.add_argument('--config', required=True, help='Config YAML file')
    sim_parser.add_argument('-t0', '--t0', type=float, default=0.0, help='Start time')
    sim_parser.add_argument('--tf', type=float, required=True, help='End time')
    sim_parser.add_argument('-dt', '--dt', type=float, default=0.01, help='Time step')
    sim_parser.add_argument('--out', required=True, help='Output directory')
    sim_parser.add_argument('--ramp-shape', choices=['exp', 'tanh'], default='exp', help='Reefing ramp shape')
    sim_parser.add_argument('--no-pickup-shrink', action='store_true', help='Disable pickup step shrinking')
    sim_parser.add_argument('--verbose', action='store_true', help='Include position/velocity columns')
    sim_parser.add_argument('--peak-window', type=float, default=5.0, help='Peak detection window (s)')
    
    # Trim command
    trim_parser = subparsers.add_parser('trim', help='Analyze trim state')
    trim_parser.add_argument('--config', required=True, help='Config YAML file')
    trim_parser.add_argument('-V', '--V', type=float, help='Desired descent velocity')
    trim_parser.add_argument('--solve', action='store_true', help='Run continuation solver')
    
    args = parser.parse_args()
    
    if args.command == 'simulate':
        cmd_simulate(args)
    elif args.command == 'trim':
        cmd_trim(args)


if __name__ == '__main__':
    main()
