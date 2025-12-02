"""Layer 2: Pressure Curve Optimization

This layer optimizes fuel and oxidizer pressure curves to input into the time series solver.
The pressure curves are 200-point arrays broken into N segments (1-20).

Each segment has:
- Segment length (in terms of points/200)
- Region type: linear OR blowdown
- Start pressure (matches previous region's end pressure)
- End pressure
- k-variable for blowdown profile between those 2 points

Pressure curves are always decreasing.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Callable, List
import numpy as np
import copy
import logging
import time
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from pintle_pipeline.config_schemas import PintleEngineConfig
from pintle_models.runner import PintleEngineRunner


def generate_pressure_curve_from_segments(
    segments: List[Dict[str, Any]],
    n_points: int = 200,
) -> np.ndarray:
    """
    Generate a 200-point pressure curve from segments.
    
    Each segment specifies:
    - length_ratio: fraction of total points (0-1)
    - type: 'linear' or 'blowdown'
    - start_pressure: pressure at start [Pa]
    - end_pressure: pressure at end [Pa] (must be <= start_pressure)
    - k: blowdown parameter (only for blowdown type)
    
    Args:
        segments: List of segment dicts
        n_points: Total number of points (default 200)
    
    Returns:
        pressure_array: Array of pressures [Pa] of length n_points
    """
    if not segments:
        # Default constant pressure
        return np.full(n_points, 5e6)  # 5 MPa default
    
    pressure_array = np.zeros(n_points)
    
    # Calculate cumulative point indices for each segment
    point_idx = 0
    for i, seg in enumerate(segments):
        length_ratio = float(np.clip(seg.get("length_ratio", 1.0 / len(segments)), 0.01, 1.0))
        seg_type = seg.get("type", "linear")
        P_start = float(seg["start_pressure"])
        P_end = float(seg["end_pressure"])
        k = float(seg.get("k", 0.3))  # Blowdown parameter
        
        # Ensure end <= start (decreasing pressure)
        if P_end > P_start:
            P_end = P_start * 0.95  # Force decrease
        
        # Calculate number of points for this segment
        if i == len(segments) - 1:
            # Last segment takes remaining points
            n_seg_points = n_points - point_idx
        else:
            n_seg_points = int(round(length_ratio * n_points))
            n_seg_points = max(1, min(n_seg_points, n_points - point_idx))
        
        if n_seg_points <= 0:
            continue
        
        # Generate local indices for this segment
        seg_indices = np.arange(point_idx, point_idx + n_seg_points)
        if len(seg_indices) == 0:
            continue
        
        # Normalized position within segment (0 to 1)
        if n_seg_points > 1:
            t_norm = np.linspace(0.0, 1.0, n_seg_points)
        else:
            t_norm = np.array([0.0])
        
        if seg_type == "linear":
            # Linear interpolation: P(t) = P_start + (P_end - P_start) * t
            pressure_array[seg_indices] = P_start + (P_end - P_start) * t_norm
        elif seg_type == "blowdown":
            # Blowdown profile: P(t) = P_end + (P_start - P_end) * exp(-k * t)
            # k controls the decay rate
            pressure_array[seg_indices] = P_end + (P_start - P_end) * np.exp(-k * t_norm)
        else:
            # Default to linear
            pressure_array[seg_indices] = P_start + (P_end - P_start) * t_norm
        
        point_idx += n_seg_points
        if point_idx >= n_points:
            break
    
    # Fill any remaining points with last value
    if point_idx < n_points:
        if point_idx > 0:
            pressure_array[point_idx:] = pressure_array[point_idx - 1]
        else:
            pressure_array[:] = segments[-1]["end_pressure"] if segments else 5e6
    
    return pressure_array


def segments_from_optimizer_vars_pressure(
    x_segments: np.ndarray,
    n_segments: int,
    initial_pressure_pa: float,
    min_pressure_pa: float = 1e6,  # 1 MPa minimum
) -> List[Dict[str, Any]]:
    """
    Convert optimizer variables to segment list for pressure curves.
    
    For each segment, optimizer provides:
    - length_ratio (0-1, fraction of total 200 points)
    - end_pressure_ratio (0-1, ratio relative to initial pressure)
    - k (0-2, blowdown parameter)
    
    Start pressure is automatically set to match previous segment's end pressure.
    First segment starts at initial_pressure_pa.
    
    Args:
        x_segments: Array of optimizer variables for segments
        n_segments: Number of segments (1-20)
        initial_pressure_pa: Initial pressure from Layer 1 [Pa]
        min_pressure_pa: Minimum pressure [Pa]
    
    Returns:
        List of segment dicts
    """
    segments = []
    vars_per_segment = 3  # length_ratio, end_pressure_ratio, k
    
    # Ensure n_segments doesn't exceed available array size
    max_available_segments = len(x_segments) // vars_per_segment
    n_segments = min(n_segments, max_available_segments)
    if n_segments < 1:
        n_segments = 1
    
    # Normalize length ratios so they sum to 1.0
    length_ratios = []
    for i in range(n_segments):
        idx_base = i * vars_per_segment
        if idx_base >= len(x_segments):
            break
        length_ratio = float(np.clip(x_segments[idx_base], 0.01, 1.0))
        length_ratios.append(length_ratio)
    
    # Normalize so sum = 1.0
    total_ratio = sum(length_ratios) if length_ratios else 1.0
    if total_ratio > 0:
        length_ratios = [lr / total_ratio for lr in length_ratios]
    
    # Build segments
    prev_end_pressure = initial_pressure_pa  # First segment starts at initial pressure from Layer 1
    
    for i in range(n_segments):
        idx_base = i * vars_per_segment
        if idx_base + 2 >= len(x_segments):
            break
        
        length_ratio = length_ratios[i] if i < len(length_ratios) else 1.0 / n_segments
        
        # All segments are modeled as blowdown; k can make them effectively linear when small.
        seg_type = "blowdown"
        
        # End pressure ratio (relative to initial pressure, but must be <= start)
        end_ratio_raw = float(np.clip(x_segments[idx_base + 1], 0.1, 1.0))
        # Ensure end <= start (decreasing)
        start_ratio = prev_end_pressure / initial_pressure_pa
        end_ratio = min(end_ratio_raw, start_ratio * 0.99)  # Slight margin
        end_pressure = initial_pressure_pa * end_ratio
        end_pressure = max(min_pressure_pa, min(end_pressure, prev_end_pressure))
        
        # k parameter for blowdown
        k = float(np.clip(x_segments[idx_base + 2] if len(x_segments) > idx_base + 2 else 0.3, 0.1, 2.0))
        
        seg = {
            "length_ratio": length_ratio,
            "type": seg_type,
            "start_pressure": prev_end_pressure,
            "end_pressure": end_pressure,
            "k": k,
        }
        
        segments.append(seg)
        prev_end_pressure = end_pressure  # Next segment starts where this one ends
    
    return segments


def calculate_required_impulse_from_mass(
    target_apogee_m: float,
    rocket_dry_mass_kg: float,
    total_propellant_mass_kg: float,
    target_burn_time_s: float,
    g: float = 9.80665,
) -> float:
    """
    Calculate minimum required total impulse to reach target apogee.
    
    Uses actual propellant mass consumed to calculate initial mass.
    Uses energy conservation with approximations for gravity and drag losses.
    
    Args:
        target_apogee_m: Target apogee altitude [m]
        rocket_dry_mass_kg: Rocket dry mass (no propellant) [kg]
            Should include: airframe + engine + lox_tank_structure + fuel_tank_structure + copv_structure
        total_propellant_mass_kg: Total propellant mass consumed [kg]
            Should be: LOX propellant + fuel propellant (from integrating mdot_O and mdot_F)
        target_burn_time_s: Target burn time [s]
        g: Gravitational acceleration [m/s²]
    
    Returns:
        Required total impulse [N·s]
    """
    # Calculate initial mass from actual propellant consumption
    # rocket_dry_mass_kg = airframe + engine + all tank structures (no propellant)
    # total_propellant_mass_kg = LOX propellant + fuel propellant consumed during burn
    initial_mass = rocket_dry_mass_kg + total_propellant_mass_kg
    
    # Minimum delta-v for vertical launch (energy conservation)
    # v_burnout^2 / 2 = g * h_apogee (ignoring losses)
    min_delta_v = np.sqrt(2.0 * g * target_apogee_m)
    
    # Account for losses:
    # - Gravity loss: ~g * t_burn (velocity lost to gravity during burn)
    # - Drag loss: ~10-20% of ideal delta-v (depends on rocket, simplified here)
    gravity_loss = g * target_burn_time_s * 0.5  # Average over burn
    drag_loss_factor = 1.15  # 15% drag loss approximation
    total_delta_v = min_delta_v * drag_loss_factor + gravity_loss
    
    # Required impulse = delta_v * initial_mass
    required_impulse = total_delta_v * initial_mass
    
    return required_impulse


# Fixed segment configuration for Layer 2 optimization
# We use a shared-segment parameterization:
#   - Shared length ratios across LOX and fuel
#   - LOX end pressures and k are optimized directly
#   - Fuel end pressures are derived from LOX end pressures and a bounded
#     LOX/Fuel pressure-ratio factor per segment, keeping segment-end
#     pressure ratios within ±25% of the initial ratio.
N_SEGMENTS = 4
VARS_PER_SEGMENT = 5  # length_ratio, lox_end_pressure_ratio, lox_k, fuel_ratio_factor, fuel_k


def run_layer2a_minimum_pressures(
    optimized_config: PintleEngineConfig,
    initial_lox_pressure_pa: float,
    initial_fuel_pressure_pa: float,
    peak_thrust: float,
    target_apogee_m: float,
    rocket_dry_mass_kg: float,
    max_lox_tank_capacity_kg: float,
    max_fuel_tank_capacity_kg: float,
    target_burn_time: float,
    n_time_points: int = 100,
    update_progress: Optional[Callable] = None,
    log_status: Optional[Callable] = None,
    min_pressure_pa: float = 1e6,
    optimal_of_ratio: Optional[float] = None,
    min_stability_margin: Optional[float] = None,
) -> Tuple[float, float, Dict[str, Any], bool]:
    """
    Layer 2a: Find minimum allowable LOX and fuel tank pressures.

    This helper layer searches for the lowest *flat* tank pressures (same value
    over the full burn) that still satisfy the key constraints:
      - Required impulse to reach target apogee
      - Tank capacity limits
      - Stability margin
      - O/F ratio tolerance

    The LOX/Fuel tank pressure ratio is kept fixed to the initial ratio from
    Layer 1 by scaling both tanks by a common factor.

    Returns:
        Tuple of (min_lox_pressure_pa, min_fuel_pressure_pa, summary, success)
    """
    from scipy.integrate import cumulative_trapezoid

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"layer2a_min_pressure_{timestamp}.log"

    layer2a_logger = logging.getLogger("layer2a_min_pressure")
    layer2a_logger.setLevel(logging.INFO)
    layer2a_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    layer2a_logger.addHandler(file_handler)
    layer2a_logger.propagate = False

    layer2a_logger.info("=" * 70)
    layer2a_logger.info("Layer 2a: Minimum Tank Pressure Search")
    layer2a_logger.info("=" * 70)
    layer2a_logger.info(f"Log file: {log_file_path}")
    layer2a_logger.info(
        f"Initial LOX pressure: {initial_lox_pressure_pa/1e6:.2f} MPa "
        f"({initial_lox_pressure_pa/6894.76:.1f} psi)"
    )
    layer2a_logger.info(
        f"Initial Fuel pressure: {initial_fuel_pressure_pa/1e6:.2f} MPa "
        f"({initial_fuel_pressure_pa/6894.76:.1f} psi)"
    )
    layer2a_logger.info(f"Target burn time: {target_burn_time:.2f} s")
    layer2a_logger.info(f"Time points (Layer 2a): {n_time_points}")
    layer2a_logger.info("")

    # Disable ablative and graphite for this layer
    config_layer2a = copy.deepcopy(optimized_config)
    if hasattr(config_layer2a, "ablative_cooling") and config_layer2a.ablative_cooling:
        config_layer2a.ablative_cooling.enabled = False
    if hasattr(config_layer2a, "graphite_insert") and config_layer2a.graphite_insert:
        config_layer2a.graphite_insert.enabled = False

    runner_layer2a = PintleEngineRunner(config_layer2a)
    time_array = np.linspace(0.0, target_burn_time, n_time_points)

    initial_ratio = initial_lox_pressure_pa / max(initial_fuel_pressure_pa, 1e-9)

    def _evaluate_scale(scale: float) -> Tuple[bool, Dict[str, Any]]:
        """Return (passes_constraints, metrics) for a given common pressure scale."""
        P_lox = np.full(n_time_points, initial_lox_pressure_pa * scale)
        P_fuel = np.full(n_time_points, initial_fuel_pressure_pa * scale)

        # Enforce absolute minimum clamp
        P_lox = np.maximum(P_lox, min_pressure_pa)
        P_fuel = np.maximum(P_fuel, min_pressure_pa)

        try:
            results = runner_layer2a.evaluate_arrays_with_time(
                time_array,
                P_lox,
                P_fuel,
                track_ablative_geometry=False,
                use_coupled_solver=False,
            )
        except Exception as e:
            msg = f"Time-series solver failed at scale={scale:.3f}: {repr(e)}"
            layer2a_logger.error(msg)
            if log_status:
                log_status("Layer 2a Error", msg)
            return False, {}

        thrust_hist = np.atleast_1d(results.get("F", np.full(n_time_points, peak_thrust)))
        thrust_hist = thrust_hist[: n_time_points]
        time_hist = time_array[: thrust_hist.shape[0]]
        if thrust_hist.shape[0] < 2:
            return False, {}

        mdot_O_hist = np.atleast_1d(results.get("mdot_O", np.zeros_like(time_hist)))
        mdot_F_hist = np.atleast_1d(results.get("mdot_F", np.zeros_like(time_hist)))
        mdot_O_hist = mdot_O_hist[: time_hist.shape[0]]
        mdot_F_hist = mdot_F_hist[: time_hist.shape[0]]

        total_lox_mass = float(np.trapezoid(mdot_O_hist, time_hist))
        total_fuel_mass = float(np.trapezoid(mdot_F_hist, time_hist))
        total_propellant_mass = total_lox_mass + total_fuel_mass

        required_impulse = calculate_required_impulse_from_mass(
            target_apogee_m,
            rocket_dry_mass_kg,
            total_propellant_mass,
            target_burn_time,
        )
        total_impulse = float(np.trapezoid(thrust_hist, time_hist))

        # Constraint 1: Impulse
        passes_impulse = total_impulse >= required_impulse

        # Constraint 2: Tank capacities
        passes_lox_capacity = total_lox_mass <= max_lox_tank_capacity_kg * 1.001
        passes_fuel_capacity = total_fuel_mass <= max_fuel_tank_capacity_kg * 1.001

        # Constraint 3: Stability
        chugging_margins = results.get("chugging_stability_margin", None)
        if chugging_margins is not None:
            min_chugging = float(np.min(np.atleast_1d(chugging_margins)))
        else:
            min_chugging = 1.0

        if min_stability_margin is not None:
            passes_stability = min_chugging >= min_stability_margin
        else:
            passes_stability = min_chugging >= 0.7

        # Constraint 4: O/F ratio
        if optimal_of_ratio is not None:
            MR_hist = np.atleast_1d(results.get("MR", np.full_like(time_hist, optimal_of_ratio)))
            MR_hist = MR_hist[: time_hist.shape[0]]
            avg_MR = float(np.mean(MR_hist))
            MR_error = abs(avg_MR - optimal_of_ratio) / max(optimal_of_ratio, 1e-9)
            passes_of = MR_error <= 0.20
        else:
            passes_of = True

        passes = passes_impulse and passes_lox_capacity and passes_fuel_capacity and passes_stability and passes_of

        metrics = {
            "scale": scale,
            "P_lox_flat_pa": float(P_lox[0]),
            "P_fuel_flat_pa": float(P_fuel[0]),
            "total_impulse": total_impulse,
            "required_impulse": required_impulse,
            "total_lox_mass": total_lox_mass,
            "total_fuel_mass": total_fuel_mass,
            "min_chugging": min_chugging,
            "initial_ratio": initial_ratio,
        }
        return passes, metrics

    # Search bounds: keep above absolute minimum and below initial pressures.
    scale_high = 1.0
    # Ensure neither tank drops below min_pressure_pa
    scale_low_lox = min_pressure_pa / max(initial_lox_pressure_pa, 1e-9)
    scale_low_fuel = min_pressure_pa / max(initial_fuel_pressure_pa, 1e-9)
    scale_low = max(scale_low_lox, scale_low_fuel)
    scale_low = min(scale_low, scale_high)

    # First check if even the initial pressures meet constraints
    passes_high, high_metrics = _evaluate_scale(scale_high)
    if not passes_high:
        layer2a_logger.warning(
            "Initial pressures do not satisfy constraints; using initial pressures as minima."
        )
        summary = {
            "min_lox_pressure_pa": float(initial_lox_pressure_pa),
            "min_fuel_pressure_pa": float(initial_fuel_pressure_pa),
            "scale": 1.0,
            "success": False,
        }
        layer2a_logger.handlers.clear()
        return initial_lox_pressure_pa, initial_fuel_pressure_pa, summary, False

    # If the lowest feasible scale also passes, we can use that directly.
    passes_low, low_metrics = _evaluate_scale(scale_low)
    if passes_low:
        min_lox = low_metrics["P_lox_flat_pa"]
        min_fuel = low_metrics["P_fuel_flat_pa"]
        summary = {
            "min_lox_pressure_pa": min_lox,
            "min_fuel_pressure_pa": min_fuel,
            "scale": scale_low,
            "success": True,
        }
        layer2a_logger.handlers.clear()
        return min_lox, min_fuel, summary, True

    # Otherwise, binary search between scale_low and 1.0
    left = scale_low
    right = scale_high
    best_scale = scale_high
    best_metrics = high_metrics

    for _ in range(10):
        mid = 0.5 * (left + right)
        passes_mid, mid_metrics = _evaluate_scale(mid)
        if passes_mid:
            best_scale = mid
            best_metrics = mid_metrics
            right = mid
        else:
            left = mid

    min_lox = best_metrics["P_lox_flat_pa"]
    min_fuel = best_metrics["P_fuel_flat_pa"]

    summary = {
        "min_lox_pressure_pa": min_lox,
        "min_fuel_pressure_pa": min_fuel,
        "scale": best_scale,
        "success": True,
    }

    # Clean up logger handlers
    layer2a_logger.handlers.clear()

    return min_lox, min_fuel, summary, True


def run_layer2_pressure(
    optimized_config: PintleEngineConfig,
    initial_lox_pressure_pa: float,
    initial_fuel_pressure_pa: float,
        peak_thrust: float,  # Initial/peak thrust target
        target_apogee_m: float,  # Target apogee for impulse calculation
        rocket_dry_mass_kg: float,  # Rocket dry mass (no propellant) = airframe + engine + lox_tank_structure + fuel_tank_structure + copv_structure
        max_lox_tank_capacity_kg: float,  # Maximum LOX tank capacity [kg]
        max_fuel_tank_capacity_kg: float,  # Maximum fuel tank capacity [kg]
    target_burn_time: float,
    n_time_points: int = 200,
    update_progress: Optional[Callable] = None,
    log_status: Optional[Callable] = None,
    min_pressure_pa: float = 1e6,  # Legacy absolute minimum clamp (~150 psi)
    optimal_of_ratio: Optional[float] = None,  # Target O/F ratio for validation
    min_stability_margin: Optional[float] = None,  # Minimum stability margin
    max_iterations: int = 30,  # Maximum optimization iterations
    max_evaluations: Optional[int] = None,  # Maximum function evaluations (None = unlimited)
    save_evaluation_plots: bool = False,  # Save PNG plots of each evaluation's pressure curves
    min_lox_pressure_floor_pa: Optional[float] = None,
    min_fuel_pressure_floor_pa: Optional[float] = None,
) -> Tuple[PintleEngineConfig, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], bool]:
    """
    Run Layer 2: Pressure Curve Optimization.
    
    Optimizes fuel and oxidizer pressure curves (200-point arrays) for time series solver.
    - Uses a fixed number of segments per tank (N_SEGMENTS)
    - Each segment can be linear or blowdown with parameters:
        - length_ratio, type, end_pressure_ratio, k
    - First performs a global search (differential_evolution), then local polish (L-BFGS-B)
    
    Returns:
        Tuple of (optimized_config, time_array, P_tank_O_array, P_tank_F_array, summary, success)
    """
    from scipy.optimize import minimize as scipy_minimize, differential_evolution
    from scipy.integrate import cumulative_trapezoid
    
    # Resolve per-tank minimum pressure floors.
    # If not provided, fall back to the legacy shared min_pressure_pa.
    if min_lox_pressure_floor_pa is None:
        min_lox_pressure_floor_pa = float(min_pressure_pa)
    if min_fuel_pressure_floor_pa is None:
        min_fuel_pressure_floor_pa = float(min_pressure_pa)

    # Set up Layer 2 logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"layer2_pressure_{timestamp}.log"
    
    # Create logger for Layer 2
    layer2_logger = logging.getLogger('layer2_pressure')
    layer2_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    layer2_logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    layer2_logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    layer2_logger.propagate = False
    
    layer2_logger.info("="*70)
    layer2_logger.info("Layer 2: Pressure Curve Optimization")
    layer2_logger.info("="*70)
    layer2_logger.info(f"Log file: {log_file_path}")
    layer2_logger.info(f"Initial LOX pressure: {initial_lox_pressure_pa/1e6:.2f} MPa ({initial_lox_pressure_pa/6894.76:.1f} psi)")
    layer2_logger.info(f"Initial Fuel pressure: {initial_fuel_pressure_pa/1e6:.2f} MPa ({initial_fuel_pressure_pa/6894.76:.1f} psi)")
    layer2_logger.info(f"Peak thrust target: {peak_thrust:.1f} N")
    layer2_logger.info(f"Target apogee: {target_apogee_m:.0f} m")
    layer2_logger.info(f"Rocket dry mass: {rocket_dry_mass_kg:.2f} kg")
    layer2_logger.info(f"Target burn time: {target_burn_time:.2f} s")
    layer2_logger.info(f"Time points: {n_time_points}")
    if optimal_of_ratio is not None:
        layer2_logger.info(f"Target O/F ratio: {optimal_of_ratio:.2f}")
    if min_stability_margin is not None:
        layer2_logger.info(f"Min stability margin: {min_stability_margin:.3f}")
    layer2_logger.info("")
    
    # Set up evaluation plot file if requested
    evaluation_plot_file = None
    if save_evaluation_plots:
        evaluation_plot_file = Path(f"layer2_evaluation_plot_{timestamp}.png")
        layer2_logger.info(f"Saving evaluation plots to: {evaluation_plot_file}")
    
    # Generate time arrays (Layer 1 doesn't provide this)
    # - Full-resolution array for local optimization and final evaluation
    # - Coarser array for the global DE search to speed up evaluations
    time_array = np.linspace(0.0, target_burn_time, n_time_points)
    n_time_points_de = min(50, n_time_points)
    time_array_de = np.linspace(0.0, target_burn_time, n_time_points_de)
    
    def save_evaluation_plot(
        eval_num: int,
        time_arr: np.ndarray,
        P_lox: np.ndarray,
        P_fuel: np.ndarray,
        objective: Optional[float] = None,
        n_seg_lox: Optional[int] = None,
        n_seg_fuel: Optional[int] = None,
    ):
        """Save a plot of the pressure curves for this evaluation (overwrites previous plot)."""
        if evaluation_plot_file is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Pressure in PSI
        ax1.plot(time_arr, P_lox / 6894.76, 'b-', linewidth=2, label='LOX Tank')
        ax1.plot(time_arr, P_fuel / 6894.76, 'r-', linewidth=2, label='Fuel Tank')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Tank Pressure [psi]')
        ax1.set_title(f'Evaluation #{eval_num} Pressure Curves')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Pressure in MPa
        ax2.plot(time_arr, P_lox / 1e6, 'b-', linewidth=2, label='LOX Tank')
        ax2.plot(time_arr, P_fuel / 1e6, 'r-', linewidth=2, label='Fuel Tank')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Tank Pressure [MPa]')
        
        # Add info text
        info_text = f"Eval #{eval_num}"
        if n_seg_lox is not None:
            info_text += f" | LOX: {n_seg_lox} seg"
        if n_seg_fuel is not None:
            info_text += f" | Fuel: {n_seg_fuel} seg"
        if objective is not None:
            info_text += f"\nObjective: {objective:.6f}"
        
        ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9, family='monospace')
        
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot (overwrites previous)
        plt.savefig(evaluation_plot_file, dpi=100, bbox_inches='tight')
        plt.close(fig)  # Close to free memory
    
    # Disable ablative and graphite for this layer
    config_layer2 = copy.deepcopy(optimized_config)
    if hasattr(config_layer2, 'ablative_cooling') and config_layer2.ablative_cooling:
        config_layer2.ablative_cooling.enabled = False
    if hasattr(config_layer2, 'graphite_insert') and config_layer2.graphite_insert:
        config_layer2.graphite_insert.enabled = False

    # Create a single PintleEngineRunner instance for this layer and reuse it
    runner_layer2 = PintleEngineRunner(config_layer2)
    
    # Optimization variables:
    # Shared-segment parameterization with fixed N_SEGMENTS segments:
    #   - Shared length ratios across LOX and fuel
    #   - LOX end pressures and k are optimized directly
    #   - Fuel end pressures are derived from LOX end pressures and a bounded
    #     LOX/Fuel pressure-ratio factor per segment, keeping segment-end
    #     pressure ratios within ±25% of the initial ratio.
    #
    # x_layer2 format (per segment, in order):
    #   - length_ratio          (0-1, shared LOX/Fuel segment length)
    #   - lox_end_pressure_ratio (0-1, relative to initial LOX tank pressure)
    #   - lox_k                 (0-2, LOX blowdown parameter)
    #   - fuel_ratio_factor     (~0.75-1.25, multiplies initial LOX/Fuel ratio)
    #   - fuel_k                (0-2, fuel blowdown parameter)

    def build_x0_compact() -> np.ndarray:
        x: list[float] = []
        initial_ratio = initial_lox_pressure_pa / max(initial_fuel_pressure_pa, 1e-9)
        # Initialize segments with shared lengths, gently decaying LOX pressure,
        # and fuel following the initial LOX/Fuel ratio (ratio_factor ~= 1.0).
        for i in range(N_SEGMENTS):
            length_ratio = 1.0 / N_SEGMENTS
            lox_end_ratio = 0.9 - 0.3 * (i / max(N_SEGMENTS - 1, 1))  # monotone decreasing guess
            lox_k = 0.3
            fuel_ratio_factor = 1.0  # start near initial pressure ratio
            fuel_k = 0.3
            x.extend([length_ratio, lox_end_ratio, lox_k, fuel_ratio_factor, fuel_k])
        return np.array(x, dtype=float)

    x0 = build_x0_compact()

    # Bounds for compact vector (per segment)
    bounds: list[tuple[float, float]] = []
    for _ in range(N_SEGMENTS):
        bounds.append((0.01, 1.0))   # length_ratio (shared LOX/Fuel)
        bounds.append((0.1, 1.0))    # lox_end_pressure_ratio (relative to initial LOX)
        bounds.append((0.1, 2.0))    # lox_k
        bounds.append((0.75, 1.25))  # fuel_ratio_factor (keeps segment-end ratios within ±25%)
        bounds.append((0.1, 2.0))    # fuel_k
    
    # Track optimization progress
    layer2_state = {
        "iter": 0,
        "max_iter": max_iterations,
        "best_obj": float("inf"),
        "last_obj": None,
        "eval_count": 0,
        "start_time": time.time()
    }
    
    def layer2_callback(xk):
        layer2_state["iter"] += 1
        frac = min(layer2_state["iter"] / max(layer2_state["max_iter"], 1), 1.0)
        progress_pct = int(frac * 100)
        elapsed = time.time() - layer2_state["start_time"]
        
        # Log progress
        if layer2_state["last_obj"] is not None:
            layer2_logger.info(f"[{progress_pct}%] Iteration {layer2_state['iter']}/{layer2_state['max_iter']} "
                            f"({elapsed:.1f}s elapsed) - "
                            f"Objective: {layer2_state['last_obj']:.6f} (Best: {layer2_state['best_obj']:.6f})")
        else:
            layer2_logger.info(f"[{progress_pct}%] Iteration {layer2_state['iter']}/{layer2_state['max_iter']} "
                            f"({elapsed:.1f}s elapsed)")
        
        # Flush log to ensure it's written immediately
        for handler in layer2_logger.handlers:
            handler.flush()
        
        # Call external progress callback if provided
        if update_progress:
            progress = 0.60 + 0.04 * frac
            update_progress(
                "Layer 2: Pressure Curve Optimization",
                progress,
                f"Layer 2 optimization {layer2_state['iter']}/{layer2_state['max_iter']} ({progress_pct}%)",
            )
    
    def decode_segments_from_x(
        x_layer2: np.ndarray,
        n_segments: int,
        initial_lox_p: float,
        initial_fuel_p: float,
        min_lox_p: float,
        min_fuel_p: float,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Decode the compact optimization vector into LOX and fuel segment lists.
        
        - Shared length ratios across LOX and fuel
        - LOX end pressures and k are optimized directly
        - Fuel end pressures are derived from LOX end pressures and a bounded
          LOX/Fuel pressure-ratio factor per segment, keeping segment-end
          pressure ratios within ±25% of the initial ratio.
        """
        segments_lox: List[Dict[str, Any]] = []
        segments_fuel: List[Dict[str, Any]] = []
        
        # Ensure we don't request more segments than available in x
        max_segments = len(x_layer2) // VARS_PER_SEGMENT
        n_segments = min(n_segments, max_segments)
        if n_segments < 1:
            n_segments = 1
        
        # Normalize shared length ratios
        raw_lengths: List[float] = []
        for i in range(n_segments):
            base = i * VARS_PER_SEGMENT
            if base >= len(x_layer2):
                break
            lr = float(np.clip(x_layer2[base], 0.01, 1.0))
            raw_lengths.append(lr)
        
        total_lr = sum(raw_lengths) if raw_lengths else 1.0
        if total_lr <= 0:
            total_lr = 1.0
        length_ratios = [lr / total_lr for lr in raw_lengths]
        
        # Build LOX and fuel segments
        prev_lox_end = initial_lox_p
        prev_fuel_end = initial_fuel_p
        initial_ratio = initial_lox_p / max(initial_fuel_p, 1e-9)
        
        for i in range(n_segments):
            base = i * VARS_PER_SEGMENT
            if base + 4 >= len(x_layer2):
                break
            
            length_ratio = length_ratios[i] if i < len(length_ratios) else 1.0 / n_segments
            
            # LOX end pressure ratio and k
            lox_end_ratio_raw = float(np.clip(x_layer2[base + 1], 0.1, 1.0))
            start_ratio_lox = prev_lox_end / max(initial_lox_p, 1e-9)
            lox_end_ratio = min(lox_end_ratio_raw, start_ratio_lox * 0.99)
            lox_end_p = initial_lox_p * lox_end_ratio
            # Ensure monotonic decrease and enforce LOX minimum floor.
            lox_end_p = max(min_lox_p, min(lox_end_p, prev_lox_end))
            lox_k = float(np.clip(x_layer2[base + 2], 0.1, 2.0))
            
            # Fuel ratio factor (keeps segment-end ratio within ±25% of initial)
            fuel_ratio_factor = float(np.clip(x_layer2[base + 3], 0.75, 1.25))
            seg_ratio = initial_ratio * fuel_ratio_factor
            fuel_end_p = lox_end_p / max(seg_ratio, 1e-9)
            # Ensure monotonic decrease and enforce fuel minimum floor.
            fuel_end_p = max(min_fuel_p, min(fuel_end_p, prev_fuel_end))
            fuel_k = float(np.clip(x_layer2[base + 4], 0.1, 2.0))
            
            seg_lox = {
                "length_ratio": length_ratio,
                "type": "blowdown",
                "start_pressure": prev_lox_end,
                "end_pressure": lox_end_p,
                "k": lox_k,
            }
            seg_fuel = {
                "length_ratio": length_ratio,
                "type": "blowdown",
                "start_pressure": prev_fuel_end,
                "end_pressure": fuel_end_p,
                "k": fuel_k,
            }
            segments_lox.append(seg_lox)
            segments_fuel.append(seg_fuel)
            prev_lox_end = lox_end_p
            prev_fuel_end = fuel_end_p
        
        return segments_lox, segments_fuel

    def layer2_objective(
        x_layer2,
        time_eval_override: Optional[np.ndarray] = None,
        n_points: Optional[int] = None,
        phase: str = "local",
    ):
        """Optimize pressure curves after initial point. Initial pressures are fixed from Layer 1.
        
        Args:
            x_layer2: Optimization variable vector.
            time_eval_override: Optional time array to use for this evaluation
                (e.g., coarse grid for DE). If None, uses the full-resolution
                `time_array` defined above.
            n_points: Optional number of time points. If None, uses the global
                `n_time_points` defined above.
            phase: Short label for logging (e.g., "DE" or "local").
        """
        eval_start_time = time.time()
        layer2_state["eval_count"] += 1
        eval_num = layer2_state["eval_count"]
        
        # Select resolution for this objective call
        if n_points is None:
            n_points = n_time_points
        if time_eval_override is None:
            time_eval = time_array.copy()
        else:
            time_eval = time_eval_override.copy()
        
        try:
            # Log every evaluation (important for long-running evaluations)
            layer2_logger.info(
                f"  → Evaluation #{eval_num} [{phase}]: {N_SEGMENTS} shared segments (LOX/Fuel, {n_points} pts)"
            )
            for handler in layer2_logger.handlers:
                handler.flush()
            
            # Decode shared parameterization into LOX and fuel segments
            lox_segments, fuel_segments = decode_segments_from_x(
                x_layer2,
                N_SEGMENTS,
                initial_lox_pressure_pa,
                initial_fuel_pressure_pa,
                min_lox_pressure_floor_pa,
                min_fuel_pressure_floor_pa,
            )
            
            # Generate pressure curves at the chosen resolution (already floored to minima)
            P_tank_O_array = generate_pressure_curve_from_segments(lox_segments, n_points)
            P_tank_F_array = generate_pressure_curve_from_segments(fuel_segments, n_points)
            
            # Ensure first point is exactly the initial pressure (fixed from Layer 1)
            P_tank_O_array[0] = initial_lox_pressure_pa
            P_tank_F_array[0] = initial_fuel_pressure_pa
            
            # Hard constraint for global search: reject pressure curves where LOX/Fuel
            # pressure ratio deviates too far from the initial ratio during the
            # physically relevant part of the burn (well above the min-pressure clamp).
            # This keeps differential_evolution from wasting evaluations on clearly
            # infeasible LOX/Fuel pressure relationships while ignoring tail-end
            # artifacts when both tanks are clamped near min_pressure_pa.
            initial_pressure_ratio = initial_lox_pressure_pa / max(initial_fuel_pressure_pa, 1e-9)
            active_min_floor = min(min_lox_pressure_floor_pa, min_fuel_pressure_floor_pa)
            active_mask = (P_tank_O_array > 1.1 * active_min_floor) & (P_tank_F_array > 1.1 * active_min_floor)
            if np.any(active_mask):
                pressure_ratio_array = P_tank_O_array[active_mask] / np.maximum(P_tank_F_array[active_mask], 1e-9)
                ratio_error_array = np.abs(pressure_ratio_array - initial_pressure_ratio) / max(
                    initial_pressure_ratio, 1e-9
                )
                if np.any(ratio_error_array > 0.25):
                    max_error = float(np.max(ratio_error_array))
                    max_excess = float(np.max(ratio_error_array - 0.25))
                    # Large objective value so DE moves away from this region, but keep it finite
                    # so the optimizer remains numerically well-behaved.
                    hard_ratio_penalty = 5e4 + max_excess * 1e4
                    layer2_logger.info(
                        f"    Skipping time series solve for eval #{eval_num}: "
                        f"pressure ratio error {max_error:.3f} (> 0.25 allowed) in active burn window"
                    )
                    for handler in layer2_logger.handlers:
                        handler.flush()
                    return hard_ratio_penalty
            
            # Save plot if requested (before running expensive solver)
            if save_evaluation_plots:
                save_evaluation_plot(
                    eval_num=eval_num,
                    time_arr=time_eval,
                    P_lox=P_tank_O_array,
                    P_fuel=P_tank_F_array,
                    n_seg_lox=N_SEGMENTS,
                    n_seg_fuel=N_SEGMENTS,
                )
            
            # Run time series evaluation (ablative/graphite disabled in config)
            layer2_logger.info(f"    Running time series solver ({n_points} points)...")
            for handler in layer2_logger.handlers:
                handler.flush()
            
            ts_start = time.time()
            results_layer2 = runner_layer2.evaluate_arrays_with_time(
                time_eval,
                P_tank_O_array,
                P_tank_F_array,
                track_ablative_geometry=False,  # Disable ablative tracking
                use_coupled_solver=False,  # Use simpler solver
            )
            ts_time = time.time() - ts_start
            layer2_logger.info(f"    Time series solver completed in {ts_time:.2f}s")
            for handler in layer2_logger.handlers:
                handler.flush()
            
            # Get thrust history
            thrust_hist = np.atleast_1d(results_layer2.get("F", np.full(n_points, peak_thrust)))
            available_n = min(thrust_hist.shape[0], n_points)
            if available_n < 1:
                return 1e6
            
            thrust_hist = thrust_hist[:available_n]
            time_hist = time_eval[:available_n]
            
            # Note: Initial pressures are fixed from Layer 1 and assumed to produce peak_thrust
            # We do not optimize or check initial thrust - it's already correct from Layer 1
            
            # Calculate actual propellant mass consumed by integrating mass flow rates
            mdot_O_hist = np.atleast_1d(results_layer2.get("mdot_O", np.zeros(available_n)))
            mdot_F_hist = np.atleast_1d(results_layer2.get("mdot_F", np.zeros(available_n)))
            mdot_O_hist = mdot_O_hist[:available_n]
            mdot_F_hist = mdot_F_hist[:available_n]
            
            # Integrate mass flow rates to get total propellant consumed
            total_lox_mass = float(np.trapezoid(mdot_O_hist, time_hist))  # kg
            total_fuel_mass = float(np.trapezoid(mdot_F_hist, time_hist))  # kg
            total_propellant_mass = total_lox_mass + total_fuel_mass
            
            # Calculate required impulse based on actual propellant consumption
            required_impulse = calculate_required_impulse_from_mass(
                target_apogee_m,
                rocket_dry_mass_kg,
                total_propellant_mass,
                target_burn_time,
            )
            
            # Check 1: Total impulse must be >= required impulse
            total_impulse = float(np.trapezoid(thrust_hist, time_hist))  # N·s
            impulse_deficit = max(0, required_impulse - total_impulse)
            impulse_penalty = (impulse_deficit / max(required_impulse, 1e-9)) * 200.0  # Large penalty if insufficient
            
            # Soft preference: keep the burn active closer to the target_burn_time.
            # We measure when 95% of the total impulse has been delivered; if that
            # happens too early relative to target_burn_time, we add a penalty.
            burn_time_penalty = 0.0
            if total_impulse > 0:
                # Cumulative impulse over time
                cumulative_impulse = cumulative_trapezoid(thrust_hist, time_hist, initial=0.0)
                impulse_95 = 0.95 * total_impulse
                idx_95 = np.searchsorted(cumulative_impulse, impulse_95, side="right")
                if idx_95 >= len(time_hist):
                    idx_95 = len(time_hist) - 1
                t95 = float(time_hist[idx_95])
                # If 95% of impulse is achieved significantly before target_burn_time,
                # penalize the "slack" time to encourage more impulse later in the burn.
                burn_completion_slack = max(0.0, target_burn_time - t95)
                # Normalize by target_burn_time so penalty is dimensionless.
                burn_time_penalty = (burn_completion_slack / max(target_burn_time, 1e-9)) * 20.0
            
            # Check 2: Propellant mass must not exceed tank capacity
            lox_capacity_exceeded = max(0, total_lox_mass - max_lox_tank_capacity_kg)
            fuel_capacity_exceeded = max(0, total_fuel_mass - max_fuel_tank_capacity_kg)
            capacity_penalty = 0.0
            if lox_capacity_exceeded > 0:
                capacity_penalty += (lox_capacity_exceeded / max(max_lox_tank_capacity_kg, 1e-9)) * 300.0
            if fuel_capacity_exceeded > 0:
                capacity_penalty += (fuel_capacity_exceeded / max(max_fuel_tank_capacity_kg, 1e-9)) * 300.0
            
            # Check 3: Stability (must pass minimum threshold)
            stability_scores = results_layer2.get("stability_score", None)
            if stability_scores is not None:
                min_stability = float(np.min(stability_scores))
            else:
                chugging = results_layer2.get("chugging_stability_margin", np.array([1.0]))
                min_stability = max(0.0, min(1.0, (float(np.min(chugging)) - 0.3) * 1.5))
            
            stability_penalty = 0.0
            if min_stability_margin is not None:
                # Check against minimum stability margin
                chugging_margins = results_layer2.get("chugging_stability_margin", np.array([1.0]))
                min_chugging = float(np.min(chugging_margins))
                if min_chugging < min_stability_margin:
                    stability_penalty = (min_stability_margin - min_chugging) * 50.0
            else:
                # Default: penalty if stability score < 0.7
                stability_penalty = max(0, 0.7 - min_stability) * 10.0
            
            # Check 4: O/F ratio (if specified)
            of_penalty = 0.0
            if optimal_of_ratio is not None:
                MR_hist = np.atleast_1d(results_layer2.get("MR", np.full(available_n, optimal_of_ratio)))
                MR_hist = MR_hist[:available_n]
                avg_MR = float(np.mean(MR_hist))
                MR_error = abs(avg_MR - optimal_of_ratio) / max(optimal_of_ratio, 1e-9)
                # Allow 20% O/F error before penalty
                if MR_error > 0.20:
                    of_penalty = (MR_error - 0.20) * 20.0
            
            # Objective: minimize penalties (no initial thrust penalty - it's fixed from Layer 1)
            obj = (
                impulse_penalty
                + burn_time_penalty
                + capacity_penalty
                + stability_penalty
                + of_penalty
            )
            
            eval_time = time.time() - eval_start_time
            
            # Update plot with objective value if plots are being saved
            if save_evaluation_plots and phase == "local":
                # Re-save plot with objective value
                save_evaluation_plot(
                    eval_num=eval_num,
                    time_arr=time_eval,
                    P_lox=P_tank_O_array,
                    P_fuel=P_tank_F_array,
                    objective=obj,
                    n_seg_lox=N_SEGMENTS,
                    n_seg_fuel=N_SEGMENTS,
                )
            
            # Update best objective
            if obj < layer2_state["best_obj"]:
                layer2_state["best_obj"] = obj
                layer2_logger.info(
                    f"    ✓ New best objective: {obj:.6f} "
                    f"(penalties: impulse={impulse_penalty:.2f}, burn_time={burn_time_penalty:.2f}, "
                    f"capacity={capacity_penalty:.2f}, stability={stability_penalty:.2f}, "
                    f"O/F={of_penalty:.2f}) "
                    f"- Evaluation took {eval_time:.2f}s"
                )
            else:
                layer2_logger.info(
                    f"    Objective: {obj:.6f} (best: {layer2_state['best_obj']:.6f}) - "
                    f"Evaluation took {eval_time:.2f}s"
                )
            
            layer2_state["last_obj"] = obj
            
            # Flush log
            for handler in layer2_logger.handlers:
                handler.flush()
            
            return obj
        except Exception as e:
            eval_time = time.time() - eval_start_time
            error_msg = f"Exception in objective evaluation #{eval_num} (took {eval_time:.2f}s): {repr(e)}"
            layer2_logger.error(error_msg)
            import traceback
            layer2_logger.error(traceback.format_exc())
            for handler in layer2_logger.handlers:
                handler.flush()
            if log_status:
                log_status("Layer 2 Pressure Error", error_msg)
            return 1e6
    
    # Wrapper for coarse global-search objective (DE) using fewer time points
    def layer2_objective_de(x_layer2: np.ndarray) -> float:
        return layer2_objective(
            x_layer2,
            time_eval_override=time_array_de,
            n_points=n_time_points_de,
            phase="DE",
        )
    
    # Wrapper for full-resolution local-search objective
    def layer2_objective_local(x_layer2: np.ndarray) -> float:
        return layer2_objective(
            x_layer2,
            time_eval_override=None,
            n_points=None,
            phase="local",
        )
    
    # Optimize
    success = False
    P_tank_O_optimized = None
    P_tank_F_optimized = None
    summary = {}
    n_segments_used = N_SEGMENTS  # Fixed number of segments per tank
    
    layer2_logger.info("Starting optimization...")
    layer2_logger.info(f"Using fixed {N_SEGMENTS} segments per tank for LOX and fuel")
    layer2_logger.info(f"Max local iterations: {layer2_state['max_iter']}")
    layer2_logger.info("")
    
    try:
        # Global search with differential evolution (coarse time grid, n_time_points_de)
        # We only want to explore a *handful* of overall pressure-curve shapes
        # and then let the local optimizer do the heavy lifting, so we keep
        # DE very small and fast.
        de_result = differential_evolution(
            layer2_objective_de,
            bounds,
            maxiter=3,     # very few generations
            popsize=1,     # minimal population → small number of shapes explored
            polish=False,
            tol=0.2,
        )
        layer2_logger.info(
            "Global search (differential_evolution) finished with objective %.6f",
            de_result.fun,
        )
        for handler in layer2_logger.handlers:
            handler.flush()

        # Local polish with L-BFGS-B (full time grid, n_time_points)
        result_layer2 = scipy_minimize(
            layer2_objective_local,
            de_result.x,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": max_iterations,
                "ftol": 1e-4,
            },
            callback=layer2_callback,
        )
        
        layer2_logger.info("")
        layer2_logger.info("Optimization completed")
        layer2_logger.info(f"Success: {result_layer2.success}")
        layer2_logger.info(f"Final objective value: {result_layer2.fun:.6f}")
        layer2_logger.info(f"Iterations: {result_layer2.nit if hasattr(result_layer2, 'nit') else 'N/A'}")
        layer2_logger.info(f"Function evaluations: {result_layer2.nfev if hasattr(result_layer2, 'nfev') else 'N/A'}")
        layer2_logger.info("")
        
        if result_layer2.success or result_layer2.fun < 1e5:
            success = True
            layer2_logger.info("✓ Optimization converged successfully")
            
            # Extract optimized segments
            layer2_logger.info(f"Optimized solution uses {N_SEGMENTS} LOX segments and {N_SEGMENTS} fuel segments")
            
            lox_segments, fuel_segments = decode_segments_from_x(
                result_layer2.x,
                N_SEGMENTS,
                initial_lox_pressure_pa,
                initial_fuel_pressure_pa,
                min_lox_pressure_floor_pa,
                min_fuel_pressure_floor_pa,
            )
            
            # Generate optimized pressure curves
            P_tank_O_optimized = generate_pressure_curve_from_segments(lox_segments, n_time_points)
            P_tank_F_optimized = generate_pressure_curve_from_segments(fuel_segments, n_time_points)
            
            if update_progress:
                update_progress(
                    "Layer 2: Pressure Curve Optimization",
                    0.64,
                    f"Optimized: {N_SEGMENTS} LOX segments, {N_SEGMENTS} fuel segments"
                )
        else:
            layer2_logger.warning("⚠ Optimization did not converge, using initial guess")
            if update_progress:
                update_progress(
                    "Layer 2: Pressure Curve Optimization",
                    0.64,
                    f"⚠️ Optimization did not converge, using initial guess"
                )
                # Use initial guess with shared parameterization
                lox_segments, fuel_segments = decode_segments_from_x(
                    x0,
                    N_SEGMENTS,
                initial_lox_pressure_pa,
                initial_fuel_pressure_pa,
                min_lox_pressure_floor_pa,
                min_fuel_pressure_floor_pa,
                )
            P_tank_O_optimized = generate_pressure_curve_from_segments(lox_segments, n_time_points)
            P_tank_F_optimized = generate_pressure_curve_from_segments(fuel_segments, n_time_points)
    
    except Exception as e:
        error_msg = f"Exception in optimization: {repr(e)}"
        layer2_logger.error(error_msg)
        import traceback
        layer2_logger.error(traceback.format_exc())
        if log_status:
            log_status("Layer 2 Pressure Error", error_msg)
        if update_progress:
            update_progress(
                "Layer 2: Pressure Curve Optimization",
                0.64,
                f"⚠️ Optimization failed: {e}, using initial guess"
            )
        # Fallback to simple linear pressure decay
        P_tank_O_optimized = np.linspace(initial_lox_pressure_pa, initial_lox_pressure_pa * 0.7, n_time_points)
        P_tank_F_optimized = np.linspace(initial_fuel_pressure_pa, initial_fuel_pressure_pa * 0.7, n_time_points)
        layer2_logger.warning("Using fallback linear pressure decay")
    
    # Build summary
    layer2_logger.info("")
    layer2_logger.info("Calculating final results...")
    if P_tank_O_optimized is not None and P_tank_F_optimized is not None:
        # Calculate final total impulse and propellant consumption for summary
        try:
            results_final = runner_layer2.evaluate_arrays_with_time(
                time_array,
                P_tank_O_optimized,
                P_tank_F_optimized,
                track_ablative_geometry=False,
                use_coupled_solver=False,
            )
            thrust_final = np.atleast_1d(results_final.get("F", [peak_thrust]))
            mdot_O_final = np.atleast_1d(results_final.get("mdot_O", np.zeros(n_time_points)))
            mdot_F_final = np.atleast_1d(results_final.get("mdot_F", np.zeros(n_time_points)))
            
            total_impulse_actual = float(np.trapezoid(thrust_final, time_array))
            initial_thrust_actual = float(thrust_final[0])
            
            # Calculate actual propellant mass consumed
            total_lox_mass_final = float(np.trapezoid(mdot_O_final, time_array))
            total_fuel_mass_final = float(np.trapezoid(mdot_F_final, time_array))
            total_propellant_mass = total_lox_mass_final + total_fuel_mass_final
            
            # Calculate required impulse from actual propellant consumption
            required_impulse_final = calculate_required_impulse_from_mass(
                target_apogee_m,
                rocket_dry_mass_kg,
                total_propellant_mass,
                target_burn_time,
            )
            
            # Log final results
            layer2_logger.info("")
            layer2_logger.info("="*70)
            layer2_logger.info("Final Results Summary")
            layer2_logger.info("="*70)
            layer2_logger.info(f"Initial thrust: {initial_thrust_actual:.1f} N (target: {peak_thrust:.1f} N)")
            layer2_logger.info(f"Total impulse: {total_impulse_actual/1000:.1f} kN·s")
            layer2_logger.info(f"Required impulse: {required_impulse_final/1000:.1f} kN·s")
            layer2_logger.info(f"Impulse ratio: {total_impulse_actual/max(required_impulse_final, 1e-9)*100:.1f}%")
            layer2_logger.info(f"LOX consumed: {total_lox_mass_final:.3f} kg ({total_lox_mass_final/max(max_lox_tank_capacity_kg, 1e-9)*100:.1f}% of capacity)")
            layer2_logger.info(f"Fuel consumed: {total_fuel_mass_final:.3f} kg ({total_fuel_mass_final/max(max_fuel_tank_capacity_kg, 1e-9)*100:.1f}% of capacity)")
            layer2_logger.info(f"Total propellant: {total_propellant_mass:.3f} kg")
            layer2_logger.info(f"LOX end pressure: {P_tank_O_optimized[-1]/6894.76:.1f} psi ({P_tank_O_optimized[-1]/1e6:.2f} MPa)")
            layer2_logger.info(f"Fuel end pressure: {P_tank_F_optimized[-1]/6894.76:.1f} psi ({P_tank_F_optimized[-1]/1e6:.2f} MPa)")
            layer2_logger.info("="*70)
        except Exception as e:
            layer2_logger.error(f"Error calculating final results: {repr(e)}")
            total_impulse_actual = 0.0
            initial_thrust_actual = 0.0
            total_lox_mass_final = 0.0
            total_fuel_mass_final = 0.0
            total_propellant_mass = 0.0
            required_impulse_final = 0.0
        
        summary = {
            "lox_segments": n_segments_used,
            "fuel_segments": n_segments_used,
            "initial_lox_pressure_pa": initial_lox_pressure_pa,
            "initial_fuel_pressure_pa": initial_fuel_pressure_pa,
            "lox_start_pressure_pa": float(P_tank_O_optimized[0]),
            "lox_end_pressure_pa": float(P_tank_O_optimized[-1]),
            "fuel_start_pressure_pa": float(P_tank_F_optimized[0]),
            "fuel_end_pressure_pa": float(P_tank_F_optimized[-1]),
            "target_burn_time": target_burn_time,
            "n_time_points": n_time_points,
            "peak_thrust": peak_thrust,
            "initial_thrust_actual": initial_thrust_actual,
            "total_lox_mass_kg": total_lox_mass_final,
            "total_fuel_mass_kg": total_fuel_mass_final,
            "total_propellant_mass_kg": total_propellant_mass,
            "max_lox_tank_capacity_kg": max_lox_tank_capacity_kg,
            "max_fuel_tank_capacity_kg": max_fuel_tank_capacity_kg,
            "lox_capacity_ratio": total_lox_mass_final / max(max_lox_tank_capacity_kg, 1e-9),
            "fuel_capacity_ratio": total_fuel_mass_final / max(max_fuel_tank_capacity_kg, 1e-9),
            "required_impulse": required_impulse_final,
            "total_impulse_actual": total_impulse_actual,
            "impulse_ratio": total_impulse_actual / max(required_impulse_final, 1e-9),
        }
    
    layer2_logger.info("")
    layer2_logger.info(f"Layer 2 optimization complete. Log saved to: {log_file_path}")
    
    # Clean up handler to prevent file handle issues
    layer2_logger.handlers.clear()
    
    return optimized_config, time_array, P_tank_O_optimized, P_tank_F_optimized, summary, success

