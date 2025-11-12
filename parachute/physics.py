"""Equations, line law, inflation, drag, rotations"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
from .model import System, BodyNode, CanopyNode, LineEdge, ReefingStage


# ============================================================================
# Inflation functions
# ============================================================================

def f_area(xi: float) -> float:
    """Area inflation function: f(xi) = 1 - exp(-xi) for xi >= 0, else 0"""
    return 0.0 if xi < 0 else 1.0 - np.exp(-xi)


def g_cd(xi: float) -> float:
    """Drag coefficient inflation function: g(xi) = tanh(xi) for xi >= 0, else 0"""
    return 0.0 if xi < 0 else np.tanh(xi)


# ============================================================================
# Rotation/quaternion utilities
# ============================================================================

def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion (scalar-first)"""
    if q is None or len(q) != 4:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    # Check for non-finite values
    if not np.all(np.isfinite(q)):
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    result = q / n
    
    # Ensure result is finite
    if not np.all(np.isfinite(result)):
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    return result


def quat_to_rotm(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (scalar-first) to rotation matrix body->inertial"""
    q = quat_normalize(q)  # Ensure normalized
    
    # Validate quaternion
    if not np.all(np.isfinite(q)) or len(q) != 4:
        return np.eye(3)  # Return identity matrix if invalid
    
    w, x, y, z = q
    R = np.array([
        [w*w + x*x - y*y - z*z, 2*(x*y - w*z),         2*(x*z + w*y)        ],
        [2*(x*y + w*z),         w*w - x*x + y*y - z*z, 2*(y*z - w*x)        ],
        [2*(x*z - w*y),         2*(y*z + w*x),         w*w - x*x - y*y + z*z]
    ])
    
    # Validate rotation matrix
    if not np.all(np.isfinite(R)):
        return np.eye(3)
    
    return R


def quat_omega_mat(q: np.ndarray) -> np.ndarray:
    """4x3 matrix so that qdot = 0.5 * Q(q) * omega (scalar-first)"""
    q = quat_normalize(q)
    w, x, y, z = q
    return 0.5 * np.array([
        [-x, -y, -z],
        [ w, -z,  y],
        [ z,  w, -x],
        [-y,  x,  w]
    ])


# ============================================================================
# Reefing parameter evaluation
# ============================================================================

def ramp_exp(t: float, t_on: float, tau: float) -> float:
    """Exponential ramp: r(t) = 1 - exp(-(t - t_on) / tau) for t >= t_on"""
    if t < t_on:
        return 0.0
    return 1.0 - np.exp(-(t - t_on) / max(tau, 1e-9))


def ramp_tanh(t: float, t_on: float, tau: float) -> float:
    """Tanh ramp: r(t) = tanh((t - t_on) / tau) for t >= t_on"""
    if t < t_on:
        return 0.0
    return np.tanh((t - t_on) / max(tau, 1e-9))


def get_effective_reefing_param(
    t: float,
    base_value: float,
    stages: list[ReefingStage],
    param_name: str,
    ramp_shape: str = 'exp'
) -> float:
    """
    Get effective reefing parameter at time t with ramping.
    
    Args:
        t: Current time
        base_value: Base value (before any reefing)
        stages: List of reefing stages (must be sorted by t_on)
        param_name: 'L0', 'k0', 'c', etc.
        ramp_shape: 'exp' or 'tanh'
    
    Returns:
        Effective parameter value at time t
    """
    if not stages:
        return base_value
    
    # Find target stage (most recent with t_on <= t)
    target_stage = None
    prev_value = base_value
    
    for stage in stages:
        if stage.t_on > t:
            break
        target_stage = stage
        # Update prev_value to what this stage would give (without ramping)
        stage_value = getattr(stage, param_name, None)
        if stage_value is not None:
            prev_value = stage_value
    
    if target_stage is None:
        return base_value
    
    stage_value = getattr(target_stage, param_name, None)
    if stage_value is None:
        # Parameter not overridden in this stage, use previous value
        return prev_value
    
    # Check if ramping
    if target_stage.ramp_tau is not None:
        # Continuous ramp
        ramp_func = ramp_exp if ramp_shape == 'exp' else ramp_tanh
        r = ramp_func(t, target_stage.t_on, target_stage.ramp_tau)
        return prev_value + r * (stage_value - prev_value)
    else:
        # Step change at t_on
        return stage_value


def get_canopy_effective_scales(
    system: System,
    canopy_id: str,
    t: float,
    ramp_shape: str = 'exp'
) -> Tuple[float, float]:
    """
    Get effective A_scale and CD_scale for a canopy by multiplying all upstream edge scales.
    
    Returns:
        (A_scale_eff, CD_scale_eff)
    """
    from .model import get_upstream_edges_for_canopy
    
    upstream_edges = get_upstream_edges_for_canopy(system, canopy_id)
    
    A_scale_prod = 1.0
    CD_scale_prod = 1.0
    
    for edge_id in upstream_edges:
        edge = system.edges[edge_id]
        # Find active stage that provides scales
        for stage in edge.reefing:
            if stage.t_on <= t:
                if stage.A_scale is not None:
                    # Check if ramping
                    if stage.ramp_tau is not None:
                        ramp_func = ramp_exp if ramp_shape == 'exp' else ramp_tanh
                        r = ramp_func(t, stage.t_on, stage.ramp_tau)
                        prev_scale = A_scale_prod
                        A_scale_prod = prev_scale + r * (stage.A_scale - prev_scale)
                    else:
                        A_scale_prod *= stage.A_scale
                
                if stage.CD_scale is not None:
                    if stage.ramp_tau is not None:
                        ramp_func = ramp_exp if ramp_shape == 'exp' else ramp_tanh
                        r = ramp_func(t, stage.t_on, stage.ramp_tau)
                        prev_scale = CD_scale_prod
                        CD_scale_prod = prev_scale + r * (stage.CD_scale - prev_scale)
                    else:
                        CD_scale_prod *= stage.CD_scale
    
    return A_scale_prod, CD_scale_prod


# ============================================================================
# Canopy inflation and drag
# ============================================================================

def canopy_area(
    t: float, 
    canopy: CanopyNode, 
    system: System, 
    ramp_shape: str = 'exp',
    t_pickup: Optional[float] = None,
    upstream_pickup_times: Optional[Dict[str, float]] = None
) -> float:
    """
    Effective canopy area at time t (includes reefing scales and hyperinflation).
    
    Inflation only starts after pickup (line goes taut) - canopy stays packed until then.
    Hyperinflation can cause area to exceed A_inf during rapid inflation.
    
    Slack management: If upstream_canopy is set, downstream canopy won't inflate until
    upstream has been picked up and inflated (to prevent premature main inflation).
    """
    # Must have deployment time/altitude reached
    if canopy.altitude_deploy is not None:
        # Altitude-based deployment: check if we've reached deployment altitude
        # CRITICAL: Validate canopy position is finite before using it
        if not np.all(np.isfinite(canopy.p)):
            # Canopy position is invalid - this can happen during separation or numerical issues
            # Use previous valid area or return 0 (conservative)
            return 0.0
        
        altitude_agl = canopy.p[2]  # z-coordinate is altitude (positive = above ground)
        
        # CRITICAL: Once a canopy has been picked up (t_pickup is set), it's deployed
        # and should continue inflating regardless of altitude changes (e.g., from separation)
        # Only check altitude BEFORE pickup (during deployment phase)
        if t_pickup is None:
            # Haven't picked up yet - check if we've reached deployment altitude
            # Deploy when descending past deployment altitude (altitude <= deploy_altitude)
            if altitude_agl > canopy.altitude_deploy:
                return 0.0  # Still above deployment altitude, not deployed yet
        # If t_pickup is set, canopy is already deployed - allow inflation regardless of altitude
    else:
        # Time-based deployment
        if t < canopy.td:
            return 0.0
    
    # Slack management: If this canopy depends on an upstream canopy, wait for it
    if canopy.upstream_canopy is not None and upstream_pickup_times is not None:
        upstream_id = canopy.upstream_canopy
        if upstream_id not in upstream_pickup_times:
            # Upstream hasn't been picked up yet - keep this canopy slack
            return 0.0
        # Check if upstream has inflated enough (simple check: if pickup time is recent enough)
        upstream_pickup = upstream_pickup_times[upstream_id]
        if upstream_pickup is not None and t < upstream_pickup + 0.1:  # Wait 100ms after upstream pickup
            return 0.0
    
    # If pickup hasn't occurred yet, canopy is still packed (no inflation)
    # CRITICAL: Check upstream_pickup_times if t_pickup is None (for multi-stage systems)
    if t_pickup is None and upstream_pickup_times is not None:
        # Try to get pickup time from upstream dict (for canopies that get pickup from edge)
        t_pickup = upstream_pickup_times.get(canopy.id)
    
    if t_pickup is None:
        return 0.0
    if t < t_pickup:
        return 0.0
    
    # Time since pickup (when inflation actually starts)
    t_since_pickup = t - t_pickup
    xi = t_since_pickup / max(canopy.tau_A, 1e-9)
    A_base = canopy.A_inf * f_area(xi)
    
    # Debug first few inflations
    if t_since_pickup < 0.1 and A_base > 0.01:
        print(f"DEBUG inflation: {canopy.id} t={t:.4f} t_pickup={t_pickup:.4f} t_since={t_since_pickup:.4f} xi={xi:.4f} A_base={A_base:.4f}")
    
    # Hyperinflation: during rapid inflation, canopy can overshoot A_inf
    # Hyperinflation factor peaks early in inflation when velocity is high
    hyperinflation_factor = 1.0
    if t_since_pickup < canopy.tau_A * 0.5:  # Early inflation phase
        # Hyperinflation more pronounced at high velocities
        # Simplified model: can overshoot by up to 20% during rapid inflation
        v_rel_mag = np.linalg.norm(canopy.v) if hasattr(canopy, 'v') else 0.0
        if v_rel_mag > 50.0:  # High velocity (m/s)
            # Peak overshoot at very beginning of inflation
            overshoot = 1.0 + 0.2 * np.exp(-xi * 3.0)  # Decays quickly
            hyperinflation_factor = min(overshoot, 1.2)  # Cap at 20% overshoot
    
    A_base *= hyperinflation_factor
    
    # Apply upstream reefing scales
    A_scale, _ = get_canopy_effective_scales(system, canopy.id, t, ramp_shape)
    return A_base * A_scale


def canopy_CD(
    t: float, 
    canopy: CanopyNode, 
    system: System, 
    ramp_shape: str = 'exp',
    t_pickup: Optional[float] = None,
    upstream_pickup_times: Optional[Dict[str, float]] = None
) -> float:
    """
    Effective canopy drag coefficient at time t (includes reefing scales).
    
    CD only increases after pickup when canopy starts inflating.
    
    Slack management: If upstream_canopy is set, downstream canopy won't inflate until
    upstream has been picked up and inflated.
    """
    # Must have deployment time/altitude reached
    if canopy.altitude_deploy is not None:
        # Altitude-based deployment: check if we've reached deployment altitude
        # CRITICAL: Validate canopy position is finite before using it
        if not np.all(np.isfinite(canopy.p)):
            return 0.0
        
        altitude_agl = canopy.p[2]  # z-coordinate is altitude (positive = above ground)
        
        # CRITICAL: Once a canopy has been picked up (t_pickup is set), it's deployed
        # and should continue inflating regardless of altitude changes (e.g., from separation)
        # Only check altitude BEFORE pickup (during deployment phase)
        if t_pickup is None:
            # Haven't picked up yet - check if we've reached deployment altitude
            # Deploy when descending past deployment altitude (altitude <= deploy_altitude)
            if altitude_agl > canopy.altitude_deploy:
                return 0.0  # Still above deployment altitude, not deployed yet
        # If t_pickup is set, canopy is already deployed - allow inflation regardless of altitude
    else:
        if t < canopy.td:
            return 0.0
    
    # Slack management: If this canopy depends on an upstream canopy, wait for it
    if canopy.upstream_canopy is not None and upstream_pickup_times is not None:
        upstream_id = canopy.upstream_canopy
        if upstream_id not in upstream_pickup_times:
            return 0.0
        upstream_pickup = upstream_pickup_times[upstream_id]
        if upstream_pickup is not None and t < upstream_pickup + 0.1:
            return 0.0
    
    # If pickup hasn't occurred yet, canopy is still packed (no drag)
    # CRITICAL: Check upstream_pickup_times if t_pickup is None (for multi-stage systems)
    if t_pickup is None and upstream_pickup_times is not None:
        # Try to get pickup time from upstream dict (for canopies that get pickup from edge)
        t_pickup = upstream_pickup_times.get(canopy.id)
    
    if t_pickup is None or t < t_pickup:
        return 0.0
    
    # Time since pickup (when inflation actually starts)
    t_since_pickup = t - t_pickup
    xi = t_since_pickup / max(canopy.tau_CD, 1e-9)
    CD_base = canopy.CD_inf * g_cd(xi)
    
    # Apply upstream reefing scales
    _, CD_scale = get_canopy_effective_scales(system, canopy.id, t, ramp_shape)
    return CD_base * CD_scale


def canopy_radius_from_area(A: float) -> float:
    """Compute canopy radius from area (assuming circular)"""
    return np.sqrt(max(A, 0.0) / np.pi)


def canopy_added_mass(
    t: float, 
    canopy: CanopyNode, 
    system: System, 
    ramp_shape: str = 'exp',
    t_pickup: Optional[float] = None,
    upstream_pickup_times: Optional[Dict[str, float]] = None
) -> float:
    """Added mass for canopy at time t (only after pickup/inflation starts)"""
    # CRITICAL: Check upstream_pickup_times if t_pickup is None (for multi-stage systems)
    if t_pickup is None and upstream_pickup_times is not None:
        t_pickup = upstream_pickup_times.get(canopy.id)
    
    A = canopy_area(t, canopy, system, ramp_shape, t_pickup, upstream_pickup_times)
    if A <= 0.0:
        return 0.0  # No added mass if canopy not inflated
    R = canopy_radius_from_area(A)
    if not np.isfinite(R) or R <= 0.0:
        return 0.0
    V = canopy.kappa * (R ** 3)
    if not np.isfinite(V) or V <= 0.0:
        return 0.0
    m_added = canopy.Ca * system.atmos.rho * V
    return max(0.0, m_added) if np.isfinite(m_added) else 0.0


def canopy_drag_force(
    t: float,
    canopy: CanopyNode,
    system: System,
    ramp_shape: str = 'exp',
    t_pickup: Optional[float] = None,
    upstream_pickup_times: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Aerodynamic drag force on canopy (inertial frame).
    
    Drag only occurs after pickup when canopy starts inflating.
    Orientation-dependent drag: drag vector opposes relative velocity,
    which accounts for body orientation effects in multi-parachute systems.
    """
    rho = system.atmos.rho
    v_air = system.atmos.v_air
    
    v_rel = v_air - canopy.v
    if not np.all(np.isfinite(v_rel)):
        return np.zeros(3)
    
    V = np.linalg.norm(v_rel)
    if not np.isfinite(V) or V < 1e-6:
        return np.zeros(3)
    
    A = canopy_area(t, canopy, system, ramp_shape, t_pickup, upstream_pickup_times)
    CD = canopy_CD(t, canopy, system, ramp_shape, t_pickup, upstream_pickup_times)
    
    if not np.isfinite(A * CD) or A * CD < 1e-9:
        return np.zeros(3)
    
    CDA = CD * A
    F = -0.5 * rho * CDA * V * v_rel
    
    if not np.all(np.isfinite(F)):
        return np.zeros(3)
    
    return F


# ============================================================================
# Line edge physics
# ============================================================================

def line_stiffness(edge: LineEdge, x: float, t: float, ramp_shape: str = 'exp') -> float:
    """Compute stiffness k(x) = k0 + k1 * x^alpha with reefing"""
    k0_eff = get_effective_reefing_param(t, edge.k0, edge.reefing, 'k0', ramp_shape)
    
    # Clamp x >= 0
    x_clamp = max(0.0, float(x)) if np.isfinite(x) else 0.0
    
    try:
        if edge.k1 == 0.0 or edge.alpha == 0.0:
            term = 0.0
        else:
            term = x_clamp ** edge.alpha
    except Exception:
        term = 0.0
    
    val = k0_eff + edge.k1 * term
    
    if not np.isfinite(val):
        return float(k0_eff)
    
    return float(val)


def tension_from_extension(
    edge: LineEdge,
    x: float,
    xdot: float,
    t: float,
    ramp_shape: str = 'exp'
) -> float:
    """Kelvin-Voigt tension: T = max(0, [k0 + k1*x^alpha]*x + c*xdot + Tpre)"""
    # Validate inputs
    if not np.isfinite(x) or not np.isfinite(xdot):
        return 0.0
    
    # Get effective parameters with reefing
    L0_eff = get_effective_reefing_param(t, edge.L0, edge.reefing, 'L0', ramp_shape)
    c_eff = get_effective_reefing_param(t, edge.c, edge.reefing, 'c', ramp_shape)
    
    # Validate parameters
    if not np.isfinite(L0_eff) or not np.isfinite(c_eff):
        return 0.0
    
    # Clamp x >= 0
    x_clamp = max(0.0, float(x))
    
    # Slack condition: if x <= 0, line is slack and T = 0 (no damping when slack!)
    if x_clamp <= 0.0:
        return 0.0
    
    # Line is taut (x > 0), compute normal tension
    k = line_stiffness(edge, x_clamp, t, ramp_shape)
    if not np.isfinite(k) or k < 0.0:
        k = edge.k0  # Fallback to base stiffness
    
    # Clamp xdot to prevent unrealistic damping forces
    xdot_clamp = np.clip(float(xdot), -1000.0, 1000.0)  # ±1000 m/s max extension rate
    
    T = k * x_clamp + c_eff * xdot_clamp + edge.T_pre
    
    # Validate T is finite
    if not np.isfinite(T):
        return 0.0
    
    # Enforce material breaking strength (physical limit, not artificial cap)
    T = min(T, edge.T_max)
    
    return max(0.0, T)


def get_edge_geometry(
    edge: LineEdge,
    system: System,
    bodies: Dict[str, BodyNode],
    canopies: Dict[str, CanopyNode]
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float]:
    """
    Compute edge geometry: positions, velocities, length, unit vector, extension, extension rate.
    
    Returns:
        (p_minus, p_plus, L, u, x, xdot)
    """
    from .model import parse_node_ref
    
    # Get minus node position/velocity
    n_minus_id, anchor_name = parse_node_ref(edge.n_minus)
    if anchor_name is not None:
        # Body anchor
        body = bodies[n_minus_id]
        R = quat_to_rotm(body.q)
        r_anchor_B = body.anchors_B[anchor_name]
        p_minus = body.r + R @ r_anchor_B
        # Anchor velocity: v_a = v_b + omega_b × (R r_a^B)
        v_minus = body.v + np.cross(body.w, R @ r_anchor_B)
    elif n_minus_id in canopies:
        # Canopy
        canopy = canopies[n_minus_id]
        p_minus = canopy.p
        v_minus = canopy.v
    else:
        raise ValueError(f"Unknown n_minus node: {edge.n_minus}")
    
    # Get plus node position/velocity
    n_plus_id, _ = parse_node_ref(edge.n_plus)
    if n_plus_id in bodies:
        # Body (though unusual for n_plus)
        body = bodies[n_plus_id]
        p_plus = body.r
        v_plus = body.v
    elif n_plus_id in canopies:
        # Canopy
        canopy = canopies[n_plus_id]
        p_plus = canopy.p
        v_plus = canopy.v
    else:
        raise ValueError(f"Unknown n_plus node: {edge.n_plus}")
    
    # Compute geometry
    dp = p_plus - p_minus
    L = np.linalg.norm(dp)
    
    if L < 1e-9:
        u = np.array([0.0, 0.0, 0.0])
        x = 0.0
        xdot = 0.0
    else:
        u = dp / L
        dv = v_plus - v_minus
        xdot = float(np.dot(dv, u))
        
        # Get effective L0 with reefing (use current time approximation)
        # In actual simulation, t will be passed from engine
        L0_eff = edge.L0  # Will be overridden by reefing evaluation in engine
        x = max(0.0, L - L0_eff)
    
    return p_minus, p_plus, L, u, x, xdot


# ============================================================================
# Body anchor utilities
# ============================================================================

def get_anchor_pose(
    body: BodyNode,
    anchor_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get anchor position and velocity in inertial frame.
    
    Returns:
        (p_anchor, v_anchor)
    """
    # Validate inputs
    if not np.all(np.isfinite(body.r)) or not np.all(np.isfinite(body.v)):
        return np.zeros(3), np.zeros(3)
    
    R = quat_to_rotm(body.q)
    if anchor_name not in body.anchors_B:
        return body.r.copy(), body.v.copy()
    
    r_anchor_B = body.anchors_B[anchor_name]
    if not np.all(np.isfinite(r_anchor_B)):
        return body.r.copy(), body.v.copy()
    
    p_anchor = body.r + R @ r_anchor_B
    
    # Validate angular velocity
    if not np.all(np.isfinite(body.w)):
        v_anchor = body.v.copy()
    else:
        v_anchor = body.v + np.cross(body.w, R @ r_anchor_B)
    
    # Ensure results are finite
    if not np.all(np.isfinite(p_anchor)):
        p_anchor = body.r.copy()
    if not np.all(np.isfinite(v_anchor)):
        v_anchor = body.v.copy()
    
    return p_anchor, v_anchor
