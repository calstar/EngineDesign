"""Integrator + event manager + observers"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
from .model import System, BodyNode, CanopyNode
from .physics import (
    quat_normalize, quat_to_rotm, quat_omega_mat,
    get_effective_reefing_param, canopy_area, canopy_CD, canopy_added_mass,
    canopy_drag_force, tension_from_extension, get_anchor_pose, line_stiffness
)
from .model import parse_node_ref as _parse_node_ref


def _perform_body_separation(
    system: System,
    state: SimulationState,
    body_id: str,
    template_state: SimulationState,
    t: float
):
    """
    Split body into two halves when separation signal triggers (after lag time).
    
    Separation is triggered by altitude/time, not by pickup events.
    Bodies separate with black powder charge force, then freefall until lines go taut.
    """
    body = state.bodies[body_id]
    
    # Separation direction: use body's longitudinal axis (typically +z in body frame)
    # This is the axis along which the charge separates the body
    R = quat_to_rotm(body.q)
    sep_dir_body = np.array([0, 0, 1])  # Body frame: +z axis (longitudinal)
    sep_dir = R @ sep_dir_body  # Transform to inertial frame
    
    # Create second body (half)
    body_half_id = f"{body_id}_half"
    
    # Split mass and inertia (conservative: each half gets half)
    m_half = body.m * 0.5
    I_half = body.I_body * 0.5  # Simplified: each half has half inertia
    
    # Position: offset slightly along separation direction
    r_offset = 0.1  # 10 cm separation initially
    r_half = body.r + sep_dir * r_offset
    
    # MOMENTUM CONSERVATION with BLACK POWDER CHARGE:
    # Black powder charge applies equal and opposite forces to both halves
    # F_charge on each half = separation_charge_force (if specified)
    # Otherwise use separation_v_mag (velocity-based)
    v_cm = body.v  # Original body velocity (center of mass)
    
    # Black powder charge: separation velocity specified directly
    # v_sep is the relative velocity between halves (6 m/s typical)
    # Each half gets: v = v_cm ± (v_sep/2) * sep_dir
    # But for equal masses: v1 = v_cm + v_sep*sep_dir, v2 = v_cm - v_sep*sep_dir
    # This conserves momentum: m_half*(v_cm + v_sep) + m_half*(v_cm - v_sep) = m_total*v_cm
    v_sep = body.separation_v_mag  # Relative separation velocity (m/s)
    
    # Halves move apart: v1 = v_cm + v_sep*sep_dir, v2 = v_cm - v_sep*sep_dir
    # This conserves momentum: m_half*(v_cm + v_sep) + m_half*(v_cm - v_sep) = m_total*v_cm
    v_half = v_cm + sep_dir * v_sep
    v_original = v_cm - sep_dir * v_sep
    
    # ANGULAR MOMENTUM CONSERVATION:
    # Pre-separation: L_total = I_total * w_cm + r_cm × (m_total * v_cm)
    # Post-separation: L1 + L2 = L_total
    # 
    # For each half: L_i = I_i * w_i + r_i × (m_i * v_i)
    # where r_i is position of half relative to original CM
    w_cm = body.w.copy()  # Original angular velocity (body frame)
    I_total = body.I_body  # Original total inertia
    
    # Calculate angular momentum before separation (in body frame)
    # L_body = I_total @ w_cm (in body frame, no position term since r_cm = 0)
    L_total_body = I_total @ w_cm
    
    # After separation: each half has half the inertia
    # L1_body + L2_body = L_total_body
    # Each half gets: L_i_body = I_half @ w_i_body
    # So: I_half @ w1 + I_half @ w2 = I_total @ w_cm
    # Since I_half = I_total / 2, we get: (I_total/2) @ (w1 + w2) = I_total @ w_cm
    # Therefore: w1 + w2 = 2 * w_cm
    #
    # But we also need to account for separation-induced angular momentum:
    # The separation velocity v_sep creates relative motion, but since separation is along
    # the body axis (sep_dir), and halves are symmetric, the angular momentum from
    # separation is zero (r1 × m1*v_sep + r2 × m2*(-v_sep) = 0 for symmetric separation)
    #
    # So: w1 + w2 = 2 * w_cm
    # We split evenly: w1 = w_cm + delta_w, w2 = w_cm - delta_w
    # where delta_w accounts for any asymmetry in separation
    
    # For symmetric separation along body axis, halves get same angular velocity
    # But we can add a small separation-induced component if needed
    # For now, assume symmetric: both halves get the same angular velocity as original
    # (This conserves angular momentum: I_half * w_cm + I_half * w_cm = I_total * w_cm ✓)
    w_half = w_cm.copy()
    w_original = w_cm.copy()
    
    # Optional: Add small separation-induced spin if separation is not perfectly symmetric
    # This would come from imperfections in charge placement, but for now we assume symmetric
    
    # Update original body
    body.v = v_original
    body.w = w_original
    
    # Create new body half
    body_half = BodyNode(
        id=body_half_id,
        m=m_half,
        I_body=I_half,
        r=r_half,
        v=v_half,
        q=body.q.copy(),  # Same orientation initially
        w=w_half,
        anchors_B={},  # No anchors on half (can be configured later)
        aero_force=body.aero_force,
        aero_moment=body.aero_moment,
        aero_area=body.aero_area,
        aero_CD=body.aero_CD,
        separation_signal_altitude=None  # Half doesn't separate further
    )
    
    # Add to system and state
    system.bodies[body_half_id] = body_half
    state.bodies[body_half_id] = body_half
    template_state.bodies[body_half_id] = body_half
    
    # Update original body mass (now it's a half too)
    body.m = m_half
    body.I_body = I_half
    
    # CRITICAL: Remove/break structural links that connect separated halves
    # When separation occurs, structural links between separated components should break
    # Find all edges connected to this body and remove structural links
    edges_to_remove = []
    for edge_id, edge in system.edges.items():
        # Check if this edge connects to the separated body
        n_minus_id, anchor_name = _parse_node_ref(edge.n_minus)
        n_plus_id, _ = _parse_node_ref(edge.n_plus)
        
        # If edge connects the separated body to another body, and it's a structural link
        # (high stiffness indicates structural link)
        is_structural = edge.k0 > 100000  # Structural links have k0 > 100 kN/m
        
        if is_structural:
            if (n_minus_id == body_id or n_plus_id == body_id or 
                n_minus_id == body_half_id or n_plus_id == body_half_id):
                # This structural link connects to the separated body - mark for removal
                edges_to_remove.append(edge_id)
    
    # Remove structural links (they break during separation)
    for edge_id in edges_to_remove:
        del system.edges[edge_id]


def _compute_pickup_jolt(
    state: SimulationState,
    system: System,
    edge_id: str,
    edge,
    t_pickup: float,
    pickup_times_dict: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute instantaneous jolt/shock when line goes taut during hyperinflation.
    
    Physics:
    - At pickup, line goes from slack to taut instantaneously
    - Relative velocity between nodes creates tension impulse
    - During hyperinflation, canopy area is maximum (peak shock)
    - Jolt: F_jolt = m * a_jolt = tension force at moment of tautening
    
    Returns:
        Dict with jolt magnitudes for each affected body/canopy
    """
    from .physics import get_anchor_pose, canopy_area, canopy_CD, canopy_added_mass
    
    n_minus_id, anchor_name = _parse_node_ref(edge.n_minus)
    if anchor_name is not None:
        body = state.bodies[n_minus_id]
        p_minus, v_minus = get_anchor_pose(body, anchor_name)
        m_minus = body.m
    elif n_minus_id in state.canopies:
        canopy_minus = state.canopies[n_minus_id]
        p_minus = canopy_minus.p
        v_minus = canopy_minus.v
        t_pickup_minus = pickup_times_dict.get(n_minus_id)
        m_added_minus = canopy_added_mass(t_pickup, canopy_minus, system, 'exp', t_pickup_minus, pickup_times_dict)
        m_minus = canopy_minus.m_canopy + m_added_minus
    else:
        return {}
    
    n_plus_id, _ = _parse_node_ref(edge.n_plus)
    if n_plus_id in state.bodies:
        body_plus = state.bodies[n_plus_id]
        p_plus = body_plus.r
        v_plus = body_plus.v
        m_plus = body_plus.m
    elif n_plus_id in state.canopies:
        canopy_plus = state.canopies[n_plus_id]
        p_plus = canopy_plus.p
        v_plus = canopy_plus.v
        t_pickup_plus = pickup_times_dict.get(n_plus_id)
        m_added_plus = canopy_added_mass(t_pickup, canopy_plus, system, 'exp', t_pickup_plus, pickup_times_dict)
        m_plus = canopy_plus.m_canopy + m_added_plus
    else:
        return {}
    
    # Line direction at moment of tautening
    dp = p_plus - p_minus
    L = np.linalg.norm(dp)
    if L < 1e-6:
        return {}
    u = dp / L
    
    # Relative velocity along line direction
    dv = v_plus - v_minus
    v_rel = float(np.dot(dv, u))
    
    # At moment of tautening, initial tension from extension rate
    # T_jolt = c * v_rel (damping term dominates initially)
    L0_eff = edge.L0
    c_eff = edge.c
    T_jolt = max(0.0, c_eff * v_rel) if v_rel > 0 else 0.0
    
    # Add hyperinflation drag contribution (peak during inflation overshoot)
    if n_plus_id in state.canopies:
        canopy = state.canopies[n_plus_id]
        t_pickup_canopy = pickup_times_dict.get(n_plus_id, t_pickup)
        A_hyper = canopy_area(t_pickup, canopy, system, 'exp', t_pickup_canopy, pickup_times_dict)
        CD_hyper = canopy_CD(t_pickup, canopy, system, 'exp', t_pickup_canopy, pickup_times_dict)
        
        v_air = system.atmos.v_air
        v_rel_aero = canopy.v - v_air
        v_rel_mag = np.linalg.norm(v_rel_aero)
        if v_rel_mag > 1e-6:
            v_rel_hat = v_rel_aero / v_rel_mag
            rho = system.atmos.rho
            F_drag_hyper = 0.5 * rho * v_rel_mag**2 * A_hyper * CD_hyper * (-v_rel_hat)
            F_drag_along_line = float(np.dot(F_drag_hyper, u))
            T_jolt += max(0.0, F_drag_along_line)
    
    # Store jolt in event (will be added to event.extra)
    return {'jolt_magnitude': float(T_jolt), 'v_rel': float(v_rel)}


@dataclass
class SimulationState:
    """Complete simulation state at time t"""
    bodies: Dict[str, BodyNode]
    canopies: Dict[str, CanopyNode]
    t: float


@dataclass
class Event:
    """Simulation event"""
    event_type: str  # 'pickup', 'deploy_on', 'reef_on'
    id: str  # edge_id or canopy_id
    t_event: float
    extra: Dict = field(default_factory=dict)


class EventManager:
    """Manages event detection and logging"""
    
    def __init__(self):
        self.events: List[Event] = []
        self.pickup_detected: Dict[str, bool] = {}
        self.deploy_detected: Dict[str, bool] = {}
        self.separation_detected: Dict[str, bool] = {}
        self.separation_signal_sent: Dict[str, float] = {}
    
    def reset(self):
        """Reset event tracking"""
        self.events.clear()
        self.pickup_detected.clear()
        self.deploy_detected.clear()
        self.separation_detected.clear()
        self.separation_signal_sent.clear()
    
    def check_pickup(
        self,
        edge_id: str,
        t: float,
        L: float,
        L0_eff: float,
        system: System,
        state: SimulationState,
        L_prev: Optional[float] = None,
        t_prev: Optional[float] = None
    ) -> Optional[Event]:
        """
        Check for pickup event (first L <= L0 -> L > L0 transition).
        
        CRITICAL: Pickup can only occur if:
        1. The line goes from slack to taut
        2. If the edge connects to a canopy, that canopy must have deployed (reached deployment altitude)
        """
        if edge_id in self.pickup_detected:
            return None
        
        # Check if this edge connects to a canopy (either n_plus or n_minus), and if so, verify it has deployed
        # Pickup can only occur if the canopy has deployed (reached deployment altitude)
        edge = system.edges[edge_id]
        n_plus_id, _ = _parse_node_ref(edge.n_plus)
        n_minus_id, _ = _parse_node_ref(edge.n_minus)
        
        # Check both ends - either could be a canopy
        for node_id in [n_plus_id, n_minus_id]:
            if node_id in system.canopies:
                canopy_id = node_id
                canopy = system.canopies[canopy_id]
                
                # Get current canopy state (may have moved)
                if canopy_id in state.canopies:
                    canopy_current = state.canopies[canopy_id]
                else:
                    canopy_current = canopy
                
                # Check if canopy has deployed (altitude-based or time-based)
                if canopy.altitude_deploy is not None:
                    # Altitude-based: must have reached deployment altitude (descending past it)
                    altitude_agl = canopy_current.p[2]  # z-coordinate is altitude
                    if altitude_agl > canopy.altitude_deploy:
                        # Canopy hasn't deployed yet - no pickup allowed
                        return None
                else:
                    # Time-based: must have reached deployment time
                    if t < canopy.td:
                        # Canopy hasn't deployed yet - no pickup allowed
                        return None
        
        if L_prev is not None and L0_eff > 0:
            was_slack = L_prev <= L0_eff
            is_taut = L > L0_eff
            
            if was_slack and is_taut:
                if t_prev is not None and t_prev < t:
                    t_interp = t_prev + (t - t_prev) * (L0_eff - L_prev) / (L - L_prev)
                else:
                    t_interp = t
                
                self.pickup_detected[edge_id] = True
                return Event(
                    event_type='pickup',
                    id=edge_id,
                    t_event=t_interp,
                    extra={'L_at_event': float(L0_eff), 'step_idx': -1}
                )
        
        return None
    
    def check_deploy(self, canopy_id: str, canopy: 'CanopyNode', state: 'SimulationState', t: float, dt: float) -> Optional[Event]:
        """Check for deployment event (time-based or altitude-based)"""
        if canopy_id in self.deploy_detected:
            return None
        
        # Check altitude-based deployment first (if set, overrides time-based)
        if canopy.altitude_deploy is not None:
            # Get altitude AGL from canopy position (assuming z-up coordinate system)
            altitude_agl = canopy.p[2]  # z-coordinate is altitude
            if altitude_agl <= canopy.altitude_deploy:
                self.deploy_detected[canopy_id] = True
                return Event(
                    event_type='deploy_on',
                    id=canopy_id,
                    t_event=t,
                    extra={'altitude_agl': float(altitude_agl), 'dt': float(dt)}
                )
        else:
            # Time-based deployment
            if t >= canopy.td:
                self.deploy_detected[canopy_id] = True
                return Event(
                    event_type='deploy_on',
                    id=canopy_id,
                    t_event=t,
                    extra={'dt': float(dt)}
                )
        
        return None
    
    def check_reefing(
        self,
        edge_id: str,
        t: float,
        t_prev: float,
        stages: List,
        dt: float
    ) -> List[Event]:
        """Check for reefing events (t crosses t_on)"""
        events = []
        for i, stage in enumerate(stages):
            if t_prev < stage.t_on <= t:
                events.append(Event(
                    event_type='reef_on',
                    id=edge_id,
                    t_event=float(stage.t_on),
                    extra={'stage_index': i, 'dt': float(dt)}
                ))
        return events
    
    def check_separation(
        self,
        body_id: str,
        body: BodyNode,
        state: SimulationState,
        t: float,
        t_prev: float
    ) -> Optional[Event]:
        """
        Check for body separation event (altitude-based trigger with lag time).
        
        Workflow:
        1. Signal sent at separation_signal_altitude
        2. Lag time (signal delay + charge delay + separation delay)
        3. Separation occurs at actual altitude (signal_altitude - altitude lost during lag)
        """
        if body.separation_signal_altitude is None:
            return None
        
        if body_id in self.separation_detected:
            return None
        
        # Get current altitude AGL (z-coordinate, assuming z-up)
        altitude_agl = body.r[2]
        
        signal_key = f"{body_id}_signal"
        if signal_key not in self.separation_signal_sent:
            # Check if we've crossed signal altitude (descending past it)
            if altitude_agl <= body.separation_signal_altitude:
                # Signal sent now
                self.separation_signal_sent[signal_key] = t
                return None  # Signal sent, but separation hasn't occurred yet
        
        # Check if lag time has passed since signal was sent
        if signal_key in self.separation_signal_sent:
            signal_time = self.separation_signal_sent[signal_key]
            if t >= signal_time + body.separation_lag_time:
                # Separation occurs now
                self.separation_detected[body_id] = True
                return Event(
                    event_type='separation',
                    id=body_id,
                    t_event=t,
                    extra={
                        'signal_altitude': float(body.separation_signal_altitude),
                        'separation_altitude': float(altitude_agl),
                        'lag_time': float(body.separation_lag_time)
                    }
                )
        
        return None


def pack_state(state: SimulationState) -> np.ndarray:
    """Pack simulation state into flat array for integrator"""
    arrs = []
    for body_id in sorted(state.bodies.keys()):
        body = state.bodies[body_id]
        arrs.append(body.r)
        arrs.append(body.v)
        arrs.append(body.q)
        arrs.append(body.w)
    for canopy_id in sorted(state.canopies.keys()):
        canopy = state.canopies[canopy_id]
        arrs.append(canopy.p)
        arrs.append(canopy.v)
    return np.concatenate(arrs)


def unpack_state(
    vec: np.ndarray,
    system: System,
    template: SimulationState
) -> SimulationState:
    """Unpack flat array into simulation state"""
    idx = 0
    bodies = {}
    
    # Use system.bodies to get all bodies (including dynamically added ones)
    all_body_ids = sorted(set(list(template.bodies.keys()) + list(system.bodies.keys())))
    for body_id in all_body_ids:
        if body_id not in template.bodies:
            # Dynamically added body - create from system
            if body_id in system.bodies:
                body_template = system.bodies[body_id]
            else:
                continue
        else:
            body_template = template.bodies[body_id]
        r = vec[idx:idx+3]; idx += 3
        v = vec[idx:idx+3]; idx += 3
        q = quat_normalize(vec[idx:idx+4]); idx += 4
        w = vec[idx:idx+3]; idx += 3
        
        bodies[body_id] = BodyNode(
            id=body_id,
            m=body_template.m,
            I_body=body_template.I_body,
            r=r, v=v, q=q, w=w,
            anchors_B=body_template.anchors_B,
            aero_force=body_template.aero_force,
            aero_moment=body_template.aero_moment,
            separation_signal_altitude=getattr(body_template, 'separation_signal_altitude', None),
            separation_lag_time=getattr(body_template, 'separation_lag_time', 0.0),
            separation_v_mag=getattr(body_template, 'separation_v_mag', 6.0),
            aero_area=getattr(body_template, 'aero_area', None),
            aero_CD=getattr(body_template, 'aero_CD', None)
        )
    
    canopies = {}
    for canopy_id in sorted(template.canopies.keys()):
        canopy_template = template.canopies[canopy_id]
        p = vec[idx:idx+3]; idx += 3
        v = vec[idx:idx+3]; idx += 3
        
        canopies[canopy_id] = CanopyNode(
            id=canopy_id,
            p=p, v=v,
            m_canopy=canopy_template.m_canopy,
            A_inf=canopy_template.A_inf,
            CD_inf=canopy_template.CD_inf,
            td=canopy_template.td,
            altitude_deploy=getattr(canopy_template, 'altitude_deploy', None),
            tau_A=canopy_template.tau_A,
            tau_CD=canopy_template.tau_CD,
            Ca=canopy_template.Ca,
            kappa=canopy_template.kappa,
            upstream_canopy=getattr(canopy_template, 'upstream_canopy', None)
        )
    
    return SimulationState(bodies=bodies, canopies=canopies, t=template.t)


def compute_rhs(
    t: float,
    state: SimulationState,
    system: System,
    ramp_shape: str = 'exp',
    pickup_times: Optional[Dict[str, float]] = None
) -> SimulationState:
    """Compute right-hand side of ODEs (derivatives)"""
    gvec = np.array([0.0, 0.0, system.atmos.g])
    
    F_body = {bid: np.zeros(3) for bid in state.bodies.keys()}
    M_body = {bid: np.zeros(3) for bid in state.bodies.keys()}
    F_canopy = {cid: np.zeros(3) for cid in state.canopies.keys()}
    
    # Process edges (use list() to avoid issues if edges are removed during iteration)
    for edge_id, edge in list(system.edges.items()):
        L0_eff = get_effective_reefing_param(t, edge.L0, edge.reefing, 'L0', ramp_shape)
        
        n_minus_id, anchor_name = _parse_node_ref(edge.n_minus)
        if anchor_name is not None:
            body = state.bodies[n_minus_id]
            p_minus, v_minus = get_anchor_pose(body, anchor_name)
        elif n_minus_id in state.canopies:
            canopy = state.canopies[n_minus_id]
            p_minus = canopy.p
            v_minus = canopy.v
        else:
            raise ValueError(f"Unknown n_minus: {edge.n_minus}")
        
        n_plus_id, _ = _parse_node_ref(edge.n_plus)
        if n_plus_id in state.bodies:
            body = state.bodies[n_plus_id]
            p_plus = body.r
            v_plus = body.v
        elif n_plus_id in state.canopies:
            canopy = state.canopies[n_plus_id]
            p_plus = canopy.p
            v_plus = canopy.v
        else:
            raise ValueError(f"Unknown n_plus: {edge.n_plus}")
        
        # Safety check: ensure positions are finite
        if not (np.all(np.isfinite(p_minus)) and np.all(np.isfinite(p_plus))):
            T = 0.0
            u = np.array([0.0, 0.0, 0.0])
        elif not np.all(np.isfinite(v_minus)) or not np.all(np.isfinite(v_plus)):
            # If velocities are invalid, still compute tension but use zero velocity difference
            dp = p_plus - p_minus
            L = np.linalg.norm(dp)
            if L < 1e-6:
                T = 0.0
                u = np.array([0.0, 0.0, 0.0])
            else:
                u = dp / L
                x = max(0.0, L - L0_eff)
                T = tension_from_extension(edge, x, 0.0, t, ramp_shape)  # Use zero xdot
                T = min(T, edge.T_max) if np.isfinite(T) else 0.0
                T = max(0.0, T)
        else:
            dp = p_plus - p_minus
            L = np.linalg.norm(dp)
            
            # Maximum line length: L0 + max extension (based on breaking strength)
            L_max = L0_eff + 2.0  # Maximum physical length (50ft + 2m extension)
            
            # Early check: if L is way beyond physical maximum, something is wrong
            # Don't compute forces with invalid geometry - position correction will fix it
            if not np.isfinite(L) or L > 100.0:  # 100m is way beyond 17.24m max
                T = 0.0
                u = np.array([0.0, 0.0, 0.0])
            elif L < 1e-6:
                T = 0.0
                u = np.array([0.0, 0.0, 0.0])
            else:
                u = dp / L
                dv = v_plus - v_minus
                xdot = float(np.dot(dv, u))
                
                # Normal extension calculation
                x = max(0.0, L - L0_eff)
                
                # Compute normal tension from material properties
                T = tension_from_extension(edge, x, xdot, t, ramp_shape)
                
                # Ensure T is finite before proceeding
                if not np.isfinite(T):
                    T = 0.0
                
                # Apply constraint force if line exceeds maximum extension
                # Maximum realistic extension: 5% for Kevlar (conservative)
                x_max_realistic = L0_eff * 0.05
                
                if x > x_max_realistic:
                    # Line has exceeded realistic extension - material has failed
                    # Apply breaking strength tension (line has broken, but we keep it for now)
                    # Use much more conservative constraint to prevent numerical explosion
                    violation = x - x_max_realistic
                    
                    # CRITICAL: Cap violation to prevent unbounded growth
                    # If violation is too large, something is wrong - cap it aggressively
                    violation = min(violation, L0_eff * 0.1)  # Cap violation at 10% of L0
                    
                    # Use moderate constraint force (not explosive)
                    # Constraint should resist further extension, but not be huge
                    k_penalty = min(edge.k0 * 10.0, 1e5)  # Max 100 kN/m (much more conservative)
                    T_constraint = k_penalty * violation
                    
                    # Add moderate damping
                    c_penalty = min(edge.c * 10.0, 1000.0)  # Max 1 kN/(m/s) (much more conservative)
                    if xdot > 0:  # Only damp if extending
                        T_constraint += c_penalty * xdot
                    
                    # Total applied force: breaking strength + moderate constraint
                    T = edge.T_max + T_constraint
                    
                    # CRITICAL: Cap total tension aggressively to prevent numerical explosion
                    # Max 5x breaking strength - very conservative to prevent explosion
                    T_max_constraint = edge.T_max * 5.0
                    T = min(T, T_max_constraint)
                    
                    # Additional safety: if T is still unreasonably high, something is very wrong
                    # In this case, the line has effectively broken - set tension to breaking strength
                    if T > 1e6:  # If tension exceeds 1 MN, something is very wrong
                        T = edge.T_max
                    
                    # Final safety: absolute maximum cap
                    T = min(T, 1e6)  # Never exceed 1 MN (1,000 kN) - this is already extremely high
                    
                    # Ensure T is finite
                    if not np.isfinite(T):
                        T = edge.T_max
                else:
                    # Normal operation: ensure tension doesn't exceed breaking strength
                    T = min(T, edge.T_max)
                
                # Final safety: ensure tension is non-negative
                T = max(0.0, T)
                
                if not np.isfinite(T) or not np.all(np.isfinite(u)):
                    T = 0.0
                    u = np.array([0.0, 0.0, 0.0])
        
        if T > 0 and np.all(np.isfinite(u)):
            F_tension = T * u
            
            # CRITICAL: Validate tension force before applying
            if not np.all(np.isfinite(F_tension)):
                F_tension = np.zeros(3)
            
            # Cap individual force components to prevent explosion
            max_force_component = 1e6  # 1 MN per component
            F_tension = np.clip(F_tension, -max_force_component, max_force_component)
            
            # Apply to plus node
            if n_plus_id in state.bodies:
                F_body[n_plus_id] += F_tension
            elif n_plus_id in state.canopies:
                F_canopy[n_plus_id] += F_tension  # Tension pulls plus node toward minus
            
            # Apply to minus node
            if anchor_name is not None:
                body = state.bodies[n_minus_id]
                F_body[n_minus_id] -= F_tension  # Equal and opposite
                R = quat_to_rotm(body.q)
                if np.all(np.isfinite(R)):
                    r_anchor_B = body.anchors_B[anchor_name]
                    F_body_frame = R.T @ (-F_tension)  # Negative for moment computation
                    if np.all(np.isfinite(F_body_frame)) and np.all(np.isfinite(r_anchor_B)):
                        M_body[n_minus_id] += np.cross(r_anchor_B, F_body_frame)
            elif n_minus_id in state.canopies:
                F_canopy[n_minus_id] -= F_tension  # Opposite direction
    
    # Add canopy drag and gravity
    # Get pickup time for each canopy (inflation only starts after pickup)
    # CRITICAL: Ensure pickup_times_dict is always a valid dict (not None)
    pickup_times_dict = pickup_times if pickup_times is not None else {}
    for canopy_id, canopy in state.canopies.items():
        # Get pickup time - check both direct lookup and upstream_pickup_times fallback
        t_pickup = pickup_times_dict.get(canopy_id)
        
        # Validate canopy state
        if not np.all(np.isfinite(canopy.p)) or not np.all(np.isfinite(canopy.v)):
            continue  # Skip invalid canopy states
        
        F_drag = canopy_drag_force(t, canopy, system, ramp_shape, t_pickup, pickup_times_dict)
        
        # Validate drag force
        if not np.all(np.isfinite(F_drag)):
            F_drag = np.zeros(3)
        
        # Gravity on canopy
        if np.isfinite(canopy.m_canopy) and canopy.m_canopy > 0:
            F_canopy[canopy_id] += canopy.m_canopy * gvec
        
        # ORIENTATION-DEPENDENT DRAG: The drag force acts on the canopy itself
        # The canopy is connected to the body via a line, so the tension in the line
        # transfers the drag force to the body. We should NOT apply drag directly to body.
        # Instead, the drag accelerates the canopy, and the tension in the line transfers
        # that force to the body. The tension calculation already handles this.
        # 
        # Apply drag force to canopy (this will create tension in the line)
        F_canopy[canopy_id] += F_drag
    
    # Add body forces
    for body_id, body in state.bodies.items():
        F_body[body_id] += body.m * gvec
        
        # Body aerodynamic drag (for freefall dynamics)
        if body.aero_area is not None and body.aero_CD is not None:
            rho = system.atmos.rho
            v_air = system.atmos.v_air
            v_rel = v_air - body.v
            if np.all(np.isfinite(v_rel)):
                V = np.linalg.norm(v_rel)
                if np.isfinite(V) and V > 1e-6:
                    v_rel_hat = v_rel / V
                    F_drag_body = 0.5 * rho * V**2 * body.aero_area * body.aero_CD * (-v_rel_hat)
                    # Cap drag force to prevent explosion
                    if np.all(np.isfinite(F_drag_body)):
                        F_drag_body = np.clip(F_drag_body, -1e6, 1e6)  # Cap at 1 MN per component
                        F_body[body_id] += F_drag_body
        
        # Custom aerodynamic force/moment (if provided)
        if body.aero_force is not None:
            F_aero = body.aero_force(state, system, body_id)
            if F_aero is not None and np.all(np.isfinite(F_aero)):
                F_aero = np.clip(F_aero, -1e6, 1e6)  # Cap at 1 MN per component
                F_body[body_id] += F_aero
        if body.aero_moment is not None:
            M_aero = body.aero_moment(state, system, body_id)
            if M_aero is not None and np.all(np.isfinite(M_aero)):
                M_aero = np.clip(M_aero, -1e6, 1e6)  # Cap at 1 MN·m per component
                M_body[body_id] += M_aero
    
    # Compute accelerations
    body_derivs = {}
    for body_id, body in state.bodies.items():
        # Validate body mass
        if not np.isfinite(body.m) or body.m <= 0:
            body.m = 1e-3  # Fallback to small mass
        
        # Validate forces and cap to prevent explosion
        if not np.all(np.isfinite(F_body[body_id])):
            F_body[body_id] = np.zeros(3)
        else:
            F_body[body_id] = np.clip(F_body[body_id], -1e6, 1e6)  # Cap at 1 MN per component
        
        if not np.all(np.isfinite(M_body[body_id])):
            M_body[body_id] = np.zeros(3)
        else:
            M_body[body_id] = np.clip(M_body[body_id], -1e6, 1e6)  # Cap at 1 MN·m per component
        
        a = F_body[body_id] / max(body.m, 1e-9)
        
        # Validate linear acceleration
        if not np.all(np.isfinite(a)):
            a = np.zeros(3)
        
        # Validate inertia tensor
        I = body.I_body
        if not np.all(np.isfinite(I)) or np.linalg.det(I) < 1e-12:
            # Fallback to identity scaled by mass
            I = np.eye(3) * body.m * 0.1
        
        # Validate angular velocity
        if not np.all(np.isfinite(body.w)):
            body.w = np.zeros(3)
        
        Iomega = I @ body.w
        omega_cross_Iomega = np.cross(body.w, Iomega)
        
        # Validate moments
        if not np.all(np.isfinite(omega_cross_Iomega)):
            omega_cross_Iomega = np.zeros(3)
        
        M_net = M_body[body_id] - omega_cross_Iomega
        if not np.all(np.isfinite(M_net)):
            M_net = np.zeros(3)
        
        try:
            alph = np.linalg.solve(I, M_net)
        except np.linalg.LinAlgError:
            # Fallback if solve fails
            alph = np.zeros(3)
        
        # Validate angular acceleration
        if not np.all(np.isfinite(alph)):
            alph = np.zeros(3)
        
        # Validate quaternion
        if not np.all(np.isfinite(body.q)) or len(body.q) != 4:
            body.q = np.array([1.0, 0.0, 0.0, 0.0])
        
        qdot = quat_omega_mat(body.q) @ body.w
        
        # Validate quaternion derivative
        if not np.all(np.isfinite(qdot)):
            qdot = np.zeros(4)
        
        # Validate velocity
        if not np.all(np.isfinite(body.v)):
            body.v = np.zeros(3)
        
        body_derivs[body_id] = {'r': body.v, 'v': a, 'q': qdot, 'w': alph}
    
    canopy_derivs = {}
    for canopy_id, canopy in state.canopies.items():
        # Include added mass in effective mass for accurate dynamics
        t_pickup = pickup_times_dict.get(canopy_id)
        # Pass upstream_pickup_times for slack management
        m_added = canopy_added_mass(t, canopy, system, ramp_shape, t_pickup, pickup_times_dict)
        
        # Validate masses
        if not np.isfinite(canopy.m_canopy) or canopy.m_canopy <= 0:
            canopy.m_canopy = 1e-3  # Fallback to small mass
        if not np.isfinite(m_added):
            m_added = 0.0
        
        m_eff = max(canopy.m_canopy + m_added, 1e-9)
        
        # Validate force and cap to prevent explosion
        if not np.all(np.isfinite(F_canopy[canopy_id])):
            F_canopy[canopy_id] = np.zeros(3)
        else:
            # Cap canopy forces to prevent numerical explosion
            F_canopy[canopy_id] = np.clip(F_canopy[canopy_id], -1e6, 1e6)  # Cap at 1 MN per component
        
        a = F_canopy[canopy_id] / m_eff
        
        # Validate acceleration
        if not np.all(np.isfinite(a)):
            a = np.zeros(3)
        
        # Validate velocity
        if not np.all(np.isfinite(canopy.v)):
            canopy.v = np.zeros(3)
        
        canopy_derivs[canopy_id] = {'p': canopy.v, 'v': a}
    
    # Build derivative state
    bodies_deriv = {}
    for body_id in sorted(state.bodies.keys()):
        body = state.bodies[body_id]
        deriv = body_derivs[body_id]
        bodies_deriv[body_id] = BodyNode(
            id=body_id, m=body.m, I_body=body.I_body,
            r=deriv['r'], v=deriv['v'], q=deriv['q'], w=deriv['w'],
            anchors_B=body.anchors_B,
            aero_force=body.aero_force, aero_moment=body.aero_moment
        )
    
    canopies_deriv = {}
    for canopy_id in sorted(state.canopies.keys()):
        canopy = state.canopies[canopy_id]
        deriv = canopy_derivs[canopy_id]
        canopies_deriv[canopy_id] = CanopyNode(
            id=canopy_id, p=deriv['p'], v=deriv['v'],
            m_canopy=canopy.m_canopy, A_inf=canopy.A_inf, CD_inf=canopy.CD_inf,
            td=canopy.td, tau_A=canopy.tau_A, tau_CD=canopy.tau_CD,
            Ca=canopy.Ca, kappa=canopy.kappa
        )
    
    return SimulationState(bodies=bodies_deriv, canopies=canopies_deriv, t=t)


def rk4_step(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    h: float
) -> np.ndarray:
    """RK4 integration step"""
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h, y + h*k3)
    return y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate(
    system: System,
    t0: float,
    tf: float,
    dt: float,
    state0: SimulationState,
    ramp_shape: str = 'exp',
    enable_pickup_shrink: bool = True,
    pickup_dt: float = 0.002,
    pickup_window: float = 0.15
) -> Tuple[List[SimulationState], List[Event]]:
    """Run simulation with RK4 integration"""
    event_mgr = EventManager()
    
    # Map edges to canopies (find which edge connects to each canopy)
    # This allows us to track when pickup occurs for each canopy
    # Check BOTH ends of the edge - canopies can be on either n_minus or n_plus
    edge_to_canopy: Dict[str, str] = {}  # edge_id -> canopy_id
    for edge_id, edge in system.edges.items():
        n_plus_id, _ = _parse_node_ref(edge.n_plus)
        n_minus_id, _ = _parse_node_ref(edge.n_minus)
        # Check both ends - canopies can be on either side
        if n_plus_id in system.canopies:
            edge_to_canopy[edge_id] = n_plus_id
        elif n_minus_id in system.canopies:
            edge_to_canopy[edge_id] = n_minus_id
    
    # Track pickup times per canopy (inflation only starts after pickup)
    pickup_times: Dict[str, float] = {}  # canopy_id -> t_pickup
    
    def make_rhs_vectorized(state_template: SimulationState):
        def f(t: float, y: np.ndarray) -> np.ndarray:
            state = unpack_state(y, system, state_template)
            deriv_state = compute_rhs(t, state, system, ramp_shape, pickup_times)
            return pack_state(deriv_state)
        return f
    
    f = make_rhs_vectorized(state0)
    y = pack_state(state0)
    t = t0
    states = [state0]
    
    edge_lengths_prev: Dict[str, float] = {}
    pickup_active_until = 0.0
    current_dt = dt
    
    while t < tf - 1e-12:
        # Check deployment events (altitude-based or time-based)
        state_current = unpack_state(y, system, state0)
        state_current.t = t
        
        # Check separation events (altitude-based with lag time)
        # Use list() to avoid dictionary changed size during iteration
        bodies_to_check = list(state0.bodies.keys())
        for body_id in bodies_to_check:
            if body_id in state_current.bodies:
                body = state_current.bodies[body_id]
                t_prev = max(t - current_dt, t0)
                event = event_mgr.check_separation(body_id, body, state_current, t, t_prev)
                if event:
                    event_mgr.events.append(event)
                    # Perform separation
                    _perform_body_separation(system, state_current, body_id, state0, t)
                    # Repack state after separation to include new body
                    y = pack_state(state_current)
                    # Update state0 to include new body for future iterations
                    if body_id + '_half' in state_current.bodies:
                        state0.bodies[body_id + '_half'] = state_current.bodies[body_id + '_half']
        
        for canopy_id, canopy in state0.canopies.items():
            # Get current canopy state for altitude check
            if canopy_id in state_current.canopies:
                canopy_current = state_current.canopies[canopy_id]
            else:
                canopy_current = canopy
            event = event_mgr.check_deploy(canopy_id, canopy_current, state_current, t, current_dt)
            if event:
                event_mgr.events.append(event)
        
        # Check reefing events
        for edge_id, edge in system.edges.items():
            if edge.reefing:
                t_prev = max(t - current_dt, t0)
                reef_events = event_mgr.check_reefing(edge_id, t, t_prev, edge.reefing, current_dt)
                event_mgr.events.extend(reef_events)
        
        # Check pickup events
        state_current = unpack_state(y, system, state0)
        state_current.t = t
        
        pickup_detected_this_step = False
        for edge_id, edge in system.edges.items():
            L0_eff = get_effective_reefing_param(t, edge.L0, edge.reefing, 'L0', ramp_shape)
            
            n_minus_id, anchor_name = _parse_node_ref(edge.n_minus)
            if anchor_name is not None:
                body = state_current.bodies[n_minus_id]
                p_minus, _ = get_anchor_pose(body, anchor_name)
            elif n_minus_id in state_current.canopies:
                p_minus = state_current.canopies[n_minus_id].p
            else:
                continue
            
            n_plus_id, _ = _parse_node_ref(edge.n_plus)
            if n_plus_id in state_current.bodies:
                p_plus = state_current.bodies[n_plus_id].r
            elif n_plus_id in state_current.canopies:
                p_plus = state_current.canopies[n_plus_id].p
            else:
                continue
            
            L = float(np.linalg.norm(p_plus - p_minus))
            L_prev = edge_lengths_prev.get(edge_id)
            
            if L_prev is not None:
                event = event_mgr.check_pickup(edge_id, t, L, L0_eff, system, state_current, L_prev, t - current_dt)
                if event:
                    event_mgr.events.append(event)
                    pickup_detected_this_step = True
                    
                    # INSTANTANEOUS JOLT CALCULATION when line goes taut
                    # This happens during hyperinflation peak - maximum shock
                    # Solve for velocity change: impulse = m * delta_v = F * dt
                    # For instantaneous jolt: F_jolt = m * (dv/dt) at moment of tautening
                    jolt_data = _compute_pickup_jolt(
                        state_current, system, edge_id, edge, event.t_event,
                        pickup_times_dict=pickup_times if pickup_times else {}
                    )
                    # Store jolt data in event for later analysis
                    event.extra.update(jolt_data)
                    
                    # Record pickup time for the canopy connected to this edge
                    # CRITICAL: Update pickup_times immediately so it's available for next RHS computation
                    if edge_id in edge_to_canopy:
                        canopy_id = edge_to_canopy[edge_id]
                        # Only set if not already set, or if this is an earlier pickup (shouldn't happen, but safety check)
                        if canopy_id not in pickup_times or event.t_event < pickup_times[canopy_id]:
                            pickup_times[canopy_id] = event.t_event
                        # Ensure pickup time is finite
                        if not np.isfinite(pickup_times[canopy_id]):
                            pickup_times[canopy_id] = event.t_event
                    
                    # Note: Separation is now triggered by altitude/time, not by pickup
                    # Bodies separate first, then freefall until lines go taut
            
            edge_lengths_prev[edge_id] = L
        
        # Pickup step size handling
        if enable_pickup_shrink and pickup_detected_this_step:
            pickup_active_until = t + pickup_window
            current_dt = min(current_dt, pickup_dt)
        
        if enable_pickup_shrink and t >= pickup_active_until:
            current_dt = dt
        
        # Check positions before integration to prevent constraint violation and debug
        # CRITICAL: Apply position correction BEFORE integration to prevent constraint violations
        state_before = unpack_state(y, system, state0)
        state_before.t = t
        position_corrected = False
        
        for edge_id, edge in system.edges.items():
            L0_eff = get_effective_reefing_param(t, edge.L0, edge.reefing, 'L0', ramp_shape)
            # Maximum physical length: L0 + 5% extension (realistic for Kevlar)
            x_max_realistic = L0_eff * 0.05
            L_max = L0_eff + x_max_realistic
            
            n_minus_id, anchor_name = _parse_node_ref(edge.n_minus)
            if anchor_name is not None:
                body = state_before.bodies[n_minus_id]
                p_minus, v_minus = get_anchor_pose(body, anchor_name)
                is_minus_body = True
            elif n_minus_id in state_before.canopies:
                p_minus = state_before.canopies[n_minus_id].p
                v_minus = state_before.canopies[n_minus_id].v
                is_minus_body = False
            else:
                continue
            
            n_plus_id, _ = _parse_node_ref(edge.n_plus)
            if n_plus_id in state_before.bodies:
                p_plus = state_before.bodies[n_plus_id].r
                v_plus = state_before.bodies[n_plus_id].v
                is_plus_body = True
            elif n_plus_id in state_before.canopies:
                p_plus = state_before.canopies[n_plus_id].p
                v_plus = state_before.canopies[n_plus_id].v
                is_plus_body = False
            else:
                continue
            
            dp = p_plus - p_minus
            L = np.linalg.norm(dp)
            dv = v_plus - v_minus
            
            # If L exceeds maximum, correct positions BEFORE integration
            if L > L_max and np.all(np.isfinite(dp)) and L > 1e-6 and L0_eff > 1e-6:
                violation = L - L_max
                u = dp / L if L > 1e-6 else np.array([0.0, 0.0, 0.0])
                
                if np.all(np.isfinite(u)):
                    # Aggressive correction before integration
                    correction = min(violation * 0.8, L0_eff * 0.1)  # Correct up to 10% of L0
                    
                    # Get masses
                    if is_plus_body:
                        m_plus = state_before.bodies[n_plus_id].m
                    else:
                        m_plus = state_before.canopies[n_plus_id].m_canopy
                    
                    if is_minus_body:
                        m_minus = state_before.bodies[n_minus_id].m
                    else:
                        m_minus = state_before.canopies[n_minus_id].m_canopy
                    
                    total_m = m_plus + m_minus
                    if total_m > 0:
                        delta_plus = -(m_minus / total_m) * correction * u
                        delta_minus = (m_plus / total_m) * correction * u
                        
                        if np.all(np.isfinite(delta_plus)) and np.all(np.isfinite(delta_minus)):
                            if is_plus_body:
                                state_before.bodies[n_plus_id].r += delta_plus
                            else:
                                state_before.canopies[n_plus_id].p += delta_plus
                            
                            if is_minus_body:
                                state_before.bodies[n_minus_id].r += delta_minus
                            else:
                                state_before.canopies[n_minus_id].p += delta_minus
                            
                            position_corrected = True
            
            # If L exceeds maximum significantly, reduce step size aggressively
            if L > L_max * 1.1:
                current_dt = min(current_dt, dt * 0.001)  # Use very small step (1000x smaller)
        
        # Repack state if position was corrected
        if position_corrected:
            y = pack_state(state_before)
            
        
        # Integration step
        h = min(current_dt, tf - t)
        y_new = rk4_step(f, t, y, h)
        
        # Normalize quaternions (CRITICAL: must do this after every integration step)
        idx = 0
        for body_id in sorted(state0.bodies.keys()):
            idx += 6  # Skip r (3) and v (3)
            q = y_new[idx:idx+4]
            q_norm = quat_normalize(q)
            # Ensure quaternion is valid before assigning
            if np.all(np.isfinite(q_norm)):
                y_new[idx:idx+4] = q_norm
            else:
                # Fallback to identity quaternion if normalization failed
                y_new[idx:idx+4] = np.array([1.0, 0.0, 0.0, 0.0])
            idx += 7  # Skip q (4) and w (3)
        
        # Validate positions are finite before proceeding
        if not np.all(np.isfinite(y_new)):
            print(f"WARNING: Non-finite state detected at t={t:.3f}s, stopping simulation")
            break
        
        y = y_new
        t += h
        
        state = unpack_state(y, system, state0)
        state.t = t
        
        # Enforce maximum length constraint (physical material limit)
        # This prevents L from exceeding what the cord can physically achieve
        position_corrected_after = False
        for edge_id, edge in system.edges.items():
            L0_eff = get_effective_reefing_param(t, edge.L0, edge.reefing, 'L0', ramp_shape)
            # Maximum realistic extension: 5% for Kevlar
            x_max_realistic = L0_eff * 0.05
            L_max = L0_eff + x_max_realistic
            
            n_minus_id, anchor_name = _parse_node_ref(edge.n_minus)
            if anchor_name is not None:
                body = state.bodies[n_minus_id]
                p_minus, _ = get_anchor_pose(body, anchor_name)
                is_minus_body = True
            elif n_minus_id in state.canopies:
                p_minus = state.canopies[n_minus_id].p
                is_minus_body = False
            else:
                continue
            
            n_plus_id, _ = _parse_node_ref(edge.n_plus)
            if n_plus_id in state.bodies:
                p_plus = state.bodies[n_plus_id].r
                is_plus_body = True
            elif n_plus_id in state.canopies:
                p_plus = state.canopies[n_plus_id].p
                is_plus_body = False
            else:
                continue
            
            dp = p_plus - p_minus
            L = np.linalg.norm(dp)
            
            # If L exceeds maximum, correct positions to enforce constraint
            # Limit correction to prevent huge jumps that cause instability
            if L > L_max and np.all(np.isfinite(dp)) and L > 1e-6 and L0_eff > 1e-6:
                # Move nodes back together proportionally by mass
                # Use more aggressive correction if violation is severe
                violation = L - L_max
                
                # If violation is very large, we need to correct more aggressively
                # But still limit to prevent huge jumps
                if violation > L0_eff * 0.2:  # Violation > 20% of L0
                    # Severe violation - correct more aggressively
                    correction = min(violation * 0.5, L0_eff * 0.15)  # Correct up to 15% of L0
                else:
                    # Moderate violation - conservative correction
                    correction = min(violation, L0_eff * 0.05)  # Limit correction to 5% of L0
                
                u = dp / L
                
                # Ensure unit vector is finite
                if not np.all(np.isfinite(u)):
                    continue  # Skip correction if geometry is invalid
                
                if is_plus_body:
                    m_plus = state.bodies[n_plus_id].m
                else:
                    m_plus = state.canopies[n_plus_id].m_canopy
                
                if is_minus_body:
                    m_minus = state.bodies[n_minus_id].m
                else:
                    m_minus = state.canopies[n_minus_id].m_canopy
                
                total_m = m_plus + m_minus
                if total_m > 0:
                    delta_plus = -(m_minus / total_m) * correction * u
                    delta_minus = (m_plus / total_m) * correction * u
                    
                    if is_plus_body:
                        state.bodies[n_plus_id].r += delta_plus
                    else:
                        state.canopies[n_plus_id].p += delta_plus
                    
                    if is_minus_body:
                        state.bodies[n_minus_id].r += delta_minus
                    else:
                        state.canopies[n_minus_id].p += delta_minus
                    
                    # Also correct velocities to prevent immediate re-separation
                    # Remove relative velocity component along separation direction
                    if is_plus_body:
                        v_plus = state.bodies[n_plus_id].v.copy()
                    else:
                        v_plus = state.canopies[n_plus_id].v.copy()
                    
                    if is_minus_body:
                        v_minus_body = state.bodies[n_minus_id].v.copy()
                        # For body anchor, need to account for rotation
                        body = state.bodies[n_minus_id]
                        R = quat_to_rotm(body.q)
                        if np.all(np.isfinite(R)):
                            r_anchor_B = body.anchors_B[anchor_name]
                            v_minus = v_minus_body + np.cross(body.w, R @ r_anchor_B)
                        else:
                            v_minus = v_minus_body
                    else:
                        v_minus = state.canopies[n_minus_id].v.copy()
                    
                    # Ensure velocities are finite
                    if not (np.all(np.isfinite(v_plus)) and np.all(np.isfinite(v_minus))):
                        continue  # Skip velocity correction if invalid
                    
                    dv = v_plus - v_minus
                    v_rel_along_u = float(np.dot(dv, u))
                    
                    # If separating (positive), reduce velocity difference
                    # Use more aggressive damping if violation is severe
                    if v_rel_along_u > 0 and np.isfinite(v_rel_along_u):
                        # Damping factor: more aggressive if violation is large
                        violation_ratio = violation / max(L0_eff, 1e-6)
                        if violation_ratio > 0.1:  # Severe violation
                            damping_factor = 0.9  # Reduce velocity difference by 90% (very aggressive)
                        else:
                            damping_factor = 0.7  # Reduce velocity difference by 70%
                        v_correction = damping_factor * v_rel_along_u
                        
                        # Mass-weighted velocity correction
                        delta_v_plus = -(m_minus / total_m) * v_correction * u
                        delta_v_minus = (m_plus / total_m) * v_correction * u
                        
                        # Ensure corrections are finite
                        if np.all(np.isfinite(delta_v_plus)) and np.all(np.isfinite(delta_v_minus)):
                            if is_plus_body:
                                state.bodies[n_plus_id].v += delta_v_plus
                            else:
                                state.canopies[n_plus_id].v += delta_v_plus
                            
                            if is_minus_body:
                                # For body, correct body velocity
                                state.bodies[n_minus_id].v += delta_v_minus
                            else:
                                state.canopies[n_minus_id].v += delta_v_minus
                    
                    position_corrected_after = True
        
        # Repack state after position correction
        if position_corrected_after:
            y = pack_state(state)
        
        # Check for impact (altitude reaches zero) - stop at ground impact
        impact = False
        for body in state.bodies.values():
            # Check if any body has reached ground (z-coordinate is altitude)
            if body.r[2] <= 0.0:
                impact = True
                print(f"Impact detected at t={t:.3f}s (body {body.id}), altitude={body.r[2]:.2f}m")
                break
        
        if not impact:
            for canopy in state.canopies.values():
                if canopy.p[2] <= 0.0:
                    impact = True
                    print(f"Impact detected at t={t:.3f}s (canopy {canopy.id}), altitude={canopy.p[2]:.2f}m")
                    break
        
        if impact:
            states.append(state)
            break
        
        # Additional validation: check positions are reasonable (numerical instability)
        invalid = False
        for body in state.bodies.values():
            if not np.all(np.isfinite(body.r)) or np.any(np.abs(body.r) > 1e6):
                invalid = True
                print(f"  Body {body.id} has invalid position: {body.r}")
                break
        for canopy in state.canopies.values():
            if not np.all(np.isfinite(canopy.p)) or np.any(np.abs(canopy.p) > 1e6):
                invalid = True
                print(f"  Canopy {canopy.id} has invalid position: {canopy.p}")
                break
        
        if invalid:
            print(f"WARNING: Invalid positions detected at t={t:.3f}s, stopping simulation")
            print(f"  This may be due to numerical instability or unrealistic forces.")
            print(f"  Check timestep size, material properties, and initial conditions.")
            # Try to find which edge is causing the problem
            for edge_id, edge in system.edges.items():
                n_minus_id, anchor_name = _parse_node_ref(edge.n_minus)
                n_plus_id, _ = _parse_node_ref(edge.n_plus)
                if anchor_name is not None and n_minus_id in state.bodies:
                    body = state.bodies[n_minus_id]
                    p_minus, _ = get_anchor_pose(body, anchor_name)
                elif n_minus_id in state.canopies:
                    p_minus = state.canopies[n_minus_id].p
                else:
                    continue
                if n_plus_id in state.bodies:
                    p_plus = state.bodies[n_plus_id].r
                elif n_plus_id in state.canopies:
                    p_plus = state.canopies[n_plus_id].p
                else:
                    continue
                L = np.linalg.norm(p_plus - p_minus)
                L0_eff = get_effective_reefing_param(t, edge.L0, edge.reefing, 'L0', 'exp')
                if L > L0_eff * 2.0:  # More than 2x L0
                    print(f"  Edge {edge_id} has length L={L:.2f}m (L0={L0_eff:.2f}m) - may be causing instability")
            break
        
        states.append(state)
    
    return states, event_mgr.events
