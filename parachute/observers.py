"""CSV writers, peak detection, trim estimator"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import csv
import json
from pathlib import Path
from .model import System
from .engine import SimulationState, Event
from .physics import (
    canopy_area, canopy_CD, canopy_added_mass, get_effective_reefing_param,
    quat_to_rotm, get_anchor_pose, canopy_drag_force
)
from .model import parse_node_ref


def compute_shock_vectors(
    t: float,
    state: SimulationState,
    system: System,
    ramp_shape: str = 'exp',
    pickup_times: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute per-body and system shock vectors.
    
    Returns:
        (F_sys, F_per_body_dict)
    """
    gvec = np.array([0.0, 0.0, system.atmos.g])
    F_per_body = {bid: np.zeros(3) for bid in state.bodies.keys()}
    
    # Accumulate tensions from edges
    for edge_id, edge in system.edges.items():
        L0_eff = get_effective_reefing_param(t, edge.L0, edge.reefing, 'L0', ramp_shape)
        
        n_minus_id, anchor_name = parse_node_ref(edge.n_minus)
        if anchor_name is not None:
            body = state.bodies[n_minus_id]
            p_minus, v_minus = get_anchor_pose(body, anchor_name)
        elif n_minus_id in state.canopies:
            p_minus = state.canopies[n_minus_id].p
            v_minus = state.canopies[n_minus_id].v
        else:
            continue
        
        n_plus_id, _ = parse_node_ref(edge.n_plus)
        if n_plus_id in state.bodies:
            p_plus = state.bodies[n_plus_id].r
            v_plus = state.bodies[n_plus_id].v
        elif n_plus_id in state.canopies:
            p_plus = state.canopies[n_plus_id].p
            v_plus = state.canopies[n_plus_id].v
        else:
            continue
        
        dp = p_plus - p_minus
        L = np.linalg.norm(dp)
        
        if L < 1e-9:
            T = 0.0
            u = np.array([0.0, 0.0, 0.0])
        else:
            u = dp / L
            x = max(0.0, L - L0_eff)
            dv = v_plus - v_minus
            xdot = float(np.dot(dv, u))
            
            # Compute tension
            from .physics import tension_from_extension
            T = tension_from_extension(edge, x, xdot, t, ramp_shape)
            
            if not np.isfinite(T) or not np.all(np.isfinite(u)):
                T = 0.0
                u = np.array([0.0, 0.0, 0.0])
        
        # Apply to body if anchor
        if anchor_name is not None:
            F_per_body[n_minus_id] += T * u
        elif n_plus_id in state.bodies:
            F_per_body[n_plus_id] += T * u
    
    # Get pickup times for each canopy (inflation only starts after pickup)
    pickup_times_dict = pickup_times if pickup_times is not None else {}
    
    # ORIENTATION-DEPENDENT: Add canopy drag forces to connected bodies
    # Multiple canopies create drag vectors at different angles, generating torques
    # CRITICAL: Must pass pickup_times_dict to canopy_drag_force for proper inflation
    # This is critical for multi-parachute systems where drag vectors may not be vertical
    for edge_id, edge in system.edges.items():
        n_plus_id, _ = parse_node_ref(edge.n_plus)
        if n_plus_id in state.canopies:
            canopy = state.canopies[n_plus_id]
            t_pickup = pickup_times_dict.get(n_plus_id)
            n_minus_id, anchor_name = parse_node_ref(edge.n_minus)
            
            # Get canopy drag force (orientation-dependent)
            F_drag = canopy_drag_force(t, canopy, system, ramp_shape, t_pickup, pickup_times_dict)
            
            # Apply drag to connected body at anchor point
            if anchor_name is not None and n_minus_id in state.bodies:
                F_per_body[n_minus_id] += F_drag  # Drag force on body
                
                # Added mass reaction force
                m_added = canopy_added_mass(t, canopy, system, ramp_shape, t_pickup, pickup_times_dict)
                if m_added > 1e-6:
                    # Estimate canopy acceleration
                    m_eff = max(canopy.m_canopy + m_added, 1e-9)
                    F_canopy_net = F_drag + canopy.m_canopy * gvec
                    # Subtract tension (already counted above)
                    L0_eff = get_effective_reefing_param(t, edge.L0, edge.reefing, 'L0', ramp_shape)
                    body = state.bodies[n_minus_id]
                    p_minus, v_minus = get_anchor_pose(body, anchor_name)
                    dp = canopy.p - p_minus
                    L = np.linalg.norm(dp)
                    if L > 1e-9:
                        u = dp / L
                        x = max(0.0, L - L0_eff)
                        dv = canopy.v - v_minus
                        xdot = float(np.dot(dv, u))
                        from .physics import tension_from_extension
                        T = tension_from_extension(edge, x, xdot, t, ramp_shape)
                        if T > 0:
                            F_canopy_net -= T * u
                    a_c = F_canopy_net / m_eff
                    F_added = m_added * a_c
                    F_per_body[n_minus_id] += F_added
    
    # Add gravity and aero
    # ORIENTATION-DEPENDENT: Gravity in body frame for inertial loading analysis
    for body_id, body in state.bodies.items():
        F_per_body[body_id] += body.m * gvec
        
        # Body aerodynamic drag (orientation-dependent)
        if body.aero_area is not None and body.aero_CD is not None:
            rho = system.atmos.rho
            v_air = system.atmos.v_air
            v_rel = v_air - body.v
            V = np.linalg.norm(v_rel)
            if V > 1e-6:
                v_rel_hat = v_rel / V
                F_drag_body = 0.5 * rho * V**2 * body.aero_area * body.aero_CD * (-v_rel_hat)
                F_per_body[body_id] += F_drag_body
        
        if body.aero_force is not None:
            F_aero = body.aero_force(state, system, body_id)
            if F_aero is not None:
                F_per_body[body_id] += F_aero
    
    # System shock
    F_sys = np.sum(np.vstack(list(F_per_body.values())), axis=0)
    
    return F_sys, F_per_body


def write_telemetry_csv(
    states: List[SimulationState],
    system: System,
    output_path: Path,
    ramp_shape: str = 'exp',
    verbose: bool = False,
    events: Optional[List] = None
):
    """Write telemetry CSV with all required columns"""
    fieldnames = ['t']
    
    # System shock
    fieldnames.extend(['Fsys_x', 'Fsys_y', 'Fsys_z'])
    
    # Per-body shocks (include dynamically added bodies from first state)
    # Get all body IDs from states (handles dynamically added bodies)
    all_body_ids = set()
    if states:
        all_body_ids = set(states[0].bodies.keys())
        for state in states:
            all_body_ids.update(state.bodies.keys())
    else:
        all_body_ids = set(system.bodies.keys())
    
    body_ids = sorted(all_body_ids)
    for bid in body_ids:
        fieldnames.extend([f'body:{bid}:Fx', f'body:{bid}:Fy', f'body:{bid}:Fz'])
    
    # Per-anchor loads (mounting hardware analysis)
    for bid in body_ids:
        # Get body from system or first state
        if bid in system.bodies:
            body = system.bodies[bid]
        elif states and bid in states[0].bodies:
            body = states[0].bodies[bid]
        else:
            continue
        for anchor_name in sorted(body.anchors_B.keys()):
            # Inertial frame (system-level analysis)
            fieldnames.extend([
                f'anchor:{bid}:{anchor_name}:Fx_I',
                f'anchor:{bid}:{anchor_name}:Fy_I',
                f'anchor:{bid}:{anchor_name}:Fz_I',
                f'anchor:{bid}:{anchor_name}:Fmag_I'
            ])
            # Body frame (mounting hardware FoS - critical for orientation-dependent loading)
            fieldnames.extend([
                f'anchor:{bid}:{anchor_name}:Fx_B',
                f'anchor:{bid}:{anchor_name}:Fy_B',
                f'anchor:{bid}:{anchor_name}:Fz_B',
                f'anchor:{bid}:{anchor_name}:Fmag_B'
            ])
    
    # Per-edge diagnostics
    edge_ids = sorted(system.edges.keys())
    for eid in edge_ids:
        fieldnames.extend([f'edge:{eid}:T', f'edge:{eid}:L', f'edge:{eid}:x'])
    
    # Per-canopy inflation
    canopy_ids = sorted(system.canopies.keys())
    for cid in canopy_ids:
        fieldnames.extend([f'canopy:{cid}:A', f'canopy:{cid}:CD', f'canopy:{cid}:F_added_mass'])
    
    # Optional verbose columns
    if verbose:
        for bid in body_ids:
            fieldnames.extend([
                f'body:{bid}:r_x', f'body:{bid}:r_y', f'body:{bid}:r_z',
                f'body:{bid}:v_x', f'body:{bid}:v_y', f'body:{bid}:v_z',
                f'body:{bid}:q_w', f'body:{bid}:q_x', f'body:{bid}:q_y', f'body:{bid}:q_z'
            ])
        for cid in canopy_ids:
            fieldnames.extend([
                f'canopy:{cid}:p_x', f'canopy:{cid}:p_y', f'canopy:{cid}:p_z',
                f'canopy:{cid}:v_x', f'canopy:{cid}:v_y', f'canopy:{cid}:v_z'
            ])
    
    # Extract pickup times from events (map edge_id -> canopy_id -> t_pickup)
    pickup_times: Dict[str, float] = {}  # canopy_id -> t_pickup
    if events:
        # Map edges to canopies
        edge_to_canopy: Dict[str, str] = {}
        for edge_id, edge in system.edges.items():
            n_plus_id, _ = parse_node_ref(edge.n_plus)
            if n_plus_id in system.canopies:
                edge_to_canopy[edge_id] = n_plus_id
        
        # Extract pickup events
        for event in events:
            if event.event_type == 'pickup' and event.id in edge_to_canopy:
                canopy_id = edge_to_canopy[event.id]
                if canopy_id not in pickup_times:
                    pickup_times[canopy_id] = event.t_event
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for state in states:
            t = state.t
            
            # Compute shocks
            F_sys, F_per_body = compute_shock_vectors(t, state, system, ramp_shape, pickup_times)
            
            row = {
                't': t,
                'Fsys_x': F_sys[0], 'Fsys_y': F_sys[1], 'Fsys_z': F_sys[2]
            }
            
            # Body shocks (handle dynamically added bodies)
            for bid in body_ids:
                # Check if body exists in current state
                if bid not in state.bodies:
                    # Body doesn't exist in this state (e.g., not yet created or removed)
                    row[f'body:{bid}:Fx'] = 0.0
                    row[f'body:{bid}:Fy'] = 0.0
                    row[f'body:{bid}:Fz'] = 0.0
                elif bid in F_per_body:
                    F = F_per_body[bid]
                    row[f'body:{bid}:Fx'] = F[0]
                    row[f'body:{bid}:Fy'] = F[1]
                    row[f'body:{bid}:Fz'] = F[2]
                else:
                    # Body exists in state but no force recorded (newly added, no forces yet)
                    row[f'body:{bid}:Fx'] = 0.0
                    row[f'body:{bid}:Fy'] = 0.0
                    row[f'body:{bid}:Fz'] = 0.0
            
            # Per-anchor loads (mounting hardware shock analysis)
            # ORIENTATION-DEPENDENT: Track forces in both inertial and body frames
            anchor_loads = {f'{bid}:{anchor_name}': {'I': np.zeros(3), 'B': np.zeros(3)} 
                           for bid in body_ids 
                           for anchor_name in system.bodies[bid].anchors_B.keys()}
            
            # 1. Tension forces from lines
            for edge_id, edge in system.edges.items():
                n_minus_id, anchor_name = parse_node_ref(edge.n_minus)
                if anchor_name is not None and n_minus_id in state.bodies:
                    body = state.bodies[n_minus_id]
                    L0_eff = get_effective_reefing_param(t, edge.L0, edge.reefing, 'L0', ramp_shape)
                    p_minus, v_minus = get_anchor_pose(body, anchor_name)
                    
                    n_plus_id, _ = parse_node_ref(edge.n_plus)
                    if n_plus_id in state.bodies:
                        p_plus = state.bodies[n_plus_id].r
                        v_plus = state.bodies[n_plus_id].v
                    elif n_plus_id in state.canopies:
                        p_plus = state.canopies[n_plus_id].p
                        v_plus = state.canopies[n_plus_id].v
                    else:
                        continue
                    dp = p_plus - p_minus
                    L = np.linalg.norm(dp)
                    if L > 1e-9:
                        u = dp / L
                        x = max(0.0, L - L0_eff)
                        dv = v_plus - v_minus
                        xdot = float(np.dot(dv, u))
                        from .physics import tension_from_extension
                        T = tension_from_extension(edge, x, xdot, t, ramp_shape)
                        if T > 0:
                            F_tension_I = T * u  # Inertial frame
                            anchor_key = f'{n_minus_id}:{anchor_name}'
                            if anchor_key in anchor_loads:
                                anchor_loads[anchor_key]['I'] += F_tension_I
                                # Convert to body frame for mounting hardware analysis
                                R = quat_to_rotm(body.q)
                                F_tension_B = R.T @ F_tension_I
                                anchor_loads[anchor_key]['B'] += F_tension_B
            
            # 2. ORIENTATION-DEPENDENT: Canopy drag forces at anchor points
            # Multiple canopies create drag vectors at different angles, generating torques
            for canopy_id, canopy in state.canopies.items():
                t_pickup = pickup_times.get(canopy_id)
                F_drag_I = canopy_drag_force(t, canopy, system, ramp_shape, t_pickup, pickup_times)
                
                # Find which body anchor this canopy connects to
                for edge_id, edge in system.edges.items():
                    n_plus_id, _ = parse_node_ref(edge.n_plus)
                    if n_plus_id == canopy_id:
                        n_minus_id, anchor_name = parse_node_ref(edge.n_minus)
                        if anchor_name is not None and n_minus_id in state.bodies:
                            body = state.bodies[n_minus_id]
                            anchor_key = f'{n_minus_id}:{anchor_name}'
                            if anchor_key in anchor_loads:
                                # Drag force in inertial frame
                                anchor_loads[anchor_key]['I'] += F_drag_I
                                # Drag force in body frame (for mounting hardware analysis)
                                R = quat_to_rotm(body.q)
                                F_drag_B = R.T @ F_drag_I
                                anchor_loads[anchor_key]['B'] += F_drag_B
                                
                                # Also account for added mass reaction force
                                m_added = canopy_added_mass(t, canopy, system, ramp_shape, t_pickup, pickup_times)
                                if m_added > 1e-6:
                                    # Estimate canopy acceleration
                                    m_eff = max(canopy.m_canopy + m_added, 1e-9)
                                    gvec = np.array([0.0, 0.0, system.atmos.g])
                                    F_canopy_net = F_drag_I + canopy.m_canopy * gvec
                                    # Subtract tension (already counted above)
                                    p_minus, v_minus = get_anchor_pose(body, anchor_name)
                                    dp = canopy.p - p_minus
                                    L = np.linalg.norm(dp)
                                    if L > 1e-9:
                                        u = dp / L
                                        L0_eff = get_effective_reefing_param(t, edge.L0, edge.reefing, 'L0', ramp_shape)
                                        x = max(0.0, L - L0_eff)
                                        dv = canopy.v - v_minus
                                        xdot = float(np.dot(dv, u))
                                        from .physics import tension_from_extension
                                        T = tension_from_extension(edge, x, xdot, t, ramp_shape)
                                        if T > 0:
                                            F_canopy_net -= T * u
                                    a_c = F_canopy_net / m_eff
                                    F_added_I = m_added * a_c
                                    anchor_loads[anchor_key]['I'] += F_added_I
                                    F_added_B = R.T @ F_added_I
                                    anchor_loads[anchor_key]['B'] += F_added_B
            
            # Write anchor loads (inertial frame and body frame)
            # Body frame is critical for mounting hardware FoS analysis (orientation-dependent)
            for bid in body_ids:
                if bid not in state.bodies:
                    # Body doesn't exist in this state - skip anchor loads for this body
                    continue
                body = state.bodies[bid]
                for anchor_name in sorted(body.anchors_B.keys()):
                    anchor_key = f'{bid}:{anchor_name}'
                    if anchor_key in anchor_loads:
                        F_anchor_I = anchor_loads[anchor_key]['I']  # Inertial frame
                        F_anchor_B = anchor_loads[anchor_key]['B']  # Body frame
                        
                        # Inertial frame (for system-level analysis)
                        row[f'anchor:{bid}:{anchor_name}:Fx_I'] = F_anchor_I[0]
                        row[f'anchor:{bid}:{anchor_name}:Fy_I'] = F_anchor_I[1]
                        row[f'anchor:{bid}:{anchor_name}:Fz_I'] = F_anchor_I[2]
                        row[f'anchor:{bid}:{anchor_name}:Fmag_I'] = float(np.linalg.norm(F_anchor_I))
                        
                        # Body frame (for mounting hardware FoS - orientation-dependent)
                        row[f'anchor:{bid}:{anchor_name}:Fx_B'] = F_anchor_B[0]
                        row[f'anchor:{bid}:{anchor_name}:Fy_B'] = F_anchor_B[1]
                        row[f'anchor:{bid}:{anchor_name}:Fz_B'] = F_anchor_B[2]
                        row[f'anchor:{bid}:{anchor_name}:Fmag_B'] = float(np.linalg.norm(F_anchor_B))
                    else:
                        # Initialize if anchor doesn't have loads yet
                        row[f'anchor:{bid}:{anchor_name}:Fx_I'] = 0.0
                        row[f'anchor:{bid}:{anchor_name}:Fy_I'] = 0.0
                        row[f'anchor:{bid}:{anchor_name}:Fz_I'] = 0.0
                        row[f'anchor:{bid}:{anchor_name}:Fmag_I'] = 0.0
                        row[f'anchor:{bid}:{anchor_name}:Fx_B'] = 0.0
                        row[f'anchor:{bid}:{anchor_name}:Fy_B'] = 0.0
                        row[f'anchor:{bid}:{anchor_name}:Fz_B'] = 0.0
                        row[f'anchor:{bid}:{anchor_name}:Fmag_B'] = 0.0
            
            # Edge diagnostics
            for eid in edge_ids:
                edge = system.edges[eid]
                L0_eff = get_effective_reefing_param(t, edge.L0, edge.reefing, 'L0', ramp_shape)
                
                n_minus_id, anchor_name = parse_node_ref(edge.n_minus)
                if anchor_name is not None:
                    if n_minus_id not in state.bodies:
                        row[f'edge:{eid}:T'] = 0.0
                        row[f'edge:{eid}:L'] = 0.0
                        row[f'edge:{eid}:x'] = 0.0
                        continue
                    body = state.bodies[n_minus_id]
                    p_minus, v_minus = get_anchor_pose(body, anchor_name)
                elif n_minus_id in state.canopies:
                    p_minus = state.canopies[n_minus_id].p
                    v_minus = state.canopies[n_minus_id].v
                else:
                    row[f'edge:{eid}:T'] = 0.0
                    row[f'edge:{eid}:L'] = 0.0
                    row[f'edge:{eid}:x'] = 0.0
                    continue
                
                n_plus_id, _ = parse_node_ref(edge.n_plus)
                if n_plus_id in state.bodies:
                    p_plus = state.bodies[n_plus_id].r
                    v_plus = state.bodies[n_plus_id].v
                elif n_plus_id in state.canopies:
                    p_plus = state.canopies[n_plus_id].p
                    v_plus = state.canopies[n_plus_id].v
                else:
                    row[f'edge:{eid}:T'] = 0.0
                    row[f'edge:{eid}:L'] = 0.0
                    row[f'edge:{eid}:x'] = 0.0
                    continue
                
                dp = p_plus - p_minus
                L = float(np.linalg.norm(dp))
                x = max(0.0, L - L0_eff)
                
                if L < 1e-9:
                    T = 0.0
                else:
                    u = dp / L
                    dv = v_plus - v_minus
                    xdot = float(np.dot(dv, u))
                    from .physics import tension_from_extension
                    T = tension_from_extension(edge, x, xdot, t, ramp_shape)
                
                row[f'edge:{eid}:T'] = T
                row[f'edge:{eid}:L'] = L
                row[f'edge:{eid}:x'] = x
            
            # Canopy inflation
            gvec = np.array([0.0, 0.0, system.atmos.g])
            for cid in canopy_ids:
                canopy = state.canopies[cid]
                t_pickup = pickup_times.get(cid)
                A = canopy_area(t, canopy, system, ramp_shape, t_pickup, pickup_times)
                CD = canopy_CD(t, canopy, system, ramp_shape, t_pickup, pickup_times)
                
                # Added mass force magnitude (for diagnostics)
                m_added = canopy_added_mass(t, canopy, system, ramp_shape, t_pickup, pickup_times)
                # Compute canopy acceleration estimate for added mass force
                F_drag = canopy_drag_force(t, canopy, system, ramp_shape, t_pickup, pickup_times)
                F_canopy_est = F_drag + canopy.m_canopy * gvec
                # Find tension from edge connecting to canopy
                T_est = 0.0
                for edge_id, edge in system.edges.items():
                    n_plus_id, _ = parse_node_ref(edge.n_plus)
                    if n_plus_id == cid:
                        n_minus_id, anchor_name = parse_node_ref(edge.n_minus)
                        if anchor_name is not None:
                            if n_minus_id not in state.bodies:
                                continue
                            body = state.bodies[n_minus_id]
                            p_minus, _ = get_anchor_pose(body, anchor_name)
                        elif n_minus_id in state.canopies:
                            p_minus = state.canopies[n_minus_id].p
                        else:
                            continue
                        dp = canopy.p - p_minus
                        L = np.linalg.norm(dp)
                        if L > 1e-9:
                            u = dp / L
                            L0_eff = get_effective_reefing_param(t, edge.L0, edge.reefing, 'L0', ramp_shape)
                            x = max(0.0, L - L0_eff)
                            if anchor_name is not None:
                                body = state.bodies[n_minus_id]
                                _, v_minus = get_anchor_pose(body, anchor_name)
                            else:
                                v_minus = state.canopies[n_minus_id].v
                            dv = canopy.v - v_minus
                            xdot = float(np.dot(dv, u))
                            from .physics import tension_from_extension
                            T_est = tension_from_extension(edge, x, xdot, t, ramp_shape)
                            if T_est > 0:
                                F_canopy_est -= T_est * u
                            break
                m_eff = max(canopy.m_canopy + m_added, 1e-9)
                a_c_est = F_canopy_est / m_eff
                F_added_mag = float(np.linalg.norm(m_added * a_c_est))
                row[f'canopy:{cid}:A'] = A
                row[f'canopy:{cid}:CD'] = CD
                row[f'canopy:{cid}:F_added_mass'] = F_added_mag
            
            # Verbose columns
            if verbose:
                for bid in body_ids:
                    if bid in state.bodies:
                        body = state.bodies[bid]
                        row[f'body:{bid}:r_x'] = body.r[0]
                        row[f'body:{bid}:r_y'] = body.r[1]
                        row[f'body:{bid}:r_z'] = body.r[2]
                        row[f'body:{bid}:v_x'] = body.v[0]
                        row[f'body:{bid}:v_y'] = body.v[1]
                        row[f'body:{bid}:v_z'] = body.v[2]
                        row[f'body:{bid}:q_w'] = body.q[0]
                        row[f'body:{bid}:q_x'] = body.q[1]
                        row[f'body:{bid}:q_y'] = body.q[2]
                        row[f'body:{bid}:q_z'] = body.q[3]
                    else:
                        # Body doesn't exist in this state
                        for suffix in ['r_x', 'r_y', 'r_z', 'v_x', 'v_y', 'v_z', 'q_w', 'q_x', 'q_y', 'q_z']:
                            row[f'body:{bid}:{suffix}'] = 0.0
                
                for cid in canopy_ids:
                    if cid in state.canopies:
                        canopy = state.canopies[cid]
                        row[f'canopy:{cid}:p_x'] = canopy.p[0]
                        row[f'canopy:{cid}:p_y'] = canopy.p[1]
                        row[f'canopy:{cid}:p_z'] = canopy.p[2]
                        row[f'canopy:{cid}:v_x'] = canopy.v[0]
                        row[f'canopy:{cid}:v_y'] = canopy.v[1]
                        row[f'canopy:{cid}:v_z'] = canopy.v[2]
                    else:
                        # Canopy doesn't exist in this state
                        for suffix in ['p_x', 'p_y', 'p_z', 'v_x', 'v_y', 'v_z']:
                            row[f'canopy:{cid}:{suffix}'] = 0.0
            
            writer.writerow(row)


def write_events_csv(events: List[Event], output_path: Path):
    """Write events CSV"""
    fieldnames = ['event_type', 'id', 't_event', 'extra']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow({
                'event_type': event.event_type,
                'id': event.id,
                't_event': event.t_event,
                'extra': json.dumps(event.extra)
            })


def detect_peaks(
    states: List[SimulationState],
    system: System,
    events: List[Event],
    peak_window: float = 5.0,
    prominence_factor: float = 2.0
) -> List[Dict]:
    """Detect peak shocks after each deployment"""
    # Get deployment times
    deploy_times = {}
    for event in events:
        if event.event_type == 'deploy_on':
            deploy_times[event.id] = event.t_event
    
    if not deploy_times:
        return []
    
    # Extract shock magnitudes
    times = [s.t for s in states]
    F_sys_mags = []
    F_body_mags = {bid: [] for bid in system.bodies.keys()}
    
    # Extract pickup times from events
    pickup_times: Dict[str, float] = {}
    if events:
        edge_to_canopy: Dict[str, str] = {}
        for edge_id, edge in system.edges.items():
            n_plus_id, _ = parse_node_ref(edge.n_plus)
            if n_plus_id in system.canopies:
                edge_to_canopy[edge_id] = n_plus_id
        for event in events:
            if event.event_type == 'pickup' and event.id in edge_to_canopy:
                canopy_id = edge_to_canopy[event.id]
                if canopy_id not in pickup_times:
                    pickup_times[canopy_id] = event.t_event
    
    for state in states:
        F_sys, F_per_body = compute_shock_vectors(state.t, state, system, ramp_shape='exp', pickup_times=pickup_times)
        F_sys_mags.append(float(np.linalg.norm(F_sys)))
        for bid, F in F_per_body.items():
            F_body_mags[bid].append(float(np.linalg.norm(F)))
    
    peaks = []
    sorted_deploys = sorted(deploy_times.items(), key=lambda x: x[1])
    
    for i, (canopy_id, t_d) in enumerate(sorted_deploys):
        # Window: from t_d to min(next_deploy, t_d + peak_window)
        if i + 1 < len(sorted_deploys):
            window_end = min(sorted_deploys[i+1][1], t_d + peak_window)
        else:
            window_end = t_d + peak_window
        
        # Find indices in window
        start_idx = next((i for i, t in enumerate(times) if t >= t_d), len(times))
        end_idx = next((i for i, t in enumerate(times) if t > window_end), len(times))
        
        if start_idx >= end_idx:
            continue
        
        # Find local max in window
        window_mags = F_sys_mags[start_idx:end_idx]
        if len(window_mags) < 3:
            continue
        
        # Simple peak detection
        median_dev = np.median(np.abs(np.diff(window_mags)))
        threshold = prominence_factor * median_dev
        
        for j in range(1, len(window_mags) - 1):
            if (window_mags[j] > window_mags[j-1] and 
                window_mags[j] > window_mags[j+1] and
                window_mags[j] - min(window_mags[j-1], window_mags[j+1]) > threshold):
                peak_idx = start_idx + j
                peaks.append({
                    'event_id': canopy_id,
                    'scope': 'system',
                    't_peak': times[peak_idx],
                    'F_peak': window_mags[j],
                    'idx_left': peak_idx - 1,
                    'idx_right': peak_idx + 1
                })
                break  # First peak only
        
        # Per-body peaks
        for bid in system.bodies.keys():
            body_mags = F_body_mags[bid][start_idx:end_idx]
            if len(body_mags) < 3:
                continue
            
            for j in range(1, len(body_mags) - 1):
                if (body_mags[j] > body_mags[j-1] and 
                    body_mags[j] > body_mags[j+1] and
                    body_mags[j] - min(body_mags[j-1], body_mags[j+1]) > threshold):
                    peak_idx = start_idx + j
                    peaks.append({
                        'event_id': canopy_id,
                        'scope': f'body:{bid}',
                        't_peak': times[peak_idx],
                        'F_peak': body_mags[j],
                        'idx_left': peak_idx - 1,
                        'idx_right': peak_idx + 1
                    })
                    break
    
    return peaks


def write_peaks_csv(peaks: List[Dict], output_path: Path):
    """Write peaks CSV"""
    if not peaks:
        # Create empty file with headers
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['event_id', 'scope', 't_peak', 'F_peak', 'idx_left', 'idx_right'])
            writer.writeheader()
        return
    
    fieldnames = ['event_id', 'scope', 't_peak', 'F_peak', 'idx_left', 'idx_right']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for peak in peaks:
            writer.writerow(peak)
