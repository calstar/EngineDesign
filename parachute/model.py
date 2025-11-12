"""Data model, graph assembly, and YAML config loader"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
import numpy as np
import yaml
from pathlib import Path


@dataclass
class BodyNode:
    """Rigid body node"""
    id: str
    m: float  # mass
    I_body: np.ndarray  # (3,3) inertia tensor in body frame
    r: np.ndarray  # (3,) position in inertial frame
    v: np.ndarray  # (3,) velocity in inertial frame
    q: np.ndarray  # (4,) quaternion (scalar-first) body->inertial
    w: np.ndarray  # (3,) angular velocity in body frame
    anchors_B: Dict[str, np.ndarray]  # name -> (3,) anchor offset in body frame
    aero_force: Optional[Callable[[Any, Any, str], np.ndarray]] = None
    aero_moment: Optional[Callable[[Any, Any, str], np.ndarray]] = None
    # Body aerodynamic properties (for freefall drag)
    aero_area: Optional[float] = None  # m² - cross-sectional area for drag
    aero_CD: Optional[float] = None  # Drag coefficient (typically 0.5-1.2 for cylinders)
    # Separation configuration: altitude-based trigger with lag time
    separation_signal_altitude: Optional[float] = None  # m AGL - altitude when signal is sent
    separation_lag_time: float = 0.0  # s - time delay from signal to actual separation (signal delay + charge delay + separation delay)
    separation_v_mag: float = 6.0  # m/s - separation velocity (relative velocity between halves, from black powder charge)
    # Note: Post-separation spin rates calculated from angular momentum conservation


@dataclass
class CanopyNode:
    """Canopy node"""
    id: str
    p: np.ndarray  # (3,) position in inertial frame
    v: np.ndarray  # (3,) velocity in inertial frame
    m_canopy: float
    A_inf: float  # fully inflated area
    CD_inf: float  # fully inflated drag coefficient
    td: float  # deployment start time (seconds) - use if altitude_deploy is None
    tau_A: float  # area inflation time constant
    tau_CD: float  # drag coefficient inflation time constant
    Ca: float  # added mass coefficient
    kappa: float  # geometry constant for volume vs radius
    altitude_deploy: Optional[float] = None  # deployment altitude (meters AGL) - overrides td if set
    upstream_canopy: Optional[str] = None  # ID of upstream canopy that must inflate first (for slack management)


@dataclass
class ReefingStage:
    """Reefing stage parameters"""
    t_on: float
    L0: Optional[float] = None
    k0: Optional[float] = None
    c: Optional[float] = None
    A_scale: Optional[float] = None
    CD_scale: Optional[float] = None
    ramp_tau: Optional[float] = None


@dataclass
class LineEdge:
    """Line edge (span) between two nodes"""
    id: str
    n_minus: str  # node id or 'body:anchor' like 'rocket:base'
    n_plus: str  # node id (canopy or body)
    L0: float  # rest length
    k0: float  # linear stiffness
    k1: float  # nonlinear stiffness coefficient
    alpha: float  # nonlinear stiffness exponent
    c: float  # damping coefficient
    T_pre: float  # preload tension
    T_max: float = 1e6  # maximum tension (breaking strength) - physical material limit
    reefing: List[ReefingStage] = field(default_factory=list)


@dataclass
class Atmosphere:
    """Atmospheric parameters"""
    rho: float = 1.225  # air density kg/m³
    g: float = 9.80665  # gravity m/s²
    v_air: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,) air velocity


@dataclass
class System:
    """Complete system topology"""
    bodies: Dict[str, BodyNode]
    canopies: Dict[str, CanopyNode]
    edges: Dict[str, LineEdge]
    atmos: Atmosphere


def parse_node_ref(ref: str) -> Tuple[str, Optional[str]]:
    """Parse node reference into (node_id, anchor_name) or (node_id, None)"""
    if ':' in ref:
        parts = ref.split(':', 1)
        return parts[0], parts[1]
    return ref, None


def validate_topology(system: System) -> None:
    """Validate graph topology: all edges reference existing nodes, no cycles, single parent per canopy"""
    # Check all edge references exist
    all_body_ids = set(system.bodies.keys())
    all_canopy_ids = set(system.canopies.keys())
    
    canopy_parents: Dict[str, str] = {}  # canopy_id -> parent_edge_id
    
    for edge_id, edge in system.edges.items():
        # Check n_minus
        n_minus_id, anchor = parse_node_ref(edge.n_minus)
        if anchor is not None:
            # Body anchor reference
            if n_minus_id not in all_body_ids:
                raise ValueError(f"Edge '{edge_id}': n_minus body '{n_minus_id}' does not exist")
            body = system.bodies[n_minus_id]
            if anchor not in body.anchors_B:
                raise ValueError(f"Edge '{edge_id}': anchor '{anchor}' not found on body '{n_minus_id}'")
        else:
            # Canopy reference
            if n_minus_id not in all_canopy_ids:
                raise ValueError(f"Edge '{edge_id}': n_minus canopy '{n_minus_id}' does not exist")
        
        # Check n_plus
        n_plus_id, _ = parse_node_ref(edge.n_plus)
        if n_plus_id in all_body_ids:
            pass  # Bodies can be in n_plus
        elif n_plus_id in all_canopy_ids:
            # Canopy parent check
            if n_plus_id in canopy_parents:
                raise ValueError(f"Canopy '{n_plus_id}' has multiple parents (edges: '{canopy_parents[n_plus_id]}', '{edge_id}')")
            canopy_parents[n_plus_id] = edge_id
        else:
            raise ValueError(f"Edge '{edge_id}': n_plus node '{n_plus_id}' does not exist")
    
    # Check all canopies have a parent
    for canopy_id in all_canopy_ids:
        if canopy_id not in canopy_parents:
            raise ValueError(f"Canopy '{canopy_id}' has no parent edge")
    
    # Check for cycles (simple DFS)
    visited = set()
    rec_stack = set()
    
    def has_cycle(node_id: str, is_canopy: bool) -> bool:
        visited.add(node_id)
        rec_stack.add(node_id)
        
        # Find edges where this node is n_minus
        for edge in system.edges.values():
            n_minus_id, _ = parse_node_ref(edge.n_minus)
            if n_minus_id == node_id:
                n_plus_id, _ = parse_node_ref(edge.n_plus)
                if n_plus_id in rec_stack:
                    return True
                if n_plus_id not in visited and has_cycle(n_plus_id, n_plus_id in all_canopy_ids):
                    return True
        
        rec_stack.remove(node_id)
        return False
    
    for canopy_id in all_canopy_ids:
        if canopy_id not in visited:
            if has_cycle(canopy_id, True):
                raise ValueError(f"Cycle detected in topology involving canopy '{canopy_id}'")


def list_to_array(data: Any) -> Any:
    """Recursively convert lists to numpy arrays"""
    if isinstance(data, list):
        # Check if it's a nested list (matrix)
        if len(data) > 0 and isinstance(data[0], list):
            return np.array(data, dtype=float)
        return np.array(data, dtype=float)
    if isinstance(data, dict):
        return {k: list_to_array(v) for k, v in data.items()}
    return data


def load_config(config_path: str | Path) -> System:
    """Load system configuration from YAML file"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Parse atmosphere
    atmos_data = data.get('atmosphere', {})
    atmos = Atmosphere(
        rho=float(atmos_data.get('rho', 1.225)),
        g=float(atmos_data.get('g', 9.80665)),
        v_air=list_to_array(atmos_data.get('v_air', [0, 0, 0]))
    )
    
    # Parse bodies
    bodies: Dict[str, BodyNode] = {}
    bodies_data = data.get('bodies', {})
    for body_id, body_data in bodies_data.items():
        anchors_B = {}
        for anchor_name, anchor_pos in body_data.get('anchors_B', {}).items():
            anchors_B[anchor_name] = list_to_array(anchor_pos)
        
        bodies[body_id] = BodyNode(
            id=body_id,
            m=float(body_data['m']),
            I_body=list_to_array(body_data['I_body']),
            r=list_to_array(body_data.get('r0', [0, 0, 0])),
            v=list_to_array(body_data.get('v0', [0, 0, 0])),
            q=list_to_array(body_data.get('q0', [1, 0, 0, 0])),
            w=list_to_array(body_data.get('w0', [0, 0, 0])),
            anchors_B=anchors_B,
            aero_force=None,  # User can set this after loading
            aero_moment=None,
            separation_signal_altitude=body_data.get('separation_signal_altitude'),  # m AGL - altitude when signal sent
            separation_lag_time=float(body_data.get('separation_lag_time', 0.0)),  # s - delay from signal to separation
            separation_v_mag=float(body_data.get('separation_v_mag', 6.0)),  # m/s - relative separation velocity
            aero_area=body_data.get('aero_area'),  # m² - body cross-sectional area for drag
            aero_CD=body_data.get('aero_CD')  # Drag coefficient (typically 0.5-1.2 for cylinders)
        )
    
    # Parse canopies
    canopies: Dict[str, CanopyNode] = {}
    canopies_data = data.get('canopies', {})
    for canopy_id, canopy_data in canopies_data.items():
        # Support altitude-based deployment (altitude_deploy in meters AGL)
        # If altitude_deploy is set, td is ignored for deployment trigger (but may be used for other purposes)
        altitude_deploy = canopy_data.get('altitude_deploy')
        if altitude_deploy is not None:
            altitude_deploy = float(altitude_deploy)
        
        canopies[canopy_id] = CanopyNode(
            id=canopy_id,
            p=list_to_array(canopy_data.get('p0', [0, 0, 0])),
            v=list_to_array(canopy_data.get('v0', [0, 0, 0])),
            m_canopy=float(canopy_data['m_canopy']),
            A_inf=float(canopy_data['A_inf']),
            CD_inf=float(canopy_data['CD_inf']),
            td=float(canopy_data.get('td', 0.0)),
            altitude_deploy=altitude_deploy,
            tau_A=float(canopy_data.get('tau_A', 1.0)),
            tau_CD=float(canopy_data.get('tau_CD', 1.0)),
            Ca=float(canopy_data.get('Ca', 1.0)),
            kappa=float(canopy_data.get('kappa', 4.0)),
            upstream_canopy=canopy_data.get('upstream_canopy')  # For slack management
        )
    
    # Parse edges
    edges: Dict[str, LineEdge] = {}
    edges_data = data.get('edges', [])
    for edge_data in edges_data:
        # Parse reefing stages
        reefing_stages = []
        for stage_data in edge_data.get('reefing', []):
            reefing_stages.append(ReefingStage(
                t_on=float(stage_data['t_on']),
                L0=float(stage_data['L0']) if 'L0' in stage_data else None,
                k0=float(stage_data['k0']) if 'k0' in stage_data else None,
                c=float(stage_data['c']) if 'c' in stage_data else None,
                A_scale=float(stage_data['A_scale']) if 'A_scale' in stage_data else None,
                CD_scale=float(stage_data['CD_scale']) if 'CD_scale' in stage_data else None,
                ramp_tau=float(stage_data['ramp_tau']) if 'ramp_tau' in stage_data else None
            ))
        
        edge = LineEdge(
            id=str(edge_data['id']),
            n_minus=str(edge_data['n_minus']),
            n_plus=str(edge_data['n_plus']),
            L0=float(edge_data['L0']),
            k0=float(edge_data['k0']),
            k1=float(edge_data.get('k1', 0.0)),
            alpha=float(edge_data.get('alpha', 1.0)),
            c=float(edge_data['c']),
            T_pre=float(edge_data.get('T_pre', 0.0)),
            T_max=float(edge_data.get('T_max', 1e6)),  # Breaking strength (default high)
            reefing=reefing_stages
        )
        edges[edge.id] = edge
    
    system = System(bodies=bodies, canopies=canopies, edges=edges, atmos=atmos)
    
    # Validate topology
    validate_topology(system)
    
    return system


def get_upstream_edges_for_canopy(system: System, canopy_id: str) -> List[str]:
    """Get list of edge IDs forming the unique upstream path from canopy to root body"""
    upstream_edges = []
    current_node_id = canopy_id
    
    while True:
        # Find edge where current_node is n_plus
        found_edge = None
        for edge_id, edge in system.edges.items():
            n_plus_id, _ = parse_node_ref(edge.n_plus)
            if n_plus_id == current_node_id:
                found_edge = edge_id
                break
        
        if found_edge is None:
            break  # Reached root
        
        upstream_edges.insert(0, found_edge)  # Prepend to build path from root
        
        # Move to n_minus
        edge = system.edges[found_edge]
        current_node_id, _ = parse_node_ref(edge.n_minus)
        
        # If it's a body anchor, we're done
        if ':' in edge.n_minus:
            break
    
    return upstream_edges
