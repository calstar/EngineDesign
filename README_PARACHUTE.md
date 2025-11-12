# Parachute Dynamics Simulation System

A high-fidelity physics-based simulation engine for multi-body parachute recovery systems. This system models the complete dynamics of parachute deployment, inflation, and descent, including multi-stage deployments, body separation, reefing, and realistic line tension models.

## Overview

**Purpose:** Model parachute recovery systems for mission-critical applications, including human-rated recovery systems.

**Key Features:**
- **Multi-body dynamics**: Rigid bodies with 6-DOF motion (position, velocity, orientation, angular velocity)
- **Multi-canopy systems**: Drogue, pilot, and main parachutes with proper sequencing
- **Realistic inflation models**: Time-varying area and drag coefficient with hyperinflation effects
- **Kelvin-Voigt line tension**: Viscoelastic model with stiffness, damping, preload, and slack/taut conditions
- **Body separation**: Black powder charge separation with momentum conservation
- **Reefing**: Time-varying parachute parameters for controlled inflation
- **Added mass**: Accounts for fluid inertial effects during canopy inflation
- **Event detection**: Automatic detection of deployment, pickup, and separation events
- **Numerical robustness**: Extensive validation and safeguards against numerical instability

## Quick Start

### 1. Installation

```bash
# Install required packages
pip install -r requirements.txt
```

Required packages:
- `numpy` - Numerical computations
- `scipy` - Scientific computing utilities
- `pandas` - Data analysis
- `matplotlib` - Plotting
- `PyYAML` - Configuration file parsing

### 2. Basic Usage

```bash
# Run a simulation
python -m parachute simulate \
    --config examples/parachute/realistic_rocket_deployment.yaml \
    --tf 120 \
    --dt 0.01 \
    --out out/my_simulation

# View results
# Output files:
#   - telemetry.csv: Time history of all states
#   - events.csv: Deployment, pickup, and separation events
#   - peaks.csv: Peak shock loads detected
```

### 3. Example Configuration

See `examples/parachute/realistic_rocket_deployment.yaml` for a complete multi-body rocket deployment scenario with:
- Nosecone, avionics, and motor bodies
- Drogue, pilot, and main parachutes
- Body separation at specified altitude
- Multi-stage deployment sequence

## System Architecture

### Core Components

```
parachute/
├── model.py          # Data models (BodyNode, CanopyNode, System, etc.)
├── physics.py        # Physics equations (inflation, drag, tension, quaternions)
├── engine.py         # Simulation engine (RK4 integration, event management)
├── observers.py      # Telemetry and event logging
└── cli.py            # Command-line interface
```

### Key Physics Models

#### 1. Canopy Inflation
- **Area evolution**: `A(t) = A_inf × f_area(ξ)` where `ξ = (t - t_pickup) / τ_A`
- **Drag coefficient**: `CD(t) = CD_inf × g_cd(ξ)`
- **Hyperinflation**: Canopy can overshoot `A_inf` by up to 20% during rapid inflation
- **Pickup-based**: Inflation only starts after line goes taut (pickup event)

#### 2. Line Tension (Kelvin-Voigt Model)
```
T = k × x + c × ẋ + T_pre
```
Where:
- `k`: Stiffness (varies with reefing)
- `c`: Damping coefficient
- `x`: Extension beyond slack length
- `T_pre`: Preload tension
- `T_max`: Breaking strength (hard limit)

#### 3. Added Mass
```
m_added = Ca × ρ × κ × R³
```
Where:
- `Ca`: Added mass coefficient (~1.0)
- `ρ`: Air density
- `κ`: Volume factor (~4.0)
- `R`: Canopy radius

#### 4. Rigid Body Dynamics
- **Translation**: `F = m × a`
- **Rotation**: `M = I × α + ω × (I × ω)` (Euler's equations)
- **Quaternion integration**: `q̇ = 0.5 × q × [0, ω]`

#### 5. Multi-Stage Deployment
- **Slack management**: Downstream canopies wait for upstream to inflate
- **Pickup detection**: Automatic detection when line goes taut
- **Reefing**: Time-varying parameters (area, drag, line length, stiffness)

## Configuration File Format

### Bodies

```yaml
bodies:
  nosecone:
    m: 2.0                    # Mass [kg]
    I_body:                    # Inertia tensor [kg·m²]
      - [0.5, 0, 0]
      - [0, 0.5, 0]
      - [0, 0, 0.1]
    r0: [0, 0, 3048]          # Initial position [m]
    v0: [0, 0, -50]            # Initial velocity [m/s]
    q0: [1, 0, 0, 0]           # Initial quaternion (scalar-first)
    w0: [0, 0, 0]              # Initial angular velocity [rad/s]
    anchors_B:                 # Anchor points in body frame
      base: [0, 0, 0]
    separation_signal_altitude: 1524.0  # Altitude to trigger separation [m]
    separation_lag_time: 0.1            # Delay from signal to separation [s]
    separation_v_mag: 6.0               # Separation velocity [m/s]
```

### Canopies

```yaml
canopies:
  drogue:
    A_inf: 3.0                # Fully inflated area [m²]
    CD_inf: 1.5                # Fully inflated drag coefficient
    altitude_deploy: 3048.0     # Deployment altitude [m] (or td: 0.0 for time-based)
    tau_A: 0.5                 # Area inflation time constant [s]
    tau_CD: 0.4                # Drag inflation time constant [s]
    m_canopy: 0.3              # Canopy mass [kg]
    Ca: 1.0                    # Added mass coefficient
    kappa: 4.0                 # Volume factor
    p0: [0, 0, 3047.75]        # Initial position [m]
    v0: [0, 0, -48]            # Initial velocity [m/s]
    upstream_canopy: null      # ID of upstream canopy (for slack management)
```

### Edges (Lines)

```yaml
edges:
  - id: leg_drogue
    n_minus: "nosecone:base"   # Minus node (body anchor or canopy)
    n_plus: drogue              # Plus node (canopy or body)
    L0: 0.5                     # Slack length [m]
    k0: 50000                   # Base stiffness [N/m]
    k1: 0                       # Nonlinear stiffness coefficient
    alpha: 1                    # Nonlinear exponent
    c: 500                      # Damping [N/(m/s)]
    T_pre: 0                    # Preload tension [N]
    T_max: 10000                # Breaking strength [N]
    reefing:                    # Optional reefing stages
      - t_start: 0.0
        t_end: 2.0
        A_scale: 0.5
        CD_scale: 0.6
        L0_scale: 0.8
        k_scale: 1.2
```

## Command-Line Interface

### Simulate Command

```bash
python -m parachute simulate \
    --config <config_file>      # YAML configuration file (required)
    --tf <time>                 # Final time [s] (required)
    --t0 <time>                 # Start time [s] (default: 0.0)
    --dt <time>                 # Time step [s] (default: 0.01)
    --out <directory>           # Output directory (required)
    --ramp-shape <exp|tanh>     # Reefing ramp shape (default: exp)
    --no-pickup-shrink          # Disable adaptive step shrinking at pickup
    --verbose                   # Include position/velocity columns in telemetry
    --peak-window <time>        # Peak detection window [s] (default: 5.0)
```

### Output Files

1. **telemetry.csv**: Time history of all system states
   - System forces: `Fsys_x`, `Fsys_y`, `Fsys_z`
   - Body states: `body:<id>:r_x`, `body:<id>:v_x`, etc.
   - Canopy states: `canopy:<id>:A`, `canopy:<id>:CD`, etc.
   - Edge states: `edge:<id>:L`, `edge:<id>:T`, etc.

2. **events.csv**: Discrete events during simulation
   - `deploy_on`: Canopy deployment triggered
   - `pickup`: Line goes taut (inflation starts)
   - `separation`: Body separation event

3. **peaks.csv**: Peak shock loads detected
   - Peak force magnitude and time
   - Per-body peak forces

## Physics Details

### Coordinate System
- **Z-axis**: Altitude (positive = above ground)
- **Gravity**: `gvec = [0, 0, 9.80665]` m/s² (downward)
- **Quaternions**: Scalar-first format `[w, x, y, z]`

### Inflation Model
- **Exponential ramp**: `f_area(ξ) = 1 - exp(-ξ)`
- **Hyperinflation**: Overshoot during rapid inflation (high velocity)
- **Pickup requirement**: Canopy stays packed until line goes taut

### Tension Model
- **Slack**: `T = 0` when `L < L0`
- **Taut**: `T = k×x + c×ẋ + T_pre` when `L ≥ L0`
- **Breaking**: Hard cap at `T_max`
- **Constraint enforcement**: Position correction when extension exceeds realistic limits

### Numerical Stability
- **Quaternion normalization**: After every integration step
- **Force capping**: All forces capped at 1 MN per component
- **Position correction**: Aggressive correction for severe constraint violations
- **Velocity damping**: Reduces relative velocity when constraints violated
- **Finite checks**: Extensive validation of all computed values

## Example Scenarios

### 1. Simple Single-Body Deployment

```yaml
# examples/parachute/simple_test.yaml
bodies:
  rocket:
    m: 18.14
    r0: [0, 0, 3048]
    v0: [67.97, 0, -1.22]

canopies:
  main:
    A_inf: 7.29
    CD_inf: 2.0
    td: 0.0
    tau_A: 0.5
    tau_CD: 0.5

edges:
  - id: shockcord
    n_minus: "rocket:base"
    n_plus: main
    L0: 15.24
    k0: 2000
    T_max: 3000
```

### 2. Multi-Body Rocket with Separation

See `examples/parachute/realistic_rocket_deployment.yaml` for:
- Three-body system (nosecone, avionics, motor)
- Three-stage parachute deployment (drogue → pilot → main)
- Body separation at specified altitude
- Slack management between stages

## Validation and Testing

The system has been extensively tested for:
- ✅ Canopy inflation after pickup
- ✅ Multi-stage deployment sequencing
- ✅ Body separation with momentum conservation
- ✅ Numerical stability under extreme conditions
- ✅ Force capping to prevent explosions
- ✅ Position/velocity constraint enforcement

## Troubleshooting

### Canopy Not Inflating
- **Check pickup events**: Ensure `pickup` events are detected in `events.csv`
- **Check deployment altitude**: Verify `altitude_deploy` is reached
- **Check pickup times**: Ensure `t_pickup` is set correctly in physics functions

### Numerical Instability
- **Reduce time step**: Try `--dt 0.005` or smaller
- **Check forces**: Look for extremely large forces in telemetry
- **Check positions**: Verify positions remain finite
- **Enable verbose**: Use `--verbose` to see position/velocity columns

### Unrealistic Results
- **Check initial conditions**: Verify velocities and positions are reasonable
- **Check line properties**: Ensure `k0`, `c`, and `T_max` are realistic
- **Check canopy parameters**: Verify `A_inf`, `CD_inf`, and `tau_A` are appropriate

### Forces Too High
- **Check breaking strength**: Ensure `T_max` is reasonable for line material
- **Check stiffness**: High `k0` can cause large forces
- **Check damping**: Increase `c` to reduce oscillations

## Recent Improvements

### Inflation Fixes
- ✅ Fixed altitude check to allow inflation after pickup
- ✅ Added validation for canopy position before altitude checks
- ✅ Fixed pickup time propagation through multi-stage systems

### Numerical Robustness
- ✅ Added force capping (1 MN per component)
- ✅ Improved position correction for constraint violations
- ✅ Enhanced velocity damping for severe violations
- ✅ Added extensive finite checks throughout

### Physics Improvements
- ✅ Proper added mass calculation
- ✅ Orientation-dependent drag forces
- ✅ Quaternion normalization after every step
- ✅ Realistic constraint enforcement

## References

- **Parachute Physics**: 
  - Knacke, T.W. "Parachute Recovery Systems Design Manual"
  - Desabrais, K.J. "Modern Parachute Systems"
  
- **Rigid Body Dynamics**:
  - Goldstein, H. "Classical Mechanics"
  - Shabana, A.A. "Dynamics of Multibody Systems"

- **Numerical Methods**:
  - Press, W.H. et al. "Numerical Recipes"
  - Hairer, E. et al. "Solving Ordinary Differential Equations"

## License

See repository for license information.

## Contributing

This system is designed for mission-critical applications. All contributions should maintain numerical robustness and physical accuracy. Please test thoroughly before submitting changes.

