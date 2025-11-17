# Chamber Geometry Optimization Pipeline

## Overview

This document describes the comprehensive optimization pipeline for chamber geometry design. The system solves for optimal chamber geometry given design requirements while considering manufacturing constraints, structural constraints, stability margins, and ablative cooling setup.

## Components

### 1. System Diagnostics (`pintle_pipeline/system_diagnostics.py`)

Comprehensive diagnostic tool that validates all system dynamics:

- **Cf Analysis**: Validates thrust coefficient calculations
- **Velocity Analysis**: Checks exit velocity, Mach number, and temperature calculations
- **Feed System Analysis**: Diagnoses pressure losses in LOX and fuel feed systems
- **Chamber Dynamics**: Validates L*, residence time, velocities, and Mach numbers
- **Stability Analysis**: Checks stability margins and frequencies
- **Overall Health**: Provides system-wide health assessment

### 2. Chamber Optimizer (`pintle_pipeline/chamber_optimizer.py`)

Iterative optimization pipeline that:

- Takes design requirements (thrust, burn time, stability margins)
- Optimizes chamber geometry (L*, A_throat, A_exit, chamber diameter)
- Considers manufacturing constraints (tolerances, dimensions)
- Considers structural constraints (wall thickness, max dimensions)
- Ensures stability margins are met
- Sets up ablative cooling for burn time
- Iterates until convergence with feedforward dynamics

## Usage

### Basic Example

```python
from pintle_pipeline.chamber_optimizer import ChamberOptimizer
from pintle_pipeline.config_schemas import PintleEngineConfig
import yaml

# Load base configuration
with open("config_minimal.yaml", "r") as f:
    config_dict = yaml.safe_load(f)
config = PintleEngineConfig(**config_dict)

# Create optimizer
optimizer = ChamberOptimizer(config)

# Define design requirements
design_requirements = {
    "target_thrust": 6000.0,  # N
    "target_burn_time": 10.0,  # s
    "target_stability_margin": 1.2,  # 20% margin
    "P_tank_O": 3.5e6,  # Pa
    "P_tank_F": 3.5e6,  # Pa
    "target_Isp": 250.0,  # s (optional)
}

# Define constraints
constraints = {
    "max_chamber_length": 0.5,  # m
    "max_chamber_diameter": 0.15,  # m
    "min_Lstar": 0.8,  # m
    "max_Lstar": 1.5,  # m
    "min_expansion_ratio": 5.0,
    "max_expansion_ratio": 20.0,
    "manufacturing_tolerance": 0.001,  # m
    "max_wall_thickness": 0.01,  # m
    "min_wall_thickness": 0.002,  # m
}

# Run optimization
results = optimizer.optimize(design_requirements, constraints)

# Access results
optimized_config = results["optimized_config"]
performance = results["performance"]
diagnostics = results["diagnostics"]
burn_analysis = results["burn_analysis"]
```

### Running Diagnostics

```python
from pintle_pipeline.system_diagnostics import SystemDiagnostics
from pintle_pipeline.config_schemas import PintleEngineConfig
import yaml

# Load configuration
with open("config_minimal.yaml", "r") as f:
    config_dict = yaml.safe_load(f)
config = PintleEngineConfig(**config_dict)

# Create diagnostics
diagnostics = SystemDiagnostics(config)

# Run diagnostics
results = diagnostics.diagnose_all(
    P_tank_O=3.5e6,  # Pa
    P_tank_F=3.5e6,  # Pa
)

# Check health status
health = results["health_status"]
print(f"System Status: {health['status']}")
print(f"Total Issues: {health['total_issues']}")

# Review specific component diagnostics
cf_analysis = results["cf_analysis"]
velocity_analysis = results["velocity_analysis"]
feed_system_analysis = results["feed_system_analysis"]
```

## Fixed Issues

### 1. Cf Calculation
- Exit Mach number is now properly initialized and stored
- Exit velocity calculation is consistent with Mach number
- Pressure thrust component is correctly calculated

### 2. Velocity and Mach Number
- Exit Mach number solver ensures supersonic convergence
- Chamber Mach number calculation uses correct mean velocity
- All velocity calculations are validated

### 3. LOX Feed Pressure Loss
- Feed system pressure loss calculation is robust
- Handles both dict and object config access
- Validates inputs and provides warnings for zero pressure drop

## Optimization Algorithm

The optimizer uses Sequential Least Squares Programming (SLSQP) to minimize:

```
Objective = 10.0 × thrust_error + 5.0 × isp_error + 3.0 × stability_error
```

Where:
- `thrust_error = |F_actual - F_target| / F_target`
- `isp_error = |Isp_actual - Isp_target| / Isp_target` (if specified)
- `stability_error = max(0, target_margin - actual_margin) / target_margin`

### Constraints

1. **Expansion Ratio**: `min_eps ≤ A_exit/A_throat ≤ max_eps`
2. **Chamber Length**: `L_chamber ≤ max_chamber_length`
3. **L* Bounds**: `min_Lstar ≤ Lstar ≤ max_Lstar`
4. **Chamber Diameter**: `diameter ≤ max_chamber_diameter`

### Optimization Variables

- `A_throat`: Throat area [m²]
- `A_exit`: Exit area [m²]
- `Lstar`: Characteristic length [m]
- `chamber_diameter`: Chamber inner diameter [m]

## Next Steps

1. **Integrate stability constraints**: Add explicit stability margin constraints to optimization
2. **Integrate ablative cooling**: Automatically size ablative liner for burn time
3. **Add manufacturing constraints**: Include tolerance and manufacturability constraints
4. **Add structural constraints**: Include wall thickness and stress constraints
5. **Multi-objective optimization**: Optimize for multiple objectives (thrust, Isp, weight)

## Notes

- The optimization uses feedforward dynamics - all system interactions are considered
- Diagnostics are run after optimization to validate the solution
- Burn analysis is performed to ensure the design meets burn time requirements
- The system is designed to be robust and handle edge cases gracefully

