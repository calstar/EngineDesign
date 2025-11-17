# Optimization Framework for Vehicle Design

## Overview

This document describes the closed-loop optimization framework that enables:
1. **User specifies target performance** (e.g., 10k feet altitude)
2. **Flight sim iterates** to find optimal:
   - Tank fill levels
   - COPV pressurant levels
   - Thrust curve
3. **System optimization** minimizes vehicle weight while optimizing:
   - Injector geometry
   - Thrust chamber design
   - System parameters

## Architecture

### 1. Input: Target Performance
```python
target = {
    "altitude": 3048.0,  # 10k feet [m]
    "payload_mass": 10.0,  # [kg]
    "constraints": {
        "max_acceleration": 10.0,  # [g]
        "min_stability_margin": 0.2,
    }
}
```

### 2. Flight Simulation Loop
```python
def optimize_vehicle(target):
    # Initialize design variables
    design_vars = {
        "tank_fill_LOX": 0.8,  # Initial guess
        "tank_fill_fuel": 0.8,
        "COPV_pressure": 3000.0,  # [psi]
        "injector_d_orifice": 0.0015,  # [m]
        "chamber_A_throat": 0.000314,  # [m²]
    }
    
    # Optimization loop
    for iteration in range(max_iterations):
        # 1. Generate thrust curve from design
        thrust_curve = generate_thrust_curve(design_vars)
        
        # 2. Run flight simulation
        flight_result = simulate_flight(thrust_curve, target)
        
        # 3. Check constraints
        if not check_constraints(flight_result, target):
            # Adjust design variables
            design_vars = adjust_design(design_vars, flight_result, target)
            continue
        
        # 4. Calculate objective (minimize weight)
        objective = calculate_vehicle_weight(design_vars)
        
        # 5. Update design (gradient descent, genetic algorithm, etc.)
        design_vars = optimize_design(design_vars, objective, flight_result)
    
    return design_vars, flight_result
```

### 3. Thrust Curve Generation
```python
def generate_thrust_curve(design_vars):
    """
    Generate complete thrust curve using fully-coupled time-varying solver.
    
    This uses the TimeVaryingCoupledSolver which accounts for:
    - Tank pressure decay
    - Ablative/graphite recession
    - Reaction chemistry changes
    - Shifting equilibrium
    - Stability evolution
    """
    # Initialize engine
    config = create_config_from_design(design_vars)
    runner = PintleEngineRunner(config)
    
    # Generate tank pressure profiles
    P_tank_O, P_tank_F, times = simulate_tank_depletion(
        design_vars["tank_fill_LOX"],
        design_vars["tank_fill_fuel"],
        design_vars["COPV_pressure"],
    )
    
    # Run fully-coupled time-varying analysis
    results = runner.evaluate_arrays_with_time(
        times,
        P_tank_O,
        P_tank_F,
        use_coupled_solver=True,  # Use fully-coupled solver
    )
    
    # Extract thrust curve
    thrust_curve = {
        "time": results["time"],
        "thrust": results["F"],
        "Isp": results["Isp"],
        "mass_flow": results["mdot_total"],
        "stability": results.get("chugging_stability_margin", None),
    }
    
    return thrust_curve
```

### 4. Flight Simulation
```python
def simulate_flight(thrust_curve, target):
    """
    Simulate vehicle flight with given thrust curve.
    
    Returns:
    - altitude_achieved
    - max_acceleration
    - burn_time
    - stability_history
    """
    # Use 6-DOF flight dynamics
    # Integrate: F = ma, drag, gravity, etc.
    # Check stability throughout flight
    
    # ... flight simulation code ...
    
    return {
        "altitude": max_altitude,
        "max_acceleration": max_accel,
        "burn_time": burn_time,
        "stability_margins": stability_history,
    }
```

### 5. Design Optimization
```python
def optimize_design(design_vars, objective, flight_result):
    """
    Optimize design variables to minimize weight while meeting constraints.
    
    Uses gradient-based or evolutionary algorithms.
    """
    # Calculate gradients or use genetic algorithm
    # Update design variables
    
    # Key optimization variables:
    # - Injector geometry (affects efficiency, stability)
    # - Chamber geometry (affects L*, efficiency)
    # - Tank fill levels (affects burn time, weight)
    # - COPV pressure (affects pressurization, weight)
    
    return updated_design_vars
```

## Key Features

### Fully-Coupled Analysis
All systems are integrated simultaneously:
- **Reaction chemistry** → affects shifting equilibrium
- **Geometry evolution** → affects chamber dynamics
- **Ablative/graphite recession** → affects geometry
- **Stability analysis** → accounts for all changes

### Time-Varying Everything
- Reaction progress changes as L* evolves
- Shifting equilibrium accounts for reaction chemistry changes
- Stability margins tracked over time
- Geometry evolution affects all downstream calculations

### Constraint Handling
- Stability margins must remain positive
- Maximum acceleration limits
- Minimum performance requirements
- Structural limits

### Weight Minimization
- Optimize tank fill levels (less propellant = less weight)
- Optimize COPV pressure (lower pressure = lighter tank)
- Optimize injector/chamber geometry (better efficiency = less propellant needed)

## Usage Example

```python
from pintle_pipeline.optimization import optimize_vehicle

# User specifies target
target = {
    "altitude": 3048.0,  # 10k feet
    "payload_mass": 10.0,  # kg
}

# Optimize
optimal_design, flight_result = optimize_vehicle(target)

print(f"Optimal altitude: {flight_result['altitude']:.1f} m")
print(f"Vehicle weight: {optimal_design['total_weight']:.1f} kg")
print(f"Tank fill LOX: {optimal_design['tank_fill_LOX']:.2%}")
print(f"Tank fill fuel: {optimal_design['tank_fill_fuel']:.2%}")
```

## Next Steps

1. **Implement flight simulation** (6-DOF dynamics)
2. **Implement optimization algorithms** (gradient descent, genetic algorithm)
3. **Create design variable mapping** (injector geometry, chamber geometry)
4. **Add constraint handling** (stability, acceleration, performance)
5. **Integrate with UI** (visualize optimization progress)

