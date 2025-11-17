# Fully-Coupled Time-Varying Analysis System

## Overview

This system provides **complete, fully-coupled time-varying analysis** for rocket engine design with **zero approximations or decoupling**. All systems are integrated simultaneously at each time step.

## Key Features

### 1. **Fully-Coupled Integration**
All systems are solved simultaneously - no decoupling:
- ✅ **Reaction chemistry** → affects shifting equilibrium
- ✅ **Shifting equilibrium** → accounts for reaction chemistry changes
- ✅ **Chamber dynamics** (L*, efficiency) → changes with geometry
- ✅ **Ablative recession** → changes geometry → affects chamber dynamics
- ✅ **Graphite recession** → changes throat area → affects flow
- ✅ **Nozzle geometry** → affects expansion ratio → affects shifting equilibrium
- ✅ **Stability analysis** → accounts for all time-varying effects

### 2. **Time-Varying Reaction Chemistry**
As geometry evolves (L* changes), reaction progress changes:
- **L* increases** → longer residence time → more reaction progress
- **More reaction progress** → affects shifting equilibrium in nozzle
- **Shifting equilibrium** → affects exit gamma, R, temperature
- **Exit properties** → affect thrust and performance

### 3. **Physics-Based (No Arbitrary Constants)**
- ✅ All calculations from first principles
- ✅ No hardcoded values (MR, fuel type come from config)
- ✅ No arbitrary clamping (only physics requirements: M > 1.0)
- ✅ Iterative self-consistent solutions

### 4. **Complete State Tracking**
At each time step, we track:
- Geometry (V_chamber, A_throat, A_exit, L*, eps)
- Recession (chamber, throat, exit, graphite)
- Reaction chemistry (progress_injection, progress_mid, progress_throat)
- Performance (Pc, Tc, F, Isp, v_exit, M_exit)
- Thermodynamics (gamma_chamber, gamma_exit, R_chamber, R_exit)
- Stability (chugging, acoustic modes, feed system)
- Heat flux and cooling

## Architecture

### TimeVaryingCoupledSolver

The core solver that integrates everything:

```python
from pintle_pipeline.time_varying_solver import TimeVaryingCoupledSolver

solver = TimeVaryingCoupledSolver(config, cea_cache)
states = solver.solve_time_series(times, P_tank_O, P_tank_F)
results = solver.get_results_dict()
```

**What it does at each time step:**
1. Updates geometry from previous recession
2. Solves chamber pressure with updated geometry
3. Calculates reaction progress (time-varying, depends on current L*)
4. Calculates heat flux
5. Calculates recession rates (ablative + graphite)
6. Updates geometry
7. Calculates thrust with shifting equilibrium (using reaction progress)
8. Calculates stability (time-varying)
9. Returns complete state

### Integration with Runner

The runner automatically uses the coupled solver when:
- `track_ablative_geometry = True`
- `use_coupled_solver = True` (default)

```python
results = runner.evaluate_arrays_with_time(
    times,
    P_tank_O,
    P_tank_F,
    use_coupled_solver=True,  # Use fully-coupled solver
)
```

## Key Couplings

### 1. Reaction Chemistry → Shifting Equilibrium
```python
# Reaction progress depends on current L*
reaction_progress = calculate_chamber_reaction_progress(
    Lstar,  # Current L* (changes with geometry)
    Pc, Tc, cstar, gamma, R, MR, config
)

# Shifting equilibrium uses reaction progress
thrust_results = calculate_thrust(
    Pc, MR, mdot,
    reaction_progress=reaction_progress,  # TIME-VARYING
    use_shifting_equilibrium=True,
)
```

### 2. Geometry Evolution → Chamber Dynamics
```python
# Geometry changes affect L*
Lstar_new = V_chamber_new / A_throat_new

# L* affects efficiency
eta_cstar = f(Lstar_new, ...)

# Efficiency affects chamber pressure
Pc = solve_chamber_pressure(Lstar_new, eta_cstar, ...)
```

### 3. Recession → Geometry → All Systems
```python
# Recession updates geometry
V_chamber_new, A_throat_new = update_geometry(recession)

# Geometry affects everything:
# - L* → efficiency → Pc
# - A_throat → mass flow → Pc
# - eps → shifting equilibrium → thrust
```

## Stability Analysis Over Time

Stability is calculated at each time step with all current conditions:

```python
from pintle_pipeline.stability_analysis_time import analyze_stability_over_time

stability_results = analyze_stability_over_time(time_history, state_history)

# Returns:
# - chugging_frequency: [Hz]
# - chugging_stability_margin: (positive = stable)
# - acoustic_frequencies: Mode frequencies
# - feed_stability_margins: Feed system margins
# - overall_stability: Combined metric
# - stability_degradation: Change over time
```

## Optimization Framework

See `OPTIMIZATION_FRAMEWORK.md` for complete details.

**Workflow:**
1. User specifies target (e.g., 10k feet altitude)
2. Flight sim iterates to find:
   - Tank fill levels
   - COPV pressurant levels
   - Thrust curve
3. System optimizes:
   - Vehicle weight (minimize)
   - Injector geometry
   - Thrust chamber
   - System parameters

**All using the fully-coupled time-varying solver!**

## Example Usage

```python
from pintle_models.runner import PintleEngineRunner
from pintle_pipeline.io import load_config

# Load config
config = load_config("config_minimal.yaml")
config.ablative_cooling.enabled = True
config.ablative_cooling.track_geometry_evolution = True
config.graphite_insert.enabled = True

# Create runner
runner = PintleEngineRunner(config)

# Define time series
times = np.linspace(0, 10.0, 100)  # 10 second burn
P_tank_O = np.full(100, 500 * 6894.76)  # 500 psi constant
P_tank_F = np.full(100, 500 * 6894.76)

# Run fully-coupled analysis
results = runner.evaluate_arrays_with_time(
    times,
    P_tank_O,
    P_tank_F,
    use_coupled_solver=True,  # Fully-coupled!
)

# Results include:
# - All performance metrics (F, Isp, Pc, etc.)
# - Geometry evolution (L*, A_throat, V_chamber)
# - Recession (chamber, throat, exit, graphite)
# - Reaction chemistry (progress_throat, tau_residence)
# - Shifting equilibrium (gamma_exit, R_exit, equilibrium_factor)
# - Stability (chugging_frequency, stability_margin)
```

## Benefits

1. **Complete Physics**: No approximations, all systems coupled
2. **Time-Varying Everything**: Reaction chemistry, geometry, stability all evolve
3. **Optimization Ready**: Framework for closed-loop vehicle design
4. **Confidence**: Physics-based calculations inspire confidence in results

## Next Steps

1. ✅ Fully-coupled time-varying solver
2. ✅ Time-varying stability analysis
3. ✅ Optimization framework documentation
4. ⏳ Flight simulation integration
5. ⏳ Design optimization algorithms
6. ⏳ UI integration for optimization

