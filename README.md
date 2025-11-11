# Pintle Injector Liquid Rocket Engine Pipeline

A comprehensive physics-based simulation pipeline that takes tank pressures as input and solves for chamber pressure, mass flow rates, and thrust output. The pipeline models the complete flow path from tank to nozzle, including feed system losses, injector flow, spray physics, combustion, and nozzle expansion.

## Overview

**Input:** Tank pressures (LOX and RP-1)  
**Output:** Thrust, mass flow rates, chamber pressure, and all performance parameters

**Key Features:**
- Chamber pressure (Pc) is **never** an input - it's always **solved** from tank pressures by balancing supply and demand.
- Modular injector architecture: pintle (LOX axial / fuel annulus), shear-coaxial, and impinging doublet models with their own flow and spray physics.
- Independent cooling models: regenerative channels, film cooling films, and ablative liners can be evaluated separately or in combination.
- **Time-varying ablative geometry:** Tracks recession and updates chamber volume, throat area, and L* over time for accurate performance predictions.

## Quick Start

### 1. Installation

```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from pathlib import Path
from pintle_pipeline.io import load_config
from pintle_models.runner import PintleEngineRunner

# Load configuration
config_path = Path("examples/pintle_engine/config_minimal.yaml")
config = load_config(str(config_path))

# Initialize runner
runner = PintleEngineRunner(config)

# Evaluate at specific tank pressures
P_tank_O = 1305 * 6894.76  # psi to Pa
P_tank_F = 974 * 6894.76   # psi to Pa

results = runner.evaluate(P_tank_O, P_tank_F)

print(f"Thrust: {results['F']/1000:.2f} kN")
print(f"Chamber Pressure: {results['Pc']/6894.76:.1f} psi")
print(f"Mass Flow: {results['mdot_total']:.3f} kg/s")
print(f"Mixture Ratio: {results['MR']:.2f}")
```

### 3. Run Example Scripts

```bash
# Run full pipeline analysis
cd examples/pintle_engine
python run_full_pipeline.py

# Generate all performance plots
python run_all_plots.py

# Interactive CLI (forward & inverse modes)
python interactive_pipeline.py

# Streamlit UI (forward & inverse modes, time-series analysis)
streamlit run ui_app.py

# Streamlit UI features:
#  - Edit key configuration parameters from the sidebar
#  - Upload alternate YAML configs
#  - Forward solver: Tank pressures → Performance
#  - Inverse solvers:
#    * Thrust only: Find tank pressures for target thrust (1D, scales baseline)
#    * Thrust + O/F: Find tank pressures for target thrust AND O/F ratio (2D, independent pressures)
#  - Time-series designer (analytic profiles, CSV upload, blowdown curves)
#  - Custom plot builder (select any variables and chart style, including heatmaps & contours)
#  - Built-in dashboards for cooling, thrust, Pc, mdot, MR, etc.
#  - Dynamic unit selection (psi/kPa/MPa for pressure, mm/m for length, kg/s for mass flow)
#  - Efficiency coupling controls (mixture, cooling, turbulence) with adjustable floors

# Pressure sweep (2D grid)
python pressure_sweep_example.py

# Validate against Huzel & Huang data
python validate_chamber_intrinsics.py
```

## Pipeline Workflow

The pipeline follows this sequence:

```
1. INPUT: Tank Pressures (P_tank_O, P_tank_F)
   ↓
2. Feed System Losses
   - Calculate pressure drops from tank to injector
   - Includes pipe friction, fittings, and regenerative cooling (fuel only)
   ↓
3. Injector Flow
   - Calculate mass flow rates through pintle injector
   - LOX: Axial flow through N orifices
   - Fuel: Radial flow through annulus gap
   - Dynamic discharge coefficients (Reynolds-dependent)
   ↓
4. Spray Physics
   - Calculate momentum flux ratio (J)
   - Validate atomization (Weber numbers, Sauter mean diameter)
   - Check evaporation length constraints
   ↓
5. Chamber Solver
   - Balance mass flow supply (from injectors) = demand (from combustion)
   - Solve for equilibrium chamber pressure Pc
   - Uses iterative root-finding (brentq)
   ↓
6. Combustion
   - Get thermochemical properties from CEA (c*, Tc, gamma, etc.)
   - Apply chamber-driven corrections (L*-based efficiency)
   - Account for finite chamber volume effects
   ↓
7. Nozzle Expansion
   - Calculate exit conditions (pressure, temperature, velocity)
   - Solve supersonic area-Mach relation
   - Calculate thrust: F = mdot × v_exit + (P_exit - Pa) × A_exit
   ↓
8. OUTPUT: Thrust, Isp, and all performance metrics
```

## Configuration

All engine parameters are defined in `examples/pintle_engine/config_minimal.yaml`. Key sections:

### Fluid Properties
- LOX: density, viscosity, surface tension
- RP-1: density, viscosity, surface tension

### Pintle Geometry
- **LOX**: Number of orifices, diameter, angle
- **Fuel**: Pintle tip diameter, gap height

### Feed System
- Pipe diameters, lengths, loss coefficients
- Regenerative cooling (optional): channel dimensions, number of channels

### Discharge Coefficients
- Dynamic model: `Cd(Re) = Cd_∞ - a_Re/√Re`
- Optional pressure and temperature corrections

### Combustion
- CEA propellant names and conditions
- L*-based efficiency model
- Chamber volume and throat area

### Nozzle
- Expansion ratio, exit area
- Ambient pressure

## Key Physics Models

### 1. Feed System Losses
Generalized model: `Δp_feed = K_eff(P) × (ρ/2) × (ṁ/(ρ×A_hyd))²`

### 2. Cooling Models (Optional)
#### Regenerative Cooling (Channels)
Models pressure drop through:
- Inlet pipe
- Manifold split (into N parallel channels)
- Parallel cooling channels (friction + entrance/exit losses)
- Manifold merge
- Outlet pipe

Dynamic discharge coefficients are applied to channel entrance and exit losses.

#### Film Cooling
- Allocates a configurable fraction of fuel mass flow to a wall film.
- Effectiveness controls the heat-flux reduction factor without altering injector hydraulics.

#### Ablative Cooling
- Energy balance between imposed heat flux and ablator recession rate.
- Supports radiative relief and surface temperature constraints.

### 3. Injector Flow
Standard orifice flow: `ṁ = Cd × A × √(2 × ρ × Δp)`

- **LOX**: Axial flow through N orifices
- **Fuel**: Radial flow through annulus gap
- Dynamic Cd varies with Reynolds number

### 4. Spray Physics
- Momentum flux ratio: `J = (ρ_O × u_O²) / (ρ_F × u_F²)`
- Spray angle: `tan(θ/2) = k × J^n`
- Weber numbers for atomization validation
- Sauter Mean Diameter (SMD) for droplet size
- Evaporation length constraints

### 5. Chamber Solver
Root-finding problem: `supply(Pc) - demand(Pc) = 0`

- **Supply**: Mass flow from injectors (depends on P_tank - Pc)
- **Demand**: Mass flow required by combustion (depends on Pc, MR, c*)

Solves iteratively using `brentq` method.

### 6. Combustion Efficiency
L*-based correction: `η_c* = 1 - C × e^(-K×L*)`

Where `L* = V_chamber / A_throat` (characteristic length)

### 7. Nozzle Thrust
High-fidelity calculation:
- Momentum thrust: `ṁ × v_exit`
- Pressure thrust: `(P_exit - Pa) × A_exit`
- Exit conditions solved from supersonic area-Mach relation

## Example Scripts

### `run_full_pipeline.py`
Complete performance analysis at a single operating point. Shows:
- Chamber performance (Pc, mdot, MR, thrust, Isp)
- Injector performance (pressure drops, velocities, Cd)
- Spray diagnostics (J, θ, Weber numbers, SMD)
- Comparison to target specifications

### `comprehensive_performance_plots.py`
Generates 16-panel dashboard showing all performance metrics and 2D pressure sweep plots.

### `pressure_sweep_example.py`
Evaluates engine performance across a 2D grid of tank pressures (200-1200 psi). Creates contour plots of:
- Thrust
- Chamber pressure
- Mixture ratio
- Specific impulse
- Mass flow
- Characteristic velocity

### `validate_chamber_intrinsics.py`
Validates combustion properties (Tc, c*, gamma) against Huzel & Huang reference data.

### `compare_ideal_vs_actual.py`
Compares CEA ideal (infinite-area equilibrium) vs. actual chamber-driven performance.

### `chamber_3d_plots.py`
Creates 3D visualizations of chamber behavior across pressure ranges.

## Output Interpretation

### Key Metrics

- **Thrust [N]**: Total engine thrust (momentum + pressure)
- **Specific Impulse [s]**: `Isp = F / (ṁ × g₀)`
- **Chamber Pressure [Pa]**: Solved equilibrium pressure
- **Mass Flow [kg/s]**: Total propellant flow rate
- **Mixture Ratio (O/F)**: Oxidizer-to-fuel mass ratio
- **c* [m/s]**: Characteristic velocity (actual, accounting for efficiency)
- **Exit Velocity [m/s]**: Nozzle exit velocity
- **Exit Pressure [Pa]**: Nozzle exit pressure

### Spray Diagnostics

- **J**: Momentum flux ratio (affects spray angle)
- **θ**: Spray angle [degrees]
- **We_O, We_F**: Weber numbers (atomization quality)
- **D32_O, D32_F**: Sauter Mean Diameter [μm] (droplet size)
- **x***: Evaporation length [mm] (mixing quality)

### Pressure Breakdown

For each propellant:
- `P_tank`: Tank pressure (input)
- `Δp_feed`: Feed system pressure loss
- `Δp_regen`: Regenerative cooling pressure loss (fuel only)
- `P_injector`: Pressure at injector inlet
- `Δp_injector`: Pressure drop across injector
- `Pc`: Chamber pressure (solved)

## File Structure

```
pintle_models/          # Core engine models
  ├── chamber_solver.py    # Solves for Pc (supply = demand)
  ├── closure.py           # Iterative flow solver with spray constraints
  ├── discharge.py         # Dynamic Cd model
  ├── geometry.py          # Injector geometry calculations
  ├── nozzle.py            # Thrust calculation
  ├── runner.py            # Main pipeline orchestrator
  └── spray.py             # Spray physics models

pintle_pipeline/        # Pipeline infrastructure
  ├── cea_cache.py         # CEA data caching and interpolation
  ├── combustion_eff.py    # L*-based efficiency models
  ├── config_schemas.py    # Pydantic validation schemas
  ├── feed_loss.py         # Feed system pressure loss model
  ├── io.py                # Configuration loading
  ├── regen_cooling.py     # Regenerative cooling model
  └── visualization.py     # Plotting utilities

examples/pintle_engine/  # Example scripts and configs
  ├── config_minimal.yaml      # Engine configuration
  ├── run_full_pipeline.py     # Complete analysis
  ├── comprehensive_performance_plots.py  # 16-panel dashboard
  ├── pressure_sweep_example.py          # 2D pressure sweeps
  ├── validate_chamber_intrinsics.py     # Validation plots
  ├── compare_ideal_vs_actual.py         # CEA comparison
  ├── chamber_3d_plots.py                # 3D visualizations
  └── pintle_engine_physics.tex           # LaTeX physics documentation
```

## Important Notes

1. **Chamber Pressure is Solved, Not Input**: The pipeline solves for Pc by balancing supply and demand. Never input Pc directly.

2. **Tank Pressures are Inputs**: Provide `P_tank_O` and `P_tank_F` in Pascals (or convert from psi: `P_Pa = P_psi × 6894.76`).

3. **CEA Cache**: First run builds a cache file (`cea_cache_LOX_RP1.npz`). This is slow but only happens once.
   - Cache metadata is checked automatically. If you change propellants, mixture range, or expansion ratio, the cache is rebuilt for you.

4. **Cooling Models**: Regenerative, film, and ablative cooling are independent. Enable only what you need in `config_minimal.yaml`.

5. **Dynamic Discharge Coefficients**: Cd varies with Reynolds number. The model is `Cd(Re) = Cd_∞ - a_Re/√Re`, with optional pressure and temperature corrections.

6. **Spray Constraints**: The pipeline validates spray quality (Weber numbers, evaporation length). If constraints are violated, Cd is reduced to enforce minimum requirements.

## Troubleshooting

### Solver Fails
- Check that tank pressures are high enough (Pc must be < P_tank)
- Verify injector geometry is reasonable
- Check that feed losses aren't too high

### Unrealistic Results
- Verify fluid properties (density, viscosity)
- Check injector geometry (areas, diameters)
- Review discharge coefficients
- Validate against known operating points

### Performance Doesn't Match Expectations
- Check mixture ratio (MR) - it affects c* significantly
- Verify chamber-driven corrections (L*, efficiency)
- Review nozzle expansion ratio
- Check spray constraints (may be reducing Cd)

### L* Changes Don't Affect Performance
If changing the characteristic length (L*) in the UI doesn't visibly change thrust or Isp, this is likely due to **efficiency coupling floors**. The combustion efficiency model includes three coupling factors:
- **Mixture efficiency**: Based on spray quality (SMD, evaporation length, Weber number)
- **Cooling efficiency**: Based on heat removed by cooling systems
- **Turbulence efficiency**: Based on injector and chamber turbulence intensity

Each coupling has a **floor** (minimum efficiency) that prevents the total efficiency from dropping below a certain value. For example, if `mixture_efficiency_floor = 0.25`, the mixture efficiency will never be less than 25%, even if spray quality is poor.

**Solution:**
1. In the UI, navigate to "Combustion & Efficiency" section
2. Look for "Efficiency Floors" (now prominently displayed)
3. Lower the floors (e.g., to 0.0) to remove the clamping effect
4. Or disable the coupling entirely by unchecking "Mixture coupling", "Cooling coupling", or "Turbulence coupling"

This will allow L* changes to have a more visible impact on performance. The default floors (0.25-0.30) are conservative and represent physically realistic lower bounds for well-designed engines.

## References

- Huzel & Huang: "Design of Liquid Propellant Rocket Engines" (chamber intrinsics validation)
- Sutton & Biblarz: "Rocket Propulsion Elements" (combustion efficiency, nozzle theory)
- Lefebvre: "Atomization and Sprays" (spray physics, SMD correlations)

## License

See repository for license information.

