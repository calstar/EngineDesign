# Comprehensive Thermal Analysis System

## Overview

This guide describes the comprehensive thermal analysis system for ablative and graphite cooling in the pintle engine. The system models:

1. **Multi-layer thermal conduction** (stainless steel + phenolic ablator + graphite insert)
2. **Pyrolysis** with char layer formation
3. **Vaporization** at high temperatures
4. **Material recession** (ablation, oxidation, erosion)
5. **Temperature profiles** through wall thickness
6. **Sizing tools** to determine required material thicknesses

## Architecture

### Core Modules

#### `pintle_pipeline/thermal_analysis.py`
Comprehensive thermal analysis engine:
- `MaterialLayer`: Defines material properties (k, ρ, cp, emissivity)
- `ThermalBoundaryConditions`: Hot gas and ambient boundary conditions
- `calculate_steady_state_temperature_profile()`: Multi-layer temperature profiles
- `calculate_pyrolysis_response()`: Pyrolysis modeling with char formation
- `calculate_vaporization_rate()`: High-temperature vaporization
- `analyze_multi_layer_system()`: Complete thermal analysis
- `calculate_required_ablative_thickness()`: Sizing function

#### `pintle_pipeline/ablative_sizing.py`
Sizing tools for ablative systems:
- `size_ablative_system()`: Complete ablative liner sizing
- `size_graphite_insert()`: Graphite insert sizing wrapper

#### Enhanced Modules
- `pintle_pipeline/ablative_cooling.py`: Enhanced with pyrolysis/vaporization
- `pintle_pipeline/graphite_cooling.py`: Oxidation and thermal ablation
- `pintle_pipeline/ablative_geometry.py`: Geometry evolution tracking

## Physics Models

### 1. Thermal Conduction

Multi-layer steady-state conduction using thermal resistance network:

```
q = (T_hot - T_cold) / R_total
R_total = R_conv_hot + Σ(R_cond_layers) + R_conv_cold
R_cond = t / (k * A)
```

Temperature profile through each layer:
```
T(x) = T_surface - (q * x / k)
```

### 2. Pyrolysis

Phenolic ablators decompose when T > T_pyrolysis (~950 K):

- **Virgin material** → **Char + Pyrolysis gases**
- Char layer forms on surface (lower conductivity: ~0.2 W/(m·K))
- Pyrolysis gases flow outward (blowing effect reduces heat flux)
- Energy consumption: ~2 MJ/kg

Pyrolysis rate:
```
ṁ_pyro = f(T_surface, T_pyrolysis) * ṁ_max
```

### 3. Vaporization

At very high temperatures (T > T_limit * 1.05), material directly vaporizes:

- No char formation (unlike pyrolysis)
- High energy consumption: ~10 MJ/kg
- Clausius-Clapeyron-like behavior (higher T, lower P → higher rate)

Vaporization rate:
```
ṁ_vap = ṁ_max * exp(-E_a / T) * (P_ref / P)^0.5
```

### 4. Graphite Oxidation

Graphite inserts oxidize when T > 800 K:

- Chemical reaction: C + O₂ → CO₂
- Oxidation rate increases with temperature and pressure
- Dominant mechanism at high temperatures (vs. thermal ablation)

Oxidation rate:
```
ṙ_ox = ṙ_ref * (1 + 10 * T_ratio) * (P / P_ref)^0.5
```

### 5. Material Recession

Total recession rate:
```
ṙ_total = ṙ_thermal + ṙ_pyrolysis + ṙ_vaporization + ṙ_oxidation
```

Where:
- `ṙ_thermal = q_net / (ρ * H_ablation)`
- `ṙ_pyrolysis = ṁ_pyro / ρ`
- `ṙ_vaporization = ṁ_vap / ρ`
- `ṙ_oxidation = f(T, P)` (graphite only)

## Usage

### Basic Thermal Analysis

```python
from pintle_pipeline.thermal_analysis import (
    MaterialLayer,
    ThermalBoundaryConditions,
    analyze_multi_layer_system,
)
from pintle_pipeline.config_schemas import AblativeCoolingConfig

# Define material layers (hot to cold)
ablative = MaterialLayer(
    name="Phenolic Ablator",
    thickness=0.01,  # 10 mm
    thermal_conductivity=0.35,  # W/(m·K)
    density=1600.0,  # kg/m³
    specific_heat=1500.0,  # J/(kg·K)
    pyrolysis_temp=950.0,  # K
)

stainless = MaterialLayer(
    name="Stainless Steel",
    thickness=0.002,  # 2 mm
    thermal_conductivity=15.0,
    density=8000.0,
    specific_heat=500.0,
)

layers = [ablative, stainless]

# Boundary conditions
bc = ThermalBoundaryConditions(
    T_hot_gas=3500.0,  # K
    h_hot_gas=5000.0,  # W/(m²·K)
    q_rad_hot=0.0,  # W/m²
    T_ambient=300.0,  # K
    h_ambient=10.0,  # W/(m²·K)
)

# Run analysis
results = analyze_multi_layer_system(
    layers,
    bc,
    ablative_config=ablative_config,
)

print(f"Hot surface temp: {results['T_surface_hot']:.1f} K")
print(f"Backface temp: {results['T_backface']:.1f} K")
print(f"Heat flux: {results['heat_flux']/1e6:.2f} MW/m²")
```

### Ablative Sizing

```python
from pintle_pipeline.ablative_sizing import size_ablative_system

sizing = size_ablative_system(
    heat_flux=2.0e6,  # W/m² (2 MW/m²)
    burn_time=10.0,  # seconds
    ablative_config=config.ablative_cooling,
    backface_temp_limit=500.0,  # K
)

print(f"Required thickness: {sizing['required_thickness']*1000:.2f} mm")
print(f"Recession allowance: {sizing['recession_allowance']*1000:.2f} mm")
print(f"Meets requirements: {sizing['meets_requirements']}")
```

### Graphite Insert Sizing

```python
from pintle_pipeline.ablative_sizing import size_graphite_insert

graphite_sizing = size_graphite_insert(
    peak_heat_flux=3.0e6,  # W/m² (throat)
    surface_temperature=2000.0,  # K
    recession_rate=1e-5,  # m/s
    burn_time=10.0,  # seconds
    graphite_config=config.graphite_insert,
)

sizing = graphite_sizing['sizing']
print(f"Initial thickness: {sizing.initial_thickness*1000:.2f} mm")
print(f"Total axial length: {sizing.total_axial_length*1000:.2f} mm")
```

## Example Script

Run the comprehensive demonstration:

```bash
python examples/pintle_engine/thermal_analysis_demo.py
```

This script:
1. Runs the engine to get operating conditions
2. Estimates heat flux (chamber and throat)
3. Sizes the ablative liner
4. Sizes the graphite insert
5. Calculates temperature profiles
6. Creates visualization plots

## Configuration

### Ablative Configuration (config_minimal.yaml)

```yaml
ablative_cooling:
  enabled: true
  material_density: 1600.0  # kg/m³
  heat_of_ablation: 2.5e6  # J/kg
  thermal_conductivity: 0.35  # W/(m·K)
  specific_heat: 1500.0  # J/(kg·K)
  initial_thickness: 0.01  # m (10 mm)
  surface_temperature_limit: 1200.0  # K
  pyrolysis_temperature: 950.0  # K
  blowing_efficiency: 0.75
  char_layer_conductivity: 0.2  # W/(m·K)
  char_layer_thickness: 0.001  # m (1 mm)
```

### Graphite Configuration

```yaml
graphite_insert:
  enabled: true
  material_density: 1800.0  # kg/m³
  heat_of_ablation: 8.0e6  # J/kg
  thermal_conductivity: 100.0  # W/(m·K)
  specific_heat: 710.0  # J/(kg·K)
  initial_thickness: 0.005  # m (5 mm)
  surface_temperature_limit: 2500.0  # K
  oxidation_temperature: 800.0  # K
  oxidation_rate: 1e-6  # m/s
```

## Key Outputs

### Thermal Analysis Results

- `T_surface_hot`: Hot surface temperature [K]
- `T_backface`: Backface temperature [K]
- `heat_flux`: Total heat flux [W/m²]
- `temperature_profile`: Full temperature profile through wall
- `pyrolysis`: Pyrolysis response (rate, char thickness, etc.)
- `vaporization`: Vaporization response (if active)

### Sizing Results

- `required_thickness`: Total required thickness [m]
- `recession_allowance`: Material lost to recession [m]
- `conduction_thickness`: Thickness for thermal protection [m]
- `safety_margin`: Recommended safety margin [m]
- `meets_requirements`: Whether design meets all requirements

## Design Workflow

1. **Estimate heat flux** from engine operating conditions (Pc, Tc, mdot)
2. **Size ablative liner** using `size_ablative_system()`
3. **Size graphite insert** using `size_graphite_insert()`
4. **Verify thermal limits** (backface temp, surface temp)
5. **Check recession** (throat area change, material consumption)
6. **Iterate** if requirements not met

## Next Steps

- [ ] Integrate thermal analysis into runner for time-varying analysis
- [ ] Add transient thermal response (not just steady-state)
- [ ] Enhance pyrolysis model with detailed chemistry
- [ ] Add thermal diagnostics to Streamlit UI
- [ ] Create thermal visualization tools

