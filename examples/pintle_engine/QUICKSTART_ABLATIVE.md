# Quick Start: Ablative Geometry Evolution

## 30-Second Setup

### 1. Enable in Config

```yaml
# config_minimal.yaml
ablative_cooling:
  enabled: true
  track_geometry_evolution: true
  throat_recession_multiplier: null  # Physics-based (recommended)
```

### 2. Run Time-Series

```python
from pintle_pipeline.io import load_config
from pintle_models.runner import PintleEngineRunner
import numpy as np

config = load_config("config_minimal.yaml")
runner = PintleEngineRunner(config)

# 10 second burn
times = np.linspace(0, 10, 100)
P_O = np.full(100, 1305 * 6894.76)  # Pa
P_F = np.full(100, 974 * 6894.76)   # Pa

# Run with geometry tracking
results = runner.evaluate_arrays_with_time(times, P_O, P_F)

# Check performance degradation
thrust_loss = (results['F'][-1] / results['F'][0] - 1) * 100
print(f"Thrust loss: {thrust_loss:.2f}%")
print(f"L* change: {(results['Lstar'][-1]/results['Lstar'][0]-1)*100:.2f}%")
```

### 3. Run Test

```bash
python test_time_varying_ablation.py
```

---

## What You Get

**New Outputs:**
- `Lstar`: Time-varying L* [m]
- `V_chamber`: Evolving chamber volume [m³]
- `A_throat`: Growing throat area [m²]
- `recession_chamber`: Cumulative recession [m]
- `recession_throat`: Throat recession [m]
- `throat_recession_multiplier`: Physics-based multiplier

**Plots:**
- Thrust degradation over time
- L* evolution
- Ablative recession (chamber & throat)
- Geometry growth percentages
- Chamber pressure evolution
- Throat recession multiplier

---

## Key Parameters

### Material Properties
```yaml
material_density: 1600.0      # kg/m³ (phenolic: 1400-1800)
heat_of_ablation: 2.5e6       # J/kg (phenolic: 2-4 MJ/kg)
pyrolysis_temperature: 950.0  # K (phenolic: 800-1000K)
```

### Throat Recession
```yaml
# Option 1: Physics-based (recommended)
throat_recession_multiplier: null

# Option 2: Fixed multiplier
throat_recession_multiplier: 1.5  # 1.2-2.0 typical
```

### Char Layer
```yaml
char_layer_conductivity: 0.2   # W/(m·K)
char_layer_thickness: 0.001    # m (1mm)
```

---

## Typical Results

**10-second burn, phenolic ablator:**
- Chamber recession: ~500 µm
- Throat recession: ~650 µm (1.3x multiplier)
- L* increase: +1.5%
- Thrust loss: -1.5%
- Isp loss: -1.1%

---

## Troubleshooting

### "No geometry evolution observed"
✅ Check: `track_geometry_evolution: true`
✅ Check: `ablative_cooling.enabled: true`
✅ Use: `evaluate_arrays_with_time()` not `evaluate_arrays()`

### "Throat multiplier is NaN"
✅ Multiplier is only calculated when ablation occurs
✅ Check first time step has valid flow conditions

### "Performance loss too high/low"
✅ Adjust `heat_of_ablation` (higher = less recession)
✅ Adjust `blowing_efficiency` (higher = more protection)
✅ Check `coverage_fraction` (< 1.0 = partial protection)

---

## Documentation

- **Full implementation:** `ABLATIVE_GEOMETRY_IMPLEMENTATION.md`
- **Cooling coupling:** `COOLING_THRUST_COUPLING.md`
- **Physics details:** `pintle_pipeline/ablative_geometry.py`

---

## Done! 🚀

You now have a fully physics-based ablative geometry evolution model integrated into your engine pipeline.

