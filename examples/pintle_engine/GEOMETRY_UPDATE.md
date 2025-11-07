# Geometry Update Summary

## Corrected Geometry Values

### LOX (Oxidizer)
- **Number of orifices**: 16
- **Area per orifice**: 0.00385154962 in² = 2.485 mm²
- **Equivalent diameter**: 1.779 mm
- **Total LOX area**: 39.77 mm²

### Fuel (Annulus)
- **Pintle tip diameter**: 0.750 in = 19.05 mm
- **Gap thickness**: 0.313 mm (corrected from 0.86 mm)
- **Fuel area (annulus)**: 19.04 mm²
- **Hydraulic diameter**: 0.626 mm

### Area Ratio
- **LOX/Fuel ratio**: 2.089 (LOX area is larger, which is correct for higher O/F)

## Updated Config Values

```yaml
pintle_geometry:
  lox:
    n_orifices: 16
    d_orifice: 0.001779  # m (1.779 mm)
    d_hydraulic: 0.001779  # m
  
  fuel:
    d_pintle_tip: 0.019050  # m (19.05 mm)
    h_gap: 0.000313  # m (0.313 mm)
    d_hydraulic: 0.000626  # m (0.626 mm)
    d_reservoir_inner: 0.019676  # m (D_p + 2×h_gap)
```

## Notes

1. **Injector pressure drop is NOT constant at 150 psi** - it varies with tank and chamber pressures as the system solves for equilibrium.

2. The corrected geometry gives more realistic results:
   - LOX area increased from 28.65 mm² to 39.77 mm²
   - Fuel area decreased from 56.36 mm² to 19.04 mm²
   - This results in a higher O/F ratio (closer to target of 2.36)

3. The pipeline correctly solves for the actual operating condition, which may have different pressure drops than any nominal design target.

