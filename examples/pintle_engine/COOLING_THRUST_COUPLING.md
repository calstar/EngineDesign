## How Cooling Affects Thrust - Complete Physics Coupling

### Summary
**YES, cooling DOES affect thrust!** The coupling is complex and involves multiple pathways. Here's the complete picture:

---

## 1. Regenerative Cooling → Thrust

### Direct Pressure Drop Effect
```
P_tank_F → [Feed losses] → [Regen channels] → P_inj_F
              ~50-100 psi      ~90 psi
                        ↓
            P_inj_F = P_tank_F - 140-190 psi
                        ↓
            ΔP_injector = P_inj_F - Pc  (REDUCED)
                        ↓
            mdot_F = Cd × A_fuel × √(2 × ρ_F × ΔP_injector)  (LOWER)
                        ↓
            O/F ratio INCREASES
                        ↓
            c* changes (depends on O/F curve)
                        ↓
            Pc changes → Thrust changes
```

**Implementation:** ✅ Correctly coupled in `pintle_models/injectors/pintle.py` lines 110-123

### Cooling Efficiency Factor
```
Heat removed by regen → Q_removed
                ↓
η_cooling = 1 - (Q_removed / (ṁ_total × cp × Tc))
                ↓
η_c* = η_L* × η_mixture × η_cooling × η_turbulence
                ↓
c*_actual = c*_ideal × η_c*  (REDUCED)
                ↓
Lower c* → Lower Pc → Lower thrust
```

**Implementation:** ✅ Correctly coupled in `chamber_solver.py` `_compute_cooling_efficiency()`

**Typical Impact:** 1-5% thrust reduction depending on regen pressure drop and heat removal

---

## 2. Film Cooling → Thrust

### Mass Diversion Effect
```
mdot_F_total → [Film mass fraction] → mdot_film
                                    ↓
                        mdot_combustion = mdot_F_total - mdot_film
                                    ↓
                        Less propellant burning efficiently
                                    ↓
                        DIRECT thrust reduction
```

**Implementation:** ✅ Correctly coupled in `film_cooling.py`

### Cooling Efficiency Factor
```
Film effectiveness → Reduces wall heat flux
                ↓
Heat removed → Cooling efficiency factor
                ↓
η_c* reduced → Thrust reduced
```

**Typical Impact:** 2-8% thrust reduction for 5-10% film mass fraction

---

## 3. Ablative Cooling → Thrust

### ⚠️ **CRITICAL MISSING COUPLING: Geometry Changes**

Ablative recession changes chamber geometry over time:

```
Ablative recession rate (µm/s)
        ↓
Total recession = recession_rate × burn_time
        ↓
Chamber diameter: D_new = D_initial + 2×recession
Throat diameter:  D_throat_new = D_throat_initial + 2×recession × 1.3
        ↓
V_chamber ↑  and  A_throat ↑
        ↓
L* = V_chamber / A_throat  (CHANGES)
        ↓
η_c* = 1 - C × exp(-K × L*)  (DEGRADES)
        ↓
c* ↓ → Pc ↓ → Thrust ↓
```

**Current Status:** ❌ **NOT IMPLEMENTED**
- We calculate `recession_rate` but don't feed it back to geometry
- Time-series simulations use constant L*
- This is a **major physics gap** for ablative engines

**Estimated Impact:** 
- For 50 µm/s recession over 10s burn: ~0.5-2% L* increase
- Corresponds to ~0.3-1.5% thrust degradation
- **Cumulative effect** - gets worse over time

**Solution:** 
- Implement `ablative_geometry.py` (✅ Created)
- Update `runner.evaluate_arrays()` to track geometry evolution
- Recalculate L* at each time step

---

## 4. Why Effects Might Seem Small

### Solver Compensation
The chamber pressure solver finds a new equilibrium:
- Fuel flow ↓ → O/F ↑
- Different O/F might have better/worse c* (depends on propellant curve)
- System "adapts" to new conditions
- Net thrust change is the **difference** between competing effects

### Efficiency Floors
Config has conservative floors:
```yaml
mixture_efficiency_floor: 0.25
cooling_efficiency_floor: 0.25
turbulence_efficiency_floor: 0.3
```

These prevent efficiency from dropping below 25-30%, masking some cooling impact.

### Percentage Basis
- 90 psi regen drop / 974 psi tank = 9% pressure loss
- But fuel flow ∝ √ΔP, so flow only drops ~4-5%
- Thrust is coupled to total mass flow and c*, so final impact is ~2-3%

---

## Complete Coupling Summary

| Cooling Type | Mechanism | Typical Thrust Impact | Currently Modeled? |
|--------------|-----------|----------------------|-------------------|
| **Regen** | Pressure drop → Lower mdot_F | -1% to -5% | ✅ Yes |
| **Regen** | Heat removal → η_cooling | -0.5% to -2% | ✅ Yes |
| **Film** | Mass diversion | -2% to -8% | ✅ Yes |
| **Film** | Heat removal → η_cooling | -0.5% to -2% | ✅ Yes |
| **Ablative** | Blowing effect (minor) | -0.1% to -0.5% | ✅ Yes |
| **Ablative** | **Geometry changes → L* ↑** | **-0.5% to -2%** | ❌ **NO** |

---

## Verification

Run these scripts to see the effects:

```bash
# Show regen/film/ablative impact on thrust
python test_cooling_impact.py

# Show ablative L* evolution (NOT currently coupled)
python test_ablative_Lstar_impact.py
```

---

## Recommendations

1. **✅ Regen & Film:** Already correctly coupled
2. **❌ Ablative Geometry:** **MUST IMPLEMENT** for time-series accuracy
3. **Consider:** Add option to disable efficiency floors for sensitivity studies
4. **Future:** Add throat erosion from combustion products (separate from ablation)

---

## Bottom Line

**Cooling DOES affect thrust through multiple pathways:**
- ✅ Pressure drops (regen)
- ✅ Mass diversion (film)
- ✅ Efficiency factors (all types)
- ❌ **Geometry evolution (ablative) - NOT YET IMPLEMENTED**

The effects are real but can be subtle (~1-10% total) because the solver compensates by finding a new equilibrium. For accurate time-series predictions with ablative liners, **geometry coupling is essential**.

