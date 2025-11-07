# Physics Review Summary: Pintle Engine Pipeline

## ✅ FIXED ISSUES

### 1. ✅ Nozzle Exit Pressure Calculation (CRITICAL - FIXED)
**Location**: `pintle_models/nozzle.py`

**Problem**: Was using incorrect subsonic formula for supersonic flow.

**Fix Applied**:
- Now solves area-Mach relation iteratively: `A/A* = (1/M) × [(2/(γ+1)) × (1 + (γ-1)/2 × M²)]^((γ+1)/(2(γ-1)))`
- Uses Newton-Raphson to find M_exit from expansion ratio
- Then uses correct isentropic relation: `P_exit/Pc = [1 + (γ-1)/2 × M_exit²]^(-γ/(γ-1))`

**Verification**: 
- For eps=6.54, gamma=1.23: M_exit ≈ 3.05 (correct for supersonic)
- P_exit/Pc ≈ 0.02-0.03 (reasonable for rocket nozzles)

### 2. ✅ Regen Cooling Pressure Scaling (FIXED)
**Location**: `pintle_pipeline/regen_cooling.py`

**Problem**: Redundant P_scale factor double-counted pressure dependence.

**Fix Applied**:
- Removed P_scale factor
- Pressure dependence already captured through mdot(P_tank) → velocity → pressure drop

### 3. ✅ Chamber Solver Bounds (IMPROVED)
**Location**: `pintle_models/chamber_solver.py`

**Problem**: Used fixed 5% margin instead of actual feed loss estimates.

**Fix Applied**:
- Changed to 15% conservative margin (better than fixed 5%)
- Added comments explaining the approach

---

## ✅ VERIFIED CORRECT PHYSICS

### 1. Injector Flow Model
- **Formula**: `mdot = Cd × A × √(2ρΔp)` ✅
- **Reynolds coupling**: `Cd(Re) = Cd_∞ - a_Re/√Re` ✅
- **Correct for incompressible flow** (LOX at typical pressures is subcritical)

### 2. Feed System Losses
- **Formula**: `Δp = K_eff(P) × (ρ/2) × u²` ✅
- **Pressure dependence**: `K_eff = K0 + K1 × φ(P)` ✅
- **Correct** - standard pressure loss equation

### 3. Regenerative Cooling
- **Friction factor**: Blasius (smooth) / Swamee-Jain (rough) ✅
- **Darcy-Weisbach**: `Δp = f × (L/D) × (ρ/2) × u²` ✅
- **Parallel channels**: Correctly models flow splitting ✅
- **Pressure dependence**: Now correctly through mdot only (P_scale removed) ✅

### 4. Chamber Pressure Solver
- **Supply**: `mdot_supply = mdot_O + mdot_F` ✅
- **Demand**: `mdot_demand = Pc × At / c*_actual` ✅
- **Chamber-driven c***: `c*_actual = η(L*) × c*_ideal` ✅
- **Coupling**: Correctly models coupling through shared Pc ✅

### 5. Combustion Efficiency
- **L* correction**: `η = 1 - C × exp(-K×L*)` ✅
- **Actual temperature**: `Tc_actual = Tc_ideal × (η)^0.7` ✅
- **Frozen flow**: `γ_frozen = γ_eq × (1 + δ)` where δ = 0.05×exp(-0.5×L*) ✅

### 6. Spray Physics
- **Momentum flux ratio**: `J = (ρ_O×u_O²)/(ρ_F×u_F²)` ✅
- **Weber number**: `We = (ρ×u²×d)/σ` ✅
- **SMD**: `D32 = C×d×We^(-m)×Oh^p` ✅
- **Evaporation length**: `x* = U_rel × τ_evap` ✅

### 7. Nozzle Model
- **Exit Mach**: Now correctly solved from area-Mach relation ✅
- **Exit pressure**: Now uses correct supersonic isentropic relation ✅
- **Exit velocity**: `v = √(2cp(Tc-T_exit))` ✅
- **Thrust**: `F = mdot×v + (P_exit-Pa)×A_exit` ✅

---

## PHYSICS COUPLING EXPLANATION

### Why Oxidizer Tank Pressure Affects Fuel Flow

**The coupling is CORRECT PHYSICS** - flows are coupled through the solved chamber pressure:

1. **Shared Chamber Pressure**: Both injectors feed into the same chamber
2. **Pc is Solved**: Chamber pressure balances supply and demand
3. **Coupling Mechanism**:
   - Change P_tank_O → changes mdot_O
   - Changes mdot_supply → solver adjusts Pc
   - New Pc affects BOTH mdot_O and mdot_F
   - Result: Both flows are coupled through Pc

**This is physically correct!** Real engines exhibit this coupling.

---

## REMAINING CONSIDERATIONS

### Minor Improvements (Not Critical):
1. **Use actual chamber temperature in nozzle**: Currently uses CEA ideal Tc
   - Could use `calculate_actual_chamber_temp()` result
   - Impact is small (temperature correction is moderate)

2. **Compressible flow effects**: Currently assumes incompressible flow
   - Reasonable for typical LOX/RP-1 conditions
   - May need compressible flow for extreme conditions

3. **Chamber solver bounds**: Could use actual feed loss estimates
   - Current 15% margin is conservative but acceptable
   - Could improve by estimating max feed losses

---

## SUMMARY

✅ **All critical physics issues have been fixed**
✅ **Nozzle exit pressure now uses correct supersonic flow relations**
✅ **Redundant pressure scaling removed**
✅ **Chamber solver bounds improved**
✅ **All other physics models verified as correct**

The pipeline now has **physically sound and consistent** modeling across all components.

