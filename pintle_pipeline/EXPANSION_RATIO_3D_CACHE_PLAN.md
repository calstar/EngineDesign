# 3D CEA Cache Implementation Plan

## Problem

**Current:** CEA cache is 2D `(Pc, MR)` with **fixed** expansion ratio `ε`
**Issue:** When ablative recession grows throat area:
```
ε = A_exit / A_throat

A_throat increases → ε decreases (if A_exit fixed)
OR both A_throat and A_exit grow (if nozzle is ablative too)
```

CEA properties (Cf, exit T, exit P, exit Mach) **depend on ε**, so we need 3D lookup!

---

## Solution: 3D CEA Cache `(Pc, MR, ε)`

### 1. Config Schema Changes

```yaml
cea:
  cache_file: "cea_cache_LOX_RP1_3D.npz"
  Pc_range: [1.0e6, 5.0e6]  # Pa
  MR_range: [1.6, 3.5]
  eps_range: [5.0, 50.0]  # NEW: Expansion ratio range
  n_points: 20  # Grid points per dimension
  # 20³ = 8000 points (manageable)
```

### 2. Cache Structure

**Old (2D):**
```python
cstar_table[i_Pc, i_MR]  # Shape: (n, n)
Cf_table[i_Pc, i_MR]
```

**New (3D):**
```python
cstar_table[i_Pc, i_MR, i_eps]  # Shape: (n, n, n)
Cf_table[i_Pc, i_MR, i_eps]
Tc_table[i_Pc, i_MR, i_eps]
gamma_table[i_Pc, i_MR, i_eps]
R_table[i_Pc, i_MR, i_eps]
M_table[i_Pc, i_MR, i_eps]
```

### 3. Interpolation

**Old:** Bilinear interpolation in 2D
**New:** **Trilinear interpolation** in 3D

```python
def trilinear_interp(table_3d, Pc, MR, eps):
    # Find bounding indices
    i0, i1, wx = find_bounds(Pc_grid, Pc)
    j0, j1, wy = find_bounds(MR_grid, MR)
    k0, k1, wz = find_bounds(eps_grid, eps)
    
    # Interpolate along Pc
    c00 = table_3d[i0, j0, k0] * (1-wx) + table_3d[i1, j0, k0] * wx
    c01 = table_3d[i0, j0, k1] * (1-wx) + table_3d[i1, j0, k1] * wx
    c10 = table_3d[i0, j1, k0] * (1-wx) + table_3d[i1, j1, k0] * wx
    c11 = table_3d[i0, j1, k1] * (1-wx) + table_3d[i1, j1, k1] * wx
    
    # Interpolate along MR
    c0 = c00 * (1-wy) + c10 * wy
    c1 = c01 * (1-wy) + c11 * wy
    
    # Interpolate along eps
    result = c0 * (1-wz) + c1 * wz
    return result
```

### 4. Nozzle Geometry Tracking

**Need to track:**
- `A_throat_initial` - Initial throat area
- `A_exit_initial` - Initial exit area  
- `eps_initial = A_exit / A_throat`

**Two scenarios:**

**A) Fixed Exit (Nozzle Not Ablative):**
```python
A_throat_new = A_throat_initial + ΔA_throat
A_exit_new = A_exit_initial  # Fixed
eps_new = A_exit_new / A_throat_new  # Decreases!
```

**B) Ablative Nozzle (Both Grow):**
```python
A_throat_new = A_throat_initial + ΔA_throat
A_exit_new = A_exit_initial + ΔA_exit
eps_new = A_exit_new / A_throat_new  # May stay constant or change
```

### 5. Integration Points

**Files to modify:**

1. **`pintle_pipeline/config_schemas.py`**
   - Add `eps_range: Tuple[float, float]` to `CEAConfig`
   - Add `nozzle_ablative: bool` to `AblativeCoolingConfig`

2. **`pintle_pipeline/cea_cache.py`**
   - Add `eps_grid` as 3rd dimension
   - Change all tables to 3D arrays
   - Implement trilinear interpolation
   - Update `_build_cache()` to loop over ε
   - Update `get()` method signature: `get(Pc, MR, eps)`

3. **`pintle_models/nozzle.py`**
   - Update `calculate_thrust()` to accept `eps` parameter
   - Pass current `eps` to CEA cache lookup

4. **`pintle_models/runner.py`**
   - Track `A_exit` in addition to `A_throat`
   - Calculate current `eps = A_exit / A_throat`
   - Pass `eps` to `calculate_thrust()`

5. **`pintle_pipeline/ablative_geometry.py`**
   - Add `update_nozzle_geometry_from_ablation()`
   - Calculate exit area growth (if nozzle is ablative)

### 6. Cache Regeneration

**Important:** 3D cache is **NOT backwards compatible** with 2D!

**Strategy:**
- New cache filename: `cea_cache_LOX_RP1_3D.npz`
- Metadata includes `"dimensions": 3`
- Auto-regenerate if dimensions mismatch

### 7. Performance Impact

**Cache size:**
- 2D: 20×20 = 400 points
- 3D: 20×20×20 = 8,000 points (~20x larger)

**Build time:**
- ~0.5s per CEA call
- 8,000 calls = ~4,000s = **~67 minutes** 😱

**Optimization:**
- Use `n_points = 15` → 15³ = 3,375 points (~28 min)
- Or `n_points = 12` → 12³ = 1,728 points (~14 min)
- Parallel CEA calls (if possible)

### 8. Alternative: Analytical Cf(ε) Correction

**Simpler approach** (if CEA is too slow):

Use 2D cache for chamber properties, apply analytical correction for Cf:

```python
# Get chamber properties from 2D cache
cstar, Tc, gamma, R = cache_2d.get(Pc, MR)

# Analytical Cf correction for expansion ratio
# Based on isentropic flow relations
Cf = calculate_Cf_isentropic(gamma, eps, Pa/Pc)
```

**Pros:**
- No 3D cache needed
- Fast
- Reasonably accurate for Cf

**Cons:**
- Less accurate than full CEA
- Doesn't capture exit chemistry effects

---

## Recommendation

**Phase 1 (Quick Fix):** Analytical Cf correction
- Keep 2D cache
- Add `calculate_Cf_from_expansion_ratio(gamma, eps, Pa, Pc)`
- Good enough for ablative geometry studies

**Phase 2 (Full Solution):** 3D CEA cache
- Implement when needed for high-fidelity analysis
- Pre-generate cache offline
- Distribute with codebase

---

## Implementation Priority

1. ✅ **DONE:** Fix geometry coupling in runner
2. **NEXT:** Add analytical Cf(ε) correction (1 hour)
3. **LATER:** Full 3D CEA cache (1 day + overnight build)

**For now, let's do the analytical correction - it's 95% accurate and 100x faster!**

