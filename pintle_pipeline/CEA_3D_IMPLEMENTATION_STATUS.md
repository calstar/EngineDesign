# 3D CEA Cache Implementation Status

## Overview
Converting CEA cache from 2D `(Pc, MR)` to 3D `(Pc, MR, ε)` to handle time-varying expansion ratio during ablative recession.

## Progress

### ✅ COMPLETED
1. Config schema updated with `eps_range` and `nozzle_ablative` flag
2. Config file updated with 3D cache parameters (`n_points=15`, `eps_range=[4.0, 15.0]`)
3. Backup of 2D cache created (`cea_cache_2d_backup.py`)

### 🔄 IN PROGRESS
4. Refactoring `CEACache` class for 3D support

### ⏳ PENDING
5. Trilinear interpolation implementation
6. Update `nozzle.py` to accept `eps` parameter
7. Update `runner.py` to track `A_exit` and calculate current `eps`
8. Add nozzle geometry evolution to `ablative_geometry.py`
9. Test and validate

## Implementation Notes

### Key Changes Needed in `cea_cache.py`

**1. Add 3D mode detection:**
```python
self.use_3d = config.eps_range is not None
if self.use_3d:
    self.eps_min, self.eps_max = config.eps_range
    self.eps_grid = np.linspace(self.eps_min, self.eps_max, self.n_points)
```

**2. Tables become 3D:**
```python
# 2D: shape (n, n)
# 3D: shape (n, n, n)
self.cstar_table = np.zeros((n, n, n)) if use_3d else np.zeros((n, n))
```

**3. Build cache loops over 3 dimensions:**
```python
for i, Pc in enumerate(Pc_grid):
    for j, MR in enumerate(MR_grid):
        if use_3d:
            for k, eps in enumerate(eps_grid):
                # CEA call with eps
        else:
            # CEA call with fixed eps
```

**4. Interpolation method:**
```python
def get(self, Pc, MR, eps=None):
    if self.use_3d:
        return self._trilinear_interp(Pc, MR, eps)
    else:
        return self._bilinear_interp(Pc, MR)
```

### Estimated Build Time
- **2D cache:** 20×20 = 400 points × 0.5s = **3.3 minutes**
- **3D cache:** 15×15×15 = 3,375 points × 0.5s = **28 minutes** ⏱️

### Backwards Compatibility
- Old 2D caches will be detected and regenerated automatically
- Metadata includes `"dimensions": 2 or 3`
- Cache filename changed to `cea_cache_LOX_RP1_3D.npz`

## Next Steps
1. Implement 3D grid initialization
2. Implement 3D cache building
3. Implement trilinear interpolation
4. Update metadata validation
5. Test with small grid first (n=5) before full build

## Testing Strategy
1. **Unit test:** Small 3D cache (5×5×5 = 125 points, ~1 min)
2. **Validation:** Compare 3D cache at fixed ε vs 2D cache
3. **Integration:** Run ablative simulation with varying ε
4. **Performance:** Full cache build (15×15×15, ~28 min)

---

**Status:** Ready to implement CEACache 3D refactor
**ETA:** ~2 hours coding + 28 min cache build

