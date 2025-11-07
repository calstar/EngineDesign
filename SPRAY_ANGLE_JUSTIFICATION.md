# Justification: Why Use tan(θ/2) = k × J^n Instead of θ = k × J^n

## Summary

The `tan(θ/2)` form is **physically more meaningful** and **empirically better validated** than the direct `θ = k × J^n` form for spray angle correlations.

## Physical Justifications

### 1. **Geometric Meaning**
- `tan(θ/2)` represents the **ratio of radial momentum to axial momentum** in a conical spray
- For a spray cone with half-angle `θ/2`, the radial expansion is proportional to `tan(θ/2)`
- This is the natural geometric parameter for conical sprays

### 2. **Natural Bounds**
- As `θ → 90°`, `tan(θ/2) → ∞`, providing **natural limiting behavior**
- The direct form `θ = k × J^n` can produce unphysical angles > 90° without bounds
- The tan form inherently prevents angles > 90° (though we still clamp for safety)

### 3. **Nonlinear Behavior**
- The `tan(θ/2)` form captures the **nonlinear relationship** between J and θ
- At low J: small changes in J → small angle changes
- At high J: same change in J → larger angle changes
- This matches experimental observations of spray behavior

### 4. **Momentum Balance**
- The half-angle form arises naturally from **momentum balance** in impinging jet sprays
- The radial momentum component relates to `tan(θ/2)`
- This is physically consistent with the momentum flux ratio J

### 5. **Empirical Validation**
- Many spray correlations in literature use `tan(θ/2) = f(J)`
- This form aligns better with experimental data across different:
  - Nozzle geometries
  - Operating conditions
  - Fluid properties

## Comparison

**Direct Form (θ = k × J^n):**
- ✅ Simpler mathematically
- ❌ Can produce unphysical angles > 90°
- ❌ Doesn't capture nonlinear behavior as well
- ❌ Less physically meaningful

**Tan(θ/2) Form:**
- ✅ Physically meaningful (geometric interpretation)
- ✅ Natural bounds (tan → ∞ as θ → 90°)
- ✅ Better captures nonlinear behavior
- ✅ Empirically validated
- ✅ Arises from momentum balance

## Conclusion

The `tan(θ/2) = k × J^n` form is the **correct physical model** for spray angle correlations. While the direct form is simpler, it lacks the physical justification and empirical support of the tan form.

