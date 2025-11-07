# Physics Intuition: Why tan(θ/2) = k × J^n

## The Core Physics: Momentum Balance in a Conical Spray

### Visual Picture
Imagine a conical spray coming out of the injector:
- The **fuel** (RP-1) flows **radially outward** (perpendicular to axis)
- The **oxidizer** (LOX) flows **axially** (along the axis) with some angle
- They **collide and mix**, creating a combined spray cone

### The Key Insight: Half-Angle is the Natural Parameter

For a **conical spray**, the half-angle `θ/2` is what matters physically, not the full angle `θ`.

**Why?** Because:
- The spray is **symmetric** about the axis
- The **radial momentum** (perpendicular to axis) determines how far the spray spreads
- The **axial momentum** (along the axis) determines how fast it goes forward
- The **ratio** of these determines the spray angle

### Momentum Balance Derivation

Consider the momentum vectors in the spray:

1. **Radial momentum component**: `p_radial = p_total × sin(θ/2)`
2. **Axial momentum component**: `p_axial = p_total × cos(θ/2)`

The **ratio** of radial to axial momentum is:
```
p_radial / p_axial = sin(θ/2) / cos(θ/2) = tan(θ/2)
```

**This is why tan(θ/2) appears naturally!**

### Connection to Momentum Flux Ratio J

The momentum flux ratio `J = (ρ_O u_O²) / (ρ_F u_F²)` tells us:
- How much **radial momentum** (from fuel) vs **axial momentum** (from oxidizer)
- When J is high → more oxidizer momentum → spray is more axial (smaller angle)
- When J is low → more fuel momentum → spray is more radial (larger angle)

The **spray angle** emerges from balancing these momentum components:
- High radial momentum → large `tan(θ/2)` → large angle
- High axial momentum → small `tan(θ/2)` → small angle

### Why Not Direct θ = k × J^n?

The direct form `θ = k × J^n` treats the angle as a **scalar quantity** without geometric meaning.

But the spray angle is **not arbitrary** - it's determined by the **vector balance** of radial and axial momenta. The `tan(θ/2)` form captures this vector relationship naturally.

### Physical Analogy

Think of it like a **projectile**:
- If you throw a ball straight up (pure axial), `θ = 0°` → `tan(θ/2) = 0`
- If you throw it sideways (pure radial), `θ = 90°` → `tan(θ/2) = 1`
- The **ratio** of horizontal to vertical velocity determines the angle

In our spray:
- The **ratio** of radial to axial momentum determines `tan(θ/2)`
- This ratio is related to the momentum flux ratio `J`
- Hence: `tan(θ/2) = k × J^n`

### The Nonlinear Behavior

Why is it nonlinear (the `J^n` term)?

Because momentum balance is **not linear**:
- At low J: Small changes in momentum ratio → small angle changes
- At high J: Same change in momentum ratio → larger angle changes (because tan grows faster)

This matches what we see: spray angles are **sensitive** to momentum ratio, but the sensitivity increases as the angle gets larger.

### Conclusion

`tan(θ/2) = k × J^n` is not just a convenient mathematical form - it's the **natural result** of momentum balance in a conical spray. The half-angle appears because the spray is symmetric, and the tangent appears because it's the ratio of radial to axial momentum components.

