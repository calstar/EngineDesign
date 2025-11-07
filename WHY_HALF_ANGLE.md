# Why Half-Angle (θ/2) Instead of Full Angle (θ)?

## The Key Insight: We Measure from the Axis

In a **conical spray**, the spray is **symmetric** about the central axis. This means we only need to consider **one side** of the spray.

### Visual Picture

```
        │  ← Central axis (axis of symmetry)
        │
        │  θ/2  ← Half-angle (measured from axis to edge)
        │  ╱
        │ ╱
        │╱  ← Edge of spray cone
       ╱│
      ╱ │
     ╱  │
    ╱   │
   ╱    │
  ╱     │
 ╱      │
╱       │
────────┼────────  ← Full angle θ (from one edge to the other)
```

### Why Half-Angle is Natural

**1. Momentum Vectors are Measured from the Axis**

When a droplet/particle travels at the edge of the spray cone:
- Its **momentum vector** makes an angle **θ/2** with the axis (not θ!)
- This is because we measure angles **from the axis**, not from the opposite edge

**2. Vector Decomposition Uses Half-Angle**

When we decompose the momentum vector of a particle at the edge:
- **Axial component**: `p_axial = p_total × cos(θ/2)` ← uses θ/2
- **Radial component**: `p_radial = p_total × sin(θ/2)` ← uses θ/2

The ratio is:
```
p_radial / p_axial = sin(θ/2) / cos(θ/2) = tan(θ/2)
```

**3. Symmetry Means We Only Need One Side**

Since the spray is symmetric:
- We don't need to consider both sides separately
- The half-angle θ/2 fully describes the geometry
- Full angle θ = 2 × (θ/2) is just a convention

### Concrete Example

Imagine a spray with **θ = 60°** (full angle):
- **Half-angle**: θ/2 = 30°
- A particle at the edge travels at **30° from the axis** (not 60°!)
- Its momentum vector makes **30° with the axis**
- When we decompose: `p_radial = p × sin(30°)`, `p_axial = p × cos(30°)`
- Ratio: `tan(30°) = 0.577`

If we incorrectly used full angle θ = 60°:
- We'd get `tan(60°) = 1.732` ← **WRONG!**
- This would be the ratio if the particle traveled at 60° from the axis
- But particles at the edge only travel at 30° from the axis!

### Why Not Full Angle?

The full angle θ is **not physically meaningful** for momentum decomposition because:
- Momentum vectors are measured **from the axis**
- Particles at the edge make angle **θ/2** with the axis, not θ
- Using θ would give us the wrong momentum components

### Analogy: Projectile Motion

Think of throwing a ball:
- If you throw it at **30° from horizontal**, the angle is **30°** (not 60°)
- The vertical component is `v × sin(30°)`
- The horizontal component is `v × cos(30°)`
- The ratio is `tan(30°)`

Similarly, in a spray:
- Particles at the edge travel at **θ/2 from the axis**
- The radial component is `p × sin(θ/2)`
- The axial component is `p × cos(θ/2)`
- The ratio is `tan(θ/2)`

### Conclusion

**Half-angle θ/2 is natural because:**
1. We measure angles **from the axis** (not from the opposite edge)
2. Momentum vectors of particles at the edge make angle **θ/2** with the axis
3. Vector decomposition uses **θ/2** (not θ)
4. The spray is symmetric, so we only need to consider one side

The full angle θ is just a convention (θ = 2 × θ/2), but it's **not the angle we use in physics calculations** because we always measure from the axis.

