# Canopy Inflation Issue Summary

## Problem
- Canopies are not inflating (area = 0) even after pickup events
- System velocities remain high (~1560 m/s) instead of slowing under parachute
- Descent from 10,000 ft should take **minutes** (60+ seconds under drogue), not seconds
- Numerical instability causes system to "fall through ground" (altitude goes to -6 million meters)

## Root Cause Analysis

1. **Pickup Events Are Detected**: `leg_drogue` pickup at t=0.100709s ✓
2. **Pickup Times Are Recorded**: `pickup_times['drogue']` should be set ✓
3. **But Canopies Still Don't Inflate**: Area remains 0.0 even after pickup

## Key Issues

### Issue 1: Edge-to-Canopy Mapping
- `leg_drogue` has `drogue` on `n_minus` (not `n_plus`)
- Code checks `n_plus_id` first, then `n_minus_id` - should work ✓
- But need to verify `edge_to_canopy['leg_drogue'] == 'drogue'`

### Issue 2: Timing of Pickup Time Updates
- Pickup event detected AFTER RK4 integration step
- `pickup_times` dict updated during event check
- Closure should capture dict by reference, so updates visible on next step
- But area still 0.0 - suggests dict not being passed or checked correctly

### Issue 3: Altitude Check
- Drogue deployment altitude: 3048.0 m
- At t=0.11s, canopy altitude: 3042.7 m
- Check: `altitude_agl > canopy.altitude_deploy` → `3042.7 > 3048.0` → False ✓
- Should allow inflation, but doesn't

### Issue 4: Pickup Time Check
- Pickup at t=0.100709s
- At t=0.11s: `t >= t_pickup` → `0.11 >= 0.100709` → True ✓
- Should allow inflation, but doesn't

## Expected Behavior

Under proper parachute deployment:
- **Drogue descent**: ~50 m/s, takes ~60 seconds from 10,000 ft to 5,000 ft
- **Main descent**: ~5 m/s, takes ~3 minutes from 1,200 ft to ground
- **Total descent time**: ~4-5 minutes from drogue deployment to impact

Current behavior:
- Velocities: 1559 m/s (way too fast - should be ~50 m/s)
- "Impact" in 5 seconds (numerical instability, not real impact)
- No drag forces (canopies not inflating)

## Next Steps

1. Verify `edge_to_canopy` mapping is correct for all edges
2. Add debug prints to verify `pickup_times` dict is updated and passed correctly
3. Check if `canopy_area()` function is being called with correct parameters
4. Verify closure captures `pickup_times` dict reference correctly
5. Fix numerical instability that causes altitude to go negative

## Files Modified
- `parachute/engine.py`: Updated `edge_to_canopy` mapping to check both ends
- `parachute/physics.py`: Altitude-based deployment and pickup checks
- `examples/realistic_rocket_deployment.yaml`: Deployment configuration

