# Surface Temperature T_s Calculation Review

## Overview
The surface temperature T_s is calculated iteratively using a Newton-Raphson method to solve the energy balance equation:
```
q''_in + q''_fb - q''_rad = q''_cond + m''_th * H*_th
```

## Issues Identified

### 1. **Energy Balance Residual When Ablating** (Lines 469-474)
**Issue**: When `is_ablating` is True, the residual is set to 0.0, but this happens AFTER the feedback loop. The residual should be calculated based on the actual energy balance, even when ablating.

**Current Code**:
```python
if is_ablating:
    residual = 0.0
else:
    residual = q_in + q_fb - q_rad - q_cond - m_dot_th * H_star_th
```

**Recommendation**: The residual should still be calculated when ablating to verify the energy balance is satisfied. When T_s is pinned, m_dot_th should balance the energy, so the residual should be near zero, but it should be calculated explicitly.

### 2. **Newton-Raphson Derivative Approximations** (Lines 484-496)
**Issue**: The derivatives for `dq_fb_dT` and `dm_dot_th_dT` use arbitrary 0.1 factors that are not physically justified.

**Current Code**:
```python
dm_dot_ox_dT = m_dot_ox * (Ea / (R_GAS * T_s**2)) * 0.1  # Small factor for stability
dm_dot_th_dT = m_dot_th * cp_s * 0.1  # Small factor for stability
```

**Recommendation**: 
- For `dm_dot_ox_dT`: The full derivative should be calculated. The kinetic-limited rate has an explicit temperature dependence: `d(m_dot_ox_kin)/dT_s = m_dot_ox_kin * (Ea / (R_GAS * T_s^2))`. The 0.1 factor should be removed or justified.
- For `dm_dot_th_dT`: In the transition region (T_s > 2800K but < T_abl), `m_dot_th` depends on T_s through the energy balance. The derivative should account for this properly.

### 3. **Transition Region Handling** (Lines 452-458)
**Issue**: When `T_s > 2800.0` but `T_s < T_abl`, thermal ablation is allowed while still iterating on T_s. This can cause oscillations near the ablation threshold.

**Current Code**:
```python
elif T_s > 2800.0 and q_net_available > 0:
    # Transition region: allow some thermal ablation but still iterate on T_s
    delta_T = max(T_s - 300.0, 0.0)
    H_star_th = graphite_config.heat_of_ablation + cp_s * delta_T
    H_star_th = max(H_star_th, 1e6)
    m_dot_th = q_net_available / H_star_th
    m_dot_th = max(m_dot_th, 0.0)
```

**Recommendation**: Consider smoothing the transition or using a more gradual approach to avoid discontinuities.

### 4. **Convergence Check** (Line 517)
**Issue**: Convergence is only checked on T_s change, not on the energy balance residual. This can lead to false convergence if the residual is non-zero.

**Current Code**:
```python
if abs(T_s - T_s_old) < tol:
    break
```

**Recommendation**: Add a residual check:
```python
residual_tol = 100.0  # W/m² - reasonable tolerance for energy balance
if abs(T_s - T_s_old) < tol and abs(residual) < residual_tol:
    break
```

### 5. **T_s Pinning Logic** (Lines 394-440)
**Issue**: When T_s is pinned at T_abl, oxidation is recalculated, but this happens inside the feedback loop. The Newton update is correctly skipped, but the residual calculation may not account for the pinned state properly.

**Current Code**: T_s is set to T_abl when `q_net_available > 0 and T_s >= T_abl`, but this check happens inside the feedback loop.

**Recommendation**: Consider checking for ablation condition before entering the feedback loop, or ensure the residual is properly calculated when T_s is pinned.

### 6. **Feedback Loop Coupling**
**Issue**: The feedback loop (lines 367-467) iterates on f_fb and m_dot_th, but when T_s changes, oxidation changes, which affects f_fb. The nested iteration is correct, but convergence may be slow.

**Recommendation**: Consider adding convergence diagnostics or adaptive iteration limits.

## Summary of Recommendations

1. **Calculate residual explicitly even when ablating** to verify energy balance
2. **Remove or justify the 0.1 factors** in derivative calculations
3. **Add residual check to convergence criteria** to ensure energy balance is satisfied
4. **Smooth transition region** to avoid oscillations
5. **Improve T_s pinning logic** to ensure consistency with energy balance
6. **Add convergence diagnostics** to track iteration progress

## Energy Balance Equation
The correct energy balance (from theory) is:
```
q''_in + q''_fb - q''_rad = q''_cond + m''_th * H*_th
```

Where:
- `q''_in = h_g * (T_g - T_s)` - Convective heat flux
- `q''_fb = f_fb * m''_ox * |Δh_ox|` - Oxidation feedback heat flux
- `q''_rad = ε * σ * (T_s^4 - T_env^4)` - Radiative cooling
- `q''_cond = k_s * (T_s - T_back) / t_eff` - Conduction into solid
- `m''_th * H*_th` - Thermal ablation energy sink

The residual should be:
```
residual = q_in + q_fb - q_rad - q_cond - m_dot_th * H_star_th
```

This should be zero (or very close to zero) when converged.

