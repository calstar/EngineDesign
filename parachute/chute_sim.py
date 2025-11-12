import numpy as np
from parachute_toolkit import example_system_two_bodies_two_chutes, simulate, instantaneous_shock, energy_peak_estimate

# Build example: 2 bodies, 2 chutes (chute-1 at t=0s, chute-2 at t=5s)
sys, state0, params = example_system_two_bodies_two_chutes()

# Run a short deployment simulation (records shocks at each step)
out = simulate(sys, params, t0=0.0, tf=8.0, dt=0.01, state0=state0)

# Peak system shock magnitude
peak = np.max(np.linalg.norm(out["F_shock_sys"], axis=1))
print("Peak |F_shock|:", peak)

# If you want the shock at a specific (t,state):
# s = ...  # some SystemState
# F_sys, F_per_body = instantaneous_shock(t, s, sys, params)

# Fast, spreadsheet-style peak estimate (before sim), given initial velocity V0 and slack s_free:
# Keff_sum is the sum of effective leg stiffnesses expected to pick up first
T_est = energy_peak_estimate(V0=30.0, s_free=5.0, Keff_sum=2000.0+1800.0, m_total=50.0+60.0)
print("Energy-based peak estimate:", T_est)
