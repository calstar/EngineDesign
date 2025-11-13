# -*- coding: utf-8 -*-

"""

flight_sim.py

--------------

Reusable RocketPy-based liquid engine flight simulation module.

"""



import numpy as np

import math

import matplotlib.pyplot as plt

from rocketpy.motors import LiquidMotor

from rocketcea.cea_obj_w_units import CEA_Obj

from rocketpy import Environment, Rocket, Flight, Function, Fluid

from rocketpy.motors import LiquidMotor, CylindricalTank

from rocketpy.motors.tank import MassBasedTank, MassFlowRateBasedTank

from pathlib import Path



g0 = 9.80665



def detect_tank_underfill_time(mdot, m_initial, burn_time, n_samples=1000):

    """

    Detect when a tank would get underfilled by integrating mdot over time.

    

    Parameters:

    -----------

    mdot : float or Function

        Mass flow rate. Can be a constant float or a RocketPy Function.

    m_initial : float

        Initial tank mass [kg]

    burn_time : float

        Total burn time [s]

    n_samples : int

        Number of time samples for integration (default: 1000)

    

    Returns:

    --------

    cutoff_time : float or None

        Time at which tank would be depleted (None if it never depletes)

    """

    # Create time array for sampling

    times = np.linspace(0, burn_time, n_samples)

    dt = burn_time / (n_samples - 1) if n_samples > 1 else burn_time

    

    # Sample mdot values

    if isinstance(mdot, Function):

        # It's a RocketPy Function - evaluate at each time

        mdot_values = np.array([mdot(t) for t in times])

    else:

        # It's a constant float

        mdot_values = np.full_like(times, float(mdot))

    

    # Integrate mdot to get cumulative mass consumed

    # Use trapezoidal integration

    cumulative_mass = np.zeros_like(times)

    for i in range(1, len(times)):

        # Trapezoidal integration:  mdot dt  (mdot[i-1] + mdot[i]) * dt / 2

        cumulative_mass[i] = cumulative_mass[i-1] + (mdot_values[i-1] + mdot_values[i]) * dt / 2.0

    

    # Find where cumulative mass exceeds initial tank mass

    # Find the first index where cumulative_mass >= m_initial

    depletion_idx = np.where(cumulative_mass >= m_initial)[0]

    

    if len(depletion_idx) > 0:

        # Tank would be depleted at this time

        cutoff_time = times[depletion_idx[0]]

        return float(cutoff_time)

    else:

        # Tank never depletes during the burn

        return None



def detect_lox_underfill_time(mdot_lox, m_lox0, burn_time, n_samples=1000):

    """

    Detect when LOX tank would get underfilled by integrating mdot_lox over time.

    

    Parameters:

    -----------

    mdot_lox : float or Function

        LOX mass flow rate. Can be a constant float or a RocketPy Function.

    m_lox0 : float

        Initial LOX mass [kg]

    burn_time : float

        Total burn time [s]

    n_samples : int

        Number of time samples for integration (default: 1000)

    

    Returns:

    --------

    cutoff_time : float or None

        Time at which LOX would be depleted (None if it never depletes)

    """

    return detect_tank_underfill_time(mdot_lox, m_lox0, burn_time, n_samples)



def detect_fuel_underfill_time(mdot_fuel, m_fuel0, burn_time, n_samples=1000):

    """

    Detect when fuel tank would get underfilled by integrating mdot_fuel over time.

    

    Parameters:

    -----------

    mdot_fuel : float or Function

        Fuel mass flow rate. Can be a constant float or a RocketPy Function.

    m_fuel0 : float

        Initial fuel mass [kg]

    burn_time : float

        Total burn time [s]

    n_samples : int

        Number of time samples for integration (default: 1000)

    

    Returns:

    --------

    cutoff_time : float or None

        Time at which fuel would be depleted (None if it never depletes)

    """

    return detect_tank_underfill_time(mdot_fuel, m_fuel0, burn_time, n_samples)



def truncate_thrust_curve(thrust_curve, cutoff_time):

    """

    Truncate thrust curve at cutoff_time, setting thrust to 0 after that point.

    

    Parameters:

    -----------

    thrust_curve : list of (t, F) tuples or Function

        Original thrust curve

    cutoff_time : float

        Time at which to cut off thrust

    

    Returns:

    --------

    truncated_curve : list of (t, F) tuples

        Thrust curve with thrust=0 after cutoff_time

    """

    if isinstance(thrust_curve, Function):

        # Convert Function to list of tuples by sampling

        # Sample up to cutoff_time, then add a point at cutoff_time with 0 thrust

        times = np.linspace(0, cutoff_time, 100)

        curve = [(float(t), float(thrust_curve(t))) for t in times]

        # Add cutoff point with 0 thrust

        curve.append((cutoff_time, 0.0))

        return curve

    elif isinstance(thrust_curve, list):

        # It's already a list of (t, F) tuples

        truncated = []

        for t, F in thrust_curve:

            if t < cutoff_time:

                truncated.append((t, F))

            elif t == cutoff_time:

                # If we hit cutoff_time exactly, use that value then add 0

                truncated.append((t, F))

                break

            else:

                # We've passed cutoff_time - interpolate and add cutoff point

                if len(truncated) > 0:

                    prev_t, prev_F = truncated[-1]

                    # Linear interpolation to cutoff_time

                    if t > prev_t:

                        F_cutoff = prev_F + (F - prev_F) * (cutoff_time - prev_t) / (t - prev_t)

                    else:

                        F_cutoff = prev_F

                    truncated.append((cutoff_time, F_cutoff))

                else:

                    # No previous points, just add cutoff with 0

                    truncated.append((cutoff_time, 0.0))

                break

        # Ensure we end with 0 thrust at cutoff_time

        if len(truncated) == 0:

            truncated.append((cutoff_time, 0.0))

        elif truncated[-1][0] < cutoff_time:

            # Add cutoff point if we haven't reached it yet

            if len(truncated) > 0:

                prev_t, prev_F = truncated[-1]

                truncated.append((cutoff_time, prev_F))

            truncated.append((cutoff_time, 0.0))

        elif truncated[-1][0] == cutoff_time and truncated[-1][1] != 0.0:

            # We're at cutoff_time but thrust isn't 0, add a 0 point

            truncated.append((cutoff_time, 0.0))

        return truncated

    else:

        raise TypeError(f"Unsupported thrust_curve type: {type(thrust_curve)}")



def truncate_mdot_function(mdot_func, cutoff_time, burn_time):

    """

    Create a new mdot function that is 0 after cutoff_time.

    

    Parameters:

    -----------

    mdot_func : float or Function

        Original mass flow rate

    cutoff_time : float

        Time at which to cut off mass flow

    burn_time : float

        Total burn time (for creating the function domain)

    

    Returns:

    --------

    truncated_func : Function

        Function that returns mdot_func(t) for t <= cutoff_time, 0 otherwise

    """

    if isinstance(mdot_func, Function):

        # Create a piecewise function

        def truncated_mdot(t):

            if t <= cutoff_time:

                return mdot_func(t)

            else:

                return 0.0

        # Convert to RocketPy Function by sampling

        times = np.linspace(0, burn_time, int(burn_time * 100) + 1)

        values = np.array([truncated_mdot(t) for t in times])

        # Ensure sorted (times should already be sorted, but just to be safe)

        order = np.argsort(times)

        times_sorted = times[order]

        values_sorted = values[order]

        # RocketPy Function expects 2D array: [[x1, y1], [x2, y2], ...]

        source = np.column_stack((times_sorted, values_sorted))

        return Function(source)

    else:

        # It's a constant - create a function that's constant until cutoff, then 0

        times = np.linspace(0, burn_time, int(burn_time * 100) + 1)

        values = np.array([float(mdot_func) if t <= cutoff_time else 0.0 for t in times])

        # Ensure sorted (times should already be sorted, but just to be safe)

        order = np.argsort(times)

        times_sorted = times[order]

        values_sorted = values[order]

        # RocketPy Function expects 2D array: [[x1, y1], [x2, y2], ...]

        source = np.column_stack((times_sorted, values_sorted))

        return Function(source)



def setup_flight(config, thrust_curve, mdot_lox, mdot_fuel, plot_results=False):

    """

    Build and simulate a RocketPy flight with configuration from config_minimal.yaml.



    Args:

        config: Configuration object with attributes matching config_minimal.yaml structure.

        plot_results (bool): whether to show diagnostic plots.



    Returns:

        dict: {

            "apogee": float,

            "max_velocity": float,

            "thrust_curve": list of (t, F),

            "flight": RocketPy Flight object,

            "params": configuration data

        }

    """







    # Extract parameters from config directly

    burn_time = config.thrust.burn_time



    # Densities from config

    rho_lox = config.fluids['oxidizer'].density

    rho_rp1 = config.fluids['fuel'].density



    # Initial masses from config

    m_lox0 = config.lox_tank.mass

    m_rp10 = config.fuel_tank.mass

    

    # Check for both LOX and fuel underfill and truncate at whichever happens first

    lox_cutoff_time = detect_lox_underfill_time(mdot_lox, m_lox0, burn_time)

    fuel_cutoff_time = detect_fuel_underfill_time(mdot_fuel, m_rp10, burn_time)

    

    # Find the earliest cutoff time (or None if neither depletes)

    cutoff_time = None

    cutoff_reason = None

    if lox_cutoff_time is not None and fuel_cutoff_time is not None:

        if lox_cutoff_time <= fuel_cutoff_time:

            cutoff_time = lox_cutoff_time

            cutoff_reason = "LOX"

        else:

            cutoff_time = fuel_cutoff_time

            cutoff_reason = "fuel"

    elif lox_cutoff_time is not None:

        cutoff_time = lox_cutoff_time

        cutoff_reason = "LOX"

    elif fuel_cutoff_time is not None:

        cutoff_time = fuel_cutoff_time

        cutoff_reason = "fuel"

    

    truncation_info = None

    if cutoff_time is not None and cutoff_time < burn_time:

        # Add small safety margin (0.01s or 1% of cutoff_time, whichever is larger) to prevent numerical precision issues
        # This ensures mass never goes negative due to integration/discretization errors
        safety_margin = max(0.01, cutoff_time * 0.01)
        safe_cutoff_time = max(0.0, cutoff_time - safety_margin)
        
        truncation_msg = f"{cutoff_reason.capitalize()} tank underfill detected at t={cutoff_time:.3f} s. Truncating thrust and mass flows at t={safe_cutoff_time:.3f} s (with safety margin)."

        # Note: Message is included in truncation_info, can be logged/displayed by caller if needed

        truncation_info = {

            "truncated": True,

            "cutoff_time": cutoff_time,

            "safe_cutoff_time": safe_cutoff_time,

            "reason": cutoff_reason,

            "message": truncation_msg

        }

        # Truncate thrust curve at safe_cutoff_time to ensure mass never goes negative
        thrust_curve = truncate_thrust_curve(thrust_curve, safe_cutoff_time)

        # Truncate mdot functions at safe_cutoff_time
        mdot_lox = truncate_mdot_function(mdot_lox, safe_cutoff_time, burn_time)

        mdot_fuel = truncate_mdot_function(mdot_fuel, safe_cutoff_time, burn_time)

        # Update burn_time to safe_cutoff_time (but keep original for tank discretization)

        effective_burn_time = safe_cutoff_time

    else:

        effective_burn_time = burn_time

        truncation_info = {"truncated": False}



    # Nozzle parameters from config

    eta_nozzle = config.nozzle.efficiency

    eps = config.nozzle.expansion_ratio

    A_t = config.nozzle.A_throat

    A_e = config.nozzle.A_exit

    

    # Check for required flight simulation config fields

    if not config.environment:

        raise ValueError("Flight simulation requires 'environment' configuration")

    if not config.rocket:

        raise ValueError("Flight simulation requires 'rocket' configuration")

    if not config.lox_tank:

        raise ValueError("Flight simulation requires 'lox_tank' configuration")

    if not config.fuel_tank:

        raise ValueError("Flight simulation requires 'fuel_tank' configuration")

    

    p_amb = config.environment.p_amb



    # Rocket parameters from config

    rocket_mass = config.rocket.mass

    rocket_inertia = config.rocket.inertia

    rocket_radius = config.rocket.radius

    cm_wo_motor = config.rocket.cm_wo_motor

    motor_dry_mass = config.rocket.motor.dry_mass

    motor_inertia = config.rocket.motor_inertia



    # Environment

    env = Environment(

        date=config.environment.date,

        latitude=config.environment.latitude,

        longitude=config.environment.longitude,

        elevation=config.environment.elevation,

    )

    # Use standard_atmosphere instead of Forecast to avoid forecast data availability issues
    # standard_atmosphere works for any date/time and doesn't require internet/forecast files
    env.set_atmospheric_model(type='standard_atmosphere')



    # Tank geometries from config

    lox_geom = CylindricalTank(radius=config.lox_tank.lox_radius, height=config.lox_tank.lox_h, spherical_caps=False)

    rp1_geom = CylindricalTank(radius=config.fuel_tank.rp1_radius, height=config.fuel_tank.rp1_h, spherical_caps=False)



    # Fluids and tanks

    lox = Fluid(name="LOX", density=rho_lox)

    rp1 = Fluid(name="RP-1", density=rho_rp1)

    pressurant = Fluid(name="LN2", density=807)  # kg/m,%% at 1 atm, 300 K



    # Convert mdot_lox and mdot_fuel to Functions if they're constants

    # (MassFlowRateBasedTank expects Functions)

    if not isinstance(mdot_lox, Function):

        times_mdot = np.linspace(0, burn_time, int(burn_time * 100) + 1)

        mdot_lox_vals = np.array([float(mdot_lox) if t <= effective_burn_time else 0.0 for t in times_mdot])

        # Ensure sorted (times should already be sorted, but just to be safe)

        order = np.argsort(times_mdot)

        times_sorted = times_mdot[order]

        vals_sorted = mdot_lox_vals[order]

        # RocketPy Function expects 2D array: [[x1, y1], [x2, y2], ...]

        source = np.column_stack((times_sorted, vals_sorted))

        mdot_lox = Function(source)

    

    if not isinstance(mdot_fuel, Function):

        times_mdot = np.linspace(0, burn_time, int(burn_time * 100) + 1)

        mdot_fuel_vals = np.array([float(mdot_fuel) if t <= effective_burn_time else 0.0 for t in times_mdot])

        # Ensure sorted (times should already be sorted, but just to be safe)

        order = np.argsort(times_mdot)

        times_sorted = times_mdot[order]

        vals_sorted = mdot_fuel_vals[order]

        # RocketPy Function expects 2D array: [[x1, y1], [x2, y2], ...]

        source = np.column_stack((times_sorted, vals_sorted))

        mdot_fuel = Function(source)



    oxidizer_tank = MassFlowRateBasedTank(

        name="LOX Tank",

        geometry=lox_geom,

        flux_time=effective_burn_time,  # Use effective_burn_time after truncation

        liquid=lox,

        gas=pressurant,

        initial_liquid_mass=m_lox0,

        initial_gas_mass=0.05,

        liquid_mass_flow_rate_in=0.0,

        liquid_mass_flow_rate_out=mdot_lox,  # Already truncated if underfill detected

        gas_mass_flow_rate_in=0.0,

        gas_mass_flow_rate_out=0.0,

        discretize=100,

    )



    fuel_tank = MassFlowRateBasedTank(

        name="RP-1 Tank",

        geometry=rp1_geom,

        flux_time=effective_burn_time,  # Use effective_burn_time after truncation

        liquid=rp1,

        gas=pressurant,

        initial_liquid_mass=m_rp10,

        initial_gas_mass=0.05,

        liquid_mass_flow_rate_in=0.0,

        liquid_mass_flow_rate_out=mdot_fuel,  # Already truncated if underfill detected

        gas_mass_flow_rate_in=0.0,

        gas_mass_flow_rate_out=0.0,

        discretize=100,

    )





    times = np.linspace(0, burn_time, int(burn_time * 100) + 1)

    # thrust_curve is already set above (may have been truncated)



    # Liquid motor - use effective_burn_time for burn_time

    liquid_motor = LiquidMotor(

        thrust_source=thrust_curve,

        center_of_dry_mass_position=0.0,

        dry_inertia=motor_inertia,

        dry_mass=motor_dry_mass,

        burn_time=(0.0, effective_burn_time),

        nozzle_radius=math.sqrt(A_e / math.pi),

        nozzle_position=-0.6,

        coordinate_system_orientation="nozzle_to_combustion_chamber",

    )



    # Rocket assembly - stack from bottom (tail) to top (nose)

    # In "tail_to_nose" system: lower position = tail, higher position = nose

    

    # Position motor at bottom, above fins

    motor_position = 0.5  # Motor center position

    

    # Add tanks relative to motor center

    # Fuel tank below LOX (negative relative to motor center)

    liquid_motor.add_tank(fuel_tank, position=config.fuel_tank.fuel_tank_pos)

    # LOX tank above fuel (positive relative to motor center)

    liquid_motor.add_tank(oxidizer_tank, position=config.lox_tank.ox_tank_pos)



    rocket = Rocket(

        radius=rocket_radius,

        mass=rocket_mass,

        inertia=rocket_inertia,

        center_of_mass_without_motor=cm_wo_motor,

        coordinate_system_orientation="tail_to_nose",

        power_off_drag=0.45,

        power_on_drag=0.45,

    )

    

    # Fins at bottom (tail) - position 0.0

    rocket.add_trapezoidal_fins(

        n=config.rocket.fins.no_fins,

        root_chord=config.rocket.fins.root_chord,

        tip_chord=config.rocket.fins.tip_chord,

        span=config.rocket.fins.fin_span,

        position=0.3  # Bottom of rocket

    )

    

    # Motor above fins

    rocket.add_motor(liquid_motor, position=motor_position)

    

    # Calculate top of highest tank to place nose above it

    # Motor center is at motor_position

    # LOX tank extends from motor_position + ox_tank_pos - lox_h/2 to motor_position + ox_tank_pos + lox_h/2

    lox_top = motor_position + config.lox_tank.ox_tank_pos + config.lox_tank.lox_h/2

    fuel_top = motor_position + config.fuel_tank.fuel_tank_pos + config.fuel_tank.rp1_h/2 if config.fuel_tank.fuel_tank_pos > 0 else 0

    

    # If pressurant tank is configured, include it

    press_top = 0

    if config.press_tank:

        press_top = motor_position + config.press_tank.pres_tank_pos + config.press_tank.press_h/2

    

    # Nose at top - above highest component

    max_height = max(lox_top, fuel_top, press_top, motor_position)

    nose_position = max_height + 4  # Small gap, then nose

    rocket.add_nose(length=0.6, kind="vonKarman", position=nose_position)



    # Flight simulation

    flight = Flight(

        rocket=rocket,

        environment=env,

        rail_length=3.35,

        inclination=90,

        heading=0,

        max_time_step=0.02,

        terminate_on_apogee=True,

    )



    apogee = float(flight.apogee)

    try:

        max_v = float(np.max(flight.vz.get_source()))

    except Exception:

        max_v = None



    print(f"Apogee [m]: {apogee:.2f}")

    if max_v is not None:

        print(f"Max velocity [m/s]: {max_v:.2f}")



    return {

        "apogee": apogee,

        "max_velocity": max_v,

        "thrust_curve": thrust_curve,

        "flight": flight,

        "params": config,

        "truncation_info": truncation_info,

    }

