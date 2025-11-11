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
    env.set_atmospheric_model(type='Forecast', file='GFS')

    print(m_lox0)
    print(m_rp10)
    print(mdot_lox)
    print(mdot_fuel)

    # Tank geometries from config
    lox_geom = CylindricalTank(radius=config.lox_tank.lox_radius, height=config.lox_tank.lox_h, spherical_caps=False)
    rp1_geom = CylindricalTank(radius=config.fuel_tank.rp1_radius, height=config.fuel_tank.rp1_h, spherical_caps=False)

    # Fluids and tanks
    lox = Fluid(name="LOX", density=rho_lox)
    rp1 = Fluid(name="RP-1", density=rho_rp1)
    pressurant = Fluid(name="LN2", density=807)  # kg/m³ at 1 atm, 300 K

    oxidizer_tank = MassFlowRateBasedTank(
        name="LOX Tank",
        geometry=lox_geom,
        flux_time=burn_time,
        liquid=lox,
        gas=pressurant,
        initial_liquid_mass=m_lox0,
        initial_gas_mass=0.05,
        liquid_mass_flow_rate_in=0.0,
        liquid_mass_flow_rate_out=mdot_lox,
        gas_mass_flow_rate_in=0.0,
        gas_mass_flow_rate_out=0.0,
        discretize=100,
    )

    fuel_tank = MassFlowRateBasedTank(
        name="RP-1 Tank",
        geometry=rp1_geom,
        flux_time=burn_time,
        liquid=rp1,
        gas=pressurant,
        initial_liquid_mass=m_rp10,
        initial_gas_mass=0.05,
        liquid_mass_flow_rate_in=0.0,
        liquid_mass_flow_rate_out=mdot_fuel,
        gas_mass_flow_rate_in=0.0,
        gas_mass_flow_rate_out=0.0,
        discretize=100,
    )


    times = np.linspace(0, burn_time, int(burn_time * 100) + 1)
    thrust_curve = thrust_curve

    # Liquid motor
    liquid_motor = LiquidMotor(
        thrust_source=thrust_curve,
        center_of_dry_mass_position=0.0,
        dry_inertia=motor_inertia,
        dry_mass=motor_dry_mass,
        burn_time=(0.0, burn_time),
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
    }