"""Streamlit UI for the pintle engine pipeline."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Tuple

import sys

import copy

import pandas as pd
import streamlit as st
import yaml

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pintle_pipeline.io import load_config
from pintle_pipeline.config_schemas import PintleEngineConfig
from pintle_models.runner import PintleEngineRunner
from examples.pintle_engine.interactive_pipeline import solve_for_thrust, ThrustSolveError

PSI_TO_PA = 6894.76
PA_TO_PSI = 1.0 / PSI_TO_PA

CONFIG_PATH = Path(__file__).parent / "config_minimal.yaml"

FLUID_LIBRARY: Dict[str, Dict[str, float]] = {
    "LOX": {
        "name": "LOX",
        "density": 1140.0,
        "viscosity": 1.8e-4,
        "surface_tension": 0.013,
        "vapor_pressure": 101325.0,
    },
    "LH2": {
        "name": "LH2",
        "density": 70.0,
        "viscosity": 1.3e-4,
        "surface_tension": 0.002,
        "vapor_pressure": 70000.0,
    },
    "RP-1": {
        "name": "RP-1",
        "density": 780.0,
        "viscosity": 2.0e-3,
        "surface_tension": 0.025,
        "vapor_pressure": 1000.0,
    },
    "Alcohol": {
        "name": "Ethanol",
        "density": 789.0,
        "viscosity": 1.2e-3,
        "surface_tension": 0.022,
        "vapor_pressure": 5800.0,
    },
}

OXIDIZER_OPTIONS = list(FLUID_LIBRARY.keys()) + ["Custom"]
FUEL_OPTIONS = list(FLUID_LIBRARY.keys()) + ["Custom"]

INJECTOR_OPTIONS = {
    "Pintle": {
        "type": "pintle",
        "supported": True,
        "description": "Axial LOX orifices with radial fuel annulus.",
    },
    "Coaxial": {
        "type": "coaxial",
        "supported": True,
        "description": "Shear coaxial: central oxidizer core with annular fuel (optional swirl).",
    },
    "Impinging": {
        "type": "impinging",
        "supported": True,
        "description": "Impinging element injector using opposing jets for atomization.",
    },
}


@lru_cache(maxsize=1)
def load_default_runner() -> PintleEngineRunner:
    config = load_config(str(CONFIG_PATH))
    return PintleEngineRunner(config)


def get_default_config_dict() -> Dict[str, Any]:
    config = load_config(str(CONFIG_PATH))
    return config.model_dump()


def summarize_results(results: Dict[str, Any]) -> None:
    Pc_psi = results["Pc"] * PA_TO_PSI
    thrust_kN = results["F"] / 1000.0
    mdot_total = results["mdot_total"]
    mdot_O = results["mdot_O"]
    mdot_F = results["mdot_F"]
    MR = results["MR"]
    Isp = results["Isp"]
    cstar = results["cstar_actual"]
    v_exit = results["v_exit"]
    P_exit_psi = results["P_exit"] * PA_TO_PSI

    st.metric("Thrust", f"{thrust_kN:.2f} kN")
    st.metric("Specific Impulse", f"{Isp:.1f} s")
    st.metric("Chamber Pressure", f"{Pc_psi:.1f} psi")
    st.metric("Total Mass Flow", f"{mdot_total:.3f} kg/s")
    st.metric("Oxidizer Flow", f"{mdot_O:.3f} kg/s")
    st.metric("Fuel Flow", f"{mdot_F:.3f} kg/s")
    st.metric("Mixture Ratio (O/F)", f"{MR:.3f}")
    st.metric("c* (actual)", f"{cstar:.1f} m/s")
    st.metric("Exit Velocity", f"{v_exit:.1f} m/s")
    st.metric("Exit Pressure", f"{P_exit_psi:.2f} psi")

    cooling = results.get("cooling", {})
    if cooling:
        st.subheader("Cooling Summary")
        regen = cooling.get("regen")
        if regen and regen.get("enabled", False):
            st.caption("Regenerative cooling")
            st.write(
                f"Coolant outlet temperature: {regen['coolant_outlet_temperature']:.1f} K"
            )
            st.write(
                f"Heat removed: {regen['heat_removed']/1000:.1f} kW | Hot-side heat flux: {regen['overall_heat_flux']/1000:.1f} kW/m²"
            )
            if "mdot_coolant" in regen:
                st.write(f"Coolant flow through channels: {regen['mdot_coolant']:.3f} kg/s")
            if "wall_temperature_coolant" in regen:
                st.write(
                    f"Wall temperature (hot/cool): {regen['wall_temperature_hot']:.1f} K / {regen['wall_temperature_coolant']:.1f} K"
                )
            if regen.get("film_effectiveness", 0.0) > 0:
                st.write(f"Film effectiveness contribution: {regen['film_effectiveness']:.2f}")
        film = cooling.get("film")
        if film and film.get("enabled", False):
            st.caption("Film cooling")
            st.write(
                f"Mass fraction: {film['mass_fraction']:.3f} | Effectiveness: {film['effectiveness']:.2f}"
            )
            st.write(
                f"Film mass flow: {film['mdot_film']:.3f} kg/s | Heat-flux factor: {film['heat_flux_factor']:.2f}"
            )
        ablative = cooling.get("ablative")
        if ablative and ablative.get("enabled", False):
            st.caption("Ablative cooling")
            st.write(
                f"Recession rate: {ablative['recession_rate']*1e6:.3f} µm/s | Effective heat flux: {ablative['effective_heat_flux']/1000:.1f} kW/m²"
            )


def forward_view(runner: PintleEngineRunner) -> None:
    st.header("Forward Mode: Tank Pressures → Performance")

    col1, col2 = st.columns(2)
    with col1:
        P_tank_O_psi = st.slider(
            "LOX Tank Pressure [psi]",
            min_value=200.0,
            max_value=1500.0,
            value=1305.0,
            step=5.0,
        )
    with col2:
        P_tank_F_psi = st.slider(
            "Fuel Tank Pressure [psi]",
            min_value=200.0,
            max_value=1500.0,
            value=974.0,
            step=5.0,
        )

    if st.button("Compute Performance", type="primary"):
        try:
            results = runner.evaluate(
                P_tank_O_psi * PSI_TO_PA,
                P_tank_F_psi * PSI_TO_PA,
            )
            summarize_results(results)
        except Exception as exc:
            st.error(f"Pipeline evaluation failed: {exc}")


def inverse_view(runner: PintleEngineRunner, config_label: str) -> None:
    st.header("Inverse Mode: Target Thrust → Tank Pressures")

    target_thrust_kN = st.number_input(
        "Desired Thrust [kN]",
        min_value=0.1,
        value=6.65,
        step=0.1,
    )

    col1, col2 = st.columns(2)
    with col1:
        base_O_psi = st.number_input(
            "Baseline LOX Tank Pressure [psi]",
            min_value=200.0,
            value=1305.0,
            step=10.0,
        )
    with col2:
        base_F_psi = st.number_input(
            "Baseline Fuel Tank Pressure [psi]",
            min_value=200.0,
            value=974.0,
            step=10.0,
        )

    if st.button("Solve for Tank Pressures", type="primary"):
        try:
            (P_tank_O_solution, P_tank_F_solution), results, diagnostics = solve_for_thrust(
                runner,
                target_thrust_kN,
                (base_O_psi, base_F_psi),
            )
        except ThrustSolveError as exc:
            st.error(str(exc))
            diag = exc.diagnostics
            st.info(
                f"Achievable thrust range: {diag['min_thrust']:.2f} - {diag['max_thrust']:.2f} kN\n"
                f"Baseline thrust (scale=1.0): {diag['baseline_thrust']:.2f} kN"
            )
            return
        except Exception as exc:
            st.error(f"Failed to find tank pressures: {exc}")
            return

        st.subheader("Required Tank Pressures")
        st.metric("LOX Tank Pressure", f"{P_tank_O_solution * PA_TO_PSI:.1f} psi")
        st.metric("Fuel Tank Pressure", f"{P_tank_F_solution * PA_TO_PSI:.1f} psi")

        st.subheader("Performance at Solution")
        summarize_results(results)

        diag = diagnostics
        st.info(
            f"Configuration: {config_label}\n"
            f"Baseline thrust (scale=1.0): {diag['baseline_thrust']:.2f} kN\n"
            f"Achievable thrust range: {diag['min_thrust']:.2f} - {diag['max_thrust']:.2f} kN"
        )

        with st.expander("Diagnostics: Thrust vs Scale"):
            st.write("This table shows sampled thrust values used to bracket the solution.")
            diag_table = {
                "Scale": list(diag["sample_scales"]),
                "Thrust [kN]": list(diag["sample_thrusts"]),
            }
            st.dataframe(diag_table)


def timeseries_view(runner: PintleEngineRunner, config_label: str) -> None:
    st.header("Time-Series Evaluation: Pressure Curve → Thrust Curve")
    st.write(
        "Upload a CSV with columns `time` (s), `P_tank_O` (psi), `P_tank_F` (psi). "
        "The pipeline will solve for chamber pressure and thrust at each time step."
    )

    uploaded_csv = st.file_uploader(
        "Upload pressure profile CSV",
        type=["csv"],
        key="timeseries_upload",
    )

    if uploaded_csv is None:
        st.info("Upload a CSV to generate thrust curves.")
        return

    try:
        df = pd.read_csv(uploaded_csv)
    except Exception as exc:
        st.error(f"Failed to read CSV: {exc}")
        return

    required_cols = {"P_tank_O", "P_tank_F"}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must contain columns: {required_cols}")
        return

    if "time" not in df.columns:
        df["time"] = range(len(df))

    results_data = []
    for _, row in df.iterrows():
        try:
            res = runner.evaluate(row["P_tank_O"] * PSI_TO_PA, row["P_tank_F"] * PSI_TO_PA)
        except Exception as exc:
            st.error(f"Pipeline failed at time={row['time']}: {exc}")
            return

        results_data.append(
            {
                "time": row["time"],
                "P_tank_O": row["P_tank_O"],
                "P_tank_F": row["P_tank_F"],
                "Pc (psi)": res["Pc"] * PA_TO_PSI,
                "Thrust (kN)": res["F"] / 1000.0,
                "Isp (s)": res["Isp"],
                "MR": res["MR"],
            }
        )

    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values("time")

    st.subheader("Thrust Curve")
    st.line_chart(results_df.set_index("time")["Thrust (kN)"])

    st.subheader("Chamber Pressure")
    st.line_chart(results_df.set_index("time")["Pc (psi)"])

    with st.expander("Detailed Results"):
        st.dataframe(results_df)
        st.download_button(
            "Download results as CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="pintle_pipeline_timeseries_results.csv",
            mime="text/csv",
        )

    st.success(
        f"Generated thrust curve using configuration '{config_label}'."
    )


def detect_fluid_choice(fluid: Dict[str, Any]) -> str:
    name = fluid.get("name")
    if name in FLUID_LIBRARY:
        defaults = FLUID_LIBRARY[name]
        tolerance = 1e-3
        if all(abs(float(fluid[key]) - defaults[key]) <= tolerance * max(1.0, abs(defaults[key])) for key in ["density", "viscosity", "surface_tension", "vapor_pressure"]):
            return name
    for candidate, defaults in FLUID_LIBRARY.items():
        tolerance = 1e-3
        if all(abs(float(fluid[key]) - defaults[key]) <= tolerance * max(1.0, abs(defaults[key])) for key in ["density", "viscosity", "surface_tension", "vapor_pressure"]):
            return candidate
    return "Custom"


def load_config_state(uploaded_file) -> Tuple[PintleEngineConfig, str]:
    if "config_dict" not in st.session_state:
        st.session_state["config_dict"] = get_default_config_dict()
        st.session_state["config_label"] = str(CONFIG_PATH)

    if uploaded_file is not None:
        try:
            config_text = uploaded_file.getvalue().decode("utf-8")
            config_dict = yaml.safe_load(config_text)
            PintleEngineConfig(**config_dict)
            st.session_state["config_dict"] = config_dict
            st.session_state["config_label"] = uploaded_file.name
        except Exception as exc:
            raise ValueError(f"Failed to load uploaded configuration: {exc}") from exc

    try:
        config_obj = PintleEngineConfig(**st.session_state["config_dict"])
    except Exception as exc:
        raise ValueError(f"Invalid configuration state: {exc}") from exc

    ox_choice = detect_fluid_choice(config_obj.fluids["oxidizer"].model_dump())
    fuel_choice = detect_fluid_choice(config_obj.fluids["fuel"].model_dump())
    st.session_state.setdefault("oxidizer_choice", ox_choice)
    st.session_state.setdefault("fuel_choice", fuel_choice)
    st.session_state.setdefault("injector_choice", "Pintle")

    return config_obj, st.session_state["config_label"]


def config_editor(config: PintleEngineConfig) -> PintleEngineConfig:
    working_copy = copy.deepcopy(st.session_state.get("config_dict", config.model_dump()))
    if "pintle_geometry" in working_copy and "injector" not in working_copy:
        working_copy["injector"] = {
            "type": "pintle",
            "geometry": working_copy.pop("pintle_geometry"),
        }

    with st.sidebar.form("config_form"):
        st.markdown("### Propellants")

        ox_choice = st.selectbox(
            "Oxidizer",
            OXIDIZER_OPTIONS,
            index=OXIDIZER_OPTIONS.index(st.session_state.get("oxidizer_choice", "Custom")),
            key="oxidizer_choice",
        )
        ox = working_copy["fluids"]["oxidizer"]
        if ox_choice != "Custom" and ox_choice in FLUID_LIBRARY:
            ox.update(FLUID_LIBRARY[ox_choice])
        ox["name"] = ox_choice if ox_choice != "Custom" else ox.get("name", "Custom Oxidizer")
        ox["density"] = st.number_input("Oxidizer density [kg/m³]", min_value=200.0, max_value=3000.0, value=float(ox["density"]))
        ox["viscosity"] = st.number_input("Oxidizer viscosity [Pa·s]", min_value=1e-5, max_value=1e-2, value=float(ox["viscosity"]))
        ox["surface_tension"] = st.number_input("Oxidizer surface tension [N/m]", min_value=1e-3, max_value=0.05, value=float(ox["surface_tension"]))
        ox["vapor_pressure"] = st.number_input("Oxidizer vapor pressure [Pa]", min_value=0.0, max_value=3e6, value=float(ox["vapor_pressure"]))

        fuel_choice = st.selectbox(
            "Fuel",
            FUEL_OPTIONS,
            index=FUEL_OPTIONS.index(st.session_state.get("fuel_choice", "Custom")),
            key="fuel_choice",
        )
        fuel = working_copy["fluids"]["fuel"]
        if fuel_choice != "Custom" and fuel_choice in FLUID_LIBRARY:
            fuel.update(FLUID_LIBRARY[fuel_choice])
        fuel["name"] = fuel_choice if fuel_choice != "Custom" else fuel.get("name", "Custom Fuel")
        fuel["density"] = st.number_input("Fuel density [kg/m³]", min_value=200.0, max_value=3000.0, value=float(fuel["density"]))
        fuel["viscosity"] = st.number_input("Fuel viscosity [Pa·s]", min_value=1e-5, max_value=1e-2, value=float(fuel["viscosity"]))
        fuel["surface_tension"] = st.number_input("Fuel surface tension [N/m]", min_value=1e-3, max_value=0.05, value=float(fuel["surface_tension"]))
        fuel["vapor_pressure"] = st.number_input("Fuel vapor pressure [Pa]", min_value=0.0, max_value=3e6, value=float(fuel["vapor_pressure"]))

        st.markdown("### Injector")
        current_choice = st.session_state.get("injector_choice", "Pintle")
        if current_choice not in INJECTOR_OPTIONS:
            current_choice = "Pintle"
        injector_choice = st.selectbox(
            "Injector Type",
            list(INJECTOR_OPTIONS.keys()),
            index=list(INJECTOR_OPTIONS.keys()).index(current_choice),
        )
        st.session_state["injector_choice"] = injector_choice
        injector_info = INJECTOR_OPTIONS[injector_choice]
        injector_type = injector_info["type"]
        st.caption(injector_info["description"])

        injector_dict = working_copy.setdefault("injector", {"type": injector_type, "geometry": {}})
        injector_dict["type"] = injector_type

        if injector_type == "pintle":
            st.markdown("#### Pintle Geometry")
            pintle_geom = injector_dict.setdefault("geometry", {})
            lox_geom = pintle_geom.setdefault("lox", {})
            fuel_geom = pintle_geom.setdefault("fuel", {})
            lox_geom.setdefault("n_orifices", 12)
            lox_geom.setdefault("d_orifice", 1.5e-3)
            fuel_geom.setdefault("d_pintle_tip", 0.02)
            fuel_geom.setdefault("h_gap", 0.0005)

            lox_geom["n_orifices"] = int(
                st.number_input("Number of LOX orifices", min_value=1, max_value=128, value=int(lox_geom["n_orifices"]))
            )
            lox_geom["d_orifice"] = st.number_input(
                "LOX orifice diameter [m]", min_value=1e-4, max_value=1e-2, value=float(lox_geom["d_orifice"]))
            fuel_geom["d_pintle_tip"] = st.number_input(
                "Fuel pintle tip diameter [m]", min_value=5e-3, max_value=0.05, value=float(fuel_geom["d_pintle_tip"]))
            fuel_geom["h_gap"] = st.number_input(
                "Fuel gap height [m]", min_value=1e-4, max_value=5e-3, value=float(fuel_geom["h_gap"]))
        elif injector_type == "coaxial":
            st.markdown("#### Coaxial Geometry")
            coax_geom = injector_dict.setdefault("geometry", {})
            core_geom = coax_geom.setdefault("core", {})
            ann_geom = coax_geom.setdefault("annulus", {})

            core_geom.setdefault("n_ports", 12)
            core_geom.setdefault("d_port", 1.4e-3)
            core_geom.setdefault("length", 0.015)
            ann_geom.setdefault("inner_diameter", 5.0e-3)
            ann_geom.setdefault("gap_thickness", 8.0e-4)
            ann_geom.setdefault("swirl_angle", 20.0)

            core_geom["n_ports"] = int(
                st.number_input("Core ports", min_value=1, max_value=256, value=int(core_geom["n_ports"]))
            )
            core_geom["d_port"] = st.number_input(
                "Core port diameter [m]", min_value=2e-4, max_value=1e-2, value=float(core_geom["d_port"])
            )
            core_length_val = st.number_input(
                "Core port length [m] (0 = auto)", min_value=0.0, max_value=0.1, value=float(core_geom.get("length") or 0.0)
            )
            core_geom["length"] = core_length_val if core_length_val > 0 else None

            ann_geom["inner_diameter"] = st.number_input(
                "Annulus inner diameter [m]", min_value=2e-3, max_value=0.05, value=float(ann_geom["inner_diameter"])
            )
            ann_geom["gap_thickness"] = st.number_input(
                "Annulus gap thickness [m]", min_value=1e-4, max_value=5e-3, value=float(ann_geom["gap_thickness"])
            )
            ann_geom["swirl_angle"] = st.slider(
                "Swirl angle [deg]", min_value=0.0, max_value=80.0, value=float(ann_geom["swirl_angle"]), step=1.0
            )

        elif injector_type == "impinging":
            st.markdown("#### Impinging Geometry")
            imp_geom = injector_dict.setdefault("geometry", {})
            ox_geom = imp_geom.setdefault("oxidizer", {})
            fuel_geom_imp = imp_geom.setdefault("fuel", {})

            for branch, geom in ("Oxidizer", ox_geom), ("Fuel", fuel_geom_imp):
                geom.setdefault("n_elements", 8)
                geom.setdefault("d_jet", 1.2e-3)
                geom.setdefault("impingement_angle", 60.0)
                geom.setdefault("spacing", 4.0e-3)

                st.subheader(f"{branch} Jets")
                geom["n_elements"] = int(
                    st.number_input(f"{branch} elements", min_value=1, max_value=128, value=int(geom["n_elements"]))
                )
                geom["d_jet"] = st.number_input(
                    f"{branch} jet diameter [m]", min_value=2e-4, max_value=5e-3, value=float(geom["d_jet"])
                )
                geom["impingement_angle"] = st.slider(
                    f"{branch} impingement angle [deg]", min_value=20.0, max_value=180.0, value=float(geom["impingement_angle"]), step=1.0
                )
                geom["spacing"] = st.number_input(
                    f"{branch} jet spacing [m]", min_value=1e-3, max_value=0.02, value=float(geom["spacing"])
                )

        st.markdown("### Feed System")
        feed = working_copy["feed_system"]
        ox_feed = feed["oxidizer"]
        fuel_feed = feed["fuel"]
        ox_feed["d_inlet"] = st.number_input("LOX inlet diameter [m]", min_value=1e-3, max_value=0.05, value=float(ox_feed["d_inlet"]))
        ox_feed["K0"] = st.number_input("LOX loss coefficient K0", min_value=0.0, max_value=10.0, value=float(ox_feed["K0"]))
        fuel_feed["d_inlet"] = st.number_input("Fuel inlet diameter [m]", min_value=1e-3, max_value=0.05, value=float(fuel_feed["d_inlet"]))
        fuel_feed["K0"] = st.number_input("Fuel loss coefficient K0", min_value=0.0, max_value=10.0, value=float(fuel_feed["K0"]))

        st.markdown("### Regenerative Cooling")
        regen = working_copy["regen_cooling"]
        regen["enabled"] = st.checkbox("Enable regenerative cooling", value=bool(regen["enabled"]))
        regen["n_channels"] = int(st.number_input("Channels", min_value=1, max_value=400, value=int(regen["n_channels"])) )
        regen["channel_width"] = st.number_input("Channel width [m]", min_value=1e-4, max_value=5e-3, value=float(regen["channel_width"]))
        regen["channel_height"] = st.number_input("Channel height [m]", min_value=1e-4, max_value=5e-3, value=float(regen["channel_height"]))
        regen["use_heat_transfer"] = st.checkbox("Enable coupled heat-transfer", value=bool(regen.get("use_heat_transfer", False)))
        regen["wall_thickness"] = st.number_input("Wall thickness [m]", min_value=1e-4, max_value=0.01, value=float(regen.get("wall_thickness", 0.002)))
        regen["wall_thermal_conductivity"] = st.number_input("Wall conductivity [W/(m·K)]", min_value=10.0, max_value=600.0, value=float(regen.get("wall_thermal_conductivity", 300.0)))
        regen["chamber_inner_diameter"] = st.number_input("Chamber inner diameter [m]", min_value=0.01, max_value=0.5, value=float(regen.get("chamber_inner_diameter", 0.08)))

        st.markdown("### Film Cooling")
        film_cfg = working_copy.setdefault("film_cooling", {
            "enabled": False,
            "mass_fraction": 0.05,
            "injection_temperature": None,
            "effectiveness_ref": 0.4,
            "decay_length": 0.1,
            "apply_to_fraction_of_length": 1.0,
        })
        film_cfg["enabled"] = st.checkbox("Enable film cooling", value=bool(film_cfg.get("enabled", False)))
        film_cfg["mass_fraction"] = st.number_input("Film mass fraction (of fuel)", min_value=0.0, max_value=0.5, value=float(film_cfg.get("mass_fraction", 0.05)))
        film_cfg["effectiveness_ref"] = st.slider("Reference effectiveness", min_value=0.0, max_value=1.0, value=float(film_cfg.get("effectiveness_ref", 0.4)), step=0.01)
        film_cfg["decay_length"] = st.number_input("Effectiveness decay length [m]", min_value=0.01, max_value=1.0, value=float(film_cfg.get("decay_length", 0.1)))
        film_cfg["apply_to_fraction_of_length"] = st.number_input("Fraction of chamber length covered", min_value=0.1, max_value=1.5, value=float(film_cfg.get("apply_to_fraction_of_length", 1.0)))
        film_temp_override = film_cfg.get("injection_temperature")
        film_cfg["injection_temperature"] = st.number_input(
            "Film injection temperature override [K] (0 = use fuel temp)",
            min_value=0.0,
            max_value=2000.0,
            value=float(film_temp_override or 0.0),
        ) or None

        st.markdown("### Ablative Cooling")
        ablative_cfg = working_copy.setdefault("ablative_cooling", {
            "enabled": False,
            "material_density": 1600.0,
            "heat_of_ablation": 2.5e6,
            "thermal_conductivity": 0.35,
            "specific_heat": 1500.0,
            "initial_thickness": 0.01,
            "surface_temperature_limit": 1200.0,
        })
        ablative_cfg["enabled"] = st.checkbox("Enable ablative liner", value=bool(ablative_cfg.get("enabled", False)))
        ablative_cfg["material_density"] = st.number_input("Ablator density [kg/m³]", min_value=200.0, max_value=4000.0, value=float(ablative_cfg.get("material_density", 1600.0)))
        ablative_cfg["heat_of_ablation"] = st.number_input("Heat of ablation [J/kg]", min_value=1e6, max_value=1e8, value=float(ablative_cfg.get("heat_of_ablation", 2.5e6)))
        ablative_cfg["initial_thickness"] = st.number_input("Initial thickness [m]", min_value=0.001, max_value=0.05, value=float(ablative_cfg.get("initial_thickness", 0.01)))
        ablative_cfg["surface_temperature_limit"] = st.number_input("Surface temperature limit [K]", min_value=500.0, max_value=2500.0, value=float(ablative_cfg.get("surface_temperature_limit", 1200.0)))

        st.markdown("### Chamber & Nozzle")
        chamber = working_copy["chamber"]
        nozzle = working_copy["nozzle"]
        chamber["A_throat"] = st.number_input("Throat area [m²]", min_value=1e-5, max_value=0.01, value=float(chamber["A_throat"]))
        chamber["Lstar"] = st.number_input("Characteristic length L* [m]", min_value=0.1, max_value=5.0, value=float(chamber["Lstar"]))
        nozzle["expansion_ratio"] = st.number_input("Expansion ratio (Ae/At)", min_value=1.0, max_value=200.0, value=float(nozzle["expansion_ratio"]))

        submitted = st.form_submit_button("Apply configuration changes")

    if submitted:
        try:
            working_copy["combustion"]["cea"]["ox_name"] = working_copy["fluids"]["oxidizer"].get("name", "Oxidizer")
            working_copy["combustion"]["cea"]["fuel_name"] = working_copy["fluids"]["fuel"].get("name", "Fuel")
            new_config = PintleEngineConfig(**working_copy)
            st.session_state["config_dict"] = working_copy
            st.success("Configuration updated.")
            return new_config
        except Exception as exc:
            st.error(f"Invalid configuration: {exc}")
            return config

    return PintleEngineConfig(**st.session_state["config_dict"])


def main():
    st.set_page_config(page_title="Pintle Engine Pipeline", layout="wide")
    st.title("Pintle Injector Engine Pipeline")

    st.sidebar.header("Configuration")
    uploaded_config = st.sidebar.file_uploader("Upload custom YAML config", type=["yaml", "yml"])

    try:
        config_obj, config_label = load_config_state(uploaded_config)
    except ValueError as exc:
        st.sidebar.error(str(exc))
        st.stop()

    config_obj = config_editor(config_obj)
    st.session_state["config_dict"] = config_obj.model_dump()
    st.session_state["config_label"] = config_label

    runner = PintleEngineRunner(config_obj)
    st.sidebar.success(f"Using configuration: {config_label}")

    tab1, tab2, tab3 = st.tabs(["Forward", "Inverse", "Time Series"])
    with tab1:
        forward_view(runner)
    with tab2:
        inverse_view(runner, config_label)
    with tab3:
        timeseries_view(runner, config_label)


if __name__ == "__main__":
    main()
