"""Experiment router — cold-flow Cd characterization and real-propellant prediction.

Per-row: mdot = weight / dT,  Cd = mdot / (A * sqrt(2 * rho_water * dP))
Exit pressure is taken as 0 (atmospheric reference), so dP = inlet pressure.
Mean Cd then predicts real-propellant mdot at operating conditions.
"""

from __future__ import annotations

import copy
import math
import statistics
from typing import List, Literal, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.state import app_state

router = APIRouter(prefix="/api/experiment", tags=["experiment"])

WATER_VAPOR_PRESSURE_PA = 2337.0   # Pa at 20 °C
WATER_VISCOSITY_PA_S    = 0.001    # Pa·s at 20 °C
PSI_TO_PA               = 6894.757


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ExperimentRow(BaseModel):
    label: str         = Field(default="Run")
    t0: float          = Field(description="Start time [s]")
    tf: float          = Field(description="End time [s]")
    delta_p_psi: float = Field(gt=0, description="Injector pressure drop during this run [psi]")
    weight: float      = Field(ge=0, description="Tank weight reading at start of this interval [kg]")


class ExperimentRequest(BaseModel):
    propellant: Literal["fuel", "lox"]
    choke_diameter_m: float      = Field(gt=0, description="Injector orifice diameter [m]")
    water_density: float         = Field(default=998.2, gt=0, description="Water density [kg/m³]")
    real_pressure_drop_pa: float = Field(gt=0, description="Real-propellant ΔP [Pa]")
    rows: List[ExperimentRow]    = Field(min_length=1)


class RowResult(BaseModel):
    label: str
    dt: float
    weight: float
    delta_p_psi: float
    mdot: float
    cd: float


class ValidationResult(BaseModel):
    cavitation_number: float
    cavitation_warning: bool
    re_water: float
    re_real: float
    re_ratio: float
    re_similarity_warning: bool


class ExperimentResponse(BaseModel):
    rows: List[RowResult]
    mean_cd: float
    std_cd: float
    cda_m2: float
    propellant_name: str
    rho_real: float
    mu_real: float
    mdot_real: float
    validation: ValidationResult


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/calculate", response_model=ExperimentResponse)
async def calculate_experiment(request: ExperimentRequest) -> ExperimentResponse:
    """Process cold-flow water test data and predict real-propellant mass flow rate."""

    area = math.pi * (request.choke_diameter_m / 2.0) ** 2

    # ---- per-row calculations (dW = weight[i] - weight[i-1], weight[-1] = 0) -
    row_results: List[RowResult] = []
    for i, row in enumerate(request.rows):
        dt = row.tf - row.t0
        if dt <= 0:
            raise HTTPException(status_code=400, detail=f"Row '{row.label}': tf must be > t0.")

        prev_weight = request.rows[i - 1].weight if i > 0 else 0.0
        dw          = row.weight - prev_weight
        if dw < 0:
            raise HTTPException(status_code=400, detail=f"Row '{row.label}': weight decreased vs previous row — check row order.")

        mdot        = dw / dt
        delta_p_pa  = row.delta_p_psi * PSI_TO_PA
        denominator = area * math.sqrt(2.0 * request.water_density * delta_p_pa)

        if denominator == 0:
            raise HTTPException(status_code=400, detail=f"Row '{row.label}': zero denominator — check diameter and ΔP.")

        cd = mdot / denominator
        row_results.append(RowResult(
            label=row.label, dt=dt, weight=dw,
            delta_p_psi=row.delta_p_psi, mdot=mdot, cd=cd,
        ))

    # ---------------------------------------------------------- summary stats
    cd_values = [r.cd for r in row_results]
    mean_cd   = statistics.mean(cd_values)
    std_cd    = statistics.stdev(cd_values) if len(cd_values) > 1 else 0.0
    cda_m2    = mean_cd * area

    # ----------------------------------------- propellant properties from config
    if not app_state.has_config():
        raise HTTPException(status_code=400, detail="No engine config loaded — needed for propellant properties.")

    fluid_key = "fuel" if request.propellant == "fuel" else "oxidizer"
    try:
        fluid = app_state.config.fluids[fluid_key]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Config has no fluids['{fluid_key}'].")

    rho_real        = fluid.density
    mu_real         = fluid.viscosity
    propellant_name = getattr(fluid, "name", fluid_key.upper())

    # ----------------------------------------- real-propellant prediction
    mdot_real = mean_cd * area * math.sqrt(2.0 * rho_real * request.real_pressure_drop_pa)

    # ---------------------------------------------------------- validation
    mean_mdot_water = statistics.mean([r.mdot for r in row_results])
    mean_dp_pa      = statistics.mean([r.delta_p_psi * PSI_TO_PA for r in row_results])

    v_water  = mean_mdot_water / (request.water_density * area) if area > 0 else 0.0
    re_water = request.water_density * v_water * request.choke_diameter_m / WATER_VISCOSITY_PA_S

    v_real   = mdot_real / (rho_real * area) if area > 0 else 0.0
    re_real  = rho_real * v_real * request.choke_diameter_m / mu_real if mu_real > 0 else 0.0
    re_ratio = (re_real / re_water) if re_water > 0 else float("inf")

    cavitation_number = mean_dp_pa / WATER_VAPOR_PRESSURE_PA

    validation = ValidationResult(
        cavitation_number=cavitation_number,
        cavitation_warning=cavitation_number < 2.0,
        re_water=re_water,
        re_real=re_real,
        re_ratio=re_ratio,
        re_similarity_warning=abs(re_ratio - 1.0) > 0.5,
    )

    return ExperimentResponse(
        rows=row_results, mean_cd=mean_cd, std_cd=std_cd, cda_m2=cda_m2,
        propellant_name=propellant_name, rho_real=rho_real, mu_real=mu_real,
        mdot_real=mdot_real, validation=validation,
    )


# ---------------------------------------------------------------------------
# Run-timeseries models and helpers
# ---------------------------------------------------------------------------

class PropellantTable(BaseModel):
    choke_diameter_m: float = Field(gt=0, description="Choke orifice diameter [m]")
    water_density: float    = Field(default=998.2, gt=0, description="Water density [kg/m³]")
    rows: List[ExperimentRow] = Field(min_length=1)


class FuelTable(PropellantTable):
    choke_diameter_m: float = Field(default=0.0079, gt=0, description="Fuel choke equivalent diameter [m]")


class LoxTable(PropellantTable):
    choke_diameter_m: float = Field(default=0.0074, gt=0, description="LOX choke equivalent diameter [m]")


class RunTimeseriesRequest(BaseModel):
    fuel: FuelTable
    lox:  LoxTable


def _extract_cd_pairs(table: PropellantTable) -> List[Tuple[float, float]]:
    """Return list of (delta_p_pa, cd) from per-row water-test data."""
    area   = math.pi * (table.choke_diameter_m / 2) ** 2
    pairs: List[Tuple[float, float]] = []
    for i, row in enumerate(table.rows):
        dt    = row.tf - row.t0
        if dt <= 0:
            continue
        prev_w = table.rows[i - 1].weight if i > 0 else 0.0
        dw     = row.weight - prev_w
        if dw < 0:
            continue
        dp_pa = row.delta_p_psi * PSI_TO_PA
        denom = area * math.sqrt(2.0 * table.water_density * dp_pa)
        if denom == 0:
            continue
        pairs.append((dp_pa, (dw / dt) / denom))
    return pairs


def _fit_cd_sqrt_dp(pairs: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Fit Cd as a linear function of sqrt(ΔP):
        Cd = a*sqrt(ΔP_pa) + b

    Returns:
        (a, b)
    """
    if not pairs:
        return 0.0, 0.0
    if len(pairs) == 1:
        # Constant Cd model: a=0, b=Cd
        return 0.0, float(pairs[0][1])

    dp = np.asarray([p for p, _ in pairs], dtype=float)
    cd = np.asarray([c for _, c in pairs], dtype=float)
    mask = np.isfinite(dp) & np.isfinite(cd) & (dp > 0.0)
    dp = dp[mask]
    cd = cd[mask]
    if dp.size == 0:
        return 0.0, 0.0
    if dp.size == 1:
        return 0.0, float(cd[0])

    x = np.sqrt(dp)
    y = cd
    a, b = np.polyfit(x, y, 1)
    if not np.isfinite(a) or not np.isfinite(b):
        return 0.0, float(np.nanmean(y)) if y.size else 0.0
    return float(a), float(b)


def _cd_from_fit(dp_pa: float, a: float, b: float) -> float:
    """Evaluate fitted Cd model at ΔP (Pa) with safety clamps."""
    if not np.isfinite(dp_pa) or dp_pa <= 0.0:
        return 0.0
    cd = a * math.sqrt(dp_pa) + b
    if not np.isfinite(cd):
        return 0.0
    return float(max(0.0, cd))


def _segments_to_curve_pa(segments, n_points: int) -> np.ndarray:
    """Build a pressure curve [Pa] using the existing segment generator."""
    from engine.optimizer.layers.layer2_pressure import generate_pressure_curve_from_segments

    if not segments:
        return np.full(n_points, 0.0, dtype=float)

    seg_dicts = []
    prev_end = None
    for i, seg in enumerate(segments):
        # Chain from previous end pressure, mirroring backend/routers/timeseries.py behavior
        start_p = float(seg.start_pressure_pa if i == 0 else (prev_end if prev_end is not None else seg.start_pressure_pa))
        end_p   = float(seg.end_pressure_pa)
        seg_dicts.append({
            "length_ratio": float(seg.length_ratio),
            "type": seg.type,
            "start_pressure": start_p,
            "end_pressure": end_p,
            "k": float(seg.k) if getattr(seg, "k", None) is not None else 0.5,
        })
        prev_end = end_p

    return np.asarray(generate_pressure_curve_from_segments(seg_dicts, n_points=n_points), dtype=float)


# ---------------------------------------------------------------------------
# /run_timeseries endpoint
# ---------------------------------------------------------------------------

@router.post("/run_timeseries")
async def run_timeseries(request: RunTimeseriesRequest):
    """Run time-series engine simulation with per-step Cd from water-test data."""
    if not app_state.has_config():
        raise HTTPException(status_code=400, detail="No engine config loaded.")

    from engine.core.runner import PintleEngineRunner

    # ---- extract (ΔP, Cd) characterisation tables ----
    fuel_pairs = _extract_cd_pairs(request.fuel)
    lox_pairs  = _extract_cd_pairs(request.lox)
    if not fuel_pairs:
        raise HTTPException(status_code=400, detail="Fuel table: not enough valid rows to compute Cd.")
    if not lox_pairs:
        raise HTTPException(status_code=400, detail="LOX table: not enough valid rows to compute Cd.")

    # ---- get pressure curves from config (hotfire-representative) ----
    cfg_active = app_state.config
    pcfg = getattr(cfg_active, "pressure_curves", None)
    if pcfg is None:
        raise HTTPException(
            status_code=400,
            detail="Active config has no 'pressure_curves' section. Upload a config that includes pressure_curves to run experiment-mode time series at operating pressures.",
        )

    n_points = int(pcfg.n_points)
    if n_points <= 1:
        raise HTTPException(status_code=400, detail="pressure_curves.n_points must be > 1.")

    duration_s = float(pcfg.target_burn_time_s)
    if duration_s <= 0:
        raise HTTPException(status_code=400, detail="pressure_curves.target_burn_time_s must be > 0.")

    times = np.linspace(0.0, duration_s, n_points, dtype=float)
    P_lox_curve_pa  = _segments_to_curve_pa(pcfg.lox_segments, n_points=n_points)
    P_fuel_curve_pa = _segments_to_curve_pa(pcfg.fuel_segments, n_points=n_points)

    # ---- deep-copy config and override injector geometry ----
    cfg = copy.deepcopy(app_state.config)

    # LOX — simple circular orifice
    cfg.injector.geometry.lox.n_orifices = 1
    cfg.injector.geometry.lox.d_orifice  = request.lox.choke_diameter_m

    # Fuel — annular gap: A = pi*h*(2R + h), solve for h
    R_tip        = cfg.injector.geometry.fuel.d_pintle_tip / 2
    A_fuel_tgt   = math.pi * (request.fuel.choke_diameter_m / 2) ** 2
    h_gap        = -R_tip + math.sqrt(R_tip ** 2 + A_fuel_tgt / math.pi)
    cfg.injector.geometry.fuel.h_gap = max(h_gap, 1e-6)

    # ---- fit/extrapolate Cd(ΔP) = a*sqrt(ΔP) + b ----
    a_fuel, b_fuel = _fit_cd_sqrt_dp(fuel_pairs)
    a_lox,  b_lox  = _fit_cd_sqrt_dp(lox_pairs)

    # ---- install fit coefficients on discharge configs (done once, not per-step) ----
    # The injector solver will re-evaluate Cd = a·√(ΔP_inj) + b at each Pc iteration
    # using the *actual* injector ΔP = P_inj − Pc, fixing the previous bug where Cd
    # was evaluated at the full tank pressure (equivalent to cold-flow ΔP ≈ 0 back-pressure).
    cfg.discharge["oxidizer"].cd_dp_fit_a = a_lox
    cfg.discharge["oxidizer"].cd_dp_fit_b = b_lox
    cfg.discharge["oxidizer"].a_Re        = 0.0
    cfg.discharge["fuel"].cd_dp_fit_a     = a_fuel
    cfg.discharge["fuel"].cd_dp_fit_b     = b_fuel
    cfg.discharge["fuel"].a_Re            = 0.0

    # ---- build runner ----
    runner = PintleEngineRunner(cfg)

    # ---- per-step loop ----
    keys = ["Pc", "mdot_O", "mdot_F", "F", "Isp", "MR", "Cd_O", "Cd_F",
            "delta_p_injector_O", "delta_p_injector_F"]
    out  = {k: np.zeros(len(times)) for k in keys}
    failures: List[str] = []

    for i in range(len(times)):
        P_O = float(P_lox_curve_pa[i])
        P_F = float(P_fuel_curve_pa[i])

        try:
            pt = runner.evaluate(P_O, P_F, debug=False, silent=True)
            for k in ["Pc", "mdot_O", "mdot_F", "F", "Isp", "MR"]:
                out[k][i] = pt.get(k, 0.0)
            out["Cd_O"][i] = pt.get("Cd_O", 0.0)
            out["Cd_F"][i] = pt.get("Cd_F", 0.0)
            inj = pt.get("injector_pressure", {}) or {}
            out["delta_p_injector_O"][i] = inj.get("delta_p_injector_O") or 0.0
            out["delta_p_injector_F"][i] = inj.get("delta_p_injector_F") or 0.0
        except Exception as e:
            msg = f"step {i} failed (P_O={P_O:.0f} Pa, P_F={P_F:.0f} Pa): {e}"
            failures.append(msg)
            print(f"[run_timeseries] {msg}")

    if failures and len(failures) == len(times):
        raise HTTPException(status_code=500, detail=f"Time-series failed for every step. First error: {failures[0]}")

    return {
        "status":  "success",
        "t":       times.tolist(),
        "results": {k: v.tolist() for k, v in out.items()},
        "fuel_cd_pressure_pairs": fuel_pairs,
        "lox_cd_pressure_pairs":  lox_pairs,
        "pressure_curves_used": {
            "P_tank_O_pa": P_lox_curve_pa.tolist(),
            "P_tank_F_pa": P_fuel_curve_pa.tolist(),
        },
        "cd_fit": {
            "fuel": {"model": "Cd = a*sqrt(dP_pa) + b", "a": a_fuel, "b": b_fuel},
            "lox":  {"model": "Cd = a*sqrt(dP_pa) + b", "a": a_lox,  "b": b_lox},
        },
    }
