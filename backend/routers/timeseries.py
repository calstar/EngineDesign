"""Time-series evaluation endpoints."""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import numpy as np
import pandas as pd

from backend.state import app_state
from engine.pipeline.time_series import generate_pressure_profile
from engine.optimizer.layers.layer2_pressure import generate_pressure_curve_from_segments
from copv.copv_solve_both import size_or_check_copv_for_polytropic_N2

# Get path to N2 Z lookup table (relative to project root)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
N2_Z_LOOKUP_CSV = str(_PROJECT_ROOT / "copv" / "n2_Z_lookup.csv")

router = APIRouter(prefix="/api/timeseries", tags=["timeseries"])


# Constants
PSI_TO_PA = 6894.76
PA_TO_PSI = 1.0 / PSI_TO_PA


def convert_numpy(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def compute_timeseries_results(
    runner,
    times: np.ndarray,
    P_tank_O_psi: np.ndarray,
    P_tank_F_psi: np.ndarray,
):
    """Compute time-series results from pressure profiles.
    
    This is a standalone version that doesn't depend on Streamlit.
    """
    P_tank_O_pa = np.asarray(P_tank_O_psi) * PSI_TO_PA
    P_tank_F_pa = np.asarray(P_tank_F_psi) * PSI_TO_PA

    # Check if ablative geometry tracking is enabled
    ablative_cfg = runner.config.ablative_cooling
    use_time_varying = (
        ablative_cfg is not None 
        and ablative_cfg.enabled 
        and getattr(ablative_cfg, 'track_geometry_evolution', False)
        and len(times) >= 2
    )
    
    if use_time_varying:
        results = runner.evaluate_arrays_with_time(
            times, 
            P_tank_O_pa, 
            P_tank_F_pa,
            use_coupled_solver=True,
        )
    else:
        results = runner.evaluate_arrays(P_tank_O_pa, P_tank_F_pa)

    # Build results dict
    result_data = {
        "time": np.asarray(times).tolist(),
        "P_tank_O_psi": np.asarray(P_tank_O_psi, dtype=float).tolist(),
        "P_tank_F_psi": np.asarray(P_tank_F_psi, dtype=float).tolist(),
        "Pc_psi": (np.asarray(results["Pc"], dtype=float) * PA_TO_PSI).tolist(),
        "thrust_kN": (np.asarray(results["F"], dtype=float) / 1000.0).tolist(),
        "Isp_s": np.asarray(results["Isp"], dtype=float).tolist(),
        "MR": np.asarray(results["MR"], dtype=float).tolist(),
        "mdot_O_kg_s": np.asarray(results["mdot_O"], dtype=float).tolist(),
        "mdot_F_kg_s": np.asarray(results["mdot_F"], dtype=float).tolist(),
        "mdot_total_kg_s": np.asarray(results["mdot_total"], dtype=float).tolist(),
        "cstar_actual_m_s": np.asarray(results["cstar_actual"], dtype=float).tolist(),
        "gamma": np.asarray(results["gamma"], dtype=float).tolist(),
    }
    
    # Add optional fields if available
    if "Cd_O" in results:
        result_data["Cd_O"] = np.asarray(results["Cd_O"], dtype=float).tolist()
    if "Cd_F" in results:
        result_data["Cd_F"] = np.asarray(results["Cd_F"], dtype=float).tolist()
    
    # Extract diagnostics data (matching Streamlit pattern)
    diagnostics_list = results.get("diagnostics", [])
    n_points = len(times)
    
    # Initialize arrays for optional data
    delta_P_injector_O_psi = []
    delta_P_injector_F_psi = []
    recession_rate_ablative_um_s = []
    recession_rate_graphite_thermal_um_s = []
    recession_rate_graphite_oxidation_um_s = []
    
    # Extract data from diagnostics (matching Streamlit pattern)
    for i, diag in enumerate(diagnostics_list):
        if not isinstance(diag, dict):
            delta_P_injector_O_psi.append(np.nan)
            delta_P_injector_F_psi.append(np.nan)
            recession_rate_ablative_um_s.append(np.nan)
            recession_rate_graphite_thermal_um_s.append(np.nan)
            recession_rate_graphite_oxidation_um_s.append(np.nan)
            continue
            
        # Injector pressure drops (check both locations like Streamlit)
        injector_pressure = diag.get("injector_pressure", {})
        if injector_pressure and isinstance(injector_pressure, dict):
            delta_P_O = injector_pressure.get("delta_P_injector_O")
            delta_P_F = injector_pressure.get("delta_P_injector_F")
        else:
            # Fallback to direct access
            delta_P_O = diag.get("delta_P_injector_O")
            delta_P_F = diag.get("delta_P_injector_F")
        
        if delta_P_O is not None:
            delta_P_injector_O_psi.append(float(delta_P_O * PA_TO_PSI))
        else:
            delta_P_injector_O_psi.append(np.nan)
            
        if delta_P_F is not None:
            delta_P_injector_F_psi.append(float(delta_P_F * PA_TO_PSI))
        else:
            delta_P_injector_F_psi.append(np.nan)
        
        # Cooling diagnostics - recession rates (matching Streamlit pattern)
        cooling = diag.get("cooling", {})
        ablative = cooling.get("ablative", {}) if cooling else {}
        graphite = cooling.get("graphite", {}) if cooling else {}
        
        # Ablative recession rate
        if ablative and isinstance(ablative, dict):
            recession_rate = ablative.get("recession_rate")
            if recession_rate is not None:
                recession_rate_ablative_um_s.append(float(recession_rate * 1e6))  # Convert m/s to µm/s
            else:
                recession_rate_ablative_um_s.append(np.nan)
        else:
            recession_rate_ablative_um_s.append(np.nan)
        
        # Graphite recession rates
        if graphite and isinstance(graphite, dict) and graphite.get("enabled", False):
            thermal_rate = graphite.get("recession_rate_thermal")
            oxidation_rate = graphite.get("oxidation_rate")
            if thermal_rate is not None:
                recession_rate_graphite_thermal_um_s.append(float(thermal_rate * 1e6))  # Convert m/s to µm/s
            else:
                recession_rate_graphite_thermal_um_s.append(np.nan)
            if oxidation_rate is not None:
                recession_rate_graphite_oxidation_um_s.append(float(oxidation_rate * 1e6))  # Convert m/s to µm/s
            else:
                recession_rate_graphite_oxidation_um_s.append(np.nan)
        else:
            recession_rate_graphite_thermal_um_s.append(np.nan)
            recession_rate_graphite_oxidation_um_s.append(np.nan)
    
    # Extract geometry evolution data (from evaluate_arrays_with_time results)
    Lstar_mm = None
    V_chamber_m3 = None
    A_throat_m2 = None
    recession_cumulative_chamber_um = None
    recession_cumulative_throat_um = None
    
    if "Lstar" in results:
        Lstar_array = np.asarray(results["Lstar"], dtype=float)
        if len(Lstar_array) == n_points:
            Lstar_mm = (Lstar_array * 1000.0).tolist()  # Convert m to mm
    
    if "V_chamber" in results:
        V_chamber_array = np.asarray(results["V_chamber"], dtype=float)
        if len(V_chamber_array) == n_points:
            V_chamber_m3 = V_chamber_array.tolist()
    
    if "A_throat" in results:
        A_throat_array = np.asarray(results["A_throat"], dtype=float)
        if len(A_throat_array) == n_points:
            A_throat_m2 = A_throat_array.tolist()
    
    if "recession_chamber" in results:
        recession_array = np.asarray(results["recession_chamber"], dtype=float)
        if len(recession_array) == n_points:
            recession_cumulative_chamber_um = (recession_array * 1e6).tolist()  # Convert m to µm
    
    if "recession_throat" in results:
        recession_array = np.asarray(results["recession_throat"], dtype=float)
        if len(recession_array) == n_points:
            recession_cumulative_throat_um = (recession_array * 1e6).tolist()  # Convert m to µm
    
    # Calculate cumulative recession from rates (matching Streamlit pattern)
    recession_cumulative_ablative_mm = None
    recession_cumulative_graphite_thermal_mm = None
    recession_cumulative_graphite_oxidation_mm = None
    
    if any(np.isfinite(recession_rate_ablative_um_s)):
        times_array = np.asarray(times)
        dt = np.diff(times_array, prepend=times_array[0] if len(times_array) > 0 else 0.0)
        recession_rate_m_s = np.asarray(recession_rate_ablative_um_s, dtype=float) * 1e-6  # Convert µm/s to m/s
        recession_rate_m_s = np.nan_to_num(recession_rate_m_s, nan=0.0)
        cumulative = np.cumsum(recession_rate_m_s * dt) * 1000.0  # Convert to mm
        recession_cumulative_ablative_mm = cumulative.tolist()
    
    if any(np.isfinite(recession_rate_graphite_thermal_um_s)):
        times_array = np.asarray(times)
        dt = np.diff(times_array, prepend=times_array[0] if len(times_array) > 0 else 0.0)
        recession_rate_m_s = np.asarray(recession_rate_graphite_thermal_um_s, dtype=float) * 1e-6
        recession_rate_m_s = np.nan_to_num(recession_rate_m_s, nan=0.0)
        cumulative = np.cumsum(recession_rate_m_s * dt) * 1000.0
        recession_cumulative_graphite_thermal_mm = cumulative.tolist()
    
    if any(np.isfinite(recession_rate_graphite_oxidation_um_s)):
        times_array = np.asarray(times)
        dt = np.diff(times_array, prepend=times_array[0] if len(times_array) > 0 else 0.0)
        recession_rate_m_s = np.asarray(recession_rate_graphite_oxidation_um_s, dtype=float) * 1e-6
        recession_rate_m_s = np.nan_to_num(recession_rate_m_s, nan=0.0)
        cumulative = np.cumsum(recession_rate_m_s * dt) * 1000.0
        recession_cumulative_graphite_oxidation_mm = cumulative.tolist()
    
    # Add optional data fields (only if data exists)
    if delta_P_injector_O_psi and any(np.isfinite(delta_P_injector_O_psi)):
        result_data["delta_P_injector_O_psi"] = delta_P_injector_O_psi
    if delta_P_injector_F_psi and any(np.isfinite(delta_P_injector_F_psi)):
        result_data["delta_P_injector_F_psi"] = delta_P_injector_F_psi
    
    if Lstar_mm and any(np.isfinite(Lstar_mm)):
        result_data["Lstar_mm"] = Lstar_mm
    
    if recession_rate_ablative_um_s and any(np.isfinite(recession_rate_ablative_um_s)):
        result_data["recession_rate_ablative_um_s"] = recession_rate_ablative_um_s
    if recession_rate_graphite_thermal_um_s and any(np.isfinite(recession_rate_graphite_thermal_um_s)):
        result_data["recession_rate_graphite_thermal_um_s"] = recession_rate_graphite_thermal_um_s
    if recession_rate_graphite_oxidation_um_s and any(np.isfinite(recession_rate_graphite_oxidation_um_s)):
        result_data["recession_rate_graphite_oxidation_um_s"] = recession_rate_graphite_oxidation_um_s
    
    if recession_cumulative_ablative_mm and any(np.isfinite(recession_cumulative_ablative_mm)):
        result_data["recession_cumulative_ablative_mm"] = recession_cumulative_ablative_mm
    if recession_cumulative_graphite_thermal_mm and any(np.isfinite(recession_cumulative_graphite_thermal_mm)):
        result_data["recession_cumulative_graphite_thermal_mm"] = recession_cumulative_graphite_thermal_mm
    if recession_cumulative_graphite_oxidation_mm and any(np.isfinite(recession_cumulative_graphite_oxidation_mm)):
        result_data["recession_cumulative_graphite_oxidation_mm"] = recession_cumulative_graphite_oxidation_mm
    
    if recession_cumulative_chamber_um and any(np.isfinite(recession_cumulative_chamber_um)):
        result_data["recession_cumulative_chamber_um"] = recession_cumulative_chamber_um
    if recession_cumulative_throat_um and any(np.isfinite(recession_cumulative_throat_um)):
        result_data["recession_cumulative_throat_um"] = recession_cumulative_throat_um
    
    if V_chamber_m3 and any(np.isfinite(V_chamber_m3)) and len(V_chamber_m3) > 0:
        result_data["V_chamber_m3"] = V_chamber_m3
        result_data["V_chamber_initial_m3"] = float(V_chamber_m3[0])
    if A_throat_m2 and any(np.isfinite(A_throat_m2)) and len(A_throat_m2) > 0:
        result_data["A_throat_m2"] = A_throat_m2
        result_data["A_throat_initial_m2"] = float(A_throat_m2[0])
    
    # Calculate summary statistics
    thrust_arr = np.asarray(results["F"], dtype=float) / 1000.0
    Pc_arr = np.asarray(results["Pc"], dtype=float) * PA_TO_PSI
    Isp_arr = np.asarray(results["Isp"], dtype=float)
    mdot_arr = np.asarray(results["mdot_total"], dtype=float)
    
    summary = {
        "avg_thrust_kN": float(np.nanmean(thrust_arr)),
        "peak_thrust_kN": float(np.nanmax(thrust_arr)),
        "min_thrust_kN": float(np.nanmin(thrust_arr)),
        "avg_Pc_psi": float(np.nanmean(Pc_arr)),
        "peak_Pc_psi": float(np.nanmax(Pc_arr)),
        "avg_Isp_s": float(np.nanmean(Isp_arr)),
        "total_impulse_kNs": float(np.trapz(thrust_arr, times)),
        "total_propellant_kg": float(np.trapz(mdot_arr, times)),
        "burn_time_s": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
    }
    
    # =========================================================================
    # COPV Sizing Analysis - Always run if press_tank config is available
    # =========================================================================
    copv_results = _run_copv_analysis(
        runner.config,
        times,
        result_data["mdot_O_kg_s"],
        result_data["mdot_F_kg_s"],
        P_tank_O_psi,
        P_tank_F_psi,
    )
    
    if copv_results:
        # Add COPV pressure trace to result data
        result_data["copv_pressure_psi"] = copv_results["copv_pressure_psi"]
        
        # Add COPV summary metrics
        summary["copv_initial_pressure_psi"] = copv_results["copv_initial_pressure_psi"]
        summary["copv_initial_mass_kg"] = copv_results["copv_initial_mass_kg"]
        summary["copv_min_margin_psi"] = copv_results["copv_min_margin_psi"]
        summary["copv_volume_L"] = copv_results["copv_volume_L"]
    
    # =========================================================================
    # Correlation Matrix - Compute correlations between key variables
    # =========================================================================
    correlation_data = _compute_correlation_matrix(result_data)
    if correlation_data:
        result_data["correlation_matrix"] = correlation_data["matrix"]
        result_data["correlation_labels"] = correlation_data["labels"]
    
    return result_data, summary


def _compute_correlation_matrix(result_data: dict) -> Optional[dict]:
    """Compute correlation matrix for key time-series variables.
    
    Returns dict with 'matrix' (2D list) and 'labels' (list of variable names).
    """
    # Define variables to include in correlation analysis
    # Map from result_data keys to display labels
    variable_map = {
        "P_tank_O_psi": "LOX Tank P",
        "P_tank_F_psi": "Fuel Tank P",
        "Pc_psi": "Chamber P",
        "thrust_kN": "Thrust",
        "Isp_s": "Isp",
        "MR": "O/F Ratio",
        "mdot_O_kg_s": "LOX mdot",
        "mdot_F_kg_s": "Fuel mdot",
        "mdot_total_kg_s": "Total mdot",
        "cstar_actual_m_s": "c*",
        "gamma": "Gamma",
        "copv_pressure_psi": "COPV P",
    }
    
    # Build dataframe with available variables
    df_data = {}
    labels = []
    
    for key, label in variable_map.items():
        if key in result_data and result_data[key] is not None:
            arr = np.asarray(result_data[key], dtype=float)
            if len(arr) > 0 and np.any(np.isfinite(arr)):
                df_data[label] = arr
                labels.append(label)
    
    if len(labels) < 2:
        return None
    
    # Create dataframe and compute correlation
    df = pd.DataFrame(df_data)
    corr = df.corr().fillna(0.0)
    
    # Convert to list format for JSON serialization
    matrix = corr.values.tolist()
    
    return {
        "matrix": matrix,
        "labels": labels,
    }


def _run_copv_analysis(
    config,
    times: np.ndarray,
    mdot_O_kg_s: list,
    mdot_F_kg_s: list,
    P_tank_O_psi: np.ndarray,
    P_tank_F_psi: np.ndarray,
) -> Optional[dict]:
    """Run COPV blowdown analysis using the polytropic N2 solver.
    
    Uses hardcoded defaults per requirements:
    - n = 1.2 (polytropic exponent)
    - T0_K = 300 (initial COPV temp)
    - R = 296.8 (gas constant for N2)
    - Oxidizer gas temp: 250K
    - Fuel gas temp: 293K
    - use_real_gas = True (use Z lookup table)
    
    Returns dict with copv_pressure_psi trace and summary metrics, or None if unavailable.
    """
    # Check if press_tank config is available
    press_tank = getattr(config, 'press_tank', None)
    if press_tank is None:
        return None
    
    # Get COPV free volume from config (default 4.5L)
    free_volume_L = getattr(press_tank, 'free_volume_L', None) or 4.5
    copv_volume_m3 = free_volume_L / 1000.0  # Convert L to m³
    
    # Build dataframe for COPV solver
    df = pd.DataFrame({
        "time": np.asarray(times),
        "mdot_O (kg/s)": np.asarray(mdot_O_kg_s),
        "mdot_F (kg/s)": np.asarray(mdot_F_kg_s),
        "P_tank_O (psi)": np.asarray(P_tank_O_psi),
        "P_tank_F (psi)": np.asarray(P_tank_F_psi),
    })
    
    try:
        # Run COPV solver with hardcoded parameters
        copv_result = size_or_check_copv_for_polytropic_N2(
            df=df,
            config=config,
            n=1.2,  # polytropic exponent
            T0_K=300.0,  # initial COPV temperature
            Tp_K=293.0,  # default propellant gas temp (overridden by branch temps)
            use_real_gas=True,  # use Z lookup table
            n2_Z_csv=N2_Z_LOOKUP_CSV,
            pressurant_R=296.8,  # gas constant for N2
            branch_temperatures_K={
                "oxidizer": 250.0,  # oxidizer gas temp
                "fuel": 293.0,      # fuel gas temp
            },
            copv_volume_m3=copv_volume_m3,
        )
        
        # Extract COPV pressure trace (Pa -> psi)
        PH_trace_Pa = np.asarray(copv_result.get("PH_trace_Pa", []))
        if len(PH_trace_Pa) == 0:
            return None
        
        copv_pressure_psi = (PH_trace_Pa * PA_TO_PSI).tolist()
        
        return {
            "copv_pressure_psi": copv_pressure_psi,
            "copv_initial_pressure_psi": float(copv_result.get("P0_Pa", 0) * PA_TO_PSI),
            "copv_initial_mass_kg": float(copv_result.get("m0_kg", 0)),
            "copv_min_margin_psi": float(copv_result.get("min_margin_Pa", 0) * PA_TO_PSI),
            "copv_volume_L": free_volume_L,
        }
        
    except Exception as e:
        # Log error but don't fail the whole request
        import logging
        logging.warning(f"COPV analysis failed: {e}")
        return None


# Request models for simple profile generation
class ProfileParams(BaseModel):
    """Parameters for a single propellant profile."""
    start_pressure_psi: float = Field(..., gt=0, description="Start pressure in psi")
    end_pressure_psi: float = Field(..., gt=0, description="End pressure in psi")
    profile_type: Literal["linear", "exponential", "power"] = Field(default="linear")
    decay_constant: Optional[float] = Field(default=3.0, ge=0.1, le=10.0)
    power: Optional[float] = Field(default=2.0, ge=0.1, le=5.0)


class GenerateProfileRequest(BaseModel):
    """Request body for simple profile generation."""
    duration_s: float = Field(..., gt=0, le=600, description="Burn duration in seconds")
    n_steps: int = Field(default=101, ge=2, le=2000, description="Number of time steps")
    lox_profile: ProfileParams
    fuel_profile: ProfileParams


class GenerateProfileResponse(BaseModel):
    """Response for profile generation."""
    status: str
    data: dict
    summary: dict


@router.post("/generate", response_model=GenerateProfileResponse)
async def generate_timeseries(request: GenerateProfileRequest):
    """Generate time-series results from simple pressure profiles.
    
    Supports linear, exponential, and power profile types.
    """
    if not app_state.has_config():
        raise HTTPException(
            status_code=400,
            detail="No config loaded. Upload a config file first."
        )
    
    try:
        # Generate LOX profile
        lox_params = {}
        if request.lox_profile.profile_type == "exponential":
            lox_params["decay_constant"] = request.lox_profile.decay_constant
        elif request.lox_profile.profile_type == "power":
            lox_params["power"] = request.lox_profile.power
            
        times, lox_pressures = generate_pressure_profile(
            request.lox_profile.profile_type,
            request.lox_profile.start_pressure_psi,
            request.lox_profile.end_pressure_psi,
            request.duration_s,
            request.n_steps,
            **lox_params,
        )
        
        # Generate Fuel profile
        fuel_params = {}
        if request.fuel_profile.profile_type == "exponential":
            fuel_params["decay_constant"] = request.fuel_profile.decay_constant
        elif request.fuel_profile.profile_type == "power":
            fuel_params["power"] = request.fuel_profile.power
            
        _, fuel_pressures = generate_pressure_profile(
            request.fuel_profile.profile_type,
            request.fuel_profile.start_pressure_psi,
            request.fuel_profile.end_pressure_psi,
            request.duration_s,
            request.n_steps,
            **fuel_params,
        )
        
        # Compute time-series results
        data, summary = compute_timeseries_results(
            app_state.runner,
            times,
            lox_pressures,
            fuel_pressures,
        )
        
        return GenerateProfileResponse(
            status="success",
            data=convert_numpy(data),
            summary=convert_numpy(summary),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Time-series evaluation failed: {str(e)}"
        )


# Request models for segment-based profiles
class PressureSegment(BaseModel):
    """A single segment of a pressure curve."""
    length_ratio: float = Field(..., ge=0.01, le=1.0, description="Fraction of total time")
    type: Literal["blowdown", "linear"] = Field(default="blowdown")
    start_pressure_psi: float = Field(..., gt=0, description="Start pressure in psi")
    end_pressure_psi: float = Field(..., gt=0, description="End pressure in psi")
    k: float = Field(default=0.5, ge=0.1, le=3.0, description="Blowdown decay constant")


class SegmentsRequest(BaseModel):
    """Request body for segment-based profile generation."""
    duration_s: float = Field(..., gt=0, le=600, description="Burn duration in seconds")
    n_points: int = Field(default=200, ge=10, le=2000, description="Number of time points")
    lox_segments: List[PressureSegment] = Field(..., min_length=1, max_length=20)
    fuel_segments: List[PressureSegment] = Field(..., min_length=1, max_length=20)


class SegmentsResponse(BaseModel):
    """Response for segment-based profile generation."""
    status: str
    data: dict
    summary: dict
    lox_curve_preview: List[float]
    fuel_curve_preview: List[float]


def segments_to_dict_list(segments: List[PressureSegment]) -> List[dict]:
    """Convert Pydantic segments to dict list for curve generation."""
    result = []
    prev_end = None
    
    for i, seg in enumerate(segments):
        # First segment uses its start pressure; subsequent segments chain from previous end
        if i == 0:
            start_p = seg.start_pressure_psi * PSI_TO_PA
        else:
            start_p = prev_end if prev_end else seg.start_pressure_psi * PSI_TO_PA
        
        end_p = seg.end_pressure_psi * PSI_TO_PA
        
        # Ensure decreasing pressure
        if end_p > start_p:
            end_p = start_p * 0.95
        
        result.append({
            "length_ratio": seg.length_ratio,
            "type": seg.type,
            "start_pressure": start_p,
            "end_pressure": end_p,
            "k": seg.k,
        })
        prev_end = end_p
    
    return result


@router.post("/from-segments", response_model=SegmentsResponse)
async def generate_from_segments(request: SegmentsRequest):
    """Generate time-series results from segment-based pressure curves.
    
    Each segment specifies:
    - length_ratio: fraction of total time (0-1)
    - type: 'blowdown' or 'linear'
    - start_pressure_psi: start pressure (auto-chained from previous segment)
    - end_pressure_psi: end pressure
    - k: blowdown decay constant (0.1-3.0)
    
    The blowdown formula: P(t) = P_end + (P_start - P_end) * exp(-k * t_norm)
    """
    if not app_state.has_config():
        raise HTTPException(
            status_code=400,
            detail="No config loaded. Upload a config file first."
        )
    
    try:
        # Convert segments to dict format
        lox_seg_dicts = segments_to_dict_list(request.lox_segments)
        fuel_seg_dicts = segments_to_dict_list(request.fuel_segments)
        
        # Generate pressure curves from segments (Pa)
        lox_curve_pa = generate_pressure_curve_from_segments(
            lox_seg_dicts,
            n_points=request.n_points,
        )
        fuel_curve_pa = generate_pressure_curve_from_segments(
            fuel_seg_dicts,
            n_points=request.n_points,
        )
        
        # Convert to psi for the API
        lox_curve_psi = lox_curve_pa * PA_TO_PSI
        fuel_curve_psi = fuel_curve_pa * PA_TO_PSI
        
        # Generate time array
        times = np.linspace(0, request.duration_s, request.n_points)
        
        # Compute time-series results
        data, summary = compute_timeseries_results(
            app_state.runner,
            times,
            lox_curve_psi,
            fuel_curve_psi,
        )
        
        return SegmentsResponse(
            status="success",
            data=convert_numpy(data),
            summary=convert_numpy(summary),
            lox_curve_preview=lox_curve_psi.tolist(),
            fuel_curve_preview=fuel_curve_psi.tolist(),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Segment-based evaluation failed: {str(e)}"
        )


# Preview endpoint for real-time curve visualization
class PreviewSegmentsRequest(BaseModel):
    """Request for curve preview without full evaluation."""
    n_points: int = Field(default=100, ge=10, le=500)
    segments: List[PressureSegment] = Field(..., min_length=1, max_length=20)


class PreviewResponse(BaseModel):
    """Response with curve preview data."""
    curve_psi: List[float]
    normalized_time: List[float]


@router.post("/preview-curve", response_model=PreviewResponse)
async def preview_curve(request: PreviewSegmentsRequest):
    """Preview a pressure curve from segments without running full evaluation.
    
    Useful for real-time curve visualization in the UI.
    """
    try:
        seg_dicts = segments_to_dict_list(request.segments)
        curve_pa = generate_pressure_curve_from_segments(
            seg_dicts,
            n_points=request.n_points,
        )
        curve_psi = (curve_pa * PA_TO_PSI).tolist()
        normalized_time = np.linspace(0, 1, request.n_points).tolist()
        
        return PreviewResponse(
            curve_psi=curve_psi,
            normalized_time=normalized_time,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Curve preview failed: {str(e)}"
        )

