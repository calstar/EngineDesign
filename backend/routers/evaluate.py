"""Engine evaluation endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np

from backend.state import app_state

router = APIRouter(prefix="/api/evaluate", tags=["evaluate"])


# Constants
PSI_TO_PA = 6894.76
PA_TO_PSI = 1.0 / PSI_TO_PA


class EvaluateRequest(BaseModel):
    """Request body for forward evaluation."""
    lox_pressure_psi: float = Field(..., gt=0, description="LOX tank pressure in psi")
    fuel_pressure_psi: float = Field(..., gt=0, description="Fuel tank pressure in psi")
    ambient_pressure_pa: float = Field(default=101325.0, gt=0, description="Ambient pressure in Pa")


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


@router.post("")
async def evaluate(request: EvaluateRequest):
    """Run forward evaluation: tank pressures -> performance.
    
    Takes LOX and fuel tank pressures in psi, returns thrust, Isp, Pc, etc.
    """
    if not app_state.has_config():
        raise HTTPException(
            status_code=400, 
            detail="No config loaded. Upload a config file first."
        )
    
    # Convert psi to Pa
    P_tank_O = request.lox_pressure_psi * PSI_TO_PA
    P_tank_F = request.fuel_pressure_psi * PSI_TO_PA
    
    try:
        results = app_state.runner.evaluate(
            P_tank_O, 
            P_tank_F, 
            P_ambient=request.ambient_pressure_pa
        )
        
        # Extract key metrics for the response
        response = {
            "status": "success",
            "inputs": {
                "lox_pressure_psi": request.lox_pressure_psi,
                "fuel_pressure_psi": request.fuel_pressure_psi,
                "ambient_pressure_pa": request.ambient_pressure_pa,
            },
            "performance": {
                "thrust_N": results["F"],
                "thrust_kN": results["F"] / 1000.0,
                "Isp_s": results["Isp"],
                "chamber_pressure_Pa": results["Pc"],
                "chamber_pressure_psi": results["Pc"] * PA_TO_PSI,
                "mdot_total_kg_s": results["mdot_total"],
                "mdot_oxidizer_kg_s": results["mdot_O"],
                "mdot_fuel_kg_s": results["mdot_F"],
                "mixture_ratio": results["MR"],
                "cstar_actual_m_s": results["cstar_actual"],
                "exit_velocity_m_s": results["v_exit"],
                "exit_pressure_Pa": results["P_exit"],
            },
            # Include full results for advanced use (converted from numpy)
            "full_results": convert_numpy(results),
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Evaluation failed: {str(e)}"
        )

