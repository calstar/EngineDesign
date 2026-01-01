"""Optimizer endpoints for design optimization and Layer 1."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import asyncio
import json
import traceback
import numpy as np
import threading
import math
import yaml

from backend.state import app_state
from backend.routers.config import config_to_dict
from engine.pipeline.config_schemas import DesignRequirementsConfig
from engine.optimizer.layers.layer1_static_optimization import run_layer1_optimization


def convert_numpy(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        # Convert numpy scalar to Python scalar, then sanitize non-finite floats
        val = obj.item()
        if isinstance(val, float) and not math.isfinite(val):
            return None
        return val
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, float):
        # JSON forbids NaN/Infinity; scrub them here so JSON.parse never fails
        return obj if math.isfinite(obj) else None
    elif isinstance(obj, (int, str, type(None))):
        return obj
    else:
        # Try to convert to string for unknown types
        try:
            return str(obj)
        except:
            return None


def safe_json_dumps(payload: Any) -> str:
    """Strict JSON serialization for SSE: converts numpy + strips NaN/Inf."""
    sanitized = convert_numpy(payload)
    # allow_nan=False ensures we never emit invalid JSON tokens like NaN
    return json.dumps(sanitized, allow_nan=False)

router = APIRouter(prefix="/api/optimizer", tags=["optimizer"])


# Request/Response models
class DesignRequirementsRequest(BaseModel):
    """Request body for saving design requirements."""
    requirements: Dict[str, Any] = Field(..., description="Design requirements dictionary")


class DesignRequirementsResponse(BaseModel):
    """Response for design requirements."""
    requirements: Optional[Dict[str, Any]] = Field(None, description="Design requirements dictionary")


class Layer1Request(BaseModel):
    """Request body for Layer 1 optimization."""
    thrust_tolerance: float = Field(default=0.1, ge=0.01, le=0.2, description="Thrust tolerance (0.1 = 10%)")
    target_burn_time: Optional[float] = Field(default=None, gt=0, description="Target burn time [s] (from design requirements if None)")


# Global state for optimization status
_optimization_status = {
    "running": False,
    "progress": 0.0,
    "stage": "",
    "message": "",
    "results": None,
    "error": None,
}


@router.post("/design-requirements")
async def save_design_requirements(request: DesignRequirementsRequest):
    """Save design requirements to config."""
    if not app_state.has_config():
        raise HTTPException(
            status_code=400,
            detail="No config loaded. Upload a config file first."
        )
    
    try:
        # Validate requirements using Pydantic
        requirements = DesignRequirementsConfig(**request.requirements)
        
        # Update config with validated requirements
        app_state.config.design_requirements = requirements
        
        return {
            "status": "success",
            "message": "Design requirements saved successfully",
            "requirements": requirements.model_dump()
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to save design requirements: {str(e)}"
        )


@router.get("/design-requirements", response_model=DesignRequirementsResponse)
async def get_design_requirements():
    """Get current design requirements from config."""
    if not app_state.has_config():
        return DesignRequirementsResponse(requirements=None)
    
    if app_state.config.design_requirements is None:
        return DesignRequirementsResponse(requirements=None)
    
    return DesignRequirementsResponse(
        requirements=app_state.config.design_requirements.model_dump()
    )


@router.get("/layer1/status")
async def get_layer1_status():
    """Get Layer 1 optimization status."""
    return {
        "running": _optimization_status["running"],
        "progress": _optimization_status["progress"],
        "stage": _optimization_status["stage"],
        "message": _optimization_status["message"],
        "has_results": _optimization_status["results"] is not None,
        "error": _optimization_status["error"],
    }


@router.get("/layer1/results")
async def get_layer1_results():
    """Get Layer 1 optimization results."""
    if _optimization_status["results"] is None:
        raise HTTPException(
            status_code=404,
            detail="No optimization results available. Run Layer 1 optimization first."
        )
    
    return {
        "status": "success",
        "results": _optimization_status["results"],
    }


@router.get("/layer1")
async def run_layer1(
    thrust_tolerance: float = 0.1,
    target_burn_time: float | None = None
):
    """Run Layer 1 optimization with Server-Sent Events for progress updates.
    
    Note: max_iterations is hardcoded in the optimizer for consistent robust convergence.
    
    Returns a stream of progress updates in SSE format.
    """
    if not app_state.has_config():
        raise HTTPException(
            status_code=400,
            detail="No config loaded. Upload a config file first."
        )
    
    if not app_state.runner:
        raise HTTPException(
            status_code=400,
            detail="Runner not initialized. Please check config."
        )
    
    # Check for design requirements
    if app_state.config.design_requirements is None:
        raise HTTPException(
            status_code=400,
            detail="No design requirements set. Save design requirements first."
        )
    
    # Check if already running
    if _optimization_status["running"]:
        raise HTTPException(
            status_code=409,
            detail="Optimization already running. Please wait for it to complete."
        )
    
    async def event_generator():
        """Generate SSE events for optimization progress."""
        global _optimization_status
        
        _optimization_status["running"] = True
        _optimization_status["progress"] = 0.0
        _optimization_status["stage"] = "Initializing"
        _optimization_status["message"] = "Starting Layer 1 optimization..."
        _optimization_status["results"] = None
        _optimization_status["error"] = None
        
        # Send initial status
        yield f"data: {safe_json_dumps({'type': 'status', 'progress': 0.0, 'stage': 'Initializing', 'message': 'Starting optimization...'})}\n\n"
        
        try:
            # Get design requirements
            requirements = app_state.config.design_requirements.model_dump()
            burn_time = target_burn_time or requirements.get("target_burn_time", 10.0)
            
            # Prepare pressure config
            pressure_config = {
                "mode": "optimizer_controlled",
                "max_lox_pressure_psi": requirements.get("max_lox_tank_pressure_psi", 700.0),
                "max_fuel_pressure_psi": requirements.get("max_fuel_tank_pressure_psi", 850.0),
                "target_burn_time": burn_time,
                "n_segments": 3,
            }
            
            # Prepare tolerances
            tolerances = {
                "thrust": thrust_tolerance,
                "apogee": 0.15,
            }
            
            # Objective history - use thread-safe list
            objective_history = []
            objective_history_lock = threading.Lock()
            last_sent_objective_count = 0
            
            # Progress callback
            def update_progress(stage: str, progress: float, message: str):
                _optimization_status["progress"] = progress
                _optimization_status["stage"] = stage
                _optimization_status["message"] = message
            
            # Objective callback - thread-safe
            def objective_callback(iteration: int, objective: float, best_objective: float):
                with objective_history_lock:
                    objective_history.append({
                        "iteration": int(iteration),
                        "objective": float(objective),
                        "best_objective": float(best_objective),
                    })
            
            # Run optimization (blocking)
            # Note: This will block the event loop, but for now we'll keep it simple
            # In production, you'd want to run this in a thread pool
            import concurrent.futures
            
            def run_optimization():
                return run_layer1_optimization(
                    config_obj=app_state.config,
                    runner=app_state.runner,
                    requirements=requirements,
                    target_burn_time=burn_time,
                    tolerances=tolerances,
                    pressure_config=pressure_config,
                    update_progress=update_progress,
                    log_status=lambda stage, msg: None,  # Not used for SSE
                    objective_callback=objective_callback,
                )
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                # Send progress updates while optimization runs
                future = loop.run_in_executor(pool, run_optimization)
                
                while not future.done():
                    # Send progress update (convert numpy types)
                    progress_data = convert_numpy({
                        'type': 'progress', 
                        'progress': _optimization_status['progress'], 
                        'stage': _optimization_status['stage'], 
                        'message': _optimization_status['message']
                    })
                    yield f"data: {safe_json_dumps(progress_data)}\n\n"
                    
                    # Check for new objective history updates and send them
                    with objective_history_lock:
                        if len(objective_history) > last_sent_objective_count:
                            # Get new entries
                            new_entries = objective_history[last_sent_objective_count:]
                            last_sent_objective_count = len(objective_history)
                            
                            # Send objective update event
                            objective_data = convert_numpy({
                                'type': 'objective',
                                'objective_history': new_entries,
                                'total_count': last_sent_objective_count,
                            })
                            yield f"data: {safe_json_dumps(objective_data)}\n\n"
                    
                    await asyncio.sleep(0.5)
                
                # Get results
                optimized_config, results = future.result()
            
            # Update config
            app_state.config = optimized_config
            
            # Store results (convert numpy types for JSON serialization)
            results_dict = convert_numpy({
                "performance": results.get("performance", {}),
                "validation": results.get("validation", {}),
                "geometry": results.get("optimized_parameters", {}),
                "objective_history": objective_history,
                "iteration_history": results.get("iteration_history", []),
                "config": config_to_dict(optimized_config),
                "config_yaml": yaml.dump(config_to_dict(optimized_config), default_flow_style=False),
            })
            _optimization_status["results"] = results_dict
            _optimization_status["progress"] = 1.0
            _optimization_status["stage"] = "Complete"
            _optimization_status["message"] = "Optimization completed successfully"
            
            # Send completion event
            yield f"data: {safe_json_dumps({'type': 'complete', 'results': results_dict})}\n\n"
            
        except Exception as e:
            error_msg = f"Optimization failed: {str(e)}"
            error_trace = traceback.format_exc()
            _optimization_status["error"] = error_msg
            _optimization_status["message"] = error_msg
            
            # Send error event
            yield f"data: {safe_json_dumps({'type': 'error', 'error': error_msg, 'traceback': error_trace})}\n\n"
        
        finally:
            _optimization_status["running"] = False
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )

