"""Closure logic: solve branch flows with spray constraints"""

from typing import Tuple, Dict, Any
from pintle_pipeline.config_schemas import PintleEngineConfig
from pintle_models.injectors import get_injector_model


def flows(
    P_tank_O: float,
    P_tank_F: float,
    Pc: float,
    config: PintleEngineConfig
) -> Tuple[float, float, Dict[str, Any]]:
    injector = get_injector_model(config)
    return injector.solve(P_tank_O, P_tank_F, Pc)



