"""Views package for UI components.

This package contains:
- tabs.py: Main tab functions for the design optimization interface
- helpers.py: Helper functions for display and comparison
"""

from .tabs import (
    _design_requirements_tab,
    _full_engine_optimization_tab,
    _injector_optimization_tab,
    _chamber_optimization_tab,
    _stability_analysis_tab,
    _flight_performance_tab,
    _results_export_tab,
)

__all__ = [
    '_design_requirements_tab',
    '_full_engine_optimization_tab',
    '_injector_optimization_tab',
    '_chamber_optimization_tab',
    '_stability_analysis_tab',
    '_flight_performance_tab',
    '_results_export_tab',
]

