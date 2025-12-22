"""
Chamber geometry solver that uses CEA lookup table to find real Cf.

This solver iteratively converges on the correct throat area and thrust coefficient
given chamber pressure and thrust, using thermochemistry from the CEA cache.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import os

# Handle both relative import (when used as module) and absolute import (when run as script)
try:
    from .chamber_geometry import (
        chamber_geometry_calc,
        area_exit_calc,
        expansion_ratio_calc,
        theta_default,
        l_star_default,
        chamber_diameter_default,
        diameter_exit_default,
    )
except ImportError:
    # Fallback for when running as script
    from engine.core.chamber_geometry import (
        chamber_geometry_calc,
        area_exit_calc,
        expansion_ratio_calc,
        theta_default,
        l_star_default,
        chamber_diameter_default,
        diameter_exit_default,
    )
from engine.pipeline.cea_cache import CEACache
from engine.pipeline.config_schemas import CEAConfig

# Default CEA cache file path (relative to project root)
DEFAULT_CEA_CACHE_FILE = "output/cache/cea_cache_LOX_RP1.npz"


def solve_chamber_geometry_with_cea(
    pc_design: float,
    thrust_design: float,
    cea_cache: CEACache,
    MR: float,
    diameter_inner: float = chamber_diameter_default,
    diameter_exit: float = diameter_exit_default,
    l_star: float = l_star_default,
    Pa: float = 101325.0,
    nozzle_efficiency: float = 1.0,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
    do_plot: bool = False,
    color_segments: bool = False,
    steps: int = 200,
    export_dxf: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, list, float, Dict[str, Any]]:
    """
    Solve for chamber geometry using CEA lookup table to find corrected Cf.
    
    This function iteratively solves for the throat area that achieves the desired
    thrust using the corrected thrust coefficient (Cf) that accounts for nozzle
    efficiency losses, matching what the pipeline uses for thrust calculations.
    
    The solver finds the optimal throat area by:
    1. Guessing an initial throat area
    2. Calculating expansion ratio eps = A_exit / A_throat
    3. Looking up ideal Cf from CEA cache using (Pc, MR, eps, Pa)
       - For 3D cache: Cf is optimized for the current expansion ratio
    4. Applying nozzle efficiency correction: Cf_corrected = nozzle_efficiency * Cf_ideal
       - This matches the corrected Cf used in the pipeline (pintle_models/nozzle.py)
    5. Calculating new throat area: A_throat = F / (Cf_corrected * Pc)
    6. Iterating until convergence (throat area stabilizes)
    
    The final Cf is the corrected thrust coefficient that accounts for:
    - Ideal Cf from CEA (isentropic flow, real gas properties)
    - Nozzle efficiency losses (boundary layer, divergence, etc.)
    - This matches the Cf used in the actual thrust calculations
    
    Parameters:
    -----------
    pc_design : float
        Design chamber pressure [Pa]
    thrust_design : float
        Design thrust [N]
    cea_cache : CEACache
        CEA cache instance for thermochemistry lookup
    MR : float
        Mixture ratio (O/F) for CEA lookup
    diameter_inner : float, optional
        Inner chamber diameter [m] (default: 3.4")
    diameter_exit : float, optional
        Exit diameter [m] (default: 4")
    l_star : float, optional
        Characteristic length [m] (default: 1.27 m)
    Pa : float, optional
        Ambient pressure [Pa] (default: sea level)
    nozzle_efficiency : float, optional
        Nozzle efficiency factor (0-1) to apply to ideal Cf (default: 1.0).
        Typical values: 0.96-0.99. This accounts for boundary layer losses,
        divergence losses, and other nozzle inefficiencies.
    tolerance : float, optional
        Convergence tolerance for throat area (default: 1e-6)
    max_iterations : int, optional
        Maximum number of iterations (default: 100)
    do_plot : bool, optional
        Whether to generate plot (default: False)
    color_segments : bool, optional
        Whether to color-code segments in plot (default: False)
    steps : int, optional
        Number of points for geometry generation (default: 200)
    export_dxf : str or None, optional
        Path to export DXF file (default: None)
    verbose : bool, optional
        Whether to print convergence information (default: False)
    
    Returns:
    --------
    chamber_pts : numpy array
        Array of (x, y) points representing the full chamber contour
    table_data : list of lists
        Table data with metric and imperial units
    total_chamber_length : float
        Total chamber length [m]
    solver_info : dict
        Dictionary containing:
        - converged: bool
        - iterations: int
        - final_Cf: float (corrected Cf with efficiency applied)
        - final_Cf_ideal: float (ideal Cf from CEA before efficiency correction)
        - final_A_throat: float
        - final_eps: float
        - convergence_history: list of (iteration, A_throat, Cf, eps, residual)
    """
    # Validate nozzle efficiency
    if not (0 < nozzle_efficiency <= 1.0):
        raise ValueError(f"nozzle_efficiency must be in (0, 1], got {nozzle_efficiency}")
    
    # Calculate exit area (fixed)
    area_exit = area_exit_calc(diameter_exit)
    
    # Initial guess for throat area using a typical Cf value
    # Use a reasonable initial guess: Cf ~ 1.4-1.6 for typical nozzles (corrected)
    # Account for nozzle efficiency in initial guess
    Cf_ideal_initial_guess = 1.5
    Cf_initial_guess = nozzle_efficiency * Cf_ideal_initial_guess
    A_throat_guess = thrust_design / (pc_design * Cf_initial_guess)
    
    # Check if CEA cache supports 3D (eps-dependent) or 2D (fixed eps)
    use_3d = cea_cache.use_3d
    
    if verbose:
        print(f"Initializing chamber geometry solver...")
        print(f"  Pc = {pc_design/1e6:.2f} MPa")
        print(f"  F = {thrust_design:.1f} N")
        print(f"  MR = {MR:.3f}")
        print(f"  Nozzle efficiency = {nozzle_efficiency:.4f}")
        print(f"  CEA cache: {'3D (eps-dependent)' if use_3d else '2D (fixed eps)'}")
        print(f"  Initial A_throat guess = {A_throat_guess*1e6:.4f} mm²")
    
    # Iteration variables
    A_throat = A_throat_guess
    convergence_history = []
    converged = False
    
    # Validate input parameters
    if pc_design <= 0:
        raise ValueError(f"Chamber pressure must be positive, got {pc_design}")
    if thrust_design <= 0:
        raise ValueError(f"Thrust must be positive, got {thrust_design}")
    if area_exit <= 0:
        raise ValueError(f"Exit area must be positive, got {area_exit}")
    
    # Typical Cf range for rocket nozzles: 1.2-2.0
    # Lower bound: underexpanded nozzles at sea level
    # Upper bound: highly overexpanded nozzles or vacuum-optimized
    Cf_min_reasonable = 1.0
    Cf_max_reasonable = 2.5
    
    for iteration in range(max_iterations):
        # Calculate expansion ratio
        eps = expansion_ratio_calc(area_exit, A_throat)
        
        # Validate expansion ratio is reasonable
        if eps < 1.0:
            raise ValueError(
                f"Expansion ratio must be >= 1.0, got {eps:.4f}. "
                f"This means exit area ({area_exit*1e6:.2f} mm²) < throat area ({A_throat*1e6:.2f} mm²)."
            )
        if eps > 100.0:
            import warnings
            warnings.warn(
                f"Very high expansion ratio ({eps:.2f}) detected. "
                f"This may indicate throat area is too small or exit area is too large."
            )
        
        # Look up Cf_ideal from CEA cache
        Cf_ideal = None  # Initialize to track if lookup succeeded
        try:
            if use_3d:
                # 3D cache: use eps in lookup (this gives us the best Cf for this expansion ratio)
                cea_props = cea_cache.eval(MR, pc_design, Pa, eps)
            else:
                # 2D cache: eps is fixed, but we still need to pass it
                # The cache will use its default eps from config
                cea_props = cea_cache.eval(MR, pc_design, Pa, None)
            
            Cf_ideal = cea_props.get("Cf_ideal", Cf_ideal_initial_guess)
            
            # Apply nozzle efficiency correction to get corrected Cf
            # This matches what the pipeline uses: Cf = nozzle_efficiency * Cf_ideal
            Cf = nozzle_efficiency * Cf_ideal
            
            # Validate Cf_ideal is finite and positive
            if not np.isfinite(Cf_ideal) or Cf_ideal <= 0:
                raise ValueError(f"Invalid Cf_ideal from CEA cache: {Cf_ideal}")
            
            # Validate corrected Cf is finite and positive
            if not np.isfinite(Cf) or Cf <= 0:
                raise ValueError(f"Invalid corrected Cf: {Cf} (from Cf_ideal={Cf_ideal}, efficiency={nozzle_efficiency})")
            
            # Validate Cf_ideal is within reasonable bounds (before efficiency correction)
            if Cf_ideal < Cf_min_reasonable or Cf_ideal > Cf_max_reasonable:
                import warnings
                warnings.warn(
                    f"Cf_ideal ({Cf_ideal:.4f}) is outside typical range [{Cf_min_reasonable:.2f}, {Cf_max_reasonable:.2f}]. "
                    f"This may indicate: (1) unusual expansion ratio ({eps:.2f}), "
                    f"(2) CEA cache issue, or (3) design point issue. "
                    f"Proceeding with caution."
                )
                
        except Exception as e:
            if verbose:
                print(f"Warning: CEA lookup failed at iteration {iteration}: {e}")
            # Fallback to initial guess (already corrected for efficiency)
            Cf = Cf_initial_guess
            # Use initial guess for Cf_ideal if lookup failed
            if Cf_ideal is None:
                Cf_ideal = Cf_ideal_initial_guess
            if iteration == 0:
                # If first iteration fails, we can't proceed
                raise RuntimeError(
                    f"CEA lookup failed on first iteration. "
                    f"Check that (Pc={pc_design/1e6:.2f} MPa, MR={MR:.3f}, eps={eps:.2f}) "
                    f"is within cache bounds."
                ) from e
        
        # Calculate new throat area from thrust equation: F = Cf * Pc * A_throat
        A_throat_new = thrust_design / (pc_design * Cf)
        
        # Validate new throat area is reasonable
        if A_throat_new <= 0 or not np.isfinite(A_throat_new):
            raise ValueError(
                f"Calculated throat area is invalid: {A_throat_new}. "
                f"This may indicate Cf={Cf:.4f} is incorrect."
            )
        
        # Check convergence (relative error)
        residual = abs(A_throat_new - A_throat) / max(A_throat, 1e-10)
        
        # Also check absolute error for very small throat areas
        residual_abs = abs(A_throat_new - A_throat)
        
        # Store both corrected Cf and ideal Cf in history
        convergence_history.append({
            'iteration': iteration,
            'A_throat': A_throat,
            'Cf': Cf,  # Corrected Cf (with efficiency applied)
            'Cf_ideal': Cf_ideal,  # Ideal Cf from CEA (actual value looked up)
            'eps': eps,
            'residual': residual,
            'residual_abs': residual_abs
        })
        
        if verbose:
            print(f"  Iter {iteration}: A_throat={A_throat*1e6:.6f} mm², "
                  f"eps={eps:.4f}, Cf_ideal={Cf_ideal:.6f}, Cf_corrected={Cf:.6f}, residual={residual:.2e}")
        
        # Check convergence (both relative and absolute)
        if residual < tolerance or residual_abs < 1e-10:
            converged = True
            A_throat = A_throat_new
            break
        
        # Update for next iteration (use adaptive relaxation for stability)
        # For large residuals, use more conservative relaxation
        if residual > 0.1:
            relaxation = 0.5  # More conservative for large changes
        elif residual > 0.01:
            relaxation = 0.7  # Standard relaxation
        else:
            relaxation = 0.9  # More aggressive near convergence
        
        A_throat = relaxation * A_throat_new + (1 - relaxation) * A_throat
        
        # Prevent oscillation: if we're oscillating, reduce relaxation further
        if iteration >= 2:
            prev_residual = convergence_history[-2]['residual']
            if residual > prev_residual * 0.9:  # Not improving much
                relaxation = max(0.3, relaxation * 0.8)  # More conservative
                A_throat = relaxation * A_throat_new + (1 - relaxation) * A_throat
    
    if not converged:
        import warnings
        warnings.warn(
            f"Chamber geometry solver did not converge after {max_iterations} iterations. "
            f"Final residual: {residual:.2e}. Using last computed values."
        )
    
    # Final Cf lookup with converged values
    eps_final = expansion_ratio_calc(area_exit, A_throat)
    if use_3d:
        # For 3D cache: This gives us the ideal Cf for the converged expansion ratio
        # This is the best achievable Cf for the given (Pc, MR, eps, Pa) conditions
        cea_props_final = cea_cache.eval(MR, pc_design, Pa, eps_final)
    else:
        # For 2D cache: Uses fixed expansion ratio from config
        cea_props_final = cea_cache.eval(MR, pc_design, Pa, None)
    
    Cf_ideal_final = cea_props_final.get("Cf_ideal", Cf / nozzle_efficiency if nozzle_efficiency > 0 else Cf)
    # Apply nozzle efficiency correction to get corrected Cf
    Cf_final = nozzle_efficiency * Cf_ideal_final
    
    # Final validation: Verify the converged solution is self-consistent
    # Check that F = Cf_final * Pc * A_throat (within tolerance)
    thrust_check = Cf_final * pc_design * A_throat
    thrust_error = abs(thrust_check - thrust_design) / thrust_design
    
    if thrust_error > 0.01:  # 1% tolerance
        import warnings
        warnings.warn(
            f"Final solution may not be self-consistent: "
            f"Requested thrust = {thrust_design:.2f} N, "
            f"Calculated thrust = {thrust_check:.2f} N "
            f"(error = {thrust_error*100:.2f}%). "
            f"This may indicate convergence issues."
        )
    
    # Validate final Cf_ideal is reasonable (before efficiency correction)
    if Cf_ideal_final < Cf_min_reasonable or Cf_ideal_final > Cf_max_reasonable:
        import warnings
        warnings.warn(
            f"Final Cf_ideal ({Cf_ideal_final:.4f}) is outside typical range "
            f"[{Cf_min_reasonable:.2f}, {Cf_max_reasonable:.2f}]. "
            f"Verify design parameters are correct."
        )
    
    if verbose:
        print(f"\nConverged after {iteration + 1} iterations:")
        print(f"  Final A_throat = {A_throat*1e6:.6f} mm²")
        print(f"  Final eps = {eps_final:.4f}")
        print(f"  Final Cf_ideal = {Cf_ideal_final:.6f} (from CEA)")
        print(f"  Final Cf_corrected = {Cf_final:.6f} (with efficiency={nozzle_efficiency:.4f})")
        print(f"  Thrust check: {thrust_check:.2f} N (requested: {thrust_design:.2f} N, error: {thrust_error*100:.2f}%)")
    
    # Now use the converged Cf to calculate full geometry
    # Convert Cf to force_coefficient for the existing function
    force_coefficient = Cf_final
    
    chamber_pts, table_data, total_chamber_length = chamber_geometry_calc(
        pc_design=pc_design,
        thrust_design=thrust_design,
        force_coeffcient=force_coefficient,
        diameter_inner=diameter_inner,
        diameter_exit=diameter_exit,
        l_star=l_star,
        do_plot=do_plot,
        color_segments=color_segments,
        steps=steps,
        export_dxf=export_dxf,
    )
    
    # Add Cf information to table data
    # Find the row with "Contraction Ratio" and insert after it
    table_data_with_cf = table_data.copy()
    for i, row in enumerate(table_data_with_cf):
        if len(row) > 0 and row[0] == 'Contraction Ratio':
            # Insert Cf row after Contraction Ratio
            cf_row = [
                'Thrust Coefficient (Cf)',
                f'{Cf_final:.6f}',
                '',
                f'{Cf_final:.6f}',
                ''
            ]
            table_data_with_cf.insert(i + 1, cf_row)
            break
    
    # Solver information
    solver_info = {
        'converged': converged,
        'iterations': iteration + 1,
        'final_Cf': float(Cf_final),  # Corrected Cf (with efficiency applied)
        'final_Cf_ideal': float(Cf_ideal_final),  # Ideal Cf from CEA
        'final_A_throat': float(A_throat),
        'final_eps': float(eps_final),
        'nozzle_efficiency': float(nozzle_efficiency),
        'convergence_history': convergence_history,
        'cea_props': {
            'cstar_ideal': cea_props_final.get('cstar_ideal', np.nan),
            'Tc': cea_props_final.get('Tc', np.nan),
            'gamma': cea_props_final.get('gamma', np.nan),
            'R': cea_props_final.get('R', np.nan),
        }
    }
    
    return chamber_pts, table_data_with_cf, total_chamber_length, solver_info


def _get_default_cea_config() -> CEAConfig:
    """
    Create a default CEA config using the standard cache file.
    
    Returns:
    --------
    CEAConfig with default cache file path
    """
    # Get project root (assume we're in chamber/ directory, go up one level)
    # Or if running from project root, use relative path
    cache_file = DEFAULT_CEA_CACHE_FILE
    
    # Try to find the cache file - check if it exists relative to current working directory
    if not os.path.exists(cache_file):
        # Try from project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_file_abs = os.path.join(project_root, DEFAULT_CEA_CACHE_FILE)
        if os.path.exists(cache_file_abs):
            cache_file = cache_file_abs
        else:
            # Use relative path anyway - CEACache will handle path resolution
            cache_file = DEFAULT_CEA_CACHE_FILE
    
    return CEAConfig(
        ox_name="LOX",
        fuel_name="RP-1",
        expansion_ratio=10.0,  # Default for 2D cache (will be overridden if 3D cache)
        cache_file=cache_file,
        Pc_range=[1.0e6, 5.0e6],  # Will be updated from cache metadata if cache exists
        MR_range=[2.0, 3.0],  # Will be updated from cache metadata if cache exists
        n_points=34,  # Will be updated from cache metadata if cache exists
    )


def solve_chamber_geometry(
    pc_design: float,
    thrust_design: float,
    MR: float,
    cea_config: Optional[CEAConfig] = None,
    diameter_inner: float = chamber_diameter_default,
    diameter_exit: float = diameter_exit_default,
    l_star: float = l_star_default,
    **kwargs
) -> Tuple[np.ndarray, list, float, Dict[str, Any]]:
    """
    Solve chamber geometry using CEA lookup table (convenience function).
    
    This is the main entry point. If no CEA config is provided, uses the default
    cache file at output/cache/cea_cache_LOX_RP1.npz.
    
    Parameters:
    -----------
    pc_design : float
        Design chamber pressure [Pa]
    thrust_design : float
        Design thrust [N]
    MR : float
        Mixture ratio (O/F)
    cea_config : CEAConfig, optional
        CEA configuration object. If None, uses default cache file.
    diameter_inner : float, optional
        Inner chamber diameter [m]
    diameter_exit : float, optional
        Exit diameter [m]
    l_star : float, optional
        Characteristic length [m]
    **kwargs
        Additional arguments passed to solve_chamber_geometry_with_cea
        (do_plot, color_segments, steps, export_dxf, verbose, etc.)
    
    Returns:
    --------
    Same as solve_chamber_geometry_with_cea
    """
    if cea_config is None:
        cea_config = _get_default_cea_config()
    
    cea_cache = CEACache(cea_config)
    return solve_chamber_geometry_with_cea(
        pc_design=pc_design,
        thrust_design=thrust_design,
        cea_cache=cea_cache,
        MR=MR,
        diameter_inner=diameter_inner,
        diameter_exit=diameter_exit,
        l_star=l_star,
        **kwargs
    )


def solve_chamber_geometry_from_config(
    pc_design: float,
    thrust_design: float,
    cea_config: CEAConfig,
    MR: float,
    diameter_inner: float = chamber_diameter_default,
    diameter_exit: float = diameter_exit_default,
    l_star: float = l_star_default,
    **kwargs
) -> Tuple[np.ndarray, list, float, Dict[str, Any]]:
    """
    Convenience function that creates a CEA cache from config and solves.
    
    Note: For most use cases, use solve_chamber_geometry() instead, which
    automatically uses the default cache file.
    
    Parameters:
    -----------
    pc_design : float
        Design chamber pressure [Pa]
    thrust_design : float
        Design thrust [N]
    cea_config : CEAConfig
        CEA configuration object
    MR : float
        Mixture ratio (O/F)
    diameter_inner : float, optional
        Inner chamber diameter [m]
    diameter_exit : float, optional
        Exit diameter [m]
    l_star : float, optional
        Characteristic length [m]
    **kwargs
        Additional arguments passed to solve_chamber_geometry_with_cea
    
    Returns:
    --------
    Same as solve_chamber_geometry_with_cea
    """
    cea_cache = CEACache(cea_config)
    return solve_chamber_geometry_with_cea(
        pc_design=pc_design,
        thrust_design=thrust_design,
        cea_cache=cea_cache,
        MR=MR,
        diameter_inner=diameter_inner,
        diameter_exit=diameter_exit,
        l_star=l_star,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage - uses default cache file automatically
    # Test parameters
    pc_design = 2.068e6  # 300 PSI
    thrust_design = 6000.0  # N
    MR = 2.5  # O/F ratio
    
    print("=" * 80)
    print("Chamber Geometry Solver Test")
    print(f"Using default CEA cache: {DEFAULT_CEA_CACHE_FILE}")
    print("=" * 80)
    
    try:
        # Use the convenience function - automatically uses default cache file
        chamber_pts, table_data, total_length, solver_info = solve_chamber_geometry(
            pc_design=pc_design,
            thrust_design=thrust_design,
            MR=MR,
            do_plot=True,
            color_segments=True,
            export_dxf='chamber/chamber_contour_cea.dxf',
            verbose=True,
        )
        
        print("\n" + "=" * 80)
        print("Solution Summary")
        print("=" * 80)
        print(f"Converged: {solver_info['converged']}")
        print(f"Iterations: {solver_info['iterations']}")
        print(f"Final Cf: {solver_info['final_Cf']:.6f}")
        print(f"Final A_throat: {solver_info['final_A_throat']*1e6:.4f} mm²")
        print(f"Final eps: {solver_info['final_eps']:.4f}")
        print(f"Total chamber length: {total_length*1000:.2f} mm")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

