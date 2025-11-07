"""RocketCEA wrapper with MR/Pc caching and bilinear interpolation"""

import numpy as np
import os
import re
from rocketcea.cea_obj import CEA_Obj
from typing import Tuple, Optional
from .config_schemas import CEAConfig


def parse_cea_basic(out: str) -> Tuple[float, float, float, float]:
    """
    Extract Tc, gamma, R, and c* from NASA CEA text output (chamber values only).
    
    Returns:
    --------
    Tc : float [K]
    gamma : float
    R : float [J/(kg·K)]
    cstar : float [m/s]
    """
    # Extract the main performance block
    block_match = re.search(r'THEORETICAL ROCKET PERFORMANCE[\s\S]+?MOLE FRACTIONS', out)
    block = block_match.group(0) if block_match else out

    # Extract chamber molecular weight M
    mw_match = re.search(r'M,\s*\(1/n\)\s+([\d.E+-]+)', block)
    if mw_match:
        M = float(mw_match.group(1))  # molecular weight [kg/kmol]
        R = 8314.462618 / M  # J/kg·K
    else:
        R = np.nan

    # Extract chamber temperature (Tc)
    Tc_match = re.search(r'T[, ]*K\s+([\d.E+-]+)', block)
    Tc = float(Tc_match.group(1)) if Tc_match else np.nan

    # Extract gamma (chamber)
    gamma_match = re.search(r'GAMMAs\s+([\d.E+-]+)', block)
    gamma = float(gamma_match.group(1)) if gamma_match else np.nan

    # Extract c*
    cstar_match = re.search(r'CSTAR[, ]*(?:M/SEC|FT/SEC)?\s+([\d.E+-]+)', block, re.IGNORECASE)
    cstar = float(cstar_match.group(1)) if cstar_match else np.nan
    # Convert ft/s → m/s if needed
    if re.search(r'CSTAR.*FT/SEC', block, re.IGNORECASE):
        cstar *= 0.3048

    return Tc, gamma, R, cstar


class CEACache:
    """CEA cache with bilinear interpolation"""
    
    def __init__(self, config: CEAConfig):
        self.config = config
        # Make cache file path absolute (relative to current working directory or project root)
        if os.path.isabs(config.cache_file):
            self.cache_file = config.cache_file
        else:
            # Try current directory first, then project root
            if os.path.exists(config.cache_file):
                self.cache_file = os.path.abspath(config.cache_file)
            else:
                # Look in parent directory (project root)
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                potential_path = os.path.join(parent_dir, config.cache_file)
                if os.path.exists(potential_path):
                    self.cache_file = potential_path
                else:
                    # Use current directory (will create new cache)
                    self.cache_file = os.path.abspath(config.cache_file)
        
        # Grid parameters
        self.Pc_min, self.Pc_max = config.Pc_range
        self.MR_min, self.MR_max = config.MR_range
        self.n_points = config.n_points
        
        # Create grids
        self.Pc_grid = np.linspace(self.Pc_min, self.Pc_max, self.n_points)
        self.MR_grid = np.linspace(self.MR_min, self.MR_max, self.n_points)
        
        # Lookup tables (initialized as None, loaded from cache or computed)
        self.cstar_table = None
        self.Cf_table = None
        self.Tc_table = None
        self.gamma_table = None
        self.R_table = None
        
        # Load from cache or build
        if os.path.exists(self.cache_file):
            self._load_cache()
        else:
            self._build_cache()
    
    def _load_cache(self):
        """Load CEA data from cache file"""
        print(f"[OK] Loading CEA cache from {self.cache_file}")
        data = np.load(self.cache_file)
        
        self.cstar_table = data["cstar"]
        self.Cf_table = data["Cf"]
        self.Tc_table = data["Tc"]
        self.gamma_table = data["gamma"]
        self.R_table = data["R"]
        
        # Verify grid matches
        Pc_loaded = data["Pc"]
        MR_loaded = data["MR"]
        
        if not np.allclose(Pc_loaded, self.Pc_grid) or not np.allclose(MR_loaded, self.MR_grid):
            print("[WARNING] Cache grid doesn't match config, rebuilding...")
            self._build_cache()
    
    def _build_cache(self):
        """Build CEA lookup tables (this takes a while)"""
        print(f"[BUILDING] Building CEA cache (this will take a while)...")
        print(f"   Grid: Pc ∈ [{self.Pc_min/1e6:.1f}, {self.Pc_max/1e6:.1f}] MPa")
        print(f"         MR ∈ [{self.MR_min:.2f}, {self.MR_max:.2f}]")
        print(f"         Points: {self.n_points}×{self.n_points} = {self.n_points**2}")
        
        chamber = CEA_Obj(oxName=self.config.ox_name, fuelName=self.config.fuel_name)
        
        # Initialize tables
        self.cstar_table = np.zeros((self.n_points, self.n_points))
        self.Cf_table = np.zeros((self.n_points, self.n_points))
        self.Tc_table = np.zeros((self.n_points, self.n_points))
        self.gamma_table = np.zeros((self.n_points, self.n_points))
        self.R_table = np.zeros((self.n_points, self.n_points))
        
        # Convert Pc from Pa to psia for CEA
        Pc_psia_grid = self.Pc_grid / 6894.76
        
        # Build lookup tables
        for i, Pc_psia in enumerate(Pc_psia_grid):
            if (i + 1) % 20 == 0:
                print(f"   Progress: {i+1}/{self.n_points} ({100*(i+1)/self.n_points:.1f}%)")
            
            for j, MR in enumerate(self.MR_grid):
                try:
                    # Get full CEA output for detailed parsing
                    out = chamber.get_full_cea_output(
                        Pc=Pc_psia,
                        MR=MR,
                        eps=self.config.expansion_ratio
                    )
                    Tc, gamma, R, cstar = parse_cea_basic(out)
                    
                    # Get Isp and calculate Cf_ideal
                    isp = chamber.estimate_Ambient_Isp(
                        Pc=Pc_psia,
                        MR=MR,
                        eps=self.config.expansion_ratio
                    )[0]
                    
                    # Cf_ideal from Isp: Cf = Isp * g0 / c*
                    # But CEA can give Cf directly, so try that first
                    try:
                        Cf_ideal = chamber.get_PambCf(
                            Pc=Pc_psia,
                            MR=MR,
                            eps=self.config.expansion_ratio
                        )[0]
                    except:
                        # Fallback: Cf ≈ Isp * g0 / c* (approximate)
                        Cf_ideal = isp * 9.80665 / cstar if cstar > 0 else np.nan
                    
                    self.cstar_table[i, j] = cstar
                    self.Cf_table[i, j] = Cf_ideal
                    self.Tc_table[i, j] = Tc
                    self.gamma_table[i, j] = gamma
                    self.R_table[i, j] = R
                    
                except Exception as e:
                    print(f"   ⚠️  Error at Pc={Pc_psia:.1f} psia, MR={MR:.2f}: {e}")
                    self.cstar_table[i, j] = np.nan
                    self.Cf_table[i, j] = np.nan
                    self.Tc_table[i, j] = np.nan
                    self.gamma_table[i, j] = np.nan
                    self.R_table[i, j] = np.nan
        
        # Save to cache
        self._save_cache()
        print(f"💾 CEA cache saved to {self.cache_file}")
    
    def _save_cache(self):
        """Save CEA data to cache file"""
        np.savez_compressed(
            self.cache_file,
            Pc=self.Pc_grid,
            MR=self.MR_grid,
            cstar=self.cstar_table,
            Cf=self.Cf_table,
            Tc=self.Tc_table,
            gamma=self.gamma_table,
            R=self.R_table,
        )
    
    def _bilinear_interpolate(self, Pc: float, MR: float, table: np.ndarray) -> float:
        """Bilinear interpolation in (Pc, MR) space"""
        # Find indices
        i_pc = np.searchsorted(self.Pc_grid, Pc)
        i_mr = np.searchsorted(self.MR_grid, MR)
        
        # Clamp to valid range
        i_pc = np.clip(i_pc, 0, len(self.Pc_grid) - 1)
        i_mr = np.clip(i_mr, 0, len(self.MR_grid) - 1)
        
        # Handle edge cases
        if i_pc == 0:
            i_pc = 1
        if i_mr == 0:
            i_mr = 1
        if i_pc >= len(self.Pc_grid):
            i_pc = len(self.Pc_grid) - 1
        if i_mr >= len(self.MR_grid):
            i_mr = len(self.MR_grid) - 1
        
        # Get surrounding points
        Pc0, Pc1 = self.Pc_grid[i_pc - 1], self.Pc_grid[i_pc]
        MR0, MR1 = self.MR_grid[i_mr - 1], self.MR_grid[i_mr]
        
        # Bilinear interpolation
        f00 = table[i_pc - 1, i_mr - 1]
        f01 = table[i_pc - 1, i_mr]
        f10 = table[i_pc, i_mr - 1]
        f11 = table[i_pc, i_mr]
        
        # Check for NaN values
        if np.isnan(f00) or np.isnan(f01) or np.isnan(f10) or np.isnan(f11):
            # Fallback to nearest neighbor
            return table[i_pc - 1, i_mr - 1]
        
        # Interpolation weights
        wx = (Pc - Pc0) / (Pc1 - Pc0) if Pc1 != Pc0 else 0
        wy = (MR - MR0) / (MR1 - MR0) if MR1 != MR0 else 0
        
        # Bilinear interpolation
        result = (f00 * (1 - wx) * (1 - wy) +
                 f10 * wx * (1 - wy) +
                 f01 * (1 - wx) * wy +
                 f11 * wx * wy)
        
        return float(result)
    
    def eval(self, MR: float, Pc: float, Pa: float = 101325.0, eps: Optional[float] = None) -> dict:
        """
        Evaluate CEA properties at given conditions.
        
        Parameters:
        -----------
        MR : float
            Mixture ratio (O/F)
        Pc : float
            Chamber pressure [Pa]
        Pa : float
            Ambient pressure [Pa] (default: sea level)
        eps : float, optional
            Expansion ratio (uses config default if None)
        
        Returns:
        --------
        dict with keys:
            cstar_ideal : float [m/s]
            Cf_ideal : float
            Tc : float [K]
            gamma : float
            R : float [J/(kg·K)]
        """
        if eps is None:
            eps = self.config.expansion_ratio
        
        # Clamp to grid bounds
        Pc = np.clip(Pc, self.Pc_min, self.Pc_max)
        MR = np.clip(MR, self.MR_min, self.MR_max)
        
        return {
            "cstar_ideal": self._bilinear_interpolate(Pc, MR, self.cstar_table),
            "Cf_ideal": self._bilinear_interpolate(Pc, MR, self.Cf_table),
            "Tc": self._bilinear_interpolate(Pc, MR, self.Tc_table),
            "gamma": self._bilinear_interpolate(Pc, MR, self.gamma_table),
            "R": self._bilinear_interpolate(Pc, MR, self.R_table),
        }
