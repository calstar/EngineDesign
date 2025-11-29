"""Pydantic schemas for YAML/JSON configuration validation"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Literal, Optional, Union
import numpy as np


class FluidConfig(BaseModel):
    """Fluid property configuration"""
    name: str
    density: float = Field(gt=0, description="Density [kg/m³]")
    viscosity: float = Field(gt=0, description="Dynamic viscosity [Pa·s]")
    surface_tension: float = Field(gt=0, description="Surface tension [N/m]")
    vapor_pressure: float = Field(ge=0, description="Vapor pressure [Pa]")
    specific_heat: float = Field(default=2200.0, gt=0, description="Specific heat at constant pressure [J/(kg·K)]")
    thermal_conductivity: float = Field(default=0.15, gt=0, description="Thermal conductivity [W/(m·K)]")
    temperature: float = Field(default=293.15, gt=0, description="Bulk fluid temperature [K]")


class PintleLOXConfig(BaseModel):
    """LOX (oxidizer) pintle geometry - Axial flow through orifices"""
    n_orifices: int = Field(gt=0, description="Number of orifices on pintle tip")
    d_orifice: float = Field(gt=0, description="Diameter of each orifice [m]")
    theta_orifice: float = Field(ge=0, le=90, description="Angle of orifices from axis [deg]")
    A_entry: float = Field(gt=0, description="Single entry hole area [m²]")
    d_hydraulic: float = Field(gt=0, description="Hydraulic diameter for Re calculation [m]")


class PintleFuelConfig(BaseModel):
    """Fuel (RP-1) pintle geometry - Reservoir with gap spillage"""
    d_pintle_tip: float = Field(gt=0, description="Outer diameter of pintle tip [m]")
    d_reservoir_inner: float = Field(gt=0, description="Inner diameter of fuel reservoir [m]")
    h_gap: float = Field(gt=0, description="Gap height between pintle tip and reservoir [m]")
    A_entry: float = Field(gt=0, description="Single entry port area into reservoir [m²]")
    d_hydraulic: float = Field(gt=0, description="Hydraulic diameter for Re calculation [m] (gap hydraulic diameter)")


class PintleGeometryConfig(BaseModel):
    """Pintle injector geometry configuration"""
    lox: PintleLOXConfig
    fuel: PintleFuelConfig


class InjectorBaseConfig(BaseModel):
    """Base injector configuration with type identifier"""
    type: Literal["pintle", "coaxial", "impinging"]


class PintleInjectorConfig(InjectorBaseConfig):
    """Complete pintle injector configuration"""
    type: Literal["pintle"] = "pintle"
    geometry: PintleGeometryConfig


class CoaxialCoreConfig(BaseModel):
    """Core (inner) element geometry for coaxial injector"""
    n_ports: int = Field(gt=0, description="Number of core ports/nozzles")
    d_port: float = Field(gt=0, description="Diameter of each core port [m]")
    length: Optional[float] = Field(default=None, gt=0, description="Port length for loss modeling [m]")


class CoaxialAnnulusConfig(BaseModel):
    """Annular (outer) element geometry for coaxial injector"""
    inner_diameter: float = Field(gt=0, description="Inner diameter of annulus (matches core OD) [m]")
    gap_thickness: float = Field(gt=0, description="Annulus gap thickness [m]")
    swirl_angle: float = Field(default=0.0, ge=0, le=90, description="Swirl angle for outer flow [deg]")


class CoaxialInjectorGeometry(BaseModel):
    """Complete geometry description for a shear coaxial injector"""
    core: CoaxialCoreConfig
    annulus: CoaxialAnnulusConfig


class CoaxialInjectorConfig(InjectorBaseConfig):
    """Coaxial injector configuration"""
    type: Literal["coaxial"] = "coaxial"
    geometry: CoaxialInjectorGeometry


class ImpingingElementConfig(BaseModel):
    """Geometry parameters for a single impinging jet element"""
    n_elements: int = Field(gt=0, description="Number of elements (pairs or triplets)")
    d_jet: float = Field(gt=0, description="Jet diameter [m]")
    impingement_angle: float = Field(gt=0, le=180, description="Included impingement angle [deg]")
    spacing: float = Field(gt=0, description="Center-to-center spacing between jets [m]")


class ImpingingInjectorGeometry(BaseModel):
    """Complete geometry for an impinging injector"""
    oxidizer: ImpingingElementConfig
    fuel: ImpingingElementConfig


class ImpingingInjectorConfig(InjectorBaseConfig):
    """Impinging-element injector configuration"""
    type: Literal["impinging"] = "impinging"
    geometry: ImpingingInjectorGeometry


class FeedSystemConfig(BaseModel):
    """Feed system configuration for one branch (O or F)"""
    d_inlet: float = Field(gt=0, description="Inlet pipe diameter [m] (e.g., 3/8\" = 0.009525 m)")
    A_hydraulic: float = Field(gt=0, description="Hydraulic area of feed line [m²] (calculated from d_inlet if not specified)")
    K0: float = Field(ge=0, description="Base loss coefficient")
    K1: float = Field(ge=0, description="Pressure dependence coefficient")
    phi_type: Literal["none", "sqrtP", "logP"] = Field(
        default="none",
        description="Pressure function type"
    )


class RegenCoolingConfig(BaseModel):
    """Regenerative cooling channel configuration"""
    enabled: bool = Field(default=False, description="Enable regen cooling model")
    d_inlet: float = Field(gt=0, description="Inlet pipe diameter [m] (e.g., 3/8\" = 0.009525 m)")
    L_inlet: float = Field(gt=0, description="Inlet pipe length [m]")
    n_channels: int = Field(gt=0, description="Number of parallel cooling channels")
    channel_width: float = Field(gt=0, description="Channel width [m]")
    channel_height: float = Field(gt=0, description="Channel height [m]")
    channel_length: float = Field(gt=0, description="Channel length [m] (typically chamber length)")
    d_outlet: Optional[float] = Field(default=None, description="Outlet pipe diameter [m] (default: same as inlet)")
    L_outlet: float = Field(gt=0, description="Outlet pipe length [m] (from merge to injector)")
    roughness: float = Field(default=0.0, ge=0, description="Surface roughness [m] (0 = smooth)")
    K_manifold_split: float = Field(default=0.5, ge=0, description="Manifold split loss coefficient")
    K_manifold_merge: float = Field(default=0.3, ge=0, description="Manifold merge loss coefficient")
    # Dynamic discharge coefficient configuration (similar to injector Cd)
    Cd_entrance_inf: float = Field(default=0.8, gt=0, le=1, description="Asymptotic Cd at high Re for channel entrance")
    a_Re_entrance: float = Field(default=0.1, ge=0, description="Reynolds correction parameter for entrance")
    Cd_entrance_min: float = Field(default=0.6, ge=0, le=1, description="Minimum Cd for entrance")
    Cd_exit_inf: float = Field(default=0.9, gt=0, le=1, description="Asymptotic Cd at high Re for channel exit")
    a_Re_exit: float = Field(default=0.1, ge=0, description="Reynolds correction parameter for exit")
    Cd_exit_min: float = Field(default=0.7, ge=0, le=1, description="Minimum Cd for exit")
    # Heat-transfer coupling (Phase 2)
    use_heat_transfer: bool = Field(default=False, description="Enable coupled heat-transfer calculations for regen cooling")
    wall_thickness: float = Field(default=0.002, gt=0, description="Hot-wall thickness between gas and coolant [m]")
    wall_thermal_conductivity: float = Field(default=300.0, gt=0, description="Wall material thermal conductivity [W/(m·K)]")
    chamber_inner_diameter: Optional[float] = Field(default=None, gt=0, description="Chamber inner diameter for hot-side area [m]")
    hot_gas_prandtl: float = Field(default=0.7, gt=0, description="Assumed hot-gas Prandtl number")
    hot_gas_viscosity: float = Field(default=4.0e-5, gt=0, description="Effective hot-gas viscosity [Pa·s]")
    hot_gas_thermal_conductivity: float = Field(default=0.1, gt=0, description="Effective hot-gas thermal conductivity [W/(m·K)]")
    radiation_emissivity_hot: float = Field(default=0.8, ge=0, le=1, description="Effective hot-side emissivity for radiation")
    radiation_view_factor: float = Field(default=1.0, ge=0, le=1, description="Radiation view factor to coolant surface")
    n_segments: int = Field(default=20, gt=0, description="Number of axial segments for heat-transfer integration")
    gas_turbulence_intensity: float = Field(default=0.1, ge=0, description="Estimated turbulence intensity of hot gas (0-1)")
    coolant_turbulence_intensity: float = Field(default=0.05, ge=0, description="Estimated turbulence intensity of coolant (0-1)")
    hot_gas_cp: float = Field(default=2200.0, gt=0, description="Hot-gas specific heat [J/(kg·K)]")
    recovery_factor: Optional[float] = Field(default=None, gt=0, le=1, description="Turbulent boundary layer recovery factor for adiabatic wall temperature (Taw = Tc × recovery_factor). Typical range: 0.90-0.98. If None, uses default from constants.")


class FilmCoolingConfig(BaseModel):
    """Film cooling configuration"""
    enabled: bool = Field(default=False, description="Enable film cooling model")
    mass_fraction: float = Field(default=0.05, ge=0, le=0.5, description="Fraction of total mass flow used for film injection")
    injection_temperature: Optional[float] = Field(default=None, gt=0, description="Film injection temperature [K] (defaults to fuel temperature)")
    effectiveness_ref: float = Field(default=0.4, ge=0, le=1, description="Reference film effectiveness at injection location")
    decay_length: float = Field(default=0.1, gt=0, description="Characteristic decay length for film effectiveness [m]")
    apply_to_fraction_of_length: float = Field(default=1.0, gt=0, description="Portion of chamber length covered by film cooling")
    slot_height: float = Field(default=3.0e-4, gt=0, description="Annular slot height for film injection [m]")
    reference_blowing_ratio: float = Field(default=0.5, gt=0, description="Reference blowing ratio for effectiveness correlation")
    blowing_exponent: float = Field(default=0.6, gt=0, description="Exponent on blowing ratio for effectiveness correlation")
    turbulence_reference_intensity: float = Field(default=0.08, gt=0, description="Reference turbulence intensity for film erosion")
    turbulence_sensitivity: float = Field(default=1.0, ge=0, description="Sensitivity of film effectiveness to turbulence intensity")
    turbulence_exponent: float = Field(default=1.0, gt=0, description="Exponent governing turbulence erosion scaling")
    turbulence_min_multiplier: float = Field(default=0.4, ge=0, le=1, description="Minimum multiplier applied to effectiveness due to turbulence erosion")
    reference_wall_temperature: float = Field(default=1100.0, gt=0, description="Reference hot wall temperature used for heat-flux estimation [K]")
    density_override: Optional[float] = Field(default=None, gt=0, description="Override density for film coolant if different from bulk fuel [kg/m³]")
    cp_override: Optional[float] = Field(default=None, gt=0, description="Override specific heat for film coolant if different from bulk fuel [J/(kg·K)]")


class GraphiteInsertConfig(BaseModel):
    """Graphite throat insert configuration (separate from chamber ablator)"""
    enabled: bool = Field(default=False, description="Enable graphite throat insert")
    material_density: float = Field(default=1800.0, gt=0, description="Graphite density [kg/m³] (typical: 1800-2200)")
    heat_of_ablation: float = Field(default=8.0e6, gt=0, description="Effective heat of ablation [J/kg] (graphite: ~8-12 MJ/kg)")
    thermal_conductivity: float = Field(default=100.0, gt=0, description="Graphite thermal conductivity [W/(m·K)] (typical: 50-150)")
    specific_heat: float = Field(default=710.0, gt=0, description="Graphite specific heat [J/(kg·K)]")
    initial_thickness: float = Field(default=0.005, gt=0, description="Initial graphite insert thickness [m]")
    surface_temperature_limit: float = Field(default=2500.0, gt=0, description="Maximum surface temperature before failure [K]")
    oxidation_temperature: float = Field(default=800.0, gt=0, description="Onset temperature for oxidation [K]")
    oxidation_rate: float = Field(default=1e-6, ge=0, description="Oxidation recession rate [m/s] at reference conditions")
    activation_energy: Optional[float] = Field(default=180e3, gt=0, description="Activation energy for Arrhenius oxidation rate [J/mol]. Typical: 150-200 kJ/mol")
    oxidation_reference_temperature: float = Field(default=1500.0, gt=0, description="Reference temperature where oxidation_rate is defined [K]. Typical: 1500-1800 K")
    oxidation_reference_pressure: float = Field(default=1.0e6, gt=0, description="Reference pressure where oxidation_rate is defined [Pa]. Typical: 1 MPa")
    recession_multiplier: Optional[float] = Field(default=None, gt=0, description="Recession multiplier vs chamber (if None, calculated from Bartz correlation). Typically 1.3-2.5")
    char_layer_conductivity: float = Field(default=5.0, gt=0, description="Thermal conductivity of protective layer [W/(m·K)]")
    char_layer_thickness: float = Field(default=0.0005, gt=0, description="Thickness of protective layer [m]")
    coverage_fraction: float = Field(default=1.0, gt=0, le=1.0, description="Fraction of throat/nozzle with graphite insert")
    emissivity: Optional[float] = Field(default=None, ge=0, le=1, description="Surface emissivity for radiation (default: 0.8)")
    ambient_temperature: Optional[float] = Field(default=None, gt=0, description="Ambient temperature for radiation [K] (default: 300 K)")
    feedback_fraction_min: Optional[float] = Field(default=None, ge=0, le=1, description="Minimum oxidation heat feedback fraction (default: 0.0)")
    feedback_fraction_max: Optional[float] = Field(default=None, ge=0, le=1, description="Maximum oxidation heat feedback fraction (default: 0.2)")
    oxidation_enthalpy: Optional[float] = Field(default=None, gt=0, description="Oxidation reaction enthalpy [J/kg C]. If None, auto-selected: 32.8e6 for CO2 (ratio=1.0), 10.1e6 for CO (ratio=2.0)")
    ablation_surface_temperature: Optional[float] = Field(default=None, gt=0, description="Surface temperature at which thermal ablation pins T_s [K] (default: 3000 K). Above this, T_s is fixed and m_dot_th balances energy.")
    oxidation_pressure_exponent: Optional[float] = Field(default=None, ge=0, description="Pressure exponent for oxidation kinetics (default: 0.5)")
    oxidation_pre_exponential: Optional[float] = Field(default=None, gt=0, description="Pre-exponential factor for Arrhenius oxidation (default: calculated from oxidation_rate)")
    mixture_mw: Optional[float] = Field(default=None, gt=0, description="Average molecular weight of combustion products [kg/mol] (default: 0.024)")
    oxidation_stoichiometry_ratio: Optional[float] = Field(default=None, gt=0, description="Moles of C per mole of O2 (1.0 for CO2, 2.0 for CO) (default: 1.0)")


class StainlessSteelCaseConfig(BaseModel):
    """Stainless steel case configuration (structural wall behind ablative/graphite)"""
    enabled: bool = Field(default=True, description="Enable stainless steel case")
    thickness: float = Field(default=0.003, gt=0, description="Stainless steel wall thickness [m]")
    thermal_conductivity: float = Field(default=15.0, gt=0, description="Thermal conductivity [W/(m·K)]")
    density: float = Field(default=8000.0, gt=0, description="Material density [kg/m³]")
    specific_heat: float = Field(default=500.0, gt=0, description="Specific heat [J/(kg·K)]")
    max_temperature: float = Field(default=1000.0, gt=0, description="Maximum allowable temperature [K] (melting point ~1700K, but limit lower for structural integrity)")
    emissivity: float = Field(default=0.3, ge=0, le=1, description="Surface emissivity")
    yield_strength: float = Field(default=200e6, gt=0, description="Yield strength at max temp [Pa]")
    youngs_modulus: float = Field(default=200e9, gt=0, description="Young's modulus [Pa]")


class AblativeCoolingConfig(BaseModel):
    """Ablative cooling configuration for chamber liner (phenolic)"""
    enabled: bool = Field(default=False, description="Enable ablative cooling model")
    material_density: float = Field(default=1600.0, gt=0, description="Ablator (phenolic) density [kg/m³]")
    heat_of_ablation: float = Field(default=2.5e6, gt=0, description="Effective heat of ablation [J/kg]")
    thermal_conductivity: float = Field(default=0.35, gt=0, description="Ablator (phenolic) thermal conductivity [W/(m·K)]")
    specific_heat: float = Field(default=1500.0, gt=0, description="Ablator (phenolic) specific heat [J/(kg·K)]")
    initial_thickness: float = Field(default=0.01, gt=0, description="Initial ablative (phenolic) thickness [m]")
    surface_temperature_limit: float = Field(default=1200.0, gt=0, description="Allowable surface temperature [K]")
    coverage_fraction: float = Field(default=1.0, gt=0, le=1.0, description="Fraction of chamber surface protected by ablative liner")
    pyrolysis_temperature: float = Field(default=900.0, gt=0, description="Characteristic pyrolysis temperature of ablator [K]")
    blowing_efficiency: float = Field(default=0.8, ge=0, le=1, description="Effectiveness of ablative gases in blocking convective heat flux (legacy constant factor, used if use_physics_based_blowing=False)")
    use_physics_based_blowing: bool = Field(default=True, description="If True, use physics-based blowing parameter B = m_dot_pyrolysis/m_dot_external. If False, use constant blowing_efficiency factor.")
    blowing_coefficient: float = Field(default=0.5, gt=0, description="Blowing coefficient c in empirical function f(B) = 1/(1 + c*B). Typical range: 0.3-0.8. Higher values = stronger blowing effect.")
    blowing_min_reduction_factor: float = Field(default=0.1, ge=0, le=1, description="Minimum convective reduction factor (maximum blowing effectiveness). Prevents blowing from reducing convective heat transfer below this fraction. Default 0.1 means maximum 90% reduction. Lower values allow stronger blowing effect.")
    turbulence_reference_intensity: float = Field(default=0.08, gt=0, description="Reference turbulence intensity for ablative augmentation")
    turbulence_sensitivity: float = Field(default=1.5, ge=0, description="Sensitivity of ablative heat flux to turbulence")
    turbulence_exponent: float = Field(default=1.0, gt=0, description="Exponent on turbulence intensity for ablative response")
    turbulence_max_multiplier: float = Field(default=3.0, gt=0, description="Maximum multiplier applied to convective heat flux due to turbulence")
    throat_recession_multiplier: Optional[float] = Field(default=None, gt=0, description="Throat recession multiplier vs chamber (if None, calculated from flow conditions). Typically 1.2-2.0")
    char_layer_conductivity: float = Field(default=0.2, gt=0, description="Thermal conductivity of char layer [W/(m·K)]")
    char_layer_thickness: float = Field(default=0.001, gt=0, description="Thickness of protective char layer [m]")
    surface_emissivity: float = Field(default=0.85, ge=0, le=1, description="Surface emissivity for radiative heat transfer (0-1, typical 0.8-0.9 for charred ablators)")
    ambient_temperature: float = Field(default=300.0, gt=0, description="Ambient/surrounding temperature for radiative heat transfer [K]. For radiation to space, use ~300K. For radiation exchange with gas, use gas temperature.")
    radiative_sink_minimum_threshold: float = Field(default=400.0, gt=0, description="Minimum ambient temperature threshold [K]. If ambient_temperature is below this, radiative_sink_fallback_temperature is used instead.")
    radiative_sink_fallback_temperature: float = Field(default=600.0, gt=0, description="Fallback radiative sink temperature [K] used when ambient_temperature is too low. Represents approximate heated steel layer temperature behind ablator.")
    track_geometry_evolution: bool = Field(default=True, description="Enable time-varying geometry tracking (L* evolution)")
    nozzle_ablative: bool = Field(default=False, description="If True, nozzle exit also recedes (A_exit grows). If False, only throat recedes (expansion ratio decreases)")


class DischargeConfig(BaseModel):
    """Discharge coefficient configuration"""
    Cd_inf: float = Field(gt=0, le=1, description="Cd at infinite Re")
    a_Re: float = Field(ge=0, description="Reynolds number correction parameter")
    Cd_min: float = Field(default=0.2, ge=0, le=1, description="Minimum Cd")
    # Pressure and temperature dependence (optional)
    use_pressure_correction: bool = Field(default=False, description="Enable pressure-dependent Cd (compressibility effects)")
    P_ref: float = Field(default=5.0e6, gt=0, description="Reference pressure for pressure correction [Pa]")
    a_P: float = Field(default=0.0, description="Pressure correction coefficient")
    use_temperature_correction: bool = Field(default=False, description="Enable temperature-dependent Cd (viscosity effects)")
    T_ref: float = Field(default=300.0, gt=0, description="Reference temperature for temperature correction [K]")
    a_T: float = Field(default=0.0, description="Temperature correction coefficient")


class SprayAngleConfig(BaseModel):
    """Spray angle model configuration"""
    model: Literal["J", "TMR"] = Field(default="TMR", description="Model type")
    k: float = Field(default=0.5, gt=0, description="J model coefficient")
    n: float = Field(default=0.5, gt=0, description="J model exponent")


class SMDConfig(BaseModel):
    """Sauter Mean Diameter configuration"""
    model: Literal["lefebvre", "nukiyama_tanasawa"] = Field(
        default="lefebvre",
        description="SMD model type"
    )
    C: float = Field(default=0.5, gt=0, description="Lefebvre constant C")
    m: float = Field(default=0.6, gt=0, description="Lefebvre exponent m")
    p: float = Field(default=0.0, description="Lefebvre exponent p")


class EvaporationConfig(BaseModel):
    """Evaporation model configuration"""
    K: float = Field(default=3e5, gt=0, description="Evaporation constant [s/m²]")
    x_star_limit: float = Field(default=0.05, gt=0, description="Max evaporation length [m]")
    use_constraint: bool = Field(default=True, description="Enable x* constraint")


class SprayConfig(BaseModel):
    """Spray/mixing model configuration"""
    momentum_flux_ratio: bool = Field(default=True, description="Enable J calculation")
    spray_angle: SprayAngleConfig = Field(default_factory=SprayAngleConfig)
    weber: dict = Field(default_factory=lambda: {"We_min": 15.0})
    smd: SMDConfig = Field(default_factory=SMDConfig)
    evaporation: EvaporationConfig = Field(default_factory=EvaporationConfig)
    use_turbulence_corrections: bool = Field(default=False, description="Enable turbulence-dependent spray corrections")
    turbulence_breakup_gain: float = Field(default=1.0, ge=0, description="Gain applied to droplet breakup due to turbulence")
    turbulence_penetration_gain: float = Field(default=0.5, ge=0, description="Gain applied to evaporation length reduction due to turbulence")


class CEAConfig(BaseModel):
    use_parallel_cea_build: bool = Field(default=True, description="Use parallel processing for CEA cache building")
    cea_parallel_workers: Optional[int] = Field(default=None, description="Number of parallel workers (None = auto-detect, limited to 8)")
    """CEA (Chemical Equilibrium Analysis) configuration"""
    ox_name: str = Field(default="LOX", description="Oxidizer name")
    fuel_name: str = Field(default="RP-1", description="Fuel name")
    expansion_ratio: float = Field(gt=1, description="Nozzle expansion ratio (initial/default value)")
    cache_file: str = Field(default="cea_cache_LOX_RP1.npz", description="Cache filename")
    Pc_range: list[float] = Field(
        default=[2.0e6, 9.0e6],
        description="Chamber pressure range [Pa]"
    )
    MR_range: list[float] = Field(
        default=[2.0, 2.8],
        description="Mixture ratio range"
    )
    eps_range: Optional[list[float]] = Field(
        default=None,
        description="Expansion ratio range for 3D cache [min, max]. If None, uses 2D cache with fixed expansion_ratio"
    )
    n_points: int = Field(default=34, gt=0, description="Number of grid points per dimension (34³ ≈ 39,300 points for 3D cache, similar to old 2D cache with ~40k points)")


class CombustionEfficiencyConfig(BaseModel):
    """Combustion efficiency (L* correction) configuration
    
    NOTE: CEA uses EQUILIBRIUM flow (not frozen). The efficiency correction
    accounts for finite chamber effects that prevent perfect equilibrium.
    
    Two models available:
    1. Simple model: Exponential L* correction (backward compatible)
    2. Advanced model: Physics-based with kinetics, mixing, turbulence
    """
    model: Literal["constant", "linear", "exponential"] = Field(
        default="exponential",
        description="Efficiency model type"
    )
    C: float = Field(default=0.3, ge=0, le=1, description="Efficiency loss parameter (for exponential/linear models)")
    K: float = Field(default=0.15, ge=0, description="Recovery rate parameter (for exponential model)")
    use_spray_correction: bool = Field(default=False, description="Apply spray quality penalty")
    spray_penalty_factor: float = Field(default=0.8, ge=0, le=1, description="Efficiency penalty if spray constraints violated")
    use_mixture_coupling: bool = Field(default=True, description="Apply mixture quality corrections")
    use_cooling_coupling: bool = Field(default=True, description="Apply cooling efficiency corrections")
    use_turbulence_coupling: bool = Field(default=False, description="Apply turbulence corrections")
    mixture_efficiency_floor: float = Field(default=0.25, ge=0, le=1, description="Minimum mixture efficiency")
    cooling_efficiency_floor: float = Field(default=0.25, ge=0, le=1, description="Minimum cooling efficiency")
    turbulence_efficiency_floor: float = Field(default=0.3, ge=0, le=1, description="Minimum turbulence efficiency")
    target_turbulence_intensity: float = Field(default=0.08, ge=0, description="Target turbulence intensity")
    turbulence_penalty_exponent: float = Field(default=1.0, gt=0, description="Turbulence penalty exponent")
    target_smd_microns: float = Field(default=50.0, gt=0, description="Target SMD for good atomization [microns]")
    xstar_limit_mm: float = Field(default=50.0, gt=0, description="Maximum evaporation length [mm]")
    xstar_penalty_exponent: float = Field(default=2.0, gt=0, description="Evaporation length penalty exponent")
    we_reference: float = Field(default=100.0, gt=0, description="Reference Weber number")
    we_penalty_exponent: float = Field(default=0.5, gt=0, description="Weber number penalty exponent")
    smd_penalty_exponent: float = Field(default=1.0, gt=0, description="SMD penalty exponent")
    use_advanced_model: bool = Field(
        default=False,
        description="Use advanced physics-based efficiency model (kinetics, mixing, turbulence)"
    )
    # Finite-rate chemistry and reaction modeling
    use_finite_rate_chemistry: bool = Field(
        default=True,
        description="Model finite-rate chemistry in chamber (reaction progress tracking)"
    )
    use_shifting_equilibrium: bool = Field(
        default=True,
        description="Use shifting equilibrium in nozzle (composition changes with expansion)"
    )


class CombustionConfig(BaseModel):
    """Combustion configuration"""
    cea: CEAConfig = Field(default_factory=CEAConfig)
    efficiency: CombustionEfficiencyConfig = Field(default_factory=CombustionEfficiencyConfig)


class ChamberConfig(BaseModel):
    """Chamber geometry configuration"""
    volume: float = Field(gt=0, description="Chamber volume [m³]")
    A_throat: float = Field(gt=0, description="Throat area [m²]")
    length: Optional[float] = Field(default=None, gt=0, description="Characteristic chamber length [m]")
    Lstar: Optional[float] = Field(
        default=None,
        gt=0,
        description="Characteristic length [m] = V_chamber / A_throat. If not specified, calculated from volume and A_throat."
    )
    chamber_inner_diameter: Optional[float] = Field(default=None, gt=0, description="Chamber inner diameter [m]")
    # Design parameters (optional, used for geometry generation)
    design_pressure: Optional[float] = Field(default=None, gt=0, description="Design chamber pressure [Pa] used for geometry calculation")
    design_thrust: Optional[float] = Field(default=None, gt=0, description="Design thrust [N] used for geometry calculation")
    design_force_coefficient: Optional[float] = Field(default=None, gt=0, description="Design force coefficient used for geometry calculation")


class NozzleConfig(BaseModel):
    """Nozzle configuration"""
    A_throat: float = Field(gt=0, description="Throat area [m²]")
    A_exit: float = Field(gt=0, description="Exit area [m²]")
    expansion_ratio: float = Field(gt=1, description="Expansion ratio (A_exit/A_throat)")
    exit_diameter: Optional[float] = Field(default=None, gt=0, description="Nozzle exit diameter [m]")
    efficiency: float = Field(default=0.98, ge=0, le=1, description="Nozzle efficiency")


class ClosureConfig(BaseModel):
    """Closure iteration configuration"""
    max_iterations: int = Field(default=6, gt=0, description="Max closure iterations")
    Cd_reduction_factor: float = Field(
        default=0.95,
        ge=0,
        le=1,
        description="Cd reduction factor if constraints violated"
    )
    tolerance: float = Field(default=1e-4, gt=0, description="Convergence tolerance")


class SolverConfig(BaseModel):
    """Solver configuration"""
    method: Literal["brentq", "secant", "newton"] = Field(
        default="brentq",
        description="Root finding method"
    )
    Pc_bounds: list[float] = Field(
        default=[100000.0, 8000000.0],
        description="Chamber pressure bounds [Pa]"
    )
    tolerance: float = Field(default=1e-6, gt=0, description="Root finding tolerance")
    max_iterations: int = Field(default=100, gt=0, description="Max iterations")
    closure: ClosureConfig = Field(default_factory=ClosureConfig)


InjectorConfig = Union[PintleInjectorConfig, CoaxialInjectorConfig, ImpingingInjectorConfig]


# Flight simulation configuration classes
class LOXTankConfig(BaseModel):
    """LOX tank geometry configuration for flight simulation"""
    lox_h: float = Field(gt=0, description="LOX tank height [m]")
    lox_radius: float = Field(gt=0, description="LOX tank radius [m]")
    ox_tank_pos: float = Field(description="Oxidizer tank position [m]")
    mass: Optional[float] = Field(default=None, gt=0, description="Initial LOX PROPELLANT mass [kg] (not tank structure). This contributes to total wet mass.")


class FuelTankConfig(BaseModel):
    """Fuel tank geometry configuration for flight simulation"""
    rp1_h: float = Field(gt=0, description="RP-1 tank height [m]")
    rp1_radius: float = Field(gt=0, description="RP-1 tank radius [m]")
    fuel_tank_pos: float = Field(description="Fuel tank position [m]")
    mass: Optional[float] = Field(default=None, gt=0, description="Initial RP-1 PROPELLANT mass [kg] (not tank structure). This contributes to total wet mass.")


class PressTankConfig(BaseModel):
    """Pressurant tank geometry configuration for flight simulation"""
    press_h: float = Field(gt=0, description="Pressurant tank height [m]")
    press_radius: float = Field(gt=0, description="Pressurant tank radius [m]")
    pres_tank_pos: float = Field(description="Pressurant tank position [m]")
    mass: Optional[float] = Field(default=None, gt=0, description="Initial fluid mass [kg] (for flight simulation)")


class FinsConfig(BaseModel):
    """Fins configuration for flight simulation"""
    no_fins: int = Field(gt=0, description="Number of fins")
    root_chord: float = Field(gt=0, description="Root chord [m]")
    tip_chord: float = Field(gt=0, description="Tip chord [m]")
    fin_span: float = Field(gt=0, description="Fin span [m]")
    fin_position: float = Field(description="Fin position [m]")


class MotorConfig(BaseModel):
    """Motor configuration for flight simulation (LEGACY - use propulsion_dry_mass instead)"""
    dry_mass: float = Field(gt=0, description="Motor dry mass [kg]")
    inertia: list[float] = Field(description="Motor inertia [kg·m²]")


class RocketConfig(BaseModel):
    """Rocket configuration for flight simulation.
    
    NEW Mass Model (recommended):
    - airframe_mass: Rocket body without propulsion (fuselage, fins, nosecone, avionics, payload)
    - engine_mass: Engine + ALL plumbing (chamber, nozzle, injector, valves, fittings, lines)
    - lox_tank_structure_mass: Empty LOX tank only (walls, no fittings)
    - fuel_tank_structure_mass: Empty fuel tank only (walls, no fittings)
    - engine_cm_offset: Height of engine CM above nozzle exit
    
    propulsion_dry_mass and propulsion_cm_offset are COMPUTED from the above.
    
    Total dry mass = airframe_mass + engine_mass + lox_tank_structure_mass + fuel_tank_structure_mass
    Total wet mass = dry mass + lox_tank.mass + fuel_tank.mass (propellants)
    
    LEGACY fields (mass, motor.dry_mass) still supported for backward compatibility.
    """
    # NEW detailed mass model
    airframe_mass: Optional[float] = Field(default=None, gt=0, description="Airframe mass (fuselage, fins, nosecone, avionics, payload) - NO propulsion [kg]")
    engine_mass: Optional[float] = Field(default=None, gt=0, description="Engine + plumbing mass (chamber, nozzle, injector, valves, ALL fittings & lines) [kg]")
    lox_tank_structure_mass: Optional[float] = Field(default=None, gt=0, description="Empty LOX tank only (walls, no fittings) [kg]")
    fuel_tank_structure_mass: Optional[float] = Field(default=None, gt=0, description="Empty fuel tank only (walls, no fittings) [kg]")
    engine_cm_offset: float = Field(default=0.15, ge=0, description="Height of engine+plumbing CM above nozzle exit [m]. Typical: 0.1-0.3m.")
    
    # COMPUTED fields (calculated from detailed breakdown)
    propulsion_dry_mass: Optional[float] = Field(default=None, gt=0, description="COMPUTED: Total propulsion dry mass (engine + tanks) [kg]")
    propulsion_cm_offset: float = Field(default=0.3, description="COMPUTED: Propulsion system CM above nozzle exit [m]")
    
    # Common parameters
    inertia: list[float] = Field(description="Total rocket inertia (dry, without propellants) [kg·m²]")
    radius: float = Field(gt=0, description="Rocket radius [m]")
    motor_position: float = Field(default=0.5, description="Nozzle exit position from rocket tail [m]")
    fins: Optional[FinsConfig] = Field(default=None, description="Fins configuration")
    
    # LEGACY fields - kept for backward compatibility
    mass: Optional[float] = Field(default=None, gt=0, description="LEGACY: Airframe mass. Use airframe_mass instead.")
    cm_wo_motor: Optional[float] = Field(default=None, description="LEGACY: CM without motor. Now auto-calculated from airframe_mass and propulsion positions.")
    dry_mass: Optional[float] = Field(default=None, gt=0, description="LEGACY: Unused field.")
    motor_inertia: Optional[list[float]] = Field(default=None, description="LEGACY: Motor inertia. Now estimated from propulsion_dry_mass.")
    motor: Optional[MotorConfig] = Field(default=None, description="LEGACY: Motor config. Use propulsion_dry_mass instead.")


class EnvironmentConfig(BaseModel):
    """Environment configuration for flight simulation"""
    date: list[int] = Field(description="Date [year, month, day, hour]")
    latitude: float = Field(ge=-90, le=90, description="Latitude [deg]")
    longitude: float = Field(ge=-180, le=180, description="Longitude [deg]")
    elevation: float = Field(description="Elevation [m]")
    p_amb: float = Field(gt=0, description="Ambient pressure [Pa]")


class ThrustConfig(BaseModel):
    """Thrust configuration for flight simulation"""
    burn_time: float = Field(gt=0, description="Burn time [s]")


class PintleEngineConfig(BaseModel):
    """Complete pintle engine configuration"""
    fluids: dict[str, FluidConfig]
    injector: InjectorConfig
    feed_system: dict[str, FeedSystemConfig]  # "oxidizer" and "fuel"
    regen_cooling: Optional[RegenCoolingConfig] = Field(default=None, description="Regenerative cooling configuration (fuel only)")
    film_cooling: Optional[FilmCoolingConfig] = Field(default=None, description="Film cooling configuration")
    ablative_cooling: Optional[AblativeCoolingConfig] = Field(default=None, description="Ablative cooling configuration for chamber liner")
    graphite_insert: Optional[GraphiteInsertConfig] = Field(default=None, description="Graphite throat insert configuration (separate from chamber ablator)")
    stainless_steel_case: Optional[StainlessSteelCaseConfig] = Field(default=None, description="Stainless steel case configuration (structural wall behind ablative/graphite)")
    discharge: dict[str, DischargeConfig]  # "oxidizer" and "fuel"
    spray: SprayConfig = Field(default_factory=SprayConfig)
    combustion: CombustionConfig = Field(default_factory=CombustionConfig)
    chamber: ChamberConfig
    nozzle: NozzleConfig
    solver: SolverConfig = Field(default_factory=SolverConfig)
    # Flight simulation fields (optional)
    lox_tank: Optional[LOXTankConfig] = Field(default=None, description="LOX tank configuration for flight simulation")
    fuel_tank: Optional[FuelTankConfig] = Field(default=None, description="Fuel tank configuration for flight simulation")
    press_tank: Optional[PressTankConfig] = Field(default=None, description="Pressurant tank configuration for flight simulation")
    rocket: Optional[RocketConfig] = Field(default=None, description="Rocket configuration for flight simulation")
    environment: Optional[EnvironmentConfig] = Field(default=None, description="Environment configuration for flight simulation")
    thrust: Optional[ThrustConfig] = Field(default=None, description="Thrust configuration for flight simulation")

    @field_validator("feed_system", "discharge")
    @classmethod
    def validate_branches(cls, v):
        """Ensure both oxidizer and fuel branches are present"""
        if "oxidizer" not in v or "fuel" not in v:
            raise ValueError("Must specify both 'oxidizer' and 'fuel' branches")
        return v

    class Config:
        extra = "allow"  # Reject unknown fields

