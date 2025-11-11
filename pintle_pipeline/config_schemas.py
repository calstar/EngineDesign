"""Pydantic schemas for YAML/JSON configuration validation"""

from pydantic import BaseModel, Field, field_validator
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


class AblativeCoolingConfig(BaseModel):
    """Ablative cooling configuration"""
    enabled: bool = Field(default=False, description="Enable ablative cooling model")
    material_density: float = Field(default=1600.0, gt=0, description="Ablator density [kg/m³]")
    heat_of_ablation: float = Field(default=2.5e6, gt=0, description="Effective heat of ablation [J/kg]")
    thermal_conductivity: float = Field(default=0.35, gt=0, description="Ablator thermal conductivity [W/(m·K)]")
    specific_heat: float = Field(default=1500.0, gt=0, description="Ablator specific heat [J/(kg·K)]")
    initial_thickness: float = Field(default=0.01, gt=0, description="Initial ablative thickness [m]")
    surface_temperature_limit: float = Field(default=1200.0, gt=0, description="Allowable surface temperature [K]")
    coverage_fraction: float = Field(default=1.0, gt=0, le=1.0, description="Fraction of chamber surface protected by ablative liner")
    pyrolysis_temperature: float = Field(default=900.0, gt=0, description="Characteristic pyrolysis temperature of ablator [K]")
    blowing_efficiency: float = Field(default=0.8, ge=0, le=1, description="Effectiveness of ablative gases in blocking convective heat flux")
    turbulence_reference_intensity: float = Field(default=0.08, gt=0, description="Reference turbulence intensity for ablative augmentation")
    turbulence_sensitivity: float = Field(default=1.5, ge=0, description="Sensitivity of ablative heat flux to turbulence")
    turbulence_exponent: float = Field(default=1.0, gt=0, description="Exponent on turbulence intensity for ablative response")
    turbulence_max_multiplier: float = Field(default=3.0, gt=0, description="Maximum multiplier applied to convective heat flux due to turbulence")


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
    """CEA (Chemical Equilibrium Analysis) configuration"""
    ox_name: str = Field(default="LOX", description="Oxidizer name")
    fuel_name: str = Field(default="RP-1", description="Fuel name")
    expansion_ratio: float = Field(gt=1, description="Nozzle expansion ratio")
    cache_file: str = Field(default="cea_cache_LOX_RP1.npz", description="Cache filename")
    Pc_range: list[float] = Field(
        default=[2.0e6, 9.0e6],
        description="Chamber pressure range [Pa]"
    )
    MR_range: list[float] = Field(
        default=[2.0, 2.8],
        description="Mixture ratio range"
    )
    n_points: int = Field(default=200, gt=0, description="Number of grid points")


class CombustionEfficiencyConfig(BaseModel):
    """Combustion efficiency (L* correction) configuration"""
    model: Literal["exponential", "constant", "linear"] = Field(
        default="exponential",
        description="Efficiency model type"
    )
    C: float = Field(default=0.3, ge=0, le=1, description="Efficiency loss parameter")
    K: float = Field(default=0.15, gt=0, description="Recovery rate parameter")
    use_spray_correction: bool = Field(
        default=False,
        description="Adjust efficiency based on spray quality"
    )
    spray_penalty_factor: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="Penalty factor if spray constraints violated"
    )
    use_mixture_coupling: bool = Field(
        default=False,
        description="Couple droplet metrics to combustion efficiency"
    )
    target_smd_microns: float = Field(
        default=50.0,
        gt=0,
        description="Target SMD for full efficiency [µm]"
    )
    smd_penalty_exponent: float = Field(
        default=1.5,
        gt=0,
        description="Exponent controlling how strongly large droplets penalize efficiency"
    )
    xstar_limit_mm: float = Field(
        default=50.0,
        gt=0,
        description="Target evaporation length [mm]"
    )
    xstar_penalty_exponent: float = Field(
        default=1.0,
        gt=0,
        description="Exponent for evaporation length penalty"
    )
    we_reference: float = Field(
        default=20.0,
        gt=0,
        description="Reference Weber number for good atomization"
    )
    we_penalty_exponent: float = Field(
        default=1.0,
        gt=0,
        description="Exponent for Weber-number-based penalty"
    )
    mixture_efficiency_floor: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="Minimum mixture efficiency multiplier"
    )
    use_cooling_coupling: bool = Field(
        default=False,
        description="Couple cooling heat removal to combustion efficiency"
    )
    cooling_efficiency_floor: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="Minimum cooling efficiency multiplier"
    )
    use_turbulence_coupling: bool = Field(
        default=False,
        description="Couple injector/chamber turbulence intensity to combustion efficiency"
    )
    target_turbulence_intensity: float = Field(
        default=0.08,
        gt=0,
        description="Target turbulence intensity for full mixing efficiency"
    )
    turbulence_penalty_exponent: float = Field(
        default=1.0,
        ge=0,
        description="Exponent for turbulence-based efficiency modifier"
    )
    turbulence_efficiency_floor: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Minimum efficiency multiplier attributable to turbulence effects"
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


class NozzleConfig(BaseModel):
    """Nozzle configuration"""
    A_throat: float = Field(gt=0, description="Throat area [m²]")
    A_exit: float = Field(gt=0, description="Exit area [m²]")
    expansion_ratio: float = Field(gt=1, description="Expansion ratio (A_exit/A_throat)")
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
    mass: Optional[float] = Field(default=None, gt=0, description="Initial fluid mass [kg] (for flight simulation)")


class FuelTankConfig(BaseModel):
    """Fuel tank geometry configuration for flight simulation"""
    rp1_h: float = Field(gt=0, description="RP-1 tank height [m]")
    rp1_radius: float = Field(gt=0, description="RP-1 tank radius [m]")
    fuel_tank_pos: float = Field(description="Fuel tank position [m]")
    mass: Optional[float] = Field(default=None, gt=0, description="Initial fluid mass [kg] (for flight simulation)")


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
    """Motor configuration for flight simulation"""
    dry_mass: float = Field(gt=0, description="Motor dry mass [kg]")
    inertia: list[float] = Field(description="Motor inertia [kg·m²]")


class RocketConfig(BaseModel):
    """Rocket configuration for flight simulation"""
    mass: float = Field(gt=0, description="Rocket mass [kg]")
    inertia: list[float] = Field(description="Rocket inertia [kg·m²]")
    radius: float = Field(gt=0, description="Rocket radius [m]")
    cm_wo_motor: float = Field(description="Center of mass without motor [m]")
    dry_mass: float = Field(gt=0, description="Dry mass [kg]")
    motor_inertia: list[float] = Field(description="Motor inertia [kg·m²]")
    fins: Optional[FinsConfig] = Field(default=None, description="Fins configuration")
    motor: Optional[MotorConfig] = Field(default=None, description="Motor configuration")


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
    ablative_cooling: Optional[AblativeCoolingConfig] = Field(default=None, description="Ablative cooling configuration")
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

