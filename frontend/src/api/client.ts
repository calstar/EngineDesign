/**
 * API client for communicating with the FastAPI backend.
 * 
 * The evaluate endpoint returns results in the same format as runner.evaluate() 
 * from the Python engine - keeping consistency with the Streamlit UI.
 */

const API_BASE = '/api';

interface ApiResponse<T> {
  data?: T;
  error?: string;
}

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<ApiResponse<T>> {
  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return { error: errorData.detail || `HTTP ${response.status}: ${response.statusText}` };
    }

    const data = await response.json();
    return { data };
  } catch (err) {
    return { error: err instanceof Error ? err.message : 'Network error' };
  }
}

// Config types
export interface EngineConfig {
  fluids: Record<string, unknown>;
  injector: Record<string, unknown>;
  feed_system: Record<string, unknown>;
  combustion: Record<string, unknown>;
  chamber: Record<string, unknown>;
  nozzle: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ConfigResponse {
  config: EngineConfig;
}

export interface UploadResponse {
  status: string;
  message: string;
  config: EngineConfig;
}

// Evaluation types - matches runner.evaluate() output from engine/core/runner.py
export interface EvaluateRequest {
  lox_pressure_psi: number;
  fuel_pressure_psi: number;
}

// Runner results - same field names as runner.evaluate() returns
export interface RunnerResults {
  // Primary performance
  Pc: number;              // Chamber pressure [Pa]
  F: number;               // Thrust [N]
  Isp: number;             // Specific impulse [s]
  mdot_O: number;          // Oxidizer mass flow [kg/s]
  mdot_F: number;          // Fuel mass flow [kg/s]
  mdot_total: number;      // Total mass flow [kg/s]
  MR: number;              // Mixture ratio (O/F)
  
  // Nozzle performance
  v_exit: number;          // Exit velocity [m/s]
  M_exit: number;          // Exit Mach number
  P_exit: number;          // Exit pressure [Pa]
  P_throat: number;        // Throat pressure [Pa]
  
  // Temperatures
  Tc: number;              // Chamber temperature [K]
  T_throat: number;        // Throat temperature [K]
  T_exit: number;          // Exit temperature [K]
  
  // Thrust coefficients
  Cf: number;              // Thrust coefficient (actual)
  Cf_actual: number;       // Thrust coefficient (actual)
  Cf_ideal: number;        // Thrust coefficient (ideal)
  Cf_theoretical: number;  // Thrust coefficient (theoretical)
  
  // Characteristic velocity
  cstar_actual: number;    // Actual c* [m/s]
  cstar_ideal: number;     // Ideal c* [m/s]
  eta_cstar: number;       // c* efficiency
  
  // Thermodynamic properties
  gamma: number;           // Ratio of specific heats (chamber)
  gamma_exit: number;      // Ratio of specific heats (exit)
  R: number;               // Gas constant (chamber) [J/(kg·K)]
  R_exit: number;          // Gas constant (exit) [J/(kg·K)]
  
  // Geometry
  eps: number;             // Expansion ratio
  A_throat: number;        // Throat area [m²]
  A_exit: number;          // Exit area [m²]
  
  // Discharge coefficients
  Cd_O: number;            // Oxidizer discharge coefficient
  Cd_F: number;            // Fuel discharge coefficient
  
  // Chamber intrinsics
  chamber_intrinsics: {
    Lstar?: number;           // Characteristic length [m]
    residence_time?: number;  // Residence time [s]
    velocity_mean?: number;   // Mean velocity [m/s]
    velocity_throat?: number; // Throat velocity [m/s]
    mach_number?: number;     // Mach number in chamber
    reynolds_number?: number; // Reynolds number
    density?: number;         // Gas density [kg/m³]
    sound_speed?: number;     // Sound speed [m/s]
  } | null;
  
  // Injector pressure diagnostics
  injector_pressure: {
    P_injector_O?: number;      // Injector pressure, oxidizer [Pa]
    P_injector_F?: number;      // Injector pressure, fuel [Pa]
    delta_p_injector_O?: number; // Injector pressure drop, oxidizer [Pa]
    delta_p_injector_F?: number; // Injector pressure drop, fuel [Pa]
    delta_p_feed_O?: number;     // Feed pressure drop, oxidizer [Pa]
    delta_p_feed_F?: number;     // Feed pressure drop, fuel [Pa]
  } | null;
  
  // Cooling results
  cooling: {
    regen?: {
      enabled: boolean;
      coolant_outlet_temperature?: number;
      heat_removed?: number;
      overall_heat_flux?: number;
      mdot_coolant?: number;
      wall_temperature_hot?: number;
      wall_temperature_coolant?: number;
    };
    film?: {
      enabled: boolean;
      mass_fraction?: number;
      effectiveness?: number;
      mdot_film?: number;
      heat_flux_factor?: number;
      blowing_ratio?: number;
    };
    ablative?: {
      enabled: boolean;
      recession_rate?: number;
      effective_heat_flux?: number;
      cooling_power?: number;
      heat_removed?: number;
      incident_heat_flux?: number;
      below_pyrolysis?: boolean;
    };
  } | null;
  
  // Stability analysis
  stability: {
    is_stable: boolean;
    stability_state: string;
    stability_score: number;
    chugging: {
      frequency?: number;
      stability_margin?: number;
      stability_index?: number;
      period?: number;
      tau_residence?: number;
      Lstar?: number;
    };
    acoustic: {
      stability_margin?: number;
      modes?: Record<string, number>;
      longitudinal_modes?: number[];
      transverse_modes?: number[];
      sound_speed?: number;
    };
    feed_system: {
      pogo_frequency?: number;
      surge_frequency?: number;
      water_hammer_margin?: number;
      stability_margin?: number;
      sound_speed?: number;
    };
    issues: string[];
    recommendations: string[];
  } | null;
  
  // Profiles (optional, for plotting)
  pressure_profile?: unknown;
  temperature_profile?: unknown;
  
  // Ambient conditions (computed from config elevation)
  P_ambient?: number;       // Ambient pressure used [Pa]
  elevation?: number;       // Elevation from config [m]
  
  // Full diagnostics
  diagnostics: Record<string, unknown>;
}

export interface EvaluateResponse {
  status: string;
  inputs: {
    lox_pressure_psi: number;
    fuel_pressure_psi: number;
    ambient_pressure_pa: number;  // Computed from elevation
    elevation_m: number;          // Elevation from config
  };
  results: RunnerResults;
}

// API functions
export async function uploadConfig(file: File): Promise<ApiResponse<UploadResponse>> {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_BASE}/config/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return { error: errorData.detail || `HTTP ${response.status}` };
    }

    const data = await response.json();
    return { data };
  } catch (err) {
    return { error: err instanceof Error ? err.message : 'Upload failed' };
  }
}

export async function getConfig(): Promise<ApiResponse<ConfigResponse>> {
  return request<ConfigResponse>('/config');
}

export async function updateConfig(updates: Partial<EngineConfig>): Promise<ApiResponse<ConfigResponse>> {
  return request<ConfigResponse>('/config', {
    method: 'PUT',
    body: JSON.stringify(updates),
  });
}

export async function evaluate(params: EvaluateRequest): Promise<ApiResponse<EvaluateResponse>> {
  return request<EvaluateResponse>('/evaluate', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function getHealth(): Promise<ApiResponse<{ status: string; config_loaded: boolean }>> {
  return request('/health');
}


// ============================================================================
// Time-Series Types and API
// ============================================================================

export type ProfileType = 'linear' | 'exponential' | 'power';
export type SegmentType = 'blowdown' | 'linear';

// Profile parameters for simple generation
export interface ProfileParams {
  start_pressure_psi: number;
  end_pressure_psi: number;
  profile_type: ProfileType;
  decay_constant?: number;  // For exponential
  power?: number;           // For power
}

// Request for simple profile generation
export interface GenerateProfileRequest {
  duration_s: number;
  n_steps: number;
  lox_profile: ProfileParams;
  fuel_profile: ProfileParams;
}

// Pressure segment for segment-based curve building
export interface PressureSegment {
  length_ratio: number;      // Fraction of total time (0-1)
  type: SegmentType;
  start_pressure_psi: number;
  end_pressure_psi: number;
  k: number;                 // Blowdown decay constant (0.1-3.0)
}

// Request for segment-based generation
export interface SegmentsRequest {
  duration_s: number;
  n_points: number;
  lox_segments: PressureSegment[];
  fuel_segments: PressureSegment[];
}

// Time-series data returned from the API
export interface TimeSeriesData {
  time: number[];
  P_tank_O_psi: number[];
  P_tank_F_psi: number[];
  Pc_psi: number[];
  thrust_kN: number[];
  Isp_s: number[];
  MR: number[];
  mdot_O_kg_s: number[];
  mdot_F_kg_s: number[];
  mdot_total_kg_s: number[];
  cstar_actual_m_s: number[];
  gamma: number[];
  Cd_O?: number[];
  Cd_F?: number[];
  // Optional fields for additional plots
  delta_P_injector_O_psi?: number[];
  delta_P_injector_F_psi?: number[];
  Lstar_mm?: number[];
  recession_rate_ablative_um_s?: number[];
  recession_rate_graphite_thermal_um_s?: number[];
  recession_rate_graphite_oxidation_um_s?: number[];
  recession_cumulative_ablative_mm?: number[];
  recession_cumulative_graphite_thermal_mm?: number[];
  recession_cumulative_graphite_oxidation_mm?: number[];
  recession_cumulative_chamber_um?: number[];
  recession_cumulative_throat_um?: number[];
  V_chamber_m3?: number[];
  A_throat_m2?: number[];
  V_chamber_initial_m3?: number;
  A_throat_initial_m2?: number;
  // COPV pressure trace
  copv_pressure_psi?: number[];
  // Correlation matrix data
  correlation_matrix?: number[][];
  correlation_labels?: string[];
}

// Summary statistics
export interface TimeSeriesSummary {
  avg_thrust_kN: number;
  peak_thrust_kN: number;
  min_thrust_kN: number;
  avg_Pc_psi: number;
  peak_Pc_psi: number;
  avg_Isp_s: number;
  total_impulse_kNs: number;
  total_propellant_kg: number;
  burn_time_s: number;
  // COPV summary metrics
  copv_initial_pressure_psi?: number;
  copv_initial_mass_kg?: number;
  copv_min_margin_psi?: number;
  copv_volume_L?: number;
}

// Response for generate endpoint
export interface GenerateProfileResponse {
  status: string;
  data: TimeSeriesData;
  summary: TimeSeriesSummary;
}

// Response for segments endpoint
export interface SegmentsResponse {
  status: string;
  data: TimeSeriesData;
  summary: TimeSeriesSummary;
  lox_curve_preview: number[];
  fuel_curve_preview: number[];
}

// Preview request for real-time curve visualization
export interface PreviewCurveRequest {
  n_points: number;
  segments: PressureSegment[];
}

// Preview response
export interface PreviewCurveResponse {
  curve_psi: number[];
  normalized_time: number[];
}

// Time-series API functions
export async function generateTimeseries(
  params: GenerateProfileRequest
): Promise<ApiResponse<GenerateProfileResponse>> {
  return request<GenerateProfileResponse>('/timeseries/generate', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function generateFromSegments(
  params: SegmentsRequest
): Promise<ApiResponse<SegmentsResponse>> {
  return request<SegmentsResponse>('/timeseries/from-segments', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function previewCurve(
  params: PreviewCurveRequest
): Promise<ApiResponse<PreviewCurveResponse>> {
  return request<PreviewCurveResponse>('/timeseries/preview-curve', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}


// ============================================================================
// Flight Simulation Types and API
// ============================================================================

export interface FlightEnvironmentConfig {
  latitude: number;
  longitude: number;
  elevation: number;
  date: [number, number, number, number]; // [year, month, day, hour]
}

export interface FlightFinsConfig {
  no_fins: number;
  root_chord: number;
  tip_chord: number;
  fin_span: number;
  fin_position: number;
}

export interface FlightRocketConfig {
  airframe_mass: number;
  engine_mass: number;
  lox_tank_structure_mass: number;
  fuel_tank_structure_mass: number;
  radius: number;
  rocket_length: number;
  motor_position: number;
  inertia: [number, number, number]; // [Ixx, Iyy, Izz]
  fins?: FlightFinsConfig;
}

export interface FlightTankConfig {
  mass: number;
  height: number;
  radius: number;
  position: number;
}

export type FlightSourceType = 'timeseries';

export interface FlightSimRequest {
  // Time-series data (required)
  time_array: number[];
  thrust_array: number[];
  mdot_O_array: number[];
  mdot_F_array: number[];
  
  // Propellant configuration
  lox_mass_kg: number;
  fuel_mass_kg: number;
  
  // Tank geometry (optional)
  lox_tank?: FlightTankConfig;
  fuel_tank?: FlightTankConfig;
  
  // Environment configuration
  environment?: FlightEnvironmentConfig;
  
  // Rocket configuration
  rocket?: FlightRocketConfig;
}

export interface FlightTrajectory {
  time: number[];
  altitude: number[];
  velocity: number[];
}

export interface FlightTruncationInfo {
  truncated: boolean;
  cutoff_time?: number;
  reason?: string;
}

export interface FlightSimResponse {
  status: string;
  apogee_m: number;
  apogee_ft: number;
  max_velocity_m_s: number;
  flight_time_s: number;
  trajectory?: FlightTrajectory;
  truncation?: FlightTruncationInfo;
  thrust_curve?: {
    time: number[];
    thrust_N: number[];
  };
  rocket_diagram?: string;  // Base64-encoded PNG
  error?: string;
}

export interface RocketPyCheckResponse {
  available: boolean;
  message: string;
  install_hint?: string;
}

// Flight simulation API functions
export async function runFlightSimulation(
  params: FlightSimRequest
): Promise<ApiResponse<FlightSimResponse>> {
  return request<FlightSimResponse>('/flight/simulate', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function checkRocketPy(): Promise<ApiResponse<RocketPyCheckResponse>> {
  return request<RocketPyCheckResponse>('/flight/check');
}


// ============================================================================
// Chamber Geometry Types and API
// ============================================================================

export interface ChamberGeometryResponse {
  positions: number[];
  R_gas: number[];
  R_ablative_outer: number[];
  R_graphite_outer: number[];
  R_stainless: number[];
  throat_position: number;
  graphite_start: number;
  graphite_end: number;
  D_chamber: number;
  D_throat: number;
  D_exit: number;
  L_chamber: number;
  L_nozzle: number;
  expansion_ratio: number;
  ablative_enabled: boolean;
  graphite_enabled: boolean;
  // Rao bell nozzle contour
  nozzle_x: number[];
  nozzle_y: number[];
  nozzle_method: string;
  // Chamber contour from CEA solver
  chamber_contour_x: number[];
  chamber_contour_y: number[];
  // Solver results
  Cf: number | null;
  Cf_ideal: number | null;
  A_throat_solved: number | null;
}

export async function getChamberGeometry(): Promise<ApiResponse<ChamberGeometryResponse>> {
  return request<ChamberGeometryResponse>('/geometry');
}
