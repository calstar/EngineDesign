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
