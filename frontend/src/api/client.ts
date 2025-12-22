/**
 * API client for communicating with the FastAPI backend.
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

// Evaluation types
export interface EvaluateRequest {
  lox_pressure_psi: number;
  fuel_pressure_psi: number;
  ambient_pressure_pa?: number;
}

export interface PerformanceMetrics {
  thrust_N: number;
  thrust_kN: number;
  Isp_s: number;
  chamber_pressure_Pa: number;
  chamber_pressure_psi: number;
  mdot_total_kg_s: number;
  mdot_oxidizer_kg_s: number;
  mdot_fuel_kg_s: number;
  mixture_ratio: number;
  cstar_actual_m_s: number;
  exit_velocity_m_s: number;
  exit_pressure_Pa: number;
}

export interface EvaluateResponse {
  status: string;
  inputs: {
    lox_pressure_psi: number;
    fuel_pressure_psi: number;
    ambient_pressure_pa: number;
  };
  performance: PerformanceMetrics;
  full_results: Record<string, unknown>;
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

