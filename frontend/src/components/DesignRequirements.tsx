import { useState, useEffect } from 'react';
import type { EngineConfig, DesignRequirements as DesignRequirementsType } from '../api/client';

interface DesignRequirementsProps {
  config: EngineConfig | null;
  onSave: (requirements: DesignRequirementsType) => void;
}

export function DesignRequirements({ config, onSave }: DesignRequirementsProps) {
  // Initialize with defaults or from config
  const [requirements, setRequirements] = useState<DesignRequirementsType>({
    // Performance targets
    target_thrust: 7000.0,
    target_apogee: 3048.0,
    optimal_of_ratio: 2.3,
    target_burn_time: 10.0,
    
    // Tank pressures
    max_lox_tank_pressure_psi: 700.0,
    max_fuel_tank_pressure_psi: 850.0,
    
    // Geometry constraints
    max_engine_length: 0.5,
    max_chamber_outer_diameter: 0.15,
    max_nozzle_exit_diameter: 0.101,
    
    // L* constraints
    min_Lstar: 0.95,
    max_Lstar: 1.27,
    
    // Stability requirements (new comprehensive analysis)
    min_stability_score: 0.75,
    require_stable_state: true,
    
    // Stability requirements (legacy margins)
    min_stability_margin: 1.2,
    chugging_margin_min: 0.2,
    acoustic_margin_min: 0.1,
    feed_stability_min: 0.15,
    
    // COPV
    copv_free_volume_L: 4.5,
  });

  // Auto-fill from config when available
  useEffect(() => {
    if (config?.design_requirements) {
      setRequirements(config.design_requirements as DesignRequirementsType);
    }
  }, [config]);

  const handleSave = () => {
    onSave(requirements);
  };

  const updateField = (field: keyof DesignRequirementsType, value: number | boolean) => {
    setRequirements(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl p-6">
        <h2 className="text-2xl font-bold text-[var(--color-text-primary)] mb-2">Design Requirements</h2>
        <p className="text-sm text-[var(--color-text-secondary)]">
          Configure your rocket and specify engine design targets. The optimizer will solve for propellant masses and engine geometry.
        </p>
      </div>

      {/* Performance Targets */}
      <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl p-6">
        <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">🎯 Performance Targets</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
              Target Peak Thrust [N]
            </label>
            <input
              type="number"
              value={requirements.target_thrust}
              onChange={(e) => updateField('target_thrust', parseFloat(e.target.value))}
              className="w-full px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="100"
              max="100000"
              step="100"
            />
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">Peak thrust during burn. Engine will be sized to achieve this.</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
              Target Apogee [m AGL]
            </label>
            <input
              type="number"
              value={requirements.target_apogee || 3048}
              onChange={(e) => updateField('target_apogee', parseFloat(e.target.value))}
              className="w-full px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="100"
              max="200000"
              step="100"
            />
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">Target altitude above ground level. Optimizer will solve for propellant masses.</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
              Optimal O/F Ratio
            </label>
            <input
              type="number"
              value={requirements.optimal_of_ratio}
              onChange={(e) => updateField('optimal_of_ratio', parseFloat(e.target.value))}
              className="w-full px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="1.5"
              max="4.0"
              step="0.1"
            />
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">Target oxidizer-to-fuel mixture ratio. LOX/RP-1 optimal: 2.4-2.8 for Isp, 2.2-2.5 for stability.</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
              Target Burn Time [s]
            </label>
            <input
              type="number"
              value={requirements.target_burn_time}
              onChange={(e) => updateField('target_burn_time', parseFloat(e.target.value))}
              className="w-full px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="1"
              max="60"
              step="1"
            />
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">Design burn time. Flight sim will truncate if propellant depletes earlier.</p>
          </div>
        </div>
      </div>

      {/* Tank Pressures */}
      <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl p-6">
        <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">🔋 Tank Pressures</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
              Max LOX Tank Pressure [psi]
            </label>
            <input
              type="number"
              value={requirements.max_lox_tank_pressure_psi}
              onChange={(e) => updateField('max_lox_tank_pressure_psi', parseFloat(e.target.value))}
              className="w-full px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="100"
              max="5000"
              step="25"
            />
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">Maximum operating pressure in LOX tank. Sets upper bound for chamber pressure.</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
              Max Fuel Tank Pressure [psi]
            </label>
            <input
              type="number"
              value={requirements.max_fuel_tank_pressure_psi}
              onChange={(e) => updateField('max_fuel_tank_pressure_psi', parseFloat(e.target.value))}
              className="w-full px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="100"
              max="5000"
              step="25"
            />
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">Maximum operating pressure in fuel tank.</p>
          </div>
        </div>
      </div>

      {/* Geometry Constraints */}
      <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl p-6">
        <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">📏 Geometry Constraints</h3>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
              Max Engine Length [m]
            </label>
            <input
              type="number"
              value={requirements.max_engine_length}
              onChange={(e) => updateField('max_engine_length', parseFloat(e.target.value))}
              className="w-full px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="0.1"
              max="3.0"
              step="0.05"
            />
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">Maximum total engine length (chamber + nozzle). Must fit in vehicle.</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
              Max Chamber OD [m]
            </label>
            <input
              type="number"
              value={requirements.max_chamber_outer_diameter}
              onChange={(e) => updateField('max_chamber_outer_diameter', parseFloat(e.target.value))}
              className="w-full px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="0.05"
              max="1.0"
              step="0.01"
            />
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">Maximum chamber outer diameter (including wall thickness and cooling jacket).</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
              Max Nozzle Exit Diameter [m]
            </label>
            <input
              type="number"
              value={requirements.max_nozzle_exit_diameter}
              onChange={(e) => updateField('max_nozzle_exit_diameter', parseFloat(e.target.value))}
              className="w-full px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="0.05"
              max="1.0"
              step="0.01"
            />
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">Maximum nozzle exit outer diameter. Constrains expansion ratio.</p>
          </div>
        </div>
      </div>

      {/* L* Constraints */}
      <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl p-6">
        <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">📐 L* (Characteristic Length) Constraints</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
              Minimum L* [m]
            </label>
            <input
              type="number"
              value={requirements.min_Lstar}
              onChange={(e) => updateField('min_Lstar', parseFloat(e.target.value))}
              className="w-full px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="0.5"
              max="3.0"
              step="0.1"
            />
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">Minimum characteristic length. Lower = smaller chamber but less complete combustion. Typical: 0.8-1.0m for LOX/RP-1.</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
              Maximum L* [m]
            </label>
            <input
              type="number"
              value={requirements.max_Lstar}
              onChange={(e) => updateField('max_Lstar', parseFloat(e.target.value))}
              className="w-full px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="0.5"
              max="3.0"
              step="0.1"
            />
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">Maximum characteristic length. Higher = better combustion but heavier/longer chamber. Typical: 1.5-2.0m for LOX/RP-1.</p>
          </div>
        </div>
      </div>

      {/* Stability Requirements */}
      <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl p-6">
        <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">🛡️ Stability Requirements</h3>
        
        <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 mb-4">
          <p className="text-sm text-blue-400">
            <strong>New Comprehensive Stability Analysis:</strong><br/>
            • Uses stability_score (0-1) and stability_state ("stable"/"marginal"/"unstable")<br/>
            • Considers chugging, acoustic modes, feed system, and mode coupling<br/>
            • <strong>Stable</strong>: score ≥ 0.75 (recommended for flight)<br/>
            • <strong>Marginal</strong>: 0.4 ≤ score &lt; 0.75 (acceptable with caution)<br/>
            • <strong>Unstable</strong>: score &lt; 0.4 (not acceptable)
          </p>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
              Minimum Stability Score
            </label>
            <input
              type="number"
              value={requirements.min_stability_score}
              onChange={(e) => updateField('min_stability_score', parseFloat(e.target.value))}
              className="w-full px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="0.0"
              max="1.0"
              step="0.05"
            />
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">Minimum stability score (0-1). 0.75 = 'stable', 0.4 = 'marginal', &lt;0.4 = 'unstable'</p>
          </div>

          <div className="flex items-center">
            <input
              type="checkbox"
              checked={requirements.require_stable_state}
              onChange={(e) => updateField('require_stable_state', e.target.checked)}
              className="w-4 h-4 text-blue-600 bg-[var(--color-bg-primary)] border-[var(--color-border)] rounded focus:ring-blue-500"
            />
            <label className="ml-2 text-sm text-[var(--color-text-primary)]">
              Require 'Stable' State (not just 'Marginal')
            </label>
          </div>

          {/* Legacy Margins */}
          <details className="mt-4">
            <summary className="cursor-pointer text-sm font-medium text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]">
              Individual Stability Margins (for detailed tracking)
            </summary>
            <div className="mt-4 space-y-3 pl-4 border-l-2 border-[var(--color-border)]">
              <p className="text-xs text-[var(--color-text-secondary)]">These are used for detailed feedback but the optimizer primarily uses stability_score above.</p>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
                    Min Overall Stability Margin (legacy)
                  </label>
                  <input
                    type="number"
                    value={requirements.min_stability_margin}
                    onChange={(e) => updateField('min_stability_margin', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500"
                    min="1.0"
                    max="5.0"
                    step="0.1"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
                    Chugging Margin (min)
                  </label>
                  <input
                    type="number"
                    value={requirements.chugging_margin_min}
                    onChange={(e) => updateField('chugging_margin_min', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500"
                    min="0.0"
                    max="10.0"
                    step="0.1"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
                    Acoustic Margin (min)
                  </label>
                  <input
                    type="number"
                    value={requirements.acoustic_margin_min}
                    onChange={(e) => updateField('acoustic_margin_min', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500"
                    min="0.0"
                    max="10.0"
                    step="0.1"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
                    Feed System Margin (min)
                  </label>
                  <input
                    type="number"
                    value={requirements.feed_stability_min}
                    onChange={(e) => updateField('feed_stability_min', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500"
                    min="0.0"
                    max="10.0"
                    step="0.1"
                  />
                </div>
              </div>
            </div>
          </details>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex justify-end">
        <button
          onClick={handleSave}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
        >
          Save Design Requirements
        </button>
      </div>

      {/* Summary */}
      <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-green-400 mb-4">📋 Design Summary</h3>
        <div className="grid grid-cols-4 gap-4">
          <div>
            <p className="text-xs text-[var(--color-text-secondary)]">Target Thrust</p>
            <p className="text-lg font-bold text-[var(--color-text-primary)]">{requirements.target_thrust.toFixed(0)} N</p>
          </div>
          <div>
            <p className="text-xs text-[var(--color-text-secondary)]">Target Apogee</p>
            <p className="text-lg font-bold text-[var(--color-text-primary)]">{(requirements.target_apogee || 0).toFixed(0)} m</p>
          </div>
          <div>
            <p className="text-xs text-[var(--color-text-secondary)]">Optimal O/F</p>
            <p className="text-lg font-bold text-[var(--color-text-primary)]">{requirements.optimal_of_ratio.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-[var(--color-text-secondary)]">Burn Time</p>
            <p className="text-lg font-bold text-[var(--color-text-primary)]">{requirements.target_burn_time.toFixed(1)} s</p>
          </div>
          <div>
            <p className="text-xs text-[var(--color-text-secondary)]">Max LOX Pressure</p>
            <p className="text-lg font-bold text-[var(--color-text-primary)]">{requirements.max_lox_tank_pressure_psi.toFixed(0)} psi</p>
          </div>
          <div>
            <p className="text-xs text-[var(--color-text-secondary)]">Max Fuel Pressure</p>
            <p className="text-lg font-bold text-[var(--color-text-primary)]">{requirements.max_fuel_tank_pressure_psi.toFixed(0)} psi</p>
          </div>
          <div>
            <p className="text-xs text-[var(--color-text-secondary)]">L* Range</p>
            <p className="text-lg font-bold text-[var(--color-text-primary)]">{requirements.min_Lstar.toFixed(2)} - {requirements.max_Lstar.toFixed(2)} m</p>
          </div>
          <div>
            <p className="text-xs text-[var(--color-text-secondary)]">Max Engine Length</p>
            <p className="text-lg font-bold text-[var(--color-text-primary)]">{(requirements.max_engine_length * 1000).toFixed(0)} mm</p>
          </div>
        </div>
      </div>
    </div>
  );
}

