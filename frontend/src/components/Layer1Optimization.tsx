import { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { 
  runLayer1Optimization, 
  getLayer1Status
} from '../api/client';
import type {
  Layer1Settings,
  Layer1ProgressEvent,
  Layer1Results,
  DesignRequirements
} from '../api/client';

interface Layer1OptimizationProps {
  requirements: DesignRequirements | null;
}

// Helper component for result cards
function ResultCard({ 
  label, 
  value, 
  unit, 
  decimals = 2, 
  color = 'blue',
  isText = false
}: { 
  label: string; 
  value: number | string | undefined; 
  unit?: string; 
  decimals?: number;
  color?: string;
  isText?: boolean;
}) {
  const colorClasses: Record<string, string> = {
    blue: 'bg-blue-500/10 border-blue-500/30',
    green: 'bg-green-500/10 border-green-500/30',
    yellow: 'bg-yellow-500/10 border-yellow-500/30',
    red: 'bg-red-500/10 border-red-500/30',
    purple: 'bg-purple-500/10 border-purple-500/30',
    orange: 'bg-orange-500/10 border-orange-500/30',
    cyan: 'bg-cyan-500/10 border-cyan-500/30',
    pink: 'bg-pink-500/10 border-pink-500/30',
    indigo: 'bg-indigo-500/10 border-indigo-500/30',
  };
  
  const textColorClasses: Record<string, string> = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    yellow: 'text-yellow-400',
    red: 'text-red-400',
    purple: 'text-purple-400',
    orange: 'text-orange-400',
    cyan: 'text-cyan-400',
    pink: 'text-pink-400',
    indigo: 'text-indigo-400',
  };
  
  const displayValue = isText 
    ? String(value || '-')
    : typeof value === 'number' 
      ? value.toFixed(decimals) 
      : value !== undefined && value !== null 
        ? String(value) 
        : '-';
  
  return (
    <div className={`rounded-lg p-3 border ${colorClasses[color] || colorClasses.blue}`}>
      <p className="text-xs text-[var(--color-text-secondary)] mb-1">{label}</p>
      <p className={`text-lg font-bold ${textColorClasses[color] || textColorClasses.blue}`}>
        {displayValue}
        {unit && <span className="text-sm font-normal text-[var(--color-text-secondary)] ml-1">{unit}</span>}
      </p>
    </div>
  );
}

// Helper component for validation cards
function ValidationCard({ label, passed }: { label: string; passed: boolean | undefined }) {
  const isPassed = passed === true;
  return (
    <div className={`rounded-lg p-3 border ${isPassed ? 'bg-green-500/10 border-green-500/30' : 'bg-red-500/10 border-red-500/30'}`}>
      <p className="text-xs text-[var(--color-text-secondary)] mb-1">{label}</p>
      <p className={`text-lg font-bold ${isPassed ? 'text-green-400' : 'text-red-400'}`}>
        {isPassed ? '✓ PASS' : '✗ FAIL'}
      </p>
    </div>
  );
}

export function Layer1Optimization({ requirements }: Layer1OptimizationProps) {
  const [settings, setSettings] = useState<Layer1Settings>({
    max_iterations: 80,
    thrust_tolerance: 0.1, // 10%
  });

  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState('');
  const [message, setMessage] = useState('');
  const [results, setResults] = useState<Layer1Results | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [objectiveHistory, setObjectiveHistory] = useState<Array<{
    iteration: number;
    objective: number;
    best_objective: number;
  }>>([]);

  // Check status on mount
  useEffect(() => {
    checkStatus();
  }, []);

  const checkStatus = async () => {
    const response = await getLayer1Status();
    if (response.data) {
      setIsRunning(response.data.running);
      setProgress(response.data.progress);
      setStage(response.data.stage);
      setMessage(response.data.message);
      if (response.data.error) {
        setError(response.data.error);
      }
    }
  };

  const handleRun = () => {
    if (!requirements) {
      setError('Please save design requirements first.');
      return;
    }

    setIsRunning(true);
    setProgress(0);
    setStage('Initializing');
    setMessage('Starting Layer 1 optimization...');
    setError(null);
    setResults(null);
    setObjectiveHistory([]);

    const eventSource = runLayer1Optimization(
      settings,
      (event: Layer1ProgressEvent) => {
        if (event.type === 'status' || event.type === 'progress') {
          if (event.progress !== undefined) setProgress(event.progress);
          if (event.stage) setStage(event.stage);
          if (event.message) setMessage(event.message);
        } else if (event.type === 'objective') {
          // Handle real-time objective updates
          if (event.objective_history && Array.isArray(event.objective_history)) {
            setObjectiveHistory(prev => [...prev, ...event.objective_history]);
          }
        } else if (event.type === 'complete') {
          setIsRunning(false);
          setProgress(1.0);
          setStage('Complete');
          setMessage('Optimization completed successfully');
          if (event.results) {
            setResults(event.results);
            // Update objective history from final results (in case we missed any)
            if (event.results.objective_history) {
              setObjectiveHistory(event.results.objective_history);
            }
          }
        } else if (event.type === 'error') {
          setIsRunning(false);
          setError(event.error || 'Unknown error');
          setMessage(event.error || 'Optimization failed');
        }
      },
      (err: string) => {
        setIsRunning(false);
        setError(err);
        setMessage('Connection error');
      }
    );

    // Cleanup on unmount
    return () => {
      eventSource.close();
    };
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl p-6">
        <h2 className="text-2xl font-bold text-[var(--color-text-primary)] mb-2">Layer 1: Static Optimization</h2>
        <p className="text-sm text-[var(--color-text-secondary)]">
          <strong>Layer 1</strong> optimizes only <strong>static</strong> quantities:
        </p>
        <ul className="text-sm text-[var(--color-text-secondary)] list-disc list-inside mt-2 space-y-1">
          <li><strong>Engine geometry</strong>: throat area, L*, expansion ratio, pintle parameters</li>
          <li><strong>Initial tank pressures</strong>: single starting LOX and fuel tank pressures (no time history)</li>
        </ul>
        <p className="text-sm text-[var(--color-text-secondary)] mt-2">
          This layer evaluates at t=0 (static) to find an engine geometry and initial tank pressures that meet the target thrust/O/F and stability requirements. All time‑varying pressure curves and thermal protection sizing are handled in downstream layers (Layer 2/3).
        </p>
      </div>

      {/* Requirements Check */}
      {!requirements && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4">
          <p className="text-yellow-400">
            ⚠️ Please set design requirements in the 'Design Requirements' tab first.
          </p>
        </div>
      )}

      {/* Settings */}
      <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl p-6">
        <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">⚙️ Optimization Settings</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
              Max Iterations
            </label>
            <input
              type="number"
              value={settings.max_iterations}
              onChange={(e) => setSettings(prev => ({ ...prev, max_iterations: parseInt(e.target.value) }))}
              className="w-full px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="20"
              max="200"
              step="10"
              disabled={isRunning}
            />
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">Maximum optimization iterations for Layer 1</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
              Thrust Tolerance [%]
            </label>
            <input
              type="number"
              value={settings.thrust_tolerance * 100}
              onChange={(e) => setSettings(prev => ({ ...prev, thrust_tolerance: parseFloat(e.target.value) / 100 }))}
              className="w-full px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="1"
              max="20"
              step="1"
              disabled={isRunning}
            />
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">Acceptable deviation from target thrust</p>
          </div>
        </div>
      </div>

      {/* Run Button */}
      <div className="flex justify-center">
        <button
          onClick={handleRun}
          disabled={isRunning || !requirements}
          className={`px-8 py-4 font-bold rounded-lg text-white text-lg transition-all ${
            isRunning || !requirements
              ? 'bg-gray-500 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700 hover:scale-105'
          }`}
        >
          {isRunning ? '🔄 Running Optimization...' : '🚀 Run Layer 1 Optimization'}
        </button>
      </div>

      {/* Progress */}
      {(isRunning || progress > 0) && (
        <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl p-6">
          <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">📊 Progress</h3>
          
          {/* Progress Bar */}
          <div className="mb-4">
            <div className="flex justify-between text-sm text-[var(--color-text-secondary)] mb-2">
              <span>{stage}</span>
              <span>{(progress * 100).toFixed(0)}%</span>
            </div>
            <div className="w-full bg-[var(--color-bg-primary)] rounded-full h-4 overflow-hidden border border-[var(--color-border)]">
              <div
                className="bg-blue-600 h-full rounded-full transition-all duration-300"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
            <p className="text-sm text-[var(--color-text-secondary)] mt-2">{message}</p>
          </div>

          {/* Objective Convergence Plot - Always visible during optimization */}
          <div className="mt-6">
            <h4 className="text-md font-semibold text-[var(--color-text-primary)] mb-2">
              Objective Convergence
              {objectiveHistory.length > 0 && (
                <span className="text-sm font-normal text-[var(--color-text-secondary)] ml-2">
                  ({objectiveHistory.length} iterations)
                </span>
              )}
            </h4>
            {objectiveHistory.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={objectiveHistory} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" opacity={0.5} />
                  <XAxis 
                    dataKey="iteration"
                    stroke="var(--color-text-secondary)"
                    tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
                    label={{ value: 'Iteration', position: 'insideBottom', offset: -5, fill: 'var(--color-text-secondary)' }}
                  />
                  <YAxis 
                    scale="log"
                    domain={['auto', 'auto']}
                    stroke="var(--color-text-secondary)"
                    tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
                    label={{ value: 'Objective Value (log)', angle: -90, position: 'insideLeft', fill: 'var(--color-text-secondary)' }}
                  />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: 'var(--color-bg-secondary)',
                      border: '1px solid var(--color-border)',
                      borderRadius: '0.5rem',
                      color: 'var(--color-text-primary)'
                    }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="objective" 
                    name="Objective" 
                    stroke="#3b82f6" 
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="best_objective" 
                    name="Best Objective" 
                    stroke="#f97316" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-64 bg-[var(--color-bg-primary)] rounded-lg border border-[var(--color-border)]">
                <p className="text-[var(--color-text-secondary)]">
                  Waiting for objective function data...
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4">
          <p className="text-red-400 font-semibold">❌ Error: {error}</p>
        </div>
      )}

      {/* Results */}
      {results && results.performance && (
        <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl p-6">
          <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">✅ Optimization Results</h3>
          
          {/* Key Performance Metrics */}
          <div className="mb-6">
            <h4 className="text-md font-semibold text-[var(--color-text-primary)] mb-3">🎯 Performance</h4>
            <div className="grid grid-cols-4 gap-4">
              <ResultCard 
                label="Thrust" 
                value={results.performance.F} 
                unit="N" 
                decimals={1}
                color="blue"
              />
              <ResultCard 
                label="O/F Ratio" 
                value={results.performance.MR} 
                decimals={2}
                color="yellow"
              />
              <ResultCard 
                label="Specific Impulse" 
                value={results.performance.Isp} 
                unit="s" 
                decimals={1}
                color="purple"
              />
              <ResultCard 
                label="Chamber Pressure" 
                value={results.performance.Pc ? (results.performance.Pc as number) / 6894.76 : undefined} 
                unit="psi" 
                decimals={1}
                color="green"
              />
              <ResultCard 
                label="Exit Pressure" 
                value={results.performance.P_exit ? (results.performance.P_exit as number) / 6894.76 : undefined} 
                unit="psi" 
                decimals={1}
                color="cyan"
              />
              <ResultCard 
                label="Thrust Coefficient" 
                value={results.performance.Cf || results.performance.Cf_actual} 
                decimals={2}
                color="orange"
              />
              <ResultCard 
                label="c* Efficiency" 
                value={results.performance.eta_cstar ? (results.performance.eta_cstar as number) * 100 : undefined} 
                unit="%" 
                decimals={1}
                color="pink"
              />
              <ResultCard 
                label="Total Mass Flow" 
                value={results.performance.mdot_total} 
                unit="kg/s" 
                decimals={3}
                color="indigo"
              />
            </div>
          </div>

          {/* Optimized Pressures */}
          <div className="mb-6">
            <h4 className="text-md font-semibold text-[var(--color-text-primary)] mb-3">🔋 Optimized Tank Pressures</h4>
            <div className="grid grid-cols-2 gap-4">
              <ResultCard 
                label="LOX Tank Pressure" 
                value={results.performance.P_O_start_psi} 
                unit="psi" 
                decimals={1}
                color="cyan"
              />
              <ResultCard 
                label="Fuel Tank Pressure" 
                value={results.performance.P_F_start_psi} 
                unit="psi" 
                decimals={1}
                color="orange"
              />
            </div>
          </div>

          {/* Stability Results */}
          <div className="mb-6">
            <h4 className="text-md font-semibold text-[var(--color-text-primary)] mb-3">🛡️ Stability Analysis</h4>
            {(() => {
              // Extract stability data from various possible locations
              const perf = results.performance;
              const stabResults = perf.stability_results as Record<string, unknown> | undefined;
              
              // Get stability score and state (might be at root or nested)
              const stabilityScore = (perf.initial_stability_score as number) ?? 
                                     (stabResults?.stability_score as number) ?? undefined;
              const stabilityState = (perf.initial_stability_state as string) ?? 
                                     (stabResults?.stability_state as string) ?? 'unknown';
              
              // Get margins - could be at root level OR nested in stability_results
              const chuggingMargin = (perf.chugging_margin as number) ??
                                     ((stabResults?.chugging as Record<string, unknown>)?.stability_margin as number) ?? undefined;
              const acousticMargin = (perf.acoustic_margin as number) ??
                                     ((stabResults?.acoustic as Record<string, unknown>)?.stability_margin as number) ?? undefined;
              const feedMargin = (perf.feed_margin as number) ??
                                 ((stabResults?.feed_system as Record<string, unknown>)?.stability_margin as number) ?? undefined;
              
              return (
                <div className="grid grid-cols-5 gap-4">
                  <ResultCard 
                    label="Stability Score" 
                    value={stabilityScore} 
                    decimals={2}
                    color={stabilityScore !== undefined && stabilityScore >= 0.75 ? 'green' : 
                           stabilityScore !== undefined && stabilityScore >= 0.4 ? 'yellow' : 'red'}
                  />
                  <ResultCard 
                    label="Stability State" 
                    value={stabilityState} 
                    isText
                    color={stabilityState === 'stable' ? 'green' : 
                           stabilityState === 'marginal' ? 'yellow' : 'red'}
                  />
                  <ResultCard 
                    label="Chugging Margin" 
                    value={chuggingMargin} 
                    decimals={3}
                    color="purple"
                  />
                  <ResultCard 
                    label="Acoustic Margin" 
                    value={acousticMargin} 
                    decimals={3}
                    color="blue"
                  />
                  <ResultCard 
                    label="Feed System Margin" 
                    value={feedMargin} 
                    decimals={3}
                    color="cyan"
                  />
                </div>
              );
            })()}
          </div>

          {/* Validation Status */}
          <div className="mb-6">
            <h4 className="text-md font-semibold text-[var(--color-text-primary)] mb-3">✓ Validation</h4>
            <div className="grid grid-cols-4 gap-4">
              <ValidationCard 
                label="Thrust Check" 
                passed={results.performance.thrust_check_passed}
              />
              <ValidationCard 
                label="O/F Check" 
                passed={results.performance.of_check_passed}
              />
              <ValidationCard 
                label="Stability Check" 
                passed={results.performance.stability_check_passed}
              />
              <ValidationCard 
                label="Pressure Candidate" 
                passed={results.performance.pressure_candidate_valid}
              />
            </div>
            {results.performance.failure_reasons && results.performance.failure_reasons.length > 0 && (
              <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                <p className="text-sm text-red-400 font-semibold mb-1">Failure Reasons:</p>
                <ul className="text-sm text-red-400 list-disc list-inside">
                  {results.performance.failure_reasons.map((reason: string, i: number) => (
                    <li key={i}>{reason}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

