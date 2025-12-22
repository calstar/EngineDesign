import type { PerformanceMetrics } from '../api/client';

interface ResultsDisplayProps {
  results: PerformanceMetrics | null;
  isLoading?: boolean;
}

interface MetricCardProps {
  label: string;
  value: string;
  unit: string;
  color?: 'blue' | 'green' | 'yellow' | 'purple' | 'cyan';
}

function MetricCard({ label, value, unit, color = 'blue' }: MetricCardProps) {
  const colorClasses = {
    blue: 'border-blue-500/30 bg-blue-500/5',
    green: 'border-green-500/30 bg-green-500/5',
    yellow: 'border-yellow-500/30 bg-yellow-500/5',
    purple: 'border-purple-500/30 bg-purple-500/5',
    cyan: 'border-cyan-500/30 bg-cyan-500/5',
  };

  const valueColorClasses = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    yellow: 'text-yellow-400',
    purple: 'text-purple-400',
    cyan: 'text-cyan-400',
  };

  return (
    <div className={`p-4 rounded-xl border ${colorClasses[color]} transition-all hover:scale-[1.02]`}>
      <div className="text-sm text-[var(--color-text-secondary)] mb-1">{label}</div>
      <div className="flex items-baseline gap-2">
        <span className={`text-2xl font-bold ${valueColorClasses[color]}`}>{value}</span>
        <span className="text-sm text-[var(--color-text-secondary)]">{unit}</span>
      </div>
    </div>
  );
}

export function ResultsDisplay({ results, isLoading }: ResultsDisplayProps) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-48">
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 border-3 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-[var(--color-text-secondary)]">Running simulation...</span>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="flex items-center justify-center h-48 text-[var(--color-text-secondary)]">
        <div className="text-center">
          <svg className="w-12 h-12 mx-auto mb-3 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <p>No results yet</p>
          <p className="text-sm mt-1">Enter tank pressures and click Evaluate</p>
        </div>
      </div>
    );
  }

  const formatNumber = (n: number, decimals: number = 2): string => {
    return n.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  return (
    <div className="space-y-4">
      {/* Primary metrics - large cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Thrust"
          value={formatNumber(results.thrust_kN, 2)}
          unit="kN"
          color="green"
        />
        <MetricCard
          label="Specific Impulse"
          value={formatNumber(results.Isp_s, 1)}
          unit="s"
          color="blue"
        />
        <MetricCard
          label="Chamber Pressure"
          value={formatNumber(results.chamber_pressure_psi, 1)}
          unit="psi"
          color="yellow"
        />
        <MetricCard
          label="Mixture Ratio (O/F)"
          value={formatNumber(results.mixture_ratio, 3)}
          unit=""
          color="purple"
        />
      </div>

      {/* Secondary metrics - smaller cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="p-3 rounded-lg bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
          <div className="text-xs text-[var(--color-text-secondary)] mb-1">Total Mass Flow</div>
          <div className="text-lg font-semibold text-[var(--color-text-primary)]">
            {formatNumber(results.mdot_total_kg_s, 3)} <span className="text-sm font-normal text-[var(--color-text-secondary)]">kg/s</span>
          </div>
        </div>
        <div className="p-3 rounded-lg bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
          <div className="text-xs text-[var(--color-text-secondary)] mb-1">Oxidizer Flow</div>
          <div className="text-lg font-semibold text-cyan-400">
            {formatNumber(results.mdot_oxidizer_kg_s, 3)} <span className="text-sm font-normal text-[var(--color-text-secondary)]">kg/s</span>
          </div>
        </div>
        <div className="p-3 rounded-lg bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
          <div className="text-xs text-[var(--color-text-secondary)] mb-1">Fuel Flow</div>
          <div className="text-lg font-semibold text-orange-400">
            {formatNumber(results.mdot_fuel_kg_s, 3)} <span className="text-sm font-normal text-[var(--color-text-secondary)]">kg/s</span>
          </div>
        </div>
        <div className="p-3 rounded-lg bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
          <div className="text-xs text-[var(--color-text-secondary)] mb-1">c* (Actual)</div>
          <div className="text-lg font-semibold text-[var(--color-text-primary)]">
            {formatNumber(results.cstar_actual_m_s, 1)} <span className="text-sm font-normal text-[var(--color-text-secondary)]">m/s</span>
          </div>
        </div>
      </div>

      {/* Nozzle metrics */}
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded-lg bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
          <div className="text-xs text-[var(--color-text-secondary)] mb-1">Exit Velocity</div>
          <div className="text-lg font-semibold text-[var(--color-text-primary)]">
            {formatNumber(results.exit_velocity_m_s, 1)} <span className="text-sm font-normal text-[var(--color-text-secondary)]">m/s</span>
          </div>
        </div>
        <div className="p-3 rounded-lg bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
          <div className="text-xs text-[var(--color-text-secondary)] mb-1">Exit Pressure</div>
          <div className="text-lg font-semibold text-[var(--color-text-primary)]">
            {formatNumber(results.exit_pressure_Pa / 1000, 2)} <span className="text-sm font-normal text-[var(--color-text-secondary)]">kPa</span>
          </div>
        </div>
      </div>
    </div>
  );
}

