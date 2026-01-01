import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { getChamberGeometry } from '../api/client';
import type { ChamberGeometryResponse, EngineConfig } from '../api/client';

interface ChamberGeometryProps {
  config: EngineConfig | null;
}

// Convert m to mm for display
const M_TO_MM = 1000;
const MM_TO_INCH = 1 / 25.4;
const INCH_TO_MM = 25.4;

// Helper function to get nice step size for a given range
function getNiceStep(range: number): number {
  const magnitude = Math.floor(Math.log10(range));
  const normalized = range / Math.pow(10, magnitude);
  
  let step;
  if (normalized <= 1) step = 1;
  else if (normalized <= 2) step = 2;
  else if (normalized <= 5) step = 5;
  else step = 10;
  
  return step * Math.pow(10, magnitude);
}

// Helper function to ensure domain includes 0 and has nice rounded bounds
function makeNiceDomain(min: number, max: number, includeZero: boolean = true): [number, number] {
  // Calculate the range
  let range = max - min;
  
  // If including zero, expand range to include it
  if (includeZero) {
    if (min > 0) {
      range = max;
      min = 0;
    } else if (max < 0) {
      range = Math.abs(min);
      max = 0;
    } else {
      range = Math.max(Math.abs(min), Math.abs(max)) * 2;
      min = -range / 2;
      max = range / 2;
    }
  }
  
  // Get a nice step size
  const step = getNiceStep(range / 8); // Aim for about 8 ticks
  
  // Round min down and max up to nice values
  const domainMin = includeZero && min <= 0 && max >= 0 
    ? Math.floor(min / step) * step
    : Math.floor(min / step) * step;
  const domainMax = includeZero && min <= 0 && max >= 0
    ? Math.ceil(max / step) * step
    : Math.ceil(max / step) * step;
  
  // Ensure 0 is included if requested
  let finalMin = domainMin;
  let finalMax = domainMax;
  if (includeZero) {
    if (finalMin > 0) finalMin = 0;
    if (finalMax < 0) finalMax = 0;
  }
  
  // Ensure min < max
  if (finalMin >= finalMax) {
    const absMax = Math.max(Math.abs(finalMin), Math.abs(finalMax));
    finalMin = -absMax;
    finalMax = absMax;
  }
  
  return [finalMin, finalMax];
}

// Helper function to format tick values nicely
function formatTick(value: number, unit: 'mm' | 'inch'): string {
  // For small values, show more decimals; for large values, show fewer
  const absValue = Math.abs(value);
  
  if (unit === 'inch') {
    if (absValue < 0.1) return value.toFixed(3);
    if (absValue < 1) return value.toFixed(2);
    if (absValue < 10) return value.toFixed(1);
    return value.toFixed(0);
  } else {
    // mm
    if (absValue < 1) return value.toFixed(2);
    if (absValue < 10) return value.toFixed(1);
    return value.toFixed(0);
  }
}

export function ChamberGeometry({ config }: ChamberGeometryProps) {
  const [geometry, setGeometry] = useState<ChamberGeometryResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showLowerHalf, setShowLowerHalf] = useState(true);
  const [ceaUnit, setCeaUnit] = useState<'mm' | 'inch'>('mm');
  const ceaContourContainerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 1000, height: 350 });

  // Fetch geometry when component mounts or config changes
  const fetchGeometry = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    const result = await getChamberGeometry();
    
    setIsLoading(false);
    
    if (result.error) {
      setError(result.error);
      setGeometry(null);
    } else if (result.data) {
      setGeometry(result.data);
    }
  }, []);

  useEffect(() => {
    if (config) {
      fetchGeometry();
    }
  }, [config, fetchGeometry]);

  // Transform geometry data for chart - create symmetric view
  // This includes chamber layers + Rao nozzle contour
  const chartData = useMemo(() => {
    if (!geometry) return [];
    
    const data: Record<string, number>[] = [];
    
    // First, add chamber region data (before throat)
    const n = geometry.positions.length;
    for (let i = 0; i < n; i++) {
      const pos = geometry.positions[i];
      // Only include chamber region (up to throat)
      if (pos > geometry.throat_position) continue;
      
      const x = pos * M_TO_MM;  // Convert to mm
      const rGas = geometry.R_gas[i] * M_TO_MM;
      const rAblative = geometry.R_ablative_outer[i] * M_TO_MM;
      const rGraphite = geometry.R_graphite_outer[i] * M_TO_MM;
      const rStainless = geometry.R_stainless[i] * M_TO_MM;
      
      // Check if this point is in graphite region
      const isGraphiteRegion = 
        pos >= geometry.graphite_start && 
        pos <= geometry.graphite_end;
      
      data.push({
        x,
        // Upper half
        R_stainless_upper: rStainless,
        R_graphite_upper: isGraphiteRegion ? rGraphite : rGas,
        R_ablative_upper: rAblative,
        R_gas_upper: rGas,
        // Lower half (negative)
        R_stainless_lower: showLowerHalf ? -rStainless : 0,
        R_graphite_lower: showLowerHalf ? (isGraphiteRegion ? -rGraphite : -rGas) : 0,
        R_ablative_lower: showLowerHalf ? -rAblative : 0,
        R_gas_lower: showLowerHalf ? -rGas : 0,
      });
    }
    
    return data;
  }, [geometry, showLowerHalf]);

  // CEA-solved chamber contour data
  const chamberContourData = useMemo(() => {
    if (!geometry || !geometry.chamber_contour_x || geometry.chamber_contour_x.length === 0) return [];
    
    // Convert from meters to selected unit
    const unitMultiplier = ceaUnit === 'mm' ? M_TO_MM : M_TO_MM * MM_TO_INCH;
    
    return geometry.chamber_contour_x.map((x, i) => ({
      x: x * unitMultiplier,
      R_chamber_upper: geometry.chamber_contour_y[i] * unitMultiplier,
      R_chamber_lower: showLowerHalf ? -geometry.chamber_contour_y[i] * unitMultiplier : 0,
    }));
  }, [geometry, showLowerHalf, ceaUnit]);

  // Measure container size for equal-scale calculation
  useEffect(() => {
    if (!ceaContourContainerRef.current) return;
    
    const updateSize = () => {
      if (ceaContourContainerRef.current) {
        const rect = ceaContourContainerRef.current.getBoundingClientRect();
        setContainerSize({ width: rect.width, height: rect.height });
      }
    };
    
    // Initial measurement
    updateSize();
    
    // Use ResizeObserver for accurate container size tracking
    const resizeObserver = new ResizeObserver(() => {
      updateSize();
    });
    
    resizeObserver.observe(ceaContourContainerRef.current);
    
    // Also listen to window resize as fallback
    window.addEventListener('resize', updateSize);
    
    return () => {
      resizeObserver.disconnect();
      window.removeEventListener('resize', updateSize);
    };
  }, [geometry]);

  // Calculate equal-scale domains for CEA contour plot (1:1 aspect ratio)
  const ceaContourDomains = useMemo(() => {
    if (!chamberContourData || chamberContourData.length === 0) {
      return { 
        xDomain: ['dataMin', 'dataMax'], 
        yDomain: showLowerHalf ? ['auto', 'auto'] : [0, 'auto'],
        xLabel: `Axial Position (${ceaUnit})`,
        yLabel: `Radius (${ceaUnit})`
      };
    }
    
    // Calculate data ranges
    const xValues = chamberContourData.map(d => d.x);
    const yValues = showLowerHalf 
      ? [...chamberContourData.map(d => d.R_chamber_upper), ...chamberContourData.map(d => d.R_chamber_lower)]
      : chamberContourData.map(d => d.R_chamber_upper);
    
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;
    
    // Container dimensions (accounting for margins: left: 20, right: 30, top: 20, bottom: 20)
    // Use default if container not measured yet
    const effectiveWidth = containerSize.width > 0 ? containerSize.width : 1000;
    const effectiveHeight = containerSize.height > 0 ? containerSize.height : 350;
    const plotWidth = effectiveWidth - 20 - 30;
    const plotHeight = effectiveHeight - 20 - 20;
    
    // Guard against invalid dimensions
    if (plotWidth <= 0 || plotHeight <= 0) {
      return { 
        xDomain: [xMin, xMax], 
        yDomain: [yMin, yMax],
        xLabel: `Axial Position (${ceaUnit})`,
        yLabel: `Radius (${ceaUnit})`
      };
    }
    
    // Calculate aspect ratio of plot area
    const plotAspectRatio = plotWidth / plotHeight;
    
    // For equal scales (1:1), we want 1mm on x-axis to equal 1mm on y-axis visually
    // This means: xRange / plotWidth should equal yRange / plotHeight
    // Or: xRange / yRange should equal plotWidth / plotHeight
    
    let xDomain: [number, number] = [xMin, xMax];
    let yDomain: [number, number] = [yMin, yMax];
    
    // Calculate the data aspect ratio (x-range / y-range)
    const dataAspectRatio = xRange / yRange;
    
    // To achieve 1:1 scale, adjust domains so that the visual representation
    // shows equal physical scales on both axes
    if (dataAspectRatio > plotAspectRatio) {
      // Data is wider relative to its height than the plot - expand y range to match
      const targetYRange = xRange / plotAspectRatio;
      const yCenter = (yMin + yMax) / 2;
      yDomain = [yCenter - targetYRange / 2, yCenter + targetYRange / 2];
    } else {
      // Data is taller relative to its width than the plot - expand x range to match
      const targetXRange = yRange * plotAspectRatio;
      const xCenter = (xMin + xMax) / 2;
      xDomain = [xCenter - targetXRange / 2, xCenter + targetXRange / 2];
    }
    
    // Add small padding (5%)
    const xPadding = (xDomain[1] - xDomain[0]) * 0.05;
    const yPadding = (yDomain[1] - yDomain[0]) * 0.05;
    xDomain = [xDomain[0] - xPadding, xDomain[1] + xPadding];
    yDomain = [yDomain[0] - yPadding, yDomain[1] + yPadding];
    
    // Round to nice values and ensure 0 is included
    // For x-axis, include 0 if the domain spans across 0
    const xIncludesZero = xDomain[0] <= 0 && xDomain[1] >= 0;
    xDomain = makeNiceDomain(xDomain[0], xDomain[1], xIncludesZero);
    
    // For y-axis, always include 0 (since it's the centerline)
    yDomain = makeNiceDomain(yDomain[0], yDomain[1], true);
    
    return {
      xDomain: xDomain as [number, number],
      yDomain: yDomain as [number, number],
      xLabel: `Axial Position (${ceaUnit})`,
      yLabel: `Radius (${ceaUnit})`
    };
  }, [chamberContourData, showLowerHalf, containerSize, ceaUnit]);

  // Calculate dimensions for display
  const dimensions = useMemo(() => {
    if (!geometry) return null;
    
    return {
      L_chamber_mm: geometry.L_chamber * M_TO_MM,
      L_nozzle_mm: geometry.L_nozzle * M_TO_MM,
      L_total_mm: (geometry.L_chamber + geometry.L_nozzle) * M_TO_MM,
      D_chamber_mm: geometry.D_chamber * M_TO_MM,
      D_throat_mm: geometry.D_throat * M_TO_MM,
      D_exit_mm: geometry.D_exit * M_TO_MM,
      throat_position_mm: geometry.throat_position * M_TO_MM,
    };
  }, [geometry]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;
    
    // Find the actual radii at this position
    const gasRadius = payload.find((p: any) => p.dataKey === 'R_gas_upper')?.value || 0;
    const ablativeRadius = payload.find((p: any) => p.dataKey === 'R_ablative_upper')?.value || 0;
    const graphiteRadius = payload.find((p: any) => p.dataKey === 'R_graphite_upper')?.value || 0;
    const stainlessRadius = payload.find((p: any) => p.dataKey === 'R_stainless_upper')?.value || 0;
    
    return (
      <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-lg p-3 shadow-xl">
        <p className="text-sm font-medium text-[var(--color-text-primary)] mb-2">
          Position: {label.toFixed(1)} mm
        </p>
        <div className="space-y-1 text-xs">
          <p className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full" style={{ backgroundColor: '#ff6b35' }} />
            <span className="text-orange-400">Gas Boundary: Ø{(gasRadius * 2).toFixed(1)} mm</span>
          </p>
          {ablativeRadius > gasRadius && (
            <p className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: '#8b4513' }} />
              <span className="text-amber-600">Ablative: Ø{(ablativeRadius * 2).toFixed(1)} mm</span>
            </p>
          )}
          {graphiteRadius > gasRadius && (
            <p className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: '#1a1a1a' }} />
              <span className="text-gray-400">Graphite: Ø{(graphiteRadius * 2).toFixed(1)} mm</span>
            </p>
          )}
          <p className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full" style={{ backgroundColor: '#6b7280' }} />
            <span className="text-gray-500">Stainless: Ø{(stainlessRadius * 2).toFixed(1)} mm</span>
          </p>
        </div>
      </div>
    );
  };

  // Empty state - no config
  if (!config) {
    return (
      <div className="flex items-center justify-center h-64 text-[var(--color-text-secondary)]">
        <div className="text-center">
          <svg className="w-12 h-12 mx-auto mb-3 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <p>No config loaded</p>
          <p className="text-sm mt-1">Upload a YAML config file first</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="p-5 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-rose-500 to-orange-600 flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
              </svg>
            </div>
            <div>
              <h2 className="text-lg font-bold text-[var(--color-text-primary)]">Chamber Geometry</h2>
              <p className="text-sm text-[var(--color-text-secondary)]">
                Cross-section visualization of thrust chamber
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
              <input
                type="checkbox"
                checked={showLowerHalf}
                onChange={(e) => setShowLowerHalf(e.target.checked)}
                className="w-4 h-4 rounded border-[var(--color-border)] text-rose-600 focus:ring-rose-500"
              />
              Show Full Cross-Section
            </label>
            
            <button
              onClick={fetchGeometry}
              disabled={isLoading}
              className="px-4 py-2 rounded-lg bg-gradient-to-r from-rose-600 to-orange-600 hover:from-rose-700 hover:to-orange-700 text-white text-sm font-medium transition-all disabled:opacity-50 flex items-center gap-2"
            >
              {isLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Loading...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  Refresh
                </>
              )}
            </button>
          </div>
        </div>

        {/* Error message */}
        {error && (
          <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
            {error}
          </div>
        )}
      </div>

      {/* Geometry Plot */}
      {geometry && chartData.length > 0 && (
        <div className="p-4 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
          <h4 className="text-sm font-semibold text-[var(--color-text-primary)] mb-4">
            Chamber Cross-Section (Side View)
          </h4>
          
          <ResponsiveContainer width="100%" height={450}>
            <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" opacity={0.3} />
              
              <XAxis
                dataKey="x"
                type="number"
                domain={['dataMin', 'dataMax']}
                stroke="var(--color-text-secondary)"
                tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
                label={{ 
                  value: 'Axial Position (mm)', 
                  position: 'insideBottom', 
                  offset: -10, 
                  fill: 'var(--color-text-secondary)' 
                }}
              />
              
              <YAxis
                domain={showLowerHalf ? ['auto', 'auto'] : [0, 'auto']}
                stroke="var(--color-text-secondary)"
                tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
                label={{ 
                  value: 'Radius (mm)', 
                  angle: -90, 
                  position: 'insideLeft', 
                  fill: 'var(--color-text-secondary)' 
                }}
              />
              
              <Tooltip content={<CustomTooltip />} />
              
              {/* Stainless Steel (outermost) - Upper */}
              <Area
                type="monotone"
                dataKey="R_stainless_upper"
                stroke="#6b7280"
                fill="#6b7280"
                fillOpacity={0.3}
                strokeWidth={1.5}
                name="Stainless Steel"
              />
              
              {/* Graphite (throat region) - Upper */}
              <Area
                type="monotone"
                dataKey="R_graphite_upper"
                stroke="#1a1a1a"
                fill="#1a1a1a"
                fillOpacity={0.5}
                strokeWidth={1.5}
                name="Graphite Insert"
              />
              
              {/* Ablative (chamber region) - Upper */}
              <Area
                type="monotone"
                dataKey="R_ablative_upper"
                stroke="#8b4513"
                fill="#8b4513"
                fillOpacity={0.4}
                strokeWidth={1.5}
                name="Ablative Liner"
              />
              
              {/* Gas Boundary - Upper */}
              <Area
                type="monotone"
                dataKey="R_gas_upper"
                stroke="#ff6b35"
                fill="#ff6b35"
                fillOpacity={0.2}
                strokeWidth={2}
                name="Gas Boundary"
              />
              
              {/* Lower half (symmetric) */}
              {showLowerHalf && (
                <>
                  <Area
                    type="monotone"
                    dataKey="R_stainless_lower"
                    stroke="#6b7280"
                    fill="#6b7280"
                    fillOpacity={0.3}
                    strokeWidth={1.5}
                    legendType="none"
                  />
                  <Area
                    type="monotone"
                    dataKey="R_graphite_lower"
                    stroke="#1a1a1a"
                    fill="#1a1a1a"
                    fillOpacity={0.5}
                    strokeWidth={1.5}
                    legendType="none"
                  />
                  <Area
                    type="monotone"
                    dataKey="R_ablative_lower"
                    stroke="#8b4513"
                    fill="#8b4513"
                    fillOpacity={0.4}
                    strokeWidth={1.5}
                    legendType="none"
                  />
                  <Area
                    type="monotone"
                    dataKey="R_gas_lower"
                    stroke="#ff6b35"
                    fill="#ff6b35"
                    fillOpacity={0.2}
                    strokeWidth={2}
                    legendType="none"
                  />
                </>
              )}
              
              {/* Throat position reference line */}
              {dimensions && (
                <ReferenceLine
                  x={dimensions.throat_position_mm}
                  stroke="#ef4444"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  label={{ 
                    value: 'Throat', 
                    position: 'top', 
                    fill: '#ef4444',
                    fontSize: 11,
                  }}
                />
              )}
              
              {/* Centerline */}
              <ReferenceLine
                y={0}
                stroke="var(--color-text-secondary)"
                strokeWidth={1}
                strokeDasharray="3 3"
              />
              
              <Legend 
                verticalAlign="top" 
                height={36}
                wrapperStyle={{ paddingBottom: '10px' }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* CEA-Solved Chamber Contour */}
      {geometry && chamberContourData.length > 0 && (
        <div 
          ref={ceaContourContainerRef}
          className="p-4 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)]"
        >
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-sm font-semibold text-[var(--color-text-primary)]">
              Chamber Contour (CEA Thermochemistry)
            </h4>
            <div className="flex items-center gap-3">
              {/* Unit switcher */}
              <div className="flex items-center gap-2 bg-[var(--color-bg-primary)] rounded-lg border border-[var(--color-border)] p-1">
                <button
                  onClick={() => setCeaUnit('mm')}
                  className={`px-3 py-1 text-xs font-medium rounded transition-all ${
                    ceaUnit === 'mm'
                      ? 'bg-rose-600 text-white'
                      : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]'
                  }`}
                >
                  mm
                </button>
                <button
                  onClick={() => setCeaUnit('inch')}
                  className={`px-3 py-1 text-xs font-medium rounded transition-all ${
                    ceaUnit === 'inch'
                      ? 'bg-rose-600 text-white'
                      : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]'
                  }`}
                >
                  in
                </button>
              </div>
              <span className="text-xs px-2 py-1 rounded bg-emerald-500/20 text-emerald-400">
                Cf = {geometry.Cf?.toFixed(4) ?? 'N/A'}
              </span>
            </div>
          </div>
          
          <ResponsiveContainer width="100%" height={350}>
            <ComposedChart data={chamberContourData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" opacity={0.3} />
              
              <XAxis
                dataKey="x"
                type="number"
                domain={ceaContourDomains.xDomain}
                stroke="var(--color-text-secondary)"
                tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
                tickFormatter={(value) => formatTick(value, ceaUnit)}
                allowDecimals={true}
                label={{ 
                  value: ceaContourDomains.xLabel, 
                  position: 'insideBottom', 
                  offset: -10, 
                  fill: 'var(--color-text-secondary)' 
                }}
              />
              
              <YAxis
                domain={ceaContourDomains.yDomain}
                stroke="var(--color-text-secondary)"
                tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
                tickFormatter={(value) => formatTick(value, ceaUnit)}
                allowDecimals={true}
                label={{ 
                  value: ceaContourDomains.yLabel, 
                  angle: -90, 
                  position: 'insideLeft', 
                  fill: 'var(--color-text-secondary)' 
                }}
              />
              
              {/* Chamber contour - Upper */}
              <Line
                type="monotone"
                dataKey="R_chamber_upper"
                stroke="#10b981"
                strokeWidth={2.5}
                dot={false}
                name="Chamber Contour (CEA)"
              />
              
              {/* Chamber contour - Lower (symmetric) */}
              {showLowerHalf && (
                <Line
                  type="monotone"
                  dataKey="R_chamber_lower"
                  stroke="#10b981"
                  strokeWidth={2.5}
                  dot={false}
                  legendType="none"
                />
              )}
              
              {/* Centerline */}
              <ReferenceLine
                y={0}
                stroke="var(--color-text-secondary)"
                strokeWidth={1}
                strokeDasharray="3 3"
              />
              
              <Legend 
                verticalAlign="top" 
                height={36}
                wrapperStyle={{ paddingBottom: '10px' }}
              />
            </ComposedChart>
          </ResponsiveContainer>
          
          <p className="text-xs text-[var(--color-text-secondary)] mt-2">
            Full chamber contour (cylindrical + contraction + nozzle) solved using CEA thermochemistry for accurate Cf.
          </p>
        </div>
      )}

      {/* Dimensions Table */}
      {geometry && dimensions && (
        <div className="p-5 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
          <h4 className="text-sm font-semibold text-[var(--color-text-primary)] mb-4">
            Chamber Dimensions
          </h4>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-3 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)]">
              <p className="text-xs text-[var(--color-text-secondary)]">Chamber Length</p>
              <p className="text-lg font-semibold text-[var(--color-text-primary)]">
                {dimensions.L_chamber_mm.toFixed(1)} mm
              </p>
            </div>
            
            <div className="p-3 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)]">
              <p className="text-xs text-[var(--color-text-secondary)]">Nozzle Length</p>
              <p className="text-lg font-semibold text-[var(--color-text-primary)]">
                {dimensions.L_nozzle_mm.toFixed(1)} mm
              </p>
            </div>
            
            <div className="p-3 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)]">
              <p className="text-xs text-[var(--color-text-secondary)]">Chamber Diameter</p>
              <p className="text-lg font-semibold text-[var(--color-text-primary)]">
                {dimensions.D_chamber_mm.toFixed(1)} mm
              </p>
            </div>
            
            <div className="p-3 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)]">
              <p className="text-xs text-[var(--color-text-secondary)]">Throat Diameter</p>
              <p className="text-lg font-semibold text-rose-400">
                {dimensions.D_throat_mm.toFixed(1)} mm
              </p>
            </div>
            
            <div className="p-3 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)]">
              <p className="text-xs text-[var(--color-text-secondary)]">Exit Diameter</p>
              <p className="text-lg font-semibold text-[var(--color-text-primary)]">
                {dimensions.D_exit_mm.toFixed(1)} mm
              </p>
            </div>
            
            <div className="p-3 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)]">
              <p className="text-xs text-[var(--color-text-secondary)]">Expansion Ratio</p>
              <p className="text-lg font-semibold text-[var(--color-text-primary)]">
                {geometry.expansion_ratio.toFixed(2)}
              </p>
            </div>
            
            <div className="p-3 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)]">
              <p className="text-xs text-[var(--color-text-secondary)]">Ablative Cooling</p>
              <p className={`text-lg font-semibold ${geometry.ablative_enabled ? 'text-green-400' : 'text-gray-500'}`}>
                {geometry.ablative_enabled ? 'Enabled' : 'Disabled'}
              </p>
            </div>
            
            <div className="p-3 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)]">
              <p className="text-xs text-[var(--color-text-secondary)]">Graphite Insert</p>
              <p className={`text-lg font-semibold ${geometry.graphite_enabled ? 'text-green-400' : 'text-gray-500'}`}>
                {geometry.graphite_enabled ? 'Enabled' : 'Disabled'}
              </p>
            </div>
            
            <div className="p-3 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)]">
              <p className="text-xs text-[var(--color-text-secondary)]">Nozzle Type</p>
              <p className="text-lg font-semibold text-blue-400">
                {geometry.nozzle_method.includes('rao') ? 'Rao Bell (80%)' : 'Conical'}
              </p>
            </div>
            
            {geometry.Cf !== null && (
              <div className="p-3 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)]">
                <p className="text-xs text-[var(--color-text-secondary)]">Thrust Coeff (Cf)</p>
                <p className="text-lg font-semibold text-emerald-400">
                  {geometry.Cf.toFixed(4)}
                </p>
              </div>
            )}
            
            {geometry.Cf_ideal !== null && (
              <div className="p-3 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)]">
                <p className="text-xs text-[var(--color-text-secondary)]">Cf Ideal (CEA)</p>
                <p className="text-lg font-semibold text-[var(--color-text-primary)]">
                  {geometry.Cf_ideal.toFixed(4)}
                </p>
              </div>
            )}
            
            {geometry.A_throat_solved !== null && (
              <div className="p-3 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)]">
                <p className="text-xs text-[var(--color-text-secondary)]">A_throat (solved)</p>
                <p className="text-lg font-semibold text-[var(--color-text-primary)]">
                  {(geometry.A_throat_solved * 1e6).toFixed(2)} mm²
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Legend explanation */}
      {geometry && (
        <div className="p-4 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
          <h4 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3">
            Structure Legend
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: '#ff6b35', opacity: 0.6 }} />
              <span className="text-[var(--color-text-secondary)]">
                <span className="text-orange-400 font-medium">Gas Boundary</span> — Hot combustion gas
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: '#8b4513', opacity: 0.6 }} />
              <span className="text-[var(--color-text-secondary)]">
                <span className="text-amber-600 font-medium">Ablative</span> — Chamber liner
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: '#1a1a1a', opacity: 0.7 }} />
              <span className="text-[var(--color-text-secondary)]">
                <span className="text-gray-400 font-medium">Graphite</span> — Throat insert
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: '#6b7280', opacity: 0.5 }} />
              <span className="text-[var(--color-text-secondary)]">
                <span className="text-gray-500 font-medium">Stainless</span> — Outer case
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Loading state */}
      {isLoading && !geometry && (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="w-12 h-12 border-4 border-rose-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-[var(--color-text-secondary)]">Loading chamber geometry...</p>
          </div>
        </div>
      )}
    </div>
  );
}

