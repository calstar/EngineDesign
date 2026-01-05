import { useState, useEffect, useMemo } from 'react';
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
    runLayer3Optimization,
    getLayer3Status,
    stopLayer3Optimization,
    uploadLayer3Config
} from '../api/client';
import type {
    Layer3Settings,
    Layer3ProgressEvent,
    Layer3Results,
    DesignRequirements
} from '../api/client';

interface Layer3OptimizationProps {
    requirements: DesignRequirements | null;
}

// Helper component for result cards
function ResultCard({
    label,
    value,
    unit,
    decimals = 2,
    color = 'orange',
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
        orange: 'bg-orange-500/10 border-orange-500/30',
        red: 'bg-red-500/10 border-red-500/30',
        yellow: 'bg-yellow-500/10 border-yellow-500/30',
        green: 'bg-green-500/10 border-green-500/30',
        blue: 'bg-blue-500/10 border-blue-500/30',
        purple: 'bg-purple-500/10 border-purple-500/30',
        cyan: 'bg-cyan-500/10 border-cyan-500/30',
    };

    const textColorClasses: Record<string, string> = {
        orange: 'text-orange-400',
        red: 'text-red-400',
        yellow: 'text-yellow-400',
        green: 'text-green-400',
        blue: 'text-blue-400',
        purple: 'text-purple-400',
        cyan: 'text-cyan-400',
    };

    const displayValue = isText
        ? String(value || '-')
        : typeof value === 'number'
            ? value.toFixed(decimals)
            : value !== undefined && value !== null
                ? String(value)
                : '-';

    return (
        <div className={`rounded-lg p-3 border ${colorClasses[color] || colorClasses.orange}`}>
            <p className="text-xs text-[var(--color-text-secondary)] mb-1">{label}</p>
            <p className={`text-lg font-bold ${textColorClasses[color] || textColorClasses.orange}`}>
                {displayValue}
                {unit && <span className="text-sm font-normal text-[var(--color-text-secondary)] ml-1">{unit}</span>}
            </p>
        </div>
    );
}

export function Layer3Optimization({ requirements }: Layer3OptimizationProps) {
    const [settings, setSettings] = useState<Layer3Settings>({
        max_iterations: 20,
        save_plots: false,
    });

    const [isRunning, setIsRunning] = useState(false);
    const [isStopping, setIsStopping] = useState(false);
    const [progress, setProgress] = useState(0);
    const [stage, setStage] = useState('');
    const [message, setMessage] = useState('');
    const [results, setResults] = useState<Layer3Results | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [successMessage, setSuccessMessage] = useState<string | null>(null);
    const [objectiveHistory, setObjectiveHistory] = useState<Array<{
        iteration: number;
        objective: number;
        best_objective: number;
    }>>([]);

    const [eventSourceRef, setEventSourceRef] = useState<EventSource | null>(null);
    const [configLoaded, setConfigLoaded] = useState(false);

    // Calculate min/max objective values for dot scaling (same as Layer 2)
    const { minObj, maxObj } = useMemo(() => {
        if (objectiveHistory.length === 0) {
            return { minObj: 1, maxObj: 1 };
        }
        const values = objectiveHistory.map(h => h.objective).filter(v => typeof v === 'number' && isFinite(v));
        if (values.length === 0) {
            return { minObj: 1, maxObj: 1 };
        }
        return {
            minObj: Math.min(...values),
            maxObj: Math.max(...values),
        };
    }, [objectiveHistory]);

    const renderDot = useMemo(() => {
        const minSize = 5;
        const maxSize = 2;
        const logMinObj = Math.log(Math.max(1e-10, minObj));
        const logMaxObj = Math.log(Math.max(1e-10, maxObj));
        const logRange = logMaxObj - logMinObj;

        return (props: any) => {
            const { cx, cy, payload } = props;
            if (!payload || typeof payload.objective !== 'number' || !isFinite(payload.objective)) {
                return <circle cx={cx} cy={cy} r={0} fill="none" />;
            }

            const isBest = payload.best_objective !== undefined &&
                Math.abs(payload.objective - payload.best_objective) < 1e-10;

            let radius = maxSize;
            if (logRange > 0) {
                const logValue = Math.log(Math.max(1e-10, payload.objective));
                const normalized = (logMaxObj - logValue) / logRange;
                radius = maxSize + (minSize - maxSize) * normalized;
            } else if (isBest) {
                radius = minSize;
            }

            return <circle cx={cx} cy={cy} r={radius} fill={isBest ? "#e80f00" : "#fb923c"} />;
        };
    }, [minObj, maxObj]);

    useEffect(() => {
        checkStatus();
    }, []);

    useEffect(() => {
        return () => {
            if (eventSourceRef) eventSourceRef.close();
        };
    }, [eventSourceRef]);

    const checkStatus = async () => {
        const response = await getLayer3Status();
        if (response.data) {
            setIsRunning(response.data.running);
            setProgress(response.data.progress);
            setStage(response.data.stage);
            setMessage(response.data.message);
        }
    };

    const handleConfigUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setError(null);
        setSuccessMessage(null);
        const result = await uploadLayer3Config(file);
        if (result.error) {
            setError(result.error);
        } else {
            setSuccessMessage('Configuration uploaded successfully for Layer 3');
            setConfigLoaded(true);
        }
    };

    const handleRun = () => {
        setIsRunning(true);
        setIsStopping(false);
        setProgress(0);
        setStage('Initializing');
        setMessage('Starting Layer 3 thermal protection optimization...');
        setError(null);
        setSuccessMessage(null);
        setResults(null);
        setObjectiveHistory([]);

        const eventSource = runLayer3Optimization(
            settings,
            (event: Layer3ProgressEvent) => {
                if (event.type === 'status' || event.type === 'progress') {
                    if (event.progress !== undefined) setProgress(event.progress);
                    if (event.stage) setStage(event.stage);
                    if (event.message) setMessage(event.message);
                } else if (event.type === 'objective') {
                    // Real-time objective history updates
                    if (event.objective_history) {
                        setObjectiveHistory(prev => [...prev, ...event.objective_history!]);
                    }
                } else if (event.type === 'complete') {
                    setIsRunning(false);
                    setIsStopping(false);
                    setProgress(1);

                    if (event.results) {
                        setResults(event.results);
                    }

                    if (event.stopped_by_user) {
                        setStage('Stopped');
                        setMessage('Optimization stopped by user - using best solution found');
                    } else {
                        setStage('Complete');
                        setMessage('Optimization completed successfully');
                    }
                } else if (event.type === 'error') {
                    setIsRunning(false);
                    setIsStopping(false);
                    if (event.error && event.error.toLowerCase().includes('stopped')) {
                        setError(null);
                        setMessage('Optimization stopped by user');
                        setStage('Stopped');
                    } else {
                        setError(event.error || 'Optimization failed');
                    }
                    setEventSourceRef(null);
                }
            },
            (err) => {
                setIsRunning(false);
                setError(err);
                setEventSourceRef(null);
            }
        );
        setEventSourceRef(eventSource);
    };

    const handleStop = async () => {
        setIsStopping(true);
        setMessage('Stopping optimization...');
        try {
            await stopLayer3Optimization();
        } catch (err) {
            console.error('Failed to stop optimization:', err);
            if (eventSourceRef) {
                eventSourceRef.close();
                setEventSourceRef(null);
            }
            setIsRunning(false);
            setIsStopping(false);
            setStage('Stopped');
            setMessage('Optimization stopped');
        }
    };

    const downloadConfig = () => {
        if (!results?.config_yaml) return;
        const blob = new Blob([results.config_yaml], { type: 'text/yaml' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'layer3_optimized_config.yaml';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const downloadCSV = () => {
        if (!results?.time_array) return;
        let csv = 'Time (s),LOX Pressure (Pa),Fuel Pressure (Pa)\\n';
        results.time_array.forEach((t: number, i: number) => {
            csv += `${t},${results.lox_pressure[i]},${results.fuel_pressure[i]}\\n`;
        });
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'layer3_results.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    return (
        <div className="space-y-6">
            {/* Description */}
            <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl p-6">
                <h2 className="text-2xl font-bold text-[var(--color-text-primary)] mb-2">Layer 3: Thermal Protection Optimization</h2>
                <p className="text-sm text-[var(--color-text-secondary)]">
                    Layer 3 optimizes <strong>ablative liner and graphite insert thicknesses</strong> to meet recession requirements with safety margin while minimizing mass. Uses the pressure curves from Layer 2.
                </p>
            </div>

            {/* Controls */}
            <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl p-6">
                <div className="flex flex-wrap items-center gap-4">
                    <div className="flex-1 min-w-[200px]">
                        <h3 className="text-sm font-semibold text-[var(--color-text-secondary)] mb-2 uppercase tracking-wider">Start Here</h3>
                        <div className="flex gap-2">
                            <label className="flex-1">
                                <span className="sr-only">Upload Config</span>
                                <div className="relative group">
                                    <input
                                        type="file"
                                        accept=".yaml,.yml"
                                        onChange={handleConfigUpload}
                                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                                        disabled={isRunning}
                                    />
                                    <div className="px-4 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] text-sm font-medium hover:border-orange-500 transition-colors flex items-center justify-center gap-2">
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a2 2 0 002 2h12a2 2 0 002-2v-1M16 8l-4-4m0 0L8 8m4-4v12" />
                                        </svg>
                                        Upload Start Config
                                    </div>
                                </div>
                            </label>
                            <button
                                onClick={handleRun}
                                disabled={isRunning || (!requirements && !configLoaded)}
                                className={`px-6 py-2 rounded-lg font-bold text-white transition-all ${isRunning || (!requirements && !configLoaded)
                                    ? 'bg-gray-500 cursor-not-allowed'
                                    : 'bg-orange-600 hover:bg-orange-700 shadow-lg shadow-orange-500/20'
                                    }`}
                            >
                                {isRunning ? '🔄 Optimizing...' : '🔥 Run Layer 3'}
                            </button>
                            {isRunning && (
                                <button
                                    onClick={handleStop}
                                    disabled={isStopping}
                                    className={`px-6 py-2 text-white rounded-lg font-bold transition-all ${isStopping
                                        ? 'bg-yellow-600 cursor-wait'
                                        : 'bg-red-600 hover:bg-red-700'
                                        }`}
                                >
                                    {isStopping ? '⏳ Stopping...' : '⏹ Stop'}
                                </button>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Progress & Status */}
            {(isRunning || progress > 0 || error || successMessage) && (
                <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl p-6">
                    <div className="flex justify-between text-sm mb-2">
                        <span className="text-[var(--color-text-primary)] font-medium">{stage}</span>
                        <span className="text-orange-400 font-bold">{(progress * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-[var(--color-bg-primary)] rounded-full h-2 mb-4">
                        <div
                            className="bg-orange-500 h-full rounded-full transition-all duration-300"
                            style={{ width: `${progress * 100}%` }}
                        />
                    </div>
                    {message && <p className="text-sm text-[var(--color-text-secondary)]">{message}</p>}
                    {error && <p className="text-sm text-red-400 mt-2 font-medium">❌ {error}</p>}
                    {successMessage && <p className="text-sm text-green-400 mt-2 font-medium">✅ {successMessage}</p>}
                </div>
            )}

            {/* Visualizations */}
            <div className="grid grid-cols-1 gap-6">
                {/* Convergence History */}
                <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl p-6">
                    <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4 flex items-center gap-2">
                        <span className="text-orange-400">📈</span> Convergence History
                    </h3>
                    <div className="h-[300px] w-full">
                        {objectiveHistory.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={objectiveHistory} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" opacity={0.3} />
                                    <XAxis dataKey="iteration" stroke="var(--color-text-secondary)" tick={{ fontSize: 12 }} />
                                    <YAxis scale="log" domain={['auto', 'auto']} stroke="var(--color-text-secondary)" tick={{ fontSize: 12 }} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: 'var(--color-bg-primary)', border: '1px solid var(--color-border)', borderRadius: '8px' }}
                                        itemStyle={{ color: '#fff' }}
                                    />
                                    <Line type="monotone" dataKey="objective" stroke="#fb923c" strokeWidth={0} dot={renderDot} isAnimationActive={false} />
                                    <Line type="monotone" dataKey="best_objective" stroke="#f97316" strokeWidth={2} strokeDasharray="5 5" dot={false} isAnimationActive={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="h-full flex items-center justify-center text-[var(--color-text-secondary)] border border-dashed border-[var(--color-border)] rounded-lg">
                                Waiting for optimization data...
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Results */}
            {results && (
                <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl p-6">
                    <div className="flex justify-between items-center mb-6">
                        <h3 className="text-lg font-semibold text-[var(--color-text-primary)]">🔥 Final Optimization Results</h3>
                        <div className="flex gap-2">
                            <button
                                onClick={downloadCSV}
                                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
                            >
                                📊 Download CSV
                            </button>
                            <button
                                onClick={downloadConfig}
                                className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium transition-colors"
                            >
                                💾 Download Config (YAML)
                            </button>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <ResultCard
                            label="Ablative Thickness"
                            value={results.summary.optimized_ablative_thickness ? results.summary.optimized_ablative_thickness * 1000 : undefined}
                            unit="mm"
                            decimals={2}
                            color="orange"
                        />
                        <ResultCard
                            label="Graphite Thickness"
                            value={results.summary.optimized_graphite_thickness ? results.summary.optimized_graphite_thickness * 1000 : undefined}
                            unit="mm"
                            decimals={2}
                            color="red"
                        />
                        <ResultCard
                            label="Max Recession (Chamber)"
                            value={results.summary.max_recession_chamber ? results.summary.max_recession_chamber * 1000 : undefined}
                            unit="mm"
                            decimals={3}
                            color="yellow"
                        />
                        <ResultCard
                            label="Max Recession (Throat)"
                            value={results.summary.max_recession_throat ? results.summary.max_recession_throat * 1000 : undefined}
                            unit="mm"
                            decimals={3}
                            color="yellow"
                        />
                        <ResultCard
                            label="Total Impulse"
                            value={results.summary.total_impulse_Ns}
                            unit="N·s"
                            decimals={0}
                            color="blue"
                        />
                        <ResultCard
                            label="Burn Time"
                            value={results.summary.burn_time_s}
                            unit="s"
                            decimals={2}
                            color="cyan"
                        />
                        <ResultCard
                            label="Min Stability"
                            value={results.summary.min_stability_margin}
                            decimals={3}
                            color="purple"
                        />
                        <ResultCard
                            label="Thermal Protection"
                            value={results.summary.thermal_protection_valid ? 'VALID' : 'INVALID'}
                            isText
                            color={results.summary.thermal_protection_valid ? 'green' : 'red'}
                        />
                    </div>
                </div>
            )}
        </div>
    );
}
