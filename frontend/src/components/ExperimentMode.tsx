import { useState, useRef } from 'react';
import type { EngineConfig, TimeSeriesData, TimeSeriesSummary } from '../api/client';
import { API_BASE } from '../api/client';
import { PressureCurveChart } from './PressureCurveChart';
import {
  ComposedChart, Scatter, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';

interface ExperimentModeProps {
  config: EngineConfig | null;
}

interface RowInput {
  id: number;
  label: string;
  t0: string;
  tf: string;
  deltaP: string;   // psi
  weight: string;   // kg — cumulative tank weight reading
}

interface TimeseriesResponse {
  status: string;
  t: number[];
  results: {
    Pc: number[];
    mdot_O: number[];
    mdot_F: number[];
    F: number[];
    Isp: number[];
    MR: number[];
    Cd_O: number[];
    Cd_F: number[];
    delta_p_injector_O: number[];
    delta_p_injector_F: number[];
  };
  fuel_cd_pressure_pairs: [number, number][];
  lox_cd_pressure_pairs:  [number, number][];
  pressure_curves_used?: {
    P_tank_O_pa: number[];
    P_tank_F_pa: number[];
  };
  cd_fit?: {
    fuel: { model: string; a: number; b: number };
    lox: { model: string; a: number; b: number };
  };
  re_similarity?: {
    mean_re_water_fuel: number;
    mean_re_water_lox: number;
    mean_re_hotfire_fuel: number;
    mean_re_hotfire_lox: number;
    ratio_hotfire_to_water_fuel: number | null;
    ratio_hotfire_to_water_lox: number | null;
    fuel_within_two_orders: boolean;
    lox_within_two_orders: boolean;
  };
}

const PSI_TO_PA = 6894.757;

function fmtN(n: number, d = 4): string { return n.toFixed(d); }
function fmtSci(n: number): string {
  if (n === 0) return '0';
  return Math.abs(Math.floor(Math.log10(Math.abs(n)))) >= 4 ? n.toExponential(3) : n.toPrecision(4);
}
function avg(arr: number[]): number { return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0; }

// ---------------------------------------------------------------------------
// Column definitions
// ---------------------------------------------------------------------------

type CellPos = { row: number; col: number };

interface InputCol {
  kind: 'input';
  key: string;
  inputKey: keyof Omit<RowInput, 'id' | 'label'>;
  header: string;
  width: number;
}
interface ComputedCol {
  kind: 'computed';
  key: string;
  header: string;
  width: number;
  compute: (row: RowInput, prevRow: RowInput | null, chokeDiamM: number, waterRho: number) => string;
  highlight?: (value: string) => 'good' | 'warn' | null;
}
type ColDef = InputCol | ComputedCol;

const COLS: ColDef[] = [
  { kind: 'input',    key: 't0',      inputKey: 't0',      header: 'T₀ [s]',      width: 90  },
  { kind: 'input',    key: 'tf',      inputKey: 'tf',      header: 'Tf [s]',       width: 90  },
  { kind: 'input',    key: 'deltaP',  inputKey: 'deltaP',  header: 'ΔP [psi]',     width: 95  },
  { kind: 'input',    key: 'weight',  inputKey: 'weight',  header: 'Weight [kg]',  width: 110 },
  {
    kind: 'computed', key: 'mdot', header: 'ṁ [kg/s]', width: 105,
    compute: (row, prevRow) => {
      const dt = parseFloat(row.tf) - parseFloat(row.t0);
      const dw = parseFloat(row.weight) - (prevRow ? parseFloat(prevRow.weight) : 0);
      if (!isFinite(dt) || dt <= 0 || !isFinite(dw) || dw < 0) return '—';
      return (dw / dt).toFixed(5);
    },
  },
  {
    kind: 'computed', key: 'cd', header: 'Cd', width: 80,
    compute: (row, prevRow, chokeDiamM, waterRho) => {
      const dt = parseFloat(row.tf) - parseFloat(row.t0);
      const dw = parseFloat(row.weight) - (prevRow ? parseFloat(prevRow.weight) : 0);
      const dp = parseFloat(row.deltaP) * PSI_TO_PA;
      if (!isFinite(dt) || dt <= 0 || !isFinite(dw) || dw < 0 || !isFinite(dp) || dp <= 0 || chokeDiamM <= 0 || waterRho <= 0) return '—';
      const area = Math.PI * (chokeDiamM / 2) ** 2;
      const mdot = dw / dt;
      return (mdot / (area * Math.sqrt(2 * waterRho * dp))).toFixed(4);
    },
  },
  {
    kind: 'computed', key: 'cn', header: 'CN', width: 75,
    compute: (row) => {
      const dp = parseFloat(row.deltaP) * PSI_TO_PA;
      if (!isFinite(dp) || dp <= 0) return '—';
      return (dp / 2337).toFixed(2);
    },
    highlight: (val) => val === '—' ? null : parseFloat(val) >= 2 ? 'good' : 'warn',
  },
  {
    kind: 'computed', key: 're_water', header: 'Re', width: 90,
    compute: (row, prevRow, chokeDiamM, waterRho) => {
      const dt = parseFloat(row.tf) - parseFloat(row.t0);
      const dw = parseFloat(row.weight) - (prevRow ? parseFloat(prevRow.weight) : 0);
      if (!isFinite(dt) || dt <= 0 || !isFinite(dw) || dw < 0 || chokeDiamM <= 0 || waterRho <= 0) return '—';
      const area  = Math.PI * (chokeDiamM / 2) ** 2;
      const mdot  = dw / dt;
      const v     = mdot / (waterRho * area);
      const re    = waterRho * v * chokeDiamM / 0.001;
      return re.toExponential(2);
    },
  },
];

const INPUT_COLS = COLS.filter((c): c is InputCol => c.kind === 'input');
const INPUT_COL_INDICES = COLS.map((c, i) => c.kind === 'input' ? i : -1).filter(i => i >= 0);
const FIRST_INPUT = INPUT_COL_INDICES[0];
const LAST_INPUT  = INPUT_COL_INDICES[INPUT_COL_INDICES.length - 1];

function newRow(n: number): RowInput {
  return { id: Date.now() + Math.random(), label: `Run ${n}`, t0: '', tf: '', deltaP: '', weight: '' };
}

// ---------------------------------------------------------------------------
// Spreadsheet grid
// ---------------------------------------------------------------------------

function SpreadsheetGrid({
  rows,
  onChange,
  chokeDiameterMm,
  waterDensity,
}: {
  rows: RowInput[];
  onChange: (rows: RowInput[]) => void;
  chokeDiameterMm: string;
  waterDensity: string;
}) {
  const [anchor, setAnchor]   = useState<CellPos | null>(null);
  const [active, setActive]   = useState<CellPos | null>(null);
  const [editing, setEditing] = useState<CellPos | null>(null);
  const [editVal, setEditVal] = useState('');
  const gridRef    = useRef<HTMLDivElement>(null);
  const inputRef   = useRef<HTMLInputElement>(null);
  const isDragging = useRef(false);

  const chokeDiamM = parseFloat(chokeDiameterMm) / 1000;
  const waterRho   = parseFloat(waterDensity) || 998.2;
  const numCols = COLS.length;
  const numRows = rows.length;

  const sel = anchor && active ? {
    r0: Math.min(anchor.row, active.row), r1: Math.max(anchor.row, active.row),
    c0: Math.min(anchor.col, active.col), c1: Math.max(anchor.col, active.col),
  } : null;

  function inSel(r: number, c: number) {
    return !!sel && r >= sel.r0 && r <= sel.r1 && c >= sel.c0 && c <= sel.c1;
  }
  function isActiveCell(r: number, c: number) { return active?.row === r && active?.col === c; }

  function getCellValue(r: number, c: number): string {
    const col = COLS[c];
    const row = rows[r];
    if (!row) return '';
    if (col.kind === 'input') return String(row[col.inputKey] ?? '');
    return col.compute(row, rows[r - 1] ?? null, chokeDiamM, waterRho);
  }

  function setCell(r: number, c: number, v: string) {
    const col = COLS[c];
    if (col.kind !== 'input') return;
    onChange(rows.map((row, i) => i === r ? { ...row, [col.inputKey]: v } : row));
  }

  function moveTo(r: number, c: number, extend = false) {
    r = Math.max(0, Math.min(r, numRows - 1));
    c = Math.max(0, Math.min(c, numCols - 1));
    setActive({ row: r, col: c });
    if (!extend) setAnchor({ row: r, col: c });
    gridRef.current?.focus();
  }

  function tabMoveTo(r: number, c: number, shift: boolean) {
    const pos = INPUT_COL_INDICES.indexOf(c);
    const effectivePos = pos >= 0 ? pos : (shift ? INPUT_COL_INDICES.length - 1 : 0);
    if (!shift && effectivePos === INPUT_COL_INDICES.length - 1) {
      if (r === numRows - 1) {
        const next = [...rows, newRow(numRows + 1)];
        onChange(next);
        setTimeout(() => moveTo(r + 1, FIRST_INPUT), 0);
      } else {
        moveTo(r + 1, FIRST_INPUT);
      }
    } else if (shift && effectivePos === 0) {
      if (r > 0) moveTo(r - 1, LAST_INPUT);
    } else {
      moveTo(r, INPUT_COL_INDICES[effectivePos + (shift ? -1 : 1)]);
    }
  }

  function startEdit(r: number, c: number, initial?: string) {
    if (COLS[c].kind !== 'input') return;
    setEditing({ row: r, col: c });
    setEditVal(initial !== undefined ? initial : getCellValue(r, c));
    setTimeout(() => { inputRef.current?.focus(); inputRef.current?.select(); }, 0);
  }

  function commitEdit() {
    if (!editing) return;
    setCell(editing.row, editing.col, editVal);
    setEditing(null);
  }

  function cancelEdit() { setEditing(null); gridRef.current?.focus(); }

  function handleGridKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
    if (!active) return;
    const { row: r, col: c } = active;
    if (editing) return;

    if (e.key === 'Enter' || e.key === 'F2') { e.preventDefault(); startEdit(r, c); return; }
    if (e.key === 'Tab')       { e.preventDefault(); tabMoveTo(r, c, e.shiftKey); return; }
    if (e.key === 'Escape')    { setAnchor({ row: r, col: c }); return; }
    if (e.key === 'ArrowUp')   { e.preventDefault(); moveTo(r - 1, c, e.shiftKey); return; }
    if (e.key === 'ArrowDown') { e.preventDefault(); moveTo(r + 1, c, e.shiftKey); return; }
    if (e.key === 'ArrowLeft') { e.preventDefault(); moveTo(r, c - 1, e.shiftKey); return; }
    if (e.key === 'ArrowRight'){ e.preventDefault(); moveTo(r, c + 1, e.shiftKey); return; }
    if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      if (sel) {
        onChange(rows.map((row, ri) => {
          if (ri < sel.r0 || ri > sel.r1) return row;
          const patch: Partial<RowInput> = {};
          for (let ci = sel.c0; ci <= sel.c1; ci++) {
            const col = COLS[ci];
            if (col.kind === 'input') patch[col.inputKey] = '';
          }
          return { ...row, ...patch };
        }));
      }
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
      e.preventDefault();
      if (sel) {
        const tsv: string[] = [];
        for (let ri = sel.r0; ri <= sel.r1; ri++)
          tsv.push(Array.from({ length: sel.c1 - sel.c0 + 1 }, (_, ci) => getCellValue(ri, sel.c0 + ci)).join('\t'));
        navigator.clipboard.writeText(tsv.join('\n'));
      }
      return;
    }
    if (!e.ctrlKey && !e.metaKey && !e.altKey && e.key.length === 1 && COLS[c].kind === 'input') {
      startEdit(r, c, e.key);
    }
  }

  function handleGridPaste(e: React.ClipboardEvent<HTMLDivElement>) {
    if (editing || !active) return;
    e.preventDefault();
    const startInputPos = INPUT_COL_INDICES.indexOf(active.col);
    if (startInputPos === -1) return;

    const text = e.clipboardData.getData('text');
    const pastedRows = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trimEnd()
      .split('\n').map(r => r.split('\t'));

    const next = rows.map(r => ({ ...r }));
    pastedRows.forEach((pr, ri) => {
      const ti = active.row + ri;
      while (next.length <= ti) next.push(newRow(next.length + 1));
      pr.forEach((val, ci) => {
        const inputIdx = startInputPos + ci;
        if (inputIdx < INPUT_COL_INDICES.length) {
          const col = COLS[INPUT_COL_INDICES[inputIdx]] as InputCol;
          next[ti] = { ...next[ti], [col.inputKey]: val.trim() };
        }
      });
    });
    onChange(next);
  }

  function handleCellMouseDown(e: React.MouseEvent, r: number, c: number) {
    if (editing) commitEdit();
    if (e.shiftKey && anchor) { setActive({ row: r, col: c }); }
    else { setAnchor({ row: r, col: c }); setActive({ row: r, col: c }); }
    isDragging.current = true;
    gridRef.current?.focus();
  }

  function handleCellMouseEnter(r: number, c: number) {
    if (isDragging.current) setActive({ row: r, col: c });
  }

  function handleInputKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!editing) return;
    const { row: r, col: c } = editing;
    if (e.key === 'Enter')  { e.preventDefault(); commitEdit(); moveTo(r + 1, c); }
    else if (e.key === 'Escape') { e.preventDefault(); cancelEdit(); }
    else if (e.key === 'Tab')    { e.preventDefault(); commitEdit(); tabMoveTo(r, c, e.shiftKey); }
    else if (e.key === 'ArrowUp' && (e.target as HTMLInputElement).selectionStart === 0) {
      e.preventDefault(); commitEdit(); moveTo(r - 1, c);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault(); commitEdit(); moveTo(r + 1, c);
    }
  }

  const borderColor  = 'rgba(255,255,255,0.1)';
  const headerBg     = 'rgba(255,255,255,0.05)';
  const computedBg   = 'rgba(255,255,255,0.02)';
  const selBg        = 'rgba(59,130,246,0.18)';
  const activeBorder = '2px solid #3b82f6';
  const rowNumBg     = 'rgba(255,255,255,0.03)';

  const cellBase: React.CSSProperties = {
    border: `1px solid ${borderColor}`,
    padding: 0,
    position: 'relative',
    height: 26,
    verticalAlign: 'middle',
    cursor: 'default',
  };

  return (
    <div
      ref={gridRef}
      tabIndex={0}
      onKeyDown={handleGridKeyDown}
      onPaste={handleGridPaste}
      onMouseUp={() => { isDragging.current = false; }}
      onMouseLeave={() => { isDragging.current = false; }}
      className="outline-none"
      style={{ userSelect: 'none', overflowX: 'auto' }}
    >
      <table style={{ borderCollapse: 'collapse', fontSize: 13, fontFamily: 'ui-monospace, monospace', tableLayout: 'fixed' }}>
        <colgroup>
          <col style={{ width: 36 }} />
          {COLS.map(c => <col key={c.key} style={{ width: c.width }} />)}
          <col style={{ width: 28 }} />
        </colgroup>
        <thead>
          <tr>
            <th style={{ ...cellBase, background: headerBg }} />
            {COLS.map(col => (
              <th key={col.key} style={{
                ...cellBase,
                background: col.kind === 'computed' ? computedBg : headerBg,
                padding: '0 6px',
                textAlign: 'left',
                fontSize: 11,
                fontWeight: 600,
                color: col.kind === 'computed' ? 'rgba(255,255,255,0.3)' : 'var(--color-text-secondary)',
                letterSpacing: '0.04em',
                whiteSpace: 'nowrap',
                fontStyle: col.kind === 'computed' ? 'italic' : undefined,
              }}>
                {col.header}
              </th>
            ))}
            <th style={{ ...cellBase, background: headerBg }} />
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={row.id}>
              <td style={{ ...cellBase, background: rowNumBg, textAlign: 'center', fontSize: 11, color: 'var(--color-text-secondary)' }}>
                {ri + 1}
              </td>

              {COLS.map((col, ci) => {
                const isEditingThis = editing?.row === ri && editing?.col === ci;
                const selected      = inSel(ri, ci);
                const isActive      = isActiveCell(ri, ci);
                const isComputed    = col.kind === 'computed';
                const displayVal    = getCellValue(ri, ci);
                const hlState       = isComputed && col.highlight ? col.highlight(displayVal) : null;

                return (
                  <td
                    key={col.key}
                    onMouseDown={e => handleCellMouseDown(e, ri, ci)}
                    onMouseEnter={() => handleCellMouseEnter(ri, ci)}
                    onDoubleClick={() => startEdit(ri, ci)}
                    style={{
                      ...cellBase,
                      background: selected ? selBg
                            : hlState === 'good' ? 'rgba(34,197,94,0.12)'
                            : hlState === 'warn' ? 'rgba(239,68,68,0.12)'
                            : isComputed ? computedBg : 'var(--color-bg-primary)',
                      outline: isActive && !isEditingThis ? activeBorder : undefined,
                      outlineOffset: '-2px',
                      overflow: 'hidden',
                    }}
                  >
                    {isEditingThis ? (
                      <input
                        ref={inputRef}
                        value={editVal}
                        onChange={e => setEditVal(e.target.value)}
                        onBlur={commitEdit}
                        onKeyDown={handleInputKeyDown}
                        style={{
                          width: '100%', height: '100%',
                          border: activeBorder, outline: 'none',
                          background: 'var(--color-bg-primary)',
                          color: 'var(--color-text-primary)',
                          padding: '1px 5px',
                          fontFamily: 'inherit', fontSize: 'inherit',
                          boxSizing: 'border-box',
                          position: 'absolute', top: 0, left: 0, zIndex: 2,
                        }}
                      />
                    ) : (
                      <div style={{
                        padding: '1px 5px',
                        color: hlState === 'good' ? '#4ade80'
                          : hlState === 'warn' ? '#f87171'
                          : isComputed
                            ? (displayVal === '—' ? 'rgba(255,255,255,0.2)' : '#f59e0b')
                            : 'var(--color-text-primary)',
                        whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                      }}>
                        {displayVal}
                      </div>
                    )}
                  </td>
                );
              })}

              <td style={{ ...cellBase, background: rowNumBg, textAlign: 'center' }}>
                {rows.length > 1 && (
                  <button
                    tabIndex={-1}
                    onMouseDown={e => { e.stopPropagation(); onChange(rows.filter((_, idx) => idx !== ri)); }}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--color-text-secondary)', fontSize: 12, padding: '0 4px' }}
                  >×</button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Grid panel (grid + header + add/copy buttons)
// ---------------------------------------------------------------------------

function GridPanel({
  title,
  accentColor,
  rows,
  onChange,
  chokeDiameterMm,
  onChokeDiamChange,
  waterDensity,
}: {
  title: string;
  accentColor: string;
  rows: RowInput[];
  onChange: (rows: RowInput[]) => void;
  chokeDiameterMm: string;
  onChokeDiamChange: (v: string) => void;
  waterDensity: string;
}) {
  const chokeDiamM = parseFloat(chokeDiameterMm) / 1000;
  const waterRho   = parseFloat(waterDensity) || 998.2;

  function copyAll() {
    const header = COLS.map(c => c.header).join('\t');
    const computedCols = COLS.filter((c): c is ComputedCol => c.kind === 'computed');
    const body = rows.map((r, i) => {
      const prev = rows[i - 1] ?? null;
      return INPUT_COLS.map(c => r[c.inputKey]).join('\t')
        + '\t' + computedCols.map(c => c.compute(r, prev, chokeDiamM, waterRho)).join('\t');
    }).join('\n');
    navigator.clipboard.writeText(header + '\n' + body);
  }

  const inputCls = "w-full px-2 py-1.5 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)] text-[var(--color-text-primary)] text-sm focus:outline-none";

  return (
    <div className="p-4 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)] flex flex-col gap-3 min-w-0">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wider" style={{ color: accentColor }}>{title}</h3>
        <div className="flex gap-2">
          <button
            onClick={() => onChange([...rows, newRow(rows.length + 1)])}
            className="px-2.5 py-1 text-xs font-medium rounded-lg border transition-colors"
            style={{ borderColor: `${accentColor}40`, color: accentColor, background: `${accentColor}10` }}
          >
            + Add Row
          </button>
          <button
            onClick={copyAll}
            className="px-2.5 py-1 text-xs font-medium rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors"
          >
            Copy All
          </button>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <label className="text-xs text-[var(--color-text-secondary)] whitespace-nowrap">Choke Ø [mm]</label>
        <input
          type="number" step="0.01" min="0"
          value={chokeDiameterMm}
          onChange={e => onChokeDiamChange(e.target.value)}
          className={inputCls}
          placeholder="3.0"
          style={{ maxWidth: 100 }}
        />
      </div>

      <SpreadsheetGrid rows={rows} onChange={onChange} chokeDiameterMm={chokeDiameterMm} waterDensity={waterDensity} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function ExperimentMode({ config }: ExperimentModeProps) {
  const [chokeDiamFuelMm,  setChokeDiamFuelMm]  = useState('3.0');
  const [chokeDiamLoxMm,   setChokeDiamLoxMm]   = useState('3.0');
  const [waterDensity,     setWaterDensity]      = useState('998.2');
  const [fuelRows, setFuelRows] = useState<RowInput[]>([
    { id: 1, label: 'Run 1', t0: '', tf: '', deltaP: '', weight: '' },
  ]);
  const [loxRows, setLoxRows] = useState<RowInput[]>([
    { id: 2, label: 'Run 1', t0: '', tf: '', deltaP: '', weight: '' },
  ]);
  const [results,  setResults]  = useState<TimeseriesResponse | null>(null);
  const [error,    setError]    = useState<string | null>(null);
  const [loading,  setLoading]  = useState(false);

  const inputCls = "w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)] text-[var(--color-text-primary)] text-sm focus:outline-none focus:border-amber-500";

  async function handleCalculate() {
    setError(null);
    setResults(null);

    const chokeFuelM = parseFloat(chokeDiamFuelMm) / 1000;
    const chokeLoxM  = parseFloat(chokeDiamLoxMm)  / 1000;
    if (isNaN(chokeFuelM) || chokeFuelM <= 0) { setError('Enter a valid Fuel choke diameter (mm).'); return; }
    if (isNaN(chokeLoxM)  || chokeLoxM  <= 0) { setError('Enter a valid LOX choke diameter (mm).'); return; }

    function buildRows(rows: RowInput[], label: string) {
      const parsed = rows.map((r, i) => ({
        label: r.label || `Run ${i + 1}`,
        t0: parseFloat(r.t0),
        tf: parseFloat(r.tf),
        delta_p_psi: parseFloat(r.deltaP),
        weight: parseFloat(r.weight),
      }));
      for (const r of parsed) {
        if (isNaN(r.t0) || isNaN(r.tf) || isNaN(r.delta_p_psi) || isNaN(r.weight)) {
          setError(`${label} row "${r.label}": all fields must be filled in.`);
          return null;
        }
      }
      return parsed;
    }

    const parsedFuel = buildRows(fuelRows, 'Fuel');
    if (!parsedFuel) return;
    const parsedLox  = buildRows(loxRows,  'LOX');
    if (!parsedLox)  return;

    setLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/experiment/run_timeseries`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fuel: {
            choke_diameter_m: chokeFuelM,
            water_density: parseFloat(waterDensity) || 998.2,
            rows: parsedFuel,
          },
          lox: {
            choke_diameter_m: chokeLoxM,
            water_density: parseFloat(waterDensity) || 998.2,
            rows: parsedLox,
          },
        }),
      });
      if (!resp.ok) {
        const e = await resp.json().catch(() => ({}));
        setError(e.detail || `HTTP ${resp.status}`);
        return;
      }
      setResults(await resp.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Network error');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="p-5 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
        <h2 className="text-xl font-bold text-[var(--color-text-primary)] mb-1">Cold-Flow Experiment</h2>
        <p className="text-sm text-[var(--color-text-secondary)]">
          Characterize injector Cd from water flow tests, then run the engine time-series with interpolated per-step Cd.
        </p>
      </div>

      {/* Water density config */}
      <div className="p-4 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
        <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider mb-3">Test Configuration</h3>
        <div className="flex items-center gap-4">
          <div style={{ maxWidth: 200 }}>
            <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Water Density [kg/m³]</label>
            <input type="number" step="0.1" min="0" value={waterDensity} onChange={e => setWaterDensity(e.target.value)} className={inputCls} placeholder="998.2" />
          </div>
          {!config && (
            <p className="text-xs text-amber-400 mt-4">No engine config loaded — time-series simulation requires a loaded config.</p>
          )}
        </div>
      </div>

      {/* Two spreadsheet grids side by side */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <GridPanel
          title="Fuel — Test Runs"
          accentColor="#f97316"
          rows={fuelRows}
          onChange={setFuelRows}
          chokeDiameterMm={chokeDiamFuelMm}
          onChokeDiamChange={setChokeDiamFuelMm}
          waterDensity={waterDensity}
        />
        <GridPanel
          title="LOX — Test Runs"
          accentColor="#60a5fa"
          rows={loxRows}
          onChange={setLoxRows}
          chokeDiameterMm={chokeDiamLoxMm}
          onChokeDiamChange={setChokeDiamLoxMm}
          waterDensity={waterDensity}
        />
      </div>

      {/* Calculate button */}
      <button
        onClick={handleCalculate}
        disabled={loading}
        className="w-full py-3 rounded-xl font-semibold text-sm transition-colors bg-amber-500 hover:bg-amber-400 text-black disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
            </svg>Running simulation…
          </span>
        ) : 'Run Time-Series Simulation'}
      </button>

      {error && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm">{error}</div>
      )}

      {/* Results */}
      {results && <TimeseriesResults results={results} />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Cd vs sqrt(ΔP) characterisation chart
// ---------------------------------------------------------------------------

function CdFitChart({
  fuelPairs, loxPairs, cdFit,
}: {
  fuelPairs: [number, number][];
  loxPairs:  [number, number][];
  cdFit?: { fuel: { a: number; b: number }; lox: { a: number; b: number } };
}) {
  const allDp_pa = [...fuelPairs.map(p => p[0]), ...loxPairs.map(p => p[0])];
  if (!allDp_pa.length) return null;

  const minDp_pa = Math.min(...allDp_pa) * 0.5;
  const maxDp_pa = Math.max(...allDp_pa) * 1.8;
  const N = 80;

  function evalFit(dp_pa: number, a: number, b: number) {
    return Math.max(0, a * Math.sqrt(dp_pa) + b);
  }

  // Build one merged dataset keyed by sqrt(ΔP [psi])
  // Each point has: sqrtDp, and optionally fuelFit, loxFit, fuelMeas, loxMeas
  type Pt = { sqrtDp: number; fuelFit?: number; loxFit?: number; fuelMeas?: number; loxMeas?: number };
  const byKey = new Map<number, Pt>();

  function getOrCreate(sqrtDp: number): Pt {
    const key = Math.round(sqrtDp * 1e6);
    if (!byKey.has(key)) byKey.set(key, { sqrtDp });
    return byKey.get(key)!;
  }

  // Fit curves
  if (cdFit) {
    for (let i = 0; i < N; i++) {
      const dp_pa = minDp_pa + (maxDp_pa - minDp_pa) * i / (N - 1);
      const sqrtDp = Math.sqrt(dp_pa / PSI_TO_PA);
      const pt = getOrCreate(sqrtDp);
      pt.fuelFit = evalFit(dp_pa, cdFit.fuel.a, cdFit.fuel.b);
      pt.loxFit  = evalFit(dp_pa, cdFit.lox.a,  cdFit.lox.b);
    }
  }

  // Measured points
  fuelPairs.forEach(([dp_pa, cd]) => {
    const pt = getOrCreate(Math.sqrt(dp_pa / PSI_TO_PA));
    pt.fuelMeas = cd;
  });
  loxPairs.forEach(([dp_pa, cd]) => {
    const pt = getOrCreate(Math.sqrt(dp_pa / PSI_TO_PA));
    pt.loxMeas = cd;
  });

  const chartData = Array.from(byKey.values()).sort((a, b) => a.sqrtDp - b.sqrtDp);

  const tip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null;
    return (
      <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-lg p-3 shadow-lg text-xs">
        <p className="text-[var(--color-text-secondary)] mb-1">ΔP = {(Number(label) ** 2).toFixed(1)} psi &nbsp;(√ΔP = {Number(label).toFixed(3)} psi^½)</p>
        {payload.map((e: any) => (
          <p key={e.dataKey} style={{ color: e.color }}>{e.name}: {Number(e.value).toFixed(4)}</p>
        ))}
      </div>
    );
  };

  return (
    <div className="p-4 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
      <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider mb-1">
        Cd Characterisation — Cold Flow Fit
      </h3>
      <p className="text-xs text-[var(--color-text-secondary)] mb-4">
        Cd = a·√ΔP<sub>pa</sub> + b &nbsp;·&nbsp; dots = measured, lines = extrapolated fit
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" opacity={0.5} />
          <XAxis
            dataKey="sqrtDp"
            type="number"
            domain={['auto', 'auto']}
            stroke="var(--color-text-secondary)"
            tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
            tickFormatter={(v: number) => v.toFixed(2)}
            label={{ value: '√ΔP (psi^½)', position: 'insideBottom', offset: -10, fill: 'var(--color-text-secondary)', fontSize: 11 }}
          />
          <YAxis
            stroke="var(--color-text-secondary)"
            tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
            label={{ value: 'Cd', angle: -90, position: 'insideLeft', fill: 'var(--color-text-secondary)', fontSize: 11 }}
            domain={['auto', 'auto']}
          />
          <Tooltip content={tip} />
          <Legend />
          {/* Fit lines */}
          <Line type="monotone" dataKey="fuelFit" name="Fuel fit" stroke="#f97316" strokeWidth={2} dot={false} connectNulls />
          <Line type="monotone" dataKey="loxFit"  name="LOX fit"  stroke="#60a5fa" strokeWidth={2} dot={false} connectNulls />
          {/* Measured scatter — lines hidden, only dots */}
          <Line type="monotone" dataKey="fuelMeas" name="Fuel meas" stroke="#f97316" strokeWidth={0} dot={{ r: 5, fill: '#f97316', strokeWidth: 2, stroke: '#fff' }} connectNulls={false} />
          <Line type="monotone" dataKey="loxMeas"  name="LOX meas"  stroke="#60a5fa" strokeWidth={0} dot={{ r: 5, fill: '#60a5fa', strokeWidth: 2, stroke: '#fff' }} connectNulls={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Convert experiment response → TimeSeriesData / TimeSeriesSummary for PressureCurveChart
// ---------------------------------------------------------------------------

function convertToTimeSeriesFormat(results: TimeseriesResponse): { data: TimeSeriesData; summary: TimeSeriesSummary } {
  const R = results.results;
  const t = results.t;
  const n = t.length;

  const nonZeroF   = R.F.filter(v => v > 0);
  const nonZeroPc  = R.Pc.filter(v => v > 0);
  const nonZeroIsp = R.Isp.filter(v => v > 0);

  const avgF   = nonZeroF.length   ? nonZeroF.reduce((a, b) => a + b, 0) / nonZeroF.length   : 0;
  const maxF   = nonZeroF.length   ? Math.max(...nonZeroF)   : 0;
  const minF   = nonZeroF.length   ? Math.min(...nonZeroF)   : 0;
  const avgPc  = nonZeroPc.length  ? nonZeroPc.reduce((a, b) => a + b, 0) / nonZeroPc.length  : 0;
  const maxPc  = nonZeroPc.length  ? Math.max(...nonZeroPc)  : 0;
  const avgIsp = nonZeroIsp.length ? nonZeroIsp.reduce((a, b) => a + b, 0) / nonZeroIsp.length : 0;

  let totalImpulse = 0;
  for (let i = 1; i < n; i++)
    totalImpulse += (R.F[i] + R.F[i - 1]) * 0.5 * (t[i] - t[i - 1]);

  const mdotTotal = R.mdot_O.map((mo, i) => mo + R.mdot_F[i]);
  let totalPropellant = 0;
  for (let i = 1; i < n; i++)
    totalPropellant += (mdotTotal[i] + mdotTotal[i - 1]) * 0.5 * (t[i] - t[i - 1]);

  const pTankO = results.pressure_curves_used?.P_tank_O_pa ?? new Array(n).fill(0);
  const pTankF = results.pressure_curves_used?.P_tank_F_pa ?? new Array(n).fill(0);

  const data: TimeSeriesData = {
    time:             t,
    P_tank_O_psi:     pTankO.map(p => p / PSI_TO_PA),
    P_tank_F_psi:     pTankF.map(p => p / PSI_TO_PA),
    Pc_psi:           R.Pc.map(p => p / PSI_TO_PA),
    thrust_kN:        R.F.map(f => f / 1000),
    Isp_s:            R.Isp,
    MR:               R.MR,
    mdot_O_kg_s:      R.mdot_O,
    mdot_F_kg_s:      R.mdot_F,
    mdot_total_kg_s:  mdotTotal,
    cstar_actual_m_s: new Array(n).fill(0),
    gamma:            new Array(n).fill(0),
    Cd_O:                    R.Cd_O,
    Cd_F:                    R.Cd_F,
    delta_P_injector_O_psi:  R.delta_p_injector_O.map(p => p / PSI_TO_PA),
    delta_P_injector_F_psi:  R.delta_p_injector_F.map(p => p / PSI_TO_PA),
  };

  const summary: TimeSeriesSummary = {
    avg_thrust_kN:       avgF / 1000,
    peak_thrust_kN:      maxF / 1000,
    min_thrust_kN:       minF / 1000,
    avg_Pc_psi:          avgPc / PSI_TO_PA,
    peak_Pc_psi:         maxPc / PSI_TO_PA,
    avg_Isp_s:           avgIsp,
    total_impulse_kNs:   totalImpulse / 1000,
    total_propellant_kg: totalPropellant,
    burn_time_s:         n > 1 ? t[n - 1] - t[0] : 0,
  };

  return { data, summary };
}

// ---------------------------------------------------------------------------
// Reynolds similarity (hotfire mean Re vs cold-flow mean Re, within 10⁻²…10²)
// ---------------------------------------------------------------------------

type ReSimilarity = NonNullable<TimeseriesResponse['re_similarity']>;

function ReSimilarityPanel({ re }: { re: ReSimilarity }) {
  const rows: {
    label: string;
    accent: string;
    meanWater: number;
    meanHot: number;
    ratio: number | null;
    ok: boolean;
  }[] = [
    {
      label: 'Fuel',
      accent: '#f97316',
      meanWater: re.mean_re_water_fuel,
      meanHot: re.mean_re_hotfire_fuel,
      ratio: re.ratio_hotfire_to_water_fuel,
      ok: re.fuel_within_two_orders,
    },
    {
      label: 'LOX',
      accent: '#60a5fa',
      meanWater: re.mean_re_water_lox,
      meanHot: re.mean_re_hotfire_lox,
      ratio: re.ratio_hotfire_to_water_lox,
      ok: re.lox_within_two_orders,
    },
  ];

  const allOk = rows.every(r => r.ok);

  return (
    <div
      className={`p-4 rounded-xl border ${
        allOk ? 'bg-emerald-500/5 border-emerald-500/25' : 'bg-amber-500/5 border-amber-500/30'
      }`}
    >
      <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider mb-1">
        Reynolds similarity (injector, choke-based)
      </h3>
      <p className="text-xs text-[var(--color-text-secondary)] mb-4">
        Mean Re from the time-series (hot, propellant properties) compared to the mean Re from water test rows.
        Pass if hotfire/water ratio is between 10<sup>-2</sup> and 10<sup>2</sup> (two orders of magnitude).
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {rows.map(row => (
          <div
            key={row.label}
            className="p-3 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)]"
          >
            <div className="flex items-center justify-between gap-2 mb-2">
              <span className="text-sm font-semibold" style={{ color: row.accent }}>{row.label}</span>
              <span
                className={`text-xs font-semibold px-2 py-0.5 rounded ${
                  row.ok ? 'bg-emerald-500/15 text-emerald-400' : 'bg-red-500/15 text-red-400'
                }`}
              >
                {row.ok ? 'Pass' : 'Outside range'}
              </span>
            </div>
            <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs font-mono">
              <dt className="text-[var(--color-text-secondary)]">Mean Re (water)</dt>
              <dd className="text-right text-[var(--color-text-primary)]">{fmtSci(row.meanWater)}</dd>
              <dt className="text-[var(--color-text-secondary)]">Mean Re (hotfire)</dt>
              <dd className="text-right" style={{ color: row.accent }}>{fmtSci(row.meanHot)}</dd>
              <dt className="text-[var(--color-text-secondary)]">Ratio (hot / water)</dt>
              <dd className="text-right text-[var(--color-text-primary)]">
                {row.ratio != null && Number.isFinite(row.ratio) ? fmtSci(row.ratio) : '—'}
              </dd>
            </dl>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Results display
// ---------------------------------------------------------------------------

function TimeseriesResults({ results }: { results: TimeseriesResponse }) {
  const R = results.results;
  const nonZero = R.F.filter(v => v > 0);

  const summaryCards = [
    { label: 'Avg Chamber Pressure', value: fmtSci(avg(R.Pc.filter(v => v > 0))), sub: 'Pa' },
    { label: 'Avg Thrust',           value: fmtN(avg(nonZero), 1),               sub: 'N' },
    { label: 'Avg Isp',              value: fmtN(avg(R.Isp.filter(v => v > 0)), 1), sub: 's' },
    { label: 'Avg O/F Ratio',        value: fmtN(avg(R.MR.filter(v => v > 0)), 3),  sub: '—' },
  ];

  return (
    <div className="space-y-5">
      {/* Cd vs sqrt(ΔP) characterisation chart */}
      <CdFitChart
        fuelPairs={results.fuel_cd_pressure_pairs}
        loxPairs={results.lox_cd_pressure_pairs}
        cdFit={results.cd_fit}
      />

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {summaryCards.map(c => (
          <div key={c.label} className="p-4 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
            <p className="text-xs text-[var(--color-text-secondary)] mb-1">{c.label}</p>
            <p className="text-xl font-bold text-amber-400">{c.value}</p>
            <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">{c.sub}</p>
          </div>
        ))}
      </div>

      {/* Reynolds: hotfire vs cold-flow (same choke-based definition as spreadsheet) */}
      {results.re_similarity && (
        <ReSimilarityPanel re={results.re_similarity} />
      )}

      {/* Fit + pressure curve metadata */}
      {(results.cd_fit || results.pressure_curves_used) && (
        <div className="p-4 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
          <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider mb-3">Operating Curves Used</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            {results.cd_fit && (
              <div className="p-3 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)]">
                <p className="text-xs text-[var(--color-text-secondary)] mb-1">Cd fit model</p>
                <p className="text-[var(--color-text-primary)] font-mono text-xs">{results.cd_fit.fuel.model}</p>
                <div className="mt-2 grid grid-cols-2 gap-2 font-mono text-xs">
                  <div>
                    <p className="text-[var(--color-text-secondary)]">Fuel a</p>
                    <p className="text-[#f97316]">{results.cd_fit.fuel.a.toExponential(3)}</p>
                  </div>
                  <div>
                    <p className="text-[var(--color-text-secondary)]">Fuel b</p>
                    <p className="text-[#f97316]">{results.cd_fit.fuel.b.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-[var(--color-text-secondary)]">LOX a</p>
                    <p className="text-[#60a5fa]">{results.cd_fit.lox.a.toExponential(3)}</p>
                  </div>
                  <div>
                    <p className="text-[var(--color-text-secondary)]">LOX b</p>
                    <p className="text-[#60a5fa]">{results.cd_fit.lox.b.toFixed(4)}</p>
                  </div>
                </div>
              </div>
            )}

            {results.pressure_curves_used && (
              <>
                <div className="p-3 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)]">
                  <p className="text-xs text-[var(--color-text-secondary)] mb-1">LOX tank pressure (avg)</p>
                  <p className="text-[#60a5fa] font-mono">{fmtSci(avg(results.pressure_curves_used.P_tank_O_pa))} Pa</p>
                  <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                    min {fmtSci(Math.min(...results.pressure_curves_used.P_tank_O_pa))} / max {fmtSci(Math.max(...results.pressure_curves_used.P_tank_O_pa))}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border)]">
                  <p className="text-xs text-[var(--color-text-secondary)] mb-1">Fuel tank pressure (avg)</p>
                  <p className="text-[#f97316] font-mono">{fmtSci(avg(results.pressure_curves_used.P_tank_F_pa))} Pa</p>
                  <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                    min {fmtSci(Math.min(...results.pressure_curves_used.P_tank_F_pa))} / max {fmtSci(Math.max(...results.pressure_curves_used.P_tank_F_pa))}
                  </p>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Cd characterisation tables */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <CdTable title="Fuel Cd vs ΔP" accentColor="#f97316" pairs={results.fuel_cd_pressure_pairs} />
        <CdTable title="LOX Cd vs ΔP"  accentColor="#60a5fa" pairs={results.lox_cd_pressure_pairs}  />
      </div>

      {/* All the same plots as Time-Series Analysis tab */}
      {(() => {
        const { data, summary } = convertToTimeSeriesFormat(results);
        return <PressureCurveChart data={data} summary={summary} />;
      })()}
    </div>
  );
}

function CdTable({ title, accentColor, pairs }: { title: string; accentColor: string; pairs: [number, number][] }) {
  return (
    <div className="p-4 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
      <h3 className="text-sm font-semibold uppercase tracking-wider mb-3" style={{ color: accentColor }}>{title}</h3>
      <table style={{ borderCollapse: 'collapse', fontSize: 12, fontFamily: 'ui-monospace, monospace', width: '100%' }}>
        <thead>
          <tr>
            {['ΔP [psi]', 'Cd'].map(h => (
              <th key={h} style={{ padding: '3px 8px', textAlign: 'right', fontSize: 11, fontWeight: 600, color: 'var(--color-text-secondary)', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {pairs.map(([dp_pa, cd], i) => (
            <tr key={i}>
              <td style={{ padding: '3px 8px', textAlign: 'right', color: 'var(--color-text-primary)' }}>{fmtN(dp_pa / PSI_TO_PA, 2)}</td>
              <td style={{ padding: '3px 8px', textAlign: 'right', color: accentColor }}>{fmtN(cd, 4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
