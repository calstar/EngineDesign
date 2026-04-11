import { useEffect } from 'react';
import type { ReactFlowInstance, Node, Edge } from '@xyflow/react';
import type { InteractionMode } from './PIDDesigner';

interface PIDToolbarProps {
  rfInstance: ReactFlowInstance | null;
  getSnapshot: () => { nodes: Node[]; edges: Edge[] };
  loadSnapshot: (data: { nodes: Node[]; edges: Edge[] }) => void;
  onClear: () => void;
  onUndo: () => void;
  onRedo: () => void;
  mode: InteractionMode;
  onModeChange: (mode: InteractionMode) => void;
}

export function PIDToolbar({ rfInstance, getSnapshot, loadSnapshot, onClear, onUndo, onRedo, mode, onModeChange }: PIDToolbarProps) {
  const fitView = () => rfInstance?.fitView({ padding: 0.1 });

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      // Undo / redo
      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) onRedo(); else onUndo();
        return;
      }
      // Mode shortcuts: V = pan, B = box select
      if (!mod && !e.shiftKey && !e.altKey) {
        if (e.key.toLowerCase() === 'v') onModeChange('pan');
        if (e.key.toLowerCase() === 'b') onModeChange('select');
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onUndo, onRedo, onModeChange]);

  const exportJSON = () => {
    const blob = new Blob([JSON.stringify(getSnapshot(), null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'pid_diagram.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const importJSON = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      try {
        const data = JSON.parse(await file.text()) as { nodes: Node[]; edges: Edge[] };
        if (Array.isArray(data.nodes) && Array.isArray(data.edges)) {
          loadSnapshot(data);
          setTimeout(() => rfInstance?.fitView({ padding: 0.1 }), 100);
        }
      } catch { alert('Invalid P&ID JSON file.'); }
    };
    input.click();
  };

  const btn = 'flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded transition-colors border';
  const def = `${btn} bg-[#1e293b] text-slate-300 hover:bg-[#334155] border-[#334155]`;
  const danger = `${btn} bg-red-900/30 text-red-400 hover:bg-red-900/50 border-red-800/50`;
  const modeBtn = (m: InteractionMode) =>
    `${btn} ${mode === m
      ? 'bg-blue-600/30 text-blue-300 border-blue-500/50'
      : 'bg-[#1e293b] text-slate-300 hover:bg-[#334155] border-[#334155]'}`;

  return (
    <div className="flex items-center gap-2 px-4 py-2 border-b border-[#1e293b] bg-[#0f172a]">
      <span className="text-sm font-semibold text-slate-200 mr-2">P&amp;ID Designer</span>

      {/* Interaction mode toggle */}
      <button onClick={() => onModeChange('pan')} className={modeBtn('pan')} title="Pan (V)">
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
        </svg>
        Pan
      </button>
      <button onClick={() => onModeChange('select')} className={modeBtn('select')} title="Box Select (B)">
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M4 5a1 1 0 011-1h4a1 1 0 010 2H6v3a1 1 0 01-2 0V5zm16 0a1 1 0 00-1-1h-4a1 1 0 000 2h3v3a1 1 0 002 0V5zM4 19a1 1 0 001 1h4a1 1 0 000-2H6v-3a1 1 0 00-2 0v4zm16 0a1 1 0 01-1 1h-4a1 1 0 010-2h3v-3a1 1 0 012 0v4z" />
        </svg>
        Select
      </button>
      <div className="w-px h-5 bg-[#334155]" />

      <button onClick={onUndo} className={def} title="Undo (Ctrl+Z)">
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M3 10h10a5 5 0 015 5v2M3 10l4-4M3 10l4 4" />
        </svg>
        Undo
      </button>
      <button onClick={onRedo} className={def} title="Redo (Ctrl+Shift+Z)">
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M21 10H11a5 5 0 00-5 5v2M21 10l-4-4M21 10l-4 4" />
        </svg>
        Redo
      </button>
      <div className="w-px h-5 bg-[#334155]" />
      <button onClick={fitView} className={def}>
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
        </svg>
        Fit View
      </button>
      <button onClick={exportJSON} className={def}>
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
        </svg>
        Export JSON
      </button>
      <button onClick={importJSON} className={def}>
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l4-4m0 0l4 4m-4-4v12" />
        </svg>
        Import JSON
      </button>
      <div className="ml-auto" />
      <button onClick={onClear} className={danger}>
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
        </svg>
        Clear
      </button>
    </div>
  );
}
