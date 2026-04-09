import type { ReactFlowInstance, Node, Edge } from '@xyflow/react';

interface PIDToolbarProps {
  rfInstance: ReactFlowInstance | null;
  getSnapshot: () => { nodes: Node[]; edges: Edge[] };
  loadSnapshot: (data: { nodes: Node[]; edges: Edge[] }) => void;
  onClear: () => void;
}

export function PIDToolbar({ rfInstance, getSnapshot, loadSnapshot, onClear }: PIDToolbarProps) {
  const fitView = () => rfInstance?.fitView({ padding: 0.1 });

  const exportJSON = () => {
    const snap = getSnapshot();
    const blob = new Blob([JSON.stringify(snap, null, 2)], { type: 'application/json' });
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
        const text = await file.text();
        const data = JSON.parse(text) as { nodes: Node[]; edges: Edge[] };
        if (Array.isArray(data.nodes) && Array.isArray(data.edges)) {
          loadSnapshot(data);
          setTimeout(() => rfInstance?.fitView({ padding: 0.1 }), 100);
        }
      } catch {
        alert('Invalid P&ID JSON file.');
      }
    };
    input.click();
  };

  const btnBase =
    'flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded transition-colors';
  const btnDefault = `${btnBase} bg-[#1e293b] text-slate-300 hover:bg-[#334155] border border-[#334155]`;
  const btnDanger  = `${btnBase} bg-red-900/30 text-red-400 hover:bg-red-900/50 border border-red-800/50`;

  return (
    <div className="flex items-center gap-2 px-4 py-2 border-b border-[#1e293b] bg-[#0f172a]">
      <span className="text-sm font-semibold text-slate-200 mr-2">P&amp;ID Designer</span>

      <button onClick={fitView} className={btnDefault} title="Fit diagram to view">
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
        </svg>
        Fit View
      </button>

      <button onClick={exportJSON} className={btnDefault} title="Export diagram as JSON">
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
        </svg>
        Export JSON
      </button>

      <button onClick={importJSON} className={btnDefault} title="Import diagram from JSON">
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l4-4m0 0l4 4m-4-4v12" />
        </svg>
        Import JSON
      </button>

      <div className="ml-auto" />

      <button onClick={onClear} className={btnDanger} title="Clear canvas">
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
        </svg>
        Clear
      </button>
    </div>
  );
}
