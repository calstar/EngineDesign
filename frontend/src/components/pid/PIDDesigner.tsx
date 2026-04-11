import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ReactFlowProvider,
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Panel,
  addEdge,
  useNodesState,
  useEdgesState,
  BackgroundVariant,
  SelectionMode,
  type ReactFlowInstance,
  type Connection,
  type Node,
  type Edge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { ComponentPalette } from './ComponentPalette';
import { PIDToolbar } from './PIDToolbar';
import { nodeTypes } from './nodes';
import { BranchableEdge } from './BranchableEdge';
import { FLUID_COLORS, COMPONENT_DEFS } from './types';
import type { PIDNodeData, ComponentType, FluidType } from './types';

export type InteractionMode = 'pan' | 'select';

let _idCounter = 1;
const genId = () => `node_${_idCounter++}`;

function defaultLabel(type: ComponentType) {
  return COMPONENT_DEFS.find(d => d.type === type)?.label ?? type;
}

// ── Undo / redo history ─────────────────────────────────────────────────────
const MAX_HISTORY = 100;

type Snapshot = { nodes: Node[]; edges: Edge[] };

function useHistory(
  nodes: Node[],
  edges: Edge[],
  setNodes: (nds: Node[]) => void,
  setEdges: (eds: Edge[]) => void,
) {
  const history   = useRef<Snapshot[]>([{ nodes: [], edges: [] }]);
  const index     = useRef(0);
  const restoring = useRef(false);
  const timer     = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Debounced push — batches rapid-fire changes (dragging, etc.)
  useEffect(() => {
    if (restoring.current) { restoring.current = false; return; }
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      const snap: Snapshot = { nodes: structuredClone(nodes), edges: structuredClone(edges) };
      const prev = history.current[index.current];
      if (JSON.stringify(prev) === JSON.stringify(snap)) return;
      history.current = history.current.slice(0, index.current + 1);
      history.current.push(snap);
      if (history.current.length > MAX_HISTORY) history.current.shift();
      index.current = history.current.length - 1;
    }, 300);
    return () => { if (timer.current) clearTimeout(timer.current); };
  }, [nodes, edges]);

  const undo = useCallback(() => {
    if (index.current <= 0) return;
    index.current -= 1;
    restoring.current = true;
    const snap = history.current[index.current];
    setNodes(structuredClone(snap.nodes));
    setEdges(structuredClone(snap.edges));
  }, [setNodes, setEdges]);

  const redo = useCallback(() => {
    if (index.current >= history.current.length - 1) return;
    index.current += 1;
    restoring.current = true;
    const snap = history.current[index.current];
    setNodes(structuredClone(snap.nodes));
    setEdges(structuredClone(snap.edges));
  }, [setNodes, setEdges]);

  const canUndo = index.current > 0;
  const canRedo = index.current < history.current.length - 1;

  return { undo, redo, canUndo, canRedo };
}

// ── Inner canvas — owns all React Flow state ─────────────────────────────────
interface CanvasProps {
  onInstance: (inst: ReactFlowInstance) => void;
  getRef:   React.MutableRefObject<() => { nodes: Node[]; edges: Edge[] }>;
  loadRef:  React.MutableRefObject<(d: { nodes: Node[]; edges: Edge[] }) => void>;
  clearRef: React.MutableRefObject<() => void>;
  undoRef:  React.MutableRefObject<() => void>;
  redoRef:  React.MutableRefObject<() => void>;
  mode:     InteractionMode;
}

function PIDCanvas({ onInstance, getRef, loadRef, clearRef, undoRef, redoRef, mode }: CanvasProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [edgeMenu, setEdgeMenu] = useState<{ id: string; x: number; y: number } | null>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [rfInst, setRfInst] = useState<ReactFlowInstance | null>(null);

  const { undo, redo } = useHistory(nodes, edges, setNodes, setEdges);

  // Expose snapshot helpers via refs so the toolbar can call them
  getRef.current   = useCallback(() => ({ nodes, edges }), [nodes, edges]);
  loadRef.current  = useCallback((d) => { setNodes(d.nodes); setEdges(d.edges); }, [setNodes, setEdges]);
  clearRef.current = useCallback(() => { setNodes([]); setEdges([]); }, [setNodes, setEdges]);
  undoRef.current  = undo;
  redoRef.current  = redo;

  const onInit = useCallback((inst: ReactFlowInstance) => {
    setRfInst(inst);
    onInstance(inst);
  }, [onInstance]);

  const edgeTypes = useMemo(() => ({ smoothstep: BranchableEdge, default: BranchableEdge }), []);

  const onConnect = useCallback((params: Connection) => {
    setEdges(eds => addEdge({
      ...params,
      type: 'smoothstep',
      style: { stroke: FLUID_COLORS.default, strokeWidth: 2 },
      data: { fluidType: 'default' as FluidType },
    }, eds));
  }, [setEdges]);

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const type = e.dataTransfer.getData('application/pid-type') as ComponentType;
    if (!type || !rfInst || !wrapperRef.current) return;
    const bounds = wrapperRef.current.getBoundingClientRect();
    const position = rfInst.screenToFlowPosition({
      x: e.clientX - bounds.left,
      y: e.clientY - bounds.top,
    });
    const nodeData = type === 'TEXT'
      ? { text: 'Text' }
      : type === 'JUNCTION'
      ? {}
      : { componentType: type, label: defaultLabel(type), fluidType: 'default' } as PIDNodeData;
    setNodes(nds => [...nds, {
      id: genId(),
      type,
      position,
      data: nodeData as unknown as Record<string, unknown>,
    }]);
  }, [rfInst, setNodes]);

  // Right-click a pipe → change fluid colour
  const onEdgeContextMenu = useCallback((e: React.MouseEvent, edge: Edge) => {
    e.preventDefault();
    e.stopPropagation();
    setEdgeMenu({ id: edge.id, x: e.clientX, y: e.clientY });
  }, []);

  const setEdgeFluid = useCallback((edgeId: string, fluid: FluidType) => {
    setEdges(eds => eds.map(e =>
      e.id === edgeId
        ? { ...e, style: { ...e.style, stroke: FLUID_COLORS[fluid], strokeWidth: 2 }, data: { ...e.data, fluidType: fluid } }
        : e,
    ));
    setEdgeMenu(null);
  }, [setEdges]);

  return (
    <div ref={wrapperRef} className="flex-1 h-full relative" onClick={() => setEdgeMenu(null)}>
      <ReactFlow
        nodes={nodes} edges={edges}
        onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
        onConnect={onConnect} onInit={onInit}
        onDrop={onDrop} onDragOver={onDragOver}
        onEdgeContextMenu={onEdgeContextMenu}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        selectionOnDrag={mode === 'select'}
        panOnDrag={mode !== 'select'}
        selectionMode={SelectionMode.Partial}
        multiSelectionKeyCode="Meta"
        deleteKeyCode="Delete"
        snapToGrid
        snapGrid={[20, 20]}
        fitView
        colorMode="dark"
        defaultEdgeOptions={{ type: 'smoothstep', style: { stroke: FLUID_COLORS.default, strokeWidth: 2 } }}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#1e293b" />
        <Controls />
        <MiniMap nodeColor={() => '#334155'} maskColor="rgba(0,0,0,0.6)"
          style={{ background: '#0f172a', border: '1px solid #1e293b' }} />
        <Panel position="bottom-center">
          <span className="text-[10px] text-slate-600 select-none">
            Drag from sidebar · Connect handles · V=Pan  B=Box select · Cmd/Ctrl+click to multi-select · Right-click pipe for fluid · Delete removes selection
          </span>
        </Panel>
      </ReactFlow>

      {edgeMenu && (
        <div
          style={{ position: 'fixed', left: edgeMenu.x, top: edgeMenu.y, zIndex: 9999 }}
          className="bg-[#1e293b] border border-[#334155] rounded-lg shadow-xl py-1 min-w-[140px]"
          onClick={e => e.stopPropagation()}
        >
          <p className="text-[10px] text-slate-500 px-3 py-1 uppercase tracking-wider">Fluid type</p>
          {(['fuel', 'lox', 'pressurant', 'default'] as FluidType[]).map(f => (
            <button key={f} onClick={() => setEdgeFluid(edgeMenu.id, f)}
              className="flex items-center gap-2 w-full px-3 py-1.5 text-sm text-slate-300 hover:bg-[#0f172a] transition-colors">
              <span className="inline-block w-3 h-3 rounded-full" style={{ background: FLUID_COLORS[f] }} />
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      )}

    </div>
  );
}

// ── Top-level designer — wires toolbar ↔ canvas ───────────────────────────────
export function PIDDesigner() {
  const [rfInstance, setRfInstance] = useState<ReactFlowInstance | null>(null);
  const [mode, setMode] = useState<InteractionMode>('pan');

  const getRef   = useRef<() => { nodes: Node[]; edges: Edge[] }>(() => ({ nodes: [], edges: [] }));
  const loadRef  = useRef<(d: { nodes: Node[]; edges: Edge[] }) => void>(() => {});
  const clearRef = useRef<() => void>(() => {});
  const undoRef  = useRef<() => void>(() => {});
  const redoRef  = useRef<() => void>(() => {});

  const handleInstance = useCallback((inst: ReactFlowInstance) => {
    setRfInstance(inst);
  }, []);

  return (
    <div className="flex flex-col h-[calc(100vh-140px)] min-h-[600px] rounded-xl overflow-hidden border border-[#1e293b]">
      <PIDToolbar
        rfInstance={rfInstance}
        getSnapshot={() => getRef.current()}
        loadSnapshot={d => loadRef.current(d)}
        onClear={() => clearRef.current()}
        onUndo={() => undoRef.current()}
        onRedo={() => redoRef.current()}
        mode={mode}
        onModeChange={setMode}
      />
      <div className="flex flex-1 overflow-hidden">
        <ComponentPalette />
        <ReactFlowProvider>
          <PIDCanvas
            onInstance={handleInstance}
            getRef={getRef}
            loadRef={loadRef}
            clearRef={clearRef}
            undoRef={undoRef}
            redoRef={redoRef}
            mode={mode}
          />
        </ReactFlowProvider>
      </div>
    </div>
  );
}
