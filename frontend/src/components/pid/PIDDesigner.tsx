import { useCallback, useRef, useState } from 'react';
import { ReactFlowProvider, type ReactFlowInstance, type Node, type Edge } from '@xyflow/react';
import { ComponentPalette } from './ComponentPalette';
import { PIDCanvas } from './PIDCanvas';
import { PIDToolbar } from './PIDToolbar';

export function PIDDesigner() {
  const rfInstanceRef = useRef<ReactFlowInstance | null>(null);
  const [, forceUpdate] = useState(0);

  // Snapshot helpers threaded from canvas
  const snapshotRef = useRef<{
    get: () => { nodes: Node[]; edges: Edge[] };
    load: (d: { nodes: Node[]; edges: Edge[] }) => void;
    clear: () => void;
  } | null>(null);

  const onClear = useCallback(() => {
    snapshotRef.current?.clear();
  }, []);

  return (
    <div className="flex flex-col h-[calc(100vh-140px)] min-h-[600px] rounded-xl overflow-hidden border border-[#1e293b]">
      <PIDToolbar
        rfInstance={rfInstanceRef.current}
        getSnapshot={() => snapshotRef.current?.get() ?? { nodes: [], edges: [] }}
        loadSnapshot={d => snapshotRef.current?.load(d)}
        onClear={onClear}
      />
      <div className="flex flex-1 overflow-hidden">
        <ComponentPalette />
        <ReactFlowProvider>
          <CanvasWithCallbacks
            onInstance={inst => {
              rfInstanceRef.current = inst;
              forceUpdate(n => n + 1); // re-render toolbar with valid rfInstance
            }}
            onSnapshotReady={helpers => {
              snapshotRef.current = helpers;
            }}
          />
        </ReactFlowProvider>
      </div>
    </div>
  );
}

// Inner component that owns the canvas and wires up snapshot helpers
function CanvasWithCallbacks({
  onInstance,
  onSnapshotReady,
}: {
  onInstance: (inst: ReactFlowInstance) => void;
  onSnapshotReady: (h: {
    get: () => { nodes: Node[]; edges: Edge[] };
    load: (d: { nodes: Node[]; edges: Edge[] }) => void;
    clear: () => void;
  }) => void;
}) {
  const getRef = useRef<() => { nodes: Node[]; edges: Edge[] }>(() => ({ nodes: [], edges: [] }));
  const loadRef = useRef<(d: { nodes: Node[]; edges: Edge[] }) => void>(() => {});
  const clearRef = useRef<() => void>(() => {});

  // Called once on canvas mount
  const handleInstance = useCallback(
    (inst: ReactFlowInstance) => {
      onInstance(inst);
      onSnapshotReady({
        get: () => getRef.current(),
        load: d => loadRef.current(d),
        clear: () => clearRef.current(),
      });
    },
    [onInstance, onSnapshotReady],
  );

  return (
    <PIDCanvasConnected
      onInstanceReady={handleInstance}
      getRef={getRef}
      loadRef={loadRef}
      clearRef={clearRef}
    />
  );
}

// Thin wrapper that injects snapshot refs into PIDCanvas
import { useNodesState, useEdgesState, type Connection, addEdge, ReactFlow, Background, Controls, MiniMap, BackgroundVariant, Panel } from '@xyflow/react';
import { nodeTypes } from './nodes';
import type { PIDNodeData, ComponentType, FluidType } from './types';
import { FLUID_COLORS, COMPONENT_DEFS } from './types';
import { useRef as useReactRef } from 'react';

let _idCounter = 1;
const genId = () => `node_${_idCounter++}`;

function defaultLabel(type: ComponentType) {
  return COMPONENT_DEFS.find(d => d.type === type)?.label ?? type;
}

function PIDCanvasConnected({
  onInstanceReady,
  getRef,
  loadRef,
  clearRef,
}: {
  onInstanceReady: (inst: ReactFlowInstance) => void;
  getRef: React.MutableRefObject<() => { nodes: Node[]; edges: Edge[] }>;
  loadRef: React.MutableRefObject<(d: { nodes: Node[]; edges: Edge[] }) => void>;
  clearRef: React.MutableRefObject<() => void>;
}) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [edgeMenu, setEdgeMenu] = useState<{ id: string; x: number; y: number } | null>(null);
  const wrapperRef = useReactRef<HTMLDivElement>(null);
  const [rfInst, setRfInst] = useState<ReactFlowInstance | null>(null);

  // Wire snapshot helpers into refs so parent can call them
  getRef.current = useCallback(() => ({ nodes, edges }), [nodes, edges]);
  loadRef.current = useCallback((d) => { setNodes(d.nodes); setEdges(d.edges); }, [setNodes, setEdges]);
  clearRef.current = useCallback(() => { setNodes([]); setEdges([]); }, [setNodes, setEdges]);

  const onInit = useCallback((inst: ReactFlowInstance) => {
    setRfInst(inst);
    onInstanceReady(inst);
  }, [onInstanceReady]);

  const onConnect = useCallback((params: Connection) => {
    setEdges(eds => addEdge({
      ...params,
      type: 'smoothstep',
      style: { stroke: FLUID_COLORS.default, strokeWidth: 2 },
      data: { fluidType: 'default' as FluidType },
    }, eds));
  }, [setEdges]);

  const onDragOver = (e: React.DragEvent) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const type = e.dataTransfer.getData('application/pid-type') as ComponentType;
    if (!type || !rfInst || !wrapperRef.current) return;
    const bounds = wrapperRef.current.getBoundingClientRect();
    const position = rfInst.screenToFlowPosition({ x: e.clientX - bounds.left, y: e.clientY - bounds.top });
    const nodeData: PIDNodeData = { componentType: type, label: defaultLabel(type), fluidType: 'default' };
    setNodes(nds => [...nds, { id: genId(), type, position, data: nodeData as unknown as Record<string, unknown> }]);
  }, [rfInst, setNodes]);

  const onEdgeClick = useCallback((_e: React.MouseEvent, edge: Edge) => {
    _e.stopPropagation();
    setEdgeMenu({ id: edge.id, x: _e.clientX, y: _e.clientY });
  }, []);

  const setEdgeFluid = useCallback((edgeId: string, fluid: FluidType) => {
    setEdges(eds => eds.map(e => e.id === edgeId
      ? { ...e, style: { ...e.style, stroke: FLUID_COLORS[fluid], strokeWidth: 2 }, data: { ...e.data, fluidType: fluid } }
      : e));
    setEdgeMenu(null);
  }, [setEdges]);

  return (
    <div ref={wrapperRef} className="flex-1 h-full relative" onClick={() => setEdgeMenu(null)}>
      <ReactFlow
        nodes={nodes} edges={edges}
        onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
        onConnect={onConnect} onInit={onInit}
        onDrop={onDrop} onDragOver={onDragOver}
        onEdgeClick={onEdgeClick}
        nodeTypes={nodeTypes}
        deleteKeyCode="Delete"
        fitView colorMode="dark"
        defaultEdgeOptions={{ type: 'smoothstep', style: { stroke: FLUID_COLORS.default, strokeWidth: 2 } }}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#1e293b" />
        <Controls />
        <MiniMap nodeColor={() => '#334155'} maskColor="rgba(0,0,0,0.6)"
          style={{ background: '#0f172a', border: '1px solid #1e293b' }} />
        <Panel position="bottom-center">
          <span className="text-[10px] text-slate-600 select-none">
            Drag from sidebar · Connect handles · Click edge to change fluid · Delete key removes selection
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
