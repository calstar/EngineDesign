import { useCallback, useRef, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  type Connection,
  type Edge,
  type Node,
  type ReactFlowInstance,
  BackgroundVariant,
  Panel,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { nodeTypes } from './nodes';
import type { PIDNodeData, ComponentType, FluidType } from './types';
import { FLUID_COLORS, COMPONENT_DEFS } from './types';

let idCounter = 1;
const newId = () => `node_${idCounter++}`;

function defaultLabel(type: ComponentType): string {
  const def = COMPONENT_DEFS.find(d => d.type === type);
  return def ? def.label : type;
}

interface EdgeMenuState {
  edgeId: string;
  x: number;
  y: number;
}

export function PIDCanvas({
  onInstanceReady,
}: {
  onInstanceReady?: (instance: ReactFlowInstance) => void;
}) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [edgeMenu, setEdgeMenu] = useState<EdgeMenuState | null>(null);
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [rfInstance, setRfInstance] = useState<ReactFlowInstance | null>(null);

  const onInit = useCallback((instance: ReactFlowInstance) => {
    setRfInstance(instance);
    onInstanceReady?.(instance);
  }, [onInstanceReady]);

  const onConnect = useCallback((params: Connection) => {
    setEdges(eds =>
      addEdge(
        {
          ...params,
          type: 'smoothstep',
          style: { stroke: FLUID_COLORS.default, strokeWidth: 2 },
          data: { fluidType: 'default' as FluidType },
        },
        eds,
      ),
    );
  }, [setEdges]);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const type = e.dataTransfer.getData('application/pid-type') as ComponentType;
      if (!type || !rfInstance || !reactFlowWrapper.current) return;

      const bounds = reactFlowWrapper.current.getBoundingClientRect();
      const position = rfInstance.screenToFlowPosition({
        x: e.clientX - bounds.left,
        y: e.clientY - bounds.top,
      });

      const nodeData: PIDNodeData = {
        componentType: type,
        label: defaultLabel(type),
        fluidType: 'default',
      };

      const newNode: Node = {
        id: newId(),
        type,
        position,
        data: nodeData as unknown as Record<string, unknown>,
      };

      setNodes(nds => [...nds, newNode]);
    },
    [rfInstance, setNodes],
  );

  const onEdgeClick = useCallback(
    (_e: React.MouseEvent, edge: Edge) => {
      _e.stopPropagation();
      setEdgeMenu({ edgeId: edge.id, x: _e.clientX, y: _e.clientY });
    },
    [],
  );

  const setEdgeFluid = useCallback(
    (edgeId: string, fluid: FluidType) => {
      setEdges(eds =>
        eds.map(e =>
          e.id === edgeId
            ? { ...e, style: { ...e.style, stroke: FLUID_COLORS[fluid], strokeWidth: 2 }, data: { ...e.data, fluidType: fluid } }
            : e,
        ),
      );
      setEdgeMenu(null);
    },
    [setEdges],
  );

  // Export/Import helpers exposed via ref-like pattern on the instance
  const getSnapshot = useCallback(() => ({ nodes, edges }), [nodes, edges]);
  const loadSnapshot = useCallback(
    (data: { nodes: Node[]; edges: Edge[] }) => {
      setNodes(data.nodes ?? []);
      setEdges(data.edges ?? []);
    },
    [setNodes, setEdges],
  );

  // Expose helpers on the canvas instance holder object
  (PIDCanvas as unknown as { _getSnapshot: typeof getSnapshot })._getSnapshot = getSnapshot;
  (PIDCanvas as unknown as { _loadSnapshot: typeof loadSnapshot })._loadSnapshot = loadSnapshot;

  return (
    <div ref={reactFlowWrapper} className="flex-1 h-full relative" onClick={() => setEdgeMenu(null)}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onInit={onInit}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onEdgeClick={onEdgeClick}
        nodeTypes={nodeTypes}
        deleteKeyCode="Delete"
        fitView
        colorMode="dark"
        defaultEdgeOptions={{
          type: 'smoothstep',
          style: { stroke: FLUID_COLORS.default, strokeWidth: 2 },
        }}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#1e293b" />
        <Controls />
        <MiniMap
          nodeColor={() => '#334155'}
          maskColor="rgba(0,0,0,0.6)"
          style={{ background: '#0f172a', border: '1px solid #1e293b' }}
        />
        <Panel position="bottom-center">
          <span className="text-[10px] text-slate-600 select-none">
            Drag components from sidebar · Connect handles · Click edge to set fluid · Delete to remove
          </span>
        </Panel>
      </ReactFlow>

      {/* Edge fluid-type context menu */}
      {edgeMenu && (
        <div
          style={{ position: 'fixed', left: edgeMenu.x, top: edgeMenu.y, zIndex: 9999 }}
          className="bg-[#1e293b] border border-[#334155] rounded-lg shadow-xl py-1 min-w-[140px]"
          onClick={e => e.stopPropagation()}
        >
          <p className="text-[10px] text-slate-500 px-3 py-1 uppercase tracking-wider">Fluid type</p>
          {(['fuel', 'lox', 'pressurant', 'default'] as FluidType[]).map(f => (
            <button
              key={f}
              onClick={() => setEdgeFluid(edgeMenu.edgeId, f)}
              className="flex items-center gap-2 w-full px-3 py-1.5 text-sm text-slate-300 hover:bg-[#0f172a] transition-colors"
            >
              <span
                className="inline-block w-3 h-3 rounded-full"
                style={{ background: FLUID_COLORS[f] }}
              />
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
