import { useState, useCallback } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { PIDNodeData } from '../types';
import { FLUID_COLORS } from '../types';

export function TankNode({ data, selected }: NodeProps<{ data: PIDNodeData }>) {
  const nodeData = data as unknown as PIDNodeData;
  const [editing, setEditing] = useState(false);
  const [editVal, setEditVal] = useState(nodeData.label);

  const commit = useCallback(() => {
    nodeData.label = editVal;
    setEditing(false);
  }, [nodeData, editVal]);

  const stroke = selected ? '#3b82f6' : '#94a3b8';
  const fluidColor = FLUID_COLORS[nodeData.fluidType ?? 'default'];
  const isInjector = nodeData.componentType === 'INJECTOR';

  if (isInjector) {
    return (
      <div className="relative flex flex-col items-center select-none">
        <Handle type="target" position={Position.Top}    id="t" style={{ background: '#94a3b8' }} />
        <Handle type="source" position={Position.Bottom} id="b" style={{ background: '#94a3b8' }} />
        <Handle type="target" position={Position.Left}   id="l" style={{ background: '#94a3b8' }} />
        <Handle type="source" position={Position.Right}  id="r" style={{ background: '#94a3b8' }} />

        <svg width="60" height="80" viewBox="0 0 60 80">
          {/* injector header box */}
          <rect x="10" y="4" width="40" height="20" rx="2"
            fill="#1e293b" stroke={stroke} strokeWidth={selected ? 2.5 : 1.5} />
          <text x="30" y="18" textAnchor="middle" fontSize="8" fill="#e2e8f0" fontFamily="monospace">INJ</text>
          {/* nozzle cone */}
          <polygon points="10,24 50,24 38,70 22,70"
            fill="#1e293b" stroke={stroke} strokeWidth={selected ? 2.5 : 1.5} />
          {/* nozzle exit */}
          <line x1="22" y1="70" x2="38" y2="70" stroke={stroke} strokeWidth={2} />
        </svg>

        {editing ? (
          <input
            autoFocus
            value={editVal}
            onChange={e => setEditVal(e.target.value)}
            onBlur={commit}
            onKeyDown={e => e.key === 'Enter' && commit()}
            className="mt-1 text-xs text-center bg-[#1e293b] border border-blue-500 text-white rounded px-1 w-20 outline-none"
          />
        ) : (
          <span onDoubleClick={() => setEditing(true)}
            className="mt-1 text-xs text-slate-300 text-center cursor-text">
            {nodeData.label}
          </span>
        )}
      </div>
    );
  }

  return (
    <div className="relative flex flex-col items-center select-none">
      <Handle type="target" position={Position.Top}    id="t" style={{ background: '#94a3b8' }} />
      <Handle type="source" position={Position.Bottom} id="b" style={{ background: '#94a3b8' }} />
      <Handle type="target" position={Position.Left}   id="l" style={{ background: '#94a3b8', top: '50%' }} />
      <Handle type="source" position={Position.Right}  id="r" style={{ background: '#94a3b8', top: '50%' }} />

      <svg width="70" height="110" viewBox="0 0 70 110">
        {/* top ellipse cap */}
        <ellipse cx="35" cy="16" rx="28" ry="10"
          fill="#1e293b" stroke={stroke} strokeWidth={selected ? 2.5 : 1.5} />
        {/* body rectangle */}
        <rect x="7" y="16" width="56" height="76"
          fill={fluidColor + '22'} stroke={stroke} strokeWidth={selected ? 2.5 : 1.5} />
        {/* bottom ellipse cap */}
        <ellipse cx="35" cy="92" rx="28" ry="10"
          fill="#1e293b" stroke={stroke} strokeWidth={selected ? 2.5 : 1.5} />
        {/* fluid type label */}
        <text x="35" y="58" textAnchor="middle" fontSize="10" fill={fluidColor} fontFamily="monospace" fontWeight="bold">
          {nodeData.fluidType?.toUpperCase() ?? 'TANK'}
        </text>
      </svg>

      {editing ? (
        <input
          autoFocus
          value={editVal}
          onChange={e => setEditVal(e.target.value)}
          onBlur={commit}
          onKeyDown={e => e.key === 'Enter' && commit()}
          className="mt-1 text-xs text-center bg-[#1e293b] border border-blue-500 text-white rounded px-1 w-20 outline-none"
        />
      ) : (
        <span onDoubleClick={() => setEditing(true)}
          className="mt-1 text-xs text-slate-300 text-center cursor-text">
          {nodeData.label}
        </span>
      )}
    </div>
  );
}
