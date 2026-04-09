import { useState, useCallback } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { PIDNodeData } from '../types';

function BowtieSVG({ selected, actuatorLabel }: { selected: boolean; actuatorLabel: string }) {
  return (
    <svg width="60" height="56" viewBox="0 0 60 56">
      {/* bowtie body */}
      <polygon
        points="8,8 52,44 52,8 8,44"
        fill="#1e293b"
        stroke={selected ? '#3b82f6' : '#94a3b8'}
        strokeWidth={selected ? 2.5 : 1.5}
      />
      {/* actuator box above center */}
      <rect x="22" y="0" width="16" height="10" rx="2"
        fill="#1e293b" stroke={selected ? '#3b82f6' : '#94a3b8'} strokeWidth={1.2} />
      <text x="30" y="9" textAnchor="middle" fontSize="7" fill="#cbd5e1" fontFamily="monospace">
        {actuatorLabel}
      </text>
      {/* stem */}
      <line x1="30" y1="10" x2="30" y2="18" stroke={selected ? '#3b82f6' : '#94a3b8'} strokeWidth={1.5} />
    </svg>
  );
}

function MidpointLine({ selected }: { selected: boolean }) {
  return (
    <svg width="60" height="56" viewBox="0 0 60 56">
      {/* bowtie */}
      <polygon
        points="8,8 52,44 52,8 8,44"
        fill="#1e293b"
        stroke={selected ? '#3b82f6' : '#94a3b8'}
        strokeWidth={selected ? 2.5 : 1.5}
      />
      {/* diamond / midpoint indicator */}
      <rect x="24" y="0" width="12" height="12" rx="2" transform="rotate(45 30 6)"
        fill="#1e293b" stroke={selected ? '#3b82f6' : '#94a3b8'} strokeWidth={1.2} />
    </svg>
  );
}

export function ValveNode({ data, selected }: NodeProps<{ data: PIDNodeData }>) {
  const nodeData = data as unknown as PIDNodeData;
  const [editing, setEditing] = useState(false);
  const [editVal, setEditVal] = useState(nodeData.label);

  const commit = useCallback(() => {
    nodeData.label = editVal;
    setEditing(false);
  }, [nodeData, editVal]);

  const actuator = nodeData.componentType === 'SOL' ? 'S' : nodeData.componentType === 'ROT' ? 'P' : 'P';
  const isMAN = nodeData.componentType === 'MAN';

  return (
    <div className="relative flex flex-col items-center select-none">
      <Handle type="target" position={Position.Left}   id="l" style={{ background: '#94a3b8', top: '55%' }} />
      <Handle type="source" position={Position.Right}  id="r" style={{ background: '#94a3b8', top: '55%' }} />
      <Handle type="target" position={Position.Top}    id="t" style={{ background: '#94a3b8' }} />
      <Handle type="source" position={Position.Bottom} id="b" style={{ background: '#94a3b8' }} />

      {isMAN ? <MidpointLine selected={!!selected} /> : <BowtieSVG selected={!!selected} actuatorLabel={actuator} />}

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
        <span
          onDoubleClick={() => setEditing(true)}
          className="mt-1 text-xs text-slate-300 text-center cursor-text"
        >
          {nodeData.label}
        </span>
      )}
    </div>
  );
}
