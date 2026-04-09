import { useState, useCallback } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { PIDNodeData } from '../types';

export function SensorNode({ data, selected }: NodeProps<{ data: PIDNodeData }>) {
  const { componentType, label } = data as unknown as PIDNodeData;
  const [editing, setEditing] = useState(false);
  const [editVal, setEditVal] = useState(label);

  const commit = useCallback(() => {
    (data as unknown as PIDNodeData).label = editVal;
    setEditing(false);
  }, [data, editVal]);

  const shortCode = componentType;

  return (
    <div className="relative flex flex-col items-center select-none">
      <Handle type="target" position={Position.Top}    id="t" style={{ background: '#94a3b8' }} />
      <Handle type="source" position={Position.Bottom} id="b" style={{ background: '#94a3b8' }} />
      <Handle type="target" position={Position.Left}   id="l" style={{ background: '#94a3b8' }} />
      <Handle type="source" position={Position.Right}  id="r" style={{ background: '#94a3b8' }} />

      <svg width="52" height="52" viewBox="0 0 52 52">
        <circle
          cx="26" cy="26" r="22"
          fill="#1e293b"
          stroke={selected ? '#3b82f6' : '#94a3b8'}
          strokeWidth={selected ? 2.5 : 1.5}
        />
        <text x="26" y="30" textAnchor="middle" fontSize="10" fill="#e2e8f0" fontFamily="monospace" fontWeight="bold">
          {shortCode}
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
        <span
          onDoubleClick={() => setEditing(true)}
          className="mt-1 text-xs text-slate-300 text-center cursor-text select-none"
        >
          {(data as unknown as PIDNodeData).label}
        </span>
      )}
    </div>
  );
}
