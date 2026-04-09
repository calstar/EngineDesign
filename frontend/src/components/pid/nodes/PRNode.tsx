import { useState, useCallback } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { PIDNodeData } from '../types';

export function PRNode({ data, selected }: NodeProps<{ data: PIDNodeData }>) {
  const nodeData = data as unknown as PIDNodeData;
  const [editing, setEditing] = useState(false);
  const [editVal, setEditVal] = useState(nodeData.label);

  const commit = useCallback(() => {
    nodeData.label = editVal;
    setEditing(false);
  }, [nodeData, editVal]);

  const stroke = selected ? '#3b82f6' : '#94a3b8';

  return (
    <div className="relative flex flex-col items-center select-none">
      <Handle type="target" position={Position.Left}   id="l" style={{ background: '#94a3b8', top: '45%' }} />
      <Handle type="source" position={Position.Right}  id="r" style={{ background: '#94a3b8', top: '45%' }} />
      <Handle type="target" position={Position.Top}    id="t" style={{ background: '#94a3b8' }} />
      <Handle type="source" position={Position.Bottom} id="b" style={{ background: '#94a3b8' }} />

      <svg width="52" height="52" viewBox="0 0 52 52">
        {/* outer square */}
        <rect x="6" y="6" width="40" height="40" rx="3"
          fill="#1e293b" stroke={stroke} strokeWidth={selected ? 2.5 : 1.5} />
        {/* diagonal arrow representing regulation */}
        <line x1="14" y1="38" x2="38" y2="14" stroke={stroke} strokeWidth={1.5} />
        <polygon points="38,14 30,16 36,22" fill={stroke} />
        {/* P label */}
        <text x="14" y="24" fontSize="9" fill="#e2e8f0" fontFamily="monospace">PR</text>
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
