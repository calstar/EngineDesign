import { useState, useCallback } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { PIDNodeData } from '../types';

export function QDNode({ data, selected }: NodeProps<{ data: PIDNodeData }>) {
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
      <Handle type="target" position={Position.Top}    id="t" style={{ background: '#94a3b8' }} />
      <Handle type="source" position={Position.Bottom} id="b" style={{ background: '#94a3b8' }} />
      <Handle type="target" position={Position.Left}   id="l" style={{ background: '#94a3b8' }} />
      <Handle type="source" position={Position.Right}  id="r" style={{ background: '#94a3b8' }} />

      <svg width="44" height="44" viewBox="0 0 44 44">
        <circle cx="22" cy="22" r="18"
          fill="#1e293b" stroke={stroke} strokeWidth={selected ? 2.5 : 1.5} />
        {/* X mark */}
        <line x1="10" y1="10" x2="34" y2="34" stroke={stroke} strokeWidth={2} />
        <line x1="34" y1="10" x2="10" y2="34" stroke={stroke} strokeWidth={2} />
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
