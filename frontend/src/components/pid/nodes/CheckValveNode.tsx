import { useState, useCallback } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { PIDNodeData } from '../types';

export function CheckValveNode({ data, selected }: NodeProps<{ data: PIDNodeData }>) {
  const nodeData = data as unknown as PIDNodeData;
  const [editing, setEditing] = useState(false);
  const [editVal, setEditVal] = useState(nodeData.label);

  const commit = useCallback(() => {
    nodeData.label = editVal;
    setEditing(false);
  }, [nodeData, editVal]);

  return (
    <div className="relative flex flex-col items-center select-none">
      <Handle type="target" position={Position.Left}   id="l" style={{ background: '#94a3b8', top: '45%' }} />
      <Handle type="source" position={Position.Right}  id="r" style={{ background: '#94a3b8', top: '45%' }} />
      <Handle type="target" position={Position.Top}    id="t" style={{ background: '#94a3b8' }} />
      <Handle type="source" position={Position.Bottom} id="b" style={{ background: '#94a3b8' }} />

      <svg width="52" height="44" viewBox="0 0 52 44">
        {/* vertical bar */}
        <line x1="26" y1="4" x2="26" y2="40"
          stroke={selected ? '#3b82f6' : '#94a3b8'} strokeWidth={1.5} />
        {/* arrow triangle */}
        <polygon
          points="26,22 46,10 46,34"
          fill="#1e293b"
          stroke={selected ? '#3b82f6' : '#94a3b8'}
          strokeWidth={selected ? 2 : 1.5}
        />
        {/* diagonal line indicating check */}
        <line x1="10" y1="4" x2="26" y2="22"
          stroke={selected ? '#3b82f6' : '#94a3b8'} strokeWidth={1.5} />
        <line x1="10" y1="40" x2="26" y2="22"
          stroke={selected ? '#3b82f6' : '#94a3b8'} strokeWidth={1.5} />
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
