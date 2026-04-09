import { useState, useCallback } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { PIDNodeData } from '../types';

export function RVNode({ data, selected }: NodeProps<{ data: PIDNodeData }>) {
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
      <Handle type="target" position={Position.Left}   id="l" style={{ background: '#94a3b8', top: '50%' }} />
      <Handle type="source" position={Position.Right}  id="r" style={{ background: '#94a3b8', top: '50%' }} />
      <Handle type="target" position={Position.Top}    id="t" style={{ background: '#94a3b8' }} />
      <Handle type="source" position={Position.Bottom} id="b" style={{ background: '#94a3b8' }} />

      <svg width="60" height="52" viewBox="0 0 60 52">
        {/* bowtie base */}
        <polygon points="6,14 54,38 54,14 6,38"
          fill="#1e293b" stroke={stroke} strokeWidth={selected ? 2.5 : 1.5} />
        {/* zigzag spring on top indicating relief */}
        <polyline
          points="20,14 24,6 28,14 32,6 36,14 40,6"
          fill="none" stroke={stroke} strokeWidth={1.5} strokeLinecap="round"
        />
        {/* vertical stem from zigzag to bowtie */}
        <line x1="30" y1="6" x2="30" y2="14" stroke={stroke} strokeWidth={1.5} />
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
