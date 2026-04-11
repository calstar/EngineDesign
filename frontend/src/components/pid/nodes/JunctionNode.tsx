import { Handle, Position, type NodeProps } from '@xyflow/react';

export function JunctionNode(_props: NodeProps) {
  // Handles are flush with the node edges; make them wide enough to grab easily.
  const handleStyle = {
    width: 10,
    height: 10,
    background: 'transparent',
    border: 'none',
  };

  return (
    <div
      style={{
        width: 10,
        height: 10,
        borderRadius: '50%',
        background: '#94a3b8',
        border: '2px solid #0f172a',
        boxShadow: '0 0 0 2px #475569',
        position: 'relative',
      }}
      className="nodrag"
    >
      {/* target handles — incoming pipes connect here */}
      <Handle type="target" position={Position.Top}   id="t" style={handleStyle} />
      <Handle type="target" position={Position.Left}  id="l" style={handleStyle} />
      {/* source handles — outgoing pipes originate here */}
      <Handle type="source" position={Position.Bottom} id="b" style={handleStyle} />
      <Handle type="source" position={Position.Right}  id="r" style={handleStyle} />
    </div>
  );
}
