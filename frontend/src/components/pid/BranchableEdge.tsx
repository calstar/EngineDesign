import { useRef, useState, useCallback } from 'react';
import { flushSync } from 'react-dom';
import {
  BaseEdge,
  getSmoothStepPath,
  Position,
  useReactFlow,
  type EdgeProps,
  type Edge,
} from '@xyflow/react';
import { FLUID_COLORS, type FluidType } from './types';

let _jid = 1;
const jid = () => `junc_${_jid++}`;

/** Half the junction node's side length (10 px ÷ 2). */
const J_HALF = 5;

export function BranchableEdge(props: EdgeProps) {
  const {
    id, source, sourceHandle, target, targetHandle,
    sourceX, sourceY, targetX, targetY,
    sourcePosition, targetPosition,
    style, data,
  } = props;

  const { setNodes, setEdges } = useReactFlow();
  const pathRef = useRef<SVGPathElement>(null);
  const [dot, setDot] = useState<{ svgX: number; svgY: number } | null>(null);

  // Sub-edges created by splitting store the correct Position values in data so
  // we don't depend on React Flow's async handle measurement for the first render.
  const overrideSourcePos = (data as { sourcePosition?: Position })?.sourcePosition;
  const overrideTargetPos = (data as { targetPosition?: Position })?.targetPosition;

  // When endpoints are nearly aligned on one axis, draw a straight line instead
  // of letting getSmoothStepPath produce an L-shape from a few-pixel offset.
  const dx = Math.abs(sourceX - targetX);
  const dy = Math.abs(sourceY - targetY);
  const ALIGN_THRESHOLD = 15;

  let edgePath: string;
  if (dx < ALIGN_THRESHOLD && dy > dx) {
    // Nearly vertical — straight line
    edgePath = `M ${sourceX},${sourceY} L ${targetX},${targetY}`;
  } else if (dy < ALIGN_THRESHOLD && dx > dy) {
    // Nearly horizontal — straight line
    edgePath = `M ${sourceX},${sourceY} L ${targetX},${targetY}`;
  } else {
    [edgePath] = getSmoothStepPath({
      sourceX, sourceY, sourcePosition: overrideSourcePos ?? sourcePosition,
      targetX, targetY, targetPosition: overrideTargetPos ?? targetPosition,
    });
  }

  const nearestPoint = useCallback((e: React.MouseEvent<SVGPathElement>) => {
    const path = pathRef.current;
    if (!path) return null;
    const svg = path.ownerSVGElement!;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const local = pt.matrixTransform(path.getScreenCTM()!.inverse());
    const len = path.getTotalLength();
    const step = Math.max(1, len / 100);
    let best = path.getPointAtLength(0), bestD = Infinity;
    for (let t = 0; t <= len; t += step) {
      const p = path.getPointAtLength(t);
      const d = Math.hypot(p.x - local.x, p.y - local.y);
      if (d < bestD) { bestD = d; best = p; }
    }
    return { svgX: best.x, svgY: best.y };
  }, []);

  const splitEdge = useCallback((
    junctionId: string,
    position: { x: number; y: number },
    isHorizontal: boolean,
    origSourcePos: Position,
    origTargetPos: Position,
  ) => {
    const es = style ?? { stroke: FLUID_COLORS.default, strokeWidth: 2 };
    const fluidType = (data as { fluidType?: FluidType })?.fluidType ?? 'default';

    const jTargetHandle = isHorizontal ? 'l' : 't';
    const jSourceHandle = isHorizontal ? 'r' : 'b';
    const jTargetPos = isHorizontal ? Position.Left : Position.Top;
    const jSourcePos = isHorizontal ? Position.Right : Position.Bottom;

    // Resolve explicit handle IDs from the edge's live positions so we never
    // pass null/undefined and let React Flow auto-pick a different handle.
    const resolvedSourceHandle = sourceHandle
      ?? (origSourcePos === Position.Right ? 'r' : origSourcePos === Position.Bottom ? 'b'
        : origSourcePos === Position.Left ? 'l' : 't');
    const resolvedTargetHandle = targetHandle
      ?? (origTargetPos === Position.Left ? 'l' : origTargetPos === Position.Top ? 't'
        : origTargetPos === Position.Right ? 'r' : 'b');

    setNodes(nds => [...nds, { id: junctionId, type: 'JUNCTION', position, data: {} }]);
    setEdges((eds: Edge[]) => [
      ...eds.filter(ex => ex.id !== id),
      {
        id: jid(),
        source, sourceHandle: resolvedSourceHandle,
        target: junctionId, targetHandle: jTargetHandle,
        type: 'smoothstep', style: es,
        data: { fluidType, sourcePosition: origSourcePos, targetPosition: jTargetPos },
      },
      {
        id: jid(),
        source: junctionId, sourceHandle: jSourceHandle,
        target, targetHandle: resolvedTargetHandle,
        type: 'smoothstep', style: es,
        data: { fluidType, sourcePosition: jSourcePos, targetPosition: origTargetPos },
      },
    ] as Edge[]);
  }, [id, source, sourceHandle, target, targetHandle, style, data, setNodes, setEdges]);

  const onMouseMove = useCallback((e: React.MouseEvent<SVGPathElement>) => {
    const p = nearestPoint(e);
    if (p) setDot(p);
  }, [nearestPoint]);

  const onDotPointerDown = useCallback((e: React.PointerEvent) => {
    e.stopPropagation();
    e.preventDefault();
    if (!dot) return;

    const junctionId = jid();
    const isHorizontal = Math.abs(targetX - sourceX) >= Math.abs(targetY - sourceY);
    const branchHandleId = isHorizontal ? 'b' : 'r';

    // Snap junction center to 10px grid on both axes.
    const GRID = 10;
    const snap = (v: number) => Math.round(v / GRID) * GRID;
    const position = {
      x: snap(dot.svgX) - J_HALF,
      y: snap(dot.svgY) - J_HALF,
    };

    flushSync(() => {
      splitEdge(junctionId, position, isHorizontal, sourcePosition, targetPosition);
    });
    setDot(null);

    // The user's pointer is still physically held down (same pointerId).
    // Forward it to the junction's branch handle so React Flow starts a
    // connection drag seamlessly from the click.
    const handle = document.querySelector(
      `[data-nodeid="${junctionId}"][data-handleid="${branchHandleId}"]`,
    ) as HTMLElement | null;

    handle?.dispatchEvent(
      new PointerEvent('pointerdown', {
        bubbles: true, cancelable: true,
        clientX: e.clientX, clientY: e.clientY,
        pointerId:   e.pointerId,
        pointerType: e.pointerType,
        isPrimary:   e.isPrimary,
        pressure:    e.pressure,
      }),
    );
  }, [dot, splitEdge, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition]);

  const fluidType = (data as { fluidType?: FluidType })?.fluidType ?? 'default';
  const lineColor = (style?.stroke as string) ?? FLUID_COLORS[fluidType];

  return (
    <>
      <BaseEdge path={edgePath} style={style} />

      {/*
        <g> wrapper prevents hover-dot flicker: sibling moves between the wide
        hit-zone path and the dot circle don't trigger onMouseLeave on the group,
        so setDot(null) only fires when the cursor truly exits the edge area.
      */}
      <g onMouseLeave={() => setDot(null)}>
        <path
          ref={pathRef}
          d={edgePath}
          fill="none"
          stroke="transparent"
          strokeWidth={20}
          style={{ cursor: 'crosshair' }}
          onMouseMove={onMouseMove}
        />
        {dot && (
          <circle
            cx={dot.svgX} cy={dot.svgY} r={6}
            fill={lineColor} stroke="#0f172a" strokeWidth={2}
            style={{ cursor: 'crosshair', pointerEvents: 'all' }}
            onPointerDown={onDotPointerDown}
            onClick={e => e.stopPropagation()}
          />
        )}
      </g>
    </>
  );
}
