import { useId, useMemo, useRef, useState } from "react";

interface Props {
  title: string;
  values: number[];
  startWeek: number;
  color?: string;
  height?: number;
  format?: (v: number) => string;
}

const W = 300;

/** Single-series line with a soft gradient wash beneath it and a hover
    crosshair + value readout. */
export function Sparkline({
  title,
  values,
  startWeek,
  color = "var(--accent)",
  height = 44,
  format = (v) => v.toFixed(3),
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const gradId = useId();
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const { path, area, min, max } = useMemo(() => {
    if (values.length === 0) return { path: "", area: "", min: 0, max: 1 };
    let mn = Math.min(...values);
    let mx = Math.max(...values);
    if (mx - mn < 1e-9) {
      mn -= 0.5;
      mx += 0.5;
    }
    const n = values.length;
    const pts = values.map((v, i) => {
      const x = n === 1 ? W / 2 : (i / (n - 1)) * W;
      const y = height - 3 - ((v - mn) / (mx - mn)) * (height - 6);
      return [x, y] as const;
    });
    const line = pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join("L");
    const first = pts[0];
    const last = pts[pts.length - 1];
    return {
      path: `M${line}`,
      area: `M${first[0].toFixed(1)},${height - 1}L${line}L${last[0].toFixed(1)},${height - 1}Z`,
      min: mn,
      max: mx,
    };
  }, [values, height]);

  const onMove = (e: React.MouseEvent) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect || values.length === 0) return;
    const frac = (e.clientX - rect.left) / rect.width;
    const idx = Math.round(frac * (values.length - 1));
    setHoverIdx(Math.max(0, Math.min(values.length - 1, idx)));
  };

  const idx = hoverIdx ?? values.length - 1;
  const cur = values[idx];
  const hoverX =
    values.length > 1 ? (idx / (values.length - 1)) * W : W / 2;

  return (
    <div className="spark-block">
      <div className="spark-title">
        <span>{title}</span>
        <b>
          {cur !== undefined
            ? `wk ${startWeek + idx} · ${format(cur)}`
            : "—"}
        </b>
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${height}`}
        style={{ width: "100%", height }}
        onMouseMove={onMove}
        onMouseLeave={() => setHoverIdx(null)}
        role="img"
        aria-label={`${title}: latest ${cur !== undefined ? format(cur) : "n/a"}, range ${format(min)} to ${format(max)}`}
      >
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.22} />
            <stop offset="100%" stopColor={color} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <line
          x1={0}
          y1={height - 1}
          x2={W}
          y2={height - 1}
          stroke="var(--hairline)"
          strokeWidth={1}
        />
        {area && <path d={area} fill={`url(#${gradId})`} />}
        {path && (
          <path
            d={path}
            fill="none"
            stroke={color}
            strokeWidth={2}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        )}
        {hoverIdx !== null && (
          <line
            x1={hoverX}
            y1={0}
            x2={hoverX}
            y2={height}
            stroke="var(--ink-muted)"
            strokeWidth={1}
            strokeDasharray="2,3"
          />
        )}
      </svg>
    </div>
  );
}
