import { useMemo, useState } from "react";
import type { NetworkResponse, WeekFrame } from "../types";
import { riskColor, SEVERITY_COLORS, SEVERITY_NAMES, TIER_LABELS } from "../color";

export type ColorMode = "severity" | "risk";

interface Props {
  network: NetworkResponse;
  latest: WeekFrame | null;
  colorMode: ColorMode;
  horizonIdx: number;
  selected: number | null;
  onSelect(id: number | null): void;
}

const VW = 1200;
const VH = 780;
const MARGIN_X = 90;
const MARGIN_Y = 46;

interface Pos {
  x: number;
  y: number;
  r: number;
}

export function NetworkView({
  network,
  latest,
  colorMode,
  horizonIdx,
  selected,
  onSelect,
}: Props) {
  const [hover, setHover] = useState<{ id: number; x: number; y: number } | null>(null);

  const layout = useMemo(() => {
    const byTier = new Map<number, number[]>();
    for (const n of network.nodes) {
      const arr = byTier.get(n.tier_idx) ?? [];
      arr.push(n.id);
      byTier.set(n.tier_idx, arr);
    }
    const nTiers = network.meta.tiers.length;
    const pos = new Map<number, Pos>();
    for (const [tier, ids] of byTier) {
      ids.sort((a, b) => a - b);
      const x = MARGIN_X + (tier / (nTiers - 1)) * (VW - 2 * MARGIN_X);
      ids.forEach((id, i) => {
        const y =
          ids.length === 1
            ? VH / 2
            : MARGIN_Y + (i / (ids.length - 1)) * (VH - 2 * MARGIN_Y);
        const cap = network.nodes[id].prod_capacity;
        pos.set(id, { x, y, r: 5 + Math.min(Math.sqrt(cap) / 4.5, 7) });
      });
    }
    return pos;
  }, [network]);

  const activeEpicenters = useMemo(() => {
    const m = new Map<number, number>(); // node -> max intensity
    for (const ep of latest?.active_episodes ?? []) {
      m.set(ep.node, Math.max(m.get(ep.node) ?? 0, ep.intensity ?? 0));
    }
    return m;
  }, [latest]);

  const fillFor = (id: number): string => {
    if (!latest) return "var(--surface-2)";
    if (colorMode === "risk") {
      const r = latest.risk?.risk[horizonIdx]?.[id];
      return r === undefined ? "var(--surface-2)" : riskColor(r);
    }
    return SEVERITY_COLORS[latest.nodes.severity[id]] ?? "var(--surface-2)";
  };

  const hoverNode = hover ? network.nodes[hover.id] : null;

  return (
    <div className="canvas-wrap">
      <svg viewBox={`0 0 ${VW} ${VH}`} preserveAspectRatio="xMidYMid meet"
           onClick={() => onSelect(null)}>
        {/* tier column headings */}
        {network.meta.tiers.map((t, i) => (
          <text
            key={t}
            x={MARGIN_X + (i / (network.meta.tiers.length - 1)) * (VW - 2 * MARGIN_X)}
            y={20}
            textAnchor="middle"
            fill="var(--ink-muted)"
            fontSize={12}
          >
            {TIER_LABELS[t] ?? t}
          </text>
        ))}

        {/* edges: opacity/width follow live flow (fraction of transport cap) */}
        <g>
          {network.edges.map((e, i) => {
            const a = layout.get(e.source)!;
            const b = layout.get(e.target)!;
            const f = latest?.edges.flow_frac[i] ?? 0;
            return (
              <line
                key={i}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                stroke={f > 0.02 ? "var(--seq-500)" : "var(--baseline)"}
                strokeWidth={0.5 + f * 2.2}
                opacity={0.12 + f * 0.55}
              >
                <title>
                  {`${network.nodes[e.source].name} → ${network.nodes[e.target].name}\nlead ${e.lead_time}w · flow ${(f * 100).toFixed(0)}% of transport cap`}
                </title>
              </line>
            );
          })}
        </g>

        {/* disruption epicenter pulses */}
        <g>
          {[...activeEpicenters.entries()].map(([id, inten]) => {
            const p = layout.get(id)!;
            return (
              <circle
                key={id}
                cx={p.x}
                cy={p.y}
                r={p.r + 5 + inten * 5}
                fill="none"
                stroke="var(--status-critical)"
                strokeWidth={1.5}
                opacity={0.35 + inten * 0.45}
              />
            );
          })}
        </g>

        {/* nodes */}
        <g>
          {network.nodes.map((n) => {
            const p = layout.get(n.id)!;
            return (
              <circle
                key={n.id}
                cx={p.x}
                cy={p.y}
                r={p.r}
                fill={fillFor(n.id)}
                stroke={
                  selected === n.id
                    ? "var(--seq-250)"
                    : n.is_sole_source
                      ? "var(--ink-2)"
                      : "var(--border)"
                }
                strokeWidth={selected === n.id ? 2.5 : n.is_sole_source ? 1.5 : 0.75}
                style={{ cursor: "pointer" }}
                onClick={(ev) => {
                  ev.stopPropagation();
                  onSelect(n.id);
                }}
                onMouseEnter={(ev) => {
                  const rect = (ev.currentTarget.ownerSVGElement as SVGSVGElement)
                    .parentElement!.getBoundingClientRect();
                  setHover({ id: n.id, x: ev.clientX - rect.left, y: ev.clientY - rect.top });
                }}
                onMouseLeave={() => setHover(null)}
              />
            );
          })}
        </g>
      </svg>

      {hoverNode && hover && latest && (
        <div
          className="tooltip"
          style={{
            left: Math.min(hover.x + 14, window.innerWidth - 620),
            top: hover.y + 10,
          }}
        >
          <div className="t-name">{hoverNode.name}</div>
          <div className="t-row">
            <span>tier</span>
            <b>{TIER_LABELS[hoverNode.tier]}{hoverNode.is_sole_source ? " · sole source" : ""}</b>
          </div>
          <div className="t-row">
            <span>severity</span>
            <b>{SEVERITY_NAMES[latest.nodes.severity[hoverNode.id]]}</b>
          </div>
          <div className="t-row">
            <span>utilization</span>
            <b>{(latest.nodes.utilization[hoverNode.id] * 100).toFixed(0)}%</b>
          </div>
          <div className="t-row">
            <span>inventory</span>
            <b>{(latest.nodes.inventory_frac[hoverNode.id] * 100).toFixed(0)}% of storage</b>
          </div>
          {latest.risk && (
            <div className="t-row">
              <span>risk ({latest.risk.horizons[horizonIdx]}w)</span>
              <b>{((latest.risk.risk[horizonIdx]?.[hoverNode.id] ?? 0) * 100).toFixed(1)}%</b>
            </div>
          )}
        </div>
      )}

      {/* legend for the active encoding */}
      <div className="legend">
        {colorMode === "severity" ? (
          <>
            {SEVERITY_NAMES.map((name, i) => (
              <div className="li" key={name}>
                <span className="swatch" style={{ background: SEVERITY_COLORS[i] }} />
                {name}
                {i > 0 && (
                  <span style={{ color: "var(--ink-muted)" }}>
                    (&ge;{(network.meta.severity_thresholds[i - 1] * 100).toFixed(0)}% cap loss)
                  </span>
                )}
              </div>
            ))}
          </>
        ) : (
          <div className="li">
            0% <span className="ramp" /> 100% — P(moderate+ disruption)
          </div>
        )}
        <div className="li">
          <span
            className="swatch"
            style={{ background: "transparent", border: "1.5px solid var(--ink-2)" }}
          />
          sole-source supplier
        </div>
        <div className="li">
          <span
            className="swatch"
            style={{ background: "transparent", border: "1.5px solid var(--status-critical)" }}
          />
          active disruption epicenter
        </div>
      </div>
    </div>
  );
}
