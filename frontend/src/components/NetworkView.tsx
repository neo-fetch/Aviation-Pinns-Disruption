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

// gentle S-curve between tier columns: control points at the horizontal
// midpoint so edges leave and arrive level, like drifting threads
function edgePath(a: Pos, b: Pos): string {
  const mx = (a.x + b.x) / 2;
  return `M${a.x},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x},${b.y}`;
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
    if (!latest) return "var(--seq-0)";
    if (colorMode === "risk") {
      const r = latest.risk?.risk[horizonIdx]?.[id];
      return r === undefined ? "var(--seq-0)" : riskColor(r);
    }
    return SEVERITY_COLORS[latest.nodes.severity[id]] ?? "var(--seq-0)";
  };

  const hoverNode = hover ? network.nodes[hover.id] : null;

  return (
    <div className="canvas-wrap">
      <svg viewBox={`0 0 ${VW} ${VH}`} preserveAspectRatio="xMidYMid meet"
           onClick={() => onSelect(null)}>
        <defs>
          {/* soft lavender halo behind every node */}
          <filter id="softGlow" x="-80%" y="-80%" width="260%" height="260%">
            <feDropShadow dx="0" dy="1" stdDeviation="4"
                          floodColor="#8b93e6" floodOpacity="0.35" />
          </filter>
          {/* wide blur for breathing disruption halos */}
          <filter id="haloBlur" x="-120%" y="-120%" width="340%" height="340%">
            <feGaussianBlur stdDeviation="4" />
          </filter>
        </defs>

        {/* tier column headings */}
        {network.meta.tiers.map((t, i) => (
          <text
            key={t}
            className="tier-label"
            x={MARGIN_X + (i / (network.meta.tiers.length - 1)) * (VW - 2 * MARGIN_X)}
            y={22}
            textAnchor="middle"
          >
            {TIER_LABELS[t] ?? t}
          </text>
        ))}

        {/* edges: drifting threads whose presence follows live flow */}
        <g>
          {network.edges.map((e, i) => {
            const a = layout.get(e.source)!;
            const b = layout.get(e.target)!;
            const f = latest?.edges.flow_frac[i] ?? 0;
            return (
              <path
                key={i}
                className="edge"
                d={edgePath(a, b)}
                stroke={f > 0.02 ? "var(--accent)" : "#c9c4e4"}
                strokeWidth={0.8 + f * 2.2}
                opacity={0.18 + f * 0.5}
              >
                <title>
                  {`${network.nodes[e.source].name} → ${network.nodes[e.target].name}\nlead ${e.lead_time}w · flow ${(f * 100).toFixed(0)}% of transport cap`}
                </title>
              </path>
            );
          })}
        </g>

        {/* disruption epicenters: breathing rose halos */}
        <g>
          {[...activeEpicenters.entries()].map(([id, inten]) => {
            const p = layout.get(id)!;
            return (
              <circle
                key={id}
                className="halo"
                cx={p.x}
                cy={p.y}
                r={p.r + 7 + inten * 4}
                fill="var(--sev-3)"
                opacity={0.3 + inten * 0.3}
                filter="url(#haloBlur)"
              />
            );
          })}
        </g>

        {/* nodes: softly glowing orbs */}
        <g filter="url(#softGlow)">
          {network.nodes.map((n) => {
            const p = layout.get(n.id)!;
            return (
              <circle
                key={n.id}
                className="node-dot"
                cx={p.x}
                cy={p.y}
                r={p.r}
                fill={fillFor(n.id)}
                stroke={
                  n.is_sole_source
                    ? "rgba(255, 255, 255, 0.95)"
                    : "rgba(255, 255, 255, 0.55)"
                }
                strokeWidth={n.is_sole_source ? 2.25 : 1}
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

        {/* selection: gentle periwinkle ring */}
        {selected != null && layout.has(selected) && (
          <circle
            className="select-ring"
            cx={layout.get(selected)!.x}
            cy={layout.get(selected)!.y}
            r={layout.get(selected)!.r + 4.5}
            fill="none"
            stroke="var(--accent)"
            strokeWidth={2}
            opacity={0.85}
            filter="url(#softGlow)"
          />
        )}
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
            style={{
              background: "var(--seq-0)",
              border: "2px solid #fff",
              boxShadow: "0 0 0 1px var(--hairline)",
            }}
          />
          sole-source supplier
        </div>
        <div className="li">
          <span
            className="swatch"
            style={{
              background: "rgba(169, 74, 116, 0.45)",
              boxShadow: "0 0 6px 2px rgba(169, 74, 116, 0.35)",
            }}
          />
          active disruption epicenter
        </div>
      </div>
    </div>
  );
}
