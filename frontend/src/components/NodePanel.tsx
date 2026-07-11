import type { NetworkResponse, WeekFrame } from "../types";
import { SEVERITY_NAMES, TIER_LABELS } from "../color";
import { Sparkline } from "./Sparkline";

interface Props {
  network: NetworkResponse;
  frames: WeekFrame[];
  latest: WeekFrame | null;
  selected: number | null;
  horizonIdx: number;
  onInjectHere(id: number): void;
}

const pct = (v: number) => `${(v * 100).toFixed(0)}%`;

export function NodePanel({
  network,
  frames,
  latest,
  selected,
  horizonIdx,
  onInjectHere,
}: Props) {
  if (selected == null) {
    return (
      <div className="panel">
        <h2>Inspector</h2>
        <div className="muted">
          Click a node in the network to inspect its live telemetry, or pick
          one from the Risk tab.
        </div>
        <h2 style={{ marginTop: 8 }}>Structural choke points</h2>
        {network.vulnerability_ranking.slice(0, 10).map((v) => (
          <div key={v.node} className="muted">
            {v.name} <b style={{ color: "var(--ink-2)" }}>{v.score.toFixed(2)}</b>
            {v.sole_source ? " · sole source" : ""}
          </div>
        ))}
      </div>
    );
  }

  const node = network.nodes[selected];
  const startWeek = frames.length > 0 ? frames[0].week : 0;
  const series = (key: keyof WeekFrame["nodes"]) =>
    frames.map((f) => f.nodes[key][selected]);
  const riskSeries = frames
    .filter((f) => f.risk)
    .map((f) => f.risk!.risk[horizonIdx][selected]);
  const riskStart =
    frames.find((f) => f.risk)?.week ?? startWeek;

  return (
    <div className="panel">
      <h2>Inspector</h2>
      <h3>{node.name}</h3>
      <dl className="kv">
        <dt>tier</dt>
        <dd>{TIER_LABELS[node.tier]}</dd>
        <dt>region</dt>
        <dd>{node.region}</dd>
        <dt>production capacity</dt>
        <dd>{node.prod_capacity.toFixed(0)} u/wk</dd>
        <dt>storage capacity</dt>
        <dd>{node.storage_capacity.toFixed(0)} u</dd>
        <dt>reliability</dt>
        <dd>{node.reliability.toFixed(3)}</dd>
        <dt>sole source</dt>
        <dd>{node.is_sole_source ? "yes" : "no"}</dd>
        <dt>suppliers / customers</dt>
        <dd>
          {node.in_degree} / {node.out_degree}
        </dd>
        {latest && (
          <>
            <dt>current severity</dt>
            <dd>{SEVERITY_NAMES[latest.nodes.severity[selected]]}</dd>
          </>
        )}
      </dl>

      <button className="primary" onClick={() => onInjectHere(selected)}>
        Inject disruption here…
      </button>

      {frames.length > 1 ? (
        <>
          <h2>Live telemetry</h2>
          <Sparkline title="Utilization" values={series("utilization")}
                     startWeek={startWeek} format={pct} />
          <Sparkline title="Inventory (frac of storage)" values={series("inventory_frac")}
                     startWeek={startWeek} format={pct} />
          <Sparkline title="Backlog (weeks of capacity)" values={series("backlog")}
                     startWeek={startWeek} format={(v) => v.toFixed(2)} />
          <Sparkline title="Capacity reduction" values={series("cap_reduction")}
                     startWeek={startWeek} format={pct} color="var(--sev-2)" />
          {riskSeries.length > 1 && (
            <Sparkline
              title={`Model risk, +${network.meta.horizons[horizonIdx]}w`}
              values={riskSeries}
              startWeek={riskStart}
              format={pct}
              color="var(--accent-deep)"
            />
          )}
        </>
      ) : (
        <div className="muted">Play the simulation to collect telemetry.</div>
      )}
    </div>
  );
}
