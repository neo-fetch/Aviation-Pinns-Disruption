import { useEffect, useState } from "react";
import type { SimControls, SimState } from "../useSimulation";
import type { NetworkResponse } from "../types";

interface Props {
  network: NetworkResponse;
  sim: SimState;
  controls: SimControls;
  injectNode: number | null;
}

const KIND_LABELS: Record<string, string> = {
  supplier_outage: "Supplier outage (fire / quality escape)",
  capacity_cut: "Capacity cut (inspection campaign)",
  leadtime_spike: "Lead-time spike (logistics chokepoint)",
  export_restriction: "Export restriction (embargo)",
  demand_surge: "Demand surge (ramp-up)",
};

export function ScenarioPanel({ network, sim, controls, injectNode }: Props) {
  const [kind, setKind] = useState("supplier_outage");
  const [node, setNode] = useState(0);
  const [magnitude, setMagnitude] = useState(0.7);
  const [duration, setDuration] = useState(6);
  const [offset, setOffset] = useState(0);

  const [seed, setSeed] = useState<string>("");
  const [autoEpisodes, setAutoEpisodes] = useState(true);
  const [nEpisodes, setNEpisodes] = useState(25);

  useEffect(() => {
    if (injectNode != null) setNode(injectNode);
  }, [injectNode]);

  const injectable = network.nodes.filter((n) => n.tier !== "customer");

  return (
    <div className="panel">
      <h2>Inject disruption</h2>
      <label className="field">
        type
        <select value={kind} onChange={(e) => setKind(e.target.value)}>
          {network.meta.disruption_types.map((k) => (
            <option key={k} value={k}>
              {KIND_LABELS[k] ?? k}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        epicenter node
        <select value={node} onChange={(e) => setNode(Number(e.target.value))}>
          {injectable.map((n) => (
            <option key={n.id} value={n.id}>
              {n.name}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        magnitude · {(magnitude * 100).toFixed(0)}%
        <input
          type="range"
          min={0.1}
          max={0.95}
          step={0.05}
          value={magnitude}
          onChange={(e) => setMagnitude(Number(e.target.value))}
        />
      </label>
      <div className="field-row">
        <label className="field">
          duration (weeks)
          <input
            type="number"
            min={1}
            max={26}
            value={duration}
            onChange={(e) => setDuration(Number(e.target.value))}
          />
        </label>
        <label className="field">
          starts in (weeks)
          <input
            type="number"
            min={0}
            max={52}
            value={offset}
            onChange={(e) => setOffset(Number(e.target.value))}
          />
        </label>
      </div>
      <button
        className="primary"
        disabled={!sim.connected}
        onClick={() =>
          controls.inject({
            kind,
            node,
            magnitude,
            duration,
            start_offset: offset,
          })
        }
      >
        Inject
      </button>

      <h2 style={{ marginTop: 10 }}>Scenario</h2>
      <div className="field-row">
        <label className="field">
          seed (blank = random)
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
          />
        </label>
        <label className="field">
          auto episodes
          <select
            value={autoEpisodes ? "on" : "off"}
            onChange={(e) => setAutoEpisodes(e.target.value === "on")}
          >
            <option value="on">on ({nEpisodes} random)</option>
            <option value="off">off (quiet baseline)</option>
          </select>
        </label>
      </div>
      {autoEpisodes && (
        <label className="field">
          number of random episodes · {nEpisodes}
          <input
            type="range"
            min={0}
            max={80}
            value={nEpisodes}
            onChange={(e) => setNEpisodes(Number(e.target.value))}
          />
        </label>
      )}
      <button
        disabled={!sim.connected}
        onClick={() =>
          controls.reset({
            seed: seed === "" ? undefined : Number(seed),
            auto_episodes: autoEpisodes,
            n_episodes: nEpisodes,
          })
        }
      >
        Reset simulation
      </button>
      <div className="muted">seed {sim.seed ?? "—"}</div>

      <h2 style={{ marginTop: 10 }}>Scheduled episodes ({sim.episodes.length})</h2>
      <div className="episode-list">
        {sim.episodes
          .slice()
          .sort((a, b) => a.start - b.start)
          .map((ep, i) => (
            <div className="episode-item" key={ep.id ?? i}>
              <span>
                {ep.kind.replace("_", " ")} · {network.nodes[ep.node]?.name}
              </span>
              <span>
                wk {ep.start}–{ep.start + ep.duration} · {(ep.magnitude * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        {sim.episodes.length === 0 && (
          <div className="muted">none — inject one above</div>
        )}
      </div>
    </div>
  );
}
