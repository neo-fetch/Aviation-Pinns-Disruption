import { useEffect, useState } from "react";
import { fetchModelStatus, fetchNetwork } from "./api";
import type { ModelStatus, NetworkResponse } from "./types";
import { useSimulation } from "./useSimulation";
import { NetworkView, type ColorMode } from "./components/NetworkView";
import { NodePanel } from "./components/NodePanel";
import { RiskPanel } from "./components/RiskPanel";
import { ScenarioPanel } from "./components/ScenarioPanel";

type Tab = "inspector" | "risk" | "scenario";

export default function App() {
  const [network, setNetwork] = useState<NetworkResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [sim, controls] = useSimulation();
  const [tab, setTab] = useState<Tab>("risk");
  const [selected, setSelected] = useState<number | null>(null);
  const [colorMode, setColorMode] = useState<ColorMode>("severity");
  const [horizonIdx, setHorizonIdx] = useState(0);
  const [injectNode, setInjectNode] = useState<number | null>(null);
  const [model, setModel] = useState<ModelStatus | null>(null);

  useEffect(() => {
    fetchNetwork().then(setNetwork).catch((e) => setLoadError(String(e)));
  }, []);

  // poll model status while it's training so the badge flips to ready
  useEffect(() => {
    let timer: number | undefined;
    const poll = async () => {
      try {
        const st = await fetchModelStatus();
        setModel(st);
        if (st.state === "training" || st.state === "untrained") {
          timer = window.setTimeout(poll, 3000);
        }
      } catch {
        timer = window.setTimeout(poll, 5000);
      }
    };
    poll();
    return () => window.clearTimeout(timer);
  }, [sim.latest?.model_state]);

  if (loadError) {
    return (
      <div className="app">
        <div className="panel">
          <h2>Backend unreachable</h2>
          <div className="muted">
            {loadError} — start it with{" "}
            <code>uvicorn server.app:app --port 8000</code> from the repo root.
          </div>
        </div>
      </div>
    );
  }
  if (!network) return <div className="app" />;

  const modelState = sim.latest?.model_state ?? model?.state ?? "…";

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>Aviation PI-GNN</h1>
          <div className="sub">
            A320 supply chain · {network.nodes.length} nodes ·{" "}
            {network.edges.length} edges
          </div>
        </div>

        <div className="week-display">
          <span>week</span>
          {sim.week}
        </div>

        <div className="controls">
          <button
            className="primary"
            disabled={!sim.connected}
            onClick={sim.playing ? controls.pause : controls.play}
          >
            {sim.playing ? "Pause" : "Play"}
          </button>
          <button disabled={!sim.connected || sim.playing} onClick={controls.step}>
            Step
          </button>
          <label className="sub" style={{ display: "flex", alignItems: "center", gap: 6 }}>
            speed
            <input
              type="range"
              min={0.5}
              max={12}
              step={0.5}
              value={sim.speed}
              onChange={(e) => controls.setSpeed(Number(e.target.value))}
            />
            {sim.speed.toFixed(1)} wk/s
          </label>
        </div>

        <div className="controls">
          <label className="sub" style={{ display: "flex", alignItems: "center", gap: 6 }}>
            color by
            <select
              value={colorMode}
              onChange={(e) => setColorMode(e.target.value as ColorMode)}
            >
              <option value="severity">actual severity</option>
              <option value="risk">predicted risk</option>
            </select>
          </label>
          {colorMode === "risk" && (
            <select
              value={horizonIdx}
              onChange={(e) => setHorizonIdx(Number(e.target.value))}
              aria-label="risk horizon"
            >
              {network.meta.horizons.map((h, i) => (
                <option key={h} value={i}>
                  +{h}w
                </option>
              ))}
            </select>
          )}
        </div>

        <span className={`badge ${modelState}`}>model: {modelState}</span>

        <div style={{ marginLeft: "auto" }}>
          {(sim.latest?.active_episodes ?? []).slice(0, 4).map((ep, i) => (
            <span className="event-chip" key={i} title={network.nodes[ep.node]?.name}>
              ⚠ {ep.kind.replace("_", " ")} @ {network.nodes[ep.node]?.name?.slice(0, 22)}
            </span>
          ))}
        </div>
      </header>

      <div className="main">
        {!sim.connected && (
          <div className="conn-banner">connecting to simulation backend…</div>
        )}
        <NetworkView
          network={network}
          latest={sim.latest}
          colorMode={colorMode}
          horizonIdx={horizonIdx}
          selected={selected}
          onSelect={(id) => {
            setSelected(id);
            if (id != null) setTab("inspector");
          }}
        />
        <aside className="sidebar">
          <div className="tabs">
            {(["inspector", "risk", "scenario"] as Tab[]).map((t) => (
              <button
                key={t}
                className={tab === t ? "active" : ""}
                onClick={() => setTab(t)}
              >
                {t[0].toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>
          {tab === "inspector" && (
            <NodePanel
              network={network}
              frames={sim.frames}
              latest={sim.latest}
              selected={selected}
              horizonIdx={horizonIdx}
              onInjectHere={(id) => {
                setInjectNode(id);
                setTab("scenario");
              }}
            />
          )}
          {tab === "risk" && (
            <RiskPanel
              network={network}
              latest={sim.latest}
              model={model}
              horizonIdx={horizonIdx}
              setHorizonIdx={setHorizonIdx}
              selected={selected}
              onSelect={(id) => {
                setSelected(id);
              }}
            />
          )}
          {tab === "scenario" && (
            <ScenarioPanel
              network={network}
              sim={sim}
              controls={controls}
              injectNode={injectNode}
            />
          )}
        </aside>
      </div>
    </div>
  );
}
