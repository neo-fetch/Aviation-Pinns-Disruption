import type { ModelStatus, NetworkResponse, WeekFrame } from "../types";
import { startTraining } from "../api";

interface Props {
  network: NetworkResponse;
  latest: WeekFrame | null;
  model: ModelStatus | null;
  horizonIdx: number;
  setHorizonIdx(i: number): void;
  selected: number | null;
  onSelect(id: number): void;
}

export function RiskPanel({
  network,
  latest,
  model,
  horizonIdx,
  setHorizonIdx,
  selected,
  onSelect,
}: Props) {
  const horizons = network.meta.horizons;
  const risk = latest?.risk;

  const ranked = risk
    ? risk.risk[horizonIdx]
        .map((r, id) => ({ id, r }))
        .sort((a, b) => b.r - a.r)
        .slice(0, 18)
    : [];

  return (
    <div className="panel">
      <h2>PI-GNN disruption risk</h2>
      <div className="muted">
        Model state: <b>{latest?.model_state ?? model?.state ?? "…"}</b>
        {model?.val_f1 != null && ` · val F1 ${model.val_f1.toFixed(2)}`}
        {model?.state === "training" && model.training_seconds != null &&
          ` · ${Math.round(model.training_seconds)}s elapsed`}
      </div>

      {(model?.state === "untrained" || model?.state === "failed") && (
        <div>
          <div className="muted" style={{ marginBottom: 8 }}>
            {model.state === "failed"
              ? `Training failed: ${model.error}`
              : "No trained model yet."}
          </div>
          <div className="field-row">
            <button className="primary" onClick={() => startTraining("fast")}>
              Train (fast)
            </button>
            <button onClick={() => startTraining("full")}>Train (full)</button>
          </div>
        </div>
      )}
      {model?.state === "ready" && (
        <div className="field-row">
          <button onClick={() => startTraining("full")}>Retrain (full quality)</button>
        </div>
      )}

      <div className="field-row" role="group" aria-label="forecast horizon">
        {horizons.map((h, i) => (
          <button
            key={h}
            className={i === horizonIdx ? "primary" : ""}
            onClick={() => setHorizonIdx(i)}
          >
            +{h}w
          </button>
        ))}
      </div>

      {!risk && (
        <div className="muted">
          Risk scores appear once the model is trained and the simulation has
          run at least {network.meta.window_weeks} weeks (the model's input
          window).
        </div>
      )}

      {risk && (
        <div>
          <div className="muted" style={{ marginBottom: 6 }}>
            P(moderate+ disruption) in {horizons[horizonIdx]} week
            {horizons[horizonIdx] > 1 ? "s" : ""} — top nodes
          </div>
          {ranked.map(({ id, r }) => (
            <div
              key={id}
              className={`risk-row${selected === id ? " selected" : ""}`}
              onClick={() => onSelect(id)}
              title={network.nodes[id].name}
            >
              <span className="name">{network.nodes[id].name}</span>
              <span className="risk-bar-track">
                <span className="risk-bar" style={{ width: `${r * 100}%` }} />
              </span>
              <span className="risk-val">{(r * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
