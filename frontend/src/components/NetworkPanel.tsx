import { useMemo, useState } from "react";
import { addCustomNode, removeCustomNode } from "../api";
import type { NetworkResponse } from "../types";

interface Props {
  network: NetworkResponse;
  onNetworkChanged(): void;
}

const TIER_LABELS: Record<string, string> = {
  raw: "Raw material",
  tier2: "Tier-2 component supplier",
  tier1: "Tier-1 system supplier",
  plant: "Airbus component plant",
  fal: "Final assembly line (FAL)",
  customer: "Customer / deliveries",
};

export function NetworkPanel({ network, onNetworkChanged }: Props) {
  const [name, setName] = useState("");
  const [tier, setTier] = useState("fal");
  const [region, setRegion] = useState("APAC");
  const [prod, setProd] = useState(8);
  const [stor, setStor] = useState(32);
  const [reliability, setReliability] = useState(0.95);
  const [soleSource, setSoleSource] = useState(false);
  const [showOverrides, setShowOverrides] = useState(false);
  const [upstream, setUpstream] = useState<string[]>([]);
  const [downstream, setDownstream] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const customNodes = network.nodes.filter((n) => n.is_custom);
  const tierIdx = network.meta.tiers.indexOf(tier);

  // eligible override targets, mirroring the server's tier-ordering rules
  const upstreamOptions = useMemo(
    () =>
      network.nodes.filter(
        (n) => n.tier_idx < tierIdx && n.tier !== "customer",
      ),
    [network, tierIdx],
  );
  const downstreamOptions = useMemo(
    () =>
      network.nodes.filter((n) => n.tier_idx > tierIdx && n.tier !== "raw"),
    [network, tierIdx],
  );

  const toggle = (list: string[], set: (v: string[]) => void, n: string) =>
    set(list.includes(n) ? list.filter((x) => x !== n) : [...list, n]);

  const submit = async () => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await addCustomNode({
        name: name.trim(),
        tier,
        region,
        prod_capacity: prod,
        storage_capacity: stor,
        reliability,
        is_sole_source: soleSource,
        upstream,
        downstream,
      });
      setName("");
      setUpstream([]);
      setDownstream([]);
      setNotice(
        "Network rebuilt — the model is retraining; risk scores return " +
          "when the badge reads ready.",
      );
      onNetworkChanged();
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (nodeName: string) => {
    if (!window.confirm(`Remove custom node "${nodeName}"?`)) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await removeCustomNode(nodeName);
      setNotice(`Removed ${nodeName} — network rebuilt, model retraining.`);
      onNetworkChanged();
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel">
      <h2>Add custom node</h2>
      <label className="field">
        name
        <input
          type="text"
          placeholder="e.g. FAL Bangalore"
          value={name}
          onChange={(e) => setName(e.target.value)}
          maxLength={60}
        />
      </label>
      <div className="field-row">
        <label className="field">
          tier
          <select value={tier} onChange={(e) => setTier(e.target.value)}>
            {network.meta.tiers.map((t) => (
              <option key={t} value={t}>
                {TIER_LABELS[t] ?? t}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          region
          <select value={region} onChange={(e) => setRegion(e.target.value)}>
            {network.meta.regions.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </label>
      </div>
      {tier !== "customer" && (
        <>
          <div className="field-row">
            <label className="field">
              production (units/week)
              <input
                type="number"
                min={1}
                max={2000}
                value={prod}
                onChange={(e) => setProd(Number(e.target.value))}
              />
            </label>
            <label className="field">
              storage (units)
              <input
                type="number"
                min={1}
                max={10000}
                value={stor}
                onChange={(e) => setStor(Number(e.target.value))}
              />
            </label>
          </div>
          <label className="field">
            reliability · {reliability.toFixed(2)}
            <input
              type="range"
              min={0.8}
              max={1.0}
              step={0.01}
              value={reliability}
              onChange={(e) => setReliability(Number(e.target.value))}
            />
          </label>
          <label
            className="field"
            style={{ flexDirection: "row", alignItems: "center", gap: 8 }}
          >
            <input
              type="checkbox"
              checked={soleSource}
              onChange={(e) => setSoleSource(e.target.checked)}
            />
            sole-source choke point
          </label>
        </>
      )}

      <button onClick={() => setShowOverrides(!showOverrides)}>
        {showOverrides ? "Hide" : "Show"} connection overrides
      </button>
      {showOverrides && (
        <>
          <div className="muted">
            Connections are auto-wired to match the built-in network (a new
            FAL is fed by plants and the engine/landing-gear suppliers).
            Picks below are added on top.
          </div>
          <div className="field-row">
            <label className="field">
              extra suppliers ({upstream.length})
              <select
                value=""
                onChange={(e) =>
                  e.target.value &&
                  toggle(upstream, setUpstream, e.target.value)
                }
              >
                <option value="">add / remove…</option>
                {upstreamOptions.map((n) => (
                  <option key={n.id} value={n.name}>
                    {(upstream.includes(n.name) ? "✓ " : "") + n.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              extra destinations ({downstream.length})
              <select
                value=""
                onChange={(e) =>
                  e.target.value &&
                  toggle(downstream, setDownstream, e.target.value)
                }
              >
                <option value="">add / remove…</option>
                {downstreamOptions.map((n) => (
                  <option key={n.id} value={n.name}>
                    {(downstream.includes(n.name) ? "✓ " : "") + n.name}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {(upstream.length > 0 || downstream.length > 0) && (
            <div className="muted">
              {upstream.map((n) => `← ${n}`).join(" · ")}
              {upstream.length > 0 && downstream.length > 0 && " · "}
              {downstream.map((n) => `→ ${n}`).join(" · ")}
            </div>
          )}
        </>
      )}

      <button
        className="primary"
        disabled={busy || !name.trim()}
        onClick={submit}
      >
        {busy ? "Rebuilding…" : "Add node"}
      </button>
      {error && (
        <div className="muted" style={{ color: "var(--sev-3)" }}>
          {error}
        </div>
      )}
      {notice && <div className="muted">{notice}</div>}

      <h2 style={{ marginTop: 10 }}>Custom nodes ({customNodes.length})</h2>
      <div className="episode-list">
        {customNodes.map((n) => (
          <div className="episode-item" key={n.id}>
            <span>
              {n.name} · {n.tier} · {n.region}
            </span>
            <button disabled={busy} onClick={() => remove(n.name)}>
              Remove
            </button>
          </div>
        ))}
        {customNodes.length === 0 && (
          <div className="muted">
            none — the built-in Airbus network is unmodified
          </div>
        )}
      </div>
    </div>
  );
}
