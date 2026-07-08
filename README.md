# Aviation PI-GNN: Airbus Supply Chain Disruption Prediction (Proof of Concept)

A proof-of-concept implementation of **Physics-Informed Graph Neural Networks
for Supply Chain Disruption Prediction and Mitigation** (Petrova & Hughes,
*Frontiers in Applied Physics and Mathematics*, 2025), applied to an
**Airbus A320-family supply chain**.

The model learns to forecast node-level disruption severity (none / minor /
moderate / major) across the multi-tier supplier network at 1, 2, 4 and 8-week
horizons, while physics-based loss terms force its internal picture of the
network to obey conservation of flow, capacity limits, and lead-time
consistency — the "laws of motion" of a supply chain.

## Positioning vs. AnyLogic (deliberately non-overlapping)

| | AnyLogic (existing) | This PoC (new layer) |
|---|---|---|
| Paradigm | Discrete-event / agent-based **simulation & optimization** | **Learned prediction** from network state telemetry |
| Question answered | *"What happens if we run policy X?"* | *"Which nodes are heading toward disruption in the next 1–8 weeks?"* |
| Inputs | Modeled process logic, distributions | Weekly node/edge telemetry (inventory, utilization, backlog, arrivals) |
| Output | KPI trajectories, optimized parameters | Node-level disruption risk scores + severity class per horizon |
| Role | Prescriptive (design & optimize) | Predictive early warning (monitor & alert) |

**Integration point:** `outputs/risk_scores.csv` — per-node, per-horizon
disruption probabilities that can seed AnyLogic disruption scenarios
(which nodes to stress, when, and how hard), replacing hand-picked what-ifs
with model-ranked ones.

## What's implemented (mapping to the paper)

| Paper section | Implementation |
|---|---|
| §3.1 Dual-framework topology analysis | `src/pignn/topology.py` — assortativity, degree distribution, centralization, percolation threshold; random vs. targeted node-removal robustness profiles; structural priors (betweenness, PageRank, clustering, k-core) appended to node features |
| §3.1 Graph representation G=(V,E,X,R) | `src/pignn/network.py` — 65-node, 187-edge directed graph: raw materials → tier-2 → tier-1 majors (engines, aerostructures, landing gear, avionics) → Airbus plants (Hamburg, Saint-Nazaire, Getafe, Stade, Broughton, Filton) → FALs (Toulouse, Hamburg, Tianjin, Mobile) → airline customers |
| §3.2 LSTM temporal memory | `src/pignn/model.py` — per-node LSTM over graph-conv embeddings with a graph-level context stream |
| §3.3 Sequential multi-horizon forecasting | 6-week input windows → severity logits at +1/+2/+4/+8 weeks |
| §3.4 Physics constraints | `src/pignn/physics.py` — flow conservation, capacity hinge penalties, lead-time consistency; exponential curriculum `λ(e)=λ_f(1−e^{−e/τ})` after a prediction-only warm-up |
| §4.1 Evaluation | `src/pignn/evaluate.py` — binary detection (P/R/F1/AUC), 4-class severity F1, F1 vs. horizon, physics-violation diagnostics |
| §4.2 Data efficiency | retraining both models at 100/50/25% training data |
| §5 Mitigation | `src/pignn/mitigate.py` — counterfactual interventions (safety stock, expedited shipments) re-scored through the trained model |

The **baseline GNN** is the identical architecture trained with physics
weights set to zero — isolating the physics terms as the experimental
variable, mirroring the paper's ablation design.

## Synthetic data (and how to replace it with real data)

No proprietary Airbus data is used. `src/pignn/network.py` builds a
structurally realistic A320-program network (sole-source choke points on
engines, nacelles, landing gear, fuselage sections; buyer-furnished-equipment
edges from engine suppliers straight to FALs). `src/pignn/simulate.py`
generates 156 weeks of flow/inventory dynamics that satisfy conservation of
flow **by construction** (verified to ~1e-8 residual), then injects 30
disruption episodes modeled on real aviation events:

- `supplier_outage` — e.g. a fuselage-panel supplier fire or quality escape
- `capacity_cut` — e.g. GTF-style engine inspection campaigns
- `leadtime_spike` — e.g. Red Sea rerouting adding weeks in transit
- `export_restriction` — e.g. titanium sanctions hitting raw-material nodes
- `demand_surge` — FAL ramp-up outpacing supplier capacity

Severity labels follow the paper: capacity reduction <10% minor, 10–30%
moderate, >30% major. To use real data, replace the simulator output with
your own weekly snapshots in the `SimResult` layout (`src/pignn/simulate.py`)
— everything downstream is agnostic to where the snapshots came from.

## Run it

```bash
pip install -r requirements.txt
python scripts/run_poc.py          # full run (~6-8 min on CPU)
python scripts/run_poc.py --fast   # 15-second smoke test
```

Outputs land in `outputs/`:

- `metrics.json` — all metrics: topology analysis, PI-GNN vs. baseline,
  data-efficiency study, mitigation counterfactuals
- `network.png` — the supply network, sole-source nodes enlarged
- `robustness.png` — LCC degradation under random vs. targeted node removal
- `horizon_decay.png` — F1 vs. forecast horizon, both models
- `data_efficiency.png` — F1 vs. training-data fraction, both models
- `risk_heatmap.png` — predicted risk over the test period vs. actual events
- `confusion.png` — 4-class severity confusion matrix
- `risk_scores.csv` — the AnyLogic-ready risk export

## Repository layout

```
config.py                  # every knob: horizons, physics weights, curriculum
src/pignn/
  network.py               # Airbus A320-family supply graph builder
  simulate.py              # weekly dynamics + disruption injection + labels
  topology.py              # dual-framework analysis (paper Fig. 1)
  dataset.py               # sliding-window tensors, temporal 60/20/20 split
  model.py                 # GraphConv + LSTM + multi-horizon & physics heads
  physics.py               # differentiable constraint losses + curriculum
  train.py                 # warm-up → curriculum training loop
  evaluate.py              # metrics suite incl. physics-violation diagnostics
  mitigate.py              # counterfactual mitigation what-ifs
scripts/run_poc.py         # end-to-end runner
```

## PoC limitations

- Single-commodity pooled material flow (no bill-of-materials explosion);
  a production system would track part families per program.
- Disruption dynamics are synthetic; magnitudes/durations are plausible but
  not calibrated to Airbus history.
- The mitigation module scores interventions through the predictive model
  only — validating a chosen intervention operationally is exactly where the
  existing AnyLogic stack takes over.
