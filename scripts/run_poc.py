"""End-to-end PoC runner: build the Airbus network, simulate 3 years of weekly
dynamics with disruptions, run the dual-framework topology analysis, train
PI-GNN and the physics-free baseline, evaluate both, run the data-efficiency
and mitigation studies, and write metrics + figures to outputs/.

Usage:  python scripts/run_poc.py [--fast]
        --fast: fewer epochs / skip data-efficiency (smoke test)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import config as cfg
from pignn.dataset import build_graph_tensors, temporal_splits
from pignn.evaluate import (data_efficiency_study, evaluate_model,
                            strip_arrays)
from pignn.mitigate import mitigation_study
from pignn.network import build_airbus_network, node_static_features
from pignn.simulate import simulate
from pignn.topology import (analytical_measures, robustness_simulation,
                            structural_node_features, vulnerability_ranking)
from pignn.train import train_model

TIER_COLORS = {"raw": "#8c6d31", "tier2": "#e6a03c", "tier1": "#d1495b",
               "plant": "#30638e", "fal": "#003d5b", "customer": "#6a8e7f"}


def plot_network(G, out):
    pos = {}
    by_tier = {}
    for n, d in G.nodes(data=True):
        by_tier.setdefault(d["tier_idx"], []).append(n)
    for ti, nodes in by_tier.items():
        for j, n in enumerate(sorted(nodes)):
            pos[n] = (ti * 2.0, j - len(nodes) / 2)
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = [TIER_COLORS[G.nodes[n]["tier"]] for n in G.nodes]
    sizes = [220 if G.nodes[n]["is_sole_source"] else 90 for n in G.nodes]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, arrowsize=6, width=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                           linewidths=0)
    for tier, color in TIER_COLORS.items():
        ax.scatter([], [], c=color, label=tier)
    ax.scatter([], [], c="gray", s=220, label="sole-source (large)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Synthetic Airbus A320-family supply network "
                 f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_robustness(rob, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    f = rob["removal_fractions"]
    ax.plot(f, rob["lcc_random"], "o-", label="random removal",
            color="#30638e")
    ax.plot(f, rob["lcc_targeted"], "s-", label="targeted (betweenness)",
            color="#d1495b")
    ax.set_xlabel("fraction of nodes removed")
    ax.set_ylabel("largest connected component (fraction)")
    ax.set_title("Robustness simulation (dual-framework, Fig. 1 analogue)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_horizon_decay(res_pignn, res_base, horizons, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    hs = [int(h) for h in horizons]
    for res, name, color in ((res_pignn, "PI-GNN", "#d1495b"),
                             (res_base, "baseline GNN", "#30638e")):
        f1s = [res["per_horizon"][f"{h}w"]["f1"] for h in horizons]
        ax.plot(hs, f1s, "o-", label=name, color=color)
    ax.set_xlabel("forecast horizon (weeks)")
    ax.set_ylabel("binary disruption F1 (moderate+)")
    ax.set_title("Multi-horizon forecast quality (test set)")
    ax.set_xticks(hs)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_data_efficiency(de, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    fracs = list(de.keys())
    x = np.arange(len(fracs))
    ax.plot(x, [de[f]["pignn"] for f in fracs], "o-", label="PI-GNN",
            color="#d1495b")
    ax.plot(x, [de[f]["baseline"] for f in fracs], "s-",
            label="baseline GNN", color="#30638e")
    ax.set_xticks(x, fracs)
    ax.set_xlabel("fraction of training data")
    ax.set_ylabel("binary disruption F1 (test, 1-week horizon)")
    ax.set_title("Data efficiency: physics priors vs. data volume")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_risk_heatmap(risk, labels, node_names, out, top_n=40):
    """risk: [W, H, N] -> heatmap of 1-week-ahead risk for riskiest nodes."""
    r = risk[:, 0, :]  # [W, N]
    order = np.argsort(-r.mean(0))[:top_n]
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(r[:, order].T, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=1)
    ax.set_yticks(range(len(order)),
                  [node_names[i][:38] for i in order], fontsize=6)
    ax.set_xlabel("test week index")
    ax.set_title("PI-GNN predicted disruption risk, 1-week horizon "
                 "(top-risk nodes)")
    fig.colorbar(im, ax=ax, label="P(moderate or major disruption)")
    # overlay true moderate+ events as dots
    true = labels[:, 0, :][:, order].T >= 2
    ys, xs = np.where(true)
    ax.scatter(xs, ys, s=4, c="black", marker="s", label="actual event")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_confusion(cm, out):
    cm = np.array(cm, dtype=float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    classes = ["none", "minor", "moderate", "major"]
    ax.set_xticks(range(4), classes)
    ax.set_yticks(range(4), classes)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("PI-GNN severity confusion (test, 1-week horizon)")
    fig.colorbar(im, ax=ax, label="row-normalized")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if args.fast:
        cfg.EPOCHS = 25
        cfg.WARMUP_EPOCHS = 8
        cfg.CURRICULUM_TAU = 5.0

    t0 = time.time()
    torch.manual_seed(cfg.SEED)
    out_dir = ROOT / cfg.OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)

    # ---- 1. network + simulation ----------------------------------------
    print("== Building Airbus A320-family supply network ==")
    G = build_airbus_network(cfg.SEED)
    node_names = [G.nodes[i]["name"] for i in range(G.number_of_nodes())]
    print(f"   {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("== Simulating weekly dynamics with disruption episodes ==")
    sim = simulate(G, cfg.N_WEEKS, cfg.N_DISRUPTION_EPISODES,
                   cfg.BASE_STOCK_WEEKS,
                   (cfg.SEVERITY_MINOR, cfg.SEVERITY_MODERATE,
                    cfg.SEVERITY_MAJOR), cfg.SEED)
    dist = np.bincount(sim.labels.ravel(), minlength=4)
    print(f"   label distribution none/minor/moderate/major: {dist.tolist()}")

    # ---- 2. dual-framework topology analysis ----------------------------
    print("== Dual-framework topology analysis ==")
    measures = analytical_measures(G)
    ranking = vulnerability_ranking(G)
    robustness = robustness_simulation(G, cfg.SEED)
    plot_network(G, out_dir / "network.png")
    plot_robustness(robustness, out_dir / "robustness.png")

    # ---- 3. datasets ------------------------------------------------------
    static = np.concatenate([node_static_features(G),
                             structural_node_features(G)], axis=1)
    g = build_graph_tensors(sim, static)
    train_ds, val_ds, test_ds = temporal_splits(
        sim, cfg.T_IN, cfg.HORIZONS, cfg.TRAIN_FRAC, cfg.VAL_FRAC)
    print(f"   windows train/val/test: {len(train_ds)}/{len(val_ds)}/"
          f"{len(test_ds)}")

    # ---- 4. train both models --------------------------------------------
    print("== Training PI-GNN (physics constraints, curriculum) ==")
    pignn, hist_p = train_model(g, train_ds, val_ds, cfg, physics=True)
    print("== Training baseline GNN (no physics) ==")
    baseline, hist_b = train_model(g, train_ds, val_ds, cfg, physics=False)

    # ---- 5. evaluation ----------------------------------------------------
    print("== Evaluating on held-out test period ==")
    res_p = evaluate_model(pignn, g, test_ds, cfg.HORIZONS)
    res_b = evaluate_model(baseline, g, test_ds, cfg.HORIZONS)
    plot_horizon_decay(res_p, res_b, cfg.HORIZONS,
                       out_dir / "horizon_decay.png")
    plot_risk_heatmap(res_p["_risk"], res_p["_labels"], node_names,
                      out_dir / "risk_heatmap.png")
    plot_confusion(res_p["confusion_1w"], out_dir / "confusion.png")

    de = {}
    if not args.fast:
        print("== Data-efficiency study (retraining at 100/50/25%) ==")
        de = data_efficiency_study(g, train_ds, val_ds, test_ds, cfg,
                                   cfg.DATA_EFFICIENCY_FRACTIONS, train_model)
        plot_data_efficiency(de, out_dir / "data_efficiency.png")

    # ---- 6. mitigation counterfactuals ------------------------------------
    print("== Counterfactual mitigation study ==")
    mitigation = mitigation_study(pignn, g, test_ds, node_names)

    # ---- 7. report ---------------------------------------------------------
    metrics = {
        "network": measures,
        "vulnerability_ranking_top15": [
            {"node": r[1], "tier": r[2], "score": r[3],
             "betweenness": r[4], "sole_source": r[5]} for r in ranking],
        "robustness": robustness,
        "label_distribution": dist.tolist(),
        "pignn": strip_arrays(res_p),
        "baseline_gnn": strip_arrays(res_b),
        "data_efficiency_f1": de,
        "mitigation_counterfactuals": mitigation,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # risk export for downstream tools (e.g. AnyLogic scenario seeding)
    risk = res_p["_risk"]  # [W, H, N]
    import csv
    with open(out_dir / "risk_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_window", "horizon_weeks", "node", "node_name",
                    "risk_moderate_plus"])
        for wi in range(risk.shape[0]):
            for hi, h in enumerate(cfg.HORIZONS):
                for ni in range(risk.shape[2]):
                    if risk[wi, hi, ni] > 0.2:
                        w.writerow([wi, h, ni, node_names[ni],
                                    round(float(risk[wi, hi, ni]), 4)])

    print("\n================ SUMMARY ================")
    print(f"PI-GNN   1w binary: {res_p['binary_1w']}")
    print(f"Baseline 1w binary: {res_b['binary_1w']}")
    print(f"PI-GNN physics violations:   {res_p['physics_violations']}")
    print(f"Baseline physics violations: {res_b['physics_violations']}")
    if de:
        print(f"Data efficiency F1: {json.dumps(de, indent=2)}")
    print(f"Outputs written to {out_dir}/ "
          f"(total {metrics['runtime_seconds']}s)")


if __name__ == "__main__":
    main()
