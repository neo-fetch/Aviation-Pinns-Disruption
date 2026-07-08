"""Counterfactual mitigation analysis (paper §5, "mitigation strategy
generation ... through counterfactual simulation").

Given a trained PI-GNN and a test window with elevated predicted risk, apply
candidate interventions to the *input state* and re-score the network:

  - safety_stock:  raise the target node's inventory ratio and days-of-supply
                   toward full storage (pre-positioning buffer stock)
  - expedite:      clear the target node's backlog and boost recent arrivals
                   (expedited shipments / premium freight)

The delta in predicted network risk quantifies the intervention's value —
a decision-support signal, not an operational plan (that remains the job of
the optimization/simulation stack, e.g. AnyLogic).
"""

import numpy as np
import torch

# dynamic feature indices (see simulate.py node_dyn layout)
F_INV, F_UTIL, F_DOS, F_BACKLOG, F_ARR, F_RED = range(6)


def _network_risk(model, g, dyn):
    with torch.no_grad():
        logits, _, _ = model(g, dyn)
        probs = torch.softmax(logits, dim=-1)
    return probs[..., 2:].sum(-1)  # [B, H, N] P(moderate)+P(major)


def apply_intervention(dyn: torch.Tensor, node: int, kind: str) -> torch.Tensor:
    d = dyn.clone()
    if kind == "safety_stock":
        d[:, :, node, F_INV] = torch.clamp(d[:, :, node, F_INV] + 0.5, max=1.0)
        d[:, :, node, F_DOS] = torch.clamp(d[:, :, node, F_DOS] + 0.8, max=2.0)
    elif kind == "expedite":
        d[:, :, node, F_BACKLOG] = 0.0
        d[:, :, node, F_ARR] = torch.clamp(d[:, :, node, F_ARR] + 0.3, max=2.0)
    else:
        raise ValueError(kind)
    return d


def mitigation_study(model, g, test_ds, node_names, top_k=5) -> list:
    """Find the riskiest (window, node) pairs on the test set and score both
    interventions on each, including downstream spillover risk reduction."""
    idxs = np.arange(len(test_ds))
    batch = test_ds.get_batch(idxs)
    base_risk = _network_risk(model, g, batch.dyn)     # [W, H, N]
    risk_h = base_risk[:, -1]                          # longest horizon [W, N]

    flat = torch.argsort(risk_h.ravel(), descending=True)
    seen, targets = set(), []
    for f in flat.tolist():
        w, n = divmod(f, risk_h.shape[1])
        if n not in seen:
            seen.add(n)
            targets.append((w, n))
        if len(targets) == top_k:
            break

    # downstream (2-hop) successor mask per node for spillover measurement
    src, dst = g.edge_index
    succ = {int(u): set() for u in range(g.static.shape[0])}
    for u, v in zip(src.tolist(), dst.tolist()):
        succ.setdefault(u, set()).add(v)

    results = []
    for w, n in targets:
        downstream = set(succ.get(n, set()))
        downstream |= {v for u in list(downstream) for v in succ.get(u, set())}
        ds_idx = sorted(downstream) or [n]
        row = {"node": int(n), "node_name": node_names[n],
               "window": int(w),
               "base_risk_node": float(risk_h[w, n]),
               "base_risk_downstream": float(risk_h[w, ds_idx].mean()),
               "interventions": {}}
        one = batch.dyn[w:w + 1]
        for kind in ("safety_stock", "expedite"):
            new_risk = _network_risk(model, g,
                                     apply_intervention(one, n, kind))[0, -1]
            row["interventions"][kind] = {
                "risk_node_after": float(new_risk[n]),
                "risk_downstream_after": float(new_risk[ds_idx].mean()),
                "node_risk_reduction_pct": float(
                    100 * (risk_h[w, n] - new_risk[n])
                    / max(float(risk_h[w, n]), 1e-6)),
            }
        results.append(row)
    return results
