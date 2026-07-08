"""Evaluation suite reproducing the paper's experiment set (§4.1-4.2):
binary disruption detection, 4-class severity, F1 vs. forecast horizon,
physics-violation diagnostics, and the data-efficiency study.

"Disruption" for binary detection = severity >= moderate (>=10% capacity
reduction), i.e. classes {2, 3}; risk score = P(moderate) + P(major).
"""

import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from .physics import violation_metrics

BINARY_THRESHOLD_CLASS = 2


@torch.no_grad()
def predict(model, g, ds, batch_size=32):
    """Return probs [W, H, N, C], labels [W, H, N], plus physics outputs."""
    probs, labels, node_phys, edge_phys, batches = [], [], [], [], []
    idxs = np.arange(len(ds))
    for s in range(0, len(idxs), batch_size):
        b = ds.get_batch(idxs[s:s + batch_size])
        logits, nph, eph = model(g, b.dyn)
        probs.append(torch.softmax(logits, dim=-1))
        labels.append(b.labels)
        node_phys.append(nph)
        edge_phys.append(eph)
        batches.append(b)
    return (torch.cat(probs), torch.cat(labels),
            torch.cat(node_phys), torch.cat(edge_phys), batches)


def _binary_metrics(y_true_cls, risk, y_pred_cls):
    y_true = (y_true_cls >= BINARY_THRESHOLD_CLASS).astype(int)
    y_pred = (y_pred_cls >= BINARY_THRESHOLD_CLASS).astype(int)
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_true.min() != y_true.max():
        out["auc"] = float(roc_auc_score(y_true, risk))
    return out


@torch.no_grad()
def evaluate_model(model, g, ds, horizons) -> dict:
    probs, labels, node_phys, edge_phys, batches = predict(model, g, ds)
    probs_np = probs.numpy()
    labels_np = labels.numpy()
    pred_cls = probs_np.argmax(-1)
    risk = probs_np[..., BINARY_THRESHOLD_CLASS:].sum(-1)

    results = {"per_horizon": {}}
    for hi, h in enumerate(horizons):
        yt, yp, rk = labels_np[:, hi].ravel(), pred_cls[:, hi].ravel(), \
            risk[:, hi].ravel()
        m = _binary_metrics(yt, rk, yp)
        m["weighted_f1_multiclass"] = float(
            f1_score(yt, yp, average="weighted", zero_division=0))
        m["macro_f1_multiclass"] = float(
            f1_score(yt, yp, average="macro", zero_division=0))
        results["per_horizon"][f"{h}w"] = m

    # headline metrics at shortest horizon
    results["binary_1w"] = results["per_horizon"][f"{horizons[0]}w"]
    yt0, yp0 = labels_np[:, 0].ravel(), pred_cls[:, 0].ravel()
    results["confusion_1w"] = confusion_matrix(
        yt0, yp0, labels=list(range(probs_np.shape[-1]))).tolist()

    # physics violations aggregated over the eval set
    viol = [violation_metrics(g, b, node_phys[i * 32:(i + 1) * 32],
                              edge_phys[i * 32:(i + 1) * 32])
            for i, b in enumerate(batches)]
    results["physics_violations"] = {
        k: float(np.mean([v[k] for v in viol])) for k in viol[0]}

    results["_risk"] = risk          # [W, H, N] for downstream plotting
    results["_labels"] = labels_np
    return results


def strip_arrays(results: dict) -> dict:
    return {k: v for k, v in results.items() if not k.startswith("_")}


def data_efficiency_study(g, train_ds, val_ds, test_ds, cfg, fractions,
                          train_fn) -> dict:
    """Retrain both models on shrinking training fractions (paper §4.2)."""
    out = {}
    for frac in fractions:
        row = {}
        for name, physics in (("pignn", True), ("baseline", False)):
            print(f"[data-efficiency] {name} @ {int(frac * 100)}% training data")
            model, _ = train_fn(g, train_ds, val_ds, cfg, physics=physics,
                                train_fraction=frac, verbose=False)
            res = evaluate_model(model, g, test_ds, cfg.HORIZONS)
            row[name] = res["binary_1w"]["f1"]
        out[f"{int(frac * 100)}%"] = row
    return out
