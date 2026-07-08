"""Training loop with physics-constraint curriculum (paper §3.4).

Stage 1 (warm-up): prediction loss only.
Stage 2: physics constraint weights ramp in exponentially via
curriculum_lambda. The baseline GNN is the same model trained with
physics=False throughout — the only difference is the constraint terms.

The prediction loss for BOTH models includes the auxiliary physical-state
regression (production/inventory/consumption/flows), so the physics terms are
the isolated experimental variable, mirroring the paper's ablation design.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .dataset import GraphTensors, WindowDataset
from .model import PIGNN
from .physics import curriculum_lambda, physics_residuals

AUX_WEIGHT = 0.5  # weight of auxiliary physical-state regression in L_pred


def class_weights(train_ds: WindowDataset, n_classes: int) -> torch.Tensor:
    labels = train_ds.sim.labels[
        train_ds.t_lasts[0]:train_ds.t_lasts[-1] + 1]
    counts = np.bincount(labels.ravel(), minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    w = np.clip(w, 0.2, 3.0)
    return torch.tensor(w, dtype=torch.float32)


def prediction_loss(logits, node_phys, edge_phys, batch, cls_w):
    B, H, N, C = logits.shape
    ce = F.cross_entropy(logits.reshape(B * H * N, C),
                         batch.labels.reshape(B * H * N), weight=cls_w)
    aux = (F.mse_loss(node_phys[..., 0], batch.prod_true)
           + F.mse_loss(node_phys[..., 1], batch.inv_true)
           + F.mse_loss(node_phys[..., 2], batch.cons_true)
           + F.mse_loss(edge_phys[..., 0], batch.flow_true)
           + F.mse_loss(edge_phys[..., 1], batch.arrival_true))
    return ce + AUX_WEIGHT * aux, ce, aux


def train_model(g: GraphTensors, train_ds: WindowDataset,
                val_ds: WindowDataset, cfg, physics: bool,
                seed: int = 0, train_fraction: float = 1.0,
                verbose: bool = True):
    """Train one model; physics=False gives the baseline GNN."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = PIGNN(g, f_dyn=train_ds.sim.node_dyn.shape[-1],
                  embed_dim=cfg.NODE_EMBED_DIM, gnn_layers=cfg.GNN_LAYERS,
                  lstm_hidden=cfg.LSTM_HIDDEN, n_horizons=len(cfg.HORIZONS),
                  n_classes=cfg.N_CLASSES)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR,
                           weight_decay=cfg.WEIGHT_DECAY)
    cls_w = class_weights(train_ds, cfg.N_CLASSES)

    n_train = max(int(len(train_ds) * train_fraction), cfg.BATCH_WINDOWS)
    # keep the most recent windows when subsampling (temporal contiguity)
    train_idxs = np.arange(len(train_ds))[-n_train:]

    best_val, best_state, patience = -1.0, None, 0
    history = []
    for epoch in range(cfg.EPOCHS):
        model.train()
        perm = rng.permutation(train_idxs)
        ep_loss, n_b = 0.0, 0
        lam = curriculum_lambda(epoch, cfg.WARMUP_EPOCHS,
                                cfg.CURRICULUM_TAU, 1.0) if physics else 0.0
        for s in range(0, len(perm), cfg.BATCH_WINDOWS):
            batch = train_ds.get_batch(perm[s:s + cfg.BATCH_WINDOWS])
            logits, node_phys, edge_phys = model(g, batch.dyn)
            loss, _, _ = prediction_loss(logits, node_phys, edge_phys,
                                         batch, cls_w)
            if lam > 0:
                lf, lc, ll = physics_residuals(g, batch, node_phys, edge_phys)
                loss = loss + lam * (cfg.LAMBDA_FLOW * lf
                                     + cfg.LAMBDA_CAPACITY * lc
                                     + cfg.LAMBDA_LEAD * ll)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.detach())
            n_b += 1

        # validation: model selection on binary disruption F1 (moderate+),
        # the metric that matters, rather than the class-weighted loss
        model.eval()
        with torch.no_grad():
            vb = val_ds.get_batch(np.arange(len(val_ds)))
            vl_logits, vnp, vep = model(g, vb.dyn)
            val_loss, _, _ = prediction_loss(vl_logits, vnp, vep, vb, cls_w)
            vp = vl_logits.argmax(-1)
            val_f1 = f1_score((vb.labels >= 2).numpy().ravel(),
                              (vp >= 2).numpy().ravel(), zero_division=0)
        history.append({"epoch": epoch, "train_loss": ep_loss / max(n_b, 1),
                        "val_loss": float(val_loss), "val_f1": float(val_f1),
                        "lambda": lam})
        if verbose and (epoch % 10 == 0 or epoch == cfg.EPOCHS - 1):
            print(f"  epoch {epoch:3d}  train {ep_loss / max(n_b, 1):.4f}  "
                  f"val {float(val_loss):.4f}  valF1 {val_f1:.3f}  "
                  f"lam {lam:.3f}")

        # early stopping only once training is mature (and, for the PI-GNN,
        # after the physics curriculum has mostly ramped in)
        min_epoch = cfg.WARMUP_EPOCHS + cfg.CURRICULUM_TAU if physics else 40
        if epoch >= min_epoch:
            if val_f1 > best_val + 1e-4:
                best_val, patience = float(val_f1), 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 25:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history
