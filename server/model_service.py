"""PI-GNN model lifecycle for the live backend: train in a background
thread on the batch simulator's output, checkpoint to disk, and score live
simulation windows into per-node / per-horizon disruption risk."""

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import config as cfg
from pignn.dataset import build_graph_tensors, temporal_splits
from pignn.model import PIGNN
from pignn.simulate import simulate
from pignn.train import train_model

SEVERITY_THRESHOLDS = (cfg.SEVERITY_MINOR, cfg.SEVERITY_MODERATE,
                       cfg.SEVERITY_MAJOR)


def _cfg_namespace(quality: str) -> SimpleNamespace:
    ns = SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg)
                            if k.isupper()})
    if quality == "fast":
        ns.EPOCHS = 25
        ns.WARMUP_EPOCHS = 8
        ns.CURRICULUM_TAU = 5.0
    return ns


class ModelService:
    """Owns the trained PI-GNN. Thread-safe: training runs in a worker
    thread and hot-swaps the model; predict_risk grabs it under a lock."""

    def __init__(self, G, static_feats: np.ndarray, ckpt_path: Path):
        self.G = G
        self.ckpt_path = Path(ckpt_path)
        self._lock = threading.Lock()
        self.model = None
        self.state = "untrained"          # untrained | training | ready | failed
        self.quality = None
        self.val_f1 = None
        self.error = None
        self._trained_at = None
        self._train_started = None

        # graph tensors are time-invariant; a fresh LiveSimulator-free shim
        # carries just the structural arrays build_graph_tensors needs
        from pignn.simulate import LiveSimulator
        live = LiveSimulator(G, cfg.BASE_STOCK_WEEKS, SEVERITY_THRESHOLDS,
                             seed=cfg.SEED)
        shim = SimpleNamespace(edge_index=live.edge_index,
                               lead_times=live.lead, trans_cap=live.tcap,
                               prod_cap=live.pcap, stor_cap=live.scap)
        self.g = build_graph_tensors(shim, static_feats)
        self.f_dyn = 6

    # ------------------------------------------------------------- lifecycle
    def _new_model(self) -> PIGNN:
        return PIGNN(self.g, f_dyn=self.f_dyn, embed_dim=cfg.NODE_EMBED_DIM,
                     gnn_layers=cfg.GNN_LAYERS, lstm_hidden=cfg.LSTM_HIDDEN,
                     n_horizons=len(cfg.HORIZONS), n_classes=cfg.N_CLASSES)

    def load_checkpoint(self) -> bool:
        if not self.ckpt_path.exists():
            return False
        try:
            payload = torch.load(self.ckpt_path, map_location="cpu",
                                 weights_only=True)
            model = self._new_model()
            model.load_state_dict(payload["state_dict"])
            model.eval()
            with self._lock:
                self.model = model
                self.state = "ready"
                self.quality = payload.get("quality")
                self.val_f1 = payload.get("val_f1")
                self._trained_at = payload.get("trained_at")
            return True
        except Exception as e:  # stale/incompatible checkpoint: retrain
            self.error = f"checkpoint load failed: {e}"
            return False

    def start_training(self, quality: str = "fast") -> bool:
        """Kick off background training; returns False if already training."""
        with self._lock:
            if self.state == "training":
                return False
            self.state = "training"
            self.quality = quality
            self.error = None
            self._train_started = time.time()
        threading.Thread(target=self._train, args=(quality,),
                         daemon=True).start()
        return True

    def _train(self, quality: str):
        try:
            ns = _cfg_namespace(quality)
            sim = simulate(self.G, ns.N_WEEKS, ns.N_DISRUPTION_EPISODES,
                           ns.BASE_STOCK_WEEKS, SEVERITY_THRESHOLDS, ns.SEED)
            train_ds, val_ds, _ = temporal_splits(
                sim, ns.T_IN, ns.HORIZONS, ns.TRAIN_FRAC, ns.VAL_FRAC)
            model, history = train_model(self.g, train_ds, val_ds, ns,
                                         physics=True, verbose=False)
            val_f1 = max((h["val_f1"] for h in history), default=None)
            trained_at = time.time()
            self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "quality": quality,
                        "val_f1": val_f1, "trained_at": trained_at},
                       self.ckpt_path)
            with self._lock:
                self.model = model
                self.state = "ready"
                self.val_f1 = val_f1
                self._trained_at = trained_at
        except Exception as e:
            with self._lock:
                self.state = "failed"
                self.error = str(e)

    def status(self) -> dict:
        with self._lock:
            out = {"state": self.state, "quality": self.quality,
                   "val_f1": self.val_f1, "horizons": list(cfg.HORIZONS),
                   "window_weeks": cfg.T_IN, "error": self.error}
            if self.state == "training" and self._train_started:
                out["training_seconds"] = round(
                    time.time() - self._train_started, 1)
        return out

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_risk(self, dyn_window: np.ndarray):
        """dyn_window: [T_IN, N, 6] -> (risk [H, N], severity [H, N]) or
        None if no trained model is available yet."""
        with self._lock:
            model = self.model
        if model is None or dyn_window.shape[0] < cfg.T_IN:
            return None
        dyn = torch.tensor(dyn_window[None, -cfg.T_IN:],
                           dtype=torch.float32)
        logits, _, _ = model(self.g, dyn)
        probs = torch.softmax(logits, dim=-1)[0]      # [H, N, C]
        risk = probs[..., 2:].sum(-1)                 # P(moderate or major)
        severity = probs.argmax(-1)
        return risk.numpy(), severity.numpy()
