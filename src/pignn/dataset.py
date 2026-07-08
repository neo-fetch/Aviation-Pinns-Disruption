"""Windowed spatio-temporal dataset built from simulator output.

Each sample is a sliding window of T_in weekly snapshots (paper §3.3) plus
targets: per-node severity classes at each forecast horizon, and the true
physical state (production, inventory, consumption, edge flows/arrivals) at
the first predicted week, used both as auxiliary regression targets and by the
physics losses.

All physical quantities are normalized by their governing capacity so that a
value of 1.0 means "at capacity" — this puts the physics residuals of every
node and edge on a comparable scale.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .simulate import SimResult


@dataclass
class GraphTensors:
    """Time-invariant graph structure shared across all windows."""
    edge_index: torch.Tensor    # [2, E]
    static: torch.Tensor        # [N, F_static] node features (incl. structural)
    edge_static: torch.Tensor   # [E, 3] lead time, transport cap, cost (norm.)
    lead_times: torch.Tensor    # [E] int
    prod_cap: torch.Tensor      # [N]
    stor_cap: torch.Tensor      # [N]
    trans_cap: torch.Tensor     # [E]
    adj_in: torch.Tensor        # [N, N] row-normalized in-neighbor averaging
    adj_out: torch.Tensor       # [N, N] row-normalized out-neighbor averaging


@dataclass
class WindowBatch:
    dyn: torch.Tensor           # [B, T_in, N, F_dyn]
    labels: torch.Tensor        # [B, H, N] severity class at each horizon
    # ground truth for the first predicted week (t_last + 1), normalized:
    prod_true: torch.Tensor     # [B, N]
    inv_true: torch.Tensor      # [B, N]
    cons_true: torch.Tensor     # [B, N]
    flow_true: torch.Tensor     # [B, E]
    arrival_true: torch.Tensor  # [B, E]
    inv_prev: torch.Tensor      # [B, N] known inventory at t_last (normalized)
    lagged_order: torch.Tensor  # [B, E] order placed at t+1-L_e (normalized)


def build_graph_tensors(sim: SimResult, static_feats: np.ndarray) -> GraphTensors:
    N = static_feats.shape[0]
    E = sim.edge_index.shape[1]
    ei = torch.tensor(sim.edge_index, dtype=torch.long)

    adj_in = torch.zeros(N, N)
    adj_out = torch.zeros(N, N)
    for e in range(E):
        u, v = int(ei[0, e]), int(ei[1, e])
        adj_in[v, u] = 1.0   # v aggregates from suppliers u
        adj_out[u, v] = 1.0  # u aggregates from customers v
    adj_in = adj_in / adj_in.sum(1, keepdim=True).clamp(min=1.0)
    adj_out = adj_out / adj_out.sum(1, keepdim=True).clamp(min=1.0)

    lead = torch.tensor(sim.lead_times, dtype=torch.long)
    tcap = torch.tensor(sim.trans_cap, dtype=torch.float32)
    edge_static = torch.stack([
        lead.float() / lead.float().max(),
        tcap / tcap.max(),
        torch.ones(E) * 0.5,
    ], dim=1)

    return GraphTensors(
        edge_index=ei,
        static=torch.tensor(static_feats, dtype=torch.float32),
        edge_static=edge_static,
        lead_times=lead,
        prod_cap=torch.tensor(sim.prod_cap, dtype=torch.float32),
        stor_cap=torch.tensor(np.minimum(sim.stor_cap, sim.stor_cap[
            sim.stor_cap < 1e8].max() * 4 if (sim.stor_cap < 1e8).any()
            else sim.stor_cap), dtype=torch.float32),
        trans_cap=tcap,
        adj_in=adj_in, adj_out=adj_out,
    )


class WindowDataset:
    def __init__(self, sim: SimResult, t_in: int, horizons, split_range):
        """split_range = (t_start, t_end) over the *last input week* index."""
        self.sim = sim
        self.t_in = t_in
        self.horizons = tuple(horizons)
        max_h = max(horizons)
        lo, hi = split_range
        lo = max(lo, t_in - 1)
        hi = min(hi, sim.n_weeks - 1 - max_h)
        self.t_lasts = list(range(lo, hi + 1))

        # normalization denominators
        self.pcap = np.maximum(sim.prod_cap, 1.0)
        scap = sim.stor_cap.copy()
        finite = scap < 1e8
        scap[~finite] = scap[finite].max() * 4 if finite.any() else 1.0
        self.scap = np.maximum(scap, 1.0)
        self.tcap = np.maximum(sim.trans_cap, 1.0)

    def __len__(self):
        return len(self.t_lasts)

    def get_batch(self, idxs) -> WindowBatch:
        sim, t_in = self.sim, self.t_in
        dyn, labels = [], []
        prod_t, inv_t, cons_t, flow_t, arr_t, inv_p, lag_ord = ([] for _ in range(7))
        for i in idxs:
            tl = self.t_lasts[i]
            tp = tl + 1  # first predicted week
            dyn.append(sim.node_dyn[tl - t_in + 1: tl + 1])
            labels.append(np.stack([sim.labels[tl + h] for h in self.horizons]))
            prod_t.append(sim.production[tp] / self.pcap)
            inv_t.append(sim.inventory[tp] / self.scap)
            cons_t.append(sim.consumption[tp] / self.pcap)
            flow_t.append(sim.edge_flow[tp] / self.tcap)
            arr_t.append(sim.edge_arrival[tp] / self.tcap)
            inv_p.append(sim.inventory[tl] / self.scap)
            # order placed L_e weeks before tp on each edge (0 if before t=0)
            lags = tp - sim.lead_times
            lo_vals = np.where(lags >= 0,
                               sim.edge_order[np.clip(lags, 0, None),
                                              np.arange(len(lags))], 0.0)
            lag_ord.append(lo_vals / self.tcap)

        def t(x, dtype=torch.float32):
            return torch.tensor(np.stack(x), dtype=dtype)

        return WindowBatch(
            dyn=t(dyn), labels=t(labels, torch.long),
            prod_true=t(prod_t), inv_true=t(inv_t), cons_true=t(cons_t),
            flow_true=t(flow_t), arrival_true=t(arr_t),
            inv_prev=t(inv_p), lagged_order=t(lag_ord),
        )


def temporal_splits(sim: SimResult, t_in, horizons, train_frac, val_frac):
    """60/20/20 temporal split over last-input-week indices (paper §4.1)."""
    T = sim.n_weeks
    usable_hi = T - 1 - max(horizons)
    usable_lo = t_in - 1
    span = usable_hi - usable_lo + 1
    tr_hi = usable_lo + int(span * train_frac) - 1
    va_hi = usable_lo + int(span * (train_frac + val_frac)) - 1
    train = WindowDataset(sim, t_in, horizons, (usable_lo, tr_hi))
    val = WindowDataset(sim, t_in, horizons, (tr_hi + 1, va_hi))
    test = WindowDataset(sim, t_in, horizons, (va_hi + 1, usable_hi))
    return train, val, test
