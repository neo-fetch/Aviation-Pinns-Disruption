"""PI-GNN model: spatial message passing + LSTM temporal memory +
multi-horizon severity heads + auxiliary physical-state heads (paper §3.2-3.3).

The physical-state heads (production, inventory, consumption per node; flow
and arrival per edge, all capacity-normalized) exist so the physics losses in
physics.py can penalize predictions that violate conservation of flow,
capacity limits, or lead-time consistency. The baseline model is this same
architecture trained with physics weights set to zero.
"""

import torch
import torch.nn as nn

from .dataset import GraphTensors


class GraphConv(nn.Module):
    """Directed message passing: separate aggregation over suppliers
    (in-neighbors) and customers (out-neighbors), plus a self path."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.w_self = nn.Linear(d_in, d_out)
        self.w_in = nn.Linear(d_in, d_out, bias=False)
        self.w_out = nn.Linear(d_in, d_out, bias=False)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, h, adj_in, adj_out):
        # h: [..., N, d_in]; adjacency matmul broadcasts over leading dims
        m = self.w_self(h) + self.w_in(adj_in @ h) + self.w_out(adj_out @ h)
        return self.norm(self.act(m))


class PIGNN(nn.Module):
    def __init__(self, g: GraphTensors, f_dyn: int, embed_dim: int,
                 gnn_layers: int, lstm_hidden: int, n_horizons: int,
                 n_classes: int):
        super().__init__()
        f_static = g.static.shape[1]
        self.n_horizons = n_horizons
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        d = f_static + f_dyn
        for _ in range(gnn_layers):
            self.convs.append(GraphConv(d, embed_dim))
            d = embed_dim

        # Per-node LSTM over the window, with a graph-level context stream
        # (mean readout) concatenated to each node's input (paper Figs. 2-3).
        self.lstm = nn.LSTM(embed_dim * 2, lstm_hidden, batch_first=True)

        self.cls_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden), nn.ReLU(),
            nn.Linear(lstm_hidden, n_horizons * n_classes))
        # node physical state at t+1: production, inventory, consumption
        self.node_phys_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden), nn.ReLU(),
            nn.Linear(lstm_hidden, 3), nn.Softplus())
        # edge physical state at t+1: flow, arrival (from endpoint states)
        self.edge_phys_head = nn.Sequential(
            nn.Linear(2 * lstm_hidden + 3, lstm_hidden), nn.ReLU(),
            nn.Linear(lstm_hidden, 2), nn.Softplus())

    def forward(self, g: GraphTensors, dyn: torch.Tensor):
        """dyn: [B, T, N, F_dyn] -> logits [B, H, N, C],
        node_phys [B, N, 3], edge_phys [B, E, 2]."""
        B, T, N, _ = dyn.shape
        static = g.static.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        h = torch.cat([static, dyn], dim=-1)
        for conv in self.convs:
            h = conv(h, g.adj_in, g.adj_out)          # [B, T, N, D]

        ctx = h.mean(dim=2, keepdim=True).expand(-1, -1, N, -1)
        seq = torch.cat([h, ctx], dim=-1)             # [B, T, N, 2D]
        seq = seq.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        out, _ = self.lstm(seq)
        h_final = out[:, -1].reshape(B, N, -1)        # [B, N, hidden]

        logits = self.cls_head(h_final).reshape(
            B, N, self.n_horizons, self.n_classes).permute(0, 2, 1, 3)
        node_phys = self.node_phys_head(h_final)      # [B, N, 3]

        src, dst = g.edge_index
        edge_in = torch.cat([h_final[:, src], h_final[:, dst],
                             g.edge_static.unsqueeze(0).expand(B, -1, -1)],
                            dim=-1)
        edge_phys = self.edge_phys_head(edge_in)      # [B, E, 2]
        return logits, node_phys, edge_phys
