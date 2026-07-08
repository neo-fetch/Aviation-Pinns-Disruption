"""Differentiable physics constraints for supply chain predictions (paper §3.4).

All quantities are capacity-normalized (1.0 = at capacity), so residuals from
different nodes/edges are commensurable. The three constraint families:

  L_flow: conservation of flow at every node —
      sum_in arrival + production = sum_out flow + consumption + dInventory
  L_capacity: production <= 1, inventory <= 1, flow <= 1 (normalized caps),
      penalized with squared hinge max(0, x - 1)^2
  L_lead: arrivals on an edge must match the order placed lead_time weeks
      earlier (known from history), squared error

Curriculum weighting (paper §3.4): lambda(e) = lambda_final*(1 - exp(-e/tau))
after a prediction-only warm-up.
"""

import math

import torch

from .dataset import GraphTensors, WindowBatch


def physics_residuals(g: GraphTensors, batch: WindowBatch,
                      node_phys: torch.Tensor, edge_phys: torch.Tensor):
    """Return per-term mean residual losses (flow, capacity, lead).

    node_phys: [B, N, 3] -> production, inventory, consumption (normalized)
    edge_phys: [B, E, 2] -> flow, arrival (normalized by transport capacity)
    """
    B, N, _ = node_phys.shape
    prod, inv, cons = node_phys.unbind(dim=-1)
    flow, arrival = edge_phys.unbind(dim=-1)
    src, dst = g.edge_index

    # de-normalize edge quantities to material units, then express node
    # balance in units of each node's production capacity
    flow_u = flow * g.trans_cap
    arr_u = arrival * g.trans_cap
    inflow = torch.zeros(B, N).index_add_(1, dst, arr_u)
    outflow = torch.zeros(B, N).index_add_(1, src, flow_u)

    pcap = g.prod_cap.clamp(min=1.0)
    scap_u = g.stor_cap.clamp(min=1.0)
    d_inv = (inv - batch.inv_prev) * scap_u
    # normalize the balance by total material throughput capacity of the node
    # so residuals from small suppliers and large hubs are commensurable
    in_cap = torch.zeros_like(g.prod_cap).index_add_(0, dst, g.trans_cap)
    node_scale = (pcap + in_cap).clamp(min=1.0)
    residual = (inflow + prod * pcap - outflow - cons * pcap - d_inv) / node_scale
    l_flow = residual.pow(2).mean()

    l_capacity = (torch.relu(prod - 1.0).pow(2).mean()
                  + torch.relu(inv - 1.0).pow(2).mean()
                  + torch.relu(flow - 1.0).pow(2).mean())

    l_lead = (arrival - batch.lagged_order).pow(2).mean()
    return l_flow, l_capacity, l_lead


def curriculum_lambda(epoch: int, warmup: int, tau: float,
                      lam_final: float) -> float:
    if epoch < warmup:
        return 0.0
    return lam_final * (1.0 - math.exp(-(epoch - warmup) / tau))


@torch.no_grad()
def violation_metrics(g: GraphTensors, batch: WindowBatch,
                      node_phys: torch.Tensor, edge_phys: torch.Tensor) -> dict:
    """Physics-violation diagnostics for evaluation (not training)."""
    l_flow, l_cap, l_lead = physics_residuals(g, batch, node_phys, edge_phys)
    prod, inv, _ = node_phys.unbind(dim=-1)
    flow, _ = edge_phys.unbind(dim=-1)
    return {
        "flow_residual_mse": float(l_flow),
        "capacity_penalty": float(l_cap),
        "leadtime_mse": float(l_lead),
        "pct_capacity_violations": float(
            ((prod > 1.0).float().mean() + (inv > 1.0).float().mean()
             + (flow > 1.0).float().mean()) / 3.0 * 100.0),
    }
