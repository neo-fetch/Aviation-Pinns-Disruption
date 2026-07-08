"""Weekly supply chain flow simulator with disruption injection.

Generates temporal snapshots G_t of the Airbus network (paper §3.1): each week
every node produces subject to capacity and material availability, ships to
downstream nodes subject to transport capacity, and receives shipments after
edge lead times. A base-stock ordering policy drives replenishment. The
resulting flows and inventories satisfy conservation of flow *by construction*,
giving the physics losses a consistent ground truth.

Disruption episodes modeled on real aviation events are injected on top of the
nominal dynamics:
  - supplier_outage:   fire/quality escape halts a supplier (e.g. fuselage panels)
  - capacity_cut:      partial capacity loss (e.g. GTF engine inspections)
  - leadtime_spike:    logistics chokepoint (e.g. Suez/Red Sea rerouting)
  - export_restriction: raw material embargo (e.g. titanium sanctions)
  - demand_surge:      FAL ramp-up outpacing supplier capacity

Severity labels per node per week come from the realized capacity reduction:
none / minor / moderate (>=10%) / major (>=30%), matching the paper (§4.1).
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np

DISRUPTION_TYPES = ("supplier_outage", "capacity_cut", "leadtime_spike",
                    "export_restriction", "demand_surge")


@dataclass
class Episode:
    kind: str
    node: int          # epicenter node
    start: int
    duration: int
    magnitude: float   # fraction of capacity lost / lead-time multiplier


@dataclass
class SimResult:
    """Arrays indexed [week, ...]. E is the number of directed edges."""
    node_dyn: np.ndarray        # [T, N, F_dyn] dynamic node features
    edge_flow: np.ndarray       # [T, E] realized flow on each edge
    edge_order: np.ndarray      # [T, E] orders placed on each edge
    edge_arrival: np.ndarray    # [T, E] arrivals (delayed orders)
    production: np.ndarray      # [T, N]
    consumption: np.ndarray     # [T, N] material consumed by production/demand
    inventory: np.ndarray       # [T, N]
    cap_reduction: np.ndarray   # [T, N] realized capacity reduction in [0,1]
    labels: np.ndarray          # [T, N] severity class 0..3
    episodes: list
    edge_index: np.ndarray      # [2, E] (src, dst)
    lead_times: np.ndarray      # [E]
    trans_cap: np.ndarray       # [E]
    prod_cap: np.ndarray        # [N]
    stor_cap: np.ndarray        # [N]

    @property
    def n_weeks(self):
        return self.node_dyn.shape[0]


def _draw_episodes(G, n_weeks, n_episodes, rng):
    """Bias epicenters toward sole-source and high-out-degree nodes."""
    candidates = [n for n, d in G.nodes(data=True) if d["tier"] != "customer"]
    weights = np.array([1.0 + 2.0 * G.nodes[n]["is_sole_source"]
                        + 0.1 * G.out_degree(n) for n in candidates])
    weights /= weights.sum()
    episodes = []
    usable = n_weeks - 22
    for i in range(n_episodes):
        kind = rng.choice(DISRUPTION_TYPES)
        if kind == "export_restriction":
            raws = [n for n in candidates if G.nodes[n]["tier"] == "raw"]
            node = int(rng.choice(raws))
        elif kind == "demand_surge":
            fals = [n for n, d in G.nodes(data=True) if d["tier"] == "fal"]
            node = int(rng.choice(fals))
        else:
            node = int(rng.choice(candidates, p=weights))
        # stratified start times so disruptions occur across the whole
        # horizon (and therefore in the train, val, AND test periods)
        start = 8 + int((i + rng.uniform(0, 1)) / n_episodes * usable)
        duration = int(rng.integers(3, 13))
        magnitude = float(rng.uniform(0.15, 0.95))
        episodes.append(Episode(kind, node, start, duration, magnitude))
    return episodes


def simulate(G: nx.DiGraph, n_weeks: int, n_episodes: int,
             base_stock_weeks: float, severity_thresholds,
             seed: int = 42) -> SimResult:
    rng = np.random.default_rng(seed)
    N = G.number_of_nodes()
    edges = list(G.edges())
    E = len(edges)
    edge_index = np.array(edges, dtype=np.int64).T
    lead = np.array([G.edges[e]["lead_time"] for e in edges], dtype=np.int64)
    tcap = np.array([G.edges[e]["transport_capacity"] for e in edges])
    pcap = np.array([G.nodes[i]["prod_capacity"] for i in range(N)])
    scap = np.array([G.nodes[i]["storage_capacity"] for i in range(N)])
    rel = np.array([G.nodes[i]["reliability"] for i in range(N)])
    tier = np.array([G.nodes[i]["tier_idx"] for i in range(N)])
    is_customer = tier == 5
    is_raw = tier == 0

    in_edges = [[] for _ in range(N)]
    out_edges = [[] for _ in range(N)]
    for e, (u, v) in enumerate(edges):
        out_edges[u].append(e)
        in_edges[v].append(e)

    episodes = _draw_episodes(G, n_weeks, n_episodes, rng)

    # steady-state-ish demand pull: customers draw aircraft from FALs;
    # upstream demand propagates via orders.
    cust_demand_base = {i: pcap[[e_src for e_src in range(N)]].mean() * 0
                        for i in range(N)}  # placeholder, set below
    fal_nodes = np.where(tier == 4)[0]
    fal_rate = pcap[fal_nodes]  # aircraft/week per FAL

    inventory = np.minimum(scap, pcap * base_stock_weeks)
    inventory[is_customer] = 0.0
    max_lead = int(lead.max())
    pipeline = np.zeros((max_lead + 1, E))  # pipeline[k, e]: arrives in k weeks

    node_dyn = np.zeros((n_weeks, N, 6), dtype=np.float32)
    edge_flow = np.zeros((n_weeks, E), dtype=np.float32)
    edge_order = np.zeros((n_weeks, E), dtype=np.float32)
    edge_arrival = np.zeros((n_weeks, E), dtype=np.float32)
    production = np.zeros((n_weeks, N), dtype=np.float32)
    consumption = np.zeros((n_weeks, N), dtype=np.float32)
    inv_hist = np.zeros((n_weeks, N), dtype=np.float32)
    cap_red = np.zeros((n_weeks, N), dtype=np.float32)
    backlog = np.zeros(N)

    for t in range(n_weeks):
        # ---- disruption state this week ---------------------------------
        cap_mult = np.ones(N)
        lead_extra = np.zeros(E, dtype=np.int64)
        demand_mult = np.ones(N)
        for ep in episodes:
            if not (ep.start <= t < ep.start + ep.duration):
                continue
            if ep.kind in ("supplier_outage", "capacity_cut", "export_restriction"):
                loss = ep.magnitude if ep.kind != "capacity_cut" else ep.magnitude * 0.6
                cap_mult[ep.node] = min(cap_mult[ep.node], 1.0 - loss)
            elif ep.kind == "leadtime_spike":
                for e in in_edges[ep.node] + out_edges[ep.node]:
                    lead_extra[e] = max(lead_extra[e],
                                        int(np.ceil(ep.magnitude * 4)))
            elif ep.kind == "demand_surge":
                demand_mult[ep.node] = 1.0 + ep.magnitude

        # random operational noise on capacity (reliability-driven)
        noise = rng.uniform(rel, 1.0, size=N)
        cap_mult = cap_mult * noise

        # ---- arrivals ----------------------------------------------------
        arrivals_e = pipeline[0].copy()
        pipeline[:-1] = pipeline[1:]
        pipeline[-1] = 0.0
        edge_arrival[t] = arrivals_e
        arrivals_n = np.zeros(N)
        for e, (u, v) in enumerate(edges):
            arrivals_n[v] += arrivals_e[e]
        # accept arrivals up to storage; overflow is disposed (scrap/secondary
        # market) and booked as consumption so conservation of flow holds
        room = np.maximum(scap - inventory, 0.0)
        accepted = np.minimum(arrivals_n, room)
        disposal = arrivals_n - accepted
        inventory = inventory + accepted

        # ---- production (needs upstream material for non-raw nodes) ------
        eff_cap = pcap * cap_mult
        prod = np.zeros(N)
        cons = np.zeros(N)
        for i in range(N):
            if is_customer[i]:
                continue
            target = eff_cap[i] * demand_mult[i]
            if is_raw[i]:
                # raw producers curtail to what they can ship or store
                ship_cap = tcap[out_edges[i]].sum() if out_edges[i] else 0.0
                headroom = max(scap[i] - inventory[i], 0.0)
                prod[i] = min(target, ship_cap + headroom)
            else:
                # 1 unit of output consumes 1 unit of pooled input material
                avail = inventory[i]
                prod[i] = min(target, avail)
                cons[i] = prod[i]
                inventory[i] -= cons[i]

        # customers "consume" delivered aircraft (inventory stays at zero so
        # conservation of flow holds at delivery nodes too)
        cons[is_customer] = arrivals_n[is_customer]
        inventory[is_customer] -= cons[is_customer]

        # ---- shipping: allocate output by downstream inventory deficit
        # (pull signal) so persistently starved nodes attract more material,
        # bounded by per-edge transport capacity
        target_inv = np.minimum(pcap * 3.0, scap)
        deficit = np.maximum(target_inv - inventory, 0.05 * pcap + 1.0)
        deficit[is_customer] = pcap[is_customer].mean() + 10.0
        ship_n = prod.copy()
        flows = np.zeros(E)
        for i in range(N):
            if is_customer[i] or not out_edges[i]:
                continue
            outs = out_edges[i]
            caps = tcap[outs]
            dsts = [edges[e][1] for e in outs]
            w = caps * deficit[dsts]
            desired = ship_n[i] * w / max(w.sum(), 1e-9)
            realized = np.minimum(desired, caps)
            flows[outs] = realized
        # goods that could not ship accumulate as producer inventory;
        # anything beyond storage is disposed (booked as consumption) so
        # conservation of flow holds exactly
        shipped = np.zeros(N)
        for e, (u, v) in enumerate(edges):
            shipped[u] += flows[e]
        leftover = np.maximum(prod - shipped, 0.0)
        overflow_out = np.maximum(inventory + leftover - scap, 0.0)
        inventory = inventory + leftover - overflow_out

        # disposal of arrival + storage overflow leaves the node this week
        cons = cons + disposal + overflow_out

        # ---- orders & pipeline insertion ---------------------------------
        for e, (u, v) in enumerate(edges):
            edge_order[t, e] = flows[e]
            k = min(int(lead[e] + lead_extra[e]), max_lead)
            pipeline[k, e] += flows[e]
        edge_flow[t] = flows

        # ---- bookkeeping --------------------------------------------------
        production[t] = prod
        consumption[t] = cons
        inv_hist[t] = inventory
        realized_red = np.zeros(N)
        active = eff_cap > 1e-9
        # realized reduction vs nominal capacity, material shortage included
        realized_red[~is_customer] = 1.0 - np.divide(
            prod[~is_customer], pcap[~is_customer],
            out=np.ones(np.sum(~is_customer)), where=pcap[~is_customer] > 0)
        realized_red = np.clip(realized_red, 0.0, 1.0)
        # remove baseline reliability noise floor so quiet weeks label as none
        baseline = 1.0 - (1.0 + rel) / 2.0
        realized_red = np.maximum(realized_red - baseline, 0.0)
        cap_red[t] = realized_red

        util = np.divide(prod, pcap, out=np.zeros(N), where=pcap > 0)
        days_supply = np.divide(inventory, cons + 1e-6,
                                out=np.full(N, 10.0), where=(cons + 1e-6) > 1e-5)
        backlog = 0.8 * backlog + np.maximum(eff_cap * demand_mult - prod, 0.0)
        node_dyn[t, :, 0] = inventory / np.maximum(scap, 1.0)
        node_dyn[t, :, 1] = util
        node_dyn[t, :, 2] = np.clip(days_supply / 10.0, 0, 2)
        node_dyn[t, :, 3] = backlog / np.maximum(pcap, 1.0)
        node_dyn[t, :, 4] = arrivals_n / np.maximum(pcap, 1.0)
        node_dyn[t, :, 5] = realized_red

    minor, moderate, major = severity_thresholds
    labels = np.zeros((n_weeks, N), dtype=np.int64)
    labels[cap_red >= minor] = 1
    labels[cap_red >= moderate] = 2
    labels[cap_red >= major] = 3

    return SimResult(node_dyn=node_dyn, edge_flow=edge_flow,
                     edge_order=edge_order, edge_arrival=edge_arrival,
                     production=production, consumption=consumption,
                     inventory=inv_hist, cap_reduction=cap_red, labels=labels,
                     episodes=episodes, edge_index=edge_index,
                     lead_times=lead, trans_cap=tcap, prod_cap=pcap,
                     stor_cap=scap)
