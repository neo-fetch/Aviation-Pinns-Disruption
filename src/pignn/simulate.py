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

Two entry points share the same dynamics:
  - simulate():     batch mode — pre-drawn episodes, fixed horizon, SimResult
  - LiveSimulator:  stepping mode — advance one week at a time and inject
                    episodes at runtime (drives the interactive web backend)
"""

import math
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
    pre_weeks: int = 5   # precursor ramp: warning signals build up gradually
    recovery_weeks: int = 3  # exponential recovery tail after the episode

    def intensity(self, t: int) -> float:
        """Fraction of full magnitude active at week t. Disruptions announce
        themselves through a precursor ramp (the paper's premise that warning
        signals accumulate over weeks) and fade through a recovery tail."""
        if self.start <= t < self.start + self.duration:
            return 1.0
        if self.start - self.pre_weeks <= t < self.start:
            return 0.35 * (t - (self.start - self.pre_weeks)) / self.pre_weeks
        end = self.start + self.duration
        if end <= t < end + self.recovery_weeks * 3:
            return math.exp(-(t - end) / self.recovery_weeks)
        return 0.0


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


class LiveSimulator:
    """Stepping version of the simulator: same dynamics as simulate(), one
    week per step() call, with episodes injectable at any time. Keeps full
    history so a SimResult can be materialized at any point."""

    def __init__(self, G: nx.DiGraph, base_stock_weeks: float,
                 severity_thresholds, seed: int = 42, rng=None,
                 episodes=None):
        self.G = G
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.severity_thresholds = tuple(severity_thresholds)
        self.episodes = list(episodes) if episodes is not None else []

        N = G.number_of_nodes()
        self.N = N
        self.edges = list(G.edges())
        self.E = len(self.edges)
        self.edge_index = np.array(self.edges, dtype=np.int64).T
        self.lead = np.array([G.edges[e]["lead_time"] for e in self.edges],
                             dtype=np.int64)
        self.tcap = np.array([G.edges[e]["transport_capacity"]
                              for e in self.edges])
        self.pcap = np.array([G.nodes[i]["prod_capacity"] for i in range(N)])
        self.scap = np.array([G.nodes[i]["storage_capacity"] for i in range(N)])
        self.rel = np.array([G.nodes[i]["reliability"] for i in range(N)])
        tier = np.array([G.nodes[i]["tier_idx"] for i in range(N)])
        self.is_customer = tier == 5
        self.is_raw = tier == 0

        self.in_edges = [[] for _ in range(N)]
        self.out_edges = [[] for _ in range(N)]
        for e, (u, v) in enumerate(self.edges):
            self.out_edges[u].append(e)
            self.in_edges[v].append(e)

        # mutable dynamic state
        self.inventory = np.minimum(self.scap, self.pcap * base_stock_weeks)
        self.inventory[self.is_customer] = 0.0
        self.max_lead = int(self.lead.max())
        # pipeline[k, e]: material on edge e arriving in k weeks
        self.pipeline = np.zeros((self.max_lead + 1, self.E))
        self.backlog = np.zeros(N)
        self.t = 0

        # per-week history (lists of arrays, stacked by result())
        self.hist_node_dyn = []
        self.hist_edge_flow = []
        self.hist_edge_order = []
        self.hist_edge_arrival = []
        self.hist_production = []
        self.hist_consumption = []
        self.hist_inventory = []
        self.hist_cap_red = []
        self.hist_labels = []

    # -------------------------------------------------------------- control
    def inject(self, kind: str, node: int, magnitude: float, duration: int,
               start_offset: int = 0, pre_weeks: int = 5,
               recovery_weeks: int = 3) -> Episode:
        """Schedule a disruption episode at runtime. start_offset=0 means the
        episode's full-intensity phase begins this coming week (its precursor
        ramp is skipped for offsets shorter than pre_weeks)."""
        if kind not in DISRUPTION_TYPES:
            raise ValueError(f"unknown disruption kind: {kind}")
        ep = Episode(kind, int(node), self.t + int(start_offset),
                     int(duration), float(magnitude),
                     pre_weeks=pre_weeks, recovery_weeks=recovery_weeks)
        self.episodes.append(ep)
        return ep

    def active_episodes(self, t=None):
        """Episodes with nonzero intensity at week t (default: current)."""
        t = self.t if t is None else t
        out = []
        for i, ep in enumerate(self.episodes):
            inten = ep.intensity(t)
            if inten > 0.0:
                out.append((i, ep, inten))
        return out

    # ----------------------------------------------------------------- step
    def step(self) -> dict:
        """Advance one week and return this week's snapshot arrays."""
        t = self.t
        N, E, edges = self.N, self.E, self.edges
        pcap, scap, tcap, lead = self.pcap, self.scap, self.tcap, self.lead
        is_customer, is_raw = self.is_customer, self.is_raw
        rng = self.rng
        inventory = self.inventory

        # ---- disruption state this week ---------------------------------
        cap_mult = np.ones(N)
        lead_extra = np.zeros(E, dtype=np.int64)
        demand_mult = np.ones(N)
        for ep in self.episodes:
            inten = ep.intensity(t)
            if inten <= 0.0:
                continue
            if ep.kind in ("supplier_outage", "capacity_cut", "export_restriction"):
                loss = ep.magnitude if ep.kind != "capacity_cut" else ep.magnitude * 0.6
                cap_mult[ep.node] = min(cap_mult[ep.node], 1.0 - loss * inten)
            elif ep.kind == "leadtime_spike":
                for e in self.in_edges[ep.node] + self.out_edges[ep.node]:
                    lead_extra[e] = max(lead_extra[e],
                                        int(np.ceil(ep.magnitude * 4 * inten)))
            elif ep.kind == "demand_surge":
                demand_mult[ep.node] = 1.0 + ep.magnitude * inten

        # random operational noise on capacity (reliability-driven)
        noise = rng.uniform(self.rel, 1.0, size=N)
        cap_mult = cap_mult * noise

        # ---- arrivals ----------------------------------------------------
        arrivals_e = self.pipeline[0].copy()
        self.pipeline[:-1] = self.pipeline[1:]
        self.pipeline[-1] = 0.0
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
                ship_cap = tcap[self.out_edges[i]].sum() if self.out_edges[i] else 0.0
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
            if is_customer[i] or not self.out_edges[i]:
                continue
            outs = self.out_edges[i]
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
        orders = np.zeros(E, dtype=np.float32)
        for e, (u, v) in enumerate(edges):
            orders[e] = flows[e]
            k = min(int(lead[e] + lead_extra[e]), self.max_lead)
            self.pipeline[k, e] += flows[e]

        # ---- bookkeeping --------------------------------------------------
        realized_red = np.zeros(N)
        # realized reduction vs nominal capacity, material shortage included
        realized_red[~is_customer] = 1.0 - np.divide(
            prod[~is_customer], pcap[~is_customer],
            out=np.ones(np.sum(~is_customer)), where=pcap[~is_customer] > 0)
        realized_red = np.clip(realized_red, 0.0, 1.0)
        # remove baseline reliability noise floor so quiet weeks label as none
        baseline = 1.0 - (1.0 + self.rel) / 2.0
        realized_red = np.maximum(realized_red - baseline, 0.0)

        util = np.divide(prod, pcap, out=np.zeros(N), where=pcap > 0)
        days_supply = np.divide(inventory, cons + 1e-6,
                                out=np.full(N, 10.0), where=(cons + 1e-6) > 1e-5)
        self.backlog = 0.8 * self.backlog + np.maximum(
            eff_cap * demand_mult - prod, 0.0)
        dyn = np.zeros((N, 6), dtype=np.float32)
        dyn[:, 0] = inventory / np.maximum(scap, 1.0)
        dyn[:, 1] = util
        dyn[:, 2] = np.clip(days_supply / 10.0, 0, 2)
        dyn[:, 3] = self.backlog / np.maximum(pcap, 1.0)
        dyn[:, 4] = arrivals_n / np.maximum(pcap, 1.0)
        dyn[:, 5] = realized_red

        minor, moderate, major = self.severity_thresholds
        labels = np.zeros(N, dtype=np.int64)
        labels[realized_red >= minor] = 1
        labels[realized_red >= moderate] = 2
        labels[realized_red >= major] = 3

        self.inventory = inventory
        self.hist_node_dyn.append(dyn)
        self.hist_edge_flow.append(flows.astype(np.float32))
        self.hist_edge_order.append(orders)
        self.hist_edge_arrival.append(arrivals_e.astype(np.float32))
        self.hist_production.append(prod.astype(np.float32))
        self.hist_consumption.append(cons.astype(np.float32))
        self.hist_inventory.append(inventory.astype(np.float32))
        self.hist_cap_red.append(realized_red.astype(np.float32))
        self.hist_labels.append(labels)
        self.t = t + 1

        return {
            "week": t,
            "node_dyn": dyn,
            "production": prod,
            "consumption": cons,
            "inventory": inventory.copy(),
            "cap_reduction": realized_red,
            "labels": labels,
            "edge_flow": flows,
            "edge_arrival": arrivals_e,
            "active_episodes": self.active_episodes(t),
        }

    # --------------------------------------------------------------- export
    def result(self) -> SimResult:
        """Materialize the full history as a SimResult (batch layout)."""
        return SimResult(
            node_dyn=np.stack(self.hist_node_dyn),
            edge_flow=np.stack(self.hist_edge_flow),
            edge_order=np.stack(self.hist_edge_order),
            edge_arrival=np.stack(self.hist_edge_arrival),
            production=np.stack(self.hist_production),
            consumption=np.stack(self.hist_consumption),
            inventory=np.stack(self.hist_inventory),
            cap_reduction=np.stack(self.hist_cap_red),
            labels=np.stack(self.hist_labels),
            episodes=self.episodes,
            edge_index=self.edge_index,
            lead_times=self.lead,
            trans_cap=self.tcap,
            prod_cap=self.pcap,
            stor_cap=self.scap,
        )


def simulate(G: nx.DiGraph, n_weeks: int, n_episodes: int,
             base_stock_weeks: float, severity_thresholds,
             seed: int = 42) -> SimResult:
    rng = np.random.default_rng(seed)
    episodes = _draw_episodes(G, n_weeks, n_episodes, rng)
    sim = LiveSimulator(G, base_stock_weeks, severity_thresholds,
                        rng=rng, episodes=episodes)
    for _ in range(n_weeks):
        sim.step()
    return sim.result()
