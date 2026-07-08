"""Synthetic Airbus A320-family supply chain network builder.

Builds a directed multi-tier graph whose structure mirrors the real A320
program: raw-material producers feed tier-2 component suppliers, which feed
tier-1 system integrators (engines, aerostructures, landing gear, avionics),
which feed Airbus component plants, which feed the four final assembly lines
(FALs), which deliver to airline customers. Names of external companies are
fictionalised role-alikes; Airbus site names are real program locations.

Every node carries production/storage capacity and every edge carries a lead
time (weeks) and transport capacity, which the simulator and the physics
losses both consume.
"""

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

TIERS = {
    "raw": 0,        # raw materials
    "tier2": 1,      # component suppliers
    "tier1": 2,      # major system suppliers
    "plant": 3,      # Airbus component plants
    "fal": 4,        # final assembly lines
    "customer": 5,   # airline delivery centers
}


@dataclass
class NodeSpec:
    name: str
    tier: str
    region: str
    prod_capacity: float          # units/week the node can produce or process
    storage_capacity: float       # max inventory units
    reliability: float            # baseline reliability score in [0.8, 1.0]
    is_sole_source: bool = False  # single-source choke point flag
    commodities: list = field(default_factory=list)


def _mk(name, tier, region, prod, stor, rel, sole=False):
    return NodeSpec(name, tier, region, prod, stor, rel, sole)


def build_airbus_network(seed: int = 42) -> nx.DiGraph:
    """Return the A320-family supply DiGraph with node/edge attributes set."""
    rng = np.random.default_rng(seed)
    specs = []

    # ---- Tier 3: raw materials -------------------------------------------
    raw_defs = [
        ("Titanium Sponge (Kazakhstan)", "CIS", True),
        ("Titanium Forgings (US)", "NA", False),
        ("Aluminium-Lithium Alloy (EU)", "EU", False),
        ("Aluminium Sheet (EU)", "EU", False),
        ("Carbon Fiber PAN Precursor (JP)", "APAC", True),
        ("Carbon Fiber Prepreg (EU)", "EU", False),
        ("Steel Alloys (EU)", "EU", False),
        ("Semiconductor Foundry (TW)", "APAC", True),
        ("Rare Earth Magnets (CN)", "APAC", True),
        ("Specialty Chemicals (DE)", "EU", False),
        ("Nickel Superalloy (US)", "NA", True),
        ("Glass & Ceramics (FR)", "EU", False),
    ]
    for name, region, sole in raw_defs:
        specs.append(_mk(name, "raw", region,
                         prod=rng.uniform(650, 1000), stor=rng.uniform(1000, 1800),
                         rel=rng.uniform(0.85, 0.98), sole=sole))

    # ---- Tier 2: component suppliers -------------------------------------
    tier2_defs = [
        ("Fastener Systems (FR)", "EU", False),
        ("Hydraulic Actuators (UK)", "EU", False),
        ("Flight Control Actuators (DE)", "EU", False),
        ("Avionics Modules (FR)", "EU", False),
        ("Avionics Modules (US)", "NA", False),
        ("Wiring Harnesses (MA)", "AFR", False),
        ("Wiring Harnesses (TN)", "AFR", False),
        ("Cabin Interiors (DE)", "EU", False),
        ("Seats (UK)", "EU", False),
        ("Machined Titanium Parts (PL)", "EU", False),
        ("Composite Panels (ES)", "EU", False),
        ("Composite Panels (KR)", "APAC", False),
        ("Engine Blades & Discs (UK)", "EU", True),
        ("Engine Control Units (US)", "NA", False),
        ("Fuel Systems (UK)", "EU", False),
        ("APU Components (US)", "NA", True),
        ("Landing Gear Forgings (CA)", "NA", False),
        ("Brakes & Wheels (FR)", "EU", False),
        ("Air Management Systems (DE)", "EU", False),
        ("Electrical Power Systems (US)", "NA", False),
        ("Window & Transparency (US)", "NA", False),
        ("Galleys (NL)", "EU", False),
        ("Bearings (SE)", "EU", False),
        ("Pumps & Valves (IT)", "EU", False),
    ]
    for name, region, sole in tier2_defs:
        specs.append(_mk(name, "tier2", region,
                         prod=rng.uniform(150, 350), stor=rng.uniform(300, 700),
                         rel=rng.uniform(0.85, 0.99), sole=sole))

    # ---- Tier 1: major system suppliers -----------------------------------
    tier1_defs = [
        ("Engine Supplier A - LEAP-class (FR/US)", "EU", False),
        ("Engine Supplier B - GTF-class (US)", "NA", False),
        ("Nacelles & Thrust Reversers (FR)", "EU", True),
        ("Landing Gear Integrator (FR)", "EU", True),
        ("Aerostructures - Fuselage Sections (US)", "NA", True),
        ("Aerostructures - Fuselage Panels (DE)", "EU", False),
        ("Aerostructures - Empennage (ES)", "EU", False),
        ("Avionics Suite Integrator (FR)", "EU", True),
        ("Avionics Suite Integrator (US)", "NA", False),
        ("APU Integrator (US)", "NA", True),
        ("Cabin & Cargo Systems (DE)", "EU", False),
        ("Pylon & Attachments (FR)", "EU", True),
        ("Wing Systems Equipment (UK)", "EU", False),
        ("Flight Control Computers (FR)", "EU", True),
    ]
    for name, region, sole in tier1_defs:
        specs.append(_mk(name, "tier1", region,
                         prod=rng.uniform(60, 120), stor=rng.uniform(120, 250),
                         rel=rng.uniform(0.88, 0.99), sole=sole))

    # ---- Airbus component plants ------------------------------------------
    plant_defs = [
        ("Airbus Hamburg - Fuselage & Cabin", "EU"),
        ("Airbus Saint-Nazaire - Forward/Center Fuselage", "EU"),
        ("Airbus Getafe - Tail & Empennage", "EU"),
        ("Airbus Stade - Composites (VTP)", "EU"),
        ("Airbus Broughton - Wings", "EU"),
        ("Airbus Filton - Wing Design & Equipping", "EU"),
    ]
    for name, region in plant_defs:
        specs.append(_mk(name, "plant", region,
                         prod=rng.uniform(55, 75), stor=rng.uniform(100, 180),
                         rel=rng.uniform(0.93, 0.995)))

    # ---- Final assembly lines ---------------------------------------------
    fal_defs = [
        ("FAL Toulouse", "EU", 16.0),
        ("FAL Hamburg", "EU", 24.0),
        ("FAL Tianjin", "APAC", 6.0),
        ("FAL Mobile", "NA", 8.0),
    ]
    for name, region, rate in fal_defs:
        specs.append(_mk(name, "fal", region,
                         prod=rate, stor=rate * 4, rel=rng.uniform(0.95, 0.995)))

    # ---- Customers ---------------------------------------------------------
    cust_defs = [
        ("Deliveries EU Airlines", "EU"),
        ("Deliveries NA Airlines", "NA"),
        ("Deliveries APAC Airlines", "APAC"),
        ("Deliveries MEA Airlines", "MEA"),
        ("Lessors & Others", "GLOBAL"),
    ]
    for name, region in cust_defs:
        specs.append(_mk(name, "customer", region,
                         prod=0.0, stor=1e9, rel=1.0))

    G = nx.DiGraph()
    for i, s in enumerate(specs):
        G.add_node(i, name=s.name, tier=s.tier, tier_idx=TIERS[s.tier],
                   region=s.region, prod_capacity=float(s.prod_capacity),
                   storage_capacity=float(s.storage_capacity),
                   reliability=float(s.reliability),
                   is_sole_source=bool(s.is_sole_source))

    by_tier = {t: [n for n, d in G.nodes(data=True) if d["tier"] == t]
               for t in TIERS}

    def connect(src_nodes, dst_nodes, out_deg_range, lead_range):
        """Wire each destination to a random subset of sources."""
        for dst in dst_nodes:
            k = int(rng.integers(*out_deg_range))
            k = min(k, len(src_nodes))
            srcs = rng.choice(src_nodes, size=k, replace=False)
            for src in srcs:
                lead = int(rng.integers(*lead_range))
                cap = G.nodes[src]["prod_capacity"] * rng.uniform(0.4, 0.9)
                G.add_edge(int(src), int(dst),
                           lead_time=lead,
                           transport_capacity=float(cap),
                           cost=float(rng.uniform(1.0, 5.0)))

    connect(by_tier["raw"], by_tier["tier2"], (2, 4), (2, 6))
    connect(by_tier["tier2"], by_tier["tier1"], (3, 6), (1, 4))
    connect(by_tier["tier1"], by_tier["plant"], (4, 8), (1, 3))
    connect(by_tier["plant"], by_tier["fal"], (3, 6), (1, 3))

    # Engines and landing gear ship directly to every FAL (buyer-furnished
    # equipment pattern), guaranteeing the classic aviation choke points.
    direct_t1 = [n for n in by_tier["tier1"]
                 if any(k in G.nodes[n]["name"]
                        for k in ("Engine Supplier", "Landing Gear Integrator"))]
    for t1 in direct_t1:
        for fal in by_tier["fal"]:
            if not G.has_edge(t1, fal):
                G.add_edge(t1, fal, lead_time=int(rng.integers(2, 5)),
                           transport_capacity=float(
                               G.nodes[t1]["prod_capacity"] * rng.uniform(0.3, 0.6)),
                           cost=float(rng.uniform(3.0, 8.0)))

    connect(by_tier["fal"], by_tier["customer"], (2, 4), (1, 2))

    # Ensure every non-customer node has at least one outgoing edge and every
    # non-raw node at least one incoming edge, so flow can traverse the graph.
    tier_order = ["raw", "tier2", "tier1", "plant", "fal", "customer"]
    for n, d in G.nodes(data=True):
        ti = tier_order.index(d["tier"])
        if d["tier"] != "customer" and G.out_degree(n) == 0:
            dst = int(rng.choice(by_tier[tier_order[ti + 1]]))
            G.add_edge(n, dst, lead_time=int(rng.integers(1, 4)),
                       transport_capacity=float(d["prod_capacity"] * 0.6),
                       cost=2.0)
        if d["tier"] != "raw" and G.in_degree(n) == 0:
            src = int(rng.choice(by_tier[tier_order[ti - 1]]))
            G.add_edge(src, n, lead_time=int(rng.integers(1, 4)),
                       transport_capacity=float(
                           G.nodes[src]["prod_capacity"] * 0.6),
                       cost=2.0)

    # Feasibility pass: every non-raw, non-customer node must be able to
    # receive at least ~1.3x its production rate, or it is starved by
    # construction rather than by disruption. Scale inbound transport
    # capacities up where needed.
    for v, d in G.nodes(data=True):
        if d["tier"] in ("raw", "customer"):
            continue
        in_e = list(G.in_edges(v))
        s = sum(G.edges[e]["transport_capacity"] for e in in_e)
        need = 1.3 * d["prod_capacity"]
        if s < need and s > 0:
            k = need / s
            for e in in_e:
                G.edges[e]["transport_capacity"] *= k

    return G


def node_static_features(G: nx.DiGraph) -> np.ndarray:
    """Static per-node features: tier one-hot, capacities, degree, choke flags."""
    n = G.number_of_nodes()
    n_tiers = len(TIERS)
    feats = np.zeros((n, n_tiers + 6), dtype=np.float32)
    prod = np.array([G.nodes[i]["prod_capacity"] for i in range(n)])
    stor = np.array([G.nodes[i]["storage_capacity"] for i in range(n)])
    prod_scale = max(prod.max(), 1.0)
    stor_scale = np.percentile(stor[stor < 1e8], 95)
    for i in range(n):
        d = G.nodes[i]
        feats[i, d["tier_idx"]] = 1.0
        feats[i, n_tiers + 0] = d["prod_capacity"] / prod_scale
        feats[i, n_tiers + 1] = min(d["storage_capacity"] / stor_scale, 2.0)
        feats[i, n_tiers + 2] = d["reliability"]
        feats[i, n_tiers + 3] = float(d["is_sole_source"])
        feats[i, n_tiers + 4] = G.in_degree(i) / 10.0
        feats[i, n_tiers + 5] = G.out_degree(i) / 10.0
    return feats
