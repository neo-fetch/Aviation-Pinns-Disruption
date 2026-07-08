"""Dual-framework topology analysis (paper §3.1, Figure 1).

Analytical branch: assortativity, degree distribution, network centralization,
percolation threshold estimate, betweenness-based choke point ranking.

Simulation branch: random vs. targeted node-removal experiments tracking the
largest weakly connected component (LCC), yielding robustness profiles.

Both branches feed the PoC twice: the vulnerability ranking is reported to the
user, and the per-node centrality scores are appended to the model's static
node features as structural priors.
"""

import networkx as nx
import numpy as np


def analytical_measures(G: nx.DiGraph) -> dict:
    UG = G.to_undirected()
    degrees = np.array([d for _, d in G.degree()])
    n = G.number_of_nodes()
    # Freeman centralization over degree
    dmax = degrees.max()
    centralization = float((dmax - degrees).sum() / ((n - 1) * (n - 2)))
    # Molloy-Reed percolation criterion: f_c = 1 - 1/(k2/k - 1)
    k1, k2 = degrees.mean(), (degrees ** 2).mean()
    kappa = k2 / k1
    perc_threshold = float(1.0 - 1.0 / (kappa - 1.0)) if kappa > 2 else 0.0
    return {
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "degree_assortativity": float(
            nx.degree_assortativity_coefficient(UG)),
        "mean_degree": float(k1),
        "max_degree": int(dmax),
        "degree_centralization": centralization,
        "percolation_threshold": perc_threshold,
        "density": float(nx.density(G)),
    }


def vulnerability_ranking(G: nx.DiGraph, top_k: int = 15) -> list:
    """Rank nodes by a composite structural vulnerability score."""
    btw = nx.betweenness_centrality(G)
    clust = nx.clustering(G.to_undirected())
    scores = []
    for i in G.nodes:
        d = G.nodes[i]
        score = (btw[i] * 2.0
                 + 0.3 * float(d["is_sole_source"])
                 + 0.1 * (1.0 - clust[i]))
        scores.append((i, d["name"], d["tier"], round(score, 4),
                       round(btw[i], 4), bool(d["is_sole_source"])))
    scores.sort(key=lambda s: -s[3])
    return scores[:top_k]


def robustness_simulation(G: nx.DiGraph, seed: int = 42,
                          n_random_runs: int = 20) -> dict:
    """LCC fraction vs. fraction of nodes removed, random vs. targeted."""
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    fractions = np.linspace(0.0, 0.5, 11)

    def lcc_frac(H):
        if H.number_of_nodes() == 0:
            return 0.0
        return max(len(c) for c in
                   nx.weakly_connected_components(H)) / n

    # random removal, averaged over runs
    random_profile = np.zeros(len(fractions))
    for _ in range(n_random_runs):
        order = rng.permutation(list(G.nodes))
        for fi, f in enumerate(fractions):
            H = G.copy()
            H.remove_nodes_from(order[:int(f * n)])
            random_profile[fi] += lcc_frac(H)
    random_profile /= n_random_runs

    # targeted removal by descending betweenness
    btw = nx.betweenness_centrality(G)
    order_t = sorted(G.nodes, key=lambda i: -btw[i])
    targeted_profile = []
    for f in fractions:
        H = G.copy()
        H.remove_nodes_from(order_t[:int(f * n)])
        targeted_profile.append(lcc_frac(H))

    return {
        "removal_fractions": fractions.tolist(),
        "lcc_random": random_profile.tolist(),
        "lcc_targeted": list(map(float, targeted_profile)),
    }


def structural_node_features(G: nx.DiGraph) -> np.ndarray:
    """Per-node structural priors appended to model inputs: betweenness,
    pagerank, clustering, core number."""
    n = G.number_of_nodes()
    btw = nx.betweenness_centrality(G)
    pr = nx.pagerank(G)
    clust = nx.clustering(G.to_undirected())
    core = nx.core_number(G.to_undirected())
    core_max = max(core.values()) or 1
    feats = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        feats[i] = [btw[i] * 10.0, pr[i] * n / 5.0, clust[i],
                    core[i] / core_max]
    return feats
