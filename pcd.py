# pcd_pubmed_all_in_one.py
# ------------------------------------------------------------
# Precomputed Cluster Distances (PCD) — single-file reproducible demo on PubMed
# - Partition: k-way Voronoi over graph metric (multi-source Dijkstra)
# - Preprocess: per-cluster "virtual source @ border nodes (0-weight)" Dijkstra
#               => cluster-to-cluster distance table d(S,T)
# - Query: exact Bidirectional Dijkstra + PCD lower-bound pruning (LB >= UB -> prune)
# - Dataset: PubMed (Planetoid) via PyTorch Geometric
# ------------------------------------------------------------

import math
import time
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import networkx as nx

# === (Optional) You need torch + torch_geometric properly installed ===
#    pip install torch torchvision
#    pip install torch_geometric --find-links https://data.pyg.org/whl/torch-<torchver>+<cuda>/html
#    (follow official PyG install docs for your env)
from torch_geometric.datasets import Planetoid


# ----------------------------
# Partitioning (k-way Voronoi)
# ----------------------------
def kway_voronoi_partition(G: nx.Graph, k: int, seed: int = 42) -> Tuple[Dict, List[List]]:
    """
    Partition nodes into k clusters using multi-source Dijkstra (Voronoi cells in graph metric).
    Returns:
      - cid: dict node -> cluster id
      - clusters: list of node lists per cluster
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    if k <= 0 or k > len(nodes):
        raise ValueError("k must be in [1, |V|]")

    centers = rng.sample(nodes, k)

    # Multi-source Dijkstra labeled by center id
    dist = {u: math.inf for u in nodes}
    cid = {u: None for u in nodes}
    pq: List[Tuple[float, int, int]] = []  # (dist, center_id, node)
    import heapq
    for i, c in enumerate(centers):
        dist[c] = 0.0
        cid[c] = i
        heapq.heappush(pq, (0.0, i, c))

    while pq:
        d, i, u = heapq.heappop(pq)
        if d != dist[u] or cid[u] != i:
            continue
        for v, edata in G[u].items():
            w = edata.get("weight", 1.0)
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                cid[v] = i
                heapq.heappush(pq, (nd, i, v))

    clusters = [[] for _ in range(k)]
    for u in nodes:
        clusters[cid[u]].append(u)
    return cid, clusters


def compute_cluster_borders(G: nx.Graph, cid: Dict) -> List[Set]:
    """Return a list of border-node sets per cluster."""
    k = max(cid.values()) + 1
    B = [set() for _ in range(k)]
    for u in G.nodes():
        cu = cid[u]
        for v in G[u]:
            if cid[v] != cu:
                B[cu].add(u)
                break
    return B


# -----------------------------------
# Precompute cluster-to-cluster table
# -----------------------------------
def dijkstra_from_virtual_border(G: nx.Graph, border_set: Set, weight: str = "weight") -> Dict:
    """
    Dijkstra from a virtual source connected with 0-weight edges to all nodes in border_set.
    Returns distances to all nodes.
    """
    import heapq
    dist = {u: math.inf for u in G.nodes()}
    pq: List[Tuple[float, int]] = []
    for u in border_set:
        dist[u] = 0.0
        heapq.heappush(pq, (0.0, u))

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, edata in G[u].items():
            w = edata.get(weight, 1.0)
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def preprocess_cluster_distances(
    G: nx.Graph, cid: Dict, clusters: List[List]
) -> Tuple[List[List[float]], List[Set]]:
    """
    For each cluster S, compute d(S,T) for all T by one Dijkstra from S's border nodes.
    Returns:
      - d_ST: k x k matrix of cluster distances
      - cluster_border: list of border-node sets per cluster
    """
    k = len(clusters)
    cluster_border = compute_cluster_borders(G, cid)

    d_ST = [[math.inf] * k for _ in range(k)]
    for i in range(k):
        dist = dijkstra_from_virtual_border(G, cluster_border[i])
        # per-target-cluster minima
        per_cluster_min = [math.inf] * k
        for v, dv in dist.items():
            j = cid[v]
            if dv < per_cluster_min[j]:
                per_cluster_min[j] = dv
        for j in range(k):
            d_ST[i][j] = per_cluster_min[j]
        d_ST[i][i] = 0.0
    return d_ST, cluster_border


# ---------------------------------------------
# Bidirectional Dijkstra with PCD-style pruning
# ---------------------------------------------
def bidij_pcd(
    G: nx.Graph, s, t, cid: Dict, d_ST: List[List[float]]
) -> Tuple[float, int, int, int, int]:
    """
    Exact bidirectional Dijkstra using PCD lower bounds for pruning.
    UB (mu) via meeting rule; LB in forward = dF[u] + d(cluster(u), cluster(t)),
    symmetrically for backward.
    Returns (dist, settled_forward, settled_backward, pops_forward, pops_backward).
    """
    if s == t:
        return (0.0, 0, 0, 0, 0)

    import heapq

    cf = cid[s]
    ct = cid[t]

    dF = defaultdict(lambda: math.inf); dF[s] = 0.0
    dB = defaultdict(lambda: math.inf); dB[t] = 0.0
    pqF: List[Tuple[float, int]] = [(0.0, s)]
    pqB: List[Tuple[float, int]] = [(0.0, t)]
    settledF, settledB = set(), set()

    mu = math.inf
    popsF = popsB = 0

    while pqF or pqB:
        # expand side with smaller frontier key
        side = "F"
        if pqB and (not pqF or pqB[0][0] < pqF[0][0]):
            side = "B"

        if side == "F":
            d, u = heapq.heappop(pqF); popsF += 1
            if u in settledF:
                continue
            # LB pruning
            lb = d + d_ST[cid[u]][ct]
            if lb >= mu:
                settledF.add(u)
                continue

            settledF.add(u)
            if u in settledB:
                mu = min(mu, d + dB[u])

            for v, edata in G[u].items():
                w = edata.get("weight", 1.0)
                nd = d + w
                if nd < dF[v]:
                    dF[v] = nd
                    heapq.heappush(pqF, (nd, v))

        else:
            d, u = heapq.heappop(pqB); popsB += 1
            if u in settledB:
                continue
            lb = d + d_ST[cid[u]][cf]
            if lb >= mu:
                settledB.add(u)
                continue

            settledB.add(u)
            if u in settledF:
                mu = min(mu, d + dF[u])

            for v, edata in G[u].items():
                w = edata.get("weight", 1.0)
                nd = d + w
                if nd < dB[v]:
                    dB[v] = nd
                    heapq.heappush(pqB, (nd, v))

        minkeyF = pqF[0][0] if pqF else math.inf
        minkeyB = pqB[0][0] if pqB else math.inf
        if min(minkeyF, minkeyB) >= mu:
            break

    return mu, len(settledF), len(settledB), popsF, popsB


# ------------------------
# Baseline bi-Dijkstra (no pruning)
# ------------------------
def bidij_baseline(G: nx.Graph, s, t) -> Tuple[float, int, int]:
    if s == t:
        return 0.0, 0, 0

    import heapq
    dF = defaultdict(lambda: math.inf); dF[s] = 0.0
    dB = defaultdict(lambda: math.inf); dB[t] = 0.0
    pqF: List[Tuple[float, int]] = [(0.0, s)]
    pqB: List[Tuple[float, int]] = [(0.0, t)]
    settledF, settledB = set(), set()
    mu = math.inf

    while pqF or pqB:
        side = "F"
        if pqB and (not pqF or pqB[0][0] < pqF[0][0]):
            side = "B"

        if side == "F":
            d, u = heapq.heappop(pqF)
            if u in settledF:
                continue
            settledF.add(u)
            if u in settledB:
                mu = min(mu, d + dB[u])
            for v, edata in G[u].items():
                w = edata.get("weight", 1.0)
                nd = d + w
                if nd < dF[v]:
                    dF[v] = nd
                    heapq.heappush(pqF, (nd, v))
        else:
            d, u = heapq.heappop(pqB)
            if u in settledB:
                continue
            settledB.add(u)
            if u in settledF:
                mu = min(mu, d + dF[u])
            for v, edata in G[u].items():
                w = edata.get("weight", 1.0)
                nd = d + w
                if nd < dB[v]:
                    dB[v] = nd
                    heapq.heappush(pqB, (nd, v))

        minkeyF = pqF[0][0] if pqF else math.inf
        minkeyB = pqB[0][0] if pqB else math.inf
        if min(minkeyF, minkeyB) >= mu:
            break

    return mu, len(settledF), len(settledB)


# ------------------------
# Dataset: PubMed (Planetoid)
# ------------------------
def load_pubmed_graph() -> nx.Graph:
    dataset = Planetoid(root="data/Planetoid", name="PubMed")
    data = dataset[0]
    # Convert to undirected NX graph with unit weights
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edge_index = data.edge_index.t().tolist()
    for u, v in edge_index:
        if u != v:
            G.add_edge(int(u), int(v), weight=1.0)

    # Drop isolates (rare), relabel to 0..n-1
    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)
        G = nx.convert_node_labels_to_integers(G)
    return G


# ------------------------
# Main & CLI
# ------------------------
def run_pubmed_experiment(k: int, Q: int, seed: int):
    print("Loading PubMed (Planetoid)...")
    G = load_pubmed_graph()
    n, m = G.number_of_nodes(), G.number_of_edges()
    print(f"Graph: PubMed |V|={n:,}, |E|={m:,}")

    if k <= 0:
        k = max(32, int(math.sqrt(n)))  # heuristic default

    print(f"Partitioning into k={k} clusters (graph-Voronoi, multi-source Dijkstra)...")
    t0 = time.time()
    cid, clusters = kway_voronoi_partition(G, k=k, seed=seed)
    t1 = time.time()
    print(f"  Partition done in {t1 - t0:.2f}s")

    print("Precomputing cluster-to-cluster distances (virtual border source)...")
    t0 = time.time()
    d_ST, cluster_border = preprocess_cluster_distances(G, cid, clusters)
    t1 = time.time()
    print(f"  Preprocessing done in {t1 - t0:.2f}s")

    # Queries
    print(f"Running {Q} random queries (baseline vs PCD)...")
    rng = random.Random(seed)
    nodes = list(G.nodes())
    rows = []
    have_path = 0
    for qi in range(Q):
        # try to sample a connected pair (PubMed mostly a big CC)
        for _ in range(200):
            s, t = rng.sample(nodes, 2)
            if nx.has_path(G, s, t):
                have_path += 1
                break
        else:
            continue

        # Baseline
        t0 = time.time()
        d0, sf0, sb0 = bidij_baseline(G, s, t)
        t1 = time.time()
        t_base = t1 - t0

        # PCD
        t0 = time.time()
        d1, sf1, sb1, pf, pb = bidij_pcd(G, s, t, cid, d_ST)
        t1 = time.time()
        t_pcd = t1 - t0

        assert abs(d0 - d1) < 1e-8, f"Distance mismatch: {d0} vs {d1}"
        settled0 = sf0 + sb0
        settled1 = sf1 + sb1
        speed_nodes = settled0 / (settled1 + 1e-12)
        speed_time = t_base / (t_pcd + 1e-12)

        rows.append((s, t, d0, settled0, settled1, speed_nodes, t_base, t_pcd, speed_time))

        if (qi + 1) % max(1, Q // 10) == 0:
            print(f"  [{qi+1:3d}/{Q}] dist={d0:.4f}  "
                  f"settled(base)={settled0:7d}  settled(PCD)={settled1:7d}  "
                  f"node-speedup={speed_nodes:6.2f}x  time-speedup={speed_time:6.2f}x")

    if not rows:
        print("No successful queries (unexpected).")
        return

    avg_node_speed = sum(r[5] for r in rows) / len(rows)
    avg_time_speed = sum(r[8] for r in rows) / len(rows)
    avg_settle_base = sum(r[3] for r in rows) / len(rows)
    avg_settle_pcd  = sum(r[4] for r in rows) / len(rows)

    print("\n=== Summary (PubMed) ===")
    print(f"Queries attempted: {Q}, connected pairs: {have_path}")
    print(f"Avg settled nodes (baseline): {avg_settle_base:,.1f}")
    print(f"Avg settled nodes (PCD)    : {avg_settle_pcd:,.1f}")
    print(f"Avg node-settle speedup    : {avg_node_speed:.2f}x")
    print(f"Avg time speedup (rough)   : {avg_time_speed:.2f}x")
    print("Done.")


def parse_args():
    ap = argparse.ArgumentParser(description="PCD all-in-one demo on PubMed (Planetoid)")
    ap.add_argument("--k", type=int, default=0, help="#clusters (default ~ sqrt(n))")
    ap.add_argument("--Q", type=int, default=100, help="#queries to test")
    ap.add_argument("--seed", type=int, default=7, help="random seed")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pubmed_experiment(k=args.k, Q=args.Q, seed=args.seed)
