# compare_partition_e.py
# ------------------------------------------------------------
# E-method pipeline with switchable partitioners: Leiden vs METIS.
# - Graph model: entity + relation nodes; each triple -> ("REL", i, r) with 0.5 edges
# - Partitioner: "leiden" uses graspologic.hierarchical_leiden if present, otherwise a fallback
#                "metis" uses nxmetis if present, otherwise a KL-bisection fallback
# - Evaluation: entity→entity pairs, reusing the same ground truth for both runs
# - Outputs: two rows (Leiden, METIS) with query_time, exact matches, MAE, preprocessing_time, and #clusters
# - Extra: enumerate connected-components in every cluster (print + optional CSV)
# ------------------------------------------------------------

import argparse
import os
import time
import random
import heapq
from collections import defaultdict, Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import networkx as nx
import pandas as pd

# ---------- optional deps ----------
try:
    from graspologic.partition import hierarchical_leiden
    HAVE_HL = True
except Exception:
    hierarchical_leiden = None
    HAVE_HL = False

try:
    import nxmetis
    HAVE_NXMETIS = True
except Exception:
    nxmetis = None
    HAVE_NXMETIS = False


# ---------- helpers ----------
def is_rel(n) -> bool:
    return isinstance(n, tuple) and len(n) > 0 and n[0] == "REL"


def node_sort_key(n):
    if is_rel(n):
        tag, idx, rel = n
        return (1, int(idx), str(rel))
    else:
        return (0, 0, str(n))


# ---------- toy graph (optional) ----------
def make_sbm_toy(n_blocks=4, block_size=80, p_in=0.06, p_out=0.005, seed=42):
    rng = np.random.default_rng(seed)
    sizes = [block_size] * n_blocks
    p = np.full((n_blocks, n_blocks), p_out)
    for i in range(n_blocks):
        p[i, i] = p_in
    G = nx.stochastic_block_model(sizes, p, seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return G


# ---------- KG loaders ----------
def load_wn18_graph_relation_nodes(path: str):
    """
    构图：每条 triple (h,t,r) 生成一个关系节点 ("REL", i, r)
    h --0.5-- ("REL", i, r) --0.5-- t
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"WN18 file not found: {path}")
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        try:
            _ = int(first.strip())
        except Exception:
            parts = first.strip().split()
            if len(parts) >= 2:
                h, t = parts[0], parts[1]
                r = parts[2] if len(parts) >= 3 else "NA"
                if h != t:
                    triples.append((h, t, r))
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            h, t = parts[0], parts[1]
            r = parts[2] if len(parts) >= 3 else "NA"
            if h == t:
                continue
            triples.append((h, t, r))

    G = nx.Graph()
    entities = set()
    for h, t, _ in triples:
        entities.add(h)
        entities.add(t)
    G.add_nodes_from(entities)
    for i, (h, t, r) in enumerate(triples):
        rel_node = ("REL", i, r)
        G.add_node(rel_node)
        G.add_edge(h, rel_node, weight=0.5)
        G.add_edge(rel_node, t, weight=0.5)
    return G


# ---------- PLL ----------
class PrunedPLLIndex:
    def __init__(self, G: nx.Graph | None = None, order=None):
        self.G = G
        self.labels = {} if G is None else {v: {} for v in G.nodes()}
        self.order = list(order) if order is not None else (list(G.nodes()) if G is not None else [])

    @classmethod
    def from_labels(cls, labels: dict):
        obj = cls(G=None, order=None)
        obj.labels = labels
        return obj

    def _pruned_dijkstra(self, root):
        dist = {root: 0.0}
        heap = [(0.0, 0, root)]
        counter = 0
        while heap:
            d, _, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if self.query(root, u) <= d:
                continue
            self.labels[u][root] = d
            for v, data in self.G[u].items():
                w = float(data.get("weight", 1.0))
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    counter += 1
                    heapq.heappush(heap, (nd, counter, v))

    def build(self):
        assert self.G is not None
        # heuristic: order by eccentricity if possible
        try:
            ecc = {}
            for comp in nx.connected_components(self.G):
                cg = self.G.subgraph(comp)
                ecc.update(nx.eccentricity(cg))
            self.order = sorted(self.G.nodes(), key=lambda n: ecc.get(n, 0))
        except Exception:
            self.order = list(self.G.nodes())
        for v in self.order:
            self._pruned_dijkstra(v)

    def query(self, u, v):
        best = float("inf")
        lu = self.labels.get(u, {})
        lv = self.labels.get(v, {})
        if len(lu) > len(lv):
            lu, lv = lv, lu
        for lm, du in lu.items():
            dv = lv.get(lm)
            if dv is not None:
                s = du + dv
                if s < best:
                    best = s
        return best


# ---------- CC enumeration per cluster ----------
def enumerate_cluster_cc(G: nx.Graph, clusters: dict[int, list]) -> pd.DataFrame:
    rows = []
    for cid, nodes in clusters.items():
        if not nodes:
            rows.append((cid, 0, 0))
            continue
        sub = G.subgraph(nodes)
        try:
            k = nx.number_connected_components(sub)
        except Exception:
            # 兜底
            k = len(list(nx.connected_components(sub)))
        rows.append((cid, len(nodes), int(k)))
    df = pd.DataFrame(rows, columns=["cluster_id", "size", "num_connected_components"])
    # 多连通分量与大簇优先显示
    return df.sort_values(
        ["num_connected_components", "size"], ascending=[False, False]
    ).reset_index(drop=True)


# ---------- partitioners ----------
def final_partition_from_hl(hl_result, G: nx.Graph) -> dict:
    node_to_lc = {}
    for h in hl_result:
        if getattr(h, "is_final_cluster", False):
            node_to_lc[h.node] = (h.level, h.cluster)
    for n in G.nodes():
        if n not in node_to_lc:
            node_to_lc[n] = (-1, f"iso_{n}")
    lc2cid, next_id = {}, 0
    part = {}
    for n, lc in node_to_lc.items():
        if lc not in lc2cid:
            lc2cid[lc] = next_id
            next_id += 1
        part[n] = lc2cid[lc]
    assert len(part) == G.number_of_nodes()
    return part


def _kl_bisect_nodes(G_sub, nodes):
    # fallback bisection using Kernighan–Lin (works without extra deps)
    if len(nodes) <= 2:
        mid = len(nodes) // 2
        return set(nodes[:mid]), set(nodes[mid:])
    H = G_sub.subgraph(nodes).copy()
    # ensure connectedness doesn't block partition; if it fails, split by order
    try:
        A, B = nx.algorithms.community.kernighan_lin_bisection(H, weight="weight")
        return set(A), set(B)
    except Exception:
        mid = len(nodes) // 2
        return set(nodes[:mid]), set(nodes[mid:])


def _nxmetis_bisect_nodes(G_sub, nodes):
    # returns two node sets using nxmetis bisection
    H = G_sub.subgraph(nodes).copy()
    try:
        _, parts = nxmetis.partition(H, 2, recursive=False)  # parts: [listA, listB]
        return set(parts[0]), set(parts[1])
    except Exception:
        # fallback to KL
        return _kl_bisect_nodes(G_sub, nodes)


def hierarchical_metis_partition(G: nx.Graph, max_cluster_size: int, prefer_nxmetis=True):
    """Recursive bisection until each cluster size <= max_cluster_size"""
    clusters = []
    # worklist of node lists
    q = deque()
    q.append(list(G.nodes()))
    while q:
        nodes = q.popleft()
        if len(nodes) <= max_cluster_size:
            clusters.append(nodes)
            continue
        # choose splitter
        if prefer_nxmetis and HAVE_NXMETIS:
            A, B = _nxmetis_bisect_nodes(G, nodes)
        else:
            A, B = _kl_bisect_nodes(G, nodes)
        if not A or not B:
            clusters.append(nodes)  # cannot split further
            continue
        q.append(list(sorted(A, key=node_sort_key)))
        q.append(list(sorted(B, key=node_sort_key)))
    # map to ids
    part = {}
    for cid, nodes in enumerate(clusters):
        for n in nodes:
            part[n] = cid
    return part


def leiden_partition(G: nx.Graph, max_cluster_size: int, resolution: float):
    if HAVE_HL:
        hl = hierarchical_leiden(
            G,
            max_cluster_size=max_cluster_size,
            resolution=resolution,
            use_modularity=True,
            random_seed=42,
            check_directed=True,
        )
        return final_partition_from_hl(hl, G)
    # Fallback: greedy modularity communities then bisection for oversized communities
    comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
    clusters = [list(sorted(c, key=node_sort_key)) for c in comms]
    # recursively split big ones via KL
    out = []
    for nodes in clusters:
        q = deque()
        q.append(nodes)
        while q:
            cur = q.popleft()
            if len(cur) <= max_cluster_size:
                out.append(cur)
            else:
                A, B = _kl_bisect_nodes(G, cur)
                if not A or not B:
                    out.append(cur)
                    continue
                q.append(list(sorted(A, key=node_sort_key)))
                q.append(list(sorted(B, key=node_sort_key)))
    part = {}
    for cid, nodes in enumerate(out):
        for n in nodes:
            part[n] = cid
    return part


# ---------- per-cluster PLL build (serial-friendly) ----------
def _build_one_cluster_serial(G, cid, nodes, neigh_map):
    nodes_set = set(nodes)
    subg = nx.Graph()
    subg.add_nodes_from(nodes)
    for u in nodes:
        for v, data in G[u].items():
            if v in nodes_set:
                subg.add_edge(u, v, weight=float(data.get("weight", 1.0)))
    # PLL order by eccentricity if possible
    try:
        ecc = {}
        for comp in nx.connected_components(subg):
            cg = subg.subgraph(comp)
            ecc.update(nx.eccentricity(cg))
        order = sorted(nodes, key=lambda n: ecc.get(n, 0))
    except Exception:
        order = list(nodes)
    t0 = time.time()
    pll = PrunedPLLIndex(subg, order)
    pll.build()
    build_time = time.time() - t0
    # boundary
    boundary = [n for n in nodes if any((nbr not in nodes_set) for nbr in neigh_map[n])]
    # boundary pairs
    pairs = {}
    for i in range(len(boundary)):
        ui = boundary[i]
        for j in range(i + 1, len(boundary)):
            vj = boundary[j]
            d = pll.query(ui, vj)
            pairs[(ui, vj)] = d
    label_entries = sum(len(m) for m in pll.labels.values())
    return cid, nodes, boundary, pairs, pll.labels, build_time, label_entries


# ---------- boundary graph for E ----------
def build_g_out(G, partition, clusters, boundary_sets, boundary_pairs):
    outside_nodes = set()
    for cid in clusters:
        outside_nodes |= boundary_sets.get(cid, set())
    G_out = nx.Graph()
    G_out.add_nodes_from(outside_nodes)
    # inter-cluster original edges
    for u, v, data in G.edges(data=True):
        if u in outside_nodes and v in outside_nodes and partition[u] != partition[v]:
            G_out.add_edge(u, v, weight=float(data.get("weight", 1.0)))
    # intra-cluster super edges
    for cid, pairs in boundary_pairs.items():
        for (u, v), d in pairs.items():
            if u == v or d == float("inf"):
                continue
            if G_out.has_edge(u, v):
                if d < G_out[u][v].get("weight", float("inf")):
                    G_out[u][v]["weight"] = float(d)
            else:
                G_out.add_edge(u, v, weight=float(d))
    return G_out


# ---------- E (two-table) ----------
def _make_adj(G):
    adj = {}
    for u in G.nodes():
        lst = []
        for v, data in G[u].items():
            lst.append((v, float(data.get("weight", 1.0))))
        adj[u] = lst
    return adj


def build_potentials_per_target(G_out, boundary_sets):
    adj_out = _make_adj(G_out)
    all_boundary = list(G_out.nodes())
    b2c = defaultdict(dict)
    # Precompute which cluster each boundary belongs to
    b2A = {}
    for A, bset in boundary_sets.items():
        for b in bset:
            b2A[b] = A
    for B in boundary_sets.keys():
        srcs = boundary_sets[B]
        if not srcs:
            continue
        dist = {x: float("inf") for x in all_boundary}
        enter = {x: None for x in all_boundary}
        heap = []
        ctr = 0
        for c in srcs:
            if c in dist:
                dist[c] = 0.0
                enter[c] = c
                heapq.heappush(heap, (0.0, ctr, c))
                ctr += 1
        while heap:
            d, _, x = heapq.heappop(heap)
            if d != dist[x]:
                continue
            for y, w in adj_out[x]:
                nd = d + w
                if nd < dist[y]:
                    dist[y] = nd
                    enter[y] = enter[x]
                    heapq.heappush(heap, (nd, ctr, y))
                    ctr += 1
        out = {b: (dist[b], enter[b]) for b in all_boundary if dist[b] < float("inf")}
        for b, val in out.items():
            b2c[b][B] = val
    return b2c


def _build_all_boundary_arrays_one_cluster(labels, nodes, blist):
    pll = PrunedPLLIndex.from_labels(labels)
    arr_map = {}
    L = len(blist)
    for n in nodes:
        arr = np.empty((L,), dtype=np.float64)
        for i, b in enumerate(blist):
            arr[i] = pll.query(n, b)
        arr_map[n] = arr
    return arr_map


def build_node_to_all_boundary_arrays(inside_pll, clusters, boundary_sets):
    allb_list = {}
    allb_index = {}
    node_to_allb_arr = {}
    for cid, pll in inside_pll.items():
        blist = sorted(boundary_sets.get(cid, set()), key=node_sort_key)
        allb_list[cid] = blist
        allb_index[cid] = {b: i for i, b in enumerate(blist)}
        arr_map = _build_all_boundary_arrays_one_cluster(pll.labels, clusters[cid], blist)
        node_to_allb_arr[cid] = arr_map
    return node_to_allb_arr, allb_list, allb_index


def build_ab_topk_from_boundary_to_cluster(boundary_to_cluster_D, boundary_sets, topk_mid=16):
    AB_topk = defaultdict(lambda: defaultdict(list))
    b2A = {}
    for A, bset in boundary_sets.items():
        for b in bset:
            b2A[b] = A
    items_by_AB = defaultdict(lambda: defaultdict(list))
    for b, perB in boundary_to_cluster_D.items():
        A = b2A.get(b)
        if A is None:
            continue
        for B, (mid, c) in perB.items():
            items_by_AB[A][B].append((mid, b, c))
    for A in items_by_AB:
        for B in items_by_AB[A]:
            items = items_by_AB[A][B]
            items.sort(key=lambda x: x[0])
            keep = items[:topk_mid]
            AB_topk[A][B] = [(b, mid, c) for (mid, b, c) in keep]
    return AB_topk


def query_cross_with_all_boundary_tables(u, v, node_cluster,
                                         node_to_allb_arr, allb_index, allb_list,
                                         boundary_to_cluster_D, AB_topk,
                                         topk_e_u, topk_e_mid,
                                         inside_pll):
    A = node_cluster[u]
    B = node_cluster[v]
    if A == B:
        return inside_pll[A].query(u, v)

    # 左端：u 最近出口 Top-K
    arr_u = node_to_allb_arr.get(A, {}).get(u)
    if arr_u is None or arr_u.size == 0:
        return float("inf")
    blistA = allb_list.get(A, [])
    if not isinstance(blistA, list):
        blistA = list(blistA)
    k = min(topk_e_u, len(arr_u))
    idx = np.argpartition(arr_u, k - 1)[:k]
    left_candidates = [(blistA[i], float(arr_u[i])) for i in idx]
    left_candidates.sort(key=lambda x: x[1])

    # 中段：优先 AB_topk，其次全表
    mid_list = AB_topk.get(A, {}).get(B, [])
    if topk_e_mid > 0 and mid_list:
        mid_list = mid_list[:topk_e_mid]
    allow_map = {bb: (mid, c) for (bb, mid, c) in mid_list} if mid_list else None

    # 右端
    arr_v = node_to_allb_arr.get(B, {}).get(v)
    if arr_v is None:
        return float("inf")
    blistB = allb_list.get(B, [])
    if not isinstance(blistB, list):
        blistB = list(blistB)

    best = float("inf")
    for (b, du) in left_candidates:
        if allow_map is not None and b in allow_map:
            mid, c = allow_map[b]
        else:
            entry = boundary_to_cluster_D.get(b, {}).get(B)
            if not entry:
                continue
            mid, c = entry
        if c is None:
            continue

        iB = allb_index[B].get(c)
        if iB is None:
            try:
                iB = blistB.index(c)
            except ValueError:
                continue
        dv = float(arr_v[iB])

        val = du + mid + dv
        if val < best:
            best = val

    return best


# ---------- sampling & ground-truth ----------
def sample_entity_pairs(G, n_pairs=1000, in_lcc=True, rng_seed=42):
    entity_nodes = [n for n in G.nodes() if not is_rel(n)]
    if not entity_nodes:
        raise RuntimeError("No entity nodes.")
    comps = list(nx.connected_components(G))
    if in_lcc and comps:
        lcc = max(comps, key=len)
        entity_nodes = [n for n in entity_nodes if n in lcc]
    if len(entity_nodes) < 2:
        raise RuntimeError("Too few entity nodes.")
    # bucket by CC
    node2cc = {}
    for i, cc in enumerate(comps):
        for n in cc:
            node2cc[n] = i
    bucket = defaultdict(list)
    for n in entity_nodes:
        bucket[node2cc[n]].append(n)
    weighted_bins = [(cid, len(nodes)) for cid, nodes in bucket.items() if len(nodes) >= 2]
    if not weighted_bins:
        raise RuntimeError("No CC with >=2 entity nodes.")
    cids, sizes = zip(*weighted_bins)
    total = sum(sizes)
    probs = [s / total for s in sizes]
    rng = random.Random(rng_seed)
    pairs = []
    trials = 0
    max_trials = n_pairs * 10
    while len(pairs) < n_pairs and trials < max_trials:
        trials += 1
        cid = rng.choices(cids, weights=probs, k=1)[0]
        nodes_in_cc = [n for n in entity_nodes if node2cc[n] == cid]
        if len(nodes_in_cc) < 2:
            continue
        u, v = rng.sample(nodes_in_cc, 2)
        pairs.append((u, v))
    return pairs


def compute_gt_distances(G, pairs):
    gt = []
    for u, v in pairs:
        d = nx.shortest_path_length(G, u, v, weight="weight")
        gt.append((u, v, d))
    return gt


# ---------- build+evaluate ----------
def build_all_e_with_partitioner(G, args, partitioner_name="leiden"):
    t0 = time.time()
    # partition
    if partitioner_name == "leiden":
        partition = leiden_partition(G, max_cluster_size=args.max_cluster_size, resolution=args.resolution)
    elif partitioner_name == "metis":
        partition = hierarchical_metis_partition(G, max_cluster_size=args.max_cluster_size, prefer_nxmetis=True)
    else:
        raise ValueError("partitioner_name must be 'leiden' or 'metis'")

    # clusters, neigh
    clusters = defaultdict(list)
    for n, cid in partition.items():
        clusters[cid].append(n)
    neigh_map = {n: list(G.neighbors(n)) for n in G.nodes()}

    # per-cluster PLL (serial to keep it robust)
    inside_pll = {}
    boundary_sets = {}
    boundary_pairs = {}
    node_cluster = {}
    for cid, nodes in clusters.items():
        cid, nodes, boundary, pairs, labels, _, _ = _build_one_cluster_serial(G, cid, nodes, neigh_map)
        inside_pll[cid] = PrunedPLLIndex.from_labels(labels)
        boundary_sets[cid] = set(boundary)
        for n in nodes:
            node_cluster[n] = cid
        boundary_pairs[cid] = pairs

    # boundary graph and E tables
    G_out = build_g_out(G, partition, clusters, boundary_sets, boundary_pairs)
    boundary_to_cluster_D = build_potentials_per_target(G_out, boundary_sets)
    node_to_allb_arr_E, allb_list_E, allb_index_E = build_node_to_all_boundary_arrays(
        inside_pll, clusters, boundary_sets
    )
    AB_topk = build_ab_topk_from_boundary_to_cluster(
        boundary_to_cluster_D, boundary_sets, topk_mid=args.topk_e_mid
    )

    preprocessing_time = time.time() - t0

    # summary sizes
    sizes = Counter(partition.values())
    top_sizes = sizes.most_common(10)
    summary = {
        "num_clusters": len(sizes),
        "top_sizes": top_sizes[:10],
        "partitioner": partitioner_name,
        "dep_flags": {
            "HAVE_HL": HAVE_HL,
            "HAVE_NXMETIS": HAVE_NXMETIS,
        },
    }

    return (
        partition,
        clusters,
        inside_pll,
        boundary_sets,
        node_cluster,
        node_to_allb_arr_E,
        allb_list_E,
        allb_index_E,
        boundary_to_cluster_D,
        AB_topk,
        preprocessing_time,
        summary,
    )


def evaluate_with_gt(
    G,
    gt,
    inside_pll,
    node_cluster,
    node_to_allb_arr_E,
    allb_index_E,
    allb_list_E,
    boundary_to_cluster_D,
    AB_topk,
    topk_e_u,
    topk_e_mid,
    preprocessing_time: float = 0.0,
    label: str = "",
):
    correct = total_eval = 0
    err = 0.0
    t0 = time.time()
    for u, v, d in gt:
        est = query_cross_with_all_boundary_tables(
            u,
            v,
            node_cluster,
            node_to_allb_arr_E,
            allb_index_E,
            allb_list_E,
            boundary_to_cluster_D,
            AB_topk,
            topk_e_u,
            topk_e_mid,
            inside_pll,
        )
        if est == d:
            correct += 1
        if est != float("inf"):
            err += abs(est - d)
            total_eval += 1
    tE = time.time() - t0
    rows = [
        [
            f"{label} (u-TopK={topk_e_u}, mid-TopK={topk_e_mid})",
            tE,
            len(gt),
            correct,
            (err / total_eval if total_eval > 0 else float("inf")),
            preprocessing_time,
        ]
    ]
    return pd.DataFrame(
        rows,
        columns=["method", "query_time_sec", "samples", "exact_matches", "mae", "preprocessing_time"],
    )


# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kg_file", type=str, default=None, help="Path to WN18.txt (overrides toy)")
    p.add_argument("--toy", action="store_true", help="Use synthetic SBM toy graph")
    p.add_argument("--toy_blocks", type=int, default=4)
    p.add_argument("--toy_block_size", type=int, default=60)
    p.add_argument("--toy_p_in", type=float, default=0.06)
    p.add_argument("--toy_p_out", type=float, default=0.005)

    p.add_argument("--eval_pairs", type=int, default=300)
    p.add_argument("--resolution", type=float, default=0.3)
    p.add_argument("--max_cluster_size", type=int, default=200)
    p.add_argument("--topk_e_u", type=int, default=8)
    p.add_argument("--topk_e_mid", type=int, default=8)

    # CSV prefix for per-cluster CC enumeration
    p.add_argument(
        "--cc_csv_prefix",
        type=str,
        default=None,
        help="If set, save per-cluster CC enumeration to <prefix>_leiden_cc.csv and <prefix>_metis_cc.csv",
    )

    args = p.parse_args()

    # build/load graph
    if args.kg_file:
        G = load_wn18_graph_relation_nodes(args.kg_file)
        src_name = f"WN18({os.path.basename(args.kg_file)})-rel_nodes"
    else:
        G = make_sbm_toy(args.toy_blocks, args.toy_block_size, args.toy_p_in, args.toy_p_out, seed=42)
        src_name = f"SBM-{args.toy_blocks}x{args.toy_block_size}"

    print(f"[INFO] Graph: {src_name}, |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

    # sample pairs and GT
    pairs = sample_entity_pairs(G, n_pairs=args.eval_pairs, in_lcc=True, rng_seed=42)
    gt = compute_gt_distances(G, pairs)

    # run Leiden
    out_leiden = build_all_e_with_partitioner(G, args, partitioner_name="leiden")
    df_leiden = evaluate_with_gt(
        G,
        gt,
        out_leiden[2],
        out_leiden[4],
        out_leiden[5],
        out_leiden[7],  # allb_index
        out_leiden[6],  # allb_list
        out_leiden[8],
        out_leiden[9],
        args.topk_e_u,
        args.topk_e_mid,
        preprocessing_time=out_leiden[10],
        label="E + Leiden",
    )

    # run METIS (or fallback) on the same GT
    out_metis = build_all_e_with_partitioner(G, args, partitioner_name="metis")
    df_metis = evaluate_with_gt(
        G,
        gt,
        out_metis[2],
        out_metis[4],
        out_metis[5],
        out_metis[7],  # allb_index
        out_metis[6],  # allb_list
        out_metis[8],
        out_metis[9],
        args.topk_e_u,
        args.topk_e_mid,
        preprocessing_time=out_metis[10],
        label=("E + METIS" if HAVE_NXMETIS else "E + (KL-bisect METIS-fallback)"),
    )

    # merge and print
    df = pd.concat([df_leiden, df_metis], ignore_index=True)
    print("\n=== Entities→Entities Evaluation (E method): Leiden vs METIS ===")
    print(df.to_string(index=False))

    # summaries
    summ_l = out_leiden[11]
    summ_m = out_metis[11]
    print("\n[SUMMARY] #clusters: Leiden={}, METIS={}".format(summ_l["num_clusters"], summ_m["num_clusters"]))
    print("[SUMMARY] deps: ", summ_l["dep_flags"])

    # Connected-components enumeration per cluster for both partitioners
    df_cc_leiden = enumerate_cluster_cc(G, out_leiden[1])
    df_cc_metis = enumerate_cluster_cc(G, out_metis[1])

    multi_l = int((df_cc_leiden["num_connected_components"] > 1).sum())
    multi_m = int((df_cc_metis["num_connected_components"] > 1).sum())

    print(
        "\n[CC] Leiden:   {} clusters; {} clusters have >1 connected component.".format(
            len(df_cc_leiden), multi_l
        )
    )
    print(
        "[CC] METIS:    {} clusters; {} clusters have >1 connected component.".format(
            len(df_cc_metis), multi_m
        )
    )

    # Show top offenders (first 10)
    def show_top(df, name):
        top = df[df["num_connected_components"] > 1].head(10)
        if top.empty:
            print(f"[CC] {name}: all clusters are single-component.")
        else:
            print(f"[CC] {name}: top multi-CC clusters (cluster_id, size, num_cc):")
            for _, row in top.iterrows():
                print(
                    "      cid={}, size={}, num_cc={}".format(
                        int(row.cluster_id), int(row.size), int(row.num_connected_components)
                    )
                )

    show_top(df_cc_leiden, "Leiden")
    show_top(df_cc_metis, "METIS")

    # Optional CSV dump
    if args.cc_csv_prefix:
        f1 = args.cc_csv_prefix + "_leiden_cc.csv"
        f2 = args.cc_csv_prefix + "_metis_cc.csv"
        df_cc_leiden.to_csv(f1, index=False)
        df_cc_metis.to_csv(f2, index=False)
        print(f"[CC] Saved CSVs: {f1} , {f2}")


if __name__ == "__main__":
    main()
