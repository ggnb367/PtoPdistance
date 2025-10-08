import argparse
import time
import random
import heapq
import os
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
from torch_geometric.datasets import Planetoid
from graspologic.partition import hierarchical_leiden


# ===============================
# Pruned PLL Index
# ===============================
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

    def build(self):
        assert self.G is not None, "build() needs a graph; use from_labels() if you already have labels."
        for v in tqdm(self.order, desc="Building PLL", unit="node"):
            self._pruned_dijkstra(v)

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


# ===============================
# Dataset Loader (with friendly fallback)
# ===============================
def load_planetoid_graph(name="Pubmed", root=None):
    """
    Load Planetoid (Cora/CiteSeer/Pubmed). Use a persistent data_root.
    If download fails (network timeout), raise clear instruction.
    """
    root = root or os.path.abspath(f"./data/{name}")
    try:
        dataset = Planetoid(root=root, name=name)
    except Exception as e:
        msg = (
            f"Failed to fetch Planetoid({name}) into '{root}'.\n"
            "Likely a GitHub access timeout.\n"
            f"Please create '{root}/raw' and put the following files there:\n"
            f"  ind.{name.lower()}.x  ind.{name.lower()}.tx  ind.{name.lower()}.allx\n"
            f"  ind.{name.lower()}.y  ind.{name.lower()}.ty  ind.{name.lower()}.ally\n"
            f"  ind.{name.lower()}.graph  ind.{name.lower()}.test.index\n"
            "Then re-run with --data_root pointing to that directory."
        )
        raise RuntimeError(msg) from e
    data = dataset[0]
    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edges = set()
    for u, v in zip(edge_index[0], edge_index[1]):
        if u == v:
            continue
        a, b = int(u), int(v)
        if a > b:
            a, b = b, a
        edges.add((a, b))
    for u, v in edges:
        G.add_edge(u, v, weight=1.0)
    return G


# ===============================
# Partition helpers
# ===============================
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


# ===============================
# Per-cluster PLL worker
# ===============================
def _build_one_cluster(args):
    """
    Return: (cid, nodes, boundary_list, boundary_pairs_dict, labels_dict, build_time, label_entries)
    """
    cid, nodes, edges, neigh_map = args
    nodes_set = set(nodes)

    subg = nx.Graph()
    subg.add_nodes_from(nodes)
    for u, v, w in edges:
        subg.add_edge(u, v, weight=float(w))

    ecc = {}
    for comp in nx.connected_components(subg):
        cg = subg.subgraph(comp)
        ecc.update(nx.eccentricity(cg))
    order = sorted(nodes, key=lambda n: ecc.get(n, 0))

    t0 = time.time()
    pll = PrunedPLLIndex(subg, order)
    for root in pll.order:
        dist = {root: 0.0}
        heap = [(0.0, 0, root)]
        counter = 0
        while heap:
            d, _, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if pll.query(root, u) <= d:
                continue
            pll.labels[u][root] = d
            for v, data in subg[u].items():
                w = float(data.get("weight", 1.0))
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    counter += 1
                    heapq.heappush(heap, (nd, counter, v))
    build_time = time.time() - t0

    # boundary: neighbors outside the cluster in ORIGINAL graph
    boundary = [n for n in nodes if any((nbr not in nodes_set) for nbr in neigh_map[n])]

    # intra-cluster boundary pairs via internal PLL
    pairs = {}
    for i in range(len(boundary)):
        ui = boundary[i]
        for j in range(i + 1, len(boundary)):
            vj = boundary[j]
            pairs[(ui, vj)] = pll.query(ui, vj)

    label_entries = sum(len(m) for m in pll.labels.values())
    return cid, nodes, boundary, pairs, pll.labels, build_time, label_entries


# ===============================
# Payload estimators (rough, content-only)
# ===============================
def payload_bytes_labels_entry_count(entry_count: int) -> int:
    return 16 * entry_count  # ~ (8B key + 8B value)

def payload_bytes_pair_id_dist(count: int) -> int:
    return (8 + 8) * count  # (id,int64 + dist,float64)

def payload_bytes_triple(count: int) -> int:
    return (8 + 8 + 8) * count  # (id,id,dist)

def payload_bytes_edges(edge_count: int) -> int:
    return (8 + 8 + 8) * edge_count  # u,v,w

def payload_bytes_nodes(node_count: int) -> int:
    return 16 * node_count  # rough for id map

def payload_bytes_floats(count: int) -> int:
    return 8 * count

def format_mib(nbytes: int) -> str:
    return f"{nbytes / (1024*1024):.2f} MiB"


# ===============================
# Build G_out & outside PLL (Hybrid A)
# ===============================
def build_g_out_and_outpll(G, partition, clusters, boundary_sets, boundary_pairs):
    outside_nodes = set()
    for cid in clusters:
        outside_nodes |= boundary_sets.get(cid, set())

    t0 = time.time()
    G_out = nx.Graph()
    G_out.add_nodes_from(outside_nodes)

    # inter-cluster original edges
    for u, v, data in G.edges(data=True):
        if u in outside_nodes and v in outside_nodes and partition[u] != partition[v]:
            G_out.add_edge(u, v, weight=float(data.get("weight", 1.0)))

    # intra-cluster super edges
    intra_super_edges = 0
    for cid, pairs in boundary_pairs.items():
        for (u, v), d in pairs.items():
            if u == v or d == float("inf"):
                continue
            if G_out.has_edge(u, v):
                if d < G_out[u][v].get("weight", float("inf")):
                    G_out[u][v]["weight"] = float(d)
            else:
                G_out.add_edge(u, v, weight=float(d))
                intra_super_edges += 1
    t_build_gout = time.time() - t0

    # outside PLL — degree descending order (highest-degree first)
    t1 = time.time()
    degree_order = sorted(G_out.degree(), key=lambda x: x[1], reverse=True)
    order_out = [n for n, _ in degree_order]  # highest degree first
    outside_pll = PrunedPLLIndex(G_out, order_out)
    outside_pll.build()
    t_build_outpll = time.time() - t1

    outside_pll_entries = sum(len(m) for m in outside_pll.labels.values())

    stats = {
        "time_sec_build_g_out": t_build_gout,
        "time_sec_build_outpll": t_build_outpll,
        "nodes": G_out.number_of_nodes(),
        "edges": G_out.number_of_edges(),
        "intra_super_edges_added": intra_super_edges,
        "payload_bytes_nodes": payload_bytes_nodes(G_out.number_of_nodes()),
        "payload_bytes_edges": payload_bytes_edges(G_out.number_of_edges()),
        "pll_label_entries": outside_pll_entries,
        "pll_payload_bytes": payload_bytes_labels_entry_count(outside_pll_entries),
    }
    return G_out, outside_pll, stats


# ===============================
# Build virtual graph (for B/C)
# ===============================
def build_virtual_graph(G_out, boundary_sets):
    t0 = time.time()
    G_rep = G_out.copy()
    virtual_of = {}
    for cid, bset in boundary_sets.items():
        vnode = ("VIRT", cid)
        virtual_of[cid] = vnode
        G_rep.add_node(vnode)
        for b in bset:
            G_rep.add_edge(vnode, b, weight=0.0)
    t = time.time() - t0
    return G_rep, virtual_of, t


# ===============================
# Adjacency builder (for workers)
# ===============================
def _make_adj(G):
    adj = {}
    for u in G.nodes():
        lst = []
        for v, data in G[u].items():
            lst.append((v, float(data.get("weight", 1.0))))
        adj[u] = lst
    return adj


# ===============================
# Hybrid B: cluster-pair shortest on virtual graph (main proc)
# ===============================
def build_hybrid_b_pairs(G_rep, boundary_sets, clusters):
    cluster_ids = sorted(clusters.keys())
    t0 = time.time()
    pair_info = {}
    for i in range(len(cluster_ids)):
        ci = cluster_ids[i]
        si = ("VIRT", ci)
        for j in range(i + 1, len(cluster_ids)):
            cj = cluster_ids[j]
            sj = ("VIRT", cj)
            try:
                d, path = nx.bidirectional_dijkstra(G_rep, si, sj, weight="weight")
                exit_u = None
                enter_v = None
                if len(path) >= 2:
                    if path[1] in boundary_sets.get(ci, set()):
                        exit_u = path[1]
                    if path[-2] in boundary_sets.get(cj, set()):
                        enter_v = path[-2]
                pair_info[(ci, cj)] = {"dist": d, "exit_u": exit_u, "enter_v": enter_v}
            except nx.NetworkXNoPath:
                pair_info[(ci, cj)] = {"dist": float("inf"), "exit_u": None, "enter_v": None}
    t = time.time() - t0
    stats = {
        "time_sec_all_pairs_shortest": t,
        "pairs": len(pair_info),
        "payload_bytes_pairs": payload_bytes_triple(len(pair_info)),
    }
    return pair_info, stats


# ===============================
# C worker: per-source SSSP on G_rep, keep Top-K routes PER (S,B) PAIR
# ===============================
def _c_sssp_topk_per_pair_worker(args):
    """
    Dijkstra from ("VIRT", S); forbid passing other virtual nodes.
    When first time reaching an external boundary node t in cluster B!=S:
      push (dist, t, exit_b) into per-B heap; keep only Top-K by dist.
    Return (S, {B: [(t, dist, exit_b), ...]})
    """
    source_cid, adj_rep, boundary_sets, partition, k_pairs = args
    vnode = ("VIRT", source_cid)

    def is_virtual(x):
        return isinstance(x, tuple) and len(x) == 2 and x[0] == "VIRT"

    dist = {}
    first_exit = {}
    heap = []
    ctr = 0

    dist[vnode] = 0.0
    first_exit[vnode] = None
    heapq.heappush(heap, (0.0, ctr, vnode)); ctr += 1

    per_B_heaps = defaultdict(list)  # B -> max-heap [(-dist, t, exit_b)]
    visited_external = set()

    while heap:
        d, _, x = heapq.heappop(heap)
        if d != dist.get(x, float("inf")):
            continue
        if is_virtual(x) and x != vnode:
            continue
        if (not is_virtual(x)) and (partition[x] != source_cid):
            if x not in visited_external:
                visited_external.add(x)
                B = partition[x]
                exit_b = first_exit.get(x, None)
                h = per_B_heaps[B]
                entry = (-d, x, exit_b)
                if len(h) < k_pairs:
                    heapq.heappush(h, entry)
                else:
                    if -h[0][0] > d:
                        heapq.heapreplace(h, entry)

        for y, w in adj_rep[x]:
            if is_virtual(y) and y != vnode:
                continue
            nd = d + w
            if nd < dist.get(y, float("inf")):
                dist[y] = nd
                if x == vnode:
                    first_exit[y] = y if y in boundary_sets[source_cid] else None
                else:
                    first_exit[y] = first_exit.get(x, None)
                heapq.heappush(heap, (nd, ctr, y)); ctr += 1

    out = {}
    for B, h in per_B_heaps.items():
        items = [(-negd, t, eb) for (negd, t, eb) in h]
        items.sort(key=lambda z: z[0])
        out[B] = items
    return source_cid, out


def build_c_topk_pairs_per_ab(G_rep, boundary_sets, partition, node_cluster, k_pairs=16, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count() or 2

    adj_rep = _make_adj(G_rep)
    args_list = [(S, adj_rep, boundary_sets, partition, k_pairs) for S in sorted(boundary_sets.keys())]

    pf_by_source = defaultdict(dict)                       # S -> {t: (dist, exit_b)}
    AB_candidates = defaultdict(lambda: defaultdict(list)) # S -> {B: [t,...]}
    kept_triplets = defaultdict(lambda: defaultdict(list)) # S -> {B: [(t, dist, exit_b), ...]}

    t0 = time.time()
    kept_entries = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_c_sssp_topk_per_pair_worker, a) for a in args_list]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Hybrid C: SSSP per source (TopK per pair)", unit="cluster"):
            S, perB = fut.result()
            for B, items in perB.items():
                kept_triplets[S][B] = items
                for (dist, t, exit_b) in items:
                    pf_by_source[S][t] = (dist, exit_b)
                    AB_candidates[S][B].append(t)
                    kept_entries += 1
    t_sssp = time.time() - t0

    stats = {
        "time_sec_c_sssp": t_sssp,
        "k_pairs": k_pairs,
        "kept_entries": kept_entries,
        "payload": payload_bytes_pair_id_dist(kept_entries),
    }
    return pf_by_source, AB_candidates, kept_triplets, stats


# ===============================
# Parallel build: node→boundary arrays (only for USED boundaries)
# ===============================
def _build_used_arrays_one_cluster(args):
    cid, nodes, used_b_list, labels = args
    pll = PrunedPLLIndex.from_labels(labels)

    arr_map = {}
    L = len(used_b_list)
    entries = 0
    if L == 0:
        for n in nodes:
            arr_map[n] = np.zeros((0,), dtype=np.float64)
        return cid, arr_map, entries

    for n in nodes:
        arr = np.empty((L,), dtype=np.float64)
        for i, b in enumerate(used_b_list):
            arr[i] = pll.query(n, b)
        arr_map[n] = arr
        entries += L
    return cid, arr_map, entries


def build_node_to_boundary_used_arrays(inside_pll, clusters, used_boundaries_sets, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count() or 2

    used_boundary_list = {}
    used_boundary_index = {}
    node_to_used_arr = {}
    total_entries = 0

    tasks = []
    for cid, pll in inside_pll.items():
        blist = sorted(used_boundaries_sets.get(cid, set()))
        used_boundary_list[cid] = blist
        used_boundary_index[cid] = {b: i for i, b in enumerate(blist)}
        tasks.append((cid, clusters[cid], blist, pll.labels))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_build_used_arrays_one_cluster, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Build node→USED-boundary arrays", unit="cluster"):
            cid, arr_map, entries = fut.result()
            node_to_used_arr[cid] = arr_map
            total_entries += entries
    t = time.time() - t0

    stats = {
        "time_sec": t,
        "entries": total_entries,
        "payload": payload_bytes_floats(total_entries),
        "used_boundary_counts": {cid: len(used_boundary_list[cid]) for cid in used_boundary_list},
    }
    return node_to_used_arr, used_boundary_list, used_boundary_index, stats


# ===============================
# From C kept_triplets -> collect USED boundaries per cluster, then arrays
# ===============================
def build_c_used_arrays_from_kept(kept_triplets, pf_by_source, node_cluster,
                                  inside_pll, clusters, max_workers=None):
    used = defaultdict(set)
    kept_pairs_count = 0
    for S, perB in kept_triplets.items():
        for B, items in perB.items():
            kept_pairs_count += len(items)
            for (t, dist, exit_b) in items:
                if exit_b is not None:
                    used[S].add(exit_b)   # source exits
                used[B].add(t)           # target entries

    node_to_used_arr, used_b_list, used_b_index, stats = build_node_to_boundary_used_arrays(
        inside_pll, clusters, used, max_workers=max_workers
    )
    stats_out = {
        "time_sec": stats["time_sec"],
        "entries": stats["entries"],
        "payload": stats["payload"],
        "kept_pairs": kept_pairs_count,
        "used_boundary_counts": stats["used_boundary_counts"],
    }
    return node_to_used_arr, used_b_list, used_b_index, stats_out


# ===============================
# D worker: per-target multi-source potentials on G_out
# ===============================
def _d_potential_one_target_worker(args):
    B, adj_out, boundary_sets, all_boundary = args
    srcs = boundary_sets[B]
    if not srcs:
        return B, {}
    dist = {x: float("inf") for x in all_boundary}
    enter = {x: None for x in all_boundary}
    heap = []
    ctr = 0
    for c in srcs:
        if c in dist:
            dist[c] = 0.0
            enter[c] = c
            heapq.heappush(heap, (0.0, ctr, c)); ctr += 1
    while heap:
        d, _, x = heapq.heappop(heap)
        if d != dist[x]:
            continue
        for y, w in adj_out[x]:
            nd = d + w
            if nd < dist[y]:
                dist[y] = nd
                enter[y] = enter[x]
                heapq.heappush(heap, (nd, ctr, y)); ctr += 1
    out = {b: (dist[b], enter[b]) for b in all_boundary if dist[b] < float("inf")}
    return B, out


def build_potentials_per_target_parallel(G_out, boundary_sets, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count() or 2

    adj_out = _make_adj(G_out)
    all_boundary = list(G_out.nodes())
    args_list = [(B, adj_out, boundary_sets, all_boundary) for B in sorted(boundary_sets.keys())]

    b2c = defaultdict(dict)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_d_potential_one_target_worker, a) for a in args_list]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Hybrid D: Potentials per target", unit="cluster"):
            B, out = fut.result()
            for b, val in out.items():
                b2c[b][B] = val
    t = time.time() - t0

    entries = sum(len(m) for m in b2c.values())
    stats = {"time_sec_potentials": t, "entries": entries, "payload": payload_bytes_pair_id_dist(entries)}
    return b2c, stats


# ===============================
# D (original version, no TopK_AB): node→TopK(u) personalized exits
# ===============================
def build_node_to_topk_boundary(inside_pll, clusters, boundary_sets, topk=16):
    t0 = time.time()
    node_to_topk = {}
    entries = 0
    for cid, pll in inside_pll.items():
        bset = sorted(boundary_sets.get(cid, set()))
        nbmap = {}
        for n in clusters[cid]:
            if not bset:
                nbmap[n] = []
                continue
            heap_k = []
            for b in bset:
                d = pll.query(n, b)
                if len(heap_k) < topk:
                    heapq.heappush(heap_k, (-d, b))
                else:
                    if d < -heap_k[0][0]:
                        heapq.heapreplace(heap_k, (-d, b))
            cand = [(b, -negd) for (negd, b) in heap_k]
            cand.sort(key=lambda x: x[1])
            nbmap[n] = cand
            entries += len(cand)
        node_to_topk[cid] = nbmap
    t = time.time() - t0
    stats = {"time_sec_n2topk": t, "entries": entries, "payload": payload_bytes_pair_id_dist(entries), "topk": topk}
    return node_to_topk, stats


# ===============================
# D (original) query: purely by potentials + node→TopK(u), right side uses PLL
# ===============================
def query_hybrid_d_original(u, v, node_cluster,
                            node_to_topk_D,  # A -> {u: [(b, du), ...]}
                            boundary_to_cluster_D,  # b -> {B: (phi_B(b), enter_B(b))}
                            inside_pll):
    A = node_cluster[u]
    B = node_cluster[v]
    if A == B:
        return inside_pll[A].query(u, v)

    topk_u = node_to_topk_D.get(A, {}).get(u, [])
    if not topk_u:
        return float("inf")

    best = float("inf")
    for b, du in topk_u:
        entry = boundary_to_cluster_D.get(b, {}).get(B, None)
        if entry is None:
            continue
        phi, c = entry
        dv = inside_pll[B].query(c, v)  # right side PLL
        val = du + phi + dv
        if val < best:
            best = val
    return best


# ===============================
# E: build node→TopK(u) for personalized exits (A侧)  — 复用 D 的构建函数
# ===============================
# (Already defined: build_node_to_topk_boundary)


# ===============================
# Queries for A/B/C/E
# ===============================
def query_hybrid_outpll(u, v, node_cluster, inside_pll, boundary_sets, outside_pll):
    cid_u = node_cluster[u]
    cid_v = node_cluster[v]
    bu = boundary_sets.get(cid_u, set())
    bv = boundary_sets.get(cid_v, set())

    if cid_u == cid_v and not bu:
        return inside_pll[cid_u].query(u, v)

    def project(x, cid, boundary_set):
        if not boundary_set:
            return None, float("inf")
        if x in boundary_set:
            return x, 0.0
        pll = inside_pll[cid]
        best = (float("inf"), None)
        for b in boundary_set:
            d = pll.query(x, b)
            if d < best[0]:
                best = (d, b)
        return best[1], best[0]

    e_u, d_u = project(u, cid_u, bu)
    e_v, d_v = project(v, cid_v, bv)
    mid = 0.0 if (e_u is None or e_v is None or e_u == e_v) else outside_pll.query(e_u, e_v)
    return d_u + mid + d_v


def query_hybrid_virtual_pairs(u, v, node_cluster, inside_pll, boundary_sets, cluster_pair_info):
    cid_u = node_cluster[u]
    cid_v = node_cluster[v]
    if cid_u == cid_v:
        return inside_pll[cid_u].query(u, v)

    key = (cid_u, cid_v) if cid_u < cid_v else (cid_v, cid_u)
    info = cluster_pair_info.get(key, None)
    if info is None or info["dist"] == float("inf"):
        return float("inf")

    if cid_u < cid_v:
        b_u = info["exit_u"]; b_v = info["enter_v"]
    else:
        b_u = info["enter_v"]; b_v = info["exit_u"]

    def best_proj_to_target_boundary(x, cid, b_target):
        pll = inside_pll[cid]
        if b_target is not None:
            return pll.query(x, b_target)
        best = float("inf")
        for b in boundary_sets.get(cid, set()):
            d = pll.query(x, b)
            if d < best:
                best = d
        return best

    du = best_proj_to_target_boundary(u, cid_u, b_u)
    dv = best_proj_to_target_boundary(v, cid_v, b_v)
    mid = info["dist"]
    return du + mid + dv


def query_hybrid_c_pairs_usedarrays(u, v, node_cluster,
                                    pf_by_source, AB_candidates,
                                    node_to_used_arr_C, used_index_C,
                                    inside_pll):
    A = node_cluster[u]
    B = node_cluster[v]
    if A == B:
        return inside_pll[A].query(u, v)

    cand_ts = AB_candidates.get(A, {}).get(B, [])
    if not cand_ts:
        return float("inf")

    arr_u = node_to_used_arr_C[A].get(u)
    arr_v = node_to_used_arr_C[B].get(v)
    if arr_u is None or arr_v is None:
        return float("inf")
    idxA = used_index_C[A]
    idxB = used_index_C[B]
    pfA = pf_by_source.get(A, {})

    best = float("inf")
    for t in cand_ts:
        dist_mid, exit_b = pfA[t]
        j = idxA.get(exit_b);  i = idxB.get(t)
        if j is None or i is None:
            continue
        du = arr_u[j];  dv = arr_v[i]
        val = du + dist_mid + dv
        if val < best:
            best = val
    return best


def query_hybrid_e_mixed(u, v, node_cluster,
                         # 来自 D 的“全局好通路”（需 d_pairs 与 USED 数组）：
                         d_pairs, node_to_used_arr_D, used_index_D,
                         # 个性化出口：TopK(u)
                         node_to_topk_E,
                         # 势能：φ_B(b), enter_B(b)
                         boundary_to_cluster_D,
                         inside_pll):
    """
    E: 混合候选 = D 的 TopK_pairs[A][B]  ∪  TopK(u)
    左端距离：优先 D 的 USED 数组；若没有，用 TopK(u) 自带 du 或一次 PLL
    右端距离：若 c 在 D 的 USED 数组里查表，否则一次簇内 PLL
    """
    A = node_cluster[u]
    B = node_cluster[v]
    if A == B:
        return inside_pll[A].query(u, v)

    cand_pairs = list(d_pairs.get(A, {}).get(B, []))  # [(b, phi, c), ...]
    topk_u = node_to_topk_E.get(A, {}).get(u, [])
    for b, du_cached in topk_u:
        entry = boundary_to_cluster_D.get(b, {}).get(B, None)
        if entry is None:
            continue
        phi, c = entry
        cand_pairs.append((b, phi, c))

    if not cand_pairs:
        return float("inf")

    arr_u = node_to_used_arr_D[A].get(u)
    arr_v = node_to_used_arr_D[B].get(v)
    idxA = used_index_D[A]
    idxB = used_index_D[B]

    best = float("inf")
    for (b, phi, c) in cand_pairs:
        iA = idxA.get(b) if arr_u is not None else None
        if iA is not None:
            du = arr_u[iA]
        else:
            du = None
            for bb, du_u in topk_u:
                if bb == b:
                    du = du_u
                    break
            if du is None:
                du = inside_pll[A].query(u, b)

        iB = idxB.get(c) if arr_v is not None else None
        if iB is not None:
            dv = arr_v[iB]
        else:
            dv = inside_pll[B].query(c, v)

        val = du + phi + dv
        if val < best:
            best = val

    return best


# ===============================
# Evaluate A/B/C/D(original)/E
# ===============================
def evaluate(G,
             inside_pll, boundary_sets, node_cluster, outside_pll,
             cluster_pair_info,
             # C:
             pf_by_source, AB_candidates, node_to_used_arr_C, used_index_C,
             # D(original):
             node_to_topk_D, boundary_to_cluster_D,
             # E:
             d_pairs, node_to_used_arr_D, used_index_D, node_to_topk_E,
             n_pairs=500):
    nodes = list(G.nodes())
    pairs = [(random.choice(nodes), random.choice(nodes)) for _ in range(n_pairs)]

    gt = []
    for u, v in pairs:
        try:
            d = nx.shortest_path_length(G, u, v, weight="weight")
            gt.append((u, v, d))
        except nx.NetworkXNoPath:
            pass

    rows = []

    # A
    correct = total = 0
    err = 0.0
    t0 = time.time()
    for u, v, d in gt:
        est = query_hybrid_outpll(u, v, node_cluster, inside_pll, boundary_sets, outside_pll)
        if est == d:
            correct += 1
        if est != float("inf"):
            err += abs(est - d); total += 1
    tA = time.time() - t0
    rows.append(["Hybrid A (leaf-PLL + G_out-PLL)", tA, total, correct, (err/total if total>0 else float("inf"))])

    # B
    correct = total = 0
    err = 0.0
    t0 = time.time()
    for u, v, d in gt:
        est = query_hybrid_virtual_pairs(u, v, node_cluster, inside_pll, boundary_sets, cluster_pair_info)
        if est == d:
            correct += 1
        if est != float("inf"):
            err += abs(est - d); total += 1
    tB = time.time() - t0
    rows.append(["Hybrid B (Virtual-Source Pairs)", tB, total, correct, (err/total if total>0 else float("inf"))])

    # C
    correct = total = 0
    err = 0.0
    t0 = time.time()
    for u, v, d in gt:
        est = query_hybrid_c_pairs_usedarrays(u, v, node_cluster,
                                              pf_by_source, AB_candidates,
                                              node_to_used_arr_C, used_index_C,
                                              inside_pll)
        if est == d:
            correct += 1
        if est != float("inf"):
            err += abs(est - d); total += 1
    tC = time.time() - t0
    rows.append(["Hybrid C (per-source SSSP + TopK_AB + USED arrays)", tC, total, correct, (err/total if total>0 else float("inf"))])

    # D (original, no TopK_AB; right side PLL)
    correct = total = 0
    err = 0.0
    t0 = time.time()
    for u, v, d in gt:
        est = query_hybrid_d_original(u, v, node_cluster,
                                      node_to_topk_D, boundary_to_cluster_D,
                                      inside_pll)
        if est == d:
            correct += 1
        if est != float("inf"):
            err += abs(est - d); total += 1
    tD = time.time() - t0
    rows.append(["Hybrid D (original: potentials + TopK(u) + right-PLL)", tD, total, correct, (err/total if total>0 else float("inf"))])

    # E
    correct = total = 0
    err = 0.0
    t0 = time.time()
    for u, v, d in gt:
        est = query_hybrid_e_mixed(u, v, node_cluster,
                                   d_pairs, node_to_used_arr_D, used_index_D,
                                   node_to_topk_E,
                                   boundary_to_cluster_D,
                                   inside_pll)
        if est == d:
            correct += 1
        if est != float("inf"):
            err += abs(est - d); total += 1
    tE = time.time() - t0
    rows.append(["Hybrid E (Mixed: TopK_pairs ∪ TopK(u) + reusing tables)", tE, total, correct, (err/total if total>0 else float("inf"))])

    return pd.DataFrame(rows, columns=["method", "query_time_sec", "samples", "exact_matches", "mae"])


# ===============================
# Build all (A/B/C/D(original)/E)
# ===============================
def build_all(G, args):
    # Partition
    hl = hierarchical_leiden(
        G,
        max_cluster_size=args.max_cluster_size,
        resolution=args.resolution,
        use_modularity=True,
        random_seed=42,
        check_directed=True,
    )
    partition = final_partition_from_hl(hl, G)

    # cluster -> nodes
    clusters = defaultdict(list)
    for n, cid in partition.items():
        clusters[cid].append(n)

    # per-cluster PLL in parallel
    neigh_map = {n: list(G.neighbors(n)) for n in G.nodes()}
    tasks = []
    for cid, nodes in clusters.items():
        nset = set(nodes)
        edges = [(u, v, data.get("weight", 1.0))
                 for u, v, data in G.edges(nodes, data=True)
                 if u in nset and v in nset]
        tasks.append((cid, nodes, edges, neigh_map))

    inside_pll = {}
    boundary_sets = {}
    boundary_pairs = {}
    node_cluster = {}
    inside_time_sum = 0.0
    inside_label_entries_sum = 0

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        it = as_completed([ex.submit(_build_one_cluster, t) for t in tasks])
        it = tqdm(it, total=len(tasks), desc="Clusters (parallel PLL)", unit="cluster")
        for fut in it:
            cid, nodes, boundary, pairs, labels, build_time, label_entries = fut.result()
            inside_time_sum += build_time
            inside_label_entries_sum += label_entries
            inside_pll[cid] = PrunedPLLIndex.from_labels(labels)
            boundary_sets[cid] = set(boundary)
            for n in nodes:
                node_cluster[n] = cid
            boundary_pairs[cid] = pairs

    # A: G_out + outside PLL
    G_out, outside_pll, stats_A = build_g_out_and_outpll(G, partition, clusters, boundary_sets, boundary_pairs)

    # B/C: Virtual graph
    G_rep, virtual_of, t_vgraph = build_virtual_graph(G_out, boundary_sets)

    # B: cluster-pair shortest paths
    cluster_pair_info, stats_B = build_hybrid_b_pairs(G_rep, boundary_sets, clusters)

    # --------- C phase ----------
    pf_by_source, AB_candidates, kept_triplets, stats_C_topk = build_c_topk_pairs_per_ab(
        G_rep, boundary_sets, partition, node_cluster,
        k_pairs=args.topk_c_pairs, max_workers=args.c_workers
    )
    node_to_used_arr_C, used_b_list_C, used_b_index_C, stats_C_used = build_c_used_arrays_from_kept(
        kept_triplets, pf_by_source, node_cluster,
        inside_pll, clusters, max_workers=args.c_workers
    )

    # --------- D(original) phase ----------
    boundary_to_cluster_D, stats_D_pot = build_potentials_per_target_parallel(
        G_out, boundary_sets, max_workers=args.d_workers
    )
    node_to_topk_D, stats_D_topk = build_node_to_topk_boundary(
        inside_pll, clusters, boundary_sets, topk=args.topk_d_u
    )

    # --------- E phase ----------
    # 仍然需要 (A,B) TopK_pairs 与 USED arrays（与之前相同）
    # 使用 D 的 potentials + C/D 的构件
    # 这里我们重用 D-pairs 的实现来给 E 提供全局好通路与 USED arrays：
    def build_d_pairs_and_used_arrays(boundary_to_cluster_D, boundary_sets, inside_pll, clusters,
                                      topk_pairs=16, max_workers=None):
        t0 = time.time()
        cluster_ids = sorted(clusters.keys())
        pairs = defaultdict(lambda: defaultdict(list))  # A -> {B: [(b, phi, c), ...]}

        used = defaultdict(set)
        kept_pairs = 0

        for A in cluster_ids:
            A_boundaries = list(boundary_sets.get(A, set()))
            for B in cluster_ids:
                if B == A:
                    continue
                cand = []
                for b in A_boundaries:
                    entry = boundary_to_cluster_D.get(b, {}).get(B, None)
                    if entry is None:
                        continue
                    phi, c = entry
                    cand.append((phi, b, c))
                if not cand:
                    continue
                cand.sort(key=lambda x: x[0])
                kept = cand[:topk_pairs]
                for phi, b, c in kept:
                    pairs[A][B].append((b, phi, c))
                    used[A].add(b)
                    if c is not None:
                        used[B].add(c)
                kept_pairs += len(kept)
        t_select = time.time() - t0

        node_to_used_arr, used_b_list, used_b_index, stats_arr = build_node_to_boundary_used_arrays(
            inside_pll, clusters, used, max_workers=max_workers
        )
        stats = {
            "time_sec_select_pairs": t_select,
            "kept_pairs": kept_pairs,
            "used_boundary_counts": stats_arr["used_boundary_counts"],
            "time_sec_arrays": stats_arr["time_sec"],
            "entries_arrays": stats_arr["entries"],
            "payload_arrays": stats_arr["payload"],
            "topk_pairs": args.topk_e_pairs,
        }
        return pairs, node_to_used_arr, used_b_list, used_b_index, stats

    d_pairs_for_E, node_to_used_arr_D_for_E, used_b_list_D_for_E, used_b_index_D_for_E, stats_E_pairs = build_d_pairs_and_used_arrays(
        boundary_to_cluster_D, boundary_sets, inside_pll, clusters,
        topk_pairs=args.topk_e_pairs, max_workers=args.d_workers
    )

    # E 也需要 node→TopK(u)，可以复用 D(original) 的构建结果：
    node_to_topk_E = node_to_topk_D

    # ===== stats aggregation =====
    stats = {
        "graph": {"V": G.number_of_nodes(), "E": G.number_of_edges()},
        "partition": {"clusters": len(clusters),
                      "boundary_total": sum(len(s) for s in boundary_sets.values())},
        "inside": {"time_sec": inside_time_sum,
                   "label_entries": inside_label_entries_sum,
                   "payload": payload_bytes_labels_entry_count(inside_label_entries_sum)},
        "g_out": {"nodes": stats_A["nodes"], "edges": stats_A["edges"],
                  "time_sec": stats_A["time_sec_build_g_out"],
                  "payload_nodes": stats_A["payload_bytes_nodes"],
                  "payload_edges": stats_A["payload_bytes_edges"]},
        "outside_pll": {"time_sec": stats_A["time_sec_build_outpll"],
                        "label_entries": stats_A["pll_label_entries"],
                        "payload": stats_A["pll_payload_bytes"]},
        "virtual_graph": {"time_sec": t_vgraph},
        "hybrid_b": {"time_sec_pairs": stats_B["time_sec_all_pairs_shortest"],
                     "pairs": stats_B["pairs"],
                     "payload": stats_B["payload_bytes_pairs"]},

        # C
        "c_topk": {"time_sec": stats_C_topk["time_sec_c_sssp"],
                   "k_pairs": stats_C_topk["k_pairs"],
                   "kept_entries": stats_C_topk["kept_entries"],
                   "payload": stats_C_topk["payload"]},
        "c_used_arrays": {"time_sec": stats_C_used["time_sec"],
                          "entries": stats_C_used["entries"],
                          "payload": stats_C_used["payload"],
                          "kept_pairs": stats_C_used["kept_pairs"],
                          "used_boundary_counts": stats_C_used["used_boundary_counts"]},

        # D(original)
        "d_potentials": {"time_sec": stats_D_pot["time_sec_potentials"],
                         "entries": stats_D_pot["entries"],
                         "payload": stats_D_pot["payload"]},
        "d_topk_u": {"time_sec": stats_D_topk["time_sec_n2topk"],
                     "entries": stats_D_topk["entries"],
                     "payload": stats_D_topk["payload"],
                     "topk": stats_D_topk["topk"]},

        # E
        "e_pairs": {"time_sec_select": stats_E_pairs["time_sec_select_pairs"],
                    "kept_pairs": stats_E_pairs["kept_pairs"],
                    "time_sec_arrays": stats_E_pairs["time_sec_arrays"],
                    "entries_arrays": stats_E_pairs["entries_arrays"],
                    "payload_arrays": stats_E_pairs["payload_arrays"],
                    "topk_pairs": stats_E_pairs["topk_pairs"],
                    "used_boundary_counts": stats_E_pairs["used_boundary_counts"]},
        "e_topk": {"time_sec": stats_D_topk["time_sec_n2topk"],
                   "entries": stats_D_topk["entries"],
                   "payload": stats_D_topk["payload"],
                   "topk": stats_D_topk["topk"]},
    }
    stats["totals"] = {
        "build_A_sec": stats["inside"]["time_sec"] + stats["g_out"]["time_sec"] + stats["outside_pll"]["time_sec"],
        "build_B_sec": stats["inside"]["time_sec"] + stats["g_out"]["time_sec"] + stats["virtual_graph"]["time_sec"] + stats["hybrid_b"]["time_sec_pairs"],
        "build_C_sec": stats["inside"]["time_sec"] + stats["g_out"]["time_sec"] + stats["virtual_graph"]["time_sec"]
                       + stats["c_topk"]["time_sec"] + stats["c_used_arrays"]["time_sec"],
        "build_D_sec": stats["inside"]["time_sec"] + stats["g_out"]["time_sec"]
                       + stats["d_potentials"]["time_sec"] + stats["d_topk_u"]["time_sec"],
        "build_E_sec": stats["inside"]["time_sec"] + stats["g_out"]["time_sec"]
                       + stats["d_potentials"]["time_sec"] + stats["e_pairs"]["time_sec_select"] + stats["e_pairs"]["time_sec_arrays"] + stats["e_topk"]["time_sec"],

        "payload_A_bytes": stats["inside"]["payload"] + stats["g_out"]["payload_nodes"] + stats["g_out"]["payload_edges"] + stats["outside_pll"]["payload"],
        "payload_B_bytes": stats["inside"]["payload"] + stats["g_out"]["payload_nodes"] + stats["g_out"]["payload_edges"] + stats["hybrid_b"]["payload"],
        "payload_C_bytes": stats["inside"]["payload"] + stats["g_out"]["payload_nodes"] + stats["g_out"]["payload_edges"]
                           + stats["c_topk"]["payload"] + stats["c_used_arrays"]["payload"],
        "payload_D_bytes": stats["inside"]["payload"] + stats["g_out"]["payload_nodes"] + stats["g_out"]["payload_edges"]
                           + stats["d_potentials"]["payload"] + stats["d_topk_u"]["payload"],
        "payload_E_bytes": stats["inside"]["payload"] + stats["g_out"]["payload_nodes"] + stats["g_out"]["payload_edges"]
                           + stats["e_pairs"]["payload_arrays"] + stats["e_topk"]["payload"],
    }

    return (partition, clusters, inside_pll, boundary_sets, node_cluster,
            G_out, outside_pll,
            # B
            cluster_pair_info,
            # C
            pf_by_source, AB_candidates,
            node_to_used_arr_C, used_b_list_C, used_b_index_C,
            # D(original)
            node_to_topk_D, boundary_to_cluster_D,
            # E
            d_pairs_for_E, node_to_used_arr_D_for_E, used_b_list_D_for_E, used_b_index_D_for_E, node_to_topk_E,
            stats)


# ===============================
# Pretty print stats
# ===============================
def print_build_stats(stats):
    print("\n=== Build Time Breakdown ===")
    rows = [
        ["Inside PLL (sum over clusters)", stats["inside"]["time_sec"]],
        ["Build G_out (boundary graph)", stats["g_out"]["time_sec"]],
        ["Outside PLL on G_out  [A] (deg-desc)", stats["outside_pll"]["time_sec"]],
        ["Virtual graph build  [B/C]", stats["virtual_graph"]["time_sec"]],
        ["All cluster-pair shortest  [B]", stats["hybrid_b"]["time_sec_pairs"]],

        [f"C: SSSP per source (TopK_AB, k={stats['c_topk']['k_pairs']})", stats["c_topk"]["time_sec"]],
        ["C: Build node→USED-boundary arrays", stats["c_used_arrays"]["time_sec"]],

        ["D: Potentials per target (multi-source)", stats["d_potentials"]["time_sec"]],
        [f"D: Build node→TopK(u) (topk={stats['d_topk_u']['topk']})", stats["d_topk_u"]["time_sec"]],

        [f"E: Select TopK_AB pairs (k={stats['e_pairs']['topk_pairs']})", stats["e_pairs"]["time_sec_select"]],
        ["E: Build node→USED-boundary arrays", stats["e_pairs"]["time_sec_arrays"]],
        [f"E: Build node→TopK(u) (topk={stats['e_topk']['topk']})", stats["e_topk"]["time_sec"]],

        ["Total Build A", stats["totals"]["build_A_sec"]],
        ["Total Build B", stats["totals"]["build_B_sec"]],
        ["Total Build C", stats["totals"]["build_C_sec"]],
        ["Total Build D", stats["totals"]["build_D_sec"]],
        ["Total Build E", stats["totals"]["build_E_sec"]],
    ]
    df_time = pd.DataFrame(rows, columns=["Step", "Time (sec)"])
    print(df_time.to_string(index=False))

    print("\n=== Storage Summary (payload-only, rough) ===")
    rows2 = [
        ["Inside PLL labels", stats["inside"]["label_entries"], format_mib(stats["inside"]["payload"])],
        ["G_out nodes", stats["g_out"]["nodes"], format_mib(stats["g_out"]["payload_nodes"])],
        ["G_out edges", stats["g_out"]["edges"], format_mib(stats["g_out"]["payload_edges"])],
        ["Outside PLL labels  [A]", stats["outside_pll"]["label_entries"], format_mib(stats["outside_pll"]["payload"])],
        ["Cluster-pair records  [B]", stats["hybrid_b"]["pairs"], format_mib(stats["hybrid_b"]["payload"])],

        [f"C: Kept (S,t) entries (TopK_AB)", stats["c_topk"]["kept_entries"], format_mib(stats["c_topk"]["payload"])],
        ["C: Node→USED-boundary entries", stats["c_used_arrays"]["entries"], format_mib(stats["c_used_arrays"]["payload"])],

        ["D: Potentials entries (b→B)", stats["d_potentials"]["entries"], format_mib(stats["d_potentials"]["payload"])],
        [f"D: Node→TopK(u) entries (topk={stats['d_topk_u']['topk']})", stats["d_topk_u"]["entries"], format_mib(stats["d_topk_u"]["payload"])],

        [f"E: Node→USED-boundary entries", stats["e_pairs"]["entries_arrays"], format_mib(stats["e_pairs"]["payload_arrays"])],
        [f"E: Kept (A,B) pairs (TopK_AB)", stats["e_pairs"]["kept_pairs"], ""],
        [f"E: Node→TopK(u) entries (topk={stats['e_topk']['topk']})", stats["e_topk"]["entries"], format_mib(stats["e_topk"]["payload"])],

        ["Total payload  [A]", "", format_mib(stats["totals"]["payload_A_bytes"])],
        ["Total payload  [B]", "", format_mib(stats["totals"]["payload_B_bytes"])],
        ["Total payload  [C]", "", format_mib(stats["totals"]["payload_C_bytes"])],
        ["Total payload  [D]", "", format_mib(stats["totals"]["payload_D_bytes"])],
        ["Total payload  [E]", "", format_mib(stats["totals"]["payload_E_bytes"])],
    ]
    df_mem = pd.DataFrame(rows2, columns=["Component", "Count", "Payload (MiB)"])
    print(df_mem.to_string(index=False))


# ===============================
# Main
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Pubmed", choices=["Cora", "CiteSeer", "Pubmed"])
    parser.add_argument("--data_root", type=str, default=None, help="Persistent root folder for Planetoid data (e.g., ./data)")
    parser.add_argument("--pairs", type=int, default=500, help="number of query pairs for evaluation")
    parser.add_argument("--resolution", type=float, default=0.3, help="Leiden resolution")
    parser.add_argument("--max_cluster_size", type=int, default=1000, help="Leiden max_cluster_size trigger")
    parser.add_argument("--max_workers", type=int, default=None, help="parallel processes for per-cluster PLL")

    # 并行度
    parser.add_argument("--c_workers", type=int, default=None, help="parallel workers for C (per-source SSSP & USED arrays)")
    parser.add_argument("--d_workers", type=int, default=None, help="parallel workers for D/E (potentials & USED arrays)")

    # Top-K
    parser.add_argument("--topk_c_pairs", type=int, default=16, help="Top-K kept routes per (source, target) pair in C")
    parser.add_argument("--topk_d_u", type=int, default=16, help="Top-K personalized exits per node for D(original) and E")
    parser.add_argument("--topk_e_pairs", type=int, default=16, help="Top-K (A,B) pairs kept for E's global good routes")

    args = parser.parse_args()

    # Load graph
    G = load_planetoid_graph(args.dataset, root=args.data_root)
    print(f"[INFO] Graph: {args.dataset}, |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

    # Build all structures
    (partition, clusters, inside_pll, boundary_sets, node_cluster,
     G_out, outside_pll,
     # B
     cluster_pair_info,
     # C
     pf_by_source, AB_candidates,
     node_to_used_arr_C, used_b_list_C, used_b_index_C,
     # D(original)
     node_to_topk_D, boundary_to_cluster_D,
     # E
     d_pairs_for_E, node_to_used_arr_D_for_E, used_b_list_D_for_E, used_b_index_D_for_E, node_to_topk_E,
     stats) = build_all(G, args)

    sizes = Counter(partition.values())
    top_sizes = sizes.most_common(10)
    print(f"[INFO] #final_clusters = {len(sizes)}, Top-10 sizes: {top_sizes[:10]}")
    print(f"[INFO] G_out: |V|={G_out.number_of_nodes()}, |E|={G_out.number_of_edges()}")

    # Build/Storage stats
    print_build_stats(stats)

    # Evaluate A/B/C/D(original)/E
    df_eval = evaluate(
        G,
        inside_pll, boundary_sets, node_cluster, outside_pll,
        cluster_pair_info,
        # C:
        pf_by_source, AB_candidates, node_to_used_arr_C, used_b_index_C,
        # D(original):
        node_to_topk_D, boundary_to_cluster_D,
        # E:
        d_pairs_for_E, node_to_used_arr_D_for_E, used_b_index_D_for_E, node_to_topk_E,
        n_pairs=args.pairs
    )
    print("\n=== Evaluation (A vs B vs C vs D(original) vs E) ===")
    print(df_eval.to_string(index=False))


if __name__ == "__main__":
    main()
