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
# Build G_out (boundary graph) — needed by potentials & virtual graph
# ===============================
def build_g_out(G, partition, clusters, boundary_sets, boundary_pairs):
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

    stats = {
        "time_sec_build_g_out": t_build_gout,
        "nodes": G_out.number_of_nodes(),
        "edges": G_out.number_of_edges(),
        "intra_super_edges_added": intra_super_edges,
        "payload_bytes_nodes": payload_bytes_nodes(G_out.number_of_nodes()),
        "payload_bytes_edges": payload_bytes_edges(G_out.number_of_edges()),
    }
    return G_out, stats


# ===============================
# Build virtual graph (for C2)
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
# Adjacency builder
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
# C2: per-source Dijkstra on G_rep, keep Top-K (S->B)
# ===============================
def _c2_per_source_worker(args):
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

    topk_per_B = defaultdict(list)  # B -> [(-dist_mid, b_exit, t_entry)]
    visited_external_boundary = set()

    while heap:
        d, _, x = heapq.heappop(heap)
        if d != dist.get(x, float("inf")):
            continue

        if is_virtual(x) and x != vnode:
            continue

        if (not is_virtual(x)) and partition[x] != source_cid:
            if x not in visited_external_boundary:
                visited_external_boundary.add(x)
                B = partition[x]
                b_exit = first_exit.get(x, None)
                if b_exit is not None:
                    h = topk_per_B[B]
                    entry = (-d, b_exit, x)
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
    for B, h in topk_per_B.items():
        items = [(-negd, b_exit, t_entry) for (negd, b_exit, t_entry) in h]
        items.sort(key=lambda z: z[0])
        out[B] = items
    return source_cid, out


def build_c2_pairs_bfs_like(G_rep, boundary_sets, partition, topk_pairs=16, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count() or 2

    adj_rep = _make_adj(G_rep)
    cids = sorted(boundary_sets.keys())
    args_list = [(S, adj_rep, boundary_sets, partition, topk_pairs) for S in cids]

    C2_pairs = defaultdict(lambda: defaultdict(list))
    used = defaultdict(set)
    kept_total = 0

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_c2_per_source_worker, a) for a in args_list]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="C2: per-source Dijkstra (TopK per (S,B))", unit="cluster"):
            S, perB = fut.result()
            for B, items in perB.items():
                C2_pairs[S][B] = items
                kept_total += len(items)
                for (dist_mid, b_exit, t_entry) in items:
                    used[S].add(b_exit)
                    used[B].add(t_entry)
    t_pairs = time.time() - t0

    stats = {
        "time_sec_c2_pairs": t_pairs,
        "topk_pairs": topk_pairs,
        "kept_total": kept_total,
        "payload_pairs": payload_bytes_triple(kept_total),
    }
    return C2_pairs, used, stats


# ===============================
# Shared: USED arrays (node -> used-boundary distances)
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


def build_used_arrays_from_c2_pairs(inside_pll, clusters, used, max_workers=None):
    return build_node_to_boundary_used_arrays(inside_pll, clusters, used, max_workers=max_workers)


# ===============================
# D-style potentials for E (multi-source on G_out)
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
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Potentials per target (for E)", unit="cluster"):
            B, out = fut.result()
            for b, val in out.items():
                b2c[b][B] = val
    t = time.time() - t0

    entries = sum(len(m) for m in b2c.values())
    stats = {"time_sec_potentials": t, "entries": entries, "payload": payload_bytes_pair_id_dist(entries)}
    return b2c, stats


# ===============================
# Node -> TopK(u) exits (for E)
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
# E: 生成 (b, φ, c) 候选 + USED arrays
# ===============================
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
        "topk_pairs": topk_pairs,
    }
    return pairs, node_to_used_arr, used_b_list, used_b_index, stats


# ===============================
# Queries
# ===============================
def query_method_without_topk_u(u, v, node_cluster,
                                C2_pairs,  # A -> {B: [(mid, b, t), ...]}
                                node_to_used_arr_C2, used_index_C2,
                                inside_pll):
    """
    Method_Without_TopK_u (原 C2):
      仅使用 per-source TopK_AB 候选 + USED 回表
      (不并入 TopK(u))
    """
    A = node_cluster[u]
    B = node_cluster[v]
    if A == B:
        return inside_pll[A].query(u, v)

    cand = C2_pairs.get(A, {}).get(B, [])
    if not cand:
        return float("inf")

    arr_u = node_to_used_arr_C2.get(A, {}).get(u)
    arr_v = node_to_used_arr_C2.get(B, {}).get(v)
    if arr_u is None or arr_v is None:
        return float("inf")

    idxA = used_index_C2.get(A, {})
    idxB = used_index_C2.get(B, {})

    best = float("inf")
    for (mid, b, t) in cand:
        iA = idxA.get(b);  iB = idxB.get(t)
        if iA is None or iB is None:
            continue
        du = arr_u[iA]
        dv = arr_v[iB]
        val = du + mid + dv
        if val < best:
            best = val
    return best


def query_method_with_topk_u(u, v, node_cluster,
                             d_pairs, node_to_used_arr_D, used_index_D,
                             node_to_topk_E,
                             boundary_to_cluster_D,
                             inside_pll):
    """
    Method_With_TopK_u (原 E):
      候选 = D 的 (b, φ, c) TopK_AB  ∪  每点 TopK(u)
      左端/右端优先 USED 回表；退化用 TopK(u) 或簇内 PLL
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

    arr_u = node_to_used_arr_D.get(A, {}).get(u)
    arr_v = node_to_used_arr_D.get(B, {}).get(v)
    idxA = used_index_D.get(A, {})
    idxB = used_index_D.get(B, {})

    best = float("inf")
    for (b, phi, c) in cand_pairs:
        # 左端 du
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

        # 右端 dv
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
# Evaluate (Only two methods kept)
# ===============================
def evaluate(G,
             inside_pll, boundary_sets, node_cluster,
             # C2:
             C2_pairs, node_to_used_arr_C2, used_index_C2,
             # With TopK(u):
             d_pairs, node_to_used_arr_D, used_index_D, node_to_topk_E, boundary_to_cluster_D,
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

    # Method_Without_TopK_u (C2)
    correct = total = 0
    err = 0.0
    t0 = time.time()
    for u, v, d in gt:
        est = query_method_without_topk_u(u, v, node_cluster,
                                          C2_pairs, node_to_used_arr_C2, used_index_C2,
                                          inside_pll)
        if est == d:
            correct += 1
        if est != float("inf"):
            err += abs(est - d); total += 1
    tC2 = time.time() - t0
    rows.append(["Method_Without_TopK_u (C2: TopK_AB + USED)", tC2, total, correct, (err/total if total>0 else float("inf"))])

    # Method_With_TopK_u (E)
    correct = total = 0
    err = 0.0
    t0 = time.time()
    for u, v, d in gt:
        est = query_method_with_topk_u(u, v, node_cluster,
                                       d_pairs, node_to_used_arr_D, used_index_D, node_to_topk_E, boundary_to_cluster_D,
                                       inside_pll)
        if est == d:
            correct += 1
        if est != float("inf"):
            err += abs(est - d); total += 1
    tE = time.time() - t0
    rows.append(["Method_With_TopK_u (E: TopK_AB ∪ TopK(u) + mixed tables)", tE, total, correct, (err/total if total>0 else float("inf"))])

    return pd.DataFrame(rows, columns=["method", "query_time_sec", "samples", "exact_matches", "mae"])


# ===============================
# Build all (only components needed by C2 & E)
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

    # Build boundary graph G_out (for potentials & virtual graph)
    G_out, stats_gout = build_g_out(G, partition, clusters, boundary_sets, boundary_pairs)

    # C2: build virtual graph & per-source TopK_AB + USED arrays
    G_rep, virtual_of, t_vgraph = build_virtual_graph(G_out, boundary_sets)
    C2_pairs, used_c2, stats_C2_pairs = build_c2_pairs_bfs_like(
        G_rep, boundary_sets, partition, topk_pairs=args.topk_c_pairs, max_workers=args.c_workers
    )
    node_to_used_arr_C2, used_b_list_C2, used_b_index_C2, stats_C2_used = build_used_arrays_from_c2_pairs(
        inside_pll, clusters, used_c2, max_workers=args.c_workers
    )

    # E: potentials (b -> B) + TopK(u) + USED arrays for E
    boundary_to_cluster_D, stats_D_pot = build_potentials_per_target_parallel(
        G_out, boundary_sets, max_workers=args.d_workers
    )
    node_to_topk_E, stats_topkE = build_node_to_topk_boundary(
        inside_pll, clusters, boundary_sets, topk=args.topk_d_u
    )
    d_pairs_for_E, node_to_used_arr_D_for_E, used_b_list_D_for_E, used_b_index_D_for_E, stats_E_pairs = build_d_pairs_and_used_arrays(
        boundary_to_cluster_D, boundary_sets, inside_pll, clusters,
        topk_pairs=args.topk_e_pairs, max_workers=args.d_workers
    )

    # ===== stats aggregation (only related parts) =====
    stats = {
        "graph": {"V": G.number_of_nodes(), "E": G.number_of_edges()},
        "partition": {"clusters": len(clusters),
                      "boundary_total": sum(len(s) for s in boundary_sets.values())},
        "inside": {"time_sec": inside_time_sum,
                   "label_entries": inside_label_entries_sum,
                   "payload": payload_bytes_labels_entry_count(inside_label_entries_sum)},
        "g_out": {"nodes": stats_gout["nodes"], "edges": stats_gout["edges"],
                  "time_sec": stats_gout["time_sec_build_g_out"],
                  "payload_nodes": stats_gout["payload_bytes_nodes"],
                  "payload_edges": stats_gout["payload_bytes_edges"]},

        # C2
        "virtual_graph": {"time_sec": t_vgraph},
        "c2_pairs": {"time_sec": stats_C2_pairs["time_sec_c2_pairs"],
                     "topk_pairs": stats_C2_pairs["topk_pairs"],
                     "kept_total": stats_C2_pairs["kept_total"],
                     "payload": stats_C2_pairs["payload_pairs"]},
        "c2_used_arrays": {"time_sec": stats_C2_used["time_sec"],
                           "entries": stats_C2_used["entries"],
                           "payload": stats_C2_used["payload"],
                           "used_boundary_counts": stats_C2_used["used_boundary_counts"]},

        # E
        "d_potentials": {"time_sec": stats_D_pot["time_sec_potentials"],
                         "entries": stats_D_pot["entries"],
                         "payload": stats_D_pot["payload"]},
        "e_pairs": {"time_sec_select": stats_E_pairs["time_sec_select_pairs"],
                    "kept_pairs": stats_E_pairs["kept_pairs"],
                    "time_sec_arrays": stats_E_pairs["time_sec_arrays"],
                    "entries_arrays": stats_E_pairs["entries_arrays"],
                    "payload_arrays": stats_E_pairs["payload_arrays"],
                    "topk_pairs": stats_E_pairs["topk_pairs"],
                    "used_boundary_counts": stats_E_pairs["used_boundary_counts"]},
        "e_topk": {"time_sec": stats_topkE["time_sec_n2topk"],
                   "entries": stats_topkE["entries"],
                   "payload": stats_topkE["payload"],
                   "topk": stats_topkE["topk"]},
    }

    stats["totals"] = {
        "build_C_sec": stats["inside"]["time_sec"] + stats["g_out"]["time_sec"] + stats["virtual_graph"]["time_sec"]
                       + stats["c2_pairs"]["time_sec"] + stats["c2_used_arrays"]["time_sec"],
        "build_E_sec": stats["inside"]["time_sec"] + stats["g_out"]["time_sec"]
                       + stats["d_potentials"]["time_sec"] + stats["e_pairs"]["time_sec_select"]
                       + stats["e_pairs"]["time_sec_arrays"] + stats["e_topk"]["time_sec"],

        "payload_C_bytes": stats["inside"]["payload"] + stats["g_out"]["payload_nodes"] + stats["g_out"]["payload_edges"]
                           + stats["c2_pairs"]["payload"] + stats["c2_used_arrays"]["payload"],
        "payload_E_bytes": stats["inside"]["payload"] + stats["g_out"]["payload_nodes"] + stats["g_out"]["payload_edges"]
                           + stats["e_pairs"]["payload_arrays"] + stats["e_topk"]["payload"],
    }

    return (partition, clusters, inside_pll, boundary_sets, node_cluster,
            G_out,
            # C2
            C2_pairs, node_to_used_arr_C2, used_b_list_C2, used_b_index_C2,
            # E
            boundary_to_cluster_D,  # potentials map (b -> {B: (phi, c)})
            d_pairs_for_E, node_to_used_arr_D_for_E, used_b_list_D_for_E, used_b_index_D_for_E, node_to_topk_E,
            stats)


# ===============================
# Pretty print stats (only C & E)
# ===============================
def print_build_stats(stats):
    print("\n=== Build Time Breakdown (Only C & E) ===")
    rows = [
        ["Inside PLL (sum over clusters)", stats["inside"]["time_sec"]],
        ["Build G_out (boundary graph)", stats["g_out"]["time_sec"]],
        ["Virtual graph build  [C]", stats["virtual_graph"]["time_sec"]],
        [f"C: per-source Dijkstra (TopK_AB, k={stats['c2_pairs']['topk_pairs']})", stats["c2_pairs"]["time_sec"]],
        ["C: Build node→USED-boundary arrays", stats["c2_used_arrays"]["time_sec"]],
        ["E: Potentials per target (multi-source)", stats["d_potentials"]["time_sec"]],
        [f"E: Select TopK_AB pairs (k={stats['e_pairs']['topk_pairs']})", stats["e_pairs"]["time_sec_select"]],
        ["E: Build node→USED-boundary arrays", stats["e_pairs"]["time_sec_arrays"]],
        [f"E: Build node→TopK(u) (topk={stats['e_topk']['topk']})", stats["e_topk"]["time_sec"]],
        ["Total Build C", stats["totals"]["build_C_sec"]],
        ["Total Build E", stats["totals"]["build_E_sec"]],
    ]
    df_time = pd.DataFrame(rows, columns=["Step", "Time (sec)"])
    print(df_time.to_string(index=False))

    print("\n=== Storage Summary (payload-only, rough) ===")
    rows2 = [
        ["Inside PLL labels", stats["inside"]["label_entries"], format_mib(stats["inside"]["payload"])],
        ["G_out nodes", stats["g_out"]["nodes"], format_mib(stats["g_out"]["payload_nodes"])],
        ["G_out edges", stats["g_out"]["edges"], format_mib(stats["g_out"]["payload_edges"])],
        [f"C: Kept (A,B) entries (TopK_AB)", stats["c2_pairs"]["kept_total"], format_mib(stats["c2_pairs"]["payload"])],
        ["C: Node→USED-boundary entries", stats["c2_used_arrays"]["entries"], format_mib(stats["c2_used_arrays"]["payload"])],
        ["E: Potentials entries (b→B)", stats["d_potentials"]["entries"], format_mib(stats["d_potentials"]["payload"])],
        [f"E: Node→USED-boundary entries", stats["e_pairs"]["entries_arrays"], format_mib(stats["e_pairs"]["payload_arrays"])],
        [f"E: Kept (A,B) pairs (TopK_AB)", stats["e_pairs"]["kept_pairs"], ""],
        [f"E: Node→TopK(u) entries (topk={stats['e_topk']['topk']})", stats["e_topk"]["entries"], format_mib(stats["e_topk"]["payload"])],
        ["Total payload  [C]", "", format_mib(stats["totals"]["payload_C_bytes"])],
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
    parser.add_argument("--pairs", type=int, default=1000, help="number of query pairs for evaluation")
    parser.add_argument("--resolution", type=float, default=0.3, help="Leiden resolution")
    parser.add_argument("--max_cluster_size", type=int, default=1000, help="Leiden max_cluster_size trigger")
    parser.add_argument("--max_workers", type=int, default=None, help="parallel processes for per-cluster PLL")

    # 并行度
    parser.add_argument("--c_workers", type=int, default=None, help="parallel workers for C (per-source Dijkstra & USED arrays)")
    parser.add_argument("--d_workers", type=int, default=None, help="parallel workers for E (potentials & USED arrays)")

    # Top-K
    parser.add_argument("--topk_c_pairs", type=int, default=16, help="Top-K kept routes per (source, target) pair in C2")
    parser.add_argument("--topk_d_u", type=int, default=16, help="Top-K personalized exits per node for E")
    parser.add_argument("--topk_e_pairs", type=int, default=16, help="Top-K (A,B) pairs kept for E's global good routes")

    args = parser.parse_args()

    # Load graph
    G = load_planetoid_graph(args.dataset, root=args.data_root)
    print(f"[INFO] Graph: {args.dataset}, |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

    # Build only parts needed by C & E
    (partition, clusters, inside_pll, boundary_sets, node_cluster,
     G_out,
     # C2
     C2_pairs, node_to_used_arr_C2, used_b_list_C2, used_b_index_C2,
     # E
     boundary_to_cluster_D,
     d_pairs_for_E, node_to_used_arr_D_for_E, used_b_list_D_for_E, used_b_index_D_for_E, node_to_topk_E,
     stats) = build_all(G, args)

    sizes = Counter(partition.values())
    top_sizes = sizes.most_common(10)
    print(f"[INFO] #final_clusters = {len(sizes)}, Top-10 sizes: {top_sizes[:10]}")
    print(f"[INFO] G_out: |V|={G_out.number_of_nodes()}, |E|={G_out.number_of_edges()}")

    # Build/Storage stats
    print_build_stats(stats)

    # Evaluate (Only two methods)
    df_eval = evaluate(
        G,
        inside_pll, boundary_sets, node_cluster,
        # C2:
        C2_pairs, node_to_used_arr_C2, used_b_index_C2,
        # E:
        d_pairs_for_E, node_to_used_arr_D_for_E, used_b_index_D_for_E, node_to_topk_E, boundary_to_cluster_D,
        n_pairs=args.pairs
    )
    print("\n=== Evaluation (Method_Without_TopK_u vs Method_With_TopK_u) ===")
    print(df_eval.to_string(index=False))


if __name__ == "__main__":
    main()
