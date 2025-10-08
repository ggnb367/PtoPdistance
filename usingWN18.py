
# ce_compare.py
# ------------------------------------------------------------
# C / E 两种方法对比（KG: entity 与 relation 均为节点；每条 triple 拆为两条半权重边）
# - 读取 WN18.txt：第一行 triple 数，之后每行: head tail relation
# - 构图：为每条 triple 生成一个独立关系节点 ("REL", i, relation)
#         head --0.5--> ("REL", i, r) --0.5--> tail
# - 其余流程（分簇/PLL/C/E评测）保持不变
# ------------------------------------------------------------

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

# Planetoid 仅在不提供 --kg_file 时可用
try:
    from torch_geometric.datasets import Planetoid
except Exception:
    Planetoid = None

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
# Loaders
# ===============================
def load_planetoid_graph(name="Pubmed", root=None):
    if Planetoid is None:
        raise RuntimeError("torch_geometric not available; install it or use --kg_file.")
    root = root or os.path.abspath(f"./data/{name}")
    dataset = Planetoid(root=root, name=name)
    data = dataset[0]
    edge_index = data.edge_index.numpy()
    G = nx.Graph()
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


def load_wn18_graph_relation_nodes(path: str):
    """
    从 WN18.txt 构建图：entity 与 relation 都是节点。
    每条 triple (h, t, r) 生成唯一关系节点 ("REL", i, r)，并连两条 0.5 权重边：
        h --0.5--> ("REL", i, r) --0.5--> t
    这样：
      - 保留所有 triples（包括多重关系）
      - 实体到实体的一跳距离仍是 1（0.5+0.5）
      - 不会把同一种 relation 全并为一个大 hub，避免产生“捷径”
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"WN18 file not found: {path}")

    triples = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        # 若首行不是数字，把它当作第一条 triple
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
                continue  # 丢弃自环
            triples.append((h, t, r))

    G = nx.Graph()
    # 确保实体节点先加入（非必须，但清晰）
    entities = set()
    for h, t, _ in triples:
        entities.add(h); entities.add(t)
    G.add_nodes_from(entities)

    # 为每条 triple 添加独立关系节点，避免跨样本“短路”
    for i, (h, t, r) in enumerate(triples):
        rel_node = ("REL", i, r)
        G.add_node(rel_node)
        G.add_edge(h, rel_node, weight=0.5)
        G.add_edge(rel_node, t, weight=0.5)

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

    # 计算分量内离心率确定顺序（失败则回退原顺序）
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

    # 边界点：在原图中与簇外相邻
    boundary = [n for n in nodes if any((nbr not in nodes_set) for nbr in neigh_map[n])]

    # 簇内边界点对最短路（用簇内PLL查询）
    pairs = {}
    for i in range(len(boundary)):
        ui = boundary[i]
        for j in range(i + 1, len(boundary)):
            vj = boundary[j]
            pairs[(ui, vj)] = pll.query(ui, vj)

    label_entries = sum(len(m) for m in pll.labels.values())
    return cid, nodes, boundary, pairs, pll.labels, build_time, label_entries


# ===============================
# Build boundary graph G_out
# ===============================
def build_g_out(G, partition, clusters, boundary_sets, boundary_pairs):
    outside_nodes = set()
    for cid in clusters:
        outside_nodes |= boundary_sets.get(cid, set())

    G_out = nx.Graph()
    G_out.add_nodes_from(outside_nodes)

    # 跨簇原始边
    for u, v, data in G.edges(data=True):
        if u in outside_nodes and v in outside_nodes and partition[u] != partition[v]:
            G_out.add_edge(u, v, weight=float(data.get("weight", 1.0)))

    # 簇内边界点“超边”
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


def build_virtual_graph(G_out, boundary_sets):
    G_rep = G_out.copy()
    for cid, bset in boundary_sets.items():
        vnode = ("VIRT", cid)
        G_rep.add_node(vnode)
        for b in bset:
            G_rep.add_edge(vnode, b, weight=0.0)
    return G_rep


def _make_adj(G):
    adj = {}
    for u in G.nodes():
        lst = []
        for v, data in G[u].items():
            lst.append((v, float(data.get("weight", 1.0))))
        adj[u] = lst
    return adj


# ===============================
# Method C: used-boundary tables
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
                heapq.heush(heap, (nd, ctr, y)); ctr += 1  # typo fix below

    out = {}
    for B, h in topk_per_B.items():
        items = [(-negd, b_exit, t_entry) for (negd, b_exit, t_entry) in h]
        items.sort(key=lambda z: z[0])
        out[B] = items
    return source_cid, out



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

    topk_per_B = defaultdict(list)
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

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_c2_per_source_worker, a) for a in args_list]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="C: per-source Dijkstra (TopK per (A,B))", unit="cluster"):
            S, perB = fut.result()
            for B, items in perB.items():
                C2_pairs[S][B] = items
                kept_total += len(items)
                for (dist_mid, b_exit, t_entry) in items:
                    used[S].add(b_exit)
                    used[B].add(t_entry)

    stats = {"kept_total": kept_total}
    return C2_pairs, used, stats


def _build_used_arrays_one_cluster(args):
    cid, nodes, used_b_list, labels = args
    pll = PrunedPLLIndex.from_labels(labels)
    arr_map = {}
    L = len(used_b_list)
    if L == 0:
        for n in nodes:
            arr_map[n] = np.zeros((0,), dtype=np.float64)
        return cid, arr_map
    for n in nodes:
        arr = np.empty((L,), dtype=np.float64)
        for i, b in enumerate(used_b_list):
            arr[i] = pll.query(n, b)
        arr_map[n] = arr
    return cid, arr_map


def build_node_to_boundary_used_arrays(inside_pll, clusters, used_boundaries_sets, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count() or 2

    used_boundary_list = {}
    used_boundary_index = {}
    node_to_used_arr = {}

    tasks = []
    for cid, pll in inside_pll.items():
        blist = sorted(used_boundaries_sets.get(cid, set()))
        used_boundary_list[cid] = blist
        used_boundary_index[cid] = {b: i for i, b in enumerate(blist)}
        tasks.append((cid, clusters[cid], blist, pll.labels))

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_build_used_arrays_one_cluster, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="C: Build node→USED-boundary arrays", unit="cluster"):
            cid, arr_map = fut.result()
            node_to_used_arr[cid] = arr_map

    return node_to_used_arr, used_boundary_list, used_boundary_index


def query_cross_with_used_boundary_tables(u, v, node_cluster,
                                          C2_pairs, node_to_used_arr_C, used_index_C,
                                          inside_pll):
    A = node_cluster[u]
    B = node_cluster[v]
    if A == B:
        return inside_pll[A].query(u, v)

    cand = C2_pairs.get(A, {}).get(B, [])
    if not cand:
        return float("inf")

    arr_u = node_to_used_arr_C.get(A, {}).get(u)
    arr_v = node_to_used_arr_C.get(B, {}).get(v)
    if arr_u is None or arr_v is None:
        return float("inf")

    idxA = used_index_C.get(A, {})
    idxB = used_index_C.get(B, {})

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


# ===============================
# Method E: all-boundary two tables
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


def _make_adj(G):
    adj = {}
    for u in G.nodes():
        lst = []
        for v, data in G[u].items():
            lst.append((v, float(data.get("weight", 1.0))))
        adj[u] = lst
    return adj


def build_potentials_per_target_parallel(G_out, boundary_sets, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count() or 2
    adj_out = _make_adj(G_out)
    all_boundary = list(G_out.nodes())
    args_list = [(B, adj_out, boundary_sets, all_boundary) for B in sorted(boundary_sets.keys())]
    b2c = defaultdict(dict)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_d_potential_one_target_worker, a) for a in args_list]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="E: Potentials per target", unit="cluster"):
            B, out = fut.result()
            for b, val in out.items():
                b2c[b][B] = val
    return b2c


def _build_all_boundary_arrays_one_cluster(args):
    cid, nodes, all_b_list, labels = args
    pll = PrunedPLLIndex.from_labels(labels)
    arr_map = {}
    L = len(all_b_list)
    for n in nodes:
        arr = np.empty((L,), dtype=np.float64)
        for i, b in enumerate(all_b_list):
            arr[i] = pll.query(n, b)
        arr_map[n] = arr
    return cid, arr_map


def build_node_to_all_boundary_arrays(inside_pll, clusters, boundary_sets, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count() or 2

    allb_list = {}
    allb_index = {}
    node_to_allb_arr = {}

    tasks = []
    for cid, pll in inside_pll.items():
        blist = sorted(boundary_sets.get(cid, set()))
        allb_list[cid] = blist
        allb_index[cid] = {b: i for i, b in enumerate(blist)}
        tasks.append((cid, clusters[cid], blist, pll.labels))

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_build_all_boundary_arrays_one_cluster, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="E: Build node→ALL-boundary arrays", unit="cluster"):
            cid, arr_map = fut.result()
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

    arr_u = node_to_allb_arr.get(A, {}).get(u)
    if arr_u is None or arr_u.size == 0:
        return float("inf")
    blistA = allb_list.get(A, [])
    k = min(topk_e_u, len(arr_u))
    idx = np.argpartition(arr_u, k-1)[:k]
    left_candidates = [(blistA[i], float(arr_u[i])) for i in idx]
    left_candidates.sort(key=lambda x: x[1])

    mid_list = AB_topk.get(A, {}).get(B, [])
    if topk_e_mid > 0 and mid_list:
        mid_list = mid_list[:topk_e_mid]
    allow_map = {bb: (mid, c) for (bb, mid, c) in mid_list} if mid_list else None

    arr_v = node_to_allb_arr.get(B, {}).get(v)
    if arr_v is None:
        return float("inf")
    blistB = allb_list.get(B, [])

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


# ===============================
# Evaluate C / E
# ===============================
def evaluate_ce(G,
                inside_pll, node_cluster,
                C2_pairs, node_to_used_arr_C, used_index_C,
                node_to_allb_arr_E, allb_index_E, allb_list_E, boundary_to_cluster_D, AB_topk,
                topk_e_u, topk_e_mid,
                n_pairs=1000):
    nodes = list(G.nodes())
    if len(nodes) == 0:
        raise RuntimeError("Empty graph.")
    pairs = [(random.choice(nodes), random.choice(nodes)) for _ in range(n_pairs)]

    gt = []
    for u, v in pairs:
        try:
            d = nx.shortest_path_length(G, u, v, weight="weight")
            gt.append((u, v, d))
        except nx.NetworkXNoPath:
            pass
    if not gt:
        raise RuntimeError("No connected pairs found for evaluation.")

    rows = []

    # C
    correct = total = 0
    err = 0.0
    t0 = time.time()
    for u, v, d in gt:
        est = query_cross_with_used_boundary_tables(u, v, node_cluster,
                                                    C2_pairs, node_to_used_arr_C, used_index_C,
                                                    inside_pll)
        if est == d:
            correct += 1
        if est != float("inf"):
            err += abs(est - d); total += 1
    tC = time.time() - t0
    rows.append(["C: used-boundary tables only", tC, total, correct, (err/total if total>0 else float("inf"))])

    # E
    correct = total = 0
    err = 0.0
    t0 = time.time()
    for u, v, d in gt:
        est = query_cross_with_all_boundary_tables(u, v, node_cluster,
                                                   node_to_allb_arr_E, allb_index_E, allb_list_E,
                                                   boundary_to_cluster_D, AB_topk,
                                                   topk_e_u, topk_e_mid,
                                                   inside_pll)
        if est == d:
            correct += 1
        if est != float("inf"):
            err += abs(est - d); total += 1
    tE = time.time() - t0
    rows.append([f"E: all-boundary tables (u-TopK={topk_e_u}, mid-TopK={topk_e_mid})", tE, total, correct, (err/total if total>0 else float("inf"))])

    return pd.DataFrame(rows, columns=["method", "query_time_sec", "samples", "exact_matches", "mae"])


# ===============================
# Build all (C & E)
# ===============================
def build_all_ce(G, args):
    # 分簇
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

    # 簇内 PLL（并行）
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

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        it = as_completed([ex.submit(_build_one_cluster, t) for t in tasks])
        it = tqdm(it, total=len(tasks), desc="Clusters (parallel PLL)", unit="cluster")
        for fut in it:
            cid, nodes, boundary, pairs, labels, _, _ = fut.result()
            inside_pll[cid] = PrunedPLLIndex.from_labels(labels)
            boundary_sets[cid] = set(boundary)
            for n in nodes:
                node_cluster[n] = cid
            boundary_pairs[cid] = pairs

    # 边界图
    G_out = build_g_out(G, partition, clusters, boundary_sets, boundary_pairs)

    # ----- C -----
    G_rep = build_virtual_graph(G_out, boundary_sets)
    C2_pairs, used_c, _ = build_c2_pairs_bfs_like(
        G_rep, boundary_sets, partition, topk_pairs=args.topk_c_pairs, max_workers=args.c_workers
    )
    node_to_used_arr_C, used_b_list_C, used_index_C = build_node_to_boundary_used_arrays(
        inside_pll, clusters, used_c, max_workers=args.c_workers
    )

    # ----- E -----
    boundary_to_cluster_D = build_potentials_per_target_parallel(
        G_out, boundary_sets, max_workers=args.d_workers
    )
    node_to_allb_arr_E, allb_list_E, allb_index_E = build_node_to_all_boundary_arrays(
        inside_pll, clusters, boundary_sets, max_workers=args.d_workers
    )
    AB_topk = build_ab_topk_from_boundary_to_cluster(
        boundary_to_cluster_D, boundary_sets, topk_mid=args.topk_e_mid
    )

    return (partition, clusters, inside_pll, boundary_sets, node_cluster,
            C2_pairs, node_to_used_arr_C, used_b_list_C, used_index_C,
            node_to_allb_arr_E, allb_list_E, allb_index_E, boundary_to_cluster_D, AB_topk)


# ===============================
# Main
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Pubmed", choices=["Cora", "CiteSeer", "Pubmed"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--kg_file", type=str, default=None, help="Path to WN18.txt. If set, overrides --dataset.")
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--resolution", type=float, default=0.3)
    parser.add_argument("--max_cluster_size", type=int, default=1000)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--c_workers", type=int, default=None)
    parser.add_argument("--d_workers", type=int, default=None)
    parser.add_argument("--topk_c_pairs", type=int, default=16)
    parser.add_argument("--topk_e_u", type=int, default=16)
    parser.add_argument("--topk_e_mid", type=int, default=16)

    args = parser.parse_args()

    # 载图
    if args.kg_file:
        G = load_wn18_graph_relation_nodes(args.kg_file)
        src_name = f"WN18({os.path.basename(args.kg_file)})-rel_nodes"
    else:
        G = load_planetoid_graph(args.dataset, root=args.data_root)
        src_name = args.dataset

    print(f"[INFO] Graph: {src_name}, |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

    # 构建
    (partition, clusters, inside_pll, boundary_sets, node_cluster,
     C2_pairs, node_to_used_arr_C, used_b_list_C, used_index_C,
     node_to_allb_arr_E, allb_list_E, allb_index_E, boundary_to_cluster_D, AB_topk) = build_all_ce(G, args)

    sizes = Counter(partition.values())
    top_sizes = sizes.most_common(10)
    print(f"[INFO] #final_clusters = {len(sizes)}, Top-10 sizes: {top_sizes[:10]}")

    # 评测
    df_eval = evaluate_ce(
        G,
        inside_pll, node_cluster,
        C2_pairs, node_to_used_arr_C, used_index_C,
        node_to_allb_arr_E, allb_index_E, allb_list_E, boundary_to_cluster_D, AB_topk,
        args.topk_e_u, args.topk_e_mid,
        n_pairs=args.pairs
    )
    print("\n=== Evaluation (C: used-boundary tables) vs (E: all-boundary two-table) ===")
    print(df_eval.to_string(index=False))


if __name__ == "__main__":
    main()
