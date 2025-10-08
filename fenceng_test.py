# fenceng_test.py
# ------------------------------------------------------------
# 只跑 E 方法（two-table）
# - 构图（改进版）：不再生成 ("REL", i, r) 关系节点；每个三元组 (h,r,t)
#   直接收缩为实体—实体边 (h,t)，边权 = 所有关系的最小权重（默认 1.0）。
#   同时把该实体对上的全部关系与权重存入边属性：
#     - rels: 关系类型元组（排序）
#     - w_by_rel: {关系: 权重}
#     - multi: 该实体对上的关系种类数
# - 构建：Leiden 分簇 + 簇内 PLL + 边界图 + E 方法两张表
# - 评测：仅抽样实体→实体（默认限制在 LCC 内），保证可达样本
# - 新增：支持将 (source,target,dist) 真值保存/加载（避免重复计算最短路）
# - 修复：node_sort_key 统一比较键；AB_topk 构造中的变量遮蔽问题
# - 输出：表格包含 preprocessing_time（构建总时长，秒）
# ------------------------------------------------------------

import argparse
import os
import time
import random
import heapq
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm

try:
    from torch_geometric.datasets import Planetoid
except Exception:
    Planetoid = None

from graspologic.partition import hierarchical_leiden


# ---------- helpers ----------
def is_rel(n) -> bool:
    """保留接口：当前不再使用关系节点，恒为 False（兼容旧评测逻辑）。"""
    return isinstance(n, tuple) and len(n) > 0 and n[0] == "REL"

def node_sort_key(n):
    """统一的节点排序键：确保 str 与 ("REL", i, r) 可比较，避免 TypeError"""
    if is_rel(n):
        tag, idx, rel = n
        return (1, int(idx), str(rel))
    else:
        return (0, 0, str(n))


# ---------- data loaders ----------
def load_planetoid_graph(name="Pubmed", root=None):
    if Planetoid is None:
        raise RuntimeError("torch_geometric 未安装；请安装它或使用 --kg_file 加载 WN18。")
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
        G.add_edge(u, v, weight=1.0, rels=("PLANETOID",), w_by_rel={"PLANETOID": 1.0}, multi=1)
    return G


def load_wn18_graph_aggregated(path: str, rel_weight: dict | None = None):
    """
    改进版构图：将每条 triple (h, t, r) 收缩为无向实体边 (h, t)，
    边权 = min_r w_r（默认每种关系 w_r=1.0，对应原先 0.5+0.5）。
    同时把该实体对的全部关系与其权重保留在边属性中。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"WN18 file not found: {path}")

    rel_weight = rel_weight or {}  # 可自定义某些关系的代价

    pair2rels: dict[tuple, dict] = {}

    def add_triple(h, t, r):
        if h == t:
            return
        # 无向合并（保持与旧版无向 REL 构图的距离等价）
        a, b = (h, t) if str(h) <= str(t) else (t, h)
        w = float(rel_weight.get(r, 1.0))
        pair2rels.setdefault((a, b), {})
        # 若同关系多条，取最小代价
        pair2rels[(a, b)][r] = min(w, pair2rels[(a, b)].get(r, float("inf")))

    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        # 第一行可能是 triple 数量；如果不是数字，则当作一条 triple
        try:
            _ = int(first.strip())
        except Exception:
            parts = first.strip().split()
            if len(parts) >= 2:
                h, t = parts[0], parts[1]
                r = parts[2] if len(parts) >= 3 else "NA"
                add_triple(h, t, r)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            h, t = parts[0], parts[1]
            r = parts[2] if len(parts) >= 3 else "NA"
            add_triple(h, t, r)

    G = nx.Graph()
    entities = set()
    for (h, t), rmap in pair2rels.items():
        entities.add(h); entities.add(t)
    G.add_nodes_from(entities)

    for (h, t), rmap in pair2rels.items():
        # 边权 = 该对实体的最小关系代价（保持实体最短路与旧法等价）
        w_min = min(rmap.values()) if rmap else 1.0
        rels_sorted = tuple(sorted(rmap.keys(), key=str))
        G.add_edge(
            h, t,
            weight=w_min,
            rels=rels_sorted,
            w_by_rel={k: float(v) for k, v in rmap.items()},
            multi=len(rmap),
        )
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
        for v in tqdm(self.order, desc="Building PLL", unit="node"):
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


# ---------- partition & per-cluster PLL ----------
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


def _build_one_cluster(args):
    """
    Return: (cid, nodes, boundary_list, boundary_pairs_dict, labels_dict, build_time, label_entries)
    （构建阶段不裁剪边界）
    """
    cid, nodes, edges, neigh_map = args
    nodes_set = set(nodes)

    subg = nx.Graph()
    subg.add_nodes_from(nodes)
    for u, v, w in edges:
        subg.add_edge(u, v, weight=float(w))

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

    # 边界点（不裁剪）
    boundary = [n for n in nodes if any((nbr not in nodes_set) for nbr in neigh_map[n])]

    # 簇内边界点对距离
    pairs = {}
    for i in range(len(boundary)):
        ui = boundary[i]
        for j in range(i + 1, len(boundary)):
            vj = boundary[j]
            pairs[(ui, vj)] = pll.query(ui, vj)

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
            w = float(data.get("weight", 1.0))
            if G_out.has_edge(u, v):
                if w < G_out[u][v].get("weight", float("inf")):
                    G_out[u][v]["weight"] = w
            else:
                G_out.add_edge(u, v, weight=w)

    # intra-cluster super edges (boundary pairs)
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
    max_workers = max_workers or (os.cpu_count() or 2)
    adj_out = _make_adj(G_out)
    all_boundary = list(G_out.nodes())
    args_list = [(B, adj_out, boundary_sets, all_boundary) for B in boundary_sets.keys()]
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
    max_workers = max_workers or (os.cpu_count() or 2)

    allb_list = {}
    allb_index = {}
    node_to_allb_arr = {}

    tasks = []
    for cid, pll in inside_pll.items():
        blist = sorted(boundary_sets.get(cid, set()), key=node_sort_key)  # ★ 关键修复：稳定顺序
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
    """
    为 (A,B) 组装 “b -> (mid,c)” 的 Top-K 候选（按 mid 升序）。
    修复了原实现中的局部变量遮蔽与死分支问题。
    """
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
            curr = items_by_AB[A][B]
            curr.sort(key=lambda x: x[0])
            keep = curr[:topk_mid]
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


# ---------- sampling & gt ----------
def sample_entity_pairs(G, n_pairs=1000, in_lcc=True, rng_seed=42):
    """只抽样实体→实体；尽量在同一连通分量抽，确保可达"""
    entity_nodes = [n for n in G.nodes() if not is_rel(n)]
    if not entity_nodes:
        raise RuntimeError("图中没有实体节点，无法评测。")

    comps = list(nx.connected_components(G))
    if in_lcc and comps:
        lcc = max(comps, key=len)
        entity_nodes = [n for n in entity_nodes if n in lcc]

    if len(entity_nodes) < 2:
        raise RuntimeError("候选实体节点过少，无法评测。请关闭 LCC 限制或检查图构建。")

    # bucket by CC
    node2cc = {}
    for i, cc in enumerate(comps):
        for n in cc:
            node2cc[n] = i

    from collections import defaultdict as dd
    bucket = dd(list)
    for n in entity_nodes:
        bucket[node2cc[n]].append(n)
    weighted_bins = [(cid, len(nodes)) for cid, nodes in bucket.items() if len(nodes) >= 2]
    if not weighted_bins:
        raise RuntimeError("没有包含≥2个实体节点的连通分量，无法评测。")

    cids, sizes = zip(*weighted_bins)
    total = sum(sizes)
    probs = [s/total for s in sizes]

    rng = random.Random(rng_seed)
    pairs = []
    trials = 0
    max_trials = n_pairs * 10
    while len(pairs) < n_pairs and trials < max_trials:
        trials += 1
        cid = rng.choices(cids, weights=probs, k=1)[0]
        nodes_in_cc = bucket[cid]
        if len(nodes_in_cc) < 2:
            continue
        u, v = rng.sample(nodes_in_cc, 2)
        pairs.append((u, v))
    return pairs


def compute_gt_distances(G, pairs):
    """给定 (u,v) 列表，计算真值最短路（加权）"""
    gt = []
    for u, v in pairs:
        d = nx.shortest_path_length(G, u, v, weight="weight")
        gt.append((u, v, d))
    return gt


def save_eval_set_csv(path, gt):
    """保存 CSV: columns = source, target, dist"""
    df = pd.DataFrame(gt, columns=["source", "target", "dist"])
    df.to_csv(path, index=False)
    print(f"[INFO] Saved eval set with ground-truth to: {path}  (rows={len(df)})")


def load_eval_set_csv(path, G, max_rows=None):
    """读取 CSV 并过滤出图中存在的节点；返回 [(u,v,d)]"""
    df = pd.read_csv(path)
    if not {"source", "target", "dist"}.issubset(df.columns):
        raise ValueError("Eval set CSV must contain columns: source,target,dist")
    pairs = []
    present = set(G.nodes())
    for _, row in df.iterrows():
        u, v, d = row["source"], row["target"], float(row["dist"])
        if u in present and v in present:
            pairs.append((u, v, d))
        if max_rows and len(pairs) >= max_rows:
            break
    print(f"[INFO] Loaded eval set from {path}: rows_used={len(pairs)}")
    return pairs


# ---------- evaluation using precomputed gt ----------
def evaluate_with_gt(
    G,
    gt,  # list of (u,v,dist)
    inside_pll, node_cluster,
    node_to_allb_arr_E, allb_index_E, allb_list_E,
    boundary_to_cluster_D, AB_topk,
    topk_e_u, topk_e_mid,
    preprocessing_time: float = 0.0,
):
    correct = total_eval = 0
    err = 0.0
    t0 = time.time()
    for u, v, d in gt:
        est = query_cross_with_all_boundary_tables(
            u, v, node_cluster,
            node_to_allb_arr_E, allb_index_E, allb_list_E,
            boundary_to_cluster_D, AB_topk,
            topk_e_u, topk_e_mid,
            inside_pll
        )
        if est == d:
            correct += 1
        if est != float("inf"):
            err += abs(est - d); total_eval += 1
    tE = time.time() - t0

    rows = [[
        f"E (entities-only, u-TopK={topk_e_u}, mid-TopK={topk_e_mid})",
        tE, len(gt), correct, (err/total_eval if total_eval>0 else float("inf")),
        preprocessing_time
    ]]
    return pd.DataFrame(rows, columns=[
        "method", "query_time_sec", "samples", "exact_matches", "mae", "preprocessing_time"
    ])


# ---------- build all (E only) ----------
def build_all_e(G, args):
    t0 = time.time()

    hl = hierarchical_leiden(
        G,
        max_cluster_size=args.max_cluster_size,
        resolution=args.resolution,
        use_modularity=True,
        random_seed=42,
        check_directed=True,
    )
    partition = final_partition_from_hl(hl, G)

    clusters = defaultdict(list)
    for n, cid in partition.items():
        clusters[cid].append(n)

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

    with ProcessPoolExecutor(max_workers=args.max_workers or (os.cpu_count() or 2)) as ex:
        it = as_completed([ex.submit(_build_one_cluster, t) for t in tasks])
        it = tqdm(it, total=len(tasks), desc="Clusters (parallel PLL)", unit="cluster")
        for fut in it:
            cid, nodes, boundary, pairs, labels, _, _ = fut.result()
            inside_pll[cid] = PrunedPLLIndex.from_labels(labels)
            boundary_sets[cid] = set(boundary)
            for n in nodes:
                node_cluster[n] = cid
            boundary_pairs[cid] = pairs

    G_out = build_g_out(G, partition, clusters, boundary_sets, boundary_pairs)

    boundary_to_cluster_D = build_potentials_per_target_parallel(
        G_out, boundary_sets, max_workers=args.d_workers
    )
    node_to_allb_arr_E, allb_list_E, allb_index_E = build_node_to_all_boundary_arrays(
        inside_pll, clusters, boundary_sets, max_workers=args.d_workers
    )
    AB_topk = build_ab_topk_from_boundary_to_cluster(
        boundary_to_cluster_D, boundary_sets, topk_mid=args.topk_e_mid
    )

    preprocessing_time = time.time() - t0

    return (partition, clusters, inside_pll, boundary_sets, node_cluster,
            node_to_allb_arr_E, allb_list_E, allb_index_E, boundary_to_cluster_D, AB_topk,
            preprocessing_time)


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Pubmed", choices=["Cora", "CiteSeer", "Pubmed"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--kg_file", type=str, default=None, help="Path to WN18.txt（提供则覆盖 --dataset）")

    parser.add_argument("--eval_pairs", type=int, default=500, help="评测样本数（只抽实体→实体）")
    parser.add_argument("--limit_to_lcc", action="store_true",
                        help="评测仅在 LCC 内抽样（默认启用，若不加则代码里会置 True）")
    parser.add_argument("--save_eval_set", type=str, default=None,
                        help="保存抽样得到的 (source,target,dist) CSV 路径")
    parser.add_argument("--load_eval_set", type=str, default=None,
                        help="从 CSV 加载 (source,target,dist) 作为评测集合（优先于抽样与真值计算）")

    parser.add_argument("--resolution", type=float, default=0.3)
    parser.add_argument("--max_cluster_size", type=int, default=1200)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--d_workers", type=int, default=None, help="workers for E")
    parser.add_argument("--topk_e_u", type=int, default=8, help="E: u 的最近出口 Top-K")
    parser.add_argument("--topk_e_mid", type=int, default=8, help="E: (A,B) 的中段 Top-K")

    args = parser.parse_args()

    # 默认：评测限制在 LCC
    if not args.limit_to_lcc:
        args.limit_to_lcc = True

    # load graph
    if args.kg_file:
        G = load_wn18_graph_aggregated(args.kg_file)  # ★ 改为聚合版加载
        src_name = f"WN18({os.path.basename(args.kg_file)})-aggregated"
    else:
        G = load_planetoid_graph(args.dataset, root=args.data_root)
        src_name = args.dataset

    num_rel_nodes = sum(1 for n in G.nodes() if is_rel(n))  # 兼容打印；现在应为 0
    print(f"[INFO] Graph: {src_name}, |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}, REL_nodes={num_rel_nodes}")

    # build (E only)
    (partition, clusters, inside_pll, boundary_sets, node_cluster,
     node_to_allb_arr_E, allb_list_E, allb_index_E, boundary_to_cluster_D, AB_topk,
     preprocessing_time) = build_all_e(G, args)

    sizes = Counter(partition.values())
    top_sizes = sizes.most_common(10)
    print(f"[INFO] #final_clusters = {len(sizes)}, Top-10 sizes: {top_sizes[:10]}")

    # --- prepare evaluation ground-truth ---
    if args.load_eval_set and os.path.isfile(args.load_eval_set):
        gt = load_eval_set_csv(args.load_eval_set, G, max_rows=args.eval_pairs)
        if not gt:
            raise RuntimeError(f"Loaded eval set has 0 usable rows: {args.load_eval_set}")
    else:
        # 抽样 + 计算真值
        pairs = sample_entity_pairs(G, n_pairs=args.eval_pairs, in_lcc=args.limit_to_lcc, rng_seed=42)
        print(f"[INFO] Sampling done. computing ground-truth distances for {len(pairs)} pairs ...")
        gt = compute_gt_distances(G, pairs)
        if args.save_eval_set:
            save_eval_set_csv(args.save_eval_set, gt)

    # evaluate with precomputed gt
    df_eval = evaluate_with_gt(
        G,
        gt,
        inside_pll, node_cluster,
        node_to_allb_arr_E, allb_index_E, allb_list_E, boundary_to_cluster_D, AB_topk,
        args.topk_e_u, args.topk_e_mid,
        preprocessing_time=preprocessing_time,
    )
    print("\n=== Entities→Entities Evaluation (E method) ===")
    print(df_eval.to_string(index=False))


if __name__ == "__main__":
    main()
