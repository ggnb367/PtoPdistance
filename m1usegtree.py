# -*- coding: utf-8 -*-
"""
gtree_m1.py  (CC-parallel HL + SUPER_ROOT + no-pruning + star-closure)

本版要点：
- 不再删除任何代表点（无预算/无裁剪），父层代表域 = parent.global_borders ∪ 子簇跨边端点全集。
- 闭包统一改为“星型”，即为每个子簇在父层代表域中的门户选一个中心，与同簇的其他门户相连（权重来自叶D_pp或子簇代表图）。
- CC 并行分层Leiden：每个连通分量独立HL，level=-1 为 CC_ROOT，level=-2 为 SUPER_ROOT（不构建代表图；不同CC查询直接 inf）。
- 叶子：PLL + TopK + 叶 D_pp(PLL) + D_np（精确）。
- 查询：在 LCA 层一次 multi-source（带 base）完成。

仍保留进度输出：
- 每个叶子：打印边界数
- 每个父簇：打印代表图节点数
"""

from __future__ import annotations
import argparse
import math
import os
import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm

import pickle
import urllib.request
import ssl
import heapq

try:
    from graspologic.partition import hierarchical_leiden
except Exception:
    hierarchical_leiden = None

INF = np.float32(np.inf)

# ----------------------------- utils -----------------------------
def is_rel(n) -> bool:
    return isinstance(n, tuple) and len(n) > 0 and n[0] == "REL"

def node_sort_key(n):
    if is_rel(n):
        try:
            _, idx, rel = n
            return (1, int(idx), str(rel))
        except Exception:
            return (1, 0, str(n))
    return (0, 0, str(n))

def _download(url: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx) as resp, open(path, "wb") as f:
        f.write(resp.read())

class HLItem:
    __slots__ = ("node", "level", "cluster")
    def __init__(self, node, level, cluster):
        self.node = node
        self.level = level
        self.cluster = cluster

# ----------------------------- loaders -----------------------------
def load_planetoid_pubmed_graph(root_dir: str = "./data/planetoid") -> nx.Graph:
    os.makedirs(root_dir, exist_ok=True)
    base = "https://github.com/kimiyoung/planetoid/raw/master/data"
    fname = "ind.pubmed.graph"
    url = f"{base}/{fname}"
    path = os.path.join(root_dir, fname)
    print(f"[Downloader] fetching {url} -> {path}")
    _download(url, path)

    with open(path, "rb") as f:
        try:
            graph = pickle.load(f, encoding="latin1")
        except TypeError:
            graph = pickle.load(f)

    G = nx.Graph()
    nodes = set(graph.keys())
    for nbrs in graph.values():
        nodes.update(nbrs)
    G.add_nodes_from(nodes)
    for u, nbrs in graph.items():
        for v in nbrs:
            if u == v:
                continue
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=1.0)
    return G

def load_wn18_graph_relation_nodes(path: str) -> nx.Graph:
    """
    读取 WN18.txt（可能首行是条数）。构图：每条 (h, t, r) 生成关系节点 ("REL", i, r)，
    然后 h --0.5-- REL --0.5-- t
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

    def to_int(x):
        try:
            return int(x)
        except Exception:
            return x

    G = nx.Graph()
    for i, (h, t, r) in enumerate(triples):
        h_i = to_int(h); t_i = to_int(t)
        rel_node = ("REL", i, to_int(r))
        G.add_edge(h_i, rel_node, weight=0.5)
        G.add_edge(rel_node, t_i, weight=0.5)
    return G

def load_edge_list_graph(path: str) -> nx.Graph:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Edge list file not found: {path}")
    G = nx.Graph()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            try:
                u = int(u); v = int(v)
            except Exception:
                pass
            w = 1.0
            if len(parts) >= 3:
                try:
                    w = float(parts[2])
                except Exception:
                    w = 1.0
            if u == v:
                continue
            if G.has_edge(u, v):
                if w < float(G[u][v].get("weight", 1.0)):
                    G[u][v]["weight"] = float(w)
            else:
                G.add_edge(u, v, weight=float(w))
    return G

# ----------------------------- HL worker (parallel per-CC) -----------------------------
def _hl_on_cc_worker(args):
    """
    对一个 CC 诱导子图跑 hierarchical_leiden。
    args: (comp_idx, nodes_cc, edges_cc, params_dict)
    return: (comp_idx, [(node, level, cluster_id), ...])
    """
    comp_idx, nodes_cc, edges_cc, params = args
    weight_attr = params["weight_attr"]
    H = nx.Graph()
    H.add_nodes_from(nodes_cc)
    for u, v, w in edges_cc:
        if u == v:
            continue
        if H.has_edge(u, v):
            if w < H[u][v].get(weight_attr, math.inf):
                H[u][v][weight_attr] = w
        else:
            H.add_edge(u, v, **{weight_attr: w})
    hl = hierarchical_leiden(
        H,
        max_cluster_size=params["max_cluster_size"],
        resolution=params["resolution"],
        randomness=params["randomness"],
        use_modularity=params["use_modularity"],
        random_seed=params["random_seed"],
        weight_attribute=weight_attr,
        check_directed=True,
    )
    out = []
    for e in hl:
        out.append((e.node, e.level, e.cluster))
    return comp_idx, out

# ----------------------------- PLL -----------------------------
class PrunedPLLIndex:
    """Weighted PLL; 构建后仅保留 labels，用于 query（兜底/叶内精确）。"""
    def __init__(self, subg: Optional[nx.Graph], order: List[Any], weight_attr: str = "weight"):
        self.G = subg
        self.order = list(order)
        self.wname = weight_attr
        self.labels: Dict[Any, Dict[Any, float]] = {v: {v: 0.0} for v in (subg.nodes() if subg is not None else [])}

    @classmethod
    def from_labels(cls, labels: Dict[Any, Dict[Any, float]], weight_attr: str = "weight") -> "PrunedPLLIndex":
        obj = cls(subg=None, order=[], weight_attr=weight_attr)
        obj.labels = labels
        return obj

    def build(self):
        if self.G is None:
            return
        for pivot in self.order:
            dist = {pivot: 0.0}
            pq = [(0.0, pivot)]
            visited = set()
            while pq:
                d, x = heapq.heappop(pq)
                if x in visited:
                    continue
                visited.add(x)

                lx = self.labels[x]
                lp = self.labels.get(pivot, {pivot: 0.0})

                # 剪枝时跳过 pivot 自身
                pruned = False
                if len(lx) < len(lp):
                    for h, dxh in lx.items():
                        if h == pivot:
                            continue
                        dvh = lp.get(h)
                        if dvh is not None and dxh + dvh <= d:
                            pruned = True
                            break
                else:
                    for h, dvh in lp.items():
                        if h == pivot:
                            continue
                        dxh = lx.get(h)
                        if dxh is not None and dxh + dvh <= d:
                            pruned = True
                            break
                if pruned:
                    continue

                self.labels[x][pivot] = d
                for nbr, edata in self.G[x].items():
                    w = float(edata.get(self.wname, 1.0))
                    nd = d + w
                    if nd < dist.get(nbr, math.inf):
                        dist[nbr] = nd
                        heapq.heappush(pq, (nd, nbr))

        self.G = None
        self.order = []

    def query(self, u: Any, v: Any) -> float:
        if u == v:
            return 0.0
        lu = self.labels.get(u); lv = self.labels.get(v)
        if not lu or not lv:
            return math.inf
        if len(lu) > len(lv):
            lu, lv = lv, lu
        best = math.inf
        for h, du in lu.items():
            dv = lv.get(h)
            if dv is not None:
                s = du + dv
                if s < best:
                    best = s
        return best

# ----------------------------- Cluster -----------------------------
@dataclass
class Cluster:
    level: int
    cid: Any
    nodes: Set[Any] = field(default_factory=set)
    parent: Optional["Cluster"] = None
    children: List["Cluster"] = field(default_factory=list)
    is_leaf: bool = False

    # 该簇所属的 CC id（SUPER_ROOT 为 None）
    cc_id: Optional[int] = None

    # 全局边界（相对全图）
    global_borders: Set[Any] = field(default_factory=set)

    # 叶子：PLL + TopK + D_pp(PLL) + D_np
    pll: Optional[PrunedPLLIndex] = None
    leaf_portals: List[Any] = field(default_factory=list)
    idx_portal_in_leaf: Dict[Any, int] = field(default_factory=dict)
    leaf_nodes_order: List[Any] = field(default_factory=list)
    idx_node_in_leaf: Dict[Any, int] = field(default_factory=dict)
    topk_by_node: Dict[Any, List[Tuple[Any, float]]] = field(default_factory=dict)
    D_pp: Optional[np.ndarray] = None  # 叶：门户×门户（用 PLL 精确填充）
    D_np: Optional[np.ndarray] = None  # 叶：节点×门户（来自 PLL）

    # 非叶代表图
    rep_nodes: List[Any] = field(default_factory=list)
    idx_portal_in_rep: Dict[Any, int] = field(default_factory=dict)
    rep_edges: List[Tuple[int, int, float]] = field(default_factory=list)

    # 父层会用到的 child→portals 映射
    child_portals: Dict["Cluster", List[Any]] = field(default_factory=lambda: defaultdict(list))
    child_portal_idx_in_child: Dict["Cluster", List[int]] = field(default_factory=lambda: defaultdict(list))
    child_portal_idx_in_parent: Dict["Cluster", List[int]] = field(default_factory=lambda: defaultdict(list))

    # （可选）跨子簇（multi-source）
    cross_tables_best: Dict[Tuple["Cluster", "Cluster"], np.ndarray] = field(default_factory=dict)
    cross_row_nodes: Dict[Tuple["Cluster", "Cluster"], List[Any]] = field(default_factory=dict)
    cross_col_nodes: Dict[Tuple["Cluster", "Cluster"], List[Any]] = field(default_factory=dict)

    def key(self) -> Tuple[int, Any]:
        return (self.level, self.cid)

    def __hash__(self):
        return id(self)

# ----------------------------- helpers -----------------------------
def _build_adj_from_edges_int(P: int, rep_edges: List[Tuple[int, int, float]]) -> List[List[Tuple[int, float]]]:
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(P)]
    for i, j, w in rep_edges:
        if i == j:
            continue
        adj[i].append((j, w))
        adj[j].append((i, w))
    return adj

def _weighted_multisource_on_rep(P: int,
                                 adj_int: List[List[Tuple[int, float]]],
                                 sources_idx: List[int],
                                 base: Optional[np.ndarray] = None) -> np.ndarray:
    """
    weighted multi-source Dijkstra on a representative graph:
      - base is None  => all sources start from 0
      - base[k]       => initial distance of sources_idx[k]
    Returns: dist_all (float32, length P)
    """
    dist = np.full(P, np.inf, dtype=np.float64)
    pq: List[Tuple[float, int]] = []
    if base is None:
        for s in sources_idx:
            dist[s] = 0.0
            heapq.heappush(pq, (0.0, s))
    else:
        assert len(base) == len(sources_idx)
        for k, s in enumerate(sources_idx):
            d0 = float(base[k])
            dist[s] = d0
            heapq.heappush(pq, (d0, s))
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj_int[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist.astype(np.float32)

# ----------------------------- workers：叶子 -----------------------------
def _leaf_pll_topk_worker(args):
    """
    输入：
      key, nodes, edges, portals, leaf_topk, wname
    输出：
      key,
      labels (PLL),
      topk_map,
      D_pp(叶, 用 PLL 精确查询得到),
      node_order,
      D_np(节点×门户, 用 PLL 精确查询得到)
    """
    key, nodes_raw, edges, portals, leaf_topk, wname = args
    nodes = sorted(nodes_raw, key=node_sort_key)

    # 1) 子图 H
    H = nx.Graph()
    H.add_nodes_from(nodes)
    for u, v, w in edges:
        if u == v:
            continue
        if H.has_edge(u, v):
            if w < H[u][v].get(wname, math.inf):
                H[u][v][wname] = w
        else:
            H.add_edge(u, v, **{wname: w})

    # 2) PLL
    order = [u for u, _ in sorted(H.degree(), key=lambda kv: kv[1], reverse=True)]
    pll = PrunedPLLIndex(H, order, weight_attr=wname)
    pll.build()

    # 3) 叶 D_pp（用 PLL 精确）
    P = len(portals)
    D_pp = np.full((P, P), INF, dtype=np.float32)
    for i in range(P):
        pi = portals[i]
        for j in range(i, P):
            pj = portals[j]
            d = pll.query(pi, pj)
            val = np.float32(d if d != math.inf else np.inf)
            D_pp[i, j] = val
            D_pp[j, i] = val
        D_pp[i, i] = np.float32(0.0)

    # 4) Top-K 与 D_np（精确）
    topk_map: Dict[Any, List[Tuple[Any, float]]] = {}
    D_np = np.full((len(nodes), P), INF, dtype=np.float32)
    for r, u in enumerate(nodes):
        dists = []
        for c, p in enumerate(portals):
            d = pll.query(u, p)
            if d != math.inf:
                val = np.float32(d)
                D_np[r, c] = val
                dists.append((p, float(d)))
        dists.sort(key=lambda x: x[1])
        if leaf_topk > 0 and len(dists) > leaf_topk:
            dists = dists[:leaf_topk]
        topk_map[u] = dists

    return key, pll.labels, topk_map, D_pp, nodes, D_np

# ----------------------------- 索引构建器 -----------------------------
class HierRepHybridTopKPLL:
    def __init__(self, G: nx.Graph,
                 max_cluster_size: int = 1200,
                 resolution: float = 1.0,
                 randomness: float = 0.001,
                 use_modularity: bool = True,
                 random_seed: int = 42,
                 weight_attr: str = "weight",
                 leaf_topk: int = 8,
                 max_workers: Optional[int] = None,
                 debug_stats: bool = False,
                 build_cross_tables: bool = False):
        if hierarchical_leiden is None:
            raise RuntimeError("graspologic.partition.hierarchical_leiden not available. pip install graspologic")
        self.G = G
        self.wname = weight_attr
        self.leaf_topk = int(max(0, leaf_topk))
        self.max_workers = max_workers or (os.cpu_count() or 2)
        self.debug_stats = debug_stats
        self.build_cross_tables = bool(build_cross_tables)

        # 缓存
        self._memo_vec: Dict[Tuple[int, Any, int, int], np.ndarray] = {}
        self._adj_cache: Dict[int, List[List[Tuple[int, float]]]] = {}

        # 0) CC split + 并行 per-CC HL
        print("[Stage] connected components split + parallel hierarchical_leiden per CC ...")
        components = list(nx.connected_components(G))
        components.sort(key=lambda s: len(s), reverse=True)

        # 准备每个CC的边集，传给子进程
        cc_nodes_list: List[List[Any]] = []
        cc_edges_list: List[List[Tuple[Any, Any, float]]] = []
        for nodes_cc in components:
            nodes_cc = list(nodes_cc)
            edges_cc = [(u, v, float(d.get(self.wname, 1.0)))
                        for u, v, d in G.subgraph(nodes_cc).edges(data=True)]
            cc_nodes_list.append(nodes_cc)
            cc_edges_list.append(edges_cc)

        params = dict(
            max_cluster_size=max_cluster_size,
            resolution=resolution,
            randomness=randomness,
            use_modularity=use_modularity,
            random_seed=random_seed,
            weight_attr=weight_attr,
        )

        hl_all: List[HLItem] = []
        with ProcessPoolExecutor(max_workers=min(self.max_workers, max(1, len(components)))) as ex:
            futs = []
            for comp_idx, (nodes_cc, edges_cc) in enumerate(zip(cc_nodes_list, cc_edges_list)):
                futs.append(ex.submit(_hl_on_cc_worker, (comp_idx, nodes_cc, edges_cc, params)))
            for fut in tqdm(as_completed(futs), total=len(futs), desc="HL per-CC", unit="cc"):
                comp_idx, tuples = fut.result()
                for node, lvl, cid in tuples:
                    # 带上 CC 前缀
                    hl_all.append(HLItem(node, lvl, ('cc', comp_idx, cid)))

        print(f"[INFO] hierarchical_leiden finished on {len(components)} CCs; aggregating hierarchy ...")
        self.root, self.all_clusters, self.node2leaf = self._build_hierarchy_with_cc_roots(
            hl_all, components
        )
        print(f"[INFO] clusters (incl. SUPER_ROOT) = {len(self.all_clusters)}")

        # 1) 全局边界
        print("[Stage] compute GLOBAL borders ...")
        self._compute_global_borders()

        # 2) 叶子：PLL + TopK + 叶 D_pp（PLL）+ D_np
        print(f"[Stage] leaf PLL + TopK (K={self.leaf_topk}) + D_pp(PLL) + D_np (parallel) ...")
        self._build_leaf_pll_topk_parallel()

        # 3) 自底向上：父层代表图 +（可选）multi-source cross tables
        print("[Stage] bottom-up representative graphs (star closure, no pruning) ...")
        self._build_rep_graphs_and_cross_tables_multisource()

        if self.debug_stats:
            self._print_debug_stats()

    # ---------- 层次（带 CC_ROOT 与 SUPER_ROOT） ----------
    def _build_hierarchy_with_cc_roots(self, hl_list, components) -> Tuple[Cluster, List[Cluster], Dict[Any, Cluster]]:
        # 先按 HL 结果生成 level / cluster map
        node_levels: Dict[Any, List[Tuple[int, Any]]] = defaultdict(list)
        for e in hl_list:
            node_levels[e.node].append((e.level, e.cluster))
        for node in node_levels:
            node_levels[node].sort(key=lambda t: t[0])

        level_map: Dict[int, Dict[Any, Set[Any]]] = defaultdict(lambda: defaultdict(set))
        for node, mems in node_levels.items():
            for lvl, cid in mems:
                level_map[lvl][cid].add(node)

        clusters: Dict[Tuple[int, Any], Cluster] = {}
        for lvl, cid_map in level_map.items():
            for cid, nset in cid_map.items():
                cc_id = cid[1] if (isinstance(cid, tuple) and len(cid) >= 2 and cid[0] == 'cc') else None
                clusters[(lvl, cid)] = Cluster(level=lvl, cid=cid, nodes=set(nset), cc_id=cc_id)

        # parent-child（按 HL 层级）
        for node, mems in node_levels.items():
            for i in range(len(mems) - 1):
                (l1, c1), (l2, c2) = mems[i], mems[i + 1]
                p = clusters[(l1, c1)]
                ch = clusters[(l2, c2)]
                if ch not in p.children:
                    p.children.append(ch)
                    ch.parent = p

        # 叶子标注
        node2leaf: Dict[Any, Cluster] = {}
        for c in clusters.values():
            if not c.children:
                c.is_leaf = True
                for u in c.nodes:
                    node2leaf[u] = c

        # 为每个 CC 创建一个 CC_ROOT（level=-1）
        cc_roots: List[Cluster] = []
        all_nodes = set(self.G.nodes())
        for comp_idx, nodes_cc in enumerate(components):
            nodes_cc = set(nodes_cc)
            cc_root = Cluster(level=-1, cid=('ccroot', comp_idx), nodes=nodes_cc, is_leaf=False, cc_id=comp_idx)
            # 选出在该 CC 内、且 parent is None 的 level=0 簇，挂到 cc_root
            for (lvl, cid), c in clusters.items():
                if lvl == 0 and c.parent is None and (c.nodes & nodes_cc):
                    c.parent = cc_root
                    cc_root.children.append(c)
            cc_roots.append(cc_root)

        # SUPER_ROOT（level=-2）
        super_root = Cluster(level=-2, cid='SUPER_ROOT', nodes=all_nodes, is_leaf=False, cc_id=None)
        for cc_root in cc_roots:
            cc_root.parent = super_root
            super_root.children.append(cc_root)

        all_clusters = list(clusters.values()) + cc_roots + [super_root]
        return super_root, all_clusters, node2leaf

    # ---------- 全局边界 ----------
    def _compute_global_borders(self):
        G = self.G
        nbrs = {u: set(G.neighbors(u)) for u in tqdm(G.nodes(), desc="scan neighbors", unit="node")}
        for c in tqdm(self.all_clusters, desc="compute borders", unit="cluster"):
            borders = set()
            S = c.nodes
            for u in S:
                if any((v not in S) for v in nbrs[u]):
                    borders.add(u)
            c.global_borders = borders

    # ---------- 叶子：PLL + TopK + 叶 D_pp（PLL） + D_np（并行） ----------
    def _build_leaf_pll_topk_parallel(self):
        leaves = [c for c in self.all_clusters if c.is_leaf]
        key2cluster = {c.key(): c for c in leaves}
        tasks = []
        for c in leaves:
            nodes = list(c.nodes)
            edges = [(u, v, float(d.get(self.wname, 1.0)))
                     for u, v, d in self.G.subgraph(nodes).edges(data=True)]
            portals = sorted([p for p in c.global_borders if p in c.nodes], key=node_sort_key)
            c.leaf_portals = portals
            c.idx_portal_in_leaf = {p: i for i, p in enumerate(portals)}
            try:
                tqdm.write(f"[leaf] level={c.level} cid={c.cid} |nodes|={len(nodes)} borders={len(portals)}")
            except Exception:
                tqdm.write("[leaf] level={} cid={} |nodes|={} borders={}".format(c.level, c.cid, len(nodes), len(portals)))
            tasks.append((c.key(), nodes, edges, portals, self.leaf_topk, self.wname))

        results = {}
        if tasks:
            with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
                futs = [ex.submit(_leaf_pll_topk_worker, t) for t in tasks]
                for fut in tqdm(as_completed(futs), total=len(futs), desc="leaf PLL+TopK+Dpp(PLL)+Dnp", unit="leaf"):
                    key, labels, topk_map, D_pp, node_order, D_np = fut.result()
                    results[key] = (labels, topk_map, D_pp, node_order, D_np)

        for key, (labels, topk_map, D_pp, node_order, D_np) in results.items():
            c = key2cluster[key]
            c.pll = PrunedPLLIndex.from_labels(labels, weight_attr=self.wname)
            c.topk_by_node = topk_map
            c.D_pp = D_pp.astype(np.float32)   # 叶 D_pp（精确，来源 PLL）
            c.leaf_nodes_order = list(node_order)
            c.idx_node_in_leaf = {u: i for i, u in enumerate(c.leaf_nodes_order)}
            c.D_np = D_np.astype(np.float32)

    # ---------- 父层代表图 +（可选）cross tables ----------
    def _build_rep_graphs_and_cross_tables_multisource(self):
        levels = sorted({c.level for c in self.all_clusters})
        # 自底向上：高level到低level（例如 3,2,1,0,-1,-2）
        levels.sort(reverse=True)
        for lvl in levels:
            Cs = [c for c in self.all_clusters if c.level == lvl]
            if not Cs:
                continue

            # 跳过 level=-2（SUPER_ROOT）：不构建代表图与跨表
            if lvl == -2:
                continue

            # 先建代表图
            for c in tqdm(Cs, desc=f"rep-graph build (level={lvl})", unit="cluster"):
                self._build_single_rep_graph_star(c)

            # 可选：基于 multi-source 生成 cross tables（非查询必需）
            if self.build_cross_tables:
                for c in tqdm(Cs, desc=f"cross tables (multi-source) (level={lvl})", unit="cluster"):
                    if c.is_leaf:
                        continue
                    self._compute_parent_cross_tables_multisource(c)

    def _get_adj_cached(self, c: Cluster) -> List[List[Tuple[int, float]]]:
        key = id(c)
        if key in self._adj_cache:
            return self._adj_cache[key]
        adj = _build_adj_from_edges_int(len(c.rep_nodes), c.rep_edges)
        self._adj_cache[key] = adj
        return adj

    def _build_single_rep_graph_star(self, c: Cluster):
        """不删任何代表点；闭包统一为 star。"""
        # 叶子：代表域=叶门户；边=叶 D_pp 的 star
        if c.is_leaf:
            portals = c.leaf_portals
            c.rep_nodes = list(portals)
            c.idx_portal_in_rep = {p: i for i, p in enumerate(c.rep_nodes)}
            c.rep_edges = []
            P = len(c.rep_nodes)
            if c.D_pp is not None and P > 1:
                # 选择中心（简单用 0；你也可以改为度最大）
                center = 0
                for j in range(P):
                    if j == center:
                        continue
                    w = float(c.D_pp[center, j])
                    if math.isinf(w):
                        continue
                    c.rep_edges.append((center, j, w))
            return

        # 非叶：收集跨子簇边端点（全集） + 强制混入自身 global_borders
        children = c.children
        node2child: Dict[Any, Cluster] = {}
        for ch in children:
            for u in ch.nodes:
                node2child[u] = ch

        selected_ports: Dict[Cluster, Set[Any]] = defaultdict(set)
        cross_edges_raw: List[Tuple[Any, Any, float]] = []
        ext_deg: Dict[Any, int] = defaultdict(int)

        for u, v, d in self.G.edges(c.nodes, data=True):
            cu = node2child.get(u); cv = node2child.get(v)
            if cu is None or cv is None or cu == cv:
                continue
            w = float(d.get(self.wname, 1.0))

            # 端点都保留（不做任何裁剪）
            if cu.is_leaf:
                if u in cu.leaf_portals:
                    selected_ports[cu].add(u); ext_deg[u] += 1
            else:
                if u in getattr(cu, "rep_nodes", []):
                    selected_ports[cu].add(u); ext_deg[u] += 1

            if cv.is_leaf:
                if v in cv.leaf_portals:
                    selected_ports[cv].add(v); ext_deg[v] += 1
            else:
                if v in getattr(cv, "rep_nodes", []):
                    selected_ports[cv].add(v); ext_deg[v] += 1

            cross_edges_raw.append((u, v, w))

        # 代表域 = 全部跨边端点 ∪ c.global_borders（不删）
        rep_set = set(c.global_borders)
        for ch in children:
            rep_set |= selected_ports.get(ch, set())

        c.rep_nodes = sorted(rep_set, key=node_sort_key)
        c.idx_portal_in_rep = {p: i for i, p in enumerate(c.rep_nodes)}
        c.rep_edges = []

        try:
            tqdm.write(f"[parent] level={c.level} cid={c.cid} children={len(children)} rep_nodes={len(c.rep_nodes)}")
        except Exception:
            tqdm.write("[parent] level={} cid={} children={} rep_nodes={}".format(c.level, c.cid, len(c.rep_nodes)))

        # child→portals
        c.child_portals.clear()
        c.child_portal_idx_in_child.clear()
        c.child_portal_idx_in_parent.clear()

        # 每个子簇：parent域中的门户集合 & star 闭包
        for ch in children:
            ports = sorted([p for p in c.rep_nodes if p in ch.nodes], key=node_sort_key)
            c.child_portals[ch] = ports
            if not ports:
                c.child_portal_idx_in_child[ch] = []
                c.child_portal_idx_in_parent[ch] = []
                continue

            parent_idx = [c.idx_portal_in_rep[p] for p in ports]
            c.child_portal_idx_in_parent[ch] = parent_idx

            # 子簇内部索引
            if ch.is_leaf:
                idx_child = [ch.idx_portal_in_leaf.get(p, None) for p in ports]
                good = [(p, i) for p, i in zip(ports, idx_child) if i is not None]
                if not good:
                    c.child_portal_idx_in_child[ch] = []
                    continue
                ports, idx_child = map(list, zip(*good))
                parent_idx = [c.idx_portal_in_rep[p] for p in ports]
                c.child_portals[ch] = ports
                c.child_portal_idx_in_parent[ch] = parent_idx
                c.child_portal_idx_in_child[ch] = idx_child

                # 叶子：用 D_pp 做 star 闭包
                if len(ports) >= 2:
                    # 选择中心：可用外部度最高；退化用 0
                    center_local = 0
                    if len(ports) >= 2:
                        # pick highest ext degree if available
                        degs = [ext_deg.get(p, 0) for p in ports]
                        center_local = int(np.argmax(degs))
                    ic = parent_idx[center_local]
                    jc = idx_child[center_local]
                    for t in range(len(ports)):
                        if t == center_local:
                            continue
                        it = parent_idx[t]
                        jt = idx_child[t]
                        w = float(ch.D_pp[jc, jt])
                        if math.isinf(w):
                            continue
                        c.rep_edges.append((ic, it, w))
            else:
                idx_child = [ch.idx_portal_in_rep.get(p, None) for p in ports]
                good = [(p, i) for p, i in zip(ports, idx_child) if i is not None]
                if not good:
                    c.child_portal_idx_in_child[ch] = []
                    continue
                ports, idx_child = map(list, zip(*good))
                parent_idx = [c.idx_portal_in_rep[p] for p in ports]
                c.child_portals[ch] = ports
                c.child_portal_idx_in_parent[ch] = parent_idx
                c.child_portal_idx_in_child[ch] = idx_child

                # 非叶：用子簇代表图距离做 star 闭包
                if len(ports) >= 2:
                    center_local = 0
                    if len(ports) >= 2:
                        degs = [ext_deg.get(p, 0) for p in ports]
                        center_local = int(np.argmax(degs))
                    ic = parent_idx[center_local]
                    jc = idx_child[center_local]

                    if ch.D_pp is not None:
                        for t in range(len(ports)):
                            if t == center_local:
                                continue
                            it = parent_idx[t]
                            jt = idx_child[t]
                            w = float(ch.D_pp[jc, jt])
                            if math.isinf(w):
                                continue
                            c.rep_edges.append((ic, it, w))
                    else:
                        Pch = len(ch.rep_nodes)
                        if Pch > 0 and ch.rep_edges:
                            adj_ch = self._get_adj_cached(ch)
                            dist = _weighted_multisource_on_rep(Pch, adj_ch, [jc], base=None)
                            for t in range(len(ports)):
                                if t == center_local:
                                    continue
                                it = parent_idx[t]
                                jt = idx_child[t]
                                w = float(dist[jt])
                                if math.isinf(w):
                                    continue
                                c.rep_edges.append((ic, it, w))

        # 跨子簇原始边（两端在 rep_nodes 内）也保留
        rep_set = set(c.rep_nodes)
        for u, v, w in cross_edges_raw:
            if u in rep_set and v in rep_set:
                i = c.idx_portal_in_rep[u]; j = c.idx_portal_in_rep[v]
                c.rep_edges.append((i, j, w))

    def _compute_parent_cross_tables_multisource(self, parent: Cluster):
        if parent.is_leaf:
            return
        P = len(parent.rep_nodes)
        if P == 0 or not parent.rep_edges:
            return
        adj = _build_adj_from_edges_int(P, parent.rep_edges)
        children = [ch for ch in parent.children if parent.child_portals.get(ch)]
        for A in children:
            A_ports = parent.child_portals.get(A, [])
            A_idx   = parent.child_portal_idx_in_parent.get(A, [])
            if not A_ports or not A_idx:
                continue
            dist_all = _weighted_multisource_on_rep(P, adj, A_idx, base=None)
            for B in children:
                if B is A:
                    continue
                B_ports = parent.child_portals.get(B, [])
                B_idx   = parent.child_portal_idx_in_parent.get(B, [])
                if not B_ports or not B_idx:
                    continue
                dvec = dist_all[B_idx]
                parent.cross_tables_best[(A, B)] = dvec.astype(np.float32, copy=True)
                parent.cross_row_nodes[(A, B)] = list(A_ports)
                parent.cross_col_nodes[(A, B)] = list(B_ports)

    # ---------- Tree helpers ----------
    def _leaf_of(self, u: Any) -> Optional[Cluster]:
        return self.node2leaf.get(u)

    def _ancestors(self, c: Cluster) -> List[Cluster]:
        res = []
        cur = c
        while cur is not None:
            res.append(cur)
            cur = cur.parent
        return res

    def _lca(self, cu: Cluster, cv: Cluster) -> Cluster:
        Au = set(self._ancestors(cu))
        cur = cv
        while cur is not None:
            if cur in Au:
                return cur
            cur = cur.parent
        return self.root

    def _direct_child_under(self, P: Cluster, D: Cluster) -> Optional[Cluster]:
        cur = D
        while cur is not None and cur.parent is not P:
            cur = cur.parent
        return cur if cur is not None and cur.parent is P else None

    # ---------- 向量：实体 → parent.child 的 portals ----------
    def _vec_entity_to_parent_child_portals(self, u: Any, parent: Cluster, child: Cluster) -> np.ndarray:
        key = (id(parent), u, id(child), 0)
        if hasattr(self, "_memo_vec") and key in self._memo_vec:
            return self._memo_vec[key]

        RN_nodes = parent.child_portals.get(child, [])
        if not RN_nodes:
            v = np.full((0,), INF, dtype=np.float32)
            self._memo_vec[key] = v
            return v

        if child.is_leaf:
            res = np.full((len(RN_nodes),), INF, dtype=np.float32)
            if child.D_np is not None and u in child.idx_node_in_leaf:
                r = child.idx_node_in_leaf[u]
                for i, p in enumerate(RN_nodes):
                    cidx = child.idx_portal_in_leaf.get(p, None)
                    if cidx is None:
                        res[i] = np.float32(np.inf)
                    else:
                        res[i] = child.D_np[r, cidx]
                self._memo_vec[key] = res
                return res

            topk = child.topk_by_node.get(u, [])
            for i, p in enumerate(RN_nodes):
                d = None
                for (pp, dd) in topk:
                    if pp == p:
                        d = dd; break
                if d is None:
                    dval = child.pll.query(u, p) if child.pll is not None else math.inf
                else:
                    dval = d
                res[i] = np.float32(dval if dval != math.inf else np.inf)
            self._memo_vec[key] = res
            return res

        # 非叶：递归抬升到 D（child 的直系子簇）
        D = self._direct_child_under(child, self._leaf_of(u))
        if D is None:
            v = np.full((len(RN_nodes),), INF, dtype=np.float32)
            self._memo_vec[key] = v
            return v

        vec_lower = self._vec_entity_to_parent_child_portals(u, child, D)
        rows_parent = child.child_portal_idx_in_parent.get(D, [])
        if not rows_parent or vec_lower.size != len(rows_parent):
            v = np.full((len(RN_nodes),), INF, dtype=np.float32)
            self._memo_vec[key] = v
            return v

        Pch = len(child.rep_nodes)
        if Pch == 0 or not child.rep_edges:
            v = np.full((len(RN_nodes),), INF, dtype=np.float32)
            self._memo_vec[key] = v
            return v
        adj_ch = _build_adj_from_edges_int(Pch, child.rep_edges)
        cols_idx = [child.idx_portal_in_rep[p] for p in RN_nodes]
        dist_all = _weighted_multisource_on_rep(Pch, adj_ch, rows_parent, base=vec_lower.astype(np.float32))
        res = dist_all[cols_idx].astype(np.float32)

        self._memo_vec[key] = res
        return res

    # ---------- 查询 ----------
    def query(self, u: Any, v: Any) -> float:
        if u == v:
            return 0.0
        Cu = self._leaf_of(u); Cv = self._leaf_of(v)
        if Cu is None or Cv is None:
            return float("inf")

        # 不同 CC 直接 inf（超根不构建代表图）
        if Cu.cc_id is not None and Cv.cc_id is not None and Cu.cc_id != Cv.cc_id:
            return float("inf")

        if Cu == Cv:
            return float(Cu.pll.query(u, v) if Cu.pll is not None else math.inf)

        L = self._lca(Cu, Cv)
        if L.level == -2:
            return float("inf")

        Au = self._direct_child_under(L, Cu)
        Bv = self._direct_child_under(L, Cv)
        if Au is None or Bv is None:
            return float("inf")

        dl = self._vec_entity_to_parent_child_portals(u, L, Au)
        dr = self._vec_entity_to_parent_child_portals(v, L, Bv)
        if dl.size == 0 or dr.size == 0:
            return float("inf")
        if not np.isfinite(dl).any() or not np.isfinite(dr).any():
            return float("inf")

        P = len(L.rep_nodes)
        if P == 0 or not L.rep_edges:
            return float("inf")
        adj_L = _build_adj_from_edges_int(P, L.rep_edges)
        A_idx = L.child_portal_idx_in_parent.get(Au, [])
        B_idx = L.child_portal_idx_in_parent.get(Bv, [])
        if not A_idx or not B_idx:
            return float("inf")
        dist_all = _weighted_multisource_on_rep(P, adj_L, A_idx, base=dl.astype(np.float32))
        cand = dist_all[B_idx] + dr
        return float(np.min(cand))

    # ---------- 调试统计 ----------
    def _print_debug_stats(self):
        levels = sorted({c.level for c in self.all_clusters})
        print("\n[DEBUG] ===== Rep stats =====")
        for lvl in levels:
            Cs = [c for c in self.all_clusters if c.level == lvl]
            if not Cs:
                continue
            rep_sizes = [len(c.rep_nodes) for c in Cs if not c.is_leaf and lvl != -2]
            if rep_sizes:
                print(f"  Level {lvl:>3}: parents={len(rep_sizes)}, rep_nodes avg={np.mean(rep_sizes):.2f}, "
                      f"min={np.min(rep_sizes)}, max={np.max(rep_sizes)}")
        print("[DEBUG] ======================\n")

# ----------------------------- 评测（整图 LCC 内） -----------------------------
def _largest_connected_subgraph(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        raise RuntimeError("Graph is empty.")
    lcc_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc_nodes).copy()

def _sample_entity_pairs_in_graph(H: nx.Graph, n_pairs=500, rng_seed=42) -> List[Tuple[Any, Any]]:
    entities = [n for n in H.nodes() if not is_rel(n)]
    if len(entities) < 2:
        raise RuntimeError("Too few entity nodes in the graph.")
    rng = random.Random(rng_seed)
    pairs = set()
    trials = 0
    max_trials = n_pairs * 50
    while len(pairs) < n_pairs and trials < max_trials:
        trials += 1
        u, v = rng.sample(entities, 2)
        if (u, v) not in pairs and (v, u) not in pairs:
            pairs.add((u, v))
    return list(pairs)

def _compute_ground_truth_on_subgraph(H: nx.Graph, pairs: List[Tuple[Any, Any]]) -> List[float]:
    gt = []
    for u, v in tqdm(pairs, desc="compute GT (LCC subgraph)", unit="pair"):
        d = nx.shortest_path_length(H, u, v, weight="weight")
        gt.append(float(d))
    return gt

def evaluate(G: nx.Graph,
             idx: HierRepHybridTopKPLL,
             n_pairs: int,
             rng_seed: int,
             preprocessing_time: float) -> pd.DataFrame:
    H = _largest_connected_subgraph(G)
    pairs = _sample_entity_pairs_in_graph(H, n_pairs=n_pairs, rng_seed=rng_seed)
    gt = _compute_ground_truth_on_subgraph(H, pairs)

    t0 = time.time()
    exact = 0
    mae_sum = 0.0
    finite_count = 0
    for (u, v), g in tqdm(list(zip(pairs, gt)), desc="eval (queries on LCC)", unit="pair"):
        est = idx.query(u, v)
        if est == g:
            exact += 1
        if not math.isinf(est):
            mae_sum += abs(est - g)
            finite_count += 1
    query_time_sec = time.time() - t0

    total = len(pairs)
    inf_count = total - finite_count
    mae = (mae_sum / finite_count) if finite_count > 0 else float("inf")

    rows = [[H.number_of_nodes(), H.number_of_edges(),
             query_time_sec, total, finite_count, inf_count, exact, mae, preprocessing_time]]
    return pd.DataFrame(rows, columns=[
        "lcc_nodes", "lcc_edges",
        "query_time_sec", "samples", "returned", "inf_count", "exact_matches", "mae",
        "preprocessing_time"
    ])

# ----------------------------- main -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Pubmed",
                        choices=["Pubmed", "WN18", "edgelist"])
    parser.add_argument("--kg_file", type=str, default="./data/WN18.txt")
    parser.add_argument("--edge_file", type=str, default="./data/edges.txt")
    parser.add_argument("--planetoid_dir", type=str, default="./data/planetoid")
    parser.add_argument("--eval_pairs", type=int, default=200)

    parser.add_argument("--max_cluster_size", type=int, default=4000)
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--leaf_topk", type=int, default=8)
    parser.add_argument("--debug_stats", action="store_true")
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--build_cross_tables", action="store_true")

    args = parser.parse_args()

    # load graph
    if args.dataset == "WN18":
        G = load_wn18_graph_relation_nodes(args.kg_file)
        src_name = os.path.basename(args.kg_file)
    elif args.dataset == "edgelist":
        G = load_edge_list_graph(args.edge_file)
        src_name = os.path.basename(args.edge_file)
    else:
        G = load_planetoid_pubmed_graph(args.planetoid_dir)
        src_name = "Planetoid: ind.pubmed.graph"

    print(f"[INFO] Graph built from {src_name}")
    print(f"[INFO] |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

    # build index
    t0 = time.time()
    idx = HierRepHybridTopKPLL(
        G,
        max_cluster_size=args.max_cluster_size,
        resolution=args.resolution,
        random_seed=args.random_seed,
        weight_attr="weight",
        leaf_topk=args.leaf_topk,
        max_workers=args.max_workers,
        debug_stats=args.debug_stats,
        build_cross_tables=args.build_cross_tables,
    )
    preprocessing_time = time.time() - t0

    # evaluate
    df = evaluate(
        G, idx,
        n_pairs=args.eval_pairs,
        rng_seed=args.random_seed,
        preprocessing_time=preprocessing_time
    )

    print("\n=== Evaluation on Global LCC (Entities→Entities) ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()

