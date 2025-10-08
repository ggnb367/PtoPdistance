# -*- coding: utf-8 -*-
"""
gtree_m1.py
(CC-parallel HL + SUPER_ROOT + index-graph + star-closure + table-only query)

本版将 WN18 的“关系节点”构图替换为“关系收缩 + 元数据保留”：
- 对每条三元组 (h, r, t) 不再新建 ("REL", i, r) 节点；而是把 (h, t) 建为一条无向边；
- 边权 = 该实体对上所有关系的最小代价（默认每种关系 1.0，对应旧法 0.5+0.5）；
- 边属性保留全部关系语义：rels（关系集合）、w_by_rel（关系→权重）、multi（关系种数）。

其余流程保持不变：CC 并行 HL，叶子 PLL+TopK，父层索引/代表图星型闭包，父层 rep_D_pp 做抬升，查询全查表。
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
    """兼容旧接口：当前不再使用 ('REL', i, r) 这样的关系节点。"""
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
                # 给普通数据集也带上统一的“关系元数据”，便于后续扩展
                G.add_edge(u, v, weight=1.0, rels=("PLANETOID",), w_by_rel={"PLANETOID": 1.0}, multi=1)
    return G

def load_wn18_graph_relation_nodes(path: str) -> nx.Graph:
    """
    改进版 WN18 构图（关系收缩 + 元数据保留）：
      - 不建 ("REL", i, r) 关系节点；
      - 每个 (h, r, t) 收缩为实体—实体边 (h, t)，无向；
      - 对同一实体对的多关系，边权取最小关系代价（默认 1.0），
        并把全部关系与各自权重保存在边属性：
          rels=tuple(sorted(rel_set)), w_by_rel={rel: weight}, multi=len(rel_set)
    这样保持实体→实体最短距离与旧法等价，同时显著缩小图规模。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"WN18 file not found: {path}")

    # 可按需定义不同关系的代价；默认均为 1.0（对应旧法 0.5+0.5）
    rel_weight: Dict[Any, float] = {}

    def to_int(x):
        try:
            return int(x)
        except Exception:
            return x

    # (min(u,v), max(u,v)) -> {rel: weight_min}
    pair2rels: Dict[Tuple[Any, Any], Dict[Any, float]] = {}

    def add_triple(h, t, r):
        if h == t:
            return
        # 统一无向对的顺序（与旧法无向一致）
        a, b = (h, t) if node_sort_key(h) <= node_sort_key(t) else (t, h)
        w = float(rel_weight.get(r, 1.0))
        mp = pair2rels.setdefault((a, b), {})
        # 同一关系若多次出现，取该关系最小代价
        mp[r] = min(w, mp.get(r, float("inf")))

    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        # 第一行可能是条数；若不是数字，把它当作一条三元组
        try:
            _ = int(str(first).strip())
        except Exception:
            parts = first.strip().split()
            if len(parts) >= 2:
                h, t = to_int(parts[0]), to_int(parts[1])
                r = to_int(parts[2]) if len(parts) >= 3 else "NA"
                add_triple(h, t, r)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            h, t = to_int(parts[0]), to_int(parts[1])
            r = to_int(parts[2]) if len(parts) >= 3 else "NA"
            add_triple(h, t, r)

    G = nx.Graph()
    for (h, t), rmap in pair2rels.items():
        w_min = min(rmap.values()) if rmap else 1.0
        rels_sorted = tuple(sorted(rmap.keys(), key=lambda x: str(x)))
        G.add_edge(
            h, t,
            weight=float(w_min),
            rels=rels_sorted,
            w_by_rel={k: float(v) for k, v in rmap.items()},
            multi=len(rmap),
        )
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
                # 也补上统一的元数据
                G.add_edge(u, v, weight=float(w), rels=("EDGE",), w_by_rel={"EDGE": float(w)}, multi=1)
    return G

# ----------------------------- HL worker (parallel per-CC) -----------------------------
def _hl_on_cc_worker(args):
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
    """Weighted PLL for undirected graphs."""
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

    cc_id: Optional[int] = None
    global_borders: Set[Any] = field(default_factory=set)

    # 叶子结构
    pll: Optional[PrunedPLLIndex] = None
    leaf_portals: List[Any] = field(default_factory=list)
    idx_portal_in_leaf: Dict[Any, int] = field(default_factory=dict)
    leaf_nodes_order: List[Any] = field(default_factory=list)
    idx_node_in_leaf: Dict[Any, int] = field(default_factory=dict)
    topk_by_node: Dict[Any, List[Tuple[Any, float]]] = field(default_factory=dict)
    D_pp: Optional[np.ndarray] = None
    D_np: Optional[np.ndarray] = None

    # 索引图（index）
    index_nodes: List[Any] = field(default_factory=list)
    idx_index_in_parent: Dict[Any, int] = field(default_factory=dict)
    index_edges: List[Tuple[int, int, float]] = field(default_factory=list)
    index_D_pp: Optional[np.ndarray] = None
    index_pll: Optional[PrunedPLLIndex] = None

    # 代表图（rep）
    rep_nodes: List[Any] = field(default_factory=list)
    idx_portal_in_rep: Dict[Any, int] = field(default_factory=dict)
    rep_edges: List[Tuple[int, int, float]] = field(default_factory=list)
    rep_D_pp: Optional[np.ndarray] = None

    child_portals: Dict["Cluster", List[Any]] = field(default_factory=lambda: defaultdict(list))
    child_portal_idx_in_child: Dict["Cluster", List[int]] = field(default_factory=lambda: defaultdict(list))
    child_portal_idx_in_parent: Dict["Cluster", List[int]] = field(default_factory=lambda: defaultdict(list))

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

def _apsp_on_adj(P: int, adj_int: List[List[Tuple[int, float]]]) -> np.ndarray:
    D = np.full((P, P), INF, dtype=np.float32)
    for s in range(P):
        dist = _weighted_multisource_on_rep(P, adj_int, [s], base=None)
        D[s, :] = dist
    return D

# ----------------------------- workers：叶子 -----------------------------
def _leaf_pll_topk_worker(args):
    key, nodes_raw, edges, portals, leaf_topk, wname = args
    nodes = sorted(nodes_raw, key=node_sort_key)

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

    order = [u for u, _ in sorted(H.degree(), key=lambda kv: kv[1], reverse=True)]
    pll = PrunedPLLIndex(H, order, weight_attr=wname)
    pll.build()

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
                 build_cross_tables: bool = False,
                 no_rep_apsp: bool = False,
                 no_index_pll: bool = False):
        if hierarchical_leiden is None:
            raise RuntimeError("graspologic.partition.hierarchical_leiden not available. pip install graspologic")
        self.G = G
        self.wname = weight_attr
        self.leaf_topk = int(max(0, leaf_topk))
        self.max_workers = max_workers or (os.cpu_count() or 2)
        self.debug_stats = debug_stats
        self.build_cross_tables = bool(build_cross_tables)
        self.no_rep_apsp = bool(no_rep_apsp)
        self.no_index_pll = bool(no_index_pll)

        # 0) CC split + 并行 per-CC HL
        print("[Stage] connected components split + parallel hierarchical_leiden per CC ...")
        components = list(nx.connected_components(G))
        components.sort(key=lambda s: len(s), reverse=True)

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
            use_modularity=True,
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

        # 3) 自底向上：INDEX 图，再 REP 图
        print("[Stage] bottom-up: build INDEX graphs (star), then REP graphs (star) ...")
        self._build_index_and_rep_graphs_bottom_up()

        # 4) 计算 D_pp：index(PLL) + rep(APSP)
        print("[Stage] compute D_pp on INDEX and REP graphs ...")
        self._compute_dpps_for_index_and_rep()

        if self.debug_stats:
            self._print_debug_stats()

    # ---------- 层次（带 CC_ROOT 与 SUPER_ROOT） ----------
    def _build_hierarchy_with_cc_roots(self, hl_list, components) -> Tuple[Cluster, List[Cluster], Dict[Any, Cluster]]:
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

        for node, mems in node_levels.items():
            for i in range(len(mems) - 1):
                (l1, c1), (l2, c2) = mems[i], mems[i + 1]
                p = clusters[(l1, c1)]
                ch = clusters[(l2, c2)]
                if ch not in p.children:
                    p.children.append(ch)
                    ch.parent = p

        node2leaf: Dict[Any, Cluster] = {}
        for c in clusters.values():
            if not c.children:
                c.is_leaf = True
                for u in c.nodes:
                    node2leaf[u] = c

        cc_roots: List[Cluster] = []
        all_nodes = set(self.G.nodes())
        for comp_idx, nodes_cc in enumerate(components):
            nodes_cc = set(nodes_cc)
            cc_root = Cluster(level=-1, cid=('ccroot', comp_idx), nodes=nodes_cc, is_leaf=False, cc_id=comp_idx)
            for (lvl, cid), c in clusters.items():
                if lvl == 0 and c.parent is None and (c.nodes & nodes_cc):
                    c.parent = cc_root
                    cc_root.children.append(c)
            cc_roots.append(cc_root)

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
                pass
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
            c.D_pp = D_pp.astype(np.float32)
            c.leaf_nodes_order = list(node_order)
            c.idx_node_in_leaf = {u: i for i, u in enumerate(c.leaf_nodes_order)}
            c.D_np = D_np.astype(np.float32)

    # ---------- 构建：索引图 & 代表图 ----------
    def _build_index_and_rep_graphs_bottom_up(self):
        levels = sorted({c.level for c in self.all_clusters})
        levels.sort(reverse=True)  # bottom-up
        for lvl in levels:
            Cs = [c for c in self.all_clusters if c.level == lvl]
            if not Cs or lvl == -2:  # 仅跳过 SUPER_ROOT；CC_ROOT(-1) 也要建
                continue

            for c in tqdm(Cs, desc=f"index-graph build (level={lvl})", unit="cluster"):
                self._build_single_index_graph_star(c)

            for c in tqdm(Cs, desc=f"rep-graph build (level={lvl})", unit="cluster"):
                self._build_single_rep_graph_from_index(c)

    def _collect_cross_edges_and_extdeg(self, c: Cluster, node2child: Dict[Any, Cluster]):
        cross_edges_raw: List[Tuple[Any, Any, float]] = []
        ext_deg: Dict[Any, int] = defaultdict(int)
        for u, v, d in self.G.edges(c.nodes, data=True):
            cu = node2child.get(u); cv = node2child.get(v)
            if cu is None or cv is None or cu == cv:
                continue
            w = float(d.get(self.wname, 1.0))
            cross_edges_raw.append((u, v, w))
            ext_deg[u] += 1
            ext_deg[v] += 1
        return cross_edges_raw, ext_deg

    def _build_single_index_graph_star(self, c: Cluster):
        if c.level == -2:
            c.index_nodes = []
            c.index_edges = []
            c.idx_index_in_parent = {}
            return

        if c.is_leaf:
            ports = c.leaf_portals
            c.index_nodes = list(ports)
            c.idx_index_in_parent = {p: i for i, p in enumerate(c.index_nodes)}
            c.index_edges = []
            P = len(c.index_nodes)
            if c.D_pp is not None and P > 1:
                center = 0
                for j in range(P):
                    if j == center:
                        continue
                    w = float(c.D_pp[center, j])
                    if math.isinf(w):
                        continue
                    c.index_edges.append((center, j, w))
            return

        children = c.children
        node2child: Dict[Any, Cluster] = {}
        for ch in children:
            for u in ch.nodes:
                node2child[u] = ch

        cross_edges_raw, ext_deg = self._collect_cross_edges_and_extdeg(c, node2child)

        index_set: Set[Any] = set()
        for ch in children:
            if ch.is_leaf:
                index_set |= set(ch.leaf_portals)
            else:
                index_set |= set(getattr(ch, "rep_nodes", []))

        c.index_nodes = sorted(index_set, key=node_sort_key)
        c.idx_index_in_parent = {p: i for i, p in enumerate(c.index_nodes)}
        c.index_edges = []

        for ch in children:
            ports = sorted([p for p in c.index_nodes if p in ch.nodes], key=node_sort_key)
            if len(ports) < 2:
                continue
            parent_idx = [c.idx_index_in_parent[p] for p in ports]

            degs = [ext_deg.get(p, 0) for p in ports]
            center_local = int(np.argmax(degs)) if len(ports) >= 2 else 0
            ic = parent_idx[center_local]

            if ch.is_leaf:
                idx_child = [ch.idx_portal_in_leaf.get(p, None) for p in ports]
                if any(i is None for i in idx_child) or ch.D_pp is None:
                    continue
                jc = idx_child[center_local]
                for t in range(len(ports)):
                    if t == center_local:
                        continue
                    jt = idx_child[t]
                    w = float(ch.D_pp[jc, jt])
                    if math.isinf(w):
                        continue
                    it = parent_idx[t]
                    c.index_edges.append((ic, it, w))
            else:
                idx_child = [ch.idx_portal_in_rep.get(p, None) for p in ports]
                if any(i is None for i in idx_child):
                    continue
                jc = idx_child[center_local]
                if ch.rep_D_pp is not None:
                    for t in range(len(ports)):
                        if t == center_local:
                            continue
                        jt = idx_child[t]
                        w = float(ch.rep_D_pp[jc, jt])
                        if math.isinf(w):
                            continue
                        it = parent_idx[t]
                        c.index_edges.append((ic, it, w))
                else:
                    Pch = len(ch.rep_nodes)
                    if Pch == 0 or not ch.rep_edges:
                        continue
                    adj_ch = _build_adj_from_edges_int(Pch, ch.rep_edges)
                    dist = _weighted_multisource_on_rep(Pch, adj_ch, [jc], base=None)
                    for t in range(len(ports)):
                        if t == center_local:
                            continue
                        jt = idx_child[t]
                        it = parent_idx[t]
                        w = float(dist[jt])
                        if math.isinf(w):
                            continue
                        c.index_edges.append((ic, it, w))

        index_set = set(c.index_nodes)
        for u, v, w in cross_edges_raw:
            if u in index_set and v in index_set:
                i = c.idx_index_in_parent[u]; j = c.idx_index_in_parent[v]
                c.index_edges.append((i, j, w))

    def _build_single_rep_graph_from_index(self, c: Cluster):
        if c.is_leaf:
            portals = c.leaf_portals
            c.rep_nodes = list(portals)
            c.idx_portal_in_rep = {p: i for i, p in enumerate(c.rep_nodes)}
            c.rep_edges = []
            P = len(c.rep_nodes)
            if c.D_pp is not None and P > 1:
                center = 0
                for j in range(P):
                    if j == center:
                        continue
                    w = float(c.D_pp[center, j])
                    if math.isinf(w):
                        continue
                    c.rep_edges.append((center, j, w))
            c.child_portals.clear()
            c.child_portal_idx_in_child.clear()
            c.child_portal_idx_in_parent.clear()
            c.rep_D_pp = c.D_pp
            return

        if c.level == -2:
            c.rep_nodes = []
            c.rep_edges = []
            c.idx_portal_in_rep = {}
            c.child_portals.clear()
            c.child_portal_idx_in_child.clear()
            c.child_portal_idx_in_parent.clear()
            c.rep_D_pp = None
            return

        children = c.children
        node2child: Dict[Any, Cluster] = {}
        for ch in children:
            for u in ch.nodes:
                node2child[u] = ch

        cross_edges_raw, ext_deg = self._collect_cross_edges_and_extdeg(c, node2child)

        parent_borders = set()
        for (u, v, _) in cross_edges_raw:
            parent_borders.add(u)
            parent_borders.add(v)
        rep_set = (set(c.index_nodes) & parent_borders) if parent_borders else set(c.index_nodes)
        if not rep_set:
            rep_set = set(c.index_nodes)
        c.rep_nodes = sorted(rep_set, key=node_sort_key)
        c.idx_portal_in_rep = {p: i for i, p in enumerate(c.rep_nodes)}
        c.rep_edges = []

        try:
            tqdm.write(f"[parent REP] level={c.level} cid={c.cid} children={len(children)} rep_nodes={len(c.rep_nodes)} (index_nodes={len(c.index_nodes)})")
        except Exception:
            pass

        c.child_portals.clear()
        c.child_portal_idx_in_child.clear()
        c.child_portal_idx_in_parent.clear()

        rep_set = set(c.rep_nodes)
        for ch in children:
            ports = sorted([p for p in c.rep_nodes if p in ch.nodes], key=node_sort_key)
            c.child_portals[ch] = ports
            if not ports:
                c.child_portal_idx_in_child[ch] = []
                c.child_portal_idx_in_parent[ch] = []
                continue

            parent_idx = [c.idx_portal_in_rep[p] for p in ports]
            c.child_portal_idx_in_parent[ch] = parent_idx

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

                if len(ports) >= 2:
                    degs = [ext_deg.get(p, 0) for p in ports]
                    center_local = int(np.argmax(degs)) if len(ports) >= 2 else 0
                    ic = parent_idx[center_local]
                    jc = idx_child[center_local]
                    for t in range(len(ports)):
                        if t == center_local:
                            continue
                        it = parent_idx[t]
                        jt = idx_child[t]
                        w = float(ch.D_pp[jc, jt]) if ch.D_pp is not None else math.inf
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

                if len(ports) >= 2:
                    degs = [ext_deg.get(p, 0) for p in ports]
                    center_local = int(np.argmax(degs)) if len(ports) >= 2 else 0
                    ic = parent_idx[center_local]
                    jc = idx_child[center_local]

                    if ch.rep_D_pp is not None:
                        for t in range(len(ports)):
                            if t == center_local:
                                continue
                            it = parent_idx[t]
                            jt = idx_child[t]
                            w = float(ch.rep_D_pp[jc, jt])
                            if math.isinf(w):
                                continue
                            c.rep_edges.append((ic, it, w))
                    else:
                        Pch = len(ch.rep_nodes)
                        if Pch > 0 and ch.rep_edges:
                            adj_ch = _build_adj_from_edges_int(Pch, ch.rep_edges)
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

        for u, v, w in cross_edges_raw:
            if u in rep_set and v in rep_set:
                i = c.idx_portal_in_rep[u]; j = c.idx_portal_in_rep[v]
                c.rep_edges.append((i, j, w))

    # ---------- 计算 D_pp：index(PLL) + rep(APSP) ----------
    def _compute_dpps_for_index_and_rep(self):
        levels = sorted({c.level for c in self.all_clusters}, reverse=True)
        for lvl in levels:
            Cs = [c for c in self.all_clusters if c.level == lvl]
            if not Cs or lvl == -2:
                continue

            if not self.no_index_pll:
                for c in tqdm(Cs, desc=f"index D_pp via PLL (level={lvl})", unit="cluster"):
                    if not c.index_nodes:
                        c.index_D_pp = None
                        c.index_pll = None
                        continue
                    H = nx.Graph()
                    H.add_nodes_from(c.index_nodes)
                    for i, j, w in c.index_edges:
                        u = c.index_nodes[i]; v = c.index_nodes[j]
                        if u == v:
                            continue
                        if H.has_edge(u, v):
                            if w < H[u][v].get(self.wname, math.inf):
                                H[u][v][self.wname] = w
                        else:
                            H.add_edge(u, v, **{self.wname: w})
                    order = [u for u, _ in sorted(H.degree(), key=lambda kv: kv[1], reverse=True)]
                    pll = PrunedPLLIndex(H, order, weight_attr=self.wname)
                    pll.build()
                    P = len(c.index_nodes)
                    D = np.full((P, P), INF, dtype=np.float32)
                    for a in range(P):
                        ua = c.index_nodes[a]
                        for b in range(a, P):
                            vb = c.index_nodes[b]
                            d = pll.query(ua, vb)
                            val = np.float32(d if d != math.inf else np.inf)
                            D[a, b] = val
                            D[b, a] = val
                        D[a, a] = 0.0
                    c.index_pll = pll
                    c.index_D_pp = D

            if not self.no_rep_apsp:
                for c in tqdm(Cs, desc=f"rep D_pp via APSP (level={lvl})", unit="cluster"):
                    if not c.rep_nodes:
                        c.rep_D_pp = None
                        continue
                    if c.is_leaf:
                        c.rep_D_pp = c.D_pp
                        continue
                    P = len(c.rep_nodes)
                    if not c.rep_edges:
                        D = np.full((P, P), INF, dtype=np.float32)
                        np.fill_diagonal(D, 0.0)
                        c.rep_D_pp = D
                        continue
                    adj = _build_adj_from_edges_int(P, c.rep_edges)
                    c.rep_D_pp = _apsp_on_adj(P, adj)

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

    # ---------- 叶 → 门户向量 ----------
    def _vec_leaf_entity_to_leaf_portals(self, u: Any, leaf: Cluster) -> np.ndarray:
        P = len(leaf.leaf_portals)
        res = np.full((P,), INF, dtype=np.float32)
        if leaf.D_np is not None and u in leaf.idx_node_in_leaf:
            r = leaf.idx_node_in_leaf[u]
            res = leaf.D_np[r, :].astype(np.float32, copy=True)
            return res
        for i, p in enumerate(leaf.leaf_portals):
            d = leaf.pll.query(u, p) if leaf.pll is not None else math.inf
            res[i] = np.float32(d if d != math.inf else np.inf)
        return res

    # ---------- 抬升：child 向量 → parent.rep_nodes 向量 ----------
    def _lift_child_vec_to_parent_rep(self, parent: Cluster, child: Cluster, vec_child: np.ndarray) -> np.ndarray:
        P = len(parent.rep_nodes)
        res = np.full((P,), INF, dtype=np.float32)
        if parent.rep_D_pp is None or P == 0:
            return res

        rows_ports = parent.child_portals.get(child, [])
        rows_parent_idx = parent.child_portal_idx_in_parent.get(child, [])
        rows_child_idx: List[int] = []
        if not rows_ports or not rows_parent_idx:
            return res

        if child.is_leaf:
            for p in rows_ports:
                idx = child.idx_portal_in_leaf.get(p, None)
                if idx is None:
                    return res
                rows_child_idx.append(idx)
        else:
            for p in rows_ports:
                idx = child.idx_portal_in_rep.get(p, None)
                if idx is None:
                    return res
                rows_child_idx.append(idx)

        base = np.array([vec_child[i] for i in rows_child_idx], dtype=np.float32)
        Dsub = parent.rep_D_pp[np.array(rows_parent_idx, dtype=int), :]
        tmp = Dsub + base[:, None]
        res = np.min(tmp, axis=0).astype(np.float32)
        return res

    def _ascend_to_ancestor_rep(self, u: Any, leaf: Cluster, ancestor: Cluster) -> np.ndarray:
        vec = self._vec_leaf_entity_to_leaf_portals(u, leaf)
        cur = leaf
        while cur is not ancestor:
            parent = cur.parent
            if parent is None or parent.level < -2:
                return np.full((0,), INF, dtype=np.float32)
            vec = self._lift_child_vec_to_parent_rep(parent, cur, vec)
            cur = parent
        if ancestor.level == -2:
            return np.full((0,), INF, dtype=np.float32)
        return vec

    # ---------- 查询（完全查表版） ----------
    def query(self, u: Any, v: Any) -> float:
        if u == v:
            return 0.0
        Cu = self._leaf_of(u); Cv = self._leaf_of(v)
        if Cu is None or Cv is None:
            return float("inf")
        if Cu.cc_id is not None and Cv.cc_id is not None and Cu.cc_id != Cv.cc_id:
            return float("inf")
        if Cu == Cv:
            return float(Cu.pll.query(u, v) if Cu.pll is not None else math.inf)

        L = self._lca(Cu, Cv)
        if L.level == -2:
            return float("inf")

        vU_L = self._ascend_to_ancestor_rep(u, Cu, L)
        vV_L = self._ascend_to_ancestor_rep(v, Cv, L)
        if vU_L.size == 0 or vV_L.size == 0:
            return float("inf")
        if not np.isfinite(vU_L).any() or not np.isfinite(vV_L).any():
            return float("inf")

        best = float(np.min(vU_L + vV_L))
        return best

    # ---------- 调试 ----------
    def _print_debug_stats(self):
        levels = sorted({c.level for c in self.all_clusters})
        print("\n[DEBUG] ===== Rep/Index stats =====")
        for lvl in levels:
            Cs = [c for c in self.all_clusters if c.level == lvl]
            if not Cs:
                continue
            rep_sizes = [len(c.rep_nodes) for c in Cs if lvl not in (-2,)]
            idx_sizes = [len(c.index_nodes) for c in Cs if lvl not in (-2,)]
            if rep_sizes:
                print(f"  Level {lvl:>3}: parents={len(rep_sizes)}, rep_nodes avg={np.mean(rep_sizes):.2f}, "
                      f"min={np.min(rep_sizes)}, max={np.max(rep_sizes)}; "
                      f"index_nodes avg={np.mean(idx_sizes):.2f}")
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

    parser.add_argument("--max_cluster_size", type=int, default=5000)
    parser.add_argument("--resolution", type=float, default=0.3)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--leaf_topk", type=int, default=8)
    parser.add_argument("--debug_stats", action="store_true")
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--build_cross_tables", action="store_true")

    parser.add_argument("--no_rep_apsp", action="store_true",
                        help="Skip computing APSP D_pp on rep graphs.")
    parser.add_argument("--no_index_pll", action="store_true",
        help="Skip building PLL and D_pp on index graphs.")

    args = parser.parse_args()

    # load graph
    if args.dataset == "WN18":
        G = load_wn18_graph_relation_nodes(args.kg_file)  # 已改为“关系收缩 + 元数据保留”
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
        no_rep_apsp=args.no_rep_apsp,
        no_index_pll=args.no_index_pll,
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
