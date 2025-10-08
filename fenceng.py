#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fenceng.py — Hierarchical per-parent representative graphs
# 规则：
# - 叶子簇内最短路：PLL
# - 非叶子簇内最短路：ALT-A（单向 A* + 子集地标）
# - 父簇代表图：每个父簇独立构建；对子簇内部“边界-边界”连边：
#     * 默认：非叶子子簇 |B| <= alt_switch_B 用 ALT-A 做点对点；
#             非叶子子簇 |B| >  alt_switch_B 用 单源截断Dijkstra(kNN) 以控开销
#     * 可用 --wire_with_alta 强制非叶子子簇用 ALT-A（谨慎！）
# - 叶子 PLL 并行；查询时按叶子/非叶子分别走 PLL/ALT-A

import argparse
import time
import random
import heapq
import os
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any

import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from torch_geometric.datasets import Planetoid
except Exception:
    Planetoid = None

try:
    from graspologic.partition import leiden
except Exception:
    leiden = None


# ===============================
# Pruned PLL Index (叶子用)
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
        assert self.G is not None
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
# ALT-A（非叶子用）: 地标 + 单向 A*
# ===============================
class AltAIndex:
    """在给定子图（节点子集）上构建 ALT-A 索引：邻接、地标与 D 矩阵"""
    def __init__(self, fullG: nx.Graph, nodes: Set[Any], k_landmarks: int = 16, seed: int = 42):
        self.nodes = set(nodes)
        self.k = k_landmarks
        self.seed = seed

        # 压缩映射
        self.orig2id: Dict[Any, int] = {}
        self.id2orig: List[Any] = []
        for i, u in enumerate(self.nodes):
            self.orig2id[u] = i
            self.id2orig.append(u)

        # 邻接（本子图诱导）
        n = len(self.nodes)
        self.adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        for u in self.nodes:
            ui = self.orig2id[u]
            for v, data in fullG[u].items():
                if v in self.nodes:
                    vi = self.orig2id[v]
                    w = float(data.get("weight", 1.0))
                    self.adj[ui].append((vi, w))

        # 选择地标 + 预计算 D
        self.landmarks = self._select_landmarks_farthest(self.k, seed)
        self.D = self._precompute_lm_dists(self.landmarks)

    # ------- 工具: Dijkstra -------
    def _dijkstra_all(self, src: int) -> np.ndarray:
        n = len(self.adj)
        INF = float("inf")
        dist = np.full(n, INF, dtype=np.float64)
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    # ------- 地标选择（最远点优先） -------
    def _select_landmarks_farthest(self, k: int, seed: int = 42) -> List[int]:
        rng = random.Random(seed)
        n = len(self.adj)
        start = rng.randrange(n)
        d0 = self._dijkstra_all(start)
        L1 = int(np.nanargmax(np.where(np.isfinite(d0), d0, -np.inf)))
        landmarks = [L1]
        mind = d0.copy()
        for _ in range(1, min(k, n)):
            cand = int(np.nanargmax(np.where(np.isfinite(mind), mind, -np.inf)))
            landmarks.append(cand)
            d = self._dijkstra_all(cand)
            mask = d < mind
            mind[mask] = d[mask]
        return landmarks

    # ------- 预计算 D[L, n] -------
    def _precompute_lm_dists(self, landmarks: List[int]) -> np.ndarray:
        n = len(self.adj); L = len(landmarks)
        D = np.empty((L, n), dtype=np.float64)
        for i, Li in enumerate(landmarks):
            D[i, :] = self._dijkstra_all(Li)
        return D

    # ------- ALT-A 子集选择 -------
    def _pick_subset_by_pair(self, s: int, t: int, L_subset: int) -> np.ndarray:
        ds = self.D[:, s]; dt = self.D[:, t]
        ok = np.isfinite(ds) & np.isfinite(dt)
        idx = np.where(ok)[0]
        if idx.size == 0:
            return np.arange(min(L_subset, self.D.shape[0]), dtype=np.int32)
        score = np.abs(ds[idx] - dt[idx])
        order = idx[np.argsort(-score)]
        return order[: min(L_subset, order.size)].astype(np.int32)

    # ------- ALT-A 启发式 -------
    def _h_subset(self, dt_sub: np.ndarray, v: int, sub_idx: np.ndarray) -> float:
        h = 0.0
        for i, Lidx in enumerate(sub_idx):
            dlt = dt_sub[i]
            dlu = self.D[Lidx, v]
            if not (np.isfinite(dlt) and np.isfinite(dlu)):
                continue
            diff = abs(dlt - dlu)
            if diff > h:
                h = diff
        return h

    # ------- 公开：点对点距离（ALT-A） -------
    def sp(self, u_orig: Any, v_orig: Any, L_subset: int = 8) -> float:
        if u_orig == v_orig:
            return 0.0
        ui = self.orig2id.get(u_orig)
        vi = self.orig2id.get(v_orig)
        if ui is None or vi is None:
            return float("inf")

        sub = self._pick_subset_by_pair(ui, vi, L_subset)
        dt_sub = np.array([self.D[Lidx, vi] for Lidx in sub], dtype=np.float64)

        n = len(self.adj)
        INF = float("inf")
        g = np.full(n, INF, dtype=np.float64)
        closed = np.zeros(n, dtype=bool)
        g[ui] = 0.0
        h_s = self._h_subset(dt_sub, ui, sub)
        pq: List[Tuple[float, int]] = [(g[ui] + h_s, ui)]

        while pq:
            f, x = heapq.heappop(pq)
            if closed[x]:
                continue
            if x == vi:
                return g[x]
            closed[x] = True
            base = g[x]
            for y, w in self.adj[x]:
                if closed[y]:
                    continue
                nd = base + w
                if nd < g[y]:
                    g[y] = nd
                    hy = self._h_subset(dt_sub, y, sub)
                    heapq.heappush(pq, (nd + hy, y))
        return float("inf")

    # ------- 单源截断Dijkstra（kNN, 仅在本子图内） -------
    def knn_from(self, src_orig: Any, boundary_set: Set[Any], k: int) -> List[Tuple[Any, float]]:
        """返回 src 到 boundary_set 中 k 个最近点（(b, dist)），若不足 k 则尽量返回"""
        ui = self.orig2id.get(src_orig)
        if ui is None:
            return []
        target_ids = {self.orig2id[b] for b in boundary_set if b in self.orig2id}
        found = 0
        dist = {ui: 0.0}
        pq = [(0.0, 0, ui)]
        push_ctr = 1
        visited = set()
        out: List[Tuple[Any, float]] = []
        while pq and found < k:
            d, _, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u != ui and u in target_ids:
                out.append((self.id2orig[u], d))
                found += 1
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, push_ctr, v))
                    push_ctr += 1
        return out


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
        entities.add(h); entities.add(t)
    G.add_nodes_from(entities)
    for i, (h, t, r) in enumerate(triples):
        rel_node = ("REL", i, r)
        G.add_node(rel_node)
        G.add_edge(h, rel_node, weight=0.5)
        G.add_edge(rel_node, t, weight=0.5)
    return G


# ===============================
# 层级簇结构
# ===============================
@dataclass(frozen=True)
class ClusterID:
    level: int
    idx: int


@dataclass
class ClusterNode:
    cid: ClusterID
    level: int
    nodes: Set[Any]
    parent: Optional["ClusterNode"] = None
    children: List["ClusterNode"] = field(default_factory=list)

    pll: Optional[PrunedPLLIndex] = None            # 叶子专用
    alta: Optional[AltAIndex] = None                # 非叶子专用
    boundary_up: Set[Any] = field(default_factory=set)
    G_rep: Optional[nx.Graph] = None

    # ---- 修复: 允许作为 set 成员/字典键（用于 LCA 搜索等） ----
    def __hash__(self) -> int:
        return hash(self.cid)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ClusterNode) and self.cid == other.cid


class HierBuilder:
    def __init__(self, G: nx.Graph,
                 max_cluster_size: int,
                 max_levels: int,
                 resolution: float,
                 random_seed: int = 42,
                 topk_u: int = 16,
                 rep_k_inner: int = 8,
                 rep_max_pairs: int = 200000,
                 alt_k: int = 16,
                 alt_subset: int = 8,
                 wire_with_alta: bool = False,
                 alt_switch_B: int = 200):
        if leiden is None:
            raise RuntimeError("graspologic.partition.leiden is required.")
        self.G = G
        self.max_cluster_size = max_cluster_size
        self.max_levels = max_levels
        self.resolution = resolution
        self.random_seed = random_seed
        self.topk_u = topk_u
        self.rep_k_inner = rep_k_inner
        self.rep_max_pairs = rep_max_pairs
        self.alt_k = alt_k
        self.alt_subset = alt_subset
        self.wire_with_alta = bool(wire_with_alta)
        self.alt_switch_B = int(alt_switch_B)

        self.node2leaf: Dict[Any, ClusterNode] = {}
        self.leaves: List[ClusterNode] = []
        self.all_clusters: List[ClusterNode] = []
        self._cid_counter = 0

    def _new_cid(self, level: int) -> ClusterID:
        c = ClusterID(level=level, idx=self._cid_counter)
        self._cid_counter += 1
        return c

    def _partition_once(self, sub_nodes: List[Any]) -> Dict[Any, int]:
        if len(sub_nodes) <= self.max_cluster_size:
            return {u: 0 for u in sub_nodes}
        H = self.G.subgraph(sub_nodes).copy()
        try:
            comm = leiden(H, resolution=self.resolution, random_seed=self.random_seed)
        except Exception:
            return {u: 0 for u in sub_nodes}
        if len(set(comm.values())) <= 1:
            return {u: 0 for u in sub_nodes}
        return comm

    def build_tree(self) -> ClusterNode:
        root = ClusterNode(cid=self._new_cid(0), level=0, nodes=set(self.G.nodes()))
        self.all_clusters.append(root)
        self._build_rec(root)
        return root

    def _build_rec(self, node: ClusterNode):
        if node.level >= self.max_levels or len(node.nodes) <= self.max_cluster_size:
            self.leaves.append(node)
            for u in node.nodes:
                self.node2leaf[u] = node
            return
        comm = self._partition_once(list(node.nodes))
        groups = defaultdict(list)
        for u, g in comm.items():
            groups[g].append(u)
        if len(groups) <= 1:
            self.leaves.append(node)
            for u in node.nodes:
                self.node2leaf[u] = node
            return
        for _, members in groups.items():
            child = ClusterNode(cid=self._new_cid(node.level + 1),
                                level=node.level + 1,
                                nodes=set(members),
                                parent=node)
            node.children.append(child)
            self.all_clusters.append(child)
            self._build_rec(child)

    # ---------- 边界（相对父） ----------
    def _compute_boundary_up_for_children(self, parent: ClusterNode):
        if not parent.children:
            return
        parent_nodes = parent.nodes
        for ch in parent.children:
            bset = set()
            for u in ch.nodes:
                for v in self.G[u]:
                    if v in parent_nodes and v not in ch.nodes:
                        bset.add(u); break
            ch.boundary_up = bset

    # ---------- 叶子：并行 PLL ----------
    @staticmethod
    def _leaf_task(nodes: List[Any], edges: List[Tuple[Any, Any, float]]) -> dict:
        subg = nx.Graph()
        subg.add_nodes_from(nodes)
        for u, v, w in edges:
            subg.add_edge(u, v, weight=float(w))
        # 用离心率做序
        try:
            ecc = {}
            for comp in nx.connected_components(subg):
                cg = subg.subgraph(comp)
                ecc.update(nx.eccentricity(cg))
            order = sorted(subg.nodes(), key=lambda n: ecc.get(n, 0))
        except Exception:
            order = list(subg.nodes())
        pll = PrunedPLLIndex(subg, order)
        # inline build（无 tqdm）
        for root in pll.order:
            dist = {root: 0.0}
            heap = [(0.0, 0, root)]
            counter = 0

            def inline_query(a, b):
                best = float("inf")
                la = pll.labels.get(a, {})
                lb = pll.labels.get(b, {})
                if len(la) > len(lb):
                    la, lb = lb, la
                for lm, da in la.items():
                    db = lb.get(lm)
                    if db is not None:
                        s = da + db
                        if s < best:
                            best = s
                return best

            while heap:
                d, _, u = heapq.heappop(heap)
                if d > dist[u]:
                    continue
                if inline_query(root, u) <= d:
                    continue
                pll.labels[u][root] = d
                for v, data in subg[u].items():
                    w = float(data.get("weight", 1.0))
                    nd = d + w
                    if nd < dist.get(v, float("inf")):
                        dist[v] = nd
                        counter += 1
                        heapq.heappush(heap, (nd, counter, v))
        return pll.labels

    def build_leaf_pll(self, pll_workers: Optional[int] = None):
        if pll_workers is None:
            pll_workers = os.cpu_count() or 2
        tasks = []
        for leaf in self.leaves:
            nset = leaf.nodes
            edges = []
            for u in nset:
                for v, data in self.G[u].items():
                    if v in nset and u <= v:
                        edges.append((u, v, data.get("weight", 1.0)))
            tasks.append((leaf, list(nset), edges))

        futures = {}
        with ProcessPoolExecutor(max_workers=pll_workers) as ex:
            for leaf, nodes, edges in tasks:
                fut = ex.submit(HierBuilder._leaf_task, nodes, edges)
                futures[fut] = leaf
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Build leaf PLL (parallel)", unit="leaf"):
                labels = fut.result()
                futures[fut].pll = PrunedPLLIndex.from_labels(labels)

    # ---------- 非叶子：构建 ALT-A 索引（供父簇和查询使用） ----------
    def build_nonleaf_alta(self):
        for c in tqdm([x for x in self.all_clusters if x.children], desc="Build ALT-A for non-leaf clusters", unit="cluster"):
            c.alta = AltAIndex(self.G, c.nodes, k_landmarks=self.alt_k, seed=self.random_seed)

    # ---------- 代表图构建（按父簇） ----------
    def build_cluster_rep_graphs(self, verbose: bool = True):
        by_level = defaultdict(list)
        for c in self.all_clusters:
            by_level[c.level].append(c)
        max_lv = max(by_level.keys())

        # boundary_up
        for lv in range(max_lv - 1, -1, -1):
            for parent in by_level[lv]:
                if parent.children:
                    self._compute_boundary_up_for_children(parent)

        # 每层自底向上构建
        for lv in range(max_lv - 1, -1, -1):
            parents = [p for p in by_level[lv] if p.children]
            for parent in tqdm(parents, desc=f"Build G_rep at level {lv}", unit="cluster"):
                rep_nodes: Set[Any] = set()
                child_idx: Dict[Any, int] = {}
                for i, ch in enumerate(parent.children):
                    for x in ch.boundary_up:
                        rep_nodes.add(x)
                        child_idx[x] = i
                Grep = nx.Graph()
                Grep.add_nodes_from(rep_nodes)

                total_pairs = 0
                for ch in parent.children:
                    blist = list(ch.boundary_up)
                    B = len(blist)
                    if B <= 1:
                        continue
                    k = self.rep_k_inner if self.rep_k_inner > 0 else (B - 1)
                    if verbose:
                        print(f"[level {lv}] parent {parent.cid.idx} child |B|={B} -> inner edges ≤ {B*k}")

                    # ——选择内部连边策略——
                    use_alta = False
                    if ch.children:  # 非叶子
                        if self.wire_with_alta:
                            use_alta = True
                        else:
                            use_alta = (B <= self.alt_switch_B)
                    # 叶子：用 PLL；非叶子：用 ALT-A 或 kNN(Dijkstra) 早停
                    if use_alta:
                        # ALT-A：对每个 u 挑 k 个最近 v（逐个 ALT-A 计算，选最小的k个）
                        for i, u in enumerate(blist):
                            dlist = []
                            for j, v in enumerate(blist):
                                if i == j:
                                    continue
                                if ch.children:
                                    d = ch.alta.sp(u, v, L_subset=self.alt_subset)
                                else:
                                    d = float(ch.pll.query(u, v))
                                if np.isfinite(d):
                                    dlist.append((v, d))
                            if not dlist:
                                continue
                            dlist.sort(key=lambda x: x[1])
                            for v, d in dlist[:k]:
                                total_pairs += 1
                                if Grep.has_edge(u, v):
                                    if d < Grep[u][v].get("weight", float("inf")):
                                        Grep[u][v]["weight"] = d
                                else:
                                    Grep.add_edge(u, v, weight=d)
                                if total_pairs >= self.rep_max_pairs:
                                    break
                            if total_pairs >= self.rep_max_pairs:
                                if verbose:
                                    print(f"[WARN] Reached rep_max_pairs={self.rep_max_pairs} in parent {parent.cid}")
                                break
                    else:
                        # 单源截断 Dijkstra（在子簇诱导子图内找kNN）
                        if ch.children:     # 非叶子：用 ALT-A 的子图邻接 + knn_from
                            for u in blist:
                                pairs = ch.alta.knn_from(u, set(blist) - {u}, k)
                                for v, d in pairs:
                                    total_pairs += 1
                                    if Grep.has_edge(u, v):
                                        if d < Grep[u][v].get("weight", float("inf")):
                                            Grep[u][v]["weight"] = d
                                    else:
                                        Grep.add_edge(u, v, weight=d)
                                    if total_pairs >= self.rep_max_pairs:
                                        break
                                if total_pairs >= self.rep_max_pairs:
                                    if verbose:
                                        print(f"[WARN] Reached rep_max_pairs={self.rep_max_pairs} in parent {parent.cid}")
                                    break
                        else:               # 叶子：在叶子子图上做 kNN（临时邻接）
                            adj = {u: [] for u in ch.nodes}
                            for u in ch.nodes:
                                for v, data in self.G[u].items():
                                    if v in ch.nodes:
                                        adj[u].append((v, float(data.get("weight", 1.0))))
                            def knn_leaf(src, k, targets):
                                found = 0
                                dist = {src: 0.0}
                                pq = [(0.0, 0, src)]
                                ctr = 1; visited=set()
                                out=[]
                                target_set=set(targets)
                                while pq and found<k:
                                    d, _, u = heapq.heappop(pq)
                                    if u in visited: continue
                                    visited.add(u)
                                    if u!=src and u in target_set:
                                        out.append((u, d)); found+=1
                                    for v, w in adj[u]:
                                        nd = d+w
                                        if nd < dist.get(v, float("inf")):
                                            dist[v]=nd; heapq.heappush(pq, (nd, ctr, v)); ctr+=1
                                return out
                            for u in blist:
                                pairs = knn_leaf(u, k, set(blist)-{u})
                                for v, d in pairs:
                                    total_pairs += 1
                                    if Grep.has_edge(u, v):
                                        if d < Grep[u][v].get("weight", float("inf")):
                                            Grep[u][v]["weight"] = d
                                    else:
                                        Grep.add_edge(u, v, weight=d)
                                    if total_pairs >= self.rep_max_pairs:
                                        break
                                if total_pairs >= self.rep_max_pairs:
                                    if verbose:
                                        print(f"[WARN] Reached rep_max_pairs={self.rep_max_pairs} in parent {parent.cid}")
                                    break

                # (2) 跨子簇原图边
                if rep_nodes:
                    for u in rep_nodes:
                        for v, data in self.G[u].items():
                            if v in rep_nodes and child_idx.get(u, -1) != child_idx.get(v, -1):
                                w = float(data.get("weight", 1.0))
                                if Grep.has_edge(u, v):
                                    if w < Grep[u][v].get("weight", float("inf")):
                                        Grep[u][v]["weight"] = w
                                else:
                                    Grep.add_edge(u, v, weight=w)

                parent.G_rep = Grep

    # ---------- 查询 ----------
    def find_lca(self, a: ClusterNode, b: ClusterNode) -> ClusterNode:
        if a is b:
            return a
        A, B = a, b
        seen = set()
        while A or B:
            if A:
                if A in seen:
                    return A
                seen.add(A)
                A = A.parent
            if B:
                if B in seen:
                    return B
                seen.add(B)
                B = B.parent
        return a

    def query_within(self, cluster: ClusterNode, u: Any, v: Any) -> float:
        # 叶子：PLL
        if not cluster.children:
            return float(cluster.pll.query(u, v)) if cluster.pll else float("inf")

        # 同一子簇：叶子=PLL；非叶子=ALT-A
        child_u = child_v = None
        for ch in cluster.children:
            if u in ch.nodes: child_u = ch
            if v in ch.nodes: child_v = ch
        if child_u is None or child_v is None:
            return float("inf")
        if child_u is child_v:
            if child_u.children:
                return child_u.alta.sp(u, v, L_subset=self.alt_subset)
            else:
                return float(child_u.pll.query(u, v)) if child_u.pll else float("inf")

        # 不同子簇：在 LCA 的 G_rep 上连接虚拟端点
        Grep = cluster.G_rep
        if Grep is None or Grep.number_of_nodes() == 0:
            return float("inf")

        Bu = child_u.boundary_up.intersection(Grep.nodes())
        Bv = child_v.boundary_up.intersection(Grep.nodes())
        if not Bu or not Bv:
            return float("inf")

        def point_to_boundaries(ch: ClusterNode, point: Any, bset: Set[Any], k: int):
            dists = []
            if ch.children:
                for b in bset:
                    d = ch.alta.sp(point, b, L_subset=self.alt_subset)
                    if np.isfinite(d): dists.append((b, d))
                dists.sort(key=lambda x: x[1])
                return dists[:min(k, len(dists))]
            else:
                for b in bset:
                    d = float(ch.pll.query(point, b))
                    if np.isfinite(d): dists.append((b, d))
                dists.sort(key=lambda x: x[1])
                return dists[:min(k, len(dists))]

        src_pairs = point_to_boundaries(child_u, u, Bu, self.topk_u)
        dst_pairs = point_to_boundaries(child_v, v, Bv, self.topk_u)
        if not src_pairs or not dst_pairs:
            return float("inf")

        H = Grep.copy()
        SRC = ("SRC", id(u), id(child_u))
        DST = ("DST", id(v), id(child_v))
        H.add_node(SRC); H.add_node(DST)
        for b, du in src_pairs: H.add_edge(SRC, b, weight=du)
        for b, dv in dst_pairs: H.add_edge(DST, b, weight=dv)

        try:
            return float(nx.shortest_path_length(H, SRC, DST, weight="weight"))
        except nx.NetworkXNoPath:
            return float("inf")

    def query_distance(self, u: Any, v: Any) -> float:
        leaf_u = self.node2leaf.get(u)
        leaf_v = self.node2leaf.get(v)
        if leaf_u is None or leaf_v is None:
            return float("inf")
        lca = self.find_lca(leaf_u, leaf_v)
        return self.query_within(lca, u, v)


# ===============================
# 评测
# ===============================
def evaluate_hier(G: nx.Graph, hb: HierBuilder, pairs: int) -> pd.DataFrame:
    nodes = list(G.nodes())
    samples = []
    # 采样可达对
    for _ in range(pairs * 3):
        u = random.choice(nodes); v = random.choice(nodes)
        if u == v:
            continue
        try:
            d = nx.shortest_path_length(G, u, v, weight="weight")
            samples.append((u, v, d))
            if len(samples) >= pairs:
                break
        except nx.NetworkXNoPath:
            continue

    correct = total = 0
    err = 0.0
    t0 = time.time()
    for u, v, d in samples:
        est = hb.query_distance(u, v)
        if est == d:
            correct += 1
        if est < float("inf"):
            err += abs(est - d); total += 1
    t = time.time() - t0
    rows = [[f"Hier (leaf=PLL, nonleaf=ALT-A, kNN={hb.rep_k_inner})",
             t, len(samples), correct, (err/total if total>0 else float('inf'))]]
    return pd.DataFrame(rows, columns=["method", "query_time_sec", "samples", "exact_matches", "mae"])


# ===============================
# Main
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Pubmed", choices=["Cora", "CiteSeer", "Pubmed"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--kg_file", type=str, default=None, help="Path to WN18.txt. If set, overrides --dataset.")
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--resolution", type=float, default=0.3)
    parser.add_argument("--max_cluster_size", type=int, default=1000)
    parser.add_argument("--hl_max_levels", type=int, default=3)
    parser.add_argument("--topk_u", type=int, default=16, help="LCA 虚拟端点两侧选择的边界 TopK")
    parser.add_argument("--rep_k_inner", type=int, default=8, help="父簇代表图：每个边界点在子簇内连 k 条最近边")
    parser.add_argument("--rep_max_pairs", type=int, default=200000, help="父簇代表图单个父簇最大内连边数上限")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--pll_workers", type=int, default=None, help="叶子 PLL 并行进程数")

    # ALT-A 相关参数
    parser.add_argument("--alt_k", type=int, default=16, help="ALT-A 的地标个数（非叶子）")
    parser.add_argument("--alt_subset", type=int, default=8, help="ALT-A 查询时的地标子集大小")
    parser.add_argument("--wire_with_alta", type=int, default=0, help="非叶子子簇内部连边是否强制使用 ALT-A（1=是）")
    parser.add_argument("--alt_switch_B", type=int, default=200, help="非叶子边界 |B| 超过该值则改用截断 Dijkstra(kNN)")

    args = parser.parse_args()

    # 加载图
    if args.kg_file:
        G = load_wn18_graph_relation_nodes(args.kg_file)
        src_name = f"WN18({os.path.basename(args.kg_file)})-rel_nodes"
    else:
        G = load_planetoid_graph(args.dataset, root=args.data_root)
        src_name = args.dataset

    print(f"[INFO] Graph: {src_name}, |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

    # 构建层次
    hb = HierBuilder(G,
                     max_cluster_size=args.max_cluster_size,
                     max_levels=args.hl_max_levels,
                     resolution=args.resolution,
                     random_seed=args.random_seed,
                     topk_u=args.topk_u,
                     rep_k_inner=args.rep_k_inner,
                     rep_max_pairs=args.rep_max_pairs,
                     alt_k=args.alt_k,
                     alt_subset=args.alt_subset,
                     wire_with_alta=bool(args.wire_with_alta),
                     alt_switch_B=args.alt_switch_B)

    root = hb.build_tree()
    sizes = Counter(len(c.nodes) for c in hb.all_clusters if not c.children)
    print(f"[INFO] #leaf_clusters = {len(hb.leaves)}, leaf size stats (top 10): {sizes.most_common(10)}")

    # 叶子 PLL（并行）
    hb.build_leaf_pll(pll_workers=args.pll_workers)
    # 非叶子 ALT-A 索引
    hb.build_nonleaf_alta()
    # 父簇代表图
    hb.build_cluster_rep_graphs(verbose=True)

    # 评测
    df_eval = evaluate_hier(G, hb, pairs=args.pairs)
    print("\n=== Evaluation: Hierarchical per-parent representative graphs (leaf=PLL, nonleaf=ALT-A) ===")
    print(df_eval.to_string(index=False))


if __name__ == "__main__":
    main()
