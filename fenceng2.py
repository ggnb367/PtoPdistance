#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fenceng2.py — G-Tree tables with GLOBAL-LANDMARK ALT at parents,
#                plus heavy precompute to make query O(K) / O(|B_LCA|)
#
# 运行示例（Pubmed）:
#   python fenceng2.py --dataset Pubmed --pll_workers 8 --pairs 300
#
# 运行示例（WN18 / 50k）:
#   python fenceng2.py --kg_file ./data/WN50k.txt --pll_workers 8 --pairs 300
#   # 或：--kg_file ./data/WN18.txt
#
# 关键优化：
# - 预计算：叶子全体节点→叶边界（PLL 查表）；
# - 自底向上 min-plus“提升”：叶→每层父簇边界（分块广播 + np.min）；
# - 根层（无边界）预计算：每对子簇 Top-K 门户 (b,c,dmid)；
# - 查询时仅数组取值 + O(|B_LCA|) 或 O(K) 运算。

import argparse
import os
import time
import heapq
import random
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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


# ---------------- utils ----------------
def stable_key(x: Any) -> Tuple[int, str]:
    return (0 if not isinstance(x, tuple) else 1, repr(x))

def tri_pack_len(n: int) -> int:
    return n * (n - 1) // 2

def tri_index(n: int, i: int, j: int) -> int:
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def unpack_get(packed: np.ndarray, n: int, i: int, j: int) -> float:
    if i == j: return 0.0
    if i > j: i, j = j, i
    return float(packed[tri_index(n, i, j)])

def unpack_set(packed: np.ndarray, n: int, i: int, j: int, v: float):
    if i == j: return
    if i > j: i, j = j, i
    packed[tri_index(n, i, j)] = v


# ---------------- PLL ----------------
class PrunedPLLIndex:
    def __init__(self, G: nx.Graph, order: Optional[List[Any]] = None):
        self.G = G
        self.labels: Dict[Any, Dict[Any, float]] = {v: {} for v in G.nodes()}
        self.order = list(order) if order is not None else list(G.nodes())

    def _inline_query(self, u, v) -> float:
        best = float("inf")
        Lu = self.labels.get(u, {})
        Lv = self.labels.get(v, {})
        if len(Lu) > len(Lv):
            Lu, Lv = Lv, Lu
        for hub, du in Lu.items():
            dv = Lv.get(hub)
            if dv is not None:
                s = du + dv
                if s < best:
                    best = s
        return best

    def build(self):
        for root in self.order:
            dist = {root: 0.0}
            pq = [(0.0, 0, root)]
            ctr = 1
            while pq:
                d, _, u = heapq.heappop(pq)
                if d != dist.get(u, float("inf")):
                    continue
                if self._inline_query(root, u) <= d:
                    continue
                self.labels[u][root] = d
                for v, data in self.G[u].items():
                    w = float(data.get("weight", 1.0))
                    nd = d + w
                    if nd < dist.get(v, float("inf")):
                        dist[v] = nd
                        heapq.heappush(pq, (nd, ctr, v)); ctr += 1

    def query(self, u, v) -> float:
        return self._inline_query(u, v)


# ---------------- loaders ----------------
def load_planetoid_graph(name="Pubmed", root=None) -> nx.Graph:
    if Planetoid is None:
        raise RuntimeError("torch_geometric not available; install it or use --kg_file.")
    root = root or os.path.abspath(f"./data/{name}")
    data = Planetoid(root=root, name=name)[0]
    ei = data.edge_index.numpy()
    G = nx.Graph()
    edges = set()
    for u, v in zip(ei[0], ei[1]):
        if u == v: continue
        a, b = int(u), int(v)
        if a > b: a, b = b, a
        edges.add((a, b))
    for u, v in edges:
        G.add_edge(u, v, weight=1.0)
    return G

def load_wn_graph_relation_nodes(path: str) -> nx.Graph:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
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
            if len(parts) < 2: continue
            h, t = parts[0], parts[1]
            r = parts[2] if len(parts) >= 3 else "NA"
            if h == t: continue
            triples.append((h, t, r))
    G = nx.Graph()
    ents = set()
    for h, t, _ in triples:
        ents.add(h); ents.add(t)
    G.add_nodes_from(ents)
    for i, (h, t, r) in enumerate(triples):
        rn = ("REL", i, r)
        G.add_node(rn)
        G.add_edge(h, rn, weight=0.5)
        G.add_edge(rn, t, weight=0.5)
    return G


# ---------------- global landmarks for ALT (parents) ----------------
def build_numeric_graph(G: nx.Graph):
    nodes = sorted(G.nodes(), key=stable_key)
    node2id = {u: i for i, u in enumerate(nodes)}
    id2node = {i: u for u, i in node2id.items()}
    n = len(nodes)
    adj = [[] for _ in range(n)]
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        ui, vi = node2id[u], node2id[v]
        adj[ui].append((vi, w))
        adj[vi].append((ui, w))
    return adj, node2id, id2node

def dijkstra_all(adj: List[List[Tuple[int, float]]], src: int) -> np.ndarray:
    n = len(adj)
    INF = float("inf")
    dist = np.full(n, INF, dtype=np.float64)
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

def select_landmarks_farthest(adj: List[List[Tuple[int, float]]], k: int, seed: int = 42):
    rng = random.Random(seed)
    n = len(adj)
    start = rng.randrange(n)
    d0 = dijkstra_all(adj, start)
    L0 = int(np.nanargmax(np.where(np.isfinite(d0), d0, -np.inf)))
    L = [L0]; Dlist = [d0]
    mind = d0.copy()
    for _ in range(1, k):
        cand = int(np.nanargmax(np.where(np.isfinite(mind), mind, -np.inf)))
        L.append(cand)
        d = dijkstra_all(adj, cand)
        Dlist.append(d)
        mask = d < mind
        # >>> FIX: use d[mask] rather than d
        mind[mask] = d[mask]
    return L, Dlist

def precompute_global_landmarks(G: nx.Graph, k: int, seed: int = 42):
    adj, node2id, id2node = build_numeric_graph(G)
    L_ids, Dlist = select_landmarks_farthest(adj, k, seed=seed)
    D_gl = np.vstack([d.astype(np.float32, copy=False) for d in Dlist])
    return D_gl, node2id, id2node


# ---------------- hierarchy ----------------
@dataclass(frozen=True)
class ClusterID:
    level: int
    idx: int

@dataclass
class Region:
    cid: ClusterID
    level: int
    nodes: Set[Any]
    parent: Optional["Region"] = None
    children: List["Region"] = field(default_factory=list)

    borders: List[Any] = field(default_factory=list)
    border_index: Dict[Any, int] = field(default_factory=dict)

    # leaf PLL
    leaf_pll: Optional[PrunedPLLIndex] = None

    # BORDER×BORDER table（上三角 pack）
    border_pack: Optional[np.ndarray] = None

    # child -> parent: |B_child| × |B_parent|
    child_to_parent: Dict[int, np.ndarray] = field(default_factory=dict)

    # 根层：每对子簇的 Top-K 门户 (b,c,mid)
    portals: Dict[Tuple[int, int], List[Tuple[Any, Any, float]]] = field(default_factory=dict)

    # 叶子的节点列表（只在叶级填充，便于预计算）
    node_list: List[Any] = field(default_factory=list)
    node_index: Dict[Any, int] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.cid)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Region) and self.cid == other.cid


class GTreeBuilderALT:
    def __init__(self, G: nx.Graph,
                 max_cluster_size: int = 1000,
                 max_levels: int = 3,
                 resolution: float = 0.3,
                 random_seed: int = 42,
                 pll_workers: Optional[int] = None,
                 alg: str = "alt",
                 L_glob: int = 32,
                 lm_seed: int = 42,
                 portal_topk: int = 16,
                 chunk_rows: int = 512):
        if leiden is None:
            raise RuntimeError("graspologic.partition.leiden is required.")
        self.G = G
        self.max_cluster_size = max_cluster_size
        self.max_levels = max_levels
        self.resolution = resolution
        self.random_seed = random_seed
        self.pll_workers = pll_workers or (os.cpu_count() or 2)

        self.alg = alg.lower().strip()
        self.L_glob = L_glob
        self.lm_seed = lm_seed
        self.portal_topk = portal_topk
        self.chunk_rows = chunk_rows  # 叶->父 min-plus 提升分块大小

        self.root: Optional[Region] = None
        self.all_regions: List[Region] = []
        self.leaves: List[Region] = []
        self.node2leaf: Dict[Any, Region] = {}
        self._cid_counter = 0

        # global landmarks
        self.D_gl: Optional[np.ndarray] = None
        self.node2id: Optional[Dict[Any, int]] = None

        # 叶子到祖先边界的矩阵缓存：leaf -> {region.cid: np.ndarray [|nodes(leaf)| × |B_region|]}
        self.leaf2anc: Dict[Region, Dict[ClusterID, np.ndarray]] = defaultdict(dict)

        # timers
        self.t_build_tree = 0.0
        self.t_borders = 0.0
        self.t_leaf_pll = 0.0
        self.t_leaf_tables = 0.0
        self.t_parent_tables = 0.0
        self.t_landmarks = 0.0
        self.t_leaf2anc = 0.0

    # --------- build tree ---------
    def _new_cid(self, level: int) -> ClusterID:
        c = ClusterID(level, self._cid_counter); self._cid_counter += 1
        return c

    def _partition_once(self, nodes: List[Any]) -> Dict[Any, int]:
        if len(nodes) <= self.max_cluster_size:
            return {u: 0 for u in nodes}
        H = self.G.subgraph(nodes).copy()
        try:
            comm = leiden(H, resolution=self.resolution, random_seed=self.random_seed)
        except Exception:
            return {u: 0 for u in nodes}
        if len(set(comm.values())) <= 1:
            return {u: 0 for u in nodes}
        return comm

    def build_tree(self) -> Region:
        t0 = time.time()
        root = Region(cid=self._new_cid(0), level=0, nodes=set(self.G.nodes()))
        self._build_rec(root)
        self.root = root
        self.t_build_tree = time.time() - t0
        return root

    def _build_rec(self, R: Region):
        self.all_regions.append(R)
        if R.level >= self.max_levels or len(R.nodes) <= self.max_cluster_size:
            self.leaves.append(R)
            for u in R.nodes:
                self.node2leaf[u] = R
            R.node_list = sorted(R.nodes, key=stable_key)
            R.node_index = {u: i for i, u in enumerate(R.node_list)}
            return
        comm = self._partition_once(sorted(R.nodes, key=stable_key))
        groups = defaultdict(list)
        for u, g in comm.items():
            groups[g].append(u)
        if len(groups) <= 1:
            self.leaves.append(R)
            for u in R.nodes:
                self.node2leaf[u] = R
            R.node_list = sorted(R.nodes, key=stable_key)
            R.node_index = {u: i for i, u in enumerate(R.node_list)}
            return
        for _, members in groups.items():
            C = Region(cid=self._new_cid(R.level + 1),
                       level=R.level + 1,
                       nodes=set(members),
                       parent=R)
            R.children.append(C)
            self._build_rec(C)

    # --------- borders ---------
    def compute_all_borders(self):
        t0 = time.time()
        for R in self.all_regions:
            bset: Set[Any] = set()
            nodes = R.nodes
            for u in nodes:
                for v in self.G[u]:
                    if v not in nodes:
                        bset.add(u); break
            R.borders = sorted(bset, key=stable_key)
            R.border_index = {b: i for i, b in enumerate(R.borders)}
        self.t_borders = time.time() - t0

    # --------- leaf PLL & leaf border table ---------
    @staticmethod
    def _leaf_pll_task(nodes: List[Any], edges: List[Tuple[Any, Any, float]]) -> Dict[Any, Dict[Any, float]]:
        subg = nx.Graph()
        subg.add_nodes_from(nodes)
        for u, v, w in edges:
            subg.add_edge(u, v, weight=float(w))
        try:
            ecc = {}
            for comp in nx.connected_components(subg):
                cg = subg.subgraph(comp)
                ecc.update(nx.eccentricity(cg))
            order = sorted(subg.nodes(), key=lambda n: ecc.get(n, 0))
        except Exception:
            order = list(subg.nodes())
        pll = PrunedPLLIndex(subg, order)
        pll.build()
        return pll.labels

    def build_leaf_pll(self):
        t0 = time.time()
        tasks = []
        for L in self.leaves:
            nset = L.nodes
            seen = set()
            edges: List[Tuple[Any, Any, float]] = []
            for u in nset:
                for v, data in self.G[u].items():
                    if v in nset:
                        key = (u, v) if stable_key(u) <= stable_key(v) else (v, u)
                        if key in seen: continue
                        seen.add(key)
                        edges.append((u, v, data.get("weight", 1.0)))
            tasks.append((L, list(nset), edges))
        futures = {}
        with ProcessPoolExecutor(max_workers=self.pll_workers) as ex:
            for L, nodes, edges in tasks:
                fut = ex.submit(GTreeBuilderALT._leaf_pll_task, nodes, edges)
                futures[fut] = L
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Build leaf PLL", unit="leaf"):
                labels = fut.result()
                L = futures[fut]
                L.leaf_pll = PrunedPLLIndex(nx.Graph(), None)
                L.leaf_pll.labels = labels
        self.t_leaf_pll = time.time() - t0

    def build_leaf_border_tables(self):
        t0 = time.time()
        for L in tqdm(self.leaves, desc="Leaf border tables (PLL)", unit="leaf"):
            B = len(L.borders)
            pack = np.full(tri_pack_len(B), np.inf, dtype=np.float32)
            for i in range(B):
                ui = L.borders[i]
                for j in range(i + 1, B):
                    vj = L.borders[j]
                    d = L.leaf_pll.query(ui, vj) if L.leaf_pll else float("inf")
                    unpack_set(pack, B, i, j, np.float32(d))
            L.border_pack = pack
        self.t_leaf_tables = time.time() - t0

    # --------- ALT (parents) ---------
    def build_global_landmarks(self):
        t0 = time.time()
        D_gl, node2id, _ = precompute_global_landmarks(self.G, k=self.L_glob, seed=self.lm_seed)
        self.D_gl = D_gl
        self.node2id = node2id
        self.t_landmarks = time.time() - t0

    def _heuristic_to_t(self, t: Any):
        D_gl = self.D_gl; node2id = self.node2id
        tt = node2id[t]
        col_t = D_gl[:, tt]
        def h(v: Any) -> float:
            vv = node2id[v]
            return float(np.max(np.abs(D_gl[:, vv] - col_t)))
        return h

    def _astar_multigoal_from_t(self, adj: Dict[Any, List[Tuple[Any, float]]], t: Any, targets: Set[Any]) -> Dict[Any, float]:
        h = self._heuristic_to_t(t)
        dist: Dict[Any, float] = {t: 0.0}
        pq = [(h(t), 0, t)]
        closed: Set[Any] = set()
        found: Dict[Any, float] = {}
        ctr = 1
        while pq and len(found) < len(targets):
            f, _, u = heapq.heappop(pq)
            if u in closed: continue
            closed.add(u)
            if u in targets and u not in found:
                found[u] = dist[u]
                if len(found) >= len(targets): break
            du = dist[u]
            for v, w in adj[u]:
                nd = du + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(pq, (nd + h(v), ctr, v)); ctr += 1
        for g in targets:
            if g not in found:
                found[g] = float("inf")
        return found

    @staticmethod
    def _dijkstra_multigoal_from_t(adj: Dict[Any, List[Tuple[Any, float]]], t: Any, targets: Set[Any]) -> Dict[Any, float]:
        dist: Dict[Any, float] = {t: 0.0}
        pq = [(0.0, 0, t)]; ctr = 1
        closed: Set[Any] = set()
        found: Dict[Any, float] = {}
        while pq and len(found) < len(targets):
            d, _, u = heapq.heappop(pq)
            if u in closed: continue
            closed.add(u)
            if u in targets and u not in found:
                found[u] = d
                if len(found) >= len(targets): break
            for v, w in adj[u]:
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, ctr, v)); ctr += 1
        for g in targets:
            if g not in found:
                found[g] = float("inf")
        return found

    # overlay 邻接：由子簇边界表和跨子簇原始边组成
    def _build_overlay_adjacency(self, P: Region) -> Tuple[Dict[Any, List[Tuple[Any, float]]], Dict[Any, int]]:
        overlay_nodes: Set[Any] = set()
        child_id: Dict[Any, int] = {}
        for ci, C in enumerate(P.children):
            for b in C.borders:
                overlay_nodes.add(b)
                child_id[b] = ci
        adj: Dict[Any, List[Tuple[Any, float]]] = {u: [] for u in overlay_nodes}
        # intra-child via child's border table
        for C in P.children:
            B = C.borders; nB = len(B)
            if nB <= 1: continue
            for i in range(nB):
                ui = B[i]
                for j in range(i + 1, nB):
                    vj = B[j]
                    w = unpack_get(C.border_pack, nB, i, j)
                    if not np.isfinite(w): continue
                    adj[ui].append((vj, float(w)))
                    adj[vj].append((ui, float(w)))
        # inter-child edges from original G
        for u in overlay_nodes:
            for v, data in self.G[u].items():
                if v in overlay_nodes and child_id[u] != child_id[v]:
                    w = float(data.get("weight", 1.0))
                    adj[u].append((v, w))
                    adj[v].append((u, w))
        return adj, child_id

    def _build_portals_for_root(self, root: Region):
        # 只在根层（|B_root|=0）需要 Top-K 门户
        if len(root.borders) != 0:
            return
        adj, child_id = self._build_overlay_adjacency(root)

        def topk_portals_pair(src_nodes: List[Any], dst_set: Set[Any], K: int) -> List[Tuple[Any, Any, float]]:
            if not src_nodes or not dst_set: return []
            dist: Dict[Any, float] = {}
            best_src: Dict[Any, Any] = {}
            pq = []
            ctr = 0
            for s in src_nodes:
                dist[s] = 0.0
                best_src[s] = s
                heapq.heappush(pq, (0.0, ctr, s)); ctr += 1
            visited: Set[Any] = set()
            out: List[Tuple[Any, Any, float]] = []
            used_dst: Set[Any] = set()
            while pq and len(out) < K:
                d, _, u = heapq.heappop(pq)
                if u in visited: continue
                visited.add(u)
                if u in dst_set and u not in used_dst:
                    out.append((best_src[u], u, d))
                    used_dst.add(u)
                    if len(out) >= K: break
                for v, w in adj[u]:
                    nd = d + w
                    if nd < dist.get(v, float("inf")):
                        dist[v] = nd
                        best_src[v] = best_src[u]
                        heapq.heappush(pq, (nd, ctr, v)); ctr += 1
            return out

        C = len(root.children)
        for i in range(C):
            Bi = list(root.children[i].borders); Si = set(Bi)
            for j in range(i + 1, C):
                Bj = list(root.children[j].borders); Sj = set(Bj)
                lst_ij = topk_portals_pair(Bi, Sj, self.portal_topk)
                lst_ji = topk_portals_pair(Bj, Si, self.portal_topk)
                merged = [(b, c, d) for (b, c, d) in lst_ij] + [(c, b, d) for (b, c, d) in lst_ji]
                merged.sort(key=lambda x: x[2])
                root.portals[(i, j)] = merged[:self.portal_topk]

    def build_parent_tables(self):
        # ALT landmarks（父簇用 ALT 时）
        if self.alg == "alt":
            self.build_global_landmarks()

        t0 = time.time()
        by_level = defaultdict(list)
        for R in self.all_regions:
            by_level[R.level].append(R)
        max_lv = max(by_level.keys())

        # 逐层自底向上构父簇：parent border tables + child->parent 矩阵
        for lv in range(max_lv - 1, -1, -1):
            parents = [p for p in by_level[lv] if p.children]
            for P in tqdm(parents, desc=f"Parent tables @ level {lv}", unit="parent"):
                # overlay 邻接
                adj, child_id = self._build_overlay_adjacency(P)

                # 父边界 pack
                Bp_list = P.borders
                Bp = len(Bp_list)
                pack = np.full(tri_pack_len(Bp), np.inf, dtype=np.float32)

                # 子->父矩阵占位
                for ci, C in enumerate(P.children):
                    if len(C.borders) == 0:
                        P.child_to_parent[ci] = np.zeros((0, Bp), dtype=np.float32)
                    else:
                        P.child_to_parent[ci] = np.full((len(C.borders), Bp), np.float32(np.inf))

                # 选择引擎
                use_alt = (self.alg == "alt" and len(Bp_list) > 0)
                engine = (self._astar_multigoal_from_t if use_alt else self._dijkstra_multigoal_from_t)

                targets_all: Set[Any] = set(adj.keys())

                # 对每个父边界 t 列，做一次多目标
                for j, tnode in enumerate(Bp_list):
                    found = engine(adj, tnode, targets_all)
                    # 填 pack
                    for i in range(j):
                        s = Bp_list[i]
                        ds = found.get(s, float("inf"))
                        if np.isfinite(ds):
                            unpack_set(pack, Bp, i, j, np.float32(ds))
                    # 填子->父（列 j）
                    for ci, C in enumerate(P.children):
                        for ri, b in enumerate(C.borders):
                            dv = found.get(b, float("inf"))
                            if np.isfinite(dv):
                                P.child_to_parent[ci][ri, j] = np.float32(dv)

                P.border_pack = pack

                # 根层（无边界）预计算 Top-K 门户
                if Bp == 0 and P is self.root:
                    self._build_portals_for_root(P)

        self.t_parent_tables = time.time() - t0

    # --------- 叶 → 祖先边界：预计算（核心提速） ---------
    def _compute_leaf_to_region_direct(self, L: Region) -> np.ndarray:
        """叶 L：返回 |nodes(L)| × |B_L|，用 PLL 查表逐元素填充。"""
        NL = len(L.node_list); BL = len(L.borders)
        M = np.full((NL, BL), np.inf, dtype=np.float32)
        if BL == 0:
            return M
        for j, b in enumerate(L.borders):
            for i, u in enumerate(L.node_list):
                d = L.leaf_pll.query(u, b) if L.leaf_pll else float("inf")
                M[i, j] = np.float32(d)
        return M

    @staticmethod
    def _minplus_chunk(A: np.ndarray, M: np.ndarray, chunk_rows: int) -> np.ndarray:
        """
        计算 B = A ⊗ M （min-plus），A:[r×m], M:[m×n] → B:[r×n]，分块按行处理。
        """
        r, m = A.shape; m2, n = M.shape
        assert m == m2
        B = np.full((r, n), np.inf, dtype=np.float32)
        for s in range(0, r, chunk_rows):
            e = min(r, s + chunk_rows)
            As = A[s:e, :]  # [rs×m]
            tmp = As[:, :, None] + M[None, :, :]
            Bs = np.min(tmp, axis=1)  # [rs×n]
            B[s:e, :] = Bs.astype(np.float32, copy=False)
        return B

    def build_leaf_to_ancestor_tables(self):
        t0 = time.time()
        # 先为所有叶填充叶→自身边界
        for L in tqdm(self.leaves, desc="Leaf→LeafBorders (PLL)", unit="leaf"):
            M_L = self._compute_leaf_to_region_direct(L)   # |V_L| × |B_L|
            self.leaf2anc[L][L.cid] = M_L

        # 对每个叶 L，沿 parent 链一路提到根
        for L in tqdm(self.leaves, desc="Leaf→AncestorBorders (min-plus lift)", unit="leaf"):
            cur = L
            A = self.leaf2anc[L][L.cid]   # 当前在 |B_cur| 上的矩阵
            while cur.parent is not None:
                P = cur.parent
                ci = P.children.index(cur)
                M = P.child_to_parent.get(ci)
                if M is None or M.size == 0:
                    Bp = len(P.borders)
                    A = np.full((len(L.node_list), Bp), np.inf, dtype=np.float32)
                else:
                    A = self._minplus_chunk(A, M, self.chunk_rows)
                self.leaf2anc[L][P.cid] = A
                cur = P
        self.t_leaf2anc = time.time() - t0

    # --------- LCA & query（全查表/Top-K O(1) 级别） ---------
    def find_lca(self, A: Region, B: Region) -> Region:
        if A is B: return A
        seen = set()
        a, b = A, B
        while a or b:
            if a:
                if a in seen: return a
                seen.add(a); a = a.parent
            if b:
                if b in seen: return b
                seen.add(b); b = b.parent
        return A

    def _child_under(self, leaf: Region, ancestor: Region) -> Region:
        cur = leaf
        while cur.parent is not ancestor:
            cur = cur.parent
        return cur

    def query_distance(self, s: Any, t: Any) -> float:
        Ls = self.node2leaf.get(s); Lt = self.node2leaf.get(t)
        if Ls is None or Lt is None: return float("inf")
        if Ls is Lt:
            return float(Ls.leaf_pll.query(s, t) if Ls.leaf_pll else float("inf"))

        A = self.find_lca(Ls, Lt)

        # LCA 有边界：直接取预计算矩阵的两行相加取最小
        if len(A.borders) > 0:
            Ms = self.leaf2anc[Ls][A.cid]  # |V_Ls| × |B_A|
            Mt = self.leaf2anc[Lt][A.cid]  # |V_Lt| × |B_A|
            is_ = Ls.node_index[s]; it_ = Lt.node_index[t]
            vs = Ms[is_, :]  # 向量
            vt = Mt[it_, :]
            return float(np.min(vs + vt))

        # LCA=根：用根的 Top-K 门户 (b,c,dmid)
        root = self.root
        cs = root.children.index(self._child_under(Ls, root))
        ct = root.children.index(self._child_under(Lt, root))
        i, j = (cs, ct) if cs < ct else (ct, cs)
        portals = root.portals.get((i, j), [])
        if not portals:
            return float("inf")

        left_child = root.children[cs]
        right_child = root.children[ct]
        Ms = self.leaf2anc[Ls][left_child.cid]   # |V_Ls| × |B_left|
        Mt = self.leaf2anc[Lt][right_child.cid]  # |V_Lt| × |B_right|
        is_ = Ls.node_index[s]; it_ = Lt.node_index[t]
        vs = Ms[is_, :]; vt = Mt[it_, :]

        idx_left = left_child.border_index
        idx_right = right_child.border_index

        best = float("inf")
        if cs < ct:
            for (b, c, mid) in portals:
                ib = idx_left.get(b, -1); ic = idx_right.get(c, -1)
                if ib < 0 or ic < 0: continue
                du = float(vs[ib]); dv = float(vt[ic])
                if not np.isfinite(du) or not np.isfinite(dv): continue
                val = du + mid + dv
                if val < best: best = val
        else:
            for (b, c, mid) in portals:  # 反向
                ib = idx_right.get(b, -1); ic = idx_left.get(c, -1)
                if ib < 0 or ic < 0: continue
                du = float(vs[ic]); dv = float(vt[ib])
                if not np.isfinite(du) or not np.isfinite(dv): continue
                val = du + mid + dv
                if val < best: best = val
        return best


# ---------------- evaluation ----------------
def evaluate(G: nx.Graph, gt: GTreeBuilderALT, pairs: int, build_total: float) -> pd.DataFrame:
    rng = random.Random(42)
    nodes = list(G.nodes())
    samples = []
    while len(samples) < pairs:
        u = rng.choice(nodes); v = rng.choice(nodes)
        if u == v: continue
        try:
            d = nx.shortest_path_length(G, u, v, weight="weight")
            samples.append((u, v, d))
        except nx.NetworkXNoPath:
            continue

    t0 = time.time()
    exact = 0; total = 0; err = 0.0
    for u, v, d in samples:
        est = gt.query_distance(u, v)
        if np.isfinite(est):
            total += 1
            if abs(est - d) <= 1e-8:
                exact += 1
            err += abs(est - d)
    qtime = time.time() - t0

    rows = [[
        "G-Tree (parents=ALT, leaf=PLL, precomputed leaf→ancestor + root Top-K)",
        build_total,
        qtime,
        len(samples),
        exact,
        (err/total if total>0 else float("inf"))
    ]]
    return pd.DataFrame(rows, columns=["method", "build_time_sec", "query_time_sec", "samples", "exact_matches", "mae"])


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="Pubmed", choices=["Cora", "CiteSeer", "Pubmed"])
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--kg_file", type=str, default=None, help="Path to WN*.txt (entity+relation nodes).")

    ap.add_argument("--resolution", type=float, default=0.3)
    ap.add_argument("--max_cluster_size", type=int, default=1000)
    ap.add_argument("--hl_max_levels", type=int, default=3)

    ap.add_argument("--pll_workers", type=int, default=None)

    ap.add_argument("--alg", type=str, default="alt", choices=["alt", "dijkstra"], help="parent overlay solver")
    ap.add_argument("--L_glob", type=int, default=32, help="#global landmarks for ALT")
    ap.add_argument("--lm_seed", type=int, default=42, help="random seed for landmark selection")

    ap.add_argument("--portal_topk", type=int, default=16, help="Top-K portals per child pair at root")
    ap.add_argument("--chunk_rows", type=int, default=512, help="row chunk for min-plus lifting")

    ap.add_argument("--pairs", type=int, default=300)

    args = ap.parse_args()

    # load graph
    if args.kg_file:
        G = load_wn_graph_relation_nodes(args.kg_file)
        src_name = f"WN18({os.path.basename(args.kg_file)})-rel_nodes"
    else:
        G = load_planetoid_graph(args.dataset, root=args.data_root)
        src_name = args.dataset

    print(f"[INFO] Graph: {src_name}, |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

    gt = GTreeBuilderALT(
        G,
        max_cluster_size=args.max_cluster_size,
        max_levels=args.hl_max_levels,
        resolution=args.resolution,
        random_seed=42,
        pll_workers=args.pll_workers,
        alg=args.alg,
        L_glob=args.L_glob,
        lm_seed=args.lm_seed,
        portal_topk=args.portal_topk,
        chunk_rows=args.chunk_rows,
    )

    # ===== build (timed) =====
    t_start = time.time()

    root = gt.build_tree()
    sizes = Counter(len(r.nodes) for r in gt.all_regions if not r.children)
    print(f"[INFO] #leaf_clusters = {len(gt.leaves)}, leaf size stats (top 10): {sizes.most_common(10)}")

    gt.compute_all_borders()
    gt.build_leaf_pll()
    gt.build_leaf_border_tables()
    gt.build_parent_tables()
    gt.build_leaf_to_ancestor_tables()

    build_total = time.time() - t_start
    print(f"[INFO] BuildTime detail:"
          f" tree={gt.t_build_tree:.3f}s,"
          f" borders={gt.t_borders:.3f}s,"
          f" leafPLL={gt.t_leaf_pll:.3f}s,"
          f" leafTables={gt.t_leaf_tables:.3f}s,"
          f" landmarks={gt.t_landmarks:.3f}s,"
          f" parentTables={gt.t_parent_tables:.3f}s,"
          f" leaf→ancestor={gt.t_leaf2anc:.3f}s,"
          f" TOTAL={build_total:.3f}s")

    # ===== evaluation =====
    df = evaluate(G, gt, pairs=args.pairs, build_total=build_total)
    print("\n=== Evaluation (G-Tree tables) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
