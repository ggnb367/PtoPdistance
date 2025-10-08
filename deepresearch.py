#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HierL-E — 全表预处理版（查询严格只查表）
- 叶内 PLL + G_nb
- 父层 index_graph（网关收缩 + kNN）
- 多源 Dijkstra 得 AB 表 (A→B)
- SAME 表（per-leaf CSC）
- 新增：Leaf→Pair-Columns 预表（L2B/L2C；查询直接取行向量 + AB 合并）
"""

from __future__ import annotations
import argparse
import os
import time
import heapq
import random
from collections import defaultdict, deque
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
    return isinstance(n, tuple) and len(n) > 0 and n[0] == "REL"

def node_sort_key(n):
    if is_rel(n):
        tag, idx, rel = n
        return (1, int(idx), str(rel))
    else:
        return (0, 0, str(n))


# ---------- data loaders ----------
def load_planetoid_graph(name="Pubmed", root=None) -> nx.Graph:
    if Planetoid is None:
        raise RuntimeError("未安装 torch_geometric；请安装或使用 --kg_file 加载 WN18。")
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


def load_wn18_graph_aggregated(path: str, rel_weight: dict | None = None) -> nx.Graph:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"WN18 file not found: {path}")
    rel_weight = rel_weight or {}
    pair2rels: dict[tuple, dict] = {}

    def add_triple(h, t, r):
        if h == t:
            return
        a, b = (h, t) if str(h) <= str(t) else (t, h)
        w = float(rel_weight.get(r, 1.0))
        pair2rels.setdefault((a, b), {})
        pair2rels[(a, b)][r] = min(w, pair2rels[(a, b)].get(r, float("inf")))

    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
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
    for (h, t), rmap in pair2rels.items():
        G.add_edge(
            h, t,
            weight=min(rmap.values()) if rmap else 1.0,
            rels=tuple(sorted(rmap.keys(), key=str)),
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

    def query(self, u, v) -> float:
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


# ---------- HL final clusters ----------
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


# ---------- Step1: skeleton ----------
def build_skeleton_leaves(G: nx.Graph, hl_resolution=0.3, hl_max_cluster_size=1200, random_seed=42):
    cluster_tree = {}
    node_to_leaf = {}
    cc_to_leaves = defaultdict(list)

    SUPER_ROOT = ("SUPER_ROOT",)
    cluster_tree[SUPER_ROOT] = dict(level=-2, parent=None, children=[], nodes=None)

    comps = list(nx.connected_components(G))
    comps_sorted = sorted(comps, key=lambda s: -len(s))
    for cc_id, nodes in enumerate(comps_sorted):
        subg = G.subgraph(nodes).copy()
        cc_cid = ("cc", cc_id)
        cluster_tree[cc_cid] = dict(level=-1, parent=SUPER_ROOT, children=[], nodes=set(nodes))
        cluster_tree[SUPER_ROOT]["children"].append(cc_cid)

        hl = hierarchical_leiden(
            subg,
            max_cluster_size=hl_max_cluster_size,
            resolution=hl_resolution,
            use_modularity=True,
            random_seed=random_seed,
            check_directed=True,
        )
        part = final_partition_from_hl(hl, subg)
        cid2nodes = defaultdict(list)
        for n, local_cid in part.items():
            cid2nodes[local_cid].append(n)
        for k, nlist in cid2nodes.items():
            leaf_cid = ("leaf", cc_id, int(k))
            cluster_tree[leaf_cid] = dict(level=0, parent=cc_cid, children=[], nodes=set(nlist))
            cluster_tree[cc_cid]["children"].append(leaf_cid)
            for n in nlist:
                node_to_leaf[n] = leaf_cid
            cc_to_leaves[cc_id].append(leaf_cid)
    return cluster_tree, node_to_leaf, cc_to_leaves, SUPER_ROOT


# ---------- Step2: per-leaf PLL + borders + kNN rep_graph ----------
def _build_one_leaf(args):
    leaf_cid, leaf_nodes, neigh_map, G, leaf_rg_k = args
    nodes = sorted(leaf_nodes, key=node_sort_key)
    node_set = set(nodes)

    border = [n for n in nodes if any((nbr not in node_set) for nbr in neigh_map[n])]
    border = sorted(border, key=node_sort_key)
    idx_map = {b: i for i, b in enumerate(border)}

    subg = nx.Graph()
    subg.add_nodes_from(nodes)
    for u in nodes:
        for v, data in G[u].items():
            if v in node_set and u < v:
                subg.add_edge(u, v, weight=float(data.get("weight", 1.0)))

    try:
        ecc = {}
        for comp in nx.connected_components(subg):
            sg = subg.subgraph(comp)
            ecc.update(nx.eccentricity(sg))
        order = sorted(nodes, key=lambda n: ecc.get(n, 0))
    except Exception:
        order = list(nodes)

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
    label_entries = sum(len(m) for m in pll.labels.values())

    # G_nb：节点→边界数组
    G_nb = {}
    L = len(border)
    for n in nodes:
        arr = np.empty((L,), dtype=np.float32)
        for i, b in enumerate(border):
            arr[i] = pll.query(n, b)
        G_nb[n] = arr

    # 叶 rep_graph：kNN 闭包
    RG = nx.Graph()
    RG.add_nodes_from(border)
    if L > 1:
        k = leaf_rg_k if leaf_rg_k > 0 else (L - 1)
        for i, bi in enumerate(border):
            arr = G_nb[bi].copy()
            arr[i] = np.inf
            if k < L - 1:
                sel = np.argpartition(arr, k)[:k]
            else:
                sel = np.where(np.isfinite(arr))[0]
            for j in sel:
                bj = border[int(j)]
                dij = float(arr[int(j)])
                if not np.isfinite(dij):
                    continue
                if RG.has_edge(bi, bj):
                    if dij < RG[bi][bj].get("weight", float("inf")):
                        RG[bi][bj]["weight"] = dij
                else:
                    RG.add_edge(bi, bj, weight=dij)

    return leaf_cid, border, pll.labels, G_nb, RG, label_entries


def build_leaves_tables(G: nx.Graph, cluster_tree: dict, node_to_leaf: dict, leaf_rg_k=8, max_workers=None):
    max_workers = max_workers or (os.cpu_count() or 2)
    neigh_map = {n: list(G.neighbors(n)) for n in G.nodes()}
    tasks = []
    for cid, meta in cluster_tree.items():
        if meta["level"] == 0:
            tasks.append((cid, meta["nodes"], neigh_map, G, leaf_rg_k))

    borders, pll_labels, G_nb_all, rep_graphs, stats = {}, {}, {}, {}, {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_build_one_leaf, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Step2: building leaves", unit="leaf"):
            leaf_cid, border, labels, G_nb, RG, label_entries = fut.result()
            borders[leaf_cid] = border
            pll_labels[leaf_cid] = labels
            G_nb_all[leaf_cid] = G_nb
            rep_graphs[leaf_cid] = RG
            stats[leaf_cid] = dict(nodes=len(cluster_tree[leaf_cid]["nodes"]),
                                   borders=len(border),
                                   pll_label_entries=label_entries,
                                   rep_edges=RG.number_of_edges())
    return borders, pll_labels, G_nb_all, rep_graphs, stats


# ---------- Step3: 父层 index_graph（网关收缩 + 子簇内网关 kNN 闭包） ----------
def _dijkstra_to_targets(graph: nx.Graph, src, targets_set: set) -> dict:
    if src not in graph:
        return {}
    dist = {src: 0.0}
    heap = [(0.0, src)]
    remain = set(x for x in targets_set if x in graph)
    out = {}
    while heap and remain:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        if u in remain:
            out[u] = d
            remain.remove(u)
            if not remain:
                break
        for v, data in graph[u].items():
            w = float(data.get("weight", 1.0))
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return out

def build_index_graphs_for_parents(
    G: nx.Graph,
    cluster_tree: dict,
    borders: dict,
    rep_graphs: dict,
    parents_subset: list | None = None,
    ig_gateway_only: bool = True,
    ig_intra_knn: int = 8,
):
    index_graphs = {}
    stats = {}

    if parents_subset is None:
        parents = [cid for cid, meta in cluster_tree.items() if meta["children"] and len(meta["children"]) > 1]
    else:
        parents = [p for p in parents_subset if cluster_tree.get(p, {}).get("children")]

    for parent in tqdm(parents, desc="Step3: building index_graphs (parents)", unit="parent"):
        children = cluster_tree[parent]["children"]
        child_to_nodes = {}
        node_to_child = {}
        for ch in children:
            RG = rep_graphs.get(ch)
            nodes_set = set(RG.nodes()) if RG is not None else set()
            child_to_nodes[ch] = nodes_set
            for x in nodes_set:
                node_to_child[x] = ch

        gateway_by_child = {ch: set() for ch in children}
        if ig_gateway_only:
            all_rep = set(node_to_child.keys())
            for b in all_rep:
                ch_b = node_to_child.get(b)
                for v in G[b]:
                    if v in all_rep:
                        ch_v = node_to_child.get(v)
                        if ch_v is not None and ch_v != ch_b:
                            gateway_by_child[ch_b].add(b)
                            gateway_by_child[ch_v].add(v)
        else:
            for ch, nodes_set in child_to_nodes.items():
                gateway_by_child[ch] = set(nodes_set)

        IG = nx.Graph()
        index_nodes = set().union(*gateway_by_child.values())
        IG.add_nodes_from(index_nodes)

        # intra: kNN on rep_graph among gateways
        for ch, gates in gateway_by_child.items():
            if not gates:
                continue
            RG = rep_graphs.get(ch)
            if RG is None or RG.number_of_nodes() == 0:
                continue
            gates_list = list(gates)
            targets_set = set(gates_list)
            for src in gates_list:
                dmap = _dijkstra_to_targets(RG, src, targets_set - {src})
                if not dmap:
                    continue
                items = list(dmap.items())
                if ig_intra_knn > 0 and len(items) > ig_intra_knn:
                    items = heapq.nsmallest(ig_intra_knn, items, key=lambda kv: kv[1])
                for tgt, dist_val in items:
                    if src == tgt or not np.isfinite(dist_val):
                        continue
                    if IG.has_edge(src, tgt):
                        if dist_val < IG[src][tgt].get("weight", float("inf")):
                            IG[src][tgt]["weight"] = float(dist_val)
                    else:
                        IG.add_edge(src, tgt, weight=float(dist_val))

        # inter: original cross-child edges (gateways only)
        all_gate = set().union(*gateway_by_child.values())
        for b in all_gate:
            ch_b = node_to_child.get(b)
            for v, data in G[b].items():
                if v not in all_gate:
                    continue
                ch_v = node_to_child.get(v)
                if ch_v is None or ch_v == ch_b:
                    continue
                w = float(data.get("weight", 1.0))
                if IG.has_edge(b, v):
                    if w < IG[b][v].get("weight", float("inf")):
                        IG[b][v]["weight"] = w
                else:
                    IG.add_edge(b, v, weight=w)

        index_graphs[parent] = IG
        stats[parent] = dict(nodes=IG.number_of_nodes(), edges=IG.number_of_edges(),
                             gateway=sum(len(s) for s in gateway_by_child.values()))
    return index_graphs, stats


# ---------- Step4: list_indexpath（多源 Dijkstra：子簇为单位） ----------
def _make_adj_from_graph(G: nx.Graph):
    return {u: [(v, float(d.get("weight", 1.0))) for v, d in G[u].items()] for u in G.nodes()}

def _multi_source_dijkstra_index_worker(args):
    parent, A, adj, child_to_gates, node_to_child = args
    seeds = list(child_to_gates.get(A, []))
    if not seeds:
        return parent, A, {}
    dist, origin = {}, {}
    heap = []
    ctr = 0
    for b in seeds:
        dist[b] = 0.0
        origin[b] = b
        heapq.heappush(heap, (0.0, ctr, b)); ctr += 1
    while heap:
        d, _, u = heapq.heappop(heap)
        if d != dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                origin[v] = origin[u]
                heapq.heappush(heap, (nd, ctr, v)); ctr += 1
    resA = defaultdict(dict)
    for c, dc in dist.items():
        b0 = origin.get(c)
        if b0 is None:
            continue
        prev = resA[b0].get(c)
        if (prev is None) or (dc < prev):
            resA[b0][c] = float(dc)
    return parent, A, resA

def build_list_indexpath_for_parents(cluster_tree, index_graphs, rep_graphs, parents_subset=None):
    L = defaultdict(lambda: defaultdict(dict))
    tasks = []

    if parents_subset is None:
        parents = [cid for cid, meta in cluster_tree.items() if meta["children"] and len(meta["children"]) > 1]
    else:
        parents = [p for p in parents_subset if cluster_tree.get(p, {}).get("children")]

    for parent in parents:
        IG = index_graphs.get(parent)
        if IG is None or IG.number_of_nodes() == 0:
            continue
        child_to_gates = {}
        node_to_child = {}
        for ch in cluster_tree[parent]["children"]:
            RG = rep_graphs.get(ch)
            nodes_set = set(RG.nodes()) if RG is not None else set()
            gates = [x for x in nodes_set if x in IG]
            child_to_gates[ch] = set(gates)
            for x in gates:
                node_to_child[x] = ch
        adj = _make_adj_from_graph(IG)
        for A, seeds in child_to_gates.items():
            if seeds:
                tasks.append((parent, A, adj, child_to_gates, node_to_child))

    with ProcessPoolExecutor(max_workers=os.cpu_count() or 2) as ex:
        futs = [ex.submit(_multi_source_dijkstra_index_worker, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Step4: list_indexpath (multi-source)", unit="job"):
            parent, A, resA = fut.result()
            L[parent][A] = resA
    return L


# ---------- Step7: AB tables ----------
def build_AB_tables(cluster_tree, rep_graphs, list_indexpath_total, mid_limit=0):
    AB_tables = defaultdict(dict)
    for L, mapA in list_indexpath_total.items():
        node_to_child = {}
        for ch in cluster_tree[L]["children"]:
            RG = rep_graphs.get(ch)
            nodes_set = RG.nodes() if RG is not None else []
            for x in nodes_set:
                node_to_child[x] = ch

        pair_bucket = defaultdict(lambda: defaultdict(list))  # (A,B) -> b -> [(mid,c)]
        for A, resA in mapA.items():
            for b, cmap in resA.items():
                for c, dist_bc in cmap.items():
                    B = node_to_child.get(c)
                    if B is None or B == A:
                        continue
                    pair_bucket[(A, B)][b].append((float(dist_bc), c))

        for (A, B), by_b in pair_bucket.items():
            b_nodes = sorted(by_b.keys(), key=node_sort_key)
            c_nodes_set = set()
            by_b_kept = {}
            for b in b_nodes:
                lst = by_b[b]
                if mid_limit and len(lst) > mid_limit:
                    keep = heapq.nsmallest(mid_limit, lst, key=lambda x: x[0])
                else:
                    keep = lst
                by_b_kept[b] = keep
                for _, c in keep:
                    c_nodes_set.add(c)
            c_nodes = sorted(c_nodes_set, key=node_sort_key)
            if not b_nodes or not c_nodes:
                continue
            b_index = {b: i for i, b in enumerate(b_nodes)}
            c_index = {c: i for i, c in enumerate(c_nodes)}

            rows, cols, mids, row_ptr = [], [], [], [0]
            for b in b_nodes:
                for dist_bc, c in by_b_kept[b]:
                    rows.append(b_index[b])
                    cols.append(c_index[c])
                    mids.append(dist_bc)
                row_ptr.append(len(rows))

            AB_tables[L][(A, B)] = dict(
                b_nodes=b_nodes, c_nodes=c_nodes,
                b_index=b_index, c_index=c_index,
                rows=np.asarray(rows, dtype=np.int32),
                cols=np.asarray(cols, dtype=np.int32),
                mids=np.asarray(mids, dtype=np.float32),
                row_ptr=np.asarray(row_ptr, dtype=np.int32),
            )
    return AB_tables


# ---------- Step7': SAME tables（per-leaf CSC） ----------
def _collect_leaves_under(cluster_tree, child):
    leaves = []
    dq = deque([child])
    while dq:
        c = dq.popleft()
        if cluster_tree[c]["level"] == 0:
            leaves.append(c)
        else:
            dq.extend(cluster_tree[c]["children"])
    return leaves

def build_SAME_tables(cluster_tree, borders, rep_graphs, G_nb_all):
    SAME = defaultdict(dict)
    for L, meta in cluster_tree.items():
        chs = meta.get("children")
        if not chs or len(chs) <= 1:
            continue
        for A in chs:
            RG = rep_graphs.get(A)
            if RG is None or RG.number_of_nodes() == 0:
                continue
            cols_nodes = sorted(RG.nodes(), key=node_sort_key)
            col_index = {b: i for i, b in enumerate(cols_nodes)}
            C = len(cols_nodes)
            leaves = _collect_leaves_under(cluster_tree, A)

            row_ptr = [0]
            cols: list[int] = []
            vals: list[float] = []
            nnz_row_index: list[int] = []
            leaf_rows: dict = {}
            total_rows = 0

            for leaf in leaves:
                blist = borders.get(leaf, [])
                if not blist:
                    leaf_rows[leaf] = (total_rows, total_rows)
                    continue
                idx_leaf = {x: i for i, x in enumerate(blist)}
                col_in_leaf = [b for b in cols_nodes if b in idx_leaf]
                col_pos = np.array([col_index[b] for b in col_in_leaf], dtype=np.int32)
                pick_idx = np.array([idx_leaf[b] for b in col_in_leaf], dtype=np.int32)

                row_start = total_rows
                for ridx_local, b0 in enumerate(blist):
                    arr = G_nb_all[leaf][b0]
                    dvals = arr[pick_idx] if pick_idx.size > 0 else np.empty((0,), dtype=np.float32)
                    finite = np.isfinite(dvals)
                    if finite.any():
                        use_cols = col_pos[finite]
                        use_vals = dvals[finite]
                        cols.extend(use_cols.tolist())
                        vals.extend(use_vals.astype(np.float32).tolist())
                        nnz_row_index.extend([ridx_local] * use_cols.size)
                    row_ptr.append(len(cols))
                    total_rows += 1
                row_end = total_rows
                leaf_rows[leaf] = (row_start, row_end)

            if total_rows == 0:
                continue

            sigma_col = np.full((C,), np.inf, dtype=np.float32)
            if len(cols) > 0:
                np.minimum.at(sigma_col, np.asarray(cols, dtype=np.int32), np.asarray(vals, dtype=np.float32))

            leaf_csc = {}
            row_ptr = np.asarray(row_ptr, dtype=np.int32)
            cols = np.asarray(cols, dtype=np.int32)
            vals = np.asarray(vals, dtype=np.float32)
            nnz_row_index = np.asarray(nnz_row_index, dtype=np.int32)

            for leaf, (r0, r1) in leaf_rows.items():
                if r0 == r1:
                    leaf_csc[leaf] = dict(col_ptr=np.zeros((C+1,), dtype=np.int32),
                                          row_idx=np.zeros((0,), dtype=np.int32),
                                          vals=np.zeros((0,), dtype=np.float32))
                    continue
                s, e = row_ptr[r0], row_ptr[r1]
                if s == e:
                    leaf_csc[leaf] = dict(col_ptr=np.zeros((C+1,), dtype=np.int32),
                                          row_idx=np.zeros((0,), dtype=np.int32),
                                          vals=np.zeros((0,), dtype=np.float32))
                    continue
                cols_se = cols[s:e]
                vals_se = vals[s:e]
                rows_local = nnz_row_index[s:e]

                counts = np.bincount(cols_se, minlength=C).astype(np.int32)
                col_ptr = np.empty((C+1,), dtype=np.int32)
                np.cumsum(np.concatenate(([0], counts)), out=col_ptr)

                row_idx = np.empty_like(rows_local)
                vals_c = np.empty_like(vals_se)
                next_pos = col_ptr[:-1].copy()
                for i in range(cols_se.size):
                    j = cols_se[i]
                    p = next_pos[j]
                    row_idx[p] = rows_local[i]
                    vals_c[p] = vals_se[i]
                    next_pos[j] = p + 1

                leaf_csc[leaf] = dict(col_ptr=col_ptr, row_idx=row_idx, vals=vals_c)

            SAME[L][A] = dict(
                cols_nodes=cols_nodes, col_index=col_index,
                row_ptr=row_ptr, cols=cols, vals=vals, nnz_row_index=nnz_row_index, leaf_rows=leaf_rows,
                sigma_col=sigma_col,
                leaf_csc=leaf_csc,
            )
    return SAME


# ---------- Step7.9: Leaf→Pair-Columns 预表（核心新增） ----------
def build_leaf2pair_tables(
    cluster_tree, borders, G_nb_all,
    SAME_tables, AB_tables,
    dtype: str = "fp16",  # "fp16" or "fp32"
):
    """
    为每个父层 L、每个 (A,B) 对，预先构建：
      L2B[L][(A,B)][leaf_under_A] : [|nodes(leaf)|, |b_nodes(AB)|] 的矩阵（与 AB 列顺序一致）
      L2C[L][(A,B)][leaf_under_B] : [|nodes(leaf)|, |c_nodes(AB)|]
    计算：d_A(n,b) = min_{b0∈border(leaf)} G_nb[n,b0] + d(b0,b)
          其中 d(b0,b) 来自 SAME[L][A].leaf_csc[leaf] 的列 j（j=col_index[b]）
    备注：矩阵 dtype 默认 float16（可切换 float32）
    """
    use_dtype = np.float16 if dtype == "fp16" else np.float32

    # 预先缓存每个叶子的“节点顺序 + 堆叠 G_nb”
    leaf_node_order = {}
    leaf_row_index = {}
    leaf_G_stack = {}
    for leaf, blist in borders.items():
        nlist = sorted(cluster_tree[leaf]["nodes"], key=node_sort_key)
        leaf_node_order[leaf] = nlist
        leaf_row_index[leaf] = {n: i for i, n in enumerate(nlist)}
        if len(blist) == 0 or len(nlist) == 0:
            leaf_G_stack[leaf] = np.zeros((0, 0), dtype=np.float32)
        else:
            # 堆叠为 [N, Lb]
            Gmat = np.vstack([G_nb_all[leaf][n] for n in nlist]).astype(np.float32, copy=False)
            leaf_G_stack[leaf] = Gmat  # float32 精度更好，末尾再转换

    L2B = defaultdict(dict)
    L2C = defaultdict(dict)

    for L, pairs in AB_tables.items():
        same_for_L = SAME_tables.get(L, {})
        if not same_for_L:
            continue

        for (A, B), AB in tqdm(pairs.items(), desc=f"Step7.9: Leaf→Pair-Columns @L={L}", unit="pair"):
            # A 侧
            SAME_A = same_for_L.get(A)
            if SAME_A is not None:
                b_nodes = AB["b_nodes"]
                col_index_A = SAME_A["col_index"]
                leaves_A = _collect_leaves_under(cluster_tree, A)
                for leaf in leaves_A:
                    nlist = leaf_node_order[leaf]
                    if len(nlist) == 0:
                        continue
                    G_stack = leaf_G_stack[leaf]      # [N, Lb]
                    Lb = G_stack.shape[1]
                    # 准备输出 [N, nb]
                    nb = len(b_nodes)
                    out = np.full((G_stack.shape[0], nb), np.inf, dtype=np.float32)

                    # 取出这个 leaf 的 CSC 视图
                    csc = SAME_A["leaf_csc"].get(leaf)
                    if csc:
                        col_ptr = csc["col_ptr"]; row_idx = csc["row_idx"]; vals = csc["vals"]
                        for j_ab, b in enumerate(b_nodes):
                            j_same = col_index_A.get(b, None)
                            if j_same is None:
                                continue
                            s, e = col_ptr[j_same], col_ptr[j_same+1]
                            if s == e:
                                continue
                            # 该列的 nnz：若干个边界行 idx (局部), 以及它们到 b 的距离
                            rows_local = row_idx[s:e]   # 这些是边界在 leaf 中的局部索引（0..Lb-1）
                            w = vals[s:e].astype(np.float32, copy=False)  # shape [k]
                            # 计算 min_n ( G_stack[:, rows_local] + w )
                            #  => 广播：G_stack[:, rows_local] shape [N,k] + w[None,:] -> [N,k]，沿 axis=1 取最小
                            cand = G_stack[:, rows_local] + w[None, :]
                            out[:, j_ab] = np.min(cand, axis=1)

                    # 存入 L2B
                    if out.size > 0:
                        L2B[L].setdefault((A, B), {})[leaf] = dict(
                            nodes=nlist, mat=out.astype(use_dtype, copy=False)  # float16/32
                        )

            # B 侧
            SAME_B = same_for_L.get(B)
            if SAME_B is not None:
                c_nodes = AB["c_nodes"]
                col_index_B = SAME_B["col_index"]
                leaves_B = _collect_leaves_under(cluster_tree, B)
                for leaf in leaves_B:
                    nlist = leaf_node_order[leaf]
                    if len(nlist) == 0:
                        continue
                    G_stack = leaf_G_stack[leaf]
                    Lb = G_stack.shape[1]
                    nc = len(c_nodes)
                    out = np.full((G_stack.shape[0], nc), np.inf, dtype=np.float32)

                    csc = SAME_B["leaf_csc"].get(leaf)
                    if csc:
                        col_ptr = csc["col_ptr"]; row_idx = csc["row_idx"]; vals = csc["vals"]
                        for j_ab, c in enumerate(c_nodes):
                            j_same = col_index_B.get(c, None)
                            if j_same is None:
                                continue
                            s, e = col_ptr[j_same], col_ptr[j_same+1]
                            if s == e:
                                continue
                            rows_local = row_idx[s:e]
                            w = vals[s:e].astype(np.float32, copy=False)
                            cand = G_stack[:, rows_local] + w[None, :]
                            out[:, j_ab] = np.min(cand, axis=1)

                    if out.size > 0:
                        L2C[L].setdefault((A, B), {})[leaf] = dict(
                            nodes=nlist, mat=out.astype(use_dtype, copy=False)
                        )

    # 返回同时附带 leaf 的 row_index（一次构建，全局复用）
    leaf_row_index = {leaf: {n: i for i, n in enumerate(nodes)} for leaf, nodes in leaf_node_order.items()}
    return L2B, L2C, leaf_row_index


# ---------- 查询辅助 ----------
def build_aux_indices_for_query(borders, pll_labels):
    leaf_border_index = {leaf: {b: i for i, b in enumerate(blist)} for leaf, blist in borders.items()}
    pll_obj = {leaf: PrunedPLLIndex.from_labels(labels) for leaf, labels in pll_labels.items()}
    return leaf_border_index, pll_obj

def get_path_to_root(cluster_tree, cid):
    path, cur = [], cid
    while cur is not None:
        path.append(cur)
        cur = cluster_tree[cur]["parent"] if cur in cluster_tree else None
    return path

def find_lca_and_children(cluster_tree, node_to_leaf, u, v):
    leaf_u = node_to_leaf[u]; leaf_v = node_to_leaf[v]
    if leaf_u == leaf_v:
        return leaf_u, leaf_u, leaf_v, leaf_u, leaf_v
    path_u = get_path_to_root(cluster_tree, leaf_u)
    path_v = get_path_to_root(cluster_tree, leaf_v)
    set_u = set(path_u)
    L = next((x for x in path_v if x in set_u), None)
    Au = leaf_u
    while cluster_tree[Au]["parent"] != L:
        Au = cluster_tree[Au]["parent"]
    Bv = leaf_v
    while cluster_tree[Bv]["parent"] != L:
        Bv = cluster_tree[Bv]["parent"]
    return L, Au, Bv, leaf_u, leaf_v


# ---------- 查询（严格表查） ----------
def query_distance_all_table(
    cluster_tree, node_to_leaf, pll_obj,
    AB_tables, L2B, L2C, leaf_row_index,
    u, v
) -> float:
    if u == v:
        return 0.0
    leaf_u = node_to_leaf.get(u); leaf_v = node_to_leaf.get(v)
    if leaf_u is None or leaf_v is None:
        return float("inf")
    if leaf_u == leaf_v:
        pll = pll_obj.get(leaf_u)
        return float("inf") if pll is None else float(pll.query(u, v))

    L, Au, Bv, leaf_u_id, leaf_v_id = find_lca_and_children(cluster_tree, node_to_leaf, u, v)

    AB = AB_tables.get(L, {}).get((Au, Bv))
    if AB is None:
        return float("inf")

    # 左/右叶→列预表
    tblL = L2B.get(L, {}).get((Au, Bv), {}).get(leaf_u_id)
    tblR = L2C.get(L, {}).get((Au, Bv), {}).get(leaf_v_id)
    if (tblL is None) or (tblR is None):
        return float("inf")

    # 取行
    rL = leaf_row_index[leaf_u_id].get(u, None)
    rR = leaf_row_index[leaf_v_id].get(v, None)
    if (rL is None) or (rR is None):
        return float("inf")

    du_row = tblL["mat"][rL, :].astype(np.float32, copy=False)   # len = nb
    dv_row = tblR["mat"][rR, :].astype(np.float32, copy=False)   # len = nc

    rows = AB["rows"]; cols = AB["cols"]; mids = AB["mids"]      # len = nnz
    # 纯向量化合并
    vals = du_row[rows] + mids + dv_row[cols]                    # [nnz]
    best = float(np.min(vals)) if vals.size > 0 else float("inf")
    return best


# ---------- 评测 ----------
def sample_entity_pairs(G, n_pairs=1000, in_lcc=True, rng_seed=42):
    entity_nodes = list(G.nodes())
    if len(entity_nodes) < 2:
        raise RuntimeError("图中节点过少，无法评测。")
    comps = list(nx.connected_components(G))
    if in_lcc and comps:
        lcc = max(comps, key=len)
        entity_nodes = [n for n in entity_nodes if n in lcc]
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
        raise RuntimeError("没有包含≥2节点的连通分量，无法评测。")
    cids, sizes = zip(*weighted_bins)
    total = sum(sizes); probs = [s/total for s in sizes]
    rng = random.Random(rng_seed)
    pairs, trials, max_trials = [], 0, n_pairs * 10
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
    gt = []
    for u, v in pairs:
        try:
            d = nx.shortest_path_length(G, u, v, weight="weight")
        except nx.NetworkXNoPath:
            d = float("inf")
        gt.append((u, v, d))
    return gt

def save_eval_set_csv(path, gt):
    df = pd.DataFrame(gt, columns=["source", "target", "dist"])
    df.to_csv(path, index=False)
    print(f"[INFO] Saved eval set with ground-truth to: {path}  (rows={len(df)})")

def load_eval_set_csv(path, G, max_rows=None):
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


def evaluate_all_table_lookup(
    cluster_tree, node_to_leaf, pll_obj,
    AB_tables, L2B, L2C, leaf_row_index,
    gt, preprocessing_time
):
    correct = total_eval = 0
    err = 0.0
    inf_pred = 0
    t0 = time.perf_counter()
    for u, v, d in gt:
        pred = query_distance_all_table(
            cluster_tree, node_to_leaf, pll_obj,
            AB_tables, L2B, L2C, leaf_row_index,
            u, v
        )
        if pred == float("inf"):
            inf_pred += 1
        if pred == d:
            correct += 1
        if (pred != float("inf")) and (d != float("inf")):
            err += abs(pred - d); total_eval += 1
    tQ = time.perf_counter() - t0

    rows = [[
        "HierL-E (all-table, vectorized)",
        tQ, len(gt), correct, (err/total_eval if total_eval>0 else float("inf")),
        inf_pred, preprocessing_time
    ]]
    return pd.DataFrame(rows, columns=[
        "method", "query_time_sec", "samples", "exact_matches", "mae", "inf_pred", "preprocessing_time"
    ])


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="Pubmed", choices=["Cora", "CiteSeer", "Pubmed"])
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--kg_file", type=str, default=None)

    ap.add_argument("--resolution", type=float, default=0.3)
    ap.add_argument("--max_cluster_size", type=int, default=1200)
    ap.add_argument("--hl_seed", type=int, default=42)
    ap.add_argument("--max_workers", type=int, default=None)

    # 压缩参数
    ap.add_argument("--leaf_rg_k", type=int, default=8, help="叶子 rep_graph kNN 闭包每点保留的近邻数")
    ap.add_argument("--ig_gateway_only", action="store_true", help="父层 index_graph 仅保留网关节点")
    ap.add_argument("--ig_intra_knn", type=int, default=8, help="父层 index_graph 各子簇网关间 kNN 边数")
    ap.add_argument("--mid_limit", type=int, default=0)

    # Leaf→Pair-Columns 预表 dtype
    ap.add_argument("--l2c_dtype", type=str, default="fp16", choices=["fp16", "fp32"],
                    help="Leaf→Pair-Columns 预表数据精度（fp16 更省内存，fp32 更准）")

    # 评测
    ap.add_argument("--eval_pairs", type=int, default=500)
    ap.add_argument("--limit_to_lcc", action="store_true")
    ap.add_argument("--save_eval_set", type=str, default=None)
    ap.add_argument("--load_eval_set", type=str, default=None)
    args = ap.parse_args()
    if not args.limit_to_lcc:
        args.limit_to_lcc = True
    if not args.ig_gateway_only:
        args.ig_gateway_only = True

    # 0) 加载图
    if args.kg_file:
        G = load_wn18_graph_aggregated(args.kg_file)
        src_name = f"WN18({os.path.basename(args.kg_file)})-aggregated"
    else:
        G = load_planetoid_graph(args.dataset, root=args.data_root)
        src_name = args.dataset
    print(f"[INFO] Graph: {src_name}, |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

    # 1) Skeleton
    tALL0 = time.time()
    cluster_tree, node_to_leaf, cc_to_leaves, SUPER_ROOT = build_skeleton_leaves(
        G, hl_resolution=args.resolution, hl_max_cluster_size=args.max_cluster_size, random_seed=args.hl_seed,
    )
    comps = [cid for cid, meta in cluster_tree.items() if meta["level"] == -1]
    leaves = [cid for cid, meta in cluster_tree.items() if meta["level"] == 0]
    print("\n[Step1] Skeleton summary")
    print(f"  SUPER_ROOT children (CC count): {len(comps)}")
    cc_sizes = []
    for cc_cid in comps:
        n_nodes = len(cluster_tree[cc_cid]["nodes"])
        n_leaves = len(cluster_tree[cc_cid]["children"])
        cc_sizes.append((cc_cid, n_nodes, n_leaves))
    cc_sizes.sort(key=lambda x: -x[1])
    for cc_cid, n_nodes, n_leaves in cc_sizes[:10]:
        print(f"    {cc_cid}: |CC_nodes|={n_nodes}, leaf_clusters={n_leaves}")
    print(f"  Total leaf clusters: {len(leaves)}")

    # 2) 叶簇（并行 PLL + kNN rep_graph）
    borders, pll_labels, G_nb_all, rep_graphs, _ = build_leaves_tables(
        G, cluster_tree, node_to_leaf, leaf_rg_k=args.leaf_rg_k, max_workers=args.max_workers
    )

    # 3) 父层 index_graph（网关收缩 + 子簇内网关 kNN 闭包）
    index_graphs_lvl1, idx_stats = build_index_graphs_for_parents(
        G, cluster_tree, borders, rep_graphs,
        parents_subset=None,
        ig_gateway_only=args.ig_gateway_only,
        ig_intra_knn=args.ig_intra_knn,
    )

    # 4) list_indexpath
    list_indexpath_lvl1 = build_list_indexpath_for_parents(
        cluster_tree, index_graphs_lvl1, rep_graphs, parents_subset=None
    )

    # 7) AB（float32）
    AB_tables_lvl1 = build_AB_tables(cluster_tree, rep_graphs, list_indexpath_lvl1, mid_limit=args.mid_limit)

    # SAME（per-leaf CSC）
    print("[INFO] Building SAME tables (per-leaf CSC) ...")
    SAME_tables = build_SAME_tables(cluster_tree, borders, rep_graphs, G_nb_all)
    print("[INFO] SAME tables ready: parents_with_SAME =", len(SAME_tables))

    # 7.9) Leaf→Pair-Columns 预表
    print("[INFO] Building Leaf→Pair-Columns tables (strict lookup) ...")
    tL2a = time.time()
    L2B, L2C, leaf_row_index = build_leaf2pair_tables(
        cluster_tree, borders, G_nb_all, SAME_tables, AB_tables_lvl1,
        dtype=args.l2c_dtype
    )
    tL2b = time.time()
    print(f"[INFO] Leaf→Pair-Columns built in {tL2b - tL2a:.3f}s; "
          f"parents_with_pairs={len(L2B)}")

    tALL1 = time.time()
    preprocessing_time = tALL1 - tALL0

    # 查询辅助
    _, pll_obj = build_aux_indices_for_query(borders, pll_labels)

    # 评测集
    if args.load_eval_set and os.path.isfile(args.load_eval_set):
        gt = load_eval_set_csv(args.load_eval_set, G, max_rows=args.eval_pairs)
        if not gt:
            raise RuntimeError(f"Loaded eval set has 0 usable rows: {args.load_eval_set}")
    else:
        pairs = sample_entity_pairs(G, n_pairs=args.eval_pairs, in_lcc=args.limit_to_lcc, rng_seed=42)
        print(f"[INFO] Sampling done. computing ground-truth distances for {len(pairs)} pairs ...")
        gt = compute_gt_distances(G, pairs)
        if args.save_eval_set:
            save_eval_set_csv(args.save_eval_set, gt)

    # 评测（纯查表）
    df_eval = evaluate_all_table_lookup(
        cluster_tree, node_to_leaf, pll_obj,
        AB_tables_lvl1, L2B, L2C, leaf_row_index,
        gt, preprocessing_time
    )
    print("\n=== Table-only LCA Query Evaluation (All-Table, strict lookup) ===")
    print(df_eval.to_string(index=False))
    print(f"\n[Summary] Preprocessing={preprocessing_time:.3f}s, "
          f"parents_with_AB={len(AB_tables_lvl1)}, parents_with_L2={len(L2B)}")
    if idx_stats:
        total_nodes = sum(s["nodes"] for s in idx_stats.values())
        total_edges = sum(s["edges"] for s in idx_stats.values())
        total_gates = sum(s["gateway"] for s in idx_stats.values())
        print(f"[IndexGraph Stats] nodes={total_nodes}, edges={total_edges}, gateways={total_gates}")

    print("\n[OK] Finished: strict table lookup (no Dijkstra nor per-query sinking).")

if __name__ == "__main__":
    main()
