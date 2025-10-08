#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hierl_e_same_opt_vec.py — Hierarchical-E with SAME & AB (全向量化抬升与合并)
- 叶簇并行 PLL
- 父层 index_graph
- AB 表（跨子簇 b->c，中段距离），预存 nnz_row_index
- SAME 表（同子簇“叶边界 b0 -> 子簇代表点 b”）按叶聚簇行，并为每个叶预打包 nnz 片段
- 查询：SAME 抬升 -> AB 合并 全向量化；中文分项计时
"""

from __future__ import annotations
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
    entities = set()
    for (h, t), rmap in pair2rels.items():
        entities.add(h); entities.add(t)
    G.add_nodes_from(entities)

    for (h, t), rmap in pair2rels.items():
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
        import heapq
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


# ---------- hierarchical leiden: extract final clusters ----------
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


# ---------- Step 1: skeleton ----------
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


# ---------- Step 2: per-leaf PLL + borders + G_nb + rep_graph ----------
def _build_one_leaf(args):
    leaf_cid, leaf_nodes, neigh_map, G = args
    nodes = sorted(leaf_nodes, key=node_sort_key)
    node_set = set(nodes)

    border = [n for n in nodes if any((nbr not in node_set) for nbr in neigh_map[n])]
    border = sorted(border, key=node_sort_key)
    b_idx = {b: i for i, b in enumerate(border)}

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

    # PLL（每个 worker 内部串行）
    t0 = time.time()
    pll = PrunedPLLIndex(subg, order)
    for root in pll.order:
        import heapq
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
    pll_build_time = time.time() - t0
    label_entries = sum(len(m) for m in pll.labels.values())

    # G_nb：簇内任一点到全部 border 的距离
    G_nb = {}
    L = len(border)
    for n in nodes:
        arr = np.empty((L,), dtype=np.float64)
        for i, b in enumerate(border):
            arr[i] = pll.query(n, b)
        G_nb[n] = arr

    # 叶 rep_graph（border 闭包）
    rep_edges = []
    for i in range(L):
        bi = border[i]
        for j in range(i + 1, L):
            bj = border[j]
            d_ij = float(G_nb[bi][b_idx[bj]])
            if d_ij < float("inf"):
                rep_edges.append((bi, bj, d_ij))

    return leaf_cid, border, pll.labels, G_nb, rep_edges, pll_build_time, label_entries


def build_leaves_tables(G: nx.Graph, cluster_tree: dict, node_to_leaf: dict, max_workers=None):
    max_workers = max_workers or (os.cpu_count() or 2)
    neigh_map = {n: list(G.neighbors(n)) for n in G.nodes()}

    tasks = []
    for cid, meta in cluster_tree.items():
        if meta["level"] == 0:  # leaf
            tasks.append((cid, meta["nodes"], neigh_map, G))

    borders = {}
    pll_labels = {}
    G_nb_all = {}
    rep_graphs = {}
    stats = {}

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_build_one_leaf, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Step2: building leaves", unit="leaf"):
            (leaf_cid, border, labels, G_nb, rep_edges, pll_time, label_entries) = fut.result()
            borders[leaf_cid] = border
            pll_labels[leaf_cid] = labels
            G_nb_all[leaf_cid] = G_nb

            RG = nx.Graph()
            RG.add_nodes_from(border)
            for u, v, w in rep_edges:
                if RG.has_edge(u, v):
                    if w < RG[u][v].get("weight", float("inf")):
                        RG[u][v]["weight"] = float(w)
                else:
                    RG.add_edge(u, v, weight=float(w))
            rep_graphs[leaf_cid] = RG

            stats[leaf_cid] = dict(
                nodes=len(cluster_tree[leaf_cid]["nodes"]),
                borders=len(border),
                pll_time=pll_time,
                pll_label_entries=label_entries,
                rep_edges=RG.number_of_edges(),
            )

    return borders, pll_labels, G_nb_all, rep_graphs, stats


# ---------- Step 3: parent index_graph ----------
def build_index_graphs_for_parents(
    G: nx.Graph,
    cluster_tree: dict,
    borders: dict,
    rep_graphs: dict,
    parents_subset: list | None = None,
):
    index_graphs = {}
    stats = {}

    if parents_subset is None:
        parents = [cid for cid, meta in cluster_tree.items() if meta["children"] and len(meta["children"]) > 1]
    else:
        parents = [p for p in parents_subset if cluster_tree.get(p, {}).get("children")]

    for parent in tqdm(parents, desc="Step3: building index_graphs (parents)", unit="parent"):
        parent_meta = cluster_tree[parent]
        children = parent_meta["children"]
        parent_nodes_set = parent_meta.get("nodes") or set()

        child_to_nodes = defaultdict(set)
        node_to_child = {}
        for ch in children:
            if cluster_tree[ch]["level"] == 0:
                nodes_set = set(borders.get(ch, []))
            else:
                RG = rep_graphs.get(ch)
                nodes_set = set(RG.nodes()) if RG is not None else set()
            if not nodes_set:
                continue
            child_to_nodes[ch] = nodes_set
            for x in nodes_set:
                node_to_child[x] = ch

        index_nodes = set().union(*child_to_nodes.values()) if child_to_nodes else set()
        IG = nx.Graph()
        if index_nodes:
            IG.add_nodes_from(index_nodes)

        # 子簇内部闭包边
        for ch, nodes_set in child_to_nodes.items():
            RG = rep_graphs.get(ch)
            if RG is None:
                continue
            for u, v, data in RG.edges(data=True):
                if (u not in index_nodes) or (v not in index_nodes):
                    continue
                w = float(data.get("weight", 1.0))
                if IG.has_edge(u, v):
                    if w < IG[u][v].get("weight", float("inf")):
                        IG[u][v]["weight"] = w
                else:
                    IG.add_edge(u, v, weight=w)

        # 跨子簇原图边（代表点之间）
        for b in index_nodes:
            if parent_nodes_set and b not in parent_nodes_set:
                continue
            ch_b = node_to_child.get(b)
            for v, data in G[b].items():
                if v not in index_nodes:
                    continue
                if parent_nodes_set and v not in parent_nodes_set:
                    continue
                ch_v = node_to_child.get(v)
                if (ch_b is None) or (ch_v is None) or (ch_b == ch_v):
                    continue
                w = float(data.get("weight", 1.0))
                if IG.has_edge(b, v):
                    if w < IG[b][v].get("weight", float("inf")):
                        IG[b][v]["weight"] = w
                else:
                    IG.add_edge(b, v, weight=w)

        index_graphs[parent] = IG
        stats[parent] = dict(nodes=IG.number_of_nodes(), edges=IG.number_of_edges(), children=len(children))

    return index_graphs, stats


# ---------- Step 4: list_indexpath（子簇为单位的多源 Dijkstra） + SAME 原始桶 ----------
def _make_adj_from_graph(G: nx.Graph):
    adj = {}
    for u in G.nodes():
        lst = []
        for v, data in G[u].items():
            lst.append((v, float(data.get("weight", 1.0))))
        adj[u] = lst
    return adj


def _multi_source_dijkstra_index_worker(args):
    """
    返回两路：
    - res_cross: b0 -> {c (B!=A): dist}    ->  AB 表
    - res_same : b0 -> {b (B==A): dist}    ->  SAME 表（要求 b0 为某个叶簇边界）
    """
    parent, A, adj, child_to_nodes, node_to_child, border_owner_leaf = args

    seeds = list(child_to_nodes.get(A, []))
    if not seeds:
        return parent, A, {}, {}

    dist = {}
    origin = {}
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

    res_cross = defaultdict(dict)
    res_same  = defaultdict(dict)

    for x, dx in dist.items():
        B = node_to_child.get(x)
        if B is None:
            continue
        b0 = origin.get(x)
        if b0 is None:
            continue

        if B != A:
            res_cross[b0][x] = float(dx)
        else:
            if border_owner_leaf.get(b0) is None:
                continue
            res_same[b0][x] = float(dx)

    return parent, A, res_cross, res_same


def build_list_indexpath_for_parents(
    cluster_tree: dict,
    index_graphs: dict,
    borders: dict,
    rep_graphs: dict,
    border_owner_leaf: dict,
    parents_subset: list | None = None,
):
    L_cross = defaultdict(lambda: defaultdict(dict))
    L_same  = defaultdict(lambda: defaultdict(dict))
    parent_stats = {}

    if parents_subset is None:
        parents = [cid for cid, meta in cluster_tree.items() if meta["children"] and len(meta["children"]) > 1]
    else:
        parents = [p for p in parents_subset if cluster_tree.get(p, {}).get("children")]

    tasks = []
    for parent in parents:
        IG = index_graphs.get(parent)
        if IG is None or IG.number_of_nodes() == 0:
            parent_stats[parent] = dict(children=len(cluster_tree[parent]["children"]), nodes=0, edges=0, jobs=0)
            continue

        node_to_child = {}
        child_to_nodes = defaultdict(set)
        for ch in cluster_tree[parent]["children"]:
            if cluster_tree[ch]["level"] == 0:
                nodes_set = set(borders.get(ch, []))
            else:
                RG = rep_graphs.get(ch)
                nodes_set = set(RG.nodes()) if RG is not None else set()
            if not nodes_set:
                continue
            for x in nodes_set:
                node_to_child[x] = ch
            child_to_nodes[ch] = nodes_set

        adj = _make_adj_from_graph(IG)

        jobs = 0
        for A, seeds in child_to_nodes.items():
            if not seeds:
                continue
            tasks.append((parent, A, adj, child_to_nodes, node_to_child, border_owner_leaf))
            jobs += 1

        parent_stats[parent] = dict(children=len(cluster_tree[parent]["children"]),
                                    nodes=IG.number_of_nodes(),
                                    edges=IG.number_of_edges(),
                                    jobs=jobs)

    with ProcessPoolExecutor(max_workers=os.cpu_count() or 2) as ex:
        futs = [ex.submit(_multi_source_dijkstra_index_worker, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Step4: list_indexpath (multi-source)", unit="job"):
            parent, A, res_cross, res_same = fut.result()
            L_cross[parent][A] = res_cross
            L_same[parent][A]  = res_same

    return L_cross, L_same, parent_stats


# ---------- Step 7A: AB 表（预存 nnz_row_index） ----------
def build_AB_tables(cluster_tree, borders, rep_graphs, list_indexpath_cross, mid_limit=0):
    AB_tables = defaultdict(dict)

    for L, mapA in list_indexpath_cross.items():
        pair_bucket = defaultdict(lambda: defaultdict(list))  # (A,B) -> b -> [(mid,c)]

        node_to_child = {}
        for ch in cluster_tree[L]["children"]:
            if cluster_tree[ch]["level"] == 0:
                nodes_set = borders.get(ch, [])
            else:
                RG = rep_graphs.get(ch)
                nodes_set = RG.nodes() if RG is not None else []
            for x in nodes_set:
                node_to_child[x] = ch

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
                if mid_limit and mid_limit > 0 and len(lst) > mid_limit:
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
            row_id_for_nnz = []
            for b in b_nodes:
                lst = by_b_kept[b]
                for dist_bc, c in lst:
                    rows.append(b_index[b])
                    cols.append(c_index[c])
                    mids.append(dist_bc)
                    row_id_for_nnz.append(b_index[b])
                row_ptr.append(len(rows))

            table = dict(
                b_nodes=b_nodes, c_nodes=c_nodes,
                b_index=b_index, c_index=c_index,
                rows=np.asarray(rows, dtype=np.int32),
                cols=np.asarray(cols, dtype=np.int32),
                mids=np.asarray(mids, dtype=np.float64),
                row_ptr=np.asarray(row_ptr, dtype=np.int32),
                nnz_row_index=np.asarray(row_id_for_nnz, dtype=np.int32),
            )
            AB_tables[L][(A, B)] = table

    return AB_tables


# ---------- Step 7B: SAME 表（按叶聚簇 + 预打包 nnz 片段） ----------
def build_SAME_tables(list_indexpath_same, border_owner_leaf, borders):
    """
    SAME[L][A] 结构：
      - b0_nodes, b_nodes, b0_index, b_index, rows, cols, vals, row_ptr（CSR）
      - leaf_rows[leaf] = np[int]         仅该叶的行
      - leaf_b0_pos[leaf] = np[int]       与 leaf_rows 对齐: b0 在 leaf 的 G_nb 向量位置
      - leaf_pack[leaf] = {
          'nnz_cols': np[int],            把 leaf_rows 的所有片段串接
          'nnz_vals': np[float],
          'nnz_row_index': np[int],       对应 leaf_rows 的行索引 (0..len(leaf_rows)-1)
        }
    """
    SAME = defaultdict(dict)

    for L, mapA in list_indexpath_same.items():
        for A, by_b0 in mapA.items():
            if not by_b0:
                continue

            b0_nodes = sorted(by_b0.keys(), key=node_sort_key)
            b_nodes_set = set()
            for b0, cmap in by_b0.items():
                b_nodes_set |= set(cmap.keys())
            b_nodes = sorted(b_nodes_set, key=node_sort_key)
            if not b0_nodes or not b_nodes:
                continue

            b0_index = {x: i for i, x in enumerate(b0_nodes)}
            b_index  = {x: j for j, x in enumerate(b_nodes)}

            rows, cols, vals, row_ptr = [], [], [], [0]
            row_owner = []
            for b0 in b0_nodes:
                cmap = by_b0.get(b0, {})
                for b, d in cmap.items():
                    rows.append(b0_index[b0])
                    cols.append(b_index[b])
                    vals.append(float(d))
                row_ptr.append(len(rows))
                row_owner.append(border_owner_leaf.get(b0))

            rows = np.asarray(rows, dtype=np.int32)
            cols = np.asarray(cols, dtype=np.int32)
            vals = np.asarray(vals, dtype=np.float64)
            row_ptr = np.asarray(row_ptr, dtype=np.int32)

            # leaf_rows & leaf_b0_pos
            leaf_rows = defaultdict(list)
            for ridx, leaf_id in enumerate(row_owner):
                if leaf_id is not None:
                    leaf_rows[leaf_id].append(ridx)
            for leaf_id in list(leaf_rows.keys()):
                leaf_rows[leaf_id] = np.asarray(leaf_rows[leaf_id], dtype=np.int32)

            leaf_b0_pos = {}
            for leaf_id, r_idx_arr in leaf_rows.items():
                blist = borders.get(leaf_id, [])
                pos_map = {b: i for i, b in enumerate(blist)}
                pos_arr = np.full((len(r_idx_arr),), -1, dtype=np.int32)
                for k in range(len(r_idx_arr)):
                    r = r_idx_arr[k]
                    b0 = b0_nodes[r]
                    pos = pos_map.get(b0, -1)
                    pos_arr[k] = pos
                leaf_b0_pos[leaf_id] = pos_arr

            # 为每个叶打包 nnz
            leaf_pack = {}
            for leaf_id, r_idx_arr in leaf_rows.items():
                if r_idx_arr.size == 0:
                    continue
                # 统计 nnz 总数
                seg_lens = row_ptr[r_idx_arr + 1] - row_ptr[r_idx_arr]
                total_nnz = int(seg_lens.sum())
                if total_nnz == 0:
                    continue
                nnz_cols = np.empty((total_nnz,), dtype=np.int32)
                nnz_vals = np.empty((total_nnz,), dtype=np.float64)
                nnz_row_index = np.empty((total_nnz,), dtype=np.int32)

                p = 0
                for k, r in enumerate(r_idx_arr):
                    s, e = row_ptr[r], row_ptr[r+1]
                    l = e - s
                    if l <= 0:
                        continue
                    nnz_cols[p:p+l] = cols[s:e]
                    nnz_vals[p:p+l] = vals[s:e]
                    nnz_row_index[p:p+l] = k  # 指向 r 在 r_idx_arr 中的索引
                    p += l

                leaf_pack[leaf_id] = dict(
                    nnz_cols=nnz_cols[:p],
                    nnz_vals=nnz_vals[:p],
                    nnz_row_index=nnz_row_index[:p],
                )

            SAME[L][A] = dict(
                b0_nodes=b0_nodes, b_nodes=b_nodes,
                b0_index=b0_index, b_index=b_index,
                rows=rows, cols=cols, vals=vals, row_ptr=row_ptr,
                leaf_rows=leaf_rows, leaf_b0_pos=leaf_b0_pos,
                leaf_pack=leaf_pack,
            )
    return SAME


# ---------- Step 5: parent rep_graph（三段式闭包） ----------
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


def _child_dist_map_to_targets(cluster_tree, child, src, targets_list, borders, G_nb_all, rep_graphs, _leaf_border_index_cache):
    if cluster_tree[child]["level"] == 0:
        blist = borders.get(child, [])
        if not blist:
            return {}
        idx_map = _leaf_border_index_cache.get(child)
        if idx_map is None:
            idx_map = {b: i for i, b in enumerate(blist)}
            _leaf_border_index_cache[child] = idx_map
        arr = G_nb_all.get(child, {}).get(src)
        if arr is None or arr.size == 0:
            return {}
        out = {}
        for t in targets_list:
            i = idx_map.get(t)
            if i is None:
                continue
            dv = float(arr[i])
            if np.isfinite(dv):
                out[t] = dv
        return out
    else:
        RG = rep_graphs.get(child)
        if RG is None:
            return {}
        return _dijkstra_to_targets(RG, src, set(targets_list))


# ---------- 向量化 AB 合并 ----------
def table_min_plus_best_vec(table, left_vec_B: np.ndarray, right_vec_C: np.ndarray) -> float:
    """
    left_vec_B: shape (len(b_nodes),), 右侧同理；值为 inf 表示不可达
    返回 min_{(row,col)∈E} left[row] + mids + right[col]
    """
    rows_idx = table["nnz_row_index"]     # (nnz,)
    cols = table["cols"]                  # (nnz,)
    mids = table["mids"]                  # (nnz,)

    lv = left_vec_B[rows_idx]
    rv = right_vec_C[cols]
    vals = lv + mids + rv
    # 过滤 inf
    mask = np.isfinite(vals)
    if not np.any(mask):
        return float("inf")
    return float(np.min(vals[mask]))


# ---------- Step 5（父层 rep_graph 闭包，保留原实现，不影响查询路径） ----------
def build_parent_rep_graphs_with_tripartite_closure(
    G: nx.Graph,
    cluster_tree: dict,
    index_graphs: dict,
    AB_tables: dict,
    rep_graphs: dict,
    borders: dict,
    G_nb_all: dict,
    parents_subset: list | None = None,
):
    stats = {}
    _leaf_border_index_cache = {}
    left_vec_cache = {}
    right_vec_cache = {}

    if parents_subset is None:
        parents = [cid for cid, meta in cluster_tree.items() if meta["children"] and len(meta["children"]) > 1]
    else:
        parents = [p for p in parents_subset if cluster_tree.get(p, {}).get("children")]

    for parent in tqdm(parents, desc="Step5: building parent rep_graphs (closure 3-part)", unit="parent"):
        IG = index_graphs.get(parent)
        if IG is None or IG.number_of_nodes() == 0:
            rep_graphs[parent] = nx.Graph()
            stats[parent] = dict(rep_nodes=0, rep_edges=0)
            continue

        parent_nodes_set = cluster_tree[parent].get("nodes") or set()
        node_to_child = {}
        child_to_nodes_all = defaultdict(set)
        for ch in cluster_tree[parent]["children"]:
            if cluster_tree[ch]["level"] == 0:
                nodes_set = set(borders.get(ch, []))
            else:
                RGch = rep_graphs.get(ch)
                nodes_set = set(RGch.nodes()) if RGch is not None else set()
            for x in nodes_set:
                node_to_child[x] = ch
            child_to_nodes_all[ch] = nodes_set

        rep_nodes = []
        rep_set = set()
        for x in IG.nodes():
            has_external = False
            for nbr in G[x]:
                if parent_nodes_set and nbr not in parent_nodes_set:
                    has_external = True
                    break
            if has_external:
                rep_nodes.append(x)
                rep_set.add(x)

        RGp = nx.Graph()
        RGp.add_nodes_from(rep_nodes)

        for ch in cluster_tree[parent]["children"]:
            RGch = rep_graphs.get(ch)
            if RGch is None:
                continue
            for u, v, data in RGch.edges(data=True):
                if (u in rep_set) and (v in rep_set):
                    w = float(data.get("weight", 1.0))
                    if RGp.has_edge(u, v):
                        if w < RGp[u][v].get("weight", float("inf")):
                            RGp[u][v]["weight"] = w
                    else:
                        RGp.add_edge(u, v, weight=w)

        ABp = AB_tables.get(parent, {})
        if ABp:
            child_to_rep_nodes = defaultdict(list)
            for x in rep_nodes:
                ch = node_to_child.get(x)
                if ch is not None:
                    child_to_rep_nodes[ch].append(x)

            for (A, B), table in ABp.items():
                repA = child_to_rep_nodes.get(A, [])
                repB = child_to_rep_nodes.get(B, [])
                if not repA or not repB:
                    continue

                b_nodes = table["b_nodes"]
                c_nodes = table["c_nodes"]

                for u in repA:
                    keyL = (A, u, id(table))
                    left_map = left_vec_cache.get(keyL)
                    if left_map is None:
                        left_map = _child_dist_map_to_targets(cluster_tree, A, u, b_nodes,
                                                              borders, G_nb_all, rep_graphs, _leaf_border_index_cache)
                        left_vec_cache[keyL] = left_map
                    if not left_map:
                        continue

                    for v in repB:
                        keyR = (B, v, id(table))
                        right_map = right_vec_cache.get(keyR)
                        if right_map is None:
                            right_map = _child_dist_map_to_targets(cluster_tree, B, v, c_nodes,
                                                                   borders, G_nb_all, rep_graphs, _leaf_border_index_cache)
                            right_vec_cache[keyR] = right_map
                        if not right_map:
                            continue

                        # 这里仍旧使用字典 + 稀疏合并（闭包只做一次，成本可接受）
                        # 若需要，也可改成向量化（同查询的 table_min_plus_best_vec）
                        # 但构建阶段只跑一遍，收益有限。
                        # 简易版：
                        C = len(c_nodes)
                        dv_full = np.full((C,), np.inf, dtype=np.float64)
                        for c, val in right_map.items():
                            j = table["c_index"].get(c)
                            if j is not None:
                                dv_full[j] = float(val)
                        best = float("inf")
                        for b, du in left_map.items():
                            r = table["b_index"].get(b)
                            if r is None:
                                continue
                            s = table["row_ptr"][r]; e = table["row_ptr"][r + 1]
                            cols_r = table["cols"][s:e]
                            mids = table["mids"][s:e]
                            dv_r = dv_full[cols_r]
                            mask = np.isfinite(dv_r)
                            if not mask.any():
                                continue
                            m = float(np.min(du + mids[mask] + dv_r[mask]))
                            if m < best:
                                best = m
                        if not np.isfinite(best):
                            continue
                        if RGp.has_edge(u, v):
                            if best < RGp[u][v].get("weight", float("inf")):
                                RGp[u][v]["weight"] = best
                        else:
                            RGp.add_edge(u, v, weight=best)

        rep_graphs[parent] = RGp
        stats[parent] = dict(rep_nodes=RGp.number_of_nodes(), rep_edges=RGp.number_of_edges())

    return rep_graphs, stats


# ---------- Step 7C: 查询辅助 ----------
def build_aux_indices_for_query(borders, pll_labels, rep_graphs):
    leaf_border_index = {}
    for leaf, blist in borders.items():
        leaf_border_index[leaf] = {b: i for i, b in enumerate(blist)}
    pll_obj = {leaf: PrunedPLLIndex.from_labels(labels) for leaf, labels in pll_labels.items()}
    rep_nodes_cache = {cid: set(rg.nodes()) if isinstance(rg, nx.Graph) else set()
                       for cid, rg in rep_graphs.items()}
    return leaf_border_index, pll_obj, rep_nodes_cache


def get_path_to_root(cluster_tree, cid):
    path = []
    cur = cid
    while cur is not None:
        path.append(cur)
        cur = cluster_tree[cur]["parent"] if cur in cluster_tree else None
    return path


def find_lca_and_children(cluster_tree, node_to_leaf, u, v):
    leaf_u = node_to_leaf[u]
    leaf_v = node_to_leaf[v]
    path_u = get_path_to_root(cluster_tree, leaf_u)
    path_v = get_path_to_root(cluster_tree, leaf_v)
    set_u = set(path_u)
    L = next((x for x in path_v if x in set_u), None)
    if L is None:
        raise RuntimeError("LCA not found")
    Au = leaf_u
    while cluster_tree[Au]["parent"] != L:
        Au = cluster_tree[Au]["parent"]
    Bv = leaf_v
    while cluster_tree[Bv]["parent"] != L:
        Bv = cluster_tree[Bv]["parent"]
    return L, Au, Bv, leaf_u, leaf_v


# ---------- SAME 抬升（全向量化） ----------
def _lift_same_vec(side_vec_arr: np.ndarray,
                   same_entry: dict,
                   leaf_id,
                   need_nodes: list) -> np.ndarray:
    """
    输入:
      - side_vec_arr: G_nb[leaf][node] 向量（长度 = 该叶 border 数）
      - same_entry: SAME[L][A] 或 SAME[L][B]
      - leaf_id: 当前叶
      - need_nodes: 需要输出的节点列表（AB 的 b_nodes 或 c_nodes）
    输出:
      - output_vec: shape (len(need_nodes),) 的数组（按 need_nodes 顺序），inf 表示不可达
    """
    # 叶下需要的行及其打包 nnz
    r_idx_arr = same_entry["leaf_rows"].get(leaf_id)
    pos_arr = same_entry["leaf_b0_pos"].get(leaf_id)
    pack = same_entry["leaf_pack"].get(leaf_id)
    if r_idx_arr is None or r_idx_arr.size == 0 or pos_arr is None or pack is None:
        return np.full((len(need_nodes),), np.inf, dtype=np.float64)

    # u/v 到这些 b0 的子向量
    valid = pos_arr >= 0
    if not np.any(valid):
        return np.full((len(need_nodes),), np.inf, dtype=np.float64)

    du_b0_sub = side_vec_arr[pos_arr[valid]]  # (R,)

    # pack 中的 nnz_row_index 是相对于 r_idx_arr 的索引；需要筛掉无效行
    # 建立一个掩码，把 nnz 的 row 映射到“有效行”的位置；无效行直接丢弃
    # 为简单起见，这里重建一个映射表：
    #  r_idx_arr_valid = r_idx_arr[valid]
    #  map from old local row id -> new compact id (或 -1 表示无效)
    #  由于 pack 是构建时针对所有 r_idx_arr 打包的，我们再打包一次更复杂。
    # 更高效做法：在 build_SAME_tables 时就为每个 leaf 只保留有效行（pos>=0）。
    # 这里实现一个快速过滤：先找哪些“有效行编号”。
    valid_rows_positions = np.nonzero(valid)[0]              # 在 r_idx_arr 中的下标
    # 构造 old->new 的映射数组
    max_row_local = (same_entry["leaf_pack"][leaf_id]["nnz_row_index"].max()
                     if pack["nnz_row_index"].size > 0 else -1)
    row_map = -np.ones((max_row_local + 1,), dtype=np.int32)
    row_map[valid_rows_positions] = np.arange(valid_rows_positions.size, dtype=np.int32)

    nnz_row_old = pack["nnz_row_index"]
    keep_mask = nnz_row_old < row_map.size
    nnz_row_old = nnz_row_old[keep_mask]
    nnz_cols = pack["nnz_cols"][keep_mask]
    nnz_vals = pack["nnz_vals"][keep_mask]
    nnz_row_new = row_map[nnz_row_old]
    keep_mask2 = nnz_row_new >= 0
    if not np.any(keep_mask2):
        return np.full((len(need_nodes),), np.inf, dtype=np.float64)

    nnz_cols = nnz_cols[keep_mask2]
    nnz_vals = nnz_vals[keep_mask2]
    nnz_row_new = nnz_row_new[keep_mask2]

    # 计算 add = du_b0_sub[nnz_row_new] + nnz_vals
    add = du_b0_sub[nnz_row_new] + nnz_vals

    # scatter-min 到 SAME 的 b_nodes 向量
    nB = len(same_entry["b_nodes"])
    du_b_full = np.full((nB,), np.inf, dtype=np.float64)
    np.minimum.at(du_b_full, nnz_cols, add)

    # 抽取 need_nodes 顺序
    same_b_index = same_entry["b_index"]
    idx = np.array([same_b_index.get(x, -1) for x in need_nodes], dtype=np.int32)
    out = np.full((len(need_nodes),), np.inf, dtype=np.float64)
    ok = idx >= 0
    if np.any(ok):
        out[ok] = du_b_full[idx[ok]]
    return out


def lift_with_SAME_left_vec(u, leaf_u, A, L, borders, G_nb_all, AB_table, SAME_tables):
    same = SAME_tables.get(L, {}).get(A)
    if same is None:
        return None  # 让调用方回退
    arr_u = G_nb_all[leaf_u].get(u)
    if arr_u is None or arr_u.size == 0:
        return np.full((len(AB_table["b_nodes"]),), np.inf, dtype=np.float64)
    return _lift_same_vec(arr_u, same, leaf_u, AB_table["b_nodes"])


def lift_with_SAME_right_vec(v, leaf_v, B, L, borders, G_nb_all, AB_table, SAME_tables):
    same = SAME_tables.get(L, {}).get(B)
    if same is None:
        return None
    arr_v = G_nb_all[leaf_v].get(v)
    if arr_v is None or arr_v.size == 0:
        return np.full((len(AB_table["c_nodes"]),), np.inf, dtype=np.float64)
    return _lift_same_vec(arr_v, same, leaf_v, AB_table["c_nodes"])


# ===== 查询（SAME 向量化抬升 + AB 向量化合并；保留轻量回退） =====
def query_distance_table_only_vectorized(
    cluster_tree, borders, G_nb_all, pll_obj, rep_graphs,
    list_indexpath_cross, AB_tables, SAME_tables, node_to_leaf,
    u, v, q_topk_u=0, q_topk_v=0, q_expand_steps=0,
    timers=None, counters=None
) -> float:

    if u == v:
        return 0.0

    t_lca0 = time.perf_counter()
    leaf_u = node_to_leaf.get(u)
    leaf_v = node_to_leaf.get(v)
    if leaf_u is None or leaf_v is None:
        return float("inf")
    if leaf_u == leaf_v:
        pll = pll_obj.get(leaf_u)
        ans = float("inf") if pll is None else float(pll.query(u, v))
        if counters is not None:
            counters["same_leaf_queries"] += 1
        return ans

    L, Au, Bv, leaf_u_id, leaf_v_id = find_lca_and_children(cluster_tree, node_to_leaf, u, v)
    t_lca1 = time.perf_counter()
    if timers is not None:
        timers["lca"] += (t_lca1 - t_lca0)

    t_tab0 = time.perf_counter()
    table = AB_tables.get(L, {}).get((Au, Bv), None)
    t_tab1 = time.perf_counter()
    if timers is not None:
        timers["ab_exist"] += (t_tab1 - t_tab0)

    if table is None:
        # 轻量回退：一次组合（字典），次数很少
        if counters is not None:
            counters["fallbacks"] += 1
        resAu = list_indexpath_cross.get(L, {}).get(Au, {})
        du_b0 = G_nb_all[leaf_u_id].get(u)
        left_map = {}
        if du_b0 is not None:
            blist_u = borders.get(leaf_u_id, [])
            for b, cmap in resAu.items():
                best = float("inf")
                for i, b0 in enumerate(blist_u):
                    mid = cmap.get(b, float("inf"))
                    if not np.isfinite(mid):
                        continue
                    du = float(du_b0[i])
                    cand = du + mid
                    if cand < best:
                        best = cand
                if np.isfinite(best):
                    left_map[b] = best
        if not left_map:
            return float("inf")
        right_candidates = set()
        for b in left_map.keys():
            right_candidates |= set(resAu.get(b, {}).keys())
        if not right_candidates:
            return float("inf")
        resBv = list_indexpath_cross.get(L, {}).get(Bv, {})
        dv_c0 = G_nb_all[leaf_v_id].get(v)
        right_map = {}
        if dv_c0 is not None:
            blist_v = borders.get(leaf_v_id, [])
            for c in right_candidates:
                best = float("inf")
                for j, c0 in enumerate(blist_v):
                    mid = resBv.get(c0, {}).get(c, float("inf"))
                    if not np.isfinite(mid):
                        continue
                    dv = float(dv_c0[j])
                    cand = dv + mid
                    if cand < best:
                        best = cand
                if np.isfinite(best):
                    right_map[c] = best
        best = float("inf")
        for b, du in left_map.items():
            cmap = resAu.get(b, {})
            for c, mid in cmap.items():
                if not np.isfinite(mid):
                    continue
                dv = right_map.get(c)
                if dv is None:
                    continue
                cand = du + mid + dv
                if cand < best:
                    best = cand
        return float(best)

    if counters is not None:
        counters["use_ab"] += 1

    # SAME 抬升（向量化）
    t_liftL0 = time.perf_counter()
    left_vec = lift_with_SAME_left_vec(u, leaf_u_id, Au, L, borders, G_nb_all, table, SAME_tables)
    t_liftL1 = time.perf_counter()
    if timers is not None:
        timers["lift_left"] += (t_liftL1 - t_liftL0)

    t_liftR0 = time.perf_counter()
    right_vec = lift_with_SAME_right_vec(v, leaf_v_id, Bv, L, borders, G_nb_all, table, SAME_tables)
    t_liftR1 = time.perf_counter()
    if timers is not None:
        timers["lift_right"] += (t_liftR1 - t_liftR0)

    # 若 SAME 缺失：退化到“叶向量 + cross 字典”的一次性组合（仍无扩容）
    if left_vec is None:
        left_vec = np.full((len(table["b_nodes"]),), np.inf, dtype=np.float64)
        du_b0 = G_nb_all[leaf_u_id].get(u)
        if du_b0 is not None:
            blist_u = borders.get(leaf_u_id, [])
            resAu = list_indexpath_cross.get(L, {}).get(Au, {})
            # 把每个 b 的最佳值放到对应行向量里
            for b, cmap in resAu.items():
                r = table["b_index"].get(b)
                if r is None:
                    continue
                best = float("inf")
                for i, b0 in enumerate(blist_u):
                    mid = cmap.get(b, float("inf"))
                    if not np.isfinite(mid):
                        continue
                    du = float(du_b0[i])
                    cand = du + mid
                    if cand < best:
                        best = cand
                left_vec[r] = best

    if right_vec is None:
        right_vec = np.full((len(table["c_nodes"]),), np.inf, dtype=np.float64)
        dv_c0 = G_nb_all[leaf_v_id].get(v)
        if dv_c0 is not None:
            blist_v = borders.get(leaf_v_id, [])
            resBv = list_indexpath_cross.get(L, {}).get(Bv, {})
            for c in table["c_nodes"]:
                j = table["c_index"].get(c)
                if j is None:
                    continue
                best = float("inf")
                for t, c0 in enumerate(blist_v):
                    mid = resBv.get(c0, {}).get(c, float("inf"))
                    if not np.isfinite(mid):
                        continue
                    dv = float(dv_c0[t])
                    cand = dv + mid
                    if cand < best:
                        best = cand
                right_vec[j] = best

    # AB 表 min-plus 向量化
    t_merge0 = time.perf_counter()
    best = table_min_plus_best_vec(table, left_vec, right_vec)
    t_merge1 = time.perf_counter()
    if timers is not None:
        timers["merge"] += (t_merge1 - t_merge0)

    if timers is not None:
        timers["expand"] += 0.0
    return float(best if np.isfinite(best) else float("inf"))


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
    total = sum(sizes)
    probs = [s/total for s in sizes]
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


def evaluate_table_query(
    G,
    cluster_tree, borders, G_nb_all, pll_obj, rep_graphs,
    list_indexpath_cross, AB_tables, SAME_tables, node_to_leaf,
    gt, preprocessing_time,
    q_topk_u=0, q_topk_v=0, q_expand_steps=0
):
    timers = defaultdict(float)
    counters = defaultdict(int)

    correct = total_eval = 0
    err = 0.0
    inf_pred = 0

    t0 = time.time()
    for u, v, d in gt:
        pred = query_distance_table_only_vectorized(
            cluster_tree, borders, G_nb_all, pll_obj, rep_graphs,
            list_indexpath_cross, AB_tables, SAME_tables, node_to_leaf,
            u, v, q_topk_u=q_topk_u, q_topk_v=q_topk_v, q_expand_steps=q_expand_steps,
            timers=timers, counters=counters
        )
        if pred == float("inf"):
            inf_pred += 1
        if pred == d:
            correct += 1
        if (pred != float("inf")) and (d != float("inf")):
            err += abs(pred - d); total_eval += 1
    tQ = time.time() - t0

    total_q = len(gt)
    def avg_ms(key): return 1000.0 * timers[key] / max(1, total_q)
    def pct(key): return 100.0 * timers[key] / max(1e-12, sum(timers.values()))

    print("\n=== 查询阶段分项用时统计（中文） ===")
    print(f"总查询数: {total_q}")
    print(f"使用 AB 表查询次数: {counters['use_ab']}")
    print(f"同叶查询次数: {counters['same_leaf_queries']}")
    print(f"触发扩容次数: {counters['expand_triggers']}")
    print(f"回退分支次数: {counters['fallbacks']}")
    print(f"预测为 inf 次数: {inf_pred}")

    print("\n【用时（秒） | 平均每次（毫秒） | 占比】")
    order = [
        ("lca", "LCA 与子簇定位"),
        ("ab_exist", "读取 AB 表（存在性）"),
        ("lift_left", "左侧抬升（u→b）"),
        ("lift_right", "右侧抬升（v→c）"),
        ("merge", "表上 min-plus 合并"),
        ("expand", "扩容阶段（总）"),
        ("same_leaf", "同叶 PLL 查询"),
    ]
    total_timer_sum = sum(timers.values())
    for k, name in order:
        print(f"- {name:18s}: {timers[k]:8.4f}s  | {avg_ms(k):7.3f} ms/次 | {pct(k):6.2f}%")
    print(f"- 合计（计时器总和）   : {total_timer_sum:8.4f}s")
    print(f"- 评测外层计时（参考） : {tQ:8.4f}s  （含循环/判定等外围开销）")

    rows = [[
        "HierL-E (SAME+AB, vectorized)",
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
    ap.add_argument("--kg_file", type=str, default=None, help="Path to WN18.txt（提供则覆盖 --dataset）")

    ap.add_argument("--resolution", type=float, default=0.3)
    ap.add_argument("--max_cluster_size", type=int, default=1200)
    ap.add_argument("--hl_seed", type=int, default=42)
    ap.add_argument("--max_workers", type=int, default=None)

    # SAME 下建议均为 0（纯查表）
    ap.add_argument("--q_topk_u", type=int, default=0)
    ap.add_argument("--q_topk_v", type=int, default=0)
    ap.add_argument("--q_expand_steps", type=int, default=0)

    ap.add_argument("--mid_limit", type=int, default=0)  # AB 表限边（默认不限）

    # 评测
    ap.add_argument("--eval_pairs", type=int, default=500)
    ap.add_argument("--limit_to_lcc", action="store_true")
    ap.add_argument("--save_eval_set", type=str, default=None)
    ap.add_argument("--load_eval_set", type=str, default=None)

    args = ap.parse_args()
    if not args.limit_to_lcc:
        args.limit_to_lcc = True

    # === 0) 加载图 ===
    if args.kg_file:
        G = load_wn18_graph_aggregated(args.kg_file)
        src_name = f"WN18({os.path.basename(args.kg_file)})-aggregated"
    else:
        G = load_planetoid_graph(args.dataset, root=args.data_root)
        src_name = args.dataset
    print(f"[INFO] Graph: {src_name}, |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

    # === 1) Skeleton ===
    tALL0 = time.time()
    cluster_tree, node_to_leaf, cc_to_leaves, SUPER_ROOT = build_skeleton_leaves(
        G,
        hl_resolution=args.resolution,
        hl_max_cluster_size=args.max_cluster_size,
        random_seed=args.hl_seed,
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

    # === 2) 叶簇表 ===
    borders, pll_labels, G_nb_all, rep_graphs, leaf_stats = build_leaves_tables(
        G, cluster_tree, node_to_leaf, max_workers=args.max_workers
    )

    # 边界点 -> 叶簇
    border_owner_leaf = {}
    for leaf, bl in borders.items():
        for b in bl:
            border_owner_leaf[b] = leaf

    # === 3) 父层 index_graph ===
    index_graphs, parent_stats = build_index_graphs_for_parents(
        G, cluster_tree, borders, rep_graphs, parents_subset=None
    )

    # === 4) 多源 Dijkstra：跨子簇 + SAME 原始桶
    list_indexpath_cross, list_indexpath_same, L_stats = build_list_indexpath_for_parents(
        cluster_tree, index_graphs, borders, rep_graphs, border_owner_leaf, parents_subset=None
    )

    # === 7A) AB 表
    AB_tables = build_AB_tables(cluster_tree, borders, rep_graphs, list_indexpath_cross, mid_limit=args.mid_limit)

    # === 7B) SAME 表（按叶打包）
    print("[INFO] Building SAME tables (leaf-packed) ...")
    t_same0 = time.time()
    SAME_tables = build_SAME_tables(list_indexpath_same, border_owner_leaf, borders)
    t_same1 = time.time()
    print(f"[INFO] SAME tables built in {t_same1 - t_same0:.3f}s; parents_with_SAME={len(SAME_tables)}")

    # === 5) rep_graph（父层，三段式闭包）
    rep_graphs, rep_parent_stats = build_parent_rep_graphs_with_tripartite_closure(
        G, cluster_tree, index_graphs, AB_tables, rep_graphs, borders, G_nb_all, parents_subset=None
    )

    tALL1 = time.time()
    preprocessing_time = tALL1 - tALL0

    # === 查询辅助 ===
    leaf_border_index, pll_obj, rep_nodes_cache = build_aux_indices_for_query(
        borders, pll_labels, rep_graphs
    )

    # === 评测集合 ===
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

    # === 评测：SAME + AB（全向量化） ===
    df_eval = evaluate_table_query(
        G,
        cluster_tree, borders, G_nb_all, pll_obj, rep_graphs,
        list_indexpath_cross, AB_tables, SAME_tables, node_to_leaf,
        gt, preprocessing_time,
        q_topk_u=args.q_topk_u, q_topk_v=args.q_topk_v, q_expand_steps=args.q_expand_steps
    )

    print("\n=== Table-only LCA Query Evaluation (SAME + AB, Vectorized) ===")
    print(df_eval.to_string(index=False))
    print(f"\n[Summary] Preprocessing={preprocessing_time:.3f}s, "
          f"parents_with_AB={len(AB_tables)}, parents_with_SAME={len(SAME_tables)}")

    print("\n[OK] Finished with SAME+AB (vectorized lifting & merging).")

if __name__ == "__main__":
    main()
