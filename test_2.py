
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hierl_e_tablevec.py — Hierarchical-E (steps 1–8)
表存储 + 向量化查询 + Top-K + 自适应扩容回退（保证查询快，容忍构建更慢）
"""

from __future__ import annotations
import argparse
import os
import time
import heapq
import random
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


# ---------- Step 1: skeleton (SUPER_ROOT → CC_ROOT → leaves) ----------
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

    # 叶内 PLL
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
    borders: dict,          # leaf_cid -> list(border nodes)
    rep_graphs: dict,       # child_cid -> nx.Graph (on representatives)
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

        # 1) child -> nodes（代表点集合）；并建立 node -> direct child
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

        # 2) 子簇内部闭包边
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

        # 3) 跨子簇原图边（代表点之间）
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


# ---------- Step 4: list_indexpath（多源 Dijkstra，b→所有 c） ----------
def _make_adj_from_graph(G: nx.Graph):
    adj = {}
    for u in G.nodes():
        lst = []
        for v, data in G[u].items():
            lst.append((v, float(data.get("weight", 1.0))))
        adj[u] = lst
    return adj


def _multi_source_dijkstra_index_worker(args):
    parent, A, adj, child_to_nodes, node_to_child = args

    seeds = list(child_to_nodes.get(A, []))
    if not seeds:
        return parent, A, {}

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

    resA = defaultdict(dict)  # b -> {c: dist}
    for c, dc in dist.items():
        B = node_to_child.get(c)
        if B is None or B == A:
            continue
        b0 = origin.get(c)
        if b0 is None:
            continue
        prev = resA[b0].get(c)
        if (prev is None) or (dc < prev):
            resA[b0][c] = float(dc)

    return parent, A, resA


def build_list_indexpath_for_parents(
    cluster_tree: dict,
    index_graphs: dict,      # parent -> nx.Graph
    borders: dict,           # 叶子边界
    rep_graphs: dict,        # child_cid -> nx.Graph
    parents_subset: list | None = None,
):
    L = defaultdict(lambda: defaultdict(dict))
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
            tasks.append((parent, A, adj, child_to_nodes, node_to_child))
            jobs += 1

        parent_stats[parent] = dict(children=len(cluster_tree[parent]["children"]),
                                    nodes=IG.number_of_nodes(),
                                    edges=IG.number_of_edges(),
                                    jobs=jobs)

    with ProcessPoolExecutor(max_workers=os.cpu_count() or 2) as ex:
        futs = [ex.submit(_multi_source_dijkstra_index_worker, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Step4: list_indexpath (multi-source)", unit="job"):
            parent, A, resA = fut.result()
            L[parent][A] = resA

    return L, parent_stats


# ---------- Step 5: parent rep_graph（抽取与闭包） ----------
def build_parent_rep_graphs(
    G: nx.Graph,
    cluster_tree: dict,
    index_graphs: dict,          # parent -> nx.Graph
    list_indexpath: dict,        # L[parent][A][b][c] = dist
    rep_graphs: dict,            # 已有：叶子 & 下层 parent rep_graphs；会新增/更新：本层 parent
    borders: dict,               # 叶子边界
    parents_subset: list | None = None,
):
    stats = {}

    if parents_subset is None:
        parents = [cid for cid, meta in cluster_tree.items() if meta["children"] and len(meta["children"]) > 1]
    else:
        parents = [p for p in parents_subset if cluster_tree.get(p, {}).get("children")]

    for parent in tqdm(parents, desc="Step5: building parent rep_graphs", unit="parent"):
        IG = index_graphs.get(parent)
        if IG is None or IG.number_of_nodes() == 0:
            rep_graphs[parent] = nx.Graph()
            stats[parent] = dict(rep_nodes=0, rep_edges=0)
            continue

        parent_nodes_set = cluster_tree[parent].get("nodes") or set()

        node_to_child = {}
        for ch in cluster_tree[parent]["children"]:
            if cluster_tree[ch]["level"] == 0:
                nodes_set = set(borders.get(ch, []))
            else:
                RG = rep_graphs.get(ch)
                nodes_set = set(RG.nodes()) if RG is not None else set()
            for x in nodes_set:
                node_to_child[x] = ch

        # 抽取 rep_nodes：与 parent 外部连通的 index 节点
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

        # 同子簇闭包边
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

        # 跨子簇查表边
        Lp = list_indexpath.get(parent, {})
        for A, resA in Lp.items():
            for b, cmap in resA.items():
                if b not in rep_set:
                    continue
                for c, dist_bc in cmap.items():
                    if c not in rep_set:
                        continue
                    if node_to_child.get(b) == node_to_child.get(c):
                        continue
                    w = float(dist_bc)
                    if RGp.has_edge(b, c):
                        if w < RGp[b][c].get("weight", float("inf")):
                            RGp[b][c]["weight"] = w
                    else:
                        RGp.add_edge(b, c, weight=w)

        rep_graphs[parent] = RGp
        stats[parent] = dict(rep_nodes=RGp.number_of_nodes(), rep_edges=RGp.number_of_edges())

    return rep_graphs, stats


# ---------- Step 6: 自底向上递推至更高层（含单子簇透传） ----------
def propagate_bottom_up_all_levels(
    G: nx.Graph,
    cluster_tree: dict,
    borders: dict,
    rep_graphs: dict,   # 将被就地补充上各父层的 rep_graph
):
    levels = sorted({meta["level"] for _, meta in cluster_tree.items()
                     if meta.get("children") is not None and meta["level"] < 0}, reverse=True)
    index_graphs_all = {}
    list_indexpath_all = {}
    per_level_report = []

    for lvl in levels:
        parents_at_level = [cid for cid, meta in cluster_tree.items()
                            if meta.get("children") and meta["level"] == lvl]
        if not parents_at_level:
            continue

        # 1) 单子簇透传
        single = [p for p in parents_at_level if len(cluster_tree[p]["children"]) == 1]
        for p in single:
            child = cluster_tree[p]["children"][0]
            rep_graphs[p] = rep_graphs.get(child, nx.Graph())

        # 2) 多子簇构建
        multi = [p for p in parents_at_level if len(cluster_tree[p]["children"]) > 1]
        if multi:
            t3a = time.time()
            idx_graphs, idx_stats = build_index_graphs_for_parents(
                G, cluster_tree, borders, rep_graphs, parents_subset=multi
            )
            index_graphs_all.update(idx_graphs)
            t3b = time.time()

            t4a = time.time()
            L, L_stats = build_list_indexpath_for_parents(
                cluster_tree, idx_graphs, borders, rep_graphs, parents_subset=multi
            )
            for parent, m in L.items():
                list_indexpath_all[parent] = m
            t4b = time.time()

            t5a = time.time()
            rep_graphs, rep_stats = build_parent_rep_graphs(
                G, cluster_tree, idx_graphs, L, rep_graphs, borders, parents_subset=multi
            )
            t5b = time.time()

            per_level_report.append(dict(
                level=lvl,
                parents=len(parents_at_level),
                single=len(single),
                multi=len(multi),
                step3_time=t3b - t3a,
                step4_time=t4b - t4a,
                step5_time=t5b - t5a,
                idx_nodes_sum=sum(s.get("nodes", 0) for s in idx_stats.values()),
                idx_edges_sum=sum(s.get("edges", 0) for s in idx_stats.values()),
                rep_nodes_sum=sum(s.get("rep_nodes", 0) for s in rep_stats.values()),
                rep_edges_sum=sum(s.get("rep_edges", 0) for s in rep_stats.values()),
            ))
        else:
            per_level_report.append(dict(
                level=lvl, parents=len(parents_at_level), single=len(single), multi=0,
                step3_time=0.0, step4_time=0.0, step5_time=0.0,
                idx_nodes_sum=0, idx_edges_sum=0, rep_nodes_sum=0, rep_edges_sum=0
            ))

    return index_graphs_all, list_indexpath_all, per_level_report, rep_graphs


# ---------- Step 7: 构建查询辅助 + AB 表 ----------
def build_aux_indices_for_query(borders, pll_labels, rep_graphs):
    """叶子 border 的下标表 & 叶内 PLL 对象"""
    leaf_border_index = {}
    for leaf, blist in borders.items():
        leaf_border_index[leaf] = {b: i for i, b in enumerate(blist)}
    pll_obj = {leaf: PrunedPLLIndex.from_labels(labels) for leaf, labels in pll_labels.items()}

    rep_nodes_cache = {}
    for cid, rg in rep_graphs.items():
        if isinstance(rg, nx.Graph):
            rep_nodes_cache[cid] = set(rg.nodes())
        else:
            rep_nodes_cache[cid] = set()
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


def _leaf_array_topk(G_nb_leaf: dict, borders_leaf: list, node, k: int):
    """从 G_nb 数组里拿 Top-K（自动扩容到所有有限项）"""
    arr = G_nb_leaf.get(node)
    if arr is None or arr.size == 0:
        return {}, []
    finite = np.isfinite(arr)
    if not finite.any():
        return {}, []
    idx_all = np.where(finite)[0]
    if k is None or k <= 0 or k >= idx_all.size:
        idx = idx_all
    else:
        sub = arr[idx_all]
        kk = min(k, sub.size)
        sub_idx = np.argpartition(sub, kk - 1)[:kk]
        idx = idx_all[sub_idx]
    out = {borders_leaf[i]: float(arr[i]) for i in idx}
    return out, idx_all.tolist()  # 返回“所有有限”的下标，便于扩容


# ===== AB 表构建 =====
def build_AB_tables(cluster_tree, borders, rep_graphs, list_indexpath_total, mid_limit=0):
    """
    由 list_indexpath_total 构建表：
    AB_tables[L][(A,B)] = {
        'b_nodes': [node,...], 'c_nodes': [node,...],
        'b_index': {node:row}, 'c_index': {node:col},
        'rows': np.int32, 'cols': np.int32, 'mids': np.float64,
        'row_ptr': np.int32  # CSR 行指针，长度 = len(b_nodes)+1
    }
    """
    AB_tables = defaultdict(dict)

    for L, mapA in list_indexpath_total.items():
        # 为该 parent 提前收集 (A,B) -> per-b 列表
        pair_bucket = defaultdict(lambda: defaultdict(list))  # (A,B) -> b -> [(mid,c)]
        # 准备 node->child 映射
        node_to_child = {}
        for ch in cluster_tree[L]["children"]:
            if cluster_tree[ch]["level"] == 0:
                nodes_set = borders.get(ch, [])
            else:
                RG = rep_graphs.get(ch)
                nodes_set = RG.nodes() if RG is not None else []
            for x in nodes_set:
                node_to_child[x] = ch

        for A, resA in mapA.items():         # resA: b -> {c: dist}
            for b, cmap in resA.items():
                for c, dist_bc in cmap.items():
                    B = node_to_child.get(c)
                    if B is None or B == A:
                        continue
                    pair_bucket[(A, B)][b].append((float(dist_bc), c))

        # 构建每个 (A,B) 的表
        for (A, B), by_b in pair_bucket.items():
            # 1) 行列节点
            b_nodes = sorted(by_b.keys(), key=node_sort_key)
            c_nodes_set = set()
            # 先按 mid_limit 过滤每个 b 的候选
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

            b_index = {b: i for i, b in enumerate(b_nodes)}
            c_index = {c: i for i, c in enumerate(c_nodes)}

            # 2) 按行展开到 rows/cols/mids，并生成 row_ptr
            rows = []
            cols = []
            mids = []
            row_ptr = [0]
            for b in b_nodes:
                for dist_bc, c in by_b_kept[b]:
                    rows.append(b_index[b])
                    cols.append(c_index[c])
                    mids.append(dist_bc)
                row_ptr.append(len(rows))
            if not rows:
                continue
            table = dict(
                b_nodes=b_nodes, c_nodes=c_nodes,
                b_index=b_index, c_index=c_index,
                rows=np.asarray(rows, dtype=np.int32),
                cols=np.asarray(cols, dtype=np.int32),
                mids=np.asarray(mids, dtype=np.float64),
                row_ptr=np.asarray(row_ptr, dtype=np.int32),
            )
            AB_tables[L][(A, B)] = table

    return AB_tables


# ===== 抬升：只到 (A,B) 所需的行/列 =====
def lift_u_to_rows_for_pair(cluster_tree, u, leaf_u, Au, L, G_nb_all, borders, list_indexpath_total, table, k_top=16, expand_all=False):
    """
    返回 dict: {b(row node in S_Au): dist(u->b)}
    - Au 为叶：直接从 G_nb_all[Au][u] 取（限制到 b∈table['b_nodes']）
    - Au 非叶：用 leaf_u 的边界 Top-K（或全部有限） + list_indexpath_total[Au][leaf_u][b0][b]
      仅计算 table['b_nodes'] 上的值，避免全量抬升
    """
    b_nodes = table.get("b_nodes", [])
    if not b_nodes:
        return {}

    if cluster_tree[Au]["level"] == 0:
        blist = borders.get(Au, [])
        u_map, _ = _leaf_array_topk(G_nb_all[Au], blist, u, k_top if not expand_all else 0)
        return {b: d for b, d in u_map.items() if b in table["b_index"]}

    # 非叶：先拿 leaf_u 对其自身边界的 Top-K/全部有限
    blist_u = borders.get(leaf_u, [])
    du_map, _ = _leaf_array_topk(G_nb_all[leaf_u], blist_u, u, k_top if not expand_all else 0)
    if not du_map:
        return {}

    # list_indexpath_total[Au][leaf_u] : b0 -> {target b in Au : mid}
    L_Au = list_indexpath_total.get(Au, {}).get(leaf_u, {})
    if not L_Au:
        return {}

    out = {}
    for b in b_nodes:
        best = float("inf")
        for b0, dub0 in du_map.items():
            cmap = L_Au.get(b0)
            if not cmap:
                continue
            mid = cmap.get(b, float("inf"))
            if mid == float("inf"):
                continue
            cand = dub0 + mid
            if cand < best:
                best = cand
        if best < float("inf"):
            out[b] = best
    return out


def lift_v_to_cols_for_pair(cluster_tree, v, leaf_v, Bv, L, G_nb_all, borders, list_indexpath_total, table, k_top=16, expand_all=False):
    """对右侧同理：返回 {c(col node in S_Bv): dist(v->c)}"""
    c_nodes = table.get("c_nodes", [])
    if not c_nodes:
        return {}

    if cluster_tree[Bv]["level"] == 0:
        blist = borders.get(Bv, [])
        v_map, _ = _leaf_array_topk(G_nb_all[Bv], blist, v, k_top if not expand_all else 0)
        return {c: d for c, d in v_map.items() if c in table["c_index"]}

    blist_v = borders.get(leaf_v, [])
    dv_map, _ = _leaf_array_topk(G_nb_all[leaf_v], blist_v, v, k_top if not expand_all else 0)
    if not dv_map:
        return {}

    L_Bv = list_indexpath_total.get(Bv, {}).get(leaf_v, {})
    if not L_Bv:
        return {}

    out = {}
    for c in c_nodes:
        best = float("inf")
        for b0, dvb0 in dv_map.items():
            cmap = L_Bv.get(b0)
            if not cmap:
                continue
            mid = cmap.get(c, float("inf"))
            if mid == float("inf"):
                continue
            cand = dvb0 + mid
            if cand < best:
                best = cand
        if best < float("inf"):
            out[c] = best
    return out


# ===== 在 AB 表上做向量化最优组合（仅遍历左侧出现的行段） =====
def table_min_plus_best(table, left_map: dict, right_map: dict) -> float:
    if not table or (not left_map) or (not right_map):
        return float("inf")

    b_index = table["b_index"]; c_index = table["c_index"]
    row_ptr = table["row_ptr"]; cols = table["cols"]; mids = table["mids"]

    # 构建右侧 dv 全数组（缺省 inf）
    C = len(c_index)
    dv_full = np.full((C,), np.inf, dtype=np.float64)
    for c, val in right_map.items():
        j = c_index.get(c)
        if j is not None:
            dv_full[j] = float(val)

    best = float("inf")
    # 仅遍历左侧存在的 b 的行段
    for b, du in left_map.items():
        r = b_index.get(b)
        if r is None:
            continue
        s = row_ptr[r]; e = row_ptr[r + 1]
        if s == e:
            continue
        cols_r = cols[s:e]
        dv_r = dv_full[cols_r]
        mask = np.isfinite(dv_r)
        if not mask.any():
            continue
        vals = du + mids[s:e][mask] + dv_r[mask]
        m = float(np.min(vals))
        if m < best:
            best = m
    return best


# ===== 查询（表 + Top-K + 扩容回退） =====
def query_distance_table_only_vectorized(
    cluster_tree, borders, G_nb_all, pll_obj, rep_graphs,
    list_indexpath_total, AB_tables, node_to_leaf,
    u, v, q_topk_u=16, q_topk_v=16, q_expand_steps=2
) -> float:
    if u == v:
        return 0.0

    leaf_u = node_to_leaf.get(u)
    leaf_v = node_to_leaf.get(v)
    if leaf_u is None or leaf_v is None:
        return float("inf")

    if leaf_u == leaf_v:
        pll = pll_obj.get(leaf_u)
        return float("inf") if pll is None else float(pll.query(u, v))

    # 找 LCA 和在 L 层的直接孩子 Au, Bv
    L, Au, Bv, leaf_u_id, leaf_v_id = find_lca_and_children(cluster_tree, node_to_leaf, u, v)

    # 找到对应 (A,B) 的 AB 表；没有表就回退 dict（极少）
    table = AB_tables.get(L, {}).get((Au, Bv), None)
    if table is None:
        # 罕见回退：只在 left_topk 所涉及的候选上做组合
        resAu = list_indexpath_total.get(L, {}).get(Au, {})
        # 左侧：构造 b 列表（用 Au 为叶/非叶的逻辑）
        left_map = {}
        if cluster_tree[Au]["level"] == 0:
            blist = borders.get(Au, [])
            left_map, _ = _leaf_array_topk(G_nb_all[Au], blist, u, q_topk_u)
        else:
            # 先取 leaf_u 的边界，再通过 Au 的表抬升到 b（只保留出现在 resAu 的 b）
            blist_u = borders.get(leaf_u_id, [])
            du_map, _ = _leaf_array_topk(G_nb_all[leaf_u_id], blist_u, u, q_topk_u)
            L_Au = list_indexpath_total.get(Au, {}).get(leaf_u_id, {})
            # 目标 b 只取 resAu 里出现的键
            b_candidates = set(resAu.keys())
            for b in b_candidates:
                best = float("inf")
                for b0, dub0 in du_map.items():
                    cmap = L_Au.get(b0, {})
                    mid = cmap.get(b, float("inf"))
                    if not np.isfinite(mid):
                        continue
                    cand = dub0 + mid
                    if cand < best:
                        best = cand
                if np.isfinite(best):
                    left_map[b] = best

        if not left_map:
            return float("inf")

        # 右侧：只计算 right_candidates = ⋃_{b∈left_keys} resAu[b].keys()
        right_candidates = set()
        for b in left_map.keys():
            right_candidates |= set(resAu.get(b, {}).keys())
        if not right_candidates:
            return float("inf")

        right_map = {}
        if cluster_tree[Bv]["level"] == 0:
            blistB = borders.get(Bv, [])
            dv_all, _ = _leaf_array_topk(G_nb_all[Bv], blistB, v, 0)  # 0 = 全部有限
            for c in right_candidates:
                dvc = dv_all.get(c)
                if dvc is not None:
                    right_map[c] = dvc
        else:
            blist_v = borders.get(leaf_v_id, [])
            dv_map0, _ = _leaf_array_topk(G_nb_all[leaf_v_id], blist_v, v, 0)
            L_Bv = list_indexpath_total.get(Bv, {}).get(leaf_v_id, {})
            for c in right_candidates:
                best = float("inf")
                for b0, dvb0 in dv_map0.items():
                    cmap = L_Bv.get(b0, {})
                    mid = cmap.get(c, float("inf"))
                    if not np.isfinite(mid):
                        continue
                    cand = dvb0 + mid
                    if cand < best:
                        best = cand
                if np.isfinite(best):
                    right_map[c] = best

        if not right_map:
            return float("inf")

        # 组合（只遍历 left_keys × right_keys 的交且有 mid 的项）
        best = float("inf")
        rk = set(right_map.keys())
        for b, du in left_map.items():
            cmap = resAu.get(b, {})
            if not cmap:
                continue
            for c in rk:
                mid = cmap.get(c, float("inf"))
                if not np.isfinite(mid):
                    continue
                cand = du + mid + right_map[c]
                if cand < best:
                    best = cand
        return float(best)

    # 正常路径：表 + Top-K 抬升 + 扩容回退
    left_map = lift_u_to_rows_for_pair(cluster_tree, u, leaf_u_id, Au, L, G_nb_all, borders, list_indexpath_total,
                                       table, k_top=q_topk_u, expand_all=False)
    right_map = lift_v_to_cols_for_pair(cluster_tree, v, leaf_v_id, Bv, L, G_nb_all, borders, list_indexpath_total,
                                        table, k_top=q_topk_v, expand_all=False)
    best = table_min_plus_best(table, left_map, right_map)
    if np.isfinite(best):
        return float(best)

    # 扩容回退
    for step in range(q_expand_steps):
        if step == 0:
            if len(left_map) <= len(right_map):
                left_map = lift_u_to_rows_for_pair(cluster_tree, u, leaf_u_id, Au, L, G_nb_all, borders, list_indexpath_total,
                                                   table, k_top=q_topk_u, expand_all=True)
            else:
                right_map = lift_v_to_cols_for_pair(cluster_tree, v, leaf_v_id, Bv, L, G_nb_all, borders, list_indexpath_total,
                                                    table, k_top=q_topk_v, expand_all=True)
        else:
            left_map = lift_u_to_rows_for_pair(cluster_tree, u, leaf_u_id, Au, L, G_nb_all, borders, list_indexpath_total,
                                               table, k_top=q_topk_u, expand_all=True)
            right_map = lift_v_to_cols_for_pair(cluster_tree, v, leaf_v_id, Bv, L, G_nb_all, borders, list_indexpath_total,
                                                table, k_top=q_topk_v, expand_all=True)
        best = table_min_plus_best(table, left_map, right_map)
        if np.isfinite(best):
            return float(best)

    return float("inf")


# ---------- Step 8: 评测与导出 ----------
def sample_entity_pairs(G, n_pairs=1000, in_lcc=True, rng_seed=42):
    entity_nodes = list(G.nodes())
    if len(entity_nodes) < 2:
        raise RuntimeError("图中节点过少，无法评测。")

    comps = list(nx.connected_components(G))
    if in_lcc and comps:
        lcc = max(comps, key=len)
        entity_nodes = [n for n in entity_nodes if n in lcc]

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
        raise RuntimeError("没有包含≥2节点的连通分量，无法评测。")

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
    list_indexpath_total, AB_tables, node_to_leaf,
    gt, preprocessing_time,
    q_topk_u=16, q_topk_v=16, q_expand_steps=2
):
    correct = total_eval = 0
    err = 0.0
    inf_pred = 0

    t0 = time.time()
    for u, v, d in gt:
        pred = query_distance_table_only_vectorized(
            cluster_tree, borders, G_nb_all, pll_obj, rep_graphs,
            list_indexpath_total, AB_tables, node_to_leaf,
            u, v, q_topk_u=q_topk_u, q_topk_v=q_topk_v, q_expand_steps=q_expand_steps
        )
        if pred == float("inf"):
            inf_pred += 1
        if pred == d:
            correct += 1
        if (pred != float("inf")) and (d != float("inf")):
            err += abs(pred - d); total_eval += 1
    tQ = time.time() - t0

    rows = [[
        "HierL-E (table-vec, Top-K+expand)",
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

    ap.add_argument("--resolution", type=float, default=0.3, help="hierarchical_leiden 的 resolution")
    ap.add_argument("--max_cluster_size", type=int, default=1200, help="hierarchical_leiden 的 max_cluster_size")
    ap.add_argument("--hl_seed", type=int, default=42)
    ap.add_argument("--max_workers", type=int, default=None)

    # 查询期 Top-K & 扩容
    ap.add_argument("--q_topk_u", type=int, default=16, help="查询：u 侧 Top-K（叶边界或抬升前置 Top-K）")
    ap.add_argument("--q_topk_v", type=int, default=16, help="查询：v 侧 Top-K")
    ap.add_argument("--q_expand_steps", type=int, default=2, help="查询：自适应扩容步数（0/1/2）")

    # 构建 AB 表时可选限边（默认 0=不限，保障召回）
    ap.add_argument("--mid_limit", type=int, default=0, help="构建 AB 表时每个 b 的 c 候选上限（0 表示不限制）")

    # 评测相关
    ap.add_argument("--eval_pairs", type=int, default=500, help="评测样本数")
    ap.add_argument("--limit_to_lcc", action="store_true",
                    help="评测仅在 LCC 内抽样（默认启用，若不加则代码里会置 True）")
    ap.add_argument("--save_eval_set", type=str, default=None,
                    help="保存抽样得到的 (source,target,dist) CSV 路径")
    ap.add_argument("--load_eval_set", type=str, default=None,
                    help="从 CSV 加载 (source,target,dist) 作为评测集合（优先于抽样与真值计算）")

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

    # === 3) level-1 parents index_graph ===
    index_graphs_lvl1, parent_stats_lvl1 = build_index_graphs_for_parents(
        G, cluster_tree, borders, rep_graphs, parents_subset=None
    )

    # === 4) list_indexpath（level-1） ===
    list_indexpath_lvl1, L_stats_lvl1 = build_list_indexpath_for_parents(
        cluster_tree, index_graphs_lvl1, borders, rep_graphs, parents_subset=None
    )

    # === 5) rep_graph（level-1 parents） ===
    rep_graphs, rep_parent_stats_lvl1 = build_parent_rep_graphs(
        G, cluster_tree, index_graphs_lvl1, list_indexpath_lvl1, rep_graphs, borders, parents_subset=None
    )

    # === 6) 自底向上递推到更高层 ===
    idx_all_2, L_all_2, per_level_report, rep_graphs = propagate_bottom_up_all_levels(
        G, cluster_tree, borders, rep_graphs
    )

    tALL1 = time.time()
    preprocessing_time = tALL1 - tALL0

    # 合并所有层
    index_graphs_total = dict(index_graphs_lvl1); index_graphs_total.update(idx_all_2)
    list_indexpath_total = dict(list_indexpath_lvl1); list_indexpath_total.update(L_all_2)

    # === 7) 查询所需辅助索引 ===
    leaf_border_index, pll_obj, rep_nodes_cache = build_aux_indices_for_query(
        borders, pll_labels, rep_graphs
    )

    # === 7.5) 构建 AB 表（表存储 + CSR） ===
    print("[INFO] Building AB tables (array/CSR) ...")
    tAB0 = time.time()
    AB_tables = build_AB_tables(cluster_tree, borders, rep_graphs, list_indexpath_total, mid_limit=args.mid_limit)
    tAB1 = time.time()
    print(f"[INFO] AB tables built in {tAB1 - tAB0:.3f}s; parents_with_tables={len(AB_tables)}")

    # === 8) 评测 ===
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

    df_eval = evaluate_table_query(
        G,
        cluster_tree, borders, G_nb_all, pll_obj, rep_graphs,
        list_indexpath_total, AB_tables, node_to_leaf,
        gt, preprocessing_time,
        q_topk_u=args.q_topk_u, q_topk_v=args.q_topk_v, q_expand_steps=args.q_expand_steps
    )

    print("\n=== Table-only LCA Query Evaluation (Table+Vectorized) ===")
    print(df_eval.to_string(index=False))
    print(f"\n[Summary] Preprocessing={preprocessing_time:.3f}s, "
          f"AB_tables_parents={len(AB_tables)}, Total parents with L-table={len(list_indexpath_total)}")

    if per_level_report:
        df6 = pd.DataFrame(per_level_report).sort_values(by="level", ascending=False)
        with pd.option_context("display.max_rows", 50, "display.max_colwidth", 60):
            print("\n[Step6] Bottom-up propagation summary (per level)")
            print(df6.to_string(index=False))

    print("\n[OK] Steps 1–8 finished (with array/CSR tables and vectorized query).")

if __name__ == "__main__":
    main()
