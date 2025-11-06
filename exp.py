#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HierL-E (Supergraph Hierarchy版)
- 变化点（满足你的需求）：
  * Step1 改为：先得到底层簇；把“每个簇抽象成一个点”，依据原图上的跨簇边构建“超图”（cluster-nodes graph）；
    对超图再次聚类，得到上一层父簇；重复此过程，形成“多层树状（不限定几叉）”层次结构。
  * Step3 起改为：父层 index_graph 采用同簇完全闭包，并在 index graph 上做多源最短路收集 child-pair top-k 路径，查询阶段用分层搜索动态拼接最短路。

依赖：
- networkx, numpy, pandas, tqdm
- graspologic (for hierarchical_leiden)
- (可选) torch_geometric 用于加载 Planetoid 数据集
"""

from __future__ import annotations
import argparse
import os
import time
import heapq
import random
from collections import defaultdict

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

def _cid_str(cid):
    if isinstance(cid, tuple):
        return ",".join(map(str, cid))
    return str(cid)


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


# ---------- PLL (query-only) ----------
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


# ---------- Leiden结果提取 ----------
def final_partition_from_hl(hl_result, G: nx.Graph) -> dict:
    """
    把 graspologic.hierarchical_leiden 的结果抽出最终簇（is_final_cluster=True）
    返回: node -> local_cluster_id
    """
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


# =========================================================
# ========== Step1（NEW）: 超图分层 —— 多层树状结构 ==========
# =========================================================
def _leiden_partition_on_graph(H: nx.Graph, resolution=0.5, max_cluster_size=10**9, seed=42):
    """
    在（超）图 H 上做一次 Leiden（底层只有一次切分），取最终簇映射: node -> cluster_id
    """
    if H.number_of_nodes() == 0:
        return {}
    hl = hierarchical_leiden(
        H,
        max_cluster_size=max_cluster_size,
        resolution=resolution,
        use_modularity=True,
        random_seed=seed,
        check_directed=True,
    )
    return final_partition_from_hl(hl, H)

def _build_supergraph_from_partition(G: nx.Graph, part: dict) -> nx.Graph:
    """
    给定原图 G 的“当前层”划分 part: node->cid
    构造超图 SG：每个簇一个超点；若原图存在跨簇边 (u,v) 使 part[u]!=part[v]，
    则在超点 (part[u], part[v]) 加一条边（权重累计为出现次数或最小权重，这里选“计数和”）。
    """
    SG = nx.Graph()
    # 添加超点
    cids = set(part.values())
    SG.add_nodes_from(cids)
    # 累计跨簇边权（用计数；也可改为和/平均/最小/最大等）
    cut_w = defaultdict(float)
    for u, v, data in G.edges(data=True):
        cu, cv = part.get(u), part.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        a, b = (cu, cv) if cu <= cv else (cv, cu)
        cut_w[(a, b)] += 1.0
    for (a, b), w in cut_w.items():
        SG.add_edge(a, b, weight=float(w))
    return SG


def _build_supergraph_from_level_sets(sub: nx.Graph, level_map: dict) -> tuple[nx.Graph, dict]:
    """构建当前层的超图，并返回节点到超点的映射。"""
    node2super = {}
    for super_cid, node_set in level_map.items():
        for n in node_set:
            node2super[n] = super_cid
    SG = _build_supergraph_from_partition(sub, node2super)
    return SG, node2super

def build_hierarchical_supergraph_tree(
    G: nx.Graph,
    base_resolution=0.3,
    base_max_cluster_size=2000,
    super_resolutions=(0.8, 0.6, 0.45, 0.3, 0.2),
    super_max_cluster_size=50,
    supernode_threshold=50,
    seed=42,
):
    """
    目标：构建一个“多层树”：
      - level=-1: CC 根（parent=None）
      - level=0 : 最底层叶簇（来自原图直接聚类）
      - level>=1: 通过超图递归聚类逐层向上合并（多叉，不限几叉）
    返回: cluster_tree, node_to_leaf, hierarchy_levels
    """
    cluster_tree = {}
    node_to_leaf = {}
    hierarchy_levels = defaultdict(list)

    comps = list(nx.connected_components(G))
    comps_sorted = sorted(comps, key=lambda s: -len(s))

    for cc_id, nodes in enumerate(comps_sorted):
        sub = G.subgraph(nodes).copy()
        cc_cid = ("cc", cc_id)
        cluster_tree[cc_cid] = dict(level=-1, parent=None, children=[], nodes=set(nodes))
        hierarchy_levels[-1].append(cc_cid)

        base_part = _leiden_partition_on_graph(
            sub, resolution=base_resolution, max_cluster_size=base_max_cluster_size, seed=seed
        )
        cid2nodes = defaultdict(list)
        for n, cid in base_part.items():
            cid2nodes[cid].append(n)
        cur_level_nodes = []
        for local_cid, nlist in cid2nodes.items():
            leaf_cid = ("leaf", cc_id, int(local_cid))
            cluster_tree[leaf_cid] = dict(level=0, parent=cc_cid, children=[], nodes=set(nlist))
            cluster_tree[cc_cid]["children"].append(leaf_cid)
            for n in nlist:
                node_to_leaf[n] = leaf_cid
            cur_level_nodes.append(leaf_cid)
            hierarchy_levels[0].append(leaf_cid)

        cur_level_map = {leaf_cid: set(cluster_tree[leaf_cid]["nodes"]) for leaf_cid in cur_level_nodes}
        level = 1
        res_list = list(super_resolutions) if super_resolutions else [0.5]
        threshold = max(1, int(supernode_threshold))
        while len(cur_level_map) > threshold:
            SG, _ = _build_supergraph_from_level_sets(sub, cur_level_map)

            if SG.number_of_edges() == 0:
                break

            res = res_list[min(level - 1, len(res_list) - 1)]
            part_super = _leiden_partition_on_graph(
                SG, resolution=res, max_cluster_size=super_max_cluster_size, seed=seed + level
            )

            groups = defaultdict(list)
            for super_node, gid in part_super.items():
                groups[gid].append(super_node)

            if len(groups) >= len(cur_level_map):
                break

            next_level_map = {}
            for gid, children_cids in groups.items():
                nodes_union = set()
                for ch in children_cids:
                    nodes_union.update(cur_level_map[ch])

                parent_cid = ("par", cc_id, level, int(gid))
                cluster_tree[parent_cid] = dict(level=level, parent=None, children=[], nodes=nodes_union)
                parents_of_children = {cluster_tree[ch]["parent"] for ch in children_cids}
                assert len(parents_of_children) == 1, "内部错误：同层children的parent不唯一"
                old_parent = list(parents_of_children)[0]
                cluster_tree[parent_cid]["parent"] = old_parent

                new_children_list = []
                for c in cluster_tree[old_parent]["children"]:
                    if c not in children_cids:
                        new_children_list.append(c)
                new_children_list.append(parent_cid)
                cluster_tree[old_parent]["children"] = new_children_list

                for ch in children_cids:
                    cluster_tree[ch]["parent"] = parent_cid
                    cluster_tree[parent_cid]["children"].append(ch)

                next_level_map[parent_cid] = nodes_union
                hierarchy_levels[level].append(parent_cid)

            cur_level_map = next_level_map
            level += 1

    levels_output = {lvl: list(ids) for lvl, ids in hierarchy_levels.items()}
    return cluster_tree, node_to_leaf, levels_output


# =========================================================
# ========== Step2: per-leaf PLL + borders + rep_graph ==========
# =========================================================
def _build_one_leaf(args):
    leaf_cid, leaf_nodes, neigh_map, G = args
    nodes = sorted(leaf_nodes, key=node_sort_key)
    node_set = set(nodes)

    # 边界点：与节点集外相邻者
    border = [n for n in nodes if any((nbr not in node_set) for nbr in neigh_map[n])]
    border = sorted(border, key=node_sort_key)
    L = len(border)

    # 构建子图
    subg = nx.Graph()
    subg.add_nodes_from(nodes)
    for u in nodes:
        for v, data in G[u].items():
            if v in node_set and u < v:
                subg.add_edge(u, v, weight=float(data.get("weight", 1.0)))

    # 选取 PLL 顺序（尽量小离心率）
    try:
        ecc = {}
        for comp in nx.connected_components(subg):
            sg = subg.subgraph(comp)
            ecc.update(nx.eccentricity(sg))
        order = sorted(nodes, key=lambda n: ecc.get(n, 0))
    except Exception:
        order = list(nodes)

    # inline build labels
    labels = {n: {} for n in nodes}
    for root in order:
        dist = {root: 0.0}
        heap = [(0.0, 0, root)]
        counter = 0

        def _query(lu, lv):
            best = float("inf")
            if len(lu) > len(lv):
                lu, lv = lv, lu
            for lm, du in lu.items():
                dv = lv.get(lm)
                if dv is not None:
                    s = du + dv
                    if s < best:
                        best = s
            return best

        while heap:
            d, _, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if _query(labels[root], labels[u]) <= d:
                continue
            labels[u][root] = d
            for v, data in subg[u].items():
                w = float(data.get("weight", 1.0))
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    counter += 1
                    heapq.heappush(heap, (nd, counter, v))
    label_entries = sum(len(m) for m in labels.values())

    # G_nb：节点→边界数组（用 PLL 合成得到距离）
    G_nb = {}
    for n in nodes:
        arr = np.empty((L,), dtype=np.float32)
        lu = labels[n]
        for i, b in enumerate(border):
            lv = labels[b]
            best = float("inf")
            if len(lu) > len(lv):
                lu2, lv2 = lv, lu
            else:
                lu2, lv2 = lu, lv
            for lm, du in lu2.items():
                dv = lv2.get(lm)
                if dv is not None:
                    s = du + dv
                    if s < best:
                        best = s
            arr[i] = best
        G_nb[n] = arr

    # 叶 rep_graph：边界点完全图（边权= PLL 距离）
    RG = nx.Graph()
    RG.add_nodes_from(border)
    if L > 1:
        for i, bi in enumerate(border):
            arr_i = G_nb[bi]
            for j in range(i + 1, L):
                bj = border[j]
                dij = float(arr_i[j])
                if np.isfinite(dij):
                    RG.add_edge(bi, bj, weight=dij)

    return leaf_cid, border, labels, G_nb, RG, label_entries


def build_leaves_tables(G: nx.Graph, cluster_tree: dict, max_workers=None):
    max_workers = max_workers or (os.cpu_count() or 2)
    neigh_map = {n: list(G.neighbors(n)) for n in G.nodes()}
    tasks = []
    for cid, meta in cluster_tree.items():
        if meta["level"] == 0:
            tasks.append((cid, meta["nodes"], neigh_map, G))

    borders, pll_labels, G_nb_all, rep_graphs, stats = {}, {}, {}, {}, {}
    with os.popen(""):  # no-op 防止个别 shell 下 tqdm 卡顿
        pass
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with tqdm(total=len(tasks), desc="Step2: building leaves", unit="leaf") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_build_one_leaf, t) for t in tasks]
            for fut in as_completed(futs):
                leaf_cid, border, labels, G_nb, RG, label_entries = fut.result()
                borders[leaf_cid] = border
                pll_labels[leaf_cid] = labels
                G_nb_all[leaf_cid] = G_nb
                rep_graphs[leaf_cid] = RG
                stats[leaf_cid] = dict(nodes=len(cluster_tree[leaf_cid]["nodes"]),
                                       borders=len(border),
                                       pll_label_entries=label_entries,
                                       rep_edges=RG.number_of_edges())
                pbar.update(1)
    return borders, pll_labels, G_nb_all, rep_graphs, stats


# ---------- Step3: 父层 index_graph（完全闭包） ----------
def _ensure_cluster_rep_graph(
    G: nx.Graph,
    cluster_tree: dict,
    borders: dict,
    rep_graphs: dict,
    cid,
):
    """
    确保任意层级的 cluster 都拥有边界点列表与 representative graph。

    之前仅为叶子构建 rep_graph，这会导致更高层 parent 的子簇
    （非叶）在 Step3 中没有可用的网关候选，从而上层 AB/L2 表为空。
    这里在需要时动态补全：
      - 计算该 cluster 的边界点（存在跨簇邻居的节点）。
      - 在 cluster 内部执行 Dijkstra，得到边界点之间的最短路，
        并用这些距离构建一个完全图作为 rep_graph。
    """

    RG = rep_graphs.get(cid)
    border = borders.get(cid)

    if RG is not None and border is not None:
        cluster_tree[cid]["border"] = border
        return RG, border

    node_set = cluster_tree[cid]["nodes"]
    if not isinstance(node_set, set):
        node_set = set(node_set)
        cluster_tree[cid]["nodes"] = node_set
    else:
        node_set = set(node_set)

    if border is None:
        border = [
            n for n in node_set
            if any((nbr not in node_set) for nbr in G[n])
        ]
        border = sorted(border, key=node_sort_key)
        borders[cid] = border
        cluster_tree[cid]["border"] = border

    if RG is not None:
        cluster_tree[cid]["border"] = border
        return RG, border

    RG = nx.Graph()
    RG.add_nodes_from(border)
    if len(border) > 1:
        allowed = node_set
        border_set = set(border)
        for src in border:
            dist = {}
            seen = {src: 0.0}
            heap = [(0.0, src)]
            while heap:
                d, u = heapq.heappop(heap)
                if d != seen[u]:
                    continue
                if u in border_set and u != src:
                    prev = dist.get(u)
                    if prev is None or d < prev:
                        dist[u] = d
                for v, data in G[u].items():
                    if v not in allowed:
                        continue
                    nd = d + float(data.get("weight", 1.0))
                    if nd < seen.get(v, float("inf")):
                        seen[v] = nd
                        heapq.heappush(heap, (nd, v))
            for tgt, d in dist.items():
                if not np.isfinite(d):
                    continue
                if RG.has_edge(src, tgt):
                    if d < RG[src][tgt].get("weight", float("inf")):
                        RG[src][tgt]["weight"] = d
                else:
                    RG.add_edge(src, tgt, weight=float(d))

    rep_graphs[cid] = RG
    borders[cid] = border
    cluster_tree[cid]["border"] = border
    return RG, border


def build_index_graphs_for_parents(
    G: nx.Graph,
    cluster_tree: dict,
    borders: dict,
    rep_graphs: dict,
    parents_subset=None,
):
    index_graphs = {}
    stats = {}

    if parents_subset is None:
        parents = [cid for cid, meta in cluster_tree.items() if meta.get("children") and len(meta["children"]) > 1]
    else:
        parents = [p for p in parents_subset if cluster_tree.get(p, {}).get("children")]

    for parent in tqdm(parents, desc="Step3: 完全闭包 index graph", unit="parent"):
        children = cluster_tree[parent]["children"]
        if len(children) <= 1:
            continue

        node_to_child = {}
        for ch in children:
            nodes = cluster_tree[ch]["nodes"]
            if not isinstance(nodes, set):
                nodes = set(nodes)
                cluster_tree[ch]["nodes"] = nodes
            for n in nodes:
                node_to_child[n] = ch

        IG = nx.Graph()

        for ch in children:
            RG, border = _ensure_cluster_rep_graph(G, cluster_tree, borders, rep_graphs, ch)
            for node in border:
                IG.add_node(node)
            for u, v, data in RG.edges(data=True):
                w = float(data.get("weight", 1.0))
                if IG.has_edge(u, v):
                    if w < IG[u][v].get("weight", float("inf")):
                        IG[u][v]["weight"] = w
                else:
                    IG.add_edge(u, v, weight=w)

        parent_nodes = cluster_tree[parent]["nodes"]
        if not isinstance(parent_nodes, set):
            parent_nodes = set(parent_nodes)
            cluster_tree[parent]["nodes"] = parent_nodes

        for n in parent_nodes:
            if n not in IG:
                continue
            ch_n = node_to_child.get(n)
            if ch_n is None:
                continue
            for nbr, data in G[n].items():
                if nbr not in IG:
                    continue
                ch_m = node_to_child.get(nbr)
                if ch_m is None or ch_m == ch_n:
                    continue
                w = float(data.get("weight", 1.0))
                if IG.has_edge(n, nbr):
                    if w < IG[n][nbr].get("weight", float("inf")):
                        IG[n][nbr]["weight"] = w
                else:
                    IG.add_edge(n, nbr, weight=w)

        index_graphs[parent] = IG
        cluster_tree[parent]["node_to_child"] = node_to_child

        parent_border = set()
        for ch in children:
            parent_border.update(cluster_tree[ch].get("border", []))
        cluster_tree[parent]["border"] = sorted(parent_border, key=node_sort_key)

        stats[parent] = dict(
            index_nodes=IG.number_of_nodes(),
            index_edges=IG.number_of_edges(),
            parent_border_count=len(parent_border),
        )

    return index_graphs, stats


def build_parent_pair_topk(
    cluster_tree: dict,
    index_graphs: dict,
    pair_topk: int = 3,
):
    """在父层 index graph 上运行多源最短路，记录 child 之间 top-k 边界组合。"""
    pair_paths = {}
    for parent, IG in index_graphs.items():
        if IG is None or IG.number_of_nodes() == 0:
            continue
        node_to_child = cluster_tree.get(parent, {}).get("node_to_child", {})
        per_pair = {}
        for child in cluster_tree[parent]["children"]:
            seeds = [n for n in IG.nodes() if node_to_child.get(n) == child]
            if not seeds:
                continue
            dist = {}
            origin = {}
            heap = []
            ctr = 0
            for b in seeds:
                dist[b] = 0.0
                origin[b] = b
                heapq.heappush(heap, (0.0, ctr, b))
                ctr += 1
            pair_maps = {}
            while heap:
                d, _, u = heapq.heappop(heap)
                if d != dist.get(u, float("inf")):
                    continue
                child_u = node_to_child.get(u)
                if child_u is not None and child_u != child:
                    key = (child, child_u)
                    pair_maps.setdefault(key, {})
                    k2 = (origin[u], u)
                    cur = pair_maps[key].get(k2)
                    if cur is None or d < cur:
                        pair_maps[key][k2] = d
                for v, data in IG[u].items():
                    nd = d + float(data.get("weight", 1.0))
                    if nd < dist.get(v, float("inf")):
                        dist[v] = nd
                        origin[v] = origin[u]
                        heapq.heappush(heap, (nd, ctr, v))
                        ctr += 1
            for key, mapping in pair_maps.items():
                per_pair.setdefault(key, {})
                for k2, val in mapping.items():
                    cur = per_pair[key].get(k2)
                    if cur is None or val < cur:
                        per_pair[key][k2] = val
        trimmed = {}
        for key, mapping in per_pair.items():
            entries = sorted(((dist, start, end) for (start, end), dist in mapping.items()), key=lambda x: x[0])
            if pair_topk and pair_topk > 0:
                entries = entries[:pair_topk]
            trimmed[key] = entries
        if trimmed:
            pair_paths[parent] = trimmed
    return pair_paths


class HierarchicalDistanceResolver:
    def __init__(
        self,
        G: nx.Graph,
        cluster_tree: dict,
        node_to_leaf: dict,
        pll_labels: dict,
        rep_graphs: dict,
        borders: dict,
        G_nb_all: dict,
        pair_paths: dict,
    ):
        self.G = G
        self.cluster_tree = cluster_tree
        self.node_to_leaf = node_to_leaf
        self.pll_obj = build_pll_objects(pll_labels)
        self.rep_graphs = rep_graphs
        self.borders = borders
        self.G_nb_all = G_nb_all
        self.pair_paths = pair_paths
        self.leaf_border_index = {
            leaf: {b: i for i, b in enumerate(border_list)}
            for leaf, border_list in borders.items()
        }
        self.node_border_cache = {}
        self.cluster_pair_cache = {}
        self.direct_cache = {}

    def _distance_within_leaf(self, leaf, u, v):
        pll = self.pll_obj.get(leaf)
        if pll is None:
            return float("inf")
        return float(pll.query(u, v))

    def _direct_distance_in_cluster(self, cluster, u, v):
        key = (cluster, u, v) if str(u) <= str(v) else (cluster, v, u)
        if key in self.direct_cache:
            return self.direct_cache[key]
        nodes = self.cluster_tree[cluster]["nodes"]
        if not isinstance(nodes, set):
            nodes = set(nodes)
        allowed = nodes
        dist = {u: 0.0}
        heap = [(0.0, u)]
        while heap:
            d, x = heapq.heappop(heap)
            if d != dist.get(x, float("inf")):
                continue
            if x == v:
                self.direct_cache[key] = d
                return d
            for y, data in self.G[x].items():
                if y not in allowed:
                    continue
                nd = d + float(data.get("weight", 1.0))
                if nd < dist.get(y, float("inf")):
                    dist[y] = nd
                    heapq.heappush(heap, (nd, y))
        self.direct_cache[key] = float("inf")
        return float("inf")

    def distance_node_to_border(self, node, border, cluster):
        key = (node, border, cluster)
        if key in self.node_border_cache:
            return self.node_border_cache[key]
        meta = self.cluster_tree[cluster]
        if meta["level"] == 0:
            idx = self.leaf_border_index.get(cluster, {}).get(border)
            arr = self.G_nb_all.get(cluster, {}).get(node)
            if idx is None or arr is None:
                val = float("inf")
            else:
                val = float(arr[idx])
            self.node_border_cache[key] = val
            return val
        node_to_child = meta.get("node_to_child") or {}
        child_node = node_to_child.get(node)
        child_border = node_to_child.get(border)
        if child_node is None or child_border is None:
            val = self._direct_distance_in_cluster(cluster, node, border)
            self.node_border_cache[key] = val
            return val
        if child_node == child_border:
            val = self.distance_node_to_border(node, border, child_node)
            self.node_border_cache[key] = val
            return val
        best = float("inf")
        pair_entries = self.pair_paths.get(cluster, {}).get((child_node, child_border), [])
        for dist_cross, start_b, end_b in pair_entries:
            d1 = self.distance_node_to_border(node, start_b, child_node)
            d2 = self.distance_node_to_border(end_b, border, child_border)
            total = d1 + dist_cross + d2
            if total < best:
                best = total
        if not pair_entries:
            direct = self._direct_distance_in_cluster(cluster, node, border)
            if direct < best:
                best = direct
        self.node_border_cache[key] = best
        return best

    def distance_between_nodes_in_cluster(self, cluster, u, v):
        if u == v:
            return 0.0
        key = (cluster, u, v) if str(u) <= str(v) else (cluster, v, u)
        if key in self.cluster_pair_cache:
            return self.cluster_pair_cache[key]
        meta = self.cluster_tree[cluster]
        if meta["level"] == 0:
            val = self._distance_within_leaf(cluster, u, v)
            self.cluster_pair_cache[key] = val
            return val
        node_to_child = meta.get("node_to_child") or {}
        child_u = node_to_child.get(u)
        child_v = node_to_child.get(v)
        if child_u is None or child_v is None:
            val = self._direct_distance_in_cluster(cluster, u, v)
            self.cluster_pair_cache[key] = val
            return val
        if child_u == child_v:
            val = self.distance_between_nodes_in_cluster(child_u, u, v)
            self.cluster_pair_cache[key] = val
            return val
        best = float("inf")
        pair_entries = self.pair_paths.get(cluster, {}).get((child_u, child_v), [])
        for dist_cross, start_b, end_b in pair_entries:
            d1 = self.distance_node_to_border(u, start_b, child_u)
            d2 = self.distance_node_to_border(v, end_b, child_v)
            total = d1 + dist_cross + d2
            if total < best:
                best = total
        if not pair_entries:
            direct = self._direct_distance_in_cluster(cluster, u, v)
            if direct < best:
                best = direct
        self.cluster_pair_cache[key] = best
        return best

    def query(self, u, v):
        if u == v:
            return 0.0
        leaf_u = self.node_to_leaf.get(u)
        leaf_v = self.node_to_leaf.get(v)
        if leaf_u is None or leaf_v is None:
            return float("inf")
        if leaf_u == leaf_v:
            return self._distance_within_leaf(leaf_u, u, v)
        L, Au, Bv, _, _ = find_lca_and_children(self.cluster_tree, self.node_to_leaf, u, v)
        if L is None:
            return float("inf")
        return self.distance_between_nodes_in_cluster(L, u, v)
# ---------- 查询辅助 ----------
def build_pll_objects(pll_labels):
    return {leaf: PrunedPLLIndex.from_labels(labels) for leaf, labels in pll_labels.items()}

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


# ---------- 查询（top-k 搜索） ----------
def query_distance_with_resolver(resolver: HierarchicalDistanceResolver, u, v) -> float:
    return resolver.query(u, v)


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


def evaluate_resolver_lookup(resolver, gt, preprocessing_time):
    correct = total_eval = 0
    err = 0.0
    inf_pred = 0
    t0 = time.perf_counter()
    for u, v, d in gt:
        pred = resolver.query(u, v)
        if pred == float("inf"):
            inf_pred += 1
        if pred == d:
            correct += 1
        if (pred != float("inf")) and (d != float("inf")):
            err += abs(pred - d)
            total_eval += 1
    tQ = time.perf_counter() - t0
    rows = [[
        "HierL-E (topk-search)",
        tQ, len(gt), correct,
        (err / total_eval if total_eval > 0 else float("inf")),
        inf_pred, preprocessing_time,
    ]]
    return pd.DataFrame(rows, columns=[
        "method", "query_time_sec", "samples", "exact_matches", "mae", "inf_pred", "preprocessing_time"
    ])


# ---------- printing helpers ----------
def print_parent_children_counts(cluster_tree):
    """
    输出每个 parent 的直接 children 数量（不是叶子数量、不是子树大小）。
    """
    parents = [(cid, meta["children"]) for cid, meta in cluster_tree.items()
               if isinstance(meta.get("children"), list) and len(meta["children"]) > 0]
    parents_sorted = sorted(parents, key=lambda x: -len(x[1]))
    print("\n[Parent → #children]")
    for cid, ch_list in parents_sorted:
        print(f"  [P] {_cid_str(cid)}  children={len(ch_list)}")


def print_hierarchy_overview(levels_map: dict):
    print("\n[Hierarchy Levels Overview]")
    for lvl in sorted(levels_map.keys()):
        nodes = levels_map[lvl]
        preview = ", ".join(_cid_str(cid) for cid in nodes[:5])
        if len(nodes) > 5:
            preview += ", ..."
        print(f"  level={lvl:<3} count={len(nodes):<5} sample=[{preview}]")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="Cora", choices=["Cora", "CiteSeer", "Pubmed"])
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--kg_file", type=str, default=None)

    # 超图分层（NEW）
    ap.add_argument("--base_resolution", type=float, default=0.3, help="底层在原图上的聚类分辨率（得到level=0叶簇）")
    ap.add_argument("--base_max_cluster_size", type=int, default=500)
    ap.add_argument("--super_resolutions", type=str, default="0.8,0.6,0.45,0.3,0.2",
                    help="在超图上的逐层分辨率列表（逗号分隔），层序向上递减或保持")
    ap.add_argument("--super_max_cluster_size", type=int, default=50,
                    help="超图分层时 Leiden 的最大簇规模限制")
    ap.add_argument("--supernode_threshold", type=int, default=50,
                    help="若当前层超点数量大于该阈值则继续向上聚合")
    ap.add_argument("--hl_seed", type=int, default=42)

    ap.add_argument("--max_workers", type=int, default=None)

    ap.add_argument("--pair_topk", type=int, default=3,
                    help="父层中每对子簇保留的跨簇最短通路数量")

    # 评测
    ap.add_argument("--eval_pairs", type=int, default=500)
    ap.add_argument("--save_eval_set", type=str, default=None)
    ap.add_argument("--load_eval_set", type=str, default=None)
    args = ap.parse_args()

    # 0) 加载图
    if args.kg_file:
        G = load_wn18_graph_aggregated(args.kg_file)
        src_name = f"WN18({os.path.basename(args.kg_file)})-aggregated"
    else:
        G = load_planetoid_graph(args.dataset, root=args.data_root)
        src_name = args.dataset
    print(f"[INFO] Graph: {src_name}, |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

    # 1) 超图分层骨架（NEW）
    tALL0 = time.time()
    super_res_list = [float(x) for x in args.super_resolutions.split(",") if x.strip()]
    cluster_tree, node_to_leaf, hierarchy_levels = build_hierarchical_supergraph_tree(
        G,
        base_resolution=args.base_resolution,
        base_max_cluster_size=args.base_max_cluster_size,
        super_resolutions=tuple(super_res_list),
        super_max_cluster_size=args.super_max_cluster_size,
        supernode_threshold=args.supernode_threshold,
        seed=args.hl_seed
    )

    print_hierarchy_overview(hierarchy_levels)
    # 输出：每个 parent 的直接 children 数量
    print_parent_children_counts(cluster_tree)

    # === 输出：节点最多的 Top-20 叶子 ===
    leaves = [(cid, len(meta["nodes"])) for cid, meta in cluster_tree.items() if meta["level"] == 0]
    leaves_sorted = sorted(leaves, key=lambda x: -x[1])[:20]
    print("\n[Top-20 Leaves by |V|]")
    for cid, sz in leaves_sorted:
        print(f"  [L] {_cid_str(cid)}  (|V|={sz})")

    # 2) 叶簇（并行 PLL + 完全图 rep_graph）
    borders, pll_labels, G_nb_all, rep_graphs, _ = build_leaves_tables(
        G, cluster_tree, max_workers=args.max_workers
    )

    # 3) 父层 index_graph（完全闭包）
    index_graphs_lvl, idx_stats = build_index_graphs_for_parents(
        G, cluster_tree, borders, rep_graphs,
        parents_subset=None,
    )

    print("\n[Parent Index/Border Summary]")
    parents = [cid for cid, meta in cluster_tree.items() if meta.get("children") and len(meta["children"]) > 1]
    for p in parents:
        st = idx_stats.get(p, {})
        print(f"  [P] {_cid_str(p)}  index_nodes={st.get('index_nodes',0)}, parent_border={st.get('parent_border_count',0)}")

    # 4) 父层 child-pair top-k 路径
    pair_paths = build_parent_pair_topk(
        cluster_tree, index_graphs_lvl, pair_topk=args.pair_topk
    )

    preprocessing_time = time.time() - tALL0

    resolver = HierarchicalDistanceResolver(
        G, cluster_tree, node_to_leaf,
        pll_labels, rep_graphs, borders, G_nb_all,
        pair_paths,
    )

    # 评测集（固定只在 LCC 内采样）
    if args.load_eval_set and os.path.isfile(args.load_eval_set):
        gt = load_eval_set_csv(args.load_eval_set, G, max_rows=args.eval_pairs)
        if not gt:
            raise RuntimeError(f"Loaded eval set has 0 usable rows: {args.load_eval_set}")
    else:
        pairs = sample_entity_pairs(G, n_pairs=args.eval_pairs, in_lcc=True, rng_seed=42)
        print(f"\n[INFO] Sampling done. computing ground-truth distances for {len(pairs)} pairs ...")
        gt = compute_gt_distances(G, pairs)
        if args.save_eval_set:
            save_eval_set_csv(args.save_eval_set, gt)

    df_eval = evaluate_resolver_lookup(resolver, gt, preprocessing_time)
    print("\n=== Hierarchical Query Evaluation (Top-k resolver) ===")
    print(df_eval.to_string(index=False))
    print(f"\n[Summary] Preprocessing={preprocessing_time:.3f}s, parents_with_pairs={len(pair_paths)}")
    if idx_stats:
        total_nodes = sum(s.get("index_nodes", 0) for s in idx_stats.values())
        total_edges = sum(s.get("index_edges", 0) for s in idx_stats.values())
        total_borders = sum(s.get("parent_border_count", 0) for s in idx_stats.values())
        print(f"[IndexGraph Stats] nodes={total_nodes}, edges={total_edges}, parent_border={total_borders}")

    print("\n[OK] Finished: hierarchical top-k resolver pipeline.")


if __name__ == "__main__":
    main()
