#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HierL-E (GPU 版, 关键算子上 GPU 化)
- Step2: 叶内 G_nb 用 cuGraph.sssp 批量生成（列=边界点；行为叶内节点）——GPU
- Step7'/7.9: L2B / L2C 的列块最小化（min-plus）用 CuPy 向量化 ——GPU
- 其它结构（index_graph/list_indexpath/AB/SAME/查询）沿用原思路，接口保持兼容
- 仍“严格查表查询”：查询阶段不跑Dijkstra
"""

from __future__ import annotations
import argparse
import os
import time
import heapq
import random
from collections import defaultdict, deque

import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm

# GPU 相关
import cupy as cp
import cudf
import cugraph
import rmm

try:
    from torch_geometric.datasets import Planetoid
except Exception:
    Planetoid = None


# -------------------- 小工具 --------------------
def node_sort_key(n):
    # 为了对混合 id（字符串/整数）的节点有稳定序
    return (isinstance(n, str), str(n))

def ensure_int_index(nodes):
    """把任意可哈希节点集合映射为 [0..N-1] 的连续整数 id。返回 (id_map, rev)"""
    uniq = sorted(set(nodes), key=node_sort_key)
    id_map = {u: i for i, u in enumerate(uniq)}
    rev = {i: u for i, u in enumerate(uniq)}
    return id_map, rev

def df_edges_from_pairs(pairs, weights=None):
    """pairs: list[(u,v)](无向); weights: 可选与 pairs 等长的 list/ndarray"""
    src = []
    dst = []
    wts = []
    for i, (u, v) in enumerate(pairs):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        src.append(a); dst.append(b)
        wts.append(1.0 if weights is None else float(weights[i]))
    df = pd.DataFrame({"src": src, "dst": dst, "weight": wts})
    return df


# -------------------- 数据加载 --------------------
def load_planetoid_edges(name="Pubmed", root=None) -> pd.DataFrame:
    if Planetoid is None:
        raise RuntimeError("未安装 torch_geometric，无法加载 Planetoid；请使用 --kg_file 加载 KG。")
    root = root or os.path.abspath(f"./data/{name}")
    dataset = Planetoid(root=root, name=name)
    data = dataset[0]
    eidx = data.edge_index.numpy()
    pairs = set()
    for u, v in zip(eidx[0], eidx[1]):
        if int(u) == int(v):
            continue
        a, b = (int(u), int(v))
        if a > b: a, b = b, a
        pairs.add((a, b))
    df = pd.DataFrame(list(pairs), columns=["src", "dst"])
    df["weight"] = 1.0
    return df

def load_wn18_edges_aggregated(path: str) -> tuple[pd.DataFrame, dict, dict]:
    """将 WN18 三元组汇总为无向边，关系取最小权（默认1.0）。映射实体为 int id。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"WN18 file not found: {path}")
    pair2w = {}
    nodes = set()
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        try:
            _ = int(first.strip())
        except Exception:
            parts = first.strip().split()
            if len(parts) >= 2:
                h, t = parts[0], parts[1]
                nodes.add(h); nodes.add(t)
                a, b = (h, t) if str(h) <= str(t) else (t, h)
                pair2w[(a, b)] = min(1.0, pair2w.get((a, b), float("inf")))
        for line in f:
            sp = line.strip().split()
            if len(sp) < 2: continue
            h, t = sp[0], sp[1]
            nodes.add(h); nodes.add(t)
            a, b = (h, t) if str(h) <= str(t) else (t, h)
            pair2w[(a, b)] = min(1.0, pair2w.get((a, b), float("inf")))
    ent2id, id2ent = ensure_int_index(nodes)
    src, dst, w = [], [], []
    for (a, b), ww in pair2w.items():
        src.append(ent2id[a]); dst.append(ent2id[b]); w.append(float(ww))
    df = pd.DataFrame({"src": src, "dst": dst, "weight": w})
    return df, ent2id, id2ent


# -------------------- Step1: 仅构叶（层次化 Leiden, 简化） --------------------
def build_leaves_by_leiden_gpu(edges_df: pd.DataFrame, resolution=0.3, max_cluster_size=1200, seed=42):
    """
    仅两层：level=-1 为 CC， level=0 为叶（递归切分，直到 size<=阈值）。
    返回：cluster_tree, node_to_leaf
    """
    # cuDF & cuGraph
    gdf = cudf.DataFrame.from_pandas(edges_df[["src", "dst", "weight"]])
    G = cugraph.Graph(directed=False)
    # 重要：src/dst 必须是整数；edges_df 已经是 int 映射并连续
    G.from_cudf_edgelist(gdf, source="src", destination="dst", edge_attr="weight", renumber=False)

    # 连通分量
    cc = cugraph.connected_components(G)  # ['vertex','component']
    cc_pd = cc.to_pandas()
    comps = defaultdict(list)
    for v, c in zip(cc_pd["vertex"].to_numpy(), cc_pd["component"].to_numpy()):
        comps[int(c)].append(int(v))

    cluster_tree = {}
    node_to_leaf = {}

    # -1 层：每个 CC 一个 parent
    for cc_id, vlist in comps.items():
        parent = ("cc", int(cc_id))
        cluster_tree[parent] = dict(level=-1, parent=None, children=[], nodes=set(vlist))

        # 队列：对超过阈值的集合递归跑 leiden
        stack = [vlist]
        leaf_idx = 0
        while stack:
            verts = stack.pop()
            if len(verts) <= max_cluster_size:
                leaf_cid = ("leaf", int(cc_id), int(leaf_idx)); leaf_idx += 1
                cluster_tree[leaf_cid] = dict(level=0, parent=parent, children=[], nodes=set(verts))
                cluster_tree[parent]["children"].append(leaf_cid)
                for n in verts: node_to_leaf[int(n)] = leaf_cid
                continue

            # 在子图上跑 Leiden
            sub = cudf.Series(verts, dtype="int32")
            SG = cugraph.subgraph(G, sub)  # returns (Graph, renumber_map)? 新 API 直接 Graph
            parts_df, _ = cugraph.leiden(SG, resolution=resolution, random_state=seed)  # ['vertex','partition']
            # 注意：renumber=False；subgraph 保持外部 id
            parts = parts_df.to_pandas()
            bins = defaultdict(list)
            for v, p in zip(parts["vertex"].to_numpy(), parts["partition"].to_numpy()):
                bins[int(p)].append(int(v))
            # 对每个分簇，如果仍超过阈值则继续递归，否则成为叶
            for _, sub_verts in bins.items():
                if len(sub_verts) <= max_cluster_size:
                    leaf_cid = ("leaf", int(cc_id), int(leaf_idx)); leaf_idx += 1
                    cluster_tree[leaf_cid] = dict(level=0, parent=parent, children=[], nodes=set(sub_verts))
                    cluster_tree[parent]["children"].append(leaf_cid)
                    for n in sub_verts: node_to_leaf[int(n)] = leaf_cid
                else:
                    stack.append(sub_verts)

    return cluster_tree, node_to_leaf, G, gdf


# -------------------- Step2: 叶内 G_nb（GPU SSSP）与 rep_graph（kNN） --------------------
def build_leaf_tables_gpu(
    full_edges_gdf: cudf.DataFrame,
    cluster_tree: dict,
    leaf_rg_k: int = 8,
):
    """
    对每个叶：
      - 取叶内边 (src/dst ∈ leaf.nodes)
      - 找 leaf border（与外部有边的点）
      - 对 border 中每个 b 跑一次 SSSP 得到 dist[:, b]，堆成 G_nb (CuPy)
      - 用 G_nb 在 border 上构 kNN rep_graph（networkx，以便后续逻辑不变）
    返回：
      borders: {leaf: [border_nodes(sorted)]}
      leaf_nodes: {leaf: [all_nodes(sorted)]}
      G_nb_all: {leaf: CuPy (N_leaf x B_leaf)}
      rep_graphs: {leaf: nx.Graph (on border nodes, weight=dist)}
    """
    # 全图边（Pandas 视图有时方便）
    full_edges_pd = full_edges_gdf.to_pandas()
    borders = {}
    leaf_nodes = {}
    G_nb_all = {}
    rep_graphs = {}

    # 预备：加速 membership 测试
    for cid, meta in tqdm(cluster_tree.items(), desc="Step2: leaves (GPU SSSP)", unit="leaf"):
        if meta["level"] != 0:
            continue
        nodes_sorted = sorted(meta["nodes"])
        leaf_nodes[cid] = nodes_sorted
        node_set = set(nodes_sorted)

        # 叶内边 & 跨边（CPU 端用 pandas 过滤再丢回 GPU，简单稳定）
        mask_src_in = full_edges_pd["src"].isin(node_set)
        mask_dst_in = full_edges_pd["dst"].isin(node_set)
        edges_in_pd = full_edges_pd[mask_src_in & mask_dst_in]
        edges_cross_pd = full_edges_pd[(mask_src_in & (~mask_dst_in)) | ((~mask_src_in) & mask_dst_in)]
        border_set = set(edges_cross_pd["src"].tolist()) & node_set
        border_set |= (set(edges_cross_pd["dst"].tolist()) & node_set)
        border_sorted = sorted(border_set)
        borders[cid] = border_sorted

        # 若没有边界或叶特别小，直接给空
        if len(nodes_sorted) == 0 or len(border_sorted) == 0 or len(edges_in_pd) == 0:
            G_nb_all[cid] = cp.zeros((len(nodes_sorted), 0), dtype=cp.float32)
            rep_graphs[cid] = nx.Graph()
            rep_graphs[cid].add_nodes_from(border_sorted)
            continue

        # 构建叶子子图（GPU）
        edges_in_gdf = cudf.DataFrame.from_pandas(edges_in_pd)
        G_leaf = cugraph.Graph(directed=False)
        G_leaf.from_cudf_edgelist(edges_in_gdf, source="src", destination="dst", edge_attr="weight", renumber=False)

        # 为了把 SSSP 结果按 leaf 节点顺序堆叠，准备一个 vertex->row 的映射
        row_index = {v: i for i, v in enumerate(nodes_sorted)}

        # 预分配 CuPy 矩阵：N x B
        N, B = len(nodes_sorted), len(border_sorted)
        G_nb = cp.full((N, B), cp.inf, dtype=cp.float32)

        # 对每个边界点 b，跑一次 SSSP（GPU）
        for j, b in enumerate(border_sorted):
            sssp_df = cugraph.sssp(G_leaf, source=int(b))  # ['vertex','distance','predecessor']
            pdf = sssp_df.to_pandas()
            for v, d in zip(pdf["vertex"].to_numpy(), pdf["distance"].to_numpy()):
                i = row_index.get(int(v), None)
                if i is not None:
                    G_nb[i, j] = np.float32(d if np.isfinite(d) else np.inf)

        G_nb_all[cid] = G_nb

        # 构建边界点之间的 kNN（weight 即 leaf 内最短路）——使用 G_nb 的“列向量”
        # 注意：距离(bi->bj) = G_nb[row_of(bj), col_of(bi)]
        border_pos = {b: i for i, b in enumerate(border_sorted)}
        RG = nx.Graph()
        RG.add_nodes_from(border_sorted)
        if len(border_sorted) > 1:
            k = leaf_rg_k if leaf_rg_k > 0 else len(border_sorted) - 1
            # 预先把“边界节点的行索引”备好
            border_rows = np.array([row_index[b] for b in border_sorted], dtype=np.int64)
            for bi in border_sorted:
                j = border_pos[bi]              # 列（源=bi）
                dvec = cp.asnumpy(G_nb[border_rows, j])  # 对其他边界的距离
                dvec[border_pos[bi]] = np.inf
                if k < len(border_sorted) - 1:
                    idx = np.argpartition(dvec, k)[:k]
                else:
                    idx = np.where(np.isfinite(dvec))[0]
                for t in idx:
                    bj = border_sorted[int(t)]
                    dij = float(dvec[int(t)])
                    if not np.isfinite(dij):
                        continue
                    if RG.has_edge(bi, bj):
                        if dij < RG[bi][bj].get("weight", float("inf")):
                            RG[bi][bj]["weight"] = dij
                    else:
                        RG.add_edge(bi, bj, weight=dij)
        rep_graphs[cid] = RG

    return borders, leaf_nodes, G_nb_all, rep_graphs


# -------------------- Step3: 父层 index_graph（仍按原规则；使用 NX） --------------------
def build_index_graphs_for_parents(
    full_G_nx: nx.Graph,
    cluster_tree: dict,
    rep_graphs: dict,
    ig_intra_knn: int = 8,
):
    """
    与原思路一致：
      - 节点：各子簇的“网关”（与其它子簇有原始跨边的边界点）
      - intra：在每个子簇的 rep_graph 上对网关做 kNN（距离=rep_graph 最短路）
      - inter：保留原始图上的跨子簇边（只保留网关）
    """
    index_graphs = {}
    stats = {}

    parents = [cid for cid, meta in cluster_tree.items() if meta.get("children") and len(meta["children"]) > 1]
    # 预计算：全体 rep_graph 的节点集合（便于判定网关）
    leaf_border_sets = {leaf: set(rg.nodes()) for leaf, rg in rep_graphs.items()}

    for parent in tqdm(parents, desc="Step3: index_graph (NX)", unit="parent"):
        children = cluster_tree[parent]["children"]

        # 先把“跨子簇的边界点”标成网关
        node_to_child = {}
        for ch in children:
            for x in leaf_border_sets.get(ch, []):
                node_to_child[x] = ch
        all_rep = set(node_to_child.keys())

        gateway_by_child = {ch: set() for ch in children}
        for u, v, data in full_G_nx.edges(data=True):
            if (u in all_rep) and (v in all_rep):
                cu, cv = node_to_child[u], node_to_child[v]
                if cu is not None and cv is not None and cu != cv:
                    gateway_by_child[cu].add(u)
                    gateway_by_child[cv].add(v)

        IG = nx.Graph()
        all_gate = set().union(*gateway_by_child.values())
        IG.add_nodes_from(all_gate)

        # intra：在每个子簇的 rep_graph 上做 kNN（以 rep_graph 的最短路为距离）
        for ch, gates in gateway_by_child.items():
            if not gates:
                continue
            RG = rep_graphs.get(ch)
            if RG is None or RG.number_of_nodes() == 0:
                continue
            # 为每个 gate 执行一次 Dijkstra（RG 很小，NX 足够快；如需可换成 cugraph + sssp）
            for src in gates:
                dist = nx.single_source_dijkstra_path_length(RG, src, weight="weight")
                # 仅取在 gates 中的项
                items = [(t, d) for t, d in dist.items() if (t in gates and t != src)]
                if ig_intra_knn > 0 and len(items) > ig_intra_knn:
                    items = heapq.nsmallest(ig_intra_knn, items, key=lambda kv: kv[1])
                for tgt, dval in items:
                    if not np.isfinite(dval):
                        continue
                    if IG.has_edge(src, tgt):
                        if dval < IG[src][tgt].get("weight", float("inf")):
                            IG[src][tgt]["weight"] = float(dval)
                    else:
                        IG.add_edge(src, tgt, weight=float(dval))

        # inter：原始跨子簇边（只保留网关）
        for u, v, data in full_G_nx.edges(data=True):
            if (u in all_gate) and (v in all_gate):
                cu, cv = node_to_child[u], node_to_child[v]
                if cu is None or cv is None or cu == cv:
                    continue
                w = float(data.get("weight", 1.0))
                if IG.has_edge(u, v):
                    if w < IG[u][v].get("weight", float("inf")):
                        IG[u][v]["weight"] = w
                else:
                    IG.add_edge(u, v, weight=w)

        index_graphs[parent] = IG
        stats[parent] = dict(nodes=IG.number_of_nodes(), edges=IG.number_of_edges(),
                             gateways=sum(len(s) for s in gateway_by_child.values()))
    return index_graphs, stats


# -------------------- Step4: list_indexpath（父层多源，按原 CPU 逻辑） --------------------
def _make_adj_from_graph_nx(G: nx.Graph):
    return {u: [(v, float(d.get("weight", 1.0))) for v, d in G[u].items()] for u in G.nodes()}

def _multi_source_dijkstra_index_worker_cpu(parent, A, adj, child_to_gates):
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

def build_list_indexpath_for_parents_cpu(cluster_tree, index_graphs, rep_graphs, parents_subset=None):
    L = defaultdict(lambda: defaultdict(dict))
    if parents_subset is None:
        parents = [cid for cid, meta in cluster_tree.items() if meta.get("children") and len(meta["children"]) > 1]
    else:
        parents = [p for p in parents_subset if cluster_tree.get(p, {}).get("children")]
    for parent in tqdm(parents, desc="Step4: list_indexpath (CPU)", unit="parent"):
        IG = index_graphs.get(parent)
        if IG is None or IG.number_of_nodes() == 0:
            continue
        child_to_gates = {}
        for ch in cluster_tree[parent]["children"]:
            RG = rep_graphs.get(ch)
            nodes_set = set(RG.nodes()) if RG is not None else set()
            gates = [x for x in nodes_set if x in IG]
            child_to_gates[ch] = set(gates)
        adj = _make_adj_from_graph_nx(IG)
        for A, seeds in child_to_gates.items():
            if not seeds:
                continue
            parent_, A_, resA = _multi_source_dijkstra_index_worker_cpu(parent, A, adj, child_to_gates)
            L[parent_][A_] = resA
    return L


# -------------------- Step7: AB 表 --------------------
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


# -------------------- Step7': SAME (per-leaf CSC) + L2B/L2C（GPU） --------------------
def collect_leaves_under(cluster_tree, child):
    leaves = []
    dq = deque([child])
    while dq:
        c = dq.popleft()
        if cluster_tree[c]["level"] == 0:
            leaves.append(c)
        else:
            dq.extend(cluster_tree[c]["children"])
    return leaves

def build_SAME_tables_gpu(cluster_tree, borders, rep_graphs, leaf_nodes, G_nb_all):
    """
    SAME[L][A]:
      - cols_nodes = A 子树“代表图”的节点（边界）
      - 对每个叶 leaf: 提取该叶所有节点到这些 cols 的最短路（来自 G_nb 的列拼后求最小）
      - 存为 per-leaf 的简易 CSC（col_ptr,row_idx,vals），但这里先保留“致密列块”便于后续 L2 取列
    """
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
            leaves = collect_leaves_under(cluster_tree, A)
            leaf_blocks = {}
            # 构建：对每个 leaf，收集该 leaf 的节点对“列=cols_nodes”的距离矩阵（由 G_nb 列拼成）
            for leaf in leaves:
                blist = borders.get(leaf, [])
                if len(blist) == 0:
                    leaf_blocks[leaf] = dict(mat=cp.zeros((len(leaf_nodes[leaf]), 0), dtype=cp.float32))
                    continue
                # G_nb_all[leaf] 形状 (N_leaf x B_leaf)，列是“该叶的边界列表”
                G_nb = G_nb_all[leaf]  # CuPy
                leaf_b_index = {b: i for i, b in enumerate(blist)}
                # 选择 A 的 cols_nodes 中那些出现在当前 leaf 的边界中的列，按 col 顺序堆叠
                pick_j = [leaf_b_index[b] for b in cols_nodes if b in leaf_b_index]
                if len(pick_j) == 0:
                    leaf_blocks[leaf] = dict(mat=cp.full((G_nb.shape[0], len(cols_nodes)), cp.inf, dtype=cp.float32))
                else:
                    sub = G_nb[:, cp.asarray(pick_j, dtype=cp.int64)]
                    # 如果 cols_nodes 有的列在本叶缺失：补为 +inf 列，并按 cols_nodes 对齐
                    if len(pick_j) < len(cols_nodes):
                        # 先构空矩阵再 scatter
                        out = cp.full((G_nb.shape[0], len(cols_nodes)), cp.inf, dtype=cp.float32)
                        # 建立列位置映射
                        pos = {b: i for i, b in enumerate(cols_nodes)}
                        for k, b in enumerate(cols_nodes):
                            if b in leaf_b_index:
                                out[:, pos[b]] = sub[:, cp.asarray([k for k2, b2 in enumerate(cols_nodes) if b2==b and b in leaf_b_index][0])]
                        leaf_blocks[leaf] = dict(mat=out)
                    else:
                        leaf_blocks[leaf] = dict(mat=sub)  # 已与 cols_nodes 同序
            SAME[L][A] = dict(cols_nodes=cols_nodes, col_index=col_index, leaf_blocks=leaf_blocks)
    return SAME

def _compute_cols_block_for_leaf_gpu(leaf_block_mat: cp.ndarray, cols_pick_pos: list[int], use_dtype):
    """从已缓存的 leaf_block 矩阵（N x |cols_nodes|）里直接 gather 指定列集，返回 (N x K)"""
    if leaf_block_mat.size == 0 or len(cols_pick_pos) == 0:
        return cp.zeros((leaf_block_mat.shape[0], 0), dtype=use_dtype)
    idx = cp.asarray(cols_pick_pos, dtype=cp.int64)
    sub = leaf_block_mat[:, idx]
    return sub.astype(use_dtype, copy=False)

def build_leaf2pair_tables_mixed_gpu(
    cluster_tree, borders, leaf_nodes, SAME_tables, AB_tables,
    dtype: str = "fp16", mode: str = "auto", union_ratio: float = 1.5
):
    """
    用 GPU（CuPy）完成 L2B/L2C 的列块切片与堆叠。
    SAME_tables[L][A]["leaf_blocks"][leaf]["mat"] 为 (N_leaf x |cols_nodes_of_A|) 的 CuPy 矩阵。
    """
    use_dtype = cp.float16 if dtype == "fp16" else cp.float32
    L2B = defaultdict(dict)
    L2C = defaultdict(dict)
    leaf_row_index = {leaf: {n: i for i, n in enumerate(leaf_nodes[leaf])} for leaf in leaf_nodes.keys()}

    for L, pairs in AB_tables.items():
        same_for_L = SAME_tables.get(L, {})
        if not same_for_L:
            continue

        # 收集 A/B 出现频率，决定是否建 union 缓存
        left_children = defaultdict(list)
        right_children = defaultdict(list)
        for (A, B) in pairs.keys():
            left_children[A].append((A, B))
            right_children[B].append((A, B))

        def decide(child, is_left=True):
            if mode in ("union", "pair"):
                return mode
            ab_list = (left_children if is_left else right_children)[child]
            if not ab_list: return "pair"
            if is_left:
                union_set, total = set(), 0
                for (A0, B0) in ab_list:
                    total += len(pairs[(A0, B0)]["b_nodes"])
                    union_set.update(pairs[(A0, B0)]["b_nodes"])
                r = total / max(1, len(union_set))
            else:
                union_set, total = set(), 0
                for (A0, B0) in ab_list:
                    total += len(pairs[(A0, B0)]["c_nodes"])
                    union_set.update(pairs[(A0, B0)]["c_nodes"])
                r = total / max(1, len(union_set))
            return "union" if r >= union_ratio else "pair"

        # per-A union
        A_union_cols = {}
        A_union_cache = {}
        for A in left_children.keys():
            if decide(A, True) != "union":
                continue
            sameA = same_for_L.get(A)
            if sameA is None:
                continue
            cols_union = []
            seen = set()
            for (A0, B0) in left_children[A]:
                for b in pairs[(A0, B0)]["b_nodes"]:
                    if b not in seen:
                        seen.add(b); cols_union.append(b)
            if len(cols_union) == 0:
                continue
            A_union_cols[A] = cols_union
            # 填缓存：对 A 的每个 leaf，直接从 leaf_blocks 中 gather union 列
            leaf_blocks = sameA["leaf_blocks"]
            posA = {b: i for i, b in enumerate(sameA["cols_nodes"])}
            pick_pos = [posA[b] for b in cols_union if b in posA]
            for leaf, blk in leaf_blocks.items():
                mat = _compute_cols_block_for_leaf_gpu(blk["mat"], pick_pos, use_dtype)
                A_union_cache[(A, leaf)] = (leaf_nodes[leaf], mat)

        # per-B union
        B_union_cols = {}
        B_union_cache = {}
        for B in right_children.keys():
            if decide(B, False) != "union":
                continue
            sameB = same_for_L.get(B)
            if sameB is None:
                continue
            cols_union = []
            seen = set()
            for (A0, B0) in right_children[B]:
                for c in pairs[(A0, B0)]["c_nodes"]:
                    if c not in seen:
                        seen.add(c); cols_union.append(c)
            if len(cols_union) == 0:
                continue
            B_union_cols[B] = cols_union
            leaf_blocks = sameB["leaf_blocks"]
            posB = {c: i for i, c in enumerate(sameB["cols_nodes"])}
            pick_pos = [posB[c] for c in cols_union if c in posB]
            for leaf, blk in leaf_blocks.items():
                mat = _compute_cols_block_for_leaf_gpu(blk["mat"], pick_pos, use_dtype)
                B_union_cache[(B, leaf)] = (leaf_nodes[leaf], mat)

        # 逐对 (A,B) 产出 L2B/L2C
        for (A, B), AB in pairs.items():
            # L2B
            sameA = same_for_L.get(A)
            if sameA is not None:
                b_nodes = AB["b_nodes"]
                leaves_A = collect_leaves_under(cluster_tree, A)
                modeA = decide(A, True)
                if modeA == "union" and A in A_union_cols:
                    union_cols = A_union_cols[A]; pos = {x: i for i, x in enumerate(union_cols)}
                    gather = [pos[b] for b in b_nodes if b in pos]
                    if gather:
                        idx = cp.asarray(gather, dtype=cp.int64)
                        for leaf in leaves_A:
                            key = (A, leaf)
                            if key not in A_union_cache:
                                continue
                            nlist, mat_union = A_union_cache[key]
                            sub = mat_union[:, idx]
                            if sub.size > 0:
                                L2B[L].setdefault((A, B), {})[leaf] = dict(nodes=nlist, mat=sub)
                else:
                    posA = {b: i for i, b in enumerate(sameA["cols_nodes"])}
                    pick_pos = [posA[b] for b in b_nodes if b in posA]
                    for leaf, blk in sameA["leaf_blocks"].items():
                        nlist = leaf_nodes[leaf]
                        sub = _compute_cols_block_for_leaf_gpu(blk["mat"], pick_pos, use_dtype)
                        if sub.size > 0:
                            L2B[L].setdefault((A, B), {})[leaf] = dict(nodes=nlist, mat=sub)

            # L2C
            sameB = same_for_L.get(B)
            if sameB is not None:
                c_nodes = AB["c_nodes"]
                leaves_B = collect_leaves_under(cluster_tree, B)
                modeB = decide(B, False)
                if modeB == "union" and B in B_union_cols:
                    union_cols = B_union_cols[B]; pos = {x: i for i, x in enumerate(union_cols)}
                    gather = [pos[c] for c in c_nodes if c in pos]
                    if gather:
                        idx = cp.asarray(gather, dtype=cp.int64)
                        for leaf in leaves_B:
                            key = (B, leaf)
                            if key not in B_union_cache:
                                continue
                            nlist, mat_union = B_union_cache[key]
                            sub = mat_union[:, idx]
                            if sub.size > 0:
                                L2C[L].setdefault((A, B), {})[leaf] = dict(nodes=nlist, mat=sub)
                else:
                    posB = {c: i for i, c in enumerate(sameB["cols_nodes"])}
                    pick_pos = [posB[c] for c in c_nodes if c in posB]
                    for leaf, blk in sameB["leaf_blocks"].items():
                        nlist = leaf_nodes[leaf]
                        sub = _compute_cols_block_for_leaf_gpu(blk["mat"], pick_pos, use_dtype)
                        if sub.size > 0:
                            L2C[L].setdefault((A, B), {})[leaf] = dict(nodes=nlist, mat=sub)

    return L2B, L2C, leaf_row_index


# -------------------- 查询（严格查表；可单次 CPU，也可批量 GPU） --------------------
def query_distance_all_table(cluster_tree, node_to_leaf, pll_stub_unused,
                             AB_tables, L2B, L2C, leaf_row_index,
                             u, v) -> float:
    if u == v: return 0.0
    leaf_u = node_to_leaf.get(u); leaf_v = node_to_leaf.get(v)
    if (leaf_u is None) or (leaf_v is None): return float("inf")
    if leaf_u == leaf_v:
        # 此 GPU 版未使用 PLL（G_nb 已精确），同叶距离可直接在 G_nb 查询，但为简化沿用 “不在表内则 inf”
        return float("inf")

    # 找 LCA=CC（本实现只有两层：CC 与叶）
    def get_path(cid):
        path = []; cur = cid
        while cur is not None:
            path.append(cur)
            cur = cluster_tree[cur]["parent"] if cur in cluster_tree else None
        return path
    path_u = set(get_path(leaf_u))
    L = None
    cur = leaf_v
    while cur is not None:
        if cur in path_u:
            L = cur if cluster_tree[cur]["level"] != 0 else cluster_tree[cur]["parent"]
            break
        cur = cluster_tree[cur]["parent"]
    if L is None:
        return float("inf")
    # 升到 L 的直接孩子
    Au = leaf_u
    while cluster_tree[Au]["parent"] != L:
        Au = cluster_tree[Au]["parent"]
    Bv = leaf_v
    while cluster_tree[Bv]["parent"] != L:
        Bv = cluster_tree[Bv]["parent"]

    AB = AB_tables.get(L, {}).get((Au, Bv))
    if AB is None: return float("inf")
    tblL = L2B.get(L, {}).get((Au, Bv), {}).get(leaf_u)
    tblR = L2C.get(L, {}).get((Au, Bv), {}).get(leaf_v)
    if (tblL is None) or (tblR is None): return float("inf")

    rL = leaf_row_index[leaf_u].get(u, None)
    rR = leaf_row_index[leaf_v].get(v, None)
    if (rL is None) or (rR is None): return float("inf")

    # 使用 CuPy 做一次性 min
    du_row = tblL["mat"][rL, :].astype(cp.float32, copy=False)  # (nb,)
    dv_row = tblR["mat"][rR, :].astype(cp.float32, copy=False)  # (nc,)
    rows = cp.asarray(AB["rows"], dtype=cp.int64)
    cols = cp.asarray(AB["cols"], dtype=cp.int64)
    mids = cp.asarray(AB["mids"], dtype=cp.float32)
    vals = du_row[rows] + mids + dv_row[cols]
    best = float(vals.min().get()) if vals.size > 0 else float("inf")
    return best


# -------------------- 评测 --------------------
def sample_entity_pairs(G_nx: nx.Graph, n_pairs=1000, in_lcc=True, rng_seed=42):
    entity_nodes = list(G_nx.nodes())
    if len(entity_nodes) < 2:
        raise RuntimeError("图中节点过少，无法评测。")
    comps = list(nx.connected_components(G_nx))
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

def compute_gt_distances(G_nx, pairs):
    gt = []
    for u, v in pairs:
        try:
            d = nx.shortest_path_length(G_nx, u, v, weight="weight")
        except nx.NetworkXNoPath:
            d = float("inf")
        gt.append((u, v, d))
    return gt

def evaluate_all_table_lookup(cluster_tree, node_to_leaf, pll_unused,
                              AB_tables, L2B, L2C, leaf_row_index,
                              gt, preprocessing_time):
    correct = total_eval = 0
    err = 0.0
    inf_pred = 0
    t0 = time.perf_counter()
    for u, v, d in gt:
        pred = query_distance_all_table(cluster_tree, node_to_leaf, None,
                                        AB_tables, L2B, L2C, leaf_row_index, u, v)
        if pred == float("inf"): inf_pred += 1
        if pred == d: correct += 1
        if (pred != float("inf")) and (d != float("inf")):
            err += abs(pred - d); total_eval += 1
    tQ = time.perf_counter() - t0
    rows = [[
        "HierL-E (GPU leaves + GPU L2, strict lookup)",
        tQ, len(gt), correct, (err/total_eval if total_eval>0 else float("inf")),
        inf_pred, preprocessing_time
    ]]
    return pd.DataFrame(rows, columns=[
        "method", "query_time_sec", "samples", "exact_matches", "mae", "inf_pred", "preprocessing_time"
    ])


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="Pubmed", choices=["Cora", "CiteSeer", "Pubmed"])
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--kg_file", type=str, default=None)

    ap.add_argument("--resolution", type=float, default=0.3)
    ap.add_argument("--max_cluster_size", type=int, default=1200)
    ap.add_argument("--hl_seed", type=int, default=42)

    ap.add_argument("--leaf_rg_k", type=int, default=8)
    ap.add_argument("--ig_intra_knn", type=int, default=8)
    ap.add_argument("--mid_limit", type=int, default=0)

    ap.add_argument("--l2c_dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--l2_mode", type=str, default="auto", choices=["auto", "union", "pair"])
    ap.add_argument("--l2_union_ratio", type=float, default=1.5)

    ap.add_argument("--eval_pairs", type=int, default=500)
    ap.add_argument("--save_eval_set", type=str, default=None)
    ap.add_argument("--load_eval_set", type=str, default=None)

    # RMM 池（建议开大一些，避免碎片）
    ap.add_argument("--rmm_pool_gb", type=int, default=28)

    args = ap.parse_args()

    # RMM 初始化
    rmm.reinitialize(pool_allocator=True, initial_pool_size=args.rmm_pool_gb * (1024**3))

    # 0) 载图（边表）
    if args.kg_file:
        edges_df, ent2id, id2ent = load_wn18_edges_aggregated(args.kg_file)
        src_name = f"WN18({os.path.basename(args.kg_file)})-agg"
    else:
        edges_df = load_planetoid_edges(args.dataset, root=args.data_root)
        src_name = args.dataset
    print(f"[INFO] Graph: {src_name}, |E|={len(edges_df)}")

    # 为评测 & 父层构图保留 NX 版全图
    G_nx = nx.Graph()
    for u, v, w in edges_df[["src","dst","weight"]].itertuples(index=False):
        G_nx.add_edge(int(u), int(v), weight=float(w))

    # 1) 分层（到叶，GPU Leiden）
    t0 = time.time()
    cluster_tree, node_to_leaf, G_cu, gdf_edges = build_leaves_by_leiden_gpu(
        edges_df, resolution=args.resolution, max_cluster_size=args.max_cluster_size, seed=args.hl_seed
    )
    comps = [cid for cid, meta in cluster_tree.items() if meta["level"] == -1]
    leaves = [cid for cid, meta in cluster_tree.items() if meta["level"] == 0]
    print("\n[Step1] Skeleton")
    print(f"  CC count={len(comps)} | leaves={len(leaves)}")

    # 2) 叶内（GPU SSSP -> G_nb；kNN rep_graph）
    borders, leaf_nodes, G_nb_all, rep_graphs = build_leaf_tables_gpu(gdf_edges, cluster_tree, leaf_rg_k=args.leaf_rg_k)

    # 3) 父层 index_graph（按原规则，NX）
    index_graphs, idx_stats = build_index_graphs_for_parents(G_nx, cluster_tree, rep_graphs, ig_intra_knn=args.ig_intra_knn)

    # 4) list_indexpath（CPU 多源 Dijkstra，结构不变）
    list_indexpath_lvl1 = build_list_indexpath_for_parents_cpu(cluster_tree, index_graphs, rep_graphs, parents_subset=None)

    # 7) AB
    AB_tables_lvl1 = build_AB_tables(cluster_tree, rep_graphs, list_indexpath_lvl1, mid_limit=args.mid_limit)

    # SAME（GPU 叶块缓存）
    print("[INFO] Building SAME tables (GPU leaf blocks) ...")
    SAME_tables = build_SAME_tables_gpu(cluster_tree, borders, rep_graphs, leaf_nodes, G_nb_all)

    # 7.9) L2B/L2C（GPU）
    print(f"[INFO] Building L2 (mode={args.l2_mode}, ratio={args.l2_union_ratio}) ...")
    L2B, L2C, leaf_row_index = build_leaf2pair_tables_mixed_gpu(
        cluster_tree, borders, leaf_nodes, SAME_tables, AB_tables_lvl1,
        dtype=args.l2c_dtype, mode=args.l2_mode, union_ratio=args.l2_union_ratio
    )

    preprocessing_time = time.time() - t0

    # 评测集
    if args.load_eval_set and os.path.isfile(args.load_eval_set):
        df = pd.read_csv(args.load_eval_set)
        gt = []
        present = set(G_nx.nodes())
        for _, row in df.iterrows():
            u, v, d = int(row["source"]), int(row["target"]), float(row["dist"])
            if (u in present) and (v in present):
                gt.append((u, v, d))
        print(f"[INFO] Loaded eval set: {len(gt)}")
        if not gt:
            raise RuntimeError("Loaded eval set has 0 usable rows.")
    else:
        pairs = sample_entity_pairs(G_nx, n_pairs=args.eval_pairs, in_lcc=True, rng_seed=42)
        print(f"[INFO] Sampling done. computing ground-truth for {len(pairs)} pairs ...")
        gt = compute_gt_distances(G_nx, pairs)
        if args.save_eval_set:
            pd.DataFrame(gt, columns=["source","target","dist"]).to_csv(args.save_eval_set, index=False)

    # 评测（严格查表）
    df_eval = evaluate_all_table_lookup(cluster_tree, node_to_leaf, None,
                                        AB_tables_lvl1, L2B, L2C, leaf_row_index,
                                        gt, preprocessing_time)
    print("\n=== Table-only LCA Query Evaluation (GPU leaves + GPU L2) ===")
    print(df_eval.to_string(index=False))
    if idx_stats:
        total_nodes = sum(s["nodes"] for s in idx_stats.values())
        total_edges = sum(s["edges"] for s in idx_stats.values())
        total_gates = sum(s["gateways"] for s in idx_stats.values())
        print(f"[IndexGraph Stats] nodes={total_nodes}, edges={total_edges}, gateways={total_gates}")
    print(f"\n[Summary] Preprocessing={preprocessing_time:.3f}s, parents_with_AB={len(AB_tables_lvl1)}, parents_with_L2={len(L2B)}")
    print("[OK] Finished: strict table lookup (heavy parts on GPU).")


if __name__ == "__main__":
    main()
