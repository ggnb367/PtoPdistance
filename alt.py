#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import time
import random
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError:
    print("Please: pip install networkx numpy")
    sys.exit(1)


# ============================================================
# 加载 WN18：实体与关系均为节点；每条 triple 拆成两条 0.5 边
# ============================================================
def load_wn18_graph_relation_nodes_weighted(
    path: str,
    rel_weight: float = 0.5,
    keep_lcc: bool = True,
) -> Tuple[List[List[Tuple[int, float]]], Dict[int, Any], Dict[Any, int]]:
    """
    读取 WN18.txt 并构建无向加权图：
      - 每个 triple (h, t, r) 建一个唯一关系节点 ("REL", i, r)
      - 加边： h --0.5--> REL_i_r --0.5--> t
    返回：
      adjw: 邻接表（0..n-1），每项为 (neighbor, weight)
      id2orig / orig2id: 压缩编号的映射
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"WN18 file not found: {path}")

    triples: List[Tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        # 首行可能是 triple 数；否则就是第一条 triple
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
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            h, t = parts[0], parts[1]
            r = parts[2] if len(parts) >= 3 else "NA"
            if h == t:
                continue
            triples.append((h, t, r))

    G = nx.Graph()
    # 先加实体节点
    entities = set()
    for h, t, _ in triples:
        entities.add(h)
        entities.add(t)
    G.add_nodes_from(entities)

    # 每条 triple 单独关系节点，避免把同关系全并成一个 hub 造成捷径
    for i, (h, t, r) in enumerate(triples):
        rel_node = ("REL", i, r)
        G.add_node(rel_node)
        G.add_edge(h, rel_node, weight=rel_weight)
        G.add_edge(rel_node, t, weight=rel_weight)

    # 保留最大连通分量（稳定评测）
    if keep_lcc and not nx.is_connected(G):
        cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(cc).copy()

    # 压缩成 0..n-1
    nodes = list(G.nodes())
    orig2id: Dict[Any, int] = {u: i for i, u in enumerate(nodes)}
    id2orig: Dict[int, Any] = {i: u for u, i in orig2id.items()}

    n = len(nodes)
    adjw: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, data in G.edges(data=True):
        ui, vi = orig2id[u], orig2id[v]
        w = float(data.get("weight", 1.0))
        adjw[ui].append((vi, w))
        adjw[vi].append((ui, w))

    m = G.number_of_edges()
    print(f"[INFO] Loaded WN18(rel-nodes, weighted): n={n} m={m} (undirected)")
    return adjw, id2orig, orig2id


# ============================================================
# 加权最短路（Dijkstra）
# ============================================================
import heapq

def dijkstra_all(adjw: List[List[Tuple[int, float]]], src: int) -> np.ndarray:
    """
    单源加权最短路。返回 float64 数组，np.inf 表示不可达。
    """
    n = len(adjw)
    dist = np.full(n, np.inf, dtype=np.float64)
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in adjw[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def dijkstra_sp(adjw: List[List[Tuple[int, float]]], s: int, t: int) -> float:
    """
    s->t 的加权最短路（提前停止）。
    """
    if s == t:
        return 0.0
    n = len(adjw)
    dist = np.full(n, np.inf, dtype=np.float64)
    dist[s] = 0.0
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u == t:
            return d
        for v, w in adjw[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return math.inf


# ============================================================
# 地标选择
# ============================================================
def select_landmarks_farthest_weighted(adjw: List[List[Tuple[int, float]]], k: int, seed: int = 42) -> List[int]:
    """
    加权版“最远点优先”：
      先从随机点出发跑 Dijkstra，选最远点 L1；
      之后每步选择“到已选集合的最小距离”最大的点。
    """
    rng = random.Random(seed)
    n = len(adjw)
    start = rng.randrange(n)
    d0 = dijkstra_all(adjw, start)
    L1 = int(np.nanargmax(np.where(np.isfinite(d0), d0, -np.inf)))
    landmarks = [L1]
    mind = d0.copy()  # 到任一已选地标的最小距离
    for _ in range(1, k):
        cand = int(np.nanargmax(np.where(np.isfinite(mind), mind, -np.inf)))
        landmarks.append(cand)
        d = dijkstra_all(adjw, cand)
        mask = d < mind
        mind[mask] = d[mask]
    return landmarks


def select_landmarks_top_degree(adjw: List[List[Tuple[int, float]]], k: int) -> List[int]:
    deg = [(i, len(adjw[i])) for i in range(len(adjw))]
    deg.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in deg[:k]]


# ============================================================
# 预计算：地标到所有点的加权距离矩阵
# ============================================================
def precompute_lm_dists_weighted(adjw: List[List[Tuple[int, float]]], landmarks: List[int]) -> np.ndarray:
    """
    返回 shape=(L, n) 的 float64 数组；np.inf 表示不可达。
    """
    n = len(adjw); L = len(landmarks)
    D = np.empty((L, n), dtype=np.float64)
    for i, Lv in enumerate(landmarks):
        D[i, :] = dijkstra_all(adjw, Lv)
    return D


# ============================================================
# ALT A：单向 A* + 子集地标
# ============================================================
def pick_subset_by_pair_weighted(D: np.ndarray, s: int, t: int, L_subset: int) -> np.ndarray:
    """
    选择最区分 (s,t) 的地标子集：
      按 |d(L,s)-d(L,t)| 降序挑前 L_subset。
    """
    ds = D[:, s]; dt = D[:, t]
    ok = np.isfinite(ds) & np.isfinite(dt)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return np.arange(min(L_subset, D.shape[0]), dtype=np.int32)
    score = np.abs(ds[idx] - dt[idx])
    order = idx[np.argsort(-score)]
    return order[: min(L_subset, order.size)].astype(np.int32)


def heuristic_h_subset_weighted(D: np.ndarray, dt_sub: np.ndarray, v: int, sub_idx: np.ndarray) -> float:
    """
    h(v)=max_L |d(L,t)-d(L,v)|，仅在子集 sub_idx 上计算。
    """
    h = 0.0
    for i, Lidx in enumerate(sub_idx):
        dlt = dt_sub[i]        # D[Lidx, t]
        dlu = D[Lidx, v]       # D[Lidx, v]
        if not (np.isfinite(dlt) and np.isfinite(dlu)):
            continue
        diff = abs(dlt - dlu)
        if diff > h:
            h = diff
    return h


import heapq
def alt_astar_single_weighted(
    adjw: List[List[Tuple[int, float]]],
    D: np.ndarray,
    s: int,
    t: int,
    L_subset: int = 8
) -> float:
    """
    单向 A*（加权），ALT 子集地标启发式。
    """
    if s == t:
        return 0.0

    n = len(adjw)
    sub = pick_subset_by_pair_weighted(D, s, t, L_subset)
    dt_sub = np.array([D[Lidx, t] for Lidx in sub], dtype=np.float64)

    INF = float("inf")
    g = np.full(n, INF, dtype=np.float64)
    closed = np.zeros(n, dtype=bool)
    g[s] = 0.0
    h_s = heuristic_h_subset_weighted(D, dt_sub, s, sub)
    pq: List[Tuple[float, int]] = [(g[s] + h_s, s)]  # (f, node)

    while pq:
        f, u = heapq.heappop(pq)
        if closed[u]:
            continue
        if u == t:
            return g[u]
        closed[u] = True
        ndu_base = g[u]
        for v, w in adjw[u]:
            if closed[v]:
                continue
            nd = ndu_base + w
            if nd < g[v]:
                g[v] = nd
                hv = heuristic_h_subset_weighted(D, dt_sub, v, sub)
                heapq.heappush(pq, (nd + hv, v))
    return math.inf


# ============================================================
# ALT B：双向 A* + 全部地标
# ============================================================
def heuristic_h_all_weighted(D: np.ndarray, dt: np.ndarray, v: int) -> float:
    """
    h(v)=max_L |d(L,t)-d(L,v)|，使用全部地标（跳过 inf）。
    """
    h = 0.0
    for i in range(D.shape[0]):
        dlt = dt[i]
        dlu = D[i, v]
        if not (np.isfinite(dlt) and np.isfinite(dlu)):
            continue
        diff = abs(dlt - dlu)
        if diff > h:
            h = diff
    return h


def alt_bidirectional_weighted(
    adjw: List[List[Tuple[int, float]]],
    D: np.ndarray,
    s: int,
    t: int
) -> float:
    """
    双向 A*（加权），两端均用 ALT 全部地标启发式。
    终止条件：min_f_forward + min_f_backward >= μ（当前最好路径长度）。
    """
    if s == t:
        return 0.0

    n = len(adjw)
    ds = D[:, s]  # 地标到 s 的距离
    dt = D[:, t]  # 地标到 t 的距离

    INF = float("inf")
    gF = np.full(n, INF, dtype=np.float64)
    gB = np.full(n, INF, dtype=np.float64)
    closedF = np.zeros(n, dtype=bool)
    closedB = np.zeros(n, dtype=bool)
    gF[s] = 0.0
    gB[t] = 0.0

    h_s = heuristic_h_all_weighted(D, dt, s)
    h_t = heuristic_h_all_weighted(D, ds, t)
    QF: List[Tuple[float, int]] = [(gF[s] + h_s, s)]
    QB: List[Tuple[float, int]] = [(gB[t] + h_t, t)]

    mu = INF

    while QF and QB:
        fF = QF[0][0] if QF else INF
        fB = QB[0][0] if QB else INF
        if fF + fB >= mu:
            break

        forward = fF <= fB
        if forward:
            f, u = heapq.heappop(QF)
            if closedF[u]:
                continue
            closedF[u] = True
            if closedB[u]:
                mu = min(mu, gF[u] + gB[u])

            base = gF[u]
            for v, w in adjw[u]:
                if closedF[v]:
                    continue
                nd = base + w
                if nd < gF[v]:
                    gF[v] = nd
                    hv = heuristic_h_all_weighted(D, dt, v)
                    heapq.heappush(QF, (nd + hv, v))
                if closedB[v]:
                    mu = min(mu, gF[v] + gB[v])
        else:
            f, u = heapq.heappop(QB)
            if closedB[u]:
                continue
            closedB[u] = True
            if closedF[u]:
                mu = min(mu, gF[u] + gB[u])

            base = gB[u]
            for v, w in adjw[u]:
                if closedB[v]:
                    continue
                nd = base + w
                if nd < gB[v]:
                    gB[v] = nd
                    hv = heuristic_h_all_weighted(D, ds, v)
                    heapq.heappush(QB, (nd + hv, v))
                if closedF[v]:
                    mu = min(mu, gF[v] + gB[v])

    return mu if math.isfinite(mu) else math.inf


# ============================================================
# 评测与打印
# ============================================================
def print_header():
    print("=== Weighted ALT on WN18 (entity+relation nodes) ===\n")
    print(f"{'method':<40}{'query_time_sec':>14}{'samples':>10}{'exact_matches':>16}{'mae':>12}")

def print_row(method: str, qtime_avg: float, samples: int, exact: int, mae: float):
    print(f"{method:<40}{qtime_avg:>14.6f}{samples:>10}{exact:>16}{mae:>12.6f}")

def print_pre_row(name: str, sec: float):
    print(f"{name:<40}{sec:>14.6f}{'':>10}{'':>16}{'':>12}")

def benchmark(
    adjw: List[List[Tuple[int, float]]],
    samples: int = 500,
    seed: int = 42,
    kA: int = 16,
    kB: int = 16,
    L_subset: int = 8,
):
    rng = random.Random(seed)
    n = len(adjw)

    # ===== 预处理计时（地标 + D矩阵） =====
    t_prep_start = time.time()
    # 地标与距离（两套）
    t_far_start = time.time()
    L_far = select_landmarks_farthest_weighted(adjw, kA, seed=seed)
    t_far_sel = time.time() - t_far_start

    t_farD_start = time.time()
    D_far = precompute_lm_dists_weighted(adjw, L_far)
    t_far_D = time.time() - t_farD_start

    t_deg_start = time.time()
    L_deg = select_landmarks_top_degree(adjw, kB)
    t_deg_sel = time.time() - t_deg_start

    t_degD_start = time.time()
    D_deg = precompute_lm_dists_weighted(adjw, L_deg)
    t_deg_D = time.time() - t_degD_start

    t_prep = time.time() - t_prep_start

    # 采样 (s,t)
    pairs: List[Tuple[int, int]] = []
    for _ in range(samples):
        s = rng.randrange(n)
        t = rng.randrange(n)
        while t == s:
            t = rng.randrange(n)
        pairs.append((s, t))

    # 加权 ground truth
    gt = [dijkstra_sp(adjw, s, t) for (s, t) in pairs]

    # ALT-A（单向 A* + 子集地标）
    tA = 0.0; exactA = 0; errA = 0.0
    for (s, t), g in zip(pairs, gt):
        t0 = time.time()
        d = alt_astar_single_weighted(adjw, D_far, s, t, L_subset=L_subset)
        tA += (time.time() - t0)
        if math.isfinite(d) and math.isfinite(g) and abs(d - g) <= 1e-9:
            exactA += 1
        errA += abs((d if math.isfinite(d) else 1e12) - g)
    qA = tA / samples
    maeA = errA / samples

    # ALT-B（双向 A* + 全部地标）
    tB = 0.0; exactB = 0; errB = 0.0
    for (s, t), g in zip(pairs, gt):
        t0 = time.time()
        d = alt_bidirectional_weighted(adjw, D_deg, s, t)
        tB += (time.time() - t0)
        if math.isfinite(d) and math.isfinite(g) and abs(d - g) <= 1e-9:
            exactB += 1
        errB += abs((d if math.isfinite(d) else 1e12) - g)
    qB = tB / samples
    maeB = errB / samples

    # ===== 打印 =====
    print_header()
    print_pre_row("pre processing_time", t_prep)
    #（如需展开细分项，放开下面四行）
    # print_pre_row("  - farthest landmark select", t_far_sel)
    # print_pre_row("  - farthest D matrix", t_far_D)
    # print_pre_row("  - top-degree landmark select", t_deg_sel)
    # print_pre_row("  - top-degree D matrix", t_deg_D)
    print_row("ALT-A (uni, farthest, subset)", qA, samples, exactA, maeA)
    print_row("ALT-B (bidir, top-degree, all)", qB, samples, exactB, maeB)


# ============================================================
# 主入口
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg_file", type=str, required=True,
                    help="Path to WN18.txt (first line may be a count).")
    ap.add_argument("--samples", type=int, default=500, help="number of (s,t) query pairs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kA", type=int, default=16, help="#landmarks for ALT-A (farthest)")
    ap.add_argument("--kB", type=int, default=16, help="#landmarks for ALT-B (top-degree)")
    ap.add_argument("--L_subset", type=int, default=8, help="ALT-A subset size")
    return ap.parse_args()


def main():
    args = parse_args()

    # === 读图计时（单独统计） ===
    t_load_start = time.time()
    adjw, id2orig, orig2id = load_wn18_graph_relation_nodes_weighted(args.kg_file, rel_weight=0.5, keep_lcc=True)
    t_load = time.time() - t_load_start

    # 先打印一行 graph_loading_time，随后由 benchmark 打印 pre processing_time 和查询行
    print_header()
    print_pre_row("graph_loading_time", t_load)

    # 评测 + 打印（内部会再打印表头；为了更清晰，复用同一表头不影响理解）
    benchmark(
        adjw,
        samples=args.samples,
        seed=args.seed,
        kA=args.kA,
        kB=args.kB,
        L_subset=args.L_subset,
    )


if __name__ == "__main__":
    main()
