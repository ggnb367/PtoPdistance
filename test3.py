import argparse
import time
import random
import heapq
import os
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import networkx as nx
import pandas as pd
from tqdm import tqdm
from torch_geometric.datasets import Planetoid
from graspologic.partition import hierarchical_leiden


# ===============================
# Pruned PLL Index (query-only friendly)
# ===============================
class PrunedPLLIndex:
    """
    Pruned Landmark Labeling for weighted graphs.
    - build() 需要 G；
    - query() 只依赖 labels，可以在没有 G 的情况下工作；
    - from_labels() 允许用子进程返回的 labels 在主进程复原查询用索引。
    """

    def __init__(self, G: nx.Graph | None = None, order=None):
        self.G = G
        self.labels = {} if G is None else {v: {} for v in G.nodes()}
        self.order = list(order) if order is not None else (list(G.nodes()) if G is not None else [])

    @classmethod
    def from_labels(cls, labels: dict):
        obj = cls(G=None, order=None)
        obj.labels = labels  # {node: {landmark: dist}}
        return obj

    def build(self):
        assert self.G is not None, "build() needs a graph; use from_labels() if you already have labels."
        for v in tqdm(self.order, desc="Building PLL", unit="node"):
            self._pruned_dijkstra(v)

    def _pruned_dijkstra(self, root):
        dist = {root: 0}
        heap = [(0, 0, root)]
        counter = 0
        while heap:
            d, _, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if self.query(root, u) <= d:
                continue
            self.labels[u][root] = d
            for v, data in self.G[u].items():
                w = data.get("weight", 1)
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    counter += 1
                    heapq.heappush(heap, (nd, counter, v))

    def query(self, u, v):
        best = float("inf")
        lu = self.labels.get(u, {})
        lv = self.labels.get(v, {})
        if len(lu) > len(lv):  # iterate the smaller label map
            lu, lv = lv, lu
        for lm, du in lu.items():
            dv = lv.get(lm)
            if dv is not None:
                s = du + dv
                if s < best:
                    best = s
        return best


# ===============================
# Helpers: dataset & partitions
# ===============================
def load_planetoid_graph(name="Pubmed"):
    dataset = Planetoid(root=f"/tmp/{name}", name=name)
    data = dataset[0]
    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    # undirected, remove self-loops, uniquify
    edges = set()
    for u, v in zip(edge_index[0], edge_index[1]):
        if u == v:
            continue
        a, b = int(u), int(v)
        if a > b:
            a, b = b, a
        edges.add((a, b))
    for u, v in edges:
        G.add_edge(u, v, weight=1)
    return G


def final_partition_from_hl(hl_result, G: nx.Graph) -> dict:
    """
    只取 is_final_cluster=True 的条目；缺失的孤立点单独成簇；
    压缩 (level, cluster) => 连续整数 cluster_id。
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

    assert len(part) == G.number_of_nodes(), "Partition must cover all nodes"
    return part


# ===============================
# Worker: build one cluster PLL in a subprocess
# ===============================
def _build_one_cluster(args):
    """
    子进程任务：在一个簇内部构建 PLL，并返回：
      (cid, nodes, boundary_list, boundary_pairs_dict, labels_dict, build_time, label_entries)
    """
    cid, nodes, edges, neigh_map = args
    nodes_set = set(nodes)

    # 构建簇内子图
    subg = nx.Graph()
    subg.add_nodes_from(nodes)
    for u, v, w in edges:
        subg.add_edge(u, v, weight=w)

    # 以 eccentricity 排序（分量内计算）
    ecc = {}
    for comp in nx.connected_components(subg):
        cg = subg.subgraph(comp)
        ecc.update(nx.eccentricity(cg))
    order = sorted(nodes, key=lambda n: ecc.get(n, 0))

    # 构建簇内 PLL
    t0 = time.time()
    pll = PrunedPLLIndex(subg, order)
    for root in pll.order:
        dist = {root: 0}
        heap = [(0, 0, root)]
        counter = 0
        while heap:
            d, _, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if pll.query(root, u) <= d:
                continue
            pll.labels[u][root] = d
            for v, data in subg[u].items():
                w = data.get("weight", 1)
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    counter += 1
                    heapq.heappush(heap, (nd, counter, v))
    build_time = time.time() - t0

    # 边界判定：原图中与簇外相邻的点
    boundary = [n for n in nodes if any((nbr not in nodes_set) for nbr in neigh_map[n])]

    # 簇内边界点两两最短距离（用内部 PLL 查询）
    pairs = {}
    for i in range(len(boundary)):
        ui = boundary[i]
        for j in range(i + 1, len(boundary)):
            vj = boundary[j]
            pairs[(ui, vj)] = pll.query(ui, vj)

    # label entry 计数
    label_entries = sum(len(m) for m in pll.labels.values())

    return cid, nodes, boundary, pairs, pll.labels, build_time, label_entries


# ===============================
# Storage estimation (payload only)
# ===============================
def payload_bytes_labels_entry_count(entry_count: int) -> int:
    """
    近似 payload 字节：每条 label (landmark->distance) 视为 8B(int64) + 8B(float64) = 16B
    不计 Python 容器开销。
    """
    return 16 * entry_count


def payload_bytes_mapping_pairs(pair_count: int) -> int:
    """
    近似 payload 字节：映射 pair(int->int) 视为 16B
    """
    return 16 * pair_count


def payload_bytes_float_count(count: int) -> int:
    return 8 * count


def format_mib(nbytes: int) -> str:
    return f"{nbytes / (1024*1024):.2f} MiB"


# ===============================
# Build hybrid structures (both variants) + stats
# ===============================
def build_hybrid_structures(G, partition, max_workers=None, show_progress=True):
    """
    构建两种混合方案共享的结构，并返回结构与构建/存储统计：
      - structures:
        inside_pll, boundary_sets, node_cluster, G_out, outside_pll, cluster_pair_info
      - stats: {
          'inside': {...},
          'g_out': {...},
          'outside_pll': {...},
          'virtual_pairs': {...},
          'totals': {...}
        }
    """
    # 1) cluster -> nodes
    clusters = defaultdict(list)
    for n, cid in partition.items():
        clusters[cid].append(n)

    # 2) 预备轻量数据
    neigh_map = {n: list(G.neighbors(n)) for n in G.nodes()}

    tasks = []
    for cid, nodes in clusters.items():
        nset = set(nodes)
        edges = [(u, v, data.get("weight", 1))
                 for u, v, data in G.edges(nodes, data=True)
                 if u in nset and v in nset]
        tasks.append((cid, nodes, edges, neigh_map))

    if max_workers is None:
        max_workers = os.cpu_count() or 2

    inside_pll = {}
    boundary_sets = {}
    boundary_pairs = {}
    node_cluster = {}

    # 3) 并行跑每个簇
    inside_time_sum = 0.0
    inside_label_entries_sum = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_build_one_cluster, t) for t in tasks]
        it = as_completed(futures)
        if show_progress:
            it = tqdm(it, total=len(futures), desc="Clusters (parallel PLL)", unit="cluster")
        for fut in it:
            cid, nodes, boundary, pairs, labels, build_time, label_entries = fut.result()
            inside_time_sum += build_time
            inside_label_entries_sum += label_entries
            inside_pll[cid] = PrunedPLLIndex.from_labels(labels)
            boundary_sets[cid] = set(boundary)
            for n in nodes:
                node_cluster[n] = cid
            boundary_pairs[cid] = pairs

    inside_stats = {
        "clusters": len(clusters),
        "parallel_workers": max_workers,
        "time_sec_sum_across_clusters": inside_time_sum,
        "label_entries_sum": inside_label_entries_sum,
        "payload_bytes_labels_sum": payload_bytes_labels_entry_count(inside_label_entries_sum),
        "boundary_nodes_sum": sum(len(s) for s in boundary_sets.values()),
    }

    # 4) G_out：所有簇的边界点构成节点集合 & 构建耗时
    t_gout0 = time.time()
    outside_nodes = set()
    for cid in clusters:
        outside_nodes |= boundary_sets.get(cid, set())

    G_out = nx.Graph()
    G_out.add_nodes_from(outside_nodes)

    # 4.1 跨簇原始边（保留）
    for u, v, data in G.edges(data=True):
        if u in outside_nodes and v in outside_nodes:
            if partition[u] != partition[v]:
                G_out.add_edge(u, v, weight=data.get("weight", 1))

    # 4.2 同簇边界点间的“超边”（权重取簇内 PLL）
    intra_super_edges = 0
    for cid, pairs in boundary_pairs.items():
        for (u, v), d in pairs.items():
            if u == v or d == float("inf"):
                continue
            if G_out.has_edge(u, v):
                if d < G_out[u][v].get("weight", float("inf")):
                    G_out[u][v]["weight"] = d
            else:
                G_out.add_edge(u, v, weight=d)
                intra_super_edges += 1
    t_gout = time.time() - t_gout0

    g_out_stats = {
        "time_sec_build_g_out": t_gout,
        "nodes": G_out.number_of_nodes(),
        "edges": G_out.number_of_edges(),
        "intra_super_edges_added": intra_super_edges,
        # payload 估算（仅节点 id 与权重）
        "payload_bytes_nodes": payload_bytes_mapping_pairs(G_out.number_of_nodes()),  # 以单节点 int64 记 8B -> 简单用 16B/pair 近似保守
        "payload_bytes_edges": (8+8+8) * G_out.number_of_edges(),  # (u,v,w)
    }

    # 5) Hybrid A: 在 G_out 上建一个 PLL
    t_outpll0 = time.time()
    degree_order = sorted(G_out.degree(), key=lambda x: x[1], reverse=True)
    order_out = [n for n, _ in degree_order]
    outside_pll = PrunedPLLIndex(G_out, order_out)
    outside_pll.build()
    t_outpll = time.time() - t_outpll0
    outside_pll_entries = sum(len(m) for m in outside_pll.labels.values())
    outside_pll_stats = {
        "time_sec_build_outside_pll": t_outpll,
        "label_entries": outside_pll_entries,
        "payload_bytes_labels": payload_bytes_labels_entry_count(outside_pll_entries),
    }

    # 6) Hybrid B: 构造带虚拟源点代表图 + 簇对最短路
    t_vs0 = time.time()
    G_rep = G_out.copy()
    virtual_of = {}  # cid -> virtual node name
    for cid, bset in boundary_sets.items():
        vnode = ("VIRT", cid)
        virtual_of[cid] = vnode
        G_rep.add_node(vnode)
        for b in bset:
            G_rep.add_edge(vnode, b, weight=0)
    t_vs_graph = time.time() - t_vs0

    cluster_ids = sorted(clusters.keys())
    t_pairs0 = time.time()
    cluster_pair_info = {}  # (mincid,maxcid) -> {'dist': d, 'exit_u': u_b, 'enter_v': v_b}
    for i in range(len(cluster_ids)):
        ci = cluster_ids[i]
        si = virtual_of[ci]
        for j in range(i + 1, len(cluster_ids)):
            cj = cluster_ids[j]
            sj = virtual_of[cj]
            try:
                d, path = nx.bidirectional_dijkstra(G_rep, si, sj, weight="weight")
                exit_u = None
                enter_v = None
                if len(path) >= 2:
                    if path[1] in boundary_sets.get(ci, set()):
                        exit_u = path[1]
                    if path[-2] in boundary_sets.get(cj, set()):
                        enter_v = path[-2]
                cluster_pair_info[(ci, cj)] = {"dist": d, "exit_u": exit_u, "enter_v": enter_v}
            except nx.NetworkXNoPath:
                cluster_pair_info[(ci, cj)] = {"dist": float("inf"), "exit_u": None, "enter_v": None}
    t_pairs = time.time() - t_pairs0

    virtual_pairs_stats = {
        "time_sec_build_virtual_graph": t_vs_graph,
        "time_sec_all_pairs_shortest": t_pairs,
        "pairs": len(cluster_pair_info),
        # payload：每对存 dist(8B) + 两个节点 id(2*8B) ≈ 24B，不含 key
        "payload_bytes_pairs": (8 + 8 + 8) * len(cluster_pair_info),
    }

    totals_stats = {
        "time_sec_total_build_A": inside_stats["time_sec_sum_across_clusters"] + g_out_stats["time_sec_build_g_out"] + outside_pll_stats["time_sec_build_outside_pll"],
        "time_sec_total_build_B": inside_stats["time_sec_sum_across_clusters"] + g_out_stats["time_sec_build_g_out"] + virtual_pairs_stats["time_sec_build_virtual_graph"] + virtual_pairs_stats["time_sec_all_pairs_shortest"],
        "payload_bytes_total_A": inside_stats["payload_bytes_labels_sum"] + g_out_stats["payload_bytes_nodes"] + g_out_stats["payload_bytes_edges"] + outside_pll_stats["payload_bytes_labels"],
        "payload_bytes_total_B": inside_stats["payload_bytes_labels_sum"] + g_out_stats["payload_bytes_nodes"] + g_out_stats["payload_bytes_edges"] + virtual_pairs_stats["payload_bytes_pairs"],
    }

    stats = {
        "inside": inside_stats,
        "g_out": g_out_stats,
        "outside_pll": outside_pll_stats,
        "virtual_pairs": virtual_pairs_stats,
        "totals": totals_stats,
    }

    return inside_pll, boundary_sets, node_cluster, G_out, outside_pll, cluster_pair_info, stats


# ===============================
# Hybrid A query: leaf-PLL + G_out-PLL
# ===============================
def query_hybrid_outpll(u, v, node_cluster, inside_pll, boundary_sets, outside_pll):
    cid_u = node_cluster[u]
    cid_v = node_cluster[v]
    bu = boundary_sets.get(cid_u, set())
    bv = boundary_sets.get(cid_v, set())

    # 同簇且无边界：直接簇内
    if cid_u == cid_v and not bu:
        return inside_pll[cid_u].query(u, v)

    def project(x, cid, boundary_set):
        if not boundary_set:
            return None, float("inf")
        if x in boundary_set:
            return x, 0.0
        pll = inside_pll[cid]
        best = (float("inf"), None)
        for b in boundary_set:
            d = pll.query(x, b)
            if d < best[0]:
                best = (d, b)
        return best[1], best[0]  # (boundary_node, dist_inside)

    e_u, d_u = project(u, cid_u, bu)
    e_v, d_v = project(v, cid_v, bv)

    mid = 0.0 if (e_u is None or e_v is None or e_u == e_v) else outside_pll.query(e_u, e_v)
    return d_u + mid + d_v


# ===============================
# Hybrid B query: leaf-PLL + cluster-virtual
# ===============================
def query_hybrid_virtual(u, v, node_cluster, inside_pll, boundary_sets, cluster_pair_info):
    cid_u = node_cluster[u]
    cid_v = node_cluster[v]

    if cid_u == cid_v:
        return inside_pll[cid_u].query(u, v)

    key = (cid_u, cid_v) if cid_u < cid_v else (cid_v, cid_u)
    info = cluster_pair_info.get(key, None)
    if info is None or info["dist"] == float("inf"):
        return float("inf")

    if cid_u < cid_v:
        b_u = info["exit_u"]
        b_v = info["enter_v"]
    else:
        b_u = info["enter_v"]
        b_v = info["exit_u"]

    def best_proj(x, cid, b_target):
        pll = inside_pll[cid]
        bset = boundary_sets.get(cid, set())
        if b_target is not None:
            return pll.query(x, b_target)
        if not bset:
            return float("inf")
        best = float("inf")
        for b in bset:
            d = pll.query(x, b)
            if d < best:
                best = d
        return best

    du = best_proj(u, cid_u, b_u)
    dv = best_proj(v, cid_v, b_v)
    mid = info["dist"]
    return du + mid + dv


# ===============================
# Evaluation (compare the two hybrids)
# ===============================
def evaluate(G, inside_pll, boundary_sets, node_cluster, outside_pll, cluster_pair_info, n_pairs=500):
    nodes = list(G.nodes())
    pairs = [(random.choice(nodes), random.choice(nodes)) for _ in range(n_pairs)]

    # Ground truth by Dijkstra on original G
    gt = []
    for u, v in pairs:
        try:
            d = nx.shortest_path_length(G, u, v, weight="weight")
            gt.append((u, v, d))
        except nx.NetworkXNoPath:
            pass

    results = []

    # Hybrid A
    correct = 0
    total = 0
    err = 0.0
    tA = time.time()
    for u, v, d in gt:
        est = query_hybrid_outpll(u, v, node_cluster, inside_pll, boundary_sets, outside_pll)
        if est == d:
            correct += 1
        if est != float("inf"):
            err += abs(est - d)
            total += 1
    qA = time.time() - tA
    results.append({
        "method": "Hybrid A (leaf-PLL + G_out-PLL)",
        "query_time_sec": qA,
        "samples": total,
        "exact_matches": correct,
        "mae": (err / total) if total > 0 else float("inf"),
    })

    # Hybrid B
    correct = 0
    total = 0
    err = 0.0
    tB = time.time()
    for u, v, d in gt:
        est = query_hybrid_virtual(u, v, node_cluster, inside_pll, boundary_sets, cluster_pair_info)
        if est == d:
            correct += 1
        if est != float("inf"):
            err += abs(est - d)
            total += 1
    qB = time.time() - tB
    results.append({
        "method": "Hybrid B (leaf-PLL + Virtual-Source)",
        "query_time_sec": qB,
        "samples": total,
        "exact_matches": correct,
        "mae": (err / total) if total > 0 else float("inf"),
    })

    return pd.DataFrame(results)


# ===============================
# Pretty print build/storage stats
# ===============================
def print_build_and_storage_stats(stats: dict):
    print("\n=== Build Time Stats ===")
    print(f"- Inside (per-cluster PLL, sum over clusters): {stats['inside']['time_sec_sum_across_clusters']:.3f}s "
          f"(clusters={stats['inside']['clusters']}, workers={stats['inside']['parallel_workers']})")
    print(f"- Build G_out (boundary graph): {stats['g_out']['time_sec_build_g_out']:.3f}s")
    print(f"- Outside PLL (on G_out) [Hybrid A]: {stats['outside_pll']['time_sec_build_outside_pll']:.3f}s")
    print(f"- Virtual graph build [Hybrid B]: {stats['virtual_pairs']['time_sec_build_virtual_graph']:.3f}s")
    print(f"- All-pairs (virtual source) shortest paths [Hybrid B]: {stats['virtual_pairs']['time_sec_all_pairs_shortest']:.3f}s")
    print(f"- Total Build A (inside + G_out + outside PLL): {stats['totals']['time_sec_total_build_A']:.3f}s")
    print(f"- Total Build B (inside + G_out + virtual graph + pairs): {stats['totals']['time_sec_total_build_B']:.3f}s")

    print("\n=== Storage Payload Estimates (content-only, no Python overhead) ===")
    print(f"- Inside PLL label entries: {stats['inside']['label_entries_sum']:,} "
          f"~ {format_mib(stats['inside']['payload_bytes_labels_sum'])}")
    print(f"- Boundary nodes (unique across clusters): {stats['inside']['boundary_nodes_sum']:,}")
    print(f"- G_out: nodes={stats['g_out']['nodes']:,}, edges={stats['g_out']['edges']:,}, "
          f"payload(nodes)~{format_mib(stats['g_out']['payload_bytes_nodes'])}, "
          f"payload(edges)~{format_mib(stats['g_out']['payload_bytes_edges'])}")
    print(f"- Outside PLL labels [Hybrid A]: {stats['outside_pll']['label_entries']:,} "
          f"~ {format_mib(stats['outside_pll']['payload_bytes_labels'])}")
    print(f"- Cluster-pair records [Hybrid B]: {stats['virtual_pairs']['pairs']:,} "
          f"~ {format_mib(stats['virtual_pairs']['payload_bytes_pairs'])}")
    print(f"- Total payload [Hybrid A]: {format_mib(stats['totals']['payload_bytes_total_A'])}")
    print(f"- Total payload [Hybrid B]: {format_mib(stats['totals']['payload_bytes_total_B'])}")
    print("(Note) Payload = 粗略估算，仅计 int64/float64 内容字节，不含 Python dict/list 等容器开销。")


# ===============================
# Main
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Pubmed", choices=["Cora", "CiteSeer", "Pubmed"])
    parser.add_argument("--pairs", type=int, default=500, help="number of query pairs for evaluation")
    parser.add_argument("--resolution", type=float, default=0.3, help="Leiden resolution")
    parser.add_argument("--max_cluster_size", type=int, default=1000, help="Leiden max_cluster_size trigger")
    parser.add_argument("--max_workers", type=int, default=None, help="parallel processes for per-cluster PLL")
    args = parser.parse_args()

    # 0) Load graph
    G = load_planetoid_graph(args.dataset)
    print(f"[INFO] Graph: {args.dataset}, |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

    # 1) Multi-level Leiden -> final clusters
    hl = hierarchical_leiden(
        G,
        max_cluster_size=args.max_cluster_size,
        resolution=args.resolution,
        use_modularity=True,
        random_seed=42,
        check_directed=True,
    )
    partition = final_partition_from_hl(hl, G)

    sizes = Counter(partition.values())
    top_sizes = sizes.most_common(10)
    print(f"[INFO] #final_clusters = {len(sizes)}, Top-10 sizes: {top_sizes[:10]}")

    # 2) Build structures + stats
    t_build0 = time.time()
    inside_pll, boundary_sets, node_cluster, G_out, outside_pll, cluster_pair_info, stats = build_hybrid_structures(
        G, partition, max_workers=args.max_workers, show_progress=True
    )
    t_build = time.time() - t_build0
    print(f"[INFO] All structures prepared in {t_build:.3f}s. "
          f"G_out: |V|={G_out.number_of_nodes()}, |E|={G_out.number_of_edges()}, "
          f"cluster-pairs: {len(cluster_pair_info)}")

    # 3) Print build/storage stats
    print_build_and_storage_stats(stats)

    # 4) Evaluate A vs B
    df = evaluate(G, inside_pll, boundary_sets, node_cluster, outside_pll, cluster_pair_info, n_pairs=args.pairs)
    print("\n=== Evaluation (A vs B) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
