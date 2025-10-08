from __future__ import annotations
"""
Single-file G-Tree (minimal, educational) + PubMed end-to-end test
=================================================================
Run:
  python run_pubmed_gtree.py \
      --root ./data --fanout 4 --tau 256 --pairs 20 --knn 10 --knn-cands 200 --use-metis

Requires:
  pip install networkx numpy torch torch-geometric
  # optional for better partitioning:
  pip install pymetis

Notes:
- This is an educational reproduction inspired by TKDE'15 G-Tree (road networks).
- Works on general undirected graphs (we load PubMed from PyTorch Geometric).
- Leaf stores (border -> vertex) distances; non-leaf stores (border -> border).
- SPSP uses assembly; kNN uses a simple best-first traversal.
- For large graphs, tune --fanout and --tau; consider installing PyMetis.
"""

import argparse, time, random, math, heapq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Iterable

import numpy as np
import networkx as nx

# ------------------------------ Dijkstra ------------------------------------

def dijkstra_distances(
    G: nx.Graph,
    sources: Iterable,
    targets: Optional[Set] = None,
    cutoff_subgraph: Optional[Set] = None,
) -> Dict:
    """Single/multi-source Dijkstra distances.
    - If cutoff_subgraph is provided, traversal is restricted to that vertex set.
    - If targets provided, early-stops when all targets settled.
    Returns: dict[source][node] = distance
    """
    if isinstance(sources, (str, int)):
        sources = [sources]
    results: Dict = {}
    for s in sources:
        dist = {s: 0.0}
        pq = [(0.0, s)]
        seen = set()
        remaining = set(targets) if targets is not None else None
        while pq:
            d, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            if remaining is not None and u in remaining:
                remaining.remove(u)
                if not remaining:
                    break
            for v, edata in G[u].items():
                if cutoff_subgraph is not None and v not in cutoff_subgraph:
                    continue
                w = edata.get("weight", 1.0)
                nd = d + float(w)
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        results[s] = dist
    return results

# ------------------------------ G-Tree --------------------------------------

@dataclass
class Node:
    id: int
    level: int
    vertices: Set
    children: List[int] = field(default_factory=list)
    borders: List = field(default_factory=list)
    is_leaf: bool = False
    # Matrices
    #  - Leaf: rows=borders, cols=vertices
    #  - Non-leaf: rows/cols=union of children borders
    dist_mat: Optional[np.ndarray] = None
    row_index: Dict = field(default_factory=dict)
    col_index: Dict = field(default_factory=dict)

class GTree:
    def __init__(self, G: nx.Graph, fanout: int = 2, tau: int = 128, use_metis: bool = True):
        self.G = G.copy()
        self.fanout = max(2, int(fanout))
        self.tau = max(4, int(tau))
        self.use_metis = use_metis
        self.nodes: Dict[int, Node] = {}
        self.root_id: int = 0
        self.vertex_to_leaf: Dict = {}
        self._build()

    # ---------- partitioning ----------
    def _partition_vertices(self, vertices: List, k: int) -> List[List]:
        subG = self.G.subgraph(vertices).copy()
        if self.use_metis:
            try:
                import pymetis
                mapping = {v: i for i, v in enumerate(subG.nodes())}
                invmap = {i: v for v, i in mapping.items()}
                xadj = [0]
                adjncy = []
                for v in subG.nodes():
                    nbrs = [mapping[n] for n in subG.neighbors(v)]
                    adjncy.extend(nbrs)
                    xadj.append(len(adjncy))
                _, parts = pymetis.part_graph(k, xadj=xadj, adjncy=adjncy)
                buckets = [[] for _ in range(k)]
                for i, p in enumerate(parts):
                    buckets[p].append(invmap[i])
                buckets = [b for b in buckets if b]
                while len(buckets) < k:
                    big = max(range(len(buckets)), key=lambda i: len(buckets[i]))
                    half = len(buckets[big]) // 2
                    buckets.append(buckets[big][half:])
                    buckets[big] = buckets[big][:half]
                return buckets
            except Exception:
                pass
        # Fallback: simple greedy BFS splits for balanced parts
        parts = []
        remaining = set(vertices)
        target = max(1, len(vertices) // k)
        while remaining and len(parts) < k - 1:
            seed = next(iter(remaining))
            q = [seed]
            comp = []
            seen = {seed}
            while q and len(comp) < target:
                u = q.pop(0)
                comp.append(u)
                for v in self.G[u]:
                    if v in remaining and v not in seen:
                        seen.add(v)
                        q.append(v)
            parts.append(comp)
            remaining.difference_update(comp)
        if remaining:
            parts.append(list(remaining))
        return parts

    def _find_borders(self, vertices: Set) -> List:
        borders = set()
        Vset = set(vertices)
        for u in vertices:
            for v in self.G[u]:
                if v not in Vset:
                    borders.add(u)
                    break
        return list(borders)

    # ---------- build ----------
    def _build(self):
        # Build topology
        nid = 0
        root_vertices = set(self.G.nodes())
        self.nodes[nid] = Node(id=nid, level=0, vertices=root_vertices)
        frontier = [nid]
        while frontier:
            cur = frontier.pop(0)
            node = self.nodes[cur]
            V = list(node.vertices)
            if len(V) <= self.tau:
                node.is_leaf = True
                node.children = []
            else:
                parts = self._partition_vertices(V, self.fanout)
                node.children = []
                for part in parts:
                    nid += 1
                    self.nodes[nid] = Node(id=nid, level=node.level + 1, vertices=set(part))
                    node.children.append(nid)
                    frontier.append(nid)
        # Borders
        for nid, node in self.nodes.items():
            node.borders = self._find_borders(node.vertices)
        # Leaf matrices (border -> vertex)
        for nid, node in self.nodes.items():
            if node.is_leaf:
                borders = node.borders
                verts = list(node.vertices)
                node.row_index = {b: i for i, b in enumerate(borders)}
                node.col_index = {v: i for i, v in enumerate(verts)}
                if borders and verts:
                    subV = set(node.vertices)
                    res = dijkstra_distances(self.G, borders, targets=set(verts), cutoff_subgraph=subV)
                    mat = np.full((len(borders), len(verts)), np.inf, dtype=float)
                    for bi, b in enumerate(borders):
                        distmap = res[b]
                        for v, di in node.col_index.items():
                            d = distmap.get(v, math.inf)
                            mat[bi, di] = d
                else:
                    mat = np.zeros((len(borders), len(verts)))
                node.dist_mat = mat
        # Non-leaf matrices bottom-up (border -> border) via contracted border graph
        levels = sorted({n.level for n in self.nodes.values()}, reverse=True)
        for lvl in levels:
            for nid, node in [(i, self.nodes[i]) for i in self.nodes if self.nodes[i].level == lvl and not self.nodes[i].is_leaf]:
                child_ids = node.children
                child_borders = []
                for cid in child_ids:
                    child_borders.extend(self.nodes[cid].borders)
                B = list(dict.fromkeys(child_borders))
                node.row_index = {b: i for i, b in enumerate(B)}
                node.col_index = node.row_index
                if not B:
                    node.dist_mat = np.zeros((0, 0))
                    continue
                H = nx.Graph(); H.add_nodes_from(B)
                for cid in child_ids:
                    c = self.nodes[cid]
                    if c.is_leaf:
                        rows = c.borders
                        if len(rows) >= 2 and c.dist_mat.size > 0:
                            for i, bi in enumerate(rows):
                                for j, bj in enumerate(rows):
                                    if i >= j:
                                        continue
                                    dv = np.min(c.dist_mat[c.row_index[bi], :] + c.dist_mat[c.row_index[bj], :])
                                    if np.isfinite(dv):
                                        H.add_edge(bi, bj, weight=float(dv))
                    else:
                        rows = list(c.row_index.keys())
                        M = c.dist_mat
                        for i, bi in enumerate(rows):
                            for j, bj in enumerate(rows):
                                if i >= j:
                                    continue
                                w = float(M[i, j])
                                if math.isfinite(w):
                                    H.add_edge(bi, bj, weight=w)
                # Add any direct original edges among these borders (preserve shortest)
                for u in B:
                    for v, edata in self.G[u].items():
                        if v in B:
                            w = float(edata.get('weight', 1.0))
                            if H.has_edge(u, v):
                                if w < H[u][v]['weight']:
                                    H[u][v]['weight'] = w
                            else:
                                H.add_edge(u, v, weight=w)
                mat = np.full((len(B), len(B)), np.inf, dtype=float)
                for s in B:
                    dmap = dijkstra_distances(H, [s])[s]
                    si = node.row_index[s]
                    for t, di in node.row_index.items():
                        mat[si, di] = float(dmap.get(t, math.inf))
                node.dist_mat = mat
        # Vertex -> leaf map
        for nid, node in self.nodes.items():
            if node.is_leaf:
                for v in node.vertices:
                    self.vertex_to_leaf[v] = nid

    # ---------- helpers ----------
    def _path_nodes_leaf_to_root(self, leaf_id: int) -> List[int]:
        chain = [leaf_id]
        cur = leaf_id
        while True:
            parent = None
            for nid, node in self.nodes.items():
                if leaf_id == self.root_id:
                    break
                if cur in node.children:
                    parent = nid
                    break
            if parent is None:
                break
            chain.append(parent)
            cur = parent
        return chain

    def _lca(self, leaf_u: int, leaf_v: int) -> int:
        A = self._path_nodes_leaf_to_root(leaf_u)
        B = self._path_nodes_leaf_to_root(leaf_v)
        Aset = set(A)
        for x in B:
            if x in Aset:
                return x
        return self.root_id

    def _border_to_border_through_node(self, node_id: int, b1, b2) -> float:
        node = self.nodes[node_id]
        if node.is_leaf:
            if node.dist_mat.size == 0:
                return math.inf
            i = node.row_index.get(b1)
            j = node.row_index.get(b2)
            if i is None or j is None:
                return math.inf
            if i == j:
                return 0.0
            return float(np.min(node.dist_mat[i, :] + node.dist_mat[j, :]))
        else:
            i = node.row_index.get(b1)
            j = node.row_index.get(b2)
            if i is None or j is None:
                return math.inf
            return float(node.dist_mat[i, j])

    # ---------- SPSP ----------
    def spsp(self, u, v) -> float:
        if u not in self.G or v not in self.G:
            return math.inf
        leafu = self.vertex_to_leaf.get(u)
        leafv = self.vertex_to_leaf.get(v)
        if leafu is None or leafv is None:
            return math.inf
        if leafu == leafv:
            leaf = self.nodes[leafu]
            dmap = dijkstra_distances(self.G, [u], targets={v}, cutoff_subgraph=leaf.vertices)[u]
            inside = dmap.get(v, math.inf)
            if leaf.borders:
                ubest = math.inf
                urow = dijkstra_distances(self.G, [u], targets=set(leaf.borders), cutoff_subgraph=leaf.vertices)[u]
                for b1 in leaf.borders:
                    d_u_b1 = urow.get(b1, math.inf)
                    if not math.isfinite(d_u_b1):
                        continue
                    for b2 in leaf.borders:
                        d_b1_b2 = self._border_to_border_through_node(leafu, b1, b2)
                        if not math.isfinite(d_b1_b2):
                            continue
                        if v in leaf.col_index:
                            d_b2_v = float(leaf.dist_mat[leaf.row_index[b2], leaf.col_index[v]])
                        else:
                            d_b2_v = math.inf
                        total = d_u_b1 + d_b1_b2 + d_b2_v
                        if total < ubest:
                            ubest = total
                return min(inside, ubest)
            else:
                return inside
        # assemble across tree (u-leaf -> LCA -> v-leaf)
        lca = self._lca(leafu, leafv)
        # up chain from u
        up_chain = []
        cur = leafu
        while cur != lca:
            up_chain.append(cur)
            parent = None
            for nid, node in self.nodes.items():
                if cur in node.children:
                    parent = nid
                    break
            cur = parent
        up_chain.append(lca)
        # down chain to v
        down_chain = []
        cur = leafv
        while cur != lca:
            down_chain.append(cur)
            parent = None
            for nid, node in self.nodes.items():
                if cur in node.children:
                    parent = nid
                    break
            cur = parent
        down_chain.append(lca)
        down_chain = list(reversed(down_chain))
        # start dp: u to borders of leafu
        leafU = self.nodes[leafu]
        u2b = dijkstra_distances(self.G, [u], targets=set(leafU.borders), cutoff_subgraph=leafU.vertices)[u]
        dp = {b: u2b.get(b, math.inf) for b in leafU.borders}
        # move up to lca (carry over common borders)
        cur_id = leafu
        for next_id in up_chain[1:]:
            next_node = self.nodes[next_id]
            new_dp = {b: math.inf for b in next_node.borders}
            for bcur, d_u_bcur in dp.items():
                if bcur in next_node.row_index:
                    new_dp[bcur] = min(new_dp[bcur], d_u_bcur)
            dp = new_dp
            cur_id = next_id
        # propagate down to leafv (relax over child border graphs)
        path_down = down_chain[1:]
        for child_id in path_down:
            # parent_id (unused directly here)
            child = self.nodes[child_id]
            new_dp = {b: math.inf for b in child.borders}
            for bpar, d_u_bpar in dp.items():
                if bpar in child.row_index:
                    new_dp[bpar] = min(new_dp[bpar], d_u_bpar)
            if child.is_leaf:
                B = child.borders
                M = child.dist_mat
                for i, bi in enumerate(B):
                    for j, bj in enumerate(B):
                        if i == j:
                            continue
                        w = float(np.min(M[i, :] + M[j, :])) if M.size > 0 else math.inf
                        if math.isfinite(w) and math.isfinite(new_dp[bi]):
                            new_dp[bj] = min(new_dp[bj], new_dp[bi] + w)
            else:
                B = list(child.row_index.keys())
                M = child.dist_mat
                for i, bi in enumerate(B):
                    for j, bj in enumerate(B):
                        if i == j:
                            continue
                        w = float(M[i, j])
                        if math.isfinite(w) and math.isfinite(new_dp[bi]):
                            new_dp[bj] = min(new_dp[bj], new_dp[bi] + w)
            dp = new_dp
        # from v-leaf borders to v
        leafV = self.nodes[leafv]
        res = math.inf
        if leafV.borders:
            d_borders_to_v = dijkstra_distances(self.G, leafV.borders, targets={v}, cutoff_subgraph=leafV.vertices)
            for b, d_u_b in dp.items():
                if b in d_borders_to_v:
                    dv = d_borders_to_v[b].get(v, math.inf)
                    if math.isfinite(d_u_b) and math.isfinite(dv):
                        res = min(res, d_u_b + dv)
        return res

    # ---------- kNN ----------
    def _min_dist_to_node(self, vq, nid: int) -> float:
        node = self.nodes[nid]
        if not node.borders:
            return math.inf
        best = math.inf
        for b in node.borders:
            best = min(best, self.spsp(vq, b))
        return best

    def knn(self, vq, candidates: Iterable, k: int) -> List[Tuple[float, object]]:
        C = set(candidates)
        leaf_to_objs: Dict[int, List] = {}
        for o in C:
            lid = self.vertex_to_leaf.get(o)
            if lid is not None:
                leaf_to_objs.setdefault(lid, []).append(o)
        pq: List[Tuple[float, str, object]] = []
        heapq.heappush(pq, (self._min_dist_to_node(vq, self.root_id), 'node', self.root_id))
        results: List[Tuple[float, object]] = []
        seen_nodes = set()
        while pq and len(results) < k:
            bound, typ, obj = heapq.heappop(pq)
            if typ == 'node':
                nid = obj
                if nid in seen_nodes:
                    continue
                seen_nodes.add(nid)
                node = self.nodes[nid]
                if node.is_leaf:
                    for o in leaf_to_objs.get(nid, []):
                        d = self.spsp(vq, o)
                        heapq.heappush(pq, (d, 'obj', o))
                else:
                    for cid in node.children:
                        b = self._min_dist_to_node(vq, cid)
                        heapq.heappush(pq, (b, 'node', cid))
            else:
                results.append((bound, obj))
        return results

# ------------------------------ PubMed loader -------------------------------

def load_pubmed_undirected(root: str = "./data") -> nx.Graph:
    try:
        from torch_geometric.datasets import Planetoid
    except Exception as e:
        raise RuntimeError(
            "Please install torch and torch-geometric: pip install torch torch-geometric"
        )
    ds = Planetoid(root=root, name="Cora")
    data = ds[0]
    edge_index = data.edge_index.cpu().numpy()  # [2, E]
    num_nodes = int(data.num_nodes)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    E = edge_index.shape[1]
    for i in range(E):
        u = int(edge_index[0, i]); v = int(edge_index[1, i])
        if u == v:
            continue
        G.add_edge(u, v, weight=1.0)
    return G

# ------------------------------ CLI / Main ----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='./data', help='dataset root for torch-geometric')
    ap.add_argument('--fanout', type=int, default=4)
    ap.add_argument('--tau', type=int, default=256)
    ap.add_argument('--use-metis', action='store_true')
    ap.add_argument('--pairs', type=int, default=20, help='num random SPSP pairs to test')
    ap.add_argument('--knn', type=int, default=10, help='k for kNN test')
    ap.add_argument('--knn-cands', type=int, default=200, help='candidate pool size for kNN test')
    args = ap.parse_args()

    print("[1/4] Loading PubMed via PyG …")
    t0 = time.time()
    G = load_pubmed_undirected(args.root)
    t1 = time.time()
    print(f"Loaded graph: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}, time={t1-t0:.2f}s")

    print(f"[2/4] Building G-tree (fanout={args.fanout}, tau={args.tau}, use_metis={args.use_metis}) …")
    tb = time.time()
    gtree = GTree(G, fanout=args.fanout, tau=args.tau, use_metis=args.use_metis)
    te = time.time()
    num_nodes = len(gtree.nodes)
    num_leaves = sum(1 for n in gtree.nodes.values() if n.is_leaf)
    print(f"G-tree built in {te-tb:.2f}s, tree-nodes={num_nodes}, leaves={num_leaves}")

    print("[3/4] SPSP random-pairs benchmark …")
    V = list(G.nodes())
    pairs = [tuple(random.sample(V, 2)) for _ in range(args.pairs)]
    t_spsp = []
    for (u, v) in pairs:
        t0 = time.time(); d = gtree.spsp(u, v); t1 = time.time()
        t_spsp.append(t1 - t0)
    print(f"SPSP avg={np.mean(t_spsp):.4f}s, median={np.median(t_spsp):.4f}s (over {len(pairs)} pairs)")

    print("[4/4] kNN benchmark …")
    vq = random.choice(V)
    cand_pool = random.sample(V, min(args.knn_cands, len(V)))
    t0 = time.time(); res = gtree.knn(vq, cand_pool, k=args.knn); t1 = time.time()
    print(f"kNN time={t1-t0:.3f}s, query={vq}, top{args.knn}:")
    for d, o in res:
        print(f"  d={d:.2f}, node={o}")

if __name__ == '__main__':
    main()
