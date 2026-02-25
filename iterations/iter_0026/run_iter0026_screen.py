from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import cdist
from scipy.stats import combine_pvalues
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


ITER_DIR = Path("iterations/iter_0026")
ITER_DIR.mkdir(parents=True, exist_ok=True)

SCGPT_RUNS_BY_DOMAIN: dict[str, dict[str, Path]] = {
    "immune": {
        "seed42_main": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle4_immune_main"
        ),
        "seed43": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle4_immune_seed43"
        ),
        "seed44": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle4_immune_seed44"
        ),
    },
    "lung": {
        "seed42_main": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle6_lung_main"
        ),
        "seed43": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle6_lung_seed43"
        ),
        "seed44": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle6_lung_seed44"
        ),
    },
    "external_lung": {
        "seed42_main": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle7_external_lung_main"
        ),
        "seed43": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle7_external_lung_seed43"
        ),
        "seed44": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle7_external_lung_seed44"
        ),
    },
}

GENEFORMER_EDGE_BY_DOMAIN = {
    "immune": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/"
        "cycle12_geneformer_immune_bootstrap/geneformer_edge_dataset.tsv"
    ),
    "lung": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/"
        "cycle12_geneformer_lung_bootstrap/geneformer_edge_dataset.tsv"
    ),
    "external_lung": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/"
        "cycle12_geneformer_external_lung_bootstrap/geneformer_edge_dataset.tsv"
    ),
}

DOROTHEA_PATH = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
    "single_cell_mechinterp/external/networks/dorothea_human.tsv"
)
GENE2GO_PATH = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
    "single_cell_mechinterp/data/perturb/gene2go_all.pkl"
)
OMNIPATH_INTERACTIONS_PATH = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
    "network_inference/data/omnipath_interactions.tsv"
)
STRING_CACHE_PATH = Path("iterations/iter_0020/h43_string_network_api.tsv")

# H61 (graph_topology): curvature + assortativity surrogate screen.
H61_LAYERS = [7, 11]
H61_GENE_CAP = 160
H61_KNN = 10
H61_NULL_PERM = 24

# H62 (cross_model_alignment): biologically anchored contrastive alignment.
H62_LAYERS = [7, 11]
H62_GENE_CAP = 220
H62_KNN = 10
H62_NULL_PERM = 24
H62_ANCHOR_QUANTILE = 0.85
H62_MIN_ANCHORS_PER_DOMAIN = 60

# H63 (intrinsic_dimensionality): layer-transition ID-gradient screen.
H63_TRANSITIONS = [(0, 3), (3, 7), (7, 11)]
H63_GENE_CAP = 170
H63_NEIGHBORS = 14
H63_NULL_PERM = 24


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    if np.unique(scores).size < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def safe_fisher_p(pvals: np.ndarray) -> float:
    values = np.asarray(pvals, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    values = np.clip(values, 1e-12, 1.0)
    _, pvalue = combine_pvalues(values, method="fisher")
    return float(pvalue)


def empirical_upper_tail_p(observed: float, null_values: np.ndarray) -> float:
    if not np.isfinite(observed):
        return float("nan")
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float((1 + np.sum(values >= observed)) / (values.size + 1))


def build_split_masks(edge_df: pd.DataFrame) -> dict[str, np.ndarray]:
    source_threshold = float(edge_df["source_idx"].median())
    target_threshold = float(edge_df["target_idx"].median())
    return {
        "source_disjoint": edge_df["source_idx"].to_numpy(dtype=float) <= source_threshold,
        "target_disjoint": edge_df["target_idx"].to_numpy(dtype=float) > target_threshold,
    }


def select_top_genes(edge_df: pd.DataFrame, gene_cap: int) -> list[int]:
    counts: dict[int, int] = {}
    for col in ["source_idx", "target_idx"]:
        for value, count in edge_df[col].value_counts().items():
            key = int(value)
            counts[key] = counts.get(key, 0) + int(count)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [gene_idx for gene_idx, _ in ranked[:gene_cap]]


def reduce_points(points: np.ndarray, n_components: int, random_state: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    points = points - points.mean(axis=0, keepdims=True)
    max_comp = min(n_components, points.shape[0] - 1, points.shape[1])
    if max_comp < 4:
        raise RuntimeError(f"Too few PCA components: {max_comp}")
    return PCA(
        n_components=max_comp,
        svd_solver="randomized",
        random_state=random_state,
    ).fit_transform(points)


def build_knn_edge_array(points: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    n_points = points.shape[0]
    k = max(2, min(n_neighbors, n_points - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(points)
    distances, indices = nbrs.kneighbors(points)

    edge_dist: dict[tuple[int, int], float] = {}
    for i in range(n_points):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            u, v = sorted((int(i), int(j)))
            if u == v:
                continue
            dval = float(dist)
            if (u, v) not in edge_dist or dval < edge_dist[(u, v)]:
                edge_dist[(u, v)] = dval

    edges = np.array(list(edge_dist.keys()), dtype=int)
    dists = np.array(list(edge_dist.values()), dtype=float)
    return edges, dists


def connect_knn_graph(points: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Build a connected weighted kNN graph by adding minimum bridges across components."""
    n_points = points.shape[0]
    k = max(2, min(n_neighbors, n_points - 1))

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(points)
    distances, indices = nbrs.kneighbors(points)

    knn_dist = np.full((n_points, n_points), np.inf, dtype=np.float64)
    np.fill_diagonal(knn_dist, 0.0)
    for src in range(n_points):
        for dist, tgt in zip(distances[src, 1:], indices[src, 1:]):
            i = int(src)
            j = int(tgt)
            d = float(dist)
            if d < knn_dist[i, j]:
                knn_dist[i, j] = d
                knn_dist[j, i] = d

    graph = csr_matrix(np.isfinite(knn_dist).astype(np.int8))
    n_components, labels = connected_components(graph, directed=False)

    while n_components > 1:
        best_i = -1
        best_j = -1
        best_dist = float("inf")
        component_ids = np.unique(labels)
        for comp_a_idx in range(component_ids.size):
            nodes_a = np.where(labels == component_ids[comp_a_idx])[0]
            for comp_b_idx in range(comp_a_idx + 1, component_ids.size):
                nodes_b = np.where(labels == component_ids[comp_b_idx])[0]
                local = cdist(points[nodes_a], points[nodes_b], metric="euclidean")
                flat_idx = int(np.argmin(local))
                d = float(local.ravel()[flat_idx])
                if d < best_dist:
                    pos_a, pos_b = np.unravel_index(flat_idx, local.shape)
                    best_i = int(nodes_a[pos_a])
                    best_j = int(nodes_b[pos_b])
                    best_dist = d
        if best_i < 0 or best_j < 0:
            break
        knn_dist[best_i, best_j] = best_dist
        knn_dist[best_j, best_i] = best_dist
        graph = csr_matrix(np.isfinite(knn_dist).astype(np.int8))
        n_components, labels = connected_components(graph, directed=False)

    return knn_dist


def geodesic_distance_matrix(points: np.ndarray, n_neighbors: int) -> np.ndarray:
    knn_dist = connect_knn_graph(points=points, n_neighbors=n_neighbors)
    geodesic = shortest_path(knn_dist, directed=False, unweighted=False)
    if np.isinf(geodesic).any():
        geodesic = cdist(points, points, metric="euclidean")
    return geodesic.astype(np.float64)


def degree_bins(values: np.ndarray, max_bins: int = 5) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return np.array([], dtype=int)
    q = min(max_bins, max(1, int(np.sqrt(x.size))))
    if q <= 1:
        return np.zeros(x.size, dtype=int)
    ranks = pd.Series(x).rank(method="average", pct=True).to_numpy(dtype=float)
    bins = np.minimum((ranks * q).astype(int), q - 1)
    return bins.astype(int)


def stratified_shuffle(values: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(values)
    s = np.asarray(strata, dtype=int)
    out = x.copy()
    for g in np.unique(s):
        idx = np.where(s == g)[0]
        if idx.size > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd < 1e-8:
        return np.zeros_like(arr, dtype=float)
    return (arr - mu) / sd


def row_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def orthogonal_procrustes_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = x.T @ y
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    return u @ vt


def random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def cross_model_edge_scores(
    x_model: np.ndarray,
    y_model: np.ndarray,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
) -> np.ndarray:
    s_to_t = np.sum(x_model[src_idx] * y_model[tgt_idx], axis=1)
    t_to_s = np.sum(x_model[tgt_idx] * y_model[src_idx], axis=1)
    return (0.5 * (s_to_t + t_to_s)).astype(float)


def load_dorothea_score_map() -> dict[tuple[str, str], int]:
    dorothea = pd.read_csv(DOROTHEA_PATH, sep="\t")
    dorothea["source"] = dorothea["source"].astype(str).str.upper()
    dorothea["target"] = dorothea["target"].astype(str).str.upper()
    confidence_map = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
    dorothea["confidence_score"] = (
        dorothea["confidence"].astype(str).str.upper().map(confidence_map).fillna(0).astype(int)
    )
    best = dorothea.groupby(["source", "target"], as_index=False)["confidence_score"].max()
    return {
        (str(row.source), str(row.target)): int(row.confidence_score)
        for row in best.itertuples(index=False)
    }


def load_omnipath_pairs() -> set[tuple[str, str]]:
    omni = pd.read_csv(OMNIPATH_INTERACTIONS_PATH, sep="\t")
    required = {"source_genesymbol", "target_genesymbol"}
    if not required.issubset(omni.columns):
        return set()
    source = omni["source_genesymbol"].astype(str).str.upper()
    target = omni["target_genesymbol"].astype(str).str.upper()
    return set(zip(source, target))


def load_gene2go_upper() -> dict[str, set[str]]:
    with open(GENE2GO_PATH, "rb") as handle:
        raw = pickle.load(handle)
    result: dict[str, set[str]] = {}
    for gene, terms in raw.items():
        if not isinstance(gene, str):
            continue
        gene_upper = gene.upper()
        if gene_upper not in result:
            result[gene_upper] = set()
        if isinstance(terms, (set, list, tuple)):
            for term in terms:
                if isinstance(term, str) and term.startswith("GO:"):
                    result[gene_upper].add(term)
    return result


def load_string_scores_from_cache(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    df = pd.read_csv(path, sep="\t")
    required = {"preferredName_A", "preferredName_B", "score"}
    if not required.issubset(df.columns):
        return {}

    mapping: dict[tuple[str, str], float] = {}
    for row in df.itertuples(index=False):
        src = str(getattr(row, "preferredName_A")).upper()
        tgt = str(getattr(row, "preferredName_B")).upper()
        score = float(getattr(row, "score"))
        if not np.isfinite(score):
            continue
        value = float(np.clip(score, 0.0, 1.0))
        ab = (src, tgt)
        ba = (tgt, src)
        mapping[ab] = max(value, mapping.get(ab, 0.0))
        mapping[ba] = max(value, mapping.get(ba, 0.0))
    return mapping


def support_score_directed(
    src: str,
    tgt: str,
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> float:
    src_u = src.upper()
    tgt_u = tgt.upper()

    dorothea_norm = float(dorothea_map.get((src_u, tgt_u), 0)) / 4.0
    omnipath_support = float((src_u, tgt_u) in omnipath_pairs)
    string_score = float(string_map.get((src_u, tgt_u), 0.0))

    go_src = gene2go_upper.get(src_u, set())
    go_tgt = gene2go_upper.get(tgt_u, set())
    union = len(go_src | go_tgt)
    go_jaccard = float(len(go_src & go_tgt) / union) if union > 0 else 0.0

    return float(0.40 * dorothea_norm + 0.25 * omnipath_support + 0.20 * string_score + 0.15 * go_jaccard)


def build_support_matrices(
    symbols_upper: list[str],
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(symbols_upper)
    support_undirected = np.zeros((n, n), dtype=np.float64)
    support_directed = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        src = symbols_upper[i]
        for j in range(i + 1, n):
            tgt = symbols_upper[j]
            s_ij = support_score_directed(
                src=src,
                tgt=tgt,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )
            s_ji = support_score_directed(
                src=tgt,
                tgt=src,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )
            support_directed[i, j] = s_ij
            support_directed[j, i] = s_ji
            und = max(s_ij, s_ji)
            support_undirected[i, j] = und
            support_undirected[j, i] = und

    return support_undirected, support_directed


def build_symbol_map(split_edges: pd.DataFrame) -> dict[int, str]:
    symbol_map: dict[int, str] = {}
    for row in split_edges[["source_idx", "source"]].drop_duplicates().itertuples(index=False):
        symbol_map[int(row.source_idx)] = str(row.source).upper()
    for row in split_edges[["target_idx", "target"]].drop_duplicates().itertuples(index=False):
        symbol_map[int(row.target_idx)] = str(row.target).upper()
    return symbol_map


def adjacency_neighbors(n_nodes: int, edges: np.ndarray) -> list[set[int]]:
    neighbors = [set() for _ in range(n_nodes)]
    for u, v in edges:
        iu = int(u)
        iv = int(v)
        if iu == iv:
            continue
        neighbors[iu].add(iv)
        neighbors[iv].add(iu)
    return neighbors


def local_clustering(neighbors: list[set[int]]) -> np.ndarray:
    n = len(neighbors)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        nbrs = list(neighbors[i])
        k = len(nbrs)
        if k < 2:
            out[i] = 0.0
            continue
        links = 0
        for a in range(k - 1):
            na = nbrs[a]
            neigh_a = neighbors[na]
            for b in range(a + 1, k):
                nb = nbrs[b]
                if nb in neigh_a:
                    links += 1
        out[i] = float(2.0 * links / (k * (k - 1)))
    return out


def local_id_two_nn(neighbor_dists: np.ndarray) -> np.ndarray:
    d = np.asarray(neighbor_dists, dtype=float)
    if d.shape[1] < 2:
        return np.ones(d.shape[0], dtype=float)
    r1 = np.clip(d[:, 0], 1e-8, None)
    r2 = np.clip(d[:, 1], 1e-8, None)
    ratio = np.clip(r2 / r1, 1.0 + 1e-8, None)
    id_est = 1.0 / np.log(ratio)
    return np.clip(id_est, 0.1, 200.0)


def local_id_mle(neighbor_dists: np.ndarray) -> np.ndarray:
    d = np.asarray(neighbor_dists, dtype=float)
    k = d.shape[1]
    if k < 3:
        return local_id_two_nn(d)
    rk = np.clip(d[:, -1], 1e-8, None)
    logs = np.log(np.clip(rk[:, None] / np.clip(d[:, :-1], 1e-8, None), 1.0 + 1e-8, None))
    denom = np.sum(logs, axis=1)
    id_est = (k - 1) / np.clip(denom, 1e-8, None)
    return np.clip(id_est, 0.1, 200.0)


def fit_signatures_scgpt(
    layer_points: np.ndarray,
    symbols: list[str],
    random_state: int,
    n_neighbors: int,
) -> pd.DataFrame:
    pts = reduce_points(layer_points, n_components=20, random_state=random_state)
    knn_edges, _ = build_knn_edge_array(points=pts, n_neighbors=n_neighbors)
    neighbors = adjacency_neighbors(pts.shape[0], knn_edges)

    degree = np.array([len(n) for n in neighbors], dtype=float) / max(1, pts.shape[0] - 1)
    clust = local_clustering(neighbors)

    mean_nbr_dist = np.zeros(pts.shape[0], dtype=float)
    for i in range(pts.shape[0]):
        neigh = sorted(neighbors[i])
        if neigh:
            mean_nbr_dist[i] = float(np.mean(np.linalg.norm(pts[neigh] - pts[i], axis=1)))

    k_id = max(4, min(14, pts.shape[0] - 1))
    nbrs = NearestNeighbors(n_neighbors=k_id + 1, metric="euclidean")
    nbrs.fit(pts)
    d_full, _ = nbrs.kneighbors(pts)
    d_local = d_full[:, 1:]
    id_two = local_id_two_nn(d_local)
    id_mle = local_id_mle(d_local)

    sig = np.column_stack(
        [
            degree,
            zscore(-mean_nbr_dist),
            clust,
            zscore(id_two),
            zscore(id_mle),
        ]
    )
    cols = [
        "deg_norm",
        "neg_mean_nbr_dist_z",
        "clustering",
        "id_two_nn_z",
        "id_mle_z",
    ]
    df = pd.DataFrame(sig, columns=cols)
    df["symbol"] = symbols
    return df.set_index("symbol")


def fit_signatures_geneformer(
    gf_df: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:
    index = {sym: i for i, sym in enumerate(symbols)}
    n = len(symbols)
    out_neighbors = [set() for _ in range(n)]
    in_neighbors = [set() for _ in range(n)]

    pos = gf_df.loc[gf_df["label"].astype(int) == 1, ["source", "target"]].copy()
    for row in pos.itertuples(index=False):
        s = str(row.source).upper()
        t = str(row.target).upper()
        if s not in index or t not in index:
            continue
        i = index[s]
        j = index[t]
        if i == j:
            continue
        out_neighbors[i].add(j)
        in_neighbors[j].add(i)

    und_neighbors = [out_neighbors[i] | in_neighbors[i] for i in range(n)]
    clust = local_clustering(und_neighbors)

    out_deg = np.array([len(x) for x in out_neighbors], dtype=float) / max(1, n - 1)
    in_deg = np.array([len(x) for x in in_neighbors], dtype=float) / max(1, n - 1)
    und_deg = np.array([len(x) for x in und_neighbors], dtype=float) / max(1, n - 1)

    reciprocity = np.zeros(n, dtype=float)
    for i in range(n):
        union = out_neighbors[i] | in_neighbors[i]
        inter = out_neighbors[i] & in_neighbors[i]
        reciprocity[i] = float(len(inter) / len(union)) if union else 0.0

    sig = np.column_stack(
        [
            out_deg,
            in_deg,
            und_deg,
            reciprocity,
            clust,
        ]
    )
    cols = [
        "out_deg_norm",
        "in_deg_norm",
        "und_deg_norm",
        "reciprocity",
        "clustering",
    ]
    df = pd.DataFrame(sig, columns=cols)
    df["symbol"] = symbols
    return df.set_index("symbol")


def zscore_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.clip(std, 1e-8, None)
    return mean, std


def zscore_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def shuffle_within_bins(values: np.ndarray, bins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(values).copy()
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        if idx.size > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def compute_graph_topology_features(n_nodes: int, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return degree, local clustering, Forman-like node curvature, and assortativity residual."""
    neighbors = adjacency_neighbors(n_nodes=n_nodes, edges=edges)
    degree = np.array([len(n) for n in neighbors], dtype=float)
    degree_norm = degree / max(1, n_nodes - 1)
    clust = local_clustering(neighbors)

    edge_curv = np.zeros(edges.shape[0], dtype=float)
    for idx, (u, v) in enumerate(edges):
        iu = int(u)
        iv = int(v)
        edge_curv[idx] = 4.0 - degree[iu] - degree[iv]

    node_curv_sum = np.zeros(n_nodes, dtype=float)
    node_curv_cnt = np.zeros(n_nodes, dtype=float)
    for idx, (u, v) in enumerate(edges):
        iu = int(u)
        iv = int(v)
        c = edge_curv[idx]
        node_curv_sum[iu] += c
        node_curv_sum[iv] += c
        node_curv_cnt[iu] += 1.0
        node_curv_cnt[iv] += 1.0
    node_curv = node_curv_sum / np.clip(node_curv_cnt, 1.0, None)

    neighbor_deg_mean = np.zeros(n_nodes, dtype=float)
    for i in range(n_nodes):
        neigh = sorted(neighbors[i])
        if neigh:
            neighbor_deg_mean[i] = float(np.mean(degree[neigh]))
        else:
            neighbor_deg_mean[i] = degree[i]
    assort_resid = degree - neighbor_deg_mean

    return degree_norm, clust, node_curv, assort_resid


def choose_anchor_symbols(
    symbols: list[str],
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
    quantile: float,
) -> tuple[list[str], dict[str, float]]:
    if not symbols:
        return [], {}

    support_und, _ = build_support_matrices(
        symbols_upper=symbols,
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    node_strength = np.max(support_und, axis=1)
    threshold = float(max(0.15, np.quantile(node_strength, quantile)))

    selected = [symbols[i] for i, val in enumerate(node_strength) if val >= threshold]
    if len(selected) < H62_MIN_ANCHORS_PER_DOMAIN:
        rank_idx = np.argsort(-node_strength)
        k = min(H62_MIN_ANCHORS_PER_DOMAIN, len(symbols))
        selected = [symbols[int(i)] for i in rank_idx[:k]]

    strength_map = {symbols[i]: float(node_strength[i]) for i in range(len(symbols))}
    return selected, strength_map


def run_h61_graph_curvature_screen() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_index, (seed_tag, run_dir) in enumerate(run_map.items()):
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            split_masks = build_split_masks(edge_df)

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                top_genes = set(select_top_genes(split_edges, gene_cap=H61_GENE_CAP))
                split_edges = split_edges.loc[
                    split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
                ].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                edge_gene_indices = np.unique(
                    np.concatenate(
                        [
                            split_edges["source_idx"].to_numpy(dtype=int),
                            split_edges["target_idx"].to_numpy(dtype=int),
                        ]
                    )
                )
                if edge_gene_indices.size < 120:
                    continue

                gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
                source_local = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
                target_local = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
                labels = split_edges["label"].to_numpy(dtype=int)

                for layer in H61_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = reduce_points(
                        points,
                        n_components=20,
                        random_state=26_610 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer,
                    )
                    knn_edges, _ = build_knn_edge_array(points=points_pca, n_neighbors=H61_KNN)
                    if knn_edges.shape[0] < 120:
                        continue

                    degree_norm, clust, node_curv, assort_resid = compute_graph_topology_features(
                        n_nodes=edge_gene_indices.size,
                        edges=knn_edges,
                    )

                    edge_dist = np.linalg.norm(points_pca[source_local] - points_pca[target_local], axis=1)
                    base_score = -edge_dist
                    curv_gap = -np.abs(node_curv[source_local] - node_curv[target_local])
                    clust_mean = 0.5 * (clust[source_local] + clust[target_local])
                    assort_gap = -np.abs(assort_resid[source_local] - assort_resid[target_local])

                    topo_score = (
                        zscore(base_score)
                        + 0.65 * zscore(curv_gap)
                        + 0.25 * zscore(clust_mean)
                        + 0.20 * zscore(assort_gap)
                    )

                    auc_base = safe_auc(labels, base_score)
                    auc_topo = safe_auc(labels, topo_score)
                    delta_auc = (
                        float(auc_topo - auc_base)
                        if np.isfinite(auc_topo) and np.isfinite(auc_base)
                        else float("nan")
                    )

                    node_deg = np.bincount(
                        np.concatenate([knn_edges[:, 0], knn_edges[:, 1]]),
                        minlength=edge_gene_indices.size,
                    )
                    node_bins = degree_bins(node_deg, max_bins=6)
                    edge_deg_eval = node_deg[source_local] + node_deg[target_local]
                    edge_bins_eval = degree_bins(edge_deg_eval, max_bins=6)

                    rng = np.random.default_rng(
                        26_611 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer
                    )
                    null_curv = np.empty(H61_NULL_PERM, dtype=float)
                    null_topofeat = np.empty(H61_NULL_PERM, dtype=float)
                    null_label = np.empty(H61_NULL_PERM, dtype=float)

                    for perm_idx in range(H61_NULL_PERM):
                        curv_perm = shuffle_within_bins(node_curv, node_bins, rng=rng)
                        score_curv_perm = (
                            zscore(base_score)
                            + 0.65 * zscore(-np.abs(curv_perm[source_local] - curv_perm[target_local]))
                            + 0.25 * zscore(clust_mean)
                            + 0.20 * zscore(assort_gap)
                        )
                        auc_perm = safe_auc(labels, score_curv_perm)
                        delta_perm = (
                            float(auc_perm - auc_base) if np.isfinite(auc_perm) and np.isfinite(auc_base) else float("nan")
                        )
                        null_curv[perm_idx] = delta_perm
                        null_rows.append(
                            {
                                "null_kind": "curvature_shuffle_within_degree_bins",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_perm),
                            }
                        )

                        clust_perm = shuffle_within_bins(clust, node_bins, rng=rng)
                        assort_perm = shuffle_within_bins(assort_resid, node_bins, rng=rng)
                        score_topo_perm = (
                            zscore(base_score)
                            + 0.65 * zscore(curv_gap)
                            + 0.25 * zscore(0.5 * (clust_perm[source_local] + clust_perm[target_local]))
                            + 0.20 * zscore(-np.abs(assort_perm[source_local] - assort_perm[target_local]))
                        )
                        auc_topo_perm = safe_auc(labels, score_topo_perm)
                        delta_topo_perm = (
                            float(auc_topo_perm - auc_base)
                            if np.isfinite(auc_topo_perm) and np.isfinite(auc_base)
                            else float("nan")
                        )
                        null_topofeat[perm_idx] = delta_topo_perm
                        null_rows.append(
                            {
                                "null_kind": "topology_feature_shuffle_within_degree_bins",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_topo_perm),
                            }
                        )

                        labels_perm = stratified_shuffle(labels, edge_bins_eval, rng=rng).astype(int)
                        auc_topo_lp = safe_auc(labels_perm, topo_score)
                        auc_base_lp = safe_auc(labels_perm, base_score)
                        delta_lp = (
                            float(auc_topo_lp - auc_base_lp)
                            if np.isfinite(auc_topo_lp) and np.isfinite(auc_base_lp)
                            else float("nan")
                        )
                        null_label[perm_idx] = delta_lp
                        null_rows.append(
                            {
                                "null_kind": "label_permutation",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_lp),
                            }
                        )

                    p_curv = empirical_upper_tail_p(delta_auc, null_curv)
                    p_topo = empirical_upper_tail_p(delta_auc, null_topofeat)
                    p_label = empirical_upper_tail_p(delta_auc, null_label)
                    p_best = np.nanmin(np.array([p_curv, p_topo, p_label], dtype=float))

                    rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "n_edges_eval": int(labels.size),
                            "n_positive_eval": int(labels.sum()),
                            "n_nodes_graph": int(edge_gene_indices.size),
                            "auc_topology_score": float(auc_topo),
                            "auc_distance_baseline": float(auc_base),
                            "delta_auc_topology_minus_baseline": float(delta_auc),
                            "mean_node_curvature": float(np.mean(node_curv)),
                            "mean_local_clustering": float(np.mean(clust)),
                            "mean_degree_norm": float(np.mean(degree_norm)),
                            "p_curvature_upper": float(p_curv),
                            "p_topology_feature_upper": float(p_topo),
                            "p_label_upper": float(p_label),
                            "p_best_upper": float(p_best),
                        }
                    )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h61_graph_curvature_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h61_graph_curvature_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    domain_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            domain_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_auc_topology_score": float(group["auc_topology_score"].mean()),
                    "mean_auc_distance_baseline": float(group["auc_distance_baseline"].mean()),
                    "mean_delta_auc_topology_minus_baseline": float(
                        group["delta_auc_topology_minus_baseline"].mean()
                    ),
                    "fraction_delta_positive": float((group["delta_auc_topology_minus_baseline"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    domain_df = pd.DataFrame(domain_rows)
    if not domain_df.empty:
        domain_df = domain_df.sort_values(["domain", "split_regime"])
    domain_path = ITER_DIR / "h61_graph_curvature_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_topology_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_split_fisher_sig": int((domain_df["combined_fisher_p_best"] < 0.05).sum())
        if not domain_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_layer_split": str(by_row_path),
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
        },
    }


def run_h62_anchor_alignment(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    by_row: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    sc_edges_seed42: dict[str, pd.DataFrame] = {}
    sc_layers_seed42: dict[str, np.ndarray] = {}
    gf_edges: dict[str, pd.DataFrame] = {}

    for domain in ["immune", "lung", "external_lung"]:
        sc_run = SCGPT_RUNS_BY_DOMAIN[domain]["seed42_main"]
        sc_edges_seed42[domain] = pd.read_csv(sc_run / "cycle1_edge_dataset.tsv", sep="\t")
        sc_layers_seed42[domain] = np.load(sc_run / "layer_gene_embeddings.npy", mmap_mode="r")
        gf_edges[domain] = pd.read_csv(GENEFORMER_EDGE_BY_DOMAIN[domain], sep="\t")

    selected_symbols_by_domain: dict[str, list[str]] = {}
    anchor_symbols_by_domain: dict[str, list[str]] = {}
    anchor_strength_by_domain: dict[str, dict[str, float]] = {}
    sc_sig_by_domain_layer: dict[tuple[str, int], pd.DataFrame] = {}
    gf_sig_by_domain: dict[str, pd.DataFrame] = {}

    for domain_index, domain in enumerate(["immune", "lung", "external_lung"]):
        sc_df = sc_edges_seed42[domain].copy()
        top_genes = set(select_top_genes(sc_df, gene_cap=H62_GENE_CAP))
        sc_df = sc_df.loc[sc_df["source_idx"].isin(top_genes) & sc_df["target_idx"].isin(top_genes)].copy()
        if sc_df.empty:
            continue

        gene_indices = np.unique(
            np.concatenate(
                [
                    sc_df["source_idx"].to_numpy(dtype=int),
                    sc_df["target_idx"].to_numpy(dtype=int),
                ]
            )
        )
        symbol_map = build_symbol_map(sc_df)
        symbols = [symbol_map[int(g)] for g in gene_indices]
        selected_symbols_by_domain[domain] = symbols

        anchors, strength_map = choose_anchor_symbols(
            symbols=symbols,
            dorothea_map=dorothea_map,
            omnipath_pairs=omnipath_pairs,
            gene2go_upper=gene2go_upper,
            string_map=string_map,
            quantile=H62_ANCHOR_QUANTILE,
        )
        anchor_symbols_by_domain[domain] = anchors
        anchor_strength_by_domain[domain] = strength_map

        gf_sig_by_domain[domain] = fit_signatures_geneformer(gf_edges[domain], symbols)

        for layer in H62_LAYERS:
            points = sc_layers_seed42[domain][layer, gene_indices, :]
            sc_sig_by_domain_layer[(domain, layer)] = fit_signatures_scgpt(
                layer_points=points,
                symbols=symbols,
                random_state=26_620 + domain_index * 10 + layer,
                n_neighbors=H62_KNN,
            )

    domains = ["immune", "lung", "external_lung"]
    for domain_index, target_domain in enumerate(domains):
        source_domains = [d for d in domains if d != target_domain]
        split_masks = build_split_masks(sc_edges_seed42[target_domain])

        for layer in H62_LAYERS:
            x_train_list: list[np.ndarray] = []
            y_train_list: list[np.ndarray] = []
            train_pair_count = 0
            anchor_fraction_values: list[float] = []

            for src_domain in source_domains:
                x_df = gf_sig_by_domain.get(src_domain)
                y_df = sc_sig_by_domain_layer.get((src_domain, layer))
                if x_df is None or y_df is None:
                    continue

                shared = sorted(set(x_df.index) & set(y_df.index))
                if len(shared) < 80:
                    continue

                base_anchor = sorted(set(anchor_symbols_by_domain.get(src_domain, [])) & set(shared))
                if len(base_anchor) < H62_MIN_ANCHORS_PER_DOMAIN:
                    strength_map = anchor_strength_by_domain.get(src_domain, {})
                    shared_sorted = sorted(shared, key=lambda s: -float(strength_map.get(s, 0.0)))
                    k = min(max(H62_MIN_ANCHORS_PER_DOMAIN, 80), len(shared_sorted))
                    base_anchor = shared_sorted[:k]
                if len(base_anchor) < H62_MIN_ANCHORS_PER_DOMAIN:
                    continue

                x_train_list.append(x_df.loc[base_anchor].to_numpy(dtype=float))
                y_train_list.append(y_df.loc[base_anchor].to_numpy(dtype=float))
                train_pair_count += len(base_anchor)
                anchor_fraction_values.append(float(len(base_anchor) / len(shared)))

            if train_pair_count < 120:
                continue

            x_train = np.vstack(x_train_list)
            y_train = np.vstack(y_train_list)
            x_mu, x_sd = zscore_fit(x_train)
            y_mu, y_sd = zscore_fit(y_train)

            x_train_z = row_normalize(zscore_apply(x_train, x_mu, x_sd))
            y_train_z = row_normalize(zscore_apply(y_train, y_mu, y_sd))
            w = orthogonal_procrustes_map(x_train_z, y_train_z)

            x_tgt_df = gf_sig_by_domain.get(target_domain)
            y_tgt_df = sc_sig_by_domain_layer.get((target_domain, layer))
            if x_tgt_df is None or y_tgt_df is None:
                continue

            shared_tgt = sorted(set(x_tgt_df.index) & set(y_tgt_df.index))
            if len(shared_tgt) < 80:
                continue

            x_tgt = x_tgt_df.loc[shared_tgt].to_numpy(dtype=float)
            y_tgt = y_tgt_df.loc[shared_tgt].to_numpy(dtype=float)
            x_tgt_z = row_normalize(zscore_apply(x_tgt, x_mu, x_sd))
            y_tgt_z = row_normalize(zscore_apply(y_tgt, y_mu, y_sd))
            mapped = row_normalize(x_tgt_z @ w)
            base = row_normalize(x_tgt_z)

            align_diag_cos = float(np.mean(np.sum(mapped * y_tgt_z, axis=1)))
            symbol_to_local = {sym: i for i, sym in enumerate(shared_tgt)}

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_df = sc_edges_seed42[target_domain].loc[split_mask].copy()
                split_df["source_u"] = split_df["source"].astype(str).str.upper()
                split_df["target_u"] = split_df["target"].astype(str).str.upper()
                keep = split_df["source_u"].isin(symbol_to_local) & split_df["target_u"].isin(symbol_to_local)
                split_df = split_df.loc[keep].copy()
                if split_df["label"].nunique() < 2 or split_df.shape[0] < 300:
                    continue

                src_idx = split_df["source_u"].map(symbol_to_local).to_numpy(dtype=int)
                tgt_idx = split_df["target_u"].map(symbol_to_local).to_numpy(dtype=int)
                labels = split_df["label"].to_numpy(dtype=int)

                transfer_scores = cross_model_edge_scores(mapped, y_tgt_z, src_idx, tgt_idx)
                baseline_scores = cross_model_edge_scores(base, y_tgt_z, src_idx, tgt_idx)

                auc_transfer = safe_auc(labels, transfer_scores)
                auc_baseline = safe_auc(labels, baseline_scores)
                delta_auc = (
                    float(auc_transfer - auc_baseline)
                    if np.isfinite(auc_transfer) and np.isfinite(auc_baseline)
                    else float("nan")
                )

                rng = np.random.default_rng(26_621 + domain_index * 100 + layer * 10 + split_index)
                null_random_map = np.empty(H62_NULL_PERM, dtype=float)
                null_signature_destroy = np.empty(H62_NULL_PERM, dtype=float)
                null_anchor_shuffle = np.empty(H62_NULL_PERM, dtype=float)

                dim = mapped.shape[1]
                for perm_idx in range(H62_NULL_PERM):
                    rand_map = random_orthogonal(dim, rng)
                    mapped_rand = row_normalize(base @ rand_map)
                    auc_rand = safe_auc(labels, cross_model_edge_scores(mapped_rand, y_tgt_z, src_idx, tgt_idx))
                    delta_rand = (
                        float(auc_rand - auc_baseline)
                        if np.isfinite(auc_rand) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_random_map[perm_idx] = delta_rand
                    null_rows.append(
                        {
                            "null_kind": "random_map_alignment",
                            "domain": target_domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_rand),
                        }
                    )

                    perm_rows = rng.permutation(mapped.shape[0])
                    mapped_perm = mapped[perm_rows]
                    auc_perm = safe_auc(labels, cross_model_edge_scores(mapped_perm, y_tgt_z, src_idx, tgt_idx))
                    delta_perm = (
                        float(auc_perm - auc_baseline)
                        if np.isfinite(auc_perm) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_signature_destroy[perm_idx] = delta_perm
                    null_rows.append(
                        {
                            "null_kind": "signature_destroy_permutation",
                            "domain": target_domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_perm),
                        }
                    )

                    y_train_shuffle = y_train_z[rng.permutation(y_train_z.shape[0])]
                    w_shuffle = orthogonal_procrustes_map(x_train_z, y_train_shuffle)
                    mapped_anchor = row_normalize(x_tgt_z @ w_shuffle)
                    auc_anchor = safe_auc(
                        labels,
                        cross_model_edge_scores(mapped_anchor, y_tgt_z, src_idx, tgt_idx),
                    )
                    delta_anchor = (
                        float(auc_anchor - auc_baseline)
                        if np.isfinite(auc_anchor) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_anchor_shuffle[perm_idx] = delta_anchor
                    null_rows.append(
                        {
                            "null_kind": "anchor_label_shuffle",
                            "domain": target_domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_anchor),
                        }
                    )

                p_random_map = empirical_upper_tail_p(delta_auc, null_random_map)
                p_destroy = empirical_upper_tail_p(delta_auc, null_signature_destroy)
                p_anchor = empirical_upper_tail_p(delta_auc, null_anchor_shuffle)
                p_best = np.nanmin(np.array([p_random_map, p_destroy, p_anchor], dtype=float))

                all_null = np.concatenate([null_random_map, null_signature_destroy, null_anchor_shuffle])
                all_null = all_null[np.isfinite(all_null)]
                null_q95 = float(np.quantile(all_null, 0.95)) if all_null.size else float("nan")
                null_gap = float(delta_auc - null_q95) if np.isfinite(delta_auc) and np.isfinite(null_q95) else float("nan")

                by_row.append(
                    {
                        "domain": target_domain,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "train_domains": "+".join(sorted(source_domains)),
                        "n_train_pairs": int(train_pair_count),
                        "n_shared_target_symbols": int(len(shared_tgt)),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_transfer": float(auc_transfer),
                        "auc_baseline": float(auc_baseline),
                        "delta_auc_transfer_minus_baseline": float(delta_auc),
                        "null_gap_q95": float(null_gap),
                        "null_q95": float(null_q95),
                        "alignment_diag_cosine": align_diag_cos,
                        "mean_anchor_fraction_train": float(np.mean(anchor_fraction_values))
                        if anchor_fraction_values
                        else float("nan"),
                        "p_random_map_upper": float(p_random_map),
                        "p_signature_destroy_upper": float(p_destroy),
                        "p_anchor_shuffle_upper": float(p_anchor),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(by_row)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h62_anchor_alignment_by_domain_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h62_anchor_alignment_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_auc_transfer": float(group["auc_transfer"].mean()),
                    "mean_auc_baseline": float(group["auc_baseline"].mean()),
                    "mean_delta_auc_transfer_minus_baseline": float(
                        group["delta_auc_transfer_minus_baseline"].mean()
                    ),
                    "mean_null_gap_q95": float(group["null_gap_q95"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_transfer_minus_baseline"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                    "mean_alignment_diag_cosine": float(group["alignment_diag_cosine"].mean()),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h62_anchor_alignment_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_transfer_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "mean_null_gap_q95": float(by_row_df["null_gap_q95"].mean()) if not by_row_df.empty else float("nan"),
        "domain_split_fisher_sig": int((summary_df["combined_fisher_p_best"] < 0.05).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_domain_layer_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h63_transition_id_gradient() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    required_layers = sorted({layer for pair in H63_TRANSITIONS for layer in pair})

    for domain_index, (domain, run_map) in enumerate(SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_index, (seed_tag, run_dir) in enumerate(run_map.items()):
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            split_masks = build_split_masks(edge_df)

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                top_genes = set(select_top_genes(split_edges, gene_cap=H63_GENE_CAP))
                split_edges = split_edges.loc[
                    split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
                ].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                edge_gene_indices = np.unique(
                    np.concatenate(
                        [
                            split_edges["source_idx"].to_numpy(dtype=int),
                            split_edges["target_idx"].to_numpy(dtype=int),
                        ]
                    )
                )
                if edge_gene_indices.size < 120:
                    continue

                gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
                source_local = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
                target_local = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
                labels = split_edges["label"].to_numpy(dtype=int)

                node_deg_eval = np.bincount(
                    np.concatenate([source_local, target_local]),
                    minlength=edge_gene_indices.size,
                )
                node_bins = degree_bins(node_deg_eval, max_bins=6)
                edge_deg_eval = node_deg_eval[source_local] + node_deg_eval[target_local]
                edge_deg_bins = degree_bins(edge_deg_eval, max_bins=6)

                layer_cache: dict[int, dict[str, np.ndarray]] = {}
                for layer in required_layers:
                    if layer >= layer_embeddings.shape[0]:
                        continue
                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = reduce_points(
                        points,
                        n_components=20,
                        random_state=26_630 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer,
                    )

                    k = max(4, min(H63_NEIGHBORS, points_pca.shape[0] - 1))
                    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
                    nbrs.fit(points_pca)
                    d_full, _ = nbrs.kneighbors(points_pca)
                    d_local = d_full[:, 1:]
                    id_two = local_id_two_nn(d_local)
                    id_mle = local_id_mle(d_local)
                    id_combined = 0.5 * (id_two + id_mle)

                    geodesic = geodesic_distance_matrix(points_pca, n_neighbors=H63_NEIGHBORS)
                    layer_cache[layer] = {
                        "id": id_combined,
                        "geodesic": geodesic,
                    }

                for t_from, t_to in H63_TRANSITIONS:
                    if t_from not in layer_cache or t_to not in layer_cache:
                        continue

                    id_a = layer_cache[t_from]["id"]
                    id_b = layer_cache[t_to]["id"]
                    geodesic_b = layer_cache[t_to]["geodesic"]

                    edge_geodesic = geodesic_b[source_local, target_local]
                    base_score = -edge_geodesic

                    delta_id = id_b - id_a
                    grad_consistency = -np.abs(delta_id[source_local] - delta_id[target_local])
                    endpoint_shift = -np.abs(
                        np.abs(id_b[source_local] - id_b[target_local])
                        - np.abs(id_a[source_local] - id_a[target_local])
                    )
                    combined_score = zscore(base_score) + 0.80 * zscore(grad_consistency) + 0.20 * zscore(endpoint_shift)

                    auc_base = safe_auc(labels, base_score)
                    auc_combined = safe_auc(labels, combined_score)
                    delta_auc = (
                        float(auc_combined - auc_base)
                        if np.isfinite(auc_combined) and np.isfinite(auc_base)
                        else float("nan")
                    )

                    dist_bins = degree_bins(edge_geodesic, max_bins=6)
                    rng = np.random.default_rng(
                        26_631
                        + domain_index * 1000
                        + seed_index * 100
                        + split_index * 20
                        + t_from * 10
                        + t_to
                    )
                    null_order = np.empty(H63_NULL_PERM, dtype=float)
                    null_endpoint = np.empty(H63_NULL_PERM, dtype=float)
                    null_label = np.empty(H63_NULL_PERM, dtype=float)

                    for perm_idx in range(H63_NULL_PERM):
                        # Layer-order null: per-node swap erases coherent progression direction.
                        swap_mask = rng.random(id_a.shape[0]) < 0.5
                        id_a_perm = id_a.copy()
                        id_b_perm = id_b.copy()
                        id_a_perm[swap_mask] = id_b[swap_mask]
                        id_b_perm[swap_mask] = id_a[swap_mask]

                        delta_perm = id_b_perm - id_a_perm
                        grad_perm = -np.abs(delta_perm[source_local] - delta_perm[target_local])
                        endpoint_perm = -np.abs(
                            np.abs(id_b_perm[source_local] - id_b_perm[target_local])
                            - np.abs(id_a_perm[source_local] - id_a_perm[target_local])
                        )
                        score_perm = zscore(base_score) + 0.80 * zscore(grad_perm) + 0.20 * zscore(endpoint_perm)
                        auc_perm = safe_auc(labels, score_perm)
                        delta_perm_auc = (
                            float(auc_perm - auc_base) if np.isfinite(auc_perm) and np.isfinite(auc_base) else float("nan")
                        )
                        null_order[perm_idx] = delta_perm_auc
                        null_rows.append(
                            {
                                "null_kind": "layer_order_swap_per_gene",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "transition_from": int(t_from),
                                "transition_to": int(t_to),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_perm_auc),
                            }
                        )

                        swapped_target = target_local.copy()
                        for b in np.unique(dist_bins):
                            idx = np.where(dist_bins == b)[0]
                            if idx.size > 1:
                                swapped_target[idx] = rng.permutation(swapped_target[idx])
                        grad_swap = -np.abs(delta_id[source_local] - delta_id[swapped_target])
                        endpoint_swap = -np.abs(
                            np.abs(id_b[source_local] - id_b[swapped_target])
                            - np.abs(id_a[source_local] - id_a[swapped_target])
                        )
                        score_swap = zscore(base_score) + 0.80 * zscore(grad_swap) + 0.20 * zscore(endpoint_swap)
                        auc_swap = safe_auc(labels, score_swap)
                        delta_swap = (
                            float(auc_swap - auc_base) if np.isfinite(auc_swap) and np.isfinite(auc_base) else float("nan")
                        )
                        null_endpoint[perm_idx] = delta_swap
                        null_rows.append(
                            {
                                "null_kind": "endpoint_swap_within_geodesic_bins",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "transition_from": int(t_from),
                                "transition_to": int(t_to),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_swap),
                            }
                        )

                        labels_perm = stratified_shuffle(labels, edge_deg_bins, rng=rng).astype(int)
                        auc_comb_lp = safe_auc(labels_perm, combined_score)
                        auc_base_lp = safe_auc(labels_perm, base_score)
                        delta_lp = (
                            float(auc_comb_lp - auc_base_lp)
                            if np.isfinite(auc_comb_lp) and np.isfinite(auc_base_lp)
                            else float("nan")
                        )
                        null_label[perm_idx] = delta_lp
                        null_rows.append(
                            {
                                "null_kind": "label_permutation",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "transition_from": int(t_from),
                                "transition_to": int(t_to),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_lp),
                            }
                        )

                    p_order = empirical_upper_tail_p(delta_auc, null_order)
                    p_endpoint = empirical_upper_tail_p(delta_auc, null_endpoint)
                    p_label = empirical_upper_tail_p(delta_auc, null_label)
                    p_best = np.nanmin(np.array([p_order, p_endpoint, p_label], dtype=float))

                    rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "transition_from": int(t_from),
                            "transition_to": int(t_to),
                            "transition": f"{t_from}->{t_to}",
                            "n_edges_eval": int(labels.size),
                            "n_positive_eval": int(labels.sum()),
                            "n_nodes_graph": int(edge_gene_indices.size),
                            "auc_transition_id_gradient": float(auc_combined),
                            "auc_geodesic_baseline": float(auc_base),
                            "delta_auc_transition_minus_baseline": float(delta_auc),
                            "mean_delta_id": float(np.mean(delta_id)),
                            "std_delta_id": float(np.std(delta_id)),
                            "mean_abs_edge_delta_id_gap": float(
                                np.mean(np.abs(delta_id[source_local] - delta_id[target_local]))
                            ),
                            "p_order_upper": float(p_order),
                            "p_endpoint_upper": float(p_endpoint),
                            "p_label_upper": float(p_label),
                            "p_best_upper": float(p_best),
                        }
                    )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(
            ["domain", "seed_tag", "split_regime", "transition_from", "transition_to"]
        )
    by_row_path = ITER_DIR / "h63_transition_id_gradient_by_seed_transition_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(
            [
                "null_kind",
                "domain",
                "seed_tag",
                "split_regime",
                "transition_from",
                "transition_to",
                "perm_idx",
            ]
        )
    null_path = ITER_DIR / "h63_transition_id_gradient_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_auc_transition_id_gradient": float(group["auc_transition_id_gradient"].mean()),
                    "mean_auc_geodesic_baseline": float(group["auc_geodesic_baseline"].mean()),
                    "mean_delta_auc_transition_minus_baseline": float(
                        group["delta_auc_transition_minus_baseline"].mean()
                    ),
                    "fraction_delta_positive": float((group["delta_auc_transition_minus_baseline"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h63_transition_id_gradient_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_transition_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_split_fisher_sig": int((summary_df["combined_fisher_p_best"] < 0.05).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_transition_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def main() -> None:
    required_paths = [
        DOROTHEA_PATH,
        GENE2GO_PATH,
        OMNIPATH_INTERACTIONS_PATH,
        STRING_CACHE_PATH,
    ]
    for run_map in SCGPT_RUNS_BY_DOMAIN.values():
        for run_dir in run_map.values():
            required_paths.append(run_dir / "cycle1_edge_dataset.tsv")
            required_paths.append(run_dir / "layer_gene_embeddings.npy")
    for p in GENEFORMER_EDGE_BY_DOMAIN.values():
        required_paths.append(p)

    missing = [str(p) for p in required_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    dorothea_map = load_dorothea_score_map()
    omnipath_pairs = load_omnipath_pairs()
    gene2go_upper = load_gene2go_upper()
    string_map = load_string_scores_from_cache(STRING_CACHE_PATH)

    h61_summary = run_h61_graph_curvature_screen()
    h62_summary = run_h62_anchor_alignment(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h63_summary = run_h63_transition_id_gradient()

    summary = {
        "iteration": "iter_0026",
        "h61": h61_summary,
        "h62": h62_summary,
        "h63": h63_summary,
        "environment": {
            "python_env": "subproject40-topology",
            "string_cache_used": str(STRING_CACHE_PATH),
        },
    }
    summary_path = ITER_DIR / "iter0026_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
