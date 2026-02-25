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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


ITER_DIR = Path("iterations/iter_0028")
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

# H67 / N329: rank-based multiparameter persistence surfaces.
H67_LAYERS = [7, 11]
H67_GENE_CAP = 170
H67_NULL_PERM = 12
H67_DIST_THRESHOLDS = [0.20, 0.35, 0.50, 0.65]
H67_MARGIN_THRESHOLDS = [0.55, 0.70, 0.85]

# H68 / N338: cycle-consistent utility-regularized cross-model mapping.
H68_LAYERS = [7, 11]
H68_GENE_CAP = 220
H68_CODEBOOK_TOKENS = 12
H68_NULL_PERM = 12
H68_MAP_ITERS = 180

# H69 / N335: multiscale geodesic triangle-defect spectrum.
H69_LAYERS = [7, 11]
H69_GENE_CAP = 170
H69_NEIGHBORS = 12
H69_NULL_PERM = 8
H69_TRIANGLE_K = [8, 12, 16]


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


def build_knn_edge_array(points: np.ndarray, n_neighbors: int) -> np.ndarray:
    n_points = points.shape[0]
    k = max(2, min(n_neighbors, n_points - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(points)
    _, indices = nbrs.kneighbors(points)

    edges = set()
    for i in range(n_points):
        for j in indices[i, 1:]:
            u, v = sorted((int(i), int(j)))
            if u != v:
                edges.add((u, v))
    if not edges:
        return np.zeros((0, 2), dtype=int)
    return np.array(sorted(edges), dtype=int)


def connect_knn_graph(points: np.ndarray, n_neighbors: int) -> np.ndarray:
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


def zscore_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.clip(std, 1e-8, None)
    return mean, std


def zscore_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def row_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def shuffle_within_bins(values: np.ndarray, bins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(values).copy()
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        if idx.size > 1:
            out[idx] = rng.permutation(out[idx])
    return out


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
    knn_edges = build_knn_edge_array(points=pts, n_neighbors=n_neighbors)
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


def component_labels_from_upper_mask(
    n_nodes: int,
    upper_i: np.ndarray,
    upper_j: np.ndarray,
    keep_mask: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    idx = np.where(keep_mask)[0]
    if idx.size == 0:
        labels = np.arange(n_nodes, dtype=int)
        return labels, int(n_nodes), 0
    rows = np.concatenate([upper_i[idx], upper_j[idx]])
    cols = np.concatenate([upper_j[idx], upper_i[idx]])
    data = np.ones(rows.shape[0], dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    n_components, labels = connected_components(graph, directed=False)
    return labels.astype(int), int(n_components), int(idx.size)


def two_axis_filtration_connectivity(
    dist_matrix: np.ndarray,
    margin_matrix: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    dist_quantiles: list[float],
    margin_quantiles: list[float],
) -> dict[str, np.ndarray]:
    n = dist_matrix.shape[0]
    upper_i, upper_j = np.triu_indices(n, k=1)
    dist_vals = dist_matrix[upper_i, upper_j]
    margin_vals = margin_matrix[upper_i, upper_j]

    dist_thresholds = [float(np.quantile(dist_vals, q)) for q in dist_quantiles]
    margin_thresholds = [float(np.quantile(margin_vals, q)) for q in margin_quantiles]

    conn_one = np.zeros((len(dist_thresholds), source_local.size), dtype=float)
    cycle_one = np.zeros(len(dist_thresholds), dtype=float)

    for d_idx, d_thr in enumerate(dist_thresholds):
        keep = dist_vals <= d_thr
        labels, n_comp, e_count = component_labels_from_upper_mask(n, upper_i, upper_j, keep)
        conn_one[d_idx] = (labels[source_local] == labels[target_local]).astype(float)
        cycle_one[d_idx] = float(max(0, e_count - n + n_comp) / max(1, n))

    conn_two = np.zeros((len(dist_thresholds) * len(margin_thresholds), source_local.size), dtype=float)
    cycle_two = np.zeros(len(dist_thresholds) * len(margin_thresholds), dtype=float)
    grid_idx = 0
    for d_thr in dist_thresholds:
        d_keep = dist_vals <= d_thr
        for m_thr in margin_thresholds:
            keep = d_keep & (margin_vals >= m_thr)
            labels, n_comp, e_count = component_labels_from_upper_mask(n, upper_i, upper_j, keep)
            conn_two[grid_idx] = (labels[source_local] == labels[target_local]).astype(float)
            cycle_two[grid_idx] = float(max(0, e_count - n + n_comp) / max(1, n))
            grid_idx += 1

    conn_one_frac = conn_one.mean(axis=0)
    conn_two_frac = conn_two.mean(axis=0)
    conn_gain = conn_two_frac - conn_one_frac
    cycle_one_mean = np.repeat(cycle_one.mean(), source_local.size)
    cycle_two_mean = np.repeat(cycle_two.mean(), source_local.size)

    return {
        "conn_one_frac": conn_one_frac,
        "conn_two_frac": conn_two_frac,
        "conn_gain": conn_gain,
        "cycle_one_mean": cycle_one_mean,
        "cycle_two_mean": cycle_two_mean,
    }


def token_affinity_from_edges(
    edge_df: pd.DataFrame,
    token_map: dict[str, int],
    n_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros((n_tokens, n_tokens), dtype=float)
    cnt = np.zeros((n_tokens, n_tokens), dtype=float)
    if edge_df.empty:
        return pos, cnt

    src = edge_df["source"].astype(str).str.upper()
    tgt = edge_df["target"].astype(str).str.upper()
    labels = edge_df["label"].to_numpy(dtype=int)

    for s, t, y in zip(src, tgt, labels):
        if s not in token_map or t not in token_map:
            continue
        i = int(token_map[s])
        j = int(token_map[t])
        cnt[i, j] += 1.0
        pos[i, j] += float(y)

    return pos, cnt


def fit_codebook(x: np.ndarray, n_tokens: int, random_state: int) -> KMeans:
    n = x.shape[0]
    k = min(n_tokens, max(4, n // 25))
    k = min(k, n)
    if k < 2:
        k = min(2, n)
    if k < 2:
        raise RuntimeError("Not enough rows to fit codebook.")
    return KMeans(n_clusters=int(k), random_state=random_state, n_init=20)


def symmetric_global_rank_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    n = values.shape[0]
    out = np.zeros((n, n), dtype=np.float64)
    upper_i, upper_j = np.triu_indices(n, k=1)
    upper_vals = values[upper_i, upper_j]
    ranks = pd.Series(upper_vals).rank(method="average", pct=True).to_numpy(dtype=np.float64)
    out[upper_i, upper_j] = ranks
    out[upper_j, upper_i] = ranks
    return out


def one_axis_rank_connectivity(
    dist_rank_matrix: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    dist_thresholds: list[float],
) -> np.ndarray:
    n = dist_rank_matrix.shape[0]
    upper_i, upper_j = np.triu_indices(n, k=1)
    dist_vals = dist_rank_matrix[upper_i, upper_j]
    conn = np.zeros((len(dist_thresholds), source_local.size), dtype=np.float64)
    for idx, threshold in enumerate(dist_thresholds):
        keep = dist_vals <= float(threshold)
        labels, _, _ = component_labels_from_upper_mask(n, upper_i, upper_j, keep)
        conn[idx] = (labels[source_local] == labels[target_local]).astype(np.float64)
    return conn.mean(axis=0)


def rank_surface_feature_bundle(
    dist_rank_matrix: np.ndarray,
    margin_rank_matrix: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    dist_thresholds: list[float],
    margin_thresholds: list[float],
) -> dict[str, np.ndarray]:
    n = dist_rank_matrix.shape[0]
    upper_i, upper_j = np.triu_indices(n, k=1)
    dist_vals = dist_rank_matrix[upper_i, upper_j]
    margin_vals = margin_rank_matrix[upper_i, upper_j]

    conn_cube = np.zeros(
        (len(dist_thresholds), len(margin_thresholds), source_local.size),
        dtype=np.float64,
    )
    cycle_grid = np.zeros((len(dist_thresholds), len(margin_thresholds)), dtype=np.float64)

    for d_idx, d_thr in enumerate(dist_thresholds):
        dist_keep = dist_vals <= float(d_thr)
        for m_idx, m_thr in enumerate(margin_thresholds):
            keep = dist_keep & (margin_vals >= float(m_thr))
            labels, n_components, e_count = component_labels_from_upper_mask(n, upper_i, upper_j, keep)
            conn_cube[d_idx, m_idx] = (labels[source_local] == labels[target_local]).astype(np.float64)
            cycle_grid[d_idx, m_idx] = float(max(0, e_count - n + n_components) / max(1, n))

    conn_mean = conn_cube.mean(axis=(0, 1))
    conn_var = conn_cube.var(axis=(0, 1))
    conn_high_minus_low_margin = conn_cube[:, -1, :].mean(axis=0) - conn_cube[:, 0, :].mean(axis=0)
    conn_scale_slope = conn_cube[-1, :, :].mean(axis=0) - conn_cube[0, :, :].mean(axis=0)
    cycle_mean = np.repeat(cycle_grid.mean(), source_local.size)

    return {
        "conn_mean": conn_mean,
        "conn_var": conn_var,
        "conn_high_minus_low_margin": conn_high_minus_low_margin,
        "conn_scale_slope": conn_scale_slope,
        "cycle_mean": cycle_mean,
    }


def symbol_positive_incidence(edge_df: pd.DataFrame, symbols: list[str]) -> pd.Series:
    symbol_upper = [str(s).upper() for s in symbols]
    counts = pd.Series(0.0, index=symbol_upper, dtype=float)
    pos_counts = pd.Series(0.0, index=symbol_upper, dtype=float)

    src_u = edge_df["source"].astype(str).str.upper()
    tgt_u = edge_df["target"].astype(str).str.upper()
    labels = edge_df["label"].to_numpy(dtype=int)

    for s, t, y in zip(src_u, tgt_u, labels):
        if s in counts.index:
            counts.loc[s] += 1.0
            pos_counts.loc[s] += float(y)
        if t in counts.index:
            counts.loc[t] += 1.0
            pos_counts.loc[t] += float(y)

    incidence = (pos_counts + 1.0) / np.clip(counts + 2.0, 1.0, None)
    return incidence.astype(float)


def weighted_ridge_map(
    src: np.ndarray,
    dst: np.ndarray,
    weights: np.ndarray,
    l2: float,
) -> np.ndarray:
    x = np.asarray(src, dtype=np.float64)
    y = np.asarray(dst, dtype=np.float64)
    w = np.sqrt(np.clip(np.asarray(weights, dtype=np.float64), 1e-8, None))[:, None]
    xw = x * w
    yw = y * w
    lhs = xw.T @ xw + float(l2) * np.eye(x.shape[1], dtype=np.float64)
    rhs = xw.T @ yw
    return np.linalg.solve(lhs, rhs)


def fit_cycle_consistent_maps(
    x_sc: np.ndarray,
    y_gf: np.ndarray,
    weights: np.ndarray,
    n_iters: int,
    lr: float,
    l2: float,
    cycle_weight: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = np.asarray(x_sc, dtype=np.float64)
    y = np.asarray(y_gf, dtype=np.float64)
    w = np.clip(np.asarray(weights, dtype=np.float64), 1e-8, None)
    w = w / max(float(np.mean(w)), 1e-8)
    w_col = w[:, None]
    n = float(x.shape[0])

    a = weighted_ridge_map(src=y, dst=x, weights=w, l2=l2)
    b = weighted_ridge_map(src=x, dst=y, weights=w, l2=l2)

    for _ in range(int(n_iters)):
        y_to_sc = y @ a
        x_to_gf = x @ b
        y_cycle = y_to_sc @ b
        x_cycle = x_to_gf @ a

        err_a = y_to_sc - x
        err_b = x_to_gf - y
        err_cycle_y = y_cycle - y
        err_cycle_x = x_cycle - x

        grad_a = (
            (y.T @ (w_col * err_a))
            + float(cycle_weight) * (y.T @ (w_col * err_cycle_y) @ b.T)
            + float(cycle_weight) * ((x @ b).T @ (w_col * err_cycle_x))
            + float(l2) * a
        ) / n
        grad_b = (
            (x.T @ (w_col * err_b))
            + float(cycle_weight) * ((y @ a).T @ (w_col * err_cycle_y))
            + float(cycle_weight) * (x.T @ (w_col * err_cycle_x) @ a.T)
            + float(l2) * b
        ) / n

        a -= float(lr) * grad_a
        b -= float(lr) * grad_b

    mapped = y @ a
    mapped_cycle = mapped @ b
    align_rmse = float(np.sqrt(np.mean((mapped - x) ** 2)))
    cycle_rmse = float(np.sqrt(np.mean((mapped_cycle - y) ** 2)))
    return a, b, align_rmse, cycle_rmse


def mean_row_cosine(x: np.ndarray, y: np.ndarray) -> float:
    x_n = row_normalize(np.asarray(x, dtype=np.float64))
    y_n = row_normalize(np.asarray(y, dtype=np.float64))
    return float(np.mean(np.sum(x_n * y_n, axis=1)))


def multiscale_triangle_defect_features(
    geodesic: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    k_values: list[int],
    rng: np.random.Generator | None = None,
    random_third: bool = False,
) -> dict[str, np.ndarray]:
    n_edges = int(source_local.size)
    n_nodes = int(geodesic.shape[0])
    nodes = np.arange(n_nodes, dtype=int)

    q50 = np.zeros((len(k_values), n_edges), dtype=np.float64)
    q90 = np.zeros((len(k_values), n_edges), dtype=np.float64)
    frac_small = np.zeros((len(k_values), n_edges), dtype=np.float64)

    for edge_idx in range(n_edges):
        u = int(source_local[edge_idx])
        v = int(target_local[edge_idx])
        if u == v:
            continue
        base = float(max(geodesic[u, v], 1e-8))
        candidates = nodes[(nodes != u) & (nodes != v)]
        if candidates.size == 0:
            continue

        ordered = None
        if not random_third:
            sums = geodesic[u, candidates] + geodesic[v, candidates]
            ordered = candidates[np.argsort(sums)]

        for k_idx, k in enumerate(k_values):
            k_eff = min(int(k), int(candidates.size))
            if k_eff <= 0:
                continue
            if random_third:
                if rng is None:
                    raise RuntimeError("rng is required when random_third=True")
                selected = rng.choice(candidates, size=k_eff, replace=False)
            else:
                selected = ordered[:k_eff]

            defects = (geodesic[u, selected] + geodesic[v, selected] - base) / base
            defects = np.clip(defects, 0.0, None)
            q50[k_idx, edge_idx] = float(np.quantile(defects, 0.50))
            q90[k_idx, edge_idx] = float(np.quantile(defects, 0.90))
            frac_small[k_idx, edge_idx] = float(np.mean(defects <= 0.05))

    feature_median = q50.mean(axis=0)
    feature_tail = q90.mean(axis=0)
    feature_close_frac = frac_small.mean(axis=0)
    feature_dispersion = (q90 - q50).mean(axis=0)
    if len(k_values) >= 2:
        feature_scale_span = q90[-1] - q90[0]
    else:
        feature_scale_span = np.zeros(n_edges, dtype=np.float64)

    return {
        "median_mean": feature_median,
        "tail_mean": feature_tail,
        "close_frac_mean": feature_close_frac,
        "dispersion_mean": feature_dispersion,
        "scale_span": feature_scale_span,
    }


def run_h67_rank_persistence_surface(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
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

                top_genes = set(select_top_genes(split_edges, gene_cap=H67_GENE_CAP))
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

                symbol_map = build_symbol_map(split_edges)
                symbols = [symbol_map[int(g)] for g in edge_gene_indices]
                support_und, support_dir = build_support_matrices(
                    symbols_upper=symbols,
                    dorothea_map=dorothea_map,
                    omnipath_pairs=omnipath_pairs,
                    gene2go_upper=gene2go_upper,
                    string_map=string_map,
                )
                margin_matrix = np.abs(support_dir - support_dir.T)

                gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
                source_local = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
                target_local = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
                labels = split_edges["label"].to_numpy(dtype=int)

                for layer in H67_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = reduce_points(
                        points,
                        n_components=20,
                        random_state=28_670 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer,
                    )
                    dist_matrix = cdist(points_pca, points_pca, metric="euclidean")
                    dist_rank = symmetric_global_rank_matrix(dist_matrix)
                    margin_rank = symmetric_global_rank_matrix(margin_matrix)

                    rank_bundle = rank_surface_feature_bundle(
                        dist_rank_matrix=dist_rank,
                        margin_rank_matrix=margin_rank,
                        source_local=source_local,
                        target_local=target_local,
                        dist_thresholds=H67_DIST_THRESHOLDS,
                        margin_thresholds=H67_MARGIN_THRESHOLDS,
                    )
                    one_axis_conn = one_axis_rank_connectivity(
                        dist_rank_matrix=dist_rank,
                        source_local=source_local,
                        target_local=target_local,
                        dist_thresholds=H67_DIST_THRESHOLDS,
                    )

                    edge_dist = dist_matrix[source_local, target_local]
                    edge_support = support_und[source_local, target_local]
                    edge_margin = margin_matrix[source_local, target_local]

                    baseline_score = zscore(-edge_dist) + 0.70 * zscore(edge_support) + 0.30 * zscore(edge_margin)
                    one_axis_score = baseline_score + 0.55 * zscore(one_axis_conn)
                    rank_surface_score = (
                        baseline_score
                        + 0.35 * zscore(rank_bundle["conn_mean"])
                        + 0.20 * zscore(rank_bundle["conn_var"])
                        + 0.20 * zscore(rank_bundle["conn_high_minus_low_margin"])
                        + 0.15 * zscore(rank_bundle["conn_scale_slope"])
                        + 0.10 * zscore(rank_bundle["conn_mean"] * edge_margin)
                    )

                    auc_baseline = safe_auc(labels, baseline_score)
                    auc_one_axis = safe_auc(labels, one_axis_score)
                    auc_rank_surface = safe_auc(labels, rank_surface_score)
                    delta_auc = (
                        float(auc_rank_surface - auc_baseline)
                        if np.isfinite(auc_rank_surface) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    delta_vs_one_axis = (
                        float(auc_rank_surface - auc_one_axis)
                        if np.isfinite(auc_rank_surface) and np.isfinite(auc_one_axis)
                        else float("nan")
                    )

                    knn_edges = build_knn_edge_array(points_pca, n_neighbors=10)
                    node_deg = np.bincount(
                        np.concatenate([knn_edges[:, 0], knn_edges[:, 1]])
                        if knn_edges.size
                        else np.array([], dtype=int),
                        minlength=edge_gene_indices.size,
                    )
                    node_bins = degree_bins(node_deg, max_bins=6)
                    edge_deg = node_deg[source_local] + node_deg[target_local]
                    edge_bins = degree_bins(edge_deg, max_bins=6)

                    rng = np.random.default_rng(
                        28_671 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer
                    )
                    null_axis_rank = np.empty(H67_NULL_PERM, dtype=np.float64)
                    null_label = np.empty(H67_NULL_PERM, dtype=np.float64)
                    node_margin_rank_strength = margin_rank.mean(axis=1)

                    for perm_idx in range(H67_NULL_PERM):
                        margin_rank_node_perm = shuffle_within_bins(node_margin_rank_strength, node_bins, rng)
                        margin_rank_perm = 0.5 * (
                            margin_rank_node_perm[:, None] + margin_rank_node_perm[None, :]
                        )
                        perm_bundle = rank_surface_feature_bundle(
                            dist_rank_matrix=dist_rank,
                            margin_rank_matrix=margin_rank_perm,
                            source_local=source_local,
                            target_local=target_local,
                            dist_thresholds=H67_DIST_THRESHOLDS,
                            margin_thresholds=H67_MARGIN_THRESHOLDS,
                        )
                        edge_margin_rank_perm = margin_rank_perm[source_local, target_local]
                        perm_score = (
                            baseline_score
                            + 0.35 * zscore(perm_bundle["conn_mean"])
                            + 0.20 * zscore(perm_bundle["conn_var"])
                            + 0.20 * zscore(perm_bundle["conn_high_minus_low_margin"])
                            + 0.15 * zscore(perm_bundle["conn_scale_slope"])
                            + 0.10 * zscore(perm_bundle["conn_mean"] * edge_margin_rank_perm)
                        )
                        auc_perm = safe_auc(labels, perm_score)
                        delta_perm = (
                            float(auc_perm - auc_baseline)
                            if np.isfinite(auc_perm) and np.isfinite(auc_baseline)
                            else float("nan")
                        )
                        null_axis_rank[perm_idx] = delta_perm
                        null_rows.append(
                            {
                                "null_kind": "axis_rank_permutation_within_degree_bins",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_perm),
                            }
                        )

                        labels_perm = stratified_shuffle(labels, edge_bins, rng).astype(int)
                        auc_lp_rank = safe_auc(labels_perm, rank_surface_score)
                        auc_lp_base = safe_auc(labels_perm, baseline_score)
                        delta_lp = (
                            float(auc_lp_rank - auc_lp_base)
                            if np.isfinite(auc_lp_rank) and np.isfinite(auc_lp_base)
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

                    p_axis = empirical_upper_tail_p(delta_auc, null_axis_rank)
                    p_label = empirical_upper_tail_p(delta_auc, null_label)
                    p_best = np.nanmin(np.array([p_axis, p_label], dtype=float))

                    rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "n_edges_eval": int(labels.size),
                            "n_positive_eval": int(labels.sum()),
                            "n_nodes_graph": int(edge_gene_indices.size),
                            "auc_rank_surface": float(auc_rank_surface),
                            "auc_one_axis": float(auc_one_axis),
                            "auc_directed_baseline": float(auc_baseline),
                            "delta_auc_rank_surface_minus_baseline": float(delta_auc),
                            "delta_auc_rank_surface_minus_one_axis": float(delta_vs_one_axis),
                            "mean_conn_rank_surface": float(np.mean(rank_bundle["conn_mean"])),
                            "mean_conn_margin_contrast": float(
                                np.mean(rank_bundle["conn_high_minus_low_margin"])
                            ),
                            "mean_cycle_rank_surface": float(np.mean(rank_bundle["cycle_mean"])),
                            "p_axis_upper": float(p_axis),
                            "p_label_upper": float(p_label),
                            "p_best_upper": float(p_best),
                        }
                    )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h67_rank_surface_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h67_rank_surface_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_auc_rank_surface": float(group["auc_rank_surface"].mean()),
                    "mean_auc_baseline": float(group["auc_directed_baseline"].mean()),
                    "mean_delta_auc_rank_surface_minus_baseline": float(
                        group["delta_auc_rank_surface_minus_baseline"].mean()
                    ),
                    "mean_delta_auc_rank_surface_minus_one_axis": float(
                        group["delta_auc_rank_surface_minus_one_axis"].mean()
                    ),
                    "fraction_delta_positive": float(
                        (group["delta_auc_rank_surface_minus_baseline"] > 0.0).mean()
                    ),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(
                        safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))
                    ),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h67_rank_surface_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_rank_surface_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "mean_delta_vs_one_axis": float(by_row_df["delta_auc_rank_surface_minus_one_axis"].mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_split_fisher_sig": int((summary_df["combined_fisher_p_best"] < 0.05).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_layer_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h68_cycle_consistent_utility_ot() -> dict[str, object]:
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

    utility_weight_by_domain: dict[str, pd.Series] = {}
    sc_sig_by_domain_layer: dict[tuple[str, int], pd.DataFrame] = {}
    gf_sig_by_domain: dict[str, pd.DataFrame] = {}

    for domain_index, domain in enumerate(["immune", "lung", "external_lung"]):
        sc_df = sc_edges_seed42[domain].copy()
        top_genes = set(select_top_genes(sc_df, gene_cap=H68_GENE_CAP))
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
        utility_weight_by_domain[domain] = symbol_positive_incidence(sc_df, symbols)
        gf_sig_by_domain[domain] = fit_signatures_geneformer(gf_edges[domain], symbols)

        for layer in H68_LAYERS:
            points = sc_layers_seed42[domain][layer, gene_indices, :]
            sc_sig_by_domain_layer[(domain, layer)] = fit_signatures_scgpt(
                layer_points=points,
                symbols=symbols,
                random_state=28_680 + domain_index * 100 + layer,
                n_neighbors=10,
            )

    domains = ["immune", "lung", "external_lung"]
    for domain_index, target_domain in enumerate(domains):
        source_domains = [d for d in domains if d != target_domain]
        split_masks = build_split_masks(sc_edges_seed42[target_domain])

        for layer in H68_LAYERS:
            train_sc_list: list[np.ndarray] = []
            train_gf_list: list[np.ndarray] = []
            train_w_list: list[np.ndarray] = []

            for src_domain in source_domains:
                sc_df = sc_sig_by_domain_layer.get((src_domain, layer))
                gf_df = gf_sig_by_domain.get(src_domain)
                utility_w = utility_weight_by_domain.get(src_domain)
                if sc_df is None or gf_df is None or utility_w is None:
                    continue

                shared = sorted(set(sc_df.index) & set(gf_df.index))
                if len(shared) < 80:
                    continue
                src_sc = sc_df.loc[shared].to_numpy(dtype=float)
                src_gf = gf_df.loc[shared].to_numpy(dtype=float)
                src_w = utility_w.loc[shared].to_numpy(dtype=float)

                train_sc_list.append(src_sc)
                train_gf_list.append(src_gf)
                train_w_list.append(src_w)

            if not train_sc_list or not train_gf_list:
                continue

            train_sc = np.vstack(train_sc_list)
            train_gf = np.vstack(train_gf_list)
            train_w = np.clip(np.concatenate(train_w_list), 0.05, None)
            if min(train_sc.shape[0], train_gf.shape[0]) < 120:
                continue

            sc_mu, sc_sd = zscore_fit(train_sc)
            gf_mu, gf_sd = zscore_fit(train_gf)
            train_sc_z = zscore_apply(train_sc, sc_mu, sc_sd)
            train_gf_z = zscore_apply(train_gf, gf_mu, gf_sd)

            try:
                sc_codebook = fit_codebook(
                    train_sc_z,
                    n_tokens=H68_CODEBOOK_TOKENS,
                    random_state=28_681 + layer,
                )
                gf_codebook = fit_codebook(
                    train_gf_z,
                    n_tokens=H68_CODEBOOK_TOKENS,
                    random_state=28_682 + layer,
                )
            except RuntimeError:
                continue

            sc_codebook.fit(train_sc_z)
            gf_codebook.fit(train_gf_z)
            n_sc = int(sc_codebook.n_clusters)
            n_gf = int(gf_codebook.n_clusters)

            map_a, map_b, train_align_rmse, train_cycle_rmse = fit_cycle_consistent_maps(
                x_sc=train_sc_z,
                y_gf=train_gf_z,
                weights=train_w,
                n_iters=H68_MAP_ITERS,
                lr=0.06,
                l2=0.05,
                cycle_weight=0.45,
            )

            sc_pos = np.ones((n_sc, n_sc), dtype=float)
            sc_cnt = np.full((n_sc, n_sc), 2.0, dtype=float)
            gf_pos = np.ones((n_gf, n_gf), dtype=float)
            gf_cnt = np.full((n_gf, n_gf), 2.0, dtype=float)

            for src_domain in source_domains:
                sc_df_sig = sc_sig_by_domain_layer.get((src_domain, layer))
                gf_df_sig = gf_sig_by_domain.get(src_domain)
                if sc_df_sig is None or gf_df_sig is None:
                    continue

                shared = sorted(set(sc_df_sig.index) & set(gf_df_sig.index))
                if len(shared) < 80:
                    continue

                sc_vals = zscore_apply(sc_df_sig.loc[shared].to_numpy(dtype=float), sc_mu, sc_sd)
                gf_vals = zscore_apply(gf_df_sig.loc[shared].to_numpy(dtype=float), gf_mu, gf_sd)
                sc_tokens = sc_codebook.predict(sc_vals)
                gf_tokens = gf_codebook.predict(gf_vals)

                sc_token_map = {sym: int(tok) for sym, tok in zip(shared, sc_tokens)}
                gf_token_map = {sym: int(tok) for sym, tok in zip(shared, gf_tokens)}

                edge_train = sc_edges_seed42[src_domain].copy()
                sc_pos_add, sc_cnt_add = token_affinity_from_edges(edge_train, sc_token_map, n_tokens=n_sc)
                gf_pos_add, gf_cnt_add = token_affinity_from_edges(edge_train, gf_token_map, n_tokens=n_gf)
                sc_pos += sc_pos_add
                sc_cnt += sc_cnt_add
                gf_pos += gf_pos_add
                gf_cnt += gf_cnt_add

            affinity_sc = sc_pos / np.clip(sc_cnt, 1.0, None)
            affinity_gf = gf_pos / np.clip(gf_cnt, 1.0, None)

            sc_tgt_df = sc_sig_by_domain_layer.get((target_domain, layer))
            gf_tgt_df = gf_sig_by_domain.get(target_domain)
            if sc_tgt_df is None or gf_tgt_df is None:
                continue

            shared_tgt = sorted(set(sc_tgt_df.index) & set(gf_tgt_df.index))
            if len(shared_tgt) < 80:
                continue

            sc_tgt_z = zscore_apply(sc_tgt_df.loc[shared_tgt].to_numpy(dtype=float), sc_mu, sc_sd)
            gf_tgt_z = zscore_apply(gf_tgt_df.loc[shared_tgt].to_numpy(dtype=float), gf_mu, gf_sd)
            mapped_sc = gf_tgt_z @ map_a
            mapped_cycle = mapped_sc @ map_b
            mapped_sc_tokens = sc_codebook.predict(mapped_sc)
            gf_tgt_tokens = gf_codebook.predict(gf_tgt_z)

            aligned_cosine = mean_row_cosine(mapped_sc, sc_tgt_z)
            cycle_cosine = mean_row_cosine(mapped_cycle, gf_tgt_z)
            sym_to_pos = {sym: idx for idx, sym in enumerate(shared_tgt)}

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_df = sc_edges_seed42[target_domain].loc[split_mask].copy()
                split_df["source_u"] = split_df["source"].astype(str).str.upper()
                split_df["target_u"] = split_df["target"].astype(str).str.upper()
                keep = split_df["source_u"].isin(sym_to_pos) & split_df["target_u"].isin(sym_to_pos)
                split_df = split_df.loc[keep].copy()
                if split_df["label"].nunique() < 2 or split_df.shape[0] < 300:
                    continue

                src_sym = split_df["source_u"].to_numpy(dtype=str)
                tgt_sym = split_df["target_u"].to_numpy(dtype=str)
                labels = split_df["label"].to_numpy(dtype=int)
                src_idx = np.array([sym_to_pos[s] for s in src_sym], dtype=int)
                tgt_idx = np.array([sym_to_pos[t] for t in tgt_sym], dtype=int)

                transfer_scores = affinity_sc[mapped_sc_tokens[src_idx], mapped_sc_tokens[tgt_idx]]
                baseline_scores = affinity_gf[gf_tgt_tokens[src_idx], gf_tgt_tokens[tgt_idx]]

                auc_transfer = safe_auc(labels, transfer_scores)
                auc_baseline = safe_auc(labels, baseline_scores)
                delta_auc = (
                    float(auc_transfer - auc_baseline)
                    if np.isfinite(auc_transfer) and np.isfinite(auc_baseline)
                    else float("nan")
                )

                rng = np.random.default_rng(28_683 + domain_index * 100 + layer * 10 + split_index)
                null_random_map = np.empty(H68_NULL_PERM, dtype=float)
                null_anchor_shuffle = np.empty(H68_NULL_PERM, dtype=float)
                null_signature_destroy = np.empty(H68_NULL_PERM, dtype=float)

                for perm_idx in range(H68_NULL_PERM):
                    random_map = rng.normal(size=map_a.shape)
                    random_map *= np.linalg.norm(map_a) / max(np.linalg.norm(random_map), 1e-8)
                    mapped_random = gf_tgt_z @ random_map
                    tokens_random = sc_codebook.predict(mapped_random)
                    auc_rand = safe_auc(
                        labels,
                        affinity_sc[tokens_random[src_idx], tokens_random[tgt_idx]],
                    )
                    delta_rand = (
                        float(auc_rand - auc_baseline)
                        if np.isfinite(auc_rand) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_random_map[perm_idx] = delta_rand
                    null_rows.append(
                        {
                            "null_kind": "random_linear_map",
                            "domain": target_domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_rand),
                        }
                    )

                    perm = rng.permutation(train_sc_z.shape[0])
                    map_a_perm, _, _, _ = fit_cycle_consistent_maps(
                        x_sc=train_sc_z[perm],
                        y_gf=train_gf_z,
                        weights=train_w,
                        n_iters=max(40, H68_MAP_ITERS // 3),
                        lr=0.05,
                        l2=0.05,
                        cycle_weight=0.45,
                    )
                    mapped_anchor_perm = gf_tgt_z @ map_a_perm
                    tokens_anchor_perm = sc_codebook.predict(mapped_anchor_perm)
                    auc_anchor = safe_auc(
                        labels,
                        affinity_sc[tokens_anchor_perm[src_idx], tokens_anchor_perm[tgt_idx]],
                    )
                    delta_anchor = (
                        float(auc_anchor - auc_baseline)
                        if np.isfinite(auc_anchor) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_anchor_shuffle[perm_idx] = delta_anchor
                    null_rows.append(
                        {
                            "null_kind": "anchor_pair_shuffle",
                            "domain": target_domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_anchor),
                        }
                    )

                    token_perm = rng.permutation(mapped_sc_tokens.shape[0])
                    tokens_destroy = mapped_sc_tokens[token_perm]
                    auc_destroy = safe_auc(
                        labels,
                        affinity_sc[tokens_destroy[src_idx], tokens_destroy[tgt_idx]],
                    )
                    delta_destroy = (
                        float(auc_destroy - auc_baseline)
                        if np.isfinite(auc_destroy) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_signature_destroy[perm_idx] = delta_destroy
                    null_rows.append(
                        {
                            "null_kind": "signature_destroy_permutation",
                            "domain": target_domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_destroy),
                        }
                    )

                p_random = empirical_upper_tail_p(delta_auc, null_random_map)
                p_anchor = empirical_upper_tail_p(delta_auc, null_anchor_shuffle)
                p_destroy = empirical_upper_tail_p(delta_auc, null_signature_destroy)
                p_best = np.nanmin(np.array([p_random, p_anchor, p_destroy], dtype=float))

                all_null = np.concatenate([null_random_map, null_anchor_shuffle, null_signature_destroy])
                all_null = all_null[np.isfinite(all_null)]
                null_q95 = float(np.quantile(all_null, 0.95)) if all_null.size else float("nan")
                null_gap = (
                    float(delta_auc - null_q95)
                    if np.isfinite(delta_auc) and np.isfinite(null_q95)
                    else float("nan")
                )

                by_row.append(
                    {
                        "domain": target_domain,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "train_domains": "+".join(sorted(source_domains)),
                        "n_train_genes": int(train_sc.shape[0]),
                        "n_shared_target_symbols": int(len(shared_tgt)),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "n_tokens_sc": int(n_sc),
                        "n_tokens_gf": int(n_gf),
                        "auc_transfer": float(auc_transfer),
                        "auc_baseline": float(auc_baseline),
                        "delta_auc_transfer_minus_baseline": float(delta_auc),
                        "null_q95": float(null_q95),
                        "null_gap_q95": float(null_gap),
                        "mapped_to_sc_cosine": float(aligned_cosine),
                        "cycle_back_cosine": float(cycle_cosine),
                        "train_align_rmse": float(train_align_rmse),
                        "train_cycle_rmse": float(train_cycle_rmse),
                        "p_random_map_upper": float(p_random),
                        "p_anchor_shuffle_upper": float(p_anchor),
                        "p_signature_destroy_upper": float(p_destroy),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(by_row)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h68_cycle_utility_ot_by_domain_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h68_cycle_utility_ot_null_summary.csv"
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
                    "mean_mapped_to_sc_cosine": float(group["mapped_to_sc_cosine"].mean()),
                    "mean_cycle_back_cosine": float(group["cycle_back_cosine"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_transfer_minus_baseline"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(
                        safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))
                    ),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h68_cycle_utility_ot_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_transfer_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "mean_null_gap_q95": float(by_row_df["null_gap_q95"].mean()) if not by_row_df.empty else float("nan"),
        "mean_alignment_cosine": float(by_row_df["mapped_to_sc_cosine"].mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_split_fisher_sig": int((summary_df["combined_fisher_p_best"] < 0.05).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_domain_layer_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h69_multiscale_triangle_defect(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
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

                top_genes = set(select_top_genes(split_edges, gene_cap=H69_GENE_CAP))
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

                symbol_map = build_symbol_map(split_edges)
                symbols = [symbol_map[int(g)] for g in edge_gene_indices]
                _, support_dir = build_support_matrices(
                    symbols_upper=symbols,
                    dorothea_map=dorothea_map,
                    omnipath_pairs=omnipath_pairs,
                    gene2go_upper=gene2go_upper,
                    string_map=string_map,
                )

                gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
                source_local = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
                target_local = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
                labels = split_edges["label"].to_numpy(dtype=int)

                for layer in H69_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = reduce_points(
                        points,
                        n_components=20,
                        random_state=28_690 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer,
                    )
                    geodesic = geodesic_distance_matrix(points_pca, n_neighbors=H69_NEIGHBORS)

                    edge_geodesic = geodesic[source_local, target_local]
                    edge_support = support_dir[source_local, target_local]
                    edge_margin = np.abs(
                        support_dir[source_local, target_local] - support_dir[target_local, source_local]
                    )
                    feature_bundle = multiscale_triangle_defect_features(
                        geodesic=geodesic,
                        source_local=source_local,
                        target_local=target_local,
                        k_values=H69_TRIANGLE_K,
                    )

                    baseline_score = zscore(-edge_geodesic) + 0.75 * zscore(edge_support) + 0.35 * zscore(edge_margin)
                    defect_score = (
                        baseline_score
                        + 0.35 * zscore(-feature_bundle["median_mean"])
                        + 0.25 * zscore(-feature_bundle["tail_mean"])
                        + 0.20 * zscore(feature_bundle["close_frac_mean"])
                        + 0.10 * zscore(-feature_bundle["scale_span"])
                        + 0.10 * zscore(-feature_bundle["dispersion_mean"])
                    )

                    auc_baseline = safe_auc(labels, baseline_score)
                    auc_defect = safe_auc(labels, defect_score)
                    delta_auc = (
                        float(auc_defect - auc_baseline)
                        if np.isfinite(auc_defect) and np.isfinite(auc_baseline)
                        else float("nan")
                    )

                    edge_bins = degree_bins(edge_geodesic, max_bins=6)
                    rng = np.random.default_rng(
                        28_691 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer
                    )
                    null_endpoint_swap = np.empty(H69_NULL_PERM, dtype=np.float64)
                    null_random_third = np.empty(H69_NULL_PERM, dtype=np.float64)
                    null_label = np.empty(H69_NULL_PERM, dtype=np.float64)

                    for perm_idx in range(H69_NULL_PERM):
                        target_swap = target_local.copy()
                        for b in np.unique(edge_bins):
                            idx = np.where(edge_bins == b)[0]
                            if idx.size > 1:
                                target_swap[idx] = rng.permutation(target_swap[idx])
                        swap_bundle = multiscale_triangle_defect_features(
                            geodesic=geodesic,
                            source_local=source_local,
                            target_local=target_swap,
                            k_values=H69_TRIANGLE_K,
                        )
                        swap_score = (
                            baseline_score
                            + 0.35 * zscore(-swap_bundle["median_mean"])
                            + 0.25 * zscore(-swap_bundle["tail_mean"])
                            + 0.20 * zscore(swap_bundle["close_frac_mean"])
                            + 0.10 * zscore(-swap_bundle["scale_span"])
                            + 0.10 * zscore(-swap_bundle["dispersion_mean"])
                        )
                        auc_swap = safe_auc(labels, swap_score)
                        delta_swap = (
                            float(auc_swap - auc_baseline)
                            if np.isfinite(auc_swap) and np.isfinite(auc_baseline)
                            else float("nan")
                        )
                        null_endpoint_swap[perm_idx] = delta_swap
                        null_rows.append(
                            {
                                "null_kind": "endpoint_swap_within_geodesic_bins",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_swap),
                            }
                        )

                        random_bundle = multiscale_triangle_defect_features(
                            geodesic=geodesic,
                            source_local=source_local,
                            target_local=target_local,
                            k_values=H69_TRIANGLE_K,
                            rng=rng,
                            random_third=True,
                        )
                        random_score = (
                            baseline_score
                            + 0.35 * zscore(-random_bundle["median_mean"])
                            + 0.25 * zscore(-random_bundle["tail_mean"])
                            + 0.20 * zscore(random_bundle["close_frac_mean"])
                            + 0.10 * zscore(-random_bundle["scale_span"])
                            + 0.10 * zscore(-random_bundle["dispersion_mean"])
                        )
                        auc_random = safe_auc(labels, random_score)
                        delta_random = (
                            float(auc_random - auc_baseline)
                            if np.isfinite(auc_random) and np.isfinite(auc_baseline)
                            else float("nan")
                        )
                        null_random_third[perm_idx] = delta_random
                        null_rows.append(
                            {
                                "null_kind": "matched_random_third_nodes",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_random),
                            }
                        )

                        labels_perm = stratified_shuffle(labels, edge_bins, rng).astype(int)
                        auc_lp = safe_auc(labels_perm, defect_score)
                        auc_lp_base = safe_auc(labels_perm, baseline_score)
                        delta_lp = (
                            float(auc_lp - auc_lp_base)
                            if np.isfinite(auc_lp) and np.isfinite(auc_lp_base)
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

                    p_swap = empirical_upper_tail_p(delta_auc, null_endpoint_swap)
                    p_random = empirical_upper_tail_p(delta_auc, null_random_third)
                    p_label = empirical_upper_tail_p(delta_auc, null_label)
                    p_best = np.nanmin(np.array([p_swap, p_random, p_label], dtype=float))

                    rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "n_edges_eval": int(labels.size),
                            "n_positive_eval": int(labels.sum()),
                            "n_nodes_graph": int(edge_gene_indices.size),
                            "auc_triangle_defect": float(auc_defect),
                            "auc_directed_geodesic_baseline": float(auc_baseline),
                            "delta_auc_triangle_defect_minus_baseline": float(delta_auc),
                            "mean_triangle_median_defect": float(np.mean(feature_bundle["median_mean"])),
                            "mean_triangle_tail_defect": float(np.mean(feature_bundle["tail_mean"])),
                            "mean_triangle_close_frac": float(np.mean(feature_bundle["close_frac_mean"])),
                            "p_swap_upper": float(p_swap),
                            "p_random_third_upper": float(p_random),
                            "p_label_upper": float(p_label),
                            "p_best_upper": float(p_best),
                        }
                    )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h69_triangle_defect_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h69_triangle_defect_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_auc_triangle_defect": float(group["auc_triangle_defect"].mean()),
                    "mean_auc_directed_geodesic_baseline": float(
                        group["auc_directed_geodesic_baseline"].mean()
                    ),
                    "mean_delta_auc_triangle_defect_minus_baseline": float(
                        group["delta_auc_triangle_defect_minus_baseline"].mean()
                    ),
                    "fraction_delta_positive": float(
                        (group["delta_auc_triangle_defect_minus_baseline"] > 0.0).mean()
                    ),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(
                        safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))
                    ),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h69_triangle_defect_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_triangle_defect_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_split_fisher_sig": int((summary_df["combined_fisher_p_best"] < 0.05).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_layer_split": str(by_row_path),
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

    h67_summary = run_h67_rank_persistence_surface(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h68_summary = run_h68_cycle_consistent_utility_ot()
    h69_summary = run_h69_multiscale_triangle_defect(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    summary = {
        "iteration": "iter_0028",
        "h67": h67_summary,
        "h68": h68_summary,
        "h69": h69_summary,
        "environment": {
            "python_env": "subproject40-topology",
            "string_cache_used": str(STRING_CACHE_PATH),
        },
    }
    summary_path = ITER_DIR / "iter0028_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
