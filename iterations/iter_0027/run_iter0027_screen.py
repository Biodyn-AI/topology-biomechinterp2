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


ITER_DIR = Path("iterations/iter_0027")
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

# H64 / N315: support-margin two-axis filtration screen.
H64_LAYERS = [7, 11]
H64_GENE_CAP = 170
H64_NULL_PERM = 12
H64_DIST_QUANTILES = [0.20, 0.35, 0.50]
H64_MARGIN_QUANTILES = [0.55, 0.70, 0.85]

# H65 / N326: topology codebook transport (cross-model rescue).
H65_LAYERS = [7, 11]
H65_GENE_CAP = 220
H65_CODEBOOK_TOKENS = 12
H65_NULL_PERM = 24

# H66 / N324: ID interactions with directed support/margin.
H66_TRANSITIONS = [(0, 3), (3, 7), (7, 11)]
H66_GENE_CAP = 170
H66_NEIGHBORS = 14
H66_NULL_PERM = 12


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


def run_h64_support_margin_two_axis(
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

                top_genes = set(select_top_genes(split_edges, gene_cap=H64_GENE_CAP))
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

                for layer in H64_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = reduce_points(
                        points,
                        n_components=20,
                        random_state=27_640 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer,
                    )
                    dist_matrix = cdist(points_pca, points_pca, metric="euclidean")

                    filt = two_axis_filtration_connectivity(
                        dist_matrix=dist_matrix,
                        margin_matrix=margin_matrix,
                        source_local=source_local,
                        target_local=target_local,
                        dist_quantiles=H64_DIST_QUANTILES,
                        margin_quantiles=H64_MARGIN_QUANTILES,
                    )

                    edge_dist = dist_matrix[source_local, target_local]
                    edge_support = support_und[source_local, target_local]
                    edge_margin = margin_matrix[source_local, target_local]

                    baseline_score = zscore(-edge_dist) + 0.70 * zscore(edge_support) + 0.30 * zscore(edge_margin)
                    one_param_score = (
                        baseline_score
                        + 0.45 * zscore(filt["conn_one_frac"])
                        + 0.15 * zscore(filt["cycle_one_mean"])
                    )
                    two_param_score = (
                        baseline_score
                        + 0.40 * zscore(filt["conn_two_frac"])
                        + 0.30 * zscore(filt["conn_gain"])
                        + 0.15 * zscore(edge_margin * filt["conn_two_frac"])
                        + 0.15 * zscore(filt["cycle_two_mean"])
                    )

                    auc_baseline = safe_auc(labels, baseline_score)
                    auc_one = safe_auc(labels, one_param_score)
                    auc_two = safe_auc(labels, two_param_score)
                    delta_auc = float(auc_two - auc_baseline) if np.isfinite(auc_two) and np.isfinite(auc_baseline) else float("nan")
                    delta_two_vs_one = float(auc_two - auc_one) if np.isfinite(auc_two) and np.isfinite(auc_one) else float("nan")

                    knn_edges = build_knn_edge_array(points_pca, n_neighbors=10)
                    node_deg = np.bincount(
                        np.concatenate([knn_edges[:, 0], knn_edges[:, 1]]) if knn_edges.size else np.array([], dtype=int),
                        minlength=edge_gene_indices.size,
                    )
                    node_bins = degree_bins(node_deg, max_bins=6)
                    edge_deg = node_deg[source_local] + node_deg[target_local]
                    edge_bins = degree_bins(edge_deg, max_bins=6)

                    node_support_strength = support_und.mean(axis=1)
                    node_margin_strength = margin_matrix.mean(axis=1)

                    rng = np.random.default_rng(
                        27_641 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer
                    )
                    null_margin = np.empty(H64_NULL_PERM, dtype=float)
                    null_support = np.empty(H64_NULL_PERM, dtype=float)
                    null_label = np.empty(H64_NULL_PERM, dtype=float)

                    for perm_idx in range(H64_NULL_PERM):
                        margin_node_perm = shuffle_within_bins(node_margin_strength, node_bins, rng)
                        margin_perm_matrix = 0.5 * (
                            margin_node_perm[:, None] + margin_node_perm[None, :]
                        )
                        filt_margin = two_axis_filtration_connectivity(
                            dist_matrix=dist_matrix,
                            margin_matrix=margin_perm_matrix,
                            source_local=source_local,
                            target_local=target_local,
                            dist_quantiles=H64_DIST_QUANTILES,
                            margin_quantiles=H64_MARGIN_QUANTILES,
                        )
                        edge_margin_perm = margin_perm_matrix[source_local, target_local]
                        score_margin_perm = (
                            zscore(-edge_dist)
                            + 0.70 * zscore(edge_support)
                            + 0.30 * zscore(edge_margin_perm)
                            + 0.40 * zscore(filt_margin["conn_two_frac"])
                            + 0.30 * zscore(filt_margin["conn_gain"])
                            + 0.15 * zscore(edge_margin_perm * filt_margin["conn_two_frac"])
                            + 0.15 * zscore(filt_margin["cycle_two_mean"])
                        )
                        auc_margin_perm = safe_auc(labels, score_margin_perm)
                        delta_margin = (
                            float(auc_margin_perm - auc_baseline)
                            if np.isfinite(auc_margin_perm) and np.isfinite(auc_baseline)
                            else float("nan")
                        )
                        null_margin[perm_idx] = delta_margin
                        null_rows.append(
                            {
                                "null_kind": "margin_shuffle_within_degree_bins",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_margin),
                            }
                        )

                        support_node_perm = shuffle_within_bins(node_support_strength, node_bins, rng)
                        edge_support_perm = 0.5 * (
                            support_node_perm[source_local] + support_node_perm[target_local]
                        )
                        score_support_perm = (
                            zscore(-edge_dist)
                            + 0.70 * zscore(edge_support_perm)
                            + 0.30 * zscore(edge_margin)
                            + 0.40 * zscore(filt["conn_two_frac"])
                            + 0.30 * zscore(filt["conn_gain"])
                            + 0.15 * zscore(edge_margin * filt["conn_two_frac"])
                            + 0.15 * zscore(filt["cycle_two_mean"])
                        )
                        auc_support_perm = safe_auc(labels, score_support_perm)
                        delta_support = (
                            float(auc_support_perm - auc_baseline)
                            if np.isfinite(auc_support_perm) and np.isfinite(auc_baseline)
                            else float("nan")
                        )
                        null_support[perm_idx] = delta_support
                        null_rows.append(
                            {
                                "null_kind": "support_score_shuffle_within_degree_bins",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_support),
                            }
                        )

                        labels_perm = stratified_shuffle(labels, edge_bins, rng).astype(int)
                        auc_two_lp = safe_auc(labels_perm, two_param_score)
                        auc_base_lp = safe_auc(labels_perm, baseline_score)
                        delta_lp = (
                            float(auc_two_lp - auc_base_lp)
                            if np.isfinite(auc_two_lp) and np.isfinite(auc_base_lp)
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

                    p_margin = empirical_upper_tail_p(delta_auc, null_margin)
                    p_support = empirical_upper_tail_p(delta_auc, null_support)
                    p_label = empirical_upper_tail_p(delta_auc, null_label)
                    p_best = np.nanmin(np.array([p_margin, p_support, p_label], dtype=float))

                    rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "n_edges_eval": int(labels.size),
                            "n_positive_eval": int(labels.sum()),
                            "n_nodes_graph": int(edge_gene_indices.size),
                            "auc_two_axis": float(auc_two),
                            "auc_one_axis": float(auc_one),
                            "auc_directed_baseline": float(auc_baseline),
                            "delta_auc_two_axis_minus_baseline": float(delta_auc),
                            "delta_auc_two_axis_minus_one_axis": float(delta_two_vs_one),
                            "mean_conn_frac_two_axis": float(np.mean(filt["conn_two_frac"])),
                            "mean_conn_gain": float(np.mean(filt["conn_gain"])),
                            "mean_edge_margin": float(np.mean(edge_margin)),
                            "p_margin_upper": float(p_margin),
                            "p_support_upper": float(p_support),
                            "p_label_upper": float(p_label),
                            "p_best_upper": float(p_best),
                        }
                    )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h64_support_margin_two_axis_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h64_support_margin_two_axis_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    domain_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            domain_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_auc_two_axis": float(group["auc_two_axis"].mean()),
                    "mean_auc_directed_baseline": float(group["auc_directed_baseline"].mean()),
                    "mean_delta_auc_two_axis_minus_baseline": float(
                        group["delta_auc_two_axis_minus_baseline"].mean()
                    ),
                    "mean_delta_auc_two_axis_minus_one_axis": float(
                        group["delta_auc_two_axis_minus_one_axis"].mean()
                    ),
                    "fraction_delta_positive": float((group["delta_auc_two_axis_minus_baseline"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    domain_df = pd.DataFrame(domain_rows)
    if not domain_df.empty:
        domain_df = domain_df.sort_values(["domain", "split_regime"])
    domain_path = ITER_DIR / "h64_support_margin_two_axis_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_two_axis_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "mean_delta_vs_one_axis": float(by_row_df["delta_auc_two_axis_minus_one_axis"].mean())
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


def run_h65_codebook_transport() -> dict[str, object]:
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

    symbol_set_by_domain: dict[str, list[str]] = {}
    sc_sig_by_domain_layer: dict[tuple[str, int], pd.DataFrame] = {}
    gf_sig_by_domain: dict[str, pd.DataFrame] = {}

    for domain_index, domain in enumerate(["immune", "lung", "external_lung"]):
        sc_df = sc_edges_seed42[domain].copy()
        top_genes = set(select_top_genes(sc_df, gene_cap=H65_GENE_CAP))
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
        symbol_set_by_domain[domain] = symbols

        gf_sig_by_domain[domain] = fit_signatures_geneformer(gf_edges[domain], symbols)
        for layer in H65_LAYERS:
            points = sc_layers_seed42[domain][layer, gene_indices, :]
            sc_sig_by_domain_layer[(domain, layer)] = fit_signatures_scgpt(
                layer_points=points,
                symbols=symbols,
                random_state=27_650 + domain_index * 100 + layer,
                n_neighbors=10,
            )

    domains = ["immune", "lung", "external_lung"]
    for domain_index, target_domain in enumerate(domains):
        source_domains = [d for d in domains if d != target_domain]
        split_masks = build_split_masks(sc_edges_seed42[target_domain])

        for layer in H65_LAYERS:
            train_sc_list: list[np.ndarray] = []
            train_gf_list: list[np.ndarray] = []
            train_meta: list[tuple[str, str]] = []

            for src_domain in source_domains:
                sc_df = sc_sig_by_domain_layer.get((src_domain, layer))
                gf_df = gf_sig_by_domain.get(src_domain)
                if sc_df is None or gf_df is None:
                    continue

                shared = sorted(set(sc_df.index) & set(gf_df.index))
                if len(shared) < 80:
                    continue

                sc_vals = sc_df.loc[shared].to_numpy(dtype=float)
                gf_vals = gf_df.loc[shared].to_numpy(dtype=float)
                train_sc_list.append(sc_vals)
                train_gf_list.append(gf_vals)
                train_meta.extend((src_domain, sym) for sym in shared)

            if not train_sc_list or not train_gf_list:
                continue

            train_sc = np.vstack(train_sc_list)
            train_gf = np.vstack(train_gf_list)
            if min(train_sc.shape[0], train_gf.shape[0]) < 120:
                continue

            sc_mu, sc_sd = zscore_fit(train_sc)
            gf_mu, gf_sd = zscore_fit(train_gf)
            train_sc_z = zscore_apply(train_sc, sc_mu, sc_sd)
            train_gf_z = zscore_apply(train_gf, gf_mu, gf_sd)

            try:
                sc_codebook = fit_codebook(train_sc_z, n_tokens=H65_CODEBOOK_TOKENS, random_state=27_651 + layer)
                gf_codebook = fit_codebook(train_gf_z, n_tokens=H65_CODEBOOK_TOKENS, random_state=27_652 + layer)
            except RuntimeError:
                continue

            sc_codebook.fit(train_sc_z)
            gf_codebook.fit(train_gf_z)
            n_sc = int(sc_codebook.n_clusters)
            n_gf = int(gf_codebook.n_clusters)

            sc_tokens_train = sc_codebook.predict(train_sc_z)
            gf_tokens_train = gf_codebook.predict(train_gf_z)

            transport_counts = np.ones((n_gf, n_sc), dtype=float)
            for gf_t, sc_t in zip(gf_tokens_train, sc_tokens_train):
                transport_counts[int(gf_t), int(sc_t)] += 1.0
            transport = transport_counts / np.clip(transport_counts.sum(axis=1, keepdims=True), 1e-8, None)

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

                sc_edge_train = sc_edges_seed42[src_domain].copy()
                pos_add, cnt_add = token_affinity_from_edges(sc_edge_train, sc_token_map, n_tokens=n_sc)
                sc_pos += pos_add
                sc_cnt += cnt_add

                pos_gf_add, cnt_gf_add = token_affinity_from_edges(sc_edge_train, gf_token_map, n_tokens=n_gf)
                gf_pos += pos_gf_add
                gf_cnt += cnt_gf_add

            affinity_sc = sc_pos / np.clip(sc_cnt, 1.0, None)
            affinity_gf = gf_pos / np.clip(gf_cnt, 1.0, None)
            transfer_matrix = transport @ affinity_sc @ transport.T

            sc_tgt_df = sc_sig_by_domain_layer.get((target_domain, layer))
            gf_tgt_df = gf_sig_by_domain.get(target_domain)
            if sc_tgt_df is None or gf_tgt_df is None:
                continue

            shared_tgt = sorted(set(sc_tgt_df.index) & set(gf_tgt_df.index))
            if len(shared_tgt) < 80:
                continue

            sc_tgt_z = zscore_apply(sc_tgt_df.loc[shared_tgt].to_numpy(dtype=float), sc_mu, sc_sd)
            gf_tgt_z = zscore_apply(gf_tgt_df.loc[shared_tgt].to_numpy(dtype=float), gf_mu, gf_sd)
            sc_tgt_tokens = sc_codebook.predict(sc_tgt_z)
            gf_tgt_tokens = gf_codebook.predict(gf_tgt_z)

            sc_token_map_tgt = {sym: int(tok) for sym, tok in zip(shared_tgt, sc_tgt_tokens)}
            gf_token_map_tgt = {sym: int(tok) for sym, tok in zip(shared_tgt, gf_tgt_tokens)}

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_df = sc_edges_seed42[target_domain].loc[split_mask].copy()
                split_df["source_u"] = split_df["source"].astype(str).str.upper()
                split_df["target_u"] = split_df["target"].astype(str).str.upper()
                keep = split_df["source_u"].isin(gf_token_map_tgt) & split_df["target_u"].isin(gf_token_map_tgt)
                split_df = split_df.loc[keep].copy()
                if split_df["label"].nunique() < 2 or split_df.shape[0] < 300:
                    continue

                src_sym = split_df["source_u"].to_numpy(dtype=str)
                tgt_sym = split_df["target_u"].to_numpy(dtype=str)
                labels = split_df["label"].to_numpy(dtype=int)

                gf_src = np.array([gf_token_map_tgt[s] for s in src_sym], dtype=int)
                gf_tgt = np.array([gf_token_map_tgt[t] for t in tgt_sym], dtype=int)

                transfer_scores = transfer_matrix[gf_src, gf_tgt]
                baseline_scores = affinity_gf[gf_src, gf_tgt]

                auc_transfer = safe_auc(labels, transfer_scores)
                auc_baseline = safe_auc(labels, baseline_scores)
                delta_auc = (
                    float(auc_transfer - auc_baseline)
                    if np.isfinite(auc_transfer) and np.isfinite(auc_baseline)
                    else float("nan")
                )

                rng = np.random.default_rng(27_653 + domain_index * 100 + layer * 10 + split_index)
                null_random_map = np.empty(H65_NULL_PERM, dtype=float)
                null_freq_shuffle = np.empty(H65_NULL_PERM, dtype=float)
                null_signature_destroy = np.empty(H65_NULL_PERM, dtype=float)

                gf_token_freq = np.bincount(gf_tgt_tokens, minlength=n_gf).astype(float)
                gf_token_prob = gf_token_freq / np.clip(gf_token_freq.sum(), 1.0, None)

                for perm_idx in range(H65_NULL_PERM):
                    sc_perm = rng.permutation(n_sc)
                    transport_rand = transport[:, sc_perm]
                    transfer_rand = transport_rand @ affinity_sc @ transport_rand.T
                    auc_rand = safe_auc(labels, transfer_rand[gf_src, gf_tgt])
                    delta_rand = (
                        float(auc_rand - auc_baseline)
                        if np.isfinite(auc_rand) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_random_map[perm_idx] = delta_rand
                    null_rows.append(
                        {
                            "null_kind": "random_codebook_mapping",
                            "domain": target_domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_rand),
                        }
                    )

                    gf_src_freq = rng.choice(n_gf, size=gf_src.size, replace=True, p=gf_token_prob)
                    gf_tgt_freq = rng.choice(n_gf, size=gf_tgt.size, replace=True, p=gf_token_prob)
                    auc_freq = safe_auc(labels, transfer_matrix[gf_src_freq, gf_tgt_freq])
                    delta_freq = (
                        float(auc_freq - auc_baseline)
                        if np.isfinite(auc_freq) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_freq_shuffle[perm_idx] = delta_freq
                    null_rows.append(
                        {
                            "null_kind": "token_frequency_matched_shuffle",
                            "domain": target_domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_freq),
                        }
                    )

                    sym_perm = rng.permutation(shared_tgt)
                    gf_token_perm_map = {
                        sym: int(gf_token_map_tgt[perm_sym]) for sym, perm_sym in zip(shared_tgt, sym_perm)
                    }
                    gf_src_perm = np.array([gf_token_perm_map[s] for s in src_sym], dtype=int)
                    gf_tgt_perm = np.array([gf_token_perm_map[t] for t in tgt_sym], dtype=int)
                    auc_destroy = safe_auc(labels, transfer_matrix[gf_src_perm, gf_tgt_perm])
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
                p_freq = empirical_upper_tail_p(delta_auc, null_freq_shuffle)
                p_destroy = empirical_upper_tail_p(delta_auc, null_signature_destroy)
                p_best = np.nanmin(np.array([p_random, p_freq, p_destroy], dtype=float))

                all_null = np.concatenate([null_random_map, null_freq_shuffle, null_signature_destroy])
                all_null = all_null[np.isfinite(all_null)]
                null_q95 = float(np.quantile(all_null, 0.95)) if all_null.size else float("nan")
                null_gap = float(delta_auc - null_q95) if np.isfinite(delta_auc) and np.isfinite(null_q95) else float("nan")

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
                        "mean_transport_diag": float(np.mean(np.diag(transport[:, : min(n_gf, n_sc)]))),
                        "p_random_map_upper": float(p_random),
                        "p_freq_shuffle_upper": float(p_freq),
                        "p_signature_destroy_upper": float(p_destroy),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(by_row)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h65_codebook_transport_by_domain_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h65_codebook_transport_null_summary.csv"
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
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h65_codebook_transport_domain_summary.csv"
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


def run_h66_id_interaction_only(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    required_layers = sorted({layer for pair in H66_TRANSITIONS for layer in pair})

    for domain_index, (domain, run_map) in enumerate(SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_index, (seed_tag, run_dir) in enumerate(run_map.items()):
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            split_masks = build_split_masks(edge_df)

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                top_genes = set(select_top_genes(split_edges, gene_cap=H66_GENE_CAP))
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

                node_deg_eval = np.bincount(
                    np.concatenate([source_local, target_local]), minlength=edge_gene_indices.size
                )
                node_bins = degree_bins(node_deg_eval, max_bins=6)

                layer_cache: dict[int, dict[str, np.ndarray]] = {}
                for layer in required_layers:
                    if layer >= layer_embeddings.shape[0]:
                        continue
                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = reduce_points(
                        points,
                        n_components=20,
                        random_state=27_660 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer,
                    )
                    k = max(4, min(H66_NEIGHBORS, points_pca.shape[0] - 1))
                    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
                    nbrs.fit(points_pca)
                    d_full, _ = nbrs.kneighbors(points_pca)
                    d_local = d_full[:, 1:]
                    id_two = local_id_two_nn(d_local)
                    id_mle = local_id_mle(d_local)
                    id_combined = 0.5 * (id_two + id_mle)
                    geodesic = geodesic_distance_matrix(points_pca, n_neighbors=H66_NEIGHBORS)

                    layer_cache[layer] = {
                        "id": id_combined,
                        "geodesic": geodesic,
                    }

                for t_from, t_to in H66_TRANSITIONS:
                    if t_from not in layer_cache or t_to not in layer_cache:
                        continue

                    id_a = layer_cache[t_from]["id"]
                    id_b = layer_cache[t_to]["id"]
                    geodesic_b = layer_cache[t_to]["geodesic"]

                    edge_geodesic = geodesic_b[source_local, target_local]
                    edge_support_dir = support_dir[source_local, target_local]
                    edge_margin = np.abs(support_dir[source_local, target_local] - support_dir[target_local, source_local])

                    id_delta = id_b - id_a
                    id_grad = -np.abs(id_delta[source_local] - id_delta[target_local])
                    id_shift = -np.abs(
                        np.abs(id_b[source_local] - id_b[target_local])
                        - np.abs(id_a[source_local] - id_a[target_local])
                    )

                    interaction_1 = edge_margin * id_grad
                    interaction_2 = edge_support_dir * id_shift

                    baseline_score = zscore(-edge_geodesic) + 0.75 * zscore(edge_support_dir) + 0.35 * zscore(edge_margin)
                    interaction_score = baseline_score + 0.60 * zscore(interaction_1) + 0.40 * zscore(interaction_2)

                    auc_base = safe_auc(labels, baseline_score)
                    auc_interaction = safe_auc(labels, interaction_score)
                    delta_auc = (
                        float(auc_interaction - auc_base)
                        if np.isfinite(auc_interaction) and np.isfinite(auc_base)
                        else float("nan")
                    )

                    dist_bins = degree_bins(edge_geodesic, max_bins=6)
                    rng = np.random.default_rng(
                        27_661
                        + domain_index * 1000
                        + seed_index * 100
                        + split_index * 20
                        + t_from * 10
                        + t_to
                    )
                    null_partner = np.empty(H66_NULL_PERM, dtype=float)
                    null_order = np.empty(H66_NULL_PERM, dtype=float)
                    null_label = np.empty(H66_NULL_PERM, dtype=float)

                    for perm_idx in range(H66_NULL_PERM):
                        id_delta_perm = shuffle_within_bins(id_delta, node_bins, rng)
                        id_grad_perm = -np.abs(id_delta_perm[source_local] - id_delta_perm[target_local])
                        interaction_1_perm = edge_margin * id_grad_perm
                        score_partner_perm = baseline_score + 0.60 * zscore(interaction_1_perm) + 0.40 * zscore(interaction_2)
                        auc_partner = safe_auc(labels, score_partner_perm)
                        delta_partner = (
                            float(auc_partner - auc_base)
                            if np.isfinite(auc_partner) and np.isfinite(auc_base)
                            else float("nan")
                        )
                        null_partner[perm_idx] = delta_partner
                        null_rows.append(
                            {
                                "null_kind": "interaction_partner_shuffle",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "transition_from": int(t_from),
                                "transition_to": int(t_to),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_partner),
                            }
                        )

                        swap_mask = rng.random(id_a.shape[0]) < 0.5
                        id_a_perm = id_a.copy()
                        id_b_perm = id_b.copy()
                        id_a_perm[swap_mask] = id_b[swap_mask]
                        id_b_perm[swap_mask] = id_a[swap_mask]
                        id_delta_swap = id_b_perm - id_a_perm
                        id_grad_swap = -np.abs(id_delta_swap[source_local] - id_delta_swap[target_local])
                        id_shift_swap = -np.abs(
                            np.abs(id_b_perm[source_local] - id_b_perm[target_local])
                            - np.abs(id_a_perm[source_local] - id_a_perm[target_local])
                        )
                        score_order_perm = baseline_score + 0.60 * zscore(edge_margin * id_grad_swap) + 0.40 * zscore(
                            edge_support_dir * id_shift_swap
                        )
                        auc_order = safe_auc(labels, score_order_perm)
                        delta_order = (
                            float(auc_order - auc_base)
                            if np.isfinite(auc_order) and np.isfinite(auc_base)
                            else float("nan")
                        )
                        null_order[perm_idx] = delta_order
                        null_rows.append(
                            {
                                "null_kind": "layer_order_permutation_per_gene",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "transition_from": int(t_from),
                                "transition_to": int(t_to),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_order),
                            }
                        )

                        labels_perm = stratified_shuffle(labels, dist_bins, rng).astype(int)
                        auc_inter_lp = safe_auc(labels_perm, interaction_score)
                        auc_base_lp = safe_auc(labels_perm, baseline_score)
                        delta_lp = (
                            float(auc_inter_lp - auc_base_lp)
                            if np.isfinite(auc_inter_lp) and np.isfinite(auc_base_lp)
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

                    p_partner = empirical_upper_tail_p(delta_auc, null_partner)
                    p_order = empirical_upper_tail_p(delta_auc, null_order)
                    p_label = empirical_upper_tail_p(delta_auc, null_label)
                    p_best = np.nanmin(np.array([p_partner, p_order, p_label], dtype=float))

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
                            "auc_id_interaction": float(auc_interaction),
                            "auc_directed_geodesic_baseline": float(auc_base),
                            "delta_auc_id_interaction_minus_baseline": float(delta_auc),
                            "mean_edge_margin": float(np.mean(edge_margin)),
                            "mean_abs_id_delta_gap": float(np.mean(np.abs(id_delta[source_local] - id_delta[target_local]))),
                            "p_partner_upper": float(p_partner),
                            "p_order_upper": float(p_order),
                            "p_label_upper": float(p_label),
                            "p_best_upper": float(p_best),
                        }
                    )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(
            ["domain", "seed_tag", "split_regime", "transition_from", "transition_to"]
        )
    by_row_path = ITER_DIR / "h66_id_interaction_by_seed_transition_split.csv"
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
    null_path = ITER_DIR / "h66_id_interaction_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_auc_id_interaction": float(group["auc_id_interaction"].mean()),
                    "mean_auc_directed_geodesic_baseline": float(group["auc_directed_geodesic_baseline"].mean()),
                    "mean_delta_auc_id_interaction_minus_baseline": float(
                        group["delta_auc_id_interaction_minus_baseline"].mean()
                    ),
                    "fraction_delta_positive": float((group["delta_auc_id_interaction_minus_baseline"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h66_id_interaction_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_id_interaction_minus_baseline"].mean())
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

    h64_summary = run_h64_support_margin_two_axis(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h65_summary = run_h65_codebook_transport()
    h66_summary = run_h66_id_interaction_only(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    summary = {
        "iteration": "iter_0027",
        "h64": h64_summary,
        "h65": h65_summary,
        "h66": h66_summary,
        "environment": {
            "python_env": "subproject40-topology",
            "string_cache_used": str(STRING_CACHE_PATH),
        },
    }
    summary_path = ITER_DIR / "iter0027_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
