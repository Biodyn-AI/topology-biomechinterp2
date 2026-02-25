from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist, pdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import combine_pvalues, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModel


ITER_DIR = Path("iterations/iter_0016")
ITER_DIR.mkdir(parents=True, exist_ok=True)

SCGPT_RUNS_BY_DOMAIN: dict[str, dict[str, Path]] = {
    "immune": {
        "seed42_main": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle4_immune_main"
        ),
        "seed43": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle4_immune_seed43"
        ),
        "seed44": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle4_immune_seed44"
        ),
    },
    "lung": {
        "seed42_main": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle6_lung_main"
        ),
        "seed43": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle6_lung_seed43"
        ),
        "seed44": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle6_lung_seed44"
        ),
    },
    "external_lung": {
        "seed42_main": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle7_external_lung_main"
        ),
        "seed43": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle7_external_lung_seed43"
        ),
        "seed44": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle7_external_lung_seed44"
        ),
    },
}

GENEFORMER_EDGE_BY_DOMAIN = {
    "immune": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "subproject_38_geometric_residual_stream_interpretability/"
        "implementation/outputs/cycle12_geneformer_immune_bootstrap/geneformer_edge_dataset.tsv"
    ),
    "lung": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "subproject_38_geometric_residual_stream_interpretability/"
        "implementation/outputs/cycle12_geneformer_lung_bootstrap/geneformer_edge_dataset.tsv"
    ),
    "external_lung": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "subproject_38_geometric_residual_stream_interpretability/"
        "implementation/outputs/cycle12_geneformer_external_lung_bootstrap/geneformer_edge_dataset.tsv"
    ),
}

PROCESSED_H5AD_BY_DOMAIN = {
    "immune": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "single_cell_mechinterp/outputs/tabula_sapiens_immune_subset_hpn_processed.h5ad"
    ),
    "lung": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "single_cell_mechinterp/outputs/invariant_causal_edges/lung/processed.h5ad"
    ),
    "external_lung": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "single_cell_mechinterp/outputs/invariant_causal_edges/external_lung/processed.h5ad"
    ),
}

CROSS_MODEL_LAYER_BY_DOMAIN = {"immune": 0, "lung": 0, "external_lung": 3}

# H31: diffusion incremental value after covariate adjustment.
H31_LAYERS = [0, 3, 7, 11]
H31_DIFFUSION_TIMES = [1, 2, 4, 8]
H31_GENE_CAP = 260
H31_NULL_PERM = 80
H31_MAX_COEXP_CELLS = 5000

# H32: convexity-deficit / detour geometry screen.
H32_LAYERS = [0, 3, 7, 11]
H32_GENE_CAP = 260
H32_NULL_PERM = 120
H32_SEED = "seed42_main"

# H33: tri-domain cycle-consistent cross-model alignment.
H33_GENE_CAP = 260
H33_NULL_PERM = 160
H33_CYCLE_LAMBDA = 0.45
H33_CYCLE_ITERS = 10


@dataclass
class CoexpressionDomainCache:
    """
    Backed expression cache used for coexpression-aware controls.

    We cap sampled cells for speed because this is a screening loop, not a
    final-scale estimate.
    """

    path: Path

    def __post_init__(self) -> None:
        self.adata = ad.read_h5ad(self.path, backed="r")
        var_names = pd.Index(self.adata.var_names.astype(str)).str.upper()
        self.gene_to_var_idx = {str(g): int(i) for i, g in enumerate(var_names)}

    def close(self) -> None:
        self.adata.file.close()

    def abs_corr_for_genes(
        self,
        gene_symbols_upper: list[str],
        max_cells: int,
        random_state: int,
    ) -> tuple[np.ndarray, float]:
        n_genes = len(gene_symbols_upper)
        corr = np.zeros((n_genes, n_genes), dtype=np.float64)
        if n_genes == 0:
            return corr, 1.0

        found_mask = np.array([g in self.gene_to_var_idx for g in gene_symbols_upper], dtype=bool)
        found_positions = np.where(found_mask)[0]
        missing_fraction = float(1.0 - found_mask.mean())
        if found_positions.size < 2:
            return corr, missing_fraction

        var_idx = [self.gene_to_var_idx[gene_symbols_upper[pos]] for pos in found_positions]
        n_cells = int(self.adata.n_obs)
        if max_cells > 0 and n_cells > max_cells:
            rng = np.random.default_rng(random_state)
            cell_idx = np.sort(rng.choice(n_cells, size=max_cells, replace=False))
        else:
            cell_idx = np.arange(n_cells, dtype=int)

        x = self.adata.X[cell_idx, :][:, var_idx]
        if sp.issparse(x):
            x_dense = x.toarray().astype(np.float64, copy=False)
        else:
            x_dense = np.asarray(x, dtype=np.float64)

        x_dense -= x_dense.mean(axis=0, keepdims=True)
        std = x_dense.std(axis=0, ddof=1)
        std = np.where(std <= 1e-12, 1.0, std)
        x_dense /= std

        co = np.corrcoef(x_dense, rowvar=False)
        co = np.nan_to_num(co, nan=0.0, posinf=0.0, neginf=0.0)
        co = np.clip(co, -1.0, 1.0)
        abs_co = np.abs(co)

        for i_local, i_global in enumerate(found_positions):
            for j_local, j_global in enumerate(found_positions):
                corr[i_global, j_global] = abs_co[i_local, j_local]
        np.fill_diagonal(corr, 0.0)
        return corr, missing_fraction


def empirical_upper_tail_p(observed: float, null_values: np.ndarray) -> float:
    if not np.isfinite(observed):
        return float("nan")
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float((1 + np.sum(values >= observed)) / (values.size + 1))


def safe_fisher_p(pvals: np.ndarray) -> float:
    values = np.asarray(pvals, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    values = np.clip(values, 1e-12, 1.0)
    _, pvalue = combine_pvalues(values, method="fisher")
    return float(pvalue)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    if np.unique(scores).size < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    rho = spearmanr(x, y).correlation
    return float(rho) if np.isfinite(rho) else float("nan")


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
    points = points.astype(np.float64, copy=False)
    points = points - points.mean(axis=0, keepdims=True)
    max_comp = min(n_components, points.shape[0] - 1, points.shape[1])
    if max_comp < 4:
        raise RuntimeError(f"Too few PCA components: {max_comp}")
    return PCA(
        n_components=max_comp,
        svd_solver="randomized",
        random_state=random_state,
    ).fit_transform(points)


def connect_knn_graph(points: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, int, bool]:
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
    used_component_bridging = False

    while n_components > 1:
        used_component_bridging = True
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

    return knn_dist, k, used_component_bridging


def geodesic_and_transition(
    points: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    knn_dist, k_eff, used_component_bridging = connect_knn_graph(points, n_neighbors=n_neighbors)
    geodesic = shortest_path(knn_dist, directed=False, unweighted=False)
    if np.isinf(geodesic).any():
        geodesic = cdist(points, points, metric="euclidean")
        used_component_bridging = True

    finite_edges = knn_dist[np.isfinite(knn_dist) & (knn_dist > 0)]
    sigma = float(np.median(finite_edges)) if finite_edges.size else 1.0
    sigma = max(sigma, 1e-6)

    weights = np.zeros_like(knn_dist)
    edge_mask = np.isfinite(knn_dist) & (knn_dist > 0)
    weights[edge_mask] = np.exp(-np.square(knn_dist[edge_mask]) / (2.0 * sigma * sigma))
    zero_row_mask = np.sum(weights, axis=1) <= 1e-12
    if np.any(zero_row_mask):
        weights[zero_row_mask, zero_row_mask] = 1.0
    row_sums = weights.sum(axis=1, keepdims=True)
    transition = weights / np.clip(row_sums, 1e-12, None)

    return geodesic.astype(np.float64), transition.astype(np.float64), k_eff, used_component_bridging


def diffusion_distance_scores(
    transition: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    t: int,
) -> np.ndarray:
    pt = np.linalg.matrix_power(transition, int(t))
    diffs = pt[source_local] - pt[target_local]
    return np.sqrt(np.sum(np.square(diffs), axis=1)).astype(np.float64)


def degree_strata(values: np.ndarray, max_bins: int = 5) -> np.ndarray:
    series = pd.Series(values.astype(float))
    n_unique = int(series.nunique(dropna=True))
    q = max(1, min(max_bins, n_unique))
    if q == 1:
        return np.zeros(values.size, dtype=int)
    strata = pd.qcut(series, q=q, labels=False, duplicates="drop")
    return strata.fillna(0).astype(int).to_numpy(dtype=int)


def combine_strata(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        raise ValueError("Need at least one stratum array")
    arrays_i = [np.asarray(a, dtype=int) for a in arrays]
    out = np.zeros_like(arrays_i[0], dtype=int)
    multiplier = 1
    for arr in arrays_i:
        out += multiplier * arr
        multiplier *= int(arr.max() + 2)
    return out


def stratified_permutation(labels: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permuted = np.asarray(labels, dtype=int).copy()
    changed = 0
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        if idx.size > 1:
            permuted[idx] = permuted[rng.permutation(idx)]
            changed += idx.size
    if changed == 0:
        return rng.permutation(permuted)
    return permuted


def stratified_shuffle(values: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        if idx.size > 1:
            out[idx] = out[rng.permutation(idx)]
    return out


def standardize_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, ddof=1, keepdims=True)
    std = np.where(std <= 1e-12, 1.0, std)
    return (x - mean) / std


def fit_logistic_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    x = standardize_columns(features)
    if np.unique(labels).size < 2:
        return np.full(labels.size, 0.5, dtype=float)

    try:
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            random_state=0,
        )
        model.fit(x, labels)
    except Exception:
        model = LogisticRegression(
            solver="liblinear",
            max_iter=500,
            random_state=0,
        )
        model.fit(x, labels)
    scores = model.predict_proba(x)[:, 1]
    return np.asarray(scores, dtype=float)


def safe_log_loss(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    return float(log_loss(labels, clipped))


def row_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def mean_knn_jaccard(x: np.ndarray, y: np.ndarray, n_neighbors: int) -> float:
    n_points = x.shape[0]
    k = max(1, min(n_neighbors, n_points - 1))
    nbrs_x = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(x)
    nbrs_y = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(y)
    _, ix = nbrs_x.kneighbors(x)
    _, iy = nbrs_y.kneighbors(y)
    jacc = np.empty(n_points, dtype=float)
    for i in range(n_points):
        a = set(ix[i, 1:].astype(int).tolist())
        b = set(iy[i, 1:].astype(int).tolist())
        union = len(a | b)
        jacc[i] = (len(a & b) / union) if union > 0 else 0.0
    return float(np.mean(jacc))


def orthogonal_procrustes_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = x.T @ y
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    return u @ vt


def cycle_return_rate(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> float:
    sim_ab = a @ b.T
    sim_bc = b @ c.T
    sim_ca = c @ a.T
    map_ab = np.argmax(sim_ab, axis=1)
    map_bc = np.argmax(sim_bc, axis=1)
    map_ca = np.argmax(sim_ca, axis=1)
    idx = np.arange(a.shape[0], dtype=int)
    return float(np.mean(map_ca[map_bc[map_ab[idx]]] == idx))


def load_geneformer_embeddings() -> np.ndarray:
    model = AutoModel.from_pretrained("ctheodoris/Geneformer")
    embeddings = model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float64, copy=False)
    del model
    return embeddings


def run_h31_diffusion_incremental(
    coexp_cache: dict[str, CoexpressionDomainCache],
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

                top_genes = select_top_genes(split_edges, gene_cap=H31_GENE_CAP)
                top_set = set(top_genes)
                split_edges = split_edges.loc[
                    split_edges["source_idx"].isin(top_set) & split_edges["target_idx"].isin(top_set)
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
                gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
                source_local = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
                target_local = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
                labels = split_edges["label"].to_numpy(dtype=int)

                gene_name_map: dict[int, str] = {}
                for row in split_edges[["source_idx", "source"]].drop_duplicates().itertuples(index=False):
                    gene_name_map[int(row.source_idx)] = str(row.source).upper()
                for row in split_edges[["target_idx", "target"]].drop_duplicates().itertuples(index=False):
                    gene_name_map[int(row.target_idx)] = str(row.target).upper()
                ordered_gene_symbols = [gene_name_map[int(g)] for g in edge_gene_indices]

                coexp_abs_corr, coexp_missing_fraction = coexp_cache[domain].abs_corr_for_genes(
                    ordered_gene_symbols,
                    max_cells=H31_MAX_COEXP_CELLS,
                    random_state=16_100 + domain_index * 100 + seed_index * 10 + split_index,
                )
                coexp_scores = coexp_abs_corr[source_local, target_local]

                source_degree_map = split_edges["source_idx"].value_counts().to_dict()
                target_degree_map = split_edges["target_idx"].value_counts().to_dict()
                source_degree = split_edges["source_idx"].map(source_degree_map).astype(float).to_numpy()
                target_degree = split_edges["target_idx"].map(target_degree_map).astype(float).to_numpy()
                mean_degree = 0.5 * (source_degree + target_degree)

                for layer in H31_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue
                    random_state = (
                        16_110
                        + domain_index * 10_000
                        + seed_index * 1_000
                        + split_index * 100
                        + layer
                    )
                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = reduce_points(points, n_components=24, random_state=random_state)

                    geodesic, transition, k_eff, used_component_bridging = geodesic_and_transition(
                        points_pca,
                        n_neighbors=24,
                    )
                    euclidean = cdist(points_pca, points_pca, metric="euclidean")
                    geodesic_distance = geodesic[source_local, target_local]
                    euclidean_distance = euclidean[source_local, target_local]

                    diffusion_cols: list[np.ndarray] = []
                    for t in H31_DIFFUSION_TIMES:
                        diff_t = diffusion_distance_scores(
                            transition=transition,
                            source_local=source_local,
                            target_local=target_local,
                            t=t,
                        )
                        diffusion_cols.append(-diff_t)

                    baseline_features = np.column_stack(
                        [
                            np.log1p(source_degree),
                            np.log1p(target_degree),
                            coexp_scores,
                            -euclidean_distance,
                            -geodesic_distance,
                        ]
                    )
                    diffusion_features = np.column_stack(diffusion_cols)
                    extended_features = np.column_stack([baseline_features, diffusion_features])

                    base_scores = fit_logistic_scores(baseline_features, labels)
                    ext_scores = fit_logistic_scores(extended_features, labels)
                    auc_base = safe_auc(labels, base_scores)
                    auc_ext = safe_auc(labels, ext_scores)
                    delta_auc = float(auc_ext - auc_base)

                    ll_base = safe_log_loss(labels, base_scores)
                    ll_ext = safe_log_loss(labels, ext_scores)
                    gain_logloss = float(ll_base - ll_ext)

                    degree_bins = degree_strata(mean_degree, max_bins=5)
                    coexp_bins = degree_strata(coexp_scores, max_bins=5)
                    geodesic_bins = degree_strata(geodesic_distance, max_bins=5)
                    strata = combine_strata(degree_bins, coexp_bins, geodesic_bins)

                    rng = np.random.default_rng(
                        16_120
                        + domain_index * 10_000
                        + seed_index * 1_000
                        + split_index * 100
                        + layer
                    )
                    null_delta_auc = np.empty(H31_NULL_PERM, dtype=float)
                    null_gain_logloss = np.empty(H31_NULL_PERM, dtype=float)

                    for perm_idx in range(H31_NULL_PERM):
                        shuffled = np.column_stack(
                            [
                                stratified_shuffle(diffusion_features[:, j], strata, rng=rng)
                                for j in range(diffusion_features.shape[1])
                            ]
                        )
                        ext_perm = np.column_stack([baseline_features, shuffled])
                        perm_scores = fit_logistic_scores(ext_perm, labels)
                        auc_perm = safe_auc(labels, perm_scores)
                        ll_perm = safe_log_loss(labels, perm_scores)
                        null_delta_auc[perm_idx] = float(auc_perm - auc_base)
                        null_gain_logloss[perm_idx] = float(ll_base - ll_perm)
                        null_rows.append(
                            {
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(null_delta_auc[perm_idx]),
                                "null_gain_logloss": float(null_gain_logloss[perm_idx]),
                            }
                        )

                    rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "n_edges": int(labels.size),
                            "n_positive": int(labels.sum()),
                            "n_unique_genes": int(edge_gene_indices.size),
                            "pca_dim": int(points_pca.shape[1]),
                            "knn_k": int(k_eff),
                            "used_component_bridging": bool(used_component_bridging),
                            "coexp_missing_fraction": float(coexp_missing_fraction),
                            "auc_baseline_covariates": float(auc_base),
                            "auc_with_diffusion": float(auc_ext),
                            "delta_auc_diffusion_incremental": float(delta_auc),
                            "logloss_baseline_covariates": float(ll_base),
                            "logloss_with_diffusion": float(ll_ext),
                            "gain_logloss_diffusion_incremental": float(gain_logloss),
                            "null_mean_delta_auc": float(np.mean(null_delta_auc)),
                            "null_mean_gain_logloss": float(np.mean(null_gain_logloss)),
                            "p_delta_auc_upper": float(empirical_upper_tail_p(delta_auc, null_delta_auc)),
                            "p_gain_logloss_upper": float(
                                empirical_upper_tail_p(gain_logloss, null_gain_logloss)
                            ),
                        }
                    )

    row_df = pd.DataFrame(rows).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    row_path = ITER_DIR / "h31_diffusion_incremental_by_seed_layer_split.csv"
    row_df.to_csv(row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["domain", "seed_tag", "split_regime", "layer", "perm_idx"]
    )
    null_path = ITER_DIR / "h31_diffusion_incremental_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    domain_rows: list[dict[str, object]] = []
    for (domain, split_regime), group in row_df.groupby(["domain", "split_regime"], sort=True):
        domain_rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(group.shape[0]),
                "mean_auc_baseline_covariates": float(group["auc_baseline_covariates"].mean()),
                "mean_auc_with_diffusion": float(group["auc_with_diffusion"].mean()),
                "mean_delta_auc_diffusion_incremental": float(
                    group["delta_auc_diffusion_incremental"].mean()
                ),
                "median_delta_auc_diffusion_incremental": float(
                    group["delta_auc_diffusion_incremental"].median()
                ),
                "mean_gain_logloss_diffusion_incremental": float(
                    group["gain_logloss_diffusion_incremental"].mean()
                ),
                "fraction_delta_auc_positive": float(
                    (group["delta_auc_diffusion_incremental"] > 0.0).mean()
                ),
                "fraction_p_delta_auc_lt_0_05": float((group["p_delta_auc_upper"] < 0.05).mean()),
                "combined_fisher_p_delta_auc": float(
                    safe_fisher_p(group["p_delta_auc_upper"].to_numpy(dtype=float))
                ),
            }
        )
    domain_df = pd.DataFrame(domain_rows).sort_values(["domain", "split_regime"])
    domain_path = ITER_DIR / "h31_diffusion_incremental_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    summary = {
        "rows_tested": int(row_df.shape[0]),
        "domain_split_groups": int(domain_df.shape[0]),
        "groups_mean_delta_positive": int(
            (domain_df["mean_delta_auc_diffusion_incremental"] > 0.0).sum()
        )
        if not domain_df.empty
        else 0,
        "groups_fisher_sig": int((domain_df["combined_fisher_p_delta_auc"] < 0.05).sum())
        if not domain_df.empty
        else 0,
        "mean_delta_auc_overall": float(row_df["delta_auc_diffusion_incremental"].mean())
        if not row_df.empty
        else float("nan"),
        "mean_gain_logloss_overall": float(row_df["gain_logloss_diffusion_incremental"].mean())
        if not row_df.empty
        else float("nan"),
        "artifact_paths": {
            "by_seed_layer_split": str(row_path),
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
        },
    }
    return summary


def run_h32_convexity_detour_screen(
    coexp_cache: dict[str, CoexpressionDomainCache],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H32_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = select_top_genes(split_edges, gene_cap=H32_GENE_CAP)
            top_set = set(top_genes)
            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(top_set) & split_edges["target_idx"].isin(top_set)
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
            gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
            source_local = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
            target_local = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
            labels = split_edges["label"].to_numpy(dtype=int)

            gene_name_map: dict[int, str] = {}
            for row in split_edges[["source_idx", "source"]].drop_duplicates().itertuples(index=False):
                gene_name_map[int(row.source_idx)] = str(row.source).upper()
            for row in split_edges[["target_idx", "target"]].drop_duplicates().itertuples(index=False):
                gene_name_map[int(row.target_idx)] = str(row.target).upper()
            ordered_gene_symbols = [gene_name_map[int(g)] for g in edge_gene_indices]

            coexp_abs_corr, coexp_missing_fraction = coexp_cache[domain].abs_corr_for_genes(
                ordered_gene_symbols,
                max_cells=H31_MAX_COEXP_CELLS,
                random_state=16_300 + domain_index * 100 + split_index * 10,
            )
            coexp_scores = coexp_abs_corr[source_local, target_local]

            source_degree_map = split_edges["source_idx"].value_counts().to_dict()
            target_degree_map = split_edges["target_idx"].value_counts().to_dict()
            source_degree = split_edges["source_idx"].map(source_degree_map).astype(float).to_numpy()
            target_degree = split_edges["target_idx"].map(target_degree_map).astype(float).to_numpy()
            mean_degree = 0.5 * (source_degree + target_degree)

            for layer in H32_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue
                random_state = 16_310 + domain_index * 100 + split_index * 10 + layer
                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = reduce_points(points, n_components=24, random_state=random_state)
                geodesic, _, k_eff, used_component_bridging = geodesic_and_transition(
                    points_pca,
                    n_neighbors=24,
                )
                euclidean = cdist(points_pca, points_pca, metric="euclidean")
                geo_edge = geodesic[source_local, target_local]
                euc_edge = np.clip(euclidean[source_local, target_local], 1e-6, None)
                detour_ratio = geo_edge / euc_edge

                n_nodes = geodesic.shape[0]
                sorted_geo = np.sort(geodesic, axis=1)
                k_local = min(8, max(1, n_nodes - 1))
                local_radius = sorted_geo[:, k_local]
                neighborhood_mask = geodesic <= local_radius[:, None]
                convexity_deficit = np.empty(labels.size, dtype=float)
                for i in range(labels.size):
                    src = int(source_local[i])
                    tgt = int(target_local[i])
                    src_mask = neighborhood_mask[src]
                    tgt_mask = neighborhood_mask[tgt]
                    inter = int(np.sum(src_mask & tgt_mask))
                    union = int(np.sum(src_mask | tgt_mask))
                    jacc = (inter / union) if union > 0 else 0.0
                    convexity_deficit[i] = 1.0 - jacc

                geodesic_score = -geo_edge
                combo_features = np.column_stack([detour_ratio, convexity_deficit, geodesic_score])
                combo_scores = fit_logistic_scores(combo_features, labels)

                auc_detour = safe_auc(labels, detour_ratio)
                auc_convexity = safe_auc(labels, convexity_deficit)
                auc_geodesic = safe_auc(labels, geodesic_score)
                auc_combo = safe_auc(labels, combo_scores)
                delta_combo = float(auc_combo - auc_geodesic)

                degree_bins = degree_strata(mean_degree, max_bins=5)
                geo_bins = degree_strata(geo_edge, max_bins=5)
                coexp_bins = degree_strata(coexp_scores, max_bins=5)
                strata = combine_strata(degree_bins, geo_bins, coexp_bins)

                rng = np.random.default_rng(16_320 + domain_index * 100 + split_index * 10 + layer)
                null_auc_combo = np.empty(H32_NULL_PERM, dtype=float)
                null_delta = np.empty(H32_NULL_PERM, dtype=float)
                for perm_idx in range(H32_NULL_PERM):
                    y_perm = stratified_permutation(labels, strata=strata, rng=rng)
                    auc_combo_perm = safe_auc(y_perm, combo_scores)
                    auc_geo_perm = safe_auc(y_perm, geodesic_score)
                    null_auc_combo[perm_idx] = float(auc_combo_perm)
                    null_delta[perm_idx] = float(auc_combo_perm - auc_geo_perm)
                    null_rows.append(
                        {
                            "domain": domain,
                            "seed_tag": H32_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_auc_combo": float(null_auc_combo[perm_idx]),
                            "null_delta_combo_minus_geodesic": float(null_delta[perm_idx]),
                        }
                    )

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": H32_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges": int(labels.size),
                        "n_positive": int(labels.sum()),
                        "n_unique_genes": int(edge_gene_indices.size),
                        "pca_dim": int(points_pca.shape[1]),
                        "knn_k": int(k_eff),
                        "used_component_bridging": bool(used_component_bridging),
                        "coexp_missing_fraction": float(coexp_missing_fraction),
                        "auc_detour_ratio": float(auc_detour),
                        "auc_convexity_deficit": float(auc_convexity),
                        "auc_geodesic_baseline": float(auc_geodesic),
                        "auc_combo_convexity_detour": float(auc_combo),
                        "delta_auc_combo_minus_geodesic": float(delta_combo),
                        "p_auc_combo_upper": float(empirical_upper_tail_p(auc_combo, null_auc_combo)),
                        "p_delta_combo_upper": float(empirical_upper_tail_p(delta_combo, null_delta)),
                        "null_mean_auc_combo": float(np.mean(null_auc_combo)),
                    }
                )

    row_df = pd.DataFrame(rows).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    row_path = ITER_DIR / "h32_convexity_detour_by_seed_layer_split.csv"
    row_df.to_csv(row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["domain", "seed_tag", "split_regime", "layer", "perm_idx"]
    )
    null_path = ITER_DIR / "h32_convexity_detour_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    domain_rows: list[dict[str, object]] = []
    for (domain, split_regime), group in row_df.groupby(["domain", "split_regime"], sort=True):
        domain_rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(group.shape[0]),
                "mean_auc_detour_ratio": float(group["auc_detour_ratio"].mean()),
                "mean_auc_convexity_deficit": float(group["auc_convexity_deficit"].mean()),
                "mean_auc_geodesic_baseline": float(group["auc_geodesic_baseline"].mean()),
                "mean_auc_combo_convexity_detour": float(group["auc_combo_convexity_detour"].mean()),
                "mean_delta_auc_combo_minus_geodesic": float(
                    group["delta_auc_combo_minus_geodesic"].mean()
                ),
                "fraction_combo_auc_gt_0_5": float((group["auc_combo_convexity_detour"] > 0.5).mean()),
                "fraction_p_delta_lt_0_05": float((group["p_delta_combo_upper"] < 0.05).mean()),
                "combined_fisher_p_delta": float(
                    safe_fisher_p(group["p_delta_combo_upper"].to_numpy(dtype=float))
                ),
            }
        )

    domain_df = pd.DataFrame(domain_rows).sort_values(["domain", "split_regime"])
    domain_path = ITER_DIR / "h32_convexity_detour_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    summary = {
        "rows_tested": int(row_df.shape[0]),
        "domain_split_groups": int(domain_df.shape[0]),
        "groups_mean_combo_auc_gt_0_5": int(
            (domain_df["mean_auc_combo_convexity_detour"] > 0.5).sum()
        )
        if not domain_df.empty
        else 0,
        "groups_fisher_sig_delta": int((domain_df["combined_fisher_p_delta"] < 0.05).sum())
        if not domain_df.empty
        else 0,
        "mean_auc_combo_overall": float(row_df["auc_combo_convexity_detour"].mean())
        if not row_df.empty
        else float("nan"),
        "mean_delta_combo_minus_geodesic_overall": float(
            row_df["delta_auc_combo_minus_geodesic"].mean()
        )
        if not row_df.empty
        else float("nan"),
        "artifact_paths": {
            "by_seed_layer_split": str(row_path),
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
        },
    }
    return summary


def run_h33_cycle_consistent_alignment() -> dict[str, object]:
    domains = ["immune", "lung", "external_lung"]
    edge_df_by_domain = {d: pd.read_csv(GENEFORMER_EDGE_BY_DOMAIN[d], sep="\t") for d in domains}

    # Build symbol-index and symbol-token maps per domain from edge tables.
    symbol_to_idx: dict[str, dict[str, int]] = {}
    symbol_to_token: dict[str, dict[str, int]] = {}
    degree_by_domain: dict[str, dict[str, int]] = {}

    for d, edge_df in edge_df_by_domain.items():
        long_map = pd.concat(
            [
                edge_df[["source", "source_idx", "source_token_id"]].rename(
                    columns={
                        "source": "gene",
                        "source_idx": "gene_idx",
                        "source_token_id": "token_id",
                    }
                ),
                edge_df[["target", "target_idx", "target_token_id"]].rename(
                    columns={
                        "target": "gene",
                        "target_idx": "gene_idx",
                        "target_token_id": "token_id",
                    }
                ),
            ],
            axis=0,
            ignore_index=True,
        )
        long_map["gene"] = long_map["gene"].astype(str).str.upper()
        map_df = (
            long_map.groupby(["gene", "gene_idx", "token_id"], as_index=False)
            .size()
            .sort_values(["gene", "size"], ascending=[True, False])
            .drop_duplicates(subset=["gene"], keep="first")
        )
        symbol_to_idx[d] = {
            str(g): int(i) for g, i in zip(map_df["gene"].tolist(), map_df["gene_idx"].tolist())
        }
        symbol_to_token[d] = {
            str(g): int(t) for g, t in zip(map_df["gene"].tolist(), map_df["token_id"].tolist())
        }

        degree_counts = (
            pd.concat([edge_df["source"].astype(str).str.upper(), edge_df["target"].astype(str).str.upper()])
            .value_counts()
            .to_dict()
        )
        degree_by_domain[d] = {str(g): int(c) for g, c in degree_counts.items()}

    shared_symbols = set(symbol_to_idx[domains[0]].keys())
    for d in domains[1:]:
        shared_symbols &= set(symbol_to_idx[d].keys())
        shared_symbols &= set(symbol_to_token[d].keys())
    shared_symbols = sorted(shared_symbols)
    if len(shared_symbols) < 80:
        raise RuntimeError(f"Too few shared symbols for H33: {len(shared_symbols)}")

    ranked_symbols = sorted(
        shared_symbols,
        key=lambda g: (
            -sum(degree_by_domain[d].get(g, 0) for d in domains),
            g,
        ),
    )
    selected_symbols = ranked_symbols[: min(H33_GENE_CAP, len(ranked_symbols))]
    selected_set = set(selected_symbols)

    token_consistency = []
    for g in selected_symbols:
        toks = [symbol_to_token[d][g] for d in domains]
        token_consistency.append(len(set(toks)) == 1)
    if not all(token_consistency):
        raise RuntimeError("Token IDs are not consistent across domains for selected symbols.")

    token_ids = np.array([symbol_to_token[domains[0]][g] for g in selected_symbols], dtype=int)
    geneformer_embeddings = load_geneformer_embeddings()
    y_raw = geneformer_embeddings[token_ids, :]

    layer_embeddings_by_domain: dict[str, np.ndarray] = {}
    x_raw_by_domain: dict[str, np.ndarray] = {}
    for d in domains:
        run_dir = SCGPT_RUNS_BY_DOMAIN[d]["seed42_main"]
        layer = int(CROSS_MODEL_LAYER_BY_DOMAIN[d])
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        if layer >= layer_embeddings.shape[0]:
            raise RuntimeError(f"Layer {layer} missing for domain {d}.")
        layer_embeddings_by_domain[d] = layer_embeddings[layer]
        gene_idx = np.array([symbol_to_idx[d][g] for g in selected_symbols], dtype=int)
        x_raw_by_domain[d] = layer_embeddings[layer, gene_idx, :].astype(np.float64)

    n_components = min(
        40,
        len(selected_symbols) - 1,
        y_raw.shape[1],
        min(x_raw_by_domain[d].shape[1] for d in domains),
    )
    if n_components < 8:
        raise RuntimeError(f"Too few PCA components for H33: {n_components}")

    y_pca = reduce_points(y_raw, n_components=n_components, random_state=16_500)
    y_norm = row_normalize(y_pca)

    x_pca_by_domain: dict[str, np.ndarray] = {}
    z_ind_by_domain: dict[str, np.ndarray] = {}
    q_ind_by_domain: dict[str, np.ndarray] = {}
    for d_idx, d in enumerate(domains):
        x_pca = reduce_points(x_raw_by_domain[d], n_components=n_components, random_state=16_510 + d_idx)
        x_pca_by_domain[d] = x_pca
        q_ind = orthogonal_procrustes_map(x_pca, y_pca)
        q_ind_by_domain[d] = q_ind
        z_ind_by_domain[d] = row_normalize(x_pca @ q_ind)

    # Joint cycle-consistent refinement toward a shared mapped manifold.
    consensus = y_pca.copy()
    q_cycle_by_domain: dict[str, np.ndarray] = {}
    z_cycle_by_domain: dict[str, np.ndarray] = {}
    for _ in range(H33_CYCLE_ITERS):
        updated = {}
        for d in domains:
            target = (1.0 - H33_CYCLE_LAMBDA) * y_pca + H33_CYCLE_LAMBDA * consensus
            q = orthogonal_procrustes_map(x_pca_by_domain[d], target)
            updated[d] = x_pca_by_domain[d] @ q
            q_cycle_by_domain[d] = q
        consensus = np.mean(np.stack([updated[d] for d in domains], axis=0), axis=0)
        for d in domains:
            z_cycle_by_domain[d] = row_normalize(updated[d])

    domain_rows: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    def edge_subset_for_domain(domain: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        edge_df = edge_df_by_domain[domain].copy()
        edge_df["source_upper"] = edge_df["source"].astype(str).str.upper()
        edge_df["target_upper"] = edge_df["target"].astype(str).str.upper()
        edge_sub = edge_df.loc[
            edge_df["source_upper"].isin(selected_set) & edge_df["target_upper"].isin(selected_set)
        ].copy()
        gene_to_pos = {g: i for i, g in enumerate(selected_symbols)}
        src = edge_sub["source_upper"].map(gene_to_pos).to_numpy(dtype=int)
        tgt = edge_sub["target_upper"].map(gene_to_pos).to_numpy(dtype=int)
        labels = edge_sub["label"].to_numpy(dtype=int)
        return src, tgt, labels

    edge_info = {d: edge_subset_for_domain(d) for d in domains}

    for d in domains:
        src, tgt, labels = edge_info[d]
        z_ind = z_ind_by_domain[d]
        z_cycle = z_cycle_by_domain[d]

        score_ind = np.sum(z_ind[src] * z_ind[tgt], axis=1)
        score_cycle = np.sum(z_cycle[src] * z_cycle[tgt], axis=1)
        auc_ind = safe_auc(labels, score_ind)
        auc_cycle = safe_auc(labels, score_cycle)

        top1_ind = float(np.mean(np.argmax(z_ind @ y_norm.T, axis=1) == np.arange(z_ind.shape[0])))
        top1_cycle = float(np.mean(np.argmax(z_cycle @ y_norm.T, axis=1) == np.arange(z_cycle.shape[0])))
        dist_rho_ind = safe_spearman(pdist(z_ind), pdist(y_norm))
        dist_rho_cycle = safe_spearman(pdist(z_cycle), pdist(y_norm))
        jacc_ind = mean_knn_jaccard(z_ind, y_norm, n_neighbors=min(10, z_ind.shape[0] - 1))
        jacc_cycle = mean_knn_jaccard(z_cycle, y_norm, n_neighbors=min(10, z_cycle.shape[0] - 1))

        domain_rows.append(
            {
                "domain": d,
                "n_symbols": int(len(selected_symbols)),
                "n_edges_eval": int(labels.size),
                "n_positive_eval": int(labels.sum()),
                "auc_independent": float(auc_ind),
                "auc_cycle_consistent": float(auc_cycle),
                "delta_auc_cycle_minus_independent": float(auc_cycle - auc_ind),
                "top1_independent": float(top1_ind),
                "top1_cycle_consistent": float(top1_cycle),
                "distance_spearman_independent": float(dist_rho_ind),
                "distance_spearman_cycle_consistent": float(dist_rho_cycle),
                "knn_jaccard_independent": float(jacc_ind),
                "knn_jaccard_cycle_consistent": float(jacc_cycle),
            }
        )

    cycle_ind = float(
        np.mean(
            [
                cycle_return_rate(
                    z_ind_by_domain["immune"],
                    z_ind_by_domain["lung"],
                    z_ind_by_domain["external_lung"],
                ),
                cycle_return_rate(
                    z_ind_by_domain["lung"],
                    z_ind_by_domain["external_lung"],
                    z_ind_by_domain["immune"],
                ),
                cycle_return_rate(
                    z_ind_by_domain["external_lung"],
                    z_ind_by_domain["immune"],
                    z_ind_by_domain["lung"],
                ),
            ]
        )
    )
    cycle_consistent = float(
        np.mean(
            [
                cycle_return_rate(
                    z_cycle_by_domain["immune"],
                    z_cycle_by_domain["lung"],
                    z_cycle_by_domain["external_lung"],
                ),
                cycle_return_rate(
                    z_cycle_by_domain["lung"],
                    z_cycle_by_domain["external_lung"],
                    z_cycle_by_domain["immune"],
                ),
                cycle_return_rate(
                    z_cycle_by_domain["external_lung"],
                    z_cycle_by_domain["immune"],
                    z_cycle_by_domain["lung"],
                ),
            ]
        )
    )

    rng = np.random.default_rng(16_600)
    null_cycle = np.empty(H33_NULL_PERM, dtype=float)
    null_auc_by_domain = {d: np.empty(H33_NULL_PERM, dtype=float) for d in domains}
    for perm_idx in range(H33_NULL_PERM):
        z_rand: dict[str, np.ndarray] = {}
        for d in domains:
            q_rand, _ = np.linalg.qr(rng.normal(size=(n_components, n_components)))
            z_rand[d] = row_normalize(x_pca_by_domain[d] @ q_rand)

        null_cycle[perm_idx] = float(
            np.mean(
                [
                    cycle_return_rate(z_rand["immune"], z_rand["lung"], z_rand["external_lung"]),
                    cycle_return_rate(z_rand["lung"], z_rand["external_lung"], z_rand["immune"]),
                    cycle_return_rate(z_rand["external_lung"], z_rand["immune"], z_rand["lung"]),
                ]
            )
        )

        for d in domains:
            src, tgt, labels = edge_info[d]
            score_rand = np.sum(z_rand[d][src] * z_rand[d][tgt], axis=1)
            null_auc_by_domain[d][perm_idx] = safe_auc(labels, score_rand)
            null_rows.append(
                {
                    "perm_idx": int(perm_idx),
                    "domain": d,
                    "null_cycle_return_rate": float(null_cycle[perm_idx]),
                    "null_edge_auc": float(null_auc_by_domain[d][perm_idx]),
                }
            )

    domain_df = pd.DataFrame(domain_rows).sort_values("domain")
    for d in domains:
        observed_auc = float(
            domain_df.loc[domain_df["domain"] == d, "auc_cycle_consistent"].to_numpy(dtype=float)[0]
        )
        domain_df.loc[domain_df["domain"] == d, "null_mean_edge_auc"] = float(
            np.mean(null_auc_by_domain[d])
        )
        domain_df.loc[domain_df["domain"] == d, "p_edge_auc_cycle_upper_vs_random"] = float(
            empirical_upper_tail_p(observed_auc, null_auc_by_domain[d])
        )

    domain_path = ITER_DIR / "h33_cycle_consistent_alignment_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    map_rows.append(
        {
            "map_variant": "independent_pairwise",
            "mean_cycle_return_rate": float(cycle_ind),
        }
    )
    map_rows.append(
        {
            "map_variant": "tri_domain_cycle_consistent",
            "mean_cycle_return_rate": float(cycle_consistent),
        }
    )
    map_df = pd.DataFrame(map_rows)
    map_df["delta_cycle_return_vs_independent"] = np.where(
        map_df["map_variant"] == "tri_domain_cycle_consistent",
        float(cycle_consistent - cycle_ind),
        0.0,
    )
    map_df["null_mean_cycle_return"] = float(np.mean(null_cycle))
    map_df["p_cycle_return_upper_vs_random"] = float(
        empirical_upper_tail_p(cycle_consistent, null_cycle)
    )
    map_path = ITER_DIR / "h33_cycle_consistent_alignment_map_quality.csv"
    map_df.to_csv(map_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(["perm_idx", "domain"])
    null_path = ITER_DIR / "h33_cycle_consistent_alignment_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary = {
        "n_domains": int(domain_df.shape[0]),
        "n_shared_symbols": int(len(selected_symbols)),
        "mean_auc_independent": float(domain_df["auc_independent"].mean()),
        "mean_auc_cycle_consistent": float(domain_df["auc_cycle_consistent"].mean()),
        "mean_delta_auc_cycle_minus_independent": float(
            domain_df["delta_auc_cycle_minus_independent"].mean()
        ),
        "mean_cycle_return_independent": float(cycle_ind),
        "mean_cycle_return_cycle_consistent": float(cycle_consistent),
        "delta_cycle_return": float(cycle_consistent - cycle_ind),
        "p_cycle_return_upper_vs_random": float(
            empirical_upper_tail_p(cycle_consistent, null_cycle)
        ),
        "domains_sig_edge_auc_cycle_vs_random": int(
            (domain_df["p_edge_auc_cycle_upper_vs_random"] < 0.05).sum()
        ),
        "artifact_paths": {
            "domain_summary": str(domain_path),
            "map_quality": str(map_path),
            "null_summary": str(null_path),
        },
    }
    return summary


def main() -> None:
    required_paths: list[Path] = []
    for run_map in SCGPT_RUNS_BY_DOMAIN.values():
        for run_dir in run_map.values():
            required_paths.append(run_dir / "layer_gene_embeddings.npy")
            required_paths.append(run_dir / "cycle1_edge_dataset.tsv")
    required_paths.extend(GENEFORMER_EDGE_BY_DOMAIN.values())
    required_paths.extend(PROCESSED_H5AD_BY_DOMAIN.values())

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    coexp_cache: dict[str, CoexpressionDomainCache] = {
        domain: CoexpressionDomainCache(path)
        for domain, path in PROCESSED_H5AD_BY_DOMAIN.items()
    }
    try:
        h31_summary = run_h31_diffusion_incremental(coexp_cache=coexp_cache)
        h32_summary = run_h32_convexity_detour_screen(coexp_cache=coexp_cache)
    finally:
        for cache in coexp_cache.values():
            cache.close()

    h33_summary = run_h33_cycle_consistent_alignment()

    iteration_summary = {
        "iteration": "iter_0016",
        "inputs": {
            "scgpt_runs_by_domain": {
                domain: {seed: str(path) for seed, path in run_map.items()}
                for domain, run_map in SCGPT_RUNS_BY_DOMAIN.items()
            },
            "geneformer_edge_by_domain": {
                domain: str(path) for domain, path in GENEFORMER_EDGE_BY_DOMAIN.items()
            },
            "processed_h5ad_by_domain": {
                domain: str(path) for domain, path in PROCESSED_H5AD_BY_DOMAIN.items()
            },
            "cross_model_layer_by_domain": CROSS_MODEL_LAYER_BY_DOMAIN,
        },
        "h31_diffusion_incremental_covariate_adjusted": h31_summary,
        "h32_convexity_detour_screen": h32_summary,
        "h33_tri_domain_cycle_consistent_alignment": h33_summary,
    }

    summary_path = ITER_DIR / "iter0016_screen_summary.json"
    summary_path.write_text(json.dumps(iteration_summary, indent=2))
    print(json.dumps(iteration_summary, indent=2))


if __name__ == "__main__":
    main()
