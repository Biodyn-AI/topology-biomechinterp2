from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import combine_pvalues, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

import ot
from transformers import AutoModel


ITER_DIR = Path("iterations/iter_0015")
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

CROSS_MODEL_LAYER_BY_DOMAIN = {"immune": 0, "lung": 0, "external_lung": 3}

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

H27_REFERENCE_DOMAIN_SUMMARY = Path("iterations/iter_0014/h27_gw_alignment_domain_summary.csv")

# H28: stronger-null diffusion screen.
H28_LAYERS = [0, 3, 7, 11]
H28_DIFFUSION_TIMES = [1, 2, 4, 8]
H28_GENE_CAP = 280
H28_NULL_PERM_RANDOM = 120
H28_NULL_PERM_MATCHED = 120
H28_MAX_COEXP_CELLS = 6000

# H29: seeded one-to-one GW rescue.
H29_GENE_CAP = 280
H29_NULL_PERM = 180
H29_EPSILON_SCHEDULE = [1.0, 0.5, 0.2]

# H30: cheap triangle-thinness broad screen.
H30_LAYERS = [0, 3, 7, 11]
H30_GENE_CAP = 260
H30_NULL_PERM = 160
H30_THIRD_NODE_SAMPLES = 12


@dataclass
class CoexpressionDomainCache:
    """
    Keeps one backed AnnData handle per domain and provides fast pairwise
    coexpression matrices for a requested gene subset.

    We use this only for null stratification, so light subsampling of cells is
    acceptable and keeps iteration runtime bounded.
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
    if not np.isfinite(rho):
        return float("nan")
    return float(rho)


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


def get_knn_indices(points: np.ndarray, n_neighbors: int) -> np.ndarray:
    n_points = points.shape[0]
    k = max(1, min(n_neighbors, n_points - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(points)
    _, indices = nbrs.kneighbors(points)
    return indices[:, 1:]


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

    # If disconnected, iteratively add the shortest cross-component edge until
    # connectivity is restored so geodesic distances are always defined.
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


def row_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def mean_knn_jaccard(x: np.ndarray, y: np.ndarray, n_neighbors: int) -> float:
    knn_x = get_knn_indices(x, n_neighbors=n_neighbors)
    knn_y = get_knn_indices(y, n_neighbors=n_neighbors)
    jaccards = np.empty(x.shape[0], dtype=float)
    for idx in range(x.shape[0]):
        nx = {int(v) for v in knn_x[idx]}
        ny = {int(v) for v in knn_y[idx]}
        union = len(nx | ny)
        if union == 0:
            jaccards[idx] = 0.0
        else:
            jaccards[idx] = len(nx & ny) / union
    return float(jaccards.mean())


def inv_sqrt_psd(matrix: np.ndarray, eps: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(matrix)
    clipped = np.clip(evals, eps, None)
    inv_sqrt_vals = 1.0 / np.sqrt(clipped)
    return (evecs * inv_sqrt_vals) @ evecs.T


def linear_cca_projection(
    x: np.ndarray, y: np.ndarray, n_components: int, reg: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    n_samples = x.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 samples for CCA")

    x_center = x - x.mean(axis=0, keepdims=True)
    y_center = y - y.mean(axis=0, keepdims=True)

    scale = float(max(n_samples - 1, 1))
    sxx = (x_center.T @ x_center) / scale + reg * np.eye(x_center.shape[1], dtype=float)
    syy = (y_center.T @ y_center) / scale + reg * np.eye(y_center.shape[1], dtype=float)
    sxy = (x_center.T @ y_center) / scale

    wx = inv_sqrt_psd(sxx, eps=reg)
    wy = inv_sqrt_psd(syy, eps=reg)
    m = wx @ sxy @ wy
    u, singular_values, vt = np.linalg.svd(m, full_matrices=False)

    k = min(n_components, singular_values.size)
    wx_u = wx @ u[:, :k]
    wy_v = wy @ vt.T[:, :k]

    x_c = x_center @ wx_u
    y_c = y_center @ wy_v
    canonical_corr = singular_values[:k]
    return x_c.astype(np.float64), y_c.astype(np.float64), canonical_corr.astype(np.float64)


def select_mapped_genes(
    edge_df: pd.DataFrame,
    gene_cap: int,
    geneformer_vocab_size: int,
) -> pd.DataFrame:
    selected_genes = select_top_genes(edge_df, gene_cap=gene_cap)
    selected_set = set(selected_genes)
    edge_sub = edge_df.loc[
        edge_df["source_idx"].isin(selected_set) & edge_df["target_idx"].isin(selected_set)
    ].copy()

    source_map = edge_sub[["source_idx", "source_token_id"]].rename(
        columns={"source_idx": "gene_idx", "source_token_id": "token_id"}
    )
    target_map = edge_sub[["target_idx", "target_token_id"]].rename(
        columns={"target_idx": "gene_idx", "target_token_id": "token_id"}
    )
    mapping_df = pd.concat([source_map, target_map], axis=0, ignore_index=True)
    mapping_df["gene_idx"] = mapping_df["gene_idx"].astype(int)
    mapping_df["token_id"] = mapping_df["token_id"].astype(int)
    mapping_df = mapping_df.loc[mapping_df["token_id"] < geneformer_vocab_size].copy()

    mapping_df = (
        mapping_df.groupby(["gene_idx", "token_id"], as_index=False)
        .size()
        .sort_values(["gene_idx", "size", "token_id"], ascending=[True, False, True])
        .drop_duplicates(subset=["gene_idx"], keep="first")
        .drop(columns=["size"])
    )

    degree_rank = {gene_idx: rank for rank, gene_idx in enumerate(selected_genes)}
    mapping_df["degree_rank"] = mapping_df["gene_idx"].map(degree_rank).fillna(10**9).astype(int)
    mapping_df = mapping_df.sort_values("degree_rank").drop(columns=["degree_rank"]).reset_index(drop=True)
    return mapping_df


def transfer_auc_from_mapping(
    map_gene_to_token_pos: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    labels: np.ndarray,
    y_points: np.ndarray,
) -> float:
    y_norm = row_normalize(y_points)
    y_cos = y_norm @ y_norm.T
    src_token_pos = map_gene_to_token_pos[source_local]
    tgt_token_pos = map_gene_to_token_pos[target_local]
    scores = y_cos[src_token_pos, tgt_token_pos]
    return safe_auc(labels, scores)


def load_geneformer_embeddings() -> np.ndarray:
    model = AutoModel.from_pretrained("ctheodoris/Geneformer")
    embeddings = model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float64, copy=False)
    del model
    return embeddings


def run_h28_diffusion_stronger_nulls(
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

                top_genes = select_top_genes(split_edges, gene_cap=H28_GENE_CAP)
                top_gene_set = set(top_genes)
                split_edges = split_edges.loc[
                    split_edges["source_idx"].isin(top_gene_set)
                    & split_edges["target_idx"].isin(top_gene_set)
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
                gene_to_local = {
                    int(gene_idx): int(local_idx)
                    for local_idx, gene_idx in enumerate(edge_gene_indices)
                }
                source_local = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
                target_local = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
                labels = split_edges["label"].to_numpy(dtype=int)

                gene_name_map = {}
                for row in split_edges[["source_idx", "source"]].drop_duplicates().itertuples(index=False):
                    gene_name_map[int(row.source_idx)] = str(row.source).upper()
                for row in split_edges[["target_idx", "target"]].drop_duplicates().itertuples(index=False):
                    gene_name_map[int(row.target_idx)] = str(row.target).upper()
                ordered_gene_symbols = [gene_name_map[int(g)] for g in edge_gene_indices]

                coexp_abs_corr, missing_fraction = coexp_cache[domain].abs_corr_for_genes(
                    ordered_gene_symbols,
                    max_cells=H28_MAX_COEXP_CELLS,
                    random_state=15_800 + domain_index * 100 + seed_index * 10 + split_index,
                )
                coexp_scores = coexp_abs_corr[source_local, target_local]

                source_degree_map = split_edges["source_idx"].value_counts().to_dict()
                target_degree_map = split_edges["target_idx"].value_counts().to_dict()
                source_degree = split_edges["source_idx"].map(source_degree_map).astype(float).to_numpy()
                target_degree = split_edges["target_idx"].map(target_degree_map).astype(float).to_numpy()
                mean_degree = 0.5 * (source_degree + target_degree)

                degree_bins = degree_strata(mean_degree, max_bins=5)
                coexp_bins = degree_strata(coexp_scores, max_bins=5)
                matched_strata = combine_strata(degree_bins, coexp_bins)

                for layer in H28_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    random_state = (
                        15_810
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
                    euclidean_scores = -euclidean[source_local, target_local]
                    geodesic_scores = -geodesic[source_local, target_local]
                    auc_euclidean = safe_auc(labels, euclidean_scores)
                    auc_geodesic = safe_auc(labels, geodesic_scores)

                    diffusion_auc_by_t: dict[int, float] = {}
                    diffusion_scores_by_t: dict[int, np.ndarray] = {}
                    for t in H28_DIFFUSION_TIMES:
                        diffusion_distance = diffusion_distance_scores(
                            transition,
                            source_local=source_local,
                            target_local=target_local,
                            t=t,
                        )
                        score = -diffusion_distance
                        diffusion_scores_by_t[t] = score
                        diffusion_auc_by_t[t] = safe_auc(labels, score)

                    best_t = int(max(diffusion_auc_by_t, key=diffusion_auc_by_t.get))
                    best_auc = float(diffusion_auc_by_t[best_t])
                    base_auc = float(max(auc_euclidean, auc_geodesic))
                    observed_delta = float(best_auc - base_auc)

                    rng = np.random.default_rng(
                        15_820
                        + domain_index * 10_000
                        + seed_index * 1_000
                        + split_index * 100
                        + layer
                    )
                    null_random = np.empty(H28_NULL_PERM_RANDOM, dtype=float)
                    for perm_idx in range(H28_NULL_PERM_RANDOM):
                        y_perm = rng.permutation(labels)
                        auc_e_perm = safe_auc(y_perm, euclidean_scores)
                        auc_g_perm = safe_auc(y_perm, geodesic_scores)
                        best_d_perm = max(
                            safe_auc(y_perm, diffusion_scores_by_t[t]) for t in H28_DIFFUSION_TIMES
                        )
                        null_random[perm_idx] = float(best_d_perm - max(auc_e_perm, auc_g_perm))
                        null_rows.append(
                            {
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "null_type": "random_label",
                                "perm_idx": int(perm_idx),
                                "delta_best_diffusion_minus_baseline": float(null_random[perm_idx]),
                            }
                        )

                    null_matched = np.empty(H28_NULL_PERM_MATCHED, dtype=float)
                    for perm_idx in range(H28_NULL_PERM_MATCHED):
                        y_perm = stratified_permutation(labels, matched_strata, rng=rng)
                        auc_e_perm = safe_auc(y_perm, euclidean_scores)
                        auc_g_perm = safe_auc(y_perm, geodesic_scores)
                        best_d_perm = max(
                            safe_auc(y_perm, diffusion_scores_by_t[t]) for t in H28_DIFFUSION_TIMES
                        )
                        null_matched[perm_idx] = float(best_d_perm - max(auc_e_perm, auc_g_perm))
                        null_rows.append(
                            {
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "null_type": "coexpression_degree_matched_label",
                                "perm_idx": int(perm_idx),
                                "delta_best_diffusion_minus_baseline": float(null_matched[perm_idx]),
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
                            "best_diffusion_t": int(best_t),
                            "best_diffusion_auc": float(best_auc),
                            "best_baseline_auc": float(base_auc),
                            "delta_best_diffusion_minus_baseline": float(observed_delta),
                            "p_random_label_upper": float(empirical_upper_tail_p(observed_delta, null_random)),
                            "p_coexp_degree_matched_upper": float(
                                empirical_upper_tail_p(observed_delta, null_matched)
                            ),
                            "null_random_mean": float(np.mean(null_random)),
                            "null_matched_mean": float(np.mean(null_matched)),
                            "coexpression_abs_mean": float(np.mean(coexp_scores)),
                            "coexpression_missing_fraction": float(missing_fraction),
                            "auc_diffusion_t1": float(diffusion_auc_by_t[1]),
                            "auc_diffusion_t2": float(diffusion_auc_by_t[2]),
                            "auc_diffusion_t4": float(diffusion_auc_by_t[4]),
                            "auc_diffusion_t8": float(diffusion_auc_by_t[8]),
                        }
                    )

    row_df = pd.DataFrame(rows).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    row_path = ITER_DIR / "h28_diffusion_coexp_by_seed_layer_split.csv"
    row_df.to_csv(row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["domain", "seed_tag", "split_regime", "layer", "null_type", "perm_idx"]
    )
    null_path = ITER_DIR / "h28_diffusion_coexp_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    domain_rows: list[dict[str, object]] = []
    for domain, group in row_df.groupby("domain", sort=True):
        domain_rows.append(
            {
                "domain": domain,
                "n_rows": int(group.shape[0]),
                "mean_best_diffusion_auc": float(group["best_diffusion_auc"].mean()),
                "mean_best_baseline_auc": float(group["best_baseline_auc"].mean()),
                "mean_delta_best_diffusion_minus_baseline": float(
                    group["delta_best_diffusion_minus_baseline"].mean()
                ),
                "fraction_delta_positive": float(
                    (group["delta_best_diffusion_minus_baseline"] > 0).mean()
                ),
                "fraction_p_random_lt_0_05": float((group["p_random_label_upper"] < 0.05).mean()),
                "fraction_p_matched_lt_0_05": float(
                    (group["p_coexp_degree_matched_upper"] < 0.05).mean()
                ),
                "combined_fisher_p_random": safe_fisher_p(
                    group["p_random_label_upper"].to_numpy(dtype=float)
                ),
                "combined_fisher_p_matched": safe_fisher_p(
                    group["p_coexp_degree_matched_upper"].to_numpy(dtype=float)
                ),
                "mean_coexpression_missing_fraction": float(
                    group["coexpression_missing_fraction"].mean()
                ),
            }
        )

    domain_df = pd.DataFrame(domain_rows).sort_values("domain")
    domain_path = ITER_DIR / "h28_diffusion_coexp_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    summary = {
        "rows_tested": int(row_df.shape[0]),
        "domains_tested": int(domain_df.shape[0]),
        "domains_positive_mean_delta": int(
            (domain_df["mean_delta_best_diffusion_minus_baseline"] > 0).sum()
        )
        if not domain_df.empty
        else 0,
        "domains_sig_fisher_matched": int((domain_df["combined_fisher_p_matched"] < 0.05).sum())
        if not domain_df.empty
        else 0,
        "mean_delta_overall": float(row_df["delta_best_diffusion_minus_baseline"].mean())
        if not row_df.empty
        else float("nan"),
        "artifact_paths": {
            "by_seed_layer_split": str(row_path),
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
        },
    }
    return summary


def run_seeded_gw(
    c1: np.ndarray,
    c2: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    g0: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    coupling = g0.astype(np.float64, copy=True)
    gw_dist = float("nan")

    for eps in H29_EPSILON_SCHEDULE:
        try:
            coupling, log = ot.gromov.entropic_gromov_wasserstein(
                c1,
                c2,
                p,
                q,
                loss_fun="square_loss",
                epsilon=float(eps),
                max_iter=500,
                tol=1e-9,
                G0=coupling,
                log=True,
                verbose=False,
            )
            gw_dist = float(log.get("gw_dist", gw_dist))
            continue
        except TypeError:
            # POT versions without G0 support: fall back to solver call without init.
            coupling, log = ot.gromov.entropic_gromov_wasserstein(
                c1,
                c2,
                p,
                q,
                loss_fun="square_loss",
                epsilon=float(eps),
                max_iter=500,
                tol=1e-9,
                log=True,
                verbose=False,
            )
            gw_dist = float(log.get("gw_dist", gw_dist))
            continue
        except Exception:
            pass

    if coupling is None or (not np.isfinite(coupling).all()) or float(np.sum(coupling)) <= 1e-12:
        coupling = g0.astype(np.float64, copy=True)

    return coupling, {"gw_dist": gw_dist}


def run_h29_seeded_gw_alignment() -> dict[str, object]:
    geneformer_embeddings = load_geneformer_embeddings()
    h27_reference = (
        pd.read_csv(H27_REFERENCE_DOMAIN_SUMMARY).set_index("domain")
        if H27_REFERENCE_DOMAIN_SUMMARY.exists()
        else pd.DataFrame()
    )

    domain_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []

    for domain_index, domain in enumerate(["immune", "lung", "external_lung"]):
        edge_df = pd.read_csv(GENEFORMER_EDGE_BY_DOMAIN[domain], sep="\t")
        mapping_df = select_mapped_genes(
            edge_df,
            gene_cap=H29_GENE_CAP,
            geneformer_vocab_size=geneformer_embeddings.shape[0],
        )
        if mapping_df.shape[0] < 120:
            continue

        gene_ids = mapping_df["gene_idx"].to_numpy(dtype=int)
        token_ids = mapping_df["token_id"].to_numpy(dtype=int)
        n_items = int(gene_ids.size)

        scgpt_run_dir = SCGPT_RUNS_BY_DOMAIN[domain]["seed42_main"]
        layer = int(CROSS_MODEL_LAYER_BY_DOMAIN[domain])
        layer_embeddings = np.load(scgpt_run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        if layer >= layer_embeddings.shape[0]:
            continue

        x_raw = layer_embeddings[layer, gene_ids, :].astype(np.float64)
        y_raw = geneformer_embeddings[token_ids, :]

        n_pca = min(48, n_items - 1, x_raw.shape[1], y_raw.shape[1])
        if n_pca < 12:
            continue

        x_pca = reduce_points(x_raw, n_components=n_pca, random_state=15_900 + domain_index * 10)
        y_pca = reduce_points(y_raw, n_components=n_pca, random_state=15_901 + domain_index * 10)

        n_cca = min(24, x_pca.shape[1], y_pca.shape[1], n_items - 1)
        x_cca, y_cca, canonical_corr = linear_cca_projection(x_pca, y_pca, n_components=n_cca, reg=1e-3)

        cca_cost = cdist(x_cca, y_cca, metric="euclidean")
        row_ind, col_ind = linear_sum_assignment(cca_cost)
        cca_seed_assignment = np.empty(n_items, dtype=int)
        cca_seed_assignment[row_ind] = col_ind

        p = np.ones(n_items, dtype=np.float64) / float(n_items)
        q = np.ones(n_items, dtype=np.float64) / float(n_items)
        g0 = np.zeros((n_items, n_items), dtype=np.float64)
        g0[np.arange(n_items), cca_seed_assignment] = 1.0 / float(n_items)

        c1 = cdist(x_cca, x_cca, metric="euclidean")
        c2 = cdist(y_cca, y_cca, metric="euclidean")
        c1_scale = float(np.median(c1[c1 > 0])) if np.any(c1 > 0) else 1.0
        c2_scale = float(np.median(c2[c2 > 0])) if np.any(c2 > 0) else 1.0
        c1 = c1 / max(c1_scale, 1e-6)
        c2 = c2 / max(c2_scale, 1e-6)

        coupling, gw_log = run_seeded_gw(c1=c1, c2=c2, p=p, q=q, g0=g0)
        if not np.isfinite(coupling).all() or float(np.sum(coupling)) <= 1e-12:
            coupling = g0

        row_ind, col_ind = linear_sum_assignment(-coupling)
        seeded_assignment = np.empty(n_items, dtype=int)
        seeded_assignment[row_ind] = col_ind

        top1_seeded = float(np.mean(seeded_assignment == np.arange(n_items)))
        reverse_seeded = np.empty(n_items, dtype=int)
        reverse_seeded[seeded_assignment] = np.arange(n_items)
        mutual_top1_seeded = float(
            np.mean((seeded_assignment == np.arange(n_items)) & (reverse_seeded == np.arange(n_items)))
        )

        top1_cca = float(np.mean(cca_seed_assignment == np.arange(n_items)))
        reverse_cca = np.argmax(np.eye(n_items)[cca_seed_assignment], axis=0)
        mutual_top1_cca = float(
            np.mean((cca_seed_assignment == np.arange(n_items)) & (reverse_cca == np.arange(n_items)))
        )

        y_seeded = y_cca[seeded_assignment]
        y_cca_map = y_cca[cca_seed_assignment]

        dist_rho_seeded = safe_spearman(pdist(x_cca), pdist(y_seeded))
        dist_rho_cca = safe_spearman(pdist(x_cca), pdist(y_cca_map))
        jaccard_seeded = mean_knn_jaccard(x_cca, y_seeded, n_neighbors=min(10, n_items - 1))
        jaccard_cca = mean_knn_jaccard(x_cca, y_cca_map, n_neighbors=min(10, n_items - 1))

        edge_sub = edge_df.loc[
            edge_df["source_idx"].isin(set(gene_ids)) & edge_df["target_idx"].isin(set(gene_ids))
        ].copy()
        gene_to_pos = {int(g): int(i) for i, g in enumerate(gene_ids)}
        source_local = edge_sub["source_idx"].map(gene_to_pos).to_numpy(dtype=int)
        target_local = edge_sub["target_idx"].map(gene_to_pos).to_numpy(dtype=int)
        labels = edge_sub["label"].to_numpy(dtype=int)

        auc_seeded = transfer_auc_from_mapping(
            seeded_assignment,
            source_local=source_local,
            target_local=target_local,
            labels=labels,
            y_points=y_cca,
        )
        auc_cca = transfer_auc_from_mapping(
            cca_seed_assignment,
            source_local=source_local,
            target_local=target_local,
            labels=labels,
            y_points=y_cca,
        )

        rng = np.random.default_rng(15_930 + domain_index * 11)
        null_top1 = np.empty(H29_NULL_PERM, dtype=float)
        null_mutual = np.empty(H29_NULL_PERM, dtype=float)
        null_auc = np.empty(H29_NULL_PERM, dtype=float)

        for perm_idx in range(H29_NULL_PERM):
            perm = rng.permutation(n_items)
            null_top1[perm_idx] = float(np.mean(perm == np.arange(n_items)))
            null_mutual[perm_idx] = float(np.mean(perm == np.arange(n_items)))
            null_auc[perm_idx] = transfer_auc_from_mapping(
                perm,
                source_local=source_local,
                target_local=target_local,
                labels=labels,
                y_points=y_cca,
            )
            null_rows.append(
                {
                    "domain": domain,
                    "perm_idx": int(perm_idx),
                    "null_top1": float(null_top1[perm_idx]),
                    "null_mutual_top1": float(null_mutual[perm_idx]),
                    "null_edge_transfer_auc": float(null_auc[perm_idx]),
                }
            )

        p_top1 = empirical_upper_tail_p(top1_seeded, null_top1)
        p_mutual = empirical_upper_tail_p(mutual_top1_seeded, null_mutual)
        p_auc = empirical_upper_tail_p(auc_seeded, null_auc)

        h27_top1 = float("nan")
        h27_auc = float("nan")
        if not h27_reference.empty and domain in h27_reference.index:
            h27_top1 = float(h27_reference.loc[domain, "top1_retrieval_gw"])
            h27_auc = float(h27_reference.loc[domain, "edge_transfer_auc_gw"])

        domain_rows.append(
            {
                "domain": domain,
                "layer": int(layer),
                "n_matched_genes": int(n_items),
                "pca_dim": int(n_pca),
                "cca_dim": int(n_cca),
                "mean_canonical_corr": float(np.mean(canonical_corr)) if canonical_corr.size else float("nan"),
                "gw_objective_seeded": float(gw_log.get("gw_dist", float("nan"))),
                "top1_seeded_gw": float(top1_seeded),
                "mutual_top1_seeded_gw": float(mutual_top1_seeded),
                "top1_cca_seed": float(top1_cca),
                "mutual_top1_cca_seed": float(mutual_top1_cca),
                "distance_spearman_seeded_gw": float(dist_rho_seeded),
                "distance_spearman_cca_seed": float(dist_rho_cca),
                "knn_jaccard_seeded_gw": float(jaccard_seeded),
                "knn_jaccard_cca_seed": float(jaccard_cca),
                "edge_transfer_auc_seeded_gw": float(auc_seeded),
                "edge_transfer_auc_cca_seed": float(auc_cca),
                "h27_unseeded_top1_reference": float(h27_top1),
                "h27_unseeded_edge_auc_reference": float(h27_auc),
                "delta_top1_vs_h27": float(top1_seeded - h27_top1) if np.isfinite(h27_top1) else float("nan"),
                "delta_auc_vs_h27": float(auc_seeded - h27_auc) if np.isfinite(h27_auc) else float("nan"),
                "null_mean_top1": float(np.mean(null_top1)),
                "null_mean_edge_transfer_auc": float(np.mean(null_auc)),
                "p_top1_upper": float(p_top1),
                "p_mutual_top1_upper": float(p_mutual),
                "p_edge_transfer_auc_upper": float(p_auc),
            }
        )

        map_rows.append(
            {
                "domain": domain,
                "n_matched_genes": int(n_items),
                "top1_seeded_gw": float(top1_seeded),
                "mutual_top1_seeded_gw": float(mutual_top1_seeded),
                "top1_cca_seed": float(top1_cca),
                "mutual_top1_cca_seed": float(mutual_top1_cca),
                "delta_top1_vs_h27": float(top1_seeded - h27_top1) if np.isfinite(h27_top1) else float("nan"),
                "p_top1_upper": float(p_top1),
                "edge_transfer_auc_seeded_gw": float(auc_seeded),
                "edge_transfer_auc_cca_seed": float(auc_cca),
                "delta_auc_vs_h27": float(auc_seeded - h27_auc) if np.isfinite(h27_auc) else float("nan"),
                "p_edge_transfer_auc_upper": float(p_auc),
            }
        )

    domain_df = pd.DataFrame(domain_rows).sort_values("domain")
    domain_path = ITER_DIR / "h29_seeded_gw_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(["domain", "perm_idx"])
    null_path = ITER_DIR / "h29_seeded_gw_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    map_df = pd.DataFrame(map_rows).sort_values("domain")
    map_path = ITER_DIR / "h29_seeded_gw_map_quality.csv"
    map_df.to_csv(map_path, index=False)

    summary = {
        "domains_tested": int(domain_df.shape[0]),
        "domains_sig_top1": int((domain_df["p_top1_upper"] < 0.05).sum()) if not domain_df.empty else 0,
        "domains_sig_transfer_auc": int((domain_df["p_edge_transfer_auc_upper"] < 0.05).sum())
        if not domain_df.empty
        else 0,
        "mean_top1_seeded": float(domain_df["top1_seeded_gw"].mean())
        if not domain_df.empty
        else float("nan"),
        "mean_transfer_auc_seeded": float(domain_df["edge_transfer_auc_seeded_gw"].mean())
        if not domain_df.empty
        else float("nan"),
        "mean_delta_top1_vs_h27": float(domain_df["delta_top1_vs_h27"].mean())
        if not domain_df.empty
        else float("nan"),
        "mean_delta_auc_vs_h27": float(domain_df["delta_auc_vs_h27"].mean())
        if not domain_df.empty
        else float("nan"),
        "combined_fisher_top1": safe_fisher_p(domain_df["p_top1_upper"].to_numpy(dtype=float))
        if not domain_df.empty
        else float("nan"),
        "combined_fisher_transfer_auc": safe_fisher_p(
            domain_df["p_edge_transfer_auc_upper"].to_numpy(dtype=float)
        )
        if not domain_df.empty
        else float("nan"),
        "artifact_paths": {
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
            "map_quality": str(map_path),
        },
    }
    return summary


def triangle_delta_from_sides(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    max_side = np.maximum(np.maximum(a, b), c)
    return 0.5 * np.maximum(0.0, a + b + c - 2.0 * max_side)


def compute_edge_thinness_scores(
    geodesic: np.ndarray,
    knn_idx: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    rng: np.random.Generator,
    n_third_samples: int,
) -> np.ndarray:
    n_nodes = geodesic.shape[0]
    all_nodes = np.arange(n_nodes, dtype=int)
    thinness = np.empty(source_local.size, dtype=np.float64)

    for i in range(source_local.size):
        src = int(source_local[i])
        tgt = int(target_local[i])

        candidates = np.unique(np.concatenate([knn_idx[src], knn_idx[tgt]])).astype(int)
        candidates = candidates[(candidates != src) & (candidates != tgt)]

        if candidates.size < 3:
            fallback = all_nodes[(all_nodes != src) & (all_nodes != tgt)]
            take = min(n_third_samples, fallback.size)
            candidates = rng.choice(fallback, size=take, replace=False)
        elif candidates.size > n_third_samples:
            candidates = rng.choice(candidates, size=n_third_samples, replace=False)

        a = np.full(candidates.size, geodesic[src, tgt], dtype=np.float64)
        b = geodesic[src, candidates]
        c = geodesic[tgt, candidates]
        delta = triangle_delta_from_sides(a, b, c)
        thinness[i] = float(np.mean(delta)) if delta.size else 0.0

    return -thinness


def run_h30_hyperbolicity_screen() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(SCGPT_RUNS_BY_DOMAIN.items()):
        # Cheap broad-screen variant: one representative seed per domain.
        seed_tag = "seed42_main"
        run_dir = run_map[seed_tag]

        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = select_top_genes(split_edges, gene_cap=H30_GENE_CAP)
            top_gene_set = set(top_genes)
            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(top_gene_set)
                & split_edges["target_idx"].isin(top_gene_set)
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
            gene_to_local = {
                int(gene_idx): int(local_idx)
                for local_idx, gene_idx in enumerate(edge_gene_indices)
            }
            source_local = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
            target_local = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
            labels = split_edges["label"].to_numpy(dtype=int)

            source_degree_map = split_edges["source_idx"].value_counts().to_dict()
            target_degree_map = split_edges["target_idx"].value_counts().to_dict()
            source_degree = split_edges["source_idx"].map(source_degree_map).astype(float).to_numpy()
            target_degree = split_edges["target_idx"].map(target_degree_map).astype(float).to_numpy()
            mean_degree = 0.5 * (source_degree + target_degree)

            for layer in H30_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                random_state = 16_000 + domain_index * 100 + split_index * 10 + layer
                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = reduce_points(points, n_components=24, random_state=random_state)
                geodesic, _, k_eff, used_component_bridging = geodesic_and_transition(
                    points_pca,
                    n_neighbors=24,
                )
                knn_idx = get_knn_indices(points_pca, n_neighbors=min(16, points_pca.shape[0] - 1))

                rng = np.random.default_rng(16_010 + domain_index * 100 + split_index * 10 + layer)
                thinness_score = compute_edge_thinness_scores(
                    geodesic=geodesic,
                    knn_idx=knn_idx,
                    source_local=source_local,
                    target_local=target_local,
                    rng=rng,
                    n_third_samples=H30_THIRD_NODE_SAMPLES,
                )
                auc_thinness = safe_auc(labels, thinness_score)

                geodesic_score = -geodesic[source_local, target_local]
                auc_geodesic = safe_auc(labels, geodesic_score)

                edge_distance = geodesic[source_local, target_local]
                degree_bins = degree_strata(mean_degree, max_bins=5)
                distance_bins = degree_strata(edge_distance, max_bins=5)
                matched_strata = combine_strata(degree_bins, distance_bins)

                null_auc = np.empty(H30_NULL_PERM, dtype=float)
                for perm_idx in range(H30_NULL_PERM):
                    y_perm = stratified_permutation(labels, matched_strata, rng=rng)
                    null_auc[perm_idx] = safe_auc(y_perm, thinness_score)
                    null_rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_auc_thinness": float(null_auc[perm_idx]),
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
                        "auc_thinness": float(auc_thinness),
                        "auc_geodesic": float(auc_geodesic),
                        "delta_auc_thinness_minus_geodesic": float(auc_thinness - auc_geodesic),
                        "p_auc_thinness_upper": float(empirical_upper_tail_p(auc_thinness, null_auc)),
                        "null_mean_auc": float(np.mean(null_auc)),
                        "null_std_auc": float(np.std(null_auc, ddof=1)) if null_auc.size > 1 else 0.0,
                    }
                )

    row_df = pd.DataFrame(rows).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    row_path = ITER_DIR / "h30_hyperbolicity_by_seed_layer_split.csv"
    row_df.to_csv(row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["domain", "seed_tag", "split_regime", "layer", "perm_idx"]
    )
    null_path = ITER_DIR / "h30_hyperbolicity_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    domain_rows: list[dict[str, object]] = []
    for (domain, split_regime), group in row_df.groupby(["domain", "split_regime"], sort=True):
        domain_rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(group.shape[0]),
                "mean_auc_thinness": float(group["auc_thinness"].mean()),
                "mean_auc_geodesic": float(group["auc_geodesic"].mean()),
                "mean_delta_auc_thinness_minus_geodesic": float(
                    group["delta_auc_thinness_minus_geodesic"].mean()
                ),
                "fraction_p_lt_0_05": float((group["p_auc_thinness_upper"] < 0.05).mean()),
                "combined_fisher_p": safe_fisher_p(group["p_auc_thinness_upper"].to_numpy(dtype=float)),
            }
        )

    domain_df = pd.DataFrame(domain_rows).sort_values(["domain", "split_regime"])
    domain_path = ITER_DIR / "h30_hyperbolicity_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    summary = {
        "rows_tested": int(row_df.shape[0]),
        "domain_split_groups": int(domain_df.shape[0]),
        "groups_mean_auc_gt_0_5": int((domain_df["mean_auc_thinness"] > 0.5).sum())
        if not domain_df.empty
        else 0,
        "groups_sig_fisher": int((domain_df["combined_fisher_p"] < 0.05).sum())
        if not domain_df.empty
        else 0,
        "mean_auc_thinness": float(row_df["auc_thinness"].mean()) if not row_df.empty else float("nan"),
        "mean_auc_geodesic": float(row_df["auc_geodesic"].mean()) if not row_df.empty else float("nan"),
        "artifact_paths": {
            "by_seed_layer_split": str(row_path),
            "domain_summary": str(domain_path),
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
    required_paths.append(H27_REFERENCE_DOMAIN_SUMMARY)

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    coexp_cache: dict[str, CoexpressionDomainCache] = {
        domain: CoexpressionDomainCache(path)
        for domain, path in PROCESSED_H5AD_BY_DOMAIN.items()
    }

    try:
        h28_summary = run_h28_diffusion_stronger_nulls(coexp_cache=coexp_cache)
        h29_summary = run_h29_seeded_gw_alignment()
        h30_summary = run_h30_hyperbolicity_screen()
    finally:
        for cache in coexp_cache.values():
            cache.close()

    iteration_summary = {
        "iteration": "iter_0015",
        "inputs": {
            "scgpt_runs_by_domain": {
                domain: {seed: str(path) for seed, path in run_dict.items()}
                for domain, run_dict in SCGPT_RUNS_BY_DOMAIN.items()
            },
            "geneformer_edge_by_domain": {
                domain: str(path) for domain, path in GENEFORMER_EDGE_BY_DOMAIN.items()
            },
            "processed_h5ad_by_domain": {
                domain: str(path) for domain, path in PROCESSED_H5AD_BY_DOMAIN.items()
            },
            "h27_reference_domain_summary": str(H27_REFERENCE_DOMAIN_SUMMARY),
            "cross_model_layer_by_domain": CROSS_MODEL_LAYER_BY_DOMAIN,
            "h28_note": "Coexpression-aware null uses absolute Pearson correlation bins computed from backed processed h5ad expression matrices and combined with degree bins.",
            "h30_note": "Cheap broad-screen uses one seed per domain (seed42_main) by design for rapid triage.",
        },
        "h28_diffusion_coexpression_matched_null": h28_summary,
        "h29_seeded_one_to_one_gw": h29_summary,
        "h30_triangle_thinness_screen": h30_summary,
    }

    summary_path = ITER_DIR / "iter0015_screen_summary.json"
    summary_path.write_text(json.dumps(iteration_summary, indent=2))
    print(json.dumps(iteration_summary, indent=2))


if __name__ == "__main__":
    main()
