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


ITER_DIR = Path("iterations/iter_0017")
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

TRRUST_PATH = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
    "single_cell_mechinterp/external/networks/trrust_human.tsv"
)

# Iteration constants for the N141/N147/N149 packet.
H34_LAYERS = [0, 3, 7, 11]
H34_DIFFUSION_TIMES = [1, 2, 4, 8]
H34_GENE_CAP = 260
H34_NULL_PERM = 80
H34_MAX_COEXP_CELLS = 5000

H35_GENE_CAP = 260
H35_NEIGHBORS = 12
H35_NULL_PERM = 300

H36_GENE_CAP = 240
H36_LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75]
H36_ANCHOR_BONUS = 2.0
H36_NULL_PERM = 120
H36_LAYER_BY_DOMAIN = {"immune": 0, "lung": 0, "external_lung": 3}
H36_CYCLE_BASELINE_PATH = Path("iterations/iter_0016/h33_cycle_consistent_alignment_domain_summary.csv")


@dataclass
class CoexpressionDomainCache:
    """
    Backed expression cache for fast coexpression controls.

    The cache is opened once per domain and reused across all tested rows to
    keep the screening iteration lightweight.
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


def empirical_lower_tail_p(observed: float, null_values: np.ndarray) -> float:
    if not np.isfinite(observed):
        return float("nan")
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float((1 + np.sum(values <= observed)) / (values.size + 1))


def empirical_two_sided_p(observed: float, null_values: np.ndarray) -> float:
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float((1 + np.sum(np.abs(values) >= abs(observed))) / (values.size + 1))


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


def safe_log_loss(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    return float(log_loss(labels, clipped))


def safe_brier_gain(labels: np.ndarray, scores_base: np.ndarray, scores_ext: np.ndarray) -> float:
    labels_f = np.asarray(labels, dtype=float)
    s_base = np.asarray(scores_base, dtype=float)
    s_ext = np.asarray(scores_ext, dtype=float)
    if labels_f.size == 0:
        return float("nan")
    brier_base = np.mean(np.square(labels_f - s_base))
    brier_ext = np.mean(np.square(labels_f - s_ext))
    return float(brier_base - brier_ext)


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


def get_knn_indices(points: np.ndarray, n_neighbors: int) -> np.ndarray:
    n_points = points.shape[0]
    k = max(1, min(n_neighbors, n_points - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(points)
    _, indices = nbrs.kneighbors(points)
    return indices[:, 1:]


def compute_local_reconstruction_errors(points: np.ndarray, neighbor_idx: np.ndarray) -> np.ndarray:
    n_points, n_neighbors = neighbor_idx.shape
    errors = np.empty(n_points, dtype=np.float64)
    ones = np.ones(n_neighbors, dtype=np.float64)
    for idx in range(n_points):
        neighborhood = points[neighbor_idx[idx]]
        z = neighborhood - points[idx]
        c = z @ z.T
        trace = float(np.trace(c))
        if trace <= 1e-12:
            errors[idx] = 0.0
            continue
        c.flat[:: n_neighbors + 1] += 1e-3 * trace
        weights = np.linalg.solve(c, ones)
        weights /= np.clip(weights.sum(), 1e-12, None)
        recon = np.sum(weights[:, None] * neighborhood, axis=0)
        numer = float(np.square(points[idx] - recon).sum())
        denom = float(np.square(points[idx]).sum()) + 1e-12
        errors[idx] = numer / denom
    return errors


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if x.size < 2:
        return 0.0, float(y.mean()) if y.size else 0.0, float("inf")
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = slope * x + intercept
    sse = float(np.sum(np.square(y - pred)))
    return float(slope), float(intercept), sse


def fit_best_piecewise_breakpoint(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    _, _, sse_linear = fit_line(x, y)

    best = {
        "breakpoint": float("nan"),
        "pre_slope": float("nan"),
        "post_slope": float("nan"),
        "sse_piecewise": float("inf"),
    }

    unique_x = np.unique(x.astype(int))
    if unique_x.size < 4:
        return {
            "breakpoint": float("nan"),
            "pre_slope": float("nan"),
            "post_slope": float("nan"),
            "sse_linear": float(sse_linear),
            "sse_piecewise": float("nan"),
            "improvement": float("nan"),
        }

    for bp in unique_x[1:-1]:
        left = x <= bp
        right = x > bp
        if left.sum() < 2 or right.sum() < 2:
            continue
        slope_left, intercept_left, sse_left = fit_line(x[left], y[left])
        slope_right, intercept_right, sse_right = fit_line(x[right], y[right])
        sse = sse_left + sse_right
        if sse < best["sse_piecewise"]:
            best = {
                "breakpoint": float(bp),
                "pre_slope": float(slope_left),
                "post_slope": float(slope_right),
                "sse_piecewise": float(sse),
            }

    if not np.isfinite(best["sse_piecewise"]):
        return {
            "breakpoint": float("nan"),
            "pre_slope": float("nan"),
            "post_slope": float("nan"),
            "sse_linear": float(sse_linear),
            "sse_piecewise": float("nan"),
            "improvement": float("nan"),
        }

    return {
        "breakpoint": best["breakpoint"],
        "pre_slope": best["pre_slope"],
        "post_slope": best["post_slope"],
        "sse_linear": float(sse_linear),
        "sse_piecewise": float(best["sse_piecewise"]),
        "improvement": float(sse_linear - best["sse_piecewise"]),
    }


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


def weighted_orthogonal_procrustes(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.sqrt(np.clip(np.asarray(weights, dtype=float), 1e-8, None))[:, None]
    return orthogonal_procrustes_map(x * w, y * w)


def load_geneformer_embeddings() -> np.ndarray:
    model = AutoModel.from_pretrained("ctheodoris/Geneformer")
    embeddings = model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float64, copy=False)
    del model
    return embeddings


def build_symbol_maps_from_edge_table(
    edge_df: pd.DataFrame,
    source_cols: tuple[str, str],
    target_cols: tuple[str, str],
) -> tuple[dict[str, int], dict[str, int]]:
    source_symbol_col, source_index_col = source_cols
    target_symbol_col, target_index_col = target_cols

    symbol_to_index: dict[str, int] = {}
    degree_counter: dict[str, int] = {}

    for symbol, idx in zip(edge_df[source_symbol_col].astype(str), edge_df[source_index_col].astype(int)):
        sym = symbol.upper()
        symbol_to_index.setdefault(sym, int(idx))
        degree_counter[sym] = degree_counter.get(sym, 0) + 1

    for symbol, idx in zip(edge_df[target_symbol_col].astype(str), edge_df[target_index_col].astype(int)):
        sym = symbol.upper()
        symbol_to_index.setdefault(sym, int(idx))
        degree_counter[sym] = degree_counter.get(sym, 0) + 1

    return symbol_to_index, degree_counter


def load_trrust_source_degree() -> dict[str, int]:
    trrust = pd.read_csv(
        TRRUST_PATH,
        sep="\t",
        header=None,
        names=["source", "target", "regulation", "pmid"],
    )
    source_series = trrust["source"].astype(str).str.upper()
    counts = source_series.value_counts()
    return {str(g): int(c) for g, c in counts.items()}


def build_positive_adjacency_from_symbol_edges(
    edge_df: pd.DataFrame,
    source_col: str,
    target_col: str,
    label_col: str,
    selected_symbols: list[str],
) -> np.ndarray:
    n = len(selected_symbols)
    symbol_to_pos = {s: i for i, s in enumerate(selected_symbols)}
    adj = np.zeros((n, n), dtype=np.float64)

    for row in edge_df[[source_col, target_col, label_col]].itertuples(index=False):
        src = str(getattr(row, source_col)).upper()
        tgt = str(getattr(row, target_col)).upper()
        label = int(getattr(row, label_col))
        if label <= 0:
            continue
        if src not in symbol_to_pos or tgt not in symbol_to_pos:
            continue
        i = symbol_to_pos[src]
        j = symbol_to_pos[tgt]
        if i == j:
            continue
        adj[i, j] += 1.0
        adj[j, i] += 1.0

    if np.sum(adj) <= 0:
        # Avoid singular all-zero spectrum.
        adj = np.eye(n, dtype=np.float64)
    return adj


def normalized_laplacian_spectral_embedding(adj: np.ndarray, n_components: int) -> np.ndarray:
    n = adj.shape[0]
    deg = np.sum(adj, axis=1)
    inv_sqrt_deg = 1.0 / np.sqrt(np.clip(deg, 1e-8, None))
    d_half = np.diag(inv_sqrt_deg)
    lap = np.eye(n, dtype=np.float64) - d_half @ adj @ d_half

    evals, evecs = np.linalg.eigh(lap)
    order = np.argsort(evals)
    # Skip the first trivial eigenvector.
    keep = order[1 : 1 + min(n_components, n - 1)]
    vecs = evecs[:, keep]
    vecs = vecs - vecs.mean(axis=0, keepdims=True)
    std = vecs.std(axis=0, ddof=1, keepdims=True)
    std = np.where(std <= 1e-12, 1.0, std)
    return vecs / std


def edge_auc_for_mask(
    z: np.ndarray,
    src_pos: np.ndarray,
    tgt_pos: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> float:
    if np.sum(mask) < 10:
        return float("nan")
    scores = np.sum(z[src_pos[mask]] * z[tgt_pos[mask]], axis=1)
    return safe_auc(labels[mask], scores)


def run_h34_convexity_detour_multiseed(
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

                top_genes = select_top_genes(split_edges, gene_cap=H34_GENE_CAP)
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
                    max_cells=H34_MAX_COEXP_CELLS,
                    random_state=17_100 + domain_index * 100 + seed_index * 10 + split_index,
                )
                coexp_scores = coexp_abs_corr[source_local, target_local]

                source_degree_map = split_edges["source_idx"].value_counts().to_dict()
                target_degree_map = split_edges["target_idx"].value_counts().to_dict()
                source_degree = split_edges["source_idx"].map(source_degree_map).astype(float).to_numpy()
                target_degree = split_edges["target_idx"].map(target_degree_map).astype(float).to_numpy()
                mean_degree = 0.5 * (source_degree + target_degree)

                for layer in H34_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue
                    random_state = (
                        17_110
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
                    euclidean_distance = np.clip(euclidean[source_local, target_local], 1e-6, None)

                    diffusion_cols: list[np.ndarray] = []
                    for t in H34_DIFFUSION_TIMES:
                        diff_t = diffusion_distance_scores(
                            transition=transition,
                            source_local=source_local,
                            target_local=target_local,
                            t=t,
                        )
                        diffusion_cols.append(-diff_t)
                    diffusion_features = np.column_stack(diffusion_cols)

                    detour_ratio = geodesic_distance / euclidean_distance

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

                    baseline_features = np.column_stack(
                        [
                            np.log1p(source_degree),
                            np.log1p(target_degree),
                            coexp_scores,
                            -euclidean_distance,
                            -geodesic_distance,
                            diffusion_features,
                        ]
                    )
                    added_features = np.column_stack([detour_ratio, convexity_deficit])
                    extended_features = np.column_stack([baseline_features, added_features])

                    base_scores = fit_logistic_scores(baseline_features, labels)
                    ext_scores = fit_logistic_scores(extended_features, labels)
                    auc_base = safe_auc(labels, base_scores)
                    auc_ext = safe_auc(labels, ext_scores)
                    delta_auc = float(auc_ext - auc_base)

                    ll_base = safe_log_loss(labels, base_scores)
                    ll_ext = safe_log_loss(labels, ext_scores)
                    gain_logloss = float(ll_base - ll_ext)
                    gain_brier = float(safe_brier_gain(labels, base_scores, ext_scores))

                    degree_bins = degree_strata(mean_degree, max_bins=5)
                    coexp_bins = degree_strata(coexp_scores, max_bins=5)
                    geodesic_bins = degree_strata(geodesic_distance, max_bins=5)
                    strata = combine_strata(degree_bins, coexp_bins, geodesic_bins)

                    rng = np.random.default_rng(
                        17_120
                        + domain_index * 10_000
                        + seed_index * 1_000
                        + split_index * 100
                        + layer
                    )
                    null_delta_auc = np.empty(H34_NULL_PERM, dtype=float)
                    null_gain_logloss = np.empty(H34_NULL_PERM, dtype=float)
                    null_gain_brier = np.empty(H34_NULL_PERM, dtype=float)

                    for perm_idx in range(H34_NULL_PERM):
                        shuffled_detour = stratified_shuffle(detour_ratio, strata, rng=rng)
                        shuffled_convexity = stratified_shuffle(convexity_deficit, strata, rng=rng)
                        ext_perm = np.column_stack(
                            [baseline_features, shuffled_detour, shuffled_convexity]
                        )
                        perm_scores = fit_logistic_scores(ext_perm, labels)
                        auc_perm = safe_auc(labels, perm_scores)
                        ll_perm = safe_log_loss(labels, perm_scores)
                        brier_gain_perm = safe_brier_gain(labels, base_scores, perm_scores)

                        null_delta_auc[perm_idx] = float(auc_perm - auc_base)
                        null_gain_logloss[perm_idx] = float(ll_base - ll_perm)
                        null_gain_brier[perm_idx] = float(brier_gain_perm)

                        null_rows.append(
                            {
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(null_delta_auc[perm_idx]),
                                "null_gain_logloss": float(null_gain_logloss[perm_idx]),
                                "null_gain_brier": float(null_gain_brier[perm_idx]),
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
                            "auc_geodesic_diffusion_base": float(auc_base),
                            "auc_with_detour_convexity": float(auc_ext),
                            "delta_auc_incremental": float(delta_auc),
                            "gain_logloss_incremental": float(gain_logloss),
                            "gain_brier_incremental": float(gain_brier),
                            "mean_detour_ratio": float(np.mean(detour_ratio)),
                            "mean_convexity_deficit": float(np.mean(convexity_deficit)),
                            "p_delta_auc_upper": float(empirical_upper_tail_p(delta_auc, null_delta_auc)),
                            "p_gain_logloss_upper": float(
                                empirical_upper_tail_p(gain_logloss, null_gain_logloss)
                            ),
                            "p_gain_brier_upper": float(
                                empirical_upper_tail_p(gain_brier, null_gain_brier)
                            ),
                        }
                    )

    row_df = pd.DataFrame(rows).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    row_path = ITER_DIR / "h34_convexity_detour_multiseed_by_seed_layer_split.csv"
    row_df.to_csv(row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["domain", "seed_tag", "split_regime", "layer", "perm_idx"]
    )
    null_path = ITER_DIR / "h34_convexity_detour_multiseed_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    domain_rows: list[dict[str, object]] = []
    for (domain, split_regime), group in row_df.groupby(["domain", "split_regime"], sort=True):
        seed_means = group.groupby("seed_tag", as_index=False)["delta_auc_incremental"].mean()
        domain_rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(group.shape[0]),
                "n_seeds": int(seed_means.shape[0]),
                "mean_auc_base": float(group["auc_geodesic_diffusion_base"].mean()),
                "mean_auc_extended": float(group["auc_with_detour_convexity"].mean()),
                "mean_delta_auc_incremental": float(group["delta_auc_incremental"].mean()),
                "median_delta_auc_incremental": float(group["delta_auc_incremental"].median()),
                "mean_gain_logloss_incremental": float(group["gain_logloss_incremental"].mean()),
                "mean_gain_brier_incremental": float(group["gain_brier_incremental"].mean()),
                "fraction_delta_auc_positive": float((group["delta_auc_incremental"] > 0.0).mean()),
                "fraction_p_delta_lt_0_05": float((group["p_delta_auc_upper"] < 0.05).mean()),
                "combined_fisher_p_delta": float(
                    safe_fisher_p(group["p_delta_auc_upper"].to_numpy(dtype=float))
                ),
                "seed_mean_delta_sign_consistency": float(
                    np.mean(np.sign(seed_means["delta_auc_incremental"].to_numpy(dtype=float)))
                ),
            }
        )

    domain_df = pd.DataFrame(domain_rows).sort_values(["domain", "split_regime"])
    domain_path = ITER_DIR / "h34_convexity_detour_multiseed_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    summary = {
        "rows_tested": int(row_df.shape[0]),
        "domain_split_groups": int(domain_df.shape[0]),
        "groups_mean_delta_positive": int(
            (domain_df["mean_delta_auc_incremental"] > 0.0).sum()
        )
        if not domain_df.empty
        else 0,
        "groups_fisher_sig_delta": int((domain_df["combined_fisher_p_delta"] < 0.05).sum())
        if not domain_df.empty
        else 0,
        "mean_delta_auc_overall": float(row_df["delta_auc_incremental"].mean())
        if not row_df.empty
        else float("nan"),
        "mean_gain_logloss_overall": float(row_df["gain_logloss_incremental"].mean())
        if not row_df.empty
        else float("nan"),
        "mean_gain_brier_overall": float(row_df["gain_brier_incremental"].mean())
        if not row_df.empty
        else float("nan"),
        "artifact_paths": {
            "by_seed_layer_split": str(row_path),
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
        },
    }
    return summary


def run_h35_linearity_breakpoint_screen() -> dict[str, object]:
    row_records: list[dict[str, object]] = []
    null_records: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_index, (seed_tag, run_dir) in enumerate(run_map.items()):
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            split_masks = build_split_masks(edge_df)

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                top_genes = select_top_genes(split_edges, gene_cap=H35_GENE_CAP)
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

                layer_rows: list[dict[str, float]] = []
                for layer in range(layer_embeddings.shape[0]):
                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = reduce_points(
                        points,
                        n_components=24,
                        random_state=17_400
                        + domain_index * 10_000
                        + seed_index * 1_000
                        + split_index * 100
                        + layer,
                    )
                    neighbor_idx = get_knn_indices(points_pca, n_neighbors=H35_NEIGHBORS)
                    recon_errors = compute_local_reconstruction_errors(points_pca, neighbor_idx)

                    edge_recon_mean = 0.5 * (recon_errors[source_local] + recon_errors[target_local])
                    edge_score = -edge_recon_mean
                    auc = safe_auc(labels, edge_score)
                    if not np.isfinite(auc):
                        continue

                    layer_rows.append(
                        {
                            "layer": float(layer),
                            "auc_edge_local_linearity": float(auc),
                            "mean_reconstruction_error": float(np.mean(recon_errors)),
                        }
                    )

                if len(layer_rows) < 6:
                    continue

                layer_df = pd.DataFrame(layer_rows).sort_values("layer")
                x = layer_df["layer"].to_numpy(dtype=float)
                y = layer_df["auc_edge_local_linearity"].to_numpy(dtype=float)

                fit_obs = fit_best_piecewise_breakpoint(x, y)
                rng = np.random.default_rng(
                    17_500 + domain_index * 10_000 + seed_index * 1_000 + split_index * 100
                )
                null_breakpoints = np.empty(H35_NULL_PERM, dtype=float)
                null_improvements = np.empty(H35_NULL_PERM, dtype=float)
                for perm_idx in range(H35_NULL_PERM):
                    y_perm = rng.permutation(y)
                    fit_perm = fit_best_piecewise_breakpoint(x, y_perm)
                    null_breakpoints[perm_idx] = float(fit_perm["breakpoint"])
                    null_improvements[perm_idx] = float(fit_perm["improvement"])
                    null_records.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "perm_idx": int(perm_idx),
                            "null_breakpoint": float(null_breakpoints[perm_idx]),
                            "null_improvement": float(null_improvements[perm_idx]),
                        }
                    )

                row_records.append(
                    {
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "n_layers": int(layer_df.shape[0]),
                        "mean_auc_edge_local_linearity": float(np.mean(y)),
                        "std_auc_edge_local_linearity": float(np.std(y, ddof=1)),
                        "breakpoint_layer": float(fit_obs["breakpoint"]),
                        "pre_break_slope": float(fit_obs["pre_slope"]),
                        "post_break_slope": float(fit_obs["post_slope"]),
                        "sse_linear": float(fit_obs["sse_linear"]),
                        "sse_piecewise": float(fit_obs["sse_piecewise"]),
                        "piecewise_improvement": float(fit_obs["improvement"]),
                        "null_mean_improvement": float(np.nanmean(null_improvements)),
                        "null_mean_breakpoint": float(np.nanmean(null_breakpoints)),
                        "p_piecewise_improvement_upper": float(
                            empirical_upper_tail_p(float(fit_obs["improvement"]), null_improvements)
                        ),
                    }
                )

    row_df = pd.DataFrame(row_records).sort_values(["domain", "seed_tag", "split_regime"])
    row_path = ITER_DIR / "h35_linearity_breakpoint_by_seed_domain_split.csv"
    row_df.to_csv(row_path, index=False)

    null_df = pd.DataFrame(null_records).sort_values(["domain", "seed_tag", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h35_linearity_breakpoint_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    paired_rows = (
        row_df.pivot_table(
            index=["domain", "seed_tag"],
            columns="split_regime",
            values=["breakpoint_layer", "piecewise_improvement", "p_piecewise_improvement_upper"],
            aggfunc="first",
        )
        .dropna()
        .reset_index()
    )
    if not paired_rows.empty:
        paired_rows.columns = [
            "domain",
            "seed_tag",
            "breakpoint_source_disjoint",
            "breakpoint_target_disjoint",
            "improvement_source_disjoint",
            "improvement_target_disjoint",
            "p_improvement_source_disjoint",
            "p_improvement_target_disjoint",
        ]
        paired_rows["breakpoint_shift_source_minus_target"] = (
            paired_rows["breakpoint_source_disjoint"] - paired_rows["breakpoint_target_disjoint"]
        )

        for domain, group in paired_rows.groupby("domain", sort=True):
            null_group = null_df.loc[null_df["domain"] == domain].copy()
            null_shift_values: list[float] = []
            for seed_tag in sorted(group["seed_tag"].unique()):
                source_null = null_group.loc[
                    (null_group["seed_tag"] == seed_tag)
                    & (null_group["split_regime"] == "source_disjoint")
                ].sort_values("perm_idx")
                target_null = null_group.loc[
                    (null_group["seed_tag"] == seed_tag)
                    & (null_group["split_regime"] == "target_disjoint")
                ].sort_values("perm_idx")
                if source_null.empty or target_null.empty:
                    continue
                n = min(source_null.shape[0], target_null.shape[0])
                shift_seed = (
                    source_null["null_breakpoint"].to_numpy(dtype=float)[:n]
                    - target_null["null_breakpoint"].to_numpy(dtype=float)[:n]
                )
                null_shift_values.append(shift_seed)

            if null_shift_values:
                stacked = np.vstack(null_shift_values)
                null_mean_shift = np.mean(stacked, axis=0)
            else:
                null_mean_shift = np.array([], dtype=float)

            observed_mean_shift = float(group["breakpoint_shift_source_minus_target"].mean())
            summary_rows.append(
                {
                    "domain": domain,
                    "n_seed_pairs": int(group.shape[0]),
                    "mean_breakpoint_source_disjoint": float(group["breakpoint_source_disjoint"].mean()),
                    "mean_breakpoint_target_disjoint": float(group["breakpoint_target_disjoint"].mean()),
                    "mean_breakpoint_shift_source_minus_target": observed_mean_shift,
                    "fraction_shift_positive": float(
                        (group["breakpoint_shift_source_minus_target"] > 0).mean()
                    ),
                    "mean_piecewise_improvement_source": float(group["improvement_source_disjoint"].mean()),
                    "mean_piecewise_improvement_target": float(group["improvement_target_disjoint"].mean()),
                    "combined_fisher_p_piecewise_source": float(
                        safe_fisher_p(group["p_improvement_source_disjoint"].to_numpy(dtype=float))
                    ),
                    "combined_fisher_p_piecewise_target": float(
                        safe_fisher_p(group["p_improvement_target_disjoint"].to_numpy(dtype=float))
                    ),
                    "p_shift_two_sided": float(
                        empirical_two_sided_p(observed_mean_shift, null_mean_shift)
                    )
                    if null_mean_shift.size
                    else float("nan"),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("domain")
    summary_path = ITER_DIR / "h35_linearity_breakpoint_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    summary = {
        "rows_tested": int(row_df.shape[0]),
        "domains_tested": int(summary_df.shape[0]),
        "domains_with_split_shift_sig": int((summary_df["p_shift_two_sided"] < 0.05).sum())
        if not summary_df.empty
        else 0,
        "domains_with_piecewise_sig_source": int(
            (summary_df["combined_fisher_p_piecewise_source"] < 0.05).sum()
        )
        if not summary_df.empty
        else 0,
        "domains_with_piecewise_sig_target": int(
            (summary_df["combined_fisher_p_piecewise_target"] < 0.05).sum()
        )
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_domain_split": str(row_path),
            "summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }
    return summary


def run_h36_anchor_spectral_alignment() -> dict[str, object]:
    geneformer_embeddings = load_geneformer_embeddings()
    trrust_source_degree = load_trrust_source_degree()

    cycle_baseline = {}
    if H36_CYCLE_BASELINE_PATH.exists():
        cycle_df = pd.read_csv(H36_CYCLE_BASELINE_PATH)
        for row in cycle_df.itertuples(index=False):
            cycle_baseline[str(row.domain)] = float(row.auc_cycle_consistent)

    domain_rows: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, domain in enumerate(["immune", "lung", "external_lung"]):
        gf_edge_df = pd.read_csv(GENEFORMER_EDGE_BY_DOMAIN[domain], sep="\t")
        sc_run_dir = SCGPT_RUNS_BY_DOMAIN[domain]["seed42_main"]
        sc_edge_df = pd.read_csv(sc_run_dir / "cycle1_edge_dataset.tsv", sep="\t")

        gf_symbol_to_token, gf_degree = build_symbol_maps_from_edge_table(
            gf_edge_df,
            source_cols=("source", "source_token_id"),
            target_cols=("target", "target_token_id"),
        )
        sc_symbol_to_idx, sc_degree = build_symbol_maps_from_edge_table(
            sc_edge_df,
            source_cols=("source", "source_idx"),
            target_cols=("target", "target_idx"),
        )

        shared_symbols = sorted(set(gf_symbol_to_token.keys()) & set(sc_symbol_to_idx.keys()))
        if len(shared_symbols) < 120:
            continue

        ranked_symbols = sorted(
            shared_symbols,
            key=lambda g: (
                -(gf_degree.get(g, 0) + sc_degree.get(g, 0)),
                g,
            ),
        )
        selected_symbols = ranked_symbols[: min(H36_GENE_CAP, len(ranked_symbols))]

        token_ids = np.array([gf_symbol_to_token[s] for s in selected_symbols], dtype=int)
        valid_mask = token_ids < geneformer_embeddings.shape[0]
        selected_symbols = [s for s, keep in zip(selected_symbols, valid_mask) if keep]
        token_ids = token_ids[valid_mask]
        if len(selected_symbols) < 100:
            continue

        gene_ids = np.array([sc_symbol_to_idx[s] for s in selected_symbols], dtype=int)
        n_symbols = len(selected_symbols)

        sc_layer = int(H36_LAYER_BY_DOMAIN[domain])
        sc_layers = np.load(sc_run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        if sc_layer >= sc_layers.shape[0]:
            continue
        x_raw = sc_layers[sc_layer, gene_ids, :].astype(np.float64)
        y_raw = geneformer_embeddings[token_ids, :].astype(np.float64)

        n_comp = min(40, n_symbols - 1, x_raw.shape[1], y_raw.shape[1])
        if n_comp < 8:
            continue

        x_pca = reduce_points(x_raw, n_components=n_comp, random_state=17_610 + domain_index * 10)
        y_pca = reduce_points(y_raw, n_components=n_comp, random_state=17_611 + domain_index * 10)

        gf_subset = gf_edge_df.copy()
        gf_subset["source_upper"] = gf_subset["source"].astype(str).str.upper()
        gf_subset["target_upper"] = gf_subset["target"].astype(str).str.upper()
        gf_subset = gf_subset.loc[
            gf_subset["source_upper"].isin(selected_symbols)
            & gf_subset["target_upper"].isin(selected_symbols)
        ].copy()
        if gf_subset["label"].nunique() < 2:
            continue

        sc_subset = sc_edge_df.copy()
        sc_subset["source_upper"] = sc_subset["source"].astype(str).str.upper()
        sc_subset["target_upper"] = sc_subset["target"].astype(str).str.upper()
        sc_subset = sc_subset.loc[
            sc_subset["source_upper"].isin(selected_symbols)
            & sc_subset["target_upper"].isin(selected_symbols)
        ].copy()

        sc_adj = build_positive_adjacency_from_symbol_edges(
            sc_subset,
            source_col="source_upper",
            target_col="target_upper",
            label_col="label",
            selected_symbols=selected_symbols,
        )
        gf_adj = build_positive_adjacency_from_symbol_edges(
            gf_subset,
            source_col="source_upper",
            target_col="target_upper",
            label_col="label",
            selected_symbols=selected_symbols,
        )

        spec_dim = min(16, n_symbols - 1)
        x_spec = normalized_laplacian_spectral_embedding(sc_adj, n_components=spec_dim)
        y_spec = normalized_laplacian_spectral_embedding(gf_adj, n_components=spec_dim)

        symbol_to_pos = {s: i for i, s in enumerate(selected_symbols)}
        src_pos = gf_subset["source_upper"].map(symbol_to_pos).to_numpy(dtype=int)
        tgt_pos = gf_subset["target_upper"].map(symbol_to_pos).to_numpy(dtype=int)
        labels = gf_subset["label"].to_numpy(dtype=int)

        source_threshold = float(gf_subset["source_idx"].median())
        target_threshold = float(gf_subset["target_idx"].median())
        source_mask = gf_subset["source_idx"].to_numpy(dtype=float) <= source_threshold
        target_mask = gf_subset["target_idx"].to_numpy(dtype=float) > target_threshold
        if np.unique(labels[source_mask]).size < 2 or np.unique(labels[target_mask]).size < 2:
            # Fallback split when index-based disjoint masks are degenerate.
            rng_fallback = np.random.default_rng(17_620 + domain_index)
            perm = rng_fallback.permutation(labels.size)
            half = labels.size // 2
            source_mask = np.zeros(labels.size, dtype=bool)
            source_mask[perm[:half]] = True
            target_mask = ~source_mask
            if np.unique(labels[source_mask]).size < 2 or np.unique(labels[target_mask]).size < 2:
                continue

        # Keep anchors sparse and biologically explicit: high-outdegree TRRUST TF sources only.
        anchor_candidates = [s for s in selected_symbols if s in trrust_source_degree]
        anchor_candidates = sorted(
            anchor_candidates,
            key=lambda s: (-trrust_source_degree[s], s),
        )
        desired_anchor_count = min(max(20, n_symbols // 6), 60)
        selected_anchors = set(anchor_candidates[: min(desired_anchor_count, len(anchor_candidates))])
        anchor_mask = np.array([s in selected_anchors for s in selected_symbols], dtype=bool)
        if np.sum(anchor_mask) < 10:
            continue
        anchor_weights = np.where(anchor_mask, 1.0 + H36_ANCHOR_BONUS, 1.0)

        # Baseline: CCA-like/procrustes equivalent at lambda=0 without anchors.
        x_base = row_normalize(x_pca)
        y_base = row_normalize(y_pca)
        q_base = orthogonal_procrustes_map(x_base, y_base)
        z_base = row_normalize(x_base @ q_base)
        auc_source_base = edge_auc_for_mask(z_base, src_pos, tgt_pos, labels, source_mask)
        auc_target_base = edge_auc_for_mask(z_base, src_pos, tgt_pos, labels, target_mask)

        best = {
            "lambda": float("nan"),
            "auc_source": float("-inf"),
            "auc_target": float("nan"),
            "z": None,
            "y": None,
            "x_mix": None,
            "y_mix": None,
        }

        for lam in H36_LAMBDA_GRID:
            x_mix = np.column_stack([(1.0 - lam) * x_pca, lam * x_spec])
            y_mix = np.column_stack([(1.0 - lam) * y_pca, lam * y_spec])
            x_mix = row_normalize(x_mix)
            y_mix = row_normalize(y_mix)

            q_anchor = weighted_orthogonal_procrustes(x_mix, y_mix, anchor_weights)
            z_anchor = row_normalize(x_mix @ q_anchor)

            auc_source = edge_auc_for_mask(z_anchor, src_pos, tgt_pos, labels, source_mask)
            auc_target = edge_auc_for_mask(z_anchor, src_pos, tgt_pos, labels, target_mask)
            if np.isfinite(auc_source) and auc_source > best["auc_source"]:
                best = {
                    "lambda": float(lam),
                    "auc_source": float(auc_source),
                    "auc_target": float(auc_target),
                    "z": z_anchor,
                    "y": y_mix,
                    "x_mix": x_mix,
                    "y_mix": y_mix,
                }

        if best["z"] is None:
            continue

        z_best = best["z"]
        y_best = best["y"]

        # Random-anchor null keeps lambda fixed to isolate anchor contribution.
        rng = np.random.default_rng(17_700 + domain_index)
        null_auc_target = np.empty(H36_NULL_PERM, dtype=float)
        null_delta_vs_base = np.empty(H36_NULL_PERM, dtype=float)
        null_auc_target_label_perm = np.empty(H36_NULL_PERM, dtype=float)
        anchor_count = int(np.sum(anchor_mask))
        score_target_anchor = np.sum(z_best[src_pos[target_mask]] * z_best[tgt_pos[target_mask]], axis=1)
        labels_target = labels[target_mask]
        for perm_idx in range(H36_NULL_PERM):
            perm_mask = np.zeros(n_symbols, dtype=bool)
            perm_idx_sel = rng.choice(n_symbols, size=anchor_count, replace=False)
            perm_mask[perm_idx_sel] = True
            perm_weights = np.where(perm_mask, 1.0 + H36_ANCHOR_BONUS, 1.0)

            q_perm = weighted_orthogonal_procrustes(
                best["x_mix"],
                best["y_mix"],
                perm_weights,
            )
            z_perm = row_normalize(best["x_mix"] @ q_perm)
            auc_perm_target = edge_auc_for_mask(z_perm, src_pos, tgt_pos, labels, target_mask)
            null_auc_target[perm_idx] = float(auc_perm_target)
            null_delta_vs_base[perm_idx] = float(auc_perm_target - auc_target_base)
            perm_labels_target = rng.permutation(labels_target)
            null_auc_target_label_perm[perm_idx] = safe_auc(perm_labels_target, score_target_anchor)
            null_rows.append(
                {
                    "domain": domain,
                    "perm_idx": int(perm_idx),
                    "null_auc_target": float(auc_perm_target),
                    "null_delta_vs_base": float(null_delta_vs_base[perm_idx]),
                    "null_auc_target_label_perm": float(null_auc_target_label_perm[perm_idx]),
                }
            )

        top1_anchor = float(np.mean(np.argmax(z_best @ y_best.T, axis=1) == np.arange(n_symbols)))
        top1_base = float(np.mean(np.argmax(z_base @ y_base.T, axis=1) == np.arange(n_symbols)))
        dist_rho_anchor = safe_spearman(pdist(z_best), pdist(y_best))
        dist_rho_base = safe_spearman(pdist(z_base), pdist(y_base))
        jacc_anchor = mean_knn_jaccard(z_best, y_best, n_neighbors=min(10, n_symbols - 1))
        jacc_base = mean_knn_jaccard(z_base, y_base, n_neighbors=min(10, n_symbols - 1))

        cycle_auc = cycle_baseline.get(domain, float("nan"))

        domain_rows.append(
            {
                "domain": domain,
                "n_symbols": int(n_symbols),
                "n_edges_eval": int(labels.size),
                "n_positive_eval": int(labels.sum()),
                "anchor_count": int(anchor_count),
                "selected_lambda": float(best["lambda"]),
                "auc_source_anchor": float(best["auc_source"]),
                "auc_target_anchor": float(best["auc_target"]),
                "auc_source_baseline": float(auc_source_base),
                "auc_target_baseline": float(auc_target_base),
                "delta_auc_target_anchor_minus_baseline": float(best["auc_target"] - auc_target_base),
                "auc_target_cycle_baseline_iter0016": float(cycle_auc),
                "delta_auc_target_anchor_minus_cycle": float(best["auc_target"] - cycle_auc)
                if np.isfinite(cycle_auc)
                else float("nan"),
                "null_mean_auc_target": float(np.nanmean(null_auc_target)),
                "p_auc_target_anchor_upper_vs_random_anchor": float(
                    empirical_upper_tail_p(float(best["auc_target"]), null_auc_target)
                ),
                "p_delta_target_upper_vs_random_anchor": float(
                    empirical_upper_tail_p(float(best["auc_target"] - auc_target_base), null_delta_vs_base)
                ),
                "null_mean_auc_target_label_perm": float(np.nanmean(null_auc_target_label_perm)),
                "p_auc_target_anchor_upper_vs_label_perm": float(
                    empirical_upper_tail_p(float(best["auc_target"]), null_auc_target_label_perm)
                ),
            }
        )

        map_rows.append(
            {
                "domain": domain,
                "map_variant": "baseline_unanchored_lambda0",
                "top1_retrieval": float(top1_base),
                "distance_spearman": float(dist_rho_base),
                "knn_jaccard": float(jacc_base),
            }
        )
        map_rows.append(
            {
                "domain": domain,
                "map_variant": "anchor_regularized_best_lambda",
                "top1_retrieval": float(top1_anchor),
                "distance_spearman": float(dist_rho_anchor),
                "knn_jaccard": float(jacc_anchor),
            }
        )

    domain_df = pd.DataFrame(domain_rows).sort_values("domain")
    domain_path = ITER_DIR / "h36_anchor_spectral_alignment_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    map_df = pd.DataFrame(map_rows).sort_values(["domain", "map_variant"])
    map_path = ITER_DIR / "h36_anchor_spectral_alignment_map_quality.csv"
    map_df.to_csv(map_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(["domain", "perm_idx"])
    null_path = ITER_DIR / "h36_anchor_spectral_alignment_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary = {
        "domains_tested": int(domain_df.shape[0]),
        "mean_auc_target_anchor": float(domain_df["auc_target_anchor"].mean())
        if not domain_df.empty
        else float("nan"),
        "mean_auc_target_baseline": float(domain_df["auc_target_baseline"].mean())
        if not domain_df.empty
        else float("nan"),
        "mean_delta_target_anchor_minus_baseline": float(
            domain_df["delta_auc_target_anchor_minus_baseline"].mean()
        )
        if not domain_df.empty
        else float("nan"),
        "domains_with_positive_delta_vs_baseline": int(
            (domain_df["delta_auc_target_anchor_minus_baseline"] > 0).sum()
        )
        if not domain_df.empty
        else 0,
        "domains_sig_vs_random_anchor": int(
            (domain_df["p_auc_target_anchor_upper_vs_random_anchor"] < 0.05).sum()
        )
        if not domain_df.empty
        else 0,
        "domains_sig_vs_label_perm": int(
            (domain_df["p_auc_target_anchor_upper_vs_label_perm"] < 0.05).sum()
        )
        if not domain_df.empty
        else 0,
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
    required_paths.append(TRRUST_PATH)

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    coexp_cache: dict[str, CoexpressionDomainCache] = {
        domain: CoexpressionDomainCache(path)
        for domain, path in PROCESSED_H5AD_BY_DOMAIN.items()
    }
    try:
        h34_summary = run_h34_convexity_detour_multiseed(coexp_cache=coexp_cache)
    finally:
        for cache in coexp_cache.values():
            cache.close()

    h35_summary = run_h35_linearity_breakpoint_screen()
    h36_summary = run_h36_anchor_spectral_alignment()

    iteration_summary = {
        "iteration": "iter_0017",
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
        },
        "h34_convexity_detour_multiseed_incremental": h34_summary,
        "h35_local_linearity_breakpoint_screen": h35_summary,
        "h36_anchor_spectral_alignment": h36_summary,
    }

    summary_path = ITER_DIR / "iter0017_screen_summary.json"
    summary_path.write_text(json.dumps(iteration_summary, indent=2))
    print(json.dumps(iteration_summary, indent=2))


if __name__ == "__main__":
    main()
