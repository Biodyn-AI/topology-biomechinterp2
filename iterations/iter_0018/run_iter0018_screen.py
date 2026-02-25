from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from ripser import ripser
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import cdist
from scipy.stats import combine_pvalues, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


ITER_DIR = Path("iterations/iter_0018")
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
DOROTHEA_PATH = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
    "single_cell_mechinterp/external/networks/dorothea_human.tsv"
)
GENE2GO_PATH = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
    "single_cell_mechinterp/data/perturb/gene2go_all.pkl"
)

H37_LAYERS = [0, 3, 7, 11]
H37_GENE_CAP = 240
H37_DIFFUSION_TIMES = [1, 2, 4, 8]
H37_NULL_PERM = 120
H37_MAX_COEXP_CELLS = 4000

H38_GENE_CAP = 240
H38_NEIGHBORS = 12
H38_NULL_PERM = 300

H39_NULL_PERM = 20


@dataclass
class CoexpressionDomainCache:
    """Open one backed h5ad per domain and reuse it across many hypothesis rows."""

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
    if x.size < 3 or y.size < 3:
        return float("nan")
    rho = spearmanr(x, y).correlation
    return float(rho) if np.isfinite(rho) else float("nan")


def empirical_upper_tail_p(observed: float, null_values: np.ndarray) -> float:
    if not np.isfinite(observed):
        return float("nan")
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float((1 + np.sum(values >= observed)) / (values.size + 1))


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

    # Bridge disconnected kNN components so geodesic distance is always defined.
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
    p_t = np.linalg.matrix_power(transition, int(t))
    src_rows = p_t[source_local]
    tgt_rows = p_t[target_local]
    diff = src_rows - tgt_rows
    return np.sqrt(np.sum(diff * diff, axis=1))


def degree_strata(values: np.ndarray, max_bins: int = 5) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return np.array([], dtype=int)
    q = min(max_bins, max(1, int(np.sqrt(x.size) // 2)))
    if q <= 1:
        return np.zeros(x.size, dtype=int)
    ranks = pd.Series(x).rank(method="average", pct=True).to_numpy(dtype=float)
    bins = np.minimum((ranks * q).astype(int), q - 1)
    return bins


def combine_strata(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        return np.array([], dtype=int)
    out = np.zeros_like(np.asarray(arrays[0], dtype=int))
    factor = 1
    for arr in arrays:
        arr_int = np.asarray(arr, dtype=int)
        out += factor * arr_int
        factor *= int(np.max(arr_int) + 1 if arr_int.size else 1)
    return out


def stratified_shuffle(values: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(values)
    strata = np.asarray(strata, dtype=int)
    out = x.copy()
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        if idx.size > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def standardize_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, ddof=1, keepdims=True)
    sd = np.where(sd <= 1e-10, 1.0, sd)
    return (x - mu) / sd


def fit_logistic_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    x = standardize_columns(features)
    y = np.asarray(labels, dtype=int)
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1200,
        class_weight="balanced",
        random_state=17_001,
    )
    clf.fit(x, y)
    return clf.predict_proba(x)[:, 1]


def get_knn_indices(points: np.ndarray, n_neighbors: int) -> np.ndarray:
    k = max(2, min(n_neighbors, points.shape[0] - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(points)
    _, indices = nbrs.kneighbors(points)
    return indices[:, 1:]


def compute_local_reconstruction_errors(points: np.ndarray, neighbor_idx: np.ndarray) -> np.ndarray:
    n, d = points.shape
    errors = np.empty(n, dtype=float)
    for i in range(n):
        neigh = neighbor_idx[i]
        y = points[i]
        x = points[neigh]
        x_center = x - x.mean(axis=0, keepdims=True)
        y_center = y - x.mean(axis=0)
        cov = x_center @ x_center.T
        cov.flat[:: cov.shape[0] + 1] += 1e-3
        w = np.linalg.solve(cov, x_center @ y_center)
        w_sum = np.sum(w)
        if abs(w_sum) > 1e-8:
            w = w / w_sum
        recon = w @ x
        errors[i] = float(np.mean((recon - y) ** 2))
    return errors


def fit_linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim == 1:
        x = x[:, None]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[0] != y.size or y.size < 3:
        return float("nan")
    design = np.column_stack([np.ones(y.size), x])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def sample_skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return float("nan")
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd <= 1e-12:
        return 0.0
    centered = (x - mu) / sd
    return float(np.mean(centered ** 3))


def compute_twonn_local_id(points: np.ndarray) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=3, metric="euclidean")
    nbrs.fit(points)
    dists, _ = nbrs.kneighbors(points)
    r1 = np.clip(dists[:, 1], 1e-8, None)
    r2 = np.clip(dists[:, 2], r1 + 1e-8, None)
    ratio = np.clip(r2 / r1, 1.0 + 1e-6, None)
    ids = 1.0 / np.log(ratio)
    return np.clip(ids, 0.0, 100.0)


def compute_local_participation_ratio(points: np.ndarray, neighbor_idx: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    pr = np.empty(n, dtype=float)
    for i in range(n):
        idx = neighbor_idx[i]
        neigh = points[idx]
        centered = neigh - neigh.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(1, neigh.shape[0] - 1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.clip(eigvals, 0.0, None)
        denom = float(np.sum(eigvals ** 2))
        if denom <= 1e-12:
            pr[i] = 0.0
        else:
            s1 = float(np.sum(eigvals))
            pr[i] = (s1 * s1) / denom
    return pr


def load_support_priors() -> tuple[set[tuple[str, str]], set[tuple[str, str]], dict[str, set[str]]]:
    trrust = pd.read_csv(
        TRRUST_PATH,
        sep="\t",
        header=None,
        names=["source", "target", "regulation", "pmid"],
    )
    trrust_pairs = {
        (str(row.source).upper(), str(row.target).upper())
        for row in trrust[["source", "target"]].itertuples(index=False)
    }

    dorothea = pd.read_csv(DOROTHEA_PATH, sep="\t")
    if "source" in dorothea.columns and "target" in dorothea.columns:
        src_col, tgt_col = "source", "target"
    else:
        src_col, tgt_col = dorothea.columns[0], dorothea.columns[1]
    dorothea_pairs = {
        (str(getattr(row, src_col)).upper(), str(getattr(row, tgt_col)).upper())
        for row in dorothea[[src_col, tgt_col]].itertuples(index=False)
    }

    with GENE2GO_PATH.open("rb") as f:
        gene2go_raw = pickle.load(f)
    gene2go = {
        str(g).upper(): {str(term) for term in terms}
        for g, terms in gene2go_raw.items()
        if terms
    }

    return trrust_pairs, dorothea_pairs, gene2go


def build_consensus_tier(
    source_symbols: np.ndarray,
    target_symbols: np.ndarray,
    trrust_pairs: set[tuple[str, str]],
    dorothea_pairs: set[tuple[str, str]],
    gene2go: dict[str, set[str]],
) -> np.ndarray:
    tiers = np.zeros(source_symbols.size, dtype=int)
    for i, (s, t) in enumerate(zip(source_symbols, target_symbols)):
        s_up = str(s).upper()
        t_up = str(t).upper()
        support = 0
        if (s_up, t_up) in trrust_pairs:
            support += 1
        if (s_up, t_up) in dorothea_pairs:
            support += 1
        go_s = gene2go.get(s_up)
        go_t = gene2go.get(t_up)
        if go_s and go_t and len(go_s & go_t) > 0:
            support += 1
        tiers[i] = support
    return tiers


def persistence_h1_metrics(points: np.ndarray) -> tuple[float, float, float, int]:
    dgms = ripser(points, maxdim=1)["dgms"]
    if len(dgms) < 2:
        return 0.0, 0.0, 0.0, 0
    h1 = dgms[1]
    if h1.size == 0:
        return 0.0, 0.0, 0.0, 0
    birth = h1[:, 0]
    death = h1[:, 1]
    finite = np.isfinite(death)
    if not np.any(finite):
        return 0.0, 0.0, 0.0, 0
    life = np.clip(death[finite] - birth[finite], 0.0, None)
    if life.size == 0:
        return 0.0, 0.0, 0.0, 0
    total = float(np.sum(life))
    mean = float(np.mean(life))
    probs = life / np.clip(total, 1e-12, None)
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
    return total, mean, entropy, int(life.size)


def run_h37_h39_screen(
    coexp_cache: dict[str, CoexpressionDomainCache],
    trrust_pairs: set[tuple[str, str]],
    dorothea_pairs: set[tuple[str, str]],
    gene2go: dict[str, set[str]],
) -> dict[str, object]:
    h37_rows: list[dict[str, object]] = []
    h37_null_rows: list[dict[str, object]] = []

    h39_rows: list[dict[str, object]] = []
    h39_null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(SCGPT_RUNS_BY_DOMAIN.items()):
        seed_tag = "seed42_main"
        run_dir = run_map[seed_tag]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = select_top_genes(split_edges, gene_cap=H37_GENE_CAP)
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

            source_symbols = split_edges["source"].astype(str).str.upper().to_numpy()
            target_symbols = split_edges["target"].astype(str).str.upper().to_numpy()
            tiers = build_consensus_tier(
                source_symbols,
                target_symbols,
                trrust_pairs=trrust_pairs,
                dorothea_pairs=dorothea_pairs,
                gene2go=gene2go,
            )

            gene_name_map: dict[int, str] = {}
            for row in split_edges[["source_idx", "source"]].drop_duplicates().itertuples(index=False):
                gene_name_map[int(row.source_idx)] = str(row.source).upper()
            for row in split_edges[["target_idx", "target"]].drop_duplicates().itertuples(index=False):
                gene_name_map[int(row.target_idx)] = str(row.target).upper()
            ordered_gene_symbols = [gene_name_map[int(g)] for g in edge_gene_indices]

            coexp_abs_corr, coexp_missing_fraction = coexp_cache[domain].abs_corr_for_genes(
                ordered_gene_symbols,
                max_cells=H37_MAX_COEXP_CELLS,
                random_state=18_100 + domain_index * 100 + split_index,
            )
            coexp_scores = coexp_abs_corr[source_local, target_local]

            source_degree_map = split_edges["source_idx"].value_counts().to_dict()
            target_degree_map = split_edges["target_idx"].value_counts().to_dict()
            source_degree = split_edges["source_idx"].map(source_degree_map).astype(float).to_numpy()
            target_degree = split_edges["target_idx"].map(target_degree_map).astype(float).to_numpy()
            mean_degree = 0.5 * (source_degree + target_degree)

            for layer in H37_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue
                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = reduce_points(
                    points,
                    n_components=24,
                    random_state=18_200 + domain_index * 1000 + split_index * 100 + layer,
                )

                geodesic, transition, k_eff, used_component_bridging = geodesic_and_transition(
                    points_pca,
                    n_neighbors=24,
                )
                euclidean = cdist(points_pca, points_pca, metric="euclidean")
                geodesic_distance = geodesic[source_local, target_local]
                euclidean_distance = np.clip(euclidean[source_local, target_local], 1e-6, None)

                diffusion_cols: list[np.ndarray] = []
                for t in H37_DIFFUSION_TIMES:
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
                extended_features = np.column_stack([baseline_features, detour_ratio, convexity_deficit])
                score_base = fit_logistic_scores(baseline_features, labels)
                score_ext = fit_logistic_scores(extended_features, labels)

                overall_delta = safe_auc(labels, score_ext) - safe_auc(labels, score_base)

                # Tier-0 is too sparse in this dataset; use <=1 as the low-support bucket.
                low_mask = tiers <= 1
                high_mask = tiers >= 2
                auc_base_low = safe_auc(labels[low_mask], score_base[low_mask])
                auc_ext_low = safe_auc(labels[low_mask], score_ext[low_mask])
                auc_base_high = safe_auc(labels[high_mask], score_base[high_mask])
                auc_ext_high = safe_auc(labels[high_mask], score_ext[high_mask])
                delta_low = auc_ext_low - auc_base_low if np.isfinite(auc_base_low) and np.isfinite(auc_ext_low) else float("nan")
                delta_high = (
                    auc_ext_high - auc_base_high
                    if np.isfinite(auc_base_high) and np.isfinite(auc_ext_high)
                    else float("nan")
                )
                tier_gap = delta_high - delta_low if np.isfinite(delta_low) and np.isfinite(delta_high) else float("nan")

                strata = combine_strata(
                    degree_strata(mean_degree, max_bins=5),
                    degree_strata(coexp_scores, max_bins=5),
                    degree_strata(geodesic_distance, max_bins=5),
                )
                rng = np.random.default_rng(
                    18_300 + domain_index * 1000 + split_index * 100 + layer
                )
                null_gap = np.empty(H37_NULL_PERM, dtype=float)
                for perm_idx in range(H37_NULL_PERM):
                    tiers_perm = stratified_shuffle(tiers, strata, rng=rng)
                    low_perm = tiers_perm <= 1
                    high_perm = tiers_perm >= 2
                    auc_base_low_p = safe_auc(labels[low_perm], score_base[low_perm])
                    auc_ext_low_p = safe_auc(labels[low_perm], score_ext[low_perm])
                    auc_base_high_p = safe_auc(labels[high_perm], score_base[high_perm])
                    auc_ext_high_p = safe_auc(labels[high_perm], score_ext[high_perm])
                    if not (
                        np.isfinite(auc_base_low_p)
                        and np.isfinite(auc_ext_low_p)
                        and np.isfinite(auc_base_high_p)
                        and np.isfinite(auc_ext_high_p)
                    ):
                        null_gap[perm_idx] = np.nan
                    else:
                        null_gap[perm_idx] = (auc_ext_high_p - auc_base_high_p) - (
                            auc_ext_low_p - auc_base_low_p
                        )
                    h37_null_rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_tier_gap_high_minus_low": float(null_gap[perm_idx]),
                        }
                    )

                h37_rows.append(
                    {
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges": int(labels.size),
                        "n_positive": int(labels.sum()),
                        "n_unique_genes": int(edge_gene_indices.size),
                        "knn_k": int(k_eff),
                        "used_component_bridging": bool(used_component_bridging),
                        "coexp_missing_fraction": float(coexp_missing_fraction),
                        "tier0_fraction": float(np.mean(tiers == 0)),
                        "tier01_fraction": float(np.mean(tiers <= 1)),
                        "tier1_fraction": float(np.mean(tiers == 1)),
                        "tier2plus_fraction": float(np.mean(tiers >= 2)),
                        "overall_delta_auc_geometry": float(overall_delta),
                        "delta_auc_tier01": float(delta_low),
                        "delta_auc_tier2plus": float(delta_high),
                        "tier_gap_high_minus_low": float(tier_gap),
                        "p_tier_gap_upper": float(empirical_upper_tail_p(tier_gap, null_gap)),
                    }
                )

                h1_total, h1_mean, h1_entropy, h1_count = persistence_h1_metrics(points_pca)
                null_totals = np.empty(H39_NULL_PERM, dtype=float)
                rng_ph = np.random.default_rng(
                    18_400 + domain_index * 1000 + split_index * 100 + layer
                )
                for perm_idx in range(H39_NULL_PERM):
                    shuffled = points_pca.copy()
                    for col in range(shuffled.shape[1]):
                        shuffled[:, col] = rng_ph.permutation(shuffled[:, col])
                    n_total, _, _, _ = persistence_h1_metrics(shuffled)
                    null_totals[perm_idx] = float(n_total)
                    h39_null_rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_h1_total_lifetime": float(n_total),
                        }
                    )

                null_mean = float(np.nanmean(null_totals))
                null_std = float(np.nanstd(null_totals, ddof=1)) if np.sum(np.isfinite(null_totals)) > 1 else float("nan")
                h1_z = (
                    (h1_total - null_mean) / null_std
                    if np.isfinite(null_std) and null_std > 1e-12
                    else float("nan")
                )

                h39_rows.append(
                    {
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_points": int(points_pca.shape[0]),
                        "h1_count": int(h1_count),
                        "h1_total_lifetime": float(h1_total),
                        "h1_mean_lifetime": float(h1_mean),
                        "h1_entropy": float(h1_entropy),
                        "null_mean_h1_total_lifetime": float(null_mean),
                        "null_std_h1_total_lifetime": float(null_std),
                        "h1_total_lifetime_z": float(h1_z),
                        "p_h1_total_upper_vs_shuffle": float(empirical_upper_tail_p(h1_total, null_totals)),
                        "geometry_delta_auc": float(overall_delta),
                    }
                )

    h37_df = pd.DataFrame(h37_rows).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    h37_path = ITER_DIR / "h37_consensus_tier_geometry_by_seed_layer_split.csv"
    h37_df.to_csv(h37_path, index=False)

    h37_null_df = pd.DataFrame(h37_null_rows).sort_values(
        ["domain", "seed_tag", "split_regime", "layer", "perm_idx"]
    )
    h37_null_path = ITER_DIR / "h37_consensus_tier_geometry_null_summary.csv"
    h37_null_df.to_csv(h37_null_path, index=False)

    h37_domain_rows: list[dict[str, object]] = []
    for (domain, split_regime), group in h37_df.groupby(["domain", "split_regime"], sort=True):
        h37_domain_rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(group.shape[0]),
                "mean_tier_gap_high_minus_low": float(group["tier_gap_high_minus_low"].mean()),
                "median_tier_gap_high_minus_low": float(group["tier_gap_high_minus_low"].median()),
                "fraction_tier_gap_positive": float((group["tier_gap_high_minus_low"] > 0.0).mean()),
                "fraction_p_tier_gap_lt_0_05": float((group["p_tier_gap_upper"] < 0.05).mean()),
                "combined_fisher_p_tier_gap": float(
                    safe_fisher_p(group["p_tier_gap_upper"].to_numpy(dtype=float))
                ),
                "mean_overall_delta_auc_geometry": float(group["overall_delta_auc_geometry"].mean()),
            }
        )
    h37_domain_df = pd.DataFrame(h37_domain_rows).sort_values(["domain", "split_regime"])
    h37_domain_path = ITER_DIR / "h37_consensus_tier_geometry_domain_summary.csv"
    h37_domain_df.to_csv(h37_domain_path, index=False)

    h39_df = pd.DataFrame(h39_rows).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    h39_path = ITER_DIR / "h39_ph_feature_shuffle_by_seed_layer_split.csv"
    h39_df.to_csv(h39_path, index=False)

    h39_null_df = pd.DataFrame(h39_null_rows).sort_values(
        ["domain", "seed_tag", "split_regime", "layer", "perm_idx"]
    )
    h39_null_path = ITER_DIR / "h39_ph_feature_shuffle_null_summary.csv"
    h39_null_df.to_csv(h39_null_path, index=False)

    h39_domain_rows: list[dict[str, object]] = []
    for (domain, split_regime), group in h39_df.groupby(["domain", "split_regime"], sort=True):
        h39_domain_rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(group.shape[0]),
                "mean_h1_z": float(group["h1_total_lifetime_z"].mean()),
                "fraction_h1_p_lt_0_05": float((group["p_h1_total_upper_vs_shuffle"] < 0.05).mean()),
                "mean_geometry_delta_auc": float(group["geometry_delta_auc"].mean()),
                "rho_h1z_vs_geometry_delta": float(
                    safe_spearman(
                        group["h1_total_lifetime_z"].to_numpy(dtype=float),
                        group["geometry_delta_auc"].to_numpy(dtype=float),
                    )
                ),
                "combined_fisher_p_h1": float(
                    safe_fisher_p(group["p_h1_total_upper_vs_shuffle"].to_numpy(dtype=float))
                ),
            }
        )
    h39_domain_df = pd.DataFrame(h39_domain_rows).sort_values(["domain", "split_regime"])
    h39_domain_path = ITER_DIR / "h39_ph_feature_shuffle_domain_summary.csv"
    h39_domain_df.to_csv(h39_domain_path, index=False)

    summary = {
        "h37": {
            "rows_tested": int(h37_df.shape[0]),
            "mean_tier_gap": float(h37_df["tier_gap_high_minus_low"].mean()) if not h37_df.empty else float("nan"),
            "domain_split_positive_gap": int((h37_domain_df["mean_tier_gap_high_minus_low"] > 0).sum()) if not h37_domain_df.empty else 0,
            "domain_split_fisher_sig": int((h37_domain_df["combined_fisher_p_tier_gap"] < 0.05).sum()) if not h37_domain_df.empty else 0,
            "artifact_paths": {
                "by_seed_layer_split": str(h37_path),
                "domain_summary": str(h37_domain_path),
                "null_summary": str(h37_null_path),
            },
        },
        "h39": {
            "rows_tested": int(h39_df.shape[0]),
            "mean_h1_z": float(h39_df["h1_total_lifetime_z"].mean()) if not h39_df.empty else float("nan"),
            "domain_split_mean_h1_positive": int((h39_domain_df["mean_h1_z"] > 0).sum()) if not h39_domain_df.empty else 0,
            "domain_split_fisher_sig_h1": int((h39_domain_df["combined_fisher_p_h1"] < 0.05).sum()) if not h39_domain_df.empty else 0,
            "artifact_paths": {
                "by_seed_layer_split": str(h39_path),
                "domain_summary": str(h39_domain_path),
                "null_summary": str(h39_null_path),
            },
        },
    }
    return summary


def run_h38_id_distribution_screen() -> dict[str, object]:
    layer_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
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

                top_genes = select_top_genes(split_edges, gene_cap=H38_GENE_CAP)
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

                one_series_rows: list[dict[str, object]] = []
                for layer in range(layer_embeddings.shape[0]):
                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = reduce_points(
                        points,
                        n_components=24,
                        random_state=18_500
                        + domain_index * 10_000
                        + seed_index * 1_000
                        + split_index * 100
                        + layer,
                    )

                    neigh_idx = get_knn_indices(points_pca, n_neighbors=H38_NEIGHBORS)
                    recon_errors = compute_local_reconstruction_errors(points_pca, neigh_idx)
                    edge_recon_mean = 0.5 * (recon_errors[source_local] + recon_errors[target_local])
                    edge_auc = safe_auc(labels, -edge_recon_mean)

                    twonn = compute_twonn_local_id(points_pca)
                    pr_local = compute_local_participation_ratio(points_pca, neigh_idx)

                    row = {
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges": int(labels.size),
                        "n_positive": int(labels.sum()),
                        "n_points": int(points_pca.shape[0]),
                        "edge_auc_local_linearity": float(edge_auc),
                        "twonn_mean": float(np.mean(twonn)),
                        "twonn_var": float(np.var(twonn, ddof=1)),
                        "twonn_skew": float(sample_skew(twonn)),
                        "pr_mean": float(np.mean(pr_local)),
                        "pr_var": float(np.var(pr_local, ddof=1)),
                        "pr_skew": float(sample_skew(pr_local)),
                    }
                    layer_rows.append(row)
                    one_series_rows.append(row)

                series_df = pd.DataFrame(one_series_rows).sort_values("layer")
                if series_df.shape[0] < 8:
                    continue

                y = series_df["edge_auc_local_linearity"].to_numpy(dtype=float)
                x_layer = series_df[["layer"]].to_numpy(dtype=float)
                x_mean = np.column_stack(
                    [
                        x_layer,
                        series_df[["twonn_mean", "pr_mean"]].to_numpy(dtype=float),
                    ]
                )
                x_full = np.column_stack(
                    [
                        x_layer,
                        series_df[
                            [
                                "twonn_mean",
                                "pr_mean",
                                "twonn_var",
                                "twonn_skew",
                                "pr_var",
                                "pr_skew",
                            ]
                        ].to_numpy(dtype=float),
                    ]
                )

                r2_mean = fit_linear_r2(x_mean, y)
                r2_full = fit_linear_r2(x_full, y)
                delta_r2 = r2_full - r2_mean

                rng = np.random.default_rng(
                    18_600 + domain_index * 10_000 + seed_index * 1_000 + split_index * 100
                )
                null_delta = np.empty(H38_NULL_PERM, dtype=float)
                for perm_idx in range(H38_NULL_PERM):
                    y_perm = rng.permutation(y)
                    delta_p = fit_linear_r2(x_full, y_perm) - fit_linear_r2(x_mean, y_perm)
                    null_delta[perm_idx] = float(delta_p)
                    null_rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "perm_idx": int(perm_idx),
                            "null_delta_r2_full_minus_mean": float(delta_p),
                        }
                    )

                fit_rows.append(
                    {
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "n_layers": int(series_df.shape[0]),
                        "r2_mean_model": float(r2_mean),
                        "r2_full_model": float(r2_full),
                        "delta_r2_full_minus_mean": float(delta_r2),
                        "p_delta_r2_upper": float(empirical_upper_tail_p(delta_r2, null_delta)),
                    }
                )

    layer_df = pd.DataFrame(layer_rows).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    layer_path = ITER_DIR / "h38_id_distribution_moments_by_seed_layer_split.csv"
    layer_df.to_csv(layer_path, index=False)

    fit_df = pd.DataFrame(fit_rows).sort_values(["domain", "seed_tag", "split_regime"])
    fit_path = ITER_DIR / "h38_id_distribution_moments_fit_by_seed_split.csv"
    fit_df.to_csv(fit_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(["domain", "seed_tag", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h38_id_distribution_moments_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    domain_rows: list[dict[str, object]] = []
    for (domain, split_regime), group in fit_df.groupby(["domain", "split_regime"], sort=True):
        domain_rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_seeds": int(group.shape[0]),
                "mean_r2_mean_model": float(group["r2_mean_model"].mean()),
                "mean_r2_full_model": float(group["r2_full_model"].mean()),
                "mean_delta_r2_full_minus_mean": float(group["delta_r2_full_minus_mean"].mean()),
                "fraction_delta_r2_positive": float((group["delta_r2_full_minus_mean"] > 0.0).mean()),
                "fraction_p_delta_lt_0_05": float((group["p_delta_r2_upper"] < 0.05).mean()),
                "combined_fisher_p_delta_r2": float(
                    safe_fisher_p(group["p_delta_r2_upper"].to_numpy(dtype=float))
                ),
            }
        )
    domain_df = pd.DataFrame(domain_rows).sort_values(["domain", "split_regime"])
    domain_path = ITER_DIR / "h38_id_distribution_moments_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    summary = {
        "rows_tested": int(fit_df.shape[0]),
        "mean_delta_r2": float(fit_df["delta_r2_full_minus_mean"].mean()) if not fit_df.empty else float("nan"),
        "domain_split_positive_delta": int((domain_df["mean_delta_r2_full_minus_mean"] > 0).sum()) if not domain_df.empty else 0,
        "domain_split_fisher_sig": int((domain_df["combined_fisher_p_delta_r2"] < 0.05).sum()) if not domain_df.empty else 0,
        "artifact_paths": {
            "by_seed_layer_split": str(layer_path),
            "fit_by_seed_split": str(fit_path),
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
        },
    }
    return summary


def main() -> None:
    required_paths = [
        TRRUST_PATH,
        DOROTHEA_PATH,
        GENE2GO_PATH,
    ]
    for domain, run_map in SCGPT_RUNS_BY_DOMAIN.items():
        required_paths.append(PROCESSED_H5AD_BY_DOMAIN[domain])
        for run_dir in run_map.values():
            required_paths.append(run_dir / "cycle1_edge_dataset.tsv")
            required_paths.append(run_dir / "layer_gene_embeddings.npy")

    missing = [str(p) for p in required_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    trrust_pairs, dorothea_pairs, gene2go = load_support_priors()

    coexp_cache = {
        domain: CoexpressionDomainCache(path)
        for domain, path in PROCESSED_H5AD_BY_DOMAIN.items()
    }
    try:
        h37_h39_summary = run_h37_h39_screen(
            coexp_cache=coexp_cache,
            trrust_pairs=trrust_pairs,
            dorothea_pairs=dorothea_pairs,
            gene2go=gene2go,
        )
    finally:
        for cache in coexp_cache.values():
            cache.close()

    h38_summary = run_h38_id_distribution_screen()

    summary = {
        "iteration": "iter_0018",
        "h37": h37_h39_summary["h37"],
        "h38": h38_summary,
        "h39": h37_h39_summary["h39"],
    }

    summary_path = ITER_DIR / "iter0018_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
