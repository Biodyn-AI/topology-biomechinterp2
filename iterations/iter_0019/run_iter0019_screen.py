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
from scipy.stats import combine_pvalues
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


ITER_DIR = Path("iterations/iter_0019")
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
OMNIPATH_INTERACTIONS_PATH = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
    "network_inference/data/omnipath_interactions.tsv"
)

H40_LAYERS = [0, 3, 7, 11]
H40_GENE_CAP = 220
H40_DIFFUSION_T = 2
H40_NULL_PERM = 60
H40_MAX_COEXP_CELLS = 3000

H41_LAYERS = [0, 3, 7, 11]
H41_SEED_TAG = "seed42_main"
H41_GENE_CAP = 170
H41_LOCAL_NEIGHBORS = 12

H42_NULL_PERM = 320


@dataclass
class CoexpressionDomainCache:
    """
    Keep one backed h5ad object open per domain to avoid repeated full reads.
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


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    if np.unique(scores).size < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


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


def fit_logistic_scores_and_coef(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = standardize_columns(features)
    y = np.asarray(labels, dtype=int)
    if x.shape[0] == 0 or np.unique(y).size < 2:
        return np.full(x.shape[0], 0.5, dtype=float), np.full(x.shape[1], np.nan, dtype=float)
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1200,
        class_weight="balanced",
        random_state=random_state,
    )
    try:
        clf.fit(x, y)
    except Exception:
        return np.full(x.shape[0], 0.5, dtype=float), np.full(x.shape[1], np.nan, dtype=float)
    score = clf.predict_proba(x)[:, 1]
    coef = clf.coef_[0].astype(float, copy=False)
    return score, coef


def quantile_mask(values: np.ndarray, q_low: float, q_high: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    if arr.size < 10:
        return np.zeros(arr.size, dtype=bool), np.zeros(arr.size, dtype=bool)
    low_thr = float(np.quantile(arr, q_low))
    high_thr = float(np.quantile(arr, q_high))
    low_mask = arr <= low_thr
    high_mask = arr >= high_thr
    return low_mask, high_mask


def compute_convexity_deficit(
    geodesic: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
) -> np.ndarray:
    n_nodes = geodesic.shape[0]
    sorted_geo = np.sort(geodesic, axis=1)
    k_local = min(8, max(1, n_nodes - 1))
    local_radius = sorted_geo[:, k_local]
    neighborhood_mask = geodesic <= local_radius[:, None]
    deficit = np.empty(source_local.size, dtype=float)
    for i in range(source_local.size):
        src = int(source_local[i])
        tgt = int(target_local[i])
        src_mask = neighborhood_mask[src]
        tgt_mask = neighborhood_mask[tgt]
        inter = int(np.sum(src_mask & tgt_mask))
        union = int(np.sum(src_mask | tgt_mask))
        jacc = (inter / union) if union > 0 else 0.0
        deficit[i] = 1.0 - jacc
    return deficit


def persistence_h1_total(points: np.ndarray) -> float:
    if points.shape[0] < 4:
        return 0.0
    dgms = ripser(points, maxdim=1)["dgms"]
    if len(dgms) < 2 or dgms[1].size == 0:
        return 0.0
    h1 = dgms[1]
    birth = h1[:, 0]
    death = h1[:, 1]
    finite = np.isfinite(death)
    if not np.any(finite):
        return 0.0
    life = np.clip(death[finite] - birth[finite], 0.0, None)
    if life.size == 0:
        return 0.0
    return float(np.sum(life))


def compute_local_h1_scores(points: np.ndarray, n_neighbors: int) -> np.ndarray:
    n = points.shape[0]
    if n < 6:
        return np.zeros(n, dtype=float)
    k = max(4, min(n_neighbors, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(points)
    _, indices = nbrs.kneighbors(points)
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        local = points[indices[i]]
        scores[i] = persistence_h1_total(local)
    return scores


def fit_linear_r2(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float:
    if x_train.ndim == 1:
        x_train = x_train[:, None]
    if x_test.ndim == 1:
        x_test = x_test[:, None]
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    if x_train.shape[0] < 3 or x_test.shape[0] < 2:
        return float("nan")
    design_train = np.column_stack([np.ones(y_train.size), x_train])
    coef, _, _, _ = np.linalg.lstsq(design_train, y_train, rcond=None)
    design_test = np.column_stack([np.ones(y_test.size), x_test])
    pred = design_test @ coef
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def load_dorothea_score_map() -> tuple[dict[tuple[str, str], int], set[str]]:
    dorothea = pd.read_csv(DOROTHEA_PATH, sep="\t")
    dorothea["source"] = dorothea["source"].astype(str).str.upper()
    dorothea["target"] = dorothea["target"].astype(str).str.upper()
    confidence_map = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
    dorothea["confidence_score"] = (
        dorothea["confidence"].astype(str).str.upper().map(confidence_map).fillna(0).astype(int)
    )
    best = dorothea.groupby(["source", "target"], as_index=False)["confidence_score"].max()
    score_map = {
        (str(row.source), str(row.target)): int(row.confidence_score)
        for row in best.itertuples(index=False)
    }
    tf_sources = set(best["source"].astype(str).unique())
    return score_map, tf_sources


def load_trrust_pairs() -> set[tuple[str, str]]:
    trrust = pd.read_csv(
        TRRUST_PATH,
        sep="\t",
        header=None,
        names=["source", "target", "regulation", "pmid"],
    )
    trrust["source"] = trrust["source"].astype(str).str.upper()
    trrust["target"] = trrust["target"].astype(str).str.upper()
    return set(zip(trrust["source"], trrust["target"]))


def load_gene2go_upper() -> dict[str, set[str]]:
    raw = pickle.load(open(GENE2GO_PATH, "rb"))
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


def load_omnipath_pairs() -> set[tuple[str, str]]:
    omni = pd.read_csv(OMNIPATH_INTERACTIONS_PATH, sep="\t")
    required = {"source_genesymbol", "target_genesymbol"}
    if not required.issubset(omni.columns):
        return set()
    source = omni["source_genesymbol"].astype(str).str.upper()
    target = omni["target_genesymbol"].astype(str).str.upper()
    return set(zip(source, target))


def build_support_arrays(
    source_symbols: np.ndarray,
    target_symbols: np.ndarray,
    dorothea_score_map: dict[tuple[str, str], int],
    trrust_pairs: set[tuple[str, str]],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
) -> dict[str, np.ndarray]:
    n = source_symbols.size
    dorothea_norm = np.zeros(n, dtype=float)
    go_jaccard = np.zeros(n, dtype=float)
    omnipath_support = np.zeros(n, dtype=float)
    trrust_support = np.zeros(n, dtype=float)
    for i, (src_raw, tgt_raw) in enumerate(zip(source_symbols, target_symbols)):
        src = str(src_raw).upper()
        tgt = str(tgt_raw).upper()
        dorothea_norm[i] = float(dorothea_score_map.get((src, tgt), 0)) / 4.0
        omnipath_support[i] = float((src, tgt) in omnipath_pairs)
        trrust_support[i] = float((src, tgt) in trrust_pairs)
        go_src = gene2go_upper.get(src, set())
        go_tgt = gene2go_upper.get(tgt, set())
        union = len(go_src | go_tgt)
        if union > 0:
            go_jaccard[i] = float(len(go_src & go_tgt) / union)
        else:
            go_jaccard[i] = 0.0

    # Leakage-safe support score: TRRUST is recorded but excluded from model terms.
    support_score = 0.45 * dorothea_norm + 0.35 * omnipath_support + 0.20 * go_jaccard
    return {
        "support_score": support_score.astype(float),
        "dorothea_norm": dorothea_norm.astype(float),
        "go_jaccard": go_jaccard.astype(float),
        "omnipath_support": omnipath_support.astype(float),
        "trrust_support_bookkeeping": trrust_support.astype(float),
    }


def run_h40_support_interaction(
    coexp_cache: dict[str, CoexpressionDomainCache],
    dorothea_score_map: dict[tuple[str, str], int],
    trrust_pairs: set[tuple[str, str]],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
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

                top_genes = select_top_genes(split_edges, gene_cap=H40_GENE_CAP)
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
                support_bundle = build_support_arrays(
                    source_symbols=source_symbols,
                    target_symbols=target_symbols,
                    dorothea_score_map=dorothea_score_map,
                    trrust_pairs=trrust_pairs,
                    omnipath_pairs=omnipath_pairs,
                    gene2go_upper=gene2go_upper,
                )
                support_score = support_bundle["support_score"]

                gene_name_map: dict[int, str] = {}
                for row in split_edges[["source_idx", "source"]].drop_duplicates().itertuples(index=False):
                    gene_name_map[int(row.source_idx)] = str(row.source).upper()
                for row in split_edges[["target_idx", "target"]].drop_duplicates().itertuples(index=False):
                    gene_name_map[int(row.target_idx)] = str(row.target).upper()
                ordered_gene_symbols = [gene_name_map[int(g)] for g in edge_gene_indices]

                coexp_abs_corr, coexp_missing_fraction = coexp_cache[domain].abs_corr_for_genes(
                    ordered_gene_symbols,
                    max_cells=H40_MAX_COEXP_CELLS,
                    random_state=19_100 + domain_index * 1000 + seed_index * 100 + split_index,
                )
                coexp_scores = coexp_abs_corr[source_local, target_local]

                source_degree_map = split_edges["source_idx"].value_counts().to_dict()
                target_degree_map = split_edges["target_idx"].value_counts().to_dict()
                source_degree = split_edges["source_idx"].map(source_degree_map).astype(float).to_numpy()
                target_degree = split_edges["target_idx"].map(target_degree_map).astype(float).to_numpy()
                mean_degree = 0.5 * (source_degree + target_degree)

                for layer in H40_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue
                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = reduce_points(
                        points,
                        n_components=24,
                        random_state=19_200
                        + domain_index * 10_000
                        + seed_index * 1_000
                        + split_index * 100
                        + layer,
                    )

                    geodesic, transition, k_eff, used_component_bridging = geodesic_and_transition(
                        points_pca,
                        n_neighbors=24,
                    )
                    euclidean = cdist(points_pca, points_pca, metric="euclidean")
                    geodesic_distance = geodesic[source_local, target_local]
                    euclidean_distance = np.clip(euclidean[source_local, target_local], 1e-6, None)
                    diffusion_distance = diffusion_distance_scores(
                        transition=transition,
                        source_local=source_local,
                        target_local=target_local,
                        t=H40_DIFFUSION_T,
                    )
                    diffusion_score = -diffusion_distance
                    geodesic_score = -geodesic_distance
                    euclidean_score = -euclidean_distance
                    detour_ratio = geodesic_distance / euclidean_distance
                    convexity_deficit = compute_convexity_deficit(
                        geodesic=geodesic,
                        source_local=source_local,
                        target_local=target_local,
                    )

                    base_features = np.column_stack(
                        [
                            np.log1p(source_degree),
                            np.log1p(target_degree),
                            coexp_scores,
                            euclidean_score,
                            geodesic_score,
                            diffusion_score,
                            detour_ratio,
                            convexity_deficit,
                            support_score,
                        ]
                    )
                    interaction_features = np.column_stack(
                        [
                            support_score * diffusion_score,
                            support_score * detour_ratio,
                            support_score * convexity_deficit,
                        ]
                    )
                    full_features = np.column_stack([base_features, interaction_features])

                    score_base, _ = fit_logistic_scores_and_coef(
                        base_features,
                        labels,
                        random_state=19_300 + layer,
                    )
                    score_full, coef_full = fit_logistic_scores_and_coef(
                        full_features,
                        labels,
                        random_state=19_320 + layer,
                    )
                    auc_base = safe_auc(labels, score_base)
                    auc_full = safe_auc(labels, score_full)
                    auc_delta = float(auc_full - auc_base) if np.isfinite(auc_base) and np.isfinite(auc_full) else float("nan")
                    interaction_coef_mean = float(np.nanmean(coef_full[-3:])) if coef_full.size >= 3 else float("nan")

                    low_mask, high_mask = quantile_mask(support_score, q_low=0.10, q_high=0.90)
                    delta_low = float("nan")
                    delta_high = float("nan")
                    top_bottom_uplift_gap = float("nan")
                    if np.any(low_mask) and np.any(high_mask):
                        auc_base_low = safe_auc(labels[low_mask], score_base[low_mask])
                        auc_full_low = safe_auc(labels[low_mask], score_full[low_mask])
                        auc_base_high = safe_auc(labels[high_mask], score_base[high_mask])
                        auc_full_high = safe_auc(labels[high_mask], score_full[high_mask])
                        if np.isfinite(auc_base_low) and np.isfinite(auc_full_low):
                            delta_low = float(auc_full_low - auc_base_low)
                        if np.isfinite(auc_base_high) and np.isfinite(auc_full_high):
                            delta_high = float(auc_full_high - auc_base_high)
                        if np.isfinite(delta_low) and np.isfinite(delta_high):
                            top_bottom_uplift_gap = float(delta_high - delta_low)

                    strata = combine_strata(
                        degree_strata(mean_degree, max_bins=5),
                        degree_strata(coexp_scores, max_bins=5),
                        degree_strata(geodesic_distance, max_bins=5),
                    )
                    rng = np.random.default_rng(
                        19_400
                        + domain_index * 10_000
                        + seed_index * 1_000
                        + split_index * 100
                        + layer
                    )
                    null_interactions = np.empty(H40_NULL_PERM, dtype=float)
                    for perm_idx in range(H40_NULL_PERM):
                        support_perm = stratified_shuffle(support_score, strata=strata, rng=rng)
                        full_perm = np.column_stack(
                            [
                                np.column_stack(
                                    [
                                        np.log1p(source_degree),
                                        np.log1p(target_degree),
                                        coexp_scores,
                                        euclidean_score,
                                        geodesic_score,
                                        diffusion_score,
                                        detour_ratio,
                                        convexity_deficit,
                                        support_perm,
                                    ]
                                ),
                                np.column_stack(
                                    [
                                        support_perm * diffusion_score,
                                        support_perm * detour_ratio,
                                        support_perm * convexity_deficit,
                                    ]
                                ),
                            ]
                        )
                        _, coef_perm = fit_logistic_scores_and_coef(
                            full_perm,
                            labels,
                            random_state=19_500 + perm_idx,
                        )
                        null_value = float(np.nanmean(coef_perm[-3:])) if coef_perm.size >= 3 else float("nan")
                        null_interactions[perm_idx] = null_value
                        null_rows.append(
                            {
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_interaction_coef_mean": float(null_value),
                            }
                        )

                    p_interaction = empirical_upper_tail_p(interaction_coef_mean, null_interactions)

                    rows.append(
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
                            "mean_support_score": float(np.mean(support_score)),
                            "mean_dorothea_norm": float(np.mean(support_bundle["dorothea_norm"])),
                            "mean_go_jaccard": float(np.mean(support_bundle["go_jaccard"])),
                            "mean_omnipath_support": float(np.mean(support_bundle["omnipath_support"])),
                            "mean_trrust_support_bookkeeping": float(np.mean(support_bundle["trrust_support_bookkeeping"])),
                            "auc_base": float(auc_base),
                            "auc_full": float(auc_full),
                            "auc_delta_full_minus_base": float(auc_delta),
                            "interaction_coef_mean": float(interaction_coef_mean),
                            "p_interaction_upper": float(p_interaction),
                            "delta_auc_low_support_decile": float(delta_low),
                            "delta_auc_high_support_decile": float(delta_high),
                            "top_bottom_uplift_gap": float(top_bottom_uplift_gap),
                        }
                    )

    by_row_df = pd.DataFrame(rows).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h40_support_interaction_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["domain", "seed_tag", "split_regime", "layer", "perm_idx"]
    )
    null_path = ITER_DIR / "h40_support_interaction_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    domain_rows: list[dict[str, object]] = []
    for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
        domain_rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(group.shape[0]),
                "mean_interaction_coef": float(group["interaction_coef_mean"].mean()),
                "fraction_interaction_positive": float((group["interaction_coef_mean"] > 0.0).mean()),
                "fraction_p_interaction_lt_0_05": float((group["p_interaction_upper"] < 0.05).mean()),
                "combined_fisher_p_interaction": float(
                    safe_fisher_p(group["p_interaction_upper"].to_numpy(dtype=float))
                ),
                "mean_auc_delta_full_minus_base": float(group["auc_delta_full_minus_base"].mean()),
                "mean_top_bottom_uplift_gap": float(group["top_bottom_uplift_gap"].mean()),
                "fraction_top_bottom_uplift_positive": float(
                    (group["top_bottom_uplift_gap"] > 0.0).mean()
                ),
            }
        )
    domain_df = pd.DataFrame(domain_rows).sort_values(["domain", "split_regime"])
    domain_path = ITER_DIR / "h40_support_interaction_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    summary = {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_interaction_coef": float(by_row_df["interaction_coef_mean"].mean()) if not by_row_df.empty else float("nan"),
        "domain_split_positive_interaction": int((domain_df["mean_interaction_coef"] > 0).sum()) if not domain_df.empty else 0,
        "domain_split_fisher_sig": int((domain_df["combined_fisher_p_interaction"] < 0.05).sum()) if not domain_df.empty else 0,
        "domain_split_positive_top_bottom_uplift": int(
            (domain_df["mean_top_bottom_uplift_gap"] > 0).sum()
        )
        if not domain_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_layer_split": str(by_row_path),
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
        },
        "string_prior_available": False,
        "string_fallback": "Used OmniPath interaction membership as a PPI prior proxy because a STRING edge-score table is not available in local inputs.",
    }
    return summary


def run_h41_split_zigzag_proxy() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H41_SEED_TAG]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = build_split_masks(edge_df)

        source_edges_raw = edge_df.loc[split_masks["source_disjoint"]].copy()
        target_edges_raw = edge_df.loc[split_masks["target_disjoint"]].copy()
        if source_edges_raw["label"].nunique() < 2 or target_edges_raw["label"].nunique() < 2:
            continue

        source_top = set(select_top_genes(source_edges_raw, gene_cap=H41_GENE_CAP))
        target_top = set(select_top_genes(target_edges_raw, gene_cap=H41_GENE_CAP))
        union_genes = np.array(sorted(source_top | target_top), dtype=int)
        intersection_genes = set(source_top & target_top)
        if union_genes.size < 80 or len(intersection_genes) < 30:
            continue

        gene_to_union_local = {int(g): int(i) for i, g in enumerate(union_genes)}
        source_positions = np.array([gene_to_union_local[g] for g in sorted(source_top)], dtype=int)
        target_positions = np.array([gene_to_union_local[g] for g in sorted(target_top)], dtype=int)
        source_genes_sorted = np.array(sorted(source_top), dtype=int)
        target_genes_sorted = np.array(sorted(target_top), dtype=int)

        layer_bundle: dict[int, dict[str, object]] = {}
        for layer in H41_LAYERS:
            if layer >= layer_embeddings.shape[0]:
                continue
            points_union = layer_embeddings[layer, union_genes, :]
            points_union_pca = reduce_points(
                points_union,
                n_components=20,
                random_state=19_600 + domain_index * 100 + layer,
            )
            geodesic, _, _, used_component_bridging = geodesic_and_transition(
                points_union_pca,
                n_neighbors=20,
            )

            points_source = points_union_pca[source_positions]
            points_target = points_union_pca[target_positions]
            source_local_h1_raw = compute_local_h1_scores(points_source, n_neighbors=H41_LOCAL_NEIGHBORS)
            target_local_h1_raw = compute_local_h1_scores(points_target, n_neighbors=H41_LOCAL_NEIGHBORS)

            source_h1_by_gene = {
                int(g): float(v) for g, v in zip(source_genes_sorted.tolist(), source_local_h1_raw.tolist())
            }
            target_h1_by_gene = {
                int(g): float(v) for g, v in zip(target_genes_sorted.tolist(), target_local_h1_raw.tolist())
            }

            zigzag_local = np.zeros(union_genes.size, dtype=float)
            source_only_local = np.zeros(union_genes.size, dtype=float)
            target_only_local = np.zeros(union_genes.size, dtype=float)
            for gene in union_genes.tolist():
                pos = gene_to_union_local[int(gene)]
                src_h1 = float(source_h1_by_gene.get(int(gene), 0.0))
                tgt_h1 = float(target_h1_by_gene.get(int(gene), 0.0))
                source_only_local[pos] = src_h1
                target_only_local[pos] = tgt_h1
                if int(gene) in intersection_genes:
                    zigzag_local[pos] = min(src_h1, tgt_h1)
                else:
                    zigzag_local[pos] = 0.25 * max(src_h1, tgt_h1)

            layer_bundle[int(layer)] = {
                "geodesic": geodesic,
                "zigzag_local": zigzag_local,
                "source_only_local": source_only_local,
                "target_only_local": target_only_local,
                "used_component_bridging": bool(used_component_bridging),
            }

        if len(layer_bundle) < 2:
            continue

        for split_regime, split_edge_df in [
            ("source_disjoint", source_edges_raw),
            ("target_disjoint", target_edges_raw),
        ]:
            split_df = split_edge_df.loc[
                split_edge_df["source_idx"].isin(union_genes) & split_edge_df["target_idx"].isin(union_genes)
            ].copy()
            if split_df["label"].nunique() < 2:
                continue

            source_local = split_df["source_idx"].map(gene_to_union_local).to_numpy(dtype=int)
            target_local = split_df["target_idx"].map(gene_to_union_local).to_numpy(dtype=int)
            labels = split_df["label"].to_numpy(dtype=int)

            for layer, bundle in sorted(layer_bundle.items()):
                geodesic = np.asarray(bundle["geodesic"], dtype=float)
                base_score = -geodesic[source_local, target_local]

                zigzag_local = np.asarray(bundle["zigzag_local"], dtype=float)
                zigzag_edge = 0.5 * (zigzag_local[source_local] + zigzag_local[target_local])
                observed_score = base_score + standardize_columns(zigzag_edge[:, None]).ravel()

                if split_regime == "source_disjoint":
                    swap_local = np.asarray(bundle["target_only_local"], dtype=float)
                else:
                    swap_local = np.asarray(bundle["source_only_local"], dtype=float)
                swap_edge = 0.5 * (swap_local[source_local] + swap_local[target_local])
                swap_score = base_score + standardize_columns(swap_edge[:, None]).ravel()

                auc_base = safe_auc(labels, base_score)
                auc_observed = safe_auc(labels, observed_score)
                auc_swap = safe_auc(labels, swap_score)
                delta_observed = float(auc_observed - auc_base) if np.isfinite(auc_observed) and np.isfinite(auc_base) else float("nan")
                delta_swap = float(auc_swap - auc_base) if np.isfinite(auc_swap) and np.isfinite(auc_base) else float("nan")

                layer_perm_deltas: list[float] = []
                for alt_layer, alt_bundle in sorted(layer_bundle.items()):
                    if alt_layer == layer:
                        continue
                    alt_local = np.asarray(alt_bundle["zigzag_local"], dtype=float)
                    alt_edge = 0.5 * (alt_local[source_local] + alt_local[target_local])
                    alt_score = base_score + standardize_columns(alt_edge[:, None]).ravel()
                    auc_alt = safe_auc(labels, alt_score)
                    if np.isfinite(auc_alt) and np.isfinite(auc_base):
                        alt_delta = float(auc_alt - auc_base)
                    else:
                        alt_delta = float("nan")
                    layer_perm_deltas.append(alt_delta)
                    null_rows.append(
                        {
                            "domain": domain,
                            "seed_tag": H41_SEED_TAG,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_layer": int(alt_layer),
                            "null_delta_auc_layer_permutation": float(alt_delta),
                        }
                    )
                layer_perm_array = np.asarray(layer_perm_deltas, dtype=float)
                p_layer_perm = empirical_upper_tail_p(delta_observed, layer_perm_array)

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": H41_SEED_TAG,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges": int(labels.size),
                        "n_positive": int(labels.sum()),
                        "n_union_genes": int(union_genes.size),
                        "n_intersection_genes": int(len(intersection_genes)),
                        "auc_base_geodesic": float(auc_base),
                        "auc_observed_zigzag_proxy": float(auc_observed),
                        "auc_split_swap_control": float(auc_swap),
                        "delta_auc_observed_minus_base": float(delta_observed),
                        "delta_auc_swap_minus_base": float(delta_swap),
                        "delta_auc_observed_minus_swap": float(delta_observed - delta_swap)
                        if np.isfinite(delta_observed) and np.isfinite(delta_swap)
                        else float("nan"),
                        "p_layer_perm_upper": float(p_layer_perm),
                        "used_component_bridging": bool(bundle["used_component_bridging"]),
                    }
                )

    by_row_df = pd.DataFrame(rows).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h41_zigzag_persistence_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["domain", "seed_tag", "split_regime", "layer", "perm_layer"]
    )
    null_path = ITER_DIR / "h41_zigzag_persistence_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    domain_rows: list[dict[str, object]] = []
    for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
        domain_rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(group.shape[0]),
                "mean_delta_auc_observed_minus_base": float(group["delta_auc_observed_minus_base"].mean()),
                "mean_delta_auc_observed_minus_swap": float(group["delta_auc_observed_minus_swap"].mean()),
                "fraction_delta_positive": float((group["delta_auc_observed_minus_base"] > 0.0).mean()),
                "fraction_p_layer_perm_lt_0_05": float((group["p_layer_perm_upper"] < 0.05).mean()),
                "combined_fisher_p_layer_perm": float(
                    safe_fisher_p(group["p_layer_perm_upper"].to_numpy(dtype=float))
                ),
            }
        )
    domain_df = pd.DataFrame(domain_rows).sort_values(["domain", "split_regime"])
    domain_path = ITER_DIR / "h41_zigzag_persistence_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    summary = {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc_observed_minus_base": float(by_row_df["delta_auc_observed_minus_base"].mean()) if not by_row_df.empty else float("nan"),
        "domain_split_positive_delta": int((domain_df["mean_delta_auc_observed_minus_base"] > 0).sum()) if not domain_df.empty else 0,
        "domain_split_fisher_sig": int((domain_df["combined_fisher_p_layer_perm"] < 0.05).sum()) if not domain_df.empty else 0,
        "artifact_paths": {
            "by_seed_layer_split": str(by_row_path),
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
        },
        "zigzag_runtime_note": "True zigzag persistence libraries are unavailable in this environment; executed a split-local PH proxy with split-swap and layer-permutation controls as a bounded fallback.",
    }
    return summary


def run_h42_id_oos() -> dict[str, object]:
    source_path = Path("iterations/iter_0018/h38_id_distribution_moments_by_seed_layer_split.csv")
    if not source_path.exists():
        raise FileNotFoundError(f"Required input for H42 not found: {source_path}")
    df = pd.read_csv(source_path)

    by_row: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    feature_mean_cols = ["layer", "twonn_mean", "pr_mean"]
    feature_full_cols = [
        "layer",
        "twonn_mean",
        "pr_mean",
        "twonn_var",
        "twonn_skew",
        "pr_var",
        "pr_skew",
    ]

    for (domain, split_regime), group in df.groupby(["domain", "split_regime"], sort=True):
        group = group.copy().sort_values(["seed_tag", "layer"])
        if group["seed_tag"].nunique() < 2 or group["layer"].nunique() < 4:
            continue

        x_mean_all = group[feature_mean_cols].to_numpy(dtype=float)
        x_full_all = group[feature_full_cols].to_numpy(dtype=float)
        y_all = group["edge_auc_local_linearity"].to_numpy(dtype=float)
        seed_all = group["seed_tag"].astype(str).to_numpy()
        layer_all = group["layer"].to_numpy(dtype=int)

        observed_mean_delta: dict[str, float] = {}

        # Leave-one-layer-out.
        layer_deltas: list[float] = []
        for hold_layer in sorted(np.unique(layer_all).tolist()):
            test_mask = layer_all == hold_layer
            train_mask = ~test_mask
            if np.sum(train_mask) < 10 or np.sum(test_mask) < 2:
                continue
            r2_mean = fit_linear_r2(
                x_train=x_mean_all[train_mask],
                y_train=y_all[train_mask],
                x_test=x_mean_all[test_mask],
                y_test=y_all[test_mask],
            )
            r2_full = fit_linear_r2(
                x_train=x_full_all[train_mask],
                y_train=y_all[train_mask],
                x_test=x_full_all[test_mask],
                y_test=y_all[test_mask],
            )
            delta = float(r2_full - r2_mean) if np.isfinite(r2_mean) and np.isfinite(r2_full) else float("nan")
            layer_deltas.append(delta)
            by_row.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "evaluation": "leave_layer_out",
                    "holdout_id": str(int(hold_layer)),
                    "n_train": int(np.sum(train_mask)),
                    "n_test": int(np.sum(test_mask)),
                    "r2_mean_model": float(r2_mean),
                    "r2_full_model": float(r2_full),
                    "delta_r2_full_minus_mean": float(delta),
                }
            )
        observed_mean_delta["leave_layer_out"] = float(np.nanmean(np.asarray(layer_deltas, dtype=float)))

        # Leave-one-seed-out.
        seed_deltas: list[float] = []
        for hold_seed in sorted(np.unique(seed_all).tolist()):
            test_mask = seed_all == hold_seed
            train_mask = ~test_mask
            if np.sum(train_mask) < 10 or np.sum(test_mask) < 4:
                continue
            r2_mean = fit_linear_r2(
                x_train=x_mean_all[train_mask],
                y_train=y_all[train_mask],
                x_test=x_mean_all[test_mask],
                y_test=y_all[test_mask],
            )
            r2_full = fit_linear_r2(
                x_train=x_full_all[train_mask],
                y_train=y_all[train_mask],
                x_test=x_full_all[test_mask],
                y_test=y_all[test_mask],
            )
            delta = float(r2_full - r2_mean) if np.isfinite(r2_mean) and np.isfinite(r2_full) else float("nan")
            seed_deltas.append(delta)
            by_row.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "evaluation": "leave_seed_out",
                    "holdout_id": str(hold_seed),
                    "n_train": int(np.sum(train_mask)),
                    "n_test": int(np.sum(test_mask)),
                    "r2_mean_model": float(r2_mean),
                    "r2_full_model": float(r2_full),
                    "delta_r2_full_minus_mean": float(delta),
                }
            )
        observed_mean_delta["leave_seed_out"] = float(np.nanmean(np.asarray(seed_deltas, dtype=float)))

        rng = np.random.default_rng(19_900 + hash((domain, split_regime)) % 10_000)
        null_by_eval: dict[str, list[float]] = {"leave_layer_out": [], "leave_seed_out": []}
        for perm_idx in range(H42_NULL_PERM):
            y_perm = rng.permutation(y_all)

            perm_layer_deltas: list[float] = []
            for hold_layer in sorted(np.unique(layer_all).tolist()):
                test_mask = layer_all == hold_layer
                train_mask = ~test_mask
                if np.sum(train_mask) < 10 or np.sum(test_mask) < 2:
                    continue
                r2_mean = fit_linear_r2(
                    x_train=x_mean_all[train_mask],
                    y_train=y_perm[train_mask],
                    x_test=x_mean_all[test_mask],
                    y_test=y_perm[test_mask],
                )
                r2_full = fit_linear_r2(
                    x_train=x_full_all[train_mask],
                    y_train=y_perm[train_mask],
                    x_test=x_full_all[test_mask],
                    y_test=y_perm[test_mask],
                )
                if np.isfinite(r2_mean) and np.isfinite(r2_full):
                    perm_layer_deltas.append(float(r2_full - r2_mean))
            null_layer_mean = float(np.nanmean(np.asarray(perm_layer_deltas, dtype=float)))
            null_by_eval["leave_layer_out"].append(null_layer_mean)
            null_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "evaluation": "leave_layer_out",
                    "perm_idx": int(perm_idx),
                    "null_mean_delta_r2": float(null_layer_mean),
                }
            )

            perm_seed_deltas: list[float] = []
            for hold_seed in sorted(np.unique(seed_all).tolist()):
                test_mask = seed_all == hold_seed
                train_mask = ~test_mask
                if np.sum(train_mask) < 10 or np.sum(test_mask) < 4:
                    continue
                r2_mean = fit_linear_r2(
                    x_train=x_mean_all[train_mask],
                    y_train=y_perm[train_mask],
                    x_test=x_mean_all[test_mask],
                    y_test=y_perm[test_mask],
                )
                r2_full = fit_linear_r2(
                    x_train=x_full_all[train_mask],
                    y_train=y_perm[train_mask],
                    x_test=x_full_all[test_mask],
                    y_test=y_perm[test_mask],
                )
                if np.isfinite(r2_mean) and np.isfinite(r2_full):
                    perm_seed_deltas.append(float(r2_full - r2_mean))
            null_seed_mean = float(np.nanmean(np.asarray(perm_seed_deltas, dtype=float)))
            null_by_eval["leave_seed_out"].append(null_seed_mean)
            null_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "evaluation": "leave_seed_out",
                    "perm_idx": int(perm_idx),
                    "null_mean_delta_r2": float(null_seed_mean),
                }
            )

        for evaluation in ["leave_layer_out", "leave_seed_out"]:
            obs_delta = float(observed_mean_delta[evaluation])
            p_upper = empirical_upper_tail_p(obs_delta, np.asarray(null_by_eval[evaluation], dtype=float))
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "evaluation": evaluation,
                    "observed_mean_delta_r2": float(obs_delta),
                    "p_delta_r2_upper": float(p_upper),
                    "fraction_holdout_delta_positive": float(
                        (
                            np.asarray(
                                [
                                    row["delta_r2_full_minus_mean"]
                                    for row in by_row
                                    if row["domain"] == domain
                                    and row["split_regime"] == split_regime
                                    and row["evaluation"] == evaluation
                                ],
                                dtype=float,
                            )
                            > 0.0
                        ).mean()
                    ),
                }
            )

    by_row_df = pd.DataFrame(by_row).sort_values(["domain", "split_regime", "evaluation", "holdout_id"])
    by_row_path = ITER_DIR / "h42_id_oos_by_seed_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(["domain", "split_regime", "evaluation", "perm_idx"])
    null_path = ITER_DIR / "h42_id_oos_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values(["domain", "split_regime", "evaluation"])
    summary_path = ITER_DIR / "h42_id_oos_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    summary = {
        "rows_tested": int(summary_df.shape[0]),
        "mean_observed_delta_r2": float(summary_df["observed_mean_delta_r2"].mean()) if not summary_df.empty else float("nan"),
        "rows_positive_delta": int((summary_df["observed_mean_delta_r2"] > 0).sum()) if not summary_df.empty else 0,
        "rows_p_lt_0_05": int((summary_df["p_delta_r2_upper"] < 0.05).sum()) if not summary_df.empty else 0,
        "artifact_paths": {
            "by_seed_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
        "source_h38_artifact": str(source_path),
    }
    return summary


def main() -> None:
    required_paths = [
        TRRUST_PATH,
        DOROTHEA_PATH,
        GENE2GO_PATH,
        OMNIPATH_INTERACTIONS_PATH,
        Path("iterations/iter_0018/h38_id_distribution_moments_by_seed_layer_split.csv"),
    ]
    for domain, run_map in SCGPT_RUNS_BY_DOMAIN.items():
        required_paths.append(PROCESSED_H5AD_BY_DOMAIN[domain])
        for run_dir in run_map.values():
            required_paths.append(run_dir / "cycle1_edge_dataset.tsv")
            required_paths.append(run_dir / "layer_gene_embeddings.npy")

    missing = [str(p) for p in required_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    dorothea_score_map, _ = load_dorothea_score_map()
    trrust_pairs = load_trrust_pairs()
    omnipath_pairs = load_omnipath_pairs()
    gene2go_upper = load_gene2go_upper()

    coexp_cache = {
        domain: CoexpressionDomainCache(path)
        for domain, path in PROCESSED_H5AD_BY_DOMAIN.items()
    }
    try:
        h40_summary = run_h40_support_interaction(
            coexp_cache=coexp_cache,
            dorothea_score_map=dorothea_score_map,
            trrust_pairs=trrust_pairs,
            omnipath_pairs=omnipath_pairs,
            gene2go_upper=gene2go_upper,
        )
    finally:
        for cache in coexp_cache.values():
            cache.close()

    h41_summary = run_h41_split_zigzag_proxy()
    h42_summary = run_h42_id_oos()

    summary = {
        "iteration": "iter_0019",
        "h40": h40_summary,
        "h41": h41_summary,
        "h42": h42_summary,
    }
    summary_path = ITER_DIR / "iter0019_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
