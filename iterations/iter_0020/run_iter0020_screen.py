from __future__ import annotations

import io
import json
import math
import pickle
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import dionysus as d
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import cdist
from scipy.stats import combine_pvalues
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


ITER_DIR = Path("iterations/iter_0020")
ITER_DIR.mkdir(parents=True, exist_ok=True)

SCGPT_RUNS_BY_DOMAIN: dict[str, dict[str, Path]] = {
    "immune": {
        "seed42_main": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle4_immune_main"
        )
    },
    "lung": {
        "seed42_main": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle6_lung_main"
        )
    },
    "external_lung": {
        "seed42_main": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle7_external_lung_main"
        )
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

H43_LAYERS = [0, 3, 7, 11]
H43_GENE_CAP = 220
H43_DIFFUSION_T = 2
H43_NULL_PERM = 80
H43_MAX_COEXP_CELLS = 2500
H43_MAX_ONTOLOGY_CELLS = 3000
H43_ONTOLOGY_TOP_CLASSES = 16
H43_STRING_CACHE_PATH = ITER_DIR / "h43_string_network_api.tsv"

H44_LAYERS = [0, 3, 7, 11]
H44_GENE_CAP = 170
H44_KNN = 10
H44_NULL_PERM = 40

H45_SOURCE_PATH = Path("iterations/iter_0018/h38_id_distribution_moments_by_seed_layer_split.csv")
H45_NULL_PERM = 140
H45_BOOTSTRAP_NULL = 800


@dataclass
class CoexpressionDomainCache:
    """
    Backed h5ad cache with fast access for coexpression and ontology profile similarity.
    """

    path: Path

    def __post_init__(self) -> None:
        self.adata = ad.read_h5ad(self.path, backed="r")
        var_names = pd.Index(self.adata.var_names.astype(str)).str.upper()
        self.gene_to_var_idx = {str(g): int(i) for i, g in enumerate(var_names)}

        if "cell_type_ontology_term_id" in self.adata.obs.columns:
            self.ontology_labels = (
                self.adata.obs["cell_type_ontology_term_id"].astype(str).fillna("NA").to_numpy()
            )
        else:
            self.ontology_labels = np.repeat("NA", int(self.adata.n_obs)).astype(str)

    def close(self) -> None:
        self.adata.file.close()

    def _sample_cell_index(self, max_cells: int, random_state: int) -> np.ndarray:
        n_cells = int(self.adata.n_obs)
        if max_cells > 0 and n_cells > max_cells:
            rng = np.random.default_rng(random_state)
            return np.sort(rng.choice(n_cells, size=max_cells, replace=False))
        return np.arange(n_cells, dtype=int)

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
        cell_idx = self._sample_cell_index(max_cells=max_cells, random_state=random_state)

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

    def ontology_similarity_for_genes(
        self,
        gene_symbols_upper: list[str],
        max_cells: int,
        top_classes: int,
        random_state: int,
    ) -> tuple[np.ndarray, float, int]:
        n_genes = len(gene_symbols_upper)
        sim = np.zeros((n_genes, n_genes), dtype=np.float64)
        if n_genes == 0:
            return sim, 1.0, 0

        found_mask = np.array([g in self.gene_to_var_idx for g in gene_symbols_upper], dtype=bool)
        found_positions = np.where(found_mask)[0]
        missing_fraction = float(1.0 - found_mask.mean())
        if found_positions.size < 2:
            return sim, missing_fraction, 0

        var_idx = [self.gene_to_var_idx[gene_symbols_upper[pos]] for pos in found_positions]
        cell_idx = self._sample_cell_index(max_cells=max_cells, random_state=random_state)
        labels = self.ontology_labels[cell_idx]
        label_counts = pd.Series(labels).value_counts()
        top_labels = label_counts.head(max(1, top_classes)).index.astype(str).tolist()

        x = self.adata.X[cell_idx, :][:, var_idx]
        if sp.issparse(x):
            detected = x.toarray() > 0
        else:
            detected = np.asarray(x) > 0

        profile = np.zeros((found_positions.size, len(top_labels)), dtype=np.float64)
        for class_idx, label in enumerate(top_labels):
            mask = labels == label
            if np.any(mask):
                profile[:, class_idx] = detected[mask].mean(axis=0)

        norms = np.linalg.norm(profile, axis=1, keepdims=True)
        norms = np.where(norms <= 1e-12, 1.0, norms)
        profile_norm = profile / norms
        found_sim = profile_norm @ profile_norm.T
        found_sim = np.nan_to_num(found_sim, nan=0.0, posinf=0.0, neginf=0.0)

        for i_local, i_global in enumerate(found_positions):
            for j_local, j_global in enumerate(found_positions):
                sim[i_global, j_global] = found_sim[i_local, j_local]
        np.fill_diagonal(sim, 0.0)
        return sim, missing_fraction, len(top_labels)


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
            dval = float(dist)
            if dval < knn_dist[i, j]:
                knn_dist[i, j] = dval
                knn_dist[j, i] = dval

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
                dval = float(local.ravel()[flat_idx])
                if dval < best_dist:
                    pos_a, pos_b = np.unravel_index(flat_idx, local.shape)
                    best_i = int(nodes_a[pos_a])
                    best_j = int(nodes_b[pos_b])
                    best_dist = dval
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
    for stratum in np.unique(strata):
        idx = np.where(strata == stratum)[0]
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


def fit_linear_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    if x_train.ndim == 1:
        x_train = x_train[:, None]
    if x_test.ndim == 1:
        x_test = x_test[:, None]
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)

    if x_train.shape[0] < 3 or x_test.shape[0] < 1:
        return np.repeat(np.nan, x_test.shape[0]).astype(float)

    design_train = np.column_stack([np.ones(y_train.size), x_train])
    coef, _, _, _ = np.linalg.lstsq(design_train, y_train, rcond=None)
    design_test = np.column_stack([np.ones(x_test.shape[0]), x_test])
    return (design_test @ coef).astype(float)


def r2_from_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(mask) < 2:
        return float("nan")
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def winsorized_r2(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    y_train = np.asarray(y_train, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2:
        return float("nan")
    low, high = np.quantile(y_train, [0.10, 0.90])
    y_true_w = np.clip(y_true, low, high)
    y_pred_w = np.clip(y_pred, low, high)
    return r2_from_prediction(y_true_w, y_pred_w)


def trimmed_r2(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    y_train = np.asarray(y_train, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 4:
        return float("nan")
    low, high = np.quantile(y_train, [0.10, 0.90])
    mask = (y_true >= low) & (y_true <= high)
    if np.sum(mask) < 2:
        return float("nan")
    return r2_from_prediction(y_true[mask], y_pred[mask])


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


def collect_h43_gene_symbols() -> set[str]:
    symbols: set[str] = set()
    for domain, run_map in SCGPT_RUNS_BY_DOMAIN.items():
        run_dir = run_map["seed42_main"]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        split_masks = build_split_masks(edge_df)
        for split_mask in split_masks.values():
            split_edges = edge_df.loc[split_mask].copy()
            top_genes = set(select_top_genes(split_edges, gene_cap=H43_GENE_CAP))
            sub = split_edges.loc[
                split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
            ]
            symbols.update(sub["source"].astype(str).str.upper().tolist())
            symbols.update(sub["target"].astype(str).str.upper().tolist())
    return symbols


def fetch_string_scores(
    gene_symbols_upper: set[str],
    cache_path: Path,
) -> tuple[dict[tuple[str, str], float], dict[str, object]]:
    text: str
    source = "cache"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        text = cache_path.read_text(encoding="utf-8")
    else:
        source = "api"
        payload = {
            "identifiers": "\r".join(sorted(gene_symbols_upper)),
            "species": "9606",
            "required_score": "150",
            "caller_identity": "subproject40_iter0020",
        }
        encoded = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://string-db.org/api/tsv/network",
            data=encoded,
            headers={"User-Agent": "subproject40-topology/iter0020"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            text = response.read().decode("utf-8", errors="replace")
        cache_path.write_text(text, encoding="utf-8")

    if not text.strip():
        return {}, {"source": source, "rows": 0, "error": "empty_response"}

    df = pd.read_csv(io.StringIO(text), sep="\t")
    required_cols = {"preferredName_A", "preferredName_B", "score"}
    if not required_cols.issubset(df.columns):
        return {}, {
            "source": source,
            "rows": int(df.shape[0]),
            "error": "missing_required_columns",
            "columns": sorted(df.columns.tolist()),
        }

    src = df["preferredName_A"].astype(str).str.upper().to_numpy()
    tgt = df["preferredName_B"].astype(str).str.upper().to_numpy()
    score = df["score"].astype(float).to_numpy()

    mapping: dict[tuple[str, str], float] = {}
    for s, t, val in zip(src, tgt, score):
        if not np.isfinite(val):
            continue
        vv = float(np.clip(val, 0.0, 1.0))
        key_ab = (str(s), str(t))
        key_ba = (str(t), str(s))
        mapping[key_ab] = max(vv, mapping.get(key_ab, 0.0))
        mapping[key_ba] = max(vv, mapping.get(key_ba, 0.0))

    return mapping, {
        "source": source,
        "rows": int(df.shape[0]),
        "pairs_indexed": int(len(mapping) // 2),
        "cache_path": str(cache_path),
    }


def build_support_arrays(
    source_symbols: np.ndarray,
    target_symbols: np.ndarray,
    dorothea_score_map: dict[tuple[str, str], int],
    trrust_pairs: set[tuple[str, str]],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_score_map: dict[tuple[str, str], float],
    ontology_similarity_edge: np.ndarray,
) -> dict[str, np.ndarray]:
    n = source_symbols.size
    dorothea_norm = np.zeros(n, dtype=float)
    go_jaccard = np.zeros(n, dtype=float)
    omnipath_support = np.zeros(n, dtype=float)
    trrust_support = np.zeros(n, dtype=float)
    string_score = np.zeros(n, dtype=float)

    for i, (src_raw, tgt_raw) in enumerate(zip(source_symbols, target_symbols)):
        src = str(src_raw).upper()
        tgt = str(tgt_raw).upper()
        dorothea_norm[i] = float(dorothea_score_map.get((src, tgt), 0)) / 4.0
        omnipath_support[i] = float((src, tgt) in omnipath_pairs)
        trrust_support[i] = float((src, tgt) in trrust_pairs)
        string_score[i] = float(string_score_map.get((src, tgt), 0.0))
        go_src = gene2go_upper.get(src, set())
        go_tgt = gene2go_upper.get(tgt, set())
        union = len(go_src | go_tgt)
        if union > 0:
            go_jaccard[i] = float(len(go_src & go_tgt) / union)
        else:
            go_jaccard[i] = 0.0

    ontology_edge = np.asarray(ontology_similarity_edge, dtype=float)
    ontology_edge = np.clip(ontology_edge, 0.0, 1.0)

    # Leakage-safe support score: TRRUST is bookkeeping only.
    support_score = (
        0.30 * dorothea_norm
        + 0.30 * string_score
        + 0.20 * go_jaccard
        + 0.10 * omnipath_support
        + 0.10 * ontology_edge
    )
    return {
        "support_score": support_score.astype(float),
        "dorothea_norm": dorothea_norm.astype(float),
        "string_score": string_score.astype(float),
        "go_jaccard": go_jaccard.astype(float),
        "omnipath_support": omnipath_support.astype(float),
        "ontology_similarity_edge": ontology_edge.astype(float),
        "trrust_support_bookkeeping": trrust_support.astype(float),
    }


def run_h43_support_interaction_ontology(
    coexp_cache: dict[str, CoexpressionDomainCache],
    dorothea_score_map: dict[tuple[str, str], int],
    trrust_pairs: set[tuple[str, str]],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    candidate_symbols = collect_h43_gene_symbols()
    string_score_map, string_info = fetch_string_scores(candidate_symbols, cache_path=H43_STRING_CACHE_PATH)

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

            top_genes = select_top_genes(split_edges, gene_cap=H43_GENE_CAP)
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
                max_cells=H43_MAX_COEXP_CELLS,
                random_state=20_100 + domain_index * 100 + split_index,
            )
            coexp_scores = coexp_abs_corr[source_local, target_local]

            ontology_sim_mat, ontology_missing_fraction, n_ontology_classes = coexp_cache[
                domain
            ].ontology_similarity_for_genes(
                ordered_gene_symbols,
                max_cells=H43_MAX_ONTOLOGY_CELLS,
                top_classes=H43_ONTOLOGY_TOP_CLASSES,
                random_state=20_200 + domain_index * 100 + split_index,
            )
            ontology_edge_scores = ontology_sim_mat[source_local, target_local]

            source_symbols = split_edges["source"].astype(str).str.upper().to_numpy()
            target_symbols = split_edges["target"].astype(str).str.upper().to_numpy()
            support_bundle = build_support_arrays(
                source_symbols=source_symbols,
                target_symbols=target_symbols,
                dorothea_score_map=dorothea_score_map,
                trrust_pairs=trrust_pairs,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_score_map=string_score_map,
                ontology_similarity_edge=ontology_edge_scores,
            )
            support_score = support_bundle["support_score"]

            source_degree_map = split_edges["source_idx"].value_counts().to_dict()
            target_degree_map = split_edges["target_idx"].value_counts().to_dict()
            source_degree = split_edges["source_idx"].map(source_degree_map).astype(float).to_numpy()
            target_degree = split_edges["target_idx"].map(target_degree_map).astype(float).to_numpy()
            mean_degree = 0.5 * (source_degree + target_degree)

            for layer in H43_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue
                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = reduce_points(
                    points,
                    n_components=24,
                    random_state=20_300 + domain_index * 100 + split_index * 10 + layer,
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
                    t=H43_DIFFUSION_T,
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
                    random_state=20_400 + layer,
                )
                score_full, coef_full = fit_logistic_scores_and_coef(
                    full_features,
                    labels,
                    random_state=20_420 + layer,
                )
                auc_base = safe_auc(labels, score_base)
                auc_full = safe_auc(labels, score_full)
                auc_delta = (
                    float(auc_full - auc_base)
                    if np.isfinite(auc_base) and np.isfinite(auc_full)
                    else float("nan")
                )
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
                    degree_strata(ontology_edge_scores, max_bins=5),
                )
                rng = np.random.default_rng(
                    20_500 + domain_index * 1000 + split_index * 100 + layer
                )
                null_interactions = np.empty(H43_NULL_PERM, dtype=float)
                for perm_idx in range(H43_NULL_PERM):
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
                        random_state=20_600 + perm_idx,
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
                        "ontology_missing_fraction": float(ontology_missing_fraction),
                        "n_ontology_classes_used": int(n_ontology_classes),
                        "mean_support_score": float(np.mean(support_score)),
                        "mean_dorothea_norm": float(np.mean(support_bundle["dorothea_norm"])),
                        "mean_string_score": float(np.mean(support_bundle["string_score"])),
                        "mean_go_jaccard": float(np.mean(support_bundle["go_jaccard"])),
                        "mean_omnipath_support": float(np.mean(support_bundle["omnipath_support"])),
                        "mean_ontology_similarity_edge": float(
                            np.mean(support_bundle["ontology_similarity_edge"])
                        ),
                        "mean_trrust_support_bookkeeping": float(
                            np.mean(support_bundle["trrust_support_bookkeeping"])
                        ),
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
    by_row_path = ITER_DIR / "h43_support_interaction_ontology_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["domain", "seed_tag", "split_regime", "layer", "perm_idx"]
    )
    null_path = ITER_DIR / "h43_support_interaction_ontology_null_summary.csv"
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
    domain_path = ITER_DIR / "h43_support_interaction_ontology_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    summary = {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_interaction_coef": float(by_row_df["interaction_coef_mean"].mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_split_positive_interaction": int((domain_df["mean_interaction_coef"] > 0).sum())
        if not domain_df.empty
        else 0,
        "domain_split_fisher_sig": int((domain_df["combined_fisher_p_interaction"] < 0.05).sum())
        if not domain_df.empty
        else 0,
        "domain_split_positive_top_bottom_uplift": int(
            (domain_df["mean_top_bottom_uplift_gap"] > 0).sum()
        )
        if not domain_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_layer_split": str(by_row_path),
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
            "string_network_cache": str(H43_STRING_CACHE_PATH),
        },
        "string_info": string_info,
    }
    return summary


def build_knn_edges_for_subset(
    points_union: np.ndarray,
    subset_positions: np.ndarray,
    n_neighbors: int,
) -> set[tuple[int, int]]:
    subset_positions = np.asarray(subset_positions, dtype=int)
    if subset_positions.size < 3:
        return set()
    k = max(2, min(n_neighbors, subset_positions.size - 1))
    sub_points = points_union[subset_positions]
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(sub_points)
    _, indices = nbrs.kneighbors(sub_points)

    edges: set[tuple[int, int]] = set()
    for i_local in range(subset_positions.size):
        src_union = int(subset_positions[i_local])
        for j_local in indices[i_local, 1:]:
            tgt_union = int(subset_positions[int(j_local)])
            if src_union == tgt_union:
                continue
            edge = (min(src_union, tgt_union), max(src_union, tgt_union))
            edges.add(edge)
    return edges


def zigzag_h1_metrics(
    n_vertices: int,
    source_edges: set[tuple[int, int]],
    target_edges: set[tuple[int, int]],
) -> dict[str, float]:
    simplices: list[d.Simplex] = []
    intervals_by_key: dict[tuple[int, ...], list[float]] = {}

    for v in range(n_vertices):
        simplex = d.Simplex([int(v)], 0.0)
        simplices.append(simplex)
        intervals_by_key[(int(v),)] = [0.0, 3.0]

    union_edges = source_edges | target_edges
    for edge in sorted(union_edges):
        u, v = int(edge[0]), int(edge[1])
        simplex = d.Simplex([u, v], 1.0)
        simplices.append(simplex)
        in_source = edge in source_edges
        in_target = edge in target_edges
        if in_source and in_target:
            interval = [0.0, 3.0]
        elif in_source:
            interval = [0.0, 2.0]
        else:
            interval = [1.0, 3.0]
        intervals_by_key[(u, v)] = interval

    filtration = d.Filtration(simplices)
    times: list[list[float]] = [[] for _ in range(len(filtration))]
    for simplex in filtration:
        verts = tuple(int(v) for v in simplex)
        key = tuple(sorted(verts))
        idx = filtration.index(simplex)
        if key in intervals_by_key:
            times[idx] = intervals_by_key[key]
        else:
            times[idx] = [0.0, 3.0]

    _, diagrams, _ = d.zigzag_homology_persistence(filtration, times)
    if len(diagrams) < 2:
        return {
            "h1_total_lifetime": 0.0,
            "h1_long_count": 0.0,
            "h1_max_lifetime": 0.0,
            "h1_count": 0.0,
        }

    h1 = diagrams[1]
    lifetimes: list[float] = []
    for point in h1:
        birth = float(point.birth)
        death = float(point.death)
        if not np.isfinite(birth) or not np.isfinite(death):
            continue
        life = max(0.0, death - birth)
        lifetimes.append(life)

    if len(lifetimes) == 0:
        return {
            "h1_total_lifetime": 0.0,
            "h1_long_count": 0.0,
            "h1_max_lifetime": 0.0,
            "h1_count": 0.0,
        }

    life_arr = np.asarray(lifetimes, dtype=float)
    return {
        "h1_total_lifetime": float(np.sum(life_arr)),
        "h1_long_count": float(np.sum(life_arr >= 1.0)),
        "h1_max_lifetime": float(np.max(life_arr)),
        "h1_count": float(life_arr.size),
    }


def run_h44_true_zigzag() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    seed_tag = "seed42_main"
    for domain_index, (domain, run_map) in enumerate(SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[seed_tag]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = build_split_masks(edge_df)

        source_edges_raw = edge_df.loc[split_masks["source_disjoint"]].copy()
        target_edges_raw = edge_df.loc[split_masks["target_disjoint"]].copy()
        if source_edges_raw["label"].nunique() < 2 or target_edges_raw["label"].nunique() < 2:
            continue

        source_top = set(select_top_genes(source_edges_raw, gene_cap=H44_GENE_CAP))
        target_top = set(select_top_genes(target_edges_raw, gene_cap=H44_GENE_CAP))
        union_genes = np.array(sorted(source_top | target_top), dtype=int)
        if union_genes.size < 90:
            continue

        gene_to_union_local = {int(g): int(i) for i, g in enumerate(union_genes)}
        source_positions = np.array([gene_to_union_local[g] for g in sorted(source_top)], dtype=int)
        target_positions = np.array([gene_to_union_local[g] for g in sorted(target_top)], dtype=int)

        for layer in H44_LAYERS:
            if layer >= layer_embeddings.shape[0]:
                continue
            points_union = layer_embeddings[layer, union_genes, :]
            points_union_pca = reduce_points(
                points_union,
                n_components=20,
                random_state=20_700 + domain_index * 100 + layer,
            )

            source_edges = build_knn_edges_for_subset(
                points_union=points_union_pca,
                subset_positions=source_positions,
                n_neighbors=H44_KNN,
            )
            target_edges = build_knn_edges_for_subset(
                points_union=points_union_pca,
                subset_positions=target_positions,
                n_neighbors=H44_KNN,
            )
            if len(source_edges) < 30 or len(target_edges) < 30:
                continue

            obs_metrics = zigzag_h1_metrics(
                n_vertices=union_genes.size,
                source_edges=source_edges,
                target_edges=target_edges,
            )
            obs_total = float(obs_metrics["h1_total_lifetime"])

            rng = np.random.default_rng(20_800 + domain_index * 100 + layer)
            null_totals = np.empty(H44_NULL_PERM, dtype=float)
            for perm_idx in range(H44_NULL_PERM):
                target_perm = np.array(
                    sorted(rng.choice(union_genes.size, size=target_positions.size, replace=False)),
                    dtype=int,
                )
                target_edges_perm = build_knn_edges_for_subset(
                    points_union=points_union_pca,
                    subset_positions=target_perm,
                    n_neighbors=H44_KNN,
                )
                perm_metrics = zigzag_h1_metrics(
                    n_vertices=union_genes.size,
                    source_edges=source_edges,
                    target_edges=target_edges_perm,
                )
                null_total = float(perm_metrics["h1_total_lifetime"])
                null_totals[perm_idx] = null_total
                null_rows.append(
                    {
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": "paired_source_target",
                        "layer": int(layer),
                        "perm_idx": int(perm_idx),
                        "null_h1_total_lifetime": float(null_total),
                    }
                )

            p_upper = empirical_upper_tail_p(obs_total, null_totals)
            rows.append(
                {
                    "domain": domain,
                    "seed_tag": seed_tag,
                    "split_regime": "paired_source_target",
                    "layer": int(layer),
                    "n_union_genes": int(union_genes.size),
                    "n_source_genes": int(source_positions.size),
                    "n_target_genes": int(target_positions.size),
                    "n_source_edges": int(len(source_edges)),
                    "n_target_edges": int(len(target_edges)),
                    "h1_total_lifetime": float(obs_metrics["h1_total_lifetime"]),
                    "h1_long_count": float(obs_metrics["h1_long_count"]),
                    "h1_max_lifetime": float(obs_metrics["h1_max_lifetime"]),
                    "h1_count": float(obs_metrics["h1_count"]),
                    "null_mean_h1_total_lifetime": float(np.nanmean(null_totals)),
                    "delta_h1_total_vs_null_mean": float(obs_total - np.nanmean(null_totals)),
                    "p_h1_total_upper": float(p_upper),
                }
            )

    by_row_df = pd.DataFrame(rows).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h44_true_zigzag_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["domain", "seed_tag", "split_regime", "layer", "perm_idx"]
    )
    null_path = ITER_DIR / "h44_true_zigzag_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    domain_rows: list[dict[str, object]] = []
    for domain, group in by_row_df.groupby("domain", sort=True):
        domain_rows.append(
            {
                "domain": domain,
                "split_regime": "paired_source_target",
                "n_rows": int(group.shape[0]),
                "mean_h1_total_lifetime": float(group["h1_total_lifetime"].mean()),
                "mean_delta_h1_total_vs_null_mean": float(group["delta_h1_total_vs_null_mean"].mean()),
                "fraction_delta_positive": float((group["delta_h1_total_vs_null_mean"] > 0.0).mean()),
                "fraction_p_h1_total_lt_0_05": float((group["p_h1_total_upper"] < 0.05).mean()),
                "combined_fisher_p_h1_total": float(
                    safe_fisher_p(group["p_h1_total_upper"].to_numpy(dtype=float))
                ),
            }
        )
    domain_df = pd.DataFrame(domain_rows).sort_values(["domain", "split_regime"])
    domain_path = ITER_DIR / "h44_true_zigzag_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    summary = {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_h1_total_lifetime": float(by_row_df["h1_total_lifetime"].mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_positive_delta": int((domain_df["mean_delta_h1_total_vs_null_mean"] > 0.0).sum())
        if not domain_df.empty
        else 0,
        "domain_fisher_sig": int((domain_df["combined_fisher_p_h1_total"] < 0.05).sum())
        if not domain_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_layer_split": str(by_row_path),
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
        },
        "tooling": "true_zigzag_dionysus",
    }
    return summary


def compute_oos_deltas(
    x_mean_all: np.ndarray,
    x_full_all: np.ndarray,
    y_all: np.ndarray,
    seed_all: np.ndarray,
    layer_all: np.ndarray,
    evaluation: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if evaluation == "leave_layer_out":
        holdout_values = sorted(np.unique(layer_all).tolist())
        make_mask = lambda hold: (layer_all == hold)  # noqa: E731
    elif evaluation == "leave_seed_out":
        holdout_values = sorted(np.unique(seed_all).tolist())
        make_mask = lambda hold: (seed_all == hold)  # noqa: E731
    else:
        raise ValueError(f"Unsupported evaluation: {evaluation}")

    for hold in holdout_values:
        test_mask = make_mask(hold)
        train_mask = ~test_mask
        if np.sum(train_mask) < 10 or np.sum(test_mask) < 2:
            continue

        pred_mean = fit_linear_predict(
            x_train=x_mean_all[train_mask],
            y_train=y_all[train_mask],
            x_test=x_mean_all[test_mask],
        )
        pred_full = fit_linear_predict(
            x_train=x_full_all[train_mask],
            y_train=y_all[train_mask],
            x_test=x_full_all[test_mask],
        )

        y_train = y_all[train_mask]
        y_test = y_all[test_mask]
        row = {
            "evaluation": evaluation,
            "holdout_id": str(hold),
            "n_train": int(np.sum(train_mask)),
            "n_test": int(np.sum(test_mask)),
            "delta_r2_raw": float(
                r2_from_prediction(y_test, pred_full) - r2_from_prediction(y_test, pred_mean)
            ),
            "delta_r2_winsor": float(
                winsorized_r2(y_test, pred_full, y_train) - winsorized_r2(y_test, pred_mean, y_train)
            ),
            "delta_r2_trimmed": float(
                trimmed_r2(y_test, pred_full, y_train) - trimmed_r2(y_test, pred_mean, y_train)
            ),
        }
        rows.append(row)
    return rows


def block_sign_bootstrap_p(
    observed_mean: float,
    deltas: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float]:
    x = np.asarray(deltas, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")

    null_vals = np.empty(n_boot, dtype=float)
    ci_vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(x, size=x.size, replace=True)
        signs = rng.choice(np.array([-1.0, 1.0]), size=x.size, replace=True)
        null_vals[i] = float(np.mean(sample * signs))
        ci_vals[i] = float(np.mean(sample))

    p_upper = empirical_upper_tail_p(observed_mean, null_vals)
    ci_low = float(np.quantile(ci_vals, 0.025))
    ci_high = float(np.quantile(ci_vals, 0.975))
    return p_upper, ci_low, ci_high


def run_h45_id_oos_robust() -> dict[str, object]:
    df = pd.read_csv(H45_SOURCE_PATH)

    by_row: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

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

        for evaluation in ["leave_layer_out", "leave_seed_out"]:
            observed_rows = compute_oos_deltas(
                x_mean_all=x_mean_all,
                x_full_all=x_full_all,
                y_all=y_all,
                seed_all=seed_all,
                layer_all=layer_all,
                evaluation=evaluation,
            )
            if not observed_rows:
                continue

            for row in observed_rows:
                row.update(
                    {
                        "domain": domain,
                        "split_regime": split_regime,
                    }
                )
            by_row.extend(observed_rows)

            obs_df = pd.DataFrame(observed_rows)

            perm_metrics: dict[str, list[float]] = {
                "delta_r2_winsor": [],
                "delta_r2_trimmed": [],
            }
            rng = np.random.default_rng(21_100 + abs(hash((domain, split_regime, evaluation))) % 100_000)
            for perm_idx in range(H45_NULL_PERM):
                y_perm = rng.permutation(y_all)
                perm_rows = compute_oos_deltas(
                    x_mean_all=x_mean_all,
                    x_full_all=x_full_all,
                    y_all=y_perm,
                    seed_all=seed_all,
                    layer_all=layer_all,
                    evaluation=evaluation,
                )
                if not perm_rows:
                    continue
                perm_df = pd.DataFrame(perm_rows)
                for metric in ["delta_r2_winsor", "delta_r2_trimmed"]:
                    val = float(np.nanmean(perm_df[metric].to_numpy(dtype=float)))
                    perm_metrics[metric].append(val)
                    null_rows.append(
                        {
                            "domain": domain,
                            "split_regime": split_regime,
                            "evaluation": evaluation,
                            "metric": metric,
                            "null_family": "permutation",
                            "null_idx": int(perm_idx),
                            "null_value": float(val),
                        }
                    )

            for metric in ["delta_r2_winsor", "delta_r2_trimmed"]:
                observed_deltas = obs_df[metric].to_numpy(dtype=float)
                observed_mean = float(np.nanmean(observed_deltas))
                p_perm = empirical_upper_tail_p(observed_mean, np.asarray(perm_metrics[metric], dtype=float))

                rng_boot = np.random.default_rng(
                    21_700 + abs(hash((domain, split_regime, evaluation, metric))) % 100_000
                )
                p_block, ci_low, ci_high = block_sign_bootstrap_p(
                    observed_mean=observed_mean,
                    deltas=observed_deltas,
                    rng=rng_boot,
                    n_boot=H45_BOOTSTRAP_NULL,
                )

                for boot_idx in range(H45_BOOTSTRAP_NULL):
                    sample = rng_boot.choice(observed_deltas, size=observed_deltas.size, replace=True)
                    signs = rng_boot.choice(np.array([-1.0, 1.0]), size=observed_deltas.size, replace=True)
                    null_val = float(np.mean(sample * signs))
                    null_rows.append(
                        {
                            "domain": domain,
                            "split_regime": split_regime,
                            "evaluation": evaluation,
                            "metric": metric,
                            "null_family": "block_sign_bootstrap",
                            "null_idx": int(boot_idx),
                            "null_value": float(null_val),
                        }
                    )

                summary_rows.append(
                    {
                        "domain": domain,
                        "split_regime": split_regime,
                        "evaluation": evaluation,
                        "metric": metric,
                        "observed_mean_delta": float(observed_mean),
                        "fraction_holdout_delta_positive": float((observed_deltas > 0.0).mean()),
                        "p_perm_upper": float(p_perm),
                        "p_block_sign_upper": float(p_block),
                        "bootstrap_ci_low": float(ci_low),
                        "bootstrap_ci_high": float(ci_high),
                    }
                )

    by_row_df = pd.DataFrame(by_row).sort_values(
        ["domain", "split_regime", "evaluation", "holdout_id"]
    )
    by_row_path = ITER_DIR / "h45_id_oos_robust_by_seed_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["domain", "split_regime", "evaluation", "metric"]
    )
    summary_path = ITER_DIR / "h45_id_oos_robust_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["domain", "split_regime", "evaluation", "metric", "null_family", "null_idx"]
    )
    null_path = ITER_DIR / "h45_id_oos_robust_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    winsor_df = summary_df.loc[summary_df["metric"] == "delta_r2_winsor"].copy()
    trimmed_df = summary_df.loc[summary_df["metric"] == "delta_r2_trimmed"].copy()

    summary = {
        "rows_tested": int(summary_df.shape[0]),
        "winsor_mean_observed_delta": float(winsor_df["observed_mean_delta"].mean())
        if not winsor_df.empty
        else float("nan"),
        "trimmed_mean_observed_delta": float(trimmed_df["observed_mean_delta"].mean())
        if not trimmed_df.empty
        else float("nan"),
        "winsor_rows_p_perm_lt_0_05": int((winsor_df["p_perm_upper"] < 0.05).sum())
        if not winsor_df.empty
        else 0,
        "trimmed_rows_p_perm_lt_0_05": int((trimmed_df["p_perm_upper"] < 0.05).sum())
        if not trimmed_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
        "source_h38_artifact": str(H45_SOURCE_PATH),
    }
    return summary


def main() -> None:
    required_paths = [
        TRRUST_PATH,
        DOROTHEA_PATH,
        GENE2GO_PATH,
        OMNIPATH_INTERACTIONS_PATH,
        H45_SOURCE_PATH,
    ]
    for domain, run_map in SCGPT_RUNS_BY_DOMAIN.items():
        required_paths.append(PROCESSED_H5AD_BY_DOMAIN[domain])
        for run_dir in run_map.values():
            required_paths.append(run_dir / "cycle1_edge_dataset.tsv")
            required_paths.append(run_dir / "layer_gene_embeddings.npy")

    missing = [str(p) for p in required_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    dorothea_score_map = load_dorothea_score_map()
    trrust_pairs = load_trrust_pairs()
    omnipath_pairs = load_omnipath_pairs()
    gene2go_upper = load_gene2go_upper()

    coexp_cache = {
        domain: CoexpressionDomainCache(path)
        for domain, path in PROCESSED_H5AD_BY_DOMAIN.items()
    }
    try:
        h43_summary = run_h43_support_interaction_ontology(
            coexp_cache=coexp_cache,
            dorothea_score_map=dorothea_score_map,
            trrust_pairs=trrust_pairs,
            omnipath_pairs=omnipath_pairs,
            gene2go_upper=gene2go_upper,
        )
    finally:
        for cache in coexp_cache.values():
            cache.close()

    h44_summary = run_h44_true_zigzag()
    h45_summary = run_h45_id_oos_robust()

    summary = {
        "iteration": "iter_0020",
        "h43": h43_summary,
        "h44": h44_summary,
        "h45": h45_summary,
        "environment_note": {
            "dionysus_installed": True,
            "dionysus_install_command": "conda run -n subproject40-topology python -m pip install dionysus",
        },
    }
    summary_path = ITER_DIR / "iter0020_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
