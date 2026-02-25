from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import combine_pvalues, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import ot
from transformers import AutoModel


ITER_DIR = Path("iterations/iter_0014")
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

# H25 screening grid and controls
H25_LAYERS = [0, 3, 7, 11]
H25_DIFFUSION_TIMES = [1, 2, 4, 8]
H25_GENE_CAP = 320
H25_NULL_PERM = 180

# H26 calibration settings
H26_GENE_CAP = 420
H26_NULL_PERM = 180
H26_BOOTSTRAP = 180

# H27 correspondence-free alignment settings
H27_GENE_CAP = 280
H27_NULL_PERM = 180


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
    if labels.size == 0:
        return float("nan")
    if np.unique(labels).size < 2:
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


def zscore(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if sd <= 1e-12:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def row_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def get_knn_indices(points: np.ndarray, n_neighbors: int) -> np.ndarray:
    n_points = points.shape[0]
    k = max(1, min(n_neighbors, n_points - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(points)
    _, indices = nbrs.kneighbors(points)
    return indices[:, 1:]


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
    """
    Build a weighted kNN graph and ensure connectivity by adding minimum-distance
    bridges across disconnected components.

    Returning an explicit dense distance matrix keeps downstream shortest-path and
    diffusion code simple and reproducible for this bounded gene count regime.
    """
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

    # If kNN leaves multiple components, add the minimum-distance cross-component
    # edges iteratively until the graph is connected.
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
        # Defensive fallback: if connectivity repair was still insufficient for a
        # rare row, use dense Euclidean distances to keep the screening packet running.
        geodesic = cdist(points, points, metric="euclidean")
        used_component_bridging = True

    # Build a row-stochastic diffusion transition matrix from Gaussian edge weights.
    finite_edges = knn_dist[np.isfinite(knn_dist) & (knn_dist > 0)]
    sigma = float(np.median(finite_edges)) if finite_edges.size else 1.0
    sigma = max(sigma, 1e-6)

    weights = np.zeros_like(knn_dist)
    edge_mask = np.isfinite(knn_dist) & (knn_dist > 0)
    weights[edge_mask] = np.exp(-np.square(knn_dist[edge_mask]) / (2.0 * sigma * sigma))
    zero_row_mask = np.sum(weights, axis=1) <= 1e-12
    if np.any(zero_row_mask):
        # Add identity self-loops so diffusion powers remain numerically well-defined.
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


def load_dorothea_and_trrust() -> tuple[dict[tuple[str, str], int], set[tuple[str, str]], set[str]]:
    dorothea = pd.read_csv(DOROTHEA_PATH, sep="\t")
    dorothea["source"] = dorothea["source"].astype(str).str.upper()
    dorothea["target"] = dorothea["target"].astype(str).str.upper()
    confidence_map = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
    dorothea["confidence_score"] = (
        dorothea["confidence"].astype(str).str.upper().map(confidence_map).fillna(-1).astype(int)
    )
    best = dorothea.groupby(["source", "target"], as_index=False)["confidence_score"].max()
    score_map = {
        (str(row.source), str(row.target)): int(row.confidence_score)
        for row in best.itertuples(index=False)
    }
    tf_set = set(best["source"].astype(str).unique())

    trrust = pd.read_csv(
        TRRUST_PATH,
        sep="\t",
        header=None,
        names=["source", "target", "regulation", "pmid"],
    )
    trrust["source"] = trrust["source"].astype(str).str.upper()
    trrust["target"] = trrust["target"].astype(str).str.upper()
    trrust_edges = set(zip(trrust["source"], trrust["target"]))
    return score_map, trrust_edges, tf_set


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


def run_h25_diffusion_distance_sweep() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_index, (seed_tag, run_dir) in enumerate(run_map.items()):
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            n_layers = layer_embeddings.shape[0]
            split_masks = build_split_masks(edge_df)

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                top_genes = select_top_genes(split_edges, gene_cap=H25_GENE_CAP)
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

                for layer in H25_LAYERS:
                    if layer >= n_layers:
                        continue

                    random_state = (
                        14_250
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
                    for t in H25_DIFFUSION_TIMES:
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
                    delta_best_vs_base = float(best_auc - base_auc)

                    rng = np.random.default_rng(
                        14_260
                        + domain_index * 10_000
                        + seed_index * 1_000
                        + split_index * 100
                        + layer
                    )
                    null_delta = np.empty(H25_NULL_PERM, dtype=float)
                    for perm_idx in range(H25_NULL_PERM):
                        labels_perm = rng.permutation(labels)
                        auc_e_perm = safe_auc(labels_perm, euclidean_scores)
                        auc_g_perm = safe_auc(labels_perm, geodesic_scores)
                        best_d_perm = max(
                            safe_auc(labels_perm, diffusion_scores_by_t[t])
                            for t in H25_DIFFUSION_TIMES
                        )
                        base_perm = max(auc_e_perm, auc_g_perm)
                        null_delta[perm_idx] = float(best_d_perm - base_perm)
                        null_rows.append(
                            {
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_best_diffusion_minus_best_baseline": float(null_delta[perm_idx]),
                            }
                        )

                    p_delta_upper = empirical_upper_tail_p(delta_best_vs_base, null_delta)

                    for t in H25_DIFFUSION_TIMES:
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
                                "diffusion_t": int(t),
                                "auc_diffusion": float(diffusion_auc_by_t[t]),
                                "auc_euclidean": float(auc_euclidean),
                                "auc_geodesic": float(auc_geodesic),
                                "best_diffusion_t": int(best_t),
                                "best_diffusion_auc": float(best_auc),
                                "best_baseline_auc": float(base_auc),
                                "delta_best_diffusion_minus_best_baseline": float(delta_best_vs_base),
                                "p_delta_best_diffusion_upper": float(p_delta_upper),
                                "null_mean_delta": float(np.mean(null_delta)),
                                "null_std_delta": float(np.std(null_delta, ddof=1))
                                if null_delta.size > 1
                                else 0.0,
                                "is_best_t": bool(t == best_t),
                            }
                        )

    row_df = pd.DataFrame(rows).sort_values(
        ["domain", "seed_tag", "split_regime", "layer", "diffusion_t"]
    )
    row_path = ITER_DIR / "h25_diffusion_distance_by_seed_layer_split.csv"
    row_df.to_csv(row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["domain", "seed_tag", "split_regime", "layer", "perm_idx"]
    )
    null_path = ITER_DIR / "h25_diffusion_distance_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    best_df = row_df.loc[row_df["is_best_t"]].copy()
    domain_rows: list[dict[str, object]] = []
    for domain, group in best_df.groupby("domain", sort=True):
        pvals = group["p_delta_best_diffusion_upper"].to_numpy(dtype=float)
        positive_fraction = float((group["delta_best_diffusion_minus_best_baseline"] > 0).mean())
        sig_fraction = float((group["p_delta_best_diffusion_upper"] < 0.05).mean())
        mean_best_delta = float(group["delta_best_diffusion_minus_best_baseline"].mean())
        domain_rows.append(
            {
                "domain": domain,
                "n_rows": int(group.shape[0]),
                "mean_best_diffusion_auc": float(group["best_diffusion_auc"].mean()),
                "mean_best_baseline_auc": float(group["best_baseline_auc"].mean()),
                "mean_delta_best_diffusion_minus_best_baseline": mean_best_delta,
                "fraction_positive_delta": positive_fraction,
                "fraction_p_lt_0_05": sig_fraction,
                "combined_fisher_p_delta_upper": safe_fisher_p(pvals),
                "median_best_diffusion_t": int(np.median(group["best_diffusion_t"].to_numpy(dtype=int))),
            }
        )

    domain_summary_df = pd.DataFrame(domain_rows).sort_values("domain")
    domain_path = ITER_DIR / "h25_diffusion_distance_domain_summary.csv"
    domain_summary_df.to_csv(domain_path, index=False)

    summary = {
        "n_rows": int(row_df.shape[0]),
        "n_best_rows": int(best_df.shape[0]),
        "domains_tested": int(domain_summary_df.shape[0]),
        "domains_with_positive_mean_delta": int(
            (domain_summary_df["mean_delta_best_diffusion_minus_best_baseline"] > 0).sum()
        ),
        "domains_with_sig_combined_fisher": int(
            (domain_summary_df["combined_fisher_p_delta_upper"] < 0.05).sum()
        ),
        "artifact_paths": {
            "by_seed_layer_split": str(row_path),
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
        },
    }
    return summary


def degree_strata(values: np.ndarray, max_bins: int = 5) -> np.ndarray:
    series = pd.Series(values.astype(float))
    n_unique = int(series.nunique(dropna=True))
    q = max(1, min(max_bins, n_unique))
    if q == 1:
        return np.zeros(values.size, dtype=int)
    strata = pd.qcut(series, q=q, labels=False, duplicates="drop")
    return strata.fillna(0).astype(int).to_numpy(dtype=int)


def fit_logistic_auc(
    labels: np.ndarray,
    geom_score: np.ndarray,
    prior_support_count: np.ndarray,
    source_degree: np.ndarray,
    target_degree: np.ndarray,
) -> tuple[float, float, float]:
    y = labels.astype(int)
    geom = zscore(geom_score)
    prior = zscore(prior_support_count)
    src_deg = zscore(source_degree)
    tgt_deg = zscore(target_degree)
    interaction = geom * prior

    x_base = np.column_stack([geom, prior, src_deg, tgt_deg])
    x_full = np.column_stack([geom, prior, src_deg, tgt_deg, interaction])

    scaler_base = StandardScaler()
    scaler_full = StandardScaler()
    xb = scaler_base.fit_transform(x_base)
    xf = scaler_full.fit_transform(x_full)

    model_base = LogisticRegression(max_iter=1000, solver="lbfgs")
    model_full = LogisticRegression(max_iter=1000, solver="lbfgs")

    model_base.fit(xb, y)
    model_full.fit(xf, y)

    pred_base = model_base.predict_proba(xb)[:, 1]
    pred_full = model_full.predict_proba(xf)[:, 1]
    auc_base = safe_auc(y, pred_base)
    auc_full = safe_auc(y, pred_full)

    interaction_coef = float(model_full.coef_[0, -1])
    return float(auc_base), float(auc_full), interaction_coef


def run_h26_biological_anchor(
    best_t_by_domain: dict[str, int],
) -> dict[str, object]:
    dorothea_score_map, trrust_edges, tf_sources = load_dorothea_and_trrust()
    gene2go_upper = load_gene2go_upper()

    edge_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, domain in enumerate(["immune", "lung", "external_lung"]):
        run_dir = SCGPT_RUNS_BY_DOMAIN[domain]["seed42_main"]
        layer = int(CROSS_MODEL_LAYER_BY_DOMAIN[domain])
        diffusion_t = int(best_t_by_domain.get(domain, 2))

        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        split_masks = build_split_masks(edge_df)
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = select_top_genes(split_edges, gene_cap=H26_GENE_CAP)
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

            points = layer_embeddings[layer, edge_gene_indices, :]
            points_pca = reduce_points(
                points,
                n_components=24,
                random_state=14_300 + domain_index * 100 + split_index,
            )
            geodesic, transition, _, _ = geodesic_and_transition(points_pca, n_neighbors=24)
            euclidean = cdist(points_pca, points_pca, metric="euclidean")
            diffusion_distance = diffusion_distance_scores(
                transition,
                source_local=source_local,
                target_local=target_local,
                t=diffusion_t,
            )

            diffusion_score = -diffusion_distance
            geodesic_score = -geodesic[source_local, target_local]
            euclidean_score = -euclidean[source_local, target_local]

            source_degree_map = split_edges["source_idx"].value_counts().to_dict()
            target_degree_map = split_edges["target_idx"].value_counts().to_dict()
            source_degree = split_edges["source_idx"].map(source_degree_map).astype(float).to_numpy()
            target_degree = split_edges["target_idx"].map(target_degree_map).astype(float).to_numpy()

            prior_support = np.zeros(labels.size, dtype=int)
            dorothea_support = np.zeros(labels.size, dtype=int)
            go_shared = np.zeros(labels.size, dtype=int)
            tf_source = np.zeros(labels.size, dtype=int)
            trrust_support = np.zeros(labels.size, dtype=int)

            for row_idx, row in enumerate(split_edges.itertuples(index=False)):
                src = str(row.source).upper()
                tgt = str(row.target).upper()

                d_score = int(dorothea_score_map.get((src, tgt), -1))
                d_support = int(d_score >= 2)
                dorothea_support[row_idx] = d_support

                src_go = gene2go_upper.get(src, set())
                tgt_go = gene2go_upper.get(tgt, set())
                shared = int(len(src_go & tgt_go) > 0)
                go_shared[row_idx] = shared

                src_tf = int(src in tf_sources)
                tf_source[row_idx] = src_tf

                t_support = int((src, tgt) in trrust_edges)
                trrust_support[row_idx] = t_support

                # TRRUST support is recorded for bookkeeping only and intentionally
                # excluded from the prior-support model terms because this edge label
                # set is TRRUST-defined.
                prior_support[row_idx] = d_support + shared + src_tf

            auc_base, auc_full, interaction_coef = fit_logistic_auc(
                labels=labels,
                geom_score=diffusion_score,
                prior_support_count=prior_support,
                source_degree=source_degree,
                target_degree=target_degree,
            )
            auc_delta = float(auc_full - auc_base)

            mean_degree = 0.5 * (source_degree + target_degree)
            strata = degree_strata(mean_degree, max_bins=5)

            rng = np.random.default_rng(14_320 + domain_index * 100 + split_index)
            null_interaction = np.empty(H26_NULL_PERM, dtype=float)
            null_auc_delta = np.empty(H26_NULL_PERM, dtype=float)

            for perm_idx in range(H26_NULL_PERM):
                perm_prior = prior_support.copy()
                for s in np.unique(strata):
                    loc = np.where(strata == s)[0]
                    if loc.size > 1:
                        perm_prior[loc] = perm_prior[rng.permutation(loc)]

                auc_b_perm, auc_f_perm, coef_perm = fit_logistic_auc(
                    labels=labels,
                    geom_score=diffusion_score,
                    prior_support_count=perm_prior,
                    source_degree=source_degree,
                    target_degree=target_degree,
                )
                null_interaction[perm_idx] = float(coef_perm)
                null_auc_delta[perm_idx] = float(auc_f_perm - auc_b_perm)
                null_rows.append(
                    {
                        "domain": domain,
                        "split_regime": split_regime,
                        "perm_idx": int(perm_idx),
                        "null_interaction_coef": float(null_interaction[perm_idx]),
                        "null_auc_delta": float(null_auc_delta[perm_idx]),
                    }
                )

            boot_interaction = np.empty(H26_BOOTSTRAP, dtype=float)
            boot_auc_delta = np.empty(H26_BOOTSTRAP, dtype=float)
            for boot_idx in range(H26_BOOTSTRAP):
                sampled_idx: list[int] = []
                for s in np.unique(strata):
                    loc = np.where(strata == s)[0]
                    if loc.size == 0:
                        continue
                    sampled_idx.extend(rng.choice(loc, size=loc.size, replace=True).tolist())
                sampled = np.asarray(sampled_idx, dtype=int)

                auc_b_boot, auc_f_boot, coef_boot = fit_logistic_auc(
                    labels=labels[sampled],
                    geom_score=diffusion_score[sampled],
                    prior_support_count=prior_support[sampled],
                    source_degree=source_degree[sampled],
                    target_degree=target_degree[sampled],
                )
                boot_interaction[boot_idx] = float(coef_boot)
                boot_auc_delta[boot_idx] = float(auc_f_boot - auc_b_boot)

            p_interaction_upper = empirical_upper_tail_p(interaction_coef, null_interaction)
            p_auc_delta_upper = empirical_upper_tail_p(auc_delta, null_auc_delta)

            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "layer": int(layer),
                    "diffusion_t": int(diffusion_t),
                    "n_edges": int(labels.size),
                    "n_positive": int(labels.sum()),
                    "positive_rate": float(labels.mean()),
                    "mean_prior_support_count": float(np.mean(prior_support)),
                    "mean_dorothea_support": float(np.mean(dorothea_support)),
                    "mean_go_shared": float(np.mean(go_shared)),
                    "mean_source_is_tf": float(np.mean(tf_source)),
                    "mean_trrust_support_bookkeeping": float(np.mean(trrust_support)),
                    "auc_baseline": float(auc_base),
                    "auc_full": float(auc_full),
                    "auc_delta_full_minus_baseline": float(auc_delta),
                    "interaction_coef": float(interaction_coef),
                    "p_interaction_upper": float(p_interaction_upper),
                    "p_auc_delta_upper": float(p_auc_delta_upper),
                    "ci95_interaction_low": float(np.quantile(boot_interaction, 0.025)),
                    "ci95_interaction_high": float(np.quantile(boot_interaction, 0.975)),
                    "ci95_auc_delta_low": float(np.quantile(boot_auc_delta, 0.025)),
                    "ci95_auc_delta_high": float(np.quantile(boot_auc_delta, 0.975)),
                }
            )

            for row_idx, row in enumerate(split_edges.itertuples(index=False)):
                edge_rows.append(
                    {
                        "domain": domain,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "diffusion_t": int(diffusion_t),
                        "source": str(row.source),
                        "target": str(row.target),
                        "source_idx": int(row.source_idx),
                        "target_idx": int(row.target_idx),
                        "label": int(row.label),
                        "diffusion_score": float(diffusion_score[row_idx]),
                        "geodesic_score": float(geodesic_score[row_idx]),
                        "euclidean_score": float(euclidean_score[row_idx]),
                        "prior_support_count": int(prior_support[row_idx]),
                        "dorothea_support": int(dorothea_support[row_idx]),
                        "go_shared": int(go_shared[row_idx]),
                        "source_is_tf": int(tf_source[row_idx]),
                        "trrust_support_bookkeeping": int(trrust_support[row_idx]),
                        "source_degree": float(source_degree[row_idx]),
                        "target_degree": float(target_degree[row_idx]),
                        "mean_degree": float(mean_degree[row_idx]),
                    }
                )

    edge_df = pd.DataFrame(edge_rows).sort_values(["domain", "split_regime", "source", "target"])
    edge_path = ITER_DIR / "h26_bio_anchor_edge_table.csv"
    edge_df.to_csv(edge_path, index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h26_bio_anchor_model_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(["domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h26_bio_anchor_permutation_null.csv"
    null_df.to_csv(null_path, index=False)

    summary = {
        "n_edges": int(edge_df.shape[0]),
        "n_domain_split_rows": int(summary_df.shape[0]),
        "rows_with_positive_interaction": int((summary_df["interaction_coef"] > 0).sum()),
        "rows_with_p_interaction_lt_0_05": int((summary_df["p_interaction_upper"] < 0.05).sum()),
        "rows_with_positive_auc_delta": int(
            (summary_df["auc_delta_full_minus_baseline"] > 0).sum()
        ),
        "rows_with_p_auc_delta_lt_0_05": int((summary_df["p_auc_delta_upper"] < 0.05).sum()),
        "artifact_paths": {
            "edge_table": str(edge_path),
            "model_summary": str(summary_path),
            "permutation_null": str(null_path),
        },
    }
    return summary


def load_geneformer_embeddings() -> np.ndarray:
    model = AutoModel.from_pretrained("ctheodoris/Geneformer")
    embeddings = model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float64, copy=False)
    del model
    return embeddings


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


def run_h27_unseeded_gw_alignment() -> dict[str, object]:
    geneformer_embeddings = load_geneformer_embeddings()

    domain_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []

    for domain_index, domain in enumerate(["immune", "lung", "external_lung"]):
        edge_df = pd.read_csv(GENEFORMER_EDGE_BY_DOMAIN[domain], sep="\t")
        mapping_df = select_mapped_genes(
            edge_df,
            gene_cap=H27_GENE_CAP,
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

        n_pca = min(32, n_items - 1, x_raw.shape[1], y_raw.shape[1])
        if n_pca < 8:
            continue

        x = reduce_points(x_raw, n_components=n_pca, random_state=14_410 + domain_index * 10)
        y = reduce_points(y_raw, n_components=n_pca, random_state=14_411 + domain_index * 10)

        c1 = cdist(x, x, metric="euclidean")
        c2 = cdist(y, y, metric="euclidean")
        # Scale costs to comparable magnitude for numerical stability.
        c1_scale = float(np.median(c1[c1 > 0])) if np.any(c1 > 0) else 1.0
        c2_scale = float(np.median(c2[c2 > 0])) if np.any(c2 > 0) else 1.0
        c1 = c1 / max(c1_scale, 1e-6)
        c2 = c2 / max(c2_scale, 1e-6)
        p = np.ones(n_items, dtype=np.float64) / float(n_items)
        q = np.ones(n_items, dtype=np.float64) / float(n_items)

        coupling = None
        gw_log: dict[str, float] = {}
        try:
            coupling, gw_log = ot.gromov.entropic_gromov_wasserstein(
                c1,
                c2,
                p,
                q,
                loss_fun="square_loss",
                epsilon=0.5,
                max_iter=400,
                tol=1e-9,
                log=True,
                verbose=False,
            )
        except Exception:
            coupling = None

        if coupling is None or (not np.isfinite(coupling).all()) or float(np.sum(coupling)) <= 1e-12:
            try:
                coupling, gw_log = ot.gromov.gromov_wasserstein(
                    c1,
                    c2,
                    p,
                    q,
                    loss_fun="square_loss",
                    max_iter=600,
                    tol_rel=1e-9,
                    tol_abs=1e-9,
                    log=True,
                    armijo=True,
                )
            except Exception:
                coupling = None

        if coupling is None or (not np.isfinite(coupling).all()) or float(np.sum(coupling)) <= 1e-12:
            # If both solvers fail, retain a valid coupling for diagnostics and classify via metrics.
            coupling = np.outer(p, q)
            gw_log = {"gw_dist": float("nan")}

        hard_map = np.argmax(coupling, axis=1).astype(int)
        reverse_map = np.argmax(coupling, axis=0).astype(int)

        top1 = float(np.mean(hard_map == np.arange(n_items)))
        mutual_top1 = float(
            np.mean((hard_map == np.arange(n_items)) & (reverse_map == np.arange(n_items)))
        )
        truth_mass = float(np.trace(coupling))

        y_gw = y[hard_map]
        dist_rho = safe_spearman(pdist(x), pdist(y_gw))
        jaccard = mean_knn_jaccard(x, y_gw, n_neighbors=min(10, n_items - 1))

        edge_sub = edge_df.loc[
            edge_df["source_idx"].isin(set(gene_ids)) & edge_df["target_idx"].isin(set(gene_ids))
        ].copy()
        gene_to_pos = {int(g): int(i) for i, g in enumerate(gene_ids)}
        source_local = edge_sub["source_idx"].map(gene_to_pos).to_numpy(dtype=int)
        target_local = edge_sub["target_idx"].map(gene_to_pos).to_numpy(dtype=int)
        labels = edge_sub["label"].to_numpy(dtype=int)

        auc_transfer_gw = transfer_auc_from_mapping(
            hard_map,
            source_local=source_local,
            target_local=target_local,
            labels=labels,
            y_points=y,
        )

        identity_map = np.arange(n_items, dtype=int)
        auc_transfer_identity = transfer_auc_from_mapping(
            identity_map,
            source_local=source_local,
            target_local=target_local,
            labels=labels,
            y_points=y,
        )
        dist_rho_identity = safe_spearman(pdist(x), pdist(y))
        jaccard_identity = mean_knn_jaccard(x, y, n_neighbors=min(10, n_items - 1))

        rng = np.random.default_rng(14_430 + domain_index * 10)
        null_top1 = np.empty(H27_NULL_PERM, dtype=float)
        null_mutual = np.empty(H27_NULL_PERM, dtype=float)
        null_dist = np.empty(H27_NULL_PERM, dtype=float)
        null_jaccard = np.empty(H27_NULL_PERM, dtype=float)
        null_auc = np.empty(H27_NULL_PERM, dtype=float)

        for perm_idx in range(H27_NULL_PERM):
            perm = rng.permutation(n_items)
            null_top1[perm_idx] = float(np.mean(perm == np.arange(n_items)))
            null_mutual[perm_idx] = float(np.mean(perm == np.arange(n_items)))

            y_perm = y[perm]
            null_dist[perm_idx] = safe_spearman(pdist(x), pdist(y_perm))
            null_jaccard[perm_idx] = mean_knn_jaccard(
                x,
                y_perm,
                n_neighbors=min(10, n_items - 1),
            )
            null_auc[perm_idx] = transfer_auc_from_mapping(
                perm,
                source_local=source_local,
                target_local=target_local,
                labels=labels,
                y_points=y,
            )

            null_rows.append(
                {
                    "domain": domain,
                    "perm_idx": int(perm_idx),
                    "null_top1_retrieval": float(null_top1[perm_idx]),
                    "null_mutual_top1": float(null_mutual[perm_idx]),
                    "null_distance_spearman": float(null_dist[perm_idx]),
                    "null_knn_jaccard": float(null_jaccard[perm_idx]),
                    "null_edge_transfer_auc": float(null_auc[perm_idx]),
                }
            )

        p_top1 = empirical_upper_tail_p(top1, null_top1)
        p_mutual = empirical_upper_tail_p(mutual_top1, null_mutual)
        p_dist = empirical_upper_tail_p(dist_rho, null_dist)
        p_jaccard = empirical_upper_tail_p(jaccard, null_jaccard)
        p_auc = empirical_upper_tail_p(auc_transfer_gw, null_auc)

        domain_rows.append(
            {
                "domain": domain,
                "layer": int(layer),
                "n_matched_genes": int(n_items),
                "pca_dim": int(n_pca),
                "gw_objective": float(gw_log.get("gw_dist", float("nan"))),
                "top1_retrieval_gw": float(top1),
                "mutual_top1_gw": float(mutual_top1),
                "truth_mass_gw": float(truth_mass),
                "distance_spearman_gw": float(dist_rho),
                "knn_jaccard_gw": float(jaccard),
                "edge_transfer_auc_gw": float(auc_transfer_gw),
                "distance_spearman_identity": float(dist_rho_identity),
                "knn_jaccard_identity": float(jaccard_identity),
                "edge_transfer_auc_identity": float(auc_transfer_identity),
                "null_mean_top1": float(np.mean(null_top1)),
                "null_mean_mutual_top1": float(np.mean(null_mutual)),
                "null_mean_distance_spearman": float(np.mean(null_dist)),
                "null_mean_knn_jaccard": float(np.mean(null_jaccard)),
                "null_mean_edge_transfer_auc": float(np.mean(null_auc)),
                "p_top1_retrieval_upper": float(p_top1),
                "p_mutual_top1_upper": float(p_mutual),
                "p_distance_spearman_upper": float(p_dist),
                "p_knn_jaccard_upper": float(p_jaccard),
                "p_edge_transfer_auc_upper": float(p_auc),
            }
        )

        map_rows.append(
            {
                "domain": domain,
                "n_matched_genes": int(n_items),
                "top1_retrieval_gw": float(top1),
                "mutual_top1_gw": float(mutual_top1),
                "truth_mass_gw": float(truth_mass),
                "p_top1_retrieval_upper": float(p_top1),
                "p_mutual_top1_upper": float(p_mutual),
            }
        )

    domain_df = pd.DataFrame(domain_rows).sort_values("domain")
    domain_path = ITER_DIR / "h27_gw_alignment_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(["domain", "perm_idx"])
    null_path = ITER_DIR / "h27_gw_alignment_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    map_df = pd.DataFrame(map_rows).sort_values("domain")
    map_path = ITER_DIR / "h27_gw_alignment_map_quality.csv"
    map_df.to_csv(map_path, index=False)

    summary = {
        "domains_tested": int(domain_df.shape[0]),
        "domains_with_sig_top1": int((domain_df["p_top1_retrieval_upper"] < 0.05).sum())
        if not domain_df.empty
        else 0,
        "domains_with_sig_distance": int((domain_df["p_distance_spearman_upper"] < 0.05).sum())
        if not domain_df.empty
        else 0,
        "domains_with_sig_transfer_auc": int((domain_df["p_edge_transfer_auc_upper"] < 0.05).sum())
        if not domain_df.empty
        else 0,
        "mean_top1_retrieval_gw": float(domain_df["top1_retrieval_gw"].mean())
        if not domain_df.empty
        else float("nan"),
        "mean_distance_spearman_gw": float(domain_df["distance_spearman_gw"].mean())
        if not domain_df.empty
        else float("nan"),
        "mean_edge_transfer_auc_gw": float(domain_df["edge_transfer_auc_gw"].mean())
        if not domain_df.empty
        else float("nan"),
        "combined_fisher_top1_p": safe_fisher_p(
            domain_df["p_top1_retrieval_upper"].to_numpy(dtype=float)
        )
        if not domain_df.empty
        else float("nan"),
        "combined_fisher_distance_p": safe_fisher_p(
            domain_df["p_distance_spearman_upper"].to_numpy(dtype=float)
        )
        if not domain_df.empty
        else float("nan"),
        "combined_fisher_transfer_auc_p": safe_fisher_p(
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


def main() -> None:
    required_paths: list[Path] = []
    for run_map in SCGPT_RUNS_BY_DOMAIN.values():
        for run_dir in run_map.values():
            required_paths.append(run_dir / "layer_gene_embeddings.npy")
            required_paths.append(run_dir / "cycle1_edge_dataset.tsv")
    required_paths.extend(GENEFORMER_EDGE_BY_DOMAIN.values())
    required_paths.extend([TRRUST_PATH, DOROTHEA_PATH, GENE2GO_PATH])

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    h25_summary = run_h25_diffusion_distance_sweep()

    h25_domain = pd.read_csv(ITER_DIR / "h25_diffusion_distance_domain_summary.csv")
    best_t_by_domain = {
        str(row.domain): int(row.median_best_diffusion_t)
        for row in h25_domain.itertuples(index=False)
    }

    h26_summary = run_h26_biological_anchor(best_t_by_domain=best_t_by_domain)
    h27_summary = run_h27_unseeded_gw_alignment()

    iteration_summary = {
        "iteration": "iter_0014",
        "inputs": {
            "scgpt_runs_by_domain": {
                domain: {seed: str(path) for seed, path in run_dict.items()}
                for domain, run_dict in SCGPT_RUNS_BY_DOMAIN.items()
            },
            "geneformer_edge_by_domain": {
                domain: str(path) for domain, path in GENEFORMER_EDGE_BY_DOMAIN.items()
            },
            "cross_model_layer_by_domain": CROSS_MODEL_LAYER_BY_DOMAIN,
            "trrust_path": str(TRRUST_PATH),
            "dorothea_path": str(DOROTHEA_PATH),
            "gene2go_path": str(GENE2GO_PATH),
            "note": "TRRUST support is recorded for bookkeeping in H26 but excluded from interaction model terms because labels are TRRUST-defined.",
        },
        "h25_diffusion_distance_sweep": h25_summary,
        "h26_biological_anchor_interaction": h26_summary,
        "h27_unseeded_gw_alignment": h27_summary,
    }

    summary_path = ITER_DIR / "iter0014_screen_summary.json"
    summary_path.write_text(json.dumps(iteration_summary, indent=2))
    print(json.dumps(iteration_summary, indent=2))


if __name__ == "__main__":
    main()
