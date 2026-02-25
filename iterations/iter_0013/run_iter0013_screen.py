from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import combine_pvalues, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


ITER_DIR = Path("iterations/iter_0013")
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


def empirical_upper_tail_p(observed: float, null_values: np.ndarray) -> float:
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float((1 + np.sum(values >= observed)) / (values.size + 1))


def empirical_lower_tail_p(observed: float, null_values: np.ndarray) -> float:
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float((1 + np.sum(values <= observed)) / (values.size + 1))


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


def build_split_masks(edge_df: pd.DataFrame) -> dict[str, np.ndarray]:
    source_threshold = float(edge_df["source_idx"].median())
    target_threshold = float(edge_df["target_idx"].median())
    return {
        "source_disjoint": edge_df["source_idx"].to_numpy(dtype=float) <= source_threshold,
        "target_disjoint": edge_df["target_idx"].to_numpy(dtype=float) > target_threshold,
    }


def phase_from_layer(layer: int, n_layers: int) -> str:
    first_cut = max(1, n_layers // 3)
    second_cut = max(first_cut + 1, (2 * n_layers) // 3)
    if layer < first_cut:
        return "early"
    if layer < second_cut:
        return "mid"
    return "late"


def select_top_genes(edge_df: pd.DataFrame, gene_cap: int) -> list[int]:
    counts: dict[int, int] = {}
    for col in ["source_idx", "target_idx"]:
        for value, count in edge_df[col].value_counts().items():
            key = int(value)
            counts[key] = counts.get(key, 0) + int(count)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [gene_idx for gene_idx, _ in ranked[:gene_cap]]


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


def run_h22_phase_transition() -> dict[str, object]:
    row_records: list[dict[str, object]] = []

    for domain_index, (domain, runs) in enumerate(SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_index, (seed_tag, run_dir) in enumerate(runs.items()):
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            if edge_df["label"].nunique() < 2:
                continue
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            n_layers = layer_embeddings.shape[0]
            split_masks = build_split_masks(edge_df)

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_edges = edge_df.loc[split_mask].copy()
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

                for layer in range(n_layers):
                    points = layer_embeddings[layer, edge_gene_indices, :].astype(np.float64)
                    points -= points.mean(axis=0, keepdims=True)
                    n_components = min(24, points.shape[0] - 1, points.shape[1])
                    if n_components < 8:
                        continue
                    points_pca = PCA(
                        n_components=n_components,
                        svd_solver="randomized",
                        random_state=13_220 + domain_index * 1_000 + seed_index * 100 + split_index * 20 + layer,
                    ).fit_transform(points)
                    neighbor_idx = get_knn_indices(points_pca, n_neighbors=12)
                    recon_error = compute_local_reconstruction_errors(points_pca, neighbor_idx)
                    edge_recon = 0.5 * (recon_error[source_local] + recon_error[target_local])
                    auc_edge_recon = safe_auc(labels, edge_recon)

                    row_records.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "phase": phase_from_layer(layer, n_layers),
                            "n_edges": int(labels.size),
                            "n_positive": int(labels.sum()),
                            "n_unique_genes": int(edge_gene_indices.size),
                            "auc_edge_reconstruction": float(auc_edge_recon),
                            "auc_centered": float(auc_edge_recon - 0.5),
                            "mean_gene_reconstruction_error": float(recon_error.mean()),
                            "std_gene_reconstruction_error": float(
                                recon_error.std(ddof=1) if recon_error.size > 1 else 0.0
                            ),
                        }
                    )

    row_df = pd.DataFrame(row_records).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    row_path = ITER_DIR / "h22_phase_transition_by_seed_layer_split.csv"
    row_df.to_csv(row_path, index=False)

    paired = (
        row_df.pivot_table(
            index=["domain", "seed_tag", "layer", "phase"],
            columns="split_regime",
            values="auc_centered",
            aggfunc="mean",
        )
        .reset_index()
        .dropna(subset=["source_disjoint", "target_disjoint"])
    )
    paired["diff_target_minus_source"] = (
        paired["target_disjoint"].astype(float) - paired["source_disjoint"].astype(float)
    )

    phase_means = (
        paired.groupby(["domain", "phase"], as_index=False)["diff_target_minus_source"]
        .mean()
        .rename(columns={"diff_target_minus_source": "mean_diff_target_minus_source"})
    )

    # For balanced design, these contrasts are equivalent to split x phase interaction terms.
    summary_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    n_boot = 2000
    n_perm = 4000
    phase_order = ["early", "mid", "late"]

    for domain_index, domain in enumerate(sorted(paired["domain"].unique())):
        domain_df = paired.loc[paired["domain"] == domain].copy()
        if domain_df.empty:
            continue
        phase_groups = {
            phase: domain_df.loc[domain_df["phase"] == phase, "diff_target_minus_source"].to_numpy(
                dtype=float
            )
            for phase in phase_order
        }
        if any(values.size == 0 for values in phase_groups.values()):
            continue

        mean_early = float(phase_groups["early"].mean())
        mean_mid = float(phase_groups["mid"].mean())
        mean_late = float(phase_groups["late"].mean())

        interaction_late_vs_early = mean_late - mean_early
        interaction_late_vs_mid = mean_late - mean_mid

        rng_boot = np.random.default_rng(13_230 + domain_index)
        boot_late = np.empty(n_boot, dtype=float)
        boot_late_vs_early = np.empty(n_boot, dtype=float)
        boot_late_vs_mid = np.empty(n_boot, dtype=float)
        for boot_idx in range(n_boot):
            sampled = {}
            for phase in phase_order:
                vals = phase_groups[phase]
                sampled[phase] = vals[rng_boot.integers(0, vals.size, size=vals.size)]
            boot_late[boot_idx] = float(np.mean(sampled["late"]))
            boot_late_vs_early[boot_idx] = float(np.mean(sampled["late"]) - np.mean(sampled["early"]))
            boot_late_vs_mid[boot_idx] = float(np.mean(sampled["late"]) - np.mean(sampled["mid"]))

        rng_perm = np.random.default_rng(13_260 + domain_index)
        diffs = domain_df["diff_target_minus_source"].to_numpy(dtype=float)
        phases = domain_df["phase"].to_numpy(dtype=str)
        null_late = np.empty(n_perm, dtype=float)
        null_late_vs_early = np.empty(n_perm, dtype=float)
        null_late_vs_mid = np.empty(n_perm, dtype=float)
        for perm_idx in range(n_perm):
            signs = rng_perm.choice(np.array([-1.0, 1.0]), size=diffs.size, replace=True)
            diffs_perm = diffs * signs
            mean_by_phase_perm = {}
            for phase in phase_order:
                phase_vals = diffs_perm[phases == phase]
                mean_by_phase_perm[phase] = float(np.mean(phase_vals))
            null_late[perm_idx] = mean_by_phase_perm["late"]
            null_late_vs_early[perm_idx] = (
                mean_by_phase_perm["late"] - mean_by_phase_perm["early"]
            )
            null_late_vs_mid[perm_idx] = mean_by_phase_perm["late"] - mean_by_phase_perm["mid"]

        summary_rows.append(
            {
                "domain": domain,
                "n_paired_seed_layer_rows": int(domain_df.shape[0]),
                "mean_diff_early": mean_early,
                "mean_diff_mid": mean_mid,
                "mean_diff_late": mean_late,
                "interaction_late_vs_early": float(interaction_late_vs_early),
                "interaction_late_vs_mid": float(interaction_late_vs_mid),
                "ci95_mean_diff_late_low": float(np.quantile(boot_late, 0.025)),
                "ci95_mean_diff_late_high": float(np.quantile(boot_late, 0.975)),
                "ci95_interaction_late_vs_early_low": float(np.quantile(boot_late_vs_early, 0.025)),
                "ci95_interaction_late_vs_early_high": float(np.quantile(boot_late_vs_early, 0.975)),
                "ci95_interaction_late_vs_mid_low": float(np.quantile(boot_late_vs_mid, 0.025)),
                "ci95_interaction_late_vs_mid_high": float(np.quantile(boot_late_vs_mid, 0.975)),
                "p_mean_diff_late_lower": empirical_lower_tail_p(mean_late, null_late),
                "p_interaction_late_vs_early_lower": empirical_lower_tail_p(
                    interaction_late_vs_early, null_late_vs_early
                ),
                "p_interaction_late_vs_mid_lower": empirical_lower_tail_p(
                    interaction_late_vs_mid, null_late_vs_mid
                ),
            }
        )

        for perm_idx in range(n_perm):
            null_rows.append(
                {
                    "domain": domain,
                    "perm_idx": int(perm_idx),
                    "null_mean_diff_late": float(null_late[perm_idx]),
                    "null_interaction_late_vs_early": float(null_late_vs_early[perm_idx]),
                    "null_interaction_late_vs_mid": float(null_late_vs_mid[perm_idx]),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("domain")
    summary_path = ITER_DIR / "h22_phase_transition_model_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(["domain", "perm_idx"])
    null_path = ITER_DIR / "h22_phase_transition_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    phase_means_path = ITER_DIR / "h22_phase_transition_phase_means.csv"
    phase_means.to_csv(phase_means_path, index=False)

    h22_summary = {
        "n_rows_seed_layer_split": int(row_df.shape[0]),
        "n_rows_paired_seed_layer": int(paired.shape[0]),
        "domains_tested": int(summary_df.shape[0]),
        "domains_with_negative_late_diff": int((summary_df["mean_diff_late"] < 0).sum()),
        "domains_with_significant_negative_late_diff": int(
            ((summary_df["mean_diff_late"] < 0) & (summary_df["p_mean_diff_late_lower"] < 0.05)).sum()
        ),
        "domains_with_significant_negative_late_vs_early": int(
            (
                (summary_df["interaction_late_vs_early"] < 0)
                & (summary_df["p_interaction_late_vs_early_lower"] < 0.05)
            ).sum()
        ),
        "artifact_paths": {
            "by_seed_layer_split": str(row_path),
            "phase_means": str(phase_means_path),
            "model_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }
    return h22_summary


def run_h23_curvature_enrichment() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    n_perm = 300
    layers_to_test = [0, 3, 7, 11]

    def compute_bin_slope_and_delta(labels: np.ndarray, neg_curvature_scores: np.ndarray) -> tuple[float, float]:
        if labels.size < 8:
            return float("nan"), float("nan")
        quantiles = np.quantile(neg_curvature_scores, [0.0, 0.25, 0.5, 0.75, 1.0])
        cut_points = np.unique(quantiles)
        if cut_points.size < 3:
            return float("nan"), float("nan")
        bins = np.digitize(neg_curvature_scores, cut_points[1:-1], right=True)

        x_vals: list[float] = []
        y_vals: list[float] = []
        for b in range(cut_points.size - 1):
            mask = bins == b
            if not np.any(mask):
                continue
            x_vals.append(float(b))
            y_vals.append(float(np.mean(labels[mask])))
        if len(x_vals) < 2:
            return float("nan"), float("nan")
        slope = float(np.polyfit(np.asarray(x_vals), np.asarray(y_vals), deg=1)[0])
        delta = float(y_vals[-1] - y_vals[0])
        return slope, delta

    for domain_index, (domain, runs) in enumerate(SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_index, (seed_tag, run_dir) in enumerate(runs.items()):
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            n_layers = layer_embeddings.shape[0]
            split_masks = build_split_masks(edge_df)

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_edges = edge_df.loc[split_mask].copy()
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

                for layer in layers_to_test:
                    if layer >= n_layers:
                        continue
                    points = layer_embeddings[layer, edge_gene_indices, :].astype(np.float64)
                    points -= points.mean(axis=0, keepdims=True)
                    n_components = min(14, points.shape[0] - 1, points.shape[1])
                    if n_components < 4:
                        continue
                    points_pca = PCA(
                        n_components=n_components,
                        svd_solver="randomized",
                        random_state=13_310 + domain_index * 1_000 + seed_index * 100 + split_index * 20 + layer,
                    ).fit_transform(points)

                    knn = NearestNeighbors(
                        n_neighbors=min(21, points_pca.shape[0]), metric="euclidean"
                    ).fit(points_pca)
                    _, indices = knn.kneighbors(points_pca)
                    graph = nx.Graph()
                    graph.add_nodes_from(range(points_pca.shape[0]))
                    for src in range(points_pca.shape[0]):
                        for tgt in indices[src, 1:]:
                            i, j = sorted((int(src), int(tgt)))
                            if i != j:
                                graph.add_edge(i, j)

                    degree = np.array([val for _, val in graph.degree()], dtype=np.int64)
                    curvature_map: dict[tuple[int, int], float] = {}
                    for u, v in graph.edges():
                        c = float(4 - degree[u] - degree[v])
                        curvature_map[(int(u), int(v))] = c
                        curvature_map[(int(v), int(u))] = c

                    edge_curvature = np.array(
                        [curvature_map.get((int(s), int(t)), np.nan) for s, t in zip(source_local, target_local)],
                        dtype=float,
                    )
                    valid_mask = np.isfinite(edge_curvature)
                    labels_valid = labels[valid_mask]
                    curvature_valid = edge_curvature[valid_mask]
                    if labels_valid.size < 50 or np.unique(labels_valid).size < 2:
                        continue

                    neg_curvature_scores = -curvature_valid
                    auc_curvature = safe_auc(labels_valid, neg_curvature_scores)
                    slope_obs, delta_obs = compute_bin_slope_and_delta(labels_valid, neg_curvature_scores)

                    rng = np.random.default_rng(
                        13_320 + domain_index * 1_000 + seed_index * 100 + split_index * 20 + layer
                    )
                    null_auc = np.empty(n_perm, dtype=float)
                    null_slope = np.empty(n_perm, dtype=float)
                    null_delta = np.empty(n_perm, dtype=float)
                    for perm_idx in range(n_perm):
                        perm_labels = rng.permutation(labels_valid)
                        null_auc[perm_idx] = safe_auc(perm_labels, neg_curvature_scores)
                        slope_perm, delta_perm = compute_bin_slope_and_delta(
                            perm_labels, neg_curvature_scores
                        )
                        null_slope[perm_idx] = slope_perm
                        null_delta[perm_idx] = delta_perm

                    rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "n_edges_total_split": int(labels.size),
                            "n_edges_in_knn_graph": int(labels_valid.size),
                            "n_positive_in_knn_graph": int(labels_valid.sum()),
                            "graph_nodes": int(graph.number_of_nodes()),
                            "graph_edges": int(graph.number_of_edges()),
                            "mean_graph_degree": float(degree.mean()) if degree.size else float("nan"),
                            "auc_neg_curvature": float(auc_curvature),
                            "slope_pos_rate_vs_neg_curvature_bin": float(slope_obs),
                            "delta_top_minus_bottom_pos_rate": float(delta_obs),
                            "p_auc_upper": empirical_upper_tail_p(auc_curvature, null_auc),
                            "p_slope_upper": empirical_upper_tail_p(slope_obs, null_slope),
                            "p_delta_upper": empirical_upper_tail_p(delta_obs, null_delta),
                            "null_mean_auc": float(np.nanmean(null_auc)),
                            "null_mean_slope": float(np.nanmean(null_slope)),
                            "null_mean_delta": float(np.nanmean(null_delta)),
                        }
                    )

    row_df = pd.DataFrame(rows).sort_values(["domain", "split_regime", "layer", "seed_tag"])
    row_path = ITER_DIR / "h23_curvature_enrichment_by_seed_layer_split.csv"
    row_df.to_csv(row_path, index=False)

    split_summary = (
        row_df.groupby(["domain", "split_regime"], as_index=False)
        .agg(
            n_rows=("auc_neg_curvature", "size"),
            mean_auc_neg_curvature=("auc_neg_curvature", "mean"),
            positive_auc_fraction=("auc_neg_curvature", lambda x: float(np.mean(x > 0.5))),
            mean_delta_top_bottom=("delta_top_minus_bottom_pos_rate", "mean"),
            positive_delta_fraction=("delta_top_minus_bottom_pos_rate", lambda x: float(np.mean(x > 0))),
            fisher_p_auc_upper=("p_auc_upper", lambda x: safe_fisher_p(np.asarray(x, dtype=float))),
            fisher_p_delta_upper=("p_delta_upper", lambda x: safe_fisher_p(np.asarray(x, dtype=float))),
        )
        .sort_values(["domain", "split_regime"])
    )
    split_summary_path = ITER_DIR / "h23_curvature_enrichment_split_summary.csv"
    split_summary.to_csv(split_summary_path, index=False)

    domain_summary = (
        row_df.groupby("domain", as_index=False)
        .agg(
            n_rows=("auc_neg_curvature", "size"),
            mean_auc_neg_curvature=("auc_neg_curvature", "mean"),
            mean_delta_top_bottom=("delta_top_minus_bottom_pos_rate", "mean"),
            fisher_p_auc_upper=("p_auc_upper", lambda x: safe_fisher_p(np.asarray(x, dtype=float))),
            fisher_p_delta_upper=("p_delta_upper", lambda x: safe_fisher_p(np.asarray(x, dtype=float))),
        )
        .sort_values("domain")
    )
    domain_summary_path = ITER_DIR / "h23_curvature_enrichment_domain_summary.csv"
    domain_summary.to_csv(domain_summary_path, index=False)

    summary = {
        "n_rows_seed_layer_split": int(row_df.shape[0]),
        "domains_tested": int(domain_summary.shape[0]),
        "domains_with_auc_positive": int((domain_summary["mean_auc_neg_curvature"] > 0.5).sum()),
        "domains_with_auc_fisher_sig": int((domain_summary["fisher_p_auc_upper"] < 0.05).sum()),
        "domains_with_delta_fisher_sig": int((domain_summary["fisher_p_delta_upper"] < 0.05).sum()),
        "artifact_paths": {
            "by_seed_layer_split": str(row_path),
            "split_summary": str(split_summary_path),
            "domain_summary": str(domain_summary_path),
        },
    }
    return summary


def run_h24_cross_model_cca_consistency() -> dict[str, object]:
    from transformers import AutoModel

    model = AutoModel.from_pretrained("ctheodoris/Geneformer")
    geneformer_embeddings = (
        model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    del model

    domain_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    n_perm = 160

    for domain_index, domain in enumerate(["immune", "lung", "external_lung"]):
        run_dir = SCGPT_RUNS_BY_DOMAIN[domain]["seed42_main"]
        layer = int(CROSS_MODEL_LAYER_BY_DOMAIN[domain])
        edge_df = pd.read_csv(GENEFORMER_EDGE_BY_DOMAIN[domain], sep="\t")
        selected_genes = select_top_genes(edge_df, gene_cap=320)
        selected_set = set(selected_genes)
        edge_df = edge_df.loc[
            edge_df["source_idx"].isin(selected_set) & edge_df["target_idx"].isin(selected_set)
        ].copy()
        if edge_df.empty:
            continue

        source_map = edge_df[["source_idx", "source_token_id"]].rename(
            columns={"source_idx": "gene_idx", "source_token_id": "token_id"}
        )
        target_map = edge_df[["target_idx", "target_token_id"]].rename(
            columns={"target_idx": "gene_idx", "target_token_id": "token_id"}
        )
        mapping_df = pd.concat([source_map, target_map], axis=0, ignore_index=True)
        mapping_df["gene_idx"] = mapping_df["gene_idx"].astype(int)
        mapping_df["token_id"] = mapping_df["token_id"].astype(int)
        mapping_df = mapping_df.loc[mapping_df["token_id"] < geneformer_embeddings.shape[0]].copy()

        mapping_df = (
            mapping_df.groupby(["gene_idx", "token_id"], as_index=False)
            .size()
            .sort_values(["gene_idx", "size", "token_id"], ascending=[True, False, True])
            .drop_duplicates(subset=["gene_idx"], keep="first")
            .drop(columns=["size"])
        )

        degree_rank = {gene_idx: rank for rank, gene_idx in enumerate(selected_genes)}
        mapping_df["degree_rank"] = mapping_df["gene_idx"].map(degree_rank).fillna(10**9).astype(int)
        mapping_df = (
            mapping_df.sort_values("degree_rank")
            .drop(columns=["degree_rank"])
            .reset_index(drop=True)
        )

        if mapping_df.shape[0] < 120:
            continue

        gene_ids = mapping_df["gene_idx"].to_numpy(dtype=int)
        token_ids = mapping_df["token_id"].to_numpy(dtype=int)
        n_items = gene_ids.size

        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        if layer >= layer_embeddings.shape[0]:
            continue

        x_raw = layer_embeddings[layer, gene_ids, :].astype(np.float64)
        y_raw = geneformer_embeddings[token_ids, :]

        x_center = x_raw - x_raw.mean(axis=0, keepdims=True)
        y_center = y_raw - y_raw.mean(axis=0, keepdims=True)

        n_pca = min(48, n_items - 1, x_center.shape[1], y_center.shape[1])
        if n_pca < 8:
            continue

        x_pca = PCA(
            n_components=n_pca,
            svd_solver="randomized",
            random_state=13_410 + domain_index * 10,
        ).fit_transform(x_center)
        y_pca = PCA(
            n_components=n_pca,
            svd_solver="randomized",
            random_state=13_411 + domain_index * 10,
        ).fit_transform(y_center)

        n_cca = min(20, n_pca, n_items - 1)
        x_cca, y_cca, canonical_corr = linear_cca_projection(x_pca, y_pca, n_components=n_cca, reg=1e-3)

        x_base = x_pca[:, :n_cca]
        y_base = y_pca[:, :n_cca]

        dist_rho_cca = safe_spearman(pdist(x_cca), pdist(y_cca))
        dist_rho_base = safe_spearman(pdist(x_base), pdist(y_base))
        jaccard_cca = mean_knn_jaccard(x_cca, y_cca, n_neighbors=min(10, n_items - 1))
        jaccard_base = mean_knn_jaccard(x_base, y_base, n_neighbors=min(10, n_items - 1))

        top1_cca = float(np.mean(np.argmin(cdist(x_cca, y_cca, metric="euclidean"), axis=1) == np.arange(n_items)))
        top1_base = float(
            np.mean(np.argmin(cdist(x_base, y_base, metric="euclidean"), axis=1) == np.arange(n_items))
        )
        mean_canonical_corr = float(np.mean(canonical_corr))

        x_unit = row_normalize(x_cca)
        y_unit = row_normalize(y_cca)
        hsic_xy = float(np.linalg.norm(x_unit.T @ y_unit, ord="fro") ** 2)
        hsic_xx = float(np.linalg.norm(x_unit.T @ x_unit, ord="fro"))
        hsic_yy = float(np.linalg.norm(y_unit.T @ y_unit, ord="fro"))
        cka_cca = hsic_xy / (max(hsic_xx, 1e-12) * max(hsic_yy, 1e-12))

        rng = np.random.default_rng(13_420 + domain_index * 100)
        null_canon = np.empty(n_perm, dtype=float)
        null_dist = np.empty(n_perm, dtype=float)
        null_jaccard = np.empty(n_perm, dtype=float)
        null_top1 = np.empty(n_perm, dtype=float)
        null_cka = np.empty(n_perm, dtype=float)

        for perm_idx in range(n_perm):
            perm = rng.permutation(n_items)
            y_perm = y_pca[perm]
            x_null, y_null, corr_null = linear_cca_projection(
                x_pca, y_perm, n_components=n_cca, reg=1e-3
            )
            null_canon[perm_idx] = float(np.mean(corr_null))
            null_dist[perm_idx] = safe_spearman(pdist(x_null), pdist(y_null))
            null_jaccard[perm_idx] = mean_knn_jaccard(
                x_null, y_null, n_neighbors=min(10, n_items - 1)
            )
            null_top1[perm_idx] = float(
                np.mean(np.argmin(cdist(x_null, y_null, metric="euclidean"), axis=1) == np.arange(n_items))
            )

            x_null_unit = row_normalize(x_null)
            y_null_unit = row_normalize(y_null)
            hsic_xy_null = float(np.linalg.norm(x_null_unit.T @ y_null_unit, ord="fro") ** 2)
            hsic_xx_null = float(np.linalg.norm(x_null_unit.T @ x_null_unit, ord="fro"))
            hsic_yy_null = float(np.linalg.norm(y_null_unit.T @ y_null_unit, ord="fro"))
            null_cka[perm_idx] = hsic_xy_null / (max(hsic_xx_null, 1e-12) * max(hsic_yy_null, 1e-12))

            null_rows.append(
                {
                    "domain": domain,
                    "perm_idx": int(perm_idx),
                    "null_mean_canonical_corr": float(null_canon[perm_idx]),
                    "null_distance_spearman": float(null_dist[perm_idx]),
                    "null_knn_jaccard": float(null_jaccard[perm_idx]),
                    "null_top1_retrieval": float(null_top1[perm_idx]),
                    "null_cka": float(null_cka[perm_idx]),
                }
            )

        domain_rows.append(
            {
                "domain": domain,
                "layer": int(layer),
                "n_matched_genes": int(n_items),
                "pca_dim": int(n_pca),
                "cca_dim": int(n_cca),
                "mean_canonical_corr": float(mean_canonical_corr),
                "distance_spearman_cca": float(dist_rho_cca),
                "distance_spearman_baseline": float(dist_rho_base),
                "distance_spearman_delta_vs_baseline": float(dist_rho_cca - dist_rho_base),
                "knn_jaccard_cca": float(jaccard_cca),
                "knn_jaccard_baseline": float(jaccard_base),
                "knn_jaccard_delta_vs_baseline": float(jaccard_cca - jaccard_base),
                "top1_retrieval_cca": float(top1_cca),
                "top1_retrieval_baseline": float(top1_base),
                "top1_retrieval_delta_vs_baseline": float(top1_cca - top1_base),
                "cka_cca": float(cka_cca),
                "p_mean_canonical_corr_upper": empirical_upper_tail_p(mean_canonical_corr, null_canon),
                "p_distance_spearman_upper": empirical_upper_tail_p(dist_rho_cca, null_dist),
                "p_knn_jaccard_upper": empirical_upper_tail_p(jaccard_cca, null_jaccard),
                "p_top1_retrieval_upper": empirical_upper_tail_p(top1_cca, null_top1),
                "p_cka_upper": empirical_upper_tail_p(cka_cca, null_cka),
                "null_mean_canonical_corr": float(np.nanmean(null_canon)),
                "null_mean_distance_spearman": float(np.nanmean(null_dist)),
                "null_mean_knn_jaccard": float(np.nanmean(null_jaccard)),
                "null_mean_top1_retrieval": float(np.nanmean(null_top1)),
                "null_mean_cka": float(np.nanmean(null_cka)),
            }
        )

    domain_df = pd.DataFrame(domain_rows).sort_values("domain")
    domain_path = ITER_DIR / "h24_cross_model_cca_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(["domain", "perm_idx"])
    null_path = ITER_DIR / "h24_cross_model_cca_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    overall_summary = pd.DataFrame(
        [
            {
                "domains_tested": int(domain_df.shape[0]),
                "mean_canonical_corr": float(domain_df["mean_canonical_corr"].mean()),
                "mean_distance_spearman_cca": float(domain_df["distance_spearman_cca"].mean()),
                "mean_knn_jaccard_cca": float(domain_df["knn_jaccard_cca"].mean()),
                "mean_top1_retrieval_cca": float(domain_df["top1_retrieval_cca"].mean()),
                "domains_distance_sig": int((domain_df["p_distance_spearman_upper"] < 0.05).sum()),
                "domains_jaccard_sig": int((domain_df["p_knn_jaccard_upper"] < 0.05).sum()),
                "domains_top1_sig": int((domain_df["p_top1_retrieval_upper"] < 0.05).sum()),
                "combined_fisher_distance_p": safe_fisher_p(
                    domain_df["p_distance_spearman_upper"].to_numpy(dtype=float)
                ),
                "combined_fisher_jaccard_p": safe_fisher_p(
                    domain_df["p_knn_jaccard_upper"].to_numpy(dtype=float)
                ),
                "combined_fisher_top1_p": safe_fisher_p(
                    domain_df["p_top1_retrieval_upper"].to_numpy(dtype=float)
                ),
            }
        ]
    )
    overall_path = ITER_DIR / "h24_cross_model_cca_overall_summary.csv"
    overall_summary.to_csv(overall_path, index=False)

    summary = {
        "domains_tested": int(domain_df.shape[0]),
        "domains_distance_sig": int((domain_df["p_distance_spearman_upper"] < 0.05).sum()),
        "domains_jaccard_sig": int((domain_df["p_knn_jaccard_upper"] < 0.05).sum()),
        "domains_top1_sig": int((domain_df["p_top1_retrieval_upper"] < 0.05).sum()),
        "mean_canonical_corr": float(domain_df["mean_canonical_corr"].mean()),
        "mean_distance_spearman_cca": float(domain_df["distance_spearman_cca"].mean()),
        "mean_knn_jaccard_cca": float(domain_df["knn_jaccard_cca"].mean()),
        "mean_top1_retrieval_cca": float(domain_df["top1_retrieval_cca"].mean()),
        "combined_fisher_distance_p": float(
            overall_summary["combined_fisher_distance_p"].iloc[0]
        ),
        "combined_fisher_jaccard_p": float(overall_summary["combined_fisher_jaccard_p"].iloc[0]),
        "combined_fisher_top1_p": float(overall_summary["combined_fisher_top1_p"].iloc[0]),
        "artifact_paths": {
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
            "overall_summary": str(overall_path),
        },
    }
    return summary


def main() -> None:
    required_paths = []
    for runs in SCGPT_RUNS_BY_DOMAIN.values():
        for run_dir in runs.values():
            required_paths.append(run_dir / "layer_gene_embeddings.npy")
            required_paths.append(run_dir / "cycle1_edge_dataset.tsv")
    required_paths.extend(GENEFORMER_EDGE_BY_DOMAIN.values())

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    h22_summary = run_h22_phase_transition()
    h23_summary = run_h23_curvature_enrichment()
    h24_summary = run_h24_cross_model_cca_consistency()

    iteration_summary = {
        "iteration": "iter_0013",
        "inputs": {
            "scgpt_runs_by_domain": {
                domain: {seed: str(path) for seed, path in run_dict.items()}
                for domain, run_dict in SCGPT_RUNS_BY_DOMAIN.items()
            },
            "geneformer_edge_by_domain": {
                domain: str(path) for domain, path in GENEFORMER_EDGE_BY_DOMAIN.items()
            },
            "cross_model_layer_by_domain": CROSS_MODEL_LAYER_BY_DOMAIN,
        },
        "h22_phase_transition": h22_summary,
        "h23_curvature_enrichment": h23_summary,
        "h24_cross_model_cca_consistency": h24_summary,
    }

    summary_path = ITER_DIR / "iter0013_screen_summary.json"
    summary_path.write_text(json.dumps(iteration_summary, indent=2))
    print(json.dumps(iteration_summary, indent=2))


if __name__ == "__main__":
    main()
