from __future__ import annotations

import itertools
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


ITER_DIR = Path("iterations/iter_0011")
ITER_DIR.mkdir(parents=True, exist_ok=True)

SCGPT_IMMUNE_RUNS = {
    "seed42_main": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "subproject_38_geometric_residual_stream_interpretability/"
        "implementation/outputs/cycle4_immune_main/layer_gene_embeddings.npy"
    ),
    "seed43": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "subproject_38_geometric_residual_stream_interpretability/"
        "implementation/outputs/cycle4_immune_seed43/layer_gene_embeddings.npy"
    ),
    "seed44": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "subproject_38_geometric_residual_stream_interpretability/"
        "implementation/outputs/cycle4_immune_seed44/layer_gene_embeddings.npy"
    ),
}

IMMUNE_EDGE_PATH = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
    "subproject_38_geometric_residual_stream_interpretability/"
    "implementation/outputs/cycle4_immune_main/cycle1_edge_dataset.tsv"
)

SCGPT_ALT_BY_DOMAIN = {
    "immune": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "subproject_38_geometric_residual_stream_interpretability/"
        "implementation/outputs/cycle4_immune_main/alt_geometry_metrics_layer0_immune_pca.csv"
    ),
    "lung": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "subproject_38_geometric_residual_stream_interpretability/"
        "implementation/outputs/cycle6_lung_main/alt_geometry_metrics_layer0_lung_pca.csv"
    ),
    "external_lung": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "subproject_38_geometric_residual_stream_interpretability/"
        "implementation/outputs/cycle7_external_lung_main/alt_geometry_metrics_layer3_external_lung_pca.csv"
    ),
}

GENEFORMER_FEATURE_BY_DOMAIN = {
    "immune": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "subproject_38_geometric_residual_stream_interpretability/"
        "implementation/outputs/cycle12_geneformer_immune_bootstrap/geneformer_feature_metrics.csv"
    ),
    "lung": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "subproject_38_geometric_residual_stream_interpretability/"
        "implementation/outputs/cycle12_geneformer_lung_bootstrap/geneformer_feature_metrics.csv"
    ),
    "external_lung": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "subproject_38_geometric_residual_stream_interpretability/"
        "implementation/outputs/cycle12_geneformer_external_lung_bootstrap/geneformer_feature_metrics.csv"
    ),
}

H13_REFERENCE_PATH = Path("iterations/iter_0010/h13_manifold_distance_by_seed_layer_split.csv")


def empirical_upper_tail_p(observed: float, null_values: np.ndarray) -> float:
    return float((1 + np.sum(null_values >= observed)) / (null_values.size + 1))


def empirical_lower_tail_p(observed: float, null_values: np.ndarray) -> float:
    return float((1 + np.sum(null_values <= observed)) / (null_values.size + 1))


def empirical_two_sided_p(observed: float, null_values: np.ndarray) -> float:
    return float((1 + np.sum(np.abs(null_values) >= abs(observed))) / (null_values.size + 1))


def safe_fisher_p(pvals: np.ndarray) -> float:
    clean = np.asarray(pvals, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan")
    clipped = np.clip(clean, 1e-12, 1.0)
    _, p = combine_pvalues(clipped, method="fisher")
    return float(p)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    if np.unique(scores).size < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def build_knn_graph(points: np.ndarray, n_neighbors: int) -> nx.Graph:
    n_points = points.shape[0]
    k = max(2, min(n_neighbors, n_points - 1))
    knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    knn.fit(points)
    distances, indices = knn.kneighbors(points)

    graph = nx.Graph()
    graph.add_nodes_from(range(n_points))

    for src in range(n_points):
        for dist, tgt in zip(distances[src, 1:], indices[src, 1:]):
            i, j = sorted((int(src), int(tgt)))
            if i == j:
                continue
            weight = float(dist)
            if graph.has_edge(i, j):
                if weight < graph[i][j]["weight"]:
                    graph[i][j]["weight"] = weight
            else:
                graph.add_edge(i, j, weight=weight)

    return graph


def positive_rate_delta(labels: np.ndarray, same_community: np.ndarray) -> tuple[float, float, float]:
    same_mask = same_community == 1
    diff_mask = ~same_mask
    pos_rate_same = float(labels[same_mask].mean()) if same_mask.any() else float("nan")
    pos_rate_diff = float(labels[diff_mask].mean()) if diff_mask.any() else float("nan")
    if np.isfinite(pos_rate_same) and np.isfinite(pos_rate_diff):
        delta = float(pos_rate_same - pos_rate_diff)
    else:
        delta = float("nan")
    return pos_rate_same, pos_rate_diff, delta


def participation_ratio_dim(points: np.ndarray) -> float:
    variances = points.var(axis=0, ddof=1)
    numerator = float(variances.sum() ** 2)
    denominator = float(np.square(variances).sum())
    if denominator <= 1e-12:
        return 0.0
    return numerator / denominator


def local_linearity_ratio(points: np.ndarray, top_k: int = 5) -> float:
    variances = points.var(axis=0, ddof=1)
    total = float(variances.sum())
    if total <= 1e-12:
        return 0.0
    k = min(top_k, variances.size)
    return float(variances[:k].sum() / total)


def run_h16_module_structure() -> dict[str, object]:
    edge_df = pd.read_csv(IMMUNE_EDGE_PATH, sep="\t")
    source_threshold = float(edge_df["source_idx"].median())
    target_threshold = float(edge_df["target_idx"].median())
    split_masks = {
        "source_disjoint": edge_df["source_idx"] <= source_threshold,
        "target_disjoint": edge_df["target_idx"] > target_threshold,
    }

    pca_dim = 14
    knn_k = 20
    n_perm = 300
    records: list[dict[str, object]] = []

    for seed_index, (seed_tag, emb_path) in enumerate(SCGPT_IMMUNE_RUNS.items()):
        layer_embeddings = np.load(emb_path, mmap_mode="r")
        n_layers = layer_embeddings.shape[0]

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
            index_map = {int(gene_idx): int(local_idx) for local_idx, gene_idx in enumerate(edge_gene_indices)}
            source_local = split_edges["source_idx"].map(index_map).to_numpy(dtype=int)
            target_local = split_edges["target_idx"].map(index_map).to_numpy(dtype=int)
            labels = split_edges["label"].to_numpy(dtype=int)

            for layer in range(n_layers):
                run_seed = 2_001_000 + seed_index * 10_000 + split_index * 1_000 + layer
                run_rng = np.random.default_rng(run_seed)

                points = layer_embeddings[layer, edge_gene_indices, :].astype(np.float64)
                points -= points.mean(axis=0, keepdims=True)
                n_components = min(pca_dim, points.shape[0] - 1, points.shape[1])
                if n_components < 2:
                    continue

                points_pca = PCA(
                    n_components=n_components,
                    svd_solver="randomized",
                    random_state=4_100 + seed_index * 100 + split_index * 20 + layer,
                ).fit_transform(points)

                graph = build_knn_graph(points_pca, n_neighbors=knn_k)
                communities = list(
                    nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
                )

                community_id = np.zeros(points_pca.shape[0], dtype=int)
                for cid, community in enumerate(communities):
                    for node in community:
                        community_id[int(node)] = int(cid)

                same_community = (community_id[source_local] == community_id[target_local]).astype(int)
                auc_same_community = safe_auc(labels, same_community.astype(float))
                pos_rate_same, pos_rate_diff, delta_pos_rate = positive_rate_delta(labels, same_community)

                null_auc = np.empty(n_perm, dtype=float)
                null_delta = np.empty(n_perm, dtype=float)
                for perm_idx in range(n_perm):
                    perm_labels = run_rng.permutation(labels)
                    null_auc[perm_idx] = safe_auc(perm_labels, same_community.astype(float))
                    _, _, perm_delta = positive_rate_delta(perm_labels, same_community)
                    null_delta[perm_idx] = float(perm_delta)

                p_auc_upper = empirical_upper_tail_p(auc_same_community, null_auc)
                p_delta_upper = (
                    empirical_upper_tail_p(delta_pos_rate, null_delta)
                    if np.isfinite(delta_pos_rate)
                    else float("nan")
                )

                modularity = float(
                    nx.algorithms.community.quality.modularity(graph, communities, weight="weight")
                )
                largest_community_fraction = float(
                    max(len(community) for community in communities) / points_pca.shape[0]
                )

                records.append(
                    {
                        "domain": "immune",
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges": int(labels.size),
                        "positive_rate": float(labels.mean()),
                        "n_unique_genes": int(edge_gene_indices.size),
                        "pca_dim": int(n_components),
                        "knn_k": int(max(2, min(knn_k, points_pca.shape[0] - 1))),
                        "n_communities": int(len(communities)),
                        "largest_community_fraction": largest_community_fraction,
                        "graph_modularity": modularity,
                        "same_community_edge_fraction": float(same_community.mean()),
                        "auc_same_community": float(auc_same_community),
                        "positive_rate_same_community": pos_rate_same,
                        "positive_rate_diff_community": pos_rate_diff,
                        "delta_positive_rate_same_minus_diff": delta_pos_rate,
                        "p_auc_upper": float(p_auc_upper),
                        "p_delta_upper": float(p_delta_upper),
                    }
                )

    by_seed_path = ITER_DIR / "h16_module_structure_by_seed_layer_split.csv"
    by_seed_df = (
        pd.DataFrame(records)
        .sort_values(["split_regime", "layer", "seed_tag"])
        .reset_index(drop=True)
    )
    by_seed_df.to_csv(by_seed_path, index=False)

    layer_records: list[dict[str, object]] = []
    for (split_regime, layer), group in by_seed_df.groupby(["split_regime", "layer"], sort=True):
        layer_records.append(
            {
                "split_regime": split_regime,
                "layer": int(layer),
                "n_seed_rows": int(group.shape[0]),
                "mean_auc_same_community": float(group["auc_same_community"].mean()),
                "mean_delta_positive_rate_same_minus_diff": float(
                    group["delta_positive_rate_same_minus_diff"].mean()
                ),
                "mean_same_community_edge_fraction": float(
                    group["same_community_edge_fraction"].mean()
                ),
                "positive_auc_fraction": float((group["auc_same_community"] > 0.5).mean()),
                "positive_delta_fraction": float(
                    (group["delta_positive_rate_same_minus_diff"] > 0).mean()
                ),
                "combined_fisher_p_auc_upper": safe_fisher_p(group["p_auc_upper"].to_numpy(dtype=float)),
                "combined_fisher_p_delta_upper": safe_fisher_p(
                    group["p_delta_upper"].to_numpy(dtype=float)
                ),
                "mean_graph_modularity": float(group["graph_modularity"].mean()),
            }
        )

    layer_df = pd.DataFrame(layer_records).sort_values(["split_regime", "layer"]).reset_index(drop=True)
    layer_path = ITER_DIR / "h16_module_structure_layer_summary.csv"
    layer_df.to_csv(layer_path, index=False)

    split_records: list[dict[str, object]] = []
    for split_regime, group in layer_df.groupby("split_regime", sort=True):
        split_records.append(
            {
                "split_regime": split_regime,
                "n_layers": int(group.shape[0]),
                "mean_auc_same_community": float(group["mean_auc_same_community"].mean()),
                "mean_delta_positive_rate_same_minus_diff": float(
                    group["mean_delta_positive_rate_same_minus_diff"].mean()
                ),
                "layers_auc_gt_0_5": int((group["mean_auc_same_community"] > 0.5).sum()),
                "layers_delta_gt_0": int(
                    (group["mean_delta_positive_rate_same_minus_diff"] > 0).sum()
                ),
                "layers_fisher_auc_sig": int((group["combined_fisher_p_auc_upper"] < 0.05).sum()),
                "layers_fisher_delta_sig": int(
                    (group["combined_fisher_p_delta_upper"] < 0.05).sum()
                ),
                "mean_graph_modularity": float(group["mean_graph_modularity"].mean()),
            }
        )

    split_df = pd.DataFrame(split_records).sort_values("split_regime").reset_index(drop=True)
    split_path = ITER_DIR / "h16_module_structure_split_summary.csv"
    split_df.to_csv(split_path, index=False)

    summary = {
        "n_rows_by_seed_layer_split": int(by_seed_df.shape[0]),
        "n_layers_tested": int(layer_df["layer"].nunique()),
        "source_mean_auc": float(
            split_df.loc[split_df["split_regime"] == "source_disjoint", "mean_auc_same_community"].iloc[0]
        ),
        "target_mean_auc": float(
            split_df.loc[split_df["split_regime"] == "target_disjoint", "mean_auc_same_community"].iloc[0]
        ),
        "source_layers_auc_gt_0_5": int(
            split_df.loc[split_df["split_regime"] == "source_disjoint", "layers_auc_gt_0_5"].iloc[0]
        ),
        "target_layers_auc_gt_0_5": int(
            split_df.loc[split_df["split_regime"] == "target_disjoint", "layers_auc_gt_0_5"].iloc[0]
        ),
        "artifact_paths": {
            "by_seed_layer_split": str(by_seed_path),
            "layer_summary": str(layer_path),
            "split_summary": str(split_path),
        },
    }
    return summary


def run_h17_cross_model_transfer() -> dict[str, object]:
    shared_features = ["centered_cosine", "dot", "cosine"]
    perm_indices = list(itertools.permutations(range(len(shared_features))))
    domain_records: list[dict[str, object]] = []
    domain_vectors: list[tuple[np.ndarray, np.ndarray]] = []

    for domain in ["immune", "lung", "external_lung"]:
        scgpt_df = pd.read_csv(SCGPT_ALT_BY_DOMAIN[domain])
        geneformer_df = pd.read_csv(GENEFORMER_FEATURE_BY_DOMAIN[domain])

        scgpt_subset = (
            scgpt_df.loc[scgpt_df["feature"].isin(shared_features), ["feature", "delta_cv_auroc"]]
            .drop_duplicates(subset="feature")
            .rename(columns={"delta_cv_auroc": "scgpt_delta_cv_auroc"})
        )
        geneformer_subset = (
            geneformer_df.loc[geneformer_df["feature"].isin(shared_features), ["feature", "delta_cv_auroc"]]
            .drop_duplicates(subset="feature")
            .rename(columns={"delta_cv_auroc": "geneformer_delta_cv_auroc"})
        )
        merged = scgpt_subset.merge(geneformer_subset, on="feature", how="inner")
        merged["feature"] = pd.Categorical(merged["feature"], categories=shared_features, ordered=True)
        merged = merged.sort_values("feature").reset_index(drop=True)

        if merged.shape[0] != len(shared_features):
            raise RuntimeError(f"Missing shared feature for domain={domain}: {merged['feature'].tolist()}")

        x = merged["scgpt_delta_cv_auroc"].to_numpy(dtype=float)
        y = merged["geneformer_delta_cv_auroc"].to_numpy(dtype=float)
        domain_vectors.append((x, y))

        rho = float(spearmanr(x, y).correlation)
        top_scgpt_index = int(np.argmax(x))
        top_geneformer_index = int(np.argmax(y))
        top_feature_match = bool(top_scgpt_index == top_geneformer_index)
        top_feature_name = shared_features[top_scgpt_index]
        transfer_gap = float(y[top_scgpt_index] - y[top_geneformer_index])

        null_rhos = np.empty(len(perm_indices), dtype=float)
        null_top_match = np.empty(len(perm_indices), dtype=int)
        for i, perm in enumerate(perm_indices):
            y_perm = y[np.array(perm, dtype=int)]
            null_rhos[i] = float(spearmanr(x, y_perm).correlation)
            null_top_match[i] = int(np.argmax(x) == np.argmax(y_perm))

        domain_records.append(
            {
                "domain": domain,
                "n_shared_features": int(len(shared_features)),
                "spearman_rho_feature_transfer": rho,
                "top_feature_scgpt": top_feature_name,
                "top_feature_geneformer": shared_features[top_geneformer_index],
                "top_feature_match": top_feature_match,
                "geneformer_gap_for_scgpt_top_minus_geneformer_top": transfer_gap,
                "p_rho_upper_exact": empirical_upper_tail_p(rho, null_rhos),
                "p_rho_two_sided_exact": empirical_two_sided_p(rho, null_rhos),
                "p_top_feature_match_exact": empirical_upper_tail_p(int(top_feature_match), null_top_match),
            }
        )

    domain_df = pd.DataFrame(domain_records).sort_values("domain").reset_index(drop=True)
    domain_path = ITER_DIR / "h17_cross_model_transfer_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    # Exact global null: independent feature-index permutations in each domain.
    global_null_records: list[dict[str, object]] = []
    observed_mean_rho = float(domain_df["spearman_rho_feature_transfer"].mean())
    observed_top_match_count = int(domain_df["top_feature_match"].sum())

    for combo in itertools.product(range(len(perm_indices)), repeat=len(domain_vectors)):
        rho_values: list[float] = []
        top_match_count = 0
        for domain_idx, perm_idx in enumerate(combo):
            x, y = domain_vectors[domain_idx]
            perm = perm_indices[perm_idx]
            y_perm = y[np.array(perm, dtype=int)]
            rho_values.append(float(spearmanr(x, y_perm).correlation))
            top_match_count += int(np.argmax(x) == np.argmax(y_perm))
        global_null_records.append(
            {
                "combo_id": "|".join(str(v) for v in combo),
                "mean_rho": float(np.mean(rho_values)),
                "top_match_count": int(top_match_count),
            }
        )

    global_null_df = pd.DataFrame(global_null_records)
    global_null_path = ITER_DIR / "h17_cross_model_transfer_global_null.csv"
    global_null_df.to_csv(global_null_path, index=False)

    summary = {
        "domains_tested": int(domain_df.shape[0]),
        "mean_spearman_rho": observed_mean_rho,
        "domains_with_positive_rho": int((domain_df["spearman_rho_feature_transfer"] > 0).sum()),
        "top_feature_match_count": observed_top_match_count,
        "p_mean_rho_upper_exact_global": empirical_upper_tail_p(
            observed_mean_rho, global_null_df["mean_rho"].to_numpy(dtype=float)
        ),
        "p_top_match_count_upper_exact_global": empirical_upper_tail_p(
            observed_top_match_count, global_null_df["top_match_count"].to_numpy(dtype=float)
        ),
        "artifact_paths": {
            "domain_summary": str(domain_path),
            "global_null": str(global_null_path),
        },
    }

    summary_path = ITER_DIR / "h17_cross_model_transfer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def run_h18_intrinsic_geodesic_coupling() -> dict[str, object]:
    edge_df = pd.read_csv(IMMUNE_EDGE_PATH, sep="\t")
    h13_df = pd.read_csv(H13_REFERENCE_PATH)

    source_threshold = float(edge_df["source_idx"].median())
    target_threshold = float(edge_df["target_idx"].median())
    split_masks = {
        "source_disjoint": edge_df["source_idx"] <= source_threshold,
        "target_disjoint": edge_df["target_idx"] > target_threshold,
    }

    intrinsic_rows: list[dict[str, object]] = []
    for seed_index, (seed_tag, emb_path) in enumerate(SCGPT_IMMUNE_RUNS.items()):
        layer_embeddings = np.load(emb_path, mmap_mode="r")
        n_layers = layer_embeddings.shape[0]

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            edge_gene_indices = np.unique(
                np.concatenate(
                    [
                        split_edges["source_idx"].to_numpy(dtype=int),
                        split_edges["target_idx"].to_numpy(dtype=int),
                    ]
                )
            )

            for layer in range(n_layers):
                points = layer_embeddings[layer, edge_gene_indices, :].astype(np.float64)
                points -= points.mean(axis=0, keepdims=True)
                n_components = min(64, points.shape[0] - 1, points.shape[1])
                if n_components < 6:
                    continue
                points_pca = PCA(
                    n_components=n_components,
                    svd_solver="randomized",
                    random_state=6_100 + seed_index * 100 + split_index * 20 + layer,
                ).fit_transform(points)

                intrinsic_rows.append(
                    {
                        "domain": "immune",
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_unique_genes": int(edge_gene_indices.size),
                        "pca_dim_intrinsic": int(n_components),
                        "participation_ratio_dim": participation_ratio_dim(points_pca),
                        "local_linearity_top5": local_linearity_ratio(points_pca, top_k=5),
                    }
                )

    intrinsic_df = pd.DataFrame(intrinsic_rows)
    merged = intrinsic_df.merge(
        h13_df[
            [
                "seed_tag",
                "split_regime",
                "layer",
                "delta_auc_geodesic_minus_euclidean",
            ]
        ],
        on=["seed_tag", "split_regime", "layer"],
        how="inner",
        validate="one_to_one",
    )

    metrics_path = ITER_DIR / "h18_intrinsic_geodesic_metrics_by_seed_layer_split.csv"
    merged.sort_values(["split_regime", "layer", "seed_tag"]).to_csv(metrics_path, index=False)

    metric_names = ["participation_ratio_dim", "local_linearity_top5"]
    by_seed_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []

    n_perm = 3000
    rng_seed_lookup = {
        ("source_disjoint", "participation_ratio_dim"): 7_301,
        ("source_disjoint", "local_linearity_top5"): 7_302,
        ("target_disjoint", "participation_ratio_dim"): 7_401,
        ("target_disjoint", "local_linearity_top5"): 7_402,
    }
    for split_regime in ["source_disjoint", "target_disjoint"]:
        split_df = merged.loc[merged["split_regime"] == split_regime].copy()
        for metric in metric_names:
            per_seed_vectors: list[tuple[np.ndarray, np.ndarray]] = []
            per_seed_rhos: list[float] = []
            for seed_tag in sorted(split_df["seed_tag"].unique()):
                seed_df = split_df.loc[split_df["seed_tag"] == seed_tag].sort_values("layer")
                x = seed_df[metric].to_numpy(dtype=float)
                y = seed_df["delta_auc_geodesic_minus_euclidean"].to_numpy(dtype=float)
                rho = float(spearmanr(x, y).correlation)
                per_seed_vectors.append((x, y))
                per_seed_rhos.append(rho)
                by_seed_records.append(
                    {
                        "split_regime": split_regime,
                        "metric": metric,
                        "seed_tag": seed_tag,
                        "n_layers": int(seed_df.shape[0]),
                        "spearman_rho": rho,
                    }
                )

            observed_mean_rho = float(np.mean(per_seed_rhos))
            rng = np.random.default_rng(rng_seed_lookup[(split_regime, metric)])
            null_mean_rho = np.empty(n_perm, dtype=float)
            for perm_idx in range(n_perm):
                rho_perm_values: list[float] = []
                for x, y in per_seed_vectors:
                    y_perm = rng.permutation(y)
                    rho_perm_values.append(float(spearmanr(x, y_perm).correlation))
                null_mean_rho[perm_idx] = float(np.mean(rho_perm_values))

            summary_records.append(
                {
                    "split_regime": split_regime,
                    "metric": metric,
                    "n_seeds": int(len(per_seed_rhos)),
                    "mean_seed_spearman_rho": observed_mean_rho,
                    "positive_seed_fraction": float(np.mean(np.array(per_seed_rhos) > 0)),
                    "p_mean_rho_upper": empirical_upper_tail_p(observed_mean_rho, null_mean_rho),
                    "p_mean_rho_lower": empirical_lower_tail_p(observed_mean_rho, null_mean_rho),
                    "p_mean_rho_two_sided": empirical_two_sided_p(observed_mean_rho, null_mean_rho),
                    "null_mean_rho_mean": float(null_mean_rho.mean()),
                    "null_mean_rho_std": float(null_mean_rho.std(ddof=1)),
                }
            )

    by_seed_df = pd.DataFrame(by_seed_records).sort_values(["split_regime", "metric", "seed_tag"])
    by_seed_path = ITER_DIR / "h18_intrinsic_geodesic_coupling_by_seed.csv"
    by_seed_df.to_csv(by_seed_path, index=False)

    summary_df = pd.DataFrame(summary_records).sort_values(["split_regime", "metric"])
    summary_path = ITER_DIR / "h18_intrinsic_geodesic_coupling_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    result = {
        "n_rows_seed_layer_split": int(merged.shape[0]),
        "metrics_tested": metric_names,
        "source_participation_mean_rho": float(
            summary_df.loc[
                (summary_df["split_regime"] == "source_disjoint")
                & (summary_df["metric"] == "participation_ratio_dim"),
                "mean_seed_spearman_rho",
            ].iloc[0]
        ),
        "target_participation_mean_rho": float(
            summary_df.loc[
                (summary_df["split_regime"] == "target_disjoint")
                & (summary_df["metric"] == "participation_ratio_dim"),
                "mean_seed_spearman_rho",
            ].iloc[0]
        ),
        "source_linearity_mean_rho": float(
            summary_df.loc[
                (summary_df["split_regime"] == "source_disjoint")
                & (summary_df["metric"] == "local_linearity_top5"),
                "mean_seed_spearman_rho",
            ].iloc[0]
        ),
        "target_linearity_mean_rho": float(
            summary_df.loc[
                (summary_df["split_regime"] == "target_disjoint")
                & (summary_df["metric"] == "local_linearity_top5"),
                "mean_seed_spearman_rho",
            ].iloc[0]
        ),
        "artifact_paths": {
            "metrics_by_seed_layer_split": str(metrics_path),
            "coupling_by_seed": str(by_seed_path),
            "coupling_summary": str(summary_path),
        },
    }
    return result


def main() -> None:
    required_paths = [
        *SCGPT_IMMUNE_RUNS.values(),
        IMMUNE_EDGE_PATH,
        *SCGPT_ALT_BY_DOMAIN.values(),
        *GENEFORMER_FEATURE_BY_DOMAIN.values(),
        H13_REFERENCE_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    h16_summary = run_h16_module_structure()
    h17_summary = run_h17_cross_model_transfer()
    h18_summary = run_h18_intrinsic_geodesic_coupling()

    iteration_summary = {
        "iteration": "iter_0011",
        "inputs": {
            "scgpt_immune_runs": {k: str(v) for k, v in SCGPT_IMMUNE_RUNS.items()},
            "immune_edge_dataset": str(IMMUNE_EDGE_PATH),
            "scgpt_alt_feature_tables": {k: str(v) for k, v in SCGPT_ALT_BY_DOMAIN.items()},
            "geneformer_feature_tables": {k: str(v) for k, v in GENEFORMER_FEATURE_BY_DOMAIN.items()},
            "h13_reference": str(H13_REFERENCE_PATH),
        },
        "h16_module_structure": h16_summary,
        "h17_cross_model_transfer": h17_summary,
        "h18_intrinsic_geodesic_coupling": h18_summary,
    }

    summary_path = ITER_DIR / "iter0011_screen_summary.json"
    summary_path.write_text(json.dumps(iteration_summary, indent=2))
    print(json.dumps(iteration_summary, indent=2))


if __name__ == "__main__":
    main()
