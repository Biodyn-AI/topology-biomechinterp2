from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from ripser import ripser
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import combine_pvalues, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.neighbors import NearestNeighbors


ITER_DIR = Path("iterations/iter_0010")
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

CROSS_MODEL_DISAGREEMENT_PATH = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
    "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/"
    "cycle15_cross_model_disagreement_stratification/disagreement_quantile_label_rates.csv"
)


def empirical_upper_tail_p(observed: float, null_values: np.ndarray) -> float:
    return float((1 + np.sum(null_values >= observed)) / (null_values.size + 1))


def empirical_lower_tail_p(observed: float, null_values: np.ndarray) -> float:
    return float((1 + np.sum(null_values <= observed)) / (null_values.size + 1))


def empirical_two_sided_p(observed: float, null_values: np.ndarray) -> float:
    return float((1 + np.sum(np.abs(null_values) >= abs(observed))) / (null_values.size + 1))


def safe_fisher_p(pvals: np.ndarray) -> float:
    clipped = np.clip(np.asarray(pvals, dtype=float), 1e-12, 1.0)
    _, p = combine_pvalues(clipped, method="fisher")
    return float(p)


def compute_h1_sum(points_or_distances: np.ndarray, distance_matrix: bool = False) -> float:
    dgms = ripser(points_or_distances, maxdim=1, distance_matrix=distance_matrix)["dgms"]
    h1 = dgms[1]
    finite = h1[np.isfinite(h1[:, 1])]
    if finite.size == 0:
        return 0.0
    return float((finite[:, 1] - finite[:, 0]).sum())


def feature_shuffle_null(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = np.empty_like(points)
    for col in range(points.shape[1]):
        shuffled[:, col] = points[rng.permutation(points.shape[0]), col]
    return shuffled


def connect_components_with_bridges(graph: nx.Graph, pairwise_euclidean: np.ndarray) -> nx.Graph:
    connected = graph.copy()
    components = [np.asarray(sorted(c), dtype=int) for c in nx.connected_components(connected)]

    while len(components) > 1:
        base = components[0]
        best_distance = float("inf")
        best_u = -1
        best_v = -1
        best_component_index = -1

        for component_index in range(1, len(components)):
            candidate = components[component_index]
            distances = pairwise_euclidean[np.ix_(base, candidate)]
            flat_index = int(np.argmin(distances))
            row_idx, col_idx = np.unravel_index(flat_index, distances.shape)
            distance = float(distances[row_idx, col_idx])
            if distance < best_distance:
                best_distance = distance
                best_u = int(base[row_idx])
                best_v = int(candidate[col_idx])
                best_component_index = component_index

        if best_component_index < 1:
            raise RuntimeError("Failed to identify bridge edge across components.")

        connected.add_edge(best_u, best_v, weight=best_distance)
        components = [np.asarray(sorted(c), dtype=int) for c in nx.connected_components(connected)]

    return connected


def build_connected_knn_graph(
    points: np.ndarray,
    pairwise_euclidean: np.ndarray,
    min_neighbors: int,
    max_neighbors: int,
) -> tuple[nx.Graph, int, bool]:
    n_points = points.shape[0]
    min_k = max(2, min(min_neighbors, n_points - 1))
    max_k = max(min_k, min(max_neighbors, n_points - 1))
    last_graph: nx.Graph | None = None

    for k in range(min_k, max_k + 1):
        knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        knn.fit(points)
        distances, indices = knn.kneighbors(points)

        graph = nx.Graph()
        graph.add_nodes_from(range(n_points))

        for source in range(n_points):
            for distance, target in zip(distances[source, 1:], indices[source, 1:]):
                i, j = sorted((int(source), int(target)))
                if i == j:
                    continue
                weight = float(distance)
                if graph.has_edge(i, j):
                    if weight < graph[i][j]["weight"]:
                        graph[i][j]["weight"] = weight
                else:
                    graph.add_edge(i, j, weight=weight)

        if nx.is_connected(graph):
            return graph, k, False
        last_graph = graph

    if last_graph is None:
        raise RuntimeError("kNN graph construction failed.")

    bridged = connect_components_with_bridges(last_graph, pairwise_euclidean)
    if nx.is_connected(bridged):
        return bridged, max_k, True

    raise RuntimeError("Unable to produce connected graph after bridge fallback.")


def graph_to_geodesic_distances(graph: nx.Graph, pairwise_euclidean: np.ndarray) -> np.ndarray:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for u, v in graph.edges():
        weight = float(pairwise_euclidean[u, v])
        rows.extend([u, v])
        cols.extend([v, u])
        vals.extend([weight, weight])

    n_nodes = graph.number_of_nodes()
    sparse_graph = csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float64)
    distances = shortest_path(sparse_graph, directed=False, unweighted=False)
    if not np.isfinite(distances).all():
        raise RuntimeError("Non-finite geodesic distances encountered.")
    return np.asarray(distances, dtype=np.float64)


def run_h13_manifold_distance() -> dict[str, object]:
    edge_df = pd.read_csv(IMMUNE_EDGE_PATH, sep="\t")
    source_threshold = float(edge_df["source_idx"].median())
    target_threshold = float(edge_df["target_idx"].median())

    split_masks = {
        "source_disjoint": edge_df["source_idx"] <= source_threshold,
        "target_disjoint": edge_df["target_idx"] > target_threshold,
    }

    pca_dim = 14
    knn_k_min = 10
    knn_k_max = 35
    n_label_permutations = 200

    records: list[dict[str, object]] = []

    for seed_index, (seed_tag, emb_path) in enumerate(SCGPT_IMMUNE_RUNS.items()):
        layer_embeddings = np.load(emb_path, mmap_mode="r")
        n_layers = layer_embeddings.shape[0]

        for split_index, (split_regime, mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[mask].copy()
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
                run_seed = 1_100_000 + seed_index * 10_000 + split_index * 1_000 + layer
                run_rng = np.random.default_rng(run_seed)

                points = layer_embeddings[layer, edge_gene_indices, :].astype(np.float64)
                points -= points.mean(axis=0, keepdims=True)
                n_components = min(pca_dim, points.shape[0] - 1, points.shape[1])
                points_pca = PCA(
                    n_components=n_components,
                    svd_solver="randomized",
                    random_state=2_300 + seed_index * 100 + split_index * 20 + layer,
                ).fit_transform(points)

                pairwise_euclidean = pairwise_distances(points_pca, metric="euclidean")
                graph, knn_k, used_component_bridging = build_connected_knn_graph(
                    points=points_pca,
                    pairwise_euclidean=pairwise_euclidean,
                    min_neighbors=knn_k_min,
                    max_neighbors=knn_k_max,
                )
                geodesic = graph_to_geodesic_distances(graph, pairwise_euclidean)

                scores_euclidean = -pairwise_euclidean[source_local, target_local]
                scores_geodesic = -geodesic[source_local, target_local]

                auc_euclidean = float(roc_auc_score(labels, scores_euclidean))
                auc_geodesic = float(roc_auc_score(labels, scores_geodesic))
                delta_auc = float(auc_geodesic - auc_euclidean)

                null_delta = np.empty(n_label_permutations, dtype=float)
                null_auc_euclidean = np.empty(n_label_permutations, dtype=float)
                null_auc_geodesic = np.empty(n_label_permutations, dtype=float)
                for perm_idx in range(n_label_permutations):
                    labels_perm = run_rng.permutation(labels)
                    null_auc_euclidean[perm_idx] = float(roc_auc_score(labels_perm, scores_euclidean))
                    null_auc_geodesic[perm_idx] = float(roc_auc_score(labels_perm, scores_geodesic))
                    null_delta[perm_idx] = float(null_auc_geodesic[perm_idx] - null_auc_euclidean[perm_idx])

                delta_null_mean = float(null_delta.mean())
                delta_null_std = float(null_delta.std(ddof=1)) if null_delta.size > 1 else 0.0
                delta_z = float((delta_auc - delta_null_mean) / (delta_null_std + 1e-9))

                records.append(
                    {
                        "domain": "immune",
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges": int(split_edges.shape[0]),
                        "positive_rate": float(labels.mean()),
                        "n_unique_genes": int(edge_gene_indices.size),
                        "pca_dim": int(n_components),
                        "knn_k": int(knn_k),
                        "used_component_bridging": bool(used_component_bridging),
                        "auc_euclidean": auc_euclidean,
                        "auc_geodesic": auc_geodesic,
                        "delta_auc_geodesic_minus_euclidean": delta_auc,
                        "null_delta_mean": delta_null_mean,
                        "null_delta_std": delta_null_std,
                        "delta_auc_z_vs_null": delta_z,
                        "p_delta_upper": empirical_upper_tail_p(delta_auc, null_delta),
                        "p_delta_two_sided": empirical_two_sided_p(delta_auc, null_delta),
                        "p_euclidean_upper": empirical_upper_tail_p(auc_euclidean, null_auc_euclidean),
                        "p_geodesic_upper": empirical_upper_tail_p(auc_geodesic, null_auc_geodesic),
                    }
                )

    by_seed_df = pd.DataFrame(records).sort_values(["split_regime", "layer", "seed_tag"]).reset_index(drop=True)
    by_seed_path = ITER_DIR / "h13_manifold_distance_by_seed_layer_split.csv"
    by_seed_df.to_csv(by_seed_path, index=False)

    layer_records: list[dict[str, object]] = []
    for (split_regime, layer), group in by_seed_df.groupby(["split_regime", "layer"], sort=True):
        layer_records.append(
            {
                "split_regime": split_regime,
                "layer": int(layer),
                "mean_auc_euclidean": float(group["auc_euclidean"].mean()),
                "mean_auc_geodesic": float(group["auc_geodesic"].mean()),
                "mean_delta_auc_geodesic_minus_euclidean": float(
                    group["delta_auc_geodesic_minus_euclidean"].mean()
                ),
                "delta_positive_fraction": float(
                    (group["delta_auc_geodesic_minus_euclidean"] > 0).mean()
                ),
                "mean_delta_auc_z_vs_null": float(group["delta_auc_z_vs_null"].mean()),
                "fisher_p_delta_upper": safe_fisher_p(group["p_delta_upper"].to_numpy(dtype=float)),
                "fisher_p_delta_two_sided": safe_fisher_p(
                    group["p_delta_two_sided"].to_numpy(dtype=float)
                ),
                "mean_knn_k": float(group["knn_k"].mean()),
                "bridge_fraction": float(group["used_component_bridging"].mean()),
            }
        )

    layer_df = pd.DataFrame(layer_records).sort_values(["split_regime", "layer"]).reset_index(drop=True)
    layer_path = ITER_DIR / "h13_manifold_distance_layer_summary.csv"
    layer_df.to_csv(layer_path, index=False)

    split_records: list[dict[str, object]] = []
    for split_regime, group in layer_df.groupby("split_regime", sort=True):
        split_records.append(
            {
                "split_regime": split_regime,
                "n_layers": int(group.shape[0]),
                "mean_delta_auc_geodesic_minus_euclidean": float(
                    group["mean_delta_auc_geodesic_minus_euclidean"].mean()
                ),
                "mean_delta_positive_fraction": float(group["delta_positive_fraction"].mean()),
                "n_layers_fisher_p_delta_upper_lt_0_05": int(
                    (group["fisher_p_delta_upper"] < 0.05).sum()
                ),
                "combined_fisher_p_delta_upper": safe_fisher_p(
                    group["fisher_p_delta_upper"].to_numpy(dtype=float)
                ),
                "combined_fisher_p_delta_two_sided": safe_fisher_p(
                    group["fisher_p_delta_two_sided"].to_numpy(dtype=float)
                ),
            }
        )

    split_df = pd.DataFrame(split_records).sort_values("split_regime").reset_index(drop=True)
    split_path = ITER_DIR / "h13_manifold_distance_split_summary.csv"
    split_df.to_csv(split_path, index=False)

    pass_rows: list[dict[str, object]] = []
    source = layer_df[layer_df["split_regime"] == "source_disjoint"].set_index("layer")
    target = layer_df[layer_df["split_regime"] == "target_disjoint"].set_index("layer")
    shared_layers = sorted(set(source.index).intersection(set(target.index)))
    for layer in shared_layers:
        source_row = source.loc[layer]
        target_row = target.loc[layer]
        pass_rows.append(
            {
                "layer": int(layer),
                "source_mean_delta_auc": float(source_row["mean_delta_auc_geodesic_minus_euclidean"]),
                "target_mean_delta_auc": float(target_row["mean_delta_auc_geodesic_minus_euclidean"]),
                "both_splits_positive_mean_delta": bool(
                    (source_row["mean_delta_auc_geodesic_minus_euclidean"] > 0)
                    and (target_row["mean_delta_auc_geodesic_minus_euclidean"] > 0)
                ),
                "both_splits_fisher_sig_upper": bool(
                    (source_row["fisher_p_delta_upper"] < 0.05)
                    and (target_row["fisher_p_delta_upper"] < 0.05)
                ),
            }
        )

    pass_df = pd.DataFrame(pass_rows).sort_values("layer").reset_index(drop=True)
    pass_path = ITER_DIR / "h13_manifold_distance_pass_matrix.csv"
    pass_df.to_csv(pass_path, index=False)

    summary = {
        "n_rows_by_seed_layer_split": int(by_seed_df.shape[0]),
        "n_layers_tested": int(layer_df["layer"].nunique()),
        "source_split_mean_delta_auc": float(
            split_df.loc[
                split_df["split_regime"] == "source_disjoint",
                "mean_delta_auc_geodesic_minus_euclidean",
            ].iloc[0]
        ),
        "target_split_mean_delta_auc": float(
            split_df.loc[
                split_df["split_regime"] == "target_disjoint",
                "mean_delta_auc_geodesic_minus_euclidean",
            ].iloc[0]
        ),
        "source_layers_sig_count": int(
            split_df.loc[
                split_df["split_regime"] == "source_disjoint",
                "n_layers_fisher_p_delta_upper_lt_0_05",
            ].iloc[0]
        ),
        "target_layers_sig_count": int(
            split_df.loc[
                split_df["split_regime"] == "target_disjoint",
                "n_layers_fisher_p_delta_upper_lt_0_05",
            ].iloc[0]
        ),
        "dual_split_positive_layers": int(pass_df["both_splits_positive_mean_delta"].sum()),
        "dual_split_significant_layers": int(pass_df["both_splits_fisher_sig_upper"].sum()),
        "artifact_paths": {
            "by_seed_layer_split": str(by_seed_path),
            "layer_summary": str(layer_path),
            "split_summary": str(split_path),
            "pass_matrix": str(pass_path),
        },
    }

    return summary


def run_h14_topology_stability() -> dict[str, object]:
    sample_sizes = [120, 180]
    pca_dims = [10, 14]
    n_bootstrap = 4
    # Need >=19 draws so empirical one-sided p-values can fall below 0.05.
    n_null = 24

    records: list[dict[str, object]] = []

    for seed_index, (seed_tag, emb_path) in enumerate(SCGPT_IMMUNE_RUNS.items()):
        layer_embeddings = np.load(emb_path, mmap_mode="r")
        n_layers, n_genes, _ = layer_embeddings.shape

        for layer in range(n_layers):
            for sample_index, sample_size in enumerate(sample_sizes):
                for pca_index, pca_dim in enumerate(pca_dims):
                    for bootstrap_idx in range(n_bootstrap):
                        run_seed = (
                            1_400_000
                            + seed_index * 100_000
                            + layer * 1_000
                            + sample_index * 100
                            + pca_index * 20
                            + bootstrap_idx
                        )
                        run_rng = np.random.default_rng(run_seed)

                        sampled_idx = run_rng.choice(
                            n_genes,
                            size=min(sample_size, n_genes),
                            replace=False,
                        )
                        points = layer_embeddings[layer, sampled_idx, :].astype(np.float64)
                        points -= points.mean(axis=0, keepdims=True)

                        n_components = min(pca_dim, points.shape[0] - 1, points.shape[1])
                        points_pca = PCA(
                            n_components=n_components,
                            svd_solver="randomized",
                            random_state=7_500
                            + seed_index * 100
                            + layer * 10
                            + sample_index * 3
                            + pca_index,
                        ).fit_transform(points)

                        h1_observed = compute_h1_sum(points_pca, distance_matrix=False)

                        null_values = np.empty(n_null, dtype=float)
                        for null_idx in range(n_null):
                            null_rng = np.random.default_rng(
                                1_600_000
                                + seed_index * 100_000
                                + layer * 1_000
                                + sample_index * 100
                                + pca_index * 20
                                + bootstrap_idx * 5
                                + null_idx
                            )
                            null_points = feature_shuffle_null(points_pca, null_rng)
                            null_values[null_idx] = compute_h1_sum(null_points, distance_matrix=False)

                        null_mean = float(null_values.mean())
                        null_std = float(null_values.std(ddof=1)) if null_values.size > 1 else 0.0
                        delta = float(h1_observed - null_mean)
                        delta_z = float(delta / (null_std + 1e-9))

                        records.append(
                            {
                                "domain": "immune",
                                "seed_tag": seed_tag,
                                "layer": int(layer),
                                "sample_size": int(sample_size),
                                "pca_dim": int(n_components),
                                "bootstrap_idx": int(bootstrap_idx),
                                "h1_observed": h1_observed,
                                "h1_null_mean": null_mean,
                                "h1_null_std": null_std,
                                "h1_delta_observed_minus_null": delta,
                                "h1_delta_z": delta_z,
                                "p_h1_upper": empirical_upper_tail_p(h1_observed, null_values),
                            }
                        )

    bootstrap_df = (
        pd.DataFrame(records)
        .sort_values(["seed_tag", "layer", "sample_size", "pca_dim", "bootstrap_idx"])
        .reset_index(drop=True)
    )
    bootstrap_path = ITER_DIR / "h14_topology_stability_bootstrap_records.csv"
    bootstrap_df.to_csv(bootstrap_path, index=False)

    seed_layer_setting_records: list[dict[str, object]] = []
    for keys, group in bootstrap_df.groupby(["seed_tag", "layer", "sample_size", "pca_dim"], sort=True):
        seed_tag, layer, sample_size, pca_dim = keys
        mean_delta = float(group["h1_delta_observed_minus_null"].mean())
        std_delta = float(group["h1_delta_observed_minus_null"].std(ddof=1))
        seed_layer_setting_records.append(
            {
                "seed_tag": seed_tag,
                "layer": int(layer),
                "sample_size": int(sample_size),
                "pca_dim": int(pca_dim),
                "n_bootstrap": int(group.shape[0]),
                "mean_h1_observed": float(group["h1_observed"].mean()),
                "mean_h1_null": float(group["h1_null_mean"].mean()),
                "mean_h1_delta": mean_delta,
                "std_h1_delta": std_delta,
                "cv_h1_delta": float(std_delta / (abs(mean_delta) + 1e-9)),
                "positive_delta_fraction": float((group["h1_delta_observed_minus_null"] > 0).mean()),
                "fisher_p_h1_upper": safe_fisher_p(group["p_h1_upper"].to_numpy(dtype=float)),
            }
        )

    seed_layer_setting_df = (
        pd.DataFrame(seed_layer_setting_records)
        .sort_values(["seed_tag", "layer", "sample_size", "pca_dim"])
        .reset_index(drop=True)
    )
    seed_layer_setting_path = ITER_DIR / "h14_topology_stability_seed_layer_setting_summary.csv"
    seed_layer_setting_df.to_csv(seed_layer_setting_path, index=False)

    layer_records: list[dict[str, object]] = []
    for layer, group in seed_layer_setting_df.groupby("layer", sort=True):
        layer_records.append(
            {
                "layer": int(layer),
                "n_seed_setting_rows": int(group.shape[0]),
                "mean_h1_delta": float(group["mean_h1_delta"].mean()),
                "median_h1_delta": float(group["mean_h1_delta"].median()),
                "delta_positive_fraction": float((group["mean_h1_delta"] > 0).mean()),
                "mean_cv_h1_delta": float(group["cv_h1_delta"].mean()),
                "fisher_sig_fraction": float((group["fisher_p_h1_upper"] < 0.05).mean()),
                "combined_fisher_p_h1_upper": safe_fisher_p(group["fisher_p_h1_upper"].to_numpy(dtype=float)),
            }
        )

    layer_df = pd.DataFrame(layer_records).sort_values("layer").reset_index(drop=True)
    layer_path = ITER_DIR / "h14_topology_stability_layer_summary.csv"
    layer_df.to_csv(layer_path, index=False)

    filtration_records: list[dict[str, object]] = []
    for (seed_tag, layer), group in seed_layer_setting_df.groupby(["seed_tag", "layer"], sort=True):
        deltas = group["mean_h1_delta"].to_numpy(dtype=float)
        if deltas.size == 0:
            continue
        filtration_records.append(
            {
                "seed_tag": seed_tag,
                "layer": int(layer),
                "n_settings": int(deltas.size),
                "delta_min": float(deltas.min()),
                "delta_max": float(deltas.max()),
                "delta_range": float(deltas.max() - deltas.min()),
                "delta_std_across_settings": float(deltas.std(ddof=1) if deltas.size > 1 else 0.0),
                "all_settings_positive": bool(np.all(deltas > 0)),
                "positive_setting_fraction": float((deltas > 0).mean()),
            }
        )

    filtration_df = (
        pd.DataFrame(filtration_records)
        .sort_values(["layer", "seed_tag"])
        .reset_index(drop=True)
    )
    filtration_path = ITER_DIR / "h14_topology_stability_filtration_sensitivity.csv"
    filtration_df.to_csv(filtration_path, index=False)

    filtration_layer_records: list[dict[str, object]] = []
    for layer, group in filtration_df.groupby("layer", sort=True):
        filtration_layer_records.append(
            {
                "layer": int(layer),
                "mean_delta_range": float(group["delta_range"].mean()),
                "mean_delta_std_across_settings": float(group["delta_std_across_settings"].mean()),
                "all_settings_positive_fraction": float(group["all_settings_positive"].mean()),
                "mean_positive_setting_fraction": float(group["positive_setting_fraction"].mean()),
            }
        )

    filtration_layer_df = (
        pd.DataFrame(filtration_layer_records).sort_values("layer").reset_index(drop=True)
    )
    filtration_layer_path = ITER_DIR / "h14_topology_stability_filtration_layer_summary.csv"
    filtration_layer_df.to_csv(filtration_layer_path, index=False)

    summary = {
        "n_bootstrap_rows": int(bootstrap_df.shape[0]),
        "n_seed_layer_setting_rows": int(seed_layer_setting_df.shape[0]),
        "n_layers_tested": int(layer_df.shape[0]),
        "mean_layer_delta": float(layer_df["mean_h1_delta"].mean()),
        "layers_with_positive_mean_delta": int((layer_df["mean_h1_delta"] > 0).sum()),
        "layers_combined_fisher_p_lt_0_05": int((layer_df["combined_fisher_p_h1_upper"] < 0.05).sum()),
        "mean_all_settings_positive_fraction": float(
            filtration_layer_df["all_settings_positive_fraction"].mean()
        ),
        "artifact_paths": {
            "bootstrap_records": str(bootstrap_path),
            "seed_layer_setting_summary": str(seed_layer_setting_path),
            "layer_summary": str(layer_path),
            "filtration_sensitivity": str(filtration_path),
            "filtration_layer_summary": str(filtration_layer_path),
        },
    }

    return summary


def run_h15_cross_model_alignment() -> dict[str, object]:
    disagreement_df = pd.read_csv(CROSS_MODEL_DISAGREEMENT_PATH)
    n_perm = 3000

    records: list[dict[str, object]] = []
    for domain_index, (domain, group) in enumerate(disagreement_df.groupby("domain", sort=True)):
        ordered = group.sort_values("disagreement_bin").reset_index(drop=True)
        x = ordered["mean_abs_disagreement"].to_numpy(dtype=float)
        y = ordered["positive_rate"].to_numpy(dtype=float)

        rho = float(spearmanr(x, y).correlation)
        slope = float(np.polyfit(x, y, deg=1)[0])

        rng = np.random.default_rng(1_800_000 + domain_index * 100)
        null_rho = np.empty(n_perm, dtype=float)
        for idx in range(n_perm):
            y_perm = rng.permutation(y)
            null_rho[idx] = float(spearmanr(x, y_perm).correlation)

        top_bin_rate = float(ordered.loc[ordered["disagreement_bin"].idxmax(), "positive_rate"])
        bottom_bin_rate = float(ordered.loc[ordered["disagreement_bin"].idxmin(), "positive_rate"])

        records.append(
            {
                "domain": domain,
                "n_bins": int(ordered.shape[0]),
                "spearman_rho_disagreement_vs_positive_rate": rho,
                "linear_slope_disagreement_vs_positive_rate": slope,
                "top_minus_bottom_positive_rate": float(top_bin_rate - bottom_bin_rate),
                "mean_positive_rate": float(y.mean()),
                "p_two_sided": empirical_two_sided_p(rho, null_rho),
                "p_negative_tail": empirical_lower_tail_p(rho, null_rho),
                "p_positive_tail": empirical_upper_tail_p(rho, null_rho),
            }
        )

    trend_df = pd.DataFrame(records).sort_values("domain").reset_index(drop=True)
    trend_path = ITER_DIR / "h15_cross_model_disagreement_trend.csv"
    trend_df.to_csv(trend_path, index=False)

    summary_payload = {
        "domains_tested": int(trend_df.shape[0]),
        "mean_spearman_rho": float(trend_df["spearman_rho_disagreement_vs_positive_rate"].mean()),
        "domains_negative_rho": int(
            (trend_df["spearman_rho_disagreement_vs_positive_rate"] < 0).sum()
        ),
        "domains_two_sided_p_lt_0_05": int((trend_df["p_two_sided"] < 0.05).sum()),
        "combined_fisher_p_two_sided": safe_fisher_p(trend_df["p_two_sided"].to_numpy(dtype=float)),
        "combined_fisher_p_negative_tail": safe_fisher_p(
            trend_df["p_negative_tail"].to_numpy(dtype=float)
        ),
        "artifact_paths": {
            "trend_by_domain": str(trend_path),
        },
    }

    summary_path = ITER_DIR / "h15_cross_model_disagreement_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    return summary_payload


def main() -> None:
    missing_paths = [
        str(path)
        for path in [*SCGPT_IMMUNE_RUNS.values(), IMMUNE_EDGE_PATH, CROSS_MODEL_DISAGREEMENT_PATH]
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(f"Missing required inputs: {missing_paths}")

    h13_summary = run_h13_manifold_distance()
    h14_summary = run_h14_topology_stability()
    h15_summary = run_h15_cross_model_alignment()

    iteration_summary = {
        "iteration": "iter_0010",
        "inputs": {
            "scgpt_immune_runs": {k: str(v) for k, v in SCGPT_IMMUNE_RUNS.items()},
            "immune_edge_dataset": str(IMMUNE_EDGE_PATH),
            "cross_model_disagreement_bins": str(CROSS_MODEL_DISAGREEMENT_PATH),
        },
        "h13_manifold_distance": h13_summary,
        "h14_topology_stability": h14_summary,
        "h15_cross_model_alignment": h15_summary,
    }

    summary_path = ITER_DIR / "iter0010_screen_summary.json"
    summary_path.write_text(json.dumps(iteration_summary, indent=2))

    print(json.dumps(iteration_summary, indent=2))


if __name__ == "__main__":
    main()
