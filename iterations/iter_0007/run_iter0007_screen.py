from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from ripser import ripser
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import combine_pvalues
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


ITER_DIR = Path("iterations/iter_0007")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def compute_h1_sum(points_or_distances: np.ndarray, distance_matrix: bool = False) -> float:
    """Return the summed finite H1 lifetimes from ripser persistence diagrams."""
    diagrams = ripser(points_or_distances, maxdim=1, distance_matrix=distance_matrix)["dgms"]
    h1 = diagrams[1]
    finite = h1[np.isfinite(h1[:, 1])]
    if finite.size == 0:
        return 0.0
    return float((finite[:, 1] - finite[:, 0]).sum())


def empirical_upper_tail_p(observed: float, null_values: np.ndarray) -> float:
    """Empirical one-sided p-value for observed >= null draws."""
    return float((1 + np.sum(null_values >= observed)) / (null_values.size + 1))


def empirical_lower_tail_p(observed: float, null_values: np.ndarray) -> float:
    """Empirical one-sided p-value for observed <= null draws."""
    return float((1 + np.sum(null_values <= observed)) / (null_values.size + 1))


def connect_components_with_bridges(graph: nx.Graph, pairwise_euclidean: np.ndarray) -> nx.Graph:
    """Connect kNN components by repeatedly adding nearest cross-component bridge edges."""
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
            raise RuntimeError("Failed to identify a valid bridge edge across components.")

        connected.add_edge(best_u, best_v, weight=best_distance)
        components = [np.asarray(sorted(c), dtype=int) for c in nx.connected_components(connected)]

    return connected


def build_connected_knn_graph(
    points: np.ndarray,
    pairwise_euclidean: np.ndarray,
    min_neighbors: int,
    max_neighbors: int,
) -> tuple[nx.Graph, int, bool]:
    """Build a symmetric kNN graph and enforce connectivity via adaptive-k + bridge fallback."""
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
        raise RuntimeError("kNN graph construction failed before connectivity checks.")

    bridged = connect_components_with_bridges(last_graph, pairwise_euclidean)
    if nx.is_connected(bridged):
        return bridged, max_k, True

    raise RuntimeError("Unable to produce connected graph after kNN + bridging fallback.")


def rewire_degree_preserving_connected(
    base_graph: nx.Graph,
    rng: np.random.Generator,
    swap_multiplier: float,
    max_retries: int,
) -> nx.Graph:
    """Generate a connected degree-preserving rewired graph via repeated edge swaps."""
    n_edges = base_graph.number_of_edges()
    n_swaps = max(1, int(round(n_edges * swap_multiplier)))
    max_tries = max(n_swaps * 20, 200)

    for _ in range(max_retries):
        rewired = base_graph.copy()
        seed = int(rng.integers(0, 2_147_483_647))
        try:
            nx.double_edge_swap(rewired, nswap=n_swaps, max_tries=max_tries, seed=seed)
        except Exception:
            continue
        if nx.is_connected(rewired):
            return rewired

    raise RuntimeError("Failed to obtain connected degree-preserving rewire graph.")


def graph_to_geodesic_distances(graph: nx.Graph, pairwise_euclidean: np.ndarray) -> np.ndarray:
    """Map graph edges to Euclidean weights and compute all-pairs geodesic distances."""
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


def mean_geodesic_distortion(geodesic: np.ndarray, euclidean: np.ndarray) -> float:
    """Return mean geodesic/euclidean ratio over unique non-degenerate pairs."""
    tri = np.triu_indices_from(euclidean, k=1)
    denom = euclidean[tri]
    numer = geodesic[tri]
    valid = denom > 1e-9
    if not np.any(valid):
        raise RuntimeError("No valid non-degenerate Euclidean pairs for distortion estimate.")
    return float(np.mean(numer[valid] / denom[valid]))


def summarize_against_null(observed: float, null_values: np.ndarray) -> dict[str, float]:
    """Compute effect-size summary for observed metric against null draws."""
    null_mean = float(null_values.mean())
    null_std = float(null_values.std(ddof=1)) if null_values.size > 1 else 0.0
    delta = float(observed - null_mean)
    z_score = float(delta / (null_std + 1e-9))
    return {
        "null_mean": null_mean,
        "null_std": null_std,
        "delta": delta,
        "z_score": z_score,
    }


domain_runs = {
    "immune": {
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
}

missing_paths = [
    str(path)
    for runs in domain_runs.values()
    for path in runs.values()
    if not path.exists()
]
if missing_paths:
    raise FileNotFoundError(f"Missing embedding files: {missing_paths}")


split_specs = {
    "source_disjoint": lambda n_genes: np.arange(0, n_genes // 2, dtype=int),
    "target_disjoint": lambda n_genes: np.arange(n_genes // 2, n_genes, dtype=int),
}

n_points = 160
pca_dim = 14
knn_k_min = 12
knn_k_max = 30
n_null_rewire = 6
rewire_swap_multiplier = 1.5
rewire_retry_per_null = 10
rewire_attempt_multiplier = 8

records: list[dict[str, object]] = []

for domain_index, (domain, runs) in enumerate(domain_runs.items()):
    for seed_index, (seed_tag, emb_path) in enumerate(runs.items()):
        layer_embeddings = np.load(emb_path, mmap_mode="r")
        n_layers, n_genes, _ = layer_embeddings.shape

        for split_index, (split_regime, pool_selector) in enumerate(split_specs.items()):
            candidate_pool = pool_selector(n_genes)
            if candidate_pool.size == 0:
                continue

            for layer in range(n_layers):
                run_seed = (
                    930000
                    + domain_index * 100000
                    + seed_index * 10000
                    + split_index * 1000
                    + layer * 10
                )
                run_rng = np.random.default_rng(run_seed)

                sample_size = min(n_points, candidate_pool.size)
                sampled_idx = run_rng.choice(candidate_pool, size=sample_size, replace=False)
                points = layer_embeddings[layer, sampled_idx, :].astype(np.float64)
                points -= points.mean(axis=0, keepdims=True)

                n_components = min(pca_dim, points.shape[0] - 1, points.shape[1])
                points_pca = PCA(
                    n_components=n_components,
                    svd_solver="randomized",
                    random_state=9400
                    + domain_index * 100
                    + seed_index * 10
                    + split_index * 3
                    + layer,
                ).fit_transform(points)

                # Euclidean observed metric (the mismatch-prone comparator from iter_0006).
                h1_observed_euclidean = compute_h1_sum(points_pca, distance_matrix=False)

                pairwise = pairwise_distances(points_pca, metric="euclidean")
                base_graph, knn_k_used, used_component_bridging = build_connected_knn_graph(
                    points=points_pca,
                    pairwise_euclidean=pairwise,
                    min_neighbors=knn_k_min,
                    max_neighbors=knn_k_max,
                )

                # Geodesic observed metric uses the same graph/metric family as the rewired null.
                base_geodesic = graph_to_geodesic_distances(base_graph, pairwise)
                h1_observed_geodesic = compute_h1_sum(base_geodesic, distance_matrix=True)
                distortion_observed = mean_geodesic_distortion(base_geodesic, pairwise)

                rewired_h1_values: list[float] = []
                rewired_distortion_values: list[float] = []
                total_rewire_attempts = 0
                max_attempts = n_null_rewire * rewire_attempt_multiplier

                while len(rewired_h1_values) < n_null_rewire and total_rewire_attempts < max_attempts:
                    total_rewire_attempts += 1
                    rw_rng = np.random.default_rng(
                        950000
                        + domain_index * 100000
                        + seed_index * 10000
                        + split_index * 1000
                        + layer * 100
                        + total_rewire_attempts
                    )
                    try:
                        rewired_graph = rewire_degree_preserving_connected(
                            base_graph=base_graph,
                            rng=rw_rng,
                            swap_multiplier=rewire_swap_multiplier,
                            max_retries=rewire_retry_per_null,
                        )
                        rewired_geodesic = graph_to_geodesic_distances(rewired_graph, pairwise)
                        rewired_h1_values.append(
                            compute_h1_sum(rewired_geodesic, distance_matrix=True)
                        )
                        rewired_distortion_values.append(
                            mean_geodesic_distortion(rewired_geodesic, pairwise)
                        )
                    except Exception:
                        continue

                if len(rewired_h1_values) < 2:
                    raise RuntimeError(
                        "Insufficient rewired null draws for stable estimates: "
                        f"domain={domain} seed={seed_tag} split={split_regime} layer={layer}"
                    )

                rewired_h1 = np.asarray(rewired_h1_values, dtype=float)
                rewired_distortion = np.asarray(rewired_distortion_values, dtype=float)

                euclidean_stats = summarize_against_null(h1_observed_euclidean, rewired_h1)
                geodesic_stats = summarize_against_null(h1_observed_geodesic, rewired_h1)
                distortion_stats = summarize_against_null(distortion_observed, rewired_distortion)

                records.append(
                    {
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_genes_pool": int(candidate_pool.size),
                        "n_points": int(points_pca.shape[0]),
                        "pca_dim": int(n_components),
                        "knn_k": int(knn_k_used),
                        "used_component_bridging": bool(used_component_bridging),
                        "n_null_draws": int(rewired_h1.size),
                        "rewire_attempts": int(total_rewire_attempts),
                        "h1_observed_euclidean": float(h1_observed_euclidean),
                        "h1_observed_geodesic": float(h1_observed_geodesic),
                        "h1_rewire_null_mean": euclidean_stats["null_mean"],
                        "h1_rewire_null_std": euclidean_stats["null_std"],
                        "h1_delta_euclidean_vs_rewire": euclidean_stats["delta"],
                        "h1_z_euclidean_vs_rewire": euclidean_stats["z_score"],
                        "h1_delta_geodesic_vs_rewire": geodesic_stats["delta"],
                        "h1_z_geodesic_vs_rewire": geodesic_stats["z_score"],
                        "h1_delta_shift_geodesic_minus_euclidean": float(
                            geodesic_stats["delta"] - euclidean_stats["delta"]
                        ),
                        "h1_p_perm_euclidean_vs_rewire": empirical_upper_tail_p(
                            h1_observed_euclidean, rewired_h1
                        ),
                        "h1_p_perm_geodesic_vs_rewire": empirical_upper_tail_p(
                            h1_observed_geodesic, rewired_h1
                        ),
                        "distortion_observed": float(distortion_observed),
                        "distortion_rewire_null_mean": distortion_stats["null_mean"],
                        "distortion_rewire_null_std": distortion_stats["null_std"],
                        "distortion_delta_observed_minus_null": distortion_stats["delta"],
                        "distortion_z_observed_minus_null": distortion_stats["z_score"],
                        "distortion_p_lower": empirical_lower_tail_p(
                            distortion_observed, rewired_distortion
                        ),
                    }
                )


by_seed_df = pd.DataFrame(records).sort_values(
    ["domain", "split_regime", "layer", "seed_tag"]
).reset_index(drop=True)
by_seed_path = ITER_DIR / "h1_immune_metric_matched_by_seed_layer.csv"
by_seed_df.to_csv(by_seed_path, index=False)


layer_summary_records: list[dict[str, object]] = []
for keys, group in by_seed_df.groupby(["domain", "split_regime", "layer"], sort=True):
    fisher_euclidean = combine_pvalues(
        group["h1_p_perm_euclidean_vs_rewire"].to_numpy(dtype=float), method="fisher"
    )
    fisher_geodesic = combine_pvalues(
        group["h1_p_perm_geodesic_vs_rewire"].to_numpy(dtype=float), method="fisher"
    )
    fisher_distortion = combine_pvalues(
        group["distortion_p_lower"].to_numpy(dtype=float), method="fisher"
    )

    layer_summary_records.append(
        {
            "domain": keys[0],
            "split_regime": keys[1],
            "layer": int(keys[2]),
            "n_seed_runs": int(group.shape[0]),
            "mean_h1_observed_euclidean": float(group["h1_observed_euclidean"].mean()),
            "mean_h1_observed_geodesic": float(group["h1_observed_geodesic"].mean()),
            "mean_h1_rewire_null": float(group["h1_rewire_null_mean"].mean()),
            "mean_h1_delta_euclidean_vs_rewire": float(
                group["h1_delta_euclidean_vs_rewire"].mean()
            ),
            "mean_h1_delta_geodesic_vs_rewire": float(
                group["h1_delta_geodesic_vs_rewire"].mean()
            ),
            "mean_h1_delta_shift_geodesic_minus_euclidean": float(
                group["h1_delta_shift_geodesic_minus_euclidean"].mean()
            ),
            "frac_seeds_geo_delta_positive": float(
                (group["h1_delta_geodesic_vs_rewire"] > 0).mean()
            ),
            "frac_seeds_euclid_delta_positive": float(
                (group["h1_delta_euclidean_vs_rewire"] > 0).mean()
            ),
            "fisher_stat_euclidean_vs_rewire": float(fisher_euclidean[0]),
            "fisher_p_euclidean_vs_rewire": float(fisher_euclidean[1]),
            "fisher_stat_geodesic_vs_rewire": float(fisher_geodesic[0]),
            "fisher_p_geodesic_vs_rewire": float(fisher_geodesic[1]),
            "mean_distortion_observed": float(group["distortion_observed"].mean()),
            "mean_distortion_rewire_null": float(group["distortion_rewire_null_mean"].mean()),
            "mean_distortion_delta_observed_minus_null": float(
                group["distortion_delta_observed_minus_null"].mean()
            ),
            "fisher_stat_distortion_lower": float(fisher_distortion[0]),
            "fisher_p_distortion_lower": float(fisher_distortion[1]),
            "mean_knn_k": float(group["knn_k"].mean()),
            "bridged_runs": int(group["used_component_bridging"].sum()),
        }
    )

layer_summary_df = pd.DataFrame(layer_summary_records).sort_values(
    ["domain", "split_regime", "layer"]
)
layer_summary_path = ITER_DIR / "h1_immune_metric_matched_layer_summary.csv"
layer_summary_df.to_csv(layer_summary_path, index=False)


pass_matrix_records: list[dict[str, object]] = []
for domain, domain_df in layer_summary_df.groupby("domain", sort=True):
    for layer, layer_df in domain_df.groupby("layer", sort=True):
        source_row = layer_df[layer_df["split_regime"] == "source_disjoint"]
        target_row = layer_df[layer_df["split_regime"] == "target_disjoint"]
        if source_row.empty or target_row.empty:
            continue

        source_geo_p = float(source_row.iloc[0]["fisher_p_geodesic_vs_rewire"])
        target_geo_p = float(target_row.iloc[0]["fisher_p_geodesic_vs_rewire"])
        source_geo_delta = float(source_row.iloc[0]["mean_h1_delta_geodesic_vs_rewire"])
        target_geo_delta = float(target_row.iloc[0]["mean_h1_delta_geodesic_vs_rewire"])

        pass_matrix_records.append(
            {
                "domain": domain,
                "layer": int(layer),
                "source_fisher_p_geodesic_vs_rewire": source_geo_p,
                "target_fisher_p_geodesic_vs_rewire": target_geo_p,
                "source_mean_h1_delta_geodesic_vs_rewire": source_geo_delta,
                "target_mean_h1_delta_geodesic_vs_rewire": target_geo_delta,
                "source_sig_geodesic": bool(source_geo_p < 0.05),
                "target_sig_geodesic": bool(target_geo_p < 0.05),
                "both_splits_sig_geodesic": bool((source_geo_p < 0.05) and (target_geo_p < 0.05)),
                "both_splits_positive_geo_delta": bool(
                    (source_geo_delta > 0.0) and (target_geo_delta > 0.0)
                ),
            }
        )

pass_matrix_df = pd.DataFrame(pass_matrix_records).sort_values(["domain", "layer"])
pass_matrix_path = ITER_DIR / "h1_immune_metric_matched_pass_matrix.csv"
pass_matrix_df.to_csv(pass_matrix_path, index=False)


domain_summary_records: list[dict[str, object]] = []
for keys, group in layer_summary_df.groupby(["domain", "split_regime"], sort=True):
    n_layers = int(group.shape[0])
    n_geo_sig = int((group["fisher_p_geodesic_vs_rewire"] < 0.05).sum())
    n_geo_pos = int((group["mean_h1_delta_geodesic_vs_rewire"] > 0.0).sum())
    n_euclid_sig = int((group["fisher_p_euclidean_vs_rewire"] < 0.05).sum())
    n_euclid_pos = int((group["mean_h1_delta_euclidean_vs_rewire"] > 0.0).sum())
    n_shift_pos = int((group["mean_h1_delta_shift_geodesic_minus_euclidean"] > 0.0).sum())
    n_distortion_lower_sig = int((group["fisher_p_distortion_lower"] < 0.05).sum())

    domain_summary_records.append(
        {
            "domain": keys[0],
            "split_regime": keys[1],
            "n_layers": n_layers,
            "n_layers_geo_sig": n_geo_sig,
            "frac_layers_geo_sig": float(n_geo_sig / n_layers if n_layers else np.nan),
            "n_layers_geo_positive_delta": n_geo_pos,
            "frac_layers_geo_positive_delta": float(n_geo_pos / n_layers if n_layers else np.nan),
            "n_layers_euclid_sig": n_euclid_sig,
            "frac_layers_euclid_sig": float(n_euclid_sig / n_layers if n_layers else np.nan),
            "n_layers_euclid_positive_delta": n_euclid_pos,
            "frac_layers_euclid_positive_delta": float(
                n_euclid_pos / n_layers if n_layers else np.nan
            ),
            "n_layers_delta_shift_positive": n_shift_pos,
            "frac_layers_delta_shift_positive": float(
                n_shift_pos / n_layers if n_layers else np.nan
            ),
            "mean_h1_delta_geodesic_vs_rewire": float(
                group["mean_h1_delta_geodesic_vs_rewire"].mean()
            ),
            "mean_h1_delta_euclidean_vs_rewire": float(
                group["mean_h1_delta_euclidean_vs_rewire"].mean()
            ),
            "mean_h1_delta_shift_geodesic_minus_euclidean": float(
                group["mean_h1_delta_shift_geodesic_minus_euclidean"].mean()
            ),
            "n_layers_distortion_lower_sig": n_distortion_lower_sig,
            "frac_layers_distortion_lower_sig": float(
                n_distortion_lower_sig / n_layers if n_layers else np.nan
            ),
            "mean_distortion_delta_observed_minus_null": float(
                group["mean_distortion_delta_observed_minus_null"].mean()
            ),
        }
    )


domain_summary_df = pd.DataFrame(domain_summary_records).sort_values(
    ["domain", "split_regime"]
)
domain_summary_path = ITER_DIR / "h1_immune_metric_matched_domain_summary.csv"
domain_summary_df.to_csv(domain_summary_path, index=False)


calibration_shift_records: list[dict[str, object]] = []
for domain, domain_df in by_seed_df.groupby("domain", sort=True):
    for split_regime, split_df in domain_df.groupby("split_regime", sort=True):
        calibration_shift_records.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(split_df.shape[0]),
                "mean_shift": float(
                    split_df["h1_delta_shift_geodesic_minus_euclidean"].mean()
                ),
                "median_shift": float(
                    split_df["h1_delta_shift_geodesic_minus_euclidean"].median()
                ),
                "min_shift": float(
                    split_df["h1_delta_shift_geodesic_minus_euclidean"].min()
                ),
                "max_shift": float(
                    split_df["h1_delta_shift_geodesic_minus_euclidean"].max()
                ),
                "frac_rows_shift_positive": float(
                    (split_df["h1_delta_shift_geodesic_minus_euclidean"] > 0.0).mean()
                ),
            }
        )

calibration_shift_df = pd.DataFrame(calibration_shift_records).sort_values(
    ["domain", "split_regime"]
)
calibration_shift_path = ITER_DIR / "h1_immune_metric_calibration_shift_summary.csv"
calibration_shift_df.to_csv(calibration_shift_path, index=False)


summary_payload = {
    "config": {
        "domain": "immune",
        "split_regimes": list(split_specs.keys()),
        "n_points_per_test": n_points,
        "pca_dim": pca_dim,
        "knn_k_min": knn_k_min,
        "knn_k_max": knn_k_max,
        "n_null_degree_preserving_rewire": n_null_rewire,
        "rewire_swap_multiplier": rewire_swap_multiplier,
        "rewire_retry_per_null": rewire_retry_per_null,
        "rewire_attempt_multiplier": rewire_attempt_multiplier,
        "observed_metrics": [
            "h1_observed_euclidean",
            "h1_observed_geodesic",
            "distortion_observed",
        ],
    },
    "inputs": {
        domain: {seed_tag: str(path) for seed_tag, path in runs.items()}
        for domain, runs in domain_runs.items()
    },
    "domain_summary": domain_summary_df.to_dict(orient="records"),
    "calibration_shift_summary": calibration_shift_df.to_dict(orient="records"),
    "artifacts": {
        "by_seed": str(by_seed_path),
        "layer_summary": str(layer_summary_path),
        "pass_matrix": str(pass_matrix_path),
        "domain_summary": str(domain_summary_path),
        "calibration_shift_summary": str(calibration_shift_path),
    },
}

summary_path = ITER_DIR / "iter0007_screen_summary.json"
summary_path.write_text(json.dumps(summary_payload, indent=2))
print(json.dumps(summary_payload, indent=2))
