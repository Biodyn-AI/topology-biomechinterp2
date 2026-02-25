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


ITER_DIR = Path("iterations/iter_0008")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def compute_h1_sum(points_or_distances: np.ndarray, distance_matrix: bool = False) -> float:
    """Return the sum of finite H1 lifetimes from ripser persistence diagrams."""
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


def summarize_against_null(observed: float, null_values: np.ndarray) -> dict[str, float]:
    """Effect-size summary for an observed scalar against null draws."""
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


def connect_components_with_bridges(graph: nx.Graph, pairwise_euclidean: np.ndarray) -> nx.Graph:
    """Connect graph components by repeatedly adding nearest cross-component edges."""
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
            raise RuntimeError("Failed to identify a valid bridge edge.")

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

    raise RuntimeError("Unable to produce connected graph after kNN + bridge fallback.")


def graph_to_geodesic_distances(graph: nx.Graph, pairwise_euclidean: np.ndarray) -> np.ndarray:
    """Assign Euclidean weights to graph edges and compute all-pairs geodesic distances."""
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


def quantile_boundaries(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Build monotonic quantile boundaries used for edge-length binning."""
    if n_bins < 2:
        return np.asarray([], dtype=float)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    boundaries = np.quantile(values, quantiles[1:-1], method="linear")
    return np.unique(np.asarray(boundaries, dtype=float))


def edge_length_histogram(
    graph: nx.Graph,
    pairwise_euclidean: np.ndarray,
    boundaries: np.ndarray,
) -> np.ndarray:
    """Compute edge-count histogram across edge-length quantile bins."""
    n_bins_effective = int(boundaries.size + 1)
    counts = np.zeros(n_bins_effective, dtype=int)
    for u, v in graph.edges():
        length = float(pairwise_euclidean[u, v])
        bin_idx = int(np.digitize(length, boundaries, right=True))
        counts[bin_idx] += 1
    return counts


def histogram_l1_ratio(counts_a: np.ndarray, counts_b: np.ndarray, n_edges: int) -> float:
    """L1 distance between edge-length histograms, normalized by number of edges."""
    return float(np.abs(counts_a - counts_b).sum() / max(1, n_edges))


def rewire_degree_preserving_connected(
    base_graph: nx.Graph,
    rng: np.random.Generator,
    swap_multiplier: float,
    max_retries: int,
) -> tuple[nx.Graph, int]:
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
            return rewired, n_swaps

    raise RuntimeError("Failed to obtain connected degree-preserving rewire graph.")


def rewire_degree_preserving_quantile_constrained_connected(
    base_graph: nx.Graph,
    pairwise_euclidean: np.ndarray,
    boundaries: np.ndarray,
    base_histogram: np.ndarray,
    rng: np.random.Generator,
    swap_multiplier: float,
    candidate_trials: int,
    target_l1_ratio: float,
) -> tuple[nx.Graph, float, int]:
    """Rewire with degree-preserving swaps and keep the best edge-length-histogram match."""
    n_edges = base_graph.number_of_edges()
    n_swaps = max(1, int(round(n_edges * swap_multiplier)))
    max_tries = max(n_swaps * 20, 200)
    best_graph: nx.Graph | None = None
    best_l1_ratio = float("inf")

    for _ in range(candidate_trials):
        rewired = base_graph.copy()
        seed = int(rng.integers(0, 2_147_483_647))
        try:
            nx.double_edge_swap(rewired, nswap=n_swaps, max_tries=max_tries, seed=seed)
        except Exception:
            continue
        if not nx.is_connected(rewired):
            continue

        hist = edge_length_histogram(rewired, pairwise_euclidean, boundaries)
        l1_ratio = histogram_l1_ratio(base_histogram, hist, n_edges=n_edges)
        if l1_ratio < best_l1_ratio:
            best_graph = rewired
            best_l1_ratio = l1_ratio
        if l1_ratio <= target_l1_ratio:
            break

    if best_graph is None:
        raise RuntimeError("Failed to obtain any connected quantile-constrained rewired graph.")
    return best_graph, best_l1_ratio, n_swaps


def knn_bucket_label(k_value: int) -> str:
    """Bucket the kNN level for bridge-conditioned diagnostics."""
    if k_value <= 20:
        return "k_le_20"
    if k_value <= 30:
        return "k_21_30"
    return "k_gt_30"


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

n_points = 150
pca_dim = 14
knn_k_min = 10
knn_k_max = 40
n_null_rewire = 5
rewire_swap_multiplier = 0.12
rewire_retry_per_null = 16
rewire_attempt_multiplier = 20
edge_length_quantile_bins = 5
quantile_constraint_target_l1_ratio = 0.22

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
                    1_080_000
                    + domain_index * 100_000
                    + seed_index * 10_000
                    + split_index * 1_000
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
                    random_state=1_090_000
                    + domain_index * 100
                    + seed_index * 10
                    + split_index * 3
                    + layer,
                ).fit_transform(points)

                pairwise = pairwise_distances(points_pca, metric="euclidean")
                base_graph, knn_k_used, used_component_bridging = build_connected_knn_graph(
                    points=points_pca,
                    pairwise_euclidean=pairwise,
                    min_neighbors=knn_k_min,
                    max_neighbors=knn_k_max,
                )
                base_geodesic = graph_to_geodesic_distances(base_graph, pairwise)
                h1_observed_geodesic = compute_h1_sum(base_geodesic, distance_matrix=True)
                distortion_observed = mean_geodesic_distortion(base_geodesic, pairwise)

                base_edge_lengths = np.asarray(
                    [pairwise[u, v] for u, v in base_graph.edges()],
                    dtype=float,
                )
                boundaries = quantile_boundaries(base_edge_lengths, edge_length_quantile_bins)
                base_hist = edge_length_histogram(base_graph, pairwise, boundaries)

                unconstrained_h1_values: list[float] = []
                unconstrained_distortion_values: list[float] = []
                unconstrained_l1_values: list[float] = []
                unconstrained_total_attempts = 0
                unconstrained_max_attempts = n_null_rewire * rewire_attempt_multiplier

                while (
                    len(unconstrained_h1_values) < n_null_rewire
                    and unconstrained_total_attempts < unconstrained_max_attempts
                ):
                    unconstrained_total_attempts += 1
                    rw_rng = np.random.default_rng(
                        1_100_000
                        + domain_index * 100_000
                        + seed_index * 10_000
                        + split_index * 1_000
                        + layer * 100
                        + unconstrained_total_attempts
                    )
                    try:
                        rewired_graph, _ = rewire_degree_preserving_connected(
                            base_graph=base_graph,
                            rng=rw_rng,
                            swap_multiplier=rewire_swap_multiplier,
                            max_retries=rewire_retry_per_null,
                        )
                        rewired_geodesic = graph_to_geodesic_distances(rewired_graph, pairwise)
                        unconstrained_h1_values.append(
                            compute_h1_sum(rewired_geodesic, distance_matrix=True)
                        )
                        unconstrained_distortion_values.append(
                            mean_geodesic_distortion(rewired_geodesic, pairwise)
                        )
                        rewired_hist = edge_length_histogram(rewired_graph, pairwise, boundaries)
                        unconstrained_l1_values.append(
                            histogram_l1_ratio(base_hist, rewired_hist, n_edges=base_graph.number_of_edges())
                        )
                    except Exception:
                        continue

                constrained_h1_values: list[float] = []
                constrained_distortion_values: list[float] = []
                constrained_l1_values: list[float] = []
                constrained_total_attempts = 0
                constrained_max_attempts = n_null_rewire * rewire_attempt_multiplier

                while (
                    len(constrained_h1_values) < n_null_rewire
                    and constrained_total_attempts < constrained_max_attempts
                ):
                    constrained_total_attempts += 1
                    rw_rng = np.random.default_rng(
                        1_200_000
                        + domain_index * 100_000
                        + seed_index * 10_000
                        + split_index * 1_000
                        + layer * 100
                        + constrained_total_attempts
                    )
                    try:
                        rewired_graph, l1_ratio, _ = (
                            rewire_degree_preserving_quantile_constrained_connected(
                                base_graph=base_graph,
                                pairwise_euclidean=pairwise,
                                boundaries=boundaries,
                                base_histogram=base_hist,
                                rng=rw_rng,
                                swap_multiplier=rewire_swap_multiplier,
                                candidate_trials=rewire_retry_per_null,
                                target_l1_ratio=quantile_constraint_target_l1_ratio,
                            )
                        )
                        rewired_geodesic = graph_to_geodesic_distances(rewired_graph, pairwise)
                        constrained_h1_values.append(
                            compute_h1_sum(rewired_geodesic, distance_matrix=True)
                        )
                        constrained_distortion_values.append(
                            mean_geodesic_distortion(rewired_geodesic, pairwise)
                        )
                        constrained_l1_values.append(float(l1_ratio))
                    except Exception:
                        continue

                if len(unconstrained_h1_values) < 2 or len(constrained_h1_values) < 2:
                    raise RuntimeError(
                        "Insufficient null draws for stable estimates: "
                        f"domain={domain} seed={seed_tag} split={split_regime} layer={layer} "
                        f"n_unconstrained={len(unconstrained_h1_values)} "
                        f"n_constrained={len(constrained_h1_values)}"
                    )

                unconstrained_h1 = np.asarray(unconstrained_h1_values, dtype=float)
                unconstrained_distortion = np.asarray(unconstrained_distortion_values, dtype=float)
                constrained_h1 = np.asarray(constrained_h1_values, dtype=float)
                constrained_distortion = np.asarray(constrained_distortion_values, dtype=float)

                unconstrained_h1_stats = summarize_against_null(
                    observed=h1_observed_geodesic,
                    null_values=unconstrained_h1,
                )
                constrained_h1_stats = summarize_against_null(
                    observed=h1_observed_geodesic,
                    null_values=constrained_h1,
                )
                unconstrained_distortion_stats = summarize_against_null(
                    observed=distortion_observed,
                    null_values=unconstrained_distortion,
                )
                constrained_distortion_stats = summarize_against_null(
                    observed=distortion_observed,
                    null_values=constrained_distortion,
                )

                common_payload = {
                    "domain": domain,
                    "seed_tag": seed_tag,
                    "split_regime": split_regime,
                    "layer": int(layer),
                    "n_genes_pool": int(candidate_pool.size),
                    "n_points": int(points_pca.shape[0]),
                    "pca_dim": int(n_components),
                    "knn_k": int(knn_k_used),
                    "knn_bucket": knn_bucket_label(int(knn_k_used)),
                    "used_component_bridging": bool(used_component_bridging),
                    "h1_observed_geodesic": float(h1_observed_geodesic),
                    "distortion_observed": float(distortion_observed),
                }

                records.append(
                    {
                        **common_payload,
                        "null_family": "degree_preserving_geodesic_rewire",
                        "n_null_draws": int(unconstrained_h1.size),
                        "rewire_attempts_total": int(unconstrained_total_attempts),
                        "mean_edge_hist_l1_ratio": float(np.mean(unconstrained_l1_values)),
                        "h1_null_mean": unconstrained_h1_stats["null_mean"],
                        "h1_null_std": unconstrained_h1_stats["null_std"],
                        "h1_delta_observed_minus_null": unconstrained_h1_stats["delta"],
                        "h1_z_observed_minus_null": unconstrained_h1_stats["z_score"],
                        "h1_p_perm_upper": empirical_upper_tail_p(
                            h1_observed_geodesic,
                            unconstrained_h1,
                        ),
                        "distortion_null_mean": unconstrained_distortion_stats["null_mean"],
                        "distortion_null_std": unconstrained_distortion_stats["null_std"],
                        "distortion_delta_observed_minus_null": unconstrained_distortion_stats[
                            "delta"
                        ],
                        "distortion_z_observed_minus_null": unconstrained_distortion_stats[
                            "z_score"
                        ],
                        "distortion_p_lower": empirical_lower_tail_p(
                            distortion_observed,
                            unconstrained_distortion,
                        ),
                    }
                )

                records.append(
                    {
                        **common_payload,
                        "null_family": "quantile_constrained_geodesic_rewire",
                        "n_null_draws": int(constrained_h1.size),
                        "rewire_attempts_total": int(constrained_total_attempts),
                        "mean_edge_hist_l1_ratio": float(np.mean(constrained_l1_values)),
                        "h1_null_mean": constrained_h1_stats["null_mean"],
                        "h1_null_std": constrained_h1_stats["null_std"],
                        "h1_delta_observed_minus_null": constrained_h1_stats["delta"],
                        "h1_z_observed_minus_null": constrained_h1_stats["z_score"],
                        "h1_p_perm_upper": empirical_upper_tail_p(
                            h1_observed_geodesic,
                            constrained_h1,
                        ),
                        "distortion_null_mean": constrained_distortion_stats["null_mean"],
                        "distortion_null_std": constrained_distortion_stats["null_std"],
                        "distortion_delta_observed_minus_null": constrained_distortion_stats[
                            "delta"
                        ],
                        "distortion_z_observed_minus_null": constrained_distortion_stats[
                            "z_score"
                        ],
                        "distortion_p_lower": empirical_lower_tail_p(
                            distortion_observed,
                            constrained_distortion,
                        ),
                    }
                )


by_seed_df = pd.DataFrame(records).sort_values(
    ["domain", "split_regime", "layer", "null_family", "seed_tag"]
).reset_index(drop=True)
by_seed_path = ITER_DIR / "h1_immune_constrained_rewire_by_seed_layer.csv"
by_seed_df.to_csv(by_seed_path, index=False)


layer_summary_records: list[dict[str, object]] = []
for keys, group in by_seed_df.groupby(["domain", "split_regime", "layer", "null_family"], sort=True):
    fisher_h1 = combine_pvalues(group["h1_p_perm_upper"].to_numpy(dtype=float), method="fisher")
    fisher_dist = combine_pvalues(group["distortion_p_lower"].to_numpy(dtype=float), method="fisher")
    layer_summary_records.append(
        {
            "domain": keys[0],
            "split_regime": keys[1],
            "layer": int(keys[2]),
            "null_family": keys[3],
            "n_seed_runs": int(group.shape[0]),
            "mean_h1_observed_geodesic": float(group["h1_observed_geodesic"].mean()),
            "mean_h1_null": float(group["h1_null_mean"].mean()),
            "mean_h1_delta_observed_minus_null": float(
                group["h1_delta_observed_minus_null"].mean()
            ),
            "frac_seed_runs_h1_delta_positive": float(
                (group["h1_delta_observed_minus_null"] > 0.0).mean()
            ),
            "fisher_stat_h1_upper": float(fisher_h1[0]),
            "fisher_p_h1_upper": float(fisher_h1[1]),
            "mean_distortion_observed": float(group["distortion_observed"].mean()),
            "mean_distortion_null": float(group["distortion_null_mean"].mean()),
            "mean_distortion_delta_observed_minus_null": float(
                group["distortion_delta_observed_minus_null"].mean()
            ),
            "fisher_stat_distortion_lower": float(fisher_dist[0]),
            "fisher_p_distortion_lower": float(fisher_dist[1]),
            "mean_knn_k": float(group["knn_k"].mean()),
            "bridged_runs": int(group["used_component_bridging"].sum()),
            "mean_edge_hist_l1_ratio": float(
                group["mean_edge_hist_l1_ratio"].dropna().mean()
            )
            if group["mean_edge_hist_l1_ratio"].notna().any()
            else np.nan,
        }
    )

layer_summary_df = pd.DataFrame(layer_summary_records).sort_values(
    ["domain", "split_regime", "layer", "null_family"]
).reset_index(drop=True)
layer_summary_path = ITER_DIR / "h1_immune_constrained_rewire_layer_summary.csv"
layer_summary_df.to_csv(layer_summary_path, index=False)


pass_matrix_records: list[dict[str, object]] = []
for null_family, family_df in layer_summary_df.groupby("null_family", sort=True):
    for layer, layer_df in family_df.groupby("layer", sort=True):
        source_row = layer_df[layer_df["split_regime"] == "source_disjoint"]
        target_row = layer_df[layer_df["split_regime"] == "target_disjoint"]
        if source_row.empty or target_row.empty:
            continue

        source_p = float(source_row.iloc[0]["fisher_p_h1_upper"])
        target_p = float(target_row.iloc[0]["fisher_p_h1_upper"])
        source_delta = float(source_row.iloc[0]["mean_h1_delta_observed_minus_null"])
        target_delta = float(target_row.iloc[0]["mean_h1_delta_observed_minus_null"])

        pass_matrix_records.append(
            {
                "domain": "immune",
                "layer": int(layer),
                "null_family": null_family,
                "source_fisher_p_h1_upper": source_p,
                "target_fisher_p_h1_upper": target_p,
                "source_mean_h1_delta_observed_minus_null": source_delta,
                "target_mean_h1_delta_observed_minus_null": target_delta,
                "source_sig_h1": bool(source_p < 0.05),
                "target_sig_h1": bool(target_p < 0.05),
                "both_splits_sig_h1": bool((source_p < 0.05) and (target_p < 0.05)),
                "both_splits_positive_h1_delta": bool(
                    (source_delta > 0.0) and (target_delta > 0.0)
                ),
            }
        )

pass_matrix_df = pd.DataFrame(pass_matrix_records).sort_values(["null_family", "layer"])
pass_matrix_path = ITER_DIR / "h1_immune_constrained_rewire_pass_matrix.csv"
pass_matrix_df.to_csv(pass_matrix_path, index=False)


domain_summary_records: list[dict[str, object]] = []
for keys, group in layer_summary_df.groupby(["domain", "split_regime", "null_family"], sort=True):
    n_layers = int(group.shape[0])
    n_h1_sig = int((group["fisher_p_h1_upper"] < 0.05).sum())
    n_h1_pos = int((group["mean_h1_delta_observed_minus_null"] > 0.0).sum())
    n_dist_sig = int((group["fisher_p_distortion_lower"] < 0.05).sum())
    domain_summary_records.append(
        {
            "domain": keys[0],
            "split_regime": keys[1],
            "null_family": keys[2],
            "n_layers": n_layers,
            "n_layers_h1_sig": n_h1_sig,
            "frac_layers_h1_sig": float(n_h1_sig / n_layers if n_layers else np.nan),
            "n_layers_h1_positive_delta": n_h1_pos,
            "frac_layers_h1_positive_delta": float(n_h1_pos / n_layers if n_layers else np.nan),
            "mean_h1_delta_observed_minus_null": float(
                group["mean_h1_delta_observed_minus_null"].mean()
            ),
            "median_h1_delta_observed_minus_null": float(
                group["mean_h1_delta_observed_minus_null"].median()
            ),
            "n_layers_distortion_lower_sig": n_dist_sig,
            "frac_layers_distortion_lower_sig": float(
                n_dist_sig / n_layers if n_layers else np.nan
            ),
            "mean_distortion_delta_observed_minus_null": float(
                group["mean_distortion_delta_observed_minus_null"].mean()
            ),
            "mean_knn_k": float(group["mean_knn_k"].mean()),
            "total_bridged_runs": int(group["bridged_runs"].sum()),
            "mean_edge_hist_l1_ratio": float(group["mean_edge_hist_l1_ratio"].dropna().mean())
            if group["mean_edge_hist_l1_ratio"].notna().any()
            else np.nan,
        }
    )

domain_summary_df = pd.DataFrame(domain_summary_records).sort_values(
    ["domain", "split_regime", "null_family"]
).reset_index(drop=True)
domain_summary_path = ITER_DIR / "h1_immune_constrained_rewire_domain_summary.csv"
domain_summary_df.to_csv(domain_summary_path, index=False)


bridge_k_strata_records: list[dict[str, object]] = []
for keys, group in by_seed_df.groupby(
    ["domain", "null_family", "used_component_bridging", "knn_bucket"],
    sort=True,
):
    bridge_k_strata_records.append(
        {
            "domain": keys[0],
            "null_family": keys[1],
            "used_component_bridging": bool(keys[2]),
            "knn_bucket": keys[3],
            "n_rows": int(group.shape[0]),
            "mean_h1_delta_observed_minus_null": float(
                group["h1_delta_observed_minus_null"].mean()
            ),
            "median_h1_delta_observed_minus_null": float(
                group["h1_delta_observed_minus_null"].median()
            ),
            "frac_rows_h1_positive_delta": float(
                (group["h1_delta_observed_minus_null"] > 0.0).mean()
            ),
            "mean_distortion_delta_observed_minus_null": float(
                group["distortion_delta_observed_minus_null"].mean()
            ),
            "mean_edge_hist_l1_ratio": float(group["mean_edge_hist_l1_ratio"].dropna().mean())
            if group["mean_edge_hist_l1_ratio"].notna().any()
            else np.nan,
        }
    )

bridge_k_strata_df = pd.DataFrame(bridge_k_strata_records).sort_values(
    ["domain", "null_family", "used_component_bridging", "knn_bucket"]
).reset_index(drop=True)
bridge_k_strata_path = ITER_DIR / "h1_immune_constrained_rewire_bridge_k_strata_summary.csv"
bridge_k_strata_df.to_csv(bridge_k_strata_path, index=False)


bridge_gap_records: list[dict[str, object]] = []
for keys, group in by_seed_df.groupby(["domain", "null_family"], sort=True):
    bridged = group[group["used_component_bridging"]]
    non_bridged = group[~group["used_component_bridging"]]
    bridge_gap_records.append(
        {
            "domain": keys[0],
            "null_family": keys[1],
            "n_rows_total": int(group.shape[0]),
            "n_rows_bridged": int(bridged.shape[0]),
            "n_rows_non_bridged": int(non_bridged.shape[0]),
            "mean_h1_delta_bridged": float(bridged["h1_delta_observed_minus_null"].mean())
            if not bridged.empty
            else np.nan,
            "mean_h1_delta_non_bridged": float(non_bridged["h1_delta_observed_minus_null"].mean())
            if not non_bridged.empty
            else np.nan,
            "bridge_minus_nonbridge_h1_delta_gap": float(
                bridged["h1_delta_observed_minus_null"].mean()
                - non_bridged["h1_delta_observed_minus_null"].mean()
            )
            if (not bridged.empty and not non_bridged.empty)
            else np.nan,
            "mean_distortion_delta_bridged": float(
                bridged["distortion_delta_observed_minus_null"].mean()
            )
            if not bridged.empty
            else np.nan,
            "mean_distortion_delta_non_bridged": float(
                non_bridged["distortion_delta_observed_minus_null"].mean()
            )
            if not non_bridged.empty
            else np.nan,
            "bridge_minus_nonbridge_distortion_delta_gap": float(
                bridged["distortion_delta_observed_minus_null"].mean()
                - non_bridged["distortion_delta_observed_minus_null"].mean()
            )
            if (not bridged.empty and not non_bridged.empty)
            else np.nan,
        }
    )

bridge_gap_df = pd.DataFrame(bridge_gap_records).sort_values(["domain", "null_family"]).reset_index(
    drop=True
)
bridge_gap_path = ITER_DIR / "h1_immune_constrained_rewire_bridge_gap_summary.csv"
bridge_gap_df.to_csv(bridge_gap_path, index=False)


paired_calibration_df = by_seed_df.pivot_table(
    index=[
        "domain",
        "seed_tag",
        "split_regime",
        "layer",
        "knn_k",
        "knn_bucket",
        "used_component_bridging",
    ],
    columns="null_family",
    values=[
        "h1_delta_observed_minus_null",
        "distortion_delta_observed_minus_null",
    ],
).reset_index()

paired_calibration_df.columns = [
    "_".join([str(part) for part in col if part != ""]).strip("_")
    if isinstance(col, tuple)
    else str(col)
    for col in paired_calibration_df.columns
]

paired_calibration_df["h1_delta_shift_constrained_minus_unconstrained"] = (
    paired_calibration_df["h1_delta_observed_minus_null_quantile_constrained_geodesic_rewire"]
    - paired_calibration_df["h1_delta_observed_minus_null_degree_preserving_geodesic_rewire"]
)
paired_calibration_df["distortion_delta_shift_constrained_minus_unconstrained"] = (
    paired_calibration_df[
        "distortion_delta_observed_minus_null_quantile_constrained_geodesic_rewire"
    ]
    - paired_calibration_df["distortion_delta_observed_minus_null_degree_preserving_geodesic_rewire"]
)
paired_calibration_path = ITER_DIR / "h1_immune_constrained_rewire_paired_shift_by_seed_layer.csv"
paired_calibration_df.to_csv(paired_calibration_path, index=False)


paired_shift_summary_records: list[dict[str, object]] = []
for keys, group in paired_calibration_df.groupby(["domain", "split_regime"], sort=True):
    paired_shift_summary_records.append(
        {
            "domain": keys[0],
            "split_regime": keys[1],
            "n_rows": int(group.shape[0]),
            "mean_h1_delta_shift_constrained_minus_unconstrained": float(
                group["h1_delta_shift_constrained_minus_unconstrained"].mean()
            ),
            "median_h1_delta_shift_constrained_minus_unconstrained": float(
                group["h1_delta_shift_constrained_minus_unconstrained"].median()
            ),
            "frac_rows_h1_shift_positive": float(
                (group["h1_delta_shift_constrained_minus_unconstrained"] > 0.0).mean()
            ),
            "mean_distortion_delta_shift_constrained_minus_unconstrained": float(
                group["distortion_delta_shift_constrained_minus_unconstrained"].mean()
            ),
            "frac_rows_distortion_shift_negative": float(
                (group["distortion_delta_shift_constrained_minus_unconstrained"] < 0.0).mean()
            ),
        }
    )

paired_shift_summary_df = pd.DataFrame(paired_shift_summary_records).sort_values(
    ["domain", "split_regime"]
)
paired_shift_summary_path = ITER_DIR / "h1_immune_constrained_rewire_paired_shift_summary.csv"
paired_shift_summary_df.to_csv(paired_shift_summary_path, index=False)


summary_payload = {
    "config": {
        "domain": "immune",
        "split_regimes": list(split_specs.keys()),
        "n_points_per_test": n_points,
        "pca_dim": pca_dim,
        "knn_k_min": knn_k_min,
        "knn_k_max": knn_k_max,
        "n_null_rewire_per_family": n_null_rewire,
        "rewire_swap_multiplier": rewire_swap_multiplier,
        "rewire_retry_per_null": rewire_retry_per_null,
        "rewire_attempt_multiplier": rewire_attempt_multiplier,
        "edge_length_quantile_bins": edge_length_quantile_bins,
        "quantile_constraint_target_l1_ratio": quantile_constraint_target_l1_ratio,
    },
    "inputs": {
        domain: {seed_tag: str(path) for seed_tag, path in runs.items()}
        for domain, runs in domain_runs.items()
    },
    "domain_summary": domain_summary_df.to_dict(orient="records"),
    "bridge_gap_summary": bridge_gap_df.to_dict(orient="records"),
    "paired_shift_summary": paired_shift_summary_df.to_dict(orient="records"),
    "artifacts": {
        "by_seed": str(by_seed_path),
        "layer_summary": str(layer_summary_path),
        "pass_matrix": str(pass_matrix_path),
        "domain_summary": str(domain_summary_path),
        "bridge_k_strata": str(bridge_k_strata_path),
        "bridge_gap_summary": str(bridge_gap_path),
        "paired_shift_by_seed_layer": str(paired_calibration_path),
        "paired_shift_summary": str(paired_shift_summary_path),
    },
}

summary_path = ITER_DIR / "iter0008_screen_summary.json"
summary_path.write_text(json.dumps(summary_payload, indent=2))
print(json.dumps(summary_payload, indent=2))
