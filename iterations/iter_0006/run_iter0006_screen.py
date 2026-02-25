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


ITER_DIR = Path("iterations/iter_0006")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def compute_h1_sum(points_or_distances: np.ndarray, distance_matrix: bool = False) -> float:
    """Return total finite H1 lifetime from ripser persistence diagrams."""
    diagrams = ripser(points_or_distances, maxdim=1, distance_matrix=distance_matrix)["dgms"]
    h1 = diagrams[1]
    finite = h1[np.isfinite(h1[:, 1])]
    if finite.size == 0:
        return 0.0
    lifetimes = finite[:, 1] - finite[:, 0]
    return float(lifetimes.sum())


def feature_shuffle_null(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle each PCA feature independently across genes."""
    shuffled = np.empty_like(points)
    for feature_idx in range(points.shape[1]):
        shuffled[:, feature_idx] = points[rng.permutation(points.shape[0]), feature_idx]
    return shuffled


def null_stats(observed: float, null_values: np.ndarray) -> dict[str, float]:
    """Effect-size and empirical p-value summary for one observed-vs-null comparison."""
    null_mean = float(null_values.mean())
    null_std = float(null_values.std(ddof=1)) if null_values.size > 1 else 0.0
    delta = float(observed - null_mean)
    z_score = float(delta / (null_std + 1e-9))
    p_perm = float((1 + np.sum(null_values >= observed)) / (null_values.size + 1))
    return {
        "null_mean": null_mean,
        "null_std": null_std,
        "delta": delta,
        "z_score": z_score,
        "p_perm": p_perm,
    }


def connect_components_with_bridges(
    graph: nx.Graph,
    pairwise_euclidean: np.ndarray,
) -> nx.Graph:
    """Connect disconnected components using nearest cross-component bridges."""
    connected = graph.copy()
    components = [np.asarray(sorted(component), dtype=int) for component in nx.connected_components(connected)]
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
            raise RuntimeError("Failed to identify bridge edge between disconnected components.")

        connected.add_edge(best_u, best_v, weight=best_distance)
        components = [np.asarray(sorted(component), dtype=int) for component in nx.connected_components(connected)]

    return connected


def build_knn_graph(
    points: np.ndarray,
    pairwise_euclidean: np.ndarray,
    n_neighbors: int,
    max_neighbors: int,
) -> tuple[nx.Graph, int, bool]:
    """Build a connected symmetric kNN graph; increase k and bridge if needed."""
    n_points = points.shape[0]
    min_k = max(2, min(n_neighbors, n_points - 1))
    max_k = max(min_k, min(max_neighbors, n_points - 1))
    last_graph: nx.Graph | None = None

    for current_k in range(min_k, max_k + 1):
        knn = NearestNeighbors(n_neighbors=current_k + 1, metric="euclidean")
        knn.fit(points)
        distances, indices = knn.kneighbors(points)

        graph = nx.Graph()
        graph.add_nodes_from(range(n_points))

        # Drop the self-neighbor in column 0 and symmetrize by taking min edge length.
        for source in range(n_points):
            for dist, target in zip(distances[source, 1:], indices[source, 1:]):
                i, j = sorted((int(source), int(target)))
                if i == j:
                    continue
                weight = float(dist)
                if graph.has_edge(i, j):
                    if weight < graph[i][j]["weight"]:
                        graph[i][j]["weight"] = weight
                else:
                    graph.add_edge(i, j, weight=weight)

        if nx.is_connected(graph):
            return graph, current_k, False

        last_graph = graph

    if last_graph is None:
        raise RuntimeError("kNN graph construction failed before connectivity check.")

    bridged = connect_components_with_bridges(last_graph, pairwise_euclidean)
    if nx.is_connected(bridged):
        return bridged, max_k, True

    raise RuntimeError("Unable to produce connected graph after kNN + component-bridge fallback.")


def rewire_degree_preserving_connected(
    base_graph: nx.Graph,
    rng: np.random.Generator,
    swap_multiplier: float,
    max_retries: int,
) -> nx.Graph:
    """Rewire edges with double-edge swaps until a connected graph is produced."""
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

    raise RuntimeError("Unable to generate connected degree-preserving rewired graph.")


def graph_to_geodesic_distances(
    graph: nx.Graph,
    pairwise_euclidean: np.ndarray,
) -> np.ndarray:
    """Assign geometric weights on rewired edges and compute all-pairs geodesic distances."""
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
        raise RuntimeError("Shortest-path matrix contains non-finite values.")
    return np.asarray(distances, dtype=np.float64)


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

n_points = 180
pca_dim = 14
n_neighbors_knn = 12
max_neighbors_knn = 30
n_null_feature_shuffle = 20
n_null_rewire_geodesic = 8
rewire_swap_multiplier = 1.5
rewire_retry_per_null = 10
rewire_attempt_multiplier = 6

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
                    620000
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
                    random_state=9300
                    + domain_index * 100
                    + seed_index * 10
                    + split_index * 3
                    + layer,
                ).fit_transform(points)

                h1_observed = compute_h1_sum(points_pca, distance_matrix=False)

                feature_shuffle_null_values: list[float] = []
                for null_idx in range(n_null_feature_shuffle):
                    fs_rng = np.random.default_rng(
                        720000
                        + domain_index * 100000
                        + seed_index * 10000
                        + split_index * 1000
                        + layer * 100
                        + null_idx
                    )
                    null_points = feature_shuffle_null(points_pca, fs_rng)
                    feature_shuffle_null_values.append(compute_h1_sum(null_points))

                pairwise = pairwise_distances(points_pca, metric="euclidean")
                base_graph, knn_k_used, used_component_bridging = build_knn_graph(
                    points=points_pca,
                    pairwise_euclidean=pairwise,
                    n_neighbors=n_neighbors_knn,
                    max_neighbors=max_neighbors_knn,
                )
                rewire_null_values: list[float] = []
                rewire_total_attempts = 0
                rewire_target_attempts = n_null_rewire_geodesic * rewire_attempt_multiplier

                while (
                    len(rewire_null_values) < n_null_rewire_geodesic
                    and rewire_total_attempts < rewire_target_attempts
                ):
                    rewire_total_attempts += 1
                    rw_rng = np.random.default_rng(
                        820000
                        + domain_index * 100000
                        + seed_index * 10000
                        + split_index * 1000
                        + layer * 100
                        + rewire_total_attempts
                    )
                    try:
                        rewired_graph = rewire_degree_preserving_connected(
                            base_graph=base_graph,
                            rng=rw_rng,
                            swap_multiplier=rewire_swap_multiplier,
                            max_retries=rewire_retry_per_null,
                        )
                        rewired_distances = graph_to_geodesic_distances(rewired_graph, pairwise)
                        rewire_null_values.append(
                            compute_h1_sum(rewired_distances, distance_matrix=True)
                        )
                    except Exception:
                        continue

                if len(rewire_null_values) < 2:
                    raise RuntimeError(
                        "Insufficient rewiring null draws for stable variance estimate: "
                        f"domain={domain} seed={seed_tag} split={split_regime} layer={layer}"
                    )

                fs_stats = null_stats(
                    observed=h1_observed,
                    null_values=np.asarray(feature_shuffle_null_values, dtype=float),
                )
                rw_stats = null_stats(
                    observed=h1_observed,
                    null_values=np.asarray(rewire_null_values, dtype=float),
                )

                records.extend(
                    [
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "n_genes_pool": int(candidate_pool.size),
                            "n_points": int(points_pca.shape[0]),
                            "pca_dim": int(n_components),
                            "null_family": "feature_shuffle",
                            "n_null_draws": int(len(feature_shuffle_null_values)),
                            "rewire_attempts": 0,
                            "knn_k": int(knn_k_used),
                            "used_component_bridging": bool(used_component_bridging),
                            "h1_sum_observed": float(h1_observed),
                            "h1_sum_null_mean": fs_stats["null_mean"],
                            "h1_sum_null_std": fs_stats["null_std"],
                            "h1_sum_delta": fs_stats["delta"],
                            "h1_sum_z": fs_stats["z_score"],
                            "h1_sum_p_perm": fs_stats["p_perm"],
                        },
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "n_genes_pool": int(candidate_pool.size),
                            "n_points": int(points_pca.shape[0]),
                            "pca_dim": int(n_components),
                            "null_family": "degree_preserving_geodesic_rewire",
                            "n_null_draws": int(len(rewire_null_values)),
                            "rewire_attempts": int(rewire_total_attempts),
                            "knn_k": int(knn_k_used),
                            "used_component_bridging": bool(used_component_bridging),
                            "h1_sum_observed": float(h1_observed),
                            "h1_sum_null_mean": rw_stats["null_mean"],
                            "h1_sum_null_std": rw_stats["null_std"],
                            "h1_sum_delta": rw_stats["delta"],
                            "h1_sum_z": rw_stats["z_score"],
                            "h1_sum_p_perm": rw_stats["p_perm"],
                        },
                    ]
                )


by_seed_df = pd.DataFrame(records).sort_values(
    ["domain", "split_regime", "layer", "null_family", "seed_tag"]
)
by_seed_path = ITER_DIR / "h1_immune_rewire_split_by_seed_layer.csv"
by_seed_df.to_csv(by_seed_path, index=False)


layer_summary_records: list[dict[str, object]] = []
group_cols = ["domain", "split_regime", "layer", "null_family"]
for keys, group in by_seed_df.groupby(group_cols, sort=True):
    fisher_stat, fisher_p = combine_pvalues(group["h1_sum_p_perm"].to_numpy(dtype=float), method="fisher")
    layer_summary_records.append(
        {
            "domain": keys[0],
            "split_regime": keys[1],
            "layer": int(keys[2]),
            "null_family": keys[3],
            "n_seed_runs": int(group.shape[0]),
            "mean_h1_sum_observed": float(group["h1_sum_observed"].mean()),
            "mean_h1_sum_null_mean": float(group["h1_sum_null_mean"].mean()),
            "mean_h1_sum_delta": float(group["h1_sum_delta"].mean()),
            "mean_h1_sum_z": float(group["h1_sum_z"].mean()),
            "delta_positive_fraction": float((group["h1_sum_delta"] > 0).mean()),
            "mean_n_null_draws": float(group["n_null_draws"].mean()),
            "mean_rewire_attempts": float(group["rewire_attempts"].mean()),
            "fisher_stat": float(fisher_stat),
            "fisher_p": float(fisher_p),
        }
    )

layer_summary_df = pd.DataFrame(layer_summary_records).sort_values(group_cols).reset_index(drop=True)
layer_summary_path = ITER_DIR / "h1_immune_rewire_split_layer_summary.csv"
layer_summary_df.to_csv(layer_summary_path, index=False)


pass_matrix_records: list[dict[str, object]] = []
for null_family, null_df in layer_summary_df.groupby("null_family", sort=True):
    for layer, layer_df in null_df.groupby("layer", sort=True):
        source_row = layer_df[layer_df["split_regime"] == "source_disjoint"]
        target_row = layer_df[layer_df["split_regime"] == "target_disjoint"]
        if source_row.empty or target_row.empty:
            continue

        source_p = float(source_row.iloc[0]["fisher_p"])
        target_p = float(target_row.iloc[0]["fisher_p"])
        source_delta = float(source_row.iloc[0]["mean_h1_sum_delta"])
        target_delta = float(target_row.iloc[0]["mean_h1_sum_delta"])

        pass_matrix_records.append(
            {
                "domain": "immune",
                "layer": int(layer),
                "null_family": null_family,
                "source_fisher_p": source_p,
                "target_fisher_p": target_p,
                "source_mean_h1_sum_delta": source_delta,
                "target_mean_h1_sum_delta": target_delta,
                "source_sig": bool(source_p < 0.05),
                "target_sig": bool(target_p < 0.05),
                "both_splits_sig": bool((source_p < 0.05) and (target_p < 0.05)),
                "both_splits_positive_delta": bool((source_delta > 0.0) and (target_delta > 0.0)),
            }
        )

pass_matrix_df = pd.DataFrame(pass_matrix_records).sort_values(["null_family", "layer"]).reset_index(drop=True)
pass_matrix_path = ITER_DIR / "h1_immune_rewire_split_pass_matrix.csv"
pass_matrix_df.to_csv(pass_matrix_path, index=False)


domain_summary_records: list[dict[str, object]] = []
for keys, group in layer_summary_df.groupby(["domain", "split_regime", "null_family"], sort=True):
    n_tests = int(group.shape[0])
    n_sig = int((group["fisher_p"] < 0.05).sum())
    n_positive = int((group["mean_h1_sum_delta"] > 0.0).sum())
    domain_summary_records.append(
        {
            "domain": keys[0],
            "split_regime": keys[1],
            "null_family": keys[2],
            "n_layer_tests": n_tests,
            "n_layer_tests_fisher_p_lt_0_05": n_sig,
            "frac_layer_tests_fisher_p_lt_0_05": float(n_sig / n_tests if n_tests else np.nan),
            "n_layer_tests_positive_delta": n_positive,
            "frac_layer_tests_positive_delta": float(n_positive / n_tests if n_tests else np.nan),
            "mean_layer_delta": float(group["mean_h1_sum_delta"].mean()),
            "median_layer_delta": float(group["mean_h1_sum_delta"].median()),
            "min_layer_delta": float(group["mean_h1_sum_delta"].min()),
            "max_layer_delta": float(group["mean_h1_sum_delta"].max()),
            "mean_layer_z": float(group["mean_h1_sum_z"].mean()),
        }
    )

domain_summary_df = pd.DataFrame(domain_summary_records).sort_values(
    ["domain", "split_regime", "null_family"]
).reset_index(drop=True)
domain_summary_path = ITER_DIR / "h1_immune_rewire_split_domain_summary.csv"
domain_summary_df.to_csv(domain_summary_path, index=False)


dual_split_summary_records: list[dict[str, object]] = []
for null_family, group in pass_matrix_df.groupby("null_family", sort=True):
    dual_split_summary_records.append(
        {
            "domain": "immune",
            "null_family": null_family,
            "n_layers": int(group.shape[0]),
            "n_layers_both_splits_sig": int(group["both_splits_sig"].sum()),
            "frac_layers_both_splits_sig": float(group["both_splits_sig"].mean()),
            "n_layers_both_splits_positive_delta": int(group["both_splits_positive_delta"].sum()),
            "frac_layers_both_splits_positive_delta": float(group["both_splits_positive_delta"].mean()),
        }
    )

dual_split_summary_df = pd.DataFrame(dual_split_summary_records).sort_values("null_family").reset_index(drop=True)
dual_split_summary_path = ITER_DIR / "h1_immune_rewire_dual_split_summary.csv"
dual_split_summary_df.to_csv(dual_split_summary_path, index=False)


iteration_summary = {
    "config": {
        "domain": "immune",
        "split_regimes": list(split_specs.keys()),
        "n_points_per_test": n_points,
        "pca_dim": pca_dim,
        "knn_k_initial": n_neighbors_knn,
        "knn_k_max": max_neighbors_knn,
        "n_null_feature_shuffle": n_null_feature_shuffle,
        "n_null_degree_preserving_geodesic_rewire": n_null_rewire_geodesic,
        "rewire_swap_multiplier": rewire_swap_multiplier,
        "rewire_retry_per_null": rewire_retry_per_null,
        "rewire_attempt_multiplier": rewire_attempt_multiplier,
    },
    "inputs": {
        domain: {seed_tag: str(path) for seed_tag, path in runs.items()}
        for domain, runs in domain_runs.items()
    },
    "domain_summary": domain_summary_df.to_dict(orient="records"),
    "dual_split_summary": dual_split_summary_df.to_dict(orient="records"),
    "artifacts": {
        "by_seed": str(by_seed_path),
        "layer_summary": str(layer_summary_path),
        "pass_matrix": str(pass_matrix_path),
        "domain_summary": str(domain_summary_path),
        "dual_split_summary": str(dual_split_summary_path),
    },
}

summary_path = ITER_DIR / "iter0006_screen_summary.json"
summary_path.write_text(json.dumps(iteration_summary, indent=2))
print(json.dumps(iteration_summary, indent=2))
