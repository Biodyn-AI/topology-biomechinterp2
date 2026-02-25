from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from scipy.stats import combine_pvalues
from sklearn.decomposition import PCA


ITER_DIR = Path("iterations/iter_0005")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def compute_h1_sum(points_or_distances: np.ndarray, distance_matrix: bool = False) -> float:
    """Return the total lifetime of finite H1 bars."""
    dgms = ripser(points_or_distances, maxdim=1, distance_matrix=distance_matrix)["dgms"]
    h1 = dgms[1]
    finite = h1[np.isfinite(h1[:, 1])]
    if finite.size == 0:
        return 0.0
    lifetimes = finite[:, 1] - finite[:, 0]
    return float(lifetimes.sum())


def feature_shuffle_null(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle each feature independently across genes to destroy geometry."""
    shuffled = np.empty_like(points)
    for feature_idx in range(points.shape[1]):
        shuffled[:, feature_idx] = points[rng.permutation(points.shape[0]), feature_idx]
    return shuffled


def null_stats(observed: float, null_values: np.ndarray) -> dict[str, float]:
    """Compute effect-size and empirical p-value summary against a null sample."""
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


def select_top_and_weak_layers() -> dict[str, dict[str, int]]:
    """Pick one strong and one weak layer per domain from prior iteration summaries."""
    selected: dict[str, dict[str, int]] = {}

    lung_summary_path = Path("iterations/iter_0003/scgpt_lung_h1_persistence_layer_summary.csv")
    lung_df = pd.read_csv(lung_summary_path)
    top_lung = int(lung_df.sort_values("mean_h1_sum_delta", ascending=False).iloc[0]["layer"])
    weak_lung = int(lung_df.sort_values("mean_h1_sum_delta", ascending=True).iloc[0]["layer"])
    selected["lung"] = {"top": top_lung, "weak": weak_lung}

    cross_domain_path = Path("iterations/iter_0004/scgpt_cross_domain_h1_layer_summary.csv")
    cross_df = pd.read_csv(cross_domain_path)
    for domain in ["immune", "external_lung"]:
        domain_df = cross_df[cross_df["domain"] == domain].copy()
        top_layer = int(
            domain_df.sort_values("mean_h1_sum_delta", ascending=False).iloc[0]["layer"]
        )
        weak_layer = int(
            domain_df.sort_values("mean_h1_sum_delta", ascending=True).iloc[0]["layer"]
        )
        selected[domain] = {"top": top_layer, "weak": weak_layer}

    return selected


domain_runs = {
    "lung": {
        "seed42_main": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle6_lung_main/layer_gene_embeddings.npy"
        ),
        "seed43": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle6_lung_seed43/layer_gene_embeddings.npy"
        ),
        "seed44": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle6_lung_seed44/layer_gene_embeddings.npy"
        ),
    },
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
    },
    "external_lung": {
        "seed42_main": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle7_external_lung_main/layer_gene_embeddings.npy"
        ),
        "seed43": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle7_external_lung_seed43/layer_gene_embeddings.npy"
        ),
        "seed44": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle7_external_lung_seed44/layer_gene_embeddings.npy"
        ),
    },
}


missing_paths = [
    str(path)
    for runs in domain_runs.values()
    for path in runs.values()
    if not path.exists()
]
if missing_paths:
    raise FileNotFoundError(f"Missing embedding files: {missing_paths}")


selected_layers = select_top_and_weak_layers()
split_specs = {
    "source_disjoint": lambda n_genes: np.arange(0, n_genes // 2, dtype=int),
    "target_disjoint": lambda n_genes: np.arange(n_genes // 2, n_genes, dtype=int),
}

n_points = 180
pca_dim = 14
n_null_feature_shuffle = 20
n_null_distance_permutation = 4

records: list[dict[str, object]] = []

for domain_index, (domain, runs) in enumerate(domain_runs.items()):
    for seed_index, (seed_tag, emb_path) in enumerate(runs.items()):
        layer_embeddings = np.load(emb_path, mmap_mode="r")
        n_layers, n_genes, _ = layer_embeddings.shape

        for split_index, (split_regime, pool_selector) in enumerate(split_specs.items()):
            candidate_pool = pool_selector(n_genes)
            if candidate_pool.size == 0:
                continue

            for layer_type, layer in selected_layers[domain].items():
                if layer >= n_layers:
                    raise ValueError(
                        f"Layer {layer} out of range for domain={domain}, seed={seed_tag}, n_layers={n_layers}"
                    )

                run_seed = (
                    510000
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
                    random_state=9100
                    + domain_index * 100
                    + seed_index * 10
                    + split_index * 3
                    + layer,
                ).fit_transform(points)

                h1_observed = compute_h1_sum(points_pca, distance_matrix=False)

                feature_shuffle_null_values: list[float] = []
                for null_idx in range(n_null_feature_shuffle):
                    fs_rng = np.random.default_rng(
                        610000
                        + domain_index * 100000
                        + seed_index * 10000
                        + split_index * 1000
                        + layer * 50
                        + null_idx
                    )
                    null_points = feature_shuffle_null(points_pca, fs_rng)
                    feature_shuffle_null_values.append(compute_h1_sum(null_points))

                condensed_distances = pdist(points_pca, metric="euclidean")
                distance_perm_null_values: list[float] = []
                for null_idx in range(n_null_distance_permutation):
                    dp_rng = np.random.default_rng(
                        710000
                        + domain_index * 100000
                        + seed_index * 10000
                        + split_index * 1000
                        + layer * 50
                        + null_idx
                    )
                    permuted = condensed_distances.copy()
                    dp_rng.shuffle(permuted)
                    distance_matrix = squareform(permuted)
                    distance_perm_null_values.append(
                        compute_h1_sum(distance_matrix, distance_matrix=True)
                    )

                fs_stats = null_stats(
                    observed=h1_observed,
                    null_values=np.asarray(feature_shuffle_null_values, dtype=float),
                )
                dp_stats = null_stats(
                    observed=h1_observed,
                    null_values=np.asarray(distance_perm_null_values, dtype=float),
                )

                records.extend(
                    [
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer_type": layer_type,
                            "layer": int(layer),
                            "n_genes_pool": int(candidate_pool.size),
                            "n_points": int(points_pca.shape[0]),
                            "pca_dim": int(n_components),
                            "null_family": "feature_shuffle",
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
                            "layer_type": layer_type,
                            "layer": int(layer),
                            "n_genes_pool": int(candidate_pool.size),
                            "n_points": int(points_pca.shape[0]),
                            "pca_dim": int(n_components),
                            "null_family": "distance_permutation",
                            "h1_sum_observed": float(h1_observed),
                            "h1_sum_null_mean": dp_stats["null_mean"],
                            "h1_sum_null_std": dp_stats["null_std"],
                            "h1_sum_delta": dp_stats["delta"],
                            "h1_sum_z": dp_stats["z_score"],
                            "h1_sum_p_perm": dp_stats["p_perm"],
                        },
                    ]
                )


by_seed_df = pd.DataFrame(records).sort_values(
    ["domain", "split_regime", "layer_type", "layer", "null_family", "seed_tag"]
)
by_seed_path = ITER_DIR / "h1_stronger_null_split_by_seed_layer.csv"
by_seed_df.to_csv(by_seed_path, index=False)


layer_summary_records: list[dict[str, object]] = []
group_cols = ["domain", "split_regime", "layer_type", "layer", "null_family"]
for keys, group in by_seed_df.groupby(group_cols, sort=True):
    fisher_stat, fisher_p = combine_pvalues(group["h1_sum_p_perm"].to_numpy(dtype=float), method="fisher")
    layer_summary_records.append(
        {
            "domain": keys[0],
            "split_regime": keys[1],
            "layer_type": keys[2],
            "layer": int(keys[3]),
            "null_family": keys[4],
            "n_seed_runs": int(group.shape[0]),
            "mean_h1_sum_observed": float(group["h1_sum_observed"].mean()),
            "mean_h1_sum_null_mean": float(group["h1_sum_null_mean"].mean()),
            "mean_h1_sum_delta": float(group["h1_sum_delta"].mean()),
            "mean_h1_sum_z": float(group["h1_sum_z"].mean()),
            "delta_positive_fraction": float((group["h1_sum_delta"] > 0).mean()),
            "fisher_stat": float(fisher_stat),
            "fisher_p": float(fisher_p),
        }
    )

layer_summary_df = pd.DataFrame(layer_summary_records).sort_values(group_cols).reset_index(drop=True)
layer_summary_path = ITER_DIR / "h1_stronger_null_split_layer_summary.csv"
layer_summary_df.to_csv(layer_summary_path, index=False)


domain_summary_records: list[dict[str, object]] = []
for keys, group in layer_summary_df.groupby(["domain", "split_regime", "null_family"], sort=True):
    n_tests = int(group.shape[0])
    n_sig = int((group["fisher_p"] < 0.05).sum())
    domain_summary_records.append(
        {
            "domain": keys[0],
            "split_regime": keys[1],
            "null_family": keys[2],
            "n_layer_tests": n_tests,
            "n_layer_tests_fisher_p_lt_0_05": n_sig,
            "frac_layer_tests_fisher_p_lt_0_05": float(n_sig / n_tests if n_tests else np.nan),
            "mean_layer_delta": float(group["mean_h1_sum_delta"].mean()),
            "min_layer_delta": float(group["mean_h1_sum_delta"].min()),
            "max_layer_delta": float(group["mean_h1_sum_delta"].max()),
        }
    )

domain_summary_df = pd.DataFrame(domain_summary_records).sort_values(
    ["domain", "split_regime", "null_family"]
).reset_index(drop=True)
domain_summary_path = ITER_DIR / "h1_stronger_null_split_domain_summary.csv"
domain_summary_df.to_csv(domain_summary_path, index=False)


iteration_summary = {
    "config": {
        "selected_layers_from_prior": selected_layers,
        "split_regimes": list(split_specs.keys()),
        "n_points_per_test": n_points,
        "pca_dim": pca_dim,
        "n_null_feature_shuffle": n_null_feature_shuffle,
        "n_null_distance_permutation": n_null_distance_permutation,
    },
    "inputs": {
        domain: {seed_tag: str(path) for seed_tag, path in runs.items()}
        for domain, runs in domain_runs.items()
    },
    "domain_summary": domain_summary_df.to_dict(orient="records"),
    "artifacts": {
        "by_seed": str(by_seed_path),
        "layer_summary": str(layer_summary_path),
        "domain_summary": str(domain_summary_path),
    },
}

summary_path = ITER_DIR / "iter0005_screen_summary.json"
summary_path.write_text(json.dumps(iteration_summary, indent=2))
print(json.dumps(iteration_summary, indent=2))
