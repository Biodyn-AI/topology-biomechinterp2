from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from ripser import ripser
from scipy.stats import combine_pvalues, spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


ITER_DIR = Path("iterations/iter_0004")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def compute_h1_lifetimes(points: np.ndarray) -> tuple[float, float, int]:
    """Return (sum_lifetime, max_lifetime, finite_bar_count) for H1 bars."""
    diagrams = ripser(points, maxdim=1)["dgms"]
    h1 = diagrams[1]
    finite = h1[np.isfinite(h1[:, 1])]
    if finite.size == 0:
        return 0.0, 0.0, 0
    lifetimes = finite[:, 1] - finite[:, 0]
    return float(lifetimes.sum()), float(lifetimes.max()), int(finite.shape[0])


def feature_shuffle_null(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Break global geometry by independently shuffling each feature."""
    shuffled = np.empty_like(points)
    for j in range(points.shape[1]):
        shuffled[:, j] = points[rng.permutation(points.shape[0]), j]
    return shuffled


def participation_ratio_dim(points: np.ndarray) -> float:
    """Effective dimension from variance spectrum."""
    variances = points.var(axis=0, ddof=1)
    numerator = float(variances.sum() ** 2)
    denominator = float(np.square(variances).sum())
    if denominator <= 1e-12:
        return 0.0
    return numerator / denominator


def local_linearity_ratio(points: np.ndarray, top_k: int = 5) -> float:
    """Share of variance captured by leading PCs as a local-linearity proxy."""
    variances = points.var(axis=0, ddof=1)
    total = float(variances.sum())
    if total <= 1e-12:
        return 0.0
    k = min(top_k, variances.size)
    return float(variances[:k].sum() / total)


def mle_intrinsic_dimension(points: np.ndarray, k: int = 10) -> float:
    """Levina-Bickel local MLE intrinsic dimension estimate."""
    n_points = points.shape[0]
    n_neighbors = min(k + 1, n_points)
    if n_neighbors <= 2:
        return float("nan")

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(points)
    dists, _ = nn.kneighbors(points)

    # Remove self-distance (first column) and keep k nearest neighbors.
    dists = dists[:, 1:]
    if dists.shape[1] < 2:
        return float("nan")

    t_k = dists[:, -1]
    t_j = dists[:, :-1]
    valid = (t_k > 0.0)[:, None] & (t_j > 0.0)
    log_terms = np.full(t_j.shape, np.nan, dtype=float)
    log_terms[valid] = np.log((t_k[:, None] + 1e-12)[valid] / (t_j + 1e-12)[valid])

    valid_counts = valid.sum(axis=1)
    sum_log = np.nansum(log_terms, axis=1)
    mean_log = np.full(sum_log.shape, np.nan, dtype=float)
    np.divide(sum_log, valid_counts, out=mean_log, where=valid_counts > 0)

    point_estimates = np.where(mean_log > 1e-12, 1.0 / mean_log, np.nan)
    estimate = float(np.nanmean(point_estimates))
    return estimate if np.isfinite(estimate) else float("nan")


def spearman_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Two-sided permutation p-value for Spearman correlation."""
    rho_obs = float(spearmanr(x, y).correlation)
    if not np.isfinite(rho_obs):
        return float("nan"), float("nan")

    null_rhos = np.empty(n_perm, dtype=float)
    for idx in range(n_perm):
        y_perm = rng.permutation(y)
        null_rhos[idx] = float(spearmanr(x, y_perm).correlation)

    p_two_sided = float((1 + np.sum(np.abs(null_rhos) >= abs(rho_obs))) / (n_perm + 1))
    return rho_obs, p_two_sided


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

n_points = 350
pca_dim = 20
n_null = 20
n_perm = 2000

missing_paths = [
    str(path)
    for runs in domain_runs.values()
    for path in runs.values()
    if not path.exists()
]
if missing_paths:
    raise FileNotFoundError(f"Missing embedding files: {missing_paths}")


records: list[dict[str, object]] = []
for domain_index, (domain, runs) in enumerate(domain_runs.items()):
    for seed_index, (seed_tag, emb_path) in enumerate(runs.items()):
        layer_embeddings = np.load(emb_path, mmap_mode="r")
        n_layers, n_genes, _ = layer_embeddings.shape

        for layer in range(n_layers):
            rng = np.random.default_rng(
                240000 + domain_index * 10000 + seed_index * 1000 + layer
            )
            sampled_idx = rng.choice(n_genes, size=min(n_points, n_genes), replace=False)
            points = layer_embeddings[layer, sampled_idx, :].astype(np.float64)
            points -= points.mean(axis=0, keepdims=True)

            n_components = min(pca_dim, points.shape[0] - 1, points.shape[1])
            points_pca = PCA(
                n_components=n_components,
                svd_solver="randomized",
                random_state=7100 + domain_index * 100 + seed_index * 10 + layer,
            ).fit_transform(points)

            h1_sum_obs, h1_max_obs, h1_count_obs = compute_h1_lifetimes(points_pca)

            null_sums: list[float] = []
            for null_idx in range(n_null):
                null_rng = np.random.default_rng(
                    880000
                    + domain_index * 100000
                    + seed_index * 10000
                    + layer * 100
                    + null_idx
                )
                null_points = feature_shuffle_null(points_pca, null_rng)
                null_sum, _, _ = compute_h1_lifetimes(null_points)
                null_sums.append(null_sum)

            null_array = np.asarray(null_sums, dtype=float)
            null_mean = float(null_array.mean())
            null_std = float(null_array.std(ddof=1)) if null_array.size > 1 else 0.0
            delta = h1_sum_obs - null_mean
            z_score = float(delta / (null_std + 1e-9))
            p_perm = float((1 + np.sum(null_array >= h1_sum_obs)) / (null_array.size + 1))

            records.append(
                {
                    "domain": domain,
                    "seed_tag": seed_tag,
                    "layer": layer,
                    "n_points": int(points.shape[0]),
                    "pca_dim": int(n_components),
                    "h1_sum_observed": h1_sum_obs,
                    "h1_sum_null_mean": null_mean,
                    "h1_sum_null_std": null_std,
                    "h1_sum_delta": float(delta),
                    "h1_sum_z": z_score,
                    "h1_sum_p_perm": p_perm,
                    "h1_max_observed": h1_max_obs,
                    "h1_count_observed": h1_count_obs,
                    "mle_intrinsic_dim": mle_intrinsic_dimension(points_pca, k=10),
                    "participation_ratio_dim": participation_ratio_dim(points_pca),
                    "linearity_top5_ratio": local_linearity_ratio(points_pca, top_k=5),
                }
            )

h1_df = (
    pd.DataFrame(records)
    .sort_values(["domain", "layer", "seed_tag"])
    .reset_index(drop=True)
)
h1_by_seed_layer_path = ITER_DIR / "scgpt_cross_domain_h1_by_seed_layer.csv"
h1_df.to_csv(h1_by_seed_layer_path, index=False)


layer_summary_records: list[dict[str, object]] = []
for (domain, layer), group in h1_df.groupby(["domain", "layer"], sort=True):
    fisher_stat, fisher_p = combine_pvalues(group["h1_sum_p_perm"].to_numpy(dtype=float), method="fisher")
    layer_summary_records.append(
        {
            "domain": domain,
            "layer": int(layer),
            "mean_h1_sum_observed": float(group["h1_sum_observed"].mean()),
            "mean_h1_sum_null_mean": float(group["h1_sum_null_mean"].mean()),
            "mean_h1_sum_delta": float(group["h1_sum_delta"].mean()),
            "mean_h1_sum_z": float(group["h1_sum_z"].mean()),
            "delta_positive_fraction": float((group["h1_sum_delta"] > 0).mean()),
            "fisher_stat": float(fisher_stat),
            "fisher_p": float(fisher_p),
            "mean_mle_intrinsic_dim": float(group["mle_intrinsic_dim"].mean()),
            "mean_participation_ratio_dim": float(group["participation_ratio_dim"].mean()),
            "mean_linearity_top5_ratio": float(group["linearity_top5_ratio"].mean()),
        }
    )

h1_layer_summary_df = (
    pd.DataFrame(layer_summary_records)
    .sort_values(["domain", "layer"])
    .reset_index(drop=True)
)
h1_layer_summary_path = ITER_DIR / "scgpt_cross_domain_h1_layer_summary.csv"
h1_layer_summary_df.to_csv(h1_layer_summary_path, index=False)


domain_summary_records: list[dict[str, object]] = []
for domain, domain_df in h1_layer_summary_df.groupby("domain", sort=True):
    n_layers = int(domain_df.shape[0])
    n_sig = int((domain_df["fisher_p"] < 0.05).sum())
    top_layer = domain_df.sort_values(
        ["mean_h1_sum_delta", "mean_h1_sum_z"],
        ascending=[False, False],
    ).iloc[0]
    domain_summary_records.append(
        {
            "domain": domain,
            "n_layers": n_layers,
            "n_layers_fisher_p_lt_0_05": n_sig,
            "frac_layers_fisher_p_lt_0_05": float(n_sig / n_layers if n_layers else np.nan),
            "mean_layer_h1_delta": float(domain_df["mean_h1_sum_delta"].mean()),
            "median_layer_h1_delta": float(domain_df["mean_h1_sum_delta"].median()),
            "min_layer_h1_delta": float(domain_df["mean_h1_sum_delta"].min()),
            "max_layer_h1_delta": float(domain_df["mean_h1_sum_delta"].max()),
            "top_layer": int(top_layer["layer"]),
            "top_layer_mean_delta": float(top_layer["mean_h1_sum_delta"]),
            "top_layer_fisher_p": float(top_layer["fisher_p"]),
        }
    )

domain_summary_df = pd.DataFrame(domain_summary_records).sort_values("domain").reset_index(drop=True)
domain_summary_path = ITER_DIR / "scgpt_cross_domain_h1_domain_summary.csv"
domain_summary_df.to_csv(domain_summary_path, index=False)


intrinsic_corr_records: list[dict[str, object]] = []
metric_columns = [
    "mle_intrinsic_dim",
    "participation_ratio_dim",
    "linearity_top5_ratio",
]

for domain_index, (domain, domain_df) in enumerate(h1_df.groupby("domain", sort=True)):
    for seed_index, (seed_tag, seed_df) in enumerate(domain_df.groupby("seed_tag", sort=True)):
        seed_df = seed_df.sort_values("layer").reset_index(drop=True)
        x = seed_df["h1_sum_delta"].to_numpy(dtype=float)
        for metric_index, metric_col in enumerate(metric_columns):
            y = seed_df[metric_col].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if int(valid.sum()) < 4:
                intrinsic_corr_records.append(
                    {
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "metric": metric_col,
                        "n_layers_valid": int(valid.sum()),
                        "spearman_rho": float("nan"),
                        "perm_p_two_sided": float("nan"),
                    }
                )
                continue

            rng = np.random.default_rng(
                970000 + domain_index * 10000 + seed_index * 1000 + metric_index * 100
            )
            rho_obs, p_perm = spearman_permutation_test(
                x=x[valid],
                y=y[valid],
                n_perm=n_perm,
                rng=rng,
            )
            intrinsic_corr_records.append(
                {
                    "domain": domain,
                    "seed_tag": seed_tag,
                    "metric": metric_col,
                    "n_layers_valid": int(valid.sum()),
                    "spearman_rho": rho_obs,
                    "perm_p_two_sided": p_perm,
                }
            )

intrinsic_corr_df = (
    pd.DataFrame(intrinsic_corr_records)
    .sort_values(["domain", "metric", "seed_tag"])
    .reset_index(drop=True)
)
intrinsic_corr_path = ITER_DIR / "intrinsic_dimensionality_h1_correlation_by_seed.csv"
intrinsic_corr_df.to_csv(intrinsic_corr_path, index=False)


intrinsic_summary_records: list[dict[str, object]] = []
for (domain, metric), metric_df in intrinsic_corr_df.groupby(["domain", "metric"], sort=True):
    valid_p = metric_df["perm_p_two_sided"].dropna().to_numpy(dtype=float)
    fisher_p = float("nan")
    fisher_stat = float("nan")
    if valid_p.size > 0:
        fisher_stat, fisher_p = combine_pvalues(valid_p, method="fisher")
    intrinsic_summary_records.append(
        {
            "domain": domain,
            "metric": metric,
            "mean_spearman_rho": float(metric_df["spearman_rho"].mean()),
            "std_spearman_rho": float(metric_df["spearman_rho"].std(ddof=1)),
            "n_seed_runs": int(metric_df.shape[0]),
            "fisher_stat": float(fisher_stat),
            "fisher_p": float(fisher_p),
            "sign_consistency": float(
                (
                    np.sign(metric_df["spearman_rho"].fillna(0.0).to_numpy(dtype=float))
                    == np.sign(metric_df["spearman_rho"].mean())
                ).mean()
            ),
        }
    )

intrinsic_summary_df = (
    pd.DataFrame(intrinsic_summary_records)
    .sort_values(["domain", "metric"])
    .reset_index(drop=True)
)
intrinsic_summary_path = ITER_DIR / "intrinsic_dimensionality_h1_correlation_summary.csv"
intrinsic_summary_df.to_csv(intrinsic_summary_path, index=False)


iteration_summary = {
    "inputs": {
        "domain_runs": {
            domain: {seed_tag: str(path) for seed_tag, path in runs.items()}
            for domain, runs in domain_runs.items()
        }
    },
    "config": {
        "n_points_per_layer": n_points,
        "pca_dim": pca_dim,
        "n_null_feature_shuffle": n_null,
        "n_perm_intrinsic_corr": n_perm,
    },
    "cross_domain_h1": domain_summary_df.to_dict(orient="records"),
    "intrinsic_corr_domain_metric": intrinsic_summary_df.to_dict(orient="records"),
    "artifacts": {
        "h1_by_seed_layer": str(h1_by_seed_layer_path),
        "h1_layer_summary": str(h1_layer_summary_path),
        "h1_domain_summary": str(domain_summary_path),
        "intrinsic_corr_by_seed": str(intrinsic_corr_path),
        "intrinsic_corr_summary": str(intrinsic_summary_path),
    },
}

summary_path = ITER_DIR / "iter0004_screen_summary.json"
summary_path.write_text(json.dumps(iteration_summary, indent=2))
print(json.dumps(iteration_summary, indent=2))
