from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ripser import ripser
from scipy.stats import combine_pvalues, spearmanr
from sklearn.decomposition import PCA


ITER_DIR = Path("iterations/iter_0003")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def compute_h1_lifetimes(points: np.ndarray) -> tuple[float, float, int]:
    """Return (sum_lifetime, max_lifetime, count) for finite H1 bars."""
    dgms = ripser(points, maxdim=1)["dgms"]
    h1 = dgms[1]
    finite = h1[np.isfinite(h1[:, 1])]
    if finite.size == 0:
        return 0.0, 0.0, 0
    lifetimes = finite[:, 1] - finite[:, 0]
    return float(lifetimes.sum()), float(lifetimes.max()), int(lifetimes.shape[0])


def feature_shuffle_null(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle each feature independently across points to break topology."""
    shuffled = np.empty_like(points)
    for j in range(points.shape[1]):
        shuffled[:, j] = points[rng.permutation(points.shape[0]), j]
    return shuffled


# H01: scGPT persistent homology with feature-shuffle null.
scgpt_runs = {
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
}

n_points = 350
pca_dim = 20
n_null = 20

h1_records: list[dict[str, object]] = []
for seed_index, (seed_tag, emb_path) in enumerate(scgpt_runs.items()):
    layer_embeddings = np.load(emb_path, mmap_mode="r")
    n_layers, n_genes, _ = layer_embeddings.shape

    for layer in range(n_layers):
        rng = np.random.default_rng(12000 + seed_index * 1000 + layer)
        sampled_idx = rng.choice(n_genes, size=min(n_points, n_genes), replace=False)
        points = layer_embeddings[layer, sampled_idx, :].astype(np.float64)
        points -= points.mean(axis=0, keepdims=True)

        n_components = min(pca_dim, points.shape[0] - 1, points.shape[1])
        points_pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=3100 + seed_index * 100 + layer,
        ).fit_transform(points)

        obs_sum, obs_max, obs_count = compute_h1_lifetimes(points_pca)

        null_sums = []
        for null_idx in range(n_null):
            null_rng = np.random.default_rng(500000 + seed_index * 10000 + layer * 100 + null_idx)
            null_points = feature_shuffle_null(points_pca, null_rng)
            null_sum, _, _ = compute_h1_lifetimes(null_points)
            null_sums.append(null_sum)

        null_array = np.asarray(null_sums, dtype=float)
        null_mean = float(null_array.mean())
        null_std = float(null_array.std(ddof=1)) if null_array.size > 1 else 0.0
        delta = obs_sum - null_mean
        z_score = delta / (null_std + 1e-9)
        p_perm = float((1 + np.sum(null_array >= obs_sum)) / (null_array.size + 1))

        h1_records.append(
            {
                "seed_tag": seed_tag,
                "layer": layer,
                "n_points": int(points.shape[0]),
                "pca_dim": int(n_components),
                "h1_sum_observed": obs_sum,
                "h1_sum_null_mean": null_mean,
                "h1_sum_null_std": null_std,
                "h1_sum_delta": delta,
                "h1_sum_z": z_score,
                "h1_sum_p_perm": p_perm,
                "h1_max_observed": obs_max,
                "h1_count_observed": obs_count,
            }
        )

h1_df = pd.DataFrame(h1_records).sort_values(["layer", "seed_tag"]).reset_index(drop=True)
h1_by_seed_path = ITER_DIR / "scgpt_lung_h1_persistence_by_seed_layer.csv"
h1_df.to_csv(h1_by_seed_path, index=False)

layer_summary_records: list[dict[str, object]] = []
for layer, group in h1_df.groupby("layer", sort=True):
    pvals = group["h1_sum_p_perm"].to_numpy(dtype=float)
    fisher_stat, fisher_p = combine_pvalues(pvals, method="fisher")
    layer_summary_records.append(
        {
            "layer": int(layer),
            "mean_h1_sum_observed": float(group["h1_sum_observed"].mean()),
            "mean_h1_sum_null_mean": float(group["h1_sum_null_mean"].mean()),
            "mean_h1_sum_delta": float(group["h1_sum_delta"].mean()),
            "mean_h1_sum_z": float(group["h1_sum_z"].mean()),
            "delta_positive_fraction": float((group["h1_sum_delta"] > 0).mean()),
            "max_h1_sum_delta": float(group["h1_sum_delta"].max()),
            "min_h1_sum_delta": float(group["h1_sum_delta"].min()),
            "fisher_stat": float(fisher_stat),
            "fisher_p": float(fisher_p),
        }
    )

h1_layer_summary_df = pd.DataFrame(layer_summary_records).sort_values("layer").reset_index(drop=True)
h1_layer_summary_path = ITER_DIR / "scgpt_lung_h1_persistence_layer_summary.csv"
h1_layer_summary_df.to_csv(h1_layer_summary_path, index=False)

# H02: Cross-model feature-profile alignment with exact permutation null.
domain_specs = [
    {
        "domain": "immune",
        "scgpt_csv": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle4_immune_main/alt_geometry_metrics_layer0_immune.csv"
        ),
        "geneformer_csv": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle12_geneformer_immune_bootstrap/geneformer_feature_metrics.csv"
        ),
    },
    {
        "domain": "lung",
        "scgpt_csv": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle6_lung_main/alt_geometry_metrics_layer0_lung.csv"
        ),
        "geneformer_csv": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle12_geneformer_lung_bootstrap/geneformer_feature_metrics.csv"
        ),
    },
    {
        "domain": "external_lung",
        "scgpt_csv": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle7_external_lung_main/alt_geometry_metrics_layer3_external_lung.csv"
        ),
        "geneformer_csv": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle12_geneformer_external_lung_bootstrap/geneformer_feature_metrics.csv"
        ),
    },
]

canonical_features = ["centered_cosine", "dot", "cosine"]
alignment_records: list[dict[str, object]] = []

for spec in domain_specs:
    scgpt_df = pd.read_csv(spec["scgpt_csv"])
    geneformer_df = pd.read_csv(spec["geneformer_csv"])

    scgpt_delta = dict(zip(scgpt_df["feature"], scgpt_df["delta_cv_auroc"]))
    geneformer_delta = dict(zip(geneformer_df["feature"], geneformer_df["delta_cv_auroc"]))

    shared_features = [
        feature
        for feature in canonical_features
        if feature in scgpt_delta and feature in geneformer_delta
    ]

    if len(shared_features) < 2:
        continue

    x = np.asarray([scgpt_delta[f] for f in shared_features], dtype=float)
    y = np.asarray([geneformer_delta[f] for f in shared_features], dtype=float)

    rho = float(spearmanr(x, y).correlation)
    cosine = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))

    permuted = list(itertools.permutations(y.tolist()))
    null_rho = np.asarray([float(spearmanr(x, np.asarray(p, dtype=float)).correlation) for p in permuted])
    null_cosine = np.asarray(
        [
            float(
                np.dot(x, np.asarray(p, dtype=float))
                / (np.linalg.norm(x) * np.linalg.norm(np.asarray(p, dtype=float)) + 1e-12)
            )
            for p in permuted
        ]
    )

    p_perm_rho = float((1 + np.sum(null_rho >= rho)) / (null_rho.size + 1))
    p_perm_cosine = float((1 + np.sum(null_cosine >= cosine)) / (null_cosine.size + 1))

    alignment_records.append(
        {
            "domain": spec["domain"],
            "n_features": len(shared_features),
            "features": "|".join(shared_features),
            "spearman_rho": rho,
            "cosine_similarity": cosine,
            "perm_null_size": int(null_rho.size),
            "p_perm_spearman": p_perm_rho,
            "p_perm_cosine": p_perm_cosine,
            "scgpt_delta_cv_auroc_vector": "|".join(f"{value:.6f}" for value in x),
            "geneformer_delta_cv_auroc_vector": "|".join(f"{value:.6f}" for value in y),
        }
    )

alignment_df = pd.DataFrame(alignment_records).sort_values("domain").reset_index(drop=True)
alignment_domain_path = ITER_DIR / "cross_model_feature_alignment_by_domain.csv"
alignment_df.to_csv(alignment_domain_path, index=False)

summary_payload: dict[str, object] = {
    "domains_evaluated": int(alignment_df.shape[0]),
    "mean_spearman_rho": float(alignment_df["spearman_rho"].mean()),
    "mean_cosine_similarity": float(alignment_df["cosine_similarity"].mean()),
    "max_cosine_similarity": float(alignment_df["cosine_similarity"].max()),
    "min_cosine_similarity": float(alignment_df["cosine_similarity"].min()),
}

rho_fisher_stat, rho_fisher_p = combine_pvalues(alignment_df["p_perm_spearman"].to_numpy(), method="fisher")
cos_fisher_stat, cos_fisher_p = combine_pvalues(alignment_df["p_perm_cosine"].to_numpy(), method="fisher")
summary_payload.update(
    {
        "combined_p_spearman_fisher": float(rho_fisher_p),
        "combined_p_cosine_fisher": float(cos_fisher_p),
        "combined_stat_spearman_fisher": float(rho_fisher_stat),
        "combined_stat_cosine_fisher": float(cos_fisher_stat),
    }
)

alignment_summary_path = ITER_DIR / "cross_model_feature_alignment_summary.json"
alignment_summary_path.write_text(json.dumps(summary_payload, indent=2))

# Consolidated iteration-level numeric summary.
top_layer_row = h1_layer_summary_df.sort_values(
    ["mean_h1_sum_delta", "mean_h1_sum_z"], ascending=[False, False]
).iloc[0]

iteration_summary = {
    "inputs": {
        "scgpt_embedding_runs": {k: str(v) for k, v in scgpt_runs.items()},
        "cross_model_domains": [
            {
                "domain": spec["domain"],
                "scgpt_csv": str(spec["scgpt_csv"]),
                "geneformer_csv": str(spec["geneformer_csv"]),
            }
            for spec in domain_specs
        ],
    },
    "h1_test": {
        "n_layers": int(h1_layer_summary_df.shape[0]),
        "n_seed_runs": int(len(scgpt_runs)),
        "n_null_per_seed_layer": n_null,
        "top_layer": int(top_layer_row["layer"]),
        "top_layer_mean_delta": float(top_layer_row["mean_h1_sum_delta"]),
        "top_layer_mean_z": float(top_layer_row["mean_h1_sum_z"]),
        "top_layer_fisher_p": float(top_layer_row["fisher_p"]),
    },
    "cross_model_alignment": summary_payload,
    "artifacts": {
        "h1_by_seed_layer": str(h1_by_seed_path),
        "h1_layer_summary": str(h1_layer_summary_path),
        "alignment_by_domain": str(alignment_domain_path),
        "alignment_summary": str(alignment_summary_path),
    },
}

(ITER_DIR / "iter0003_screen_summary.json").write_text(json.dumps(iteration_summary, indent=2))
print(json.dumps(iteration_summary, indent=2))
