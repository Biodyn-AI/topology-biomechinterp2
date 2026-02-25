from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import combine_pvalues, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


ITER_DIR = Path("iterations/iter_0012")
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

DOROTHEA_PATH = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
    "single_cell_mechinterp/external/networks/dorothea_human.tsv"
)

TRRUST_PATH = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
    "single_cell_mechinterp/external/networks/trrust_human.tsv"
)

H13_REFERENCE_PATH = Path("iterations/iter_0010/h13_manifold_distance_by_seed_layer_split.csv")

SCGPT_RUNS_BY_DOMAIN = {
    "immune": {
        "run_dir": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle4_immune_main"
        ),
        "layer": 0,
        "gene_cap": 600,
    },
    "lung": {
        "run_dir": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle6_lung_main"
        ),
        "layer": 0,
        "gene_cap": 600,
    },
    "external_lung": {
        "run_dir": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/"
            "implementation/outputs/cycle7_external_lung_main"
        ),
        "layer": 3,
        "gene_cap": 600,
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


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    rho = spearmanr(x, y).correlation
    return float(rho) if np.isfinite(rho) else float("nan")


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


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


def orthogonal_procrustes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = x.T @ y
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    return u @ vt


def get_knn_indices(points: np.ndarray, n_neighbors: int) -> np.ndarray:
    n_points = points.shape[0]
    k = max(1, min(n_neighbors, n_points - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(points)
    _, indices = nbrs.kneighbors(points)
    return indices[:, 1:]


def mapping_mean_jaccard(
    mapping_gene_to_token: np.ndarray, knn_x: np.ndarray, knn_y: np.ndarray
) -> float:
    n = mapping_gene_to_token.size
    jaccards = np.empty(n, dtype=float)
    for gene_pos in range(n):
        token_pos = int(mapping_gene_to_token[gene_pos])
        mapped_neighbors = {int(mapping_gene_to_token[idx]) for idx in knn_x[gene_pos]}
        y_neighbors = {int(idx) for idx in knn_y[token_pos]}
        union_size = len(mapped_neighbors | y_neighbors)
        if union_size == 0:
            jaccards[gene_pos] = 0.0
        else:
            jaccards[gene_pos] = len(mapped_neighbors & y_neighbors) / union_size
    return float(jaccards.mean())


def transfer_edge_auc(
    mapping_gene_to_token: np.ndarray,
    src_token_pos: np.ndarray,
    tgt_token_pos: np.ndarray,
    labels: np.ndarray,
    cos_x: np.ndarray,
) -> float:
    inverse_map = np.empty_like(mapping_gene_to_token)
    inverse_map[mapping_gene_to_token] = np.arange(mapping_gene_to_token.size, dtype=int)
    src_gene_pos = inverse_map[src_token_pos]
    tgt_gene_pos = inverse_map[tgt_token_pos]
    scores = cos_x[src_gene_pos, tgt_gene_pos]
    return safe_auc(labels, scores)


def confidence_tier_from_dorothea_score(score: int) -> str:
    if score >= 3:
        return "high"
    if score >= 1:
        return "medium"
    return "low"


def build_dorothea_score_map() -> dict[tuple[str, str], int]:
    dorothea = pd.read_csv(DOROTHEA_PATH, sep="\t")
    dorothea["source"] = dorothea["source"].astype(str).str.upper()
    dorothea["target"] = dorothea["target"].astype(str).str.upper()
    score_map = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
    dorothea["confidence_score"] = (
        dorothea["confidence"].astype(str).str.upper().map(score_map).fillna(-1).astype(int)
    )
    dorothea_best = dorothea.groupby(["source", "target"], as_index=False)["confidence_score"].max()
    return {
        (str(row.source), str(row.target)): int(row.confidence_score)
        for row in dorothea_best.itertuples(index=False)
    }


def run_h19_confidence_community() -> dict[str, object]:
    edge_df = pd.read_csv(IMMUNE_EDGE_PATH, sep="\t")
    edge_df["source"] = edge_df["source"].astype(str).str.upper()
    edge_df["target"] = edge_df["target"].astype(str).str.upper()

    # TRRUST is loaded for anchor bookkeeping. We intentionally do not use TRRUST membership
    # as a confidence tier because it is label-defining in this edge dataset.
    trrust = pd.read_csv(
        TRRUST_PATH,
        sep="\t",
        header=None,
        names=["source", "target", "regulation", "pmid"],
    )
    trrust["source"] = trrust["source"].astype(str).str.upper()
    trrust["target"] = trrust["target"].astype(str).str.upper()
    trrust_edges = set(zip(trrust["source"], trrust["target"]))

    dorothea_score_map = build_dorothea_score_map()
    edge_pairs = list(zip(edge_df["source"], edge_df["target"]))
    edge_df["in_trrust"] = [pair in trrust_edges for pair in edge_pairs]
    edge_df["dorothea_score"] = [dorothea_score_map.get(pair, -1) for pair in edge_pairs]
    edge_df["confidence_tier"] = edge_df["dorothea_score"].apply(confidence_tier_from_dorothea_score)

    source_threshold = float(edge_df["source_idx"].median())
    target_threshold = float(edge_df["target_idx"].median())
    split_masks = {
        "source_disjoint": edge_df["source_idx"] <= source_threshold,
        "target_disjoint": edge_df["target_idx"] > target_threshold,
    }

    tier_order = ["low", "medium", "high"]
    tier_to_code = {tier: idx for idx, tier in enumerate(tier_order)}
    n_perm = 400

    bin_rows: list[dict[str, object]] = []
    monotonic_rows: list[dict[str, object]] = []

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
            tier_codes = split_edges["confidence_tier"].map(tier_to_code).to_numpy(dtype=int)

            for layer in range(n_layers):
                run_seed = 12_190_000 + seed_index * 10_000 + split_index * 1_000 + layer
                rng = np.random.default_rng(run_seed)

                points = layer_embeddings[layer, edge_gene_indices, :].astype(np.float64)
                points -= points.mean(axis=0, keepdims=True)
                n_components = min(14, points.shape[0] - 1, points.shape[1])
                if n_components < 2:
                    continue
                points_pca = PCA(
                    n_components=n_components,
                    svd_solver="randomized",
                    random_state=12_191 + seed_index * 100 + split_index * 20 + layer,
                ).fit_transform(points)

                graph = build_knn_graph(points_pca, n_neighbors=20)
                communities = list(
                    nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
                )
                community_id = np.zeros(points_pca.shape[0], dtype=int)
                for cid, community in enumerate(communities):
                    for node in community:
                        community_id[int(node)] = int(cid)
                same_community = (community_id[source_local] == community_id[target_local]).astype(int)

                tier_auc_map: dict[str, float] = {}
                tier_delta_map: dict[str, float] = {}
                for tier in tier_order:
                    tier_mask = split_edges["confidence_tier"].to_numpy(dtype=str) == tier
                    labels_tier = labels[tier_mask]
                    same_tier = same_community[tier_mask]
                    auc_tier = safe_auc(labels_tier, same_tier.astype(float))
                    _, _, delta_tier = positive_rate_delta(labels_tier, same_tier)
                    tier_auc_map[tier] = auc_tier
                    tier_delta_map[tier] = delta_tier

                    bin_rows.append(
                        {
                            "domain": "immune",
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "tier": tier,
                            "tier_code": int(tier_to_code[tier]),
                            "n_edges": int(labels_tier.size),
                            "n_positive": int(labels_tier.sum()),
                            "positive_rate": float(labels_tier.mean()) if labels_tier.size else float("nan"),
                            "auc_same_community": float(auc_tier),
                            "delta_positive_rate_same_minus_diff": float(delta_tier),
                            "same_community_edge_fraction": float(same_tier.mean())
                            if same_tier.size
                            else float("nan"),
                        }
                    )

                auc_values = np.array([tier_auc_map[tier] for tier in tier_order], dtype=float)
                delta_values = np.array([tier_delta_map[tier] for tier in tier_order], dtype=float)
                x = np.array([0.0, 1.0, 2.0], dtype=float)

                def slope_with_null(metric_values: np.ndarray, metric_name: str) -> dict[str, float]:
                    result = {
                        f"{metric_name}_slope": float("nan"),
                        f"{metric_name}_rho": float("nan"),
                        f"{metric_name}_p_slope_upper": float("nan"),
                        f"{metric_name}_p_slope_two_sided": float("nan"),
                        f"{metric_name}_null_mean": float("nan"),
                        f"{metric_name}_null_std": float("nan"),
                        f"{metric_name}_n_perm_valid": 0,
                    }
                    if not np.isfinite(metric_values).all():
                        return result

                    slope_obs = float(np.polyfit(x, metric_values, deg=1)[0])
                    rho_obs = safe_spearman(x, metric_values)
                    null_slopes = np.empty(n_perm, dtype=float)

                    for perm_idx in range(n_perm):
                        perm_codes = rng.permutation(tier_codes)
                        perm_values = []
                        for tier in tier_order:
                            code = tier_to_code[tier]
                            tier_mask_perm = perm_codes == code
                            labels_perm_tier = labels[tier_mask_perm]
                            same_perm_tier = same_community[tier_mask_perm]
                            if metric_name == "auc":
                                val = safe_auc(labels_perm_tier, same_perm_tier.astype(float))
                            else:
                                _, _, val = positive_rate_delta(labels_perm_tier, same_perm_tier)
                            perm_values.append(val)
                        perm_values_arr = np.array(perm_values, dtype=float)
                        if np.isfinite(perm_values_arr).all():
                            null_slopes[perm_idx] = float(np.polyfit(x, perm_values_arr, deg=1)[0])
                        else:
                            null_slopes[perm_idx] = float("nan")

                    valid_null = null_slopes[np.isfinite(null_slopes)]
                    if valid_null.size == 0:
                        return result

                    result.update(
                        {
                            f"{metric_name}_slope": slope_obs,
                            f"{metric_name}_rho": rho_obs,
                            f"{metric_name}_p_slope_upper": empirical_upper_tail_p(slope_obs, valid_null),
                            f"{metric_name}_p_slope_two_sided": empirical_two_sided_p(
                                slope_obs, valid_null
                            ),
                            f"{metric_name}_null_mean": float(valid_null.mean()),
                            f"{metric_name}_null_std": float(valid_null.std(ddof=1))
                            if valid_null.size > 1
                            else 0.0,
                            f"{metric_name}_n_perm_valid": int(valid_null.size),
                        }
                    )
                    return result

                auc_monotonic = slope_with_null(auc_values, "auc")
                delta_monotonic = slope_with_null(delta_values, "delta")
                monotonic_rows.append(
                    {
                        "domain": "immune",
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        **auc_monotonic,
                        **delta_monotonic,
                        "auc_low": float(auc_values[0]),
                        "auc_medium": float(auc_values[1]),
                        "auc_high": float(auc_values[2]),
                        "delta_low": float(delta_values[0]),
                        "delta_medium": float(delta_values[1]),
                        "delta_high": float(delta_values[2]),
                    }
                )

    bin_df = pd.DataFrame(bin_rows).sort_values(["split_regime", "layer", "seed_tag", "tier"])
    bin_path = ITER_DIR / "h19_confidence_community_by_seed_layer_split_bin.csv"
    bin_df.to_csv(bin_path, index=False)

    monotonic_df = pd.DataFrame(monotonic_rows).sort_values(["split_regime", "layer", "seed_tag"])
    monotonic_path = ITER_DIR / "h19_confidence_community_monotonicity_tests.csv"
    monotonic_df.to_csv(monotonic_path, index=False)

    layer_split_rows: list[dict[str, object]] = []
    for (split_regime, layer), group in monotonic_df.groupby(["split_regime", "layer"], sort=True):
        layer_split_rows.append(
            {
                "split_regime": split_regime,
                "layer": int(layer),
                "n_seed_rows": int(group.shape[0]),
                "mean_auc_slope": float(group["auc_slope"].mean()),
                "positive_auc_slope_fraction": float((group["auc_slope"] > 0).mean()),
                "fisher_p_auc_slope_upper": safe_fisher_p(group["auc_p_slope_upper"].to_numpy(dtype=float)),
                "mean_delta_slope": float(group["delta_slope"].mean()),
                "positive_delta_slope_fraction": float((group["delta_slope"] > 0).mean()),
                "fisher_p_delta_slope_upper": safe_fisher_p(
                    group["delta_p_slope_upper"].to_numpy(dtype=float)
                ),
                "mean_auc_low": float(group["auc_low"].mean()),
                "mean_auc_medium": float(group["auc_medium"].mean()),
                "mean_auc_high": float(group["auc_high"].mean()),
            }
        )

    layer_split_df = pd.DataFrame(layer_split_rows).sort_values(["split_regime", "layer"])
    layer_split_path = ITER_DIR / "h19_confidence_community_layer_split_summary.csv"
    layer_split_df.to_csv(layer_split_path, index=False)

    split_summary: dict[str, dict[str, float]] = {}
    for split_regime, group in layer_split_df.groupby("split_regime", sort=True):
        split_summary[split_regime] = {
            "mean_auc_slope": float(group["mean_auc_slope"].mean()),
            "layers_with_positive_auc_slope": int((group["mean_auc_slope"] > 0).sum()),
            "layers_fisher_auc_slope_sig": int((group["fisher_p_auc_slope_upper"] < 0.05).sum()),
            "mean_auc_low": float(group["mean_auc_low"].mean()),
            "mean_auc_medium": float(group["mean_auc_medium"].mean()),
            "mean_auc_high": float(group["mean_auc_high"].mean()),
        }

    summary = {
        "n_rows_by_seed_layer_split_bin": int(bin_df.shape[0]),
        "n_monotonic_rows": int(monotonic_df.shape[0]),
        "split_summary": split_summary,
        "artifact_paths": {
            "by_seed_layer_split_bin": str(bin_path),
            "layer_split_summary": str(layer_split_path),
            "monotonicity_tests": str(monotonic_path),
        },
    }
    return summary


def select_top_genes(edge_df: pd.DataFrame, gene_cap: int) -> list[int]:
    degree_counter: dict[int, int] = {}
    for col in ["source_idx", "target_idx"]:
        for value, count in edge_df[col].value_counts().items():
            key = int(value)
            degree_counter[key] = degree_counter.get(key, 0) + int(count)
    ordered = sorted(degree_counter.items(), key=lambda x: (-x[1], x[0]))
    return [gene_idx for gene_idx, _ in ordered[:gene_cap]]


def reduce_to_common_space(x_raw: np.ndarray, y_raw: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    x_center = x_raw - x_raw.mean(axis=0, keepdims=True)
    y_center = y_raw - y_raw.mean(axis=0, keepdims=True)
    n_comp = min(64, x_center.shape[0] - 1, x_center.shape[1], y_center.shape[1])
    if n_comp < 4:
        raise RuntimeError(f"Too few components for alignment: n_comp={n_comp}")
    x_proj = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed).fit_transform(x_center)
    y_proj = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed + 1).fit_transform(y_center)
    x_proj = x_proj - x_proj.mean(axis=0, keepdims=True)
    y_proj = y_proj - y_proj.mean(axis=0, keepdims=True)
    x_proj = x_proj / (np.linalg.norm(x_proj) + 1e-12)
    y_proj = y_proj / (np.linalg.norm(y_proj) + 1e-12)
    return x_proj.astype(np.float64), y_proj.astype(np.float64), int(n_comp)


def run_h20_cross_model_transfer() -> dict[str, object]:
    from transformers import AutoModel

    model = AutoModel.from_pretrained("ctheodoris/Geneformer")
    geneformer_emb = (
        model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    del model

    n_null = 300
    domain_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, domain in enumerate(["immune", "lung", "external_lung"]):
        cfg = SCGPT_RUNS_BY_DOMAIN[domain]
        edge_df = pd.read_csv(GENEFORMER_EDGE_BY_DOMAIN[domain], sep="\t")
        selected_genes = select_top_genes(edge_df, gene_cap=cfg["gene_cap"])
        selected_set = set(selected_genes)

        edge_df = edge_df.loc[
            edge_df["source_idx"].isin(selected_set) & edge_df["target_idx"].isin(selected_set)
        ].copy()
        if edge_df.empty:
            continue

        source_pairs = edge_df[["source_idx", "source_token_id"]].rename(
            columns={"source_idx": "gene_idx", "source_token_id": "token_id"}
        )
        target_pairs = edge_df[["target_idx", "target_token_id"]].rename(
            columns={"target_idx": "gene_idx", "target_token_id": "token_id"}
        )
        mapping_df = pd.concat([source_pairs, target_pairs], axis=0, ignore_index=True)
        mapping_df["gene_idx"] = mapping_df["gene_idx"].astype(int)
        mapping_df["token_id"] = mapping_df["token_id"].astype(int)

        # Keep the most frequent gene->token pair when ambiguities exist.
        mapping_df = (
            mapping_df.groupby(["gene_idx", "token_id"], as_index=False)
            .size()
            .sort_values(["gene_idx", "size", "token_id"], ascending=[True, False, True])
            .drop_duplicates(subset=["gene_idx"], keep="first")
            .drop(columns=["size"])
        )
        mapping_df = mapping_df.loc[mapping_df["token_id"] < geneformer_emb.shape[0]].copy()

        degree_rank = {gene: rank for rank, gene in enumerate(selected_genes)}
        mapping_df["degree_rank"] = mapping_df["gene_idx"].map(degree_rank).fillna(10**9).astype(int)
        mapping_df = mapping_df.sort_values("degree_rank").drop(columns=["degree_rank"]).reset_index(drop=True)

        if mapping_df.shape[0] < 120:
            continue

        gene_ids = mapping_df["gene_idx"].to_numpy(dtype=int)
        token_ids = mapping_df["token_id"].to_numpy(dtype=int)
        n_genes = gene_ids.size
        if n_genes > cfg["gene_cap"]:
            gene_ids = gene_ids[: cfg["gene_cap"]]
            token_ids = token_ids[: cfg["gene_cap"]]
            n_genes = gene_ids.size

        gene_to_pos = {int(gene_idx): idx for idx, gene_idx in enumerate(gene_ids)}
        token_to_pos = {int(token_id): idx for idx, token_id in enumerate(token_ids)}
        keep_edges = (
            edge_df["source_idx"].map(gene_to_pos).notna()
            & edge_df["target_idx"].map(gene_to_pos).notna()
            & edge_df["source_token_id"].map(token_to_pos).notna()
            & edge_df["target_token_id"].map(token_to_pos).notna()
        )
        edge_eval = edge_df.loc[keep_edges].copy()
        if edge_eval["label"].nunique() < 2:
            continue

        src_token_pos = edge_eval["source_token_id"].map(token_to_pos).to_numpy(dtype=int)
        tgt_token_pos = edge_eval["target_token_id"].map(token_to_pos).to_numpy(dtype=int)
        labels = edge_eval["label"].to_numpy(dtype=int)

        layer_embeddings = np.load(cfg["run_dir"] / "layer_gene_embeddings.npy", mmap_mode="r")
        x_raw = layer_embeddings[cfg["layer"], gene_ids, :].astype(np.float64)
        y_raw = geneformer_emb[token_ids, :]
        x_proj, y_proj, n_comp = reduce_to_common_space(
            x_raw, y_raw, seed=12_200 + domain_index * 10
        )

        x_unit = normalize_rows(x_proj)
        cos_x = x_unit @ x_unit.T

        r = orthogonal_procrustes(x_proj, y_proj)
        x_aligned = x_proj @ r
        procrustes_dist = cdist(x_aligned, y_proj, metric="euclidean")
        procrustes_top1 = float(np.mean(np.argmin(procrustes_dist, axis=1) == np.arange(n_genes)))

        cost_matrix = cdist(x_proj, y_proj, metric="sqeuclidean")
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        ot_mapping = col_ind[np.argsort(row_ind)].astype(int)
        ot_top1 = float(np.mean(ot_mapping == np.arange(n_genes)))

        k_neighbors = min(10, n_genes - 1)
        knn_x = get_knn_indices(x_proj, n_neighbors=k_neighbors)
        knn_y = get_knn_indices(y_proj, n_neighbors=k_neighbors)

        true_mapping = np.arange(n_genes, dtype=int)
        jaccard_true = mapping_mean_jaccard(true_mapping, knn_x=knn_x, knn_y=knn_y)
        jaccard_ot = mapping_mean_jaccard(ot_mapping, knn_x=knn_x, knn_y=knn_y)

        auc_true = transfer_edge_auc(
            mapping_gene_to_token=true_mapping,
            src_token_pos=src_token_pos,
            tgt_token_pos=tgt_token_pos,
            labels=labels,
            cos_x=cos_x,
        )
        auc_ot = transfer_edge_auc(
            mapping_gene_to_token=ot_mapping,
            src_token_pos=src_token_pos,
            tgt_token_pos=tgt_token_pos,
            labels=labels,
            cos_x=cos_x,
        )

        rng = np.random.default_rng(12_280 + domain_index)
        jaccard_null = np.empty(n_null, dtype=float)
        auc_null = np.empty(n_null, dtype=float)
        procrustes_top1_null = np.empty(n_null, dtype=float)
        random_top1_null = np.empty(n_null, dtype=float)

        for null_idx in range(n_null):
            perm = rng.permutation(n_genes)

            # Null for Procrustes retrieval: fit with shuffled correspondences.
            y_perm = y_proj[perm]
            r_null = orthogonal_procrustes(x_proj, y_perm)
            x_null = x_proj @ r_null
            dist_null = cdist(x_null, y_proj, metric="euclidean")
            procrustes_top1_null[null_idx] = float(
                np.mean(np.argmin(dist_null, axis=1) == np.arange(n_genes))
            )
            random_top1_null[null_idx] = float(np.mean(perm == np.arange(n_genes)))

            jaccard_null[null_idx] = mapping_mean_jaccard(perm, knn_x=knn_x, knn_y=knn_y)
            auc_null[null_idx] = transfer_edge_auc(
                mapping_gene_to_token=perm,
                src_token_pos=src_token_pos,
                tgt_token_pos=tgt_token_pos,
                labels=labels,
                cos_x=cos_x,
            )

            null_rows.append(
                {
                    "domain": domain,
                    "null_idx": int(null_idx),
                    "procrustes_top1_null": float(procrustes_top1_null[null_idx]),
                    "random_top1_null": float(random_top1_null[null_idx]),
                    "jaccard_null": float(jaccard_null[null_idx]),
                    "transfer_auc_null": float(auc_null[null_idx]),
                }
            )

        domain_rows.append(
            {
                "domain": domain,
                "layer": int(cfg["layer"]),
                "n_genes": int(n_genes),
                "n_edges_eval": int(edge_eval.shape[0]),
                "n_positive_edges": int(labels.sum()),
                "pca_dim_alignment": int(n_comp),
                "k_neighbors": int(k_neighbors),
                "procrustes_top1_retrieval": float(procrustes_top1),
                "ot_top1_assignment_recovery": float(ot_top1),
                "mean_jaccard_true_map": float(jaccard_true),
                "mean_jaccard_ot_map": float(jaccard_ot),
                "transfer_auc_true_map": float(auc_true),
                "transfer_auc_ot_map": float(auc_ot),
                "p_procrustes_top1_upper": empirical_upper_tail_p(procrustes_top1, procrustes_top1_null),
                "p_ot_top1_upper": empirical_upper_tail_p(ot_top1, random_top1_null),
                "p_jaccard_true_upper": empirical_upper_tail_p(jaccard_true, jaccard_null),
                "p_jaccard_ot_upper": empirical_upper_tail_p(jaccard_ot, jaccard_null),
                "p_transfer_auc_true_upper": empirical_upper_tail_p(auc_true, auc_null),
                "p_transfer_auc_ot_upper": empirical_upper_tail_p(auc_ot, auc_null),
                "null_mean_jaccard": float(jaccard_null.mean()),
                "null_mean_transfer_auc": float(auc_null.mean()),
            }
        )

    by_domain_df = pd.DataFrame(domain_rows).sort_values("domain").reset_index(drop=True)
    by_domain_path = ITER_DIR / "h20_cross_model_transfer_by_domain_layer.csv"
    by_domain_df.to_csv(by_domain_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(["domain", "null_idx"]).reset_index(drop=True)
    null_path = ITER_DIR / "h20_cross_model_transfer_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    alignment_summary = {
        "domains_tested": int(by_domain_df.shape[0]),
        "mean_procrustes_top1": float(by_domain_df["procrustes_top1_retrieval"].mean()),
        "mean_ot_top1": float(by_domain_df["ot_top1_assignment_recovery"].mean()),
        "mean_jaccard_true_map": float(by_domain_df["mean_jaccard_true_map"].mean()),
        "mean_jaccard_ot_map": float(by_domain_df["mean_jaccard_ot_map"].mean()),
        "mean_transfer_auc_true_map": float(by_domain_df["transfer_auc_true_map"].mean()),
        "mean_transfer_auc_ot_map": float(by_domain_df["transfer_auc_ot_map"].mean()),
        "domains_jaccard_true_sig": int((by_domain_df["p_jaccard_true_upper"] < 0.05).sum()),
        "domains_transfer_true_sig": int((by_domain_df["p_transfer_auc_true_upper"] < 0.05).sum()),
        "combined_fisher_p_jaccard_true_upper": safe_fisher_p(
            by_domain_df["p_jaccard_true_upper"].to_numpy(dtype=float)
        ),
        "combined_fisher_p_transfer_true_upper": safe_fisher_p(
            by_domain_df["p_transfer_auc_true_upper"].to_numpy(dtype=float)
        ),
        "artifact_paths": {
            "by_domain_layer": str(by_domain_path),
            "null_summary": str(null_path),
        },
    }

    alignment_summary_path = ITER_DIR / "h20_cross_model_transfer_alignment_summary.csv"
    pd.DataFrame([alignment_summary]).to_csv(alignment_summary_path, index=False)
    alignment_summary["artifact_paths"]["alignment_summary"] = str(alignment_summary_path)
    return alignment_summary


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


def run_h21_local_reconstruction() -> dict[str, object]:
    edge_df = pd.read_csv(IMMUNE_EDGE_PATH, sep="\t")
    h13_df = pd.read_csv(H13_REFERENCE_PATH)

    source_threshold = float(edge_df["source_idx"].median())
    target_threshold = float(edge_df["target_idx"].median())
    split_masks = {
        "source_disjoint": edge_df["source_idx"] <= source_threshold,
        "target_disjoint": edge_df["target_idx"] > target_threshold,
    }

    n_perm_row = 400
    row_records: list[dict[str, object]] = []

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
            index_map = {int(gene_idx): int(local_idx) for local_idx, gene_idx in enumerate(edge_gene_indices)}
            source_local = split_edges["source_idx"].map(index_map).to_numpy(dtype=int)
            target_local = split_edges["target_idx"].map(index_map).to_numpy(dtype=int)
            labels = split_edges["label"].to_numpy(dtype=int)

            for layer in range(n_layers):
                run_seed = 12_210_000 + seed_index * 10_000 + split_index * 1_000 + layer
                rng = np.random.default_rng(run_seed)

                points = layer_embeddings[layer, edge_gene_indices, :].astype(np.float64)
                points -= points.mean(axis=0, keepdims=True)
                n_components = min(32, points.shape[0] - 1, points.shape[1])
                if n_components < 8:
                    continue

                points_pca = PCA(
                    n_components=n_components,
                    svd_solver="randomized",
                    random_state=12_211 + seed_index * 100 + split_index * 20 + layer,
                ).fit_transform(points)
                neighbor_idx = get_knn_indices(points_pca, n_neighbors=12)
                recon_errors = compute_local_reconstruction_errors(points_pca, neighbor_idx)

                edge_recon_mean = 0.5 * (recon_errors[source_local] + recon_errors[target_local])
                edge_recon_absdiff = np.abs(recon_errors[source_local] - recon_errors[target_local])

                auc_edge_recon = safe_auc(labels, edge_recon_mean)
                auc_edge_recon_absdiff = safe_auc(labels, edge_recon_absdiff)
                pos_mean = float(edge_recon_mean[labels == 1].mean())
                neg_mean = float(edge_recon_mean[labels == 0].mean())
                delta_pos_minus_neg = float(pos_mean - neg_mean)

                null_auc = np.empty(n_perm_row, dtype=float)
                null_delta = np.empty(n_perm_row, dtype=float)
                for perm_idx in range(n_perm_row):
                    labels_perm = rng.permutation(labels)
                    null_auc[perm_idx] = safe_auc(labels_perm, edge_recon_mean)
                    perm_pos_mean = float(edge_recon_mean[labels_perm == 1].mean())
                    perm_neg_mean = float(edge_recon_mean[labels_perm == 0].mean())
                    null_delta[perm_idx] = perm_pos_mean - perm_neg_mean

                row_records.append(
                    {
                        "domain": "immune",
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges": int(labels.size),
                        "n_positive": int(labels.sum()),
                        "n_unique_genes": int(edge_gene_indices.size),
                        "pca_dim": int(n_components),
                        "mean_gene_reconstruction_error": float(recon_errors.mean()),
                        "std_gene_reconstruction_error": float(recon_errors.std(ddof=1)),
                        "mean_edge_reconstruction_error": float(edge_recon_mean.mean()),
                        "mean_edge_absdiff_reconstruction_error": float(edge_recon_absdiff.mean()),
                        "positive_edge_recon_mean": pos_mean,
                        "negative_edge_recon_mean": neg_mean,
                        "delta_edge_recon_pos_minus_neg": delta_pos_minus_neg,
                        "auc_edge_recon_mean": float(auc_edge_recon),
                        "auc_edge_recon_absdiff": float(auc_edge_recon_absdiff),
                        "p_auc_edge_recon_upper": empirical_upper_tail_p(auc_edge_recon, null_auc),
                        "p_auc_edge_recon_two_sided": empirical_two_sided_p(auc_edge_recon, null_auc),
                        "p_delta_pos_minus_neg_upper": empirical_upper_tail_p(
                            delta_pos_minus_neg, null_delta
                        ),
                        "p_delta_pos_minus_neg_two_sided": empirical_two_sided_p(
                            delta_pos_minus_neg, null_delta
                        ),
                    }
                )

    row_df = pd.DataFrame(row_records).sort_values(["split_regime", "layer", "seed_tag"]).reset_index(
        drop=True
    )
    row_path = ITER_DIR / "h21_local_reconstruction_edge_features.csv"
    row_df.to_csv(row_path, index=False)

    merged = row_df.merge(
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

    coupling_by_seed_rows: list[dict[str, object]] = []
    trend_rows: list[dict[str, object]] = []
    n_perm_coupling = 3000

    for split_regime in ["source_disjoint", "target_disjoint"]:
        split_df = merged.loc[merged["split_regime"] == split_regime].copy()

        # Edge-level predictive trend summary by split.
        trend_rows.append(
            {
                "analysis_type": "edge_predictive",
                "split_regime": split_regime,
                "metric": "auc_edge_recon_mean",
                "n_rows": int(split_df.shape[0]),
                "mean_value": float(split_df["auc_edge_recon_mean"].mean()),
                "positive_fraction": float((split_df["auc_edge_recon_mean"] > 0.5).mean()),
                "fisher_p_upper": safe_fisher_p(split_df["p_auc_edge_recon_upper"].to_numpy(dtype=float)),
            }
        )
        trend_rows.append(
            {
                "analysis_type": "edge_predictive",
                "split_regime": split_regime,
                "metric": "delta_edge_recon_pos_minus_neg",
                "n_rows": int(split_df.shape[0]),
                "mean_value": float(split_df["delta_edge_recon_pos_minus_neg"].mean()),
                "positive_fraction": float(
                    (split_df["delta_edge_recon_pos_minus_neg"] > 0).mean()
                ),
                "fisher_p_upper": safe_fisher_p(
                    split_df["p_delta_pos_minus_neg_upper"].to_numpy(dtype=float)
                ),
            }
        )

        for metric in ["mean_edge_reconstruction_error", "delta_edge_recon_pos_minus_neg"]:
            per_seed_rho: list[float] = []
            per_seed_vectors: list[tuple[np.ndarray, np.ndarray]] = []
            for seed_tag in sorted(split_df["seed_tag"].unique()):
                seed_df = split_df.loc[split_df["seed_tag"] == seed_tag].sort_values("layer")
                x = seed_df[metric].to_numpy(dtype=float)
                y = seed_df["delta_auc_geodesic_minus_euclidean"].to_numpy(dtype=float)
                rho = safe_spearman(x, y)
                per_seed_rho.append(rho)
                per_seed_vectors.append((x, y))
                coupling_by_seed_rows.append(
                    {
                        "split_regime": split_regime,
                        "metric": metric,
                        "seed_tag": seed_tag,
                        "n_layers": int(seed_df.shape[0]),
                        "spearman_rho": float(rho),
                    }
                )

            observed_mean_rho = float(np.nanmean(np.asarray(per_seed_rho, dtype=float)))
            rng = np.random.default_rng(
                12_230 + (0 if split_regime == "source_disjoint" else 100) + (0 if metric == "mean_edge_reconstruction_error" else 10)
            )
            null_mean_rho = np.empty(n_perm_coupling, dtype=float)
            for perm_idx in range(n_perm_coupling):
                rho_perm_values = []
                for x, y in per_seed_vectors:
                    y_perm = rng.permutation(y)
                    rho_perm_values.append(safe_spearman(x, y_perm))
                null_mean_rho[perm_idx] = float(np.nanmean(np.asarray(rho_perm_values, dtype=float)))

            trend_rows.append(
                {
                    "analysis_type": "coupling_to_h13",
                    "split_regime": split_regime,
                    "metric": metric,
                    "n_rows": int(len(per_seed_rho)),
                    "mean_value": observed_mean_rho,
                    "positive_fraction": float((np.asarray(per_seed_rho) > 0).mean()),
                    "fisher_p_upper": float("nan"),
                    "p_mean_rho_upper": empirical_upper_tail_p(observed_mean_rho, null_mean_rho),
                    "p_mean_rho_lower": empirical_lower_tail_p(observed_mean_rho, null_mean_rho),
                    "p_mean_rho_two_sided": empirical_two_sided_p(observed_mean_rho, null_mean_rho),
                    "null_mean_rho": float(null_mean_rho.mean()),
                    "null_std_rho": float(null_mean_rho.std(ddof=1)),
                }
            )

    coupling_by_seed_df = pd.DataFrame(coupling_by_seed_rows).sort_values(
        ["split_regime", "metric", "seed_tag"]
    )
    coupling_by_seed_path = ITER_DIR / "h21_local_reconstruction_coupling_by_seed.csv"
    coupling_by_seed_df.to_csv(coupling_by_seed_path, index=False)

    trend_df = pd.DataFrame(trend_rows).sort_values(["analysis_type", "split_regime", "metric"])
    trend_path = ITER_DIR / "h21_local_reconstruction_trend_summary.csv"
    trend_df.to_csv(trend_path, index=False)

    summary = {
        "n_rows_seed_layer_split": int(row_df.shape[0]),
        "mean_auc_edge_recon": float(row_df["auc_edge_recon_mean"].mean()),
        "mean_delta_pos_minus_neg": float(row_df["delta_edge_recon_pos_minus_neg"].mean()),
        "artifact_paths": {
            "edge_features": str(row_path),
            "trend_summary": str(trend_path),
            "coupling_by_seed": str(coupling_by_seed_path),
        },
    }
    return summary


def main() -> None:
    required_paths = [
        *SCGPT_IMMUNE_RUNS.values(),
        IMMUNE_EDGE_PATH,
        DOROTHEA_PATH,
        TRRUST_PATH,
        H13_REFERENCE_PATH,
        *GENEFORMER_EDGE_BY_DOMAIN.values(),
        *[cfg["run_dir"] / "layer_gene_embeddings.npy" for cfg in SCGPT_RUNS_BY_DOMAIN.values()],
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    h19_summary = run_h19_confidence_community()
    h20_summary = run_h20_cross_model_transfer()
    h21_summary = run_h21_local_reconstruction()

    iteration_summary = {
        "iteration": "iter_0012",
        "inputs": {
            "scgpt_immune_runs": {k: str(v) for k, v in SCGPT_IMMUNE_RUNS.items()},
            "immune_edge_dataset": str(IMMUNE_EDGE_PATH),
            "dorothea_path": str(DOROTHEA_PATH),
            "trrust_path": str(TRRUST_PATH),
            "h13_reference_path": str(H13_REFERENCE_PATH),
            "cross_model_domains": {
                domain: {
                    "scgpt_run_dir": str(cfg["run_dir"]),
                    "geneformer_edge_dataset": str(GENEFORMER_EDGE_BY_DOMAIN[domain]),
                    "layer": int(cfg["layer"]),
                }
                for domain, cfg in SCGPT_RUNS_BY_DOMAIN.items()
            },
        },
        "h19_confidence_community": h19_summary,
        "h20_cross_model_transfer": h20_summary,
        "h21_local_reconstruction": h21_summary,
    }

    summary_path = ITER_DIR / "iter0012_screen_summary.json"
    summary_path.write_text(json.dumps(iteration_summary, indent=2))
    print(json.dumps(iteration_summary, indent=2))


if __name__ == "__main__":
    main()
