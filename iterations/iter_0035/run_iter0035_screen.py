from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ITER_DIR = Path("iterations/iter_0035")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base")

# H88 / N448: multiseed sparse descriptor consensus robustness.
H88_SEEDS = ["seed42_main", "seed43", "seed44"]
H88_LAYERS = [0, 3, 7, 11]
H88_GENE_CAP = 170
H88_NEIGHBORS = 12
H88_TRIANGLE_K = [8, 12, 16]
H88_NULL_PERM = 4
H88_CV_SPLITS = 4
H88_L1_C = 0.2
H88_EDGE_SAMPLE = 300

# H89 / N441: local linearity phase-boundary screen.
H89_LAYERS = [0, 3, 7, 11]
H89_GENE_CAP = 170
H89_NEIGHBORS = 12
H89_ID_K = 12
H89_LOCAL_DIM = 4
H89_NULL_PERM = 6
H89_CV_SPLITS = 4
H89_L1_C = 0.25
H89_EDGE_SAMPLE = 320

# H90 / N438: perturbation topology stability screen.
H90_LAYERS = [7, 11]
H90_GENE_CAP = 170
H90_NEIGHBORS = 12
H90_TRIANGLE_K = [8, 12, 16]
H90_NULL_PERM = 8
H90_CV_SPLITS = 4
H90_L1_C = 0.25
H90_EDGE_SAMPLE = 320
H90_K_VARIANTS = [10, 16]
H90_JITTER_SIGMAS = [0.005, 0.01]


def ensure_required_inputs() -> None:
    required_paths = [
        BASE.DOROTHEA_PATH,
        BASE.GENE2GO_PATH,
        BASE.OMNIPATH_INTERACTIONS_PATH,
        BASE.STRING_CACHE_PATH,
    ]
    for run_map in BASE.SCGPT_RUNS_BY_DOMAIN.values():
        for run_dir in run_map.values():
            required_paths.append(run_dir / "cycle1_edge_dataset.tsv")
            required_paths.append(run_dir / "layer_gene_embeddings.npy")

    missing = [str(p) for p in required_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def min_class_count(labels: np.ndarray) -> int:
    y = np.asarray(labels, dtype=int)
    counts = np.bincount(y, minlength=2)
    return int(np.min(counts))


def stratified_index_sample(labels: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    n = y.size
    if n <= max_n:
        return np.arange(n, dtype=int)

    idx_all = np.arange(n, dtype=int)
    idx_pos = idx_all[y == 1]
    idx_neg = idx_all[y == 0]
    if idx_pos.size == 0 or idx_neg.size == 0:
        return np.sort(rng.choice(idx_all, size=max_n, replace=False))

    frac_pos = idx_pos.size / n
    n_pos = int(round(max_n * frac_pos))
    n_pos = max(1, min(n_pos, idx_pos.size - 1))
    n_neg = max(1, min(max_n - n_pos, idx_neg.size - 1))

    choose_pos = rng.choice(idx_pos, size=n_pos, replace=False)
    choose_neg = rng.choice(idx_neg, size=n_neg, replace=False)
    chosen = np.sort(np.concatenate([choose_pos, choose_neg]))

    if chosen.size < max_n:
        pool = np.setdiff1d(idx_all, chosen, assume_unique=False)
        extra = rng.choice(pool, size=max_n - chosen.size, replace=False)
        chosen = np.sort(np.concatenate([chosen, extra]))

    return chosen


def cross_validated_auc(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    penalty: str,
    c_value: float = 1.0,
    n_splits: int = 4,
) -> float:
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)

    if x.ndim != 2 or x.shape[0] != y.size or np.unique(y).size < 2:
        return float("nan")

    max_splits = min(n_splits, min_class_count(y))
    if max_splits < 2:
        return float("nan")

    cv = StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=random_state)
    probs = np.full(y.shape[0], np.nan, dtype=float)

    for fold_idx, (tr, te) in enumerate(cv.split(x, y)):
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x[tr])
        x_te = scaler.transform(x[te])

        if penalty == "none":
            model = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=1200,
                random_state=random_state + fold_idx,
            )
        elif penalty == "l1":
            model = LogisticRegression(
                penalty="l1",
                C=float(c_value),
                solver="liblinear",
                max_iter=1200,
                random_state=random_state + fold_idx,
            )
        else:
            raise ValueError(f"Unsupported penalty={penalty}")

        model.fit(x_tr, y[tr])
        probs[te] = model.predict_proba(x_te)[:, 1]

    return BASE.safe_auc(y, probs)


def fit_l1_coefficients(features: np.ndarray, labels: np.ndarray, c_value: float, random_state: int) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    if x.ndim != 2 or x.shape[0] != y.size or np.unique(y).size < 2:
        return np.zeros(x.shape[1], dtype=float)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(
        penalty="l1",
        C=float(c_value),
        solver="liblinear",
        max_iter=1200,
        random_state=random_state,
    )
    model.fit(x_scaled, y)
    return np.asarray(model.coef_, dtype=float).ravel()


def compute_h70_scores(
    geodesic: np.ndarray,
    support_dir: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    triangle_k: list[int],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    edge_geodesic = geodesic[source_local, target_local]
    edge_support = support_dir[source_local, target_local]
    edge_margin = np.abs(support_dir[source_local, target_local] - support_dir[target_local, source_local])

    feature_bundle = BASE.multiscale_triangle_defect_features(
        geodesic=geodesic,
        source_local=source_local,
        target_local=target_local,
        k_values=triangle_k,
    )

    baseline = BASE.zscore(-edge_geodesic) + 0.75 * BASE.zscore(edge_support) + 0.35 * BASE.zscore(edge_margin)
    defect = (
        baseline
        + 0.35 * BASE.zscore(-feature_bundle["median_mean"])
        + 0.25 * BASE.zscore(-feature_bundle["tail_mean"])
        + 0.20 * BASE.zscore(feature_bundle["close_frac_mean"])
        + 0.10 * BASE.zscore(-feature_bundle["scale_span"])
        + 0.10 * BASE.zscore(-feature_bundle["dispersion_mean"])
    )
    return baseline, defect, feature_bundle


def build_sparse_descriptors(
    geodesic: np.ndarray,
    support_dir: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    points_pca: np.ndarray,
    tri_bundle: dict[str, np.ndarray],
    n_neighbors: int,
) -> np.ndarray:
    n_nodes = points_pca.shape[0]
    knn_edges = BASE.build_knn_edge_array(points_pca, n_neighbors=n_neighbors)
    neighbors = BASE.adjacency_neighbors(n_nodes, knn_edges)
    clust = BASE.local_clustering(neighbors)
    degree = np.asarray([len(v) for v in neighbors], dtype=float)

    node_bridge = degree * (1.0 - clust)
    node_curv = 1.0 - clust

    edge_bridge = 0.5 * (node_bridge[source_local] + node_bridge[target_local])
    edge_curv_var = 0.5 * (node_curv[source_local] - node_curv[target_local]) ** 2
    edge_margin = np.abs(support_dir[source_local, target_local] - support_dir[target_local, source_local])
    edge_support = support_dir[source_local, target_local]
    edge_defect_med = -tri_bundle["median_mean"]
    edge_defect_tail = -tri_bundle["tail_mean"]
    edge_close_frac = tri_bundle["close_frac_mean"]
    edge_degree_sum = degree[source_local] + degree[target_local]

    return np.column_stack(
        [
            edge_bridge,
            edge_curv_var,
            edge_margin,
            edge_support,
            edge_defect_med,
            edge_defect_tail,
            edge_close_frac,
            edge_degree_sum,
        ]
    )


def local_reconstruction_error(points: np.ndarray, n_neighbors: int, local_dim: int) -> np.ndarray:
    x = np.asarray(points, dtype=float)
    n = x.shape[0]
    k = max(3, min(int(n_neighbors), n - 1))
    d_eff = max(1, min(int(local_dim), x.shape[1] - 1, k - 1))

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(x)
    _, ind = nbrs.kneighbors(x)

    out = np.zeros(n, dtype=float)
    for i in range(n):
        neigh = ind[i, 1:]
        patch = x[neigh]
        patch = patch - patch.mean(axis=0, keepdims=True)
        if patch.shape[0] < 3:
            out[i] = 1.0
            continue
        # SVD of local neighborhood provides a stable local linearity estimate.
        _, s, _ = np.linalg.svd(patch, full_matrices=False)
        eigvals = (s**2) / max(1, patch.shape[0] - 1)
        total = float(np.sum(eigvals))
        if not np.isfinite(total) or total <= 1e-12:
            out[i] = 1.0
            continue
        keep = float(np.sum(eigvals[:d_eff]))
        out[i] = float(np.clip(1.0 - keep / total, 0.0, 1.0))
    return out


def local_id_from_points(points: np.ndarray, n_neighbors: int) -> np.ndarray:
    x = np.asarray(points, dtype=float)
    n = x.shape[0]
    k = max(3, min(int(n_neighbors), n - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(x)
    d_full, _ = nbrs.kneighbors(x)
    d_local = d_full[:, 1:]
    return BASE.local_id_mle(d_local)


def phase_boundary_edge_descriptors(
    recon_error: np.ndarray,
    id_jump: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
) -> np.ndarray:
    recon_z = BASE.zscore(recon_error)
    jump_z = BASE.zscore(id_jump)

    # "Phase boundary" proxy: moderate reconstruction error (near center) and low depth jump.
    node_boundary = -np.abs(recon_z)
    node_id_stable = -np.abs(jump_z)

    edge_boundary_mean = 0.5 * (node_boundary[source_local] + node_boundary[target_local])
    edge_boundary_min = np.minimum(node_boundary[source_local], node_boundary[target_local])
    edge_id_stable_mean = 0.5 * (node_id_stable[source_local] + node_id_stable[target_local])
    edge_pair_smooth = -np.abs(recon_z[source_local] - recon_z[target_local])

    return np.column_stack(
        [
            edge_boundary_mean,
            edge_boundary_min,
            edge_id_stable_mean,
            edge_pair_smooth,
        ]
    )


def pairwise_jaccard(nonzero_sets: list[set[int]]) -> float:
    vals: list[float] = []
    for a, b in itertools.combinations(nonzero_sets, 2):
        union = len(a | b)
        if union == 0:
            vals.append(1.0)
        else:
            vals.append(len(a & b) / union)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def pairwise_sign_agreement(sign_vectors: list[np.ndarray], nonzero_sets: list[set[int]]) -> float:
    vals: list[float] = []
    for i, j in itertools.combinations(range(len(sign_vectors)), 2):
        inter = nonzero_sets[i] & nonzero_sets[j]
        if len(inter) == 0:
            continue
        idx = np.asarray(sorted(inter), dtype=int)
        agree = np.mean(sign_vectors[i][idx] == sign_vectors[j][idx])
        vals.append(float(agree))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def run_h88_multiseed_sparse_descriptor_consensus(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    coef_registry: dict[tuple[str, str, int], list[tuple[str, np.ndarray]]] = {}

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_index, seed_tag in enumerate(H88_SEEDS):
            run_dir = run_map[seed_tag]
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            split_masks = BASE.build_split_masks(edge_df)

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H88_GENE_CAP))
                split_edges = split_edges.loc[
                    split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
                ].copy()
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
                if edge_gene_indices.size < 120:
                    continue

                symbol_map = BASE.build_symbol_map(split_edges)
                symbols = [symbol_map[int(g)] for g in edge_gene_indices]
                _, support_dir = BASE.build_support_matrices(
                    symbols_upper=symbols,
                    dorothea_map=dorothea_map,
                    omnipath_pairs=omnipath_pairs,
                    gene2go_upper=gene2go_upper,
                    string_map=string_map,
                )

                gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
                source_local_all = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
                target_local_all = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
                labels_all = split_edges["label"].to_numpy(dtype=int)

                for layer in H88_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    rng = np.random.default_rng(
                        35_880 + domain_index * 10_000 + seed_index * 1000 + split_index * 100 + layer
                    )
                    sample_idx = stratified_index_sample(labels_all, max_n=H88_EDGE_SAMPLE, rng=rng)
                    if sample_idx.size < 120:
                        continue

                    source_local = source_local_all[sample_idx]
                    target_local = target_local_all[sample_idx]
                    labels = labels_all[sample_idx]

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = BASE.reduce_points(
                        points,
                        n_components=20,
                        random_state=35_881 + domain_index * 10_000 + seed_index * 1000 + split_index * 100 + layer,
                    )
                    geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H88_NEIGHBORS)

                    _, h70_defect, tri_bundle = compute_h70_scores(
                        geodesic=geodesic,
                        support_dir=support_dir,
                        source_local=source_local,
                        target_local=target_local,
                        triangle_k=H88_TRIANGLE_K,
                    )
                    descriptors = build_sparse_descriptors(
                        geodesic=geodesic,
                        support_dir=support_dir,
                        source_local=source_local,
                        target_local=target_local,
                        points_pca=points_pca,
                        tri_bundle=tri_bundle,
                        n_neighbors=H88_NEIGHBORS,
                    )

                    edge_geodesic = geodesic[source_local, target_local]
                    bins = BASE.degree_bins(edge_geodesic, max_bins=6)

                    x_base = h70_defect[:, None]
                    x_blend = np.column_stack([h70_defect, descriptors])

                    auc_base = cross_validated_auc(
                        x_base,
                        labels,
                        random_state=35_882 + domain_index * 10_000 + seed_index * 1000 + split_index * 100 + layer,
                        penalty="none",
                        n_splits=H88_CV_SPLITS,
                    )
                    auc_blend = cross_validated_auc(
                        x_blend,
                        labels,
                        random_state=35_883 + domain_index * 10_000 + seed_index * 1000 + split_index * 100 + layer,
                        penalty="l1",
                        c_value=H88_L1_C,
                        n_splits=H88_CV_SPLITS,
                    )
                    delta_auc = float(auc_blend - auc_base) if np.isfinite(auc_base) and np.isfinite(auc_blend) else float("nan")

                    coef = fit_l1_coefficients(
                        x_blend,
                        labels,
                        c_value=H88_L1_C,
                        random_state=35_884 + domain_index * 10_000 + seed_index * 1000 + split_index * 100 + layer,
                    )
                    descriptor_coef = coef[1:]
                    coef_registry.setdefault((domain, split_regime, int(layer)), []).append((seed_tag, descriptor_coef))

                    null_feature = np.empty(H88_NULL_PERM, dtype=float)
                    null_endpoint = np.empty(H88_NULL_PERM, dtype=float)
                    null_label = np.empty(H88_NULL_PERM, dtype=float)

                    for perm_idx in range(H88_NULL_PERM):
                        shuffled_desc = np.column_stack(
                            [BASE.shuffle_within_bins(descriptors[:, j], bins, rng) for j in range(descriptors.shape[1])]
                        )
                        auc_feat = cross_validated_auc(
                            np.column_stack([h70_defect, shuffled_desc]),
                            labels,
                            random_state=35_890
                            + domain_index * 100_000
                            + seed_index * 10_000
                            + split_index * 1000
                            + layer * 10
                            + perm_idx,
                            penalty="l1",
                            c_value=H88_L1_C,
                            n_splits=H88_CV_SPLITS,
                        )
                        null_feature[perm_idx] = float(auc_feat - auc_base) if np.isfinite(auc_feat) and np.isfinite(auc_base) else float("nan")
                        null_rows.append(
                            {
                                "hypothesis_id": "H88",
                                "null_kind": "descriptor_shuffle_within_geodesic_bins",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_value": float(null_feature[perm_idx]),
                            }
                        )

                        target_swap = target_local.copy()
                        for b in np.unique(bins):
                            idx = np.where(bins == b)[0]
                            if idx.size > 1:
                                target_swap[idx] = rng.permutation(target_swap[idx])
                        tri_swap = BASE.multiscale_triangle_defect_features(
                            geodesic=geodesic,
                            source_local=source_local,
                            target_local=target_swap,
                            k_values=H88_TRIANGLE_K,
                        )
                        desc_swap = build_sparse_descriptors(
                            geodesic=geodesic,
                            support_dir=support_dir,
                            source_local=source_local,
                            target_local=target_swap,
                            points_pca=points_pca,
                            tri_bundle=tri_swap,
                            n_neighbors=H88_NEIGHBORS,
                        )
                        auc_swap = cross_validated_auc(
                            np.column_stack([h70_defect, desc_swap]),
                            labels,
                            random_state=36_090
                            + domain_index * 100_000
                            + seed_index * 10_000
                            + split_index * 1000
                            + layer * 10
                            + perm_idx,
                            penalty="l1",
                            c_value=H88_L1_C,
                            n_splits=H88_CV_SPLITS,
                        )
                        null_endpoint[perm_idx] = float(auc_swap - auc_base) if np.isfinite(auc_swap) and np.isfinite(auc_base) else float("nan")
                        null_rows.append(
                            {
                                "hypothesis_id": "H88",
                                "null_kind": "endpoint_swap_within_geodesic_bins",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_value": float(null_endpoint[perm_idx]),
                            }
                        )

                        labels_perm = BASE.stratified_shuffle(labels, bins, rng).astype(int)
                        auc_lp_base = cross_validated_auc(
                            x_base,
                            labels_perm,
                            random_state=36_290
                            + domain_index * 100_000
                            + seed_index * 10_000
                            + split_index * 1000
                            + layer * 10
                            + perm_idx,
                            penalty="none",
                            n_splits=H88_CV_SPLITS,
                        )
                        auc_lp_blend = cross_validated_auc(
                            x_blend,
                            labels_perm,
                            random_state=36_490
                            + domain_index * 100_000
                            + seed_index * 10_000
                            + split_index * 1000
                            + layer * 10
                            + perm_idx,
                            penalty="l1",
                            c_value=H88_L1_C,
                            n_splits=H88_CV_SPLITS,
                        )
                        null_label[perm_idx] = (
                            float(auc_lp_blend - auc_lp_base)
                            if np.isfinite(auc_lp_blend) and np.isfinite(auc_lp_base)
                            else float("nan")
                        )
                        null_rows.append(
                            {
                                "hypothesis_id": "H88",
                                "null_kind": "label_permutation",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_value": float(null_label[perm_idx]),
                            }
                        )

                    all_null = np.concatenate([null_feature, null_endpoint, null_label])
                    q95 = float(np.nanquantile(all_null, 0.95))
                    p_feat = BASE.empirical_upper_tail_p(delta_auc, null_feature)
                    p_end = BASE.empirical_upper_tail_p(delta_auc, null_endpoint)
                    p_lab = BASE.empirical_upper_tail_p(delta_auc, null_label)
                    p_best = np.nanmin(np.array([p_feat, p_end, p_lab], dtype=float))

                    rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "n_edges_eval": int(labels.size),
                            "n_positive_eval": int(labels.sum()),
                            "auc_h70_baseline": float(auc_base),
                            "auc_sparse_descriptor_blend": float(auc_blend),
                            "delta_auc_sparse_descriptor_blend_minus_h70": float(delta_auc),
                            "descriptor_nonzero_count": int(np.sum(np.abs(descriptor_coef) > 1e-8)),
                            "q95_null_delta_auc": float(q95),
                            "null_gap_q95_delta_auc": float(delta_auc - q95),
                            "p_feature_shuffle_upper": float(p_feat),
                            "p_endpoint_swap_upper": float(p_end),
                            "p_label_shuffle_upper": float(p_lab),
                            "p_best_upper": float(p_best),
                        }
                    )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h88_multiseed_sparse_descriptor_by_seed_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h88_multiseed_sparse_descriptor_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    stability_rows: list[dict[str, object]] = []
    for (domain, split_regime, layer), items in sorted(coef_registry.items()):
        if len(items) < 2:
            continue
        seeds_here = [seed for seed, _ in items]
        vecs = [coef for _, coef in items]
        nonzero_sets = [set(np.where(np.abs(v) > 1e-8)[0].tolist()) for v in vecs]
        signs = [np.sign(v) for v in vecs]

        jacc = pairwise_jaccard(nonzero_sets)
        sign_agree = pairwise_sign_agreement(signs, nonzero_sets)
        core_set = set.intersection(*nonzero_sets) if nonzero_sets else set()

        mask = (
            (by_row_df["domain"] == domain)
            & (by_row_df["split_regime"] == split_regime)
            & (by_row_df["layer"] == int(layer))
        )
        grp = by_row_df.loc[mask]

        stability_rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "layer": int(layer),
                "n_seeds": int(len(items)),
                "seed_tags": ";".join(seeds_here),
                "descriptor_nonzero_jaccard_mean": float(jacc),
                "descriptor_sign_agreement_mean": float(sign_agree),
                "descriptor_core_size": int(len(core_set)),
                "mean_delta_auc_sparse_descriptor_blend_minus_h70": float(
                    grp["delta_auc_sparse_descriptor_blend_minus_h70"].mean()
                )
                if not grp.empty
                else float("nan"),
                "mean_null_gap_q95_delta_auc": float(grp["null_gap_q95_delta_auc"].mean()) if not grp.empty else float("nan"),
            }
        )

    stability_df = pd.DataFrame(stability_rows)
    if not stability_df.empty:
        stability_df = stability_df.sort_values(["domain", "split_regime", "layer"])
    stability_path = ITER_DIR / "h88_multiseed_sparse_descriptor_stability.csv"
    stability_df.to_csv(stability_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            stab = stability_df.loc[
                (stability_df["domain"] == domain) & (stability_df["split_regime"] == split_regime)
            ]
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_sparse_descriptor_blend_minus_h70": float(
                        group["delta_auc_sparse_descriptor_blend_minus_h70"].mean()
                    ),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_sparse_descriptor_blend_minus_h70"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                    "mean_descriptor_nonzero_jaccard": float(stab["descriptor_nonzero_jaccard_mean"].mean())
                    if not stab.empty
                    else float("nan"),
                    "mean_descriptor_sign_agreement": float(stab["descriptor_sign_agreement_mean"].mean())
                    if not stab.empty
                    else float("nan"),
                    "mean_descriptor_core_size": float(stab["descriptor_core_size"].mean()) if not stab.empty else float("nan"),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h88_multiseed_sparse_descriptor_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_sparse_descriptor_blend_minus_h70"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_sparse_descriptor_blend_minus_h70"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95_delta_auc"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "mean_descriptor_jaccard": float(summary_df["mean_descriptor_nonzero_jaccard"].mean())
        if not summary_df.empty
        else float("nan"),
        "artifact_paths": {
            "by_seed_split_layer": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
            "stability": str(stability_path),
        },
    }


def run_h89_phase_boundary_screen(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        seed_tag = "seed42_main"
        run_dir = run_map[seed_tag]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H89_GENE_CAP))
            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
            ].copy()
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
            if edge_gene_indices.size < 120:
                continue

            symbol_map = BASE.build_symbol_map(split_edges)
            symbols = [symbol_map[int(g)] for g in edge_gene_indices]
            _, support_dir = BASE.build_support_matrices(
                symbols_upper=symbols,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )

            gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
            source_local_all = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
            target_local_all = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
            labels_all = split_edges["label"].to_numpy(dtype=int)

            rng_split = np.random.default_rng(35_890 + domain_index * 100 + split_index)
            sample_idx = stratified_index_sample(labels_all, max_n=H89_EDGE_SAMPLE, rng=rng_split)
            if sample_idx.size < 120:
                continue

            source_local = source_local_all[sample_idx]
            target_local = target_local_all[sample_idx]
            labels = labels_all[sample_idx]

            # Precompute per-layer node metrics so depth-jump descriptors are consistent per split.
            layer_node_recon: dict[int, np.ndarray] = {}
            layer_node_id: dict[int, np.ndarray] = {}
            layer_h70: dict[int, np.ndarray] = {}
            layer_bins: dict[int, np.ndarray] = {}

            for layer in H89_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue
                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=24,
                    random_state=35_891 + domain_index * 10_000 + split_index * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H89_NEIGHBORS)

                _, h70_defect, _ = compute_h70_scores(
                    geodesic=geodesic,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H88_TRIANGLE_K,
                )
                edge_geodesic = geodesic[source_local, target_local]

                layer_h70[layer] = h70_defect
                layer_bins[layer] = BASE.degree_bins(edge_geodesic, max_bins=6)
                layer_node_recon[layer] = local_reconstruction_error(
                    points_pca,
                    n_neighbors=H89_ID_K,
                    local_dim=H89_LOCAL_DIM,
                )
                layer_node_id[layer] = local_id_from_points(points_pca, n_neighbors=H89_ID_K)

            available_layers = sorted(layer_h70.keys())
            if len(available_layers) < 2:
                continue

            layer_node_jump: dict[int, np.ndarray] = {}
            for idx, layer in enumerate(available_layers):
                prev_layer = available_layers[max(0, idx - 1)]
                next_layer = available_layers[min(len(available_layers) - 1, idx + 1)]
                id_here = layer_node_id[layer]
                id_prev = layer_node_id[prev_layer]
                id_next = layer_node_id[next_layer]
                layer_node_jump[layer] = 0.5 * (np.abs(id_here - id_prev) + np.abs(id_next - id_here))

            phase_desc_by_layer: dict[int, np.ndarray] = {}
            for layer in available_layers:
                phase_desc_by_layer[layer] = phase_boundary_edge_descriptors(
                    recon_error=layer_node_recon[layer],
                    id_jump=layer_node_jump[layer],
                    source_local=source_local,
                    target_local=target_local,
                )

            layer_to_pos = {layer: idx for idx, layer in enumerate(available_layers)}

            for layer in available_layers:
                h70_defect = layer_h70[layer]
                bins = layer_bins[layer]
                phase_desc = phase_desc_by_layer[layer]

                x_base = h70_defect[:, None]
                x_phase = np.column_stack([h70_defect, phase_desc])

                auc_base = cross_validated_auc(
                    x_base,
                    labels,
                    random_state=35_892 + domain_index * 10_000 + split_index * 100 + layer,
                    penalty="none",
                    n_splits=H89_CV_SPLITS,
                )
                auc_phase = cross_validated_auc(
                    x_phase,
                    labels,
                    random_state=35_893 + domain_index * 10_000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H89_L1_C,
                    n_splits=H89_CV_SPLITS,
                )
                delta_auc = float(auc_phase - auc_base) if np.isfinite(auc_phase) and np.isfinite(auc_base) else float("nan")

                coef = fit_l1_coefficients(
                    x_phase,
                    labels,
                    c_value=H89_L1_C,
                    random_state=35_894 + domain_index * 10_000 + split_index * 100 + layer,
                )

                null_layer = np.empty(H89_NULL_PERM, dtype=float)
                null_feature = np.empty(H89_NULL_PERM, dtype=float)
                null_label = np.empty(H89_NULL_PERM, dtype=float)

                for perm_idx in range(H89_NULL_PERM):
                    # Layer-order null: swap descriptors across depth while keeping labels and baseline fixed.
                    perm_layers = available_layers.copy()
                    rng_split.shuffle(perm_layers)
                    mapped_layer = perm_layers[layer_to_pos[layer]]
                    desc_layer_perm = phase_desc_by_layer[mapped_layer]
                    auc_layer = cross_validated_auc(
                        np.column_stack([h70_defect, desc_layer_perm]),
                        labels,
                        random_state=36_892 + domain_index * 100_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H89_L1_C,
                        n_splits=H89_CV_SPLITS,
                    )
                    null_layer[perm_idx] = float(auc_layer - auc_base) if np.isfinite(auc_layer) and np.isfinite(auc_base) else float("nan")
                    null_rows.append(
                        {
                            "hypothesis_id": "H89",
                            "null_kind": "layer_order_permutation",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_layer[perm_idx]),
                        }
                    )

                    shuffled_desc = np.column_stack(
                        [BASE.shuffle_within_bins(phase_desc[:, j], bins, rng_split) for j in range(phase_desc.shape[1])]
                    )
                    auc_feat = cross_validated_auc(
                        np.column_stack([h70_defect, shuffled_desc]),
                        labels,
                        random_state=37_892 + domain_index * 100_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H89_L1_C,
                        n_splits=H89_CV_SPLITS,
                    )
                    null_feature[perm_idx] = float(auc_feat - auc_base) if np.isfinite(auc_feat) and np.isfinite(auc_base) else float("nan")
                    null_rows.append(
                        {
                            "hypothesis_id": "H89",
                            "null_kind": "descriptor_shuffle_within_geodesic_bins",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_feature[perm_idx]),
                        }
                    )

                    labels_perm = BASE.stratified_shuffle(labels, bins, rng_split).astype(int)
                    auc_lp_base = cross_validated_auc(
                        x_base,
                        labels_perm,
                        random_state=38_892 + domain_index * 100_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="none",
                        n_splits=H89_CV_SPLITS,
                    )
                    auc_lp_phase = cross_validated_auc(
                        x_phase,
                        labels_perm,
                        random_state=39_892 + domain_index * 100_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H89_L1_C,
                        n_splits=H89_CV_SPLITS,
                    )
                    null_label[perm_idx] = (
                        float(auc_lp_phase - auc_lp_base)
                        if np.isfinite(auc_lp_phase) and np.isfinite(auc_lp_base)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H89",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_layer, null_feature, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_layer = BASE.empirical_upper_tail_p(delta_auc, null_layer)
                p_feat = BASE.empirical_upper_tail_p(delta_auc, null_feature)
                p_lab = BASE.empirical_upper_tail_p(delta_auc, null_label)
                p_best = np.nanmin(np.array([p_layer, p_feat, p_lab], dtype=float))

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70_baseline": float(auc_base),
                        "auc_phase_boundary_blend": float(auc_phase),
                        "delta_auc_phase_boundary_minus_h70": float(delta_auc),
                        "phase_nonzero_count": int(np.sum(np.abs(coef[1:]) > 1e-8)),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95_delta_auc": float(delta_auc - q95),
                        "p_layer_order_upper": float(p_layer),
                        "p_feature_shuffle_upper": float(p_feat),
                        "p_label_shuffle_upper": float(p_lab),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h89_phase_boundary_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h89_phase_boundary_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_phase_boundary_minus_h70": float(group["delta_auc_phase_boundary_minus_h70"].mean()),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_phase_boundary_minus_h70"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h89_phase_boundary_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_phase_boundary_minus_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_phase_boundary_minus_h70"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95_delta_auc"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_domain_split_layer": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h90_topology_stability_screen(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        seed_tag = "seed42_main"
        run_dir = run_map[seed_tag]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H90_GENE_CAP))
            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
            ].copy()
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
            if edge_gene_indices.size < 120:
                continue

            symbol_map = BASE.build_symbol_map(split_edges)
            symbols = [symbol_map[int(g)] for g in edge_gene_indices]
            _, support_dir = BASE.build_support_matrices(
                symbols_upper=symbols,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )

            gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
            source_local_all = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
            target_local_all = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
            labels_all = split_edges["label"].to_numpy(dtype=int)

            for layer in H90_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                rng = np.random.default_rng(35_990 + domain_index * 1000 + split_index * 100 + layer)
                sample_idx = stratified_index_sample(labels_all, max_n=H90_EDGE_SAMPLE, rng=rng)
                if sample_idx.size < 120:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=22,
                    random_state=35_991 + domain_index * 1000 + split_index * 100 + layer,
                )
                geodesic_base = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H90_NEIGHBORS)

                _, h70_base, _ = compute_h70_scores(
                    geodesic=geodesic_base,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H90_TRIANGLE_K,
                )

                perturbed_scores = []

                for k_var in H90_K_VARIANTS:
                    geodesic_k = BASE.geodesic_distance_matrix(points_pca, n_neighbors=k_var)
                    _, h70_k, _ = compute_h70_scores(
                        geodesic=geodesic_k,
                        support_dir=support_dir,
                        source_local=source_local,
                        target_local=target_local,
                        triangle_k=H90_TRIANGLE_K,
                    )
                    perturbed_scores.append(h70_k)

                for sigma in H90_JITTER_SIGMAS:
                    jitter = rng.normal(loc=0.0, scale=float(sigma), size=points_pca.shape)
                    points_j = points_pca + jitter
                    points_j = PCA(
                        n_components=min(points_j.shape[1], points_j.shape[0] - 1, 22),
                        svd_solver="randomized",
                        random_state=36_000 + int(1000 * sigma),
                    ).fit_transform(points_j)
                    geodesic_j = BASE.geodesic_distance_matrix(points_j, n_neighbors=H90_NEIGHBORS)
                    _, h70_j, _ = compute_h70_scores(
                        geodesic=geodesic_j,
                        support_dir=support_dir,
                        source_local=source_local,
                        target_local=target_local,
                        triangle_k=H90_TRIANGLE_K,
                    )
                    perturbed_scores.append(h70_j)

                perturbed = np.vstack(perturbed_scores)
                delta_mat = perturbed - h70_base[None, :]
                stability = -np.median(np.abs(delta_mat), axis=0)
                stability_dispersion = -np.std(delta_mat, axis=0)
                stability_features = np.column_stack([stability, stability_dispersion])

                edge_geodesic = geodesic_base[source_local, target_local]
                bins = BASE.degree_bins(edge_geodesic, max_bins=6)

                x_base = h70_base[:, None]
                x_aug = np.column_stack([h70_base, stability_features])

                auc_base = cross_validated_auc(
                    x_base,
                    labels,
                    random_state=35_992 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="none",
                    n_splits=H90_CV_SPLITS,
                )
                auc_aug = cross_validated_auc(
                    x_aug,
                    labels,
                    random_state=35_993 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H90_L1_C,
                    n_splits=H90_CV_SPLITS,
                )
                delta_auc = float(auc_aug - auc_base) if np.isfinite(auc_aug) and np.isfinite(auc_base) else float("nan")

                stability_pos_neg_gap = float(np.mean(stability[labels == 1]) - np.mean(stability[labels == 0]))

                null_feature = np.empty(H90_NULL_PERM, dtype=float)
                null_profile = np.empty(H90_NULL_PERM, dtype=float)
                null_label = np.empty(H90_NULL_PERM, dtype=float)

                for perm_idx in range(H90_NULL_PERM):
                    shuffled_desc = np.column_stack(
                        [BASE.shuffle_within_bins(stability_features[:, j], bins, rng) for j in range(stability_features.shape[1])]
                    )
                    auc_feat = cross_validated_auc(
                        np.column_stack([h70_base, shuffled_desc]),
                        labels,
                        random_state=36_992 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H90_L1_C,
                        n_splits=H90_CV_SPLITS,
                    )
                    null_feature[perm_idx] = float(auc_feat - auc_base) if np.isfinite(auc_feat) and np.isfinite(auc_base) else float("nan")
                    null_rows.append(
                        {
                            "hypothesis_id": "H90",
                            "null_kind": "stability_feature_shuffle_within_geodesic_bins",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_feature[perm_idx]),
                        }
                    )

                    shuffled_profiles = np.vstack(
                        [BASE.shuffle_within_bins(perturbed[p], bins, rng) for p in range(perturbed.shape[0])]
                    )
                    delta_shuf = shuffled_profiles - h70_base[None, :]
                    stability_shuf = -np.median(np.abs(delta_shuf), axis=0)
                    disp_shuf = -np.std(delta_shuf, axis=0)
                    auc_profile = cross_validated_auc(
                        np.column_stack([h70_base, stability_shuf, disp_shuf]),
                        labels,
                        random_state=37_992 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H90_L1_C,
                        n_splits=H90_CV_SPLITS,
                    )
                    null_profile[perm_idx] = (
                        float(auc_profile - auc_base) if np.isfinite(auc_profile) and np.isfinite(auc_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H90",
                            "null_kind": "perturbation_profile_shuffle_within_geodesic_bins",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_profile[perm_idx]),
                        }
                    )

                    labels_perm = BASE.stratified_shuffle(labels, bins, rng).astype(int)
                    auc_lp_base = cross_validated_auc(
                        x_base,
                        labels_perm,
                        random_state=38_992 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="none",
                        n_splits=H90_CV_SPLITS,
                    )
                    auc_lp_aug = cross_validated_auc(
                        x_aug,
                        labels_perm,
                        random_state=39_992 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H90_L1_C,
                        n_splits=H90_CV_SPLITS,
                    )
                    null_label[perm_idx] = (
                        float(auc_lp_aug - auc_lp_base)
                        if np.isfinite(auc_lp_aug) and np.isfinite(auc_lp_base)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H90",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_feature, null_profile, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_feat = BASE.empirical_upper_tail_p(delta_auc, null_feature)
                p_prof = BASE.empirical_upper_tail_p(delta_auc, null_profile)
                p_lab = BASE.empirical_upper_tail_p(delta_auc, null_label)
                p_best = np.nanmin(np.array([p_feat, p_prof, p_lab], dtype=float))

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70_baseline": float(auc_base),
                        "auc_stability_blend": float(auc_aug),
                        "delta_auc_stability_blend_minus_h70": float(delta_auc),
                        "stability_pos_minus_neg": float(stability_pos_neg_gap),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95_delta_auc": float(delta_auc - q95),
                        "p_feature_shuffle_upper": float(p_feat),
                        "p_profile_shuffle_upper": float(p_prof),
                        "p_label_shuffle_upper": float(p_lab),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h90_topology_stability_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h90_topology_stability_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_stability_blend_minus_h70": float(group["delta_auc_stability_blend_minus_h70"].mean()),
                    "mean_stability_pos_minus_neg": float(group["stability_pos_minus_neg"].mean()),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_stability_blend_minus_h70"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h90_topology_stability_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_stability_blend_minus_h70"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_stability_blend_minus_h70"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95_delta_auc"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_domain_split_layer": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def main() -> None:
    ensure_required_inputs()

    dorothea_map = BASE.load_dorothea_score_map()
    omnipath_pairs = BASE.load_omnipath_pairs()
    gene2go_upper = BASE.load_gene2go_upper()
    string_map = BASE.load_string_scores_from_cache(BASE.STRING_CACHE_PATH)

    h88_summary = run_h88_multiseed_sparse_descriptor_consensus(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h89_summary = run_h89_phase_boundary_screen(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h90_summary = run_h90_topology_stability_screen(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    summary = {
        "iteration": "iter_0035",
        "h88": h88_summary,
        "h89": h89_summary,
        "h90": h90_summary,
    }
    summary_path = ITER_DIR / "iter0035_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
