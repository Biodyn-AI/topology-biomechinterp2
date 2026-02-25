from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ITER_DIR = Path("iterations/iter_0040")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reuse the mature geometry/topology utility surface from iter_0028.
BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base")


# H103 / N508: interaction-only derivative rescue over H91/H93-like backbone.
H103_SEED = "seed42_main"
H103_LAYERS = [0, 3, 7, 11]
H103_GENE_CAP = 170
H103_NEIGHBORS = 12
H103_TRIANGLE_K = [8, 12, 16]
H103_EDGE_SAMPLE = 220
H103_CV_SPLITS = 4
H103_L1_C = 0.22
H103_NULL_PERM = 16
H103_DIST_QUANTILES = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
H103_MARGIN_QUANTILES = [0.45, 0.60, 0.75]
H103_INTERACTION_COLS = [0, 1, 2, 3, 8, 9, 10, 11, 16]

# H104 / N520: depth motif grammar on layered descriptor states.
H104_SEED = "seed42_main"
H104_LAYERS = [0, 3, 7, 11]
H104_GENE_CAP = 180
H104_NEIGHBORS = 12
H104_TRIANGLE_K = [8, 12, 16]
H104_EDGE_SAMPLE = 230
H104_CV_SPLITS = 4
H104_NULL_PERM = 16
H104_TOKEN_BINS = 3

# H105 / N519: STRING-conditioned null calibration check.
H105_SEED = "seed42_main"
H105_LAYERS = [7, 11]
H105_GENE_CAP = 180
H105_NEIGHBORS = 12
H105_TRIANGLE_K = [8, 12, 16]
H105_EDGE_SAMPLE = 240
H105_CV_SPLITS = 4
H105_L1_C = 0.22
H105_NULL_PERM = 20


def ensure_required_inputs() -> None:
    required_paths = [
        BASE.DOROTHEA_PATH,
        BASE.GENE2GO_PATH,
        BASE.OMNIPATH_INTERACTIONS_PATH,
        BASE.STRING_CACHE_PATH,
    ]
    for run_map in BASE.SCGPT_RUNS_BY_DOMAIN.values():
        run_dir = run_map[H103_SEED]
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


def confidence_weighted_geodesic(geodesic: np.ndarray, support_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sym_support = 0.5 * (support_dir + support_dir.T)
    sym_support = np.clip(sym_support, 0.0, 1.0)

    weighted = geodesic / (0.35 + sym_support)
    weighted = np.asarray(weighted, dtype=float)
    np.fill_diagonal(weighted, 0.0)
    return weighted, sym_support


def edge_degree_sum(points: np.ndarray, n_neighbors: int, source_local: np.ndarray, target_local: np.ndarray) -> np.ndarray:
    knn_edges = BASE.build_knn_edge_array(points=points, n_neighbors=n_neighbors)
    neighbors = BASE.adjacency_neighbors(points.shape[0], knn_edges)
    deg = np.asarray([len(nb) for nb in neighbors], dtype=float)
    return deg[source_local] + deg[target_local]


def build_edge_strata(edge_length: np.ndarray, degree_sum: np.ndarray, max_len_bins: int, max_deg_bins: int) -> np.ndarray:
    bins_len = BASE.degree_bins(edge_length, max_bins=max_len_bins)
    bins_deg = BASE.degree_bins(degree_sum, max_bins=max_deg_bins)
    return (bins_len * 16 + bins_deg).astype(int)


def filtration_connectivity_trajectories(
    dist_matrix: np.ndarray,
    margin_matrix: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    dist_quantiles: list[float],
    margin_quantiles: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = dist_matrix.shape[0]
    upper_i, upper_j = np.triu_indices(n_nodes, k=1)
    dist_vals = dist_matrix[upper_i, upper_j]
    margin_vals = margin_matrix[upper_i, upper_j]

    dist_thresholds = [float(np.quantile(dist_vals, q)) for q in dist_quantiles]
    margin_thresholds = [float(np.quantile(margin_vals, q)) for q in margin_quantiles]

    traj_one = np.zeros((source_local.size, len(dist_thresholds)), dtype=float)
    traj_two = np.zeros((source_local.size, len(dist_thresholds)), dtype=float)

    for d_idx, d_thr in enumerate(dist_thresholds):
        keep_dist = dist_vals <= d_thr
        labels_one, _, _ = BASE.component_labels_from_upper_mask(
            n_nodes=n_nodes,
            upper_i=upper_i,
            upper_j=upper_j,
            keep_mask=keep_dist,
        )
        traj_one[:, d_idx] = (labels_one[source_local] == labels_one[target_local]).astype(float)

        margin_conn = np.zeros((len(margin_thresholds), source_local.size), dtype=float)
        for m_idx, m_thr in enumerate(margin_thresholds):
            keep = keep_dist & (margin_vals >= m_thr)
            labels_two, _, _ = BASE.component_labels_from_upper_mask(
                n_nodes=n_nodes,
                upper_i=upper_i,
                upper_j=upper_j,
                keep_mask=keep,
            )
            margin_conn[m_idx] = (labels_two[source_local] == labels_two[target_local]).astype(float)
        traj_two[:, d_idx] = margin_conn.mean(axis=0)

    return traj_one, traj_two


def first_true_index(mask: np.ndarray) -> np.ndarray:
    out = np.argmax(mask, axis=1).astype(float)
    none_true = ~mask.any(axis=1)
    out[none_true] = float(mask.shape[1])
    return out


def derivative_spectrum_features(trajectory: np.ndarray) -> np.ndarray:
    traj = np.asarray(trajectory, dtype=float)
    n_edges, n_steps = traj.shape

    if n_steps < 2:
        return np.zeros((n_edges, 8), dtype=float)

    d1 = np.diff(traj, axis=1)
    if d1.shape[1] >= 2:
        d2 = np.diff(d1, axis=1)
    else:
        d2 = np.zeros((n_edges, 1), dtype=float)

    auc = np.mean(traj, axis=1)
    end_minus_start = traj[:, -1] - traj[:, 0]
    onset = first_true_index(traj > 0.5) / max(1.0, float(n_steps - 1))

    abs_d1 = np.abs(d1)
    mass = np.sum(abs_d1, axis=1)
    prob = abs_d1 / np.clip(mass[:, None], 1e-8, None)
    entropy = -np.sum(prob * np.log(np.clip(prob, 1e-8, None)), axis=1) / np.log(max(2, d1.shape[1]))

    d1_mean = np.mean(d1, axis=1)
    d1_abs_mean = np.mean(abs_d1, axis=1)
    peak_idx = np.argmax(abs_d1, axis=1).astype(float) / max(1.0, float(abs_d1.shape[1] - 1))
    d2_abs_mean = np.mean(np.abs(d2), axis=1)

    return np.column_stack(
        [
            auc,
            onset,
            end_minus_start,
            entropy,
            d1_mean,
            d1_abs_mean,
            peak_idx,
            d2_abs_mean,
        ]
    )


def build_h101_feature_matrix(traj_one: np.ndarray, traj_two: np.ndarray) -> np.ndarray:
    f1 = derivative_spectrum_features(traj_one)
    f2 = derivative_spectrum_features(traj_two)
    diff = f2 - f1
    diff_sel = diff[:, [0, 1, 2, 3, 5, 7]]
    return np.column_stack([f1, f2, diff_sel])


def permute_trajectory_order(trajectory: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(trajectory.shape[1])
    return trajectory[:, order]


def quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    # Rank-based bins keep tokenization stable under repeated values.
    ranks = pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).to_numpy(dtype=float)
    bins = np.minimum((ranks * n_bins).astype(int), n_bins - 1)
    return bins.astype(int)


def build_depth_tokens(
    base_by_layer: dict[int, np.ndarray],
    gain_by_layer: dict[int, np.ndarray],
    layers: list[int],
    n_bins: int,
) -> np.ndarray:
    tokens = []
    for layer in layers:
        b = quantile_bins(base_by_layer[layer], n_bins=n_bins)
        g = quantile_bins(gain_by_layer[layer], n_bins=n_bins)
        token = b * n_bins + g
        tokens.append(token)
    return np.column_stack(tokens)


def motif_cv_scores(
    tokens: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    n_splits: int,
    alpha: float = 1.0,
) -> np.ndarray:
    x = np.asarray(tokens, dtype=int)
    y = np.asarray(labels, dtype=int)

    if x.ndim != 2 or x.shape[0] != y.size or np.unique(y).size < 2:
        return np.full(y.size, np.nan, dtype=float)

    max_splits = min(n_splits, min_class_count(y))
    if max_splits < 2:
        return np.full(y.size, np.nan, dtype=float)

    n_layers = x.shape[1]
    n_tokens = int(np.max(x)) + 1
    out_scores = np.full(y.shape[0], np.nan, dtype=float)

    cv = StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=random_state)

    for tr, te in cv.split(x, y):
        pos_counts = np.full((n_layers - 1, n_tokens, n_tokens), alpha, dtype=float)
        neg_counts = np.full((n_layers - 1, n_tokens, n_tokens), alpha, dtype=float)

        x_tr = x[tr]
        y_tr = y[tr]
        for row_idx in range(x_tr.shape[0]):
            arr = x_tr[row_idx]
            is_pos = int(y_tr[row_idx]) == 1
            for step in range(n_layers - 1):
                src = int(arr[step])
                tgt = int(arr[step + 1])
                if is_pos:
                    pos_counts[step, src, tgt] += 1.0
                else:
                    neg_counts[step, src, tgt] += 1.0

        pos_probs = pos_counts / np.clip(pos_counts.sum(axis=2, keepdims=True), 1e-8, None)
        neg_probs = neg_counts / np.clip(neg_counts.sum(axis=2, keepdims=True), 1e-8, None)

        x_te = x[te]
        fold_scores = np.zeros(x_te.shape[0], dtype=float)
        for row_idx in range(x_te.shape[0]):
            arr = x_te[row_idx]
            score = 0.0
            for step in range(n_layers - 1):
                src = int(arr[step])
                tgt = int(arr[step + 1])
                p_pos = float(np.clip(pos_probs[step, src, tgt], 1e-8, 1.0))
                p_neg = float(np.clip(neg_probs[step, src, tgt], 1e-8, 1.0))
                score += float(np.log(p_pos / p_neg))
            fold_scores[row_idx] = score
        out_scores[te] = fold_scores

    return out_scores


def motif_cv_auc(tokens: np.ndarray, labels: np.ndarray, random_state: int, n_splits: int) -> tuple[float, np.ndarray]:
    scores = motif_cv_scores(tokens=tokens, labels=labels, random_state=random_state, n_splits=n_splits)
    auc = BASE.safe_auc(np.asarray(labels, dtype=int), scores)
    return auc, scores


def run_h103_interaction_derivative_rescue(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H103_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H103_GENE_CAP))
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

            for layer in H103_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                rng = np.random.default_rng(40_100 + domain_index * 1000 + split_index * 100 + layer)
                sample_idx = stratified_index_sample(labels_all, max_n=H103_EDGE_SAMPLE, rng=rng)
                if sample_idx.size < 120:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=22,
                    random_state=40_101 + domain_index * 1000 + split_index * 100 + layer,
                )

                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H103_NEIGHBORS)
                geodesic_w, support_sym = confidence_weighted_geodesic(geodesic, support_dir)

                _, h70_base, _ = compute_h70_scores(
                    geodesic=geodesic,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H103_TRIANGLE_K,
                )
                _, h70_weighted, _ = compute_h70_scores(
                    geodesic=geodesic_w,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H103_TRIANGLE_K,
                )

                weighted_gain = h70_weighted - h70_base
                edge_margin = support_dir[source_local, target_local] - support_dir[target_local, source_local]
                edge_confidence = support_sym[source_local, target_local]

                margin_matrix = np.abs(support_dir - support_dir.T)
                traj_one, traj_two = filtration_connectivity_trajectories(
                    dist_matrix=geodesic_w,
                    margin_matrix=margin_matrix,
                    source_local=source_local,
                    target_local=target_local,
                    dist_quantiles=H103_DIST_QUANTILES,
                    margin_quantiles=H103_MARGIN_QUANTILES,
                )
                deriv_features = build_h101_feature_matrix(traj_one=traj_one, traj_two=traj_two)
                interaction_base = deriv_features[:, H103_INTERACTION_COLS]

                x_backbone = np.column_stack([h70_base, h70_weighted, weighted_gain, edge_margin, edge_confidence])

                interaction_block = np.column_stack(
                    [
                        weighted_gain[:, None] * interaction_base,
                        edge_margin[:, None] * interaction_base,
                        edge_confidence[:, None] * interaction_base,
                    ]
                )
                x_interaction = np.column_stack([x_backbone, interaction_block])

                auc_backbone = cross_validated_auc(
                    x_backbone,
                    labels,
                    random_state=40_102 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H103_L1_C,
                    n_splits=H103_CV_SPLITS,
                )
                auc_interaction = cross_validated_auc(
                    x_interaction,
                    labels,
                    random_state=40_103 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H103_L1_C,
                    n_splits=H103_CV_SPLITS,
                )
                delta_auc = (
                    float(auc_interaction - auc_backbone)
                    if np.isfinite(auc_backbone) and np.isfinite(auc_interaction)
                    else float("nan")
                )

                interaction_signal = float(
                    np.mean(interaction_block[labels == 1, 0]) - np.mean(interaction_block[labels == 0, 0])
                )

                edge_length = geodesic_w[source_local, target_local]
                degree_sum = edge_degree_sum(
                    points=points_pca,
                    n_neighbors=H103_NEIGHBORS,
                    source_local=source_local,
                    target_local=target_local,
                )
                strata = build_edge_strata(edge_length=edge_length, degree_sum=degree_sum, max_len_bins=6, max_deg_bins=4)

                null_order = np.empty(H103_NULL_PERM, dtype=float)
                null_partner = np.empty(H103_NULL_PERM, dtype=float)
                null_label = np.empty(H103_NULL_PERM, dtype=float)

                for perm_idx in range(H103_NULL_PERM):
                    # Null 1: filtration-quantile order permutation before derivative extraction.
                    traj_one_perm = permute_trajectory_order(traj_one, rng)
                    traj_two_perm = permute_trajectory_order(traj_two, rng)
                    deriv_perm = build_h101_feature_matrix(traj_one=traj_one_perm, traj_two=traj_two_perm)
                    int_base_perm = deriv_perm[:, H103_INTERACTION_COLS]
                    int_block_perm = np.column_stack(
                        [
                            weighted_gain[:, None] * int_base_perm,
                            edge_margin[:, None] * int_base_perm,
                            edge_confidence[:, None] * int_base_perm,
                        ]
                    )
                    x_order = np.column_stack([x_backbone, int_block_perm])
                    auc_order = cross_validated_auc(
                        x_order,
                        labels,
                        random_state=40_104 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H103_L1_C,
                        n_splits=H103_CV_SPLITS,
                    )
                    null_order[perm_idx] = (
                        float(auc_order - auc_backbone) if np.isfinite(auc_order) and np.isfinite(auc_backbone) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H103",
                            "null_kind": "derivative_quantile_order_permutation",
                            "domain": domain,
                            "seed_tag": H103_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_order[perm_idx]),
                        }
                    )

                    # Null 2: interaction-partner shuffle within geodesic/degree strata.
                    gain_perm = BASE.shuffle_within_bins(weighted_gain, strata, rng)
                    margin_perm = BASE.shuffle_within_bins(edge_margin, strata, rng)
                    conf_perm = BASE.shuffle_within_bins(edge_confidence, strata, rng)
                    int_block_partner = np.column_stack(
                        [
                            gain_perm[:, None] * interaction_base,
                            margin_perm[:, None] * interaction_base,
                            conf_perm[:, None] * interaction_base,
                        ]
                    )
                    x_partner = np.column_stack([x_backbone, int_block_partner])
                    auc_partner = cross_validated_auc(
                        x_partner,
                        labels,
                        random_state=40_105 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H103_L1_C,
                        n_splits=H103_CV_SPLITS,
                    )
                    null_partner[perm_idx] = (
                        float(auc_partner - auc_backbone)
                        if np.isfinite(auc_partner) and np.isfinite(auc_backbone)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H103",
                            "null_kind": "interaction_partner_shuffle_within_geodesic_bins",
                            "domain": domain,
                            "seed_tag": H103_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_partner[perm_idx]),
                        }
                    )

                    # Null 3: stratified label permutation.
                    labels_perm = BASE.stratified_shuffle(labels, strata, rng).astype(int)
                    auc_lp_backbone = cross_validated_auc(
                        x_backbone,
                        labels_perm,
                        random_state=40_106 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H103_L1_C,
                        n_splits=H103_CV_SPLITS,
                    )
                    auc_lp_inter = cross_validated_auc(
                        x_interaction,
                        labels_perm,
                        random_state=40_107 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H103_L1_C,
                        n_splits=H103_CV_SPLITS,
                    )
                    null_label[perm_idx] = (
                        float(auc_lp_inter - auc_lp_backbone)
                        if np.isfinite(auc_lp_backbone) and np.isfinite(auc_lp_inter)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H103",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H103_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_order, null_partner, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_order = BASE.empirical_upper_tail_p(delta_auc, null_order)
                p_partner = BASE.empirical_upper_tail_p(delta_auc, null_partner)
                p_label = BASE.empirical_upper_tail_p(delta_auc, null_label)
                p_best = np.nanmin(np.array([p_order, p_partner, p_label], dtype=float))

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": H103_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h93_backbone": float(auc_backbone),
                        "auc_interaction_rescue": float(auc_interaction),
                        "delta_auc_interaction_derivative_minus_h91_h93": float(delta_auc),
                        "interaction_signal_pos_minus_neg": float(interaction_signal),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95_delta_auc": float(delta_auc - q95),
                        "p_order_permutation_upper": float(p_order),
                        "p_partner_shuffle_upper": float(p_partner),
                        "p_label_shuffle_upper": float(p_label),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h103_interaction_derivative_rescue_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h103_interaction_derivative_rescue_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_interaction_derivative_minus_h91_h93": float(
                        group["delta_auc_interaction_derivative_minus_h91_h93"].mean()
                    ),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "mean_interaction_signal_pos_minus_neg": float(group["interaction_signal_pos_minus_neg"].mean()),
                    "fraction_delta_positive": float(
                        (group["delta_auc_interaction_derivative_minus_h91_h93"] > 0.0).mean()
                    ),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h103_interaction_derivative_rescue_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_interaction_derivative_minus_h91_h93"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_interaction_derivative_minus_h91_h93"] > 0.0).sum())
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


def run_h104_depth_motif_grammar(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H104_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H104_GENE_CAP))
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

            rng = np.random.default_rng(40_200 + domain_index * 1000 + split_index * 100)
            sample_idx = stratified_index_sample(labels_all, max_n=H104_EDGE_SAMPLE, rng=rng)
            if sample_idx.size < 120:
                continue

            source_local = source_local_all[sample_idx]
            target_local = target_local_all[sample_idx]
            labels = labels_all[sample_idx]

            base_by_layer: dict[int, np.ndarray] = {}
            gain_by_layer: dict[int, np.ndarray] = {}
            edge_length = None
            degree_sum = None

            for layer in H104_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=22,
                    random_state=40_201 + domain_index * 1000 + split_index * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H104_NEIGHBORS)
                geodesic_w, _ = confidence_weighted_geodesic(geodesic, support_dir)

                _, h70_base, _ = compute_h70_scores(
                    geodesic=geodesic,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H104_TRIANGLE_K,
                )
                _, h70_weighted, _ = compute_h70_scores(
                    geodesic=geodesic_w,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H104_TRIANGLE_K,
                )

                base_by_layer[layer] = h70_base
                gain_by_layer[layer] = h70_weighted - h70_base

                if layer == H104_LAYERS[-1]:
                    edge_length = geodesic_w[source_local, target_local]
                    degree_sum = edge_degree_sum(
                        points=points_pca,
                        n_neighbors=H104_NEIGHBORS,
                        source_local=source_local,
                        target_local=target_local,
                    )

            if len(base_by_layer) != len(H104_LAYERS):
                continue
            if edge_length is None or degree_sum is None:
                continue

            tokens = build_depth_tokens(
                base_by_layer=base_by_layer,
                gain_by_layer=gain_by_layer,
                layers=H104_LAYERS,
                n_bins=H104_TOKEN_BINS,
            )
            baseline_score = base_by_layer[H104_LAYERS[-1]]
            baseline_auc = BASE.safe_auc(labels, baseline_score)

            motif_auc, motif_scores = motif_cv_auc(
                tokens=tokens,
                labels=labels,
                random_state=40_202 + domain_index * 1000 + split_index * 100,
                n_splits=H104_CV_SPLITS,
            )
            delta_auc = float(motif_auc - baseline_auc) if np.isfinite(motif_auc) and np.isfinite(baseline_auc) else float("nan")
            motif_gap = float(np.mean(motif_scores[labels == 1]) - np.mean(motif_scores[labels == 0]))

            strata = build_edge_strata(edge_length=edge_length, degree_sum=degree_sum, max_len_bins=6, max_deg_bins=4)

            null_order = np.empty(H104_NULL_PERM, dtype=float)
            null_token = np.empty(H104_NULL_PERM, dtype=float)
            null_label = np.empty(H104_NULL_PERM, dtype=float)

            for perm_idx in range(H104_NULL_PERM):
                # Null 1: layer-order permutation (marginal token counts preserved).
                order = rng.permutation(tokens.shape[1])
                tokens_order = tokens[:, order]
                auc_order, _ = motif_cv_auc(
                    tokens=tokens_order,
                    labels=labels,
                    random_state=40_203 + domain_index * 10_000 + split_index * 1000 + perm_idx,
                    n_splits=H104_CV_SPLITS,
                )
                null_order[perm_idx] = (
                    float(auc_order - baseline_auc) if np.isfinite(auc_order) and np.isfinite(baseline_auc) else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H104",
                        "null_kind": "layer_order_permutation",
                        "domain": domain,
                        "seed_tag": H104_SEED,
                        "split_regime": split_regime,
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_order[perm_idx]),
                    }
                )

                # Null 2: token shuffle within each layer.
                tokens_shuffle = tokens.copy()
                for col in range(tokens_shuffle.shape[1]):
                    tokens_shuffle[:, col] = rng.permutation(tokens_shuffle[:, col])
                auc_token, _ = motif_cv_auc(
                    tokens=tokens_shuffle,
                    labels=labels,
                    random_state=40_204 + domain_index * 10_000 + split_index * 1000 + perm_idx,
                    n_splits=H104_CV_SPLITS,
                )
                null_token[perm_idx] = (
                    float(auc_token - baseline_auc) if np.isfinite(auc_token) and np.isfinite(baseline_auc) else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H104",
                        "null_kind": "token_shuffle_within_layer",
                        "domain": domain,
                        "seed_tag": H104_SEED,
                        "split_regime": split_regime,
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_token[perm_idx]),
                    }
                )

                # Null 3: stratified label permutation.
                labels_perm = BASE.stratified_shuffle(labels, strata, rng).astype(int)
                auc_lp_motif, _ = motif_cv_auc(
                    tokens=tokens,
                    labels=labels_perm,
                    random_state=40_205 + domain_index * 10_000 + split_index * 1000 + perm_idx,
                    n_splits=H104_CV_SPLITS,
                )
                auc_lp_base = BASE.safe_auc(labels_perm, baseline_score)
                null_label[perm_idx] = (
                    float(auc_lp_motif - auc_lp_base)
                    if np.isfinite(auc_lp_motif) and np.isfinite(auc_lp_base)
                    else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H104",
                        "null_kind": "label_permutation",
                        "domain": domain,
                        "seed_tag": H104_SEED,
                        "split_regime": split_regime,
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_label[perm_idx]),
                    }
                )

            all_null = np.concatenate([null_order, null_token, null_label])
            q95 = float(np.nanquantile(all_null, 0.95))
            p_order = BASE.empirical_upper_tail_p(delta_auc, null_order)
            p_token = BASE.empirical_upper_tail_p(delta_auc, null_token)
            p_label = BASE.empirical_upper_tail_p(delta_auc, null_label)
            p_best = np.nanmin(np.array([p_order, p_token, p_label], dtype=float))

            rows.append(
                {
                    "domain": domain,
                    "seed_tag": H104_SEED,
                    "split_regime": split_regime,
                    "layer": -1,
                    "n_edges_eval": int(labels.size),
                    "n_positive_eval": int(labels.sum()),
                    "auc_h70_baseline": float(baseline_auc),
                    "auc_depth_motif_grammar": float(motif_auc),
                    "delta_auc_motif_grammar_minus_h70": float(delta_auc),
                    "motif_loglik_pos_minus_neg": float(motif_gap),
                    "q95_null_delta_auc": float(q95),
                    "null_gap_q95_delta_auc": float(delta_auc - q95),
                    "p_layer_order_upper": float(p_order),
                    "p_token_shuffle_upper": float(p_token),
                    "p_label_shuffle_upper": float(p_label),
                    "p_best_upper": float(p_best),
                }
            )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime"])
    by_row_path = ITER_DIR / "h104_depth_motif_grammar_by_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h104_depth_motif_grammar_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_motif_grammar_minus_h70": float(group["delta_auc_motif_grammar_minus_h70"].mean()),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "mean_motif_loglik_pos_minus_neg": float(group["motif_loglik_pos_minus_neg"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_motif_grammar_minus_h70"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h104_depth_motif_grammar_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_motif_grammar_minus_h70"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_motif_grammar_minus_h70"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95_delta_auc"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_domain_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h105_string_conditioned_null_calibration(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H105_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H105_GENE_CAP))
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

            for layer in H105_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                rng = np.random.default_rng(40_300 + domain_index * 1000 + split_index * 100 + layer)
                sample_idx = stratified_index_sample(labels_all, max_n=H105_EDGE_SAMPLE, rng=rng)
                if sample_idx.size < 120:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=22,
                    random_state=40_301 + domain_index * 1000 + split_index * 100 + layer,
                )

                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H105_NEIGHBORS)
                geodesic_w, support_sym = confidence_weighted_geodesic(geodesic, support_dir)

                _, h70_base, _ = compute_h70_scores(
                    geodesic=geodesic,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H105_TRIANGLE_K,
                )
                _, h70_weighted, _ = compute_h70_scores(
                    geodesic=geodesic_w,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H105_TRIANGLE_K,
                )

                weighted_gain = h70_weighted - h70_base
                edge_margin = support_dir[source_local, target_local] - support_dir[target_local, source_local]
                edge_confidence = support_sym[source_local, target_local]

                string_scores = np.asarray(
                    [
                        float(string_map.get((symbols[int(s)], symbols[int(t)]), 0.0))
                        for s, t in zip(source_local, target_local)
                    ],
                    dtype=float,
                )

                x_base = h70_base[:, None]
                x_h93 = np.column_stack([h70_base, h70_weighted, weighted_gain, edge_margin, edge_confidence, string_scores])

                auc_base = cross_validated_auc(
                    x_base,
                    labels,
                    random_state=40_302 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="none",
                    n_splits=H105_CV_SPLITS,
                )
                auc_h93 = cross_validated_auc(
                    x_h93,
                    labels,
                    random_state=40_303 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H105_L1_C,
                    n_splits=H105_CV_SPLITS,
                )
                delta_auc = float(auc_h93 - auc_base) if np.isfinite(auc_h93) and np.isfinite(auc_base) else float("nan")

                edge_length = geodesic_w[source_local, target_local]
                degree_sum = edge_degree_sum(
                    points=points_pca,
                    n_neighbors=H105_NEIGHBORS,
                    source_local=source_local,
                    target_local=target_local,
                )
                string_bins = BASE.degree_bins(string_scores, max_bins=10)
                degree_bins = BASE.degree_bins(degree_sum, max_bins=5)
                cond_strata = (string_bins * 16 + degree_bins).astype(int)

                null_uncond = np.empty(H105_NULL_PERM, dtype=float)
                null_cond = np.empty(H105_NULL_PERM, dtype=float)

                for perm_idx in range(H105_NULL_PERM):
                    # Null A: unconditioned label permutation.
                    labels_uncond = rng.permutation(labels)
                    auc_u_base = cross_validated_auc(
                        x_base,
                        labels_uncond,
                        random_state=40_304 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="none",
                        n_splits=H105_CV_SPLITS,
                    )
                    auc_u_h93 = cross_validated_auc(
                        x_h93,
                        labels_uncond,
                        random_state=40_305 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H105_L1_C,
                        n_splits=H105_CV_SPLITS,
                    )
                    null_uncond[perm_idx] = (
                        float(auc_u_h93 - auc_u_base)
                        if np.isfinite(auc_u_h93) and np.isfinite(auc_u_base)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H105",
                            "null_kind": "unconditioned_label_permutation",
                            "domain": domain,
                            "seed_tag": H105_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_uncond[perm_idx]),
                        }
                    )

                    # Null B: STRING-decile x degree-bin conditioned label permutation.
                    labels_cond = BASE.stratified_shuffle(labels, cond_strata, rng).astype(int)
                    auc_c_base = cross_validated_auc(
                        x_base,
                        labels_cond,
                        random_state=40_306 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="none",
                        n_splits=H105_CV_SPLITS,
                    )
                    auc_c_h93 = cross_validated_auc(
                        x_h93,
                        labels_cond,
                        random_state=40_307 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H105_L1_C,
                        n_splits=H105_CV_SPLITS,
                    )
                    null_cond[perm_idx] = (
                        float(auc_c_h93 - auc_c_base)
                        if np.isfinite(auc_c_h93) and np.isfinite(auc_c_base)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H105",
                            "null_kind": "string_degree_conditioned_label_permutation",
                            "domain": domain,
                            "seed_tag": H105_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_cond[perm_idx]),
                        }
                    )

                q95_uncond = float(np.nanquantile(null_uncond, 0.95))
                q95_cond = float(np.nanquantile(null_cond, 0.95))
                null_gap_uncond = float(delta_auc - q95_uncond)
                null_gap_cond = float(delta_auc - q95_cond)
                cond_gain = float(null_gap_cond - null_gap_uncond)

                p_uncond = BASE.empirical_upper_tail_p(delta_auc, null_uncond)
                p_cond = BASE.empirical_upper_tail_p(delta_auc, null_cond)

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": H105_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70_baseline": float(auc_base),
                        "auc_h93_with_string": float(auc_h93),
                        "delta_auc_h93_with_string_minus_h70": float(delta_auc),
                        "q95_null_unconditioned": float(q95_uncond),
                        "q95_null_conditioned": float(q95_cond),
                        "null_gap_q95_unconditioned": float(null_gap_uncond),
                        "null_gap_q95_conditioned": float(null_gap_cond),
                        "conditioned_minus_unconditioned_null_gap": float(cond_gain),
                        "p_unconditioned_upper": float(p_uncond),
                        "p_conditioned_upper": float(p_cond),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h105_string_conditioned_null_calibration_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h105_string_conditioned_null_calibration_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_h93_with_string_minus_h70": float(group["delta_auc_h93_with_string_minus_h70"].mean()),
                    "mean_null_gap_q95_unconditioned": float(group["null_gap_q95_unconditioned"].mean()),
                    "mean_null_gap_q95_conditioned": float(group["null_gap_q95_conditioned"].mean()),
                    "mean_conditioned_minus_unconditioned_null_gap": float(
                        group["conditioned_minus_unconditioned_null_gap"].mean()
                    ),
                    "fraction_conditional_gain_positive": float(
                        (group["conditioned_minus_unconditioned_null_gap"] > 0.0).mean()
                    ),
                    "fraction_p_conditioned_lt_0_05": float((group["p_conditioned_upper"] < 0.05).mean()),
                    "combined_fisher_p_conditioned": float(
                        BASE.safe_fisher_p(group["p_conditioned_upper"].to_numpy(dtype=float))
                    ),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h105_string_conditioned_null_calibration_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_h93_with_string_minus_h70"].mean())
        if not by_row_df.empty
        else float("nan"),
        "mean_conditioned_null_gap_gain": float(by_row_df["conditioned_minus_unconditioned_null_gap"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits_conditioned_gain": int(
            (summary_df["mean_conditioned_minus_unconditioned_null_gap"] > 0.0).sum()
        )
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

    h103_summary = run_h103_interaction_derivative_rescue(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h104_summary = run_h104_depth_motif_grammar(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h105_summary = run_h105_string_conditioned_null_calibration(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    summary = {
        "iteration": "iter_0040",
        "h103": h103_summary,
        "h104": h104_summary,
        "h105": h105_summary,
    }
    summary_path = ITER_DIR / "iter0040_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
