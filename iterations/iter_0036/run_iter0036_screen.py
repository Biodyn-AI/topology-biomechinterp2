from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ITER_DIR = Path("iterations/iter_0036")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base")

# H91 / N449: stability-selected sparse descriptor consensus (single carry-over refinement).
H91_SEEDS = ["seed42_main", "seed43", "seed44"]
H91_LAYERS = [0, 3, 7, 11]
H91_GENE_CAP = 170
H91_NEIGHBORS = 12
H91_TRIANGLE_K = [8, 12, 16]
H91_EDGE_SAMPLE = 240
H91_CV_SPLITS = 4
H91_L1_C = 0.20
H91_NULL_PERM = 4
H91_BOOTSTRAPS = 12
H91_BOOTSTRAP_FRAC = 0.75
H91_STABILITY_FREQ = 0.60
H91_STABILITY_SIGN = 0.70

# H92 / N452: scale-space lifetime trajectory descriptors (new method).
H92_SEED = "seed42_main"
H92_LAYERS = [0, 3, 7, 11]
H92_GENE_CAP = 170
H92_K_VALUES = [8, 10, 12, 14, 16]
H92_REF_K = 12
H92_TRIANGLE_K = [8, 12, 16]
H92_EDGE_SAMPLE = 300
H92_CV_SPLITS = 4
H92_L1_C = 0.25
H92_NULL_PERM = 6

# H93 / N458: confidence/sign-weighted filtration rescue (new biologically anchored method).
H93_SEED = "seed42_main"
H93_LAYERS = [7, 11]
H93_GENE_CAP = 170
H93_NEIGHBORS = 12
H93_TRIANGLE_K = [8, 12, 16]
H93_EDGE_SAMPLE = 300
H93_CV_SPLITS = 4
H93_L1_C = 0.25
H93_NULL_PERM = 6


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


def bootstrap_stable_descriptor_mask(
    h70_defect: np.ndarray,
    descriptors: np.ndarray,
    labels: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=int)
    x = np.column_stack([h70_defect, descriptors])
    n = y.size
    d = descriptors.shape[1]

    count_nonzero = np.zeros(d, dtype=float)
    sign_sum = np.zeros(d, dtype=float)

    rng = np.random.default_rng(random_state)
    boot_n = max(100, min(n, int(round(H91_BOOTSTRAP_FRAC * n))))

    for boot_idx in range(H91_BOOTSTRAPS):
        sub_idx = stratified_index_sample(y, max_n=boot_n, rng=rng)
        coef = fit_l1_coefficients(
            x[sub_idx],
            y[sub_idx],
            c_value=H91_L1_C,
            random_state=random_state + boot_idx + 1,
        )
        desc_coef = coef[1:]
        nz = np.abs(desc_coef) > 1e-8
        count_nonzero[nz] += 1.0
        sign_sum[nz] += np.sign(desc_coef[nz])

    freq = count_nonzero / float(max(1, H91_BOOTSTRAPS))
    sign_agree = np.zeros(d, dtype=float)
    mask = count_nonzero > 0
    sign_agree[mask] = np.abs(sign_sum[mask]) / count_nonzero[mask]

    stable_mask = (freq >= H91_STABILITY_FREQ) & (sign_agree >= H91_STABILITY_SIGN)
    if not np.any(stable_mask):
        order = np.lexsort((-sign_agree, -freq))
        top_k = min(2, d)
        chosen = order[:top_k]
        stable_mask = np.zeros(d, dtype=bool)
        stable_mask[chosen] = True

    return stable_mask, freq, sign_agree


def trajectory_shape_features(trajectory: np.ndarray, scales: np.ndarray) -> np.ndarray:
    traj = np.asarray(trajectory, dtype=float)
    scale_arr = np.asarray(scales, dtype=float)

    if traj.ndim != 2 or traj.shape[1] != scale_arr.size:
        raise ValueError("trajectory and scales shape mismatch")

    t = scale_arr.copy()
    t = (t - np.mean(t)) / max(np.std(t), 1e-8)

    design = np.column_stack([np.ones(t.size), t, t**2])
    pinv = np.linalg.pinv(design)
    coefs = traj @ pinv.T

    slope = coefs[:, 1]
    curvature = coefs[:, 2]
    area = np.trapezoid(traj, t, axis=1)
    amplitude = np.max(traj, axis=1) - np.min(traj, axis=1)
    end_minus_start = traj[:, -1] - traj[:, 0]

    return np.column_stack([slope, curvature, area, amplitude, end_minus_start])


def confidence_weighted_geodesic(geodesic: np.ndarray, support_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sym_support = 0.5 * (support_dir + support_dir.T)
    sym_support = np.clip(sym_support, 0.0, 1.0)

    weighted = geodesic / (0.35 + sym_support)
    weighted = np.asarray(weighted, dtype=float)
    np.fill_diagonal(weighted, 0.0)
    return weighted, sym_support


def random_sign_flip_support(support_dir: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    s = np.asarray(support_dir, dtype=float)
    sym = 0.5 * (s + s.T)
    asym = 0.5 * (s - s.T)

    n = s.shape[0]
    upper_i, upper_j = np.triu_indices(n, k=1)
    flip = rng.choice(np.array([-1.0, 1.0], dtype=float), size=upper_i.size, replace=True)

    asym_new = np.zeros_like(asym)
    asym_new[upper_i, upper_j] = asym[upper_i, upper_j] * flip
    asym_new[upper_j, upper_i] = -asym_new[upper_i, upper_j]

    out = sym + asym_new
    np.fill_diagonal(out, 0.0)
    return out


def run_h91_stability_selected_sparse_descriptor_consensus(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    coef_registry: dict[tuple[str, str, int], list[tuple[str, np.ndarray]]] = {}

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_index, seed_tag in enumerate(H91_SEEDS):
            run_dir = run_map[seed_tag]
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            split_masks = BASE.build_split_masks(edge_df)

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H91_GENE_CAP))
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

                for layer in H91_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    rng = np.random.default_rng(
                        36_910 + domain_index * 10_000 + seed_index * 1000 + split_index * 100 + layer
                    )
                    sample_idx = stratified_index_sample(labels_all, max_n=H91_EDGE_SAMPLE, rng=rng)
                    if sample_idx.size < 120:
                        continue

                    source_local = source_local_all[sample_idx]
                    target_local = target_local_all[sample_idx]
                    labels = labels_all[sample_idx]

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = BASE.reduce_points(
                        points,
                        n_components=20,
                        random_state=36_911 + domain_index * 10_000 + seed_index * 1000 + split_index * 100 + layer,
                    )
                    geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H91_NEIGHBORS)

                    _, h70_defect, tri_bundle = compute_h70_scores(
                        geodesic=geodesic,
                        support_dir=support_dir,
                        source_local=source_local,
                        target_local=target_local,
                        triangle_k=H91_TRIANGLE_K,
                    )
                    descriptors = build_sparse_descriptors(
                        geodesic=geodesic,
                        support_dir=support_dir,
                        source_local=source_local,
                        target_local=target_local,
                        points_pca=points_pca,
                        tri_bundle=tri_bundle,
                        n_neighbors=H91_NEIGHBORS,
                    )

                    stable_mask, freq, sign_agree = bootstrap_stable_descriptor_mask(
                        h70_defect=h70_defect,
                        descriptors=descriptors,
                        labels=labels,
                        random_state=36_912 + domain_index * 10_000 + seed_index * 1000 + split_index * 100 + layer,
                    )
                    selected_idx = np.where(stable_mask)[0]
                    selected_desc = descriptors[:, selected_idx]

                    x_base = h70_defect[:, None]
                    x_stable = np.column_stack([h70_defect, selected_desc])

                    auc_base = cross_validated_auc(
                        x_base,
                        labels,
                        random_state=36_913 + domain_index * 10_000 + seed_index * 1000 + split_index * 100 + layer,
                        penalty="none",
                        n_splits=H91_CV_SPLITS,
                    )
                    auc_stable = cross_validated_auc(
                        x_stable,
                        labels,
                        random_state=36_914 + domain_index * 10_000 + seed_index * 1000 + split_index * 100 + layer,
                        penalty="l1",
                        c_value=H91_L1_C,
                        n_splits=H91_CV_SPLITS,
                    )
                    delta_auc = float(auc_stable - auc_base) if np.isfinite(auc_stable) and np.isfinite(auc_base) else float("nan")

                    coef_sel = fit_l1_coefficients(
                        x_stable,
                        labels,
                        c_value=H91_L1_C,
                        random_state=36_915 + domain_index * 10_000 + seed_index * 1000 + split_index * 100 + layer,
                    )
                    full_coef = np.zeros(descriptors.shape[1], dtype=float)
                    if selected_idx.size > 0:
                        full_coef[selected_idx] = coef_sel[1:]
                    coef_registry.setdefault((domain, split_regime, int(layer)), []).append((seed_tag, full_coef))

                    edge_geodesic = geodesic[source_local, target_local]
                    bins = BASE.degree_bins(edge_geodesic, max_bins=6)

                    null_random_subset = np.empty(H91_NULL_PERM, dtype=float)
                    null_feature = np.empty(H91_NULL_PERM, dtype=float)
                    null_endpoint = np.empty(H91_NULL_PERM, dtype=float)
                    null_label = np.empty(H91_NULL_PERM, dtype=float)

                    subset_size = max(1, int(selected_idx.size))
                    for perm_idx in range(H91_NULL_PERM):
                        rand_idx = np.sort(rng.choice(descriptors.shape[1], size=subset_size, replace=False))
                        x_rand = np.column_stack([h70_defect, descriptors[:, rand_idx]])
                        auc_rand = cross_validated_auc(
                            x_rand,
                            labels,
                            random_state=36_916
                            + domain_index * 100_000
                            + seed_index * 10_000
                            + split_index * 1000
                            + layer * 10
                            + perm_idx,
                            penalty="l1",
                            c_value=H91_L1_C,
                            n_splits=H91_CV_SPLITS,
                        )
                        null_random_subset[perm_idx] = (
                            float(auc_rand - auc_base) if np.isfinite(auc_rand) and np.isfinite(auc_base) else float("nan")
                        )
                        null_rows.append(
                            {
                                "hypothesis_id": "H91",
                                "null_kind": "random_feature_subset_control",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_value": float(null_random_subset[perm_idx]),
                            }
                        )

                        shuffled_desc = np.column_stack(
                            [BASE.shuffle_within_bins(selected_desc[:, j], bins, rng) for j in range(selected_desc.shape[1])]
                        )
                        auc_feat = cross_validated_auc(
                            np.column_stack([h70_defect, shuffled_desc]),
                            labels,
                            random_state=37_016
                            + domain_index * 100_000
                            + seed_index * 10_000
                            + split_index * 1000
                            + layer * 10
                            + perm_idx,
                            penalty="l1",
                            c_value=H91_L1_C,
                            n_splits=H91_CV_SPLITS,
                        )
                        null_feature[perm_idx] = (
                            float(auc_feat - auc_base) if np.isfinite(auc_feat) and np.isfinite(auc_base) else float("nan")
                        )
                        null_rows.append(
                            {
                                "hypothesis_id": "H91",
                                "null_kind": "stability_descriptor_shuffle_within_geodesic_bins",
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
                            k_values=H91_TRIANGLE_K,
                        )
                        desc_swap = build_sparse_descriptors(
                            geodesic=geodesic,
                            support_dir=support_dir,
                            source_local=source_local,
                            target_local=target_swap,
                            points_pca=points_pca,
                            tri_bundle=tri_swap,
                            n_neighbors=H91_NEIGHBORS,
                        )
                        auc_swap = cross_validated_auc(
                            np.column_stack([h70_defect, desc_swap[:, selected_idx]]),
                            labels,
                            random_state=37_116
                            + domain_index * 100_000
                            + seed_index * 10_000
                            + split_index * 1000
                            + layer * 10
                            + perm_idx,
                            penalty="l1",
                            c_value=H91_L1_C,
                            n_splits=H91_CV_SPLITS,
                        )
                        null_endpoint[perm_idx] = (
                            float(auc_swap - auc_base) if np.isfinite(auc_swap) and np.isfinite(auc_base) else float("nan")
                        )
                        null_rows.append(
                            {
                                "hypothesis_id": "H91",
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
                            random_state=37_216
                            + domain_index * 100_000
                            + seed_index * 10_000
                            + split_index * 1000
                            + layer * 10
                            + perm_idx,
                            penalty="none",
                            n_splits=H91_CV_SPLITS,
                        )
                        auc_lp_stable = cross_validated_auc(
                            x_stable,
                            labels_perm,
                            random_state=37_316
                            + domain_index * 100_000
                            + seed_index * 10_000
                            + split_index * 1000
                            + layer * 10
                            + perm_idx,
                            penalty="l1",
                            c_value=H91_L1_C,
                            n_splits=H91_CV_SPLITS,
                        )
                        null_label[perm_idx] = (
                            float(auc_lp_stable - auc_lp_base)
                            if np.isfinite(auc_lp_stable) and np.isfinite(auc_lp_base)
                            else float("nan")
                        )
                        null_rows.append(
                            {
                                "hypothesis_id": "H91",
                                "null_kind": "label_permutation",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_value": float(null_label[perm_idx]),
                            }
                        )

                    all_null = np.concatenate([null_random_subset, null_feature, null_endpoint, null_label])
                    q95 = float(np.nanquantile(all_null, 0.95))
                    p_random = BASE.empirical_upper_tail_p(delta_auc, null_random_subset)
                    p_feat = BASE.empirical_upper_tail_p(delta_auc, null_feature)
                    p_end = BASE.empirical_upper_tail_p(delta_auc, null_endpoint)
                    p_lab = BASE.empirical_upper_tail_p(delta_auc, null_label)
                    p_best = np.nanmin(np.array([p_random, p_feat, p_end, p_lab], dtype=float))

                    rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "n_edges_eval": int(labels.size),
                            "n_positive_eval": int(labels.sum()),
                            "selected_feature_count": int(selected_idx.size),
                            "mean_selected_frequency": float(np.mean(freq[selected_idx])) if selected_idx.size > 0 else float("nan"),
                            "mean_selected_sign_agreement": float(np.mean(sign_agree[selected_idx]))
                            if selected_idx.size > 0
                            else float("nan"),
                            "auc_h70_baseline": float(auc_base),
                            "auc_stability_selected_blend": float(auc_stable),
                            "delta_auc_stability_selected_blend_minus_h70": float(delta_auc),
                            "delta_auc_random_subset_minus_h70_mean": float(np.nanmean(null_random_subset)),
                            "q95_null_delta_auc": float(q95),
                            "null_gap_q95_delta_auc": float(delta_auc - q95),
                            "p_random_subset_upper": float(p_random),
                            "p_feature_shuffle_upper": float(p_feat),
                            "p_endpoint_swap_upper": float(p_end),
                            "p_label_shuffle_upper": float(p_lab),
                            "p_best_upper": float(p_best),
                        }
                    )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h91_stability_selected_sparse_descriptor_by_seed_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h91_stability_selected_sparse_descriptor_null_summary.csv"
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
        sign_match = pairwise_sign_agreement(signs, nonzero_sets)
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
                "descriptor_sign_agreement_mean": float(sign_match),
                "descriptor_core_size": int(len(core_set)),
                "mean_delta_auc_stability_selected_blend_minus_h70": float(
                    grp["delta_auc_stability_selected_blend_minus_h70"].mean()
                )
                if not grp.empty
                else float("nan"),
                "mean_null_gap_q95_delta_auc": float(grp["null_gap_q95_delta_auc"].mean()) if not grp.empty else float("nan"),
            }
        )

    stability_df = pd.DataFrame(stability_rows)
    if not stability_df.empty:
        stability_df = stability_df.sort_values(["domain", "split_regime", "layer"])
    stability_path = ITER_DIR / "h91_stability_selected_sparse_descriptor_stability.csv"
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
                    "mean_delta_auc_stability_selected_blend_minus_h70": float(
                        group["delta_auc_stability_selected_blend_minus_h70"].mean()
                    ),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_stability_selected_blend_minus_h70"] > 0.0).mean()),
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
    summary_path = ITER_DIR / "h91_stability_selected_sparse_descriptor_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_stability_selected_blend_minus_h70"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_stability_selected_blend_minus_h70"] > 0.0).sum())
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


def run_h92_scale_space_lifetime_trajectory(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H92_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H92_GENE_CAP))
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

            for layer in H92_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                rng = np.random.default_rng(36_920 + domain_index * 1000 + split_index * 100 + layer)
                sample_idx = stratified_index_sample(labels_all, max_n=H92_EDGE_SAMPLE, rng=rng)
                if sample_idx.size < 120:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=22,
                    random_state=36_921 + domain_index * 1000 + split_index * 100 + layer,
                )

                geodesic_by_k: dict[int, np.ndarray] = {}
                for k in H92_K_VALUES:
                    geodesic_by_k[int(k)] = BASE.geodesic_distance_matrix(points_pca, n_neighbors=int(k))

                geodesic_ref = geodesic_by_k[H92_REF_K]
                _, h70_ref, _ = compute_h70_scores(
                    geodesic=geodesic_ref,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H92_TRIANGLE_K,
                )

                trajectory_cols = []
                for k in H92_K_VALUES:
                    tri = BASE.multiscale_triangle_defect_features(
                        geodesic=geodesic_by_k[int(k)],
                        source_local=source_local,
                        target_local=target_local,
                        k_values=H92_TRIANGLE_K,
                    )
                    trajectory_cols.append(-np.asarray(tri["median_mean"], dtype=float))
                trajectory = np.column_stack(trajectory_cols)
                traj_features = trajectory_shape_features(trajectory, scales=np.asarray(H92_K_VALUES, dtype=float))

                x_base = h70_ref[:, None]
                x_aug = np.column_stack([h70_ref, traj_features])

                auc_base = cross_validated_auc(
                    x_base,
                    labels,
                    random_state=36_922 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="none",
                    n_splits=H92_CV_SPLITS,
                )
                auc_aug = cross_validated_auc(
                    x_aug,
                    labels,
                    random_state=36_923 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H92_L1_C,
                    n_splits=H92_CV_SPLITS,
                )
                delta_auc = float(auc_aug - auc_base) if np.isfinite(auc_aug) and np.isfinite(auc_base) else float("nan")

                slope = traj_features[:, 0]
                slope_gap = float(np.mean(slope[labels == 1]) - np.mean(slope[labels == 0]))

                edge_geodesic = geodesic_ref[source_local, target_local]
                bins = BASE.degree_bins(edge_geodesic, max_bins=6)

                null_scale = np.empty(H92_NULL_PERM, dtype=float)
                null_feature = np.empty(H92_NULL_PERM, dtype=float)
                null_label = np.empty(H92_NULL_PERM, dtype=float)

                for perm_idx in range(H92_NULL_PERM):
                    perm_order = rng.permutation(len(H92_K_VALUES))
                    traj_perm = trajectory[:, perm_order]
                    feat_scale = trajectory_shape_features(traj_perm, scales=np.asarray(H92_K_VALUES, dtype=float))
                    auc_scale = cross_validated_auc(
                        np.column_stack([h70_ref, feat_scale]),
                        labels,
                        random_state=36_924 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H92_L1_C,
                        n_splits=H92_CV_SPLITS,
                    )
                    null_scale[perm_idx] = (
                        float(auc_scale - auc_base) if np.isfinite(auc_scale) and np.isfinite(auc_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H92",
                            "null_kind": "scale_order_permutation",
                            "domain": domain,
                            "seed_tag": H92_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_scale[perm_idx]),
                        }
                    )

                    feat_shuf = np.column_stack(
                        [BASE.shuffle_within_bins(traj_features[:, j], bins, rng) for j in range(traj_features.shape[1])]
                    )
                    auc_feat = cross_validated_auc(
                        np.column_stack([h70_ref, feat_shuf]),
                        labels,
                        random_state=36_925 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H92_L1_C,
                        n_splits=H92_CV_SPLITS,
                    )
                    null_feature[perm_idx] = (
                        float(auc_feat - auc_base) if np.isfinite(auc_feat) and np.isfinite(auc_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H92",
                            "null_kind": "trajectory_feature_shuffle_within_geodesic_bins",
                            "domain": domain,
                            "seed_tag": H92_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_feature[perm_idx]),
                        }
                    )

                    labels_perm = BASE.stratified_shuffle(labels, bins, rng).astype(int)
                    auc_lp_base = cross_validated_auc(
                        x_base,
                        labels_perm,
                        random_state=36_926 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="none",
                        n_splits=H92_CV_SPLITS,
                    )
                    auc_lp_aug = cross_validated_auc(
                        x_aug,
                        labels_perm,
                        random_state=36_927 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H92_L1_C,
                        n_splits=H92_CV_SPLITS,
                    )
                    null_label[perm_idx] = (
                        float(auc_lp_aug - auc_lp_base) if np.isfinite(auc_lp_aug) and np.isfinite(auc_lp_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H92",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H92_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_scale, null_feature, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_scale = BASE.empirical_upper_tail_p(delta_auc, null_scale)
                p_feat = BASE.empirical_upper_tail_p(delta_auc, null_feature)
                p_lab = BASE.empirical_upper_tail_p(delta_auc, null_label)
                p_best = np.nanmin(np.array([p_scale, p_feat, p_lab], dtype=float))

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": H92_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70_baseline": float(auc_base),
                        "auc_scale_trajectory_blend": float(auc_aug),
                        "delta_auc_scale_trajectory_minus_h70": float(delta_auc),
                        "slope_pos_minus_neg": float(slope_gap),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95_delta_auc": float(delta_auc - q95),
                        "p_scale_order_upper": float(p_scale),
                        "p_feature_shuffle_upper": float(p_feat),
                        "p_label_shuffle_upper": float(p_lab),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h92_scale_space_lifetime_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h92_scale_space_lifetime_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_scale_trajectory_minus_h70": float(group["delta_auc_scale_trajectory_minus_h70"].mean()),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "mean_slope_pos_minus_neg": float(group["slope_pos_minus_neg"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_scale_trajectory_minus_h70"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h92_scale_space_lifetime_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_scale_trajectory_minus_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_scale_trajectory_minus_h70"] > 0.0).sum())
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


def run_h93_confidence_sign_weighted_filtration(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H93_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H93_GENE_CAP))
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

            for layer in H93_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                rng = np.random.default_rng(36_930 + domain_index * 1000 + split_index * 100 + layer)
                sample_idx = stratified_index_sample(labels_all, max_n=H93_EDGE_SAMPLE, rng=rng)
                if sample_idx.size < 120:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=22,
                    random_state=36_931 + domain_index * 1000 + split_index * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H93_NEIGHBORS)

                _, h70_base, _ = compute_h70_scores(
                    geodesic=geodesic,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H93_TRIANGLE_K,
                )

                geodesic_weighted, support_sym = confidence_weighted_geodesic(geodesic, support_dir)
                _, h70_weighted, _ = compute_h70_scores(
                    geodesic=geodesic_weighted,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H93_TRIANGLE_K,
                )

                edge_confidence = support_sym[source_local, target_local]
                edge_signed_margin = support_dir[source_local, target_local] - support_dir[target_local, source_local]
                weighted_gain = h70_weighted - h70_base

                x_base = h70_base[:, None]
                x_aug = np.column_stack(
                    [
                        h70_base,
                        h70_weighted,
                        weighted_gain,
                        edge_signed_margin,
                        edge_confidence,
                    ]
                )

                auc_base = cross_validated_auc(
                    x_base,
                    labels,
                    random_state=36_932 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="none",
                    n_splits=H93_CV_SPLITS,
                )
                auc_aug = cross_validated_auc(
                    x_aug,
                    labels,
                    random_state=36_933 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H93_L1_C,
                    n_splits=H93_CV_SPLITS,
                )
                delta_auc = float(auc_aug - auc_base) if np.isfinite(auc_aug) and np.isfinite(auc_base) else float("nan")

                gain_pos_neg = float(np.mean(weighted_gain[labels == 1]) - np.mean(weighted_gain[labels == 0]))

                edge_geodesic = geodesic[source_local, target_local]
                bins = BASE.degree_bins(edge_geodesic, max_bins=6)

                null_conf = np.empty(H93_NULL_PERM, dtype=float)
                null_sign = np.empty(H93_NULL_PERM, dtype=float)
                null_label = np.empty(H93_NULL_PERM, dtype=float)

                for perm_idx in range(H93_NULL_PERM):
                    conf_shuffle = BASE.shuffle_within_bins(edge_confidence, bins, rng)
                    margin_shuffle = BASE.shuffle_within_bins(edge_signed_margin, bins, rng)
                    weighted_shuffle = BASE.shuffle_within_bins(h70_weighted, bins, rng)
                    x_conf = np.column_stack(
                        [
                            h70_base,
                            weighted_shuffle,
                            weighted_shuffle - h70_base,
                            margin_shuffle,
                            conf_shuffle,
                        ]
                    )
                    auc_conf = cross_validated_auc(
                        x_conf,
                        labels,
                        random_state=36_934 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H93_L1_C,
                        n_splits=H93_CV_SPLITS,
                    )
                    null_conf[perm_idx] = (
                        float(auc_conf - auc_base) if np.isfinite(auc_conf) and np.isfinite(auc_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H93",
                            "null_kind": "confidence_shuffle_within_geodesic_bins",
                            "domain": domain,
                            "seed_tag": H93_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_conf[perm_idx]),
                        }
                    )

                    support_flip = random_sign_flip_support(support_dir, rng)
                    _, h70_flip, _ = compute_h70_scores(
                        geodesic=geodesic_weighted,
                        support_dir=support_flip,
                        source_local=source_local,
                        target_local=target_local,
                        triangle_k=H93_TRIANGLE_K,
                    )
                    margin_flip = support_flip[source_local, target_local] - support_flip[target_local, source_local]
                    x_sign = np.column_stack(
                        [
                            h70_base,
                            h70_flip,
                            h70_flip - h70_base,
                            margin_flip,
                            edge_confidence,
                        ]
                    )
                    auc_sign = cross_validated_auc(
                        x_sign,
                        labels,
                        random_state=36_935 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H93_L1_C,
                        n_splits=H93_CV_SPLITS,
                    )
                    null_sign[perm_idx] = (
                        float(auc_sign - auc_base) if np.isfinite(auc_sign) and np.isfinite(auc_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H93",
                            "null_kind": "sign_flip_control",
                            "domain": domain,
                            "seed_tag": H93_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_sign[perm_idx]),
                        }
                    )

                    labels_perm = BASE.stratified_shuffle(labels, bins, rng).astype(int)
                    auc_lp_base = cross_validated_auc(
                        x_base,
                        labels_perm,
                        random_state=36_936 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="none",
                        n_splits=H93_CV_SPLITS,
                    )
                    auc_lp_aug = cross_validated_auc(
                        x_aug,
                        labels_perm,
                        random_state=36_937 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H93_L1_C,
                        n_splits=H93_CV_SPLITS,
                    )
                    null_label[perm_idx] = (
                        float(auc_lp_aug - auc_lp_base) if np.isfinite(auc_lp_aug) and np.isfinite(auc_lp_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H93",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H93_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_conf, null_sign, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_conf = BASE.empirical_upper_tail_p(delta_auc, null_conf)
                p_sign = BASE.empirical_upper_tail_p(delta_auc, null_sign)
                p_lab = BASE.empirical_upper_tail_p(delta_auc, null_label)
                p_best = np.nanmin(np.array([p_conf, p_sign, p_lab], dtype=float))

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": H93_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70_baseline": float(auc_base),
                        "auc_weighted_filtration_blend": float(auc_aug),
                        "delta_auc_weighted_filtration_minus_h70": float(delta_auc),
                        "weighted_gain_pos_minus_neg": float(gain_pos_neg),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95_delta_auc": float(delta_auc - q95),
                        "p_confidence_shuffle_upper": float(p_conf),
                        "p_sign_flip_upper": float(p_sign),
                        "p_label_shuffle_upper": float(p_lab),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h93_confidence_sign_weighted_filtration_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h93_confidence_sign_weighted_filtration_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_weighted_filtration_minus_h70": float(
                        group["delta_auc_weighted_filtration_minus_h70"].mean()
                    ),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "mean_weighted_gain_pos_minus_neg": float(group["weighted_gain_pos_minus_neg"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_weighted_filtration_minus_h70"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h93_confidence_sign_weighted_filtration_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_weighted_filtration_minus_h70"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_weighted_filtration_minus_h70"] > 0.0).sum())
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

    h91_summary = run_h91_stability_selected_sparse_descriptor_consensus(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h92_summary = run_h92_scale_space_lifetime_trajectory(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h93_summary = run_h93_confidence_sign_weighted_filtration(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    summary = {
        "iteration": "iter_0036",
        "h91": h91_summary,
        "h92": h92_summary,
        "h93": h93_summary,
    }
    summary_path = ITER_DIR / "iter0036_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
