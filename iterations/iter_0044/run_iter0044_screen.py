from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ITER_DIR = Path("iterations/iter_0044")
ITER_DIR.mkdir(parents=True, exist_ok=True)

H115_SEED = "seed42_main"
H115_LAYERS = [0, 3, 7, 11]
H115_GENE_CAP = 170
H115_NEIGHBORS = 12
H115_EDGE_SAMPLE = 260
H115_CV_SPLITS = 4
H115_NULL_PERM = 16
H115_SUBSPACE_DIM = 5

H116_SEED = "seed42_main"
H116_LAYER = 11
H116_GENE_CAP = 190
H116_NEIGHBORS = 12
H116_EDGE_SAMPLE = 300
H116_CV_SPLITS = 4
H116_NULL_PERM = 20


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base")


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


def cv_auc_logit(features: np.ndarray, labels: np.ndarray, random_state: int, n_splits: int, l1_c: float = 0.25) -> float:
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
        model = LogisticRegression(
            penalty="l1",
            C=float(l1_c),
            solver="liblinear",
            max_iter=1500,
            random_state=random_state + fold_idx,
        )
        model.fit(x_tr, y[tr])
        probs[te] = model.predict_proba(x_te)[:, 1]

    return BASE.safe_auc(y, probs)


def confidence_weighted_geodesic(geodesic: np.ndarray, support_dir: np.ndarray) -> np.ndarray:
    sym_support = 0.5 * (support_dir + support_dir.T)
    sym_support = np.clip(sym_support, 0.0, 1.0)
    weighted = geodesic / (0.35 + sym_support)
    weighted = np.asarray(weighted, dtype=float)
    np.fill_diagonal(weighted, 0.0)
    return weighted


def compute_h70_scores(
    geodesic: np.ndarray,
    support_dir: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    triangle_k: list[int],
) -> np.ndarray:
    edge_geodesic = geodesic[source_local, target_local]
    edge_support = support_dir[source_local, target_local]
    edge_margin = np.abs(support_dir[source_local, target_local] - support_dir[target_local, source_local])

    bundle = BASE.multiscale_triangle_defect_features(
        geodesic=geodesic,
        source_local=source_local,
        target_local=target_local,
        k_values=triangle_k,
    )

    baseline = BASE.zscore(-edge_geodesic) + 0.75 * BASE.zscore(edge_support) + 0.35 * BASE.zscore(edge_margin)
    return (
        baseline
        + 0.35 * BASE.zscore(-bundle["median_mean"])
        + 0.25 * BASE.zscore(-bundle["tail_mean"])
        + 0.20 * BASE.zscore(bundle["close_frac_mean"])
        + 0.10 * BASE.zscore(-bundle["scale_span"])
        + 0.10 * BASE.zscore(-bundle["dispersion_mean"])
    )


def build_tangent_bases(points: np.ndarray, n_neighbors: int, n_components: int) -> list[np.ndarray]:
    n_nodes, dim = points.shape
    k_eff = max(4, min(n_neighbors, n_nodes - 1))
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nbrs.fit(points)
    _, inds = nbrs.kneighbors(points)

    bases: list[np.ndarray] = []
    for i in range(n_nodes):
        local_idx = inds[i, 1:]
        local_pts = points[local_idx] - points[i]
        if local_pts.shape[0] < n_components:
            u = np.eye(dim, n_components)
        else:
            pca = PCA(n_components=n_components, svd_solver="full")
            pca.fit(local_pts)
            u = pca.components_.T
        bases.append(u)
    return bases


def mean_principal_angle(u: np.ndarray, v: np.ndarray) -> float:
    m = np.clip(np.linalg.svd(u.T @ v, compute_uv=False), 0.0, 1.0)
    return float(np.mean(np.arccos(m)))


def angle_profile_features(angle_mat: np.ndarray) -> np.ndarray:
    slope = angle_mat[:, -1] - angle_mat[:, 0]
    first_diff = np.diff(angle_mat, axis=1)
    if first_diff.shape[1] >= 2:
        accel = np.mean(np.abs(np.diff(first_diff, axis=1)), axis=1)
    else:
        accel = np.zeros(angle_mat.shape[0], dtype=float)
    jerk_proxy = np.var(first_diff, axis=1)
    return np.column_stack([slope, accel, jerk_proxy])


def domain_split_summary(rows: list[dict[str, object]], hypothesis_id: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    sub = df.loc[df["hypothesis_id"] == hypothesis_id].copy()
    if sub.empty:
        return pd.DataFrame()
    grp = sub.groupby(["domain", "split_regime"], as_index=False).agg(
        mean_delta=("delta_vs_h70", "mean"),
        mean_null_gap=("null_gap", "mean"),
        n=("delta_vs_h70", "count"),
    )
    grp["direction_pass"] = (grp["mean_delta"] > 0).astype(int)
    grp["null_pass"] = (grp["mean_null_gap"] > 0).astype(int)
    return grp


def run_h115_tangent_acceleration(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_idx, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H115_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_idx, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H115_GENE_CAP))
            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
            ].copy()
            if split_edges["label"].nunique() < 2:
                continue

            edge_gene_indices = np.unique(np.concatenate([
                split_edges["source_idx"].to_numpy(dtype=int),
                split_edges["target_idx"].to_numpy(dtype=int),
            ]))
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

            rng = np.random.default_rng(44_150 + domain_idx * 1000 + split_idx * 100)
            sample_idx = stratified_index_sample(labels_all, max_n=H115_EDGE_SAMPLE, rng=rng)
            if sample_idx.size < 120:
                continue

            source_local = source_local_all[sample_idx]
            target_local = target_local_all[sample_idx]
            labels = labels_all[sample_idx]

            angle_layers: list[np.ndarray] = []
            h70_deep = None
            for layer in H115_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    angle_layers = []
                    break
                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(points, n_components=20, random_state=44_151 + layer)

                bases = build_tangent_bases(points_pca, n_neighbors=H115_NEIGHBORS, n_components=H115_SUBSPACE_DIM)
                angles = np.asarray([
                    mean_principal_angle(bases[s], bases[t]) for s, t in zip(source_local, target_local)
                ], dtype=float)
                angle_layers.append(angles)

                if layer == H115_LAYERS[-1]:
                    geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H115_NEIGHBORS)
                    geodesic_w = confidence_weighted_geodesic(geodesic, support_dir)
                    h70_deep = compute_h70_scores(
                        geodesic=geodesic_w,
                        support_dir=support_dir,
                        source_local=source_local,
                        target_local=target_local,
                        triangle_k=[8, 12, 16],
                    )

            if len(angle_layers) != len(H115_LAYERS) or h70_deep is None:
                continue

            angle_mat = np.column_stack(angle_layers)
            feat = angle_profile_features(angle_mat)
            model_feat = np.column_stack([h70_deep, feat])

            auc_h70 = BASE.safe_auc(labels, h70_deep)
            auc_model = cv_auc_logit(model_feat, labels, random_state=44_152 + domain_idx * 1000 + split_idx * 100, n_splits=H115_CV_SPLITS)
            delta = float(auc_model - auc_h70) if np.isfinite(auc_model) and np.isfinite(auc_h70) else float("nan")

            null_layer = np.empty(H115_NULL_PERM, dtype=float)
            null_label = np.empty(H115_NULL_PERM, dtype=float)

            strata = BASE.degree_bins(np.asarray(h70_deep, dtype=float), max_bins=6)
            for perm_idx in range(H115_NULL_PERM):
                order = rng.permutation(angle_mat.shape[1])
                feat_perm = angle_profile_features(angle_mat[:, order])
                auc_perm = cv_auc_logit(
                    np.column_stack([h70_deep, feat_perm]),
                    labels,
                    random_state=44_153 + domain_idx * 10_000 + split_idx * 1000 + perm_idx,
                    n_splits=H115_CV_SPLITS,
                )
                null_layer[perm_idx] = float(auc_perm - auc_h70) if np.isfinite(auc_perm) and np.isfinite(auc_h70) else float("nan")
                null_rows.append({
                    "hypothesis_id": "H115",
                    "null_kind": "layer_order_permutation",
                    "domain": domain,
                    "seed_tag": H115_SEED,
                    "split_regime": split_regime,
                    "perm_idx": int(perm_idx),
                    "null_value": float(null_layer[perm_idx]),
                })

                y_perm = BASE.stratified_shuffle(labels, strata, rng).astype(int)
                auc_lp = cv_auc_logit(
                    model_feat,
                    y_perm,
                    random_state=44_154 + domain_idx * 10_000 + split_idx * 1000 + perm_idx,
                    n_splits=H115_CV_SPLITS,
                )
                auc_h70_lp = BASE.safe_auc(y_perm, h70_deep)
                null_label[perm_idx] = float(auc_lp - auc_h70_lp) if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp) else float("nan")
                null_rows.append({
                    "hypothesis_id": "H115",
                    "null_kind": "label_permutation",
                    "domain": domain,
                    "seed_tag": H115_SEED,
                    "split_regime": split_regime,
                    "perm_idx": int(perm_idx),
                    "null_value": float(null_label[perm_idx]),
                })

            null_pool = np.concatenate([null_layer[np.isfinite(null_layer)], null_label[np.isfinite(null_label)]])
            null_gap = float(delta - np.nanmean(null_pool)) if null_pool.size else float("nan")
            rows.append({
                "hypothesis_id": "H115",
                "domain": domain,
                "seed_tag": H115_SEED,
                "split_regime": split_regime,
                "auc_h70": float(auc_h70),
                "auc_model": float(auc_model),
                "delta_vs_h70": delta,
                "null_gap": null_gap,
                "n_edges": int(labels.size),
            })

    raw_df = pd.DataFrame(rows)
    null_df = pd.DataFrame(null_rows)
    raw_path = ITER_DIR / "h115_tangent_acceleration_by_domain_split.csv"
    null_path = ITER_DIR / "h115_tangent_acceleration_null_summary.csv"
    dom_path = ITER_DIR / "h115_tangent_acceleration_domain_summary.csv"
    raw_df.to_csv(raw_path, index=False)
    null_df.to_csv(null_path, index=False)
    dom_df = domain_split_summary(rows, "H115")
    dom_df.to_csv(dom_path, index=False)

    return {
        "raw": rows,
        "domain_summary": dom_df,
        "artifact_paths": [str(raw_path), str(null_path), str(dom_path)],
    }


def run_h116_trrust_sign_motif(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    dorothea_signed = {(s.upper(), t.upper()): int(v) for (s, t), v in dorothea_map.items() if int(v) != 0}

    for domain_idx, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H116_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_idx, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H116_GENE_CAP))
            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
            ].copy()
            if split_edges["label"].nunique() < 2:
                continue

            edge_gene_indices = np.unique(np.concatenate([
                split_edges["source_idx"].to_numpy(dtype=int),
                split_edges["target_idx"].to_numpy(dtype=int),
            ]))
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

            rng = np.random.default_rng(44_260 + domain_idx * 1000 + split_idx * 100)
            sample_idx = stratified_index_sample(labels_all, max_n=H116_EDGE_SAMPLE, rng=rng)
            if sample_idx.size < 120:
                continue

            source_local = source_local_all[sample_idx]
            target_local = target_local_all[sample_idx]
            labels = labels_all[sample_idx]

            points = layer_embeddings[H116_LAYER, edge_gene_indices, :]
            points_pca = BASE.reduce_points(points, n_components=20, random_state=44_261 + domain_idx * 1000 + split_idx * 100)
            geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H116_NEIGHBORS)
            geodesic_w = confidence_weighted_geodesic(geodesic, support_dir)

            h70 = compute_h70_scores(
                geodesic=geodesic_w,
                support_dir=support_dir,
                source_local=source_local,
                target_local=target_local,
                triangle_k=[8, 12, 16],
            )

            src_sym = [symbols[i].upper() for i in source_local]
            tgt_sym = [symbols[i].upper() for i in target_local]
            edge_margin = support_dir[source_local, target_local] - support_dir[target_local, source_local]
            margin_sign = np.sign(edge_margin)

            motif_sign = np.asarray([dorothea_signed.get((s, t), 0) for s, t in zip(src_sym, tgt_sym)], dtype=int)
            motif_present = (motif_sign != 0).astype(float)
            sign_consistent = ((motif_sign * margin_sign) > 0).astype(float)

            feat = np.column_stack([
                h70,
                motif_present,
                sign_consistent,
                h70 * motif_present,
                h70 * sign_consistent,
            ])

            auc_h70 = BASE.safe_auc(labels, h70)
            auc_model = cv_auc_logit(feat, labels, random_state=44_262 + domain_idx * 1000 + split_idx * 100, n_splits=H116_CV_SPLITS)
            delta = float(auc_model - auc_h70) if np.isfinite(auc_model) and np.isfinite(auc_h70) else float("nan")

            tf_degree = np.asarray([sum(1 for (s, _) in dorothea_signed.keys() if s == sym) for sym in src_sym], dtype=float)
            tf_bins = BASE.degree_bins(tf_degree, max_bins=4)
            strata = BASE.degree_bins(np.asarray(h70, dtype=float), max_bins=6)

            null_motif = np.empty(H116_NULL_PERM, dtype=float)
            null_label = np.empty(H116_NULL_PERM, dtype=float)
            for perm_idx in range(H116_NULL_PERM):
                motif_perm = motif_sign.copy()
                for b in np.unique(tf_bins):
                    idx = np.where(tf_bins == b)[0]
                    if idx.size > 1:
                        motif_perm[idx] = motif_perm[rng.permutation(idx)]

                sign_cons_perm = ((motif_perm * margin_sign) > 0).astype(float)
                feat_perm = np.column_stack([h70, (motif_perm != 0).astype(float), sign_cons_perm, h70 * (motif_perm != 0).astype(float), h70 * sign_cons_perm])
                auc_mp = cv_auc_logit(
                    feat_perm,
                    labels,
                    random_state=44_263 + domain_idx * 10_000 + split_idx * 1000 + perm_idx,
                    n_splits=H116_CV_SPLITS,
                )
                null_motif[perm_idx] = float(auc_mp - auc_h70) if np.isfinite(auc_mp) and np.isfinite(auc_h70) else float("nan")
                null_rows.append({
                    "hypothesis_id": "H116",
                    "null_kind": "motif_sign_permutation_within_tf_bin",
                    "domain": domain,
                    "seed_tag": H116_SEED,
                    "split_regime": split_regime,
                    "perm_idx": int(perm_idx),
                    "null_value": float(null_motif[perm_idx]),
                })

                y_perm = BASE.stratified_shuffle(labels, strata, rng).astype(int)
                auc_lp = cv_auc_logit(
                    feat,
                    y_perm,
                    random_state=44_264 + domain_idx * 10_000 + split_idx * 1000 + perm_idx,
                    n_splits=H116_CV_SPLITS,
                )
                auc_h70_lp = BASE.safe_auc(y_perm, h70)
                null_label[perm_idx] = float(auc_lp - auc_h70_lp) if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp) else float("nan")
                null_rows.append({
                    "hypothesis_id": "H116",
                    "null_kind": "label_permutation",
                    "domain": domain,
                    "seed_tag": H116_SEED,
                    "split_regime": split_regime,
                    "perm_idx": int(perm_idx),
                    "null_value": float(null_label[perm_idx]),
                })

            null_pool = np.concatenate([null_motif[np.isfinite(null_motif)], null_label[np.isfinite(null_label)]])
            null_gap = float(delta - np.nanmean(null_pool)) if null_pool.size else float("nan")
            rows.append({
                "hypothesis_id": "H116",
                "domain": domain,
                "seed_tag": H116_SEED,
                "split_regime": split_regime,
                "auc_h70": float(auc_h70),
                "auc_model": float(auc_model),
                "delta_vs_h70": delta,
                "null_gap": null_gap,
                "motif_coverage": float(np.mean(motif_present)),
                "sign_consistency_rate": float(np.mean(sign_consistent[motif_present > 0])) if np.any(motif_present > 0) else 0.0,
                "n_edges": int(labels.size),
            })

    raw_df = pd.DataFrame(rows)
    null_df = pd.DataFrame(null_rows)
    raw_path = ITER_DIR / "h116_trrust_sign_motif_by_domain_split.csv"
    null_path = ITER_DIR / "h116_trrust_sign_motif_null_summary.csv"
    dom_path = ITER_DIR / "h116_trrust_sign_motif_domain_summary.csv"
    raw_df.to_csv(raw_path, index=False)
    null_df.to_csv(null_path, index=False)
    dom_df = domain_split_summary(rows, "H116")
    dom_df.to_csv(dom_path, index=False)

    return {
        "raw": rows,
        "domain_summary": dom_df,
        "artifact_paths": [str(raw_path), str(null_path), str(dom_path)],
    }


def main() -> None:
    dorothea_map = BASE.load_dorothea_score_map()
    omnipath_pairs = BASE.load_omnipath_pairs()
    gene2go_upper = BASE.load_gene2go_upper()
    string_map = BASE.load_string_scores_from_cache(BASE.STRING_CACHE_PATH)

    h115 = run_h115_tangent_acceleration(dorothea_map, omnipath_pairs, gene2go_upper, string_map)
    h116 = run_h116_trrust_sign_motif(dorothea_map, omnipath_pairs, gene2go_upper, string_map)

    summary = {
        "iteration": "iter_0044",
        "hypotheses": {
            "H115": {
                "n_rows": int(len(h115["raw"])),
                "mean_delta": float(np.nanmean(pd.DataFrame(h115["raw"])["delta_vs_h70"])) if h115["raw"] else float("nan"),
                "mean_null_gap": float(np.nanmean(pd.DataFrame(h115["raw"])["null_gap"])) if h115["raw"] else float("nan"),
                "direction_passes": int(h115["domain_summary"]["direction_pass"].sum()) if not h115["domain_summary"].empty else 0,
                "null_passes": int(h115["domain_summary"]["null_pass"].sum()) if not h115["domain_summary"].empty else 0,
                "artifact_paths": h115["artifact_paths"],
            },
            "H116": {
                "n_rows": int(len(h116["raw"])),
                "mean_delta": float(np.nanmean(pd.DataFrame(h116["raw"])["delta_vs_h70"])) if h116["raw"] else float("nan"),
                "mean_null_gap": float(np.nanmean(pd.DataFrame(h116["raw"])["null_gap"])) if h116["raw"] else float("nan"),
                "direction_passes": int(h116["domain_summary"]["direction_pass"].sum()) if not h116["domain_summary"].empty else 0,
                "null_passes": int(h116["domain_summary"]["null_pass"].sum()) if not h116["domain_summary"].empty else 0,
                "artifact_paths": h116["artifact_paths"],
            },
        },
    }

    out_path = ITER_DIR / "iter0044_screen_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
