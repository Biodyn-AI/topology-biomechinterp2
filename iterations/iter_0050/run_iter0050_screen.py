from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors


ITER_DIR = Path("iterations/iter_0050")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base_iter0050")
PREV = load_module(Path("iterations/iter_0047/run_iter0047_screen.py"), "iter0047_prev_iter0050")
ITER49 = load_module(Path("iterations/iter_0049/run_iter0049_screen.py"), "iter0049_prev_iter0050")


# H133 / N663-style slot: persistent-homology surrogate via rank-surface filtration features.
H133_SEED = "seed42_main"
H133_LAYER = 11
H133_SPLITS = ("source_disjoint", "target_disjoint")
H133_GENE_CAP = 180
H133_MIN_GENE_NODES = 120
H133_EDGE_SAMPLE = 240
H133_NEIGHBORS = 12
H133_NULL_PERM = 20
H133_DIST_THRESHOLDS = [0.20, 0.35, 0.50, 0.65]
H133_MARGIN_THRESHOLDS = [0.55, 0.70, 0.85]

# H134 / N667-style slot: intrinsic-dimension phase descriptors along directed paths.
H134_SEED = "seed42_main"
H134_LAYERS = [7, 11]
H134_SPLITS = ("source_disjoint", "target_disjoint")
H134_GENE_CAP = 170
H134_MIN_GENE_NODES = 120
H134_EDGE_SAMPLE = 230
H134_NEIGHBORS = 12
H134_CV_SPLITS = 4
H134_NULL_PERM = 18

# H135 / N670-style carry-over refinement: hard-slice semantic hardening rerun.
H135_SEEDS = ["seed42_main", "seed43", "seed44"]
H135_NULL_PERM = 32


def ensure_required_inputs() -> None:
    required = [
        BASE.DOROTHEA_PATH,
        BASE.GENE2GO_PATH,
        BASE.OMNIPATH_INTERACTIONS_PATH,
        BASE.STRING_CACHE_PATH,
        PREV.TRRUST_PATH,
    ]

    for domain, run_map in BASE.SCGPT_RUNS_BY_DOMAIN.items():
        # H133 and H134 use seed42 on all domains.
        run_dir = run_map[H133_SEED]
        required.append(run_dir / "cycle1_edge_dataset.tsv")
        required.append(run_dir / "layer_gene_embeddings.npy")

    # H135 uses immune + lung hard-slices across three seeds.
    for domain in ("immune", "lung"):
        run_map = BASE.SCGPT_RUNS_BY_DOMAIN[domain]
        for seed_tag in H135_SEEDS:
            run_dir = run_map[seed_tag]
            required.append(run_dir / "cycle1_edge_dataset.tsv")
            required.append(run_dir / "layer_gene_embeddings.npy")

    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def finite_q95(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, 0.95))


def finite_fisher(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(BASE.safe_fisher_p(vals))


def summarize_by_domain_split(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (domain, split_regime), group in df.groupby(["domain", "split_regime"], sort=True):
        rows.append(
            {
                "domain": str(domain),
                "split_regime": str(split_regime),
                "n_rows": int(group.shape[0]),
                "mean_delta_vs_h70": float(group["delta_vs_h70"].mean()),
                "mean_null_gap_q95": float(group["null_gap_q95"].mean()),
                "fraction_delta_positive": float((group["delta_vs_h70"] > 0.0).mean()),
                "fraction_null_gap_positive": float((group["null_gap_q95"] > 0.0).mean()),
                "combined_fisher_p_best": finite_fisher(group["p_best_upper"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows).sort_values(["domain", "split_regime"])


def id_sign_flip_count(values: np.ndarray, eps: float = 1e-6) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 3:
        return 0.0
    delta = np.diff(arr)
    signs = np.zeros(delta.size, dtype=int)
    signs[delta > eps] = 1
    signs[delta < -eps] = -1
    kept = signs[signs != 0]
    if kept.size < 2:
        return 0.0
    return float(np.sum(kept[1:] != kept[:-1]))


def id_phase_descriptor(seq_forward: np.ndarray, seq_reverse: np.ndarray) -> np.ndarray:
    fwd = np.asarray(seq_forward, dtype=float)
    rev = np.asarray(seq_reverse, dtype=float)
    if fwd.size == 0:
        fwd = np.array([1.0], dtype=float)
    if rev.size == 0:
        rev = np.array([1.0], dtype=float)

    rev_aligned = rev[::-1]
    m = int(min(fwd.size, rev_aligned.size))
    if m > 0:
        denom = float(np.mean(np.abs(np.concatenate([fwd[:m], rev_aligned[:m]]))))
        denom = max(denom, 1e-8)
        hysteresis = float(np.mean(np.abs(fwd[:m] - rev_aligned[:m])) / denom)
    else:
        hysteresis = 0.0

    slope_f = float(fwd[-1] - fwd[0]) if fwd.size >= 2 else 0.0
    # Keep reverse-path direction (target->source) for directional asymmetry.
    slope_r_dir = float(rev[-1] - rev[0]) if rev.size >= 2 else 0.0

    flip_total = id_sign_flip_count(fwd) + id_sign_flip_count(rev)
    slope_asym = slope_f + slope_r_dir
    path_len_asym = float(fwd.size - rev.size)
    mean_id = float(0.5 * (np.mean(fwd) + np.mean(rev_aligned)))
    disp_id = float(0.5 * (np.std(fwd) + np.std(rev_aligned)))

    return np.asarray(
        [
            flip_total,
            hysteresis,
            slope_asym,
            path_len_asym,
            mean_id,
            disp_id,
        ],
        dtype=float,
    )


def swapped_id_phase_descriptor(desc: np.ndarray) -> np.ndarray:
    out = np.asarray(desc, dtype=float).copy()
    # Direction-sensitive coordinates flip sign under source/target swap.
    out[2] = -out[2]  # slope_asym
    out[3] = -out[3]  # path_len_asym
    return out


def build_h134_feature_matrix(h70: np.ndarray, desc: np.ndarray) -> np.ndarray:
    x = np.asarray(desc, dtype=float)
    return np.column_stack(
        [
            h70,
            x,
            h70 * x[:, 1],  # hysteresis interaction
            h70 * x[:, 2],  # slope-asym interaction
        ]
    )


def run_h133_rank_surface_persistence(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_idx, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H133_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_idx, split_regime in enumerate(H133_SPLITS):
            split_mask = split_masks.get(split_regime)
            if split_mask is None:
                continue

            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H133_GENE_CAP))
            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
            ].copy()
            if split_edges["label"].nunique() < 2:
                continue

            edge_gene_indices, gene_to_local, symbols, support_dir = PREV.build_symbol_resources(
                split_edges=split_edges,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )
            if edge_gene_indices.size < H133_MIN_GENE_NODES:
                continue

            source_local_all = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
            target_local_all = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
            labels_all = split_edges["label"].to_numpy(dtype=int)

            rng = np.random.default_rng(50_133 + domain_idx * 1000 + split_idx * 100)
            sample_idx = PREV.stratified_index_sample(labels_all, max_n=H133_EDGE_SAMPLE, rng=rng)
            if sample_idx.size < 120:
                continue

            source_local = source_local_all[sample_idx]
            target_local = target_local_all[sample_idx]
            labels = labels_all[sample_idx]

            if H133_LAYER >= layer_embeddings.shape[0]:
                continue
            points = layer_embeddings[H133_LAYER, edge_gene_indices, :]
            points_pca = BASE.reduce_points(
                points,
                n_components=20,
                random_state=50_134 + domain_idx * 1000 + split_idx * 100,
            )

            geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H133_NEIGHBORS)
            geodesic_w = PREV.confidence_weighted_geodesic(geodesic, support_dir)
            h70 = PREV.compute_h70_scores(
                geodesic=geodesic_w,
                support_dir=support_dir,
                source_local=source_local,
                target_local=target_local,
                triangle_k=[8, 12, 16],
            )

            dist_matrix = cdist(points_pca, points_pca, metric="euclidean")
            margin_matrix = np.abs(support_dir - support_dir.T)
            dist_rank = BASE.symmetric_global_rank_matrix(dist_matrix)
            margin_rank = BASE.symmetric_global_rank_matrix(margin_matrix)

            rank_bundle = BASE.rank_surface_feature_bundle(
                dist_rank_matrix=dist_rank,
                margin_rank_matrix=margin_rank,
                source_local=source_local,
                target_local=target_local,
                dist_thresholds=H133_DIST_THRESHOLDS,
                margin_thresholds=H133_MARGIN_THRESHOLDS,
            )
            one_axis_conn = BASE.one_axis_rank_connectivity(
                dist_rank_matrix=dist_rank,
                source_local=source_local,
                target_local=target_local,
                dist_thresholds=H133_DIST_THRESHOLDS,
            )

            score = (
                h70
                + 0.35 * BASE.zscore(rank_bundle["conn_mean"])
                + 0.20 * BASE.zscore(rank_bundle["conn_var"])
                + 0.20 * BASE.zscore(rank_bundle["conn_high_minus_low_margin"])
                + 0.15 * BASE.zscore(rank_bundle["conn_scale_slope"])
                + 0.10 * BASE.zscore(rank_bundle["cycle_mean"])
            )

            auc_h70 = BASE.safe_auc(labels, h70)
            auc_model = BASE.safe_auc(labels, score)
            delta = float(auc_model - auc_h70) if np.isfinite(auc_h70) and np.isfinite(auc_model) else float("nan")

            knn_edges = BASE.build_knn_edge_array(points=points_pca, n_neighbors=H133_NEIGHBORS)
            if knn_edges.size > 0:
                node_deg = np.bincount(
                    np.concatenate([knn_edges[:, 0], knn_edges[:, 1]]),
                    minlength=edge_gene_indices.size,
                )
            else:
                node_deg = np.zeros(edge_gene_indices.size, dtype=int)
            node_bins = BASE.degree_bins(node_deg.astype(float), max_bins=6)
            node_margin_rank_strength = margin_rank.mean(axis=1)

            edge_len = geodesic_w[source_local, target_local]
            deg_sum = PREV.edge_degree_sum(points_pca, H133_NEIGHBORS, source_local, target_local)
            edge_strata = PREV.build_edge_strata(edge_len, deg_sum, max_len_bins=6, max_deg_bins=4)

            null_axis = np.empty(H133_NULL_PERM, dtype=float)
            null_label = np.empty(H133_NULL_PERM, dtype=float)

            for perm_idx in range(H133_NULL_PERM):
                margin_perm_node = BASE.shuffle_within_bins(node_margin_rank_strength, node_bins, rng)
                margin_rank_perm = 0.5 * (margin_perm_node[:, None] + margin_perm_node[None, :])
                perm_bundle = BASE.rank_surface_feature_bundle(
                    dist_rank_matrix=dist_rank,
                    margin_rank_matrix=margin_rank_perm,
                    source_local=source_local,
                    target_local=target_local,
                    dist_thresholds=H133_DIST_THRESHOLDS,
                    margin_thresholds=H133_MARGIN_THRESHOLDS,
                )
                perm_score = (
                    h70
                    + 0.35 * BASE.zscore(perm_bundle["conn_mean"])
                    + 0.20 * BASE.zscore(perm_bundle["conn_var"])
                    + 0.20 * BASE.zscore(perm_bundle["conn_high_minus_low_margin"])
                    + 0.15 * BASE.zscore(perm_bundle["conn_scale_slope"])
                    + 0.10 * BASE.zscore(perm_bundle["cycle_mean"])
                )
                auc_perm = BASE.safe_auc(labels, perm_score)
                null_axis[perm_idx] = (
                    float(auc_perm - auc_h70) if np.isfinite(auc_perm) and np.isfinite(auc_h70) else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H133",
                        "null_kind": "axis_rank_permutation_within_degree_bins",
                        "domain": domain,
                        "seed_tag": H133_SEED,
                        "split_regime": split_regime,
                        "layer": int(H133_LAYER),
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_axis[perm_idx]),
                    }
                )

                y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                auc_lp = BASE.safe_auc(y_perm, score)
                auc_h70_lp = BASE.safe_auc(y_perm, h70)
                null_label[perm_idx] = (
                    float(auc_lp - auc_h70_lp) if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp) else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H133",
                        "null_kind": "label_permutation",
                        "domain": domain,
                        "seed_tag": H133_SEED,
                        "split_regime": split_regime,
                        "layer": int(H133_LAYER),
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_label[perm_idx]),
                    }
                )

            all_null = np.concatenate([null_axis, null_label])
            q95 = finite_q95(all_null)
            p_axis = BASE.empirical_upper_tail_p(delta, null_axis)
            p_label = BASE.empirical_upper_tail_p(delta, null_label)
            p_best = float(np.nanmin(np.asarray([p_axis, p_label], dtype=float)))

            rows.append(
                {
                    "hypothesis_id": "H133",
                    "domain": domain,
                    "seed_tag": H133_SEED,
                    "split_regime": split_regime,
                    "layer": int(H133_LAYER),
                    "n_edges_eval": int(labels.size),
                    "n_positive_eval": int(labels.sum()),
                    "auc_h70": float(auc_h70),
                    "auc_rank_surface_model": float(auc_model),
                    "delta_vs_h70": float(delta),
                    "q95_null_delta_auc": float(q95),
                    "null_gap_q95": float(delta - q95),
                    "p_axis_upper": float(p_axis),
                    "p_label_upper": float(p_label),
                    "p_best_upper": float(p_best),
                    "mean_conn_mean": float(np.mean(rank_bundle["conn_mean"])),
                    "mean_conn_high_minus_low_margin": float(np.mean(rank_bundle["conn_high_minus_low_margin"])),
                    "mean_one_axis_conn": float(np.mean(one_axis_conn)),
                }
            )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime"])
    by_row_path = ITER_DIR / "h133_rank_surface_persistence_by_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h133_rank_surface_persistence_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_df = summarize_by_domain_split(by_row_df)
    summary_path = ITER_DIR / "h133_rank_surface_persistence_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_vs_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_vs_h70"] > 0.0).sum()) if not summary_df.empty else 0,
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95"] > 0.0).sum()) if not summary_df.empty else 0,
        "artifact_paths": {
            "by_domain_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h134_id_phase_descriptor_screen(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_idx, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H134_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_idx, split_regime in enumerate(H134_SPLITS):
            split_mask = split_masks.get(split_regime)
            if split_mask is None:
                continue
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H134_GENE_CAP))
            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
            ].copy()
            if split_edges["label"].nunique() < 2:
                continue

            edge_gene_indices, gene_to_local, symbols, support_dir = PREV.build_symbol_resources(
                split_edges=split_edges,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )
            if edge_gene_indices.size < H134_MIN_GENE_NODES:
                continue

            source_local_all = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
            target_local_all = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
            labels_all = split_edges["label"].to_numpy(dtype=int)

            for layer in H134_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                rng = np.random.default_rng(50_210 + domain_idx * 1000 + split_idx * 100 + layer)
                sample_idx = PREV.stratified_index_sample(labels_all, max_n=H134_EDGE_SAMPLE, rng=rng)
                if sample_idx.size < 120:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=50_211 + domain_idx * 1000 + split_idx * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H134_NEIGHBORS)
                geodesic_w = PREV.confidence_weighted_geodesic(geodesic, support_dir)
                h70 = PREV.compute_h70_scores(
                    geodesic=geodesic_w,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=[8, 12, 16],
                )

                _, graph_directed = PREV.build_directed_knn_weighted_graph(
                    points=points_pca,
                    support_dir=support_dir,
                    n_neighbors=H134_NEIGHBORS,
                )
                _, predecessors = shortest_path(
                    graph_directed,
                    directed=True,
                    unweighted=False,
                    return_predecessors=True,
                )

                nbrs = NearestNeighbors(n_neighbors=3, metric="euclidean")
                nbrs.fit(points_pca)
                d_full, _ = nbrs.kneighbors(points_pca)
                node_id = BASE.local_id_two_nn(np.asarray(d_full[:, 1:3], dtype=float))

                desc = np.zeros((labels.size, 6), dtype=float)
                swapped_desc = np.zeros((labels.size, 6), dtype=float)
                seq_forward: list[np.ndarray] = []
                seq_reverse: list[np.ndarray] = []

                for i, (src, tgt) in enumerate(zip(source_local, target_local)):
                    p_f = PREV.path_nodes_from_predecessor(predecessors, src=int(src), tgt=int(tgt))
                    p_r = PREV.path_nodes_from_predecessor(predecessors, src=int(tgt), tgt=int(src))

                    seq_f = node_id[np.asarray(p_f, dtype=int)]
                    seq_r = node_id[np.asarray(p_r, dtype=int)]
                    seq_forward.append(np.asarray(seq_f, dtype=float))
                    seq_reverse.append(np.asarray(seq_r, dtype=float))

                    d_vec = id_phase_descriptor(seq_f, seq_r)
                    desc[i] = d_vec
                    swapped_desc[i] = swapped_id_phase_descriptor(d_vec)

                feat = build_h134_feature_matrix(h70, desc)
                auc_h70 = BASE.safe_auc(labels, h70)
                auc_model = PREV.cv_auc_logit(
                    feat,
                    labels,
                    random_state=50_212 + domain_idx * 1000 + split_idx * 100 + layer,
                    n_splits=H134_CV_SPLITS,
                )
                delta = float(auc_model - auc_h70) if np.isfinite(auc_model) and np.isfinite(auc_h70) else float("nan")

                edge_len = geodesic_w[source_local, target_local]
                deg_sum = PREV.edge_degree_sum(points_pca, H134_NEIGHBORS, source_local, target_local)
                edge_strata = PREV.build_edge_strata(edge_len, deg_sum, max_len_bins=6, max_deg_bins=4)

                null_profile = np.empty(H134_NULL_PERM, dtype=float)
                null_swap = np.empty(H134_NULL_PERM, dtype=float)
                null_label = np.empty(H134_NULL_PERM, dtype=float)

                for perm_idx in range(H134_NULL_PERM):
                    # Null 1: destroy profile order while keeping marginal ID values per path.
                    desc_profile = np.zeros_like(desc, dtype=float)
                    for i in range(labels.size):
                        sf = seq_forward[i].copy()
                        sr = seq_reverse[i].copy()
                        if sf.size > 1:
                            sf = sf[rng.permutation(sf.size)]
                        if sr.size > 1:
                            sr = sr[rng.permutation(sr.size)]
                        desc_profile[i] = id_phase_descriptor(sf, sr)

                    feat_profile = build_h134_feature_matrix(h70, desc_profile)
                    auc_profile = PREV.cv_auc_logit(
                        feat_profile,
                        labels,
                        random_state=50_213 + domain_idx * 10_000 + split_idx * 1000 + layer * 100 + perm_idx,
                        n_splits=H134_CV_SPLITS,
                    )
                    null_profile[perm_idx] = (
                        float(auc_profile - auc_h70)
                        if np.isfinite(auc_profile) and np.isfinite(auc_h70)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H134",
                            "null_kind": "id_profile_permutation_along_path",
                            "domain": domain,
                            "seed_tag": H134_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_profile[perm_idx]),
                        }
                    )

                    # Null 2: endpoint swap stochasticity within edge strata.
                    desc_swapped = desc.copy()
                    for g in np.unique(edge_strata):
                        idx = np.where(edge_strata == g)[0]
                        if idx.size <= 1:
                            continue
                        choose = rng.random(idx.size) < 0.5
                        if np.any(choose):
                            desc_swapped[idx[choose]] = swapped_desc[idx[choose]]

                    feat_swap = build_h134_feature_matrix(h70, desc_swapped)
                    auc_swap = PREV.cv_auc_logit(
                        feat_swap,
                        labels,
                        random_state=50_214 + domain_idx * 10_000 + split_idx * 1000 + layer * 100 + perm_idx,
                        n_splits=H134_CV_SPLITS,
                    )
                    null_swap[perm_idx] = (
                        float(auc_swap - auc_h70) if np.isfinite(auc_swap) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H134",
                            "null_kind": "endpoint_swap_within_distance_degree_bins",
                            "domain": domain,
                            "seed_tag": H134_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_swap[perm_idx]),
                        }
                    )

                    y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                    auc_lp = PREV.cv_auc_logit(
                        feat,
                        y_perm,
                        random_state=50_215 + domain_idx * 10_000 + split_idx * 1000 + layer * 100 + perm_idx,
                        n_splits=H134_CV_SPLITS,
                    )
                    auc_h70_lp = BASE.safe_auc(y_perm, h70)
                    null_label[perm_idx] = (
                        float(auc_lp - auc_h70_lp) if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H134",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H134_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_profile, null_swap, null_label])
                q95 = finite_q95(all_null)
                p_profile = BASE.empirical_upper_tail_p(delta, null_profile)
                p_swap = BASE.empirical_upper_tail_p(delta, null_swap)
                p_label = BASE.empirical_upper_tail_p(delta, null_label)
                p_best = float(np.nanmin(np.asarray([p_profile, p_swap, p_label], dtype=float)))

                rows.append(
                    {
                        "hypothesis_id": "H134",
                        "domain": domain,
                        "seed_tag": H134_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70": float(auc_h70),
                        "auc_id_phase_model": float(auc_model),
                        "delta_vs_h70": float(delta),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95": float(delta - q95),
                        "p_profile_upper": float(p_profile),
                        "p_swap_upper": float(p_swap),
                        "p_label_upper": float(p_label),
                        "p_best_upper": float(p_best),
                        "id_sign_flip_count_mean": float(np.mean(desc[:, 0])),
                        "id_hysteresis_mean": float(np.mean(desc[:, 1])),
                        "id_slope_asymmetry_mean": float(np.mean(desc[:, 2])),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h134_id_phase_descriptor_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h134_id_phase_descriptor_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_df = summarize_by_domain_split(by_row_df)
    summary_path = ITER_DIR / "h134_id_phase_descriptor_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_vs_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_vs_h70"] > 0.0).sum()) if not summary_df.empty else 0,
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95"] > 0.0).sum()) if not summary_df.empty else 0,
        "artifact_paths": {
            "by_domain_split_layer": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h135_hard_slice_semantic_refinement(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
    trrust_sign_map: dict[tuple[str, str], int],
    trrust_tf_out_degree: dict[str, int],
) -> dict[str, object]:
    # Reuse the validated H130 engine but constrain to hard slices and higher null budget.
    iter49 = ITER49
    iter49.ITER_DIR = ITER_DIR

    orig_runs = iter49.BASE.SCGPT_RUNS_BY_DOMAIN
    orig_split_fn = iter49.PREV.build_split_masks_plus
    orig_seeds = list(iter49.H130_SEEDS)
    orig_null_perm = int(iter49.H130_NULL_PERM)

    try:
        iter49.H130_SEEDS = list(H135_SEEDS)
        iter49.H130_NULL_PERM = int(H135_NULL_PERM)
        iter49.BASE.SCGPT_RUNS_BY_DOMAIN = {
            "immune": orig_runs["immune"],
            "lung": orig_runs["lung"],
        }

        def hard_slice_masks(edge_df: pd.DataFrame) -> dict[str, np.ndarray]:
            masks = orig_split_fn(edge_df)
            out: dict[str, np.ndarray] = {}
            source = masks.get("source_disjoint")
            dual = masks.get("dual_axis_disjoint")
            if source is not None:
                out["source_disjoint"] = source
            if dual is not None:
                out["dual_axis_disjoint"] = dual
            return out

        iter49.PREV.build_split_masks_plus = hard_slice_masks

        _ = iter49.run_h130_semantic_hardening(
            dorothea_map=dorothea_map,
            omnipath_pairs=omnipath_pairs,
            gene2go_upper=gene2go_upper,
            string_map=string_map,
            trrust_sign_map=trrust_sign_map,
            trrust_tf_out_degree=trrust_tf_out_degree,
        )
    finally:
        iter49.BASE.SCGPT_RUNS_BY_DOMAIN = orig_runs
        iter49.PREV.build_split_masks_plus = orig_split_fn
        iter49.H130_SEEDS = orig_seeds
        iter49.H130_NULL_PERM = orig_null_perm

    base_by_row_path = ITER_DIR / "h130_semantic_go_string_hardening_by_seed_domain_split.csv"
    base_null_path = ITER_DIR / "h130_semantic_go_string_hardening_null_summary.csv"

    by_row_df = pd.read_csv(base_by_row_path)
    by_row_df["hypothesis_id"] = "H135"
    by_row_df = by_row_df.loc[
        by_row_df["split_regime"].isin(["source_disjoint", "dual_axis_disjoint"])
        & by_row_df["domain"].isin(["immune", "lung"])
    ].copy()
    by_row_df = by_row_df.sort_values(["seed_tag", "domain", "split_regime"])

    null_df = pd.read_csv(base_null_path)
    null_df["hypothesis_id"] = "H135"
    null_df = null_df.loc[
        null_df["split_regime"].isin(["source_disjoint", "dual_axis_disjoint"])
        & null_df["domain"].isin(["immune", "lung"])
    ].copy()
    null_df = null_df.sort_values(["null_kind", "seed_tag", "domain", "split_regime", "perm_idx"])

    summary_df = summarize_by_domain_split(by_row_df)

    by_row_path = ITER_DIR / "h135_hard_slice_semantic_refinement_by_seed_domain_split.csv"
    null_path = ITER_DIR / "h135_hard_slice_semantic_refinement_null_summary.csv"
    summary_path = ITER_DIR / "h135_hard_slice_semantic_refinement_domain_summary.csv"
    by_row_df.to_csv(by_row_path, index=False)
    null_df.to_csv(null_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    lung_dual = by_row_df.loc[
        (by_row_df["domain"] == "lung") & (by_row_df["split_regime"] == "dual_axis_disjoint")
    ]
    immune_source = by_row_df.loc[
        (by_row_df["domain"] == "immune") & (by_row_df["split_regime"] == "source_disjoint")
    ]

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_vs_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_vs_h70"] > 0.0).sum()) if not summary_df.empty else 0,
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95"] > 0.0).sum()) if not summary_df.empty else 0,
        "lung_dual_axis_mean_null_gap": float(lung_dual["null_gap_q95"].mean()) if not lung_dual.empty else float("nan"),
        "immune_source_mean_null_gap": float(immune_source["null_gap_q95"].mean()) if not immune_source.empty else float("nan"),
        "artifact_paths": {
            "by_seed_domain_split": str(by_row_path),
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
    trrust_sign_map, trrust_tf_out_degree = PREV.load_trrust_signed_map()

    h133_summary = run_h133_rank_surface_persistence(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h134_summary = run_h134_id_phase_descriptor_screen(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h135_summary = run_h135_hard_slice_semantic_refinement(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
        trrust_sign_map=trrust_sign_map,
        trrust_tf_out_degree=trrust_tf_out_degree,
    )

    summary = {
        "iteration": "iter_0050",
        "h133_rank_surface_persistence": h133_summary,
        "h134_id_phase_descriptor": h134_summary,
        "h135_hard_slice_semantic_refinement": h135_summary,
    }
    summary_path = ITER_DIR / "iter0050_screen_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
