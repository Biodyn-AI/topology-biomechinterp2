from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit

ITER_DIR = Path("iterations/iter_0029")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def load_iter0028_module():
    """Load the previous iteration runner as a helper module for shared utilities."""
    module_path = Path("iterations/iter_0028/run_iter0028_screen.py")
    spec = importlib.util.spec_from_file_location("iter0028_base", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_iter0028_module()

# H70 / N343: robustness expansion of H69 with stronger null calibration.
H70_LAYERS = [7, 11]
H70_GENE_CAP = 170
H70_NEIGHBORS = 12
H70_TRIANGLE_K = [8, 12, 16]
H70_NULL_PERM = 48
H70_BOOTSTRAP_N = 4000

# H71 / N350: topology-signature distillation transfer (major-change rescue).
H71_LAYERS = [7, 11]
H71_GENE_CAP = 220
H71_CODEBOOK_TOKENS = 12
H71_NULL_PERM = 24

# H72 / N355: edge trajectory motif class pilot screen.
H72_LAYERS = [0, 3, 7, 11]
H72_SEED_TAG = "seed42_main"
H72_GENE_CAP = 170
H72_NEIGHBORS = 12
H72_TRIANGLE_K = [8, 12]
H72_MOTIF_K = 5
H72_NULL_PERM = 12


def safe_logit(p: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(p, dtype=float)
    clipped = np.clip(arr, 1e-4, 1.0 - 1e-4)
    out = np.log(clipped / (1.0 - clipped))
    if np.isscalar(p):
        return float(out)
    return out


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        value = float(arr[0])
        return value, value, value

    means = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = float(np.mean(sample))
    center = float(np.mean(arr))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return center, lo, hi


def motif_rate_table(
    motif_ids: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    motif = np.asarray(motif_ids, dtype=int)
    y = np.asarray(labels, dtype=float)
    counts = np.bincount(motif, minlength=n_clusters).astype(float)
    positives = np.bincount(motif, weights=y, minlength=n_clusters).astype(float)
    rates = (positives + 1.0) / np.clip(counts + 2.0, 1.0, None)
    return counts, positives, rates


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
    for p in BASE.GENEFORMER_EDGE_BY_DOMAIN.values():
        required_paths.append(p)

    missing = [str(p) for p in required_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def run_h70_triangle_defect_hard_null(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_index, (seed_tag, run_dir) in enumerate(run_map.items()):
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            split_masks = BASE.build_split_masks(edge_df)

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H70_GENE_CAP))
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
                source_local = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
                target_local = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
                labels = split_edges["label"].to_numpy(dtype=int)

                for layer in H70_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = BASE.reduce_points(
                        points,
                        n_components=20,
                        random_state=29_700 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer,
                    )
                    geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H70_NEIGHBORS)

                    edge_geodesic = geodesic[source_local, target_local]
                    edge_support = support_dir[source_local, target_local]
                    edge_margin = np.abs(
                        support_dir[source_local, target_local] - support_dir[target_local, source_local]
                    )
                    feature_bundle = BASE.multiscale_triangle_defect_features(
                        geodesic=geodesic,
                        source_local=source_local,
                        target_local=target_local,
                        k_values=H70_TRIANGLE_K,
                    )

                    baseline_score = (
                        BASE.zscore(-edge_geodesic)
                        + 0.75 * BASE.zscore(edge_support)
                        + 0.35 * BASE.zscore(edge_margin)
                    )
                    defect_score = (
                        baseline_score
                        + 0.35 * BASE.zscore(-feature_bundle["median_mean"])
                        + 0.25 * BASE.zscore(-feature_bundle["tail_mean"])
                        + 0.20 * BASE.zscore(feature_bundle["close_frac_mean"])
                        + 0.10 * BASE.zscore(-feature_bundle["scale_span"])
                        + 0.10 * BASE.zscore(-feature_bundle["dispersion_mean"])
                    )

                    auc_baseline = BASE.safe_auc(labels, baseline_score)
                    auc_defect = BASE.safe_auc(labels, defect_score)
                    delta_auc = (
                        float(auc_defect - auc_baseline)
                        if np.isfinite(auc_defect) and np.isfinite(auc_baseline)
                        else float("nan")
                    )

                    edge_bins = BASE.degree_bins(edge_geodesic, max_bins=6)
                    rng = np.random.default_rng(
                        29_701 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer
                    )
                    null_endpoint_swap = np.empty(H70_NULL_PERM, dtype=np.float64)
                    null_random_third = np.empty(H70_NULL_PERM, dtype=np.float64)
                    null_label = np.empty(H70_NULL_PERM, dtype=np.float64)

                    for perm_idx in range(H70_NULL_PERM):
                        target_swap = target_local.copy()
                        for b in np.unique(edge_bins):
                            idx = np.where(edge_bins == b)[0]
                            if idx.size > 1:
                                target_swap[idx] = rng.permutation(target_swap[idx])
                        swap_bundle = BASE.multiscale_triangle_defect_features(
                            geodesic=geodesic,
                            source_local=source_local,
                            target_local=target_swap,
                            k_values=H70_TRIANGLE_K,
                        )
                        swap_score = (
                            baseline_score
                            + 0.35 * BASE.zscore(-swap_bundle["median_mean"])
                            + 0.25 * BASE.zscore(-swap_bundle["tail_mean"])
                            + 0.20 * BASE.zscore(swap_bundle["close_frac_mean"])
                            + 0.10 * BASE.zscore(-swap_bundle["scale_span"])
                            + 0.10 * BASE.zscore(-swap_bundle["dispersion_mean"])
                        )
                        auc_swap = BASE.safe_auc(labels, swap_score)
                        delta_swap = (
                            float(auc_swap - auc_baseline)
                            if np.isfinite(auc_swap) and np.isfinite(auc_baseline)
                            else float("nan")
                        )
                        null_endpoint_swap[perm_idx] = delta_swap
                        null_rows.append(
                            {
                                "null_kind": "endpoint_swap_within_geodesic_bins",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_swap),
                            }
                        )

                        random_bundle = BASE.multiscale_triangle_defect_features(
                            geodesic=geodesic,
                            source_local=source_local,
                            target_local=target_local,
                            k_values=H70_TRIANGLE_K,
                            rng=rng,
                            random_third=True,
                        )
                        random_score = (
                            baseline_score
                            + 0.35 * BASE.zscore(-random_bundle["median_mean"])
                            + 0.25 * BASE.zscore(-random_bundle["tail_mean"])
                            + 0.20 * BASE.zscore(random_bundle["close_frac_mean"])
                            + 0.10 * BASE.zscore(-random_bundle["scale_span"])
                            + 0.10 * BASE.zscore(-random_bundle["dispersion_mean"])
                        )
                        auc_random = BASE.safe_auc(labels, random_score)
                        delta_random = (
                            float(auc_random - auc_baseline)
                            if np.isfinite(auc_random) and np.isfinite(auc_baseline)
                            else float("nan")
                        )
                        null_random_third[perm_idx] = delta_random
                        null_rows.append(
                            {
                                "null_kind": "matched_random_third_nodes",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_random),
                            }
                        )

                        labels_perm = BASE.stratified_shuffle(labels, edge_bins, rng).astype(int)
                        auc_lp = BASE.safe_auc(labels_perm, defect_score)
                        auc_lp_base = BASE.safe_auc(labels_perm, baseline_score)
                        delta_lp = (
                            float(auc_lp - auc_lp_base)
                            if np.isfinite(auc_lp) and np.isfinite(auc_lp_base)
                            else float("nan")
                        )
                        null_label[perm_idx] = delta_lp
                        null_rows.append(
                            {
                                "null_kind": "label_permutation",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_lp),
                            }
                        )

                    p_swap = BASE.empirical_upper_tail_p(delta_auc, null_endpoint_swap)
                    p_random = BASE.empirical_upper_tail_p(delta_auc, null_random_third)
                    p_label = BASE.empirical_upper_tail_p(delta_auc, null_label)
                    p_best = np.nanmin(np.array([p_swap, p_random, p_label], dtype=float))

                    q95_swap = float(np.quantile(null_endpoint_swap[np.isfinite(null_endpoint_swap)], 0.95))
                    q95_random = float(np.quantile(null_random_third[np.isfinite(null_random_third)], 0.95))
                    q95_label = float(np.quantile(null_label[np.isfinite(null_label)], 0.95))

                    rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "n_edges_eval": int(labels.size),
                            "n_positive_eval": int(labels.sum()),
                            "n_nodes_graph": int(edge_gene_indices.size),
                            "auc_triangle_defect": float(auc_defect),
                            "auc_directed_geodesic_baseline": float(auc_baseline),
                            "delta_auc_triangle_defect_minus_baseline": float(delta_auc),
                            "mean_triangle_median_defect": float(np.mean(feature_bundle["median_mean"])),
                            "mean_triangle_tail_defect": float(np.mean(feature_bundle["tail_mean"])),
                            "mean_triangle_close_frac": float(np.mean(feature_bundle["close_frac_mean"])),
                            "q95_endpoint_swap": q95_swap,
                            "q95_random_third": q95_random,
                            "q95_label": q95_label,
                            "null_gap_q95_endpoint_swap": float(delta_auc - q95_swap),
                            "null_gap_q95_random_third": float(delta_auc - q95_random),
                            "null_gap_q95_label": float(delta_auc - q95_label),
                            "p_swap_upper": float(p_swap),
                            "p_random_third_upper": float(p_random),
                            "p_label_upper": float(p_label),
                            "p_best_upper": float(p_best),
                        }
                    )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h70_triangle_defect_robust_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h70_triangle_defect_robust_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            rng = np.random.default_rng(abs(hash((domain, split_regime, "h70"))) % (2**32))
            mean_delta, ci_lo, ci_hi = bootstrap_mean_ci(
                group["delta_auc_triangle_defect_minus_baseline"].to_numpy(dtype=float),
                rng=rng,
                n_boot=H70_BOOTSTRAP_N,
                alpha=0.05,
            )
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_auc_triangle_defect": float(group["auc_triangle_defect"].mean()),
                    "mean_auc_directed_geodesic_baseline": float(
                        group["auc_directed_geodesic_baseline"].mean()
                    ),
                    "mean_delta_auc_triangle_defect_minus_baseline": float(mean_delta),
                    "bootstrap_ci95_delta_lo": float(ci_lo),
                    "bootstrap_ci95_delta_hi": float(ci_hi),
                    "mean_null_gap_q95_endpoint_swap": float(group["null_gap_q95_endpoint_swap"].mean()),
                    "mean_null_gap_q95_random_third": float(group["null_gap_q95_random_third"].mean()),
                    "mean_null_gap_q95_label": float(group["null_gap_q95_label"].mean()),
                    "fraction_delta_positive": float(
                        (group["delta_auc_triangle_defect_minus_baseline"] > 0.0).mean()
                    ),
                    "fraction_random_null_gap_positive": float((group["null_gap_q95_random_third"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(
                        BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))
                    ),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h70_triangle_defect_robust_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    global_mean_delta = (
        float(by_row_df["delta_auc_triangle_defect_minus_baseline"].mean()) if not by_row_df.empty else float("nan")
    )
    positive_domain_splits = (
        int((summary_df["mean_delta_auc_triangle_defect_minus_baseline"] > 0.0).sum())
        if not summary_df.empty
        else 0
    )
    random_gap_positive_domain_splits = (
        int((summary_df["mean_null_gap_q95_random_third"] > 0.0).sum()) if not summary_df.empty else 0
    )

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": global_mean_delta,
        "positive_domain_splits": positive_domain_splits,
        "random_gap_positive_domain_splits": random_gap_positive_domain_splits,
        "artifact_paths": {
            "by_seed_layer_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h71_topology_signature_distillation() -> dict[str, object]:
    by_row: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    domains = ["immune", "lung", "external_lung"]
    sc_edges_seed42: dict[str, pd.DataFrame] = {}
    sc_layers_seed42: dict[str, np.ndarray] = {}
    gf_edges: dict[str, pd.DataFrame] = {}

    for domain in domains:
        sc_run = BASE.SCGPT_RUNS_BY_DOMAIN[domain]["seed42_main"]
        sc_edges_seed42[domain] = pd.read_csv(sc_run / "cycle1_edge_dataset.tsv", sep="\t")
        sc_layers_seed42[domain] = np.load(sc_run / "layer_gene_embeddings.npy", mmap_mode="r")
        gf_edges[domain] = pd.read_csv(BASE.GENEFORMER_EDGE_BY_DOMAIN[domain], sep="\t")

    utility_weight_by_domain: dict[str, pd.Series] = {}
    sc_sig_by_domain_layer: dict[tuple[str, int], pd.DataFrame] = {}
    gf_sig_by_domain: dict[str, pd.DataFrame] = {}

    for domain_index, domain in enumerate(domains):
        sc_df = sc_edges_seed42[domain].copy()
        top_genes = set(BASE.select_top_genes(sc_df, gene_cap=H71_GENE_CAP))
        sc_df = sc_df.loc[sc_df["source_idx"].isin(top_genes) & sc_df["target_idx"].isin(top_genes)].copy()
        if sc_df.empty:
            continue

        gene_indices = np.unique(
            np.concatenate(
                [
                    sc_df["source_idx"].to_numpy(dtype=int),
                    sc_df["target_idx"].to_numpy(dtype=int),
                ]
            )
        )
        symbol_map = BASE.build_symbol_map(sc_df)
        symbols = [symbol_map[int(g)] for g in gene_indices]
        utility_weight_by_domain[domain] = BASE.symbol_positive_incidence(sc_df, symbols)
        gf_sig_by_domain[domain] = BASE.fit_signatures_geneformer(gf_edges[domain], symbols)

        for layer in H71_LAYERS:
            if layer >= sc_layers_seed42[domain].shape[0]:
                continue
            points = sc_layers_seed42[domain][layer, gene_indices, :]
            sc_sig = BASE.fit_signatures_scgpt(
                layer_points=points,
                symbols=symbols,
                random_state=29_710 + domain_index * 100 + layer,
                n_neighbors=10,
            )
            sc_sig_by_domain_layer[(domain, layer)] = sc_sig

    for domain_index, target_domain in enumerate(domains):
        source_domains = [d for d in domains if d != target_domain]
        split_masks = BASE.build_split_masks(sc_edges_seed42[target_domain])

        for layer in H71_LAYERS:
            train_sc_list: list[np.ndarray] = []
            train_gf_list: list[np.ndarray] = []
            train_w_list: list[np.ndarray] = []

            for src_domain in source_domains:
                sc_df = sc_sig_by_domain_layer.get((src_domain, layer))
                gf_df = gf_sig_by_domain.get(src_domain)
                utility_w = utility_weight_by_domain.get(src_domain)
                if sc_df is None or gf_df is None or utility_w is None:
                    continue

                shared = sorted(set(sc_df.index) & set(gf_df.index))
                if len(shared) < 80:
                    continue
                src_sc = sc_df.loc[shared].to_numpy(dtype=float)
                src_gf = gf_df.loc[shared].to_numpy(dtype=float)
                src_w = utility_w.loc[shared].to_numpy(dtype=float)

                train_sc_list.append(src_sc)
                train_gf_list.append(src_gf)
                train_w_list.append(src_w)

            if not train_sc_list or not train_gf_list:
                continue

            train_sc = np.vstack(train_sc_list)
            train_gf = np.vstack(train_gf_list)
            train_w = np.clip(np.concatenate(train_w_list), 0.05, None)
            if min(train_sc.shape[0], train_gf.shape[0]) < 120:
                continue

            sc_mu, sc_sd = BASE.zscore_fit(train_sc)
            gf_mu, gf_sd = BASE.zscore_fit(train_gf)
            train_sc_z = BASE.zscore_apply(train_sc, sc_mu, sc_sd)
            train_gf_z = BASE.zscore_apply(train_gf, gf_mu, gf_sd)

            map_w = BASE.weighted_ridge_map(
                src=train_gf_z,
                dst=train_sc_z,
                weights=train_w,
                l2=0.08,
            )

            try:
                sc_codebook = BASE.fit_codebook(
                    train_sc_z,
                    n_tokens=H71_CODEBOOK_TOKENS,
                    random_state=29_711 + layer,
                )
                gf_codebook = BASE.fit_codebook(
                    train_gf_z,
                    n_tokens=H71_CODEBOOK_TOKENS,
                    random_state=29_712 + layer,
                )
            except RuntimeError:
                continue

            sc_codebook.fit(train_sc_z)
            gf_codebook.fit(train_gf_z)
            n_sc = int(sc_codebook.n_clusters)
            n_gf = int(gf_codebook.n_clusters)

            sc_pos = np.ones((n_sc, n_sc), dtype=float)
            sc_cnt = np.full((n_sc, n_sc), 2.0, dtype=float)
            gf_pos = np.ones((n_gf, n_gf), dtype=float)
            gf_cnt = np.full((n_gf, n_gf), 2.0, dtype=float)

            for src_domain in source_domains:
                sc_df_sig = sc_sig_by_domain_layer.get((src_domain, layer))
                gf_df_sig = gf_sig_by_domain.get(src_domain)
                if sc_df_sig is None or gf_df_sig is None:
                    continue

                shared = sorted(set(sc_df_sig.index) & set(gf_df_sig.index))
                if len(shared) < 80:
                    continue

                sc_vals = BASE.zscore_apply(sc_df_sig.loc[shared].to_numpy(dtype=float), sc_mu, sc_sd)
                gf_vals = BASE.zscore_apply(gf_df_sig.loc[shared].to_numpy(dtype=float), gf_mu, gf_sd)
                sc_tokens = sc_codebook.predict(sc_vals)
                gf_tokens = gf_codebook.predict(gf_vals)

                sc_token_map = {sym: int(tok) for sym, tok in zip(shared, sc_tokens)}
                gf_token_map = {sym: int(tok) for sym, tok in zip(shared, gf_tokens)}

                edge_train = sc_edges_seed42[src_domain].copy()
                sc_pos_add, sc_cnt_add = BASE.token_affinity_from_edges(edge_train, sc_token_map, n_tokens=n_sc)
                gf_pos_add, gf_cnt_add = BASE.token_affinity_from_edges(edge_train, gf_token_map, n_tokens=n_gf)
                sc_pos += sc_pos_add
                sc_cnt += sc_cnt_add
                gf_pos += gf_pos_add
                gf_cnt += gf_cnt_add

            affinity_sc = sc_pos / np.clip(sc_cnt, 1.0, None)
            affinity_gf = gf_pos / np.clip(gf_cnt, 1.0, None)

            sc_tgt_df = sc_sig_by_domain_layer.get((target_domain, layer))
            gf_tgt_df = gf_sig_by_domain.get(target_domain)
            if sc_tgt_df is None or gf_tgt_df is None:
                continue

            shared_tgt = sorted(set(sc_tgt_df.index) & set(gf_tgt_df.index))
            if len(shared_tgt) < 80:
                continue

            sc_tgt_z = BASE.zscore_apply(sc_tgt_df.loc[shared_tgt].to_numpy(dtype=float), sc_mu, sc_sd)
            gf_tgt_z = BASE.zscore_apply(gf_tgt_df.loc[shared_tgt].to_numpy(dtype=float), gf_mu, gf_sd)
            mapped_sc = gf_tgt_z @ map_w
            mapped_tokens = sc_codebook.predict(mapped_sc)
            gf_tgt_tokens = gf_codebook.predict(gf_tgt_z)

            aligned_cosine = BASE.mean_row_cosine(mapped_sc, sc_tgt_z)
            sym_to_pos = {sym: idx for idx, sym in enumerate(shared_tgt)}

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_df = sc_edges_seed42[target_domain].loc[split_mask].copy()
                split_df["source_u"] = split_df["source"].astype(str).str.upper()
                split_df["target_u"] = split_df["target"].astype(str).str.upper()
                keep = split_df["source_u"].isin(sym_to_pos) & split_df["target_u"].isin(sym_to_pos)
                split_df = split_df.loc[keep].copy()
                if split_df["label"].nunique() < 2 or split_df.shape[0] < 300:
                    continue

                src_sym = split_df["source_u"].to_numpy(dtype=str)
                tgt_sym = split_df["target_u"].to_numpy(dtype=str)
                labels = split_df["label"].to_numpy(dtype=int)
                src_idx = np.array([sym_to_pos[s] for s in src_sym], dtype=int)
                tgt_idx = np.array([sym_to_pos[t] for t in tgt_sym], dtype=int)

                transfer_scores = affinity_sc[mapped_tokens[src_idx], mapped_tokens[tgt_idx]]
                baseline_scores = affinity_gf[gf_tgt_tokens[src_idx], gf_tgt_tokens[tgt_idx]]

                auc_transfer = BASE.safe_auc(labels, transfer_scores)
                auc_baseline = BASE.safe_auc(labels, baseline_scores)
                delta_auc = (
                    float(auc_transfer - auc_baseline)
                    if np.isfinite(auc_transfer) and np.isfinite(auc_baseline)
                    else float("nan")
                )

                rng = np.random.default_rng(29_713 + domain_index * 100 + layer * 10 + split_index)
                null_teacher_assign = np.empty(H71_NULL_PERM, dtype=float)
                null_anchor_shuffle = np.empty(H71_NULL_PERM, dtype=float)
                null_signature_destroy = np.empty(H71_NULL_PERM, dtype=float)

                for perm_idx in range(H71_NULL_PERM):
                    # Null 1: teacher signature assignment shuffle before fitting the map.
                    train_sc_perm = train_sc_z[rng.permutation(train_sc_z.shape[0])]
                    map_teacher = BASE.weighted_ridge_map(
                        src=train_gf_z,
                        dst=train_sc_perm,
                        weights=train_w,
                        l2=0.08,
                    )
                    mapped_teacher = gf_tgt_z @ map_teacher
                    tokens_teacher = sc_codebook.predict(mapped_teacher)
                    auc_teacher = BASE.safe_auc(
                        labels,
                        affinity_sc[tokens_teacher[src_idx], tokens_teacher[tgt_idx]],
                    )
                    delta_teacher = (
                        float(auc_teacher - auc_baseline)
                        if np.isfinite(auc_teacher) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_teacher_assign[perm_idx] = delta_teacher
                    null_rows.append(
                        {
                            "null_kind": "random_teacher_signature_assignment",
                            "domain": target_domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_teacher),
                        }
                    )

                    # Null 2: anchor-label shuffle by permuting source anchors relative to targets.
                    perm = rng.permutation(train_gf_z.shape[0])
                    map_anchor = BASE.weighted_ridge_map(
                        src=train_gf_z[perm],
                        dst=train_sc_z,
                        weights=train_w,
                        l2=0.08,
                    )
                    mapped_anchor = gf_tgt_z @ map_anchor
                    tokens_anchor = sc_codebook.predict(mapped_anchor)
                    auc_anchor = BASE.safe_auc(
                        labels,
                        affinity_sc[tokens_anchor[src_idx], tokens_anchor[tgt_idx]],
                    )
                    delta_anchor = (
                        float(auc_anchor - auc_baseline)
                        if np.isfinite(auc_anchor) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_anchor_shuffle[perm_idx] = delta_anchor
                    null_rows.append(
                        {
                            "null_kind": "anchor_label_shuffle",
                            "domain": target_domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_anchor),
                        }
                    )

                    # Null 3: destroy mapped signatures at inference by token permutation.
                    token_perm = rng.permutation(mapped_tokens.shape[0])
                    tokens_destroy = mapped_tokens[token_perm]
                    auc_destroy = BASE.safe_auc(
                        labels,
                        affinity_sc[tokens_destroy[src_idx], tokens_destroy[tgt_idx]],
                    )
                    delta_destroy = (
                        float(auc_destroy - auc_baseline)
                        if np.isfinite(auc_destroy) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_signature_destroy[perm_idx] = delta_destroy
                    null_rows.append(
                        {
                            "null_kind": "signature_destroy_permutation",
                            "domain": target_domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_destroy),
                        }
                    )

                p_teacher = BASE.empirical_upper_tail_p(delta_auc, null_teacher_assign)
                p_anchor = BASE.empirical_upper_tail_p(delta_auc, null_anchor_shuffle)
                p_destroy = BASE.empirical_upper_tail_p(delta_auc, null_signature_destroy)
                p_best = np.nanmin(np.array([p_teacher, p_anchor, p_destroy], dtype=float))

                all_null = np.concatenate([null_teacher_assign, null_anchor_shuffle, null_signature_destroy])
                all_null = all_null[np.isfinite(all_null)]
                null_q95 = float(np.quantile(all_null, 0.95)) if all_null.size else float("nan")
                null_gap = (
                    float(delta_auc - null_q95)
                    if np.isfinite(delta_auc) and np.isfinite(null_q95)
                    else float("nan")
                )

                by_row.append(
                    {
                        "domain": target_domain,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "train_domains": "+".join(sorted(source_domains)),
                        "n_train_genes": int(train_sc.shape[0]),
                        "n_shared_target_symbols": int(len(shared_tgt)),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "n_tokens_sc": int(n_sc),
                        "n_tokens_gf": int(n_gf),
                        "auc_transfer": float(auc_transfer),
                        "auc_baseline": float(auc_baseline),
                        "delta_auc_transfer_minus_baseline": float(delta_auc),
                        "null_q95": float(null_q95),
                        "null_gap_q95": float(null_gap),
                        "mapped_to_sc_cosine": float(aligned_cosine),
                        "p_teacher_assign_upper": float(p_teacher),
                        "p_anchor_shuffle_upper": float(p_anchor),
                        "p_signature_destroy_upper": float(p_destroy),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(by_row)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h71_topology_signature_distill_by_domain_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h71_topology_signature_distill_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_auc_transfer": float(group["auc_transfer"].mean()),
                    "mean_auc_baseline": float(group["auc_baseline"].mean()),
                    "mean_delta_auc_transfer_minus_baseline": float(
                        group["delta_auc_transfer_minus_baseline"].mean()
                    ),
                    "mean_null_gap_q95": float(group["null_gap_q95"].mean()),
                    "mean_mapped_to_sc_cosine": float(group["mapped_to_sc_cosine"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_transfer_minus_baseline"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(
                        BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))
                    ),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h71_topology_signature_distill_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_transfer_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "mean_null_gap_q95": float(by_row_df["null_gap_q95"].mean()) if not by_row_df.empty else float("nan"),
        "mean_alignment_cosine": float(by_row_df["mapped_to_sc_cosine"].mean())
        if not by_row_df.empty
        else float("nan"),
        "artifact_paths": {
            "by_domain_layer_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def build_edge_trajectory_features(
    layer_embeddings: np.ndarray,
    edge_gene_indices: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    support_dir: np.ndarray,
    random_state_base: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Build per-edge trajectory vectors from multiple layers.

    The trajectory packs two channels per layer: a geodesic-support baseline channel and
    a triangle-defect channel. This lets motif clustering track how edge geometry evolves
    over depth rather than relying on any single layer snapshot.
    """
    available_layers = [layer for layer in H72_LAYERS if layer < layer_embeddings.shape[0]]
    if len(available_layers) < 2:
        raise RuntimeError("Need at least two available layers for trajectory motif scan.")

    traj_blocks: list[np.ndarray] = []
    baseline_layers: list[np.ndarray] = []
    geodesic_ref = np.zeros(source_local.size, dtype=float)

    for layer_offset, layer in enumerate(available_layers):
        points = layer_embeddings[layer, edge_gene_indices, :]
        points_pca = BASE.reduce_points(
            points,
            n_components=20,
            random_state=random_state_base + 100 * layer_offset + layer,
        )
        geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H72_NEIGHBORS)

        edge_geodesic = geodesic[source_local, target_local]
        edge_support = support_dir[source_local, target_local]
        edge_margin = np.abs(support_dir[source_local, target_local] - support_dir[target_local, source_local])

        feature_bundle = BASE.multiscale_triangle_defect_features(
            geodesic=geodesic,
            source_local=source_local,
            target_local=target_local,
            k_values=H72_TRIANGLE_K,
        )

        baseline_component = (
            BASE.zscore(-edge_geodesic) + 0.75 * BASE.zscore(edge_support) + 0.35 * BASE.zscore(edge_margin)
        )
        defect_component = (
            0.60 * BASE.zscore(-feature_bundle["median_mean"])
            + 0.40 * BASE.zscore(feature_bundle["close_frac_mean"])
        )

        traj_blocks.append(np.column_stack([BASE.zscore(baseline_component), BASE.zscore(defect_component)]))
        baseline_layers.append(baseline_component)
        geodesic_ref = edge_geodesic

    trajectory = np.concatenate(traj_blocks, axis=1)
    baseline_reference = np.mean(np.column_stack(baseline_layers), axis=1)
    return trajectory, baseline_reference, geodesic_ref, available_layers


def run_h72_edge_trajectory_motif_scan(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map.get(H72_SEED_TAG)
        if run_dir is None:
            continue

        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H72_GENE_CAP))
            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
            ].copy()
            if split_edges["label"].nunique() < 2 or split_edges.shape[0] < 600:
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
            source_local = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
            target_local = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
            labels = split_edges["label"].to_numpy(dtype=int)

            try:
                trajectory, baseline_reference, edge_geodesic_ref, used_layers = build_edge_trajectory_features(
                    layer_embeddings=layer_embeddings,
                    edge_gene_indices=edge_gene_indices,
                    source_local=source_local,
                    target_local=target_local,
                    support_dir=support_dir,
                    random_state_base=29_720 + domain_index * 100 + split_index * 10,
                )
            except RuntimeError:
                continue

            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=29_721 + domain_index * 11 + split_index)
            train_idx, test_idx = next(splitter.split(trajectory, labels))

            train_x = trajectory[train_idx]
            test_x = trajectory[test_idx]
            train_y = labels[train_idx].astype(int)
            test_y = labels[test_idx].astype(int)
            baseline_test = baseline_reference[test_idx]

            feat_mu, feat_sd = BASE.zscore_fit(train_x)
            train_x_z = BASE.zscore_apply(train_x, feat_mu, feat_sd)
            test_x_z = BASE.zscore_apply(test_x, feat_mu, feat_sd)

            n_clusters = int(min(H72_MOTIF_K, max(3, train_x_z.shape[0] // 240)))
            motif_model = KMeans(n_clusters=n_clusters, n_init=15, random_state=29_722 + domain_index * 10 + split_index)
            motif_model.fit(train_x_z)
            motif_train = motif_model.labels_.astype(int)
            motif_test = motif_model.predict(test_x_z).astype(int)

            _, _, motif_rates = motif_rate_table(motif_train, train_y, n_clusters=n_clusters)
            global_rate = float(np.mean(train_y))
            best_motif_enrichment = float(np.max(motif_rates - global_rate))
            motif_signal = safe_logit(motif_rates)

            auc_baseline = BASE.safe_auc(test_y, baseline_test)
            motif_augmented_score = baseline_test + 0.35 * motif_signal[motif_test]
            auc_motif_augmented = BASE.safe_auc(test_y, motif_augmented_score)
            delta_auc = (
                float(auc_motif_augmented - auc_baseline)
                if np.isfinite(auc_motif_augmented) and np.isfinite(auc_baseline)
                else float("nan")
            )

            rng = np.random.default_rng(29_723 + domain_index * 100 + split_index)
            test_bins = BASE.degree_bins(edge_geodesic_ref[test_idx], max_bins=6)
            train_bins = BASE.degree_bins(edge_geodesic_ref[train_idx], max_bins=6)
            n_layers = len(used_layers)

            null_layer_order = np.empty(H72_NULL_PERM, dtype=float)
            null_label_shuffle = np.empty(H72_NULL_PERM, dtype=float)
            null_enrichment = np.empty(H72_NULL_PERM, dtype=float)

            for perm_idx in range(H72_NULL_PERM):
                # Null A: destroy trajectory order by permuting layer blocks before motif discovery.
                perm_layers = rng.permutation(n_layers)
                perm_cols: list[int] = []
                for idx in perm_layers:
                    perm_cols.extend([2 * int(idx), 2 * int(idx) + 1])

                train_perm = train_x_z[:, perm_cols]
                test_perm = test_x_z[:, perm_cols]
                motif_perm_model = KMeans(
                    n_clusters=n_clusters,
                    n_init=10,
                    random_state=29_724 + domain_index * 100 + split_index * 10 + perm_idx,
                )
                motif_perm_model.fit(train_perm)
                motif_train_perm = motif_perm_model.labels_.astype(int)
                motif_test_perm = motif_perm_model.predict(test_perm).astype(int)
                _, _, rates_perm = motif_rate_table(motif_train_perm, train_y, n_clusters=n_clusters)
                signal_perm = safe_logit(rates_perm)
                aug_perm = baseline_test + 0.35 * signal_perm[motif_test_perm]
                auc_aug_perm = BASE.safe_auc(test_y, aug_perm)
                delta_perm = (
                    float(auc_aug_perm - auc_baseline)
                    if np.isfinite(auc_aug_perm) and np.isfinite(auc_baseline)
                    else float("nan")
                )
                null_layer_order[perm_idx] = delta_perm
                null_rows.append(
                    {
                        "null_kind": "layer_order_permutation",
                        "domain": domain,
                        "split_regime": split_regime,
                        "perm_idx": int(perm_idx),
                        "null_delta_auc": float(delta_perm),
                    }
                )

                # Null B: shuffle motif class assignments in degree bins on the held-out set.
                motif_test_shuffle = BASE.stratified_shuffle(motif_test, test_bins, rng).astype(int)
                aug_shuffle = baseline_test + 0.35 * motif_signal[motif_test_shuffle]
                auc_aug_shuffle = BASE.safe_auc(test_y, aug_shuffle)
                delta_shuffle = (
                    float(auc_aug_shuffle - auc_baseline)
                    if np.isfinite(auc_aug_shuffle) and np.isfinite(auc_baseline)
                    else float("nan")
                )
                null_label_shuffle[perm_idx] = delta_shuffle
                null_rows.append(
                    {
                        "null_kind": "motif_label_shuffle_within_degree_bins",
                        "domain": domain,
                        "split_regime": split_regime,
                        "perm_idx": int(perm_idx),
                        "null_delta_auc": float(delta_shuffle),
                    }
                )

                motif_train_shuffle = BASE.stratified_shuffle(motif_train, train_bins, rng).astype(int)
                _, _, rates_shuffle = motif_rate_table(motif_train_shuffle, train_y, n_clusters=n_clusters)
                null_enrichment[perm_idx] = float(np.max(rates_shuffle - global_rate))

            p_layer = BASE.empirical_upper_tail_p(delta_auc, null_layer_order)
            p_shuffle = BASE.empirical_upper_tail_p(delta_auc, null_label_shuffle)
            p_best = np.nanmin(np.array([p_layer, p_shuffle], dtype=float))
            p_enrichment = BASE.empirical_upper_tail_p(best_motif_enrichment, null_enrichment)

            all_null = np.concatenate([null_layer_order, null_label_shuffle])
            all_null = all_null[np.isfinite(all_null)]
            null_q95 = float(np.quantile(all_null, 0.95)) if all_null.size else float("nan")
            null_gap = float(delta_auc - null_q95) if np.isfinite(delta_auc) and np.isfinite(null_q95) else float("nan")

            rows.append(
                {
                    "domain": domain,
                    "seed_tag": H72_SEED_TAG,
                    "split_regime": split_regime,
                    "layers_used": ",".join(str(x) for x in used_layers),
                    "n_edges_eval": int(labels.size),
                    "n_train_edges": int(train_idx.size),
                    "n_test_edges": int(test_idx.size),
                    "n_positive_eval": int(labels.sum()),
                    "n_motifs": int(n_clusters),
                    "auc_motif_augmented": float(auc_motif_augmented),
                    "auc_baseline": float(auc_baseline),
                    "delta_auc_motif_minus_baseline": float(delta_auc),
                    "best_motif_enrichment": float(best_motif_enrichment),
                    "p_enrichment_upper": float(p_enrichment),
                    "null_q95": float(null_q95),
                    "null_gap_q95": float(null_gap),
                    "p_layer_order_upper": float(p_layer),
                    "p_label_shuffle_upper": float(p_shuffle),
                    "p_best_upper": float(p_best),
                }
            )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime"])
    by_row_path = ITER_DIR / "h72_edge_trajectory_motif_by_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h72_edge_trajectory_motif_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_auc_motif_augmented": float(group["auc_motif_augmented"].mean()),
                    "mean_auc_baseline": float(group["auc_baseline"].mean()),
                    "mean_delta_auc_motif_minus_baseline": float(
                        group["delta_auc_motif_minus_baseline"].mean()
                    ),
                    "mean_best_motif_enrichment": float(group["best_motif_enrichment"].mean()),
                    "mean_null_gap_q95": float(group["null_gap_q95"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_motif_minus_baseline"] > 0.0).mean()),
                    "fraction_enrichment_p_lt_0_05": float((group["p_enrichment_upper"] < 0.05).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(
                        BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))
                    ),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h72_edge_trajectory_motif_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_motif_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "mean_enrichment": float(by_row_df["best_motif_enrichment"].mean()) if not by_row_df.empty else float("nan"),
        "artifact_paths": {
            "by_domain_split": str(by_row_path),
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

    h70_summary = run_h70_triangle_defect_hard_null(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h71_summary = run_h71_topology_signature_distillation()
    h72_summary = run_h72_edge_trajectory_motif_scan(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    summary = {
        "iteration": "iter_0029",
        "h70": h70_summary,
        "h71": h71_summary,
        "h72": h72_summary,
        "environment": {
            "python_env": "subproject40-topology",
            "string_cache_used": str(BASE.STRING_CACHE_PATH),
        },
    }
    summary_path = ITER_DIR / "iter0029_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
