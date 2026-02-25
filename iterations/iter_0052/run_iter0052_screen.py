from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ITER_DIR = Path("iterations/iter_0052")
ITER_DIR.mkdir(parents=True, exist_ok=True)


# H139 (carry-over refinement): sectional anisotropy robustness expansion.
H139_SEEDS = ["seed42_main", "seed43", "seed44"]
H139_LAYER = 11
H139_SPLITS = ("source_disjoint", "target_disjoint", "dual_axis_disjoint")
H139_GENE_CAP = 170
H139_GENE_CAP_LUNG_DUAL = 240
H139_MIN_GENE_NODES = 120
H139_MIN_GENE_NODES_LUNG_DUAL = 90
H139_EDGE_SAMPLE = 220
H139_EDGE_SAMPLE_LUNG_DUAL = 190
H139_NEIGHBORS = 12
H139_CV_SPLITS = 3
H139_NULL_PERM = 8

# H140 (new family): neighborhood-size stability check.
H140_SEED = "seed42_main"
H140_LAYER = 11
H140_SPLITS = ("source_disjoint", "target_disjoint", "dual_axis_disjoint")
H140_NEIGHBOR_GRID = [8, 12, 16]
H140_GENE_CAP = 170
H140_GENE_CAP_LUNG_DUAL = 240
H140_MIN_GENE_NODES = 120
H140_MIN_GENE_NODES_LUNG_DUAL = 90
H140_EDGE_SAMPLE = 220
H140_EDGE_SAMPLE_LUNG_DUAL = 190
H140_CV_SPLITS = 3


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base_iter0052")
PREV = load_module(Path("iterations/iter_0047/run_iter0047_screen.py"), "iter0047_prev_iter0052")
ITER51 = load_module(Path("iterations/iter_0051/run_iter0051_screen.py"), "iter0051_prev_iter0052")


def ensure_required_inputs() -> None:
    required = [
        BASE.DOROTHEA_PATH,
        BASE.GENE2GO_PATH,
        BASE.OMNIPATH_INTERACTIONS_PATH,
        BASE.STRING_CACHE_PATH,
        PREV.TRRUST_PATH,
    ]

    for domain, run_map in BASE.SCGPT_RUNS_BY_DOMAIN.items():
        for seed_tag in sorted(set(H139_SEEDS + [H140_SEED])):
            run_dir = run_map[seed_tag]
            required.append(run_dir / "cycle1_edge_dataset.tsv")
            required.append(run_dir / "layer_gene_embeddings.npy")
        required.append(BASE.GENEFORMER_EDGE_BY_DOMAIN[domain])

    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def finite_q95(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, 0.95))


def finite_fisher(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(BASE.safe_fisher_p(arr))


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


def summarize_by_domain(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for domain, group in df.groupby("domain", sort=True):
        rows.append(
            {
                "domain": str(domain),
                "n_rows": int(group.shape[0]),
                "mean_delta_vs_h70": float(group["delta_vs_h70"].mean()),
                "mean_null_gap_q95": float(group["null_gap_q95"].mean()),
                "fraction_delta_positive": float((group["delta_vs_h70"] > 0.0).mean()),
                "fraction_null_gap_positive": float((group["null_gap_q95"] > 0.0).mean()),
                "combined_fisher_p_best": finite_fisher(group["p_best_upper"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows).sort_values("domain")


DATA_CACHE: dict[tuple[str, str], tuple[pd.DataFrame, np.ndarray]] = {}


def load_run_payload(domain: str, seed_tag: str) -> tuple[pd.DataFrame, np.ndarray]:
    key = (domain, seed_tag)
    if key in DATA_CACHE:
        return DATA_CACHE[key]

    run_dir = BASE.SCGPT_RUNS_BY_DOMAIN[domain][seed_tag]
    edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
    layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
    DATA_CACHE[key] = (edge_df, layer_embeddings)
    return edge_df, layer_embeddings


def prepare_split_bundle(
    *,
    domain: str,
    seed_tag: str,
    split_regime: str,
    gene_cap: int,
    min_gene_nodes: int,
    edge_sample: int,
    rng_seed: int,
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object] | None:
    edge_df, layer_embeddings = load_run_payload(domain=domain, seed_tag=seed_tag)
    split_masks = PREV.build_split_masks_plus(edge_df)
    split_mask = split_masks.get(split_regime)
    if split_mask is None:
        return None

    split_edges = edge_df.loc[split_mask].copy()
    if split_edges["label"].nunique() < 2:
        return None

    top_genes = set(BASE.select_top_genes(split_edges, gene_cap=gene_cap))
    split_edges = split_edges.loc[
        split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
    ].copy()
    if split_edges["label"].nunique() < 2:
        return None

    edge_gene_indices, gene_to_local, _, support_dir = PREV.build_symbol_resources(
        split_edges=split_edges,
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    if edge_gene_indices.size < min_gene_nodes:
        return None

    source_local_all = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
    target_local_all = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
    labels_all = split_edges["label"].to_numpy(dtype=int)

    rng = np.random.default_rng(rng_seed)
    sample_idx = PREV.stratified_index_sample(labels_all, max_n=edge_sample, rng=rng)
    if sample_idx.size < 120:
        return None

    return {
        "layer_embeddings": layer_embeddings,
        "edge_gene_indices": edge_gene_indices,
        "support_dir": support_dir,
        "source_local": source_local_all[sample_idx],
        "target_local": target_local_all[sample_idx],
        "labels": labels_all[sample_idx],
    }


def compute_sectional_packet(
    *,
    bundle: dict[str, object],
    layer: int,
    n_neighbors: int,
    pca_random_state: int,
) -> dict[str, np.ndarray] | None:
    layer_embeddings = bundle["layer_embeddings"]
    edge_gene_indices = np.asarray(bundle["edge_gene_indices"], dtype=int)
    support_dir = np.asarray(bundle["support_dir"], dtype=float)
    source_local = np.asarray(bundle["source_local"], dtype=int)
    target_local = np.asarray(bundle["target_local"], dtype=int)
    labels = np.asarray(bundle["labels"], dtype=int)

    if layer >= layer_embeddings.shape[0]:
        return None

    points = layer_embeddings[layer, edge_gene_indices, :]
    points_pca = BASE.reduce_points(points, n_components=20, random_state=pca_random_state)

    geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=n_neighbors)
    geodesic_w = PREV.confidence_weighted_geodesic(geodesic, support_dir)

    h70 = PREV.compute_h70_scores(
        geodesic=geodesic_w,
        support_dir=support_dir,
        source_local=source_local,
        target_local=target_local,
        triangle_k=[8, 12, 16],
    )

    node_geo = ITER51.local_sectional_geometry(points_pca, n_neighbors=n_neighbors)
    sec_feat = ITER51.sectional_edge_features(
        points=points_pca,
        source_local=source_local,
        target_local=target_local,
        support_dir=support_dir,
        node_geo=node_geo,
    )

    edge_len = geodesic_w[source_local, target_local]
    len_bins = PREV.quantile_bins(edge_len, n_bins=6)
    degree_sum = PREV.edge_degree_sum(points_pca, n_neighbors, source_local, target_local)
    edge_strata = PREV.build_edge_strata(edge_len, degree_sum, max_len_bins=6, max_deg_bins=4)

    return {
        "h70": np.asarray(h70, dtype=float),
        "sec_feat": np.asarray(sec_feat, dtype=float),
        "labels": labels,
        "len_bins": np.asarray(len_bins, dtype=int),
        "edge_strata": np.asarray(edge_strata, dtype=int),
    }


def evaluate_delta(
    *,
    h70: np.ndarray,
    sec_feat: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    n_splits: int,
) -> tuple[float, float, float]:
    auc_h70 = BASE.safe_auc(labels, h70)
    auc_model = PREV.cv_auc_logit(
        np.column_stack([h70, sec_feat]),
        labels,
        random_state=random_state,
        n_splits=n_splits,
    )
    delta = float(auc_model - auc_h70) if np.isfinite(auc_model) and np.isfinite(auc_h70) else float("nan")
    return float(auc_h70), float(auc_model), delta


def run_h139_sectional_seed_robustness(
    *,
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_idx, domain in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.keys()):
        for seed_idx, seed_tag in enumerate(H139_SEEDS):
            for split_idx, split_regime in enumerate(H139_SPLITS):
                is_lung_dual = domain == "lung" and split_regime == "dual_axis_disjoint"
                gene_cap = H139_GENE_CAP_LUNG_DUAL if is_lung_dual else H139_GENE_CAP
                min_gene_nodes = H139_MIN_GENE_NODES_LUNG_DUAL if is_lung_dual else H139_MIN_GENE_NODES
                edge_sample = H139_EDGE_SAMPLE_LUNG_DUAL if is_lung_dual else H139_EDGE_SAMPLE

                bundle = prepare_split_bundle(
                    domain=domain,
                    seed_tag=seed_tag,
                    split_regime=split_regime,
                    gene_cap=gene_cap,
                    min_gene_nodes=min_gene_nodes,
                    edge_sample=edge_sample,
                    rng_seed=52_139 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100,
                    dorothea_map=dorothea_map,
                    omnipath_pairs=omnipath_pairs,
                    gene2go_upper=gene2go_upper,
                    string_map=string_map,
                )
                if bundle is None:
                    continue

                packet = compute_sectional_packet(
                    bundle=bundle,
                    layer=H139_LAYER,
                    n_neighbors=H139_NEIGHBORS,
                    pca_random_state=52_200 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100,
                )
                if packet is None:
                    continue

                h70 = packet["h70"]
                sec_feat = packet["sec_feat"]
                labels = packet["labels"]
                len_bins = packet["len_bins"]
                edge_strata = packet["edge_strata"]

                auc_h70, auc_model, delta = evaluate_delta(
                    h70=h70,
                    sec_feat=sec_feat,
                    labels=labels,
                    random_state=52_300 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100,
                    n_splits=H139_CV_SPLITS,
                )

                rng = np.random.default_rng(52_400 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100)
                null_swap = np.empty(H139_NULL_PERM, dtype=float)
                null_feat_shuffle = np.empty(H139_NULL_PERM, dtype=float)
                null_label = np.empty(H139_NULL_PERM, dtype=float)

                for perm_idx in range(H139_NULL_PERM):
                    feat_swap = PREV.permute_rows_within_strata(
                        ITER51.swapped_sectional_features(sec_feat),
                        strata=len_bins,
                        rng=rng,
                    )
                    auc_swap = PREV.cv_auc_logit(
                        np.column_stack([h70, feat_swap]),
                        labels,
                        random_state=52_500
                        + domain_idx * 100_000
                        + seed_idx * 10_000
                        + split_idx * 1000
                        + perm_idx,
                        n_splits=H139_CV_SPLITS,
                    )
                    null_swap[perm_idx] = (
                        float(auc_swap - auc_h70) if np.isfinite(auc_swap) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H139",
                            "null_kind": "endpoint_swap_within_distance_bins",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H139_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_swap[perm_idx]),
                        }
                    )

                    feat_shuffle = PREV.permute_rows_within_strata(
                        sec_feat,
                        strata=len_bins,
                        rng=rng,
                    )
                    auc_shuffle = PREV.cv_auc_logit(
                        np.column_stack([h70, feat_shuffle]),
                        labels,
                        random_state=52_600
                        + domain_idx * 100_000
                        + seed_idx * 10_000
                        + split_idx * 1000
                        + perm_idx,
                        n_splits=H139_CV_SPLITS,
                    )
                    null_feat_shuffle[perm_idx] = (
                        float(auc_shuffle - auc_h70)
                        if np.isfinite(auc_shuffle) and np.isfinite(auc_h70)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H139",
                            "null_kind": "sectional_row_shuffle",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H139_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_feat_shuffle[perm_idx]),
                        }
                    )

                    y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                    auc_lp = PREV.cv_auc_logit(
                        np.column_stack([h70, sec_feat]),
                        y_perm,
                        random_state=52_700
                        + domain_idx * 100_000
                        + seed_idx * 10_000
                        + split_idx * 1000
                        + perm_idx,
                        n_splits=H139_CV_SPLITS,
                    )
                    auc_h70_lp = BASE.safe_auc(y_perm, h70)
                    null_label[perm_idx] = (
                        float(auc_lp - auc_h70_lp)
                        if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H139",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H139_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_swap, null_feat_shuffle, null_label])
                q95 = finite_q95(all_null)
                p_swap = BASE.empirical_upper_tail_p(delta, null_swap)
                p_shuffle = BASE.empirical_upper_tail_p(delta, null_feat_shuffle)
                p_label = BASE.empirical_upper_tail_p(delta, null_label)
                p_best = float(np.nanmin(np.asarray([p_swap, p_shuffle, p_label], dtype=float)))

                rows.append(
                    {
                        "hypothesis_id": "H139",
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(H139_LAYER),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70": float(auc_h70),
                        "auc_sectional_model": float(auc_model),
                        "delta_vs_h70": float(delta),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95": float(delta - q95),
                        "p_endpoint_swap_upper": float(p_swap),
                        "p_sectional_shuffle_upper": float(p_shuffle),
                        "p_label_upper": float(p_label),
                        "p_best_upper": float(p_best),
                        "mean_anisotropy_gap": float(np.mean(sec_feat[:, 2])),
                        "mean_pair_alignment": float(np.mean(sec_feat[:, 9])),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["seed_tag", "domain", "split_regime"])
    by_row_path = ITER_DIR / "h139_sectional_seed_robustness_by_seed_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "seed_tag", "domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h139_sectional_seed_robustness_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    split_summary_df = summarize_by_domain_split(by_row_df)
    split_summary_path = ITER_DIR / "h139_sectional_seed_robustness_domain_split_summary.csv"
    split_summary_df.to_csv(split_summary_path, index=False)

    domain_summary_df = summarize_by_domain(by_row_df)
    domain_summary_path = ITER_DIR / "h139_sectional_seed_robustness_domain_summary.csv"
    domain_summary_df.to_csv(domain_summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_vs_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((split_summary_df["mean_delta_vs_h70"] > 0.0).sum()) if not split_summary_df.empty else 0,
        "positive_null_gap_domain_splits": int((split_summary_df["mean_null_gap_q95"] > 0.0).sum())
        if not split_summary_df.empty
        else 0,
        "positive_rows_strict": int((by_row_df["null_gap_q95"] > 0.0).sum()) if not by_row_df.empty else 0,
        "artifact_paths": {
            "by_seed_domain_split": str(by_row_path),
            "domain_split_summary": str(split_summary_path),
            "domain_summary": str(domain_summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h140_neighborhood_scaling(
    *,
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    by_k_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    for domain_idx, domain in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.keys()):
        for split_idx, split_regime in enumerate(H140_SPLITS):
            is_lung_dual = domain == "lung" and split_regime == "dual_axis_disjoint"
            gene_cap = H140_GENE_CAP_LUNG_DUAL if is_lung_dual else H140_GENE_CAP
            min_gene_nodes = H140_MIN_GENE_NODES_LUNG_DUAL if is_lung_dual else H140_MIN_GENE_NODES
            edge_sample = H140_EDGE_SAMPLE_LUNG_DUAL if is_lung_dual else H140_EDGE_SAMPLE

            bundle = prepare_split_bundle(
                domain=domain,
                seed_tag=H140_SEED,
                split_regime=split_regime,
                gene_cap=gene_cap,
                min_gene_nodes=min_gene_nodes,
                edge_sample=edge_sample,
                rng_seed=52_800 + domain_idx * 10_000 + split_idx * 100,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )
            if bundle is None:
                continue

            delta_real: list[float] = []
            delta_swap: list[float] = []

            for k_idx, neighbors in enumerate(H140_NEIGHBOR_GRID):
                packet = compute_sectional_packet(
                    bundle=bundle,
                    layer=H140_LAYER,
                    n_neighbors=neighbors,
                    pca_random_state=52_900 + domain_idx * 10_000 + split_idx * 100 + k_idx,
                )
                if packet is None:
                    continue

                h70 = packet["h70"]
                sec_feat = packet["sec_feat"]
                labels = packet["labels"]
                len_bins = packet["len_bins"]

                auc_h70, auc_model, delta = evaluate_delta(
                    h70=h70,
                    sec_feat=sec_feat,
                    labels=labels,
                    random_state=53_000 + domain_idx * 10_000 + split_idx * 100 + k_idx,
                    n_splits=H140_CV_SPLITS,
                )

                rng = np.random.default_rng(53_100 + domain_idx * 10_000 + split_idx * 100 + k_idx)
                feat_swap = PREV.permute_rows_within_strata(
                    ITER51.swapped_sectional_features(sec_feat),
                    strata=len_bins,
                    rng=rng,
                )
                auc_swap = PREV.cv_auc_logit(
                    np.column_stack([h70, feat_swap]),
                    labels,
                    random_state=53_200 + domain_idx * 10_000 + split_idx * 100 + k_idx,
                    n_splits=H140_CV_SPLITS,
                )
                delta_swap_k = (
                    float(auc_swap - auc_h70) if np.isfinite(auc_swap) and np.isfinite(auc_h70) else float("nan")
                )

                delta_real.append(float(delta))
                delta_swap.append(float(delta_swap_k))

                by_k_rows.append(
                    {
                        "hypothesis_id": "H140",
                        "domain": domain,
                        "seed_tag": H140_SEED,
                        "split_regime": split_regime,
                        "layer": int(H140_LAYER),
                        "neighbors": int(neighbors),
                        "n_edges_eval": int(labels.size),
                        "auc_h70": float(auc_h70),
                        "auc_model": float(auc_model),
                        "delta_vs_h70": float(delta),
                        "delta_swap_null": float(delta_swap_k),
                        "delta_gain_vs_swap": float(delta - delta_swap_k),
                    }
                )

            if not delta_real:
                continue

            arr_real = np.asarray(delta_real, dtype=float)
            arr_swap = np.asarray(delta_swap, dtype=float)
            split_rows.append(
                {
                    "hypothesis_id": "H140",
                    "domain": domain,
                    "seed_tag": H140_SEED,
                    "split_regime": split_regime,
                    "layer": int(H140_LAYER),
                    "n_neighbors_tested": int(arr_real.size),
                    "mean_delta_vs_h70": float(np.nanmean(arr_real)),
                    "mean_delta_swap_null": float(np.nanmean(arr_swap)),
                    "mean_delta_gain_vs_swap": float(np.nanmean(arr_real - arr_swap)),
                    "stability_span_delta": float(np.nanmax(arr_real) - np.nanmin(arr_real)),
                    "fraction_delta_positive": float(np.mean(arr_real > 0.0)),
                    "fraction_gain_positive": float(np.mean((arr_real - arr_swap) > 0.0)),
                }
            )

    by_k_df = pd.DataFrame(by_k_rows)
    if not by_k_df.empty:
        by_k_df = by_k_df.sort_values(["domain", "split_regime", "neighbors"])
    by_k_path = ITER_DIR / "h140_neighborhood_scaling_by_domain_split_k.csv"
    by_k_df.to_csv(by_k_path, index=False)

    split_summary_df = pd.DataFrame(split_rows)
    if not split_summary_df.empty:
        split_summary_df = split_summary_df.sort_values(["domain", "split_regime"])
    split_summary_path = ITER_DIR / "h140_neighborhood_scaling_domain_split_summary.csv"
    split_summary_df.to_csv(split_summary_path, index=False)

    domain_summary_rows: list[dict[str, object]] = []
    if not split_summary_df.empty:
        for domain, group in split_summary_df.groupby("domain", sort=True):
            domain_summary_rows.append(
                {
                    "domain": str(domain),
                    "n_rows": int(group.shape[0]),
                    "mean_delta_vs_h70": float(group["mean_delta_vs_h70"].mean()),
                    "mean_delta_gain_vs_swap": float(group["mean_delta_gain_vs_swap"].mean()),
                    "mean_stability_span_delta": float(group["stability_span_delta"].mean()),
                    "fraction_gain_positive": float((group["mean_delta_gain_vs_swap"] > 0.0).mean()),
                }
            )
    domain_summary_df = pd.DataFrame(domain_summary_rows)
    if not domain_summary_df.empty:
        domain_summary_df = domain_summary_df.sort_values("domain")
    domain_summary_path = ITER_DIR / "h140_neighborhood_scaling_domain_summary.csv"
    domain_summary_df.to_csv(domain_summary_path, index=False)

    return {
        "rows_tested": int(by_k_df.shape[0]),
        "mean_delta_auc": float(by_k_df["delta_vs_h70"].mean()) if not by_k_df.empty else float("nan"),
        "mean_gain_vs_swap": float(by_k_df["delta_gain_vs_swap"].mean()) if not by_k_df.empty else float("nan"),
        "positive_split_gains": int((split_summary_df["mean_delta_gain_vs_swap"] > 0.0).sum())
        if not split_summary_df.empty
        else 0,
        "artifact_paths": {
            "by_domain_split_k": str(by_k_path),
            "domain_split_summary": str(split_summary_path),
            "domain_summary": str(domain_summary_path),
        },
    }


def run_h141_strict_null_sensitivity(*, h139_row_path: Path, h139_null_path: Path) -> dict[str, object]:
    if not h139_row_path.exists() or not h139_null_path.exists():
        raise FileNotFoundError("H141 requires H139 artifacts to exist before execution")

    row_df = pd.read_csv(h139_row_path)
    null_df = pd.read_csv(h139_null_path)

    if row_df.empty or null_df.empty:
        empty_paths = {
            "row_summary": ITER_DIR / "h141_strict_null_sensitivity_row_summary.csv",
            "domain_split_summary": ITER_DIR / "h141_strict_null_sensitivity_domain_split_summary.csv",
            "domain_summary": ITER_DIR / "h141_strict_null_sensitivity_domain_summary.csv",
            "nullkind_summary": ITER_DIR / "h141_strict_null_sensitivity_nullkind_summary.csv",
        }
        for path in empty_paths.values():
            pd.DataFrame().to_csv(path, index=False)
        return {
            "rows_tested": 0,
            "strict_positive_rows": 0,
            "artifact_paths": {k: str(v) for k, v in empty_paths.items()},
        }

    key_cols = ["domain", "seed_tag", "split_regime", "layer"]
    q95_df = (
        null_df.groupby(key_cols + ["null_kind"], as_index=False)["null_value"]
        .quantile(0.95)
        .rename(columns={"null_value": "q95_null"})
    )
    mean_df = (
        null_df.groupby(key_cols + ["null_kind"], as_index=False)["null_value"]
        .mean()
        .rename(columns={"null_value": "mean_null"})
    )
    q95_df = q95_df.merge(mean_df, on=key_cols + ["null_kind"], how="left")

    q95_wide = q95_df.pivot(index=key_cols, columns="null_kind", values="q95_null")
    q95_wide.columns = [f"q95_{str(col)}" for col in q95_wide.columns]
    q95_null_cols = list(q95_wide.columns)

    mean_wide = q95_df.pivot(index=key_cols, columns="null_kind", values="mean_null")
    mean_wide.columns = [f"mean_{str(col)}" for col in mean_wide.columns]

    row_summary = row_df.merge(q95_wide.reset_index(), on=key_cols, how="left")
    row_summary = row_summary.merge(mean_wide.reset_index(), on=key_cols, how="left")

    for col in q95_null_cols:
        suffix = col.replace("q95_", "")
        row_summary[f"margin_vs_{suffix}"] = row_summary["delta_vs_h70"] - row_summary[col]

    row_summary["strict_q95_max"] = row_summary[q95_null_cols].max(axis=1)
    row_summary["strict_margin"] = row_summary["delta_vs_h70"] - row_summary["strict_q95_max"]
    row_summary["strict_positive"] = (row_summary["strict_margin"] > 0.0).astype(int)

    row_summary_path = ITER_DIR / "h141_strict_null_sensitivity_row_summary.csv"
    row_summary.to_csv(row_summary_path, index=False)

    split_rows: list[dict[str, object]] = []
    for (domain, split_regime), group in row_summary.groupby(["domain", "split_regime"], sort=True):
        split_rows.append(
            {
                "domain": str(domain),
                "split_regime": str(split_regime),
                "n_rows": int(group.shape[0]),
                "mean_delta_vs_h70": float(group["delta_vs_h70"].mean()),
                "mean_strict_margin": float(group["strict_margin"].mean()),
                "fraction_strict_positive": float((group["strict_margin"] > 0.0).mean()),
                "fraction_directional_positive": float((group["delta_vs_h70"] > 0.0).mean()),
            }
        )
    split_summary_df = pd.DataFrame(split_rows).sort_values(["domain", "split_regime"])
    split_summary_path = ITER_DIR / "h141_strict_null_sensitivity_domain_split_summary.csv"
    split_summary_df.to_csv(split_summary_path, index=False)

    domain_rows: list[dict[str, object]] = []
    for domain, group in row_summary.groupby("domain", sort=True):
        domain_rows.append(
            {
                "domain": str(domain),
                "n_rows": int(group.shape[0]),
                "mean_delta_vs_h70": float(group["delta_vs_h70"].mean()),
                "mean_strict_margin": float(group["strict_margin"].mean()),
                "fraction_strict_positive": float((group["strict_margin"] > 0.0).mean()),
            }
        )
    domain_summary_df = pd.DataFrame(domain_rows).sort_values("domain")
    domain_summary_path = ITER_DIR / "h141_strict_null_sensitivity_domain_summary.csv"
    domain_summary_df.to_csv(domain_summary_path, index=False)

    nullkind_rows: list[dict[str, object]] = []
    margin_cols = [c for c in row_summary.columns if c.startswith("margin_vs_")]
    for col in sorted(margin_cols):
        series = row_summary[col].to_numpy(dtype=float)
        nullkind_rows.append(
            {
                "null_kind": col.replace("margin_vs_", ""),
                "n_rows": int(series.size),
                "mean_margin": float(np.nanmean(series)),
                "fraction_positive_margin": float(np.nanmean(series > 0.0)),
            }
        )
    nullkind_summary_df = pd.DataFrame(nullkind_rows).sort_values("null_kind")
    nullkind_summary_path = ITER_DIR / "h141_strict_null_sensitivity_nullkind_summary.csv"
    nullkind_summary_df.to_csv(nullkind_summary_path, index=False)

    return {
        "rows_tested": int(row_summary.shape[0]),
        "strict_positive_rows": int((row_summary["strict_margin"] > 0.0).sum()),
        "strict_positive_domain_splits": int((split_summary_df["mean_strict_margin"] > 0.0).sum())
        if not split_summary_df.empty
        else 0,
        "mean_strict_margin": float(row_summary["strict_margin"].mean()) if not row_summary.empty else float("nan"),
        "artifact_paths": {
            "row_summary": str(row_summary_path),
            "domain_split_summary": str(split_summary_path),
            "domain_summary": str(domain_summary_path),
            "nullkind_summary": str(nullkind_summary_path),
        },
    }


def main() -> None:
    ensure_required_inputs()

    dorothea_map = BASE.load_dorothea_score_map()
    omnipath_pairs = BASE.load_omnipath_pairs()
    gene2go_upper = BASE.load_gene2go_upper()
    string_map = BASE.load_string_scores_from_cache(BASE.STRING_CACHE_PATH)

    h139 = run_h139_sectional_seed_robustness(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    h140 = run_h140_neighborhood_scaling(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    h141 = run_h141_strict_null_sensitivity(
        h139_row_path=Path(h139["artifact_paths"]["by_seed_domain_split"]),
        h139_null_path=Path(h139["artifact_paths"]["null_summary"]),
    )

    summary = {
        "iteration": "iter_0052",
        "h139": h139,
        "h140": h140,
        "h141": h141,
    }
    summary_path = ITER_DIR / "iter0052_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
