from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

ITER_DIR = Path("iterations/iter_0030")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def load_iter0028_module():
    module_path = Path("iterations/iter_0028/run_iter0028_screen.py")
    spec = importlib.util.spec_from_file_location("iter0028_base", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_iter0028_module()

# H73 / N368: support-concordance anchoring of H70 with hard controls.
H73_LAYERS = [7, 11]
H73_GENE_CAP = 170
H73_NEIGHBORS = 12
H73_TRIANGLE_K = [8, 12, 16]
H73_NULL_PERM = 32
H73_BOOTSTRAP_N = 4000

# H74 / N365: relational spectral cross-model alignment pilot.
H74_LAYERS = [7, 11]
H74_GENE_CAP = 220
H74_SPECTRAL_DIM = 8
H74_SPECTRAL_K = 12
H74_NULL_PERM = 24

# H75 / N361: geodesic curvature-acceleration screen.
H75_LAYERS = [0, 3, 7, 11]
H75_SEED_TAG = "seed42_main"
H75_GENE_CAP = 170
H75_NEIGHBORS = 12
H75_TRIANGLE_K = [8, 12]
H75_NULL_PERM = 32


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


def subset_delta_auc(
    labels: np.ndarray,
    baseline_score: np.ndarray,
    alt_score: np.ndarray,
    mask: np.ndarray,
) -> float:
    idx = np.where(mask)[0]
    if idx.size < 40:
        return float("nan")
    y = labels[idx]
    if np.unique(y).size < 2:
        return float("nan")
    auc_alt = BASE.safe_auc(y, alt_score[idx])
    auc_base = BASE.safe_auc(y, baseline_score[idx])
    if not np.isfinite(auc_alt) or not np.isfinite(auc_base):
        return float("nan")
    return float(auc_alt - auc_base)


def support_interaction_delta(
    labels: np.ndarray,
    baseline_score: np.ndarray,
    defect_score: np.ndarray,
    support_values: np.ndarray,
) -> tuple[float, float, float, int, int]:
    support = np.asarray(support_values, dtype=float)
    finite = np.isfinite(support)
    if finite.sum() < 120:
        return float("nan"), float("nan"), float("nan"), 0, 0

    lo = float(np.quantile(support[finite], 1.0 / 3.0))
    hi = float(np.quantile(support[finite], 2.0 / 3.0))

    low_mask = finite & (support <= lo)
    high_mask = finite & (support >= hi)
    n_low = int(low_mask.sum())
    n_high = int(high_mask.sum())

    delta_low = subset_delta_auc(labels, baseline_score, defect_score, low_mask)
    delta_high = subset_delta_auc(labels, baseline_score, defect_score, high_mask)
    interaction = (
        float(delta_high - delta_low)
        if np.isfinite(delta_high) and np.isfinite(delta_low)
        else float("nan")
    )
    return interaction, delta_high, delta_low, n_high, n_low


def sample_within_strata_with_replacement(
    values: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    s = np.asarray(strata, dtype=int)
    out = x.copy()
    for g in np.unique(s):
        idx = np.where(s == g)[0]
        if idx.size > 1:
            out[idx] = rng.choice(x[idx], size=idx.size, replace=True)
    return out


def build_weighted_knn_graph(points: np.ndarray, n_neighbors: int) -> np.ndarray:
    n = points.shape[0]
    k = max(2, min(int(n_neighbors), n - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(points)
    distances, indices = nbrs.kneighbors(points)

    scale = float(np.median(distances[:, 1:]))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0

    w = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for d, j in zip(distances[i, 1:], indices[i, 1:]):
            jj = int(j)
            val = float(np.exp(-float(d) / scale))
            if val > w[i, jj]:
                w[i, jj] = val
            if val > w[jj, i]:
                w[jj, i] = val
    np.fill_diagonal(w, 0.0)
    return w


def spectral_embedding_from_signatures(
    signature_df: pd.DataFrame,
    n_components: int,
    n_neighbors: int,
) -> pd.DataFrame:
    x = signature_df.to_numpy(dtype=float)
    x_mu, x_sd = BASE.zscore_fit(x)
    x_z = BASE.zscore_apply(x, x_mu, x_sd)

    w = build_weighted_knn_graph(x_z, n_neighbors=n_neighbors)
    degree = np.sum(w, axis=1)
    degree = np.clip(degree, 1e-8, None)
    inv_sqrt = 1.0 / np.sqrt(degree)
    m = (inv_sqrt[:, None] * w) * inv_sqrt[None, :]

    evals, evecs = np.linalg.eigh(m)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    use = min(n_components, max(1, evecs.shape[1] - 1))
    cols = np.arange(1, use + 1)
    emb = evecs[:, cols] * np.sqrt(np.clip(evals[cols], 1e-8, None))[None, :]
    emb = BASE.row_normalize(emb)

    out = pd.DataFrame(emb, index=signature_df.index)
    out.columns = [f"spec_{i+1}" for i in range(out.shape[1])]
    return out


def orthogonal_map(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_centered = src - src.mean(axis=0, keepdims=True)
    dst_centered = dst - dst.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(src_centered.T @ dst_centered, full_matrices=False)
    return u @ vt


def random_orthogonal_matrix(dim: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    sign = np.sign(np.diag(r))
    sign[sign == 0.0] = 1.0
    return q * sign[None, :]


def edge_pair_features(embedding: np.ndarray, src_idx: np.ndarray, tgt_idx: np.ndarray) -> np.ndarray:
    src = embedding[src_idx]
    tgt = embedding[tgt_idx]
    return np.concatenate([src, tgt, np.abs(src - tgt), src * tgt], axis=1)


def run_h73_support_concordance_anchor(
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

                top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H73_GENE_CAP))
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

                for layer in H73_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = BASE.reduce_points(
                        points,
                        n_components=20,
                        random_state=30_730 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer,
                    )
                    geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H73_NEIGHBORS)

                    edge_geodesic = geodesic[source_local, target_local]
                    edge_support = support_dir[source_local, target_local]
                    edge_margin = np.abs(
                        support_dir[source_local, target_local] - support_dir[target_local, source_local]
                    )

                    feature_bundle = BASE.multiscale_triangle_defect_features(
                        geodesic=geodesic,
                        source_local=source_local,
                        target_local=target_local,
                        k_values=H73_TRIANGLE_K,
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
                    support_concordance = edge_support - 0.40 * edge_margin

                    auc_baseline = BASE.safe_auc(labels, baseline_score)
                    auc_defect = BASE.safe_auc(labels, defect_score)
                    delta_auc = (
                        float(auc_defect - auc_baseline)
                        if np.isfinite(auc_defect) and np.isfinite(auc_baseline)
                        else float("nan")
                    )

                    interaction_delta, delta_high, delta_low, n_high, n_low = support_interaction_delta(
                        labels=labels,
                        baseline_score=baseline_score,
                        defect_score=defect_score,
                        support_values=support_concordance,
                    )
                    if not np.isfinite(interaction_delta):
                        continue

                    geodesic_bins = BASE.degree_bins(edge_geodesic, max_bins=6)
                    support_bins = BASE.degree_bins(support_concordance, max_bins=4)
                    strata = geodesic_bins * 6 + support_bins

                    rng = np.random.default_rng(
                        30_731 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer
                    )
                    null_shuffle = np.empty(H73_NULL_PERM, dtype=float)
                    null_matched = np.empty(H73_NULL_PERM, dtype=float)
                    null_label = np.empty(H73_NULL_PERM, dtype=float)

                    for perm_idx in range(H73_NULL_PERM):
                        # Null A: shuffle support-concordance within matched geodesic/support strata.
                        support_perm = BASE.stratified_shuffle(support_concordance, strata, rng)
                        interaction_perm, _, _, _, _ = support_interaction_delta(
                            labels=labels,
                            baseline_score=baseline_score,
                            defect_score=defect_score,
                            support_values=support_perm,
                        )
                        null_shuffle[perm_idx] = interaction_perm
                        null_rows.append(
                            {
                                "null_kind": "support_shuffle_within_geodesic_support_bins",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_interaction_delta": float(interaction_perm),
                            }
                        )

                        # Null B: matched random support draw within the same strata.
                        support_matched = sample_within_strata_with_replacement(support_concordance, strata, rng)
                        interaction_matched, _, _, _, _ = support_interaction_delta(
                            labels=labels,
                            baseline_score=baseline_score,
                            defect_score=defect_score,
                            support_values=support_matched,
                        )
                        null_matched[perm_idx] = interaction_matched
                        null_rows.append(
                            {
                                "null_kind": "matched_random_support_within_strata",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_interaction_delta": float(interaction_matched),
                            }
                        )

                        # Null C: label shuffle within geodesic bins.
                        labels_perm = BASE.stratified_shuffle(labels, geodesic_bins, rng).astype(int)
                        interaction_label, _, _, _, _ = support_interaction_delta(
                            labels=labels_perm,
                            baseline_score=baseline_score,
                            defect_score=defect_score,
                            support_values=support_concordance,
                        )
                        null_label[perm_idx] = interaction_label
                        null_rows.append(
                            {
                                "null_kind": "label_shuffle_within_geodesic_bins",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_interaction_delta": float(interaction_label),
                            }
                        )

                    p_shuffle = BASE.empirical_upper_tail_p(interaction_delta, null_shuffle)
                    p_matched = BASE.empirical_upper_tail_p(interaction_delta, null_matched)
                    p_label = BASE.empirical_upper_tail_p(interaction_delta, null_label)
                    p_best = np.nanmin(np.array([p_shuffle, p_matched, p_label], dtype=float))

                    all_null = np.concatenate([null_shuffle, null_matched, null_label])
                    all_null = all_null[np.isfinite(all_null)]
                    null_q95 = float(np.quantile(all_null, 0.95)) if all_null.size else float("nan")
                    null_gap = (
                        float(interaction_delta - null_q95)
                        if np.isfinite(interaction_delta) and np.isfinite(null_q95)
                        else float("nan")
                    )

                    rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "n_edges_eval": int(labels.size),
                            "n_positive_eval": int(labels.sum()),
                            "n_high_support_edges": int(n_high),
                            "n_low_support_edges": int(n_low),
                            "auc_baseline": float(auc_baseline),
                            "auc_triangle_defect": float(auc_defect),
                            "delta_auc_triangle_minus_baseline": float(delta_auc),
                            "delta_auc_high_support": float(delta_high),
                            "delta_auc_low_support": float(delta_low),
                            "interaction_delta_high_minus_low": float(interaction_delta),
                            "mean_support_concordance": float(np.mean(support_concordance)),
                            "null_q95_interaction": float(null_q95),
                            "null_gap_q95_interaction": float(null_gap),
                            "p_support_shuffle_upper": float(p_shuffle),
                            "p_matched_support_upper": float(p_matched),
                            "p_label_shuffle_upper": float(p_label),
                            "p_best_upper": float(p_best),
                        }
                    )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h73_support_concordance_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h73_support_concordance_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            rng = np.random.default_rng(abs(hash((domain, split_regime, "h73"))) % (2**32))
            mean_interaction, ci_lo, ci_hi = bootstrap_mean_ci(
                group["interaction_delta_high_minus_low"].to_numpy(dtype=float),
                rng=rng,
                n_boot=H73_BOOTSTRAP_N,
                alpha=0.05,
            )
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_triangle_minus_baseline": float(
                        group["delta_auc_triangle_minus_baseline"].mean()
                    ),
                    "mean_interaction_delta_high_minus_low": float(mean_interaction),
                    "bootstrap_ci95_interaction_lo": float(ci_lo),
                    "bootstrap_ci95_interaction_hi": float(ci_hi),
                    "mean_null_gap_q95_interaction": float(group["null_gap_q95_interaction"].mean()),
                    "fraction_interaction_positive": float(
                        (group["interaction_delta_high_minus_low"] > 0.0).mean()
                    ),
                    "fraction_null_gap_positive": float((group["null_gap_q95_interaction"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(
                        BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))
                    ),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h73_support_concordance_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_triangle_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "mean_interaction": float(by_row_df["interaction_delta_high_minus_low"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_interaction_domain_splits": int(
            (summary_df["mean_interaction_delta_high_minus_low"] > 0.0).sum()
        )
        if not summary_df.empty
        else 0,
        "null_surviving_domain_splits": int((summary_df["mean_null_gap_q95_interaction"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_layer_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h74_relational_spectral_alignment() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    domains = ["immune", "lung", "external_lung"]

    sc_edges_seed42: dict[str, pd.DataFrame] = {}
    sc_layers_seed42: dict[str, np.ndarray] = {}
    gf_edges: dict[str, pd.DataFrame] = {}
    sc_spectral: dict[tuple[str, int], pd.DataFrame] = {}
    gf_spectral: dict[tuple[str, int], pd.DataFrame] = {}

    for domain_index, domain in enumerate(domains):
        sc_run = BASE.SCGPT_RUNS_BY_DOMAIN[domain]["seed42_main"]
        sc_edges_seed42[domain] = pd.read_csv(sc_run / "cycle1_edge_dataset.tsv", sep="\t")
        sc_layers_seed42[domain] = np.load(sc_run / "layer_gene_embeddings.npy", mmap_mode="r")
        gf_edges[domain] = pd.read_csv(BASE.GENEFORMER_EDGE_BY_DOMAIN[domain], sep="\t")

        sc_df = sc_edges_seed42[domain].copy()
        top_genes = set(BASE.select_top_genes(sc_df, gene_cap=H74_GENE_CAP))
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

        gf_sig = BASE.fit_signatures_geneformer(gf_edges[domain], symbols)
        for layer in H74_LAYERS:
            if layer >= sc_layers_seed42[domain].shape[0]:
                continue
            points = sc_layers_seed42[domain][layer, gene_indices, :]
            sc_sig = BASE.fit_signatures_scgpt(
                layer_points=points,
                symbols=symbols,
                random_state=30_740 + domain_index * 100 + layer,
                n_neighbors=10,
            )

            shared = sorted(set(sc_sig.index) & set(gf_sig.index))
            if len(shared) < 90:
                continue
            sc_sig_shared = sc_sig.loc[shared]
            gf_sig_shared = gf_sig.loc[shared]

            sc_spectral[(domain, layer)] = spectral_embedding_from_signatures(
                sc_sig_shared,
                n_components=H74_SPECTRAL_DIM,
                n_neighbors=H74_SPECTRAL_K,
            )
            gf_spectral[(domain, layer)] = spectral_embedding_from_signatures(
                gf_sig_shared,
                n_components=H74_SPECTRAL_DIM,
                n_neighbors=H74_SPECTRAL_K,
            )

    for domain_index, target_domain in enumerate(domains):
        source_domains = [d for d in domains if d != target_domain]
        split_masks = BASE.build_split_masks(sc_edges_seed42[target_domain])
        source_split_masks = {
            src_domain: BASE.build_split_masks(sc_edges_seed42[src_domain])
            for src_domain in source_domains
        }

        for layer in H74_LAYERS:
            source_sc_stack: list[np.ndarray] = []
            source_gf_stack: list[np.ndarray] = []

            train_x: list[np.ndarray] = []
            train_y: list[np.ndarray] = []

            for src_domain in source_domains:
                sc_emb_df = sc_spectral.get((src_domain, layer))
                gf_emb_df = gf_spectral.get((src_domain, layer))
                if sc_emb_df is None or gf_emb_df is None:
                    continue

                shared = sorted(set(sc_emb_df.index) & set(gf_emb_df.index))
                if len(shared) < 80:
                    continue
                source_sc_stack.append(sc_emb_df.loc[shared].to_numpy(dtype=float))
                source_gf_stack.append(gf_emb_df.loc[shared].to_numpy(dtype=float))

            if not source_sc_stack or not source_gf_stack:
                continue

            sc_train_aln = np.vstack(source_sc_stack)
            gf_train_aln = np.vstack(source_gf_stack)
            if min(sc_train_aln.shape[0], gf_train_aln.shape[0]) < 160:
                continue

            map_r = orthogonal_map(src=gf_train_aln, dst=sc_train_aln)

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                train_x.clear()
                train_y.clear()

                for src_domain in source_domains:
                    sc_emb_df = sc_spectral.get((src_domain, layer))
                    if sc_emb_df is None:
                        continue

                    src_mask = source_split_masks[src_domain].get(split_regime)
                    if src_mask is None:
                        continue
                    split_df = sc_edges_seed42[src_domain].loc[src_mask].copy()
                    split_df["source_u"] = split_df["source"].astype(str).str.upper()
                    split_df["target_u"] = split_df["target"].astype(str).str.upper()

                    symbols = sorted(set(sc_emb_df.index))
                    symbol_to_idx = {sym: i for i, sym in enumerate(symbols)}
                    keep = split_df["source_u"].isin(symbol_to_idx) & split_df["target_u"].isin(symbol_to_idx)
                    split_df = split_df.loc[keep].copy()
                    if split_df["label"].nunique() < 2 or split_df.shape[0] < 280:
                        continue

                    emb = sc_emb_df.loc[symbols].to_numpy(dtype=float)
                    src_idx = np.array([symbol_to_idx[s] for s in split_df["source_u"].to_numpy(dtype=str)], dtype=int)
                    tgt_idx = np.array([symbol_to_idx[t] for t in split_df["target_u"].to_numpy(dtype=str)], dtype=int)
                    feat = edge_pair_features(emb, src_idx, tgt_idx)
                    lbl = split_df["label"].to_numpy(dtype=int)
                    train_x.append(feat)
                    train_y.append(lbl)

                if not train_x:
                    continue

                x_train = np.vstack(train_x)
                y_train = np.concatenate(train_y)
                if x_train.shape[0] < 700 or np.unique(y_train).size < 2:
                    continue

                clf = LogisticRegression(
                    max_iter=1200,
                    solver="lbfgs",
                    C=1.0,
                    random_state=30_741 + domain_index * 100 + split_index * 10 + layer,
                )
                clf.fit(x_train, y_train)

                sc_tgt_df = sc_spectral.get((target_domain, layer))
                gf_tgt_df = gf_spectral.get((target_domain, layer))
                if sc_tgt_df is None or gf_tgt_df is None:
                    continue

                shared_tgt = sorted(set(sc_tgt_df.index) & set(gf_tgt_df.index))
                if len(shared_tgt) < 80:
                    continue

                sc_tgt = sc_tgt_df.loc[shared_tgt].to_numpy(dtype=float)
                gf_tgt = gf_tgt_df.loc[shared_tgt].to_numpy(dtype=float)
                mapped_gf_tgt = gf_tgt @ map_r

                sym_to_idx = {sym: i for i, sym in enumerate(shared_tgt)}
                eval_df = sc_edges_seed42[target_domain].loc[split_mask].copy()
                eval_df["source_u"] = eval_df["source"].astype(str).str.upper()
                eval_df["target_u"] = eval_df["target"].astype(str).str.upper()
                keep = eval_df["source_u"].isin(sym_to_idx) & eval_df["target_u"].isin(sym_to_idx)
                eval_df = eval_df.loc[keep].copy()
                if eval_df["label"].nunique() < 2 or eval_df.shape[0] < 280:
                    continue

                labels = eval_df["label"].to_numpy(dtype=int)
                src_idx = np.array([sym_to_idx[s] for s in eval_df["source_u"].to_numpy(dtype=str)], dtype=int)
                tgt_idx = np.array([sym_to_idx[t] for t in eval_df["target_u"].to_numpy(dtype=str)], dtype=int)

                feat_transfer = edge_pair_features(mapped_gf_tgt, src_idx, tgt_idx)
                feat_baseline = edge_pair_features(gf_tgt, src_idx, tgt_idx)

                score_transfer = clf.predict_proba(feat_transfer)[:, 1]
                score_baseline = clf.predict_proba(feat_baseline)[:, 1]

                auc_transfer = BASE.safe_auc(labels, score_transfer)
                auc_baseline = BASE.safe_auc(labels, score_baseline)
                delta_auc = (
                    float(auc_transfer - auc_baseline)
                    if np.isfinite(auc_transfer) and np.isfinite(auc_baseline)
                    else float("nan")
                )

                alignment_cosine = BASE.mean_row_cosine(mapped_gf_tgt, sc_tgt)

                rng = np.random.default_rng(30_742 + domain_index * 100 + split_index * 10 + layer)
                dim = mapped_gf_tgt.shape[1]

                null_spec_perm = np.empty(H74_NULL_PERM, dtype=float)
                null_rand_basis = np.empty(H74_NULL_PERM, dtype=float)
                null_destroy = np.empty(H74_NULL_PERM, dtype=float)

                for perm_idx in range(H74_NULL_PERM):
                    # Null A: eigen-spectrum permutation (dimension order + sign flips).
                    dim_perm = rng.permutation(dim)
                    signs = rng.choice(np.array([-1.0, 1.0]), size=dim)
                    mapped_perm = mapped_gf_tgt[:, dim_perm] * signs[None, :]
                    score_perm = clf.predict_proba(edge_pair_features(mapped_perm, src_idx, tgt_idx))[:, 1]
                    auc_perm = BASE.safe_auc(labels, score_perm)
                    delta_perm = (
                        float(auc_perm - auc_baseline)
                        if np.isfinite(auc_perm) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_spec_perm[perm_idx] = delta_perm
                    null_rows.append(
                        {
                            "null_kind": "eigenspectrum_permutation",
                            "domain": target_domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_perm),
                        }
                    )

                    # Null B: random orthogonal basis map.
                    rand_q = random_orthogonal_matrix(dim, rng)
                    mapped_rand = gf_tgt @ rand_q
                    score_rand = clf.predict_proba(edge_pair_features(mapped_rand, src_idx, tgt_idx))[:, 1]
                    auc_rand = BASE.safe_auc(labels, score_rand)
                    delta_rand = (
                        float(auc_rand - auc_baseline)
                        if np.isfinite(auc_rand) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_rand_basis[perm_idx] = delta_rand
                    null_rows.append(
                        {
                            "null_kind": "random_orthogonal_basis",
                            "domain": target_domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_rand),
                        }
                    )

                    # Null C: signature-destroy row permutation.
                    row_perm = rng.permutation(mapped_gf_tgt.shape[0])
                    mapped_destroy = mapped_gf_tgt[row_perm]
                    score_destroy = clf.predict_proba(edge_pair_features(mapped_destroy, src_idx, tgt_idx))[:, 1]
                    auc_destroy = BASE.safe_auc(labels, score_destroy)
                    delta_destroy = (
                        float(auc_destroy - auc_baseline)
                        if np.isfinite(auc_destroy) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_destroy[perm_idx] = delta_destroy
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

                p_spec = BASE.empirical_upper_tail_p(delta_auc, null_spec_perm)
                p_rand = BASE.empirical_upper_tail_p(delta_auc, null_rand_basis)
                p_destroy = BASE.empirical_upper_tail_p(delta_auc, null_destroy)
                p_best = np.nanmin(np.array([p_spec, p_rand, p_destroy], dtype=float))

                all_null = np.concatenate([null_spec_perm, null_rand_basis, null_destroy])
                all_null = all_null[np.isfinite(all_null)]
                null_q95 = float(np.quantile(all_null, 0.95)) if all_null.size else float("nan")
                null_gap = (
                    float(delta_auc - null_q95)
                    if np.isfinite(delta_auc) and np.isfinite(null_q95)
                    else float("nan")
                )

                rows.append(
                    {
                        "domain": target_domain,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "train_domains": "+".join(sorted(source_domains)),
                        "n_train_edges": int(x_train.shape[0]),
                        "n_eval_edges": int(labels.size),
                        "n_eval_positive": int(labels.sum()),
                        "n_shared_target_symbols": int(len(shared_tgt)),
                        "auc_transfer": float(auc_transfer),
                        "auc_baseline": float(auc_baseline),
                        "delta_auc_transfer_minus_baseline": float(delta_auc),
                        "mapped_to_sc_cosine": float(alignment_cosine),
                        "null_q95": float(null_q95),
                        "null_gap_q95": float(null_gap),
                        "p_eigenspectrum_upper": float(p_spec),
                        "p_random_basis_upper": float(p_rand),
                        "p_signature_destroy_upper": float(p_destroy),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h74_relational_spectral_alignment_by_domain_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h74_relational_spectral_alignment_null_summary.csv"
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
    summary_path = ITER_DIR / "h74_relational_spectral_alignment_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_transfer_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "mean_null_gap_q95": float(by_row_df["null_gap_q95"].mean()) if not by_row_df.empty else float("nan"),
        "mean_alignment_cosine": float(by_row_df["mapped_to_sc_cosine"].mean()) if not by_row_df.empty else float("nan"),
        "artifact_paths": {
            "by_domain_layer_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h75_curvature_acceleration_scan(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map.get(H75_SEED_TAG)
        if run_dir is None:
            continue

        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H75_GENE_CAP))
            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
            ].copy()
            if split_edges["label"].nunique() < 2 or split_edges.shape[0] < 500:
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

            curvature_layers: list[np.ndarray] = []
            baseline_layers: list[np.ndarray] = []
            geodesic_ref = np.zeros(labels.size, dtype=float)
            used_layers: list[int] = []

            for layer in H75_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=30_750 + domain_index * 100 + split_index * 10 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H75_NEIGHBORS)
                edge_geodesic = geodesic[source_local, target_local]
                edge_support = support_dir[source_local, target_local]
                edge_margin = np.abs(support_dir[source_local, target_local] - support_dir[target_local, source_local])

                feature_bundle = BASE.multiscale_triangle_defect_features(
                    geodesic=geodesic,
                    source_local=source_local,
                    target_local=target_local,
                    k_values=H75_TRIANGLE_K,
                )
                curvature_proxy = -feature_bundle["median_mean"]
                baseline_component = (
                    BASE.zscore(-edge_geodesic)
                    + 0.75 * BASE.zscore(edge_support)
                    + 0.35 * BASE.zscore(edge_margin)
                )

                curvature_layers.append(curvature_proxy)
                baseline_layers.append(baseline_component)
                geodesic_ref = edge_geodesic
                used_layers.append(int(layer))

            if len(curvature_layers) < 4:
                continue

            curvature_mat = np.column_stack(curvature_layers)
            baseline_score = np.mean(np.column_stack(baseline_layers), axis=1)
            slope = np.diff(curvature_mat, axis=1)
            accel = np.diff(slope, axis=1)

            mean_abs_accel = np.mean(np.abs(accel), axis=1)
            max_abs_accel = np.max(np.abs(accel), axis=1)
            mean_slope = np.mean(slope, axis=1)
            end_minus_start = curvature_mat[:, -1] - curvature_mat[:, 0]

            curvature_score = (
                baseline_score
                + 0.30 * BASE.zscore(-mean_abs_accel)
                + 0.20 * BASE.zscore(-max_abs_accel)
                + 0.15 * BASE.zscore(mean_slope)
                + 0.10 * BASE.zscore(end_minus_start)
            )

            auc_baseline = BASE.safe_auc(labels, baseline_score)
            auc_curvature = BASE.safe_auc(labels, curvature_score)
            delta_auc = (
                float(auc_curvature - auc_baseline)
                if np.isfinite(auc_curvature) and np.isfinite(auc_baseline)
                else float("nan")
            )

            bins = BASE.degree_bins(geodesic_ref, max_bins=6)
            rng = np.random.default_rng(30_751 + domain_index * 100 + split_index)

            null_layer_order = np.empty(H75_NULL_PERM, dtype=float)
            null_curvature_shuffle = np.empty(H75_NULL_PERM, dtype=float)
            null_label = np.empty(H75_NULL_PERM, dtype=float)

            for perm_idx in range(H75_NULL_PERM):
                # Null A: permute layer order before slope/acceleration extraction.
                perm = rng.permutation(curvature_mat.shape[1])
                perm_mat = curvature_mat[:, perm]
                slope_perm = np.diff(perm_mat, axis=1)
                accel_perm = np.diff(slope_perm, axis=1)
                score_perm = (
                    baseline_score
                    + 0.30 * BASE.zscore(-np.mean(np.abs(accel_perm), axis=1))
                    + 0.20 * BASE.zscore(-np.max(np.abs(accel_perm), axis=1))
                    + 0.15 * BASE.zscore(np.mean(slope_perm, axis=1))
                    + 0.10 * BASE.zscore(perm_mat[:, -1] - perm_mat[:, 0])
                )
                auc_perm = BASE.safe_auc(labels, score_perm)
                delta_perm = (
                    float(auc_perm - auc_baseline)
                    if np.isfinite(auc_perm) and np.isfinite(auc_baseline)
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

                # Null B: shuffle curvature smoothness terms within geodesic bins.
                accel_shuffle = BASE.stratified_shuffle(mean_abs_accel, bins, rng)
                max_accel_shuffle = BASE.stratified_shuffle(max_abs_accel, bins, rng)
                slope_shuffle = BASE.stratified_shuffle(mean_slope, bins, rng)
                closure_shuffle = BASE.stratified_shuffle(end_minus_start, bins, rng)
                score_shuffle = (
                    baseline_score
                    + 0.30 * BASE.zscore(-accel_shuffle)
                    + 0.20 * BASE.zscore(-max_accel_shuffle)
                    + 0.15 * BASE.zscore(slope_shuffle)
                    + 0.10 * BASE.zscore(closure_shuffle)
                )
                auc_shuffle = BASE.safe_auc(labels, score_shuffle)
                delta_shuffle = (
                    float(auc_shuffle - auc_baseline)
                    if np.isfinite(auc_shuffle) and np.isfinite(auc_baseline)
                    else float("nan")
                )
                null_curvature_shuffle[perm_idx] = delta_shuffle
                null_rows.append(
                    {
                        "null_kind": "curvature_shuffle_within_geodesic_bins",
                        "domain": domain,
                        "split_regime": split_regime,
                        "perm_idx": int(perm_idx),
                        "null_delta_auc": float(delta_shuffle),
                    }
                )

                # Null C: label permutation within geodesic bins.
                labels_perm = BASE.stratified_shuffle(labels, bins, rng).astype(int)
                auc_lp = BASE.safe_auc(labels_perm, curvature_score)
                auc_lp_base = BASE.safe_auc(labels_perm, baseline_score)
                delta_lp = (
                    float(auc_lp - auc_lp_base)
                    if np.isfinite(auc_lp) and np.isfinite(auc_lp_base)
                    else float("nan")
                )
                null_label[perm_idx] = delta_lp
                null_rows.append(
                    {
                        "null_kind": "label_shuffle_within_geodesic_bins",
                        "domain": domain,
                        "split_regime": split_regime,
                        "perm_idx": int(perm_idx),
                        "null_delta_auc": float(delta_lp),
                    }
                )

            p_layer = BASE.empirical_upper_tail_p(delta_auc, null_layer_order)
            p_curvature = BASE.empirical_upper_tail_p(delta_auc, null_curvature_shuffle)
            p_label = BASE.empirical_upper_tail_p(delta_auc, null_label)
            p_best = np.nanmin(np.array([p_layer, p_curvature, p_label], dtype=float))

            all_null = np.concatenate([null_layer_order, null_curvature_shuffle, null_label])
            all_null = all_null[np.isfinite(all_null)]
            null_q95 = float(np.quantile(all_null, 0.95)) if all_null.size else float("nan")
            null_gap = float(delta_auc - null_q95) if np.isfinite(delta_auc) and np.isfinite(null_q95) else float("nan")

            rows.append(
                {
                    "domain": domain,
                    "seed_tag": H75_SEED_TAG,
                    "split_regime": split_regime,
                    "layers_used": ",".join(str(x) for x in used_layers),
                    "n_edges_eval": int(labels.size),
                    "n_positive_eval": int(labels.sum()),
                    "auc_baseline": float(auc_baseline),
                    "auc_curvature_accel": float(auc_curvature),
                    "delta_auc_curvature_minus_baseline": float(delta_auc),
                    "mean_abs_accel": float(np.mean(mean_abs_accel)),
                    "mean_slope": float(np.mean(mean_slope)),
                    "null_q95": float(null_q95),
                    "null_gap_q95": float(null_gap),
                    "p_layer_order_upper": float(p_layer),
                    "p_curvature_shuffle_upper": float(p_curvature),
                    "p_label_shuffle_upper": float(p_label),
                    "p_best_upper": float(p_best),
                }
            )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime"])
    by_row_path = ITER_DIR / "h75_curvature_acceleration_by_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h75_curvature_acceleration_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_auc_curvature_accel": float(group["auc_curvature_accel"].mean()),
                    "mean_auc_baseline": float(group["auc_baseline"].mean()),
                    "mean_delta_auc_curvature_minus_baseline": float(
                        group["delta_auc_curvature_minus_baseline"].mean()
                    ),
                    "mean_null_gap_q95": float(group["null_gap_q95"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_curvature_minus_baseline"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(
                        BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))
                    ),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h75_curvature_acceleration_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_curvature_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_curvature_minus_baseline"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "significant_domain_splits": int((summary_df["fraction_p_best_lt_0_05"] > 0.0).sum())
        if not summary_df.empty
        else 0,
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

    h73_summary = run_h73_support_concordance_anchor(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h74_summary = run_h74_relational_spectral_alignment()
    h75_summary = run_h75_curvature_acceleration_scan(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    summary = {
        "iteration": "iter_0030",
        "h73": h73_summary,
        "h74": h74_summary,
        "h75": h75_summary,
        "environment": {
            "python_env": "subproject40-topology",
            "string_cache_used": str(BASE.STRING_CACHE_PATH),
        },
    }
    summary_path = ITER_DIR / "iter0030_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
