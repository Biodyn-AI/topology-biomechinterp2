from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.neighbors import NearestNeighbors

ITER_DIR = Path("iterations/iter_0031")
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

# H76 / N382 rescue-once major change: coexpression-aware support-concordance interaction v2.
H76_LAYERS = [7, 11]
H76_GENE_CAP = 170
H76_NEIGHBORS = 12
H76_TRIANGLE_K = [8, 12, 16]
H76_NULL_PERM = 24
H76_BOOTSTRAP_N = 4000

# H77 / N379 rescue-once major change: cross-model relational rank agreement endpoint.
H77_LAYERS = [7, 11]
H77_GENE_CAP = 220
H77_SPECTRAL_DIM = 8
H77_SPECTRAL_K = 12
H77_NULL_PERM = 24

# H78 / N376 new method: geodesic detour elasticity under neighborhood-size perturbation.
H78_LAYERS = [0, 3, 7, 11]
H78_SEED_TAG = "seed42_main"
H78_GENE_CAP = 170
H78_NEIGHBORS_LIST = [8, 12, 16]
H78_TRIANGLE_K = [8, 12]
H78_NULL_PERM = 24


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


def combine_strata(parts: list[np.ndarray]) -> np.ndarray:
    strata = np.zeros(parts[0].size, dtype=np.int64)
    factor = np.int64(1)
    for arr in parts:
        vals = np.asarray(arr, dtype=np.int64)
        if vals.size != strata.size:
            raise ValueError("All strata arrays must have equal length.")
        vmin = int(vals.min()) if vals.size else 0
        vals = vals - vmin
        vmax = int(vals.max()) if vals.size else 0
        strata += factor * vals
        factor *= np.int64(max(2, vmax + 1))
    return strata.astype(int)


def build_go_jaccard_matrix(
    symbols_upper: list[str],
    gene2go_upper: dict[str, set[str]],
) -> np.ndarray:
    n = len(symbols_upper)
    out = np.zeros((n, n), dtype=float)
    go_sets = [gene2go_upper.get(sym, set()) for sym in symbols_upper]
    for i in range(n):
        g_i = go_sets[i]
        for j in range(i + 1, n):
            g_j = go_sets[j]
            union = len(g_i | g_j)
            value = float(len(g_i & g_j) / union) if union > 0 else 0.0
            out[i, j] = value
            out[j, i] = value
    return out


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < 3:
        return float("nan")
    xr = pd.Series(xa[mask]).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(ya[mask]).rank(method="average").to_numpy(dtype=float)
    if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def topk_overlap_smallest(ref: np.ndarray, other: np.ndarray, topk: int) -> float:
    x = np.asarray(ref, dtype=float)
    y = np.asarray(other, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < max(10, topk):
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    k = int(min(topk, xv.size))
    if k <= 0:
        return float("nan")
    idx_x = np.argpartition(xv, k - 1)[:k]
    idx_y = np.argpartition(yv, k - 1)[:k]
    overlap = np.intersect1d(idx_x, idx_y).size
    return float(overlap / max(1, k))


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


def run_h76_coexpression_support_interaction_v2(
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

                top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H76_GENE_CAP))
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
                go_jaccard = build_go_jaccard_matrix(symbols, gene2go_upper)

                gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
                source_local = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
                target_local = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
                labels = split_edges["label"].to_numpy(dtype=int)

                for layer in H76_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = BASE.reduce_points(
                        points,
                        n_components=20,
                        random_state=31_760 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer,
                    )
                    geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H76_NEIGHBORS)
                    edge_geodesic = geodesic[source_local, target_local]
                    edge_support = support_dir[source_local, target_local]
                    edge_margin = np.abs(
                        support_dir[source_local, target_local] - support_dir[target_local, source_local]
                    )
                    edge_ontology = go_jaccard[source_local, target_local]

                    # Coexpression proxy: cosine agreement in the residual-space local chart.
                    points_norm = BASE.row_normalize(points_pca)
                    cos_sim = np.clip(points_norm @ points_norm.T, -1.0, 1.0)
                    edge_coexp_proxy = cos_sim[source_local, target_local]

                    knn_edges = BASE.build_knn_edge_array(points_pca, n_neighbors=H76_NEIGHBORS)
                    neighbors = BASE.adjacency_neighbors(points_pca.shape[0], knn_edges)
                    degree = np.array([len(n) for n in neighbors], dtype=float)
                    edge_degree = 0.5 * (degree[source_local] + degree[target_local])

                    feature_bundle = BASE.multiscale_triangle_defect_features(
                        geodesic=geodesic,
                        source_local=source_local,
                        target_local=target_local,
                        k_values=H76_TRIANGLE_K,
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

                    geodesic_bins = BASE.degree_bins(edge_geodesic, max_bins=5)
                    degree_bins = BASE.degree_bins(edge_degree, max_bins=4)
                    coexp_bins = BASE.degree_bins(edge_coexp_proxy, max_bins=4)
                    ontology_bins = BASE.degree_bins(edge_ontology, max_bins=3)
                    strata = combine_strata([geodesic_bins, degree_bins, coexp_bins, ontology_bins])

                    rng = np.random.default_rng(
                        31_761 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer
                    )
                    null_shuffle = np.empty(H76_NULL_PERM, dtype=float)
                    null_matched = np.empty(H76_NULL_PERM, dtype=float)
                    null_label = np.empty(H76_NULL_PERM, dtype=float)

                    for perm_idx in range(H76_NULL_PERM):
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
                                "null_kind": "support_shuffle_within_gxdxcxontology",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_interaction_delta": float(interaction_perm),
                            }
                        )

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
                            "mean_coexpression_proxy": float(np.mean(edge_coexp_proxy)),
                            "mean_ontology_overlap": float(np.mean(edge_ontology)),
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
    by_row_path = ITER_DIR / "h76_coexpression_support_interaction_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h76_coexpression_support_interaction_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            rng = np.random.default_rng(abs(hash((domain, split_regime, "h76"))) % (2**32))
            mean_interaction, ci_lo, ci_hi = bootstrap_mean_ci(
                group["interaction_delta_high_minus_low"].to_numpy(dtype=float),
                rng=rng,
                n_boot=H76_BOOTSTRAP_N,
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
    summary_path = ITER_DIR / "h76_coexpression_support_interaction_domain_summary.csv"
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


def run_h77_relational_rank_agreement() -> dict[str, object]:
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
        top_genes = set(BASE.select_top_genes(sc_df, gene_cap=H77_GENE_CAP))
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
        for layer in H77_LAYERS:
            if layer >= sc_layers_seed42[domain].shape[0]:
                continue
            points = sc_layers_seed42[domain][layer, gene_indices, :]
            sc_sig = BASE.fit_signatures_scgpt(
                layer_points=points,
                symbols=symbols,
                random_state=31_770 + domain_index * 100 + layer,
                n_neighbors=10,
            )

            shared = sorted(set(sc_sig.index) & set(gf_sig.index))
            if len(shared) < 90:
                continue
            sc_sig_shared = sc_sig.loc[shared]
            gf_sig_shared = gf_sig.loc[shared]

            sc_spectral[(domain, layer)] = spectral_embedding_from_signatures(
                sc_sig_shared,
                n_components=H77_SPECTRAL_DIM,
                n_neighbors=H77_SPECTRAL_K,
            )
            gf_spectral[(domain, layer)] = spectral_embedding_from_signatures(
                gf_sig_shared,
                n_components=H77_SPECTRAL_DIM,
                n_neighbors=H77_SPECTRAL_K,
            )

    for domain_index, target_domain in enumerate(domains):
        source_domains = [d for d in domains if d != target_domain]
        split_masks = BASE.build_split_masks(sc_edges_seed42[target_domain])

        for layer in H77_LAYERS:
            source_sc_stack: list[np.ndarray] = []
            source_gf_stack: list[np.ndarray] = []

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
                sc_tgt_df = sc_spectral.get((target_domain, layer))
                gf_tgt_df = gf_spectral.get((target_domain, layer))
                if sc_tgt_df is None or gf_tgt_df is None:
                    continue

                shared_tgt = sorted(set(sc_tgt_df.index) & set(gf_tgt_df.index))
                if len(shared_tgt) < 80:
                    continue

                eval_df = sc_edges_seed42[target_domain].loc[split_mask].copy()
                eval_df["source_u"] = eval_df["source"].astype(str).str.upper()
                eval_df["target_u"] = eval_df["target"].astype(str).str.upper()
                split_symbols = sorted(set(eval_df["source_u"]) | set(eval_df["target_u"]))
                split_symbols = [s for s in split_symbols if s in set(shared_tgt)]
                if len(split_symbols) < 40:
                    continue

                sym_to_idx = {sym: i for i, sym in enumerate(shared_tgt)}
                local_idx = np.array([sym_to_idx[s] for s in split_symbols], dtype=int)

                sc_tgt = sc_tgt_df.loc[shared_tgt].to_numpy(dtype=float)[local_idx]
                gf_tgt = gf_tgt_df.loc[shared_tgt].to_numpy(dtype=float)[local_idx]
                mapped_gf_tgt = gf_tgt @ map_r

                sc_rel = pdist(BASE.row_normalize(sc_tgt), metric="euclidean")
                mapped_rel = pdist(BASE.row_normalize(mapped_gf_tgt), metric="euclidean")
                baseline_rel = pdist(BASE.row_normalize(gf_tgt), metric="euclidean")

                spearman_mapped = safe_spearman(sc_rel, mapped_rel)
                spearman_baseline = safe_spearman(sc_rel, baseline_rel)
                delta_spearman = (
                    float(spearman_mapped - spearman_baseline)
                    if np.isfinite(spearman_mapped) and np.isfinite(spearman_baseline)
                    else float("nan")
                )

                n_pairs = int(sc_rel.size)
                topk = max(50, int(round(0.10 * n_pairs)))
                overlap_mapped = topk_overlap_smallest(sc_rel, mapped_rel, topk=topk)
                overlap_baseline = topk_overlap_smallest(sc_rel, baseline_rel, topk=topk)
                delta_overlap = (
                    float(overlap_mapped - overlap_baseline)
                    if np.isfinite(overlap_mapped) and np.isfinite(overlap_baseline)
                    else float("nan")
                )
                if not np.isfinite(delta_spearman):
                    continue

                rng = np.random.default_rng(31_771 + domain_index * 100 + split_index * 10 + layer)
                dim = mapped_gf_tgt.shape[1]

                null_symbol_perm = np.empty(H77_NULL_PERM, dtype=float)
                null_rand_basis = np.empty(H77_NULL_PERM, dtype=float)
                null_signature_destroy = np.empty(H77_NULL_PERM, dtype=float)

                for perm_idx in range(H77_NULL_PERM):
                    perm_rows = rng.permutation(mapped_gf_tgt.shape[0])
                    mapped_perm = mapped_gf_tgt[perm_rows]
                    rel_perm = pdist(BASE.row_normalize(mapped_perm), metric="euclidean")
                    spearman_perm = safe_spearman(sc_rel, rel_perm)
                    delta_perm = (
                        float(spearman_perm - spearman_baseline)
                        if np.isfinite(spearman_perm) and np.isfinite(spearman_baseline)
                        else float("nan")
                    )
                    null_symbol_perm[perm_idx] = delta_perm
                    null_rows.append(
                        {
                            "null_kind": "symbol_permutation",
                            "domain": target_domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_spearman": float(delta_perm),
                        }
                    )

                    rand_q = random_orthogonal_matrix(dim, rng)
                    mapped_rand = gf_tgt @ rand_q
                    rel_rand = pdist(BASE.row_normalize(mapped_rand), metric="euclidean")
                    spearman_rand = safe_spearman(sc_rel, rel_rand)
                    delta_rand = (
                        float(spearman_rand - spearman_baseline)
                        if np.isfinite(spearman_rand) and np.isfinite(spearman_baseline)
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
                            "null_delta_spearman": float(delta_rand),
                        }
                    )

                    dim_perm = rng.permutation(dim)
                    signs = rng.choice(np.array([-1.0, 1.0]), size=dim)
                    mapped_destroy = mapped_gf_tgt[:, dim_perm] * signs[None, :]
                    rel_destroy = pdist(BASE.row_normalize(mapped_destroy), metric="euclidean")
                    spearman_destroy = safe_spearman(sc_rel, rel_destroy)
                    delta_destroy = (
                        float(spearman_destroy - spearman_baseline)
                        if np.isfinite(spearman_destroy) and np.isfinite(spearman_baseline)
                        else float("nan")
                    )
                    null_signature_destroy[perm_idx] = delta_destroy
                    null_rows.append(
                        {
                            "null_kind": "signature_basis_destroy",
                            "domain": target_domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_spearman": float(delta_destroy),
                        }
                    )

                p_symbol = BASE.empirical_upper_tail_p(delta_spearman, null_symbol_perm)
                p_rand = BASE.empirical_upper_tail_p(delta_spearman, null_rand_basis)
                p_destroy = BASE.empirical_upper_tail_p(delta_spearman, null_signature_destroy)
                p_best = np.nanmin(np.array([p_symbol, p_rand, p_destroy], dtype=float))

                all_null = np.concatenate([null_symbol_perm, null_rand_basis, null_signature_destroy])
                all_null = all_null[np.isfinite(all_null)]
                null_q95 = float(np.quantile(all_null, 0.95)) if all_null.size else float("nan")
                null_gap = (
                    float(delta_spearman - null_q95)
                    if np.isfinite(delta_spearman) and np.isfinite(null_q95)
                    else float("nan")
                )

                rows.append(
                    {
                        "domain": target_domain,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_symbols_eval": int(len(split_symbols)),
                        "n_rel_pairs": int(n_pairs),
                        "spearman_mapped_vs_sc": float(spearman_mapped),
                        "spearman_baseline_vs_sc": float(spearman_baseline),
                        "delta_spearman_mapped_minus_baseline": float(delta_spearman),
                        "topk_overlap_mapped": float(overlap_mapped),
                        "topk_overlap_baseline": float(overlap_baseline),
                        "delta_topk_overlap": float(delta_overlap),
                        "null_q95": float(null_q95),
                        "null_gap_q95": float(null_gap),
                        "p_symbol_perm_upper": float(p_symbol),
                        "p_random_basis_upper": float(p_rand),
                        "p_signature_destroy_upper": float(p_destroy),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h77_relational_rank_agreement_by_domain_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h77_relational_rank_agreement_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_spearman_mapped_minus_baseline": float(
                        group["delta_spearman_mapped_minus_baseline"].mean()
                    ),
                    "mean_delta_topk_overlap": float(group["delta_topk_overlap"].mean()),
                    "mean_null_gap_q95": float(group["null_gap_q95"].mean()),
                    "fraction_delta_spearman_positive": float(
                        (group["delta_spearman_mapped_minus_baseline"] > 0.0).mean()
                    ),
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
    summary_path = ITER_DIR / "h77_relational_rank_agreement_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_spearman": float(by_row_df["delta_spearman_mapped_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "mean_null_gap_q95": float(by_row_df["null_gap_q95"].mean()) if not by_row_df.empty else float("nan"),
        "mean_delta_topk_overlap": float(by_row_df["delta_topk_overlap"].mean()) if not by_row_df.empty else float("nan"),
        "artifact_paths": {
            "by_domain_layer_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h78_geodesic_detour_elasticity(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map.get(H78_SEED_TAG)
        if run_dir is None:
            continue

        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H78_GENE_CAP))
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

            for layer in H78_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=31_780 + domain_index * 100 + split_index * 10 + layer,
                )

                euclidean = cdist(points_pca, points_pca, metric="euclidean")
                euclidean = np.clip(euclidean, 1e-8, None)

                geodesic_by_k: dict[int, np.ndarray] = {}
                for k in H78_NEIGHBORS_LIST:
                    geodesic_by_k[k] = BASE.geodesic_distance_matrix(points_pca, n_neighbors=k)

                edge_support = support_dir[source_local, target_local]
                edge_margin = np.abs(support_dir[source_local, target_local] - support_dir[target_local, source_local])

                geod_ref = geodesic_by_k[12][source_local, target_local]
                euclid_edge = euclidean[source_local, target_local]

                detour_stack = []
                for k in H78_NEIGHBORS_LIST:
                    geod = geodesic_by_k[k][source_local, target_local]
                    detour_stack.append(geod / np.clip(euclid_edge, 1e-8, None))
                detour_mat = np.column_stack(detour_stack)

                mean_detour = np.mean(detour_mat, axis=1)
                detour_span = np.max(detour_mat, axis=1) - np.min(detour_mat, axis=1)
                detour_cv = np.std(detour_mat, axis=1) / np.clip(np.abs(mean_detour), 1e-6, None)

                # Reuse triangle-defect statistics as a stable geometric anchor term.
                feature_bundle = BASE.multiscale_triangle_defect_features(
                    geodesic=geodesic_by_k[12],
                    source_local=source_local,
                    target_local=target_local,
                    k_values=H78_TRIANGLE_K,
                )

                baseline_score = (
                    BASE.zscore(-geod_ref)
                    + 0.75 * BASE.zscore(edge_support)
                    + 0.35 * BASE.zscore(edge_margin)
                )
                elasticity_score = (
                    baseline_score
                    + 0.30 * BASE.zscore(-detour_span)
                    + 0.20 * BASE.zscore(-detour_cv)
                    + 0.15 * BASE.zscore(-mean_detour)
                    + 0.10 * BASE.zscore(-feature_bundle["tail_mean"])
                )

                auc_baseline = BASE.safe_auc(labels, baseline_score)
                auc_elasticity = BASE.safe_auc(labels, elasticity_score)
                delta_auc = (
                    float(auc_elasticity - auc_baseline)
                    if np.isfinite(auc_elasticity) and np.isfinite(auc_baseline)
                    else float("nan")
                )

                bins = BASE.degree_bins(geod_ref, max_bins=6)
                rng = np.random.default_rng(31_781 + domain_index * 100 + split_index * 10 + layer)

                null_endpoint_swap = np.empty(H78_NULL_PERM, dtype=float)
                null_feature_shuffle = np.empty(H78_NULL_PERM, dtype=float)
                null_label = np.empty(H78_NULL_PERM, dtype=float)

                for perm_idx in range(H78_NULL_PERM):
                    target_swap = target_local.copy()
                    for b in np.unique(bins):
                        idx = np.where(bins == b)[0]
                        if idx.size > 1:
                            target_swap[idx] = rng.permutation(target_swap[idx])

                    geod_swap_stack = []
                    euclid_swap = euclidean[source_local, target_swap]
                    for k in H78_NEIGHBORS_LIST:
                        geod_swap = geodesic_by_k[k][source_local, target_swap]
                        geod_swap_stack.append(geod_swap / np.clip(euclid_swap, 1e-8, None))
                    detour_swap = np.column_stack(geod_swap_stack)

                    span_swap = np.max(detour_swap, axis=1) - np.min(detour_swap, axis=1)
                    cv_swap = np.std(detour_swap, axis=1) / np.clip(np.abs(np.mean(detour_swap, axis=1)), 1e-6, None)
                    mean_swap = np.mean(detour_swap, axis=1)

                    score_swap = (
                        baseline_score
                        + 0.30 * BASE.zscore(-span_swap)
                        + 0.20 * BASE.zscore(-cv_swap)
                        + 0.15 * BASE.zscore(-mean_swap)
                        + 0.10 * BASE.zscore(-feature_bundle["tail_mean"])
                    )
                    auc_swap = BASE.safe_auc(labels, score_swap)
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
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_swap),
                        }
                    )

                    span_shuffle = BASE.stratified_shuffle(detour_span, bins, rng)
                    cv_shuffle = BASE.stratified_shuffle(detour_cv, bins, rng)
                    mean_shuffle = BASE.stratified_shuffle(mean_detour, bins, rng)
                    tail_shuffle = BASE.stratified_shuffle(feature_bundle["tail_mean"], bins, rng)
                    score_shuffle = (
                        baseline_score
                        + 0.30 * BASE.zscore(-span_shuffle)
                        + 0.20 * BASE.zscore(-cv_shuffle)
                        + 0.15 * BASE.zscore(-mean_shuffle)
                        + 0.10 * BASE.zscore(-tail_shuffle)
                    )
                    auc_shuffle = BASE.safe_auc(labels, score_shuffle)
                    delta_shuffle = (
                        float(auc_shuffle - auc_baseline)
                        if np.isfinite(auc_shuffle) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_feature_shuffle[perm_idx] = delta_shuffle
                    null_rows.append(
                        {
                            "null_kind": "elasticity_feature_shuffle_within_geodesic_bins",
                            "domain": domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_shuffle),
                        }
                    )

                    labels_perm = BASE.stratified_shuffle(labels, bins, rng).astype(int)
                    auc_lp = BASE.safe_auc(labels_perm, elasticity_score)
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
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_lp),
                        }
                    )

                p_swap = BASE.empirical_upper_tail_p(delta_auc, null_endpoint_swap)
                p_shuffle = BASE.empirical_upper_tail_p(delta_auc, null_feature_shuffle)
                p_label = BASE.empirical_upper_tail_p(delta_auc, null_label)
                p_best = np.nanmin(np.array([p_swap, p_shuffle, p_label], dtype=float))

                all_null = np.concatenate([null_endpoint_swap, null_feature_shuffle, null_label])
                all_null = all_null[np.isfinite(all_null)]
                null_q95 = float(np.quantile(all_null, 0.95)) if all_null.size else float("nan")
                null_gap = float(delta_auc - null_q95) if np.isfinite(delta_auc) and np.isfinite(null_q95) else float("nan")

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": H78_SEED_TAG,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_baseline": float(auc_baseline),
                        "auc_detour_elasticity": float(auc_elasticity),
                        "delta_auc_elasticity_minus_baseline": float(delta_auc),
                        "mean_detour_ratio": float(np.mean(mean_detour)),
                        "mean_detour_span": float(np.mean(detour_span)),
                        "mean_detour_cv": float(np.mean(detour_cv)),
                        "null_q95": float(null_q95),
                        "null_gap_q95": float(null_gap),
                        "p_endpoint_swap_upper": float(p_swap),
                        "p_feature_shuffle_upper": float(p_shuffle),
                        "p_label_shuffle_upper": float(p_label),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h78_geodesic_detour_elasticity_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h78_geodesic_detour_elasticity_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_auc_detour_elasticity": float(group["auc_detour_elasticity"].mean()),
                    "mean_auc_baseline": float(group["auc_baseline"].mean()),
                    "mean_delta_auc_elasticity_minus_baseline": float(
                        group["delta_auc_elasticity_minus_baseline"].mean()
                    ),
                    "mean_null_gap_q95": float(group["null_gap_q95"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_elasticity_minus_baseline"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(
                        BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))
                    ),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h78_geodesic_detour_elasticity_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_elasticity_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_elasticity_minus_baseline"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "significant_domain_splits": int((summary_df["fraction_p_best_lt_0_05"] > 0.0).sum())
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

    h76_summary = run_h76_coexpression_support_interaction_v2(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h77_summary = run_h77_relational_rank_agreement()
    h78_summary = run_h78_geodesic_detour_elasticity(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    summary = {
        "iteration": "iter_0031",
        "h76": h76_summary,
        "h77": h77_summary,
        "h78": h78_summary,
        "environment": {
            "python_env": "subproject40-topology",
            "string_cache_used": str(BASE.STRING_CACHE_PATH),
        },
    }
    summary_path = ITER_DIR / "iter0031_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
