from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

ITER_DIR = Path("iterations/iter_0032")
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

# H79 / N395: TF-module conditioned support calibration (major-change rescue on H70 lineage).
H79_LAYERS = [7, 11]
H79_GENE_CAP = 170
H79_NEIGHBORS = 12
H79_TRIANGLE_K = [8, 12, 16]
H79_NULL_PERM = 24
H79_MIN_MODULE_EDGES = 50

# H80 / N392: pathway-centroid cross-model geometric alignment (edge-free objective).
H80_LAYERS = [7, 11]
H80_GENE_CAP = 220
H80_SPECTRAL_DIM = 8
H80_SPECTRAL_K = 12
H80_MIN_MODULE_SIZE = 8
H80_MAX_MODULE_SIZE = 80
H80_MAX_GO_MODULES = 24
H80_MAX_TRRUST_MODULES = 24
H80_NULL_PERM = 24

# H81 / N389: neighbor-dropout detour elasticity v2 (major-change rescue on H78 lineage).
H81_LAYERS = [0, 3, 7, 11]
H81_SEED_TAG = "seed42_main"
H81_GENE_CAP = 170
H81_NEIGHBORS = 12
H81_DROPOUT_RATES = [0.10, 0.20, 0.30]
H81_RANDOM_DROPOUT_REPEATS = 4
H81_NULL_PERM = 16


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


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x0 = np.asarray(x, dtype=float)
    y0 = np.asarray(y, dtype=float)
    if x0.shape[0] < 3 or y0.shape[0] < 3 or x0.shape[0] != y0.shape[0]:
        return float("nan")
    x0 = x0 - x0.mean(axis=0, keepdims=True)
    y0 = y0 - y0.mean(axis=0, keepdims=True)
    kx = x0 @ x0.T
    ky = y0 @ y0.T
    hsic = float(np.sum(kx * ky))
    norm = float(np.sqrt(np.sum(kx * kx) * np.sum(ky * ky)))
    if norm < 1e-12:
        return float("nan")
    return float(hsic / norm)


def top1_profile_retrieval(score_matrix: np.ndarray) -> float:
    s = np.asarray(score_matrix, dtype=float)
    if s.ndim != 2 or s.shape[0] == 0 or s.shape[0] != s.shape[1]:
        return float("nan")
    pred = np.argmax(s, axis=1)
    truth = np.arange(s.shape[0])
    return float(np.mean(pred == truth))


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


def load_dorothea_tf_targets() -> dict[str, set[str]]:
    dorothea = pd.read_csv(BASE.DOROTHEA_PATH, sep="\t")
    dorothea["source"] = dorothea["source"].astype(str).str.upper()
    dorothea["target"] = dorothea["target"].astype(str).str.upper()
    tf_targets: dict[str, set[str]] = {}
    for row in dorothea[["source", "target"]].drop_duplicates().itertuples(index=False):
        tf = str(row.source)
        tg = str(row.target)
        tf_targets.setdefault(tf, set()).add(tg)
    return tf_targets


def build_term_to_genes(
    symbols_upper: list[str],
    gene2go_upper: dict[str, set[str]],
) -> dict[str, set[str]]:
    term_to_genes: dict[str, set[str]] = {}
    for sym in symbols_upper:
        for term in gene2go_upper.get(sym, set()):
            term_to_genes.setdefault(term, set()).add(sym)
    return term_to_genes


def edge_source_density_tiers(
    symbols_upper: list[str],
    string_map: dict[tuple[str, str], float],
) -> dict[str, str]:
    density: dict[str, float] = {}
    for src in symbols_upper:
        vals = [string_map.get((src, tgt), 0.0) for tgt in symbols_upper if tgt != src]
        density[src] = float(np.mean(vals)) if vals else 0.0

    arr = np.array([density[s] for s in symbols_upper], dtype=float)
    q1 = float(np.quantile(arr, 1.0 / 3.0))
    q2 = float(np.quantile(arr, 2.0 / 3.0))

    out: dict[str, str] = {}
    for sym in symbols_upper:
        d = density[sym]
        if d <= q1:
            out[sym] = "low"
        elif d >= q2:
            out[sym] = "high"
        else:
            out[sym] = "mid"
    return out


def fit_module_weights(
    module_labels: np.ndarray,
    support_values: np.ndarray,
    labels: np.ndarray,
    min_edges: int,
) -> dict[str, float]:
    modules = np.asarray(module_labels)
    support = np.asarray(support_values, dtype=float)
    y = np.asarray(labels, dtype=int)

    weights: dict[str, float] = {}
    valid = np.isfinite(support)
    global_weight = 0.0
    if valid.sum() >= min_edges:
        lo_g = float(np.quantile(support[valid], 1.0 / 3.0))
        hi_g = float(np.quantile(support[valid], 2.0 / 3.0))
        hi_mask = valid & (support >= hi_g)
        lo_mask = valid & (support <= lo_g)
        if hi_mask.sum() >= 20 and lo_mask.sum() >= 20:
            global_weight = float(y[hi_mask].mean() - y[lo_mask].mean())

    for m in np.unique(modules):
        mask = modules == m
        if mask.sum() < min_edges:
            continue
        sup_m = support[mask]
        y_m = y[mask]
        finite = np.isfinite(sup_m)
        if finite.sum() < min_edges or np.unique(y_m[finite]).size < 2:
            continue
        lo = float(np.quantile(sup_m[finite], 1.0 / 3.0))
        hi = float(np.quantile(sup_m[finite], 2.0 / 3.0))
        hi_mask = finite & (sup_m >= hi)
        lo_mask = finite & (sup_m <= lo)
        if hi_mask.sum() < 20 or lo_mask.sum() < 20:
            continue
        weight = float(y_m[hi_mask].mean() - y_m[lo_mask].mean())
        weights[str(m)] = weight

    # Ensure every observed module has a defined calibration weight.
    for m in np.unique(modules):
        key = str(m)
        if key not in weights:
            weights[key] = global_weight
    return weights


def module_interaction_metric(
    module_labels: np.ndarray,
    support_values: np.ndarray,
    labels: np.ndarray,
    baseline_score: np.ndarray,
    defect_score: np.ndarray,
    min_edges: int,
) -> float:
    modules = np.asarray(module_labels)
    support = np.asarray(support_values, dtype=float)
    y = np.asarray(labels, dtype=int)
    baseline = np.asarray(baseline_score, dtype=float)
    defect = np.asarray(defect_score, dtype=float)

    interactions: list[float] = []
    weights: list[float] = []

    for m in np.unique(modules):
        mask = modules == m
        if mask.sum() < min_edges:
            continue
        sup_m = support[mask]
        finite = np.isfinite(sup_m)
        if finite.sum() < min_edges or np.unique(y[mask][finite]).size < 2:
            continue

        lo = float(np.quantile(sup_m[finite], 1.0 / 3.0))
        hi = float(np.quantile(sup_m[finite], 2.0 / 3.0))
        high = np.zeros(mask.size, dtype=bool)
        low = np.zeros(mask.size, dtype=bool)
        high_idx = np.where(mask)[0][finite & (sup_m >= hi)]
        low_idx = np.where(mask)[0][finite & (sup_m <= lo)]
        high[high_idx] = True
        low[low_idx] = True

        delta_high = subset_delta_auc(y, baseline, defect, high)
        delta_low = subset_delta_auc(y, baseline, defect, low)
        if np.isfinite(delta_high) and np.isfinite(delta_low):
            interactions.append(float(delta_high - delta_low))
            weights.append(float(mask.sum()))

    if not interactions:
        return float("nan")
    w = np.asarray(weights, dtype=float)
    v = np.asarray(interactions, dtype=float)
    return float(np.sum(w * v) / np.clip(np.sum(w), 1e-8, None))


def score_with_module_calibration(
    defect_score: np.ndarray,
    support_values: np.ndarray,
    module_labels: np.ndarray,
    module_weights: dict[str, float],
) -> np.ndarray:
    z_sup = BASE.zscore(support_values)
    out = np.asarray(defect_score, dtype=float).copy()
    labels = np.asarray(module_labels)
    for m in np.unique(labels):
        idx = np.where(labels == m)[0]
        w = float(module_weights.get(str(m), 0.0))
        out[idx] += w * z_sup[idx]
    return out


def local_betweenness_surrogate(points: np.ndarray, n_neighbors: int) -> np.ndarray:
    knn_edges = BASE.build_knn_edge_array(points, n_neighbors=n_neighbors)
    neigh = BASE.adjacency_neighbors(points.shape[0], knn_edges)
    centrality = np.zeros(points.shape[0], dtype=float)
    for i, nbrs in enumerate(neigh):
        nbr_list = list(nbrs)
        k = len(nbr_list)
        if k < 2:
            centrality[i] = 0.0
            continue
        missing = 0
        for a in range(k - 1):
            na = nbr_list[a]
            na_neigh = neigh[na]
            for b in range(a + 1, k):
                nb = nbr_list[b]
                if nb not in na_neigh:
                    missing += 1
        centrality[i] = float(missing)
    return centrality


def geodesic_with_dropped_nodes(
    points: np.ndarray,
    n_neighbors: int,
    drop_nodes: np.ndarray,
) -> np.ndarray:
    n = points.shape[0]
    keep = np.ones(n, dtype=bool)
    keep[np.asarray(drop_nodes, dtype=int)] = False
    kept_idx = np.where(keep)[0]

    out = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(out, 0.0)

    if kept_idx.size < max(20, n_neighbors + 2):
        return out

    sub = points[kept_idx]
    sub_geo = BASE.geodesic_distance_matrix(sub, n_neighbors=min(n_neighbors, kept_idx.size - 1))
    out[np.ix_(kept_idx, kept_idx)] = sub_geo
    return out


def mean_random_dropout_geodesic(
    points: np.ndarray,
    n_neighbors: int,
    n_drop: int,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = points.shape[0]
    accum = np.zeros((n, n), dtype=float)
    counts = np.zeros((n, n), dtype=float)

    nodes = np.arange(n, dtype=int)
    for _ in range(int(repeats)):
        drop = rng.choice(nodes, size=n_drop, replace=False)
        g = geodesic_with_dropped_nodes(points, n_neighbors=n_neighbors, drop_nodes=drop)
        finite = np.isfinite(g)
        accum[finite] += g[finite]
        counts[finite] += 1.0

    out = np.full((n, n), np.inf, dtype=float)
    ok = counts > 0
    out[ok] = accum[ok] / counts[ok]
    np.fill_diagonal(out, 0.0)
    return out


def inflation_from_distance(
    base_dist: np.ndarray,
    perturbed_dist: np.ndarray,
) -> np.ndarray:
    b = np.clip(np.asarray(base_dist, dtype=float), 1e-8, None)
    p = np.asarray(perturbed_dist, dtype=float)
    out = np.full(b.shape[0], np.nan, dtype=float)
    finite = np.isfinite(p)
    out[finite] = (p[finite] - b[finite]) / b[finite]
    out[finite] = np.clip(out[finite], 0.0, 8.0)
    return out


def columnwise_nanmean(arr2d: np.ndarray) -> np.ndarray:
    values = np.asarray(arr2d, dtype=float)
    valid = np.isfinite(values)
    numer = np.where(valid, values, 0.0).sum(axis=0)
    denom = valid.sum(axis=0)
    out = np.full(values.shape[1], np.nan, dtype=float)
    keep = denom > 0
    out[keep] = numer[keep] / denom[keep]
    return out


def run_h79_tf_module_conditioned_rescue(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
    tf_targets: dict[str, set[str]],
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

                top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H79_GENE_CAP))
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
                density_tier = edge_source_density_tiers(symbols, string_map)

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

                src_symbols = split_edges["source"].astype(str).str.upper().to_numpy()
                tgt_symbols = split_edges["target"].astype(str).str.upper().to_numpy()

                module_labels = []
                for s_sym, t_sym in zip(src_symbols, tgt_symbols):
                    tf_bucket = "tfsrc" if s_sym in tf_targets else "other"
                    if s_sym in tf_targets and t_sym in tf_targets[s_sym]:
                        pair_bucket = "in_module"
                    elif s_sym in tf_targets:
                        pair_bucket = "out_module"
                    else:
                        pair_bucket = "na"
                    tier = density_tier.get(s_sym, "mid")
                    module_labels.append(f"{tf_bucket}:{pair_bucket}:{tier}")
                module_labels = np.asarray(module_labels, dtype=object)

                for layer in H79_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = BASE.reduce_points(
                        points,
                        n_components=20,
                        random_state=32_790 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer,
                    )
                    geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H79_NEIGHBORS)

                    edge_geodesic = geodesic[source_local, target_local]
                    edge_support = support_dir[source_local, target_local]
                    edge_margin = np.abs(
                        support_dir[source_local, target_local] - support_dir[target_local, source_local]
                    )
                    support_concordance = edge_support - 0.40 * edge_margin

                    feature_bundle = BASE.multiscale_triangle_defect_features(
                        geodesic=geodesic,
                        source_local=source_local,
                        target_local=target_local,
                        k_values=H79_TRIANGLE_K,
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

                    module_weights = fit_module_weights(
                        module_labels=module_labels,
                        support_values=support_concordance,
                        labels=labels,
                        min_edges=H79_MIN_MODULE_EDGES,
                    )
                    module_score = score_with_module_calibration(
                        defect_score=defect_score,
                        support_values=support_concordance,
                        module_labels=module_labels,
                        module_weights=module_weights,
                    )

                    auc_defect = BASE.safe_auc(labels, defect_score)
                    auc_module = BASE.safe_auc(labels, module_score)
                    delta_auc = (
                        float(auc_module - auc_defect)
                        if np.isfinite(auc_module) and np.isfinite(auc_defect)
                        else float("nan")
                    )

                    interaction = module_interaction_metric(
                        module_labels=module_labels,
                        support_values=support_concordance,
                        labels=labels,
                        baseline_score=baseline_score,
                        defect_score=defect_score,
                        min_edges=H79_MIN_MODULE_EDGES,
                    )

                    edge_bins = BASE.degree_bins(edge_geodesic, max_bins=6)
                    rng = np.random.default_rng(
                        32_791 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer
                    )

                    null_module_perm = np.empty(H79_NULL_PERM, dtype=float)
                    null_weight_perm = np.empty(H79_NULL_PERM, dtype=float)
                    null_label = np.empty(H79_NULL_PERM, dtype=float)

                    unique_modules = sorted({str(x) for x in module_labels})
                    observed_weight_values = np.array([module_weights[m] for m in unique_modules], dtype=float)

                    for perm_idx in range(H79_NULL_PERM):
                        mod_perm = rng.permutation(module_labels)
                        w_perm = fit_module_weights(
                            module_labels=mod_perm,
                            support_values=support_concordance,
                            labels=labels,
                            min_edges=H79_MIN_MODULE_EDGES,
                        )
                        score_perm = score_with_module_calibration(
                            defect_score=defect_score,
                            support_values=support_concordance,
                            module_labels=mod_perm,
                            module_weights=w_perm,
                        )
                        auc_perm = BASE.safe_auc(labels, score_perm)
                        delta_perm = (
                            float(auc_perm - auc_defect)
                            if np.isfinite(auc_perm) and np.isfinite(auc_defect)
                            else float("nan")
                        )
                        null_module_perm[perm_idx] = delta_perm
                        null_rows.append(
                            {
                                "null_kind": "module_label_permutation",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_perm),
                            }
                        )

                        permuted_weights = rng.permutation(observed_weight_values)
                        weight_map_perm = {
                            mod: float(val) for mod, val in zip(unique_modules, permuted_weights, strict=False)
                        }
                        score_weight_perm = score_with_module_calibration(
                            defect_score=defect_score,
                            support_values=support_concordance,
                            module_labels=module_labels,
                            module_weights=weight_map_perm,
                        )
                        auc_weight_perm = BASE.safe_auc(labels, score_weight_perm)
                        delta_weight = (
                            float(auc_weight_perm - auc_defect)
                            if np.isfinite(auc_weight_perm) and np.isfinite(auc_defect)
                            else float("nan")
                        )
                        null_weight_perm[perm_idx] = delta_weight
                        null_rows.append(
                            {
                                "null_kind": "matched_random_module_weights",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_weight),
                            }
                        )

                        labels_perm = BASE.stratified_shuffle(labels, edge_bins, rng).astype(int)
                        w_label = fit_module_weights(
                            module_labels=module_labels,
                            support_values=support_concordance,
                            labels=labels_perm,
                            min_edges=H79_MIN_MODULE_EDGES,
                        )
                        score_label = score_with_module_calibration(
                            defect_score=defect_score,
                            support_values=support_concordance,
                            module_labels=module_labels,
                            module_weights=w_label,
                        )
                        auc_lp_mod = BASE.safe_auc(labels_perm, score_label)
                        auc_lp_def = BASE.safe_auc(labels_perm, defect_score)
                        delta_lp = (
                            float(auc_lp_mod - auc_lp_def)
                            if np.isfinite(auc_lp_mod) and np.isfinite(auc_lp_def)
                            else float("nan")
                        )
                        null_label[perm_idx] = delta_lp
                        null_rows.append(
                            {
                                "null_kind": "label_shuffle_within_geodesic_bins",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_lp),
                            }
                        )

                    p_mod = BASE.empirical_upper_tail_p(delta_auc, null_module_perm)
                    p_w = BASE.empirical_upper_tail_p(delta_auc, null_weight_perm)
                    p_l = BASE.empirical_upper_tail_p(delta_auc, null_label)
                    p_best = np.nanmin(np.array([p_mod, p_w, p_l], dtype=float))

                    all_null = np.concatenate([null_module_perm, null_weight_perm, null_label])
                    all_null = all_null[np.isfinite(all_null)]
                    q95 = float(np.quantile(all_null, 0.95)) if all_null.size else float("nan")
                    null_gap = float(delta_auc - q95) if np.isfinite(delta_auc) and np.isfinite(q95) else float("nan")

                    rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "n_edges_eval": int(labels.size),
                            "n_positive_eval": int(labels.sum()),
                            "n_modules": int(len(unique_modules)),
                            "auc_defect": float(auc_defect),
                            "auc_module_calibrated": float(auc_module),
                            "delta_auc_module_minus_defect": float(delta_auc),
                            "interaction_delta_high_minus_low": float(interaction),
                            "mean_module_weight": float(np.mean(list(module_weights.values()))),
                            "q95_null": float(q95),
                            "null_gap_q95": float(null_gap),
                            "p_module_perm_upper": float(p_mod),
                            "p_weight_perm_upper": float(p_w),
                            "p_label_upper": float(p_l),
                            "p_best_upper": float(p_best),
                        }
                    )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h79_tf_module_conditioned_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h79_tf_module_conditioned_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_module_minus_defect": float(group["delta_auc_module_minus_defect"].mean()),
                    "mean_interaction_delta_high_minus_low": float(
                        group["interaction_delta_high_minus_low"].mean()
                    ),
                    "mean_null_gap_q95": float(group["null_gap_q95"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_module_minus_defect"] > 0.0).mean()),
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
    summary_path = ITER_DIR / "h79_tf_module_conditioned_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_module_minus_defect"].mean())
        if not by_row_df.empty
        else float("nan"),
        "immune_source_delta_auc": float(
            summary_df.loc[
                (summary_df["domain"] == "immune") & (summary_df["split_regime"] == "source_disjoint"),
                "mean_delta_auc_module_minus_defect",
            ].iloc[0]
        )
        if not summary_df.empty
        and ((summary_df["domain"] == "immune") & (summary_df["split_regime"] == "source_disjoint")).any()
        else float("nan"),
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_layer_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h80_pathway_centroid_alignment(
    gene2go_upper: dict[str, set[str]],
    tf_targets: dict[str, set[str]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    domains = ["immune", "lung", "external_lung"]

    sc_edges: dict[str, pd.DataFrame] = {}
    sc_layers: dict[str, np.ndarray] = {}
    gf_edges: dict[str, pd.DataFrame] = {}

    for domain in domains:
        sc_run = BASE.SCGPT_RUNS_BY_DOMAIN[domain]["seed42_main"]
        sc_edges[domain] = pd.read_csv(sc_run / "cycle1_edge_dataset.tsv", sep="\t")
        sc_layers[domain] = np.load(sc_run / "layer_gene_embeddings.npy", mmap_mode="r")
        gf_edges[domain] = pd.read_csv(BASE.GENEFORMER_EDGE_BY_DOMAIN[domain], sep="\t")

    for domain_index, domain in enumerate(domains):
        sc_df = sc_edges[domain].copy()
        top_genes = set(BASE.select_top_genes(sc_df, gene_cap=H80_GENE_CAP))
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

        gf_sig_full = BASE.fit_signatures_geneformer(gf_edges[domain], symbols)

        for layer in H80_LAYERS:
            if layer >= sc_layers[domain].shape[0]:
                continue

            points = sc_layers[domain][layer, gene_indices, :]
            sc_sig = BASE.fit_signatures_scgpt(
                layer_points=points,
                symbols=symbols,
                random_state=32_800 + domain_index * 100 + layer,
                n_neighbors=10,
            )

            shared = sorted(set(sc_sig.index) & set(gf_sig_full.index))
            if len(shared) < 100:
                continue

            sc_spec = spectral_embedding_from_signatures(
                sc_sig.loc[shared],
                n_components=H80_SPECTRAL_DIM,
                n_neighbors=H80_SPECTRAL_K,
            )
            gf_spec = spectral_embedding_from_signatures(
                gf_sig_full.loc[shared],
                n_components=H80_SPECTRAL_DIM,
                n_neighbors=H80_SPECTRAL_K,
            )

            shared_set = set(shared)
            term_to_genes = build_term_to_genes(shared, gene2go_upper)

            modules: list[tuple[str, list[str]]] = []

            # TRRUST-driven TF modules.
            trrust_candidates = []
            for tf, tgts in tf_targets.items():
                genes = sorted(({tf} | set(tgts)) & shared_set)
                if H80_MIN_MODULE_SIZE <= len(genes) <= H80_MAX_MODULE_SIZE:
                    trrust_candidates.append((tf, genes))
            trrust_candidates = sorted(trrust_candidates, key=lambda x: (-len(x[1]), x[0]))
            for tf, genes in trrust_candidates[:H80_MAX_TRRUST_MODULES]:
                modules.append((f"TRRUST::{tf}", genes))

            # GO-driven modules.
            go_candidates = []
            for term, genes in term_to_genes.items():
                if H80_MIN_MODULE_SIZE <= len(genes) <= H80_MAX_MODULE_SIZE:
                    go_candidates.append((term, sorted(genes)))
            go_candidates = sorted(go_candidates, key=lambda x: (-len(x[1]), x[0]))
            for term, genes in go_candidates[:H80_MAX_GO_MODULES]:
                modules.append((f"GO::{term}", genes))

            # De-duplicate exact gene sets to avoid inflated correspondence.
            deduped: list[tuple[str, list[str]]] = []
            seen_sets: set[tuple[str, ...]] = set()
            for name, genes in modules:
                key = tuple(genes)
                if key in seen_sets:
                    continue
                seen_sets.add(key)
                deduped.append((name, genes))
            modules = deduped

            if len(modules) < 12:
                continue

            sc_arr = sc_spec.to_numpy(dtype=float)
            gf_arr = gf_spec.to_numpy(dtype=float)
            idx_map = {sym: i for i, sym in enumerate(shared)}

            sc_centroids = []
            gf_centroids = []
            module_names = []
            module_sizes = []
            module_genes_used: list[list[str]] = []

            for name, genes in modules:
                loc = [idx_map[g] for g in genes if g in idx_map]
                if len(loc) < H80_MIN_MODULE_SIZE:
                    continue
                sc_centroids.append(sc_arr[loc].mean(axis=0))
                gf_centroids.append(gf_arr[loc].mean(axis=0))
                module_names.append(name)
                module_sizes.append(len(loc))
                module_genes_used.append(genes)

            if len(sc_centroids) < 12:
                continue

            sc_centroids_arr = np.asarray(sc_centroids, dtype=float)
            gf_centroids_arr = np.asarray(gf_centroids, dtype=float)

            d_sc = cdist(sc_centroids_arr, sc_centroids_arr, metric="euclidean")
            d_gf = cdist(gf_centroids_arr, gf_centroids_arr, metric="euclidean")
            upper_sc = d_sc[np.triu_indices(d_sc.shape[0], k=1)]
            upper_gf = d_gf[np.triu_indices(d_gf.shape[0], k=1)]

            spearman = safe_spearman(upper_sc, upper_gf)
            cka = linear_cka(sc_centroids_arr, gf_centroids_arr)

            sc_profile = BASE.row_normalize(d_sc)
            gf_profile = BASE.row_normalize(d_gf)
            profile_score = sc_profile @ gf_profile.T
            top1 = top1_profile_retrieval(profile_score)

            rng = np.random.default_rng(32_801 + domain_index * 100 + layer)
            null_label_perm = np.empty(H80_NULL_PERM, dtype=float)
            null_membership_perm = np.empty(H80_NULL_PERM, dtype=float)
            null_signature_destroy = np.empty(H80_NULL_PERM, dtype=float)

            genes_all = np.array(shared, dtype=object)
            module_sizes_arr = np.asarray(module_sizes, dtype=int)

            for perm_idx in range(H80_NULL_PERM):
                perm = rng.permutation(len(module_names))
                d_gf_perm = d_gf[perm][:, perm]
                upper_perm = d_gf_perm[np.triu_indices(d_gf_perm.shape[0], k=1)]
                spearman_perm = safe_spearman(upper_sc, upper_perm)
                null_label_perm[perm_idx] = spearman_perm
                null_rows.append(
                    {
                        "null_kind": "module_label_permutation",
                        "domain": domain,
                        "layer": int(layer),
                        "perm_idx": int(perm_idx),
                        "null_spearman": float(spearman_perm),
                    }
                )

                rand_centroids = []
                for size in module_sizes_arr:
                    chosen = rng.choice(genes_all, size=int(size), replace=False)
                    loc = np.array([idx_map[g] for g in chosen], dtype=int)
                    rand_centroids.append(gf_arr[loc].mean(axis=0))
                rand_centroids_arr = np.asarray(rand_centroids, dtype=float)
                d_rand = cdist(rand_centroids_arr, rand_centroids_arr, metric="euclidean")
                upper_rand = d_rand[np.triu_indices(d_rand.shape[0], k=1)]
                spearman_rand = safe_spearman(upper_sc, upper_rand)
                null_membership_perm[perm_idx] = spearman_rand
                null_rows.append(
                    {
                        "null_kind": "module_membership_permutation",
                        "domain": domain,
                        "layer": int(layer),
                        "perm_idx": int(perm_idx),
                        "null_spearman": float(spearman_rand),
                    }
                )

                gf_destroy = gf_arr.copy()
                for col in range(gf_destroy.shape[1]):
                    gf_destroy[:, col] = gf_destroy[rng.permutation(gf_destroy.shape[0]), col]
                gf_destroy_centroids = []
                for genes in module_genes_used:
                    loc = np.array([idx_map[g] for g in genes if g in idx_map], dtype=int)
                    gf_destroy_centroids.append(gf_destroy[loc].mean(axis=0))
                gf_destroy_centroids_arr = np.asarray(gf_destroy_centroids, dtype=float)
                d_destroy = cdist(gf_destroy_centroids_arr, gf_destroy_centroids_arr, metric="euclidean")
                upper_destroy = d_destroy[np.triu_indices(d_destroy.shape[0], k=1)]
                spearman_destroy = safe_spearman(upper_sc, upper_destroy)
                null_signature_destroy[perm_idx] = spearman_destroy
                null_rows.append(
                    {
                        "null_kind": "signature_destroy_permutation",
                        "domain": domain,
                        "layer": int(layer),
                        "perm_idx": int(perm_idx),
                        "null_spearman": float(spearman_destroy),
                    }
                )

            p_label = BASE.empirical_upper_tail_p(spearman, null_label_perm)
            p_membership = BASE.empirical_upper_tail_p(spearman, null_membership_perm)
            p_destroy = BASE.empirical_upper_tail_p(spearman, null_signature_destroy)
            p_best = np.nanmin(np.array([p_label, p_membership, p_destroy], dtype=float))

            all_null = np.concatenate([null_label_perm, null_membership_perm, null_signature_destroy])
            all_null = all_null[np.isfinite(all_null)]
            q95 = float(np.quantile(all_null, 0.95)) if all_null.size else float("nan")
            null_gap = float(spearman - q95) if np.isfinite(spearman) and np.isfinite(q95) else float("nan")

            rows.append(
                {
                    "domain": domain,
                    "split_regime": "heldout_domain_seed42",
                    "layer": int(layer),
                    "n_shared_genes": int(len(shared)),
                    "n_modules": int(len(module_names)),
                    "mean_module_size": float(np.mean(module_sizes_arr)),
                    "spearman_centroid_distance": float(spearman),
                    "cka_centroid_geometry": float(cka),
                    "top1_profile_retrieval": float(top1),
                    "q95_null_spearman": float(q95),
                    "null_gap_q95_spearman": float(null_gap),
                    "p_label_perm_upper": float(p_label),
                    "p_membership_perm_upper": float(p_membership),
                    "p_signature_destroy_upper": float(p_destroy),
                    "p_best_upper": float(p_best),
                }
            )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "layer"])
    by_row_path = ITER_DIR / "h80_pathway_centroid_alignment_by_domain_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "layer", "perm_idx"])
    null_path = ITER_DIR / "h80_pathway_centroid_alignment_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for domain, group in by_row_df.groupby("domain", sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "n_rows": int(group.shape[0]),
                    "mean_spearman_centroid_distance": float(group["spearman_centroid_distance"].mean()),
                    "mean_cka_centroid_geometry": float(group["cka_centroid_geometry"].mean()),
                    "mean_top1_profile_retrieval": float(group["top1_profile_retrieval"].mean()),
                    "mean_null_gap_q95_spearman": float(group["null_gap_q95_spearman"].mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_spearman"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(
                        BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))
                    ),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain"])
    summary_path = ITER_DIR / "h80_pathway_centroid_alignment_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_spearman": float(by_row_df["spearman_centroid_distance"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_null_gap_domains": int((summary_df["mean_null_gap_q95_spearman"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_domain_layer": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h81_neighbor_dropout_detour_elasticity(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map.get(H81_SEED_TAG)
        if run_dir is None:
            continue

        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H81_GENE_CAP))
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

            for layer in H81_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=32_810 + domain_index * 100 + split_index * 10 + layer,
                )

                base_geo = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H81_NEIGHBORS)
                edge_geodesic = base_geo[source_local, target_local]
                edge_support = support_dir[source_local, target_local]
                edge_margin = np.abs(support_dir[source_local, target_local] - support_dir[target_local, source_local])

                baseline = (
                    BASE.zscore(-edge_geodesic)
                    + 0.75 * BASE.zscore(edge_support)
                    + 0.35 * BASE.zscore(edge_margin)
                )

                centrality = local_betweenness_surrogate(points_pca, n_neighbors=H81_NEIGHBORS)
                order_desc = np.argsort(centrality)[::-1]

                geo_target_by_rate: list[np.ndarray] = []
                geo_random_by_rate: list[np.ndarray] = []

                rng = np.random.default_rng(32_811 + domain_index * 100 + split_index * 10 + layer)
                n_nodes = points_pca.shape[0]

                for rate in H81_DROPOUT_RATES:
                    n_drop = max(1, int(np.floor(rate * n_nodes)))
                    drop_target = order_desc[:n_drop]

                    geo_target = geodesic_with_dropped_nodes(
                        points_pca,
                        n_neighbors=H81_NEIGHBORS,
                        drop_nodes=drop_target,
                    )
                    geo_random = mean_random_dropout_geodesic(
                        points_pca,
                        n_neighbors=H81_NEIGHBORS,
                        n_drop=n_drop,
                        repeats=H81_RANDOM_DROPOUT_REPEATS,
                        rng=rng,
                    )

                    geo_target_by_rate.append(geo_target)
                    geo_random_by_rate.append(geo_random)

                target_rows = []
                random_rows = []
                for geo_target, geo_random in zip(geo_target_by_rate, geo_random_by_rate, strict=False):
                    target_rows.append(
                        inflation_from_distance(
                            base_geo[source_local, target_local],
                            geo_target[source_local, target_local],
                        )
                    )
                    random_rows.append(
                        inflation_from_distance(
                            base_geo[source_local, target_local],
                            geo_random[source_local, target_local],
                        )
                    )

                target_arr = np.vstack(target_rows)
                random_arr = np.vstack(random_rows)

                target_mean = columnwise_nanmean(target_arr)
                random_mean = columnwise_nanmean(random_arr)
                target_high = target_arr[-1]
                target_slope = target_arr[-1] - target_arr[0]
                advantage = random_mean - target_mean

                # Replace NaNs with conservative neutral values for score construction.
                for arr in [target_mean, random_mean, target_high, target_slope, advantage]:
                    if np.isnan(arr).any():
                        fill = float(np.nanmedian(arr)) if np.isfinite(arr).any() else 0.0
                        arr[np.isnan(arr)] = fill

                dropout_component = (
                    0.35 * BASE.zscore(-target_mean)
                    + 0.30 * BASE.zscore(advantage)
                    + 0.20 * BASE.zscore(-target_slope)
                    + 0.15 * BASE.zscore(-target_high)
                )
                dropout_score = baseline + dropout_component

                auc_base = BASE.safe_auc(labels, baseline)
                auc_drop = BASE.safe_auc(labels, dropout_score)
                delta_auc = (
                    float(auc_drop - auc_base)
                    if np.isfinite(auc_drop) and np.isfinite(auc_base)
                    else float("nan")
                )

                edge_bins = BASE.degree_bins(edge_geodesic, max_bins=6)
                null_endpoint_swap = np.empty(H81_NULL_PERM, dtype=float)
                null_feature_shuffle = np.empty(H81_NULL_PERM, dtype=float)
                null_label = np.empty(H81_NULL_PERM, dtype=float)

                for perm_idx in range(H81_NULL_PERM):
                    # Endpoint swap within geodesic bins.
                    tgt_swap = target_local.copy()
                    for b in np.unique(edge_bins):
                        idx = np.where(edge_bins == b)[0]
                        if idx.size > 1:
                            tgt_swap[idx] = rng.permutation(tgt_swap[idx])

                    swap_target_rates = []
                    swap_random_rates = []
                    for geo_target, geo_random in zip(geo_target_by_rate, geo_random_by_rate, strict=False):
                        swap_target_rates.append(
                            inflation_from_distance(
                                base_geo[source_local, tgt_swap],
                                geo_target[source_local, tgt_swap],
                            )
                        )
                        swap_random_rates.append(
                            inflation_from_distance(
                                base_geo[source_local, tgt_swap],
                                geo_random[source_local, tgt_swap],
                            )
                        )
                    swap_target = columnwise_nanmean(np.vstack(swap_target_rates))
                    swap_random = columnwise_nanmean(np.vstack(swap_random_rates))
                    swap_adv = swap_random - swap_target
                    for arr in [swap_target, swap_adv]:
                        if np.isnan(arr).any():
                            fill = float(np.nanmedian(arr)) if np.isfinite(arr).any() else 0.0
                            arr[np.isnan(arr)] = fill
                    swap_component = 0.6 * BASE.zscore(-swap_target) + 0.4 * BASE.zscore(swap_adv)
                    score_swap = baseline + swap_component
                    auc_swap = BASE.safe_auc(labels, score_swap)
                    delta_swap = (
                        float(auc_swap - auc_base)
                        if np.isfinite(auc_swap) and np.isfinite(auc_base)
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

                    comp_perm = BASE.stratified_shuffle(dropout_component, edge_bins, rng)
                    score_perm = baseline + comp_perm
                    auc_perm = BASE.safe_auc(labels, score_perm)
                    delta_perm = (
                        float(auc_perm - auc_base)
                        if np.isfinite(auc_perm) and np.isfinite(auc_base)
                        else float("nan")
                    )
                    null_feature_shuffle[perm_idx] = delta_perm
                    null_rows.append(
                        {
                            "null_kind": "feature_shuffle_within_geodesic_bins",
                            "domain": domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_perm),
                        }
                    )

                    labels_perm = BASE.stratified_shuffle(labels, edge_bins, rng).astype(int)
                    auc_lp_drop = BASE.safe_auc(labels_perm, dropout_score)
                    auc_lp_base = BASE.safe_auc(labels_perm, baseline)
                    delta_lp = (
                        float(auc_lp_drop - auc_lp_base)
                        if np.isfinite(auc_lp_drop) and np.isfinite(auc_lp_base)
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
                p_feat = BASE.empirical_upper_tail_p(delta_auc, null_feature_shuffle)
                p_lab = BASE.empirical_upper_tail_p(delta_auc, null_label)
                p_best = np.nanmin(np.array([p_swap, p_feat, p_lab], dtype=float))

                all_null = np.concatenate([null_endpoint_swap, null_feature_shuffle, null_label])
                all_null = all_null[np.isfinite(all_null)]
                q95 = float(np.quantile(all_null, 0.95)) if all_null.size else float("nan")
                null_gap = float(delta_auc - q95) if np.isfinite(delta_auc) and np.isfinite(q95) else float("nan")

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": H81_SEED_TAG,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_baseline": float(auc_base),
                        "auc_dropout_elasticity": float(auc_drop),
                        "delta_auc_dropout_minus_baseline": float(delta_auc),
                        "mean_targeted_inflation": float(np.mean(target_mean)),
                        "mean_random_inflation": float(np.mean(random_mean)),
                        "mean_advantage_random_minus_targeted": float(np.mean(advantage)),
                        "mean_targeted_slope_30_minus_10": float(np.mean(target_slope)),
                        "q95_null": float(q95),
                        "null_gap_q95": float(null_gap),
                        "p_swap_upper": float(p_swap),
                        "p_feature_shuffle_upper": float(p_feat),
                        "p_label_upper": float(p_lab),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h81_neighbor_dropout_detour_elasticity_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h81_neighbor_dropout_detour_elasticity_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_dropout_minus_baseline": float(
                        group["delta_auc_dropout_minus_baseline"].mean()
                    ),
                    "mean_null_gap_q95": float(group["null_gap_q95"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_dropout_minus_baseline"] > 0.0).mean()),
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
    summary_path = ITER_DIR / "h81_neighbor_dropout_detour_elasticity_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_dropout_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_dropout_minus_baseline"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95"] > 0.0).sum())
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
    tf_targets = load_dorothea_tf_targets()

    h79_summary = run_h79_tf_module_conditioned_rescue(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
        tf_targets=tf_targets,
    )
    h80_summary = run_h80_pathway_centroid_alignment(
        gene2go_upper=gene2go_upper,
        tf_targets=tf_targets,
    )
    h81_summary = run_h81_neighbor_dropout_detour_elasticity(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    summary = {
        "iteration": "iter_0032",
        "h79": h79_summary,
        "h80": h80_summary,
        "h81": h81_summary,
    }

    summary_path = ITER_DIR / "iter0032_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
