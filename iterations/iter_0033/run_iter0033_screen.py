from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

ITER_DIR = Path("iterations/iter_0033")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base")

# H82 / N399: Local witness-cycle persistence on H70 hotspots.
H82_LAYERS = [7, 11]
H82_GENE_CAP = 170
H82_NEIGHBORS = 12
H82_TRIANGLE_K = [8, 12, 16]
H82_EDGE_SAMPLE = 300
H82_LOCAL_NEIGHBOR_K = 12
H82_FILTRATION_QUANTILES = [0.20, 0.35, 0.50, 0.65, 0.80]
H82_NULL_PERM = 24
H82_WEIGHT = 0.30

# H83 / N407: Cross-model pathway trajectory invariance.
H83_LAYERS = [0, 3, 7, 11]
H83_GENE_CAP = 220
H83_SPECTRAL_DIM = 8
H83_SPECTRAL_K = 12
H83_MIN_MODULE_SIZE = 8
H83_MAX_MODULE_SIZE = 70
H83_MAX_GO_MODULES = 20
H83_MAX_TRRUST_MODULES = 20
H83_NULL_PERM = 24

# H84 / N412: Shortcut-bridge competition index.
H84_LAYERS = [0, 3, 7, 11]
H84_GENE_CAP = 170
H84_NEIGHBORS = 12
H84_TRIANGLE_K = [8, 12, 16]
H84_NULL_PERM = 24
H84_WEIGHT = 0.35


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


def build_weighted_knn_graph(points: np.ndarray, n_neighbors: int) -> np.ndarray:
    n = points.shape[0]
    k = max(2, min(int(n_neighbors), n - 1))
    nbrs = BASE.NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
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


def cycle_rank(n_nodes: int, edges: np.ndarray) -> float:
    if n_nodes <= 1:
        return 0.0
    if edges.size == 0:
        n_comp = n_nodes
    else:
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.ones(rows.size, dtype=np.float64)
        graph = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        n_comp, _ = connected_components(graph, directed=False, connection="weak")
    e_count = int(edges.shape[0])
    beta = max(0, e_count - n_nodes + int(n_comp))
    return float(beta / max(1, n_nodes))


def local_cycle_persistence_surrogate(points_local: np.ndarray, q_grid: list[float]) -> tuple[float, float]:
    n = int(points_local.shape[0])
    if n < 8:
        return float("nan"), float("nan")

    d = cdist(points_local, points_local, metric="euclidean")
    tri = d[np.triu_indices(n, k=1)]
    tri = tri[np.isfinite(tri)]
    if tri.size < 8:
        return float("nan"), float("nan")

    betti_vals = []
    for q in q_grid:
        threshold = float(np.quantile(tri, q))
        mask = np.triu(d <= threshold, k=1)
        edges = np.argwhere(mask)
        betti_vals.append(cycle_rank(n_nodes=n, edges=edges))
    b = np.asarray(betti_vals, dtype=float)
    if b.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(b)), float(np.max(b))


def edge_local_cycle_features(
    points_pca: np.ndarray,
    geodesic: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    n_local_neighbors: int,
    q_grid: list[float],
) -> np.ndarray:
    n = points_pca.shape[0]
    nodes = np.arange(n, dtype=int)
    out = np.full(source_local.shape[0], np.nan, dtype=float)

    for i, (u, v) in enumerate(zip(source_local, target_local)):
        uu = int(u)
        vv = int(v)
        if uu == vv:
            continue
        finite_mask = np.isfinite(geodesic[uu]) & np.isfinite(geodesic[vv])
        finite_nodes = nodes[finite_mask]
        if finite_nodes.size < 10:
            continue
        order = np.argsort(geodesic[uu, finite_nodes] + geodesic[vv, finite_nodes])
        use = finite_nodes[order[: min(finite_nodes.size, 2 * n_local_neighbors)]]
        local_nodes = np.unique(np.concatenate([[uu, vv], use]))
        if local_nodes.size < 8:
            continue
        auc, _ = local_cycle_persistence_surrogate(points_local=points_pca[local_nodes], q_grid=q_grid)
        out[i] = auc
    return out


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


def matched_control_indices(
    hotspot_idx: np.ndarray,
    bins: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    h = np.asarray(hotspot_idx, dtype=int)
    y = np.asarray(labels, dtype=int)
    b = np.asarray(bins, dtype=int)
    all_idx = np.arange(y.size, dtype=int)
    h_set = set(h.tolist())
    controls = []
    for idx in h:
        same = all_idx[(b == b[idx]) & (y == y[idx])]
        same = np.array([j for j in same if j not in h_set], dtype=int)
        if same.size == 0:
            same = all_idx[b == b[idx]]
            same = np.array([j for j in same if j not in h_set], dtype=int)
        if same.size == 0:
            same = np.array([j for j in all_idx if j not in h_set], dtype=int)
        if same.size == 0:
            continue
        controls.append(int(rng.choice(same)))
    if not controls:
        return np.array([], dtype=int)
    return np.asarray(controls, dtype=int)


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


def build_h83_modules(
    shared_symbols: list[str],
    gene2go_upper: dict[str, set[str]],
    tf_targets: dict[str, set[str]],
) -> list[tuple[str, list[str]]]:
    shared_set = set(shared_symbols)
    modules: list[tuple[str, list[str]]] = []

    trrust_candidates = []
    for tf, tgts in tf_targets.items():
        genes = sorted(({tf} | set(tgts)) & shared_set)
        if H83_MIN_MODULE_SIZE <= len(genes) <= H83_MAX_MODULE_SIZE:
            trrust_candidates.append((tf, genes))
    trrust_candidates = sorted(trrust_candidates, key=lambda x: (-len(x[1]), x[0]))
    for tf, genes in trrust_candidates[:H83_MAX_TRRUST_MODULES]:
        modules.append((f"TRRUST::{tf}", genes))

    term_to_genes = build_term_to_genes(shared_symbols, gene2go_upper)
    go_candidates = []
    for term, genes in term_to_genes.items():
        if H83_MIN_MODULE_SIZE <= len(genes) <= H83_MAX_MODULE_SIZE:
            go_candidates.append((term, sorted(genes)))
    go_candidates = sorted(go_candidates, key=lambda x: (-len(x[1]), x[0]))
    for term, genes in go_candidates[:H83_MAX_GO_MODULES]:
        modules.append((f"GO::{term}", genes))

    deduped: list[tuple[str, list[str]]] = []
    seen_sets: set[tuple[str, ...]] = set()
    for name, genes in modules:
        key = tuple(genes)
        if key in seen_sets:
            continue
        seen_sets.add(key)
        deduped.append((name, genes))
    return deduped


def module_distance_matrices(
    embeddings_by_depth: list[pd.DataFrame],
    modules: list[tuple[str, list[str]]],
    shared_symbols: list[str],
) -> tuple[list[np.ndarray], np.ndarray]:
    idx_map = {g: i for i, g in enumerate(shared_symbols)}
    valid_modules = []
    module_sizes = []
    module_loc = []

    for name, genes in modules:
        loc = [idx_map[g] for g in genes if g in idx_map]
        if len(loc) < H83_MIN_MODULE_SIZE:
            continue
        valid_modules.append(name)
        module_sizes.append(len(loc))
        module_loc.append(np.asarray(loc, dtype=int))

    if len(valid_modules) < 10:
        return [], np.array([], dtype=int)

    depth_mats = []
    for emb_df in embeddings_by_depth:
        arr = emb_df.loc[shared_symbols].to_numpy(dtype=float)
        centroids = np.asarray([arr[loc].mean(axis=0) for loc in module_loc], dtype=float)
        d = cdist(centroids, centroids, metric="euclidean")
        depth_mats.append(d)
    return depth_mats, np.asarray(module_sizes, dtype=int)


def trajectory_concordance_from_distances(
    dist_sc: list[np.ndarray],
    dist_gf: list[np.ndarray],
) -> tuple[float, float, float]:
    if len(dist_sc) != len(dist_gf) or len(dist_sc) == 0:
        return float("nan"), float("nan"), float("nan")
    n_depth = len(dist_sc)
    n_modules = dist_sc[0].shape[0]
    if n_modules < 3:
        return float("nan"), float("nan"), float("nan")

    pair_traj_sc = []
    pair_traj_gf = []
    pair_rhos = []
    for i in range(n_modules - 1):
        for j in range(i + 1, n_modules):
            s = np.array([dist_sc[d][i, j] for d in range(n_depth)], dtype=float)
            g = np.array([dist_gf[d][i, j] for d in range(n_depth)], dtype=float)
            pair_traj_sc.append(s)
            pair_traj_gf.append(g)
            pair_rhos.append(safe_spearman(s, g))

    sc_mat = np.asarray(pair_traj_sc, dtype=float)
    gf_mat = np.asarray(pair_traj_gf, dtype=float)
    rho_mean = float(np.nanmean(np.asarray(pair_rhos, dtype=float)))
    cka = linear_cka(sc_mat, gf_mat)

    retrieval_vals = []
    for d in range(n_depth):
        sc_profile = BASE.row_normalize(dist_sc[d])
        gf_profile = BASE.row_normalize(dist_gf[d])
        profile_score = sc_profile @ gf_profile.T
        retrieval_vals.append(top1_profile_retrieval(profile_score))
    top1 = float(np.nanmean(np.asarray(retrieval_vals, dtype=float)))
    return rho_mean, cka, top1


def run_h82_local_witness_cycle_persistence(
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

                top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H82_GENE_CAP))
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

                for layer in H82_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    rng = np.random.default_rng(
                        33_820 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer
                    )
                    sample_idx = stratified_index_sample(labels_all, max_n=H82_EDGE_SAMPLE, rng=rng)
                    if sample_idx.size < 120:
                        continue

                    source_local = source_local_all[sample_idx]
                    target_local = target_local_all[sample_idx]
                    labels = labels_all[sample_idx]

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = BASE.reduce_points(
                        points,
                        n_components=20,
                        random_state=33_821 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer,
                    )
                    geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H82_NEIGHBORS)

                    _, h70_defect_all, _ = compute_h70_scores(
                        geodesic=geodesic,
                        support_dir=support_dir,
                        source_local=source_local_all,
                        target_local=target_local_all,
                        triangle_k=H82_TRIANGLE_K,
                    )
                    h70_defect = h70_defect_all[sample_idx]
                    edge_geodesic = geodesic[source_local, target_local]
                    geodesic_bins = BASE.degree_bins(edge_geodesic, max_bins=6)

                    local_cycle = edge_local_cycle_features(
                        points_pca=points_pca,
                        geodesic=geodesic,
                        source_local=source_local,
                        target_local=target_local,
                        n_local_neighbors=H82_LOCAL_NEIGHBOR_K,
                        q_grid=H82_FILTRATION_QUANTILES,
                    )
                    finite_cycle = np.isfinite(local_cycle)
                    if finite_cycle.sum() < 80:
                        continue
                    cycle_filled = local_cycle.copy()
                    cycle_filled[~finite_cycle] = float(np.nanmedian(local_cycle[finite_cycle]))

                    score_base = h70_defect.copy()
                    score_aug = score_base + H82_WEIGHT * BASE.zscore(cycle_filled)

                    auc_base = BASE.safe_auc(labels, score_base)
                    auc_aug = BASE.safe_auc(labels, score_aug)
                    delta_auc = (
                        float(auc_aug - auc_base) if np.isfinite(auc_aug) and np.isfinite(auc_base) else float("nan")
                    )

                    hotspot_threshold = float(np.quantile(score_base[finite_cycle], 0.80))
                    hotspot_idx = np.where(finite_cycle & (score_base >= hotspot_threshold))[0]
                    if hotspot_idx.size < 20:
                        continue
                    control_idx = matched_control_indices(
                        hotspot_idx=hotspot_idx,
                        bins=geodesic_bins,
                        labels=labels,
                        rng=rng,
                    )
                    if control_idx.size < 20:
                        continue
                    local_gap = float(np.mean(cycle_filled[hotspot_idx]) - np.mean(cycle_filled[control_idx]))

                    null_random_gap = np.empty(H82_NULL_PERM, dtype=float)
                    null_feature_delta = np.empty(H82_NULL_PERM, dtype=float)
                    null_label_delta = np.empty(H82_NULL_PERM, dtype=float)

                    for perm_idx in range(H82_NULL_PERM):
                        pseudo_hotspot = []
                        for hidx in hotspot_idx:
                            pool = np.where((geodesic_bins == geodesic_bins[hidx]) & finite_cycle)[0]
                            if pool.size == 0:
                                pool = np.where(finite_cycle)[0]
                            pseudo_hotspot.append(int(rng.choice(pool)))
                        pseudo_hotspot = np.asarray(pseudo_hotspot, dtype=int)
                        pseudo_control = matched_control_indices(
                            hotspot_idx=pseudo_hotspot,
                            bins=geodesic_bins,
                            labels=labels,
                            rng=rng,
                        )
                        if pseudo_control.size == 0:
                            null_random_gap[perm_idx] = float("nan")
                        else:
                            null_random_gap[perm_idx] = float(
                                np.mean(cycle_filled[pseudo_hotspot]) - np.mean(cycle_filled[pseudo_control])
                            )
                        null_rows.append(
                            {
                                "hypothesis_id": "H82",
                                "null_kind": "matched_random_hotspot_set",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_value": float(null_random_gap[perm_idx]),
                            }
                        )

                        shuffled_cycle = BASE.shuffle_within_bins(cycle_filled, geodesic_bins, rng)
                        score_feat = score_base + H82_WEIGHT * BASE.zscore(shuffled_cycle)
                        auc_feat = BASE.safe_auc(labels, score_feat)
                        null_feature_delta[perm_idx] = (
                            float(auc_feat - auc_base) if np.isfinite(auc_feat) and np.isfinite(auc_base) else float("nan")
                        )
                        null_rows.append(
                            {
                                "hypothesis_id": "H82",
                                "null_kind": "feature_shuffle_within_geodesic_bins",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_value": float(null_feature_delta[perm_idx]),
                            }
                        )

                        labels_perm = BASE.stratified_shuffle(labels, geodesic_bins, rng).astype(int)
                        auc_lp_aug = BASE.safe_auc(labels_perm, score_aug)
                        auc_lp_base = BASE.safe_auc(labels_perm, score_base)
                        null_label_delta[perm_idx] = (
                            float(auc_lp_aug - auc_lp_base)
                            if np.isfinite(auc_lp_aug) and np.isfinite(auc_lp_base)
                            else float("nan")
                        )
                        null_rows.append(
                            {
                                "hypothesis_id": "H82",
                                "null_kind": "label_permutation",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_value": float(null_label_delta[perm_idx]),
                            }
                        )

                    q95_gap = float(np.nanquantile(null_random_gap, 0.95))
                    q95_feat = float(np.nanquantile(null_feature_delta, 0.95))
                    q95_lab = float(np.nanquantile(null_label_delta, 0.95))
                    p_feat = BASE.empirical_upper_tail_p(delta_auc, null_feature_delta)
                    p_label = BASE.empirical_upper_tail_p(delta_auc, null_label_delta)
                    p_gap = BASE.empirical_upper_tail_p(local_gap, null_random_gap)
                    p_best = np.nanmin(np.array([p_feat, p_label, p_gap], dtype=float))

                    rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "n_edges_eval": int(labels.size),
                            "n_positive_eval": int(labels.sum()),
                            "n_finite_local_cycle": int(finite_cycle.sum()),
                            "auc_h70_baseline": float(auc_base),
                            "auc_h82_local_cycle_augmented": float(auc_aug),
                            "delta_auc_local_cycle_plus_h70_minus_h70": float(delta_auc),
                            "local_cycle_hotspot_gap": float(local_gap),
                            "q95_null_local_cycle_hotspot_gap": float(q95_gap),
                            "null_gap_q95_local_cycle_hotspot_gap": float(local_gap - q95_gap),
                            "q95_null_delta_feature_shuffle": float(q95_feat),
                            "q95_null_delta_label_shuffle": float(q95_lab),
                            "null_gap_q95_feature_shuffle": float(delta_auc - q95_feat),
                            "null_gap_q95_label_shuffle": float(delta_auc - q95_lab),
                            "p_feature_shuffle_upper": float(p_feat),
                            "p_label_shuffle_upper": float(p_label),
                            "p_gap_upper": float(p_gap),
                            "p_best_upper": float(p_best),
                        }
                    )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h82_local_witness_cycle_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h82_local_witness_cycle_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_local_cycle_plus_h70_minus_h70": float(
                        group["delta_auc_local_cycle_plus_h70_minus_h70"].mean()
                    ),
                    "mean_local_cycle_hotspot_gap": float(group["local_cycle_hotspot_gap"].mean()),
                    "mean_null_gap_q95_local_cycle_hotspot_gap": float(
                        group["null_gap_q95_local_cycle_hotspot_gap"].mean()
                    ),
                    "fraction_delta_positive": float(
                        (group["delta_auc_local_cycle_plus_h70_minus_h70"] > 0.0).mean()
                    ),
                    "fraction_null_gap_positive": float(
                        (group["null_gap_q95_local_cycle_hotspot_gap"] > 0.0).mean()
                    ),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h82_local_witness_cycle_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_local_cycle_plus_h70_minus_h70"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int(
            (summary_df["mean_delta_auc_local_cycle_plus_h70_minus_h70"] > 0.0).sum()
        )
        if not summary_df.empty
        else 0,
        "positive_null_gap_domain_splits": int(
            (summary_df["mean_null_gap_q95_local_cycle_hotspot_gap"] > 0.0).sum()
        )
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_layer_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h83_pathway_trajectory_invariance(
    gene2go_upper: dict[str, set[str]],
    tf_targets: dict[str, set[str]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    domains = ["immune", "lung", "external_lung"]
    for domain_index, domain in enumerate(domains):
        sc_run = BASE.SCGPT_RUNS_BY_DOMAIN[domain]["seed42_main"]
        sc_edges = pd.read_csv(sc_run / "cycle1_edge_dataset.tsv", sep="\t")
        sc_layers = np.load(sc_run / "layer_gene_embeddings.npy", mmap_mode="r")
        gf_edges = pd.read_csv(BASE.GENEFORMER_EDGE_BY_DOMAIN[domain], sep="\t")

        top_genes = set(BASE.select_top_genes(sc_edges, gene_cap=H83_GENE_CAP))
        sc_edges = sc_edges.loc[
            sc_edges["source_idx"].isin(top_genes) & sc_edges["target_idx"].isin(top_genes)
        ].copy()
        if sc_edges.empty:
            continue

        edge_gene_indices = np.unique(
            np.concatenate(
                [
                    sc_edges["source_idx"].to_numpy(dtype=int),
                    sc_edges["target_idx"].to_numpy(dtype=int),
                ]
            )
        )
        symbol_map = BASE.build_symbol_map(sc_edges)
        symbols = [symbol_map[int(g)] for g in edge_gene_indices]

        gf_rank_proxy = (
            gf_edges["source_token_id"].to_numpy(dtype=float) + gf_edges["target_token_id"].to_numpy(dtype=float)
        ) / 2.0
        quantiles = np.quantile(gf_rank_proxy, [0.0, 0.25, 0.50, 0.75, 1.0])
        gf_depth_edges: list[pd.DataFrame] = []
        for i in range(4):
            lo = float(quantiles[i])
            hi = float(quantiles[i + 1])
            if i < 3:
                mask = (gf_rank_proxy >= lo) & (gf_rank_proxy < hi)
            else:
                mask = (gf_rank_proxy >= lo) & (gf_rank_proxy <= hi)
            gf_depth_edges.append(gf_edges.loc[mask].copy())

        sc_sig_by_depth: list[pd.DataFrame] = []
        gf_sig_by_depth: list[pd.DataFrame] = []
        for depth_idx, layer in enumerate(H83_LAYERS):
            if layer >= sc_layers.shape[0]:
                continue
            points = sc_layers[layer, edge_gene_indices, :]
            sc_sig = BASE.fit_signatures_scgpt(
                layer_points=points,
                symbols=symbols,
                random_state=33_830 + domain_index * 100 + depth_idx,
                n_neighbors=10,
            )
            gf_sig = BASE.fit_signatures_geneformer(gf_depth_edges[depth_idx], symbols)
            sc_sig_by_depth.append(sc_sig)
            gf_sig_by_depth.append(gf_sig)

        if len(sc_sig_by_depth) != 4 or len(gf_sig_by_depth) != 4:
            continue
        shared = sorted(
            set(sc_sig_by_depth[0].index)
            & set(sc_sig_by_depth[1].index)
            & set(sc_sig_by_depth[2].index)
            & set(sc_sig_by_depth[3].index)
            & set(gf_sig_by_depth[0].index)
            & set(gf_sig_by_depth[1].index)
            & set(gf_sig_by_depth[2].index)
            & set(gf_sig_by_depth[3].index)
        )
        if len(shared) < 120:
            continue

        modules = build_h83_modules(shared_symbols=shared, gene2go_upper=gene2go_upper, tf_targets=tf_targets)
        if len(modules) < 12:
            continue

        sc_embed_depth = [
            spectral_embedding_from_signatures(df.loc[shared], n_components=H83_SPECTRAL_DIM, n_neighbors=H83_SPECTRAL_K)
            for df in sc_sig_by_depth
        ]
        gf_embed_depth = [
            spectral_embedding_from_signatures(df.loc[shared], n_components=H83_SPECTRAL_DIM, n_neighbors=H83_SPECTRAL_K)
            for df in gf_sig_by_depth
        ]

        dist_sc, module_sizes = module_distance_matrices(sc_embed_depth, modules, shared)
        dist_gf, _ = module_distance_matrices(gf_embed_depth, modules, shared)
        if len(dist_sc) != 4 or len(dist_gf) != 4 or module_sizes.size < 10:
            continue

        rho_mean, cka, top1 = trajectory_concordance_from_distances(dist_sc=dist_sc, dist_gf=dist_gf)
        rng = np.random.default_rng(33_831 + domain_index)
        null_label_perm = np.empty(H83_NULL_PERM, dtype=float)
        null_depth_perm = np.empty(H83_NULL_PERM, dtype=float)
        null_signature_destroy = np.empty(H83_NULL_PERM, dtype=float)

        n_modules = dist_sc[0].shape[0]
        n_genes_shared = len(shared)

        for perm_idx in range(H83_NULL_PERM):
            perm_modules = rng.permutation(n_modules)
            dist_gf_perm = [d[perm_modules][:, perm_modules] for d in dist_gf]
            rho_perm, _, _ = trajectory_concordance_from_distances(dist_sc=dist_sc, dist_gf=dist_gf_perm)
            null_label_perm[perm_idx] = rho_perm
            null_rows.append(
                {
                    "hypothesis_id": "H83",
                    "null_kind": "module_label_permutation",
                    "domain": domain,
                    "perm_idx": int(perm_idx),
                    "null_value": float(rho_perm),
                }
            )

            perm_depth = rng.permutation(len(dist_gf))
            dist_gf_depth = [dist_gf[i] for i in perm_depth]
            rho_depth, _, _ = trajectory_concordance_from_distances(dist_sc=dist_sc, dist_gf=dist_gf_depth)
            null_depth_perm[perm_idx] = rho_depth
            null_rows.append(
                {
                    "hypothesis_id": "H83",
                    "null_kind": "depth_order_permutation",
                    "domain": domain,
                    "perm_idx": int(perm_idx),
                    "null_value": float(rho_depth),
                }
            )

            destroyed = []
            for depth_df in gf_embed_depth:
                arr = depth_df.loc[shared].to_numpy(dtype=float)
                arr_destroy = arr[rng.permutation(n_genes_shared)]
                destroyed.append(pd.DataFrame(arr_destroy, index=shared, columns=depth_df.columns))
            dist_destroy, _ = module_distance_matrices(destroyed, modules, shared)
            rho_destroy, _, _ = trajectory_concordance_from_distances(dist_sc=dist_sc, dist_gf=dist_destroy)
            null_signature_destroy[perm_idx] = rho_destroy
            null_rows.append(
                {
                    "hypothesis_id": "H83",
                    "null_kind": "signature_destroy_permutation",
                    "domain": domain,
                    "perm_idx": int(perm_idx),
                    "null_value": float(rho_destroy),
                }
            )

        all_null = np.concatenate([null_label_perm, null_depth_perm, null_signature_destroy])
        q95 = float(np.nanquantile(all_null, 0.95))
        null_gap = float(rho_mean - q95) if np.isfinite(rho_mean) and np.isfinite(q95) else float("nan")
        p_label = BASE.empirical_upper_tail_p(rho_mean, null_label_perm)
        p_depth = BASE.empirical_upper_tail_p(rho_mean, null_depth_perm)
        p_destroy = BASE.empirical_upper_tail_p(rho_mean, null_signature_destroy)
        p_best = np.nanmin(np.array([p_label, p_depth, p_destroy], dtype=float))

        rows.append(
            {
                "domain": domain,
                "split_regime": "heldout_domain_seed42",
                "n_modules": int(n_modules),
                "mean_module_size": float(np.mean(module_sizes)),
                "n_shared_genes": int(len(shared)),
                "trajectory_spearman_mean": float(rho_mean),
                "trajectory_cka": float(cka),
                "trajectory_top1_profile_retrieval": float(top1),
                "q95_null_trajectory_spearman": float(q95),
                "null_gap_q95_trajectory_spearman": float(null_gap),
                "p_label_perm_upper": float(p_label),
                "p_depth_perm_upper": float(p_depth),
                "p_signature_destroy_upper": float(p_destroy),
                "p_best_upper": float(p_best),
            }
        )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain"])
    by_row_path = ITER_DIR / "h83_pathway_trajectory_invariance_by_domain.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "perm_idx"])
    null_path = ITER_DIR / "h83_pathway_trajectory_invariance_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        summary_rows.append(
            {
                "n_domains": int(by_row_df.shape[0]),
                "mean_trajectory_spearman": float(by_row_df["trajectory_spearman_mean"].mean()),
                "mean_trajectory_cka": float(by_row_df["trajectory_cka"].mean()),
                "mean_top1_profile_retrieval": float(by_row_df["trajectory_top1_profile_retrieval"].mean()),
                "mean_null_gap_q95_trajectory_spearman": float(
                    by_row_df["null_gap_q95_trajectory_spearman"].mean()
                ),
                "positive_null_gap_domains": int((by_row_df["null_gap_q95_trajectory_spearman"] > 0.0).sum()),
                "fraction_p_best_lt_0_05": float((by_row_df["p_best_upper"] < 0.05).mean()),
                "combined_fisher_p_best": float(BASE.safe_fisher_p(by_row_df["p_best_upper"].to_numpy(dtype=float))),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = ITER_DIR / "h83_pathway_trajectory_invariance_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_trajectory_spearman": float(by_row_df["trajectory_spearman_mean"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_null_gap_domains": int((by_row_df["null_gap_q95_trajectory_spearman"] > 0.0).sum())
        if not by_row_df.empty
        else 0,
        "artifact_paths": {
            "by_domain": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h84_shortcut_bridge_competition(
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

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H84_GENE_CAP))
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

            for layer in H84_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=33_840 + domain_index * 100 + split_index * 20 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H84_NEIGHBORS)

                baseline_score, _, tri_bundle = compute_h70_scores(
                    geodesic=geodesic,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H84_TRIANGLE_K,
                )

                knn_edges = BASE.build_knn_edge_array(points_pca, n_neighbors=H84_NEIGHBORS)
                neighbors = BASE.adjacency_neighbors(points_pca.shape[0], knn_edges)
                clust = BASE.local_clustering(neighbors)
                degree = np.array([len(n) for n in neighbors], dtype=float)
                node_bridge = degree * (1.0 - clust)

                edge_bridge = 0.5 * (node_bridge[source_local] + node_bridge[target_local])
                edge_triangle = tri_bundle["close_frac_mean"]
                edge_sbc = BASE.zscore(edge_bridge) - BASE.zscore(edge_triangle)

                alt_score = baseline_score + H84_WEIGHT * BASE.zscore(edge_sbc)
                auc_base = BASE.safe_auc(labels, baseline_score)
                auc_alt = BASE.safe_auc(labels, alt_score)
                delta_auc = (
                    float(auc_alt - auc_base) if np.isfinite(auc_alt) and np.isfinite(auc_base) else float("nan")
                )

                edge_geodesic = geodesic[source_local, target_local]
                bins = BASE.degree_bins(edge_geodesic, max_bins=6)
                rng = np.random.default_rng(33_841 + domain_index * 100 + split_index * 20 + layer)
                null_endpoint = np.empty(H84_NULL_PERM, dtype=float)
                null_feature = np.empty(H84_NULL_PERM, dtype=float)
                null_label = np.empty(H84_NULL_PERM, dtype=float)

                for perm_idx in range(H84_NULL_PERM):
                    target_swap = target_local.copy()
                    for b in np.unique(bins):
                        idx = np.where(bins == b)[0]
                        if idx.size > 1:
                            target_swap[idx] = rng.permutation(target_swap[idx])
                    tri_swap = BASE.multiscale_triangle_defect_features(
                        geodesic=geodesic,
                        source_local=source_local,
                        target_local=target_swap,
                        k_values=H84_TRIANGLE_K,
                    )
                    edge_bridge_swap = 0.5 * (node_bridge[source_local] + node_bridge[target_swap])
                    edge_sbc_swap = BASE.zscore(edge_bridge_swap) - BASE.zscore(tri_swap["close_frac_mean"])
                    alt_swap = baseline_score + H84_WEIGHT * BASE.zscore(edge_sbc_swap)
                    auc_swap = BASE.safe_auc(labels, alt_swap)
                    null_endpoint[perm_idx] = (
                        float(auc_swap - auc_base) if np.isfinite(auc_swap) and np.isfinite(auc_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H84",
                            "null_kind": "endpoint_swap_within_geodesic_bins",
                            "domain": domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_endpoint[perm_idx]),
                        }
                    )

                    sbc_shuffle = BASE.shuffle_within_bins(edge_sbc, bins, rng)
                    alt_feat = baseline_score + H84_WEIGHT * BASE.zscore(sbc_shuffle)
                    auc_feat = BASE.safe_auc(labels, alt_feat)
                    null_feature[perm_idx] = (
                        float(auc_feat - auc_base) if np.isfinite(auc_feat) and np.isfinite(auc_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H84",
                            "null_kind": "feature_shuffle_within_geodesic_bins",
                            "domain": domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_feature[perm_idx]),
                        }
                    )

                    labels_perm = BASE.stratified_shuffle(labels, bins, rng).astype(int)
                    auc_lp_alt = BASE.safe_auc(labels_perm, alt_score)
                    auc_lp_base = BASE.safe_auc(labels_perm, baseline_score)
                    null_label[perm_idx] = (
                        float(auc_lp_alt - auc_lp_base)
                        if np.isfinite(auc_lp_alt) and np.isfinite(auc_lp_base)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H84",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_endpoint, null_feature, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_end = BASE.empirical_upper_tail_p(delta_auc, null_endpoint)
                p_feat = BASE.empirical_upper_tail_p(delta_auc, null_feature)
                p_lab = BASE.empirical_upper_tail_p(delta_auc, null_label)
                p_best = np.nanmin(np.array([p_end, p_feat, p_lab], dtype=float))

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_sbc_index": float(auc_alt),
                        "auc_directed_geodesic_baseline": float(auc_base),
                        "delta_auc_sbc_index_minus_baseline": float(delta_auc),
                        "mean_edge_bridge": float(np.mean(edge_bridge)),
                        "mean_edge_triangle_close_frac": float(np.mean(edge_triangle)),
                        "mean_sbc_index": float(np.mean(edge_sbc)),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95_delta_auc": float(delta_auc - q95),
                        "p_endpoint_upper": float(p_end),
                        "p_feature_upper": float(p_feat),
                        "p_label_upper": float(p_lab),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h84_shortcut_bridge_competition_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h84_shortcut_bridge_competition_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_sbc_index_minus_baseline": float(
                        group["delta_auc_sbc_index_minus_baseline"].mean()
                    ),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_sbc_index_minus_baseline"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h84_shortcut_bridge_competition_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_sbc_index_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_sbc_index_minus_baseline"] > 0.0).sum())
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
    tf_targets = load_dorothea_tf_targets()

    h82_summary = run_h82_local_witness_cycle_persistence(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h83_summary = run_h83_pathway_trajectory_invariance(
        gene2go_upper=gene2go_upper,
        tf_targets=tf_targets,
    )
    h84_summary = run_h84_shortcut_bridge_competition(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    summary = {
        "iteration": "iter_0033",
        "h82": h82_summary,
        "h83": h83_summary,
        "h84": h84_summary,
    }
    summary_path = ITER_DIR / "iter0033_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
