from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ITER_DIR = Path("iterations/iter_0034")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base")

# H85 / N420: dual-filtration local witness persistence on H82/H70 hotspots.
H85_LAYERS = [7, 11]
H85_GENE_CAP = 170
H85_NEIGHBORS = 12
H85_TRIANGLE_K = [8, 12, 16]
H85_EDGE_SAMPLE = 240
H85_LOCAL_NEIGHBOR_K = 12
H85_DIST_QUANTILES = [0.20, 0.35, 0.50, 0.65]
H85_MARGIN_QUANTILES = [0.40, 0.60, 0.80]
H85_NULL_PERM = 16
H85_WEIGHT = 0.32

# H86 / N429: cross-model barcode OT depth alignment pilot.
H86_LAYERS = [0, 3, 7, 11]
H86_GENE_CAP = 220
H86_MIN_MODULE_SIZE = 8
H86_MAX_MODULE_SIZE = 70
H86_MAX_GO_MODULES = 20
H86_MAX_TRRUST_MODULES = 20
H86_NULL_PERM = 24
H86_OT_EPS = 0.18
H86_OT_ITERS = 120

# H87 / N433: cheap sparse descriptor blend breadth screen.
H87_LAYERS = [0, 3, 7, 11]
H87_GENE_CAP = 170
H87_NEIGHBORS = 12
H87_TRIANGLE_K = [8, 12, 16]
H87_NULL_PERM = 8
H87_CV_SPLITS = 4
H87_L1_C = 0.2


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


def dual_filtration_cycle_surface(
    points_local: np.ndarray,
    margin_local: np.ndarray,
    dist_quantiles: list[float],
    margin_quantiles: list[float],
) -> tuple[float, float]:
    # Two-parameter filtration surrogate: edge appears if distance is small and support-margin is strong.
    n = int(points_local.shape[0])
    if n < 8:
        return float("nan"), float("nan")

    d = cdist(points_local, points_local, metric="euclidean")
    tri_d = d[np.triu_indices(n, k=1)]
    tri_m = margin_local[np.triu_indices(n, k=1)]
    tri_d = tri_d[np.isfinite(tri_d)]
    tri_m = tri_m[np.isfinite(tri_m)]
    if tri_d.size < 8 or tri_m.size < 8:
        return float("nan"), float("nan")

    values = []
    for qd in dist_quantiles:
        d_thr = float(np.quantile(tri_d, qd))
        for qm in margin_quantiles:
            m_thr = float(np.quantile(tri_m, qm))
            edge_mask = np.triu((d <= d_thr) & (margin_local >= m_thr), k=1)
            edges = np.argwhere(edge_mask)
            values.append(cycle_rank(n_nodes=n, edges=edges))

    if not values:
        return float("nan"), float("nan")
    b = np.asarray(values, dtype=float)
    return float(np.mean(b)), float(np.max(b))


def edge_local_dual_cycle_features(
    points_pca: np.ndarray,
    geodesic: np.ndarray,
    margin_matrix: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    n_local_neighbors: int,
    dist_quantiles: list[float],
    margin_quantiles: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    n = points_pca.shape[0]
    nodes = np.arange(n, dtype=int)
    out_mean = np.full(source_local.shape[0], np.nan, dtype=float)
    out_max = np.full(source_local.shape[0], np.nan, dtype=float)

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

        margin_local = margin_matrix[np.ix_(local_nodes, local_nodes)]
        m, mx = dual_filtration_cycle_surface(
            points_local=points_pca[local_nodes],
            margin_local=margin_local,
            dist_quantiles=dist_quantiles,
            margin_quantiles=margin_quantiles,
        )
        out_mean[i] = m
        out_max[i] = mx

    return out_mean, out_max


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


def build_h86_modules(
    shared_symbols: list[str],
    gene2go_upper: dict[str, set[str]],
    tf_targets: dict[str, set[str]],
) -> list[tuple[str, list[str]]]:
    shared_set = set(shared_symbols)
    modules: list[tuple[str, list[str]]] = []

    trrust_candidates = []
    for tf, tgts in tf_targets.items():
        genes = sorted(({tf} | set(tgts)) & shared_set)
        if H86_MIN_MODULE_SIZE <= len(genes) <= H86_MAX_MODULE_SIZE:
            trrust_candidates.append((tf, genes))
    trrust_candidates = sorted(trrust_candidates, key=lambda x: (-len(x[1]), x[0]))
    for tf, genes in trrust_candidates[:H86_MAX_TRRUST_MODULES]:
        modules.append((f"TRRUST::{tf}", genes))

    term_to_genes = build_term_to_genes(shared_symbols, gene2go_upper)
    go_candidates = []
    for term, genes in term_to_genes.items():
        if H86_MIN_MODULE_SIZE <= len(genes) <= H86_MAX_MODULE_SIZE:
            go_candidates.append((term, sorted(genes)))
    go_candidates = sorted(go_candidates, key=lambda x: (-len(x[1]), x[0]))
    for term, genes in go_candidates[:H86_MAX_GO_MODULES]:
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


def module_distance_matrix_from_signatures(
    signature_df: pd.DataFrame,
    modules: list[tuple[str, list[str]]],
    shared_symbols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    idx_map = {g: i for i, g in enumerate(shared_symbols)}
    arr = signature_df.loc[shared_symbols].to_numpy(dtype=float)

    module_loc = []
    module_sizes = []
    for _, genes in modules:
        loc = [idx_map[g] for g in genes if g in idx_map]
        if len(loc) < H86_MIN_MODULE_SIZE:
            continue
        module_loc.append(np.asarray(loc, dtype=int))
        module_sizes.append(len(loc))

    if len(module_loc) < 10:
        return np.zeros((0, 0), dtype=float), np.array([], dtype=int)

    centroids = np.asarray([arr[loc].mean(axis=0) for loc in module_loc], dtype=float)
    d = cdist(centroids, centroids, metric="euclidean")
    return d, np.asarray(module_sizes, dtype=int)


def persistence_descriptor_from_distance_matrix(dist: np.ndarray) -> np.ndarray:
    n = dist.shape[0]
    if n < 3:
        return np.full(9, np.nan, dtype=float)

    tri = dist[np.triu_indices(n, k=1)]
    tri = tri[np.isfinite(tri)]
    if tri.size < 6:
        return np.full(9, np.nan, dtype=float)

    quantiles = np.quantile(tri, [0.20, 0.40, 0.60, 0.80])
    betti_vals = []
    for q in [0.20, 0.35, 0.50, 0.65, 0.80]:
        thr = float(np.quantile(tri, q))
        edge_mask = np.triu(dist <= thr, k=1)
        edges = np.argwhere(edge_mask)
        betti_vals.append(cycle_rank(n_nodes=n, edges=edges))
    return np.asarray(list(quantiles) + list(betti_vals), dtype=float)


def sinkhorn_uniform_plan(cost: np.ndarray, epsilon: float, n_iter: int) -> np.ndarray:
    c = np.asarray(cost, dtype=float)
    n, m = c.shape
    a = np.full(n, 1.0 / n, dtype=float)
    b = np.full(m, 1.0 / m, dtype=float)

    scale = float(np.nanmedian(c[np.isfinite(c)]))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    kernel = np.exp(-c / max(epsilon * scale, 1e-8))
    kernel = np.clip(kernel, 1e-12, None)

    u = np.ones(n, dtype=float)
    v = np.ones(m, dtype=float)
    for _ in range(max(20, n_iter)):
        kv = kernel @ v
        kv = np.clip(kv, 1e-12, None)
        u = a / kv
        ktu = kernel.T @ u
        ktu = np.clip(ktu, 1e-12, None)
        v = b / ktu

    plan = (u[:, None] * kernel) * v[None, :]
    total = float(plan.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return np.full((n, m), 1.0 / (n * m), dtype=float)
    return plan / total


def random_doubly_stochastic(n: int, rng: np.random.Generator, n_iter: int = 30) -> np.ndarray:
    mat = rng.random((n, n), dtype=float) + 1e-8
    for _ in range(n_iter):
        mat = mat / np.clip(mat.sum(axis=1, keepdims=True), 1e-12, None)
        mat = mat / np.clip(mat.sum(axis=0, keepdims=True), 1e-12, None)
    mat = mat / np.clip(mat.sum(), 1e-12, None)
    return mat


def ot_depth_alignment_metrics(cost: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    plan = sinkhorn_uniform_plan(cost, epsilon=H86_OT_EPS, n_iter=H86_OT_ITERS)
    n = cost.shape[0]
    depth_idx = np.arange(n, dtype=float)

    mismatch = float(np.sum(plan * np.abs(depth_idx[:, None] - depth_idx[None, :])))
    mismatch /= float(max(1, n - 1))
    monotonic_score = float(1.0 - mismatch)

    row_mass = np.clip(plan.sum(axis=1), 1e-12, None)
    mapped_depth = (plan @ depth_idx) / row_mass
    rho = safe_spearman(depth_idx, mapped_depth)
    ot_cost = float(np.sum(plan * cost))
    return monotonic_score, rho, ot_cost, plan


def min_class_count(labels: np.ndarray) -> int:
    y = np.asarray(labels, dtype=int)
    counts = np.bincount(y, minlength=2)
    return int(np.min(counts))


def cross_validated_auc(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    model_kind: str,
) -> float:
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)

    if x.ndim != 2 or x.shape[0] != y.size or np.unique(y).size < 2:
        return float("nan")

    max_splits = min(H87_CV_SPLITS, min_class_count(y))
    if max_splits < 2:
        return float("nan")

    cv = StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=random_state)
    probs = np.full(y.shape[0], np.nan, dtype=float)

    for fold_idx, (tr, te) in enumerate(cv.split(x, y)):
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x[tr])
        x_te = scaler.transform(x[te])

        if model_kind == "baseline":
            model = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=1000,
                random_state=random_state + fold_idx,
            )
        elif model_kind == "blend":
            model = LogisticRegression(
                penalty="l1",
                C=H87_L1_C,
                solver="liblinear",
                max_iter=1000,
                random_state=random_state + fold_idx,
            )
        else:
            raise ValueError(f"Unknown model_kind={model_kind}")

        model.fit(x_tr, y[tr])
        probs[te] = model.predict_proba(x_te)[:, 1]

    return BASE.safe_auc(y, probs)


def fit_blend_nonzero_count(features: np.ndarray, labels: np.ndarray) -> int:
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    if x.ndim != 2 or x.shape[0] != y.size or np.unique(y).size < 2:
        return 0
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(
        penalty="l1",
        C=H87_L1_C,
        solver="liblinear",
        max_iter=1000,
        random_state=4034,
    )
    model.fit(x_scaled, y)
    coef = np.asarray(model.coef_, dtype=float).ravel()
    return int(np.sum(np.abs(coef) > 1e-8))


def build_h87_descriptors(
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


def run_h85_dual_filtration_local_witness(
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

                top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H85_GENE_CAP))
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
                support_margin_matrix = np.abs(support_dir - support_dir.T)

                gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
                source_local_all = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
                target_local_all = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
                labels_all = split_edges["label"].to_numpy(dtype=int)

                for layer in H85_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    rng = np.random.default_rng(
                        34_850 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer
                    )
                    sample_idx = stratified_index_sample(labels_all, max_n=H85_EDGE_SAMPLE, rng=rng)
                    if sample_idx.size < 120:
                        continue

                    source_local = source_local_all[sample_idx]
                    target_local = target_local_all[sample_idx]
                    labels = labels_all[sample_idx]

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = BASE.reduce_points(
                        points,
                        n_components=20,
                        random_state=34_851 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer,
                    )
                    geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H85_NEIGHBORS)

                    _, h70_defect_all, _ = compute_h70_scores(
                        geodesic=geodesic,
                        support_dir=support_dir,
                        source_local=source_local_all,
                        target_local=target_local_all,
                        triangle_k=H85_TRIANGLE_K,
                    )
                    h70_defect = h70_defect_all[sample_idx]
                    edge_geodesic = geodesic[source_local, target_local]
                    geodesic_bins = BASE.degree_bins(edge_geodesic, max_bins=6)

                    dual_mean, dual_max = edge_local_dual_cycle_features(
                        points_pca=points_pca,
                        geodesic=geodesic,
                        margin_matrix=support_margin_matrix,
                        source_local=source_local,
                        target_local=target_local,
                        n_local_neighbors=H85_LOCAL_NEIGHBOR_K,
                        dist_quantiles=H85_DIST_QUANTILES,
                        margin_quantiles=H85_MARGIN_QUANTILES,
                    )
                    finite = np.isfinite(dual_mean) & np.isfinite(dual_max)
                    if finite.sum() < 80:
                        continue

                    dual_mean_fill = dual_mean.copy()
                    dual_max_fill = dual_max.copy()
                    dual_mean_fill[~np.isfinite(dual_mean_fill)] = float(np.nanmedian(dual_mean_fill[finite]))
                    dual_max_fill[~np.isfinite(dual_max_fill)] = float(np.nanmedian(dual_max_fill[finite]))
                    dual_score = BASE.zscore(dual_mean_fill) + 0.35 * BASE.zscore(dual_max_fill)

                    score_base = h70_defect.copy()
                    score_aug = score_base + H85_WEIGHT * dual_score

                    auc_base = BASE.safe_auc(labels, score_base)
                    auc_aug = BASE.safe_auc(labels, score_aug)
                    delta_auc = (
                        float(auc_aug - auc_base) if np.isfinite(auc_aug) and np.isfinite(auc_base) else float("nan")
                    )

                    hotspot_threshold = float(np.quantile(score_base[finite], 0.80))
                    hotspot_idx = np.where(finite & (score_base >= hotspot_threshold))[0]
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
                    local_gap = float(np.mean(dual_score[hotspot_idx]) - np.mean(dual_score[control_idx]))

                    null_random_gap = np.empty(H85_NULL_PERM, dtype=float)
                    null_feature_delta = np.empty(H85_NULL_PERM, dtype=float)
                    null_label_delta = np.empty(H85_NULL_PERM, dtype=float)

                    for perm_idx in range(H85_NULL_PERM):
                        pseudo_hotspot = []
                        for hidx in hotspot_idx:
                            pool = np.where((geodesic_bins == geodesic_bins[hidx]) & finite)[0]
                            if pool.size == 0:
                                pool = np.where(finite)[0]
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
                                np.mean(dual_score[pseudo_hotspot]) - np.mean(dual_score[pseudo_control])
                            )
                        null_rows.append(
                            {
                                "hypothesis_id": "H85",
                                "null_kind": "matched_random_hotspot_set",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_value": float(null_random_gap[perm_idx]),
                            }
                        )

                        shuffled = BASE.shuffle_within_bins(dual_score, geodesic_bins, rng)
                        score_feat = score_base + H85_WEIGHT * BASE.zscore(shuffled)
                        auc_feat = BASE.safe_auc(labels, score_feat)
                        null_feature_delta[perm_idx] = (
                            float(auc_feat - auc_base) if np.isfinite(auc_feat) and np.isfinite(auc_base) else float("nan")
                        )
                        null_rows.append(
                            {
                                "hypothesis_id": "H85",
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
                                "hypothesis_id": "H85",
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
                            "n_finite_dual_cycle": int(finite.sum()),
                            "auc_h70_baseline": float(auc_base),
                            "auc_h85_dual_cycle_augmented": float(auc_aug),
                            "delta_auc_local_dual_filtration_plus_h70_minus_h70": float(delta_auc),
                            "dual_cycle_hotspot_gap": float(local_gap),
                            "q95_null_dual_cycle_hotspot_gap": float(q95_gap),
                            "null_gap_q95_local_dual_filtration_hotspot_gap": float(local_gap - q95_gap),
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
    by_row_path = ITER_DIR / "h85_dual_filtration_witness_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h85_dual_filtration_witness_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_local_dual_filtration_plus_h70_minus_h70": float(
                        group["delta_auc_local_dual_filtration_plus_h70_minus_h70"].mean()
                    ),
                    "mean_dual_cycle_hotspot_gap": float(group["dual_cycle_hotspot_gap"].mean()),
                    "mean_null_gap_q95_local_dual_filtration_hotspot_gap": float(
                        group["null_gap_q95_local_dual_filtration_hotspot_gap"].mean()
                    ),
                    "fraction_delta_positive": float(
                        (group["delta_auc_local_dual_filtration_plus_h70_minus_h70"] > 0.0).mean()
                    ),
                    "fraction_null_gap_positive": float(
                        (group["null_gap_q95_local_dual_filtration_hotspot_gap"] > 0.0).mean()
                    ),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h85_dual_filtration_witness_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_local_dual_filtration_plus_h70_minus_h70"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int(
            (summary_df["mean_delta_auc_local_dual_filtration_plus_h70_minus_h70"] > 0.0).sum()
        )
        if not summary_df.empty
        else 0,
        "positive_null_gap_domain_splits": int(
            (summary_df["mean_null_gap_q95_local_dual_filtration_hotspot_gap"] > 0.0).sum()
        )
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_layer_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h86_barcode_ot_depth_alignment(
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

        top_genes = set(BASE.select_top_genes(sc_edges, gene_cap=H86_GENE_CAP))
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
        for depth_idx, layer in enumerate(H86_LAYERS):
            if layer >= sc_layers.shape[0]:
                continue
            points = sc_layers[layer, edge_gene_indices, :]
            sc_sig = BASE.fit_signatures_scgpt(
                layer_points=points,
                symbols=symbols,
                random_state=34_860 + domain_index * 100 + depth_idx,
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

        modules = build_h86_modules(shared_symbols=shared, gene2go_upper=gene2go_upper, tf_targets=tf_targets)
        if len(modules) < 12:
            continue

        sc_dist_depth = []
        gf_dist_depth = []
        module_sizes_ref = None
        for depth_idx in range(4):
            sc_dist, sc_sizes = module_distance_matrix_from_signatures(sc_sig_by_depth[depth_idx], modules, shared)
            gf_dist, gf_sizes = module_distance_matrix_from_signatures(gf_sig_by_depth[depth_idx], modules, shared)
            if sc_dist.size == 0 or gf_dist.size == 0 or sc_dist.shape != gf_dist.shape:
                sc_dist_depth = []
                break
            if module_sizes_ref is None:
                module_sizes_ref = sc_sizes
            elif module_sizes_ref.size != sc_sizes.size:
                sc_dist_depth = []
                break
            if sc_sizes.size != gf_sizes.size:
                sc_dist_depth = []
                break
            sc_dist_depth.append(sc_dist)
            gf_dist_depth.append(gf_dist)
        if len(sc_dist_depth) != 4 or module_sizes_ref is None or module_sizes_ref.size < 10:
            continue

        sc_barcodes = np.vstack([persistence_descriptor_from_distance_matrix(d) for d in sc_dist_depth])
        gf_barcodes = np.vstack([persistence_descriptor_from_distance_matrix(d) for d in gf_dist_depth])
        finite_barcodes = np.isfinite(sc_barcodes).all() and np.isfinite(gf_barcodes).all()
        if not finite_barcodes:
            continue

        combo = np.vstack([sc_barcodes, gf_barcodes])
        mu, sd = BASE.zscore_fit(combo)
        sc_z = BASE.zscore_apply(sc_barcodes, mu, sd)
        gf_z = BASE.zscore_apply(gf_barcodes, mu, sd)

        cost = cdist(sc_z, gf_z, metric="euclidean")
        score, rho, ot_cost, _ = ot_depth_alignment_metrics(cost)

        rng = np.random.default_rng(34_861 + domain_index)
        null_random_transport = np.empty(H86_NULL_PERM, dtype=float)
        null_depth_perm = np.empty(H86_NULL_PERM, dtype=float)
        null_component_perm = np.empty(H86_NULL_PERM, dtype=float)

        for perm_idx in range(H86_NULL_PERM):
            plan_rand = random_doubly_stochastic(cost.shape[0], rng)
            idx = np.arange(cost.shape[0], dtype=float)
            mismatch_rand = float(np.sum(plan_rand * np.abs(idx[:, None] - idx[None, :]))) / float(
                max(1, cost.shape[0] - 1)
            )
            null_random_transport[perm_idx] = float(1.0 - mismatch_rand)
            null_rows.append(
                {
                    "hypothesis_id": "H86",
                    "null_kind": "random_transport_fixed_marginals",
                    "domain": domain,
                    "perm_idx": int(perm_idx),
                    "null_value": float(null_random_transport[perm_idx]),
                }
            )

            perm_depth = rng.permutation(cost.shape[1])
            cost_depth = cost[:, perm_depth]
            depth_score, _, _, _ = ot_depth_alignment_metrics(cost_depth)
            null_depth_perm[perm_idx] = depth_score
            null_rows.append(
                {
                    "hypothesis_id": "H86",
                    "null_kind": "depth_order_permutation",
                    "domain": domain,
                    "perm_idx": int(perm_idx),
                    "null_value": float(null_depth_perm[perm_idx]),
                }
            )

            gf_destroy = np.vstack([row[rng.permutation(row.size)] for row in gf_z])
            cost_destroy = cdist(sc_z, gf_destroy, metric="euclidean")
            destroy_score, _, _, _ = ot_depth_alignment_metrics(cost_destroy)
            null_component_perm[perm_idx] = destroy_score
            null_rows.append(
                {
                    "hypothesis_id": "H86",
                    "null_kind": "barcode_component_permutation",
                    "domain": domain,
                    "perm_idx": int(perm_idx),
                    "null_value": float(null_component_perm[perm_idx]),
                }
            )

        all_null = np.concatenate([null_random_transport, null_depth_perm, null_component_perm])
        q95 = float(np.nanquantile(all_null, 0.95))
        null_gap = float(score - q95) if np.isfinite(score) and np.isfinite(q95) else float("nan")

        p_rand = BASE.empirical_upper_tail_p(score, null_random_transport)
        p_depth = BASE.empirical_upper_tail_p(score, null_depth_perm)
        p_comp = BASE.empirical_upper_tail_p(score, null_component_perm)
        p_best = np.nanmin(np.array([p_rand, p_depth, p_comp], dtype=float))

        rows.append(
            {
                "domain": domain,
                "split_regime": "heldout_domain_seed42",
                "n_modules": int(module_sizes_ref.size),
                "mean_module_size": float(np.mean(module_sizes_ref)),
                "n_shared_genes": int(len(shared)),
                "barcode_ot_depth_alignment_score": float(score),
                "ot_depth_spearman": float(rho),
                "ot_transport_cost": float(ot_cost),
                "q95_null_barcode_ot_depth_alignment_score": float(q95),
                "null_gap_q95_barcode_ot_depth_alignment_score": float(null_gap),
                "p_random_transport_upper": float(p_rand),
                "p_depth_permutation_upper": float(p_depth),
                "p_component_permutation_upper": float(p_comp),
                "p_best_upper": float(p_best),
            }
        )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain"])
    by_row_path = ITER_DIR / "h86_barcode_ot_depth_alignment_by_domain.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "perm_idx"])
    null_path = ITER_DIR / "h86_barcode_ot_depth_alignment_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        summary_rows.append(
            {
                "n_domains": int(by_row_df.shape[0]),
                "mean_barcode_ot_depth_alignment_score": float(by_row_df["barcode_ot_depth_alignment_score"].mean()),
                "mean_ot_depth_spearman": float(by_row_df["ot_depth_spearman"].mean()),
                "mean_ot_transport_cost": float(by_row_df["ot_transport_cost"].mean()),
                "mean_null_gap_q95_barcode_ot_depth_alignment_score": float(
                    by_row_df["null_gap_q95_barcode_ot_depth_alignment_score"].mean()
                ),
                "positive_null_gap_domains": int(
                    (by_row_df["null_gap_q95_barcode_ot_depth_alignment_score"] > 0.0).sum()
                ),
                "fraction_p_best_lt_0_05": float((by_row_df["p_best_upper"] < 0.05).mean()),
                "combined_fisher_p_best": float(BASE.safe_fisher_p(by_row_df["p_best_upper"].to_numpy(dtype=float))),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = ITER_DIR / "h86_barcode_ot_depth_alignment_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_barcode_ot_depth_alignment_score": float(by_row_df["barcode_ot_depth_alignment_score"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_null_gap_domains": int(
            (by_row_df["null_gap_q95_barcode_ot_depth_alignment_score"] > 0.0).sum()
        )
        if not by_row_df.empty
        else 0,
        "artifact_paths": {
            "by_domain": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h87_sparse_descriptor_blend(
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

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H87_GENE_CAP))
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

            for layer in H87_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=34_870 + domain_index * 100 + split_index * 20 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H87_NEIGHBORS)

                _, baseline_score, tri_bundle = compute_h70_scores(
                    geodesic=geodesic,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H87_TRIANGLE_K,
                )
                descriptors = build_h87_descriptors(
                    geodesic=geodesic,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    points_pca=points_pca,
                    tri_bundle=tri_bundle,
                    n_neighbors=H87_NEIGHBORS,
                )

                edge_geodesic = geodesic[source_local, target_local]
                bins = BASE.degree_bins(edge_geodesic, max_bins=6)
                rng = np.random.default_rng(34_871 + domain_index * 100 + split_index * 20 + layer)

                x_base = baseline_score[:, None]
                x_blend = np.column_stack([baseline_score, descriptors])
                auc_base = cross_validated_auc(
                    features=x_base,
                    labels=labels,
                    random_state=34_872 + domain_index * 100 + split_index * 20 + layer,
                    model_kind="baseline",
                )
                auc_blend = cross_validated_auc(
                    features=x_blend,
                    labels=labels,
                    random_state=34_873 + domain_index * 100 + split_index * 20 + layer,
                    model_kind="blend",
                )
                delta_auc = (
                    float(auc_blend - auc_base) if np.isfinite(auc_blend) and np.isfinite(auc_base) else float("nan")
                )

                nonzero_features = fit_blend_nonzero_count(x_blend, labels)

                null_feature = np.empty(H87_NULL_PERM, dtype=float)
                null_endpoint = np.empty(H87_NULL_PERM, dtype=float)
                null_label = np.empty(H87_NULL_PERM, dtype=float)

                for perm_idx in range(H87_NULL_PERM):
                    shuffled_desc = np.column_stack(
                        [BASE.shuffle_within_bins(descriptors[:, j], bins, rng) for j in range(descriptors.shape[1])]
                    )
                    auc_feat = cross_validated_auc(
                        features=np.column_stack([baseline_score, shuffled_desc]),
                        labels=labels,
                        random_state=34_880
                        + domain_index * 1000
                        + split_index * 100
                        + layer * 10
                        + perm_idx,
                        model_kind="blend",
                    )
                    null_feature[perm_idx] = (
                        float(auc_feat - auc_base) if np.isfinite(auc_feat) and np.isfinite(auc_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H87",
                            "null_kind": "descriptor_shuffle_within_geodesic_bins",
                            "domain": domain,
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
                        k_values=H87_TRIANGLE_K,
                    )
                    desc_swap = build_h87_descriptors(
                        geodesic=geodesic,
                        support_dir=support_dir,
                        source_local=source_local,
                        target_local=target_swap,
                        points_pca=points_pca,
                        tri_bundle=tri_swap,
                        n_neighbors=H87_NEIGHBORS,
                    )
                    auc_swap = cross_validated_auc(
                        features=np.column_stack([baseline_score, desc_swap]),
                        labels=labels,
                        random_state=34_980
                        + domain_index * 1000
                        + split_index * 100
                        + layer * 10
                        + perm_idx,
                        model_kind="blend",
                    )
                    null_endpoint[perm_idx] = (
                        float(auc_swap - auc_base) if np.isfinite(auc_swap) and np.isfinite(auc_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H87",
                            "null_kind": "endpoint_swap_within_geodesic_bins",
                            "domain": domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_endpoint[perm_idx]),
                        }
                    )

                    labels_perm = BASE.stratified_shuffle(labels, bins, rng).astype(int)
                    auc_lp_base = cross_validated_auc(
                        features=x_base,
                        labels=labels_perm,
                        random_state=35_080
                        + domain_index * 1000
                        + split_index * 100
                        + layer * 10
                        + perm_idx,
                        model_kind="baseline",
                    )
                    auc_lp_blend = cross_validated_auc(
                        features=x_blend,
                        labels=labels_perm,
                        random_state=35_180
                        + domain_index * 1000
                        + split_index * 100
                        + layer * 10
                        + perm_idx,
                        model_kind="blend",
                    )
                    null_label[perm_idx] = (
                        float(auc_lp_blend - auc_lp_base)
                        if np.isfinite(auc_lp_blend) and np.isfinite(auc_lp_base)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H87",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_feature, null_endpoint, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_feat = BASE.empirical_upper_tail_p(delta_auc, null_feature)
                p_end = BASE.empirical_upper_tail_p(delta_auc, null_endpoint)
                p_lab = BASE.empirical_upper_tail_p(delta_auc, null_label)
                p_best = np.nanmin(np.array([p_feat, p_end, p_lab], dtype=float))

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70_baseline": float(auc_base),
                        "auc_descriptor_blend": float(auc_blend),
                        "delta_auc_descriptor_blend_minus_h70": float(delta_auc),
                        "blend_nonzero_feature_count": int(nonzero_features),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95_delta_auc": float(delta_auc - q95),
                        "p_feature_shuffle_upper": float(p_feat),
                        "p_endpoint_swap_upper": float(p_end),
                        "p_label_shuffle_upper": float(p_lab),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h87_sparse_descriptor_blend_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h87_sparse_descriptor_blend_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_descriptor_blend_minus_h70": float(
                        group["delta_auc_descriptor_blend_minus_h70"].mean()
                    ),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "mean_blend_nonzero_feature_count": float(group["blend_nonzero_feature_count"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_descriptor_blend_minus_h70"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h87_sparse_descriptor_blend_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_descriptor_blend_minus_h70"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_descriptor_blend_minus_h70"] > 0.0).sum())
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

    h85_summary = run_h85_dual_filtration_local_witness(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h86_summary = run_h86_barcode_ot_depth_alignment(
        gene2go_upper=gene2go_upper,
        tf_targets=tf_targets,
    )
    h87_summary = run_h87_sparse_descriptor_blend(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    summary = {
        "iteration": "iter_0034",
        "h85": h85_summary,
        "h86": h86_summary,
        "h87": h87_summary,
    }
    summary_path = ITER_DIR / "iter0034_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
