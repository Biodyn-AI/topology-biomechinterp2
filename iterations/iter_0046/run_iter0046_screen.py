from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ITER_DIR = Path("iterations/iter_0046")
ITER_DIR.mkdir(parents=True, exist_ok=True)

TRRUST_PATH = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
    "single_cell_mechinterp/external/networks/trrust_human.tsv"
)

# H121 / N605: directional geodesic asymmetry.
H121_SEED = "seed42_main"
H121_LAYERS = [7, 11]
H121_GENE_CAP = 170
H121_NEIGHBORS = 12
H121_EDGE_SAMPLE = 240
H121_CV_SPLITS = 4
H121_NULL_PERM = 12

# H122 / N609: cross-model landscape transport (major objective reset).
H122_SEED = "seed42_main"
H122_LAYERS = [7, 11]
H122_GENE_CAP = 220
H122_NEIGHBORS = 12
H122_MODULE_MIN_SIZE = 8
H122_MODULE_MAX_SIZE = 48
H122_MAX_MODULES = 32
H122_NULL_PERM = 16

# H123 / N600: strict hardening of H118 with dual-axis split + stricter nulls.
H123_SEEDS = ["seed42_main", "seed43", "seed44"]
H123_LAYER = 11
H123_GENE_CAP = 190
H123_NEIGHBORS = 12
H123_EDGE_SAMPLE = 260
H123_CV_SPLITS = 4
H123_NULL_PERM = 12


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base")


def ensure_required_inputs() -> None:
    required = [
        BASE.DOROTHEA_PATH,
        BASE.GENE2GO_PATH,
        BASE.OMNIPATH_INTERACTIONS_PATH,
        BASE.STRING_CACHE_PATH,
        TRRUST_PATH,
    ]
    for domain, run_map in BASE.SCGPT_RUNS_BY_DOMAIN.items():
        for seed_tag in set(H123_SEEDS + [H121_SEED, H122_SEED]):
            run_dir = run_map[seed_tag]
            required.append(run_dir / "cycle1_edge_dataset.tsv")
            required.append(run_dir / "layer_gene_embeddings.npy")
        required.append(BASE.GENEFORMER_EDGE_BY_DOMAIN[domain])
    missing = [str(p) for p in required if not Path(p).exists()]
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


def cv_auc_logit(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    n_splits: int,
    l1_c: float = 0.25,
) -> float:
    x = np.asarray(features, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
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
            max_iter=1600,
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


def quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    ranks = pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).to_numpy(dtype=float)
    bins = np.minimum((ranks * n_bins).astype(int), n_bins - 1)
    return bins.astype(int)


def permute_within_strata(values: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(values).copy()
    s = np.asarray(strata, dtype=int)
    for g in np.unique(s):
        idx = np.where(s == g)[0]
        if idx.size > 1:
            out[idx] = out[rng.permutation(idx)]
    return out


def permute_rows_within_strata(matrix: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(matrix, dtype=float).copy()
    s = np.asarray(strata, dtype=int)
    for g in np.unique(s):
        idx = np.where(s == g)[0]
        if idx.size > 1:
            out[idx] = out[rng.permutation(idx)]
    return out


def edge_degree_sum(points: np.ndarray, n_neighbors: int, source_local: np.ndarray, target_local: np.ndarray) -> np.ndarray:
    knn_edges = BASE.build_knn_edge_array(points=points, n_neighbors=n_neighbors)
    neighbors = BASE.adjacency_neighbors(points.shape[0], knn_edges)
    deg = np.asarray([len(nb) for nb in neighbors], dtype=float)
    return deg[source_local] + deg[target_local]


def build_edge_strata(edge_length: np.ndarray, degree_sum: np.ndarray, max_len_bins: int, max_deg_bins: int) -> np.ndarray:
    bins_len = BASE.degree_bins(edge_length, max_bins=max_len_bins)
    bins_deg = BASE.degree_bins(degree_sum, max_bins=max_deg_bins)
    return (bins_len * 16 + bins_deg).astype(int)


def fit_linear_map(src: np.ndarray, dst: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    x = np.asarray(src, dtype=float)
    y = np.asarray(dst, dtype=float)
    lhs = x.T @ x + float(l2) * np.eye(x.shape[1], dtype=float)
    rhs = x.T @ y
    return np.linalg.solve(lhs, rhs)


def label_propagation_communities(
    neighbors: list[set[int]],
    rng: np.random.Generator,
    max_iter: int = 20,
) -> np.ndarray:
    # Lightweight graph community proxy for fast screening.
    n = len(neighbors)
    labels = np.arange(n, dtype=int)
    order = np.arange(n, dtype=int)
    for _ in range(max_iter):
        changed = False
        order = rng.permutation(order)
        for node in order:
            nbrs = list(neighbors[int(node)])
            if not nbrs:
                continue
            lab = labels[np.asarray(nbrs, dtype=int)]
            vals, cnts = np.unique(lab, return_counts=True)
            max_count = int(np.max(cnts))
            candidate = vals[cnts == max_count]
            new_label = int(rng.choice(candidate))
            if new_label != labels[int(node)]:
                labels[int(node)] = new_label
                changed = True
        if not changed:
            break
    uniq = np.unique(labels)
    remap = {int(v): int(i) for i, v in enumerate(uniq)}
    return np.asarray([remap[int(v)] for v in labels], dtype=int)


def load_trrust_signed_map() -> tuple[dict[tuple[str, str], int], dict[str, int]]:
    df = pd.read_csv(
        TRRUST_PATH,
        sep="\t",
        header=None,
        names=["source", "target", "mode", "pmid"],
        dtype={"source": str, "target": str, "mode": str},
    )
    mode_map = {"ACTIVATION": 1, "REPRESSION": -1}
    signed: dict[tuple[str, str], int] = {}
    out_degree: dict[str, int] = {}

    for row in df.itertuples(index=False):
        src = str(row.source).upper()
        tgt = str(row.target).upper()
        sign = mode_map.get(str(row.mode).upper(), 0)
        if sign == 0:
            continue
        key = (src, tgt)
        if key in signed and signed[key] != sign:
            continue
        signed[key] = sign
        out_degree[src] = out_degree.get(src, 0) + 1
    return signed, out_degree


def domain_split_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (domain, split_regime), group in df.groupby(["domain", "split_regime"], sort=True):
        rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(group.shape[0]),
                "mean_delta_vs_h70": float(group["delta_vs_h70"].mean()),
                "mean_null_gap_q95": float(group["null_gap_q95"].mean()),
                "fraction_delta_positive": float((group["delta_vs_h70"] > 0.0).mean()),
                "fraction_null_gap_positive": float((group["null_gap_q95"] > 0.0).mean()),
                "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
            }
        )
    return pd.DataFrame(rows).sort_values(["domain", "split_regime"])


def build_symbol_resources(
    split_edges: pd.DataFrame,
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> tuple[np.ndarray, dict[int, int], list[str], np.ndarray]:
    edge_gene_indices = np.unique(
        np.concatenate(
            [
                split_edges["source_idx"].to_numpy(dtype=int),
                split_edges["target_idx"].to_numpy(dtype=int),
            ]
        )
    )
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
    return edge_gene_indices, gene_to_local, symbols, support_dir


def build_split_masks_plus(edge_df: pd.DataFrame) -> dict[str, np.ndarray]:
    masks = BASE.build_split_masks(edge_df)
    source_threshold = float(edge_df["source_idx"].median())
    target_threshold = float(edge_df["target_idx"].median())
    masks["dual_axis_disjoint"] = (
        edge_df["source_idx"].to_numpy(dtype=float) <= source_threshold
    ) & (edge_df["target_idx"].to_numpy(dtype=float) > target_threshold)
    return masks


def build_directed_knn_weighted_graph(
    points: np.ndarray,
    support_dir: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, csr_matrix]:
    edges = BASE.build_knn_edge_array(points=points, n_neighbors=n_neighbors)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for u, v in edges:
        iu = int(u)
        iv = int(v)
        dist = float(np.linalg.norm(points[iu] - points[iv]))
        w_uv = dist / (0.35 + float(support_dir[iu, iv]))
        w_vu = dist / (0.35 + float(support_dir[iv, iu]))
        rows.extend([iu, iv])
        cols.extend([iv, iu])
        vals.extend([w_uv, w_vu])
    graph = csr_matrix((vals, (rows, cols)), shape=(points.shape[0], points.shape[0]))
    return edges, graph


def path_nodes_from_predecessor(predecessors: np.ndarray, src: int, tgt: int) -> list[int]:
    if src == tgt:
        return [src]
    cur = int(tgt)
    path = [cur]
    max_hops = predecessors.shape[0] + 3
    hops = 0
    while cur != src and hops < max_hops:
        prev = int(predecessors[src, cur])
        if prev < 0:
            break
        path.append(prev)
        cur = prev
        hops += 1
    if path[-1] != src:
        return [src, tgt]
    return path[::-1]


def mean_path_support(path: list[int], support_dir: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    vals = [float(support_dir[int(u), int(v)]) for u, v in zip(path[:-1], path[1:])]
    return float(np.mean(vals)) if vals else 0.0


def directional_path_features(
    source_local: np.ndarray,
    target_local: np.ndarray,
    dist_mat: np.ndarray,
    predecessors: np.ndarray,
    support_dir: np.ndarray,
) -> np.ndarray:
    # Features encode directional asymmetry between source->target and reverse geodesics.
    eps = 1e-8
    feat = np.zeros((source_local.size, 7), dtype=float)
    for i, (s, t) in enumerate(zip(source_local, target_local)):
        src = int(s)
        tgt = int(t)
        d_fwd = float(dist_mat[src, tgt])
        d_rev = float(dist_mat[tgt, src])
        if not np.isfinite(d_fwd):
            d_fwd = float(np.nanmax(dist_mat[np.isfinite(dist_mat)])) if np.isfinite(dist_mat).any() else 1.0
        if not np.isfinite(d_rev):
            d_rev = float(np.nanmax(dist_mat[np.isfinite(dist_mat)])) if np.isfinite(dist_mat).any() else 1.0

        p_fwd = path_nodes_from_predecessor(predecessors, src=src, tgt=tgt)
        p_rev = path_nodes_from_predecessor(predecessors, src=tgt, tgt=src)
        hop_fwd = float(max(1, len(p_fwd) - 1))
        hop_rev = float(max(1, len(p_rev) - 1))
        sup_fwd = mean_path_support(p_fwd, support_dir)
        sup_rev = mean_path_support(p_rev, support_dir)

        ratio = float(np.log((d_rev + eps) / (d_fwd + eps)))
        hop_gap = float(hop_rev - hop_fwd)
        sup_gap = float(sup_fwd - sup_rev)
        dist_mean = 0.5 * (d_fwd + d_rev)
        dist_abs_gap = abs(d_fwd - d_rev)
        margin = float(support_dir[src, tgt] - support_dir[tgt, src])

        feat[i] = np.asarray([ratio, d_fwd, d_rev, hop_gap, sup_gap, dist_mean, margin], dtype=float)
    return feat


def swapped_directional_features(feat: np.ndarray) -> np.ndarray:
    x = np.asarray(feat, dtype=float)
    out = x.copy()
    out[:, 0] = -x[:, 0]  # log distance ratio flips under source/target swap
    out[:, 1] = x[:, 2]   # d_fwd <-> d_rev
    out[:, 2] = x[:, 1]
    out[:, 3] = -x[:, 3]  # hop gap flips
    out[:, 4] = -x[:, 4]  # support gap flips
    out[:, 5] = x[:, 5]   # symmetric mean distance
    out[:, 6] = -x[:, 6]  # direct support margin flips
    return out


def flip_directional_columns_within_bins(
    feat: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
    directional_cols: tuple[int, ...] = (0, 3, 4, 6),
) -> np.ndarray:
    out = np.asarray(feat, dtype=float).copy()
    s = np.asarray(strata, dtype=int)
    for g in np.unique(s):
        idx = np.where(s == g)[0]
        if idx.size <= 1:
            continue
        signs = rng.choice(np.array([-1.0, 1.0]), size=idx.size, replace=True)
        for col in directional_cols:
            out[idx, col] = out[idx, col] * signs
    return out


def run_h121_directional_geodesic_asymmetry(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_idx, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H121_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_idx, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H121_GENE_CAP))
            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
            ].copy()
            if split_edges["label"].nunique() < 2:
                continue

            edge_gene_indices, gene_to_local, symbols, support_dir = build_symbol_resources(
                split_edges=split_edges,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )
            if edge_gene_indices.size < 120:
                continue

            source_local_all = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
            target_local_all = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
            labels_all = split_edges["label"].to_numpy(dtype=int)

            rng = np.random.default_rng(46_100 + domain_idx * 1000 + split_idx * 100)
            sample_idx = stratified_index_sample(labels_all, max_n=H121_EDGE_SAMPLE, rng=rng)
            if sample_idx.size < 120:
                continue

            source_local = source_local_all[sample_idx]
            target_local = target_local_all[sample_idx]
            labels = labels_all[sample_idx]

            for layer in H121_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=46_101 + domain_idx * 1000 + split_idx * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H121_NEIGHBORS)
                geodesic_w = confidence_weighted_geodesic(geodesic, support_dir)

                h70 = compute_h70_scores(
                    geodesic=geodesic_w,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=[8, 12, 16],
                )

                _, directed_graph = build_directed_knn_weighted_graph(
                    points=points_pca,
                    support_dir=support_dir,
                    n_neighbors=H121_NEIGHBORS,
                )
                dist_dir, pred_dir = shortest_path(
                    directed_graph,
                    directed=True,
                    unweighted=False,
                    return_predecessors=True,
                )
                asym_feat = directional_path_features(
                    source_local=source_local,
                    target_local=target_local,
                    dist_mat=dist_dir,
                    predecessors=pred_dir,
                    support_dir=support_dir,
                )

                model_feat = np.column_stack([h70, asym_feat])
                auc_h70 = BASE.safe_auc(labels, h70)
                auc_model = cv_auc_logit(
                    model_feat,
                    labels,
                    random_state=46_102 + domain_idx * 1000 + split_idx * 100 + layer,
                    n_splits=H121_CV_SPLITS,
                )
                delta = float(auc_model - auc_h70) if np.isfinite(auc_model) and np.isfinite(auc_h70) else float("nan")

                dist_bins = quantile_bins(asym_feat[:, 5], n_bins=6)
                deg_sum = edge_degree_sum(points_pca, H121_NEIGHBORS, source_local, target_local)
                edge_len = geodesic_w[source_local, target_local]
                edge_strata = build_edge_strata(edge_len, deg_sum, max_len_bins=6, max_deg_bins=4)

                null_flip = np.empty(H121_NULL_PERM, dtype=float)
                null_swap = np.empty(H121_NULL_PERM, dtype=float)
                null_label = np.empty(H121_NULL_PERM, dtype=float)

                for perm_idx in range(H121_NULL_PERM):
                    feat_flip = flip_directional_columns_within_bins(asym_feat, strata=dist_bins, rng=rng)
                    auc_flip = cv_auc_logit(
                        np.column_stack([h70, feat_flip]),
                        labels,
                        random_state=46_103 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H121_CV_SPLITS,
                    )
                    null_flip[perm_idx] = (
                        float(auc_flip - auc_h70) if np.isfinite(auc_flip) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H121",
                            "null_kind": "direction_flip_within_path_length_bins",
                            "domain": domain,
                            "seed_tag": H121_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_flip[perm_idx]),
                        }
                    )

                    feat_swap = swapped_directional_features(asym_feat)
                    feat_swap = permute_rows_within_strata(feat_swap, strata=dist_bins, rng=rng)
                    auc_swap = cv_auc_logit(
                        np.column_stack([h70, feat_swap]),
                        labels,
                        random_state=46_104 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H121_CV_SPLITS,
                    )
                    null_swap[perm_idx] = (
                        float(auc_swap - auc_h70) if np.isfinite(auc_swap) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H121",
                            "null_kind": "endpoint_swap_within_distance_bins",
                            "domain": domain,
                            "seed_tag": H121_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_swap[perm_idx]),
                        }
                    )

                    y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                    auc_lp = cv_auc_logit(
                        model_feat,
                        y_perm,
                        random_state=46_105 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H121_CV_SPLITS,
                    )
                    auc_h70_lp = BASE.safe_auc(y_perm, h70)
                    null_label[perm_idx] = (
                        float(auc_lp - auc_h70_lp)
                        if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H121",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H121_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_flip, null_swap, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_flip = BASE.empirical_upper_tail_p(delta, null_flip)
                p_swap = BASE.empirical_upper_tail_p(delta, null_swap)
                p_label = BASE.empirical_upper_tail_p(delta, null_label)
                p_best = float(np.nanmin(np.asarray([p_flip, p_swap, p_label], dtype=float)))

                rows.append(
                    {
                        "hypothesis_id": "H121",
                        "domain": domain,
                        "seed_tag": H121_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70": float(auc_h70),
                        "auc_directional_geodesic_model": float(auc_model),
                        "delta_vs_h70": float(delta),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95": float(delta - q95),
                        "p_direction_flip_upper": float(p_flip),
                        "p_endpoint_swap_upper": float(p_swap),
                        "p_label_upper": float(p_label),
                        "p_best_upper": float(p_best),
                        "mean_log_distance_ratio": float(np.mean(asym_feat[:, 0])),
                        "mean_support_gap": float(np.mean(asym_feat[:, 4])),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h121_directional_geodesic_asymmetry_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h121_directional_geodesic_asymmetry_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_df = domain_split_summary(by_row_df)
    summary_path = ITER_DIR / "h121_directional_geodesic_asymmetry_domain_summary.csv"
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


def build_go_modules(
    symbols: list[str],
    gene2go_upper: dict[str, set[str]],
    min_size: int,
    max_size: int,
    max_modules: int,
) -> list[tuple[str, np.ndarray]]:
    term_to_indices: dict[str, list[int]] = {}
    upper_symbols = [s.upper() for s in symbols]
    for idx, sym in enumerate(upper_symbols):
        for term in gene2go_upper.get(sym, set()):
            term_to_indices.setdefault(str(term), []).append(int(idx))

    modules: list[tuple[str, np.ndarray]] = []
    for term, idxs in term_to_indices.items():
        uniq = np.array(sorted(set(int(v) for v in idxs)), dtype=int)
        if min_size <= uniq.size <= max_size:
            modules.append((term, uniq))
    modules.sort(key=lambda item: (-item[1].size, item[0]))
    return modules[:max_modules]


def module_landscape_vector(dist_mat: np.ndarray, module_idx: np.ndarray) -> np.ndarray:
    if module_idx.size < 3:
        return np.zeros(7, dtype=float)
    sub = dist_mat[np.ix_(module_idx, module_idx)]
    tri = sub[np.triu_indices(module_idx.size, k=1)]
    tri = tri[np.isfinite(tri)]
    if tri.size == 0:
        return np.zeros(7, dtype=float)
    return np.asarray(
        [
            float(np.quantile(tri, 0.10)),
            float(np.quantile(tri, 0.25)),
            float(np.quantile(tri, 0.50)),
            float(np.quantile(tri, 0.75)),
            float(np.quantile(tri, 0.90)),
            float(np.mean(tri)),
            float(np.std(tri)),
        ],
        dtype=float,
    )


def build_geneformer_distance_matrix(gf_edges: pd.DataFrame, symbols: list[str]) -> np.ndarray:
    symbol_to_idx = {str(sym).upper(): i for i, sym in enumerate(symbols)}
    use = gf_edges.copy()
    use["source"] = use["source"].astype(str).str.upper()
    use["target"] = use["target"].astype(str).str.upper()
    use = use.loc[use["source"].isin(symbol_to_idx) & use["target"].isin(symbol_to_idx)].copy()
    if use.empty:
        return np.full((len(symbols), len(symbols)), np.inf, dtype=float)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for row in use.itertuples(index=False):
        s = symbol_to_idx[str(row.source)]
        t = symbol_to_idx[str(row.target)]
        if s == t:
            continue
        rows.extend([s, t])
        cols.extend([t, s])
        vals.extend([1.0, 1.0])
    graph = csr_matrix((vals, (rows, cols)), shape=(len(symbols), len(symbols)))
    dist = shortest_path(graph, directed=False, unweighted=True)
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return np.full_like(dist, fill_value=float(len(symbols)), dtype=float)
    fill = float(np.max(finite) + 1.0)
    dist = np.where(np.isfinite(dist), dist, fill)
    return dist.astype(float)


def run_h122_cross_model_landscape_transport(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_idx, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H122_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)
        gf_df = pd.read_csv(BASE.GENEFORMER_EDGE_BY_DOMAIN[domain], sep="\t")

        for split_idx, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H122_GENE_CAP))
            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
            ].copy()
            if split_edges.empty:
                continue

            idx_symbol = BASE.build_symbol_map(split_edges)
            gf_symbols = set(gf_df["source"].astype(str).str.upper()) | set(gf_df["target"].astype(str).str.upper())
            common_gene_ids = [g for g in sorted(idx_symbol.keys()) if idx_symbol[g].upper() in gf_symbols]
            if len(common_gene_ids) < 130:
                continue

            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(common_gene_ids) & split_edges["target_idx"].isin(common_gene_ids)
            ].copy()
            if split_edges.empty:
                continue

            edge_gene_indices, _, symbols, support_dir = build_symbol_resources(
                split_edges=split_edges,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )
            if edge_gene_indices.size < 130:
                continue

            gf_dist = build_geneformer_distance_matrix(gf_edges=gf_df, symbols=symbols)
            modules = build_go_modules(
                symbols=symbols,
                gene2go_upper=gene2go_upper,
                min_size=H122_MODULE_MIN_SIZE,
                max_size=H122_MODULE_MAX_SIZE,
                max_modules=H122_MAX_MODULES,
            )
            if len(modules) < 12:
                continue

            for layer in H122_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue
                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=46_202 + domain_idx * 1000 + split_idx * 100 + layer,
                )
                sc_geo = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H122_NEIGHBORS)
                sc_geo = confidence_weighted_geodesic(sc_geo, support_dir)

                sc_vecs: list[np.ndarray] = []
                gf_vecs: list[np.ndarray] = []
                module_sizes: list[int] = []
                for _, idxs in modules:
                    sc_vecs.append(module_landscape_vector(sc_geo, idxs))
                    gf_vecs.append(module_landscape_vector(gf_dist, idxs))
                    module_sizes.append(int(idxs.size))

                sc_arr = np.vstack(sc_vecs).astype(float)
                gf_arr = np.vstack(gf_vecs).astype(float)
                m = sc_arr.shape[0]
                if m < 12:
                    continue

                rng = np.random.default_rng(46_203 + domain_idx * 1000 + split_idx * 100 + layer)
                perm = rng.permutation(m)
                n_train = int(np.floor(0.6 * m))
                n_train = max(6, min(n_train, m - 4))
                train_idx = perm[:n_train]
                test_idx = perm[n_train:]
                if test_idx.size < 4:
                    continue

                map_obs = fit_linear_map(gf_arr[train_idx], sc_arr[train_idx], l2=1e-3)
                pred_obs = gf_arr[test_idx] @ map_obs
                mse_obs = float(np.mean((pred_obs - sc_arr[test_idx]) ** 2))
                observed = float(-mse_obs)

                null_map = np.empty(H122_NULL_PERM, dtype=float)
                null_bin = np.empty(H122_NULL_PERM, dtype=float)
                for perm_idx in range(H122_NULL_PERM):
                    row_perm = rng.permutation(m)
                    gf_map = gf_arr[row_perm]
                    map_rand = fit_linear_map(gf_map[train_idx], sc_arr[train_idx], l2=1e-3)
                    pred_rand = gf_map[test_idx] @ map_rand
                    mse_rand = float(np.mean((pred_rand - sc_arr[test_idx]) ** 2))
                    null_map[perm_idx] = -mse_rand
                    null_rows.append(
                        {
                            "hypothesis_id": "H122",
                            "null_kind": "random_module_mapping",
                            "domain": domain,
                            "seed_tag": H122_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_map[perm_idx]),
                        }
                    )

                    gf_bin = gf_arr.copy()
                    for row_i in range(gf_bin.shape[0]):
                        gf_bin[row_i] = gf_bin[row_i, rng.permutation(gf_bin.shape[1])]
                    map_bin = fit_linear_map(gf_bin[train_idx], sc_arr[train_idx], l2=1e-3)
                    pred_bin = gf_bin[test_idx] @ map_bin
                    mse_bin = float(np.mean((pred_bin - sc_arr[test_idx]) ** 2))
                    null_bin[perm_idx] = -mse_bin
                    null_rows.append(
                        {
                            "hypothesis_id": "H122",
                            "null_kind": "landscape_bin_permutation",
                            "domain": domain,
                            "seed_tag": H122_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_bin[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_map, null_bin])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_map = BASE.empirical_upper_tail_p(observed, null_map)
                p_bin = BASE.empirical_upper_tail_p(observed, null_bin)
                p_best = float(np.nanmin(np.asarray([p_map, p_bin], dtype=float)))

                rows.append(
                    {
                        "hypothesis_id": "H122",
                        "domain": domain,
                        "seed_tag": H122_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_modules_eval": int(m),
                        "n_modules_train": int(train_idx.size),
                        "n_modules_test": int(test_idx.size),
                        "delta_vs_h70": float(observed),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95": float(observed - q95),
                        "p_mapping_upper": float(p_map),
                        "p_landscape_bin_upper": float(p_bin),
                        "p_best_upper": float(p_best),
                        "mse_observed": float(mse_obs),
                        "mean_module_size": float(np.mean(module_sizes)),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h122_landscape_transport_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h122_landscape_transport_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_df = domain_split_summary(by_row_df)
    summary_path = ITER_DIR / "h122_landscape_transport_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_transport_score": float(by_row_df["delta_vs_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_vs_h70"] > 0.0).sum()) if not summary_df.empty else 0,
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95"] > 0.0).sum()) if not summary_df.empty else 0,
        "artifact_paths": {
            "by_domain_split_layer": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def shuffle_sign_within_tf(
    trrust_sign: np.ndarray,
    src_symbols: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.asarray(trrust_sign, dtype=int).copy()
    by_tf: dict[str, list[int]] = {}
    for i, tf in enumerate(src_symbols):
        by_tf.setdefault(str(tf), []).append(i)
    for idxs in by_tf.values():
        idx = np.asarray(idxs, dtype=int)
        if idx.size > 1:
            out[idx] = out[rng.permutation(idx)]
    return out


def run_h123_signed_motif_module_hardening(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
    trrust_sign_map: dict[tuple[str, str], int],
    trrust_tf_out_degree: dict[str, int],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_idx, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_idx, seed_tag in enumerate(H123_SEEDS):
            run_dir = run_map[seed_tag]
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            split_masks = build_split_masks_plus(edge_df)

            for split_idx, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H123_GENE_CAP))
                split_edges = split_edges.loc[
                    split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
                ].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                edge_gene_indices, gene_to_local, symbols, support_dir = build_symbol_resources(
                    split_edges=split_edges,
                    dorothea_map=dorothea_map,
                    omnipath_pairs=omnipath_pairs,
                    gene2go_upper=gene2go_upper,
                    string_map=string_map,
                )
                if edge_gene_indices.size < 120:
                    continue

                source_local_all = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
                target_local_all = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
                labels_all = split_edges["label"].to_numpy(dtype=int)

                rng = np.random.default_rng(46_300 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100)
                sample_idx = stratified_index_sample(labels_all, max_n=H123_EDGE_SAMPLE, rng=rng)
                if sample_idx.size < 120:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                if H123_LAYER >= layer_embeddings.shape[0]:
                    continue

                points = layer_embeddings[H123_LAYER, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=46_301 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H123_NEIGHBORS)
                geodesic_w = confidence_weighted_geodesic(geodesic, support_dir)

                h70 = compute_h70_scores(
                    geodesic=geodesic_w,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=[8, 12, 16],
                )

                knn_edges = BASE.build_knn_edge_array(points=points_pca, n_neighbors=H123_NEIGHBORS)
                neighbors = BASE.adjacency_neighbors(points_pca.shape[0], knn_edges)
                community_labels = label_propagation_communities(neighbors=neighbors, rng=rng, max_iter=20)
                same_community = (community_labels[source_local] == community_labels[target_local]).astype(float)

                src_sym = [symbols[i].upper() for i in source_local]
                tgt_sym = [symbols[i].upper() for i in target_local]
                trrust_sign = np.asarray([trrust_sign_map.get((s, t), 0) for s, t in zip(src_sym, tgt_sym)], dtype=int)
                motif_present = (trrust_sign != 0).astype(float)

                edge_margin = support_dir[source_local, target_local] - support_dir[target_local, source_local]
                margin_sign = np.sign(edge_margin)
                sign_consistent = ((trrust_sign * margin_sign) > 0).astype(float)

                feat = np.column_stack(
                    [
                        h70,
                        same_community,
                        motif_present,
                        sign_consistent,
                        same_community * motif_present,
                        same_community * sign_consistent,
                        h70 * same_community * sign_consistent,
                    ]
                )

                auc_h70 = BASE.safe_auc(labels, h70)
                auc_model = cv_auc_logit(
                    feat,
                    labels,
                    random_state=46_302 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100,
                    n_splits=H123_CV_SPLITS,
                )
                delta = float(auc_model - auc_h70) if np.isfinite(auc_model) and np.isfinite(auc_h70) else float("nan")

                tf_deg = np.asarray([trrust_tf_out_degree.get(s, 0) for s in src_sym], dtype=float)
                tf_bins = BASE.degree_bins(tf_deg, max_bins=4)

                tgt_deg_proxy = np.sum(support_dir > 0.55, axis=0).astype(float)
                tgt_deg_bins = BASE.degree_bins(tgt_deg_proxy[target_local], max_bins=4)
                motif_strata = (tf_bins * 16 + tgt_deg_bins).astype(int)

                edge_len = geodesic_w[source_local, target_local]
                deg_sum = edge_degree_sum(points_pca, H123_NEIGHBORS, source_local, target_local)
                edge_strata = build_edge_strata(edge_len, deg_sum, max_len_bins=6, max_deg_bins=4)

                null_tf_sign = np.empty(H123_NULL_PERM, dtype=float)
                null_motif_decoy = np.empty(H123_NULL_PERM, dtype=float)
                null_label = np.empty(H123_NULL_PERM, dtype=float)

                for perm_idx in range(H123_NULL_PERM):
                    sign_perm = shuffle_sign_within_tf(trrust_sign, src_symbols=src_sym, rng=rng)
                    motif_perm_tf = (sign_perm != 0).astype(float)
                    sign_cons_perm_tf = ((sign_perm * margin_sign) > 0).astype(float)
                    feat_tf = np.column_stack(
                        [
                            h70,
                            same_community,
                            motif_perm_tf,
                            sign_cons_perm_tf,
                            same_community * motif_perm_tf,
                            same_community * sign_cons_perm_tf,
                            h70 * same_community * sign_cons_perm_tf,
                        ]
                    )
                    auc_tf = cv_auc_logit(
                        feat_tf,
                        labels,
                        random_state=46_303 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H123_CV_SPLITS,
                    )
                    null_tf_sign[perm_idx] = (
                        float(auc_tf - auc_h70) if np.isfinite(auc_tf) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H123",
                            "null_kind": "tf_identity_preserving_sign_shuffle",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H123_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_tf_sign[perm_idx]),
                        }
                    )

                    motif_perm = permute_within_strata(motif_present, strata=motif_strata, rng=rng).astype(float)
                    sign_cons_pool = permute_within_strata(sign_consistent, strata=motif_strata, rng=rng).astype(float)
                    sign_cons_perm = motif_perm * sign_cons_pool
                    feat_decoy = np.column_stack(
                        [
                            h70,
                            same_community,
                            motif_perm,
                            sign_cons_perm,
                            same_community * motif_perm,
                            same_community * sign_cons_perm,
                            h70 * same_community * sign_cons_perm,
                        ]
                    )
                    auc_decoy = cv_auc_logit(
                        feat_decoy,
                        labels,
                        random_state=46_304 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H123_CV_SPLITS,
                    )
                    null_motif_decoy[perm_idx] = (
                        float(auc_decoy - auc_h70) if np.isfinite(auc_decoy) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H123",
                            "null_kind": "motif_decoy_shuffle_matched_tf_target_degree",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H123_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_motif_decoy[perm_idx]),
                        }
                    )

                    y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                    auc_lp = cv_auc_logit(
                        feat,
                        y_perm,
                        random_state=46_305 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H123_CV_SPLITS,
                    )
                    auc_h70_lp = BASE.safe_auc(y_perm, h70)
                    null_label[perm_idx] = (
                        float(auc_lp - auc_h70_lp)
                        if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H123",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H123_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_tf_sign, null_motif_decoy, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_tf = BASE.empirical_upper_tail_p(delta, null_tf_sign)
                p_decoy = BASE.empirical_upper_tail_p(delta, null_motif_decoy)
                p_label = BASE.empirical_upper_tail_p(delta, null_label)
                p_best = float(np.nanmin(np.asarray([p_tf, p_decoy, p_label], dtype=float)))

                rows.append(
                    {
                        "hypothesis_id": "H123",
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(H123_LAYER),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70": float(auc_h70),
                        "auc_signed_module_model": float(auc_model),
                        "delta_vs_h70": float(delta),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95": float(delta - q95),
                        "p_tf_sign_upper": float(p_tf),
                        "p_motif_decoy_upper": float(p_decoy),
                        "p_label_upper": float(p_label),
                        "p_best_upper": float(p_best),
                        "motif_coverage": float(np.mean(motif_present)),
                        "sign_consistency_rate": float(np.mean(sign_consistent[motif_present > 0]))
                        if np.any(motif_present > 0)
                        else 0.0,
                        "same_community_rate": float(np.mean(same_community)),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["seed_tag", "domain", "split_regime"])
    by_row_path = ITER_DIR / "h123_signed_motif_module_hardening_by_seed_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "seed_tag", "domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h123_signed_motif_module_hardening_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_df = domain_split_summary(by_row_df)
    summary_path = ITER_DIR / "h123_signed_motif_module_hardening_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_vs_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_vs_h70"] > 0.0).sum()) if not summary_df.empty else 0,
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95"] > 0.0).sum()) if not summary_df.empty else 0,
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
    trrust_sign_map, trrust_tf_out_degree = load_trrust_signed_map()

    h121 = run_h121_directional_geodesic_asymmetry(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h122 = run_h122_cross_model_landscape_transport(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h123 = run_h123_signed_motif_module_hardening(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
        trrust_sign_map=trrust_sign_map,
        trrust_tf_out_degree=trrust_tf_out_degree,
    )

    summary = {
        "iteration": "iter_0046",
        "h121": h121,
        "h122": h122,
        "h123": h123,
    }
    summary_path = ITER_DIR / "iter0046_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
