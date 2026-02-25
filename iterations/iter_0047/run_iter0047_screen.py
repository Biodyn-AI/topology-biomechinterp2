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


ITER_DIR = Path("iterations/iter_0047")
ITER_DIR.mkdir(parents=True, exist_ok=True)

TRRUST_PATH = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
    "single_cell_mechinterp/external/networks/trrust_human.tsv"
)

# H124 / N625: signed motif-community hardening with STRING conditioning.
H124_SEEDS = ["seed42_main", "seed43", "seed44"]
H124_LAYER = 11
H124_GENE_CAP = 200
H124_GENE_CAP_LUNG_DUAL = 260
H124_MIN_GENE_NODES = 120
H124_MIN_GENE_NODES_LUNG_DUAL = 90
H124_NEIGHBORS = 12
H124_EDGE_SAMPLE = 240
H124_EDGE_SAMPLE_LUNG_DUAL = 220
H124_CV_SPLITS = 3
H124_NULL_PERM = 64

# H125 / N622: anchor-constrained cycle-consistent cross-model alignment.
H125_SEED = "seed42_main"
H125_LAYERS = [7, 11]
H125_GENE_CAP = 220
H125_MIN_SHARED = 90
H125_NULL_PERM = 32
H125_MAP_ITERS = 120
H125_MAP_ITERS_NULL = 52
H125_SPLITS = ("source_disjoint", "target_disjoint")

# H126 / N620: geodesic torsion and turning-angle asymmetry.
H126_SEED = "seed42_main"
H126_LAYERS = [7, 11]
H126_GENE_CAP = 170
H126_NEIGHBORS = 12
H126_EDGE_SAMPLE = 240
H126_CV_SPLITS = 4
H126_NULL_PERM = 24


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
        for seed_tag in set(H124_SEEDS + [H125_SEED, H126_SEED]):
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
    l1_c: float = 0.30,
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


def label_propagation_communities(
    neighbors: list[set[int]],
    rng: np.random.Generator,
    max_iter: int = 20,
) -> np.ndarray:
    # Lightweight graph community proxy used throughout this screening loop.
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


def domain_only_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for domain, group in df.groupby("domain", sort=True):
        rows.append(
            {
                "domain": domain,
                "n_rows": int(group.shape[0]),
                "mean_delta_vs_h70": float(group["delta_vs_h70"].mean()),
                "mean_null_gap_q95": float(group["null_gap_q95"].mean()),
                "mean_cycle_consistency_error": float(group["cycle_consistency_error"].mean()),
                "fraction_delta_positive": float((group["delta_vs_h70"] > 0.0).mean()),
                "fraction_null_gap_positive": float((group["null_gap_q95"] > 0.0).mean()),
                "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
            }
        )
    return pd.DataFrame(rows).sort_values(["domain"])


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


def _safe_unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros_like(v)
    return v / norm


def turn_torsion_stats(path_points: np.ndarray) -> tuple[float, float, float, float]:
    if path_points.shape[0] < 3:
        return 0.0, 0.0, 0.0, 0.0

    vec = np.diff(path_points, axis=0)
    if vec.shape[0] < 2:
        return 0.0, 0.0, 0.0, 0.0
    unit = np.vstack([_safe_unit(v) for v in vec])

    turn_angles: list[float] = []
    for i in range(unit.shape[0] - 1):
        dot = float(np.clip(np.dot(unit[i], unit[i + 1]), -1.0, 1.0))
        turn_angles.append(float(np.arccos(dot)))

    torsion_signed: list[float] = []
    if unit.shape[0] >= 3:
        for i in range(unit.shape[0] - 2):
            n1 = np.cross(unit[i], unit[i + 1])
            n2 = np.cross(unit[i + 1], unit[i + 2])
            d = float(np.linalg.norm(n1) * np.linalg.norm(n2))
            if d < 1e-10:
                continue
            cosv = float(np.clip(np.dot(n1, n2) / d, -1.0, 1.0))
            sign = float(np.sign(np.dot(np.cross(n1, n2), unit[i + 1])))
            torsion_signed.append(sign * float(np.arccos(cosv)))

    turn_arr = np.asarray(turn_angles, dtype=float)
    tors_arr = np.asarray(torsion_signed, dtype=float)
    turn_mean = float(np.mean(turn_arr)) if turn_arr.size else 0.0
    turn_std = float(np.std(turn_arr)) if turn_arr.size else 0.0
    tors_abs = float(np.mean(np.abs(tors_arr))) if tors_arr.size else 0.0
    tors_signed_mean = float(np.mean(tors_arr)) if tors_arr.size else 0.0
    return turn_mean, turn_std, tors_abs, tors_signed_mean


def torsion_directional_features(
    source_local: np.ndarray,
    target_local: np.ndarray,
    dist_mat: np.ndarray,
    predecessors: np.ndarray,
    support_dir: np.ndarray,
    points_pca: np.ndarray,
) -> np.ndarray:
    # Directional geodesic-path descriptors: turning-angle and torsion asymmetry.
    eps = 1e-8
    xyz = np.asarray(points_pca[:, :3], dtype=float)
    feat = np.zeros((source_local.size, 17), dtype=float)

    finite_dist = dist_mat[np.isfinite(dist_mat)]
    fallback_dist = float(np.max(finite_dist)) if finite_dist.size else 1.0

    for i, (s, t) in enumerate(zip(source_local, target_local)):
        src = int(s)
        tgt = int(t)
        d_fwd = float(dist_mat[src, tgt])
        d_rev = float(dist_mat[tgt, src])
        if not np.isfinite(d_fwd):
            d_fwd = fallback_dist
        if not np.isfinite(d_rev):
            d_rev = fallback_dist

        p_fwd = path_nodes_from_predecessor(predecessors, src=src, tgt=tgt)
        p_rev = path_nodes_from_predecessor(predecessors, src=tgt, tgt=src)

        hop_fwd = float(max(1, len(p_fwd) - 1))
        hop_rev = float(max(1, len(p_rev) - 1))
        sup_fwd = mean_path_support(p_fwd, support_dir)
        sup_rev = mean_path_support(p_rev, support_dir)

        t_mean_f, t_std_f, tors_abs_f, tors_signed_f = turn_torsion_stats(xyz[np.asarray(p_fwd, dtype=int)])
        t_mean_r, t_std_r, tors_abs_r, tors_signed_r = turn_torsion_stats(xyz[np.asarray(p_rev, dtype=int)])

        ratio = float(np.log((d_rev + eps) / (d_fwd + eps)))
        hop_gap = float(hop_rev - hop_fwd)
        sup_gap = float(sup_fwd - sup_rev)
        margin = float(support_dir[src, tgt] - support_dir[tgt, src])

        feat[i] = np.asarray(
            [
                ratio,
                hop_gap,
                sup_gap,
                t_mean_f,
                t_mean_r,
                t_mean_f - t_mean_r,
                t_std_f,
                t_std_r,
                t_std_f - t_std_r,
                tors_abs_f,
                tors_abs_r,
                tors_abs_f - tors_abs_r,
                tors_signed_f,
                tors_signed_r,
                tors_signed_f - tors_signed_r,
                0.5 * (d_fwd + d_rev),
                margin,
            ],
            dtype=float,
        )
    return feat


def swapped_torsion_features(feat: np.ndarray) -> np.ndarray:
    x = np.asarray(feat, dtype=float)
    out = x.copy()
    out[:, 0] = -x[:, 0]
    out[:, 1] = -x[:, 1]
    out[:, 2] = -x[:, 2]

    out[:, 3] = x[:, 4]
    out[:, 4] = x[:, 3]
    out[:, 5] = -x[:, 5]

    out[:, 6] = x[:, 7]
    out[:, 7] = x[:, 6]
    out[:, 8] = -x[:, 8]

    out[:, 9] = x[:, 10]
    out[:, 10] = x[:, 9]
    out[:, 11] = -x[:, 11]

    out[:, 12] = x[:, 13]
    out[:, 13] = x[:, 12]
    out[:, 14] = -x[:, 14]

    out[:, 15] = x[:, 15]
    out[:, 16] = -x[:, 16]
    return out


def path_reversal_within_bins(feat: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(feat, dtype=float)
    swapped = swapped_torsion_features(x)
    out = x.copy()
    s = np.asarray(strata, dtype=int)
    for g in np.unique(s):
        idx = np.where(s == g)[0]
        if idx.size <= 1:
            continue
        choose = rng.random(idx.size) < 0.5
        if np.any(choose):
            out[idx[choose]] = swapped[idx[choose]]
    return out


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


def select_anchor_go_terms(
    symbols: list[str],
    gene2go_upper: dict[str, set[str]],
    trrust_gene_set: set[str],
    max_terms: int = 48,
) -> set[str]:
    trrust_terms: dict[str, int] = {}
    global_terms: dict[str, int] = {}
    for sym in symbols:
        sym_u = str(sym).upper()
        terms = gene2go_upper.get(sym_u, set())
        for term in terms:
            global_terms[term] = global_terms.get(term, 0) + 1
            if sym_u in trrust_gene_set:
                trrust_terms[term] = trrust_terms.get(term, 0) + 1

    ordered = sorted(trrust_terms.items(), key=lambda kv: (-kv[1], kv[0]))
    if not ordered:
        ordered = sorted(global_terms.items(), key=lambda kv: (-kv[1], kv[0]))
    return {term for term, _ in ordered[:max_terms]}


def build_anchor_weights(
    symbols: list[str],
    gene2go_upper: dict[str, set[str]],
    trrust_gene_set: set[str],
    anchor_terms: set[str],
) -> np.ndarray:
    out = np.ones(len(symbols), dtype=float)
    for i, sym in enumerate(symbols):
        sym_u = str(sym).upper()
        is_trrust = 1.0 if sym_u in trrust_gene_set else 0.0
        go_hit = 1.0 if len(gene2go_upper.get(sym_u, set()) & anchor_terms) > 0 else 0.0
        out[i] = 1.0 + 1.6 * is_trrust + 0.8 * go_hit
    return out


def build_scaffold_signature_tables(
    domain: str,
    edge_df: pd.DataFrame,
    layer_embeddings: np.ndarray,
    gf_df: pd.DataFrame,
    random_base: int,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    top_genes = set(BASE.select_top_genes(edge_df, gene_cap=H125_GENE_CAP))
    use = edge_df.loc[edge_df["source_idx"].isin(top_genes) & edge_df["target_idx"].isin(top_genes)].copy()
    if use.empty:
        return {}, pd.DataFrame()

    idx_symbol = BASE.build_symbol_map(use)
    symbols = [idx_symbol[int(g)] for g in sorted(idx_symbol.keys())]

    sc_tables: dict[int, pd.DataFrame] = {}
    gene_indices = np.array(sorted(idx_symbol.keys()), dtype=int)
    for layer in H125_LAYERS:
        points = layer_embeddings[layer, gene_indices, :]
        sc_tables[layer] = BASE.fit_signatures_scgpt(
            layer_points=points,
            symbols=symbols,
            random_state=random_base + layer,
            n_neighbors=10,
        )

    gf_table = BASE.fit_signatures_geneformer(gf_df, symbols)
    return sc_tables, gf_table


def edge_pair_cosine_scores(
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    left_mat: np.ndarray,
    right_mat: np.ndarray,
) -> np.ndarray:
    left_n = BASE.row_normalize(left_mat)
    right_n = BASE.row_normalize(right_mat)
    return np.sum(left_n[src_idx] * right_n[tgt_idx], axis=1)


def run_h124_signed_motif_string_hardening(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
    trrust_sign_map: dict[tuple[str, str], int],
    trrust_tf_out_degree: dict[str, int],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    ordered_splits = ["source_disjoint", "target_disjoint", "dual_axis_disjoint"]

    for domain_idx, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_idx, seed_tag in enumerate(H124_SEEDS):
            run_dir = run_map[seed_tag]
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            split_masks = build_split_masks_plus(edge_df)

            for split_idx, split_regime in enumerate(ordered_splits):
                split_mask = split_masks.get(split_regime)
                if split_mask is None:
                    continue

                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                is_lung_dual = domain == "lung" and split_regime == "dual_axis_disjoint"
                gene_cap = H124_GENE_CAP_LUNG_DUAL if is_lung_dual else H124_GENE_CAP
                min_gene_nodes = H124_MIN_GENE_NODES_LUNG_DUAL if is_lung_dual else H124_MIN_GENE_NODES
                sample_cap = H124_EDGE_SAMPLE_LUNG_DUAL if is_lung_dual else H124_EDGE_SAMPLE

                top_genes = set(BASE.select_top_genes(split_edges, gene_cap=gene_cap))
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
                if edge_gene_indices.size < min_gene_nodes:
                    continue

                source_local_all = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
                target_local_all = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
                labels_all = split_edges["label"].to_numpy(dtype=int)

                rng = np.random.default_rng(47_100 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100)
                sample_idx = stratified_index_sample(labels_all, max_n=sample_cap, rng=rng)
                if sample_idx.size < 110:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                if H124_LAYER >= layer_embeddings.shape[0]:
                    continue

                points = layer_embeddings[H124_LAYER, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=47_101 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H124_NEIGHBORS)
                geodesic_w = confidence_weighted_geodesic(geodesic, support_dir)

                h70 = compute_h70_scores(
                    geodesic=geodesic_w,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=[8, 12, 16],
                )

                knn_edges = BASE.build_knn_edge_array(points=points_pca, n_neighbors=H124_NEIGHBORS)
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

                string_conf = np.asarray([float(string_map.get((s, t), 0.0)) for s, t in zip(src_sym, tgt_sym)], dtype=float)
                string_high = (string_conf >= 0.70).astype(float)

                feat = np.column_stack(
                    [
                        h70,
                        same_community,
                        motif_present,
                        sign_consistent,
                        string_conf,
                        string_high,
                        same_community * motif_present,
                        same_community * sign_consistent,
                        motif_present * string_conf,
                        sign_consistent * string_conf,
                        same_community * string_conf,
                        h70 * same_community * sign_consistent,
                        h70 * sign_consistent * string_conf,
                    ]
                )

                auc_h70 = BASE.safe_auc(labels, h70)
                auc_model = cv_auc_logit(
                    feat,
                    labels,
                    random_state=47_102 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100,
                    n_splits=H124_CV_SPLITS,
                )
                delta = float(auc_model - auc_h70) if np.isfinite(auc_model) and np.isfinite(auc_h70) else float("nan")

                tf_deg = np.asarray([trrust_tf_out_degree.get(s, 0) for s in src_sym], dtype=float)
                tf_bins = BASE.degree_bins(tf_deg, max_bins=4)

                tgt_deg_proxy = np.sum(support_dir > 0.55, axis=0).astype(float)
                tgt_deg_bins = BASE.degree_bins(tgt_deg_proxy[target_local], max_bins=4)
                str_bins = quantile_bins(string_conf, n_bins=4)
                motif_strata = (tf_bins * 64 + tgt_deg_bins * 8 + str_bins).astype(int)

                edge_len = geodesic_w[source_local, target_local]
                deg_sum = edge_degree_sum(points_pca, H124_NEIGHBORS, source_local, target_local)
                edge_strata = build_edge_strata(edge_len, deg_sum, max_len_bins=6, max_deg_bins=4)

                null_tf_sign = np.empty(H124_NULL_PERM, dtype=float)
                null_motif_decoy = np.empty(H124_NULL_PERM, dtype=float)
                null_string_bin = np.empty(H124_NULL_PERM, dtype=float)
                null_label = np.empty(H124_NULL_PERM, dtype=float)

                for perm_idx in range(H124_NULL_PERM):
                    sign_perm = shuffle_sign_within_tf(trrust_sign, src_symbols=src_sym, rng=rng)
                    motif_perm_tf = (sign_perm != 0).astype(float)
                    sign_cons_perm_tf = ((sign_perm * margin_sign) > 0).astype(float)
                    feat_tf = np.column_stack(
                        [
                            h70,
                            same_community,
                            motif_perm_tf,
                            sign_cons_perm_tf,
                            string_conf,
                            string_high,
                            same_community * motif_perm_tf,
                            same_community * sign_cons_perm_tf,
                            motif_perm_tf * string_conf,
                            sign_cons_perm_tf * string_conf,
                            same_community * string_conf,
                            h70 * same_community * sign_cons_perm_tf,
                            h70 * sign_cons_perm_tf * string_conf,
                        ]
                    )
                    auc_tf = cv_auc_logit(
                        feat_tf,
                        labels,
                        random_state=47_103 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H124_CV_SPLITS,
                    )
                    null_tf_sign[perm_idx] = (
                        float(auc_tf - auc_h70) if np.isfinite(auc_tf) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H124",
                            "null_kind": "tf_identity_preserving_sign_shuffle",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H124_LAYER),
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
                            string_conf,
                            string_high,
                            same_community * motif_perm,
                            same_community * sign_cons_perm,
                            motif_perm * string_conf,
                            sign_cons_perm * string_conf,
                            same_community * string_conf,
                            h70 * same_community * sign_cons_perm,
                            h70 * sign_cons_perm * string_conf,
                        ]
                    )
                    auc_decoy = cv_auc_logit(
                        feat_decoy,
                        labels,
                        random_state=47_104 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H124_CV_SPLITS,
                    )
                    null_motif_decoy[perm_idx] = (
                        float(auc_decoy - auc_h70) if np.isfinite(auc_decoy) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H124",
                            "null_kind": "motif_decoy_shuffle_matched_tf_target_degree",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H124_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_motif_decoy[perm_idx]),
                        }
                    )

                    str_perm = permute_within_strata(string_conf, strata=motif_strata, rng=rng).astype(float)
                    str_high_perm = (str_perm >= 0.70).astype(float)
                    feat_str = np.column_stack(
                        [
                            h70,
                            same_community,
                            motif_present,
                            sign_consistent,
                            str_perm,
                            str_high_perm,
                            same_community * motif_present,
                            same_community * sign_consistent,
                            motif_present * str_perm,
                            sign_consistent * str_perm,
                            same_community * str_perm,
                            h70 * same_community * sign_consistent,
                            h70 * sign_consistent * str_perm,
                        ]
                    )
                    auc_str = cv_auc_logit(
                        feat_str,
                        labels,
                        random_state=47_105 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H124_CV_SPLITS,
                    )
                    null_string_bin[perm_idx] = (
                        float(auc_str - auc_h70) if np.isfinite(auc_str) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H124",
                            "null_kind": "string_confidence_bin_permutation_within_degree_strata",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H124_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_string_bin[perm_idx]),
                        }
                    )

                    y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                    auc_lp = cv_auc_logit(
                        feat,
                        y_perm,
                        random_state=47_106 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H124_CV_SPLITS,
                    )
                    auc_h70_lp = BASE.safe_auc(y_perm, h70)
                    null_label[perm_idx] = (
                        float(auc_lp - auc_h70_lp)
                        if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H124",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H124_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_tf_sign, null_motif_decoy, null_string_bin, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_tf = BASE.empirical_upper_tail_p(delta, null_tf_sign)
                p_decoy = BASE.empirical_upper_tail_p(delta, null_motif_decoy)
                p_string = BASE.empirical_upper_tail_p(delta, null_string_bin)
                p_label = BASE.empirical_upper_tail_p(delta, null_label)
                p_best = float(np.nanmin(np.asarray([p_tf, p_decoy, p_string, p_label], dtype=float)))

                rows.append(
                    {
                        "hypothesis_id": "H124",
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(H124_LAYER),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70": float(auc_h70),
                        "auc_signed_string_model": float(auc_model),
                        "delta_vs_h70": float(delta),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95": float(delta - q95),
                        "p_tf_sign_upper": float(p_tf),
                        "p_motif_decoy_upper": float(p_decoy),
                        "p_string_bin_upper": float(p_string),
                        "p_label_upper": float(p_label),
                        "p_best_upper": float(p_best),
                        "motif_coverage": float(np.mean(motif_present)),
                        "string_conf_mean": float(np.mean(string_conf)),
                        "same_community_rate": float(np.mean(same_community)),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["seed_tag", "domain", "split_regime"])
    by_row_path = ITER_DIR / "h124_signed_string_hardening_by_seed_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "seed_tag", "domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h124_signed_string_hardening_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_df = domain_split_summary(by_row_df)
    summary_path = ITER_DIR / "h124_signed_string_hardening_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    lung_dual = by_row_df.loc[
        (by_row_df["domain"] == "lung") & (by_row_df["split_regime"] == "dual_axis_disjoint")
    ] if not by_row_df.empty else pd.DataFrame()

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_vs_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_vs_h70"] > 0.0).sum()) if not summary_df.empty else 0,
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95"] > 0.0).sum()) if not summary_df.empty else 0,
        "lung_dual_axis_rows": int(lung_dual.shape[0]),
        "lung_dual_axis_mean_null_gap": float(lung_dual["null_gap_q95"].mean()) if not lung_dual.empty else float("nan"),
        "artifact_paths": {
            "by_seed_domain_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h125_anchor_cycle_alignment(
    gene2go_upper: dict[str, set[str]],
    trrust_sign_map: dict[tuple[str, str], int],
) -> dict[str, object]:
    by_row: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    trrust_gene_set = set()
    for src, tgt in trrust_sign_map.keys():
        trrust_gene_set.add(str(src).upper())
        trrust_gene_set.add(str(tgt).upper())

    sc_edges: dict[str, pd.DataFrame] = {}
    sc_layers: dict[str, np.ndarray] = {}
    gf_edges: dict[str, pd.DataFrame] = {}
    sc_sig: dict[tuple[str, int], pd.DataFrame] = {}
    gf_sig: dict[str, pd.DataFrame] = {}

    for domain_idx, domain in enumerate(["immune", "lung", "external_lung"]):
        run_dir = BASE.SCGPT_RUNS_BY_DOMAIN[domain][H125_SEED]
        sc_edges[domain] = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        sc_layers[domain] = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        gf_edges[domain] = pd.read_csv(BASE.GENEFORMER_EDGE_BY_DOMAIN[domain], sep="\t")

        sc_tables, gf_table = build_scaffold_signature_tables(
            domain=domain,
            edge_df=sc_edges[domain],
            layer_embeddings=sc_layers[domain],
            gf_df=gf_edges[domain],
            random_base=47_200 + domain_idx * 100,
        )
        for layer, table in sc_tables.items():
            sc_sig[(domain, layer)] = table
        gf_sig[domain] = gf_table

    domains = ["immune", "lung", "external_lung"]
    for domain_idx, target_domain in enumerate(domains):
        source_domains = [d for d in domains if d != target_domain]
        split_masks = BASE.build_split_masks(sc_edges[target_domain])

        for layer in H125_LAYERS:
            train_sc_list: list[np.ndarray] = []
            train_gf_list: list[np.ndarray] = []
            train_w_list: list[np.ndarray] = []

            for src_domain in source_domains:
                sc_df = sc_sig.get((src_domain, layer))
                gf_df = gf_sig.get(src_domain)
                if sc_df is None or gf_df is None or sc_df.empty or gf_df.empty:
                    continue

                shared = sorted(set(sc_df.index) & set(gf_df.index))
                if len(shared) < H125_MIN_SHARED:
                    continue

                anchor_terms = select_anchor_go_terms(shared, gene2go_upper, trrust_gene_set)
                weights = build_anchor_weights(shared, gene2go_upper, trrust_gene_set, anchor_terms)

                train_sc_list.append(sc_df.loc[shared].to_numpy(dtype=float))
                train_gf_list.append(gf_df.loc[shared].to_numpy(dtype=float))
                train_w_list.append(weights)

            if not train_sc_list:
                continue

            train_sc = np.vstack(train_sc_list)
            train_gf = np.vstack(train_gf_list)
            train_w = np.concatenate(train_w_list)
            if train_sc.shape[0] < 2 * H125_MIN_SHARED:
                continue

            sc_mu, sc_sd = BASE.zscore_fit(train_sc)
            gf_mu, gf_sd = BASE.zscore_fit(train_gf)
            train_sc_z = BASE.zscore_apply(train_sc, sc_mu, sc_sd)
            train_gf_z = BASE.zscore_apply(train_gf, gf_mu, gf_sd)

            map_a, map_b, align_rmse, cycle_rmse = BASE.fit_cycle_consistent_maps(
                x_sc=train_sc_z,
                y_gf=train_gf_z,
                weights=train_w,
                n_iters=H125_MAP_ITERS,
                lr=0.055,
                l2=0.05,
                cycle_weight=0.50,
            )

            sc_tgt_df = sc_sig.get((target_domain, layer))
            gf_tgt_df = gf_sig.get(target_domain)
            if sc_tgt_df is None or gf_tgt_df is None or sc_tgt_df.empty or gf_tgt_df.empty:
                continue

            shared_tgt = sorted(set(sc_tgt_df.index) & set(gf_tgt_df.index))
            if len(shared_tgt) < H125_MIN_SHARED:
                continue

            sc_tgt_z = BASE.zscore_apply(sc_tgt_df.loc[shared_tgt].to_numpy(dtype=float), sc_mu, sc_sd)
            gf_tgt_z = BASE.zscore_apply(gf_tgt_df.loc[shared_tgt].to_numpy(dtype=float), gf_mu, gf_sd)
            mapped_sc = gf_tgt_z @ map_a
            mapped_cycle = mapped_sc @ map_b

            self_transfer = np.sum(BASE.row_normalize(mapped_sc) * BASE.row_normalize(sc_tgt_z), axis=1)
            sym_to_pos = {sym: idx for idx, sym in enumerate(shared_tgt)}

            for split_idx, split_regime in enumerate(H125_SPLITS):
                split_mask = split_masks.get(split_regime)
                if split_mask is None:
                    continue

                split_df = sc_edges[target_domain].loc[split_mask].copy()
                split_df["source_u"] = split_df["source"].astype(str).str.upper()
                split_df["target_u"] = split_df["target"].astype(str).str.upper()
                keep = split_df["source_u"].isin(sym_to_pos) & split_df["target_u"].isin(sym_to_pos)
                split_df = split_df.loc[keep].copy()
                if split_df["label"].nunique() < 2 or split_df.shape[0] < 250:
                    continue

                src_sym = split_df["source_u"].to_numpy(dtype=str)
                tgt_sym = split_df["target_u"].to_numpy(dtype=str)
                labels = split_df["label"].to_numpy(dtype=int)
                src_idx = np.array([sym_to_pos[s] for s in src_sym], dtype=int)
                tgt_idx = np.array([sym_to_pos[t] for t in tgt_sym], dtype=int)

                pair_transfer = edge_pair_cosine_scores(src_idx, tgt_idx, mapped_sc, sc_tgt_z)
                transfer_scores = 0.60 * pair_transfer + 0.40 * (self_transfer[src_idx] + self_transfer[tgt_idx]) * 0.5
                baseline_scores = edge_pair_cosine_scores(src_idx, tgt_idx, gf_tgt_z, gf_tgt_z)

                auc_transfer = BASE.safe_auc(labels, transfer_scores)
                auc_baseline = BASE.safe_auc(labels, baseline_scores)
                delta_auc = (
                    float(auc_transfer - auc_baseline)
                    if np.isfinite(auc_transfer) and np.isfinite(auc_baseline)
                    else float("nan")
                )

                edge_strata = build_edge_strata(
                    edge_length=np.abs(transfer_scores),
                    degree_sum=np.abs(baseline_scores),
                    max_len_bins=6,
                    max_deg_bins=4,
                )

                rng = np.random.default_rng(47_210 + domain_idx * 1000 + layer * 100 + split_idx * 10)
                null_anchor = np.empty(H125_NULL_PERM, dtype=float)
                null_random = np.empty(H125_NULL_PERM, dtype=float)
                null_label = np.empty(H125_NULL_PERM, dtype=float)

                for perm_idx in range(H125_NULL_PERM):
                    perm_w = train_w[rng.permutation(train_w.shape[0])]
                    map_a_anchor, _, _, _ = BASE.fit_cycle_consistent_maps(
                        x_sc=train_sc_z,
                        y_gf=train_gf_z,
                        weights=perm_w,
                        n_iters=H125_MAP_ITERS_NULL,
                        lr=0.050,
                        l2=0.05,
                        cycle_weight=0.50,
                    )
                    mapped_anchor = gf_tgt_z @ map_a_anchor
                    tr_anchor = edge_pair_cosine_scores(src_idx, tgt_idx, mapped_anchor, sc_tgt_z)
                    tr_anchor = 0.60 * tr_anchor + 0.40 * (self_transfer[src_idx] + self_transfer[tgt_idx]) * 0.5
                    auc_anchor = BASE.safe_auc(labels, tr_anchor)
                    null_anchor[perm_idx] = (
                        float(auc_anchor - auc_baseline)
                        if np.isfinite(auc_anchor) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H125",
                            "null_kind": "anchor_label_permutation_preserving_size",
                            "domain": target_domain,
                            "seed_tag": H125_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_anchor[perm_idx]),
                        }
                    )

                    perm = rng.permutation(train_sc_z.shape[0])
                    map_a_rand, _, _, _ = BASE.fit_cycle_consistent_maps(
                        x_sc=train_sc_z[perm],
                        y_gf=train_gf_z,
                        weights=train_w,
                        n_iters=H125_MAP_ITERS_NULL,
                        lr=0.050,
                        l2=0.05,
                        cycle_weight=0.50,
                    )
                    mapped_rand = gf_tgt_z @ map_a_rand
                    tr_rand = edge_pair_cosine_scores(src_idx, tgt_idx, mapped_rand, sc_tgt_z)
                    tr_rand = 0.60 * tr_rand + 0.40 * (self_transfer[src_idx] + self_transfer[tgt_idx]) * 0.5
                    auc_rand = BASE.safe_auc(labels, tr_rand)
                    null_random[perm_idx] = (
                        float(auc_rand - auc_baseline)
                        if np.isfinite(auc_rand) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H125",
                            "null_kind": "random_correspondence_baseline",
                            "domain": target_domain,
                            "seed_tag": H125_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_random[perm_idx]),
                        }
                    )

                    y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                    auc_tp = BASE.safe_auc(y_perm, transfer_scores)
                    auc_bp = BASE.safe_auc(y_perm, baseline_scores)
                    null_label[perm_idx] = (
                        float(auc_tp - auc_bp)
                        if np.isfinite(auc_tp) and np.isfinite(auc_bp)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H125",
                            "null_kind": "label_permutation",
                            "domain": target_domain,
                            "seed_tag": H125_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                p_anchor = BASE.empirical_upper_tail_p(delta_auc, null_anchor)
                p_random = BASE.empirical_upper_tail_p(delta_auc, null_random)
                p_label = BASE.empirical_upper_tail_p(delta_auc, null_label)
                p_best = float(np.nanmin(np.asarray([p_anchor, p_random, p_label], dtype=float)))

                all_null = np.concatenate([null_anchor, null_random, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                cycle_err_local = float(np.sqrt(np.mean((mapped_cycle - gf_tgt_z) ** 2)))

                by_row.append(
                    {
                        "hypothesis_id": "H125",
                        "domain": target_domain,
                        "seed_tag": H125_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_transfer": float(auc_transfer),
                        "auc_baseline": float(auc_baseline),
                        "delta_vs_h70": float(delta_auc),
                        "transfer_delta_auc_vs_h70": float(delta_auc),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95": float(delta_auc - q95),
                        "cycle_consistency_error": float(cycle_err_local),
                        "alignment_rmse_train": float(align_rmse),
                        "cycle_rmse_train": float(cycle_rmse),
                        "p_anchor_upper": float(p_anchor),
                        "p_random_upper": float(p_random),
                        "p_label_upper": float(p_label),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(by_row)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h125_anchor_cycle_alignment_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h125_anchor_cycle_alignment_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    split_summary_df = domain_split_summary(by_row_df)
    split_summary_path = ITER_DIR / "h125_anchor_cycle_alignment_domain_split_summary.csv"
    split_summary_df.to_csv(split_summary_path, index=False)

    domain_summary_df = domain_only_summary(by_row_df)
    domain_summary_path = ITER_DIR / "h125_anchor_cycle_alignment_domain_summary.csv"
    domain_summary_df.to_csv(domain_summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_vs_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_count": int((domain_summary_df["mean_delta_vs_h70"] > 0.0).sum()) if not domain_summary_df.empty else 0,
        "positive_null_gap_domain_count": int((domain_summary_df["mean_null_gap_q95"] > 0.0).sum()) if not domain_summary_df.empty else 0,
        "immune_mean_null_gap": float(domain_summary_df.loc[domain_summary_df["domain"] == "immune", "mean_null_gap_q95"].mean())
        if not domain_summary_df.empty
        else float("nan"),
        "artifact_paths": {
            "by_domain_split_layer": str(by_row_path),
            "domain_split_summary": str(split_summary_path),
            "domain_summary": str(domain_summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h126_geodesic_torsion_asymmetry(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_idx, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H126_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_idx, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H126_GENE_CAP))
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

            rng = np.random.default_rng(47_300 + domain_idx * 1000 + split_idx * 100)
            sample_idx = stratified_index_sample(labels_all, max_n=H126_EDGE_SAMPLE, rng=rng)
            if sample_idx.size < 120:
                continue

            source_local = source_local_all[sample_idx]
            target_local = target_local_all[sample_idx]
            labels = labels_all[sample_idx]

            for layer in H126_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=47_301 + domain_idx * 1000 + split_idx * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H126_NEIGHBORS)
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
                    n_neighbors=H126_NEIGHBORS,
                )
                dist_dir, pred_dir = shortest_path(
                    directed_graph,
                    directed=True,
                    unweighted=False,
                    return_predecessors=True,
                )
                torsion_feat = torsion_directional_features(
                    source_local=source_local,
                    target_local=target_local,
                    dist_mat=dist_dir,
                    predecessors=pred_dir,
                    support_dir=support_dir,
                    points_pca=points_pca,
                )

                model_feat = np.column_stack([h70, torsion_feat])
                auc_h70 = BASE.safe_auc(labels, h70)
                auc_model = cv_auc_logit(
                    model_feat,
                    labels,
                    random_state=47_302 + domain_idx * 1000 + split_idx * 100 + layer,
                    n_splits=H126_CV_SPLITS,
                )
                delta = float(auc_model - auc_h70) if np.isfinite(auc_model) and np.isfinite(auc_h70) else float("nan")

                edge_len = geodesic_w[source_local, target_local]
                len_bins = quantile_bins(edge_len, n_bins=6)
                deg_sum = edge_degree_sum(points_pca, H126_NEIGHBORS, source_local, target_local)
                edge_strata = build_edge_strata(edge_len, deg_sum, max_len_bins=6, max_deg_bins=4)

                null_reverse = np.empty(H126_NULL_PERM, dtype=float)
                null_swap = np.empty(H126_NULL_PERM, dtype=float)
                null_label = np.empty(H126_NULL_PERM, dtype=float)

                for perm_idx in range(H126_NULL_PERM):
                    feat_rev = path_reversal_within_bins(torsion_feat, strata=len_bins, rng=rng)
                    auc_rev = cv_auc_logit(
                        np.column_stack([h70, feat_rev]),
                        labels,
                        random_state=47_303 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H126_CV_SPLITS,
                    )
                    null_reverse[perm_idx] = (
                        float(auc_rev - auc_h70) if np.isfinite(auc_rev) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H126",
                            "null_kind": "path_reversal_within_length_bins",
                            "domain": domain,
                            "seed_tag": H126_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_reverse[perm_idx]),
                        }
                    )

                    feat_swap = swapped_torsion_features(torsion_feat)
                    feat_swap = permute_rows_within_strata(feat_swap, strata=len_bins, rng=rng)
                    auc_swap = cv_auc_logit(
                        np.column_stack([h70, feat_swap]),
                        labels,
                        random_state=47_304 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H126_CV_SPLITS,
                    )
                    null_swap[perm_idx] = (
                        float(auc_swap - auc_h70) if np.isfinite(auc_swap) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H126",
                            "null_kind": "endpoint_swap_within_distance_bins",
                            "domain": domain,
                            "seed_tag": H126_SEED,
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
                        random_state=47_305 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H126_CV_SPLITS,
                    )
                    auc_h70_lp = BASE.safe_auc(y_perm, h70)
                    null_label[perm_idx] = (
                        float(auc_lp - auc_h70_lp)
                        if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H126",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H126_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_reverse, null_swap, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_rev = BASE.empirical_upper_tail_p(delta, null_reverse)
                p_swap = BASE.empirical_upper_tail_p(delta, null_swap)
                p_label = BASE.empirical_upper_tail_p(delta, null_label)
                p_best = float(np.nanmin(np.asarray([p_rev, p_swap, p_label], dtype=float)))

                rows.append(
                    {
                        "hypothesis_id": "H126",
                        "domain": domain,
                        "seed_tag": H126_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70": float(auc_h70),
                        "auc_torsion_model": float(auc_model),
                        "delta_vs_h70": float(delta),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95": float(delta - q95),
                        "p_path_reversal_upper": float(p_rev),
                        "p_endpoint_swap_upper": float(p_swap),
                        "p_label_upper": float(p_label),
                        "p_best_upper": float(p_best),
                        "mean_turning_gap": float(np.mean(torsion_feat[:, 5])),
                        "mean_torsion_abs_gap": float(np.mean(torsion_feat[:, 11])),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h126_geodesic_torsion_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h126_geodesic_torsion_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_df = domain_split_summary(by_row_df)
    summary_path = ITER_DIR / "h126_geodesic_torsion_domain_summary.csv"
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


def main() -> None:
    ensure_required_inputs()

    dorothea_map = BASE.load_dorothea_score_map()
    omnipath_pairs = BASE.load_omnipath_pairs()
    gene2go_upper = BASE.load_gene2go_upper()
    string_map = BASE.load_string_scores_from_cache(BASE.STRING_CACHE_PATH)
    trrust_sign_map, trrust_tf_out_degree = load_trrust_signed_map()

    h124 = run_h124_signed_motif_string_hardening(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
        trrust_sign_map=trrust_sign_map,
        trrust_tf_out_degree=trrust_tf_out_degree,
    )
    h125 = run_h125_anchor_cycle_alignment(
        gene2go_upper=gene2go_upper,
        trrust_sign_map=trrust_sign_map,
    )
    h126 = run_h126_geodesic_torsion_asymmetry(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    summary = {
        "iteration": "iter_0047",
        "h124": h124,
        "h125": h125,
        "h126": h126,
    }
    summary_path = ITER_DIR / "iter0047_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
