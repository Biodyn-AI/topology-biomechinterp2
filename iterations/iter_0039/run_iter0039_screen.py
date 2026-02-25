from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ITER_DIR = Path("iterations/iter_0039")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base")

# H100 / N493: relative persistence contrast against matched background complexes.
H100_SEED = "seed42_main"
H100_LAYERS = [7, 11]
H100_GENE_CAP = 170
H100_NEIGHBORS = 12
H100_TRIANGLE_K = [8, 12, 16]
H100_EDGE_SAMPLE = 280
H100_CV_SPLITS = 4
H100_L1_C = 0.22
H100_NULL_PERM = 24
H100_DIST_QUANTILES = [0.20, 0.35, 0.50, 0.65]
H100_MARGIN_QUANTILES = [0.55, 0.70, 0.85]

# H101 / N497: persistence derivative spectrum over filtration quantiles.
H101_SEED = "seed42_main"
H101_LAYERS = [0, 3, 7, 11]
H101_GENE_CAP = 170
H101_NEIGHBORS = 12
H101_TRIANGLE_K = [8, 12, 16]
H101_EDGE_SAMPLE = 260
H101_CV_SPLITS = 4
H101_L1_C = 0.22
H101_NULL_PERM = 20
H101_DIST_QUANTILES = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
H101_MARGIN_QUANTILES = [0.45, 0.60, 0.75]

# H102 / N501: cross-model OT + monotone depth warp (fast-fail pilot).
H102_SEED = "seed42_main"
H102_LAYERS = [7, 11]
H102_GENE_CAP = 220
H102_MODULE_MIN = 8
H102_MODULE_MAX = 42
H102_MAX_MODULES = 64
H102_NULL_PERM = 32
H102_WARP_CANDIDATES = [(0.80, 0.20), (0.60, 0.40), (0.50, 0.50), (0.40, 0.60), (0.20, 0.80)]


def ensure_required_inputs() -> None:
    required_paths = [
        BASE.DOROTHEA_PATH,
        BASE.GENE2GO_PATH,
        BASE.OMNIPATH_INTERACTIONS_PATH,
        BASE.STRING_CACHE_PATH,
    ]
    for run_map in BASE.SCGPT_RUNS_BY_DOMAIN.values():
        run_dir = run_map[H100_SEED]
        required_paths.append(run_dir / "cycle1_edge_dataset.tsv")
        required_paths.append(run_dir / "layer_gene_embeddings.npy")
    for gf_path in BASE.GENEFORMER_EDGE_BY_DOMAIN.values():
        required_paths.append(gf_path)

    missing = [str(p) for p in required_paths if not Path(p).exists()]
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


def cross_validated_auc(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    penalty: str,
    c_value: float = 1.0,
    n_splits: int = 4,
) -> float:
    x = np.asarray(features, dtype=float)
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

        if penalty == "none":
            model = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=1200,
                random_state=random_state + fold_idx,
            )
        elif penalty == "l1":
            model = LogisticRegression(
                penalty="l1",
                C=float(c_value),
                solver="liblinear",
                max_iter=1200,
                random_state=random_state + fold_idx,
            )
        else:
            raise ValueError(f"Unsupported penalty={penalty}")

        model.fit(x_tr, y[tr])
        probs[te] = model.predict_proba(x_te)[:, 1]

    return BASE.safe_auc(y, probs)


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


def confidence_weighted_geodesic(geodesic: np.ndarray, support_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sym_support = 0.5 * (support_dir + support_dir.T)
    sym_support = np.clip(sym_support, 0.0, 1.0)

    weighted = geodesic / (0.35 + sym_support)
    weighted = np.asarray(weighted, dtype=float)
    np.fill_diagonal(weighted, 0.0)
    return weighted, sym_support


def zscore_cols(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mu = np.mean(arr, axis=0, keepdims=True)
    sd = np.std(arr, axis=0, keepdims=True)
    sd = np.clip(sd, 1e-8, None)
    return (arr - mu) / sd


def make_non_self_targets(
    source_local: np.ndarray,
    target_local: np.ndarray,
    n_nodes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.asarray(target_local, dtype=int).copy()
    src = np.asarray(source_local, dtype=int)
    bad = np.where(out == src)[0]
    for idx in bad:
        pool = np.delete(np.arange(n_nodes, dtype=int), src[idx])
        out[idx] = int(rng.choice(pool))
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


def relative_filtration_bundle(
    dist_matrix: np.ndarray,
    margin_matrix: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    dist_quantiles: list[float],
    margin_quantiles: list[float],
) -> dict[str, np.ndarray]:
    return BASE.two_axis_filtration_connectivity(
        dist_matrix=dist_matrix,
        margin_matrix=margin_matrix,
        source_local=source_local,
        target_local=target_local,
        dist_quantiles=dist_quantiles,
        margin_quantiles=margin_quantiles,
    )


def assemble_h100_features(
    h70_base: np.ndarray,
    h70_weighted: np.ndarray,
    edge_margin: np.ndarray,
    edge_confidence: np.ndarray,
    anchor_bundle: dict[str, np.ndarray],
    background_bundle: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    weighted_gain = h70_weighted - h70_base
    x_h93 = np.column_stack(
        [
            h70_base,
            h70_weighted,
            weighted_gain,
            edge_margin,
            edge_confidence,
        ]
    )

    rel_one = anchor_bundle["conn_one_frac"] - background_bundle["conn_one_frac"]
    rel_two = anchor_bundle["conn_two_frac"] - background_bundle["conn_two_frac"]
    rel_gain = anchor_bundle["conn_gain"] - background_bundle["conn_gain"]

    x_rel = np.column_stack(
        [
            x_h93,
            anchor_bundle["conn_one_frac"],
            anchor_bundle["conn_two_frac"],
            anchor_bundle["conn_gain"],
            background_bundle["conn_one_frac"],
            background_bundle["conn_two_frac"],
            background_bundle["conn_gain"],
            rel_one,
            rel_two,
            rel_gain,
        ]
    )
    return x_h93, x_rel


def filtration_connectivity_trajectories(
    dist_matrix: np.ndarray,
    margin_matrix: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    dist_quantiles: list[float],
    margin_quantiles: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = dist_matrix.shape[0]
    upper_i, upper_j = np.triu_indices(n_nodes, k=1)
    dist_vals = dist_matrix[upper_i, upper_j]
    margin_vals = margin_matrix[upper_i, upper_j]

    dist_thresholds = [float(np.quantile(dist_vals, q)) for q in dist_quantiles]
    margin_thresholds = [float(np.quantile(margin_vals, q)) for q in margin_quantiles]

    traj_one = np.zeros((source_local.size, len(dist_thresholds)), dtype=float)
    traj_two = np.zeros((source_local.size, len(dist_thresholds)), dtype=float)

    for d_idx, d_thr in enumerate(dist_thresholds):
        keep_dist = dist_vals <= d_thr
        labels_one, _, _ = BASE.component_labels_from_upper_mask(
            n_nodes=n_nodes,
            upper_i=upper_i,
            upper_j=upper_j,
            keep_mask=keep_dist,
        )
        traj_one[:, d_idx] = (labels_one[source_local] == labels_one[target_local]).astype(float)

        margin_conn = np.zeros((len(margin_thresholds), source_local.size), dtype=float)
        for m_idx, m_thr in enumerate(margin_thresholds):
            keep = keep_dist & (margin_vals >= m_thr)
            labels_two, _, _ = BASE.component_labels_from_upper_mask(
                n_nodes=n_nodes,
                upper_i=upper_i,
                upper_j=upper_j,
                keep_mask=keep,
            )
            margin_conn[m_idx] = (labels_two[source_local] == labels_two[target_local]).astype(float)
        traj_two[:, d_idx] = margin_conn.mean(axis=0)

    return traj_one, traj_two


def first_true_index(mask: np.ndarray) -> np.ndarray:
    out = np.argmax(mask, axis=1).astype(float)
    none_true = ~mask.any(axis=1)
    out[none_true] = float(mask.shape[1])
    return out


def derivative_spectrum_features(trajectory: np.ndarray) -> np.ndarray:
    traj = np.asarray(trajectory, dtype=float)
    n_edges, n_steps = traj.shape

    if n_steps < 2:
        return np.zeros((n_edges, 8), dtype=float)

    d1 = np.diff(traj, axis=1)
    if d1.shape[1] >= 2:
        d2 = np.diff(d1, axis=1)
    else:
        d2 = np.zeros((n_edges, 1), dtype=float)

    auc = np.mean(traj, axis=1)
    end_minus_start = traj[:, -1] - traj[:, 0]
    onset = first_true_index(traj > 0.5) / max(1.0, float(n_steps - 1))

    abs_d1 = np.abs(d1)
    mass = np.sum(abs_d1, axis=1)
    prob = abs_d1 / np.clip(mass[:, None], 1e-8, None)
    entropy = -np.sum(prob * np.log(np.clip(prob, 1e-8, None)), axis=1) / np.log(max(2, d1.shape[1]))

    d1_mean = np.mean(d1, axis=1)
    d1_abs_mean = np.mean(abs_d1, axis=1)

    peak_idx = np.argmax(abs_d1, axis=1).astype(float) / max(1.0, float(abs_d1.shape[1] - 1))

    d2_abs_mean = np.mean(np.abs(d2), axis=1)

    return np.column_stack(
        [
            auc,
            onset,
            end_minus_start,
            entropy,
            d1_mean,
            d1_abs_mean,
            peak_idx,
            d2_abs_mean,
        ]
    )


def build_h101_feature_matrix(traj_one: np.ndarray, traj_two: np.ndarray) -> np.ndarray:
    f1 = derivative_spectrum_features(traj_one)
    f2 = derivative_spectrum_features(traj_two)
    diff = f2 - f1
    # Keep only stable summary columns from the difference branch to avoid overparameterization.
    diff_sel = diff[:, [0, 1, 2, 3, 5, 7]]
    return np.column_stack([f1, f2, diff_sel])


def permute_trajectory_order(trajectory: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(trajectory.shape[1])
    return trajectory[:, order]


def randomize_derivative_signs(feature_matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(feature_matrix, dtype=float).copy()
    # Columns tied to signed derivative behavior in the concatenated feature matrix.
    derivative_cols = [4, 12, 20]
    for col in derivative_cols:
        signs = rng.choice(np.array([-1.0, 1.0]), size=x.shape[0], replace=True)
        x[:, col] = x[:, col] * signs
    return x


def spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    a = pd.Series(np.asarray(x, dtype=float)).rank(method="average")
    b = pd.Series(np.asarray(y, dtype=float)).rank(method="average")
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.corr(b, method="pearson"))


def select_go_modules(symbols: list[str], gene2go_upper: dict[str, set[str]]) -> list[tuple[str, list[str]]]:
    term_to_genes: dict[str, list[str]] = {}
    symbol_set = set(symbols)
    for sym in symbols:
        for term in gene2go_upper.get(sym.upper(), set()):
            if term not in term_to_genes:
                term_to_genes[term] = []
            term_to_genes[term].append(sym)

    modules: list[tuple[str, list[str]]] = []
    for term, genes in term_to_genes.items():
        uniq = sorted(set(g for g in genes if g in symbol_set))
        if H102_MODULE_MIN <= len(uniq) <= H102_MODULE_MAX:
            modules.append((term, uniq))

    modules.sort(key=lambda item: (-len(item[1]), item[0]))
    return modules[:H102_MAX_MODULES]


def module_role_vectors(
    modules: list[tuple[str, list[str]]],
    role_df: pd.DataFrame,
) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    vectors: list[np.ndarray] = []
    for term, genes in modules:
        genes_use = [g for g in genes if g in role_df.index]
        if len(genes_use) < H102_MODULE_MIN:
            continue
        vec = role_df.loc[genes_use].to_numpy(dtype=float).mean(axis=0)
        names.append(term)
        vectors.append(vec)

    if not vectors:
        return [], np.zeros((0, role_df.shape[1]), dtype=float)
    return names, np.vstack(vectors)


def role_transition_matrix(role_vectors: np.ndarray) -> np.ndarray:
    x = zscore_cols(role_vectors)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.clip(norms, 1e-12, None)
    sim = x @ x.T
    np.fill_diagonal(sim, 0.0)
    rank = BASE.symmetric_global_rank_matrix(sim)
    row_sum = rank.sum(axis=1, keepdims=True)
    return rank / np.clip(row_sum, 1e-8, None)


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    upper_i, upper_j = np.triu_indices(matrix.shape[0], k=1)
    return matrix[upper_i, upper_j]


def top_edge_jaccard(matrix_a: np.ndarray, matrix_b: np.ndarray, frac: float = 0.20) -> float:
    a = upper_triangle_values(matrix_a)
    b = upper_triangle_values(matrix_b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    k = max(3, int(round(frac * a.size)))
    idx_a = set(np.argsort(-a)[:k].tolist())
    idx_b = set(np.argsort(-b)[:k].tolist())
    union = idx_a | idx_b
    if not union:
        return 0.0
    return float(len(idx_a & idx_b) / len(union))


def build_module_size_preserving_permutation(
    symbols: list[str],
    modules: list[tuple[str, list[str]]],
    rng: np.random.Generator,
) -> list[tuple[str, list[str]]]:
    # Module sizes are preserved term-wise while allowing overlap between modules,
    # matching the overlapping structure of GO modules in the observed data.
    symbol_arr = np.asarray(symbols, dtype=object)
    out: list[tuple[str, list[str]]] = []
    for term, genes in modules:
        size = int(len(genes))
        if size <= 0:
            out.append((term, []))
            continue
        size_eff = min(size, symbol_arr.size)
        chosen = rng.choice(symbol_arr, size=size_eff, replace=False)
        out.append((term, sorted(chosen.tolist())))
    return out


def align_module_vectors(
    modules: list[tuple[str, list[str]]],
    sc_layer_roles: dict[int, pd.DataFrame],
    gf_roles: pd.DataFrame,
) -> tuple[list[str], dict[int, np.ndarray], np.ndarray]:
    name_map: dict[int, list[str]] = {}
    vec_map: dict[int, np.ndarray] = {}

    for layer, role_df in sc_layer_roles.items():
        names, vecs = module_role_vectors(modules=modules, role_df=role_df)
        name_map[layer] = names
        vec_map[layer] = vecs

    names_gf, vecs_gf = module_role_vectors(modules=modules, role_df=gf_roles)
    if len(names_gf) == 0:
        return [], {}, np.zeros((0, gf_roles.shape[1]), dtype=float)

    common = set(names_gf)
    for layer in sc_layer_roles:
        common &= set(name_map[layer])
    common_sorted = sorted(common)

    if len(common_sorted) < H102_MODULE_MIN:
        return [], {}, np.zeros((0, gf_roles.shape[1]), dtype=float)

    gf_idx = {name: i for i, name in enumerate(names_gf)}
    gf_out = np.vstack([vecs_gf[gf_idx[name]] for name in common_sorted])

    sc_out: dict[int, np.ndarray] = {}
    for layer in sc_layer_roles:
        idx = {name: i for i, name in enumerate(name_map[layer])}
        sc_out[layer] = np.vstack([vec_map[layer][idx[name]] for name in common_sorted])

    return common_sorted, sc_out, gf_out


def depth_warp_ot_alignment(
    sc_layer_vectors: dict[int, np.ndarray],
    gf_vectors: np.ndarray,
    warp_candidates: list[tuple[float, float]],
) -> dict[str, float]:
    sc7 = sc_layer_vectors[H102_LAYERS[0]]
    sc11 = sc_layer_vectors[H102_LAYERS[1]]
    gf = zscore_cols(gf_vectors)

    best: dict[str, float] | None = None

    for w7, w11 in warp_candidates:
        sc_warp = zscore_cols(w7 * sc7 + w11 * sc11)
        cost = cdist(sc_warp, gf, metric="euclidean")
        row_idx, col_idx = linear_sum_assignment(cost)

        gf_aligned = np.zeros_like(sc_warp)
        gf_aligned[row_idx] = gf[col_idx]

        sc_t = role_transition_matrix(sc_warp)
        gf_t = role_transition_matrix(gf_aligned)

        concordance = spearman_rank_corr(upper_triangle_values(sc_t), upper_triangle_values(gf_t))
        transport_cost = float(np.mean(cost[row_idx, col_idx]))
        top_jacc = top_edge_jaccard(sc_t, gf_t, frac=0.20)
        rmse = float(np.sqrt(np.mean((sc_warp - gf_aligned) ** 2)))

        candidate = {
            "concordance": float(concordance),
            "transport_cost": transport_cost,
            "top_jaccard": float(top_jacc),
            "alignment_rmse": rmse,
            "warp_w7": float(w7),
            "warp_w11": float(w11),
        }

        if best is None or candidate["transport_cost"] < best["transport_cost"]:
            best = candidate

    if best is None:
        return {
            "concordance": float("nan"),
            "transport_cost": float("nan"),
            "top_jaccard": float("nan"),
            "alignment_rmse": float("nan"),
            "warp_w7": float("nan"),
            "warp_w11": float("nan"),
        }
    return best


def depth_warp_ot_alignment_fixed(
    sc_layer_vectors: dict[int, np.ndarray],
    gf_vectors: np.ndarray,
    w7: float,
    w11: float,
) -> float:
    sc7 = sc_layer_vectors[H102_LAYERS[0]]
    sc11 = sc_layer_vectors[H102_LAYERS[1]]
    sc_warp = zscore_cols(w7 * sc7 + w11 * sc11)
    gf = zscore_cols(gf_vectors)
    cost = cdist(sc_warp, gf, metric="euclidean")
    row_idx, col_idx = linear_sum_assignment(cost)
    gf_aligned = np.zeros_like(sc_warp)
    gf_aligned[row_idx] = gf[col_idx]

    sc_t = role_transition_matrix(sc_warp)
    gf_t = role_transition_matrix(gf_aligned)
    return spearman_rank_corr(upper_triangle_values(sc_t), upper_triangle_values(gf_t))


def random_orthogonal_matrix(dim: int, rng: np.random.Generator) -> np.ndarray:
    mat = rng.normal(size=(dim, dim))
    q, _ = np.linalg.qr(mat)
    return q


def run_h100_relative_persistence_contrast(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H100_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H100_GENE_CAP))
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

            for layer in H100_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                rng = np.random.default_rng(39_100 + domain_index * 1000 + split_index * 100 + layer)
                sample_idx = stratified_index_sample(labels_all, max_n=H100_EDGE_SAMPLE, rng=rng)
                if sample_idx.size < 120:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=22,
                    random_state=39_101 + domain_index * 1000 + split_index * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H100_NEIGHBORS)
                geodesic_w, support_sym = confidence_weighted_geodesic(geodesic, support_dir)

                _, h70_base, _ = compute_h70_scores(
                    geodesic=geodesic,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H100_TRIANGLE_K,
                )
                _, h70_weighted, _ = compute_h70_scores(
                    geodesic=geodesic_w,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H100_TRIANGLE_K,
                )

                edge_margin = support_dir[source_local, target_local] - support_dir[target_local, source_local]
                edge_confidence = support_sym[source_local, target_local]

                edge_length = geodesic_w[source_local, target_local]
                edge_deg_sum = edge_degree_sum(
                    points=points_pca,
                    n_neighbors=H100_NEIGHBORS,
                    source_local=source_local,
                    target_local=target_local,
                )
                strata = build_edge_strata(
                    edge_length=edge_length,
                    degree_sum=edge_deg_sum,
                    max_len_bins=6,
                    max_deg_bins=4,
                )

                margin_matrix = np.abs(support_dir - support_dir.T)
                anchor_bundle = relative_filtration_bundle(
                    dist_matrix=geodesic_w,
                    margin_matrix=margin_matrix,
                    source_local=source_local,
                    target_local=target_local,
                    dist_quantiles=H100_DIST_QUANTILES,
                    margin_quantiles=H100_MARGIN_QUANTILES,
                )

                bg_target = BASE.shuffle_within_bins(target_local, strata, rng).astype(int)
                bg_target = make_non_self_targets(
                    source_local=source_local,
                    target_local=bg_target,
                    n_nodes=geodesic_w.shape[0],
                    rng=rng,
                )
                bg_bundle = relative_filtration_bundle(
                    dist_matrix=geodesic_w,
                    margin_matrix=margin_matrix,
                    source_local=source_local,
                    target_local=bg_target,
                    dist_quantiles=H100_DIST_QUANTILES,
                    margin_quantiles=H100_MARGIN_QUANTILES,
                )

                x_h93, x_rel = assemble_h100_features(
                    h70_base=h70_base,
                    h70_weighted=h70_weighted,
                    edge_margin=edge_margin,
                    edge_confidence=edge_confidence,
                    anchor_bundle=anchor_bundle,
                    background_bundle=bg_bundle,
                )

                auc_h93 = cross_validated_auc(
                    x_h93,
                    labels,
                    random_state=39_102 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H100_L1_C,
                    n_splits=H100_CV_SPLITS,
                )
                auc_rel = cross_validated_auc(
                    x_rel,
                    labels,
                    random_state=39_103 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H100_L1_C,
                    n_splits=H100_CV_SPLITS,
                )
                delta_auc = float(auc_rel - auc_h93) if np.isfinite(auc_rel) and np.isfinite(auc_h93) else float("nan")

                rel_two = anchor_bundle["conn_two_frac"] - bg_bundle["conn_two_frac"]
                rel_two_gap = float(np.mean(rel_two[labels == 1]) - np.mean(rel_two[labels == 0]))

                null_anchor = np.empty(H100_NULL_PERM, dtype=float)
                null_bg = np.empty(H100_NULL_PERM, dtype=float)
                null_label = np.empty(H100_NULL_PERM, dtype=float)

                for perm_idx in range(H100_NULL_PERM):
                    # Null 1: anchor permutation preserving edge-length and degree strata.
                    perm_source = BASE.shuffle_within_bins(source_local, strata, rng).astype(int)
                    perm_target = BASE.shuffle_within_bins(target_local, strata, rng).astype(int)
                    perm_target = make_non_self_targets(
                        source_local=perm_source,
                        target_local=perm_target,
                        n_nodes=geodesic_w.shape[0],
                        rng=rng,
                    )
                    perm_anchor = relative_filtration_bundle(
                        dist_matrix=geodesic_w,
                        margin_matrix=margin_matrix,
                        source_local=perm_source,
                        target_local=perm_target,
                        dist_quantiles=H100_DIST_QUANTILES,
                        margin_quantiles=H100_MARGIN_QUANTILES,
                    )
                    x_h93_perm, x_rel_perm = assemble_h100_features(
                        h70_base=h70_base,
                        h70_weighted=h70_weighted,
                        edge_margin=edge_margin,
                        edge_confidence=edge_confidence,
                        anchor_bundle=perm_anchor,
                        background_bundle=bg_bundle,
                    )
                    auc_h93_perm = cross_validated_auc(
                        x_h93_perm,
                        labels,
                        random_state=39_104 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H100_L1_C,
                        n_splits=H100_CV_SPLITS,
                    )
                    auc_rel_perm = cross_validated_auc(
                        x_rel_perm,
                        labels,
                        random_state=39_105 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H100_L1_C,
                        n_splits=H100_CV_SPLITS,
                    )
                    null_anchor[perm_idx] = (
                        float(auc_rel_perm - auc_h93_perm)
                        if np.isfinite(auc_h93_perm) and np.isfinite(auc_rel_perm)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H100",
                            "null_kind": "anchor_permutation_within_length_degree_bins",
                            "domain": domain,
                            "seed_tag": H100_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_anchor[perm_idx]),
                        }
                    )

                    # Null 2: replace matched background with unconstrained random background pairs.
                    bg_rand = rng.integers(0, geodesic_w.shape[0], size=source_local.size, endpoint=False)
                    bg_rand = make_non_self_targets(
                        source_local=source_local,
                        target_local=bg_rand,
                        n_nodes=geodesic_w.shape[0],
                        rng=rng,
                    )
                    rand_bundle = relative_filtration_bundle(
                        dist_matrix=geodesic_w,
                        margin_matrix=margin_matrix,
                        source_local=source_local,
                        target_local=bg_rand,
                        dist_quantiles=H100_DIST_QUANTILES,
                        margin_quantiles=H100_MARGIN_QUANTILES,
                    )
                    x_h93_rand, x_rel_rand = assemble_h100_features(
                        h70_base=h70_base,
                        h70_weighted=h70_weighted,
                        edge_margin=edge_margin,
                        edge_confidence=edge_confidence,
                        anchor_bundle=anchor_bundle,
                        background_bundle=rand_bundle,
                    )
                    auc_h93_rand = cross_validated_auc(
                        x_h93_rand,
                        labels,
                        random_state=39_106 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H100_L1_C,
                        n_splits=H100_CV_SPLITS,
                    )
                    auc_rel_rand = cross_validated_auc(
                        x_rel_rand,
                        labels,
                        random_state=39_107 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H100_L1_C,
                        n_splits=H100_CV_SPLITS,
                    )
                    null_bg[perm_idx] = (
                        float(auc_rel_rand - auc_h93_rand)
                        if np.isfinite(auc_h93_rand) and np.isfinite(auc_rel_rand)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H100",
                            "null_kind": "random_background_complex",
                            "domain": domain,
                            "seed_tag": H100_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_bg[perm_idx]),
                        }
                    )

                    # Null 3: stratified label permutation.
                    labels_perm = BASE.stratified_shuffle(labels, strata, rng).astype(int)
                    auc_lp_h93 = cross_validated_auc(
                        x_h93,
                        labels_perm,
                        random_state=39_108 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H100_L1_C,
                        n_splits=H100_CV_SPLITS,
                    )
                    auc_lp_rel = cross_validated_auc(
                        x_rel,
                        labels_perm,
                        random_state=39_109 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H100_L1_C,
                        n_splits=H100_CV_SPLITS,
                    )
                    null_label[perm_idx] = (
                        float(auc_lp_rel - auc_lp_h93)
                        if np.isfinite(auc_lp_h93) and np.isfinite(auc_lp_rel)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H100",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H100_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_anchor, null_bg, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_anchor = BASE.empirical_upper_tail_p(delta_auc, null_anchor)
                p_bg = BASE.empirical_upper_tail_p(delta_auc, null_bg)
                p_lab = BASE.empirical_upper_tail_p(delta_auc, null_label)
                p_best = np.nanmin(np.array([p_anchor, p_bg, p_lab], dtype=float))

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": H100_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h93_backbone": float(auc_h93),
                        "auc_relative_persistence": float(auc_rel),
                        "delta_auc_relative_ph_minus_h93": float(delta_auc),
                        "relative_conn_two_pos_minus_neg": float(rel_two_gap),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95_delta_auc": float(delta_auc - q95),
                        "p_anchor_permutation_upper": float(p_anchor),
                        "p_random_background_upper": float(p_bg),
                        "p_label_shuffle_upper": float(p_lab),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h100_relative_persistence_contrast_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h100_relative_persistence_contrast_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_relative_ph_minus_h93": float(group["delta_auc_relative_ph_minus_h93"].mean()),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "mean_relative_conn_two_pos_minus_neg": float(group["relative_conn_two_pos_minus_neg"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_relative_ph_minus_h93"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h100_relative_persistence_contrast_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_relative_ph_minus_h93"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_relative_ph_minus_h93"] > 0.0).sum())
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


def run_h101_persistence_derivative_spectrum(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H101_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H101_GENE_CAP))
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

            for layer in H101_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                rng = np.random.default_rng(39_200 + domain_index * 1000 + split_index * 100 + layer)
                sample_idx = stratified_index_sample(labels_all, max_n=H101_EDGE_SAMPLE, rng=rng)
                if sample_idx.size < 120:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=22,
                    random_state=39_201 + domain_index * 1000 + split_index * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H101_NEIGHBORS)
                geodesic_w, _ = confidence_weighted_geodesic(geodesic, support_dir)
                margin_matrix = np.abs(support_dir - support_dir.T)

                _, h70_base, _ = compute_h70_scores(
                    geodesic=geodesic,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H101_TRIANGLE_K,
                )

                traj_one, traj_two = filtration_connectivity_trajectories(
                    dist_matrix=geodesic_w,
                    margin_matrix=margin_matrix,
                    source_local=source_local,
                    target_local=target_local,
                    dist_quantiles=H101_DIST_QUANTILES,
                    margin_quantiles=H101_MARGIN_QUANTILES,
                )
                traj_features = build_h101_feature_matrix(traj_one=traj_one, traj_two=traj_two)

                x_base = h70_base[:, None]
                x_aug = np.column_stack([h70_base, traj_features])

                auc_base = cross_validated_auc(
                    x_base,
                    labels,
                    random_state=39_202 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="none",
                    n_splits=H101_CV_SPLITS,
                )
                auc_aug = cross_validated_auc(
                    x_aug,
                    labels,
                    random_state=39_203 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H101_L1_C,
                    n_splits=H101_CV_SPLITS,
                )
                delta_auc = float(auc_aug - auc_base) if np.isfinite(auc_aug) and np.isfinite(auc_base) else float("nan")

                feature_gap = float(np.mean(traj_features[labels == 1, 8]) - np.mean(traj_features[labels == 0, 8]))

                edge_length = geodesic_w[source_local, target_local]
                degree_sum = edge_degree_sum(
                    points=points_pca,
                    n_neighbors=H101_NEIGHBORS,
                    source_local=source_local,
                    target_local=target_local,
                )
                strata = build_edge_strata(edge_length=edge_length, degree_sum=degree_sum, max_len_bins=6, max_deg_bins=4)

                null_order = np.empty(H101_NULL_PERM, dtype=float)
                null_sign = np.empty(H101_NULL_PERM, dtype=float)
                null_label = np.empty(H101_NULL_PERM, dtype=float)

                for perm_idx in range(H101_NULL_PERM):
                    # Null 1: quantile-order permutation of filtration trajectories.
                    traj_one_perm = permute_trajectory_order(traj_one, rng)
                    traj_two_perm = permute_trajectory_order(traj_two, rng)
                    feat_order = build_h101_feature_matrix(traj_one=traj_one_perm, traj_two=traj_two_perm)
                    x_order = np.column_stack([h70_base, feat_order])
                    auc_order = cross_validated_auc(
                        x_order,
                        labels,
                        random_state=39_204 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H101_L1_C,
                        n_splits=H101_CV_SPLITS,
                    )
                    null_order[perm_idx] = (
                        float(auc_order - auc_base) if np.isfinite(auc_order) and np.isfinite(auc_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H101",
                            "null_kind": "quantile_order_permutation",
                            "domain": domain,
                            "seed_tag": H101_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_order[perm_idx]),
                        }
                    )

                    # Null 2: derivative-sign randomization on derived spectrum features.
                    feat_sign = randomize_derivative_signs(traj_features, rng)
                    x_sign = np.column_stack([h70_base, feat_sign])
                    auc_sign = cross_validated_auc(
                        x_sign,
                        labels,
                        random_state=39_205 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H101_L1_C,
                        n_splits=H101_CV_SPLITS,
                    )
                    null_sign[perm_idx] = (
                        float(auc_sign - auc_base) if np.isfinite(auc_sign) and np.isfinite(auc_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H101",
                            "null_kind": "derivative_sign_randomization",
                            "domain": domain,
                            "seed_tag": H101_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_sign[perm_idx]),
                        }
                    )

                    # Null 3: stratified label permutation.
                    labels_perm = BASE.stratified_shuffle(labels, strata, rng).astype(int)
                    auc_lp_base = cross_validated_auc(
                        x_base,
                        labels_perm,
                        random_state=39_206 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="none",
                        n_splits=H101_CV_SPLITS,
                    )
                    auc_lp_aug = cross_validated_auc(
                        x_aug,
                        labels_perm,
                        random_state=39_207 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H101_L1_C,
                        n_splits=H101_CV_SPLITS,
                    )
                    null_label[perm_idx] = (
                        float(auc_lp_aug - auc_lp_base)
                        if np.isfinite(auc_lp_aug) and np.isfinite(auc_lp_base)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H101",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H101_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_order, null_sign, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_order = BASE.empirical_upper_tail_p(delta_auc, null_order)
                p_sign = BASE.empirical_upper_tail_p(delta_auc, null_sign)
                p_lab = BASE.empirical_upper_tail_p(delta_auc, null_label)
                p_best = np.nanmin(np.array([p_order, p_sign, p_lab], dtype=float))

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": H101_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70_baseline": float(auc_base),
                        "auc_persistence_derivative_blend": float(auc_aug),
                        "delta_auc_persistence_derivative_minus_h70": float(delta_auc),
                        "derivative_spectrum_pos_minus_neg": float(feature_gap),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95_delta_auc": float(delta_auc - q95),
                        "p_quantile_order_upper": float(p_order),
                        "p_derivative_sign_upper": float(p_sign),
                        "p_label_shuffle_upper": float(p_lab),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h101_persistence_derivative_spectrum_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h101_persistence_derivative_spectrum_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_persistence_derivative_minus_h70": float(
                        group["delta_auc_persistence_derivative_minus_h70"].mean()
                    ),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "mean_derivative_spectrum_pos_minus_neg": float(group["derivative_spectrum_pos_minus_neg"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_persistence_derivative_minus_h70"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h101_persistence_derivative_spectrum_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_persistence_derivative_minus_h70"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_persistence_derivative_minus_h70"] > 0.0).sum())
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


def run_h102_ot_monotone_depth_warp(
    gene2go_upper: dict[str, set[str]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, domain in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.keys()):
        sc_run = BASE.SCGPT_RUNS_BY_DOMAIN[domain][H102_SEED]
        gf_path = BASE.GENEFORMER_EDGE_BY_DOMAIN[domain]

        sc_edge_df = pd.read_csv(sc_run / "cycle1_edge_dataset.tsv", sep="\t")
        gf_df = pd.read_csv(gf_path, sep="\t")
        sc_layer_embeddings = np.load(sc_run / "layer_gene_embeddings.npy", mmap_mode="r")

        top_genes = set(BASE.select_top_genes(sc_edge_df, gene_cap=H102_GENE_CAP))
        sc_sub = sc_edge_df.loc[
            sc_edge_df["source_idx"].isin(top_genes) & sc_edge_df["target_idx"].isin(top_genes)
        ].copy()
        if sc_sub.empty:
            continue

        idx_to_symbol: dict[int, str] = {}
        for row in sc_sub[["source_idx", "source"]].drop_duplicates().itertuples(index=False):
            idx_to_symbol[int(row.source_idx)] = str(row.source).upper()
        for row in sc_sub[["target_idx", "target"]].drop_duplicates().itertuples(index=False):
            idx_to_symbol[int(row.target_idx)] = str(row.target).upper()

        gf_symbols = set(gf_df["source"].astype(str).str.upper()) | set(gf_df["target"].astype(str).str.upper())
        common_indices = [g for g in sorted(idx_to_symbol.keys()) if idx_to_symbol[g] in gf_symbols]
        if len(common_indices) < 100:
            continue

        symbols = [idx_to_symbol[g] for g in common_indices]
        modules = select_go_modules(symbols=symbols, gene2go_upper=gene2go_upper)
        if len(modules) < 10:
            continue

        gf_pos = gf_df.loc[gf_df["label"].astype(int) == 1, ["source", "target"]].copy()
        gf_pos["source"] = gf_pos["source"].astype(str).str.upper()
        gf_pos["target"] = gf_pos["target"].astype(str).str.upper()
        gf_pos = gf_pos.loc[gf_pos["source"].isin(symbols) & gf_pos["target"].isin(symbols)]

        gf_roles = BASE.fit_signatures_geneformer(gf_df=gf_pos.assign(label=1), symbols=symbols)

        sc_layer_roles: dict[int, pd.DataFrame] = {}
        for layer in H102_LAYERS:
            if layer >= sc_layer_embeddings.shape[0]:
                continue
            points = sc_layer_embeddings[layer, np.asarray(common_indices, dtype=int), :]
            sc_layer_roles[layer] = BASE.fit_signatures_scgpt(
                layer_points=points,
                symbols=symbols,
                random_state=39_300 + domain_index * 100 + layer,
                n_neighbors=12,
            )
        if len(sc_layer_roles) != len(H102_LAYERS):
            continue

        module_names, sc_vectors_by_layer, gf_vectors = align_module_vectors(
            modules=modules,
            sc_layer_roles=sc_layer_roles,
            gf_roles=gf_roles,
        )
        if len(module_names) < H102_MODULE_MIN:
            continue

        observed = depth_warp_ot_alignment(
            sc_layer_vectors=sc_vectors_by_layer,
            gf_vectors=gf_vectors,
            warp_candidates=H102_WARP_CANDIDATES,
        )

        concordance = observed["concordance"]

        rng = np.random.default_rng(39_301 + domain_index * 1000)
        null_module = np.empty(H102_NULL_PERM, dtype=float)
        null_depth = np.empty(H102_NULL_PERM, dtype=float)
        null_warp = np.empty(H102_NULL_PERM, dtype=float)
        null_subspace = np.empty(H102_NULL_PERM, dtype=float)

        for perm_idx in range(H102_NULL_PERM):
            # Null 1: module-membership permutation with preserved module sizes.
            perm_modules = build_module_size_preserving_permutation(symbols=symbols, modules=modules, rng=rng)
            _, sc_perm_by_layer, gf_perm = align_module_vectors(
                modules=perm_modules,
                sc_layer_roles=sc_layer_roles,
                gf_roles=gf_roles,
            )
            if len(sc_perm_by_layer) != len(H102_LAYERS) or gf_perm.shape[0] < H102_MODULE_MIN:
                null_module[perm_idx] = float("nan")
            else:
                out_perm = depth_warp_ot_alignment(
                    sc_layer_vectors=sc_perm_by_layer,
                    gf_vectors=gf_perm,
                    warp_candidates=H102_WARP_CANDIDATES,
                )
                null_module[perm_idx] = out_perm["concordance"]
            null_rows.append(
                {
                    "hypothesis_id": "H102",
                    "null_kind": "module_membership_permutation",
                    "domain": domain,
                    "seed_tag": H102_SEED,
                    "split_regime": "other",
                    "layer": -1,
                    "perm_idx": int(perm_idx),
                    "null_value": float(null_module[perm_idx]),
                }
            )

            # Null 2: depth-order permutation (swap shallow/deep layers).
            sc_depth_swap = {
                H102_LAYERS[0]: sc_vectors_by_layer[H102_LAYERS[1]],
                H102_LAYERS[1]: sc_vectors_by_layer[H102_LAYERS[0]],
            }
            out_depth = depth_warp_ot_alignment(
                sc_layer_vectors=sc_depth_swap,
                gf_vectors=gf_vectors,
                warp_candidates=H102_WARP_CANDIDATES,
            )
            null_depth[perm_idx] = out_depth["concordance"]
            null_rows.append(
                {
                    "hypothesis_id": "H102",
                    "null_kind": "depth_order_permutation",
                    "domain": domain,
                    "seed_tag": H102_SEED,
                    "split_regime": "other",
                    "layer": -1,
                    "perm_idx": int(perm_idx),
                    "null_value": float(null_depth[perm_idx]),
                }
            )

            # Null 3: random monotone warp baseline.
            alpha = float(rng.uniform(0.0, 1.0))
            w7 = min(alpha, 1.0 - alpha)
            w11 = max(alpha, 1.0 - alpha)
            null_warp[perm_idx] = depth_warp_ot_alignment_fixed(
                sc_layer_vectors=sc_vectors_by_layer,
                gf_vectors=gf_vectors,
                w7=w7,
                w11=w11,
            )
            null_rows.append(
                {
                    "hypothesis_id": "H102",
                    "null_kind": "random_monotone_warp",
                    "domain": domain,
                    "seed_tag": H102_SEED,
                    "split_regime": "other",
                    "layer": -1,
                    "perm_idx": int(perm_idx),
                    "null_value": float(null_warp[perm_idx]),
                }
            )

            # Null 4: random-subspace baseline.
            q = random_orthogonal_matrix(gf_vectors.shape[1], rng)
            gf_rot = gf_vectors @ q
            out_rot = depth_warp_ot_alignment(
                sc_layer_vectors=sc_vectors_by_layer,
                gf_vectors=gf_rot,
                warp_candidates=H102_WARP_CANDIDATES,
            )
            null_subspace[perm_idx] = out_rot["concordance"]
            null_rows.append(
                {
                    "hypothesis_id": "H102",
                    "null_kind": "random_subspace_alignment",
                    "domain": domain,
                    "seed_tag": H102_SEED,
                    "split_regime": "other",
                    "layer": -1,
                    "perm_idx": int(perm_idx),
                    "null_value": float(null_subspace[perm_idx]),
                }
            )

        all_null = np.concatenate([null_module, null_depth, null_warp, null_subspace])
        q95 = float(np.nanquantile(all_null, 0.95))
        p_module = BASE.empirical_upper_tail_p(concordance, null_module)
        p_depth = BASE.empirical_upper_tail_p(concordance, null_depth)
        p_warp = BASE.empirical_upper_tail_p(concordance, null_warp)
        p_sub = BASE.empirical_upper_tail_p(concordance, null_subspace)
        p_best = np.nanmin(np.array([p_module, p_depth, p_warp, p_sub], dtype=float))

        rows.append(
            {
                "domain": domain,
                "seed_tag": H102_SEED,
                "split_regime": "other",
                "layer": -1,
                "n_modules": int(len(module_names)),
                "module_persistence_ot_concordance": float(concordance),
                "warped_ot_transport_cost": float(observed["transport_cost"]),
                "top_module_overlap_jaccard": float(observed["top_jaccard"]),
                "alignment_rmse": float(observed["alignment_rmse"]),
                "best_warp_w7": float(observed["warp_w7"]),
                "best_warp_w11": float(observed["warp_w11"]),
                "q95_null_concordance": float(q95),
                "null_gap_q95_concordance": float(concordance - q95),
                "p_module_membership_upper": float(p_module),
                "p_depth_order_upper": float(p_depth),
                "p_random_warp_upper": float(p_warp),
                "p_random_subspace_upper": float(p_sub),
                "p_best_upper": float(p_best),
            }
        )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain"])
    by_row_path = ITER_DIR / "h102_ot_monotone_depth_warp_by_domain.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "perm_idx"])
    null_path = ITER_DIR / "h102_ot_monotone_depth_warp_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for domain, group in by_row_df.groupby("domain", sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "n_rows": int(group.shape[0]),
                    "mean_module_persistence_ot_concordance": float(group["module_persistence_ot_concordance"].mean()),
                    "mean_warped_ot_transport_cost": float(group["warped_ot_transport_cost"].mean()),
                    "mean_null_gap_q95_concordance": float(group["null_gap_q95_concordance"].mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_concordance"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain"])
    summary_path = ITER_DIR / "h102_ot_monotone_depth_warp_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_concordance": float(by_row_df["module_persistence_ot_concordance"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_count": int((summary_df["mean_null_gap_q95_concordance"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_domain": str(by_row_path),
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

    h100_summary = run_h100_relative_persistence_contrast(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h101_summary = run_h101_persistence_derivative_spectrum(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h102_summary = run_h102_ot_monotone_depth_warp(
        gene2go_upper=gene2go_upper,
    )

    summary = {
        "iteration": "iter_0039",
        "h100": h100_summary,
        "h101": h101_summary,
        "h102": h102_summary,
    }
    summary_path = ITER_DIR / "iter0039_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
