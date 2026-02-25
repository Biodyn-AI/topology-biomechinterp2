from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import dionysus as d
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ITER_DIR = Path("iterations/iter_0043")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base")

# H112 / N565: semi-Markov biologically anchored grammar rescue.
H112_SEED = "seed42_main"
H112_LAYERS = [0, 3, 7, 11]
H112_GENE_CAP = 190
H112_NEIGHBORS = 12
H112_EDGE_SAMPLE = 300
H112_CV_SPLITS = 4
H112_NULL_PERM = 24
H112_TF_BINS = 3
H112_SUPPORT_BINS = 3

# H113 / N552: depth-transition zigzag long-bar mass + birth-depth entropy.
H113_SEED = "seed42_main"
H113_LAYERS = [0, 3, 7, 11]
H113_GENE_CAP = 130
H113_KNN = 8
H113_NULL_PERM = 6

# H114 / N559: intrinsic-dimension hysteresis screen.
H114_SEED = "seed42_main"
H114_LAYERS = [0, 3, 7, 11]
H114_GENE_CAP = 180
H114_NEIGHBORS = 12
H114_EDGE_SAMPLE = 240
H114_CV_SPLITS = 4
H114_NULL_PERM = 12
H114_L1_C = 0.22
H114_K_VALUES = [4, 6, 8, 10, 12, 16]


def ensure_required_inputs() -> None:
    required_paths = [
        BASE.DOROTHEA_PATH,
        BASE.GENE2GO_PATH,
        BASE.OMNIPATH_INTERACTIONS_PATH,
        BASE.STRING_CACHE_PATH,
    ]
    for run_map in BASE.SCGPT_RUNS_BY_DOMAIN.values():
        run_dir = run_map[H112_SEED]
        required_paths.append(run_dir / "cycle1_edge_dataset.tsv")
        required_paths.append(run_dir / "layer_gene_embeddings.npy")

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
) -> tuple[np.ndarray, np.ndarray]:
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
    return baseline, defect


def confidence_weighted_geodesic(geodesic: np.ndarray, support_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sym_support = 0.5 * (support_dir + support_dir.T)
    sym_support = np.clip(sym_support, 0.0, 1.0)

    weighted = geodesic / (0.35 + sym_support)
    weighted = np.asarray(weighted, dtype=float)
    np.fill_diagonal(weighted, 0.0)
    return weighted, sym_support


def edge_degree_sum(points: np.ndarray, n_neighbors: int, source_local: np.ndarray, target_local: np.ndarray) -> np.ndarray:
    knn_edges = BASE.build_knn_edge_array(points=points, n_neighbors=n_neighbors)
    neighbors = BASE.adjacency_neighbors(points.shape[0], knn_edges)
    deg = np.asarray([len(nb) for nb in neighbors], dtype=float)
    return deg[source_local] + deg[target_local]


def build_edge_strata(edge_length: np.ndarray, degree_sum: np.ndarray, max_len_bins: int, max_deg_bins: int) -> np.ndarray:
    bins_len = BASE.degree_bins(edge_length, max_bins=max_len_bins)
    bins_deg = BASE.degree_bins(degree_sum, max_bins=max_deg_bins)
    return (bins_len * 16 + bins_deg).astype(int)


def quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    ranks = pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).to_numpy(dtype=float)
    bins = np.minimum((ranks * n_bins).astype(int), n_bins - 1)
    return bins.astype(int)


def sign_states(values: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.ones(arr.shape[0], dtype=int)
    out[arr > threshold] = 2
    out[arr < -threshold] = 0
    return out


def tf_activity_by_symbol(symbols: list[str], dorothea_map: dict[tuple[str, str], int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for sym in symbols:
        key = sym.upper()
        count = 0
        for src, _ in dorothea_map.keys():
            if src == key:
                count += 1
        out[key] = float(count)
    return out


def sequence_transition_entropy(seq: np.ndarray) -> float:
    if seq.size < 2:
        return 0.0
    trans = [f"{int(seq[i])}_{int(seq[i+1])}" for i in range(seq.size - 1)]
    _, counts = np.unique(trans, return_counts=True)
    probs = counts / np.clip(np.sum(counts), 1e-8, None)
    return float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))


def sequence_dwell_entropy(seq: np.ndarray) -> float:
    if seq.size == 0:
        return 0.0
    runs: list[int] = []
    current = int(seq[0])
    run_len = 1
    for val in seq[1:]:
        iv = int(val)
        if iv == current:
            run_len += 1
        else:
            runs.append(run_len)
            current = iv
            run_len = 1
    runs.append(run_len)
    bins = [0, 0, 0]
    for r in runs:
        if r <= 1:
            bins[0] += 1
        elif r == 2:
            bins[1] += 1
        else:
            bins[2] += 1
    probs = np.asarray(bins, dtype=float)
    probs = probs / np.clip(np.sum(probs), 1e-8, None)
    return float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))


def sequence_segments(seq: np.ndarray, n_dwell_bins: int = 3) -> list[int]:
    if seq.size == 0:
        return []

    segments: list[int] = []
    current = int(seq[0])
    run_len = 1

    def to_dwell_bin(length: int) -> int:
        if length <= 1:
            return 0
        if length == 2:
            return 1
        return 2

    for val in seq[1:]:
        iv = int(val)
        if iv == current:
            run_len += 1
        else:
            db = to_dwell_bin(run_len)
            segments.append(current * n_dwell_bins + db)
            current = iv
            run_len = 1

    db_last = to_dwell_bin(run_len)
    segments.append(current * n_dwell_bins + db_last)
    return segments


def signature_preserving_sequence_permutation(tokens: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(tokens, dtype=int)
    out = x.copy()

    signatures: dict[tuple[tuple[int, ...], int], list[int]] = {}
    for i in range(x.shape[0]):
        seq = x[i]
        sorted_tok = tuple(sorted(int(v) for v in seq.tolist()))
        n_changes = int(np.sum(seq[1:] != seq[:-1]))
        key = (sorted_tok, n_changes)
        signatures.setdefault(key, []).append(i)

    for idx in signatures.values():
        if len(idx) > 1:
            perm = rng.permutation(np.asarray(idx, dtype=int))
            out[np.asarray(idx, dtype=int)] = out[perm]

    return out


def gaussian_llr(value: float, pos_mean: float, pos_sd: float, neg_mean: float, neg_sd: float) -> float:
    p_pos = norm.logpdf(value, loc=pos_mean, scale=max(pos_sd, 1e-6))
    p_neg = norm.logpdf(value, loc=neg_mean, scale=max(neg_sd, 1e-6))
    return float(p_pos - p_neg)


def second_order_sequence_scores(
    tokens: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    n_splits: int,
    alpha: float = 1.0,
) -> np.ndarray:
    x = np.asarray(tokens, dtype=int)
    y = np.asarray(labels, dtype=int)

    if x.ndim != 2 or x.shape[0] != y.size or np.unique(y).size < 2:
        return np.full(y.size, np.nan, dtype=float)

    max_splits = min(n_splits, min_class_count(y))
    if max_splits < 2:
        return np.full(y.size, np.nan, dtype=float)

    n_states = int(np.max(x)) + 1
    n_layers = x.shape[1]
    if n_layers < 3:
        return np.full(y.size, np.nan, dtype=float)

    cv = StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=random_state)
    out_scores = np.full(y.shape[0], np.nan, dtype=float)

    for tr, te in cv.split(x, y):
        x_tr = x[tr]
        y_tr = y[tr]

        pos_start = np.full(n_states, alpha, dtype=float)
        neg_start = np.full(n_states, alpha, dtype=float)

        pos_second = np.full((n_states, n_states), alpha, dtype=float)
        neg_second = np.full((n_states, n_states), alpha, dtype=float)

        pos_tri = np.full((n_states, n_states, n_states), alpha, dtype=float)
        neg_tri = np.full((n_states, n_states, n_states), alpha, dtype=float)

        for seq, lbl in zip(x_tr, y_tr):
            s0, s1 = int(seq[0]), int(seq[1])
            if int(lbl) == 1:
                pos_start[s0] += 1.0
                pos_second[s0, s1] += 1.0
            else:
                neg_start[s0] += 1.0
                neg_second[s0, s1] += 1.0

            for t in range(2, n_layers):
                a = int(seq[t - 2])
                b = int(seq[t - 1])
                c = int(seq[t])
                if int(lbl) == 1:
                    pos_tri[a, b, c] += 1.0
                else:
                    neg_tri[a, b, c] += 1.0

        pos_start = pos_start / np.clip(pos_start.sum(), 1e-8, None)
        neg_start = neg_start / np.clip(neg_start.sum(), 1e-8, None)

        pos_second = pos_second / np.clip(pos_second.sum(axis=1, keepdims=True), 1e-8, None)
        neg_second = neg_second / np.clip(neg_second.sum(axis=1, keepdims=True), 1e-8, None)

        pos_tri = pos_tri / np.clip(pos_tri.sum(axis=2, keepdims=True), 1e-8, None)
        neg_tri = neg_tri / np.clip(neg_tri.sum(axis=2, keepdims=True), 1e-8, None)

        for row_idx, seq in zip(te, x[te]):
            s0, s1 = int(seq[0]), int(seq[1])
            llr = float(np.log(np.clip(pos_start[s0], 1e-8, 1.0) / np.clip(neg_start[s0], 1e-8, 1.0)))
            llr += float(np.log(np.clip(pos_second[s0, s1], 1e-8, 1.0) / np.clip(neg_second[s0, s1], 1e-8, 1.0)))

            for t in range(2, n_layers):
                a = int(seq[t - 2])
                b = int(seq[t - 1])
                c = int(seq[t])
                llr += float(np.log(np.clip(pos_tri[a, b, c], 1e-8, 1.0) / np.clip(neg_tri[a, b, c], 1e-8, 1.0)))

            out_scores[row_idx] = llr

    return out_scores


def second_order_sequence_auc(
    tokens: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    n_splits: int,
) -> tuple[float, np.ndarray]:
    scores = second_order_sequence_scores(tokens=tokens, labels=labels, random_state=random_state, n_splits=n_splits)
    auc = BASE.safe_auc(np.asarray(labels, dtype=int), scores)
    return auc, scores


def semi_markov_sequence_scores(
    tokens: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    n_splits: int,
    alpha: float = 1.0,
    forbidden_tau: float = 0.015,
    forbidden_ratio: float = 2.5,
    forbidden_penalty: float = 0.9,
) -> np.ndarray:
    x = np.asarray(tokens, dtype=int)
    y = np.asarray(labels, dtype=int)

    if x.ndim != 2 or x.shape[0] != y.size or np.unique(y).size < 2:
        return np.full(y.size, np.nan, dtype=float)

    max_splits = min(n_splits, min_class_count(y))
    if max_splits < 2:
        return np.full(y.size, np.nan, dtype=float)

    n_states = int(np.max(x)) + 1
    n_seg_states = n_states * 3

    trans_entropy = np.asarray([sequence_transition_entropy(seq) for seq in x], dtype=float)
    dwell_entropy = np.asarray([sequence_dwell_entropy(seq) for seq in x], dtype=float)
    segments = [sequence_segments(seq, n_dwell_bins=3) for seq in x]

    cv = StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=random_state)
    scores = np.full(y.shape[0], np.nan, dtype=float)

    for tr, te in cv.split(x, y):
        y_tr = y[tr]

        pos_start = np.full(n_seg_states, alpha, dtype=float)
        neg_start = np.full(n_seg_states, alpha, dtype=float)
        pos_trans = np.full((n_seg_states, n_seg_states), alpha, dtype=float)
        neg_trans = np.full((n_seg_states, n_seg_states), alpha, dtype=float)

        for idx in tr:
            seg = segments[idx]
            if len(seg) == 0:
                continue
            if int(y[idx]) == 1:
                pos_start[seg[0]] += 1.0
                for a, b in zip(seg[:-1], seg[1:]):
                    pos_trans[a, b] += 1.0
            else:
                neg_start[seg[0]] += 1.0
                for a, b in zip(seg[:-1], seg[1:]):
                    neg_trans[a, b] += 1.0

        pos_start = pos_start / np.clip(np.sum(pos_start), 1e-8, None)
        neg_start = neg_start / np.clip(np.sum(neg_start), 1e-8, None)
        pos_trans = pos_trans / np.clip(np.sum(pos_trans, axis=1, keepdims=True), 1e-8, None)
        neg_trans = neg_trans / np.clip(np.sum(neg_trans, axis=1, keepdims=True), 1e-8, None)

        forbidden_mask = (pos_trans < forbidden_tau) & (neg_trans > forbidden_ratio * pos_trans)

        pos_te = trans_entropy[tr][y_tr == 1]
        neg_te = trans_entropy[tr][y_tr == 0]
        pos_de = dwell_entropy[tr][y_tr == 1]
        neg_de = dwell_entropy[tr][y_tr == 0]

        pos_te_mu = float(np.mean(pos_te)) if pos_te.size else 0.0
        neg_te_mu = float(np.mean(neg_te)) if neg_te.size else 0.0
        pos_te_sd = float(np.std(pos_te)) if pos_te.size else 1.0
        neg_te_sd = float(np.std(neg_te)) if neg_te.size else 1.0

        pos_de_mu = float(np.mean(pos_de)) if pos_de.size else 0.0
        neg_de_mu = float(np.mean(neg_de)) if neg_de.size else 0.0
        pos_de_sd = float(np.std(pos_de)) if pos_de.size else 1.0
        neg_de_sd = float(np.std(neg_de)) if neg_de.size else 1.0

        for idx in te:
            seg = segments[idx]
            if len(seg) == 0:
                scores[idx] = 0.0
                continue

            llr = float(np.log(np.clip(pos_start[seg[0]], 1e-8, 1.0) / np.clip(neg_start[seg[0]], 1e-8, 1.0)))

            forbidden_count = 0
            trans_count = 0
            for a, b in zip(seg[:-1], seg[1:]):
                llr += float(np.log(np.clip(pos_trans[a, b], 1e-8, 1.0) / np.clip(neg_trans[a, b], 1e-8, 1.0)))
                if forbidden_mask[a, b]:
                    forbidden_count += 1
                trans_count += 1

            forbidden_rate = float(forbidden_count / max(1, trans_count))
            llr -= forbidden_penalty * forbidden_rate

            llr += gaussian_llr(
                value=float(trans_entropy[idx]),
                pos_mean=pos_te_mu,
                pos_sd=pos_te_sd,
                neg_mean=neg_te_mu,
                neg_sd=neg_te_sd,
            )
            llr += gaussian_llr(
                value=float(dwell_entropy[idx]),
                pos_mean=pos_de_mu,
                pos_sd=pos_de_sd,
                neg_mean=neg_de_mu,
                neg_sd=neg_de_sd,
            )
            scores[idx] = llr

    return scores


def semi_markov_sequence_auc(
    tokens: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    n_splits: int,
) -> tuple[float, np.ndarray]:
    scores = semi_markov_sequence_scores(
        tokens=tokens,
        labels=labels,
        random_state=random_state,
        n_splits=n_splits,
    )
    auc = BASE.safe_auc(np.asarray(labels, dtype=int), scores)
    return auc, scores


def build_knn_edges_for_subset(
    points_union: np.ndarray,
    subset_positions: np.ndarray,
    n_neighbors: int,
) -> set[tuple[int, int]]:
    subset_positions = np.asarray(subset_positions, dtype=int)
    if subset_positions.size < 3:
        return set()

    k = max(2, min(n_neighbors, subset_positions.size - 1))
    sub_points = points_union[subset_positions]
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(sub_points)
    _, indices = nbrs.kneighbors(sub_points)

    edges: set[tuple[int, int]] = set()
    for i_local in range(subset_positions.size):
        src_union = int(subset_positions[i_local])
        for j_local in indices[i_local, 1:]:
            tgt_union = int(subset_positions[int(j_local)])
            if src_union == tgt_union:
                continue
            edge = (min(src_union, tgt_union), max(src_union, tgt_union))
            edges.add(edge)
    return edges


def zigzag_pair_metrics(
    n_vertices: int,
    edges_a: set[tuple[int, int]],
    edges_b: set[tuple[int, int]],
) -> dict[str, float]:
    simplices: list[d.Simplex] = []
    intervals_by_key: dict[tuple[int, ...], list[float]] = {}

    for v in range(n_vertices):
        simplex = d.Simplex([int(v)], 0.0)
        simplices.append(simplex)
        intervals_by_key[(int(v),)] = [0.0, 3.0]

    union_edges = edges_a | edges_b
    for edge in sorted(union_edges):
        u, v = int(edge[0]), int(edge[1])
        simplex = d.Simplex([u, v], 1.0)
        simplices.append(simplex)

        in_a = edge in edges_a
        in_b = edge in edges_b
        if in_a and in_b:
            interval = [0.0, 3.0]
        elif in_a:
            interval = [0.0, 2.0]
        else:
            interval = [1.0, 3.0]
        intervals_by_key[(u, v)] = interval

    filtration = d.Filtration(simplices)
    times: list[list[float]] = [[] for _ in range(len(filtration))]
    for simplex in filtration:
        verts = tuple(int(v) for v in simplex)
        key = tuple(sorted(verts))
        idx = filtration.index(simplex)
        times[idx] = intervals_by_key.get(key, [0.0, 3.0])

    _, diagrams, _ = d.zigzag_homology_persistence(filtration, times)
    if len(diagrams) < 2:
        return {
            "h1_total_lifetime": 0.0,
            "h1_long_mass": 0.0,
            "h1_long_count": 0.0,
            "h1_birth_centroid": 0.0,
        }

    life: list[float] = []
    births: list[float] = []
    long_life: list[float] = []
    for point in diagrams[1]:
        birth = float(point.birth)
        death = float(point.death)
        if not np.isfinite(birth) or not np.isfinite(death):
            continue
        lt = max(0.0, death - birth)
        life.append(lt)
        births.append(birth)
        if lt >= 1.0:
            long_life.append(lt)

    if not life:
        return {
            "h1_total_lifetime": 0.0,
            "h1_long_mass": 0.0,
            "h1_long_count": 0.0,
            "h1_birth_centroid": 0.0,
        }

    if long_life:
        long_mass = float(np.sum(np.asarray(long_life, dtype=float)))
        long_count = float(len(long_life))
        birth_centroid = float(np.mean(np.asarray(births, dtype=float)))
    else:
        long_mass = 0.0
        long_count = 0.0
        birth_centroid = 0.0

    return {
        "h1_total_lifetime": float(np.sum(np.asarray(life, dtype=float))),
        "h1_long_mass": long_mass,
        "h1_long_count": long_count,
        "h1_birth_centroid": birth_centroid,
    }


def entropy_from_vector(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = np.clip(arr, 0.0, None)
    if arr.size == 0 or float(np.sum(arr)) <= 0.0:
        return 0.0
    p = arr / np.sum(arr)
    return float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))


def degree_bin_relabel_positions(
    points_ref: np.ndarray,
    positions: np.ndarray,
    n_neighbors: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pos = np.asarray(positions, dtype=int)
    if pos.size < 6:
        return pos.copy()

    sub_points = points_ref[pos]
    sub_edges = BASE.build_knn_edge_array(sub_points, n_neighbors=n_neighbors)
    neighbors = BASE.adjacency_neighbors(sub_points.shape[0], sub_edges)
    deg = np.asarray([len(nb) for nb in neighbors], dtype=float)
    bins = BASE.degree_bins(deg, max_bins=5)

    relabel = np.arange(pos.size, dtype=int)
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        if idx.size > 1:
            relabel[idx] = rng.permutation(idx)

    return pos[relabel]


def compute_id_curves(
    points: np.ndarray,
    k_values: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    k_max = max(k_values)
    n = points.shape[0]
    k_eff = min(k_max, n - 1)
    if k_eff < 4:
        raise RuntimeError("Too few neighbors for ID hysteresis computation")

    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nbrs.fit(points)
    d_full, _ = nbrs.kneighbors(points)
    d_local = np.asarray(d_full[:, 1:], dtype=float)

    forward_cols: list[np.ndarray] = []
    reverse_cols: list[np.ndarray] = []

    for k in k_values:
        k_use = min(k, d_local.shape[1])
        if k_use < 3:
            k_use = min(3, d_local.shape[1])
        forward_cols.append(BASE.local_id_mle(d_local[:, :k_use]))

        start = max(0, k_use - 1)
        shell = d_local[:, start:]
        if shell.shape[1] < 3:
            shell = d_local[:, : min(3, d_local.shape[1])]
        reverse_cols.append(BASE.local_id_mle(shell))

    return np.column_stack(forward_cols), np.column_stack(reverse_cols)


def node_hysteresis_bundle(
    id_forward: np.ndarray,
    id_reverse: np.ndarray,
) -> np.ndarray:
    forward = np.asarray(id_forward, dtype=float)
    reverse = np.asarray(id_reverse, dtype=float)

    area = np.mean(np.abs(forward - reverse), axis=1)
    endpoint_gap = np.abs(forward[:, 0] - reverse[:, 0])

    df = np.diff(forward, axis=1)
    dr = np.diff(reverse, axis=1)
    flip = np.mean((np.sign(df) != np.sign(dr)).astype(float), axis=1) if df.shape[1] > 0 else np.zeros(forward.shape[0])

    return np.column_stack([area, endpoint_gap, flip])


def edge_features_from_node_bundle(
    node_bundle: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
) -> np.ndarray:
    src_feat = node_bundle[source_local]
    tgt_feat = node_bundle[target_local]
    mean_feat = 0.5 * (src_feat + tgt_feat)
    abs_diff = np.abs(src_feat - tgt_feat)
    return np.column_stack([mean_feat, abs_diff])


def run_h112_semimarkov_biogrammar_rescue(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H112_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H112_GENE_CAP))
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
            tf_activity = tf_activity_by_symbol(symbols=symbols, dorothea_map=dorothea_map)

            gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
            source_local_all = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
            target_local_all = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
            labels_all = split_edges["label"].to_numpy(dtype=int)

            rng = np.random.default_rng(43_120 + domain_index * 1000 + split_index * 100)
            sample_idx = stratified_index_sample(labels_all, max_n=H112_EDGE_SAMPLE, rng=rng)
            if sample_idx.size < 120:
                continue

            source_local = source_local_all[sample_idx]
            target_local = target_local_all[sample_idx]
            labels = labels_all[sample_idx]

            edge_margin = support_dir[source_local, target_local] - support_dir[target_local, source_local]
            sign_state = sign_states(edge_margin)

            tf_source_activity = np.asarray([tf_activity[symbols[idx].upper()] for idx in source_local], dtype=float)
            tf_bin = quantile_bins(tf_source_activity, n_bins=H112_TF_BINS)

            token_layers: list[np.ndarray] = []
            h70_deep = None
            edge_length_deep = None
            degree_sum_deep = None

            for layer in H112_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    token_layers = []
                    break

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=43_121 + domain_index * 1000 + split_index * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H112_NEIGHBORS)
                geodesic_w, support_sym = confidence_weighted_geodesic(geodesic, support_dir)

                edge_support_layer = support_sym[source_local, target_local] / np.clip(
                    geodesic_w[source_local, target_local],
                    1e-8,
                    None,
                )
                support_bin = quantile_bins(edge_support_layer, n_bins=H112_SUPPORT_BINS)

                token = tf_bin * (H112_SUPPORT_BINS * 3) + support_bin * 3 + sign_state
                token_layers.append(token.astype(int))

                if layer == H112_LAYERS[-1]:
                    _, h70_deep = compute_h70_scores(
                        geodesic=geodesic_w,
                        support_dir=support_dir,
                        source_local=source_local,
                        target_local=target_local,
                        triangle_k=[8, 12, 16],
                    )
                    edge_length_deep = geodesic_w[source_local, target_local]
                    degree_sum_deep = edge_degree_sum(
                        points=points_pca,
                        n_neighbors=H112_NEIGHBORS,
                        source_local=source_local,
                        target_local=target_local,
                    )

            if len(token_layers) != len(H112_LAYERS) or h70_deep is None:
                continue

            tokens = np.column_stack(token_layers)

            auc_h70 = BASE.safe_auc(labels, h70_deep)
            auc_second, _ = second_order_sequence_auc(
                tokens=tokens,
                labels=labels,
                random_state=43_122 + domain_index * 1000 + split_index * 100,
                n_splits=H112_CV_SPLITS,
            )
            auc_semi, _ = semi_markov_sequence_auc(
                tokens=tokens,
                labels=labels,
                random_state=43_123 + domain_index * 1000 + split_index * 100,
                n_splits=H112_CV_SPLITS,
            )

            delta_vs_second = float(auc_semi - auc_second) if np.isfinite(auc_semi) and np.isfinite(auc_second) else float("nan")
            delta_vs_h70 = float(auc_semi - auc_h70) if np.isfinite(auc_semi) and np.isfinite(auc_h70) else float("nan")

            strata = build_edge_strata(
                edge_length=np.asarray(edge_length_deep, dtype=float),
                degree_sum=np.asarray(degree_sum_deep, dtype=float),
                max_len_bins=6,
                max_deg_bins=4,
            )

            null_sig = np.empty(H112_NULL_PERM, dtype=float)
            null_order = np.empty(H112_NULL_PERM, dtype=float)
            null_label = np.empty(H112_NULL_PERM, dtype=float)

            for perm_idx in range(H112_NULL_PERM):
                tokens_sig = signature_preserving_sequence_permutation(tokens=tokens, rng=rng)
                auc2_sig, _ = second_order_sequence_auc(
                    tokens=tokens_sig,
                    labels=labels,
                    random_state=43_124 + domain_index * 10_000 + split_index * 1000 + perm_idx,
                    n_splits=H112_CV_SPLITS,
                )
                aucs_sig, _ = semi_markov_sequence_auc(
                    tokens=tokens_sig,
                    labels=labels,
                    random_state=43_125 + domain_index * 10_000 + split_index * 1000 + perm_idx,
                    n_splits=H112_CV_SPLITS,
                )
                null_sig[perm_idx] = (
                    float(aucs_sig - auc2_sig)
                    if np.isfinite(aucs_sig) and np.isfinite(auc2_sig)
                    else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H112",
                        "null_kind": "occupancy_transition_signature_sequence_permutation",
                        "domain": domain,
                        "seed_tag": H112_SEED,
                        "split_regime": split_regime,
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_sig[perm_idx]),
                    }
                )

                order = rng.permutation(tokens.shape[1])
                tokens_ord = tokens[:, order]
                auc2_ord, _ = second_order_sequence_auc(
                    tokens=tokens_ord,
                    labels=labels,
                    random_state=43_126 + domain_index * 10_000 + split_index * 1000 + perm_idx,
                    n_splits=H112_CV_SPLITS,
                )
                aucs_ord, _ = semi_markov_sequence_auc(
                    tokens=tokens_ord,
                    labels=labels,
                    random_state=43_127 + domain_index * 10_000 + split_index * 1000 + perm_idx,
                    n_splits=H112_CV_SPLITS,
                )
                null_order[perm_idx] = (
                    float(aucs_ord - auc2_ord)
                    if np.isfinite(aucs_ord) and np.isfinite(auc2_ord)
                    else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H112",
                        "null_kind": "layer_order_permutation",
                        "domain": domain,
                        "seed_tag": H112_SEED,
                        "split_regime": split_regime,
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_order[perm_idx]),
                    }
                )

                labels_perm = BASE.stratified_shuffle(labels, strata, rng).astype(int)
                auc2_lp, _ = second_order_sequence_auc(
                    tokens=tokens,
                    labels=labels_perm,
                    random_state=43_128 + domain_index * 10_000 + split_index * 1000 + perm_idx,
                    n_splits=H112_CV_SPLITS,
                )
                aucs_lp, _ = semi_markov_sequence_auc(
                    tokens=tokens,
                    labels=labels_perm,
                    random_state=43_129 + domain_index * 10_000 + split_index * 1000 + perm_idx,
                    n_splits=H112_CV_SPLITS,
                )
                null_label[perm_idx] = (
                    float(aucs_lp - auc2_lp)
                    if np.isfinite(aucs_lp) and np.isfinite(auc2_lp)
                    else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H112",
                        "null_kind": "label_permutation",
                        "domain": domain,
                        "seed_tag": H112_SEED,
                        "split_regime": split_regime,
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_label[perm_idx]),
                    }
                )

            all_null = np.concatenate([null_sig, null_order, null_label])
            q95 = float(np.nanquantile(all_null, 0.95))

            p_sig = BASE.empirical_upper_tail_p(delta_vs_second, null_sig)
            p_order = BASE.empirical_upper_tail_p(delta_vs_second, null_order)
            p_label = BASE.empirical_upper_tail_p(delta_vs_second, null_label)
            p_best = np.nanmin(np.array([p_sig, p_order, p_label], dtype=float))

            rows.append(
                {
                    "domain": domain,
                    "seed_tag": H112_SEED,
                    "split_regime": split_regime,
                    "layer": -1,
                    "n_edges_eval": int(labels.size),
                    "n_positive_eval": int(labels.sum()),
                    "auc_h70_deep_layer": float(auc_h70),
                    "auc_second_order_fsm": float(auc_second),
                    "auc_semimarkov_bio_grammar": float(auc_semi),
                    "delta_auc_semimarkov_minus_second_order": float(delta_vs_second),
                    "delta_auc_semimarkov_minus_h70": float(delta_vs_h70),
                    "q95_null_delta_auc": float(q95),
                    "null_gap_q95_delta_auc": float(delta_vs_second - q95),
                    "p_signature_shuffle_upper": float(p_sig),
                    "p_layer_order_upper": float(p_order),
                    "p_label_shuffle_upper": float(p_label),
                    "p_best_upper": float(p_best),
                }
            )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime"])
    by_row_path = ITER_DIR / "h112_semimarkov_biogrammar_by_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h112_semimarkov_biogrammar_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_semimarkov_minus_second_order": float(
                        group["delta_auc_semimarkov_minus_second_order"].mean()
                    ),
                    "mean_delta_auc_semimarkov_minus_h70": float(group["delta_auc_semimarkov_minus_h70"].mean()),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_semimarkov_minus_second_order"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h112_semimarkov_biogrammar_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_semimarkov_minus_second_order"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_semimarkov_minus_second_order"] > 0.0).sum()) if not summary_df.empty else 0,
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95_delta_auc"] > 0.0).sum()) if not summary_df.empty else 0,
        "artifact_paths": {
            "by_domain_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h113_depth_zigzag_longbar() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    transition_pairs = list(zip(H113_LAYERS[:-1], H113_LAYERS[1:]))

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H113_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H113_GENE_CAP))
            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
            ].copy()
            if split_edges["label"].nunique() < 2:
                continue

            # Positive/negative gene pools are the biological anchor for this topology test.
            pos_genes = set(
                pd.concat(
                    [
                        split_edges.loc[split_edges["label"].astype(int) == 1, "source_idx"],
                        split_edges.loc[split_edges["label"].astype(int) == 1, "target_idx"],
                    ]
                ).astype(int)
            )
            neg_genes = set(
                pd.concat(
                    [
                        split_edges.loc[split_edges["label"].astype(int) == 0, "source_idx"],
                        split_edges.loc[split_edges["label"].astype(int) == 0, "target_idx"],
                    ]
                ).astype(int)
            )

            union_genes = np.array(sorted(pos_genes | neg_genes), dtype=int)
            if union_genes.size < 90:
                continue

            union_gene_set = set(int(g) for g in union_genes.tolist())
            pos_only = np.array(sorted([g for g in pos_genes if g in union_gene_set]), dtype=int)
            neg_only = np.array(sorted([g for g in neg_genes if g in union_gene_set]), dtype=int)
            if pos_only.size < 35 or neg_only.size < 35:
                continue

            gene_to_union = {int(g): int(i) for i, g in enumerate(union_genes)}
            pos_positions = np.array([gene_to_union[int(g)] for g in pos_only], dtype=int)
            neg_positions = np.array([gene_to_union[int(g)] for g in neg_only], dtype=int)

            layer_points: dict[int, np.ndarray] = {}
            for layer in H113_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    layer_points = {}
                    break
                points = layer_embeddings[layer, union_genes, :]
                layer_points[layer] = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=43_310 + domain_index * 1000 + split_index * 100 + layer,
                )
            if len(layer_points) != len(H113_LAYERS):
                continue

            # Cache class-specific kNN graphs per layer to avoid rebuilding them in every null draw.
            edges_pos_by_layer: dict[int, set[tuple[int, int]]] = {}
            edges_neg_by_layer: dict[int, set[tuple[int, int]]] = {}
            cache_ok = True
            for layer in H113_LAYERS:
                edges_pos_by_layer[layer] = build_knn_edges_for_subset(layer_points[layer], pos_positions, H113_KNN)
                edges_neg_by_layer[layer] = build_knn_edges_for_subset(layer_points[layer], neg_positions, H113_KNN)
                if len(edges_pos_by_layer[layer]) < 20 or len(edges_neg_by_layer[layer]) < 20:
                    cache_ok = False
                    break
            if not cache_ok:
                continue

            pos_mass = []
            neg_mass = []

            for la, lb in transition_pairs:
                edges_pos_a = edges_pos_by_layer[la]
                edges_pos_b = edges_pos_by_layer[lb]
                edges_neg_a = edges_neg_by_layer[la]
                edges_neg_b = edges_neg_by_layer[lb]

                met_pos = zigzag_pair_metrics(n_vertices=union_genes.size, edges_a=edges_pos_a, edges_b=edges_pos_b)
                met_neg = zigzag_pair_metrics(n_vertices=union_genes.size, edges_a=edges_neg_a, edges_b=edges_neg_b)

                pos_mass.append(met_pos["h1_long_mass"])
                neg_mass.append(met_neg["h1_long_mass"])

            if len(pos_mass) != len(transition_pairs):
                continue

            pos_mass_arr = np.asarray(pos_mass, dtype=float)
            neg_mass_arr = np.asarray(neg_mass, dtype=float)

            delta_long_mass = float(np.mean(pos_mass_arr - neg_mass_arr))
            pos_birth_entropy = entropy_from_vector(pos_mass_arr)
            neg_birth_entropy = entropy_from_vector(neg_mass_arr)
            entropy_gap = float(pos_birth_entropy - neg_birth_entropy)

            rng = np.random.default_rng(43_311 + domain_index * 1000 + split_index * 100)
            null_layer_order = np.empty(H113_NULL_PERM, dtype=float)
            null_degree_relabel = np.empty(H113_NULL_PERM, dtype=float)
            null_label = np.empty(H113_NULL_PERM, dtype=float)

            for perm_idx in range(H113_NULL_PERM):
                # Null 1: permute layer order before building transitions.
                order = rng.permutation(len(H113_LAYERS))
                perm_layers = [H113_LAYERS[i] for i in order]
                perm_pairs = list(zip(perm_layers[:-1], perm_layers[1:]))
                pm = []
                nm = []
                for la, lb in perm_pairs:
                    edges_pos_a = edges_pos_by_layer[la]
                    edges_pos_b = edges_pos_by_layer[lb]
                    edges_neg_a = edges_neg_by_layer[la]
                    edges_neg_b = edges_neg_by_layer[lb]
                    met_pos = zigzag_pair_metrics(n_vertices=union_genes.size, edges_a=edges_pos_a, edges_b=edges_pos_b)
                    met_neg = zigzag_pair_metrics(n_vertices=union_genes.size, edges_a=edges_neg_a, edges_b=edges_neg_b)
                    pm.append(met_pos["h1_long_mass"])
                    nm.append(met_neg["h1_long_mass"])
                null_layer_order[perm_idx] = float(np.mean(np.asarray(pm, dtype=float) - np.asarray(nm, dtype=float)))
                null_rows.append(
                    {
                        "hypothesis_id": "H113",
                        "null_kind": "layer_order_permutation",
                        "domain": domain,
                        "seed_tag": H113_SEED,
                        "split_regime": split_regime,
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_layer_order[perm_idx]),
                    }
                )

                # Null 2: degree-bin relabeling for layer-B node identities.
                pm = []
                nm = []
                for la, lb in transition_pairs:
                    pos_b_relabel = degree_bin_relabel_positions(
                        points_ref=layer_points[la],
                        positions=pos_positions,
                        n_neighbors=H113_KNN,
                        rng=rng,
                    )
                    neg_b_relabel = degree_bin_relabel_positions(
                        points_ref=layer_points[la],
                        positions=neg_positions,
                        n_neighbors=H113_KNN,
                        rng=rng,
                    )
                    edges_pos_a = edges_pos_by_layer[la]
                    edges_pos_b = build_knn_edges_for_subset(layer_points[lb], pos_b_relabel, H113_KNN)
                    edges_neg_a = edges_neg_by_layer[la]
                    edges_neg_b = build_knn_edges_for_subset(layer_points[lb], neg_b_relabel, H113_KNN)
                    met_pos = zigzag_pair_metrics(n_vertices=union_genes.size, edges_a=edges_pos_a, edges_b=edges_pos_b)
                    met_neg = zigzag_pair_metrics(n_vertices=union_genes.size, edges_a=edges_neg_a, edges_b=edges_neg_b)
                    pm.append(met_pos["h1_long_mass"])
                    nm.append(met_neg["h1_long_mass"])
                null_degree_relabel[perm_idx] = float(np.mean(np.asarray(pm, dtype=float) - np.asarray(nm, dtype=float)))
                null_rows.append(
                    {
                        "hypothesis_id": "H113",
                        "null_kind": "local_complex_degree_bin_node_relabel",
                        "domain": domain,
                        "seed_tag": H113_SEED,
                        "split_regime": split_regime,
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_degree_relabel[perm_idx]),
                    }
                )

                # Null 3: shuffle labels before creating positive/negative gene pools.
                labels_perm = split_edges["label"].to_numpy(dtype=int).copy()
                labels_perm = rng.permutation(labels_perm)
                split_edges_perm = split_edges.copy()
                split_edges_perm["label_perm"] = labels_perm

                pos_perm = set(
                    pd.concat(
                        [
                            split_edges_perm.loc[split_edges_perm["label_perm"] == 1, "source_idx"],
                            split_edges_perm.loc[split_edges_perm["label_perm"] == 1, "target_idx"],
                        ]
                    ).astype(int)
                )
                neg_perm = set(
                    pd.concat(
                        [
                            split_edges_perm.loc[split_edges_perm["label_perm"] == 0, "source_idx"],
                            split_edges_perm.loc[split_edges_perm["label_perm"] == 0, "target_idx"],
                        ]
                    ).astype(int)
                )

                posp = np.array([gene_to_union[g] for g in sorted(pos_perm) if g in gene_to_union], dtype=int)
                negp = np.array([gene_to_union[g] for g in sorted(neg_perm) if g in gene_to_union], dtype=int)

                if posp.size < 20 or negp.size < 20:
                    null_label[perm_idx] = float("nan")
                else:
                    pm = []
                    nm = []
                    for la, lb in transition_pairs:
                        edges_pos_a = build_knn_edges_for_subset(layer_points[la], posp, H113_KNN)
                        edges_pos_b = build_knn_edges_for_subset(layer_points[lb], posp, H113_KNN)
                        edges_neg_a = build_knn_edges_for_subset(layer_points[la], negp, H113_KNN)
                        edges_neg_b = build_knn_edges_for_subset(layer_points[lb], negp, H113_KNN)
                        met_pos = zigzag_pair_metrics(n_vertices=union_genes.size, edges_a=edges_pos_a, edges_b=edges_pos_b)
                        met_neg = zigzag_pair_metrics(n_vertices=union_genes.size, edges_a=edges_neg_a, edges_b=edges_neg_b)
                        pm.append(met_pos["h1_long_mass"])
                        nm.append(met_neg["h1_long_mass"])
                    null_label[perm_idx] = float(np.mean(np.asarray(pm, dtype=float) - np.asarray(nm, dtype=float)))

                null_rows.append(
                    {
                        "hypothesis_id": "H113",
                        "null_kind": "label_permutation",
                        "domain": domain,
                        "seed_tag": H113_SEED,
                        "split_regime": split_regime,
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_label[perm_idx]),
                    }
                )

            all_null = np.concatenate([null_layer_order, null_degree_relabel, null_label])
            q95 = float(np.nanquantile(all_null, 0.95))

            p_layer = BASE.empirical_upper_tail_p(delta_long_mass, null_layer_order)
            p_relabel = BASE.empirical_upper_tail_p(delta_long_mass, null_degree_relabel)
            p_label = BASE.empirical_upper_tail_p(delta_long_mass, null_label)
            p_best = np.nanmin(np.asarray([p_layer, p_relabel, p_label], dtype=float))

            rows.append(
                {
                    "domain": domain,
                    "seed_tag": H113_SEED,
                    "split_regime": split_regime,
                    "layer": -1,
                    "n_union_genes": int(union_genes.size),
                    "n_pos_genes": int(pos_positions.size),
                    "n_neg_genes": int(neg_positions.size),
                    "mean_long_bar_mass_positive": float(np.mean(pos_mass_arr)),
                    "mean_long_bar_mass_negative": float(np.mean(neg_mass_arr)),
                    "delta_long_bar_mass_positive_minus_negative": float(delta_long_mass),
                    "birth_depth_entropy_positive": float(pos_birth_entropy),
                    "birth_depth_entropy_negative": float(neg_birth_entropy),
                    "birth_depth_entropy_gap": float(entropy_gap),
                    "q95_null_delta_long_bar_mass": float(q95),
                    "null_gap_q95_delta_long_bar_mass": float(delta_long_mass - q95),
                    "p_layer_order_upper": float(p_layer),
                    "p_degree_relabel_upper": float(p_relabel),
                    "p_label_shuffle_upper": float(p_label),
                    "p_best_upper": float(p_best),
                }
            )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime"])
    by_row_path = ITER_DIR / "h113_depth_zigzag_longbar_by_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h113_depth_zigzag_longbar_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_long_bar_mass_positive_minus_negative": float(
                        group["delta_long_bar_mass_positive_minus_negative"].mean()
                    ),
                    "mean_birth_depth_entropy_gap": float(group["birth_depth_entropy_gap"].mean()),
                    "mean_null_gap_q95_delta_long_bar_mass": float(group["null_gap_q95_delta_long_bar_mass"].mean()),
                    "fraction_delta_positive": float((group["delta_long_bar_mass_positive_minus_negative"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_long_bar_mass"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h113_depth_zigzag_longbar_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_long_bar_mass": float(by_row_df["delta_long_bar_mass_positive_minus_negative"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_long_bar_mass_positive_minus_negative"] > 0.0).sum()) if not summary_df.empty else 0,
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95_delta_long_bar_mass"] > 0.0).sum()) if not summary_df.empty else 0,
        "artifact_paths": {
            "by_domain_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h114_id_hysteresis_screen(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H114_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H114_GENE_CAP))
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

            for layer in H114_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                rng = np.random.default_rng(43_410 + domain_index * 1000 + split_index * 100 + layer)
                sample_idx = stratified_index_sample(labels_all, max_n=H114_EDGE_SAMPLE, rng=rng)
                if sample_idx.size < 120:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=22,
                    random_state=43_411 + domain_index * 1000 + split_index * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H114_NEIGHBORS)
                geodesic_w, _ = confidence_weighted_geodesic(geodesic, support_dir)

                _, h70_defect = compute_h70_scores(
                    geodesic=geodesic_w,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=[8, 12, 16],
                )

                id_forward, id_reverse = compute_id_curves(points_pca, k_values=H114_K_VALUES)
                node_bundle = node_hysteresis_bundle(id_forward, id_reverse)
                edge_bundle = edge_features_from_node_bundle(node_bundle, source_local=source_local, target_local=target_local)

                x_base = np.column_stack([h70_defect])
                x_aug = np.column_stack([h70_defect, edge_bundle])

                auc_base = cross_validated_auc(
                    x_base,
                    labels,
                    random_state=43_412 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H114_L1_C,
                    n_splits=H114_CV_SPLITS,
                )
                auc_aug = cross_validated_auc(
                    x_aug,
                    labels,
                    random_state=43_413 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H114_L1_C,
                    n_splits=H114_CV_SPLITS,
                )
                delta_auc = float(auc_aug - auc_base) if np.isfinite(auc_aug) and np.isfinite(auc_base) else float("nan")

                edge_length = geodesic_w[source_local, target_local]
                degree_sum = edge_degree_sum(
                    points=points_pca,
                    n_neighbors=H114_NEIGHBORS,
                    source_local=source_local,
                    target_local=target_local,
                )
                strata = build_edge_strata(edge_length=edge_length, degree_sum=degree_sum, max_len_bins=6, max_deg_bins=4)

                knn_edges = BASE.build_knn_edge_array(points_pca, n_neighbors=H114_NEIGHBORS)
                neighbors = BASE.adjacency_neighbors(points_pca.shape[0], knn_edges)
                node_deg = np.asarray([len(nb) for nb in neighbors], dtype=float)
                node_bins = BASE.degree_bins(node_deg, max_bins=6)

                null_radius = np.empty(H114_NULL_PERM, dtype=float)
                null_neighborhood = np.empty(H114_NULL_PERM, dtype=float)
                null_label = np.empty(H114_NULL_PERM, dtype=float)

                for perm_idx in range(H114_NULL_PERM):
                    # Null 1: radius-order permutation in the forward branch.
                    order = rng.permutation(id_forward.shape[1])
                    node_perm = node_hysteresis_bundle(id_forward[:, order], id_reverse)
                    edge_perm = edge_features_from_node_bundle(node_perm, source_local=source_local, target_local=target_local)
                    x_perm = np.column_stack([h70_defect, edge_perm])
                    auc_perm = cross_validated_auc(
                        x_perm,
                        labels,
                        random_state=43_414 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H114_L1_C,
                        n_splits=H114_CV_SPLITS,
                    )
                    null_radius[perm_idx] = float(auc_perm - auc_base) if np.isfinite(auc_perm) and np.isfinite(auc_base) else float("nan")
                    null_rows.append(
                        {
                            "hypothesis_id": "H114",
                            "null_kind": "radius_order_permutation",
                            "domain": domain,
                            "seed_tag": H114_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_radius[perm_idx]),
                        }
                    )

                    # Null 2: neighborhood reshuffle via node-bin permutation.
                    perm_index = np.arange(node_bundle.shape[0], dtype=int)
                    for b in np.unique(node_bins):
                        idx = np.where(node_bins == b)[0]
                        if idx.size > 1:
                            perm_index[idx] = rng.permutation(idx)
                    node_shuff = node_bundle[perm_index]
                    edge_shuff = edge_features_from_node_bundle(node_shuff, source_local=source_local, target_local=target_local)
                    x_shuff = np.column_stack([h70_defect, edge_shuff])
                    auc_shuff = cross_validated_auc(
                        x_shuff,
                        labels,
                        random_state=43_415 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H114_L1_C,
                        n_splits=H114_CV_SPLITS,
                    )
                    null_neighborhood[perm_idx] = (
                        float(auc_shuff - auc_base) if np.isfinite(auc_shuff) and np.isfinite(auc_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H114",
                            "null_kind": "local_neighborhood_reshuffle",
                            "domain": domain,
                            "seed_tag": H114_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_neighborhood[perm_idx]),
                        }
                    )

                    labels_perm = BASE.stratified_shuffle(labels, strata, rng).astype(int)
                    auc_lp = cross_validated_auc(
                        x_aug,
                        labels_perm,
                        random_state=43_416 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H114_L1_C,
                        n_splits=H114_CV_SPLITS,
                    )
                    auc_lp_base = cross_validated_auc(
                        x_base,
                        labels_perm,
                        random_state=43_417 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H114_L1_C,
                        n_splits=H114_CV_SPLITS,
                    )
                    null_label[perm_idx] = (
                        float(auc_lp - auc_lp_base) if np.isfinite(auc_lp) and np.isfinite(auc_lp_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H114",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H114_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_radius, null_neighborhood, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))

                p_radius = BASE.empirical_upper_tail_p(delta_auc, null_radius)
                p_neigh = BASE.empirical_upper_tail_p(delta_auc, null_neighborhood)
                p_label = BASE.empirical_upper_tail_p(delta_auc, null_label)
                p_best = np.nanmin(np.asarray([p_radius, p_neigh, p_label], dtype=float))

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": H114_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70_defect": float(auc_base),
                        "auc_id_hysteresis_blend": float(auc_aug),
                        "delta_auc_id_hysteresis_minus_h70": float(delta_auc),
                        "mean_node_hysteresis_area": float(np.mean(node_bundle[:, 0])),
                        "mean_node_endpoint_gap": float(np.mean(node_bundle[:, 1])),
                        "mean_node_flip_rate": float(np.mean(node_bundle[:, 2])),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95_delta_auc": float(delta_auc - q95),
                        "p_radius_order_upper": float(p_radius),
                        "p_neighborhood_reshuffle_upper": float(p_neigh),
                        "p_label_shuffle_upper": float(p_label),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h114_id_hysteresis_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h114_id_hysteresis_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_id_hysteresis_minus_h70": float(group["delta_auc_id_hysteresis_minus_h70"].mean()),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_id_hysteresis_minus_h70"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h114_id_hysteresis_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_id_hysteresis_minus_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_id_hysteresis_minus_h70"] > 0.0).sum()) if not summary_df.empty else 0,
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95_delta_auc"] > 0.0).sum()) if not summary_df.empty else 0,
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

    h112_summary = run_h112_semimarkov_biogrammar_rescue(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h113_summary = run_h113_depth_zigzag_longbar()
    h114_summary = run_h114_id_hysteresis_screen(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    summary = {
        "iteration": "iter_0043",
        "h112": h112_summary,
        "h113": h113_summary,
        "h114": h114_summary,
    }

    summary_path = ITER_DIR / "iter0043_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
