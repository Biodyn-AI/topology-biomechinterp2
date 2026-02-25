from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ITER_DIR = Path("iterations/iter_0038")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base")

# H97 / N482-rescue: bridge-curvature with structure-matched rewiring nulls.
H97_SEED = "seed42_main"
H97_LAYERS = [0, 3, 7, 11]
H97_GENE_CAP = 170
H97_NEIGHBORS = 12
H97_TRIANGLE_K = [8, 12, 16]
H97_EDGE_SAMPLE = 300
H97_CV_SPLITS = 4
H97_L1_C = 0.25
H97_NULL_PERM = 18
H97_LENGTH_BINS = 5
H97_BRIDGE_BINS = 4

# H98 / N486: multi-radius ID heterogeneity entropy screen.
H98_SEED = "seed42_main"
H98_LAYERS = [0, 3, 7, 11]
H98_GENE_CAP = 170
H98_NEIGHBORS = 12
H98_TRIANGLE_K = [8, 12, 16]
H98_EDGE_SAMPLE = 300
H98_CV_SPLITS = 4
H98_L1_C = 0.20
H98_NULL_PERM = 18
H98_ID_K = [4, 6, 8, 10, 12]

# H99 / N487: cross-model module role-graph alignment structural reset.
H99_SEED = "seed42_main"
H99_LAYERS = [7, 11]
H99_GENE_CAP = 220
H99_MODULE_MIN = 8
H99_MODULE_MAX = 42
H99_MAX_MODULES = 64
H99_NULL_PERM = 64


def ensure_required_inputs() -> None:
    required_paths = [
        BASE.DOROTHEA_PATH,
        BASE.GENE2GO_PATH,
        BASE.OMNIPATH_INTERACTIONS_PATH,
        BASE.STRING_CACHE_PATH,
    ]
    for run_map in BASE.SCGPT_RUNS_BY_DOMAIN.values():
        run_dir = run_map[H97_SEED]
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


def edge_features_from_neighbors(
    neighbors: list[set[int]],
    source_local: np.ndarray,
    target_local: np.ndarray,
    support_dir: np.ndarray,
) -> np.ndarray:
    n = len(neighbors)
    degree = np.asarray([len(nb) for nb in neighbors], dtype=float)
    clustering = BASE.local_clustering(neighbors)

    edge_forman = np.zeros(source_local.size, dtype=float)
    edge_bridge = np.zeros(source_local.size, dtype=float)
    edge_jaccard = np.zeros(source_local.size, dtype=float)
    edge_curv_gap = np.zeros(source_local.size, dtype=float)
    edge_cycle_proxy = np.zeros(source_local.size, dtype=float)

    for i in range(source_local.size):
        u = int(source_local[i])
        v = int(target_local[i])
        if u < 0 or v < 0 or u >= n or v >= n or u == v:
            continue
        nu = neighbors[u]
        nv = neighbors[v]
        inter = len(nu & nv)
        union = len(nu | nv)
        jacc = float(inter / union) if union > 0 else 0.0
        edge_jaccard[i] = jacc
        edge_bridge[i] = 1.0 - jacc

        edge_forman[i] = 4.0 - degree[u] - degree[v]
        edge_curv_gap[i] = abs((1.0 - clustering[u]) - (1.0 - clustering[v]))

        denom = max(1, min(int(degree[u]), int(degree[v])) - 1)
        edge_cycle_proxy[i] = float(inter / denom)

    edge_support = support_dir[source_local, target_local]
    edge_margin = np.abs(support_dir[source_local, target_local] - support_dir[target_local, source_local])

    return np.column_stack(
        [
            edge_forman,
            edge_bridge,
            edge_jaccard,
            edge_curv_gap,
            edge_cycle_proxy,
            edge_support,
            edge_margin,
        ]
    )


def degree_preserving_edge_swap(edges: np.ndarray, rng: np.random.Generator, attempts: int) -> np.ndarray:
    if edges.size == 0:
        return edges.copy()

    edge_set = set()
    for u, v in np.asarray(edges, dtype=int):
        a, b = sorted((int(u), int(v)))
        if a != b:
            edge_set.add((a, b))

    edge_list = list(edge_set)
    m = len(edge_list)
    if m < 2:
        return np.array(sorted(edge_set), dtype=int)

    for _ in range(max(1, attempts)):
        i, j = rng.choice(m, size=2, replace=False)
        a, b = edge_list[i]
        c, d = edge_list[j]

        if len({a, b, c, d}) < 4:
            continue

        cand1 = tuple(sorted((a, d)))
        cand2 = tuple(sorted((c, b)))
        if cand1[0] == cand1[1] or cand2[0] == cand2[1]:
            continue
        if cand1 == cand2:
            continue
        if cand1 in edge_set or cand2 in edge_set:
            continue

        edge_set.remove((a, b))
        edge_set.remove((c, d))
        edge_set.add(cand1)
        edge_set.add(cand2)
        edge_list[i] = cand1
        edge_list[j] = cand2

    out = np.array(sorted(edge_set), dtype=int)
    if out.size == 0:
        return np.zeros((0, 2), dtype=int)
    return out


def pair_bin_matrix(values: np.ndarray, max_bins: int) -> np.ndarray:
    n = values.shape[0]
    upper_i, upper_j = np.triu_indices(n, k=1)
    bins = BASE.degree_bins(values[upper_i, upper_j], max_bins=max_bins)
    out = np.zeros((n, n), dtype=int)
    out[upper_i, upper_j] = bins
    out[upper_j, upper_i] = bins
    return out


def pair_bridge_proxy_matrix(neighbors: list[set[int]]) -> np.ndarray:
    n = len(neighbors)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        ni = neighbors[i]
        for j in range(i + 1, n):
            nj = neighbors[j]
            union = len(ni | nj)
            inter = len(ni & nj)
            jacc = float(inter / union) if union > 0 else 0.0
            out[i, j] = 1.0 - jacc
            out[j, i] = out[i, j]
    return out


def constrained_degree_swap(
    edges: np.ndarray,
    strata_matrix: np.ndarray,
    rng: np.random.Generator,
    attempts: int,
) -> np.ndarray:
    if edges.size == 0:
        return edges.copy()

    edge_set = set()
    for u, v in np.asarray(edges, dtype=int):
        a, b = sorted((int(u), int(v)))
        if a != b:
            edge_set.add((a, b))

    edge_list = list(edge_set)
    m = len(edge_list)
    if m < 2:
        return np.array(sorted(edge_set), dtype=int)

    for _ in range(max(1, attempts)):
        i, j = rng.choice(m, size=2, replace=False)
        a, b = edge_list[i]
        c, d = edge_list[j]
        if len({a, b, c, d}) < 4:
            continue

        old_labels = sorted([int(strata_matrix[a, b]), int(strata_matrix[c, d])])
        proposals = [
            (tuple(sorted((a, d))), tuple(sorted((c, b)))),
            (tuple(sorted((a, c))), tuple(sorted((b, d)))),
        ]
        if rng.random() < 0.5:
            proposals = [proposals[1], proposals[0]]

        accepted = False
        for cand1, cand2 in proposals:
            if cand1[0] == cand1[1] or cand2[0] == cand2[1]:
                continue
            if cand1 == cand2:
                continue
            if cand1 in edge_set or cand2 in edge_set:
                continue

            new_labels = sorted([int(strata_matrix[cand1[0], cand1[1]]), int(strata_matrix[cand2[0], cand2[1]])])
            if new_labels != old_labels:
                continue

            edge_set.remove((a, b))
            edge_set.remove((c, d))
            edge_set.add(cand1)
            edge_set.add(cand2)
            edge_list[i] = cand1
            edge_list[j] = cand2
            accepted = True
            break

        if not accepted:
            continue

    out = np.array(sorted(edge_set), dtype=int)
    if out.size == 0:
        return np.zeros((0, 2), dtype=int)
    return out


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
        if H99_MODULE_MIN <= len(uniq) <= H99_MODULE_MAX:
            modules.append((term, uniq))

    modules.sort(key=lambda item: (-len(item[1]), item[0]))
    return modules[:H99_MAX_MODULES]


def module_role_vectors(
    modules: list[tuple[str, list[str]]],
    role_df: pd.DataFrame,
) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    vectors: list[np.ndarray] = []
    for term, genes in modules:
        genes_use = [g for g in genes if g in role_df.index]
        if len(genes_use) < H99_MODULE_MIN:
            continue
        vec = role_df.loc[genes_use].to_numpy(dtype=float).mean(axis=0)
        names.append(term)
        vectors.append(vec)

    if not vectors:
        return [], np.zeros((0, role_df.shape[1]), dtype=float)
    return names, np.vstack(vectors)


def zscore_cols(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mu = np.mean(arr, axis=0, keepdims=True)
    sd = np.std(arr, axis=0, keepdims=True)
    sd = np.clip(sd, 1e-8, None)
    return (arr - mu) / sd


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


def orthogonal_alignment_rmse(sc_vectors: np.ndarray, gf_vectors: np.ndarray) -> float:
    x = zscore_cols(sc_vectors)
    y = zscore_cols(gf_vectors)
    m = y.T @ x
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    q = u @ vt
    y_map = y @ q
    return float(np.sqrt(np.mean((y_map - x) ** 2)))


def build_module_size_preserving_permutation(
    symbols: list[str],
    modules: list[tuple[str, list[str]]],
    rng: np.random.Generator,
) -> list[tuple[str, list[str]]]:
    sizes = [len(genes) for _, genes in modules]
    perm_symbols = list(np.asarray(symbols)[rng.permutation(len(symbols))])
    out: list[tuple[str, list[str]]] = []
    cursor = 0
    for (term, _), size in zip(modules, sizes):
        chunk = perm_symbols[cursor : cursor + size]
        cursor += size
        out.append((term, sorted(chunk)))
    return out


def compute_node_id_matrix(points: np.ndarray, k_values: list[int]) -> np.ndarray:
    max_k = int(max(k_values))
    nbrs = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean")
    nbrs.fit(points)
    dist, _ = nbrs.kneighbors(points)
    local = np.clip(dist[:, 1:], 1e-8, None)

    id_rows: list[np.ndarray] = []
    for k in k_values:
        local_k = local[:, : int(k)]
        rk = np.clip(local_k[:, -1], 1e-8, None)
        logs = np.log(np.clip(rk[:, None] / np.clip(local_k[:, :-1], 1e-8, None), 1.0 + 1e-8, None))
        denom = np.sum(logs, axis=1)
        id_est = (local_k.shape[1] - 1) / np.clip(denom, 1e-8, None)
        id_rows.append(np.clip(id_est, 0.1, 200.0))
    return np.column_stack(id_rows)


def node_id_feature_bundle(id_matrix: np.ndarray) -> dict[str, np.ndarray]:
    mat = np.asarray(id_matrix, dtype=float)
    mat = np.clip(mat, 1e-8, None)
    n_r = mat.shape[1]

    probs = mat / np.clip(mat.sum(axis=1, keepdims=True), 1e-8, None)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-8, None)), axis=1) / np.log(max(2, n_r))

    weights = np.linspace(0.6, 1.4, num=n_r, dtype=float)
    weighted = mat * weights[None, :]
    probs_w = weighted / np.clip(weighted.sum(axis=1, keepdims=True), 1e-8, None)
    entropy_weighted = -np.sum(probs_w * np.log(np.clip(probs_w, 1e-8, None)), axis=1) / np.log(max(2, n_r))

    dispersion = np.std(mat, axis=1)
    slope = mat[:, -1] - mat[:, 0]
    level = np.mean(mat, axis=1)
    span = np.max(mat, axis=1) - np.min(mat, axis=1)

    return {
        "entropy": entropy,
        "entropy_weighted": entropy_weighted,
        "dispersion": dispersion,
        "slope": slope,
        "level": level,
        "span": span,
    }


def edge_features_from_node_bundle(
    node_bundle: dict[str, np.ndarray],
    source_local: np.ndarray,
    target_local: np.ndarray,
) -> np.ndarray:
    cols: list[np.ndarray] = []
    keys = ["entropy", "entropy_weighted", "dispersion", "slope", "level", "span"]
    for key in keys:
        values = np.asarray(node_bundle[key], dtype=float)
        cols.append(0.5 * (values[source_local] + values[target_local]))
    for key in ["entropy", "dispersion", "slope", "span"]:
        values = np.asarray(node_bundle[key], dtype=float)
        cols.append(np.abs(values[source_local] - values[target_local]))
    return np.column_stack(cols)


def run_h97_calibrated_bridge_curvature(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H97_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H97_GENE_CAP))
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

            for layer in H97_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                rng = np.random.default_rng(38_700 + domain_index * 1000 + split_index * 100 + layer)
                sample_idx = stratified_index_sample(labels_all, max_n=H97_EDGE_SAMPLE, rng=rng)
                if sample_idx.size < 120:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=22,
                    random_state=38_701 + domain_index * 1000 + split_index * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H97_NEIGHBORS)
                _, h70_defect, _ = compute_h70_scores(
                    geodesic=geodesic,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H97_TRIANGLE_K,
                )

                knn_edges = BASE.build_knn_edge_array(points_pca, n_neighbors=H97_NEIGHBORS)
                neighbors = BASE.adjacency_neighbors(points_pca.shape[0], knn_edges)
                descriptors = edge_features_from_neighbors(
                    neighbors=neighbors,
                    source_local=source_local,
                    target_local=target_local,
                    support_dir=support_dir,
                )

                x_base = h70_defect[:, None]
                x_aug = np.column_stack([h70_defect, descriptors])

                auc_base = cross_validated_auc(
                    x_base,
                    labels,
                    random_state=38_702 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="none",
                    n_splits=H97_CV_SPLITS,
                )
                auc_aug = cross_validated_auc(
                    x_aug,
                    labels,
                    random_state=38_703 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H97_L1_C,
                    n_splits=H97_CV_SPLITS,
                )
                delta_auc = float(auc_aug - auc_base) if np.isfinite(auc_aug) and np.isfinite(auc_base) else float("nan")

                length_bins = pair_bin_matrix(geodesic, max_bins=H97_LENGTH_BINS)
                bridge_bins = pair_bin_matrix(pair_bridge_proxy_matrix(neighbors), max_bins=H97_BRIDGE_BINS)
                length_bridge_bins = length_bins * H97_BRIDGE_BINS + bridge_bins

                edge_geodesic = geodesic[source_local, target_local]
                edge_bins = BASE.degree_bins(edge_geodesic, max_bins=6)

                null_degree = np.empty(H97_NULL_PERM, dtype=float)
                null_len = np.empty(H97_NULL_PERM, dtype=float)
                null_len_bridge = np.empty(H97_NULL_PERM, dtype=float)
                null_label = np.empty(H97_NULL_PERM, dtype=float)

                swap_attempts = max(80, int(2.5 * max(1, knn_edges.shape[0])))
                for perm_idx in range(H97_NULL_PERM):
                    rw_deg = degree_preserving_edge_swap(knn_edges, rng=rng, attempts=swap_attempts)
                    desc_deg = edge_features_from_neighbors(
                        neighbors=BASE.adjacency_neighbors(points_pca.shape[0], rw_deg),
                        source_local=source_local,
                        target_local=target_local,
                        support_dir=support_dir,
                    )
                    auc_deg = cross_validated_auc(
                        np.column_stack([h70_defect, desc_deg]),
                        labels,
                        random_state=38_704 + domain_index * 100_000 + split_index * 10_000 + layer * 100 + perm_idx,
                        penalty="l1",
                        c_value=H97_L1_C,
                        n_splits=H97_CV_SPLITS,
                    )
                    null_degree[perm_idx] = (
                        float(auc_deg - auc_base) if np.isfinite(auc_deg) and np.isfinite(auc_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H97",
                            "null_kind": "degree_preserving_edge_swap",
                            "domain": domain,
                            "seed_tag": H97_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_degree[perm_idx]),
                        }
                    )

                    rw_len = constrained_degree_swap(
                        knn_edges,
                        strata_matrix=length_bins,
                        rng=rng,
                        attempts=swap_attempts,
                    )
                    desc_len = edge_features_from_neighbors(
                        neighbors=BASE.adjacency_neighbors(points_pca.shape[0], rw_len),
                        source_local=source_local,
                        target_local=target_local,
                        support_dir=support_dir,
                    )
                    auc_len = cross_validated_auc(
                        np.column_stack([h70_defect, desc_len]),
                        labels,
                        random_state=38_804 + domain_index * 100_000 + split_index * 10_000 + layer * 100 + perm_idx,
                        penalty="l1",
                        c_value=H97_L1_C,
                        n_splits=H97_CV_SPLITS,
                    )
                    null_len[perm_idx] = (
                        float(auc_len - auc_base) if np.isfinite(auc_len) and np.isfinite(auc_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H97",
                            "null_kind": "degree_plus_length_bin_swap",
                            "domain": domain,
                            "seed_tag": H97_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_len[perm_idx]),
                        }
                    )

                    rw_len_bridge = constrained_degree_swap(
                        knn_edges,
                        strata_matrix=length_bridge_bins,
                        rng=rng,
                        attempts=swap_attempts,
                    )
                    desc_len_bridge = edge_features_from_neighbors(
                        neighbors=BASE.adjacency_neighbors(points_pca.shape[0], rw_len_bridge),
                        source_local=source_local,
                        target_local=target_local,
                        support_dir=support_dir,
                    )
                    auc_len_bridge = cross_validated_auc(
                        np.column_stack([h70_defect, desc_len_bridge]),
                        labels,
                        random_state=38_904 + domain_index * 100_000 + split_index * 10_000 + layer * 100 + perm_idx,
                        penalty="l1",
                        c_value=H97_L1_C,
                        n_splits=H97_CV_SPLITS,
                    )
                    null_len_bridge[perm_idx] = (
                        float(auc_len_bridge - auc_base)
                        if np.isfinite(auc_len_bridge) and np.isfinite(auc_base)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H97",
                            "null_kind": "degree_plus_length_bridge_strata_swap",
                            "domain": domain,
                            "seed_tag": H97_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_len_bridge[perm_idx]),
                        }
                    )

                    labels_perm = BASE.stratified_shuffle(labels, edge_bins, rng).astype(int)
                    auc_lp_base = cross_validated_auc(
                        x_base,
                        labels_perm,
                        random_state=39_004 + domain_index * 100_000 + split_index * 10_000 + layer * 100 + perm_idx,
                        penalty="none",
                        n_splits=H97_CV_SPLITS,
                    )
                    auc_lp_aug = cross_validated_auc(
                        x_aug,
                        labels_perm,
                        random_state=39_104 + domain_index * 100_000 + split_index * 10_000 + layer * 100 + perm_idx,
                        penalty="l1",
                        c_value=H97_L1_C,
                        n_splits=H97_CV_SPLITS,
                    )
                    null_label[perm_idx] = (
                        float(auc_lp_aug - auc_lp_base)
                        if np.isfinite(auc_lp_aug) and np.isfinite(auc_lp_base)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H97",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H97_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_degree, null_len, null_len_bridge, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_deg = BASE.empirical_upper_tail_p(delta_auc, null_degree)
                p_len = BASE.empirical_upper_tail_p(delta_auc, null_len)
                p_len_bridge = BASE.empirical_upper_tail_p(delta_auc, null_len_bridge)
                p_lab = BASE.empirical_upper_tail_p(delta_auc, null_label)
                p_best = np.nanmin(np.array([p_deg, p_len, p_len_bridge, p_lab], dtype=float))

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": H97_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70_baseline": float(auc_base),
                        "auc_graph_bridge_calibrated": float(auc_aug),
                        "delta_auc_graph_bridge_calibrated_minus_h70": float(delta_auc),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95_delta_auc": float(delta_auc - q95),
                        "p_degree_swap_upper": float(p_deg),
                        "p_length_swap_upper": float(p_len),
                        "p_length_bridge_swap_upper": float(p_len_bridge),
                        "p_label_shuffle_upper": float(p_lab),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h97_graph_bridge_calibrated_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h97_graph_bridge_calibrated_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_graph_bridge_calibrated_minus_h70": float(
                        group["delta_auc_graph_bridge_calibrated_minus_h70"].mean()
                    ),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "fraction_delta_positive": float(
                        (group["delta_auc_graph_bridge_calibrated_minus_h70"] > 0.0).mean()
                    ),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h97_graph_bridge_calibrated_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_graph_bridge_calibrated_minus_h70"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_graph_bridge_calibrated_minus_h70"] > 0.0).sum())
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


def run_h98_id_entropy_screen(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H98_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H98_GENE_CAP))
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

            for layer in H98_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                rng = np.random.default_rng(38_800 + domain_index * 1000 + split_index * 100 + layer)
                sample_idx = stratified_index_sample(labels_all, max_n=H98_EDGE_SAMPLE, rng=rng)
                if sample_idx.size < 120:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=22,
                    random_state=38_801 + domain_index * 1000 + split_index * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H98_NEIGHBORS)
                _, h70_defect, _ = compute_h70_scores(
                    geodesic=geodesic,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H98_TRIANGLE_K,
                )

                id_matrix = compute_node_id_matrix(points_pca, k_values=H98_ID_K)
                node_bundle = node_id_feature_bundle(id_matrix)
                edge_id_features = edge_features_from_node_bundle(node_bundle, source_local=source_local, target_local=target_local)

                x_base = h70_defect[:, None]
                x_aug = np.column_stack([h70_defect, edge_id_features])

                auc_base = cross_validated_auc(
                    x_base,
                    labels,
                    random_state=38_802 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="none",
                    n_splits=H98_CV_SPLITS,
                )
                auc_aug = cross_validated_auc(
                    x_aug,
                    labels,
                    random_state=38_803 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H98_L1_C,
                    n_splits=H98_CV_SPLITS,
                )
                delta_auc = float(auc_aug - auc_base) if np.isfinite(auc_aug) and np.isfinite(auc_base) else float("nan")

                knn_edges = BASE.build_knn_edge_array(points_pca, n_neighbors=H98_NEIGHBORS)
                neighbors = BASE.adjacency_neighbors(points_pca.shape[0], knn_edges)
                node_degree = np.asarray([len(nb) for nb in neighbors], dtype=float)
                degree_bins = BASE.degree_bins(node_degree, max_bins=5)

                edge_geodesic = geodesic[source_local, target_local]
                edge_bins = BASE.degree_bins(edge_geodesic, max_bins=6)

                null_radius = np.empty(H98_NULL_PERM, dtype=float)
                null_neighbor = np.empty(H98_NULL_PERM, dtype=float)
                null_label = np.empty(H98_NULL_PERM, dtype=float)

                for perm_idx in range(H98_NULL_PERM):
                    id_perm = id_matrix.copy()
                    for node_idx in range(id_perm.shape[0]):
                        order = rng.permutation(id_perm.shape[1])
                        id_perm[node_idx] = id_perm[node_idx, order]
                    feat_radius = edge_features_from_node_bundle(
                        node_id_feature_bundle(id_perm),
                        source_local=source_local,
                        target_local=target_local,
                    )
                    auc_radius = cross_validated_auc(
                        np.column_stack([h70_defect, feat_radius]),
                        labels,
                        random_state=38_804 + domain_index * 100_000 + split_index * 10_000 + layer * 100 + perm_idx,
                        penalty="l1",
                        c_value=H98_L1_C,
                        n_splits=H98_CV_SPLITS,
                    )
                    null_radius[perm_idx] = (
                        float(auc_radius - auc_base) if np.isfinite(auc_radius) and np.isfinite(auc_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H98",
                            "null_kind": "radius_order_permutation",
                            "domain": domain,
                            "seed_tag": H98_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_radius[perm_idx]),
                        }
                    )

                    perm_index = np.arange(id_matrix.shape[0], dtype=int)
                    for bin_value in np.unique(degree_bins):
                        idx = np.where(degree_bins == bin_value)[0]
                        if idx.size > 1:
                            perm_index[idx] = rng.permutation(idx)
                    bundle_neighbor = {
                        key: np.asarray(value)[perm_index]
                        for key, value in node_bundle.items()
                    }
                    feat_neighbor = edge_features_from_node_bundle(
                        bundle_neighbor,
                        source_local=source_local,
                        target_local=target_local,
                    )
                    auc_neighbor = cross_validated_auc(
                        np.column_stack([h70_defect, feat_neighbor]),
                        labels,
                        random_state=38_904 + domain_index * 100_000 + split_index * 10_000 + layer * 100 + perm_idx,
                        penalty="l1",
                        c_value=H98_L1_C,
                        n_splits=H98_CV_SPLITS,
                    )
                    null_neighbor[perm_idx] = (
                        float(auc_neighbor - auc_base)
                        if np.isfinite(auc_neighbor) and np.isfinite(auc_base)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H98",
                            "null_kind": "neighborhood_assignment_shuffle",
                            "domain": domain,
                            "seed_tag": H98_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_neighbor[perm_idx]),
                        }
                    )

                    labels_perm = BASE.stratified_shuffle(labels, edge_bins, rng).astype(int)
                    auc_lp_base = cross_validated_auc(
                        x_base,
                        labels_perm,
                        random_state=39_004 + domain_index * 100_000 + split_index * 10_000 + layer * 100 + perm_idx,
                        penalty="none",
                        n_splits=H98_CV_SPLITS,
                    )
                    auc_lp_aug = cross_validated_auc(
                        x_aug,
                        labels_perm,
                        random_state=39_104 + domain_index * 100_000 + split_index * 10_000 + layer * 100 + perm_idx,
                        penalty="l1",
                        c_value=H98_L1_C,
                        n_splits=H98_CV_SPLITS,
                    )
                    null_label[perm_idx] = (
                        float(auc_lp_aug - auc_lp_base)
                        if np.isfinite(auc_lp_aug) and np.isfinite(auc_lp_base)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H98",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H98_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_radius, null_neighbor, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_radius = BASE.empirical_upper_tail_p(delta_auc, null_radius)
                p_neighbor = BASE.empirical_upper_tail_p(delta_auc, null_neighbor)
                p_lab = BASE.empirical_upper_tail_p(delta_auc, null_label)
                p_best = np.nanmin(np.array([p_radius, p_neighbor, p_lab], dtype=float))

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": H98_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70_baseline": float(auc_base),
                        "auc_id_entropy_blend": float(auc_aug),
                        "delta_auc_id_entropy_minus_h70": float(delta_auc),
                        "mean_edge_entropy": float(np.mean(edge_id_features[:, 0])),
                        "mean_edge_entropy_weighted": float(np.mean(edge_id_features[:, 1])),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95_delta_auc": float(delta_auc - q95),
                        "p_radius_permutation_upper": float(p_radius),
                        "p_neighbor_shuffle_upper": float(p_neighbor),
                        "p_label_shuffle_upper": float(p_lab),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h98_id_entropy_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h98_id_entropy_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_id_entropy_minus_h70": float(group["delta_auc_id_entropy_minus_h70"].mean()),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_id_entropy_minus_h70"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h98_id_entropy_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_id_entropy_minus_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_id_entropy_minus_h70"] > 0.0).sum())
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


def run_h99_cross_model_role_graph_alignment(
    gene2go_upper: dict[str, set[str]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, domain in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.keys()):
        sc_run = BASE.SCGPT_RUNS_BY_DOMAIN[domain][H99_SEED]
        gf_path = BASE.GENEFORMER_EDGE_BY_DOMAIN[domain]

        sc_edge_df = pd.read_csv(sc_run / "cycle1_edge_dataset.tsv", sep="\t")
        gf_df = pd.read_csv(gf_path, sep="\t")
        sc_layer_embeddings = np.load(sc_run / "layer_gene_embeddings.npy", mmap_mode="r")

        top_genes = set(BASE.select_top_genes(sc_edge_df, gene_cap=H99_GENE_CAP))
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
        rng = np.random.default_rng(38_900 + domain_index * 1000)

        for layer in H99_LAYERS:
            if layer >= sc_layer_embeddings.shape[0]:
                continue

            points = sc_layer_embeddings[layer, np.asarray(common_indices, dtype=int), :]
            sc_roles = BASE.fit_signatures_scgpt(
                layer_points=points,
                symbols=symbols,
                random_state=38_901 + domain_index * 100 + layer,
                n_neighbors=12,
            )

            module_names, sc_vectors = module_role_vectors(modules=modules, role_df=sc_roles)
            _, gf_vectors = module_role_vectors(modules=modules, role_df=gf_roles)
            if len(module_names) < 8:
                continue

            sc_transition = role_transition_matrix(sc_vectors)
            gf_transition = role_transition_matrix(gf_vectors)

            concordance = spearman_rank_corr(upper_triangle_values(sc_transition), upper_triangle_values(gf_transition))
            top_jacc = top_edge_jaccard(sc_transition, gf_transition, frac=0.20)
            align_rmse = orthogonal_alignment_rmse(sc_vectors, gf_vectors)

            if not np.isfinite(concordance):
                continue

            null_module = np.empty(H99_NULL_PERM, dtype=float)
            null_role = np.empty(H99_NULL_PERM, dtype=float)
            null_subspace = np.empty(H99_NULL_PERM, dtype=float)

            for perm_idx in range(H99_NULL_PERM):
                perm_modules = build_module_size_preserving_permutation(symbols=symbols, modules=modules, rng=rng)
                _, sc_perm = module_role_vectors(modules=perm_modules, role_df=sc_roles)
                _, gf_perm = module_role_vectors(modules=perm_modules, role_df=gf_roles)
                if sc_perm.shape[0] < 8 or gf_perm.shape[0] < 8:
                    null_module[perm_idx] = float("nan")
                else:
                    null_module[perm_idx] = spearman_rank_corr(
                        upper_triangle_values(role_transition_matrix(sc_perm)),
                        upper_triangle_values(role_transition_matrix(gf_perm)),
                    )
                null_rows.append(
                    {
                        "hypothesis_id": "H99",
                        "null_kind": "module_membership_permutation",
                        "domain": domain,
                        "seed_tag": H99_SEED,
                        "split_regime": "other",
                        "layer": int(layer),
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_module[perm_idx]),
                    }
                )

                col_perm = rng.permutation(gf_vectors.shape[1])
                gf_role_perm = gf_vectors[:, col_perm]
                null_role[perm_idx] = spearman_rank_corr(
                    upper_triangle_values(sc_transition),
                    upper_triangle_values(role_transition_matrix(gf_role_perm)),
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H99",
                        "null_kind": "role_label_permutation",
                        "domain": domain,
                        "seed_tag": H99_SEED,
                        "split_regime": "other",
                        "layer": int(layer),
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_role[perm_idx]),
                    }
                )

                dim = gf_vectors.shape[1]
                rand = rng.normal(size=(dim, dim))
                q, _ = np.linalg.qr(rand)
                gf_subspace = gf_vectors @ q
                null_subspace[perm_idx] = spearman_rank_corr(
                    upper_triangle_values(sc_transition),
                    upper_triangle_values(role_transition_matrix(gf_subspace)),
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H99",
                        "null_kind": "random_subspace_rotation",
                        "domain": domain,
                        "seed_tag": H99_SEED,
                        "split_regime": "other",
                        "layer": int(layer),
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_subspace[perm_idx]),
                    }
                )

            all_null = np.concatenate([null_module, null_role, null_subspace])
            q95 = float(np.nanquantile(all_null, 0.95))
            p_module = BASE.empirical_upper_tail_p(concordance, null_module)
            p_role = BASE.empirical_upper_tail_p(concordance, null_role)
            p_subspace = BASE.empirical_upper_tail_p(concordance, null_subspace)
            p_best = np.nanmin(np.array([p_module, p_role, p_subspace], dtype=float))

            rows.append(
                {
                    "domain": domain,
                    "seed_tag": H99_SEED,
                    "split_regime": "other",
                    "layer": int(layer),
                    "n_modules": int(sc_vectors.shape[0]),
                    "module_role_graph_concordance": float(concordance),
                    "top_role_graph_jaccard": float(top_jacc),
                    "module_role_alignment_rmse": float(align_rmse),
                    "q95_null_concordance": float(q95),
                    "null_gap_q95_concordance": float(concordance - q95),
                    "p_module_membership_upper": float(p_module),
                    "p_role_label_upper": float(p_role),
                    "p_random_subspace_upper": float(p_subspace),
                    "p_best_upper": float(p_best),
                }
            )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "layer"])
    by_row_path = ITER_DIR / "h99_cross_model_role_graph_by_domain_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "layer", "perm_idx"])
    null_path = ITER_DIR / "h99_cross_model_role_graph_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for domain, group in by_row_df.groupby("domain", sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "n_rows": int(group.shape[0]),
                    "mean_module_role_graph_concordance": float(group["module_role_graph_concordance"].mean()),
                    "mean_top_role_graph_jaccard": float(group["top_role_graph_jaccard"].mean()),
                    "mean_null_gap_q95_concordance": float(group["null_gap_q95_concordance"].mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_concordance"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain"])
    summary_path = ITER_DIR / "h99_cross_model_role_graph_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_concordance": float(by_row_df["module_role_graph_concordance"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_count": int((summary_df["mean_null_gap_q95_concordance"] > 0.0).sum()) if not summary_df.empty else 0,
        "artifact_paths": {
            "by_domain_layer": str(by_row_path),
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

    h97_summary = run_h97_calibrated_bridge_curvature(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h98_summary = run_h98_id_entropy_screen(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h99_summary = run_h99_cross_model_role_graph_alignment(
        gene2go_upper=gene2go_upper,
    )

    summary = {
        "iteration": "iter_0038",
        "h97": h97_summary,
        "h98": h98_summary,
        "h99": h99_summary,
    }
    summary_path = ITER_DIR / "iter0038_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
