from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ITER_DIR = Path("iterations/iter_0041")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base")


# H106 / N538: STRING triad-closure weighted filtration rescue over H93 backbone.
H106_SEED = "seed42_main"
H106_LAYERS = [7, 11]
H106_GENE_CAP = 170
H106_NEIGHBORS = 12
H106_TRIANGLE_K = [8, 12, 16]
H106_EDGE_SAMPLE = 280
H106_CV_SPLITS = 4
H106_L1_C = 0.22
H106_NULL_PERM = 24
H106_TRIAD_ALPHA = 0.45

# H107 / N537: finite-state descriptor motif screen with second-order sequence model.
H107_SEED = "seed42_main"
H107_LAYERS = [0, 3, 7, 11]
H107_GENE_CAP = 180
H107_NEIGHBORS = 12
H107_TRIANGLE_K = [8, 12, 16]
H107_EDGE_SAMPLE = 260
H107_CV_SPLITS = 4
H107_NULL_PERM = 20
H107_TOKEN_BINS = 3

# H108 / N531: cross-model perturbation-response alignment rescue.
H108_SEED = "seed42_main"
H108_LAYERS = [7, 11]
H108_GENE_CAP = 220
H108_MODULE_MIN = 8
H108_MODULE_MAX = 42
H108_MAX_MODULES = 64
H108_PERTURB_PER_KIND = 8
H108_NULL_PERM = 96


def ensure_required_inputs() -> None:
    required_paths = [
        BASE.DOROTHEA_PATH,
        BASE.GENE2GO_PATH,
        BASE.OMNIPATH_INTERACTIONS_PATH,
        BASE.STRING_CACHE_PATH,
    ]
    for run_map in BASE.SCGPT_RUNS_BY_DOMAIN.values():
        run_dir = run_map[H106_SEED]
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


def edge_degree_sum(points: np.ndarray, n_neighbors: int, source_local: np.ndarray, target_local: np.ndarray) -> np.ndarray:
    knn_edges = BASE.build_knn_edge_array(points=points, n_neighbors=n_neighbors)
    neighbors = BASE.adjacency_neighbors(points.shape[0], knn_edges)
    deg = np.asarray([len(nb) for nb in neighbors], dtype=float)
    return deg[source_local] + deg[target_local]


def build_edge_strata(edge_length: np.ndarray, degree_sum: np.ndarray, max_len_bins: int, max_deg_bins: int) -> np.ndarray:
    bins_len = BASE.degree_bins(edge_length, max_bins=max_len_bins)
    bins_deg = BASE.degree_bins(degree_sum, max_bins=max_deg_bins)
    return (bins_len * 16 + bins_deg).astype(int)


def spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    a = pd.Series(np.asarray(x, dtype=float)).rank(method="average")
    b = pd.Series(np.asarray(y, dtype=float)).rank(method="average")
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.corr(b, method="pearson"))


def quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    ranks = pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).to_numpy(dtype=float)
    bins = np.minimum((ranks * n_bins).astype(int), n_bins - 1)
    return bins.astype(int)


def build_string_matrix(symbols: list[str], string_map: dict[tuple[str, str], float]) -> np.ndarray:
    n = len(symbols)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        si = symbols[i]
        for j in range(i + 1, n):
            sj = symbols[j]
            score = float(max(string_map.get((si, sj), 0.0), string_map.get((sj, si), 0.0)))
            out[i, j] = score
            out[j, i] = score
    return out


def triad_closure_matrix(string_matrix: np.ndarray) -> np.ndarray:
    # Triad closure proxy via weighted common-neighbor intensity.
    s = np.asarray(string_matrix, dtype=float)
    if s.size == 0:
        return s
    s = np.clip(s, 0.0, 1.0)
    triad = s @ s.T
    np.fill_diagonal(triad, 0.0)
    upper = triad[np.triu_indices_from(triad, k=1)]
    denom = float(np.nanmax(upper)) if upper.size else 1.0
    if not np.isfinite(denom) or denom <= 1e-8:
        return np.zeros_like(triad)
    triad = np.clip(triad / denom, 0.0, 1.0)
    np.fill_diagonal(triad, 0.0)
    return triad


def build_triad_weighted_geodesic(
    geodesic: np.ndarray,
    support_sym: np.ndarray,
    triad_matrix: np.ndarray,
    triad_alpha: float,
) -> np.ndarray:
    weight = 0.35 + np.clip(support_sym, 0.0, 1.0) + float(triad_alpha) * np.clip(triad_matrix, 0.0, 1.0)
    out = geodesic / np.clip(weight, 1e-6, None)
    out = np.asarray(out, dtype=float)
    np.fill_diagonal(out, 0.0)
    return out


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
        if H108_MODULE_MIN <= len(uniq) <= H108_MODULE_MAX:
            modules.append((term, uniq))

    modules.sort(key=lambda item: (-len(item[1]), item[0]))
    return modules[:H108_MAX_MODULES]


def module_role_vectors(
    modules: list[tuple[str, list[str]]],
    role_df: pd.DataFrame,
) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    vectors: list[np.ndarray] = []
    for term, genes in modules:
        genes_use = [g for g in genes if g in role_df.index]
        if len(genes_use) < H108_MODULE_MIN:
            continue
        vec = role_df.loc[genes_use].to_numpy(dtype=float).mean(axis=0)
        names.append(term)
        vectors.append(vec)

    if not vectors:
        return [], np.zeros((0, role_df.shape[1]), dtype=float)
    return names, np.vstack(vectors)


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

    if len(common_sorted) < H108_MODULE_MIN:
        return [], {}, np.zeros((0, gf_roles.shape[1]), dtype=float)

    gf_idx = {name: i for i, name in enumerate(names_gf)}
    gf_out = np.vstack([vecs_gf[gf_idx[name]] for name in common_sorted])

    sc_out: dict[int, np.ndarray] = {}
    for layer in sc_layer_roles:
        idx = {name: i for i, name in enumerate(name_map[layer])}
        sc_out[layer] = np.vstack([vec_map[layer][idx[name]] for name in common_sorted])

    return common_sorted, sc_out, gf_out


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


def fit_linear_map(src: np.ndarray, dst: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    x = np.asarray(src, dtype=float)
    y = np.asarray(dst, dtype=float)
    lhs = x.T @ x + float(l2) * np.eye(x.shape[1], dtype=float)
    rhs = x.T @ y
    return np.linalg.solve(lhs, rhs)


def build_perturbation_specs(n_modules: int, n_each: int, rng: np.random.Generator) -> list[tuple[str, int, int]]:
    n_eff = min(max(4, n_each), max(1, n_modules - 1))
    idx = np.arange(n_modules, dtype=int)

    # Build three perturbation families: dropout, sign flip, and local rewiring.
    specs: list[tuple[str, int, int]] = []

    order = rng.permutation(idx)
    for i in order[:n_eff]:
        specs.append(("dropout", int(i), -1))

    order = rng.permutation(idx)
    for i in order[:n_eff]:
        specs.append(("sign_flip", int(i), -1))

    for _ in range(n_eff):
        a, b = rng.choice(idx, size=2, replace=False)
        specs.append(("rewire_swap", int(a), int(b)))

    return specs


def perturb_response_vector(vectors: np.ndarray, specs: list[tuple[str, int, int]]) -> np.ndarray:
    base_t = role_transition_matrix(vectors)
    out = np.zeros(len(specs), dtype=float)

    for i, (kind, a, b) in enumerate(specs):
        pert = np.asarray(vectors, dtype=float).copy()
        if kind == "dropout":
            pert[a, :] = 0.0
        elif kind == "sign_flip":
            pert[a, :] = -pert[a, :]
        elif kind == "rewire_swap":
            pert[[a, b], :] = pert[[b, a], :]
        else:
            raise ValueError(f"Unknown perturbation kind={kind}")

        pert_t = role_transition_matrix(pert)
        out[i] = float(np.mean(np.abs(pert_t - base_t)))

    return out


def module_vectors_from_names(
    module_names: list[str],
    module_gene_map: dict[str, list[str]],
    role_df: pd.DataFrame,
) -> np.ndarray:
    vecs: list[np.ndarray] = []
    for name in module_names:
        genes = [g for g in module_gene_map[name] if g in role_df.index]
        if len(genes) < H108_MODULE_MIN:
            return np.zeros((0, role_df.shape[1]), dtype=float)
        vecs.append(role_df.loc[genes].to_numpy(dtype=float).mean(axis=0))
    if not vecs:
        return np.zeros((0, role_df.shape[1]), dtype=float)
    return np.vstack(vecs)


def run_h106_string_triad_weighted_filtration(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H106_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H106_GENE_CAP))
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

            string_local = build_string_matrix(symbols=symbols, string_map=string_map)
            triad_matrix = triad_closure_matrix(string_local)

            gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
            source_local_all = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
            target_local_all = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
            labels_all = split_edges["label"].to_numpy(dtype=int)

            for layer in H106_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                rng = np.random.default_rng(41_610 + domain_index * 1000 + split_index * 100 + layer)
                sample_idx = stratified_index_sample(labels_all, max_n=H106_EDGE_SAMPLE, rng=rng)
                if sample_idx.size < 120:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=22,
                    random_state=41_611 + domain_index * 1000 + split_index * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H106_NEIGHBORS)
                geodesic_h93, support_sym = confidence_weighted_geodesic(geodesic, support_dir)
                geodesic_n538 = build_triad_weighted_geodesic(
                    geodesic=geodesic,
                    support_sym=support_sym,
                    triad_matrix=triad_matrix,
                    triad_alpha=H106_TRIAD_ALPHA,
                )

                _, h70_base, _ = compute_h70_scores(
                    geodesic=geodesic,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H106_TRIANGLE_K,
                )
                _, h70_h93, _ = compute_h70_scores(
                    geodesic=geodesic_h93,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H106_TRIANGLE_K,
                )
                _, h70_n538, _ = compute_h70_scores(
                    geodesic=geodesic_n538,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H106_TRIANGLE_K,
                )

                edge_conf = support_sym[source_local, target_local]
                edge_margin = support_dir[source_local, target_local] - support_dir[target_local, source_local]
                triad_edge = triad_matrix[source_local, target_local]

                gain_h93 = h70_h93 - h70_base
                gain_triad = h70_n538 - h70_h93

                x_h93 = np.column_stack([h70_base, h70_h93, gain_h93, edge_margin, edge_conf])
                x_n538 = np.column_stack(
                    [
                        x_h93,
                        h70_n538,
                        gain_triad,
                        triad_edge,
                        triad_edge * gain_h93,
                    ]
                )

                auc_h93 = cross_validated_auc(
                    x_h93,
                    labels,
                    random_state=41_612 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H106_L1_C,
                    n_splits=H106_CV_SPLITS,
                )
                auc_n538 = cross_validated_auc(
                    x_n538,
                    labels,
                    random_state=41_613 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H106_L1_C,
                    n_splits=H106_CV_SPLITS,
                )
                delta_auc = float(auc_n538 - auc_h93) if np.isfinite(auc_n538) and np.isfinite(auc_h93) else float("nan")

                triad_pos_neg = float(np.mean(triad_edge[labels == 1]) - np.mean(triad_edge[labels == 0]))

                edge_length = geodesic_h93[source_local, target_local]
                degree_sum = edge_degree_sum(
                    points=points_pca,
                    n_neighbors=H106_NEIGHBORS,
                    source_local=source_local,
                    target_local=target_local,
                )
                strata = build_edge_strata(edge_length=edge_length, degree_sum=degree_sum, max_len_bins=6, max_deg_bins=4)

                null_string_shuffle = np.empty(H106_NULL_PERM, dtype=float)
                null_triad_shuffle = np.empty(H106_NULL_PERM, dtype=float)
                null_label = np.empty(H106_NULL_PERM, dtype=float)

                for perm_idx in range(H106_NULL_PERM):
                    triad_edge_shuff = BASE.shuffle_within_bins(triad_edge, strata, rng)
                    gain_triad_shuff = BASE.shuffle_within_bins(gain_triad, strata, rng)
                    h70_n538_shuff = h70_h93 + gain_triad_shuff
                    x_string = np.column_stack(
                        [
                            x_h93,
                            h70_n538_shuff,
                            gain_triad_shuff,
                            triad_edge_shuff,
                            triad_edge_shuff * gain_h93,
                        ]
                    )
                    auc_string = cross_validated_auc(
                        x_string,
                        labels,
                        random_state=41_614 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H106_L1_C,
                        n_splits=H106_CV_SPLITS,
                    )
                    null_string_shuffle[perm_idx] = (
                        float(auc_string - auc_h93) if np.isfinite(auc_string) and np.isfinite(auc_h93) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H106",
                            "null_kind": "string_weight_shuffle_within_length_degree_bins",
                            "domain": domain,
                            "seed_tag": H106_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_string_shuffle[perm_idx]),
                        }
                    )

                    perm_nodes = rng.permutation(triad_matrix.shape[0])
                    triad_perm = triad_matrix[np.ix_(perm_nodes, perm_nodes)]
                    geodesic_perm = build_triad_weighted_geodesic(
                        geodesic=geodesic,
                        support_sym=support_sym,
                        triad_matrix=triad_perm,
                        triad_alpha=H106_TRIAD_ALPHA,
                    )
                    _, h70_perm, _ = compute_h70_scores(
                        geodesic=geodesic_perm,
                        support_dir=support_dir,
                        source_local=source_local,
                        target_local=target_local,
                        triangle_k=H106_TRIANGLE_K,
                    )
                    triad_edge_perm = triad_perm[source_local, target_local]
                    gain_perm = h70_perm - h70_h93
                    x_triad = np.column_stack(
                        [
                            x_h93,
                            h70_perm,
                            gain_perm,
                            triad_edge_perm,
                            triad_edge_perm * gain_h93,
                        ]
                    )
                    auc_triad = cross_validated_auc(
                        x_triad,
                        labels,
                        random_state=41_615 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H106_L1_C,
                        n_splits=H106_CV_SPLITS,
                    )
                    null_triad_shuffle[perm_idx] = (
                        float(auc_triad - auc_h93) if np.isfinite(auc_triad) and np.isfinite(auc_h93) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H106",
                            "null_kind": "triad_closure_node_permutation",
                            "domain": domain,
                            "seed_tag": H106_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_triad_shuffle[perm_idx]),
                        }
                    )

                    labels_perm = BASE.stratified_shuffle(labels, strata, rng).astype(int)
                    auc_lp_h93 = cross_validated_auc(
                        x_h93,
                        labels_perm,
                        random_state=41_616 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H106_L1_C,
                        n_splits=H106_CV_SPLITS,
                    )
                    auc_lp_n538 = cross_validated_auc(
                        x_n538,
                        labels_perm,
                        random_state=41_617 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H106_L1_C,
                        n_splits=H106_CV_SPLITS,
                    )
                    null_label[perm_idx] = (
                        float(auc_lp_n538 - auc_lp_h93)
                        if np.isfinite(auc_lp_h93) and np.isfinite(auc_lp_n538)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H106",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H106_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_string_shuffle, null_triad_shuffle, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_string = BASE.empirical_upper_tail_p(delta_auc, null_string_shuffle)
                p_triad = BASE.empirical_upper_tail_p(delta_auc, null_triad_shuffle)
                p_label = BASE.empirical_upper_tail_p(delta_auc, null_label)
                p_best = np.nanmin(np.array([p_string, p_triad, p_label], dtype=float))

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": H106_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h93_backbone": float(auc_h93),
                        "auc_n538_string_triad_weighted": float(auc_n538),
                        "delta_auc_string_triad_weighted_minus_h93": float(delta_auc),
                        "triad_closure_pos_minus_neg": float(triad_pos_neg),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95_delta_auc": float(delta_auc - q95),
                        "p_string_weight_shuffle_upper": float(p_string),
                        "p_triad_closure_shuffle_upper": float(p_triad),
                        "p_label_shuffle_upper": float(p_label),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h106_string_triad_weighted_filtration_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h106_string_triad_weighted_filtration_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_string_triad_weighted_minus_h93": float(
                        group["delta_auc_string_triad_weighted_minus_h93"].mean()
                    ),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "mean_triad_closure_pos_minus_neg": float(group["triad_closure_pos_minus_neg"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_string_triad_weighted_minus_h93"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h106_string_triad_weighted_filtration_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_string_triad_weighted_minus_h93"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_string_triad_weighted_minus_h93"] > 0.0).sum())
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


def run_h107_finite_state_descriptor_motif(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H107_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H107_GENE_CAP))
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

            rng = np.random.default_rng(41_700 + domain_index * 1000 + split_index * 100)
            sample_idx = stratified_index_sample(labels_all, max_n=H107_EDGE_SAMPLE, rng=rng)
            if sample_idx.size < 120:
                continue

            source_local = source_local_all[sample_idx]
            target_local = target_local_all[sample_idx]
            labels = labels_all[sample_idx]

            token_layers: list[np.ndarray] = []
            h70_deep = None
            edge_length_deep = None
            degree_sum_deep = None

            for layer in H107_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    token_layers = []
                    break

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=41_701 + domain_index * 1000 + split_index * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H107_NEIGHBORS)
                geodesic_w, support_sym = confidence_weighted_geodesic(geodesic, support_dir)

                euclid_edge = np.linalg.norm(points_pca[source_local] - points_pca[target_local], axis=1)
                geo_edge = geodesic_w[source_local, target_local]
                stretch = geo_edge / np.clip(euclid_edge, 1e-8, None)

                _, h70_weighted, tri_bundle = compute_h70_scores(
                    geodesic=geodesic_w,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H107_TRIANGLE_K,
                )

                dispersion = tri_bundle["dispersion_mean"]
                state_a = quantile_bins(h70_weighted, n_bins=H107_TOKEN_BINS)
                state_b = quantile_bins(stretch, n_bins=H107_TOKEN_BINS)
                state_c = quantile_bins(dispersion, n_bins=H107_TOKEN_BINS)
                token = state_a * (H107_TOKEN_BINS**2) + state_b * H107_TOKEN_BINS + state_c
                token_layers.append(token.astype(int))

                if layer == H107_LAYERS[-1]:
                    h70_deep = h70_weighted.copy()
                    edge_length_deep = geo_edge.copy()
                    degree_sum_deep = edge_degree_sum(
                        points=points_pca,
                        n_neighbors=H107_NEIGHBORS,
                        source_local=source_local,
                        target_local=target_local,
                    )

            if len(token_layers) != len(H107_LAYERS) or h70_deep is None:
                continue

            tokens = np.column_stack(token_layers)
            auc_motif, _ = second_order_sequence_auc(
                tokens=tokens,
                labels=labels,
                random_state=41_702 + domain_index * 1000 + split_index * 100,
                n_splits=H107_CV_SPLITS,
            )
            auc_h70 = BASE.safe_auc(labels, h70_deep)
            delta_auc = float(auc_motif - auc_h70) if np.isfinite(auc_motif) and np.isfinite(auc_h70) else float("nan")

            strata = build_edge_strata(
                edge_length=np.asarray(edge_length_deep, dtype=float),
                degree_sum=np.asarray(degree_sum_deep, dtype=float),
                max_len_bins=6,
                max_deg_bins=4,
            )

            null_layer_order = np.empty(H107_NULL_PERM, dtype=float)
            null_token_shuffle = np.empty(H107_NULL_PERM, dtype=float)
            null_label = np.empty(H107_NULL_PERM, dtype=float)

            for perm_idx in range(H107_NULL_PERM):
                order = rng.permutation(tokens.shape[1])
                tokens_order = tokens[:, order]
                auc_order, _ = second_order_sequence_auc(
                    tokens=tokens_order,
                    labels=labels,
                    random_state=41_703 + domain_index * 10_000 + split_index * 1000 + perm_idx,
                    n_splits=H107_CV_SPLITS,
                )
                null_layer_order[perm_idx] = (
                    float(auc_order - auc_h70) if np.isfinite(auc_order) and np.isfinite(auc_h70) else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H107",
                        "null_kind": "layer_order_permutation",
                        "domain": domain,
                        "seed_tag": H107_SEED,
                        "split_regime": split_regime,
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_layer_order[perm_idx]),
                    }
                )

                tokens_shuffle = tokens.copy()
                for col in range(tokens_shuffle.shape[1]):
                    tokens_shuffle[:, col] = rng.permutation(tokens_shuffle[:, col])
                auc_token, _ = second_order_sequence_auc(
                    tokens=tokens_shuffle,
                    labels=labels,
                    random_state=41_704 + domain_index * 10_000 + split_index * 1000 + perm_idx,
                    n_splits=H107_CV_SPLITS,
                )
                null_token_shuffle[perm_idx] = (
                    float(auc_token - auc_h70) if np.isfinite(auc_token) and np.isfinite(auc_h70) else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H107",
                        "null_kind": "token_shuffle_within_layer",
                        "domain": domain,
                        "seed_tag": H107_SEED,
                        "split_regime": split_regime,
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_token_shuffle[perm_idx]),
                    }
                )

                labels_perm = BASE.stratified_shuffle(labels, strata, rng).astype(int)
                auc_lp_motif, _ = second_order_sequence_auc(
                    tokens=tokens,
                    labels=labels_perm,
                    random_state=41_705 + domain_index * 10_000 + split_index * 1000 + perm_idx,
                    n_splits=H107_CV_SPLITS,
                )
                auc_lp_h70 = BASE.safe_auc(labels_perm, h70_deep)
                null_label[perm_idx] = (
                    float(auc_lp_motif - auc_lp_h70)
                    if np.isfinite(auc_lp_motif) and np.isfinite(auc_lp_h70)
                    else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H107",
                        "null_kind": "label_permutation",
                        "domain": domain,
                        "seed_tag": H107_SEED,
                        "split_regime": split_regime,
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_label[perm_idx]),
                    }
                )

            all_null = np.concatenate([null_layer_order, null_token_shuffle, null_label])
            q95 = float(np.nanquantile(all_null, 0.95))
            p_order = BASE.empirical_upper_tail_p(delta_auc, null_layer_order)
            p_token = BASE.empirical_upper_tail_p(delta_auc, null_token_shuffle)
            p_label = BASE.empirical_upper_tail_p(delta_auc, null_label)
            p_best = np.nanmin(np.array([p_order, p_token, p_label], dtype=float))

            rows.append(
                {
                    "domain": domain,
                    "seed_tag": H107_SEED,
                    "split_regime": split_regime,
                    "layer": -1,
                    "n_edges_eval": int(labels.size),
                    "n_positive_eval": int(labels.sum()),
                    "auc_h70_deep_layer": float(auc_h70),
                    "auc_finite_state_motif": float(auc_motif),
                    "delta_auc_dfa_motif_minus_h70": float(delta_auc),
                    "q95_null_delta_auc": float(q95),
                    "null_gap_q95_delta_auc": float(delta_auc - q95),
                    "p_layer_order_upper": float(p_order),
                    "p_token_shuffle_upper": float(p_token),
                    "p_label_shuffle_upper": float(p_label),
                    "p_best_upper": float(p_best),
                }
            )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime"])
    by_row_path = ITER_DIR / "h107_finite_state_descriptor_motif_by_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h107_finite_state_descriptor_motif_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_dfa_motif_minus_h70": float(group["delta_auc_dfa_motif_minus_h70"].mean()),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_dfa_motif_minus_h70"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h107_finite_state_descriptor_motif_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_dfa_motif_minus_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_dfa_motif_minus_h70"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95_delta_auc"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_domain_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h108_cross_model_perturbation_response_alignment(
    gene2go_upper: dict[str, set[str]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, domain in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.keys()):
        sc_run = BASE.SCGPT_RUNS_BY_DOMAIN[domain][H108_SEED]
        gf_path = BASE.GENEFORMER_EDGE_BY_DOMAIN[domain]

        sc_edge_df = pd.read_csv(sc_run / "cycle1_edge_dataset.tsv", sep="\t")
        gf_df = pd.read_csv(gf_path, sep="\t")
        sc_layer_embeddings = np.load(sc_run / "layer_gene_embeddings.npy", mmap_mode="r")

        top_genes = set(BASE.select_top_genes(sc_edge_df, gene_cap=H108_GENE_CAP))
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

        module_gene_map = {name: genes for name, genes in modules}

        gf_pos = gf_df.loc[gf_df["label"].astype(int) == 1, ["source", "target"]].copy()
        gf_pos["source"] = gf_pos["source"].astype(str).str.upper()
        gf_pos["target"] = gf_pos["target"].astype(str).str.upper()
        gf_pos = gf_pos.loc[gf_pos["source"].isin(symbols) & gf_pos["target"].isin(symbols)]

        gf_roles = BASE.fit_signatures_geneformer(gf_df=gf_pos.assign(label=1), symbols=symbols)

        sc_layer_roles: dict[int, pd.DataFrame] = {}
        for layer in H108_LAYERS:
            if layer >= sc_layer_embeddings.shape[0]:
                continue
            points = sc_layer_embeddings[layer, np.asarray(common_indices, dtype=int), :]
            sc_layer_roles[layer] = BASE.fit_signatures_scgpt(
                layer_points=points,
                symbols=symbols,
                random_state=41_800 + domain_index * 100 + layer,
                n_neighbors=12,
            )
        if len(sc_layer_roles) != len(H108_LAYERS):
            continue

        module_names, sc_vectors_by_layer, gf_vectors = align_module_vectors(
            modules=modules,
            sc_layer_roles=sc_layer_roles,
            gf_roles=gf_roles,
        )
        if len(module_names) < H108_MODULE_MIN:
            continue

        sc_blend = zscore_cols(0.5 * sc_vectors_by_layer[H108_LAYERS[0]] + 0.5 * sc_vectors_by_layer[H108_LAYERS[1]])
        gf_base = zscore_cols(gf_vectors)

        map_mat = fit_linear_map(src=gf_base, dst=sc_blend, l2=1e-3)
        gf_aligned = gf_base @ map_mat

        rng = np.random.default_rng(41_801 + domain_index * 1000)
        specs = build_perturbation_specs(
            n_modules=sc_blend.shape[0],
            n_each=H108_PERTURB_PER_KIND,
            rng=rng,
        )

        sc_response = perturb_response_vector(sc_blend, specs=specs)
        gf_response = perturb_response_vector(gf_aligned, specs=specs)
        observed = spearman_rank_corr(sc_response, gf_response)

        null_schedule = np.empty(H108_NULL_PERM, dtype=float)
        null_module_shuffle = np.empty(H108_NULL_PERM, dtype=float)
        null_random_mapping = np.empty(H108_NULL_PERM, dtype=float)

        for perm_idx in range(H108_NULL_PERM):
            schedule_perm = rng.permutation(gf_response)
            null_schedule[perm_idx] = spearman_rank_corr(sc_response, schedule_perm)
            null_rows.append(
                {
                    "hypothesis_id": "H108",
                    "null_kind": "perturbation_schedule_permutation",
                    "domain": domain,
                    "seed_tag": H108_SEED,
                    "split_regime": "other",
                    "layer": -1,
                    "perm_idx": int(perm_idx),
                    "null_value": float(null_schedule[perm_idx]),
                }
            )

            gf_module_perm = gf_aligned[rng.permutation(gf_aligned.shape[0]), :]
            resp_module = perturb_response_vector(gf_module_perm, specs=specs)
            null_module_shuffle[perm_idx] = spearman_rank_corr(sc_response, resp_module)
            null_rows.append(
                {
                    "hypothesis_id": "H108",
                    "null_kind": "module_label_shuffle",
                    "domain": domain,
                    "seed_tag": H108_SEED,
                    "split_regime": "other",
                    "layer": -1,
                    "perm_idx": int(perm_idx),
                    "null_value": float(null_module_shuffle[perm_idx]),
                }
            )

            gf_roles_rand = gf_roles.copy()
            gf_roles_rand.index = rng.permutation(gf_roles_rand.index.to_numpy())
            gf_rand_vecs = module_vectors_from_names(
                module_names=module_names,
                module_gene_map=module_gene_map,
                role_df=gf_roles_rand,
            )
            if gf_rand_vecs.shape[0] != sc_blend.shape[0]:
                null_random_mapping[perm_idx] = float("nan")
            else:
                gf_rand_base = zscore_cols(gf_rand_vecs)
                map_rand = fit_linear_map(src=gf_rand_base, dst=sc_blend, l2=1e-3)
                gf_rand_aligned = gf_rand_base @ map_rand
                resp_rand = perturb_response_vector(gf_rand_aligned, specs=specs)
                null_random_mapping[perm_idx] = spearman_rank_corr(sc_response, resp_rand)

            null_rows.append(
                {
                    "hypothesis_id": "H108",
                    "null_kind": "random_gene_mapping_baseline",
                    "domain": domain,
                    "seed_tag": H108_SEED,
                    "split_regime": "other",
                    "layer": -1,
                    "perm_idx": int(perm_idx),
                    "null_value": float(null_random_mapping[perm_idx]),
                }
            )

        all_null = np.concatenate([null_schedule, null_module_shuffle, null_random_mapping])
        q95 = float(np.nanquantile(all_null, 0.95))
        p_schedule = BASE.empirical_upper_tail_p(observed, null_schedule)
        p_module = BASE.empirical_upper_tail_p(observed, null_module_shuffle)
        p_mapping = BASE.empirical_upper_tail_p(observed, null_random_mapping)
        p_best = np.nanmin(np.array([p_schedule, p_module, p_mapping], dtype=float))

        rows.append(
            {
                "domain": domain,
                "seed_tag": H108_SEED,
                "split_regime": "other",
                "layer": -1,
                "n_modules": int(sc_blend.shape[0]),
                "n_perturbations": int(len(specs)),
                "module_response_rank_spearman": float(observed),
                "mean_sc_response": float(np.mean(sc_response)),
                "mean_gf_response": float(np.mean(gf_response)),
                "response_mean_gap_sc_minus_gf": float(np.mean(sc_response) - np.mean(gf_response)),
                "q95_null_response_concordance": float(q95),
                "null_gap_q95_response_concordance": float(observed - q95),
                "p_schedule_permutation_upper": float(p_schedule),
                "p_module_shuffle_upper": float(p_module),
                "p_random_mapping_upper": float(p_mapping),
                "p_best_upper": float(p_best),
            }
        )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain"])
    by_row_path = ITER_DIR / "h108_cross_model_perturbation_response_by_domain.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "perm_idx"])
    null_path = ITER_DIR / "h108_cross_model_perturbation_response_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for domain, group in by_row_df.groupby("domain", sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "n_rows": int(group.shape[0]),
                    "mean_module_response_rank_spearman": float(group["module_response_rank_spearman"].mean()),
                    "mean_null_gap_q95_response_concordance": float(group["null_gap_q95_response_concordance"].mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_response_concordance"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain"])
    summary_path = ITER_DIR / "h108_cross_model_perturbation_response_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_concordance": float(by_row_df["module_response_rank_spearman"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_count": int((summary_df["mean_null_gap_q95_response_concordance"] > 0.0).sum())
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

    h106_summary = run_h106_string_triad_weighted_filtration(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h107_summary = run_h107_finite_state_descriptor_motif(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h108_summary = run_h108_cross_model_perturbation_response_alignment(
        gene2go_upper=gene2go_upper,
    )

    summary = {
        "iteration": "iter_0041",
        "h106": h106_summary,
        "h107": h107_summary,
        "h108": h108_summary,
    }
    summary_path = ITER_DIR / "iter0041_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
