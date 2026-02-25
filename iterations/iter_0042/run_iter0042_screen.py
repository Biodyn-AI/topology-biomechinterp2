from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ITER_DIR = Path("iterations/iter_0042")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base")


# H109 / N546: cross-model perturbation Jacobian alignment (multi-seed rescue).
H109_SEEDS = ["seed42_main", "seed43", "seed44"]
H109_LAYERS = [7, 11]
H109_GENE_CAP = 220
H109_MODULE_MIN = 8
H109_MODULE_MAX = 46
H109_MAX_MODULES = 72
H109_PERTURB_PER_KIND = 8
H109_STRENGTHS = [0.25, 0.50, 0.75, 1.00]
H109_NULL_PERM = 64

# H110 / N539: perturbation persistence vineyards (seed42 breadth).
H110_SEED = "seed42_main"
H110_LAYERS = [7, 11]
H110_GENE_CAP = 180
H110_NEIGHBORS = 12
H110_EDGE_SAMPLE = 260
H110_CV_SPLITS = 4
H110_L1_C = 0.22
H110_STRENGTHS = [0.00, 0.25, 0.50, 0.75, 1.00]
H110_NULL_PERM = 32

# H111 / N551: biologically anchored finite-state grammar.
H111_SEED = "seed42_main"
H111_LAYERS = [0, 3, 7, 11]
H111_GENE_CAP = 190
H111_NEIGHBORS = 12
H111_EDGE_SAMPLE = 280
H111_CV_SPLITS = 4
H111_NULL_PERM = 24
H111_TF_BINS = 3
H111_SUPPORT_BINS = 3


def ensure_required_inputs() -> None:
    required_paths = [
        BASE.DOROTHEA_PATH,
        BASE.GENE2GO_PATH,
        BASE.OMNIPATH_INTERACTIONS_PATH,
        BASE.STRING_CACHE_PATH,
    ]

    for domain_runs in BASE.SCGPT_RUNS_BY_DOMAIN.values():
        for seed_tag in H109_SEEDS:
            run_dir = domain_runs[seed_tag]
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
        if H109_MODULE_MIN <= len(uniq) <= H109_MODULE_MAX:
            modules.append((term, uniq))

    modules.sort(key=lambda item: (-len(item[1]), item[0]))
    return modules[:H109_MAX_MODULES]


def module_role_vectors(
    modules: list[tuple[str, list[str]]],
    role_df: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names: list[str] = []
    vectors: list[np.ndarray] = []
    sizes: list[int] = []
    for term, genes in modules:
        genes_use = [g for g in genes if g in role_df.index]
        if len(genes_use) < H109_MODULE_MIN:
            continue
        vec = role_df.loc[genes_use].to_numpy(dtype=float).mean(axis=0)
        names.append(term)
        vectors.append(vec)
        sizes.append(len(genes_use))

    if not vectors:
        return [], np.zeros((0, role_df.shape[1]), dtype=float), np.zeros(0, dtype=float)
    return names, np.vstack(vectors), np.asarray(sizes, dtype=float)


def align_module_vectors(
    modules: list[tuple[str, list[str]]],
    sc_layer_roles: dict[int, pd.DataFrame],
    gf_roles: pd.DataFrame,
) -> tuple[list[str], dict[int, np.ndarray], np.ndarray, np.ndarray]:
    name_map: dict[int, list[str]] = {}
    vec_map: dict[int, np.ndarray] = {}
    size_map: dict[int, np.ndarray] = {}

    for layer, role_df in sc_layer_roles.items():
        names, vecs, sizes = module_role_vectors(modules=modules, role_df=role_df)
        name_map[layer] = names
        vec_map[layer] = vecs
        size_map[layer] = sizes

    names_gf, vecs_gf, sizes_gf = module_role_vectors(modules=modules, role_df=gf_roles)
    if len(names_gf) == 0:
        return [], {}, np.zeros((0, gf_roles.shape[1]), dtype=float), np.zeros(0, dtype=float)

    common = set(names_gf)
    for layer in sc_layer_roles:
        common &= set(name_map[layer])
    common_sorted = sorted(common)

    if len(common_sorted) < H109_MODULE_MIN:
        return [], {}, np.zeros((0, gf_roles.shape[1]), dtype=float), np.zeros(0, dtype=float)

    gf_idx = {name: i for i, name in enumerate(names_gf)}
    gf_out = np.vstack([vecs_gf[gf_idx[name]] for name in common_sorted])
    size_out = np.asarray([sizes_gf[gf_idx[name]] for name in common_sorted], dtype=float)

    sc_out: dict[int, np.ndarray] = {}
    for layer in sc_layer_roles:
        idx = {name: i for i, name in enumerate(name_map[layer])}
        sc_out[layer] = np.vstack([vec_map[layer][idx[name]] for name in common_sorted])

    return common_sorted, sc_out, gf_out, size_out


def zscore_cols(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mu = np.mean(arr, axis=0, keepdims=True)
    sd = np.std(arr, axis=0, keepdims=True)
    sd = np.clip(sd, 1e-8, None)
    return (arr - mu) / sd


def row_norm(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norm, 1e-8, None)


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


def apply_module_perturbation(
    vectors: np.ndarray,
    kind: str,
    a: int,
    b: int,
    strength: float,
) -> np.ndarray:
    out = np.asarray(vectors, dtype=float).copy()
    s = float(np.clip(strength, 0.0, 1.0))

    if kind == "dropout":
        out[a, :] = (1.0 - s) * out[a, :]
    elif kind == "sign_flip":
        out[a, :] = (1.0 - 2.0 * s) * out[a, :]
    elif kind == "rewire_swap":
        va = out[a, :].copy()
        vb = out[b, :].copy()
        out[a, :] = (1.0 - s) * va + s * vb
        out[b, :] = (1.0 - s) * vb + s * va
    else:
        raise ValueError(f"Unknown perturbation kind={kind}")

    return out


def perturb_response_profile(
    vectors: np.ndarray,
    specs: list[tuple[str, int, int]],
    strengths: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    base_t = role_transition_matrix(vectors)
    n_rows = len(specs) * len(strengths)
    scalar = np.zeros(n_rows, dtype=float)
    jac = np.zeros((n_rows, vectors.shape[0]), dtype=float)

    row = 0
    for kind, a, b in specs:
        for strength in strengths:
            pert = apply_module_perturbation(vectors=vectors, kind=kind, a=a, b=b, strength=strength)
            pert_t = role_transition_matrix(pert)
            delta_t = np.abs(pert_t - base_t)
            scalar[row] = float(np.mean(delta_t))
            jac[row] = delta_t.mean(axis=0) + delta_t.mean(axis=1)
            row += 1

    return scalar, jac


def schedule_permute_response(
    scalar: np.ndarray,
    jac: np.ndarray,
    n_specs: int,
    n_strengths: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    scalar_grid = np.asarray(scalar, dtype=float).reshape(n_specs, n_strengths)
    jac_grid = np.asarray(jac, dtype=float).reshape(n_specs, n_strengths, jac.shape[1])

    scalar_perm = scalar_grid.copy()
    jac_perm = jac_grid.copy()
    for spec_idx in range(n_specs):
        order = rng.permutation(n_strengths)
        scalar_perm[spec_idx] = scalar_grid[spec_idx, order]
        jac_perm[spec_idx] = jac_grid[spec_idx, order, :]

    return scalar_perm.reshape(-1), jac_perm.reshape(n_specs * n_strengths, jac.shape[1])


def subspace_mean_cosine(x: np.ndarray, y: np.ndarray, top_k: int = 4) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        return float("nan")

    a_center = a - a.mean(axis=0, keepdims=True)
    b_center = b - b.mean(axis=0, keepdims=True)

    try:
        _, _, vta = np.linalg.svd(a_center, full_matrices=False)
        _, _, vtb = np.linalg.svd(b_center, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan")

    k = min(top_k, vta.shape[0], vtb.shape[0])
    if k < 1:
        return float("nan")

    ua = vta[:k].T
    ub = vtb[:k].T
    m = ua.T @ ub
    svals = np.linalg.svd(m, compute_uv=False)
    return float(np.mean(np.clip(svals, 0.0, 1.0)))


def module_vectors_from_names(
    module_names: list[str],
    module_gene_map: dict[str, list[str]],
    role_df: pd.DataFrame,
) -> np.ndarray:
    vecs: list[np.ndarray] = []
    for name in module_names:
        genes = [g for g in module_gene_map[name] if g in role_df.index]
        if len(genes) < H109_MODULE_MIN:
            return np.zeros((0, role_df.shape[1]), dtype=float)
        vecs.append(role_df.loc[genes].to_numpy(dtype=float).mean(axis=0))
    if not vecs:
        return np.zeros((0, role_df.shape[1]), dtype=float)
    return np.vstack(vecs)


def normalize_module_vectors(vectors: np.ndarray, domain: str) -> np.ndarray:
    x = zscore_cols(vectors)
    # Immune-specific rescue: normalize per-module amplitude to reduce high-variance module dominance.
    if domain == "immune":
        x = row_norm(x)
    return x


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


def sign_states(values: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.ones(arr.shape[0], dtype=int)
    out[arr > threshold] = 2
    out[arr < -threshold] = 0
    return out


def build_support_permutation_with_degree_bins(
    support_sym: np.ndarray,
    node_bins: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = support_sym.shape[0]
    perm = np.arange(n, dtype=int)
    for b in np.unique(node_bins):
        idx = np.where(node_bins == b)[0]
        if idx.size > 1:
            perm[idx] = rng.permutation(idx)
    out = support_sym[np.ix_(perm, perm)]
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 0.0)
    return out


def edge_score_from_support(
    geodesic: np.ndarray,
    support_sym: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    edge_margin: np.ndarray,
) -> np.ndarray:
    geodesic_w = geodesic / (0.35 + np.clip(support_sym, 0.0, 1.0))
    np.fill_diagonal(geodesic_w, 0.0)
    edge_geo = geodesic_w[source_local, target_local]
    edge_sup = support_sym[source_local, target_local]
    return BASE.zscore(-edge_geo) + 0.70 * BASE.zscore(edge_sup) + 0.30 * BASE.zscore(np.abs(edge_margin))


def vineyard_descriptors(traj: np.ndarray, strengths: list[float]) -> np.ndarray:
    arr = np.asarray(traj, dtype=float)
    s = np.asarray(strengths, dtype=float)

    slope = np.zeros(arr.shape[0], dtype=float)
    rough = np.zeros(arr.shape[0], dtype=float)
    curvature = np.zeros(arr.shape[0], dtype=float)
    crossings = np.zeros(arr.shape[0], dtype=float)

    centered_s = s - s.mean()
    denom = float(np.sum(centered_s**2))

    for i in range(arr.shape[0]):
        y = arr[i]
        centered_y = y - y.mean()
        slope[i] = float(np.sum(centered_s * centered_y) / max(1e-8, denom))

        d1 = np.diff(y)
        d2 = np.diff(d1)
        rough[i] = float(np.mean(np.abs(d1))) if d1.size else 0.0
        curvature[i] = float(np.mean(np.abs(d2))) if d2.size else 0.0

        if d1.size >= 2:
            signs = np.sign(d1)
            crossings[i] = float(np.sum(signs[1:] * signs[:-1] < 0))
        else:
            crossings[i] = 0.0

    span = arr.max(axis=1) - arr.min(axis=1)
    level = arr.mean(axis=1)
    end_minus_start = arr[:, -1] - arr[:, 0]

    return np.column_stack([slope, rough, curvature, crossings, span, level, end_minus_start])


def run_h109_cross_model_jacobian_alignment(
    gene2go_upper: dict[str, set[str]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, domain in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.keys()):
        gf_path = BASE.GENEFORMER_EDGE_BY_DOMAIN[domain]
        gf_df = pd.read_csv(gf_path, sep="\t")

        for seed_index, seed_tag in enumerate(H109_SEEDS):
            sc_run = BASE.SCGPT_RUNS_BY_DOMAIN[domain][seed_tag]
            sc_edge_df = pd.read_csv(sc_run / "cycle1_edge_dataset.tsv", sep="\t")
            sc_layer_embeddings = np.load(sc_run / "layer_gene_embeddings.npy", mmap_mode="r")

            top_genes = set(BASE.select_top_genes(sc_edge_df, gene_cap=H109_GENE_CAP))
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
            if len(modules) < 12:
                continue
            module_gene_map = {name: genes for name, genes in modules}

            gf_pos = gf_df.loc[gf_df["label"].astype(int) == 1, ["source", "target"]].copy()
            gf_pos["source"] = gf_pos["source"].astype(str).str.upper()
            gf_pos["target"] = gf_pos["target"].astype(str).str.upper()
            gf_pos = gf_pos.loc[gf_pos["source"].isin(symbols) & gf_pos["target"].isin(symbols)]

            gf_roles = BASE.fit_signatures_geneformer(gf_df=gf_pos.assign(label=1), symbols=symbols)

            sc_layer_roles: dict[int, pd.DataFrame] = {}
            for layer in H109_LAYERS:
                if layer >= sc_layer_embeddings.shape[0]:
                    continue
                points = sc_layer_embeddings[layer, np.asarray(common_indices, dtype=int), :]
                sc_layer_roles[layer] = BASE.fit_signatures_scgpt(
                    layer_points=points,
                    symbols=symbols,
                    random_state=42_900 + domain_index * 1000 + seed_index * 100 + layer,
                    n_neighbors=12,
                )
            if len(sc_layer_roles) != len(H109_LAYERS):
                continue

            module_names, sc_vectors_by_layer, gf_vectors, module_sizes = align_module_vectors(
                modules=modules,
                sc_layer_roles=sc_layer_roles,
                gf_roles=gf_roles,
            )
            if len(module_names) < H109_MODULE_MIN:
                continue

            sc_blend = normalize_module_vectors(
                0.5 * sc_vectors_by_layer[H109_LAYERS[0]] + 0.5 * sc_vectors_by_layer[H109_LAYERS[1]],
                domain=domain,
            )
            gf_base = normalize_module_vectors(gf_vectors, domain=domain)

            map_mat = fit_linear_map(src=gf_base, dst=sc_blend, l2=1e-3)
            gf_aligned = gf_base @ map_mat

            rng = np.random.default_rng(42_901 + domain_index * 10_000 + seed_index * 1000)
            specs = build_perturbation_specs(
                n_modules=sc_blend.shape[0],
                n_each=H109_PERTURB_PER_KIND,
                rng=rng,
            )

            sc_response, sc_jac = perturb_response_profile(
                vectors=sc_blend,
                specs=specs,
                strengths=H109_STRENGTHS,
            )
            gf_response, gf_jac = perturb_response_profile(
                vectors=gf_aligned,
                specs=specs,
                strengths=H109_STRENGTHS,
            )

            observed_rho = spearman_rank_corr(sc_response, gf_response)
            observed_jac_cos = subspace_mean_cosine(sc_jac, gf_jac, top_k=4)

            size_bins = BASE.degree_bins(module_sizes, max_bins=4)
            var_bins = BASE.degree_bins(np.var(gf_aligned, axis=1), max_bins=4)
            module_strata = (size_bins * 8 + var_bins).astype(int)

            null_schedule_rho = np.empty(H109_NULL_PERM, dtype=float)
            null_schedule_jac = np.empty(H109_NULL_PERM, dtype=float)
            null_module_rho = np.empty(H109_NULL_PERM, dtype=float)
            null_module_jac = np.empty(H109_NULL_PERM, dtype=float)
            null_mapping_rho = np.empty(H109_NULL_PERM, dtype=float)
            null_mapping_jac = np.empty(H109_NULL_PERM, dtype=float)

            for perm_idx in range(H109_NULL_PERM):
                n_specs = len(specs)
                n_strengths = len(H109_STRENGTHS)
                gf_resp_sched, gf_jac_sched = schedule_permute_response(
                    scalar=gf_response,
                    jac=gf_jac,
                    n_specs=n_specs,
                    n_strengths=n_strengths,
                    rng=rng,
                )
                null_schedule_rho[perm_idx] = spearman_rank_corr(sc_response, gf_resp_sched)
                null_schedule_jac[perm_idx] = subspace_mean_cosine(sc_jac, gf_jac_sched, top_k=4)
                null_rows.append(
                    {
                        "hypothesis_id": "H109",
                        "null_kind": "perturbation_schedule_permutation",
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": "other",
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_response_value": float(null_schedule_rho[perm_idx]),
                        "null_jacobian_value": float(null_schedule_jac[perm_idx]),
                    }
                )

                perm_idx_vec = np.arange(gf_aligned.shape[0], dtype=int)
                for strata_id in np.unique(module_strata):
                    idx = np.where(module_strata == strata_id)[0]
                    if idx.size > 1:
                        perm_idx_vec[idx] = rng.permutation(idx)
                gf_module_perm = gf_aligned[perm_idx_vec]

                resp_module, jac_module = perturb_response_profile(
                    vectors=gf_module_perm,
                    specs=specs,
                    strengths=H109_STRENGTHS,
                )
                null_module_rho[perm_idx] = spearman_rank_corr(sc_response, resp_module)
                null_module_jac[perm_idx] = subspace_mean_cosine(sc_jac, jac_module, top_k=4)
                null_rows.append(
                    {
                        "hypothesis_id": "H109",
                        "null_kind": "module_size_variance_matched_shuffle",
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": "other",
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_response_value": float(null_module_rho[perm_idx]),
                        "null_jacobian_value": float(null_module_jac[perm_idx]),
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
                    null_mapping_rho[perm_idx] = float("nan")
                    null_mapping_jac[perm_idx] = float("nan")
                else:
                    gf_rand_base = normalize_module_vectors(gf_rand_vecs, domain=domain)
                    map_rand = fit_linear_map(src=gf_rand_base, dst=sc_blend, l2=1e-3)
                    gf_rand_aligned = gf_rand_base @ map_rand
                    resp_rand, jac_rand = perturb_response_profile(
                        vectors=gf_rand_aligned,
                        specs=specs,
                        strengths=H109_STRENGTHS,
                    )
                    null_mapping_rho[perm_idx] = spearman_rank_corr(sc_response, resp_rand)
                    null_mapping_jac[perm_idx] = subspace_mean_cosine(sc_jac, jac_rand, top_k=4)

                null_rows.append(
                    {
                        "hypothesis_id": "H109",
                        "null_kind": "random_gene_mapping_baseline",
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": "other",
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_response_value": float(null_mapping_rho[perm_idx]),
                        "null_jacobian_value": float(null_mapping_jac[perm_idx]),
                    }
                )

            all_null_rho = np.concatenate([null_schedule_rho, null_module_rho, null_mapping_rho])
            all_null_jac = np.concatenate([null_schedule_jac, null_module_jac, null_mapping_jac])

            q95_rho = float(np.nanquantile(all_null_rho, 0.95))
            q95_jac = float(np.nanquantile(all_null_jac, 0.95))

            p_schedule_rho = BASE.empirical_upper_tail_p(observed_rho, null_schedule_rho)
            p_module_rho = BASE.empirical_upper_tail_p(observed_rho, null_module_rho)
            p_mapping_rho = BASE.empirical_upper_tail_p(observed_rho, null_mapping_rho)
            p_best_rho = np.nanmin(np.array([p_schedule_rho, p_module_rho, p_mapping_rho], dtype=float))

            p_schedule_jac = BASE.empirical_upper_tail_p(observed_jac_cos, null_schedule_jac)
            p_module_jac = BASE.empirical_upper_tail_p(observed_jac_cos, null_module_jac)
            p_mapping_jac = BASE.empirical_upper_tail_p(observed_jac_cos, null_mapping_jac)
            p_best_jac = np.nanmin(np.array([p_schedule_jac, p_module_jac, p_mapping_jac], dtype=float))

            rows.append(
                {
                    "domain": domain,
                    "seed_tag": seed_tag,
                    "split_regime": "other",
                    "layer": -1,
                    "n_modules": int(sc_blend.shape[0]),
                    "n_perturbations": int(len(specs) * len(H109_STRENGTHS)),
                    "module_response_rank_spearman": float(observed_rho),
                    "jacobian_subspace_mean_cosine": float(observed_jac_cos),
                    "q95_null_response_concordance": float(q95_rho),
                    "q95_null_jacobian_subspace": float(q95_jac),
                    "null_gap_q95_response_concordance": float(observed_rho - q95_rho),
                    "null_gap_q95_jacobian_subspace": float(observed_jac_cos - q95_jac),
                    "p_best_response_upper": float(p_best_rho),
                    "p_best_jacobian_upper": float(p_best_jac),
                }
            )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["seed_tag", "domain"])
    by_row_path = ITER_DIR / "h109_cross_model_jacobian_alignment_by_seed_domain.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "seed_tag", "domain", "perm_idx"])
    null_path = ITER_DIR / "h109_cross_model_jacobian_alignment_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (seed_tag, domain), group in by_row_df.groupby(["seed_tag", "domain"], sort=True):
            summary_rows.append(
                {
                    "seed_tag": seed_tag,
                    "domain": domain,
                    "n_rows": int(group.shape[0]),
                    "mean_module_response_rank_spearman": float(group["module_response_rank_spearman"].mean()),
                    "mean_jacobian_subspace_mean_cosine": float(group["jacobian_subspace_mean_cosine"].mean()),
                    "mean_null_gap_q95_response_concordance": float(group["null_gap_q95_response_concordance"].mean()),
                    "mean_null_gap_q95_jacobian_subspace": float(group["null_gap_q95_jacobian_subspace"].mean()),
                    "fraction_response_null_gap_positive": float(
                        (group["null_gap_q95_response_concordance"] > 0.0).mean()
                    ),
                    "fraction_jacobian_null_gap_positive": float(
                        (group["null_gap_q95_jacobian_subspace"] > 0.0).mean()
                    ),
                    "fraction_p_best_response_lt_0_05": float((group["p_best_response_upper"] < 0.05).mean()),
                    "fraction_p_best_jacobian_lt_0_05": float((group["p_best_jacobian_upper"] < 0.05).mean()),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["seed_tag", "domain"])
    summary_path = ITER_DIR / "h109_cross_model_jacobian_alignment_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_response_concordance": float(by_row_df["module_response_rank_spearman"].mean())
        if not by_row_df.empty
        else float("nan"),
        "mean_jacobian_subspace_cosine": float(by_row_df["jacobian_subspace_mean_cosine"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_response_null_gap_rows": int((by_row_df["null_gap_q95_response_concordance"] > 0.0).sum())
        if not by_row_df.empty
        else 0,
        "positive_jacobian_null_gap_rows": int((by_row_df["null_gap_q95_jacobian_subspace"] > 0.0).sum())
        if not by_row_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_domain": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h110_persistence_vineyards(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H110_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H110_GENE_CAP))
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
            support_sym = 0.5 * (support_dir + support_dir.T)
            support_sym = np.clip(support_sym, 0.0, 1.0)

            gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
            source_local_all = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
            target_local_all = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
            labels_all = split_edges["label"].to_numpy(dtype=int)

            for layer in H110_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                rng = np.random.default_rng(42_100 + domain_index * 1000 + split_index * 100 + layer)
                sample_idx = stratified_index_sample(labels_all, max_n=H110_EDGE_SAMPLE, rng=rng)
                if sample_idx.size < 120:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=22,
                    random_state=42_101 + domain_index * 1000 + split_index * 100 + layer,
                )

                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H110_NEIGHBORS)
                geodesic_h93, support_sym_chk = confidence_weighted_geodesic(geodesic, support_dir)

                _, h70_base, _ = compute_h70_scores(
                    geodesic=geodesic,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=[8, 12, 16],
                )
                _, h70_h93, _ = compute_h70_scores(
                    geodesic=geodesic_h93,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=[8, 12, 16],
                )

                edge_margin = support_dir[source_local, target_local] - support_dir[target_local, source_local]
                edge_conf = support_sym_chk[source_local, target_local]
                gain_h93 = h70_h93 - h70_base

                x_h93 = np.column_stack([h70_base, h70_h93, gain_h93, edge_margin, edge_conf])
                auc_h93 = cross_validated_auc(
                    x_h93,
                    labels,
                    random_state=42_102 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H110_L1_C,
                    n_splits=H110_CV_SPLITS,
                )

                traj = np.zeros((labels.size, len(H110_STRENGTHS)), dtype=float)
                for s_idx, strength in enumerate(H110_STRENGTHS):
                    support_mix = (1.0 - strength) * support_sym + strength * support_sym_chk
                    traj[:, s_idx] = edge_score_from_support(
                        geodesic=geodesic,
                        support_sym=support_mix,
                        source_local=source_local,
                        target_local=target_local,
                        edge_margin=edge_margin,
                    )

                vine_feat = vineyard_descriptors(traj=traj, strengths=H110_STRENGTHS)
                x_vine = np.column_stack([x_h93, vine_feat, vine_feat[:, 0] * edge_margin])

                auc_vine = cross_validated_auc(
                    x_vine,
                    labels,
                    random_state=42_103 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H110_L1_C,
                    n_splits=H110_CV_SPLITS,
                )

                delta_auc = float(auc_vine - auc_h93) if np.isfinite(auc_vine) and np.isfinite(auc_h93) else float("nan")

                degree_sum = edge_degree_sum(
                    points=points_pca,
                    n_neighbors=H110_NEIGHBORS,
                    source_local=source_local,
                    target_local=target_local,
                )
                edge_length = geodesic_h93[source_local, target_local]
                strata = build_edge_strata(edge_length=edge_length, degree_sum=degree_sum, max_len_bins=6, max_deg_bins=4)

                knn_edges = BASE.build_knn_edge_array(points=points_pca, n_neighbors=H110_NEIGHBORS)
                neighbors = BASE.adjacency_neighbors(points_pca.shape[0], knn_edges)
                node_deg = np.asarray([len(nb) for nb in neighbors], dtype=float)
                node_bins = BASE.degree_bins(node_deg, max_bins=6)

                null_schedule = np.empty(H110_NULL_PERM, dtype=float)
                null_rewire = np.empty(H110_NULL_PERM, dtype=float)
                null_label = np.empty(H110_NULL_PERM, dtype=float)

                for perm_idx in range(H110_NULL_PERM):
                    order = rng.permutation(traj.shape[1])
                    traj_perm = traj[:, order]
                    feat_perm = vineyard_descriptors(traj=traj_perm, strengths=H110_STRENGTHS)
                    x_perm = np.column_stack([x_h93, feat_perm, feat_perm[:, 0] * edge_margin])
                    auc_perm = cross_validated_auc(
                        x_perm,
                        labels,
                        random_state=42_104 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H110_L1_C,
                        n_splits=H110_CV_SPLITS,
                    )
                    null_schedule[perm_idx] = (
                        float(auc_perm - auc_h93) if np.isfinite(auc_perm) and np.isfinite(auc_h93) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H110",
                            "null_kind": "perturbation_schedule_permutation",
                            "domain": domain,
                            "seed_tag": H110_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_schedule[perm_idx]),
                        }
                    )

                    support_perm = build_support_permutation_with_degree_bins(
                        support_sym=support_sym,
                        node_bins=node_bins,
                        rng=rng,
                    )
                    traj_rewire = np.zeros_like(traj)
                    for s_idx, strength in enumerate(H110_STRENGTHS):
                        mix = (1.0 - strength) * support_sym + strength * support_perm
                        traj_rewire[:, s_idx] = edge_score_from_support(
                            geodesic=geodesic,
                            support_sym=mix,
                            source_local=source_local,
                            target_local=target_local,
                            edge_margin=edge_margin,
                        )
                    feat_rewire = vineyard_descriptors(traj=traj_rewire, strengths=H110_STRENGTHS)
                    x_rewire = np.column_stack([x_h93, feat_rewire, feat_rewire[:, 0] * edge_margin])
                    auc_rewire = cross_validated_auc(
                        x_rewire,
                        labels,
                        random_state=42_105 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H110_L1_C,
                        n_splits=H110_CV_SPLITS,
                    )
                    null_rewire[perm_idx] = (
                        float(auc_rewire - auc_h93)
                        if np.isfinite(auc_rewire) and np.isfinite(auc_h93)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H110",
                            "null_kind": "degree_preserving_local_rewiring",
                            "domain": domain,
                            "seed_tag": H110_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_rewire[perm_idx]),
                        }
                    )

                    labels_perm = BASE.stratified_shuffle(labels, strata, rng).astype(int)
                    auc_lp_h93 = cross_validated_auc(
                        x_h93,
                        labels_perm,
                        random_state=42_106 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H110_L1_C,
                        n_splits=H110_CV_SPLITS,
                    )
                    auc_lp_vine = cross_validated_auc(
                        x_vine,
                        labels_perm,
                        random_state=42_107 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H110_L1_C,
                        n_splits=H110_CV_SPLITS,
                    )
                    null_label[perm_idx] = (
                        float(auc_lp_vine - auc_lp_h93)
                        if np.isfinite(auc_lp_h93) and np.isfinite(auc_lp_vine)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H110",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H110_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_schedule, null_rewire, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))

                p_schedule = BASE.empirical_upper_tail_p(delta_auc, null_schedule)
                p_rewire = BASE.empirical_upper_tail_p(delta_auc, null_rewire)
                p_label = BASE.empirical_upper_tail_p(delta_auc, null_label)
                p_best = np.nanmin(np.array([p_schedule, p_rewire, p_label], dtype=float))

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": H110_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h93_backbone": float(auc_h93),
                        "auc_vineyard_features": float(auc_vine),
                        "delta_auc_vineyard_features_minus_h93": float(delta_auc),
                        "mean_vineyard_slope": float(vine_feat[:, 0].mean()),
                        "mean_vineyard_curvature": float(vine_feat[:, 2].mean()),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95_delta_auc": float(delta_auc - q95),
                        "p_schedule_permutation_upper": float(p_schedule),
                        "p_rewire_upper": float(p_rewire),
                        "p_label_shuffle_upper": float(p_label),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h110_persistence_vineyard_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h110_persistence_vineyard_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_vineyard_features_minus_h93": float(
                        group["delta_auc_vineyard_features_minus_h93"].mean()
                    ),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_vineyard_features_minus_h93"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h110_persistence_vineyard_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_vineyard_features_minus_h93"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_vineyard_features_minus_h93"] > 0.0).sum())
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


def run_h111_biologically_anchored_fsm(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H111_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H111_GENE_CAP))
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

            rng = np.random.default_rng(42_400 + domain_index * 1000 + split_index * 100)
            sample_idx = stratified_index_sample(labels_all, max_n=H111_EDGE_SAMPLE, rng=rng)
            if sample_idx.size < 120:
                continue

            source_local = source_local_all[sample_idx]
            target_local = target_local_all[sample_idx]
            labels = labels_all[sample_idx]

            edge_margin = support_dir[source_local, target_local] - support_dir[target_local, source_local]
            sign_state = sign_states(edge_margin)

            tf_source_activity = np.asarray([tf_activity[symbols[idx].upper()] for idx in source_local], dtype=float)
            tf_bin = quantile_bins(tf_source_activity, n_bins=H111_TF_BINS)

            token_layers: list[np.ndarray] = []
            h70_deep = None
            edge_length_deep = None
            degree_sum_deep = None

            for layer in H111_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    token_layers = []
                    break

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=42_401 + domain_index * 1000 + split_index * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H111_NEIGHBORS)
                geodesic_w, support_sym = confidence_weighted_geodesic(geodesic, support_dir)

                edge_support_layer = support_sym[source_local, target_local] / np.clip(
                    geodesic_w[source_local, target_local],
                    1e-8,
                    None,
                )
                support_bin = quantile_bins(edge_support_layer, n_bins=H111_SUPPORT_BINS)

                token = tf_bin * (H111_SUPPORT_BINS * 3) + support_bin * 3 + sign_state
                token_layers.append(token.astype(int))

                if layer == H111_LAYERS[-1]:
                    _, h70_deep, _ = compute_h70_scores(
                        geodesic=geodesic_w,
                        support_dir=support_dir,
                        source_local=source_local,
                        target_local=target_local,
                        triangle_k=[8, 12, 16],
                    )
                    edge_length_deep = geodesic_w[source_local, target_local]
                    degree_sum_deep = edge_degree_sum(
                        points=points_pca,
                        n_neighbors=H111_NEIGHBORS,
                        source_local=source_local,
                        target_local=target_local,
                    )

            if len(token_layers) != len(H111_LAYERS) or h70_deep is None:
                continue

            tokens = np.column_stack(token_layers)

            auc_motif, _ = second_order_sequence_auc(
                tokens=tokens,
                labels=labels,
                random_state=42_402 + domain_index * 1000 + split_index * 100,
                n_splits=H111_CV_SPLITS,
            )
            auc_h70 = BASE.safe_auc(labels, h70_deep)
            delta_auc = float(auc_motif - auc_h70) if np.isfinite(auc_motif) and np.isfinite(auc_h70) else float("nan")

            strata = build_edge_strata(
                edge_length=np.asarray(edge_length_deep, dtype=float),
                degree_sum=np.asarray(degree_sum_deep, dtype=float),
                max_len_bins=6,
                max_deg_bins=4,
            )

            null_state_shuffle = np.empty(H111_NULL_PERM, dtype=float)
            null_layer_order = np.empty(H111_NULL_PERM, dtype=float)
            null_label = np.empty(H111_NULL_PERM, dtype=float)

            for perm_idx in range(H111_NULL_PERM):
                tokens_state = tokens.copy()
                for col in range(tokens_state.shape[1]):
                    tokens_state[:, col] = rng.permutation(tokens_state[:, col])
                auc_state, _ = second_order_sequence_auc(
                    tokens=tokens_state,
                    labels=labels,
                    random_state=42_403 + domain_index * 10_000 + split_index * 1000 + perm_idx,
                    n_splits=H111_CV_SPLITS,
                )
                null_state_shuffle[perm_idx] = (
                    float(auc_state - auc_h70) if np.isfinite(auc_state) and np.isfinite(auc_h70) else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H111",
                        "null_kind": "state_frequency_matched_token_shuffle",
                        "domain": domain,
                        "seed_tag": H111_SEED,
                        "split_regime": split_regime,
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_state_shuffle[perm_idx]),
                    }
                )

                order = rng.permutation(tokens.shape[1])
                tokens_order = tokens[:, order]
                auc_order, _ = second_order_sequence_auc(
                    tokens=tokens_order,
                    labels=labels,
                    random_state=42_404 + domain_index * 10_000 + split_index * 1000 + perm_idx,
                    n_splits=H111_CV_SPLITS,
                )
                null_layer_order[perm_idx] = (
                    float(auc_order - auc_h70) if np.isfinite(auc_order) and np.isfinite(auc_h70) else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H111",
                        "null_kind": "layer_order_permutation",
                        "domain": domain,
                        "seed_tag": H111_SEED,
                        "split_regime": split_regime,
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_layer_order[perm_idx]),
                    }
                )

                labels_perm = BASE.stratified_shuffle(labels, strata, rng).astype(int)
                auc_lp_motif, _ = second_order_sequence_auc(
                    tokens=tokens,
                    labels=labels_perm,
                    random_state=42_405 + domain_index * 10_000 + split_index * 1000 + perm_idx,
                    n_splits=H111_CV_SPLITS,
                )
                auc_lp_h70 = BASE.safe_auc(labels_perm, h70_deep)
                null_label[perm_idx] = (
                    float(auc_lp_motif - auc_lp_h70)
                    if np.isfinite(auc_lp_motif) and np.isfinite(auc_lp_h70)
                    else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H111",
                        "null_kind": "label_permutation",
                        "domain": domain,
                        "seed_tag": H111_SEED,
                        "split_regime": split_regime,
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_label[perm_idx]),
                    }
                )

            all_null = np.concatenate([null_state_shuffle, null_layer_order, null_label])
            q95 = float(np.nanquantile(all_null, 0.95))

            p_state = BASE.empirical_upper_tail_p(delta_auc, null_state_shuffle)
            p_order = BASE.empirical_upper_tail_p(delta_auc, null_layer_order)
            p_label = BASE.empirical_upper_tail_p(delta_auc, null_label)
            p_best = np.nanmin(np.array([p_state, p_order, p_label], dtype=float))

            rows.append(
                {
                    "domain": domain,
                    "seed_tag": H111_SEED,
                    "split_regime": split_regime,
                    "layer": -1,
                    "n_edges_eval": int(labels.size),
                    "n_positive_eval": int(labels.sum()),
                    "auc_h70_deep_layer": float(auc_h70),
                    "auc_bio_anchored_fsm": float(auc_motif),
                    "delta_auc_biofsm_minus_h70": float(delta_auc),
                    "q95_null_delta_auc": float(q95),
                    "null_gap_q95_delta_auc": float(delta_auc - q95),
                    "p_state_shuffle_upper": float(p_state),
                    "p_layer_order_upper": float(p_order),
                    "p_label_shuffle_upper": float(p_label),
                    "p_best_upper": float(p_best),
                }
            )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime"])
    by_row_path = ITER_DIR / "h111_bio_anchored_fsm_by_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h111_bio_anchored_fsm_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_biofsm_minus_h70": float(group["delta_auc_biofsm_minus_h70"].mean()),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "fraction_delta_positive": float((group["delta_auc_biofsm_minus_h70"] > 0.0).mean()),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h111_bio_anchored_fsm_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_biofsm_minus_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_biofsm_minus_h70"] > 0.0).sum())
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


def main() -> None:
    ensure_required_inputs()

    dorothea_map = BASE.load_dorothea_score_map()
    omnipath_pairs = BASE.load_omnipath_pairs()
    gene2go_upper = BASE.load_gene2go_upper()
    string_map = BASE.load_string_scores_from_cache(BASE.STRING_CACHE_PATH)

    h109_summary = run_h109_cross_model_jacobian_alignment(gene2go_upper=gene2go_upper)
    h110_summary = run_h110_persistence_vineyards(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h111_summary = run_h111_biologically_anchored_fsm(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    summary = {
        "iteration": "iter_0042",
        "h109": h109_summary,
        "h110": h110_summary,
        "h111": h111_summary,
    }

    summary_path = ITER_DIR / "iter0042_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
