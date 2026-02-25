from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ITER_DIR = Path("iterations/iter_0037")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base")

# H94 / N474-rescue: ontology-stratified weighted filtration refinement.
H94_SEEDS = ["seed42_main", "seed43", "seed44"]
H94_LAYERS = [7, 11]
H94_GENE_CAP = 170
H94_NEIGHBORS = 12
H94_TRIANGLE_K = [8, 12, 16]
H94_EDGE_SAMPLE = 280
H94_CV_SPLITS = 4
H94_L1_C = 0.20
H94_NULL_PERM = 6

# H95 / N467-inspired but graph-topology family: bridge-curvature descriptors.
H95_SEED = "seed42_main"
H95_LAYERS = [0, 3, 7, 11]
H95_GENE_CAP = 170
H95_NEIGHBORS = 12
H95_TRIANGLE_K = [8, 12, 16]
H95_EDGE_SAMPLE = 300
H95_CV_SPLITS = 4
H95_L1_C = 0.25
H95_NULL_PERM = 6

# H96 / N473-rescue rationale: module-level cross-model topology concordance.
H96_SEED = "seed42_main"
H96_LAYERS = [7, 11]
H96_GENE_CAP = 220
H96_MODULE_MIN = 8
H96_MODULE_MAX = 40
H96_MAX_MODULES = 64
H96_NULL_PERM = 128


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


def go_jaccard(src: str, tgt: str, gene2go_upper: dict[str, set[str]]) -> float:
    src_terms = gene2go_upper.get(src.upper(), set())
    tgt_terms = gene2go_upper.get(tgt.upper(), set())
    union = len(src_terms | tgt_terms)
    if union == 0:
        return 0.0
    return float(len(src_terms & tgt_terms) / union)


def stratify_three_bins(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return np.array([], dtype=int)
    q1 = float(np.quantile(x, 1.0 / 3.0))
    q2 = float(np.quantile(x, 2.0 / 3.0))
    out = np.zeros(x.size, dtype=int)
    out[x > q1] = 1
    out[x > q2] = 2
    return out


def build_h94_stratified_matrix(
    h70_base: np.ndarray,
    h70_weighted: np.ndarray,
    edge_margin_signed: np.ndarray,
    edge_confidence: np.ndarray,
    strata: np.ndarray,
) -> np.ndarray:
    # Build blockwise features so each ontology stratum can carry distinct coefficients.
    gain = h70_weighted - h70_base
    cols = [h70_base]
    for k in [0, 1, 2]:
        mask = (strata == k).astype(float)
        cols.append(h70_weighted * mask)
        cols.append(gain * mask)
        cols.append(edge_margin_signed * mask)
        cols.append(edge_confidence * mask)
    return np.column_stack(cols)


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


def degree_preserving_edge_swap(edges: np.ndarray, n_nodes: int, rng: np.random.Generator, attempts: int) -> np.ndarray:
    # Undirected simple-graph edge swaps preserve node degrees while randomizing local topology.
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

        # Apply swap.
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
        if H96_MODULE_MIN <= len(uniq) <= H96_MODULE_MAX:
            modules.append((term, uniq))

    modules.sort(key=lambda t: (-len(t[1]), t[0]))
    return modules[:H96_MAX_MODULES]


def module_density_from_edges(pos_edges: set[tuple[str, str]], module_genes: list[str]) -> float:
    if len(module_genes) < 2:
        return float("nan")
    genes = module_genes
    possible = len(genes) * (len(genes) - 1)
    if possible <= 0:
        return float("nan")
    hit = 0
    for s in genes:
        su = s.upper()
        for t in genes:
            tu = t.upper()
            if su == tu:
                continue
            if (su, tu) in pos_edges:
                hit += 1
    return float(hit / possible)


def run_h94_ontology_stratified_weighted_filtration(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_index, seed_tag in enumerate(H94_SEEDS):
            run_dir = run_map[seed_tag]
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            split_masks = BASE.build_split_masks(edge_df)

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H94_GENE_CAP))
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

                source_symbol_all = split_edges["source"].astype(str).str.upper().to_numpy()
                target_symbol_all = split_edges["target"].astype(str).str.upper().to_numpy()
                go_jaccard_all = np.asarray(
                    [go_jaccard(s, t, gene2go_upper) for s, t in zip(source_symbol_all, target_symbol_all)],
                    dtype=float,
                )

                for layer in H94_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    rng = np.random.default_rng(
                        37_410 + domain_index * 100_000 + seed_index * 10_000 + split_index * 1000 + layer
                    )
                    sample_idx = stratified_index_sample(labels_all, max_n=H94_EDGE_SAMPLE, rng=rng)
                    if sample_idx.size < 120:
                        continue

                    source_local = source_local_all[sample_idx]
                    target_local = target_local_all[sample_idx]
                    labels = labels_all[sample_idx]
                    edge_go = go_jaccard_all[sample_idx]

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = BASE.reduce_points(
                        points,
                        n_components=22,
                        random_state=37_411 + domain_index * 100_000 + seed_index * 10_000 + split_index * 1000 + layer,
                    )
                    geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H94_NEIGHBORS)

                    _, h70_base, _ = compute_h70_scores(
                        geodesic=geodesic,
                        support_dir=support_dir,
                        source_local=source_local,
                        target_local=target_local,
                        triangle_k=H94_TRIANGLE_K,
                    )

                    geodesic_weighted, support_sym = confidence_weighted_geodesic(geodesic, support_dir)
                    _, h70_weighted, _ = compute_h70_scores(
                        geodesic=geodesic_weighted,
                        support_dir=support_dir,
                        source_local=source_local,
                        target_local=target_local,
                        triangle_k=H94_TRIANGLE_K,
                    )

                    edge_confidence = support_sym[source_local, target_local]
                    edge_margin_signed = support_dir[source_local, target_local] - support_dir[target_local, source_local]
                    strata = stratify_three_bins(edge_go)

                    x_global = np.column_stack(
                        [
                            h70_base,
                            h70_weighted,
                            h70_weighted - h70_base,
                            edge_margin_signed,
                            edge_confidence,
                        ]
                    )
                    x_strat = build_h94_stratified_matrix(
                        h70_base=h70_base,
                        h70_weighted=h70_weighted,
                        edge_margin_signed=edge_margin_signed,
                        edge_confidence=edge_confidence,
                        strata=strata,
                    )

                    auc_global = cross_validated_auc(
                        x_global,
                        labels,
                        random_state=37_412 + domain_index * 100_000 + seed_index * 10_000 + split_index * 1000 + layer,
                        penalty="l1",
                        c_value=H94_L1_C,
                        n_splits=H94_CV_SPLITS,
                    )
                    auc_strat = cross_validated_auc(
                        x_strat,
                        labels,
                        random_state=37_413 + domain_index * 100_000 + seed_index * 10_000 + split_index * 1000 + layer,
                        penalty="l1",
                        c_value=H94_L1_C,
                        n_splits=H94_CV_SPLITS,
                    )
                    delta_auc = float(auc_strat - auc_global) if np.isfinite(auc_strat) and np.isfinite(auc_global) else float("nan")

                    gain = h70_weighted - h70_base
                    stratum_means = []
                    for k in [0, 1, 2]:
                        mask = strata == k
                        if np.any(mask):
                            stratum_means.append(float(np.mean(gain[mask])))
                    between_strata_var = float(np.var(stratum_means)) if stratum_means else float("nan")

                    edge_geodesic = geodesic[source_local, target_local]
                    bins = BASE.degree_bins(edge_geodesic, max_bins=6)

                    null_strata = np.empty(H94_NULL_PERM, dtype=float)
                    null_conf = np.empty(H94_NULL_PERM, dtype=float)
                    null_label = np.empty(H94_NULL_PERM, dtype=float)

                    for perm_idx in range(H94_NULL_PERM):
                        strata_perm = BASE.shuffle_within_bins(strata, bins, rng).astype(int)
                        x_strat_perm = build_h94_stratified_matrix(
                            h70_base=h70_base,
                            h70_weighted=h70_weighted,
                            edge_margin_signed=edge_margin_signed,
                            edge_confidence=edge_confidence,
                            strata=strata_perm,
                        )
                        auc_sp = cross_validated_auc(
                            x_strat_perm,
                            labels,
                            random_state=37_414
                            + domain_index * 1_000_000
                            + seed_index * 100_000
                            + split_index * 10_000
                            + layer * 100
                            + perm_idx,
                            penalty="l1",
                            c_value=H94_L1_C,
                            n_splits=H94_CV_SPLITS,
                        )
                        null_strata[perm_idx] = (
                            float(auc_sp - auc_global) if np.isfinite(auc_sp) and np.isfinite(auc_global) else float("nan")
                        )
                        null_rows.append(
                            {
                                "hypothesis_id": "H94",
                                "null_kind": "ontology_stratum_shuffle_within_geodesic_bins",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_value": float(null_strata[perm_idx]),
                            }
                        )

                        conf_perm = BASE.shuffle_within_bins(edge_confidence, bins, rng)
                        margin_perm = BASE.shuffle_within_bins(edge_margin_signed, bins, rng)
                        weighted_perm = BASE.shuffle_within_bins(h70_weighted, bins, rng)
                        x_conf = build_h94_stratified_matrix(
                            h70_base=h70_base,
                            h70_weighted=weighted_perm,
                            edge_margin_signed=margin_perm,
                            edge_confidence=conf_perm,
                            strata=strata,
                        )
                        auc_cp = cross_validated_auc(
                            x_conf,
                            labels,
                            random_state=37_514
                            + domain_index * 1_000_000
                            + seed_index * 100_000
                            + split_index * 10_000
                            + layer * 100
                            + perm_idx,
                            penalty="l1",
                            c_value=H94_L1_C,
                            n_splits=H94_CV_SPLITS,
                        )
                        null_conf[perm_idx] = (
                            float(auc_cp - auc_global) if np.isfinite(auc_cp) and np.isfinite(auc_global) else float("nan")
                        )
                        null_rows.append(
                            {
                                "hypothesis_id": "H94",
                                "null_kind": "confidence_feature_shuffle_within_geodesic_bins",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_value": float(null_conf[perm_idx]),
                            }
                        )

                        labels_perm = BASE.stratified_shuffle(labels, bins, rng).astype(int)
                        auc_gp = cross_validated_auc(
                            x_global,
                            labels_perm,
                            random_state=37_614
                            + domain_index * 1_000_000
                            + seed_index * 100_000
                            + split_index * 10_000
                            + layer * 100
                            + perm_idx,
                            penalty="l1",
                            c_value=H94_L1_C,
                            n_splits=H94_CV_SPLITS,
                        )
                        auc_sp = cross_validated_auc(
                            x_strat,
                            labels_perm,
                            random_state=37_714
                            + domain_index * 1_000_000
                            + seed_index * 100_000
                            + split_index * 10_000
                            + layer * 100
                            + perm_idx,
                            penalty="l1",
                            c_value=H94_L1_C,
                            n_splits=H94_CV_SPLITS,
                        )
                        null_label[perm_idx] = (
                            float(auc_sp - auc_gp) if np.isfinite(auc_sp) and np.isfinite(auc_gp) else float("nan")
                        )
                        null_rows.append(
                            {
                                "hypothesis_id": "H94",
                                "null_kind": "label_permutation",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_value": float(null_label[perm_idx]),
                            }
                        )

                    all_null = np.concatenate([null_strata, null_conf, null_label])
                    q95 = float(np.nanquantile(all_null, 0.95))
                    p_strata = BASE.empirical_upper_tail_p(delta_auc, null_strata)
                    p_conf = BASE.empirical_upper_tail_p(delta_auc, null_conf)
                    p_lab = BASE.empirical_upper_tail_p(delta_auc, null_label)
                    p_best = np.nanmin(np.array([p_strata, p_conf, p_lab], dtype=float))

                    rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "n_edges_eval": int(labels.size),
                            "n_positive_eval": int(labels.sum()),
                            "auc_global_weighted": float(auc_global),
                            "auc_ontology_stratified_weighted": float(auc_strat),
                            "delta_auc_ontology_weighted_minus_global_weighted": float(delta_auc),
                            "mean_go_jaccard": float(np.mean(edge_go)),
                            "between_strata_weighted_gain_variance": float(between_strata_var),
                            "q95_null_delta_auc": float(q95),
                            "null_gap_q95_delta_auc": float(delta_auc - q95),
                            "p_stratum_shuffle_upper": float(p_strata),
                            "p_conf_shuffle_upper": float(p_conf),
                            "p_label_shuffle_upper": float(p_lab),
                            "p_best_upper": float(p_best),
                        }
                    )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h94_ontology_stratified_weighted_filtration_by_seed_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h94_ontology_stratified_weighted_filtration_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_ontology_weighted_minus_global_weighted": float(
                        group["delta_auc_ontology_weighted_minus_global_weighted"].mean()
                    ),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "mean_between_strata_weighted_gain_variance": float(
                        group["between_strata_weighted_gain_variance"].mean()
                    ),
                    "fraction_delta_positive": float(
                        (group["delta_auc_ontology_weighted_minus_global_weighted"] > 0.0).mean()
                    ),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h94_ontology_stratified_weighted_filtration_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_ontology_weighted_minus_global_weighted"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_ontology_weighted_minus_global_weighted"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95_delta_auc"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_split_layer": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h95_graph_bridge_curvature_blend(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H95_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H95_GENE_CAP))
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

            for layer in H95_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                rng = np.random.default_rng(37_420 + domain_index * 1000 + split_index * 100 + layer)
                sample_idx = stratified_index_sample(labels_all, max_n=H95_EDGE_SAMPLE, rng=rng)
                if sample_idx.size < 120:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=22,
                    random_state=37_421 + domain_index * 1000 + split_index * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H95_NEIGHBORS)

                _, h70_defect, _ = compute_h70_scores(
                    geodesic=geodesic,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=H95_TRIANGLE_K,
                )

                knn_edges = BASE.build_knn_edge_array(points_pca, n_neighbors=H95_NEIGHBORS)
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
                    random_state=37_422 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="none",
                    n_splits=H95_CV_SPLITS,
                )
                auc_aug = cross_validated_auc(
                    x_aug,
                    labels,
                    random_state=37_423 + domain_index * 1000 + split_index * 100 + layer,
                    penalty="l1",
                    c_value=H95_L1_C,
                    n_splits=H95_CV_SPLITS,
                )
                delta_auc = float(auc_aug - auc_base) if np.isfinite(auc_aug) and np.isfinite(auc_base) else float("nan")

                edge_geodesic = geodesic[source_local, target_local]
                bins = BASE.degree_bins(edge_geodesic, max_bins=6)

                null_rewire = np.empty(H95_NULL_PERM, dtype=float)
                null_feature = np.empty(H95_NULL_PERM, dtype=float)
                null_label = np.empty(H95_NULL_PERM, dtype=float)

                for perm_idx in range(H95_NULL_PERM):
                    rewired_edges = degree_preserving_edge_swap(
                        edges=knn_edges,
                        n_nodes=points_pca.shape[0],
                        rng=rng,
                        attempts=max(50, int(2 * max(1, knn_edges.shape[0]))),
                    )
                    rewired_neighbors = BASE.adjacency_neighbors(points_pca.shape[0], rewired_edges)
                    desc_rewired = edge_features_from_neighbors(
                        neighbors=rewired_neighbors,
                        source_local=source_local,
                        target_local=target_local,
                        support_dir=support_dir,
                    )
                    auc_rw = cross_validated_auc(
                        np.column_stack([h70_defect, desc_rewired]),
                        labels,
                        random_state=37_424 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H95_L1_C,
                        n_splits=H95_CV_SPLITS,
                    )
                    null_rewire[perm_idx] = (
                        float(auc_rw - auc_base) if np.isfinite(auc_rw) and np.isfinite(auc_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H95",
                            "null_kind": "degree_preserving_edge_swap",
                            "domain": domain,
                            "seed_tag": H95_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_rewire[perm_idx]),
                        }
                    )

                    feat_perm = np.column_stack(
                        [BASE.shuffle_within_bins(descriptors[:, j], bins, rng) for j in range(descriptors.shape[1])]
                    )
                    auc_feat = cross_validated_auc(
                        np.column_stack([h70_defect, feat_perm]),
                        labels,
                        random_state=37_524 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H95_L1_C,
                        n_splits=H95_CV_SPLITS,
                    )
                    null_feature[perm_idx] = (
                        float(auc_feat - auc_base) if np.isfinite(auc_feat) and np.isfinite(auc_base) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H95",
                            "null_kind": "descriptor_shuffle_within_geodesic_bins",
                            "domain": domain,
                            "seed_tag": H95_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_feature[perm_idx]),
                        }
                    )

                    labels_perm = BASE.stratified_shuffle(labels, bins, rng).astype(int)
                    auc_lp_base = cross_validated_auc(
                        x_base,
                        labels_perm,
                        random_state=37_624 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="none",
                        n_splits=H95_CV_SPLITS,
                    )
                    auc_lp_aug = cross_validated_auc(
                        x_aug,
                        labels_perm,
                        random_state=37_724 + domain_index * 10_000 + split_index * 1000 + layer * 10 + perm_idx,
                        penalty="l1",
                        c_value=H95_L1_C,
                        n_splits=H95_CV_SPLITS,
                    )
                    null_label[perm_idx] = (
                        float(auc_lp_aug - auc_lp_base)
                        if np.isfinite(auc_lp_aug) and np.isfinite(auc_lp_base)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H95",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H95_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_rewire, null_feature, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_rewire = BASE.empirical_upper_tail_p(delta_auc, null_rewire)
                p_feature = BASE.empirical_upper_tail_p(delta_auc, null_feature)
                p_lab = BASE.empirical_upper_tail_p(delta_auc, null_label)
                p_best = np.nanmin(np.array([p_rewire, p_feature, p_lab], dtype=float))

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": H95_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70_baseline": float(auc_base),
                        "auc_graph_bridge_curvature_blend": float(auc_aug),
                        "delta_auc_graph_bridge_curvature_minus_h70": float(delta_auc),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95_delta_auc": float(delta_auc - q95),
                        "p_rewire_upper": float(p_rewire),
                        "p_feature_shuffle_upper": float(p_feature),
                        "p_label_shuffle_upper": float(p_lab),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h95_graph_bridge_curvature_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h95_graph_bridge_curvature_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "split_regime": split_regime,
                    "n_rows": int(group.shape[0]),
                    "mean_delta_auc_graph_bridge_curvature_minus_h70": float(
                        group["delta_auc_graph_bridge_curvature_minus_h70"].mean()
                    ),
                    "mean_null_gap_q95_delta_auc": float(group["null_gap_q95_delta_auc"].mean()),
                    "fraction_delta_positive": float(
                        (group["delta_auc_graph_bridge_curvature_minus_h70"] > 0.0).mean()
                    ),
                    "fraction_null_gap_positive": float((group["null_gap_q95_delta_auc"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h95_graph_bridge_curvature_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_graph_bridge_curvature_minus_h70"].mean())
        if not by_row_df.empty
        else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_auc_graph_bridge_curvature_minus_h70"] > 0.0).sum())
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


def run_h96_cross_model_module_topology_concordance(
    gene2go_upper: dict[str, set[str]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, domain in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.keys()):
        sc_run = BASE.SCGPT_RUNS_BY_DOMAIN[domain][H96_SEED]
        gf_path = BASE.GENEFORMER_EDGE_BY_DOMAIN[domain]

        sc_edge_df = pd.read_csv(sc_run / "cycle1_edge_dataset.tsv", sep="\t")
        gf_df = pd.read_csv(gf_path, sep="\t")
        sc_layer_embeddings = np.load(sc_run / "layer_gene_embeddings.npy", mmap_mode="r")

        top_genes = set(BASE.select_top_genes(sc_edge_df, gene_cap=H96_GENE_CAP))
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
        if len(common_indices) < 80:
            continue

        symbols = [idx_to_symbol[g] for g in common_indices]
        symbol_to_local = {sym: i for i, sym in enumerate(symbols)}
        modules = select_go_modules(symbols=symbols, gene2go_upper=gene2go_upper)
        if len(modules) < 8:
            continue

        gf_pos = gf_df.loc[gf_df["label"].astype(int) == 1, ["source", "target"]].copy()
        gf_pos["source"] = gf_pos["source"].astype(str).str.upper()
        gf_pos["target"] = gf_pos["target"].astype(str).str.upper()
        gf_pos = gf_pos.loc[gf_pos["source"].isin(symbol_to_local) & gf_pos["target"].isin(symbol_to_local)]
        gf_pos_edges = set(zip(gf_pos["source"], gf_pos["target"]))

        gf_sig = BASE.fit_signatures_geneformer(gf_df=gf_pos.assign(label=1), symbols=symbols)
        gf_node_score = (
            BASE.zscore(gf_sig["und_deg_norm"].to_numpy(dtype=float))
            + 0.6 * BASE.zscore(gf_sig["clustering"].to_numpy(dtype=float))
            + 0.4 * BASE.zscore(gf_sig["reciprocity"].to_numpy(dtype=float))
        )
        gf_node_score_map = {sym: float(gf_node_score[i]) for i, sym in enumerate(symbols)}

        for layer in H96_LAYERS:
            if layer >= sc_layer_embeddings.shape[0]:
                continue

            points = sc_layer_embeddings[layer, np.asarray(common_indices, dtype=int), :]
            points_pca = BASE.reduce_points(
                points,
                n_components=22,
                random_state=37_430 + domain_index * 100 + layer,
            )
            knn_edges = BASE.build_knn_edge_array(points_pca, n_neighbors=12)
            neighbors = BASE.adjacency_neighbors(points_pca.shape[0], knn_edges)
            sc_degree = np.asarray([len(nb) for nb in neighbors], dtype=float) / max(1, points_pca.shape[0] - 1)
            sc_clust = BASE.local_clustering(neighbors)
            sc_node_score = BASE.zscore(sc_degree) + 0.6 * BASE.zscore(sc_clust)

            edge_set_und = {tuple(sorted((int(u), int(v)))) for u, v in np.asarray(knn_edges, dtype=int)}

            module_sc = []
            module_gf = []
            module_names = []
            for term, module_genes in modules:
                local_idx = [symbol_to_local[g] for g in module_genes if g in symbol_to_local]
                if len(local_idx) < H96_MODULE_MIN:
                    continue

                # scGPT module score combines node topology and within-module edge density.
                sc_mean = float(np.mean(sc_node_score[np.asarray(local_idx, dtype=int)]))
                possible_und = len(local_idx) * (len(local_idx) - 1) / 2.0
                sc_hit = 0
                for i in range(len(local_idx)):
                    for j in range(i + 1, len(local_idx)):
                        e = tuple(sorted((int(local_idx[i]), int(local_idx[j]))))
                        if e in edge_set_und:
                            sc_hit += 1
                sc_density = float(sc_hit / possible_und) if possible_und > 0 else 0.0
                sc_score = sc_mean + 0.5 * BASE.zscore(np.array([sc_density, 0.0]))[0]

                gf_mean = float(np.mean([gf_node_score_map[g] for g in module_genes if g in gf_node_score_map]))
                gf_density = module_density_from_edges(gf_pos_edges, module_genes)
                if not np.isfinite(gf_density):
                    continue
                gf_score = gf_mean + 0.5 * BASE.zscore(np.array([gf_density, 0.0]))[0]

                module_names.append(term)
                module_sc.append(sc_score)
                module_gf.append(gf_score)

            if len(module_sc) < 8:
                continue

            module_sc_arr = np.asarray(module_sc, dtype=float)
            module_gf_arr = np.asarray(module_gf, dtype=float)
            corr = spearman_rank_corr(module_sc_arr, module_gf_arr)

            if not np.isfinite(corr):
                continue

            top_k = max(3, int(round(0.2 * len(module_sc_arr))))
            sc_top = set(np.argsort(-module_sc_arr)[:top_k].tolist())
            gf_top = set(np.argsort(-module_gf_arr)[:top_k].tolist())
            top_jacc = float(len(sc_top & gf_top) / len(sc_top | gf_top)) if (sc_top | gf_top) else 0.0

            rng = np.random.default_rng(37_431 + domain_index * 100 + layer)
            null_corr = np.empty(H96_NULL_PERM, dtype=float)
            null_jacc = np.empty(H96_NULL_PERM, dtype=float)
            for perm_idx in range(H96_NULL_PERM):
                perm = rng.permutation(module_gf_arr.size)
                gf_perm = module_gf_arr[perm]
                corr_perm = spearman_rank_corr(module_sc_arr, gf_perm)
                null_corr[perm_idx] = float(corr_perm) if np.isfinite(corr_perm) else float("nan")

                gf_top_perm = set(np.argsort(-gf_perm)[:top_k].tolist())
                jacc_perm = float(len(sc_top & gf_top_perm) / len(sc_top | gf_top_perm)) if (sc_top | gf_top_perm) else 0.0
                null_jacc[perm_idx] = jacc_perm

                null_rows.append(
                    {
                        "hypothesis_id": "H96",
                        "null_kind": "module_score_permutation",
                        "domain": domain,
                        "seed_tag": H96_SEED,
                        "split_regime": "other",
                        "layer": int(layer),
                        "perm_idx": int(perm_idx),
                        "null_value_corr": float(null_corr[perm_idx]),
                        "null_value_top_jaccard": float(null_jacc[perm_idx]),
                    }
                )

            q95_corr = float(np.nanquantile(null_corr, 0.95))
            p_corr = BASE.empirical_upper_tail_p(corr, null_corr)
            q95_jacc = float(np.nanquantile(null_jacc, 0.95))
            p_jacc = BASE.empirical_upper_tail_p(top_jacc, null_jacc)
            p_best = np.nanmin(np.array([p_corr, p_jacc], dtype=float))

            rows.append(
                {
                    "domain": domain,
                    "seed_tag": H96_SEED,
                    "split_regime": "other",
                    "layer": int(layer),
                    "n_modules": int(module_sc_arr.size),
                    "module_spearman_scgpt_geneformer": float(corr),
                    "top_module_jaccard": float(top_jacc),
                    "q95_null_spearman": float(q95_corr),
                    "null_gap_q95_spearman": float(corr - q95_corr),
                    "q95_null_top_jaccard": float(q95_jacc),
                    "null_gap_q95_top_jaccard": float(top_jacc - q95_jacc),
                    "p_spearman_upper": float(p_corr),
                    "p_top_jaccard_upper": float(p_jacc),
                    "p_best_upper": float(p_best),
                }
            )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "layer"])
    by_row_path = ITER_DIR / "h96_cross_model_module_topology_by_domain_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["domain", "layer", "perm_idx"])
    null_path = ITER_DIR / "h96_cross_model_module_topology_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not by_row_df.empty:
        for domain, group in by_row_df.groupby("domain", sort=True):
            summary_rows.append(
                {
                    "domain": domain,
                    "n_rows": int(group.shape[0]),
                    "mean_module_spearman_scgpt_geneformer": float(group["module_spearman_scgpt_geneformer"].mean()),
                    "mean_top_module_jaccard": float(group["top_module_jaccard"].mean()),
                    "mean_null_gap_q95_spearman": float(group["null_gap_q95_spearman"].mean()),
                    "mean_null_gap_q95_top_jaccard": float(group["null_gap_q95_top_jaccard"].mean()),
                    "fraction_spearman_null_gap_positive": float((group["null_gap_q95_spearman"] > 0.0).mean()),
                    "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                    "combined_fisher_p_best": float(BASE.safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain"])
    summary_path = ITER_DIR / "h96_cross_model_module_topology_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_spearman": float(by_row_df["module_spearman_scgpt_geneformer"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_count": int((summary_df["mean_null_gap_q95_spearman"] > 0.0).sum()) if not summary_df.empty else 0,
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

    h94_summary = run_h94_ontology_stratified_weighted_filtration(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h95_summary = run_h95_graph_bridge_curvature_blend(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h96_summary = run_h96_cross_model_module_topology_concordance(
        gene2go_upper=gene2go_upper,
    )

    summary = {
        "iteration": "iter_0037",
        "h94": h94_summary,
        "h95": h95_summary,
        "h96": h96_summary,
    }
    summary_path = ITER_DIR / "iter0037_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
