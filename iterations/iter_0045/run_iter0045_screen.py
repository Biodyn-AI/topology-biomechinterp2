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


ITER_DIR = Path("iterations/iter_0045")
ITER_DIR.mkdir(parents=True, exist_ok=True)

TRRUST_PATH = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
    "single_cell_mechinterp/external/networks/trrust_human.tsv"
)

# H118: multiseed robustness check for module-structure + signed motif interaction.
H118_SEEDS = ["seed42_main", "seed43", "seed44"]
H118_LAYER = 11
H118_GENE_CAP = 190
H118_NEIGHBORS = 12
H118_EDGE_SAMPLE = 280
H118_CV_SPLITS = 4
H118_NULL_PERM = 10

# H119: disagreement-conditioned cross-model transfer.
H119_SEED = "seed42_main"
H119_LAYER = 11
H119_GENE_CAP = 220
H119_EDGE_SAMPLE = 300
H119_CV_SPLITS = 4
H119_NULL_PERM = 12

# H120: geodesic-path curvature drift descriptors.
H120_SEED = "seed42_main"
H120_LAYERS = [7, 11]
H120_GENE_CAP = 170
H120_NEIGHBORS = 12
H120_EDGE_SAMPLE = 240
H120_CV_SPLITS = 4
H120_NULL_PERM = 8


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base")


def ensure_required_inputs() -> None:
    required = [BASE.DOROTHEA_PATH, BASE.GENE2GO_PATH, BASE.OMNIPATH_INTERACTIONS_PATH, BASE.STRING_CACHE_PATH, TRRUST_PATH]
    for domain, run_map in BASE.SCGPT_RUNS_BY_DOMAIN.items():
        for seed_tag in set(H118_SEEDS + [H119_SEED, H120_SEED]):
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


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    num = np.sum(aa * bb, axis=1)
    den = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1)
    return num / np.clip(den, 1e-8, None)


def fit_linear_map(src: np.ndarray, dst: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    x = np.asarray(src, dtype=float)
    y = np.asarray(dst, dtype=float)
    lhs = x.T @ x + float(l2) * np.eye(x.shape[1], dtype=float)
    rhs = x.T @ y
    return np.linalg.solve(lhs, rhs)


def disagreement_gated_features(base_feat: np.ndarray, disagreement: np.ndarray, bins: np.ndarray) -> np.ndarray:
    b = np.asarray(base_feat, dtype=float)
    d = np.asarray(disagreement, dtype=float)
    q = np.asarray(bins, dtype=int)
    mid = (q == 1).astype(float)[:, None]
    high = (q == 2).astype(float)[:, None]
    return np.column_stack([b, d, mid[:, 0], high[:, 0], b * mid, b * high])


def label_propagation_communities(
    neighbors: list[set[int]],
    rng: np.random.Generator,
    max_iter: int = 20,
) -> np.ndarray:
    # Lightweight graph community proxy: asynchronous label propagation on kNN neighborhoods.
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


def run_h118_signed_motif_module_robustness(
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
        for seed_idx, seed_tag in enumerate(H118_SEEDS):
            run_dir = run_map[seed_tag]
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            split_masks = BASE.build_split_masks(edge_df)

            for split_idx, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H118_GENE_CAP))
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

                rng = np.random.default_rng(45_180 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100)
                sample_idx = stratified_index_sample(labels_all, max_n=H118_EDGE_SAMPLE, rng=rng)
                if sample_idx.size < 120:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                if H118_LAYER >= layer_embeddings.shape[0]:
                    continue

                points = layer_embeddings[H118_LAYER, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=45_181 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H118_NEIGHBORS)
                geodesic_w = confidence_weighted_geodesic(geodesic, support_dir)

                h70 = compute_h70_scores(
                    geodesic=geodesic_w,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=[8, 12, 16],
                )

                knn_edges = BASE.build_knn_edge_array(points=points_pca, n_neighbors=H118_NEIGHBORS)
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
                    random_state=45_182 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100,
                    n_splits=H118_CV_SPLITS,
                )
                delta = float(auc_model - auc_h70) if np.isfinite(auc_model) and np.isfinite(auc_h70) else float("nan")

                tf_deg = np.asarray([trrust_tf_out_degree.get(s, 0) for s in src_sym], dtype=float)
                tf_bins = BASE.degree_bins(tf_deg, max_bins=4)
                edge_len = geodesic_w[source_local, target_local]
                deg_sum = edge_degree_sum(points_pca, H118_NEIGHBORS, source_local, target_local)
                edge_strata = build_edge_strata(edge_len, deg_sum, max_len_bins=6, max_deg_bins=4)

                null_comm = np.empty(H118_NULL_PERM, dtype=float)
                null_sign = np.empty(H118_NULL_PERM, dtype=float)
                null_label = np.empty(H118_NULL_PERM, dtype=float)

                for perm_idx in range(H118_NULL_PERM):
                    comm_perm = community_labels[rng.permutation(community_labels.shape[0])]
                    same_comm_perm = (comm_perm[source_local] == comm_perm[target_local]).astype(float)
                    feat_comm = np.column_stack(
                        [
                            h70,
                            same_comm_perm,
                            motif_present,
                            sign_consistent,
                            same_comm_perm * motif_present,
                            same_comm_perm * sign_consistent,
                            h70 * same_comm_perm * sign_consistent,
                        ]
                    )
                    auc_comm = cv_auc_logit(
                        feat_comm,
                        labels,
                        random_state=45_183 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H118_CV_SPLITS,
                    )
                    null_comm[perm_idx] = (
                        float(auc_comm - auc_h70) if np.isfinite(auc_comm) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H118",
                            "null_kind": "community_size_preserving_label_shuffle",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H118_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_comm[perm_idx]),
                        }
                    )

                    sign_perm = permute_within_strata(trrust_sign, tf_bins, rng).astype(int)
                    motif_perm = (sign_perm != 0).astype(float)
                    sign_cons_perm = ((sign_perm * margin_sign) > 0).astype(float)
                    feat_sign = np.column_stack(
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
                    auc_sign = cv_auc_logit(
                        feat_sign,
                        labels,
                        random_state=45_184 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H118_CV_SPLITS,
                    )
                    null_sign[perm_idx] = (
                        float(auc_sign - auc_h70) if np.isfinite(auc_sign) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H118",
                            "null_kind": "trrust_sign_shuffle_within_tf_degree_bin",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H118_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_sign[perm_idx]),
                        }
                    )

                    y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                    auc_lp = cv_auc_logit(
                        feat,
                        y_perm,
                        random_state=45_185 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H118_CV_SPLITS,
                    )
                    auc_h70_lp = BASE.safe_auc(y_perm, h70)
                    null_label[perm_idx] = (
                        float(auc_lp - auc_h70_lp)
                        if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H118",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H118_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_comm, null_sign, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_comm = BASE.empirical_upper_tail_p(delta, null_comm)
                p_sign = BASE.empirical_upper_tail_p(delta, null_sign)
                p_label = BASE.empirical_upper_tail_p(delta, null_label)
                p_best = float(np.nanmin(np.asarray([p_comm, p_sign, p_label], dtype=float)))

                rows.append(
                    {
                        "hypothesis_id": "H118",
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(H118_LAYER),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70": float(auc_h70),
                        "auc_signed_module_model": float(auc_model),
                        "delta_vs_h70": float(delta),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95": float(delta - q95),
                        "p_comm_upper": float(p_comm),
                        "p_sign_upper": float(p_sign),
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
    by_row_path = ITER_DIR / "h118_signed_motif_module_by_seed_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "seed_tag", "domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h118_signed_motif_module_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_df = domain_split_summary(by_row_df)
    summary_path = ITER_DIR / "h118_signed_motif_module_domain_summary.csv"
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


def run_h119_disagreement_conditioned_transfer(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_idx, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H119_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        gf_df = pd.read_csv(BASE.GENEFORMER_EDGE_BY_DOMAIN[domain], sep="\t")

        gf_symbols = set(gf_df["source"].astype(str).str.upper()) | set(gf_df["target"].astype(str).str.upper())
        split_masks = BASE.build_split_masks(edge_df)

        for split_idx, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H119_GENE_CAP))
            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
            ].copy()
            if split_edges["label"].nunique() < 2:
                continue

            idx_symbol = BASE.build_symbol_map(split_edges)
            common_gene_ids = [g for g in sorted(idx_symbol.keys()) if idx_symbol[g].upper() in gf_symbols]
            if len(common_gene_ids) < 120:
                continue

            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(common_gene_ids) & split_edges["target_idx"].isin(common_gene_ids)
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

            rng = np.random.default_rng(45_290 + domain_idx * 10_000 + split_idx * 100)
            sample_idx = stratified_index_sample(labels_all, max_n=H119_EDGE_SAMPLE, rng=rng)
            if sample_idx.size < 120:
                continue

            source_local = source_local_all[sample_idx]
            target_local = target_local_all[sample_idx]
            labels = labels_all[sample_idx]

            if H119_LAYER >= layer_embeddings.shape[0]:
                continue

            points = layer_embeddings[H119_LAYER, edge_gene_indices, :]
            points_pca = BASE.reduce_points(points, n_components=20, random_state=45_291 + domain_idx * 100 + split_idx)

            geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=12)
            geodesic_w = confidence_weighted_geodesic(geodesic, support_dir)
            edge_len = geodesic_w[source_local, target_local]
            deg_sum = edge_degree_sum(points_pca, 12, source_local, target_local)
            edge_strata = build_edge_strata(edge_len, deg_sum, max_len_bins=6, max_deg_bins=4)

            h70 = compute_h70_scores(
                geodesic=geodesic_w,
                support_dir=support_dir,
                source_local=source_local,
                target_local=target_local,
                triangle_k=[8, 12, 16],
            )

            sc_roles_df = BASE.fit_signatures_scgpt(
                layer_points=points,
                symbols=symbols,
                random_state=45_292 + domain_idx * 100 + split_idx,
                n_neighbors=12,
            )

            gf_use = gf_df.copy()
            gf_use["source"] = gf_use["source"].astype(str).str.upper()
            gf_use["target"] = gf_use["target"].astype(str).str.upper()
            gf_use = gf_use.loc[gf_use["source"].isin(symbols) & gf_use["target"].isin(symbols)].copy()
            if gf_use.empty or gf_use["label"].nunique() < 2:
                continue

            gf_roles_df = BASE.fit_signatures_geneformer(gf_df=gf_use, symbols=symbols)
            if gf_roles_df.shape[0] != sc_roles_df.shape[0]:
                continue

            sc_arr = sc_roles_df.loc[symbols].to_numpy(dtype=float)
            gf_arr = gf_roles_df.loc[symbols].to_numpy(dtype=float)
            map_mat = fit_linear_map(src=gf_arr, dst=sc_arr, l2=1e-3)
            gf_aligned = gf_arr @ map_mat

            sc_src = sc_arr[source_local]
            sc_tgt = sc_arr[target_local]
            gf_src = gf_aligned[source_local]
            gf_tgt = gf_aligned[target_local]

            sc_cos = row_cosine(sc_src, sc_tgt)
            gf_cos = row_cosine(gf_src, gf_tgt)
            sc_l2 = np.linalg.norm(sc_src - sc_tgt, axis=1)
            gf_l2 = np.linalg.norm(gf_src - gf_tgt, axis=1)

            base_feat = np.column_stack([sc_cos, gf_cos, BASE.zscore(-sc_l2), BASE.zscore(-gf_l2), h70])
            disagreement = np.abs(BASE.zscore(sc_cos) - BASE.zscore(gf_cos)) + np.abs(
                BASE.zscore(sc_l2) - BASE.zscore(gf_l2)
            )
            dis_bins = quantile_bins(disagreement, n_bins=3)
            gate_feat = disagreement_gated_features(base_feat=base_feat, disagreement=disagreement, bins=dis_bins)

            auc_base = cv_auc_logit(
                base_feat,
                labels,
                random_state=45_293 + domain_idx * 100 + split_idx,
                n_splits=H119_CV_SPLITS,
            )
            auc_gate = cv_auc_logit(
                gate_feat,
                labels,
                random_state=45_294 + domain_idx * 100 + split_idx,
                n_splits=H119_CV_SPLITS,
            )
            delta = float(auc_gate - auc_base) if np.isfinite(auc_gate) and np.isfinite(auc_base) else float("nan")

            null_dis = np.empty(H119_NULL_PERM, dtype=float)
            null_map = np.empty(H119_NULL_PERM, dtype=float)
            null_label = np.empty(H119_NULL_PERM, dtype=float)

            gf_row_bins = BASE.degree_bins(np.linalg.norm(gf_aligned, axis=1), max_bins=5)

            for perm_idx in range(H119_NULL_PERM):
                dis_perm = permute_within_strata(dis_bins, edge_strata, rng)
                gate_perm = disagreement_gated_features(base_feat=base_feat, disagreement=disagreement, bins=dis_perm)
                auc_perm = cv_auc_logit(
                    gate_perm,
                    labels,
                    random_state=45_295 + domain_idx * 100_000 + split_idx * 10_000 + perm_idx,
                    n_splits=H119_CV_SPLITS,
                )
                null_dis[perm_idx] = (
                    float(auc_perm - auc_base) if np.isfinite(auc_perm) and np.isfinite(auc_base) else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H119",
                        "null_kind": "disagreement_bin_permutation_within_edge_strata",
                        "domain": domain,
                        "seed_tag": H119_SEED,
                        "split_regime": split_regime,
                        "layer": int(H119_LAYER),
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_dis[perm_idx]),
                    }
                )

                perm_idx_vec = np.arange(gf_aligned.shape[0], dtype=int)
                for g in np.unique(gf_row_bins):
                    idx = np.where(gf_row_bins == g)[0]
                    if idx.size > 1:
                        perm_idx_vec[idx] = rng.permutation(idx)
                gf_rand = gf_aligned[perm_idx_vec]

                gr_src = gf_rand[source_local]
                gr_tgt = gf_rand[target_local]
                gf_cos_rand = row_cosine(gr_src, gr_tgt)
                gf_l2_rand = np.linalg.norm(gr_src - gr_tgt, axis=1)
                base_rand = np.column_stack([sc_cos, gf_cos_rand, BASE.zscore(-sc_l2), BASE.zscore(-gf_l2_rand), h70])
                dis_rand = np.abs(BASE.zscore(sc_cos) - BASE.zscore(gf_cos_rand)) + np.abs(
                    BASE.zscore(sc_l2) - BASE.zscore(gf_l2_rand)
                )
                bins_rand = quantile_bins(dis_rand, n_bins=3)
                gate_rand = disagreement_gated_features(base_feat=base_rand, disagreement=dis_rand, bins=bins_rand)

                auc_base_rand = cv_auc_logit(
                    base_rand,
                    labels,
                    random_state=45_296 + domain_idx * 100_000 + split_idx * 10_000 + perm_idx,
                    n_splits=H119_CV_SPLITS,
                )
                auc_gate_rand = cv_auc_logit(
                    gate_rand,
                    labels,
                    random_state=45_297 + domain_idx * 100_000 + split_idx * 10_000 + perm_idx,
                    n_splits=H119_CV_SPLITS,
                )
                null_map[perm_idx] = (
                    float(auc_gate_rand - auc_base_rand)
                    if np.isfinite(auc_gate_rand) and np.isfinite(auc_base_rand)
                    else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H119",
                        "null_kind": "random_gene_mapping_baseline",
                        "domain": domain,
                        "seed_tag": H119_SEED,
                        "split_regime": split_regime,
                        "layer": int(H119_LAYER),
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_map[perm_idx]),
                    }
                )

                y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                auc_base_lp = cv_auc_logit(
                    base_feat,
                    y_perm,
                    random_state=45_298 + domain_idx * 100_000 + split_idx * 10_000 + perm_idx,
                    n_splits=H119_CV_SPLITS,
                )
                auc_gate_lp = cv_auc_logit(
                    gate_feat,
                    y_perm,
                    random_state=45_299 + domain_idx * 100_000 + split_idx * 10_000 + perm_idx,
                    n_splits=H119_CV_SPLITS,
                )
                null_label[perm_idx] = (
                    float(auc_gate_lp - auc_base_lp)
                    if np.isfinite(auc_gate_lp) and np.isfinite(auc_base_lp)
                    else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H119",
                        "null_kind": "label_permutation",
                        "domain": domain,
                        "seed_tag": H119_SEED,
                        "split_regime": split_regime,
                        "layer": int(H119_LAYER),
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_label[perm_idx]),
                    }
                )

            all_null = np.concatenate([null_dis, null_map, null_label])
            q95 = float(np.nanquantile(all_null, 0.95))
            p_dis = BASE.empirical_upper_tail_p(delta, null_dis)
            p_map = BASE.empirical_upper_tail_p(delta, null_map)
            p_label = BASE.empirical_upper_tail_p(delta, null_label)
            p_best = float(np.nanmin(np.asarray([p_dis, p_map, p_label], dtype=float)))

            rows.append(
                {
                    "hypothesis_id": "H119",
                    "domain": domain,
                    "seed_tag": H119_SEED,
                    "split_regime": split_regime,
                    "layer": int(H119_LAYER),
                    "n_edges_eval": int(labels.size),
                    "n_positive_eval": int(labels.sum()),
                    "auc_base_transfer": float(auc_base),
                    "auc_disagreement_gated_transfer": float(auc_gate),
                    "delta_vs_h70": float(delta),
                    "q95_null_delta_auc": float(q95),
                    "null_gap_q95": float(delta - q95),
                    "p_disagreement_upper": float(p_dis),
                    "p_mapping_upper": float(p_map),
                    "p_label_upper": float(p_label),
                    "p_best_upper": float(p_best),
                    "mean_disagreement": float(np.mean(disagreement)),
                    "high_disagreement_frac": float(np.mean(dis_bins == 2)),
                }
            )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime"])
    by_row_path = ITER_DIR / "h119_disagreement_gated_transfer_by_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h119_disagreement_gated_transfer_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_df = domain_split_summary(by_row_df)
    summary_path = ITER_DIR / "h119_disagreement_gated_transfer_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_vs_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_vs_h70"] > 0.0).sum()) if not summary_df.empty else 0,
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95"] > 0.0).sum()) if not summary_df.empty else 0,
        "artifact_paths": {
            "by_domain_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def build_weighted_knn_graph(points: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray, list[set[int]]]:
    edges = BASE.build_knn_edge_array(points=points, n_neighbors=n_neighbors)
    neighbors = BASE.adjacency_neighbors(points.shape[0], edges)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for u, v in edges:
        iu = int(u)
        iv = int(v)
        w = float(np.linalg.norm(points[iu] - points[iv]))
        rows.extend([iu, iv])
        cols.extend([iv, iu])
        vals.extend([w, w])
    mat = csr_matrix((vals, (rows, cols)), shape=(points.shape[0], points.shape[0]))
    return edges, mat, neighbors


def edge_forman_curvature(edges: np.ndarray, neighbors: list[set[int]]) -> dict[tuple[int, int], float]:
    deg = np.asarray([len(nb) for nb in neighbors], dtype=float)
    out: dict[tuple[int, int], float] = {}
    for u, v in edges:
        iu = int(u)
        iv = int(v)
        curv = float(4.0 - deg[iu] - deg[iv])
        out[(iu, iv)] = curv
        out[(iv, iu)] = curv
    return out


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


def curvature_features_for_pairs(
    source_local: np.ndarray,
    target_local: np.ndarray,
    dist_mat: np.ndarray,
    predecessors: np.ndarray,
    curvatures: dict[tuple[int, int], float],
) -> np.ndarray:
    cache: dict[tuple[int, int], np.ndarray] = {}
    feats = np.zeros((source_local.size, 5), dtype=float)
    for i, (s, t) in enumerate(zip(source_local, target_local)):
        a = int(s)
        b = int(t)
        key = (min(a, b), max(a, b))
        if key not in cache:
            path = path_nodes_from_predecessor(predecessors, src=a, tgt=b)
            edge_curv = []
            for u, v in zip(path[:-1], path[1:]):
                edge_curv.append(float(curvatures.get((int(u), int(v)), 0.0)))
            if edge_curv:
                arr = np.asarray(edge_curv, dtype=float)
                mean_curv = float(np.mean(arr))
                min_curv = float(np.min(arr))
                std_curv = float(np.std(arr))
                drift_abs = float(np.abs(arr[-1] - arr[0]))
                path_len = float(len(path) - 1)
            else:
                mean_curv = 0.0
                min_curv = 0.0
                std_curv = 0.0
                drift_abs = 0.0
                path_len = 1.0
            cache[key] = np.asarray([mean_curv, min_curv, std_curv, drift_abs, path_len], dtype=float)
        feats[i] = cache[key]
    return feats


def run_h120_geodesic_curvature_drift(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_idx, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H120_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_idx, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H120_GENE_CAP))
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

            rng = np.random.default_rng(45_390 + domain_idx * 1000 + split_idx * 100)
            sample_idx = stratified_index_sample(labels_all, max_n=H120_EDGE_SAMPLE, rng=rng)
            if sample_idx.size < 120:
                continue

            source_local = source_local_all[sample_idx]
            target_local = target_local_all[sample_idx]
            labels = labels_all[sample_idx]

            for layer in H120_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue
                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=45_391 + domain_idx * 1000 + split_idx * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H120_NEIGHBORS)
                geodesic_w = confidence_weighted_geodesic(geodesic, support_dir)

                h70 = compute_h70_scores(
                    geodesic=geodesic_w,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=[8, 12, 16],
                )

                edges, weighted_graph, neighbors = build_weighted_knn_graph(points=points_pca, n_neighbors=H120_NEIGHBORS)
                curv_map = edge_forman_curvature(edges=edges, neighbors=neighbors)
                dist_mat, predecessors = shortest_path(
                    weighted_graph,
                    directed=False,
                    unweighted=False,
                    return_predecessors=True,
                )
                curv_feat = curvature_features_for_pairs(
                    source_local=source_local,
                    target_local=target_local,
                    dist_mat=dist_mat,
                    predecessors=predecessors,
                    curvatures=curv_map,
                )

                model_feat = np.column_stack([h70, curv_feat])
                auc_h70 = BASE.safe_auc(labels, h70)
                auc_model = cv_auc_logit(
                    model_feat,
                    labels,
                    random_state=45_392 + domain_idx * 1000 + split_idx * 100 + layer,
                    n_splits=H120_CV_SPLITS,
                )
                delta = float(auc_model - auc_h70) if np.isfinite(auc_model) and np.isfinite(auc_h70) else float("nan")

                edge_len = geodesic_w[source_local, target_local]
                dist_bins = BASE.degree_bins(edge_len, max_bins=6)
                deg_sum = edge_degree_sum(points_pca, H120_NEIGHBORS, source_local, target_local)
                deg_bins = BASE.degree_bins(deg_sum, max_bins=6)
                edge_strata = build_edge_strata(edge_len, deg_sum, max_len_bins=6, max_deg_bins=4)

                null_dist = np.empty(H120_NULL_PERM, dtype=float)
                null_deg = np.empty(H120_NULL_PERM, dtype=float)
                null_label = np.empty(H120_NULL_PERM, dtype=float)

                for perm_idx in range(H120_NULL_PERM):
                    feat_dist = permute_rows_within_strata(curv_feat, dist_bins, rng)
                    auc_dist = cv_auc_logit(
                        np.column_stack([h70, feat_dist]),
                        labels,
                        random_state=45_393 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H120_CV_SPLITS,
                    )
                    null_dist[perm_idx] = (
                        float(auc_dist - auc_h70) if np.isfinite(auc_dist) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H120",
                            "null_kind": "distance_bin_curvature_feature_shuffle",
                            "domain": domain,
                            "seed_tag": H120_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_dist[perm_idx]),
                        }
                    )

                    feat_deg = permute_rows_within_strata(curv_feat, deg_bins, rng)
                    auc_deg = cv_auc_logit(
                        np.column_stack([h70, feat_deg]),
                        labels,
                        random_state=45_394 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H120_CV_SPLITS,
                    )
                    null_deg[perm_idx] = (
                        float(auc_deg - auc_h70) if np.isfinite(auc_deg) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H120",
                            "null_kind": "degree_bin_curvature_feature_shuffle",
                            "domain": domain,
                            "seed_tag": H120_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_deg[perm_idx]),
                        }
                    )

                    y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                    auc_lp = cv_auc_logit(
                        model_feat,
                        y_perm,
                        random_state=45_395 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H120_CV_SPLITS,
                    )
                    auc_h70_lp = BASE.safe_auc(y_perm, h70)
                    null_label[perm_idx] = (
                        float(auc_lp - auc_h70_lp)
                        if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H120",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H120_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_dist, null_deg, null_label])
                q95 = float(np.nanquantile(all_null, 0.95))
                p_dist = BASE.empirical_upper_tail_p(delta, null_dist)
                p_deg = BASE.empirical_upper_tail_p(delta, null_deg)
                p_label = BASE.empirical_upper_tail_p(delta, null_label)
                p_best = float(np.nanmin(np.asarray([p_dist, p_deg, p_label], dtype=float)))

                rows.append(
                    {
                        "hypothesis_id": "H120",
                        "domain": domain,
                        "seed_tag": H120_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70": float(auc_h70),
                        "auc_curvature_drift_model": float(auc_model),
                        "delta_vs_h70": float(delta),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95": float(delta - q95),
                        "p_distance_shuffle_upper": float(p_dist),
                        "p_degree_shuffle_upper": float(p_deg),
                        "p_label_upper": float(p_label),
                        "p_best_upper": float(p_best),
                        "mean_curvature_along_path": float(np.mean(curv_feat[:, 0])),
                        "mean_path_len": float(np.mean(curv_feat[:, 4])),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h120_geodesic_curvature_drift_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h120_geodesic_curvature_drift_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_df = domain_split_summary(by_row_df)
    summary_path = ITER_DIR / "h120_geodesic_curvature_drift_domain_summary.csv"
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

    h118 = run_h118_signed_motif_module_robustness(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
        trrust_sign_map=trrust_sign_map,
        trrust_tf_out_degree=trrust_tf_out_degree,
    )
    h119 = run_h119_disagreement_conditioned_transfer(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h120 = run_h120_geodesic_curvature_drift(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    summary = {
        "iteration": "iter_0045",
        "h118": h118,
        "h119": h119,
        "h120": h120,
    }
    summary_path = ITER_DIR / "iter0045_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
