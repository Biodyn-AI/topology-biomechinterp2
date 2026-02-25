from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path
from sklearn.cluster import KMeans


ITER_DIR = Path("iterations/iter_0049")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base_iter0049")
PREV = load_module(Path("iterations/iter_0047/run_iter0047_screen.py"), "iter0047_prev_iter0049")

TRRUST_PATH = PREV.TRRUST_PATH

# H130 / N656: continuous GO semantic hardening of H127 stack.
H130_SEEDS = ["seed42_main", "seed43", "seed44"]
H130_LAYER = 11
H130_GENE_CAP = 220
H130_GENE_CAP_LUNG_DUAL = 260
H130_MIN_GENE_NODES = 120
H130_MIN_GENE_NODES_LUNG_DUAL = 90
H130_NEIGHBORS = 12
H130_EDGE_SAMPLE = 240
H130_EDGE_SAMPLE_LUNG_DUAL = 220
H130_CV_SPLITS = 3
H130_NULL_PERM = 24

# H131 / N653: chart/sheaf-style cross-model alignment rescue.
H131_SEED = "seed42_main"
H131_LAYERS = [7, 11]
H131_SPLITS = ("source_disjoint", "target_disjoint")
H131_MIN_SHARED = 90
H131_EDGE_SAMPLE = 260
H131_NULL_PERM = 24
H131_CHARTS = 4
H131_MIN_CHART = 20

# H132 / N650: local chart-fracture manifold diagnostic.
H132_SEED = "seed42_main"
H132_LAYERS = [7, 11]
H132_GENE_CAP = 180
H132_NEIGHBORS = 12
H132_EDGE_SAMPLE = 240
H132_CV_SPLITS = 4
H132_NULL_PERM = 24
H132_CHART_WINDOW = 5
H132_ANGLE_DEG = 35.0


def ensure_required_inputs() -> None:
    required = [
        BASE.DOROTHEA_PATH,
        BASE.GENE2GO_PATH,
        BASE.OMNIPATH_INTERACTIONS_PATH,
        BASE.STRING_CACHE_PATH,
        TRRUST_PATH,
    ]
    for domain, run_map in BASE.SCGPT_RUNS_BY_DOMAIN.items():
        for seed_tag in set(H130_SEEDS + [H131_SEED, H132_SEED]):
            run_dir = run_map[seed_tag]
            required.append(run_dir / "cycle1_edge_dataset.tsv")
            required.append(run_dir / "layer_gene_embeddings.npy")
        required.append(BASE.GENEFORMER_EDGE_BY_DOMAIN[domain])

    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def finite_q95(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, 0.95))


def finite_fisher(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(BASE.safe_fisher_p(vals))


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


def term_ic_map(symbols_upper: list[str], gene2go_upper: dict[str, set[str]]) -> dict[str, float]:
    freq: dict[str, int] = {}
    total = 0
    for sym in symbols_upper:
        terms = gene2go_upper.get(sym, set())
        if not terms:
            continue
        total += 1
        for term in terms:
            freq[term] = freq.get(term, 0) + 1
    denom = max(1, total)
    return {term: float(np.log((denom + 1.0) / (count + 1.0))) for term, count in freq.items()}


def semantic_pair_scores(
    src_terms: set[str],
    tgt_terms: set[str],
    ic_map: dict[str, float],
) -> tuple[float, float, float, float]:
    if not src_terms or not tgt_terms:
        return 0.0, 0.0, 0.0, 0.0

    inter = src_terms & tgt_terms
    union = src_terms | tgt_terms
    if not inter:
        return 0.0, 0.0, 0.0, 0.0

    jaccard = float(len(inter) / max(1, len(union)))
    ic_inter = float(sum(ic_map.get(t, 0.0) for t in inter))
    ic_src = float(sum(ic_map.get(t, 0.0) for t in src_terms))
    ic_tgt = float(sum(ic_map.get(t, 0.0) for t in tgt_terms))
    lin = float((2.0 * ic_inter) / max(1e-8, ic_src + ic_tgt))
    resnik = float(max(ic_map.get(t, 0.0) for t in inter))

    denom = float(np.sqrt(max(1, len(src_terms) * len(tgt_terms))))
    overlap_norm = float(len(inter) / max(1e-8, denom))
    return jaccard, lin, resnik, overlap_norm


def build_go_semantic_features(
    src_sym: list[str],
    tgt_sym: list[str],
    gene2go_upper: dict[str, set[str]],
    anchor_terms: set[str],
    ic_map: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(src_sym)
    sem_global = np.zeros(n, dtype=float)
    sem_anchor = np.zeros(n, dtype=float)
    resnik_norm = np.zeros(n, dtype=float)
    overlap_norm = np.zeros(n, dtype=float)

    max_ic = float(max(ic_map.values())) if ic_map else 1.0
    max_ic = max(max_ic, 1e-8)

    cache_terms: dict[str, set[str]] = {}

    def terms_for(sym: str) -> set[str]:
        key = str(sym).upper()
        if key not in cache_terms:
            cache_terms[key] = set(gene2go_upper.get(key, set()))
        return cache_terms[key]

    for i, (src, tgt) in enumerate(zip(src_sym, tgt_sym)):
        src_terms = terms_for(src)
        tgt_terms = terms_for(tgt)

        j, lin, res, ov = semantic_pair_scores(src_terms, tgt_terms, ic_map)
        sem_global[i] = float(0.55 * lin + 0.30 * j + 0.15 * ov)
        resnik_norm[i] = float(res / max_ic)
        overlap_norm[i] = float(ov)

        src_anchor = src_terms & anchor_terms
        tgt_anchor = tgt_terms & anchor_terms
        j_a, lin_a, res_a, ov_a = semantic_pair_scores(src_anchor, tgt_anchor, ic_map)
        sem_anchor[i] = float(0.65 * lin_a + 0.20 * j_a + 0.15 * ov_a + 0.10 * (res_a / max_ic))

    return sem_global, sem_anchor, resnik_norm, overlap_norm


def summarize_by_domain_split(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (domain, split_regime), group in df.groupby(["domain", "split_regime"], sort=True):
        rows.append(
            {
                "domain": str(domain),
                "split_regime": str(split_regime),
                "n_rows": int(group.shape[0]),
                "mean_delta_vs_h70": float(group["delta_vs_h70"].mean()),
                "mean_null_gap_q95": float(group["null_gap_q95"].mean()),
                "fraction_delta_positive": float((group["delta_vs_h70"] > 0.0).mean()),
                "fraction_null_gap_positive": float((group["null_gap_q95"] > 0.0).mean()),
                "combined_fisher_p_best": finite_fisher(group["p_best_upper"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows).sort_values(["domain", "split_regime"])


def summarize_by_domain(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for domain, group in df.groupby("domain", sort=True):
        rows.append(
            {
                "domain": str(domain),
                "n_rows": int(group.shape[0]),
                "mean_delta_vs_h70": float(group["delta_vs_h70"].mean()),
                "mean_null_gap_q95": float(group["null_gap_q95"].mean()),
                "fraction_delta_positive": float((group["delta_vs_h70"] > 0.0).mean()),
                "fraction_null_gap_positive": float((group["null_gap_q95"] > 0.0).mean()),
                "combined_fisher_p_best": finite_fisher(group["p_best_upper"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows).sort_values("domain")


def run_h130_semantic_hardening(
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

    trrust_gene_set: set[str] = set()
    for src, tgt in trrust_sign_map.keys():
        trrust_gene_set.add(str(src).upper())
        trrust_gene_set.add(str(tgt).upper())

    for domain_idx, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_idx, seed_tag in enumerate(H130_SEEDS):
            run_dir = run_map[seed_tag]
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            split_masks = PREV.build_split_masks_plus(edge_df)

            for split_idx, split_regime in enumerate(ordered_splits):
                split_mask = split_masks.get(split_regime)
                if split_mask is None:
                    continue
                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                is_lung_dual = domain == "lung" and split_regime == "dual_axis_disjoint"
                gene_cap = H130_GENE_CAP_LUNG_DUAL if is_lung_dual else H130_GENE_CAP
                min_gene_nodes = H130_MIN_GENE_NODES_LUNG_DUAL if is_lung_dual else H130_MIN_GENE_NODES
                sample_cap = H130_EDGE_SAMPLE_LUNG_DUAL if is_lung_dual else H130_EDGE_SAMPLE

                top_genes = set(BASE.select_top_genes(split_edges, gene_cap=gene_cap))
                split_edges = split_edges.loc[
                    split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
                ].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                edge_gene_indices, gene_to_local, symbols, support_dir = PREV.build_symbol_resources(
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

                rng = np.random.default_rng(49_100 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100)
                sample_idx = PREV.stratified_index_sample(labels_all, max_n=sample_cap, rng=rng)
                if sample_idx.size < 110:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]
                if H130_LAYER >= layer_embeddings.shape[0]:
                    continue

                points = layer_embeddings[H130_LAYER, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=49_101 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H130_NEIGHBORS)
                geodesic_w = PREV.confidence_weighted_geodesic(geodesic, support_dir)

                h70 = PREV.compute_h70_scores(
                    geodesic=geodesic_w,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=[8, 12, 16],
                )

                knn_edges = BASE.build_knn_edge_array(points=points_pca, n_neighbors=H130_NEIGHBORS)
                neighbors = BASE.adjacency_neighbors(points_pca.shape[0], knn_edges)
                community_labels = PREV.label_propagation_communities(neighbors=neighbors, rng=rng, max_iter=20)
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

                all_symbols_upper = [str(s).upper() for s in symbols]
                ic_map = term_ic_map(all_symbols_upper, gene2go_upper)
                anchor_terms = PREV.select_anchor_go_terms(
                    symbols=symbols,
                    gene2go_upper=gene2go_upper,
                    trrust_gene_set=trrust_gene_set,
                    max_terms=80,
                )
                go_sem, go_anchor_sem, go_resnik, go_overlap = build_go_semantic_features(
                    src_sym=src_sym,
                    tgt_sym=tgt_sym,
                    gene2go_upper=gene2go_upper,
                    anchor_terms=anchor_terms,
                    ic_map=ic_map,
                )

                feat = np.column_stack(
                    [
                        h70,
                        same_community,
                        motif_present,
                        sign_consistent,
                        string_conf,
                        string_high,
                        go_sem,
                        go_anchor_sem,
                        go_resnik,
                        go_overlap,
                        same_community * go_sem,
                        sign_consistent * go_sem,
                        string_conf * go_sem,
                        string_conf * go_anchor_sem,
                        go_sem * sign_consistent * same_community,
                        h70 * go_sem,
                    ]
                )

                auc_h70 = BASE.safe_auc(labels, h70)
                auc_model = PREV.cv_auc_logit(
                    feat,
                    labels,
                    random_state=49_102 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100,
                    n_splits=H130_CV_SPLITS,
                )
                delta = float(auc_model - auc_h70) if np.isfinite(auc_model) and np.isfinite(auc_h70) else float("nan")

                tf_deg = np.asarray([trrust_tf_out_degree.get(s, 0) for s in src_sym], dtype=float)
                tf_bins = BASE.degree_bins(tf_deg, max_bins=4)
                tgt_deg_proxy = np.sum(support_dir > 0.55, axis=0).astype(float)
                tgt_deg_bins = BASE.degree_bins(tgt_deg_proxy[target_local], max_bins=4)
                str_bins = PREV.quantile_bins(string_conf, n_bins=4)
                depth_bins = PREV.quantile_bins(go_resnik, n_bins=4)
                semantic_strata = (tf_bins * 256 + tgt_deg_bins * 32 + str_bins * 4 + depth_bins).astype(int)

                edge_len = geodesic_w[source_local, target_local]
                deg_sum = PREV.edge_degree_sum(points_pca, H130_NEIGHBORS, source_local, target_local)
                edge_strata = PREV.build_edge_strata(edge_len, deg_sum, max_len_bins=6, max_deg_bins=4)

                null_tf_sign = np.empty(H130_NULL_PERM, dtype=float)
                null_motif_decoy = np.empty(H130_NULL_PERM, dtype=float)
                null_string_bin = np.empty(H130_NULL_PERM, dtype=float)
                null_go_sem = np.empty(H130_NULL_PERM, dtype=float)
                null_label = np.empty(H130_NULL_PERM, dtype=float)

                for perm_idx in range(H130_NULL_PERM):
                    sign_perm = PREV.shuffle_sign_within_tf(trrust_sign, src_symbols=src_sym, rng=rng)
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
                            go_sem,
                            go_anchor_sem,
                            go_resnik,
                            go_overlap,
                            same_community * go_sem,
                            sign_cons_perm_tf * go_sem,
                            string_conf * go_sem,
                            string_conf * go_anchor_sem,
                            go_sem * sign_cons_perm_tf * same_community,
                            h70 * go_sem,
                        ]
                    )
                    auc_tf = PREV.cv_auc_logit(
                        feat_tf,
                        labels,
                        random_state=49_103 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H130_CV_SPLITS,
                    )
                    null_tf_sign[perm_idx] = float(auc_tf - auc_h70) if np.isfinite(auc_tf) and np.isfinite(auc_h70) else float("nan")
                    null_rows.append(
                        {
                            "hypothesis_id": "H130",
                            "null_kind": "tf_identity_preserving_sign_shuffle",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H130_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_tf_sign[perm_idx]),
                        }
                    )

                    motif_perm = PREV.permute_within_strata(motif_present, strata=semantic_strata, rng=rng).astype(float)
                    sign_pool = PREV.permute_within_strata(sign_consistent, strata=semantic_strata, rng=rng).astype(float)
                    sign_cons_perm = motif_perm * sign_pool
                    feat_decoy = np.column_stack(
                        [
                            h70,
                            same_community,
                            motif_perm,
                            sign_cons_perm,
                            string_conf,
                            string_high,
                            go_sem,
                            go_anchor_sem,
                            go_resnik,
                            go_overlap,
                            same_community * go_sem,
                            sign_cons_perm * go_sem,
                            string_conf * go_sem,
                            string_conf * go_anchor_sem,
                            go_sem * sign_cons_perm * same_community,
                            h70 * go_sem,
                        ]
                    )
                    auc_decoy = PREV.cv_auc_logit(
                        feat_decoy,
                        labels,
                        random_state=49_104 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H130_CV_SPLITS,
                    )
                    null_motif_decoy[perm_idx] = (
                        float(auc_decoy - auc_h70) if np.isfinite(auc_decoy) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H130",
                            "null_kind": "motif_decoy_shuffle_matched_tf_target_degree",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H130_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_motif_decoy[perm_idx]),
                        }
                    )

                    str_perm = PREV.permute_within_strata(string_conf, strata=semantic_strata, rng=rng).astype(float)
                    str_high_perm = (str_perm >= 0.70).astype(float)
                    feat_str = np.column_stack(
                        [
                            h70,
                            same_community,
                            motif_present,
                            sign_consistent,
                            str_perm,
                            str_high_perm,
                            go_sem,
                            go_anchor_sem,
                            go_resnik,
                            go_overlap,
                            same_community * go_sem,
                            sign_consistent * go_sem,
                            str_perm * go_sem,
                            str_perm * go_anchor_sem,
                            go_sem * sign_consistent * same_community,
                            h70 * go_sem,
                        ]
                    )
                    auc_str = PREV.cv_auc_logit(
                        feat_str,
                        labels,
                        random_state=49_105 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H130_CV_SPLITS,
                    )
                    null_string_bin[perm_idx] = (
                        float(auc_str - auc_h70) if np.isfinite(auc_str) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H130",
                            "null_kind": "string_confidence_bin_permutation_within_degree_strata",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H130_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_string_bin[perm_idx]),
                        }
                    )

                    go_sem_perm = PREV.permute_within_strata(go_sem, strata=semantic_strata, rng=rng).astype(float)
                    go_anchor_perm = PREV.permute_within_strata(go_anchor_sem, strata=semantic_strata, rng=rng).astype(float)
                    go_resnik_perm = PREV.permute_within_strata(go_resnik, strata=semantic_strata, rng=rng).astype(float)
                    go_overlap_perm = PREV.permute_within_strata(go_overlap, strata=semantic_strata, rng=rng).astype(float)
                    feat_go = np.column_stack(
                        [
                            h70,
                            same_community,
                            motif_present,
                            sign_consistent,
                            string_conf,
                            string_high,
                            go_sem_perm,
                            go_anchor_perm,
                            go_resnik_perm,
                            go_overlap_perm,
                            same_community * go_sem_perm,
                            sign_consistent * go_sem_perm,
                            string_conf * go_sem_perm,
                            string_conf * go_anchor_perm,
                            go_sem_perm * sign_consistent * same_community,
                            h70 * go_sem_perm,
                        ]
                    )
                    auc_go = PREV.cv_auc_logit(
                        feat_go,
                        labels,
                        random_state=49_106 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H130_CV_SPLITS,
                    )
                    null_go_sem[perm_idx] = (
                        float(auc_go - auc_h70) if np.isfinite(auc_go) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H130",
                            "null_kind": "go_semantic_rewiring_within_depth_proxy_strata",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H130_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_go_sem[perm_idx]),
                        }
                    )

                    y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                    auc_lp = PREV.cv_auc_logit(
                        feat,
                        y_perm,
                        random_state=49_107 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H130_CV_SPLITS,
                    )
                    auc_h70_lp = BASE.safe_auc(y_perm, h70)
                    null_label[perm_idx] = (
                        float(auc_lp - auc_h70_lp) if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H130",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H130_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_tf_sign, null_motif_decoy, null_string_bin, null_go_sem, null_label])
                q95 = finite_q95(all_null)
                p_tf = BASE.empirical_upper_tail_p(delta, null_tf_sign)
                p_decoy = BASE.empirical_upper_tail_p(delta, null_motif_decoy)
                p_string = BASE.empirical_upper_tail_p(delta, null_string_bin)
                p_go = BASE.empirical_upper_tail_p(delta, null_go_sem)
                p_label = BASE.empirical_upper_tail_p(delta, null_label)
                p_best = float(np.nanmin(np.asarray([p_tf, p_decoy, p_string, p_go, p_label], dtype=float)))

                rows.append(
                    {
                        "hypothesis_id": "H130",
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(H130_LAYER),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70": float(auc_h70),
                        "auc_semantic_hardening_model": float(auc_model),
                        "delta_vs_h70": float(delta),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95": float(delta - q95),
                        "p_tf_sign_upper": float(p_tf),
                        "p_motif_decoy_upper": float(p_decoy),
                        "p_string_bin_upper": float(p_string),
                        "p_go_semantic_upper": float(p_go),
                        "p_label_upper": float(p_label),
                        "p_best_upper": float(p_best),
                        "motif_coverage": float(np.mean(motif_present)),
                        "string_conf_mean": float(np.mean(string_conf)),
                        "same_community_rate": float(np.mean(same_community)),
                        "go_semantic_mean": float(np.mean(go_sem)),
                        "go_anchor_semantic_mean": float(np.mean(go_anchor_sem)),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["seed_tag", "domain", "split_regime"])
    by_row_path = ITER_DIR / "h130_semantic_go_string_hardening_by_seed_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "seed_tag", "domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h130_semantic_go_string_hardening_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_df = summarize_by_domain_split(by_row_df)
    summary_path = ITER_DIR / "h130_semantic_go_string_hardening_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    lung_dual = (
        by_row_df.loc[(by_row_df["domain"] == "lung") & (by_row_df["split_regime"] == "dual_axis_disjoint")]
        if not by_row_df.empty
        else pd.DataFrame()
    )
    immune_source = (
        by_row_df.loc[(by_row_df["domain"] == "immune") & (by_row_df["split_regime"] == "source_disjoint")]
        if not by_row_df.empty
        else pd.DataFrame()
    )

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_vs_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((summary_df["mean_delta_vs_h70"] > 0.0).sum()) if not summary_df.empty else 0,
        "positive_null_gap_domain_splits": int((summary_df["mean_null_gap_q95"] > 0.0).sum()) if not summary_df.empty else 0,
        "lung_dual_axis_mean_null_gap": float(lung_dual["null_gap_q95"].mean()) if not lung_dual.empty else float("nan"),
        "immune_source_mean_null_gap": float(immune_source["null_gap_q95"].mean()) if not immune_source.empty else float("nan"),
        "artifact_paths": {
            "by_seed_domain_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def orthogonal_map(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    x = np.asarray(src, dtype=float)
    y = np.asarray(dst, dtype=float)
    m = x.T @ y
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    return u @ vt


def fit_chart_maps(
    train_sc: np.ndarray,
    train_gf: np.ndarray,
    chart_labels: np.ndarray,
    n_charts: int,
    min_points: int,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    global_map = orthogonal_map(train_gf, train_sc)
    maps: dict[int, np.ndarray] = {}
    for chart_id in range(n_charts):
        idx = np.where(chart_labels == chart_id)[0]
        if idx.size < min_points:
            maps[chart_id] = global_map
            continue
        maps[chart_id] = orthogonal_map(train_gf[idx], train_sc[idx])
    return maps, global_map


def apply_chart_maps(
    features: np.ndarray,
    chart_labels: np.ndarray,
    maps: dict[int, np.ndarray],
    fallback_map: np.ndarray,
) -> np.ndarray:
    out = np.zeros_like(features, dtype=float)
    for i in range(features.shape[0]):
        chart = int(chart_labels[i])
        mapper = maps.get(chart, fallback_map)
        out[i] = features[i] @ mapper
    return out


def cycle_consistency_residual(maps: dict[int, np.ndarray], n_charts: int) -> float:
    ids = [idx for idx in range(n_charts) if idx in maps]
    if len(ids) < 3:
        return float("nan")
    d = next(iter(maps.values())).shape[0]
    eye = np.eye(d, dtype=float)
    residuals: list[float] = []
    for i in range(len(ids) - 2):
        a = ids[i]
        b = ids[i + 1]
        c = ids[i + 2]
        t_ab = maps[b].T @ maps[a]
        t_bc = maps[c].T @ maps[b]
        t_ca = maps[a].T @ maps[c]
        cycle = t_ca @ t_bc @ t_ab
        residuals.append(float(np.linalg.norm(cycle - eye, ord="fro") / max(1.0, d)))
    if not residuals:
        return float("nan")
    return float(np.mean(residuals))


def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(d, d))
    q, r = np.linalg.qr(raw)
    s = np.sign(np.diag(r))
    s[s == 0] = 1.0
    return q * s


def run_h131_chart_sheaf_alignment(
    gene2go_upper: dict[str, set[str]],
    trrust_sign_map: dict[tuple[str, str], int],
) -> dict[str, object]:
    by_row: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    trrust_gene_set: set[str] = set()
    for src, tgt in trrust_sign_map.keys():
        trrust_gene_set.add(str(src).upper())
        trrust_gene_set.add(str(tgt).upper())

    sc_edges: dict[str, pd.DataFrame] = {}
    sc_layers: dict[str, np.ndarray] = {}
    gf_edges: dict[str, pd.DataFrame] = {}
    sc_sig: dict[tuple[str, int], pd.DataFrame] = {}
    gf_sig: dict[str, pd.DataFrame] = {}

    domains = ["immune", "lung", "external_lung"]
    for domain_idx, domain in enumerate(domains):
        run_dir = BASE.SCGPT_RUNS_BY_DOMAIN[domain][H131_SEED]
        sc_edges[domain] = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        sc_layers[domain] = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        gf_edges[domain] = pd.read_csv(BASE.GENEFORMER_EDGE_BY_DOMAIN[domain], sep="\t")

        sc_tables, gf_table = PREV.build_scaffold_signature_tables(
            domain=domain,
            edge_df=sc_edges[domain],
            layer_embeddings=sc_layers[domain],
            gf_df=gf_edges[domain],
            random_base=49_200 + domain_idx * 100,
        )
        for layer, table in sc_tables.items():
            sc_sig[(domain, layer)] = table
        gf_sig[domain] = gf_table

    for domain_idx, target_domain in enumerate(domains):
        source_domains = [d for d in domains if d != target_domain]
        split_masks = BASE.build_split_masks(sc_edges[target_domain])

        for layer in H131_LAYERS:
            train_sc_list: list[np.ndarray] = []
            train_gf_list: list[np.ndarray] = []
            train_sym_list: list[list[str]] = []

            for src_domain in source_domains:
                sc_df = sc_sig.get((src_domain, layer))
                gf_df = gf_sig.get(src_domain)
                if sc_df is None or gf_df is None or sc_df.empty or gf_df.empty:
                    continue
                shared = sorted(set(sc_df.index) & set(gf_df.index))
                if len(shared) < H131_MIN_SHARED:
                    continue
                train_sc_list.append(sc_df.loc[shared].to_numpy(dtype=float))
                train_gf_list.append(gf_df.loc[shared].to_numpy(dtype=float))
                train_sym_list.append(shared)

            if not train_sc_list:
                continue

            train_sc = np.vstack(train_sc_list)
            train_gf = np.vstack(train_gf_list)
            train_symbols = [sym for grp in train_sym_list for sym in grp]
            if train_sc.shape[0] < 2 * H131_MIN_SHARED:
                continue

            sc_mu, sc_sd = BASE.zscore_fit(train_sc)
            gf_mu, gf_sd = BASE.zscore_fit(train_gf)
            train_sc_z = BASE.zscore_apply(train_sc, sc_mu, sc_sd)
            train_gf_z = BASE.zscore_apply(train_gf, gf_mu, gf_sd)

            kmeans = KMeans(n_clusters=H131_CHARTS, random_state=49_210 + domain_idx * 100 + layer, n_init=20)
            train_chart = kmeans.fit_predict(train_sc_z)
            chart_maps, global_map = fit_chart_maps(
                train_sc=train_sc_z,
                train_gf=train_gf_z,
                chart_labels=train_chart,
                n_charts=H131_CHARTS,
                min_points=H131_MIN_CHART,
            )
            cycle_resid = cycle_consistency_residual(chart_maps, n_charts=H131_CHARTS)

            sc_tgt_df = sc_sig.get((target_domain, layer))
            gf_tgt_df = gf_sig.get(target_domain)
            if sc_tgt_df is None or gf_tgt_df is None or sc_tgt_df.empty or gf_tgt_df.empty:
                continue

            shared_tgt = sorted(set(sc_tgt_df.index) & set(gf_tgt_df.index))
            if len(shared_tgt) < H131_MIN_SHARED:
                continue

            sc_tgt_z = BASE.zscore_apply(sc_tgt_df.loc[shared_tgt].to_numpy(dtype=float), sc_mu, sc_sd)
            gf_tgt_z = BASE.zscore_apply(gf_tgt_df.loc[shared_tgt].to_numpy(dtype=float), gf_mu, gf_sd)
            tgt_chart = kmeans.predict(sc_tgt_z)
            mapped_tgt = apply_chart_maps(gf_tgt_z, tgt_chart, chart_maps, global_map)

            self_cos = np.sum(BASE.row_normalize(mapped_tgt) * BASE.row_normalize(sc_tgt_z), axis=1)
            sym_to_pos = {sym: idx for idx, sym in enumerate(shared_tgt)}

            # Keep GO anchor traceability for diagnostics.
            anchor_terms = PREV.select_anchor_go_terms(
                symbols=train_symbols,
                gene2go_upper=gene2go_upper,
                trrust_gene_set=trrust_gene_set,
                max_terms=64,
            )
            anchor_hits = np.array(
                [float(bool(gene2go_upper.get(sym.upper(), set()) & anchor_terms)) for sym in shared_tgt],
                dtype=float,
            )

            for split_idx, split_regime in enumerate(H131_SPLITS):
                split_mask = split_masks.get(split_regime)
                if split_mask is None:
                    continue
                split_df = sc_edges[target_domain].loc[split_mask].copy()
                split_df["source_u"] = split_df["source"].astype(str).str.upper()
                split_df["target_u"] = split_df["target"].astype(str).str.upper()
                keep = split_df["source_u"].isin(sym_to_pos) & split_df["target_u"].isin(sym_to_pos)
                split_df = split_df.loc[keep].copy()
                if split_df["label"].nunique() < 2 or split_df.shape[0] < 220:
                    continue

                labels_all = split_df["label"].to_numpy(dtype=int)
                rng = np.random.default_rng(49_220 + domain_idx * 1000 + layer * 100 + split_idx * 10)
                sample_idx = PREV.stratified_index_sample(labels_all, max_n=H131_EDGE_SAMPLE, rng=rng)
                split_df = split_df.iloc[sample_idx].copy()

                src_sym = split_df["source_u"].to_numpy(dtype=str)
                tgt_sym = split_df["target_u"].to_numpy(dtype=str)
                labels = split_df["label"].to_numpy(dtype=int)
                src_idx = np.array([sym_to_pos[s] for s in src_sym], dtype=int)
                tgt_idx = np.array([sym_to_pos[t] for t in tgt_sym], dtype=int)

                pair_transfer = PREV.edge_pair_cosine_scores(src_idx, tgt_idx, mapped_tgt, sc_tgt_z)
                pair_baseline = PREV.edge_pair_cosine_scores(src_idx, tgt_idx, gf_tgt_z, gf_tgt_z)
                chart_same = (tgt_chart[src_idx] == tgt_chart[tgt_idx]).astype(float)
                anchor_pair = 0.5 * (anchor_hits[src_idx] + anchor_hits[tgt_idx])

                transfer_scores = (
                    0.55 * pair_transfer
                    + 0.25 * (self_cos[src_idx] + self_cos[tgt_idx]) * 0.5
                    + 0.10 * chart_same
                    + 0.10 * anchor_pair
                    - 0.05 * (0.0 if not np.isfinite(cycle_resid) else cycle_resid)
                )
                baseline_scores = pair_baseline

                auc_transfer = BASE.safe_auc(labels, transfer_scores)
                auc_baseline = BASE.safe_auc(labels, baseline_scores)
                delta = float(auc_transfer - auc_baseline) if np.isfinite(auc_transfer) and np.isfinite(auc_baseline) else float("nan")

                edge_strata = PREV.build_edge_strata(
                    edge_length=np.abs(transfer_scores),
                    degree_sum=np.abs(baseline_scores),
                    max_len_bins=6,
                    max_deg_bins=4,
                )

                null_chart = np.empty(H131_NULL_PERM, dtype=float)
                null_cycle = np.empty(H131_NULL_PERM, dtype=float)
                null_random = np.empty(H131_NULL_PERM, dtype=float)
                null_label = np.empty(H131_NULL_PERM, dtype=float)

                for perm_idx in range(H131_NULL_PERM):
                    # Null A: chart-label permutation preserving chart counts.
                    perm_chart = train_chart[rng.permutation(train_chart.shape[0])]
                    maps_perm, global_perm = fit_chart_maps(
                        train_sc=train_sc_z,
                        train_gf=train_gf_z,
                        chart_labels=perm_chart,
                        n_charts=H131_CHARTS,
                        min_points=H131_MIN_CHART,
                    )
                    mapped_perm = apply_chart_maps(gf_tgt_z, tgt_chart, maps_perm, global_perm)
                    tr_perm = PREV.edge_pair_cosine_scores(src_idx, tgt_idx, mapped_perm, sc_tgt_z)
                    tr_perm = (
                        0.55 * tr_perm
                        + 0.25 * (self_cos[src_idx] + self_cos[tgt_idx]) * 0.5
                        + 0.10 * chart_same
                        + 0.10 * anchor_pair
                    )
                    auc_perm = BASE.safe_auc(labels, tr_perm)
                    null_chart[perm_idx] = float(auc_perm - auc_baseline) if np.isfinite(auc_perm) and np.isfinite(auc_baseline) else float("nan")
                    null_rows.append(
                        {
                            "hypothesis_id": "H131",
                            "null_kind": "chart_label_permutation_preserving_sizes",
                            "domain": target_domain,
                            "seed_tag": H131_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_chart[perm_idx]),
                        }
                    )

                    # Null B: cycle-order shuffle (reassign learned chart maps to different chart ids).
                    chart_order = np.arange(H131_CHARTS, dtype=int)
                    shuffled = rng.permutation(chart_order)
                    remapped: dict[int, np.ndarray] = {}
                    for src_chart, dst_chart in zip(chart_order, shuffled):
                        remapped[int(src_chart)] = chart_maps.get(int(dst_chart), global_map)
                    mapped_cycle = apply_chart_maps(gf_tgt_z, tgt_chart, remapped, global_map)
                    tr_cycle = PREV.edge_pair_cosine_scores(src_idx, tgt_idx, mapped_cycle, sc_tgt_z)
                    tr_cycle = (
                        0.55 * tr_cycle
                        + 0.25 * (self_cos[src_idx] + self_cos[tgt_idx]) * 0.5
                        + 0.10 * chart_same
                        + 0.10 * anchor_pair
                    )
                    auc_cycle = BASE.safe_auc(labels, tr_cycle)
                    null_cycle[perm_idx] = (
                        float(auc_cycle - auc_baseline) if np.isfinite(auc_cycle) and np.isfinite(auc_baseline) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H131",
                            "null_kind": "cycle_order_shuffle",
                            "domain": target_domain,
                            "seed_tag": H131_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_cycle[perm_idx]),
                        }
                    )

                    # Null C: random orthogonal maps baseline.
                    rand_maps: dict[int, np.ndarray] = {}
                    dim = train_sc_z.shape[1]
                    for chart in range(H131_CHARTS):
                        rand_maps[int(chart)] = random_orthogonal(dim, rng)
                    mapped_rand = apply_chart_maps(gf_tgt_z, tgt_chart, rand_maps, np.eye(dim))
                    tr_rand = PREV.edge_pair_cosine_scores(src_idx, tgt_idx, mapped_rand, sc_tgt_z)
                    tr_rand = (
                        0.55 * tr_rand
                        + 0.25 * (self_cos[src_idx] + self_cos[tgt_idx]) * 0.5
                        + 0.10 * chart_same
                        + 0.10 * anchor_pair
                    )
                    auc_rand = BASE.safe_auc(labels, tr_rand)
                    null_random[perm_idx] = (
                        float(auc_rand - auc_baseline) if np.isfinite(auc_rand) and np.isfinite(auc_baseline) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H131",
                            "null_kind": "random_orthogonal_map_baseline",
                            "domain": target_domain,
                            "seed_tag": H131_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_random[perm_idx]),
                        }
                    )

                    # Null D: label permutation.
                    y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                    auc_tp = BASE.safe_auc(y_perm, transfer_scores)
                    auc_bp = BASE.safe_auc(y_perm, baseline_scores)
                    null_label[perm_idx] = float(auc_tp - auc_bp) if np.isfinite(auc_tp) and np.isfinite(auc_bp) else float("nan")
                    null_rows.append(
                        {
                            "hypothesis_id": "H131",
                            "null_kind": "label_permutation",
                            "domain": target_domain,
                            "seed_tag": H131_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_chart, null_cycle, null_random, null_label])
                q95 = finite_q95(all_null)
                p_chart = BASE.empirical_upper_tail_p(delta, null_chart)
                p_cycle = BASE.empirical_upper_tail_p(delta, null_cycle)
                p_rand = BASE.empirical_upper_tail_p(delta, null_random)
                p_label = BASE.empirical_upper_tail_p(delta, null_label)
                p_best = float(np.nanmin(np.asarray([p_chart, p_cycle, p_rand, p_label], dtype=float)))

                by_row.append(
                    {
                        "hypothesis_id": "H131",
                        "domain": target_domain,
                        "seed_tag": H131_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_transfer": float(auc_transfer),
                        "auc_baseline": float(auc_baseline),
                        "delta_vs_h70": float(delta),
                        "alignment_delta_vs_random": float(delta),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95": float(delta - q95),
                        "cycle_consistency_error": float(cycle_resid),
                        "p_chart_upper": float(p_chart),
                        "p_cycle_upper": float(p_cycle),
                        "p_random_upper": float(p_rand),
                        "p_label_upper": float(p_label),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(by_row)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h131_chart_sheaf_alignment_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h131_chart_sheaf_alignment_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    split_summary_df = summarize_by_domain_split(by_row_df)
    split_summary_path = ITER_DIR / "h131_chart_sheaf_alignment_domain_split_summary.csv"
    split_summary_df.to_csv(split_summary_path, index=False)

    domain_summary_df = summarize_by_domain(by_row_df)
    domain_summary_path = ITER_DIR / "h131_chart_sheaf_alignment_domain_summary.csv"
    domain_summary_df.to_csv(domain_summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_vs_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_count": int((domain_summary_df["mean_null_gap_q95"] > 0.0).sum()) if not domain_summary_df.empty else 0,
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


def _safe_unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-10:
        return np.zeros_like(v)
    return v / norm


def first_pc(points: np.ndarray) -> np.ndarray:
    x = np.asarray(points, dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] < 2:
        return np.zeros(x.shape[1], dtype=float)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return _safe_unit(vt[0])


def principal_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    ua = _safe_unit(a)
    ub = _safe_unit(b)
    dot = float(np.clip(np.abs(np.dot(ua, ub)), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def chart_fracture_stats(
    path: list[int],
    points: np.ndarray,
    window: int,
    angle_threshold_deg: float,
) -> tuple[float, float, float, float, float, float]:
    if len(path) < max(window + 1, 4):
        return 0.0, 0.0, 0.0, 0.0, 0.0, float(max(1, len(path) - 1))

    idx = np.asarray(path, dtype=int)
    angles: list[float] = []
    for start in range(len(path) - window):
        w_a = points[idx[start : start + window]]
        w_b = points[idx[start + 1 : start + 1 + window]]
        pc_a = first_pc(w_a)
        pc_b = first_pc(w_b)
        angles.append(principal_angle_deg(pc_a, pc_b))

    arr = np.asarray(angles, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, float(max(1, len(path) - 1))
    count = float(np.sum(arr > angle_threshold_deg))
    density = float(count / max(1, arr.size))
    return (
        count,
        density,
        float(np.max(arr)),
        float(np.mean(arr)),
        float(np.std(arr)),
        float(max(1, len(path) - 1)),
    )


def fracture_directional_features(
    source_local: np.ndarray,
    target_local: np.ndarray,
    dist_mat: np.ndarray,
    predecessors: np.ndarray,
    points_pca: np.ndarray,
    window: int,
    angle_threshold_deg: float,
) -> np.ndarray:
    feat = np.zeros((source_local.size, 18), dtype=float)
    for i, (s, t) in enumerate(zip(source_local, target_local)):
        src = int(s)
        tgt = int(t)
        p_fwd = PREV.path_nodes_from_predecessor(predecessors, src=src, tgt=tgt)
        p_rev = PREV.path_nodes_from_predecessor(predecessors, src=tgt, tgt=src)

        c_f, d_f, max_f, mean_f, std_f, hops_f = chart_fracture_stats(
            p_fwd, points_pca, window=window, angle_threshold_deg=angle_threshold_deg
        )
        c_r, d_r, max_r, mean_r, std_r, hops_r = chart_fracture_stats(
            p_rev, points_pca, window=window, angle_threshold_deg=angle_threshold_deg
        )

        d_fwd = float(dist_mat[src, tgt]) if np.isfinite(dist_mat[src, tgt]) else float(np.nanmax(dist_mat[np.isfinite(dist_mat)]))
        d_rev = float(dist_mat[tgt, src]) if np.isfinite(dist_mat[tgt, src]) else float(np.nanmax(dist_mat[np.isfinite(dist_mat)]))
        dist_ratio = float(np.log((d_rev + 1e-8) / (d_fwd + 1e-8)))

        feat[i] = np.asarray(
            [
                c_f,
                c_r,
                c_f - c_r,
                d_f,
                d_r,
                d_f - d_r,
                max_f,
                max_r,
                max_f - max_r,
                mean_f,
                mean_r,
                mean_f - mean_r,
                std_f,
                std_r,
                std_f - std_r,
                hops_f,
                hops_r,
                dist_ratio,
            ],
            dtype=float,
        )
    return feat


def swapped_fracture_features(feat: np.ndarray) -> np.ndarray:
    x = np.asarray(feat, dtype=float)
    out = x.copy()
    out[:, 0] = x[:, 1]
    out[:, 1] = x[:, 0]
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

    out[:, 15] = x[:, 16]
    out[:, 16] = x[:, 15]
    out[:, 17] = -x[:, 17]
    return out


def path_order_shuffle_within_bins(
    feat: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.asarray(feat, dtype=float)
    swapped = swapped_fracture_features(x)
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


def run_h132_chart_fracture_diagnostic(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_idx, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H132_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_idx, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue
            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H132_GENE_CAP))
            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(top_genes) & split_edges["target_idx"].isin(top_genes)
            ].copy()
            if split_edges["label"].nunique() < 2:
                continue

            edge_gene_indices, gene_to_local, _, support_dir = PREV.build_symbol_resources(
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

            rng = np.random.default_rng(49_300 + domain_idx * 1000 + split_idx * 100)
            sample_idx = PREV.stratified_index_sample(labels_all, max_n=H132_EDGE_SAMPLE, rng=rng)
            if sample_idx.size < 120:
                continue

            source_local = source_local_all[sample_idx]
            target_local = target_local_all[sample_idx]
            labels = labels_all[sample_idx]

            for layer in H132_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue
                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=49_301 + domain_idx * 1000 + split_idx * 100 + layer,
                )

                _, graph = PREV.build_directed_knn_weighted_graph(
                    points=points_pca,
                    support_dir=support_dir,
                    n_neighbors=H132_NEIGHBORS,
                )
                dist_mat, predecessors = shortest_path(
                    graph,
                    directed=True,
                    unweighted=False,
                    return_predecessors=True,
                )
                finite_dist = dist_mat[np.isfinite(dist_mat)]
                fallback_dist = float(np.max(finite_dist)) if finite_dist.size else 1.0
                dist_h70 = np.asarray(dist_mat, dtype=float).copy()
                dist_h70[~np.isfinite(dist_h70)] = fallback_dist

                h70 = PREV.compute_h70_scores(
                    geodesic=dist_h70,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=[8, 12, 16],
                )

                fracture_feat = fracture_directional_features(
                    source_local=source_local,
                    target_local=target_local,
                    dist_mat=dist_h70,
                    predecessors=predecessors,
                    points_pca=points_pca,
                    window=H132_CHART_WINDOW,
                    angle_threshold_deg=H132_ANGLE_DEG,
                )

                model_feat = np.column_stack([h70, fracture_feat])
                auc_h70 = BASE.safe_auc(labels, h70)
                auc_model = PREV.cv_auc_logit(
                    model_feat,
                    labels,
                    random_state=49_302 + domain_idx * 1000 + split_idx * 100 + layer,
                    n_splits=H132_CV_SPLITS,
                )
                delta = float(auc_model - auc_h70) if np.isfinite(auc_model) and np.isfinite(auc_h70) else float("nan")

                edge_len = dist_h70[source_local, target_local]
                len_bins = BASE.degree_bins(edge_len, max_bins=6)
                deg_sum = PREV.edge_degree_sum(points_pca, H132_NEIGHBORS, source_local, target_local)
                edge_strata = PREV.build_edge_strata(edge_len, deg_sum, max_len_bins=6, max_deg_bins=4)

                null_basis = np.empty(H132_NULL_PERM, dtype=float)
                null_order = np.empty(H132_NULL_PERM, dtype=float)
                null_swap = np.empty(H132_NULL_PERM, dtype=float)
                null_label = np.empty(H132_NULL_PERM, dtype=float)

                for perm_idx in range(H132_NULL_PERM):
                    feat_basis = fracture_feat.copy()
                    basis_cols = [6, 7, 8, 9, 10, 11, 12, 13, 14]
                    feat_basis[:, basis_cols] = PREV.permute_rows_within_strata(
                        feat_basis[:, basis_cols],
                        strata=len_bins,
                        rng=rng,
                    )
                    auc_basis = PREV.cv_auc_logit(
                        np.column_stack([h70, feat_basis]),
                        labels,
                        random_state=49_303 + domain_idx * 10_000 + split_idx * 1000 + layer * 100 + perm_idx,
                        n_splits=H132_CV_SPLITS,
                    )
                    null_basis[perm_idx] = (
                        float(auc_basis - auc_h70) if np.isfinite(auc_basis) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H132",
                            "null_kind": "chart_basis_rotation_shuffle_within_length_bins",
                            "domain": domain,
                            "seed_tag": H132_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_basis[perm_idx]),
                        }
                    )

                    feat_order = path_order_shuffle_within_bins(fracture_feat, strata=len_bins, rng=rng)
                    auc_order = PREV.cv_auc_logit(
                        np.column_stack([h70, feat_order]),
                        labels,
                        random_state=49_304 + domain_idx * 10_000 + split_idx * 1000 + layer * 100 + perm_idx,
                        n_splits=H132_CV_SPLITS,
                    )
                    null_order[perm_idx] = (
                        float(auc_order - auc_h70) if np.isfinite(auc_order) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H132",
                            "null_kind": "path_order_permutation",
                            "domain": domain,
                            "seed_tag": H132_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_order[perm_idx]),
                        }
                    )

                    feat_swap = PREV.permute_rows_within_strata(fracture_feat, strata=edge_strata, rng=rng)
                    auc_swap = PREV.cv_auc_logit(
                        np.column_stack([h70, feat_swap]),
                        labels,
                        random_state=49_305 + domain_idx * 10_000 + split_idx * 1000 + layer * 100 + perm_idx,
                        n_splits=H132_CV_SPLITS,
                    )
                    null_swap[perm_idx] = (
                        float(auc_swap - auc_h70) if np.isfinite(auc_swap) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H132",
                            "null_kind": "endpoint_swap_within_distance_bins",
                            "domain": domain,
                            "seed_tag": H132_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_swap[perm_idx]),
                        }
                    )

                    y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                    auc_lp = PREV.cv_auc_logit(
                        model_feat,
                        y_perm,
                        random_state=49_306 + domain_idx * 10_000 + split_idx * 1000 + layer * 100 + perm_idx,
                        n_splits=H132_CV_SPLITS,
                    )
                    auc_h70_lp = BASE.safe_auc(y_perm, h70)
                    null_label[perm_idx] = (
                        float(auc_lp - auc_h70_lp) if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H132",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H132_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_basis, null_order, null_swap, null_label])
                q95 = finite_q95(all_null)
                p_basis = BASE.empirical_upper_tail_p(delta, null_basis)
                p_order = BASE.empirical_upper_tail_p(delta, null_order)
                p_swap = BASE.empirical_upper_tail_p(delta, null_swap)
                p_label = BASE.empirical_upper_tail_p(delta, null_label)
                p_best = float(np.nanmin(np.asarray([p_basis, p_order, p_swap, p_label], dtype=float)))

                rows.append(
                    {
                        "hypothesis_id": "H132",
                        "domain": domain,
                        "seed_tag": H132_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70": float(auc_h70),
                        "auc_chart_fracture_model": float(auc_model),
                        "delta_vs_h70": float(delta),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95": float(delta - q95),
                        "p_basis_upper": float(p_basis),
                        "p_path_order_upper": float(p_order),
                        "p_endpoint_swap_upper": float(p_swap),
                        "p_label_upper": float(p_label),
                        "p_best_upper": float(p_best),
                        "mean_fracture_count_gap": float(np.mean(fracture_feat[:, 2])),
                        "mean_fracture_density_gap": float(np.mean(fracture_feat[:, 5])),
                        "mean_max_angle_gap": float(np.mean(fracture_feat[:, 8])),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h132_chart_fracture_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h132_chart_fracture_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_df = summarize_by_domain_split(by_row_df)
    summary_path = ITER_DIR / "h132_chart_fracture_domain_summary.csv"
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

    h130 = run_h130_semantic_hardening(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
        trrust_sign_map=trrust_sign_map,
        trrust_tf_out_degree=trrust_tf_out_degree,
    )
    h131 = run_h131_chart_sheaf_alignment(
        gene2go_upper=gene2go_upper,
        trrust_sign_map=trrust_sign_map,
    )
    h132 = run_h132_chart_fracture_diagnostic(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    summary = {
        "iteration": "iter_0049",
        "hypotheses": {
            "H130": h130,
            "H131": h131,
            "H132": h132,
        },
    }
    summary_path = ITER_DIR / "iter0049_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
