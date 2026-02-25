from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path


ITER_DIR = Path("iterations/iter_0048")
ITER_DIR.mkdir(parents=True, exist_ok=True)


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base_iter0048")
PREV = load_module(Path("iterations/iter_0047/run_iter0047_screen.py"), "iter0047_prev_iter0048")

TRRUST_PATH = PREV.TRRUST_PATH

# H127 / N641 (module_structure refinement): signed motif-community + STRING + GO hardening.
H127_SEEDS = ["seed42_main"]
H127_LAYER = 11
H127_GENE_CAP = 210
H127_GENE_CAP_LUNG_DUAL = 250
H127_MIN_GENE_NODES = 120
H127_MIN_GENE_NODES_LUNG_DUAL = 90
H127_NEIGHBORS = 12
H127_EDGE_SAMPLE = 240
H127_EDGE_SAMPLE_LUNG_DUAL = 220
H127_CV_SPLITS = 3
H127_NULL_PERM = 32

# H128 / N645 (graph_topology novel family): curvature/community surrogates.
H128_SEED = "seed42_main"
H128_LAYERS = [7, 11]
H128_GENE_CAP = 180
H128_NEIGHBORS = 12
H128_EDGE_SAMPLE = 240
H128_CV_SPLITS = 4
H128_NULL_PERM = 24

# H129 / N634 (manifold rescue): multi-scale torsion spectrum.
H129_SEED = "seed42_main"
H129_LAYERS = [7, 11]
H129_GENE_CAP = 170
H129_EDGE_SAMPLE = 230
H129_CV_SPLITS = 4
H129_NULL_PERM = 24
H129_SCALES = [8, 12, 16]
H129_BASE_SCALE = 12


def ensure_required_inputs() -> None:
    required = [
        BASE.DOROTHEA_PATH,
        BASE.GENE2GO_PATH,
        BASE.OMNIPATH_INTERACTIONS_PATH,
        BASE.STRING_CACHE_PATH,
        TRRUST_PATH,
    ]
    for domain, run_map in BASE.SCGPT_RUNS_BY_DOMAIN.items():
        for seed_tag in set(H127_SEEDS + [H128_SEED, H129_SEED]):
            run_dir = run_map[seed_tag]
            required.append(run_dir / "cycle1_edge_dataset.tsv")
            required.append(run_dir / "layer_gene_embeddings.npy")
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def finite_q95(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, 0.95))


def build_go_pair_features(
    src_sym: list[str],
    tgt_sym: list[str],
    gene2go_upper: dict[str, set[str]],
    anchor_terms: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shared = np.zeros(len(src_sym), dtype=float)
    anchor_pair = np.zeros(len(src_sym), dtype=float)
    overlap_norm = np.zeros(len(src_sym), dtype=float)

    cache_terms: dict[str, set[str]] = {}

    def terms_for(sym: str) -> set[str]:
        key = str(sym).upper()
        if key not in cache_terms:
            cache_terms[key] = set(gene2go_upper.get(key, set()))
        return cache_terms[key]

    for i, (src, tgt) in enumerate(zip(src_sym, tgt_sym)):
        src_terms = terms_for(src)
        tgt_terms = terms_for(tgt)
        inter = src_terms & tgt_terms
        shared[i] = 1.0 if inter else 0.0

        src_anchor = src_terms & anchor_terms
        tgt_anchor = tgt_terms & anchor_terms
        anchor_pair[i] = 1.0 if src_anchor and tgt_anchor else 0.0

        denom = float(np.sqrt(max(1, len(src_terms) * len(tgt_terms))))
        overlap_norm[i] = float(len(inter) / denom)

    return shared, anchor_pair, overlap_norm


def compute_graph_topology_features(
    n_nodes: int,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    neighbors = BASE.adjacency_neighbors(n_nodes=n_nodes, edges=edges)
    degree = np.asarray([len(nb) for nb in neighbors], dtype=float)
    degree_norm = degree / max(1, n_nodes - 1)
    clustering = BASE.local_clustering(neighbors)

    edge_curvature = np.zeros(edges.shape[0], dtype=float)
    for idx, (u, v) in enumerate(edges):
        iu = int(u)
        iv = int(v)
        edge_curvature[idx] = 4.0 - degree[iu] - degree[iv]

    node_curv_sum = np.zeros(n_nodes, dtype=float)
    node_curv_cnt = np.zeros(n_nodes, dtype=float)
    for idx, (u, v) in enumerate(edges):
        iu = int(u)
        iv = int(v)
        c = float(edge_curvature[idx])
        node_curv_sum[iu] += c
        node_curv_sum[iv] += c
        node_curv_cnt[iu] += 1.0
        node_curv_cnt[iv] += 1.0
    node_curvature = node_curv_sum / np.clip(node_curv_cnt, 1.0, None)

    neighbor_deg_mean = np.zeros(n_nodes, dtype=float)
    for i in range(n_nodes):
        neigh = sorted(neighbors[i])
        if neigh:
            neighbor_deg_mean[i] = float(np.mean(degree[np.asarray(neigh, dtype=int)]))
        else:
            neighbor_deg_mean[i] = degree[i]
    assort_resid = degree - neighbor_deg_mean

    return degree_norm, clustering, node_curvature, assort_resid, degree


def aggregate_multiscale_torsion(torsion_stack: np.ndarray) -> np.ndarray:
    # stack shape: [n_edges, n_scales, n_feat(17)]
    mean_feat = np.mean(torsion_stack, axis=1)

    select_idx = np.asarray([0, 5, 8, 11, 14], dtype=int)
    select_slice = torsion_stack[:, :, select_idx]
    std_sel = np.std(select_slice, axis=1)
    slope_sel = select_slice[:, -1, :] - select_slice[:, 0, :]

    scale_consistency = -np.mean(std_sel, axis=1, keepdims=True)
    sign_stability = np.mean(np.sign(torsion_stack[:, :, 14]), axis=1, keepdims=True)

    return np.column_stack([mean_feat, std_sel, slope_sel, scale_consistency, sign_stability])


def permute_scale_order_per_edge(torsion_stack: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(torsion_stack, dtype=float).copy()
    n_edges, n_scales, _ = out.shape
    for i in range(n_edges):
        order = rng.permutation(n_scales)
        out[i] = out[i, order, :]
    return out


def run_h127_signed_string_go_hardening(
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
        for seed_idx, seed_tag in enumerate(H127_SEEDS):
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
                gene_cap = H127_GENE_CAP_LUNG_DUAL if is_lung_dual else H127_GENE_CAP
                min_gene_nodes = H127_MIN_GENE_NODES_LUNG_DUAL if is_lung_dual else H127_MIN_GENE_NODES
                sample_cap = H127_EDGE_SAMPLE_LUNG_DUAL if is_lung_dual else H127_EDGE_SAMPLE

                top_genes = set(BASE.select_top_genes(split_edges, gene_cap=gene_cap))
                split_edges = split_edges.loc[
                    split_edges["source_idx"].isin(top_genes)
                    & split_edges["target_idx"].isin(top_genes)
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

                rng = np.random.default_rng(48_100 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100)
                sample_idx = PREV.stratified_index_sample(labels_all, max_n=sample_cap, rng=rng)
                if sample_idx.size < 110:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                if H127_LAYER >= layer_embeddings.shape[0]:
                    continue

                points = layer_embeddings[H127_LAYER, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=48_101 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H127_NEIGHBORS)
                geodesic_w = PREV.confidence_weighted_geodesic(geodesic, support_dir)

                h70 = PREV.compute_h70_scores(
                    geodesic=geodesic_w,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=[8, 12, 16],
                )

                knn_edges = BASE.build_knn_edge_array(points=points_pca, n_neighbors=H127_NEIGHBORS)
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

                anchor_terms = PREV.select_anchor_go_terms(
                    symbols=symbols,
                    gene2go_upper=gene2go_upper,
                    trrust_gene_set=trrust_gene_set,
                    max_terms=64,
                )
                go_shared, go_anchor_pair, go_overlap_norm = build_go_pair_features(
                    src_sym=src_sym,
                    tgt_sym=tgt_sym,
                    gene2go_upper=gene2go_upper,
                    anchor_terms=anchor_terms,
                )

                feat = np.column_stack(
                    [
                        h70,
                        same_community,
                        motif_present,
                        sign_consistent,
                        string_conf,
                        string_high,
                        go_shared,
                        go_anchor_pair,
                        go_overlap_norm,
                        same_community * motif_present,
                        same_community * sign_consistent,
                        motif_present * string_conf,
                        sign_consistent * string_conf,
                        same_community * string_conf,
                        motif_present * go_shared,
                        sign_consistent * go_anchor_pair,
                        string_conf * go_overlap_norm,
                        h70 * sign_consistent * string_conf,
                        h70 * go_anchor_pair,
                    ]
                )

                auc_h70 = BASE.safe_auc(labels, h70)
                auc_model = PREV.cv_auc_logit(
                    feat,
                    labels,
                    random_state=48_102 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100,
                    n_splits=H127_CV_SPLITS,
                )
                delta = float(auc_model - auc_h70) if np.isfinite(auc_model) and np.isfinite(auc_h70) else float("nan")

                tf_deg = np.asarray([trrust_tf_out_degree.get(s, 0) for s in src_sym], dtype=float)
                tf_bins = BASE.degree_bins(tf_deg, max_bins=4)

                tgt_deg_proxy = np.sum(support_dir > 0.55, axis=0).astype(float)
                tgt_deg_bins = BASE.degree_bins(tgt_deg_proxy[target_local], max_bins=4)
                str_bins = PREV.quantile_bins(string_conf, n_bins=4)
                go_bins = PREV.quantile_bins(go_overlap_norm, n_bins=4)
                motif_strata = (tf_bins * 256 + tgt_deg_bins * 32 + str_bins * 4 + go_bins).astype(int)

                edge_len = geodesic_w[source_local, target_local]
                deg_sum = PREV.edge_degree_sum(points_pca, H127_NEIGHBORS, source_local, target_local)
                edge_strata = PREV.build_edge_strata(edge_len, deg_sum, max_len_bins=6, max_deg_bins=4)

                null_tf_sign = np.empty(H127_NULL_PERM, dtype=float)
                null_motif_decoy = np.empty(H127_NULL_PERM, dtype=float)
                null_string_bin = np.empty(H127_NULL_PERM, dtype=float)
                null_go_membership = np.empty(H127_NULL_PERM, dtype=float)
                null_label = np.empty(H127_NULL_PERM, dtype=float)

                for perm_idx in range(H127_NULL_PERM):
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
                            go_shared,
                            go_anchor_pair,
                            go_overlap_norm,
                            same_community * motif_perm_tf,
                            same_community * sign_cons_perm_tf,
                            motif_perm_tf * string_conf,
                            sign_cons_perm_tf * string_conf,
                            same_community * string_conf,
                            motif_perm_tf * go_shared,
                            sign_cons_perm_tf * go_anchor_pair,
                            string_conf * go_overlap_norm,
                            h70 * sign_cons_perm_tf * string_conf,
                            h70 * go_anchor_pair,
                        ]
                    )
                    auc_tf = PREV.cv_auc_logit(
                        feat_tf,
                        labels,
                        random_state=48_103 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H127_CV_SPLITS,
                    )
                    null_tf_sign[perm_idx] = (
                        float(auc_tf - auc_h70) if np.isfinite(auc_tf) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H127",
                            "null_kind": "tf_identity_preserving_sign_shuffle",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H127_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_tf_sign[perm_idx]),
                        }
                    )

                    motif_perm = PREV.permute_within_strata(motif_present, strata=motif_strata, rng=rng).astype(float)
                    sign_pool = PREV.permute_within_strata(sign_consistent, strata=motif_strata, rng=rng).astype(float)
                    sign_cons_perm = motif_perm * sign_pool
                    feat_decoy = np.column_stack(
                        [
                            h70,
                            same_community,
                            motif_perm,
                            sign_cons_perm,
                            string_conf,
                            string_high,
                            go_shared,
                            go_anchor_pair,
                            go_overlap_norm,
                            same_community * motif_perm,
                            same_community * sign_cons_perm,
                            motif_perm * string_conf,
                            sign_cons_perm * string_conf,
                            same_community * string_conf,
                            motif_perm * go_shared,
                            sign_cons_perm * go_anchor_pair,
                            string_conf * go_overlap_norm,
                            h70 * sign_cons_perm * string_conf,
                            h70 * go_anchor_pair,
                        ]
                    )
                    auc_decoy = PREV.cv_auc_logit(
                        feat_decoy,
                        labels,
                        random_state=48_104 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H127_CV_SPLITS,
                    )
                    null_motif_decoy[perm_idx] = (
                        float(auc_decoy - auc_h70) if np.isfinite(auc_decoy) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H127",
                            "null_kind": "motif_decoy_shuffle_matched_tf_target_degree",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H127_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_motif_decoy[perm_idx]),
                        }
                    )

                    str_perm = PREV.permute_within_strata(string_conf, strata=motif_strata, rng=rng).astype(float)
                    str_high_perm = (str_perm >= 0.70).astype(float)
                    feat_str = np.column_stack(
                        [
                            h70,
                            same_community,
                            motif_present,
                            sign_consistent,
                            str_perm,
                            str_high_perm,
                            go_shared,
                            go_anchor_pair,
                            go_overlap_norm,
                            same_community * motif_present,
                            same_community * sign_consistent,
                            motif_present * str_perm,
                            sign_consistent * str_perm,
                            same_community * str_perm,
                            motif_present * go_shared,
                            sign_consistent * go_anchor_pair,
                            str_perm * go_overlap_norm,
                            h70 * sign_consistent * str_perm,
                            h70 * go_anchor_pair,
                        ]
                    )
                    auc_str = PREV.cv_auc_logit(
                        feat_str,
                        labels,
                        random_state=48_105 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H127_CV_SPLITS,
                    )
                    null_string_bin[perm_idx] = (
                        float(auc_str - auc_h70) if np.isfinite(auc_str) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H127",
                            "null_kind": "string_confidence_bin_permutation_within_degree_strata",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H127_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_string_bin[perm_idx]),
                        }
                    )

                    go_shared_perm = PREV.permute_within_strata(go_shared, strata=motif_strata, rng=rng).astype(float)
                    go_anchor_perm = PREV.permute_within_strata(go_anchor_pair, strata=motif_strata, rng=rng).astype(float)
                    go_overlap_perm = PREV.permute_within_strata(go_overlap_norm, strata=motif_strata, rng=rng).astype(float)
                    feat_go = np.column_stack(
                        [
                            h70,
                            same_community,
                            motif_present,
                            sign_consistent,
                            string_conf,
                            string_high,
                            go_shared_perm,
                            go_anchor_perm,
                            go_overlap_perm,
                            same_community * motif_present,
                            same_community * sign_consistent,
                            motif_present * string_conf,
                            sign_consistent * string_conf,
                            same_community * string_conf,
                            motif_present * go_shared_perm,
                            sign_consistent * go_anchor_perm,
                            string_conf * go_overlap_perm,
                            h70 * sign_consistent * string_conf,
                            h70 * go_anchor_perm,
                        ]
                    )
                    auc_go = PREV.cv_auc_logit(
                        feat_go,
                        labels,
                        random_state=48_106 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H127_CV_SPLITS,
                    )
                    null_go_membership[perm_idx] = (
                        float(auc_go - auc_h70) if np.isfinite(auc_go) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H127",
                            "null_kind": "go_membership_permutation_within_degree_strata",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H127_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_go_membership[perm_idx]),
                        }
                    )

                    y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                    auc_lp = PREV.cv_auc_logit(
                        feat,
                        y_perm,
                        random_state=48_107 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H127_CV_SPLITS,
                    )
                    auc_h70_lp = BASE.safe_auc(y_perm, h70)
                    null_label[perm_idx] = (
                        float(auc_lp - auc_h70_lp)
                        if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H127",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H127_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate(
                    [
                        null_tf_sign,
                        null_motif_decoy,
                        null_string_bin,
                        null_go_membership,
                        null_label,
                    ]
                )
                q95 = finite_q95(all_null)
                p_tf = BASE.empirical_upper_tail_p(delta, null_tf_sign)
                p_decoy = BASE.empirical_upper_tail_p(delta, null_motif_decoy)
                p_string = BASE.empirical_upper_tail_p(delta, null_string_bin)
                p_go = BASE.empirical_upper_tail_p(delta, null_go_membership)
                p_label = BASE.empirical_upper_tail_p(delta, null_label)
                p_best = float(np.nanmin(np.asarray([p_tf, p_decoy, p_string, p_go, p_label], dtype=float)))

                rows.append(
                    {
                        "hypothesis_id": "H127",
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(H127_LAYER),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70": float(auc_h70),
                        "auc_signed_string_go_model": float(auc_model),
                        "delta_vs_h70": float(delta),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95": float(delta - q95),
                        "p_tf_sign_upper": float(p_tf),
                        "p_motif_decoy_upper": float(p_decoy),
                        "p_string_bin_upper": float(p_string),
                        "p_go_permutation_upper": float(p_go),
                        "p_label_upper": float(p_label),
                        "p_best_upper": float(p_best),
                        "motif_coverage": float(np.mean(motif_present)),
                        "string_conf_mean": float(np.mean(string_conf)),
                        "same_community_rate": float(np.mean(same_community)),
                        "go_shared_rate": float(np.mean(go_shared)),
                        "go_anchor_pair_rate": float(np.mean(go_anchor_pair)),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["seed_tag", "domain", "split_regime"])
    by_row_path = ITER_DIR / "h127_signed_string_go_hardening_by_seed_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "seed_tag", "domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h127_signed_string_go_hardening_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_df = PREV.domain_split_summary(by_row_df)
    summary_path = ITER_DIR / "h127_signed_string_go_hardening_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    lung_dual = (
        by_row_df.loc[
            (by_row_df["domain"] == "lung") & (by_row_df["split_regime"] == "dual_axis_disjoint")
        ]
        if not by_row_df.empty
        else pd.DataFrame()
    )

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


def run_h128_graph_topology_surrogate(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_idx, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H128_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_idx, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H128_GENE_CAP))
            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(top_genes)
                & split_edges["target_idx"].isin(top_genes)
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

            rng = np.random.default_rng(48_200 + domain_idx * 1000 + split_idx * 100)
            sample_idx = PREV.stratified_index_sample(labels_all, max_n=H128_EDGE_SAMPLE, rng=rng)
            if sample_idx.size < 120:
                continue

            source_local = source_local_all[sample_idx]
            target_local = target_local_all[sample_idx]
            labels = labels_all[sample_idx]

            for layer in H128_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=48_201 + domain_idx * 1000 + split_idx * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H128_NEIGHBORS)
                geodesic_w = PREV.confidence_weighted_geodesic(geodesic, support_dir)

                h70 = PREV.compute_h70_scores(
                    geodesic=geodesic_w,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=[8, 12, 16],
                )

                knn_edges = BASE.build_knn_edge_array(points=points_pca, n_neighbors=H128_NEIGHBORS)
                if knn_edges.shape[0] < 120:
                    continue
                neighbors = BASE.adjacency_neighbors(points_pca.shape[0], knn_edges)
                degree_norm, clust, node_curv, assort_resid, node_deg = compute_graph_topology_features(
                    n_nodes=edge_gene_indices.size,
                    edges=knn_edges,
                )

                community = PREV.label_propagation_communities(neighbors=neighbors, rng=rng, max_iter=20)
                same_community = (community[source_local] == community[target_local]).astype(float)

                curv_gap = -np.abs(node_curv[source_local] - node_curv[target_local])
                clust_mean = 0.5 * (clust[source_local] + clust[target_local])
                assort_gap = -np.abs(assort_resid[source_local] - assort_resid[target_local])
                degree_balance = -np.abs(degree_norm[source_local] - degree_norm[target_local])

                topo_feat = np.column_stack(
                    [
                        curv_gap,
                        clust_mean,
                        assort_gap,
                        same_community,
                        degree_balance,
                        curv_gap * same_community,
                    ]
                )
                model_feat = np.column_stack(
                    [
                        h70,
                        topo_feat,
                        h70 * curv_gap,
                        h70 * same_community,
                    ]
                )

                auc_h70 = BASE.safe_auc(labels, h70)
                auc_model = PREV.cv_auc_logit(
                    model_feat,
                    labels,
                    random_state=48_202 + domain_idx * 1000 + split_idx * 100 + layer,
                    n_splits=H128_CV_SPLITS,
                )
                delta = float(auc_model - auc_h70) if np.isfinite(auc_model) and np.isfinite(auc_h70) else float("nan")

                node_bins = BASE.degree_bins(node_deg, max_bins=6)
                edge_len = geodesic_w[source_local, target_local]
                edge_deg_eval = node_deg[source_local] + node_deg[target_local]
                edge_strata = PREV.build_edge_strata(edge_len, edge_deg_eval, max_len_bins=6, max_deg_bins=4)
                edge_bin_eval = BASE.degree_bins(edge_deg_eval, max_bins=6)

                null_curv = np.empty(H128_NULL_PERM, dtype=float)
                null_topo = np.empty(H128_NULL_PERM, dtype=float)
                null_rewire = np.empty(H128_NULL_PERM, dtype=float)
                null_label = np.empty(H128_NULL_PERM, dtype=float)

                for perm_idx in range(H128_NULL_PERM):
                    curv_perm = BASE.shuffle_within_bins(node_curv, node_bins, rng)
                    curv_gap_perm = -np.abs(curv_perm[source_local] - curv_perm[target_local])
                    feat_curv = np.column_stack(
                        [
                            h70,
                            curv_gap_perm,
                            clust_mean,
                            assort_gap,
                            same_community,
                            degree_balance,
                            curv_gap_perm * same_community,
                            h70 * curv_gap_perm,
                            h70 * same_community,
                        ]
                    )
                    auc_curv = PREV.cv_auc_logit(
                        feat_curv,
                        labels,
                        random_state=48_203 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H128_CV_SPLITS,
                    )
                    null_curv[perm_idx] = (
                        float(auc_curv - auc_h70) if np.isfinite(auc_curv) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H128",
                            "null_kind": "curvature_shuffle_within_degree_bins",
                            "domain": domain,
                            "seed_tag": H128_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_curv[perm_idx]),
                        }
                    )

                    clust_perm = BASE.shuffle_within_bins(clust, node_bins, rng)
                    assort_perm = BASE.shuffle_within_bins(assort_resid, node_bins, rng)
                    comm_perm = BASE.shuffle_within_bins(community.astype(float), node_bins, rng).astype(int)
                    same_comm_perm = (comm_perm[source_local] == comm_perm[target_local]).astype(float)
                    feat_topo = np.column_stack(
                        [
                            h70,
                            curv_gap,
                            0.5 * (clust_perm[source_local] + clust_perm[target_local]),
                            -np.abs(assort_perm[source_local] - assort_perm[target_local]),
                            same_comm_perm,
                            degree_balance,
                            curv_gap * same_comm_perm,
                            h70 * curv_gap,
                            h70 * same_comm_perm,
                        ]
                    )
                    auc_topo = PREV.cv_auc_logit(
                        feat_topo,
                        labels,
                        random_state=48_204 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H128_CV_SPLITS,
                    )
                    null_topo[perm_idx] = (
                        float(auc_topo - auc_h70) if np.isfinite(auc_topo) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H128",
                            "null_kind": "community_topology_feature_shuffle",
                            "domain": domain,
                            "seed_tag": H128_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_topo[perm_idx]),
                        }
                    )

                    perm_order = BASE.shuffle_within_bins(np.arange(labels.size), edge_bin_eval, rng).astype(int)
                    feat_rewire = np.column_stack(
                        [
                            h70,
                            curv_gap[perm_order],
                            clust_mean[perm_order],
                            assort_gap[perm_order],
                            same_community[perm_order],
                            degree_balance[perm_order],
                            curv_gap[perm_order] * same_community[perm_order],
                            h70 * curv_gap[perm_order],
                            h70 * same_community[perm_order],
                        ]
                    )
                    auc_rw = PREV.cv_auc_logit(
                        feat_rewire,
                        labels,
                        random_state=48_205 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H128_CV_SPLITS,
                    )
                    null_rewire[perm_idx] = (
                        float(auc_rw - auc_h70) if np.isfinite(auc_rw) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H128",
                            "null_kind": "degree_bin_edge_feature_rewiring",
                            "domain": domain,
                            "seed_tag": H128_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_rewire[perm_idx]),
                        }
                    )

                    y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                    auc_lp = PREV.cv_auc_logit(
                        model_feat,
                        y_perm,
                        random_state=48_206 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H128_CV_SPLITS,
                    )
                    auc_h70_lp = BASE.safe_auc(y_perm, h70)
                    null_label[perm_idx] = (
                        float(auc_lp - auc_h70_lp)
                        if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H128",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H128_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_curv, null_topo, null_rewire, null_label])
                q95 = finite_q95(all_null)
                p_curv = BASE.empirical_upper_tail_p(delta, null_curv)
                p_topo = BASE.empirical_upper_tail_p(delta, null_topo)
                p_rewire = BASE.empirical_upper_tail_p(delta, null_rewire)
                p_label = BASE.empirical_upper_tail_p(delta, null_label)
                p_best = float(np.nanmin(np.asarray([p_curv, p_topo, p_rewire, p_label], dtype=float)))

                rows.append(
                    {
                        "hypothesis_id": "H128",
                        "domain": domain,
                        "seed_tag": H128_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70": float(auc_h70),
                        "auc_graph_topology_model": float(auc_model),
                        "delta_vs_h70": float(delta),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95": float(delta - q95),
                        "p_curvature_upper": float(p_curv),
                        "p_topology_upper": float(p_topo),
                        "p_rewire_upper": float(p_rewire),
                        "p_label_upper": float(p_label),
                        "p_best_upper": float(p_best),
                        "mean_node_curvature": float(np.mean(node_curv)),
                        "mean_local_clustering": float(np.mean(clust)),
                        "mean_same_community": float(np.mean(same_community)),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h128_graph_topology_surrogate_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h128_graph_topology_surrogate_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_df = PREV.domain_split_summary(by_row_df)
    summary_path = ITER_DIR / "h128_graph_topology_surrogate_domain_summary.csv"
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


def run_h129_multiscale_torsion_spectrum(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_idx, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H129_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_idx, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H129_GENE_CAP))
            split_edges = split_edges.loc[
                split_edges["source_idx"].isin(top_genes)
                & split_edges["target_idx"].isin(top_genes)
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

            rng = np.random.default_rng(48_300 + domain_idx * 1000 + split_idx * 100)
            sample_idx = PREV.stratified_index_sample(labels_all, max_n=H129_EDGE_SAMPLE, rng=rng)
            if sample_idx.size < 120:
                continue

            source_local = source_local_all[sample_idx]
            target_local = target_local_all[sample_idx]
            labels = labels_all[sample_idx]

            for layer in H129_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=48_301 + domain_idx * 1000 + split_idx * 100 + layer,
                )

                torsion_by_scale: list[np.ndarray] = []
                h70_by_scale: dict[int, np.ndarray] = {}
                geodesic_w_by_scale: dict[int, np.ndarray] = {}

                for scale in H129_SCALES:
                    geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=scale)
                    geodesic_w = PREV.confidence_weighted_geodesic(geodesic, support_dir)
                    geodesic_w_by_scale[scale] = geodesic_w

                    h70_scale = PREV.compute_h70_scores(
                        geodesic=geodesic_w,
                        support_dir=support_dir,
                        source_local=source_local,
                        target_local=target_local,
                        triangle_k=[8, 12, 16],
                    )
                    h70_by_scale[scale] = h70_scale

                    _, directed_graph = PREV.build_directed_knn_weighted_graph(
                        points=points_pca,
                        support_dir=support_dir,
                        n_neighbors=scale,
                    )
                    dist_dir, pred_dir = shortest_path(
                        directed_graph,
                        directed=True,
                        unweighted=False,
                        return_predecessors=True,
                    )
                    torsion_feat = PREV.torsion_directional_features(
                        source_local=source_local,
                        target_local=target_local,
                        dist_mat=dist_dir,
                        predecessors=pred_dir,
                        support_dir=support_dir,
                        points_pca=points_pca,
                    )
                    torsion_by_scale.append(torsion_feat)

                torsion_stack = np.stack(torsion_by_scale, axis=1)
                agg_feat = aggregate_multiscale_torsion(torsion_stack)
                h70 = h70_by_scale[H129_BASE_SCALE]
                model_feat = np.column_stack([h70, agg_feat])

                auc_h70 = BASE.safe_auc(labels, h70)
                auc_model = PREV.cv_auc_logit(
                    model_feat,
                    labels,
                    random_state=48_302 + domain_idx * 1000 + split_idx * 100 + layer,
                    n_splits=H129_CV_SPLITS,
                )
                delta = float(auc_model - auc_h70) if np.isfinite(auc_model) and np.isfinite(auc_h70) else float("nan")

                edge_len = geodesic_w_by_scale[H129_BASE_SCALE][source_local, target_local]
                len_bins = PREV.quantile_bins(edge_len, n_bins=6)
                deg_sum = PREV.edge_degree_sum(points_pca, H129_BASE_SCALE, source_local, target_local)
                edge_strata = PREV.build_edge_strata(edge_len, deg_sum, max_len_bins=6, max_deg_bins=4)

                null_reverse = np.empty(H129_NULL_PERM, dtype=float)
                null_swap = np.empty(H129_NULL_PERM, dtype=float)
                null_scale = np.empty(H129_NULL_PERM, dtype=float)
                null_label = np.empty(H129_NULL_PERM, dtype=float)

                for perm_idx in range(H129_NULL_PERM):
                    rev_list: list[np.ndarray] = []
                    for arr in torsion_by_scale:
                        rev_list.append(PREV.path_reversal_within_bins(arr, strata=len_bins, rng=rng))
                    feat_rev = np.column_stack([h70, aggregate_multiscale_torsion(np.stack(rev_list, axis=1))])
                    auc_rev = PREV.cv_auc_logit(
                        feat_rev,
                        labels,
                        random_state=48_303 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H129_CV_SPLITS,
                    )
                    null_reverse[perm_idx] = (
                        float(auc_rev - auc_h70) if np.isfinite(auc_rev) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H129",
                            "null_kind": "path_reversal_within_length_bins",
                            "domain": domain,
                            "seed_tag": H129_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_reverse[perm_idx]),
                        }
                    )

                    swap_list: list[np.ndarray] = []
                    for arr in torsion_by_scale:
                        swapped = PREV.swapped_torsion_features(arr)
                        swapped = PREV.permute_rows_within_strata(swapped, strata=len_bins, rng=rng)
                        swap_list.append(swapped)
                    feat_swap = np.column_stack([h70, aggregate_multiscale_torsion(np.stack(swap_list, axis=1))])
                    auc_swap = PREV.cv_auc_logit(
                        feat_swap,
                        labels,
                        random_state=48_304 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H129_CV_SPLITS,
                    )
                    null_swap[perm_idx] = (
                        float(auc_swap - auc_h70) if np.isfinite(auc_swap) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H129",
                            "null_kind": "endpoint_swap_within_distance_bins",
                            "domain": domain,
                            "seed_tag": H129_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_swap[perm_idx]),
                        }
                    )

                    stack_perm = permute_scale_order_per_edge(torsion_stack, rng=rng)
                    feat_scale = np.column_stack([h70, aggregate_multiscale_torsion(stack_perm)])
                    auc_scale = PREV.cv_auc_logit(
                        feat_scale,
                        labels,
                        random_state=48_305 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H129_CV_SPLITS,
                    )
                    null_scale[perm_idx] = (
                        float(auc_scale - auc_h70)
                        if np.isfinite(auc_scale) and np.isfinite(auc_h70)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H129",
                            "null_kind": "scale_order_permutation",
                            "domain": domain,
                            "seed_tag": H129_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_scale[perm_idx]),
                        }
                    )

                    y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                    auc_lp = PREV.cv_auc_logit(
                        model_feat,
                        y_perm,
                        random_state=48_306 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H129_CV_SPLITS,
                    )
                    auc_h70_lp = BASE.safe_auc(y_perm, h70)
                    null_label[perm_idx] = (
                        float(auc_lp - auc_h70_lp)
                        if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H129",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H129_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_reverse, null_swap, null_scale, null_label])
                q95 = finite_q95(all_null)
                p_reverse = BASE.empirical_upper_tail_p(delta, null_reverse)
                p_swap = BASE.empirical_upper_tail_p(delta, null_swap)
                p_scale = BASE.empirical_upper_tail_p(delta, null_scale)
                p_label = BASE.empirical_upper_tail_p(delta, null_label)
                p_best = float(np.nanmin(np.asarray([p_reverse, p_swap, p_scale, p_label], dtype=float)))

                rows.append(
                    {
                        "hypothesis_id": "H129",
                        "domain": domain,
                        "seed_tag": H129_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70": float(auc_h70),
                        "auc_multiscale_torsion_model": float(auc_model),
                        "delta_vs_h70": float(delta),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95": float(delta - q95),
                        "p_path_reversal_upper": float(p_reverse),
                        "p_endpoint_swap_upper": float(p_swap),
                        "p_scale_order_upper": float(p_scale),
                        "p_label_upper": float(p_label),
                        "p_best_upper": float(p_best),
                        "mean_scale_consistency": float(np.mean(agg_feat[:, -2])),
                        "mean_sign_stability": float(np.mean(agg_feat[:, -1])),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h129_multiscale_torsion_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h129_multiscale_torsion_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_df = PREV.domain_split_summary(by_row_df)
    summary_path = ITER_DIR / "h129_multiscale_torsion_domain_summary.csv"
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
    trrust_sign_map, trrust_tf_out_degree = PREV.load_trrust_signed_map()

    h127 = run_h127_signed_string_go_hardening(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
        trrust_sign_map=trrust_sign_map,
        trrust_tf_out_degree=trrust_tf_out_degree,
    )
    h128 = run_h128_graph_topology_surrogate(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h129 = run_h129_multiscale_torsion_spectrum(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    summary = {
        "iteration": "iter_0048",
        "h127": h127,
        "h128": h128,
        "h129": h129,
    }
    summary_path = ITER_DIR / "iter0048_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
