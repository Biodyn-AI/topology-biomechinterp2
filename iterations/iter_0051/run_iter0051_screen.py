from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


ITER_DIR = Path("iterations/iter_0051")
ITER_DIR.mkdir(parents=True, exist_ok=True)


# H136 / N680: sectional anisotropy broad-screen.
H136_SEED = "seed42_main"
H136_LAYERS = [7, 11]
H136_SPLITS = ("source_disjoint", "target_disjoint")
H136_GENE_CAP = 170
H136_MIN_GENE_NODES = 120
H136_EDGE_SAMPLE = 230
H136_NEIGHBORS = 12
H136_CV_SPLITS = 4
H136_NULL_PERM = 24

# H137 / N684 rescue-once (major change): correspondence-free cross-model alignment.
H137_SEED = "seed42_main"
H137_LAYERS = [7, 11]
H137_SPLITS = ("source_disjoint", "target_disjoint")
H137_GENE_CAP = 220
H137_MIN_GENES = 110
H137_SIG_NEIGHBORS = 10
H137_NULL_PERM = 32

# H138 / N686: ontology-sheaf hard-slice rescue over signed motif/community backbone.
H138_SEEDS = ["seed42_main", "seed43", "seed44"]
H138_LAYER = 11
H138_SPLITS = ("source_disjoint", "target_disjoint", "dual_axis_disjoint")
H138_GENE_CAP = 220
H138_GENE_CAP_LUNG_DUAL = 260
H138_MIN_GENE_NODES = 120
H138_MIN_GENE_NODES_LUNG_DUAL = 90
H138_EDGE_SAMPLE = 230
H138_EDGE_SAMPLE_LUNG_DUAL = 210
H138_NEIGHBORS = 12
H138_CV_SPLITS = 3
H138_NULL_PERM = 24
H138_CHART_TERMS = 12


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base_iter0051")
PREV = load_module(Path("iterations/iter_0047/run_iter0047_screen.py"), "iter0047_prev_iter0051")

TRRUST_PATH = PREV.TRRUST_PATH


def ensure_required_inputs() -> None:
    required = [
        BASE.DOROTHEA_PATH,
        BASE.GENE2GO_PATH,
        BASE.OMNIPATH_INTERACTIONS_PATH,
        BASE.STRING_CACHE_PATH,
        TRRUST_PATH,
    ]
    for domain, run_map in BASE.SCGPT_RUNS_BY_DOMAIN.items():
        required.append(run_map[H136_SEED] / "cycle1_edge_dataset.tsv")
        required.append(run_map[H136_SEED] / "layer_gene_embeddings.npy")
        required.append(run_map[H137_SEED] / "cycle1_edge_dataset.tsv")
        required.append(run_map[H137_SEED] / "layer_gene_embeddings.npy")
        required.append(BASE.GENEFORMER_EDGE_BY_DOMAIN[domain])
        for seed in H138_SEEDS:
            required.append(run_map[seed] / "cycle1_edge_dataset.tsv")
            required.append(run_map[seed] / "layer_gene_embeddings.npy")

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


def _safe_unit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=float)
    nrm = float(np.linalg.norm(x))
    if nrm < 1e-10:
        return np.zeros_like(x)
    return x / nrm


def local_sectional_geometry(points: np.ndarray, n_neighbors: int) -> dict[str, np.ndarray]:
    n_nodes = points.shape[0]
    k = max(4, min(n_neighbors, n_nodes - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(points)
    _, idx = nbrs.kneighbors(points)

    anis = np.zeros(n_nodes, dtype=float)
    planarity = np.zeros(n_nodes, dtype=float)
    omni = np.zeros(n_nodes, dtype=float)
    pc1 = np.zeros((n_nodes, points.shape[1]), dtype=float)
    degree = np.zeros(n_nodes, dtype=float)

    for i in range(n_nodes):
        neigh = idx[i, 1:].astype(int)
        degree[i] = float(neigh.size)
        if neigh.size < 3:
            continue
        block = points[neigh]
        block = block - block.mean(axis=0, keepdims=True)
        cov = (block.T @ block) / max(1, block.shape[0] - 1)
        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals[::-1], 1e-8, None)
        vecs = evecs[:, ::-1]

        denom = float(evals[0] + evals[1] + evals[2])
        anis[i] = float((evals[0] - evals[1]) / max(1e-8, denom))
        planarity[i] = float((evals[1] - evals[2]) / max(1e-8, evals[0]))
        omni[i] = float((evals[0] * evals[1] * evals[2]) ** (1.0 / 3.0) / max(1e-8, denom))
        pc1[i] = _safe_unit(vecs[:, 0])

    return {
        "anisotropy": anis,
        "planarity": planarity,
        "omnivariance": omni,
        "pc1": pc1,
        "degree": degree,
    }


def sectional_edge_features(
    points: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    support_dir: np.ndarray,
    node_geo: dict[str, np.ndarray],
) -> np.ndarray:
    anis = node_geo["anisotropy"]
    planarity = node_geo["planarity"]
    omni = node_geo["omnivariance"]
    pc1 = node_geo["pc1"]

    src = np.asarray(source_local, dtype=int)
    tgt = np.asarray(target_local, dtype=int)
    edge_vec = points[tgt] - points[src]
    edge_u = np.vstack([_safe_unit(v) for v in edge_vec])

    src_align = np.abs(np.sum(pc1[src] * edge_u, axis=1))
    tgt_align = np.abs(np.sum(pc1[tgt] * (-edge_u), axis=1))
    pair_align = np.abs(np.sum(pc1[src] * pc1[tgt], axis=1))
    support_margin = support_dir[src, tgt] - support_dir[tgt, src]

    anis_src = anis[src]
    anis_tgt = anis[tgt]
    anis_gap = np.abs(anis_src - anis_tgt)
    anis_mean = 0.5 * (anis_src + anis_tgt)

    base = np.column_stack(
        [
            anis_src,
            anis_tgt,
            anis_gap,
            anis_mean,
            0.5 * (planarity[src] + planarity[tgt]),
            0.5 * (omni[src] + omni[tgt]),
            support_margin,
            src_align,
            tgt_align,
            pair_align,
            anis_gap * support_margin,
            anis_mean * pair_align,
        ]
    )
    return np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)


def swapped_sectional_features(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    out = x.copy()
    out[:, 0] = x[:, 1]
    out[:, 1] = x[:, 0]
    out[:, 2] = x[:, 2]
    out[:, 3] = x[:, 3]
    out[:, 4] = x[:, 4]
    out[:, 5] = x[:, 5]
    out[:, 6] = -x[:, 6]
    out[:, 7] = x[:, 8]
    out[:, 8] = x[:, 7]
    out[:, 9] = x[:, 9]
    out[:, 10] = -x[:, 10]
    out[:, 11] = x[:, 11]
    return out


def random_orientation_columns(
    points: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n_nodes, n_dim = points.shape
    rnd = rng.normal(0.0, 1.0, size=(n_nodes, n_dim))
    rnd = np.vstack([_safe_unit(v) for v in rnd])

    src = np.asarray(source_local, dtype=int)
    tgt = np.asarray(target_local, dtype=int)
    edge_vec = points[tgt] - points[src]
    edge_u = np.vstack([_safe_unit(v) for v in edge_vec])

    src_align = np.abs(np.sum(rnd[src] * edge_u, axis=1))
    tgt_align = np.abs(np.sum(rnd[tgt] * (-edge_u), axis=1))
    pair_align = np.abs(np.sum(rnd[src] * rnd[tgt], axis=1))
    return np.column_stack([src_align, tgt_align, pair_align])


def topology_descriptor_from_signature_table(sig_df: pd.DataFrame, n_neighbors: int) -> np.ndarray:
    x = np.asarray(sig_df.to_numpy(dtype=float), dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if x.shape[0] < 6:
        return np.zeros(12, dtype=float)

    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    x = x / np.clip(sd, 1e-6, None)

    n = x.shape[0]
    k = max(3, min(n_neighbors, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(x)
    d_knn, i_knn = nbrs.kneighbors(x)

    neighbors = [set() for _ in range(n)]
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for d, j in zip(d_knn[i, 1:], i_knn[i, 1:]):
            u = int(i)
            v = int(j)
            if u == v:
                continue
            neighbors[u].add(v)
            neighbors[v].add(u)
            w = float(d)
            if adj[u, v] <= 0.0 or w < adj[u, v]:
                adj[u, v] = w
                adj[v, u] = w

    deg = np.asarray([len(nb) for nb in neighbors], dtype=float)
    deg_norm = deg / max(1.0, n - 1.0)
    clust = BASE.local_clustering(neighbors)

    edge_w = adj[np.triu_indices(n, k=1)]
    edge_w = edge_w[edge_w > 0.0]
    if edge_w.size == 0:
        edge_w = np.array([0.0], dtype=float)

    sigma = float(np.median(edge_w))
    sigma = max(sigma, 1e-6)
    sim = np.exp(-(adj**2) / (2.0 * sigma**2))
    sim[adj <= 0.0] = 0.0
    np.fill_diagonal(sim, 0.0)

    deg_w = sim.sum(axis=1)
    inv = 1.0 / np.sqrt(np.clip(deg_w, 1e-8, None))
    lap = np.eye(n) - (inv[:, None] * sim * inv[None, :])
    eigvals = np.linalg.eigvalsh(lap)
    spectral_gap = float(eigvals[1]) if eigvals.size > 1 else 0.0
    trace_norm = float(np.trace(lap) / max(1, n))

    dist_full = cdist(x, x, metric="euclidean")
    mst = minimum_spanning_tree(dist_full)
    mst_vals = np.asarray(mst.data, dtype=float)
    if mst_vals.size == 0:
        mst_vals = np.array([0.0], dtype=float)

    prob_deg = deg + 1.0
    prob_deg = prob_deg / np.sum(prob_deg)
    entropy_deg = float(-np.sum(prob_deg * np.log(np.clip(prob_deg, 1e-12, None))) / np.log(max(2, n)))

    return np.asarray(
        [
            float(np.mean(deg_norm)),
            float(np.std(deg_norm)),
            float(np.mean(clust)),
            float(np.std(clust)),
            float(np.mean(edge_w)),
            float(np.std(edge_w)),
            float(np.mean(mst_vals)),
            float(np.std(mst_vals)),
            float(np.quantile(mst_vals, 0.90)),
            spectral_gap,
            trace_norm,
            entropy_deg,
        ],
        dtype=float,
    )


def descriptor_similarity(a: np.ndarray, b: np.ndarray) -> float:
    xa = np.asarray(a, dtype=float)
    xb = np.asarray(b, dtype=float)
    scale = float(np.median(np.abs(np.concatenate([xa, xb]))))
    scale = max(scale, 1e-3)
    dist2 = float(np.mean(((xa - xb) / scale) ** 2))
    return float(np.exp(-dist2))


def select_chart_terms(
    symbols: list[str],
    gene2go_upper: dict[str, set[str]],
    trrust_gene_set: set[str],
    max_terms: int,
) -> list[str]:
    trrust_terms: dict[str, int] = {}
    global_terms: dict[str, int] = {}

    for sym in symbols:
        terms = gene2go_upper.get(str(sym).upper(), set())
        for term in terms:
            global_terms[term] = global_terms.get(term, 0) + 1
            if str(sym).upper() in trrust_gene_set:
                trrust_terms[term] = trrust_terms.get(term, 0) + 1

    if trrust_terms:
        ranked = sorted(trrust_terms.items(), key=lambda kv: (-kv[1], kv[0]))
    else:
        ranked = sorted(global_terms.items(), key=lambda kv: (-kv[1], kv[0]))
    return [term for term, _ in ranked[:max_terms]]


def assign_chart_indices(
    symbols: list[str],
    chart_terms: list[str],
    gene2go_upper: dict[str, set[str]],
) -> np.ndarray:
    if not symbols:
        return np.zeros(0, dtype=int)
    if not chart_terms:
        return -np.ones(len(symbols), dtype=int)

    term_to_idx = {term: i for i, term in enumerate(chart_terms)}
    out = -np.ones(len(symbols), dtype=int)
    for i, sym in enumerate(symbols):
        terms = gene2go_upper.get(str(sym).upper(), set())
        if not terms:
            continue
        hit = [term_to_idx[t] for t in terms if t in term_to_idx]
        if hit:
            out[i] = int(min(hit))
    return out


def permute_chart_assignments(chart_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(chart_idx, dtype=int).copy()
    known = np.where(out >= 0)[0]
    if known.size > 1:
        vals = out[known].copy()
        out[known] = vals[rng.permutation(vals.size)]
    return out


def build_sheaf_bundle(
    chart_idx: np.ndarray,
    source_local: np.ndarray,
    target_local: np.ndarray,
    edge_margin: np.ndarray,
) -> dict[str, np.ndarray]:
    src_chart = np.asarray(chart_idx[np.asarray(source_local, dtype=int)], dtype=int)
    tgt_chart = np.asarray(chart_idx[np.asarray(target_local, dtype=int)], dtype=int)

    known = chart_idx[chart_idx >= 0]
    n_known = int(known.max()) + 1 if known.size else 0
    unknown_id = n_known
    n_total = unknown_id + 1

    src_idx = np.where(src_chart >= 0, src_chart, unknown_id)
    tgt_idx = np.where(tgt_chart >= 0, tgt_chart, unknown_id)

    pair_counts = np.zeros((n_total, n_total), dtype=float)
    pair_sums = np.zeros((n_total, n_total), dtype=float)
    np.add.at(pair_counts, (src_idx, tgt_idx), 1.0)
    np.add.at(pair_sums, (src_idx, tgt_idx), np.asarray(edge_margin, dtype=float))
    pair_mean = np.divide(pair_sums, pair_counts, out=np.zeros_like(pair_sums), where=pair_counts > 0)

    expected = pair_mean[src_idx, tgt_idx]
    obstruction = np.abs(np.asarray(edge_margin, dtype=float) - expected)
    compatibility = np.exp(-obstruction)
    pair_freq = pair_counts[src_idx, tgt_idx] / max(1.0, float(edge_margin.size))
    boundary = (src_idx != tgt_idx).astype(float)
    unknown = ((src_chart < 0) | (tgt_chart < 0)).astype(float)

    return {
        "boundary": np.asarray(boundary, dtype=float),
        "unknown": np.asarray(unknown, dtype=float),
        "obstruction": np.asarray(obstruction, dtype=float),
        "compatibility": np.asarray(compatibility, dtype=float),
        "pair_freq": np.asarray(pair_freq, dtype=float),
    }


def shuffled_sheaf_sections(
    bundle: dict[str, np.ndarray],
    strata: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    out = {k: np.asarray(v, dtype=float).copy() for k, v in bundle.items()}
    out["obstruction"] = PREV.permute_within_strata(out["obstruction"], strata=strata, rng=rng).astype(float)
    out["compatibility"] = np.exp(-out["obstruction"])
    out["pair_freq"] = PREV.permute_within_strata(out["pair_freq"], strata=strata, rng=rng).astype(float)
    return out


def build_h138_feature_matrix(
    h70: np.ndarray,
    same_community: np.ndarray,
    motif_present: np.ndarray,
    sign_consistent: np.ndarray,
    string_conf: np.ndarray,
    string_high: np.ndarray,
    sheaf_bundle: dict[str, np.ndarray],
) -> np.ndarray:
    boundary = sheaf_bundle["boundary"]
    unknown = sheaf_bundle["unknown"]
    obstruction = sheaf_bundle["obstruction"]
    compatibility = sheaf_bundle["compatibility"]
    pair_freq = sheaf_bundle["pair_freq"]

    return np.column_stack(
        [
            h70,
            same_community,
            motif_present,
            sign_consistent,
            string_conf,
            string_high,
            boundary,
            unknown,
            obstruction,
            compatibility,
            pair_freq,
            obstruction * sign_consistent,
            obstruction * string_conf,
            boundary * same_community,
            h70 * compatibility,
            h70 * sign_consistent * compatibility,
        ]
    )


def run_h136_sectional_anisotropy(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_idx, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        run_dir = run_map[H136_SEED]
        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = BASE.build_split_masks(edge_df)

        for split_idx, split_regime in enumerate(H136_SPLITS):
            split_mask = split_masks.get(split_regime)
            if split_mask is None:
                continue

            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(BASE.select_top_genes(split_edges, gene_cap=H136_GENE_CAP))
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
            if edge_gene_indices.size < H136_MIN_GENE_NODES:
                continue

            source_local_all = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
            target_local_all = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
            labels_all = split_edges["label"].to_numpy(dtype=int)

            rng = np.random.default_rng(51_136 + domain_idx * 1000 + split_idx * 100)
            sample_idx = PREV.stratified_index_sample(labels_all, max_n=H136_EDGE_SAMPLE, rng=rng)
            if sample_idx.size < 120:
                continue

            source_local = source_local_all[sample_idx]
            target_local = target_local_all[sample_idx]
            labels = labels_all[sample_idx]

            for layer in H136_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=51_137 + domain_idx * 1000 + split_idx * 100 + layer,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H136_NEIGHBORS)
                geodesic_w = PREV.confidence_weighted_geodesic(geodesic, support_dir)

                h70 = PREV.compute_h70_scores(
                    geodesic=geodesic_w,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=[8, 12, 16],
                )
                node_geo = local_sectional_geometry(points_pca, n_neighbors=H136_NEIGHBORS)
                sec_feat = sectional_edge_features(
                    points=points_pca,
                    source_local=source_local,
                    target_local=target_local,
                    support_dir=support_dir,
                    node_geo=node_geo,
                )

                model_feat = np.column_stack([h70, sec_feat])
                auc_h70 = BASE.safe_auc(labels, h70)
                auc_model = PREV.cv_auc_logit(
                    model_feat,
                    labels,
                    random_state=51_138 + domain_idx * 1000 + split_idx * 100 + layer,
                    n_splits=H136_CV_SPLITS,
                )
                delta = float(auc_model - auc_h70) if np.isfinite(auc_model) and np.isfinite(auc_h70) else float("nan")

                edge_len = geodesic_w[source_local, target_local]
                len_bins = PREV.quantile_bins(edge_len, n_bins=6)
                deg_sum = PREV.edge_degree_sum(points_pca, H136_NEIGHBORS, source_local, target_local)
                edge_strata = PREV.build_edge_strata(edge_len, deg_sum, max_len_bins=6, max_deg_bins=4)

                null_swap = np.empty(H136_NULL_PERM, dtype=float)
                null_rot = np.empty(H136_NULL_PERM, dtype=float)
                null_label = np.empty(H136_NULL_PERM, dtype=float)

                for perm_idx in range(H136_NULL_PERM):
                    feat_swap = PREV.permute_rows_within_strata(
                        swapped_sectional_features(sec_feat),
                        strata=len_bins,
                        rng=rng,
                    )
                    auc_swap = PREV.cv_auc_logit(
                        np.column_stack([h70, feat_swap]),
                        labels,
                        random_state=51_139 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H136_CV_SPLITS,
                    )
                    null_swap[perm_idx] = (
                        float(auc_swap - auc_h70) if np.isfinite(auc_swap) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H136",
                            "null_kind": "endpoint_swap_within_distance_bins",
                            "domain": domain,
                            "seed_tag": H136_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_swap[perm_idx]),
                        }
                    )

                    orient = random_orientation_columns(
                        points=points_pca,
                        source_local=source_local,
                        target_local=target_local,
                        rng=rng,
                    )
                    feat_rot = np.asarray(sec_feat, dtype=float).copy()
                    feat_rot[:, 7] = orient[:, 0]
                    feat_rot[:, 8] = orient[:, 1]
                    feat_rot[:, 9] = orient[:, 2]
                    feat_rot[:, 11] = feat_rot[:, 3] * feat_rot[:, 9]
                    auc_rot = PREV.cv_auc_logit(
                        np.column_stack([h70, feat_rot]),
                        labels,
                        random_state=51_140 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H136_CV_SPLITS,
                    )
                    null_rot[perm_idx] = (
                        float(auc_rot - auc_h70) if np.isfinite(auc_rot) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H136",
                            "null_kind": "tangent_basis_random_rotation",
                            "domain": domain,
                            "seed_tag": H136_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_rot[perm_idx]),
                        }
                    )

                    y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                    auc_lp = PREV.cv_auc_logit(
                        model_feat,
                        y_perm,
                        random_state=51_141 + domain_idx * 100_000 + split_idx * 10_000 + layer * 100 + perm_idx,
                        n_splits=H136_CV_SPLITS,
                    )
                    auc_h70_lp = BASE.safe_auc(y_perm, h70)
                    null_label[perm_idx] = (
                        float(auc_lp - auc_h70_lp)
                        if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H136",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": H136_SEED,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_swap, null_rot, null_label])
                q95 = finite_q95(all_null)
                p_swap = BASE.empirical_upper_tail_p(delta, null_swap)
                p_rot = BASE.empirical_upper_tail_p(delta, null_rot)
                p_label = BASE.empirical_upper_tail_p(delta, null_label)
                p_best = float(np.nanmin(np.asarray([p_swap, p_rot, p_label], dtype=float)))

                rows.append(
                    {
                        "hypothesis_id": "H136",
                        "domain": domain,
                        "seed_tag": H136_SEED,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70": float(auc_h70),
                        "auc_sectional_model": float(auc_model),
                        "delta_vs_h70": float(delta),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95": float(delta - q95),
                        "p_endpoint_swap_upper": float(p_swap),
                        "p_tangent_rotation_upper": float(p_rot),
                        "p_label_upper": float(p_label),
                        "p_best_upper": float(p_best),
                        "mean_anisotropy_gap": float(np.mean(sec_feat[:, 2])),
                        "mean_pair_alignment": float(np.mean(sec_feat[:, 9])),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h136_sectional_anisotropy_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h136_sectional_anisotropy_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    split_summary_df = summarize_by_domain_split(by_row_df)
    split_summary_path = ITER_DIR / "h136_sectional_anisotropy_domain_split_summary.csv"
    split_summary_df.to_csv(split_summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_vs_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((split_summary_df["mean_delta_vs_h70"] > 0.0).sum()) if not split_summary_df.empty else 0,
        "positive_null_gap_domain_splits": int((split_summary_df["mean_null_gap_q95"] > 0.0).sum())
        if not split_summary_df.empty
        else 0,
        "artifact_paths": {
            "by_domain_split_layer": str(by_row_path),
            "domain_split_summary": str(split_summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h137_correspondence_free_alignment() -> dict[str, object]:
    descriptor_records: list[dict[str, object]] = []

    for domain_idx, domain in enumerate(["immune", "lung", "external_lung"]):
        run_dir = BASE.SCGPT_RUNS_BY_DOMAIN[domain][H137_SEED]
        sc_edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        gf_edge_df = pd.read_csv(BASE.GENEFORMER_EDGE_BY_DOMAIN[domain], sep="\t")

        sc_split_masks = BASE.build_split_masks(sc_edge_df)
        gf_split_masks = BASE.build_split_masks(gf_edge_df)

        for split_idx, split_regime in enumerate(H137_SPLITS):
            sc_mask = sc_split_masks.get(split_regime)
            gf_mask = gf_split_masks.get(split_regime)
            if sc_mask is None or gf_mask is None:
                continue

            sc_split = sc_edge_df.loc[sc_mask].copy()
            gf_split = gf_edge_df.loc[gf_mask].copy()
            if sc_split.empty or gf_split.empty:
                continue

            top_genes = set(BASE.select_top_genes(sc_split, gene_cap=H137_GENE_CAP))
            sc_split = sc_split.loc[
                sc_split["source_idx"].isin(top_genes) & sc_split["target_idx"].isin(top_genes)
            ].copy()
            if sc_split.empty:
                continue

            idx_symbol = BASE.build_symbol_map(sc_split)
            gene_indices = np.array(sorted(idx_symbol.keys()), dtype=int)
            symbols = [idx_symbol[int(i)].upper() for i in gene_indices]
            if len(symbols) < H137_MIN_GENES:
                continue

            gf_sig = BASE.fit_signatures_geneformer(gf_split, symbols=symbols)
            if gf_sig.empty:
                continue
            desc_gf = topology_descriptor_from_signature_table(gf_sig, n_neighbors=H137_SIG_NEIGHBORS)

            for layer in H137_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue
                points = layer_embeddings[layer, gene_indices, :]
                sc_sig = BASE.fit_signatures_scgpt(
                    layer_points=points,
                    symbols=symbols,
                    random_state=51_500 + domain_idx * 1000 + split_idx * 100 + layer,
                    n_neighbors=10,
                )
                if sc_sig.empty:
                    continue
                desc_sc = topology_descriptor_from_signature_table(sc_sig, n_neighbors=H137_SIG_NEIGHBORS)

                descriptor_records.append(
                    {
                        "domain": domain,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_symbols": int(len(symbols)),
                        "desc_sc": desc_sc,
                        "desc_gf": desc_gf,
                    }
                )

    if not descriptor_records:
        out = {
            "rows_tested": 0,
            "mean_delta_auc": float("nan"),
            "positive_domain_count": 0,
            "positive_null_gap_domain_count": 0,
            "immune_mean_null_gap": float("nan"),
            "artifact_paths": {
                "by_domain_split_layer": str(ITER_DIR / "h137_correspondence_free_alignment_by_domain_split_layer.csv"),
                "domain_split_summary": str(ITER_DIR / "h137_correspondence_free_alignment_domain_split_summary.csv"),
                "domain_summary": str(ITER_DIR / "h137_correspondence_free_alignment_domain_summary.csv"),
                "null_summary": str(ITER_DIR / "h137_correspondence_free_alignment_null_summary.csv"),
            },
        }
        pd.DataFrame().to_csv(ITER_DIR / "h137_correspondence_free_alignment_by_domain_split_layer.csv", index=False)
        pd.DataFrame().to_csv(ITER_DIR / "h137_correspondence_free_alignment_domain_split_summary.csv", index=False)
        pd.DataFrame().to_csv(ITER_DIR / "h137_correspondence_free_alignment_domain_summary.csv", index=False)
        pd.DataFrame().to_csv(ITER_DIR / "h137_correspondence_free_alignment_null_summary.csv", index=False)
        return out

    rec_df = pd.DataFrame(descriptor_records)
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for ridx, row in rec_df.iterrows():
        domain = str(row["domain"])
        split_regime = str(row["split_regime"])
        layer = int(row["layer"])
        desc_sc = np.asarray(row["desc_sc"], dtype=float)
        desc_gf = np.asarray(row["desc_gf"], dtype=float)

        same_bucket_other = rec_df.loc[
            (rec_df["split_regime"] == split_regime)
            & (rec_df["layer"] == layer)
            & (rec_df["domain"] != domain)
        ]
        if same_bucket_other.empty:
            continue

        baseline_random_scores = [
            descriptor_similarity(desc_sc, np.asarray(v, dtype=float))
            for v in same_bucket_other["desc_gf"].tolist()
        ]
        random_baseline = float(np.mean(baseline_random_scores))
        align_true = descriptor_similarity(desc_sc, desc_gf)
        delta = float(align_true - random_baseline)

        candidate_pool = rec_df.loc[
            ~(
                (rec_df["domain"] == domain)
                & (rec_df["split_regime"] == split_regime)
                & (rec_df["layer"] == layer)
            )
        ]
        if candidate_pool.empty:
            continue

        rng = np.random.default_rng(51_600 + int(ridx) * 97)
        null_pairing = np.empty(H137_NULL_PERM, dtype=float)
        null_spectrum = np.empty(H137_NULL_PERM, dtype=float)

        for perm_idx in range(H137_NULL_PERM):
            perm_row = candidate_pool.iloc[int(rng.integers(0, candidate_pool.shape[0]))]
            perm_desc = np.asarray(perm_row["desc_gf"], dtype=float)
            sim_pair = descriptor_similarity(desc_sc, perm_desc)
            null_pairing[perm_idx] = float(sim_pair - random_baseline)
            null_rows.append(
                {
                    "hypothesis_id": "H137",
                    "null_kind": "cross_model_pairing_shuffle",
                    "domain": domain,
                    "seed_tag": H137_SEED,
                    "split_regime": split_regime,
                    "layer": int(layer),
                    "perm_idx": int(perm_idx),
                    "null_value": float(null_pairing[perm_idx]),
                }
            )

            perm = rng.permutation(desc_gf.size)
            sim_spec = descriptor_similarity(desc_sc, desc_gf[perm])
            null_spectrum[perm_idx] = float(sim_spec - random_baseline)
            null_rows.append(
                {
                    "hypothesis_id": "H137",
                    "null_kind": "kernel_spectrum_permutation",
                    "domain": domain,
                    "seed_tag": H137_SEED,
                    "split_regime": split_regime,
                    "layer": int(layer),
                    "perm_idx": int(perm_idx),
                    "null_value": float(null_spectrum[perm_idx]),
                }
            )

        all_null = np.concatenate([null_pairing, null_spectrum])
        q95 = finite_q95(all_null)
        p_pair = BASE.empirical_upper_tail_p(delta, null_pairing)
        p_spec = BASE.empirical_upper_tail_p(delta, null_spectrum)
        p_best = float(np.nanmin(np.asarray([p_pair, p_spec], dtype=float)))

        rows.append(
            {
                "hypothesis_id": "H137",
                "domain": domain,
                "seed_tag": H137_SEED,
                "split_regime": split_regime,
                "layer": int(layer),
                "n_symbols": int(row["n_symbols"]),
                "alignment_true": float(align_true),
                "alignment_random_baseline": float(random_baseline),
                "alignment_delta_vs_random": float(delta),
                "delta_vs_h70": float(delta),
                "q95_null_alignment_delta": float(q95),
                "q95_null_delta_auc": float(q95),
                "null_gap_q95": float(delta - q95),
                "p_pairing_upper": float(p_pair),
                "p_spectrum_upper": float(p_spec),
                "p_best_upper": float(p_best),
            }
        )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h137_correspondence_free_alignment_by_domain_split_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h137_correspondence_free_alignment_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    split_summary_df = summarize_by_domain_split(by_row_df)
    split_summary_path = ITER_DIR / "h137_correspondence_free_alignment_domain_split_summary.csv"
    split_summary_df.to_csv(split_summary_path, index=False)

    domain_summary_df = summarize_by_domain(by_row_df)
    domain_summary_path = ITER_DIR / "h137_correspondence_free_alignment_domain_summary.csv"
    domain_summary_df.to_csv(domain_summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["alignment_delta_vs_random"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_count": int((domain_summary_df["mean_delta_vs_h70"] > 0.0).sum()) if not domain_summary_df.empty else 0,
        "positive_null_gap_domain_count": int((domain_summary_df["mean_null_gap_q95"] > 0.0).sum())
        if not domain_summary_df.empty
        else 0,
        "immune_mean_null_gap": float(
            domain_summary_df.loc[domain_summary_df["domain"] == "immune", "mean_null_gap_q95"].mean()
        )
        if not domain_summary_df.empty
        else float("nan"),
        "artifact_paths": {
            "by_domain_split_layer": str(by_row_path),
            "domain_split_summary": str(split_summary_path),
            "domain_summary": str(domain_summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h138_ontology_sheaf_hardening(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
    trrust_sign_map: dict[tuple[str, str], int],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    trrust_gene_set: set[str] = set()
    for s, t in trrust_sign_map.keys():
        trrust_gene_set.add(str(s).upper())
        trrust_gene_set.add(str(t).upper())

    for domain_idx, (domain, run_map) in enumerate(BASE.SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_idx, seed_tag in enumerate(H138_SEEDS):
            run_dir = run_map[seed_tag]
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            split_masks = PREV.build_split_masks_plus(edge_df)

            for split_idx, split_regime in enumerate(H138_SPLITS):
                split_mask = split_masks.get(split_regime)
                if split_mask is None:
                    continue

                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                is_lung_dual = domain == "lung" and split_regime == "dual_axis_disjoint"
                gene_cap = H138_GENE_CAP_LUNG_DUAL if is_lung_dual else H138_GENE_CAP
                min_gene_nodes = H138_MIN_GENE_NODES_LUNG_DUAL if is_lung_dual else H138_MIN_GENE_NODES
                edge_sample = H138_EDGE_SAMPLE_LUNG_DUAL if is_lung_dual else H138_EDGE_SAMPLE

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

                rng = np.random.default_rng(51_700 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100)
                sample_idx = PREV.stratified_index_sample(labels_all, max_n=edge_sample, rng=rng)
                if sample_idx.size < 110:
                    continue

                source_local = source_local_all[sample_idx]
                target_local = target_local_all[sample_idx]
                labels = labels_all[sample_idx]

                if H138_LAYER >= layer_embeddings.shape[0]:
                    continue
                points = layer_embeddings[H138_LAYER, edge_gene_indices, :]
                points_pca = BASE.reduce_points(
                    points,
                    n_components=20,
                    random_state=51_701 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100,
                )
                geodesic = BASE.geodesic_distance_matrix(points_pca, n_neighbors=H138_NEIGHBORS)
                geodesic_w = PREV.confidence_weighted_geodesic(geodesic, support_dir)

                h70 = PREV.compute_h70_scores(
                    geodesic=geodesic_w,
                    support_dir=support_dir,
                    source_local=source_local,
                    target_local=target_local,
                    triangle_k=[8, 12, 16],
                )

                knn_edges = BASE.build_knn_edge_array(points=points_pca, n_neighbors=H138_NEIGHBORS)
                neighbors = BASE.adjacency_neighbors(points_pca.shape[0], knn_edges)
                communities = PREV.label_propagation_communities(neighbors=neighbors, rng=rng, max_iter=20)
                same_community = (communities[source_local] == communities[target_local]).astype(float)

                src_sym = [symbols[i].upper() for i in source_local]
                tgt_sym = [symbols[i].upper() for i in target_local]
                trrust_sign = np.asarray([trrust_sign_map.get((s, t), 0) for s, t in zip(src_sym, tgt_sym)], dtype=int)
                motif_present = (trrust_sign != 0).astype(float)

                edge_margin = support_dir[source_local, target_local] - support_dir[target_local, source_local]
                margin_sign = np.sign(edge_margin)
                sign_consistent = ((trrust_sign * margin_sign) > 0).astype(float)
                string_conf = np.asarray([float(string_map.get((s, t), 0.0)) for s, t in zip(src_sym, tgt_sym)], dtype=float)
                string_high = (string_conf >= 0.70).astype(float)

                chart_terms = select_chart_terms(
                    symbols=symbols,
                    gene2go_upper=gene2go_upper,
                    trrust_gene_set=trrust_gene_set,
                    max_terms=H138_CHART_TERMS,
                )
                chart_idx = assign_chart_indices(symbols=symbols, chart_terms=chart_terms, gene2go_upper=gene2go_upper)
                sheaf = build_sheaf_bundle(
                    chart_idx=chart_idx,
                    source_local=source_local,
                    target_local=target_local,
                    edge_margin=edge_margin,
                )
                feat = build_h138_feature_matrix(
                    h70=h70,
                    same_community=same_community,
                    motif_present=motif_present,
                    sign_consistent=sign_consistent,
                    string_conf=string_conf,
                    string_high=string_high,
                    sheaf_bundle=sheaf,
                )

                auc_h70 = BASE.safe_auc(labels, h70)
                auc_model = PREV.cv_auc_logit(
                    feat,
                    labels,
                    random_state=51_702 + domain_idx * 10_000 + seed_idx * 1000 + split_idx * 100,
                    n_splits=H138_CV_SPLITS,
                )
                delta = float(auc_model - auc_h70) if np.isfinite(auc_model) and np.isfinite(auc_h70) else float("nan")

                edge_len = geodesic_w[source_local, target_local]
                deg_sum = PREV.edge_degree_sum(points_pca, H138_NEIGHBORS, source_local, target_local)
                edge_strata = PREV.build_edge_strata(edge_len, deg_sum, max_len_bins=6, max_deg_bins=4)

                null_chart = np.empty(H138_NULL_PERM, dtype=float)
                null_section = np.empty(H138_NULL_PERM, dtype=float)
                null_label = np.empty(H138_NULL_PERM, dtype=float)

                for perm_idx in range(H138_NULL_PERM):
                    chart_perm = permute_chart_assignments(chart_idx, rng=rng)
                    sheaf_chart = build_sheaf_bundle(
                        chart_idx=chart_perm,
                        source_local=source_local,
                        target_local=target_local,
                        edge_margin=edge_margin,
                    )
                    feat_chart = build_h138_feature_matrix(
                        h70=h70,
                        same_community=same_community,
                        motif_present=motif_present,
                        sign_consistent=sign_consistent,
                        string_conf=string_conf,
                        string_high=string_high,
                        sheaf_bundle=sheaf_chart,
                    )
                    auc_chart = PREV.cv_auc_logit(
                        feat_chart,
                        labels,
                        random_state=51_703 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H138_CV_SPLITS,
                    )
                    null_chart[perm_idx] = (
                        float(auc_chart - auc_h70) if np.isfinite(auc_chart) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H138",
                            "null_kind": "chart_relabel_preserving_chart_size",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H138_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_chart[perm_idx]),
                        }
                    )

                    sheaf_sec = shuffled_sheaf_sections(sheaf, strata=edge_strata, rng=rng)
                    feat_sec = build_h138_feature_matrix(
                        h70=h70,
                        same_community=same_community,
                        motif_present=motif_present,
                        sign_consistent=sign_consistent,
                        string_conf=string_conf,
                        string_high=string_high,
                        sheaf_bundle=sheaf_sec,
                    )
                    auc_sec = PREV.cv_auc_logit(
                        feat_sec,
                        labels,
                        random_state=51_704 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H138_CV_SPLITS,
                    )
                    null_section[perm_idx] = (
                        float(auc_sec - auc_h70) if np.isfinite(auc_sec) and np.isfinite(auc_h70) else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H138",
                            "null_kind": "section_shuffle_within_degree_strata",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H138_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_section[perm_idx]),
                        }
                    )

                    y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                    auc_lp = PREV.cv_auc_logit(
                        feat,
                        y_perm,
                        random_state=51_705 + domain_idx * 100_000 + seed_idx * 10_000 + split_idx * 1000 + perm_idx,
                        n_splits=H138_CV_SPLITS,
                    )
                    auc_h70_lp = BASE.safe_auc(y_perm, h70)
                    null_label[perm_idx] = (
                        float(auc_lp - auc_h70_lp)
                        if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp)
                        else float("nan")
                    )
                    null_rows.append(
                        {
                            "hypothesis_id": "H138",
                            "null_kind": "label_permutation",
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(H138_LAYER),
                            "perm_idx": int(perm_idx),
                            "null_value": float(null_label[perm_idx]),
                        }
                    )

                all_null = np.concatenate([null_chart, null_section, null_label])
                q95 = finite_q95(all_null)
                p_chart = BASE.empirical_upper_tail_p(delta, null_chart)
                p_section = BASE.empirical_upper_tail_p(delta, null_section)
                p_label = BASE.empirical_upper_tail_p(delta, null_label)
                p_best = float(np.nanmin(np.asarray([p_chart, p_section, p_label], dtype=float)))

                rows.append(
                    {
                        "hypothesis_id": "H138",
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(H138_LAYER),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_h70": float(auc_h70),
                        "auc_ontology_sheaf_model": float(auc_model),
                        "delta_vs_h70": float(delta),
                        "q95_null_delta_auc": float(q95),
                        "null_gap_q95": float(delta - q95),
                        "p_chart_upper": float(p_chart),
                        "p_section_upper": float(p_section),
                        "p_label_upper": float(p_label),
                        "p_best_upper": float(p_best),
                        "chart_term_count": int(len(chart_terms)),
                        "mean_sheaf_obstruction": float(np.mean(sheaf["obstruction"])),
                        "mean_chart_boundary_cross": float(np.mean(sheaf["boundary"])),
                    }
                )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["seed_tag", "domain", "split_regime"])
    by_row_path = ITER_DIR / "h138_ontology_sheaf_hardening_by_seed_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "seed_tag", "domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h138_ontology_sheaf_hardening_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    split_summary_df = summarize_by_domain_split(by_row_df)
    split_summary_path = ITER_DIR / "h138_ontology_sheaf_hardening_domain_split_summary.csv"
    split_summary_df.to_csv(split_summary_path, index=False)

    domain_summary_df = summarize_by_domain(by_row_df)
    domain_summary_path = ITER_DIR / "h138_ontology_sheaf_hardening_domain_summary.csv"
    domain_summary_df.to_csv(domain_summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_vs_h70"].mean()) if not by_row_df.empty else float("nan"),
        "positive_domain_splits": int((split_summary_df["mean_delta_vs_h70"] > 0.0).sum()) if not split_summary_df.empty else 0,
        "positive_null_gap_domain_splits": int((split_summary_df["mean_null_gap_q95"] > 0.0).sum())
        if not split_summary_df.empty
        else 0,
        "immune_source_null_gap": float(
            split_summary_df.loc[
                (split_summary_df["domain"] == "immune")
                & (split_summary_df["split_regime"] == "source_disjoint"),
                "mean_null_gap_q95",
            ].mean()
        )
        if not split_summary_df.empty
        else float("nan"),
        "lung_dual_null_gap": float(
            split_summary_df.loc[
                (split_summary_df["domain"] == "lung")
                & (split_summary_df["split_regime"] == "dual_axis_disjoint"),
                "mean_null_gap_q95",
            ].mean()
        )
        if not split_summary_df.empty
        else float("nan"),
        "artifact_paths": {
            "by_seed_domain_split": str(by_row_path),
            "domain_split_summary": str(split_summary_path),
            "domain_summary": str(domain_summary_path),
            "null_summary": str(null_path),
        },
    }


def main() -> None:
    ensure_required_inputs()

    dorothea_map = BASE.load_dorothea_score_map()
    omnipath_pairs = BASE.load_omnipath_pairs()
    gene2go_upper = BASE.load_gene2go_upper()
    string_map = BASE.load_string_scores_from_cache(BASE.STRING_CACHE_PATH)
    trrust_sign_map, _ = PREV.load_trrust_signed_map()

    h136 = run_h136_sectional_anisotropy(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h137 = run_h137_correspondence_free_alignment()
    h138 = run_h138_ontology_sheaf_hardening(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
        trrust_sign_map=trrust_sign_map,
    )

    summary = {
        "iteration": "iter_0051",
        "h136": h136,
        "h137": h137,
        "h138": h138,
    }
    summary_path = ITER_DIR / "iter0051_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
