from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


ITER_DIR = Path("iterations/iter_0053")
ITER_DIR.mkdir(parents=True, exist_ok=True)


# H142 / Q01: adaptive-neighborhood rescue on known strict-margin fail slices.
H142_SEEDS = ["seed42_main", "seed43", "seed44"]
H142_LAYER = 11
H142_FAIL_SLICES = [
    ("external_lung", "source_disjoint"),
    ("external_lung", "target_disjoint"),
    ("external_lung", "dual_axis_disjoint"),
    ("lung", "source_disjoint"),
    ("lung", "target_disjoint"),
    ("immune", "dual_axis_disjoint"),
]
H142_NEIGHBOR_MIN = 8
H142_NEIGHBOR_MAX = 20
H142_CV_SPLITS = 3
H142_NULL_PERM = 8

# H143 / Q07: TF-module Betti-curve cross-model pilot.
H143_SEED = "seed42_main"
H143_LAYER = 11
H143_SPLITS = ("source_disjoint", "target_disjoint")
H143_GENE_CAP = 220
H143_MIN_SHARED = 100
H143_MIN_MODULE_GENES = 6
H143_MIN_MODULES = 4
H143_MAX_MODULES = 12
H143_NULL_PERM = 24
H143_EPS_QUANTILES = (0.15, 0.30, 0.45, 0.60, 0.75)

# H144 / Q11: depth motif token screen.
H144_SEED = "seed42_main"
H144_LAYERS = [0, 3, 7, 11]
H144_SPLITS = ("source_disjoint", "target_disjoint", "dual_axis_disjoint")
H144_NEIGHBORS = 12
H144_CV_SPLITS = 3
H144_LABEL_NULL_PERM = 16


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(Path("iterations/iter_0028/run_iter0028_screen.py"), "iter0028_base_iter0053")
PREV = load_module(Path("iterations/iter_0047/run_iter0047_screen.py"), "iter0047_prev_iter0053")
ITER51 = load_module(Path("iterations/iter_0051/run_iter0051_screen.py"), "iter0051_prev_iter0053")
ITER52 = load_module(Path("iterations/iter_0052/run_iter0052_screen.py"), "iter0052_prev_iter0053")

TRRUST_PATH = PREV.TRRUST_PATH


def ensure_required_inputs() -> None:
    """Fail fast when source artifacts for this bounded screening packet are missing."""
    required = [
        BASE.DOROTHEA_PATH,
        BASE.GENE2GO_PATH,
        BASE.OMNIPATH_INTERACTIONS_PATH,
        BASE.STRING_CACHE_PATH,
        TRRUST_PATH,
    ]

    for domain, run_map in BASE.SCGPT_RUNS_BY_DOMAIN.items():
        for seed_tag in sorted(set(H142_SEEDS + [H143_SEED, H144_SEED])):
            run_dir = run_map[seed_tag]
            required.append(run_dir / "cycle1_edge_dataset.tsv")
            required.append(run_dir / "layer_gene_embeddings.npy")
        required.append(BASE.GENEFORMER_EDGE_BY_DOMAIN[domain])

    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def slice_sampling_params(domain: str, split_regime: str) -> tuple[int, int, int]:
    """
    Slice-specific data budget.

    The fail-slice rescue intentionally relaxes selection thresholds for historically sparse
    slices so we can test whether adaptive neighborhoods recover strict margins rather than
    failing purely due to low evaluation coverage.
    """
    gene_cap = 170
    min_gene_nodes = 120
    edge_sample = 220

    if domain == "external_lung" and split_regime == "dual_axis_disjoint":
        gene_cap = 280
        min_gene_nodes = 75
        edge_sample = 170
    elif domain == "external_lung":
        gene_cap = 220
        min_gene_nodes = 95
        edge_sample = 200
    elif domain == "immune" and split_regime == "dual_axis_disjoint":
        gene_cap = 210
        min_gene_nodes = 90
        edge_sample = 195

    return gene_cap, min_gene_nodes, edge_sample


def choose_adaptive_k(points: np.ndarray, k_min: int, k_max: int) -> int:
    """
    Pick neighborhood size from geometric density heterogeneity.

    Higher distance CV indicates heterogeneous local density and receives smaller k.
    Larger gene pools receive a modest positive k adjustment.
    """
    n_nodes = int(points.shape[0])
    if n_nodes <= max(4, k_min + 1):
        return int(max(4, min(k_max, n_nodes - 1)))

    probe_k = int(max(4, min(7, n_nodes - 1)))
    nbrs = NearestNeighbors(n_neighbors=probe_k + 1, metric="euclidean")
    nbrs.fit(points)
    dists, _ = nbrs.kneighbors(points)
    radial = np.asarray(dists[:, -1], dtype=float)

    mean_radial = float(np.mean(radial))
    std_radial = float(np.std(radial))
    cv = std_radial / max(1e-8, mean_radial)

    size_shift = (float(n_nodes) - 160.0) / 80.0
    raw_k = 14.0 - 10.0 * cv + 2.0 * size_shift
    return int(np.clip(int(round(raw_k)), k_min, k_max))


def summarize_h142_by_domain_split(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (domain, split_regime), group in df.groupby(["domain", "split_regime"], sort=True):
        rows.append(
            {
                "domain": str(domain),
                "split_regime": str(split_regime),
                "n_rows": int(group.shape[0]),
                "mean_adaptive_k": float(group["adaptive_k"].mean()),
                "mean_delta_vs_h70": float(group["delta_vs_h70"].mean()),
                "mean_strict_margin": float(group["strict_margin"].mean()),
                "fraction_strict_positive": float((group["strict_margin"] > 0.0).mean()),
                "fraction_directional_positive": float((group["delta_vs_h70"] > 0.0).mean()),
                "combined_fisher_p_best": ITER52.finite_fisher(group["p_best_upper"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows).sort_values(["domain", "split_regime"])


def summarize_h142_by_domain(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for domain, group in df.groupby("domain", sort=True):
        rows.append(
            {
                "domain": str(domain),
                "n_rows": int(group.shape[0]),
                "mean_adaptive_k": float(group["adaptive_k"].mean()),
                "mean_delta_vs_h70": float(group["delta_vs_h70"].mean()),
                "mean_strict_margin": float(group["strict_margin"].mean()),
                "fraction_strict_positive": float((group["strict_margin"] > 0.0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("domain")


def run_h142_adaptive_fail_slice(
    *,
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for slice_idx, (domain, split_regime) in enumerate(H142_FAIL_SLICES):
        for seed_idx, seed_tag in enumerate(H142_SEEDS):
            gene_cap, min_gene_nodes, edge_sample = slice_sampling_params(domain, split_regime)
            bundle = ITER52.prepare_split_bundle(
                domain=domain,
                seed_tag=seed_tag,
                split_regime=split_regime,
                gene_cap=gene_cap,
                min_gene_nodes=min_gene_nodes,
                edge_sample=edge_sample,
                rng_seed=53_142 + slice_idx * 10_000 + seed_idx * 100,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )
            if bundle is None:
                continue

            layer_embeddings = bundle["layer_embeddings"]
            edge_gene_indices = np.asarray(bundle["edge_gene_indices"], dtype=int)
            if H142_LAYER >= layer_embeddings.shape[0]:
                continue

            points = layer_embeddings[H142_LAYER, edge_gene_indices, :]
            points_pca = BASE.reduce_points(
                points,
                n_components=20,
                random_state=53_143 + slice_idx * 10_000 + seed_idx * 100,
            )
            adaptive_k = choose_adaptive_k(
                points=points_pca,
                k_min=H142_NEIGHBOR_MIN,
                k_max=H142_NEIGHBOR_MAX,
            )

            packet = ITER52.compute_sectional_packet(
                bundle=bundle,
                layer=H142_LAYER,
                n_neighbors=adaptive_k,
                pca_random_state=53_144 + slice_idx * 10_000 + seed_idx * 100,
            )
            if packet is None:
                continue

            h70 = np.asarray(packet["h70"], dtype=float)
            sec_feat = np.asarray(packet["sec_feat"], dtype=float)
            labels = np.asarray(packet["labels"], dtype=int)
            len_bins = np.asarray(packet["len_bins"], dtype=int)
            edge_strata = np.asarray(packet["edge_strata"], dtype=int)

            auc_h70, auc_model, delta = ITER52.evaluate_delta(
                h70=h70,
                sec_feat=sec_feat,
                labels=labels,
                random_state=53_145 + slice_idx * 10_000 + seed_idx * 100,
                n_splits=H142_CV_SPLITS,
            )

            rng = np.random.default_rng(53_146 + slice_idx * 10_000 + seed_idx * 100)
            null_swap = np.empty(H142_NULL_PERM, dtype=float)
            null_feat = np.empty(H142_NULL_PERM, dtype=float)
            null_label = np.empty(H142_NULL_PERM, dtype=float)

            for perm_idx in range(H142_NULL_PERM):
                feat_swap = PREV.permute_rows_within_strata(
                    ITER51.swapped_sectional_features(sec_feat),
                    strata=len_bins,
                    rng=rng,
                )
                auc_swap = PREV.cv_auc_logit(
                    np.column_stack([h70, feat_swap]),
                    labels,
                    random_state=53_147
                    + slice_idx * 100_000
                    + seed_idx * 10_000
                    + perm_idx,
                    n_splits=H142_CV_SPLITS,
                )
                null_swap[perm_idx] = (
                    float(auc_swap - auc_h70) if np.isfinite(auc_swap) and np.isfinite(auc_h70) else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H142",
                        "null_kind": "endpoint_swap_within_distance_bins",
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(H142_LAYER),
                        "adaptive_k": int(adaptive_k),
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_swap[perm_idx]),
                    }
                )

                feat_shuffle = PREV.permute_rows_within_strata(sec_feat, strata=len_bins, rng=rng)
                auc_shuffle = PREV.cv_auc_logit(
                    np.column_stack([h70, feat_shuffle]),
                    labels,
                    random_state=53_148
                    + slice_idx * 100_000
                    + seed_idx * 10_000
                    + perm_idx,
                    n_splits=H142_CV_SPLITS,
                )
                null_feat[perm_idx] = (
                    float(auc_shuffle - auc_h70)
                    if np.isfinite(auc_shuffle) and np.isfinite(auc_h70)
                    else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H142",
                        "null_kind": "sectional_row_shuffle",
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(H142_LAYER),
                        "adaptive_k": int(adaptive_k),
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_feat[perm_idx]),
                    }
                )

                y_perm = BASE.stratified_shuffle(labels, edge_strata, rng).astype(int)
                auc_lp = PREV.cv_auc_logit(
                    np.column_stack([h70, sec_feat]),
                    y_perm,
                    random_state=53_149
                    + slice_idx * 100_000
                    + seed_idx * 10_000
                    + perm_idx,
                    n_splits=H142_CV_SPLITS,
                )
                auc_h70_lp = BASE.safe_auc(y_perm, h70)
                null_label[perm_idx] = (
                    float(auc_lp - auc_h70_lp)
                    if np.isfinite(auc_lp) and np.isfinite(auc_h70_lp)
                    else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H142",
                        "null_kind": "label_permutation",
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(H142_LAYER),
                        "adaptive_k": int(adaptive_k),
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_label[perm_idx]),
                    }
                )

            q95_swap = ITER52.finite_q95(null_swap)
            q95_feat = ITER52.finite_q95(null_feat)
            q95_label = ITER52.finite_q95(null_label)
            strict_q95 = float(np.nanmax(np.asarray([q95_swap, q95_feat, q95_label], dtype=float)))

            p_swap = BASE.empirical_upper_tail_p(delta, null_swap)
            p_feat = BASE.empirical_upper_tail_p(delta, null_feat)
            p_label = BASE.empirical_upper_tail_p(delta, null_label)
            p_best = float(np.nanmin(np.asarray([p_swap, p_feat, p_label], dtype=float)))

            rows.append(
                {
                    "hypothesis_id": "H142",
                    "domain": domain,
                    "seed_tag": seed_tag,
                    "split_regime": split_regime,
                    "layer": int(H142_LAYER),
                    "adaptive_k": int(adaptive_k),
                    "n_edges_eval": int(labels.size),
                    "n_positive_eval": int(labels.sum()),
                    "auc_h70": float(auc_h70),
                    "auc_adaptive_model": float(auc_model),
                    "delta_vs_h70": float(delta),
                    "q95_endpoint_swap": float(q95_swap),
                    "q95_row_shuffle": float(q95_feat),
                    "q95_label": float(q95_label),
                    "strict_q95_max": float(strict_q95),
                    "strict_margin": float(delta - strict_q95),
                    "strict_positive": int((delta - strict_q95) > 0.0),
                    "p_endpoint_swap_upper": float(p_swap),
                    "p_row_shuffle_upper": float(p_feat),
                    "p_label_upper": float(p_label),
                    "p_best_upper": float(p_best),
                }
            )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime", "seed_tag"])
    by_row_path = ITER_DIR / "h142_adaptive_fail_slice_by_seed_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "seed_tag", "perm_idx"])
    null_path = ITER_DIR / "h142_adaptive_fail_slice_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    split_summary_df = summarize_h142_by_domain_split(by_row_df)
    split_summary_path = ITER_DIR / "h142_adaptive_fail_slice_domain_split_summary.csv"
    split_summary_df.to_csv(split_summary_path, index=False)

    domain_summary_df = summarize_h142_by_domain(by_row_df)
    domain_summary_path = ITER_DIR / "h142_adaptive_fail_slice_domain_summary.csv"
    domain_summary_df.to_csv(domain_summary_path, index=False)

    ext_lung_dual_rows = int(
        by_row_df.loc[
            (by_row_df["domain"] == "external_lung")
            & (by_row_df["split_regime"] == "dual_axis_disjoint")
        ].shape[0]
    ) if not by_row_df.empty else 0

    fail_slice_non_negative = int((split_summary_df["mean_strict_margin"] >= 0.0).sum()) if not split_summary_df.empty else 0

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_vs_h70"].mean()) if not by_row_df.empty else float("nan"),
        "mean_strict_margin": float(by_row_df["strict_margin"].mean()) if not by_row_df.empty else float("nan"),
        "strict_positive_rows": int((by_row_df["strict_margin"] > 0.0).sum()) if not by_row_df.empty else 0,
        "strict_non_negative_fail_slices": int(fail_slice_non_negative),
        "external_lung_dual_axis_rows": int(ext_lung_dual_rows),
        "artifact_paths": {
            "by_seed_domain_split": str(by_row_path),
            "domain_split_summary": str(split_summary_path),
            "domain_summary": str(domain_summary_path),
            "null_summary": str(null_path),
        },
    }


def load_trrust_modules(trrust_path: Path) -> dict[str, set[str]]:
    """Build TF -> target set map used as biological anchors for module definitions."""
    trrust_df = pd.read_csv(
        trrust_path,
        sep="\t",
        header=None,
        names=["tf", "target", "sign", "pmid"],
    )
    modules: dict[str, set[str]] = {}
    for row in trrust_df.itertuples(index=False):
        tf = str(row.tf).upper()
        target = str(row.target).upper()
        modules.setdefault(tf, set()).add(target)
    return modules


def betti_curve_signature(distance_matrix: np.ndarray, eps_quantiles: tuple[float, ...]) -> np.ndarray:
    """
    Coarse persistence-like signature from filtration over pairwise distances.

    For each epsilon quantile, we derive graph Betti-0 (component count) and Betti-1
    (cycle rank) on the thresholded neighborhood graph.
    """
    d = np.asarray(distance_matrix, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or d.shape[0] < 4:
        return np.zeros(2 * len(eps_quantiles), dtype=float)

    n = int(d.shape[0])
    tri = d[np.triu_indices(n, k=1)]
    finite = tri[np.isfinite(tri)]
    if finite.size < 4:
        return np.zeros(2 * len(eps_quantiles), dtype=float)

    eps_grid = np.quantile(finite, np.asarray(eps_quantiles, dtype=float))
    feats: list[float] = []

    for eps in eps_grid:
        adj = ((d <= float(eps)) & (d > 0.0)).astype(np.int8)
        graph = csr_matrix(adj)
        n_components, _ = connected_components(graph, directed=False, return_labels=True)
        n_edges = float(adj.sum() / 2.0)
        b1 = max(0.0, n_edges - float(n) + float(n_components))

        feats.append(float(n_components) / max(1.0, float(n)))
        feats.append(float(b1) / max(1.0, float(n)))

    return np.asarray(feats, dtype=float)


def centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < 1e-10:
        return 0.0
    return float(np.dot(x, y) / denom)


def safe_graph_distance_matrix(n_nodes: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """
    Undirected shortest-path distances with finite fallback for disconnected pairs.
    """
    base = np.full((n_nodes, n_nodes), np.inf, dtype=float)
    np.fill_diagonal(base, 0.0)
    for u, v in edges:
        if u == v:
            continue
        base[u, v] = 1.0
        base[v, u] = 1.0

    dist = shortest_path(base, directed=False, unweighted=False)
    finite = dist[np.isfinite(dist) & (dist > 0.0)]
    fill = float(np.max(finite) + 1.0) if finite.size else float(max(2, n_nodes // 2))
    out = np.asarray(dist, dtype=float)
    out[~np.isfinite(out)] = fill
    return out


def summarize_h143_by_domain(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for domain, group in df.groupby("domain", sort=True):
        rows.append(
            {
                "domain": str(domain),
                "n_rows": int(group.shape[0]),
                "mean_modules": float(group["n_modules"].mean()),
                "mean_alignment_delta": float(group["delta_vs_perm_baseline"].mean()),
                "mean_null_gap_q95": float(group["null_gap_q95"].mean()),
                "fraction_null_gap_positive": float((group["null_gap_q95"] > 0.0).mean()),
                "combined_fisher_p_best": ITER52.finite_fisher(group["p_best_upper"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows).sort_values("domain")


def run_h143_tf_module_cross_model() -> dict[str, object]:
    trrust_modules = load_trrust_modules(TRRUST_PATH)

    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_idx, domain in enumerate(["immune", "lung", "external_lung"]):
        sc_run_dir = BASE.SCGPT_RUNS_BY_DOMAIN[domain][H143_SEED]
        sc_edge_df = pd.read_csv(sc_run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        sc_embeddings = np.load(sc_run_dir / "layer_gene_embeddings.npy", mmap_mode="r")

        gf_edge_df = pd.read_csv(BASE.GENEFORMER_EDGE_BY_DOMAIN[domain], sep="\t")

        sc_split_masks = PREV.build_split_masks_plus(sc_edge_df)
        gf_split_masks = PREV.build_split_masks_plus(gf_edge_df)

        for split_idx, split_regime in enumerate(H143_SPLITS):
            sc_mask = sc_split_masks.get(split_regime)
            gf_mask = gf_split_masks.get(split_regime)
            if sc_mask is None or gf_mask is None:
                continue

            sc_split = sc_edge_df.loc[sc_mask].copy()
            gf_split = gf_edge_df.loc[gf_mask].copy()
            if sc_split.empty or gf_split.empty:
                continue

            top_genes = set(BASE.select_top_genes(sc_split, gene_cap=H143_GENE_CAP))
            sc_split = sc_split.loc[
                sc_split["source_idx"].isin(top_genes) & sc_split["target_idx"].isin(top_genes)
            ].copy()
            if sc_split.empty:
                continue

            sc_symbol_map = BASE.build_symbol_map(sc_split)
            symbol_to_idx = {
                str(sym).upper(): int(idx)
                for idx, sym in sc_symbol_map.items()
                if int(idx) in top_genes
            }

            gf_symbols = set(gf_split["source"].astype(str).str.upper()) | set(
                gf_split["target"].astype(str).str.upper()
            )
            shared_symbols = sorted(set(symbol_to_idx.keys()) & gf_symbols)
            if len(shared_symbols) < H143_MIN_SHARED:
                continue

            if H143_LAYER >= sc_embeddings.shape[0]:
                continue

            shared_indices = np.asarray([symbol_to_idx[sym] for sym in shared_symbols], dtype=int)
            sc_points = sc_embeddings[H143_LAYER, shared_indices, :]
            sc_points_pca = BASE.reduce_points(
                sc_points,
                n_components=20,
                random_state=53_210 + domain_idx * 100 + split_idx,
            )
            sc_dist = cdist(sc_points_pca, sc_points_pca, metric="euclidean")

            symbol_to_local = {sym: i for i, sym in enumerate(shared_symbols)}
            gf_pos = gf_split.loc[
                gf_split["label"].astype(int) == 1,
                ["source", "target"],
            ].copy()
            gf_edges: list[tuple[int, int]] = []
            for row in gf_pos.itertuples(index=False):
                s = str(row.source).upper()
                t = str(row.target).upper()
                if s not in symbol_to_local or t not in symbol_to_local:
                    continue
                i = int(symbol_to_local[s])
                j = int(symbol_to_local[t])
                if i != j:
                    gf_edges.append((i, j))
            gf_dist = safe_graph_distance_matrix(n_nodes=len(shared_symbols), edges=gf_edges)

            module_defs: list[tuple[str, list[int]]] = []
            for tf, targets in trrust_modules.items():
                module_symbols = {tf} | set(targets)
                kept = [symbol_to_local[s] for s in shared_symbols if s in module_symbols]
                if len(kept) >= H143_MIN_MODULE_GENES:
                    module_defs.append((tf, kept))

            module_defs = sorted(module_defs, key=lambda x: (-len(x[1]), x[0]))[:H143_MAX_MODULES]
            if len(module_defs) < H143_MIN_MODULES:
                continue

            sc_signatures: list[np.ndarray] = []
            gf_signatures: list[np.ndarray] = []
            module_sizes: list[int] = []
            module_ids: list[str] = []

            for tf, local_ids in module_defs:
                idx = np.asarray(sorted(set(local_ids)), dtype=int)
                if idx.size < H143_MIN_MODULE_GENES:
                    continue

                sig_sc = betti_curve_signature(sc_dist[np.ix_(idx, idx)], eps_quantiles=H143_EPS_QUANTILES)
                sig_gf = betti_curve_signature(gf_dist[np.ix_(idx, idx)], eps_quantiles=H143_EPS_QUANTILES)

                sc_signatures.append(sig_sc)
                gf_signatures.append(sig_gf)
                module_sizes.append(int(idx.size))
                module_ids.append(tf)

            if len(sc_signatures) < H143_MIN_MODULES:
                continue

            true_sims = [
                centered_cosine(sig_sc, sig_gf)
                for sig_sc, sig_gf in zip(sc_signatures, gf_signatures)
            ]
            true_mean = float(np.mean(true_sims))

            rng = np.random.default_rng(53_220 + domain_idx * 100 + split_idx)
            null_module_perm = np.empty(H143_NULL_PERM, dtype=float)
            null_anchor_shuffle = np.empty(H143_NULL_PERM, dtype=float)

            n_modules = len(sc_signatures)
            n_shared = len(shared_symbols)

            for perm_idx in range(H143_NULL_PERM):
                perm = rng.permutation(n_modules)
                sims_perm = [
                    centered_cosine(sc_signatures[i], gf_signatures[int(perm[i])])
                    for i in range(n_modules)
                ]
                null_module_perm[perm_idx] = float(np.mean(sims_perm))
                null_rows.append(
                    {
                        "hypothesis_id": "H143",
                        "null_kind": "module_permutation",
                        "domain": domain,
                        "seed_tag": H143_SEED,
                        "split_regime": split_regime,
                        "layer": int(H143_LAYER),
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_module_perm[perm_idx]),
                    }
                )

                sims_anchor: list[float] = []
                for mod_idx, size in enumerate(module_sizes):
                    draw = rng.choice(n_shared, size=size, replace=False)
                    gf_sig_rand = betti_curve_signature(
                        gf_dist[np.ix_(draw, draw)],
                        eps_quantiles=H143_EPS_QUANTILES,
                    )
                    sims_anchor.append(centered_cosine(sc_signatures[mod_idx], gf_sig_rand))

                null_anchor_shuffle[perm_idx] = float(np.mean(sims_anchor))
                null_rows.append(
                    {
                        "hypothesis_id": "H143",
                        "null_kind": "anchor_gene_set_shuffle",
                        "domain": domain,
                        "seed_tag": H143_SEED,
                        "split_regime": split_regime,
                        "layer": int(H143_LAYER),
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_anchor_shuffle[perm_idx]),
                    }
                )

            perm_baseline = float(np.mean(null_module_perm))
            delta = float(true_mean - perm_baseline)

            delta_null_perm = null_module_perm - perm_baseline
            delta_null_anchor = null_anchor_shuffle - perm_baseline
            q95 = ITER52.finite_q95(np.concatenate([delta_null_perm, delta_null_anchor]))

            p_perm = BASE.empirical_upper_tail_p(delta, delta_null_perm)
            p_anchor = BASE.empirical_upper_tail_p(delta, delta_null_anchor)
            p_best = float(np.nanmin(np.asarray([p_perm, p_anchor], dtype=float)))

            rows.append(
                {
                    "hypothesis_id": "H143",
                    "domain": domain,
                    "seed_tag": H143_SEED,
                    "split_regime": split_regime,
                    "layer": int(H143_LAYER),
                    "n_shared_symbols": int(len(shared_symbols)),
                    "n_modules": int(len(sc_signatures)),
                    "module_ids": ";".join(module_ids),
                    "alignment_true_mean": float(true_mean),
                    "alignment_perm_baseline": float(perm_baseline),
                    "delta_vs_perm_baseline": float(delta),
                    "q95_null_delta": float(q95),
                    "null_gap_q95": float(delta - q95),
                    "p_module_permutation_upper": float(p_perm),
                    "p_anchor_shuffle_upper": float(p_anchor),
                    "p_best_upper": float(p_best),
                }
            )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime"])
    by_row_path = ITER_DIR / "h143_tf_module_cross_model_by_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h143_tf_module_cross_model_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    domain_summary_df = summarize_h143_by_domain(by_row_df)
    domain_summary_path = ITER_DIR / "h143_tf_module_cross_model_domain_summary.csv"
    domain_summary_df.to_csv(domain_summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_alignment_delta": float(by_row_df["delta_vs_perm_baseline"].mean()) if not by_row_df.empty else float("nan"),
        "positive_null_gap_rows": int((by_row_df["null_gap_q95"] > 0.0).sum()) if not by_row_df.empty else 0,
        "positive_null_gap_domains": int((domain_summary_df["mean_null_gap_q95"] > 0.0).sum()) if not domain_summary_df.empty else 0,
        "artifact_paths": {
            "by_domain_split": str(by_row_path),
            "domain_summary": str(domain_summary_path),
            "null_summary": str(null_path),
        },
    }


def summarize_h144_by_domain(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for domain, group in df.groupby("domain", sort=True):
        rows.append(
            {
                "domain": str(domain),
                "n_rows": int(group.shape[0]),
                "fraction_motif_hit": float(group["motif_hit"].mean()),
                "mean_motif_excess": float(group["motif_excess_vs_permuted_order"].mean()),
                "mean_layer11_delta": float(group["layer11_delta_vs_h70"].mean()),
                "mean_strict_margin": float(group["strict_margin_label_null"].mean()),
                "fraction_strict_positive": float((group["strict_margin_label_null"] > 0.0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("domain")


def motif_hit_from_sequence(d0: float, d3: float, d7: float, d11: float) -> int:
    rise_early = d3 > d0
    rise_mid = d7 > d3
    stabilize_late = d11 >= (d7 - 0.003)
    return int(rise_early and rise_mid and stabilize_late)


def run_h144_depth_motif_tokens(
    *,
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_idx, domain in enumerate(["immune", "lung", "external_lung"]):
        for split_idx, split_regime in enumerate(H144_SPLITS):
            gene_cap, min_gene_nodes, edge_sample = slice_sampling_params(domain, split_regime)
            bundle = ITER52.prepare_split_bundle(
                domain=domain,
                seed_tag=H144_SEED,
                split_regime=split_regime,
                gene_cap=gene_cap,
                min_gene_nodes=min_gene_nodes,
                edge_sample=edge_sample,
                rng_seed=53_310 + domain_idx * 1000 + split_idx * 100,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )
            if bundle is None:
                continue

            per_layer_delta: dict[int, float] = {}
            layer11_packet: dict[str, np.ndarray] | None = None

            for layer in H144_LAYERS:
                packet = ITER52.compute_sectional_packet(
                    bundle=bundle,
                    layer=layer,
                    n_neighbors=H144_NEIGHBORS,
                    pca_random_state=53_311 + domain_idx * 1000 + split_idx * 100 + layer,
                )
                if packet is None:
                    per_layer_delta = {}
                    break

                h70 = np.asarray(packet["h70"], dtype=float)
                sec_feat = np.asarray(packet["sec_feat"], dtype=float)
                labels = np.asarray(packet["labels"], dtype=int)

                _, _, delta = ITER52.evaluate_delta(
                    h70=h70,
                    sec_feat=sec_feat,
                    labels=labels,
                    random_state=53_312 + domain_idx * 1000 + split_idx * 100 + layer,
                    n_splits=H144_CV_SPLITS,
                )
                per_layer_delta[layer] = float(delta)

                if layer == 11:
                    layer11_packet = packet

            if set(H144_LAYERS) - set(per_layer_delta.keys()):
                continue
            if layer11_packet is None:
                continue

            d0 = float(per_layer_delta[0])
            d3 = float(per_layer_delta[3])
            d7 = float(per_layer_delta[7])
            d11 = float(per_layer_delta[11])

            motif_hit = motif_hit_from_sequence(d0, d3, d7, d11)
            motif_strength = float((d3 - d0) + (d7 - d3) - abs(d11 - d7))

            perm_hits: list[int] = []
            seq_vals = [d0, d3, d7, d11]
            for perm_idx, perm in enumerate(itertools.permutations(range(4))):
                if perm == (0, 1, 2, 3):
                    continue
                pd0, pd3, pd7, pd11 = [seq_vals[i] for i in perm]
                hit = motif_hit_from_sequence(pd0, pd3, pd7, pd11)
                perm_hits.append(hit)
                null_rows.append(
                    {
                        "hypothesis_id": "H144",
                        "null_kind": "layer_order_permutation",
                        "domain": domain,
                        "seed_tag": H144_SEED,
                        "split_regime": split_regime,
                        "layer": -1,
                        "perm_idx": int(perm_idx),
                        "null_value": float(hit),
                    }
                )

            motif_null_mean = float(np.mean(perm_hits)) if perm_hits else float("nan")
            motif_excess = float(motif_hit - motif_null_mean) if np.isfinite(motif_null_mean) else float("nan")

            h70_11 = np.asarray(layer11_packet["h70"], dtype=float)
            sec_feat_11 = np.asarray(layer11_packet["sec_feat"], dtype=float)
            labels_11 = np.asarray(layer11_packet["labels"], dtype=int)
            edge_strata_11 = np.asarray(layer11_packet["edge_strata"], dtype=int)

            auc_h70_11 = BASE.safe_auc(labels_11, h70_11)
            null_label = np.empty(H144_LABEL_NULL_PERM, dtype=float)
            rng = np.random.default_rng(53_313 + domain_idx * 1000 + split_idx * 100)
            for perm_idx in range(H144_LABEL_NULL_PERM):
                y_perm = BASE.stratified_shuffle(labels_11, edge_strata_11, rng).astype(int)
                auc_model_perm = PREV.cv_auc_logit(
                    np.column_stack([h70_11, sec_feat_11]),
                    y_perm,
                    random_state=53_314 + domain_idx * 10_000 + split_idx * 1000 + perm_idx,
                    n_splits=H144_CV_SPLITS,
                )
                auc_h70_perm = BASE.safe_auc(y_perm, h70_11)
                null_label[perm_idx] = (
                    float(auc_model_perm - auc_h70_perm)
                    if np.isfinite(auc_model_perm) and np.isfinite(auc_h70_perm)
                    else float("nan")
                )
                null_rows.append(
                    {
                        "hypothesis_id": "H144",
                        "null_kind": "label_permutation_layer11",
                        "domain": domain,
                        "seed_tag": H144_SEED,
                        "split_regime": split_regime,
                        "layer": 11,
                        "perm_idx": int(perm_idx),
                        "null_value": float(null_label[perm_idx]),
                    }
                )

            q95_label = ITER52.finite_q95(null_label)
            p_label = BASE.empirical_upper_tail_p(d11, null_label)

            rows.append(
                {
                    "hypothesis_id": "H144",
                    "domain": domain,
                    "seed_tag": H144_SEED,
                    "split_regime": split_regime,
                    "layers": "0;3;7;11",
                    "layer0_delta_vs_h70": float(d0),
                    "layer3_delta_vs_h70": float(d3),
                    "layer7_delta_vs_h70": float(d7),
                    "layer11_delta_vs_h70": float(d11),
                    "mean_layer_delta_vs_h70": float(np.mean([d0, d3, d7, d11])),
                    "motif_hit": int(motif_hit),
                    "motif_strength": float(motif_strength),
                    "motif_null_mean_hit": float(motif_null_mean),
                    "motif_excess_vs_permuted_order": float(motif_excess),
                    "q95_label_null_layer11": float(q95_label),
                    "strict_margin_label_null": float(d11 - q95_label),
                    "p_label_upper": float(p_label),
                    "n_edges_eval": int(labels_11.size),
                    "n_positive_eval": int(labels_11.sum()),
                    "auc_h70_layer11": float(auc_h70_11),
                }
            )

    by_row_df = pd.DataFrame(rows)
    if not by_row_df.empty:
        by_row_df = by_row_df.sort_values(["domain", "split_regime"])
    by_row_path = ITER_DIR / "h144_depth_motif_tokens_by_domain_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df = null_df.sort_values(["null_kind", "domain", "split_regime", "perm_idx"])
    null_path = ITER_DIR / "h144_depth_motif_tokens_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    domain_summary_df = summarize_h144_by_domain(by_row_df)
    domain_summary_path = ITER_DIR / "h144_depth_motif_tokens_domain_summary.csv"
    domain_summary_df.to_csv(domain_summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_motif_excess": float(by_row_df["motif_excess_vs_permuted_order"].mean()) if not by_row_df.empty else float("nan"),
        "motif_hit_rows": int(by_row_df["motif_hit"].sum()) if not by_row_df.empty else 0,
        "strict_positive_rows": int((by_row_df["strict_margin_label_null"] > 0.0).sum()) if not by_row_df.empty else 0,
        "artifact_paths": {
            "by_domain_split": str(by_row_path),
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

    h142 = run_h142_adaptive_fail_slice(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    h143 = run_h143_tf_module_cross_model()

    h144 = run_h144_depth_motif_tokens(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )

    summary = {
        "iteration": "iter_0053",
        "h142": h142,
        "h143": h143,
        "h144": h144,
    }
    summary_path = ITER_DIR / "iter0053_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
