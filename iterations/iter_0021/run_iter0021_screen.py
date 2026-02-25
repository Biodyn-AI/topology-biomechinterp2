from __future__ import annotations

import io
import json
import pickle
from pathlib import Path

import dionysus as d
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from scipy.stats import combine_pvalues
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModel


ITER_DIR = Path("iterations/iter_0021")
ITER_DIR.mkdir(parents=True, exist_ok=True)

SCGPT_RUNS_BY_DOMAIN: dict[str, dict[str, Path]] = {
    "immune": {
        "seed42_main": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle4_immune_main"
        ),
        "seed43": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle4_immune_seed43"
        ),
        "seed44": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle4_immune_seed44"
        ),
    },
    "lung": {
        "seed42_main": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle6_lung_main"
        ),
        "seed43": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle6_lung_seed43"
        ),
        "seed44": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle6_lung_seed44"
        ),
    },
    "external_lung": {
        "seed42_main": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle7_external_lung_main"
        ),
        "seed43": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle7_external_lung_seed43"
        ),
        "seed44": Path(
            "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
            "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle7_external_lung_seed44"
        ),
    },
}

GENEFORMER_EDGE_BY_DOMAIN = {
    "immune": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/"
        "cycle12_geneformer_immune_bootstrap/geneformer_edge_dataset.tsv"
    ),
    "lung": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/"
        "cycle12_geneformer_lung_bootstrap/geneformer_edge_dataset.tsv"
    ),
    "external_lung": Path(
        "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
        "subproject_38_geometric_residual_stream_interpretability/implementation/outputs/"
        "cycle12_geneformer_external_lung_bootstrap/geneformer_edge_dataset.tsv"
    ),
}

H31_UTILITY_PATH = Path("iterations/iter_0016/h31_diffusion_incremental_by_seed_layer_split.csv")

TRRUST_PATH = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
    "single_cell_mechinterp/external/networks/trrust_human.tsv"
)
DOROTHEA_PATH = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
    "single_cell_mechinterp/external/networks/dorothea_human.tsv"
)
GENE2GO_PATH = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
    "single_cell_mechinterp/data/perturb/gene2go_all.pkl"
)
OMNIPATH_INTERACTIONS_PATH = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/"
    "network_inference/data/omnipath_interactions.tsv"
)
STRING_CACHE_PATH = Path("iterations/iter_0020/h43_string_network_api.tsv")

# H46 configuration (refinement of H44 with support-weighted filtering).
H46_LAYERS = [0, 3, 7, 11]
H46_GENE_CAP = 150
H46_KNN = 10
H46_NULL_PERM = 36

# H47 configuration (new method: bifiltration-like cycle score vs distance-only ablation).
H47_LAYERS = [0, 3, 7, 11]
H47_GENE_CAP = 180
H47_KNN = 10
H47_NULL_PERM = 36

# H48 configuration (cheap cross-model motif overlap screen).
H48_LAYERS = [7, 11]
H48_TOP_K = [50, 100, 200]
H48_NULL_PERM = 90


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    if np.unique(scores).size < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def empirical_upper_tail_p(observed: float, null_values: np.ndarray) -> float:
    if not np.isfinite(observed):
        return float("nan")
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float((1 + np.sum(values >= observed)) / (values.size + 1))


def safe_fisher_p(pvals: np.ndarray) -> float:
    values = np.asarray(pvals, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    values = np.clip(values, 1e-12, 1.0)
    _, pvalue = combine_pvalues(values, method="fisher")
    return float(pvalue)


def build_split_masks(edge_df: pd.DataFrame) -> dict[str, np.ndarray]:
    source_threshold = float(edge_df["source_idx"].median())
    target_threshold = float(edge_df["target_idx"].median())
    return {
        "source_disjoint": edge_df["source_idx"].to_numpy(dtype=float) <= source_threshold,
        "target_disjoint": edge_df["target_idx"].to_numpy(dtype=float) > target_threshold,
    }


def select_top_genes(edge_df: pd.DataFrame, gene_cap: int) -> list[int]:
    counts: dict[int, int] = {}
    for col in ["source_idx", "target_idx"]:
        for value, count in edge_df[col].value_counts().items():
            key = int(value)
            counts[key] = counts.get(key, 0) + int(count)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [gene_idx for gene_idx, _ in ranked[:gene_cap]]


def reduce_points(points: np.ndarray, n_components: int, random_state: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    points = points - points.mean(axis=0, keepdims=True)
    max_comp = min(n_components, points.shape[0] - 1, points.shape[1])
    if max_comp < 4:
        raise RuntimeError(f"Too few PCA components: {max_comp}")
    return PCA(
        n_components=max_comp,
        svd_solver="randomized",
        random_state=random_state,
    ).fit_transform(points)


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


def zigzag_h1_total_lifetime(
    n_vertices: int,
    source_edges: set[tuple[int, int]],
    target_edges: set[tuple[int, int]],
) -> float:
    simplices: list[d.Simplex] = []
    intervals_by_key: dict[tuple[int, ...], list[float]] = {}

    for v in range(n_vertices):
        simplex = d.Simplex([int(v)], 0.0)
        simplices.append(simplex)
        intervals_by_key[(int(v),)] = [0.0, 3.0]

    union_edges = source_edges | target_edges
    for edge in sorted(union_edges):
        u, v = int(edge[0]), int(edge[1])
        simplex = d.Simplex([u, v], 1.0)
        simplices.append(simplex)
        in_source = edge in source_edges
        in_target = edge in target_edges
        if in_source and in_target:
            interval = [0.0, 3.0]
        elif in_source:
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
        return 0.0

    h1 = diagrams[1]
    lifetimes: list[float] = []
    for point in h1:
        birth = float(point.birth)
        death = float(point.death)
        if not np.isfinite(birth) or not np.isfinite(death):
            continue
        life = max(0.0, death - birth)
        lifetimes.append(life)

    if len(lifetimes) == 0:
        return 0.0
    return float(np.sum(np.asarray(lifetimes, dtype=float)))


def load_dorothea_score_map() -> dict[tuple[str, str], int]:
    dorothea = pd.read_csv(DOROTHEA_PATH, sep="\t")
    dorothea["source"] = dorothea["source"].astype(str).str.upper()
    dorothea["target"] = dorothea["target"].astype(str).str.upper()
    confidence_map = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
    dorothea["confidence_score"] = (
        dorothea["confidence"].astype(str).str.upper().map(confidence_map).fillna(0).astype(int)
    )
    best = dorothea.groupby(["source", "target"], as_index=False)["confidence_score"].max()
    return {
        (str(row.source), str(row.target)): int(row.confidence_score)
        for row in best.itertuples(index=False)
    }


def load_trrust_pairs() -> set[tuple[str, str]]:
    trrust = pd.read_csv(
        TRRUST_PATH,
        sep="\t",
        header=None,
        names=["source", "target", "regulation", "pmid"],
    )
    trrust["source"] = trrust["source"].astype(str).str.upper()
    trrust["target"] = trrust["target"].astype(str).str.upper()
    return set(zip(trrust["source"], trrust["target"]))


def load_omnipath_pairs() -> set[tuple[str, str]]:
    omni = pd.read_csv(OMNIPATH_INTERACTIONS_PATH, sep="\t")
    required = {"source_genesymbol", "target_genesymbol"}
    if not required.issubset(omni.columns):
        return set()
    source = omni["source_genesymbol"].astype(str).str.upper()
    target = omni["target_genesymbol"].astype(str).str.upper()
    return set(zip(source, target))


def load_gene2go_upper() -> dict[str, set[str]]:
    with open(GENE2GO_PATH, "rb") as handle:
        raw = pickle.load(handle)
    result: dict[str, set[str]] = {}
    for gene, terms in raw.items():
        if not isinstance(gene, str):
            continue
        gene_upper = gene.upper()
        if gene_upper not in result:
            result[gene_upper] = set()
        if isinstance(terms, (set, list, tuple)):
            for term in terms:
                if isinstance(term, str) and term.startswith("GO:"):
                    result[gene_upper].add(term)
    return result


def load_string_scores_from_cache(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    df = pd.read_csv(io.StringIO(text), sep="\t")
    required = {"preferredName_A", "preferredName_B", "score"}
    if not required.issubset(df.columns):
        return {}

    mapping: dict[tuple[str, str], float] = {}
    for row in df.itertuples(index=False):
        src = str(getattr(row, "preferredName_A")).upper()
        tgt = str(getattr(row, "preferredName_B")).upper()
        score = float(getattr(row, "score"))
        if not np.isfinite(score):
            continue
        value = float(np.clip(score, 0.0, 1.0))
        ab = (src, tgt)
        ba = (tgt, src)
        mapping[ab] = max(value, mapping.get(ab, 0.0))
        mapping[ba] = max(value, mapping.get(ba, 0.0))
    return mapping


def support_score_pair(
    src: str,
    tgt: str,
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> float:
    src_u = src.upper()
    tgt_u = tgt.upper()
    dorothea_norm = float(dorothea_map.get((src_u, tgt_u), 0)) / 4.0
    omnipath_support = float((src_u, tgt_u) in omnipath_pairs)
    string_score = float(string_map.get((src_u, tgt_u), 0.0))

    go_src = gene2go_upper.get(src_u, set())
    go_tgt = gene2go_upper.get(tgt_u, set())
    union = len(go_src | go_tgt)
    go_jaccard = float(len(go_src & go_tgt) / union) if union > 0 else 0.0

    # We intentionally use a simple biologically anchored weighted sum.
    # This keeps screening fast and preserves directional support semantics.
    return float(0.35 * dorothea_norm + 0.35 * string_score + 0.20 * go_jaccard + 0.10 * omnipath_support)


def build_support_matrix(
    symbols_upper: list[str],
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> np.ndarray:
    n = len(symbols_upper)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        src = symbols_upper[i]
        for j in range(i + 1, n):
            tgt = symbols_upper[j]
            s_ij = support_score_pair(
                src=src,
                tgt=tgt,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )
            s_ji = support_score_pair(
                src=tgt,
                tgt=src,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )
            value = max(s_ij, s_ji)
            mat[i, j] = value
            mat[j, i] = value
    return mat


def load_h31_utility_table() -> pd.DataFrame:
    df = pd.read_csv(H31_UTILITY_PATH)
    utility = (
        df.groupby(["domain", "seed_tag", "layer"], as_index=False)["delta_auc_diffusion_incremental"]
        .mean()
        .rename(columns={"delta_auc_diffusion_incremental": "utility_delta_auc_mean"})
    )
    return utility


def linear_loo_r2(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    groups = groups[mask]
    if x.size < 6:
        return float("nan")

    y_pred = np.full(y.size, np.nan, dtype=float)
    unique_groups = np.unique(groups)
    for hold in unique_groups:
        test_mask = groups == hold
        train_mask = ~test_mask
        if np.sum(train_mask) < 3 or np.sum(test_mask) < 1:
            continue
        x_train = x[train_mask]
        y_train = y[train_mask]
        design = np.column_stack([np.ones(y_train.size), x_train])
        coef, _, _, _ = np.linalg.lstsq(design, y_train, rcond=None)
        x_test = x[test_mask]
        y_pred[test_mask] = coef[0] + coef[1] * x_test

    valid = np.isfinite(y_pred)
    if np.sum(valid) < 3:
        return float("nan")
    y_true = y[valid]
    y_hat = y_pred[valid]
    ss_res = float(np.sum((y_true - y_hat) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def stratified_shuffle(values: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(values)
    strata = np.asarray(strata, dtype=int)
    out = x.copy()
    for stratum in np.unique(strata):
        idx = np.where(strata == stratum)[0]
        if idx.size > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def run_h46_weighted_zigzag(
    dorothea_map: dict[tuple[str, str], int],
    trrust_pairs: set[tuple[str, str]],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    utility_df = load_h31_utility_table()

    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_index, (seed_tag, run_dir) in enumerate(run_map.items()):
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            split_masks = build_split_masks(edge_df)

            source_edges_raw = edge_df.loc[split_masks["source_disjoint"]].copy()
            target_edges_raw = edge_df.loc[split_masks["target_disjoint"]].copy()
            if source_edges_raw["label"].nunique() < 2 or target_edges_raw["label"].nunique() < 2:
                continue

            source_top = set(select_top_genes(source_edges_raw, gene_cap=H46_GENE_CAP))
            target_top = set(select_top_genes(target_edges_raw, gene_cap=H46_GENE_CAP))
            union_genes = np.array(sorted(source_top | target_top), dtype=int)
            if union_genes.size < 90:
                continue

            gene_to_union_local = {int(g): int(i) for i, g in enumerate(union_genes)}
            source_positions = np.array([gene_to_union_local[g] for g in sorted(source_top)], dtype=int)
            target_positions = np.array([gene_to_union_local[g] for g in sorted(target_top)], dtype=int)

            symbol_map: dict[int, str] = {}
            for row in edge_df[["source_idx", "source"]].drop_duplicates().itertuples(index=False):
                symbol_map[int(row.source_idx)] = str(row.source).upper()
            for row in edge_df[["target_idx", "target"]].drop_duplicates().itertuples(index=False):
                symbol_map[int(row.target_idx)] = str(row.target).upper()
            ordered_symbols = [symbol_map[int(g)] for g in union_genes]

            support_matrix = build_support_matrix(
                symbols_upper=ordered_symbols,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )

            for layer in H46_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue
                points_union = layer_embeddings[layer, union_genes, :]
                points_union_pca = reduce_points(
                    points_union,
                    n_components=20,
                    random_state=21_460 + domain_index * 100 + seed_index * 10 + layer,
                )

                source_edges_set = build_knn_edges_for_subset(
                    points_union=points_union_pca,
                    subset_positions=source_positions,
                    n_neighbors=H46_KNN,
                )
                target_edges_set = build_knn_edges_for_subset(
                    points_union=points_union_pca,
                    subset_positions=target_positions,
                    n_neighbors=H46_KNN,
                )
                if len(source_edges_set) < 30 or len(target_edges_set) < 30:
                    continue

                source_edges = sorted(source_edges_set)
                target_edges = sorted(target_edges_set)
                source_support = np.array([support_matrix[i, j] for i, j in source_edges], dtype=float)
                target_support = np.array([support_matrix[i, j] for i, j in target_edges], dtype=float)
                all_support = np.concatenate([source_support, target_support])

                q50 = float(np.quantile(all_support, 0.50))
                q75 = float(np.quantile(all_support, 0.75))

                source_mid = {e for e, s in zip(source_edges, source_support) if s >= q50}
                target_mid = {e for e, s in zip(target_edges, target_support) if s >= q50}
                source_hi = {e for e, s in zip(source_edges, source_support) if s >= q75}
                target_hi = {e for e, s in zip(target_edges, target_support) if s >= q75}

                obs_unweighted = zigzag_h1_total_lifetime(
                    n_vertices=union_genes.size,
                    source_edges=set(source_edges),
                    target_edges=set(target_edges),
                )
                obs_weighted_mid = zigzag_h1_total_lifetime(
                    n_vertices=union_genes.size,
                    source_edges=source_mid,
                    target_edges=target_mid,
                )
                obs_weighted_hi = zigzag_h1_total_lifetime(
                    n_vertices=union_genes.size,
                    source_edges=source_hi,
                    target_edges=target_hi,
                )
                obs_weighted = 0.5 * (obs_weighted_mid + obs_weighted_hi)

                rng = np.random.default_rng(21_470 + domain_index * 100 + seed_index * 10 + layer)
                null_unweighted = np.empty(H46_NULL_PERM, dtype=float)
                null_weighted = np.empty(H46_NULL_PERM, dtype=float)
                for perm_idx in range(H46_NULL_PERM):
                    target_perm = np.array(
                        sorted(rng.choice(union_genes.size, size=target_positions.size, replace=False)),
                        dtype=int,
                    )
                    target_perm_edges_set = build_knn_edges_for_subset(
                        points_union=points_union_pca,
                        subset_positions=target_perm,
                        n_neighbors=H46_KNN,
                    )
                    target_perm_edges = sorted(target_perm_edges_set)
                    target_perm_support = np.array(
                        [support_matrix[i, j] for i, j in target_perm_edges],
                        dtype=float,
                    )

                    all_support_perm = np.concatenate([source_support, target_perm_support])
                    q50_perm = float(np.quantile(all_support_perm, 0.50))
                    q75_perm = float(np.quantile(all_support_perm, 0.75))
                    target_perm_mid = {e for e, s in zip(target_perm_edges, target_perm_support) if s >= q50_perm}
                    target_perm_hi = {e for e, s in zip(target_perm_edges, target_perm_support) if s >= q75_perm}

                    null_un = zigzag_h1_total_lifetime(
                        n_vertices=union_genes.size,
                        source_edges=set(source_edges),
                        target_edges=set(target_perm_edges),
                    )
                    null_mid = zigzag_h1_total_lifetime(
                        n_vertices=union_genes.size,
                        source_edges=source_mid,
                        target_edges=target_perm_mid,
                    )
                    null_hi = zigzag_h1_total_lifetime(
                        n_vertices=union_genes.size,
                        source_edges=source_hi,
                        target_edges=target_perm_hi,
                    )
                    null_w = 0.5 * (null_mid + null_hi)

                    null_unweighted[perm_idx] = null_un
                    null_weighted[perm_idx] = null_w
                    null_rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": "paired_source_target",
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_unweighted_h1_total": float(null_un),
                            "null_weighted_h1_total": float(null_w),
                        }
                    )

                p_unweighted = empirical_upper_tail_p(obs_unweighted, null_unweighted)
                p_weighted = empirical_upper_tail_p(obs_weighted, null_weighted)
                excess_unweighted = float(obs_unweighted - np.nanmean(null_unweighted))
                excess_weighted = float(obs_weighted - np.nanmean(null_weighted))

                util_match = utility_df.loc[
                    (utility_df["domain"] == domain)
                    & (utility_df["seed_tag"] == seed_tag)
                    & (utility_df["layer"] == int(layer)),
                    "utility_delta_auc_mean",
                ]
                utility_value = float(util_match.iloc[0]) if not util_match.empty else float("nan")

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": "paired_source_target",
                        "layer": int(layer),
                        "n_union_genes": int(union_genes.size),
                        "n_source_edges": int(len(source_edges)),
                        "n_target_edges": int(len(target_edges)),
                        "obs_unweighted_h1_total": float(obs_unweighted),
                        "obs_weighted_h1_total": float(obs_weighted),
                        "null_mean_unweighted_h1_total": float(np.nanmean(null_unweighted)),
                        "null_mean_weighted_h1_total": float(np.nanmean(null_weighted)),
                        "excess_unweighted_h1_total": float(excess_unweighted),
                        "excess_weighted_h1_total": float(excess_weighted),
                        "p_unweighted_upper": float(p_unweighted),
                        "p_weighted_upper": float(p_weighted),
                        "utility_delta_auc_mean": float(utility_value),
                        "mean_trrust_support_bookkeeping": float(
                            np.mean(
                                [
                                    float((ordered_symbols[i], ordered_symbols[j]) in trrust_pairs)
                                    for i, j in source_edges[: min(200, len(source_edges))]
                                ]
                            )
                            if source_edges
                            else 0.0
                        ),
                    }
                )

    by_row_df = pd.DataFrame(rows).sort_values(["domain", "seed_tag", "layer"])
    by_row_path = ITER_DIR / "h46_weighted_zigzag_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(["domain", "seed_tag", "layer", "perm_idx"])
    null_path = ITER_DIR / "h46_weighted_zigzag_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    domain_rows: list[dict[str, object]] = []
    for domain, group in by_row_df.groupby("domain", sort=True):
        uw = group["excess_unweighted_h1_total"].to_numpy(dtype=float)
        ww = group["excess_weighted_h1_total"].to_numpy(dtype=float)
        yy = group["utility_delta_auc_mean"].to_numpy(dtype=float)

        corr_unweighted = float(pd.Series(uw).corr(pd.Series(yy), method="spearman"))
        corr_weighted = float(pd.Series(ww).corr(pd.Series(yy), method="spearman"))

        domain_rows.append(
            {
                "domain": domain,
                "n_rows": int(group.shape[0]),
                "mean_excess_unweighted_h1_total": float(np.nanmean(uw)),
                "mean_excess_weighted_h1_total": float(np.nanmean(ww)),
                "spearman_utility_vs_excess_unweighted": float(corr_unweighted),
                "spearman_utility_vs_excess_weighted": float(corr_weighted),
                "weighted_minus_unweighted_spearman": float(corr_weighted - corr_unweighted),
                "combined_fisher_p_unweighted": float(
                    safe_fisher_p(group["p_unweighted_upper"].to_numpy(dtype=float))
                ),
                "combined_fisher_p_weighted": float(
                    safe_fisher_p(group["p_weighted_upper"].to_numpy(dtype=float))
                ),
            }
        )

    domain_df = pd.DataFrame(domain_rows).sort_values("domain")
    domain_path = ITER_DIR / "h46_weighted_zigzag_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    loo_r2_weighted = linear_loo_r2(
        x=by_row_df["excess_weighted_h1_total"].to_numpy(dtype=float),
        y=by_row_df["utility_delta_auc_mean"].to_numpy(dtype=float),
        groups=by_row_df["domain"].to_numpy(),
    )
    loo_r2_unweighted = linear_loo_r2(
        x=by_row_df["excess_unweighted_h1_total"].to_numpy(dtype=float),
        y=by_row_df["utility_delta_auc_mean"].to_numpy(dtype=float),
        groups=by_row_df["domain"].to_numpy(),
    )

    summary = {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_excess_weighted_h1_total": float(by_row_df["excess_weighted_h1_total"].mean())
        if not by_row_df.empty
        else float("nan"),
        "mean_excess_unweighted_h1_total": float(by_row_df["excess_unweighted_h1_total"].mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_weighted_better_count": int(
            (
                domain_df["weighted_minus_unweighted_spearman"].to_numpy(dtype=float) > 0.0
            ).sum()
        )
        if not domain_df.empty
        else 0,
        "loo_r2_weighted": float(loo_r2_weighted),
        "loo_r2_unweighted": float(loo_r2_unweighted),
        "artifact_paths": {
            "by_seed_layer_split": str(by_row_path),
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
        },
    }
    return summary


def build_knn_edge_array(points: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    n_points = points.shape[0]
    k = max(2, min(n_neighbors, n_points - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(points)
    distances, indices = nbrs.kneighbors(points)

    edge_dist: dict[tuple[int, int], float] = {}
    for i in range(n_points):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            u, v = sorted((int(i), int(j)))
            if u == v:
                continue
            dval = float(dist)
            if (u, v) not in edge_dist or dval < edge_dist[(u, v)]:
                edge_dist[(u, v)] = dval

    edges = np.array(list(edge_dist.keys()), dtype=int)
    dists = np.array(list(edge_dist.values()), dtype=float)
    return edges, dists


def cycle_rank(n_nodes: int, edges: np.ndarray) -> float:
    m = int(edges.shape[0])
    if m == 0:
        return 0.0
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(row.size, dtype=np.int8)
    graph = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    n_components, _ = connected_components(graph, directed=False)
    return float(max(0, m - n_nodes + n_components))


def bifiltration_scores(
    n_nodes: int,
    edges: np.ndarray,
    dists: np.ndarray,
    support: np.ndarray,
    d_quantiles: list[float],
    s_quantiles: list[float],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    n_edges = edges.shape[0]
    d_thresholds = np.quantile(dists, d_quantiles)
    s_thresholds = np.quantile(support, s_quantiles)

    score_bif = np.zeros(n_edges, dtype=float)
    count_bif = np.zeros(n_edges, dtype=float)

    score_dist = np.zeros(n_edges, dtype=float)
    count_dist = np.zeros(n_edges, dtype=float)

    mean_beta_bif = []
    mean_beta_dist = []

    for d_thr in d_thresholds:
        mask_d = dists <= float(d_thr)
        edges_d = edges[mask_d]
        beta_d = cycle_rank(n_nodes=n_nodes, edges=edges_d)
        mean_beta_dist.append(beta_d)

        score_dist += beta_d * mask_d.astype(float)
        count_dist += mask_d.astype(float)

        for s_thr in s_thresholds:
            mask = mask_d & (support >= float(s_thr))
            edges_ds = edges[mask]
            beta_ds = cycle_rank(n_nodes=n_nodes, edges=edges_ds)
            mean_beta_bif.append(beta_ds)

            score_bif += beta_ds * mask.astype(float)
            count_bif += mask.astype(float)

    score_bif = score_bif / np.clip(count_bif, 1.0, None)
    score_dist = score_dist / np.clip(count_dist, 1.0, None)

    return (
        score_bif,
        score_dist,
        float(np.mean(mean_beta_bif)) if mean_beta_bif else 0.0,
        float(np.mean(mean_beta_dist)) if mean_beta_dist else 0.0,
    )


def run_h47_bifiltration(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(SCGPT_RUNS_BY_DOMAIN.items()):
        seed_tag = "seed42_main"
        run_dir = run_map[seed_tag]

        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = build_split_masks(edge_df)

        for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
            split_edges = edge_df.loc[split_mask].copy()
            if split_edges["label"].nunique() < 2:
                continue

            top_genes = set(select_top_genes(split_edges, gene_cap=H47_GENE_CAP))
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
            if edge_gene_indices.size < 100:
                continue

            gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
            source_local = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
            target_local = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
            labels = split_edges["label"].to_numpy(dtype=int)

            symbol_map: dict[int, str] = {}
            for row in split_edges[["source_idx", "source"]].drop_duplicates().itertuples(index=False):
                symbol_map[int(row.source_idx)] = str(row.source).upper()
            for row in split_edges[["target_idx", "target"]].drop_duplicates().itertuples(index=False):
                symbol_map[int(row.target_idx)] = str(row.target).upper()
            ordered_symbols = [symbol_map[int(g)] for g in edge_gene_indices]

            support_matrix = build_support_matrix(
                symbols_upper=ordered_symbols,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )

            for layer in H47_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue
                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = reduce_points(
                    points,
                    n_components=20,
                    random_state=21_570 + domain_index * 100 + split_index * 10 + layer,
                )

                knn_edges, knn_dists = build_knn_edge_array(points=points_pca, n_neighbors=H47_KNN)
                if knn_edges.shape[0] < 80:
                    continue

                knn_support = np.array(
                    [support_matrix[i, j] for i, j in knn_edges],
                    dtype=float,
                )
                distance_bins = pd.qcut(knn_dists, q=5, labels=False, duplicates="drop")
                distance_bins = np.asarray(distance_bins, dtype=int)

                bif_score, dist_score, mean_beta_bif, mean_beta_dist = bifiltration_scores(
                    n_nodes=edge_gene_indices.size,
                    edges=knn_edges,
                    dists=knn_dists,
                    support=knn_support,
                    d_quantiles=[0.35, 0.50, 0.65, 0.80],
                    s_quantiles=[0.20, 0.40, 0.60, 0.80],
                )
                pair_to_bif = {
                    (int(i), int(j)): float(score)
                    for (i, j), score in zip(knn_edges.tolist(), bif_score.tolist())
                }
                pair_to_dist = {
                    (int(i), int(j)): float(score)
                    for (i, j), score in zip(knn_edges.tolist(), dist_score.tolist())
                }

                eval_pairs = [
                    (min(int(s), int(t)), max(int(s), int(t)))
                    for s, t in zip(source_local.tolist(), target_local.tolist())
                ]
                eval_bif = np.array([pair_to_bif.get(p, 0.0) for p in eval_pairs], dtype=float)
                eval_dist = np.array([pair_to_dist.get(p, 0.0) for p in eval_pairs], dtype=float)

                auc_bif = safe_auc(labels, eval_bif)
                auc_dist = safe_auc(labels, eval_dist)
                delta_auc = float(auc_bif - auc_dist) if np.isfinite(auc_bif) and np.isfinite(auc_dist) else float("nan")

                rng = np.random.default_rng(21_580 + domain_index * 100 + split_index * 10 + layer)
                null_delta = np.empty(H47_NULL_PERM, dtype=float)
                for perm_idx in range(H47_NULL_PERM):
                    support_perm = stratified_shuffle(knn_support, strata=distance_bins, rng=rng)
                    bif_perm, _, _, _ = bifiltration_scores(
                        n_nodes=edge_gene_indices.size,
                        edges=knn_edges,
                        dists=knn_dists,
                        support=support_perm,
                        d_quantiles=[0.35, 0.50, 0.65, 0.80],
                        s_quantiles=[0.20, 0.40, 0.60, 0.80],
                    )
                    pair_to_bif_perm = {
                        (int(i), int(j)): float(score)
                        for (i, j), score in zip(knn_edges.tolist(), bif_perm.tolist())
                    }
                    eval_bif_perm = np.array([pair_to_bif_perm.get(p, 0.0) for p in eval_pairs], dtype=float)
                    auc_bif_perm = safe_auc(labels, eval_bif_perm)
                    delta_perm = (
                        float(auc_bif_perm - auc_dist)
                        if np.isfinite(auc_bif_perm) and np.isfinite(auc_dist)
                        else float("nan")
                    )
                    null_delta[perm_idx] = delta_perm
                    null_rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc_bif_minus_distance": float(delta_perm),
                        }
                    )

                p_delta = empirical_upper_tail_p(delta_auc, null_delta)

                rows.append(
                    {
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "n_nodes_graph": int(edge_gene_indices.size),
                        "n_knn_edges": int(knn_edges.shape[0]),
                        "auc_bifiltration": float(auc_bif),
                        "auc_distance_only": float(auc_dist),
                        "delta_auc_bif_minus_distance": float(delta_auc),
                        "p_delta_auc_upper": float(p_delta),
                        "null_mean_delta_auc": float(np.nanmean(null_delta)),
                        "mean_beta_bif": float(mean_beta_bif),
                        "mean_beta_distance_only": float(mean_beta_dist),
                    }
                )

    by_row_df = pd.DataFrame(rows).sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h47_bifiltration_by_domain_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["domain", "split_regime", "layer", "perm_idx"]
    )
    null_path = ITER_DIR / "h47_bifiltration_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    domain_rows: list[dict[str, object]] = []
    for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
        domain_rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(group.shape[0]),
                "mean_auc_bifiltration": float(group["auc_bifiltration"].mean()),
                "mean_auc_distance_only": float(group["auc_distance_only"].mean()),
                "mean_delta_auc_bif_minus_distance": float(group["delta_auc_bif_minus_distance"].mean()),
                "fraction_delta_positive": float((group["delta_auc_bif_minus_distance"] > 0.0).mean()),
                "fraction_p_delta_lt_0_05": float((group["p_delta_auc_upper"] < 0.05).mean()),
                "combined_fisher_p_delta": float(
                    safe_fisher_p(group["p_delta_auc_upper"].to_numpy(dtype=float))
                ),
            }
        )

    domain_df = pd.DataFrame(domain_rows).sort_values(["domain", "split_regime"])
    domain_path = ITER_DIR / "h47_bifiltration_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    summary = {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_bif_minus_distance"].mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_split_positive_delta": int(
            (domain_df["mean_delta_auc_bif_minus_distance"] > 0.0).sum()
        )
        if not domain_df.empty
        else 0,
        "domain_split_fisher_sig": int((domain_df["combined_fisher_p_delta"] < 0.05).sum())
        if not domain_df.empty
        else 0,
        "artifact_paths": {
            "by_domain_layer_split": str(by_row_path),
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
        },
    }
    return summary


def compute_scgpt_centered_cosine(
    layer_embeddings: np.ndarray,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    pca_dim: int,
    seed: int,
) -> np.ndarray:
    centered = layer_embeddings - layer_embeddings.mean(axis=0, keepdims=True)
    n_comp = min(pca_dim, centered.shape[0] - 1, centered.shape[1])
    proj = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed).fit_transform(centered)
    src_proj = proj[src_idx]
    tgt_proj = proj[tgt_idx]
    num = np.sum(src_proj * tgt_proj, axis=1)
    den = np.clip(np.linalg.norm(src_proj, axis=1) * np.linalg.norm(tgt_proj, axis=1), 1e-8, None)
    return (num / den).astype(float)


def compute_geneformer_centered_cosine(
    centered_emb: np.ndarray,
    src_tok: np.ndarray,
    tgt_tok: np.ndarray,
) -> np.ndarray:
    src_center = centered_emb[src_tok]
    tgt_center = centered_emb[tgt_tok]
    num = np.sum(src_center * tgt_center, axis=1)
    den = np.clip(np.linalg.norm(src_center, axis=1) * np.linalg.norm(tgt_center, axis=1), 1e-8, None)
    return (num / den).astype(float)


def motif_ffl(edges: list[tuple[str, str]]) -> set[tuple[str, str, str]]:
    out: dict[str, set[str]] = {}
    for src, tgt in edges:
        if src == tgt:
            continue
        out.setdefault(src, set()).add(tgt)

    motifs: set[tuple[str, str, str]] = set()
    for a, a_out in out.items():
        for b in a_out:
            b_out = out.get(b, set())
            common = a_out & b_out
            for c in common:
                if c == a or c == b:
                    continue
                motifs.add((a, b, c))
    return motifs


def motif_bifan(edges: list[tuple[str, str]]) -> set[tuple[str, str, str, str]]:
    out: dict[str, set[str]] = {}
    for src, tgt in edges:
        if src == tgt:
            continue
        out.setdefault(src, set()).add(tgt)

    sources = sorted(out.keys())
    motifs: set[tuple[str, str, str, str]] = set()
    for i, a in enumerate(sources):
        targets_a = out[a]
        if len(targets_a) < 2:
            continue
        for b in sources[i + 1 :]:
            common = sorted((targets_a & out[b]) - {a, b})
            if len(common) < 2:
                continue
            for x in range(len(common) - 1):
                for y in range(x + 1, len(common)):
                    c = common[x]
                    d = common[y]
                    motifs.add((a, b, c, d))
    return motifs


def permute_edges_preserve_degree(
    sources: np.ndarray,
    targets: np.ndarray,
    rng: np.random.Generator,
    max_restarts: int = 40,
) -> list[tuple[str, str]]:
    src = np.asarray(sources, dtype=object)
    tgt = np.asarray(targets, dtype=object)

    for _ in range(max_restarts):
        perm_tgt = rng.permutation(tgt)
        if np.any(src == perm_tgt):
            continue
        edges = list(zip(src.tolist(), perm_tgt.tolist()))
        if len(set(edges)) == len(edges):
            return edges

    # Fallback keeps degree multiset but may include a small number of duplicates.
    perm_tgt = rng.permutation(tgt)
    edges = list(zip(src.tolist(), perm_tgt.tolist()))
    dedup = list(dict.fromkeys(edges))
    return dedup


def run_h48_cross_model_motif_overlap() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    model = AutoModel.from_pretrained("ctheodoris/Geneformer")
    emb_weight = model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float64, copy=False)
    centered_emb = emb_weight - emb_weight.mean(axis=0, keepdims=True)
    del model

    for domain_index, domain in enumerate(["immune", "lung", "external_lung"]):
        sc_edge_df = pd.read_csv(
            SCGPT_RUNS_BY_DOMAIN[domain]["seed42_main"] / "cycle1_edge_dataset.tsv",
            sep="\t",
        )[["source", "target", "source_idx", "target_idx"]].copy()

        gf_edge_df = pd.read_csv(GENEFORMER_EDGE_BY_DOMAIN[domain], sep="\t")[[
            "source",
            "target",
            "label",
            "source_token_id",
            "target_token_id",
        ]].copy()

        merged = gf_edge_df.merge(sc_edge_df, on=["source", "target"], how="inner")
        merged = merged.drop_duplicates(subset=["source", "target"]).reset_index(drop=True)
        if merged.shape[0] < 300:
            continue

        src_tok = merged["source_token_id"].to_numpy(dtype=int)
        tgt_tok = merged["target_token_id"].to_numpy(dtype=int)
        gf_score = compute_geneformer_centered_cosine(centered_emb=centered_emb, src_tok=src_tok, tgt_tok=tgt_tok)

        src_gene_idx = merged["source_idx"].to_numpy(dtype=int)
        tgt_gene_idx = merged["target_idx"].to_numpy(dtype=int)
        source_symbols = merged["source"].astype(str).to_numpy()
        target_symbols = merged["target"].astype(str).to_numpy()

        layer_embeddings = np.load(
            SCGPT_RUNS_BY_DOMAIN[domain]["seed42_main"] / "layer_gene_embeddings.npy",
            mmap_mode="r",
        )

        for layer in H48_LAYERS:
            if layer >= layer_embeddings.shape[0]:
                continue
            sc_score = compute_scgpt_centered_cosine(
                layer_embeddings=layer_embeddings[layer],
                src_idx=src_gene_idx,
                tgt_idx=tgt_gene_idx,
                pca_dim=64,
                seed=21_680 + domain_index * 10 + layer,
            )

            for k in H48_TOP_K:
                k_eff = min(k, merged.shape[0])
                if k_eff < 20:
                    continue

                idx_sc = np.argpartition(-sc_score, k_eff - 1)[:k_eff]
                idx_gf = np.argpartition(-gf_score, k_eff - 1)[:k_eff]

                sc_edges = list(zip(source_symbols[idx_sc].tolist(), target_symbols[idx_sc].tolist()))
                gf_edges = list(zip(source_symbols[idx_gf].tolist(), target_symbols[idx_gf].tolist()))

                ffl_sc = motif_ffl(sc_edges)
                ffl_gf = motif_ffl(gf_edges)
                bifan_sc = motif_bifan(sc_edges)
                bifan_gf = motif_bifan(gf_edges)

                overlap_ffl = int(len(ffl_sc & ffl_gf))
                overlap_bifan = int(len(bifan_sc & bifan_gf))
                overlap_total = int(overlap_ffl + overlap_bifan)

                rng = np.random.default_rng(21_690 + domain_index * 100 + layer * 10 + k_eff)
                null_total = np.empty(H48_NULL_PERM, dtype=float)
                for perm_idx in range(H48_NULL_PERM):
                    sc_perm_edges = permute_edges_preserve_degree(
                        sources=source_symbols[idx_sc],
                        targets=target_symbols[idx_sc],
                        rng=rng,
                    )
                    gf_perm_edges = permute_edges_preserve_degree(
                        sources=source_symbols[idx_gf],
                        targets=target_symbols[idx_gf],
                        rng=rng,
                    )

                    ffl_sc_perm = motif_ffl(sc_perm_edges)
                    ffl_gf_perm = motif_ffl(gf_perm_edges)
                    bifan_sc_perm = motif_bifan(sc_perm_edges)
                    bifan_gf_perm = motif_bifan(gf_perm_edges)

                    overlap_perm = int(len(ffl_sc_perm & ffl_gf_perm) + len(bifan_sc_perm & bifan_gf_perm))
                    null_total[perm_idx] = float(overlap_perm)
                    null_rows.append(
                        {
                            "domain": domain,
                            "layer": int(layer),
                            "top_k": int(k_eff),
                            "perm_idx": int(perm_idx),
                            "null_overlap_total": float(overlap_perm),
                        }
                    )

                p_overlap = empirical_upper_tail_p(float(overlap_total), null_total)
                null_mean = float(np.nanmean(null_total))
                null_std = float(np.nanstd(null_total))
                delta_overlap = float(overlap_total - null_mean)
                if null_std <= 1e-12:
                    # Zero-variance nulls are common in sparse motif regimes.
                    # Avoid huge artificial z-scores when dividing by ~0.
                    z_overlap = 0.0 if abs(delta_overlap) <= 1e-12 else float("nan")
                else:
                    z_overlap = float(delta_overlap / null_std)

                rows.append(
                    {
                        "domain": domain,
                        "layer": int(layer),
                        "top_k": int(k_eff),
                        "overlap_ffl": int(overlap_ffl),
                        "overlap_bifan": int(overlap_bifan),
                        "overlap_total": int(overlap_total),
                        "sc_ffl_count": int(len(ffl_sc)),
                        "gf_ffl_count": int(len(ffl_gf)),
                        "sc_bifan_count": int(len(bifan_sc)),
                        "gf_bifan_count": int(len(bifan_gf)),
                        "null_mean_overlap_total": float(null_mean),
                        "null_std_overlap_total": float(null_std),
                        "delta_overlap_total": float(delta_overlap),
                        "z_overlap_total": float(z_overlap),
                        "p_overlap_upper": float(p_overlap),
                    }
                )

    by_row_df = pd.DataFrame(rows).sort_values(["domain", "layer", "top_k"])
    by_row_path = ITER_DIR / "h48_cross_model_motif_overlap_by_domain_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(["domain", "layer", "top_k", "perm_idx"])
    null_path = ITER_DIR / "h48_cross_model_motif_overlap_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    for domain, group in by_row_df.groupby("domain", sort=True):
        summary_rows.append(
            {
                "domain": domain,
                "n_rows": int(group.shape[0]),
                "mean_overlap_total": float(group["overlap_total"].mean()),
                "mean_delta_overlap_total": float(group["delta_overlap_total"].mean()),
                "mean_z_overlap_total": float(group["z_overlap_total"].mean()),
                "fraction_delta_positive": float((group["delta_overlap_total"] > 0.0).mean()),
                "fraction_z_positive": float((group["z_overlap_total"] > 0.0).mean()),
                "fraction_p_overlap_lt_0_05": float((group["p_overlap_upper"] < 0.05).mean()),
                "combined_fisher_p_overlap": float(
                    safe_fisher_p(group["p_overlap_upper"].to_numpy(dtype=float))
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("domain")
    summary_path = ITER_DIR / "h48_cross_model_motif_overlap_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    summary = {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_overlap_total": float(by_row_df["delta_overlap_total"].mean())
        if not by_row_df.empty
        else float("nan"),
        "mean_z_overlap_total": float(by_row_df["z_overlap_total"].mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_positive_delta": int((summary_df["mean_delta_overlap_total"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "domain_positive_z": int((summary_df["mean_z_overlap_total"] > 0.0).sum())
        if not summary_df.empty
        else 0,
        "domain_fisher_sig": int((summary_df["combined_fisher_p_overlap"] < 0.05).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_domain_layer": str(by_row_path),
            "summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }
    return summary


def main() -> None:
    required_paths = [
        TRRUST_PATH,
        DOROTHEA_PATH,
        GENE2GO_PATH,
        OMNIPATH_INTERACTIONS_PATH,
        H31_UTILITY_PATH,
        STRING_CACHE_PATH,
    ]
    for domain, run_map in SCGPT_RUNS_BY_DOMAIN.items():
        required_paths.append(GENEFORMER_EDGE_BY_DOMAIN[domain])
        for run_dir in run_map.values():
            required_paths.append(run_dir / "cycle1_edge_dataset.tsv")
            required_paths.append(run_dir / "layer_gene_embeddings.npy")

    missing = [str(p) for p in required_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    dorothea_map = load_dorothea_score_map()
    trrust_pairs = load_trrust_pairs()
    omnipath_pairs = load_omnipath_pairs()
    gene2go_upper = load_gene2go_upper()
    string_map = load_string_scores_from_cache(STRING_CACHE_PATH)

    h46_summary = run_h46_weighted_zigzag(
        dorothea_map=dorothea_map,
        trrust_pairs=trrust_pairs,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h47_summary = run_h47_bifiltration(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h48_summary = run_h48_cross_model_motif_overlap()

    summary = {
        "iteration": "iter_0021",
        "h46": h46_summary,
        "h47": h47_summary,
        "h48": h48_summary,
        "environment": {
            "python_env": "subproject40-topology",
            "string_cache_used": str(STRING_CACHE_PATH),
            "geneformer_model": "ctheodoris/Geneformer",
        },
    }
    summary_path = ITER_DIR / "iter0021_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
