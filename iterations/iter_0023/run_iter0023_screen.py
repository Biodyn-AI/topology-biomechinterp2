from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import combine_pvalues
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


ITER_DIR = Path("iterations/iter_0023")
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

# H52: directed/signed multiseed replication with stricter controls.
H52_LAYERS = [0, 3, 7, 11]
H52_GENE_CAP = 180
H52_KNN = 10
H52_NULL_PERM = 16

# H53: directed path-homology surrogate pilot (high-risk/high-reward).
H53_LAYERS = [7, 11]
H53_GENE_CAP = 140
H53_KNN = 8
H53_NULL_PERM = 6
H53_MAX_TRIANGLES = 1600

# H54: local linearity rupture index (cheap broad screen).
H54_LAYERS = [0, 3, 7, 11]
H54_GENE_CAP = 180
H54_NEIGHBORS = 12
H54_NULL_PERM = 16


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


def distance_cycle_score_matrix(
    n_nodes: int,
    edges: np.ndarray,
    dists: np.ndarray,
    d_quantiles: list[float],
) -> np.ndarray:
    score = np.zeros((n_nodes, n_nodes), dtype=float)
    count = np.zeros((n_nodes, n_nodes), dtype=float)

    for d_thr in np.quantile(dists, d_quantiles):
        mask = dists <= float(d_thr)
        edges_d = edges[mask]
        beta = cycle_rank(n_nodes=n_nodes, edges=edges_d)
        for u, v in edges_d:
            iu = int(u)
            iv = int(v)
            score[iu, iv] += beta
            score[iv, iu] += beta
            count[iu, iv] += 1.0
            count[iv, iu] += 1.0

    return score / np.clip(count, 1.0, None)


def directed_signed_score_matrix(
    n_nodes: int,
    edges: np.ndarray,
    dists: np.ndarray,
    margins: np.ndarray,
    d_quantiles: list[float],
    m_quantiles: list[float],
) -> np.ndarray:
    total = np.zeros((n_nodes, n_nodes), dtype=float)
    n_steps = 0

    for d_thr in np.quantile(dists, d_quantiles):
        mask_d = dists <= float(d_thr)
        if int(mask_d.sum()) < 6:
            continue

        edges_d = edges[mask_d]
        margins_d = margins[mask_d]
        abs_margin_d = np.abs(margins_d)
        if np.all(abs_margin_d <= 0.0):
            continue

        for m_thr in np.quantile(abs_margin_d, m_quantiles):
            strong = abs_margin_d >= float(m_thr)
            if int(strong.sum()) < 6:
                continue

            edges_s = edges_d[strong]
            margins_s = margins_d[strong]

            adj = np.zeros((n_nodes, n_nodes), dtype=np.int16)
            signed = np.zeros((n_nodes, n_nodes), dtype=float)

            for (u, v), margin in zip(edges_s, margins_s):
                iu = int(u)
                iv = int(v)
                mval = float(margin)
                if mval >= 0.0:
                    adj[iu, iv] = 1
                else:
                    adj[iv, iu] = 1
                signed[iu, iv] = mval
                signed[iv, iu] = -mval

            paths2 = adj @ adj
            signed_component = np.tanh(4.0 * signed)
            component = (paths2 - paths2.T).astype(float) + signed_component

            total += component
            n_steps += 1

    if n_steps == 0:
        return total
    return total / float(n_steps)


def degree_bins(values: np.ndarray, max_bins: int = 5) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return np.array([], dtype=int)
    q = min(max_bins, max(1, int(np.sqrt(x.size) // 2)))
    if q <= 1:
        return np.zeros(x.size, dtype=int)
    ranks = pd.Series(x).rank(method="average", pct=True).to_numpy(dtype=float)
    bins = np.minimum((ranks * q).astype(int), q - 1)
    return bins


def stratified_shuffle(values: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(values)
    strata = np.asarray(strata, dtype=int)
    out = x.copy()
    for stratum in np.unique(strata):
        idx = np.where(strata == stratum)[0]
        if idx.size > 1:
            out[idx] = rng.permutation(out[idx])
    return out


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
    df = pd.read_csv(Path(path), sep="\t")
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


def support_score_directed(
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

    return float(0.40 * dorothea_norm + 0.25 * omnipath_support + 0.20 * string_score + 0.15 * go_jaccard)


def build_support_matrices(
    symbols_upper: list[str],
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(symbols_upper)
    support_undirected = np.zeros((n, n), dtype=np.float64)
    support_directed = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        src = symbols_upper[i]
        for j in range(i + 1, n):
            tgt = symbols_upper[j]
            s_ij = support_score_directed(
                src=src,
                tgt=tgt,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )
            s_ji = support_score_directed(
                src=tgt,
                tgt=src,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )
            support_directed[i, j] = s_ij
            support_directed[j, i] = s_ji
            und = max(s_ij, s_ji)
            support_undirected[i, j] = und
            support_undirected[j, i] = und

    return support_undirected, support_directed


def directed_flag_beta1(
    n_nodes: int,
    directed_edges: list[tuple[int, int]],
    max_triangles: int,
    rng: np.random.Generator,
) -> float:
    edge_set = {(int(u), int(v)) for u, v in directed_edges if int(u) != int(v)}
    if not edge_set:
        return 0.0

    edge_list = sorted(edge_set)
    m = len(edge_list)
    edge_to_idx = {edge: idx for idx, edge in enumerate(edge_list)}

    b1 = np.zeros((n_nodes, m), dtype=np.float32)
    for idx, (u, v) in enumerate(edge_list):
        b1[u, idx] = -1.0
        b1[v, idx] = 1.0
    rank1 = int(np.linalg.matrix_rank(b1))

    out_neighbors: dict[int, set[int]] = {}
    for u, v in edge_list:
        out_neighbors.setdefault(u, set()).add(v)

    triangles: list[tuple[int, int, int]] = []
    for u, outs in out_neighbors.items():
        for v in outs:
            common = out_neighbors.get(v, set()) & outs
            for w in common:
                if w == u or w == v:
                    continue
                triangles.append((u, v, w))

    if len(triangles) > max_triangles:
        keep = rng.choice(len(triangles), size=max_triangles, replace=False)
        triangles = [triangles[int(i)] for i in keep]

    if not triangles:
        return float(max(0, m - rank1))

    b2 = np.zeros((m, len(triangles)), dtype=np.float32)
    for col, (u, v, w) in enumerate(triangles):
        b2[edge_to_idx[(u, v)], col] += 1.0
        b2[edge_to_idx[(v, w)], col] += 1.0
        b2[edge_to_idx[(u, w)], col] -= 1.0

    rank2 = int(np.linalg.matrix_rank(b2))
    beta1 = max(0, (m - rank1) - rank2)
    return float(beta1)


def directed_path_homology_score_matrix(
    n_nodes: int,
    edges: np.ndarray,
    dists: np.ndarray,
    margins: np.ndarray,
    d_quantiles: list[float],
    m_quantiles: list[float],
    max_triangles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    total = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    count = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    d_thresholds = np.unique(np.quantile(dists, d_quantiles))
    for d_thr in d_thresholds:
        mask_d = dists <= float(d_thr)
        if int(mask_d.sum()) < 8:
            continue

        edges_d = edges[mask_d]
        margins_d = margins[mask_d]
        abs_margin_d = np.abs(margins_d)
        if np.all(abs_margin_d <= 0.0):
            continue

        m_thresholds = np.unique(np.quantile(abs_margin_d, m_quantiles))
        for m_thr in m_thresholds:
            strong = abs_margin_d >= float(m_thr)
            if int(strong.sum()) < 8:
                continue

            edges_s = edges_d[strong]
            margins_s = margins_d[strong]

            oriented_edges: list[tuple[int, int]] = []
            for (u, v), margin in zip(edges_s, margins_s):
                iu = int(u)
                iv = int(v)
                if float(margin) >= 0.0:
                    oriented_edges.append((iu, iv))
                else:
                    oriented_edges.append((iv, iu))

            beta1 = directed_flag_beta1(
                n_nodes=n_nodes,
                directed_edges=oriented_edges,
                max_triangles=max_triangles,
                rng=rng,
            )
            for u, v in oriented_edges:
                total[u, v] += beta1
                count[u, v] += 1.0

    return total / np.clip(count, 1.0, None)


def get_knn_indices(points: np.ndarray, n_neighbors: int) -> np.ndarray:
    k = max(2, min(n_neighbors, points.shape[0] - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(points)
    _, indices = nbrs.kneighbors(points)
    return indices[:, 1:]


def compute_local_reconstruction_errors(points: np.ndarray, neighbor_idx: np.ndarray) -> np.ndarray:
    n, _ = points.shape
    errors = np.empty(n, dtype=float)

    # This is a direct LLE-style local linear reconstruction error.
    # We use it as the per-node local linearity base signal.
    for i in range(n):
        neigh = neighbor_idx[i]
        y = points[i]
        x = points[neigh]
        x_center = x - x.mean(axis=0, keepdims=True)
        y_center = y - x.mean(axis=0)
        cov = x_center @ x_center.T
        cov.flat[:: cov.shape[0] + 1] += 1e-3
        w = np.linalg.solve(cov, x_center @ y_center)
        w_sum = np.sum(w)
        if abs(w_sum) > 1e-8:
            w = w / w_sum
        recon = w @ x
        errors[i] = float(np.mean((recon - y) ** 2))

    return errors


def compute_rupture_from_layer_errors(layer_errors: np.ndarray) -> np.ndarray:
    # Rupture score at each layer = normalized local slope magnitude in the
    # depth trajectory of reconstruction errors for each node.
    errors = np.asarray(layer_errors, dtype=float)
    if errors.ndim != 2:
        raise ValueError("layer_errors must be 2D [n_layers, n_nodes]")
    n_layers, _ = errors.shape
    if n_layers < 2:
        return np.zeros_like(errors)

    diffs = np.abs(np.diff(errors, axis=0))
    scale = np.median(diffs, axis=0)
    scale = np.clip(scale, 1e-6, None)

    rupture = np.zeros_like(errors)
    for li in range(n_layers):
        if li == 0:
            local_delta = diffs[0]
        elif li == n_layers - 1:
            local_delta = diffs[-1]
        else:
            local_delta = 0.5 * (diffs[li - 1] + diffs[li])
        rupture[li] = local_delta / scale

    return rupture


def build_symbol_map(split_edges: pd.DataFrame) -> dict[int, str]:
    symbol_map: dict[int, str] = {}
    for row in split_edges[["source_idx", "source"]].drop_duplicates().itertuples(index=False):
        symbol_map[int(row.source_idx)] = str(row.source).upper()
    for row in split_edges[["target_idx", "target"]].drop_duplicates().itertuples(index=False):
        symbol_map[int(row.target_idx)] = str(row.target).upper()
    return symbol_map


def run_h52_directed_signed_multiseed(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_index, (seed_tag, run_dir) in enumerate(run_map.items()):
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            split_masks = build_split_masks(edge_df)

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                top_genes = set(select_top_genes(split_edges, gene_cap=H52_GENE_CAP))
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

                gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
                source_local = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
                target_local = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
                labels = split_edges["label"].to_numpy(dtype=int)

                node_deg = np.bincount(
                    np.concatenate([source_local, target_local]),
                    minlength=edge_gene_indices.size,
                )
                edge_deg = node_deg[source_local] + node_deg[target_local]
                edge_deg_bins = degree_bins(edge_deg, max_bins=5)

                symbol_map = build_symbol_map(split_edges)
                ordered_symbols = [symbol_map[int(g)] for g in edge_gene_indices]
                _, support_directed = build_support_matrices(
                    symbols_upper=ordered_symbols,
                    dorothea_map=dorothea_map,
                    omnipath_pairs=omnipath_pairs,
                    gene2go_upper=gene2go_upper,
                    string_map=string_map,
                )

                for layer in H52_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = reduce_points(
                        points,
                        n_components=20,
                        random_state=23_520 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer,
                    )

                    knn_edges, knn_dists = build_knn_edge_array(points=points_pca, n_neighbors=H52_KNN)
                    if knn_edges.shape[0] < 100:
                        continue

                    margins = np.array(
                        [support_directed[i, j] - support_directed[j, i] for i, j in knn_edges],
                        dtype=float,
                    )

                    directed_matrix = directed_signed_score_matrix(
                        n_nodes=edge_gene_indices.size,
                        edges=knn_edges,
                        dists=knn_dists,
                        margins=margins,
                        d_quantiles=[0.35, 0.50, 0.65, 0.80],
                        m_quantiles=[0.40, 0.60, 0.80],
                    )
                    dist_matrix = distance_cycle_score_matrix(
                        n_nodes=edge_gene_indices.size,
                        edges=knn_edges,
                        dists=knn_dists,
                        d_quantiles=[0.35, 0.50, 0.65, 0.80],
                    )

                    eval_dir = directed_matrix[source_local, target_local]
                    eval_dist = dist_matrix[source_local, target_local]

                    auc_dir = safe_auc(labels, eval_dir)
                    auc_dist = safe_auc(labels, eval_dist)
                    delta_auc = (
                        float(auc_dir - auc_dist) if np.isfinite(auc_dir) and np.isfinite(auc_dist) else float("nan")
                    )

                    rng = np.random.default_rng(
                        23_521 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer
                    )

                    null_degree = np.empty(H52_NULL_PERM, dtype=float)
                    null_sign = np.empty(H52_NULL_PERM, dtype=float)
                    null_label = np.empty(H52_NULL_PERM, dtype=float)

                    for perm_idx in range(H52_NULL_PERM):
                        perm = rng.permutation(edge_gene_indices.size)
                        inv = np.empty_like(perm)
                        inv[perm] = np.arange(perm.size)
                        eval_dir_deg = directed_matrix[inv[source_local], inv[target_local]]
                        auc_deg = safe_auc(labels, eval_dir_deg)
                        delta_deg = (
                            float(auc_deg - auc_dist) if np.isfinite(auc_deg) and np.isfinite(auc_dist) else float("nan")
                        )
                        null_degree[perm_idx] = delta_deg
                        null_rows.append(
                            {
                                "null_kind": "degree_orientation_null",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_deg),
                            }
                        )

                        sign_flip = rng.choice([-1.0, 1.0], size=margins.size, replace=True)
                        margins_sign = np.abs(margins) * sign_flip
                        directed_sign = directed_signed_score_matrix(
                            n_nodes=edge_gene_indices.size,
                            edges=knn_edges,
                            dists=knn_dists,
                            margins=margins_sign,
                            d_quantiles=[0.35, 0.50, 0.65, 0.80],
                            m_quantiles=[0.40, 0.60, 0.80],
                        )
                        eval_dir_sign = directed_sign[source_local, target_local]
                        auc_sign = safe_auc(labels, eval_dir_sign)
                        delta_sign = (
                            float(auc_sign - auc_dist)
                            if np.isfinite(auc_sign) and np.isfinite(auc_dist)
                            else float("nan")
                        )
                        null_sign[perm_idx] = delta_sign
                        null_rows.append(
                            {
                                "null_kind": "sign_shuffle_null",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_sign),
                            }
                        )

                        labels_perm = stratified_shuffle(labels, edge_deg_bins, rng=rng).astype(int)
                        auc_dir_lp = safe_auc(labels_perm, eval_dir)
                        auc_dist_lp = safe_auc(labels_perm, eval_dist)
                        delta_lp = (
                            float(auc_dir_lp - auc_dist_lp)
                            if np.isfinite(auc_dir_lp) and np.isfinite(auc_dist_lp)
                            else float("nan")
                        )
                        null_label[perm_idx] = delta_lp
                        null_rows.append(
                            {
                                "null_kind": "split_label_placebo",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_lp),
                            }
                        )

                    p_degree = empirical_upper_tail_p(delta_auc, null_degree)
                    p_sign = empirical_upper_tail_p(delta_auc, null_sign)
                    p_label = empirical_upper_tail_p(delta_auc, null_label)
                    p_best = np.nanmin(np.array([p_degree, p_sign, p_label], dtype=float))

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
                            "auc_directed_signed": float(auc_dir),
                            "auc_distance_only": float(auc_dist),
                            "delta_auc_directed_minus_distance": float(delta_auc),
                            "null_degree_mean_delta": float(np.nanmean(null_degree)),
                            "null_sign_mean_delta": float(np.nanmean(null_sign)),
                            "null_label_mean_delta": float(np.nanmean(null_label)),
                            "p_degree_upper": float(p_degree),
                            "p_sign_upper": float(p_sign),
                            "p_label_upper": float(p_label),
                            "p_best_upper": float(p_best),
                        }
                    )

    by_row_df = pd.DataFrame(rows).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h52_directed_signed_multiseed_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"]
    )
    null_path = ITER_DIR / "h52_directed_signed_multiseed_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
        summary_rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(group.shape[0]),
                "mean_auc_directed_signed": float(group["auc_directed_signed"].mean()),
                "mean_auc_distance_only": float(group["auc_distance_only"].mean()),
                "mean_delta_auc_directed_minus_distance": float(
                    group["delta_auc_directed_minus_distance"].mean()
                ),
                "fraction_delta_positive": float((group["delta_auc_directed_minus_distance"] > 0.0).mean()),
                "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                "combined_fisher_p_best": float(safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h52_directed_signed_multiseed_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_directed_minus_distance"].mean())
        if not by_row_df.empty
        else float("nan"),
        "fraction_delta_positive": float((by_row_df["delta_auc_directed_minus_distance"] > 0.0).mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_split_fisher_sig": int((summary_df["combined_fisher_p_best"] < 0.05).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_layer_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h53_directed_path_homology(
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

            top_genes = set(select_top_genes(split_edges, gene_cap=H53_GENE_CAP))
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
            if edge_gene_indices.size < 90:
                continue

            gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
            source_local = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
            target_local = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
            labels = split_edges["label"].to_numpy(dtype=int)

            symbol_map = build_symbol_map(split_edges)
            ordered_symbols = [symbol_map[int(g)] for g in edge_gene_indices]
            _, support_directed = build_support_matrices(
                symbols_upper=ordered_symbols,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )

            for layer in H53_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = reduce_points(
                    points,
                    n_components=18,
                    random_state=23_530 + domain_index * 100 + split_index * 20 + layer,
                )

                knn_edges, knn_dists = build_knn_edge_array(points=points_pca, n_neighbors=H53_KNN)
                if knn_edges.shape[0] < 70:
                    continue

                margins = np.array(
                    [support_directed[i, j] - support_directed[j, i] for i, j in knn_edges],
                    dtype=float,
                )

                base_rng = np.random.default_rng(
                    23_531 + domain_index * 100 + split_index * 20 + layer
                )
                path_matrix = directed_path_homology_score_matrix(
                    n_nodes=edge_gene_indices.size,
                    edges=knn_edges,
                    dists=knn_dists,
                    margins=margins,
                    d_quantiles=[0.50, 0.70],
                    m_quantiles=[0.50, 0.75],
                    max_triangles=H53_MAX_TRIANGLES,
                    rng=base_rng,
                )
                dist_matrix = distance_cycle_score_matrix(
                    n_nodes=edge_gene_indices.size,
                    edges=knn_edges,
                    dists=knn_dists,
                    d_quantiles=[0.40, 0.60, 0.80],
                )

                eval_path = path_matrix[source_local, target_local]
                eval_dist = dist_matrix[source_local, target_local]
                auc_path = safe_auc(labels, eval_path)
                auc_dist = safe_auc(labels, eval_dist)
                delta_auc = (
                    float(auc_path - auc_dist)
                    if np.isfinite(auc_path) and np.isfinite(auc_dist)
                    else float("nan")
                )

                rng = np.random.default_rng(
                    23_532 + domain_index * 100 + split_index * 20 + layer
                )
                null_degree = np.empty(H53_NULL_PERM, dtype=float)
                null_sign = np.empty(H53_NULL_PERM, dtype=float)
                null_random_map = np.empty(H53_NULL_PERM, dtype=float)

                for perm_idx in range(H53_NULL_PERM):
                    perm = rng.permutation(edge_gene_indices.size)
                    inv = np.empty_like(perm)
                    inv[perm] = np.arange(perm.size)
                    eval_path_deg = path_matrix[inv[source_local], inv[target_local]]
                    auc_deg = safe_auc(labels, eval_path_deg)
                    delta_deg = (
                        float(auc_deg - auc_dist) if np.isfinite(auc_deg) and np.isfinite(auc_dist) else float("nan")
                    )
                    null_degree[perm_idx] = delta_deg
                    null_rows.append(
                        {
                            "null_kind": "degree_orientation_null",
                            "domain": domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_deg),
                        }
                    )

                    sign_flip = rng.choice([-1.0, 1.0], size=margins.size, replace=True)
                    margins_sign = np.abs(margins) * sign_flip
                    sign_rng = np.random.default_rng(
                        23_700 + domain_index * 1000 + split_index * 100 + layer * 10 + perm_idx
                    )
                    path_sign = directed_path_homology_score_matrix(
                        n_nodes=edge_gene_indices.size,
                        edges=knn_edges,
                        dists=knn_dists,
                        margins=margins_sign,
                        d_quantiles=[0.50, 0.70],
                        m_quantiles=[0.50, 0.75],
                        max_triangles=H53_MAX_TRIANGLES,
                        rng=sign_rng,
                    )
                    eval_path_sign = path_sign[source_local, target_local]
                    auc_sign = safe_auc(labels, eval_path_sign)
                    delta_sign = (
                        float(auc_sign - auc_dist)
                        if np.isfinite(auc_sign) and np.isfinite(auc_dist)
                        else float("nan")
                    )
                    null_sign[perm_idx] = delta_sign
                    null_rows.append(
                        {
                            "null_kind": "sign_shuffle_null",
                            "domain": domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_sign),
                        }
                    )

                    margins_map = rng.permutation(margins)
                    map_rng = np.random.default_rng(
                        23_900 + domain_index * 1000 + split_index * 100 + layer * 10 + perm_idx
                    )
                    path_map = directed_path_homology_score_matrix(
                        n_nodes=edge_gene_indices.size,
                        edges=knn_edges,
                        dists=knn_dists,
                        margins=margins_map,
                        d_quantiles=[0.50, 0.70],
                        m_quantiles=[0.50, 0.75],
                        max_triangles=H53_MAX_TRIANGLES,
                        rng=map_rng,
                    )
                    eval_path_map = path_map[source_local, target_local]
                    auc_map = safe_auc(labels, eval_path_map)
                    delta_map = (
                        float(auc_map - auc_dist)
                        if np.isfinite(auc_map) and np.isfinite(auc_dist)
                        else float("nan")
                    )
                    null_random_map[perm_idx] = delta_map
                    null_rows.append(
                        {
                            "null_kind": "random_map_null",
                            "domain": domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_map),
                        }
                    )

                p_degree = empirical_upper_tail_p(delta_auc, null_degree)
                p_sign = empirical_upper_tail_p(delta_auc, null_sign)
                p_map = empirical_upper_tail_p(delta_auc, null_random_map)
                p_best = np.nanmin(np.array([p_degree, p_sign, p_map], dtype=float))

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
                        "auc_path_homology": float(auc_path),
                        "auc_distance_only": float(auc_dist),
                        "delta_auc_path_minus_distance": float(delta_auc),
                        "null_degree_mean_delta": float(np.nanmean(null_degree)),
                        "null_sign_mean_delta": float(np.nanmean(null_sign)),
                        "null_random_map_mean_delta": float(np.nanmean(null_random_map)),
                        "p_degree_upper": float(p_degree),
                        "p_sign_upper": float(p_sign),
                        "p_random_map_upper": float(p_map),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(rows).sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h53_directed_path_homology_by_domain_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h53_directed_path_homology_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
        summary_rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(group.shape[0]),
                "mean_auc_path_homology": float(group["auc_path_homology"].mean()),
                "mean_auc_distance_only": float(group["auc_distance_only"].mean()),
                "mean_delta_auc_path_minus_distance": float(group["delta_auc_path_minus_distance"].mean()),
                "fraction_delta_positive": float((group["delta_auc_path_minus_distance"] > 0.0).mean()),
                "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                "combined_fisher_p_best": float(safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h53_directed_path_homology_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_path_minus_distance"].mean())
        if not by_row_df.empty
        else float("nan"),
        "fraction_delta_positive": float((by_row_df["delta_auc_path_minus_distance"] > 0.0).mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_split_fisher_sig": int((summary_df["combined_fisher_p_best"] < 0.05).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_domain_layer_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h54_linearity_rupture() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_index, (seed_tag, run_dir) in enumerate(run_map.items()):
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            split_masks = build_split_masks(edge_df)

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                top_genes = set(select_top_genes(split_edges, gene_cap=H54_GENE_CAP))
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

                gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
                source_local = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
                target_local = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
                labels = split_edges["label"].to_numpy(dtype=int)

                node_deg = np.bincount(
                    np.concatenate([source_local, target_local]),
                    minlength=edge_gene_indices.size,
                )
                edge_deg = node_deg[source_local] + node_deg[target_local]
                edge_deg_bins = degree_bins(edge_deg, max_bins=5)

                layer_errors: list[np.ndarray] = []
                edge_baseline_scores: list[np.ndarray] = []

                for layer in H54_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue
                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = reduce_points(
                        points,
                        n_components=20,
                        random_state=23_540 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer,
                    )
                    neigh_idx = get_knn_indices(points_pca, n_neighbors=H54_NEIGHBORS)
                    recon_errors = compute_local_reconstruction_errors(points_pca, neigh_idx)
                    layer_errors.append(recon_errors)
                    # Baseline is local linearity itself (smaller reconstruction error = more linear neighborhood).
                    edge_baseline_scores.append(-0.5 * (recon_errors[source_local] + recon_errors[target_local]))

                if len(layer_errors) != len(H54_LAYERS):
                    continue

                layer_error_matrix = np.vstack(layer_errors)
                rupture_matrix = compute_rupture_from_layer_errors(layer_error_matrix)

                rng = np.random.default_rng(
                    23_541 + domain_index * 1000 + seed_index * 100 + split_index * 20
                )

                for layer_pos, layer in enumerate(H54_LAYERS):
                    rupture_scores = 0.5 * (
                        rupture_matrix[layer_pos, source_local] + rupture_matrix[layer_pos, target_local]
                    )
                    baseline_scores = edge_baseline_scores[layer_pos]

                    auc_rupture = safe_auc(labels, rupture_scores)
                    auc_baseline = safe_auc(labels, baseline_scores)
                    delta_auc = (
                        float(auc_rupture - auc_baseline)
                        if np.isfinite(auc_rupture) and np.isfinite(auc_baseline)
                        else float("nan")
                    )

                    null_layer_shuffle = np.empty(H54_NULL_PERM, dtype=float)
                    null_endpoint_swap = np.empty(H54_NULL_PERM, dtype=float)
                    null_label_perm = np.empty(H54_NULL_PERM, dtype=float)

                    for perm_idx in range(H54_NULL_PERM):
                        perm_order = rng.permutation(len(H54_LAYERS))
                        perm_error_matrix = layer_error_matrix[perm_order]
                        perm_rupture_matrix = compute_rupture_from_layer_errors(perm_error_matrix)
                        perm_scores = 0.5 * (
                            perm_rupture_matrix[layer_pos, source_local]
                            + perm_rupture_matrix[layer_pos, target_local]
                        )
                        auc_layer_perm = safe_auc(labels, perm_scores)
                        delta_layer = (
                            float(auc_layer_perm - auc_baseline)
                            if np.isfinite(auc_layer_perm) and np.isfinite(auc_baseline)
                            else float("nan")
                        )
                        null_layer_shuffle[perm_idx] = delta_layer
                        null_rows.append(
                            {
                                "null_kind": "layer_order_shuffle",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_layer),
                            }
                        )

                        swapped_target = target_local.copy()
                        for b in np.unique(edge_deg_bins):
                            idx = np.where(edge_deg_bins == b)[0]
                            if idx.size > 1:
                                swapped_target[idx] = rng.permutation(swapped_target[idx])
                        swap_scores = 0.5 * (
                            rupture_matrix[layer_pos, source_local] + rupture_matrix[layer_pos, swapped_target]
                        )
                        auc_swap = safe_auc(labels, swap_scores)
                        delta_swap = (
                            float(auc_swap - auc_baseline)
                            if np.isfinite(auc_swap) and np.isfinite(auc_baseline)
                            else float("nan")
                        )
                        null_endpoint_swap[perm_idx] = delta_swap
                        null_rows.append(
                            {
                                "null_kind": "endpoint_swap_degree_bin",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_swap),
                            }
                        )

                        labels_perm = stratified_shuffle(labels, edge_deg_bins, rng=rng).astype(int)
                        auc_r_perm = safe_auc(labels_perm, rupture_scores)
                        auc_b_perm = safe_auc(labels_perm, baseline_scores)
                        delta_label = (
                            float(auc_r_perm - auc_b_perm)
                            if np.isfinite(auc_r_perm) and np.isfinite(auc_b_perm)
                            else float("nan")
                        )
                        null_label_perm[perm_idx] = delta_label
                        null_rows.append(
                            {
                                "null_kind": "label_permutation",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_label),
                            }
                        )

                    p_layer = empirical_upper_tail_p(delta_auc, null_layer_shuffle)
                    p_endpoint = empirical_upper_tail_p(delta_auc, null_endpoint_swap)
                    p_label = empirical_upper_tail_p(delta_auc, null_label_perm)
                    p_best = np.nanmin(np.array([p_layer, p_endpoint, p_label], dtype=float))

                    rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "n_edges_eval": int(labels.size),
                            "n_positive_eval": int(labels.sum()),
                            "n_nodes_graph": int(edge_gene_indices.size),
                            "auc_rupture": float(auc_rupture),
                            "auc_local_linearity_baseline": float(auc_baseline),
                            "delta_auc_rupture_minus_baseline": float(delta_auc),
                            "null_layer_mean_delta": float(np.nanmean(null_layer_shuffle)),
                            "null_endpoint_mean_delta": float(np.nanmean(null_endpoint_swap)),
                            "null_label_mean_delta": float(np.nanmean(null_label_perm)),
                            "p_layer_upper": float(p_layer),
                            "p_endpoint_upper": float(p_endpoint),
                            "p_label_upper": float(p_label),
                            "p_best_upper": float(p_best),
                        }
                    )

    by_row_df = pd.DataFrame(rows).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h54_linearity_rupture_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"]
    )
    null_path = ITER_DIR / "h54_linearity_rupture_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
        summary_rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(group.shape[0]),
                "mean_auc_rupture": float(group["auc_rupture"].mean()),
                "mean_auc_local_linearity_baseline": float(group["auc_local_linearity_baseline"].mean()),
                "mean_delta_auc_rupture_minus_baseline": float(group["delta_auc_rupture_minus_baseline"].mean()),
                "fraction_delta_positive": float((group["delta_auc_rupture_minus_baseline"] > 0.0).mean()),
                "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                "combined_fisher_p_best": float(safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h54_linearity_rupture_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_rupture_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "fraction_delta_positive": float((by_row_df["delta_auc_rupture_minus_baseline"] > 0.0).mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_split_fisher_sig": int((summary_df["combined_fisher_p_best"] < 0.05).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_layer_split": str(by_row_path),
            "domain_summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def main() -> None:
    required_paths = [DOROTHEA_PATH, GENE2GO_PATH, OMNIPATH_INTERACTIONS_PATH, STRING_CACHE_PATH]
    for run_map in SCGPT_RUNS_BY_DOMAIN.values():
        for run_dir in run_map.values():
            required_paths.append(run_dir / "cycle1_edge_dataset.tsv")
            required_paths.append(run_dir / "layer_gene_embeddings.npy")

    missing = [str(p) for p in required_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    dorothea_map = load_dorothea_score_map()
    omnipath_pairs = load_omnipath_pairs()
    gene2go_upper = load_gene2go_upper()
    string_map = load_string_scores_from_cache(STRING_CACHE_PATH)

    h52_summary = run_h52_directed_signed_multiseed(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h53_summary = run_h53_directed_path_homology(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h54_summary = run_h54_linearity_rupture()

    summary = {
        "iteration": "iter_0023",
        "h52": h52_summary,
        "h53": h53_summary,
        "h54": h54_summary,
        "environment": {
            "python_env": "subproject40-topology",
            "string_cache_used": str(STRING_CACHE_PATH),
        },
    }
    summary_path = ITER_DIR / "iter0023_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
