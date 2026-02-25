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


ITER_DIR = Path("iterations/iter_0024")
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

# H55: directed/signed topology with increased null resolution + failure-slice diagnostics.
H55_LAYERS = [7, 11]
H55_GENE_CAP = 150
H55_KNN = 10
H55_NULL_PERM = 64

# H56: densified directed path-homology rescue with utility-transfer endpoint.
H56_LAYERS = [7, 11]
H56_GENE_CAP = 150
H56_KNN_SWEEP = [8, 12]
H56_NULL_PERM = 24
H56_MAX_TRIANGLES = 2400

# H57: geodesic neighborhood anisotropy-tail broad screen.
H57_LAYERS = [0, 3, 7, 11]
H57_GENE_CAP = 170
H57_NEIGHBORS = 14
H57_NULL_PERM = 24


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    if np.unique(scores).size < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def safe_precision(labels: np.ndarray, preds: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    tp = int(np.sum((labels == 1) & (preds == 1)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def safe_recall(labels: np.ndarray, preds: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    tp = int(np.sum((labels == 1) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def safe_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    p = safe_precision(labels, preds)
    r = safe_recall(labels, preds)
    if p + r <= 0.0:
        return 0.0
    return float(2.0 * p * r / (p + r))


def choose_best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.size == 0:
        return 0.0
    quantiles = np.linspace(0.05, 0.95, 37)
    thresholds = np.unique(np.quantile(scores, quantiles))
    if thresholds.size == 0:
        return float(np.median(scores))

    best_t = float(thresholds[0])
    best_f1 = -1.0
    for threshold in thresholds:
        preds = (scores >= float(threshold)).astype(int)
        f1 = safe_f1(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(threshold)
    return best_t


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

    d_thresholds = np.unique(np.quantile(dists, d_quantiles))
    for d_thr in d_thresholds:
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

    d_thresholds = np.unique(np.quantile(dists, d_quantiles))
    for d_thr in d_thresholds:
        mask_d = dists <= float(d_thr)
        if int(mask_d.sum()) < 6:
            continue

        edges_d = edges[mask_d]
        margins_d = margins[mask_d]
        abs_margin_d = np.abs(margins_d)
        if np.all(abs_margin_d <= 0.0):
            continue

        m_thresholds = np.unique(np.quantile(abs_margin_d, m_quantiles))
        for m_thr in m_thresholds:
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
            signed_component = np.tanh(3.5 * signed)
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


def build_symbol_map(split_edges: pd.DataFrame) -> dict[int, str]:
    symbol_map: dict[int, str] = {}
    for row in split_edges[["source_idx", "source"]].drop_duplicates().itertuples(index=False):
        symbol_map[int(row.source_idx)] = str(row.source).upper()
    for row in split_edges[["target_idx", "target"]].drop_duplicates().itertuples(index=False):
        symbol_map[int(row.target_idx)] = str(row.target).upper()
    return symbol_map


def get_knn_indices(points: np.ndarray, n_neighbors: int) -> np.ndarray:
    k = max(2, min(n_neighbors, points.shape[0] - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(points)
    _, indices = nbrs.kneighbors(points)
    return indices[:, 1:]


def oriented_edges_from_margins(edges: np.ndarray, margins: np.ndarray) -> np.ndarray:
    out = np.empty_like(edges)
    for idx, ((u, v), margin) in enumerate(zip(edges, margins)):
        if float(margin) >= 0.0:
            out[idx, 0] = int(u)
            out[idx, 1] = int(v)
        else:
            out[idx, 0] = int(v)
            out[idx, 1] = int(u)
    return out


def graph_regime_diagnostics(
    n_nodes: int,
    edges: np.ndarray,
    margins: np.ndarray,
) -> dict[str, float]:
    oriented = oriented_edges_from_margins(edges, margins)
    m = oriented.shape[0]
    if m == 0:
        return {
            "orientation_entropy": 0.0,
            "sign_balance": 0.0,
            "knn_density": 0.0,
            "degree_assortativity": 0.0,
            "margin_iqr": 0.0,
        }

    pos = float(np.mean(margins >= 0.0))
    neg = 1.0 - pos
    entropy = 0.0
    for p in [pos, neg]:
        if p > 0:
            entropy -= p * float(np.log2(p))
    entropy = entropy / np.log2(2.0)

    sign_balance = float(pos - neg)
    density = float((2.0 * m) / max(1.0, n_nodes * (n_nodes - 1)))

    out_degree = np.bincount(oriented[:, 0], minlength=n_nodes).astype(float)
    in_degree = np.bincount(oriented[:, 1], minlength=n_nodes).astype(float)
    src_deg = out_degree[oriented[:, 0]]
    tgt_deg = in_degree[oriented[:, 1]]
    if np.std(src_deg) < 1e-8 or np.std(tgt_deg) < 1e-8:
        assort = 0.0
    else:
        assort = float(np.corrcoef(src_deg, tgt_deg)[0, 1])

    margin_iqr = float(np.quantile(np.abs(margins), 0.75) - np.quantile(np.abs(margins), 0.25))
    return {
        "orientation_entropy": float(entropy),
        "sign_balance": sign_balance,
        "knn_density": density,
        "degree_assortativity": assort,
        "margin_iqr": margin_iqr,
    }


def metric_matched_randomized_knn(
    n_nodes: int,
    n_edges: int,
    dists: np.ndarray,
    node_degree: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if n_edges <= 0:
        return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=float)

    probs = np.asarray(node_degree, dtype=float)
    probs = np.clip(probs, 1e-3, None)
    probs = probs / probs.sum()

    edge_set: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = n_edges * 40
    while len(edge_set) < n_edges and attempts < max_attempts:
        attempts += 1
        u = int(rng.choice(n_nodes, p=probs))
        v = int(rng.choice(n_nodes, p=probs))
        if u == v:
            continue
        a, b = sorted((u, v))
        edge_set.add((a, b))

    if len(edge_set) < max(4, n_edges // 2):
        # Fallback: deterministic fill to avoid pathological duplicate-heavy draws.
        for u in range(n_nodes):
            for v in range(u + 1, n_nodes):
                edge_set.add((u, v))
                if len(edge_set) >= n_edges:
                    break
            if len(edge_set) >= n_edges:
                break

    edges = np.array(sorted(edge_set)[:n_edges], dtype=int)
    if edges.shape[0] == 0:
        return edges, np.zeros(0, dtype=float)

    dist_perm = rng.permutation(np.asarray(dists, dtype=float))
    if dist_perm.size < edges.shape[0]:
        repeats = int(np.ceil(edges.shape[0] / max(1, dist_perm.size)))
        dist_perm = np.tile(dist_perm, repeats)
    out_dists = dist_perm[: edges.shape[0]].copy()
    return edges, out_dists


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


def directed_path_homology_v2_score_matrix(
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
        if int(mask_d.sum()) < 10:
            continue

        edges_d = edges[mask_d]
        margins_d = margins[mask_d]
        abs_margin_d = np.abs(margins_d)
        if np.all(abs_margin_d <= 0.0):
            continue

        m_thresholds = np.unique(np.quantile(abs_margin_d, m_quantiles))
        for m_thr in m_thresholds:
            strong = abs_margin_d >= float(m_thr)
            if int(strong.sum()) < 10:
                continue

            edges_s = edges_d[strong]
            margins_s = margins_d[strong]

            oriented_edges: list[tuple[int, int]] = []
            signed = np.zeros((n_nodes, n_nodes), dtype=float)
            for (u, v), margin in zip(edges_s, margins_s):
                iu = int(u)
                iv = int(v)
                mval = float(margin)
                if mval >= 0.0:
                    oriented_edges.append((iu, iv))
                else:
                    oriented_edges.append((iv, iu))
                signed[iu, iv] = mval
                signed[iv, iu] = -mval

            beta1 = directed_flag_beta1(
                n_nodes=n_nodes,
                directed_edges=oriented_edges,
                max_triangles=max_triangles,
                rng=rng,
            )

            adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
            for u, v in oriented_edges:
                adj[u, v] = 1

            path2 = adj @ adj
            path3 = path2 @ adj
            path_signal = (path2 - path2.T).astype(float) + 0.35 * (path3 - path3.T).astype(float)
            signed_component = 0.20 * np.tanh(3.0 * signed)

            for u, v in oriented_edges:
                value = beta1 + path_signal[u, v] + signed_component[u, v]
                total[u, v] += value
                count[u, v] += 1.0

    return total / np.clip(count, 1.0, None)


def compute_local_anisotropy_tail(
    points: np.ndarray,
    neighbor_idx: np.ndarray,
    neighbor_dists: np.ndarray,
) -> np.ndarray:
    n_nodes = points.shape[0]
    out = np.zeros(n_nodes, dtype=float)

    for i in range(n_nodes):
        neigh = neighbor_idx[i]
        x = points[neigh] - points[i]
        if x.shape[0] < 3:
            out[i] = 0.0
            continue

        cov = (x.T @ x) / max(1, x.shape[0] - 1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.clip(eigvals, 1e-8, None)
        spectral_ratio = float(eigvals[-1] / eigvals[0])

        d = np.asarray(neighbor_dists[i], dtype=float)
        tail_ratio = float(np.quantile(d, 0.90) / max(1e-8, np.quantile(d, 0.50)))
        out[i] = float(np.log1p(spectral_ratio) * tail_ratio)

    return out


def run_h55_directed_signed_highperm(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    diag_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(SCGPT_RUNS_BY_DOMAIN.items()):
        for seed_index, (seed_tag, run_dir) in enumerate(run_map.items()):
            edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            split_masks = build_split_masks(edge_df)

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                top_genes = set(select_top_genes(split_edges, gene_cap=H55_GENE_CAP))
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
                if edge_gene_indices.size < 110:
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

                for layer in H55_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = reduce_points(
                        points,
                        n_components=20,
                        random_state=24_550 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer,
                    )

                    knn_edges, knn_dists = build_knn_edge_array(points=points_pca, n_neighbors=H55_KNN)
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
                        d_quantiles=[0.40, 0.55, 0.70],
                        m_quantiles=[0.45, 0.65, 0.85],
                    )
                    dist_matrix = distance_cycle_score_matrix(
                        n_nodes=edge_gene_indices.size,
                        edges=knn_edges,
                        dists=knn_dists,
                        d_quantiles=[0.40, 0.55, 0.70],
                    )

                    eval_dir = directed_matrix[source_local, target_local]
                    eval_dist = dist_matrix[source_local, target_local]

                    auc_dir = safe_auc(labels, eval_dir)
                    auc_dist = safe_auc(labels, eval_dist)
                    delta_auc = (
                        float(auc_dir - auc_dist) if np.isfinite(auc_dir) and np.isfinite(auc_dist) else float("nan")
                    )

                    diagnostics = graph_regime_diagnostics(
                        n_nodes=edge_gene_indices.size,
                        edges=knn_edges,
                        margins=margins,
                    )

                    rng = np.random.default_rng(
                        24_551 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer
                    )

                    null_degree = np.empty(H55_NULL_PERM, dtype=float)
                    null_sign = np.empty(H55_NULL_PERM, dtype=float)
                    null_label = np.empty(H55_NULL_PERM, dtype=float)
                    null_metric = np.empty(H55_NULL_PERM, dtype=float)

                    for perm_idx in range(H55_NULL_PERM):
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
                            d_quantiles=[0.40, 0.55, 0.70],
                            m_quantiles=[0.45, 0.65, 0.85],
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

                        rand_edges, rand_dists = metric_matched_randomized_knn(
                            n_nodes=edge_gene_indices.size,
                            n_edges=int(knn_edges.shape[0]),
                            dists=knn_dists,
                            node_degree=node_deg,
                            rng=rng,
                        )
                        if rand_edges.shape[0] < 10:
                            null_metric[perm_idx] = float("nan")
                        else:
                            rand_margins = rng.permutation(margins)
                            if rand_margins.size < rand_edges.shape[0]:
                                repeats = int(np.ceil(rand_edges.shape[0] / max(1, rand_margins.size)))
                                rand_margins = np.tile(rand_margins, repeats)
                            rand_margins = rand_margins[: rand_edges.shape[0]]

                            directed_rand = directed_signed_score_matrix(
                                n_nodes=edge_gene_indices.size,
                                edges=rand_edges,
                                dists=rand_dists,
                                margins=rand_margins,
                                d_quantiles=[0.40, 0.55, 0.70],
                                m_quantiles=[0.45, 0.65, 0.85],
                            )
                            dist_rand = distance_cycle_score_matrix(
                                n_nodes=edge_gene_indices.size,
                                edges=rand_edges,
                                dists=rand_dists,
                                d_quantiles=[0.40, 0.55, 0.70],
                            )
                            eval_dir_rand = directed_rand[source_local, target_local]
                            eval_dist_rand = dist_rand[source_local, target_local]
                            auc_rand = safe_auc(labels, eval_dir_rand)
                            auc_rand_base = safe_auc(labels, eval_dist_rand)
                            delta_metric_perm = (
                                float(auc_rand - auc_rand_base)
                                if np.isfinite(auc_rand) and np.isfinite(auc_rand_base)
                                else float("nan")
                            )
                            null_metric[perm_idx] = delta_metric_perm

                        null_rows.append(
                            {
                                "null_kind": "metric_matched_knn_randomization",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(null_metric[perm_idx]),
                            }
                        )

                    p_degree = empirical_upper_tail_p(delta_auc, null_degree)
                    p_sign = empirical_upper_tail_p(delta_auc, null_sign)
                    p_label = empirical_upper_tail_p(delta_auc, null_label)
                    p_metric = empirical_upper_tail_p(delta_auc, null_metric)
                    p_best = np.nanmin(np.array([p_degree, p_sign, p_label, p_metric], dtype=float))

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
                            "null_metric_mean_delta": float(np.nanmean(null_metric)),
                            "p_degree_upper": float(p_degree),
                            "p_sign_upper": float(p_sign),
                            "p_label_upper": float(p_label),
                            "p_metric_upper": float(p_metric),
                            "p_best_upper": float(p_best),
                            "orientation_entropy": diagnostics["orientation_entropy"],
                            "sign_balance": diagnostics["sign_balance"],
                            "knn_density": diagnostics["knn_density"],
                            "degree_assortativity": diagnostics["degree_assortativity"],
                            "margin_iqr": diagnostics["margin_iqr"],
                        }
                    )

                    diag_rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "delta_auc_directed_minus_distance": float(delta_auc),
                            "is_failure_slice": int(domain == "lung" and split_regime == "source_disjoint"),
                            **diagnostics,
                        }
                    )

    by_row_df = pd.DataFrame(rows).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h55_directed_signed_highperm_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"]
    )
    null_path = ITER_DIR / "h55_directed_signed_highperm_null_summary.csv"
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
    summary_path = ITER_DIR / "h55_directed_signed_highperm_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    diag_df = pd.DataFrame(diag_rows)
    assoc_rows: list[dict[str, object]] = []
    for col in ["orientation_entropy", "sign_balance", "knn_density", "degree_assortativity", "margin_iqr"]:
        x = diag_df[col].to_numpy(dtype=float)
        y = diag_df["delta_auc_directed_minus_distance"].to_numpy(dtype=float)
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x, y)[0, 1])

        lung_fail = diag_df.loc[
            (diag_df["domain"] == "lung") & (diag_df["split_regime"] == "source_disjoint"), col
        ]
        other = diag_df.loc[
            ~((diag_df["domain"] == "lung") & (diag_df["split_regime"] == "source_disjoint")), col
        ]
        diff = float(lung_fail.mean() - other.mean()) if not lung_fail.empty and not other.empty else float("nan")
        assoc_rows.append(
            {
                "domain": "all",
                "seed_tag": "summary",
                "split_regime": "all",
                "layer": -1,
                "delta_auc_directed_minus_distance": float("nan"),
                "is_failure_slice": 2,
                "orientation_entropy": float("nan"),
                "sign_balance": float("nan"),
                "knn_density": float("nan"),
                "degree_assortativity": float("nan"),
                "margin_iqr": float("nan"),
                "diagnostic_name": col,
                "delta_corr_global": corr,
                "failure_slice_minus_other_mean": diff,
            }
        )

    diag_df["diagnostic_name"] = "row"
    diag_df["delta_corr_global"] = float("nan")
    diag_df["failure_slice_minus_other_mean"] = float("nan")
    diag_out = pd.concat([diag_df, pd.DataFrame(assoc_rows)], ignore_index=True)
    diag_path = ITER_DIR / "h55_directed_signed_failure_slice_diagnostics.csv"
    diag_out.to_csv(diag_path, index=False)

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
            "failure_slice_diagnostics": str(diag_path),
        },
    }


def run_h56_path_homology_v2(
    dorothea_map: dict[tuple[str, str], int],
    omnipath_pairs: set[tuple[str, str]],
    gene2go_upper: dict[str, set[str]],
    string_map: dict[tuple[str, str], float],
) -> dict[str, object]:
    by_split_rows: list[dict[str, object]] = []
    transfer_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for domain_index, (domain, run_map) in enumerate(SCGPT_RUNS_BY_DOMAIN.items()):
        seed_tag = "seed42_main"
        run_dir = run_map[seed_tag]

        edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        split_masks = build_split_masks(edge_df)

        for layer in H56_LAYERS:
            if layer >= layer_embeddings.shape[0]:
                continue

            split_payload: dict[str, dict[str, object]] = {}

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                top_genes = set(select_top_genes(split_edges, gene_cap=H56_GENE_CAP))
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

                symbol_map = build_symbol_map(split_edges)
                ordered_symbols = [symbol_map[int(g)] for g in edge_gene_indices]
                _, support_directed = build_support_matrices(
                    symbols_upper=ordered_symbols,
                    dorothea_map=dorothea_map,
                    omnipath_pairs=omnipath_pairs,
                    gene2go_upper=gene2go_upper,
                    string_map=string_map,
                )

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = reduce_points(
                    points,
                    n_components=20,
                    random_state=24_560 + domain_index * 100 + split_index * 20 + layer,
                )

                matrices_path: list[np.ndarray] = []
                matrices_dist: list[np.ndarray] = []
                knn_edges_cache: list[np.ndarray] = []
                knn_dists_cache: list[np.ndarray] = []
                margins_cache: list[np.ndarray] = []

                for k_idx, knn_k in enumerate(H56_KNN_SWEEP):
                    knn_edges, knn_dists = build_knn_edge_array(points=points_pca, n_neighbors=knn_k)
                    if knn_edges.shape[0] < 80:
                        continue

                    margins = np.array(
                        [support_directed[i, j] - support_directed[j, i] for i, j in knn_edges],
                        dtype=float,
                    )
                    rng_matrix = np.random.default_rng(
                        24_561 + domain_index * 1000 + split_index * 100 + layer * 10 + k_idx
                    )
                    path_matrix = directed_path_homology_v2_score_matrix(
                        n_nodes=edge_gene_indices.size,
                        edges=knn_edges,
                        dists=knn_dists,
                        margins=margins,
                        d_quantiles=[0.45, 0.60, 0.75, 0.85],
                        m_quantiles=[0.40, 0.60, 0.80],
                        max_triangles=H56_MAX_TRIANGLES,
                        rng=rng_matrix,
                    )
                    dist_matrix = distance_cycle_score_matrix(
                        n_nodes=edge_gene_indices.size,
                        edges=knn_edges,
                        dists=knn_dists,
                        d_quantiles=[0.45, 0.60, 0.75, 0.85],
                    )
                    matrices_path.append(path_matrix)
                    matrices_dist.append(dist_matrix)
                    knn_edges_cache.append(knn_edges)
                    knn_dists_cache.append(knn_dists)
                    margins_cache.append(margins)

                if not matrices_path:
                    continue

                path_matrix_mean = np.mean(np.stack(matrices_path, axis=0), axis=0)
                dist_matrix_mean = np.mean(np.stack(matrices_dist, axis=0), axis=0)

                path_scores = path_matrix_mean[source_local, target_local]
                dist_scores = dist_matrix_mean[source_local, target_local]

                auc_path = safe_auc(labels, path_scores)
                auc_dist = safe_auc(labels, dist_scores)
                delta_auc = (
                    float(auc_path - auc_dist)
                    if np.isfinite(auc_path) and np.isfinite(auc_dist)
                    else float("nan")
                )

                rng = np.random.default_rng(24_562 + domain_index * 100 + split_index * 20 + layer)
                null_rewire = np.empty(H56_NULL_PERM, dtype=float)
                null_sign = np.empty(H56_NULL_PERM, dtype=float)

                for perm_idx in range(H56_NULL_PERM):
                    perm = rng.permutation(edge_gene_indices.size)
                    inv = np.empty_like(perm)
                    inv[perm] = np.arange(perm.size)
                    auc_rewire = safe_auc(labels, path_matrix_mean[inv[source_local], inv[target_local]])
                    delta_rewire = (
                        float(auc_rewire - auc_dist)
                        if np.isfinite(auc_rewire) and np.isfinite(auc_dist)
                        else float("nan")
                    )
                    null_rewire[perm_idx] = delta_rewire
                    null_rows.append(
                        {
                            "null_kind": "directed_degree_rewire",
                            "domain": domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "direction": "na",
                            "perm_idx": int(perm_idx),
                            "null_value": float(delta_rewire),
                        }
                    )

                    sign_mats: list[np.ndarray] = []
                    for k_idx, knn_edges in enumerate(knn_edges_cache):
                        knn_dists = knn_dists_cache[k_idx]
                        margins = margins_cache[k_idx]
                        sign_flip = rng.choice([-1.0, 1.0], size=margins.size, replace=True)
                        margins_sign = np.abs(margins) * sign_flip
                        sign_rng = np.random.default_rng(
                            24_800 + domain_index * 1000 + split_index * 100 + layer * 10 + perm_idx + k_idx
                        )
                        sign_mats.append(
                            directed_path_homology_v2_score_matrix(
                                n_nodes=edge_gene_indices.size,
                                edges=knn_edges,
                                dists=knn_dists,
                                margins=margins_sign,
                                d_quantiles=[0.45, 0.60, 0.75, 0.85],
                                m_quantiles=[0.40, 0.60, 0.80],
                                max_triangles=H56_MAX_TRIANGLES,
                                rng=sign_rng,
                            )
                        )
                    sign_matrix_mean = np.mean(np.stack(sign_mats, axis=0), axis=0)
                    auc_sign = safe_auc(labels, sign_matrix_mean[source_local, target_local])
                    delta_sign = (
                        float(auc_sign - auc_dist) if np.isfinite(auc_sign) and np.isfinite(auc_dist) else float("nan")
                    )
                    null_sign[perm_idx] = delta_sign
                    null_rows.append(
                        {
                            "null_kind": "sign_shuffle_null",
                            "domain": domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "direction": "na",
                            "perm_idx": int(perm_idx),
                            "null_value": float(delta_sign),
                        }
                    )

                p_rewire = empirical_upper_tail_p(delta_auc, null_rewire)
                p_sign = empirical_upper_tail_p(delta_auc, null_sign)
                p_best = np.nanmin(np.array([p_rewire, p_sign], dtype=float))

                by_split_rows.append(
                    {
                        "domain": domain,
                        "seed_tag": seed_tag,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "n_nodes_graph": int(edge_gene_indices.size),
                        "auc_path_homology_v2": float(auc_path),
                        "auc_distance_only": float(auc_dist),
                        "delta_auc_path_minus_distance": float(delta_auc),
                        "null_rewire_mean_delta": float(np.nanmean(null_rewire)),
                        "null_sign_mean_delta": float(np.nanmean(null_sign)),
                        "p_rewire_upper": float(p_rewire),
                        "p_sign_upper": float(p_sign),
                        "p_best_upper": float(p_best),
                    }
                )

                split_payload[split_regime] = {
                    "labels": labels,
                    "path_scores": path_scores,
                    "dist_scores": dist_scores,
                }

            # Utility-transfer objective: tune threshold on one split, evaluate on the other split.
            if "source_disjoint" in split_payload and "target_disjoint" in split_payload:
                for direction, src_split, tgt_split in [
                    ("source_to_target", "source_disjoint", "target_disjoint"),
                    ("target_to_source", "target_disjoint", "source_disjoint"),
                ]:
                    src = split_payload[src_split]
                    tgt = split_payload[tgt_split]

                    src_labels = np.asarray(src["labels"], dtype=int)
                    tgt_labels = np.asarray(tgt["labels"], dtype=int)
                    src_path = np.asarray(src["path_scores"], dtype=float)
                    src_dist = np.asarray(src["dist_scores"], dtype=float)
                    tgt_path = np.asarray(tgt["path_scores"], dtype=float)
                    tgt_dist = np.asarray(tgt["dist_scores"], dtype=float)

                    thr_path = choose_best_f1_threshold(src_path, src_labels)
                    thr_dist = choose_best_f1_threshold(src_dist, src_labels)

                    tgt_preds_path = (tgt_path >= thr_path).astype(int)
                    tgt_preds_dist = (tgt_dist >= thr_dist).astype(int)

                    precision_path = safe_precision(tgt_labels, tgt_preds_path)
                    precision_dist = safe_precision(tgt_labels, tgt_preds_dist)
                    recall_path = safe_recall(tgt_labels, tgt_preds_path)
                    recall_dist = safe_recall(tgt_labels, tgt_preds_dist)
                    f1_path = safe_f1(tgt_labels, tgt_preds_path)
                    f1_dist = safe_f1(tgt_labels, tgt_preds_dist)
                    utility_lift_f1 = float(f1_path - f1_dist)

                    rng_transfer = np.random.default_rng(
                        24_563 + domain_index * 1000 + layer * 10 + (0 if direction == "source_to_target" else 1)
                    )
                    null_transfer = np.empty(H56_NULL_PERM, dtype=float)
                    for perm_idx in range(H56_NULL_PERM):
                        src_labels_perm = rng_transfer.permutation(src_labels)
                        thr_path_perm = choose_best_f1_threshold(src_path, src_labels_perm)
                        thr_dist_perm = choose_best_f1_threshold(src_dist, src_labels_perm)

                        preds_path_perm = (tgt_path >= thr_path_perm).astype(int)
                        preds_dist_perm = (tgt_dist >= thr_dist_perm).astype(int)
                        f1_path_perm = safe_f1(tgt_labels, preds_path_perm)
                        f1_dist_perm = safe_f1(tgt_labels, preds_dist_perm)
                        lift_perm = float(f1_path_perm - f1_dist_perm)
                        null_transfer[perm_idx] = lift_perm

                        null_rows.append(
                            {
                                "null_kind": "random_map_transfer_control",
                                "domain": domain,
                                "split_regime": "dual_axis_disjoint",
                                "layer": int(layer),
                                "direction": direction,
                                "perm_idx": int(perm_idx),
                                "null_value": float(lift_perm),
                            }
                        )

                    p_transfer = empirical_upper_tail_p(utility_lift_f1, null_transfer)
                    transfer_rows.append(
                        {
                            "domain": domain,
                            "layer": int(layer),
                            "direction": direction,
                            "f1_path_homology_v2": float(f1_path),
                            "f1_distance_only": float(f1_dist),
                            "f1_utility_lift": float(utility_lift_f1),
                            "precision_path_homology_v2": float(precision_path),
                            "precision_distance_only": float(precision_dist),
                            "recall_path_homology_v2": float(recall_path),
                            "recall_distance_only": float(recall_dist),
                            "p_transfer_upper": float(p_transfer),
                            "null_transfer_mean": float(np.nanmean(null_transfer)),
                        }
                    )

    by_split_df = pd.DataFrame(by_split_rows).sort_values(["domain", "split_regime", "layer"])
    by_split_path = ITER_DIR / "h56_path_homology_v2_by_domain_layer_split.csv"
    by_split_df.to_csv(by_split_path, index=False)

    transfer_df = pd.DataFrame(transfer_rows).sort_values(["domain", "layer", "direction"])
    transfer_path = ITER_DIR / "h56_path_homology_v2_utility_transfer_summary.csv"
    transfer_df.to_csv(transfer_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["null_kind", "domain", "split_regime", "layer", "direction", "perm_idx"]
    )
    null_path = ITER_DIR / "h56_path_homology_v2_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    transfer_domain_rows: list[dict[str, object]] = []
    for domain, group in transfer_df.groupby("domain", sort=True):
        transfer_domain_rows.append(
            {
                "domain": domain,
                "n_rows": int(group.shape[0]),
                "mean_f1_utility_lift": float(group["f1_utility_lift"].mean()),
                "fraction_positive_lift": float((group["f1_utility_lift"] > 0.0).mean()),
                "fraction_transfer_p_lt_0_05": float((group["p_transfer_upper"] < 0.05).mean()),
                "combined_fisher_p_transfer": float(safe_fisher_p(group["p_transfer_upper"].to_numpy(dtype=float))),
            }
        )
    transfer_domain_df = pd.DataFrame(transfer_domain_rows)

    return {
        "rows_tested": int(by_split_df.shape[0]),
        "mean_delta_auc": float(by_split_df["delta_auc_path_minus_distance"].mean())
        if not by_split_df.empty
        else float("nan"),
        "mean_transfer_f1_lift": float(transfer_df["f1_utility_lift"].mean())
        if not transfer_df.empty
        else float("nan"),
        "domain_transfer_fisher_sig": int((transfer_domain_df["combined_fisher_p_transfer"] < 0.05).sum())
        if not transfer_domain_df.empty
        else 0,
        "artifact_paths": {
            "by_domain_layer_split": str(by_split_path),
            "utility_transfer_summary": str(transfer_path),
            "null_summary": str(null_path),
        },
    }


def run_h57_geodesic_anisotropy() -> dict[str, object]:
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

                top_genes = set(select_top_genes(split_edges, gene_cap=H57_GENE_CAP))
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
                if edge_gene_indices.size < 130:
                    continue

                gene_to_local = {int(g): int(i) for i, g in enumerate(edge_gene_indices)}
                source_local = split_edges["source_idx"].map(gene_to_local).to_numpy(dtype=int)
                target_local = split_edges["target_idx"].map(gene_to_local).to_numpy(dtype=int)
                labels = split_edges["label"].to_numpy(dtype=int)

                node_deg = np.bincount(
                    np.concatenate([source_local, target_local]),
                    minlength=edge_gene_indices.size,
                )
                node_deg_bins = degree_bins(node_deg, max_bins=5)
                edge_deg = node_deg[source_local] + node_deg[target_local]
                edge_deg_bins = degree_bins(edge_deg, max_bins=5)

                for layer in H57_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = reduce_points(
                        points,
                        n_components=20,
                        random_state=24_570 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer,
                    )

                    k = max(2, min(H57_NEIGHBORS, points_pca.shape[0] - 1))
                    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
                    nbrs.fit(points_pca)
                    neighbor_dists_full, neighbor_idx_full = nbrs.kneighbors(points_pca)
                    neighbor_idx = neighbor_idx_full[:, 1:]
                    neighbor_dists = neighbor_dists_full[:, 1:]

                    anisotropy = compute_local_anisotropy_tail(
                        points=points_pca,
                        neighbor_idx=neighbor_idx,
                        neighbor_dists=neighbor_dists,
                    )

                    edge_aniso = 0.5 * (anisotropy[source_local] + anisotropy[target_local])
                    edge_dist = np.linalg.norm(points_pca[source_local] - points_pca[target_local], axis=1)
                    baseline = -edge_dist

                    auc_aniso = safe_auc(labels, edge_aniso)
                    auc_base = safe_auc(labels, baseline)
                    delta_auc = (
                        float(auc_aniso - auc_base)
                        if np.isfinite(auc_aniso) and np.isfinite(auc_base)
                        else float("nan")
                    )

                    rng = np.random.default_rng(
                        24_571 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer
                    )
                    null_swap = np.empty(H57_NULL_PERM, dtype=float)
                    null_neigh_perm = np.empty(H57_NULL_PERM, dtype=float)
                    null_label = np.empty(H57_NULL_PERM, dtype=float)

                    # Distance-matched bins for endpoint swaps.
                    dist_bins = degree_bins(edge_dist, max_bins=6)

                    for perm_idx in range(H57_NULL_PERM):
                        swapped_target = target_local.copy()
                        for b in np.unique(dist_bins):
                            idx = np.where(dist_bins == b)[0]
                            if idx.size > 1:
                                swapped_target[idx] = rng.permutation(swapped_target[idx])
                        swap_scores = 0.5 * (anisotropy[source_local] + anisotropy[swapped_target])
                        auc_swap = safe_auc(labels, swap_scores)
                        delta_swap = (
                            float(auc_swap - auc_base) if np.isfinite(auc_swap) and np.isfinite(auc_base) else float("nan")
                        )
                        null_swap[perm_idx] = delta_swap
                        null_rows.append(
                            {
                                "null_kind": "distance_matched_endpoint_swap",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_swap),
                            }
                        )

                        aniso_perm = anisotropy.copy()
                        for b in np.unique(node_deg_bins):
                            idx = np.where(node_deg_bins == b)[0]
                            if idx.size > 1:
                                aniso_perm[idx] = rng.permutation(aniso_perm[idx])
                        neigh_scores = 0.5 * (aniso_perm[source_local] + aniso_perm[target_local])
                        auc_neigh = safe_auc(labels, neigh_scores)
                        delta_neigh = (
                            float(auc_neigh - auc_base)
                            if np.isfinite(auc_neigh) and np.isfinite(auc_base)
                            else float("nan")
                        )
                        null_neigh_perm[perm_idx] = delta_neigh
                        null_rows.append(
                            {
                                "null_kind": "neighborhood_permutation",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_neigh),
                            }
                        )

                        labels_perm = stratified_shuffle(labels, edge_deg_bins, rng=rng).astype(int)
                        auc_aniso_lp = safe_auc(labels_perm, edge_aniso)
                        auc_base_lp = safe_auc(labels_perm, baseline)
                        delta_lp = (
                            float(auc_aniso_lp - auc_base_lp)
                            if np.isfinite(auc_aniso_lp) and np.isfinite(auc_base_lp)
                            else float("nan")
                        )
                        null_label[perm_idx] = delta_lp
                        null_rows.append(
                            {
                                "null_kind": "label_permutation",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_lp),
                            }
                        )

                    p_swap = empirical_upper_tail_p(delta_auc, null_swap)
                    p_neigh = empirical_upper_tail_p(delta_auc, null_neigh_perm)
                    p_label = empirical_upper_tail_p(delta_auc, null_label)
                    p_best = np.nanmin(np.array([p_swap, p_neigh, p_label], dtype=float))

                    rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "n_edges_eval": int(labels.size),
                            "n_positive_eval": int(labels.sum()),
                            "n_nodes_graph": int(edge_gene_indices.size),
                            "auc_anisotropy_tail": float(auc_aniso),
                            "auc_geodesic_distance_baseline": float(auc_base),
                            "delta_auc_anisotropy_minus_baseline": float(delta_auc),
                            "null_swap_mean_delta": float(np.nanmean(null_swap)),
                            "null_neighborhood_mean_delta": float(np.nanmean(null_neigh_perm)),
                            "null_label_mean_delta": float(np.nanmean(null_label)),
                            "p_swap_upper": float(p_swap),
                            "p_neighborhood_upper": float(p_neigh),
                            "p_label_upper": float(p_label),
                            "p_best_upper": float(p_best),
                        }
                    )

    by_row_df = pd.DataFrame(rows).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h57_geodesic_anisotropy_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"]
    )
    null_path = ITER_DIR / "h57_geodesic_anisotropy_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
        summary_rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(group.shape[0]),
                "mean_auc_anisotropy_tail": float(group["auc_anisotropy_tail"].mean()),
                "mean_auc_geodesic_distance_baseline": float(group["auc_geodesic_distance_baseline"].mean()),
                "mean_delta_auc_anisotropy_minus_baseline": float(
                    group["delta_auc_anisotropy_minus_baseline"].mean()
                ),
                "fraction_delta_positive": float((group["delta_auc_anisotropy_minus_baseline"] > 0.0).mean()),
                "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                "combined_fisher_p_best": float(safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["domain", "split_regime"])
    summary_path = ITER_DIR / "h57_geodesic_anisotropy_domain_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_anisotropy_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "fraction_delta_positive": float((by_row_df["delta_auc_anisotropy_minus_baseline"] > 0.0).mean())
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

    h55_summary = run_h55_directed_signed_highperm(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h56_summary = run_h56_path_homology_v2(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h57_summary = run_h57_geodesic_anisotropy()

    summary = {
        "iteration": "iter_0024",
        "h55": h55_summary,
        "h56": h56_summary,
        "h57": h57_summary,
        "environment": {
            "python_env": "subproject40-topology",
            "string_cache_used": str(STRING_CACHE_PATH),
        },
    }
    summary_path = ITER_DIR / "iter0024_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
