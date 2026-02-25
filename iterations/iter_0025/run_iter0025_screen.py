from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


ITER_DIR = Path("iterations/iter_0025")
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

# H58: biologically weighted directed/signed rescue.
H58_LAYERS = [7, 11]
H58_GENE_CAP = 150
H58_KNN = 10
H58_NULL_PERM = 32

# H59: cross-model topology-signature transfer pilot.
H59_LAYERS = [7, 11]
H59_GENE_CAP = 220
H59_KNN = 10
H59_NULL_PERM = 24

# H60: ID-jump broad screen.
H60_LAYERS = [0, 3, 7, 11]
H60_GENE_CAP = 170
H60_NEIGHBORS = 14
H60_NULL_PERM = 24


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    if np.unique(scores).size < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def safe_fisher_p(pvals: np.ndarray) -> float:
    values = np.asarray(pvals, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    values = np.clip(values, 1e-12, 1.0)
    _, pvalue = combine_pvalues(values, method="fisher")
    return float(pvalue)


def empirical_upper_tail_p(observed: float, null_values: np.ndarray) -> float:
    if not np.isfinite(observed):
        return float("nan")
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float((1 + np.sum(values >= observed)) / (values.size + 1))


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

    parent = np.arange(n_nodes, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for u, v in edges:
        union(int(u), int(v))

    roots = {find(i) for i in range(n_nodes)}
    n_components = len(roots)
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


def zscore(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd < 1e-8:
        return np.zeros_like(arr, dtype=float)
    return (arr - mu) / sd


def row_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def orthogonal_procrustes_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = x.T @ y
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    return u @ vt


def random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def edge_cosine_scores(vectors: np.ndarray, src_idx: np.ndarray, tgt_idx: np.ndarray) -> np.ndarray:
    src = vectors[src_idx]
    tgt = vectors[tgt_idx]
    num = np.sum(src * tgt, axis=1)
    den = np.clip(np.linalg.norm(src, axis=1) * np.linalg.norm(tgt, axis=1), 1e-8, None)
    return (num / den).astype(float)


def cross_model_edge_scores(
    x_model: np.ndarray,
    y_model: np.ndarray,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
) -> np.ndarray:
    # Symmetric directed compatibility across mapped-vs-target model signatures.
    s_to_t = np.sum(x_model[src_idx] * y_model[tgt_idx], axis=1)
    t_to_s = np.sum(x_model[tgt_idx] * y_model[src_idx], axis=1)
    return (0.5 * (s_to_t + t_to_s)).astype(float)


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
    df = pd.read_csv(path, sep="\t")
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


def adjacency_neighbors(n_nodes: int, edges: np.ndarray) -> list[set[int]]:
    neighbors = [set() for _ in range(n_nodes)]
    for u, v in edges:
        iu = int(u)
        iv = int(v)
        if iu == iv:
            continue
        neighbors[iu].add(iv)
        neighbors[iv].add(iu)
    return neighbors


def local_clustering(neighbors: list[set[int]]) -> np.ndarray:
    n = len(neighbors)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        nbrs = list(neighbors[i])
        k = len(nbrs)
        if k < 2:
            out[i] = 0.0
            continue
        links = 0
        for a in range(k - 1):
            na = nbrs[a]
            neigh_a = neighbors[na]
            for b in range(a + 1, k):
                nb = nbrs[b]
                if nb in neigh_a:
                    links += 1
        out[i] = float(2.0 * links / (k * (k - 1)))
    return out


def two_hop_reach(neighbors: list[set[int]]) -> np.ndarray:
    n = len(neighbors)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        reach: set[int] = set()
        for j in neighbors[i]:
            reach.update(neighbors[j])
        reach.discard(i)
        out[i] = float(len(reach) / max(1, n - 1))
    return out


def local_id_two_nn(neighbor_dists: np.ndarray) -> np.ndarray:
    d = np.asarray(neighbor_dists, dtype=float)
    if d.shape[1] < 2:
        return np.ones(d.shape[0], dtype=float)
    r1 = np.clip(d[:, 0], 1e-8, None)
    r2 = np.clip(d[:, 1], 1e-8, None)
    ratio = np.clip(r2 / r1, 1.0 + 1e-8, None)
    id_est = 1.0 / np.log(ratio)
    return np.clip(id_est, 0.1, 200.0)


def local_id_mle(neighbor_dists: np.ndarray) -> np.ndarray:
    d = np.asarray(neighbor_dists, dtype=float)
    k = d.shape[1]
    if k < 3:
        return local_id_two_nn(d)
    rk = np.clip(d[:, -1], 1e-8, None)
    logs = np.log(np.clip(rk[:, None] / np.clip(d[:, :-1], 1e-8, None), 1.0 + 1e-8, None))
    denom = np.sum(logs, axis=1)
    id_est = (k - 1) / np.clip(denom, 1e-8, None)
    return np.clip(id_est, 0.1, 200.0)


def fit_signatures_scgpt(
    layer_points: np.ndarray,
    symbols: list[str],
    random_state: int,
    n_neighbors: int,
) -> pd.DataFrame:
    pts = reduce_points(layer_points, n_components=20, random_state=random_state)
    knn_edges, knn_dists = build_knn_edge_array(points=pts, n_neighbors=n_neighbors)
    neighbors = adjacency_neighbors(pts.shape[0], knn_edges)

    degree = np.array([len(n) for n in neighbors], dtype=float) / max(1, pts.shape[0] - 1)
    clust = local_clustering(neighbors)
    reach2 = two_hop_reach(neighbors)

    # Mean distance to graph neighbors.
    mean_nbr_dist = np.zeros(pts.shape[0], dtype=float)
    for i in range(pts.shape[0]):
        neigh = sorted(neighbors[i])
        if neigh:
            mean_nbr_dist[i] = float(np.mean(np.linalg.norm(pts[neigh] - pts[i], axis=1)))

    # Local ID estimates from metric neighborhoods.
    k_id = max(4, min(14, pts.shape[0] - 1))
    nbrs = NearestNeighbors(n_neighbors=k_id + 1, metric="euclidean")
    nbrs.fit(pts)
    d_full, _ = nbrs.kneighbors(pts)
    d_local = d_full[:, 1:]
    id_two = local_id_two_nn(d_local)
    id_mle = local_id_mle(d_local)

    sig = np.column_stack(
        [
            degree,
            zscore(-mean_nbr_dist),
            clust,
            reach2,
            zscore(id_two),
            zscore(id_mle),
        ]
    )
    cols = [
        "deg_norm",
        "neg_mean_nbr_dist_z",
        "clustering",
        "two_hop_reach",
        "id_two_nn_z",
        "id_mle_z",
    ]
    df = pd.DataFrame(sig, columns=cols)
    df["symbol"] = symbols
    return df.set_index("symbol")


def fit_signatures_geneformer(
    gf_df: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:
    index = {sym: i for i, sym in enumerate(symbols)}
    n = len(symbols)
    out_neighbors = [set() for _ in range(n)]
    in_neighbors = [set() for _ in range(n)]

    pos = gf_df.loc[gf_df["label"].astype(int) == 1, ["source", "target"]].copy()
    for row in pos.itertuples(index=False):
        s = str(row.source).upper()
        t = str(row.target).upper()
        if s not in index or t not in index:
            continue
        i = index[s]
        j = index[t]
        if i == j:
            continue
        out_neighbors[i].add(j)
        in_neighbors[j].add(i)

    und_neighbors = [out_neighbors[i] | in_neighbors[i] for i in range(n)]
    clust = local_clustering(und_neighbors)

    out_deg = np.array([len(x) for x in out_neighbors], dtype=float) / max(1, n - 1)
    in_deg = np.array([len(x) for x in in_neighbors], dtype=float) / max(1, n - 1)
    und_deg = np.array([len(x) for x in und_neighbors], dtype=float) / max(1, n - 1)

    reciprocity = np.zeros(n, dtype=float)
    twohop_out = np.zeros(n, dtype=float)
    for i in range(n):
        union = out_neighbors[i] | in_neighbors[i]
        inter = out_neighbors[i] & in_neighbors[i]
        reciprocity[i] = float(len(inter) / len(union)) if union else 0.0

        reach: set[int] = set()
        for j in out_neighbors[i]:
            reach.update(out_neighbors[j])
        reach.discard(i)
        twohop_out[i] = float(len(reach) / max(1, n - 1))

    sig = np.column_stack(
        [
            out_deg,
            in_deg,
            und_deg,
            reciprocity,
            clust,
            twohop_out,
        ]
    )
    cols = [
        "out_deg_norm",
        "in_deg_norm",
        "und_deg_norm",
        "reciprocity",
        "clustering",
        "twohop_out",
    ]
    df = pd.DataFrame(sig, columns=cols)
    df["symbol"] = symbols
    return df.set_index("symbol")


def zscore_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.clip(std, 1e-8, None)
    return mean, std


def zscore_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def run_h58_weighted_directed_signed(
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

                top_genes = set(select_top_genes(split_edges, gene_cap=H58_GENE_CAP))
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

                symbol_map = build_symbol_map(split_edges)
                ordered_symbols = [symbol_map[int(g)] for g in edge_gene_indices]
                support_undirected, support_directed = build_support_matrices(
                    symbols_upper=ordered_symbols,
                    dorothea_map=dorothea_map,
                    omnipath_pairs=omnipath_pairs,
                    gene2go_upper=gene2go_upper,
                    string_map=string_map,
                )

                for layer in H58_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = reduce_points(
                        points,
                        n_components=20,
                        random_state=25_580 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer,
                    )
                    knn_edges, knn_dists = build_knn_edge_array(points=points_pca, n_neighbors=H58_KNN)
                    if knn_edges.shape[0] < 100:
                        continue

                    margins = np.array(
                        [support_directed[i, j] - support_directed[j, i] for i, j in knn_edges],
                        dtype=float,
                    )
                    edge_weights = np.array(
                        [1.0 + 1.8 * support_undirected[i, j] for i, j in knn_edges],
                        dtype=float,
                    )
                    weighted_margins = margins * edge_weights

                    directed_unweighted = directed_signed_score_matrix(
                        n_nodes=edge_gene_indices.size,
                        edges=knn_edges,
                        dists=knn_dists,
                        margins=margins,
                        d_quantiles=[0.40, 0.55, 0.70],
                        m_quantiles=[0.45, 0.65, 0.85],
                    )
                    directed_weighted = directed_signed_score_matrix(
                        n_nodes=edge_gene_indices.size,
                        edges=knn_edges,
                        dists=knn_dists,
                        margins=weighted_margins,
                        d_quantiles=[0.40, 0.55, 0.70],
                        m_quantiles=[0.45, 0.65, 0.85],
                    )
                    dist_matrix = distance_cycle_score_matrix(
                        n_nodes=edge_gene_indices.size,
                        edges=knn_edges,
                        dists=knn_dists,
                        d_quantiles=[0.40, 0.55, 0.70],
                    )

                    eval_unweighted = directed_unweighted[source_local, target_local]
                    eval_weighted = directed_weighted[source_local, target_local]
                    eval_dist = dist_matrix[source_local, target_local]

                    auc_unweighted = safe_auc(labels, eval_unweighted)
                    auc_weighted = safe_auc(labels, eval_weighted)
                    auc_dist = safe_auc(labels, eval_dist)

                    delta_weighted_vs_dist = (
                        float(auc_weighted - auc_dist)
                        if np.isfinite(auc_weighted) and np.isfinite(auc_dist)
                        else float("nan")
                    )
                    delta_weighted_vs_unweighted = (
                        float(auc_weighted - auc_unweighted)
                        if np.isfinite(auc_weighted) and np.isfinite(auc_unweighted)
                        else float("nan")
                    )

                    node_deg_knn = np.bincount(
                        np.concatenate([knn_edges[:, 0], knn_edges[:, 1]]),
                        minlength=edge_gene_indices.size,
                    )
                    edge_deg_knn = node_deg_knn[knn_edges[:, 0]] + node_deg_knn[knn_edges[:, 1]]
                    edge_bins = degree_bins(edge_deg_knn, max_bins=6)
                    eval_edge_deg = node_deg_knn[source_local] + node_deg_knn[target_local]
                    eval_edge_bins = degree_bins(eval_edge_deg, max_bins=6)

                    rng = np.random.default_rng(
                        25_581 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer
                    )
                    null_weight_shuffle = np.empty(H58_NULL_PERM, dtype=float)
                    null_placebo = np.empty(H58_NULL_PERM, dtype=float)
                    null_label = np.empty(H58_NULL_PERM, dtype=float)

                    for perm_idx in range(H58_NULL_PERM):
                        shuffled_w = stratified_shuffle(edge_weights, edge_bins, rng=rng)
                        weighted_perm = margins * shuffled_w
                        directed_perm = directed_signed_score_matrix(
                            n_nodes=edge_gene_indices.size,
                            edges=knn_edges,
                            dists=knn_dists,
                            margins=weighted_perm,
                            d_quantiles=[0.40, 0.55, 0.70],
                            m_quantiles=[0.45, 0.65, 0.85],
                        )
                        eval_perm = directed_perm[source_local, target_local]
                        auc_perm = safe_auc(labels, eval_perm)
                        delta_perm = (
                            float(auc_perm - auc_dist) if np.isfinite(auc_perm) and np.isfinite(auc_dist) else float("nan")
                        )
                        null_weight_shuffle[perm_idx] = delta_perm
                        null_rows.append(
                            {
                                "null_kind": "bio_weight_shuffle_within_edge_degree_bins",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_perm),
                            }
                        )

                        random_bins = rng.integers(0, max(2, edge_bins.max() + 1), size=edge_bins.size)
                        placebo_w = stratified_shuffle(edge_weights, random_bins, rng=rng)
                        weighted_placebo = margins * placebo_w
                        directed_placebo = directed_signed_score_matrix(
                            n_nodes=edge_gene_indices.size,
                            edges=knn_edges,
                            dists=knn_dists,
                            margins=weighted_placebo,
                            d_quantiles=[0.40, 0.55, 0.70],
                            m_quantiles=[0.45, 0.65, 0.85],
                        )
                        eval_placebo = directed_placebo[source_local, target_local]
                        auc_placebo = safe_auc(labels, eval_placebo)
                        delta_placebo = (
                            float(auc_placebo - auc_dist)
                            if np.isfinite(auc_placebo) and np.isfinite(auc_dist)
                            else float("nan")
                        )
                        null_placebo[perm_idx] = delta_placebo
                        null_rows.append(
                            {
                                "null_kind": "random_strata_weight_placebo",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_placebo),
                            }
                        )

                        labels_perm = stratified_shuffle(labels, eval_edge_bins, rng=rng).astype(int)
                        auc_w_lp = safe_auc(labels_perm, eval_weighted)
                        auc_d_lp = safe_auc(labels_perm, eval_dist)
                        delta_lp = (
                            float(auc_w_lp - auc_d_lp)
                            if np.isfinite(auc_w_lp) and np.isfinite(auc_d_lp)
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

                    p_weight = empirical_upper_tail_p(delta_weighted_vs_dist, null_weight_shuffle)
                    p_placebo = empirical_upper_tail_p(delta_weighted_vs_dist, null_placebo)
                    p_label = empirical_upper_tail_p(delta_weighted_vs_dist, null_label)
                    p_best = np.nanmin(np.array([p_weight, p_placebo, p_label], dtype=float))

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
                            "auc_weighted_directed_signed": float(auc_weighted),
                            "auc_unweighted_directed_signed": float(auc_unweighted),
                            "auc_distance_only": float(auc_dist),
                            "delta_auc_weighted_minus_distance": float(delta_weighted_vs_dist),
                            "delta_auc_weighted_minus_unweighted": float(delta_weighted_vs_unweighted),
                            "mean_edge_weight": float(np.mean(edge_weights)),
                            "mean_abs_margin": float(np.mean(np.abs(margins))),
                            "p_weight_shuffle_upper": float(p_weight),
                            "p_placebo_upper": float(p_placebo),
                            "p_label_upper": float(p_label),
                            "p_best_upper": float(p_best),
                        }
                    )

    by_row_df = pd.DataFrame(rows).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h58_weighted_directed_signed_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"]
    )
    null_path = ITER_DIR / "h58_weighted_directed_signed_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    domain_rows: list[dict[str, object]] = []
    for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
        domain_rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(group.shape[0]),
                "mean_auc_weighted_directed_signed": float(group["auc_weighted_directed_signed"].mean()),
                "mean_auc_unweighted_directed_signed": float(group["auc_unweighted_directed_signed"].mean()),
                "mean_auc_distance_only": float(group["auc_distance_only"].mean()),
                "mean_delta_auc_weighted_minus_distance": float(group["delta_auc_weighted_minus_distance"].mean()),
                "mean_delta_auc_weighted_minus_unweighted": float(group["delta_auc_weighted_minus_unweighted"].mean()),
                "fraction_delta_positive": float((group["delta_auc_weighted_minus_distance"] > 0.0).mean()),
                "fraction_weight_gain_positive": float((group["delta_auc_weighted_minus_unweighted"] > 0.0).mean()),
                "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                "combined_fisher_p_best": float(safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
            }
        )
    domain_df = pd.DataFrame(domain_rows).sort_values(["domain", "split_regime"])
    domain_path = ITER_DIR / "h58_weighted_directed_signed_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    failure_rows: list[dict[str, object]] = []
    for domain in ["lung", "external_lung"]:
        subset = by_row_df.loc[
            (by_row_df["domain"] == domain) & (by_row_df["split_regime"] == "source_disjoint")
        ]
        if subset.empty:
            continue
        failure_rows.append(
            {
                "domain": domain,
                "split_regime": "source_disjoint",
                "n_rows": int(subset.shape[0]),
                "mean_delta_auc_weighted_minus_distance": float(subset["delta_auc_weighted_minus_distance"].mean()),
                "mean_delta_auc_weighted_minus_unweighted": float(subset["delta_auc_weighted_minus_unweighted"].mean()),
                "fraction_weight_gain_positive": float((subset["delta_auc_weighted_minus_unweighted"] > 0.0).mean()),
                "fraction_delta_positive": float((subset["delta_auc_weighted_minus_distance"] > 0.0).mean()),
            }
        )
    failure_df = pd.DataFrame(failure_rows).sort_values(["domain", "split_regime"])
    failure_path = ITER_DIR / "h58_weighted_directed_signed_failure_slice_summary.csv"
    failure_df.to_csv(failure_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_weighted_minus_distance"].mean())
        if not by_row_df.empty
        else float("nan"),
        "mean_weight_gain_delta": float(by_row_df["delta_auc_weighted_minus_unweighted"].mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_split_fisher_sig": int((domain_df["combined_fisher_p_best"] < 0.05).sum())
        if not domain_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_layer_split": str(by_row_path),
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
            "failure_slice_summary": str(failure_path),
        },
    }


def run_h59_cross_model_topology_signature_transfer() -> dict[str, object]:
    by_row: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    sc_edges_seed42: dict[str, pd.DataFrame] = {}
    sc_layers_seed42: dict[str, np.ndarray] = {}
    gf_edges: dict[str, pd.DataFrame] = {}

    for domain in ["immune", "lung", "external_lung"]:
        sc_run = SCGPT_RUNS_BY_DOMAIN[domain]["seed42_main"]
        sc_edges_seed42[domain] = pd.read_csv(sc_run / "cycle1_edge_dataset.tsv", sep="\t")
        sc_layers_seed42[domain] = np.load(sc_run / "layer_gene_embeddings.npy", mmap_mode="r")
        gf_edges[domain] = pd.read_csv(GENEFORMER_EDGE_BY_DOMAIN[domain], sep="\t")

    # Build per-domain selected symbols and model signatures.
    selected_symbols_by_domain: dict[str, list[str]] = {}
    sc_sig_by_domain_layer: dict[tuple[str, int], pd.DataFrame] = {}
    gf_sig_by_domain: dict[str, pd.DataFrame] = {}

    for domain_index, domain in enumerate(["immune", "lung", "external_lung"]):
        sc_df = sc_edges_seed42[domain].copy()
        top_genes = set(select_top_genes(sc_df, gene_cap=H59_GENE_CAP))
        sc_df = sc_df.loc[sc_df["source_idx"].isin(top_genes) & sc_df["target_idx"].isin(top_genes)].copy()
        if sc_df.empty:
            continue

        gene_indices = np.unique(
            np.concatenate(
                [
                    sc_df["source_idx"].to_numpy(dtype=int),
                    sc_df["target_idx"].to_numpy(dtype=int),
                ]
            )
        )
        symbol_map = build_symbol_map(sc_df)
        symbols = [symbol_map[int(g)] for g in gene_indices]
        selected_symbols_by_domain[domain] = symbols

        gf_sig_by_domain[domain] = fit_signatures_geneformer(gf_edges[domain], symbols)

        for layer in H59_LAYERS:
            points = sc_layers_seed42[domain][layer, gene_indices, :]
            sc_sig_by_domain_layer[(domain, layer)] = fit_signatures_scgpt(
                layer_points=points,
                symbols=symbols,
                random_state=25_590 + domain_index * 10 + layer,
                n_neighbors=H59_KNN,
            )

    domains = ["immune", "lung", "external_lung"]
    for target_domain in domains:
        source_domains = [d for d in domains if d != target_domain]

        split_masks = build_split_masks(sc_edges_seed42[target_domain])

        for layer in H59_LAYERS:
            x_train_list: list[np.ndarray] = []
            y_train_list: list[np.ndarray] = []
            train_pair_count = 0

            for src_domain in source_domains:
                if src_domain not in gf_sig_by_domain:
                    continue
                x_df = gf_sig_by_domain[src_domain]
                y_df = sc_sig_by_domain_layer.get((src_domain, layer))
                if y_df is None:
                    continue

                shared = sorted(set(x_df.index) & set(y_df.index))
                if len(shared) < 60:
                    continue
                x_train_list.append(x_df.loc[shared].to_numpy(dtype=float))
                y_train_list.append(y_df.loc[shared].to_numpy(dtype=float))
                train_pair_count += len(shared)

            if train_pair_count < 120:
                continue

            x_train = np.vstack(x_train_list)
            y_train = np.vstack(y_train_list)
            x_mu, x_sd = zscore_fit(x_train)
            y_mu, y_sd = zscore_fit(y_train)

            x_train_z = row_normalize(zscore_apply(x_train, x_mu, x_sd))
            y_train_z = row_normalize(zscore_apply(y_train, y_mu, y_sd))
            w = orthogonal_procrustes_map(x_train_z, y_train_z)

            x_tgt_df = gf_sig_by_domain.get(target_domain)
            y_tgt_df = sc_sig_by_domain_layer.get((target_domain, layer))
            if x_tgt_df is None or y_tgt_df is None:
                continue

            shared_tgt = sorted(set(x_tgt_df.index) & set(y_tgt_df.index))
            if len(shared_tgt) < 80:
                continue

            x_tgt = x_tgt_df.loc[shared_tgt].to_numpy(dtype=float)
            y_tgt = y_tgt_df.loc[shared_tgt].to_numpy(dtype=float)
            x_tgt_z = row_normalize(zscore_apply(x_tgt, x_mu, x_sd))
            y_tgt_z = row_normalize(zscore_apply(y_tgt, y_mu, y_sd))
            mapped = row_normalize(x_tgt_z @ w)
            base = row_normalize(x_tgt_z)

            align_diag_cos = float(np.mean(np.sum(mapped * y_tgt_z, axis=1)))
            symbol_to_local = {sym: i for i, sym in enumerate(shared_tgt)}

            for split_regime, split_mask in split_masks.items():
                split_df = sc_edges_seed42[target_domain].loc[split_mask].copy()
                split_df["source_u"] = split_df["source"].astype(str).str.upper()
                split_df["target_u"] = split_df["target"].astype(str).str.upper()
                keep = split_df["source_u"].isin(symbol_to_local) & split_df["target_u"].isin(symbol_to_local)
                split_df = split_df.loc[keep].copy()
                if split_df["label"].nunique() < 2 or split_df.shape[0] < 300:
                    continue

                src_idx = split_df["source_u"].map(symbol_to_local).to_numpy(dtype=int)
                tgt_idx = split_df["target_u"].map(symbol_to_local).to_numpy(dtype=int)
                labels = split_df["label"].to_numpy(dtype=int)

                transfer_scores = cross_model_edge_scores(mapped, y_tgt_z, src_idx, tgt_idx)
                baseline_scores = cross_model_edge_scores(base, y_tgt_z, src_idx, tgt_idx)

                auc_transfer = safe_auc(labels, transfer_scores)
                auc_baseline = safe_auc(labels, baseline_scores)
                delta_auc = (
                    float(auc_transfer - auc_baseline)
                    if np.isfinite(auc_transfer) and np.isfinite(auc_baseline)
                    else float("nan")
                )

                rng = np.random.default_rng(25_591 + layer * 100 + len(shared_tgt))
                null_random_map = np.empty(H59_NULL_PERM, dtype=float)
                null_signature_destroy = np.empty(H59_NULL_PERM, dtype=float)

                dim = mapped.shape[1]
                for perm_idx in range(H59_NULL_PERM):
                    rand_map = random_orthogonal(dim, rng)
                    mapped_rand = row_normalize(base @ rand_map)
                    auc_rand = safe_auc(labels, cross_model_edge_scores(mapped_rand, y_tgt_z, src_idx, tgt_idx))
                    delta_rand = (
                        float(auc_rand - auc_baseline)
                        if np.isfinite(auc_rand) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_random_map[perm_idx] = delta_rand
                    null_rows.append(
                        {
                            "null_kind": "random_map_alignment",
                            "domain": target_domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_rand),
                        }
                    )

                    perm_rows = rng.permutation(mapped.shape[0])
                    mapped_perm = mapped[perm_rows]
                    auc_perm = safe_auc(labels, cross_model_edge_scores(mapped_perm, y_tgt_z, src_idx, tgt_idx))
                    delta_perm = (
                        float(auc_perm - auc_baseline)
                        if np.isfinite(auc_perm) and np.isfinite(auc_baseline)
                        else float("nan")
                    )
                    null_signature_destroy[perm_idx] = delta_perm
                    null_rows.append(
                        {
                            "null_kind": "signature_destroy_permutation",
                            "domain": target_domain,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_perm),
                        }
                    )

                p_random_map = empirical_upper_tail_p(delta_auc, null_random_map)
                p_destroy = empirical_upper_tail_p(delta_auc, null_signature_destroy)
                p_best = np.nanmin(np.array([p_random_map, p_destroy], dtype=float))

                by_row.append(
                    {
                        "domain": target_domain,
                        "split_regime": split_regime,
                        "layer": int(layer),
                        "train_domains": "+".join(sorted(source_domains)),
                        "n_train_pairs": int(train_pair_count),
                        "n_shared_target_symbols": int(len(shared_tgt)),
                        "n_edges_eval": int(labels.size),
                        "n_positive_eval": int(labels.sum()),
                        "auc_transfer": float(auc_transfer),
                        "auc_baseline": float(auc_baseline),
                        "delta_auc_transfer_minus_baseline": float(delta_auc),
                        "alignment_diag_cosine": align_diag_cos,
                        "p_random_map_upper": float(p_random_map),
                        "p_signature_destroy_upper": float(p_destroy),
                        "p_best_upper": float(p_best),
                    }
                )

    by_row_df = pd.DataFrame(by_row).sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h59_cross_model_topology_signature_transfer_by_domain_layer.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(["null_kind", "domain", "split_regime", "layer", "perm_idx"])
    null_path = ITER_DIR / "h59_cross_model_topology_signature_transfer_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    for domain, group in by_row_df.groupby("domain", sort=True):
        summary_rows.append(
            {
                "domain": domain,
                "n_rows": int(group.shape[0]),
                "mean_auc_transfer": float(group["auc_transfer"].mean()),
                "mean_auc_baseline": float(group["auc_baseline"].mean()),
                "mean_delta_auc_transfer_minus_baseline": float(group["delta_auc_transfer_minus_baseline"].mean()),
                "fraction_delta_positive": float((group["delta_auc_transfer_minus_baseline"] > 0.0).mean()),
                "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                "combined_fisher_p_best": float(safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
                "mean_alignment_diag_cosine": float(group["alignment_diag_cosine"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("domain")
    summary_path = ITER_DIR / "h59_cross_model_topology_signature_transfer_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_transfer_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_fisher_sig": int((summary_df["combined_fisher_p_best"] < 0.05).sum())
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_domain_layer": str(by_row_path),
            "summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }


def run_h60_id_jump_screen() -> dict[str, object]:
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

                top_genes = set(select_top_genes(split_edges, gene_cap=H60_GENE_CAP))
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

                node_deg_eval = np.bincount(
                    np.concatenate([source_local, target_local]),
                    minlength=edge_gene_indices.size,
                )
                node_deg_bins = degree_bins(node_deg_eval, max_bins=6)
                edge_deg_eval = node_deg_eval[source_local] + node_deg_eval[target_local]
                edge_deg_bins = degree_bins(edge_deg_eval, max_bins=6)

                for layer in H60_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = reduce_points(
                        points,
                        n_components=20,
                        random_state=25_600 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer,
                    )

                    k = max(3, min(H60_NEIGHBORS, points_pca.shape[0] - 1))
                    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
                    nbrs.fit(points_pca)
                    d_full, idx_full = nbrs.kneighbors(points_pca)
                    d_local = d_full[:, 1:]
                    idx_local = idx_full[:, 1:]

                    id_two = local_id_two_nn(d_local)
                    id_mle = local_id_mle(d_local)
                    id_combined = 0.5 * (id_two + id_mle)

                    edge_dist = np.linalg.norm(points_pca[source_local] - points_pca[target_local], axis=1)
                    base_score = -edge_dist
                    id_jump = -np.abs(id_combined[source_local] - id_combined[target_local])
                    id_mean = -0.5 * (id_combined[source_local] + id_combined[target_local])
                    combined_score = zscore(base_score) + 0.75 * zscore(id_jump) + 0.25 * zscore(id_mean)

                    auc_base = safe_auc(labels, base_score)
                    auc_combined = safe_auc(labels, combined_score)
                    delta_auc = (
                        float(auc_combined - auc_base)
                        if np.isfinite(auc_combined) and np.isfinite(auc_base)
                        else float("nan")
                    )

                    rng = np.random.default_rng(
                        25_601 + domain_index * 1000 + seed_index * 100 + split_index * 20 + layer
                    )
                    null_endpoint_swap = np.empty(H60_NULL_PERM, dtype=float)
                    null_estimator = np.empty(H60_NULL_PERM, dtype=float)
                    null_label = np.empty(H60_NULL_PERM, dtype=float)

                    dist_bins = degree_bins(edge_dist, max_bins=6)
                    for perm_idx in range(H60_NULL_PERM):
                        swapped_target = target_local.copy()
                        for b in np.unique(dist_bins):
                            idx = np.where(dist_bins == b)[0]
                            if idx.size > 1:
                                swapped_target[idx] = rng.permutation(swapped_target[idx])
                        id_jump_swap = -np.abs(id_combined[source_local] - id_combined[swapped_target])
                        id_mean_swap = -0.5 * (id_combined[source_local] + id_combined[swapped_target])
                        combined_swap = zscore(base_score) + 0.75 * zscore(id_jump_swap) + 0.25 * zscore(id_mean_swap)
                        auc_swap = safe_auc(labels, combined_swap)
                        delta_swap = (
                            float(auc_swap - auc_base) if np.isfinite(auc_swap) and np.isfinite(auc_base) else float("nan")
                        )
                        null_endpoint_swap[perm_idx] = delta_swap
                        null_rows.append(
                            {
                                "null_kind": "endpoint_swap_within_distance_bins",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_swap),
                            }
                        )

                        id_perm = id_combined.copy()
                        for b in np.unique(node_deg_bins):
                            node_idx = np.where(node_deg_bins == b)[0]
                            if node_idx.size > 1:
                                id_perm[node_idx] = rng.permutation(id_perm[node_idx])
                        id_jump_perm = -np.abs(id_perm[source_local] - id_perm[target_local])
                        id_mean_perm = -0.5 * (id_perm[source_local] + id_perm[target_local])
                        combined_perm = zscore(base_score) + 0.75 * zscore(id_jump_perm) + 0.25 * zscore(id_mean_perm)
                        auc_est = safe_auc(labels, combined_perm)
                        delta_est = (
                            float(auc_est - auc_base) if np.isfinite(auc_est) and np.isfinite(auc_base) else float("nan")
                        )
                        null_estimator[perm_idx] = delta_est
                        null_rows.append(
                            {
                                "null_kind": "estimator_randomization_placebo",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_est),
                            }
                        )

                        labels_perm = stratified_shuffle(labels, edge_deg_bins, rng=rng).astype(int)
                        auc_comb_lp = safe_auc(labels_perm, combined_score)
                        auc_base_lp = safe_auc(labels_perm, base_score)
                        delta_lp = (
                            float(auc_comb_lp - auc_base_lp)
                            if np.isfinite(auc_comb_lp) and np.isfinite(auc_base_lp)
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

                    p_swap = empirical_upper_tail_p(delta_auc, null_endpoint_swap)
                    p_est = empirical_upper_tail_p(delta_auc, null_estimator)
                    p_label = empirical_upper_tail_p(delta_auc, null_label)
                    p_best = np.nanmin(np.array([p_swap, p_est, p_label], dtype=float))

                    mean_id = float(np.mean(id_combined))
                    std_id = float(np.std(id_combined))
                    mean_jump = float(np.mean(np.abs(id_combined[source_local] - id_combined[target_local])))

                    rows.append(
                        {
                            "domain": domain,
                            "seed_tag": seed_tag,
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "n_edges_eval": int(labels.size),
                            "n_positive_eval": int(labels.sum()),
                            "n_nodes_graph": int(edge_gene_indices.size),
                            "auc_combined_id_jump": float(auc_combined),
                            "auc_geodesic_baseline": float(auc_base),
                            "delta_auc_combined_minus_baseline": float(delta_auc),
                            "mean_local_id": mean_id,
                            "std_local_id": std_id,
                            "mean_edge_id_jump": mean_jump,
                            "p_swap_upper": float(p_swap),
                            "p_estimator_upper": float(p_est),
                            "p_label_upper": float(p_label),
                            "p_best_upper": float(p_best),
                        }
                    )

    by_row_df = pd.DataFrame(rows).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h60_id_jump_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"]
    )
    null_path = ITER_DIR / "h60_id_jump_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    domain_rows: list[dict[str, object]] = []
    for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
        domain_rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(group.shape[0]),
                "mean_auc_combined_id_jump": float(group["auc_combined_id_jump"].mean()),
                "mean_auc_geodesic_baseline": float(group["auc_geodesic_baseline"].mean()),
                "mean_delta_auc_combined_minus_baseline": float(group["delta_auc_combined_minus_baseline"].mean()),
                "fraction_delta_positive": float((group["delta_auc_combined_minus_baseline"] > 0.0).mean()),
                "fraction_p_best_lt_0_05": float((group["p_best_upper"] < 0.05).mean()),
                "combined_fisher_p_best": float(safe_fisher_p(group["p_best_upper"].to_numpy(dtype=float))),
            }
        )
    domain_df = pd.DataFrame(domain_rows).sort_values(["domain", "split_regime"])
    domain_path = ITER_DIR / "h60_id_jump_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    return {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_combined_minus_baseline"].mean())
        if not by_row_df.empty
        else float("nan"),
        "fraction_delta_positive": float((by_row_df["delta_auc_combined_minus_baseline"] > 0.0).mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_split_fisher_sig": int((domain_df["combined_fisher_p_best"] < 0.05).sum())
        if not domain_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_layer_split": str(by_row_path),
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
        },
    }


def main() -> None:
    required_paths = [
        DOROTHEA_PATH,
        GENE2GO_PATH,
        OMNIPATH_INTERACTIONS_PATH,
        STRING_CACHE_PATH,
    ]
    for run_map in SCGPT_RUNS_BY_DOMAIN.values():
        for run_dir in run_map.values():
            required_paths.append(run_dir / "cycle1_edge_dataset.tsv")
            required_paths.append(run_dir / "layer_gene_embeddings.npy")
    for p in GENEFORMER_EDGE_BY_DOMAIN.values():
        required_paths.append(p)

    missing = [str(p) for p in required_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    dorothea_map = load_dorothea_score_map()
    omnipath_pairs = load_omnipath_pairs()
    gene2go_upper = load_gene2go_upper()
    string_map = load_string_scores_from_cache(STRING_CACHE_PATH)

    h58_summary = run_h58_weighted_directed_signed(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h59_summary = run_h59_cross_model_topology_signature_transfer()
    h60_summary = run_h60_id_jump_screen()

    summary = {
        "iteration": "iter_0025",
        "h58": h58_summary,
        "h59": h59_summary,
        "h60": h60_summary,
        "environment": {
            "python_env": "subproject40-topology",
            "string_cache_used": str(STRING_CACHE_PATH),
        },
    }
    summary_path = ITER_DIR / "iter0025_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
