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
from transformers import AutoModel


ITER_DIR = Path("iterations/iter_0022")
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

# H49: multiseed extension of H47.
H49_LAYERS = [0, 3, 7, 11]
H49_GENE_CAP = 180
H49_KNN = 10
H49_NULL_PERM = 20
H49_LAYER_PLACEBO_PERM = 200

# H50: directed/signed topology pilot.
H50_LAYERS = [7, 11]
H50_GENE_CAP = 180
H50_KNN = 10
H50_NULL_PERM = 24

# H51: expanded anti-sparsity motif screen.
H51_LAYERS = [7, 11]
H51_TOP_K = [100, 200, 400]
H51_NULL_PERM = 36
H51_MODULE_TOP_TERMS = 40


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


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 3:
        return float("nan")
    corr = pd.Series(a[mask]).corr(pd.Series(b[mask]), method="spearman")
    return float(corr) if pd.notna(corr) else float("nan")


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


def stratified_shuffle(values: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(values)
    strata = np.asarray(strata, dtype=int)
    out = x.copy()
    for stratum in np.unique(strata):
        idx = np.where(strata == stratum)[0]
        if idx.size > 1:
            out[idx] = rng.permutation(out[idx])
    return out


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

    d_thresholds = np.quantile(dists, d_quantiles)
    for d_thr in d_thresholds:
        mask_d = dists <= float(d_thr)
        if int(mask_d.sum()) < 6:
            continue

        edges_d = edges[mask_d]
        margins_d = margins[mask_d]
        abs_margin_d = np.abs(margins_d)
        if np.all(abs_margin_d <= 0.0):
            continue

        m_thresholds = np.quantile(abs_margin_d, m_quantiles)
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
            signed_component = np.tanh(4.0 * signed)
            component = (paths2 - paths2.T).astype(float) + signed_component

            total += component
            n_steps += 1

    if n_steps == 0:
        return total
    return total / float(n_steps)


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

    # Directional priors (DoRothEA, OmniPath) are weighted most heavily;
    # GO/STRING provide soft biological support for robust fallback behavior.
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


def load_h31_utility_table() -> pd.DataFrame:
    df = pd.read_csv(H31_UTILITY_PATH)
    utility = (
        df.groupby(["domain", "seed_tag", "layer"], as_index=False)["delta_auc_diffusion_incremental"]
        .mean()
        .rename(columns={"delta_auc_diffusion_incremental": "utility_delta_auc_mean"})
    )
    return utility


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

    perm_tgt = rng.permutation(tgt)
    edges = list(zip(src.tolist(), perm_tgt.tolist()))
    return list(dict.fromkeys(edges))


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


def motif_feedback_triad(edges: list[tuple[str, str]]) -> set[tuple[str, str, str]]:
    out: dict[str, set[str]] = {}
    for src, tgt in edges:
        if src == tgt:
            continue
        out.setdefault(src, set()).add(tgt)

    motifs: set[tuple[str, str, str]] = set()
    for a, a_out in out.items():
        for b in a_out:
            for c in out.get(b, set()):
                if c == a or c == b:
                    continue
                if a in out.get(c, set()):
                    motifs.add(tuple(sorted((a, b, c))))
    return motifs


def motif_feedforward_chain(edges: list[tuple[str, str]]) -> set[tuple[str, str, str]]:
    out: dict[str, set[str]] = {}
    for src, tgt in edges:
        if src == tgt:
            continue
        out.setdefault(src, set()).add(tgt)

    motifs: set[tuple[str, str, str]] = set()
    for a, a_out in out.items():
        for b in a_out:
            for c in out.get(b, set()):
                if c == a or c == b:
                    continue
                if c in out.get(a, set()):
                    continue
                motifs.add((a, b, c))
    return motifs


def motif_multi_input(edges: list[tuple[str, str]]) -> set[tuple[str, str, str]]:
    incoming: dict[str, set[str]] = {}
    out: dict[str, set[str]] = {}
    for src, tgt in edges:
        if src == tgt:
            continue
        incoming.setdefault(tgt, set()).add(src)
        out.setdefault(src, set()).add(tgt)

    motifs: set[tuple[str, str, str]] = set()
    for tgt, srcs in incoming.items():
        ordered = sorted(srcs - {tgt})
        if len(ordered) < 2:
            continue
        for i in range(len(ordered) - 1):
            for j in range(i + 1, len(ordered)):
                a = ordered[i]
                b = ordered[j]
                if b in out.get(a, set()) or a in out.get(b, set()):
                    continue
                motifs.add((a, b, tgt))
    return motifs


def motif_panel(edges: list[tuple[str, str]]) -> dict[str, set[tuple]]:
    return {
        "ffl": motif_ffl(edges),
        "bifan": motif_bifan(edges),
        "feedback_triad": motif_feedback_triad(edges),
        "feedforward_chain": motif_feedforward_chain(edges),
        "multi_input": motif_multi_input(edges),
    }


def motif_overlap_counts(
    motifs_a: dict[str, set[tuple]],
    motifs_b: dict[str, set[tuple]],
) -> tuple[dict[str, int], int]:
    overlap_by_type: dict[str, int] = {}
    total = 0
    for key in motifs_a.keys():
        overlap = int(len(motifs_a[key] & motifs_b[key]))
        overlap_by_type[key] = overlap
        total += overlap
    return overlap_by_type, int(total)


def assign_go_modules(genes: np.ndarray, gene2go_upper: dict[str, set[str]], top_terms: int) -> dict[str, str]:
    term_counts: dict[str, int] = {}
    for gene in genes:
        for term in gene2go_upper.get(str(gene).upper(), set()):
            term_counts[term] = term_counts.get(term, 0) + 1

    ranked_terms = sorted(term_counts.items(), key=lambda x: (-x[1], x[0]))
    top = [term for term, _ in ranked_terms[:top_terms]]
    top_set = set(top)

    module_map: dict[str, str] = {}
    for gene in genes:
        gene_u = str(gene).upper()
        terms = sorted(gene2go_upper.get(gene_u, set()) & top_set)
        module_map[gene_u] = terms[0] if terms else "GO:OTHER"
    return module_map


def collapse_edges_by_module(edges: list[tuple[str, str]], module_map: dict[str, str]) -> list[tuple[str, str]]:
    collapsed = {
        (module_map.get(str(src).upper(), "GO:OTHER"), module_map.get(str(tgt).upper(), "GO:OTHER"))
        for src, tgt in edges
        if module_map.get(str(src).upper(), "GO:OTHER") != module_map.get(str(tgt).upper(), "GO:OTHER")
    }
    return sorted(collapsed)


def run_h49_bifiltration_multiseed(
    dorothea_map: dict[tuple[str, str], int],
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

            for split_index, (split_regime, split_mask) in enumerate(split_masks.items()):
                split_edges = edge_df.loc[split_mask].copy()
                if split_edges["label"].nunique() < 2:
                    continue

                top_genes = set(select_top_genes(split_edges, gene_cap=H49_GENE_CAP))
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

                support_undirected, _ = build_support_matrices(
                    symbols_upper=ordered_symbols,
                    dorothea_map=dorothea_map,
                    omnipath_pairs=omnipath_pairs,
                    gene2go_upper=gene2go_upper,
                    string_map=string_map,
                )

                for layer in H49_LAYERS:
                    if layer >= layer_embeddings.shape[0]:
                        continue

                    points = layer_embeddings[layer, edge_gene_indices, :]
                    points_pca = reduce_points(
                        points,
                        n_components=20,
                        random_state=22_490 + domain_index * 100 + seed_index * 20 + split_index * 10 + layer,
                    )

                    knn_edges, knn_dists = build_knn_edge_array(points=points_pca, n_neighbors=H49_KNN)
                    if knn_edges.shape[0] < 80:
                        continue

                    knn_support = np.array(
                        [support_undirected[i, j] for i, j in knn_edges],
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
                    delta_auc = (
                        float(auc_bif - auc_dist) if np.isfinite(auc_bif) and np.isfinite(auc_dist) else float("nan")
                    )

                    rng = np.random.default_rng(
                        22_491 + domain_index * 100 + seed_index * 20 + split_index * 10 + layer
                    )
                    null_delta = np.empty(H49_NULL_PERM, dtype=float)

                    for perm_idx in range(H49_NULL_PERM):
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
                                "null_kind": "support_shuffle",
                                "domain": domain,
                                "seed_tag": seed_tag,
                                "split_regime": split_regime,
                                "layer": int(layer),
                                "perm_idx": int(perm_idx),
                                "null_delta_auc": float(delta_perm),
                                "null_spearman": float("nan"),
                            }
                        )

                    p_delta = empirical_upper_tail_p(delta_auc, null_delta)

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
                            "utility_delta_auc_mean": float(utility_value),
                        }
                    )

    by_row_df = pd.DataFrame(rows).sort_values(["domain", "seed_tag", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h49_bifiltration_multiseed_by_seed_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    domain_summary_rows: list[dict[str, object]] = []

    for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
        domain_summary_rows.append(
            {
                "summary_type": "domain_split",
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(group.shape[0]),
                "mean_delta_auc_bif_minus_distance": float(group["delta_auc_bif_minus_distance"].mean()),
                "fraction_delta_positive": float((group["delta_auc_bif_minus_distance"] > 0.0).mean()),
                "fraction_p_delta_lt_0_05": float((group["p_delta_auc_upper"] < 0.05).mean()),
                "combined_fisher_p_delta": float(
                    safe_fisher_p(group["p_delta_auc_upper"].to_numpy(dtype=float))
                ),
                "spearman_utility_vs_delta": float("nan"),
                "p_layer_placebo_upper": float("nan"),
            }
        )

    domain_seed_offsets = {"immune": 0, "lung": 1, "external_lung": 2}
    for domain, domain_group in by_row_df.groupby("domain", sort=True):
        observed_corr = safe_spearman(
            domain_group["delta_auc_bif_minus_distance"].to_numpy(dtype=float),
            domain_group["utility_delta_auc_mean"].to_numpy(dtype=float),
        )

        rng = np.random.default_rng(22_599 + 100 * domain_seed_offsets.get(str(domain), 0))
        null_corr = np.empty(H49_LAYER_PLACEBO_PERM, dtype=float)

        # Layer-placebo: permute delta values within each seed/split block,
        # preserving per-block distributions but breaking depth alignment.
        base = domain_group[["seed_tag", "split_regime", "delta_auc_bif_minus_distance", "utility_delta_auc_mean"]].copy()
        for perm_idx in range(H49_LAYER_PLACEBO_PERM):
            permuted = []
            for (_, _), block in base.groupby(["seed_tag", "split_regime"], sort=False):
                vals = block["delta_auc_bif_minus_distance"].to_numpy(dtype=float)
                permuted.append(rng.permutation(vals))
            delta_perm = np.concatenate(permuted)
            corr_perm = safe_spearman(delta_perm, base["utility_delta_auc_mean"].to_numpy(dtype=float))
            null_corr[perm_idx] = corr_perm
            null_rows.append(
                {
                    "null_kind": "layer_placebo",
                    "domain": domain,
                    "seed_tag": "all",
                    "split_regime": "all",
                    "layer": -1,
                    "perm_idx": int(perm_idx),
                    "null_delta_auc": float("nan"),
                    "null_spearman": float(corr_perm),
                }
            )

        p_layer_placebo = empirical_upper_tail_p(observed_corr, null_corr)

        domain_summary_rows.append(
            {
                "summary_type": "domain_utility",
                "domain": domain,
                "split_regime": "all",
                "n_rows": int(domain_group.shape[0]),
                "mean_delta_auc_bif_minus_distance": float(domain_group["delta_auc_bif_minus_distance"].mean()),
                "fraction_delta_positive": float((domain_group["delta_auc_bif_minus_distance"] > 0.0).mean()),
                "fraction_p_delta_lt_0_05": float((domain_group["p_delta_auc_upper"] < 0.05).mean()),
                "combined_fisher_p_delta": float(
                    safe_fisher_p(domain_group["p_delta_auc_upper"].to_numpy(dtype=float))
                ),
                "spearman_utility_vs_delta": float(observed_corr),
                "p_layer_placebo_upper": float(p_layer_placebo),
            }
        )

    domain_df = pd.DataFrame(domain_summary_rows).sort_values(["summary_type", "domain", "split_regime"])
    domain_path = ITER_DIR / "h49_bifiltration_multiseed_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["null_kind", "domain", "seed_tag", "split_regime", "layer", "perm_idx"]
    )
    null_path = ITER_DIR / "h49_bifiltration_multiseed_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    split_df = domain_df.loc[domain_df["summary_type"] == "domain_split"].copy()
    util_df = domain_df.loc[domain_df["summary_type"] == "domain_utility"].copy()

    summary = {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_bif_minus_distance"].mean())
        if not by_row_df.empty
        else float("nan"),
        "fraction_delta_positive": float((by_row_df["delta_auc_bif_minus_distance"] > 0.0).mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_split_fisher_sig": int((split_df["combined_fisher_p_delta"] < 0.05).sum())
        if not split_df.empty
        else 0,
        "domain_utility_positive": int((util_df["spearman_utility_vs_delta"] > 0.0).sum())
        if not util_df.empty
        else 0,
        "domain_utility_placebo_sig": int((util_df["p_layer_placebo_upper"] < 0.05).sum())
        if not util_df.empty
        else 0,
        "artifact_paths": {
            "by_seed_layer_split": str(by_row_path),
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
        },
    }
    return summary


def run_h50_directed_signed_topology(
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

            top_genes = set(select_top_genes(split_edges, gene_cap=H50_GENE_CAP))
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

            _, support_directed = build_support_matrices(
                symbols_upper=ordered_symbols,
                dorothea_map=dorothea_map,
                omnipath_pairs=omnipath_pairs,
                gene2go_upper=gene2go_upper,
                string_map=string_map,
            )

            for layer in H50_LAYERS:
                if layer >= layer_embeddings.shape[0]:
                    continue

                points = layer_embeddings[layer, edge_gene_indices, :]
                points_pca = reduce_points(
                    points,
                    n_components=20,
                    random_state=22_590 + domain_index * 100 + split_index * 10 + layer,
                )

                knn_edges, knn_dists = build_knn_edge_array(points=points_pca, n_neighbors=H50_KNN)
                if knn_edges.shape[0] < 80:
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

                rng = np.random.default_rng(22_591 + domain_index * 100 + split_index * 10 + layer)

                null_degree = np.empty(H50_NULL_PERM, dtype=float)
                null_sign = np.empty(H50_NULL_PERM, dtype=float)

                for perm_idx in range(H50_NULL_PERM):
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
                            "split_regime": split_regime,
                            "layer": int(layer),
                            "perm_idx": int(perm_idx),
                            "null_delta_auc": float(delta_sign),
                        }
                    )

                p_degree = empirical_upper_tail_p(delta_auc, null_degree)
                p_sign = empirical_upper_tail_p(delta_auc, null_sign)

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
                        "p_degree_upper": float(p_degree),
                        "p_sign_upper": float(p_sign),
                    }
                )

    by_row_df = pd.DataFrame(rows).sort_values(["domain", "split_regime", "layer"])
    by_row_path = ITER_DIR / "h50_directed_signed_topology_by_domain_layer_split.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["null_kind", "domain", "split_regime", "layer", "perm_idx"]
    )
    null_path = ITER_DIR / "h50_directed_signed_topology_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    domain_rows: list[dict[str, object]] = []
    for (domain, split_regime), group in by_row_df.groupby(["domain", "split_regime"], sort=True):
        p_combined = safe_fisher_p(
            np.nanmin(
                np.column_stack(
                    [
                        group["p_degree_upper"].to_numpy(dtype=float),
                        group["p_sign_upper"].to_numpy(dtype=float),
                    ]
                ),
                axis=1,
            )
        )

        domain_rows.append(
            {
                "domain": domain,
                "split_regime": split_regime,
                "n_rows": int(group.shape[0]),
                "mean_auc_directed_signed": float(group["auc_directed_signed"].mean()),
                "mean_auc_distance_only": float(group["auc_distance_only"].mean()),
                "mean_delta_auc_directed_minus_distance": float(
                    group["delta_auc_directed_minus_distance"].mean()
                ),
                "fraction_delta_positive": float(
                    (group["delta_auc_directed_minus_distance"] > 0.0).mean()
                ),
                "fraction_p_degree_lt_0_05": float((group["p_degree_upper"] < 0.05).mean()),
                "fraction_p_sign_lt_0_05": float((group["p_sign_upper"] < 0.05).mean()),
                "combined_fisher_p_best_null": float(p_combined),
            }
        )

    domain_df = pd.DataFrame(domain_rows).sort_values(["domain", "split_regime"])
    domain_path = ITER_DIR / "h50_directed_signed_topology_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)

    summary = {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_auc": float(by_row_df["delta_auc_directed_minus_distance"].mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_split_positive_delta": int(
            (domain_df["mean_delta_auc_directed_minus_distance"] > 0.0).sum()
        )
        if not domain_df.empty
        else 0,
        "domain_split_fisher_sig": int((domain_df["combined_fisher_p_best_null"] < 0.05).sum())
        if not domain_df.empty
        else 0,
        "artifact_paths": {
            "by_domain_layer_split": str(by_row_path),
            "domain_summary": str(domain_path),
            "null_summary": str(null_path),
        },
    }
    return summary


def run_h51_cross_model_motif_fingerprint(
    gene2go_upper: dict[str, set[str]],
) -> dict[str, object]:
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
        if merged.shape[0] < 350:
            continue

        src_tok = merged["source_token_id"].to_numpy(dtype=int)
        tgt_tok = merged["target_token_id"].to_numpy(dtype=int)
        gf_score = compute_geneformer_centered_cosine(centered_emb=centered_emb, src_tok=src_tok, tgt_tok=tgt_tok)

        src_gene_idx = merged["source_idx"].to_numpy(dtype=int)
        tgt_gene_idx = merged["target_idx"].to_numpy(dtype=int)
        source_symbols = merged["source"].astype(str).str.upper().to_numpy()
        target_symbols = merged["target"].astype(str).str.upper().to_numpy()
        unique_symbols = np.unique(np.concatenate([source_symbols, target_symbols]))
        module_map = assign_go_modules(unique_symbols, gene2go_upper=gene2go_upper, top_terms=H51_MODULE_TOP_TERMS)

        layer_embeddings = np.load(
            SCGPT_RUNS_BY_DOMAIN[domain]["seed42_main"] / "layer_gene_embeddings.npy",
            mmap_mode="r",
        )

        for layer in H51_LAYERS:
            if layer >= layer_embeddings.shape[0]:
                continue

            sc_score = compute_scgpt_centered_cosine(
                layer_embeddings=layer_embeddings[layer],
                src_idx=src_gene_idx,
                tgt_idx=tgt_gene_idx,
                pca_dim=64,
                seed=22_710 + domain_index * 10 + layer,
            )

            for k in H51_TOP_K:
                k_eff = min(k, merged.shape[0])
                if k_eff < 50:
                    continue

                idx_sc = np.argpartition(-sc_score, k_eff - 1)[:k_eff]
                idx_gf = np.argpartition(-gf_score, k_eff - 1)[:k_eff]

                sc_edges = list(zip(source_symbols[idx_sc].tolist(), target_symbols[idx_sc].tolist()))
                gf_edges = list(zip(source_symbols[idx_gf].tolist(), target_symbols[idx_gf].tolist()))

                row_variants = {
                    "gene": {
                        "sc_edges": sc_edges,
                        "gf_edges": gf_edges,
                    },
                    "module": {
                        "sc_edges": collapse_edges_by_module(sc_edges, module_map=module_map),
                        "gf_edges": collapse_edges_by_module(gf_edges, module_map=module_map),
                    },
                }

                for variant, edge_view in row_variants.items():
                    sc_view_edges = edge_view["sc_edges"]
                    gf_view_edges = edge_view["gf_edges"]

                    motifs_sc = motif_panel(sc_view_edges)
                    motifs_gf = motif_panel(gf_view_edges)
                    overlap_by_type, overlap_total = motif_overlap_counts(motifs_sc, motifs_gf)

                    rng = np.random.default_rng(22_720 + domain_index * 100 + layer * 10 + k_eff + (variant == "module"))

                    null_degree = np.empty(H51_NULL_PERM, dtype=float)
                    null_module = np.empty(H51_NULL_PERM, dtype=float)
                    null_module[:] = np.nan

                    for perm_idx in range(H51_NULL_PERM):
                        sc_perm_edges_gene = permute_edges_preserve_degree(
                            sources=source_symbols[idx_sc],
                            targets=target_symbols[idx_sc],
                            rng=rng,
                        )
                        gf_perm_edges_gene = permute_edges_preserve_degree(
                            sources=source_symbols[idx_gf],
                            targets=target_symbols[idx_gf],
                            rng=rng,
                        )

                        if variant == "gene":
                            sc_perm_edges = sc_perm_edges_gene
                            gf_perm_edges = gf_perm_edges_gene
                        else:
                            sc_perm_edges = collapse_edges_by_module(sc_perm_edges_gene, module_map=module_map)
                            gf_perm_edges = collapse_edges_by_module(gf_perm_edges_gene, module_map=module_map)

                        overlap_perm = motif_overlap_counts(
                            motif_panel(sc_perm_edges),
                            motif_panel(gf_perm_edges),
                        )[1]
                        null_degree[perm_idx] = float(overlap_perm)
                        null_rows.append(
                            {
                                "null_kind": "degree_rewire",
                                "domain": domain,
                                "layer": int(layer),
                                "top_k": int(k_eff),
                                "variant": variant,
                                "perm_idx": int(perm_idx),
                                "null_overlap_total": float(overlap_perm),
                            }
                        )

                        if variant == "module":
                            shuffled_genes = unique_symbols.copy()
                            shuffled_mod_sc = rng.permutation([module_map[g] for g in shuffled_genes])
                            shuffled_mod_gf = rng.permutation([module_map[g] for g in shuffled_genes])
                            module_map_sc = {g: m for g, m in zip(shuffled_genes.tolist(), shuffled_mod_sc.tolist())}
                            module_map_gf = {g: m for g, m in zip(shuffled_genes.tolist(), shuffled_mod_gf.tolist())}

                            sc_mod_shuffle = collapse_edges_by_module(sc_edges, module_map=module_map_sc)
                            gf_mod_shuffle = collapse_edges_by_module(gf_edges, module_map=module_map_gf)
                            overlap_mod = motif_overlap_counts(
                                motif_panel(sc_mod_shuffle),
                                motif_panel(gf_mod_shuffle),
                            )[1]
                            null_module[perm_idx] = float(overlap_mod)
                            null_rows.append(
                                {
                                    "null_kind": "module_shuffle",
                                    "domain": domain,
                                    "layer": int(layer),
                                    "top_k": int(k_eff),
                                    "variant": variant,
                                    "perm_idx": int(perm_idx),
                                    "null_overlap_total": float(overlap_mod),
                                }
                            )

                    p_degree = empirical_upper_tail_p(float(overlap_total), null_degree)
                    degree_mean = float(np.nanmean(null_degree))
                    degree_std = float(np.nanstd(null_degree))
                    delta_degree = float(overlap_total - degree_mean)
                    z_degree = float(delta_degree / degree_std) if degree_std > 1e-12 else float("nan")

                    if variant == "module":
                        p_module = empirical_upper_tail_p(float(overlap_total), null_module)
                        module_mean = float(np.nanmean(null_module))
                        module_std = float(np.nanstd(null_module))
                        delta_module = float(overlap_total - module_mean)
                        z_module = float(delta_module / module_std) if module_std > 1e-12 else float("nan")
                    else:
                        p_module = float("nan")
                        module_mean = float("nan")
                        module_std = float("nan")
                        delta_module = float("nan")
                        z_module = float("nan")

                    rows.append(
                        {
                            "domain": domain,
                            "layer": int(layer),
                            "top_k": int(k_eff),
                            "variant": variant,
                            "overlap_total": int(overlap_total),
                            "overlap_ffl": int(overlap_by_type["ffl"]),
                            "overlap_bifan": int(overlap_by_type["bifan"]),
                            "overlap_feedback_triad": int(overlap_by_type["feedback_triad"]),
                            "overlap_feedforward_chain": int(overlap_by_type["feedforward_chain"]),
                            "overlap_multi_input": int(overlap_by_type["multi_input"]),
                            "null_degree_mean": float(degree_mean),
                            "null_degree_std": float(degree_std),
                            "delta_degree": float(delta_degree),
                            "z_degree": float(z_degree),
                            "p_degree_upper": float(p_degree),
                            "null_module_mean": float(module_mean),
                            "null_module_std": float(module_std),
                            "delta_module": float(delta_module),
                            "z_module": float(z_module),
                            "p_module_upper": float(p_module),
                        }
                    )

    by_row_df = pd.DataFrame(rows).sort_values(["domain", "variant", "layer", "top_k"])
    by_row_path = ITER_DIR / "h51_cross_model_motif_fingerprint_by_domain_layer_k.csv"
    by_row_df.to_csv(by_row_path, index=False)

    null_df = pd.DataFrame(null_rows).sort_values(
        ["null_kind", "domain", "variant", "layer", "top_k", "perm_idx"]
    )
    null_path = ITER_DIR / "h51_cross_model_motif_fingerprint_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    summary_rows: list[dict[str, object]] = []
    for (domain, variant), group in by_row_df.groupby(["domain", "variant"], sort=True):
        nondegenerate = float((group["null_degree_std"] > 0.0).mean())

        if variant == "module":
            pvals = np.nanmin(
                np.column_stack(
                    [
                        group["p_degree_upper"].to_numpy(dtype=float),
                        group["p_module_upper"].to_numpy(dtype=float),
                    ]
                ),
                axis=1,
            )
        else:
            pvals = group["p_degree_upper"].to_numpy(dtype=float)

        summary_rows.append(
            {
                "domain": domain,
                "variant": variant,
                "n_rows": int(group.shape[0]),
                "mean_overlap_total": float(group["overlap_total"].mean()),
                "mean_delta_degree": float(group["delta_degree"].mean()),
                "fraction_delta_degree_positive": float((group["delta_degree"] > 0.0).mean()),
                "fraction_p_degree_lt_0_05": float((group["p_degree_upper"] < 0.05).mean()),
                "mean_delta_module": float(group["delta_module"].mean()),
                "fraction_p_module_lt_0_05": float((group["p_module_upper"] < 0.05).mean()),
                "combined_fisher_p_best": float(safe_fisher_p(pvals)),
                "fraction_nondegenerate_degree_null": float(nondegenerate),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["domain", "variant"])
    summary_path = ITER_DIR / "h51_cross_model_motif_fingerprint_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    summary = {
        "rows_tested": int(by_row_df.shape[0]),
        "mean_delta_degree": float(by_row_df["delta_degree"].mean())
        if not by_row_df.empty
        else float("nan"),
        "domain_variant_fisher_sig": int((summary_df["combined_fisher_p_best"] < 0.05).sum())
        if not summary_df.empty
        else 0,
        "domain_with_any_sig_variant": int(
            (summary_df.groupby("domain")["combined_fisher_p_best"].min() < 0.05).sum()
        )
        if not summary_df.empty
        else 0,
        "artifact_paths": {
            "by_domain_layer_k": str(by_row_path),
            "summary": str(summary_path),
            "null_summary": str(null_path),
        },
    }
    return summary


def main() -> None:
    required_paths = [
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
    omnipath_pairs = load_omnipath_pairs()
    gene2go_upper = load_gene2go_upper()
    string_map = load_string_scores_from_cache(STRING_CACHE_PATH)

    h49_summary = run_h49_bifiltration_multiseed(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h50_summary = run_h50_directed_signed_topology(
        dorothea_map=dorothea_map,
        omnipath_pairs=omnipath_pairs,
        gene2go_upper=gene2go_upper,
        string_map=string_map,
    )
    h51_summary = run_h51_cross_model_motif_fingerprint(gene2go_upper=gene2go_upper)

    summary = {
        "iteration": "iter_0022",
        "h49": h49_summary,
        "h50": h50_summary,
        "h51": h51_summary,
        "environment": {
            "python_env": "subproject40-topology",
            "string_cache_used": str(STRING_CACHE_PATH),
            "geneformer_model": "ctheodoris/Geneformer",
        },
    }
    summary_path = ITER_DIR / "iter0022_screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
