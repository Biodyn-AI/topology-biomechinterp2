# Brainstormer Hypothesis Roadmap — iter_0005

## Candidate Hypotheses (Next Iteration)

| ID | Family | Hypothesis | Concrete follow-up experiment | Falsifier | value | cost |
|---|---|---|---|---|---|---|
| N21 | topology | H1 signal survives a biologically interpretable stronger null (degree-preserving/geodesic rewiring). | Replace `distance_permutation` with kNN graph rewiring (preserve node degree, recompute geodesic distances), keep the same 3-domain, 2-split, top+weak layer grid and Fisher combine across seeds. | Fewer than 25% of tested domain-layer-split units remain significant (`Fisher p < 0.05`) under rewiring null. | high | medium |
| N22 | topology | Split robustness has depth structure (not random failures) in immune. | Expand immune from 2 layers to all 12 layers under feature-shuffle + rewiring nulls for both source/target splits; produce a layer-by-split pass matrix. | No coherent depth pattern and at most 2/12 layers pass both splits. | high | medium |
| N23 | topology | Positive topology is metric-stable across persistence summaries. | Recompute per-layer effects using multiple PH summaries (`H1 sum`, `max H1 lifetime`, `Betti curve area`, `persistent entropy`) and test sign/significance consistency. | Effect direction flips or becomes null for most metrics in top layers. | medium | low |
| N24 | manifold geometry | Layers with stronger topology also show stronger geodesic nonlinearity. | Per layer, compute geodesic-vs-Euclidean distortion (Isomap residual variance or trustworthiness/continuity gap) and correlate with H1 delta across seeds/domains with permutation nulls. | Correlations are sign-inconsistent and non-significant after pooling domains. | medium | medium |
| N25 | manifold geometry | Topology-strength layers have lower local tangent consistency. | Estimate local PCA tangent alignment (neighbor patch angle dispersion) per layer; test coupling with H1 deltas. | No reproducible coupling in at least two of three domains. | medium | medium |
| N26 | cross-model alignment | Topological signatures align across scGPT and Geneformer when matched by gene/layer/domain. | Materialize matched-gene residual embeddings, compute persistence images/Betti curves per layer, then test similarity with gene-label permutation nulls. | Similarity is null-like in at least 2/3 domains. | high | high |
| N27 | cross-model alignment | Layers that are split-robust in scGPT align better with Geneformer than split-brittle layers. | After N22, stratify layers into robust vs brittle and compare CKA/Procrustes alignment distributions across strata. | No alignment gap between robust and brittle strata. | medium | medium |
| N28 | biological anchoring | Split-robust topology genes are enriched for regulatory programs. | For robust layers, rank genes by H1 contribution (ablation-drop proxy), run TRRUST/DoRothEA/GO enrichment with FDR control by domain. | No enrichment (`FDR < 0.1`) in at least two domains. | high | medium |
| N29 | biological anchoring | Topological-core genes improve network recovery over matched controls. | Restrict GRN recovery/evaluation to topological-core genes and compare AUROC/AUPR versus expression/degree-matched random gene sets. | Core sets are not better than matched controls across domains. | medium | high |
| N30 | controls/null stress test | Topology effect is not an expression-variance or degree confound. | Build confound-matched null gene subsets (match mean expression, variance, graph degree) and recompute H1 deltas with bootstrap CIs. | Observed deltas fall inside matched-null CIs for most tested layers. | high | medium |

## Prioritized Shortlist
- First wave: `N21`, `N22`, `N28`.
- Second wave: `N30`, `N23`.
- Stretch branch: `N26` (only if matched-gene Geneformer residual tensors are available).
