# Brainstormer Hypothesis Roadmap — iter_0003

## Candidate Hypotheses (Next Iteration)

| ID | Family | Hypothesis | Concrete follow-up experiment | Falsifier | value | cost |
|---|---|---|---|---|---|---|
| N01 | topology | H1 persistence signal generalizes beyond lung to immune and external-lung scGPT embeddings. | Re-run current H1-vs-feature-shuffle protocol on `cycle4_immune_*` and `cycle7_external_lung_*` (3 seeds each), layer-wise Fisher combine. | Any domain has `<50%` layers with Fisher p `<0.05`. | high | medium |
| N02 | topology | Topological signal is stable under split robustness (source-disjoint / target-disjoint gene partitions). | Build disjoint gene subsets from regulator-source vs target-role partitions; recompute H1 deltas per subset/layer. | Held-out partition collapses to null across most layers. | high | medium |
| N03 | topology + controls | H1 signal survives stronger structure-breaking nulls. | For top layers (L0/L1/L7/L9), test nulls: distance-matrix permutation, kNN rewiring, and random orthogonal projection controls. | Empirical p becomes non-significant (`>0.1`) for most tested nulls. | high | medium |
| N04 | manifold geometry | Layers with stronger H1 also show distinct manifold statistics (intrinsic dimension, anisotropy, geodesic distortion). | Compute per-layer manifold metrics and correlate with `mean_h1_sum_delta` across layers/seeds. | Correlations weak/inconsistent sign across seeds. | medium | medium |
| N05 | manifold geometry | H1 conclusions are robust to embedding compression choice. | Sweep PCA dims (`10/20/32/64`) and random projections; compare effect-size rank stability by layer. | Layer ranking/effect sizes are unstable or collapse outside one dimension choice. | medium | low |
| N06 | cross-model alignment | Matched-gene residual spaces in scGPT and Geneformer are above-null aligned. | Materialize matched-gene vectors per domain/layer and run linear CKA + Procrustes with gene-label permutation nulls. | CKA/Procrustes not above permutation in at least 2/3 domains. | high | high |
| N07 | cross-model alignment | Cross-model agreement is stronger on topological core genes than on all genes. | Define topological-core genes from high-persistence layers; compute scGPT/Geneformer edge-score concordance restricted to this set vs full set. | Concordance does not improve on topological-core subset. | medium | medium |
| N08 | biological anchoring | High-persistence gene neighborhoods are enriched for known regulator programs. | Extract genes contributing to top persistent cycles and run TRRUST/DoRothEA/GO enrichment with FDR control. | No meaningful enrichment (`FDR<0.1`) in top-cycle neighborhoods. | high | medium |
| N09 | biological anchoring | Late-layer weakness (L11) reflects biologically different gene content (less regulatory concentration). | Compare L0 vs L11 high-contribution genes on TF density, regulon degree, and pathway categories. | No TF/regulon depletion in weak layers. | medium | low |
| N10 | null stress test | Positive H1 is not a byproduct of trivial expression/degree confounds. | Recompute H1 on expression-matched and network-degree-matched random gene sets (same size as observed). | Matched controls reproduce observed deltas. | high | medium |

## Notes
- N01 + N03 is the fastest way to convert current lung-only success into a robust cross-domain claim.
- N06 remains strategically important but should be treated as a higher-cost stretch unless matched-gene residual tensors are already available.
