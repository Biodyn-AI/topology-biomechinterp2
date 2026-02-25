# Executor Next Steps — iter_0003

1. Replicate H01 persistent-homology null test on immune and external-lung scGPT embeddings to check domain generality of high H1 deltas.
2. Add split-regime robustness for H01 with source-disjoint and target-disjoint edge subsets, keeping identical topological summary metrics.
3. Promote H02 from feature-summary alignment to residual-level alignment by materializing Geneformer gene/token embedding tensors for matched genes and running CKA/Procrustes + permutation null.
4. Test biological anchoring of H01 by comparing high-persistence genes/modules against TRRUST/GO module enrichment and evaluating whether high-persistence neighborhoods are regulator-enriched.
5. Stress-test null sensitivity by adding distance-matrix permutation and kNN graph rewiring controls for the same layers where H1 deltas peak.
