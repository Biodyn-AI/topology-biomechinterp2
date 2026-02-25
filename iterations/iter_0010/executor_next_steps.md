# Executor Next Steps — iter_0010

1. **Promote H13 to cross-domain replication (novel domain check)**
- Re-run geodesic-vs-euclidean edge AUROC packet on `lung` and `external_lung` scGPT embeddings with the same split protocol and permutation null.
- Gate: require positive mean delta in both splits and at least `>=4/12` dual-split significant layers per domain.

2. **Stress-test H14 with stronger topology controls (one carry-over refinement)**
- Keep the same bootstrap/filtration grid, add one additional null family (distance-preserving row permutation or kNN-edge rewiring surrogate) while retaining feature-shuffle.
- Gate: maintain `>75%` settings-positive fraction per layer and at least `>=8/12` layers with combined p<0.05 under both null families.

3. **Resolve H15 domain heterogeneity**
- Extend disagreement-trend analysis using per-edge cross-model score artifacts (if surfaced) to avoid bin-level aggregation loss.
- Gate: determine whether lung-negative trend persists after controlling for source-degree and baseline edge prevalence.

4. **Biological anchoring follow-through (high-upside branch)**
- For H13/H14 robust layers, perform TF/pathway enrichment on top cycle-contributing genes and test overlap with TRRUST/DoRothEA priors.
- Gate: reproducible enrichment direction across both source/target splits.
