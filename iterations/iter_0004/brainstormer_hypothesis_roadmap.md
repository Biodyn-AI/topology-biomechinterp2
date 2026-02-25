# Brainstormer Hypothesis Roadmap — iter_0004

## Candidate Hypotheses (Next Iteration)

| ID | Family | Hypothesis | Concrete follow-up experiment | Falsifier | value | cost |
|---|---|---|---|---|---|---|
| N11 | topology | Cross-domain H1 signal survives stronger null families, not just feature-shuffle. | For top layers per domain (lung L0, immune L7, external-lung L0 + one weak late layer), test at least two stronger nulls: (1) pairwise-distance permutation, (2) degree-preserving kNN rewiring. Recompute empirical p-values and Fisher combine across seeds. | Most tested layer-domain pairs lose significance (`p > 0.1`) under all stronger nulls. | high | medium |
| N12 | topology | H1 signal is split-robust under disjoint gene partitions. | Partition genes into source-disjoint and target-disjoint sets (TRRUST/DoRothEA roles or deterministic random halves if roles unavailable), rerun H1 protocol per split/layer/domain. | Any split collapses to near-null (`<50%` layers with Fisher `p<0.05`) in at least two domains. | high | medium |
| N13 | topology | Layer-level topology profile is conserved across tissues. | Compute cross-domain correlation of layer H1-delta vectors (lung vs immune vs external-lung), calibrate with layer-label permutation. | Cross-domain layer-profile correlation is null-like in all domain pairs. | medium | low |
| N14 | manifold geometry | H1 couples with manifold spread/linearity after replacing unstable local-ID metric. | Replace `mle_intrinsic_dim` with robust estimators (TwoNN and local PCA dimension), include lung + immune + external-lung, meta-analyze sign and Fisher p across seeds/domains. | Pooled effects are non-significant and sign-inconsistent across domains. | high | medium |
| N15 | manifold geometry | High-H1 layers have greater nonlinearity beyond PCA proxies. | Measure Isomap residual variance (or geodesic distortion) per layer and test correlation with H1 delta under layer permutations. | No consistent positive association across seeds and domains. | medium | medium |
| N16 | cross-model alignment | Residual-level scGPT/Geneformer spaces are above-null aligned when matched by gene and layer/domain. | Materialize matched-gene residual vectors, run linear CKA + Procrustes with gene-label permutation nulls per domain/layer. | Alignment is not above null in at least 2 of 3 domains. | high | high |
| N17 | cross-model alignment | Topological signatures (not just vector geometry) align across models. | Compare persistence images/Betti curves between matched scGPT and Geneformer layer embeddings; test similarity against permutation null. | Similarity is indistinguishable from null in most layers/domains. | medium | high |
| N18 | biological anchoring | High-persistence genes are enriched for known regulator programs. | In each top layer/domain, rank genes by contribution to H1 delta (ablation-drop or cycle-neighborhood membership) and run TRRUST/DoRothEA/GO enrichment with FDR control. | No regulator/pathway enrichment (`FDR < 0.1`) in top gene sets. | high | medium |
| N19 | biological anchoring | Topological-core genes improve GRN recovery versus matched controls. | Build edge predictions restricted to topological-core genes and compare AUROC/AUPR to expression- and degree-matched random gene sets using TRRUST/STRING references. | Core-gene restricted recovery is not better than matched controls. | medium | high |
| N20 | controls/null stress test | Positive H1 deltas are not driven by simple confounds (expression variance, gene degree, sampling). | Run confound-matched resampling nulls and gene-bootstrap confidence intervals for H1 deltas across layers/domains. | Observed deltas fall within matched-null intervals for most layers. | high | medium |

## Shortlist for Immediate Execution
- First wave: `N11`, `N12`, `N18`.
- Keep `N16` as a stretch branch unless matched-gene residual tensors are already available.
