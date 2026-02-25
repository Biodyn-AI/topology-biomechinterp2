# Brainstormer Hypothesis Roadmap — iter_0012

## Retire / Deprioritize
| Direction | Evidence | Action |
|---|---|---|
| Rewiring-survival branch (`H07/H09/H12`) | Multiple calibrated runs (`iter_0006`-`iter_0008`) stayed uniformly negative with no rescue trend | `retire_now` |
| Distortion-lower-tail rewiring rescue | Repeated non-significant outcomes and no directional correction | `retire_now` |
| Plain Hungarian OT unsupervised map (`H20` OT variant as implemented) | `iter_0012` mean top-1 `0.0024`, `0/3` domains significant | `retire_now` |
| Confidence-tier monotonicity in raw DoRothEA tiers (`H19` current form) | Direction flips opposite to claim in both splits; tier prevalence is saturated at high confidence | `rescue_once_with_major_change` |
| Intrinsic mechanism framed as positive coupling (`H21` current sign) | Significant inverse coupling in target split (`rho=-0.4079`, `p=0.0190`) | `rescue_once_with_major_change` |
| Coarse disagreement-bin trend (`H15` style) | Domain-sign instability and coarse-bin confounding | `rescue_once_with_major_change` |
| Bridge-conditioned rewiring explanation (`H11`) | Identifiability failure due split-confounded bridge strata | `rescue_once_with_major_change` |

## New Hypothesis Portfolio
| ID | One-sentence hypothesis | Concrete test design | Expected signal if true | Null / control | Value | Cost |
|---|---|---|---|---|---|---|
| N84 | A geodesic-distance x prior-confidence bifiltration will reveal stable persistence zones that single-parameter filtrations miss. | Build bifiltration slices over geodesic radius and confidence weight (DoRothEA/TRRUST/STRING), compute rank-invariant proxies per layer/split. | Long-lived Betti features concentrate in high-confidence/high-geodesic-structure region and replicate across seeds. | Tier-label shuffle plus geodesic-distance shuffle with matched marginals. | high | high |
| N85 | Signed directed topology on TF→target edges carries stronger regulatory signal than undirected community structure. | Construct directed/signed flag complexes on layer-wise TF-target neighborhoods and compare persistence summaries to H16 baseline. | Directed/signed persistence features outperform same-community AUROC and show consistent split gains. | Edge-sign permutation and direction randomization with preserved degree. | high | high |
| N86 | Cross-layer zigzag persistence will isolate modules that persist through depth and correspond to high geodesic-lift layers. | Compute zigzag persistence across consecutive layers on shared gene subsets and align interval lifetimes to H13 profiles. | Long zigzag intervals peak near layers with strong geodesic-over-euclidean gain. | Layer-order permutation and random layer stitching. | high | high |
| N87 | Density-equalized witness complexes will retain topology-stability signal while reducing prevalence confounds seen in H19. | Build witness complexes with tier-balanced landmark sampling and rerun H14-style stability summaries by split. | Positive persistence-vs-null deltas remain after balancing tier/degree density. | Landmark re-sampling null and degree-matched random witnesses. | medium | medium |
| N88 | Strongly negative Ricci/Forman curvature edges are enriched for true regulatory interactions. | Compute edge curvature on kNN graphs per layer/split and test edge-label enrichment across curvature quantiles. | Monotonic increase in positive-edge rate toward most-negative curvature bins. | Degree-preserving rewiring curvature baseline plus label permutation. | high | medium |
| N89 | Geodesic detour ratio (path length / Euclidean length) identifies non-linear regulatory conduits missed by local reconstruction features. | Compute detour ratio for candidate edges and evaluate AUROC/trend by split/layer/domain. | High-detour edges have higher positive-rate enrichment and stronger effect in high-H13 layers. | Path-endpoint shuffle with distance-matched controls. | medium | low |
| N90 | A depth-phase transition in local linearity explains why H21 is source-positive but target-negative in late layers. | Fit split x layer interaction models on local-linearity/reconstruction features across immune/lung/external-lung. | Significant negative interaction in late layers for target-disjoint splits replicates across domains. | Layer-shuffle and split-label permutation. | high | low |
| N91 | Intrinsic-dimension distribution shape (variance, skew, tail mass), not mean ID, predicts geodesic lift and H21 sign changes. | Estimate local ID with two estimators, compute distribution-shape statistics per layer/split, regress against H13/H21 metrics. | Shape statistics show stable sign-consistent associations where mean ID failed. | Estimator swap robustness and within-seed layer permutations. | high | medium |
| N92 | CCA-warm-start Sinkhorn OT can recover meaningful unsupervised cross-model maps where Hungarian OT collapsed. | Initialize shared latent space with CCA/PCA, run entropic OT, evaluate top-1 recovery, neighborhood Jaccard, and transfer AUROC in 3 domains. | Top-1 and transfer metrics improve over Hungarian OT in at least 2 domains and pass null tests. | Random latent rotation and random-map permutation nulls. | high | high |
| N93 | Gromov-Wasserstein graph alignment will preserve cross-model neighborhood geometry better than pointwise OT. | Align scGPT/Geneformer neighborhood graphs with GW and compare module overlap/transfer AUROC to Procrustes and OT baselines. | GW improves neighborhood preservation and yields comparable or better transfer AUROC than Procrustes in at least 1 domain. | Degree-matched random graph correspondences. | high | high |
| N94 | Cross-model depth is non-isomorphic, and dynamic-time-warped layer alignment will outperform fixed-layer matching. | Build layerwise geometric signatures (curvature, ID-shape, geodesic gain) and run DTW alignment across models/domains. | DTW-aligned layer pairs produce higher transfer metrics than fixed (e.g., layer 0/3) matches. | Random layer-pair baseline and reversed-layer control. | medium | medium |
| N95 | Multi-prior biological anchoring with prevalence-adjusted models will recover a robust confidence signal despite H19 monotonic failure. | Combine DoRothEA, TRRUST, STRING confidence into calibrated edge priors; fit prevalence- and degree-adjusted mixed models for community/geometry effects. | Adjusted effect sizes remain positive and significant even if raw monotonic slopes are not. | Prior-label permutation and prevalence-matched stratified bootstrap. | high | medium |
| N96 | Topology/geometry effects are concentrated in GO and Cell Ontology programs rather than global edge pools. | Rerun H13/H14/H16-style tests within marker/pathway-restricted subgraphs (immune signaling, epithelial, stromal, etc.). | Specific ontology/pathway strata show amplified and reproducible topology/geometry signal. | Size/expression-matched random gene-set controls. | medium | medium |
| N97 | Feed-forward and feedback motif edges occupy a distinct representation manifold region with persistent geometric signatures across layers. | Extract 3-node motifs from prior networks, embed motif edges in representation features (curvature, detour, community, ID-shape), and test motif-class separability by layer. | Motif classes separate with stable margins and align with layers showing high transfer/geodesic gains. | Degree-preserving motif rewiring and motif-label permutation. | medium | medium |

## Top 3 for Immediate Execution
1. High-probability discovery candidate: **N95 (multi-prior, prevalence-adjusted biological anchoring)**.
- Why: it directly rescues a failed branch (`H19`) with a materially better design while leveraging a stable positive base (`H16`).
- Immediate gate: adjusted community/geometry coefficients remain positive in both source and target splits with permutation-calibrated `p < 0.05`.

2. High-risk/high-reward candidate: **N92 (CCA-warm-start Sinkhorn OT unsupervised rescue)**.
- Why: success would convert cross-model transfer from map-aware only (`H20` Procrustes) to truly unsupervised structure discovery.
- Immediate gate: OT top-1 recovery and transfer AUROC both beat Hungarian OT and random-map nulls in at least `2/3` domains.

3. Cheap broad-screen candidate: **N90 (split x depth phase-transition screen for local linearity/reconstruction)**.
- Why: low-cost and directly explains the strongest unresolved pattern in `H21` (late-layer target inversion).
- Immediate gate: significant negative target-split interaction in late layers reproduced in at least two domains.
