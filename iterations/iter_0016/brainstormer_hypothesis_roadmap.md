# Brainstormer Hypothesis Roadmap — iter_0016

## Retire / Deprioritize
| Direction | Evidence | Decision |
|---|---|---|
| GW-primary correspondence recovery (`H27/H29`) | Repeated controlled failures for both map quality and transfer utility | `retire_now` |
| Rewiring-null survival lineage (`H07/H09/H12`) | Long negative streak with no rescue trend across calibration variants | `retire_now` |
| Raw Forman-curvature enrichment (`H23` form) | Opposite-direction, below-chance outcomes across domains | `retire_now` |
| Triangle-thinness/hyperbolicity edge score (`H30` form) | Underperforms geodesic baseline in nearly all strata | `retire_now` |
| Cycle-consistency-first alignment objective (`H33` form) | Improves cycle metrics without transfer gain | `rescue_once_with_major_change` |
| Pooled diffusion claim without stratified mechanism (`H28/H31` framing) | Positive mean effects but weak cross-domain robustness | `rescue_once_with_major_change` |
| Universal-sign intrinsic coupling framing (`H04/H18/H21/H22`) | Repeated sign flips by domain/split/depth | `rescue_once_with_major_change` |

## New Hypothesis Portfolio
| ID | Theme | One-sentence hypothesis | Concrete test design | Expected signal if true | Null / control | Value | Cost |
|---|---|---|---|---|---|---|---|
| N141 | Graph geometry | Convexity-deficit + detour features remain predictive across seeds and add signal beyond diffusion covariates. | Re-run `H32` on seeds `42/43/44` with both splits and layers `0/3/7/11`; compare nested models `geodesic+diffusion` vs `geodesic+diffusion+convexity+detour`. | Positive incremental delta in `>=2/3` domains with seed-stable sign. | Degree x coexpression x geodesic matched permutation and feature-shuffle ablation. | high | low |
| N142 | Topology (filtration variants) | Regulatory signal is a consensus PH effect across filtration choices, not a single-filtration artifact. | Compute PH summaries under Vietoris-Rips, witness, and kNN-clique filtrations on matched gene subsets; test consensus score against edge labels. | Consensus score outperforms each single filtration and is split-robust. | Filtration-label permutation and coordinate bootstrap. | high | medium |
| N143 | Topology (stability) | True signal has lower diffusion-time barcode drift than matched nulls. | Build diffusion-time vineyards (`t=1,2,4,8`) and quantify barcode drift per layer/split/domain; relate drift to edge AUROC. | Positive edges concentrate in low-drift regions with significant drift-gap vs null. | Diffusion-kernel row/column permutation preserving degree profile. | medium | medium |
| N144 | Topology (directed complexes) | Directed/signed flag-complex features from TF->target orientation outperform undirected PH features. | Construct directed simplicial complexes using signed priors and compute directed topology descriptors per layer/split. | Directed features improve AUROC and calibration over undirected baselines in `>=2` domains. | Direction/sign randomization preserving in/out-degree and sign prevalence. | high | high |
| N145 | Curvature geometry | Residual Ollivier-Ricci curvature magnitude (after regressing degree and distance) captures regulatory edges better than raw curvature sign. | Estimate OR curvature on kNN graphs, regress curvature on degree/geodesic/coexpression, and use residual magnitude in edge models. | Positive residual-magnitude coefficient and significant incremental gain. | Degree-preserving rewiring and residual-label permutation. | medium | medium |
| N146 | Geodesic geometry | Regulatory positives are enriched on high-detour geodesic corridors with central bridge endpoints. | Compute edge-level detour, endpoint betweenness, and geodesic corridor occupancy; test joint model vs geodesic-only baseline. | Joint geometry model shows robust positive delta and higher recall at fixed precision. | Endpoint-matched permutations by degree/geodesic/coexpression bins. | medium | low |
| N147 | Local linearity | Depth-specific local-linearity breakpoints explain split asymmetry better than global averages. | Fit piecewise depth models of local linearity/reconstruction error for each domain/split and test breakpoint-location association with edge AUROC shifts. | Reproducible breakpoint-depth shift between source and target splits in at least two domains. | Layer-order permutation and random breakpoint placement control. | medium | low |
| N148 | Intrinsic dimension | ID tail-shape and anisotropy (not mean ID) predict where geometry-to-edge coupling flips sign. | Estimate local ID with TWO-NN and participation-ratio; derive skew/tail metrics plus anisotropy and fit interaction models. | Significant ID-shape x split interactions with consistent direction by domain. | Estimator-swap robustness and within-layer label permutation. | medium | low |
| N149 | Cross-model alignment | Anchor-regularized spectral alignment optimized for transfer utility can beat both CCA-only and cycle-only objectives. | Align Laplacian eigenbases after CCA whitening with soft TRRUST anchors; train objective directly on held-out transfer AUROC proxy. | Transfer AUROC improves over CCA and `H33` in `>=2/3` domains. | Random-anchor sets and eigenvector sign/permutation controls. | high | high |
| N150 | Cross-model topology transfer | Persistence-image transport across models preserves functional structure even when gene correspondence is imperfect. | Convert PH summaries to persistence images for scGPT/Geneformer and align via CCA/Procrustes in PI space; test transfer to edge labels. | PI-space transfer AUROC exceeds random-map and matches/exceeds gene-level map baselines. | PI-pixel permutation and correspondence permutation controls. | medium | medium |
| N151 | Cross-model cycle rescue | Tri-domain cycle consistency becomes useful only when biologically anchored edges are explicitly up-weighted in the loss. | Re-run tri-domain objective with anchor-weighted loss and holdout-anchor evaluation; report both cycle metrics and transfer AUROC. | Positive transfer delta and improved held-out-anchor retrieval, not just cycle-return gain. | Anchor-label shuffle and weight-swap ablations. | high | medium |
| N152 | Biological anchoring | Convexity/detour signal is strongest in high-consensus prior edges (TRRUST+STRING+GO). | Build consensus prior tiers and fit geometry x tier interaction models by domain/split/layer. | Positive interaction slope for consensus tiers with significant incremental fit gain. | Prior-tier permutation within degree/coexpression strata. | high | medium |
| N153 | Cell ontology | Geometry-to-function coupling is cell-ontology specific and diluted in pooled analysis. | Stratify edge tests by cell ontology groups and run hierarchical mixed-effects models with domain/split random effects. | At least one reproducible ontology stratum per domain with strong positive geometry effect. | Ontology-label shuffle and prevalence-matched subsampling controls. | high | medium |
| N154 | Mechanistic motifs | Feed-forward, feedback, and bifan motifs occupy separable geometry/topology regions that transfer across models. | Label motifs from prior networks, train motif classifier on geometry+topology features in scGPT, and evaluate transfer to Geneformer. | Above-chance motif classification and stable transfer margins across domains. | Motif-label permutation and degree-preserving motif rewiring. | medium | medium |

## Top 3 for Immediate Execution
1. High-probability discovery candidate: **N141 (multi-seed convexity/detour incremental test)**.
- Why now: strongest current positive branch with low implementation lift.
- Immediate gate: positive incremental delta vs `geodesic+diffusion` in `>=2/3` domains and both splits.

2. High-risk/high-reward candidate: **N149 (anchor-regularized utility-optimized spectral alignment)**.
- Why now: it directly addresses the `H33` failure mode by optimizing the downstream metric instead of cycle quality.
- Immediate gate: transfer AUROC improvement over CCA-only and current cycle-consistent baseline in `>=2/3` domains.

3. Cheap broad-screen candidate: **N147 (local-linearity breakpoint screen)**.
- Why now: uses existing features and gives rapid information about split/depth mechanism heterogeneity.
- Immediate gate: reproducible split-specific breakpoint pattern in at least two domains.
