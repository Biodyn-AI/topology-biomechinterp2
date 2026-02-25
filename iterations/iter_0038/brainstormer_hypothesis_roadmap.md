# Brainstormer Hypothesis Roadmap - iter_0038

## Retire / Deprioritize
1. Cross-model unsupervised concordance lineage (`H74/H77/H80/H83/H86/H96/H99`) -> `retire_now`.
Reason: repeated null-fragility with `0/3` domain-level null-gap support even after structural resets.

2. Additive bridge-curvature utility lineage (`H95/H97`) -> `retire_now`.
Reason: repeated strong directional lift with persistent `0/6` positive mean null-gap domain-splits.

3. Standalone/additive intrinsic-dimension utility forms (`H54/H60/H63/H66/H89/H98`) -> `retire_now`.
Reason: no robust promotion signal after multiple retries.

4. GO-overlap additive stratification on weighted filtration (`H94` form) -> `retire_now`.
Reason: broad underperformance versus the global weighted filtration backbone.

5. Standalone additive topology-stability trajectory forms (`H90/H92`) -> `retire_now`.
Reason: repeated near-zero directional effects that collapse under null controls.

6. Global pooled interaction overlays (`H73/H76/H79`) -> `rescue_once_with_major_change`.
Required change: hierarchical/local partial pooling with explicit shrinkage and hard robustness gates.

## New Hypothesis Portfolio
| ID | Category | One-sentence hypothesis | Concrete test design | Expected signal if true | Null/control | Value | Cost |
|---|---|---|---|---|---|---|---|
| `N493` | Topology (relative PH) | Relative persistence between edge-anchored neighborhood complexes and matched background complexes carries robust regulatory signal beyond absolute barcodes. | For each candidate edge, build edge-anchored and matched-background simplicial complexes, compute relative H1/H2 lifetime descriptors, and test incremental utility over `H93/H91` across domains/splits/layers. | Positive mean delta in `>=5/6` domain-splits and positive mean null-gap in `>=4/6`. | Anchor-node permutation (degree/length matched), complex-size matched random backgrounds, label permutation. | high | medium |
| `N494` | Topology (zigzag across depth) | True edges show higher zigzag barcode continuity across layers (`0->3->7->11`) than negatives. | Compute per-edge barcodes per layer and zigzag continuity metrics (bottleneck trajectory statistics), then add continuity features to `H91`. | Positive utility with positive mean null-gap in `>=4/6` domain-splits. | Layer-order permutation, barcode-pair randomization within layer bins, label permutation. | high | medium-high |
| `N495` | Topology (fibered bifiltration) | A bifiltration over geodesic distance and signed-support confidence isolates stable causal loop signatures. | Build 2-parameter filtration surfaces, extract fibered summaries on angle grid, and evaluate added utility on top of `H93`. | Positive mean delta in `>=4/6` and positive mean null-gap in `>=3/6` domain-splits. | Axis swap, support-sign shuffle within degree bins, confidence shuffle, label permutation. | high | high |
| `N496` | Topology (local homology) | Positive edges reside in neighborhoods with higher stable local homology rank around edge midpoints. | Construct midpoint-centered balls at multiple radii, compute local homology rank/lifetime descriptors, and test additive plus interaction terms with `H91`. | Stable positive coefficients and positive mean null-gap in `>=3/6` domain-splits. | Midpoint randomization with geodesic-length matching, radius-order permutation, label permutation. | medium | medium |
| `N497` | Topology (lifetime derivative spectrum) | Derivative shape of persistence-lifetime curves across filtration quantiles captures weak but robust signal missed by raw lifetimes. | Reuse existing weighted-filtration outputs to compute slope/curvature/inflection descriptors, run seed42 breadth screen over all domains/splits/layers `{0,3,7,11}`. | Positive mean delta in `>=4/6` domain-splits with positive mean null-gap in `>=2/6`. | Quantile-order permutation, derivative-sign randomization, label permutation. | medium | low |
| `N498` | Manifold geometry (geodesic corridors) | True edges occupy low-entropy high-multiplicity geodesic corridors in the kNN manifold. | Compute k-shortest-path multiplicity, corridor entropy, and corridor perturbation sensitivity; test incremental lift over `H91`. | Positive mean null-gap in `>=3/6` domain-splits, strongest at layers `7/11`. | Endpoint-matched random pairs, path-order permutation, degree-preserving graph rewiring. | medium | medium |
| `N499` | Manifold geometry (anisotropic curvature) | Directional curvature asymmetry around source-to-target neighborhoods is predictive even when scalar curvature means fail. | Estimate multi-scale directional curvature proxies (including Ollivier-style local transport curvature) and anisotropy ratios, then test over baseline features. | Consistent effect sign across domains and positive mean null-gap in `>=3/6` domain-splits. | Direction swap, transport-plan randomization, scale permutation, label permutation. | medium | medium-high |
| `N500` | Manifold geometry x ID interactions | Intrinsic-dimension gradients become informative only via interaction with high weighted persistence, not as standalone features. | Compute ID gradient along source-target geodesics and fit interaction-only terms with `H93` persistence features. | Positive interaction coefficients with positive mean null-gap in `>=3/6` domain-splits. | Interaction-term permutation within persistence bins, ID-gradient shuffle, label permutation. | high | medium |
| `N501` | Cross-model alignment (OT + depth warp) | Cross-model alignment emerges when GO-module persistence images are matched by OT with monotone depth warping. | Build module persistence images in scGPT/Geneformer, fit entropic OT and monotone layer-warp on one domain, evaluate held-out domain concordance. | Positive primary null-gap in `>=2/3` domains and lower warped OT cost than nulls. | Module-membership permutation, depth-order permutation, random-warp baseline, random-subspace baseline. | high | high |
| `N502` | Cross-model alignment (spectral structure) | Shared module-graph Laplacian eigenspectra are aligned even when pairwise role concordance is weak. | Derive module interaction graphs from robust in-model descriptors and compare leading spectral fingerprints across models/domains/layers. | Positive eigenspectrum-concordance null-gap in `>=2/3` domains. | Degree-preserving module-graph rewiring, eigenvector sign randomization, label permutation. | medium-high | medium-high |
| `N503` | Cross-model transfer (anchored core) | Restricting alignment to the robust in-model edge core unlocks transferable non-core structure. | Define high-confidence `H91/H93` core, learn CCA/Procrustes mapping on core features, and evaluate concordance on held-out non-core edges. | Held-out non-core concordance null-gap positive in `>=2/3` domains. | Core-membership shuffle (size-matched), random mapping baseline, depth scramble. | medium-high | medium |
| `N504` | Biological anchoring (TRRUST sign) | Separate activation/repression temperature calibration on weighted filtration improves weak-slice robustness without harming strong slices. | Fit sign-specific calibration temperatures for `H93` features with nested CV across domains/splits/layers and evaluate weak-slice lift. | At least one weak slice flips to positive mean null-gap while existing strong slices remain positive. | Sign-label permutation, shared-temperature baseline, label permutation. | high | low-medium |
| `N505` | Biological anchoring (Cell Ontology + STRING) | Hierarchical Cell Ontology partial pooling with STRING-informed priors generalizes better than flat additive stratification. | Fit hierarchical logistic calibration where ontology-stratum offsets are shrinkage-regularized by STRING confidence; compare with global and flat models. | Positive mean null-gap in `>=4/6` domain-splits and reduced between-domain variance versus flat stratification. | Ontology-label permutation (size-preserving), shuffled STRING priors, label permutation. | high | medium-high |
| `N506` | Algorithmic signatures (depth motif grammar) | Positive edges follow reproducible depth-transition motifs in descriptor state space that negatives rarely express. | Discretize robust descriptors across layers into motif tokens, estimate Markov transition likelihood ratios, and test predictive lift. | Motif-likelihood enrichment plus positive mean null-gap in `>=3/6` domain-splits. | Layer-order permutation with marginal preservation, token shuffle within layer, label permutation. | medium | low |
| `N507` | Mechanistic motif validation (counterfactual surgery) | Counterfactual perturbation of top descriptors induces monotonic probability shifts for positives but weak/non-monotonic shifts for negatives. | Run matched-edge descriptor interventions (ablation/amplification) and quantify monotonic dose-response slope differences across domains/splits. | Positive-vs-negative monotonic slope gap remains positive under controls. | Random descriptor surgery, unmatched swap controls, label permutation. | medium | low-medium |

## Top 3 for Immediate Execution
1. High-probability discovery candidate: `N493` (relative persistence contrast on `H93/H91` backbone).
Why: closest to proven robust signal while adding a genuinely new endpoint class (relative topology, not additive blend).
Keep gate: positive mean delta in `>=5/6` domain-splits and positive mean null-gap in `>=4/6`.
Fast-fail: retire this exact formulation if positive mean null-gap is `<=1/6` in pilot.

2. High-risk/high-reward candidate: `N501` (cross-model OT + monotone depth warping on module persistence images).
Why: strongest remaining structural-reset path to reopen cross-model branch.
Keep gate: positive primary-metric null-gap in `>=2/3` domains.
Fast-fail: immediate re-retire cross-model slot if `0/3` domains clear null-gap.

3. Cheap broad-screen candidate: `N497` (lifetime derivative spectrum).
Why: low engineering cost, broad layer/domain coverage, and orthogonal to retired additive ID/topology-stability forms.
Keep gate: positive mean delta in `>=4/6` domain-splits and positive mean null-gap in `>=2/6`.
Fast-fail: stop if positive mean delta is `<3/6` domain-splits.
