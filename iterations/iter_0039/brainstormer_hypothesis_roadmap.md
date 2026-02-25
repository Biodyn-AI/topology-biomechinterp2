# Brainstormer Hypothesis Roadmap - iter_0039

## Retire / Deprioritize
1. Cross-model unsupervised concordance/OT-depth family (`H65/H68/H71/H74/H77/H80/H83/H86/H96/H99/H102`) -> `retire_now`.
Reason: `11` consecutive negatives and repeated `0/3` domain-level null-gap support.

2. Exact relative-background persistent-homology construction (`H100`) -> `retire_now`.
Reason: under baseline and null-gap negative in all `6/6` domain-splits.

3. Additive derivative-spectrum form (`H101` as tested) -> `rescue_once_with_major_change`.
Required change: interaction-only terms with `H91/H93`; ban standalone additive derivative bundle.

4. Additive bridge-curvature utility lineage (`H95/H97`) -> `retire_now`.
Reason: repeated directional lift with persistent robustness failure (`0/6` positive mean null-gap domain-splits).

5. Standalone/additive intrinsic-dimension utility lineage (`H54/H60/H63/H66/H89/H98`) -> `retire_now`.
Reason: long negative tail with no robust rescue.

6. Flat pooled interaction overlays (`H73/H76/H79` style) -> `rescue_once_with_major_change`.
Required change: hierarchical partial pooling over GO/Cell Ontology/module strata.

7. Standalone additive topology-stability trajectories (`H90/H92`) -> `retire_now`.
Reason: low directional effect and repeated null collapse.

## New Hypothesis Portfolio
| ID | Category | One-sentence hypothesis | Concrete test design | Expected signal if true | Null/control | Value | Cost |
|---|---|---|---|---|---|---|---|
| `N508` | Topology (interaction rescue) | Persistence derivative-spectrum signal becomes robust only when coupled to `H93` weighted-filtration gain and `H91` stable descriptors. | Build interaction-only features (`derivative x H93_gain`, `derivative x stable_descriptor_core`) across domains/splits/layers `{0,3,7,11}`; compare against `H91+H93` backbone. | Positive mean null-gap in `>=3/6` domain-splits with strongest lift in lung. | Quantile-order permutation, interaction-partner shuffle within geodesic bins, label permutation. | high | low |
| `N509` | Topology (local homology) | True edges sit near local homology transition zones around edge midpoints rather than in uniformly connected neighborhoods. | For each edge, build midpoint-centered geodesic balls at multiple radii and extract local homology transition descriptors (rank/lifetime jumps); test incremental utility over `H93`. | Positive null-gap in `>=3/6` domain-splits and consistent positive transition-gap (pos vs neg). | Midpoint swap with length/degree matching, radius-order permutation, label permutation. | high | medium |
| `N510` | Topology (vineyard stability) | True edges have smoother persistence vineyards under kNN-scale perturbations than negatives. | Compute per-edge barcode summaries over k-scale sweeps and perturbations; derive vineyard path-length/variance metrics and test as interaction features with `H70`/`H93`. | Lower instability for positives and positive utility null-gap in `>=3/6` domain-splits. | Scale-order permutation, perturbation schedule randomization, endpoint swap, label permutation. | medium | medium |
| `N511` | Topology (directed bifiltration) | Direction-aware bifiltration (distance x signed support asymmetry) captures regulatory directionality missed by symmetric filtration. | Replace absolute margin axis with signed margin axis; compute source->target vs target->source contrastive connectivity features and evaluate over `H93`. | Positive directional contrast and positive null-gap in `>=4/6` domain-splits. | Sign-flip within TF bins, direction swap, label permutation. | high | medium |
| `N512` | Manifold geometry (corridor multiplicity) | Positive edges lie on low-entropy, high-redundancy geodesic corridors in the kNN graph. | Extract k-shortest-path multiplicity, corridor entropy, and detour volatility features for each edge; add interaction terms with triangle-defect baseline. | Positive edges show lower corridor entropy and higher multiplicity, with positive null-gap in `>=3/6` domain-splits. | Degree-preserving rewiring, endpoint-matched random pairs, path-order permutation. | medium | medium |
| `N513` | Manifold geometry (anisotropic curvature) | Local directional curvature asymmetry around edge endpoints is predictive even when scalar bridge-curvature failed. | Compute outgoing vs incoming neighborhood curvature proxies and anisotropy ratios around source/target; test contrastive features over `H91`. | Stable anisotropy sign across domains and positive null-gap in `>=3/6` domain-splits. | Neighborhood orientation randomization, degree-stratified node swap, label permutation. | medium | high |
| `N514` | Manifold geometry x ID interactions | Intrinsic-dimension gradients become useful only through interaction with robust persistence confidence, not as standalone features. | Estimate local ID (`2NN`/`MLE`) at endpoints and along geodesic corridor; fit interaction-only terms with `H93` weighted gain. | Positive interaction coefficients and positive null-gap in `>=3/6` domain-splits. | ID-profile shuffle within geodesic bins, interaction-term permutation, label permutation. | high | medium |
| `N515` | Cross-model alignment (anchored transfer) | Cross-model transfer is recoverable when mapping is learned on high-confidence in-model topology core and evaluated on held-out non-core modules. | Define core using top `H91/H93` confidence edges, fit CCA+Procrustes mapping on core module-role vectors, test concordance/utility on non-core modules only. | Positive domain null-gap in `>=2/3` domains on held-out non-core evaluation. | Core-membership shuffle (size-matched), random-anchor mapping, depth-order permutation, random-subspace control. | high | high |
| `N516` | Cross-model structure response | Models may disagree on static geometry but agree on relative module response ranks under controlled perturbations. | Apply matched perturbation library (module dropout/edge masking), compute per-model persistence-response ranks, and compare rank concordance across models. | Rank-response concordance clears null-gap in `>=2/3` domains while static concordance may remain weak. | Perturbation schedule permutation, module-label shuffle, random mapping baseline. | medium | high |
| `N517` | Biological anchoring (TRRUST sign calibration) | Activation/repression-specific temperature calibration of weighted filtration can recover weak slices without damaging strong ones. | Fit sign-specific calibration parameters in nested CV on `H93` features and evaluate weak-slice uplift plus strong-slice retention. | At least one currently weak slice flips to positive null-gap while no strong slice flips negative. | Sign-label permutation, shared-temperature baseline, label permutation. | high | low |
| `N518` | Biological anchoring (GO hierarchy contrasts) | Persistence signal is concentrated in ancestor-child GO contrast structure, not flat GO strata. | Build parent-child GO module pairs and use parent-minus-child persistence contrast features interacting with `H93`. | Positive null-gap in `>=3/6` domain-splits and reduced domain heterogeneity vs flat GO stratification. | GO hierarchy edge rewiring preserving depth, module-size matched random hierarchies, label permutation. | medium | medium |
| `N519` | Biological anchoring (STRING-conditioned controls) | Some apparent null failures are due to confidence heteroskedasticity and disappear under STRING-conditioned null calibration. | Compute conditional q95 nulls stratified by STRING decile + degree bins and compare to unconditioned q95 results. | Conditional null-gap improves in weak slices without inflating false positives in strong slices. | Unconditioned null baseline, within-decile label permutation. | medium | low |
| `N520` | Algorithmic signatures (depth motif grammar) | True edges follow a compact depth-transition grammar in descriptor state space across layers `{0,3,7,11}`. | Discretize `H91/H93` descriptor trajectories into tokens and score motif log-likelihood ratios as predictive features. | Positive motif enrichment and positive null-gap in `>=3/6` domain-splits. | Layer-order permutation with marginal preservation, token shuffle within layer, label permutation. | medium | low |
| `N521` | Mechanistic motif validation (counterfactual response) | Top stable descriptors are mechanistic if controlled interventions produce monotonic score shifts for positives but not negatives. | Run descriptor ablation/amplification sweeps on held-out edges and summarize monotonicity slope gaps. | Positive-vs-negative monotonicity gap remains positive under controls in all domains. | Random descriptor intervention (magnitude-matched), unmatched edge swaps, label permutation. | medium | medium |
| `N522` | Biological anchoring (Cell Ontology pooling) | Cell-ontology-aware partial pooling stabilizes cross-domain generalization of topology descriptors better than flat models. | Fit hierarchical model with Cell Ontology branch effects on `H91/H93` interaction features; compare to flat logistic baseline on source/target disjoint splits. | Lower between-domain variance and positive null-gap in `>=4/6` domain-splits. | Ontology-label permutation (size-preserving), flat-model baseline, label permutation. | medium | high |

## Top 3 for Immediate Execution
1. High-probability discovery candidate: `N508` (interaction-only derivative-spectrum rescue on `H91/H93`).
Why: `H101` showed directional lift and one positive null-gap row; interaction-only coupling addresses the additive failure mode directly.
Keep gate: positive mean null-gap in `>=3/6` domain-splits and positive mean delta in `>=4/6`.
Fast-fail: retire this rescue if positive mean null-gap is `<=1/6`.

2. High-risk/high-reward candidate: `N515` (anchored core-to-noncore cross-model transfer).
Why: cross-model branch only deserves one slot, and this is a fundamental objective change from raw concordance.
Keep gate: positive null-gap in `>=2/3` domains on held-out non-core modules.
Fast-fail: if `0/3` domains pass, re-retire cross-model branch for at least three loops.

3. Cheap broad-screen candidate: `N520` (depth motif grammar).
Why: low engineering cost, orthogonal to recent additive failures, and directly tests mechanistic structure.
Keep gate: positive mean delta in `>=4/6` domain-splits and positive mean null-gap in `>=2/6`.
Fast-fail: stop if positive mean delta is `<3/6`.
