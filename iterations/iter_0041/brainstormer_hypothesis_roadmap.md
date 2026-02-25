# Brainstormer Hypothesis Roadmap - iter_0041

## Retire / Deprioritize
1. Additive weighted-filtration rescue lineage on `H93` (`H94/H100/H103/H106`) -> `retire_now`.
Evidence: four consecutive formulations with `0/6` positive mean null-gap domain-splits.

2. Static cross-model concordance lineage (module topology rank, role-graph concordance, OT depth warp; `H96/H99/H102` and related static endpoints) -> `retire_now`.
Evidence: repeated domain-level robustness failure (`0/3` positive null-gap domains).

3. Conditioned-null-as-rescue objective (`H105`) -> `retire_now`.
Evidence: calibration tightening without rescue gain in any domain-split.

4. Standalone intrinsic-dimension additive utility lineage (`H89/H98` plus earlier `H60/H63`) -> `retire_now`.
Evidence: persistent negative tail with no robust promotion signal.

5. Coarse tokenized motif grammar lineage (`H104/H107`) -> `rescue_once_with_major_change`.
Required major change: biologically anchored state construction and occupancy-matched nulls.

6. `H108` immune failure mode -> `rescue_once_with_major_change`.
Required major change: immune-specific perturbation balancing + module-size/variance matched null package; no global method pivot yet.

## New Hypothesis Portfolio
| ID | Category | One-sentence hypothesis | Concrete test design | Expected signal if true | Null / control | Value | Cost |
|---|---|---|---|---|---|---|---|
| `N539` | topology (PH variant) | Regulatory edges have smoother and more model-concordant persistence vineyards under graded perturbations than non-edges. | For each edge neighborhood, run perturbation strengths `{0,0.25,0.5,0.75,1.0}` in both models, compute vineyard trajectory descriptors (lifetime slope, curvature, crossing count), and add to `H93/H108` baselines. | Positive edges show lower vineyard roughness and higher cross-model vineyard correlation; positive mean null-gap in `>=3/6` in-model domain-splits. | Perturbation-schedule permutation, degree-preserving local rewiring, label permutation. | high | high |
| `N540` | topology (stability) | Positive edges exhibit lower bootstrap Wasserstein variance of persistence images across split-resampled manifolds. | Bootstrap cells per domain/split/layer, compute persistence images and Wasserstein variance per edge-local complex, then test incremental AUROC over `H93`. | Lower variance for positives and positive mean null-gap in `>=3/6` domain-splits. | Within-layer feature shuffle, bootstrap-label permutation, split-swap control. | medium | medium |
| `N541` | topology (filtration variant) | Signed-confidence bifiltration (direction margin x biological support) captures robust signal missed by one-axis filtrations. | Build 2-axis filtration using directed margin and support score; summarize with rank-invariant slices/Hilbert signatures and evaluate against `H93`. | Positive mean null-gap in `>=3/6` domain-splits with improved source/target symmetry vs additive rescues. | Margin-sign shuffle within degree bins, support-bin permutation, label permutation. | high | high |
| `N542` | manifold geometry (curvature) | Endpoint-neighborhood curvature anisotropy is systematically different for true regulatory edges. | Estimate discrete curvature tensors (Forman/Ollivier proxies) on endpoint neighborhoods at radii `{1,2,3}`; use anisotropy ratios and interactions with geodesic baseline. | Stable anisotropy direction across domains and positive mean null-gap in `>=3/6` domain-splits. | Neighborhood membership shuffle (degree-matched), radius-order permutation, label permutation. | medium | medium |
| `N543` | manifold geometry (geodesic sensitivity) | True edges lie on geodesics with lower Jacobi-like divergence under local perturbation. | Perform geodesic shooting from source to target with small neighborhood perturbations; compute endpoint spread and path instability features. | Positives have lower geodesic divergence and positive mean null-gap in `>=3/6` domain-splits. | Endpoint-matched random pair control, perturbation-direction shuffle, label permutation. | high | high |
| `N544` | manifold geometry (intrinsic dimension) | Positive edges traverse a characteristic intrinsic-dimension valley (drop then recovery) along source-to-target paths. | Estimate pathwise ID profile with TWO-NN/local PCA along shortest geodesic paths and derive valley depth/location features. | Positive edges show stronger valley depth and consistent valley position; positive mean null-gap in `>=2/6` domain-splits. | Path-order permutation, endpoint swap within distance bins, label permutation. | medium | low |
| `N545` | manifold geometry (local linearity) | True edges preserve local tangent-subspace continuity across depth transitions better than negatives. | Compute local tangent spaces at layers `{0,3,7,11}`, derive cumulative principal-angle drift per edge, and test incremental utility over `H70`. | Lower cumulative principal-angle drift for positives and positive mean null-gap in `>=3/6`. | Layer-order permutation, neighborhood shuffle, label permutation. | medium | medium |
| `N546` | cross-model structure transfer | Cross-model perturbation Jacobian subspaces align after variance-normalized module scaling, including immune. | Extend `H108` by estimating module-response Jacobians for perturbation panels, compare canonical angles + rank concordance across models with immune-specific scaling. | Domain-level null-gap positive in `>=2/3` domains with immune improving from negative to near-zero/positive. | Perturbation-schedule permutation, module-size/variance matched shuffle, random gene mapping. | high | medium |
| `N547` | cross-model alignment | Alignment learned on two domains transfers zero-shot to the third domain only when constrained by GO hierarchy. | Train alignment on lung+external_lung, evaluate immune zero-shot perturbation-response concordance; repeat leave-one-domain-out. | Zero-shot target domain has positive null-gap in at least one leave-one-domain-out test; GO-constrained beats unconstrained. | Domain-label permutation, GO-tree rewiring (depth-preserving), random-map baseline. | high | high |
| `N548` | biological anchoring (TRRUST/STRING) | TRRUST signed feed-forward/feedback motifs correspond to high-persistence local cycles around true edges. | Map edge endpoints to TRRUST motifs + STRING support; compute motif-conditioned cycle persistence descriptors and interaction terms with `H93`. | Positive motif-interaction coefficient and positive mean null-gap in `>=3/6` domain-splits. | Motif sign permutation, motif membership shuffle preserving degree/support bins, label permutation. | high | medium |
| `N549` | biological anchoring (cell ontology) | Cell-ontology partial pooling stabilizes coefficients and rescues weak domain-splits without inflating false positives. | Fit hierarchical logistic model with cell-ontology random effects over topological/geometric descriptors; compare weakest-split null-gap vs flat model. | Worst-split null-gap improves by `>=0.01` while keeping strong-split performance stable. | Ontology-label permutation (size-matched), flat-model baseline, label permutation. | medium | medium |
| `N550` | biological anchoring (GO/STRING) | GO parent-child boundary genes carry boundary-topology signatures that enrich regulatory-edge prediction. | Construct parent-child boundary scores and boundary-conditioned PH descriptors; test incremental utility over `H93`. | Boundary-conditioned descriptors show positive interaction and positive mean null-gap in `>=3/6`. | GO hierarchy-edge rewiring preserving depth/size, boundary-score shuffle, label permutation. | medium | medium |
| `N551` | algorithmic signatures / mechanistic motifs | A biologically anchored finite-state grammar (TF activity x support x sign states) outperforms coarse quantile tokens and survives null calibration. | Replace `H107` tokenization with biologically anchored states, train second-order Markov/DFA likelihood-ratio features, and evaluate over all domains/splits. | Positive mean delta in `>=4/6` domain-splits and positive mean null-gap in `>=2/6` domain-splits. | State-frequency matched token shuffle, layer-order permutation, label permutation. | high | low |
| `N552` | algorithmic signatures / mechanism | True edges are enriched for stable feedback-dominance signatures in local linearized dynamics across depth. | Estimate local linear dynamics between layers, compute feedback-dominance/loop-gain proxies near endpoints, and test additive + interaction utility. | Feedback-dominance index is higher for positives and yields positive mean null-gap in `>=3/6` domain-splits. | Spectrum-preserving random dynamics control, endpoint swap, label permutation. | medium | medium |

## Top 3 for Immediate Execution
1. High-probability discovery candidate: `N546` (cross-model perturbation Jacobian alignment with immune normalization).
Why: direct continuation of `H108` which already passed `2/3` domains; changes are targeted at the observed immune failure mode, not a full reset.
Pilot design: seeds `{42,43,44}`, 3 domains, same perturbation families as `H108` plus variance-normalized Jacobian features.
Keep gate: positive domain-level null-gap in `>=2/3` domains for at least `2/3` seeds.
Fail-fast: retire this rescue if immune remains negative in all seeds.

2. High-risk/high-reward candidate: `N539` (perturbation persistence vineyards).
Why: introduces a genuinely new topological object (trajectory of barcodes under perturbation intensity) that could reveal mechanism-level structure absent in static PH summaries.
Pilot design: seed42 first-pass on layers `{7,11}` and both split regimes, then expand only if gate clears.
Keep gate: positive mean null-gap in `>=3/6` domain-splits.
Fail-fast: if positive mean null-gap is `0/6`, retire immediately.

3. Cheap broad-screen candidate: `N551` (biologically anchored finite-state grammar).
Why: low engineering cost and directly addresses why `H107` was directional but non-robust (coarse state alphabet likely under-specifies biology).
Pilot design: seed42 across all 3 domains and both split regimes using existing descriptor traces.
Keep gate: positive mean delta in `>=4/6` and positive mean null-gap in `>=2/6`.
Fail-fast: if positive mean delta `<3/6`, stop lineage.
