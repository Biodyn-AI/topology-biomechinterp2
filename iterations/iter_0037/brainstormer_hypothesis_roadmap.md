# Brainstormer Hypothesis Roadmap - iter_0037

## Retire / Deprioritize
1. `cross_model_alignment` endpoint lineage (`H71/H74/H77/H80/H83/H86/H96`) -> `retire_now`.
Reason: repeated null-gap failures after multiple objective resets.
Reopen only if: one structural-reset objective with module-level invariants and strict `0/3` fast-fail.

2. GO-overlap additive stratification refinement (`H94` form) -> `retire_now`.
Reason: universal underperformance against global weighted filtration and `0/6` null-gap support.

3. Standalone additive topology-stability/trajectory utility forms (`H72/H90/H92`) -> `retire_now`.
Reason: repeated small directional effects with robustness collapse.

4. Standalone additive intrinsic-dimension/phase-boundary utility forms (`H54/H60/H63/H66/H89`) -> `retire_now`.
Reason: no robust signal after repeated retries.

5. Global pooled interaction overlays (`H73/H76/H79`) -> `rescue_once_with_major_change`.
Required change: local/hierarchical conditioning (ontology or module strata), not pooled global coefficients.

6. Bridge-curvature blend as currently null-tested (`H95` form) -> `rescue_once_with_major_change`.
Required change: structure-matched nulls (degree + edge-length-bin + bridge-rate preservation) with higher null resolution.

## New Hypothesis Portfolio
| ID | Category | One-sentence hypothesis | Concrete test design | Expected signal if true | Null/control | Value | Cost |
|---|---|---|---|---|---|---|---|
| `N478` | Topology (relative PH) | Relative homology between high-confidence subcomplex and full complex captures regulatory signal missed by absolute barcodes. | Build high-confidence subcomplexes per row, compute relative Betti/lifetime descriptors, and add to `H91/H93` baseline across domains/splits/seeds. | Positive mean `delta_AUROC` in `>=4/6` domain-splits with positive mean null-gap in `>=3/6`. | Confidence-label permutation (size-preserving), subcomplex-size matched random subsets, label permutation. | high | medium |
| `N479` | Topology (fibered bifiltration) | Fibered bifiltration over geodesic distance x signed-support asymmetry recovers robust local edge signatures. | Compute fibered barcodes over angle grid, summarize persistence surfaces, and test additive utility over `H70/H91`. | Positive mean delta in `>=4/6` and positive mean null-gap in `>=3/6` domain-splits. | Axis-swap, support-sign shuffle within degree bins, label permutation. | high | high |
| `N480` | Topology (critical-event typing) | True edges are enriched for shortcut-closure death events rather than collapse deaths during filtration. | Label local PH events into closure/collapse classes, derive event frequency ratios, and evaluate predictive lift. | Closure-event ratio significantly higher for positives and positive null-gap in `>=3/6` domain-splits. | Event-label permutation within scale bins, matched-random neighborhoods, label permutation. | medium | medium |
| `N481` | Topology (landmark witness stability) | Landmark-witness persistence envelopes are more stable and transferable than full-neighborhood PH features. | Sample landmark sets repeatedly, compute witness-complex descriptors, and aggregate mean/variance features into classifier. | Utility lift with lower bootstrap variance and positive null-gap in `>=3/6` domain-splits. | Landmark-index permutation, witness-neighborhood shuffle, label permutation. | medium | medium |
| `N482` | Graph topology (H95 rescue) | Bridge-curvature gain is real but masked by an over-permissive null that ignores edge-length and bridge-rate structure. | Re-run `H95` with nulls preserving degree sequence, edge-length bins, and bridge-rate strata; increase null draws to `>=32` per row. | Mean null-gap becomes positive in `>=3/6` domain-splits while keeping positive mean delta in `6/6`. | Existing degree-preserving swap as baseline control, descriptor shuffle, label permutation. | high | low-medium |
| `N483` | Manifold geometry (geodesic corridors) | True edges lie on high-multiplicity near-shortest geodesic corridors with low corridor entropy. | Compute k-shortest-path multiplicity, corridor entropy, and perturbation sensitivity; test incremental lift over `H91`. | Positive mean delta in `>=4/6` domain-splits, strongest at layers `7/11`. | Endpoint-matched random pairs, path-order shuffle, label permutation. | medium | medium |
| `N484` | Manifold geometry (anisotropic curvature) | Directional curvature anisotropy around edge neighborhoods is predictive even when scalar curvature means fail. | Estimate directional curvature proxies (multiple directions/scales), derive anisotropy ratios, and add to baseline models. | Positive mean null-gap in `>=2/6` domain-splits with split-consistent effect direction. | Direction randomization, scale-order permutation, label permutation. | medium | medium |
| `N485` | Manifold x topology interaction | Local linearity residual contributes only when weighted persistence is high, not as a standalone additive signal. | Fit interaction terms between reconstruction-error quantiles and `H93` weighted-persistence bins in cross-validated models. | Interaction coefficient remains positive under controls and yields positive null-gap in `>=3/6` domain-splits. | Interaction-term permutation within bins, residual shuffle, label permutation. | high | medium |
| `N486` | Intrinsic dimension (heterogeneity) | Multi-radius ID heterogeneity entropy, rather than mean ID, discriminates positives from negatives. | Compute neighborhood ID at multiple radii, derive entropy/dispersion features, and run seed42 breadth screen across all domains/splits/layers. | Positive mean delta in `>=4/6` domain-splits with positive mean null-gap in `>=2/6`. | Radius-order permutation, neighborhood assignment shuffle, label permutation. | medium | low |
| `N487` | Cross-model structural alignment | Cross-model agreement emerges in module role-graph topology, not raw edge or module-score concordance. | Build per-module role graphs from persistence descriptors in each model, align transition matrices with Procrustes/CCA, and test domain-wise concordance. | Positive primary-metric null-gap in `>=2/3` domains. | Module-membership permutation (size-preserving), role-label shuffle, depth-order permutation. | high | high |
| `N488` | Cross-model depth alignment | Cross-model module persistence trajectories align after learned monotone depth warping. | Compute module trajectory descriptors by depth/rank-strata, fit monotone warping on source domains, evaluate held-out concordance. | Held-out concordance null-gap positive in `>=2/3` domains. | Depth-order permutation, warping-function randomization, module-label permutation. | high | medium-high |
| `N489` | Biological anchoring (cell ontology) | Hierarchical Cell Ontology conditioning can improve `H93` robustness without the overfitting seen in direct GO additive stratification. | Fit hierarchical shrinkage weights per ontology stratum (shared prior + stratum offsets) and compare against global `H93`. | Positive mean null-gap in `>=5/6` domain-splits and reduced stratum-variance vs `H93` global weights. | Ontology-label permutation (size-preserving), random-shrinkage baseline, label permutation. | high | medium |
| `N490` | Biological anchoring (confidence calibration) | Weighted filtration has a domain-stable confidence temperature optimum that lifts weak slices without harming strong ones. | Sweep confidence/sign temperature `tau`, fit calibration curves per domain/split/layer, and select robust `tau` by nested CV. | At least one weak slice flips to positive null-gap and no strong slice loses sign. | Confidence shuffle within degree bins, random-`tau` baseline, label permutation. | medium-high | low |
| `N491` | Algorithmic signatures | Positive edges follow reproducible descriptor-state transition motifs across depth that are absent in negatives. | Discretize key descriptors over layers `{0,3,7,11}`, mine transition motifs (Markov features), and test utility/enrichment. | Motif enrichment plus positive null-gap in `>=3/6` domain-splits. | Sequence-order permutation preserving layer marginals, motif-token shuffle, label permutation. | medium | medium |
| `N492` | Mechanistic motif validation | Counterfactual descriptor surgery should induce monotonic probability shifts for positives but not negatives. | Perform matched-edge descriptor swaps/ablations on top features, then measure monotonic dose-response of predicted probabilities. | Positive class shows steeper monotonic slopes and stronger intervention effect across domains/splits. | Random surgery, unmatched swap controls, label permutation. | medium | low-medium |

## Top 3 for Immediate Execution
1. High-probability discovery candidate: `N482` (H95 rescue with structure-matched high-resolution nulls).
Why: existing signal is strong (`24/24` positive rows) and likely under-resolved under current null design.
Keep gate: positive mean delta in `6/6` domain-splits and positive mean null-gap in `>=3/6`.
Fast-fail: stop after pilot if positive mean null-gap remains `0/6`.

2. High-risk/high-reward candidate: `N487` (cross-model module role-graph alignment).
Why: this is a true structural reset for a stale branch and can either reopen or cleanly retire cross-model work.
Keep gate: positive primary null-gap in `>=2/3` domains.
Fast-fail: retire immediately if `0/3` domains clear null-gap.

3. Cheap broad-screen candidate: `N486` (multi-radius ID heterogeneity entropy).
Why: low implementation cost, broad coverage, and materially different from previously retired additive ID forms.
Keep gate: positive mean delta in `>=4/6` domain-splits and positive mean null-gap in `>=2/6`.
