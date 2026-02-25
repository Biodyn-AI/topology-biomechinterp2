# Retire / Deprioritize

1. Cross-model edge-transfer and edge-rank endpoint family (`H68/H71/H74/H77/H80/H83/H86`) -> `retire_now`.
Reason: repeated null-gap collapse across endpoint redesigns.

2. Standalone additive intrinsic-dimension/local-linearity utility forms (`H54/H60/H63/H66/H89`) -> `retire_now`.
Reason: directional effects repeatedly fail robustness.

3. Standalone additive topology-stability/trajectory forms (`H72/H90/H92`) -> `retire_now`.
Reason: small deltas with repeated `0/6` null-gap support.

4. Pooled global biological interaction overlays (`H73/H76/H79`) -> `rescue_once_with_major_change`.
Required change: local or ontology-stratified interaction modeling.

5. Fixed-threshold dual-filtration witness refinement (`H85`) -> `rescue_once_with_major_change`.
Required change: adaptive threshold calibration with uncertainty weighting.

# New Hypothesis Portfolio

| ID | Category | One-sentence hypothesis | Concrete test design | Expected signal if true | Null/control | Value | Cost |
|---|---|---|---|---|---|---|---|
| `N463` | Topology (multiparameter PH) | Joint geodesic-distance x confidence filtration captures true-edge topology missed by one-axis barcodes. | Compute fibered barcodes over an angle grid per edge neighborhood and add persistence-surface descriptors on top of `H91`. | Positive mean delta and positive mean null-gap in `>=4/6` domain-splits. | Confidence-bin shuffle, angle-order shuffle, label permutation. | high | high |
| `N464` | Topology (directed PH) | Signed directed local complexes encode regulatory directionality better than undirected descriptors. | Build signed directed flag complexes (from support sign/direction), extract extended-persistence features, and compare vs `H93`/`H91`. | `>=0.01` AUROC gain over undirected variant in `>=4/6` domain-splits with positive null-gap. | Sign-flip, orientation randomization (degree-preserving), label permutation. | high | medium-high |
| `N465` | Topology (zigzag across depth) | True edges show stable zigzag persistence intervals across layer transitions that false edges do not. | Compute zigzag PH across layers `{0,3,7,11}` for edge-local neighborhoods and score interval-length/turnover features. | Positive null-gap in `>=3/6` domain-splits, strongest on late-layer transitions. | Layer-order permutation, inter-layer edge shuffle, label permutation. | medium-high | high |
| `N466` | Topology (relative homology) | Relative topology between high-confidence subcomplexes and full complexes isolates biologically meaningful signal. | Define confidence-thresholded subcomplexes and compute relative Betti/lifetime deltas as add-ons to `H93`. | Improvement concentrated in currently weak slices (e.g., immune target L11, lung source L7) with positive null-gap. | Threshold randomization (size-preserving), confidence shuffle, label permutation. | high | medium |
| `N467` | Topology (entropy spectrum) | Persistence-entropy slope is a scale-robust signal even when raw lifetime trajectory additives fail. | Reuse `H92` scale pipeline but replace trajectory features with entropy/slope summaries and test over `H70`/`H91`. | Positive mean delta in `>=4/6` and positive mean null-gap in `>=2/6` domain-splits. | Scale-order permutation, entropy-bin shuffle, label permutation. | medium | low |
| `N468` | Manifold geometry (curvature anisotropy) | Curvature anisotropy around edges is informative even when scalar curvature averages are weak. | Estimate directional curvature proxies around each edge and add anisotropy ratios as features. | Positive null-gap in `>=3/6` domain-splits with strongest effect in target-disjoint. | Neighbor reassignment with degree matching, direction shuffle, label permutation. | medium | medium |
| `N469` | Manifold geometry (geodesic corridors) | True edges lie on high-multiplicity near-shortest geodesic corridors with low perturbation variance. | Compute k-shortest path multiplicity plus noise-stability of path sets and blend with `H91`. | Small but consistent uplift and positive null-gap in `>=2/6` domain-splits. | Endpoint-matched random pairs, path-order shuffle, label permutation. | medium | medium |
| `N470` | Manifold geometry x topology interaction | Local linearity residual is predictive only when coupled with high weighted-persistence context. | Fit interaction terms between local reconstruction error quantiles and `H93` persistence bins (no standalone ID additive). | Positive interaction effect with positive null-gap in `>=3/6` domain-splits. | Interaction-term permutation within bins, residual shuffle, label permutation. | high | medium |
| `N471` | Manifold geometry (ID heterogeneity) | Neighborhood ID heterogeneity entropy (not mean ID) separates true from false edges. | Estimate ID over multiple radii and use entropy/dispersion features as add-ons to `H91`. | Lift concentrated at layers `7/11` with positive null-gap in `>=2/6` domain-splits. | Radius-order permutation, neighborhood-label shuffle, label permutation. | medium | low |
| `N472` | Cross-model structural alignment | Cross-model agreement appears in module-level barcode rank order rather than edge-transfer utility. | Compute TRRUST/GO module barcode quantiles for scGPT and Geneformer, then evaluate depth-wise rank concordance per domain. | Positive null-gap in `>=2/3` domains for rank-concordance metrics. | Module-membership permutation (size-preserving), depth-order shuffle, quantile shuffle. | high | medium-high |
| `N473` | Cross-model structural alignment | A low-dimensional shared latent of module persistence images can align models if cycle consistency holds. | Learn CCA/Procrustes latent on source domains and test held-out domain retrieval + cycle reconstruction error. | Held-out concordance and retrieval exceed nulls in `>=2/3` domains. | Random subspace baseline, shuffled correspondences, depth permutation. | high | high |
| `N474` | Biological anchoring (cell ontology) | Ontology-stratified confidence/sign weighting will amplify `H93` and reduce slice fragility. | Stratify edges by Cell Ontology module tags, tune per-stratum weight temperature, and meta-analyze vs global `H93`. | Higher mean null-gap and lower between-strata variance vs global weights. | Ontology-label permutation (size-preserving), random temperature baseline, label permutation. | high | medium |
| `N475` | Biological anchoring (confidence calibration) | Robustness has a temperature optimum in confidence weighting that can be found by calibration sweep. | Sweep confidence temperature `tau` for `H93` filtration and track delta/null-gap/reliability across domain-splits. | Broad robust optimum that improves weak rows without harming strong rows. | Confidence shuffle within degree bins, random `tau` baseline, label permutation. | medium-high | low |
| `N476` | Algorithmic signatures (motifs) | True edges follow reproducible descriptor-state transition motifs across depths. | Discretize top `H91/H93` descriptors into states across `{0,3,7,11}`, mine motifs, and test motif-feature utility. | Motif enrichment plus positive mean null-gap in `>=3/6` domain-splits. | Sequence-order permutation (per-layer marginals preserved), motif-token shuffle, label permutation. | high | medium |
| `N477` | Mechanistic signature validation | Causal descriptor surgery should induce monotonic probability shifts for positives but not negatives. | Perform matched-edge descriptor swap interventions and measure monotonic dose-response slopes. | Significantly steeper monotonic slopes for positives across domains/splits. | Random descriptor surgery, unmatched swap control, label permutation. | medium | low-medium |

# Top 3 for Immediate Execution

1. High-probability discovery candidate: `N474` (ontology-stratified weighted filtration on top of `H93`).
Why: `H93` is already strong; this is the most likely near-term upgrade with biological interpretability and direct robustness gain.
Execution gate: positive mean delta in `6/6`, positive mean null-gap in `>=5/6`, and reduced between-strata variance vs global `H93`.

2. High-risk/high-reward candidate: `N473` (cross-model shared latent for module persistence images).
Why: this is a real structural reset for a stale family; success would reopen cross-model claims with a defensible invariant objective.
Execution gate: positive primary-metric null-gap in `>=2/3` domains.
Fail-fast: if `0/3` domains positive in pilot, retire this exact formulation immediately.

3. Cheap broad-screen candidate: `N467` (persistence-entropy slope spectrum).
Why: very low implementation overhead using existing `H92` machinery but materially different signal representation.
Execution gate: positive mean delta in `>=4/6` domain-splits and positive mean null-gap in `>=2/6`.
