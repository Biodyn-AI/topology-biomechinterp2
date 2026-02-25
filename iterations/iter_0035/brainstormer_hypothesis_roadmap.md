# Retire / Deprioritize

1. `cross_model_alignment` global transfer/order utility endpoints (`H71/H74/H77/H80/H83/H86` lineage) -> `retire_now`.
Reason: repeated null-gap failure despite endpoint changes.
Reopen rule: only one major reset using module-level topological invariants with strict in-model positive-control preflight.

2. Standalone additive intrinsic-dimension and phase-boundary utility forms (`H54/H60/H63/H66/H89`) -> `retire_now`.
Reason: repeated directional-only effects that collapse under nulls.
Reopen rule: diagnostics/interaction-only on top of an already-positive topology backbone.

3. Standalone additive perturbation-stability utility forms (`H90` and related elasticity-style add-ons) -> `retire_now`.
Reason: weak lift and universal null-gap failure.
Reopen rule: use perturbations as stress-test strata or uncertainty estimates, not direct predictive features.

4. Fixed-threshold dual-filtration witness refinement (`H85` form) -> `rescue_once_with_major_change`.
Reason: near-miss signal but unstable robustness.
Required change: adaptive filtration calibration by domain/split/layer with uncertainty-aware weighting.

5. Global pooled support-interaction overlays (`H73/H76/H79` forms) -> `rescue_once_with_major_change`.
Reason: utility direction sometimes positive, but interaction robustness repeatedly fails.
Required change: local/stratified biological conditioning (cell ontology/module strata) instead of global pooled interaction terms.

# New Hypothesis Portfolio

1. `N449` `[topology + robustness]`
Hypothesis: stability-selection on sparse descriptors will convert the weak `immune/source_disjoint` H88 slice into a robust positive while increasing cross-seed descriptor-core overlap.
Test design: rerun H88 in all domains/splits/layers with bootstrap stability selection (subsampled folds + seed consensus), then compare utility and core Jaccard vs current H88.
Expected signal if true: positive mean null-gap in all `6/6` domain-splits and descriptor-core `Jaccard >= 0.65`.
Null/control: descriptor shuffle within geodesic bins, endpoint swap, label permutation, and random-feature-subset control.
Value: `high`; Cost: `medium`.

2. `N450` `[topology: multiparameter PH]`
Hypothesis: two-parameter persistence summaries (geodesic distance x directed-support margin) capture predictive structure that one-axis descriptors miss.
Test design: compute local rank-invariant summaries or Hilbert-surface descriptors per edge neighborhood and add them to H70/H88 backbones.
Expected signal if true: positive mean delta and positive mean null-gap in `>=4/6` domain-splits.
Null/control: axis-swap null, support-margin shuffle within bins, label permutation.
Value: `high`; Cost: `high`.

3. `N451` `[topology: directed persistence]`
Hypothesis: directed/signed extended-persistence descriptors of local support-flow complexes outperform undirected local-cycle descriptors.
Test design: build directed local complexes from support asymmetry, extract directed barcode features, and test incremental AUROC over H70.
Expected signal if true: strongest gains in source-disjoint slices and positive null-gap in `>=4/6` domain-splits.
Null/control: direction randomization, sign-flip permutation, endpoint swap within geodesic bins.
Value: `high`; Cost: `medium-high`.

4. `N452` `[topology: scale-space]`
Hypothesis: slope/curvature of local H1 lifetime across filtration scale is more robust than raw lifetime magnitude.
Test design: compute per-edge lifetime trajectories across k/radius scales and use trajectory-shape features (slope, curvature, area) on top of H70.
Expected signal if true: positive mean delta in `>=4/6` domain-splits with better null-gap than fixed-scale features.
Null/control: scale-order permutation, trajectory-value shuffle within bins, label permutation.
Value: `medium`; Cost: `low`.

5. `N453` `[manifold geometry: curvature anisotropy]`
Hypothesis: curvature anisotropy (directional spread) is informative even when scalar curvature averages were negative.
Test design: estimate multi-direction local curvature proxies around each edge and test anisotropy ratios as additive features.
Expected signal if true: positive uplift in at least two domains with non-negative mean null-gap.
Null/control: direction-label randomization, neighborhood rotation shuffle, label permutation.
Value: `medium`; Cost: `medium`.

6. `N454` `[manifold geometry: geodesic corridor]`
Hypothesis: positives lie in low-divergence geodesic corridors with high near-shortest-path multiplicity.
Test design: compute k-shortest path degeneracy, detour growth under small endpoint perturbation, and corridor dispersion features.
Expected signal if true: positive edges show lower divergence and utility lift over geodesic baseline, especially at layers `7/11`.
Null/control: endpoint-matched random path controls, path-order shuffles, label permutation.
Value: `medium`; Cost: `medium`.

7. `N455` `[manifold + intrinsic dimension]`
Hypothesis: directional local-ID shear along geodesic tangent fields is predictive even though absolute ID-jump additives failed.
Test design: estimate local tangent frames and directional ID change vectors across depths, then score edge-aligned shear features.
Expected signal if true: positive mean delta in `>=3/6` domain-splits with at least `>=2/6` positive mean null-gap.
Null/control: tangent-direction randomization, depth-order permutation, label permutation.
Value: `medium`; Cost: `medium`.

8. `N456` `[cross-model structural reset]`
Hypothesis: cross-model agreement appears at module-level persistence-image trajectories even when direct transfer endpoints fail.
Test design: compute TRRUST/GO module persistence images by depth for each model, align with CCA/Procrustes in module space, and evaluate held-out module retrieval/concordance.
Expected signal if true: positive null-gap in `>=2/3` domains for trajectory-concordance metrics.
Null/control: module-membership permutation (size-preserving), depth-order permutation, random subspace alignment baseline.
Value: `high`; Cost: `high`.

9. `N457` `[cross-model structural reset]`
Hypothesis: relative depth-order of topological event quantiles is conserved cross-model within biological modules.
Test design: for each module, compare ordered birth/death quantile curves across depths between models and score rank concordance.
Expected signal if true: positive depth-order concordance null-gap in at least two domains.
Null/control: depth-order permutation, module-label shuffle, event-quantile shuffle.
Value: `medium`; Cost: `low-medium`.

10. `N458` `[biological anchoring]`
Hypothesis: confidence-and-sign weighted filtrations (STRING confidence + DoRothEA direction) suppress spurious cycles and rescue weak H88 slices.
Test design: modify edge birth times with confidence/sign weights, recompute local descriptors, and compare against unweighted H88 in failing slices first.
Expected signal if true: `immune/source_disjoint` mean null-gap turns positive and at least one additional weak slice improves.
Null/control: confidence shuffle within degree/support bins, sign-flip control, label permutation.
Value: `high`; Cost: `low-medium`.

11. `N459` `[biological anchoring]`
Hypothesis: moderate global descriptor stability is a mixture artifact and becomes high within cell-ontology strata.
Test design: stratify edges by Cell Ontology gene-set membership, rerun H88 descriptor-core stability per stratum, and meta-analyze with mixed effects.
Expected signal if true: within-stratum Jaccard exceeds pooled Jaccard by meaningful margin (target `+0.1` or more).
Null/control: ontology-label permutation preserving stratum sizes and degree bins.
Value: `high`; Cost: `low`.

12. `N460` `[algorithmic signatures]`
Hypothesis: positives follow a small set of reproducible descriptor-state motifs across layers that act as mechanistic signatures.
Test design: discretize key descriptors into states across layers `{0,3,7,11}`, fit motif-transition features (Markov/HMM), and test enrichment + utility.
Expected signal if true: a compact motif subset is enriched in positives across all domains with positive null-gap utility.
Null/control: sequence-order permutation preserving per-layer marginals, motif-token shuffle, label permutation.
Value: `medium`; Cost: `medium`.

13. `N461` `[mechanistic validation]`
Hypothesis: causal descriptor surgery on triangle-defect and support-margin components produces monotonic score shifts for true edges but not false edges.
Test design: perform local counterfactual edits by swapping descriptor values with matched neighbors and measure monotonicity of predicted probabilities.
Expected signal if true: significantly steeper monotonic response for positives in most domain-split slices.
Null/control: random-descriptor surgery, unmatched swap control, label permutation.
Value: `medium`; Cost: `low-medium`.

14. `N462` `[topology uncertainty]`
Hypothesis: uncertainty-aware weighting of topological descriptors (using bootstrap variance) improves robustness even when mean descriptors already lift AUROC.
Test design: bootstrap descriptor extraction per row, compute uncertainty features/weights, and refit blend models with and without uncertainty.
Expected signal if true: improved mean null-gap and reduced slice volatility versus baseline H88 blend.
Null/control: bootstrap-index permutation, random-weight baseline, label permutation.
Value: `medium`; Cost: `low`.

# Top 3 for Immediate Execution

1. High-probability discovery candidate: `N449` (stability-selected H88 rescue).
Why now: it directly upgrades the strongest live branch and targets the only failing domain-split.
Execution gate: positive mean delta in `6/6`, positive mean null-gap in `>=5/6`, and descriptor-core `Jaccard >= 0.65`.

2. High-risk/high-reward candidate: `N456` (cross-model module persistence-image trajectory alignment).
Why now: it is a true structural reset away from repeatedly-failing transfer endpoints and can retire cross-model work cleanly if it fails.
Execution gate: positive null-gap in `>=2/3` domains on primary trajectory-concordance metric.

3. Cheap broad-screen candidate: `N452` (scale-space lifetime trajectory features).
Why now: low implementation cost, strong topological novelty, and minimal dependence on fragile alignment assumptions.
Execution gate: positive mean delta in `>=4/6` domain-splits and positive mean null-gap in `>=2/6`.
