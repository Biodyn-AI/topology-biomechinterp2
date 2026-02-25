# Brainstormer Structured Feedback - iter_0035

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0035/executor_research_validation.json`).
- Full 3-slot discovery planning is valid; no recovery-only restriction is required.

## Iteration Readout
- `H88` is the only promotable signal this iteration.
  - Mean `delta_auc_sparse_descriptor_blend_minus_h70 = +0.07603`.
  - Positive rows `72/72`; positive mean delta in `6/6` domain-splits.
  - Positive mean null-gap in `5/6` domain-splits.
  - Mechanistic stability is moderate, not high (`mean descriptor_nonzero_jaccard = 0.49263`; weakest slice `immune/source_disjoint` with mean null-gap `-0.00264`).
- `H89` is negative as a standalone additive manifold/ID utility formulation.
  - Directional lift exists (`mean delta = +0.01676`) but robustness fails (`0/6` positive mean null-gap domain-splits).
- `H90` is negative as a standalone additive stability utility formulation.
  - Lift is weak (`mean delta = +0.00449`) and robustness fails (`0/6` positive mean null-gap domain-splits).

## Cumulative Pattern (Paper + Master Log)
- Strongest live lineages are still topology-first and in-model:
  - `H70` triangle-defect branch,
  - `H82` local witness-cycle branch,
  - `H87/H88` sparse descriptor branch.
- Cross-model alignment/transfer has repeated null-gap collapse across multiple endpoint redesigns (`H71/H74/H77/H80/H83/H86` and earlier map-transfer failures).
- Standalone intrinsic-dimension additive utility forms repeatedly fail robustness (`H54/H60/H63/H66/H89`).
- Standalone perturbation-stability additive utility forms are not rescuing utility (`H90`, plus prior weak elasticity/dropout-style variants).

## Stale Direction Triage
- `cross_model_alignment` global transfer/order endpoints -> `retire_now`.
  - Failure sequence is long and consistent; next attempt must be a structural reset to within-model invariant comparison, not another direct transfer utility endpoint.
- Standalone additive intrinsic-dimension/phase-boundary utility forms -> `retire_now`.
  - Use ID only as conditional diagnostics or interaction terms on a positive backbone.
- Standalone additive perturbation-stability utility forms -> `retire_now`.
  - Keep perturbations for stress-testing/calibration strata, not as direct predictors.
- Fixed-threshold dual-filtration witness refinement (`H85` style) -> `rescue_once_with_major_change`.
  - Rescue only with adaptive filtration calibration and uncertainty-aware weighting.
- Global support-interaction overlays (`H73/H76/H79` style) -> `rescue_once_with_major_change`.
  - Rescue only with local/stratified biological conditioning (cell-ontology/module strata), not pooled global interaction coefficients.

## Navigation for iter_0036
- Keep one carry-over slot on `H88` focused on stability tightening and the `immune/source_disjoint` weakness.
- Use one high-risk slot for cross-model structural reset based on module-level topological invariants.
- Use one cheap orthogonal topology/manifold screen that is not another standalone additive ID or additive stability score.
