# Brainstormer Structured Feedback - iter_0036

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0036/executor_research_validation.json`).
- Full 3-slot experimentation is valid; no recovery-only restriction is needed for this loop.

## Iteration Evidence Snapshot
- `H91` is a confirmed positive branch, not just directional:
  - mean `delta_auc_stability_selected_blend_minus_h70 = +0.07424`
  - positive rows `72/72`
  - positive mean null-gap in `6/6` domain-splits
  - descriptor stability hit target (`mean nonzero-set Jaccard = 0.65046`, sign agreement `1.0`).
- `H93` is a strong positive lead with biological anchoring:
  - mean `delta_auc_weighted_filtration_minus_h70 = +0.08443`
  - positive rows `12/12`
  - positive mean null-gap in `6/6` domain-splits.
  - caveat: null resolution is coarse (small permutation budget; many `p_best_upper = 0.142857`).
- `H92` is negative in tested form:
  - mean `delta_auc_scale_trajectory_minus_h70 = +0.00386`
  - positive mean null-gap in `0/6` domain-splits.

## Cumulative Pattern (Master Log + Paper)
- Strongest current lineages are still in-model topology + biologically anchored filtration (`H70 -> H87/H88/H91`, `H82/H85/H93`).
- Repeated cross-model endpoint families remain stale: post-`iter_0028` attempts (`H68/H71/H74/H77/H80/H83/H86`) are uniformly null-gap negative.
- Repeated standalone additive intrinsic-dimension/local-linearity utility forms remain stale (`H54/H60/H63/H66/H89`).
- Repeated standalone additive topology-stability/trajectory utilities remain stale (`H72/H90/H92`).

## Stale Direction Triage
1. Cross-model edge-utility transfer/rank endpoint family (`H68/H71/H74/H77/H80/H83/H86`) -> `retire_now`.
Reason: long repeated negative sequence under multiple redesigns.
Reopen rule: only with a structural-reset objective (module-level invariants), strict `0/3` fast-fail.

2. Standalone additive intrinsic-dimension and local-linearity utilities (`H54/H60/H63/H66/H89`) -> `retire_now`.
Reason: repeated directional-only lift collapses under null controls.
Reopen rule: interaction-only on top of a positive topology backbone.

3. Standalone additive topology-stability and scale-trajectory utilities (`H72/H90/H92`) -> `retire_now`.
Reason: repeated weak deltas and `0/6` null-gap support pattern.
Reopen rule: diagnostics or calibration strata only, not direct predictors.

4. Global pooled biological interaction overlays (`H73/H76/H79`) -> `rescue_once_with_major_change`.
Reason: utility sometimes positive, interaction robustness repeatedly weak.
Required change: local/ontology-stratified conditioning instead of pooled interaction coefficients.

5. Fixed-threshold dual-filtration witness refinement (`H85`) -> `rescue_once_with_major_change`.
Reason: near-miss branch with mixed robustness.
Required change: adaptive, uncertainty-aware thresholding by domain/split/layer.

## Navigation for Next Loop
- Keep carry-over budget tight: one carry-over slot only, centered on upgrading `H93` null resolution and multiseed replication.
- Use remaining budget on geometric/topological novelty, not another retry of retired additive families.
- Cross-model work is allowed only as one high-risk structural-reset slot with hard fast-fail.

## Minimal Recovery Plan (if a future gate fails)
1. Run seed42-only `H93` replication (`layers {7,11}`, both splits, 16 null draws) to re-establish a valid positive quickly.
2. Run one cheap topology novelty screen on seed42 (`layers {0,3,7,11}`) with reduced null budget.
3. Skip cross-model execution until gate returns `true`.
