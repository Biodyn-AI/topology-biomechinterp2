# Brainstormer Structured Feedback - iter_0038

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0038/executor_research_validation.json`).
- Full 3-slot experimentation is allowed next loop.

## Iteration Evidence Snapshot
- `H97` (`graph_topology`) stayed directionally strong but non-robust:
  - mean `delta_auc_graph_bridge_calibrated_minus_h70 = +0.07852`
  - positive rows: `24/24`
  - positive mean null-gap domain-splits: `0/6`.
- `H98` (`intrinsic_dimensionality`) was negative:
  - mean `delta_auc_id_entropy_minus_h70 = -0.00773`
  - positive mean domain-splits: `1/6`
  - positive mean null-gap domain-splits: `0/6`.
- `H99` (`cross_model_alignment`) remained null-fragile after structural reset:
  - mean concordance `+0.03934`
  - positive null-gap domains: `0/3`.

## Cumulative Pattern (Master Log + Paper + Screens)
- Active in-model leaders are still `H91` and `H93`.
- `cross_model_alignment` is a stale branch: `7` consecutive negatives in `iter_0030-iter_0038` (`H74/H77/H80/H83/H86/H96/H99`).
- Additive standalone intrinsic-dimension forms are stale (`H89`, `H98`, plus older lineage) and still non-robust.
- Bridge-curvature additive lineage (`H95/H97`) appears to be a repeated directional-but-null-failing pattern (q95 null-gaps near zero but still negative in all domain-splits).

## Stale Direction Triage
1. Cross-model unsupervised concordance endpoints (`H74/H77/H80/H83/H86/H96/H99`) -> `retire_now`.
Reason: repeated `0/3` domain-level null-gap support after multiple endpoint resets.
Reopen only if: objective is anchored to shared robust in-model topology states, not raw concordance.

2. Additive bridge-curvature utility lineage (`H95/H97`) -> `retire_now`.
Reason: two consecutive runs with strong directional lift but `0/6` positive mean null-gap domain-splits.
Reopen only if: endpoint is non-additive/contrastive (relative topology objective), not another additive AUROC blend.

3. Standalone/additive intrinsic-dimension entropy lineage (`H89/H98`) -> `retire_now`.
Reason: repeated null failure and underperformance versus baseline.
Reopen only if: interaction-only design on top of robust topology backbones (`H91/H93`).

4. GO-stratified additive refinement (`H94` form) -> `retire_now`.
Reason: direct underperformance against global weighted filtration.

5. Global pooled interaction overlays (`H73/H76/H79`) -> `rescue_once_with_major_change`.
Required change: hierarchical/local conditioning (ontology/module-level partial pooling), not pooled global coefficients.

## Navigation for Next Loop
- Allocate `2/3` slots to non-retired in-model topology/manifold hypotheses anchored to `H91/H93`.
- Allow at most `1/3` slot for cross-model work, and only with a hard fast-fail (`0/3` domains => immediate re-retire).
- Prefer relative/contrastive topology endpoints and interaction-only manifold terms over additive overlays.

## Minimal Recovery Plan (if a future gate flips false)
1. Run one cheap broad screen on a new topology derivative feature family at `seed42` and layers `{0,3,7,11}`.
2. Run one reduced-budget high-probability relative-topology test (`>=16` null draws) on layers `{7,11}`.
3. Skip cross-model execution until gate returns `true`.
