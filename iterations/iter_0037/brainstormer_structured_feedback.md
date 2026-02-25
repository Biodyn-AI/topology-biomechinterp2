# Brainstormer Structured Feedback - iter_0037

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0037/executor_research_validation.json`).
- Full 3-slot experimentation remains valid for the next loop.

## Iteration Evidence Snapshot
- `H94` (GO-ontology-stratified weighted filtration) is decisively negative:
  - mean `delta_auc_ontology_weighted_minus_global_weighted = -0.00933`
  - positive mean domain-splits: `0/6`
  - positive mean null-gap domain-splits: `0/6`.
- `H95` (bridge-curvature blend) is a near-miss, not a clear negative:
  - mean `delta_auc_graph_bridge_curvature_minus_h70 = +0.07710`
  - positive rows: `24/24`
  - positive mean null-gap domain-splits: `0/6` (domain means are close to zero in several slices).
- `H96` (cross-model module concordance rescue) is negative:
  - mean module Spearman `-0.00555`
  - mean null-gap(q95 Spearman) `-0.21467`
  - positive null-gap domains: `0/3`.

## Cumulative Pattern (Master Log + Paper)
- Top in-model branches remain `H91` and `H93` with broad null-robust support.
- Cross-model branch has repeated null-gap failures after multiple major objective changes; expected rescue yield is now low without new data view/supervision.
- Repeated additive standalone trajectories (ID/phase-boundary/stability trajectory) keep showing directional-only effects that fail null robustness.

## Stale Direction Triage
1. Cross-model endpoint family (`H71/H74/H77/H80/H83/H86/H96`) -> `retire_now`.
Reason: repeated objective redesigns still fail null-gap criteria.
Reopen rule: only with a single structural reset using shared module-level invariants plus strict fast-fail.

2. GO-overlap stratified additive refinement on weighted filtration (`H94` form) -> `retire_now`.
Reason: broad underperformance vs global weighted baseline and universal null-gap failure.
Reopen rule: only if stratification objective changes from additive feature blocks to hierarchical calibration or relative topology contrasts.

3. Standalone additive trajectory/stability utility forms (`H72/H90/H92`) -> `retire_now`.
Reason: repeated weak lift with `0/6` null-gap support pattern.
Reopen rule: diagnostics only, not direct utility endpoint.

4. Standalone additive intrinsic-dimension/phase-boundary forms (`H54/H60/H63/H66/H89`) -> `retire_now`.
Reason: repeated directional effects collapse under controls.
Reopen rule: interaction-only terms on top of proven topology backbones.

5. H95 bridge-curvature family -> `rescue_once_with_major_change`.
Reason: strong directional effect but no domain-level null-gap clearance.
Required change: nulls must preserve degree and edge-length-bin/bridge-rate structure, with higher permutation resolution.

## Navigation for Next Loop
- Keep one carry-over slot for `H95` only if null-model calibration is materially upgraded; do not rerun unchanged.
- Spend remaining slots on new topology/manifold directions that are not recycled additive failures.
- Allow at most one high-risk cross-model structural-reset slot with hard `0/3` fast-fail.

## Minimal Recovery Plan (only if a future gate flips false)
1. Seed42, layers `{7,11}`, rerun calibrated `H95` rescue with reduced but valid null budget (`>=16` draws/null family).
2. Run one cheap broad screen on a new manifold/topology idea (single seed, layers `{0,3,7,11}`).
3. Skip cross-model execution until the gate returns `true`.
