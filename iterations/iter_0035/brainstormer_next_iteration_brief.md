# Next Iteration Brief - iter_0035 -> iter_0036

## Research Gate
- `passed_min_research_gate = true`.
- Run a full 3-slot packet.

## Packet Objective
- Convert `H88` from strong utility to stable mechanistic evidence.
- Run exactly one cross-model structural-reset test with strict fast-fail.
- Add one cheap orthogonal topology screen to avoid local overfitting.

## Required 3-Slot Execution Packet

1. Slot A (high-probability): `N449` stability-selected sparse descriptor consensus.
Scope: domains `immune/lung/external_lung`; seeds `42/43/44`; splits `source_disjoint/target_disjoint`; layers `{0,3,7,11}`.
Primary metric: `delta_auc_stability_selected_blend_minus_h70`.
Secondary metrics: `null_gap_q95_delta_auc`, descriptor-core stability (`nonzero-set Jaccard`, sign agreement), per-slice variance.
Keep gate: positive mean delta in `6/6` domain-splits, positive mean null-gap in `>=5/6`, and `Jaccard >= 0.65`.
Fail-fast rule: if after seed42+seed43 there are `<3/6` positive mean null-gap domain-splits, stop and mark as unstable.

2. Slot B (high-risk/high-reward): `N456` module-level cross-model persistence-image trajectory alignment.
Scope: seed42 pilot across `immune/lung/external_lung`; expand seeds only if pilot passes.
Primary metric: `module_trajectory_concordance_null_gap`.
Secondary metrics: module retrieval top-1 and depth-order concordance.
Keep gate: positive primary-metric null-gap in `>=2/3` domains.
Fail-fast rule: if pilot is `0/3` positive null-gap domains, retire this exact formulation immediately.

3. Slot C (cheap broad-screen): `N452` scale-space lifetime trajectory descriptors.
Scope: seed42 breadth run across all domains/splits/layers `{0,3,7,11}`.
Primary metric: `delta_auc_scale_trajectory_minus_h70`.
Secondary metric: `null_gap_q95_delta_auc`.
Keep gate: positive mean delta in `>=4/6` domain-splits and positive mean null-gap in `>=2/6`.
Fail-fast rule: if positive mean delta is `<3/6`, retire without reseeding.

## Pre-Registered Nulls
- Slot A: descriptor shuffle within geodesic bins, endpoint swap, label permutation, random-feature-subset control.
- Slot B: module-membership permutation (size-preserving), depth-order permutation, random subspace alignment baseline.
- Slot C: scale-order permutation, trajectory-feature shuffle within bins, label permutation.

## Hard Retire Rules for iter_0036
- Do not run another global cross-model transfer/order endpoint if Slot B fails.
- Do not reopen standalone additive intrinsic-dimension or additive perturbation-stability utility forms in this packet.
- Do not allocate more than one slot to interaction-overlay rescues.

## Minimal Recovery Plan (only if a future executor gate is false)
1. Run seed42-only `N449` on layers `{7,11}` and both splits with reduced null draws to re-establish one valid positive branch quickly.
2. Run seed42-only `N452` on a single split (`source_disjoint`) for cheap orthogonal screening.
3. Skip cross-model Slot B until gate returns true.
