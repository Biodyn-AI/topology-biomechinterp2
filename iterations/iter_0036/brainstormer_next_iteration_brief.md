# Next Iteration Brief - iter_0036 -> iter_0037

## Research Gate
- `passed_min_research_gate = true`.
- Run a full 3-slot packet.

## Packet Objective
- Convert `H93` from seed42-positive to multiseed, high-null-resolution evidence.
- Add one high-risk structural-reset cross-model test with strict fast-fail.
- Add one cheap topology screen that is not another raw trajectory additive retry.

## Required 3-Slot Execution Packet

1. Slot A (high-probability): `N474` ontology-stratified weighted filtration (`H93` extension).
Scope: domains `immune/lung/external_lung`; seeds `42/43/44`; splits `source_disjoint/target_disjoint`; layers `{7,11}`.
Primary metric: `delta_auc_ontology_weighted_filtration_minus_h70`.
Secondary metrics: `null_gap_q95_delta_auc`, variance across ontology strata, gain over global `H93`.
Keep gate: positive mean delta in `6/6`, positive mean null-gap in `>=5/6`, and aggregate gain over global `H93` in `>=4/6` domain-splits.
Fail-fast: after seeds `42+43`, stop if `<3/6` domain-splits have positive mean null-gap.

2. Slot B (high-risk/high-reward): `N473` cross-model shared-latent persistence-image alignment.
Scope: seed42 pilot across `immune/lung/external_lung`; expand seeds only if pilot passes.
Primary metric: `module_persistence_alignment_null_gap`.
Secondary metrics: held-out module retrieval top-1 and cycle-consistency reconstruction error.
Keep gate: positive primary-metric null-gap in `>=2/3` domains.
Fail-fast: retire immediately if pilot is `0/3` positive domains.

3. Slot C (cheap broad-screen): `N467` persistence-entropy slope spectrum.
Scope: seed42 breadth run across domains, splits, and layers `{0,3,7,11}`.
Primary metric: `delta_auc_entropy_slope_minus_h70`.
Secondary metric: `null_gap_q95_delta_auc`.
Keep gate: positive mean delta in `>=4/6` domain-splits and positive mean null-gap in `>=2/6`.
Fail-fast: if positive mean delta is `<3/6`, retire without reseeding.

## Pre-Registered Nulls and Budgets
- Slot A: ontology-label permutation (size-preserving), confidence shuffle within degree bins, sign-flip, label permutation; target `>=64` draws per null family.
- Slot B: module-membership permutation, depth-order permutation, shuffled correspondence baseline, random subspace baseline; target `>=32` draws per null family.
- Slot C: scale-order permutation, entropy-feature shuffle within bins, label permutation; target `>=24` draws per null family.

## Hard Retire Rules for iter_0037
- Do not run any additional cross-model edge-transfer/rank endpoint if Slot B fails.
- Do not reopen standalone additive intrinsic-dimension or additive topology-stability trajectories in this packet.
- Do not allocate more than one carry-over slot from `H91/H93` lineages.

## Minimal Recovery Plan (only if a future gate becomes false)
1. Run seed42-only Slot A on layers `{7,11}` with `16` null draws to recover one valid positive branch quickly.
2. Run seed42-only Slot C on `source_disjoint` to preserve breadth at minimal cost.
3. Skip Slot B until `passed_min_research_gate` returns `true`.
