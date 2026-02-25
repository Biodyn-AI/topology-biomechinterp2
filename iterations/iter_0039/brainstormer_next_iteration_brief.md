# Next Iteration Brief - iter_0039 -> iter_0040

## Research Gate
- `passed_min_research_gate = true`.
- Run a full 3-slot packet.

## Packet Objective
- Stop spending budget on known additive failure modes.
- Exploit robust anchors (`H91`, `H93`) with interaction/contrastive topology.
- Allow exactly one cross-model attempt, but only with a fundamentally new anchor and strict fast-fail.

## Required 3-Slot Execution Packet

1. Slot A (high-probability discovery): `N508` interaction-only derivative rescue.
Scope:
- `seed42` pilot over `immune/lung/external_lung`, `source_disjoint/target_disjoint`, layers `{0,3,7,11}`.
- Promote to seeds `{43,44}` only if pilot keep gate is met.
Primary metric:
- `delta_auc_interaction_derivative_minus_h91_h93`.
Robustness metric:
- `null_gap_q95_delta_auc`.
Null package:
- derivative quantile-order permutation,
- interaction-partner shuffle within geodesic bins,
- label permutation.
Keep gate:
- positive mean delta in `>=4/6` domain-splits,
- positive mean null-gap in `>=3/6` domain-splits.
Fail-fast:
- if positive mean null-gap is `<=1/6`, retire this rescue line immediately.

2. Slot B (high-risk/high-reward): `N515` anchored core-to-noncore cross-model transfer.
Scope:
- `seed42` pilot over `immune/lung/external_lung`, layers `{7,11}`.
- Core anchor learned from high-confidence `H91/H93` in-model topology states; evaluate on held-out non-core modules.
Primary metric:
- `noncore_module_transfer_concordance` (plus transfer AUROC if available).
Robustness metric:
- domain-level `null_gap_q95_concordance`.
Null package:
- size-matched core-membership shuffle,
- random-anchor mapping,
- depth-order permutation,
- random-subspace alignment.
Keep gate:
- positive null-gap in `>=2/3` domains.
Fail-fast:
- if `0/3` domains pass, retire cross-model for at least the next three loops.

3. Slot C (cheap broad screen): `N520` depth motif grammar.
Scope:
- `seed42` breadth over all domains, both disjoint splits, layers `{0,3,7,11}`.
Primary metric:
- `delta_auc_motif_grammar_minus_h70`.
Robustness metric:
- `null_gap_q95_delta_auc`.
Null package:
- layer-order permutation with marginal preservation,
- token shuffle within layer,
- label permutation.
Keep gate:
- positive mean delta in `>=4/6` domain-splits,
- positive mean null-gap in `>=2/6` domain-splits.
Fail-fast:
- stop if positive mean delta is `<3/6` domain-splits.

## Null Budgets and Reporting Requirements
- Slot A: `>=24` permutations per null family in pilot; `>=40` for promotion.
- Slot B: `>=96` permutations per null family (small cross-model sample requires higher resolution).
- Slot C: `>=20` permutations per null family.
- Report by-row and domain-split/domain summaries, q95 null-gaps, and `p_best` distributions.

## Hard Discipline Rules
- Do not rerun exact `H100` or additive `H101` forms.
- Do not rerun additive bridge-curvature (`H95/H97`) or standalone ID entropy (`H98`) forms.
- Do not allocate more than one cross-model slot.
- Do not treat directional lift as success without null-gap pass.

## Contingency if Gate Unexpectedly Fails
1. Run Slot C first with reduced null budget (`16`) to re-establish valid machine output quickly.
2. Run Slot A on layers `{7,11}` only with reduced null budget (`16`).
3. Skip Slot B until gate returns `true`.
