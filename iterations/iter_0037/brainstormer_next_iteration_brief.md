# Next Iteration Brief - iter_0037 -> iter_0038

## Research Gate
- `passed_min_research_gate = true`.
- Run a full 3-slot packet.

## Packet Objective
- Adjudicate the strongest near-miss (`H95`) with a materially stronger null design.
- Take one high-risk structural-reset shot on cross-model alignment.
- Add one cheap breadth screen in manifold/ID space that is not a retired additive form.

## Required 3-Slot Execution Packet

1. Slot A (high-probability discovery): `N482` calibrated bridge-curvature rescue.
Scope: `seed42` pilot first across `immune/lung/external_lung`, both disjoint splits, layers `{0,3,7,11}`; expand to seeds `43/44` only if pilot passes.
Primary metric: `delta_auc_graph_bridge_curvature_minus_h70`.
Key robustness metric: `null_gap_q95_delta_auc` under structure-matched nulls.
Null package:
- degree-preserving edge swap (baseline),
- degree + edge-length-bin preserving swap,
- degree + edge-length-bin + bridge-rate-stratum preserving swap,
- descriptor shuffle and label permutation.
Keep gate: positive mean delta in `6/6` domain-splits and positive mean null-gap in `>=3/6`.
Fail-fast: if pilot remains `0/6` positive mean null-gap, retire the bridge-curvature additive branch.

2. Slot B (high-risk/high-reward): `N487` cross-model module role-graph alignment.
Scope: seed42 pilot across `immune/lung/external_lung`, layers `{7,11}`.
Primary metric: `module_role_graph_concordance_null_gap`.
Secondary metrics: top-k role retrieval and cycle-consistency reconstruction error.
Null package:
- module-membership permutation (size-preserving),
- role-label permutation,
- depth-order permutation,
- random-subspace alignment baseline.
Keep gate: positive primary-metric null-gap in `>=2/3` domains.
Fail-fast: if `0/3` domains positive, retire this exact cross-model formulation immediately.

3. Slot C (cheap broad-screen): `N486` multi-radius ID heterogeneity entropy.
Scope: seed42 breadth across all three domains, both disjoint splits, layers `{0,3,7,11}`.
Primary metric: `delta_auc_id_entropy_minus_h70`.
Robustness metric: `null_gap_q95_delta_auc`.
Null package:
- radius-order permutation,
- neighborhood assignment shuffle,
- label permutation.
Keep gate: positive mean delta in `>=4/6` domain-splits and positive mean null-gap in `>=2/6`.
Fail-fast: retire if positive mean delta `<3/6`.

## Null Budgets and Reporting Requirements
- Slot A: `>=32` permutations per null family in pilot; `>=48` on promotion run.
- Slot B: `>=64` permutations per null family (small-row cross-model pilot needs higher resolution).
- Slot C: `>=24` permutations per null family.
- Report domain-split (or domain) means, row-level pass counts, and null-gap distributions, not only aggregate means.

## Hard Discipline Rules
- Do not run another GO-overlap additive stratification variant of `H94`.
- Do not run standalone additive topology-stability trajectory variants (`H72/H90/H92` forms).
- Do not run additional cross-model endpoints beyond Slot B in this packet.

## Minimal Recovery Plan (only if next gate unexpectedly fails)
1. Run Slot A on seed42 with reduced null budget (`16`) to re-establish one valid branch quickly.
2. Run Slot C only on `source_disjoint` as a cheap breadth sanity check.
3. Skip Slot B until gate returns `true`.
