# Next Iteration Brief - iter_0038 -> iter_0039

## Research Gate
- `passed_min_research_gate = true`.
- Execute a full 3-slot packet.

## Packet Objective
- Pivot away from stale additive lineages (`H95/H97`, `H98`) and repeated cross-model null failures (`H99` lineage).
- Anchor one slot in high-probability in-model relative topology.
- Take exactly one high-risk cross-model structural reset with strict fast-fail.
- Keep one low-cost broad screen to preserve exploration breadth.

## Required 3-Slot Execution Packet

1. Slot A (high-probability discovery): `N493` relative persistence contrast.
Scope:
- Pilot: `seed42`, domains `immune/lung/external_lung`, splits `source_disjoint/target_disjoint`, layers `{7,11}`.
- Promote to layers `{0,3,7,11}` and seeds `43/44` only if pilot passes gate.
Primary metric:
- `delta_auc_relative_ph_contrast_minus_h93`.
Robustness metric:
- `null_gap_q95_delta_auc`.
Null package:
- edge-anchor permutation preserving degree/length bins,
- size-matched random background complexes,
- label permutation.
Keep gate:
- positive mean delta in `>=5/6` domain-splits,
- positive mean null-gap in `>=4/6` domain-splits.
Fail-fast:
- if positive mean null-gap is `<=1/6` in pilot, retire this exact formulation.

2. Slot B (high-risk/high-reward): `N501` cross-model OT + monotone depth warp.
Scope:
- `seed42` pilot across `immune/lung/external_lung`, layers `{7,11}`.
Primary metric:
- `module_persistence_ot_concordance`.
Robustness metric:
- `null_gap_q95_concordance` (domain-level).
Secondary diagnostics:
- warped OT transport cost,
- top-k module retrieval overlap,
- alignment RMSE.
Null package:
- module-membership permutation (size-preserving),
- depth-order permutation,
- random monotone warp,
- random-subspace alignment baseline.
Keep gate:
- positive primary null-gap in `>=2/3` domains.
Fail-fast:
- if `0/3` domains positive, re-retire cross-model branch immediately.

3. Slot C (cheap broad-screen): `N497` persistence derivative spectrum.
Scope:
- `seed42` breadth over all domains, both disjoint splits, layers `{0,3,7,11}`.
Primary metric:
- `delta_auc_persistence_derivative_minus_h70`.
Robustness metric:
- `null_gap_q95_delta_auc`.
Null package:
- quantile-order permutation,
- derivative-sign randomization,
- label permutation.
Keep gate:
- positive mean delta in `>=4/6` domain-splits,
- positive mean null-gap in `>=2/6` domain-splits.
Fail-fast:
- stop if positive mean delta is `<3/6` domain-splits.

## Null Budgets and Reporting Requirements
- Slot A: `>=32` permutations per null family in pilot; `>=48` in promotion run.
- Slot B: `>=80` permutations per null family (small-row cross-model pilot needs higher resolution).
- Slot C: `>=24` permutations per null family.
- Report domain-split/domain means, row-level pass counts, and q95 null-gap distributions.

## Hard Discipline Rules
- Do not run another additive bridge-curvature utility rerun (`H95/H97` form).
- Do not run another standalone/additive ID entropy screen (`H98` form).
- Do not run more than one cross-model slot in this packet.
- Do not treat directional-only lift as success; null-gap gate is mandatory.

## Contingency (only if next gate unexpectedly fails)
1. Run Slot C first with reduced null budget (`16`) to re-establish valid machine output quickly.
2. Run Slot A on layers `{7,11}` only with `16` null draws.
3. Skip Slot B until gate returns `true`.
