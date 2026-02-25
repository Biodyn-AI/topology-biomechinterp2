# Next Iteration Brief - iter_0034 -> iter_0035

## Research Gate
- `passed_min_research_gate = true`, so run a full 3-slot discovery packet.

## Packet Goal
- Convert `H87` from single-seed signal into robust, mechanistic evidence.
- Attempt one high-risk cross-model structural reset with strict fast-fail.
- Keep one cheap orthogonal geometric screen to preserve breadth.

## Required 3-Slot Execution Packet

1. Slot A (high-probability): `N448` multiseed sparse-descriptor consensus.
Scope: domains `immune/lung/external_lung`, seeds `42/43/44`, splits `source_disjoint/target_disjoint`, layers `{0,3,7,11}`.
Primary metric: `delta_auc_descriptor_blend_minus_h70`.
Secondary metrics: `null_gap_q95_delta_auc`, descriptor-core stability (`nonzero-set Jaccard`, coefficient-sign agreement).
Keep gate: positive mean delta in `>=5/6` domain-splits, positive mean null-gap in `>=5/6`, and descriptor-core `Jaccard >= 0.6`.
Fail-fast: after seed42, if positive mean null-gap is `<2/6`, stop multiseed expansion and mark formulation unstable.

2. Slot B (high-risk/high-reward): `N442` cross-model topological role-transition alignment.
Scope: seed42 pilot on `immune/lung/external_lung`; expand seeds only if pilot passes.
Primary metric: `role_transition_concordance_score` (post role-space GW/OT alignment).
Secondary metrics: depth-order Spearman in role space, role-retrieval top-1.
Keep gate: positive primary-metric null-gap in `>=2/3` domains.
Fail-fast: if pilot is `0/3` positive null-gap domains, retire this exact formulation immediately.

3. Slot C (cheap broad-screen): `N441` local linearity phase-boundary screen.
Scope: seed42 breadth run across all domains, splits, layers `{0,3,7,11}`.
Primary metric: `delta_auc_phase_boundary_minus_h70`.
Secondary metric: `null_gap_q95_delta_auc`.
Keep gate: positive mean delta in `>=4/6` domain-splits and positive mean null-gap in `>=2/6`.
Fail-fast: if positive mean delta is `<3/6`, retire without reseeding.

## Pre-Registered Nulls
- Slot A: descriptor shuffle within geodesic bins, endpoint swap within bins, label permutation, coefficient-sign randomization.
- Slot B: role-label permutation, depth-order permutation, random doubly-stochastic transport baseline.
- Slot C: layer-order permutation, feature shuffle within bins, label permutation.

## Hard Retire Rules for iter_0035
- Do not run another cross-model endpoint if Slot B fails `0/3` on domain null-gap.
- Do not reopen standalone additive graph-topology or standalone intrinsic-dimension utility objectives in this packet.
- If Slot A fails stability gate, demote sparse-blend as exploratory-only and switch next topology slot to `N434` once.

## Recovery Contingency (only if a future gate fails)
- Minimal executable fallback packet:
  1. Run seed42-only `N448` on layers `{7,11}` with reduced null draws to re-establish one validated positive branch.
  2. Run seed42-only `N441` single-split (`source_disjoint`) for quick orthogonal signal check.
  3. Skip cross-model slot until gate is green again.
