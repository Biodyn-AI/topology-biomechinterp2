# Next Iteration Brief - iter_0040 -> iter_0041

## Research Gate
- Current status: `passed_min_research_gate = true`.
- Execute full 3-slot packet.

## Packet Objective
- Keep one anchor-adjacent high-probability slot to restore discovery momentum.
- Spend one slot on a true high-upside cross-model reset with strict fast-fail.
- Keep one cheap broad-screen slot for exploration breadth.

## Required 3-Slot Execution Packet

1. Slot A (high-probability discovery): `N538` STRING triad-closure weighted filtration.
Primary metric:
- `delta_auc_string_triad_weighted_minus_h93`
Robustness metric:
- `null_gap_q95_delta_auc`
Scope:
- seed42 pilot, domains `{immune, lung, external_lung}`
- splits `{source_disjoint, target_disjoint}`
- layers `{7,11}`
Null package:
- STRING-weight shuffle within degree bins
- triad-closure shuffle
- label permutation
Keep gate:
- positive mean null-gap in `>=3/6` domain-splits
Fail-fast:
- retire immediately if positive mean null-gap is `0/6`

2. Slot B (high-risk/high-reward): `N531` cross-model perturbation-response alignment.
Primary metric:
- `module_response_rank_spearman`
Robustness metric:
- `null_gap_q95_response_concordance` (domain-level)
Scope:
- seed42 pilot across all 3 domains on shared genes/modules
- perturbation panel includes module dropout, sign flips, and local rewiring
Null package:
- perturbation-schedule permutation
- module-label shuffle
- random gene mapping baseline
Keep gate:
- positive domain-level null-gap in `>=2/3` domains
Fail-fast:
- if `0/3`, freeze cross-model experiments for next 3 loops

3. Slot C (cheap broad-screen): `N537` finite-state descriptor motif screen.
Primary metric:
- `delta_auc_dfa_motif_minus_h70`
Robustness metric:
- `null_gap_q95_delta_auc`
Scope:
- seed42 breadth run over all domains/splits
- descriptor trajectories over layers `{0,3,7,11}`
Null package:
- layer-order permutation preserving marginals
- token shuffle within layer
- label permutation
Keep gate:
- positive mean delta in `>=4/6` domain-splits and positive mean null-gap in `>=2/6`
Fail-fast:
- stop lineage if positive mean delta `<3/6`

## Common Execution Rules
- Null budgets:
  - Slot A: `>=24` draws per null family
  - Slot B: `>=96` draws per null family (small sample; needs tighter q95)
  - Slot C: `>=20` draws per null family
- Report format:
  - by-row CSV, domain-split summary CSV, null summary CSV, machine-readable JSON summary
- Promotion rule:
  - no promotion without passing pre-registered null-gap gate
- Breadth discipline:
  - no more than one carry-over refinement slot

## Minimal Fallback Plan (use only if gate flips false)
1. Run Slot C first with reduced null budget (`16`) to produce valid machine artifacts quickly.
2. Run Slot A restricted to layers `{7,11}` with reduced null budget (`16`).
3. Skip Slot B until gate returns `true`.
