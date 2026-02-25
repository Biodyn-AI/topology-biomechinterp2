# Next Iteration Brief - iter_0046 -> iter_0047

## Research Gate
- Current status: `passed_min_research_gate = true`.
- Execute the full 3-slot packet.

## Packet Objective
- Convert the strongest branch (`H123` lineage) from robust-but-incomplete to fully covered and promotion-ready.
- Attempt one truly new cross-model alignment objective with biological anchors built in.
- Run one cheap geometry probe to map `H121` source-disjoint failure regions.

## Required 3-Slot Execution Packet

1. Slot A (high-probability discovery): `N625` TRRUST+STRING hardening and coverage completion.
Primary metric:
- `delta_vs_h70`
Robustness metrics:
- `mean_null_gap_q95` and `fraction_null_gap_positive` by `(domain, split_regime)`
Scope:
- seeds `{42,43,44}`
- domains `{immune, lung, external_lung}`
- splits `{source_disjoint, target_disjoint, dual_axis_disjoint}`
- layer `{11}`
- force retention of `lung/dual_axis_disjoint` rows (adjust sampling thresholds if needed)
Null package:
- TF-identity-preserving sign shuffle
- motif-decoy shuffle matched on TF/target degree strata
- STRING-confidence bin permutation within matched strata
- label permutation
Null budget:
- `>=64` permutations per null family
Keep gate:
- positive direction in `>=8/9` domain-splits
- positive mean null-gap in `>=8/9` domain-splits
- explicit `lung/dual_axis_disjoint` mean null-gap positive
Fail-fast:
- if interim run (seeds `{42,43}`) still misses `lung/dual_axis_disjoint`, stop and emit blocker artifact instead of scaling up

2. Slot B (high-risk/high-reward): `N622` anchor-constrained cycle-consistent cross-model alignment.
Primary metrics:
- `transfer_delta_auc_vs_h70`
- `cycle_consistency_error` (lower is better, converted to signed utility)
Robustness metric:
- domain-level `null_gap_q95` on transfer lift
Scope:
- seed42 pilot
- domains `{immune, lung, external_lung}`
- splits `{source_disjoint, target_disjoint}`
- layers `{7,11}`
- anchor set from TRRUST TF modules + GO process labels
Null package:
- anchor-label permutation preserving anchor cardinality
- random correspondence baseline
- label permutation
Null budget:
- `>=32` permutations per null family
Keep gate:
- positive domain null-gap in `>=2/3` domains
- immune domain mean null-gap `>=0`
Fail-fast:
- if `0/3` domains are positive-null-gap at pilot scale, retire endpoint immediately

3. Slot C (cheap broad-screen): `N620` geodesic torsion and turning-angle asymmetry.
Primary metric:
- `delta_vs_h70`
Robustness metric:
- `null_gap_q95` by `(domain, split_regime)`
Scope:
- seed42
- domains `{immune, lung, external_lung}`
- splits `{source_disjoint, target_disjoint}`
- layers `{7,11}`
Null package:
- path reversal within path-length bins
- endpoint swap within distance bins
- label permutation
Null budget:
- `>=24` permutations per null family
Keep gate:
- positive mean delta in `>=4/6` domain-splits
- positive mean null-gap in `>=2/6` domain-splits
- at least one source-disjoint domain-split with positive mean null-gap
Fail-fast:
- if all source-disjoint mean deltas are non-positive, stop branch

## Common Execution Rules
- Exactly one exploitation slot (`N625`), one major-reset exploration slot (`N622`), one cheap broad-screen slot (`N620`).
- Keep artifacts per slot mandatory:
  - row-level CSV
  - domain/split summary CSV
  - null summary CSV
  - iteration-level summary JSON update
- Promotion rule:
  - no branch promotion without pre-registered null-gap gate pass.

## Minimal Fallback Plan (if next gate flips false)
1. Run Slot C first with reduced null budget (`>=12`) to quickly restore valid machine artifacts.
2. Run narrowed Slot A on `lung` only (layer 11, seeds `{42,43}`) with full strict nulls.
3. Defer Slot B until gate returns `true`.
