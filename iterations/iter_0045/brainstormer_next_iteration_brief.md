# Next Iteration Brief - iter_0045 -> iter_0046

## Research Gate
- Current status: `passed_min_research_gate = true`.
- Run a full 3-slot packet.

## Packet Objective
- Convert `H118` from promising to robust or retire quickly under stricter controls.
- Try one true topological objective reset for cross-model transfer.
- Run one low-cost geometry screen that directly attacks the `H120` source-disjoint weakness.

## Required 3-Slot Execution Packet

1. Slot A (high-probability discovery): `N600` strict `H118` hardening.
Primary metric:
- `delta_vs_h70`
Robustness metrics:
- `mean_null_gap_q95` by `(domain, split_regime)`
- `fraction_null_gap_positive` by `(domain, split_regime)`
Scope:
- seeds `{42,43,44}`
- domains `{immune, lung, external_lung}`
- splits `{source_disjoint, target_disjoint, dual_axis_disjoint}`
- layer `{11}`
Null package:
- TF-identity-preserving TRRUST sign shuffle
- motif-decoy shuffle matched on TF degree, target degree, and motif prevalence
- label permutation
Null budget:
- `>=64` permutations per null family
Keep gate:
- positive direction in `>=8/9` domain-splits
- positive mean null-gap in `>=5/9` domain-splits
Fail-fast:
- if interim run (seed42+seed43) has positive mean null-gap in `<=3/9`, do not expand further; retire endpoint

2. Slot B (high-risk/high-reward): `N609` persistence-landscape transport for cross-model alignment.
Primary metrics:
- `module_landscape_transport_residual`
- `transfer_delta_auc_vs_h70`
Robustness metric:
- domain-level `null_gap_q95` on transfer lift
Scope:
- seed42 pilot
- domains `{immune, lung, external_lung}`
- split `{source_disjoint, target_disjoint}`
- layers `{7,11}`
Null package:
- random module mapping
- landscape-bin permutation
- label permutation
Null budget:
- `>=32` permutations per null family
Keep gate:
- positive domain null-gap in `>=2/3` domains
- immune domain null-gap `>=0`
Fail-fast:
- if positive domain null-gap is `0/3`, retire this formulation immediately

3. Slot C (cheap broad-screen): `N605` directional geodesic asymmetry.
Primary metric:
- `delta_vs_h70` from asymmetry features
Robustness metric:
- `null_gap_q95` by domain/split
Scope:
- seed42
- domains `{immune, lung, external_lung}`
- splits `{source_disjoint, target_disjoint}`
- layers `{7,11}`
Null package:
- direction-flip within path-length bins
- endpoint swap within distance bins
- label permutation
Null budget:
- `>=24` permutations per null family
Keep gate:
- positive mean delta in `>=4/6` domain-splits
- positive mean null-gap in `>=2/6` domain-splits
- at least one `source_disjoint` split with positive mean null-gap
Fail-fast:
- if all three `source_disjoint` mean null-gaps remain negative, stop this branch

## Common Execution Rules
- Keep breadth discipline: exactly one carry-over hardening slot (`N600`), one major-reset slot (`N609`), one cheap broad screen (`N605`).
- Required artifacts per slot:
  - row-level CSV
  - domain/split summary CSV
  - null summary CSV
  - iteration-level machine summary JSON entry
- Promotion rule:
  - no promotion without passing pre-registered null-gap gates

## Minimal Fallback Plan (only if next gate flips false)
1. Run Slot C first with reduced null budget (`>=12`) to quickly restore valid machine artifacts.
2. Run a narrowed Slot A (`lung`, `layer=11`, seeds `{42,43}`) with full null package.
3. Defer Slot B until gate returns `true`.
