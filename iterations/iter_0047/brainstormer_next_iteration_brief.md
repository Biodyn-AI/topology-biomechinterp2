# Next Iteration Brief - iter_0047 -> iter_0048

## Research Gate
- Current status: `passed_min_research_gate = true`.
- Execute a full 3-slot breadth packet.

## Packet Objective
- Convert `H124` from directional-only to null-robust in hard splits.
- Re-open cross-model work with one truly new structure-invariant objective.
- Expand `H126` with a cheap multi-scale geometry screen to map failure slices.

## Required 3-Slot Execution Packet

1. Slot A (high-probability discovery): `N641` biological hardening of `H124`.
Primary metric:
- `delta_vs_h70`
Robustness metrics:
- `mean_null_gap_q95` and `fraction_null_gap_positive` by `(domain, split_regime)`
Scope:
- seeds `{42,43,44}`
- domains `{immune, lung, external_lung}`
- splits `{source_disjoint, target_disjoint, dual_axis_disjoint}`
- layer `{11}`
- keep forced `lung/dual_axis_disjoint` coverage
Feature package:
- existing signed motif-community + STRING terms
- add GO co-membership interactions
- add adversarial motif decoy features matched by TF-degree-community strata
Null package:
- TF-identity-preserving sign shuffle
- motif decoy shuffle matched by TF/target degree strata
- STRING-bin permutation within strata
- GO-membership permutation within strata
- label permutation
Null budget:
- `>=64` per null family (use `>=96` if runtime allows)
Keep gate:
- positive mean null-gap in `>=6/9` domain-splits
- `lung/dual_axis_disjoint` mean null-gap `> 0`
- no domain with all three splits null-negative
Fail-fast:
- if interim seeds `{42,43}` already show `lung/dual_axis_disjoint <= 0` and `immune/source <= 0`, stop expansion and emit diagnostics

2. Slot B (high-risk/high-reward): `N638` cross-model perturbation-field persistence alignment.
Primary metrics:
- `alignment_delta_vs_random`
- `alignment_null_gap_q95` at domain level
Secondary metrics:
- cycle-residual stability
- perturbation-response rank agreement (diagnostic only)
Scope:
- seed42 pilot
- domains `{immune, lung, external_lung}`
- layers `{7,11}`
- splits `{source_disjoint, target_disjoint}`
Method:
- generate perturbation-response complexes for matched module sets in scGPT and Geneformer
- compute persistence images / landscapes per perturbation family
- align via sliced-Wasserstein or Procrustes on persistence features
Null package:
- module remap permutation preserving module sizes
- perturbation-family shuffle
- random-subspace projection baseline
- label permutation
Null budget:
- `>=32` per null family
Keep gate:
- positive domain null-gap in `>=2/3` domains
- immune domain null-gap `>=0`
Fail-fast:
- if all domains have non-positive null-gap in a pilot at layer `7`, retire endpoint immediately

3. Slot C (cheap broad-screen): `N634` multi-scale torsion spectrum over `H126`.
Primary metric:
- `delta_vs_h70`
Robustness metric:
- `null_gap_q95` by `(domain, split_regime, layer)`
Scope:
- seed42
- domains `{immune, lung, external_lung}`
- splits `{source_disjoint, target_disjoint}`
- layers `{7,11}`
- scales `{8,12,16}` neighbors or equivalent path-neighborhood scales
Feature package:
- turning-angle and torsion moments (mean, tail, variance)
- scale-consistency index
- directional asymmetry index
Null package:
- path reversal within length bins
- endpoint swap within distance bins
- scale-order permutation
- label permutation
Null budget:
- `>=24` per null family
Keep gate:
- positive mean delta in `>=4/6` domain-splits
- positive mean null-gap in `>=2/6`
- at least one source-disjoint domain-split null-positive
Fail-fast:
- if all source-disjoint rows are delta-negative, stop branch

## Common Execution Rules
- Exactly one exploitation slot (`N641`), one high-risk reset slot (`N638`), one cheap scan (`N634`).
- Promotion requires pre-registered null-gap gate pass; directional lift alone is insufficient.
- Required artifacts per slot:
  - row-level CSV
  - domain/split summary CSV
  - null summary CSV
  - iteration summary JSON update

## Minimal Recovery Plan (only if a future gate flips to `false`)
1. Run Slot C first with null budget `>=12` to restore fast, valid machine artifacts.
2. Run narrowed Slot A on `lung + immune` dual-axis/source splits only with full strict nulls.
3. Defer Slot B until gate returns to `true`.
