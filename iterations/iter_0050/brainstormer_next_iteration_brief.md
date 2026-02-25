# Next Iteration Brief - iter_0050 -> iter_0051

## Research Gate
- Current status: `passed_min_research_gate = true`.
- Execute a full 3-slot packet.

## Packet Objective
- Recover strict-null robustness on hard slices by replacing H130-style additive semantics with a new biological-topological mechanism.
- Re-open cross-model discovery using correspondence-free structure agreement, not map learning.
- Run one cheap geometric screen to keep exploration breadth high.

## Required 3-Slot Execution Packet

1. Slot A (high-probability discovery): `N686` cell-ontology sheaf hard-slice rescue.
Primary metric:
- `delta_vs_h70`
Robustness metrics:
- `mean_null_gap_q95` and `fraction_null_gap_positive` by `(domain, split_regime)`
Scope:
- seeds `{42,43,44}`
- domains `{immune, lung, external_lung}`
- splits `{source_disjoint, target_disjoint, dual_axis_disjoint}`
- layer `{11}`
Feature package:
- start from H123-level robust core (signed motif-community hardening)
- add `ontology_sheaf_obstruction_energy`
- add `chart_boundary_crossing_count`
- add interactions: `sheaf_obstruction x sign_consistency`, `sheaf_obstruction x string_confidence`, `chart_boundary_crossing x same_community`
Null package:
- ontology chart relabel preserving chart sizes
- section shuffle preserving node degree strata
- TF-identity-preserving sign shuffle
- motif decoy shuffle matched by TF/target degree
- label permutation
Null budget:
- `>=96` draws per null family for hard slices (`immune/source_disjoint`, `lung/dual_axis_disjoint`)
- `>=64` draws per null family for other slices
Keep gate:
- `immune/source_disjoint` mean null-gap `>= 0`
- `lung/dual_axis_disjoint` mean null-gap `>= 0`
- positive mean null-gap in `>=4/9` domain-splits
Fail-fast:
- after seeds `{42,43}`, if both hard slices remain below `-0.004`, stop seed44 and emit hard-slice diagnostics

2. Slot B (high-risk/high-reward): `N684` correspondence-free cross-model persistence-kernel alignment.
Primary metrics:
- `alignment_delta_vs_random`
- `domain_null_gap_q95`
Secondary diagnostics:
- per-domain kernel alignment score
- per-layer contribution decomposition (`7` vs `11`)
Scope:
- seed42 pilot (expand to seed43 only if keep gate passes)
- domains `{immune, lung, external_lung}`
- splits `{source_disjoint, target_disjoint}`
- layers `{7,11}`
Method:
- compute persistence images/landscapes for matched domain/split/layer slices in scGPT and Geneformer
- compute correspondence-free kernel agreement features (no gene map)
- test whether agreement features improve edge utility over random cross-model pairing baseline
Null package:
- cross-model pairing shuffle within domain/split/layer
- kernel-spectrum permutation
- anchor set shuffle in degree bins
- label permutation
Null budget:
- `>=32` draws per null family
Keep gate:
- positive domain null-gap in `>=2/3` domains
- immune domain null-gap `>= 0`
Fail-fast:
- if all immune rows are delta-nonpositive and null-negative at layer `7`, retire endpoint immediately

3. Slot C (cheap broad-screen): `N680` sectional-curvature anisotropy screen.
Primary metric:
- `delta_vs_h70`
Robustness metric:
- `null_gap_q95` by `(domain, split_regime, layer)`
Scope:
- seed42
- domains `{immune, lung, external_lung}`
- splits `{source_disjoint, target_disjoint}`
- layers `{7,11}`
Method:
- estimate local tangent planes at source/target neighborhoods
- compute sectional curvature anisotropy and source-target asymmetry summaries
- evaluate additive lift over H70
Null package:
- endpoint swap within distance bins
- tangent basis random rotation
- path direction reversal
- label permutation
Null budget:
- `>=24` draws per null family
Keep gate:
- positive mean delta in `>=4/6` domain-splits
- positive mean null-gap in `>=2/6` domain-splits
- at least one source-disjoint split with positive mean null-gap
Fail-fast:
- if both source-disjoint domains are delta-nonpositive at layer `7`, stop this branch

## Common Execution Rules
- Exactly one exploitation slot (`N686`), one high-risk reset slot (`N684`), one cheap broad-screen slot (`N680`).
- Use strict null-gap gates for keep/retire decisions; directional lift alone does not pass.
- Required artifacts per slot:
  - row-level CSV
  - domain/split summary CSV
  - null summary CSV
  - inclusion in iteration summary JSON

## Recovery Mode (only if next gate is `false`)
1. Run Slot C first with reduced null budget (`>=12`) for fast valid artifacts.
2. Run Slot A only on the two hard slices with full null budgets.
3. Defer Slot B until the gate returns to `true`.
