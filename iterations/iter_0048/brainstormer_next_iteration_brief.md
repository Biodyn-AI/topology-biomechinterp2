# Next Iteration Brief - iter_0048 -> iter_0049

## Research Gate
- Current status: `passed_min_research_gate = true`.
- Execute a full 3-slot packet.

## Packet Objective
- Convert the strongest active branch (`H127`) from directional-only to strict-null robust on hard slices.
- Re-open cross-model discovery with a topology-invariant objective, not direct transfer utility.
- Run one cheap manifold diagnostic to map mechanistic failure regions quickly.

## Required 3-Slot Execution Packet

1. Slot A (high-probability discovery): `N656` continuous GO-STRING semantic hardening.
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
- start from `H127` stack
- replace binary GO co-membership with continuous GO semantic similarity features (depth-weighted overlap / Resnik-like proxy)
- include interaction terms: `GO_semantic x STRING_confidence`, `GO_semantic x sign_consistency`, `GO_semantic x same_community`
Null package:
- TF-identity-preserving sign shuffle
- motif-decoy shuffle matched by TF/target degree strata
- STRING-bin permutation within strata
- GO-graph rewiring/permutation within ontology-depth strata
- label permutation
Null budget:
- `>=64` per null family (`>=96` preferred for hard slices)
Keep gate:
- positive mean null-gap in `>=6/9` domain-splits
- `lung/dual_axis_disjoint` mean null-gap `> 0`
- `immune/source_disjoint` mean null-gap `>= 0`
Fail-fast:
- after seeds `{42,43}`, if both hard slices remain negative by more than `-0.003`, stop scale-up and emit diagnostic artifact

2. Slot B (high-risk/high-reward): `N653` cross-model chart/sheaf consistency alignment.
Primary metrics:
- `alignment_delta_vs_random`
- `domain_null_gap_q95`
Secondary diagnostics:
- chart cycle-consistency residual
- immune-specific failure decomposition
Scope:
- seed42 pilot
- domains `{immune, lung, external_lung}`
- splits `{source_disjoint, target_disjoint}`
- layers `{7,11}`
Method:
- partition shared genes/modules by cell-ontology chart
- fit per-chart orthogonal alignment maps between scGPT and Geneformer feature spaces
- compute sheaf-like cycle inconsistency and alignment utility features
Null package:
- chart-label permutation preserving chart sizes
- cycle-order shuffle
- random map baseline
- label permutation
Null budget:
- `>=32` per null family
Keep gate:
- positive domain null-gap in `>=2/3` domains
- immune domain null-gap `>=0`
Fail-fast:
- if all three domains are null-negative at layer `7`, retire endpoint immediately

3. Slot C (cheap broad-screen): `N650` local chart-fracture manifold diagnostic.
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
- build local PCA charts along source-target geodesics
- derive fracture count, fracture density, max chart-angle jump, and fracture asymmetry features
Null package:
- chart-basis random rotation within neighborhoods
- path-order permutation
- endpoint swap within distance bins
- label permutation
Null budget:
- `>=24` per null family
Keep gate:
- positive mean delta in `>=4/6` domain-splits
- positive mean null-gap in `>=2/6`
- at least one source-disjoint split null-positive
Fail-fast:
- if all source-disjoint rows are delta-nonpositive, stop branch

## Common Execution Rules
- Exactly one exploitation slot (`N656`), one high-risk reset slot (`N653`), one cheap broad-screen slot (`N650`).
- Promotion requires pre-registered null-gap gates; directional lift alone is insufficient.
- Required artifacts per slot:
  - row-level CSV
  - domain/split summary CSV
  - null summary CSV
  - iteration summary JSON entry

## Minimal Recovery Plan (only if gate is `false`)
1. Run Slot C first with reduced null budget (`>=12`) to restore valid machine artifacts fast.
2. Run narrowed Slot A only on `lung/dual_axis_disjoint` and `immune/source_disjoint` with full strict nulls.
3. Defer Slot B until the gate returns to `true`.
