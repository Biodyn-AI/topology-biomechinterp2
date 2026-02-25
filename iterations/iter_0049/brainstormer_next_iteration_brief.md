# Next Iteration Brief - iter_0049 -> iter_0050

## Research Gate
- Current status: `passed_min_research_gate = true`.
- Execute a full 3-slot packet.

## Packet Objective
- Convert the only active lineage (`H130`) from directional-only to strict-null survival on hard slices.
- Re-open cross-model discovery with a correspondence-free invariant objective (not map-learning transfer).
- Run one cheap manifold screen to quickly test non-redundant geometry signal.

## Required 3-Slot Execution Packet

1. Slot A (high-probability discovery): `N670` hard-slice ontology-barrier rescue on top of `H130`.
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
- start from `H130` feature stack
- add `ontology_barrier_energy(source,target)`
- add `go_semantic_residual` (semantic score minus expected value in degree+depth strata)
- add `string_triangle_closure_support`
- add interactions: `ontology_barrier x go_semantic_residual`, `go_semantic_residual x sign_consistency`, `string_triangle_closure x same_community`
Null package:
- conditional randomization preserving TF-degree bin, target-degree bin, GO-depth bin, ontology-barrier quantile, STRING bin
- TF-identity-preserving sign shuffle
- motif-decoy shuffle matched by TF/target degree
- label permutation
Null budget:
- `>=96` draws per null family on hard slices (`lung/dual_axis_disjoint`, `immune/source_disjoint`)
- `>=64` draws per null family on other slices
Keep gate:
- positive mean null-gap in `>=4/9` domain-splits
- `lung/dual_axis_disjoint` mean null-gap `> 0`
- `immune/source_disjoint` mean null-gap `>= 0`
- both hard slices improve by at least `+0.01` versus `H130` null-gap
Fail-fast:
- after seeds `{42,43}`, if both hard slices are still below `-0.003`, stop seed44 and emit hard-slice diagnostics

2. Slot B (high-risk/high-reward): `N668` correspondence-free cross-model topological invariant alignment.
Primary metrics:
- `alignment_delta_vs_random`
- `domain_null_gap_q95`
Secondary diagnostics:
- per-domain invariant similarity (`sliced_wasserstein`, `mmd`)
- immune-specific failure decomposition
Scope:
- seed42 pilot (expand to seed43 only if keep gate passes)
- domains `{immune, lung, external_lung}`
- splits `{source_disjoint, target_disjoint}`
- layers `{7,11}`
Method:
- build matched edge-neighborhood filtrations in scGPT and Geneformer
- compute persistence images/Betti curves per neighborhood
- score correspondence-free agreement with distributional metrics (no learned map)
- use agreement features for utility lift vs random cross-model pairing baseline
Null package:
- cross-model pairing shuffle within domain/split/layer
- anchor-set shuffle within degree strata
- barcode-lifetime permutation within bins
- label permutation
Null budget:
- `>=32` draws per null family
Keep gate:
- positive domain null-gap in `>=2/3` domains
- immune domain null-gap `>=0`
Fail-fast:
- if all layer-7 domain rows are both delta-nonpositive and null-negative, retire endpoint immediately

3. Slot C (cheap broad-screen): `N667` intrinsic-dimension phase descriptor screen.
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
- compute TWO-NN intrinsic-dimension profile along directed geodesic paths
- derive `id_sign_flip_count`, `id_hysteresis`, `id_slope_asymmetry`
- evaluate additive lift over `H70`
Null package:
- ID-profile permutation along path
- path-order permutation
- endpoint swap within distance bins
- label permutation
Null budget:
- `>=24` draws per null family
Keep gate:
- positive mean delta in `>=4/6` domain-splits
- positive mean null-gap in `>=2/6`
- at least one source-disjoint split with positive mean null-gap
Fail-fast:
- if all source-disjoint rows are delta-nonpositive, stop branch

## Common Execution Rules
- Exactly one exploitation slot (`N670`), one high-risk reset slot (`N668`), one cheap broad-screen slot (`N667`).
- Promotion requires passing pre-registered null-gap gates; directional lift alone is insufficient.
- Required artifacts per slot:
  - row-level CSV
  - domain/split summary CSV
  - null summary CSV
  - iteration summary JSON entry

## Minimal Executable Recovery Plan (if a future gate is `false`)
1. Run Slot C first with reduced null budget (`>=12`) to quickly restore valid machine outputs.
2. Run narrowed Slot A only on `lung/dual_axis_disjoint` and `immune/source_disjoint` with full strict nulls.
3. Defer Slot B until gate returns to `true`.
