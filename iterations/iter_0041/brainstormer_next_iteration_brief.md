# Next Iteration Brief - iter_0041 -> iter_0042

## Research Gate
- Current status: `passed_min_research_gate = true`.
- Execute full 3-slot packet.

## Packet Objective
- Convert the new cross-model win (`H108`) into a robust multi-seed claim or retire it quickly.
- Open one genuinely new topology direction (vineyards) rather than another additive tweak.
- Keep one cheap, broad exploratory slot with a materially changed motif construction.

## Required 3-Slot Execution Packet

1. Slot A (high-probability discovery): `N546` cross-model perturbation Jacobian alignment.
Primary metrics:
- `module_response_rank_spearman`
- `jacobian_subspace_mean_cosine` (or canonical-angle complement)
Robustness metric:
- `null_gap_q95_response_concordance` at domain level
Scope:
- seeds `{42,43,44}`
- domains `{immune, lung, external_lung}`
- perturbations: module dropout, sign-flip, local rewiring (same panel as `H108`)
Null package:
- perturbation-schedule permutation
- module-size/variance matched module shuffle
- random gene mapping
Keep gate:
- positive domain-level null-gap in `>=2/3` domains for at least `2/3` seeds
Fail-fast:
- if immune domain null-gap is negative in all three seeds, retire this rescue family

2. Slot B (high-risk/high-reward): `N539` perturbation persistence vineyards.
Primary metric:
- `delta_auc_vineyard_features_minus_h93`
Robustness metric:
- `null_gap_q95_delta_auc`
Scope:
- seed42 pilot
- domains `{immune, lung, external_lung}`
- splits `{source_disjoint, target_disjoint}`
- layers `{7,11}`
- perturbation strengths `{0,0.25,0.5,0.75,1.0}`
Null package:
- perturbation-schedule permutation
- degree-preserving local rewiring
- label permutation
Keep gate:
- positive mean null-gap in `>=3/6` domain-splits
Fail-fast:
- retire immediately if positive mean null-gap is `0/6`

3. Slot C (cheap broad-screen): `N551` biologically anchored finite-state grammar.
Primary metric:
- `delta_auc_biofsm_minus_h70`
Robustness metric:
- `null_gap_q95_delta_auc`
Scope:
- seed42
- domains `{immune, lung, external_lung}`
- splits `{source_disjoint, target_disjoint}`
- layers `{0,3,7,11}`
- state alphabet: TF-activity bin x support bin x sign-state
Null package:
- state-frequency matched token shuffle
- layer-order permutation
- label permutation
Keep gate:
- positive mean delta in `>=4/6` and positive mean null-gap in `>=2/6`
Fail-fast:
- stop lineage if positive mean delta `<3/6`

## Common Execution Rules
- Null budgets:
  - Slot A: `>=128` permutations per null family (domain-level cross-model gate is sensitive)
  - Slot B: `>=32` permutations per null family
  - Slot C: `>=24` permutations per null family
- Artifact contract (required for each slot):
  - by-row CSV
  - domain/split summary CSV
  - null summary CSV
  - machine-readable iteration summary JSON
- Promotion discipline:
  - no promotion without pre-registered null-gap gate pass
  - max one carry-over refinement slot in the following iteration

## Minimal Fallback Plan (use only if gate flips false)
1. Run Slot C with reduced null budget (`16`) to restore valid machine artifacts quickly.
2. Run Slot B on a single domain (`lung`) and layers `{7,11}` only.
3. Defer Slot A until gate returns `true`.
