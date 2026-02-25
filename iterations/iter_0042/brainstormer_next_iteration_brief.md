# Next Iteration Brief - iter_0042 -> iter_0043

## Research Gate
- Current status: `passed_min_research_gate = true`.
- Run full 3-slot packet.

## Packet Objective
- Convert the strongest directional near-miss (`H111` lineage) into a robust result or retire quickly.
- Open one materially new topology object (zigzag persistence) instead of another additive refinement.
- Preserve exploration breadth with one low-cost manifold screen.

## Required 3-Slot Execution Packet

1. Slot A (high-probability discovery): `N565` semi-Markov biologically anchored grammar.
Primary metrics:
- `delta_auc_semimarkov_biofsm_minus_h70`
- `null_gap_q95_delta_auc`
Scope:
- seeds `{42,43,44}`
- domains `{immune, lung, external_lung}`
- splits `{source_disjoint, target_disjoint}`
- layers `{0,3,7,11}`
- state factors: TF-activity bin x support bin x sign-state x dwell-time bin
Null package:
- occupancy+transition-count matched sequence shuffle
- layer-order permutation
- label permutation
Keep gate:
- positive mean null-gap in `>=2/6` domain-splits (stretch `>=3/6`)
- positive mean delta in `>=4/6` domain-splits
Fail-fast:
- if positive mean delta `<4/6` or immune+lung both remain null-gap negative in both splits, retire lineage.

2. Slot B (high-risk/high-reward): `N552` depth zigzag persistence.
Primary metrics:
- `delta_auc_zigzag_ph_minus_h93` (or `-h70` if no stable `H93` integration)
- `null_gap_q95_delta_auc`
Scope:
- seed42 pilot first
- domains `{immune, lung, external_lung}`
- splits `{source_disjoint, target_disjoint}`
- layers `{0,3,7,11}`
- edge-local complexes with support-weighted filtration
Null package:
- layer-order permutation
- local-complex node relabeling within degree bins
- label permutation
Keep gate:
- positive mean null-gap in `>=2/6` domain-splits
Fail-fast:
- retire if positive mean null-gap is `0/6` with near-zero/negative mean utility.

3. Slot C (cheap broad-screen): `N559` intrinsic-dimension hysteresis.
Primary metrics:
- `delta_auc_id_hysteresis_minus_h70`
- `null_gap_q95_delta_auc`
Scope:
- seed42
- domains `{immune, lung, external_lung}`
- splits `{source_disjoint, target_disjoint}`
- layers `{0,3,7,11}`
- neighborhood scales `k={4,6,8,10,12,16}`
Null package:
- radius-order permutation
- neighborhood assignment shuffle (degree-matched)
- label permutation
Keep gate:
- positive mean delta in `>=4/6` domain-splits and positive mean null-gap in `>=1/6`
Fail-fast:
- retire if positive mean delta `<=2/6`.

## Common Execution Rules
- Null budgets:
  - Slot A: `>=64` per null family (sequence null quality matters).
  - Slot B: `>=32` per null family.
  - Slot C: `>=24` per null family.
- Artifact contract per slot:
  - by-row CSV
  - domain/split summary CSV
  - null summary CSV
  - machine-readable iteration summary JSON
- Promotion rule:
  - no promotion without meeting pre-registered null-gap gate.

## Minimal Fallback Plan (only if gate flips false)
1. Run Slot C first with null budget `16` to re-establish valid machine artifacts quickly.
2. Run Slot A on one domain (`lung`) with both splits to test whether null redesign is working.
3. Defer Slot B until gate recovers.
