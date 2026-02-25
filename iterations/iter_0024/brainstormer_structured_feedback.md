# Brainstormer Structured Feedback - iter_0024

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0024/executor_research_validation.json`).
- Implication: run a full discovery packet next iteration; no gate-recovery-only cycle is required.

## Iteration Readout (What Matters)
- `H55` is the only strong live lead: mean `delta_AUROC=+0.01169`, `25/36` positive rows, Fisher-significant in `6/6` domain-splits under higher-permutation nulls.
- `H55` failure is now localized, not global: source-disjoint negatives are concentrated in `lung` (`-0.00315`) and `external_lung` (`-0.00440`) and track weaker margin concentration / sign structure.
- `H56` failed its defining objective for a second path-homology cycle (`H53` then `H56`): discrimination moved positive, but transfer utility remained exactly zero.
- `H57` failed broad-screen keep criteria: net incremental value vs geodesic baseline is negative (`-0.01779`) and only `3/6` domain-splits are directionally positive.

## Stale Direction Triage

### `retire_now`
1. Directed path-homology utility-transfer line (`H53` + `H56`) as currently defined.
2. Standalone geodesic anisotropy-tail endpoint (`H57`) as currently defined.
3. Rewiring-survival lineage (`H05-H12`) after repeated calibrated negatives.
4. GW-first correspondence recovery (`H27`, `H29`) after repeated map/transfer failure.
5. Local-linearity rupture endpoint (`H54`) and weighted-zigzag coupling endpoint (`H46`).
6. Motif-overlap-as-endpoint formulation (`H48`, `H51`) without module-shuffle support.
7. Current OOS intrinsic-dimension `delta_R2` objective (`H42`, `H45`).

### `rescue_once_with_major_change`
1. Bifiltration utility claim (`H49`): one rescue only if endpoint changes from threshold-transfer to ranking calibration transfer.
2. Curvature family (`H23`, `H30`): one rescue only as interaction terms with support/detour, not standalone score.
3. Cross-model utility alignment (`H33`, `H36`): one rescue only if objective is topology-signature transfer, not correspondence recovery.

## Navigation Rules for iter_0025
1. Keep one slot on the strongest positive branch (`H55`) but force a biologically anchored failure-slice test.
2. Use one slot for a genuinely new geometric/topological mechanism, not a parameter tweak on retired negatives.
3. Require every candidate to include both a geometry-preserving null and a biology-anchor falsification control.
4. Predefine continuation gates before running to avoid post-hoc justification.

## Selected Immediate Candidates
- High-probability discovery: `N255` (biologically weighted directed/signed failure-slice rescue; proposed `H58`).
- High-risk/high-reward: `N263` (cross-model persistence-diagram transfer without correspondence; proposed `H59`).
- Cheap broad-screen: `N262` (intrinsic-dimension jump screen on edge endpoints; proposed `H60`).
