# Brainstormer Structured Feedback - iter_0023

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0023/executor_research_validation.json`).
- Implication: proceed with a full 3-slot hypothesis packet; no gate-recovery-only run is required.

## Iteration Readout (What Matters)
- `H52` is the active winner and now the portfolio anchor: mean `delta_AUROC=+0.01461`, positive in `58/72` rows, Fisher-significant in `6/6` domain-split aggregates.
- `H52` has one clear failure slice that must be diagnosed, not ignored: `lung/source_disjoint` mean `delta=-0.00359` with seed concentration (`seed43` and `seed44` mostly negative) and strongest weakness at layers `7` and `11`.
- `H53` is not yet a productive branch in its current form: pilot signal is weak (`+0.00276`) and non-robust (`0/6` Fisher-significant aggregates).
- `H54` is decisively negative and should be closed as formulated: mean `delta=-0.04527`, strong failures in immune and lung, and only one weakly positive domain-split pocket.

## Stale Direction Triage

### `retire_now`
1. Local-linearity rupture endpoint (`H54`) - clear negative effect size and broad cross-domain underperformance.
2. Rewiring-survival lineage (`H05-H12`) - repeated negatives under multiple calibrated stronger-null redesigns.
3. GW-first correspondence recovery (`H27`, `H29`) - repeated failure on both mapping quality and transfer utility.
4. Raw curvature/thinness direct-score variants (`H23`, `H30`) - repeatedly below geometry baselines.
5. Weighted-zigzag coupling objective (`H46`) - explicit objective failure (`0/36` weighted wins).
6. Motif-overlap-as-endpoint objective (`H48`, `H51`) - mixed descriptive signal without module-level causal support.
7. Current OOS ID `delta_R2` formulation (`H42`, `H45`) - unstable and contradictory holdout behavior.

### `rescue_once_with_major_change`
1. `H53` path-homology branch - only if endpoint changes to utility/transfer and directed complex construction is materially denser.
2. `H49` utility-coupling claim - one redesign with explicit held-out utility target and placebo-calibrated continuation gate.
3. Curvature family - only as interaction terms with support/detour/geodesic context, not standalone scores.

## Navigation Guidance
- Keep one slot on `H52` escalation because it is currently the only robust positive topology branch with cross-domain support.
- Do not spend a full slot on any branch that has already failed twice without a changed objective.
- Force each new candidate to include both: 
  - a topology/geometry-preserving null,
  - and one biology-anchor falsification control (TRRUST/GO/STRING/cell-ontology shuffle or matched-set randomization).

## Selected Immediate Candidates
- High-probability discovery: `N241` (`H55`) - high-permutation directed/signed replication plus targeted `lung/source_disjoint` failure diagnosis.
- High-risk/high-reward: `N244` (`H56`) - path-homology v2 with densified directed complex and utility-first objective.
- Cheap broad-screen: `N247` (`H57`) - local geodesic anisotropy-tail features as a low-cost manifold screen.
