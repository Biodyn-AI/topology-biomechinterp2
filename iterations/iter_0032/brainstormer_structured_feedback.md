# Brainstormer Structured Feedback - iter_0032

## Gate Status
- `passed_min_research_gate`: `true` (from `executor_research_validation.json`).
- No recovery-only mode needed; proceed with discovery-first next loop.

## Iteration Readout (What Happened)
- `H79` (TF-module conditioned rescue): directional utility gain is real (`34/36` rows positive), but robustness is weak (`6/36` rows with positive `null_gap_q95`; only `1/6` domain-splits with positive mean null-gap).
- `H80` (pathway-centroid cross-model): all rows directional positive (`6/6` Spearman > 0), but robustness failed everywhere (`0/6` rows with positive `null_gap_q95_spearman`; `0/3` domains positive on mean null-gap).
- `H81` (neighbor-dropout detour elasticity v2): decisive negative (`1/24` positive delta rows; `0/6` domain-splits positive on mean delta and mean null-gap).

## Stale Direction Triage
- `cross_model_alignment` utility-promotion endpoints -> `retire_now`.
  - Evidence: last 10 iterations contain `6` negatives and `2` inconclusives for this family; latest `H80` still fails null-gap in all domains.
- Global detour/dropout elasticity as utility feature (`H78` -> `H81`) -> `retire_now`.
  - Evidence: null-gap stayed non-positive after major redesign; `H81` also shows feature degeneracy (`14/24` rows with zero targeted inflation).
- H70 support-interaction overlays (`H73`, `H76`, `H79`) -> `rescue_once_with_major_change`.
  - Evidence: base geometric lift persists, but interaction/null-gap gate repeatedly fails.
- Standalone ID-lift utility endpoints (`H60`, `H63`, `H66`) -> `retire_now`.
  - Evidence: repeated negative mean deltas and no robust split/domain support.
- Curvature-as-direct-score branch (`H75` after earlier curvature negatives) -> `rescue_once_with_major_change`.
  - Evidence: weak directional signal but universal null-gap failure in current parameterization.

## Active Backbone To Preserve
- Keep the H70 lineage as the discovery anchor (triangle-defect branch remains the strongest robust utility signal).
- Keep one topology-stability lane active (directed/signed and zigzag style tests showed prior positive evidence).

## What Should Change Immediately
- Stop spending slots on cross-model utility claims; if cross-model stays in the portfolio, switch endpoint to invariance/stability diagnostics instead of AUROC transfer claims.
- Replace global dropout perturbations with edge-local witness-path interventions (local causal geometry, not global node removal).
- For biology anchoring, test localized/conditional mechanisms (depth-conditional or motif-conditional), not global module-weight overlays.
