# Brainstormer Structured Feedback - iter_0030

## Gate Status
- `passed_min_research_gate`: `true` (`iterations/iter_0030/executor_research_validation.json`).
- Recovery-only mode is not required for this pass.

## Iteration Readout (What Matters)
- `H73` confirms the core `H70` geometry signal persists (`mean delta_AUROC=+0.02498`), but the tested support-concordance interaction is not credible (`mean interaction=-0.00032`, interaction `null_gap_q95` mean negative, `0/6` domain-splits null-surviving).
- `H74` is another cross-model edge-transfer failure (`mean null_gap_q95=-0.09881`), with a hard immune collapse (`immune mean delta=-0.16020`).
- `H75` has weak directional signal (`mean delta_AUROC=+0.00210`) but zero null-gap survival (`0/6` positive), so it cannot be promoted.
- Most informative machine-level pattern this iteration: H73/H75 effects can look positive by raw delta but collapse under null-gap criteria.

## Stale Direction Triage
| Direction | Status | Evidence anchor | Next action |
|---|---|---|---|
| Cross-model **edge-utility transfer** endpoints (`H29/H33/H59/H62/H65/H68/H71/H74`) | `retire_now` | Repeated utility-negative or null-gap-negative outcomes across major redesigns. | Stop spending on edge-transfer AUROC endpoint; only revisit with non-edge relational endpoint. |
| Standalone intrinsic-dimension lift endpoints (`H42/H54/H60/H63/H66`) | `retire_now` | Consecutive negatives with no stable rescue domain. | Keep ID only as interaction/covariate terms. |
| Rewiring-survival null branch (`iter_0005`-`iter_0008`) | `retire_now` | Uniformly non-supportive despite calibration variants. | Keep closed unless graph-construction regime is materially changed. |
| Two-axis/rank-surface persistence lift forms (`H64/H67`) | `retire_now` | Back-to-back all-split negative utility deltas. | Reopen only for localization/stability endpoint, not direct AUROC lift. |
| Support-interaction anchoring in current formulation (`H58/H73`) | `rescue_once_with_major_change` | Prior support branch showed signal (`H40/H43`), but current interaction implementation fails null robustness. | One coexpression-aware, cell-ontology-aware redesign; retire if null-gap still fails. |
| Curvature family as currently parameterized (`H23/H61/H75`) | `rescue_once_with_major_change` | Static and acceleration variants are negative/inconclusive and null-gap weak. | One redesign using discrete Ricci/transport geometry, then retire if non-robust. |
| Motif trajectory pilot in current form (`H72`) | `rescue_once_with_major_change` | Near-zero effect and weak/degenerate null informativeness in prior form. | One state-machine redesign with non-degenerate temporal nulls. |

## Strategic Pivot for Next Loop
- Treat `H70` as the anchor and test mechanisms that explain **where** it works (coexpression/cell-ontology strata), not whether it exists.
- Shift cross-model work away from edge utility transfer to relational agreement metrics (rank consistency, motif-distribution similarity, cycle consistency).
- In geometry, prioritize measures that are inherently structural (Ricci flow, detour elasticity, tangent transport inconsistency) rather than another small-weight feature blend.
- Keep at least one cheap breadth probe each loop to avoid local overfitting to the `H70` lineage.

## Top-3 Recommendation Snapshot
1. High-probability discovery: coexpression-aware support-concordance interaction v2 on `H70` (`N382`).
2. High-risk/high-reward: cross-model relational agreement endpoint (non-edge transfer) (`N379`).
3. Cheap broad-screen: geodesic detour-elasticity screen (`N376`).

## Minimal Recovery Plan (only if a future gate fails)
1. Run `N376` on `seed42`, layers `{0,3,7,11}`, all domains/splits, 24 permutations.
2. Run reduced `N382` on immune + lung at layers `{7,11}` with geodesic/degree/coexpression strata.
3. Emit valid `executor_hypothesis_screen.json` and `iterXXXX_screen_summary.json` even if only two slots complete.
