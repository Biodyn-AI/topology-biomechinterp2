# Brainstormer Structured Feedback - iter_0031

## Gate Status
- `passed_min_research_gate`: `true` (`iterations/iter_0031/executor_research_validation.json`).
- Recovery-only mode is not required for this pass.

## Iteration Readout (Action-Relevant)
- `H76` is still not promotable: geometry lift persists (`mean delta_AUROC=+0.02323`) but anchoring interaction is near-zero (`+0.00041`) with only `1/6` domain-splits null-surviving; immune/source remains negative (`-0.00460`).
- `H77` is decisively negative: endpoint change to relational ranks produced effectively zero effect (`delta_spearman~0`) and `0/12` positive null-gap rows.
- `H78` is directionally useful but not robust: small positive breadth signal (`4/6` positive domain-split means) but null-gap stays negative in all domain-splits (`0/6`).
- Most important mechanistic clue: in `H76`, layer `11` has positive interaction mean while layer `7` is negative, suggesting depth-conditional signal instead of global interaction.

## Stale-Direction Triage
| Direction | Status | Evidence anchor | Action |
|---|---|---|---|
| Global cross-model mapping endpoints (`H59/H62/H65/H68/H71/H74/H77`) | `retire_now` | Multiple major redesigns, still null-gap negative or utility-negative. | Stop global map objectives; only re-enter with biology-stratified invariants. |
| Rewiring-survival stronger-null branch (`iter_0005`-`iter_0008`) | `retire_now` | Uniform non-support under calibrated variants. | Keep closed unless graph-construction regime changes materially. |
| Standalone intrinsic-dimension lift endpoints (`H42/H54/H60/H63/H66`) | `retire_now` | Repeated negatives with no stable rescue slice. | Keep ID only as interaction/covariate features. |
| Two-axis/rank-surface persistence lift (`H64/H67`) | `retire_now` | Consecutive all-split negatives. | Re-open only for localization/stability endpoints. |
| Support-interaction recipe as currently parameterized (`H73/H76`) | `rescue_once_with_major_change` | Core geometry survives, but interaction objective fails robustness. | One targeted immune/source redesign with TF-module conditioning. |
| Curvature-as-score family (`H23/H61/H75`) | `rescue_once_with_major_change` | Static/acceleration versions are weak or negative under null-gap. | One Ricci-flow/transport redesign, then retire if still non-robust. |
| Detour-elasticity current recipe (`H78`) | `rescue_once_with_major_change` | Directional signal but universal null-gap failure. | Keep only as low-cost probe with explicit neighbor-dropout perturbation. |

## Strategic Pivot
- Keep `H70` lineage as anchor, but target the known failure slice directly (immune/source) instead of global interaction averages.
- Push topology work toward stability/filtration agreement metrics, not another direct AUROC-lift variant.
- Keep exactly one cross-model shot, but make it pathway/stratum-level and edge-free.
- Keep one cheap breadth probe each loop to avoid local overfitting to one family.

## Recommended Top 3 Snapshot
1. High-probability discovery: `N395` (immune/source TF-module conditioned biological anchoring rescue).
2. High-risk/high-reward: `N392` (cross-model pathway-centroid manifold alignment).
3. Cheap broad-screen: `N389` (neighbor-dropout geodesic robustness / detour elasticity v2).

## Minimal Recovery Plan (only if a future gate fails)
1. Run `N389` only (seed42, all domains/splits, layers `{0,3,7,11}`, 24 permutations).
2. Run reduced `N395` on immune/lung, layers `{7,11}`, with module-label permutation control.
3. Emit valid `executor_hypothesis_screen.json` and `iterXXXX_screen_summary.json` even if only two slots complete.
