# Brainstormer Structured Feedback - iter_0027

## Gate Status
- `passed_min_research_gate`: `true` (`iterations/iter_0027/executor_research_validation.json`).
- Recovery-only mode is not required for this pass.

## Iteration Readout (What Matters)
- `H64` failed decisively: mean `delta_AUROC(two-axis - baseline) = -0.03184`, positive rows `0/36`, positive mean domain-splits `0/6`.
- `H65` exposed endpoint mismatch: mean `null_gap_q95 = +0.13671` (positive in all `6/6` domain-splits) while mean `delta_AUROC(transfer - baseline) = -0.10204` (positive in `0/6`).
- `H66` is strongly negative everywhere: mean `delta_AUROC = -0.13176`, positive rows `0/54`, Fisher-significant domain-splits `0/6`.
- Practical implication: current loop is repeatedly finding structure that survives nulls but does not improve predictive utility; scoring objectives need to be utility-constrained by design.

## Stale Direction Triage
| Direction | Status | Evidence anchor | Next action |
|---|---|---|---|
| Two-axis persistence AUROC-lift endpoint (`H64` form) | `retire_now` | Uniformly negative across all domains/splits/layers. | Stop this endpoint form; only re-open with new endpoint (calibration/ranking or perturbation-linked). |
| Standalone ID endpoint lineage (`H54/H60/H63/H66`) | `retire_now` | Four consecutive ID variants are net negative or decisively negative. | Use ID only as interaction/gating signal, never as primary score. |
| Path-homology utility-transfer endpoint (`H53/H56`) | `retire_now` | Directional discrimination but repeated utility-transfer gate failure. | Keep closed unless endpoint and transfer target both change materially. |
| Directed/signed weighting rescue tweaks (`H58`, plus weighted zigzag coupling `H46`) | `retire_now` | Repeated failure to improve the known source-disjoint failure slices. | No more weight-polishing on the same representation. |
| Static curvature direct scoring (`H23/H61`) | `retire_now` | Multiple curvature surrogates remain net negative or fragile. | Re-open only as dynamic/interacting feature family. |
| Cross-model transfer utility objective (`H59/H62/H65` forms) | `rescue_once_with_major_change` | Directional/null-gap signals exist, but transfer utility is non-robust or negative. | One final redesign with explicit utility-regularized objective and cycle consistency; retire if still non-positive. |
| Bifiltration utility-coupling claim (`H49` endpoint form) | `rescue_once_with_major_change` | Discrimination signal is stable, but utility coupling remains weak/placebo-sensitive. | One redesign focused on ranking/calibration utility instead of raw AUROC lift. |

## Strategic Pivot
- Keep pressure on topology families that are structurally different from retired forms: rank-based multiparameter persistence surfaces, vineyards, and persistent-Laplacian features.
- In geometry, prioritize dynamic and multiscale descriptors (transition slopes, triangle-defect spectra, tangent transport mismatch) over static scalar scores.
- In cross-model, enforce dual constraints jointly (`null_gap > 0` and non-negative transfer utility) to prevent another robustness-without-utility failure.
- Add biological anchoring directly into filtrations and evaluation strata (TRRUST/GO/STRING/cell ontology), not only as post-hoc weighting.

## Recommended Next 3
1. High-probability discovery: `N329` (rank-based multiparameter persistence surface with utility gate).
2. High-risk/high-reward: `N338` (cycle-consistent utility-regularized cross-model transport).
3. Cheap broad-screen: `N335` (multiscale geodesic triangle-defect spectrum).
