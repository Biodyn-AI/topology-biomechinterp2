# Brainstormer Structured Feedback - iter_0028

## Gate Status
- `passed_min_research_gate`: `true` (`iterations/iter_0028/executor_research_validation.json`).
- Recovery-only mode is not required for this pass.

## Iteration Readout (What Matters)
- `H67` failed decisively despite major redesign: mean `delta_AUROC(rank_surface-baseline) = -0.03048`, positive mean domain-splits `0/6`.
- `H68` failed utility and robustness together: mean `delta_AUROC(transfer-baseline) = -0.30464`, mean `null_gap_q95 = -0.03847`, immune null-gap strongly negative.
- `H69` is the only live signal: mean `delta_AUROC(triangle_defect-baseline) = +0.02617`, positive rows `30/36`, positive mean domain-splits `6/6`.
- Limiting factor is statistical resolution, not direction: `H69` uses `8` permutations/null family (`p` floor `0.111`), and matched-random-third remains the hard control (null mean `+0.01350`).

## Stale Direction Triage
| Direction | Status | Evidence anchor | Next action |
|---|---|---|---|
| Multiparameter persistence utility-lift lineage (`H64`, `H67`) | `retire_now` | Two consecutive major variants remain uniformly negative on utility and failure slices. | Stop AUROC-lift endpoint forms; only re-open with new endpoint type (calibration/ranking/perturbation sensitivity). |
| Cross-model global mapping utility-transfer lineage (`H59`, `H62`, `H65`, `H68`) | `retire_now` | Four runs are inconclusive-to-negative, latest is strongly negative in all domain-splits. | Retire global-map transfer objective in current form. |
| Standalone intrinsic-dimension endpoint lineage (`H54`, `H60`, `H63`, `H66`) | `retire_now` | Repeated negative outcomes with no rescue signal. | Keep ID only as interaction/gating covariate. |
| Path-homology utility-transfer endpoint (`H53`, `H56`) | `retire_now` | Directional discrimination without utility-transfer success. | Do not spend another slot on this endpoint form. |
| Weight-polish rescues (`H46`, `H58`) | `retire_now` | Did not fix known source-disjoint failures. | No more weighting tweaks on unchanged representations. |
| Biological anchoring as coefficient-only endpoint (`H40`, `H43`) | `rescue_once_with_major_change` | Interaction coefficients are reproducibly positive, but direct utility lift is near zero. | One redesign where biological anchoring is part of representation/filtering, not only model terms. |
| Cross-model branch as a whole | `rescue_once_with_major_change` | Legacy global mapping failed, but earlier geometric consistency (`H20/H24`) suggests latent transferable structure exists. | One final representation-first transfer attempt (topology-signature distillation), then retire if non-positive. |

## Strategic Pivot
- Keep `H69` and make it falsifiable quickly with higher null resolution and hard-null stratification.
- Shift topology work toward dynamic/multiscale summaries (vineyards, persistent Laplacian, defect-weighted filtrations), not another direct persistence-lift rerun.
- Push cross-model to representation transfer of topology signatures, not direct embedding-map transfer.
- Force biological anchoring into filtration axes or stratified alignment objectives.

## Recommended Next 3
1. High-probability discovery: `N343` (H69 robustness + hard-null beat-down with higher permutation budget).
2. High-risk/high-reward: `N350` (cross-model topology-signature distillation transfer).
3. Cheap broad-screen: `N355` (edge trajectory motif classes from layerwise geometry/topology traces).
