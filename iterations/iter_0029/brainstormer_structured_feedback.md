# Brainstormer Structured Feedback - iter_0029

## Gate Status
- `passed_min_research_gate`: `true` (`iterations/iter_0029/executor_research_validation.json`).
- Recovery-only mode is not required for this pass.

## Iteration Readout (What Matters)
- `H70` is the only robust positive branch: mean `delta_AUROC = +0.02637`, positive mean in `6/6` domain-splits, and mean matched-random-third `null_gap_q95 = +0.01010`.
- `H70` weakness is localized, not global: immune/source and immune/target means are small (`+0.00543`, `+0.01125`) with CIs crossing zero.
- `H71` is decisively negative: mean `delta_AUROC = -0.42758`, positive rows `0/12`, mean `null_gap_q95 = -0.14795`.
- `H72` is inconclusive and method-limited: mean `delta_AUROC = +0.00008`, no significant rows, and layer-order nulls are degenerate (identical deltas), so this implementation cannot support a mechanistic claim.

## Stale Direction Triage
| Direction | Status | Evidence anchor | Next action |
|---|---|---|---|
| Cross-model edge-utility transfer lineage (`H59`, `H62`, `H65`, `H68`, `H71`) | `retire_now` | Repeated inconclusive/negative outcomes; latest run is strongly negative in every domain-split. | Stop map-based edge transfer objective in current loop. |
| Standalone intrinsic-dimension AUROC-lift lineage (`H54`, `H60`, `H63`, `H66`) | `retire_now` | Four consecutive negatives with no rescue. | Keep ID only as moderator/covariate, not primary endpoint. |
| Multiparameter PH utility-lift lineage (`H64`, `H67`) | `retire_now` | Two major variants remained negative across all domain-splits. | Re-open only with a different endpoint (localization/stability, not direct lift). |
| Weight-polish rescue lineage (`H46`, `H58`) | `retire_now` | No rescue of known source-disjoint failure slices. | No more weighting tweaks without representation change. |
| Edge-trajectory motifs in current `H72` form | `rescue_once_with_major_change` | Signal is near zero and one null family is non-informative due order-invariant implementation. | One rerun only with non-degenerate temporal features + multiseed + harder nulls. |
| Cross-model family overall | `rescue_once_with_major_change` | Utility transfer failed, but earlier geometric-consistency results suggest latent shared structure. | One final relational-invariant alignment attempt; retire on failure. |

## Strategic Pivot
- Keep `H70` active, but shift next tests from "is there a signal" to "where is it biologically credible and where does it fail".
- Prioritize topology localization/stability methods (relative PH, persistent Laplacian, vineyard drift) over more global lift variants that are already stale.
- Reframe manifold geometry around dynamic quantities (curvature acceleration, tangent anisotropy, transport entropy), not static summary statistics.
- Restrict cross-model work to one high-risk representation-level attempt based on relational invariants instead of direct edge utility transfer.

## Recommended Next 3
1. High-probability discovery: `N368` (H70 support-concordance biology anchor test with hard controls).
2. High-risk/high-reward: `N365` (cross-model relational spectral alignment of topology signatures).
3. Cheap broad-screen: `N361` (geodesic curvature-acceleration scan across layers).

## Minimal Recovery Plan (if next gate unexpectedly fails)
1. Run `N361` on `seed42`, layers `{0,3,7,11}`, all domains/splits, with 24 permutations and a single summary CSV.
2. Run a reduced `N368` packet on `seed42`, layers `{7,11}`, with degree+coexpression-matched support shuffle.
3. Emit valid `executor_hypothesis_screen.json` and `iterXXXX_screen_summary.json` even if only two slots complete.
