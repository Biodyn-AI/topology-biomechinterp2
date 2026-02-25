# Brainstormer Structured Feedback — iter_0014

## Inputs Inspected
- `iterations/iter_0014/executor_iteration_report.md`
- `iterations/iter_0014/executor_hypothesis_screen.json`
- `iterations/iter_0014/executor_research_validation.json`
- New machine artifacts in `iterations/iter_0014/`:
  - `h25_diffusion_distance_by_seed_layer_split.csv`
  - `h25_diffusion_distance_domain_summary.csv`
  - `h25_diffusion_distance_null_summary.csv`
  - `h26_bio_anchor_edge_table.csv`
  - `h26_bio_anchor_model_summary.csv`
  - `h26_bio_anchor_permutation_null.csv`
  - `h27_gw_alignment_domain_summary.csv`
  - `h27_gw_alignment_null_summary.csv`
  - `h27_gw_alignment_map_quality.csv`
  - `iter0014_screen_summary.json`
  - `run_iter0014_screen.py`
- Cumulative context:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex`

## Research Gate
- `passed_min_research_gate=true`.
- No gate-recovery patch is required for this iteration.

## Iteration Signal Assessment
1. `H25` is the strongest active branch and now has cross-domain support.
- Mean best diffusion uplift over best baseline is `+0.0173`.
- Domain deltas are positive in all domains (immune `+0.0161`, lung `+0.0249`, external-lung `+0.0109`).
- Domain Fisher tests are significant in `3/3` domains.

2. `H26` is not validating the interaction claim in current form.
- Interaction term is significant in `0/6` domain-split rows and positive in only `1/6`.
- Full-vs-baseline calibration gain is small (`+0.00068` mean) but non-zero in aggregate.
- Interpretation: useful as calibration feature engineering, not yet a mechanistic interaction finding.

3. `H27` shows a geometry-correspondence gap.
- Coarse geometry transfer is strong (distance Spearman significant in `3/3` domains).
- Correspondence recovery collapses (mean top-1 `0.00119`, combined `p=0.990`).
- Edge transfer is weak-borderline (`mean AUROC=0.5186`), with only immune significant.

## Stale Direction Triage

### `retire_now`
| Direction | Evidence | Action |
|---|---|---|
| Rewiring-null survival lineage (`H07/H09/H12`) | Repeated calibrated negatives across `iter_0006`-`iter_0008` with no rescue trend | `retire_now` |
| Distortion-lower-tail rewiring rescue | Repeated non-significance and no directional correction | `retire_now` |
| Raw Forman negative-curvature enrichment (`H23` as implemented) | Below-chance AUROC and opposite-direction enrichment in all domains | `retire_now` |
| Plain unsupervised correspondence-free map recovery (Hungarian OT + unseeded GW as default) | `H20` OT and `H27` GW both fail top-1 recovery despite strong geometry similarity | `retire_now` |

### `rescue_once_with_major_change`
| Direction | Why not retired yet | Required major change | Action |
|---|---|---|---|
| Geometry x prior interaction (`H26` form) | Small but real calibration gain exists | Add independent STRING support + ontology strata + consensus-prior modeling | `rescue_once_with_major_change` |
| Intrinsic branch framed as domain-invariant sign (`H04/H18/H21/H22`) | Repeated heterogeneity suggests regime effects, not pure null | Model domain x split x phase interactions with anisotropy features | `rescue_once_with_major_change` |
| Cross-model unseeded transport | Coarse manifold agreement is strong, correspondence is not | Use CCA-seeded + one-to-one-regularized GW with cycle-consistency checks | `rescue_once_with_major_change` |
| Coarse disagreement-bin trend (`H15` style) | Aggregated signal exists but sign flips by domain | Replace bins with per-edge mixed models and prevalence controls | `rescue_once_with_major_change` |

## Navigation Guidance for Next Loop
1. Exploit the strongest confirmed signal (`H25`) with tougher nulls before branching to expensive new methods.
2. Push one major cross-model rescue with seeded constrained transport rather than repeating unseeded OT/GW.
3. Keep one low-cost, broad geometry screen in the packet to guarantee a valid artifact even if the high-risk run fails.

## Minimal Recovery Plan (Only if a Future Gate Fails)
1. Run only the diffusion robustness check (coexpression-matched null) on one domain with both disjoint splits.
2. Emit mandatory files: `executor_iteration_report.md`, `executor_hypothesis_screen.json`, and one machine CSV summary.
3. Resume full 3-hypothesis packet after validation gate is restored.
