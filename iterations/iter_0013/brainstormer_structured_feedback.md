# Brainstormer Structured Feedback — iter_0013

## Inputs Inspected
- `iterations/iter_0013/executor_iteration_report.md`
- `iterations/iter_0013/executor_hypothesis_screen.json`
- `iterations/iter_0013/executor_research_validation.json`
- New machine artifacts in `iterations/iter_0013/`:
  - `h22_phase_transition_by_seed_layer_split.csv`
  - `h22_phase_transition_phase_means.csv`
  - `h22_phase_transition_model_summary.csv`
  - `h22_phase_transition_null_summary.csv`
  - `h23_curvature_enrichment_by_seed_layer_split.csv`
  - `h23_curvature_enrichment_split_summary.csv`
  - `h23_curvature_enrichment_domain_summary.csv`
  - `h24_cross_model_cca_domain_summary.csv`
  - `h24_cross_model_cca_null_summary.csv`
  - `h24_cross_model_cca_overall_summary.csv`
  - `iter0013_screen_summary.json`
  - `run_iter0013_screen.py`
- Cumulative context:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex`

## Research Gate
- `passed_min_research_gate=true`.
- No gate-recovery patch is required for this iteration.

## Iteration Signal Assessment
1. `H24` is the strongest active branch and now has multi-metric, cross-domain support.
- Mean canonical correlation `0.7968`.
- Distance Spearman / kNN Jaccard / top-1 retrieval are significant in `3/3` domains (`combined Fisher p=3.17e-05` for each metric family).
- CCA beats PCA baseline in every domain for all three transfer metrics.

2. `H23` is directionally negative, not just underpowered.
- Domain mean AUROC for negative-curvature score is below chance in all domains (`0.3406`, `0.3894`, `0.3905`).
- Top-vs-bottom curvature-bin positive-rate deltas are negative in all domains (`-0.3062`, `-0.1826`, `-0.1941`).
- This exact curvature definition should be considered closed.

3. `H22` is domain-conditional and currently not promotable.
- Immune has a clear late negative split difference (`late diff=-0.0215`, `p=7.5e-4`).
- Lung and external-lung fail directional replication (late diffs `+0.0037` and `+0.0112`).
- Cross-domain gate was missed (`1/3` domains negative in late phase).

## Stale Direction Triage

### `retire_now`
| Direction | Evidence | Action |
|---|---|---|
| Rewiring-null survival branch (`H07/H09/H12`) | Repeated calibrated negatives across `iter_0006`-`iter_0008`; no supportive trend | `retire_now` |
| Rewiring distortion-lower-tail rescue sub-branch | Repeated non-significance and no directional rescue | `retire_now` |
| Plain Hungarian OT unsupervised map | `iter_0012` top-1 near zero (`0.0024`) with `0/3` significant domains | `retire_now` |
| Raw Forman negative-curvature enrichment (`H23` as implemented) | Strong, consistent opposite-direction results in `iter_0013` | `retire_now` |

### `rescue_once_with_major_change`
| Direction | Why not retired yet | Required major change | Action |
|---|---|---|---|
| Intrinsic-dimensionality positive-coupling storyline (`H04/H18/H21/H22`) | Repeated neutral/mixed outcomes with structured heterogeneity suggest mechanism mismatch, not total absence | Reframe as sign-heterogeneous regime model (domain x split x phase) with biological-anchor interactions | `rescue_once_with_major_change` |
| Confidence-tier monotonicity (`H19`) | Direction failed, but tier prevalence/confidence saturation confounds are real | Replace raw tier slope with prevalence-adjusted multi-prior model (TRRUST + STRING + GO) | `rescue_once_with_major_change` |
| Bridge-conditioned graph-topology explanation (`H11`) | Prior run had identifiability failure due split-confounded bridge strata | Fixed-k protocol that forces bridge/non-bridge support in both splits before any claim | `rescue_once_with_major_change` |
| Coarse disagreement-bin trend (`H15`) | Domain-heterogeneous sign may hide real per-edge effects | Move to per-edge mixed models with explicit domain interaction and prevalence controls | `rescue_once_with_major_change` |

## Navigation Guidance for Next Loop
1. Allocate most budget to cross-model structure transfer/alignment and biological anchoring of the already-positive geometry signal.
2. Keep one cheap geometry screen in every loop to avoid high-cost single-point failure.
3. Do not rerun retired branches without an explicitly different method definition and control set.

## Minimal Recovery Plan (Only if a Future Gate Fails)
1. Run one low-cost screen (`N105` diffusion-distance triage) on one domain with both disjoint splits.
2. Emit required files (`executor_iteration_report.md`, `executor_hypothesis_screen.json`, and at least one machine CSV summary).
3. Re-enter full portfolio only after validation gate is restored.
