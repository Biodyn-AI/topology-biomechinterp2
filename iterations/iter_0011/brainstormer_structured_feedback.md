# Brainstormer Structured Feedback — iter_0011

## Inputs Inspected
- `iterations/iter_0011/executor_iteration_report.md`
- `iterations/iter_0011/executor_hypothesis_screen.json`
- `iterations/iter_0011/executor_research_validation.json`
- New machine artifacts in `iterations/iter_0011/`:
  - `h16_module_structure_by_seed_layer_split.csv`
  - `h16_module_structure_layer_summary.csv`
  - `h16_module_structure_split_summary.csv`
  - `h17_cross_model_transfer_domain_summary.csv`
  - `h17_cross_model_transfer_global_null.csv`
  - `h17_cross_model_transfer_summary.json`
  - `h18_intrinsic_geodesic_metrics_by_seed_layer_split.csv`
  - `h18_intrinsic_geodesic_coupling_by_seed.csv`
  - `h18_intrinsic_geodesic_coupling_summary.csv`
  - `iter0011_screen_summary.json`
  - `run_iter0011_screen.py`
- Cumulative context:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex`
  - `tracking/prompt.md`

## Research Gate
- `passed_min_research_gate=true`.
- No gate-recovery run is needed for this iteration.

## Iteration Signal Assessment
1. `H16` (`module_structure`) is a valid promoted direction.
- Both splits are consistently above chance: source mean AUC `0.5387`, target mean AUC `0.5413`.
- All layers support the effect in both splits (`12/12` AUC>0.5; `12/12` Fisher-significant for AUC and rate-gap).
- Effect size is modest but stable, which is appropriate for follow-up via biological stratification rather than brute-force reruns.

2. `H17` (`cross_model_alignment`) is promising but fragile to feature-panel size.
- Positive transfer in all domains (`rho`: `1.0`, `0.5`, `1.0`) and top-feature match `3/3`.
- Exact global null is significant (`p_mean_rho=0.0369`, `p_top_match=0.0415`).
- Limitation is structural: only 3 shared features, so one matched panel expansion is mandatory before strong promotion.

3. `H18` (`intrinsic_dimensionality`) remains split-conditional.
- Target-disjoint is significant and direction-consistent across all seeds (`local_linearity +`, `participation_ratio -`).
- Source-disjoint is non-significant for both metrics (`|rho|=0.2354`, `p~0.17`).
- This should not be rerun in the same form; rescue requires richer intrinsic geometry descriptors.

## Stale Direction Triage

### `retire_now`
1. Rewiring-survival as a primary branch (`H07/H09/H12` lineage in `null_sensitivity`).
- Repeated calibrated negatives across multiple iterations with no rescue and persistent negative deltas.

2. Distortion-lower-tail as a rescue mechanism for rewiring failure.
- Repeated non-supportive outcomes with no directional recovery.

### `rescue_once_with_major_change`
1. Bridge-conditioned rewiring explanation (`H11`).
- Current evidence is split-confounded and not identifiable; allow only one fixed-k, within-split factorial rescue if explicitly prioritized.

2. Coarse disagreement-bin cross-model trend (`H15` form).
- Domain-heterogeneous sign suggests confounding; next attempt must be per-edge with degree/prevalence controls.

3. Mean-only intrinsic coupling (`H04/H18` form).
- Simple rank coupling is unstable; next attempt should use local-ID distribution shape (variance/IQR/skew) and mixed-effects models.

## Navigation Guidance for iter_0012
1. Spend next budget on mechanistic interpretation of already-positive branches (`H13`, `H14`, `H16`, `H17`) rather than old rewiring stress tests.
2. Tie geometry/topology to external biological priors early (DoRothEA/TRRUST/STRING/Cell Ontology) so signals become decision-useful.
3. For cross-model work, prioritize structure transfer and expanded feature matching over additional low-dimensional summary correlations.

## Minimal Recovery Plan (future contingency only)
- If a future executor gate fails, run one compact continuity packet:
1. Execute one low-cost mechanistic screen with one explicit null (recommended: local reconstruction error vs geodesic gain).
2. Emit mandatory artifacts: one machine CSV, `executor_hypothesis_screen.json`, and short reproducible report.
3. Resume high-cost branches only after gate validity is restored.
