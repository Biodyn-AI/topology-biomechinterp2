# Brainstormer Structured Feedback — iter_0010

## Inputs Inspected
- `iterations/iter_0010/executor_iteration_report.md`
- `iterations/iter_0010/executor_hypothesis_screen.json`
- `iterations/iter_0010/executor_research_validation.json`
- New machine artifacts in `iterations/iter_0010/`:
  - `h13_manifold_distance_by_seed_layer_split.csv`
  - `h13_manifold_distance_layer_summary.csv`
  - `h13_manifold_distance_pass_matrix.csv`
  - `h13_manifold_distance_split_summary.csv`
  - `h14_topology_stability_bootstrap_records.csv`
  - `h14_topology_stability_filtration_layer_summary.csv`
  - `h14_topology_stability_filtration_sensitivity.csv`
  - `h14_topology_stability_layer_summary.csv`
  - `h14_topology_stability_seed_layer_setting_summary.csv`
  - `h15_cross_model_disagreement_summary.json`
  - `h15_cross_model_disagreement_trend.csv`
  - `iter0010_screen_summary.json`
  - `run_iter0010_screen.py`
- Cumulative context:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex`
  - `tracking/prompt.md`

## Research Gate
- `passed_min_research_gate`: `true`
- Iteration is valid for forward hypothesis expansion; no gate-recovery patch is required.

## Iteration Signal Assessment
1. `H13` is a real forward path, not noise.
- Dual-split positive in all layers (`12/12`) with dual-split significance in `7/12`.
- Strongest combined layers: `4`, `8`, `9` by mean source/target AUROC delta.

2. `H14` is robust under bounded perturbations.
- Positive mean H1 deltas in `12/12` layers; combined Fisher significance in `12/12`.
- Filtration robustness is strong (`all_settings_positive_fraction=1.0`, mean delta-range `3.285`).

3. `H15` remains unresolved but not dead.
- Lung is strongly negative (`rho=-0.9758`, `p=3.33e-4`), immune flips sign (`rho=+0.4012`, `p=0.2496`).
- This is a heterogeneity question, not a global monotonic law yet.

## Stale Direction Triage

### `retire_now`
1. Rewiring-survival as a primary branch (`null_sensitivity` H07/H09/H12 lineage).
- Repeated strong negatives across three consecutive calibrated iterations (`iter_0006` to `iter_0008`) with zero significant rescue and persistent negative deltas.
- Further same-family reruns are low expected value.

2. Distortion-lower-tail rescue claim under rewiring controls.
- Also repeatedly non-supportive (`0/24` in iter_0008; no directional rescue in prior iterations).

### `rescue_once_with_major_change`
1. Bridge-conditioned rewiring explanation (`graph_topology` H11).
- Current evidence is confounded by split (`source` almost always bridged; `target` mostly not).
- Only one more attempt is justified, and only with within-split bridge-identifiable factorial design.

2. Cross-model alignment trend branch (`cross_model_alignment` H02/H15 lineage).
- Coarse-bin statistics are unstable and domain-heterogeneous.
- Rescue requires per-edge models with degree/prevalence controls and matched-gene structure transfer, not another bin-only rerun.

3. Intrinsic-dimension coupling (`H04` style simple rank correlation).
- Mixed once and then stalled.
- Rescue only if upgraded to local-ID distributional features plus mixed-effects controls.

## Directional Guidance for Next Loop
1. Spend budget on new geometry/topology signatures that explain why `H13` and `H14` are positive, not on more rewiring-survival diagnostics.
2. Tie new tests to biological anchors early (TRRUST/DoRothEA/STRING/Cell Ontology) so positive geometry translates to mechanistic value.
3. Resolve `H15` with controlled per-edge analysis; this is a salvageable branch with clear upside.

## Gate-Failure Recovery Note
- Not needed for `iter_0010` because gate passed.
- A minimal fallback plan is still provided in `brainstormer_next_iteration_brief.md` for future failure cases.
