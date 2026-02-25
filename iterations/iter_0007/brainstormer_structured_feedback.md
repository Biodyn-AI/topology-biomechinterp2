# Brainstormer Structured Feedback — iter_0007

## Inputs Inspected
- `iterations/iter_0007/executor_iteration_report.md`
- `iterations/iter_0007/executor_hypothesis_screen.json`
- `iterations/iter_0007/executor_research_validation.json`
- New machine artifacts in `iterations/iter_0007/`:
  - `h1_immune_metric_matched_by_seed_layer.csv`
  - `h1_immune_metric_matched_layer_summary.csv`
  - `h1_immune_metric_matched_pass_matrix.csv`
  - `h1_immune_metric_matched_domain_summary.csv`
  - `h1_immune_metric_calibration_shift_summary.csv`
  - `iter0007_screen_summary.json`
  - `run_iter0007_screen.py`
- Cumulative context:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex`
  - `tracking/prompt.md`

## Research Gate
- `passed_min_research_gate`: `true`
- Executor artifacts are valid and complete for one full screening pass.

## Hypothesis Triage

### Promising
1. **Feature-shuffle topology branch remains the strongest live signal (cumulative).**
- Direct evidence: iter_0004 had `12/12` significant layers in both immune and external-lung; iter_0006 retained `16/24` significant immune layer-split tests under feature-shuffle, with dual-split passes at layers `7, 9, 10, 11`.
- Interpretation: persistent topological structure is still credible under weak-to-moderate nulls and disjoint splits, especially in late immune layers.

2. **Late-layer depth structure is repeatable and biologically actionable.**
- Direct evidence: dual-split feature-shuffle passes in immune are concentrated in late layers (`7, 9, 10, 11`), not uniformly distributed.
- Interpretation: depth-specific mechanism is a better target than global all-layer claims.

3. **Connectivity regime is a high-value explanatory lever for rewiring behavior.**
- Direct evidence (iter_0007 by-seed artifact): bridged rows have mean geodesic delta `-108.22` (`n=61`) vs non-bridged rows `-24.03` (`n=11`); all source rows are at `k=30`; overall `corr(k, delta_geo)=-0.332`.
- Interpretation: rewiring failure may be partly concentrated in high-k/bridge-heavy graph regimes; this is testable, not speculative.

### Neutral
1. **H10 (metric mismatch dominance) is directional but too small to change conclusions.**
- Direct evidence: mean calibration shift `(delta_geo - delta_euclid) = +0.180`, positive in `22/24` layer-split aggregates, but geodesic significance remains `0/24`.
- Interpretation: mismatch contributes slightly but is not the primary failure driver.

2. **Split robustness remains partial rather than global.**
- Direct evidence: iter_0006 dual-split feature-shuffle pass rate is `4/12` in immune.
- Interpretation: the branch is informative but still below broad-invariance threshold.

3. **Cross-model alignment remains unresolved at residual geometry level.**
- Direct evidence: iter_0003 feature-summary alignment was high-magnitude but non-significant (`Fisher p` around `0.35-0.41`), and residual-level matched tensors are still not fully integrated into this loop.
- Interpretation: keep active as a secondary branch once matched residual artifacts are available.

### Negative
1. **H09 (metric-matched rewiring rescue) is negative in immune.**
- Direct evidence: geodesic-vs-rewire significance `0/24`, minimum Fisher `p=0.6913`, dual-split pass `0/12`, mean geodesic delta `-95.356`.
- Interpretation: metric matching does not rescue rewiring-null support in current immune protocol.

2. **H07/H09 rewiring-survival branch is currently non-supportive across two consecutive iterations.**
- Direct evidence: iter_0006 `0/24` significant with mean deltas `-140.519` and `-129.702`; iter_0007 remains `0/24` after calibration.
- Interpretation: this branch should not be promoted without additional null-design controls.

3. **Distance-permutation stronger null remains over-adversarial for interpretation (historical negative control).**
- Direct evidence: iter_0005 distance-permutation yielded `0/12` significant and mean delta `-850.942`.
- Interpretation: deprioritize this null family for decision-driving conclusions.

## Decision-Oriented Guidance
1. Keep the rewiring-survival claim negative **for immune under current graph construction regime**.
2. Do not spend another full iteration on identical rewiring settings; prioritize one targeted calibration/control test that directly attacks the bridge/high-k confound.
3. In parallel, keep progress on the positive branch by anchoring late-layer robust topology (`7,9,10,11`) to biological programs.
