# Brainstormer Structured Feedback — iter_0006

## Inputs Inspected
- `iterations/iter_0006/executor_iteration_report.md`
- `iterations/iter_0006/executor_hypothesis_screen.json`
- `iterations/iter_0006/executor_research_validation.json`
- New machine artifacts in `iterations/iter_0006/`:
  - `h1_immune_rewire_split_by_seed_layer.csv`
  - `h1_immune_rewire_split_layer_summary.csv`
  - `h1_immune_rewire_split_pass_matrix.csv`
  - `h1_immune_rewire_split_domain_summary.csv`
  - `h1_immune_rewire_dual_split_summary.csv`
  - `iter0006_screen_summary.json`
- Cumulative context:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex`
  - `tracking/prompt.md`

## Research Gate
- `passed_min_research_gate`: `true`
- Executor output quality checks passed; no recovery action required for this iteration.

## Hypothesis Triage

### Promising
1. **Feature-shuffle topology signal remains strong in immune source-disjoint split.**
- Direct evidence: source-disjoint under `feature_shuffle` is `12/12` significant with mean layer delta `+6.646`.
- Interpretation: immune topology signal is not a narrow single-layer artifact; it survives broad source-side disjoint sampling.

2. **Late-layer dual-split robustness cluster is repeatably visible.**
- Direct evidence: `feature_shuffle` dual-split significant layers are `[7, 9, 10, 11]` (`4/12`).
- Interpretation: robustness concentrates at depth, consistent with a layer-structured mechanism rather than uniform topology across depth.

3. **Depth asymmetry itself is a useful mechanistic lead.**
- Direct evidence: target-disjoint has only `4/12` significant layers but `9/12` positive deltas; strongest target support appears at layers `9-11`.
- Interpretation: many layers are directionally positive but underpowered on target split; this suggests signal dilution, not complete absence.

### Neutral
1. **Split robustness is partial, not global.**
- Direct evidence: only `4/12` layers pass both splits; target-disjoint mean delta is `+0.875`.
- Interpretation: branch is informative but below promotion threshold for broad invariance claims.

2. **Robustness boundary layers are plausible follow-up targets.**
- Direct evidence: layer `8` target Fisher `p=0.075` (near threshold), layer `6` target Fisher `p=0.132`.
- Interpretation: small power or calibration changes may convert near-miss layers, so binary pass/fail is currently unstable near threshold.

3. **Cross-domain status of this full-layer split pattern is unresolved.**
- Direct evidence: iter_0006 tested immune only.
- Interpretation: cannot yet claim domain-general depth asymmetry under this exact protocol.

### Negative
1. **Rewiring-null survival hypothesis failed in the current implementation regime.**
- Direct evidence: `degree_preserving_geodesic_rewire` has `0/24` significant layer-tests, mean deltas `-140.519` (source) and `-129.702` (target).
- Interpretation: H07 is negative as currently parameterized.

2. **Current rewiring null appears calibration-mismatched to the observed metric.**
- Direct evidence: observed PH is computed from Euclidean point clouds, but rewiring null uses geodesic distances; rewiring p-values are uniformly `1.0` at seed-layer level.
- Interpretation: this is likely too adversarial for interpretability screening and should be recalibrated before broad scientific conclusions.

3. **Graph connectivity diagnostics indicate a fragile null construction regime.**
- Direct evidence: component bridging used in `142/144` rows; source split uses `k=30` in all rows.
- Interpretation: conclusions under rewiring null are entangled with near-connectivity-boundary behavior.

## What To Do Next (Decision-Oriented)
1. Prioritize a **metric-matched null calibration** (observed geodesic vs rewired geodesic) before any further promote/reject decision on stronger-null survival.
2. Replicate full-layer split mapping in **external-lung** to test whether the late-layer dual-split cluster is immune-specific or cross-domain.
3. Run **biological anchoring** on layers `7, 9, 10, 11` to convert topological patterns into biological claims.
