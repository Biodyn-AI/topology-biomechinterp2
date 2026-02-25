# Executor Iteration Report: iter_0052

## Objective
Run a breadth-first 3-hypothesis screen with one carry-over branch and two materially new tests:
- `H139` (`manifold_distance`): seed-expanded robustness test for sectional anisotropy descriptors.
- `H140` (`topology_stability`): neighborhood-size scaling stability on the same geometric signal.
- `H141` (`null_sensitivity`): strict null-fragility audit using this iteration's multi-null outputs.

## Environment
- Python environment: `conda run -n subproject40-topology`
- Runner: `iterations/iter_0052/run_iter0052_screen.py`

## Command Trace
```bash
conda run -n subproject40-topology python -m py_compile iterations/iter_0052/run_iter0052_screen.py
PYTHONWARNINGS=ignore conda run -n subproject40-topology python iterations/iter_0052/run_iter0052_screen.py
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```

## Results (Quantitative)

| Hypothesis | Family | Primary metric | Result | Null/control summary | Decision |
|---|---|---|---:|---:|---|
| H139 | manifold_distance | mean `delta_vs_h70` | `+0.03135` (25 rows) | positive mean null-gap domain-splits `6/9`; row-level positive null-gap `16/25` | promising |
| H140 | topology_stability | mean `delta_gain_vs_swap` | `+0.03374` (24 rows) | positive gain domain-splits `6/8`; positive gain rows `19/24` | neutral |
| H141 | null_sensitivity | mean `strict_margin` | `-0.00523` (25 rows) | strict-positive domain-splits `3/9`; strict-positive rows `15/25` | inconclusive |

### H139 details (carry-over robustness expansion)
- Scope: seeds `{42,43,44}`; domains `{immune, lung, external_lung}`; splits `{source_disjoint, target_disjoint, dual_axis_disjoint}`; layer `11`.
- Method: H136 sectional anisotropy/tangent descriptors over H70 with three null families (endpoint swap within distance bins, sectional row shuffle, label permutation; 8 each).
- Key outcomes:
  - mean `delta_vs_h70 = +0.03135`.
  - positive directional rows `20/25`.
  - positive null-gap rows `16/25`.
  - positive mean null-gap domain-splits `6/9`.
  - hard slices now non-negative on average:
    - `immune/source_disjoint` mean null-gap `+0.03052`
    - `lung/dual_axis_disjoint` mean null-gap `+0.01718`
- Coverage caveat: `external_lung/dual_axis_disjoint` had only one evaluable seed row (seed43), with mean null-gap `-0.06927`.
- Artifacts:
  - `iterations/iter_0052/h139_sectional_seed_robustness_by_seed_domain_split.csv`
  - `iterations/iter_0052/h139_sectional_seed_robustness_domain_split_summary.csv`
  - `iterations/iter_0052/h139_sectional_seed_robustness_domain_summary.csv`
  - `iterations/iter_0052/h139_sectional_seed_robustness_null_summary.csv`

### H140 details (new topology-stability test)
- Scope: seed42; domains `{immune, lung, external_lung}`; source/target/dual-axis where evaluable; layer `11`; neighborhood grid `k={8,12,16}`.
- Method: measure `delta_vs_h70` across neighborhood scales and compare against matched endpoint-swap control (`delta_gain_vs_swap`).
- Key outcomes:
  - mean `delta_vs_h70 = +0.02693`.
  - mean `delta_gain_vs_swap = +0.03374`.
  - positive gain rows `19/24`, positive gain domain-splits `6/8`.
  - by-k means:
    - `k=8`: mean delta `+0.02996`, mean gain `+0.03880`
    - `k=12`: mean delta `+0.02494`, mean gain `+0.03174`
    - `k=16`: mean delta `+0.02588`, mean gain `+0.03068`
- Interpretation: signal is generally stable across neighborhood scales, but not universal (`immune/source_disjoint` and `lung/dual_axis_disjoint` were gain-negative).
- Artifacts:
  - `iterations/iter_0052/h140_neighborhood_scaling_by_domain_split_k.csv`
  - `iterations/iter_0052/h140_neighborhood_scaling_domain_split_summary.csv`
  - `iterations/iter_0052/h140_neighborhood_scaling_domain_summary.csv`

### H141 details (new strict null-sensitivity audit)
- Scope: post-hoc strict-margin analysis over H139 outputs.
- Method: for each row, compute strict margin against max q95 across the three null families.
- Key outcomes:
  - mean strict margin `-0.00523`.
  - strict-positive rows `15/25`.
  - strict-positive domain-splits `3/9`.
  - domain means:
    - immune `+0.01164`
    - lung `-0.00795`
    - external_lung `-0.02344`
- Interpretation: H139 remains partially null-fragile outside immune despite strong directional and moderate null-gap performance.
- Artifacts:
  - `iterations/iter_0052/h141_strict_null_sensitivity_row_summary.csv`
  - `iterations/iter_0052/h141_strict_null_sensitivity_domain_split_summary.csv`
  - `iterations/iter_0052/h141_strict_null_sensitivity_domain_summary.csv`
  - `iterations/iter_0052/h141_strict_null_sensitivity_nullkind_summary.csv`

## Interpretation
- `H139` is the strongest branch this iteration and clears the planned robustness expansion gate on available rows.
- `H140` provides novel, low-cost evidence that the anisotropy signal is reasonably neighborhood-scale stable.
- `H141` adds decisive caution: strict max-null margins are still negative on average, concentrated in external-lung and some lung slices.

## Blockers / Runtime Notes
- No data/package blocker.
- Coverage limitation: `external_lung/dual_axis_disjoint` had only one evaluable seed row after filtering/sampling constraints.
- Non-blocking warning class observed during model fitting: scikit-learn `penalty` deprecation / `l1_ratio` consistency warnings.

## Machine-Readable Summary
- `iterations/iter_0052/iter0052_screen_summary.json`

## Paper Update
- Updated `paper/autoloop_research_paper.tex` with section marker `ITERATION UPDATE: iter_0052` and artifact-backed quantitative claims.
- Recompiled `paper/autoloop_research_paper.pdf` via `latexmk`.
