# Executor Iteration Report: iter_0040

## Objective
Run a breadth packet with 3 hypotheses emphasizing novelty and fast rejection:
- `H103` (`persistent_homology`): interaction-only derivative rescue (`N508`) on top of H91/H93-style backbone.
- `H104` (`manifold_distance`): depth motif grammar screen (`N520`).
- `H105` (`null_sensitivity`): STRING-conditioned null calibration check (`N519`).

## Environment
- Python environment: `conda run -n subproject40-topology`
- Runner: `iterations/iter_0040/run_iter0040_screen.py`

## Command Trace
```bash
conda run -n subproject40-topology python -m py_compile iterations/iter_0040/run_iter0040_screen.py
conda run -n subproject40-topology python iterations/iter_0040/run_iter0040_screen.py
latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```

## Results (Quantitative)

| Hypothesis | Family | Primary Metric | Result | Null Robustness | Decision |
|---|---|---|---:|---:|---|
| H103 | persistent_homology | mean `delta_auc_interaction_derivative_minus_h91_h93` | `-0.00304` (24 rows) | positive mean null-gap domain-splits: `0/6` | negative |
| H104 | manifold_distance | mean `delta_auc_motif_grammar_minus_h70` | `-0.00908` (6 rows) | positive mean null-gap domain-splits: `0/6` | negative |
| H105 | null_sensitivity | mean `conditioned_minus_unconditioned_null_gap` | `-0.05125` (12 rows) | positive conditioned-gain domain-splits: `0/6` | negative |

### H103 details (`N508` rescue)
- Directional signal was weak/inconsistent: positive mean delta in `2/6` domain-splits, negative overall mean.
- Robustness failed decisively: mean null-gap negative in all `6/6` domain-splits.
- Best row still failed control: lung/source-disjoint/layer11 had `delta=+0.01298` but `null_gap=-0.00579`.
- Artifacts:
  - `iterations/iter_0040/h103_interaction_derivative_rescue_by_domain_split_layer.csv`
  - `iterations/iter_0040/h103_interaction_derivative_rescue_domain_summary.csv`
  - `iterations/iter_0040/h103_interaction_derivative_rescue_null_summary.csv`

### H104 details (`N520` cheap broad screen)
- Mixed directionality but no robustness: `4/6` domain-splits had positive delta, yet all `6/6` had negative mean null-gap.
- Mean effect is negative overall (`-0.00908`).
- Best slice remained below null threshold: lung/target-disjoint `delta=+0.03081`, `null_gap=-0.00244`.
- Artifacts:
  - `iterations/iter_0040/h104_depth_motif_grammar_by_domain_split.csv`
  - `iterations/iter_0040/h104_depth_motif_grammar_domain_summary.csv`
  - `iterations/iter_0040/h104_depth_motif_grammar_null_summary.csv`

### H105 details (`N519` conditioned null sensitivity)
- Base utility stayed positive (`mean delta_auc_h93_with_string_minus_h70 = +0.07099`), but conditional null calibration worsened margins.
- Mean conditioned minus unconditioned null-gap was negative (`-0.05125`), with positive gain in `0/6` domain-splits.
- Unconditioned null-gap was positive only in lung/source-disjoint; conditioned null-gap was negative for all tested rows.
- Artifacts:
  - `iterations/iter_0040/h105_string_conditioned_null_calibration_by_domain_split_layer.csv`
  - `iterations/iter_0040/h105_string_conditioned_null_calibration_domain_summary.csv`
  - `iterations/iter_0040/h105_string_conditioned_null_calibration_null_summary.csv`

## Interpretation
- This packet produced decisive negative evidence for all three tested formulations under explicit controls.
- `N508` rescue failed its core robustness objective and should be retired in this interaction form.
- `N520` motif grammar is currently directional-only and non-robust; do not carry forward unchanged.
- `N519` indicates conditional STRING+degree nulls are stricter than unconditioned calibration in this setup, not a rescue path.

## Execution Notes
- No data/runtime blockers.
- `sklearn` emitted many non-blocking deprecation warnings for logistic `penalty` handling; outputs were generated successfully.

## Machine-Readable Summary
- `iterations/iter_0040/iter0040_screen_summary.json`
