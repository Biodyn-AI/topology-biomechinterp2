# Executor Iteration Report: iter_0046

## Objective
Execute the pre-registered 3-slot breadth packet from `iter_0045` brainstormer guidance:
- `H121` (`manifold_distance`, new method): directional geodesic asymmetry (`N605`).
- `H122` (`cross_model_alignment`, major-reset method): cross-model landscape transport (`N609`).
- `H123` (`module_structure`, refinement): stricter hardening of signed motif-community signal (`N600`).

## Environment
- Python environment: `conda run -n subproject40-topology`
- Runner: `iterations/iter_0046/run_iter0046_screen.py`

## Command Trace
```bash
conda run -n subproject40-topology python -m py_compile iterations/iter_0046/run_iter0046_screen.py
conda run -n subproject40-topology python iterations/iter_0046/run_iter0046_screen.py
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```

## Results (Quantitative)

| Hypothesis | Family | Primary Metric | Result | Null Robustness | Decision |
|---|---|---|---:|---:|---|
| H121 | manifold_distance | mean `delta_vs_h70` | `+0.03273` (12 rows; 3 domains x 2 splits x 2 layers) | positive mean null-gap domain-splits: `2/6` | neutral |
| H122 | cross_model_alignment | mean `transport_score_neg_mse` | `-100.54990` (12 rows) | positive mean null-gap domain-splits: `0/6` | negative |
| H123 | module_structure | mean `delta_vs_h70` | `+0.09351` (22 rows; seeds {42,43,44}) | positive mean null-gap domain-splits: `8/8` observed | promising |

### H121 details (`manifold_distance`, directional geodesic asymmetry)
- Directional lift was broadly positive (`11/12` rows positive; all `6/6` domain-splits positive on mean delta).
- Null-calibrated support was partial but non-zero (`3/12` row-level positive null-gap; `2/6` domain-splits positive null-gap).
- The pre-registered cheap-screen gate was met (`>=2/6` positive null-gap domain-splits, and at least one `source_disjoint` null-gap positive).
- Best row: `lung/target_disjoint/layer7` (`delta=+0.07505`, `null_gap=+0.02468`).
- Artifacts:
  - `iterations/iter_0046/h121_directional_geodesic_asymmetry_by_domain_split_layer.csv`
  - `iterations/iter_0046/h121_directional_geodesic_asymmetry_domain_summary.csv`
  - `iterations/iter_0046/h121_directional_geodesic_asymmetry_null_summary.csv`

### H122 details (`cross_model_alignment`, landscape transport reset)
- The tested transport objective was decisively negative.
- Observed transport score remained strongly below null references in all slices (`12/12` negative row-level null gaps; `0/6` positive domain-split null gaps).
- Mean observed module transport MSE was high (`100.5499`) and no domain/split recovered positive robustness.
- Artifacts:
  - `iterations/iter_0046/h122_landscape_transport_by_domain_split_layer.csv`
  - `iterations/iter_0046/h122_landscape_transport_domain_summary.csv`
  - `iterations/iter_0046/h122_landscape_transport_null_summary.csv`

### H123 details (`module_structure`, strict H118 hardening)
- Signal remained strong under stricter controls (`22/22` rows positive; mean `delta=+0.09351`).
- Null robustness was strong in all observed domain-splits (`22/22` positive row-level null gaps; `8/8` positive mean null-gap domain-splits).
- Dual-axis coverage was substantial but incomplete (observed in `immune` and `external_lung`; `lung/dual_axis_disjoint` not evaluable under current sampling/class constraints).
- Strongest observed aggregate was `external_lung/target_disjoint` (`mean delta=+0.11979`, `mean null_gap=+0.04994`).
- Artifacts:
  - `iterations/iter_0046/h123_signed_motif_module_hardening_by_seed_domain_split.csv`
  - `iterations/iter_0046/h123_signed_motif_module_hardening_domain_summary.csv`
  - `iterations/iter_0046/h123_signed_motif_module_hardening_null_summary.csv`

## Interpretation
- `H123` is the strongest branch this iteration: high directional consistency and broad null-gap survival under stricter controls.
- `H121` provides a viable manifold-distance rescue variant with directional consistency and limited but real null support.
- `H122` fails as a major-reset cross-model topology transport objective; this endpoint should be retired.

## Blockers / Runtime Notes
- No data-access blocker.
- `sklearn` produced high-volume deprecation warnings (`penalty`/`l1_ratio`) during logistic CV; outputs are unaffected.
- Coverage gap: `lung/dual_axis_disjoint` rows were not retained in `H123` after filtering/sampling constraints; follow-up should force this split explicitly.

## Machine-Readable Summary
- `iterations/iter_0046/iter0046_screen_summary.json`
