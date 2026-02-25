# Executor Iteration Report: iter_0045

## Objective
Run a breadth-first 3-hypothesis packet with one carry-over refinement and two materially new methods:
- `H118` (`module_structure`, refinement from `H116`): multiseed signed TRRUST motif x graph-community interaction over H70.
- `H119` (`cross_model_alignment`, new method): disagreement-conditioned cross-model transfer utility (scGPT vs Geneformer).
- `H120` (`manifold_distance`, new method): geodesic-path curvature-drift descriptors over H70.

## Environment
- Python environment: `conda run -n subproject40-topology`
- Runner: `iterations/iter_0045/run_iter0045_screen.py`

## Command Trace
```bash
conda run -n subproject40-topology python -m py_compile iterations/iter_0045/run_iter0045_screen.py
conda run -n subproject40-topology python iterations/iter_0045/run_iter0045_screen.py
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```

## Results (Quantitative)

| Hypothesis | Family | Primary Metric | Result | Null Robustness | Decision |
|---|---|---|---:|---:|---|
| H118 | module_structure | mean `delta_vs_h70` | `+0.09885` (18 rows; 3 seeds x 3 domains x 2 splits) | positive mean null-gap domain-splits: `3/6` | promising |
| H119 | cross_model_alignment | mean `delta_vs_h70` | `+0.00060` (6 domain-split rows) | positive mean null-gap domain-splits: `1/6` | negative |
| H120 | manifold_distance | mean `delta_vs_h70` | `+0.03854` (12 rows; layers {7,11}) | positive mean null-gap domain-splits: `3/6` | neutral |

### H118 details (`module_structure`, refinement of H116)
- Used true signed TRRUST edges (Activation/Repression only), not unsigned confidence scores.
- Directional effect was strong and stable across seeds: `18/18` rows had positive deltas (`mean delta=+0.09885`, `std=0.02775`).
- Null robustness was partial: `6/18` rows with positive q95 null-gap; domain-split mean null-gap positive in `3/6`.
- Strongest domain-split mean delta: `lung/source_disjoint = +0.14174`; weakest: `immune/source_disjoint = +0.06761`.
- Artifacts:
  - `iterations/iter_0045/h118_signed_motif_module_by_seed_domain_split.csv`
  - `iterations/iter_0045/h118_signed_motif_module_domain_summary.csv`
  - `iterations/iter_0045/h118_signed_motif_module_null_summary.csv`

### H119 details (`cross_model_alignment`, disagreement-conditioned transfer)
- Disagreement-gating did not materially improve transfer over ungated baseline (`mean delta=+0.00060`, `std=0.01274`).
- Domain-split direction mixed (`3/6` positive deltas) and null-gap largely negative (`1/6` positive mean null-gap).
- Largest negative split: `external_lung/source_disjoint` (`delta=-0.01022`, `null_gap=-0.04818`).
- Largest positive split: `immune/target_disjoint` (`delta=+0.01750`, `null_gap=-0.00202`, still below q95).
- Artifacts:
  - `iterations/iter_0045/h119_disagreement_gated_transfer_by_domain_split.csv`
  - `iterations/iter_0045/h119_disagreement_gated_transfer_domain_summary.csv`
  - `iterations/iter_0045/h119_disagreement_gated_transfer_null_summary.csv`

### H120 details (`manifold_distance`, geodesic curvature drift)
- Directional lift was consistent (`12/12` rows positive; `mean delta=+0.03854`, `std=0.02122`).
- Robustness remained partial (`5/12` rows positive q95 null-gap; domain-split mean null-gap positive in `3/6`).
- Strongest row: `lung/source_disjoint/layer11` (`delta=+0.07278`), but null-gap negative (`-0.02220`).
- Best null-robust row: `immune/target_disjoint/layer7` (`delta=+0.06028`, `null_gap=+0.03505`).
- Artifacts:
  - `iterations/iter_0045/h120_geodesic_curvature_drift_by_domain_split_layer.csv`
  - `iterations/iter_0045/h120_geodesic_curvature_drift_domain_summary.csv`
  - `iterations/iter_0045/h120_geodesic_curvature_drift_null_summary.csv`

## Interpretation
- `H118`: best signal this iteration. Multiseed directional stability plus partial null survival makes this a high-priority follow-up, but not yet a fully robust claim.
- `H119`: major-change cross-model rescue failed to produce robust utility; treat this endpoint as negative.
- `H120`: broad directional signal exists, but null-calibrated robustness is incomplete; keep as neutral pending stricter controls.

## Blockers / Runtime Notes
- No data blockers.
- `sklearn` emitted repetitive deprecation warnings (`penalty`/`l1_ratio`) during logistic CV; artifacts were generated successfully.
- Null budgets were intentionally bounded for screening speed, which sets a floor on attainable empirical p-values (e.g., `1/(n_perm+1)`).

## Machine-Readable Summary
- `iterations/iter_0045/iter0045_screen_summary.json`
