# Executor Iteration Report: iter_0041

## Objective
Execute the pre-registered breadth packet from iter_0040 brainstormer guidance with three materially distinct slots:
- `H106` (`persistent_homology`, `N538`): STRING triad-closure weighted filtration over H93 backbone.
- `H107` (`topology_stability`, `N537` rescue): finite-state descriptor motif screen with a second-order sequence model.
- `H108` (`cross_model_alignment`, `N531` rescue): cross-model perturbation-response rank alignment.

## Environment
- Python environment: `conda run -n subproject40-topology`
- Runner: `iterations/iter_0041/run_iter0041_screen.py`

## Command Trace
```bash
conda run -n subproject40-topology python -m py_compile iterations/iter_0041/run_iter0041_screen.py
conda run -n subproject40-topology python iterations/iter_0041/run_iter0041_screen.py
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```

## Results (Quantitative)

| Hypothesis | Family | Primary Metric | Result | Null Robustness | Decision |
|---|---|---|---:|---:|---|
| H106 | persistent_homology | mean `delta_auc_string_triad_weighted_minus_h93` | `-0.00118` (12 rows) | positive mean null-gap domain-splits: `0/6` | negative |
| H107 | topology_stability | mean `delta_auc_dfa_motif_minus_h70` | `+0.04563` (6 rows) | positive mean null-gap domain-splits: `0/6` | inconclusive |
| H108 | cross_model_alignment | mean `module_response_rank_spearman` | `+0.73159` (3 domains) | positive domain null-gap count: `2/3` | promising |

### H106 details (`N538`: STRING triad-closure weighted filtration)
- Gate outcome: failed fail-fast gate (`positive mean null-gap = 0/6` domain-splits).
- Mean delta was near-zero and slightly negative (`-0.00118`).
- Best directional row still failed control: lung/target-disjoint/layer11 had `delta=+0.00852` but `q95_null=0.01709` (`null_gap=-0.00857`).
- Artifacts:
  - `iterations/iter_0041/h106_string_triad_weighted_filtration_by_domain_split_layer.csv`
  - `iterations/iter_0041/h106_string_triad_weighted_filtration_domain_summary.csv`
  - `iterations/iter_0041/h106_string_triad_weighted_filtration_null_summary.csv`

### H107 details (`N537`: finite-state descriptor motif, major method change)
- Directional effect was consistently positive (`delta > 0` in `6/6` domain-splits; mean `+0.04563`).
- Robustness failed under q95 null-gap in all splits (`0/6` positive null-gap).
- Example: immune/source-disjoint had largest raw lift (`delta=+0.09176`) but still below q95 null (`null_gap=-0.01615`).
- Artifacts:
  - `iterations/iter_0041/h107_finite_state_descriptor_motif_by_domain_split.csv`
  - `iterations/iter_0041/h107_finite_state_descriptor_motif_domain_summary.csv`
  - `iterations/iter_0041/h107_finite_state_descriptor_motif_null_summary.csv`

### H108 details (`N531`: cross-model perturbation-response alignment)
- Rescue rationale used: static concordance was retired; this test switched to perturbation-response rank agreement.
- Primary metric improved materially: mean response-rank Spearman `+0.73159` across domains.
- Null-gap gate passed at domain level (`2/3` domains positive):
  - external_lung: `rho=0.78348`, `null_gap=+0.03452`
  - lung: `rho=0.80870`, `null_gap=+0.01422`
  - immune: `rho=0.60261`, `null_gap=-0.21491` (fails)
- Artifacts:
  - `iterations/iter_0041/h108_cross_model_perturbation_response_by_domain.csv`
  - `iterations/iter_0041/h108_cross_model_perturbation_response_domain_summary.csv`
  - `iterations/iter_0041/h108_cross_model_perturbation_response_null_summary.csv`

## Interpretation
- `H106/N538` produced decisive negative evidence in this pilot and should be retired in this exact form.
- `H107/N537` showed consistent directional lift but failed robustness controls; treat as inconclusive rather than promoted.
- `H108/N531` is the first cross-model branch in recent loops to pass the pre-registered null-gap gate (`2/3` domains), so it is a promising rescue line with one domain-specific failure mode (immune).

## Blockers / Runtime Notes
- No data/runtime blockers.
- Non-blocking warning volume from sklearn logistic penalty deprecation; outputs were generated successfully.

## Machine-Readable Summary
- `iterations/iter_0041/iter0041_screen_summary.json`
