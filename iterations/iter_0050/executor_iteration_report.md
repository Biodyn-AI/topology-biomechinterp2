# Executor Iteration Report: iter_0050

## Objective
Execute a breadth-oriented 3-slot packet with one carry-over refinement and two materially novel tests:
- `H133` (`persistent_homology`, novel family slot): rank-surface filtration topology surrogate.
- `H134` (`intrinsic_dimensionality`, novel method slot): TWO-NN path-phase descriptors.
- `H135` (`module_structure`, carry-over refinement): hard-slice-focused rerun of semantic motif-community hardening (`H130` lineage).

## Environment
- Python environment: `conda run -n subproject40-topology`
- Runner: `iterations/iter_0050/run_iter0050_screen.py`

## Command Trace
```bash
conda run -n subproject40-topology python -m py_compile iterations/iter_0050/run_iter0050_screen.py
conda run -n subproject40-topology python iterations/iter_0050/run_iter0050_screen.py
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```

## Results (Quantitative)

| Hypothesis | Family | Primary metric | Result | Null robustness | Decision |
|---|---|---|---:|---:|---|
| H133 | persistent_homology | mean `delta_vs_h70` | `-0.04611` (6 rows) | positive mean null-gap domain-splits: `0/6`; row-level positive null-gap: `0/6` | negative |
| H134 | intrinsic_dimensionality | mean `delta_vs_h70` | `+0.01132` (12 rows) | positive mean null-gap domain-splits: `0/6`; row-level positive null-gap: `0/12` | negative |
| H135 | module_structure | mean `delta_vs_h70` | `+0.13870` (12 rows) | positive mean null-gap domain-splits: `0/4`; hard slices remain negative (`lung/dual_axis=-0.00502`, `immune/source=-0.01473`) | negative |

### H133 details (`persistent_homology`, rank-surface filtration)
- Scope: seed42; domains `{immune, lung, external_lung}`; splits `{source_disjoint, target_disjoint}`; layer `11`.
- Directional outcome: all tested domain-splits were negative (`6/6` mean deltas < 0).
- Strict-null outcome: no support under axis-rank and label nulls (`0/6` positive mean null-gap domain-splits).
- Interpretation: this filtration-topology surrogate is decisively non-supportive in the tested regime.
- Artifacts:
  - `iterations/iter_0050/h133_rank_surface_persistence_by_domain_split.csv`
  - `iterations/iter_0050/h133_rank_surface_persistence_domain_summary.csv`
  - `iterations/iter_0050/h133_rank_surface_persistence_null_summary.csv`

### H134 details (`intrinsic_dimensionality`, ID path-phase descriptors)
- Scope: seed42; domains `{immune, lung, external_lung}`; splits `{source_disjoint, target_disjoint}`; layers `{7,11}`.
- Directional outcome: weak positive directional lift in all 6 domain-splits (`mean delta_vs_h70 > 0`), strongest at `lung/source_disjoint` (`+0.02802`).
- Strict-null outcome: no null-surviving support (`0/6` positive mean null-gap domain-splits; all mean null-gaps negative).
- Descriptor diagnostics: `id_sign_flip_count_mean` and `id_hysteresis_mean` were non-trivial, but this did not translate to null-robust gains.
- Interpretation: directional signal exists but is null-fragile; this endpoint is not promotion-ready.
- Artifacts:
  - `iterations/iter_0050/h134_id_phase_descriptor_by_domain_split_layer.csv`
  - `iterations/iter_0050/h134_id_phase_descriptor_domain_summary.csv`
  - `iterations/iter_0050/h134_id_phase_descriptor_null_summary.csv`

### H135 details (`module_structure`, hard-slice semantic refinement)
- Scope: seeds `{42,43,44}`; domains `{immune, lung}`; hard slices `{source_disjoint, dual_axis_disjoint}`; layer `11`; null budget increased to `32` per family.
- Directional outcome: strong and consistent (`mean delta_vs_h70 = +0.13870`; positive deltas `12/12`).
- Strict-null outcome: still fails hard-slice gate (`positive mean null-gap domain-splits = 0/4`), including:
  - `immune/source_disjoint`: mean null-gap `-0.01473`
  - `lung/dual_axis_disjoint`: mean null-gap `-0.00502`
- Interpretation: this H130-style rescue remains directional-only after targeted hard-slice rerun; endpoint is not worth continued refinement without materially changed features/nulls.
- Artifacts:
  - `iterations/iter_0050/h135_hard_slice_semantic_refinement_by_seed_domain_split.csv`
  - `iterations/iter_0050/h135_hard_slice_semantic_refinement_domain_summary.csv`
  - `iterations/iter_0050/h135_hard_slice_semantic_refinement_null_summary.csv`

## Interpretation
- `H133`: decisive negative evidence.
- `H134`: mild directional signal but no null survival; negative as a screening endpoint.
- `H135`: second hard-slice-focused failure for this semantic-hardening variant; treat this endpoint as negative/retired.

## Blockers / Runtime Notes
- No data or package blockers.
- High-volume scikit-learn warnings (`penalty` deprecation and `l1_ratio` consistency warnings) were emitted during logistic-CV loops; runs completed successfully.

## Machine-Readable Summary
- `iterations/iter_0050/iter0050_screen_summary.json`

## Paper Update
- Updated `paper/autoloop_research_paper.tex` with section marker `ITERATION UPDATE: iter_0050`.
- Recompiled `paper/autoloop_research_paper.pdf` via `latexmk`.
