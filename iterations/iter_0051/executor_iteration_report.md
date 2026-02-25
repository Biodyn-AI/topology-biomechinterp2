# Executor Iteration Report: iter_0051

## Objective
Execute a breadth-oriented 3-slot screening packet with materially novel methods:
- `H136` (`manifold_distance`, `N680`): sectional anisotropy/tangent-orientation manifold screen.
- `H137` (`cross_model_alignment`, `N684` rescue with major method change): correspondence-free cross-model topology-descriptor alignment.
- `H138` (`module_structure`, `N686`): ontology-chart sheaf-obstruction hardening on signed motif-community backbone.

## Environment
- Python environment: `conda run -n subproject40-topology`
- Runner: `iterations/iter_0051/run_iter0051_screen.py`

## Command Trace
```bash
conda run -n subproject40-topology python -m py_compile iterations/iter_0051/run_iter0051_screen.py
conda run -n subproject40-topology python iterations/iter_0051/run_iter0051_screen.py
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```

## Results (Quantitative)

| Hypothesis | Family | Primary metric | Result | Null robustness | Decision |
|---|---|---|---:|---:|---|
| H136 | manifold_distance | mean `delta_vs_h70` | `+0.02572` (12 rows) | positive mean null-gap domain-splits: `2/6`; row-level positive null-gap: `2/12` | promising |
| H137 | cross_model_alignment | mean `alignment_delta_vs_random` | `+0.00194` (12 rows) | positive mean null-gap domains: `0/3`; immune mean null-gap: `-0.09748` | negative |
| H138 | module_structure | mean `delta_vs_h70` | `+0.13381` (27 rows) | positive mean null-gap domain-splits: `0/9`; hard slices remain negative | negative |

### H136 details (`N680`: sectional anisotropy)
- Scope: seed42; domains `{immune, lung, external_lung}`; splits `{source_disjoint, target_disjoint}`; layers `{7,11}`.
- Directionality:
  - mean `delta_vs_h70 = +0.02572`
  - positive mean delta domain-splits `5/6`
  - row-level positive deltas `9/12`
- Strict-null outcome:
  - positive mean null-gap domain-splits `2/6`
  - row-level positive null-gap `2/12`
  - source-disjoint null-positive domain-split present (`lung/source_disjoint`, mean null-gap `+0.01017`)
- Interpretation: this cheap broad-screen passed its pre-registered keep gate (`>=4/6` directional and `>=2/6` null-positive domain-splits) and is the only promoted branch from this iteration.
- Artifacts:
  - `iterations/iter_0051/h136_sectional_anisotropy_by_domain_split_layer.csv`
  - `iterations/iter_0051/h136_sectional_anisotropy_domain_split_summary.csv`
  - `iterations/iter_0051/h136_sectional_anisotropy_null_summary.csv`

### H137 details (`N684`: correspondence-free cross-model alignment)
- Scope: seed42; domains `{immune, lung, external_lung}`; splits `{source_disjoint, target_disjoint}`; layers `{7,11}`.
- Method: correspondence-free topology-descriptor matching between scGPT and Geneformer signature manifolds, benchmarked against random cross-domain pairing and kernel-spectrum permutation nulls.
- Outcome:
  - mean `alignment_delta_vs_random = +0.00194` (near zero)
  - positive mean delta domain count `1/3`
  - positive mean null-gap domain count `0/3`
  - immune mean null-gap `-0.09748`
- Interpretation: major-method rescue failed strict domain-level null gates; this endpoint is retired.
- Artifacts:
  - `iterations/iter_0051/h137_correspondence_free_alignment_by_domain_split_layer.csv`
  - `iterations/iter_0051/h137_correspondence_free_alignment_domain_split_summary.csv`
  - `iterations/iter_0051/h137_correspondence_free_alignment_domain_summary.csv`
  - `iterations/iter_0051/h137_correspondence_free_alignment_null_summary.csv`

### H138 details (`N686`: ontology sheaf hardening)
- Scope: seeds `{42,43,44}`; domains `{immune, lung, external_lung}`; splits `{source_disjoint, target_disjoint, dual_axis_disjoint}`; layer `11`.
- Directionality:
  - mean `delta_vs_h70 = +0.13381`
  - positive mean delta domain-splits `9/9`
  - row-level positive deltas `27/27`
- Strict-null outcome:
  - positive mean null-gap domain-splits `0/9`
  - row-level positive null-gap `1/27`
  - hard slices remain negative:
    - `immune/source_disjoint` mean null-gap `-0.00567`
    - `lung/dual_axis_disjoint` mean null-gap `-0.00476`
- Interpretation: strong directional signal remains null-fragile; this additive ontology-sheaf endpoint is negative in current form.
- Artifacts:
  - `iterations/iter_0051/h138_ontology_sheaf_hardening_by_seed_domain_split.csv`
  - `iterations/iter_0051/h138_ontology_sheaf_hardening_domain_split_summary.csv`
  - `iterations/iter_0051/h138_ontology_sheaf_hardening_domain_summary.csv`
  - `iterations/iter_0051/h138_ontology_sheaf_hardening_null_summary.csv`

## Interpretation
- `H136` is the only branch with non-trivial null-surviving lift and is promoted.
- `H137` remains negative even after the required major method reset; retire this endpoint.
- `H138` reproduces the now-familiar directional-only pattern and fails the hard-slice strict-null gate again.

## Blockers / Runtime Notes
- No data/package blocker.
- High-volume scikit-learn deprecation warnings (`penalty`/`l1_ratio`) were emitted during logistic-CV loops; runs completed successfully.

## Machine-Readable Summary
- `iterations/iter_0051/iter0051_screen_summary.json`

## Paper Update
- Updated `paper/autoloop_research_paper.tex` with section marker `ITERATION UPDATE: iter_0051`.
- Recompiled `paper/autoloop_research_paper.pdf` via `latexmk`.
