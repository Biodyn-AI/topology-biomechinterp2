# Executor Iteration Report: iter_0047

## Objective
Execute the pre-registered 3-slot breadth packet from `iter_0046` brainstormer guidance:
- `H124` (`module_structure`, refinement; `N625`): signed motif-community hardening with STRING conditioning and forced dual-axis coverage.
- `H125` (`cross_model_alignment`, major-reset exploration; `N622`): anchor-constrained cycle-consistent cross-model alignment.
- `H126` (`manifold_distance`, new method; `N620`): geodesic torsion and turning-angle asymmetry over H70.

## Environment
- Python environment: `conda run -n subproject40-topology`
- Runner: `iterations/iter_0047/run_iter0047_screen.py`

## Command Trace
```bash
conda run -n subproject40-topology python -m py_compile iterations/iter_0047/run_iter0047_screen.py
conda run -n subproject40-topology python iterations/iter_0047/run_iter0047_screen.py
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```

## Results (Quantitative)

| Hypothesis | Family | Primary Metric | Result | Null Robustness | Decision |
|---|---|---|---:|---:|---|
| H124 | module_structure | mean `delta_vs_h70` | `+0.13098` (27 rows; seeds {42,43,44} x 3 domains x 3 splits) | positive mean null-gap domain-splits: `4/9`; lung dual-axis mean null-gap `-0.00326` | neutral |
| H125 | cross_model_alignment | mean `transfer_delta_auc_vs_h70` | `+0.09855` (12 rows) | positive domain null-gap count: `0/3`; immune mean null-gap `-0.01894` | negative |
| H126 | manifold_distance | mean `delta_vs_h70` | `+0.04421` (12 rows; 3 domains x 2 splits x 2 layers) | positive mean null-gap domain-splits: `2/6`; source-disjoint positive-null-gap count: `1` | promising |

### H124 details (`module_structure`, N625)
- Directionality was fully consistent (`27/27` positive row-level deltas; all `9/9` domain-splits positive mean delta).
- Robustness gate was not met under strict null calibration (positive mean null-gap in `4/9` domain-splits, below the `>=8/9` target).
- Forced `lung/dual_axis_disjoint` coverage succeeded (`3` rows present), but null-gap remained slightly negative on average (`-0.00326`).
- Best robust split was `immune/dual_axis_disjoint` (mean delta `+0.12326`, mean null-gap `+0.00490`).
- Artifacts:
  - `iterations/iter_0047/h124_signed_string_hardening_by_seed_domain_split.csv`
  - `iterations/iter_0047/h124_signed_string_hardening_domain_summary.csv`
  - `iterations/iter_0047/h124_signed_string_hardening_null_summary.csv`

### H125 details (`cross_model_alignment`, N622)
- Raw transfer lift was high only in immune slices (immune mean delta `+0.32667`), but this did not survive null controls.
- Domain-level robustness failed decisively (positive mean null-gap domains `0/3`; immune null-gap still negative).
- Pre-registered keep gate failed (`>=2/3` positive-null-gap domains and immune `>=0` were both missed).
- Artifacts:
  - `iterations/iter_0047/h125_anchor_cycle_alignment_by_domain_split_layer.csv`
  - `iterations/iter_0047/h125_anchor_cycle_alignment_domain_split_summary.csv`
  - `iterations/iter_0047/h125_anchor_cycle_alignment_domain_summary.csv`
  - `iterations/iter_0047/h125_anchor_cycle_alignment_null_summary.csv`

### H126 details (`manifold_distance`, N620)
- Directional signal was broad (`11/12` positive row-level deltas; `6/6` domain-splits positive mean delta).
- Null-calibrated support was limited but non-zero (`6/12` row-level positive null-gap; `2/6` domain-split positive mean null-gap).
- Pre-registered cheap-screen gate passed: positive mean delta in `>=4/6`, positive mean null-gap in `>=2/6`, and at least one source-disjoint split with positive null-gap (`immune/source_disjoint`).
- Best robust split was `immune/source_disjoint` (mean delta `+0.10097`, mean null-gap `+0.01757`).
- Artifacts:
  - `iterations/iter_0047/h126_geodesic_torsion_by_domain_split_layer.csv`
  - `iterations/iter_0047/h126_geodesic_torsion_domain_summary.csv`
  - `iterations/iter_0047/h126_geodesic_torsion_null_summary.csv`

## Interpretation
- `H124` confirms strong directional signal in the `H123` lineage, but strict null robustness and lung dual-axis robustness remain insufficient for promotion.
- `H125` provides decisive negative evidence for this anchor-constrained cycle-consistency endpoint; it should be retired in this form.
- `H126` is a viable manifold-geometry branch: modest but reproducible null-surviving signal with gate pass in the fast screen.

## Blockers / Runtime Notes
- No data-access or package blocker.
- High-volume scikit-learn deprecation/user warnings were emitted during logistic-CV loops (`penalty`/`l1_ratio`); metrics/artifacts were still produced.

## Machine-Readable Summary
- `iterations/iter_0047/iter0047_screen_summary.json`

## Paper Update
- Updated `paper/autoloop_research_paper.tex` with section marker `ITERATION UPDATE: iter_0047`.
- Compiled `paper/autoloop_research_paper.pdf` via `latexmk` (33 pages in this build).
