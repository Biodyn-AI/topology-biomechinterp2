# Executor Iteration Report: iter_0049

## Objective
Run the pre-registered breadth packet from iter_0048 brainstorming guidance with one refinement and two materially changed probes:
- `H130` (`module_structure`, refinement of `H127` / `N656`): continuous GO semantic x STRING hardening.
- `H131` (`cross_model_alignment`, major method reset from `H125` / `N653`): chart/sheaf-style local alignment with cycle-consistency diagnostics.
- `H132` (`manifold_distance`, major method change from torsion lineage / `N650`): local chart-fracture manifold diagnostics.

## Environment
- Python environment: `conda run -n subproject40-topology`
- Runner: `iterations/iter_0049/run_iter0049_screen.py`

## Command Trace
```bash
conda run -n subproject40-topology python -m py_compile iterations/iter_0049/run_iter0049_screen.py
conda run -n subproject40-topology python iterations/iter_0049/run_iter0049_screen.py
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```

## Results (Quantitative)

| Hypothesis | Family | Primary metric | Result | Null robustness | Decision |
|---|---|---|---:|---:|---|
| H130 | module_structure | mean `delta_vs_h70` | `+0.13096` (27 rows; 3 seeds x 3 domains x 3 splits) | positive mean null-gap domain-splits: `0/9`; row-level positive null-gap `4/27`; lung dual-axis mean null-gap `-0.00541`; immune source mean null-gap `-0.01486` | neutral |
| H131 | cross_model_alignment | mean `alignment_delta_vs_random` (`delta_vs_h70`) | `-0.00293` (12 rows; 3 domains x 2 splits x 2 layers) | positive mean null-gap domains: `0/3`; row-level positive null-gap `0/12`; immune mean null-gap `-0.14339` | negative |
| H132 | manifold_distance | mean `delta_vs_h70` | `+0.01637` (12 rows; 3 domains x 2 splits x 2 layers) | positive mean null-gap domain-splits: `0/6`; row-level positive null-gap `0/12` | negative |

### H130 details (`module_structure`, N656 semantic hardening)
- Directional lift remained universally positive (`27/27` positive row deltas; `9/9` positive mean-delta domain-splits).
- Strict-null robustness failed globally (`0/9` positive mean null-gap domain-splits).
- Hard slices stayed negative after semantic hardening:
  - `lung/dual_axis_disjoint`: mean null-gap `-0.00541`.
  - `immune/source_disjoint`: mean null-gap `-0.01486`.
- Interpretation: continuous GO semantics improved directional signal but did not survive null calibration.
- Artifacts:
  - `iterations/iter_0049/h130_semantic_go_string_hardening_by_seed_domain_split.csv`
  - `iterations/iter_0049/h130_semantic_go_string_hardening_domain_summary.csv`
  - `iterations/iter_0049/h130_semantic_go_string_hardening_null_summary.csv`

### H131 details (`cross_model_alignment`, N653 chart/sheaf reset)
- Cross-model chart/sheaf objective did not recover utility under controls.
- Mean delta was slightly negative overall (`-0.00293`).
- Even where immune directional deltas were positive in subset rows, null gaps stayed negative in all rows (`0/12` positive row null-gap).
- Domain-level robustness gate failed (`0/3` positive mean null-gap domains; immune mean null-gap `-0.14339`).
- Interpretation: this rescue endpoint is decisively non-robust; treat as negative evidence.
- Artifacts:
  - `iterations/iter_0049/h131_chart_sheaf_alignment_by_domain_split_layer.csv`
  - `iterations/iter_0049/h131_chart_sheaf_alignment_domain_split_summary.csv`
  - `iterations/iter_0049/h131_chart_sheaf_alignment_domain_summary.csv`
  - `iterations/iter_0049/h131_chart_sheaf_alignment_null_summary.csv`

### H132 details (`manifold_distance`, N650 chart-fracture broad screen)
- Directional lift was modest (`mean delta +0.01637`; positive deltas `9/12`; positive mean-delta domain-splits `4/6`).
- Strict-null robustness failed everywhere (`0/6` positive mean null-gap domain-splits).
- Interpretation: chart-fracture descriptors capture weak directional structure but do not survive controls in this formulation.
- Artifacts:
  - `iterations/iter_0049/h132_chart_fracture_by_domain_split_layer.csv`
  - `iterations/iter_0049/h132_chart_fracture_domain_summary.csv`
  - `iterations/iter_0049/h132_chart_fracture_null_summary.csv`

## Interpretation
- No branch met a robust promotion gate this iteration.
- `H130` remains the strongest directional branch but is still null-fragile on the hardest disjoint slices.
- `H131` and `H132` add negative evidence for their current rescue formulations and should not be iterated without materially different objectives.

## Blockers / Runtime Notes
- No data-access or package-install blockers.
- High-volume scikit-learn logistic warnings (`penalty` deprecation/inconsistent `l1_ratio`) occurred during CV loops; runs completed and artifacts were written.

## Machine-Readable Summary
- `iterations/iter_0049/iter0049_screen_summary.json`

## Paper Update
- Updated `paper/autoloop_research_paper.tex` with section marker `ITERATION UPDATE: iter_0049`.
- Compiled `paper/autoloop_research_paper.pdf` via `latexmk`.
