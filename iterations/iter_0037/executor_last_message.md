Completed exactly one executor iteration (`iter_0037`) with concrete experiments, outputs, and paper/log updates.

**What I ran**
- `conda run -n subproject40-topology python -m py_compile iterations/iter_0037/run_iter0037_screen.py`
- `PYTHONWARNINGS=ignore conda run --no-capture-output -n subproject40-topology python iterations/iter_0037/run_iter0037_screen.py`
- `latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex` (in `paper/`)

**Hypothesis outcomes**
- `H94` (`persistent_homology`, refinement): **negative**  
  Mean delta AUC `-0.00933`, positive null-gap domain-splits `0/6`.
- `H95` (`graph_topology`, new method): **inconclusive**  
  Mean delta AUC `+0.07710`, but positive null-gap domain-splits `0/6`.
- `H96` (`cross_model_alignment`, rescue-once major change): **negative**  
  Mean module Spearman `-0.00555`, positive null-gap domains `0/3`.

**Required iteration artifacts written**
- `iterations/iter_0037/executor_iteration_report.md`
- `iterations/iter_0037/executor_next_steps.md`
- `iterations/iter_0037/executor_hypothesis_screen.json`
- Machine-readable results including:
  - `iterations/iter_0037/iter0037_screen_summary.json`
  - `iterations/iter_0037/h94_ontology_stratified_weighted_filtration_*.csv`
  - `iterations/iter_0037/h95_graph_bridge_curvature_*.csv`
  - `iterations/iter_0037/h96_cross_model_module_topology_*.csv`

**Cumulative updates completed**
- `reports/autoloop_master_log.md` updated with `iter_0037`.
- `paper/autoloop_research_paper.tex` updated with section marker line `ITERATION UPDATE: iter_0037`.
- `paper/autoloop_research_paper.pdf` rebuilt successfully.
- `tracking/prompt.md` updated with iter_0037 progress/decisions/blockers/next step.