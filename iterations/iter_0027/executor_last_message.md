Iteration `iter_0027` is complete with one full executor cycle and stop.

Implemented + ran:
- `iterations/iter_0027/run_iter0027_screen.py`
- `conda run -n subproject40-topology python -m py_compile iterations/iter_0027/run_iter0027_screen.py`
- `conda run -n subproject40-topology python iterations/iter_0027/run_iter0027_screen.py`
- `latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex` (in `paper/`)

Key quantitative outcomes:
- `H64` (support-margin two-axis filtration): mean `delta_AUROC = -0.03184` (negative; source-disjoint failure slices stayed negative).
- `H65` (cross-model codebook transport): mean `delta_AUROC = -0.10204` with mean `null_gap_q95 = +0.13671` (null-robust but utility-negative vs baseline).
- `H66` (ID interaction-only): mean `delta_AUROC = -0.13176` (negative; 0/6 Fisher-significant domain-splits).

Required artifacts written:
- `iterations/iter_0027/executor_iteration_report.md`
- `iterations/iter_0027/executor_next_steps.md`
- `iterations/iter_0027/executor_hypothesis_screen.json`
- Machine outputs including `iterations/iter_0027/iter0027_screen_summary.json` and all H64/H65/H66 CSVs.

Cumulative updates completed:
- `reports/autoloop_master_log.md` (added `iter_0027` block)
- `paper/autoloop_research_paper.tex` (added section marker `ITERATION UPDATE: iter_0027`)
- `paper/autoloop_research_paper.pdf` (recompiled)
- `tracking/prompt.md` (added `iter_0027` progress/decisions/blockers/next step)

No hard blockers occurred.