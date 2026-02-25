Executed `iter_0040` end-to-end with concrete experiments and artifacts.

- Ran:
  - `conda run -n subproject40-topology python -m py_compile iterations/iter_0040/run_iter0040_screen.py`
  - `conda run -n subproject40-topology python iterations/iter_0040/run_iter0040_screen.py`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex` (in `paper/`)

- Key results:
  - `H103` (`persistent_homology`, N508 rescue): mean delta AUROC `-0.00304`, positive mean null-gap domain-splits `0/6` → negative.
  - `H104` (`manifold_distance`, N520): mean delta AUROC `-0.00908`, positive mean null-gap domain-splits `0/6` → negative.
  - `H105` (`null_sensitivity`, N519): mean conditioned-minus-unconditioned null-gap `-0.05125`, positive conditioned-gain domain-splits `0/6` → negative for rescue objective.

- Required iteration files written:
  - `iterations/iter_0040/executor_iteration_report.md`
  - `iterations/iter_0040/executor_next_steps.md`
  - `iterations/iter_0040/executor_hypothesis_screen.json`

- Machine-readable artifacts generated this iteration:
  - `iterations/iter_0040/iter0040_screen_summary.json`
  - `iterations/iter_0040/h103_interaction_derivative_rescue_by_domain_split_layer.csv`
  - `iterations/iter_0040/h103_interaction_derivative_rescue_domain_summary.csv`
  - `iterations/iter_0040/h103_interaction_derivative_rescue_null_summary.csv`
  - `iterations/iter_0040/h104_depth_motif_grammar_by_domain_split.csv`
  - `iterations/iter_0040/h104_depth_motif_grammar_domain_summary.csv`
  - `iterations/iter_0040/h104_depth_motif_grammar_null_summary.csv`
  - `iterations/iter_0040/h105_string_conditioned_null_calibration_by_domain_split_layer.csv`
  - `iterations/iter_0040/h105_string_conditioned_null_calibration_domain_summary.csv`
  - `iterations/iter_0040/h105_string_conditioned_null_calibration_null_summary.csv`

- Cumulative updates completed:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex` (includes exact marker line `ITERATION UPDATE: iter_0040`)
  - `paper/autoloop_research_paper.pdf` (compiled successfully)
  - `tracking/prompt.md`

- Note:
  - Run emitted many non-blocking sklearn deprecation warnings (`penalty` handling), but all outputs were produced correctly.

