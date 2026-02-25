Completed `iter_0022` with real experiments, artifacts, log updates, and paper compile.

Key outcomes:
- `H49` (multiseed bifiltration): robust discrimination signal but weak utility coupling.  
  - Mean `delta_AUROC=+0.00599`, positive `69/72`, Fisher-significant domain-splits `6/6`, utility-placebo significant `0/3`.
- `H50` (directed/signed topology pilot): strongest positive this iteration.  
  - Mean `delta_AUROC=+0.01585`, positive `11/12`, domain-split Fisher-significant `6/6`.
- `H51` (expanded cross-model motif fingerprint): mixed.  
  - Degree-null enrichment broadened (`5/6` summary rows Fisher-significant), but module-shuffle control failed (`0/18`).

Required iteration files written:
- `iterations/iter_0022/executor_iteration_report.md`
- `iterations/iter_0022/executor_next_steps.md`
- `iterations/iter_0022/executor_hypothesis_screen.json`

Machine-readable experiment artifacts generated (this iteration):
- `iterations/iter_0022/h49_bifiltration_multiseed_by_seed_layer_split.csv`
- `iterations/iter_0022/h49_bifiltration_multiseed_domain_summary.csv`
- `iterations/iter_0022/h49_bifiltration_multiseed_null_summary.csv`
- `iterations/iter_0022/h50_directed_signed_topology_by_domain_layer_split.csv`
- `iterations/iter_0022/h50_directed_signed_topology_domain_summary.csv`
- `iterations/iter_0022/h50_directed_signed_topology_null_summary.csv`
- `iterations/iter_0022/h51_cross_model_motif_fingerprint_by_domain_layer_k.csv`
- `iterations/iter_0022/h51_cross_model_motif_fingerprint_summary.csv`
- `iterations/iter_0022/h51_cross_model_motif_fingerprint_null_summary.csv`
- `iterations/iter_0022/iter0022_screen_summary.json`

Cumulative updates completed:
- `reports/autoloop_master_log.md` updated with `iter_0022`.
- `paper/autoloop_research_paper.tex` updated with exact marker `ITERATION UPDATE: iter_0022`.
- `paper/autoloop_research_paper.pdf` recompiled via `latexmk -pdf`.
- `tracking/prompt.md` updated with progress/decisions/blockers/next steps.

Iteration stopped after this single execution cycle.